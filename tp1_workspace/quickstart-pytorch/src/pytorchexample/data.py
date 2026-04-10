# data.py
# Gestion des données pour le Federated Learning (TP1).
#
# Deux fonctions principales :
#   - generate_distributed_datasets : crée les splits de données par client
#     selon une distribution de Dirichlet (non-IID)
#   - load_client_data : charge les données d'un client spécifique

from typing import Tuple
import os
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset, random_split
from torchvision import datasets, transforms


def generate_distributed_datasets(k: int, alpha: float, save_dir: str, min_samples: int = 10) -> None:
    """
    Distribue le dataset FashionMNIST entre k clients selon une distribution de Dirichlet.

    Paramètres :
      k           : nombre de clients (ex: 50)
      alpha       : paramètre de la distribution de Dirichlet
                    - alpha petit (0.1) → distribution très hétérogène (non-IID)
                      chaque client a surtout des images d'une seule classe
                    - alpha grand (100) → distribution quasi-uniforme (IID)
                      chaque client a des images de toutes les classes
      save_dir    : dossier où sauvegarder les fichiers .pt des clients
      min_samples : nombre minimum d'échantillons garantis par client
                    (évite les clients complètement vides)

    Résultat : k fichiers client_0.pt ... client_{k-1}.pt dans save_dir/
    Chaque fichier contient {"x": tenseur_images, "y": tenseur_labels}
    """
    os.makedirs(save_dir, exist_ok=True)  # Créer le dossier si nécessaire

    # Télécharger FashionMNIST (60 000 images train, 28×28 pixels, 10 classes)
    # Classes : T-shirt, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag, Boot
    transform = transforms.ToTensor()
    dataset = datasets.FashionMNIST(root="./data", train=True, download=True, transform=transform)

    X = dataset.data     # Tenseur des images : shape (60000, 28, 28)
    y = dataset.targets  # Tenseur des labels : shape (60000,)

    num_classes = 10
    data_per_client = [[] for _ in range(k)]  # Liste d'indices pour chaque client

    # Distribution de Dirichlet par classe :
    # Pour chaque classe c (0 à 9), on distribue les indices de ses exemples
    # entre les k clients selon un tirage Dirichlet(alpha).
    # Dirichlet(alpha) génère un vecteur de k proportions qui somment à 1.
    for c in range(num_classes):
        # Indices de tous les exemples de la classe c
        idx_c = np.where(y.numpy() == c)[0]
        np.random.shuffle(idx_c)  # Mélanger pour éviter les biais d'ordre

        # Tirer les proportions selon Dirichlet(alpha, alpha, ..., alpha)
        # Un petit alpha concentre la masse sur quelques composantes
        proportions = np.random.dirichlet(np.repeat(alpha, k))

        # Convertir les proportions en indices de coupure (cumsum → positions de split)
        proportions = (np.cumsum(proportions) * len(idx_c)).astype(int)[:-1]

        # Découper le tableau d'indices selon ces positions
        split = np.split(idx_c, proportions)

        # Assigner les indices à chaque client
        for i in range(k):
            data_per_client[i].extend(split[i])

    # Garantir un minimum d'échantillons pour chaque client
    # (si un client est trop pauvre, on lui prend des exemples du client le plus riche)
    for i in range(k):
        while len(data_per_client[i]) < min_samples:
            richest = max(range(k), key=lambda j: len(data_per_client[j]))
            data_per_client[i].append(data_per_client[richest].pop())

    # Sauvegarder les données de chaque client dans un fichier .pt
    for i in range(k):
        idxs = data_per_client[i]
        x_client = X[idxs]  # Images du client i
        y_client = y[idxs]  # Labels du client i

        # Sauvegarder sous forme de dictionnaire PyTorch
        torch.save({"x": x_client, "y": y_client}, f"{save_dir}/client_{i}.pt")

    # Afficher des statistiques sur la distribution
    sizes = [len(data_per_client[i]) for i in range(k)]
    print(f"✅ {k} clients datasets generated in {save_dir}")
    print(f"   min={min(sizes)}, max={max(sizes)}, avg={int(np.mean(sizes))} samples/client")


def load_client_data(cid: int, data_dir: str, batch_size: int) -> Tuple[DataLoader, DataLoader]:
    """
    Charge les données d'un client spécifique et retourne deux DataLoaders.

    Le fichier client_{cid}.pt doit avoir été créé par generate_distributed_datasets().

    Retourne :
      train_loader : DataLoader avec 80% des données du client (entraînement local)
      val_loader   : DataLoader avec 20% des données du client (évaluation locale)

    Preprocessing :
      - Les images sont converties en float32 et normalisées dans [0, 1]
      - .unsqueeze(1) ajoute une dimension channel → (N, 1, 28, 28) pour les Conv2D
      - / 255.0 normalise les pixels de [0, 255] vers [0.0, 1.0]
    """
    # Charger le fichier de données du client
    data = torch.load(f"{data_dir}/client_{cid}.pt")

    # Préparer les tenseurs :
    # - float() : convertir uint8 → float32 (nécessaire pour les calculs)
    # - unsqueeze(1) : ajouter la dimension channel (28,28) → (1,28,28)
    # - / 255.0 : normaliser [0,255] → [0.0,1.0]
    X = data["x"].float().unsqueeze(1) / 255.0
    y = data["y"]  # Labels : entiers de 0 à 9

    # Créer un dataset PyTorch standard à partir des tenseurs
    dataset = TensorDataset(X, y)

    # Diviser 80% entraînement / 20% validation (split local au client)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size

    # random_split fait un split aléatoire reproductible (sans seed ici)
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # DataLoaders :
    # - shuffle=True pour train : mélanger les batches à chaque époque
    # - shuffle=False pour val : pas besoin de mélanger pour l'évaluation
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader
