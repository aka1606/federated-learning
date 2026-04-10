"""
tp3example.data — Distributed dataset generation and loading

Fonctions pour :
  1. Générer des datasets non-IID (hétérogènes) pour k clients
     - Utilise la distribution Dirichlet pour contrôler l'hétérogénéité
     - alpha élevé → quasi-IID (données similaires sur tous les clients)
     - alpha faible → très hétérogène (données spécialisées par client)
  
  2. Charger les données d'un client et les diviser train/validation
     - Split 80% train, 20% validation
     - Retourne des DataLoaders pour l'entraînement distribué
"""

from typing import Tuple
import os
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset, random_split
from torchvision import datasets, transforms


def generate_distributed_datasets(k: int, alpha: float, save_dir: str, min_samples: int = 10) -> None:
    """
    Génère k datasets distribués pour les clients avec hétérogénéité contrôlée.
    
    Paramètres:
        k: nombre de clients
        alpha: paramètre Dirichlet pour contrôler l'hétérogénéité
               - alpha=0.1: très hétérogène (peu de classes par client)
               - alpha=10: quasi-IID (distribution équilibrée)
        save_dir: répertoire où sauvegarder les fichiers client_*.pt
        min_samples: nombre minimum d'échantillons par client
    
    Processus:
        1. Télécharge FashionMNIST (60k images d'entraînement)
        2. Pour chaque classe: divise les indices selon Dirichlet
        3. Assemble les données de chaque client
        4. Garantit un minimum de min_samples par client
        5. Sauvegarde chaque client en fichier .pt
    """
    os.makedirs(save_dir, exist_ok=True)

    # Charger FashionMNIST depuis torchvision
    transform = transforms.ToTensor()
    dataset = datasets.FashionMNIST(root="./data", train=True, download=True, transform=transform)

    X = dataset.data
    y = dataset.targets

    # Distribution des 10 classes sur k clients selon Dirichlet
    num_classes = 10
    data_per_client = [[] for _ in range(k)]

    # Pour CHAQUE classe: générer des proportions aléatoires et diviser l'indice
    for c in range(num_classes):
        idx_c = np.where(y.numpy() == c)[0]  # Tous les indices de la classe c
        np.random.shuffle(idx_c)  # Mélanger les indices
        
        # Générer k proportions qui somment à 1 (distribution Dirichlet)
        proportions = np.random.dirichlet(np.repeat(alpha, k))
        # Convertir en indices de split
        proportions = (np.cumsum(proportions) * len(idx_c)).astype(int)[:-1]
        # Diviser les indices selon les proportions
        split = np.split(idx_c, proportions)
        
        # Assigner les données de la classe c à chaque client
        for i in range(k):
            data_per_client[i].extend(split[i])

    # Garantir un minimum de min_samples par client
    # (évite les clients avec trop peu de données)
    for i in range(k):
        while len(data_per_client[i]) < min_samples:
            # Trouver le client avec le plus de données et en transférer une
            richest = max(range(k), key=lambda j: len(data_per_client[j]))
            data_per_client[i].append(data_per_client[richest].pop())

    # Sauvegarder les datasets en fichiers individuels
    for i in range(k):
        idxs = data_per_client[i]
        torch.save({"x": X[idxs], "y": y[idxs]}, f"{save_dir}/client_{i}.pt")

    # Afficher statistiques
    sizes = [len(data_per_client[i]) for i in range(k)]
    print(f"✅ {k} clients datasets generated in {save_dir}")
    print(f"   min={min(sizes)}, max={max(sizes)}, avg={int(np.mean(sizes))} samples/client")


def load_client_data(cid: int, data_dir: str, batch_size: int) -> Tuple[DataLoader, DataLoader]:
    """
    Charge les données d'un client et les divise en train/validation.
    
    Paramètres:
        cid: client ID (0 à k-1)
        data_dir: répertoire contenant les fichiers client_*.pt
        batch_size: taille des batches pour les DataLoaders
    
    Retourne:
        (train_loader, val_loader): DataLoaders pour entraînement et validation
        
    Processus:
        1. Charge le fichier client_{cid}.pt
        2. Normalise les images (pixels / 255)
        3. Ajoute une dimension canal (C=1 pour FashionMNIST)
        4. Split 80/20 train/validation
        5. Crée les DataLoaders avec shuffle (train) et sans shuffle (val)
    """
    data = torch.load(f"{data_dir}/client_{cid}.pt")
    X = data["x"].float().unsqueeze(1) / 255.0  # Normaliser [0,1]
    y = data["y"]

    # Créer un TensorDataset et le diviser 80/20
    dataset = TensorDataset(X, y)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # Créer les DataLoaders (shuffle=True pour train, False pour validation)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False)
    return train_loader, val_loader
