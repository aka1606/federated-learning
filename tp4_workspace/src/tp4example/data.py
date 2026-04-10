"""
tp4example.data — CIFAR-10 Distributed Dataset Generation (FedPer TP4)

Charges CIFAR-10 et le distribue entre k clients avec hétérogénéité contrôlée.
Chaque client reçoit train + validation locales sauvegardées sous forme d'indices.

Utilise:
  - Dirichlet(alpha) pour générer des distributions non-IID
  - Sauvegarde des indices plutôt que des copies (économise de l'espace)
"""

from __future__ import annotations
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset, random_split
from torchvision import datasets, transforms


# Normalization CIFAR-10 : moyenne et écart-type par canal (RGB)
# Ces valeurs sont calculées sur le dataset complet et utilisées pour la normalisation
TRANSFORM = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465),  # RGB mean
                         (0.2470, 0.2435, 0.2616)),  # RGB std
])


def generate_distributed_datasets(
    k: int,
    alpha: float,
    save_dir: str | Path,
    val_ratio: float = 0.1,
    seed: int = 42,
) -> None:
    """
    Génère k datasets distribués CIFAR-10 avec hétérogénéité Dirichlet.
    
    Processus:
      1. Télécharger CIFAR-10 train (50k) et test (10k)
      2. Partitioner train en k clients selon Dirichlet(alpha)
      3. Pour chaque client:
        - Train/val split (90/10 par défaut)
        - Sauvegarder les indices sous client_i.pt
      4. Sauvegarder les indices test communs sous test_indices.pt
    
    Paramètres:
      k: nombre de clients
      alpha: paramètre Dirichlet
        - alpha=0.1: très hétérogène (peu de classes par client)
        - alpha=10: quasi-IID (distribution équilibrée)
      save_dir: répertoire de sortie
      val_ratio: fraction train→validation (10% par défaut)
      seed: seed pour reproductibilité
    
    Fichiers générés:
      - client_0.pt, ..., client_{k-1}.pt: indices pour chaque client
      - test_indices.pt: indices test communs
      - cifar10_raw/: données CIFAR-10 brutes (téléchargées une seule fois)
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(seed)

    # Télécharger CIFAR-10 complet (train + test)
    print("  Downloading CIFAR-10...")
    full_train = datasets.CIFAR10(
        root=str(save_dir / "cifar10_raw"),
        train=True, download=True, transform=TRANSFORM,
    )
    full_test = datasets.CIFAR10(
        root=str(save_dir / "cifar10_raw"),
        train=False, download=True, transform=TRANSFORM,
    )

    targets = np.array(full_train.targets)
    num_classes = 10
    n_total = len(targets)

    # Partitionnement Dirichlet : pour chaque classe, diviser ses échantillons
    # entre les k clients selon Dirichlet(alpha)
    class_indices = [np.where(targets == c)[0] for c in range(num_classes)]
    client_indices = [[] for _ in range(k)]

    for c in range(num_classes):
        idx_c = class_indices[c]
        rng.shuffle(idx_c)
        # Générer k proportions qui somment à 1
        proportions = rng.dirichlet(alpha * np.ones(k))
        proportions = (proportions * len(idx_c)).astype(int)
        # Corriger les arrondis (s'assurer que la somme égale len(idx_c))
        proportions[-1] = len(idx_c) - proportions[:-1].sum()
        splits = np.split(idx_c, np.cumsum(proportions[:-1]))
        for i, split in enumerate(splits):
            client_indices[i].extend(split.tolist())

    # Sauvegarder les indices pour chaque client
    sizes = []
    for i, indices in enumerate(client_indices):
        indices = np.array(indices)
        rng.shuffle(indices)
        # Split train/val pour ce client
        n_val = max(1, int(len(indices) * val_ratio))
        train_idx = indices[n_val:].tolist()  # 90% pour train
        val_idx   = indices[:n_val].tolist()  # 10% pour val

        # Sauvegarder sous forme d'indices (pas les données elles-mêmes)
        data = {
            "train_indices": train_idx,
            "val_indices":   val_idx,
        }
        torch.save(data, save_dir / f"client_{i}.pt")
        sizes.append(len(train_idx))

    # Sauvegarder les indices test communs (test set partagé pour l'évaluation)
    torch.save(list(range(len(full_test))), save_dir / "test_indices.pt")

    print(f"✅ {k} CIFAR-10 client datasets generated in {save_dir}")
    print(f"   min={min(sizes)}, max={max(sizes)}, avg={int(np.mean(sizes))} train samples/client")


def load_client_data(
    cid: int,
    data_dir: str | Path,
    batch_size: int = 64,
) -> Tuple[DataLoader, DataLoader]:
    """
    Charge les DataLoaders train/val pour un client.
    
    Processus:
      1. Charger le dataset CIFAR-10 complet (une seule fois, en cache)
      2. Lire les indices pour ce client (train_idx, val_idx)
      3. Créer des Subsets du dataset complet
      4. Retourner two DataLoaders (train avec shuffle, val sans shuffle)
    
    Paramètres:
      cid: client ID (0 à k-1)
      data_dir: répertoire contenant les fichiers client_*.pt et cifar10_raw/
      batch_size: taille des batches pour les DataLoaders
    
    Retourne:
      (train_loader, val_loader): DataLoaders pour entraînement et validation
    """
    data_dir = Path(data_dir)

    full_train = datasets.CIFAR10(
        root=str(data_dir / "cifar10_raw"),
        train=True, download=False, transform=TRANSFORM,
    )

    client_data = torch.load(data_dir / f"client_{cid}.pt", weights_only=True)
    train_set = Subset(full_train, client_data["train_indices"])
    val_set   = Subset(full_train, client_data["val_indices"])

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_set,   batch_size=batch_size, shuffle=False)

    return train_loader, val_loader
