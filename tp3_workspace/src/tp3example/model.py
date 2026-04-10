"""tp3example.model: CustomFashionModel — FedAvg + attack methods for TP3

Modèle CNN pour FashionMNIST (28x28, 1 canal) :
  - Architecture simple : 2 Conv2D + Flatten + 2 Linear
  - 10 classes de sortie (FashionMNIST)
  
Méthodes:
  ✓ train_epoch / test_epoch : entraînement standard FedAvg
  ✓ train_epoch_data_poison : attaque par empoisonnement des labels
  ✓ Paramètres de modèle (get/set) pour la communication Flower
"""

from __future__ import annotations
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


class CustomFashionModel(nn.Module):

    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Linear(128, 10),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    # ------------------------------------------------------------------
    # Standard training / evaluation
    # Entraînement honnête et évaluation standard
    # ------------------------------------------------------------------

    def train_epoch(
        self,
        train_loader: DataLoader,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
    ) -> Tuple[float, float]:
        """
        Entraîne le modèle pour une époque.
        
        Processus:
          1. Itérer sur les batches
          2. Forward pass, calcul de la loss
          3. Backward pass et optimisation
          4. Accumuler les métriques (loss, accuracy)
        
        Retourne:
          (avg_loss, accuracy) sur les données d'entraînement
        """
        self.train()
        total_loss, correct, total = 0.0, 0, 0
        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = self(x_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * x_batch.size(0)
            correct += outputs.max(1)[1].eq(y_batch).sum().item()
            total += x_batch.size(0)
        return (total_loss / total, correct / total) if total > 0 else (0.0, 0.0)

    @torch.no_grad()
    def test_epoch(
        self,
        test_loader: DataLoader,
        criterion: nn.Module,
        device: torch.device,
    ) -> Tuple[float, float]:
        """
        Évalue le modèle en mode validation/test.
        
        @torch.no_grad() désactive le calcul des gradients (économise mémoire)
        
        Retourne:
          (avg_loss, accuracy) sur les données de test/validation
        """
        self.eval()
        total_loss, correct, total = 0.0, 0, 0
        for x_batch, y_batch in test_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            outputs = self(x_batch)
            loss = criterion(outputs, y_batch)
            total_loss += loss.item() * x_batch.size(0)
            correct += outputs.max(1)[1].eq(y_batch).sum().item()
            total += x_batch.size(0)
        return (total_loss / total, correct / total) if total > 0 else (0.0, 0.0)

    # ------------------------------------------------------------------
    # Attack methods (TP3)
    # Méthodes d'attaque : empoisonnement des données et du modèle
    # ------------------------------------------------------------------

    def train_epoch_data_poison(
        self,
        train_loader: DataLoader,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        num_classes: int = 10,
    ) -> Tuple[float, float]:
        """
        Entraînement avec EMPOISONNEMENT DES DONNÉES (label flipping).
        
        Attaque:
          Les labels sont inversés durant l'entraînement :
            class_i → class_(num_classes-1-i)
          Exemple (10 classes):
            T-shirt (0) → Boot (9)
            Trouser (1) → Bag (8)
            ...
            Boot (9) → T-shirt (0)
        
        Effet:
          Le modèle apprend des associations complètement inversées,
          ce qui perturbe l'apprentissage fédéré global.
        
        Les images X ne sont PAS modifiées, seuls les labels y sont flippés.
        
        Retourne:
          (avg_loss, accuracy) sur les labels FLIPPÉS
        """
        self.train()
        total_loss, correct, total = 0.0, 0, 0
        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            # FLIP les labels pour créer l'attaque
            y_flipped = (num_classes - 1) - y_batch   # Inversion : i → 9-i
            optimizer.zero_grad()
            outputs = self(x_batch)
            # La loss est calculée avec les labels flippés
            loss = criterion(outputs, y_flipped)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * x_batch.size(0)
            # Accuracy calculée sur les labels flippés
            correct += outputs.max(1)[1].eq(y_flipped).sum().item()
            total += x_batch.size(0)
        return (total_loss / total, correct / total) if total > 0 else (0.0, 0.0)

    # ------------------------------------------------------------------
    # Parameter exchange helpers
    # Utilitaires pour sérialiser/désérialiser les paramètres du modèle
    # (nécessaire pour la communication avec Flower)
    # ------------------------------------------------------------------

    def get_model_parameters(self) -> List[np.ndarray]:
        """
        Exporte les paramètres du modèle sous forme NumPy.
        
        Utilisé par Flower pour :
          - Sérialiser le modèle pour envoi au serveur
          - Créer les mises à jour centralisées
        
        Retourne:
          Liste des tenseurs convertis en arrays NumPy
        """
        return [val.cpu().numpy() for _, val in self.state_dict().items()]

    def set_model_parameters(self, params: List[np.ndarray]) -> None:
        """
        Importe les paramètres du modèle depuis NumPy.
        
        Utilisé par Flower pour :
          - Charger les poids globaux du serveur
          - Restaurer le modèle après une mise à jour
        
        Paramètres:
          params: Liste des arrays NumPy correspondant aux poids du state_dict
        """
        keys = list(self.state_dict().keys())
        self.load_state_dict(
            {k: torch.tensor(v) for k, v in zip(keys, params)}, strict=True
        )
