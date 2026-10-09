"""
tp4example.model: Net — CIFAR-10 CNN avec split base/personnalisation (FedPer)

Architecture CNN classique:
  Conv2D(3→6, 5×5) → MaxPool → Conv2D(6→16, 5×5) → MaxPool
  → Flatten → FC(400→120) → FC(120→84) → FC(84→10)

FedPer : découpe le modèle en KP couches de personnalisation
  - Couches base (shared): agrégées par le serveur
  - Couches personnalisées (private): gardées localement par chaque client

Utilitaires:
  - Split du modèle selon KP
  - Export/import de paramètres numpy (communication Flower)
  - Entraînement et évaluation standard
"""

from __future__ import annotations
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


# Noms des couches entraînables (dans l'ordre forward)
# Pool est excluded (pas de poids à apprendre)
LAYER_NAMES = ["conv1", "conv2", "fc1", "fc2", "fc3"]
NUM_LAYERS   = len(LAYER_NAMES)  # 5 couches dans ce modèle


class Net(nn.Module):
    """CIFAR-10 CNN — architecture from TP4 specification."""

    def __init__(self) -> None:
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool  = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1   = nn.Linear(16 * 5 * 5, 120)
        self.fc2   = nn.Linear(120, 84)
        self.fc3   = nn.Linear(84, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

    # ------------------------------------------------------------------
    # Utilitaires de split des couches
    # Divise le modèle en couches base (partagées) et personnalisées (locales)
    # ------------------------------------------------------------------

    @staticmethod
    def _base_layer_names(num_personalization_layers: int) -> List[str]:
        """
        Retourne les noms des couches BASE (partagées).
        
        Avec KP couches de personnalisation:
          - KP premières couches: personnalisées
          - reste: base
        
        Exemple: KP=2 → base=["conv1", "conv2", "fc1"], pers=["fc2", "fc3"]
        """
        kp = min(num_personalization_layers, NUM_LAYERS)
        return LAYER_NAMES[:NUM_LAYERS - kp]

    @staticmethod
    def _personalized_layer_names(num_personalization_layers: int) -> List[str]:
        """
        Retourne les noms des couches PERSONNALISÉES (locales).
        
        Les KP dernières couches sont personnalisées.
        Exemple: KP=2 → pers=["fc2", "fc3"]
        """
        kp = min(num_personalization_layers, NUM_LAYERS)
        return LAYER_NAMES[NUM_LAYERS - kp:]

    # ------------------------------------------------------------------
    # Extraction / chargement des paramètres
    # Convertit entre state_dict PyTorch et listes numpy (pour Flower)
    # ------------------------------------------------------------------

    def get_base_model_parameters(self, num_personalization_layers: int) -> List[np.ndarray]:
        """
        Exporte les paramètres des couches BASE en listes numpy.
        
        Utilisé par le client pour envoyer les couches base au serveur.
        Retourne: [conv1.weight, conv1.bias, conv2.weight, conv2.bias, ...]
        """
        base_names = self._base_layer_names(num_personalization_layers)
        params = []
        sd = self.state_dict()
        for name in base_names:
            params.append(sd[f"{name}.weight"].cpu().numpy())
            params.append(sd[f"{name}.bias"].cpu().numpy())
        return params

    def set_base_model_parameters(
        self,
        params: List[np.ndarray],
        num_personalization_layers: int,
    ) -> None:
        """
        Charge les paramètres des couches BASE depuis des arrays numpy.
        
        Utilisé par le client pour charger les poids globaux du serveur.
        Modifie le state_dict en-place.
        """
        base_names = self._base_layer_names(num_personalization_layers)
        sd = self.state_dict()
        idx = 0
        for name in base_names:
            sd[f"{name}.weight"] = torch.tensor(params[idx]);     idx += 1
            sd[f"{name}.bias"]   = torch.tensor(params[idx]);     idx += 1
        self.load_state_dict(sd)

    def get_personalized_model_parameters(self, num_personalization_layers: int) -> List[np.ndarray]:
        """
        Exporte les paramètres des couches PERSONNALISÉES en listes numpy.
        
        Utilisé par le client pour sauvegarder ses couches personnalisées
        dans context.state (persistance locale entre les rounds).
        """
        pers_names = self._personalized_layer_names(num_personalization_layers)
        params = []
        sd = self.state_dict()
        for name in pers_names:
            params.append(sd[f"{name}.weight"].cpu().numpy())
            params.append(sd[f"{name}.bias"].cpu().numpy())
        return params

    def set_personalized_model_parameters(
        self,
        params: List[np.ndarray],
        num_personalization_layers: int,
    ) -> None:
        """
        Charge les paramètres des couches PERSONNALISÉES depuis des arrays numpy.
        
        Utilisé par le client pour restaurer ses couches personnalisées
        depuis context.state.
        """
        pers_names = self._personalized_layer_names(num_personalization_layers)
        sd = self.state_dict()
        idx = 0
        for name in pers_names:
            sd[f"{name}.weight"] = torch.tensor(params[idx]);     idx += 1
            sd[f"{name}.bias"]   = torch.tensor(params[idx]);     idx += 1
        self.load_state_dict(sd)

    # ------------------------------------------------------------------
    # Entraînement et évaluation
    # ------------------------------------------------------------------

    def train_epoch(
        self,
        train_loader: DataLoader,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
    ) -> Tuple[float, float]:
        """
        Entraîne le modèle pour une époque locale.
        
        Processus standard:
          1. Mode train (active Dropout)
          2. Pour chaque batch:
            - Forward pass
            - Calcul de loss
            - Backward + optimisation
            - Accumulation des métriques
        
        Retourne: (loss_moyenne, accuracy)
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
            correct    += outputs.max(1)[1].eq(y_batch).sum().item()
            total      += x_batch.size(0)
        return (total_loss / total, correct / total) if total > 0 else (0.0, 0.0)

    @torch.no_grad()
    def test_epoch(
        self,
        test_loader: DataLoader,
        criterion: nn.Module,
        device: torch.device,
    ) -> Tuple[float, float]:
        """
        Évalue le modèle sur un dataset de validation/test.
        
        @torch.no_grad() : désactive le calcul des gradients
        (économise mémoire, accélère l'inférence)
        
        Retourne: (loss_moyenne, accuracy)
        """
        self.eval()
        total_loss, correct, total = 0.0, 0, 0
        for x_batch, y_batch in test_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            outputs = self(x_batch)
            loss = criterion(outputs, y_batch)
            total_loss += loss.item() * x_batch.size(0)
            correct    += outputs.max(1)[1].eq(y_batch).sum().item()
            total      += x_batch.size(0)
        return (total_loss / total, correct / total) if total > 0 else (0.0, 0.0)
    