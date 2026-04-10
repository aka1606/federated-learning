"""
model.py — Modèle CNN pour la classification FashionMNIST (TP1).

Ce fichier contient :
  - CustomFashionModel : architecture CNN + toutes les méthodes d'entraînement
    nécessaires au Federated Learning (FedAvg, FedSGD, FedProx, SCAFFOLD)
  - Méthodes d'échange de paramètres (get/set) pour la communication avec Flower
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


class CustomFashionModel(nn.Module):
    """
    CNN pour la classification FashionMNIST (entrée : images 28×28 grayscale, 10 classes).

    Architecture :
      Conv2D(1→32, 3×3) → ReLU → MaxPool(2×2)   # (1,28,28) → (32,14,14)
      Conv2D(32→64, 3×3) → ReLU → MaxPool(2×2)  # (32,14,14) → (64,7,7)
      Flatten → Linear(3136→128) → ReLU
      Linear(128→10)                              # 10 classes FashionMNIST

    Méthodes d'entraînement :
      - train_epoch()           : entraînement standard (FedAvg)
      - train_one_step()        : une seule étape SGD (FedSGD)
      - train_epoch_fedprox()   : FedAvg + terme proximal (FedProx)
      - train_epoch_scaffold()  : FedAvg + correction de gradient (SCAFFOLD)
      - train_epoch_data_poison() : empoisonnement par inversion de labels (attaque)

    Méthodes d'échange :
      - get_model_parameters() / set_model_parameters() : conversion numpy ↔ state_dict
    """

    def __init__(self) -> None:
        super().__init__()
        # Architecture séquentielle CNN → MLP
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),   # padding=1 → conserve la taille (28×28)
            nn.ReLU(),
            nn.MaxPool2d(2),                               # ÷2 → (32, 14, 14)
            nn.Conv2d(32, 64, kernel_size=3, padding=1),  # (64, 14, 14)
            nn.ReLU(),
            nn.MaxPool2d(2),                               # ÷2 → (64, 7, 7)
            nn.Flatten(),                                  # 64×7×7 = 3136 features
            nn.Linear(64 * 7 * 7, 128),                   # Couche dense
            nn.ReLU(),
            nn.Linear(128, 10),                            # 10 logits = 10 classes
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Passe avant : retourne les logits (avant softmax)."""
        return self.net(x)

    # ------------------------------------------------------------------
    # Entraînement et évaluation locaux
    # ------------------------------------------------------------------

    def train_epoch(
        self,
        train_loader: DataLoader,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
    ) -> Tuple[float, float]:
        """
        Une époque d'entraînement local standard (utilisée par FedAvg).

        À chaque batch :
          1. Forward pass : calcul des prédictions
          2. Calcul de la loss (CrossEntropy)
          3. Backward pass : calcul des gradients
          4. Mise à jour des poids par l'optimiseur

        Retourne : (perte_moyenne, précision)
        """
        self.train()  # Mode entraînement (active Dropout, BatchNorm si présents)
        total_loss = 0.0
        correct = 0
        total = 0

        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()              # Réinitialiser les gradients
            outputs = self(x_batch)            # Forward pass
            loss = criterion(outputs, y_batch) # CrossEntropy loss
            loss.backward()                    # Backpropagation
            optimizer.step()                   # Mise à jour des poids

            total_loss += loss.item() * x_batch.size(0)  # Accumulation pondérée
            _, predicted = outputs.max(1)      # Classe avec le logit le plus élevé
            correct += predicted.eq(y_batch).sum().item()
            total += x_batch.size(0)

        avg_loss = total_loss / total if total > 0 else 0.0
        accuracy = correct / total if total > 0 else 0.0
        return avg_loss, accuracy

    @torch.no_grad()  # Décorateur : désactive le calcul de gradients (plus rapide)
    def test_epoch(
        self,
        test_loader: DataLoader,
        criterion: nn.Module,
        device: torch.device,
    ) -> Tuple[float, float]:
        """
        Une époque d'évaluation locale.

        @torch.no_grad() évite de calculer les gradients (inutiles en inférence)
        → économise de la mémoire et accélère le calcul.

        Retourne : (perte_moyenne, précision)
        """
        self.eval()  # Mode évaluation (désactive Dropout)
        total_loss = 0.0
        correct = 0
        total = 0

        for x_batch, y_batch in test_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            outputs = self(x_batch)
            loss = criterion(outputs, y_batch)

            total_loss += loss.item() * x_batch.size(0)
            _, predicted = outputs.max(1)
            correct += predicted.eq(y_batch).sum().item()
            total += x_batch.size(0)

        avg_loss = total_loss / total if total > 0 else 0.0
        accuracy = correct / total if total > 0 else 0.0
        return avg_loss, accuracy

    def train_epoch_data_poison(
        self,
        train_loader: DataLoader,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        num_classes: int = 10,
    ) -> Tuple[float, float]:
        """
        Empoisonnement par inversion de labels (Data Poisoning — TP3).

        Stratégie d'attaque : les images restent intactes, mais les labels sont
        inversés : classe i → num_classes - 1 - i
        Exemples : 0 (T-shirt) ↔ 9 (Boot), 1 (Trouser) ↔ 8 (Bag), etc.

        Impact : le client malveillant entraîne le modèle à faire des erreurs
        systématiques. Après agrégation FedAvg, le modèle global est dégradé.

        Retourne : (perte_moyenne, précision sur les labels INVERSÉS)
        """
        self.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            # Inverser les labels — les images ne sont pas modifiées
            y_batch = (num_classes - 1) - y_batch
            optimizer.zero_grad()
            outputs = self(x_batch)
            loss = criterion(outputs, y_batch)  # Entraîner sur les labels inversés
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * x_batch.size(0)
            _, predicted = outputs.max(1)
            correct += predicted.eq(y_batch).sum().item()
            total += x_batch.size(0)

        avg_loss = total_loss / total if total > 0 else 0.0
        accuracy = correct / total if total > 0 else 0.0
        return avg_loss, accuracy

    def train_epoch_fedprox(
        self,
        train_loader: DataLoader,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        global_params: List[torch.Tensor],
        mu: float,
    ) -> Tuple[float, float]:
        """
        Une époque d'entraînement avec la régularisation proximale FedProx.

        FedProx ajoute un terme de régularisation à la loss :
          Loss_total = CrossEntropy + (mu/2) * ||w_local - w_global||²

        Ce terme "proximal" empêche les poids locaux de trop s'éloigner
        du modèle global, ce qui améliore la convergence en milieu non-IID.

        mu (fedprox-mu) : force du terme proximal
          - mu = 0   → identique à FedAvg (pas de régularisation)
          - mu grand → forte contrainte, les clients s'éloignent peu du global

        Retourne : (perte_moyenne_totale, précision)
        """
        self.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = self(x_batch)
            loss = criterion(outputs, y_batch)

            # Terme proximal : somme des ||w_i - w_global_i||² pour chaque couche
            # w_g.to(device) : déplacer les poids globaux sur le bon device
            prox = sum(
                ((w - w_g.to(device)) ** 2).sum()
                for w, w_g in zip(self.parameters(), global_params)
            )
            loss = loss + (mu / 2.0) * prox  # Ajouter le terme proximal à la loss

            loss.backward()
            optimizer.step()

            total_loss += loss.item() * x_batch.size(0)
            _, predicted = outputs.max(1)
            correct += predicted.eq(y_batch).sum().item()
            total += x_batch.size(0)

        avg_loss = total_loss / total if total > 0 else 0.0
        accuracy = correct / total if total > 0 else 0.0
        return avg_loss, accuracy

    def train_epoch_scaffold(
        self,
        train_loader: DataLoader,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        c_k: List[torch.Tensor],   # Variateur de contrôle local du client k
        c: List[torch.Tensor],     # Variateur de contrôle global (du serveur)
    ) -> Tuple[float, float]:
        """
        Une époque d'entraînement avec correction de gradient SCAFFOLD.

        SCAFFOLD corrige le "client drift" (dérive des clients en non-IID) en
        appliquant une correction au gradient avant la mise à jour :
          grad_corrigé = grad - c_k + c_global

        Où :
          c_k      = variateur de contrôle local (estimé par le client k)
          c_global = variateur de contrôle global (maintenu par le serveur)

        Cette correction compense la différence entre le gradient local
        (biaisé vers les données du client) et le gradient global.

        Retourne : (perte_moyenne, précision)
        """
        self.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = self(x_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()  # Calcul du gradient standard

            # Application de la correction SCAFFOLD :
            # grad ← grad - c_k + c_global
            # pour chaque paramètre p et ses variateurs correspondants
            for p, ck_i, c_i in zip(self.parameters(), c_k, c):
                if p.grad is not None:
                    p.grad.data.add_(-ck_i.to(device) + c_i.to(device))

            optimizer.step()  # Mise à jour avec le gradient corrigé

            total_loss += loss.item() * x_batch.size(0)
            _, predicted = outputs.max(1)
            correct += predicted.eq(y_batch).sum().item()
            total += x_batch.size(0)

        avg_loss = total_loss / total if total > 0 else 0.0
        accuracy = correct / total if total > 0 else 0.0
        return avg_loss, accuracy

    def train_one_step(
        self,
        train_loader: DataLoader,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
    ) -> Tuple[float, float]:
        """
        FedSGD : une seule étape de gradient sur un seul mini-batch.

        Contrairement à FedAvg (plusieurs époques locales), FedSGD ne fait
        qu'une seule mise à jour par round. Cela correspond à du gradient
        stochastique distribué.

        Avantages : convergence théoriquement plus stable
        Inconvénients : nécessite BEAUCOUP plus de rounds de communication

        Le DataLoader doit être créé avec shuffle=True pour que le premier
        batch soit aléatoire à chaque appel.

        Retourne : (perte_batch, précision_batch)
        """
        self.train()
        # Prendre uniquement le PREMIER batch (next(iter(...)))
        x_batch, y_batch = next(iter(train_loader))
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        outputs = self(x_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()

        _, predicted = outputs.max(1)
        acc = predicted.eq(y_batch).sum().item() / x_batch.size(0)
        return loss.item(), acc

    # ------------------------------------------------------------------
    # Échange de paramètres avec Flower
    # ------------------------------------------------------------------

    def get_model_parameters(self) -> List[np.ndarray]:
        """
        Exporte tous les paramètres du modèle comme liste de tableaux NumPy.

        Utilisé pour envoyer les poids au serveur ou pour les recevoir.
        L'ordre est déterministe (suit l'ordre de state_dict()).

        Retourne : [np.array(conv1.weight), np.array(conv1.bias), ...]
        """
        return [val.cpu().numpy() for _, val in self.state_dict().items()]

    def set_model_parameters(self, params: List[np.ndarray]) -> None:
        """
        Charge les paramètres dans le modèle depuis une liste de tableaux NumPy.

        L'ordre doit correspondre exactement à get_model_parameters().
        strict=True vérifie que toutes les clés correspondent.
        """
        keys = list(self.state_dict().keys())
        new_state_dict = {k: torch.tensor(v) for k, v in zip(keys, params)}
        self.load_state_dict(new_state_dict, strict=True)
