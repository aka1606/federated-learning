"""
server_app.py — Serveur Flower pour le Federated Learning (TP1).

Ce fichier implémente la boucle d'orchestration FL côté serveur :
  1. Lire les hyperparamètres depuis pyproject.toml (via context.run_config)
  2. Initialiser le modèle global
  3. Pour chaque round : envoyer les poids → entraînement local → récupérer → agréger
  4. Sauvegarder les résultats dans results/<run_id>.json

Algorithmes supportés : FedAvg, FedProx, SCAFFOLD, FedSGD
"""

import json
import random
from pathlib import Path
from typing import Dict, List

import torch

from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
from flwr.serverapp import Grid, ServerApp

from pytorchexample.model import CustomFashionModel

# Instanciation de l'application serveur Flower
app = ServerApp()


# -----------------------------------------------------------------------
# Fonctions utilitaires
# -----------------------------------------------------------------------

def sample_clients(
    node_ids: List[str],
    fraction: float,
    global_rng: random.Random,
) -> List[str]:
    """
    Sélectionne aléatoirement une fraction des clients disponibles.

    fraction : proportion de clients participants (ex: 0.5 = 50% des clients)
    global_rng : générateur aléatoire avec seed fixe → résultats reproductibles

    Garantit au moins 1 client sélectionné (max(1, ...)).
    """
    k = max(1, int(len(node_ids) * fraction))
    return global_rng.sample(node_ids, k)


def fedavg(
    state_dicts: List[Dict[str, torch.Tensor]],
    num_examples: List[int],
) -> Dict[str, torch.Tensor]:
    """
    Agrégation FedAvg : moyenne pondérée des poids des clients.

    Formule : W_global = Σ_k (n_k / Σ_j n_j) × W_k

    Chaque client k contribue proportionnellement à la taille de son dataset.
    Un client avec plus de données a plus d'influence sur le modèle global.

    Args :
      state_dicts  : liste des state_dict locaux (un par client)
      num_examples : nombre d'exemples d'entraînement de chaque client

    Retourne : state_dict agrégé (le nouveau modèle global)
    """
    total = sum(num_examples)  # Total d'exemples sur tous les clients sélectionnés
    aggregated: Dict[str, torch.Tensor] = {}
    for key in state_dicts[0].keys():
        # Moyenne pondérée couche par couche
        aggregated[key] = sum(
            state_dicts[i][key].float() * (num_examples[i] / total)
            for i in range(len(state_dicts))
        )
    return aggregated


def aggregate_metrics(
    metrics: List[Dict[str, float]],
    num_examples: List[int],
) -> Dict[str, float]:
    """
    Calcule la moyenne pondérée des métriques scalaires entre les clients.

    Utilisé pour agréger train_loss, train_acc, eval_loss, eval_acc.
    La pondération est proportionnelle au nombre d'exemples de chaque client.
    """
    total = sum(num_examples)
    aggregated: Dict[str, float] = {}
    for key in metrics[0].keys():
        aggregated[key] = sum(
            metrics[i][key] * (num_examples[i] / total)
            for i in range(len(metrics))
        )
    return aggregated


# -----------------------------------------------------------------------
# Boucle principale du serveur
# -----------------------------------------------------------------------

@app.main()
def main(grid: Grid, context: Context) -> None:
    """
    Point d'entrée principal du serveur Flower.

    grid : interface pour communiquer avec les clients (envoyer/recevoir messages)
    context : accès à la configuration (run_config) et à l'identifiant du run

    Flux d'exécution par round :
      1. Sélectionner les clients participants
      2. Envoyer les poids globaux + lancer l'entraînement local
      3. Récupérer les poids mis à jour + métriques
      4. Agréger avec FedAvg (ou variante selon l'algorithme)
      5. Envoyer les poids agrégés + lancer l'évaluation
      6. Récupérer les métriques d'évaluation
      7. Logger les résultats du round
    """

    # --- Lecture des hyperparamètres depuis pyproject.toml ---
    num_rounds: int          = int(context.run_config["num-server-rounds"])   # Nombre de rounds FL
    fraction_train: float    = float(context.run_config["fraction-train"])    # Fraction clients participants
    lr: float                = float(context.run_config["learning-rate"])     # Learning rate (pour SCAFFOLD)
    seed: int                = int(context.run_config["seed"])                # Graine pour reproductibilité
    local_epochs: int        = int(context.run_config["local-epochs"])        # Époques locales (pour SCAFFOLD)
    batch_size: int          = int(context.run_config["batch-size"])          # Taille batch (pour SCAFFOLD)
    data_dir: str            = str(context.run_config["data-dir"])            # Dossier données
    client_optimizer: str    = str(context.run_config["client-optimizer"])    # "sgd" ou "adam"
    client_algorithm: str    = str(context.run_config["client-algorithm"])    # "fedavg", "fedprox", "scaffold", "fedsgd"
    run_id: str              = str(context.run_id)                            # ID unique de cette simulation

    # --- Initialisation du modèle global ---
    global_model = CustomFashionModel()
    global_weights = global_model.state_dict()  # Poids initiaux (aléatoires)

    # --- SCAFFOLD : variateur de contrôle global c (initialisé paresseusement au round 1) ---
    # c_global corrige la dérive des gradients locaux en milieu non-IID
    c_global: dict = {}   # {"0": tensor_layer0, "1": tensor_layer1, ...}

    # Générateur aléatoire avec seed fixe pour la sélection des clients
    global_rng = random.Random(seed)
    results = []  # Liste des métriques de chaque round (sera sauvegardée en JSON)

    for round_num in range(1, num_rounds + 1):

        # ----------------------------------------------------------------
        # 1. Sélection des clients participants pour ce round
        # ----------------------------------------------------------------
        all_node_ids = [str(nid) for nid in grid.get_node_ids()]
        sampled_ids = sample_clients(all_node_ids, fraction_train, global_rng)

        # ----------------------------------------------------------------
        # 2. Round d'entraînement
        # ----------------------------------------------------------------

        # SCAFFOLD : initialiser c_global à zéros au premier round
        # (un zéro = pas de correction au début, s'affine au fil des rounds)
        if client_algorithm == "scaffold" and not c_global:
            c_global = {str(i): torch.zeros_like(v) for i, v in enumerate(global_weights.values())}

        # Construire les messages d'entraînement pour chaque client sélectionné
        train_msgs = []
        for node_id in sampled_ids:
            # Le message contient les poids globaux actuels
            content_dict = {"arrays": ArrayRecord(global_weights)}
            if client_algorithm == "scaffold":
                # SCAFFOLD : inclure aussi le variateur de contrôle global
                content_dict["scaffold_c"] = ArrayRecord(c_global)
            content = RecordDict(content_dict)
            msg = Message(
                content=content,
                message_type="train",         # Type de message : demande d'entraînement
                dst_node_id=int(node_id),     # Destinataire : ce client spécifique
                group_id=str(round_num),      # Grouper tous les messages du même round
            )
            train_msgs.append(msg)

        # Envoyer tous les messages et attendre les réponses
        train_replies = list(grid.send_and_receive(train_msgs))

        # Parser les réponses d'entraînement
        state_dicts, num_examples_train, train_metrics_list = [], [], []
        delta_c_list = []  # Pour SCAFFOLD : deltas des variateurs de contrôle locaux
        for reply in train_replies:
            # Extraire les poids mis à jour du client
            sd = reply.content["arrays"].to_torch_state_dict()
            state_dicts.append(sd)
            # Extraire les métriques
            m = reply.content["metrics"]
            n = int(m["num_examples"])
            num_examples_train.append(n)
            train_metrics_list.append({
                "train_loss": float(m["train_loss"]),
                "train_acc":  float(m["train_acc"]),
            })
            if client_algorithm == "scaffold":
                # Récupérer le delta du variateur de contrôle local
                dc_sd = reply.content["scaffold_delta_ck"].to_torch_state_dict()
                delta_c_list.append(dc_sd)

        # Agrégation FedAvg des poids mis à jour
        global_weights = fedavg(state_dicts, num_examples_train)
        agg_train = aggregate_metrics(train_metrics_list, num_examples_train)

        # SCAFFOLD : mettre à jour le variateur de contrôle global
        # c_global ← c_global + (1/|S|) × Σ(delta_c_k)
        # |S| = nombre de clients participants, delta_c_k = correction du client k
        if client_algorithm == "scaffold" and delta_c_list:
            n_clients = len(delta_c_list)
            for key in c_global:
                c_global[key] = c_global[key] + sum(dc[key] for dc in delta_c_list) / n_clients

        # ----------------------------------------------------------------
        # 3. Round d'évaluation
        # ----------------------------------------------------------------
        # On évalue le modèle global agrégé sur les mêmes clients qui ont entraîné
        eval_msgs = []
        for node_id in sampled_ids:
            content = RecordDict({
                "arrays": ArrayRecord(global_weights),  # Poids globaux agrégés
            })
            msg = Message(
                content=content,
                message_type="evaluate",       # Type de message : demande d'évaluation
                dst_node_id=int(node_id),
                group_id=str(round_num),
            )
            eval_msgs.append(msg)

        eval_replies = list(grid.send_and_receive(eval_msgs))

        eval_metrics_list, num_examples_eval = [], []
        for reply in eval_replies:
            m = reply.content["metrics"]
            n = int(m["num_examples"])
            num_examples_eval.append(n)
            eval_metrics_list.append({
                "eval_loss": float(m["eval_loss"]),
                "eval_acc":  float(m["eval_acc"]),
            })

        agg_eval = aggregate_metrics(eval_metrics_list, num_examples_eval)

        # ----------------------------------------------------------------
        # 4. Log des statistiques du round
        # ----------------------------------------------------------------
        round_log = {
            "round":         round_num,
            "num_clients":   len(sampled_ids),         # Clients participants ce round
            "train_loss":    agg_train["train_loss"],  # Perte d'entraînement agrégée
            "train_acc":     agg_train["train_acc"],   # Précision d'entraînement agrégée
            "fed_eval_loss": agg_eval["eval_loss"],    # Perte d'évaluation du modèle global
            "fed_eval_acc":  agg_eval["eval_acc"],     # Précision du modèle global ← métrique principale
        }
        results.append(round_log)

        print(
            f"[Round {round_num:>2}/{num_rounds}] "
            f"clients={len(sampled_ids)} | "
            f"train_loss={round_log['train_loss']:.4f} | "
            f"train_acc={round_log['train_acc']:.4f} | "
            f"eval_loss={round_log['fed_eval_loss']:.4f} | "
            f"eval_acc={round_log['fed_eval_acc']:.4f}"
        )

    # ----------------------------------------------------------------
    # 5. Sauvegarde des résultats en JSON
    # ----------------------------------------------------------------
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    results_file = results_dir / f"{run_id}.json"

    # sanitize() remplace NaN/Inf par None (JSON ne supporte pas ces valeurs)
    import math
    def sanitize(obj):
        if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
            return None
        if isinstance(obj, dict):
            return {k: sanitize(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [sanitize(v) for v in obj]
        return obj

    with open(results_file, "w") as f:
        json.dump(sanitize(results), f, indent=2)
    print(f"\nResults saved to {results_file}")
