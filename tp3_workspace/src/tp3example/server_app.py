"""tp3example.server_app: ServerApp — FedAvg / FedMedian / Krum orchestration

Application serveur Flower (Flwr) :
  - Orchestre l'apprentissage fédéré sur plusieurs tours
  - Reçoit les mises à jour des clients
  - Les agrège selon une stratégie (FedAvg, FedMedian, Krum)
  - Envoie le modèle global mis à jour aux clients
  - Logs les métriques en JSON

Stratégies d'agrégation:
  ✓ FedAvg: moyenne pondérée par nombre d'exemples
  ✓ FedMedian: médiane coordonnée (robuste aux outliers)
  ✓ Krum: sélection d'une mise à jour proche de ses voisins
"""

import json
import math
import random
from pathlib import Path
from typing import Dict, List

import torch

from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
from flwr.serverapp import Grid, ServerApp

from tp3example.model import CustomFashionModel

app = ServerApp()


# ── Helpers ──────────────────────────────────────────────────────────────────

def sample_clients(node_ids, fraction, rng):
    """
    Sélectionne aléatoirement une fraction des clients disponibles.
    
    Paramètres:
      node_ids: liste de tous les IDs de clients disponibles
      fraction: fraction à sélectionner (ex: 1.0 = tous, 0.5 = moitié)
      rng: numpy.random.Random avec seed pour reproductibilité
    
    Retourne:
      Liste des IDs sélectionnés (de taille max(1, len(node_ids) * fraction))
    """
    k = max(1, int(len(node_ids) * fraction))
    return rng.sample(node_ids, k)


def sanitize(obj):
    """
    Remplace les valeurs NaN/Inf par None pour garder JSON valide.
    
    Utile car les modèles empoisonnés peuvent produire NaN/Inf,
    et JSON n'accepte que None (pas NaN/Inf).
    """
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    if isinstance(obj, dict):
        return {k: sanitize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [sanitize(v) for v in obj]
    return obj


# ── Aggregation strategies ────────────────────────────────────────────────────

def fedavg(state_dicts, num_examples):
    """
    FedAvg — Moyenne pondérée des mises à jour client.
    
    Formule:
      w_global = Σ (num_examples_i / N) * w_i
    
    où:
      - w_i: poids reçus du client i
      - num_examples_i: nombre d'échantillons d'entraînement du client i
      - N: nombre total d'échantillons
    
    Robustesse: SENSIBLE aux outliers (clients malveillants avec model poisoning)
    """
    total = sum(num_examples)
    aggregated = {}
    for key in state_dicts[0]:
        aggregated[key] = sum(
            state_dicts[i][key].float() * (num_examples[i] / total)
            for i in range(len(state_dicts))
        )
    return aggregated


def fedmedian(state_dicts, num_examples):
    """
    FedMedian — Médiane coordonnée par coordonnée.
    
    Processus:
      1. Pour chaque paramètre w[layer][coord]:
        - Récupérer les valeurs de tous les clients
        - Calculer la médiane
      2. Retourner les paramètres médians uniquement
    
    Avantages:
      - Robuste aux outliers bizantins (clients malveillants)
      - Ignore complètement les mises à jour extrêmes
      - Pas d'importance au nombre d'exemples
    
    Note: Ignores num_examples (traite tous les clients égaux)
    """
    aggregated = {}
    for key in state_dicts[0]:
        stacked = torch.stack([sd[key].float() for sd in state_dicts])
        aggregated[key] = torch.median(stacked, dim=0).values
    return aggregated


def krum(state_dicts, num_examples, f):
    """
    Krum — Sélection simple d'une mise à jour 'bonne'.
    
    Algorithme:
      1. For each update vector w_i:
        - Calculer les distances L2 avec tous les autres vecteurs
        - Prendre les n-f-2 plus petites distances
        - Score = somme de ces distances
      2. Sélectionner w_best = argmin(score)
    
    Paramètres:
      - n: nombre total de clients
      - f: nombre de clients supposés malveillants
      
    Intérêt:
      - Très robuste : ignore complètement f+2 pires clients
      - Sélectionne une mise à jour proche de ses voisins
      - Défense contre model poisoning (vecteurs * -5 sont loin)
    """
    n = len(state_dicts)
    # Convertir chaque state_dict en vecteur 1-D (concaténation de tous les poids)
    vecs = [
        torch.cat([p.float().flatten() for p in sd.values()])
        for sd in state_dicts
    ]

    scores = []
    # Pour chaque client i, calculer son score
    for i in range(n):
        # Distances L2 de client i avec tous les autres
        dists = sorted(
            [torch.sum((vecs[i] - vecs[j]) ** 2).item() for j in range(n) if j != i]
        )
        # Prendre les n-f-2 plus proches voisins et sommer leurs distances
        neighbours = max(1, n - f - 2)
        scores.append(sum(dists[:neighbours]))

    # Sélectionner le client avec le plus petit score (le plus proche de ses voisins)
    best = int(torch.tensor(scores).argmin().item())
    return state_dicts[best]


def aggregate_metrics(metrics, num_examples):
    """
    Agrège les métriques des clients en une seule métrique (moyenne pondérée).
    
    Paramètres:
      metrics: list of dicts {"metric_name": value, ...}
      num_examples: list of sample counts per client
    
    Retourne:
      dict avec les mêmes clés, valeurs moyennes pondérées
    """
    total = sum(num_examples)
    aggregated = {}
    for key in metrics[0]:
        aggregated[key] = sum(
            metrics[i][key] * (num_examples[i] / total)
            for i in range(len(metrics))
        )
    return aggregated


# ── Main server loop ──────────────────────────────────────────────────────────

@app.main()
def main(grid: Grid, context: Context) -> None:
    """
    Boucle principale du serveur Flower.
    
    Processus par tour (round):
      1. Sample des clients
      2. Envoyer le modèle à l'entraînement
      3. Recevoir les mises à jour + métriques
      4. Agréger les mises à jour (FedAvg/FedMedian/Krum)
      5. Évaluer le modèle via les clients
      6. Logger et continuer
    
    Configuration (pyproject.toml):
      - num-server-rounds: nombre total de tours
      - fraction-train: fraction des clients à samplernumpy par tour
      - strategy: "fedavg" | "fedmedian" | "krum"
      - malicious-ratio: fraction des clients attaquants
    """
    cfg = context.run_config

    # Charger la configuration
    num_rounds      = int(cfg["num-server-rounds"])
    fraction_train  = float(cfg["fraction-train"])
    seed            = int(cfg["seed"])
    output_dir      = str(cfg.get("output-dir", "results"))
    strategy_name   = str(cfg.get("strategy", "fedavg")).lower()
    malicious_ratio = float(cfg.get("malicious-ratio", 0.0))
    num_clients     = int(cfg.get("num-clients", 50))
    run_id          = str(context.run_id)

    # Nombre de clients à considérer comme malveillants (pour Krum)
    f = int(num_clients * malicious_ratio)

    # Initialiser le modèle global et les résultats
    global_model   = CustomFashionModel()
    global_weights = global_model.state_dict()
    rng            = random.Random(seed)
    results        = []

    print(f"\n  Strategy : {strategy_name.upper()}  |  malicious-ratio={malicious_ratio}  |  f={f}")

    # ── Boucles principale sur les rounds ──────────────────────────────
    for round_num in range(1, num_rounds + 1):

        # ── Phase d'entraînement ──────────────────────────────────
        # Samplernumpy une fraction des clients
        all_ids     = [str(nid) for nid in grid.get_node_ids()]
        sampled_ids = sample_clients(all_ids, fraction_train, rng)

        # Construire les messages d'entraînement pour les clients samplesé
        train_msgs = [
            Message(
                content=RecordDict({"arrays": ArrayRecord(global_weights)}),
                message_type="train",
                dst_node_id=int(nid),
                group_id=str(round_num),
            )
            for nid in sampled_ids
        ]
        
        # Envoyer et attendre les réponses des clients
        train_replies = list(grid.send_and_receive(train_msgs))

        # Extraire les poids et métriques reçus
        state_dicts, n_train, train_metrics = [], [], []
        for reply in train_replies:
            state_dicts.append(reply.content["arrays"].to_torch_state_dict())
            m = reply.content["metrics"]
            n = int(m["num_examples"])
            n_train.append(n)
            train_metrics.append({
                "train_loss": float(m["train_loss"]),
                "train_acc":  float(m["train_acc"]),
            })
                "train_acc":  float(m["train_acc"]),
            })

        # ── Phase d'agrégation ──────────────────────────────────
        # Choisir la stratégie d'agrégation selon la config
        if strategy_name == "fedmedian":
            global_weights = fedmedian(state_dicts, n_train)
        elif strategy_name == "krum":
            global_weights = krum(state_dicts, n_train, f)
        else:  # Default: FedAvg
            global_weights = fedavg(state_dicts, n_train)

        # Agréger les métriques d'entraînement (moyenne pondérée)
        agg_train = aggregate_metrics(train_metrics, n_train)

        # ── Phase d'évaluation ───────────────────────────────────
        # Construire les messages d'évaluation
        eval_msgs = [
            Message(
                content=RecordDict({"arrays": ArrayRecord(global_weights)}),
                message_type="evaluate",
                dst_node_id=int(nid),
                group_id=str(round_num),
            )
            for nid in sampled_ids
        ]
        eval_replies = list(grid.send_and_receive(eval_msgs))

        # Extraire les métriques d'évaluation
        eval_metrics, n_eval = [], []
        for reply in eval_replies:
            m = reply.content["metrics"]
            n = int(m["num_examples"])
            n_eval.append(n)
            eval_metrics.append({
                "eval_loss": float(m["eval_loss"]),
                "eval_acc":  float(m["eval_acc"]),
            })
        agg_eval = aggregate_metrics(eval_metrics, n_eval)

        # ── Logging ──────────────────────────────────────────────────────
        # Construire une ligne de résultats
        row = {
            "round":         round_num,
            "num_clients":   len(sampled_ids),
            "train_loss":    agg_train["train_loss"],
            "train_acc":     agg_train["train_acc"],
            "fed_eval_loss": agg_eval["eval_loss"],
            "fed_eval_acc":  agg_eval["eval_acc"],
        }
        results.append(row)
        
        # Afficher les métriques du round
        print(
            f"[Round {round_num:>2}/{num_rounds}] "
            f"train_loss={row['train_loss']:.4f} | "
            f"train_acc={row['train_acc']:.4f} | "
            f"eval_loss={row['fed_eval_loss']:.4f} | "
            f"eval_acc={row['fed_eval_acc']:.4f}"
        )

    # ── Sauvegarde des résultats ────────────────────────────────────────
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    result_file = out / f"{run_id}.json"
    with open(result_file, "w") as f_out:
        json.dump(sanitize(results), f_out, indent=2)
    print(f"\nResults saved to {result_file}")
