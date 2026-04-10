"""
tp4example.server_app: ServerApp — Serveur FedPer (agrège seulement les couches BASE)

Rôle du serveur FedPer:
  1. Maintenir les poids GLOBAUX des couches BASE
  2. Envoyer les couches base aux clients
  3. Recevoir les mises à jour des couches base uniquement
  4. Agréger (FedAvg) les mises à jour base
  5. Agréger les métriques locales (mean/std des accuracies)
  6. Logger les résultats

NOTE: Les couches personnalisées ne sont JAMAIS envoyées au serveur.
Elles restent complètement locales chez les clients.
"""

import json
import math
import random
from pathlib import Path

import numpy as np
import torch

from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
from flwr.serverapp import Grid, ServerApp

from tp4example.model import Net

app = ServerApp()


def _base_keys(num_personalization_layers: int) -> list[str]:
    """Retourne les clés du state_dict pour les couches BASE."""
    kp = min(num_personalization_layers, NUM_LAYERS)
    base_names = LAYER_NAMES[:NUM_LAYERS - kp]
    return [f"{n}.{p}" for n in base_names for p in ("weight", "bias")]


def sample_clients(node_ids, fraction, rng):
    """
    Sélectionne aléatoirement une fraction des clients disponibles.
    """
    return rng.sample(node_ids, k)


def fedavg_base(state_dicts: list[dict], num_examples: list[int]) -> dict:
    """
    Agrège les couches BASE selon FedAvg (moyenne pondérée).
    
    Seules les couches base sont traitées (les pers restent locales).
    """
    total = sum(num_examples)
    aggregated = {}
    for key in state_dicts[0]:
        aggregated[key] = sum(
            state_dicts[i][key].float() * (num_examples[i] / total)
            for i in range(len(state_dicts))
        )
    return aggregated


def sanitize(obj):
    """
    Remplace NaN/Inf par None pour que le JSON reste valide.
    """
        return None
    if isinstance(obj, dict):
        return {k: sanitize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [sanitize(v) for v in obj]
    return obj


@app.main()
def main(grid: Grid, context: Context) -> None:
    """
    Fonction principale du serveur FedPer.
    
    Orchestre num_rounds tours de:
      1. Sampleler les clients
      2. Envoyer les couches base globales
      3. Recevoir les mises à jour base + métriques
      4. Agréger les mises à jour base (FedAvg)
      5. Agréger les métriques (mean local acc, std local acc, loss)
      6. Logger pour cette ronde
    
    Output:
      Fichier JSON des résultats (1 ligne par round avec métriques agrégées)
    """

    # Initialize only base layer parameters
    init_model    = Net()
    base_keys     = _base_keys(kp)
    full_sd       = init_model.state_dict()
    global_base   = {k: full_sd[k] for k in base_keys}

    rng     = random.Random(seed)
    results = []

    for round_num in range(1, num_rounds + 1):

        # ── Train ────────────────────────────────────────────────────
        all_ids     = [str(nid) for nid in grid.get_node_ids()]
        sampled_ids = sample_clients(all_ids, fraction_train, rng)

        train_msgs = [
            Message(
                content=RecordDict({"arrays": ArrayRecord(global_base)}),
                message_type="train",
                dst_node_id=int(nid),
                group_id=str(round_num),
            )
            for nid in sampled_ids
        ]
        train_replies = list(grid.send_and_receive(train_msgs))

        base_state_dicts, n_train, train_metrics = [], [], []
        for reply in train_replies:
            base_state_dicts.append(reply.content["arrays"].to_torch_state_dict())
            m = reply.content["metrics"]
            n = int(m["num_examples"])
            n_train.append(n)
            train_metrics.append({
                "train_loss": float(m["train_loss"]),
                "train_acc":  float(m["train_acc"]),
            })

        # Aggregate base layers only
        global_base = fedavg_base(base_state_dicts, n_train)

        total = sum(n_train)
        agg_train_loss = sum(train_metrics[i]["train_loss"] * n_train[i] / total
                             for i in range(len(n_train)))
        agg_train_acc  = sum(train_metrics[i]["train_acc"]  * n_train[i] / total
                             for i in range(len(n_train)))

        # ── Eval ─────────────────────────────────────────────────────
        eval_msgs = [
            Message(
                content=RecordDict({"arrays": ArrayRecord(global_base)}),
                message_type="evaluate",
                dst_node_id=int(nid),
                group_id=str(round_num),
            )
            for nid in sampled_ids
        ]
        eval_replies = list(grid.send_and_receive(eval_msgs))

        eval_accs, n_eval = [], []
        for reply in eval_replies:
            m = reply.content["metrics"]
            eval_accs.append(float(m["eval_acc"]))
            n_eval.append(int(m["num_examples"]))

        # Mean and std of local accuracies across clients
        acc_array       = np.array(eval_accs)
        mean_local_acc  = float(acc_array.mean())
        std_local_acc   = float(acc_array.std())

        total_eval = sum(n_eval)
        agg_eval_loss = sum(
            float(eval_replies[i].content["metrics"]["eval_loss"]) * n_eval[i] / total_eval
            for i in range(len(n_eval))
        )

        # ── Log ──────────────────────────────────────────────────────
        row = {
            "round":           round_num,
            "num_clients":     len(sampled_ids),
            "train_loss":      agg_train_loss,
            "train_acc":       agg_train_acc,
            "mean_local_acc":  mean_local_acc,
            "std_local_acc":   std_local_acc,
            "eval_loss":       agg_eval_loss,
        }
        results.append(row)
        print(
            f"[Round {round_num:>2}/{num_rounds}] "
            f"train_acc={agg_train_acc:.4f} | "
            f"local_acc={mean_local_acc:.4f} ± {std_local_acc:.4f}"
        )

    # ── Save ─────────────────────────────────────────────────────────
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    result_file = out / f"{run_id}.json"
    with open(result_file, "w") as f:
        json.dump(sanitize(results), f, indent=2)
    print(f"\nResults saved to {result_file}")
