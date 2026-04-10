"""
analysis_tp3.py — Analyse des résultats FedAvg sous attaques de poisoning

But: Charger les résultats JSON de run_tp3_attacks.py et générer des visualisations
+ tableau récapitulatif.

Résultats utilisés:
  results/attacks/baseline.json
  results/attacks/data_ratio=0.25.json
  results/attacks/data_ratio=0.50.json
  results/attacks/model_ratio=0.25.json
  results/attacks/model_ratio=0.50.json

Output:
  1. Tableau de synthèse (final accuracy, loss, etc.)
  2. 4 graphiques (fed_eval_acc, fed_eval_loss, train_acc, train_loss)
     sauvegardés dans results/figures/attacks/

Usage:
    python scripts/analysis_tp3.py
"""

from __future__ import annotations
import json
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
from prettytable import PrettyTable


def load(path: Path) -> List[dict]:
    """
    Charge un fichier JSON et nettoie les données.
    
    Processus:
      1. Lire le JSON (contient une liste de rounds)
      2. Remplacer None (NaN sérialisé) par float("nan") pour les plots
    
    Retourne:
      Liste de dicts {round, train_loss, train_acc, fed_eval_loss, fed_eval_acc}
    """


RUNS = [
    ("Baseline (0%)",          "baseline"),
    ("Data poison  25%",       "data_ratio=0.25"),
    ("Data poison  50%",       "data_ratio=0.50"),
    ("Model poison 25%",       "model_ratio=0.25"),
    ("Model poison 50%",       "model_ratio=0.50"),
]

COLORS = {
    "Baseline (0%)":    "tab:blue",
    "Data poison  25%": "tab:orange",
    "Data poison  50%": "tab:red",
    "Model poison 25%": "tab:green",
    "Model poison 50%": "tab:purple",
}


def plot(runs: Dict[str, List[dict]], metric: str, fig_dir: Path, title: str) -> None:
    """
    Génère un graphique de comparaison pour une métrique donnée.
    
    Paramètres:
      runs: dict { "Scenario name" : [list of rounds with metrics] }
      metric: clé dans les dicts des rounds (ex: "fed_eval_acc")
      fig_dir: répertoire de sortie des figures
      title: titre du graphique
    
    Output:
      Fichier PNG sauvegardé dans fig_dir/{metric}.png
    """
    fig_dir.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(9, 5))
    for name, data in runs.items():
        x = [r["round"] for r in data]
        y = [r.get(metric) for r in data]
        plt.plot(x, y, label=name, color=COLORS.get(name), marker="o", markersize=2)
    plt.xlabel("Round")
    plt.ylabel(metric.replace("_", " ").title())
    plt.title(title)
    plt.legend(fontsize=8)
    plt.grid(True)
    plt.tight_layout()
    out = fig_dir / f"{metric}.png"
    plt.savefig(out, dpi=120)
    plt.close()
    print(f"  Saved: {out}")


def main():
    """
    Fonction principale de l'analyse TP3.
    
    Étapes:
      1. Vérifier que tous les fichiers JSON existent
      2. Les charger et les regrouper par scénario
      3. Générer un tableau de synthèse (dernière accuracy/loss)
      4. Générer 4 graphiques (1 par métrique)
      5. Afficher où les fichiers sont sauvegardés
    """
    res_dir  = base / "results" / "attacks"
    fig_dir  = base / "results" / "figures" / "attacks"

    missing = [str(res_dir / f"{r}.json") for _, r in RUNS if not (res_dir / f"{r}.json").exists()]
    if missing:
        print("Missing files:"); [print(f"  {m}") for m in missing]
        return

    runs = {name: load(res_dir / f"{fname}.json") for name, fname in RUNS}

    # ── Summary table ──────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  TP3 — Attack Scenarios Summary")
    print("=" * 70)
    table = PrettyTable()
    table.field_names = ["Scenario", "Final Train Acc", "Final Eval Acc", "Final Eval Loss"]
    for name, data in runs.items():
        last = data[-1]
        table.add_row([
            name,
            f"{last.get('train_acc', float('nan')):.4f}",
            f"{last.get('fed_eval_acc', float('nan')):.4f}",
            f"{last.get('fed_eval_loss', float('nan')):.4f}",
        ])
    print(table)

    # ── Plots ─────────────────────────────────────────────────────────
    print("\nGenerating plots...")
    for metric in ["fed_eval_acc", "fed_eval_loss", "train_acc", "train_loss"]:
        plot(runs, metric, fig_dir, f"FedAvg under attacks — {metric}")

    print(f"\nAll plots saved to: {fig_dir}")


if __name__ == "__main__":
    main()
