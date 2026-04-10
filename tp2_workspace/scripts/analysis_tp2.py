# analysis_tp2.py
# TP2 Partie 1 — Analyse des résultats FedAvg sous différentes hétérogénéités.
#
# Charge les 3 fichiers JSON de results/vary_alpha/ (un par alpha),
# affiche un tableau récapitulatif et génère 4 graphiques de comparaison.
#
# Usage (depuis tp2_workspace/) :
#   python scripts/analysis_tp2.py

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
from prettytable import PrettyTable


class ResultsVisualizer:
    """
    Analyseur de résultats FL multi-expériences.

    Chaque run = un fichier JSON contenant une liste de dicts par round :
      round, train_loss, train_acc, fed_eval_loss, fed_eval_acc, num_clients
    """

    def __init__(self) -> None:
        # Dictionnaire : nom_run → liste de dicts par round
        self.runs: Dict[str, List[dict]] = {}

    def add_run(self, name: str, file_name: str) -> None:
        """Charge un fichier JSON et l'enregistre sous 'name'."""
        with open(file_name, "r") as f:
            self.runs[name] = json.load(f)
        print(f"Loaded run '{name}' ({len(self.runs[name])} rounds) from {file_name}")

    def print_run_summary_table(self) -> None:
        """
        Affiche un tableau avec les métriques finales de chaque run.
        Prend le DERNIER round de chaque expérience comme résultat final.
        """
        table = PrettyTable()
        table.field_names = [
            "Run",
            "Final Train Loss",  # Perte sur données d'entraînement
            "Final Train Acc",   # Précision d'entraînement
            "Final Eval Loss",   # Perte sur données de validation
            "Final Eval Acc",    # Précision de validation ← métrique principale
        ]
        for name, rounds in self.runs.items():
            last = rounds[-1]  # Dernier round
            table.add_row([
                name,
                f"{last.get('train_loss',    float('nan')):.4f}",
                f"{last.get('train_acc',     float('nan')):.4f}",
                f"{last.get('fed_eval_loss', float('nan')):.4f}",
                f"{last.get('fed_eval_acc',  float('nan')):.4f}",
            ])
        print(table)

    def plot_metric(self, metric: str, fig_directory: str) -> None:
        """
        Trace l'évolution d'une métrique vs round pour tous les runs.

        Chaque run = une courbe avec markers.
        Sauvegarde en PNG dans fig_directory/<metric>.png
        """
        Path(fig_directory).mkdir(parents=True, exist_ok=True)
        plt.figure(figsize=(9, 5))
        for name, rounds in self.runs.items():
            x = [r["round"] for r in rounds]
            y = [r.get(metric, float("nan")) for r in rounds]
            plt.plot(x, y, marker="o", markersize=3, label=name)
        plt.xlabel("Round")
        plt.ylabel(metric.replace("_", " ").title())
        # Titre explicite indiquant qu'on fait varier α (Dirichlet)
        plt.title(f"FedAvg — {metric} vs Round  (varying α)")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        save_path = str(Path(fig_directory) / f"{metric}.png")
        plt.savefig(save_path, dpi=120)
        plt.close()
        print(f"Saved: {save_path}")

    def plot_all(self, fig_directory: str) -> None:
        """Trace les 4 métriques standard FL."""
        for metric in ["train_loss", "train_acc", "fed_eval_loss", "fed_eval_acc"]:
            self.plot_metric(metric, fig_directory)


# ── Point d'entrée ─────────────────────────────────────────────────────────────

def main() -> None:
    base = Path(__file__).parent.parent   # tp2_workspace/
    results_dir = base / "results" / "vary_alpha"  # Résultats FedAvg
    figures_dir = str(base / "results" / "figures" / "vary_alpha")

    # Ordre croissant d'hétérogénéité pour la lisibilité du graphique
    # alpha élevé = IID = facile, alpha faible = non-IID = difficile
    runs = [
        ("α=100  (low heterogeneity)",      results_dir / "alpha=100.json"),
        ("α=1    (moderate heterogeneity)", results_dir / "alpha=1.json"),
        ("α=0.01 (high heterogeneity)",     results_dir / "alpha=0.01.json"),
    ]

    # Vérifier que tous les fichiers de résultats existent
    missing = [str(p) for _, p in runs if not p.exists()]
    if missing:
        print("Missing result files — run experiments first:")
        for m in missing:
            print(f"  {m}")
        print("\nCommand:  python scripts/run_tp2_part1.py")
        return

    # Charger et afficher les résultats
    viz = ResultsVisualizer()
    for name, path in runs:
        viz.add_run(name, str(path))

    print("\n" + "="*60)
    print("  TP2 Part 1 — Summary Table")
    print("="*60)
    viz.print_run_summary_table()

    print("\nGenerating plots...")
    viz.plot_all(figures_dir)
    print(f"\nAll plots saved to: {figures_dir}")


if __name__ == "__main__":
    main()
