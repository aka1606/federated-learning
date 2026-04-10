"""
analysis_tp2_final.py — TP2 Analyse finale : FedAvg vs FedProx vs SCAFFOLD

Compare les 3 algorithmes sur les 3 valeurs d'alpha.
C'est le graphique de conclusion du TP2.

Sources :
  results/vary_alpha/  ← FedAvg  (Part 1)
  results/fedprox/     ← FedProx (Part 2)
  results/scaffold/    ← SCAFFOLD (Part 3)

Sorties :
  - Tableau récapitulatif avec convergence
  - Graphiques globaux (9 courbes)
  - Graphiques par alpha (3 courbes : un algo par courbe)

Usage :
    python scripts/analysis_tp2_final.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
from prettytable import PrettyTable


def load(path: Path) -> List[dict]:
    """Charge un fichier JSON de résultats."""
    with open(path) as f:
        return json.load(f)


def convergence_round(rounds: List[dict], metric: str = "fed_eval_acc", threshold: float = 0.80) -> int:
    """
    Retourne le round auquel la métrique dépasse le seuil pour la première fois.

    Utilisé pour mesurer la VITESSE de convergence des algorithmes.
    Ex: SCAFFOLD atteint-il 80% d'accuracy avant FedAvg ?

    Retourne -1 si le seuil n'est jamais atteint (ex: en cas de non-IID extrême).
    """
    for r in rounds:
        if r.get(metric, 0) >= threshold:
            return r["round"]  # Premier round où la métrique ≥ threshold
    return -1  # Le seuil n'a jamais été atteint


def plot_metric(runs: Dict[str, List[dict]], metric: str, fig_path: Path, title: str) -> None:
    """
    Trace une métrique pour tous les runs avec un code couleur et style cohérent.

    Code couleur : bleu=FedAvg, orange=FedProx, vert=SCAFFOLD
    Style de ligne : continu=α=100, tirets=α=1, pointillé=α=0.01
    Permet de lire le graphique même à 9 courbes simultanées.
    """
    fig_path.parent.mkdir(parents=True, exist_ok=True)

    # Code couleur par algorithme
    colors  = {"FedAvg": "tab:blue", "FedProx": "tab:orange", "SCAFFOLD": "tab:green"}
    # Style de ligne par niveau d'hétérogénéité
    styles  = {"α=100": "-", "α=1": "--", "α=0.01": ":"}

    plt.figure(figsize=(10, 5))
    for name, rounds in runs.items():
        # Détecter l'algorithme et l'alpha depuis le nom du run
        algo  = next((a for a in colors if a in name), "FedAvg")
        alpha = next((a for a in styles if a in name), "α=1")
        x = [r["round"] for r in rounds]
        y = [r.get(metric, float("nan")) for r in rounds]
        plt.plot(x, y, label=name, color=colors[algo],
                 linestyle=styles[alpha], marker="o", markersize=2)
    plt.xlabel("Round")
    plt.ylabel(metric.replace("_", " ").title())
    plt.title(title)
    plt.legend(fontsize=7, ncol=3)  # 3 colonnes pour tenir en largeur
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(fig_path, dpi=120)
    plt.close()
    print(f"  Saved: {fig_path}")


def main() -> None:
    base         = Path(__file__).parent.parent
    fedavg_dir   = base / "results" / "vary_alpha"
    fedprox_dir  = base / "results" / "fedprox"
    scaffold_dir = base / "results" / "scaffold"
    fig_dir      = base / "results" / "figures" / "final"

    alphas = [("alpha=100", "α=100"), ("alpha=1", "α=1"), ("alpha=0.01", "α=0.01")]
    algos  = [("FedAvg",  fedavg_dir), ("FedProx", fedprox_dir), ("SCAFFOLD", scaffold_dir)]

    # Vérifier que tous les 9 fichiers de résultats existent (3 algos × 3 alphas)
    missing = [
        str(d / f"{fname}.json")
        for fname, _ in alphas
        for _, d in algos
        if not (d / f"{fname}.json").exists()
    ]
    if missing:
        print("Missing result files:")
        for m in missing:
            print(f"  {m}")
        return

    # Charger tous les runs
    all_runs: Dict[str, List[dict]] = {}
    for fname, label in alphas:
        for algo, d in algos:
            all_runs[f"{algo} {label}"] = load(d / f"{fname}.json")

    # ── Tableau récapitulatif avec métriques de convergence ───────────────────
    print("\n" + "=" * 80)
    print("  TP2 Final — FedAvg vs FedProx vs SCAFFOLD — Summary")
    print("=" * 80)

    table = PrettyTable()
    table.field_names = [
        "Algorithm", "α",
        "Final Train Acc",       # Précision d'entraînement final
        "Final Eval Acc",        # Précision d'évaluation finale
        "Best Eval Acc",         # Meilleure précision sur tous les rounds
        "Convergence (≥80%)"     # Round où 80% d'accuracy est atteint
    ]
    for fname, label in alphas:
        for algo, d in algos:
            rounds = load(d / f"{fname}.json")
            last   = rounds[-1]
            best   = max(r["fed_eval_acc"] for r in rounds)  # Meilleur accuracy atteint
            conv   = convergence_round(rounds)                # Round de convergence
            conv_s = f"round {conv}" if conv > 0 else "never"
            table.add_row([
                algo, label,
                f"{last['train_acc']:.4f}",
                f"{last['fed_eval_acc']:.4f}",
                f"{best:.4f}",
                conv_s,
            ])
        # Séparateur visuel entre les alphas
        table.add_row(["-"*8]*6)
    print(table)

    # ── Graphiques globaux : toutes les 9 courbes ensemble ────────────────────
    # Permet de voir d'un coup l'effet combiné de l'algorithme et de l'hétérogénéité
    print("\nGenerating global comparison plots...")
    for metric in ["fed_eval_acc", "fed_eval_loss", "train_loss", "train_acc"]:
        plot_metric(
            all_runs, metric,
            fig_dir / f"global_{metric}.png",
            f"FedAvg vs FedProx vs SCAFFOLD — {metric}"
        )

    # ── Graphiques par alpha : 3 algorithmes pour chaque valeur d'alpha ───────
    # Permet de voir comment les algorithmes se comportent sous une hétérogénéité spécifique
    print("\nGenerating per-alpha plots...")
    for fname, label in alphas:
        pair = {
            f"FedAvg {label}":   load(fedavg_dir  / f"{fname}.json"),
            f"FedProx {label}":  load(fedprox_dir / f"{fname}.json"),
            f"SCAFFOLD {label}": load(scaffold_dir / f"{fname}.json"),
        }
        for metric in ["fed_eval_acc", "fed_eval_loss"]:
            plot_metric(
                pair, metric,
                fig_dir / fname / f"{metric}.png",
                f"{label} — {metric}"
            )

    print(f"\nAll plots saved to: {fig_dir}")


if __name__ == "__main__":
    main()
