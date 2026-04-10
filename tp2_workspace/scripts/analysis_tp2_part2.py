"""
analysis_tp2_part2.py — TP2 Partie 2 : Comparaison FedAvg vs FedProx

Charge les résultats des deux algorithmes pour les 3 valeurs d'alpha
et génère des graphiques comparatifs côte à côte.

Sources :
  results/vary_alpha/  ← résultats FedAvg (Part 1)
  results/fedprox/     ← résultats FedProx (Part 2)

Graphiques générés :
  - Comparaison globale (toutes les 6 courbes sur le même graphique)
  - Comparaison par alpha (2 courbes : FedAvg vs FedProx pour chaque alpha)

Sauvegardés dans : results/figures/fedprox/

Usage :
    python scripts/analysis_tp2_part2.py
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


def plot_metric(runs: Dict[str, List[dict]], metric: str, fig_dir: Path, title: str) -> None:
    """
    Trace une métrique pour tous les runs avec des styles différents par algorithme.

    Style des courbes :
      - FedAvg  : ligne continue (-)  avec cercles (o)
      - FedProx : ligne pointillée (--) avec carrés (s)

    Cela facilite la distinction visuelle même en noir et blanc.
    """
    fig_dir.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(9, 5))

    # Styles différents selon l'algorithme pour comparaison claire
    styles = {
        "FedAvg":  {"linestyle": "-",  "marker": "o"},   # Continu + cercle
        "FedProx": {"linestyle": "--", "marker": "s"},   # Pointillé + carré
    }

    for name, rounds in runs.items():
        # Détecter l'algorithme depuis le nom du run
        algo = "FedProx" if "FedProx" in name else "FedAvg"
        x = [r["round"] for r in rounds]
        y = [r.get(metric, float("nan")) for r in rounds]
        plt.plot(x, y, label=name, markersize=3, **styles[algo])

    plt.xlabel("Round")
    plt.ylabel(metric.replace("_", " ").title())
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    out = fig_dir / f"{metric}.png"
    plt.savefig(out, dpi=120)
    plt.close()
    print(f"  Saved: {out}")


def summary_table(runs: Dict[str, List[dict]]) -> None:
    """Affiche un tableau des métriques finales pour tous les runs."""
    table = PrettyTable()
    table.field_names = ["Run", "Final Train Loss", "Final Train Acc", "Final Eval Loss", "Final Eval Acc"]
    for name, rounds in runs.items():
        last = rounds[-1]
        table.add_row([
            name,
            f"{last.get('train_loss',    float('nan')):.4f}",
            f"{last.get('train_acc',     float('nan')):.4f}",
            f"{last.get('fed_eval_loss', float('nan')):.4f}",
            f"{last.get('fed_eval_acc',  float('nan')):.4f}",
        ])
    print(table)


def main() -> None:
    base        = Path(__file__).parent.parent
    fedavg_dir  = base / "results" / "vary_alpha"  # Résultats FedAvg de Part 1
    fedprox_dir = base / "results" / "fedprox"     # Résultats FedProx de Part 2
    fig_base    = base / "results" / "figures" / "fedprox"

    alphas = [("alpha=100", "α=100"), ("alpha=1", "α=1"), ("alpha=0.01", "α=0.01")]

    # Vérifier que tous les fichiers existent (FedAvg ET FedProx)
    missing = []
    for fname, _ in alphas:
        for d in [fedavg_dir, fedprox_dir]:
            p = d / f"{fname}.json"
            if not p.exists():
                missing.append(str(p))
    if missing:
        print("Missing result files:")
        for m in missing:
            print(f"  {m}")
        return

    # ── Comparaison globale : toutes les 6 courbes ensemble ───────────────────
    # Permet de voir d'un coup d'œil quel algorithme fonctionne mieux pour chaque alpha
    all_runs: Dict[str, List[dict]] = {}
    for fname, label in alphas:
        all_runs[f"FedAvg  {label}"]  = load(fedavg_dir  / f"{fname}.json")
        all_runs[f"FedProx {label}"]  = load(fedprox_dir / f"{fname}.json")

    print("\n" + "=" * 65)
    print("  TP2 Part 2 — FedAvg vs FedProx — Summary Table")
    print("=" * 65)
    summary_table(all_runs)

    print("\nGenerating global comparison plots...")
    for metric in ["train_loss", "train_acc", "fed_eval_loss", "fed_eval_acc"]:
        plot_metric(all_runs, metric, fig_base, f"FedAvg vs FedProx — {metric} (all α)")

    # ── Comparaison par alpha : 2 courbes par graphique (FedAvg vs FedProx) ──
    # Permet de voir précisément l'impact de FedProx pour un alpha donné
    print("\nGenerating per-alpha plots...")
    for fname, label in alphas:
        pair = {
            f"FedAvg  {label}": load(fedavg_dir  / f"{fname}.json"),
            f"FedProx {label}": load(fedprox_dir / f"{fname}.json"),
        }
        fig_dir = fig_base / fname  # Sous-dossier par alpha
        for metric in ["train_loss", "train_acc", "fed_eval_loss", "fed_eval_acc"]:
            plot_metric(pair, metric, fig_dir, f"FedAvg vs FedProx — {metric}  ({label})")

    print(f"\nAll plots saved to: {fig_base}")


if __name__ == "__main__":
    main()
