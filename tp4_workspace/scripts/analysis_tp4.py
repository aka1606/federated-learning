"""
analysis_tp4.py — Analyse des résultats FedPer (TP4)

But: Générer des graphiques et tableaux pour les deux parties.

Résultats utilisés:
  PART A: results/fedper/kp_variation/ (KP=0,1,2,3 × alpha=0.1,10)
  PART B: results/fedper/partial_training/ (fraction=1.0,0.5,0.2)

Output:
  Tableaux ASCII récapitulatifs
  Graphiques avec courbes mean_local_acc ± std_local_acc
  Sauvegardés dans results/figures/fedper/

Usage:
    python scripts/analysis_tp4.py
"""

from __future__ import annotations
import json
import math
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from prettytable import PrettyTable

BASE      = Path(__file__).parent.parent
PART_A_DIR = BASE / "results" / "fedper" / "kp_variation"     # KP variation
PART_B_DIR = BASE / "results" / "fedper" / "partial_training" # Partial training
FIG_BASE   = BASE / "results" / "figures" / "fedper"

COLORS = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple"]


def load(path: Path) -> Optional[List[dict]]:
    """
    Charge un fichier JSON de résultats.
    
    Retourne None si le fichier n'existe pas.
    """
        return None
    with open(path) as f:
        return json.load(f)


def plot_with_std(
    runs: Dict[str, List[dict]],
    fig_dir: Path,
    title: str,
    filename: str,
) -> None:
    """
    Génère un graphique avec bandes d'incertitude (mean ± std).
    
    Trace la métrique mean_local_acc avec la bande std_local_acc.
    Chaque courbe représente une configuration différente.
    """
    fig_dir.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 5))
    for (name, data), color in zip(runs.items(), COLORS):
        if data is None:
            continue
        x    = [r["round"] for r in data]
        mean = [r.get("mean_local_acc", float("nan")) for r in data]
        std  = [r.get("std_local_acc",  0.0) for r in data]
        mean_arr = np.array(mean)
        std_arr  = np.array(std)
        plt.plot(x, mean_arr, label=name, color=color, marker="o", markersize=2)
        plt.fill_between(x, mean_arr - std_arr, mean_arr + std_arr,
                         alpha=0.15, color=color)
    plt.xlabel("Round")
    plt.ylabel("Local Accuracy (mean ± std)")
    plt.title(title)
    plt.legend(fontsize=8)
    plt.grid(True)
    plt.tight_layout()
    out = fig_dir / f"{filename}.png"
    plt.savefig(out, dpi=120)
    plt.close()
    print(f"  Saved: {out}")


def summary_table(runs: Dict[str, Optional[List[dict]]], title: str) -> None:
    """
    Affiche un tableau ASCII de synthèse.
    
    Colonnes:
      - Run: nom de la config
      - Final Mean/Std Acc: accuracy locale finale (mean ± std)
      - Final Train Acc: accuracy d'entraînement finale
    """
    print("=" * 65)
    table = PrettyTable()
    table.field_names = ["Run", "Final Mean Acc", "Final Std Acc", "Final Train Acc"]
    for name, data in runs.items():
        if data is None:
            table.add_row([name, "N/A", "N/A", "N/A"])
            continue
        last = data[-1]
        def fmt(v): return f"{v:.4f}" if v is not None and not math.isnan(v) else "NaN"
        table.add_row([name,
                       fmt(last.get("mean_local_acc")),
                       fmt(last.get("std_local_acc")),
                       fmt(last.get("train_acc"))])
    print(table)


def part_a():
    """
    PART A—Impact de KP (nombre de couches personnalisées).
    
    Pour chaque alpha ∈ {0.1, 10}:
      - Charger les résultats des 4 configs (KP=0,1,2,3)
      - Afficher tableau comparant les KP
      - Générer graphique
    """
    print("=" * 65)

    for alpha in [0.1, 10.0]:
        runs = {}
        for kp in [0, 1, 2, 3]:
            label = f"KP={kp}" + (" (FedAvg)" if kp == 0 else "")
            fname = f"kp={kp}_alpha={alpha}.json"
            runs[label] = load(PART_A_DIR / fname)

        title    = f"FedPer — alpha={alpha} — KP variation"
        filename = f"kp_variation_alpha={alpha}"
        summary_table(runs, title)
        plot_with_std(runs, FIG_BASE / "kp_variation", title, filename)


def part_b():
    """
    PART B—Impact du partial training (fraction d'entraînement).
    
    Étude de la robustesse avec fraction ∈ {1.0, 0.5, 0.2}:
      - Charger les résultats des 3 configs
      - Afficher tableau comparant les fractions
      - Générer graphique
    """
    print("=" * 65)

    runs = {}
    for fraction in [1.0, 0.5, 0.2]:
        label = f"fraction={fraction}"
        fname = f"fraction={fraction}_alpha=0.1_kp=1.json"
        runs[label] = load(PART_B_DIR / fname)

    title    = "FedPer — Partial training (alpha=0.1, KP=1)"
    filename = "partial_training"
    summary_table(runs, title)
    plot_with_std(runs, FIG_BASE / "partial_training", title, filename)


def main():
    """
    Fonction principale de l'analyse TP4.
    
    Orchestre les deux analyses:
      1. PART A: Impact du nombre de couches personnalisées
      2. PART B: Impact du partial training
    
    Output:
      Tableaux affichés dans le terminal
      Graphiques sauvegardés dans results/figures/fedper/
    """
    print(f"\n  All figures saved under: {FIG_BASE}")


if __name__ == "__main__":
    main()
