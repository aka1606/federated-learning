"""
analysis_tp3_defenses.py — Analyse des défenses byzantines (FedMedian, Krum)

But: Comparer FedAvg vs FedMedian vs Krum sous attaques et hétérogénéité.

Résultats utilisés:
  PART A: results/defenses/strategy_comparison/ (data/model × 0%,25%,50%)
  PART B: results/defenses/heterogeneity/ (alpha ∈ {10,1,0.1} × strategies)

Output:
  - Tableaux de synthèse (Part A + Part B)
  - Graphiques de comparaison (fed_eval_acc, fed_eval_loss, train_acc)
  - Sauvegardés dans results/figures/defenses/{strategy_comparison,heterogeneity}/

Usage:
    python scripts/analysis_tp3_defenses.py
"""

from __future__ import annotations
import json
import math
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
from prettytable import PrettyTable

BASE = Path(__file__).parent.parent
ATTACKS_DIR  = BASE / "results" / "attacks"        # Résultats FedAvg (Part 1)
STRATEGY_DIR = BASE / "results" / "defenses" / "strategy_comparison"  # Part A
HETERO_DIR   = BASE / "results" / "defenses" / "heterogeneity"        # Part B
FIG_BASE     = BASE / "results" / "figures" / "defenses"


def load(path: Path) -> Optional[List[dict]]:
    """
    Charge un fichier JSON et nettoie les données NaN.
    
    Retourne:
      Liste des rounds ou None si le fichier n'existe pas
    """
    with open(path) as f:
        data = json.load(f)
    for r in data:
        for k, v in r.items():
            if v is None:
                r[k] = float("nan")
    return data


def plot_comparison(runs: Dict[str, List[dict]], metric: str,
                    fig_dir: Path, title: str, filename: str) -> None:
    """
    Génère un graphique de comparaison pour plusieurs stratégies.
    
    Paramètres:
      runs: dict { "FedAvg" : [...], "FedMedian" : [...], "Krum" : [...] }
      metric: clé à tracer (ex: "fed_eval_acc")
      fig_dir, title, filename: configuration du fichier de sortie
    
    Affiche les courbes de toutes les stratégies qui ont des données.
    """
    plt.figure(figsize=(10, 5))
    for name, data in runs.items():
        if data is None:
            continue
        x = [r["round"] for r in data]
        y = [r.get(metric, float("nan")) for r in data]
        plt.plot(x, y, label=name, marker="o", markersize=2)
    plt.xlabel("Round")
    plt.ylabel(metric.replace("_", " ").title())
    plt.title(title)
    plt.legend(fontsize=7)
    plt.grid(True)
    plt.tight_layout()
    out = fig_dir / f"{filename}.png"
    plt.savefig(out, dpi=120)
    plt.close()
    print(f"  Saved: {out}")


def summary_table(runs: Dict[str, Optional[List[dict]]], title: str) -> None:
    """
    Affiche un tableau ASCII de synthèse des résultats.
    
    Colonnes:
      - Scenario: nom du scénario (ex: "FedAvg", "FedMedian", etc.)
      - Final Train/Eval Acc: dernière accuracy
      - Final Eval Loss: dernère loss
    
    Si les données sont None (fichier manquant), affiche "N/A".
    """
    table = PrettyTable()
    table.field_names = ["Scenario", "Final Train Acc", "Final Eval Acc", "Final Eval Loss"]
    for name, data in runs.items():
        if data is None:
            table.add_row([name, "N/A", "N/A", "N/A"])
            continue
        last = data[-1]
        def fmt(v):
            return f"{v:.4f}" if v is not None and not math.isnan(v) else "NaN"
        table.add_row([name,
                        fmt(last.get("train_acc")),
                        fmt(last.get("fed_eval_acc")),
                        fmt(last.get("fed_eval_loss"))])
    print(table)


# ── Part A helpers ────────────────────────────────────────────────────────────
# Fonctions pour charger et afficher les résultats de Part A (comparaison stratégies)

def load_strategy_runs(attack: str, ratio: float):
    """
    Charge les résultats de FedAvg, FedMedian, Krum pour un scénario d'attaque.
    
    Paramètres:
      attack: "data" ou "model"
      ratio: 0.0, 0.25, 0.50
    
    Processus:
      - Charger FedAvg depuis results/attacks/ (results de Part 1)
      - Charger FedMedian depuis results/defenses/strategy_comparison/
      - Charger Krum depuis results/defenses/strategy_comparison/
      - Pour baseline (ratio=0), Krum = FedMedian (même résultat sans attaque)
    """
    ratio_str = f"{ratio:.2f}"
    if ratio == 0.0:
        fedavg_path = ATTACKS_DIR / "baseline.json"
    else:
        fedavg_path = ATTACKS_DIR / f"{attack}_ratio={ratio_str[1:]}.json"
        # attacks/ uses ratio without leading zero: e.g. data_ratio=0.25
        # try both formats
        if not fedavg_path.exists():
            short = f"{ratio:.2f}".lstrip("0") or "0"
            fedavg_path = ATTACKS_DIR / f"{attack}_ratio={short}.json"

    runs = {
        "FedAvg":    load(fedavg_path),
        "FedMedian": load(STRATEGY_DIR / f"fedmedian_{attack}_ratio={ratio_str}.json"),
        "Krum":      load(STRATEGY_DIR / f"krum_{attack}_ratio={ratio_str}.json"),
    }
    # Krum baseline = same as FedMedian baseline (no attack)
    if ratio == 0.0:
        runs["Krum"] = load(STRATEGY_DIR / f"fedmedian_{attack}_ratio={ratio_str}.json")
    return runs


def part_a():
    """
    PART A — Comparaison des stratégies sous attaques (alpha=10 fixe).
    
    Processus:
      Pour chaque combinaison (attack × ratio):
        1. Charger les résultats de FedAvg, FedMedian, Krum
        2. Afficher un tableau récapitulatif
        3. Générer 3 graphiques (eval_acc, eval_loss, train_acc)
    
    Résultats sauvegardés dans : results/figures/defenses/strategy_comparison/
    """
    print("=" * 70)

    for attack in ["data", "model"]:
        for ratio in [0.0, 0.25, 0.50]:
            label = f"{attack.capitalize()} Poison — {int(ratio*100)}% malicious"
            runs  = load_strategy_runs(attack, ratio)

            fig_dir  = FIG_BASE / "strategy_comparison"
            filename = f"{attack}_ratio={ratio:.2f}"

            summary_table(runs, label)
            plot_comparison(runs, "fed_eval_acc",  fig_dir,
                            f"{label} — Eval Accuracy",  f"{filename}_eval_acc")
            plot_comparison(runs, "fed_eval_loss", fig_dir,
                            f"{label} — Eval Loss",      f"{filename}_eval_loss")
            plot_comparison(runs, "train_acc",     fig_dir,
                            f"{label} — Train Accuracy", f"{filename}_train_acc")


# ── Part B helpers ────────────────────────────────────────────────────────────
# Fonctions pour afficher les résultats de Part B (impact hétérogénéité)

def part_b():
    """
    PART B — Impact de l'hétérogénéité (25% malveillants, data poisoning fixe).
    
    Processus:
      1. Pour chaque stratégie (FedAvg, FedMedian, Krum):
        - Charger les résultats pour alpha ∈ {10, 1, 0.1}
        - Afficher tableau comparant les alphas
        - Générer 2 graphiques (eval_acc, eval_loss)
      
      2. Pour chaque alpha ∈ {10, 1, 0.1}:
        - Afficher tableau comparant les stratégies
        - Générer graphique de comparaison des stratégies
    
    Résultats sauvegardés dans : results/figures/defenses/heterogeneity/
    """
    print("=" * 70)

    for strategy in ["fedavg", "fedmedian", "krum"]:
        runs = {}
        for alpha in [10.0, 1.0, 0.1]:
            fname = f"hetero_alpha={alpha}_{strategy}.json"
            runs[f"alpha={alpha}"] = load(HETERO_DIR / fname)

        label    = f"{strategy.upper()} — varying alpha (25% data poison)"
        fig_dir  = FIG_BASE / "heterogeneity"
        filename = f"{strategy}_hetero"

        summary_table(runs, label)
        plot_comparison(runs, "fed_eval_acc",  fig_dir,
                        f"{label} — Eval Accuracy",  f"{filename}_eval_acc")
        plot_comparison(runs, "fed_eval_loss", fig_dir,
                        f"{label} — Eval Loss",      f"{filename}_eval_loss")

    # Cross-strategy view for each alpha
    for alpha in [10.0, 1.0, 0.1]:
        runs = {}
        for strategy in ["fedavg", "fedmedian", "krum"]:
            fname = f"hetero_alpha={alpha}_{strategy}.json"
            runs[strategy.upper()] = load(HETERO_DIR / fname)
        label    = f"alpha={alpha} — strategy comparison (25% data poison)"
        fig_dir  = FIG_BASE / "heterogeneity"
        filename = f"alpha={alpha}_strategies"
        summary_table(runs, label)
        plot_comparison(runs, "fed_eval_acc", fig_dir,
                        label + " — Eval Acc", f"{filename}_eval_acc")


# ── Point d'entrée ────────────────────────────────────────────────────────────

def main():
    """
    Fonction principale de l'analyse des défenses.
    
    Orchestre deux analyses complémentaires:
      1. PART A : Comparer les stratégies sous des attaques spécifiques (alpha fixe)
      2. PART B : Étudier l'impact de l'hétérogénéité sur les stratégies
    
    Output:
      Tableaux ASCII affichés dans le terminal
      Graphiques sauvegardés dans results/figures/defenses/
    """
    part_a()
    part_b()
    print(f"\n  All figures saved under: {FIG_BASE}")


if __name__ == "__main__":
    main()
