# analysis.py
# Script d'analyse des résultats des simulations Federated Learning.
# Charge plusieurs fichiers JSON (un par expérience), affiche un tableau
# récapitulatif et génère des graphiques de courbes d'apprentissage.
#
# Usage :
#   python3 analysis.py   (depuis tp1_workspace/quickstart-pytorch/)

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt   # Pour tracer les graphiques
from prettytable import PrettyTable  # Pour afficher des tableaux formatés dans le terminal


class ResultsVisualizer:
    """
    Classe principale d'analyse des résultats FL multi-expériences.

    Chaque "run" correspond à une expérience :
      - Identifiée par un nom (ex: "alpha=0.1", "clients=50")
      - Pointe vers un fichier JSON produit par le serveur Flower
      - Ce JSON contient une liste de dicts, un par round, avec les clés :
          round, train_loss, train_acc, fed_eval_loss, fed_eval_acc, num_clients

    Exemple d'utilisation :
        viz = ResultsVisualizer()
        viz.add_run("alpha=0.1", "results/vary_alpha_alpha=0.1.json")
        viz.add_run("alpha=5",   "results/vary_alpha_alpha=5.json")
        viz.print_run_summary_table()
        viz.plot_all("results/figures/vary_alpha")
    """

    def __init__(self) -> None:
        # Dictionnaire : nom_du_run → liste de dicts par round
        # Ex: {"alpha=0.1": [{"round": 1, "train_loss": 0.8, ...}, ...], ...}
        self.runs: Dict[str, List[dict]] = {}

    def add_run(self, name: str, file_name: str) -> None:
        """
        Charge un fichier JSON de résultats et l'enregistre sous 'name'.

        Le fichier JSON est produit automatiquement par server_app.py
        à la fin de chaque simulation Flower.
        """
        with open(file_name, "r") as f:
            self.runs[name] = json.load(f)
        print(f"Loaded run '{name}' ({len(self.runs[name])} rounds) from {file_name}")

    def print_run_summary_table(self) -> None:
        """
        Affiche un tableau récapitulatif des résultats finaux.

        Chaque ligne = une expérience (un run).
        Colonnes : perte d'entraînement, précision d'entraînement,
                   perte d'évaluation fédérée, précision d'évaluation fédérée.

        On prend uniquement le DERNIER round (rounds[-1]) comme métrique finale.
        C'est la valeur obtenue après convergence.
        """
        table = PrettyTable()
        table.field_names = [
            "Run",
            "Final Train Loss",   # Perte sur les données d'entraînement (dernier round)
            "Final Train Acc",    # Précision sur entraînement (dernier round)
            "Final Eval Loss",    # Perte sur données de validation (dernier round)
            "Final Eval Acc",     # Précision sur validation (dernier round) — métrique principale
        ]
        for name, rounds in self.runs.items():
            last = rounds[-1]  # Dernier round = état final du modèle global
            table.add_row([
                name,
                f"{last.get('train_loss', float('nan')):.4f}",
                f"{last.get('train_acc', float('nan')):.4f}",
                f"{last.get('fed_eval_loss', float('nan')):.4f}",
                f"{last.get('fed_eval_acc', float('nan')):.4f}",
            ])
        print(table)

    def plot_metric(self, metric: str, fig_directory: str) -> None:
        """
        Trace l'évolution d'une métrique en fonction du round pour tous les runs.

        Chaque run apparaît comme une courbe sur le même graphique.
        La légende utilise les noms des runs.

        metric : une des clés du JSON, ex: "train_loss", "fed_eval_acc"
        fig_directory : dossier où sauvegarder l'image PNG
        """
        Path(fig_directory).mkdir(parents=True, exist_ok=True)
        plt.figure(figsize=(8, 5))

        for name, rounds in self.runs.items():
            x = [r["round"] for r in rounds]              # Axe X = numéro de round
            y = [r.get(metric, float("nan")) for r in rounds]  # Axe Y = valeur de la métrique
            plt.plot(x, y, marker="o", label=name)        # Une courbe par expérience

        plt.xlabel("Round")                                 # Label axe X
        plt.ylabel(metric.replace("_", " ").title())       # Label axe Y (ex: "Fed Eval Acc")
        plt.title(f"{metric} vs Round")
        plt.legend()                                        # Afficher la légende
        plt.grid(True)
        plt.tight_layout()

        save_path = str(Path(fig_directory) / f"{metric}.png")
        plt.savefig(save_path)  # Sauvegarder en PNG
        plt.close()             # Fermer la figure pour libérer la mémoire
        print(f"Saved plot: {save_path}")

    def plot_all(self, fig_directory: str) -> None:
        """
        Trace les 4 métriques standard FL en une seule commande :
          - train_loss    : convergence de l'entraînement
          - train_acc     : précision locale des clients
          - fed_eval_loss : perte du modèle global sur les données de validation
          - fed_eval_acc  : précision du modèle global (métrique principale à analyser)
        """
        for metric in ["train_loss", "train_acc", "fed_eval_loss", "fed_eval_acc"]:
            self.plot_metric(metric, fig_directory)


# ---------------------------------------------------------------------------
# Fonctions utilitaires
# ---------------------------------------------------------------------------

def compare_group(group_prefix: str, fig_subdir: str) -> None:
    """
    Charge automatiquement tous les fichiers results/<group_prefix>_*.json,
    affiche le tableau récapitulatif et génère les graphiques.

    Exemple : compare_group("vary_alpha", "vary_alpha")
    → charge results/vary_alpha_alpha=0.1.json, results/vary_alpha_alpha=5.json, etc.
    → sauvegarde les graphiques dans results/figures/vary_alpha/
    """
    # Trouver tous les fichiers JSON dont le nom commence par group_prefix
    files = sorted(Path("results").glob(f"{group_prefix}_*.json"))

    if not files:
        print(f"Aucun fichier trouvé pour le groupe '{group_prefix}'.")
        print(f"Lance d'abord : python3 run_experiments.py")
        return

    viz = ResultsVisualizer()
    for f in files:
        # Extraire le label du run depuis le nom du fichier
        # Ex: "vary_alpha_alpha=0.1" → label = "alpha=0.1"
        label = f.stem.replace(f"{group_prefix}_", "")
        viz.add_run(label, str(f))

    print(f"\n{'='*50}")
    print(f"  Groupe : {group_prefix}")
    print(f"{'='*50}")
    viz.print_run_summary_table()
    viz.plot_all(f"results/figures/{fig_subdir}")
    print(f"Graphiques sauvegardés dans results/figures/{fig_subdir}/\n")


def main() -> None:
    """
    Compare tous les groupes d'expériences générés par run_experiments.py.

    Step 7 : Impact des hyperparamètres
      - vary_alpha    : impact de l'hétérogénéité (alpha Dirichlet)
      - vary_clients  : impact du nombre de clients
      - vary_fraction : impact de la fraction de participation

    Step 8 : Comparaison des algorithmes
      - vary_algo     : FedAvg+SGD vs FedAvg+Adam vs FedSGD
    """
    # Step 7 — Hyperparamètres
    compare_group("vary_alpha",    "vary_alpha")
    compare_group("vary_clients",  "vary_clients")
    compare_group("vary_fraction", "vary_fraction")

    # Step 8 — Algorithmes
    compare_group("vary_algo",     "vary_algo")


if __name__ == "__main__":
    main()
