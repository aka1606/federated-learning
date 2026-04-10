"""
run_tp2_part1.py — TP2 Partie 1 : FedAvg avec différents niveaux d'hétérogénéité

But : étudier l'impact du paramètre alpha de Dirichlet sur la convergence de FedAvg.
On fait varier alpha ∈ {100, 1, 0.01} :
  - alpha=100  → distribution quasi-IID (facile)
  - alpha=1    → distribution modérément non-IID
  - alpha=0.01 → distribution très non-IID (difficile)

Ce script réutilise le code de TP1 (quickstart-pytorch) :
  - Il modifie temporairement pyproject.toml de TP1
  - Lance `flwr run .` depuis le dossier TP1
  - Restaure pyproject.toml à la fin

Résultats sauvegardés dans : tp2_workspace/results/vary_alpha/

Usage :
    python tp2_workspace/scripts/run_tp2_part1.py
"""

import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

# ── Chemins absolus ────────────────────────────────────────────────────────────
SCRIPT_DIR  = Path(__file__).parent.resolve()      # tp2_workspace/scripts/
TP2_DIR     = SCRIPT_DIR.parent                    # tp2_workspace/
TP1_DIR     = TP2_DIR.parent / "tp1_workspace" / "quickstart-pytorch"  # Le projet Flower à lancer
TOML        = TP1_DIR / "pyproject.toml"           # Fichier de config à modifier
RESULTS_DIR = TP2_DIR / "results" / "vary_alpha"   # Dossier de sortie des résultats TP2
PYTHON      = sys.executable                       # Chemin Python courant (ex: .venv/bin/python3)
FLWR        = Path(PYTHON).parent / "flwr"         # Chemin de la commande flwr

# ── Hyperparamètres fixes pour toutes les expériences TP2 ──────────────────────
# Ces valeurs sont communes à toutes les expériences de ce TP.
# Seul alpha-dirichlet varie d'une expérience à l'autre.
TP2_PARAMS = {
    "seed":              42,     # Graine aléatoire (reproductibilité)
    "num-clients":       50,     # 50 clients FL
    "num-server-rounds": 50,     # 50 rounds de communication (plus que TP1 pour mieux observer)
    "fraction-train":    1.0,    # Tous les clients participent à chaque round
    "local-epochs":      3,      # 3 époques locales par round
    "learning-rate":     0.01,   # LR pour SGD
    "batch-size":        64,     # Plus grand batch que TP1 (64 au lieu de 32)
    "client-optimizer":  "sgd",  # Optimiseur SGD
    "client-algorithm":  "fedavg",  # Algorithme FedAvg de base
}

# ── 3 expériences : une par valeur d'alpha ─────────────────────────────────────
# Chaque tuple : (label, alpha)
# label = nom du fichier de résultats
# alpha = paramètre Dirichlet à injecter dans pyproject.toml
EXPERIMENTS = [
    ("alpha=100",  100.0),   # Cas quasi-IID : données bien distribuées entre clients
    ("alpha=1",    1.0),     # Cas intermédiaire
    ("alpha=0.01", 0.01),    # Cas très non-IID : chaque client a presque une seule classe
]


# ── Fonctions utilitaires ──────────────────────────────────────────────────────

def patch_toml(overrides: dict) -> None:
    """
    Modifie pyproject.toml de TP1 avec les nouvelles valeurs.

    Pour chaque clé dans overrides, on cherche la ligne correspondante
    dans pyproject.toml et on la remplace par la nouvelle valeur.
    Utilise des regex (re.sub) pour le remplacement.
    """
    text = TOML.read_text()
    for key, val in overrides.items():
        # Formater selon le type : string avec guillemets, numérique sans
        repl = f'{key} = "{val}"' if isinstance(val, str) else f"{key} = {val}"
        # Remplacer la ligne "key = ancienne_valeur" par "key = nouvelle_valeur"
        text = re.sub(rf"^{key}\s*=.*$", repl, text, flags=re.MULTILINE)
    TOML.write_text(text)


def generate_data(alpha: float) -> None:
    """
    Génère les données distribuées pour les 50 clients avec le paramètre alpha donné.

    Lance generate_distributed_datasets() dans un sous-processus Python
    avec le bon PYTHONPATH pour accéder aux modules de TP1.

    On doit régénérer les données à chaque changement d'alpha car alpha
    contrôle la distribution des classes entre les clients.
    """
    print(f"  → Generating data  (alpha={alpha})...")
    subprocess.run(
        [PYTHON, "-c", f"""
import sys
sys.path.insert(0, r'{TP1_DIR / "src"}')
from pytorchexample.data import generate_distributed_datasets
generate_distributed_datasets(k=50, alpha={alpha}, save_dir=r'{TP1_DIR / "data"}')
"""],
        check=True,
    )


def run_flwr() -> Path:
    """
    Lance `flwr run .` depuis le dossier TP1 et retourne le fichier JSON de résultats.

    Étapes :
      1. Installer le package TP1 en mode éditable (pour que flwr trouve pytorchexample)
      2. Tuer les processus flwr/ray résiduels
      3. Lancer la simulation
      4. Trouver le nouveau fichier JSON de résultats

    Retourne le chemin du fichier JSON créé par cette simulation.
    """
    # Installation du package TP1 (nécessaire pour que flwr importe pytorchexample)
    subprocess.run([PYTHON, "-m", "pip", "install", "-e", str(TP1_DIR), "-q"], check=True)

    # Nettoyer les processus résiduels pour éviter les conflits de ports
    os.system("pkill -f flwr 2>/dev/null; pkill -f ray 2>/dev/null")
    time.sleep(2)

    raw_logs = TP1_DIR / "results" / "raw_logs"
    raw_logs.mkdir(parents=True, exist_ok=True)
    results_root = TP1_DIR / "results"

    # Mémoriser les fichiers JSON existants avant la simulation
    before = set(results_root.glob("*.json")) | set(raw_logs.glob("*.json"))

    print("  → Running flwr run .  (patience)...")
    # Lancer la simulation Flower depuis le dossier TP1
    subprocess.run([str(FLWR), "run", ".", "--stream"], check=True, cwd=str(TP1_DIR))

    # Trouver le nouveau fichier JSON créé par cette simulation
    after = set(results_root.glob("*.json")) | set(raw_logs.glob("*.json"))
    new = after - before
    if not new:
        raise RuntimeError("No JSON result found after simulation.")

    # Prendre le fichier le plus récent si plusieurs nouveaux fichiers
    result = max(new, key=lambda p: p.stat().st_mtime)

    # Déplacer dans raw_logs si le fichier atterrit à la racine de results/
    if result.parent == results_root:
        dest = raw_logs / result.name
        shutil.move(str(result), str(dest))
        result = dest
    return result


def save(src: Path, label: str) -> Path:
    """
    Copie le fichier de résultats brut vers le dossier de résultats TP2
    avec un nom lisible (ex: "alpha=0.01.json").
    """
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    dest = RESULTS_DIR / f"{label}.json"
    shutil.copy(str(src), str(dest))
    print(f"  ✅ Saved: {dest}")
    return dest


# ── Point d'entrée ─────────────────────────────────────────────────────────────

def main() -> None:
    # Vérifier que TP1 existe
    if not TP1_DIR.exists():
        sys.exit(f"ERROR: TP1 not found at {TP1_DIR}")

    # Configurer Flower pour utiliser 50 supernodes locaux (= 50 clients simulés)
    flwr_cfg = Path.home() / ".flwr" / "config.toml"
    flwr_cfg.parent.mkdir(exist_ok=True)
    flwr_cfg.write_text(
        '[superlink]\ndefault = "local"\n\n[superlink.local]\noptions.num-supernodes = 50\n'
    )

    # Sauvegarder le contenu original de pyproject.toml pour le restaurer après
    original_toml = TOML.read_text()

    print("\n" + "=" * 60)
    print("  TP2 Part 1 — FedAvg  alpha ∈ {100, 1, 0.01}")
    print("=" * 60)

    try:
        for label, alpha in EXPERIMENTS:
            dest = RESULTS_DIR / f"{label}.json"

            # Skip si déjà fait (permet de reprendre après une interruption)
            if dest.exists():
                print(f"\n  ⏭  {label} — already done, skipping.")
                continue

            print(f"\n--- {label} ---")
            # Configurer TP1 avec les paramètres TP2 + l'alpha courant
            patch_toml({**TP2_PARAMS, "alpha-dirichlet": alpha})
            # Générer les données avec ce nouvel alpha
            generate_data(alpha)
            # Lancer la simulation et récupérer les résultats
            result = run_flwr()
            # Sauvegarder avec un nom explicite
            save(result, label)

    finally:
        # TOUJOURS restaurer pyproject.toml de TP1 même si une erreur survient
        TOML.write_text(original_toml)
        print("\n  TP1 pyproject.toml restored.")

    print("\n" + "=" * 60)
    print("  Done! Results in:", RESULTS_DIR)
    print("  Next: python scripts/analysis_tp2.py")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
