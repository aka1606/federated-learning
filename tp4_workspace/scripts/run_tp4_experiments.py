"""
run_tp4_experiments.py — TP4 : Expériences FedPer

But: Étudier l'impact de la personnalisation (KP) et du partial training.

Deux parties d'expériences:

  PART A — Impact du nombre de couches personnalisées (KP):
    Pour chaque alpha ∈ {0.1, 10}:
      Pour chaque KP ∈ {0, 1, 2, 3}:
        → Lancer FedPer avec KP couches personnalisées
        → KP=0 = FedAvg (baseline, pas de personnalisation)
    
    Objectif: Voir comment la personnalisation affecte la convergence
    en milieu très hétérogène (alpha=0.1) vs quasi-IID (alpha=10)

  PART B — Impact du partial training (fraction d'clients):
    Pour chaque fraction ∈ {1.0, 0.5, 0.2}:
      → Lancer FedPer avec alpha=0.1, KP=1 (meilleur config)
      → Simuler une participation partielle (20%, 50%, 100% des clients)
    
    Objectif: Étudier la robustesse du FedPer quand les clients
    ne participent pas à tous les rounds

Résultats sauvegardés dans:
  results/fedper/kp_variation/       (Part A)
  results/fedper/partial_training/   (Part B)

Usage (depuis tp4_workspace/):
    python scripts/run_tp4_experiments.py
"""

import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

TP4_DIR = Path(__file__).parent.parent.resolve()
TOML    = TP4_DIR / "pyproject.toml"
PYTHON  = sys.executable
FLWR    = Path(PYTHON).parent / "flwr"

BASE_PARAMS = {
    "seed":              42,
    "num-clients":       10,
    "num-server-rounds": 30,
    "fraction-train":    1.0,
    "local-epochs":      3,
    "learning-rate":     0.01,
    "batch-size":        64,
}

# ── Part A: Variation de KP ───────────────────────────────────────────────────
# Étudie l'impact du nombre de couches personnalisées sur deux niveaux d'hétérogénéité
PART_A = []
for alpha in [0.1, 10.0]:  # Très hétérogène vs quasi-IID
    for kp in [0, 1, 2, 3]:  # KP=0 est FedAvg (baseline)
        label = f"kp={kp}_alpha={alpha}"
        PART_A.append((label, {
            "alpha-dirichlet":           alpha,
            "num-personalization-layers": kp,
            "fraction-train":            1.0,  # Tous les clients participent
        }))

# ── Part B: Partial training (fraction d'entraînement) ──────────────────────
# Étudie la robustesse quand les clients ne participent pas à tous les rounds
# Utilise la meilleure config de Part A: alpha=0.1, KP=1
PART_B = []
for fraction in [1.0, 0.5, 0.2]:  # % de clients qui participent
    label = f"fraction={fraction}_alpha=0.1_kp=1"
    PART_B.append((label, {
        "alpha-dirichlet":           0.1,  # Très hétérogène (meilleur pour voir les différences)
        "num-personalization-layers": 1,   # Meilleur KP trouvé en Part A
        "fraction-train":            fraction,
    }))


def patch_toml(overrides: dict) -> None:
    """
    Modifie dynamiquement les paramètres dans pyproject.toml.
    
    Utilisé pour changer les hyperparamètres entre expériences
    sans édition manuelle du fichier.
    """


def generate_data(alpha: float) -> None:
    """
    Génère les datasets CIFAR-10 distribués pour 10 clients avec alpha donné.
    """
    subprocess.run(
        [PYTHON, "-c", f"""
import sys; sys.path.insert(0, r'{TP4_DIR / "src"}')
from tp4example.data import generate_distributed_datasets
generate_distributed_datasets(k=10, alpha={alpha}, save_dir=r'{TP4_DIR / "data"}')
"""],
        check=True,
    )


def run_flwr(raw_dir: Path) -> Path:
    """
    Lance 'flwr run .' et retourne le chemin du JSON résultant.
    
    Utilise timestamp pour détecter les nouveaux fichiers.
    """
    os.system("pkill -f flwr 2>/dev/null; pkill -f ray 2>/dev/null")
    time.sleep(8)

    raw_dir.mkdir(parents=True, exist_ok=True)
    t_start = time.time()

    print("  → Running flwr run . (patience)...")
    subprocess.run([str(FLWR), "run", ".", "--stream"], check=True, cwd=str(TP4_DIR))

    new = [p for p in raw_dir.glob("*.json") if p.stat().st_mtime >= t_start]
    if not new:
        raise RuntimeError("No JSON result found after simulation.")
    return max(new, key=lambda p: p.stat().st_mtime)


def save(src: Path, dest_dir: Path, label: str) -> None:
    """Copie le fichier résultats avec un nom explicite."""
    dest = dest_dir / f"{label}.json"
    shutil.copy(str(src), str(dest))
    print(f"  ✅ Saved: {dest}")


def setup_flwr():
    """Configure Flower pour 10 supernodes locaux."""
    flwr_cfg.parent.mkdir(exist_ok=True)
    flwr_cfg.write_text(
        '[superlink]\ndefault = "local"\n\n[superlink.local]\n'
        'options.num-supernodes = 10\n'
    )


def run_batch(experiments, dest_dir, raw_dir, last_alpha=None):
    """
    Exécute un lot d'expériences.
    
    Paramètres:
      experiments: liste (label, overrides)
      dest_dir: où sauvegarder les résultats
      raw_dir: dossier temporaire pour JSON bruts
      last_alpha: [alpha_courant] pour tracker quand régénérer les données
    """
    try:
        for label, extra in experiments:
            dest = dest_dir / f"{label}.json"
            if dest.exists():
                print(f"\n  ⏭  {label} — already done, skipping.")
                continue

            print(f"\n--- {label} ---")
            alpha = extra.get("alpha-dirichlet", 0.1)
            if last_alpha is None or last_alpha[0] != alpha:
                generate_data(alpha)
                if last_alpha is not None:
                    last_alpha[0] = alpha

            overrides = {**BASE_PARAMS, **extra, "output-dir": str(raw_dir)}
            patch_toml(overrides)
            result = run_flwr(raw_dir)
            save(result, dest_dir, label)
    finally:
        TOML.write_text(original_toml)
        print("\n  pyproject.toml restored.")


def main():
    """
    Fonction principale : orchestre PART A et PART B.
    
    PART A:
      - Générer les données une fois (les deux alphas sur le même ensemble)
      - Lancer toutes les expériences (4×2=8 configurations)
    
    PART B:
      - Lancer les 3 expériences de partial training
    
    À la fin, affiche où chercher les résultats.
    """
    part_a_dir = TP4_DIR / "results" / "fedper" / "kp_variation"
    part_b_dir = TP4_DIR / "results" / "fedper" / "partial_training"

    print("\n" + "=" * 60)
    print("  TP4 — Part A: KP variation (KP ∈ {0,1,2,3})")
    print("=" * 60)
    last_alpha = [None]
    run_batch(PART_A, part_a_dir, raw_dir, last_alpha)

    print("\n" + "=" * 60)
    print("  TP4 — Part B: Partial training (fraction ∈ {1.0, 0.5, 0.2})")
    print("=" * 60)
    run_batch(PART_B, part_b_dir, raw_dir)

    print("\n" + "=" * 60)
    print("  Done!")
    print(f"  Part A: {part_a_dir}")
    print(f"  Part B: {part_b_dir}")
    print("  Next: python scripts/analysis_tp4.py")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
