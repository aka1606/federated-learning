"""
run_tp3_defenses.py — TP3 Défenses : FedMedian & Krum contre les attaques

But : Comparer les stratégies de défense byzantines contre FedAvg.

Deux parties d'expériences:

  PART A — Comparaison des stratégies (alpha=10, quasi-IID):
    Pour chaque type d'attaque (data & model) × ratio (0%, 25%, 50%):
      → Lancer FedAvg (résultats de attacks/, réutilisés)
      → Lancer FedMedian (nouvelle)
      → Lancer Krum (nouvelle)

  PART B — Impact de l'hétérogénéité (25% malveillants, data poisoning):
    Pour chaque alpha ∈ {10, 1, 0.1}:
      → Lancer FedAvg, FedMedian, Krum
      → Voir comment ils se comportent en milieu très hétérogène

Stratégies de défense:
  - FedMedian: médiane coordonnée (robuste mais perd info)
  - Krum: sélection simple (très robuste, sélectionne 1 vecteur)

Résultats sauvegardés dans:
  results/defenses/strategy_comparison/  (Part A)
  results/defenses/heterogeneity/        (Part B)

Usage (depuis tp3_workspace/):
    python scripts/run_tp3_defenses.py
"""

import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

TP3_DIR = Path(__file__).parent.parent.resolve()
TOML    = TP3_DIR / "pyproject.toml"
PYTHON  = sys.executable
FLWR    = Path(PYTHON).parent / "flwr"

# Hyperparamètres de base (identiques pour tous)
BASE_PARAMS = {
    "seed":              42,
    "num-clients":       50,
    "num-server-rounds": 50,
    "fraction-train":    1.0,
    "local-epochs":      3,
    "learning-rate":     0.01,
    "batch-size":        64,
    "alpha-dirichlet":   10.0,
}

# ── PART A: Comparaison des stratégies (alpha=10) ─────────────────────────────
# Génère N expériences testant FedMedian et Krum sous différentes attaques
STRATEGY_EXPERIMENTS = []
for attack in ["data", "model"]:
    for ratio in [0.0, 0.25, 0.50]:
        for strategy in ["fedmedian", "krum"]:
            # ← Optimisation : baseline (0% malveillants) pareil pour fed/median/krum
            if ratio == 0.0 and strategy == "krum":
                continue
            label = f"{strategy}_{attack}_ratio={ratio:.2f}"
            STRATEGY_EXPERIMENTS.append((label, {
                "attack-type":    attack if ratio > 0 else "none",
                "malicious-ratio": ratio,
                "strategy":       strategy,
            }))

# ── PART B: Impact de l'hétérogénéité (25% malveillants, data poisoning) ──────
# Teste les 3 stratégies avec différents niveaux d'hétérogénéité (alpha)
HETERO_EXPERIMENTS = []
for alpha in [10.0, 1.0, 0.1]:
    for strategy in ["fedavg", "fedmedian", "krum"]:
        label = f"hetero_alpha={alpha}_{strategy}"
        HETERO_EXPERIMENTS.append((label, {
            "alpha-dirichlet":  alpha,         # ← Fait varier l'hétérogénéité
            "attack-type":      "data",        # ← Fixe: data poisoning
            "malicious-ratio":  0.25,          # ← Fixe: 25% malveillants
            "strategy":         strategy,
        }))


def patch_toml(overrides: dict) -> None:
    """Modifie le TOML avec les overrides donnés."""
    text = TOML.read_text()
    for key, val in overrides.items():
        repl = f'{key} = "{val}"' if isinstance(val, str) else f"{key} = {val}"
        text = re.sub(rf"^{key}\s*=.*$", repl, text, flags=re.MULTILINE)
    TOML.write_text(text)


def generate_data(alpha: float) -> None:
    """
    Génère les données pour 50 clients avec un alpha Dirichlet donné.
    
    Utilisé dans les deux parts pour régénérer les datasets
    avec les différentes valeurs d'alpha.
    """
    subprocess.run(
        [PYTHON, "-c", f"""
import sys; sys.path.insert(0, r'{TP3_DIR / "src"}')
from tp3example.data import generate_distributed_datasets
generate_distributed_datasets(k=50, alpha={alpha}, save_dir=r'{TP3_DIR / "data"}')
"""],
        check=True,
    )


def run_flwr(raw_dir: Path) -> Path:
    """
    Lance la simulation Flower (flwr run .) et retourne le fichier JSON résultant.
    
    Étapes:
      1. Installer le package tp3 en mode éditable (-e)
      2. Tuer les processus Flower/Ray précédents
      3. Attendre la stabilisation (time.sleep)
      4. Lancer flwr run . avec streaming
      5. Récupérer le JSON créé via timestamp
    """
    os.system("pkill -f flwr 2>/dev/null; pkill -f ray 2>/dev/null")
    time.sleep(8)

    raw_dir.mkdir(parents=True, exist_ok=True)
    t_start = time.time()

    print("  → Running flwr run . (patience)...")
    subprocess.run([str(FLWR), "run", ".", "--stream"], check=True, cwd=str(TP3_DIR))

    new = [p for p in raw_dir.glob("*.json") if p.stat().st_mtime >= t_start]
    if not new:
        raise RuntimeError("No JSON result found after simulation.")
    return max(new, key=lambda p: p.stat().st_mtime)


def save(src: Path, dest_dir: Path, label: str) -> None:
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / f"{label}.json"
    shutil.copy(str(src), str(dest))
    print(f"  ✅ Saved: {dest}")


def run_batch(experiments, dest_dir: Path, raw_dir: Path,
              extra_base: dict = None, regen_data: bool = False,
              last_alpha: list = None):
    """
    Exécute un lot d'expériences de manière séquentielle.
    
    Paramètres:
      experiments: liste de (label, overrides) à exécuter
      dest_dir: répertoire de destination pour les résultats
      raw_dir: répertoire temporaire pour les fichiers JSON bruts
      extra_base: paramètres supplémentaires appliqués à toutes les exps
      regen_data: si True, régénère les données quand alpha change
      last_alpha: liste [alpha_courant] pour tracker les changements d'alpha
    
    Processus:
      1. Pour chaque expérience:
        - Vérifier si déjà exécutée (skip)
        - Si alpha change ET regen_data=True: régénérer les données
        - Patcher le TOML (+ extra_base + extra)
        - Lancer flwr run .
        - Sauvegarder dans dest_dir
    """
    original_toml = TOML.read_text()
    try:
        for label, extra in experiments:
            dest = dest_dir / f"{label}.json"
            if dest.exists():
                print(f"\n  ⏭  {label} — already done, skipping.")
                continue

            print(f"\n--- {label} ---")

            alpha = extra.get("alpha-dirichlet", BASE_PARAMS["alpha-dirichlet"])
            if regen_data and (last_alpha is None or last_alpha[0] != alpha):
                generate_data(alpha)
                if last_alpha is not None:
                    last_alpha[0] = alpha

            overrides = {**BASE_PARAMS, **(extra_base or {}), **extra,
                         "output-dir": str(raw_dir)}
            patch_toml(overrides)
            result = run_flwr(raw_dir)
            save(result, dest_dir, label)

    finally:
        TOML.write_text(original_toml)
        print("\n  pyproject.toml restored.")


def setup_flwr():
    """
    Configure Flower pour utiliser 50 supernodes locaux.
    
    Écrit ~/.flwr/config.toml avec:
      - default superlink: local
      - num-supernodes: 50 (simuler 50 clients localement sur une machine)
    """
    flwr_cfg.parent.mkdir(exist_ok=True)
    flwr_cfg.write_text(
        '[superlink]\ndefault = "local"\n\n[superlink.local]\n'
        'options.num-supernodes = 50\n'
    )


def main():
    """
    Fonction principale : orchestre PART A et PART B des défenses.
    
    PART A — Comparaison des stratégies (alpha=10 fixe):
      - Générer les données une fois (alpha=10)
      - Lancer toutes les stratégies (FedMedian, Krum) contre les attaques
      - Sauvegarder dans strategy_comparison/
    
    PART B — Impact de l'hétérogénéité (25% data poison fixe):
      - Pour cada alpha ∈ {10, 1, 0.1}:
        - Régénérer les données (regen_data=True)
        - Lancer FedAvg, FedMedian, Krum
      - Sauvegarder dans heterogeneity/
    """
    strategy_dir = TP3_DIR / "results" / "defenses" / "strategy_comparison"
    hetero_dir   = TP3_DIR / "results" / "defenses" / "heterogeneity"

    print("\n" + "=" * 65)
    print("  TP3 — Part A: Strategy comparison (FedMedian / Krum)")
    print("=" * 65)
    generate_data(10.0)
    run_batch(STRATEGY_EXPERIMENTS, strategy_dir, raw_dir)

    print("\n" + "=" * 65)
    print("  TP3 — Part B: Heterogeneity impact (alpha ∈ {10, 1, 0.1})")
    print("=" * 65)
    last_alpha = [None]
    run_batch(HETERO_EXPERIMENTS, hetero_dir, raw_dir,
              regen_data=True, last_alpha=last_alpha)

    print("\n" + "=" * 65)
    print("  Done!")
    print(f"  Strategy results : {strategy_dir}")
    print(f"  Hetero results   : {hetero_dir}")
    print("  Next: python scripts/analysis_tp3_defenses.py")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()
