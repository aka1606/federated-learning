"""
run_tp2_part2.py — TP2 Partie 2 : FedProx avec différents niveaux d'hétérogénéité

But : comparer FedProx vs FedAvg (résultats de Part 1) sur les mêmes 3 valeurs d'alpha.
FedProx ajoute un terme proximal à la loss locale :
  Loss = CrossEntropy + (mu/2) * ||w_local - w_global||²

Ce terme empêche les clients de trop s'éloigner du modèle global,
ce qui est particulièrement utile en milieu non-IID (alpha petit).

Résultats sauvegardés dans : tp2_workspace/results/fedprox/

Usage :
    python tp2_workspace/scripts/run_tp2_part2.py
"""

import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

# ── Chemins absolus ────────────────────────────────────────────────────────────
SCRIPT_DIR  = Path(__file__).parent.resolve()
TP2_DIR     = SCRIPT_DIR.parent
TP1_DIR     = TP2_DIR.parent / "tp1_workspace" / "quickstart-pytorch"
TOML        = TP1_DIR / "pyproject.toml"
RESULTS_DIR = TP2_DIR / "results" / "fedprox"   # Dossier séparé de FedAvg
PYTHON      = sys.executable
FLWR        = Path(PYTHON).parent / "flwr"

# ── Hyperparamètres fixes ── identiques à Part 1 pour une comparaison équitable ──
TP2_PARAMS = {
    "seed":              42,
    "num-clients":       50,
    "num-server-rounds": 50,
    "fraction-train":    1.0,
    "local-epochs":      3,
    "learning-rate":     0.01,
    "batch-size":        64,
    "client-optimizer":  "sgd",
    "client-algorithm":  "fedprox",  # ← DIFFÉRENCE avec Part 1 : FedProx au lieu de FedAvg
    "fedprox-mu":        0.1,         # Coefficient mu du terme proximal
    # mu=0.1 est une valeur standard (pas trop fort pour ne pas trop contraindre les clients)
}

# ── 3 expériences ─ mêmes alpha qu'en Part 1 pour comparaison directe ─────────
EXPERIMENTS = [
    ("alpha=100",  100.0),
    ("alpha=1",    1.0),
    ("alpha=0.01", 0.01),
]


# ── Fonctions utilitaires ── (identiques à Part 1) ────────────────────────────

def patch_toml(overrides: dict) -> None:
    """Modifie les lignes correspondantes dans pyproject.toml de TP1."""
    text = TOML.read_text()
    for key, val in overrides.items():
        repl = f'{key} = "{val}"' if isinstance(val, str) else f"{key} = {val}"
        text = re.sub(rf"^{key}\s*=.*$", repl, text, flags=re.MULTILINE)
    TOML.write_text(text)


def generate_data(alpha: float) -> None:
    """Génère les datasets distribués pour 50 clients avec le paramètre alpha donné."""
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
    """Lance `flwr run .` depuis TP1 et retourne le chemin du JSON de résultats."""
    # Installation du package pour que flwr trouve pytorchexample
    subprocess.run([PYTHON, "-m", "pip", "install", "-e", str(TP1_DIR), "-q"], check=True)

    # Nettoyer les processus flwr/ray résiduels
    os.system("pkill -f flwr 2>/dev/null; pkill -f ray 2>/dev/null")
    time.sleep(2)

    raw_logs = TP1_DIR / "results" / "raw_logs"
    raw_logs.mkdir(parents=True, exist_ok=True)
    results_root = TP1_DIR / "results"

    before = set(results_root.glob("*.json")) | set(raw_logs.glob("*.json"))

    print("  → Running flwr run .  (patience)...")
    subprocess.run([str(FLWR), "run", ".", "--stream"], check=True, cwd=str(TP1_DIR))

    after = set(results_root.glob("*.json")) | set(raw_logs.glob("*.json"))
    new = after - before
    if not new:
        raise RuntimeError("No JSON result found after simulation.")

    result = max(new, key=lambda p: p.stat().st_mtime)
    if result.parent == results_root:
        dest = raw_logs / result.name
        shutil.move(str(result), str(dest))
        result = dest
    return result


def save(src: Path, label: str) -> Path:
    """Copie le fichier de résultats avec un nom lisible."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    dest = RESULTS_DIR / f"{label}.json"
    shutil.copy(str(src), str(dest))
    print(f"  ✅ Saved: {dest}")
    return dest


# ── Point d'entrée ─────────────────────────────────────────────────────────────

def main() -> None:
    if not TP1_DIR.exists():
        sys.exit(f"ERROR: TP1 not found at {TP1_DIR}")

    # Configurer Flower
    flwr_cfg = Path.home() / ".flwr" / "config.toml"
    flwr_cfg.parent.mkdir(exist_ok=True)
    flwr_cfg.write_text(
        '[superlink]\ndefault = "local"\n\n[superlink.local]\noptions.num-supernodes = 50\n'
    )

    # Sauvegarder pyproject.toml original pour restauration
    original_toml = TOML.read_text()

    print("\n" + "=" * 60)
    print("  TP2 Part 2 — FedProx  alpha ∈ {100, 1, 0.01}  mu=0.1")
    print("=" * 60)

    try:
        for label, alpha in EXPERIMENTS:
            dest = RESULTS_DIR / f"{label}.json"
            if dest.exists():
                print(f"\n  ⏭  {label} — already done, skipping.")
                continue

            print(f"\n--- {label} ---")
            patch_toml({**TP2_PARAMS, "alpha-dirichlet": alpha})
            generate_data(alpha)
            result = run_flwr()
            save(result, label)

    finally:
        # Restaurer pyproject.toml dans tous les cas
        TOML.write_text(original_toml)
        print("\n  TP1 pyproject.toml restored.")

    print("\n" + "=" * 60)
    print("  Done! Results in:", RESULTS_DIR)
    print("  Next: python scripts/analysis_tp2_part2.py")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
