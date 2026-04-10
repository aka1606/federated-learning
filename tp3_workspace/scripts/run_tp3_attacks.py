"""
run_tp3_attacks.py — TP3 : FedAvg sous attaques de poisoning

But : étudier l'impact des attaques sur la robustesse de FedAvg.

Types d'attaques :
  - Data Poisoning (empoisonnement des données) :
      Les clients malveillants inversent leurs labels : i → 9-i
      Ex: T-shirt (0) → Boot (9), Trouser (1) → Bag (8)
      Effet : le modèle apprend des associations incorrectes

  - Model Poisoning (empoisonnement du modèle) :
      Les clients malveillants entraînent normalement puis MULTIPLIENT
      leurs poids par -5 avant de les envoyer au serveur
      Effet : les mises à jour sont extrêmement perturbées

Pour chaque type d'attaque, on teste 3 proportions de clients malveillants :
  0% (baseline), 25%, 50%

Total : 5 expériences = 1 baseline + 2 data poison + 2 model poison

Résultats sauvegardés dans : tp3_workspace/results/attacks/

Usage (depuis tp3_workspace/) :
    python scripts/run_tp3_attacks.py
"""

import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

# ── Chemins absolus ────────────────────────────────────────────────────────────
TP3_DIR = Path(__file__).parent.parent.resolve()  # tp3_workspace/
TOML    = TP3_DIR / "pyproject.toml"              # Config Flower TP3
RESULTS = TP3_DIR / "results" / "attacks"         # Dossier de sortie
PYTHON  = sys.executable
FLWR    = Path(PYTHON).parent / "flwr"

# ── Hyperparamètres de base ────────────────────────────────────────────────────
# alpha=10 → distribution modérément hétérogène (quasi-IID)
# On choisit un alpha modéré pour que les attaques soient l'unique variable
BASE_PARAMS = {
    "seed":             42,
    "num-clients":      50,
    "num-server-rounds":50,
    "fraction-train":   1.0,
    "local-epochs":     3,
    "learning-rate":    0.01,
    "batch-size":       64,
    "alpha-dirichlet":  10.0,   # Quasi-IID pour isoler l'effet des attaques
}

# ── 5 expériences d'attaque ────────────────────────────────────────────────────
# Chaque tuple : (label, overrides)
# malicious-ratio : fraction des clients qui sont malveillants
# attack-type     : "none" (baseline), "data" (label flipping), "model" (scale -5)
EXPERIMENTS = [
    ("baseline",         {"malicious-ratio": 0.0,  "attack-type": "none"}),   # Référence sans attaque
    ("data_ratio=0.25",  {"malicious-ratio": 0.25, "attack-type": "data"}),   # 25% data poison
    ("data_ratio=0.50",  {"malicious-ratio": 0.5,  "attack-type": "data"}),   # 50% data poison
    ("model_ratio=0.25", {"malicious-ratio": 0.25, "attack-type": "model"}),  # 25% model poison
    ("model_ratio=0.50", {"malicious-ratio": 0.5,  "attack-type": "model"}),  # 50% model poison
]


def patch_toml(overrides: dict) -> None:
    """
    Modifie dynamiquement les lignes correspondantes dans pyproject.toml.
    
    Processus:
      1. Lire le contenu du TOML
      2. Pour chaque clé=valeur dans overrides:
        - Utiliser regex pour remplacer la ligne correspondante
      3. Réécrire le fichier modifié
    
    Utile pour changer les hyperparamètres entre expériences
    sans éditer manuellement le fichier.
    """
    text = TOML.read_text()
    for key, val in overrides.items():
        repl = f'{key} = "{val}"' if isinstance(val, str) else f"{key} = {val}"
        text = re.sub(rf"^{key}\s*=.*$", repl, text, flags=re.MULTILINE)
    TOML.write_text(text)


def generate_data(alpha: float) -> None:
    """
    Génère les datasets pour 50 clients avec alpha donné.
    Utilise le module de données de TP3 (tp3example.data).
    """
    print(f"  → Generating data (alpha={alpha})...")
    subprocess.run(
        [PYTHON, "-c", f"""
import sys; sys.path.insert(0, r'{TP3_DIR / "src"}')
from tp3example.data import generate_distributed_datasets
generate_distributed_datasets(k=50, alpha={alpha}, save_dir=r'{TP3_DIR / "data"}')
"""],
        check=True,
    )


def run_flwr(label: str) -> Path:
    """
    Lance `flwr run .` depuis TP3_DIR et retourne le chemin du JSON de résultats.

    Utilise une détection par timestamp (t_start) pour trouver le nouveau JSON.
    Plus robuste que la comparaison "avant/après" si des fichiers préexistants
    ont le même nom.
    """
    subprocess.run([PYTHON, "-m", "pip", "install", "-e", str(TP3_DIR), "-q"], check=True)
    os.system("pkill -f flwr 2>/dev/null; pkill -f ray 2>/dev/null")
    time.sleep(8)  # TP3 a besoin de plus de temps pour se stabiliser

    raw_dir = TP3_DIR / "results" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    # Mémoriser le timestamp de début pour détecter les nouveaux fichiers
    t_start = time.time()

    print("  → Running flwr run .  (patience)...")
    subprocess.run([str(FLWR), "run", ".", "--stream"], check=True, cwd=str(TP3_DIR))

    # Trouver les fichiers JSON créés APRÈS t_start
    new = [p for p in raw_dir.glob("*.json") if p.stat().st_mtime >= t_start]
    if not new:
        raise RuntimeError("No JSON result found after simulation.")
    return max(new, key=lambda p: p.stat().st_mtime)


def save(src: Path, label: str) -> None:
    """Copie le fichier de résultats avec un nom explicite."""
    RESULTS.mkdir(parents=True, exist_ok=True)
    dest = RESULTS / f"{label}.json"
    shutil.copy(str(src), str(dest))
    print(f"  ✅ Saved: {dest}")


def main():
    """
    Fonction principale : orchestre les 5 expériences d'attaque.
    
    Étapes:
      1. Configurer Flower pour 50 clients locaux
      2. Générer les datasets (alpha=10 fixe)
      3. Pour chaque expérience (baseline + 4 attaques):
        - Patcher le TOML avec les paramètres d'attaque
        - Lancer la simulation Flower
        - Sauvegarder les résultats
      4. Restaurer le TOML original
      5. Afficher où chercher les résultats
    """
    flwr_cfg = Path.home() / ".flwr" / "config.toml"
    flwr_cfg.parent.mkdir(exist_ok=True)
    flwr_cfg.write_text('[superlink]\ndefault = "local"\n\n[superlink.local]\noptions.num-supernodes = 50\n')

    # Sauvegarder pyproject.toml pour restauration après
    original_toml = TOML.read_text()

    print("\n" + "=" * 60)
    print("  TP3 — FedAvg under poisoning attacks  (alpha=10)")
    print("=" * 60)

    # Les données sont les mêmes pour toutes les expériences (alpha=10 fixe)
    # On génère une seule fois (l'attaque se passe dans client_app.py, pas dans les données)
    generate_data(10.0)

    try:
        for label, extra in EXPERIMENTS:
            dest = RESULTS / f"{label}.json"

            # Skip si déjà fait
            if dest.exists():
                print(f"\n  ⏭  {label} — already done, skipping.")
                continue

            print(f"\n--- {label} ---")
            # Configurer les paramètres + paramètres d'attaque + dossier de sortie
            patch_toml({**BASE_PARAMS, **extra, "output-dir": "results/raw"})
            result = run_flwr(label)
            save(result, label)

    finally:
        TOML.write_text(original_toml)
        print("\n  pyproject.toml restored.")

    print("\n" + "=" * 60)
    print("  Done! Results in:", RESULTS)
    print("  Next: python scripts/analysis_tp3.py")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
