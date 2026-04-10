"""
generate_data.py — Génération des datasets distribués pour TP3

But: Créer des datasets non-IID pour k clients à partir de la config pyproject.toml

Utilisation (depuis tp3_workspace/):
    python scripts/generate_data.py

Output:
    data/client_0.pt, data/client_1.pt, ..., data/client_{k-1}.pt
"""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Charger le fichier TOML (configuration Flower)
try:
    import tomllib  # Python 3.11+
    def _load(p): return tomllib.load(open(p, "rb"))
except ImportError:
    import tomli  # Python < 3.11
    def _load(p): return tomli.loads(Path(p).read_text())

from tp3example.data import generate_distributed_datasets


def main():
    """
    Lit la configuration, puis génère les datasets distribués.
    
    Configuration extraite de pyproject.toml:
      - num-clients: nombre de clients k
      - alpha-dirichlet: paramètre Dirichlet pour hétérogénéité
      - data-dir: répertoire de sortie
    """
    cfg = _load("pyproject.toml")["tool"]["flwr"]["app"]["config"]
    generate_distributed_datasets(
        k=int(cfg["num-clients"]),
        alpha=float(cfg["alpha-dirichlet"]),
        save_dir=str(cfg["data-dir"]),
    )

if __name__ == "__main__":
    main()
