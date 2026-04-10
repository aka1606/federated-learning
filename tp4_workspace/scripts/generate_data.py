"""
generate_data.py — Génère les datasets distribués CIFAR-10 pour TP4

But: Créer 10 clients avec données hétérogènes (alpha=0.1) pour TP4

Utilisation (depuis tp4_workspace/):
    python scripts/generate_data.py

Output:
    data/client_0.pt, ..., data/client_9.pt
    data/test_indices.pt
    data/cifar10_raw/ (téléchargement CIFAR-10)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from tp4example.data import generate_distributed_datasets

if __name__ == "__main__":
    generate_distributed_datasets(
        k=10,
        alpha=0.1,
        save_dir=Path(__file__).parent.parent / "data",
    )
