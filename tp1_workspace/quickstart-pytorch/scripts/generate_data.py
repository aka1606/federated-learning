# generate_data.py
# Génère les datasets distribués pour la simulation Federated Learning.
#
# Ce script lit les paramètres (nombre de clients K, alpha Dirichlet, data-dir)
# directement depuis pyproject.toml pour être toujours synchronisé avec
# la configuration de l'expérience en cours.
#
# Usage :
#   python3 scripts/generate_data.py   (depuis quickstart-pytorch/)

from pathlib import Path

# Python 3.11+ inclut tomllib nativement.
# Pour Python 3.10 et antérieur, on utilise tomli (bibliothèque externe).
try:
    import tomllib  # Built-in Python 3.11+
    def _load_toml(path: str) -> dict:
        # tomllib nécessite l'ouverture en mode binaire ("rb")
        with open(path, "rb") as f:
            return tomllib.load(f)
except ImportError:
    import tomli   # pip install tomli
    def _load_toml(path: str) -> dict:
        # tomli lit le fichier comme du texte
        return tomli.loads(Path(path).read_text())

# Import de la fonction de génération des données depuis le module src/
# (nécessite que PYTHONPATH soit configuré ou que le package soit installé)
from pytorchexample.data import generate_distributed_datasets


def main() -> None:
    # Lire la configuration depuis [tool.flwr.app.config] dans pyproject.toml
    # Cela garantit que generate_data.py utilise exactement les mêmes paramètres
    # que ce qui est configuré pour la simulation Flower.
    cfg = _load_toml("pyproject.toml")["tool"]["flwr"]["app"]["config"]

    # Extraire les paramètres nécessaires à la génération
    num_clients: int  = int(cfg["num-clients"])    # Nombre de clients FL (ex: 50)
    alpha: float      = float(cfg["alpha-dirichlet"])  # Paramètre Dirichlet (ex: 0.1)
    data_dir: str     = str(cfg["data-dir"])        # Dossier où sauvegarder les fichiers

    print(f"Generating datasets for {num_clients} clients")
    print(f"  Dirichlet alpha = {alpha}")
    print(f"  Save directory  = {data_dir}")

    # Appel de la fonction principale de génération
    # Elle crée K fichiers client_0.pt, client_1.pt, ..., client_{K-1}.pt
    # dans data_dir/, chacun contenant les données d'un client.
    generate_distributed_datasets(k=num_clients, alpha=alpha, save_dir=data_dir)


if __name__ == "__main__":
    main()
