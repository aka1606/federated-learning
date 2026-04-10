# run_experiments.py
# Lance automatiquement toutes les expériences du TP (Steps 7 et 8).
# Modifie pyproject.toml, régénère les données si besoin,
# lance flwr run ./, puis sauvegarde les résultats avec un nom explicite.
#
# Usage :
#   python3 run_experiments.py

import os       # Pour manipuler les chemins, variables d'environnement, et tuer des processus
import shutil   # Pour copier/déplacer des fichiers de résultats
import subprocess  # Pour lancer des commandes shell (ex: flwr run, generate_data.py)
import sys      # Pour récupérer le chemin de l'interpréteur Python courant
import time     # Pour mettre le script en pause (laisser les processus se terminer)
from pathlib import Path  # Pour manipuler les chemins de fichiers de façon portable (Linux/Mac/Windows)

# sys.executable = chemin absolu vers le Python qui exécute ce script
# Ex: /usr/bin/python3 ou /home/user/.venv/bin/python3
PYTHON = sys.executable

# ─────────────────────────────────────────────────────────────────────────────
# Utilitaires
# ─────────────────────────────────────────────────────────────────────────────

def update_toml(overrides: dict) -> None:
    """
    Met à jour les paramètres dans pyproject.toml.

    pyproject.toml est le fichier de configuration central du projet Flower.
    La section [tool.flwr.app.config] contient tous les hyperparamètres
    (learning rate, nombre de clients, nombre de rounds, etc.).

    Cette fonction reçoit un dictionnaire d'overrides (ex: {"learning-rate": 0.001})
    et remplace les lignes correspondantes dans le fichier TOML.
    Cela évite de modifier le fichier à la main entre chaque expérience.
    """
    # Chemin vers pyproject.toml : on remonte de 2 niveaux depuis ce script
    # scripts/ -> quickstart-pytorch/ -> pyproject.toml
    toml_path = Path(__file__).parent.parent / "pyproject.toml"

    # Lire tout le contenu du fichier comme une chaîne de texte
    text = toml_path.read_text()

    for key, value in overrides.items():
        import re  # Module d'expressions régulières pour chercher/remplacer du texte

        # Formater la valeur selon son type :
        # - str  → avec guillemets  ex: client-optimizer = "adam"
        # - int/float → sans guillemets  ex: learning-rate = 0.001
        if isinstance(value, str):
            replacement = f'{key} = "{value}"'
        else:
            replacement = f"{key} = {value}"

        # re.sub remplace la ligne existante par la nouvelle valeur
        # ^ et $ = début/fin de ligne, re.MULTILINE pour chercher dans tout le fichier
        text = re.sub(rf'^{key}\s*=.*$', replacement, text, flags=re.MULTILINE)

    # Écrire le fichier modifié (écrase l'ancien)
    toml_path.write_text(text)


def generate_data() -> None:
    """
    Régénère les datasets distribués entre les clients.

    Dans le federated learning, chaque client possède ses propres données locales.
    La distribution des données entre clients est contrôlée par alpha-dirichlet :
    - alpha petit (ex: 0.1) → distribution très hétérogène (non-IID) : chaque client
      a surtout des exemples d'une ou deux classes → cas réaliste mais difficile
    - alpha grand (ex: 100) → distribution quasi-uniforme (proche de IID) : chaque
      client a des données représentatives de toutes les classes

    Cette fonction appelle scripts/generate_data.py pour recréer les splits de données.
    On doit régénérer les données si on change num-clients ou alpha-dirichlet.
    """
    print("  → Génération des données...")

    # Chemin vers le script de génération de données
    script_path = Path(__file__).parent / "generate_data.py"

    # Copier les variables d'environnement actuelles et ajouter PYTHONPATH
    # PYTHONPATH permet à generate_data.py d'importer les modules dans src/
    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path(__file__).parent.parent / "src")

    # Lancer generate_data.py comme un sous-processus
    # check=True : lève une exception si le script plante (code retour != 0)
    subprocess.run([PYTHON, str(script_path)], check=True, env=env)


def run_simulation() -> Path:
    """
    Lance une simulation Flower et retourne le chemin du fichier JSON de résultats.

    Flower orchestre la simulation fédérée :
    - Un ServerApp agrège les mises à jour des clients (FedAvg, FedSGD, etc.)
    - Plusieurs ClientApp s'entraînent localement sur leurs données
    - À chaque round, les clients envoient leurs poids au serveur qui les agrège

    La commande `flwr run ./` lit la configuration depuis pyproject.toml
    et lance la simulation avec tous les paramètres définis.
    """
    # --- NETTOYAGE DES PROCESSUS PRÉCÉDENTS ---
    # Si une simulation précédente a planté, des processus flwr/ray peuvent
    # encore tourner en arrière-plan et bloquer la prochaine simulation
    print("  → Nettoyage des processus Flower/Ray...")
    os.system("pkill -f flwr 2>/dev/null || true")  # Tuer tous les processus flwr
    os.system("pkill -f ray 2>/dev/null || true")   # Tuer tous les processus Ray (moteur de simulation)
    time.sleep(5)  # Attendre 5 secondes que les processus se terminent proprement

    base_dir = Path(__file__).parent.parent  # Racine du projet
    results_dir = base_dir / "results"       # Dossier où Flower écrit les résultats
    raw_logs_dir = results_dir / "raw_logs"  # Sous-dossier pour les logs bruts
    raw_logs_dir.mkdir(exist_ok=True)        # Créer le dossier s'il n'existe pas

    # Mémoriser les fichiers JSON existants AVANT la simulation
    # pour détecter quel nouveau fichier a été créé par cette simulation
    results_before = set(results_dir.glob("*.json")) | set(raw_logs_dir.glob("*.json"))

    print("  → Lancement de flwr run ./  (patience...)")

    # Lancer `flwr run . --stream` :
    # - `flwr run .` : exécute le projet Flower dans le répertoire courant
    # - `--stream` : affiche les logs en temps réel dans le terminal
    # - check=True : lève une exception si la commande échoue
    # - cwd : répertoire de travail = racine du projet
    subprocess.run(
        [str(Path(PYTHON).parent / "flwr"), "run", ".", "--stream"],
        check=True,
        cwd=str(base_dir)
    )

    # Trouver le nouveau fichier JSON créé par cette simulation
    results_after = set(results_dir.glob("*.json")) | set(raw_logs_dir.glob("*.json"))
    new_files = results_after - results_before  # Différence = fichiers nouvellement créés

    if not new_files:
        raise RuntimeError("Aucun fichier de résultats trouvé après la simulation.")

    # Si plusieurs fichiers ont été créés, prendre le plus récent (par date de modification)
    new_file = max(new_files, key=lambda p: p.stat().st_mtime)

    # Déplacer le fichier de résultats dans raw_logs/ s'il est à la racine de results/
    if new_file.parent == results_dir:
        dest = raw_logs_dir / new_file.name
        shutil.move(new_file, dest)
        return dest
    return new_file


def save_result(src: Path, dest_name: str) -> None:
    """
    Copie le fichier de résultats brut avec un nom lisible et explicite.

    Ex: le fichier brut "run_2024-01-15_14-32-01.json" devient
        "vary_alpha_alpha=0.1.json" pour savoir exactement quelle expérience c'est.
    """
    base_dir = Path(__file__).parent.parent
    dest = base_dir / "results" / dest_name
    shutil.copy(src, dest)
    print(f"  ✅ Résultats sauvegardés : {dest}")


# ─────────────────────────────────────────────────────────────────────────────
# Définition des expériences
# ─────────────────────────────────────────────────────────────────────────────

# Valeurs par défaut utilisées pour tous les paramètres NON variés dans une expérience.
# Cela garantit que chaque expérience ne fait varier qu'UN SEUL paramètre à la fois
# (principe du contrôle des variables en science expérimentale).
DEFAULTS = {
    "num-clients":       50,   # Nombre total de clients dans la simulation
    "alpha-dirichlet":   0.1,  # Hétérogénéité des données (petit = très non-IID)
    "fraction-train":    1.0,  # Fraction de clients qui participent à chaque round (1.0 = tous)
    "num-server-rounds": 10,   # Nombre de rounds de communication serveur-clients
    "local-epochs":      2,    # Nombre d'époques d'entraînement local par client par round
    "batch-size":        32,   # Taille des mini-batches pour l'entraînement local
    "learning-rate":     0.01, # Taux d'apprentissage de l'optimiseur local
    "seed":              42,   # Graine aléatoire pour la reproductibilité des résultats
    "client-optimizer":  "sgd",    # Optimiseur : "sgd" (classique) ou "adam" (adaptatif)
    "client-algorithm":  "fedavg", # Algorithme fédéré : "fedavg", "fedsgd", ou "fedprox"
}

# ── Step 7 : Varier les hyperparamètres ──────────────────────────────────────
# Chaque liste contient des tuples : (label, overrides, regenerate_data)
# - label         : nom de l'expérience (utilisé dans le nom du fichier de résultats)
# - overrides     : paramètres à modifier par rapport aux DEFAULTS
# - regenerate_data : True si on doit recréer les splits de données (nécessaire si
#                     on change num-clients ou alpha-dirichlet)

# Expériences Step 7.1 : Impact de l'hétérogénéité des données (alpha de Dirichlet)
# alpha contrôle la distribution des classes entre les clients :
# - alpha=0.1 → très non-IID : clients spécialisés sur quelques classes → convergence difficile
# - alpha=5   → modérément hétérogène
# - alpha=100 → quasi-IID : distribution uniforme → convergence facile
VARY_ALPHA = [
    ("alpha=0.1",  {"alpha-dirichlet": 0.1},   True),  # Cas non-IID extrême
    ("alpha=5",    {"alpha-dirichlet": 5.0},   True),  # Cas intermédiaire
    ("alpha=100",  {"alpha-dirichlet": 100.0}, True),  # Cas quasi-IID
]

# Expériences Step 7.2 : Impact du nombre de clients
# Plus il y a de clients, plus les données sont fragmentées par client,
# mais aussi plus l'agrégation FedAvg est représentative de la population globale.
VARY_CLIENTS = [
    ("clients=5",  {"num-clients": 5},  True),  # Peu de clients → beaucoup de données chacun
    ("clients=20", {"num-clients": 20}, True),  # Cas intermédiaire
    ("clients=50", {"num-clients": 50}, True),  # Beaucoup de clients → peu de données chacun
]

# Expériences Step 7.3 : Impact de la fraction de clients participants (fraction-train)
# En vrai federated learning, on ne peut pas forcer tous les clients à participer
# (appareils hors-ligne, contraintes réseau...). fraction-train simule cela.
# - fraction=0.1 → seulement 10% des clients participent à chaque round → plus rapide mais moins stable
# - fraction=1.0 → tous les clients participent → résultats plus stables mais plus lent
# Pas besoin de régénérer les données car on ne change pas leur distribution.
VARY_FRACTION = [
    ("fraction=0.1", {"fraction-train": 0.1}, False),
    ("fraction=0.5", {"fraction-train": 0.5}, False),
    ("fraction=1.0", {"fraction-train": 1.0}, False),
]

# ── Step 8 : Comparer les algorithmes fédérés ────────────────────────────────
# FedAvg + SGD  : algorithme de base, chaque client fait du SGD local puis le
#                 serveur fait la moyenne des poids → simple et efficace
# FedAvg + Adam : remplace SGD par Adam (taux d'apprentissage adaptatif) → peut
#                 converger plus vite mais sensible au learning rate
# FedSGD        : cas limite où local-epochs=1 et batch=∞ → équivalent à du
#                 gradient descent distribué sur un round complet
#
# Pour FedSGD, il faut plus de rounds pour compenser le fait qu'on fait moins
# de travail par round. Formule : rounds_fedsgd = rounds × epochs × (N/K / batch_size)
# Avec N=60000 exemples, K=50 clients, batch=32 → ≈ 38 itérations/client/round
# → rounds_fedsgd = 10 × 2 × 38 = 760
FEDSGD_ROUNDS = 760

# Expérience Step 8 : FedAvg + Adam avec learning rate réduit à 0.001
# (Adam est sensible au LR, 0.01 peut être trop grand et causer de l'instabilité)
VARY_ALGORITHM = [
    ("fedavg_adam_v2", {
        "client-algorithm":  "fedavg",
        "client-optimizer":  "adam",
        "num-server-rounds": 10,
        "learning-rate":     0.001,  # LR plus petit pour Adam (0.001 au lieu de 0.01)
    }, False),  # Pas besoin de régénérer les données
]


# ─────────────────────────────────────────────────────────────────────────────
# Lancement des expériences
# ─────────────────────────────────────────────────────────────────────────────

def run_group(group_name: str, experiments: list) -> None:
    """
    Lance un groupe d'expériences séquentiellement.

    Pour chaque expérience dans la liste :
    1. Fusionner les DEFAULTS avec les overrides spécifiques à cette expérience
    2. Écrire la config dans pyproject.toml
    3. Régénérer les données si nécessaire
    4. Lancer la simulation Flower
    5. Sauvegarder les résultats avec un nom explicite

    group_name : préfixe pour les fichiers de résultats (ex: "vary_alpha")
    experiments : liste de tuples (label, overrides, regenerate_data)
    """
    print(f"\n{'='*60}\n  GROUPE : {group_name}\n{'='*60}")
    base_dir = Path(__file__).parent.parent
    (base_dir / "results").mkdir(exist_ok=True)  # Créer le dossier results/ si nécessaire

    for label, overrides, regen in experiments:
        print(f"\n--- Expérience : {label} ---")

        # Fusionner : on part des DEFAULTS et on écrase les clés spécifiées dans overrides
        # Ex: DEFAULTS + {"alpha-dirichlet": 5.0} → tous les defaults sauf alpha qui vaut 5.0
        config = {**DEFAULTS, **overrides}

        # Mettre à jour pyproject.toml avec cette configuration
        update_toml(config)

        # Régénérer les données seulement si la distribution a changé
        # (changement de num-clients ou alpha-dirichlet)
        if regen:
            generate_data()

        # Lancer la simulation et récupérer le chemin du fichier de résultats
        result_file = run_simulation()

        # Construire le nom final du fichier : "vary_alpha_alpha=0.1.json"
        dest_name = f"{group_name}_{label}.json"
        save_result(result_file, dest_name)

    print(f"\n✅ Groupe '{group_name}' terminé.\n")


def main() -> None:
    """
    Point d'entrée principal du script.

    Actuellement configuré pour lancer uniquement l'expérience corrective
    Adam avec LR=0.001 (expérience du Step 8).

    Pour lancer d'autres groupes d'expériences (Step 7), il suffit de
    décommenter les appels run_group correspondants.
    """
    # Se placer dans le répertoire racine du projet pour que `flwr run ./` fonctionne
    base_dir = Path(__file__).parent.parent
    os.chdir(base_dir)

    print("\n🚀 Lancement de l'expérience corrective : Adam avec LR=0.001\n")

    # --- CONFIGURATION DE FLOWER ---
    # Flower a besoin d'un fichier ~/.flwr/config.toml pour savoir combien de
    # "supernodes" (nœuds de simulation) utiliser.
    # num-supernodes doit correspondre à num-clients dans pyproject.toml.
    print("  → Configuration de ~/.flwr/config.toml...")
    flwr_dir = Path.home() / ".flwr"
    flwr_dir.mkdir(exist_ok=True)
    config_content = """[superlink]
default = "local"

[superlink.local]
options.num-supernodes = 50
"""
    # Écrire la config Flower dans le dossier home de l'utilisateur
    (flwr_dir / "config.toml").write_text(config_content)

    # Lancer uniquement le groupe "vary_algo" avec l'expérience Adam corrigée
    # Pour activer les autres groupes, décommenter :
    # run_group("vary_alpha",    VARY_ALPHA)
    # run_group("vary_clients",  VARY_CLIENTS)
    # run_group("vary_fraction", VARY_FRACTION)
    run_group("vary_algo", VARY_ALGORITHM)

    print("\n🎉 Expérience terminée !")
    print("Lance maintenant : python3 analysis.py\n")
    # analysis.py lit les fichiers JSON dans results/ et génère les graphiques


if __name__ == "__main__":
    # Ce bloc s'exécute seulement si on lance le script directement
    # (pas si on l'importe comme module dans un autre script)
    main()
