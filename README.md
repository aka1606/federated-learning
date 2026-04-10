# Federated Learning — Travaux Pratiques

Série de 4 TPs progressifs sur l'apprentissage fédéré (Federated Learning), réalisés avec **PyTorch** et le framework [Flower](https://flower.ai/).

Dataset commun : **FashionMNIST** (70 000 images, 10 classes).  
Distribution des données : partitionnement **Dirichlet(α)** pour simuler l'hétérogénéité entre clients.

---

## Structure du projet

```
Federated_learning/
├── tp1_workspace/       # Introduction — FedAvg, FedSGD, hyperparamètres
├── tp2_workspace/       # Algorithmes avancés — FedProx, SCAFFOLD
├── tp3_workspace/       # Sécurité — attaques et défenses
└── tp4_workspace/       # Personnalisation — FedPer
```

---

## TP1 — Introduction au Federated Learning

**Objectif** : comprendre l'impact des hyperparamètres et des algorithmes d'agrégation sur la convergence d'un modèle fédéré.

**Modèle** : CNN à deux couches de convolution (`CustomFashionModel`).

**Expériences** :
| Paramètre varié | Valeurs testées |
|---|---|
| Hétérogénéité (α Dirichlet) | 0.1, 5, 100 |
| Nombre de clients | 5, 20, 50 |
| Fraction de participation | 0.1, 0.5, 1.0 |
| Algorithme | FedAvg+SGD, FedAvg+Adam, FedSGD |

**Résultats clés** :
- α élevé (quasi-IID) → meilleure convergence (84.07% avec α=100)
- FedAvg + Adam (LR=0.001) > FedAvg + SGD en 10 rounds (81.05% vs 78.67%)
- FedSGD nécessite ~76× plus de rounds que FedAvg pour un résultat inférieur

```bash
cd tp1_workspace/quickstart-pytorch
pip install -e .
python scripts/generate_data.py
flwr run ./                        # simulation unique
python scripts/run_experiments.py  # toutes les expériences
python scripts/analysis.py         # graphiques
```

---

## TP2 — Algorithmes robustes à l'hétérogénéité

**Objectif** : comparer FedAvg, **FedProx** et **SCAFFOLD** face au *client drift* causé par des données hétérogènes.

**Protocole** : 50 clients, 50 rounds, α ∈ {0.01, 1, 100}.

**Résultats clés** (eval accuracy à α=0.01) :
| Algorithme | Acc finale |
|---|---|
| FedAvg | 66.9% |
| FedProx (μ=0.1) | ~70% |
| SCAFFOLD | ~75% |

- Avec α=0.01 (forte hétérogénéité), FedAvg plafonne à ~67% — phénomène de *client drift*
- SCAFFOLD corrige le drift via des variables de contrôle et surpasse FedProx

```bash
cd tp2_workspace
pip install -e .
python scripts/run_tp2_part1.py   # FedAvg baseline
python scripts/run_tp2_part2.py   # FedProx
python scripts/run_tp2_part3.py   # SCAFFOLD
python scripts/analysis_tp2_final.py
```

---

## TP3 — Sécurité du Federated Learning

**Objectif** : étudier les attaques de *poisoning* et l'efficacité des stratégies d'agrégation robustes.

**Attaques implémentées** :
| Type | Mécanisme |
|---|---|
| Data poisoning | Inversion des labels : classe `i` → `9-i` |
| Model poisoning | Multiplication des poids par `−5` avant envoi |

**Défenses** :
| Stratégie | Principe |
|---|---|
| FedAvg | Moyenne pondérée (aucune robustesse) |
| FedMedian | Médiane coordonnée par coordonnée |
| Krum | Sélection du client le moins divergent |

**Scénarios** : 0%, 25%, 50% de clients malveillants × 3 stratégies.

```bash
cd tp3_workspace
pip install -e .
python scripts/generate_data.py
python scripts/run_tp3_attacks.py    # expériences d'attaques
python scripts/run_tp3_defenses.py   # expériences de défenses
python scripts/analysis_tp3.py
python scripts/analysis_tp3_defenses.py
```

---

## TP4 — Personnalisation : FedPer

**Objectif** : implémenter **FedPer** (Federated Personalization) — partager uniquement les premières couches et garder les dernières personnalisées par client.

**Paramètre clé** : `KP` = nombre de couches personnalisées (non partagées).
- `KP=0` → FedAvg standard (baseline)
- `KP=1` → dernière couche FC personnalisée
- `KP=2` → 2 dernières couches FC personnalisées
- `KP=3` → toutes les FC personnalisées

**Expériences** :
- Part A : impact de KP pour α=0.1 et α=10
- Part B : impact de la participation partielle (20%, 50%, 100%) avec KP=1, α=0.1

```bash
cd tp4_workspace
pip install -e .
python scripts/generate_data.py
python scripts/run_tp4_experiments.py
python scripts/analysis_tp4.py
```

---

## Prérequis

- Python ≥ 3.10
- PyTorch ≥ 2.0
- `flwr[simulation]` ≥ 1.26.0

Chaque workspace est autonome avec son propre `pyproject.toml`. Il est recommandé d'utiliser un environnement virtuel par TP :

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

---

## Ce qui n'est pas versionné

Les éléments suivants sont exclus du dépôt (voir [.gitignore](.gitignore)) :
- Données brutes : `data/`, `FashionMNIST/`, fichiers `.pt`, `.gz`
- Résultats générés : `results/`, fichiers `.json`, `.png`
- Environnements virtuels : `.venv/`, `venv/`

Pour reproduire les expériences, lancer d'abord `generate_data.py` puis le script `run_*.py` correspondant.
