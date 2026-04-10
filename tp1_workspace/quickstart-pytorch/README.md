# TP1 — Federated Learning avec PyTorch et Flower

## Description

Ce TP implémente et expérimente l'apprentissage fédéré (Federated Learning) sur le dataset **FashionMNIST** en utilisant le framework [Flower](https://flower.ai/) avec PyTorch.

L'objectif est de comprendre comment les hyperparamètres et les algorithmes d'agrégation influencent la convergence d'un modèle entraîné de façon distribuée sur des clients aux données hétérogènes.

---

## Structure du projet

```
quickstart-pytorch/
├── src/pytorchexample/
│   ├── client_app.py     # Handler train/evaluate côté client
│   ├── server_app.py     # Orchestration FL côté serveur (FedAvg, FedSGD)
│   ├── model.py          # CNN FashionMNIST (CustomFashionModel)
│   ├── data.py           # Génération des partitions Dirichlet
│   └── task.py
├── scripts/
│   ├── run_experiments.py  # Automatisation de toutes les expériences
│   ├── generate_data.py    # Génération des datasets distribués
│   └── analysis.py         # Analyse et génération des graphiques
├── results/
│   ├── vary_algo_*.json
│   ├── vary_alpha_*.json
│   ├── vary_clients_*.json
│   ├── vary_fraction_*.json
│   └── figures/            # Graphiques générés
└── pyproject.toml
```

---

## Modèle

**`CustomFashionModel`** — CNN à deux couches de convolution :

```
Conv2d(1→32) + ReLU + MaxPool
Conv2d(32→64) + ReLU + MaxPool
Flatten → Linear(3136→128) → ReLU → Linear(128→10)
```

Supporte trois méthodes d'entraînement :
- `train_epoch()` : une epoch locale complète (FedAvg)
- `train_one_step()` : un seul mini-batch (FedSGD)
- `test_epoch()` : évaluation locale

---

## Algorithmes implémentés

### FedAvg (McMahan et al., 2017)
Chaque client sélectionné entraîne le modèle local pendant `local-epochs` epochs, puis le serveur agrège les poids par moyenne pondérée selon le nombre d'exemples.

### FedSGD
Chaque client effectue un seul pas de gradient sur un mini-batch. Pour comparer équitablement avec FedAvg (10 rounds, 2 epochs, batch=32, 50 clients), le nombre de rounds a été calculé :

```
rounds_FedSGD = 10 × 2 × (60000/50 / 32) ≈ 760 rounds
```

### Agrégation serveur

```python
W_global = sum_k (n_k / N_total) * W_k
```

---

## Installation et lancement

```bash
# Installer les dépendances
pip install -e .

# Générer les données partitionnées
python scripts/generate_data.py

# Lancer une simulation
flwr run ./

# Lancer toutes les expériences automatiquement
python scripts/run_experiments.py

# Générer les graphiques et tableaux de résultats
python scripts/analysis.py
```

---

## Expériences réalisées

### Step 7 — Variation des hyperparamètres

Toutes les expériences utilisent les valeurs par défaut suivantes sauf le paramètre varié :

| Paramètre | Valeur par défaut |
|-----------|------------------|
| num-clients | 50 |
| alpha-dirichlet | 0.1 |
| fraction-train | 1.0 |
| num-server-rounds | 10 |
| local-epochs | 2 |
| batch-size | 32 |
| learning-rate | 0.01 |
| optimizer | SGD |

#### 1. Hétérogénéité des données (alpha Dirichlet)

| Alpha | Acc finale (fed_eval) |
|-------|----------------------|
| 0.1 (très hétérogène) | 79.28% |
| 5 | 82.59% |
| 100 (quasi-IID) | **84.07%** |

**Observation** : Plus alpha est élevé, plus les données sont homogènes entre clients (IID), et meilleures sont les performances. Avec alpha=0.1, chaque client voit une distribution très biaisée, ce qui ralentit la convergence.

#### 2. Nombre de clients

| Clients | Acc finale |
|---------|-----------|
| 5 | **89.12%** |
| 20 | 82.58% |
| 50 | 77.27% |

**Observation** : Avec peu de clients, chaque participant représente une fraction plus large des données totales, ce qui améliore l'agrégation sur 10 rounds. Avec 50 clients, le bruit d'agrégation est plus élevé.

#### 3. Fraction de participation (fraction-train)

| Fraction | Acc finale |
|----------|-----------|
| 0.1 (5 clients/round) | 85.50% |
| 0.5 (25 clients/round) | 75.82% |
| 1.0 (50 clients/round) | 78.42% |

**Observation** : fraction=0.1 donne une variance élevée round-à-round (courbe très irrégulière) car seulement 5 clients participent — les résultats dépendent fortement du tirage aléatoire. fraction=1.0 est plus stable mais légèrement moins performant sur 10 rounds.

---

### Step 8 — Comparaison des algorithmes

| Algorithme | LR | Rounds | Acc finale |
|-----------|----|--------|-----------|
| FedAvg + SGD | 0.01 | 10 | 78.67% |
| FedAvg + Adam | 0.01 | 10 | **10.24%** ← divergence |
| FedAvg + Adam v2 | 0.001 | 10 | **81.05%** |
| FedSGD | 0.01 | 760 | 74.19% |

---

## Problèmes rencontrés et solutions

### Problème 1 — FedAvg + Adam diverge (~10% d'accuracy)

**Symptôme** : Avec Adam et `learning-rate=0.01`, le modèle reste bloqué à ~10% d'accuracy (performance aléatoire sur 10 classes), dès le premier round et tout au long de l'entraînement.

**Cause** : Adam adapte déjà les gradients par paramètre via ses estimations de moment. Un learning rate de 0.01 — correct pour SGD — est beaucoup trop élevé pour Adam en contexte fédéré, où les gradients locaux sont hétérogènes. Résultat : les mises à jour explosent et le modèle diverge complètement.

**Solution** : Réduction du LR à **0.001** (10× plus petit). Une nouvelle expérience `fedavg_adam_v2` a été lancée avec ce paramètre corrigé. Résultat : convergence propre à **81.05%** en 10 rounds, surpassant même FedAvg+SGD.

```
# LR trop grand → divergence
learning-rate = 0.01  + Adam  →  acc ≈ 10%

# LR adapté → convergence
learning-rate = 0.001 + Adam  →  acc ≈ 81%
```

### Problème 2 — FedSGD nécessite beaucoup plus de rounds

**Symptôme** : FedSGD avec 10 rounds ne converge pas du tout (moins de 20% d'accuracy).

**Cause** : FedSGD n'effectue qu'un seul pas de gradient par round, contre plusieurs epochs entières pour FedAvg. Le travail par round est incomparable.

**Solution** : Calcul du nombre de rounds équivalent pour une comparaison juste :

```
rounds_FedSGD = rounds_FedAvg × local_epochs × (N / (K × batch_size))
              = 10 × 2 × (60000 / (50 × 32))
              ≈ 760 rounds
```

FedSGD atteint **74.19%** en 760 rounds, légèrement inférieur à FedAvg SGD (78.67%) en seulement 10 rounds. FedAvg est donc plus efficace en nombre de communications.

---

## Résultats finaux

| Expérience | Meilleure config | Acc finale |
|-----------|-----------------|-----------|
| Vary alpha | alpha=100 | 84.07% |
| Vary clients | clients=5 | 89.12% |
| Vary fraction | fraction=0.1 | 85.50% |
| Vary algo | FedAvg + Adam (LR=0.001) | **81.05%** |

---

## Conclusions

1. **FedAvg + Adam (LR=0.001)** offre la meilleure convergence parmi les algorithmes testés sur 10 rounds.
2. **L'hétérogénéité des données** (faible alpha) dégrade significativement les performances — un défi central du Federated Learning.
3. **Moins de clients** améliore la convergence en peu de rounds, mais réduit la diversité des données agrégées.
4. **FedSGD** converge mais nécessite ~76× plus de rounds que FedAvg pour un résultat inférieur.
5. **Le choix du learning rate est critique** pour Adam en FL : un LR inadapté cause une divergence totale dès le premier round.
