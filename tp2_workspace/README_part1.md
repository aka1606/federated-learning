# TP2 — Partie 1 : Impact de l'hétérogénéité des données sur FedAvg

## Protocole expérimental

Nous avons exécuté 3 simulations FedAvg en faisant varier uniquement le paramètre α de la
distribution de Dirichlet, qui contrôle le degré d'hétérogénéité des données entre clients.

**Hyperparamètres fixes :**
| Paramètre | Valeur |
|---|---|
| Nombre de clients | 50 |
| Rounds fédérés | 50 |
| Fraction de clients par round | 1.0 (tous) |
| Epochs locales | 3 |
| Taille de batch | 64 |
| Learning rate | 0.01 |
| Optimiseur | SGD |
| Algorithme | FedAvg |
| Dataset | FashionMNIST |

**Valeurs d'α testées :**
- **α = 100** : distribution quasi-uniforme → données proches de l'IID
- **α = 1** : hétérogénéité modérée
- **α = 0.01** : distribution très concentrée → chaque client ne voit presque qu'une seule classe

---

## Résultats

### Tableau de synthèse (round 50)

| α | Train Loss | Train Acc | Eval Loss | Eval Acc |
|---|---|---|---|---|
| 100 (faible hétéro.) | 0.2755 | 89.9% | 0.2786 | **90.0%** |
| 1 (hétéro. modérée) | 0.2254 | 91.9% | 0.2870 | 89.3% |
| 0.01 (forte hétéro.) | 0.0253 | **99.3%** | 0.8591 | 66.9% |

### Convergence (round 1 → round 50)

| α | Eval Acc round 1 | Eval Acc round 50 | Meilleure Eval Acc |
|---|---|---|---|
| 100 | 62.3% | 90.0% | 90.1% (round 49) |
| 1 | 52.0% | 89.3% | 90.3% (round 49) |
| 0.01 | **10.0%** | 66.9% | 67.1% (round 28) |

---

## Analyse et observations

### 1. Cas quasi-IID (α = 100)

Avec α = 100, chaque client reçoit une portion représentative de toutes les classes.
L'entraînement est stable : la loss d'évaluation suit de près la loss d'entraînement
(0.2786 vs 0.2755), signe d'une bonne généralisation. Le modèle converge régulièrement
vers ~90% d'accuracy en évaluation.

### 2. Hétérogénéité modérée (α = 1)

Les données sont distribuées de façon non-uniforme mais chaque client conserve une
certaine diversité de classes. Les performances finales sont comparables au cas IID
(89.3% eval), mais la convergence est légèrement moins stable au début (52% au round 1
contre 62%). La train accuracy dépasse légèrement l'eval accuracy, ce qui indique un
début de sur-apprentissage local.

### 3. Forte hétérogénéité (α = 0.01) — phénomène de *client drift*

C'est le cas le plus instructif. Dès le round 1, l'écart entre train accuracy (97.2%)
et eval accuracy (10.0%) est massif : chaque client maîtrise parfaitement ses quelques
classes mais le modèle global ne sait pas encore généraliser.

Au fil des rounds, FedAvg agrège des gradients très divergents (chaque client pousse le
modèle vers ses classes locales). Le modèle global n'arrive jamais à converger
correctement : la meilleure eval accuracy atteinte est seulement 67.1% (round 28),
puis la performance stagne voire régresse légèrement.

Ce phénomène s'appelle le **client drift** : les mises à jour locales tirent le modèle
global dans des directions contradictoires, et la simple moyenne des poids (FedAvg)
ne suffit pas à les réconcilier.

---

## Conclusion

L'hétérogénéité des données est le principal défi du Federated Learning. Plus α est
faible, plus les distributions locales divergent, et plus FedAvg peine à produire un
modèle global performant. Le cas α = 0.01 illustre clairement les limites de FedAvg :

- **Train accuracy élevée** → les clients sur-apprennent leurs données locales
- **Eval accuracy faible** → le modèle global ne généralise pas

Ces observations motivent l'utilisation d'algorithmes plus robustes à l'hétérogénéité,
comme **FedProx** (qui ajoute un terme de régularisation proximal) ou **SCAFFOLD**
(qui corrige le drift par des variables de contrôle), que nous explorerons dans la
suite du TP.
