"""
client_app.py — Client Flower pour le Federated Learning (TP1).

Ce fichier implémente les deux handlers côté client :
  - train()    : reçoit les poids globaux, entraîne localement, renvoie les poids mis à jour
  - evaluate() : reçoit les poids globaux, évalue sur données locales, renvoie les métriques

Algorithmes supportés via client-algorithm :
  - fedavg   : plusieurs époques locales standard
  - fedsgd   : une seule étape de gradient (équivalent GD distribué)
  - fedprox  : FedAvg + terme de régularisation proximal
  - scaffold : FedAvg + correction de gradient (lutte contre le client drift)
"""

import torch
import torch.nn as nn

from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp

from pytorchexample.data import load_client_data
from pytorchexample.model import CustomFashionModel

# Instanciation de l'application client Flower
app = ClientApp()


@app.train()
def train(msg: Message, context: Context) -> Message:
    """
    Handler d'entraînement côté client.

    Appelé par le serveur à chaque round d'entraînement.

    Flux :
      1. Charger les poids globaux depuis le message du serveur
      2. Lire la configuration (learning rate, époques, algorithme...)
      3. Charger les données locales du client
      4. Entraîner localement (selon l'algorithme choisi)
      5. Renvoyer les poids mis à jour + métriques au serveur

    Métriques renvoyées : train_loss, train_acc, num_examples
    """

    # --- Charger le modèle depuis le message du serveur ---
    # Le serveur envoie les poids globaux encapsulés dans un ArrayRecord
    model = CustomFashionModel()
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())

    # --- Sélection du device (GPU Apple Silicon > GPU CUDA > CPU) ---
    if torch.backends.mps.is_available():
        device = torch.device("mps")   # GPU Apple Silicon (M1/M2/M3)
    elif torch.cuda.is_available():
        device = torch.device("cuda:0")  # GPU NVIDIA
    else:
        device = torch.device("cpu")    # CPU (simulation sur laptop)

    model.to(device)

    # --- Lire la configuration depuis pyproject.toml ---
    # context.run_config est un dict avec tous les paramètres de [tool.flwr.app.config]
    cfg = context.run_config
    lr               = float(cfg["learning-rate"])     # Taux d'apprentissage
    local_epochs     = int(cfg["local-epochs"])        # Nombre d'époques locales
    batch_size       = int(cfg["batch-size"])          # Taille des mini-batches
    data_dir         = str(cfg["data-dir"])            # Dossier des données
    client_optimizer = str(cfg["client-optimizer"])    # "sgd" ou "adam"
    client_algorithm = str(cfg["client-algorithm"])    # "fedavg", "fedsgd", "fedprox", "scaffold"
    fedprox_mu       = float(cfg.get("fedprox-mu", 0.0))  # Coefficient proximal (FedProx uniquement)

    # --- Identifier ce client ---
    # partition-id est attribué par Flower à chaque nœud (0, 1, ..., K-1)
    partition_id = int(context.node_config["partition-id"])

    # --- Charger les données locales du client ---
    # Retourne (train_loader, val_loader) — on n'utilise que train_loader ici
    train_loader, _ = load_client_data(
        cid=partition_id,
        data_dir=data_dir,
        batch_size=batch_size,
    )

    # --- Construire l'optimiseur selon la configuration ---
    criterion = nn.CrossEntropyLoss().to(device)  # Fonction de perte standard pour classification
    if client_optimizer == "adam":
        # Adam : taux d'apprentissage adaptatif, converge souvent plus vite
        # Mais sensible au LR (recommandé : 0.001)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    elif client_algorithm == "scaffold":
        # SCAFFOLD requiert SGD sans momentum pour que la correction de gradient soit correcte
        # Le momentum interfère avec la correction SCAFFOLD
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.0)
    else:
        # SGD standard avec momentum=0.9 (valeur classique)
        # Le momentum accélère la convergence et réduit les oscillations
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)

    # --- Sauvegarder les poids globaux (avant entraînement local) ---
    # Utilisé par FedProx et SCAFFOLD pour calculer les termes de correction
    global_params = [p.detach().clone().cpu() for p in model.parameters()]

    # --- Entraînement selon l'algorithme ---
    if client_algorithm == "fedsgd":
        # FedSGD : une seule étape de gradient sur UN SEUL mini-batch
        # Équivalent à un gradient descent distribué
        # Avantage : convergence théoriquement stable
        # Inconvénient : nécessite beaucoup plus de rounds que FedAvg
        train_loss, train_acc = model.train_one_step(
            train_loader, criterion, optimizer, device
        )

    elif client_algorithm == "fedprox":
        # FedProx : entraînement standard + terme proximal
        # Loss = CrossEntropy + (mu/2) * ||w_local - w_global||²
        # Empêche les clients de trop diverger du modèle global
        total_loss, total_acc = 0.0, 0.0
        for _ in range(local_epochs):
            loss, acc = model.train_epoch_fedprox(
                train_loader, criterion, optimizer, device, global_params, fedprox_mu
            )
            total_loss += loss
            total_acc += acc
        train_loss = total_loss / local_epochs
        train_acc = total_acc / local_epochs

    elif client_algorithm == "scaffold":
        # SCAFFOLD : entraînement avec correction de gradient
        # Corrige le "client drift" causé par la non-IID-ité des données

        # Charger le variateur de contrôle local c_k depuis l'état persistant
        # (context.state est un stockage persistant entre les rounds pour ce client)
        if "scaffold_ck" in context.state.array_records:
            # Rounds suivants : restaurer c_k sauvegardé
            ck_sd = context.state.array_records["scaffold_ck"].to_torch_state_dict()
            n_params = len(list(model.parameters()))
            c_k = [ck_sd[str(i)].cpu() for i in range(n_params)]
        else:
            # Premier round : initialiser c_k à zéros
            c_k = [torch.zeros_like(p.data).cpu() for p in model.parameters()]

        # Charger le variateur de contrôle global c depuis le message du serveur
        c_sd = msg.content["scaffold_c"].to_torch_state_dict()
        n_params = len(list(model.parameters()))
        c_global = [c_sd[str(i)].cpu() for i in range(n_params)]

        # T = nombre total d'étapes locales = époques × batches_par_époque
        T = local_epochs * len(train_loader)

        # Entraînement avec correction SCAFFOLD
        total_loss, total_acc = 0.0, 0.0
        for _ in range(local_epochs):
            loss, acc = model.train_epoch_scaffold(
                train_loader, criterion, optimizer, device, c_k, c_global
            )
            total_loss += loss
            total_acc += acc
        train_loss = total_loss / local_epochs
        train_acc  = total_acc  / local_epochs

        # Mettre à jour c_k :
        # c_k_new = c_k - c_global + (1/(lr*T)) * (w_avant - w_après)
        # Cette formule estime le gradient local moyen sur T étapes
        w_t1 = [p.data.detach().clone().cpu() for p in model.parameters()]
        c_k_new = [
            ck_i - c_i + (1.0 / (lr * T)) * (wt_i - wt1_i)
            for ck_i, c_i, wt_i, wt1_i in zip(c_k, c_global, global_params, w_t1)
        ]

        # delta_c_k = différence entre nouveau et ancien c_k
        # Envoyé au serveur pour qu'il mette à jour c_global
        delta_c_k = {str(i): (c_k_new[i] - c_k[i]) for i in range(len(c_k))}

        # Sauvegarder c_k mis à jour dans l'état persistant du client
        context.state.array_records["scaffold_ck"] = ArrayRecord(
            {str(i): c_k_new[i] for i in range(len(c_k_new))}
        )

    else:
        # FedAvg : entraînement standard sur plusieurs époques locales
        # L'algorithme de base du Federated Learning
        total_loss, total_acc = 0.0, 0.0
        for _ in range(local_epochs):
            loss, acc = model.train_epoch(train_loader, criterion, optimizer, device)
            total_loss += loss
            total_acc += acc
        train_loss = total_loss / local_epochs
        train_acc = total_acc / local_epochs

    num_examples = len(train_loader.dataset)  # Nombre d'exemples d'entraînement

    # --- Construire le message de réponse ---
    # Encapsuler les poids mis à jour dans un ArrayRecord
    model_record  = ArrayRecord(model.state_dict())
    metric_record = MetricRecord({
        "train_loss":   float(train_loss),
        "train_acc":    float(train_acc),
        "num_examples": int(num_examples),  # Nécessaire pour la pondération FedAvg
    })
    content_dict = {"arrays": model_record, "metrics": metric_record}

    if client_algorithm == "scaffold":
        # SCAFFOLD : inclure aussi le delta du variateur de contrôle
        content_dict["scaffold_delta_ck"] = ArrayRecord(delta_c_k)

    content = RecordDict(content_dict)
    return Message(content=content, reply_to=msg)  # reply_to = référence au message du serveur


@app.evaluate()
def evaluate(msg: Message, context: Context) -> Message:
    """
    Handler d'évaluation côté client.

    Appelé par le serveur à chaque round d'évaluation (après l'agrégation).
    Évalue le modèle global agrégé sur les données de VALIDATION locales.

    Métriques renvoyées : eval_loss, eval_acc, num_examples
    """

    # --- Charger le modèle global agrégé depuis le message ---
    model = CustomFashionModel()
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())

    # --- Sélection du device ---
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")

    model.to(device)

    # --- Lire la configuration ---
    cfg = context.run_config
    batch_size = int(cfg["batch-size"])
    data_dir   = str(cfg["data-dir"])

    # --- Identifier ce client ---
    partition_id = int(context.node_config["partition-id"])

    # --- Charger les données de VALIDATION locales ---
    # On n'utilise que val_loader ici (le _ ignore train_loader)
    _, val_loader = load_client_data(
        cid=partition_id,
        data_dir=data_dir,
        batch_size=batch_size,
    )

    # --- Évaluer le modèle ---
    criterion = nn.CrossEntropyLoss().to(device)
    eval_loss, eval_acc = model.test_epoch(val_loader, criterion, device)
    num_examples = len(val_loader.dataset)

    # --- Construire la réponse ---
    metric_record = MetricRecord({
        "eval_loss":    float(eval_loss),
        "eval_acc":     float(eval_acc),
        "num_examples": int(num_examples),
    })
    content = RecordDict({"metrics": metric_record})
    return Message(content=content, reply_to=msg)
