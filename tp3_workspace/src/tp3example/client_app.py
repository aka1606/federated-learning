"""tp3example.client_app: ClientApp — FedAvg with data/model poisoning support

Application client Flower (Flwr) :
  - Reçoit le modèle global du serveur
  - Entraîne localement sur ses données
  - Envoie les mises à jour au serveur
  - Supporte les attaques : data poisoning (label flipping) et model poisoning (scaling)

Flow:
  1. train(msg): reçoit le modèle, entraîne, retourne les mises à jour
  2. evaluate(msg): reçoit le modèle, évalue sur validation, retourne les métriques
"""

import torch
import torch.nn as nn

from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp

from tp3example.data import load_client_data
from tp3example.model import CustomFashionModel

app = ClientApp()


@app.train()
def train(msg: Message, context: Context) -> Message:
    """
    Fonction d'entraînement client Flower.
    
    Étapes:
      1. Charger le modèle global depuis le message du serveur
      2. Charger les données locales du client
      3. Déterminer si ce client est malveillant (c.-à-d., doit attaquer)
      4. Entraîner localement (honnêtement ou en attaquant)
      5. Pour model poisoning: multiplier les poids par -5
      6. Retourner le modèle entraîné et les métriques
    
    Configuration (depuis pyproject.toml):
      - learning-rate: taux d'apprentissage
      - local-epochs: nombre d'épodes locales
      - batch-size: taille des batches
      - data-dir: répertoire des données
      - num-clients: nombre total de clients (pour le calcul malicious)
      - malicious-ratio: fraction des clients malveillants
      - attack-type: "none" | "data" | "model"
    """
    # --- Load global model ---
    # Récupérer le modèle global envoyé par le serveur
    model = CustomFashionModel()
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())

    # --- Device selection ---
    # Utiliser GPU si disponible (Apple Silicon: MPS, NVIDIA: CUDA, sinon CPU)
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
    model.to(device)

    # --- Load configuration ---
    # Récupérer les paramètres depuis la config (pyproject.toml modifiée)
    cfg              = context.run_config
    lr               = float(cfg["learning-rate"])
    local_epochs     = int(cfg["local-epochs"])
    batch_size       = int(cfg["batch-size"])
    data_dir         = str(cfg["data-dir"])
    num_clients      = int(cfg.get("num-clients", 50))
    malicious_ratio  = float(cfg.get("malicious-ratio", 0.0))
    attack_type      = str(cfg.get("attack-type", "none"))

    # --- Determine if this client is malicious ---
    # Les k premiers clients (selon malicious_ratio) sont malveillants
    partition_id  = int(context.node_config["partition-id"])
    num_malicious = int(num_clients * malicious_ratio)
    is_malicious  = (partition_id < num_malicious) and (attack_type != "none")

    # --- Load data ---
    # Charger les données locales du client
    train_loader, _ = load_client_data(cid=partition_id, data_dir=data_dir, batch_size=batch_size)

    # --- Setup optimizer & loss ---
    # Définir la fonction de loss et l'optimiseur
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)

    # --- Training phase ---
    # Entraîner localement (honnêtement ou en attaquant)
    total_loss, total_acc = 0.0, 0.0

    if is_malicious and attack_type == "data":
        # ATTAQUE: Data poisoning (label flipping)
        # Les labels sont inversés : i → 9-i
        for _ in range(local_epochs):
            loss, acc = model.train_epoch_data_poison(train_loader, criterion, optimizer, device)
            total_loss += loss
            total_acc  += acc
    else:
        # Entraînement HONNÊTE (utilisé aussi comme base pour model poisoning)
        for _ in range(local_epochs):
            loss, acc = model.train_epoch(train_loader, criterion, optimizer, device)
            total_loss += loss
            total_acc  += acc

    train_loss = total_loss / local_epochs
    train_acc  = total_acc  / local_epochs

    # --- Model poisoning attack ---
    # ATTAQUE: Model poisoning - multiplier tous les poids par -5
    # Cette perturbation extrême désorganise l'agrégation FedAvg
    if is_malicious and attack_type == "model":
        with torch.no_grad():
            for param in model.parameters():
                param.data.mul_(-5.0)

    num_examples = len(train_loader.dataset)

    # --- Send updates and metrics to server ---
    # Construire le message de réponse avec le modèle entraîné et les métriques
    content = RecordDict({
        "arrays":  ArrayRecord(model.state_dict()),  # Modèle entraîné localement
        "metrics": MetricRecord({
            "train_loss":   float(train_loss),
            "train_acc":    float(train_acc),
            "num_examples": int(num_examples),
        }),
    })
    return Message(content=content, reply_to=msg)


@app.evaluate()
def evaluate(msg: Message, context: Context) -> Message:
    """
    Fonction d'évaluation client Flower.
    
    Étapes:
      1. Charger le modèle global reçu du serveur
      2. Évaluer sur le dataset de validation local
      3. Retourner les métriques (loss, accuracy)
    
    Remarque: L'évaluation n'est PAS empoisonnée (même si le client est malveillant).
              On teste la qualité réelle du modèle.
    """
    # Charger le modèle global du message serveur
    model = CustomFashionModel()
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())

    # Sélectionner le device (GPU si disponible)
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
    model.to(device)

    # Récupérer la configuration
    cfg        = context.run_config
    batch_size = int(cfg["batch-size"])
    data_dir   = str(cfg["data-dir"])
    partition_id = int(context.node_config["partition-id"])

    # Charger les données de validation du client (80/20 split effectué dans load_client_data)
    _, val_loader = load_client_data(cid=partition_id, data_dir=data_dir, batch_size=batch_size)

    # Évaluer sur les données de validation
    criterion = nn.CrossEntropyLoss().to(device)
    eval_loss, eval_acc = model.test_epoch(val_loader, criterion, device)

    # Retourner les métriques au serveur
    content = RecordDict({
        "metrics": MetricRecord({
            "eval_loss":    float(eval_loss),
            "eval_acc":     float(eval_acc),
            "num_examples": int(len(val_loader.dataset)),
        }),
    })
    return Message(content=content, reply_to=msg)
