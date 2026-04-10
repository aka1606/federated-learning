"""
tp4example.client_app: ClientApp—FedPer Client avec Couches Personnalisées

Le client FedPer:
  1. Reçoit les couches BASE du serveur
  2. Restaure ses couches PERSONNALISÉES depuis context.state (persistance locale)
  3. Assemble le modèle complet (base + pers)
  4. Entraîne localement (tous les poids)
  5. Sauvegarde les couches pers pour le prochain round
  6. Envoie seulement les couches base au serveur

Cette architecture permet:
  - Les couches base → convergence globale
  - Les couches pers → adaptation locale (FedPer)
"""

import numpy as np
import torch
import torch.nn as nn

from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp

from tp4example.data import load_client_data
from tp4example.model import Net

app = ClientApp()


def _get_device() -> torch.device:
    """Sélectionne le meilleur device disponible."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    return torch.device("cpu")


def _arrays_to_numpy(record: ArrayRecord) -> list[np.ndarray]:
    """Convertit ArrayRecord (Flower) en liste de numpy arrays."""
    sd = record.to_torch_state_dict()
    return [v.cpu().numpy() for v in sd.values()]


def _numpy_to_record(params: list[np.ndarray], keys: list[str]) -> ArrayRecord:
    """Convertit liste numpy + clés en ArrayRecord (Flower)."""
    sd = {k: torch.tensor(v) for k, v in zip(keys, params)}
    return ArrayRecord(sd)


def _base_keys(num_personalization_layers: int) -> list[str]:
    """Retourne les clés du state_dict pour les couches BASE."""
    from tp4example.model import LAYER_NAMES, NUM_LAYERS
    kp = min(num_personalization_layers, NUM_LAYERS)
    base_names = LAYER_NAMES[:NUM_LAYERS - kp]
    return [f"{n}.{p}" for n in base_names for p in ("weight", "bias")]


def _pers_keys(num_personalization_layers: int) -> list[str]:
    """Retourne les clés du state_dict pour les couches PERSONNALISÉES."""
    from tp4example.model import LAYER_NAMES, NUM_LAYERS
    kp = min(num_personalization_layers, NUM_LAYERS)
    pers_names = LAYER_NAMES[NUM_LAYERS - kp:]
    return [f"{n}.{p}" for n in pers_names for p in ("weight", "bias")]


@app.train()
def train(msg: Message, context: Context) -> Message:
    """
    Fonction d'entraînement FedPer client.
    
    Étapes clés:
      1. Recevoir les couches BASE du serveur
      2. Restaurer les couches PERSONNALISÉES depuis context.state
        (première fois: initialiser, après: charger de l'état précédent)
      3. Assembler le modèle complet
      4. Entraîner localement (tous les poids)
      5. Sauvegarder les couches pers dans context.state
      6. Envoyer SEULEMENT les couches base au serveur
    
    Persistance:
      - context.state : stockage local pour les couches pers (survit entre rounds)
      - context.node_config : partition-id, autre config client
      - context.run_config : parameters globaux (lr, epochs, kp, etc.)
    """
    device = _get_device()

    cfg                       = context.run_config
    lr                        = float(cfg["learning-rate"])
    local_epochs              = int(cfg["local-epochs"])
    batch_size                = int(cfg["batch-size"])
    data_dir                  = str(cfg["data-dir"])
    kp                        = int(cfg.get("num-personalization-layers", 1))
    partition_id              = int(context.node_config["partition-id"])

    # ── 1. Receive base weights from server ───────────────────────────
    base_params = _arrays_to_numpy(msg.content["arrays"])

    # ── 2. Load / initialize personalized weights ─────────────────────
    pers_key = "personalized_params"
    if pers_key in context.state and len(context.state[pers_key]) > 0:
        # Subsequent rounds — restore saved personalized params
        pers_record = context.state[pers_key]
        pers_params = _arrays_to_numpy(pers_record)
    else:
        # First time — initialize personalized layers from a fresh model
        init_model = Net()
        pers_params = init_model.get_personalized_model_parameters(kp)

    # ── 3. Assemble full local model ──────────────────────────────────
    model = Net().to(device)
    model.set_base_model_parameters(base_params, kp)
    if kp > 0:
        model.set_personalized_model_parameters(pers_params, kp)

    # ── 4. Local training ─────────────────────────────────────────────
    train_loader, _ = load_client_data(partition_id, data_dir, batch_size)
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)

    total_loss, total_acc = 0.0, 0.0
    for _ in range(local_epochs):
        loss, acc = model.train_epoch(train_loader, criterion, optimizer, device)
        total_loss += loss
        total_acc  += acc

    train_loss = total_loss / local_epochs
    train_acc  = total_acc  / local_epochs

    # ── 5. Save personalized weights to context.state ─────────────────
    updated_pers = model.get_personalized_model_parameters(kp)
    if kp > 0:
        context.state[pers_key] = _numpy_to_record(
            updated_pers, _pers_keys(kp)
        )

    # ── 6. Send only base layer weights back to server ─────────────────
    updated_base = model.get_base_model_parameters(kp)

    content = RecordDict({
        "arrays": _numpy_to_record(updated_base, _base_keys(kp)),
        "metrics": MetricRecord({
            "train_loss":   float(train_loss),
            "train_acc":    float(train_acc),
            "num_examples": int(len(train_loader.dataset)),
        }),
    })
    return Message(content=content, reply_to=msg)


@app.evaluate()
def evaluate(msg: Message, context: Context) -> Message:
    """
    Fonction d'évaluation FedPer client.
    
    Étapes:
      1. Recevoir les couches BASE du serveur
      2. Restaurer les couches PERSONNALISÉES depuis context.state
      3. Assembler le modèle complet
      4. Évaluer sur les données de validation locales
      5. Retourner les métriques d'évaluation
    
    Remarque: L'évaluation utilise le modèle complet (base + pers)
    pour évaluer la performance locale de chaque client.
    """

    # ── 1. Receive base weights from server ───────────────────────────
    base_params = _arrays_to_numpy(msg.content["arrays"])

    # ── 2. Load personalized weights (if any) ─────────────────────────
    pers_key = "personalized_params"
    if pers_key in context.state and len(context.state[pers_key]) > 0:
        pers_params = _arrays_to_numpy(context.state[pers_key])
    else:
        init_model  = Net()
        pers_params = init_model.get_personalized_model_parameters(kp)

    # ── 3. Assemble full local model ──────────────────────────────────
    model = Net().to(device)
    model.set_base_model_parameters(base_params, kp)
    if kp > 0:
        model.set_personalized_model_parameters(pers_params, kp)

    # ── 4. Evaluate on local val data ─────────────────────────────────
    _, val_loader = load_client_data(partition_id, data_dir, batch_size)
    criterion = nn.CrossEntropyLoss().to(device)
    eval_loss, eval_acc = model.test_epoch(val_loader, criterion, device)

    content = RecordDict({
        "metrics": MetricRecord({
            "eval_loss":    float(eval_loss),
            "eval_acc":     float(eval_acc),
            "num_examples": int(len(val_loader.dataset)),
        }),
    })
    return Message(content=content, reply_to=msg)
