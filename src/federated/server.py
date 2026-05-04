# src/federated/server.py
"""
Federated learning simulation runner.
Uses Flower's simulation API — all clients run in-process,
no networking required. Perfect for single-machine experiments.
"""

import json
import os
import torch
import numpy as np
from typing import Dict, List
from torch.utils.data import random_split, Subset

import flwr as fl
from flwr.common import ndarrays_to_parameters

from src.model.resnet_tb import build_model
from src.model.model_utils import get_model_weights, save_model
from src.data.dataset import TBX11KDataset, ShenzhenDataset
from src.data.federated_split import create_federated_splits
from src.federated.client import TBClient
from src.federated.strategies import FedAvgStrategy, FedProxStrategy


def _make_val_split(dataset, val_ratio: float = 0.15, seed: int = 42):
    """Split a dataset into train/val subsets."""
    n_val   = max(1, int(len(dataset) * val_ratio))
    n_train = len(dataset) - n_val
    generator = torch.Generator().manual_seed(seed)
    return random_split(dataset, [n_train, n_val], generator=generator)


def build_client_fn(node_datasets: Dict, val_datasets: Dict, cfg: dict):
    """
    Returns a Flower client_fn that Flower simulation calls per round.
    Maps partition_id (0/1/2) → node name → TBClient instance.
    """
    node_keys = list(node_datasets.keys())   # ["india", "south_africa", "eastern_europe"]
    clients   = {}

    for node_key in node_keys:
        clients[node_key] = TBClient(
            node_name      = node_key,
            train_dataset  = node_datasets[node_key],
            val_dataset    = val_datasets[node_key],
            num_classes    = cfg.get("num_classes", 3),
            local_epochs   = cfg.get("local_epochs", 5),
            batch_size     = cfg.get("batch_size", 32),
            learning_rate  = cfg.get("learning_rate", 0.001),
            mu             = cfg.get("mu", 0.01),
            device         = "cpu",
        )

    def client_fn(cid: str) -> fl.client.Client:
        node_key = node_keys[int(cid)]
        return clients[node_key].to_client()

    return client_fn, clients


def run_federated_training(
    tbx_root: str = "data/raw/TBX11K",
    shenzhen_root: str = "data/raw/Shenzhen",
    num_rounds: int = 25,
    strategy_name: str = "fedprox",   # "fedavg" | "fedprox"
    local_epochs: int = 5,
    batch_size: int = 32,
    learning_rate: float = 0.001,
    mu: float = 0.01,
    num_classes: int = 3,
    use_mock: bool = False,
    save_dir: str = "checkpoints",
    results_path: str = "runs/results.json",
    verbose: bool = True,
) -> Dict:
    """
    Main entry point for federated training.
    Returns dict with round-by-round metrics.
    """
    os.makedirs(save_dir,   exist_ok=True)
    os.makedirs("runs",     exist_ok=True)

    # ── Load datasets ─────────────────────────────────────────────────────────
    print("\n" + "="*55)
    print(f"  Federated Training  |  Strategy: {strategy_name.upper()}")
    print(f"  Rounds: {num_rounds}  |  Local epochs: {local_epochs}")
    print("="*55)

    tbx_train = TBX11KDataset(tbx_root, split="train", use_mock=use_mock)
    tbx_val   = TBX11KDataset(tbx_root, split="val",   use_mock=use_mock)
    shenzhen  = ShenzhenDataset(shenzhen_root,          use_mock=use_mock)

    # ── Federated Non-IID split ───────────────────────────────────────────────
    print("\n  Partitioning data across nodes (Non-IID)...")
    node_train = create_federated_splits(
        tbx_train, shenzhen, verbose=verbose
    )

    # Validation: give each node a slice of tbx_val
    node_val: Dict = {}
    val_size  = max(1, len(tbx_val) // 3)
    for i, node_key in enumerate(node_train.keys()):
        start = i * val_size
        end   = start + val_size if i < 2 else len(tbx_val)
        indices = list(range(start, min(end, len(tbx_val))))
        node_val[node_key] = Subset(tbx_val, indices)

    # ── Initial global model weights ──────────────────────────────────────────
    global_model = build_model(num_classes=num_classes, pretrained=True)
    init_weights = [w.numpy() for w in get_model_weights(global_model)]
    init_params  = ndarrays_to_parameters(init_weights)

    # ── Strategy ──────────────────────────────────────────────────────────────
    cfg = dict(
        num_classes=num_classes, local_epochs=local_epochs,
        batch_size=batch_size, learning_rate=learning_rate, mu=mu,
    )
    client_fn, client_objects = build_client_fn(node_train, node_val, cfg)

    strategy_kwargs = dict(
        fraction_fit        = 1.0,
        fraction_evaluate   = 1.0,
        min_fit_clients     = 3,
        min_evaluate_clients= 3,
        min_available_clients=3,
        initial_parameters  = init_params,
    )

    if strategy_name == "fedprox":
        strategy = FedProxStrategy(mu=mu, num_rounds=num_rounds,
                                   **strategy_kwargs)
    else:
        strategy = FedAvgStrategy(num_rounds=num_rounds, **strategy_kwargs)

    # ── Run simulation ────────────────────────────────────────────────────────
    print(f"\n  Starting Flower simulation ({num_rounds} rounds)...\n")
    ray_init_args = {"num_gpus": 0, "ignore_reinit_error": True}
    history = fl.simulation.start_simulation(
        client_fn          = client_fn,
        num_clients        = 3,
        config             = fl.server.ServerConfig(num_rounds=num_rounds),
        strategy           = strategy,
        client_resources   = {"num_cpus": 1, "num_gpus": 0.0},
        ray_init_args      = ray_init_args,
    )

    # ── Save final model ──────────────────────────────────────────────────────
    final_weights = strategy.round_metrics[-1] if strategy.round_metrics else {}
    save_path     = os.path.join(
        save_dir, f"global_model_{strategy_name}_r{num_rounds}.pth"
    )
    save_model(global_model, save_path,
               metadata={"strategy": strategy_name, "rounds": num_rounds})

    # ── Save results ──────────────────────────────────────────────────────────
    results = {
        "strategy":      strategy_name,
        "num_rounds":    num_rounds,
        "local_epochs":  local_epochs,
        "mu":            mu,
        "round_metrics": strategy.round_metrics,
        "history":       {
            "losses_distributed":   history.losses_distributed,
            "losses_centralized":   history.losses_centralized,
            "metrics_distributed":  str(history.metrics_distributed),
            "metrics_centralized":  str(history.metrics_centralized),
        },
    }
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved → {results_path}")

    return results