# src/federated/client.py
"""
Flower federated learning client.
Each client = one hospital node (India / South Africa / Eastern Europe).
Performs local training per round with class-weighted loss, then uploads weights.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from typing import Dict, List, Tuple
import flwr as fl
import numpy as np

from src.model.resnet_tb import build_model
from src.model.model_utils import get_model_weights, set_model_weights


class TBClient(fl.client.NumPyClient):

    def __init__(
        self,
        node_name: str,
        train_dataset: Dataset,
        val_dataset: Dataset,
        num_classes: int = 3,
        local_epochs: int = 5,
        batch_size: int = 32,
        learning_rate: float = 0.001,
        mu: float = 0.01,
        device: str = "cpu",
    ):
        self.node_name     = node_name
        self.train_dataset = train_dataset
        self.val_dataset   = val_dataset
        self.num_classes   = num_classes
        self.local_epochs  = local_epochs
        self.batch_size    = batch_size
        self.learning_rate = learning_rate
        self.mu            = mu
        self.device        = torch.device(device)

        self.model = build_model(num_classes=num_classes, pretrained=False)
        self.model.to(self.device)

        self.train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            sampler=self._make_balanced_sampler(),
            num_workers=0,
            pin_memory=False
        )
        self.val_loader = DataLoader(
            val_dataset, batch_size=batch_size,
            shuffle=False, num_workers=0, pin_memory=False
        )

        # Compute class weights once at init
        self.class_weights = self._compute_class_weights()
        print(f"  ⚖  [{node_name}] class weights: "
              f"Healthy={self.class_weights[0]:.2f}  "
              f"ActiveTB={self.class_weights[1]:.2f}  "
              f"LatentTB={self.class_weights[2]:.2f}")

    def _compute_class_weights(self) -> torch.Tensor:
        """No class weighting in loss — balanced sampler handles imbalance."""
        return torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32)

    def _make_balanced_sampler(self):
        """
        WeightedRandomSampler — oversample minority classes so each
        batch sees roughly equal Healthy / Active TB / Latent TB.
        Unlike class-weighted loss, this doesn't push predictions
        toward TB — it just ensures the model sees enough TB examples.
        """
        from torch.utils.data import WeightedRandomSampler

        labels = []

        def collect_labels(ds):
            if hasattr(ds, 'samples'):
                for _, lbl in ds.samples:
                    labels.append(lbl)
            elif hasattr(ds, 'dataset') and hasattr(ds, 'indices'):
                for i in ds.indices:
                    _, lbl = ds.dataset.samples[i]
                    labels.append(lbl)
            elif hasattr(ds, 'dataset_a'):
                collect_labels(ds.dataset_a)
                collect_labels(ds.dataset_b)

        collect_labels(self.train_dataset)

        # Weight per sample = 1 / class_count
        from collections import Counter
        counts = Counter(labels)
        weight_per_class = {
            cls: 1.0 / max(count, 1)
            for cls, count in counts.items()
        }
        sample_weights = [
            weight_per_class.get(lbl, 1.0) for lbl in labels
        ]

        print(f"  ⚖  [{self.node_name}] balanced sampler: "
            f"{dict(sorted(counts.items()))}")

        return WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True
        )

    # ── Flower interface ──────────────────────────────────────────────────────

    def get_parameters(self, config: Dict) -> List[np.ndarray]:
        return [w.cpu().numpy() for w in get_model_weights(self.model)]

    def set_parameters(self, parameters: List[np.ndarray]):
        weights = [torch.tensor(p) for p in parameters]
        set_model_weights(self.model, weights)

    def fit(
        self, parameters: List[np.ndarray], config: Dict
    ) -> Tuple[List[np.ndarray], int, Dict]:
        self.set_parameters(parameters)

        # Save global weights for FedProx proximal term
        global_weights = [
            w.clone().detach() for w in get_model_weights(self.model)
        ]

        optimizer = torch.optim.Adam(
            self.model.get_trainable_params(),
            lr=self.learning_rate,
            weight_decay=1e-4,
        )
        criterion = nn.CrossEntropyLoss(
            weight=self.class_weights.to(self.device)
        )

        self.model.train()
        total_loss    = 0.0
        total_correct = 0
        total_samples = 0

        for epoch in range(self.local_epochs):
            for images, labels in self.train_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)

                optimizer.zero_grad()
                outputs = self.model(images)

                ce_loss = criterion(outputs, labels)

                # FedProx proximal term: (mu/2) * ||w - w_global||^2
                prox_loss = torch.tensor(0.0, device=self.device)
                if self.mu > 0:
                    for w_curr, w_glob in zip(
                        get_model_weights(self.model), global_weights
                    ):
                        prox_loss += torch.norm(
                            w_curr.to(self.device) - w_glob.to(self.device)
                        ) ** 2
                    prox_loss = (self.mu / 2) * prox_loss

                loss = ce_loss + prox_loss
                loss.backward()
                optimizer.step()

                total_loss    += ce_loss.item() * len(labels)
                preds          = outputs.argmax(dim=1)
                total_correct += (preds == labels).sum().item()
                total_samples += len(labels)

        avg_loss = total_loss / max(total_samples, 1)
        accuracy = total_correct / max(total_samples, 1)

        return (
            self.get_parameters(config={}),
            total_samples,
            {"train_loss": avg_loss, "train_accuracy": accuracy,
             "node": self.node_name},
        )

    def evaluate(
        self, parameters: List[np.ndarray], config: Dict
    ) -> Tuple[float, int, Dict]:
        self.set_parameters(parameters)
        criterion = nn.CrossEntropyLoss()   # unweighted for true eval loss

        self.model.eval()
        total_loss    = 0.0
        total_correct = 0
        total_samples = 0

        with torch.no_grad():
            for images, labels in self.val_loader:
                images  = images.to(self.device)
                labels  = labels.to(self.device)
                outputs = self.model(images)
                loss    = criterion(outputs, labels)

                total_loss    += loss.item() * len(labels)
                preds          = outputs.argmax(dim=1)
                total_correct += (preds == labels).sum().item()
                total_samples += len(labels)

        avg_loss = total_loss / max(total_samples, 1)
        accuracy = total_correct / max(total_samples, 1)

        return (
            avg_loss,
            total_samples,
            {"val_accuracy": accuracy, "node": self.node_name},
        )