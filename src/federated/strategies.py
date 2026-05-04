# src/federated/strategies.py
"""
FedAvg (baseline) and FedProx aggregation strategies.
Both are implemented as Flower Strategy subclasses.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from flwr.common import (
    FitRes, Parameters, Scalar,
    ndarrays_to_parameters, parameters_to_ndarrays,
)
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg


class FedAvgStrategy(FedAvg):
    """
    Standard FedAvg — weighted average by dataset size.
    w_global = Σ (n_k / N) * w_k
    """

    def __init__(self, num_rounds: int = 25, **kwargs):
        super().__init__(**kwargs)
        self.num_rounds  = num_rounds
        self.round_metrics: List[Dict] = []

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures,
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:

        # Collect weights and sample counts
        weights_list = []
        sample_counts = []
        metrics_list  = []

        for _, fit_res in results:
            weights      = parameters_to_ndarrays(fit_res.parameters)
            n_samples    = fit_res.num_examples
            weights_list.append(weights)
            sample_counts.append(n_samples)
            metrics_list.append(fit_res.metrics)

        # Weighted average
        total = sum(sample_counts)
        aggregated = [
            sum(
                (n / total) * w[i]
                for w, n in zip(weights_list, sample_counts)
            )
            for i in range(len(weights_list[0]))
        ]

        # Log round metrics
        avg_train_acc = np.mean([
            m.get("train_accuracy", 0) for m in metrics_list
        ])
        round_log = {
            "round": server_round,
            "strategy": "FedAvg",
            "avg_train_accuracy": float(avg_train_acc),
            "total_samples": total,
        }
        self.round_metrics.append(round_log)
        print(f"  [FedAvg] Round {server_round:>2} | "
              f"avg_train_acc={avg_train_acc:.4f} | "
              f"clients={len(results)}")

        return ndarrays_to_parameters(aggregated), round_log


class FedProxStrategy(FedAvg):
    """
    FedProx — same aggregation as FedAvg on the server side,
    but clients add proximal term during local training (handled in client.py).
    This class tracks convergence metrics for comparison with FedAvg.
    """

    def __init__(self, mu: float = 0.01, num_rounds: int = 25, **kwargs):
        super().__init__(**kwargs)
        self.mu          = mu
        self.num_rounds  = num_rounds
        self.round_metrics: List[Dict] = []

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures,
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:

        weights_list  = []
        sample_counts = []
        metrics_list  = []

        for _, fit_res in results:
            weights      = parameters_to_ndarrays(fit_res.parameters)
            n_samples    = fit_res.num_examples
            weights_list.append(weights)
            sample_counts.append(n_samples)
            metrics_list.append(fit_res.metrics)

        # Weighted average (same as FedAvg — proximal constraint was on clients)
        total = sum(sample_counts)
        aggregated = [
            sum(
                (n / total) * w[i]
                for w, n in zip(weights_list, sample_counts)
            )
            for i in range(len(weights_list[0]))
        ]

        avg_train_acc = np.mean([
            m.get("train_accuracy", 0) for m in metrics_list
        ])
        round_log = {
            "round": server_round,
            "strategy": "FedProx",
            "mu": self.mu,
            "avg_train_accuracy": float(avg_train_acc),
            "total_samples": total,
        }
        self.round_metrics.append(round_log)
        print(f"  [FedProx μ={self.mu}] Round {server_round:>2} | "
              f"avg_train_acc={avg_train_acc:.4f} | "
              f"clients={len(results)}")

        return ndarrays_to_parameters(aggregated), round_log