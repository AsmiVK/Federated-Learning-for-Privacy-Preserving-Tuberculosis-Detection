# scripts/run_training.py
"""
Main training script. Run this to execute the full experiment.

Usage:
    # Quick smoke test (mock data, 2 rounds):
    python scripts/run_training.py --rounds 2 --mock

    # Full run on real data:
    python scripts/run_training.py --rounds 25 --strategy fedprox

    # Compare both strategies:
    python scripts/run_training.py --rounds 25 --compare
"""

import sys, argparse, json, os, torch
sys.path.insert(0, ".")

from torch.utils.data import DataLoader
from src.data.dataset import TBX11KDataset, ShenzhenDataset
from src.model.resnet_tb import build_model
from src.model.model_utils import load_model
from src.training.metrics import (
    evaluate_model, plot_confusion_matrix, plot_convergence
)
from src.federated.server import run_federated_training


def evaluate_saved_model(
    checkpoint_path: str,
    tbx_root: str,
    use_mock: bool,
    strategy_name: str,
):
    """Load a saved checkpoint and evaluate on the test set."""
    print(f"\n  Evaluating {strategy_name} model on test set...")

    test_ds = TBX11KDataset(tbx_root, split="test", use_mock=use_mock)
    if len(test_ds) == 0:
        print("  ⚠  Test set empty, skipping evaluation")
        return {}

    loader = DataLoader(test_ds, batch_size=32, shuffle=False, num_workers=0)

    if os.path.isfile(checkpoint_path):
        model = load_model(checkpoint_path)
    else:
        print(f"  ⚠  Checkpoint not found: {checkpoint_path}")
        print("  Using untrained model for metric demo...")
        model = build_model(num_classes=3, pretrained=False)

    metrics = evaluate_model(
        model, loader,
        device=torch.device("cpu"),
        verbose=True,
    )
    return metrics


def main():
    parser = argparse.ArgumentParser(
        description="Federated TB Detection Training"
    )
    parser.add_argument("--rounds",    type=int,   default=5)
    parser.add_argument("--strategy",  type=str,   default="fedprox",
                        choices=["fedavg", "fedprox"])
    parser.add_argument("--epochs",    type=int,   default=5)
    parser.add_argument("--batch",     type=int,   default=32)
    parser.add_argument("--lr", type=float, default=0.0003)
    parser.add_argument("--mu", type=float, default=0.001)
    parser.add_argument("--mock",      action="store_true",
                        help="Use mock data (for testing)")
    parser.add_argument("--compare",   action="store_true",
                        help="Run both FedAvg and FedProx and compare")
    parser.add_argument("--tbx",       type=str,
                        default="data/raw/TBX11K")
    parser.add_argument("--shenzhen",  type=str,
                        default="data/raw/Shenzhen")
    args = parser.parse_args()

    os.makedirs("runs", exist_ok=True)
    all_results = {}

    strategies = ["fedavg", "fedprox"] if args.compare else [args.strategy]

    for strat in strategies:
        print(f"\n{'='*55}")
        print(f"  Running: {strat.upper()}")
        print(f"{'='*55}")

        results = run_federated_training(
            tbx_root      = args.tbx,
            shenzhen_root = args.shenzhen,
            num_rounds    = args.rounds,
            strategy_name = strat,
            local_epochs  = args.epochs,
            batch_size    = args.batch,
            learning_rate = args.lr,
            mu            = args.mu,
            use_mock      = args.mock,
            save_dir      = "checkpoints",
            results_path  = f"runs/results_{strat}.json",
        )
        all_results[strat] = results

        # Evaluate saved model
        ckpt = f"checkpoints/global_model_{strat}_r{args.rounds}.pth"
        metrics = evaluate_saved_model(ckpt, args.tbx, args.mock, strat)

        if metrics:
            plot_confusion_matrix(
                metrics,
                save_path=f"runs/confusion_{strat}.png",
                title=f"Confusion Matrix — {strat.upper()}",
            )
            all_results[strat]["test_metrics"] = metrics

    # Convergence plot (compare strategies if both ran)
    fedavg_rounds  = all_results.get("fedavg",  {}).get("round_metrics", [])
    fedprox_rounds = all_results.get("fedprox", {}).get("round_metrics", [])
    if fedavg_rounds or fedprox_rounds:
        plot_convergence(
            fedavg_rounds, fedprox_rounds,
            save_path="runs/convergence.png",
        )

    # Save combined results
    with open("runs/all_results.json", "w") as f:
        json.dump(
            {k: {kk: vv for kk, vv in v.items()
                 if kk != "history"}        # skip non-serializable history
             for k, v in all_results.items()},
            f, indent=2, default=str
        )
    print("\n  ✅ All results saved to runs/all_results.json")
    print("  ✅ Plots saved to runs/")


if __name__ == "__main__":
    main()