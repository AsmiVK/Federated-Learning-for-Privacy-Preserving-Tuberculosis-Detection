# src/training/metrics.py
"""
Evaluation metrics for TB detection.
All metrics computed from model predictions on a test DataLoader.

Required metrics (from project spec):
  - Weighted F1-Score      (primary — handles class imbalance)
  - AUC-ROC (OvR)          (threshold-independent)
  - Cohen's Kappa          (agreement beyond chance)
  - Per-class Sensitivity  (recall per class)
  - Per-class Specificity
  - Accuracy
  - Confusion Matrix
"""

import torch
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import (
    f1_score, roc_auc_score, cohen_kappa_score,
    confusion_matrix, accuracy_score,
    classification_report,
)
from typing import Dict, Tuple
import matplotlib
matplotlib.use("Agg")   # non-interactive backend — safe on Windows
import matplotlib.pyplot as plt
import seaborn as sns
import os


CLASS_NAMES = ["Healthy", "Active TB", "Latent TB"]


def run_inference(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Run model on all batches.
    Returns:
        all_labels  : (N,)    ground truth class indices
        all_preds   : (N,)    predicted class indices
        all_probs   : (N, 3)  softmax probabilities
    """
    model.eval()
    all_labels, all_preds, all_probs = [], [], []

    with torch.no_grad():
        for images, labels in loader:
            images  = images.to(device)
            outputs = model(images)
            probs   = torch.softmax(outputs, dim=1).cpu().numpy()
            preds   = outputs.argmax(dim=1).cpu().numpy()

            all_labels.append(labels.numpy())
            all_preds.append(preds)
            all_probs.append(probs)

    return (
        np.concatenate(all_labels),
        np.concatenate(all_preds),
        np.concatenate(all_probs),
    )


def compute_metrics(
    labels: np.ndarray,
    preds: np.ndarray,
    probs: np.ndarray,
    verbose: bool = True,
) -> Dict:
    """
    Compute all required metrics from predictions.
    Returns a dict with every metric value.
    """
    # ── Core metrics ──────────────────────────────────────────────────────────
    accuracy = accuracy_score(labels, preds)
    f1_weighted = f1_score(labels, preds, average="weighted", zero_division=0)
    kappa = cohen_kappa_score(labels, preds)

    # AUC-ROC: needs probabilities for all classes present in labels
    unique_classes = np.unique(labels)
    if len(unique_classes) >= 2:
        try:
            # Use only columns for classes present in labels
            auc_roc = roc_auc_score(
                labels,
                probs[:, :len(unique_classes)] if len(unique_classes) < 3
                else probs,
                multi_class="ovr",
                average="weighted",
                labels=list(unique_classes),
            )
        except Exception:
            auc_roc = float("nan")
    else:
        auc_roc = float("nan")

    # ── Confusion matrix ──────────────────────────────────────────────────────
    present = sorted(unique_classes)
    cm = confusion_matrix(labels, preds, labels=present)

    # ── Per-class Sensitivity & Specificity ───────────────────────────────────
    sensitivity = {}    # True Positive Rate = TP / (TP + FN)
    specificity = {}    # True Negative Rate = TN / (TN + FP)

    for i, cls_idx in enumerate(present):
        cls_name = CLASS_NAMES[cls_idx]
        if i < len(cm):
            tp = cm[i, i]
            fn = cm[i, :].sum() - tp
            fp = cm[:, i].sum() - tp
            tn = cm.sum() - tp - fn - fp

            sensitivity[cls_name] = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            specificity[cls_name] = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    # ── Per-class F1 ──────────────────────────────────────────────────────────
    f1_per_class_vals = f1_score(
        labels, preds, average=None,
        labels=present, zero_division=0
    )
    f1_per_class = {
        CLASS_NAMES[cls_idx]: float(f1_per_class_vals[i])
        for i, cls_idx in enumerate(present)
    }

    metrics = {
        "accuracy":      float(accuracy),
        "f1_weighted":   float(f1_weighted),
        "auc_roc":       float(auc_roc),
        "cohen_kappa":   float(kappa),
        "sensitivity":   sensitivity,
        "specificity":   specificity,
        "f1_per_class":  f1_per_class,
        "confusion_matrix": cm.tolist(),
        "n_samples":     int(len(labels)),
        "classes_present": [CLASS_NAMES[i] for i in present],
    }

    if verbose:
        _print_metrics(metrics)

    return metrics


def _print_metrics(metrics: Dict):
    """Pretty-print the metrics table."""
    print("\n  ┌─────────────────────────────────────────────────┐")
    print("  │              Evaluation Results                 │")
    print("  ├─────────────────────────────────────────────────┤")
    print(f"  │  Accuracy       : {metrics['accuracy']:.4f}                      │")
    print(f"  │  F1 (weighted)  : {metrics['f1_weighted']:.4f}  ← PRIMARY METRIC  │")
    print(f"  │  AUC-ROC (OvR)  : {metrics['auc_roc']:.4f}                      │")
    print(f"  │  Cohen's Kappa  : {metrics['cohen_kappa']:.4f}                      │")
    print("  ├─────────────────────────────────────────────────┤")
    print("  │  Per-Class Sensitivity (Recall):                │")
    for cls, val in metrics["sensitivity"].items():
        flag = "  ⚠ LOW" if cls == "Active TB" and val < 0.80 else ""
        print(f"  │    {cls:<12}: {val:.4f}{flag:<20}         │")
    print("  ├─────────────────────────────────────────────────┤")
    print("  │  Per-Class Specificity:                         │")
    for cls, val in metrics["specificity"].items():
        print(f"  │    {cls:<12}: {val:.4f}                         │")
    print("  └─────────────────────────────────────────────────┘")


def evaluate_model(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device = torch.device("cpu"),
    verbose: bool = True,
) -> Dict:
    """Full evaluation pipeline — runs inference then computes all metrics."""
    labels, preds, probs = run_inference(model, loader, device)
    return compute_metrics(labels, preds, probs, verbose=verbose)


def plot_confusion_matrix(
    metrics: Dict,
    save_path: str = "runs/confusion_matrix.png",
    title: str = "Confusion Matrix",
):
    """Save a styled confusion matrix heatmap."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    cm     = np.array(metrics["confusion_matrix"])
    labels = metrics["classes_present"]

    fig, ax = plt.subplots(figsize=(7, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=labels, yticklabels=labels,
        linewidths=0.5, ax=ax,
    )
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("Actual",    fontsize=12)
    ax.set_title(title,        fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  ✓ Confusion matrix saved → {save_path}")


def plot_convergence(
    round_metrics_fedavg: list,
    round_metrics_fedprox: list,
    save_path: str = "runs/convergence.png",
):
    """Plot FedAvg vs FedProx accuracy convergence curves."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 5))

    if round_metrics_fedavg:
        rounds = [r["round"] for r in round_metrics_fedavg]
        accs   = [r["avg_train_accuracy"] for r in round_metrics_fedavg]
        ax.plot(rounds, accs, "r-o", label="FedAvg (baseline)", linewidth=2)

    if round_metrics_fedprox:
        rounds = [r["round"] for r in round_metrics_fedprox]
        accs   = [r["avg_train_accuracy"] for r in round_metrics_fedprox]
        ax.plot(rounds, accs, "b-o", label="FedProx (ours)", linewidth=2)

    ax.set_xlabel("Communication Rounds", fontsize=12)
    ax.set_ylabel("Avg Train Accuracy",   fontsize=12)
    ax.set_title("FedAvg vs FedProx Convergence", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  ✓ Convergence plot saved → {save_path}")