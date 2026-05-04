# src/model/model_utils.py
"""Utilities for saving, loading, and inspecting model weights."""

import torch
import os
from typing import List
from src.model.resnet_tb import TBResNet18, build_model


def save_model(model: TBResNet18, path: str, metadata: dict = None):
    """Save model weights + optional metadata."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = {"state_dict": model.state_dict()}
    if metadata:
        payload["metadata"] = metadata
    torch.save(payload, path)
    print(f"  ✅ Model saved → {path}")


def load_model(path: str, num_classes: int = 3) -> TBResNet18:
    """Load model weights from a checkpoint file."""
    model = build_model(num_classes=num_classes, pretrained=False)
    payload = torch.load(path, map_location="cpu")
    state = payload["state_dict"] if "state_dict" in payload else payload
    model.load_state_dict(state)
    print(f"  ✅ Model loaded ← {path}")
    return model


def get_model_weights(model: TBResNet18) -> List[torch.Tensor]:
    """Extract trainable weights as a flat list (used by Flower FL)."""
    return [p.data.clone() for p in model.parameters() if p.requires_grad]


def set_model_weights(model: TBResNet18, weights: List[torch.Tensor]):
    """Inject aggregated weights back into model (used by Flower FL)."""
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    assert len(trainable_params) == len(weights), (
        f"Weight count mismatch: model has {len(trainable_params)} "
        f"trainable params, got {len(weights)} weights"
    )
    with torch.no_grad():
        for param, weight in zip(trainable_params, weights):
            param.copy_(weight)


def print_model_summary(model: TBResNet18):
    """Print a clean summary of layers and freeze status."""
    stats = model.get_param_stats()
    print("\n  ┌─────────────────────────────────────────────┐")
    print("  │         ResNet-18 for TB Detection          │")
    print("  ├─────────────────────────────────────────────┤")
    layers = [
        ("conv1 + bn1 + relu + maxpool", model.conv1, "🔒 FROZEN"),
        ("layer1  (64ch,  2 blocks)",    model.layer1, "🔒 FROZEN"),
        ("layer2  (128ch, 2 blocks)",    model.layer2, "🔒 FROZEN"),
        ("layer3  (256ch, 2 blocks)",    model.layer3, "🔧 TRAINABLE"),
        ("layer4  (512ch, 2 blocks)",    model.layer4, "🔧 TRAINABLE"),
        ("avgpool → flatten",            model.avgpool, "🔧 TRAINABLE"),
        ("fc      (512 → 3)",            model.fc,     "🔧 TRAINABLE"),
    ]
    for name, _, status in layers:
        print(f"  │  {status}  {name:<35}│")
    print("  ├─────────────────────────────────────────────┤")
    print(f"  │  Total params    : {stats['total']:>10,}              │")
    print(f"  │  Trainable params: {stats['trainable']:>10,}              │")
    print(f"  │  Frozen params   : {stats['frozen']:>10,}              │")
    print("  └─────────────────────────────────────────────┘\n")