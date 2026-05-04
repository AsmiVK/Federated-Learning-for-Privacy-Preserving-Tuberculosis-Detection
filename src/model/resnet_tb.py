# src/model/resnet_tb.py
"""
ResNet-18 modified for TB detection (3-class).
Architecture law:
  - Layer1, Layer2  → FROZEN  (ImageNet generic features)
  - Layer3, Layer4  → TRAINABLE (TB-specific: cavities, opacities)
  - FC: 512 → 3    → MODIFIED output head
"""

import torch
import torch.nn as nn
from torchvision import models
from typing import Dict, Tuple


class TBResNet18(nn.Module):
    def __init__(self, num_classes: int = 3, pretrained: bool = True):
        super(TBResNet18, self).__init__()

        # ── Load backbone ─────────────────────────────────────────
        weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = models.resnet18(weights=weights)

        # ── Keep all layers except the original FC ─────────────────
        self.conv1   = backbone.conv1    # 7×7, 64, stride 2
        self.bn1     = backbone.bn1
        self.relu    = backbone.relu
        self.maxpool = backbone.maxpool

        self.layer1  = backbone.layer1   # 64ch,  2 blocks — FROZEN
        self.layer2  = backbone.layer2   # 128ch, 2 blocks — FROZEN
        self.layer3  = backbone.layer3   # 256ch, 2 blocks — TRAINABLE
        self.layer4  = backbone.layer4   # 512ch, 2 blocks — TRAINABLE

        self.avgpool = backbone.avgpool  # Global Average Pooling

        # ── Modified FC: 512 → 3 (Healthy / Active TB / Latent TB) ─
        self.fc = nn.Linear(512, num_classes)

        # ── Apply freeze / unfreeze policy ────────────────────────
        self._apply_freeze_policy()

    def _apply_freeze_policy(self):
        """Freeze early layers, unfreeze TB-specific layers."""
        # Freeze: conv1, bn1, layer1, layer2
        frozen_modules = [self.conv1, self.bn1, self.layer1, self.layer2]
        for module in frozen_modules:
            for param in module.parameters():
                param.requires_grad = False

        # Trainable: layer3, layer4, fc (already requires_grad=True by default)
        trainable_modules = [self.layer3, self.layer4, self.fc]
        for module in trainable_modules:
            for param in module.parameters():
                param.requires_grad = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input: (B, 3, 224, 224)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)       # → (B, 64, 56, 56)

        x = self.layer1(x)        # → (B, 64,  56, 56)  FROZEN
        x = self.layer2(x)        # → (B, 128, 28, 28)  FROZEN
        x = self.layer3(x)        # → (B, 256, 14, 14)  TRAINABLE
        x = self.layer4(x)        # → (B, 512,  7,  7)  TRAINABLE

        x = self.avgpool(x)       # → (B, 512, 1, 1)
        x = torch.flatten(x, 1)   # → (B, 512)
        x = self.fc(x)            # → (B, 3)
        return x

    def get_param_stats(self) -> Dict[str, int]:
        """Return counts of trainable vs frozen parameters."""
        total    = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen   = total - trainable
        return {"total": total, "trainable": trainable, "frozen": frozen}

    def get_trainable_params(self):
        """Return only parameters that will be updated during training."""
        return [p for p in self.parameters() if p.requires_grad]


def build_model(num_classes: int = 3, pretrained: bool = True) -> TBResNet18:
    """Factory function — use this everywhere instead of direct instantiation."""
    return TBResNet18(num_classes=num_classes, pretrained=pretrained)