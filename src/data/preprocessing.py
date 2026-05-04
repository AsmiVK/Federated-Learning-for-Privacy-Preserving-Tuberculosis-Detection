# src/data/preprocessing.py
"""
Image transforms for TB X-ray classification.
All images are grayscale chest X-rays → converted to 3-channel RGB
for ResNet-18 compatibility.
"""

from torchvision import transforms


def get_train_transforms(image_size: int = 224) -> transforms.Compose:
    """Augmented pipeline for training — adds robustness."""
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.Grayscale(num_output_channels=3),   # X-ray → 3ch RGB
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(                           # ImageNet stats
            mean=[0.485, 0.456, 0.406],
            std= [0.229, 0.224, 0.225]
        ),
    ])


def get_val_transforms(image_size: int = 224) -> transforms.Compose:
    """No augmentation for val/test — deterministic."""
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std= [0.229, 0.224, 0.225]
        ),
    ])