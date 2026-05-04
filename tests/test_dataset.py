# tests/test_dataset.py
import sys, torch
sys.path.insert(0, ".")

from torch.utils.data import DataLoader
from src.data.dataset import TBX11KDataset, ShenzhenDataset, CLASS_NAMES
from src.data.federated_split import create_federated_splits

def test_dataset():
    print("\n" + "="*50)
    print("  CHECKPOINT 3 — Data Pipeline Test")
    print("="*50)

    # 1. TBX11K mock dataset
    tbx = TBX11KDataset(root_dir="data/raw/TBX11K", split="train", use_mock=True)
    assert len(tbx) > 0, "❌ TBX11K dataset is empty"
    img, label = tbx[0]
    assert img.shape == (3, 224, 224), f"❌ Wrong shape: {img.shape}"
    assert label in [0, 1, 2],        f"❌ Invalid label: {label}"
    print(f"\n  ✓ TBX11K mock loaded — {len(tbx)} samples")
    print(f"  ✓ Sample shape: {img.shape}, label: {CLASS_NAMES[label]}")

    counts = tbx.get_class_counts()
    print(f"  ✓ Class distribution: {counts}")

    # 2. Shenzhen mock dataset
    shen = ShenzhenDataset(root_dir="data/raw/Shenzhen", use_mock=True)
    assert len(shen) > 0
    img2, label2 = shen[0]
    assert img2.shape == (3, 224, 224)
    print(f"\n  ✓ Shenzhen mock loaded — {len(shen)} samples")

    # 3. DataLoader
    loader = DataLoader(tbx, batch_size=8, shuffle=True, num_workers=0)
    batch_imgs, batch_labels = next(iter(loader))
    assert batch_imgs.shape == (8, 3, 224, 224)
    assert batch_labels.shape == (8,)
    print(f"\n  ✓ DataLoader OK — batch shape: {batch_imgs.shape}")

    # 4. Pixel value range after normalization
    assert batch_imgs.min() < 0, "❌ Images don't appear normalized (min should be < 0)"
    print(f"  ✓ Normalization OK — pixel range: [{batch_imgs.min():.2f}, {batch_imgs.max():.2f}]")

    # 5. Federated split
    print(f"\n  Creating Non-IID federated split...")
    node_datasets = create_federated_splits(tbx, shen, verbose=True)

    assert set(node_datasets.keys()) == {"india", "south_africa", "eastern_europe"}
    for node_name, ds in node_datasets.items():
        assert len(ds) > 0, f"❌ Node {node_name} is empty"

    # Node 3 should be larger due to Shenzhen addition
    n3 = len(node_datasets["eastern_europe"])
    n1 = len(node_datasets["india"])
    print(f"\n  ✓ Node sizes — India: {n1}, "
          f"S.Africa: {len(node_datasets['south_africa'])}, "
          f"E.Europe: {n3} (incl. Shenzhen)")

    # 6. Each node returns valid batches
    for node_key, ds in node_datasets.items():
        loader = DataLoader(ds, batch_size=4, shuffle=False, num_workers=0)
        imgs, labels = next(iter(loader))
        assert imgs.shape[1:] == (3, 224, 224)
    print(f"  ✓ All 3 nodes produce valid DataLoader batches")

    print("\n✅  CHECKPOINT 3 PASSED — Data pipeline is ready!\n")

if __name__ == "__main__":
    test_dataset()