# src/data/federated_split.py
"""
Non-IID federated distribution across 3 hospital nodes.

Real TBX11K class counts (train split):
  Healthy:   ~5320  (majority)
  Active TB:  ~489  (minority)
  Latent TB:   ~73  (rare)

Strategy: distribute EACH CLASS separately with different per-node
ratios to create Non-IID statistical heterogeneity.

Per-class node fractions (must sum to 1.0):
              Healthy  ActiveTB  LatentTB
  India:        0.40     0.35      0.40   ← moderate TB burden
  S.Africa:     0.35     0.50      0.45   ← highest TB burden
  E.Europe:     0.25     0.15      0.15   ← lowest TB + Shenzhen domain shift

This guarantees every node sees all 3 classes.
"""

import numpy as np
from torch.utils.data import Dataset, Subset
from typing import Dict, List, Tuple
from src.data.dataset import TBX11KDataset, ShenzhenDataset, CLASS_NAMES

# Per-class split fractions across nodes (each column sums to 1.0)
NODE_CLASS_FRACTIONS = {
    #              label0    label1    label2
    "india":        {0: 0.40, 1: 0.35, 2: 0.40},
    "south_africa": {0: 0.35, 1: 0.50, 2: 0.45},
    "eastern_europe":{0: 0.25, 1: 0.15, 2: 0.15},
}

NODE_NAMES = {
    "india":         "Node 1 — India (High TB)",
    "south_africa":  "Node 2 — South Africa (Very High TB)",
    "eastern_europe":"Node 3 — Eastern Europe (MDR-TB)",
}


def _get_indices_by_class(dataset: Dataset) -> Dict[int, List[int]]:
    """Return {class_label: [sample_indices]} for all classes present."""
    class_indices: Dict[int, List[int]] = {}
    for idx, (_, label) in enumerate(dataset.samples):
        class_indices.setdefault(label, []).append(idx)
    return class_indices


def create_federated_splits(
    tbx_dataset: TBX11KDataset,
    shenzhen_dataset: ShenzhenDataset,
    seed: int = 42,
    verbose: bool = True,
) -> Dict[str, Dataset]:
    """
    Partition TBX11K into Non-IID node subsets by distributing
    each class independently across nodes.
    """
    rng           = np.random.default_rng(seed)
    class_indices = _get_indices_by_class(tbx_dataset)

    if verbose:
        print(f"\n  Available classes in TBX11K train split:")
        for label, indices in sorted(class_indices.items()):
            name = CLASS_NAMES[label] if label < len(CLASS_NAMES) else f"class_{label}"
            print(f"    label {label} ({name:<12}): {len(indices)} samples")

    # Shuffle each class's indices once
    shuffled: Dict[int, List[int]] = {}
    for label, indices in class_indices.items():
        arr = np.array(indices)
        rng.shuffle(arr)
        shuffled[label] = arr.tolist()

    # Assign indices per node per class
    node_indices: Dict[str, List[int]] = {k: [] for k in NODE_CLASS_FRACTIONS}

    for label, indices in shuffled.items():
        n = len(indices)
        node_keys = list(NODE_CLASS_FRACTIONS.keys())

        # Calculate how many samples each node gets for this class
        counts = []
        for nk in node_keys:
            frac = NODE_CLASS_FRACTIONS[nk].get(label, 0.0)
            counts.append(int(n * frac))

        # Give any remainder to node with largest fraction
        remainder = n - sum(counts)
        if remainder > 0:
            biggest = max(range(len(node_keys)),
                         key=lambda i: NODE_CLASS_FRACTIONS[node_keys[i]].get(label, 0))
            counts[biggest] += remainder

        # Slice and assign
        pos = 0
        for nk, cnt in zip(node_keys, counts):
            node_indices[nk] += indices[pos: pos + cnt]
            pos += cnt

    # Build Dataset objects
    node_datasets: Dict[str, Dataset] = {}
    for node_key, indices in node_indices.items():
        subset = Subset(tbx_dataset, indices)

        if node_key == "eastern_europe" and len(shenzhen_dataset) > 0:
            combined = CombinedDataset(subset, shenzhen_dataset)
            node_datasets[node_key] = combined
        else:
            node_datasets[node_key] = subset

        if verbose:
            _print_node_stats(
                node_key, NODE_NAMES[node_key], indices,
                tbx_dataset,
                node_key == "eastern_europe",
                shenzhen_dataset,
            )

    return node_datasets


class CombinedDataset(Dataset):
    """Concatenates two datasets — used for Node 3 (TBX + Shenzhen)."""

    def __init__(self, dataset_a: Dataset, dataset_b: Dataset):
        self.dataset_a = dataset_a
        self.dataset_b = dataset_b
        self.len_a     = len(dataset_a)

    def __len__(self) -> int:
        return len(self.dataset_a) + len(self.dataset_b)

    def __getitem__(self, idx: int) -> Tuple:
        if idx < self.len_a:
            return self.dataset_a[idx]
        return self.dataset_b[idx - self.len_a]


def _print_node_stats(
    node_key: str,
    node_name: str,
    indices: List[int],
    dataset: TBX11KDataset,
    use_shenzhen: bool,
    shenzhen_ds: ShenzhenDataset,
):
    label_counts: Dict[str, int] = {}
    for i in indices:
        _, lbl = dataset.samples[i]
        name = CLASS_NAMES[lbl] if lbl < len(CLASS_NAMES) else f"class_{lbl}"
        label_counts[name] = label_counts.get(name, 0) + 1

    total = len(indices)
    shen  = f" + {len(shenzhen_ds)} Shenzhen" if use_shenzhen else ""
    print(f"\n  📍 {node_name}  ({total}{shen} samples)")
    for cls in CLASS_NAMES:
        cnt = label_counts.get(cls, 0)
        pct = cnt / total * 100 if total > 0 else 0
        bar = "█" * int(pct / 5)
        print(f"     {cls:<12}: {cnt:>4}  ({pct:4.1f}%)  {bar}")