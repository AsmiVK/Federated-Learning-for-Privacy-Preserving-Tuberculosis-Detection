# src/data/dataset.py
"""
PyTorch Dataset classes for TBX11K and Shenzhen datasets.

TBX11K actual folder structure:
    data/raw/TBX11K/imgs/
        ├── health/   h*.png   → Healthy   (label 0)
        ├── sick/     s*.png   → Non-TB    (label 0, merged with healthy)
        └── tb/       tb*.png  → Active TB (label 1)

Shenzhen actual folder structure:
    data/raw/Shenzhen/
        ├── images/   CHNCXR_XXXX_0/1.png
        └── shenzhen_metadata.csv  (study_id, sex, age, findings)

Label scheme:
    0 = Healthy   (health/ + sick/ from TBX11K, "normal" from Shenzhen)
    1 = Active TB (tb/ from TBX11K, any TB finding from Shenzhen)
    2 = Latent TB (reserved — insufficient labels in raw data to use)

Note on 3-class: TBX11K's tb/ folder does not distinguish active vs latent
at the file level. We use 2 classes from real data and keep label 2
available for future annotation-based splitting.
"""

import os
import torch
import numpy as np
import pandas as pd
import json
from PIL import Image
from torch.utils.data import Dataset
from typing import List, Tuple, Optional
from src.data.preprocessing import get_train_transforms, get_val_transforms


CLASS_NAMES  = ["Healthy", "Active TB", "Latent TB"]
CLASS_TO_IDX = {"healthy": 0, "active_tb": 1, "latent_tb": 2}


class TBX11KDataset(Dataset):
    """
    Loads TBX11K dataset directly from folder structure:
        imgs/health/ → label 0
        imgs/sick/   → label 0  (sick non-TB treated as non-TB)
        imgs/tb/     → label 1  (active TB)
    Falls back to mock data if dataset not found.
    """

    def __init__(
        self,
        root_dir: str,
        split: str = "train",
        transform=None,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        seed: int = 42,
        use_mock: bool = False,
    ):
        self.root_dir  = root_dir
        self.split     = split
        self.transform = transform or (
            get_train_transforms() if split == "train" else get_val_transforms()
        )
        self.samples: List[Tuple[str, int]] = []
        self._use_mock = False

        if use_mock or not os.path.isdir(root_dir):
            print(f"  ⚠  TBX11K not found at '{root_dir}' — using mock data")
            self._load_mock_data()
        else:
            all_samples = self._scan_folders()
            if not all_samples:
                print(f"  ⚠  No images found in TBX11K folders — using mock data")
                self._load_mock_data()
            else:
                self.samples = self._split_samples(
                    all_samples, split, val_ratio, test_ratio, seed
                )
                print(f"  ✓  TBX11K [{split}]: {len(self.samples)} samples loaded")

    def _scan_folders(self) -> List[Tuple[str, int]]:
        """
        Load labels from COCO JSON annotations.
        Category mapping:
            1 = ActiveTuberculosis          → label 1 (Active TB)
            2 = ObsoletePulmonaryTuberculosis → label 2 (Latent TB)
            3 = PulmonaryTuberculosis       → label 1 (Active TB)
        health/ and sick/ images           → label 0 (Healthy)
        TB images with no annotation       → label 1 (Active TB, conservative)
        """
        imgs_dir   = os.path.join(self.root_dir, "imgs")
        if not os.path.isdir(imgs_dir):
            imgs_dir = self.root_dir

        # ── Step 1: Parse JSON to get per-image TB category ───────────
        # Use TBX11K_train.json (covers all 6600 images with TB annotations)
        json_path = os.path.join(
            self.root_dir, "annotations", "json", "TBX11K_train.json"
        )
        # For val split use TBX11K_val.json
        if self.split == "val":
            json_path = os.path.join(
                self.root_dir, "annotations", "json", "TBX11K_val.json"
            )

        # image_id → filename, annotation image_id → category_id
        img_id_to_fname: dict = {}
        img_id_to_label: dict = {}   # will hold final TB label per image_id

        if os.path.isfile(json_path):
            with open(json_path) as f:
                coco = json.load(f)

            for img in coco.get("images", []):
                img_id_to_fname[img["id"]] = img["file_name"]  # e.g. "tb/tb0005.png"

            # category_id → our label
            # 1=ActiveTB→1, 2=ObsoletePTB(Latent)→2, 3=PTB(Active)→1
            cat_to_label = {1: 1, 2: 2, 3: 1}

            for ann in coco.get("annotations", []):
                img_id  = ann["image_id"]
                cat_id  = ann["category_id"]
                new_lbl = cat_to_label.get(cat_id, 1)
                # If image has multiple annotations, take most severe
                # (Active TB > Latent TB)
                existing = img_id_to_label.get(img_id, 99)
                if new_lbl < existing:   # lower number = more severe
                    img_id_to_label[img_id] = new_lbl

            # Build fname → label dict (only for tb/ images)
            fname_to_label: dict = {}
            for img_id, fname in img_id_to_fname.items():
                if fname.startswith("tb/"):
                    lbl = img_id_to_label.get(img_id, 1)  # default Active TB
                    # fname is like "tb/tb0005.png" — get just the filename
                    fname_to_label[os.path.basename(fname)] = lbl

            print(f"  ✓  JSON parsed: {len(fname_to_label)} TB images with labels")
            active_count = sum(1 for v in fname_to_label.values() if v == 1)
            latent_count = sum(1 for v in fname_to_label.values() if v == 2)
            print(f"       Active TB (cat 1+3): {active_count}")
            print(f"       Latent TB (cat 2):   {latent_count}")
        else:
            print(f"  ⚠  JSON not found at {json_path}, falling back to all-Active-TB")
            fname_to_label = {}

        # ── Step 2: Scan folders and assign labels ─────────────────────
        all_samples = []

        # health/ and sick/ → Healthy (label 0)
        for folder in ["health", "sick"]:
            folder_path = os.path.join(imgs_dir, folder)
            if not os.path.isdir(folder_path):
                print(f"  ⚠  Folder not found: {folder_path}")
                continue
            files = sorted([
                f for f in os.listdir(folder_path)
                if f.lower().endswith((".png", ".jpg", ".jpeg"))
            ])
            for fname in files:
                all_samples.append((os.path.join(folder_path, fname), 0))
            print(f"  ✓  TBX11K imgs/{folder}/: {len(files)} images → label 0 (Healthy)")

        # tb/ → Active TB (1) or Latent TB (2) from JSON
        tb_path = os.path.join(imgs_dir, "tb")
        if os.path.isdir(tb_path):
            tb_files = sorted([
                f for f in os.listdir(tb_path)
                if f.lower().endswith((".png", ".jpg", ".jpeg"))
            ])
            active_used, latent_used, fallback = 0, 0, 0
            for fname in tb_files:
                if fname in fname_to_label:
                    lbl = fname_to_label[fname]
                else:
                    lbl = 1   # not in JSON → treat as Active TB
                    fallback += 1
                all_samples.append((os.path.join(tb_path, fname), lbl))
                if lbl == 1: active_used += 1
                else:        latent_used += 1

            print(f"  ✓  TBX11K imgs/tb/: {active_used} Active TB (label=1)"
                f" + {latent_used} Latent TB (label=2)"
                f" [{fallback} fallback to Active]")

        return all_samples

    def _split_samples(
        self,
        all_samples: List[Tuple[str, int]],
        split: str,
        val_ratio: float,
        test_ratio: float,
        seed: int,
    ) -> List[Tuple[str, int]]:
        """Deterministic train/val/test split stratified by class."""
        rng = np.random.default_rng(seed)

        # Group by label
        by_class: dict = {}
        for path, label in all_samples:
            by_class.setdefault(label, []).append((path, label))

        train_s, val_s, test_s = [], [], []
        for label, items in by_class.items():
            items = list(items)
            rng.shuffle(items)
            n     = len(items)
            n_val  = int(n * val_ratio)
            n_test = int(n * test_ratio)
            test_s  += items[:n_test]
            val_s   += items[n_test:n_test + n_val]
            train_s += items[n_test + n_val:]

        return {"train": train_s, "val": val_s, "test": test_s}[split]

    def _load_mock_data(self, n: int = 200):
        self._use_mock = True
        # Simulate class imbalance: Healthy 50%, Active TB 40%, Latent 10%
        dist = [(0, int(n * 0.50)), (1, int(n * 0.40)), (2, int(n * 0.10))]
        for label, count in dist:
            for _ in range(count):
                self.samples.append((f"__mock__{label}", label))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        path, label = self.samples[idx]
        if path.startswith("__mock__"):
            arr   = np.random.randint(0, 256, (224, 224), dtype=np.uint8)
            image = Image.fromarray(arr, mode="L")
        else:
            image = Image.open(path).convert("L")
        if self.transform:
            image = self.transform(image)
        return image, label

    def get_class_counts(self) -> dict:
        counts = {name: 0 for name in CLASS_NAMES}
        for _, label in self.samples:
            if label < len(CLASS_NAMES):
                counts[CLASS_NAMES[label]] += 1
        return counts


class ShenzhenDataset(Dataset):
    """
    Shenzhen dataset loader.

    Reads shenzhen_metadata.csv for accurate labels:
        findings == "normal"  → label 0 (Healthy)
        anything else         → label 1 (Active TB)

    Falls back to filename suffix (_0 = normal, _1 = TB)
    if CSV not found.
    """

    def __init__(
        self,
        root_dir: str,
        transform=None,
        use_mock: bool = False,
    ):
        self.root_dir  = root_dir
        self.transform = transform or get_val_transforms()
        self.samples: List[Tuple[str, int]] = []
        self._use_mock = False

        if use_mock or not os.path.isdir(root_dir):
            print(f"  ⚠  Shenzhen not found at '{root_dir}' — using mock data")
            self._load_mock_data()
        else:
            loaded = self._load_from_csv() or self._load_from_filenames()
            if not loaded:
                print(f"  ⚠  No Shenzhen images found — using mock data")
                self._load_mock_data()
            else:
                print(f"  ✓  Shenzhen: {len(self.samples)} samples loaded")

    def _load_from_csv(self) -> bool:
        """Primary: use shenzhen_metadata.csv for labels."""
        csv_path = os.path.join(self.root_dir, "shenzhen_metadata.csv")
        img_dir  = os.path.join(self.root_dir, "images")

        if not os.path.isfile(csv_path) or not os.path.isdir(img_dir):
            return False

        df = pd.read_csv(csv_path)
        # Expected columns: study_id, sex, age, findings
        if "study_id" not in df.columns or "findings" not in df.columns:
            print(f"  ⚠  CSV missing expected columns, falling back to filenames")
            return False

        normal_count, tb_count, skip_count = 0, 0, 0
        for _, row in df.iterrows():
            fname    = str(row["study_id"]).strip()
            finding  = str(row["findings"]).strip().lower()
            img_path = os.path.join(img_dir, fname)

            if not os.path.isfile(img_path):
                skip_count += 1
                continue

            # "normal" → Healthy(0), anything else → Active TB(1)
            if finding == "normal":
                label = 0
                normal_count += 1
            else:
                label = 1
                tb_count += 1

            self.samples.append((img_path, label))

        print(f"  ✓  Shenzhen CSV: normal={normal_count}, TB={tb_count}"
              f", skipped={skip_count}")
        return len(self.samples) > 0

    def _load_from_filenames(self) -> bool:
        """Fallback: infer label from filename suffix (_0=normal, _1=TB)."""
        img_dir = os.path.join(self.root_dir, "images")
        if not os.path.isdir(img_dir):
            img_dir = self.root_dir

        for fname in sorted(os.listdir(img_dir)):
            if not fname.lower().endswith((".png", ".jpg", ".jpeg")):
                continue
            stem   = os.path.splitext(fname)[0]
            suffix = stem.split("_")[-1]
            if suffix == "0":
                label = 0
            elif suffix == "1":
                label = 1
            else:
                continue
            self.samples.append((os.path.join(img_dir, fname), label))

        return len(self.samples) > 0

    def _load_mock_data(self, n: int = 60):
        self._use_mock = True
        for i in range(n):
            self.samples.append((f"__mock__{i % 2}", i % 2))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        path, label = self.samples[idx]
        if path.startswith("__mock__"):
            arr   = np.random.randint(0, 256, (224, 224), dtype=np.uint8)
            image = Image.fromarray(arr, mode="L")
        else:
            image = Image.open(path).convert("L")
        if self.transform:
            image = self.transform(image)
        return image, label