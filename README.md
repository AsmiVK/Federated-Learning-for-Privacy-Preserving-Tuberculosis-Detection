# 🫁 FedTB — Federated Learning for Tuberculosis Detection

> Privacy-preserving multi-institutional AI for TB screening using chest X-rays — no patient data ever leaves the hospital.

[![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2.2-orange?logo=pytorch)](https://pytorch.org)
[![Flower](https://img.shields.io/badge/Flower-1.8.0-green)](https://flower.ai)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.35.0-red?logo=streamlit)](https://streamlit.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

---

## 📋 Table of Contents
- [Overview](#-overview)
- [Key Results](#-key-results)
- [Architecture](#-architecture)
- [Project Structure](#-project-structure)
- [Setup](#-setup)
- [Datasets](#-datasets)
- [Running the Project](#-running-the-project)
- [Dashboard](#-dashboard)
- [Experimental Findings](#-experimental-findings)
- [Team](#-team)
- [References](#-references)

---

## 🔍 Overview

Tuberculosis affects 10+ million people annually worldwide. AI-based chest X-ray analysis can automate TB screening — but hospitals **cannot legally share patient data** (HIPAA, GDPR). This project solves that problem using **Federated Learning (FL)**: a distributed training paradigm where only model weights are shared, never patient images.

### The Problem
- Centralized AI requires pooling all hospital data → privacy violation
- Each hospital alone has insufficient data to train a robust model
- Regional TB patterns vary dramatically (India: 2.5% MDR-TB, Eastern Europe: 35% MDR-TB)

### Our Solution
- **Federated Learning**: 3 hospital nodes train collaboratively without sharing X-rays
- **FedProx**: Proximal regularization prevents client drift on Non-IID data
- **ResNet-18**: Transfer learning from ImageNet for efficient TB-specific fine-tuning
- **3-class detection**: Healthy | Active TB | Latent TB

---

## 📊 Key Results

| Method | Accuracy | F1 (Weighted) | Healthy Sensitivity | Active TB Sensitivity |
|--------|----------|---------------|--------------------|-----------------------|
| FedAvg (baseline) | 0.016 | 0.008 | 0.004 | 0.000 |
| **FedProx (ours)** | **0.726** | **0.767** | **0.787** | **0.163** |

> **FedAvg collapses on Non-IID data** due to client drift. FedProx's proximal regularization term prevents this, achieving 72.6% accuracy across 3 geographically distinct hospital nodes — with **zero patient data sharing**.

Training configuration: 20 communication rounds · 2 local epochs · balanced mini-batch sampling · LR=0.0003 · μ=0.001

---

## 🏗️ Architecture

### Model: ResNet-18 (Transfer Learning)
```
Input: 224×224 Chest X-Ray (Grayscale → 3ch RGB)
    ↓
Conv1 + BN + ReLU + MaxPool  [FROZEN — ImageNet generic features]
    ↓
Layer 1 (64ch,  2 blocks)    [FROZEN]
    ↓
Layer 2 (128ch, 2 blocks)    [FROZEN]
    ↓
Layer 3 (256ch, 2 blocks)    [TRAINABLE — lung structures]
    ↓
Layer 4 (512ch, 2 blocks)    [TRAINABLE — TB-specific: cavities, opacities]
    ↓
Global Average Pooling
    ↓
FC: 512 → 3                  [TRAINABLE — output head]
    ↓
Output: [Healthy | Active TB | Latent TB]
```
**Parameters**: 11,178,051 total · 10,494,979 trainable · 683,072 frozen

### Federated Setup: Non-IID Hospital Nodes
```
                    ┌─────────────────────────────┐
                    │      Central Server          │
                    │  (Flower + FedProx/FedAvg)  │
                    │   Aggregates weights only    │
                    └──────┬──────────────┬───────┘
                           │              │
              ┌────────────┘              └────────────┐
              ▼                                        ▼
   ┌─────────────────────┐              ┌─────────────────────┐
   │  Node 1 — India     │              │ Node 2 — S. Africa  │
   │  2,329 images       │              │  2,140 images        │
   │  Healthy: 91.4%     │              │  Healthy: 87.0%      │
   │  Active TB: 7.3%    │              │  Active TB: 11.4%    │
   │  Latent TB: 1.2%    │              │  Latent TB: 1.6%     │
   └─────────────────────┘              └─────────────────────┘
              ▲
              │
   ┌─────────────────────┐
   │  Node 3 — E. Europe │
   │  1,413 + 662 Shen.  │
   │  Healthy: 94.1%     │
   │  Active TB: 5.2%    │   ← Domain shift: Shenzhen dataset
   │  Latent TB: 0.7%    │     (different X-ray equipment)
   └─────────────────────┘
```

### FedProx Loss Function
```
L_client(w) = L_data(w) + (μ/2) × ||w - w_global||²
              ───────────   ────────────────────────
              Cross-entropy    Proximal term (μ=0.001)
              loss             prevents client drift
```

---

## 📁 Project Structure

```
FedTB/
├── config/
│   └── config.yaml              # All hyperparameters
├── src/
│   ├── data/
│   │   ├── preprocessing.py     # Image transforms (augmentation, normalization)
│   │   ├── dataset.py           # TBX11K + Shenzhen dataset classes (COCO JSON parsing)
│   │   └── federated_split.py   # Non-IID partition across 3 nodes
│   ├── model/
│   │   ├── resnet_tb.py         # ResNet-18 modified for 3-class TB detection
│   │   └── model_utils.py       # Save/load/get/set weights (Flower compatible)
│   ├── federated/
│   │   ├── client.py            # Flower FL client (one per hospital node)
│   │   ├── strategies.py        # FedAvg and FedProx strategy classes
│   │   └── server.py            # FL simulation runner
│   └── training/
│       └── metrics.py           # F1, AUC-ROC, Kappa, Sensitivity, Specificity
├── app/
│   └── dashboard.py             # Streamlit 4-tab interactive dashboard
├── scripts/
│   ├── run_training.py          # Main training script (CLI)
│   └── verify_env.py            # Environment verification
├── tests/
│   ├── test_model.py
│   ├── test_dataset.py
│   └── test_federated.py
├── requirements.txt
└── README.md
```

---

## ⚙️ Setup

### Prerequisites
- Python 3.11
- Windows / Linux / macOS
- No GPU required (CPU training supported)

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/YOUR_USERNAME/FedTB.git
cd FedTB

# 2. Create virtual environment
python -m venv venv

# Windows
.\venv\Scripts\Activate.ps1

# Linux/Mac
source venv/bin/activate

# 3. Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

### Verify Installation

```bash
python scripts/verify_env.py
```

Expected output:
```
✅  CHECKPOINT 0 PASSED — Environment is ready!
```

---

## 📦 Datasets

Download both datasets from Kaggle and place them as shown:

### TBX11K Dataset
**Download:** https://www.kaggle.com/datasets/usmanshams/tbx-11

```
data/raw/TBX11K/
├── imgs/
│   ├── health/     ← 3,800 healthy X-rays
│   ├── sick/       ← 3,800 sick non-TB X-rays
│   └── tb/         ← 800 TB X-rays (active + latent)
└── annotations/
    └── json/
        ├── TBX11K_train.json   ← COCO format (3 TB categories)
        └── TBX11K_val.json
```

### Shenzhen Dataset
**Download:** https://www.kaggle.com/datasets/raddar/tuberculosis-chest-xrays-shenzhen

```
data/raw/Shenzhen/
├── images/                  ← 662 chest X-rays
└── shenzhen_metadata.csv    ← Labels (findings column)
```

> ⚠️ If datasets are not found, the system automatically falls back to synthetic mock data for pipeline testing.

---

## 🚀 Running the Project

### Run Tests First
```bash
python tests/test_model.py       # Verify ResNet-18 architecture
python tests/test_dataset.py     # Verify data pipeline
python tests/test_federated.py   # Verify FL engine (2-round mock)
```

### Training

```bash
# Quick test with mock data (2 rounds, ~5 mins)
python scripts/run_training.py --rounds 2 --mock --strategy fedprox

# Single strategy on real data
python scripts/run_training.py --rounds 20 --epochs 2 --strategy fedprox

# Compare both strategies (recommended — produces convergence curves)
python scripts/run_training.py --rounds 20 --epochs 2 --compare
```

### Training Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--rounds` | 5 | Number of FL communication rounds |
| `--epochs` | 5 | Local training epochs per round |
| `--strategy` | fedprox | `fedavg` or `fedprox` |
| `--compare` | False | Run both strategies sequentially |
| `--lr` | 0.0003 | Learning rate |
| `--mu` | 0.001 | FedProx proximal coefficient |
| `--batch` | 32 | Batch size |
| `--mock` | False | Use synthetic data (no download needed) |

### Output Files
```
checkpoints/
└── global_model_fedprox_r20.pth   ← Saved model weights

runs/
├── results_fedprox.json           ← Round-by-round metrics
├── results_fedavg.json
├── confusion_fedprox.png          ← Confusion matrix plots
├── confusion_fedavg.png
└── convergence.png                ← FedAvg vs FedProx curves
```

---

## 🖥️ Dashboard

```bash
streamlit run app/dashboard.py
```

Opens at **http://localhost:8501**

### Tabs

| Tab | Description |
|-----|-------------|
| 🏠 Overview | Project summary, node distribution, architecture |
| 🚀 Run Training | Launch FL experiments from UI with sliders |
| 📊 Results | Live metrics, convergence curves, confusion matrices |
| 🔬 Predict | Upload chest X-ray → get TB classification |

---

## 🔬 Experimental Findings

### What We Tried (Full Iteration History)

| # | Configuration | FedProx Accuracy | F1 | Key Finding |
|---|--------------|------------------|----|-------------|
| 1 | Naive split (R5, E5) | 0.314 | 0.451 | Minority class starvation at nodes 2&3 |
| 2 | Fixed split, no weights (R5, E5) | 0.038 | 0.013 | Model predicted all TB |
| 3 | Raw inverse-frequency weights (R10, E2) | 0.501* | 0.623* | Validation loss explosion (0.9→11.7) |
| 4 | Sqrt weights + cap (R20, E2) | 0.353 | 0.483 | Per-node weight conflict in aggregation |
| 5 | Fixed weights [0.5, 2.0, 4.0] (R20, E1) | 0.084 | 0.016 | Predicted all Active TB |
| 6 | **Balanced sampler, uniform loss (R20, E2)** | **0.726** | **0.767** | **Best result** ✅ |

\* FedAvg result, FedProx collapsed in this configuration

### Key Lessons Learned

1. **Federated partition matters critically**: Naive fraction-based splits exhaust minority classes at primary nodes. Per-class independent distribution is essential.

2. **Class-weighted loss fails in FL**: Different per-node weights create conflicting gradients that FedAvg cannot reconcile. Balanced sampling (WeightedRandomSampler) with uniform loss is the correct approach.

3. **FedAvg collapses on Non-IID data**: Client drift causes each node to overfit its local distribution under balanced sampling. The aggregated model inherits this bias and collapses to single-class prediction. FedProx's proximal term is essential.

4. **Train accuracy ≠ test performance in FL**: FedAvg achieved 96.2% train accuracy but 1.6% test accuracy. FedProx achieved 95.9% train accuracy and 72.6% test accuracy. Validation loss is a better convergence indicator.

5. **AUC-ROC interpretation with imbalanced test sets**: FedProx AUC-ROC is 0.479 (below random) because balanced training miscalibrates softmax probabilities relative to the 90.4% Healthy test distribution. Hard-decision accuracy (72.6%) is more meaningful here.

---

## 👥 Team

| Name | SRN | Section |
|------|-----|---------|
| Anvita Agarwal | PES2UG23CS068 | B |
| Asmi Vishal Kapadnis | PES2UG23CS100 | B |

**PES University EC Campus, Bengaluru**
Topics in Deep Learning Mini Project

---

## 📚 References

1. Liu et al., "Rethinking Computer-Aided Tuberculosis Diagnosis," CVPR 2020
2. Jaeger et al., "Two public chest X-ray datasets," Quant. Imag. Med. Surg., 2014
3. McMahan et al., "Communication-Efficient Learning of Deep Networks from Decentralized Data," AISTATS 2017 — **FedAvg**
4. Li et al., "Federated Optimization in Heterogeneous Networks," MLSys 2020 — **FedProx**
5. Xu et al., "Federated Learning for Healthcare Informatics," J. Healthc. Inform. Res., 2021
6. Lakhani & Sundaram, "Deep Learning at Chest Radiography," Radiology, 2017
7. Kaissis et al., "Secure, privacy-preserving and federated ML in medical imaging," Nature MI, 2020
8. Beutel et al., "Flower: A Friendly Federated Learning Framework," arXiv:2007.14390, 2020
9. He et al., "Deep Residual Learning for Image Recognition," CVPR 2016
10. Raghu et al., "Transfusion: Understanding Transfer Learning for Medical Imaging," NeurIPS 2019
11. Zhao et al., "Federated Learning with Non-IID Data," arXiv:1806.00582, 2018
12. WHO, Global Tuberculosis Report 2023

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

---

<p align="center">
  Built with PyTorch · Flower · Streamlit · ❤️
</p>
