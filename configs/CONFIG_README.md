# Dataset Configuration Reference

Complete configuration details for all datasets and models.

---

## 📊 Configuration Overview

| Config Name           | Model     | 2D Input    | 3D Output     | Batch | Params  | GPU Mem |
|-----------------------|-----------|-------------|---------------|-------|---------|---------|
| `swinx2s_femur`       | SwinX2S   | 128×128     | 128×128×128   | 4     | 32.5M   | ~8 GB   |
| `swinx2s_hip`         | SwinX2S   | 288×288     | 160×288×288   | 2     | 130M    | ~24 GB  |
| `swinx2s_rib`         | SwinX2S   | 320×320     | 160×320×320   | 1     | 314M    | ~40 GB  |
| `swinx2s_verse19`     | SwinX2S   | 128×128     | 96×128×128    | 2     | 48M     | ~16 GB  |
| `swinx2s_verse20`     | SwinX2S   | 128×128     | 96×128×128    | 2     | 48M     | ~16 GB  |
| `swinx2s_ctpelvic`    | SwinX2S   | 256×256     | 96×256×256    | 1     | 32M     | ~40 GB  |
| `transvert_femur`     | TransVert | 128×128     | 128×128×128   | 4     | 0.56M   | ~6 GB   |
| `transvert_hip`       | TransVert | 256×256     | 128×256×256   | 2     | 0.56M   | ~10 GB  |
| `transvert_rib`       | TransVert | 256×256     | 128×256×256   | 2     | 0.56M   | ~10 GB  |
| `transvert_verse19`   | TransVert | 96×96       | 96×96×96      | 4     | 0.56M   | ~4 GB   |
| `transvert_verse20`   | TransVert | 96×96       | 96×96×96      | 4     | 0.56M   | ~4 GB   |
| `transvert_ctpelvic`  | TransVert | 256×256     | 96×256×256    | 1     | 0.56M   | ~50 GB  |

---

## 🔧 Detailed Configurations

### 1. Femur Dataset

**Dataset Info:**
- Samples: 931
- Original Volume: 85×85×85
- Source: TotalSegmentator

#### SwinX2S Configuration
```python
Config: swinx2s_femur
Model: Swin-X2S-Tiny

# Input/Output
2D Input Size:    128×128 (AP + Lateral X-rays)
3D Output Size:   128×128×128

# Architecture
Feature Size:     32
Depths:           (2, 2, 6, 2)
Num Heads:        (1, 2, 4, 8)
Window Size:      (6, 6)
Fusion Depths:    (2, 2, 2, 2, 2, 2)

# Training
Batch Size:       4
Learning Rate:    1e-4
Epochs:           150 (recommended)
Parameters:       32.51M
GPU Memory:       ~8 GB
```

#### TransVert Configuration
```python
Config: transvert_femur
Model: TransVert

# Input/Output
2D Input Size:    128×128 (AP + Lateral X-rays)
3D Output Size:   128×128×128

# Architecture
Mid Channels:     32
Intermediate Depth: 32

# Training
Batch Size:       4
Learning Rate:    1e-4
Epochs:           150 (recommended)
Parameters:       0.56M
GPU Memory:       ~6 GB
```

---

### 2. Hip Dataset

**Dataset Info:**
- Samples: ~800
- Original Volume: 191-192³
- Source: TotalSegmentator

#### SwinX2S Configuration
```python
Config: swinx2s_hip
Model: Swin-X2S-Small

# Input/Output
2D Input Size:    288×288
3D Output Size:   160×288×288

# Architecture
Feature Size:     64
Depths:           (2, 2, 6, 2)
Num Heads:        (2, 4, 8, 16)
Window Size:      (8, 8)
Fusion Depths:    (2, 2, 2, 2, 2, 2)

# Training
Batch Size:       2
Learning Rate:    1e-4
Epochs:           150 (recommended)
Parameters:       129.97M
GPU Memory:       ~24 GB
```

#### TransVert Configuration
```python
Config: transvert_hip
Model: TransVert

# Input/Output
2D Input Size:    256×256
3D Output Size:   128×256×256

# Architecture
Mid Channels:     32
Intermediate Depth: 32

# Training
Batch Size:       2
Learning Rate:    1e-4
Epochs:           150 (recommended)
Parameters:       0.56M
GPU Memory:       ~10 GB
```

---

### 3. Rib Dataset

**Dataset Info:**
- Samples: ~700
- Original Volume: 213×213×213
- Source: TotalSegmentator

#### SwinX2S Configuration
```python
Config: swinx2s_rib
Model: Swin-X2S-Base

# Input/Output
2D Input Size:    320×320
3D Output Size:   160×320×320

# Architecture
Feature Size:     96
Depths:           (2, 2, 18, 2)
Num Heads:        (3, 6, 12, 24)
Window Size:      (8, 8)
Fusion Depths:    (2, 2, 2, 2, 2, 2)

# Training
Batch Size:       1
Learning Rate:    1e-4
Epochs:           150 (recommended)
Parameters:       313.68M
GPU Memory:       ~40 GB
```

#### TransVert Configuration
```python
Config: transvert_rib
Model: TransVert

# Input/Output
2D Input Size:    256×256
3D Output Size:   128×256×256

# Architecture
Mid Channels:     48
Intermediate Depth: 32

# Training
Batch Size:       2
Learning Rate:    1e-4
Epochs:           150 (recommended)
Parameters:       0.56M
GPU Memory:       ~10 GB
```

---

### 4. Verse19 Dataset

**Dataset Info:**
- Samples: 1168 (largest)
- Original Volume: 96×96×31-48 (variable, anisotropic)
- Source: VerSe Challenge 2019

#### SwinX2S Configuration
```python
Config: swinx2s_verse19
Model: Swin-X2S-Small

# Input/Output
2D Input Size:    128×128
3D Output Size:   96×128×128

# Architecture
Feature Size:     48
Depths:           (2, 2, 2, 2)
Num Heads:        (2, 4, 6, 8)
Window Size:      (6, 6)
Fusion Depths:    (2, 2, 2, 2, 2, 2)

# Training
Batch Size:       2
Learning Rate:    1e-4
Epochs:           150 (recommended)
Parameters:       ~48M
GPU Memory:       ~16 GB
```

#### TransVert Configuration
```python
Config: transvert_verse19
Model: TransVert

# Input/Output
2D Input Size:    96×96
3D Output Size:   96×96×96

# Architecture
Mid Channels:     32
Intermediate Depth: 24

# Training
Batch Size:       4
Learning Rate:    1e-4
Epochs:           150 (recommended)
Parameters:       0.56M
GPU Memory:       ~4 GB
```

---

### 5. Verse20 Dataset

**Dataset Info:**
- Samples: 133 (smallest)
- Original Volume: 104-147³ (variable, anisotropic)
- Source: VerSe Challenge 2020

#### SwinX2S Configuration
```python
Config: swinx2s_verse20
Model: Swin-X2S-Tiny

# Input/Output
2D Input Size:    128×128
3D Output Size:   96×128×128

# Architecture
Feature Size:     48
Depths:           (2, 2, 2, 2)
Num Heads:        (2, 4, 6, 8)
Window Size:      (6, 6)
Fusion Depths:    (2, 2, 2, 2, 2, 2)

# Training
Batch Size:       4
Learning Rate:    1e-4
Epochs:           100 (recommended - small dataset)
Parameters:       ~48M
GPU Memory:       ~8 GB
```

#### TransVert Configuration
```python
Config: transvert_verse20
Model: TransVert

# Input/Output
2D Input Size:    96×96
3D Output Size:   96×96×96

# Architecture
Mid Channels:     32
Intermediate Depth: 24

# Training
Batch Size:       4
Learning Rate:    1e-4
Epochs:           100 (recommended - small dataset)
Parameters:       0.56M
GPU Memory:       ~4 GB
```

---

### 6. CTPelvic Dataset

**Dataset Info:**
- Samples: ~200
- Original Volume: 294-546³ (highly variable, largest volumes)
- Source: CTPelvic1K

#### SwinX2S Configuration
```python
Config: swinx2s_ctpelvic
Model: Swin-X2S-Tiny

# Input/Output
2D Input Size:    256×256
3D Output Size:   96×256×256

# Architecture
Feature Size:     32
Depths:           (2, 2, 2, 2)
Num Heads:        (2, 4, 6, 8)
Window Size:      (6, 6)
Fusion Depths:    (2, 2, 2, 2, 2, 2)

# Training
Batch Size:       1 (required)
Learning Rate:    1e-4
Epochs:           150 (recommended)
Parameters:       ~32M
GPU Memory:       ~40 GB (requires A100 80GB)
```

#### TransVert Configuration
```python
Config: transvert_ctpelvic
Model: TransVert

# Input/Output
2D Input Size:    256×256
3D Output Size:   96×256×256

# Architecture
Mid Channels:     24
Intermediate Depth: 24

# Training
Batch Size:       1 (required)
Learning Rate:    1e-4
Epochs:           150 (recommended)
Parameters:       0.56M
GPU Memory:       ~50 GB (requires A100 80GB)
```

---

## 🎯 Common Configuration Parameters

All configurations share these base parameters:

```python
# Loss Function
BCE Weight:       0.5
Dice Weight:      0.5

# Optimizer
Optimizer:        Adam
Learning Rate:    1e-4
Gradient Clip:    1.0

# Scheduler
Type:             ReduceLROnPlateau
Factor:           0.5
Patience:         5

# Early Stopping
Patience:         10 epochs

# Data
Random Seed:      42
Target Key:       "vol"
Binary Threshold: 0.5

# Training
Mixed Precision:  FP16 (automatic on CUDA)
```

---

## 📁 Data Directory Structure

Expected directory structure for each dataset:

```
datasets/datasets/
├── Total Segmentator/
│   ├── Femur/
│   │   ├── train/
│   │   ├── val/
│   │   └── test/
│   ├── Hip/
│   │   ├── train/
│   │   ├── val/
│   │   └── test/
│   └── Rib/
│       ├── train/
│       ├── val/
│       └── test/
├── Verse/
│   ├── verse19/
│   │   ├── train/
│   │   ├── val/
│   │   └── test/
│   └── verse20/
│       ├── train/
│       ├── val/
│       └── test/
└── ctpelvik/
    ├── train/
    ├── val/
    └── test/
```

Each split folder contains:
- `xa/` - AP X-ray projections (.npy)
- `xb/` - Lateral X-ray projections (.npy)
- `vol/` - 3D ground truth volumes (.npy)

---

## 🚀 Usage Examples

### List All Configurations
```bash
python train.py --list_configs
```

### Train with Specific Configuration
```bash
# SwinX2S on Femur
python train.py --config swinx2s_femur --epochs 150 --batch_size 4

# TransVert on Hip
python train.py --config transvert_hip --epochs 150 --batch_size 2

# With custom settings
python train.py --config swinx2s_verse19 --epochs 100 --batch_size 2 --no_wandb
```

### Override Configuration Parameters
```bash
# Change batch size
python train.py --config transvert_femur --batch_size 2

# Change epochs
python train.py --config swinx2s_hip --epochs 200

# Disable W&B logging
python train.py --config transvert_verse19 --no_wandb
```

---

## 💾 Output Locations

### Checkpoints
```
saved_models/
├── swinx2s/
│   ├── tiny/
│   │   └── femur/
│   │       ├── best_epoch045_dice0.9100_loss0.0432.pth
│   │       └── best_model_weights.pth
│   ├── small/
│   └── base/
└── transvert/
    ├── femur/
    │   ├── best_epoch056_dice0.8292_loss0.1410.pth
    │   └── best_model_weights.pth
    └── hip/
```

### Training Logs
```
saved_models/
├── transvert_femur_training.log
├── swinx2s_femur_training.log
└── ...
```

---

## 📊 Expected Performance

| Dataset  | TransVert Dice | SwinX2S Dice | Training Time (TransVert) | Training Time (SwinX2S) |
|----------|----------------|--------------|---------------------------|-------------------------|
| Femur    | 0.82-0.83      | 0.90-0.91    | 1.5 hours                 | 6 hours                 |
| Hip      | 0.80-0.82      | 0.88-0.90    | 2 hours                   | 8 hours                 |
| Rib      | 0.78-0.80      | 0.86-0.88    | 2 hours                   | 10 hours                |
| Verse19  | 0.80-0.82      | 0.88-0.90    | 2 hours                   | 8 hours                 |
| Verse20  | 0.75-0.78      | 0.82-0.85    | 30 min                    | 3 hours                 |
| CTPelvic | 0.75-0.78      | 0.82-0.85    | 4 hours                   | 16 hours                |

*Times based on NVIDIA A100 80GB GPU*

---

## 🔍 Configuration Selection Guide

### Choose SwinX2S when:
- Maximum accuracy is required
- GPU memory is available (16GB+)
- Training time is not critical
- Clinical metrics are important

### Choose TransVert when:
- Fast training is needed
- Limited GPU memory (<16GB)
- Quick prototyping/debugging
- Efficiency is prioritized over accuracy

### Model Size Selection (SwinX2S):
- **Tiny**: Small datasets (<500 samples), limited GPU
- **Small**: Best balance, recommended for most cases
- **Base**: Maximum accuracy, requires 24GB+ GPU

---

## 📝 Notes

- All configurations use early stopping (patience=10)
- Mixed precision (FP16) is enabled automatically on CUDA
- Clinical metrics are computed only for Femur dataset
- CTPelvic requires A100 80GB GPU due to large volume sizes
- Verse datasets have variable volume sizes (anisotropic)
