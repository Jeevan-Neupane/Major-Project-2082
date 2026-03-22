# X-Ray to 3D Bone Reconstruction

Deep learning pipeline for reconstructing 3D bone volumes from 2D X-ray images.  
Two orthogonal views (AP + Lateral) are fused to predict a full 3D binary segmentation mask.

## Models

| Model         | Type                                               | Parameters | Strengths         |
| ------------- | -------------------------------------------------- | ---------- | ----------------- |
| **SwinX2S**   | Swin Transformer + Cross-Attention + UNETR Decoder | 32M–558M   | Highest accuracy  |
| **TransVert** | CNN Encoder + Volumetric Expansion + CNN Decoder   | ~0.5M      | Lightweight, fast |

## Datasets

| Dataset      | Source           | Volumes | Volume Size | X-ray Size |
| ------------ | ---------------- | ------- | ----------- | ---------- |
| **Femur**    | TotalSegmentator | 931     | 85³         | 128×128    |
| **Hip**      | TotalSegmentator | ~800    | 191–192³    | 288×288    |
| **Rib**      | TotalSegmentator | ~700    | 213³        | 320×320    |
| **Verse19**  | VerSe Challenge  | 1168    | 96×96×31–48 | 96×96      |
| **Verse20**  | VerSe Challenge  | 133     | 104–147³    | 96×96      |
| **CTPelvic** | CTPelvic1K       | ~200    | 294–546³    | 288×288    |

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. List available configs
python train.py --list_configs

# 3. Train (example)
python train.py --config swinx2s_femur --model_size small --epochs 150
```

## Project Structure

```
├── train.py                 # Main training entry point
├── test_hip_metrics.py      # Hip clinical metrics testing
├── requirements.txt         # Python dependencies
├── configs/
│   ├── config.py            # All dataset/model configurations
│   └── args_parser.py       # CLI argument parsing
├── models/
│   ├── swinx2s.py           # SwinX2S model
│   ├── transvert.py         # TransVert model (self-contained)
│   └── swinX2S/             # Swin Transformer components
│       ├── swin_transformer.py
│       ├── transfusion.py
│       └── unetr_blocks.py
├── utils/
│   ├── data/                # Data loading
│   │   └── dataset.py       # BoneDataset & collate_bone
│   ├── training/            # Training infrastructure
│   │   ├── trainer.py       # Train/validate epoch loops
│   │   ├── losses.py        # DiceBCE loss
│   │   ├── metrics.py       # Dice score computation
│   │   ├── checkpoint.py    # Model checkpoint management
│   │   └── logger.py        # Training logger (file + CSV)
│   └── clinical/            # Clinical metric evaluation
│       ├── femur_metrics.py  # Diameter, NSA, length
│       ├── hip_metrics.py    # Pelvic width, acetabular diameter/depth
│       └── verse_metrics.py  # Height, width, depth
├── logs/                    # Training logs (.log + .csv + _clinical.csv)
├── saved_models/            # Best checkpoints per config
└── datasets/datasets/       # Data root
    ├── Total Segmentator/   (Femur/, Hip/, Rib/)
    ├── Verse/               (verse19/, verse20/)
    └── ctpelvik/
```

---

## Training Commands — All Datasets

> **Flags:** `--no_wandb` disables W&B logging. `--model_size` applies only to SwinX2S.  
> Logs are saved automatically to `logs/<config>_<timestamp>.log` and `.csv`.

### Femur

| Model     | Size      | Command                                                                                 | Batch | GPU Mem |
| --------- | --------- | --------------------------------------------------------------------------------------- | ----- | ------- |
| SwinX2S   | Tiny      | `python train.py --config swinx2s_femur --model_size tiny --epochs 100 --batch_size 4`  | 4     | ~8 GB   |
| SwinX2S   | **Small** | `python train.py --config swinx2s_femur --model_size small --epochs 150 --batch_size 4` | 4     | ~16 GB  |
| SwinX2S   | Base      | `python train.py --config swinx2s_femur --model_size base --epochs 200 --batch_size 2`  | 2     | ~24 GB  |
| TransVert | —         | `python train.py --config transvert_femur --epochs 150 --batch_size 4`                  | 4     | ~6 GB   |

### Hip

| Model     | Size      | Command                                                                               | Batch | GPU Mem |
| --------- | --------- | ------------------------------------------------------------------------------------- | ----- | ------- |
| SwinX2S   | Tiny      | `python train.py --config swinx2s_hip --model_size tiny --epochs 100 --batch_size 2`  | 2     | ~12 GB  |
| SwinX2S   | **Small** | `python train.py --config swinx2s_hip --model_size small --epochs 150 --batch_size 2` | 2     | ~24 GB  |
| SwinX2S   | Base      | `python train.py --config swinx2s_hip --model_size base --epochs 200 --batch_size 1`  | 1     | ~40 GB  |
| TransVert | —         | `python train.py --config transvert_hip --epochs 150 --batch_size 2`                  | 2     | ~10 GB  |

### Rib

| Model     | Size      | Command                                                                               | Batch | GPU Mem |
| --------- | --------- | ------------------------------------------------------------------------------------- | ----- | ------- |
| SwinX2S   | Tiny      | `python train.py --config swinx2s_rib --model_size tiny --epochs 100 --batch_size 2`  | 2     | ~14 GB  |
| SwinX2S   | **Small** | `python train.py --config swinx2s_rib --model_size small --epochs 150 --batch_size 1` | 1     | ~24 GB  |
| SwinX2S   | Base      | `python train.py --config swinx2s_rib --model_size base --epochs 200 --batch_size 1`  | 1     | ~40 GB  |
| TransVert | —         | `python train.py --config transvert_rib --epochs 150 --batch_size 2`                  | 2     | ~10 GB  |

### Verse19

| Model     | Size      | Command                                                                                   | Batch | GPU Mem |
| --------- | --------- | ----------------------------------------------------------------------------------------- | ----- | ------- |
| SwinX2S   | Tiny      | `python train.py --config swinx2s_verse19 --model_size tiny --epochs 100 --batch_size 4`  | 4     | ~8 GB   |
| SwinX2S   | **Small** | `python train.py --config swinx2s_verse19 --model_size small --epochs 150 --batch_size 2` | 2     | ~16 GB  |
| SwinX2S   | Base      | `python train.py --config swinx2s_verse19 --model_size base --epochs 200 --batch_size 1`  | 1     | ~24 GB  |
| TransVert | —         | `python train.py --config transvert_verse19 --epochs 150 --batch_size 4`                  | 4     | ~4 GB   |

### Verse20

| Model     | Size      | Command                                                                                   | Batch | GPU Mem |
| --------- | --------- | ----------------------------------------------------------------------------------------- | ----- | ------- |
| SwinX2S   | Tiny      | `python train.py --config swinx2s_verse20 --model_size tiny --epochs 100 --batch_size 4`  | 4     | ~8 GB   |
| SwinX2S   | **Small** | `python train.py --config swinx2s_verse20 --model_size small --epochs 150 --batch_size 2` | 2     | ~16 GB  |
| SwinX2S   | Base      | `python train.py --config swinx2s_verse20 --model_size base --epochs 200 --batch_size 1`  | 1     | ~24 GB  |
| TransVert | —         | `python train.py --config transvert_verse20 --epochs 150 --batch_size 4`                  | 4     | ~4 GB   |

### CTPelvic

| Model     | Size      | Command                                                                                    | Batch | GPU Mem |
| --------- | --------- | ------------------------------------------------------------------------------------------ | ----- | ------- |
| SwinX2S   | Tiny      | `python train.py --config swinx2s_ctpelvic --model_size tiny --epochs 100 --batch_size 1`  | 1     | ~40 GB  |
| SwinX2S   | **Small** | `python train.py --config swinx2s_ctpelvic --model_size small --epochs 150 --batch_size 1` | 1     | ~60 GB  |
| SwinX2S   | Base      | `python train.py --config swinx2s_ctpelvic --model_size base --epochs 200 --batch_size 1`  | 1     | ~80 GB  |
| TransVert | —         | `python train.py --config transvert_ctpelvic --epochs 150 --batch_size 1`                  | 1     | ~50 GB  |

> **Note:** CTPelvic volumes (up to 546³) require A100 GPUs. All CTPelvic runs use batch_size=1.

---

## Optimal Recommended Commands

If you only want to run **one command per dataset** (best accuracy/speed trade-off), use **SwinX2S Small**:

```bash
# Femur
python train.py --config swinx2s_femur --model_size small --epochs 150 --batch_size 4

# Hip
python train.py --config swinx2s_hip --model_size small --epochs 150 --batch_size 2

# Rib
python train.py --config swinx2s_rib --model_size small --epochs 150 --batch_size 1

# Verse19
python train.py --config swinx2s_verse19 --model_size small --epochs 150 --batch_size 2

# Verse20
python train.py --config swinx2s_verse20 --model_size small --epochs 150 --batch_size 2

# CTPelvic (requires A100 60GB+)
python train.py --config swinx2s_ctpelvic --model_size small --epochs 150 --batch_size 1
```

---

## Training Features

- **Early Stopping** — halts if Dice score doesn't improve for 10 epochs
- **Clinical Metrics** — per-dataset anatomical measurements tracked each epoch
- **Best Model Selection** — saves checkpoint with highest validation Dice score
- **Structured Logging** — every run produces a `.log` (human-readable) and `.csv` (machine-readable) in `logs/`
- **W&B Integration** — optional experiment tracking (disable with `--no_wandb`)
- **Mixed Precision** — automatic FP16 on CUDA for faster training and lower memory

## Log Files

Each training run creates up to three files in `logs/`:

| File                                | Format | Content                                                         |
| ----------------------------------- | ------ | --------------------------------------------------------------- |
| `<config>_<timestamp>.log`          | Text   | Epoch-by-epoch metrics, config summary, per-sample examples     |
| `<config>_<timestamp>.csv`          | CSV    | All numeric metrics per epoch for plotting/analysis             |
| `<config>_<timestamp>_clinical.csv` | CSV    | Per-sample clinical measurements (pred / gt / error) each epoch |

Example log output:

```
2026-03-04 07:15:23 | [Epoch 1/150] Train Loss: 0.4321 (BCE 0.2100 + Dice 0.2221) | Val Loss: 0.3987 (BCE 0.1900 + Dice 0.2087) | Val Dice: 0.7654 | LR: 1.00e-04 | Patience: 0/10 | Time: 45.2s
2026-03-04 07:15:23 |   ** New best Dice: 0.7654 (epoch 1, loss 0.3987)
2026-03-04 07:15:23 |   Clinical (femur): Diameter MAE: 3.21 | Nsa MAE: 4.56 | Length MAE: 6.78
2026-03-04 07:15:23 |     Sample p_0001: diameter: pred=42.50 gt=44.10 err=1.60 | nsa: pred=128.30 gt=130.00 err=1.70
```

## Saved Models

```
saved_models/
├── swinx2s/
│   ├── small/
│   │   ├── femur/
│   │   │   ├── best_epoch045_dice0.9100_loss0.0432.pth
│   │   │   └── best_model_weights.pth
│   │   ├── hip/
│   │   └── ...
│   ├── tiny/
│   └── base/
└── transvert/
    ├── femur/
    ├── hip/
    └── ...
```

## Clinical Metrics

| Dataset           | Metric 1              | Metric 2                 | Metric 3              |
| ----------------- | --------------------- | ------------------------ | --------------------- |
| Femur             | Head Diameter (mm)    | Neck-Shaft Angle (°)     | Length (mm)           |
| Hip / CTPelvic    | Pelvic Width (mm)     | Acetabular Diameter (mm) | Acetabular Depth (mm) |
| Verse19 / Verse20 | Vertebral Height (mm) | Width (mm)               | Depth (mm)            |

### Testing Hip Metrics

```bash
python test_hip_metrics.py "path/to/hip_mask.nii.gz"
```

## Troubleshooting

**Out of Memory?** Reduce `--batch_size` or use a smaller `--model_size`.

```bash
# Try tiny model with batch 1
python train.py --config swinx2s_hip --model_size tiny --epochs 100 --batch_size 1
```

**No GPU?** Training will automatically fall back to CPU (very slow for production use).

**W&B errors?** Use `--no_wandb` to skip, or run `wandb login --relogin`.

## Hardware Requirements

| Model Size     | Min GPU Memory | Recommended GPU  |
| -------------- | -------------- | ---------------- |
| Tiny           | 8 GB           | RTX 3060+        |
| Small          | 16–24 GB       | RTX 3090 / A5000 |
| Base           | 24–40 GB       | A100 40GB        |
| CTPelvic (any) | 40–80 GB       | A100 80GB        |
| TransVert      | 4–10 GB        | Any modern GPU   |
