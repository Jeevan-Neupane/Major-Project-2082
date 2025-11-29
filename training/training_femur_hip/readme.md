# SwinX2S & TransVert - 3D Reconstruction from X-rays

Two deep learning models for 3D bone reconstruction from orthogonal 2D X-ray views:
- **SwinX2S**: Swin Transformer-based architecture (~52M parameters)
- **TransVert**: Lightweight CNN-based architecture (~0.5M parameters)

---

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone <your-repo-url>
cd swinx2s-project

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Test Models

```bash
# Test both models (checks input/output shapes and parameters)
python test/model_test.py
```

Expected output:
```
🧪 TESTING SWINX2S MODEL
✅ Model created successfully
   Parameters: 52,145,025 (52.15M)

🧪 TESTING TRANSVERT MODEL
✅ Model created successfully
   Parameters: 546,593 (0.55M)
```

---

## 🏋️ Training

### Basic Usage

```bash
# Train SwinX2S (default)
python train.py --model swinx2s --data-dir ./data --epochs 100 --batch-size 2

# Train TransVert
python train.py --model transvert --data-dir ./data --epochs 100 --batch-size 4
```

### With Weights & Biases Logging

```bash
# Train with W&B logging enabled
python train.py --model swinx2s --data-dir ./data --epochs 100 --wandb

# Train TransVert with W&B
python train.py --model transvert --data-dir ./data --epochs 150 --batch-size 4 --wandb
```

### Custom Save Directory

```bash
# Save checkpoints to custom directory
python train.py \
    --model swinx2s \
    --data-dir ./data \
    --save-dir ./my_checkpoints \
    --epochs 100 \
    --batch-size 2
```

### Adjust Training Split

```bash
# Use 80% data for training (default is 70%)
python train.py \
    --model transvert \
    --data-dir ./data \
    --train-ratio 0.8 \
    --epochs 100
```

---

## 📝 Command Line Arguments

| Argument | Type | Required | Default | Description |
|----------|------|----------|---------|-------------|
| `--model` | str | No | `swinx2s` | Model architecture (`swinx2s` or `transvert`) |
| `--data-dir` | str | **Yes** | - | Path to dataset directory |
| `--epochs` | int | No | `100` | Number of training epochs |
| `--batch-size` | int | No | `2` | Batch size for training |
| `--train-ratio` | float | No | `0.7` | Ratio of data for training (0-1) |
| `--save-dir` | str | No | `./checkpoints` | Directory to save model checkpoints |
| `--wandb` | flag | No | False | Enable Weights & Biases logging |

---

## 📊 Model Specifications

### SwinX2S
- **Architecture**: Swin Transformer with cross-view attention
- **Parameters**: ~52.1M
- **Input**: Two 2D X-ray views (sagittal & coronal) [B, 1, 128, 128]
- **Output**: 3D volume [B, 1, 64, 128, 128]
- **Memory**: ~8-14 GB GPU (batch size 1-2)
- **Best For**: High-quality reconstruction

### TransVert
- **Architecture**: Lightweight CNN with volumetric expansion
- **Parameters**: ~0.55M
- **Input**: Two 2D X-ray views (sagittal & coronal) [B, 1, 128, 128]
- **Output**: 3D volume [B, 1, 64, 128, 128]
- **Memory**: ~2-3 GB GPU (batch size 1-4)
- **Best For**: Fast training/inference, limited GPU memory

---

## 💾 GPU Memory Requirements

| Model | Batch=1 | Batch=2 | Batch=4 |
|-------|---------|---------|---------|
| SwinX2S | ~8 GB | ~14 GB | ~24 GB |
| TransVert | ~2 GB | ~3 GB | ~5 GB |

**Recommendation**:
- **RTX 3090 / A100**: Use SwinX2S with batch size 2
- **RTX 3060 / 2080 Ti**: Use TransVert with batch size 4
- **Limited GPU**: Use TransVert with batch size 1-2

---

## 📈 Training Examples

### Example 1: Quick Test (TransVert)
```bash
python train.py \
    --model transvert \
    --data-dir ./data \
    --epochs 50 \
    --batch-size 4
```

### Example 2: Full Training (SwinX2S with W&B)
```bash
python train.py \
    --model swinx2s \
    --data-dir ./data \
    --epochs 200 \
    --batch-size 2 \
    --train-ratio 0.8 \
    --save-dir ./results/swinx2s \
    --wandb
```

### Example 3: Small Dataset (TransVert)
```bash
python train.py \
    --model transvert \
    --data-dir ./data \
    --epochs 100 \
    --batch-size 8 \
    --train-ratio 0.6
```

---

## 📁 Project Structure

```
swinx2s-project/
├── configs/
│   └── config.py              # Default configuration
├── models/
│   ├── swinx2s.py             # SwinX2S model (~52M params)
│   ├── transvert_model.py     # TransVert model (~0.5M params)
│   ├── swinX2S/               # SwinX2S components
│   └── transvert/             # TransVert components
├── utils/
│   ├── dataset.py             # Dataset loaders
│   ├── losses.py              # Loss functions
│   ├── trainer.py             # Training utilities
│   ├── checkpoint.py          # Checkpoint management
│   └── metrics.py             # Evaluation metrics
├── test/
│   └── model_test.py          # Model testing script
├── train.py                   # Main training script
├── requirements.txt           # Dependencies
└── README.md                  # This file
```

---

## 📊 Expected Training Time

On RTX 3090:

| Model | Batch Size | Time/Epoch | 100 Epochs |
|-------|-----------|------------|------------|
| SwinX2S | 2 | ~15 min | ~25 hours |
| TransVert | 4 | ~2 min | ~3 hours |

---

## 🐛 Troubleshooting

### Out of Memory Error
```bash
# Use smaller batch size
--batch-size 1

# Or use TransVert instead
--model transvert
```

### Dataset Not Found
```bash
# Make sure data directory exists and contains data
ls ./data  # Check if directory has files

# Specify correct path
--data-dir /path/to/your/data
```

### Slow Training
```bash
# Check if CUDA is available
python -c "import torch; print(torch.cuda.is_available())"

# Use larger batch size if memory allows
--batch-size 4
```

---

## 📧 Contact

For questions or issues, please open an issue on GitHub.

---

## 🙏 Acknowledgments

- SwinX2S based on Swin Transformer architecture
- TransVert custom lightweight architecture