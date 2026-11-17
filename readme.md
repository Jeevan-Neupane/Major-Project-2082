# X-Ray to 3D Bone Reconstruction — Training Repository

Training-only codebase for reconstructing binary 3D bone masks from paired AP
and lateral X-rays.

## Models

| Model | Type | Purpose |
| --- | --- | --- |
| `DualViewCNN` | lightweight CNN | single-scale baseline |
| `MultiScaleDualViewCNN` | multi-scale CNN | stronger CNN baseline |
| `Swin-X2S` | transformer | cross-view transformer baseline |

## Train

```bash
python train.py --list_configs
python train.py --config dualviewcnn_femur
python train.py --config multiscale_dualviewcnn_femur
python train.py --config swinx2s_femur
```

Use `--dry_run --no_wandb --num_workers 0` to verify a configuration without
starting an experiment.

## Paper (November 2026)

The paper is in `paper/arxiv_v3/` (`final_paper_v3.pdf`, arXiv package `arxiv_v3.zip`); the version for the major
project report is in `paper/major_report_v3/`. Its code, data splits and per-case results are in the repository
https://github.com/Jeevan-Neupane/biplanar-bone-reconstruction (release v1.0), with the configs
`fair_<model>_<dataset>`, for example `fair_multiscale_hip`. Trained models are on Hugging Face:
`jeevan1234/biplanar-bone-reconstruction-models`.

## Data protocol

The source data lives in `datasets/`. Reproducible disjoint 80/10/10 PID
partitions are stored in `splits/` and are required by training. Regenerate
them with:

```bash
python scripts/create_patient_splits.py --seed 42
```

See [data split documentation](docs/DATA_SPLITS.md) and
[training configurations](docs/TRAINING_CONFIGS.md) for details.

## Repository layout

```text
configs/     training presets and CLI arguments
datasets/    local AP/LAT X-rays and 3D mask targets (Git-ignored)
docs/        experiment protocol documentation
models/      CNN and transformer architectures
scripts/     split-generation utility
splits/      reproducible train/validation/test PID manifests
utils/       data loading, losses, metrics, checkpoints, logging
train.py     training entry point
```
