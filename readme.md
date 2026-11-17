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

## Fair benchmark (paper version 2, November 2026)

The results in `paper/arxiv_v2/` come from the configs `fair_<model>_<dataset>`, for example
`fair_multiscale_hip` or `fair_swinx2s_femur`. Compared with the older configs, they add:

- AP images mirrored on loading (`FLIP_AP_LR`);
- both views placed in 3D according to the X-ray geometry (`VIEW_GEOMETRY="correct"`);
- standard Swin windows with drop path, averaging of the two SwinX2S outputs and the official two-view loss;
- GroupNorm (DualViewCNN) and InstanceNorm (SwinX2S);
- the same settings for every model: 200 epochs, batch size 2, early stopping after 40 epochs;
- the deduplicated CTPelvic split (`splits/ctpelvic_seed42_dedup.json`).

```bash
python train.py --config fair_multiscale_hip --no_wandb
python scripts/check_geometry.py --part models          # how each model places the two views in 3D
python scripts/check_geometry.py --part data --flip-ap  # do the images match projections of the masks?
python scripts/evaluate_fair.py --split val             # validation check (test: once, after this passes)
python scripts/analyze_fair.py                          # paired statistics
```

The old configs (without `fair_`) and their default options reproduce the first version exactly.
Trained models are on Hugging Face: `jeevan1234/biplanar-bone-reconstruction-models` (folder `fair_v2/`).

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
