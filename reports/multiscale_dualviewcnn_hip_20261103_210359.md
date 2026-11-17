# Experiment report: multiscale_dualviewcnn_hip_20261103_210359

Status: **in progress**. This report is generated from the local CSV and log.

## Protocol

- Model: `multiscale_dualviewcnn`
- Dataset: `hip`
- Split: `splits/hip_seed42.json | Train: 285 | Val: 36 | Test: 35`
- Seed: `42`
- Augmentation: `True`
- Optimizer / LR: `adam` / `0.0002`
- Epochs configured / recorded: `200` / `146`

## Best validation epoch

| Epoch | Dice | IoU | Precision | Recall | Val. loss |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 116 | 0.8022 | 0.6749 | 0.7890 | 0.8168 | 0.1404 |

## Latest epoch

| Epoch | Train loss | Val. loss | Dice | Learning rate |
| ---: | ---: | ---: | ---: | ---: |
| 146 | 0.0679 | 0.1419 | 0.8017 | 6.25e-06 |

## Reporting note

These are validation results only. Do not report held-out test results until a final checkpoint is selected without consulting the test split.
