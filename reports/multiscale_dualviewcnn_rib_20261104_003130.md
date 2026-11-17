# Experiment report: multiscale_dualviewcnn_rib_20261104_003130

Status: **in progress**. This report is generated from the local CSV and log.

## Protocol

- Model: `multiscale_dualviewcnn`
- Dataset: `rib`
- Split: `splits/rib_seed42.json | Train: 307 | Val: 38 | Test: 39`
- Seed: `42`
- Augmentation: `True`
- Optimizer / LR: `adam` / `0.0002`
- Epochs configured / recorded: `200` / `169`

## Best validation epoch

| Epoch | Dice | IoU | Precision | Recall | Val. loss |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 139 | 0.3946 | 0.2531 | 0.3759 | 0.4180 | 0.3374 |

## Latest epoch

| Epoch | Train loss | Val. loss | Dice | Learning rate |
| ---: | ---: | ---: | ---: | ---: |
| 169 | 0.2082 | 0.3396 | 0.3926 | 2.5e-05 |

## Reporting note

These are validation results only. Do not report held-out test results until a final checkpoint is selected without consulting the test split.
