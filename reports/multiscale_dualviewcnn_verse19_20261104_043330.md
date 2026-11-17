# Experiment report: multiscale_dualviewcnn_verse19_20261104_043330

Status: **in progress**. This report is generated from the local CSV and log.

## Protocol

- Model: `multiscale_dualviewcnn`
- Dataset: `verse19`
- Split: `splits/verse19_seed42.json | Train: 747 | Val: 93 | Test: 94`
- Seed: `42`
- Augmentation: `True`
- Optimizer / LR: `adam` / `0.0002`
- Epochs configured / recorded: `200` / `167`

## Best validation epoch

| Epoch | Dice | IoU | Precision | Recall | Val. loss |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 137 | 0.8595 | 0.7619 | 0.8527 | 0.8684 | 0.0981 |

## Latest epoch

| Epoch | Train loss | Val. loss | Dice | Learning rate |
| ---: | ---: | ---: | ---: | ---: |
| 167 | 0.0555 | 0.0990 | 0.8589 | 6.25e-06 |

## Reporting note

These are validation results only. Do not report held-out test results until a final checkpoint is selected without consulting the test split.
