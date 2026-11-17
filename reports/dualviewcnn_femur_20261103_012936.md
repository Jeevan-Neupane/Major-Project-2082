# Experiment report: dualviewcnn_femur_20261103_012936

Status: **in progress**. This report is generated from the local CSV and log.

## Protocol

- Model: `dualviewcnn`
- Dataset: `femur`
- Split: `splits/femur_seed42.json | Train: 595 | Val: 74 | Test: 75`
- Seed: `42`
- Augmentation: `True`
- Optimizer / LR: `adam` / `0.0001`
- Epochs configured / recorded: `200` / `60`

## Best validation epoch

| Epoch | Dice | IoU | Precision | Recall | Val. loss |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 56 | 0.7650 | 0.6397 | 0.7606 | 0.7748 | 0.1852 |

## Latest epoch

| Epoch | Train loss | Val. loss | Dice | Learning rate |
| ---: | ---: | ---: | ---: | ---: |
| 60 | 0.1800 | 0.1937 | 0.7542 | 0.0001 |

## Reporting note

These are validation results only. Do not report held-out test results until a final checkpoint is selected without consulting the test split.
