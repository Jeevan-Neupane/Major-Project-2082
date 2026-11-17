# Experiment report: dualviewcnn_ctpelvic_20261110_081701

Status: **in progress**. This report is generated from the local CSV and log.

## Protocol

- Model: `dualviewcnn`
- Dataset: `ctpelvic`
- Split: `splits/ctpelvic_seed42.json | Train: 433 | Val: 54 | Test: 54`
- Seed: `42`
- Augmentation: `True`
- Optimizer / LR: `adam` / `0.0001`
- Epochs configured / recorded: `200` / `126`

## Best validation epoch

| Epoch | Dice | IoU | Precision | Recall | Val. loss |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 96 | 0.5893 | 0.4224 | 0.5790 | 0.6112 | 0.2659 |

## Latest epoch

| Epoch | Train loss | Val. loss | Dice | Learning rate |
| ---: | ---: | ---: | ---: | ---: |
| 126 | 0.2245 | 0.2700 | 0.5836 | 3.125e-06 |

## Reporting note

These are validation results only. Do not report held-out test results until a final checkpoint is selected without consulting the test split.
