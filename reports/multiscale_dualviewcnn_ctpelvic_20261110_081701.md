# Experiment report: multiscale_dualviewcnn_ctpelvic_20261110_081701

Status: **in progress**. This report is generated from the local CSV and log.

## Protocol

- Model: `multiscale_dualviewcnn`
- Dataset: `ctpelvic`
- Split: `splits/ctpelvic_seed42.json | Train: 433 | Val: 54 | Test: 54`
- Seed: `42`
- Augmentation: `True`
- Optimizer / LR: `adam` / `0.0002`
- Epochs configured / recorded: `200` / `200`

## Best validation epoch

| Epoch | Dice | IoU | Precision | Recall | Val. loss |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 187 | 0.8275 | 0.7089 | 0.8164 | 0.8397 | 0.1175 |

## Latest epoch

| Epoch | Train loss | Val. loss | Dice | Learning rate |
| ---: | ---: | ---: | ---: | ---: |
| 200 | 0.0594 | 0.1178 | 0.8273 | 3.125e-06 |

## Reporting note

These are validation results only. Do not report held-out test results until a final checkpoint is selected without consulting the test split.
