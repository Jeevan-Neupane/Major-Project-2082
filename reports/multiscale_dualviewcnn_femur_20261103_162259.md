# Experiment report: multiscale_dualviewcnn_femur_20261103_162259

Status: **in progress**. This report is generated from the local CSV and log.

## Protocol

- Model: `multiscale_dualviewcnn`
- Dataset: `femur`
- Split: `splits/femur_seed42.json | Train: 595 | Val: 74 | Test: 75`
- Seed: `42`
- Augmentation: `True`
- Optimizer / LR: `adam` / `0.0002`
- Epochs configured / recorded: `200` / `182`

## Best validation epoch

| Epoch | Dice | IoU | Precision | Recall | Val. loss |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 152 | 0.9253 | 0.8730 | 0.9294 | 0.9228 | 0.0645 |

## Latest epoch

| Epoch | Train loss | Val. loss | Dice | Learning rate |
| ---: | ---: | ---: | ---: | ---: |
| 182 | 0.0339 | 0.0647 | 0.9251 | 7.8125e-07 |

## Reporting note

These are validation results only. Do not report held-out test results until a final checkpoint is selected without consulting the test split.
