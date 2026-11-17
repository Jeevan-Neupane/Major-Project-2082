# Experiment report: multiscale_dualviewcnn_verse20_20261104_110801

Status: **in progress**. This report is generated from the local CSV and log.

## Protocol

- Model: `multiscale_dualviewcnn`
- Dataset: `verse20`
- Split: `splits/verse20_seed42.json | Train: 85 | Val: 11 | Test: 10`
- Seed: `42`
- Augmentation: `True`
- Optimizer / LR: `adam` / `0.0002`
- Epochs configured / recorded: `200` / `146`

## Best validation epoch

| Epoch | Dice | IoU | Precision | Recall | Val. loss |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 116 | 0.8497 | 0.7412 | 0.8600 | 0.8407 | 0.1144 |

## Latest epoch

| Epoch | Train loss | Val. loss | Dice | Learning rate |
| ---: | ---: | ---: | ---: | ---: |
| 146 | 0.0558 | 0.1176 | 0.8484 | 2.5e-05 |

## Reporting note

These are validation results only. Do not report held-out test results until a final checkpoint is selected without consulting the test split.
