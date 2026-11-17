# Validation metrics by dataset

Updated: 2026-11-04 06:59 UTC.

Metrics are from the epoch with the highest recorded validation Dice for each run. Scores are percentages; loss is unscaled. These are validation results, not held-out test results.

| Dataset | Model | Status | Best epoch | Dice (%) | IoU (%) | Precision (%) | Recall (%) | Val loss |
|---|---|---|---:|---:|---:|---:|---:|---:|
| femur | DualViewCNN | Complete | 172 | 78.66 | 66.85 | 79.62 | 78.19 | 0.1679 |
| femur | MultiScale DualViewCNN | Complete | 152 | 92.53 | 87.30 | 92.94 | 92.28 | 0.0645 |
| femur | SwinX2S | Complete | 136 | 90.25 | 83.59 | 91.41 | 89.76 | 0.0821 |
| hip | DualViewCNN | Complete | 104 | 58.22 | 42.78 | 61.69 | 59.45 | 0.2733 |
| hip | MultiScale DualViewCNN | Complete | 116 | 80.22 | 67.49 | 78.90 | 81.68 | 0.1404 |
| hip | SwinX2S | Complete | 32 | 70.52 | 54.68 | 58.23 | 89.83 | 0.5480 |
| rib | DualViewCNN | Complete | 122 | 19.36 | 10.95 | 16.05 | 25.42 | 0.4426 |
| rib | MultiScale DualViewCNN | Complete | 139 | 39.46 | 25.31 | 37.59 | 41.80 | 0.3374 |
| rib | SwinX2S | Complete | 25 | 15.68 | 8.59 | 9.50 | 49.68 | 0.6495 |
| verse19 | DualViewCNN | Complete | 140 | 47.17 | 32.35 | 44.69 | 52.29 | 0.3447 |
| verse19 | MultiScale DualViewCNN | Provisional | 117 | 85.78 | 75.95 | 86.32 | 85.43 | 0.0985 |
| verse19 | SwinX2S | Provisional | 41 | 79.43 | 66.99 | 82.01 | 77.67 | 0.1383 |
