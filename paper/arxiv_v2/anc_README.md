# Ancillary files

The paper reports 18 models (3 models x 6 datasets) trained under one protocol. All numbers in the paper can be
traced to the files below. The same code, splits and results are in the code repository
https://github.com/Jeevan-Neupane/biplanar-bone-reconstruction (release v1.0).

## Results
- `results_fair.json`: per-dataset test summaries, parameter counts, best epochs and paired statistics (MultiScale
  vs the other two models, Holm over 12 tests; exploratory SwinX2S vs DualViewCNN, Holm over 6). Made by
  `scripts/analyze_fair.py`.
- `val/fair_<model>_<dataset>.json`, `test/fair_<model>_<dataset>.json`: per-case metrics (Dice, IoU, precision,
  recall, HD95, ASSD, NSD at 1.5 mm) of every validation and test case, with the run folder, best epoch and the
  validation Dice stored in the checkpoint. Made by `scripts/evaluate_fair.py`.
  - `val/verification.json`: the validation re-scoring matched the stored values for all 18 runs (max 1.1e-5).
  - `test/test_run_info.json`: the test splits were scored once, on 17 Nov 2026.

## Data splits
- `splits/<dataset>_seed42.json`: seed-42 splits by sample folder (80/10/10).
- `splits/ctpelvic_seed42_dedup.json`: CTPelvic split with one copy of each scan stored twice (412/48/44); made by
  `scripts/make_clean_splits.py`.

## Geometry checks
- `geometry_check/data_flip_ap.json`: for every sample, whether the loaded AP and lateral images match projections
  of the mask better than their mirror images (AP mirrored on loading, as in training).
- `geometry_check/models_femur.json`, `models_hip.json`: sensitivity test of how each model places the two views
  in 3D (random weights, femur and hip sizes). Made by `scripts/check_geometry.py`.

## Training logs
- `logs/<run>.csv` (one row per epoch) and `logs/<run>.log` (settings and progress) for the 18 runs.

## Code
- `code/`: the training and model code (configs `fair_<model>_<dataset>` in `configs/config.py`). `scripts/`:
  evaluation, statistics, checks, tables/figures and the cloud/local launch scripts. The processed image data are not
  included (derived from the datasets; see the paper).

## Licence of the adapted Swin-X2S code

The Swin-X2S code is used under the following licence:

```
MIT License

Copyright (c) 2025 liukuan5625

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
