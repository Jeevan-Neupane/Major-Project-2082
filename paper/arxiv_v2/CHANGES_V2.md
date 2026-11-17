# Version 2: what changed and why (17 Nov 2026)

Version 1 is kept unchanged in `paper/arxiv_submission/`.

## Why a new version
A post-hoc check of version 1 found implementation errors that made the model comparison unfair:
1. **View placement.**
   - All three models used the AP image as stored, which is mirrored relative to the volume's W axis.
   - DualViewCNN and SwinX2S placed the lateral image columns on W instead of D.
   - Only MultiScale placed the lateral view correctly.
2. **Swin encoder code.** The vendored copy (models/swinX2S/swin_transformer.py) built the attention windows with
   `unfold` + `view`. This mixed pixels and channels, and the reverse step did not undo it. It also never applied
   drop path. The official code does neither.
3. **Training settings.**
   - Epoch limits, batch sizes and patience differed per model.
   - SwinX2S used BatchNorm at batch size 1 and an extra 1x1x1 output-fusion layer, and the Small model on
     hip/rib/CTPelvic.
4. **CTPelvic duplicates.** 37 duplicated scans were still in the training and validation splits. In version 1
   they were only excluded from the test statistics.

## What was done
- **Code fixes**, all opt-in so the version-1 models still load and give identical outputs:
  - AP flip on loading (`FLIP_AP_LR`);
  - correct lateral placement for DualViewCNN and SwinX2S (`VIEW_GEOMETRY="correct"`);
  - standard Swin windows plus drop path (`SWIN_IMPL="official"`);
  - official averaging of the two SwinX2S outputs and the official two-view loss (`SWIN_OUTPUT_MODE="average"`);
  - GroupNorm for DualViewCNN and InstanceNorm for SwinX2S (official default);
  - SwinX2S Tiny on every dataset;
  - for all models: 200 epochs, batch size 2, patience 40;
  - deduplicated CTPelvic split (412/48/44).
- **Verification:**
  - Projection test on every sample (`scripts/check_geometry.py --part data`).
  - Sensitivity test of every model (`--part models`).
  - Three independent code reviews and a line-by-line comparison with the official Swin-X2S code.
  - Unit test of the window fix against an independent reference (`reports/verification/test_swin_windows.py`).
- **Training:**
  - All 18 models retrained: 14 on the local RTX 5090 and 4 SwinX2S runs on Modal L40S GPUs.
  - SwinX2S hip and CTPelvic stopped at the cloud time limit at epochs 195 and 190; the learning rate was already
    below 1e-6.
  - SwinX2S CTPelvic was preempted once and restarted from scratch; only the complete restarted run is used.
- **Evaluation:**
  - Validation was rescored first; all 18 runs matched the stored values within 1.1e-5.
  - The test split was then scored once, on 17 Nov 2026 (`scripts/evaluate_fair.py`).
  - Statistics were computed with the same method as version 1 (`scripts/analyze_fair.py`).

## Effect (same 297 test cases)
Mean Dice over the six datasets, version 1 → version 2:

| Model | Version 1 | Version 2 |
|---|---|---|
| DualViewCNN | 53.8 | 72.6 |
| SwinX2S | 69.1 | 80.0 |
| MultiScale | 78.9 | 83.7 |

MultiScale is still best on all six datasets. It is significantly better than DualViewCNN on 6/6 and than
SwinX2S on 5/6 (Holm); the exception is VerSe20, n=10.

## Text changes
- Abstract, introduction (contributions), problem setup (AP mirroring, data check on every sample), splits
  (CTPelvic dedup), preprocessing, model descriptions, Table 1 (models), training (shared settings, hardware,
  time limits, preemption), Table 2 (training settings), evaluation protocol (new paragraph "Relation to an
  earlier version"), statistics (exploratory SwinX2S vs DualViewCNN), all results text and tables, new
  Section "Before and after the corrections" with Table "beforeafter", discussion, limitations, future work,
  conclusion and code availability.
- The models will be released on Hugging Face. They still need to be uploaded with the user's token (see
  `submission_checklist.md`).
- Figures:
  - schematic figures redrawn: overview, SwinX2S, DualViewCNN;
  - data figures regenerated: test Dice distribution, validation curves, loss curves, qualitative;
  - the CTPelvic duplicates table was removed.
