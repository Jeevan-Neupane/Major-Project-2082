# Data splits

The source folders previously had validation and test directories copied from
the training directory, causing PID overlap. The original files are preserved
as an immutable sample store; reproducible manifests under `splits/` now define
disjoint 80/10/10 train/validation/test partitions.

Create or reproduce them with:

```bash
python scripts/create_patient_splits.py --seed 42
```

Training loads only PIDs assigned to its manifest and validates that all three
sets are disjoint. The held-out test set is never loaded by the training loop.

The available folder identifiers are `p_####`. For TotalSegmentator and
CTPelvic these are treated as patient/sample IDs. The processed VerSe folders
do not retain original scan IDs, so its split is guaranteed at `p_####` sample
level only; obtain original scan-to-vertebra mappings before claiming a
strictly scan-level VerSe patient split in a publication.

## Training augmentation

Only training data is augmented. The pipeline uses paired geometric
reflections that transform the AP image, lateral image, and target volume in
the corresponding axes, plus mild per-view gamma, gain, bias, and Gaussian
noise perturbations. Validation and test samples remain unmodified.

Arbitrary 2D rotation, crop, or translation is intentionally not used: with
biplanar projections, applying such transforms without a calibrated matching
3D transform would corrupt input-target alignment.
