# DRR Generation — Improvement Research & Recommendations

> **Scope:** Research review of the X-Ray-to-3D Bone Reconstruction project's DRR generation stage
> (based on [SeverineHabert/DRR-renderer](https://github.com/SeverineHabert/DRR-renderer)),
> with concrete improvement suggestions and a comparison of alternative open-source DRR generators.
>
> Sources reviewed: `data_preparation/drr_generation/` C++ sources + [readme.md](../data_preparation/drr_generation/readme.md),
> `data_preparation/notebooks/`, `src/datasets/bone_dataset.py`, `src/configs/config.py`, upstream repo code, and candidate repos.

---

## Table of Contents

1. [Current Pipeline Summary](#1-current-pipeline-summary)
2. [What Was Inherited vs. Changed](#2-what-was-inherited-vs-changed)
3. [Improvements Needed (Ranked)](#3-improvements-needed-ranked)
4. [Alternative Open-Source DRR Generators](#4-alternative-open-source-drr-generators)
5. [Recommended Adoption Path](#5-recommended-adoption-path)
6. [Project-Level Cautions (Beyond DRR)](#6-project-level-cautions-beyond-drr)
7. [Quick-Win Checklist](#7-quick-win-checklist)

---

## 1) Current Pipeline Summary

The project reconstructs 3D bone volumes (femur, hip, rib, spine — VerSe19/20) from 2 views
(AP + Lateral X-ray) using **SwinX2S** (Swin Transformer + Cross-Attention + UNETR decoder, 32M–558M params)
and **TransVert** (CNN encoder + volumetric expansion + CNN decoder, ~0.5M params), trained on
TotalSegmentator / VerSe / CTPelvic1K datasets (~1300 volumes max per dataset).

The DRR stage (`data_preparation/drr_generation/`) is a standalone C++/CMake program:

| Aspect | Current setting |
|---|---|
| Inputs | `.mhd + .raw` (MetaImage), `.nii` / `.nii.gz` (NIfTI-1) |
| Camera | Source 4000 mm from CT center, detector 2000 mm, 1024×1024 px @ 1 mm |
| Rays | 6 axis-aligned directions (`+X … −Z`) only — AP/LAT chosen per-format (`-Y/+X` for mhd, `+Y/-X` for nifti) |
| Sampling | Fixed step size = smallest voxel spacing; trilinear interpolation of HU |
| Physics | HU ≤ 200 → μ = 0; HU > 200 → μ = μ_water · (1 + HU/1000), μ_water = 0.022 (NIST 50 keV) |
| Integral | **Average** of μ over samples (not a line integral) |
| Output | Min-max normalized, inverted (bone = white), 8-bit 3-channel PNG |

Verdict: a solid teaching/reference baseline, but it currently produces **stylized bone-only
sum Maps, not physical X-rays**, and it is slow (multi-minute, single-threaded per volume).

---

## 2) What Was Inherited vs. Changed vs. Original

| Aspect | Upstream DRR-renderer (2017, TUM) | This project |
|---|---|---|
| Language | C++ + OpenCV + CUDA + OpenMP | C++17, no OpenCV/OpenMP/CUDA |
| Acceleration | GPU (CUDA kernel) + `#pragma omp parallel for` | Single-threaded CPU |
| Formats | `.mhd` only | `.mhd` + `.nii`/`.nii.gz` (own gzread loader) |
| Metadata | Reads spacing/offset; **does not use translation** | Full index↔world transform + direction matrix |
| HU map | Threshold −800 (or 100 for CT-typed volumes) → `μ = μ_water·(1 + HU/1000)` | Threshold raised to **200** → same linear map |
| Integral | **Mean over ray samples** (upstream bug inherited) | Same — mean, not line integral |
| Resolution | 640×480, f=2200 mm | 1024×1024, source 4000 mm / detector 2000 mm |
| Output | 8-bit cv::Mat | 8-bit PNG via stb_image_write |

Key insight: the **averaging-instead-of-integral behavior was inherited from upstream** — it was
never physically consistent in either codebase, so fixing it is safe and worthwhile.

---

## 3) Improvements Needed (Ranked)

### 🔴 Priority 1 — Physics / units of the ray integral (affects what the network learns)

#### (a) Averaging instead of a line integral — `src/RayMarcher.cpp:51`

```cpp
return (count > 0) ? (sum / count) : 0.0f;   // ← mean μ per sample, units are wrong
```

Beer-Lambert requires the line integral $\int \mu(s)\,ds$. Averaging makes the same femur appear
dimmer when the CT bounding box adds more air samples — brightness becomes ray-length-dependent
per patient, breaking cross-patient consistency.

**Fix (one line):**

```cpp
return sum * stepSize;   // units: μ(mm⁻¹)·mm = dimensionless line integral
```

Then either transmit $I/I_0 = \exp(-\text{integral})$ (film intensity) or feed the negative log
directly (detector log-response). For segmentation training the monotone transform matters less,
but the `× stepSize` fix is essential for cross-patient consistency.

#### (b) HU threshold at 200 is too aggressive — `src/RayMarcher.cpp:5,32`

- Real X-rays: soft tissue contributes ~5–10% of attenuation — it is never zero.
- Trabecular (spongy) bone spans **~150–350 HU**; the HU ≤ 200 cutoff zeroes out half of it, so
  cortical bone dominates and thin structures (rib edges, vertebra endplates, acetabular wall)
  fade or vanish.
- Upstream used −800 (or 100); this project raised it to 200.

**Fix:** replace the hard step with a smooth ramp — e.g. `HU ≤ −800 → μ ≈ 0`, then
$\mu \propto (1 + \mathrm{HU}/1000)$ everywhere (optionally with a small soft-tissue weight).
Alternative: render from the **segmented bone masks** (femur/pelvis/spine labels already exist)
for a controllable bone-only variant, and keep the full-HU variant for realism.

#### (c) Threshold-after-interpolation causes stair-stepping — `src/RayMarcher.cpp:45-47`

HU is trilinearly interpolated, *then* the discontinuous threshold is applied. Interpolate
**after** applying μ (or use cubic B-spline like DeepDRR's projector) — otherwise bone
boundaries develop banding artifacts.

### 🟠 Priority 2 — Geometry realism

#### (d) Camera geometry is not clinical — `src/XRayCamera.cpp:5-9,28`

`SOURCE_DISTANCE=4000`, `DETECTOR_DISTANCE=2000`, 1 mm pixels → SDD = 6000 mm, magnification 1.5×,
only ±5° ray divergence — **effectively a parallel projection**. Real C-arms: SDD ≈ 1020–1100 mm,
source-to-isocenter ≈ 700–800 mm. Re-parameterize to clinical values if DRRs must resemble real
radiographs, and record the effective magnification in the config.

#### (e) Orientation handled by per-format axis flips — `src/main.cpp:118-119`

`.mhd` uses `-Y/+X`, `.nii` uses `+Y/-X` as a "practical correction". This is data-dependent
guesswork: one TotalSegmentator export with a different `TransformMatrix` silently produces
mirrored AP views; an AP↔LAT label swap would corrupt supervised pairs.

**Fix:** derive AP/LAT camera axes from the actual patient orientation (LPS vs RAS) in the
direction matrix; at minimum add a QC script that overlays projected label silhouettes on the
DRR to detect flipped/mirrored views.

#### (f) No pose augmentation

Training uses only perfectly orthogonal AP/LAT views. Real X-rays have 2–15° obliquity. The
camera supports only 6 axis directions (`src/XRayCamera.cpp:18-25`); adding yaw/pitch (±10°) and
small translations would substantially improve robustness at inference on real images.

### 🟡 Priority 3 — Speed & data format

#### (g) Single-threaded rendering — `src/DRRGenerator.cpp:13-20`

1024² rays × ~500 samples × trilinear interpolation (8 voxel reads each), zero parallelism;
upstream at least had `#pragma omp parallel for`. Options (ascending effort):

1. `#pragma omp parallel for collapse(2)` on the pixel loops + enable OpenMP in `CMakeLists.txt`
   → ~8–16× speedup for free.
2. `stepSize = min_spacing / 2`, or implement **Siddon's exact algorithm** (see §4).
3. Offload rendering to DiffDRR/nanodrr (GPU) → ~25–30 ms per image.

#### (h) Min-max normalization per image — `src/DRRGenerator.cpp:24-38`

Every DRR is stretched to full [0,1] independently, so a patient with low bone density gets
scaled to look identical to a dense one — absolute intensity carries no signal.

**Fix:** with the `× stepSize` fix, use a **fixed global window/scale across the whole dataset**;
save the raw float integral (`.npy` or 16-bit PNG) and let the dataset loader
(`src/datasets/bone_dataset.py: _load_xray`) own normalization and augmentation.

#### (i) 8-bit PNG only

3-channel 8-bit PNG is saved, then the training loader resizes to 96–320 px. Save **16-bit
grayscale PNG** (stb_image_write supports it) at native resolution; 8-bit throws away
low-contrast detail (thin bone edges) the models are supposed to learn.

### ⚪ Priority 4 — Realism extras (only if targeting real-X-ray transfer)

- **Noise/blur:** add Poisson (quantum) noise + small Gaussian detector blur before PNG save —
  cheap, big robustness gain. DeepDRR goes fully photorealistic (material decomposition via a
  V-net + Monte Carlo Rayleigh scatter + noise injection) if Linux + NVIDIA is available.
- **Beam quality:** constant μ ignores the energy spectrum; a per-dataset scale factor is
  usually enough for the downstream segmentation task.

---

## 4) Alternative Open-Source DRR Generators

| Repo | Stars | Approach | GPU | Maintained | Fit for this project |
|---|---|---|---|---|---|
| [eigenvivek/DiffDRR](https://github.com/eigenvivek/DiffDRR) (MICCAI 2022) | 300 | Siddon ray-tracing, differentiable, PyTorch module | ✅ CUDA | ✅ Active (2026) | **Best overall fit.** `pip install diffdrr`, pure Python, ~25 ms for 200² on RTX 2080 Ti, arbitrary camera poses (pose augmentation for free), 3D-labelmap 2D projection built in, MIT license |
| [eigenvivek/nanodrr](https://github.com/eigenvivek/nanodrr) | new | DiffDRR re-written for speed (fused Triton kernel) | ✅ CUDA | ✅ Most current | Same author recommends it over DiffDRR: ~5× faster (PyTorch backend), ~40× with Triton, 4× less memory, `torch.compile` compatible |
| [arcadelab/deepdrr](https://github.com/arcadelab/deepdrr) (DeepDRR, arXiv 1803.08606) | 284 | Material decomposition + Monte Carlo scatter + noise = photorealistic X-ray | ✅ CUDA (cupy) | ✅ Active | **Most realistic** — the standard for closing the synthetic→real gap. Requires Linux + conda + NVIDIA; heavier to set up |
| [fabio86d/CUDA_DigitallyReconstructedRadiographs](https://github.com/fabio86d/CUDA_DigitallyReconstructedRadiographs) | 64 | Exact CUDA **Siddon** algorithm, Python + C++/CUDA | ✅ CUDA | ⚠️ 2018-era | Gold-standard *exact* line integrals; useful reference implementation for fixing the custom C++ |
| [fabio86d/HipHop_2D3Dregistration](https://github.com/fabio86d/HipHop_2D3Dregistration) | 173 | 2D/3D registration toolchain with DRR projection | C++/Cython | ⚠️ | Useful for evaluation/registration add-ons |
| [farrell236/DRR-RATE](https://github.com/farrell236/DRR-RATE), [elda27/pydrr](https://github.com/elda27/pydrr), [roulbac/2D3DAutoReg](https://github.com/roulbac/2D3DAutoReg), [yuta-hi/pycuda_drr](https://github.com/yuta-hi/pycuda_drr) | 2–26 | Various CUDA/Python DRR renderers | ✅ | mixed | Smaller/less maintained; handy as Siddon/CUDA code references |
| [eigenvivek/DiffDRR-Datasets](https://github.com/eigenvivek/DiffDRR-Datasets) | 25 | Pre-built paired 2D/3D registration datasets | — | ✅ | Free extra training/eval pairs (TotalSegmentator-based), compatible with this project's data |

**Bottom line:** keep the C++ renderer as a teaching/reference implementation, and move the
production DRR source to **DiffDRR or nanodrr** — they slot directly into the PyTorch training
stack, support arbitrary poses for augmentation, are fast enough to render DRRs **on-the-fly
inside the DataLoader** (removing the pre-generated PNG dependency), and provide exact line
integrals with correct HU→μ mapping.

Example parity with the current setup:

```python
import torch
from diffdrr.drr import DRR

drr = DRR(
    subject,          # CT volume + origin + spacing (SimpleITK-loadable)
    sdd=1020.0,       # clinical source-to-detector distance (instead of 6000 mm)
    height=512,
    delx=0.7,         # match voxel spacing
).to("cuda")

rotations    = torch.tensor([[0.0, 0.0, 0.0]])   # yaw, pitch, roll → randomize ±10°
translations = torch.tensor([[0.0, 850.0, 0.0]])
img = drr(rotations, translations, parameterization="euler_angles", convention="ZXY")
```

---

## 5) Recommended Adoption Path

1. **Short term (keep the C++, ~1 hour of work):**
   - Fix (a): `sum * stepSize` line integral in `RayMarcher.cpp:51`
   - Fix (c): apply μ before interpolation, or interpolate-then-apply smooth map
   - Fix (b): smooth −800-based HU→μ map instead of the hard 200 cutoff
   - Add OpenMP (`collapse(2)`) for a free ~8–16× speedup
   - Save 16-bit grayscale PNGs; keep min-max constant across the dataset
2. **Medium term (recommended):** adopt **DiffDRR / nanodrr** as the DRR source; generate
   on-the-fly in the DataLoader; enable yaw/pitch pose augmentation (±10°).
3. **If real-X-ray generalization matters** (the frontend accepts user X-rays): generate a
   second training set with **DeepDRR**'s noise/scatter/beam models; train or fine-tune the same
   architectures on it.

---

## 6) Project-Level Cautions (Beyond DRR)

- **Cross-config patient leakage:** femur/hip/rib all draw from TotalSegmentator. Verify no
  patient appears in both train and test across configs, otherwise Dice scores are optimistic.
- **Anisotropic spine volumes:** VerSe19 volumes are 96×96×31–48 and are upscaled to 128 mm
  depth — the network invents slices that do not exist in the source. Worth a limitation note
  in the report; consider a depth-conditional config.
- **Orientation QC:** add an automated check that AP/LAT views are not mirrored (see 3e).

---

## 7) Quick-Win Checklist

- [ ] Line integral: `sum * stepSize` (RayMarcher.cpp:51)
- [ ] Smooth HU→μ map, remove hard HU ≤ 200 → 0 (RayMarcher.cpp:5,32)
- [ ] Interpolate μ, not HU (RayMarcher.cpp:45-47)
- [ ] OpenMP parallel pixel loops + CMake flag (DRRGenerator.cpp:13-20, CMakeLists.txt)
- [ ] Global fixed normalization; save 16-bit PNG / float .npy (DRRGenerator.cpp:24-38)
- [ ] Orientation-derived AP/LAT axes + QC overlay script (main.cpp:118-119)
- [ ] Clinical camera parameter set as a config option (XRayCamera.cpp:5-9)
- [ ] Trial: DiffDRR/nanodrr on-the-fly DRR + pose augmentation in bone_dataset.py
- [ ] Optional: Poisson noise + Gaussian blur augmentation
- [ ] Verify no cross-config patient leakage