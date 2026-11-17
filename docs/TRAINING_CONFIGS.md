# Training configuration protocol

The configuration registry is the source of truth for training commands. These
presets are paper-aligned starting points, not claims that one architecture is
universally best before held-out evaluation.

## Recommended primary runs

| Dataset | Swin-X2S | DualViewCNN baseline | Output volume |
| --- | --- | --- | --- |
| Femur | Tiny, batch 1 | 32 channels, batch 4 | 96 x 96 x 128 |
| Hip | Small, batch 1 | 32 channels, batch 2 | 128 x 128 x 160 |
| Rib | Small, batch 1 | 48 channels, batch 2 | 128 x 128 x 160 |
| VerSe19 | Tiny, batch 1 | 32 channels, batch 4 | 96 x 96 x 128 |
| VerSe20 | Tiny, batch 1 | 32 channels, batch 4 | 96 x 96 x 128 |
| CTPelvic | Small, batch 1 | 32 channels, batch 1 | 128 x 128 x 160 |

Swin-X2S uses 250 epochs, AdamW (3e-5), weight decay 0.05, 20-epoch
linear warm-up, cosine decay, a 7x7 window, and one cross-view fusion layer.
The original implementation used a 0.5 regularization weight together with
layer-wise parameter groups; this project does not implement those groups, so
0.05 is the safer comparable AdamW baseline.

DualViewCNN uses 200 epochs, Adam (1e-4), and ReduceLROnPlateau. It is a
general direct-mask CNN baseline, deliberately named separately from TransVert.

## Evidence

- The [official Swin-X2S repository](https://github.com/liukuan5625/Swin-X2S)
  supplies Tiny and Base scripts with 96x96x128 or 128x128x160 inputs,
  batch size 1, AdamW at 3e-5, 20 warm-up epochs, and 250 total epochs.
- The [Swin-X2S paper](https://arxiv.org/abs/2501.05961) evaluates biplanar
  X-ray-to-3D reconstruction across femur, hip, spine, rib, and pelvic data.
- The [NeurIPS biplanar benchmark](https://proceedings.neurips.cc/paper_files/paper/2023/hash/412732f172bdd5ad0efde2fafa110700-Paper-Datasets_and_Benchmarks.pdf)
  evaluates CNN encoder-decoder baselines across femur, hip, spine, and rib.

Run a one-epoch smoke test before every full experiment, then record the exact
config name, seed, code commit, and checkpoint path in the paper experiment log.

## MultiScaleDualViewCNN

`MultiScaleDualViewCNN` is the second CNN comparator. It independently encodes
AP and lateral X-rays, then fuses their lifted 3D features at every decoder
scale. It uses four scales, GroupNorm (safe for batch size one), Adam at
2e-4, and the same input/output sizes and patient exclusions as the other
models. Its implementation is a clean-room segmentation adaptation of the
multi-scale fusion family evaluated in the
[NAAMII NeurIPS benchmark](https://proceedings.neurips.cc/paper_files/paper/2023/hash/412732f172bdd5ad0efde2fafa110700-Paper-Datasets_and_Benchmarks.pdf),
not an X2CT-GAN CT-intensity reproduction.

Use `multiscale_dualviewcnn_<dataset>`, for example:

```bash
python train.py --config multiscale_dualviewcnn_femur
```
