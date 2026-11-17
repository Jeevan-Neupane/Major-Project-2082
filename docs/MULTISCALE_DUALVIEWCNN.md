# MultiScaleDualViewCNN

`MultiScaleDualViewCNN` is this repository's second CNN comparator for
biplanar X-ray to 3D binary bone-mask reconstruction.

It is a clean-room segmentation implementation of the multi-scale fusion
architecture family benchmarked by Shakya and Khanal at NeurIPS 2023. It does
not copy their GPL-licensed implementation and it does not claim to reproduce
X2CT-GAN, whose objective is CT-intensity synthesis rather than binary mask
segmentation.

## Architecture

```text
AP X-ray  -> independent 2D pyramid encoder -> 2D decoder --+-> AP 3D lift --+
                                                                    per scale  |-> 3D fusion decoder -> mask logits
LAT X-ray -> independent 2D pyramid encoder -> 2D decoder --+-> LAT 3D lift -+
```

Four coarse-to-fine resolutions are used. At each scale, AP features are
expanded along depth, lateral features are mapped so their horizontal axis
contributes to depth, and both are fused with the upsampled 3D feature from
the previous scale. GroupNorm avoids batch-size-one BatchNorm instability.

## Presets

All presets use Adam (2e-4), ReduceLROnPlateau, 200 maximum epochs, four
scales, and 16 base channels:

| Dataset group | Input X-ray | Output mask | Batch |
| --- | --- | --- | ---: |
| Femur, VerSe19, VerSe20 | 96 x 128 | 96 x 96 x 128 | 2 |
| Hip, Rib, CTPelvic | 128 x 160 | 128 x 128 x 160 | 1 |

Run with, for example:

```bash
python train.py --config multiscale_dualviewcnn_femur
```

## Citation context

The NeurIPS benchmark compares single-scale and multi-scale CNN
encoder-decoders on public femur, hip, rib, and vertebra datasets under a
shared protocol. Cite it for the baseline family and describe this code as an
adaptation for binary 3D segmentation:

> Shakya, M. and Khanal, B. *Benchmarking Encoder-Decoder Architectures for
> Biplanar X-ray to 3D Bone Shape Reconstruction*. NeurIPS 2023 Datasets and
> Benchmarks Track.

Paper: <https://proceedings.neurips.cc/paper_files/paper/2023/hash/412732f172bdd5ad0efde2fafa110700-Paper-Datasets_and_Benchmarks.pdf>
