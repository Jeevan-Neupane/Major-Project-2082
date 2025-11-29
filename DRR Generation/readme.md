# DRRGeneration

Generates Digitally Reconstructed Radiographs (DRRs) from a CT volume stored in MetaImage (.mhd + .raw) format.

## Core Components

- CT volume loading and coordinate transforms: [`CTVolume`](src/CTVolume.cpp) / header [include/CTVolume.h](include/CTVolume.h)
  - [`CTVolume::loadMHD`](src/CTVolume.cpp) parses header, gets spacing, origin, direction.
  - [`CTVolume::loadRawData`](src/CTVolume.cpp) loads voxel data.
  - [`CTVolume::interpolate`](src/CTVolume.cpp) trilinear sampling.
  - [`CTVolume::getWorldBounds`](src/CTVolume.cpp), [`CTVolume::getCenter`](src/CTVolume.cpp).
- X‑ray acquisition geometry: [`XRayCamera`](src/XRayCamera.cpp) / header [include/XRayCamera.h](include/XRayCamera.h)
  - Constants: SOURCE_DISTANCE_FROM_CT, DETECTOR_DISTANCE_FROM_CT, DETECTOR_ROWS, DETECTOR_COLS, DETECTOR_PIXEL_SIZE.
  - [`XRayCamera::setup`](src/XRayCamera.cpp) positions source & detector based on axis (+X, -X, +Y, -Y, +Z, -Z).
  - [`XRayCamera::getPixelWorldPos`](src/XRayCamera.cpp) maps detector pixels to world.
  - [`XRayCamera::printSetupInfo`](src/XRayCamera.cpp) prints geometry.
- Ray marching / attenuation: [`RayMarcher`](src/RayMarcher.cpp) / header [include/RayMarcher.h](include/RayMarcher.h)
  - [`RayMarcher::rayBoxIntersect`](src/RayMarcher.cpp) clips ray to CT bounds.
  - [`RayMarcher::huToMu`](src/RayMarcher.cpp) converts HU to linear attenuation.
  - [`RayMarcher::marchRay`](src/RayMarcher.cpp) accumulates attenuation along sampled points using `stepSize`.
- DRR orchestration: [`DRRGenerator`](src/DRRGenerator.cpp) / header [include/DRRGenerator.h](include/DRRGenerator.h)
  - [`DRRGenerator::generate`](src/DRRGenerator.cpp) loops pixels, launches rays.
  - [`DRRGenerator::postProcess`](src/DRRGenerator.cpp) normalizes / optional inversion (see `INVERT_INTENSITY`).
  - [`DRRGenerator::saveAsPNG`](src/DRRGenerator.cpp) writes PNG via stb: [libs/stb_image_write.h](libs/stb_image_write.h), implementation [src/stb_image_write_impl.cpp](src/stb_image_write_impl.cpp).

## Entry Point

`main`: [src/main.cpp](src/main.cpp)
- Directory helpers: [`ensureDirectoryExists`](src/main.cpp), [`getFilenameStem`](src/main.cpp).
- View generation helper: [`generateView`](src/main.cpp).
- Generates either:
  - Two default views: AP (-Y) and Lat (+X), or
  - Single specified axis.

Constants in `main`:
- `RAY_STEP_SIZE = 0.0f` → auto uses min voxel spacing.
- `OUTPUT_PNG = true` → enables PNG export.

## Input Data

Place .mhd + .raw pairs under e.g. [mhd_files/patient_01/patient_01.mhd](mhd_files/patient_01/patient_01.mhd). Header fields used:
- `DimSize` → nx, ny, nz
- `ElementSpacing` → voxel size
- `Offset` / `TransformMatrix` → origin & orientation (stored into direction matrices)
- `ElementType` → voxel type (e.g. MET_SHORT)
- `ElementDataFile` → raw file name

## Pipeline Summary

1. Parse MHD: [`CTVolume::loadMHD`](src/CTVolume.cpp) then raw via [`CTVolume::loadRawData`](src/CTVolume.cpp).
2. Compute world bounds: [`CTVolume::getWorldBounds`](src/CTVolume.cpp).
3. Configure camera: [`XRayCamera::setup`](src/XRayCamera.cpp) for chosen axis.
4. For each detector pixel:
   - Obtain world position: [`XRayCamera::getPixelWorldPos`](src/XRayCamera.cpp).
   - Form ray to source, clip: [`RayMarcher::rayBoxIntersect`](src/RayMarcher.cpp).
   - Sample HU → μ: [`RayMarcher::huToMu`](src/RayMarcher.cpp).
   - Accumulate attenuation: [`RayMarcher::marchRay`](src/RayMarcher.cpp).
5. Normalize image: [`DRRGenerator::postProcess`](src/DRRGenerator.cpp).
6. Save PNG: [`DRRGenerator::saveAsPNG`](src/DRRGenerator.cpp) if enabled.

## Build

Requires CMake ≥ 3.10, C++17 (set in [CMakeLists.txt](CMakeLists.txt)).

```sh
mkdir build
cd build
cmake ..
make
```

Output executable: `build/drr_gen`.

## Run

Default (generate AP and Lat):
```sh
./build/drr_gen mhd_files/patient_01/patient_01.mhd
```

Single axis (e.g. +Z):
```sh
./build/drr_gen mhd_files/patient_01/patient_01.mhd +Z
```

Valid axes: +X -X +Y -Y +Z -Z.

## Output

PNG(s) written to:
```
outputs/<stem>/AP.png
outputs/<stem>/Lat.png
```
(or `<axis>.png` if single view). Directory creation via [`ensureDirectoryExists`](src/main.cpp).

## Adjustments

- Change detector resolution / geometry: edit constants in [src/XRayCamera.cpp](src/XRayCamera.cpp).
- Invert intensity: toggle `INVERT_INTENSITY` in [src/DRRGenerator.cpp](src/DRRGenerator.cpp).
- Force ray step size: set `RAY_STEP_SIZE` in [src/main.cpp](src/main.cpp) (>0 overrides auto).
- Extend preprocessing: modify [`DRRGenerator::postProcess`](src/DRRGenerator.cpp).

## Notes

- HU to μ conversion currently linear and water‑based in [`RayMarcher::huToMu`](src/RayMarcher.cpp).
- Interpolation assumes data loaded into `CTVolume::data` as floats.
- Direction matrix supports non‑identity orientations from MHD header.

## File Index

- Build script: [CMakeLists.txt](CMakeLists.txt)
- Core sources: [src/CTVolume.cpp](src/CTVolume.cpp), [src/XRayCamera.cpp](src/XRayCamera.cpp), [src/RayMarcher.cpp](src/RayMarcher.cpp), [src/DRRGenerator.cpp](src/DRRGenerator.cpp), [src/main.cpp](src/main.cpp)
- Headers: [include/CTVolume.h](include/CTVolume.h), [include/XRayCamera.h](include/XRayCamera.h), [include/RayMarcher.h](include/RayMarcher.h), [include/DRRGenerator.h](include/DRRGenerator.h), [include/MathUtils.h](include/MathUtils.h)
- Image writing: [libs/stb_image_write.h](libs/stb_image_write.h), [src/stb_image_write_impl.cpp](src/stb_image_write_impl.cpp)