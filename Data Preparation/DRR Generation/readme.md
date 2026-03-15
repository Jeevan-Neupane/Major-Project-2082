# DRR Generation — Beginner's Complete Guide

> **What is a DRR?**
> A **Digitally Reconstructed Radiograph (DRR)** is a synthetic X-ray image generated from a 3D CT scan using a computer. Instead of exposing a patient to real X-rays again, we simulate the physics of X-ray projection mathematically. The result looks like a traditional X-ray but is computed entirely from existing CT data.

---

## Table of Contents

1. [What This Project Does](#1-what-this-project-does)
2. [Background — CT Scans, X-rays, and DRRs](#2-background--ct-scans-x-rays-and-drrs)
3. [Prerequisites and Libraries](#3-prerequisites-and-libraries)
4. [Build Instructions](#4-build-instructions)
5. [Run Instructions](#5-run-instructions)
6. [Pipeline — How It Works Step by Step](#6-pipeline--how-it-works-step-by-step)
7. [File Formats Explained](#7-file-formats-explained)
8. [Orientation and Axis Convention](#8-orientation-and-axis-convention)
9. [Reading the Console Output](#9-reading-the-console-output)
10. [Troubleshooting](#10-troubleshooting)
11. [Tunable Parameters](#11-tunable-parameters)
12. [Code Map](#12-code-map)

---

## 1) What This Project Does

This project takes a **3D CT scan** (a stack of medical images) and produces **2D X-ray-like images** from it — called DRRs.

Supported input formats:

- **MetaImage**: `.mhd` + `.raw` (a text header + binary data pair)
- **NIfTI**: `.nii` or `.nii.gz` (a single file, possibly compressed)

What happens internally, in plain English:

1. Load the 3D CT volume (all the voxel data and its physical measurements).
2. Set up a virtual X-ray machine — a source (the "bulb") and a detector (the "film").
3. For every pixel on the detector, shoot a ray from the source through the CT volume.
4. Along each ray, read CT values, convert them to how much they block X-rays, and sum it up.
5. Normalize the result and save it as a PNG image.

Default output: two views — **AP** (front-to-back) and **Lateral** (side-to-side).

<!-- IMAGE PLACEHOLDER: diagram showing a 3D CT volume with two arrows — one going front-to-back (AP) and one going left-to-right (Lateral), each producing a flat 2D image -->
![AP and Lateral DRR views](docs/images/ap_lat_overview.png)

---

## 2) Background — CT Scans, X-rays, and DRRs

### What is a CT scan?

A CT (Computed Tomography) scan is a 3D grid of numbers. Each tiny cube in the grid is called a **voxel** (like a 3D pixel). Each voxel stores a number called a **Hounsfield Unit (HU)** that tells you what kind of tissue is there:

| HU Range | Tissue |
|----------|--------|
| ~-1000   | Air    |
| ~-100    | Fat    |
| ~0       | Water / Soft tissue |
| ~400–1000 | Bone  |

<!-- IMAGE PLACEHOLDER: a grayscale CT slice showing different tissues labeled — air (black), soft tissue (gray), bone (white) -->
![CT slice with tissue labels](docs/images/ct_slice_tissues.png)

### What is an X-ray?

An X-ray machine fires radiation through the body. Dense materials (like bone) absorb more radiation and appear **white** on the film. Soft tissue absorbs less and appears **gray**. Air absorbs almost nothing and appears **black**.

The physics behind this is called the **Beer-Lambert Law**:

$$
I = I_0 \cdot \exp\!\left(-\int \mu(s)\,ds\right)
$$

- `I₀` = initial X-ray intensity
- `I` = intensity after passing through tissue
- `μ(s)` = how strongly the tissue at position `s` absorbs X-rays (attenuation coefficient)
- The integral sums up all the absorption along the ray path

<!-- IMAGE PLACEHOLDER: diagram of Beer-Lambert law — a horizontal arrow (X-ray beam) passing through a block of material, with I₀ on the left and I on the right, and the formula below -->
![Beer-Lambert Law diagram](docs/images/beer_lambert.png)

### What is a DRR?

A DRR simulates this process digitally. Instead of real X-rays, we:

1. Use the CT HU values to estimate `μ` (attenuation) at every point.
2. Cast virtual rays through the CT volume.
3. Sum up the attenuation along each ray.
4. Map the result to a grayscale image.

The result looks like an X-ray but is computed entirely from the CT data — no additional radiation needed.

<!-- IMAGE PLACEHOLDER: side-by-side comparison — left: real X-ray of a chest, right: DRR generated from CT of the same patient -->
![Real X-ray vs DRR comparison](docs/images/xray_vs_drr.png)

---

## 3) Prerequisites and Libraries

### What you need installed

| Tool | Why |
|------|-----|
| C++17 compiler (GCC/G++) | Compiles the source code |
| CMake >= 3.10 | Build system that generates the Makefile |
| Zlib development library | Reads `.nii.gz` compressed files |

### Why Zlib?

`.nii.gz` files are NIfTI files compressed with gzip. The loader in [src/CTVolume.cpp](src/CTVolume.cpp) uses Zlib functions (`gzopen`, `gzread`) to decompress and read the voxel data on the fly — without needing to manually unzip the file first.

### Install everything on Ubuntu / WSL (one command)

```bash
sudo apt update && sudo apt install -y build-essential cmake zlib1g-dev
```

- `build-essential` → installs GCC/G++ and Make
- `cmake` → installs CMake
- `zlib1g-dev` → installs Zlib headers and libraries

---

## 4) Build Instructions

> **What is building?** "Building" means turning the human-readable C++ source code into a binary program your computer can actually run.

### Step-by-step

```bash
# Step 1: Go to the project root folder
cd drr-generation

# Step 2: Create a separate folder for build files (keeps things clean)
mkdir -p build

# Step 3: Enter the build folder
cd build

# Step 4: Let CMake read CMakeLists.txt and prepare the build
cmake ..

# Step 5: Compile the code (use all CPU cores with -j)
make -j
```

After a successful build, you will have:

```
build/drr_gen
```

This is the executable program.

<!-- IMAGE PLACEHOLDER: screenshot of a terminal showing successful cmake .. and make -j output, with the final line showing "drr_gen" being linked -->
![Successful build terminal output](docs/images/build_success.png)

### What CMake does

CMake reads [CMakeLists.txt](CMakeLists.txt) and:

- Detects your compiler
- Finds Zlib on your system
- Generates a `Makefile`
- `make` then uses that `Makefile` to compile all `.cpp` files and link them into `drr_gen`

---

## 5) Run Instructions

### A) Generate default AP + Lateral views

For a `.mhd` file:

```bash
./build/drr_gen mhd_files/patient_01/patient_01.mhd
```

For a `.nii.gz` file:

```bash
./build/drr_gen patient_02.nii.gz
```

This generates two PNG files — one AP view and one Lateral view.

### B) Generate a single custom axis view

```bash
./build/drr_gen patient_02.nii.gz +Z
```

Valid axis options:

| Axis | Meaning |
|------|---------|
| `+X` | Ray travels in the positive X direction |
| `-X` | Ray travels in the negative X direction |
| `+Y` | Ray travels in the positive Y direction |
| `-Y` | Ray travels in the negative Y direction |
| `+Z` | Ray travels in the positive Z direction (top-down) |
| `-Z` | Ray travels in the negative Z direction (bottom-up) |

<!-- IMAGE PLACEHOLDER: diagram of a 3D coordinate system (X, Y, Z axes) with a CT volume in the center and arrows showing the 6 possible ray directions -->
![6 axis directions diagram](docs/images/axis_directions.png)

### Output location

All PNG files are saved under:

```
outputs/<input_filename_without_extension>/
```

Examples:

```
outputs/patient_01/AP.png
outputs/patient_01/Lat.png
outputs/patient_02/+Z.png
```

---

## 6) Pipeline — How It Works Step by Step

This section walks through exactly what happens inside the program from the moment you run it to when the PNG is saved.

<!-- IMAGE PLACEHOLDER: flowchart with boxes: "Load CT Volume" → "Setup Camera Geometry" → "For each pixel: cast ray" → "March ray through volume" → "Accumulate attenuation" → "Normalize" → "Save PNG" -->
![DRR pipeline flowchart](docs/images/pipeline_flowchart.png)

---

### Step 1: Volume Loading

**File:** [src/main.cpp](src/main.cpp) calls `CTVolume::loadVolume(...)` in [src/CTVolume.cpp](src/CTVolume.cpp)

The program first reads the CT file from disk. Based on the file extension:

- `.mhd` → calls `loadMHD(...)` to read the text header, then `loadRawData(...)` to read the binary voxel data
- `.nii` / `.nii.gz` → calls `loadNIfTI(...)` which reads the header and voxel data from one file (decompressing if needed)

After loading, the volume stores:

- `nx, ny, nz` — number of voxels in each direction
- Voxel spacing in mm (physical size of each voxel)
- Origin — where in 3D space the volume starts
- Direction matrix — how the volume is oriented in space

> **Analogy:** Think of the CT volume as a 3D grid of tiny boxes. Each box has a known size (spacing) and position (origin + direction). Loading reads all the numbers inside those boxes into memory.

<!-- IMAGE PLACEHOLDER: diagram of a 3D voxel grid with labels showing nx/ny/nz dimensions, spacing arrows between voxels, and an origin point at one corner -->
![CT volume voxel grid](docs/images/voxel_grid.png)

---

### Step 2: Spatial Transform Setup

**File:** [src/CTVolume.cpp](src/CTVolume.cpp)

CT data is stored as a grid of integer indices `(i, j, k)`. But rays travel in real-world 3D space measured in millimeters. We need to convert between the two.

Two transforms are set up:

- **Index → World** (`indexToWorld`): converts a voxel grid position like `(100, 200, 50)` into a real-world coordinate like `(45.5 mm, 102.0 mm, 25.0 mm)`
- **World → Index** (`worldToIndex`): the reverse — converts a world position back to a grid index so we can look up the voxel value

> **Why this matters:** When a ray is marching through space in mm, we need to know which voxel it's currently inside to read the CT value there.

<!-- IMAGE PLACEHOLDER: diagram showing a voxel grid on the left with index coordinates (i,j,k) and a world space on the right with mm coordinates (x,y,z), with a two-headed arrow labeled "indexToWorld / worldToIndex" between them -->
![Index to world transform](docs/images/index_world_transform.png)

---

### Step 3: World Bounding Box

**File:** [src/CTVolume.cpp](src/CTVolume.cpp) — `getWorldBounds(...)`

Before casting rays, the program needs to know the physical extent of the CT volume in world space — i.e., where does the volume start and end in mm?

It does this by transforming all 8 corners of the voxel grid into world coordinates and finding the min/max in each direction. This gives an **Axis-Aligned Bounding Box (AABB)**.

> **Why this matters:** When a ray is cast, we first check if it even intersects this box. If it doesn't, we skip it immediately (no computation needed). If it does, we know exactly where the ray enters and exits the volume.

<!-- IMAGE PLACEHOLDER: diagram of a 3D box (the AABB) with 8 labeled corners, and a ray arrow that enters one face and exits another face -->
![AABB bounding box with ray intersection](docs/images/aabb_ray.png)

---

### Step 4: Camera Geometry Setup

**File:** [src/XRayCamera.cpp](src/XRayCamera.cpp)

This sets up the virtual X-ray machine. The geometry is:

```
[X-ray Source]  ----ray---->  [CT Volume]  ----ray---->  [Detector]
    4000 mm away from center              2000 mm away from center
```

Fixed constants:

| Parameter | Value |
|-----------|-------|
| Source distance from CT center | 4000 mm |
| Detector distance from CT center | 2000 mm |
| Detector size | 1024 × 1024 pixels |
| Pixel size | 1 mm |

`XRayCamera::setup(...)` positions the source and detector based on which axis you're projecting along, and sets up two direction vectors (`uDir`, `vDir`) that define the horizontal and vertical directions of the detector plane.

<!-- IMAGE PLACEHOLDER: side-view diagram showing the X-ray source on the left, the CT volume (box) in the center, and the detector panel on the right. Label the 4000mm and 2000mm distances. Show a single ray going from source through the volume to a pixel on the detector. -->
![X-ray camera geometry](docs/images/camera_geometry.png)

---

### Step 5: Ray Casting

**File:** [src/DRRGenerator.cpp](src/DRRGenerator.cpp) — `DRRGenerator::generate()`

For every pixel `(u, v)` on the 1024×1024 detector:

1. Compute the 3D world position of that detector pixel.
2. Form a ray: starting at the source, pointing toward that detector pixel.
3. Pass the ray to `RayMarcher::marchRay(...)`.
4. Store the returned attenuation value in the output image at `(u, v)`.

> **Analogy:** Imagine standing at the X-ray source and looking through a 1024×1024 grid of tiny windows. For each window, you ask: "how much stuff did the X-ray pass through to reach this window?" That's one ray cast.

<!-- IMAGE PLACEHOLDER: diagram of the detector grid (1024x1024) with one pixel highlighted, and a line drawn from the source through that pixel into the CT volume -->
![Ray casting per detector pixel](docs/images/ray_casting.png)

---

### Step 6: HU to Attenuation Conversion

**File:** [src/RayMarcher.cpp](src/RayMarcher.cpp) — `RayMarcher::huToMu(...)`

As the ray marches through the volume, at each sample point it reads a CT HU value and converts it to an attenuation coefficient `μ`:

- If `HU ≤ 200` (air, fat, soft tissue) → `μ = 0` (transparent to X-rays in this simplified model)
- If `HU > 200` (bone, dense tissue) → linear mapping:

$$
\mu = \mu_{water} \cdot \left(1 + \frac{HU}{1000}\right), \quad \mu_{water} = 0.022
$$

> **Why threshold at 200?** This is a simplification to make bones stand out clearly in the DRR. Soft tissue contributes very little to real X-ray contrast, so we ignore it below HU=200.

<!-- IMAGE PLACEHOLDER: graph with HU on the x-axis (from -1000 to 1000) and μ on the y-axis. Flat line at μ=0 from -1000 to 200, then a rising line from HU=200 onward. Label the threshold at HU=200 and the slope. -->
![HU to mu mapping graph](docs/images/hu_to_mu_graph.png)

---

### Step 7: Ray Marching (Sampling Along the Ray)

**File:** [src/RayMarcher.cpp](src/RayMarcher.cpp) — `marchRay(...)`

This is the core computation. For a single ray:

1. **Clip to AABB**: find where the ray enters and exits the CT volume using `rayBoxIntersect`. If the ray misses the volume entirely, return 0.
2. **Step along the ray**: starting from the entry point, move in small steps (`stepSize`) toward the exit point.
3. **At each step**:
   - Convert the current 3D world position to a voxel index.
   - Use **trilinear interpolation** (`CTVolume::interpolate`) to get a smooth HU value (avoids blocky artifacts from reading the nearest voxel).
   - Convert HU to `μ` using `huToMu(...)`.
   - Accumulate the `μ` value.
4. Return the average accumulated attenuation.

> **What is trilinear interpolation?** Instead of snapping to the nearest voxel, we blend the values of the 8 surrounding voxels based on how close we are to each one. This gives a smoother result.

> **What is step size?** If set to `0`, it defaults to the smallest voxel spacing (e.g., if voxels are 1mm × 1mm × 2mm, step size = 1mm). Smaller steps = more accurate but slower.

<!-- IMAGE PLACEHOLDER: diagram of a ray entering a voxel grid, with dots along the ray showing sample points. One sample point is highlighted with the 8 surrounding voxels shown, illustrating trilinear interpolation. -->
![Ray marching with trilinear interpolation](docs/images/ray_marching_trilinear.png)

---

### Step 8: Post-Processing and Saving

**File:** [src/DRRGenerator.cpp](src/DRRGenerator.cpp) — `postProcess()` and `saveAsPNG(...)`

After all rays are cast, the 2D result is a floating-point image. Two steps finalize it:

1. **Min-max normalization**: scales all values to the range `[0, 1]`
   - The pixel with the lowest attenuation becomes 0 (black)
   - The pixel with the highest attenuation becomes 1 (white)

2. **Optional inversion** (controlled by `INVERT_INTENSITY`): flips black and white so bones appear white on a dark background — matching the look of a real X-ray.

3. **PNG export**: uses the `stb_image_write` library ([libs/stb_image_write.h](libs/stb_image_write.h)) to write a grayscale PNG file.

<!-- IMAGE PLACEHOLDER: three small images side by side — left: raw floating point output (may look washed out), center: after normalization (full contrast), right: after inversion (bone=white, background=black, like a real X-ray) -->
![Post-processing steps](docs/images/postprocessing_steps.png)

---

## 7) File Formats Explained

### CT Intensity — Hounsfield Units (HU)

Every voxel in a CT scan stores a number in **Hounsfield Units**:

$$
HU = 1000 \cdot \frac{\mu - \mu_{water}}{\mu_{water}}
$$

- `μ` = linear attenuation coefficient of the tissue
- `μ_water` = attenuation of water (reference)
- Water = 0 HU, Air = -1000 HU, Dense bone = +1000 HU

<!-- IMAGE PLACEHOLDER: vertical bar (like a color scale) showing HU values from -1000 (black, air) through 0 (gray, water/soft tissue) to +1000 (white, bone), with tissue labels on the side -->
![Hounsfield Unit scale](docs/images/hu_scale.png)

---

### `.mhd + .raw` (MetaImage format)

This is a two-file format:

- `.mhd` — a plain text file containing metadata (open it in any text editor)
- `.raw` — a binary file containing the actual voxel numbers

Example `.mhd` content:

```
ObjectType = Image
NDims = 3
DimSize = 512 512 300
ElementSpacing = 0.7 0.7 1.25
Offset = -180.0 -180.0 -375.0
TransformMatrix = 1 0 0 0 1 0 0 0 1
ElementType = MET_SHORT
ElementDataFile = patient_01.raw
```

Fields used by this project:

| Field | Meaning |
|-------|---------|
| `DimSize` | Number of voxels in X, Y, Z |
| `ElementSpacing` | Physical size of each voxel in mm |
| `Offset` / `Origin` | World position of the first voxel |
| `TransformMatrix` / `Orientation` | How the volume is rotated in space |
| `ElementType` | Data type (e.g., `MET_SHORT` = 16-bit integer) |
| `ElementDataFile` | Name of the `.raw` binary file |

---

### `.nii` / `.nii.gz` (NIfTI format)

NIfTI packs everything into one file:

- `.nii` — uncompressed (header + voxel data in one file)
- `.nii.gz` — gzip-compressed version (smaller file, needs Zlib to read)

The loader in this project supports:

- NIfTI-1 header parsing (`sizeof_hdr = 348`)
- Multiple data types (INT16, FLOAT32, etc.) converted to float
- Voxel spacing from `pixdim[1..3]`
- Intensity scaling via `scl_slope` and `scl_inter` (applied as: `value = slope × raw + intercept`)
- Physical orientation and origin from the `sform` matrix (when `sform_code > 0`)

> **Note:** NIfTI-2 (`sizeof_hdr = 540`) is detected but not fully supported in the current loader.

<!-- IMAGE PLACEHOLDER: diagram showing the structure of a .nii file — a header block at the top (labeled "348 bytes") and a voxel data block below it, with arrows pointing to key fields like pixdim, sform_matrix, scl_slope -->
![NIfTI file structure](docs/images/nifti_structure.png)

---

## 8) Orientation and Axis Convention

In [src/main.cpp](src/main.cpp), the default AP and Lateral source axes are chosen differently depending on the file format:

| Format | AP axis | Lateral axis |
|--------|---------|--------------|
| `.mhd` | `-Y` | `+X` |
| `.nii` / `.nii.gz` | `+Y` | `-X` |

**Why the difference?** MHD and NIfTI files often store the CT volume with different orientation conventions (e.g., RAS vs LPS coordinate systems). This axis flip is a practical correction to ensure the generated DRR looks anatomically correct (head up, left-right not mirrored) for typical datasets in each format.

<!-- IMAGE PLACEHOLDER: two diagrams side by side — left: MHD coordinate system with AP=-Y and Lat=+X labeled on a body outline, right: NIfTI coordinate system with AP=+Y and Lat=-X labeled. Show how both produce the same anatomical view despite different axis signs. -->
![Orientation convention for MHD vs NIfTI](docs/images/orientation_convention.png)

---

## 9) Reading the Console Output

When you run the program, it prints geometry information before starting ray casting. Here is what each part means:

```
Source position: (-4000, 0, 0)       ← where the virtual X-ray bulb is (in mm)
Detector center: (2000, 0, 0)        ← center of the virtual detector panel
Detector uDir: (0, 1, 0)             ← horizontal direction of the detector
Detector vDir: (0, 0, 1)             ← vertical direction of the detector
Corner (0,0): (2000, -512, -512)     ← world position of detector corner pixel
Corner (1023,1023): (2000, 511, 511) ← world position of opposite corner pixel
```

If you see this output followed by the program running without errors, the geometry is configured correctly.

Common messages and what they mean:

| Message | Meaning |
|---------|---------|
| `Loaded volume: 512 x 512 x 300` | CT volume dimensions read successfully |
| `Spacing: 0.70 x 0.70 x 1.25 mm` | Physical voxel size |
| `Generating AP view...` | Starting ray casting for the front-to-back projection |
| `Saved: outputs/patient_02/AP.png` | DRR successfully written to disk |

---

## 10) Troubleshooting

### `camke: command not found`

This is a typo. Use:

```bash
cmake ..
```

### `cmake: command not found`

CMake is not installed. Fix:

```bash
sudo apt install -y cmake
```

### `Zlib not found` during CMake

The Zlib development headers are missing. Fix:

```bash
sudo apt install -y zlib1g-dev
```

### `Unsupported file extension`

The input file must end in `.mhd`, `.nii`, or `.nii.gz`. Check your filename carefully.

### `Invalid axis argument`

The second argument must be exactly one of: `+X`, `-X`, `+Y`, `-Y`, `+Z`, `-Z` (case-sensitive, with the `+` or `-` sign).

### Output PNG is completely black or white

This usually means the HU values in the volume are outside the expected range, or the volume loaded incorrectly. Check:

- That the `.raw` file is in the same folder as the `.mhd` file
- That the NIfTI file is not corrupted (try opening it in a viewer like ITK-SNAP)

### DRR looks anatomically flipped

Try the opposite axis sign. For example, if `+Y` gives a flipped image, try `-Y`.

---

## 11) Tunable Parameters

You can adjust these values to change the output:

| What to change | Where | Parameter |
|----------------|-------|-----------|
| Ray step size (accuracy vs speed) | [src/main.cpp](src/main.cpp) | `stepSize` (0 = auto) |
| Source/detector distance | [src/XRayCamera.cpp](src/XRayCamera.cpp) | `SOURCE_DIST`, `DETECTOR_DIST` |
| Detector resolution | [src/XRayCamera.cpp](src/XRayCamera.cpp) | `DETECTOR_WIDTH`, `DETECTOR_HEIGHT` |
| Pixel size in mm | [src/XRayCamera.cpp](src/XRayCamera.cpp) | `PIXEL_SIZE` |
| HU threshold for bone | [src/RayMarcher.cpp](src/RayMarcher.cpp) | `HU_THRESHOLD` (currently 200) |
| Water attenuation constant | [src/RayMarcher.cpp](src/RayMarcher.cpp) | `MU_WATER` (currently 0.022) |
| Invert output (bone=white vs bone=dark) | [src/DRRGenerator.cpp](src/DRRGenerator.cpp) | `INVERT_INTENSITY` |

---

## 12) Code Map

```
drr-generation/
├── CMakeLists.txt                  ← Build configuration
├── src/
│   ├── main.cpp                    ← Entry point: parses args, loads volume, runs pipeline
│   ├── CTVolume.cpp                ← Loads MHD/NIfTI, stores voxels + spatial transforms
│   ├── XRayCamera.cpp              ← Sets up source/detector geometry per axis
│   ├── RayMarcher.cpp              ← Marches rays, HU→μ conversion, trilinear interpolation
│   ├── DRRGenerator.cpp            ← Orchestrates ray casting, normalization, PNG export
│   └── stb_image_write_impl.cpp    ← Compiles the stb PNG writer
├── include/
│   ├── CTVolume.h
│   ├── XRayCamera.h
│   ├── RayMarcher.h
│   ├── DRRGenerator.h
│   └── MathUtils.h                 ← Vector/matrix math utilities
└── libs/
    └── stb_image_write.h           ← Single-header PNG writing library
```

### Data flow summary

```
main.cpp
  └─ CTVolume::loadVolume()         reads CT file → voxel array + spatial metadata
  └─ XRayCamera::setup()            positions source + detector for chosen axis
  └─ DRRGenerator::generate()
       └─ for each detector pixel:
            └─ RayMarcher::marchRay()
                 └─ rayBoxIntersect()          clips ray to volume AABB
                 └─ CTVolume::interpolate()    trilinear sample at each step
                 └─ huToMu()                   converts HU → attenuation
  └─ DRRGenerator::postProcess()    normalize + optional invert
  └─ saveAsPNG()                    write grayscale PNG via stb
```
