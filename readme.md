# Major Project 2082

Welcome to the **Major Project 2082** repository. This project focuses on advanced 3D medical imaging techniques, simulation of X-ray projections, and medical data preparation.

## 📌 Project Overview

The core objective of this project is to bridge the gap between 3D volumetric data (like CT scans) and 2D radiographic imaging. By digitally reconstructing radiographs (DRRs) and exploring X-ray projections, this project aims to support tasks such as medical image alignment, synthetic data generation, and potentially 3D anatomy reconstruction from 2D views.



## 📂 Repository Structure

The project is currently divided into specialized data preparation and visualization modules.

### [Data Preparation](./Data%20Preparation)

This directory contains all the workflows and codebases for processing medical data.

- **[DRR Generation](./Data%20Preparation/DRR%20Generation/readme.md)**  
  A high-performance C++ based pipeline that generates synthetic X-ray images (Digitally Reconstructed Radiographs) from 3D CT scans. It simulates the physics of X-ray projection and supports both `.mhd` and `.nii.gz` formats. Check out its [dedicated README](./Data%20Preparation/DRR%20Generation/readme.md) for build and run instructions.
  
- **[Notebooks](./Data%20Preparation/notebooks)**  
  A collection of Python scripts and Jupyter Notebooks for prototyping and data conversion:
  - `DRR Generation.ipynb`: Prototyping and visualizations for the DRR generation algorithm.
  - `NIIGZ To MHD.ipynb`: Conversion utilities to transform NIfTI compressed files (`.nii.gz`) into MetaImage headers and raw binary data (`.mhd` + `.raw`).
  - `X_ray_to_3d_projection.ipynb`: 3D visual plotting of the X-ray source, detector geometry, and the CT volume using Matplotlib.

## Datasets

- **CT Scans**: 3D medical imaging data used as input for DRR generation. Datasets should be placed in the appropriate data directory as described in the module documentation.
- **DRR Images**: Synthetic X-ray images generated from CT scans. Output location and format are configurable in the DRR Generation module.

## Models

- **DRR Generation Model**: The C++ implementation in `Data Preparation/DRR Generation/` simulates X-ray projections from CT volumes using ray marching and camera models. See the [DRR Generation readme](Data%20Preparation/DRR%20Generation/readme.md) for technical details.
- **Notebooks**: Jupyter notebooks in `Data Preparation/notebooks/` may include or reference machine learning models for further analysis or visualization. Refer to each notebook for model-specific information.

## 🛠️ Technologies Used

- **C/C++ & CMake**: For computationally intensive ray-marching and DRR generation.
- **Python**: For data manipulation, prototyping, and 3D visualization.
- **Jupyter Notebook**: For interactive development and data analysis.
- **Medical Formats Supported**: NIfTI (`.nii`, `.nii.gz`) and MetaImage (`.mhd`, `.raw`).

## 🚀 Getting Started
### Prerequisites
- **C++17** or later
- **CMake** (for building C++ code)
- **Python 3.x** (for notebooks and data conversion)
- **Jupyter Notebook/Lab** (for running notebooks)
- Recommended: Python packages such as `numpy`, `matplotlib`, and `SimpleITK` (see notebook cells for details)

### Building the DRR Generation Module
1. Open a terminal and navigate to the DRR Generation directory:
   ```sh
   cd Data Preparation/DRR Generation
   ```
2. Create a build directory and run CMake:
   ```sh
   mkdir build
   cd build
   cmake ..
   make
   ```
3. The resulting executable will be available in the `build` directory.

### Running the Notebooks
1. Navigate to the notebooks directory:
   ```sh
   cd Data Preparation/notebooks
   ```
2. Launch Jupyter Lab or Notebook:
   ```sh
   jupyter lab
   # or
   jupyter notebook
   ```
3. Open and run the desired notebook (e.g., `DRR Generation.ipynb`).