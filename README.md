# U-Net-Cell-TFM

This repository provides an integrated pipeline to process live-cell microscopy images using Traction Force Microscopy (TFM) and predict traction forces from fluorescence channels using a deep learning model based on U-Net.

The workflow processes bead and reference images to compute displacements and traction maps using Fourier Transform Traction Cytometry (FTTC), stacks them with protein fluorescence channels (e.g., zyxin, actin), and trains a U-Net to infer force distributions from image features.

---

## Repository Structure

```
u-net-cell-tfm/
├── notebooks/                  # Jupyter notebooks (pipeline, training, inference)
│   ├── Full_TFM_UNet_Pipeline.ipynb
│   ├── DataProcessing.ipynb
│   ├── dataset_viewer.ipynb
│   ├── train_unet.ipynb
│   └── load_trained_unet.ipynb
├── tfm/                        # TFM-related Python scripts (image registration, FTTC, etc.)
├── utils/                      # U-Net architecture and dataset utilities
├── example_dataset/           # Output folder for .npy files and dataset.csv
├── input_data/                # Input directory to place cell folders with .tif files
├── requirements.txt           # Minimal required Python packages
├── cellstress_venv.yml        # Full Conda environment (Python 3.7, CUDA 10.2)
└── README.md                  # This file
```

---

## Input Folder Format

Raw microscopy input data must be placed in `input_data/` under subfolders for each cell, like so:

```
input_data/
├── Cell001/
│   ├── beads.tif
│   ├── reference.tif
│   ├── zyxin.tif
│   └── actin.tif
├── Cell002/
│   └── ...
```

Each subfolder must include:
- `beads.tif`: image of substrate under force
- `reference.tif`: image of relaxed substrate (no force)
- `zyxin.tif` and `actin.tif`: fluorescence channels

---

## Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/EdmundTehEs/u-net-cell-tfm.git
cd u-net-cell-tfm
```

### 2. Install the Environment

Option A: Install via Conda (recommended for full reproducibility)

```bash
conda env create -f cellstress_venv.yml
conda activate cellstress_env
```

Option B: Install minimal dependencies using pip

```bash
pip install -r requirements.txt
```

---

## How to Use

### Workflow Summary
1. **Register beads** and compute displacements using scripts in `tfm/`.
2. **Calculate traction maps** with `TFM_calculation` and generate masks.
   The core functions now accept `save_files=False` to return arrays directly
   without writing to disk.
3. **Assemble training stacks** via the notebooks in `notebooks/`.
4. **Train** the U-Net (`train_unet.ipynb`) and **predict** new force maps.

### Option 1: End-to-End in Notebook

Run the notebook:

```bash
notebooks/Full_TFM_UNet_Pipeline.ipynb
```

This will:
- Run TFM preprocessing on all folders in `input_data/`
- Create `.npy` stacks and register them in `example_dataset/dataset.csv`
- Train the U-Net model

### Option 2: Manual Execution

You can also run each step separately:

- Dataset preparation: `notebooks/Full_TFM_UNet_Pipeline.ipynb` (or modify it to generate only the dataset)
- Training: `notebooks/train_unet.ipynb`
- Inference: `notebooks/load_trained_unet.ipynb`

---

## Output Format

Each `.npy` stack contains 8 channels:

```
[ux, uy, fx, fy, mask, forcemask, zyxin, actin]
```

These are used for supervised learning to predict traction forces (`fx`, `fy`) from the other input channels.

The dataset is indexed via `example_dataset/dataset.csv`.

---

## Acknowledgments

This repository was developed as part of a research project in cell mechanics and super-resolution imaging, integrating microscopy image analysis with machine learning techniques. The TFM code was adapted from experimental pipelines in the Mechanobiology Institute (MBI) at the National University of Singapore, with mentorship and guidance from Wang Wei and Professor Tony Kanchanawong.

The deep learning methodology and force prediction pipeline are adapted from the work of:

> Schmitt, M.S., Colen, J., Sala, S., Devany, J., Seetharaman, S., Caillier, A., Gardel, M.L., Oakes, P.W., and Vitelli, V.  
> *Machine learning interpretable models of cell mechanics from protein images.* Cell (2023).  
> https://doi.org/10.1016/j.cell.2023.11.041

Their publication and open-source model implementation provided the foundational structure and conceptual basis for predicting traction force distributions from focal adhesion protein images using U-Net-based architectures.

## License

This project is licensed under the [MIT License](LICENSE).
