# 🧬 U-Net Cell Traction Force Prediction

This repository contains a complete pipeline for predicting **cell traction forces** from fluorescence microscopy images using a combination of **Traction Force Microscopy (TFM)** and a deep learning model based on **U-Net**.

---

## 📦 What This Repository Does

### 1. **TFM Preprocessing**
Given raw images of:
- `beads.tif` (fluorescent beads under tension)
- `reference.tif` (relaxed beads after cell removal)
- Protein fluorescence channels (`zyxin.tif`, `actin.tif`)

The TFM pipeline computes:
- **Displacement maps** (`ux`, `uy`)
- **Traction force maps** (`fx`, `fy`)
- **Cell masks** (`mask`, `forcemask`)

### 2. **Dataset Preparation**
All outputs are stacked into an 8-channel `.npy` file per cell:
```
[ux, uy, fx, fy, mask, forcemask, zyxin, actin]
```
These are saved in `example_dataset/` and listed in a `dataset.csv`.

### 3. **U-Net Training**
A U-Net model (`utils/UNeXt.py`) is trained to predict traction force maps from the image stack.

---

## 🗂 Folder Structure

```
u-net-cell-tfm/
│
├── tfm/                      # TFM utilities
├── utils/                    # U-Net model & data loading
├── example_dataset/          # .npy files + dataset.csv
├── input_data/               # Your raw input data (see below)
├── Full_TFM_UNet_Pipeline.ipynb  # Main notebook
├── prepare_dataset.py        # Script version of pipeline
├── train_unet.ipynb
├── load_trained_unet.ipynb
└── README.md
```

---

## 📁 Input Folder Format

Your input images should be organized as:

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

---

## 🚀 How to Run the Pipeline

### 🧪 From Jupyter Notebook

Open:
```bash
Full_TFM_UNet_Pipeline.ipynb
```

This will:
1. Loop through each `input_data/Cell*/` folder
2. Run TFM and create `.npy` files
3. Register them in `example_dataset/dataset.csv`
4. Train U-Net on the result

### ⚙️ From Script

If you prefer CLI:
```bash
python prepare_dataset.py --image_dir ./Cell001 --fluorescence_dir ./Cell001 --output_dir ./example_dataset
```

---

## 📋 Dependencies

Install the environment using:
```bash
conda env create -f environment.yml
conda activate cellstress_env
```

> You can edit `environment.yml` for Python >= 3.8, PyTorch >= 1.12

---

## 📣 Credits

Built using image processing + machine learning pipelines under guidance from Prof. Tony Kanchanawong's lab. Includes work from:
- TFM-main scripts
- Custom U-Net segmentation

---

## 📌 License
MIT License. See LICENSE for more details.
