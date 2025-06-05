import os
import glob
import argparse
import numpy as np
import pandas as pd
import skimage.io as io

# Import TFM utilities
from tfm.TFM_Image_registration import TFM_Image_registration
from tfm.TFM_displacement_tools import TFM_optical_flow
from tfm.TFM_tools import TFM_calculation, cellmask_threshold


def run_tfm(cell_dir):
    """Run the TFM pipeline inside ``cell_dir``."""
    cwd = os.getcwd()
    os.chdir(cell_dir)
    try:
        # Image registration and flat field correction
        TFM_Image_registration()
        # Compute bead displacements
        TFM_optical_flow()
        # Calculate traction forces
        TFM_calculation()
        # Generate masks from fluorescence channel
        cellmask_threshold('zyxin.tif')
    finally:
        os.chdir(cwd)


def stack_npy(cell_dir, out_dir):
    """Stack TFM outputs into ``.npy`` dataset files."""
    disp_u = sorted(glob.glob(os.path.join(cell_dir, 'displacement_files', 'disp_u*.tif')))
    disp_v = sorted(glob.glob(os.path.join(cell_dir, 'displacement_files', 'disp_v*.tif')))
    fx = sorted(glob.glob(os.path.join(cell_dir, 'traction_files', 'fx_*.tif')))
    fy = sorted(glob.glob(os.path.join(cell_dir, 'traction_files', 'fy_*.tif')))
    mask = io.imread(os.path.join(cell_dir, 'cellmask.tif'))
    fmask = io.imread(os.path.join(cell_dir, 'forcemask.tif'))
    zyxin = io.imread(os.path.join(cell_dir, 'zyxin.tif'))
    actin = io.imread(os.path.join(cell_dir, 'actin.tif'))

    os.makedirs(out_dir, exist_ok=True)

    frames = len(disp_u)
    rows = []
    for i in range(frames):
        ux = io.imread(disp_u[i])
        uy = io.imread(disp_v[i])
        fx_i = io.imread(fx[i])
        fy_i = io.imread(fy[i])
        channels = [ux, uy, fx_i, fy_i]
        if mask.ndim == 3:
            channels.append(mask[i])
            channels.append(fmask[i])
            channels.append(zyxin[i])
            channels.append(actin[i])
        else:
            channels.append(mask)
            channels.append(fmask)
            channels.append(zyxin)
            channels.append(actin)
        arr = np.stack(channels)
        fname = f'frame_{i}.npy'
        np.save(os.path.join(out_dir, fname), arr)
        rows.append({'folder': os.path.basename(out_dir), 'filename': fname})

    return rows


def main(raw_root, output_root):
    cell_dirs = [os.path.join(raw_root, d) for d in os.listdir(raw_root)
                 if os.path.isdir(os.path.join(raw_root, d))]
    dataset_rows = []
    for cell in cell_dirs:
        print('Processing', cell)
        run_tfm(cell)
        out_dir = os.path.join(output_root, os.path.basename(cell))
        rows = stack_npy(cell, out_dir)
        dataset_rows.extend(rows)

    df = pd.DataFrame(dataset_rows)
    df.to_csv(os.path.join(output_root, 'dataset.csv'), index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Prepare dataset from raw SDCM images')
    parser.add_argument('input_dir', help='Folder with raw cell subdirectories')
    parser.add_argument('output_dir', help='Destination dataset directory')
    args = parser.parse_args()
    main(args.input_dir, args.output_dir)
