import csv
import os
from pathlib import Path

import numpy as np
from skimage import io
import pandas as pd

from utils import data_processing as dp


def create_stacks(input_root: Path, output_root: Path) -> Path:
    npy_files = []
    for folder in sorted(os.listdir(input_root)):
        cell_path = input_root / folder
        if not cell_path.is_dir():
            continue
        ux = io.imread(cell_path / 'disp_u_0.tif')
        uy = io.imread(cell_path / 'disp_v_0.tif')
        fx = io.imread(cell_path / 'fx_0.tif')
        fy = io.imread(cell_path / 'fy_0.tif')
        mask = io.imread(cell_path / 'cellmask.tif')
        fmask = io.imread(cell_path / 'forcemask.tif')
        zyxin = io.imread(cell_path / 'zyxin.tif')
        actin = io.imread(cell_path / 'actin.tif')
        stack = np.stack([ux, uy, fx, fy, mask, fmask, zyxin, actin], axis=0)
        folder_out = output_root / folder
        folder_out.mkdir(exist_ok=True)
        npy_name = f"{folder}.npy"
        np.save(folder_out / npy_name, stack)
        npy_files.append((folder, npy_name))
    csv_path = output_root / 'dataset.csv'
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['folder', 'filename'])
        for folder, name in npy_files:
            writer.writerow([folder, name])
    return csv_path


def test_create_npy_and_csv(minimal_dataset):
    input_root, output_root = minimal_dataset
    csv_path = create_stacks(input_root, output_root)
    assert csv_path.exists()
    df = pd.read_csv(csv_path)
    assert len(df) == 2
    for _, row in df.iterrows():
        npy_file = output_root / row['folder'] / row['filename']
        assert npy_file.exists()
        arr = np.load(npy_file)
        assert arr.shape == (8, 4, 4)


def test_celldataset_loading(minimal_dataset):
    input_root, output_root = minimal_dataset
    create_stacks(input_root, output_root)
    kwargs = dict(
        root=str(output_root),
        force_load=False,
        test_split='bycell',
        test_cells=None,
        in_channels=[6],
        out_channels=[2,3],
        transform_kwargs=dict(
            output_channels=[2,3],
            vector_components=[],
            crop_size=0,
            norm_output={},
            perturb_input={},
            perturb_output={},
            add_noise={},
            magnitude_only=False,
            angmag=False,
            rotate=False,
        ),
        frames_to_keep=10000,
        input_baseline_normalization=None,
        output_baseline_normalization=None,
        validation_split=0.5,
        remake_dataset_csv=False,
        exclude_frames=None,
    )
    dataset = dp.CellDataset(**kwargs)
    assert len(dataset) == 2
    # load first item without mask cropping
    item = dataset.__getitem__(0, mask_crop=False)
    assert 'mask' in item and 'zyxin' in item and 'output' in item
