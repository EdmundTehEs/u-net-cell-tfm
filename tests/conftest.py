import os
import numpy as np
from skimage import io
import pytest

@pytest.fixture
def minimal_dataset(tmp_path):
    input_root = tmp_path / "input"
    output_root = tmp_path / "dataset"
    input_root.mkdir()
    output_root.mkdir()
    channels = ['fx_0.tif','fy_0.tif','disp_u_0.tif','disp_v_0.tif',
                'cellmask.tif','forcemask.tif','zyxin.tif','actin.tif']
    for cell in ['Cell001','Cell002']:
        cell_dir = input_root / cell
        cell_dir.mkdir()
        for ch in channels:
            arr = np.random.randint(0, 256, size=(4,4), dtype=np.uint8)
            io.imsave(str(cell_dir/ch), arr)
    return input_root, output_root
