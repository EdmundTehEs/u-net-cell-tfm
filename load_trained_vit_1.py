#!/usr/bin/env python3
import argparse
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patch
import matplotlib.colors as pltclr

from utils.ViTNet import ViTNet           # assumes ViTNet.py is in PYTHONPATH or same dir
from utils.data_processing import CellDataset, SubsetSampler
from utils.nb_utils import make_vector_field

def main(args):
    # Set device
    device = torch.device('cuda:0' if (args.use_gpu and torch.cuda.is_available()) else 'cpu')
    print(f"Using device: {device}")

    # Load checkpoint
    checkpoint = torch.load(args.model_path, map_location=\'cpu\')

    # Update dataset paths as needed (hardcoded fallback)
    dataset_kwargs = checkpoint[\\'dataset_kwargs\\']
    if args.data_root is not None:
        dataset_kwargs[\\'root\\'] = args.data_root
    test_cells = dataset_kwargs[\\'test_cells\\']

    # Set evaluation crop size (can be overridden)
    dataset_kwargs[\\'transform_kwargs\\'][\\'crop_size\\'] = args.crop_size
    dataset_kwargs[\\'transform_kwargs\\'][\\'rotate\\'] = False

    # Build model
    model_kwargs = checkpoint[\\'model_kwargs\\']
    # Pass device to ViTNet during instantiation
    model = ViTNet(**model_kwargs, model_idx=0, device=device)
    model.load_state_dict(checkpoint[\\'model_state_dict\\'])
    # model = model.to(device) # Redundant if device is handled in __init__
    model.eval()

    # Build dataset and loader
    dataset = CellDataset(**dataset_kwargs)
    sampler = SubsetSampler(np.arange(len(dataset)))
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=1, shuffle=False, sampler=sampler, pin_memory=True
    )

    # Find a test cell/sample
    if args.cell_name is None:
        # Use the first test cell by default
        cell_name = dataset.info.folder.unique()[0]
    else:
        cell_name = args.cell_name

    idxs = np.sort(dataset.info[dataset.info.folder == cell_name].index.values)
    if len(idxs) == 0:
        raise RuntimeError(f"No frames found for cell {cell_name}")
    idx = idxs[args.frame_idx]

    sample = dataset[idx]
    # Only unsqueeze tensors that are model inputs or targets
    # Assuming \'zyxin\', \'output\', \'displacements\', \'mask\' are the relevant tensors
    # You might need to adjust this based on your actual dataset structure
    if \'zyxin\' in sample: sample[\'zyxin\'] = sample[\'zyxin\'].unsqueeze(0).to(device)
    if \'output\' in sample: sample[\'output\'] = sample[\'output\'].unsqueeze(0).to(device)
    if \'displacements\' in sample: sample[\'displacements\'] = sample[\'displacements\'].unsqueeze(0).to(device)
    if \'mask\' in sample: sample[\'mask\'] = sample[\'mask\'].unsqueeze(0).to(device)
    # Inference (multiple runs for robustness)
    preds = []
    for _ in range(5):
        with torch.no_grad():
            pred = model(model.select_inputs(model.input_type, sample)).detach().cpu().numpy().squeeze()
        preds.append(pred)
    pred = np.mean(preds, axis=0)
    target = sample['output'].detach().cpu().numpy().squeeze()
    zyx = sample['zyxin'].detach().cpu().numpy().squeeze()

    # Convert polar (mag, angle) to (fx, fy)
    fxp, fyp = pred[0] * np.cos(pred[1]), pred[0] * np.sin(pred[1])
    fxt, fyt = target[0] * np.cos(target[1]), target[0] * np.sin(target[1])

    # Error map and metrics
    diff = np.sqrt((fxp - fxt) ** 2 + (fyp - fyt) ** 2)
    diff[target[0] < 0.5] = 0

    # Mean absolute error & correlation (excluding background)
    mask = target[0] > 0.5
    mae = np.mean(np.abs(pred[0][mask] - target[0][mask]))
    corr = np.corrcoef(pred[0][mask].flatten(), target[0][mask].flatten())[0, 1]

    print(f"Mean Absolute Error: {mae:.3f}")
    print(f"Pearson Correlation: {corr:.3f}")

    # Plot
    fig, ax = plt.subplots(1, 4, figsize=(16, 4), dpi=200)
    vmax = 3

    # Four panels: input, ground truth, prediction, error
    ax[0].imshow(zyx / zyx.max(), vmax=0.3, vmin=0.0, cmap='gray', origin='lower')
    ax[0].set_title('Input (Zyxin)')
    ax[1].imshow(target[0], vmax=vmax, vmin=0.4, cmap='inferno', origin='lower')
    ax[1].quiver(*make_vector_field(*target, downsample=16, threshold=0.5, angmag=True),
                    color='w', scale=20, width=0.003, alpha=0.8)
    ax[1].set_title('Ground Truth')
    ax[2].imshow(pred[0], vmax=vmax, vmin=0.4, cmap='inferno', origin='lower')
    ax[2].quiver(*make_vector_field(*pred, downsample=16, threshold=0.5, angmag=True),
                    color='w', scale=20, width=0.003, alpha=0.8)
    ax[2].set_title('Prediction')
    ax[3].imshow(diff, vmax=vmax, vmin=0., cmap='Reds', origin='lower')
    ax[3].set_title('Error (|F_pred - F_true|)')

    for a in ax.flat:
        a.set_xticks([])
        a.set_yticks([])
        

    # Add text with metrics to the error panel
    ax[3].text(0.05, 0.95, f"MAE: {mae:.2f}\nCorr: {corr:.2f}",
               color='k', fontsize=12, ha='left', va='top', transform=ax[3].transAxes,
               bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))

    fig.tight_layout()
    os.makedirs(args.output_dir, exist_ok=True)
    outpath = os.path.join(args.output_dir, f"{cell_name}_frame{args.frame_idx}_prediction.png")
    fig.savefig(outpath, bbox_inches='tight')
    print(f"Saved figure to: {outpath}")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load and visualize ViTNet force predictions.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to model.pt checkpoint")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save output figure")
    parser.add_argument("--data_root", type=str, default=None, help="Override data root directory")
    parser.add_argument("--cell_name", type=str, default=None, help="Cell name (default: first in test set)")
    parser.add_argument("--frame_idx", type=int, default=2, help="Frame index within the cell to visualize")
    parser.add_argument("--crop_size", type=int, default=512, help="Image crop size for evaluation")
    parser.add_argument("--use_gpu", action="store_true", help="Use GPU if available")

    args = parser.parse_args()
    main(args)
