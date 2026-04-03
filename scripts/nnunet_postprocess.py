"""
nnU-Net Post-processing: Connected Component + Volume Threshold
Improve Dice from 0.367 by removing small false-positive blobs.
"""

import nibabel as nib
import numpy as np
from scipy import ndimage
from pathlib import Path

pred_dir = Path("F:/ADDS/nnUNet_predictions/fold_0_validation")
label_dir = Path("F:/ADDS/nnUNet_raw/Dataset011_ColonMasked/labelsTr")

pred_ids = ['colon_004','colon_013','colon_016','colon_019',
            'colon_024','colon_025','colon_033','colon_039']


def compute_dice(pred, label):
    """Compute Dice score."""
    pred_bin = (pred > 0).astype(float)
    label_bin = (label > 0).astype(float)
    intersection = np.sum(pred_bin * label_bin)
    union = np.sum(pred_bin) + np.sum(label_bin)
    if union == 0:
        return 1.0 if np.sum(pred_bin) == 0 else 0.0
    return 2.0 * intersection / union


def postprocess_prediction(pred, min_volume=500):
    """
    Post-processing:
    1. Connected component analysis
    2. Keep only largest component (and components > min_volume)
    3. Binary morphological closing to fill holes
    """
    pred_bin = (pred > 0).astype(np.uint8)
    
    # Connected components
    labeled, n_components = ndimage.label(pred_bin)
    
    if n_components == 0:
        return pred_bin
    
    # Get component sizes
    sizes = ndimage.sum(pred_bin, labeled, range(1, n_components + 1))
    
    # Keep largest component + any > min_volume
    largest_idx = np.argmax(sizes) + 1
    largest_size = sizes[largest_idx - 1]
    
    filtered = np.zeros_like(pred_bin)
    for i in range(1, n_components + 1):
        if sizes[i-1] >= min_volume or i == largest_idx:
            filtered[labeled == i] = 1
    
    # Morphological closing to smooth boundaries
    struct = ndimage.generate_binary_structure(3, 1)
    filtered = ndimage.binary_closing(filtered, structure=struct, iterations=2).astype(np.uint8)
    
    return filtered


# Test multiple thresholds
thresholds = [100, 300, 500, 1000, 2000]

print("=== nnU-Net Post-processing Results ===\n")

# Baseline (raw predictions)
print("--- Baseline (raw) ---")
baseline_dices = []
for pid in pred_ids:
    pred = nib.load(str(pred_dir / f"{pid}.nii.gz")).get_fdata().astype(np.uint8)
    label = nib.load(str(label_dir / f"{pid}.nii.gz")).get_fdata().astype(np.uint8)
    dice = compute_dice(pred, label)
    baseline_dices.append(dice)
    print(f"  {pid}: Dice={dice:.4f}")
print(f"  Mean: {np.mean(baseline_dices):.4f}\n")

# Post-processed
best_threshold = 0
best_mean_dice = 0

for min_vol in thresholds:
    print(f"--- min_volume={min_vol} ---")
    pp_dices = []
    for pid in pred_ids:
        pred = nib.load(str(pred_dir / f"{pid}.nii.gz")).get_fdata().astype(np.uint8)
        label = nib.load(str(label_dir / f"{pid}.nii.gz")).get_fdata().astype(np.uint8)
        
        pred_pp = postprocess_prediction(pred, min_volume=min_vol)
        dice = compute_dice(pred_pp, label)
        pp_dices.append(dice)
        
        removed = int(np.sum(pred > 0)) - int(np.sum(pred_pp))
        print(f"  {pid}: Dice={dice:.4f} (removed {removed} voxels)")
    
    mean_dice = np.mean(pp_dices)
    print(f"  Mean: {mean_dice:.4f} (delta={mean_dice - np.mean(baseline_dices):+.4f})\n")
    
    if mean_dice > best_mean_dice:
        best_mean_dice = mean_dice
        best_threshold = min_vol

print(f"=== BEST: min_volume={best_threshold}, Mean Dice={best_mean_dice:.4f} ===")
print(f"=== Improvement: {best_mean_dice - np.mean(baseline_dices):+.4f} over baseline ===")
