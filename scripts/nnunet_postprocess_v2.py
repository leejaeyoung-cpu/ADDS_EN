"""
nnU-Net Improved Post-processing
================================
Current baseline: Dice=0.367 → simple post-proc: 0.397
This script tries:
1. Probability threshold optimization (using .npz softmax)
2. Multi-scale morphological operations
3. Boundary refinement with conditional dilation
4. Adaptive volume threshold per case
"""

import nibabel as nib
import numpy as np
from scipy import ndimage
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

PRED_DIR = Path("F:/ADDS/nnUNet_predictions/fold_0_validation")
LABEL_DIR = Path("F:/ADDS/nnUNet_raw/Dataset011_ColonMasked/labelsTr")

PRED_IDS = ['colon_004','colon_013','colon_016','colon_019',
            'colon_024','colon_025','colon_033','colon_039']


def compute_dice(pred, label):
    pred_bin = (pred > 0).astype(float)
    label_bin = (label > 0).astype(float)
    intersection = np.sum(pred_bin * label_bin)
    union = np.sum(pred_bin) + np.sum(label_bin)
    if union == 0:
        return 1.0 if np.sum(pred_bin) == 0 else 0.0
    return 2.0 * intersection / union


def compute_metrics(pred, label):
    """Compute Dice, precision, recall."""
    pred_bin = (pred > 0).astype(float)
    label_bin = (label > 0).astype(float)
    tp = np.sum(pred_bin * label_bin)
    fp = np.sum(pred_bin * (1 - label_bin))
    fn = np.sum((1 - pred_bin) * label_bin)
    
    dice = 2 * tp / (2*tp + fp + fn + 1e-8)
    precision = tp / (tp + fp + 1e-8) 
    recall = tp / (tp + fn + 1e-8)
    return dice, precision, recall


# ================================================================
# Strategy 1: Probability Threshold Optimization
# ================================================================
def prob_threshold_optimization():
    """Use softmax probabilities from .npz to optimize threshold."""
    logger.info("=== Strategy 1: Probability Threshold ===")
    
    # Check if we have npz files with probabilities
    test_npz = PRED_DIR / f"{PRED_IDS[0]}.npz"
    if not test_npz.exists():
        logger.info("No .npz probability files found, skipping")
        return None
    
    # Load one to understand format
    data = np.load(test_npz)
    logger.info("NPZ keys: %s", list(data.keys()))
    probabilities = data['probabilities'] if 'probabilities' in data else data[list(data.keys())[0]]
    logger.info("Probability shape: %s", probabilities.shape)
    
    best_thresh = 0.5
    best_mean_dice = 0
    
    for thresh in [0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7]:
        dices = []
        for pid in PRED_IDS:
            npz_path = PRED_DIR / f"{pid}.npz"
            label_path = LABEL_DIR / f"{pid}.nii.gz"
            
            if not npz_path.exists() or not label_path.exists():
                continue
            
            data = np.load(npz_path)
            probs = data['probabilities'] if 'probabilities' in data else data[list(data.keys())[0]]
            
            # probs shape: (C, D, H, W) where C=num_classes
            if probs.ndim == 4 and probs.shape[0] >= 2:
                # Use foreground probability (class 1)
                fg_prob = probs[1]
            elif probs.ndim == 3:
                fg_prob = probs
            else:
                continue
            
            # Transpose from (D,H,W) to (H,W,D) to match nii.gz label
            label = nib.load(str(label_path)).get_fdata().astype(np.uint8)
            if fg_prob.shape != label.shape:
                fg_prob = np.transpose(fg_prob, (1, 2, 0))
            
            pred_bin = (fg_prob > thresh).astype(np.uint8)
            
            dice = compute_dice(pred_bin, label)
            dices.append(dice)
        
        if dices:
            mean_dice = np.mean(dices)
            logger.info("  thresh=%.2f: mean Dice=%.4f", thresh, mean_dice)
            if mean_dice > best_mean_dice:
                best_mean_dice = mean_dice
                best_thresh = thresh
    
    logger.info("  Best: thresh=%.2f, Dice=%.4f", best_thresh, best_mean_dice)
    return best_thresh


# ================================================================
# Strategy 2: Advanced Morphological Post-processing
# ================================================================
def advanced_postprocess(pred, min_volume=2000, fill_holes=True, smooth=True):
    """
    Multi-step post-processing:
    1. Connected component analysis + volume filter
    2. Hole filling
    3. Gaussian smoothing + re-threshold
    4. Morphological closing
    """
    pred_bin = (pred > 0).astype(np.uint8)
    
    # Step 1: Connected components + volume filter
    labeled, n_comp = ndimage.label(pred_bin)
    if n_comp == 0:
        return pred_bin
    
    sizes = ndimage.sum(pred_bin, labeled, range(1, n_comp + 1))
    largest_idx = np.argmax(sizes) + 1
    
    filtered = np.zeros_like(pred_bin)
    for i in range(1, n_comp + 1):
        if sizes[i-1] >= min_volume or i == largest_idx:
            filtered[labeled == i] = 1
    
    # Step 2: Fill holes in each slice
    if fill_holes:
        for z in range(filtered.shape[0]):
            filtered[z] = ndimage.binary_fill_holes(filtered[z]).astype(np.uint8)
    
    # Step 3: Gaussian smooth + re-threshold
    if smooth:
        smoothed = ndimage.gaussian_filter(filtered.astype(np.float32), sigma=1.0)
        filtered = (smoothed > 0.3).astype(np.uint8)
    
    # Step 4: Morphological closing
    struct = ndimage.generate_binary_structure(3, 2)
    filtered = ndimage.binary_closing(filtered, structure=struct, iterations=2).astype(np.uint8)
    
    return filtered


# ================================================================
# Strategy 3: Probability + Morphology Combined
# ================================================================
def combined_postprocess(probs_fg, min_volume=2000, prob_thresh=0.5):
    """
    Combined: threshold probabilities + morphological cleanup
    """
    pred_bin = (probs_fg > prob_thresh).astype(np.uint8)
    
    # CC analysis
    labeled, n_comp = ndimage.label(pred_bin)
    if n_comp == 0:
        return pred_bin
    
    sizes = ndimage.sum(pred_bin, labeled, range(1, n_comp + 1))
    largest_idx = np.argmax(sizes) + 1
    
    filtered = np.zeros_like(pred_bin)
    for i in range(1, n_comp + 1):
        if sizes[i-1] >= min_volume or i == largest_idx:
            filtered[labeled == i] = 1
    
    # Fill holes per slice
    for z in range(filtered.shape[0]):
        filtered[z] = ndimage.binary_fill_holes(filtered[z]).astype(np.uint8)
    
    # Morphological closing
    struct = ndimage.generate_binary_structure(3, 2)
    filtered = ndimage.binary_closing(filtered, structure=struct, iterations=2).astype(np.uint8)
    
    return filtered


def main():
    logger.info("=" * 60)
    logger.info("nnU-Net Improved Post-processing")
    logger.info("=" * 60)
    
    # ----------------------------------------------------------------
    # Baseline
    # ----------------------------------------------------------------
    logger.info("\n--- Baseline (raw predictions) ---")
    baseline = {}
    for pid in PRED_IDS:
        pred_path = PRED_DIR / f"{pid}.nii.gz"
        label_path = LABEL_DIR / f"{pid}.nii.gz"
        if not pred_path.exists() or not label_path.exists():
            logger.warning("Missing: %s", pid)
            continue
        pred = nib.load(str(pred_path)).get_fdata().astype(np.uint8)
        label = nib.load(str(label_path)).get_fdata().astype(np.uint8)
        dice, prec, rec = compute_metrics(pred, label)
        baseline[pid] = {'dice': dice, 'precision': prec, 'recall': rec}
        logger.info("  %s: Dice=%.4f, Prec=%.4f, Rec=%.4f", pid, dice, prec, rec)
    
    mean_baseline = np.mean([v['dice'] for v in baseline.values()])
    logger.info("  Mean baseline: %.4f\n", mean_baseline)
    
    # ----------------------------------------------------------------
    # Strategy 1: Probability threshold
    # ----------------------------------------------------------------
    best_thresh = prob_threshold_optimization()
    
    # ----------------------------------------------------------------
    # Strategy 2: Advanced morphological (on raw predictions)
    # ----------------------------------------------------------------
    logger.info("\n=== Strategy 2: Advanced Morphological ===")
    for min_vol in [1000, 2000, 3000, 5000]:
        dices = []
        for pid in PRED_IDS:
            pred = nib.load(str(PRED_DIR / f"{pid}.nii.gz")).get_fdata().astype(np.uint8)
            label = nib.load(str(LABEL_DIR / f"{pid}.nii.gz")).get_fdata().astype(np.uint8)
            pp = advanced_postprocess(pred, min_volume=min_vol)
            dice = compute_dice(pp, label)
            dices.append(dice)
        mean_d = np.mean(dices)
        logger.info("  min_vol=%d: mean Dice=%.4f (delta=%+.4f)", min_vol, mean_d, mean_d - mean_baseline)
    
    # ----------------------------------------------------------------
    # Strategy 3: Combined probs + morphology 
    # ----------------------------------------------------------------
    logger.info("\n=== Strategy 3: Combined P+M ===")
    has_npz = (PRED_DIR / f"{PRED_IDS[0]}.npz").exists()
    
    if has_npz:
        best_combined = {'dice': 0, 'params': {}}
        for thresh in [0.35, 0.4, 0.45, 0.5, 0.55, 0.6]:
            for min_vol in [1000, 2000, 3000]:
                dices = []
                per_case = {}
                for pid in PRED_IDS:
                    data = np.load(PRED_DIR / f"{pid}.npz")
                    probs = data['probabilities'] if 'probabilities' in data else data[list(data.keys())[0]]
                    label = nib.load(str(LABEL_DIR / f"{pid}.nii.gz")).get_fdata().astype(np.uint8)
                    
                    if probs.ndim == 4 and probs.shape[0] >= 2:
                        fg = probs[1]
                    elif probs.ndim == 3:
                        fg = probs
                    else:
                        continue
                    
                    # Transpose from (D,H,W) to (H,W,D) to match nii.gz label
                    if fg.shape != label.shape:
                        fg = np.transpose(fg, (1, 2, 0))
                    
                    pp = combined_postprocess(fg, min_volume=min_vol, prob_thresh=thresh)
                    dice = compute_dice(pp, label)
                    dices.append(dice)
                    per_case[pid] = dice
                
                if dices:
                    mean_d = np.mean(dices)
                    if mean_d > best_combined['dice']:
                        best_combined = {
                            'dice': mean_d,
                            'params': {'thresh': thresh, 'min_vol': min_vol},
                            'per_case': per_case,
                        }
        
        logger.info("  Best combined: Dice=%.4f, params=%s", 
                    best_combined['dice'], best_combined['params'])
        logger.info("  Delta vs baseline: %+.4f", best_combined['dice'] - mean_baseline)
        
        if best_combined.get('per_case'):
            for pid, dice in best_combined['per_case'].items():
                base = baseline.get(pid, {}).get('dice', 0)
                logger.info("    %s: %.4f → %.4f (%+.4f)", pid, base, dice, dice - base)
    else:
        logger.info("  No .npz files, skipping probability-based strategies")
    
    # ----------------------------------------------------------------
    # Final summary
    # ----------------------------------------------------------------
    logger.info("\n" + "=" * 60)
    logger.info("SUMMARY")
    logger.info("=" * 60)
    logger.info("  Baseline:     Dice=%.4f", mean_baseline)
    logger.info("  Previous PP:  Dice=0.397 (min_vol=2000)")
    if has_npz and best_combined['dice'] > 0:
        logger.info("  Best new:     Dice=%.4f (%+.4f vs baseline)", 
                    best_combined['dice'], best_combined['dice'] - mean_baseline)
    logger.info("  Advanced morph with best min_vol tested above")


if __name__ == "__main__":
    main()
