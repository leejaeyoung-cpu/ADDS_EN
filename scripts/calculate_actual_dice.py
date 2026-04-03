"""
Phase 1: Calculate Actual Dice Score
실행: python scripts/calculate_actual_dice.py
"""

import numpy as np
import nibabel as nib
from pathlib import Path
import pandas as pd
from scipy.spatial.distance import directed_hausdorff
import json

def calculate_dice(pred, gt):
    """실제 Dice coefficient 계산"""
    pred_bool = pred > 0.5
    gt_bool = gt > 0
    
    intersection = np.sum(pred_bool * gt_bool)
    union = np.sum(pred_bool) + np.sum(gt_bool)
    
    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    
    return (2.0 * intersection) / (union + 1e-8)

def calculate_metrics(pred_path, gt_path):
    """Per-case 상세 metrics 계산"""
    print(f"Processing: {pred_path.name}...", end=' ')
    
    try:
        pred = nib.load(pred_path).get_fdata()
        gt = nib.load(gt_path).get_fdata()
        
        pred_bool = pred > 0.5
        gt_bool = gt > 0
        
        # Dice
        dice = calculate_dice(pred, gt)
        
        # Confusion matrix
        tp = np.sum(pred_bool * gt_bool)
        fp = np.sum(pred_bool * (~gt_bool))
        fn = np.sum((~pred_bool) * gt_bool)
        tn = np.sum((~pred_bool) * (~gt_bool))
        
        # Derived metrics
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        specificity = tn / (tn + fp + 1e-8)
        
        # Hausdorff Distance (경계 정확도)
        hd = np.inf
        if tp > 0:
            try:
                pred_points = np.argwhere(pred_bool)
                gt_points = np.argwhere(gt_bool)
                if len(pred_points) > 0 and len(gt_points) > 0:
                    hd = max(
                        directed_hausdorff(pred_points, gt_points)[0],
                        directed_hausdorff(gt_points, pred_points)[0]
                    )
            except:
                hd = -1  # Error computing HD
        
        print(f"Dice={dice:.4f}")
        
        return {
            'dice': float(dice),
            'precision': float(precision),
            'recall': float(recall),
            'f1': float(f1),
            'specificity': float(specificity),
            'tp': int(tp),
            'fp': int(fp),
            'fn': int(fn),
            'tn': int(tn),
            'hausdorff_distance': float(hd)
        }
    except Exception as e:
        print(f"ERROR: {e}")
        return None

def main(pred_dir=None, gt_dir=None):
    """Main evaluation function"""
    
    if pred_dir is None:
        pred_dir = Path("f:/ADDS/nnUNet_predictions/fold_0_validation")
    if gt_dir is None:
        gt_dir = Path("f:/ADDS/nnUNet_raw/Dataset011_ColonMasked/labelsTs")
    
    pred_dir = Path(pred_dir)
    gt_dir = Path(gt_dir)
    
    print("\n" + "="*70)
    print("PHASE 1: Calculating Actual Dice Scores")
    print("="*70)
    print(f"Predictions: {pred_dir}")
    print(f"Ground Truth: {gt_dir}")
    print("-"*70)
    
    # Find all prediction files
    pred_files = sorted(pred_dir.glob("*.nii.gz"))
    print(f"Found {len(pred_files)} prediction files\n")
    
    results = []
    for pred_file in pred_files:
        case_id = pred_file.stem.replace('.nii', '')
        gt_file = gt_dir / f"{case_id}.nii.gz"
        
        if not gt_file.exists():
            print(f"WARNING: Ground truth not found for {case_id}")
            continue
        
        metrics = calculate_metrics(pred_file, gt_file)
        if metrics:
            metrics['case_id'] = case_id
            results.append(metrics)
    
    if not results:
        print("ERROR: No results computed!")
        return None
    
    # Convert to DataFrame
    df = pd.DataFrame(results)
    
    # Summary statistics
    print("\n" + "="*70)
    print("VALIDATION SUMMARY STATISTICS")
    print("="*70)
    
    print(f"\n[DICE SCORE]")
    print(f"  Mean:   {df['dice'].mean():.4f} +/- {df['dice'].std():.4f}")
    print(f"  Median: {df['dice'].median():.4f}")
    print(f"  Min:    {df['dice'].min():.4f} (Case: {df.loc[df['dice'].idxmin(), 'case_id']})")
    print(f"  Max:    {df['dice'].max():.4f} (Case: {df.loc[df['dice'].idxmax(), 'case_id']})")
    
    print(f"\n[PRECISION & RECALL]")
    print(f"  Precision: {df['precision'].mean():.4f} +/- {df['precision'].std():.4f}")
    print(f"  Recall:    {df['recall'].mean():.4f} +/- {df['recall'].std():.4f}")
    print(f"  F1 Score:  {df['f1'].mean():.4f} +/- {df['f1'].std():.4f}")
    
    # False Positive/Negative Rates
    total_tp = df['tp'].sum()
    total_fp = df['fp'].sum()
    total_fn = df['fn'].sum()
    total_tn = df['tn'].sum()
    
    fp_rate = total_fp / (total_fp + total_tn) if (total_fp + total_tn) > 0 else 0
    fn_rate = total_fn / (total_fn + total_tp) if (total_fn + total_tp) > 0 else 0
    
    print(f"\n[ERROR RATES]")
    print(f"  False Positive Rate: {fp_rate*100:.2f}% ({total_fp:,} voxels)")
    print(f"  False Negative Rate: {fn_rate*100:.2f}% ({total_fn:,} voxels)")
    
    print(f"\n[HAUSDORFF DISTANCE]")
    valid_hd = df[df['hausdorff_distance'] > 0]['hausdorff_distance']
    if len(valid_hd) > 0:
        print(f"  Mean: {valid_hd.mean():.2f} voxels")
        print(f"  Median: {valid_hd.median():.2f} voxels")
    else:
        print("  N/A (no valid cases)")
    
    # Clinical categorization
    print(f"\n[CLINICAL CATEGORIZATION]")
    excellent = len(df[df['dice'] >= 0.75])
    good = len(df[(df['dice'] >= 0.60) & (df['dice'] < 0.75)])
    fair = len(df[(df['dice'] >= 0.45) & (df['dice'] < 0.60)])
    poor = len(df[df['dice'] < 0.45])
    
    total = len(df)
    print(f"  Excellent (≥0.75): {excellent:2d}/{total} ({excellent/total*100:.1f}%)")
    print(f"  Good (0.60-0.75):  {good:2d}/{total} ({good/total*100:.1f}%)")
    print(f"  Fair (0.45-0.60):  {fair:2d}/{total} ({fair/total*100:.1f}%)")
    print(f"  Poor (<0.45):      {poor:2d}/{total} ({poor/total*100:.1f}%)")
    
    # VERDICT
    print("\n" + "="*70)
    mean_dice = df['dice'].mean()
    
    if mean_dice >= 0.62:
        verdict = "✅ PASS"
        interpretation = "Training metrics reflect actual performance"
    elif mean_dice >= 0.55:
        verdict = "⚠️ WARNING"
        interpretation = "Lower than Pseudo Dice - overfitting suspected"
    else:
        verdict = "❌ FAIL"
        interpretation = '"실제로는 모른다" CONFIRMED!'
    
    print(f"VERDICT: {verdict}")
    print(f"Mean Dice: {mean_dice:.4f}")
    print(f"Interpretation: {interpretation}")
    print("="*70)
    
    # Save results
    output_csv = "validation_metrics_detailed.csv"
    df.to_csv(output_csv, index=False)
    print(f"\n💾 Detailed results saved to: {output_csv}")
    
    # Save summary JSON
    summary = {
        'mean_dice': float(mean_dice),
        'std_dice': float(df['dice'].std()),
        'median_dice': float(df['dice'].median()),
        'min_dice': float(df['dice'].min()),
        'max_dice': float(df['dice'].max()),
        'mean_precision': float(df['precision'].mean()),
        'mean_recall': float(df['recall'].mean()),
        'fp_rate': float(fp_rate),
        'fn_rate': float(fn_rate),
        'verdict': verdict,
        'num_cases': len(df)
    }
    
    with open("validation_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"💾 Summary saved to: validation_summary.json")
    
    return summary

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 2:
        pred_dir = sys.argv[1]
        gt_dir = sys.argv[2]
        main(pred_dir, gt_dir)
    else:
        print("Usage: python calculate_actual_dice.py [pred_dir] [gt_dir]")
        print("Or run with defaults:")
        main()
