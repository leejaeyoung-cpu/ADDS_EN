"""
nnU-Net 모델 검증 스크립트
Validation 데이터로 inference 수행 및 평가
"""
import os
import json
import numpy as np
import nibabel as nib
from pathlib import Path

# Fold 0 validation cases
VAL_CASES = [
    "colon_004", "colon_013", "colon_016", "colon_019", "colon_024",
    "colon_025", "colon_033", "colon_039", "colon_040", "colon_045",
    "colon_047", "colon_048", "colon_049", "colon_054", "colon_060",
    "colon_063", "colon_065", "colon_069", "colon_071", "colon_088",
    "colon_092", "colon_093", "colon_096", "colon_097", "colon_099",
    "colon_106"
]

INPUT_DIR = r"F:\ADDS\nnUNet_raw\Dataset011_ColonMasked\imagesTr"
LABEL_DIR = r"F:\ADDS\nnUNet_raw\Dataset011_ColonMasked\labelsTr"
OUTPUT_DIR = r"F:\ADDS\evaluation_results\validation_inference"

def prepare_validation_input():
    """Validation cases를 inference 입력 폴더로 복사"""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    input_for_inference = os.path.join(OUTPUT_DIR, "input")
    os.makedirs(input_for_inference, exist_ok=True)
    
    print(f"Preparing {len(VAL_CASES)} validation cases for inference...")
    
    for case_id in VAL_CASES:
        src = os.path.join(INPUT_DIR, f"{case_id}_0000.nii.gz")
        if os.path.exists(src):
            # Symbolic link or copy
            dst = os.path.join(input_for_inference, f"{case_id}_0000.nii.gz")
            if not os.path.exists(dst):
                # Windows에서는 copy 사용
                import shutil
                shutil.copy2(src, dst)
                print(f"  Copied: {case_id}")
        else:
            print(f"  WARNING: {case_id} not found!")
    
    print(f"\nInput prepared at: {input_for_inference}")
    return input_for_inference

def calculate_dice(pred, gt):
    """Dice coefficient 계산"""
    pred_binary = (pred > 0).astype(np.float32)
    gt_binary = (gt > 0).astype(np.float32)
    
    intersection = np.sum(pred_binary * gt_binary)
    union = np.sum(pred_binary) + np.sum(gt_binary)
    
    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    
    return 2.0 * intersection / union

def evaluate_predictions(pred_dir, label_dir):
    """예측 결과 평가"""
    results = {}
    dice_scores = []
    
    print("\nEvaluating predictions...")
    
    for case_id in VAL_CASES:
        pred_file = os.path.join(pred_dir, f"{case_id}.nii.gz")
        label_file = os.path.join(label_dir, f"{case_id}.nii.gz")
        
        if not os.path.exists(pred_file):
            print(f"  WARNING: Prediction for {case_id} not found!")
            continue
        
        if not os.path.exists(label_file):
            print(f"  WARNING: Label for {case_id} not found!")
            continue
        
        # Load prediction and ground truth
        pred = nib.load(pred_file).get_fdata()
        gt = nib.load(label_file).get_fdata()
        
        # Calculate Dice
        dice = calculate_dice(pred, gt)
        dice_scores.append(dice)
        
        # Calculate volume metrics
        pred_volume = np.sum(pred > 0)
        gt_volume = np.sum(gt > 0)
        
        results[case_id] = {
            "dice": float(dice),
            "pred_volume_voxels": int(pred_volume),
            "gt_volume_voxels": int(gt_volume),
            "volume_ratio": float(pred_volume / gt_volume) if gt_volume > 0 else 0.0
        }
        
        print(f"  {case_id}: Dice = {dice:.4f}, Vol ratio = {results[case_id]['volume_ratio']:.2f}")
    
    # Summary statistics
    summary = {
        "mean_dice": float(np.mean(dice_scores)),
        "std_dice": float(np.std(dice_scores)),
        "min_dice": float(np.min(dice_scores)),
        "max_dice": float(np.max(dice_scores)),
        "median_dice": float(np.median(dice_scores)),
        "num_cases": len(dice_scores)
    }
    
    print(f"\n{'='*60}")
    print(f"VALIDATION SUMMARY (n={summary['num_cases']})")
    print(f"{'='*60}")
    print(f"Mean Dice:   {summary['mean_dice']:.4f} ± {summary['std_dice']:.4f}")
    print(f"Median Dice: {summary['median_dice']:.4f}")
    print(f"Range:       {summary['min_dice']:.4f} - {summary['max_dice']:.4f}")
    print(f"{'='*60}")
    
    return results, summary

if __name__ == "__main__":
    print("="*60)
    print("nnU-Net Model Evaluation Script")
    print("="*60)
    
    # Step 1: Prepare input
    input_dir = prepare_validation_input()
    
    print("\n" + "="*60)
    print("NEXT STEPS:")
    print("="*60)
    print("1. Run nnU-Net inference:")
    print(f"   nnUNetv2_predict -i {input_dir} \\")
    print(f"      -o {os.path.join(OUTPUT_DIR, 'predictions')} \\")
    print(f"      -d 011 -c 3d_fullres -f 0 --save_probabilities")
    print("\n2. After inference completes, run evaluation:")
    print("   python -c \"from evaluate_model import evaluate_predictions; ")
    print(f"   evaluate_predictions(r'{os.path.join(OUTPUT_DIR, 'predictions')}', r'{LABEL_DIR}')\"")
    print("="*60)
