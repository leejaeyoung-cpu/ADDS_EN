"""
Validation 데이터 분석 스크립트
각 validation 케이스의 종양 분포와 center crop 영역의 종양 포함 여부 분석
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import nibabel as nib
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend

def analyze_validation_fold(data_root: str = "data/medical_decathlon/Task10_Colon", fold: int = 0):
    """Validation fold의 종양 분포 분석"""
    
    print("=" * 80)
    print(f"Validation Fold {fold} Analysis - Tumor Distribution")
    print("=" * 80)
    
    data_root = Path(data_root)
    images_dir = data_root / "imagesTr"
    labels_dir = data_root / "labelsTr"
    
    # Load all cases
    image_files = sorted(list(images_dir.glob("*.nii.gz")))
    
    all_samples = []
    for img_path in image_files:
        label_path = labels_dir / img_path.name
        if label_path.exists():
            all_samples.append({
                'image': str(img_path),
                'label': str(label_path),
                'case_id': img_path.stem.replace('.nii', '')
            })
    
    # K-Fold split (same as dataset)
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    indices = list(range(len(all_samples)))
    folds = list(kfold.split(indices))
    _, val_idx = folds[fold]
    
    val_samples = [all_samples[i] for i in val_idx]
    
    print(f"\nTotal validation cases: {len(val_samples)}\n")
    
    # Analyze each validation case
    results = []
    patch_size = (96, 96, 96)
    
    for i, sample in enumerate(val_samples):
        print(f"[{i+1}/{len(val_samples)}] Analyzing {sample['case_id']}...")
        
        # Load label
        nifti_label = nib.load(sample['label'], mmap=False)
        label_vol = np.asarray(nifti_label.dataobj, dtype=np.float32)
        
        # Calculate tumor statistics
        total_voxels = label_vol.size
        tumor_voxels = (label_vol > 0).sum()
        tumor_ratio = tumor_voxels / total_voxels * 100
        
        # Check center crop region
        vol_shape = label_vol.shape
        start_z = max(0, (vol_shape[0] - patch_size[0]) // 2)
        start_y = max(0, (vol_shape[1] - patch_size[1]) // 2)
        start_x = max(0, (vol_shape[2] - patch_size[2]) // 2)
        
        center_patch = label_vol[
            start_z:start_z + patch_size[0],
            start_y:start_y + patch_size[1],
            start_x:start_x + patch_size[2]
        ]
        
        center_tumor_voxels = (center_patch > 0).sum()
        center_tumor_ratio = center_tumor_voxels / center_patch.size * 100 if center_patch.size > 0 else 0
        
        # Find tumor center of mass
        if tumor_voxels > 0:
            tumor_coords = np.where(label_vol > 0)
            tumor_center = (
                int(np.mean(tumor_coords[0])),
                int(np.mean(tumor_coords[1])),
                int(np.mean(tumor_coords[2]))
            )
        else:
            tumor_center = None
        
        results.append({
            'case_id': sample['case_id'],
            'volume_shape': vol_shape,
            'total_voxels': total_voxels,
            'tumor_voxels': tumor_voxels,
            'tumor_ratio': tumor_ratio,
            'center_tumor_voxels': center_tumor_voxels,
            'center_tumor_ratio': center_tumor_ratio,
            'tumor_center': tumor_center
        })
        
        print(f"  Volume shape: {vol_shape}")
        print(f"  Total tumor voxels: {tumor_voxels:,} ({tumor_ratio:.4f}%)")
        print(f"  Center patch tumor: {center_tumor_voxels:,} ({center_tumor_ratio:.4f}%)")
        print(f"  Tumor center: {tumor_center}")
        print()
    
    # Summary statistics
    print("=" * 80)
    print("SUMMARY STATISTICS")
    print("=" * 80)
    
    total_tumor_ratios = [r['tumor_ratio'] for r in results]
    center_tumor_ratios = [r['center_tumor_ratio'] for r in results]
    cases_with_tumor = sum(1 for r in results if r['tumor_voxels'] > 0)
    cases_with_center_tumor = sum(1 for r in results if r['center_tumor_voxels'] > 0)
    
    print(f"\nCases with tumor: {cases_with_tumor}/{len(val_samples)} ({cases_with_tumor/len(val_samples)*100:.1f}%)")
    print(f"Cases with tumor in CENTER CROP: {cases_with_center_tumor}/{len(val_samples)} ({cases_with_center_tumor/len(val_samples)*100:.1f}%)")
    print(f"\nAverage tumor ratio (whole volume): {np.mean(total_tumor_ratios):.4f}%")
    print(f"Average tumor ratio (center crop): {np.mean(center_tumor_ratios):.4f}%")
    print(f"\n⚠️  CENTER CROP MISSES TUMOR IN: {len(val_samples) - cases_with_center_tumor}/{len(val_samples)} cases")
    
    # Detailed table
    print("\n" + "-" * 80)
    print(f"{'Case ID':<20} {'Total Tumor %':<15} {'Center Tumor %':<15} {'Center Has Tumor'}")
    print("-" * 80)
    for r in results:
        has_center = "✓" if r['center_tumor_voxels'] > 0 else "✗"
        print(f"{r['case_id']:<20} {r['tumor_ratio']:>12.4f}%  {r['center_tumor_ratio']:>12.4f}%   {has_center:^10}")
    
    # Save results
    output_dir = Path("analysis_output")
    output_dir.mkdir(exist_ok=True)
    
    with open(output_dir / f"validation_fold{fold}_analysis.txt", "w") as f:
        f.write(f"Validation Fold {fold} Analysis\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Cases with tumor in CENTER CROP: {cases_with_center_tumor}/{len(val_samples)}\n")
        f.write(f"CENTER CROP MISSES TUMOR IN: {len(val_samples) - cases_with_center_tumor}/{len(val_samples)} cases\n\n")
        
        for r in results:
            f.write(f"{r['case_id']}: {r['tumor_ratio']:.4f}% total, {r['center_tumor_ratio']:.4f}% center\n")
    
    print(f"\n✓ Results saved to: {output_dir / f'validation_fold{fold}_analysis.txt'}")
    
    # Visualization
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Tumor ratio distribution
    axes[0].bar(range(len(results)), [r['tumor_ratio'] for r in results], alpha=0.7, label='Whole Volume')
    axes[0].bar(range(len(results)), [r['center_tumor_ratio'] for r in results], alpha=0.7, label='Center Crop')
    axes[0].set_xlabel('Validation Case Index')
    axes[0].set_ylabel('Tumor Ratio (%)')
    axes[0].set_title('Tumor Distribution: Whole Volume vs Center Crop')
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    
    # Plot 2: Missing tumor cases
    missing_count = len(val_samples) - cases_with_center_tumor
    axes[1].bar(['Has Tumor', 'No Tumor'], [cases_with_center_tumor, missing_count], color=['green', 'red'], alpha=0.7)
    axes[1].set_ylabel('Number of Cases')
    axes[1].set_title(f'Center Crop Tumor Coverage (Fold {fold})')
    axes[1].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / f'validation_fold{fold}_plot.png', dpi=150)
    print(f"✓ Plot saved to: {output_dir / f'validation_fold{fold}_plot.png'}")
    
    print("\n" + "=" * 80)
    print("CONCLUSION:")
    print("=" * 80)
    if cases_with_center_tumor < len(val_samples) * 0.5:
        print("⚠️  CRITICAL: More than 50% of validation cases have NO tumor in center crop!")
        print("⚠️  This explains the extremely low validation Dice score (0.0069)")
        print("⚠️  ACTION REQUIRED: Fix validation sampling strategy")
    else:
        print("✓ Most validation cases have tumor in center crop")
        print("  (But this may not be the primary cause of low Dice)")
    
    print("=" * 80)
    
    return results


if __name__ == "__main__":
    analyze_validation_fold(fold=0)
