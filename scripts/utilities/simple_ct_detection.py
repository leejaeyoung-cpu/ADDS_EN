"""
간단한 CT 종양 검출 스크립트 (Post-processing 비활성화)
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from pathlib import Path
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from medical_imaging.inference.predictor import SOTAPredictor


def load_nifti(path):
    """Load NIfTI file"""
    nii = nib.load(str(path))
    volume = nii.get_fdata()
    spacing = nii.header.get_zooms()
    return volume, spacing


def detect_tumor(ct_image_path, model_path, disable_postprocessing=True):
    """
    CT 이미지에서 종양 검출
    
    Args:
        ct_image_path: CT 이미지 경로 (.nii.gz)
        model_path: 모델 체크포인트 경로
        disable_postprocessing: Post-processing 비활성화 여부 (기본: True)
    
    Returns:
        prediction: 종양 예측 마스크
        volume: 원본 CT 볼륨
    """
    # Load CT image
    print(f"\n📂 Loading: {ct_image_path}")
    volume, spacing = load_nifti(ct_image_path)
    print(f"   Image shape: {volume.shape}")
    print(f"   Spacing: {spacing}")
    
    # Initialize predictor
    print(f"\n🔧 Initializing predictor...")
    print(f"   Post-processing: {'DISABLED' if disable_postprocessing else 'ENABLED'}")
    
    predictor = SOTAPredictor(
        checkpoint_path=model_path,
        device='cuda',
        apply_postprocessing=not disable_postprocessing
    )
    
    # Convert to 3-channel (모델 요구사항)
    print(f"\n🔄 Converting to 3-channel format...")
    volume_3ch = np.stack([volume, volume, volume], axis=0)
    
    # Run detection
    print(f"\n🎯 Running tumor detection...")
    prediction = predictor.predict(volume_3ch, return_probabilities=False)
    
    # Results
    tumor_voxels = np.sum(prediction > 0.5)
    total_voxels = prediction.size
    tumor_percentage = (tumor_voxels / total_voxels) * 100
    
    print(f"\n✅ Detection Complete:")
    print(f"   Tumor voxels: {tumor_voxels:,}")
    print(f"   Tumor percentage: {tumor_percentage:.4f}%")
    print(f"   Has tumor: {'YES' if tumor_voxels > 0 else 'NO'}")
    
    return prediction, volume


def visualize_detection(volume, prediction, output_path="detection_result.png"):
    """Visualize detection result"""
    # Find middle slice
    mid_slice = volume.shape[2] // 2
    
    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Original CT
    axes[0].imshow(volume[:, :, mid_slice].T, cmap='gray', origin='lower')
    axes[0].set_title('Original CT (Middle Slice)', fontsize=14)
    axes[0].axis('off')
    
    # Prediction
    axes[1].imshow(prediction[:, :, mid_slice].T, cmap='hot', origin='lower')
    axes[1].set_title('Tumor Prediction', fontsize=14)
    axes[1].axis('off')
    
    # Overlay
    axes[2].imshow(volume[:, :, mid_slice].T, cmap='gray', origin='lower', alpha=0.7)
    axes[2].imshow(prediction[:, :, mid_slice].T, cmap='hot', origin='lower', alpha=0.5)
    axes[2].set_title('Overlay', fontsize=14)
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\n💾 Saved visualization: {output_path}")


def main():
    print("="*80)
    print("Simple CT Tumor Detection (Post-processing Disabled)")
    print("="*80)
    
    # Paths
    ct_path = "data/medical_decathlon/Task10_Colon/imagesTr/colon_001.nii.gz"
    model_path = "models/sota_combo_v2/fold_0/best_model.pth"
    output_path = "simple_detection_result.png"
    
    # Check files exist
    if not os.path.exists(ct_path):
        print(f"\n❌ CT image not found: {ct_path}")
        print("   Please update ct_path in the script")
        return
    
    if not os.path.exists(model_path):
        print(f"\n❌ Model not found: {model_path}")
        return
    
    # Run detection
    prediction, volume = detect_tumor(ct_path, model_path, disable_postprocessing=True)
    
    # Visualize
    visualize_detection(volume, prediction, output_path)
    
    print("\n" + "="*80)
    print("✅ Done!")
    print("="*80)


if __name__ == "__main__":
    main()
