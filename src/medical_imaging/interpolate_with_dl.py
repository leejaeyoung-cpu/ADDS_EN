"""
Inference script for trained slice interpolator

Uses trained model to enhance CT volumes by interpolating intermediate slices.
"""

import torch
import torch.nn.functional as F
import numpy as np
import nibabel as nib
from pathlib import Path
from typing import Tuple
import logging

from dl_slice_interpolator import create_interpolator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_trained_model(model_path: Path, device: str = 'cuda') -> torch.nn.Module:
    """Load trained interpolator model"""
    checkpoint = torch.load(model_path, map_location=device)
    model_type = checkpoint.get('model_type', 'lightweight')
    
    model = create_interpolator(model_type, device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    logger.info(f"Loaded {model_type} model from {model_path}")
    logger.info(f"  Epoch: {checkpoint['epoch']}")
    logger.info(f"  Val Loss: {checkpoint['val_loss']:.4f}")
    
    return model


def interpolate_volume_dl(
    volume: np.ndarray,
    model: torch.nn.Module,
    device: str = 'cuda',
    interpolation_factor: int = 2
) -> np.ndarray:
    """
    Interpolate CT volume using deep learning model
    
    Args:
        volume: Input volume (Z, H, W)
        model: Trained interpolator model
        device: Device for inference
        interpolation_factor: How many slices to insert between each pair
    
    Returns:
        Interpolated volume with more slices
    """
    logger.info(f"Interpolating volume with DL model (factor: {interpolation_factor}x)")
    logger.info(f"  Original shape: {volume.shape}")
    
    z, h, w = volume.shape
    
    # Normalize volume
    hu_min, hu_max = -100, 400
    volume_norm = np.clip((volume - hu_min) / (hu_max - hu_min), 0, 1)
    
    # Initialize output volume
    new_z = (z - 1) * interpolation_factor + z
    interpolated = np.zeros((new_z, h, w), dtype=np.float32)
    
    # Copy original slices
    for i in range(z):
        interpolated[i * (interpolation_factor + 1)] = volume_norm[i]
    
    # Interpolate between slices
    model.eval()
    with torch.no_grad():
        for i in range(z - 1):
            # Get adjacent slices
            slice_before = volume_norm[i]
            slice_after = volume_norm[i + 1]
            
            # Prepare input
            input_pair = torch.from_numpy(
                np.stack([slice_before, slice_after], axis=0)
            ).float().unsqueeze(0).to(device)  # (1, 2, H, W)
            
            # Generate intermediate slices
            for j in range(1, interpolation_factor + 1):
                # Weight for interpolation (0.0 to 1.0)
                weight = j / (interpolation_factor + 1)
                
                # Generate slice
                pred = model(input_pair).squeeze(0).squeeze(0).cpu().numpy()
                
                # Weighted blend with linear interpolation (for stability)
                linear_interp = (1 - weight) * slice_before + weight * slice_after
                pred_blended = 0.7 * pred + 0.3 * linear_interp
                
                # Store
                output_idx = i * (interpolation_factor + 1) + j
                interpolated[output_idx] = pred_blended
    
    # Denormalize
    interpolated = interpolated * (hu_max - hu_min) + hu_min
    
    logger.info(f"  Interpolated shape: {interpolated.shape}")
    
    return interpolated


def enhance_ct_volume_dl(
    input_path: Path,
    output_path: Path,
    model_path: Path,
    interpolation_factor: int = 2,
    device: str = 'cuda'
):
    """
    Enhance CT volume using DL interpolation
    
    Args:
        input_path: Input .nii.gz file
        output_path: Output .nii.gz file
        model_path: Path to trained model
        interpolation_factor: Z-axis interpolation factor
        device: Device for inference
    """
    logger.info(f"Enhancing CT volume: {input_path.name}")
    
    # Load model
    model = load_trained_model(model_path, device)
    
    # Load volume
    nii = nib.load(input_path)
    volume = nii.get_fdata()
    spacing = nii.header.get_zooms()
    
    # Interpolate
    interpolated = interpolate_volume_dl(
        volume,
        model,
        device,
        interpolation_factor
    )
    
    # Update spacing
    new_spacing = (
        spacing[0] / (interpolation_factor + 1),
        spacing[1],
        spacing[2]
    )
    
    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    new_affine = nii.affine.copy()
    new_affine[2, 2] = new_affine[2, 2] / (interpolation_factor + 1)
    
    new_nii = nib.Nifti1Image(interpolated.astype(volume.dtype), new_affine, nii.header)
    new_nii.header.set_zooms(new_spacing)
    
    nib.save(new_nii, output_path)
    
    logger.info(f"Saved: {output_path}")
    logger.info(f"  Spacing: {spacing} -> {new_spacing}")


if __name__ == "__main__":
    # Example usage
    model_path = Path("models/slice_interpolator/best_interpolator_lightweight.pth")
    
    if not model_path.exists():
        print(f"[ERROR] Model not found: {model_path}")
        print("Please train the model first using train_interpolator.py")
    else:
        # Test on a CT volume
        test_volume = Path("outputs/inha_ct_analysis/inha_ct_volume.nii.gz")
        
        if test_volume.exists():
            enhance_ct_volume_dl(
                test_volume,
                Path("outputs/inha_ct_analysis/inha_ct_volume_dl_enhanced.nii.gz"),
                model_path,
                interpolation_factor=2,
                device='cuda' if torch.cuda.is_available() else 'cpu'
            )
        else:
            print(f"[INFO] Test volume not found: {test_volume}")
            print("Model loaded successfully and ready for inference!")
