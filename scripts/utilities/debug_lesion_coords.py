"""
Debug lesion coordinates and volume dimensions
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import nibabel as nib
from src.medical_imaging.detection.anatomy_based_detector import quick_detect_colon_tumors

case_name = "colon_001"
image_path = f"data/medical_decathlon/Task10_Colon/imagesTr/{case_name}.nii.gz"

print(f"\n[*] Loading: {image_path}")
nii = nib.load(image_path)
volume = nii.get_fdata()
spacing = nii.header.get_zooms()

print(f"\nVolume shape: {volume.shape}")
print(f"Spacing: {spacing}")

print(f"\n[*] Running detection...")
lesions = quick_detect_colon_tumors(ct_volume=volume, spacing=spacing, device="gpu")

print(f"\n[*] Found {len(lesions)} lesions")
print(f"\nLesion details:")
for i, lesion in enumerate(lesions[:10]):
    print(f"\nLesion {i+1}:")
    print(f"  Centroid: {lesion.centroid}")
    print(f"  BBox: z={lesion.bbox['z_min']}-{lesion.bbox['z_max']}, "
          f"y={lesion.bbox['y_min']}-{lesion.bbox['y_max']}, "
          f"x={lesion.bbox['x_min']}-{lesion.bbox['x_max']}")
    print(f"  Mask shape: {lesion.mask.shape}")
    print(f"  Classification: {lesion.classification}")
    print(f"  HU: {lesion.mean_hu:.1f}")

print(f"\n[*] Volume shape vs lesion coords:")
print(f"  Volume: {volume.shape}")
print(f"  Max lesion z: {max(l.bbox['z_max'] for l in lesions)}")
print(f"  Max lesion y: {max(l.bbox['y_max'] for l in lesions)}")
print(f"  Max lesion x: {max(l.bbox['x_max'] for l in lesions)}")
