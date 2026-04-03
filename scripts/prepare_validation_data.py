"""
Prepare validation cases for inference
nnUNetv2_predict requires raw NIfTI images, not preprocessed data
"""

import json
import shutil
from pathlib import Path

print("="*70)
print("Preparing Fold 0 Validation Cases for Inference")
print("="*70)

# Load splits
with open("f:/ADDS/nnUNet_preprocessed/Dataset011_ColonMasked/splits_final.json") as f:
    splits = json.load(f)

fold_0_val = splits[0]['val']  # 26 cases
print(f"\nFold 0 validation: {len(fold_0_val)} cases")

# Create test directories
test_imgs_dir = Path("f:/ADDS/nnUNet_raw/Dataset011_ColonMasked/imagesTs_fold0")
test_labels_dir = Path("f:/ADDS/nnUNet_raw/Dataset011_ColonMasked/labelsTs_fold0")

test_imgs_dir.mkdir(parents=True, exist_ok=True)
test_labels_dir.mkdir(parents=True, exist_ok=True)

# Source directories
train_imgs_dir = Path("f:/ADDS/nnUNet_raw/Dataset011_ColonMasked/imagesTr")
train_labels_dir = Path("f:/ADDS/nnUNet_raw/Dataset011_ColonMasked/labelsTr")

print(f"\nSource images: {train_imgs_dir}")
print(f"Target images: {test_imgs_dir}")

# Create symlinks or copy (symlinks are faster)
copied = 0
for case_id in fold_0_val:
    img_file = f"{case_id}_0000.nii.gz"
    label_file = f"{case_id}.nii.gz"
    
    # Image
    src_img = train_imgs_dir / img_file
    dst_img = test_imgs_dir / img_file
    if src_img.exists() and not dst_img.exists():
        # Use copy instead of symlink for compatibility
        shutil.copy2(src_img, dst_img)
        copied += 1
    
    # Label (for later dice calculation)
    src_label = train_labels_dir / label_file
    dst_label = test_labels_dir / label_file
    if src_label.exists() and not dst_label.exists():
        shutil.copy2(src_label, dst_label)

print(f"\n✅ Prepared {copied} validation images")
print(f"✅ Images directory: {test_imgs_dir}")
print(f"✅ Labels directory: {test_labels_dir}")

print("\n" + "="*70)
print("Ready for inference!")
print("="*70)
print(f"\nNext command:")
print(f"nnUNetv2_predict -i {test_imgs_dir} -o f:/ADDS/nnUNet_predictions/fold_0_validation -d 011 -c 3d_fullres -f 0")
