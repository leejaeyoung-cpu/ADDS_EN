"""
Custom split file for single-case nnU-Net training
Creates a manual train/val split for Dataset100_CRC_CT
"""

import json
from pathlib import Path

# Output directory
output_dir = Path("F:/ADDS/nnUNet_data/nnUNet_preprocessed/Dataset100_CRC_CT")
output_dir.mkdir(parents=True, exist_ok=True)

# Create custom split with single case in train and val
# This allows training with just 1 case
splits = [
    {
        "train": ["CRC_CT_001"],
        "val": ["CRC_CT_001"]
    }
]

# Save splits file
splits_file = output_dir / "splits_final.json"
with open(splits_file, 'w') as f:
    json.dump(splits, f, indent=2)

print(f"✅ Created custom split file: {splits_file}")
print(f"   Train cases: {splits[0]['train']}")
print(f"   Val cases: {splits[0]['val']}")
print("\nYou can now train with fold 0 using:")
print("nnUNetv2_train 100 2d 0 --npz")
