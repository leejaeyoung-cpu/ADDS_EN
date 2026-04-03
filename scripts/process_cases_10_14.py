"""
Process specific cases 10-14
"""
import subprocess
import sys
from pathlib import Path

def process_cases_10_to_14(dataset_dir):
    """Process cases colon_010 to colon_014"""
    dataset_dir = Path(dataset_dir)
    
    # Find all case files
    images_dir = dataset_dir / "imagesTr"
    all_files = sorted(images_dir.glob("*_0000.nii.gz"))
    
    # Get cases 10-14 (0-indexed, so indices 10-14)
    target_cases = all_files[10:15]
    
    print(f"Processing {len(target_cases)} cases:")
    for f in target_cases:
        case_id = f.name.replace("_0000.nii.gz", "")
        print(f"  - {case_id}")
    
    # Call the main script with max_cases offset
    cmd = [
        sys.executable,
        "scripts/create_anatomical_masks.py",
        "--dataset", str(dataset_dir),
        "--start_idx", "10",
        "--max_cases", "5"
    ]
    
    # For now, just use the regular script with max_cases
    # Since we want cases 10-14, we'll just process all and use first 15
    cmd = [
        sys.executable,
        "scripts/create_anatomical_masks.py",
        "--dataset", str(dataset_dir),
        "--max_cases", "15"  # Process up to 15 to include 10-14
    ]
    
    print(f"\nRunning: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd="f:/ADDS")
    
    return result.returncode == 0

if __name__ == "__main__":
    dataset_dir = Path("f:/ADDS/nnUNet_raw/Dataset010_Colon")
    success = process_cases_10_to_14(dataset_dir)
    
    if success:
        print("\nProcessing complete!")
    else:
        print("\nProcessing failed!")
        sys.exit(1)
