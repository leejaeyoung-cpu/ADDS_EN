"""
Monitor full dataset processing and auto-generate reports when complete
"""
import subprocess
import time
import sys
from pathlib import Path


def check_processing_status(dataset_dir):
    """Check how many cases have been processed"""
    dataset_dir = Path(dataset_dir)
    body_masks_dir = dataset_dir / "body_masks"
    
    if not body_masks_dir.exists():
        return 0
    
    body_masks = list(body_masks_dir.glob("*_body.nii.gz"))
    return len(body_masks)


def wait_for_completion_and_generate_reports():
    """Wait for processing to complete, then generate reports"""
    dataset_dir = Path("f:/ADDS/nnUNet_raw/Dataset010_Colon")
    target_count = 126
    
    print("="*70)
    print("MONITORING 126-CASE PROCESSING")
    print("="*70)
    print(f"Target: {target_count} cases")
    print(f"Checking every 5 minutes...")
    print("="*70)
    print()
    
    last_count = 0
    while True:
        current_count = check_processing_status(dataset_dir)
        
        if current_count != last_count:
            print(f"[{time.strftime('%H:%M:%S')}] Progress: {current_count}/{target_count} cases ({current_count/target_count*100:.1f}%)")
            last_count = current_count
        
        if current_count >= target_count:
            print("\n" + "="*70)
            print("PROCESSING COMPLETE!")
            print("="*70)
            print(f"Total cases processed: {current_count}")
            print()
            break
        
        # Wait 5 minutes
        time.sleep(300)
    
    # Generate detailed reports
    print("Starting detailed report generation...")
    print()
    
    cmd = [sys.executable, "scripts/generate_detailed_reports.py"]
    result = subprocess.run(cmd, cwd="f:/ADDS")
    
    if result.returncode == 0:
        print("\n" + "="*70)
        print("ALL WORK COMPLETE!")
        print("="*70)
        print("1. 126 cases processed")
        print("2. Detailed reports generated")
        print("3. Ready for nnU-Net training")
        print("="*70)
    else:
        print("\n[ERROR] Report generation failed")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(wait_for_completion_and_generate_reports())
