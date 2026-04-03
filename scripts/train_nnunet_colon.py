"""
nnU-Net Training Pipeline for Colon Tumor Detection
====================================================
Complete training pipeline using existing Dataset010_Colon
"""
import subprocess
import sys
from pathlib import Path
import os


def check_environment():
    """Check nnU-Net environment variables"""
    print("\n" + "="*60)
    print("1. CHECKING ENVIRONMENT")
    print("="*60)
    
    required_vars = ['nnUNet_raw', 'nnUNet_preprocessed', 'nnUNet_results']
    
    for var in required_vars:
        value = os.getenv(var)
        if value:
            print(f"✅ {var}: {value}")
        else:
            print(f"❌ {var}: NOT SET")
            print(f"\n[ERROR] Please set environment variable:")
            print(f"  $env:{var} = 'C:\\nnUNet_data\\{var}'")
            return False
    
    return True


def verify_dataset():
    """Verify dataset structure"""
    print("\n" + "="*60)
    print("2. VERIFYING DATASET")
    print("="*60)
    
    dataset_path = Path(os.getenv('nnUNet_raw')) / "Dataset010_Colon"
    
    if not dataset_path.exists():
        print(f"❌ Dataset not found: {dataset_path}")
        return False
    
    print(f"✅ Dataset found: {dataset_path}")
    
    # Check structure
    imagesTr = dataset_path / "imagesTr"
    labelsTr = dataset_path / "labelsTr"
    
    num_images = len(list(imagesTr.glob("*.nii.gz"))) if imagesTr.exists() else 0
    num_labels = len(list(labelsTr.glob("*.nii.gz"))) if labelsTr.exists() else 0
    
    print(f"  - Training images: {num_images}")
    print(f"  - Training labels: {num_labels}")
    
    if num_images == 0 or num_labels == 0:
        print("❌ No training data found!")
        return False
    
    if num_images != num_labels:
        print(f"⚠️  Mismatch: {num_images} images vs {num_labels} labels")
    
    return True


def run_preprocessing():
    """Run nnU-Net preprocessing"""
    print("\n" + "="*60)
    print("3. PREPROCESSING")
    print("="*60)
    
    print("\n[*] Starting nnU-Net preprocessing...")
    print("    This will:")
    print("    - Analyze dataset properties")
    print("    - Create training plans")
    print("    - Preprocess all images")
    print("    - Estimate: 10-30 minutes")
    print()
    
    cmd = [
        "nnUNetv2_plan_and_preprocess",
        "-d", "010",
        "--verify_dataset_integrity"
    ]
    
    print(f"[CMD] {' '.join(cmd)}")
    print()
    
    try:
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=False,
            text=True
        )
        print("\n✅ Preprocessing complete!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Preprocessing failed: {e}")
        return False
    except FileNotFoundError:
        print("\n❌ nnUNetv2_plan_and_preprocess not found!")
        print("   Make sure nnU-Net is installed:")
        print("   pip install nnunetv2")
        return False


def start_training(fold=0):
    """Start nnU-Net training"""
    print("\n" + "="*60)
    print("4. TRAINING")
    print("="*60)
    
    print(f"\n[*] Starting nnU-Net training (Fold {fold})...")
    print("    Configuration: 3d_fullres")
    print("    Trainer: nnUNetTrainer")
    print("    Estimated time: 12-48 hours (depends on GPU)")
    print()
    print("    ⚠️  This will take a LONG time!")
    print("    Consider running in background or tmux/screen")
    print()
    
    cmd = [
        "nnUNetv2_train",
        "010",  # Dataset ID
        "3d_fullres",  # Configuration
        str(fold),  # Fold number
        "--npz"  # Save space
    ]
    
    print(f"[CMD] {' '.join(cmd)}")
    print()
    
    response = input("Start training now? (y/n): ").strip().lower()
    
    if response != 'y':
        print("\n[!] Training cancelled by user")
        print(f"\n[INFO] To start training later, run:")
        print(f"  {' '.join(cmd)}")
        return False
    
    try:
        subprocess.run(cmd, check=True)
        print("\n✅ Training complete!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Training failed: {e}")
        return False
    except KeyboardInterrupt:
        print("\n\n[!] Training interrupted by user")
        print("    Training can be resumed by running the same command")
        return False


def main():
    """Main training pipeline"""
    print("\n" + "="*80)
    print("nnU-Net TRAINING PIPELINE")
    print("Dataset: 010_Colon (Medical Decathlon)")
    print("="*80)
    
    # Step 1: Check environment
    if not check_environment():
        sys.exit(1)
    
    # Step 2: Verify dataset
    if not verify_dataset():
        sys.exit(1)
    
    # Step 3: Preprocessing
    print("\n" + "-"*60)
    response = input("\nRun preprocessing? (y/n): ").strip().lower()
    
    if response == 'y':
        if not run_preprocessing():
            print("\n[ERROR] Preprocessing failed. Cannot continue.")
            sys.exit(1)
    else:
        print("[!] Skipping preprocessing (assuming already done)")
    
    # Step 4: Training
    print("\n" + "-"*60)
    print("\n[INFO] Ready to start training!")
    print("       You can train all 5 folds or just fold 0")
    print()
    
    fold_choice = input("Train which fold? (0-4, or 'all'): ").strip().lower()
    
    if fold_choice == 'all':
        for fold in range(5):
            print(f"\n[*] Training fold {fold}/5...")
            if not start_training(fold):
                break
    elif fold_choice.isdigit() and 0 <= int(fold_choice) <= 4:
        start_training(int(fold_choice))
    else:
        print("[!] Invalid choice. Exiting.")
    
    print("\n" + "="*80)
    print("PIPELINE COMPLETE")
    print("="*80)
    print()
    print("Next steps:")
    print("  1. Wait for training to complete")
    print("  2. Find best model: nnUNetv2_find_best_configuration -d 010")
    print("  3. Run inference: nnUNetv2_predict -i INPUT -o OUTPUT -d 010 -c 3d_fullres")
    print()


if __name__ == "__main__":
    main()
