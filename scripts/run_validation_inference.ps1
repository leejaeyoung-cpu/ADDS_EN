# nnU-Net Validation Inference - Fixed Script
# Sets environment variables and runs inference on Fold 0 validation set

Write-Host "======================================" -ForegroundColor Cyan
Write-Host "nnU-Net Phase 1 Validation Inference" -ForegroundColor Cyan
Write-Host "======================================" -ForegroundColor Cyan
Write-Host ""

# Set environment variables
Write-Host "[1/4] Setting environment variables..." -ForegroundColor Yellow
$env:nnUNet_raw = "f:\ADDS\nnUNet_raw"
$env:nnUNet_preprocessed = "f:\ADDS\nnUNet_preprocessed"
$env:nnUNet_results = "f:\ADDS\nnUNet_results"

Write-Host "  nnUNet_raw: $env:nnUNet_raw" -ForegroundColor Green
Write-Host "  nnUNet_preprocessed: $env:nnUNet_preprocessed" -ForegroundColor Green
Write-Host "  nnUNet_results: $env:nnUNet_results" -ForegroundColor Green
Write-Host ""

# Verify model exists
Write-Host "[2/4] Verifying trained model..." -ForegroundColor Yellow
$model_path = "f:\ADDS\nnUNet_results\Dataset011_ColonMasked\nnUNetTrainer__nnUNetPlans__3d_fullres\fold_0\checkpoint_best.pth"
if (Test-Path $model_path) {
    $model_size_mb = [math]::Round((Get-Item $model_path).Length / 1MB, 2)
    Write-Host "  Model found: checkpoint_best.pth ($model_size_mb MB)" -ForegroundColor Green
}
else {
    Write-Host "  ERROR: Model not found at $model_path" -ForegroundColor Red
    exit 1
}
Write-Host ""

# Verify input images
Write-Host "[3/4] Verifying validation images..." -ForegroundColor Yellow
$input_dir = "f:\ADDS\nnUNet_raw\Dataset011_ColonMasked\imagesTs_fold0"
$image_count = (Get-ChildItem -Path $input_dir -Filter "*.nii.gz" | Measure-Object).Count
Write-Host "  Input directory: $input_dir" -ForegroundColor Green
Write-Host "  Validation images: $image_count cases" -ForegroundColor Green

if ($image_count -ne 26) {
    Write-Host "  WARNING: Expected 26 cases, found $image_count" -ForegroundColor Yellow
}
Write-Host ""

# Create output directory
$output_dir = "f:\ADDS\nnUNet_predictions\fold_0_validation"
New-Item -ItemType Directory -Force -Path $output_dir | Out-Null

# Run inference
Write-Host "[4/4] Running inference..." -ForegroundColor Yellow
Write-Host "  This will take approximately 1 hour (26 cases x 2-3 min)" -ForegroundColor Cyan
Write-Host "  Configuration: 3d_fullres, Fold 0" -ForegroundColor Cyan
Write-Host "  Device: CUDA (GPU)" -ForegroundColor Cyan
Write-Host ""

Write-Host "Starting inference at $(Get-Date -Format 'HH:mm:ss')..." -ForegroundColor Yellow
Write-Host ""

# Execute inference
try {
    nnUNetv2_predict `
        -i $input_dir `
        -o $output_dir `
        -d 011 `
        -c 3d_fullres `
        -f 0 `
        -device cuda `
        --verbose
    
    Write-Host ""
    Write-Host "======================================" -ForegroundColor Green
    Write-Host "Inference completed at $(Get-Date -Format 'HH:mm:ss')!" -ForegroundColor Green
    Write-Host "======================================" -ForegroundColor Green
    Write-Host ""
    Write-Host "Predictions saved to: $output_dir" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "Next step: Calculate actual Dice scores" -ForegroundColor Cyan
    Write-Host "Run: python f:\ADDS\scripts\calculate_actual_dice.py" -ForegroundColor Yellow
    
}
catch {
    Write-Host ""
    Write-Host "======================================" -ForegroundColor Red
    Write-Host "Inference failed!" -ForegroundColor Red
    Write-Host "======================================" -ForegroundColor Red
    Write-Host "Error: $_" -ForegroundColor Red
    exit 1
}
