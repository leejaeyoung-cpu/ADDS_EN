# nnU-Net Training Scripts for Dataset100_CRC_CT
# Run these commands after preprocessing is complete

# Set environment variables (run first!)
$env:nnUNet_raw = 'F:\ADDS\nnUNet_data\nnUNet_raw'
$env:nnUNet_preprocessed = 'F:\ADDS\nnUNet_data\nnUNet_preprocessed'
$env:nnUNet_results = 'F:\ADDS\nnUNet_data\nnUNet_results'

# Option 1: Train 2D network (fastest, recommended for single case)
# Estimated time: 1-2 hours with GPU
nnUNetv2_train 100 2d 0 --npz

# Option 2: Train 3D full resolution network (more accurate but slower)
# Estimated time: 3-6 hours with GPU
# nnUNetv2_train 100 3d_fullres 0

# Option 3: Train 3D low resolution network
# nnUNetv2_train 100 3d_lowres 0

# Option 4: Train all folds (for cross-validation)
# for ($i=0; $i -lt 5; $i++) { nnUNetv2_train 100 2d $i }

# After training, predict on test set (if you have one)
# nnUNetv2_predict -i /path/to/test/images -o /path/to/output -d 100 -c 2d -f 0

# Ensemble prediction (if trained multiple folds)
# nnUNetv2_predict -i /path/to/test/images -o /path/to/output -d 100 -c 2d -f all
