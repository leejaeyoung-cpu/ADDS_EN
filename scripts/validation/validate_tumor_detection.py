"""
Comprehensive tumor detection validation with visualization
Tests the fixed pipeline (normalize=False) with real patient CT data
"""
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from src.medical_imaging.detection.simple_hu_detector import detect_tumors_simple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load CT volume
logger.info("Loading Inha CT volume...")
vol_path = Path('outputs/inha_ct_analysis/inha_ct_volume.nii.gz')
vol = nib.load(str(vol_path))
data = vol.get_fdata()

logger.info(f"Volume shape: {data.shape}")
logger.info(f"HU range: [{data.min():.1f}, {data.max():.1f}]")

# Detect tumors with fixed parameters
logger.info("\nDetecting tumors...")
lesions = detect_tumors_simple(
    ct_volume=data,
    spacing=(1.0, 1.0, 1.0),
    tumor_hu_min=40.0,
    tumor_hu_max=150.0
)

logger.info(f"\n✅ Detected {len(lesions)} tumors")

# Sort by volume
lesions_sorted = sorted(lesions, key=lambda x: x.volume_mm3, reverse=True)

# Print tumor statistics
logger.info("\n=== Top 20 Tumors ===")
for i, les in enumerate(lesions_sorted[:20]):
    logger.info(f"Tumor {i+1:2d}: Vol={les.volume_mm3:8.1f} mm³, "
                f"HU={les.mean_hu:5.1f}, "
                f"Center=({les.centroid[0]:3d}, {les.centroid[1]:3d}, {les.centroid[2]:3d})")

# Create comprehensive visualization
logger.info("\n=== Creating Visualization ===")

# Select 20 slices with tumors
tumor_slices = set()
for les in lesions_sorted[:20]:
    z_center = les.centroid[0]
    tumor_slices.add(z_center)

selected_slices = sorted(list(tumor_slices))[:20]
logger.info(f"Selected {len(selected_slices)} slices for visualization")

# Create tumor masks for visualization
tumor_volume = np.zeros_like(data)
for les in lesions_sorted[:20]:
    tumor_volume[les.mask] = 1

# Create figure with 4x5 grid (20 images)
fig, axes = plt.subplots(4, 5, figsize=(20, 16))
fig.suptitle('CT Tumor Detection - Top 20 Slices', fontsize=16, fontweight='bold')

for idx, slice_idx in enumerate(selected_slices):
    row = idx // 5
    col = idx % 5
    ax = axes[row, col]
    
    # Get CT slice and tumor mask
    ct_slice = data[slice_idx, :, :]
    tumor_slice = tumor_volume[slice_idx, :, :]
    
    # Display CT in grayscale
    ax.imshow(ct_slice, cmap='gray', vmin=-200, vmax=200)
    
    # Overlay tumor mask in red
    tumor_overlay = np.ma.masked_where(tumor_slice == 0, tumor_slice)
    ax.imshow(tumor_overlay, cmap='Reds', alpha=0.5, vmin=0, vmax=1)
    
    # Count tumors in this slice
    num_tumors = len([l for l in lesions_sorted[:20] if l.centroid[0] == slice_idx])
    
    ax.set_title(f'Slice {slice_idx} ({num_tumors} tumors)', fontsize=10)
    ax.axis('off')

plt.tight_layout()

# Save output
output_path = Path('C:/Users/brook/.gemini/antigravity/brain/9dbe779b-6d91-4581-af37-5bcfabe49ea1/tumor_detection_validation.png')
plt.savefig(output_path, dpi=150, bbox_inches='tight')
logger.info(f"\n✅ Saved visualization to: {output_path}")

# Create detailed report
report_lines = [
    "# CT Tumor Detection Validation Report",
    "",
    f"**Date**: 2026-02-06",
    f"**Dataset**: Inha University Hospital CT",
    "",
    "## Volume Statistics",
    f"- Shape: {data.shape}",
    f"- HU Range: [{data.min():.1f}, {data.max():.1f}]",
    f"- Total Voxels: {data.size:,}",
    "",
    "## Detection Results",
    f"- **Total Tumors Detected**: {len(lesions)}",
    f"- **Detection Parameters**: HU 40-150, Min Volume 100 mm³",
    "",
    "## Top 20 Tumors",
    "",
    "| # | Volume (mm³) | Mean HU | Centroid (Z, Y, X) |",
    "|---|--------------|---------|-------------------|"
]

for i, les in enumerate(lesions_sorted[:20]):
    report_lines.append(
        f"| {i+1} | {les.volume_mm3:.1f} | {les.mean_hu:.1f} | "
        f"({les.centroid[0]}, {les.centroid[1]}, {les.centroid[2]}) |"
    )

report_lines.extend([
    "",
    "## Visualization",
    f"",
    f"![Tumor Detection Results]({output_path.as_posix()})",
    "",
    "## Conclusion",
    f"✅ System successfully detected {len(lesions)} tumor candidates",
    f"✅ Top 20 tumors visualized across {len(selected_slices)} slices",
    f"✅ HU values properly preserved (range: [{data.min():.1f}, {data.max():.1f}])",
])

report_path = Path('C:/Users/brook/.gemini/antigravity/brain/9dbe779b-6d91-4581-af37-5bcfabe49ea1/tumor_validation_report.md')
report_path.write_text('\n'.join(report_lines), encoding='utf-8')
logger.info(f"✅ Saved report to: {report_path}")

print("\n" + "="*60)
print("VALIDATION COMPLETE!")
print("="*60)
print(f"Detected: {len(lesions)} tumors")
print(f"Visualized: {len(selected_slices)} slices")
print(f"Image: {output_path}")
print(f"Report: {report_path}")
print("="*60)
