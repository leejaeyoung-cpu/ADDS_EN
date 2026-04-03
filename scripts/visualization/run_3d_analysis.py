"""
Full 3D Tumor Analysis Pipeline

Combines:
1. 2D detection results
2. 3D clustering
3. 3D measurements
4. Visualization
"""
import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import nibabel as nib

# Import our 3D clustering module
from tumor_3d_clustering import cluster_detections_3d, calculate_3d_size, format_tumor_summary, Tumor3D


def load_detection_results(json_path: Path) -> Tuple[List[Dict], Tuple[float, float, float]]:
    """Load detection results and spacing"""
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Extract spacing
    spacing = tuple(data.get('spacing', (0.78125, 0.78125, 5.0)))
    
    # Extract detection results
    results = data.get('results', [])
    
    return results, spacing


def visualize_3d_tumors(tumors_3d: List[Tumor3D], volume_path: Path, output_dir: Path):
    """
    Create 3D tumor visualization
    
    Shows:
    - Top 6 tumors by size
    - Multiple views per tumor
    - 3D measurements overlay
    """
    # Load volume
    nii = nib.load(volume_path)
    volume = nii.get_fdata()
    
    # Sort by volume
    tumors_sorted = sorted(tumors_3d, key=lambda t: t.volume_mm3, reverse=True)[:6]
    
    # Create figure with 2 rows x 3 columns
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    fig.suptitle('Top 6 Tumors by Volume (3D Analysis)', fontsize=20, fontweight='bold', y=0.98)
    
    for idx, (ax, tumor) in enumerate(zip(axes, tumors_sorted)):
        # Get middle slice of this tumor
        mid_slice_idx = tumor.slices[len(tumor.slices) // 2]
        
        # Get the slice
        ct_slice = volume[mid_slice_idx, :, :]
        
        # Display
        ax.imshow(ct_slice, cmap='gray', vmin=-200, vmax=300)
        
        # Draw tumor locations on this slice
        slice_positions = [i for i, s in enumerate(tumor.slices) if s == mid_slice_idx]
        
        for pos in slice_positions:
            cx, cy = tumor.centroids_2d[pos]
            area_mm2 = tumor.areas_mm2[pos]
            radius_px = np.sqrt(area_mm2 / np.pi) / 0.78125  # Convert to pixels
            
            # Draw circle
            circle = Circle((cy, cx), radius_px, fill=False, color='red', linewidth=2)
            ax.add_patch(circle)
        
        # Add text annotations
        info_text = (
            f"Tumor #{tumor.tumor_id}\n"
            f"Volume: {tumor.volume_mm3:.0f} mm³\n"
            f"Diameter: {tumor.max_diameter_mm:.1f} mm\n"
            f"Slices: {len(tumor.slices)}\n"
            f"Conf: {tumor.confidence:.0f}%"
        )
        
        ax.text(10, 30, info_text, color='yellow', fontsize=10, 
                fontweight='bold', verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))
        
        ax.set_title(f'Tumor #{tumor.tumor_id} - Slice {mid_slice_idx}', 
                    fontsize=12, fontweight='bold')
        ax.axis('off')
    
    plt.tight_layout()
    
    # Save
    output_path = output_dir / '3d_tumor_analysis.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"[+] Saved visualization: {output_path}")
    
    plt.close()


def save_clinical_report(tumors_3d: List[Tumor3D], summary: Dict, output_dir: Path):
    """
    Generate clinical report in markdown format
    """
    report_path = output_dir / '3d_tumor_report.md'
    
    # Sort by volume
    tumors_sorted = sorted(tumors_3d, key=lambda t: t.volume_mm3, reverse=True)
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# 3D CT Tumor Analysis Report\n\n")
        f.write(f"**Analysis Date**: {np.datetime64('now')}\n\n")
        f.write("---\n\n")
        
        f.write("## Executive Summary\n\n")
        f.write(f"- **Total Tumors Identified**: {summary['tumor_count']}\n")
        f.write(f"- **Total Volume**: {summary['total_volume_ml']:.2f} mL\n")
        f.write(f"- **Largest Tumor**: {summary['largest_tumor_mm']:.1f} mm diameter\n")
        f.write(f"- **Slices Analyzed**: {summary['tumor_slices']}\n\n")
        
        f.write("---\n\n")
        f.write("## Individual Tumor Analysis\n\n")
        
        # Create table
        f.write("| ID | Volume (mm³) | Diameter (mm) | Z-Range (mm) | Slices | Confidence |\n")
        f.write("|----|-------------|---------------|-------------|--------|------------|\n")
        
        for tumor in tumors_sorted[:20]:  # Top 20
            f.write(f"| {tumor.tumor_id} | {tumor.volume_mm3:.0f} | "
                   f"{tumor.max_diameter_mm:.1f} | {tumor.z_range_mm:.1f} | "
                   f"{len(tumor.slices)} | {tumor.confidence:.0f}% |\n")
        
        f.write("\n---\n\n")
        f.write("## Size Distribution\n\n")
        
        # Group by size
        small = sum(1 for t in tumors_3d if t.max_diameter_mm < 10)
        medium = sum(1 for t in tumors_3d if 10 <= t.max_diameter_mm < 20)
        large = sum(1 for t in tumors_3d if t.max_diameter_mm >= 20)
        
        f.write(f"- **Small (<10mm)**: {small} tumors\n")
        f.write(f"- **Medium (10-20mm)**: {medium} tumors\n")
        f.write(f"- **Large (≥20mm)**: {large} tumors\n\n")
        
        f.write("---\n\n")
        f.write("## Clinical Significance\n\n")
        f.write("> **Note**: This is an automated AI analysis. ")
        f.write("All findings must be reviewed and confirmed by a qualified radiologist.\n\n")
        
        # Flag large tumors
        if large > 0:
            f.write("### ⚠️ Large Lesions Detected\n\n")
            for tumor in tumors_sorted:
                if tumor.max_diameter_mm >= 20:
                    f.write(f"- **Tumor #{tumor.tumor_id}**: {tumor.max_diameter_mm:.1f}mm diameter, "
                           f"{tumor.volume_mm3:.0f}mm³ volume\n")
            f.write("\n")
        
        f.write("### Recommendations\n\n")
        f.write("1. Radiologist review of all detected lesions\n")
        f.write("2. Clinical correlation with patient history\n")
        f.write("3. Consider follow-up imaging for lesions >10mm\n")
        f.write("4. Pathological confirmation if clinically indicated\n\n")
        
    print(f"[+] Saved clinical report: {report_path}")


def main():
    print("="*80)
    print("FULL 3D TUMOR ANALYSIS PIPELINE")
    print("="*80)
    
    # Paths
    # CRITICAL FIX: Use detection WITH masks (same detection run!)
    detection_json = Path("outputs/inha_ct_detection_with_masks/detection_summary.json")
    volume_path = Path("outputs/inha_ct_analysis/inha_ct_volume.nii.gz")
    output_dir = Path("outputs/inha_3d_analysis")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n[*] Loading detection results: {detection_json}")
    results, spacing = load_detection_results(detection_json)
    print(f"[*] Spacing: {spacing}")
    print(f"[*] Total detections: {len(results)} slices")
    
    # 3D clustering
    print(f"\n[*] Running 3D clustering...")
    tumors_3d = cluster_detections_3d(results, spacing, max_distance_mm=20.0)
    print(f"[+] Identified {len(tumors_3d)} 3D tumors")
    
    # Calculate measurements
    print(f"\n[*] Calculating 3D measurements...")
    for tumor in tumors_3d:
        calculate_3d_size(tumor, spacing)
    
    # Count total detections
    total_detections = sum(len(r.get('detections', [])) for r in results)
    total_slices = len([r for r in results if len(r.get('detections', [])) > 0])
    
    # Format summary
    summary = format_tumor_summary(tumors_3d, total_detections, total_slices)
    
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"3D Tumors: {summary['tumor_count']}")
    print(f"Total detections: {total_detections}")
    print(f"Tumor-positive slices: {total_slices}")
    print(f"Largest tumor: {summary['largest_tumor_mm']:.1f} mm")
    print(f"Total volume: {summary['total_volume_ml']:.2f} mL")
    print(f"\n{summary['summary_text']}")
    
    # Show top tumors
    print("\n" + "="*80)
    print("TOP 10 TUMORS")
    print("="*80)
    tumors_sorted = sorted(tumors_3d, key=lambda t: t.volume_mm3, reverse=True)[:10]
    for tumor in tumors_sorted:
        print(f"  Tumor #{tumor.tumor_id}: {tumor.max_diameter_mm:.1f}mm diameter, "
              f"{tumor.volume_mm3:.0f}mm^3, {len(tumor.slices)} slices, conf={tumor.confidence:.0f}%")
    
    # Save results
    print("\n" + "="*80)
    print("SAVING RESULTS")
    print("="*80)
    
    # Save JSON
    json_output = output_dir / 'tumors_3d.json'
    with open(json_output, 'w', encoding='utf-8') as f:
        json.dump({
            'summary': summary,
            'tumors': [
                {
                    'tumor_id': t.tumor_id,
                    'volume_mm3': float(t.volume_mm3),
                    'max_diameter_mm': float(t.max_diameter_mm),
                    'z_range_mm': float(t.z_range_mm),
                    'num_slices': len(t.slices),
                    'slices': t.slices,  # For reference
                    'confidence': float(t.confidence),
                    'mean_hu': float(t.mean_hu),
                    # CRITICAL FIX: Store detection indices for perfect reconstruction
                    'detection_indices': t.detection_indices if t.detection_indices else []
                }
                for t in tumors_3d
            ]
        }, f, indent=2)
    print(f"[+] Saved JSON: {json_output}")
    
    # Create visualizations
    print(f"\n[*] Creating visualizations...")
    visualize_3d_tumors(tumors_3d, volume_path, output_dir)
    
    # Create clinical report
    print(f"\n[*] Generating clinical report...")
    save_clinical_report(tumors_3d, summary, output_dir)
    
    print("\n" + "="*80)
    print("[OK] 3D ANALYSIS COMPLETE")
    print("="*80)
    print(f"\nResults saved to: {output_dir.absolute()}")


if __name__ == "__main__":
    main()
