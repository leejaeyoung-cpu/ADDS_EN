"""
Visualize Hybrid Detection Results with Images
===============================================
Run hybrid detection on a single CT case and create visualization images
showing tumor candidates with bounding boxes and confidence scores
"""
import sys
from pathlib import Path
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.gridspec import GridSpec
import json

# Add project root
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.medical_imaging.inference.hybrid_predictor import HybridPredictor


def load_nifti(filepath):
    """Load NIfTI volume"""
    nii = nib.load(filepath)
    volume = nii.get_fdata()
    spacing = nii.header.get_zooms()
    return volume, spacing


def visualize_detection_results(
    volume, 
    candidates, 
    output_path="outputs/hybrid_detection_visualization.png",
    slice_idx=None
):
    """
    Create visualization with detected candidates
    
    Args:
        volume: 3D CT volume
        candidates: List of TumorCandidate objects
        output_path: Where to save the visualization
        slice_idx: Which slice to visualize (None = auto-select best)
    """
    # Auto-select slice with most candidates if not specified
    if slice_idx is None:
        # Since candidates fromrule-based detection come from 2D slices,
        # their slice_index might not match the 3D volume dimensions
        # Just pick the middle slice
        slice_idx = volume.shape[2] // 2
    
    # Ensure slice_idx is within bounds
    slice_idx = max(0, min(slice_idx, volume.shape[2] - 1))
    
    # Create figure with subplots
    fig = plt.figure(figsize=(20, 10))
    gs = GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.3)
    
    # Get CT slice
    ct_slice = volume[:, :, slice_idx]
    
    # Normalize to HU range for visualization
    ct_display = np.clip(ct_slice, -160, 240)
    ct_display = (ct_display + 160) / 400  # Normalize to 0-1
    
    # 1. Original CT slice
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(ct_display.T, cmap='gray', origin='lower')
    ax1.set_title(f'Original CT Slice (z={slice_idx})', fontsize=14, fontweight='bold')
    ax1.axis('off')
    
    # 2. CT with all candidates
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.imshow(ct_display.T, cmap='gray', origin='lower')
    ax2.set_title(f'All Candidates (n={len(candidates)})', fontsize=14, fontweight='bold')
    
    # Show first 100 candidates to avoid overcrowding
    # Since these are 2D detections, just show them on the selected slice
    display_candidates = candidates[:100]  # Limit for visualization
    for candidate in display_candidates:
        x, y = candidate.centroid
        # Estimate radius from area
        radius = max(5, min(50, np.sqrt(candidate.area_pixels / np.pi)))  # Clamp radius
        conf = candidate.confidence_score
        
        # Color based on confidence
        color = 'yellow' if conf > 0.8 else 'orange'
        alpha = 0.3 + 0.4 * conf
        
        circle = plt.Circle((x, y), radius, color=color, fill=False, 
                          linewidth=2, alpha=alpha)
        ax2.add_patch(circle)
    if len(candidates) > 100:
        ax2.text(10, 10, f'Showing 100/{len(candidates)} candidates', 
                color='yellow', fontsize=10, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='black', alpha=0.7))
    ax2.axis('off')
    
    # 3. High-confidence candidates only
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.imshow(ct_display.T, cmap='gray', origin='lower')
    
    high_conf_candidates = [c for c in candidates if c.confidence_score > 0.8]
    ax3.set_title(f'High-Confidence (>80%, n={len(high_conf_candidates)})', 
                 fontsize=14, fontweight='bold')
    
    # Show top 20 high-confidence
    for candidate in high_conf_candidates[:20]:
        x, y = candidate.centroid
        radius = max(5, min(50, np.sqrt(candidate.area_pixels / np.pi)))
        conf = candidate.confidence_score
        
        circle = plt.Circle((x, y), radius, color='red', fill=False, 
                          linewidth=3, alpha=0.8)
        ax3.add_patch(circle)
        
        # Add confidence label
        ax3.text(x, y - radius - 5, f'{conf:.1%}', 
                color='red', fontsize=9, fontweight='bold',
                ha='center', va='bottom',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    ax3.axis('off')
    
    # 4. 3D scatter plot of all candidates
    ax4 = fig.add_subplot(gs[1, 0], projection='3d')
    if candidates:
        # Extract centroids and slice indices for 3D positions
        positions = np.array([[c.centroid[0], c.centroid[1], c.slice_index] for c in candidates])
        confidences = np.array([c.confidence_score for c in candidates])
        
        scatter = ax4.scatter(positions[:, 0], positions[:, 1], positions[:, 2],
                            c=confidences, cmap='hot', s=50, alpha=0.6)
        ax4.set_xlabel('X')
        ax4.set_ylabel('Y')
        ax4.set_zlabel('Z (Slice)')
        ax4.set_title('3D Candidate Distribution', fontsize=14, fontweight='bold')
        plt.colorbar(scatter, ax=ax4, label='Confidence', shrink=0.6)
    
    # 5. Statistics text
    ax5 = fig.add_subplot(gs[1, 1])
    ax5.axis('off')
    
    high_conf_count = len([c for c in candidates if c.confidence_score > 0.8])
    med_conf_count = len([c for c in candidates if 0.5 < c.confidence_score <= 0.8])
    max_conf = max([c.confidence_score for c in candidates]) if candidates else 0
    mean_conf = np.mean([c.confidence_score for c in candidates]) if candidates else 0
    
    stats_text = f"""
    Detection Statistics
    {'='*40}
    Total Candidates: {len(candidates)}
    High-Confidence (>80%): {high_conf_count}
    Medium-Confidence (50-80%): {med_conf_count}
    
    Max Confidence: {max_conf* 100:.2f}%
    Mean Confidence: {mean_conf * 100:.2f}%
    
    Volume Shape: {volume.shape}
    Selected Slice: {slice_idx}/{volume.shape[2]}
    Visualization: Showing sample candidates
    """
    
    ax5.text(0.1, 0.5, stats_text, fontsize=12, family='monospace',
            verticalalignment='center',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    # 6. Confidence histogram
    ax6 = fig.add_subplot(gs[1, 2])
    if candidates:
        confidences = [c.confidence_score for c in candidates]
        ax6.hist(confidences, bins=20, color='steelblue', alpha=0.7, edgecolor='black')
        ax6.axvline(x=0.8, color='red', linestyle='--', linewidth=2, label='High-Conf Threshold')
        ax6.set_xlabel('Confidence Score', fontsize=12)
        ax6.set_ylabel('Number of Candidates', fontsize=12)
        ax6.set_title('Confidence Distribution', fontsize=14, fontweight='bold')
        ax6.legend()
        ax6.grid(True, alpha=0.3)
    
    plt.suptitle('Hybrid Detection System - Tumor Candidate Visualization', 
                fontsize=16, fontweight='bold', y=0.98)
    
    # Save figure
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n[+] Visualization saved to: {output_path}")
    
    return output_path


def main():
    """Run hybrid detection on colon_001 and visualize"""
    
    # Setup paths
    case_name = "colon_001"
    image_path = f"data/medical_decathlon/Task10_Colon/imagesTr/{case_name}.nii.gz"
    output_path = f"outputs/hybrid_detection_{case_name}.png"
    
    print(f"\n[*] Running Hybrid Detection on {case_name}...")
    print("=" * 60)
    
    # Load volume
    print(f"\n[*] Loading: {image_path}")
    volume, spacing = load_nifti(image_path)
    print(f"   Volume shape: {volume.shape}")
    print(f"   Spacing: {spacing}")
    
    # Initialize predictor (rule-only mode)
    print(f"\n[*] Initializing Hybrid Predictor (rule-only mode)...")
    predictor = HybridPredictor(
        dl_checkpoint_path=None,  # No DL model needed for rule-only mode
        device="cpu",
        detection_mode="rule_only"
    )
    
    # Run prediction
    print(f"\n[*] Running detection...")
    result = predictor.predict(volume, spacing)
    
    # Extract candidates
    candidates = result.get('candidates', [])
    print(f"\n[*] Detection Results:")
    print(f"   Total candidates: {len(candidates)}")
    print(f"   High-confidence (>80%): {len([c for c in candidates if c.confidence_score > 0.8])}")
    if candidates:
        max_conf = max([c.confidence_score for c in candidates])
        print(f"   Max confidence: {max_conf * 100:.2f}%")
    
    # Create visualization
    print(f"\n[*] Creating visualization...")
    viz_path = visualize_detection_results(
        volume=volume,
        candidates=candidates,
        output_path=output_path
    )
    
    # Save results JSON
    json_path = output_path.replace('.png', '_results.json')
    with open(json_path, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"[+] Results saved to: {json_path}")
    
    print(f"\n[+] Done! Check the visualization at: {viz_path}")
    
    return viz_path



if __name__ == "__main__":
    main()
