"""
Visualize Hybrid Detection Results
===================================
Generate images with tumor candidates marked with bounding boxes and confidence scores
"""

import sys
from pathlib import Path
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.gridspec import GridSpec

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.medical_imaging.inference.hybrid_predictor import HybridPredictor


def load_nifti(filepath: str):
    """Load NIfTI volume and spacing"""
    nifti_img = nib.load(filepath)
    volume = np.asarray(nifti_img.dataobj, dtype=np.float32)
    header = nifti_img.header
    spacing = header.get_zooms()[:3]
    spacing = (spacing[2], spacing[1], spacing[0])
    return volume, spacing


def visualize_detection_results(
    case_path: str,
    output_dir: str = "outputs/visualizations",
    mode: str = "rule_only",
    max_slices_to_show: int = 6
):
    """
    Visualize detection results with bounding boxes
    
    Args:
        case_path: Path to NIfTI file
        output_dir: Output directory for images
        mode: Detection mode
        max_slices_to_show: Maximum number of slices to visualize
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    case_id = Path(case_path).stem.replace('.nii', '')
    
    print(f"Visualizing: {case_id}")
    print("=" * 80)
    
    # Load volume
    volume, spacing = load_nifti(case_path)
    print(f"Volume shape: {volume.shape}")
    print(f"Spacing: {spacing} mm")
    
    # Initialize predictor
    predictor = HybridPredictor(
        dl_checkpoint_path=None,
        device="cuda",
        detection_mode=mode
    )
    
    # Run detection
    print("Running detection...")
    results = predictor.predict(
        volume=volume,
        spacing=spacing,
        return_candidates=True
    )
    
    mask = results['mask']
    candidates = results.get('candidates', [])
    
    print(f"Detected {len(candidates)} candidates")
    
    # Group candidates by slice
    candidates_by_slice = {}
    for c in candidates:
        z = c.slice_index
        if z not in candidates_by_slice:
            candidates_by_slice[z] = []
        candidates_by_slice[z].append(c)
    
    # Get slices with most high-confidence candidates
    slice_scores = {}
    for z, cands in candidates_by_slice.items():
        high_conf = [c for c in cands if c.confidence_score > 0.5]
        slice_scores[z] = len(high_conf)
    
    top_slices = sorted(slice_scores.items(), key=lambda x: x[1], reverse=True)[:max_slices_to_show]
    
    print(f"Visualizing top {len(top_slices)} slices with most detections")
    
    # Create visualization (OpenAI-style: larger, higher quality)
    n_slices = len(top_slices)
    fig = plt.figure(figsize=(24, 6 * n_slices))  # Increased from 20x4
    gs = GridSpec(n_slices, 3, figure=fig, hspace=0.3, wspace=0.3)
    
    # Apply proper CT windowing for abdomen (W400, L40)
    window_min = -160  # L - W/2 = 40 - 400/2
    window_max = 240   # L + W/2 = 40 + 400/2
    
    for idx, (z, count) in enumerate(top_slices):
        # Ensure z is within bounds
        if z >= volume.shape[2]:
            print(f"Warning: Slice {z} out of bounds, skipping")
            continue
        
        slice_img = volume[:, :, z]
        slice_candidates = candidates_by_slice[z]
        
        # Sort by confidence
        slice_candidates = sorted(slice_candidates, key=lambda x: x.confidence_score, reverse=True)
        
        # Original image with proper abdomen windowing
        ax1 = fig.add_subplot(gs[idx, 0])
        ax1.imshow(slice_img.T, cmap='gray', origin='lower', vmin=window_min, vmax=window_max)
        ax1.set_title(f'Slice {z} - Original\n{len(slice_candidates)} candidates',  fontsize=14, fontweight='bold')
        ax1.axis('off')
        
        # With bounding boxes
        ax2 = fig.add_subplot(gs[idx, 1])
        ax2.imshow(slice_img.T, cmap='gray', origin='lower', vmin=window_min, vmax=window_max)
        
        for i, cand in enumerate(slice_candidates[:20]):  # Top 20 per slice
            bbox = cand.bounding_box
            conf = cand.confidence_score
            
            # Color by confidence
            if conf > 0.8:
                color = 'red'
            elif conf > 0.5:
                color = 'yellow'
            else:
                color = 'cyan'
            
            # Draw bounding box (note: bbox is (min_row, min_col, max_row, max_col))
            width = bbox[3] - bbox[1]
            height = bbox[2] - bbox[0]
            
            rect = patches.Rectangle(
                (bbox[1], bbox[0]),  # (x, y) in image coords
                width, height,
                linewidth=3,  # Thicker for better visibility
                edgecolor=color,
                facecolor='none'
            )
            ax2.add_patch(rect)
            
            # Label (only for top 5)
            if i < 5:
                ax2.text(
                    bbox[1] + width/2, bbox[0] - 5,
                    f'{conf:.2f}',
                    color='white',
                    fontsize=10,  # Larger font
                    fontweight='bold',
                    ha='center',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor=color, alpha=0.7)
                )
        
        ax2.set_title(f'Detections (Top 20)\nRed: >0.8, Yellow: >0.5', fontsize=14, fontweight='bold')
        ax2.axis('off')
        
        # Statistics
        ax3 = fig.add_subplot(gs[idx, 2])
        ax3.axis('off')
        
        high_conf = [c for c in slice_candidates if c.confidence_score > 0.5]
        
        stats_text = f"Slice {z} Statistics:\\n"
        stats_text += "=" * 40 + "\\n\\n"
        stats_text += f"Total candidates: {len(slice_candidates)}\\n"
        stats_text += f"High conf (>0.5): {len(high_conf)}\\n"
        stats_text += f"Max confidence: {max(c.confidence_score for c in slice_candidates):.2f}\\n\\n"
        
        stats_text += "Top 5 Candidates:\\n"
        stats_text += "-" * 40 + "\\n"
        for i, c in enumerate(slice_candidates[:5], 1):
            stats_text += f"#{i}:\\n"
            stats_text += f"  Area: {c.area_mm2:.1f} mm²\\n"
            stats_text += f"  HU: {c.mean_hu:.1f}\\n"
            stats_text += f"  Conf: {c.confidence_score:.2f}\\n\\n"
        
        ax3.text(
            0.05, 0.95, stats_text,
            transform=ax3.transAxes,
            fontsize=11,  # Larger for readability
            verticalalignment='top',
            fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3)
        )
    
    # Save with high DPI for professional quality (OpenAI-style)
    output_path = output_dir / f"{case_id}_detection_visualization.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')  # Increased from 150
    plt.close()
    
    print(f"\\n[OK] Saved visualization: {output_path}")
    print("")
    
    return str(output_path)


def visualize_all_cases(
    data_dir: str = "data/medical_decathlon/Task10_Colon",
    num_cases: int = 3,
    output_dir: str = "outputs/visualizations"
):
    """Visualize multiple cases"""
    dataset_root = Path(data_dir)
    images_dir = dataset_root / "imagesTr"
    
    image_files = sorted(images_dir.glob("*.nii.gz"))[:num_cases]
    
    print("=" * 80)
    print("DETECTION VISUALIZATION")
    print("=" * 80)
    print(f"Cases: {num_cases}")
    print(f"Output: {output_dir}")
    print("")
    
    saved_files = []
    for idx, image_path in enumerate(image_files, 1):
        print(f"[{idx}/{num_cases}]")
        output_path = visualize_detection_results(
            str(image_path),
            output_dir=output_dir,
            max_slices_to_show=3  # 3 slices per case
        )
        saved_files.append(output_path)
    
    print("=" * 80)
    print("VISUALIZATION COMPLETE")
    print("=" * 80)
    print(f"\\nSaved {len(saved_files)} visualization files:")
    for f in saved_files:
        print(f"  - {f}")
    print(f"\\nFolder: {Path(output_dir).absolute()}")
    print("")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Visualize detection results")
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data/medical_decathlon/Task10_Colon",
        help="Path to data"
    )
    parser.add_argument(
        "--num_cases",
        type=int,
        default=3,
        help="Number of cases to visualize"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/visualizations",
        help="Output directory"
    )
    
    args = parser.parse_args()
    
    visualize_all_cases(
        data_dir=args.data_dir,
        num_cases=args.num_cases,
        output_dir=args.output_dir
    )
