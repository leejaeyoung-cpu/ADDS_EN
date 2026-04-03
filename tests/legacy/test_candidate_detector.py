"""
Test txt Pipeline Integration on Medical Decathlon Data
OpenAI-style candidate detection test
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from pathlib import Path

# Import new detection module
from medical_imaging.detection import TumorDetector, TumorCandidate, merge_candidates

def load_nifti(path):
    """Load NIfTI file"""
    nii = nib.load(str(path))
    volume = nii.get_fdata()
    spacing = nii.header.get_zooms()
    return volume, spacing


def visualize_candidates_openai_style(ct_slice, candidates, output_path, slice_idx):
    """OpenAI-style visualization with bounding boxes"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # 1. Original CT
    axes[0].imshow(ct_slice.T, cmap='gray', origin='lower')
    axes[0].set_title(f'Original CT (Slice {slice_idx})', fontsize=14, fontweight='bold')
    axes[0].axis('off')
    
    # 2. Candidates with bounding boxes
    axes[1].imshow(ct_slice.T, cmap='gray', origin='lower', alpha=0.8)
    
    for i, cand in enumerate(candidates[:25]):  # Top 25 like OpenAI
        bbox = cand.bounding_box
        conf = cand.confidence_score
        
        # Color by confidence
        if conf > 0.7:
            color = 'red'
            label_color = 'red'
        elif conf > 0.5:
            color = 'yellow'
            label_color = 'yellow'
        else:
            color = 'blue'
            label_color = 'blue'
        
        # Draw bounding box (bbox is min_row, min_col, max_row, max_col)
        rect = Rectangle(
            (bbox[1], bbox[0]),  # (x, y) in transposed coords
            bbox[3] - bbox[1],   # width
            bbox[2] - bbox[0],   # height
            edgecolor=color,
            facecolor='none',
            linewidth=2
        )
        axes[1].add_patch(rect)
        
        # Label
        label_text = f"#{i+1}\nConf:{conf:.2f}"
        axes[1].text(
            bbox[1] + 2, bbox[0] + 2,
            label_text,
            color='white',
            fontsize=8,
            bbox=dict(boxstyle='round', facecolor=label_color, alpha=0.7)
        )
    
    axes[1].set_title(f'Detected: {len(candidates)} candidates', fontsize=14, fontweight='bold')
    axes[1].axis('off')
    
    # 3. Detection summary (OpenAI style)
    axes[2].axis('off')
    
    high_conf = [c for c in candidates if c.confidence_score > 0.5]
    
    summary_text = f"""
Candidate Details:
{'='*40}

Total candidates: {len(candidates)}
High confidence (>0.5): {len(high_conf)}
Max confidence: {max([c.confidence_score for c in candidates]) if candidates else 0:.2f}

Top 10:
"""
    
    for i, cand in enumerate(candidates[:10]):
        summary_text += f"""
#{i+1}:
  Area: {cand.area_mm2:.1f} mm²
  Mean HU: {cand.mean_hu:.1f}
  Circularity: {cand.circularity:.2f}
  Confidence: {cand.confidence_score:.2f}
"""
    
    axes[2].text(
        0.1, 0.9,
        summary_text,
        transform=axes[2].transAxes,
        fontsize=10,
        verticalalignment='top',
        fontfamily='monospace',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    )
    
    axes[2].set_title('Detection Summary', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"[OK] Saved visualization: {output_path}")



def main():
    print("="*80)
    print("txt Pipeline Integration Test - Medical Decathlon Data")
    print("OpenAI-style Candidate Detection")
    print("="*80)
    
    # Paths
    data_path = Path("data/medical_decathlon/Task10_Colon/imagesTr")
    gt_path = Path("data/medical_decathlon/Task10_Colon/labelsTr")
    output_dir = Path("results/candidate_detection_test")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Test on colon_001
    test_file = "colon_001.nii.gz"
    ct_file = data_path / test_file
    label_file = gt_path / test_file
    
    if not ct_file.exists():
        print(f"❌ CT file not found: {ct_file}")
        return
    
    # Load data
    print(f"\nLoading: {test_file}")
    volume, spacing = load_nifti(ct_file)
    print(f"   Volume shape: {volume.shape}")
    print(f"   Spacing: {spacing}")
    
    # Load ground truth if available
    has_gt = label_file.exists()
    if has_gt:
        gt_volume, _ = load_nifti(label_file)
        print(f"   Ground truth: {gt_volume.shape}")
        gt_slices_with_tumor = [
            i for i in range(gt_volume.shape[2]) if gt_volume[:, :, i].sum() > 100
        ]
        print(f"   Slices with tumor: {len(gt_slices_with_tumor)}")
    else:
        gt_slices_with_tumor = []
    
    # Initialize detector (OpenAI style parameters)
    detector = TumorDetector(
        min_area_mm2=10.0,      # <- KEY: 10 not 1000!
        max_area_mm2=10000.0,
        hu_range=(-50, 200)
    )
    
    print(f"\nDetector Settings:")
    print(f"   min_area_mm2: {detector.min_area_mm2}")
    print(f"   max_area_mm2: {detector.max_area_mm2}")
    print(f"   hu_range: {detector.hu_range}")
    
    # Test on middle slices
    print(f"\nRunning detection...")

    mid_slice = volume.shape[2] // 2
    test_slices = [mid_slice - 10, mid_slice, mid_slice + 10]
    
    # If GT available, test on tumor slices
    if gt_slices_with_tumor:
        test_slices = gt_slices_with_tumor[:3]
        print(f"   Testing on GT tumor slices: {test_slices}")
    
    all_results = []
    
    for slice_idx in test_slices:
        print(f"\n   Processing slice {slice_idx}...")
        
        # Extract 2D slice
        ct_slice = volume[:, :, slice_idx]
        
        # Detect candidates
        candidates = detector.detect_candidates_2d(
            hu_slice=ct_slice,
            pixel_spacing=(spacing[0], spacing[1]),
            body_mask=None,  # Auto-generate
            slice_index=slice_idx,
            method='multi_threshold'
        )
        
        print(f"      Detected: {len(candidates)} candidates")
        
        if candidates:
            high_conf = [c for c in candidates if c.confidence_score > 0.5]
            print(f"      High confidence (>0.5): {len(high_conf)}")
            
            # Show top 3
            for i, c in enumerate(candidates[:3]):
                print(f"         #{i+1}: Area={c.area_mm2:.1f}mm², "
                      f"HU={c.mean_hu:.1f}, "
                      f"Circ={c.circularity:.2f}, "
                      f"Conf={c.confidence_score:.2f}")
        
        # Visualize
        output_path = output_dir / f"slice_{slice_idx:03d}_detection.png"
        visualize_candidates_openai_style(ct_slice, candidates, output_path, slice_idx)
        
        all_results.append((slice_idx, candidates))
    
    # Summary
    print("\n" + "="*80)
    print("Detection Complete!")
    print("="*80)
    
    total_candidates = sum(len(candidates) for _, candidates in all_results)
    high_conf_total = sum(
        len([c for c in candidates if c.confidence_score > 0.5]) 
        for _, candidates in all_results
    )
    
    print(f"\nTotal slices tested: {len(all_results)}")
    print(f"Total candidates: {total_candidates}")
    print(f"High confidence candidates: {high_conf_total}")
    
    if total_candidates > 0:
        all_confidences = []
        for _, candidates in all_results:
            all_confidences.extend([c.confidence_score for c in candidates])
        
        print(f"Max confidence: {max(all_confidences):.3f}")
        print(f"Average confidence: {np.mean(all_confidences):.3f}")
    
    print(f"\nResults saved to: {output_dir.absolute()}")
    
    # Compare with OpenAI targets
    print("\nComparison with OpenAI Performance:")
    print(f"   OpenAI (Delay Phase): 125 candidates, 79 high-conf (63.2%), max 0.99")
    print(f"   ADDS (txt pipeline): {total_candidates} candidates, {high_conf_total} high-conf, max {max(all_confidences) if all_confidences else 0:.2f}")
    
    if total_candidates > 0:
        print("\nSUCCESS: Candidate detection is working!")
    else:
        print("\nWARNING: No candidates detected. Check parameters.")



if __name__ == "__main__":
    main()
