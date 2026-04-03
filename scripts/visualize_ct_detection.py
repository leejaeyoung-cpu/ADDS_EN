#!/usr/bin/env python3
"""
인하대병원 CT Detection 시각화
검출된 종양 후보 영역을 이미지로 표시
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from medical_imaging.ct_analyzer import CTAnalyzer
import pydicom
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib import cm
import json

def visualize_detection_results():
    """검출 결과 시각화"""
    
    print("\n" + "="*70)
    print("CT Detection 시각화")
    print("="*70)
    
    # Load candidates from previous detection
    with open('inha_ct_candidates_v2.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    candidates = data['candidates']
    
    print(f"\nTotal candidates: {len(candidates)}")
    
    # CT Analyzer
    analyzer = CTAnalyzer(use_gpu=True, use_nnunet=False, enable_ai_research=False)
    
    # DICOM directory
    dicom_dir = Path("CTdata/CTdcm")
    
    # Output directory
    output_dir = Path("CTdata/visualizations")
    output_dir.mkdir(exist_ok=True)
    
    # Visualize each candidate
    for i, cand in enumerate(candidates, 1):
        slice_num = cand['slice']
        dcm_file = dicom_dir / f"{slice_num}.dcm"
        
        if not dcm_file.exists():
            continue
        
        print(f"\n[{i}/{len(candidates)}] Visualizing {dcm_file.name}...")
        
        try:
            # Re-analyze to get mask
            result = analyzer.analyze_ct_image(
                image_path=str(dcm_file),
                cancer_type="Colorectal"
            )
            
            if result['status'] != 'success':
                print(f"  Error: {result.get('error')}")
                continue
            
            # Load DICOM image
            dcm = pydicom.dcmread(dcm_file)
            image_array = dcm.pixel_array.astype(float)
            
            # Apply HU conversion
            if hasattr(dcm, 'RescaleSlope') and hasattr(dcm, 'RescaleIntercept'):
                image_array = image_array * dcm.RescaleSlope + dcm.RescaleIntercept
            
            # Get segmentation mask
            mask = result['segmentation'].get('segmentation_mask')
            
            if mask is None or mask.sum() == 0:
                print(f"  No mask available")
                continue
            
            # Create visualization
            fig, axes = plt.subplots(1, 3, figsize=(18, 6))
            
            # 1. Original CT
            ax = axes[0]
            ax.imshow(image_array, cmap='gray', vmin=-200, vmax=200)
            ax.set_title(f'Original CT\nSlice {slice_num}', fontsize=12, fontweight='bold')
            ax.axis('off')
            
            # 2. Detection Overlay
            ax = axes[1]
            ax.imshow(image_array, cmap='gray', vmin=-200, vmax=200)
            # Overlay mask in red with transparency
            masked = np.ma.masked_where(mask == 0, mask)
            ax.imshow(masked, cmap='Reds', alpha=0.5, vmin=0, vmax=1)
            
            # Draw bounding box
            bbox = cand['bbox']
            rect = mpatches.Rectangle(
                (bbox['x'], bbox['y']),
                bbox['width'], bbox['height'],
                linewidth=2, edgecolor='yellow', facecolor='none'
            )
            ax.add_patch(rect)
            
            ax.set_title(f'Detection Overlay\nProbability: {cand["probability"]:.1%}', 
                        fontsize=12, fontweight='bold')
            ax.axis('off')
            
            # 3. Mask Only
            ax = axes[2]
            ax.imshow(mask, cmap='hot')
            ax.set_title(f'Segmentation Mask\nSize: {cand["volume_voxels"]:,} voxels', 
                        fontsize=12, fontweight='bold')
            ax.axis('off')
            
            # Add info text
            info_text = f"""
Detection Info:
• Probability: {cand['probability']:.1%} ({cand['confidence']})
• Size: {cand['volume_voxels']:,} voxels
• Diameter: {cand['diameter_mm']:.1f} mm
• Intensity Mean: {cand['features']['intensity_mean']:.1f}
• Intensity Std: {cand['features']['intensity_std']:.1f}
• Texture Entropy: {cand['features']['texture_entropy']:.2f}
            """.strip()
            
            fig.text(0.5, 0.02, info_text, ha='center', fontsize=10, 
                    family='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
            
            plt.tight_layout(rect=[0, 0.08, 1, 1])
            
            # Save
            output_file = output_dir / f"detection_{slice_num}.png"
            plt.savefig(output_file, dpi=150, bbox_inches='tight')
            plt.close()
            
            print(f"  ✓ Saved: {output_file.name}")
        
        except Exception as e:
            print(f"  Error: {e}")
            import traceback
            traceback.print_exc()
    
    # Create summary figure
    print(f"\n{'─'*70}")
    print("Creating summary figure...")
    
    try:
        n_candidates = min(6, len(candidates))
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        for idx, cand in enumerate(candidates[:n_candidates]):
            slice_num = cand['slice']
            dcm_file = dicom_dir / f"{slice_num}.dcm"
            
            # Load image
            dcm = pydicom.dcmread(dcm_file)
            image_array = dcm.pixel_array.astype(float)
            if hasattr(dcm, 'RescaleSlope') and hasattr(dcm, 'RescaleIntercept'):
                image_array = image_array * dcm.RescaleSlope + dcm.RescaleIntercept
            
            # Get mask
            result = analyzer.analyze_ct_image(str(dcm_file), "Colorectal")
            mask = result['segmentation'].get('segmentation_mask')
            
            # Plot
            ax = axes[idx]
            ax.imshow(image_array, cmap='gray', vmin=-200, vmax=200)
            if mask is not None and mask.sum() > 0:
                masked = np.ma.masked_where(mask == 0, mask)
                ax.imshow(masked, cmap='Reds', alpha=0.5)
            
            ax.set_title(f'Slice {slice_num}\nProb: {cand["probability"]:.1%}', 
                        fontsize=10)
            ax.axis('off')
        
        plt.suptitle('인하대병원 CT Detection Results - All Candidates', 
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        summary_file = output_dir / "detection_summary.png"
        plt.savefig(summary_file, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Summary saved: {summary_file.name}")
    
    except Exception as e:
        print(f"Error creating summary: {e}")
    
    print(f"\n{'='*70}")
    print(f"Visualization complete!")
    print(f"Output directory: {output_dir.absolute()}")
    print(f"Total images: {len(list(output_dir.glob('*.png')))}")
    print("="*70 + "\n")
    
    print("Files created:")
    for img_file in sorted(output_dir.glob('*.png')):
        print(f"  - {img_file.name}")

if __name__ == '__main__':
    visualize_detection_results()
