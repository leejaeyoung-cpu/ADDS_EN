#!/usr/bin/env python3
"""
Batch MedSAM Annotation for Inha CT Data
Semi-automated annotation with interactive review
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from medical_imaging.medsam_detector import MedSAMDetector
import pydicom
import numpy as np
import matplotlib.pyplot as plt
import cv2
import json
from typing import List, Dict

def select_representative_slices(dicom_dir: Path, n_slices: int = 20) -> List[Path]:
    """
    Select representative slices from CT series
    
    Strategy:
    - Evenly spaced through volume
    - Focus on arterial phase (best for tumor detection)
    """
    arterial_series = sorted(dicom_dir.glob("200*.dcm"))  # Series 2: Abdomen Artery
    
    if len(arterial_series) == 0:
        print("No arterial phase found, using all DCM files")
        arterial_series = sorted(dicom_dir.glob("*.dcm"))
    
    print(f"\nTotal arterial phase slices: {len(arterial_series)}")
    
    # Select evenly spaced slices
    step = max(1, len(arterial_series) // n_slices)
    selected = arterial_series[::step][:n_slices]
    
    print(f"Selected {len(selected)} slices (every {step})")
    
    return selected

def get_auto_tumor_point(image: np.ndarray) -> tuple:
    """
    Automatically detect likely tumor location
    
    Simple heuristic:
    - Find brightest regions in center-lower area
    - Typical location for colorectal tumors
    """
    h, w = image.shape[:2]
    
    # Focus on center-lower region (typical tumor location)
    roi_y1 = int(h * 0.3)
    roi_y2 = int(h * 0.8)
    roi_x1 = int(w * 0.3)
    roi_x2 = int(w * 0.7)
    
    roi = image[roi_y1:roi_y2, roi_x1:roi_x2]
    
    # Find brightest local region
    kernel_size = 20
    local_max = cv2.dilate(roi, np.ones((kernel_size, kernel_size)))
    max_val = roi.max()
    
    # Find coordinates
    mask = (roi == max_val)
    if mask.sum() > 0:
        coords = np.argwhere(mask)
        center = coords.mean(axis=0).astype(int)
        
        # Convert back to full image coordinates
        point_y = roi_y1 + center[0]
        point_x = roi_x1 + center[1]
    else:
        # Fallback to center
        point_x = w // 2
        point_y = int(h * 0.6)
    
    return int(point_x), int(point_y)

def batch_annotate(
    detector: MedSAMDetector,
    dicom_files: List[Path],
    output_dir: Path,
    auto_point: bool = True
) -> Dict:
    """
    Batch annotate CT slices with MedSAM
    
    Args:
        detector: MedSAM detector instance
        dicom_files: List of DICOM files to annotate
        output_dir: Output directory for visualizations
        auto_point: Use automatic point detection
    
    Returns:
        Dictionary of annotations
    """
    output_dir.mkdir(exist_ok=True, parents=True)
    annotations = {}
    
    print(f"\n{'='*70}")
    print(f"Batch Annotation: {len(dicom_files)} slices")
    print(f"{'='*70}\n")
    
    for idx, dcm_file in enumerate(dicom_files, 1):
        print(f"[{idx}/{len(dicom_files)}] Processing: {dcm_file.name}")
        
        try:
            # Load CT slice
            image_rgb, image_gray = detector.load_ct_slice(dcm_file)
            
            # Get tumor point (auto or manual)
            if auto_point:
                point_x, point_y = get_auto_tumor_point(image_gray)
                print(f"  Auto point: ({point_x}, {point_y})")
            else:
                # Use center as default
                h, w = image_gray.shape
                point_x, point_y = w // 2, int(h * 0.6)
                print(f"  Default point: ({point_x}, {point_y})")
            
            # Run MedSAM segmentation
            points = np.array([[point_x, point_y]])
            masks, scores, logits = detector.predict_with_points(
                image_rgb,
                points,
                labels=np.array([1])
            )
            
            # Select best mask
            best_idx = np.argmax(scores)
            best_mask = masks[best_idx]
            best_score = scores[best_idx]
            
            print(f"  Score: {best_score:.3f}")
            
            # Create visualization
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            
            # Original + point
            axes[0].imshow(image_gray, cmap='gray')
            axes[0].scatter(point_x, point_y, c='red', s=200, marker='*', 
                          edgecolors='yellow', linewidths=2)
            axes[0].set_title('CT + Point', fontsize=12)
            axes[0].axis('off')
            
            # Segmentation overlay
            axes[1].imshow(image_gray, cmap='gray')
            axes[1].imshow(best_mask, alpha=0.5, cmap='Reds')
            axes[1].scatter(point_x, point_y, c='red', s=100, marker='*')
            axes[1].set_title(f'Segm (Score: {best_score:.3f})', fontsize=12)
            axes[1].axis('off')
            
            # Mask only
            axes[2].imshow(best_mask, cmap='hot')
            axes[2].set_title('Mask', fontsize=12)
            axes[2].axis('off')
            
            plt.suptitle(f'{dcm_file.name} - MedSAM Annotation', 
                        fontsize=14, fontweight='bold')
            plt.tight_layout()
            
            # Save visualization
            viz_file = output_dir / f"annotation_{dcm_file.stem}.png"
            plt.savefig(viz_file, dpi=100, bbox_inches='tight')
            plt.close()
            
            print(f"  Saved: {viz_file.name}")
            
            # Store annotation
            annotations[dcm_file.name] = {
                'file': dcm_file.name,
                'point': [point_x, point_y],
                'mask_shape': best_mask.shape,
                'mask_size': int(best_mask.sum()),
                'score': float(best_score),
                'auto_point': auto_point,
                'visualization': str(viz_file)
            }
            
        except Exception as e:
            print(f"  Error: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n{'='*70}")
    print(f"Annotation complete: {len(annotations)} successful")
    print(f"{'='*70}\n")
    
    return annotations

def main():
    """Main batch annotation workflow"""
    
    print("\n" + "="*70)
    print("MedSAM Batch Annotation Tool")
    print("="*70)
    
    # Initialize MedSAM
    print("\nInitializing MedSAM...")
    detector = MedSAMDetector(
        model_type="vit_h",
        checkpoint_path="models/sam_vit_h_4b8939.pth"
    )
    
    # Select slices
    dicom_dir = Path("CTdata/CTdcm")
    n_slices = 20
    
    print(f"\nSelecting {n_slices} representative slices...")
    selected_files = select_representative_slices(dicom_dir, n_slices)
    
    # Batch annotate
    output_dir = Path("annotations/batch_medsam")
    annotations = batch_annotate(
        detector,
        selected_files,
        output_dir,
        auto_point=True
    )
    
    # Save annotations
    annotation_file = Path("annotations/medsam_batch_annotations.json")
    annotation_file.parent.mkdir(exist_ok=True)
    
    with open(annotation_file, 'w') as f:
        json.dump(annotations, f, indent=2)
    
    print(f"\n✅ Annotations saved: {annotation_file}")
    
    # Summary
    print(f"\n{'='*70}")
    print("Summary")
    print(f"{'='*70}")
    print(f"Total slices annotated: {len(annotations)}")
    
    if annotations:
        scores = [ann['score'] for ann in annotations.values()]
        print(f"Average confidence: {np.mean(scores):.3f}")
        print(f"Min confidence: {np.min(scores):.3f}")
        print(f"Max confidence: {np.max(scores):.3f}")
        
        sizes = [ann['mask_size'] for ann in annotations.values()]
        print(f"\nMask sizes:")
        print(f"  Average: {np.mean(sizes):.0f} pixels")
        print(f"  Range: {np.min(sizes):.0f} - {np.max(sizes):.0f}")
    
    print(f"\nVisualizations: {output_dir}/")
    print(f"Annotations JSON: {annotation_file}")
    
    print("\nNext steps:")
    print("1. Review visualizations")
    print("2. Remove any incorrect annotations")
    print("3. Run fine-tuning (optional)")
    print("4. Full inference on all 426 slices")

if __name__ == '__main__':
    main()
