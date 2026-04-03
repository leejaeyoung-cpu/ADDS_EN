"""
Batch Tumor Detection for DICOM Files
======================================
Process 100 random DICOM slices from CTdcm folder
Detect tumors and create masked visualization images
"""
import sys
import random
from pathlib import Path
import numpy as np
import pydicom
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from tqdm import tqdm
import json
from datetime import datetime

# Add project root
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.medical_imaging.inference.hybrid_predictor import HybridPredictor


def load_dicom_slice(filepath):
    """Load a single DICOM slice"""
    try:
        dcm = pydicom.dcmread(filepath)
        
        # Get pixel array
        pixel_array = dcm.pixel_array.astype(np.float32)
        
        # Apply DICOM rescale
        if hasattr(dcm, 'RescaleSlope') and hasattr(dcm, 'RescaleIntercept'):
            pixel_array = pixel_array * dcm.RescaleSlope + dcm.RescaleIntercept
        
        # Get spacing if available
        try:
            spacing = (float(dcm.PixelSpacing[0]), float(dcm.PixelSpacing[1]), 
                      float(dcm.SliceThickness) if hasattr(dcm, 'SliceThickness') else 1.0)
        except:
            spacing = (1.0, 1.0, 1.0)
        
        return pixel_array, spacing, True
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return None, None, False


def create_2d_predictor():
    """Create predictor for single slice analysis"""
    from src.medical_imaging.detection.candidate_detector import TumorDetector
    detector = TumorDetector()
    return detector


def detect_and_visualize(
    slice_data, 
    slice_name,
    output_dir,
    detector
):
    """
    Detect tumors in a single slice and create visualization
    
    Returns:
        dict: Detection results with tumor found status
    """
    try:
        # Run detection on the 2D slice
        candidates = detector.detect_candidates_2d(
            hu_slice=slice_data,
            pixel_spacing=(1.0, 1.0),  # Default spacing for DICOM
            slice_index=0,
            method='multi_threshold'
        )
        
        # Filter high-confidence candidates
        high_conf_candidates = [c for c in candidates if c.confidence_score > 0.7]
        
        has_tumor = len(high_conf_candidates) > 0
        
        # Create visualization
        fig = plt.figure(figsize=(15, 5))
        gs = GridSpec(1, 3, figure=fig, hspace=0.3, wspace=0.3)
        
        # Normalize for visualization
        ct_display = np.clip(slice_data, -160, 240)
        ct_display = (ct_display + 160) / 400
        
        # 1. Original slice
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.imshow(ct_display.T, cmap='gray', origin='lower')
        ax1.set_title(f'Original CT\n{slice_name}', fontsize=12, fontweight='bold')
        ax1.axis('off')
        
        # 2. All candidates
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.imshow(ct_display.T, cmap='gray', origin='lower')
        ax2.set_title(f'All Candidates (n={len(candidates)})', fontsize=12, fontweight='bold')
        
        for candidate in candidates[:50]:  # Limit display
            x, y = candidate.centroid
            radius = max(5, min(30, np.sqrt(candidate.area_pixels / np.pi)))
            conf = candidate.confidence_score
            
            color = 'yellow' if conf > 0.7 else 'orange'
            alpha = 0.3 + 0.4 * conf
            
            circle = plt.Circle((x, y), radius, color=color, fill=False, 
                              linewidth=2, alpha=alpha)
            ax2.add_patch(circle)
        ax2.axis('off')
        
        # 3. High-confidence with mask
        ax3 = fig.add_subplot(gs[0, 2])
        ax3.imshow(ct_display.T, cmap='gray', origin='lower')
        
        if has_tumor:
            # Create red mask overlay
            mask = np.zeros_like(slice_data, dtype=bool)
            for candidate in high_conf_candidates:
                x, y = candidate.centroid
                radius = max(5, min(30, np.sqrt(candidate.area_pixels / np.pi)))
                
                # Create circular mask
                yy, xx = np.ogrid[:slice_data.shape[1], :slice_data.shape[0]]
                circle_mask = (xx - x)**2 + (yy - y)**2 <= radius**2
                mask[circle_mask.T] = True
                
                # Draw circle
                circle = plt.Circle((x, y), radius, color='red', fill=False, 
                                  linewidth=3, alpha=0.9)
                ax3.add_patch(circle)
                
                # Add confidence label
                ax3.text(x, y - radius - 5, f'{candidate.confidence_score:.1%}', 
                        color='red', fontsize=9, fontweight='bold',
                        ha='center', va='bottom',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9))
            
            # Apply red overlay
            red_overlay = np.zeros((*slice_data.shape, 4))
            red_overlay[mask.T] = [1, 0, 0, 0.3]  # Red with alpha
            ax3.imshow(red_overlay, origin='lower')
            
            ax3.set_title(f'✓ TUMOR DETECTED (n={len(high_conf_candidates)})', 
                         fontsize=12, fontweight='bold', color='red')
        else:
            ax3.set_title('✓ NO TUMOR FOUND', fontsize=12, fontweight='bold', color='green')
        
        ax3.axis('off')
        
        # Add overall title
        status = "TUMOR DETECTED" if has_tumor else "NO TUMOR"
        status_color = "red" if has_tumor else "green"
        plt.suptitle(f'{slice_name} - {status}', 
                    fontsize=14, fontweight='bold', color=status_color)
        
        # Save
        output_path = output_dir / f"{slice_name}_detection.png"
        plt.savefig(output_path, dpi=100, bbox_inches='tight')
        plt.close()
        
        return {
            'slice_name': slice_name,
            'has_tumor': has_tumor,
            'total_candidates': len(candidates),
            'high_conf_candidates': len(high_conf_candidates),
            'max_confidence': max([c.confidence_score for c in candidates]) if candidates else 0,
            'output_image': str(output_path)
        }
        
    except Exception as e:
        print(f"Error processing {slice_name}: {e}")
        return {
            'slice_name': slice_name,
            'has_tumor': False,
            'error': str(e)
        }


def main():
    """Process 100 random DICOM files"""
    
    # Setup
    ctdcm_dir = Path("CTdcm")
    output_dir = Path("outputs/dcm_tumor_detection")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print("BATCH TUMOR DETECTION - DICOM FILES")
    print("=" * 80)
    
    # Get all DICOM files
    dcm_files = list(ctdcm_dir.glob("*.dcm"))
    print(f"\n[*] Found {len(dcm_files)} DICOM files in {ctdcm_dir}")
    
    # Select 100 random files
    num_files = min(100, len(dcm_files))
    selected_files = random.sample(dcm_files, num_files)
    print(f"[*] Randomly selected {num_files} files for processing")
    
    # Initialize detector
    print(f"\n[*] Initializing tumor detector...")
    detector = create_2d_predictor()
    
    # Process each file
    print(f"\n[*] Processing files...")
    results = []
    tumor_count = 0
    error_count = 0
    
    for dcm_file in tqdm(selected_files, desc="Detecting tumors"):
        # Load DICOM
        slice_data, spacing, success = load_dicom_slice(dcm_file)
        
        if not success:
            error_count += 1
            continue
        
        # Detect and visualize
        result = detect_and_visualize(
            slice_data=slice_data,
            slice_name=dcm_file.stem,
            output_dir=output_dir,
            detector=detector
        )
        
        results.append(result)
        
        if result.get('has_tumor', False):
            tumor_count += 1
    
    # Summary statistics
    print("\n" + "=" * 80)
    print("DETECTION SUMMARY")
    print("=" * 80)
    print(f"Total processed: {len(results)}")
    print(f"Tumors detected: {tumor_count} ({tumor_count/len(results)*100:.1f}%)")
    print(f"No tumors: {len(results) - tumor_count} ({(len(results)-tumor_count)/len(results)*100:.1f}%)")
    print(f"Errors: {error_count}")
    
    # Save statistics JSON
    summary = {
        'timestamp': datetime.now().isoformat(),
        'total_files': len(dcm_files),
        'processed': len(results),
        'tumor_detected': tumor_count,
        'no_tumor': len(results) - tumor_count,
        'errors': error_count,
        'detection_rate': f"{tumor_count/len(results)*100:.2f}%",
        'results': results
    }
    
    summary_path = output_dir / "detection_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n[+] Summary saved to: {summary_path}")
    print(f"[+] Visualizations saved to: {output_dir}")
    
    # List tumor-positive cases
    if tumor_count > 0:
        print(f"\n[*] Tumor-positive cases ({tumor_count}):")
        for result in results:
            if result.get('has_tumor', False):
                print(f"  - {result['slice_name']}: "
                      f"{result['high_conf_candidates']} candidates, "
                      f"max conf={result['max_confidence']:.1%}")
    
    print(f"\n[+] Done! Check results in: {output_dir}")


if __name__ == "__main__":
    main()
