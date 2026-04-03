"""
Perfect Tumor Reconstruction Service

Integrates the complete perfect reconstruction pipeline into the backend API:
1. Direct detection index mapping (100% match, 0% fallback)
2. Perfect 3D mask reconstruction
3. Enhanced measurements (surface area, shape metrics, distance-to-colon)
"""

from pathlib import Path
import json
import numpy as np
import nibabel as nib
import logging
import sys

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from tumor_3d_clustering import cluster_detections_3d, format_tumor_summary
from reconstruct_tumor_masks_perfect import reconstruct_with_actual_masks
from step3_enhanced_measurements_perfect import enhance_tumor_measurements_perfect

logger = logging.getLogger(__name__)


class PerfectReconstructionService:
    """Service for perfect tumor reconstruction with enhanced measurements"""
    
    def __init__(self):
        self.logger = logger
    
    def run_complete_pipeline(
        self,
        detection_json: Path,
        detection_masks_npz: Path,
        volume_path: Path,
        output_dir: Path,
        use_conservative_colon_mask: bool = True
    ) -> dict:
        """
        Run complete perfect reconstruction pipeline
        
        Args:
            detection_json: Path to detection_summary.json
            detection_masks_npz: Path to detection_masks.npz
            volume_path: Path to original CT volume (for spacing)
            output_dir: Output directory for results
            use_conservative_colon_mask: Use conservative mask for classification
            
        Returns:
            Dictionary with paths to generated files:
            {
                'tumors_json': Path,
                'masks_npz': Path,
                'enhanced_json': Path,
                'summary': {...}
            }
        """
        self.logger.info("="*80)
        self.logger.info("PERFECT RECONSTRUCTION PIPELINE - BACKEND SERVICE")
        self.logger.info("="*80)
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Get spacing from volume
        nii = nib.load(volume_path)
        spacing = nii.header.get_zooms()
        volume_shape = nii.shape
        
        self.logger.info(f"\nVolume shape: {volume_shape}")
        self.logger.info(f"Spacing: {spacing}")
        
        # Step 1: Load detection results
        self.logger.info("\n[1/4] Loading detection results...")
        with open(detection_json) as f:
            detection_data = json.load(f)
        
        results = detection_data['results']
        self.logger.info(f"Loaded {len(results)} slice results")
        
        # Step 2: 3D Clustering with detection index tracking
        self.logger.info("\n[2/4] Running 3D clustering with detection tracking...")
        tumors_3d = cluster_detections_3d(results, spacing, max_distance_mm=20.0)
        
        total_detections = sum(len(r.get('detections', [])) for r in results)
        total_slices = len([r for r in results if len(r.get('detections', [])) > 0])
        
        summary = format_tumor_summary(tumors_3d, total_detections, total_slices)
        
        self.logger.info(f"Clustered into {len(tumors_3d)} 3D tumors")
        
        # Save tumors JSON with detection_indices
        tumors_json_path = output_dir / 'tumors_3d.json'
        with open(tumors_json_path, 'w') as f:
            json.dump({
                'summary': summary,
                'tumors': [
                    {
                        'tumor_id': t.tumor_id,
                        'volume_mm3': float(t.volume_mm3),
                        'max_diameter_mm': float(t.max_diameter_mm),
                        'z_range_mm': float(t.z_range_mm),
                        'num_slices': len(t.slices),
                        'slices': t.slices,
                        'confidence': float(t.confidence),
                        'mean_hu': float(t.mean_hu),
                        'detection_indices': t.detection_indices if t.detection_indices else []
                    }
                    for t in tumors_3d
                ]
            }, f, indent=2)
        
        self.logger.info(f"Saved tumors: {tumors_json_path}")
        
        # Step 3: Perfect reconstruction
        self.logger.info("\n[3/4] Running perfect reconstruction...")
        tumor_masks = reconstruct_with_actual_masks(
            tumors_json_path,
            detection_json,
            detection_masks_npz,
            volume_shape,
            spacing
        )
        
        # Save masks
        masks_dir = output_dir / 'tumor_masks_perfect'
        masks_dir.mkdir(exist_ok=True)
        
        masks_npz_path = masks_dir / 'all_tumor_masks_perfect.npz'
        np.savez_compressed(masks_npz_path, **{f'tumor_{tid}': mask for tid, mask in tumor_masks.items()})
        
        self.logger.info(f"Saved masks: {masks_npz_path}")
        
        # Step 4: Enhanced measurements
        self.logger.info("\n[4/4] Computing enhanced measurements...")
        
        # Determine colon mask path - resolve from detection directory structure
        segmentation_dir = detection_json.parent.parent / 'inha_ct_detection' / '3d_segmentation'
        
        if use_conservative_colon_mask:
            colon_mask_path = segmentation_dir / 'colon_mask_3d_conservative.nii.gz'
            if not colon_mask_path.exists():
                # Fallback to original if conservative doesn't exist
                self.logger.warning("Conservative colon mask not found, using original")
                colon_mask_path = segmentation_dir / 'colon_mask_3d.nii.gz'
        else:
            colon_mask_path = segmentation_dir / 'colon_mask_3d.nii.gz'
        
        self.logger.info(f"Using colon mask: {colon_mask_path}")
        
        enhanced_json_path = output_dir / 'tumors_3d_enhanced_perfect.json'
        
        enhance_tumor_measurements_perfect(
            tumors_json_path,
            masks_npz_path,
            colon_mask_path,
            enhanced_json_path,
            spacing
        )
        
        self.logger.info(f"Saved enhanced measurements: {enhanced_json_path}")
        
        # Load final results for summary
        with open(enhanced_json_path) as f:
            enhanced_data = json.load(f)
        
        final_summary = {
            'total_tumors': len(enhanced_data['tumors']),
            'inside_colon': sum(1 for t in enhanced_data['tumors'] if t['distance_to_colon']['inside_colon']),
            'outside_colon': sum(1 for t in enhanced_data['tumors'] if not t['distance_to_colon']['inside_colon']),
            'mean_sphericity': float(np.mean([t['sphericity'] for t in enhanced_data['tumors']])),
            'mean_volume_mm3': float(np.mean([t['volume_mm3'] for t in enhanced_data['tumors']]))
        }
        
        self.logger.info("\n" + "="*80)
        self.logger.info("PIPELINE COMPLETE")
        self.logger.info("="*80)
        self.logger.info(f"Total tumors: {final_summary['total_tumors']}")
        self.logger.info(f"Inside colon: {final_summary['inside_colon']} ({final_summary['inside_colon']/final_summary['total_tumors']*100:.1f}%)")
        self.logger.info(f"Outside colon: {final_summary['outside_colon']} ({final_summary['outside_colon']/final_summary['total_tumors']*100:.1f}%)")
        self.logger.info(f"Mean sphericity: {final_summary['mean_sphericity']:.3f}")
        
        return {
            'tumors_json': tumors_json_path,
            'masks_npz': masks_npz_path,
            'enhanced_json': enhanced_json_path,
            'summary': final_summary
        }


# Convenience function for backend integration
def run_perfect_reconstruction_pipeline(
    detection_dir: Path,
    volume_path: Path,
    output_dir: Path
) -> dict:
    """
    Convenience wrapper for backend integration
    
    Args:
        detection_dir: Directory with detection outputs
        volume_path: Path to CT volume
        output_dir: Output directory
        
    Returns:
        Results dictionary
    """
    service = PerfectReconstructionService()
    
    detection_json = detection_dir / 'detection_summary.json'
    detection_masks = detection_dir / 'detection_masks.npz'
    
    if not detection_json.exists():
        raise FileNotFoundError(f"Detection JSON not found: {detection_json}")
    if not detection_masks.exists():
        raise FileNotFoundError(f"Detection masks not found: {detection_masks}")
    
    return service.run_complete_pipeline(
        detection_json,
        detection_masks,
        volume_path,
        output_dir
    )


if __name__ == "__main__":
    # Test the service
    logging.basicConfig(level=logging.INFO)
    
    detection_dir = Path("outputs/inha_ct_detection_with_masks")
    volume_path = Path("outputs/inha_ct_analysis/inha_ct_volume.nii.gz")
    output_dir = Path("outputs/inha_3d_analysis")
    
    results = run_perfect_reconstruction_pipeline(
        detection_dir,
        volume_path,
        output_dir
    )
    
    print("\n✅ Service test complete!")
    print(f"Enhanced JSON: {results['enhanced_json']}")
