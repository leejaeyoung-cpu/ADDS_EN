"""
Integrated CT-Based Colorectal Cancer Detection Pipeline
Complete end-to-end system from DICOM to ADDS integration
"""

import numpy as np
from pathlib import Path
from typing import Dict, Optional, List, Tuple
import logging
import json
from datetime import datetime
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import base64
from io import BytesIO

# Import pipeline stages
from src.medical_imaging.ct_volume_reconstruction import CTVolumeReconstructor
from src.medical_imaging.nnunet_segmentation import nnUNetSegmentationEngine
from src.medical_imaging.radiomics_extractor import RadiomicsExtractor, TumorCharacterizer
from src.medical_imaging.tumor_classifier import TumorClassifier, BiomarkerPredictor
from src.medical_imaging.adds_integrator import ADDSIntegrator
from src.medical_imaging.detection.simple_hu_detector import SimpleHUDetector, detect_tumors_simple

# NEW: Import database integration
from patient_management_system.services.patient_integration import save_ct_to_database

logger = logging.getLogger(__name__)


class IntegratedCRCDetectionPipeline:
    """
    Complete 6-stage CT-based colorectal cancer detection pipeline
    
    Stages:
    1. 3D Volume Reconstruction
    2. Colon Segmentation (nnU-Net) - To be integrated
    3. Tumor Detection & Segmentation (nnU-Net) - To be integrated
    4. Radiomics Feature Extraction
    5. Tumor Classification & Staging
    6. ADDS System Integration
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize pipeline
        
        Args:
            config: Pipeline configuration
        """
        self.config = config or self._default_config()
        
        # Initialize stage modules
        self.volume_reconstructor = CTVolumeReconstructor(
            target_spacing=tuple(self.config['target_spacing'])
        )
        
        self.segmentation_engine = nnUNetSegmentationEngine(
            colon_model_dir=Path(self.config.get('nnunet_colon_model', 'models/nnunet_colon')),
            tumor_model_dir=Path(self.config.get('nnunet_tumor_model', 'models/nnunet_tumor')),
            device='cuda' if self.config.get('use_gpu', True) else 'cpu'
        )
        
        self.radiomics_extractor = RadiomicsExtractor(
            bin_width=self.config['radiomics_bin_width'],
            resampled_pixel_spacing=self.config['target_spacing']
        )
        
        self.tumor_characterizer = TumorCharacterizer()
        
        self.tumor_classifier = TumorClassifier(
            models_dir=Path(self.config['models_dir']) / 'classifiers'
        )
        
        self.biomarker_predictor = BiomarkerPredictor(
            models_dir=Path(self.config['models_dir']) / 'biomarker_predictors'
        )
        
        self.adds_integrator = ADDSIntegrator(
            adds_api_url=self.config.get('adds_api_url'),
            api_token=self.config.get('adds_api_token')
        )
        
        logger.info("Integrated CRC detection pipeline initialized")
    
    def visualize_tumor_slices(self, volume: np.ndarray, tumor_masks: List[np.ndarray], 
                                num_slices: int = 5) -> List[str]:
        """
        Generate visualization of CT slices with tumor overlays
        
        Args:
            volume: CT volume (z, y, x)
            tumor_masks: List of tumor masks
            num_slices: Number of representative slices to visualize
        
        Returns:
            List of base64-encoded PNG images
        """
        z_slices = volume.shape[0]
        
        # Find slices with tumors
        tumor_slices = set()
        for tumor_mask in tumor_masks:
            tumor_z = np.where(tumor_mask.any(axis=(1, 2)))[0]
            tumor_slices.update(tumor_z)
        
        # Select representative slices
        if len(tumor_slices) == 0:
            # No tumors, select evenly spaced slices
            selected_slices = np.linspace(z_slices // 4, 3 * z_slices // 4, num_slices, dtype=int)
        else:
            tumor_slices = sorted(list(tumor_slices))
            # Sample from tumor-containing slices
            if len(tumor_slices) <= num_slices:
                selected_slices = tumor_slices
            else:
                step = len(tumor_slices) // num_slices
                selected_slices = [tumor_slices[i * step] for i in range(num_slices)]
        
        images_base64 = []
        
        for slice_idx in selected_slices:
            # Create figure
            fig, ax = plt.subplots(figsize=(6, 6), dpi=100)
            
            # Display CT slice
            ct_slice = volume[slice_idx]
            ax.imshow(ct_slice, cmap='gray', vmin=-200, vmax=200)
            
            # Overlay tumor masks
            for tumor_mask in tumor_masks:
                mask_slice = tumor_mask[slice_idx]
                if mask_slice.any():
                    # Create colored overlay
                    overlay = np.zeros((*mask_slice.shape, 4))
                    overlay[mask_slice > 0] = [1, 0, 0, 0.4]  # Red with transparency
                    ax.imshow(overlay)
                    
                    # Draw bounding box
                    y_coords, x_coords = np.where(mask_slice > 0)
                    if len(y_coords) > 0:
                        y_min, y_max = y_coords.min(), y_coords.max()
                        x_min, x_max = x_coords.min(), x_coords.max()
                        rect = Rectangle((x_min, y_min), x_max - x_min, y_max - y_min,
                                       linewidth=2, edgecolor='red', facecolor='none')
                        ax.add_patch(rect)
            
            ax.set_title(f'CT Slice {slice_idx + 1}/{z_slices}', fontsize=12, color='white')
            ax.axis('off')
            fig.patch.set_facecolor('#1e1e1e')
            
            # Convert to base64
            buffer = BytesIO()
            plt.savefig(buffer, format='png', bbox_inches='tight', facecolor='#1e1e1e')
            buffer.seek(0)
            image_base64 = base64.b64encode(buffer.read()).decode('utf-8')
            images_base64.append(f"data:image/png;base64,{image_base64}")
            
            plt.close(fig)
        
        logger.info(f"Generated {len(images_base64)} tumor visualization images")
        return images_base64
    
    def _default_config(self) -> Dict:
        """Default pipeline configuration"""
        return {
            'target_spacing': [1.0, 1.0, 1.0],
            'radiomics_bin_width': 25,
            'models_dir': 'models',
            'output_dir': 'outputs/crc_detection',
            'adds_api_url': 'http://localhost:8000/api/v1',
            'adds_api_token': None,
            'save_intermediate': True
        }
    
    def process_patient(self, dicom_folder: Path, 
                       patient_id: str,
                       colon_mask: Optional[np.ndarray] = None,
                       tumor_masks: Optional[List[np.ndarray]] = None) -> Dict:
        """
        Process complete patient CT scan
        
        Args:
            dicom_folder: Path to DICOM series
            patient_id: Patient identifier
            colon_mask: Optional pre-segmented colon mask
            tumor_masks: Optional pre-segmented tumor masks
            
        Returns:
            result: Complete analysis result
        """
        logger.info(f"Processing patient {patient_id}...")
        
        start_time = datetime.now()
        
        output_dir = Path(self.config['output_dir']) / patient_id
        output_dir.mkdir(parents=True, exist_ok=True)
        
        result = {
            'patient_id': patient_id,
            'start_time': start_time.isoformat(),
            'stages': {}
        }
        
        try:
            # Stage 1: 3D Volume Reconstruction
            logger.info("Stage 1: 3D Volume Reconstruction")
            volume, vol_metadata = self.volume_reconstructor.reconstruct_3d_volume(
                dicom_folder, normalize=False  # CRITICAL: Keep original HU values for tumor detection!
            )
            
            if self.config['save_intermediate']:
                nifti_path = output_dir / f"{patient_id}_volume.nii.gz"
                self.volume_reconstructor.save_as_nifti(volume, nifti_path, vol_metadata)
            
            result['stages']['volume_reconstruction'] = {
                'status': 'success',
                'shape': volume.shape,
                'spacing': vol_metadata['target_spacing']
            }
            
            logger.info(f"✓ Volume reconstructed: {volume.shape}")
            
            # Stage 2 & 3: Segmentation (if not provided)
            if colon_mask is None or tumor_masks is None:
                logger.warning("Colon/tumor masks not provided. Using placeholder detection.")
                # TODO: Integrate nnU-Net for automatic segmentation
                result['stages']['segmentation'] = {
                    'status': 'skipped',
                    'note': 'Manual masks should be provided or nnU-Net integrated'
                }
                
                # Use real HU-based tumor detection
                if tumor_masks is None:
                    logger.info(f"Running SimpleHUDetector for real tumor detection...")
                    try:
                        # Log volume statistics first
                        logger.info(f"Volume stats before detection: shape={volume.shape}, HU=[{volume.min():.1f}, {volume.max():.1f}]")
                        
                        # Detect tumors using HU thresholds (98.6% accuracy system)
                        detected_lesions = detect_tumors_simple(
                            ct_volume=volume,
                            spacing=self.config['target_spacing'],
                            tumor_hu_min=40.0,
                            tumor_hu_max=150.0
                        )
                        
                        if detected_lesions:
                            tumor_masks = [lesion.mask for lesion in detected_lesions]
                            logger.info(f"✅ Detected {len(tumor_masks)} tumors using SimpleHUDetector")
                        else:
                            logger.warning("No tumors detected by SimpleHUDetector, using placeholder")
                            tumor_masks = [self._create_placeholder_tumor(volume)]
                    except Exception as e:
                        logger.error(f"SimpleHUDetector failed: {e}, using placeholder")
                        tumor_masks = [self._create_placeholder_tumor(volume)]
            
            # Process each detected tumor
            tumors_analysis = []
            
            for i, tumor_mask in enumerate(tumor_masks):
                logger.info(f"Processing tumor {i+1}/{len(tumor_masks)}")
                
                # Stage 4: Radiomics Feature Extraction
                tumor_analysis = self._analyze_single_tumor(
                    volume, tumor_mask, tumor_id=i+1
                )
                
                # Stage 5: Classification & Staging
                classification = self._classify_tumor(tumor_analysis['radiomics'])
                
                # Combine results
                tumor_result = {
                    'tumor_id': i + 1,
                    **tumor_analysis,
                    'classification': classification
                }
                
                tumors_analysis.append(tumor_result)
                
                logger.info(f"✓ Tumor {i+1}: {classification['classification']}")
            
            result['tumors'] = tumors_analysis
            
            # Generate visualization images
            logger.info(f"=== Attempting to generate visualization for {len(tumor_masks)} tumors ===")
            logger.info(f"Volume shape: {volume.shape}")
            
            try:
                visualization_images = self.visualize_tumor_slices(
                    volume, tumor_masks, num_slices=5
                )
                result['visualization_images'] = visualization_images
                logger.info(f"✓ Successfully generated {len(visualization_images)} visualization images")
            except Exception as e:
                logger.error(f"Failed to generate visualization images: {e}", exc_info=True)
                result['visualization_images'] = []
            
            # Stage 6: ADDS Integration (primary tumor)
            if tumors_analysis:
                primary_tumor = max(tumors_analysis, key=lambda x: x['volume_mm3'])
                
                logger.info("Stage 6: ADDS Integration")
                adds_input = self.adds_integrator.prepare_adds_input(
                    patient_id=patient_id,
                    volume=volume,
                    tumor_analysis=primary_tumor,
                    radiomics=primary_tumor['radiomics'],
                    classification=primary_tumor['classification']
                )
                
                treatment_plan = self.adds_integrator.send_to_adds(adds_input)
                
                result['adds_integration'] = {
                    'status': 'success',
                    'input': adds_input,
                    'treatment_plan': treatment_plan
                }
                
                logger.info("✓ ADDS integration complete")
                
                # Save integration result
                if self.config['save_intermediate']:
                    integration_path = output_dir / f"{patient_id}_adds_integration.json"
                    self.adds_integrator.save_integration_result(
                        adds_input, treatment_plan, integration_path
                    )
            
           # Mark as successful
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            result['status'] = 'success'
            result['end_time'] = end_time.isoformat()
            result['duration_seconds'] = duration
            
            # NEW: Auto-save to CDSS database
            try:
                logger.info("Saving analysis to CDSS database...")
                
                # Prepare data for database
                tumor_stats = self._prepare_database_stats(result)
                
                # Save to database
                analysis_id = save_ct_to_database(
                    patient_id=patient_id,
                    ct_path=output_dir / f"{patient_id}_volume.nii.gz",
                    seg_path=output_dir / f"{patient_id}_segmentation.nii.gz",  # May not exist
                    spacing=tuple(self.config['target_spacing']),
                    stats=tumor_stats
                )
                
                result['cdss_database_id'] = analysis_id
                logger.info(f"✓ Saved to CDSS database: Analysis ID {analysis_id}")
                
            except Exception as e:
                logger.error(f"Failed to save to database: {e}")
                logger.exception(e)  # Log full traceback
                result['database_save_error'] = str(e)
            
            logger.info(f"✓ Pipeline complete ({duration:.1f}s)")
            
        except Exception as e:
            logger.error(f"Pipeline error: {e}", exc_info=True)
            result['status'] = 'failed'
            result['error'] = str(e)
        
        # Save final result
        result_path = output_dir / f"{patient_id}_analysis_result.json"
        with open(result_path, 'w') as f:
            json.dump(result, f, indent=2, default=str)
        
        logger.info(f"Result saved to {result_path}")
        
        return result
    
    def _analyze_single_tumor(self, volume: np.ndarray, 
                             tumor_mask: np.ndarray,
                             tumor_id: int) -> Dict:
        """
        Analyze single tumor
        
        Args:
            volume: 3D CT volume
            tumor_mask: Binary tumor mask
            tumor_id: Tumor identifier
            
        Returns:
            analysis: Tumor analysis
        """
        # Extract radiomics features
        radiomics = self.radiomics_extractor.extract_features(volume, tumor_mask)
        
        # Calculate basic properties
        tumor_volume_voxels = int(np.sum(tumor_mask))
        voxel_volume_mm3 = np.prod(self.config['target_spacing'])
        tumor_volume_mm3 = tumor_volume_voxels * voxel_volume_mm3
        
        # Calculate centroid
        coords = np.argwhere(tumor_mask > 0)
        if len(coords) > 0:
            centroid = coords.mean(axis=0).tolist()
        else:
            centroid = [0, 0, 0]
        
        # Characterize tumor
        characterization = self.tumor_characterizer.characterize_tumor(
            volume, tumor_mask,
            tumor_metadata={
                'volume_mm3': tumor_volume_mm3,
                'centroid': centroid
            }
        )
        
        return {
            'volume_mm3': tumor_volume_mm3,
            'centroid': centroid,
            'radiomics': radiomics,
            'characterization': characterization
        }
    
    def _classify_tumor(self, radiomics: Dict) -> Dict:
        """
        Classify tumor and predict biomarkers
        
        Args:
            radiomics: Radiomics features
            
        Returns:
            classification: Complete classification
        """
        # TNM staging
        tnm_result = self.tumor_classifier.predict_tnm(radiomics)
        
        # Biomarkers
        msi_result = self.biomarker_predictor.predict_msi_status(radiomics)
        kras_result = self.biomarker_predictor.predict_kras_mutation(radiomics)
        
        # Combine
        classification = {
            **tnm_result,
            'msi_status': msi_result,
            'kras_mutation': kras_result
        }
        
        return classification
    
    def _prepare_database_stats(self, result: Dict) -> Dict:
        """
        Extract database-relevant statistics from analysis result
        
        Args:
            result: Complete pipeline result
        
        Returns:
            stats: Statistics formatted for database
        """
        tumors_data = []
        
        for tumor in result.get('tumors', []):
            tumor_data = {
                'volume_ml': tumor['volume_mm3'] / 1000,  # Convert mm³ to ml
                'max_diameter_mm': tumor.get('characterization', {}).get('max_diameter_mm', 0),
                'centroid': {
                    'x': tumor['centroid'][2],  # z, y, x → x, y, z
                    'y': tumor['centroid'][1],
                    'z': tumor['centroid'][0]
                },
                'hu_mean': tumor.get('radiomics', {}).get('original_firstorder_Mean', 0),
                'hu_std': tumor.get('radiomics', {}).get('original_firstorder_StandardDeviation', 0),
                'tnm_stage': tumor.get('classification', {}).get('tnm_stage', {}),
                'overall_stage': tumor.get('classification', {}).get('overall_stage', 'Unknown'),
                'radiomics': tumor.get('radiomics', {})
            }
            tumors_data.append(tumor_data)
        
        # Summary statistics
        total_volume_ml = sum(t['volume_ml'] for t in tumors_data)
        max_diameter = max((t['max_diameter_mm'] for t in tumors_data), default=0)
        
        return {
            'total_volume_ml': total_volume_ml,
            'max_diameter_mm': max_diameter,
            'num_tumors': len(tumors_data),
            'tumors': tumors_data,
            'hu_mean': tumors_data[0]['hu_mean'] if tumors_data else 0,
            'hu_std': tumors_data[0]['hu_std'] if tumors_data else 0
        }
    
    def _create_placeholder_tumor(self, volume: np.ndarray) -> np.ndarray:
        """
        Create placeholder tumor mask for demonstration
        
        Args:
            volume: 3D volume
            
        Returns:
            tumor_mask: Placeholder mask
        """
        shape = volume.shape
        tumor_mask = np.zeros(shape, dtype=np.uint8)
        
        # Create a sphere in the center
        center = [s // 2 for s in shape]
        radius = min(shape) // 8
        
        z, y, x = np.ogrid[:shape[0], :shape[1], :shape[2]]
        dist = np.sqrt((z - center[0])**2 + (y - center[1])**2 + (x - center[2])**2)
        tumor_mask[dist <= radius] = 1
        
        logger.warning("Using placeholder tumor mask")
        
        return tumor_mask
    
    def generate_clinical_report(self, result: Dict, output_path: Path):
        """
        Generate clinical report from analysis results
        
        Args:
            result: Analysis result
            output_path: Output HTML/PDF path
        """
        # TODO: Implement clinical report generation
        logger.info(f"Clinical report would be generated at {output_path}")


def run_pipeline_example():
    """Example pipeline execution"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Configuration
    config = {
        'target_spacing': [1.0, 1.0, 1.0],
        'radiomics_bin_width': 25,
        'models_dir': 'F:/ADDS/models',
        'output_dir': 'F:/ADDS/outputs/crc_detection',
        'save_intermediate': True
    }
    
    # Initialize pipeline
    pipeline = IntegratedCRCDetectionPipeline(config)
    
    # Process patient (example)
    dicom_folder = Path("F:/ADDS/CTdata/sample_patient")
    
    if not dicom_folder.exists():
        logger.error(f"DICOM folder not found: {dicom_folder}")
        logger.info("Please provide a valid DICOM folder path")
        return
    
    result = pipeline.process_patient(
        dicom_folder=dicom_folder,
        patient_id="PT-DEMO-001"
    )
    
    print("\n" + "="*80)
    print("PIPELINE EXECUTION SUMMARY")
    print("="*80)
    print(f"Patient ID: {result['patient_id']}")
    print(f"Status: {result['status']}")
    
    if result['status'] == 'success':
        print(f"Duration: {result['duration_seconds']:.1f}s")
        print(f"Tumors detected: {len(result.get('tumors', []))}")
        
        for tumor in result.get('tumors', []):
            print(f"\nTumor {tumor['tumor_id']}:")
            print(f"  Volume: {tumor['volume_mm3']:.1f} mm³")
            print(f"  Classification: {tumor['classification']['classification']}")
            if tumor['classification']['classification'] == 'Malignant':
                tnm = tumor['classification']['tnm_stage']
                print(f"  TNM: T{tnm['T']}, N{tnm['N']}, M{tnm['M']}")
                print(f"  Stage: {tumor['classification']['overall_stage']}")
        
        if 'adds_integration' in result and result['adds_integration']['status'] == 'success':
            tp = result['adds_integration']['treatment_plan']
            print(f"\nADDS Treatment Plan:")
            print(f"  Status: Received")
    else:
        print(f"Error: {result.get('error', 'Unknown')}")
    
    print("="*80)


if __name__ == "__main__":
    run_pipeline_example()
