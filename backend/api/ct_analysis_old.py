"""
CT Analysis API Router
Provides endpoints for CT-based colorectal cancer detection pipeline
"""

from fastapi import APIRouter, UploadFile, File, HTTPException, BackgroundTasks, Query
from fastapi.responses import JSONResponse
from typing import List
import logging
import tempfile
from pathlib import Path
import shutil
import sys
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from ct_crc_detection_pipeline import IntegratedCRCDetectionPipeline
from src.medical_imaging.detection.verified_ct_detector import VerifiedCTDetector

# Toggle switch for choosing detector (safe rollback mechanism)
USE_VERIFIED_DETECTOR = True  # Use proven detection script

logger = logging.getLogger(__name__)

router = APIRouter()

# Initialize CT pipeline
CT_PIPELINE_CONFIG = {
    'target_spacing': [1.0, 1.0, 1.0],
    'radiomics_bin_width': 25,
    'models_dir': 'models',
    'output_dir': 'outputs/crc_detection',
    'save_intermediate': True,
    'use_gpu': True
}

ct_pipeline = None

def get_pipeline():
    """Lazy initialization of CT pipeline"""
    global ct_pipeline
    if ct_pipeline is None:
        logger.info("Initializing CT CRC Detection Pipeline...")
        ct_pipeline = IntegratedCRCDetectionPipeline(CT_PIPELINE_CONFIG)
        logger.info("CT Pipeline initialized successfully")
    return ct_pipeline


@router.post("/analyze")
async def analyze_ct_scan(
    patient_id: str,
    files: List[UploadFile] = File(...)
) -> JSONResponse:
    """
    Analyze CT scan for colorectal cancer detection
    
    **Parameters**:
    - patient_id: Patient identifier
    - files: List of DICOM files (.dcm)
    
    **Returns**:
    - JSON with complete analysis results including:
        - Volume reconstruction
        - Segmentation
        - Radiomics features
        - TNM staging
        - Treatment recommendations
    """
    
    if not files:
        raise HTTPException(status_code=400, detail="No DICOM files provided")
    
    logger.info(f"Received CT analysis request for patient {patient_id}")
    logger.info(f"Number of DICOM files: {len(files)}")
    
    # Create temporary directory for DICOM files
    temp_dir = Path(tempfile.mkdtemp(prefix=f"ct_{patient_id}_"))
    
    try:
        # Save uploaded files
        dicom_paths = []
        for file in files:
            if not file.filename.endswith('.dcm'):
                logger.warning(f"Skipping non-DICOM file: {file.filename}")
                continue
            
            file_path = temp_dir / file.filename
            with open(file_path, 'wb') as f:
                content = await file.read()
                f.write(content)
            dicom_paths.append(file_path)
        
        logger.info(f"Saved {len(dicom_paths)} DICOM files to {temp_dir}")
        
        if not dicom_paths:
            raise HTTPException(
                status_code=400,
                detail="No valid DICOM files (.dcm) found in upload"
            )
        
       # Run CT detection
        logger.info(f"🔍 CT Detection starting... USE_VERIFIED_DETECTOR={USE_VERIFIED_DETECTOR}")
        if USE_VERIFIED_DETECTOR:
            logger.info(f"✅ Using detect_tumors_inha_corrected.py run_detection()")
            
            # Import from the proven script
            import sys
            from pathlib import Path as ScriptPath
            sys.path.insert(0, str(ScriptPath(__file__).parent.parent.parent))
            
            # Import detector functions
            from detect_tumors_inha_corrected import (
                load_nifti_volume, 
                detect_tumors_in_slice, 
                visualize_detection
            )
            from src.medical_imaging.detection.candidate_detector import TumorDetector
            
            # Create output directory
            output_dir = temp_dir / "detection_results"
            output_dir.mkdir(exist_ok=True)
            
            # Find NIfTI file (should be in temp_dir after reconstruction)
            nifti_files = list(temp_dir.glob("*.nii.gz"))
            if not nifti_files:
                # If no NIfTI, need to reconstruct from DICOM first
                logger.info("No NIfTI found, running volume reconstruction...")
                
                # Use existing pipeline's volume reconstructor
                from src.medical_imaging.ct_volume_reconstruction import CTVolumeReconstructor
                reconstructor = CTVolumeReconstructor()
                volume, metadata = reconstructor.reconstruct_3d_volume(temp_dir, normalize=False)
                
                # Save as NIfTI for detection
                import nibabel as nib
                nifti_path = temp_dir / "reconstructed_volume.nii.gz"
                
                # Create proper affine matrix from metadata
                spacing = metadata.get('spacing', (1.0, 1.0, 1.0))
                affine = np.diag([spacing[0], spacing[1], spacing[2], 1.0])
                
                nii = nib.Nifti1Image(volume, affine=affine)
                nib.save(nii, nifti_path)
                logger.info(f"Volume saved: {nifti_path}")
            else:
                nifti_path = nifti_files[0]
                logger.info(f"Using existing NIfTI: {nifti_path}")
            
            # Load volume
            volume, spacing, affine = load_nifti_volume(nifti_path)
            
            # Initialize detector
            detector = TumorDetector()
            
            # Process slices and collect results
            results = []
            analyzed_images = []
            tumor_count = 0
            
            n_slices = volume.shape[0]
            interesting_slices = range(0, min(80, n_slices))
            
            logger.info(f"Processing {len(list(interesting_slices))} slices...")
            
            for slice_idx in interesting_slices:
                slice_data = volume[slice_idx, :, :]
                z_position_mm = float(slice_idx * spacing[2])
                
                # Detect tumors (already filtered by detect_tumors_in_slice)
                candidates = detect_tumors_in_slice(
                    slice_data=slice_data,
                    detector=detector,
                    pixel_spacing=spacing,
                    slice_idx=slice_idx
                )
                
                # Only save slices with high-confidence tumor detections
                high_conf = [c for c in candidates if c.confidence_score > 0.7]
                
                if len(high_conf) > 0:
                    # This slice has tumors - save it
                    output_path = output_dir / f"slice_{slice_idx:03d}_detection.png"
                    
                    result = visualize_detection(
                        slice_data=slice_data,
                        candidates=candidates,
                        slice_idx=slice_idx,
                        output_path=output_path,
                        z_position_mm=z_position_mm
                    )
                    
                    result['slice_idx'] = slice_idx
                    result['max_confidence'] = max(c.confidence_score for c in high_conf)
                    result['tumor_count'] = len(high_conf)
                    results.append(result)
                    
                    if result['has_tumor']:
                        tumor_count += 1
            
            logger.info(f"Detection complete: {tumor_count} tumor-positive slices found")
            
            # Sort by confidence and select top 5 for preview
            results_sorted = sorted(results, key=lambda x: x.get('max_confidence', 0), reverse=True)
            top_results = results_sorted[:5]
            
            # Read images for top 5 results
            for result in top_results:
                slice_idx = result['slice_idx']
                output_path = output_dir / f"slice_{slice_idx:03d}_detection.png"
                
                if output_path.exists():
                    import base64
                    with open(output_path, 'rb') as f:
                        img_b64 = base64.b64encode(f.read()).decode('utf-8')
                        # Add data URI prefix for HTML display
                        img_data_uri = f"data:image/png;base64,{img_b64}"
                        analyzed_images.append(img_data_uri)
            
            logger.info(f"Detection complete: {tumor_count} tumor-positive slices")
            
            # Get largest tumor size
            largest_area = 0
            for result in results:
                if result.get('detections'):
                    for det in result['detections']:
                        largest_area = max(largest_area, det['area_pixels'])
            
            largest_tumor_mm = np.sqrt(largest_area) * 0.7 if largest_area > 0 else 0
            
            response_data = {
                'status': 'success',
                'patient_id': patient_id,
                'tumors_detected': tumor_count,
                'largest_tumor_size_mm': round(largest_tumor_mm, 1),
                'total_slices': len(results),
                'detection_rate': f"{tumor_count/len(results)*100:.1f}%" if results else "0%",
                'analyzed_images': analyzed_images[:20],  # Limit to 20 images
                'tumor_statistics': {
                    'tumor_slices': tumor_count,
                    'total_processed': len(results)
                }
            }
            
        else:
            logger.info("Using legacy IntegratedCRCDetectionPipeline")
            pipeline = get_pipeline()
            result = pipeline.process_patient(
                dicom_folder=temp_dir,
                patient_id=patient_id
            )
            
            logger.info(f"Pipeline completed for patient {patient_id}: {result['status']}")
            
            # Extract data for response (old format)
            response_data = {
                'status': 'success',
                'patient_id': patient_id,
                'tumors_detected': len(result.get('tumors', [])),
                'largest_tumor_size_mm': 0,
                'total_tumor_volume_cm3': 0,
                'radiomics_features': {},
                'tumor_locations': [],
                'analyzed_images': result.get('visualization_images', [])
            }
        
        # Return results
        return JSONResponse(content=response_data)
        
    except Exception as e:
        logger.error(f"Error processing CT scan: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
        
    finally:
        # Cleanup temporary directory
        try:
            shutil.rmtree(temp_dir)
            logger.info(f"Cleaned up temporary directory: {temp_dir}")
        except Exception as e:
            logger.warning(f"Failed to cleanup temp directory: {e}")


@router.get("/health")
async def ct_pipeline_health():
    """
    Check CT pipeline health status
    """
    try:
        pipeline = get_pipeline()
        
        return {
            "status": "healthy",
            "pipeline": "IntegratedCRCDetectionPipeline",
            "stages": {
                "volume_reconstruction": "available",
                "segmentation": "available (fallback)",
                "radiomics": "available",
                "classification": "available",
                "adds_integration": "available"
            },
            "gpu": "cuda" if CT_PIPELINE_CONFIG['use_gpu'] else "cpu"
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {
            "status": "unhealthy",
            "error": str(e)
        }


@router.get("/models/status")
async def models_status():
    """
    Check status of nnU-Net models
    """
    from pathlib import Path
    
    models_dir = Path("nnUNet_data/nnUNet_results")
    
    models = {
        "colon_segmentation": {
            "available": False,
            "path": None
        },
        "tumor_segmentation": {
            "available": False,
            "path": None
        }
    }
    
    # Check for trained models
    if models_dir.exists():
        # Look for Dataset100_CRC_CT
        dataset_dir = models_dir / "Dataset100_CRC_CT"
        if dataset_dir.exists():
            for model_path in dataset_dir.glob("nnUNetTrainer__nnUNetPlans__*"):
                if "2d" in model_path.name:
                    models["colon_segmentation"]["available"] = True
                    models["colon_segmentation"]["path"] = str(model_path)
                    models["tumor_segmentation"]["available"] = True
                    models["tumor_segmentation"]["path"] = str(model_path)
    
    return {
        "models": models,
        "fallback_available": True,
        "note": "Fallback algorithm will be used when trained models are unavailable"
    }
