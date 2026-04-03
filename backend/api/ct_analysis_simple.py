"""
SIMPLIFIED CT Analysis API - Direct call to proven detection script
"""
from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from typing import List
import tempfile
import shutil
from pathlib import Path
import logging
import numpy as np

router = APIRouter()
logger = logging.getLogger(__name__)

@router.post("/analyze")
async def analyze_ct_scan(
    patient_id: str,
    files: List[UploadFile] = File(...)
) -> JSONResponse:
    """
    Analyze CT scan using proven detection script
    """
    
    if not files:
        raise HTTPException(status_code=400, detail="No DICOM files provided")
    
    logger.info(f"Received CT analysis request for patient {patient_id}")
    logger.info(f"Number of DICOM files: {len(files)}")
    
    # Create temporary directory
    temp_dir = Path(tempfile.mkdtemp(prefix=f"ct_{patient_id}_"))
    
    try:
        # Save uploaded files
        dicom_paths = []
        for file in files:
            content = await file.read()
            file_path = temp_dir / file.filename
            with open(file_path, 'wb') as f:
                f.write(content)
            dicom_paths.append(file_path)
        
        logger.info(f"Saved {len(dicom_paths)} DICOM files")
        
        # Find or create NIfTI
        nifti_files = list(temp_dir.glob("*.nii.gz"))
        if not nifti_files:
            logger.info("Reconstructing volume from DICOM...")
            from src.medical_imaging.ct_volume_reconstruction import CTVolumeReconstructor
            reconstructor = CTVolumeReconstructor()
            volume, metadata = reconstructor.reconstruct_3d_volume(temp_dir, normalize=False)
            
            import nibabel as nib
            nifti_path = temp_dir / "volume.nii.gz"
            spacing = metadata.get('spacing', (1.0, 1.0, 1.0))
            affine = np.diag([spacing[0], spacing[1], spacing[2], 1.0])
            nii = nib.Nifti1Image(volume, affine=affine)
            nib.save(nii, nifti_path)
            logger.info(f"Volume saved: {nifti_path}")
        else:
            nifti_path = nifti_files[0]
            logger.info(f"Using existing NIfTI: {nifti_path}")
        
        # Call proven detection script
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent.parent))
        from detect_tumors_inha_corrected import run_detection
        
        output_dir = temp_dir / "results"
        output_dir.mkdir(exist_ok=True)
        
        logger.info(f"Running detection...")
        result = run_detection(nifti_path, output_dir)
        
        if result['status'] == 'error':
            raise Exception(result.get('message', 'Detection failed'))
        
        logger.info(f"Detection complete: {result['tumor_count']} tumors")
        
        # Load images
        analyzed_images = []
        tumor_results = [r for r in result['results'] if r.get('has_tumor')]
        top5 = sorted(tumor_results, key=lambda x: x.get('max_confidence', 0), reverse=True)[:5]
        
        for r in top5:
            img_path = output_dir / f"slice_{r['slice_idx']:03d}_detection.png"
            if img_path.exists():
                import base64
                with open(img_path, 'rb') as f:
                    b64 = base64.b64encode(f.read()).decode('utf-8')
                    analyzed_images.append(f"data:image/png;base64,{b64}")
        
        logger.info(f"Loaded {len(analyzed_images)} images")
        
        return JSONResponse(content={
            'status': 'success',
            'patient_id': patient_id,
            'tumors_detected': result['tumor_count'],
            'total_slices': result['total_slices'],
            'detection_rate': result['detection_rate'],
            'analyzed_images': analyzed_images
        })
        
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
        
    finally:
        try:
            shutil.rmtree(temp_dir)
        except:
            pass
