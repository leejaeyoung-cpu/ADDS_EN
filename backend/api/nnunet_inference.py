"""
nnU-Net Inference API
Provides endpoints for tumor segmentation using trained nnU-Net models
"""
import os as _os
from pathlib import Path as _Path
BASE_DIR = _Path(_os.environ.get("ADDS_BASE_DIR", str(_Path(__file__).resolve().parent.parent.parent)))

from fastapi import APIRouter, HTTPException, UploadFile, File
from pydantic import BaseModel
from typing import Dict, List, Optional, Any
from pathlib import Path
import tempfile
import shutil
import logging
import numpy as np
import os

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/nnunet", tags=["nnunet"])

# nnU-Net model paths
NNUNET_BASE = Path(os.environ.get("nnUNet_results", "nnUNet_results"))
DEFAULT_DATASET = "Dataset011_ColonMasked"
DEFAULT_CONFIG = "nnUNetTrainer__nnUNetPlans__3d_fullres"


class SegmentationRequest(BaseModel):
    """Request for segmentation from stored NIfTI file"""
    nifti_path: str
    dataset: str = DEFAULT_DATASET
    use_ensemble: bool = False
    folds: Optional[List[int]] = None


class SegmentationResponse(BaseModel):
    """Segmentation result"""
    status: str
    tumor_detected: bool = False
    tumor_volume_voxels: int = 0
    tumor_volume_ml: float = 0.0
    num_tumor_regions: int = 0
    bounding_box: Optional[Dict[str, int]] = None
    segmentation_path: Optional[str] = None
    method: str = "nnU-Net"
    message: str = ""


def _get_model_path(dataset: str = DEFAULT_DATASET) -> Path:
    """Resolve nnU-Net model path"""
    # Check multiple possible locations
    candidates = [
        NNUNET_BASE / dataset / DEFAULT_CONFIG,
        Path(BASE_DIR / "nnUNet_results") / dataset / DEFAULT_CONFIG,
        Path("C:/nnUNet_data/nnUNet_results") / dataset / DEFAULT_CONFIG,
    ]
    for p in candidates:
        if p.exists():
            return p
    return candidates[0]  # Return first candidate even if not found


def _get_predictor(model_path: Path, use_ensemble: bool = False, folds: list = None):
    """Initialize nnU-Net predictor"""
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))
    
    from medical_imaging.nnunet_predictor import nnUNetPredictorEnhanced
    
    predictor = nnUNetPredictorEnhanced(
        model_path=str(model_path),
        use_gpu=True,
        use_ensemble=use_ensemble,
        folds=folds
    )
    return predictor


@router.post("/segment", response_model=SegmentationResponse)
async def segment_tumor(request: SegmentationRequest):
    """
    Run nnU-Net tumor segmentation on a NIfTI volume
    
    Args:
        request: SegmentationRequest with nifti_path and options
    
    Returns:
        Segmentation result with tumor metrics
    """
    nifti_path = Path(request.nifti_path)
    if not nifti_path.exists():
        raise HTTPException(status_code=404, detail=f"NIfTI file not found: {nifti_path}")
    
    model_path = _get_model_path(request.dataset)
    
    logger.info(f"Running nnU-Net segmentation: {nifti_path}")
    logger.info(f"Model: {model_path}")
    
    try:
        predictor = _get_predictor(
            model_path, 
            use_ensemble=request.use_ensemble,
            folds=request.folds
        )
        
        result = predictor.predict_from_nifti(str(nifti_path))
        
        if result.get('status') == 'error':
            return SegmentationResponse(
                status="error",
                message=result.get('error', 'Unknown error')
            )
        
        # Calculate volume in ml (assuming 1mm³ spacing if not provided)
        tumor_volume_voxels = result.get('tumor_volume', 0)
        tumor_volume_ml = tumor_volume_voxels / 1000.0  # mm³ → ml
        
        # Count separate tumor regions
        mask = result.get('segmentation_mask')
        num_regions = 0
        if mask is not None and np.any(mask > 0):
            from scipy.ndimage import label
            labeled, num_regions = label(mask > 0)
        
        return SegmentationResponse(
            status="success",
            tumor_detected=result.get('tumor_detected', False),
            tumor_volume_voxels=tumor_volume_voxels,
            tumor_volume_ml=round(tumor_volume_ml, 2),
            num_tumor_regions=num_regions,
            bounding_box=result.get('bounding_box'),
            method=result.get('method', 'nnU-Net'),
            message=f"Segmentation complete: {num_regions} tumor region(s) detected"
        )
    
    except ImportError as e:
        logger.warning(f"nnU-Net not available: {e}")
        return SegmentationResponse(
            status="fallback",
            message="nnU-Net package not installed. Using fallback HU-based detection."
        )
    except Exception as e:
        logger.error(f"nnU-Net segmentation failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/models")
async def list_available_models():
    """List available nnU-Net models"""
    models = []
    
    # Check multiple result directories
    result_dirs = [
        NNUNET_BASE,
        Path(BASE_DIR / "nnUNet_results"),
    ]
    
    for results_dir in result_dirs:
        if not results_dir.exists():
            continue
        for dataset_dir in results_dir.iterdir():
            if dataset_dir.is_dir() and dataset_dir.name.startswith("Dataset"):
                for trainer_dir in dataset_dir.iterdir():
                    if trainer_dir.is_dir() and "nnUNet" in trainer_dir.name:
                        # Check for checkpoints
                        folds = []
                        for fold in range(5):
                            checkpoint = trainer_dir / f"fold_{fold}" / "checkpoint_best.pth"
                            if checkpoint.exists():
                                folds.append(fold)
                        
                        if folds:
                            models.append({
                                "dataset": dataset_dir.name,
                                "trainer": trainer_dir.name,
                                "folds_available": folds,
                                "path": str(trainer_dir)
                            })
    
    return {"models": models, "count": len(models)}


@router.get("/health")
async def health_check():
    """Health check"""
    model_path = _get_model_path()
    model_exists = model_path.exists()
    
    try:
        import nnunetv2
        nnunet_installed = True
    except ImportError:
        nnunet_installed = False
    
    return {
        "status": "healthy",
        "service": "nnunet",
        "model_path": str(model_path),
        "model_exists": model_exists,
        "nnunet_installed": nnunet_installed
    }
