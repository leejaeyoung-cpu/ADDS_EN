"""
Segmentation API
Cellpose-based cell segmentation endpoint
"""

from fastapi import APIRouter, File, UploadFile, HTTPException
from typing import Optional
import numpy as np
from pathlib import Path
import tempfile
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))

from backend.services.segmentation_service import SegmentationService
from backend.schemas.cell_schema import SegmentationRequest, SegmentationResponse

router = APIRouter()
service = SegmentationService()

@router.post("/", response_model=SegmentationResponse)
async def segment_image(
    file: UploadFile = File(...),
    diameter: Optional[float] = None,
    flow_threshold: float = 0.6,
    cellprob_threshold: float = -1.0,
    batch_size: int = 8
):
    """
    Segment cells in microscopy image using Cellpose
    
    Args:
        file: Image file (TIFF, PNG, JPG)
        diameter: Cell diameter in pixels (None=auto-detect)
        flow_threshold: Flow error threshold (0.1-3.0)
        cellprob_threshold: Cell probability threshold
        batch_size: Batch size for GPU processing
        
    Returns:
        SegmentationResponse with masks, cell count, and metadata
    """
    
    # Validate file type
    if not file.filename.lower().endswith(('.tif', '.tiff', '.png', '.jpg', '.jpeg')):
        raise HTTPException(status_code=400, detail="Invalid file type. Use TIFF, PNG, or JPG")
    
    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.filename).suffix) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name
    
    try:
        # Perform segmentation
        result = await service.segment(
            image_path=tmp_path,
            diameter=diameter,
            flow_threshold=flow_threshold,
            cellprob_threshold=cellprob_threshold,
            batch_size=batch_size
        )
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Segmentation failed: {str(e)}")
        
    finally:
        # Cleanup
        Path(tmp_path).unlink(missing_ok=True)

@router.get("/models")
async def list_models():
    """List available Cellpose models"""
    return {
        "models": ["cyto", "cyto2", "cyto3", "nuclei", "custom"],
        "default": "cyto2",
        "recommended": "cyto2"
    }
