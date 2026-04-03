"""Cell Culture Microscopy Analysis API"""

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File
from sqlalchemy.orm import Session
from pathlib import Path
from datetime import datetime
import shutil
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from database import get_db
from database.models import CTAnalysis
from services.cell_culture_service import CellCultureService

router = APIRouter()
cell_culture_service = CellCultureService()

# Directory for microscopy uploads
MICROSCOPY_DIR = Path(__file__).parent.parent.parent / "outputs" / "microscopy"
MICROSCOPY_DIR.mkdir(parents=True, exist_ok=True)


@router.post("/analysis/{analysis_id}/upload_microscopy")
async def upload_microscopy_image(
    analysis_id: int,
    file: UploadFile = File(...),
    pixel_size_um: float = 0.5,
    db: Session = Depends(get_db)
):
    """
    Upload microscopy image for cell culture analysis
    
    Args:
        analysis_id: CT Analysis ID
        file: Microscopy image (TIFF, PNG, JPG)
        pixel_size_um: Pixel size in micrometers
        
    Returns:
        Cell culture analysis results
    """
    # Get analysis
    analysis = db.query(CTAnalysis).filter(CTAnalysis.id == analysis_id).first()
    if not analysis:
        raise HTTPException(status_code=404, detail="Analysis not found")
    
    # Validate file type
    allowed_extensions = ['.tif', '.tiff', '.png', '.jpg', '.jpeg']
    file_ext = Path(file.filename).suffix.lower()
    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type. Allowed: {', '.join(allowed_extensions)}"
        )
    
    # Save microscopy image
    microscopy_dir = MICROSCOPY_DIR / f"analysis_{analysis_id}"
    microscopy_dir.mkdir(parents=True, exist_ok=True)
    
    image_path = microscopy_dir / f"cell_culture{file_ext}"
    with open(image_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    # Run Cellpose analysis
    try:
        cell_analysis = cell_culture_service.analyze_microscopy_image(
            image_path=str(image_path),
            pixel_size_um=pixel_size_um
        )
        
        # Save results to database
        analysis.microscopy_image_path = str(image_path)
        analysis.cell_culture_data = cell_analysis
        db.commit()
        
        return {
            "analysis_id": analysis_id,
            "status": "completed",
            "microscopy_path": str(image_path),
            "cell_analysis": cell_analysis
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Cell culture analysis failed: {str(e)}"
        )


@router.get("/analysis/{analysis_id}/cell_culture")
async def get_cell_culture_results(
    analysis_id: int,
    db: Session = Depends(get_db)
):
    """
    Get cell culture analysis results
    
    Args:
        analysis_id: CT Analysis ID
        
    Returns:
        Cell culture analysis data
    """
    analysis = db.query(CTAnalysis).filter(CTAnalysis.id == analysis_id).first()
    if not analysis:
        raise HTTPException(status_code=404, detail="Analysis not found")
    
    if not analysis.cell_culture_data:
        raise HTTPException(
            status_code=404,
            detail="No cell culture analysis available for this analysis"
        )
    
    return {
        "analysis_id": analysis_id,
        "microscopy_image_path": analysis.microscopy_image_path,
        "cell_culture_data": analysis.cell_culture_data
    }


@router.post("/analysis/{analysis_id}/integrate_data")
async def integrate_ct_and_cell_culture(
    analysis_id: int,
    db: Session = Depends(get_db)
):
    """
    Integrate CT and cell culture data for comprehensive pharmacokinetic dataset
    
    Args:
        analysis_id: CT Analysis ID
        
    Returns:
        Integrated feature set for ADDS
    """
    analysis = db.query(CTAnalysis).filter(CTAnalysis.id == analysis_id).first()
    if not analysis:
        raise HTTPException(status_code=404, detail="Analysis not found")
    
    # Check required data
    if not analysis.radiomics_features:
        raise HTTPException(status_code=400, detail="CT radiomics not available")
    
    if not analysis.tumor_characteristics:
        raise HTTPException(status_code=400, detail="CT tumor characteristics not available")
    
    if not analysis.cell_culture_data:
        raise HTTPException(status_code=400, detail="Cell culture data not available")
    
    # Integrate data
    try:
        integrated_data = cell_culture_service.integrate_with_ct_data(
            ct_radiomics=analysis.radiomics_features,
            ct_tumor_chars=analysis.tumor_characteristics,
            cell_culture_data=analysis.cell_culture_data
        )
        
        # Update cell culture data with integrated features
        analysis.cell_culture_data["integrated_features"] = integrated_data
        db.commit()
        
        return {
            "analysis_id": analysis_id,
            "status": "integrated",
            "integrated_data": integrated_data,
            "message": "CT and cell culture data successfully integrated for pharmacokinetic modeling"
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Data integration failed: {str(e)}"
        )
