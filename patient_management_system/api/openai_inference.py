"""
OpenAI Inference API
GPT-4 medical analysis and drug recommendations
"""

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from database import get_db
from database.models import CTAnalysis
from database.schemas import ComparisonResult
from services.openai_service import OpenAIService

router = APIRouter()
openai_service = OpenAIService()


@router.post("/analysis/{analysis_id}/infer")
async def run_openai_inference(
    analysis_id: int,
    db: Session = Depends(get_db)
):
    """
    Run OpenAI GPT-4 medical inference
    
    Args:
        analysis_id: CT Analysis ID
        
    Returns:
        OpenAI drug recommendations and analysis
    """
    # Get analysis
    analysis = db.query(CTAnalysis).filter(CTAnalysis.id == analysis_id).first()
    if not analysis:
        raise HTTPException(status_code=404, detail="Analysis not found")
    
    # Check if CT analysis is complete
    if analysis.status != "completed":
        raise HTTPException(
            status_code=400,
            detail=f"CT analysis must be completed first. Current status: {analysis.status}"
        )
    
    # Check required data
    if not analysis.radiomics_features or not analysis.tumor_characteristics:
        raise HTTPException(
            status_code=400,
            detail="CT analysis must include radiomics features and tumor characteristics"
        )
    
    # Prepare clinical data
    clinical_data = {
        "patient_id": analysis.patient.patient_id,
        "tumor_location": analysis.tumor_location or "Unknown",
        "tnm_stage": analysis.tnm_stage or "Unknown",
        "msi_status": analysis.msi_status or "Unknown",
        "kras_mutation": analysis.kras_mutation or "Unknown"
    }
    
    # Run OpenAI inference
    try:
        result = openai_service.run_medical_inference(
            radiomics=analysis.radiomics_features,
            tumor_characteristics=analysis.tumor_characteristics,
            clinical_data=clinical_data
        )
        
        # Save result
        analysis.openai_result = result
        db.commit()
        
        return {
            "analysis_id": analysis_id,
            "status": "completed",
            "result": result
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"OpenAI inference failed: {str(e)}")


@router.get("/analysis/{analysis_id}/result")
async def get_openai_result(
    analysis_id: int,
    db: Session = Depends(get_db)
):
    """
    Get OpenAI inference result
    
    Args:
        analysis_id: CT Analysis ID
        
    Returns:
        OpenAI inference result
    """
    analysis = db.query(CTAnalysis).filter(CTAnalysis.id == analysis_id).first()
    if not analysis:
        raise HTTPException(status_code=404, detail="Analysis not found")
    
    if not analysis.openai_result:
        raise HTTPException(
            status_code=404,
            detail="OpenAI inference has not been run for this analysis"
        )
    
    return analysis.openai_result


@router.get("/analysis/{analysis_id}/compare", response_model=ComparisonResult)
async def compare_results(
    analysis_id: int,
    db: Session = Depends(get_db)
):
    """
    Compare ADDS and OpenAI inference results
    
    Args:
        analysis_id: CT Analysis ID
        
    Returns:
        Comparison analysis with drug overlap and consensus
    """
    analysis = db.query(CTAnalysis).filter(CTAnalysis.id == analysis_id).first()
    if not analysis:
        raise HTTPException(status_code=404, detail="Analysis not found")
    
    if not analysis.adds_result:
        raise HTTPException(status_code=400, detail="ADDS inference not available")
    
    if not analysis.openai_result:
        raise HTTPException(status_code=400, detail="OpenAI inference not available")
    
    # Extract drug lists
    adds_drugs = set(analysis.adds_result.get("recommended_drugs", []))
    openai_drugs = set(analysis.openai_result.get("recommended_drugs", []))
    
    # Calculate overlap
    overlap_drugs = adds_drugs.intersection(openai_drugs)
    unique_to_adds = list(adds_drugs - openai_drugs)
    unique_to_openai = list(openai_drugs - adds_drugs)
    consensus_drugs = list(overlap_drugs)
    
    # Calculate overlap percentage
    total_unique_drugs = len(adds_drugs.union(openai_drugs))
    drug_overlap = len(overlap_drugs) / total_unique_drugs if total_unique_drugs > 0 else 0.0
    
    result = {
        "adds": analysis.adds_result,
        "openai": analysis.openai_result,
        "drug_overlap": drug_overlap,
        "consensus_drugs": consensus_drugs,
        "unique_to_adds": unique_to_adds,
        "unique_to_openai": unique_to_openai
    }
    
    return result
