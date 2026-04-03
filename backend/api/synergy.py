"""
Drug Synergy API
Calculate drug combination synergy scores
"""

from fastapi import APIRouter, HTTPException
from typing import Optional

from backend.services.synergy_service import SynergyService
from backend.schemas.analysis_schema import SynergyRequest, SynergyResponse

router = APIRouter()
service = SynergyService()

@router.post("/calculate", response_model=SynergyResponse)
async def calculate_synergy(request: SynergyRequest):
    """
    Calculate drug combination synergy
    
    Models:
    - bliss: Bliss Independence
    - loewe: Loewe Additivity
    - hsa: Highest Single Agent
    - zip: Zero Interaction Potency
    
    Returns:
        Synergy score, significance, and model details
    """
    
    try:
        result = await service.calculate(
            drug_a_effect=request.drug_a_effect,
            drug_b_effect=request.drug_b_effect,
            combination_effect=request.combination_effect,
            model=request.model
        )
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Synergy calculation failed: {str(e)}")

@router.get("/models")
async def list_models():
    """List available synergy models"""
    return {
        "models": {
            "bliss": "Bliss Independence (most common)",
            "loewe": "Loewe Additivity (dose-based)",
            "hsa": "Highest Single Agent (conservative)",
            "zip": "Zero Interaction Potency (advanced)"
        },
        "default": "bliss"
    }
