"""
Pharmacokinetics Analysis API
2-compartment PK modeling with drug-specific parameters
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Dict, List, Optional, Any
import logging

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/pharmacokinetics", tags=["pharmacokinetics"])

# Import the PK engine
from backend.services.pharmacokinetics_service import PharmacokineticEngine, DRUG_PK_PARAMS

pk_engine = PharmacokineticEngine()


class PharmacokineticRequest(BaseModel):
    """Request model for PK analysis"""
    patient_id: str
    drug_name: str = "5-fluorouracil"
    ct_analysis: Optional[Dict[str, Any]] = None
    cell_analysis: Optional[Dict[str, Any]] = None
    pathology: Optional[Dict[str, Any]] = None
    body_surface_area: float = 1.7
    body_weight: float = 70.0
    renal_function: float = 100.0
    hepatic_function: str = "normal"
    age: float = 60.0
    infusion_duration_hours: float = 2.0
    dose_mg_m2: Optional[float] = None


class PharmacokineticResponse(BaseModel):
    """Response model for PK analysis"""
    patient_id: str
    drug_name: str
    clearance_l_h: float
    volume_central_l: float
    volume_peripheral_l: float
    half_life_alpha_hours: float
    half_life_beta_hours: float
    half_life_effective_hours: float
    auc_ug_h_ml: float
    cmax_ug_ml: float
    trough_ug_ml: float
    optimal_dose_mg_m2: float
    dosing_interval_hours: int
    predicted_efficacy: float
    toxicity_risk: str
    therapeutic_window: List[float]
    recommendations: List[str]
    concentration_profile: Optional[Dict[str, Any]] = None
    
    # Legacy compatibility
    clearance_ml_min: float = 0
    volume_distribution_l: float = 0
    half_life_hours: float = 0


@router.post("/analyze", response_model=PharmacokineticResponse)
async def analyze_pharmacokinetics(request: PharmacokineticRequest):
    """
    Analyze pharmacokinetics using 2-compartment model
    
    Supports drug-specific PK parameters for:
    5-fluorouracil, oxaliplatin, irinotecan, bevacizumab, cetuximab, pembrolizumab
    """
    try:
        logger.info(f"PK analysis for patient {request.patient_id}, drug={request.drug_name}")
        
        # Extract clinical parameters
        tumor_volume_cm3 = 0.0
        ki67_index = 0.0
        
        if request.ct_analysis:
            tumor_volume_cm3 = request.ct_analysis.get('total_tumor_volume_cm3', 0.0)
        
        if request.cell_analysis:
            ki67_index = request.cell_analysis.get('ki67_index', 0.0)
        
        # Run 2-compartment PK analysis
        result = pk_engine.analyze(
            drug_name=request.drug_name,
            tumor_volume_cm3=tumor_volume_cm3,
            ki67_index=ki67_index,
            body_surface_area=request.body_surface_area,
            body_weight=request.body_weight,
            renal_function=request.renal_function,
            hepatic_function=request.hepatic_function,
            age=request.age,
            infusion_duration=request.infusion_duration_hours,
            dose_mg_m2=request.dose_mg_m2
        )
        
        # Build recommendations
        recommendations = []
        
        if result.toxicity_risk == "High":
            recommendations.append("⚠️ High toxicity risk — consider dose reduction (75%)")
        
        if result.cmax > result.therapeutic_window[1]:
            recommendations.append(f"Cmax ({result.cmax:.2f}) exceeds upper limit — extend infusion duration")
        
        if result.trough < result.therapeutic_window[0]:
            recommendations.append(f"Trough ({result.trough:.2f}) may be subtherapeutic — consider shorter interval")
        
        if tumor_volume_cm3 > 100:
            recommendations.append("Large tumor burden — neoadjuvant therapy may improve drug exposure")
        
        if ki67_index > 50:
            recommendations.append("High Ki-67 proliferation — chemotherapy likely effective")
        
        if request.renal_function < 60:
            recommendations.append("Impaired renal function — mandatory dose adjustment per protocol")
        
        if request.hepatic_function != "normal":
            recommendations.append(f"Hepatic impairment ({request.hepatic_function}) — clearance reduced")
        
        if abs(result.optimal_dose - (request.dose_mg_m2 or result.optimal_dose)) > 20:
            recommendations.append(
                f"Dose optimization suggests {result.optimal_dose:.0f} mg/m² (current protocol may differ)"
            )
        
        if not recommendations:
            recommendations.append("Standard dosing protocol appropriate — no adjustments needed")
        
        # Legacy compatibility fields (convert L/h → mL/min)
        cl_ml_min = result.clearance * 1000 / 60
        
        logger.info(f"✓ PK complete: {request.drug_name}, t½={result.half_life_effective:.1f}h, "
                     f"dose={result.optimal_dose:.0f} mg/m²")
        
        return PharmacokineticResponse(
            patient_id=request.patient_id,
            drug_name=result.drug_name,
            clearance_l_h=result.clearance,
            volume_central_l=result.volume_central,
            volume_peripheral_l=result.volume_peripheral,
            half_life_alpha_hours=result.half_life_alpha,
            half_life_beta_hours=result.half_life_beta,
            half_life_effective_hours=result.half_life_effective,
            auc_ug_h_ml=result.auc,
            cmax_ug_ml=result.cmax,
            trough_ug_ml=result.trough,
            optimal_dose_mg_m2=result.optimal_dose,
            dosing_interval_hours=result.dosing_interval,
            predicted_efficacy=result.predicted_efficacy,
            toxicity_risk=result.toxicity_risk,
            therapeutic_window=list(result.therapeutic_window),
            recommendations=recommendations,
            concentration_profile=result.concentration_profile,
            # Legacy
            clearance_ml_min=round(cl_ml_min, 2),
            volume_distribution_l=result.volume_central,
            half_life_hours=result.half_life_effective
        )
    
    except Exception as e:
        logger.error(f"Error in pharmacokinetics analysis: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/drugs")
async def list_available_drugs():
    """List drugs with available PK parameters"""
    drugs = []
    for name, params in DRUG_PK_PARAMS.items():
        if name == "default":
            continue
        drugs.append({
            "name": name,
            "typical_dose_mg_m2": params["typical_dose"],
            "route": params["route"],
            "therapeutic_range": params["therapeutic_range"]
        })
    return {"drugs": drugs}


@router.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "pharmacokinetics", "model": "2-compartment"}
