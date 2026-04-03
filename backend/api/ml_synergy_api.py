"""
ML Drug Synergy Prediction API
Exposes trained XGBoost models for drug combination synergy prediction.
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)

router = APIRouter()

from backend.services.ml_synergy_service import MLSynergyService

_service = None


def _get_service() -> MLSynergyService:
    global _service
    if _service is None:
        _service = MLSynergyService()
    return _service


class SynergyPredictRequest(BaseModel):
    model_config = {"protected_namespaces": ()}
    drug_a: str
    drug_b: str
    cell_line: str = "HCT116"
    model_variant: str = "combined"


class SynergyCompareRequest(BaseModel):
    drug_a: str
    drug_b: str
    cell_line: str = "HCT116"


@router.post("/predict")
async def predict_synergy(request: SynergyPredictRequest):
    """
    Predict drug combination synergy using XGBoost model.

    Model variants: morgan, combined, fixAB, realexpr
    Data: O'Neil et al. 2016 (23,052 combos, 38 drugs, 39 cell lines)
    """
    svc = _get_service()
    if not svc.available_models:
        raise HTTPException(status_code=503, detail="No ML synergy models available")

    result = svc.predict(
        drug_a=request.drug_a,
        drug_b=request.drug_b,
        cell_line=request.cell_line,
        model_variant=request.model_variant,
    )

    if result.get("status") == "error":
        raise HTTPException(status_code=400, detail=result.get("message"))

    return result


@router.post("/compare")
async def compare_synergy_models(request: SynergyCompareRequest):
    """
    Compare predictions across all 4 XGBoost model variants.

    Returns per-model scores + consensus classification.
    """
    svc = _get_service()
    result = svc.compare_models(
        drug_a=request.drug_a,
        drug_b=request.drug_b,
        cell_line=request.cell_line,
    )

    if result.get("status") == "error":
        raise HTTPException(status_code=400, detail=result.get("message"))

    return result


@router.get("/drugs")
async def list_available_drugs():
    """List drugs available for synergy prediction."""
    svc = _get_service()
    return {
        "drugs": svc.available_drugs,
        "count": len(svc.available_drugs),
    }


@router.get("/models")
async def list_synergy_models():
    """List available XGBoost synergy model variants."""
    svc = _get_service()
    return {
        "models": svc.available_models,
        "count": len(svc.available_models),
    }


@router.get("/model-info")
async def get_model_info():
    """Get synergy model metadata and data source info."""
    svc = _get_service()
    return svc.get_model_info()


class DeepSynergyRequest(BaseModel):
    drug_a: str
    drug_b: str
    cell_line: str = "HCT116"


@router.post("/deep-synergy/predict")
async def predict_deep_synergy(request: DeepSynergyRequest):
    """
    Predict synergy using DeepSynergy v5 (PyTorch MLP).

    Features: Morgan FP (2048d) + CL expression (256d DepMap) + tissue (12d).
    Trained on 927K DrugComb records covering 4,246 drugs.
    Drug-pair holdout r=0.707 (62K unique pairs, 5-fold CV).
    """
    svc = _get_service()
    result = svc.predict_deep_synergy(
        drug_a=request.drug_a,
        drug_b=request.drug_b,
        cell_line=request.cell_line,
    )
    if result.get("status") == "error":
        raise HTTPException(status_code=400, detail=result.get("message"))
    return result
