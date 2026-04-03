"""
Treatment Response Prediction API
Exposes trained XGBoost model for CRC treatment response prediction.
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Dict, Optional, Any
import logging

logger = logging.getLogger(__name__)

router = APIRouter()

from backend.services.treatment_response_service import TreatmentResponsePredictor

_predictor = None


def _get_predictor() -> TreatmentResponsePredictor:
    global _predictor
    if _predictor is None:
        _predictor = TreatmentResponsePredictor()
    return _predictor


class TreatmentResponseRequest(BaseModel):
    """Request for treatment response prediction."""
    stage: Optional[int] = None          # TNM stage (1-4)
    kras_mut: Optional[int] = None       # 0 or 1
    braf_mut: Optional[int] = None       # 0 or 1
    tp53_mut: Optional[int] = None       # 0 or 1
    mmr_deficient: Optional[int] = None  # 0 or 1
    proximal: Optional[int] = None       # 0 or 1
    male: Optional[int] = None           # 0 or 1
    age: Optional[float] = None
    chemo_5fu: Optional[int] = None      # 0 or 1
    chemo_folfox: Optional[int] = None   # 0 or 1
    chemo_folfiri: Optional[int] = None  # 0 or 1
    subtype_C1: Optional[int] = None
    subtype_C2: Optional[int] = None
    subtype_C3: Optional[int] = None
    subtype_C4: Optional[int] = None
    subtype_C5: Optional[int] = None
    subtype_C6: Optional[int] = None
    additional_features: Optional[Dict[str, float]] = None


@router.post("/predict")
async def predict_treatment_response(request: TreatmentResponseRequest):
    """
    Predict CRC treatment response (3-year relapse-free probability).

    Model: XGBoost trained on GSE39582 (211 CRC chemo patients)
    CV Performance: AUC = 0.64 ± 0.08
    """
    predictor = _get_predictor()
    if not predictor.is_available:
        raise HTTPException(
            status_code=503,
            detail="Treatment response model not available"
        )

    # Build features dict from request
    features = {}
    for field in [
        "stage", "kras_mut", "braf_mut", "tp53_mut", "mmr_deficient",
        "proximal", "male", "age", "chemo_5fu", "chemo_folfox", "chemo_folfiri",
        "subtype_C1", "subtype_C2", "subtype_C3", "subtype_C4",
        "subtype_C5", "subtype_C6",
    ]:
        val = getattr(request, field)
        if val is not None:
            features[field] = val

    # Add any additional features (e.g., gene expression probe IDs)
    if request.additional_features:
        features.update(request.additional_features)

    result = predictor.predict(features)
    return result


@router.get("/features")
async def get_required_features():
    """Get the list of features used by the model."""
    predictor = _get_predictor()
    info = predictor.get_model_info()

    clinical = [
        "stage", "kras_mut", "braf_mut", "tp53_mut", "mmr_deficient",
        "proximal", "male", "age", "chemo_5fu", "chemo_folfox", "chemo_folfiri",
    ]
    subtypes = [f"subtype_C{i}" for i in range(1, 7)]
    gene_probes = [f for f in info.get("feature_names", [])
                   if f not in clinical and f not in subtypes]

    return {
        "clinical_features": clinical,
        "subtype_features": subtypes,
        "gene_expression_probes": gene_probes,
        "total": info.get("n_features", 0),
    }


@router.get("/feature-importance")
async def get_feature_importance(top_n: int = 20):
    """Get top feature importances from the model."""
    predictor = _get_predictor()
    return {"importances": predictor.get_feature_importance(top_n=top_n)}


@router.get("/model-info")
async def get_model_info():
    """Get model metadata and performance."""
    predictor = _get_predictor()
    return predictor.get_model_info()
