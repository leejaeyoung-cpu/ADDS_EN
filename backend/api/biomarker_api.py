"""
Biomarker Prediction API
Predicts KRAS mutation, MSI status, and BRAF mutation from radiomics features.

When trained models are available: uses Logistic Regression from radiomics shape/texture.
When no training data: returns explicit 'model_not_trained' status.

Ref: Liang et al. "Prediction of KRAS mutation status in CRC
     using CT-based radiomic features" (Radiology 2020)
"""

import os as _os
from pathlib import Path as _Path
BASE_DIR = _Path(_os.environ.get("ADDS_BASE_DIR", str(_Path(__file__).resolve().parent.parent.parent)))

import os as _os
from pathlib import Path as _Path
BASE_DIR = _Path(_os.environ.get("ADDS_BASE_DIR", str(_Path(__file__).resolve().parent.parent.parent)))

import os as _os
from pathlib import Path as _Path
BASE_DIR = _Path(_os.environ.get("ADDS_BASE_DIR", str(_Path(__file__).resolve().parent.parent.parent)))

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Dict, List, Optional, Any
import logging
import numpy as np
from pathlib import Path

logger = logging.getLogger(__name__)

router = APIRouter()

MODEL_DIR = Path(BASE_DIR / "models/biomarker")


class BiomarkerPredictionRequest(BaseModel):
    """Radiomics or clinical features for biomarker prediction."""
    # Radiomics features (from PyRadiomics)
    tumor_volume_cm3: Optional[float] = None
    surface_area_cm2: Optional[float] = None
    sphericity: Optional[float] = None
    compactness: Optional[float] = None
    elongation: Optional[float] = None
    flatness: Optional[float] = None
    # Texture features (GLCM)
    glcm_contrast: Optional[float] = None
    glcm_correlation: Optional[float] = None
    glcm_energy: Optional[float] = None
    glcm_homogeneity: Optional[float] = None
    # First-order statistics
    mean_hu: Optional[float] = None
    std_hu: Optional[float] = None
    skewness: Optional[float] = None
    kurtosis: Optional[float] = None
    entropy: Optional[float] = None
    # Location
    tumor_location: Optional[str] = None  # proximal, distal, rectum
    # Patient clinical
    age: Optional[float] = None
    sex: Optional[str] = None


class BiomarkerModel:
    """Simple radiomics-based biomarker predictor."""

    def __init__(self, biomarker_name: str):
        self.biomarker_name = biomarker_name
        self.model = None
        self.scaler = None
        self._load_model()

    def _load_model(self):
        """Try to load a trained model."""
        model_path = MODEL_DIR / f"{self.biomarker_name}_model.pkl"
        if model_path.exists():
            try:
                import pickle
                with open(model_path, "rb") as f:
                    saved = pickle.load(f)
                self.model = saved.get("model")
                self.scaler = saved.get("scaler")
                logger.info(f"Loaded biomarker model: {self.biomarker_name}")
            except Exception as e:
                logger.warning(f"Failed to load {self.biomarker_name} model: {e}")

    @property
    def is_trained(self) -> bool:
        return self.model is not None

    def predict(self, features: Dict[str, float]) -> Dict[str, Any]:
        if not self.is_trained:
            # Return explicit untrained status — NOT a fake prediction
            return {
                "biomarker": self.biomarker_name,
                "status": "model_not_trained",
                "message": (
                    f"No trained model for {self.biomarker_name}. "
                    "Requires radiomics data from TCGA/TCIA for training. "
                    "Ref: Liang et al. Radiology 2020."
                ),
                "probability": None,
                "prediction": None,
            }

        # Build feature vector
        feature_keys = [
            "tumor_volume_cm3", "surface_area_cm2", "sphericity", "compactness",
            "elongation", "flatness", "glcm_contrast", "glcm_correlation",
            "glcm_energy", "glcm_homogeneity", "mean_hu", "std_hu",
            "skewness", "kurtosis", "entropy",
        ]
        x = np.array([features.get(k, 0.0) for k in feature_keys]).reshape(1, -1)

        if self.scaler:
            x = self.scaler.transform(x)

        try:
            prob = self.model.predict_proba(x)[0]
            pred = int(self.model.predict(x)[0])
            positive_prob = float(prob[1]) if len(prob) > 1 else float(prob[0])

            return {
                "biomarker": self.biomarker_name,
                "status": "success",
                "probability": round(positive_prob, 4),
                "prediction": "positive" if pred == 1 else "negative",
                "confidence": round(abs(positive_prob - 0.5) * 2, 4),
            }
        except Exception as e:
            return {
                "biomarker": self.biomarker_name,
                "status": "error",
                "message": str(e),
            }


# Lazy-loaded biomarker models
_kras_model = None
_msi_model = None
_braf_model = None


def _get_kras():
    global _kras_model
    if _kras_model is None:
        _kras_model = BiomarkerModel("kras")
    return _kras_model


def _get_msi():
    global _msi_model
    if _msi_model is None:
        _msi_model = BiomarkerModel("msi")
    return _msi_model


def _get_braf():
    global _braf_model
    if _braf_model is None:
        _braf_model = BiomarkerModel("braf")
    return _braf_model


def _request_to_features(request: BiomarkerPredictionRequest) -> Dict[str, float]:
    features = {}
    for field in request.model_fields:
        val = getattr(request, field)
        if val is not None and isinstance(val, (int, float)):
            features[field] = float(val)
    return features


@router.post("/predict-kras")
async def predict_kras_mutation(request: BiomarkerPredictionRequest):
    """
    Predict KRAS mutation status from radiomics features.

    Returns explicit 'model_not_trained' if no training data available.
    Ref: Liang et al. "CT-based radiomic KRAS prediction" (Radiology 2020)
    """
    model = _get_kras()
    features = _request_to_features(request)
    return model.predict(features)


@router.post("/predict-msi")
async def predict_msi_status(request: BiomarkerPredictionRequest):
    """
    Predict MSI (Microsatellite Instability) status from radiomics.

    Ref: Fan et al. "Radiomic analysis of MSI in CRC" (EJSO 2019)
    """
    model = _get_msi()
    features = _request_to_features(request)
    return model.predict(features)


@router.post("/predict-braf")
async def predict_braf_mutation(request: BiomarkerPredictionRequest):
    """
    Predict BRAF V600E mutation from radiomics.

    Ref: Liang et al. (Radiology 2020)
    """
    model = _get_braf()
    features = _request_to_features(request)
    return model.predict(features)


@router.get("/available")
async def list_biomarkers():
    """List biomarker prediction capabilities and their status."""


    return {
        "biomarkers": [
            {
                "name": "kras",
                "description": "KRAS mutation (G12D, G12V, G13D, etc.)",
                "trained": _get_kras().is_trained,
                "reference": "Liang et al. Radiology 2020",
            },
            {
                "name": "msi",
                "description": "Microsatellite Instability (MSI-H vs MSS)",
                "trained": _get_msi().is_trained,
                "reference": "Fan et al. EJSO 2019",
            },
            {
                "name": "braf",
                "description": "BRAF V600E mutation",
                "trained": _get_braf().is_trained,
                "reference": "Liang et al. Radiology 2020",
            },
        ]
    }
