"""
Treatment Response Prediction Service
Loads trained XGBoost model (GSE39582) for CRC treatment response prediction.

Data: GSE39582 — 211 CRC chemo patients, 145 features
Label: responded (1=no relapse at 3yr, 0=relapsed)
CV: AUC 0.64 ± 0.08
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any

import numpy as np

logger = logging.getLogger(__name__)

MODEL_DIR = Path(BASE_DIR / "models/treatment_response")
SYNERGY_DIR = Path(BASE_DIR / "models/synergy")


class TreatmentResponsePredictor:
    """XGBoost-based CRC treatment response predictor."""

    def __init__(self):
        self.model = None
        self.metadata = None
        self.feature_names: List[str] = []
        self._load_model()

    def _load_model(self):
        """Load the best available treatment response model."""
        # Try XGBoost JSON format first (treatment_response dir)
        json_path = MODEL_DIR / "xgb_treatment_response.json"
        meta_path = MODEL_DIR / "model_metadata.json"

        if meta_path.exists():
            with open(meta_path) as f:
                self.metadata = json.load(f)
            self.feature_names = self.metadata.get("feature_names", [])

        if json_path.exists():
            try:
                import xgboost as xgb
                self.model = xgb.XGBClassifier()
                self.model.load_model(str(json_path))
                logger.info(
                    f"Treatment response model loaded: {len(self.feature_names)} features, "
                    f"AUC={self.metadata.get('cv_results', {}).get('auc', 'N/A')}"
                )
                return
            except Exception as e:
                logger.warning(f"Failed to load XGBoost JSON model: {e}")

        # Fallback: try pkl in synergy dir (v5 is latest)
        for ver in ["v5", "v4", "v3", "v2"]:
            pkl_path = SYNERGY_DIR / f"treatment_response_{ver}.pkl"
            if pkl_path.exists():
                try:
                    import pickle
                    with open(pkl_path, "rb") as f:
                        self.model = pickle.load(f)
                    logger.info(f"Treatment response model loaded from {pkl_path}")
                    return
                except Exception as e:
                    logger.warning(f"Failed to load {pkl_path}: {e}")

        logger.warning("No treatment response model found")

    @property
    def is_available(self) -> bool:
        return self.model is not None

    def predict(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """
        Predict treatment response probability.

        Args:
            features: Dict with clinical + genomic features.
                Required keys (clinical): stage, kras_mut, braf_mut, tp53_mut,
                    mmr_deficient, proximal, male, age, chemo_5fu, chemo_folfox, chemo_folfiri
                Optional keys (subtypes): subtype_C1..C6
                Optional keys (gene probes): Affymetrix probe IDs

        Returns:
            Dict with response_probability, classification, confidence, features_used
        """
        if not self.is_available:
            return {
                "status": "model_not_available",
                "message": "Treatment response model not loaded",
            }

        # Build feature vector in correct order
        x = np.zeros(len(self.feature_names))
        features_found = 0
        for i, fname in enumerate(self.feature_names):
            if fname in features:
                x[i] = float(features[fname])
                features_found += 1

        if features_found == 0:
            return {
                "status": "error",
                "message": f"No matching features. Expected: {self.feature_names[:10]}...",
            }

        # Predict
        try:
            x_2d = x.reshape(1, -1)
            prob = self.model.predict_proba(x_2d)[0]
            pred_class = int(self.model.predict(x_2d)[0])

            response_prob = float(prob[1]) if len(prob) > 1 else float(prob[0])

            # Classification
            if response_prob >= 0.6:
                classification = "likely_responder"
            elif response_prob <= 0.4:
                classification = "likely_non_responder"
            else:
                classification = "uncertain"

            # Confidence
            confidence = abs(response_prob - 0.5) * 2  # 0 at 0.5, 1 at 0 or 1

            return {
                "status": "success",
                "response_probability": round(response_prob, 4),
                "predicted_class": pred_class,
                "classification": classification,
                "confidence": round(confidence, 4),
                "features_used": features_found,
                "total_features": len(self.feature_names),
                "model_info": {
                    "data_source": self.metadata.get("description", "GSE39582"),
                    "n_samples": self.metadata.get("n_samples", 211),
                    "cv_auc": self.metadata.get("cv_results", {}).get("auc", "N/A"),
                    "label": self.metadata.get("label", "3yr relapse-free"),
                },
            }
        except Exception as e:
            logger.error(f"Prediction failed: {e}", exc_info=True)
            return {"status": "error", "message": str(e)}

    def get_feature_importance(self, top_n: int = 20) -> List[Dict[str, Any]]:
        """Get feature importance from XGBoost model."""
        if not self.is_available:
            return []

        try:
            importances = self.model.feature_importances_
            indices = np.argsort(importances)[::-1][:top_n]
            return [
                {
                    "feature": self.feature_names[i] if i < len(self.feature_names) else f"f{i}",
                    "importance": round(float(importances[i]), 4),
                    "rank": rank + 1,
                }
                for rank, i in enumerate(indices)
            ]
        except Exception:
            return []

    def get_model_info(self) -> Dict[str, Any]:
        """Return model metadata."""

import os as _os
from pathlib import Path as _Path
# ADDS_BASE_DIR environment variable overrides automatic detection
BASE_DIR = _Path(_os.environ.get("ADDS_BASE_DIR", str(_Path(__file__).resolve().parent.parent)))

        return {
            "available": self.is_available,
            "metadata": self.metadata,
            "n_features": len(self.feature_names),
            "feature_names": self.feature_names,
        }
