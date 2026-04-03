"""
ML-based Drug Synergy Prediction Service
Loads trained models for drug synergy prediction:
  - XGBoost variants: morgan, combined, fixAB, realexpr
  - DeepSynergy v3: PyTorch MLP (927K DrugComb, 4246 drugs)

Data: O'Neil et al. 2016 + DrugComb 1.17M
"""

import json
import logging
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Any

import numpy as np

logger = logging.getLogger(__name__)

MODEL_DIR = Path(BASE_DIR / "models/synergy")

# Tissue keywords for cell line features (DeepSynergy v3)
TISSUE_KEYWORDS = {
    'COLON': ['HCT','SW480','SW620','COLO','DLD','HT29','LOVO','RKO'],
    'BREAST': ['MCF','MDA','BT','T47D','SKBR'],
    'LUNG': ['A549','NCI','H460','H1299','H1975','HCC827','PC9'],
    'OVARIAN': ['SKOV','A2780','OVCAR','IGROV'],
    'LEUKEMIA': ['K562','HL60','MOLT','JURKAT','CCRF'],
    'MELANOMA': ['A375','SKMEL','MALME','UACC'],
    'PROSTATE': ['PC3','DU145','LNCAP'],
    'LIVER': ['HEPG2','HEP3B','HUH'],
    'BRAIN': ['U87','U251','SF','SNB','A172'],
    'PANCREAS': ['PANC','MIAPACA','BXPC'],
    'KIDNEY': ['786','A498','ACHN','CAKI'],
    'STOMACH': ['AGS','KATO','MKN','SNU'],
}


class MLSynergyService:
    """XGBoost-based drug synergy predictor with multiple model variants."""

    VARIANTS = ["morgan", "combined", "fixAB", "realexpr"]

    def __init__(self):
        self.models: Dict[str, Any] = {}
        self.metadata: Dict[str, Any] = {}
        self.drug_smiles: Dict[str, str] = {}
        self.drug_fingerprints: Dict[str, Any] = {}
        self.deep_synergy_model = None
        self.deep_synergy_scaler = None
        self._load_models()

    def _load_models(self):
        """Load all available XGBoost synergy model variants."""
        # Load drug SMILES mapping
        smiles_path = MODEL_DIR / "drug_smiles.json"
        if smiles_path.exists():
            with open(smiles_path) as f:
                self.drug_smiles = json.load(f)
            logger.info(f"Loaded {len(self.drug_smiles)} drug SMILES mappings")

        # Load model metadata
        for ver in ["v3", "v2", "v1"]:
            meta_path = MODEL_DIR / f"model_metadata_{ver}.json"
            if meta_path.exists():
                with open(meta_path) as f:
                    self.metadata = json.load(f)
                break

        if not self.metadata:
            meta_path = MODEL_DIR / "model_metadata.json"
            if meta_path.exists():
                with open(meta_path) as f:
                    self.metadata = json.load(f)

        # Load drug fingerprints
        fp_path = MODEL_DIR / "drug_fingerprints_morgan.pkl"
        if fp_path.exists():
            try:
                with open(fp_path, "rb") as f:
                    self.drug_fingerprints = pickle.load(f)
                logger.info(f"Loaded fingerprints for {len(self.drug_fingerprints)} drugs")
            except Exception as e:
                logger.warning(f"Failed to load fingerprints: {e}")

        # Load XGBoost models
        for variant in self.VARIANTS:
            pkl_path = MODEL_DIR / f"xgboost_synergy_{variant}.pkl"
            if pkl_path.exists():
                try:
                    with open(pkl_path, "rb") as f:
                        self.models[variant] = pickle.load(f)
                    logger.info(f"Loaded XGBoost synergy model: {variant}")
                except Exception as e:
                    logger.warning(f"Failed to load {variant}: {e}")

        # Also try JSON format
        for ver in ["v3", "v2", "v1"]:
            json_path = MODEL_DIR / f"xgb_synergy_{ver}.json"
            if json_path.exists() and f"json_{ver}" not in self.models:
                try:
                    import xgboost as xgb
                    model = xgb.XGBRegressor()
                    model.load_model(str(json_path))
                    self.models[f"json_{ver}"] = model
                    logger.info(f"Loaded XGBoost synergy JSON model: {ver}")
                except Exception as e:
                    logger.warning(f"Failed to load JSON model {ver}: {e}")

        # Load DeepSynergy v3 (PyTorch)
        self._load_deep_synergy()

        logger.info(f"Total synergy models loaded: {len(self.models)}")

    @property
    def available_models(self) -> List[str]:
        return list(self.models.keys())

    @property
    def available_drugs(self) -> List[str]:
        return list(self.drug_smiles.keys())

    def predict(
        self,
        drug_a: str,
        drug_b: str,
        cell_line: str = "HCT116",
        model_variant: str = "combined",
    ) -> Dict[str, Any]:
        """
        Predict synergy score for a drug pair using XGBoost.

        Args:
            drug_a: Drug A name
            drug_b: Drug B name
            cell_line: Cell line name
            model_variant: Which model variant to use

        Returns:
            Dict with synergy_score, classification, model details
        """
        if model_variant not in self.models:
            available = self.available_models
            if not available:
                return {
                    "status": "error",
                    "message": "No synergy models loaded",
                }
            model_variant = available[0]

        model = self.models[model_variant]

        # Build feature vector
        features = self._build_features(drug_a, drug_b, cell_line, model_variant)
        if features is None:
            return {
                "status": "error",
                "message": f"Cannot compute features for {drug_a} + {drug_b}",
            }

        try:
            x = np.array(features).reshape(1, -1)
            score = float(model.predict(x)[0])

            # Classification (based on Bliss model convention)
            if score > 5:
                classification = "synergistic"
            elif score < -5:
                classification = "antagonistic"
            else:
                classification = "additive"

            return {
                "status": "success",
                "drug_a": drug_a,
                "drug_b": drug_b,
                "cell_line": cell_line,
                "synergy_score": round(score, 2),
                "classification": classification,
                "model_variant": model_variant,
                "drug_a_known": drug_a in self.drug_smiles,
                "drug_b_known": drug_b in self.drug_smiles,
                "data_source": self.metadata.get("data_source", "O'Neil et al. 2016"),
            }
        except Exception as e:
            logger.error(f"Prediction failed: {e}", exc_info=True)
            return {"status": "error", "message": str(e)}

    def compare_models(
        self, drug_a: str, drug_b: str, cell_line: str = "HCT116"
    ) -> Dict[str, Any]:
        """Predict with all available model variants and compare."""
        results = {}
        for variant in self.models:
            result = self.predict(drug_a, drug_b, cell_line, variant)
            if result.get("status") == "success":
                results[variant] = {
                    "score": result["synergy_score"],
                    "classification": result["classification"],
                }

        if not results:
            return {"status": "error", "message": "No models could produce predictions"}

        scores = [r["score"] for r in results.values()]
        consensus_score = float(np.mean(scores))

        return {
            "status": "success",
            "drug_a": drug_a,
            "drug_b": drug_b,
            "cell_line": cell_line,
            "per_model": results,
            "consensus_score": round(consensus_score, 2),
            "consensus_classification": (
                "synergistic" if consensus_score > 5
                else "antagonistic" if consensus_score < -5
                else "additive"
            ),
            "model_agreement": len(set(r["classification"] for r in results.values())) == 1,
            "n_models": len(results),
        }

    def _build_features(
        self, drug_a: str, drug_b: str, cell_line: str, variant: str
    ) -> Optional[List[float]]:
        """Build feature vector for a drug pair."""
        # Get fingerprints for both drugs
        fp_a = self.drug_fingerprints.get(drug_a)
        fp_b = self.drug_fingerprints.get(drug_b)

        if fp_a is None or fp_b is None:
            # Try to generate from SMILES
            fp_a = fp_a or self._generate_fingerprint(drug_a)
            fp_b = fp_b or self._generate_fingerprint(drug_b)

        if fp_a is None or fp_b is None:
            return None

        # Concatenate drug features
        fp_a = np.array(fp_a, dtype=np.float32).flatten()
        fp_b = np.array(fp_b, dtype=np.float32).flatten()

        features = np.concatenate([fp_a, fp_b])
        return features.tolist()

    def _generate_fingerprint(self, drug_name: str) -> Optional[np.ndarray]:
        """Generate Morgan fingerprint from SMILES."""
        # Case-insensitive lookup
        smiles = (self.drug_smiles.get(drug_name)
                  or self.drug_smiles.get(drug_name.upper())
                  or self.drug_smiles.get(drug_name.lower())
                  or self.drug_smiles.get(drug_name.title()))
        if not smiles:
            return None

        try:
            from rdkit import Chem
            from rdkit.Chem import AllChem

            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return None

            fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=1024)
            return np.array(fp, dtype=np.float32)
        except ImportError:
            logger.warning("RDKit not available for fingerprint generation")
            return None

    # ================================================================
    # DeepSynergy v5 (PyTorch) — with CL expression embedding
    # ================================================================

    def _load_deep_synergy(self):
        """Load DeepSynergy v5 PyTorch model with CL expression."""
        # Try v5 first, fall back to v3
        model_path = MODEL_DIR / "deep_synergy_v5.pt"
        model_version = "v5"
        input_dim = 2316  # FP(2048) + CL_expr(256) + tissue(12)

        if not model_path.exists():
            model_path = MODEL_DIR / "deep_synergy_v3.pt"
            model_version = "v3"
            input_dim = 2060  # FP(2048) + tissue(12)

        if not model_path.exists():
            logger.info("DeepSynergy model not found, skipping")
            return

        try:
            import torch
            import torch.nn as nn

            hidden = [2048, 1024, 512, 256]

            # Build model
            layers = []
            prev = input_dim
            for i, h in enumerate(hidden):
                layers.extend([
                    nn.Linear(prev, h), nn.BatchNorm1d(h),
                    nn.GELU(), nn.Dropout(0.3 if i < 2 else 0.2)
                ])
                prev = h
            layers.append(nn.Linear(prev, 1))
            model = nn.Sequential(*layers)
            state_dict = torch.load(model_path, map_location="cpu", weights_only=True)
            # Strip 'net.' prefix if present (from wrapper class)
            cleaned = {}
            for k, v in state_dict.items():
                cleaned[k.replace("net.", "")] = v
            model.load_state_dict(cleaned)
            model.eval()

            self.deep_synergy_model = model
            self.deep_synergy_version = model_version
            self.models[f"deep_synergy_{model_version}"] = "pytorch"

            # Load CL expression embeddings for v5
            self.cl_expression_map = {}
            if model_version == "v5":
                cl_paths = [
                    Path(BASE_DIR / "data/ml_training/cell_line_expression_256_expanded_v2.pkl"),
                    Path(BASE_DIR / "data/ml_training/cell_line_expression_256_expanded.pkl"),
                ]
                for cp in cl_paths:
                    if cp.exists():
                        import pickle
                        with open(cp, 'rb') as f:
                            self.cl_expression_map = pickle.load(f)
                        logger.info("Loaded CL expression: %d entries from %s",
                                    len(self.cl_expression_map), cp.name)
                        break

            logger.info("Loaded DeepSynergy %s (PyTorch, %d params, dim=%d)",
                        model_version, sum(p.numel() for p in model.parameters()),
                        input_dim)

        except Exception as e:
            logger.warning(f"Failed to load DeepSynergy: {e}")

    def _get_cell_line_features(self, cell_line: str) -> np.ndarray:
        """Get tissue-type features for a cell line."""
        vec = np.zeros(len(TISSUE_KEYWORDS), dtype=np.float32)
        cl = cell_line.upper().replace('-','').replace('_','').replace(' ','')
        for i, (tissue, keywords) in enumerate(TISSUE_KEYWORDS.items()):
            for kw in keywords:
                if kw.replace('-','') in cl:
                    vec[i] = 1.0
                    break
        return vec

    def _get_cl_expression(self, cell_line: str) -> np.ndarray:
        """Get 256d CL expression embedding for a cell line."""
        if not self.cl_expression_map:
            return np.zeros(256, dtype=np.float32)
        cl_upper = cell_line.upper().strip()
        cl_clean = cl_upper.replace('-','').replace('_','').replace(' ','')
        expr = self.cl_expression_map.get(cl_upper)
        if expr is None:
            expr = self.cl_expression_map.get(cl_clean)
        if expr is not None:
            return np.array(expr, dtype=np.float32)
        return np.zeros(256, dtype=np.float32)

    def predict_deep_synergy(
        self, drug_a: str, drug_b: str, cell_line: str = "HCT116"
    ) -> Dict[str, Any]:
        """Predict synergy using DeepSynergy v5 PyTorch model."""
        if self.deep_synergy_model is None:
            return {"status": "error", "message": "DeepSynergy not loaded"}

        fp_a = self._generate_fingerprint(drug_a)
        fp_b = self._generate_fingerprint(drug_b)
        if fp_a is None or fp_b is None:
            return {
                "status": "error",
                "message": f"SMILES not found for {drug_a if fp_a is None else drug_b}",
            }

        try:
            import torch

            version = getattr(self, 'deep_synergy_version', 'v3')
            cl_tissue = self._get_cell_line_features(cell_line)

            if version == "v5":
                cl_expr = self._get_cl_expression(cell_line)
                features = np.concatenate([fp_a.flatten(), fp_b.flatten(), cl_expr, cl_tissue])
            else:
                features = np.concatenate([fp_a.flatten(), fp_b.flatten(), cl_tissue])

            x = torch.FloatTensor(features).unsqueeze(0)

            with torch.no_grad():
                score = float(self.deep_synergy_model(x).squeeze().item())

            classification = (
                "synergistic" if score > 5
                else "antagonistic" if score < -5
                else "additive"
            )

            return {
                "status": "success",
                "drug_a": drug_a,
                "drug_b": drug_b,
                "cell_line": cell_line,
                "synergy_score": round(score, 2),
                "classification": classification,
                "model": f"deep_synergy_{version}",
                "model_type": "PyTorch MLP",
                "training_data": "DrugComb 927K (4246 drugs)",
                "cl_expression_available": version == "v5" and bool(self.cl_expression_map),
                "drug_pair_cv_r": 0.7074 if version == "v5" else 0.6044,
            }
        except Exception as e:
            logger.error(f"DeepSynergy prediction failed: {e}", exc_info=True)
            return {"status": "error", "message": str(e)}

    def get_model_info(self) -> Dict[str, Any]:
        """Return model metadata."""

import os as _os
from pathlib import Path as _Path
# ADDS_BASE_DIR environment variable overrides automatic detection
BASE_DIR = _Path(_os.environ.get("ADDS_BASE_DIR", str(_Path(__file__).resolve().parent.parent)))

        info = {
            "available_models": self.available_models,
            "n_drugs": len(self.drug_smiles),
            "metadata": self.metadata,
            "data_source": self.metadata.get("data_source", "O'Neil et al. 2016"),
            "n_samples": self.metadata.get("n_samples", 23052),
        }
        if self.deep_synergy_model is not None:
            version = getattr(self, 'deep_synergy_version', 'v3')
            info[f"deep_synergy_{version}"] = {
                "status": "loaded",
                "training_samples": 927011,
                "unique_drugs": 4246,
                "drug_pair_cv_r": 0.7074 if version == "v5" else 0.6044,
                "cl_expression_entries": len(self.cl_expression_map) if hasattr(self, 'cl_expression_map') else 0,
                "features": "FP(2048) + CL_expr(256) + tissue(12)" if version == "v5" else "FP(2048) + tissue(12)",
            }
        return info
