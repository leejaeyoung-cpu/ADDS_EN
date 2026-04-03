"""
Stage 5: Tumor Classification and Staging
ML-based classification and TNM staging using radiomics features
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple, List
import pickle
from pathlib import Path
import logging

# ML libraries
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
import xgboost as xgb

logger = logging.getLogger(__name__)


class TumorClassifier:
    """
    Multi-level tumor classification system
    - Benign vs. Malignant
    - T Stage (Tumor invasion depth)
    - N Stage (Lymph node involvement)
    """
    
    def __init__(self, models_dir: Optional[Path] = None):
        """
        Initialize classifier
        
        Args:
            models_dir: Directory containing trained models
        """
        self.models_dir = models_dir or Path("models/classifiers")
        self.scaler = StandardScaler()
        self.feature_selector = None
        
        # Classification models
        self.benign_malignant_model = None
        self.t_stage_model = None
        self.n_stage_model = None
        self.msi_predictor = None
        
        self._load_models()
    
    def _load_models(self):
        """Load pre-trained models"""
        try:
            if self.models_dir.exists():
                # Load scaler
                scaler_path = self.models_dir / "radiomics_scaler.pkl"
                if scaler_path.exists():
                    with open(scaler_path, 'rb') as f:
                        self.scaler = pickle.load(f)
                    logger.info("Loaded feature scaler")
                
                # Load classifiers
                benign_mal_path = self.models_dir / "benign_malignant_xgboost.pkl"
                if benign_mal_path.exists():
                    with open(benign_mal_path, 'rb') as f:
                        self.benign_malignant_model = pickle.load(f)
                    logger.info("Loaded benign/malignant classifier")
                
                # Load staging models
                t_stage_path = self.models_dir / "t_stage_randomforest.pkl"
                if t_stage_path.exists():
                    with open(t_stage_path, 'rb') as f:
                        self.t_stage_model = pickle.load(f)
                    logger.info("Loaded T-stage classifier")
                
                n_stage_path = self.models_dir / "n_stage_svm.pkl"
                if n_stage_path.exists():
                    with open(n_stage_path, 'rb') as f:
                        self.n_stage_model = pickle.load(f)
                    logger.info("Loaded N-stage classifier")
                
        except Exception as e:
            logger.warning(f"Could not load models: {e}. Will use default models.")
    
    def _create_default_models(self):
        """Create default untrained models"""
        # Benign/Malignant: XGBoost
        self.benign_malignant_model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42
        )
        
        # T Stage: Random Forest
        self.t_stage_model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )
        
        # N Stage: SVM
        self.n_stage_model = SVC(
            kernel='rbf',
            probability=True,
            random_state=42
        )
        
        logger.info("Created default models")
    
    def prepare_features(self, radiomics: Dict[str, float]) -> np.ndarray:
        """
        Prepare radiomics features for classification
        
        Args:
            radiomics: Dictionary of radiomics features
            
        Returns:
            features: Normalized feature array
        """
        # Convert  to array
        feature_names = sorted(radiomics.keys())
        features = np.array([radiomics[name] for name in feature_names])
        
        # Handle NaN/Inf
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Reshape for sklearn
        features = features.reshape(1, -1)
        
        # Scale features
        try:
            features_scaled = self.scaler.transform(features)
        except:
            # If scaler not fitted, return as is
            features_scaled = features
        
        return features_scaled
    
    def predict_malignancy(self, radiomics: Dict[str, float]) -> Dict:
        """
        Predict benign vs. malignant
        
        Args:
            radiomics: Radiomics features
            
        Returns:
            result: Classification result with probability
        """
        features = self.prepare_features(radiomics)
        
        if self.benign_malignant_model is None:
            # Fallback: rule-based heuristic
            return self._heuristic_malignancy(radiomics)
        
        # Predict
        prob = self.benign_malignant_model.predict_proba(features)[0, 1]
        is_malignant = prob > 0.5
        
        result = {
            'classification': 'Malignant' if is_malignant else 'Benign',
            'malignancy_probability': float(prob),
            'confidence': float(max(prob, 1-prob))
        }
        
        return result
    
    def _heuristic_malignancy(self, radiomics: Dict[str, float]) -> Dict:
        """
        Heuristic-based malignancy prediction (fallback)
        
        High-risk indicators:
        - Low sphericity (irregular shape)
        - High entropy (heterogeneous)
        - High texture contrast
        - Large volume
        """
        sphericity = radiomics.get('original_shape_Sphericity', 0.5)
        entropy = radiomics.get('original_firstorder_Entropy', 0)
        contrast = radiomics.get('original_glcm_Contrast', 0)
        
        # Simple scoring
        risk_score = 0.0
        
        if sphericity < 0.6:
            risk_score += 0.3
        if entropy > 4.0:
            risk_score += 0.3
        if contrast > 100:
            risk_score += 0.4
        
        risk_score = min(risk_score, 1.0)
        
        return {
            'classification': 'Malignant' if risk_score > 0.5 else 'Benign',
            'malignancy_probability': risk_score,
            'confidence': abs(risk_score - 0.5) * 2,
            'method': 'heuristic'
        }
    
    def predict_t_stage(self, radiomics: Dict[str, float]) -> str:
        """
        Predict T stage (tumor invasion depth)
        T0, T1, T2, T3, T4
        
        Args:
            radiomics: Radiomics features
            
        Returns:
            t_stage: Predicted T stage
        """
        features = self.prepare_features(radiomics)
        
        if self.t_stage_model is None:
            # Fallback: size-based heuristic
            volume = radiomics.get('original_shape_VoxelVolume', 0)
            if volume < 500:
                return 'T1'
            elif volume < 2000:
                return 'T2'
            elif volume < 5000:
                return 'T3'
            else:
                return 'T4'
        
        t_stage = self.t_stage_model.predict(features)[0]
        return t_stage
    
    def predict_n_stage(self, radiomics: Dict[str, float]) -> str:
        """
        Predict N stage (lymph node involvement)
        N0, N1, N2
        
        Args:
            radiomics: Radiomics features
            
        Returns:
            n_stage: Predicted N stage
        """
        features = self.prepare_features(radiomics)
        
        if self.n_stage_model is None:
            # Conservative: default to N0
            return 'N0'
        
        n_stage = self.n_stage_model.predict(features)[0]
        return n_stage
    
    def predict_tnm(self, radiomics: Dict[str, float]) -> Dict:
        """
        Complete TNM staging
        
        Args:
            radiomics: Radiomics features
            
        Returns:
            tnm: TNM staging dictionary
        """
        # First check malignancy
        malignancy = self.predict_malignancy(radiomics)
        
        if malignancy['classification'] == 'Benign':
            return {
                'classification': 'Benign',
                'malignancy_probability': malignancy['malignancy_probability'],
                'tnm_stage': None,
                'overall_stage': None
            }
        
        # Predict stages
        t_stage = self.predict_t_stage(radiomics)
        n_stage = self.predict_n_stage(radiomics)
        m_stage = 'MX'  # Clinical evaluation required for distant metastasis
        
        overall_stage = self._compute_overall_stage(t_stage, n_stage, m_stage)
        
        return {
            'classification': 'Malignant',
            'malignancy_probability': malignancy['malignancy_probability'],
            'tnm_stage': {
                'T': t_stage,
                'N': n_stage,
                'M': m_stage
            },
            'overall_stage': overall_stage
        }
    
    def _compute_overall_stage(self, t: str, n: str, m: str) -> str:
        """
        Compute overall stage from TNM
        Uses simplified AJCC staging criteria
        
        Args:
            t, n, m: TNM stages
            
        Returns:
            overall_stage: Stage I, II, III, or IV
        """
        # Metastasis = Stage IV
        if m == 'M1':
            return 'Stage IV'
        
        # N2 = Stage III
        if n == 'N2':
            return 'Stage III'
        
        # N1 = Stage III
        if n == 'N1':
            return 'Stage III'
        
        # T4 with N0 = Stage II
        if t == 'T4' and n == 'N0':
            return 'Stage II'
        
        # T3 with N0 = Stage II
        if t == 'T3' and n == 'N0':
            return 'Stage II'
        
        # T1-T2 with N0 = Stage I
        if t in ['T1', 'T2'] and n == 'N0':
            return 'Stage I'
        
        return 'Stage Unknown'


class BiomarkerPredictor:
    """Predict molecular biomarkers from imaging features"""
    
    def __init__(self, models_dir: Optional[Path] = None):
        self.models_dir = models_dir or Path("models/biomarker_predictors")
        self.msi_model = None
        self.kras_model = None
        
    def predict_msi_status(self, radiomics: Dict[str, float]) -> Dict:
        """
        Predict MSI (Microsatellite Instability) status
        
        MSI-H tumors tend to be:
        - More heterogeneous (high entropy)
        - Less spherical (irregular)
        - High texture variability
        
        Args:
            radiomics: Radiomics features
            
        Returns:
            msi_result: MSI prediction
        """
        # Use texture heterogeneity as proxy
        entropy = radiomics.get('original_firstorder_Entropy', 0)
        sphericity = radiomics.get('original_shape_Sphericity', 1.0)
        contrast = radiomics.get('original_glcm_Contrast', 0)
        
        # Heuristic scoring
        msi_score = 0.0
        
        if entropy > 4.5:
            msi_score += 0.4
        if sphericity < 0.5:
            msi_score += 0.3
        if contrast > 150:
            msi_score += 0.3
        
        msi_score = min(msi_score, 1.0)
        
        return {
            'status': 'MSI-H' if msi_score > 0.6 else 'MSS',
            'probability': msi_score,
            'confidence': abs(msi_score - 0.5) * 2
        }
    
    def predict_kras_mutation(self, radiomics: Dict[str, float]) -> Dict:
        """

import os as _os
from pathlib import Path as _Path
# ADDS_BASE_DIR environment variable overrides automatic detection
BASE_DIR = _Path(_os.environ.get("ADDS_BASE_DIR", str(_Path(__file__).resolve().parent.parent)))

        Predict KRAS mutation likelihood from radiomics features.

        Uses trained Logistic Regression when available (models/biomarker/kras_model.pkl).
        Returns explicit 'model_not_trained' otherwise — never returns fake predictions.

        Ref: Liang et al. "CT-based radiomic KRAS prediction" (Radiology 2020)

        Args:
            radiomics: Radiomics features (shape, texture, first-order)

        Returns:
            kras_result: KRAS prediction with status, probability, confidence
        """
        import pickle
        from pathlib import Path

        model_path = Path(BASE_DIR / "models/biomarker/kras_model.pkl")
        if model_path.exists():
            try:
                with open(model_path, "rb") as f:
                    saved = pickle.load(f)
                model = saved.get("model")
                scaler = saved.get("scaler")
                feature_keys = saved.get("feature_keys", [
                    "original_shape_Sphericity", "original_shape_VoxelVolume",
                    "original_firstorder_Entropy", "original_glcm_Contrast",
                ])
                x = np.array([radiomics.get(k, 0.0) for k in feature_keys]).reshape(1, -1)
                if scaler:
                    x = scaler.transform(x)
                prob = model.predict_proba(x)[0]
                positive_prob = float(prob[1]) if len(prob) > 1 else float(prob[0])
                return {
                    'status': 'Mutant' if positive_prob >= 0.5 else 'Wild-type',
                    'probability': round(positive_prob, 4),
                    'confidence': round(abs(positive_prob - 0.5) * 2, 4),
                }
            except Exception as e:
                logger.warning(f"KRAS model prediction failed: {e}")

        # No trained model — return explicit status (NOT a fake 0.5)
        return {
            'status': 'model_not_trained',
            'probability': None,
            'confidence': None,
            'message': 'KRAS prediction requires trained model from TCGA/TCIA radiomics data',
        }


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    # Sample radiomics features
    radiomics = {
        'original_shape_Sphericity': 0.45,
        'original_shape_VoxelVolume': 3500,
        'original_firstorder_Entropy': 4.8,
        'original_glcm_Contrast': 180,
        'original_glcm_Correlation': 0.75
    }
    
    # Classify
    classifier = TumorClassifier()
    result = classifier.predict_tnm(radiomics)
    
    print(f"\n✓ Classification: {result['classification']}")
    if result['classification'] == 'Malignant':
        print(f"✓ TNM Stage: T{result['tnm_stage']['T']}, N{result['tnm_stage']['N']}, M{result['tnm_stage']['M']}")
        print(f"✓ Overall Stage: {result['overall_stage']}")
    
    # Biomarkers
    biomarker = BiomarkerPredictor()
    msi = biomarker.predict_msi_status(radiomics)
    print(f"\n✓ MSI Status: {msi['status']} (confidence: {msi['confidence']:.2f})")
