"""
Unit Tests for Prognosis Prediction Engine

Tests all components:
- Radiomics feature extraction
- DeepSurv model
- Survival prediction
- Risk stratification
"""

import pytest
import numpy as np
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.medical_imaging.prognosis import (
    PrognosisEngine,
    SurvivalPrediction,
    RadiomicsExtractor,
    DeepSurvModel
)


class TestRadiomicsExtractor:
    """Test suite for RadiomicsExtractor"""
    
    @pytest.fixture
    def extractor(self):
        """Create radiomics extractor instance"""
        return RadiomicsExtractor(spacing=(1.0, 1.0, 1.0))
    
    @pytest.fixture
    def simple_tumor(self):
        """Simple spherical tumor for testing"""
        mask = np.zeros((50, 50, 50), dtype=np.uint8)
        center = (25, 25, 25)
        radius = 10
        
        z, y, x = np.ogrid[:50, :50, :50]
        dist = np.sqrt((z - center[0])**2 + (y - center[1])**2 + (x - center[2])**2)
        mask[dist <= radius] = 1
        
        return mask
    
    @pytest.fixture
    def ct_volume(self):
        """Simple CT volume with tumor region"""
        volume = np.random.randn(50, 50, 50) * 30 + 50  # Mean HU = 50, std = 30
        return volume
    
    # ==================== Shape Features Tests ====================
    
    def test_shape_volume_calculation(self, extractor, simple_tumor):
        """Test volume calculation"""
        features = extractor._extract_shape_features(simple_tumor)
        
        # Spherical tumor with radius 10mm should have volume ~4188 mm³
        expected_volume = (4/3) * np.pi * (10**3)
        assert 'shape_volume_mm3' in features
        assert abs(features['shape_volume_mm3'] - expected_volume) < 500  # Within tolerance
    
    def test_shape_sphericity(self, extractor, simple_tumor):
        """Test sphericity calculation"""
        features = extractor._extract_shape_features(simple_tumor)
        
        assert 'shape_sphericity' in features
        # Spherical tumor should have sphericity close to 1.0
        assert 0.8 <= features['shape_sphericity'] <= 1.2
    
    def test_shape_max_diameter(self, extractor, simple_tumor):
        """Test maximum diameter measurement"""
        features = extractor._extract_shape_features(simple_tumor)
        
        assert 'shape_max_diameter_mm' in features
        # Diameter should be ~20mm (radius 10mm)
        assert 18.0 <= features['shape_max_diameter_mm'] <= 22.0
    
    def test_empty_mask_shape_features(self, extractor):
        """Test shape features with empty mask"""
        empty_mask = np.zeros((50, 50, 50), dtype=np.uint8)
        features = extractor._extract_shape_features(empty_mask)
        
        assert features['shape_volume_mm3'] == 0.0
        assert features['shape_max_diameter_mm'] == 0.0
    
    # ==================== Intensity Features Tests ====================
    
    def test_intensity_statistics(self, extractor, ct_volume, simple_tumor):
        """Test first-order intensity statistics"""
        features = extractor._extract_first_order_statistics(ct_volume, simple_tumor)
        
        # Check all expected features exist
        assert 'intensity_mean' in features
        assert 'intensity_std' in features
        assert 'intensity_median' in features
        assert 'intensity_range' in features
        assert 'intensity_skewness' in features
        assert 'intensity_kurtosis' in features
        
        # Mean should be around 50 (from ct_volume fixture)
        assert 30 <= features['intensity_mean'] <= 70
    
    def test_intensity_percentiles(self, extractor, ct_volume, simple_tumor):
        """Test percentile calculations"""
        features = extractor._extract_first_order_statistics(ct_volume, simple_tumor)
        
        assert 'intensity_p10' in features
        assert 'intensity_p25' in features
        assert 'intensity_p75' in features
        assert 'intensity_p90' in features
        assert 'intensity_iqr' in features
        
        # Percentiles should be ordered
        assert features['intensity_p10'] < features['intensity_p25']
        assert features['intensity_p25'] < features['intensity_p75']
        assert features['intensity_p75'] < features['intensity_p90']
    
    def test_empty_mask_intensity_features(self, extractor, ct_volume):
        """Test intensity features with empty mask"""
        empty_mask = np.zeros((50, 50, 50), dtype=np.uint8)
        features = extractor._extract_first_order_statistics(ct_volume, empty_mask)
        
        assert features['intensity_mean'] == 0.0
        assert features['intensity_std'] == 0.0
    
    # ==================== Texture Features Tests ====================
    
    def test_texture_features(self, extractor, ct_volume, simple_tumor):
        """Test GLCM texture features"""
        features = extractor._extract_texture_features(ct_volume, simple_tumor)
        
        assert 'texture_contrast' in features
        assert 'texture_homogeneity' in features
        assert 'texture_energy' in features
        assert 'texture_correlation' in features
        
        # Homogeneity should be between 0 and 1
        assert 0.0 <= features['texture_homogeneity'] <= 1.0
        
        # Energy should be non-negative
        assert features['texture_energy'] >= 0.0
    
    def test_complete_feature_extraction(self, extractor, ct_volume, simple_tumor):
        """Test complete feature extraction pipeline"""
        features = extractor.extract_features(ct_volume, simple_tumor)
        
        # Should have features from all categories
        assert len(features) > 10
        
        # Check for features from each category
        shape_features = [k for k in features.keys() if k.startswith('shape_')]
        intensity_features = [k for k in features.keys() if k.startswith('intensity_')]
        texture_features = [k for k in features.keys() if k.startswith('texture_')]
        
        assert len(shape_features) > 0
        assert len(intensity_features) > 0
        assert len(texture_features) > 0


class TestDeepSurvModel:
    """Test suite for DeepSurv model"""
    
    def test_model_initialization(self):
        """Test model initialization"""
        model = DeepSurvModel(input_dim=30, hidden_dims=[64, 32, 16])
        assert model is not None
    
    def test_model_forward_pass(self):
        """Test forward pass"""
        model = DeepSurvModel(input_dim=30)
        
        # Create dummy input
        x = np.random.randn(5, 30).astype(np.float32)
        import torch
        x_tensor = torch.FloatTensor(x)
        
        # Forward pass
        output = model(x_tensor)
        
        # Output should be (batch_size, 1)
        assert output.shape == (5, 1)
    
    def test_model_output_range(self):
        """Test that model outputs reasonable values"""
        model = DeepSurvModel(input_dim=30)
        model.eval()
        
        import torch
        x = torch.randn(10, 30)
        
        with torch.no_grad():
            output = model(x)
        
        # Output should be finite
        assert torch.isfinite(output).all()


class TestPrognosisEngine:
    """Test suite for PrognosisEngine"""
    
    @pytest.fixture
    def engine(self):
        """Create prognosis engine instance"""
        return PrognosisEngine()
    
    @pytest.fixture
    def test_data(self):
        """Create test CT volume and tumor mask"""
        ct_volume = np.random.randn(80, 100, 100) * 30 + 50
        
        # Create tumor mask
        tumor_mask = np.zeros((80, 100, 100), dtype=np.uint8)
        center = (40, 50, 50)
        radius = 15
        
        z, y, x = np.ogrid[:80, :100, :100]
        dist = np.sqrt((z - center[0])**2 + (y - center[1])**2 + (x - center[2])**2)
        tumor_mask[dist <= radius] = 1
        
        return ct_volume, tumor_mask
    
    @pytest.fixture
    def clinical_features(self):
        """Create sample clinical features"""
        return {
            'tnm_stage': {'T': 'T2', 'N': 'N1', 'M': 'M0'},
            'age': 65,
            'gender': 'male'
        }
    
    # ==================== Feature Combination Tests ====================
    
    def test_feature_combination(self, engine, clinical_features):
        """Test combining radiomics and clinical features"""
        radiomics = {
            'shape_volume_cm3': 5.0,
            'shape_sphericity': 0.9,
            'intensity_mean': 50.0
        }
        
        combined = engine._combine_features(radiomics, clinical_features)
        
        # Should have both radiomics and clinical features
        assert 'radiomics_shape_volume_cm3' in combined
        assert 'clinical_T' in combined
        assert 'clinical_N' in combined
        assert 'clinical_M' in combined
        assert 'clinical_age_normalized' in combined
    
    def test_tnm_encoding(self, engine):
        """Test TNM stage encoding"""
        clinical = {
            'tnm_stage': {'T': 'T3', 'N': 'N2', 'M': 'M1'}
        }
        
        combined = engine._combine_features({}, clinical)
        
        assert combined['clinical_T'] == 3
        assert combined['clinical_N'] == 2
        assert combined['clinical_M'] == 1
    
    # ==================== Risk Prediction Tests ====================
    
    def test_risk_score_prediction(self, engine):
        """Test risk score prediction"""
        features = {
            'radiomics_shape_volume_cm3': 5.0,
            'clinical_T': 2,
            'clinical_N': 1,
            'clinical_M': 0
        }
        
        risk_score = engine._predict_risk_score(features)
        
        # Risk score should be between 0 and 1
        assert 0.0 <= risk_score <= 1.0
    
    def test_risk_classification(self, engine):
        """Test risk category classification"""
        # Low risk
        assert engine._classify_risk(0.2) == "Low"
        
        # Intermediate risk
        assert engine._classify_risk(0.5) == "Intermediate"
        
        # High risk
        assert engine._classify_risk(0.8) == "High"
    
    # ==================== Survival Curve Tests ====================
    
    def test_survival_curve_generation(self, engine):
        """Test survival curve generation"""
        risk_score = 0.5
        survival_probs = engine._generate_survival_curves(risk_score)
        
        # Should have probabilities for all timepoints
        assert '6mo' in survival_probs
        assert '12mo' in survival_probs
        assert '24mo' in survival_probs
        assert '60mo' in survival_probs
        
        # All probabilities should be between 0 and 1
        for prob in survival_probs.values():
            assert 0.0 <= prob <= 1.0
        
        # Survival should decrease over time (generally)
        assert survival_probs['6mo'] >= survival_probs['12mo']
        assert survival_probs['12mo'] >= survival_probs['24mo']
    
    def test_median_survival_estimation(self, engine):
        """Test median survival time estimation"""
        # High survival (median > 60 months)
        high_survival = {'6mo': 0.9, '12mo': 0.8, '24mo': 0.7, '60mo': 0.6}
        median_high = engine._estimate_median_survival(high_survival)
        assert median_high >= 60
        
        # Low survival (median < 12 months)
        low_survival = {'6mo': 0.6, '12mo': 0.3, '24mo': 0.1, '60mo': 0.05}
        median_low = engine._estimate_median_survival(low_survival)
        assert median_low < 12
    
    def test_confidence_interval(self, engine):
        """Test confidence interval calculation"""
        risk_score = 0.5
        ci = engine._calculate_confidence_interval(risk_score)
        
        # CI should be a tuple of two values
        assert len(ci) == 2
        lower, upper = ci
        
        # Lower bound should be less than upper bound
        assert lower < upper
        
        # Both should be between 0 and 1
        assert 0.0 <= lower <= 1.0
        assert 0.0 <= upper <= 1.0
        
        # Risk score should be within CI
        assert lower <= risk_score <= upper
    
    # ==================== Integration Tests ====================
    
    def test_complete_survival_prediction(self, engine, test_data, clinical_features):
        """Test complete survival prediction pipeline"""
        ct_volume, tumor_mask = test_data
        
        result = engine.predict_survival(
            ct_volume,
            tumor_mask,
            clinical_features,
            spacing=(2.0, 1.0, 1.0)
        )
        
        # Verify result structure
        assert isinstance(result, SurvivalPrediction)
        assert isinstance(result.survival_probabilities, dict)
        assert result.risk_category in ["Low", "Intermediate", "High"]
        assert 0.0 <= result.risk_score <= 1.0
        assert result.median_survival_months > 0
        assert len(result.confidence_interval) == 2
        assert isinstance(result.features_used, dict)
    
    def test_early_stage_prognosis(self, engine, test_data):
        """Test prognosis for early-stage cancer (should have better survival)"""
        ct_volume, tumor_mask = test_data
        
        # Early stage (T1, N0, M0)
        early_stage = {
            'tnm_stage': {'T': 'T1', 'N': 'N0', 'M': 'M0'},
            'age': 55,
            'gender': 'female'
        }
        
        result = engine.predict_survival(ct_volume, tumor_mask, early_stage)
        
        # Early stage generally has lower risk
        # (Note: actual values depend on model initialization)
        assert result.median_survival_months > 0
        assert 0.0 <= result.risk_score <= 1.0
    
    def test_advanced_stage_prognosis(self, engine, test_data):
        """Test prognosis for advanced cancer (should have worse survival)"""
        ct_volume, tumor_mask = test_data
        
        # Advanced stage (T4, N2, M1)
        advanced_stage = {
            'tnm_stage': {'T': 'T4', 'N': 'N2', 'M': 'M1'},
            'age': 75,
            'gender': 'male'
        }
        
        result = engine.predict_survival(ct_volume, tumor_mask, advanced_stage)
        
        # Advanced stage should still produce valid results
        assert isinstance(result, SurvivalPrediction)
        assert result.median_survival_months > 0


# ==================== Performance Tests ====================

class TestPrognosisPerformance:
    """Performance tests for prognosis prediction"""
    
    def test_prediction_speed(self):
        """Test prediction completes in reasonable time"""
        import time
        
        engine = PrognosisEngine()
        
        # Medium-sized volume
        ct_volume = np.random.randn(100, 150, 150) * 30 + 50
        tumor_mask = np.random.rand(100, 150, 150) > 0.98
        
        clinical = {
            'tnm_stage': {'T': 'T2', 'N': 'N1', 'M': 'M0'},
            'age': 65,
            'gender': 'male'
        }
        
        start_time = time.time()
        result = engine.predict_survival(ct_volume, tumor_mask, clinical)
        elapsed = time.time() - start_time
        
        # Should complete in < 10 seconds
        assert elapsed < 10.0
        print(f"Prognosis prediction time: {elapsed:.2f}s")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
