import pytest
import numpy as np
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.medical_imaging.prognosis import (
    PrognosisEngine, SurvivalPrediction, RadiomicsExtractor, DeepSurvModel
)

class TestRadiomicsExtractor:
    @pytest.fixture
    def extractor(self):
        return RadiomicsExtractor(spacing=(1.0, 1.0, 1.0))
    @pytest.fixture
    def simple_tumor(self):
        mask = np.zeros((50, 50, 50), dtype=np.uint8)
        z, y, x = np.ogrid[:50, :50, :50]
        dist = np.sqrt((z-25)**2 + (y-25)**2 + (x-25)**2)
        mask[dist <= 10] = 1
        return mask
    @pytest.fixture
    def ct_volume(self):
        return np.random.randn(50, 50, 50) * 30 + 50
    def test_shape_volume_calculation(self, extractor, simple_tumor):
        features = extractor._extract_shape_features(simple_tumor)
        expected_volume = (4/3) * np.pi * (10**3)
        assert 'shape_volume_mm3' in features
        assert abs(features['shape_volume_mm3'] - expected_volume) < 500
    def test_shape_sphericity(self, extractor, simple_tumor):
        features = extractor._extract_shape_features(simple_tumor)
        assert 'shape_sphericity' in features
        assert features['shape_sphericity'] > 0
    def test_shape_max_diameter(self, extractor, simple_tumor):
        features = extractor._extract_shape_features(simple_tumor)
        assert 'shape_max_diameter_mm' in features
        assert 14.0 <= features['shape_max_diameter_mm'] <= 24.0
    def test_empty_mask_shape_features(self, extractor):
        empty_mask = np.zeros((50, 50, 50), dtype=np.uint8)
        features = extractor._extract_shape_features(empty_mask)
        assert features['shape_volume_mm3'] == 0.0
        assert features['shape_max_diameter_mm'] == 0.0
    def test_intensity_statistics(self, extractor, ct_volume, simple_tumor):
        features = extractor._extract_first_order_statistics(ct_volume, simple_tumor)
        assert 'intensity_mean' in features
        assert 'intensity_std' in features
        assert 'intensity_median' in features
        assert 30 <= features['intensity_mean'] <= 70
    def test_intensity_percentiles(self, extractor, ct_volume, simple_tumor):
        features = extractor._extract_first_order_statistics(ct_volume, simple_tumor)
        assert features['intensity_p10'] < features['intensity_p25']
        assert features['intensity_p25'] < features['intensity_p75']
        assert features['intensity_p75'] < features['intensity_p90']
    def test_empty_mask_intensity_features(self, extractor, ct_volume):
        empty_mask = np.zeros((50, 50, 50), dtype=np.uint8)
        features = extractor._extract_first_order_statistics(ct_volume, empty_mask)
        assert features['intensity_mean'] == 0.0
    def test_texture_features(self, extractor, ct_volume, simple_tumor):
        features = extractor._extract_texture_features(ct_volume, simple_tumor)
        assert 'texture_contrast' in features
        assert 'texture_homogeneity' in features
        assert 0.0 <= features['texture_homogeneity'] <= 1.0
    def test_complete_feature_extraction(self, extractor, ct_volume, simple_tumor):
        features = extractor.extract_features(ct_volume, simple_tumor)
        assert len(features) > 10
        assert len([k for k in features if k.startswith('shape_')]) > 0
        assert len([k for k in features if k.startswith('intensity_')]) > 0
        assert len([k for k in features if k.startswith('texture_')]) > 0

class TestDeepSurvModel:
    def test_model_initialization(self):
        model = DeepSurvModel(input_dim=30, hidden_dims=[64, 32, 16])
        assert model is not None
    def test_model_forward_pass(self):
        model = DeepSurvModel(input_dim=30)
        x = np.random.randn(5, 30).astype(np.float32)
        import torch
        output = model(torch.FloatTensor(x))
        assert output.shape == (5, 1)
    def test_model_output_range(self):
        import torch
        model = DeepSurvModel(input_dim=30)
        model.eval()
        with torch.no_grad():
            output = model(torch.randn(10, 30))
        assert torch.isfinite(output).all()

class TestPrognosisEngine:
    @pytest.fixture
    def engine(self):
        return PrognosisEngine()
    @pytest.fixture
    def test_data(self):
        ct_volume = np.random.randn(80, 100, 100) * 30 + 50
        tumor_mask = np.zeros((80, 100, 100), dtype=np.uint8)
        z, y, x = np.ogrid[:80, :100, :100]
        dist = np.sqrt((z-40)**2 + (y-50)**2 + (x-50)**2)
        tumor_mask[dist <= 15] = 1
        return ct_volume, tumor_mask
    @pytest.fixture
    def clinical_features(self):
        return {'tnm_stage': {'T': 'T2', 'N': 'N1', 'M': 'M0'}, 'age': 65, 'gender': 'male'}
    def test_feature_combination(self, engine, clinical_features):
        radiomics = {'shape_volume_cm3': 5.0, 'shape_sphericity': 0.9, 'intensity_mean': 50.0}
        combined = engine._combine_features(radiomics, clinical_features)
        assert 'radiomics_shape_volume_cm3' in combined
        assert 'clinical_T' in combined
        assert 'clinical_age_normalized' in combined
    def test_tnm_encoding(self, engine):
        clinical = {'tnm_stage': {'T': 'T3', 'N': 'N2', 'M': 'M1'}}
        combined = engine._combine_features({}, clinical)
        assert combined['clinical_T'] == 3
        assert combined['clinical_N'] == 2
        assert combined['clinical_M'] == 1
    def test_risk_score_prediction(self, engine):
        features = {'radiomics_shape_volume_cm3': 5.0, 'clinical_T': 2, 'clinical_N': 1, 'clinical_M': 0}
        risk_score = engine._predict_risk_score(features)
        assert 0.0 <= risk_score <= 1.0
    def test_risk_classification(self, engine):
        assert engine._classify_risk(0.2) == "Low"
        assert engine._classify_risk(0.5) == "Intermediate"
        assert engine._classify_risk(0.8) == "High"
    def test_survival_curve_generation(self, engine):
        survival_probs = engine._generate_survival_curves(0.5)
        assert '6mo' in survival_probs
        assert '12mo' in survival_probs
        assert '24mo' in survival_probs
        assert '60mo' in survival_probs
        for prob in survival_probs.values():
            assert 0.0 <= prob <= 1.0
        assert survival_probs['6mo'] >= survival_probs['12mo']
        assert survival_probs['12mo'] >= survival_probs['24mo']
    def test_median_survival_estimation(self, engine):
        # Fix: all probs > 0.5, so median survival > max timepoint (60)
        high_survival = {'6mo': 0.95, '12mo': 0.90, '24mo': 0.85, '60mo': 0.80}
        median_high = engine._estimate_median_survival(high_survival)
        assert median_high >= 60  # All probs > 0.5, so median > 60
        low_survival = {'6mo': 0.3, '12mo': 0.1, '24mo': 0.05, '60mo': 0.01}
        median_low = engine._estimate_median_survival(low_survival)
        assert median_low < 12
    def test_confidence_interval(self, engine):
        ci = engine._calculate_confidence_interval(0.5)
        assert len(ci) == 2
        lower, upper = ci
        assert lower < upper
        assert 0.0 <= lower <= 1.0
        assert 0.0 <= upper <= 1.0
        assert lower <= 0.5 <= upper
    def test_complete_survival_prediction(self, engine, test_data, clinical_features):
        ct_volume, tumor_mask = test_data
        result = engine.predict_survival(ct_volume, tumor_mask, clinical_features, spacing=(2.0, 1.0, 1.0))
        assert isinstance(result, SurvivalPrediction)
        assert result.risk_category in ["Low", "Intermediate", "High"]
        assert 0.0 <= result.risk_score <= 1.0
        assert result.median_survival_months > 0
    def test_early_stage_prognosis(self, engine, test_data):
        ct_volume, tumor_mask = test_data
        result = engine.predict_survival(ct_volume, tumor_mask, {'tnm_stage': {'T': 'T1', 'N': 'N0', 'M': 'M0'}, 'age': 55, 'gender': 'female'})
        assert result.median_survival_months > 0
    def test_advanced_stage_prognosis(self, engine, test_data):
        ct_volume, tumor_mask = test_data
        result = engine.predict_survival(ct_volume, tumor_mask, {'tnm_stage': {'T': 'T4', 'N': 'N2', 'M': 'M1'}, 'age': 75, 'gender': 'male'})
        assert isinstance(result, SurvivalPrediction)
        assert result.median_survival_months > 0

class TestPrognosisPerformance:
    def test_prediction_speed(self):
        import time
        engine = PrognosisEngine()
        ct_volume = np.random.randn(100, 150, 150) * 30 + 50
        tumor_mask = np.random.rand(100, 150, 150) > 0.98
        clinical = {'tnm_stage': {'T': 'T2', 'N': 'N1', 'M': 'M0'}, 'age': 65, 'gender': 'male'}
        start_time = time.time()
        result = engine.predict_survival(ct_volume, tumor_mask, clinical)
        elapsed = time.time() - start_time
        assert elapsed < 10.0
        print(f"Prognosis prediction time: {elapsed:.2f}s")

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
