"""
Unit Tests for TNM Staging Engine

Tests all components of the TNM classification system:
- T stage classification (tumor size/invasion)
- N stage classification (lymph node involvement)
- M stage classification (metastasis detection)
- Overall stage mapping
"""

import pytest
import numpy as np
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.medical_imaging.staging import TNMStagingEngine, TNMStage


class TestTNMStagingEngine:
    """Test suite for TNM Staging Engine"""
    
    @pytest.fixture
    def engine(self):
        """Create TNM staging engine instance"""
        return TNMStagingEngine(spacing=(1.0, 1.0, 1.0))
    
    @pytest.fixture
    def small_tumor_mask(self):
        """Small tumor (≤2cm diameter) for T1 classification"""
        mask = np.zeros((100, 100, 100), dtype=np.uint8)
        # Create sphere with ~10mm radius (2cm diameter)
        center = (50, 50, 50)
        radius = 10
        
        z, y, x = np.ogrid[:100, :100, :100]
        dist = np.sqrt((z - center[0])**2 + (y - center[1])**2 + (x - center[2])**2)
        mask[dist <= radius] = 1
        
        return mask
    
    @pytest.fixture
    def medium_tumor_mask(self):
        """Medium tumor (2-5cm diameter) for T2 classification"""
        mask = np.zeros((100, 150, 150), dtype=np.uint8)
        center = (50, 75, 75)
        radius = 20  # ~4cm diameter
        
        z, y, x = np.ogrid[:100, :150, :150]
        dist = np.sqrt((z - center[0])**2 + (y - center[1])**2 + (x - center[2])**2)
        mask[dist <= radius] = 1
        
        return mask
    
    @pytest.fixture
    def large_tumor_mask(self):
        """Large tumor (>5cm diameter) for T3 classification"""
        mask = np.zeros((150, 200, 200), dtype=np.uint8)
        center = (75, 100, 100)
        radius = 30  # ~6cm diameter
        
        z, y, x = np.ogrid[:150, :200, :200]
        dist = np.sqrt((z - center[0])**2 + (y - center[1])**2 + (x - center[2])**2)
        mask[dist <= radius] = 1
        
        return mask
    
    @pytest.fixture
    def simple_organ_masks(self, small_tumor_mask):
        """Simple organ masks for testing (non-overlapping with tumor)"""
        # Create organ masks that don't overlap with tumor
        organs = {
            'colon': np.zeros_like(small_tumor_mask),
            'liver': np.zeros_like(small_tumor_mask)
        }
        # Place colon away from tumor center (tumor is at 50,50,50)
        organs['colon'][0:30, 0:30, 0:30] = 1
        organs['liver'][70:100, 70:100, 70:100] = 1
        return organs
    
    # ==================== T Stage Tests ====================
    
    def test_t0_classification_no_tumor(self, engine, simple_organ_masks):
        """Test T0 classification for no tumor"""
        empty_mask = np.zeros((100, 100, 100), dtype=np.uint8)
        
        t_stage, details = engine.classify_t_stage(empty_mask, simple_organ_masks)
        
        assert t_stage == "T0"
        assert details['volume_cm3'] == 0.0
    
    def test_t1_classification_small_tumor(self, engine, small_tumor_mask, simple_organ_masks):
        """Test T1 classification for small tumor (≤2cm)"""
        t_stage, details = engine.classify_t_stage(small_tumor_mask, simple_organ_masks)
        
        assert t_stage == "T1"
        assert details['max_diameter_mm'] <= 20.0  # ≤2cm
        assert details['invasion_detected'] == False
    
    def test_t2_classification_medium_tumor(self, engine, medium_tumor_mask, simple_organ_masks):
        """Test T2 classification for medium tumor (2-5cm)"""
        t_stage, details = engine.classify_t_stage(medium_tumor_mask, simple_organ_masks)
        
        assert t_stage == "T2"
        assert 20.0 < details['max_diameter_mm'] <= 50.0  # 2-5cm
        assert details['invasion_detected'] == False
    
    def test_t3_classification_large_tumor(self, engine, large_tumor_mask, simple_organ_masks):
        """Test T3 classification for large tumor (>5cm)"""
        t_stage, details = engine.classify_t_stage(large_tumor_mask, simple_organ_masks)
        
        assert t_stage == "T3"
        assert details['max_diameter_mm'] > 50.0  # >5cm
    
    def test_t4_classification_with_invasion(self, engine, medium_tumor_mask):
        """Test T4 classification when tumor invades adjacent organs"""
        # Create organ masks that overlap with tumor (simulating invasion)
        organ_masks = {
            'colon': medium_tumor_mask.copy(),
            'liver': medium_tumor_mask.copy()  # Liver overlaps with tumor
        }
        
        t_stage, details = engine.classify_t_stage(medium_tumor_mask, organ_masks)
        
        assert t_stage == "T4"
        assert details['invasion_detected'] == True
        assert len(details['invaded_organs']) > 0
    
    def test_max_diameter_measurement(self, engine):
        """Test accurate maximum diameter measurement"""
        # Create elongated tumor (20mm x 10mm x 10mm)
        mask = np.zeros((50, 50, 50), dtype=np.uint8)
        mask[15:35, 20:30, 20:30] = 1  # 20mm long
        
        max_diameter = engine._measure_max_diameter(mask)
        
        # Should be close to 20mm (with some tolerance)
        assert 18.0 <= max_diameter <= 25.0
    
    # ==================== N Stage Tests ====================
    
    def test_n0_classification_no_nodes(self, engine, simple_organ_masks):
        """Test N0 classification when no lymph nodes detected"""
        n_stage, details = engine.classify_n_stage(simple_organ_masks, None)
        
        assert n_stage == "N0"
        assert details['suspicious_nodes'] == 0
    
    def test_n1_classification_few_nodes(self, engine):
        """Test N1 classification for 1-3 lymph nodes"""
        # Create organ masks with 2 lymph nodes
        lymph_mask = np.zeros((100, 100, 100), dtype=np.uint8)
        
        # Node 1 (size > 10mm)
        lymph_mask[20:35, 20:35, 20:35] = 1
        
        # Node 2 (size > 10mm)
        lymph_mask[50:65, 50:65, 50:65] = 1
        
        organ_masks = {
            'lymph_node_right': lymph_mask
        }
        
        n_stage, details = engine.classify_n_stage(organ_masks, None)
        
        assert n_stage == "N1"
        assert 1 <= details['suspicious_nodes'] <= 3
    
    def test_n2_classification_many_nodes(self, engine):
        """Test N2 classification for ≥4 lymph nodes"""
        # Create organ masks with 5 lymph nodes
        lymph_mask = np.zeros((150, 150, 150), dtype=np.uint8)
        
        # Create 5 separate nodes
        positions = [(20, 20, 20), (50, 50, 50), (80, 80, 80), 
                     (30, 70, 30), (70, 30, 70)]
        
        for pos in positions:
            z, y, x = pos
            lymph_mask[z:z+15, y:y+15, x:x+15] = 1
        
        organ_masks = {
            'lymph_node_right': lymph_mask,
            'lymph_node_left': lymph_mask
        }
        
        n_stage, details = engine.classify_n_stage(organ_masks, None)
        
        assert n_stage == "N2"
        assert details['suspicious_nodes'] >= 4
    
    # ==================== M Stage Tests ====================
    
    def test_m0_classification_no_metastasis(self, engine, small_tumor_mask):
        """Test M0 classification when no distant metastasis"""
        # Tumor only in colon (primary site)
        organ_masks = {
            'colon': small_tumor_mask.copy(),
            'liver': np.zeros_like(small_tumor_mask),
            'lung_upper_lobe_left': np.zeros_like(small_tumor_mask)
        }
        
        m_stage, details = engine.classify_m_stage(small_tumor_mask, organ_masks)
        
        assert m_stage == "M0"
        assert details['metastasis_detected'] == False
        assert len(details['metastasis_sites']) == 0
    
    def test_m1_classification_liver_metastasis(self, engine, small_tumor_mask):
        """Test M1 classification with liver metastasis"""
        # Create separate tumor in liver (not continuous with colon)
        # Primary tumor at (40-60, 40-60, 40-60)
        # Liver tumor at separate location (70-85, 70-85, 70-85)
        liver_tumor = np.zeros_like(small_tumor_mask)
        liver_tumor[70:85, 70:85, 70:85] = 1  # Separate location
        
        combined_tumor = np.logical_or(small_tumor_mask, liver_tumor).astype(np.uint8)
        
        # Colon mask covers only primary tumor area
        colon_mask = np.zeros_like(small_tumor_mask)
        colon_mask[30:70, 30:70, 30:70] = 1
        
        # Liver mask covers the liver tumor area (not overlapping with colon)
        liver_mask = np.zeros_like(small_tumor_mask)
        liver_mask[60:100, 60:100, 60:100] = 1
        
        organ_masks = {
            'colon': colon_mask,
            'liver': liver_mask
        }
        
        m_stage, details = engine.classify_m_stage(combined_tumor, organ_masks)
        
        assert m_stage == "M1", f"Expected M1, got {m_stage}. Details: {details}"
        assert details['metastasis_detected'] == True
        assert 'liver' in details['metastasis_sites']
    
    # ==================== Overall Stage Mapping Tests ====================
    
    def test_stage_i_mapping(self, engine):
        """Test Stage I mapping (T1-T2, N0, M0)"""
        stage = engine._map_tnm_to_stage("T1", "N0", "M0")
        assert stage == "I"
        
        stage = engine._map_tnm_to_stage("T2", "N0", "M0")
        assert stage == "I"
    
    def test_stage_iia_mapping(self, engine):
        """Test Stage IIA mapping (T3, N0, M0)"""
        stage = engine._map_tnm_to_stage("T3", "N0", "M0")
        assert stage == "IIA"
    
    def test_stage_iib_mapping(self, engine):
        """Test Stage IIB mapping (T4, N0, M0)"""
        stage = engine._map_tnm_to_stage("T4", "N0", "M0")
        assert stage == "IIB"
    
    def test_stage_iiia_mapping(self, engine):
        """Test Stage IIIA mapping (T1-T2, N1, M0)"""
        stage = engine._map_tnm_to_stage("T1", "N1", "M0")
        assert stage == "IIIA"
        
        stage = engine._map_tnm_to_stage("T2", "N1", "M0")
        assert stage == "IIIA"
    
    def test_stage_iiib_mapping(self, engine):
        """Test Stage IIIB mapping"""
        # T3-T4, N1, M0
        stage = engine._map_tnm_to_stage("T3", "N1", "M0")
        assert stage == "IIIB"
        
        stage = engine._map_tnm_to_stage("T4", "N1", "M0")
        assert stage == "IIIB"
        
        # T1-T2, N2, M0
        stage = engine._map_tnm_to_stage("T1", "N2", "M0")
        assert stage == "IIIB"
    
    def test_stage_iiic_mapping(self, engine):
        """Test Stage IIIC mapping (T3-T4, N2, M0)"""
        stage = engine._map_tnm_to_stage("T3", "N2", "M0")
        assert stage == "IIIC"
        
        stage = engine._map_tnm_to_stage("T4", "N2", "M0")
        assert stage == "IIIC"
    
    def test_stage_iv_mapping(self, engine):
        """Test Stage IV mapping (any M1)"""
        # Any T, any N, M1 = Stage IV
        stage = engine._map_tnm_to_stage("T1", "N0", "M1")
        assert stage == "IV"
        
        stage = engine._map_tnm_to_stage("T4", "N2", "M1")
        assert stage == "IV"
    
    # ==================== Integration Tests ====================
    
    def test_complete_tnm_classification(self, engine, medium_tumor_mask, simple_organ_masks):
        """Test complete TNM classification pipeline"""
        result = engine.classify_tnm(medium_tumor_mask, simple_organ_masks)
        
        # Verify result structure
        assert isinstance(result, TNMStage)
        assert result.T in ["T0", "T1", "T2", "T3", "T4"]
        assert result.N in ["N0", "N1", "N2"]
        assert result.M in ["M0", "M1"]
        assert result.stage in ["I", "IIA", "IIB", "IIIA", "IIIB", "IIIC", "IV", "Unknown"]
        assert 0.0 <= result.confidence <= 1.0
        
        # Verify details
        assert 'T' in result.details
        assert 'N' in result.details
        assert 'M' in result.details
    
    def test_early_stage_cancer(self, engine, small_tumor_mask):
        """Test classification of early-stage cancer (Stage I)"""
        organ_masks = {
            'colon': small_tumor_mask.copy(),
            'liver': np.zeros_like(small_tumor_mask)
        }
        
        result = engine.classify_tnm(small_tumor_mask, organ_masks)
        
        # Should be Stage I (T1, N0, M0)
        assert result.T == "T1"
        assert result.N == "N0"
        assert result.M == "M0"
        assert result.stage == "I"
    
    def test_advanced_stage_cancer(self, engine, large_tumor_mask):
        """Test classification of advanced cancer with metastasis"""
        # Create liver metastasis at separate location
        liver_met = np.zeros_like(large_tumor_mask)
        liver_met[100:120, 140:160, 140:160] = 1  # Separate from primary tumor
        
        combined_tumor = np.logical_or(large_tumor_mask, liver_met).astype(np.uint8)
        
        # Create non-overlapping organ masks
        colon_mask = np.zeros_like(large_tumor_mask)
        colon_mask[50:100, 80:130, 80:130] = 1  # Covers primary tumor area
        
        liver_mask = np.zeros_like(large_tumor_mask)
        liver_mask[90:150, 130:200, 130:200] = 1  # Covers liver met area
        
        organ_masks = {
            'colon': colon_mask,
            'liver': liver_mask
        }
        
        result = engine.classify_tnm(combined_tumor, organ_masks)
        
        # Should be Stage IV (any T, any N, M1)
        assert result.M == "M1", f"Expected M1, got {result.M}. Details: {result.details['M']}"
        assert result.stage == "IV"
        assert 'liver' in result.details['M']['metastasis_sites']


# ==================== Performance Tests ====================

class TestTNMStagingPerformance:
    """Performance tests for TNM staging"""
    
    def test_large_volume_performance(self):
        """Test performance with large CT volumes"""
        import time
        
        engine = TNMStagingEngine(spacing=(1.0, 1.0, 1.0))
        
        # Large volume (512³)
        large_tumor = np.random.rand(512, 512, 512) > 0.99
        organ_masks = {
            'colon': np.random.rand(512, 512, 512) > 0.95,
            'liver': np.random.rand(512, 512, 512) > 0.95
        }
        
        start_time = time.time()
        result = engine.classify_tnm(large_tumor, organ_masks)
        elapsed_time = time.time() - start_time
        
        # Should complete in reasonable time (< 30 seconds)
        assert elapsed_time < 30.0
        print(f"Large volume TNM classification: {elapsed_time:.2f}s")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
