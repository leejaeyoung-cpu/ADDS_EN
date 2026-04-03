"""
Test suite for CDSS metadata extraction pipeline

Run with: pytest patient_management_system/tests/test_metadata_extraction.py -v
"""

import pytest
import numpy as np
import tempfile
from pathlib import Path
from PIL import Image

from patient_management_system.services.metadata_extraction import (
    CTMetadataExtractor,
    CellImageFeatureExtractor,
    MetadataAggregator
)


class TestCTMetadataExtractor:
    """Test CT metadata extraction"""
    
    def test_extract_tumor_metadata(self):
        """Test tumor metadata extraction from CT volume"""
        # Create mock CT volume and segmentation
        ct_volume = np.random.randint(0, 100, size=(50, 128, 128), dtype=np.int16)
        segmentation = np.zeros((50, 128, 128), dtype=np.uint8)
        
        # Add tumor region
        segmentation[20:30, 50:70, 50:70] = 1
        
        spacing = (2.5, 1.0, 1.0)  # z, y, x in mm
        
        # Extract metadata
        metadata = CTMetadataExtractor.extract_tumor_metadata(
            ct_volume, segmentation, spacing
        )
        
        # Assertions
        assert 'volume_ml' in metadata
        assert metadata['volume_ml'] > 0
        assert 'max_diameter_mm' in metadata
        assert 'hu_stats' in metadata
        assert 'mean' in metadata['hu_stats']
        assert 'shape' in metadata
        
        print(f"✓ Tumor volume: {metadata['volume_ml']:.2f} ml")
        print(f"✓ Max diameter: {metadata['max_diameter_mm']:.2f} mm")
    
    def test_shape_features(self):
        """Test shape feature calculation"""
        # Spherical tumor
        seg_sphere = np.zeros((30, 30, 30), dtype=np.uint8)
        center = 15
        radius = 8
        
        for z in range(30):
            for y in range(30):
                for x in range(30):
                    if (z-center)**2 + (y-center)**2 + (x-center)**2 <= radius**2:
                        seg_sphere[z, y, x] = 1
        
        ct_volume = np.ones((30, 30, 30), dtype=np.int16) * 50
        spacing = (1.0, 1.0, 1.0)
        
        metadata = CTMetadataExtractor.extract_tumor_metadata(
            ct_volume, seg_sphere, spacing
        )
        
        # Sphere should have high sphericity
        assert metadata['shape']['sphericity'] > 0.5
        print(f"✓ Sphericity: {metadata['shape']['sphericity']:.3f}")


class TestCellImageFeatureExtractor:
    """Test cell image feature extraction"""
    
    def setup_method(self):
        """Create temporary directory for test images"""
        self.temp_dir = tempfile.mkdtemp()
        self.extractor = CellImageFeatureExtractor()
    
    def test_extract_from_real_image(self):
        """Test feature extraction from real image"""
        # Create synthetic cell image
        img_array = np.random.randint(0, 255, size=(224, 224, 3), dtype=np.uint8)
        
        # Add some cell-like structures
        from skimage.draw import disk
        rr, cc = disk((112, 112), 30)
        img_array[rr, cc, :] = [100, 50, 150]  # Purple nucleus
        
        # Save image
        img_path = Path(self.temp_dir) / "test_cell.jpg"
        img = Image.fromarray(img_array)
        img.save(img_path)
        
        # Extract features
        features = self.extractor.extract_morphology_features(img_path)
        
        # Assertions
        assert features.get('success') == True
        assert 'cnn_features' in features
        assert features['cnn_feature_dim'] == 2048  # ResNet50
        assert 'cell_density' in features
        assert 'texture' in features
        assert 'color_stats' in features
        
        print(f"✓ CNN features: {features['cnn_feature_dim']}-dim")
        print(f"✓ Cell density: {features['cell_density']:.3f}")
        print(f"✓ Texture contrast: {features['texture']['contrast']:.2f}")
    
    def test_nonexistent_image(self):
        """Test handling of nonexistent image"""
        features = self.extractor.extract_morphology_features(
            Path("/nonexistent/image.jpg")
        )
        
        assert 'error' in features
        print("✓ Correctly handles missing images")


class TestMetadataAggregator:
    """Test metadata aggregation for ML"""
    
    def test_feature_stats_calculation(self):
        """Test feature statistics computation"""
        aggregator = MetadataAggregator()
        
        # Calculate stats (will work with existing DB)
        stats = aggregator._compute_feature_stats()
        
        assert isinstance(stats, dict)
        print(f"✓ Feature statistics computed: {list(stats.keys())}")
    
    def test_treatment_pattern_analysis(self):
        """Test treatment pattern analysis"""
        aggregator = MetadataAggregator()
        
        patterns = aggregator._analyze_treatment_patterns()
        
        assert isinstance(patterns, dict)
        assert 'unique_combinations' in patterns
        print(f"✓ Treatment patterns analyzed")


if __name__ == "__main__":
    print("="*80)
    print("CDSS Metadata Extraction Test Suite")
    print("="*80)
    
    pytest.main([__file__, "-v", "-s"])
