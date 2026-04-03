"""
Stage 4: Radiomics Feature Extraction Module
Extracts quantitative features from medical images
"""

import numpy as np
from typing import Dict, Optional, List, Tuple
try:
    import SimpleITK as sitk
    from radiomics import featureextractor
    RADIOMICS_AVAILABLE = True
    SITK_BSPLINE = sitk.sitkBSpline
except ImportError:
    print("Warning: radiomics or SimpleITK not installed. Using simplified feature extraction.")
    RADIOMICS_AVAILABLE = False
    SITK_BSPLINE = None  # Fallback value
    sitk = None
from scipy import ndimage
import logging
from pathlib import Path
import pandas as pd

logger = logging.getLogger(__name__)


class RadiomicsExtractor:
    """Radiomics feature extraction from tumor masks"""
    
    def __init__(self, bin_width: int = 25, 
                 resampled_pixel_spacing: Optional[Tuple[float, float, float]] = None):
        """
        Initialize radiomics extractor
        
        Args:
            bin_width: Bin width for discretization
            resampled_pixel_spacing: Target pixel spacing for resampling
        """
        self.bin_width = bin_width
        self.resampled_pixel_spacing = resampled_pixel_spacing or (1.0, 1.0, 1.0)
        
        if RADIOMICS_AVAILABLE:
            # PyRadiomics settings
            settings = {
                'binWidth': bin_width,
                'resampledPixelSpacing': self.resampled_pixel_spacing,
                'interpolator': SITK_BSPLINE,  # Use the safe constant
                'label': 1
            }
            
            self.extractor = featureextractor.RadiomicsFeatureExtractor(**settings)
        else:
            self.extractor = None
        
        # Enable feature classes
        self.enabled_features = {
            'shape': [],  # All shape features
            'firstorder': [],  # All first-order features
            'glcm': [],  # Gray Level Co-occurrence Matrix
            'glrlm': [],  # Gray Level Run Length Matrix
            'glszm': [],  # Gray Level Size Zone Matrix
            'gldm': [],  # Gray Level Dependence Matrix
            'ngtdm': []  # Neighboring Gray Tone Difference Matrix
        }
        
        self._initialize_extractor()
    
    def _initialize_extractor(self):
        """Initialize PyRadiomics feature extractor"""
        if not RADIOMICS_AVAILABLE or self.extractor is None:
            logger.warning("PyRadiomics not available, skipping initialization")
            return
            
        # Enable all features
        for feature_class in self.enabled_features:
            self.extractor.enableFeatureClassByName(feature_class)
        
        # Enable wavelet features for multi-scale analysis
        self.extractor.enableImageTypeByName('Wavelet')
        
        logger.info("Radiomics extractor initialized")

    def _extract_simplified_features(self, volume: np.ndarray, mask: np.ndarray) -> Dict[str, float]:
        """
        Extract simplified features when PyRadiomics is not available.
        This is a placeholder for basic feature extraction.
        """
        logger.warning("PyRadiomics not available. Returning simplified features.")
        # Example: return tumor volume
        tumor_volume_voxels = np.sum(mask)
        # Assuming pixel spacing is available or default to 1.0
        pixel_volume = np.prod(self.resampled_pixel_spacing) if self.resampled_pixel_spacing else 1.0
        actual_volume = tumor_volume_voxels * pixel_volume

        # Add a simple centroid calculation
        if tumor_volume_voxels > 0:
            coords = np.argwhere(mask)
            centroid = np.mean(coords, axis=0)
            # Convert to float for consistency with radiomics output
            centroid_x, centroid_y, centroid_z = float(centroid[0]), float(centroid[1]), float(centroid[2])
        else:
            centroid_x, centroid_y, centroid_z = np.nan, np.nan, np.nan

        return {
            'simplified_tumor_volume_voxels': float(tumor_volume_voxels),
            'simplified_tumor_volume_mm3': float(actual_volume),
            'simplified_centroid_x': centroid_x,
            'simplified_centroid_y': centroid_y,
            'simplified_centroid_z': centroid_z,
            # Add other basic features as needed
        }
    
    def extract_features(self, volume: np.ndarray, 
                        mask: np.ndarray) -> Dict[str, float]:
        """
        Extract radiomics features from tumor region
        
        Args:
            volume: 3D CT volume
            mask: Binary 3D tumor mask
            
        Returns:
            features: Dictionary of feature name -> value
        """
        if not RADIOMICS_AVAILABLE or sitk is None:
            logger.warning("SimpleITK not available, using simplified features")
            return self._extract_simplified_features(volume, mask)
            
        # Convert to SimpleITK images
        image_sitk = sitk.GetImageFromArray(volume.astype(np.float32))
        mask_sitk = sitk.GetImageFromArray(mask.astype(np.uint8))
        
        # Set same spacing for both
        spacing = self.resampled_pixel_spacing
        image_sitk.SetSpacing(spacing)
        mask_sitk.SetSpacing(spacing)
        
        # Extract features
        logger.info("Extracting radiomics features...")
        features_raw = self.extractor.execute(image_sitk, mask_sitk)
        
        # Filter out diagnostics info
        features = {}
        for key, val in features_raw.items():
            if not key.startswith('diagnostics'):
                try:
                    features[key] = float(val)
                except (ValueError, TypeError):
                    logger.warning(f"Could not convert feature {key} to float: {val}")
        
        logger.info(f"Extracted {len(features)} radiomics features")
        
        return features
    
    def extract_features_batch(self, volume: np.ndarray, 
                              masks: List[np.ndarray]) -> List[Dict[str, float]]:
        """
        Extract features from multiple tumor regions
        
        Args:
            volume: 3D CT volume
            masks: List of binary tumor masks
            
        Returns:
            features_list: List of feature dictionaries
        """
        features_list = []
        
        for i, mask in enumerate(masks):
            logger.info(f"Processing tumor {i+1}/{len(masks)}")
            features = self.extract_features(volume, mask)
            features['tumor_id'] = i + 1
            features_list.append(features)
        
        return features_list
    
    def get_feature_names(self) -> List[str]:
        """Get list of all feature names"""
        # Create a dummy image and mask
        dummy_image = sitk.GetImageFromArray(np.ones((10, 10, 10), dtype=np.float32))
        dummy_mask = sitk.GetImageFromArray(np.ones((10, 10, 10), dtype=np.uint8))
        
        # Extract to get feature names
        features_raw = self.extractor.execute(dummy_image, dummy_mask)
        
        feature_names = [key for key in features_raw.keys() 
                        if not key.startswith('diagnostics')]
        
        return feature_names
    
    def features_to_dataframe(self, features_list: List[Dict[str, float]]) -> pd.DataFrame:
        """
        Convert features list to pandas DataFrame
        
        Args:
            features_list: List of feature dictionaries
            
        Returns:
            df: DataFrame with features
        """
        df = pd.DataFrame(features_list)
        return df
    
    def save_features(self, features: Dict[str, float], output_path: Path):
        """
        Save features to CSV
        
        Args:
            features: Feature dictionary
            output_path: Output CSV path
        """
        df = pd.DataFrame([features])
        df.to_csv(output_path, index=False)
        logger.info(f"Features saved to {output_path}")


class TumorCharacterizer:
    """High-level tumor characterization using radiomics"""
    
    def __init__(self):
        self.radiomics_extractor = RadiomicsExtractor()
    
    def characterize_tumor(self, volume: np.ndarray, 
                          tumor_mask: np.ndarray,
                          tumor_metadata: Optional[Dict] = None) -> Dict:
        """
        Complete tumor characterization
        
        Args:
            volume: 3D CT volume
            tumor_mask: Binary tumor mask
            tumor_metadata: Optional metadata (volume, centroid, etc.)
            
        Returns:
            characterization: Complete characterization dictionary
        """
        # Extract radiomics features
        radiomics = self.radiomics_extractor.extract_features(volume, tumor_mask)
        
        # Calculate basic properties
        tumor_volume_voxels = np.sum(tumor_mask)
        
        characterization = {
            'radiomics_features': radiomics,
            'tumor_volume_voxels': int(tumor_volume_voxels),
        }
        
        # Add metadata if provided
        if tumor_metadata:
            characterization.update(tumor_metadata)
        
        # Extract key clinical indicators
        characterization['sphericity'] = radiomics.get('original_shape_Sphericity', None)
        characterization['surface_area_volume_ratio'] = radiomics.get(
            'original_shape_SurfaceVolumeRatio', None
        )
        characterization['heterogeneity_entropy'] = radiomics.get(
            'original_firstorder_Entropy', None
        )
        characterization['texture_contrast'] = radiomics.get(
            'original_glcm_Contrast', None
        )
        
        return characterization
    
    def estimate_vascularity(self, radiomics: Dict[str, float]) -> float:
        """
        Estimate tumor vascularity from texture features
        
        Args:
            radiomics: Radiomics feature dictionary
            
        Returns:
            vascularity_score: Estimated vascularity (0-1)
        """
        # Use texture heterogeneity as proxy
        # High entropy + high contrast = high vascularity
        entropy = radiomics.get('original_firstorder_Entropy', 0)
        contrast = radiomics.get('original_glcm_Contrast', 0)
        
        # Normalize (these are rough estimates)
        entropy_norm = min(entropy / 5.0, 1.0)
        contrast_norm = min(contrast / 200.0, 1.0)
        
        vascularity_score = (entropy_norm + contrast_norm) / 2
        
        return vascularity_score


if __name__ == "__main__":
    # Example usage
    import sys
    
    logging.basicConfig(level=logging.INFO)
    
    # Create sample data
    volume = np.random.rand(100, 100, 100).astype(np.float32) * 100
    tumor_mask = np.zeros((100, 100, 100), dtype=np.uint8)
    tumor_mask[40:60, 40:60, 40:60] = 1  # Cube tumor
    
    # Extract features
    extractor = RadiomicsExtractor()
    features = extractor.extract_features(volume, tumor_mask)
    
    print(f"\n✓ Extracted {len(features)} features")
    print("\nSample features:")
    for i, (key, val) in enumerate(list(features.items())[:10]):
        print(f"  {key}: {val:.4f}")
    
    # Characterize tumor
    characterizer = TumorCharacterizer()
    characterization = characterizer.characterize_tumor(volume, tumor_mask)
    
    print(f"\n✓ Sphericity: {characterization['sphericity']:.4f}")
    print(f"✓ Heterogeneity: {characterization['heterogeneity_entropy']:.4f}")
