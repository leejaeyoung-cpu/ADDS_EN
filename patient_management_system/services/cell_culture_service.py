"""
Cell Culture Analysis Service
Integrates Cellpose for microscopy image analysis
"""

from pathlib import Path
from typing import Dict, Any, Optional
import sys
import numpy as np
from PIL import Image
import json

# Add ADDS src to path
ADDS_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ADDS_ROOT))
sys.path.insert(0, str(ADDS_ROOT / "src"))

# Import existing Cellpose components
try:
    from preprocessing.image_processor import CellposeProcessor
    from ui.cdss_cellpose_pipeline import analyze_cellpose_with_pipeline
    from ai.dataset_builder import IntegratedDatasetBuilder
    CELLPOSE_AVAILABLE = True
except ImportError:
    CELLPOSE_AVAILABLE = False
    print("Warning: Cellpose components not available. Using simulated analysis.")


class CellCultureService:
    """
    Cell Culture Microscopy Analysis Service
    
    Uses Cellpose for:
    - Cell segmentation
    - Cell counting and morphology analysis
    - Spatial distribution analysis
    - Integration with pharmacokinetic dataset
    """
    
    def __init__(self):
        """Initialize cell culture analyzer"""
        self.cellpose_available = CELLPOSE_AVAILABLE
        if CELLPOSE_AVAILABLE:
            self.processor = CellposeProcessor()
            self.dataset_builder = IntegratedDatasetBuilder()
        else:
            self.processor = None
            self.dataset_builder = None
    
    def analyze_microscopy_image(
        self,
        image_path: str,
        pixel_size_um: float = 0.5,
        diameter: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Analyze microscopy image using Cellpose
        
        Args:
            image_path: Path to microscopy image (TIFF, PNG, JPG)
            pixel_size_um: Pixel size in micrometers (default: 0.5)
            diameter: Expected cell diameter in pixels (None = auto-detect)
            
        Returns:
            Comprehensive cell culture analysis results
        """
        if not self.cellpose_available:
            return self._simulated_analysis(image_path)
        
        try:
            # Use existing ADDS Cellpose pipeline
            results, visualizernum_cells, metadata = analyze_cellpose_with_pipeline(
                image_file=image_path,
                pixel_size_um=pixel_size_um,
                diameter=diameter
            )
            
            # Extract comprehensive features
            analysis = {
                "status": "success",
                "cellpose_version": metadata.get("cellpose_version", "2.0"),
                "processing_params": {
                    "pixel_size_um": pixel_size_um,
                    "diameter": diameter or "auto-detected"
                },
                
                # Core counts
                "num_cells": results.get("num_cells", 0),
                "total_area": results.get("total_area", 0),
                "coverage": results.get("coverage", 0),
                
                # Morphology features
                "morphology": {
                    "mean_area": results.get("mean_area", 0),
                    "std_area": results.get("std_area", 0),
                    "cv_area": results.get("cv_area", 0),  # Coefficient of variation
                    "mean_perimeter": results.get("mean_perimeter", 0),
                    "mean_circularity": results.get("mean_circularity", 0),
                    "mean_eccentricity": results.get("mean_eccentricity", 0),
                    "size_distribution": results.get("size_distribution", {})
                },
                
                # Spatial distribution
                "spatial_distribution": {
                    "clark_evans_index": results.get("clark_evans_index", 1.0),
                    "clustered_ratio": results.get("clustered_ratio", 0),
                    "num_clusters": results.get("num_clusters", 0),
                    "mean_nnd": results.get("mean_nnd", 0),  # Mean nearest neighbor distance
                    "spatial_pattern": results.get("spatial_pattern", "random")
                },
                
                # Heterogeneity analysis
                "heterogeneity": {
                    "overall_score": results.get("overall_heterogeneity", 0),
                    "grade": results.get("heterogeneity_grade", "Unknown"),
                    "size_entropy": results.get("size_entropy", 0),
                    "shape_diversity": results.get("shape_diversity", 0)
                },
                
                # Cell viability indicators (from morphology)
                "viability_indicators": {
                    "regular_shape_ratio": results.get("regular_shape_ratio", 0),
                    "fragmented_cells": results.get("fragmented_cells", 0),
                    "estimated_viability": results.get("estimated_viability", 0)
                },
                
                # Pharmacokinetic-relevant features
                "pharmacokinetic_features": self._extract_pk_features(results),
                
                # Processing metadata
                "processing_time_seconds": metadata.get("processing_time", 0),
                "image_dimensions": metadata.get("image_shape", [])
            }
            
            return analysis
            
        except Exception as e:
            print(f"Cellpose analysis error: {e}")
            return {
                "status": "error",
                "error_message": str(e),
                "fallback": self._simulated_analysis(image_path)
            }
    
    def _extract_pk_features(self, cellpose_results: Dict) -> Dict:
        """
        Extract pharmacokinetic-relevant features from cell culture
        
        These features help predict drug response and are used by ADDS
        """
        num_cells = cellpose_results.get("num_cells", 0)
        heterogeneity = cellpose_results.get("overall_heterogeneity", 0)
        clark_evans = cellpose_results.get("clark_evans_index", 1.0)
        
        # Cell density (cells per mm²)
        total_area_mm2 = cellpose_results.get("total_area", 1) / 1000000  # Convert µm² to mm²
        cell_density = num_cells / total_area_mm2 if total_area_mm2 > 0 else 0
        
        # Drug resistance indicators
        # High heterogeneity → potential drug resistance
        # Clustered pattern → microenvironment protection
        drug_resistance_score = (heterogeneity * 0.6) + \
                               ((1.0 - clark_evans) * 0.4)  # Low CE = clustered
        
        # Proliferation indicators
        # Large cells, high shape variation → active proliferation
        cv_area = cellpose_results.get("cv_area", 0)
        proliferation_score = min(1.0, cv_area / 0.5)  # Normalize CV
        
        return {
            "cell_density_per_mm2": float(cell_density),
            "drug_resistance_score": float(drug_resistance_score),
            "proliferation_score": float(proliferation_score),
            "microenvironment_complexity": float(heterogeneity),
            "spatial_organization": "clustered" if clark_evans < 0.9 else "dispersed",
            
            # Feature vector for ML model
            "feature_vector": [
                float(cell_density),
                float(heterogeneity),
                float(clark_evans),
                float(cv_area),
                float(cellpose_results.get("mean_circularity", 0))
            ]
        }
    
    def integrate_with_ct_data(
        self,
        ct_radiomics: Dict,
        ct_tumor_chars: Dict,
        cell_culture_data: Dict
    ) -> Dict:
        """
        Integrate cell culture data with CT analysis for comprehensive dataset
        
        Args:
            ct_radiomics: CT radiomics features
            ct_tumor_chars: CT tumor characteristics
            cell_culture_data: Cell culture analysis results
            
        Returns:
            Combined feature set for ADDS pharmacokinetic system
        """
        integrated = {
            "data_sources": ["CT_radiomics", "CT_tumor_characteristics", "cell_culture"],
            
            # CT features
            "ct_features": {
                "tumor_volume_mm3": ct_tumor_chars.get("morphology", {}).get("area_mm2", 0) * 5,
                "tumor_sphericity": ct_radiomics.get("shape_Sphericity", 0),
                "texture_entropy": ct_radiomics.get("firstorder_Entropy", 0),
                "mean_hu": ct_tumor_chars.get("intensity", {}).get("mean_hu", 0)
            },
            
            # Cell culture features
            "cell_culture_features": {
                "cell_density": cell_culture_data.get("pharmacokinetic_features", {}).get("cell_density_per_mm2", 0),
                "heterogeneity": cell_culture_data.get("heterogeneity", {}).get("overall_score", 0),
                "drug_resistance_score": cell_culture_data.get("pharmacokinetic_features", {}).get("drug_resistance_score", 0),
                "proliferation_score": cell_culture_data.get("pharmacokinetic_features", {}).get("proliferation_score", 0)
            },
            
            # Combined ML feature vector (14 features)
            "ml_feature_vector": self._create_ml_feature_vector(
                ct_radiomics, ct_tumor_chars, cell_culture_data
            ),
            
            # Risk assessment
            "integrated_risk_assessment": self._assess_integrated_risk(
                ct_tumor_chars, cell_culture_data
            )
        }
        
        return integrated
    
    def _create_ml_feature_vector(
        self,
        ct_radiomics: Dict,
        ct_tumor_chars: Dict,
        cell_culture: Dict
    ) -> list:
        """Create 14-dimensional feature vector for ML models"""
        
        # CT features (7 dimensions)
        ct_vector = [
            ct_radiomics.get("shape_Sphericity", 0),
            ct_radiomics.get("firstorder_Entropy", 0),
            ct_radiomics.get("glcm_Contrast", 0),
            ct_tumor_chars.get("morphology", {}).get("area_mm2", 0) / 1000,  # Normalize
            ct_tumor_chars.get("morphology", {}).get("circularity", 0),
            ct_tumor_chars.get("intensity", {}).get("mean_hu", 0) / 100,  # Normalize
            ct_tumor_chars.get("confidence_score", 0)
        ]
        
        # Cell culture features (7 dimensions)
        pk_features = cell_culture.get("pharmacokinetic_features", {})
        cell_vector = [
            pk_features.get("cell_density_per_mm2", 0) / 1000,  # Normalize
            pk_features.get("drug_resistance_score", 0),
            pk_features.get("proliferation_score", 0),
            pk_features.get("microenvironment_complexity", 0),
            cell_culture.get("morphology", {}).get("mean_circularity", 0),
            cell_culture.get("spatial_distribution", {}).get("clark_evans_index", 1.0),
            cell_culture.get("viability_indicators", {}).get("estimated_viability", 0)
        ]
        
        return ct_vector + cell_vector
    
    def _assess_integrated_risk(
        self,
        ct_tumor_chars: Dict,
        cell_culture: Dict
    ) -> Dict:
        """Assess risk based on integrated CT + cell culture data"""
        
        risk_factors = []
        
        # From CT
        tumor_size = ct_tumor_chars.get("morphology", {}).get("area_mm2", 0)
        if tumor_size > 500:
            risk_factors.append("Large tumor size (CT)")
        
        # From cell culture
        pk = cell_culture.get("pharmacokinetic_features", {})
        if pk.get("drug_resistance_score", 0) > 0.7:
            risk_factors.append("High drug resistance potential (cell culture)")
        
        heterogeneity = cell_culture.get("heterogeneity", {}).get("overall_score", 0)
        if heterogeneity > 0.7:
            risk_factors.append("High tumor heterogeneity (cell culture)")
        
        return {
            "risk_factors": risk_factors,
            "risk_level": "High" if len(risk_factors) >= 2 else "Moderate" if len(risk_factors) == 1 else "Low",
            "pharmacokinetic_complexity": "High" if len(risk_factors) >= 2 else "Standard"
        }
    
    def _simulated_analysis(self, image_path: str) -> Dict:
        """Generate simulated cell culture analysis when Cellpose is unavailable"""
        
        # Load image to get dimensions
        try:
            img = Image.open(image_path)
            width, height = img.size
        except:
            width, height = 2048, 2048
        
        # Simulate realistic cell culture metrics
        num_cells = np.random.randint(300, 800)
        heterogeneity = np.random.uniform(0.4, 0.8)
        
        return {
            "status": "simulated",
            "note": "Cellpose not available - using simulated data",
            "num_cells": int(num_cells),
            "morphology": {
                "mean_area": float(np.random.uniform(150, 300)),
                "cv_area": float(np.random.uniform(0.3, 0.6)),
                "mean_circularity": float(np.random.uniform(0.7, 0.9))
            },
            "spatial_distribution": {
                "clark_evans_index": float(np.random.uniform(0.8, 1.2)),
                "spatial_pattern": "random"
            },
            "heterogeneity": {
                "overall_score": float(heterogeneity),
                "grade": "Moderate" if heterogeneity < 0.7 else "High"
            },
            "pharmacokinetic_features": {
                "cell_density_per_mm2": float(num_cells / 4.0),  # Assume 2mm² image
                "drug_resistance_score": float(heterogeneity * 0.8),
                "proliferation_score": float(np.random.uniform(0.5, 0.8)),
                "feature_vector": [float(x) for x in np.random.rand(5)]
            },
            "image_dimensions": [width, height]
        }
