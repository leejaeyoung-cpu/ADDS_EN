"""
Metadata Extraction Pipeline

Extracts structured metadata from:
1. CT analyses → Tumor characteristics
2. Cell images → Morphological features
3. Treatment records → Drug-tumor response patterns

This metadata is used for continuous learning.
"""

import numpy as np
import nibabel as nib
import torch
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime
import logging

from patient_management_system.database.db_enhanced import get_session
from patient_management_system.database.models_enhanced import (
    CTAnalysis, TumorMeasurement, CellImage, Treatment, TreatmentOutcome
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CTMetadataExtractor:
    """Extract metadata from CT analyses"""
    
    @staticmethod
    def extract_tumor_metadata(ct_volume: np.ndarray, segmentation: np.ndarray, 
                               spacing: tuple) -> Dict[str, Any]:
        """
        Extract comprehensive tumor metadata
        
        Args:
            ct_volume: CT volume array
            segmentation: Tumor segmentation mask
            spacing: Voxel spacing (z, y, x) in mm
        
        Returns:
            Dictionary of tumor characteristics
        """
        metadata = {}
        
        # Volume calculation
        voxel_volume_mm3 = spacing[0] * spacing[1] * spacing[2]
        tumor_voxels = np.sum(segmentation > 0)
        metadata['volume_ml'] = float(tumor_voxels * voxel_volume_mm3 / 1000)
        
        # Size measurements
        tumor_coords = np.argwhere(segmentation > 0)
        if len(tumor_coords) > 0:
            # Bounding box
            min_coords = tumor_coords.min(axis=0)
            max_coords = tumor_coords.max(axis=0)
            
            # Physical dimensions (mm)
            dims_voxels = max_coords - min_coords + 1
            dims_mm = dims_voxels * np.array(spacing)
            
            metadata['max_diameter_mm'] = float(dims_mm.max())
            metadata['min_diameter_mm'] = float(dims_mm.min())
            
            # Centroid
            center = tumor_coords.mean(axis=0)
            metadata['centroid'] = {
                'z': float(center[0] * spacing[0]),
                'y': float(center[1] * spacing[1]),
                'x': float(center[2] * spacing[2])
            }
            
            # HU statistics
            tumor_region = ct_volume[segmentation > 0]
            metadata['hu_stats'] = {
                'mean': float(tumor_region.mean()),
                'std': float(tumor_region.std()),
                'min': float(tumor_region.min()),
                'max': float(tumor_region.max()),
                'median': float(np.median(tumor_region)),
                'q25': float(np.percentile(tumor_region, 25)),
                'q75': float(np.percentile(tumor_region, 75))
            }
            
            # Shape features
            metadata['shape'] = {
                'compactness': CTMetadataExtractor._calculate_compactness(segmentation, spacing),
                'sphericity': CTMetadataExtractor._calculate_sphericity(segmentation, spacing)
            }
        
        return metadata
    
    @staticmethod
    def _calculate_compactness(mask: np.ndarray, spacing: tuple) -> float:
        """Calculate compactness (how round the tumor is)"""
        volume = np.sum(mask > 0) * np.prod(spacing)
        
        # Surface area approximation (simple gradient-based)
        from scipy import ndimage
        surface = ndimage.binary_dilation(mask).astype(int) - mask
        surface_area = np.sum(surface) * spacing[1] * spacing[2]  # Approximate
        
        if surface_area > 0:
            compactness = (volume ** (2/3)) / surface_area
            return float(compactness)
        return 0.0
    
    @staticmethod
    def _calculate_sphericity(mask: np.ndarray, spacing: tuple) -> float:
        """Calculate sphericity (0-1, 1 = perfect sphere)"""
        volume = np.sum(mask > 0) * np.prod(spacing)
        
        # Approximate equivalent sphere radius
        radius_equiv = (3 * volume / (4 * np.pi)) ** (1/3)
        
        # Actual size
        coords = np.argwhere(mask > 0)
        if len(coords) > 0:
            center = coords.mean(axis=0)
            distances = np.linalg.norm((coords - center) * spacing, axis=1)
            radius_actual = distances.max()
            
            if radius_actual > 0:
                return float(radius_equiv / radius_actual)
        
        return 0.0


class CellImageFeatureExtractor:
    """Extract features from cell images using CNN and classical methods"""
    
    def __init__(self):
        """Initialize feature extractor with pretrained CNN"""
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = None
        self.transform = None
    
    def _load_model(self):
        """Lazy load ResNet50 model for feature extraction"""
        if self.model is None:
            import torchvision.models as models
            import torchvision.transforms as transforms
            
            # Load pretrained ResNet50
            self.model = models.resnet50(pretrained=True)
            # Remove final classification layer to get features
            self.model = torch.nn.Sequential(*list(self.model.children())[:-1])
            self.model = self.model.to(self.device)
            self.model.eval()
            
            # Standard ImageNet normalization
            self.transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
            ])
    
    def extract_morphology_features(self, image_path: Path) -> Dict[str, Any]:
        """
        Extract comprehensive features from cell image
        
        Args:
            image_path: Path to cell image
            
        Returns:
            Dictionary containing CNN features and classical morphology metrics
        """
        from PIL import Image
        import cv2
        
        if not Path(image_path).exists():
            logger.warning(f"Image not found: {image_path}")
            return {'error': 'Image not found'}
        
        features = {}
        
        try:
            # === 1. CNN Feature Extraction ===
            self._load_model()
            
            # Load and preprocess image
            image_pil = Image.open(image_path).convert('RGB')
            image_tensor = self.transform(image_pil).unsqueeze(0).to(self.device)
            
            # Extract features
            with torch.no_grad():
                cnn_features = self.model(image_tensor)
                cnn_features = cnn_features.squeeze().cpu().numpy()
            
            # Store as list (2048-dim for ResNet50)
            features['cnn_features'] = cnn_features.tolist()
            features['cnn_feature_dim'] = len(cnn_features)
            
            # === 2. Classical Morphology Features ===
            image_cv = cv2.imread(str(image_path))
            gray = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY)
            
            # Cell density estimation (simple threshold-based)
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            cell_pixels = np.sum(binary > 0)
            total_pixels = gray.shape[0] * gray.shape[1]
            features['cell_density'] = float(cell_pixels / total_pixels)
            
            # Texture analysis using GLCM
            from skimage.feature import graycomatrix, graycoprops
            glcm = graycomatrix(gray, distances=[1], angles=[0], levels=256, symmetric=True, normed=True)
            features['texture'] = {
                'contrast': float(graycoprops(glcm, 'contrast')[0, 0]),
                'dissimilarity': float(graycoprops(glcm, 'dissimilarity')[0, 0]),
                'homogeneity': float(graycoprops(glcm, 'homogeneity')[0, 0]),
                'energy': float(graycoprops(glcm, 'energy')[0, 0]),
                'correlation': float(graycoprops(glcm, 'correlation')[0, 0])
            }
            
            # Color statistics
            hsv = cv2.cvtColor(image_cv, cv2.COLOR_BGR2HSV)
            features['color_stats'] = {
                'hue_mean': float(hsv[:,:,0].mean()),
                'saturation_mean': float(hsv[:,:,1].mean()),
                'value_mean': float(hsv[:,:,2].mean()),
                'hue_std': float(hsv[:,:,0].std()),
                'saturation_std': float(hsv[:,:,1].std()),
                'value_std': float(hsv[:,:,2].std())
            }
            
            # Nuclear features (simplified - would need proper segmentation in production)
            features['nuclear_estimate'] = {
                'intensity_mean': float(gray.mean()),
                'intensity_std': float(gray.std()),
                'pixel_area': int(cell_pixels)
            }
            
            features['extraction_method'] = 'ResNet50_CNN + Classical'
            features['success'] = True
            
        except Exception as e:
            logger.error(f"Error extracting features from {image_path}: {e}")
            features['error'] = str(e)
            features['success'] = False
        
        return features


class MetadataAggregator:
    """Aggregate metadata from multiple patients for ML"""
    
    def __init__(self):
        self.db = get_session()
    
    def create_training_dataset(self) -> Dict[str, Any]:
        """
        Create training dataset from all patient metadata
        
        Returns:
            Dataset ready for ML training
        """
        logger.info("Creating training dataset from patient metadata...")
        
        dataset = {
            'created_at': datetime.utcnow().isoformat(),
            'samples': [],
            'feature_names': [],
            'statistics': {}
        }
        
        # Get all patients with complete data
        from sqlalchemy import and_
        
        analyses = (
            self.db.query(CTAnalysis)
            .filter(and_(
                CTAnalysis.status == 'completed',
                CTAnalysis.tumor_characteristics.isnot(None)
            ))
            .all()
        )
        
        logger.info(f"Found {len(analyses)} complete CT analyses")
        
        for analysis in analyses:
            # Get associated treatment
            treatment = (
                self.db.query(Treatment)
                .filter(Treatment.ct_analysis_id == analysis.id)
                .first()
            )
            
            if not treatment:
                continue
            
            # Get outcome
            outcome = (
                self.db.query(TreatmentOutcome)
                .filter(TreatmentOutcome.treatment_id == treatment.id)
                .first()
            )
            
            if not outcome:
                continue
            
            # Create sample
            sample = {
                'patient_id': analysis.patient.patient_id,
                'features': {
                    'tumor_volume_ml': analysis.volume_ml,
                    'max_diameter_mm': analysis.max_diameter_mm,
                    'hu_mean': analysis.hu_mean,
                    'hu_std': analysis.hu_std,
                    # Add more features from tumor_characteristics JSON
                    **analysis.tumor_characteristics
                },
                'treatment': {
                    'drugs': treatment.drug_cocktail,
                    'duration_days': (treatment.end_date - treatment.start_date).days if treatment.end_date else None
                },
                'outcome': {
                    'response_type': outcome.response_type.value,
                    'tumor_change_percent': outcome.tumor_size_change_percent,
                    'pfs_days': outcome.pfs_days
                }
            }
            
            dataset['samples'].append(sample)
        
        logger.info(f"Created dataset with {len(dataset['samples'])} samples")
        
        return dataset
    
    def save_metadata_snapshot(self) -> int:
        """Save current metadata snapshot to database"""
        from patient_management_system.database.models_enhanced import MetadataSnapshot
        
        # Create snapshot
        snapshot = MetadataSnapshot(
            total_patients=self.db.query(Patient).count(),
            total_analyses=self.db.query(CTAnalysis).count(),
            total_treatments=self.db.query(Treatment).count(),
            feature_statistics=self._compute_feature_stats(),
            treatment_patterns=self._analyze_treatment_patterns(),
            outcome_statistics=self._compute_outcome_stats()
        )
        
        self.db.add(snapshot)
        self.db.commit()
        
        logger.info(f"Saved metadata snapshot #{snapshot.id}")
        return snapshot.id
    
    def _compute_feature_stats(self) -> Dict[str, Any]:
        """Compute statistics of features"""
        analyses = self.db.query(CTAnalysis).filter(CTAnalysis.volume_ml.isnot(None)).all()
        
        if not analyses:
            return {}
        
        volumes = [a.volume_ml for a in analyses if a.volume_ml]
        diameters = [a.max_diameter_mm for a in analyses if a.max_diameter_mm]
        
        return {
            'tumor_volume_ml': {
                'mean': float(np.mean(volumes)) if volumes else 0,
                'std': float(np.std(volumes)) if volumes else 0,
                'count': len(volumes)
            },
            'max_diameter_mm': {
                'mean': float(np.mean(diameters)) if diameters else 0,
                'std': float(np.std(diameters)) if diameters else 0,
                'count': len(diameters)
            }
        }
    
    def _analyze_treatment_patterns(self) -> Dict[str, Any]:
        """Analyze common treatment patterns"""
        treatments = self.db.query(Treatment).all()
        
        drug_combinations = {}
        for t in treatments:
            drugs = tuple(sorted([d['drug_name'] for d in t.drug_cocktail]))
            drug_combinations[drugs] = drug_combinations.get(drugs, 0) + 1
        
        return {
            'unique_combinations': len(drug_combinations),
            'most_common': sorted(drug_combinations.items(), key=lambda x: x[1], reverse=True)[:5]
        }
    
    def _compute_outcome_stats(self) -> Dict[str, Any]:
        """Compute outcome statistics"""
        outcomes = self.db.query(TreatmentOutcome).all()
        
        response_counts = {}
        for o in outcomes:
            response_counts[o.response_type.value] = response_counts.get(o.response_type.value, 0) + 1
        
        return {
            'total_outcomes': len(outcomes),
            'response_distribution': response_counts
        }


if __name__ == "__main__":
    print("=" * 80)
    print("Metadata Extraction System Test")
    print("=" * 80)
    
    # Test aggregator
    aggregator = MetadataAggregator()
    
    # Create snapshot
    snapshot_id = aggregator.save_metadata_snapshot()
    print(f"\nCreated snapshot: {snapshot_id}")
    
    # Create dataset
    dataset = aggregator.create_training_dataset()
    print(f"\nDataset samples: {len(dataset['samples'])}")
