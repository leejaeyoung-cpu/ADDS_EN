"""
Feature Service
Real morphological feature extraction using skimage regionprops and GLCM
"""

import sys
from pathlib import Path
import numpy as np
from typing import Dict, List, Optional, Any
import logging

sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))

from preprocessing.image_processor import CellposeProcessor

logger = logging.getLogger(__name__)


class FeatureService:
    """Service for morphological feature extraction from segmented cells"""
    
    def __init__(self):
        try:
            import torch
            gpu_available = torch.cuda.is_available()
            print(f"FeatureService: Initializing with GPU={'enabled' if gpu_available else 'disabled'}")
            self.processor = CellposeProcessor(gpu=gpu_available)
        except ImportError:
            print("FeatureService: PyTorch not found, using CPU")
            self.processor = CellposeProcessor(gpu=False)
    
    async def extract(
        self,
        image_id: str,
        masks: Any,
        feature_set: str = "basic",
        image: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        Extract morphological features from segmented cell masks
        
        Args:
            image_id: Identifier for the image
            masks: 2D label array from segmentation (each cell = unique int)
            feature_set: "basic", "advanced", or "all"
            image: Optional intensity image for intensity-based features
            
        Returns:
            Dict with per-cell features and summary statistics
        """
        from skimage.measure import regionprops, regionprops_table
        
        masks_arr = np.array(masks) if not isinstance(masks, np.ndarray) else masks
        
        if masks_arr.ndim != 2:
            raise ValueError(f"Masks must be 2D, got shape {masks_arr.shape}")
        
        num_cells = int(masks_arr.max())
        if num_cells == 0:
            return {
                "image_id": image_id,
                "num_cells": 0,
                "features": {},
                "summary": {},
                "feature_set": feature_set
            }
        
        logger.info(f"Extracting {feature_set} features for {num_cells} cells")
        
        # Basic morphological features via regionprops
        basic_props = [
            'label', 'area', 'perimeter', 'eccentricity',
            'solidity', 'extent', 'major_axis_length', 'minor_axis_length',
            'orientation', 'equivalent_diameter_area'
        ]
        
        # Add intensity features if image provided
        intensity_props = []
        if image is not None:
            intensity_props = ['mean_intensity', 'max_intensity', 'min_intensity']
        
        props = basic_props + intensity_props
        table = regionprops_table(masks_arr, intensity_image=image, properties=props)
        
        # Calculate derived features
        features_per_cell = {}
        n = len(table['label'])
        
        for i in range(n):
            cell_id = int(table['label'][i])
            cell_feat = {}
            
            # Basic morphology
            area = float(table['area'][i])
            perimeter = float(table['perimeter'][i])
            
            cell_feat['area'] = area
            cell_feat['perimeter'] = perimeter
            cell_feat['eccentricity'] = float(table['eccentricity'][i])
            cell_feat['solidity'] = float(table['solidity'][i])
            cell_feat['extent'] = float(table['extent'][i])
            cell_feat['major_axis'] = float(table['major_axis_length'][i])
            cell_feat['minor_axis'] = float(table['minor_axis_length'][i])
            cell_feat['orientation'] = float(table['orientation'][i])
            cell_feat['equivalent_diameter'] = float(table['equivalent_diameter_area'][i])
            
            # Derived: circularity  (4π × area / perimeter²)
            if perimeter > 0:
                cell_feat['circularity'] = float(4 * np.pi * area / (perimeter ** 2))
            else:
                cell_feat['circularity'] = 0.0
            
            # Derived: aspect ratio
            minor = cell_feat['minor_axis']
            if minor > 0:
                cell_feat['aspect_ratio'] = float(cell_feat['major_axis'] / minor)
            else:
                cell_feat['aspect_ratio'] = 1.0
            
            # Intensity features
            if image is not None:
                cell_feat['mean_intensity'] = float(table['mean_intensity'][i])
                cell_feat['max_intensity'] = float(table['max_intensity'][i])
                cell_feat['min_intensity'] = float(table['min_intensity'][i])
                cell_feat['intensity_range'] = cell_feat['max_intensity'] - cell_feat['min_intensity']
                cell_feat['integrated_intensity'] = cell_feat['mean_intensity'] * area
            
            features_per_cell[str(cell_id)] = cell_feat
        
        # Advanced spatial features
        if feature_set in ("advanced", "all"):
            features_per_cell = self._add_spatial_features(
                features_per_cell, masks_arr
            )
        
        # Texture (GLCM) features
        if feature_set == "all" and image is not None:
            features_per_cell = self._add_texture_features(
                features_per_cell, masks_arr, image
            )
        
        # Summary statistics across all cells
        summary = self._compute_summary(features_per_cell)
        
        return {
            "image_id": image_id,
            "num_cells": num_cells,
            "features": features_per_cell,
            "summary": summary,
            "feature_set": feature_set
        }
    
    def _add_spatial_features(
        self, features: Dict, masks: np.ndarray
    ) -> Dict:
        """Add spatial distribution features (nearest neighbor, density)"""
        from skimage.measure import regionprops
        
        props = regionprops(masks)
        centroids = np.array([p.centroid for p in props])
        labels = [p.label for p in props]
        
        if len(centroids) < 2:
            for label in labels:
                features[str(label)]['nearest_neighbor_dist'] = 0.0
                features[str(label)]['cell_density'] = 0.0
            return features
        
        # Compute pairwise distances
        from scipy.spatial.distance import cdist
        dists = cdist(centroids, centroids)
        np.fill_diagonal(dists, np.inf)
        
        nn_dists = np.min(dists, axis=1)
        
        # Image area for density calculation
        img_area = masks.shape[0] * masks.shape[1]
        cell_density = float(len(centroids) / img_area * 1e6)  # cells per million pixels
        
        for i, label in enumerate(labels):
            key = str(label)
            if key in features:
                features[key]['nearest_neighbor_dist'] = float(nn_dists[i])
                features[key]['cell_density'] = cell_density
        
        # Uniformity: coefficient of variation of NN distances
        if np.mean(nn_dists) > 0:
            uniformity = float(1.0 - np.std(nn_dists) / np.mean(nn_dists))
        else:
            uniformity = 0.0
        
        for label in labels:
            features[str(label)]['spatial_uniformity'] = max(0, uniformity)
        
        return features
    
    def _add_texture_features(
        self, features: Dict, masks: np.ndarray, image: np.ndarray
    ) -> Dict:
        """Add GLCM texture features per cell"""
        from skimage.feature import graycomatrix, graycoprops
        from skimage.measure import regionprops
        
        props = regionprops(masks, intensity_image=image)
        
        for prop in props:
            key = str(prop.label)
            if key not in features:
                continue
            
            # Get cell bounding box region
            min_r, min_c, max_r, max_c = prop.bbox
            cell_img = image[min_r:max_r, min_c:max_c].copy()
            cell_mask = masks[min_r:max_r, min_c:max_c] == prop.label
            
            # Mask out non-cell pixels
            cell_img[~cell_mask] = 0
            
            # Normalize to 0-255 uint8 for GLCM
            if cell_img.max() > 0:
                cell_uint8 = ((cell_img - cell_img.min()) / 
                              (cell_img.max() - cell_img.min()) * 255).astype(np.uint8)
            else:
                cell_uint8 = np.zeros_like(cell_img, dtype=np.uint8)
            
            if cell_uint8.shape[0] < 2 or cell_uint8.shape[1] < 2:
                features[key]['glcm_contrast'] = 0.0
                features[key]['glcm_dissimilarity'] = 0.0
                features[key]['glcm_homogeneity'] = 1.0
                features[key]['glcm_energy'] = 1.0
                features[key]['glcm_correlation'] = 0.0
                continue
            
            try:
                glcm = graycomatrix(
                    cell_uint8, distances=[1], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4],
                    levels=256, symmetric=True, normed=True
                )
                features[key]['glcm_contrast'] = float(np.mean(graycoprops(glcm, 'contrast')))
                features[key]['glcm_dissimilarity'] = float(np.mean(graycoprops(glcm, 'dissimilarity')))
                features[key]['glcm_homogeneity'] = float(np.mean(graycoprops(glcm, 'homogeneity')))
                features[key]['glcm_energy'] = float(np.mean(graycoprops(glcm, 'energy')))
                features[key]['glcm_correlation'] = float(np.mean(graycoprops(glcm, 'correlation')))
            except Exception as e:
                logger.warning(f"GLCM failed for cell {key}: {e}")
                features[key]['glcm_contrast'] = 0.0
                features[key]['glcm_dissimilarity'] = 0.0
                features[key]['glcm_homogeneity'] = 1.0
                features[key]['glcm_energy'] = 1.0
                features[key]['glcm_correlation'] = 0.0
        
        return features
    
    def _compute_summary(self, features: Dict) -> Dict[str, Any]:
        """Compute summary statistics across all cells"""
        if not features:
            return {}
        
        # Collect all feature names
        all_keys = set()
        for cell_feat in features.values():
            all_keys.update(cell_feat.keys())
        
        summary = {}
        for feat_name in sorted(all_keys):
            values = [
                cell_feat[feat_name] 
                for cell_feat in features.values() 
                if feat_name in cell_feat
            ]
            if not values:
                continue
            
            arr = np.array(values, dtype=float)
            summary[feat_name] = {
                "mean": float(np.mean(arr)),
                "std": float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0,
                "median": float(np.median(arr)),
                "min": float(np.min(arr)),
                "max": float(np.max(arr)),
                "q25": float(np.percentile(arr, 25)),
                "q75": float(np.percentile(arr, 75))
            }
        
        return summary
