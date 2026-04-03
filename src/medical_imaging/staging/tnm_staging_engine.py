"""
TNM Staging Engine for Automated Cancer Classification

Based on:
- YOLOv8 + TNMClassifier (2024, IEEE)
- TR-Net (Transformer-ResNet) for TNM Staging (2024, NIH)

TNM Classification System:
- T (Tumor): Primary tumor size/invasion
- N (Node): Lymph node involvement
- M (Metastasis): Distant metastasis
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, Tuple, Optional, List
from pathlib import Path
import logging
from scipy import ndimage
from dataclasses import dataclass


@dataclass
class TNMStage:
    """TNM staging result data class"""
    T: str  # T0, T1, T2, T3, T4
    N: str  # N0, N1, N2
    M: str  # M0, M1
    stage: str  # I, IIA, IIB, IIIA, IIIB, IIIC, IV
    confidence: float
    details: Dict[str, any]


class TNMStagingEngine:
    """
    Automated TNM staging system for colon cancer.
    
    Classification criteria (AJCC 8th edition for colon cancer):
    
    T Stage (Tumor):
    - T0: No evidence of primary tumor
    - T1: Tumor invades submucosa (≤2cm)
    - T2: Tumor invades muscularis propria (2-5cm)
    - T3: Tumor invades through muscularis propria (>5cm or wall penetration)
    - T4: Tumor invades adjacent organs
    
    N Stage (Lymph Nodes):
    - N0: No regional lymph node metastasis
    - N1: 1-3 regional lymph nodes involved
    - N2: ≥4 regional lymph nodes involved
    
    M Stage (Metastasis):
    - M0: No distant metastasis
    - M1: Distant metastasis present (liver, lung, peritoneum, etc.)
    
    Overall Stage:
    - I: T1-2, N0, M0
    - IIA: T3, N0, M0
    - IIB: T4, N0, M0
    - IIIA: T1-2, N1, M0
    - IIIB: T3-4, N1, M0 or T1-2, N2, M0
    - IIIC: T3-4, N2, M0
    - IV: Any T, Any N, M1
    """
    
    def __init__(
        self,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0)
    ):
        """
        Initialize TNM Staging Engine.
        
        Args:
            device: Computation device ('cuda' or 'cpu')
            spacing: Voxel spacing in mm (z, y, x)
        """
        self.device = device
        self.spacing = spacing
        self.logger = logging.getLogger(__name__)
        
        # AJCC criteria thresholds
        self.t_stage_thresholds = {
            'size_small': 20.0,  # mm (2cm)
            'size_medium': 50.0,  # mm (5cm)
        }
        
        self.n_stage_thresholds = {
            'min_lymph_node_size': 10.0,  # mm
            'few_nodes': 3,  # 1-3 nodes = N1
        }
        
        self.logger.info(f"Initialized TNMStagingEngine (device={device})")
    
    def classify_tnm(
        self,
        tumor_mask: np.ndarray,
        organ_masks: Dict[str, np.ndarray],
        ct_volume: Optional[np.ndarray] = None
    ) -> TNMStage:
        """
        Perform complete TNM staging.
        
        Args:
            tumor_mask: Binary mask of detected tumor (D, H, W)
            organ_masks: Dictionary of organ segmentation masks
            ct_volume: Original CT volume (optional, for advanced analysis)
            
        Returns:
            TNMStage object with classification results
        """
        self.logger.info("Starting TNM classification...")
        
        # T Stage: Tumor size and invasion
        t_stage, t_details = self.classify_t_stage(tumor_mask, organ_masks)
        
        # N Stage: Lymph node involvement
        n_stage, n_details = self.classify_n_stage(organ_masks, ct_volume)
        
        # M Stage: Distant metastasis
        m_stage, m_details = self.classify_m_stage(tumor_mask, organ_masks)
        
        # Overall stage
        overall_stage = self._map_tnm_to_stage(t_stage, n_stage, m_stage)
        
        # Calculate confidence (simplified - in production, use model confidence)
        confidence = self._calculate_confidence(t_details, n_details, m_details)
        
        result = TNMStage(
            T=t_stage,
            N=n_stage,
            M=m_stage,
            stage=overall_stage,
            confidence=confidence,
            details={
                'T': t_details,
                'N': n_details,
                'M': m_details
            }
        )
        
        self.logger.info(f"TNM Classification: {t_stage} {n_stage} {m_stage} → Stage {overall_stage}")
        
        return result
    
    def classify_t_stage(
        self,
        tumor_mask: np.ndarray,
        organ_masks: Dict[str, np.ndarray]
    ) -> Tuple[str, Dict]:
        """
        Classify T stage based on tumor size and invasion.
        
        Args:
            tumor_mask: Binary tumor mask
            organ_masks: Organ segmentation masks
            
        Returns:
            Tuple of (T stage string, details dict)
        """
        # Calculate tumor volume and maximum diameter
        tumor_volume_mm3 = np.sum(tumor_mask) * np.prod(self.spacing)
        tumor_volume_cm3 = tumor_volume_mm3 / 1000.0
        
        # Estimate maximum diameter from volume (assuming spherical approximation)
        radius_mm = ((3 * tumor_volume_mm3) / (4 * np.pi)) ** (1/3)
        max_diameter_mm = 2 * radius_mm
        
        # More accurate: measure actual maximum diameter using distance transform
        if np.sum(tumor_mask) > 0:
            actual_max_diameter_mm = self._measure_max_diameter(tumor_mask)
            max_diameter_mm = max(max_diameter_mm, actual_max_diameter_mm)
        
        # Check for invasion into adjacent organs
        invasion_detected = self._detect_organ_invasion(tumor_mask, organ_masks)
        
        # Classify T stage
        if np.sum(tumor_mask) == 0:
            t_stage = "T0"
        elif invasion_detected:
            t_stage = "T4"
        elif max_diameter_mm > self.t_stage_thresholds['size_medium']:
            t_stage = "T3"
        elif max_diameter_mm > self.t_stage_thresholds['size_small']:
            t_stage = "T2"
        else:
            t_stage = "T1"
        
        details = {
            'volume_cm3': tumor_volume_cm3,
            'max_diameter_mm': max_diameter_mm,
            'invasion_detected': invasion_detected,
            'invaded_organs': self._get_invaded_organs(tumor_mask, organ_masks) if invasion_detected else []
        }
        
        return t_stage, details
    
    def classify_n_stage(
        self,
        organ_masks: Dict[str, np.ndarray],
        ct_volume: Optional[np.ndarray] = None
    ) -> Tuple[str, Dict]:
        """
        Classify N stage based on lymph node involvement.
        
        Args:
            organ_masks: Organ segmentation masks (should include lymph nodes if available)
            ct_volume: CT volume for intensity-based lymph node detection
            
        Returns:
            Tuple of (N stage string, details dict)
        """
        # Detect lymph nodes from organ segmentation
        lymph_nodes = self._detect_lymph_nodes(organ_masks, ct_volume)
        
        # Count suspicious nodes (size > threshold)
        suspicious_nodes = [node for node in lymph_nodes if node['size_mm'] >= self.n_stage_thresholds['min_lymph_node_size']]
        num_nodes = len(suspicious_nodes)
        
        # Classify N stage
        if num_nodes == 0:
            n_stage = "N0"
        elif num_nodes <= self.n_stage_thresholds['few_nodes']:
            n_stage = "N1"
        else:
            n_stage = "N2"
        
        details = {
            'total_lymph_nodes': len(lymph_nodes),
            'suspicious_nodes': num_nodes,
            'node_details': suspicious_nodes[:5]  # Top 5 largest nodes
        }
        
        return n_stage, details
    
    def classify_m_stage(
        self,
        tumor_mask: np.ndarray,
        organ_masks: Dict[str, np.ndarray]
    ) -> Tuple[str, Dict]:
        """
        Classify M stage based on distant metastasis detection.
        
        Common metastasis sites for colon cancer:
        - Liver (most common)
        - Lung
        - Peritoneum
        - Distant lymph nodes
        
        Args:
            tumor_mask: Primary tumor mask
            organ_masks: Organ segmentation masks
            
        Returns:
            Tuple of (M stage string, details dict)
        """
        # Get colon mask to identify primary tumor location
        colon_mask = organ_masks.get('colon', np.zeros_like(tumor_mask))
        
        # Detect metastasis in key organs
        metastasis_sites = []
        
        # Check liver
        if 'liver' in organ_masks:
            liver_mets = self._detect_organ_metastasis(
                tumor_mask, organ_masks['liver'], colon_mask, 'liver'
            )
            if liver_mets['detected']:
                metastasis_sites.append('liver')
        
        # Check lungs
        for lung_key in ['lung_upper_lobe_left', 'lung_upper_lobe_right', 
                         'lung_lower_lobe_left', 'lung_lower_lobe_right']:
            if lung_key in organ_masks:
                lung_mets = self._detect_organ_metastasis(
                    tumor_mask, organ_masks[lung_key], colon_mask, lung_key
                )
                if lung_mets['detected']:
                    metastasis_sites.append('lung')
                    break
        
        # Classify M stage
        m_stage = "M1" if len(metastasis_sites) > 0 else "M0"
        
        details = {
            'metastasis_detected': len(metastasis_sites) > 0,
            'metastasis_sites': metastasis_sites,
            'number_of_sites': len(metastasis_sites)
        }
        
        return m_stage, details
    
    def _measure_max_diameter(self, mask: np.ndarray) -> float:
        """
        Measure maximum diameter of a 3D mask using distance transform.
        
        Args:
            mask: Binary 3D mask
            
        Returns:
            Maximum diameter in mm
        """
        if np.sum(mask) == 0:
            return 0.0
        
        # Get all tumor voxel coordinates
        coords = np.argwhere(mask > 0)
        
        if len(coords) < 2:
            return 0.0
        
        # Calculate all pairwise distances (expensive for large tumors, so sample)
        max_samples = 1000
        if len(coords) > max_samples:
            indices = np.random.choice(len(coords), max_samples, replace=False)
            coords = coords[indices]
        
        # Compute distance matrix
        from scipy.spatial.distance import pdist
        
        # Scale coordinates by voxel spacing
        scaled_coords = coords * np.array(self.spacing)
        
        # Maximum pairwise distance
        if len(scaled_coords) > 1:
            distances = pdist(scaled_coords)
            max_diameter_mm = np.max(distances)
        else:
            max_diameter_mm = 0.0
        
        return max_diameter_mm
    
    def _detect_organ_invasion(
        self,
        tumor_mask: np.ndarray,
        organ_masks: Dict[str, np.ndarray]
    ) -> bool:
        """
        Detect if tumor invades adjacent organs.
        
        Args:
            tumor_mask: Tumor mask
            organ_masks: Dictionary of organ masks
            
        Returns:
            True if invasion detected
        """
        # Dilate tumor mask slightly to detect contact
        dilated_tumor = ndimage.binary_dilation(tumor_mask, iterations=2)
        
        # Check overlap with adjacent organs (excluding colon itself)
        adjacent_organs = ['liver', 'kidney_right', 'kidney_left', 'stomach', 
                          'pancreas', 'spleen', 'urinary_bladder']
        
        for organ_name in adjacent_organs:
            if organ_name in organ_masks:
                organ_mask = organ_masks[organ_name]
                # Ensure shapes match
                if organ_mask.shape != tumor_mask.shape:
                    continue  # Skip if shapes don't match
                overlap = np.logical_and(dilated_tumor, organ_mask)
                if np.sum(overlap) > 0:
                    return True
        
        return False
    
    def _get_invaded_organs(
        self,
        tumor_mask: np.ndarray,
        organ_masks: Dict[str, np.ndarray]
    ) -> List[str]:
        """Get list of invaded organ names"""
        dilated_tumor = ndimage.binary_dilation(tumor_mask, iterations=2)
        
        invaded = []
        adjacent_organs = ['liver', 'kidney_right', 'kidney_left', 'stomach', 
                          'pancreas', 'spleen', 'urinary_bladder']
        
        for organ_name in adjacent_organs:
            if organ_name in organ_masks:
                organ_mask = organ_masks[organ_name]
                if organ_mask.shape != tumor_mask.shape:
                    continue
                overlap = np.logical_and(dilated_tumor, organ_mask)
                if np.sum(overlap) > 0:
                    invaded.append(organ_name)
        
        return invaded
    
    def _detect_lymph_nodes(
        self,
        organ_masks: Dict[str, np.ndarray],
        ct_volume: Optional[np.ndarray] = None
    ) -> List[Dict]:
        """
        Detect lymph nodes from segmentation or CT intensity.
        
        Args:
            organ_masks: Organ masks (may include lymph node regions)
            ct_volume: CT volume for intensity-based detection
            
        Returns:
            List of detected lymph node dictionaries
        """
        lymph_nodes = []
        
        # Method 1: From organ segmentation (if available)
        for key in ['lymph_node_right', 'lymph_node_left']:
            if key in organ_masks:
                # Label connected components
                labeled, num_features = ndimage.label(organ_masks[key])
                
                for i in range(1, num_features + 1):
                    node_mask = (labeled == i)
                    node_volume = np.sum(node_mask) * np.prod(self.spacing)
                    node_diameter = 2 * ((3 * node_volume) / (4 * np.pi)) ** (1/3)
                    
                    lymph_nodes.append({
                        'location': key,
                        'size_mm': node_diameter,
                        'volume_mm3': node_volume
                    })
        
        # If no lymph nodes from segmentation, return empty list
        # In production, implement intensity-based detection from CT volume
        
        return lymph_nodes
    
    def _detect_organ_metastasis(
        self,
        tumor_mask: np.ndarray,
        organ_mask: np.ndarray,
        primary_site_mask: np.ndarray,
        organ_name: str
    ) -> Dict:
        """
        Detect metastasis in a specific organ.
        
        Logic: Significant tumor volume found in non-primary organ = metastasis
        
        Args:
            tumor_mask: Detected tumor regions
            organ_mask: Target organ mask
            primary_site_mask: Primary tumor site mask (e.g., colon)
            organ_name: Name of target organ
            
        Returns:
            Dictionary with detection results
        """
        # Ensure shapes match
        if organ_mask.shape != tumor_mask.shape or primary_site_mask.shape != tumor_mask.shape:
            return {'detected': False, 'organ': organ_name}
        
        # Check for tumor in this organ
        tumor_in_organ = np.logical_and(tumor_mask, organ_mask)
        
        # If no tumor in organ, no metastasis
        if np.sum(tumor_in_organ) == 0:
            return {'detected': False, 'organ': organ_name}
        
        # Calculate volume
        metastasis_volume = np.sum(tumor_in_organ) * np.prod(self.spacing)
        
        # Check if this organ overlaps with primary site (colon)
        # If organ is part of primary site, it's not metastasis
        organ_in_primary = np.logical_and(organ_mask, primary_site_mask)
        overlap_volume = np.sum(organ_in_primary) * np.prod(self.spacing)
        
        # If substantial overlap with primary site, this is the primary tumor location
        if overlap_volume > 1000.0:  # mm³
            return {'detected': False, 'organ': organ_name}
        
        # Threshold for clinically significant metastasis
        min_metastasis_volume = 100.0  # mm³ (0.1 cm³)
        
        detected = metastasis_volume > min_metastasis_volume
        
        return {
            'detected': detected,
            'organ': organ_name,
            'volume_mm3': metastasis_volume if detected else 0.0
        }
    
    def _map_tnm_to_stage(self, t_stage: str, n_stage: str, m_stage: str) -> str:
        """
        Map TNM classification to overall cancer stage (AJCC 8th edition).
        
        Args:
            t_stage: T classification
            n_stage: N classification
            m_stage: M classification
            
        Returns:
            Overall stage (I, IIA, IIB, IIIA, IIIB, IIIC, IV)
        """
        # Stage IV: Any metastasis
        if m_stage == "M1":
            return "IV"
        
        # Stage III: Node positive
        if n_stage in ["N1", "N2"]:
            if t_stage in ["T1", "T2"]:
                if n_stage == "N1":
                    return "IIIA"
                else:  # N2
                    return "IIIB"
            else:  # T3, T4
                if n_stage == "N1":
                    return "IIIB"
                else:  # N2
                    return "IIIC"
        
        # Stage II: Node negative, advanced tumor
        if n_stage == "N0":
            if t_stage == "T3":
                return "IIA"
            elif t_stage == "T4":
                return "IIB"
            elif t_stage in ["T1", "T2"]:
                return "I"
        
        # Default fallback
        return "Unknown"
    
    def _calculate_confidence(
        self,
        t_details: Dict,
        n_details: Dict,
        m_details: Dict
    ) -> float:
        """
        Calculate overall confidence score for TNM classification.
        
        In production, this would use model confidence scores.
        For now, use heuristic based on detection quality.
        
        Args:
            t_details: T stage details
            n_details: N stage details
            m_details: M stage details
            
        Returns:
            Confidence score (0.0 - 1.0)
        """
        confidence = 0.0
        
        # T stage confidence (based on tumor detection)
        if t_details['volume_cm3'] > 0:
            confidence += 0.4
        
        # N stage confidence (based on lymph node detection)
        if n_details['total_lymph_nodes'] > 0 or n_details['suspicious_nodes'] == 0:
            confidence += 0.3
        
        # M stage confidence (always have organ masks)
        confidence += 0.3
        
        return min(confidence, 1.0)


def main():
    """Example usage"""
    # Create dummy data for demonstration
    tumor_mask = np.random.rand(100, 256, 256) > 0.95
    organ_masks = {
        'colon': np.random.rand(100, 256, 256) > 0.9,
        'liver': np.random.rand(100, 256, 256) > 0.95
    }
    
    # Initialize engine
    engine = TNMStagingEngine(spacing=(2.0, 1.0, 1.0))
    
    # Classify
    result = engine.classify_tnm(tumor_mask, organ_masks)
    
    print(f"TNM Classification: {result.T} {result.N} {result.M}")
    print(f"Overall Stage: {result.stage}")
    print(f"Confidence: {result.confidence:.2f}")
    print(f"\nDetails:")
    print(f"  T: {result.details['T']}")
    print(f"  N: {result.details['N']}")
    print(f"  M: {result.details['M']}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
