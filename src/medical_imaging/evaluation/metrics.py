"""
Segmentation Evaluation Metrics
Dice, Hausdorff, Surface Dice, Volume Similarity
"""

import numpy as np
from scipy import ndimage
from scipy.spatial import distance
from typing import Dict, Tuple
import logging

logger = logging.getLogger(__name__)


def calculate_dice(
    pred: np.ndarray,
    gt: np.ndarray,
    smooth: float = 1e-5
) -> float:
    """
    Calculate Dice coefficient
    
    Dice = 2 * |A ∩ B| / (|A| + |B|)
    
    Args:
        pred: Prediction mask (D, H, W)
        gt: Ground truth mask (D, H, W)
        smooth: Smoothing factor
    
    Returns:
        dice: Dice coefficient [0, 1]
    """
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    
    intersection = (pred & gt).sum()
    union = pred.sum() + gt.sum()
    
    dice = (2.0 * intersection + smooth) / (union + smooth)
    
    return float(dice)


def calculate_iou(
    pred: np.ndarray,
    gt: np.ndarray,
    smooth: float = 1e-5
) -> float:
    """
    Calculate Intersection over Union (IoU / Jaccard)
    
    IoU = |A ∩ B| / |A ∪ B|
    
    Args:
        pred: Prediction mask
        gt: Ground truth mask
        smooth: Smoothing factor
    
    Returns:
        iou: IoU score [0, 1]
    """
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    
    intersection = (pred & gt).sum()
    union = (pred | gt).sum()
    
    iou = (intersection + smooth) / (union + smooth)
    
    return float(iou)


def calculate_hausdorff(
    pred: np.ndarray,
    gt: np.ndarray,
    percentile: int = 95
) -> float:
    """
    Calculate Hausdorff distance
    
    Measures maximum distance from pred surface to GT surface
    
    Args:
        pred: Prediction mask
        gt: Ground truth mask
        percentile: Percentile for robust HD (95 is standard)
    
    Returns:
        hd: Hausdorff distance in voxels
    """
    # Get surface points
    pred_surface = get_surface_points(pred)
    gt_surface = get_surface_points(gt)
    
    if len(pred_surface) == 0 or len(gt_surface) == 0:
        return float('inf')
    
    # Calculate directed Hausdorff distances
    dist_pred_to_gt = distance.cdist(pred_surface, gt_surface, metric='euclidean')
    dist_gt_to_pred = distance.cdist(gt_surface, pred_surface, metric='euclidean')
    
    # Get minimum distances
    min_dist_pred_to_gt = dist_pred_to_gt.min(axis=1)
    min_dist_gt_to_pred = dist_gt_to_pred.min(axis=1)
    
    # Get percentile
    hd_pred_to_gt = np.percentile(min_dist_pred_to_gt, percentile)
    hd_gt_to_pred = np.percentile(min_dist_gt_to_pred, percentile)
    
    # Hausdorff is the maximum of the two directed distances
    hd = max(hd_pred_to_gt, hd_gt_to_pred)
    
    return float(hd)


def get_surface_points(mask: np.ndarray) -> np.ndarray:
    """
    Extract surface points from binary mask
    
    Args:
        mask: Binary mask (D, H, W)
    
    Returns:
        surface_points: Array of surface coordinates (N, 3)
    """
    # Erode mask
    eroded = ndimage.binary_erosion(mask)
    
    # Surface is mask - eroded
    surface = mask & ~eroded
    
    # Get coordinates
    surface_points = np.argwhere(surface)
    
    return surface_points


def calculate_surface_dice(
    pred: np.ndarray,
    gt: np.ndarray,
    tolerance: float = 2.0
) -> float:
    """
    Calculate Surface Dice coefficient
    
    Measures agreement of surfaces within tolerance distance
    
    Args:
        pred: Prediction mask
        gt: Ground truth mask
        tolerance: Tolerance distance in voxels
    
    Returns:
        surface_dice: Surface Dice [0, 1]
    """
    # Get surface points
    pred_surface = get_surface_points(pred)
    gt_surface = get_surface_points(gt)
    
    if len(pred_surface) == 0 or len(gt_surface) == 0:
        return 0.0
    
    # Calculate distances
    dist_pred_to_gt = distance.cdist(pred_surface, gt_surface, metric='euclidean')
    dist_gt_to_pred = distance.cdist(gt_surface, pred_surface, metric='euclidean')
    
    # Count surface points within tolerance
    pred_within_tolerance = (dist_pred_to_gt.min(axis=1) <= tolerance).sum()
    gt_within_tolerance = (dist_gt_to_pred.min(axis=1) <= tolerance).sum()
    
    # Surface Dice
    surface_dice = (pred_within_tolerance + gt_within_tolerance) / (len(pred_surface) + len(gt_surface))
    
    return float(surface_dice)


def calculate_volume_similarity(
    pred: np.ndarray,
    gt: np.ndarray
) -> float:
    """
    Calculate volume similarity
    
    VS = 1 - |V_pred - V_gt| / (V_pred + V_gt)
    
    Args:
        pred: Prediction mask
        gt: Ground truth mask
    
    Returns:
        vs: Volume similarity [0, 1]
    """
    vol_pred = pred.sum()
    vol_gt = gt.sum()
    
    if vol_pred + vol_gt == 0:
        return 1.0
    
    vs = 1.0 - abs(vol_pred - vol_gt) / (vol_pred + vol_gt)
    
    return float(vs)


class SegmentationMetrics:
    """
    Comprehensive segmentation metrics calculator
    
    Metrics:
    - Dice coefficient
    - IoU (Jaccard)
    - Hausdorff distance (95th percentile)
    - Surface Dice
    - Volume similarity
    - Sensitivity (Recall)
    - Specificity
    - Precision
    """
    
    def __init__(self):
        self.metrics = {}
    
    def calculate_all(
        self,
        pred: np.ndarray,
        gt: np.ndarray,
        spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0)
    ) -> Dict[str, float]:
        """
        Calculate all metrics
        
        Args:
            pred: Prediction mask (D, H, W)
            gt: Ground truth mask (D, H, W)
            spacing: Voxel spacing (mm)
        
        Returns:
            metrics: Dictionary of metrics
        """
        pred_bool = pred.astype(bool)
        gt_bool = gt.astype(bool)
        
        # Overlap metrics
        self.metrics['dice'] = calculate_dice(pred, gt)
        self.metrics['iou'] = calculate_iou(pred, gt)
        
        # Distance metrics
        self.metrics['hausdorff_95'] = calculate_hausdorff(pred, gt, percentile=95)
        self.metrics['surface_dice'] = calculate_surface_dice(pred, gt, tolerance=2.0)
        
        # Volume metrics
        self.metrics['volume_similarity'] = calculate_volume_similarity(pred, gt)
        self.metrics['volume_pred'] = float(pred.sum() * np.prod(spacing))  # mm³
        self.metrics['volume_gt'] = float(gt.sum() * np.prod(spacing))  # mm³
        
        # Classification metrics
        tp = (pred_bool & gt_bool).sum()
        fp = (pred_bool & ~gt_bool).sum()
        fn = (~pred_bool & gt_bool).sum()
        tn = (~pred_bool & ~gt_bool).sum()
        
        self.metrics['sensitivity'] = float(tp / (tp + fn + 1e-10))  # Recall
        self.metrics['specificity'] = float(tn / (tn + fp + 1e-10))
        self.metrics['precision'] = float(tp / (tp + fp + 1e-10))
        
        # F1 score (harmonic mean of precision and recall)
        self.metrics['f1_score'] = float(
            2 * (self.metrics['precision'] * self.metrics['sensitivity']) /
            (self.metrics['precision'] + self.metrics['sensitivity'] + 1e-10)
        )
        
        return self.metrics
    
    def print_metrics(self):
        """Print metrics in readable format"""
        logger.info("=" * 60)
        logger.info("Segmentation Metrics")
        logger.info("=" * 60)
        
        for name, value in self.metrics.items():
            if 'volume' in name:
                logger.info(f"{name:20s}: {value:.2f} mm³")
            elif 'hausdorff' in name:
                logger.info(f"{name:20s}: {value:.2f} voxels")
            else:
                logger.info(f"{name:20s}: {value:.4f}")
        
        logger.info("=" * 60)


# Test metrics
if __name__ == "__main__":
    print("Testing segmentation metrics...")
    
    # Create dummy masks
    pred = np.zeros((64, 128, 128), dtype=np.uint8)
    gt = np.zeros((64, 128, 128), dtype=np.uint8)
    
    # Add overlapping regions
    pred[20:35, 50:75, 50:75] = 1
    gt[22:37, 52:77, 52:77] = 1
    
    # Calculate metrics
    metrics_calc = SegmentationMetrics()
    metrics = metrics_calc.calculate_all(pred, gt)
    
    print("\nMetrics:")
    for name, value in metrics.items():
        print(f"  {name}: {value:.4f}")
    
    print(f"\nDice score: {metrics['dice']:.4f}")
    print("[OK] Metrics test passed!")
