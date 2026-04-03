"""
Post-processing Utilities for Medical Segmentation
Connected components, morphological operations, hole filling
"""

import numpy as np
from scipy import ndimage
from skimage import morphology, measure
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)


def remove_small_components(
    mask: np.ndarray,
    min_size: int = 1000,
    max_size: Optional[int] = None
) -> np.ndarray:
    """
    Remove small connected components
    
    Args:
        mask: Binary segmentation mask (D, H, W)
        min_size: Minimum component size in voxels
        max_size: Maximum component size (optional)
    
    Returns:
        cleaned_mask: Mask with small components removed
    """
    # Label connected components
    labeled, num_features = ndimage.label(mask)
    
    # Calculate component sizes
    component_sizes = np.bincount(labeled.ravel())
    
    # Create mask for valid components
    too_small = component_sizes < min_size
    too_small[0] = False  # Keep background
    
    if max_size is not None:
        too_large = component_sizes > max_size
        too_large[0] = False
        invalid = too_small | too_large
    else:
        invalid = too_small
    
    # Remove invalid components
    remove_mask = invalid[labeled]
    cleaned = mask.copy()
    cleaned[remove_mask] = 0
    
    logger.info(f"Removed {invalid.sum()} components (kept {num_features - invalid.sum()})")
    
    return cleaned


def fill_holes(mask: np.ndarray, max_hole_size: int = 500) -> np.ndarray:
    """
    Fill holes in segmentation mask
    
    Args:
        mask: Binary segmentation mask (D, H, W)
        max_hole_size: Maximum hole size to fill
    
    Returns:
        filled_mask: Mask with holes filled
    """
    filled = ndimage.binary_fill_holes(mask).astype(mask.dtype)
    
    # Only fill small holes
    if max_hole_size:
        holes = filled & ~mask
        labeled_holes, _ = ndimage.label(holes)
        hole_sizes = np.bincount(labeled_holes.ravel())
        
        # Remove large holes (likely not actual holes)
        large_holes = hole_sizes > max_hole_size
        large_holes[0] = False
        keep_holes_mask = large_holes[labeled_holes]
        
        filled[keep_holes_mask] = 0
    
    return filled


def morphological_cleanup(
    mask: np.ndarray,
    operation: str = "close",
    kernel_size: int = 3
) -> np.ndarray:
    """
    Apply morphological operations to clean up mask
    
    Args:
        mask: Binary segmentation mask (D, H, W)
        operation: "open", "close", "erode", or "dilate"
        kernel_size: Size of structuring element
    
    Returns:
        cleaned_mask: Morphologically processed mask
    """
    # Create 3D ball structuring element
    struct_elem = morphology.ball(kernel_size)
    
    if operation == "open":
        cleaned = ndimage.binary_opening(mask, structure=struct_elem)
    elif operation == "close":
        cleaned = ndimage.binary_closing(mask, structure=struct_elem)
    elif operation == "erode":
        cleaned = ndimage.binary_erosion(mask, structure=struct_elem)
    elif operation == "dilate":
        cleaned = ndimage.binary_dilation(mask, structure=struct_elem)
    else:
        raise ValueError(f"Unknown operation: {operation}")
    
    return cleaned.astype(mask.dtype)


def get_largest_component(mask: np.ndarray) -> np.ndarray:
    """
    Keep only the largest connected component
    
    Args:
        mask: Binary segmentation mask (D, H, W)
    
    Returns:
        largest_mask: Mask with only largest component
    """
    labeled, num_features = ndimage.label(mask)
    
    if num_features == 0:
        return mask
    
    # Find largest component
    component_sizes = np.bincount(labeled.ravel())
    component_sizes[0] = 0  # Ignore background
    largest_label = component_sizes.argmax()
    
    # Keep only largest
    largest_mask = (labeled == largest_label).astype(mask.dtype)
    
    logger.info(f"Kept largest component ({component_sizes[largest_label]} voxels)")
    
    return largest_mask


def postprocess_segmentation(
    mask: np.ndarray,
    min_size: int = 1000,
    max_size: Optional[int] = 50000,
    fill_holes_size: int = 500,
    apply_morphology: bool = True,
    keep_largest_only: bool = False
) -> Tuple[np.ndarray, dict]:
    """
    Complete post-processing pipeline for segmentation masks
    
    Pipeline:
    1. Remove small components
    2. Fill holes
    3. Morphological cleanup (closing then opening)
    4. Optional: Keep only largest component
    
    Args:
        mask: Raw segmentation mask (D, H, W)
        min_size: Minimum component size
        max_size: Maximum component size
        fill_holes_size: Maximum hole size to fill
        apply_morphology: Apply morphological operations
        keep_largest_only: Keep only the largest component
    
    Returns:
        cleaned_mask: Post-processed mask
        stats: Processing statistics
    """
    logger.info("Starting post-processing pipeline...")
    
    original_volume = mask.sum()
    stats = {'original_volume': int(original_volume)}
    
    # Step 1: Remove small/large components
    mask = remove_small_components(mask, min_size=min_size, max_size=max_size)
    stats['after_size_filter'] = int(mask.sum())
    
    # Step 2: Fill holes
    if fill_holes_size > 0:
        mask = fill_holes(mask, max_hole_size=fill_holes_size)
        stats['after_hole_fill'] = int(mask.sum())
    
    # Step 3: Morphological cleanup
    if apply_morphology:
        # Close to connect nearby regions
        mask = morphological_cleanup(mask, operation="close", kernel_size=3)
        # Open to remove small protrusions
        mask = morphological_cleanup(mask, operation="open", kernel_size=2)
        stats['after_morphology'] = int(mask.sum())
    
    # Step 4: Keep largest component (optional)
    if keep_largest_only:
        mask = get_largest_component(mask)
        stats['after_largest'] = int(mask.sum())
    
    final_volume = mask.sum()
    stats['final_volume'] = int(final_volume)
    stats['volume_change'] = float((final_volume - original_volume) / original_volume * 100)
    
    logger.info(f"Post-processing complete. Volume change: {stats['volume_change']:.1f}%")
    
    return mask, stats


def smooth_boundaries(
    mask: np.ndarray,
    sigma: float = 1.0
) -> np.ndarray:
    """
    Smooth segmentation boundaries using Gaussian filter
    
    Args:
        mask: Binary segmentation mask (D, H, W)
        sigma: Gaussian kernel sigma
    
    Returns:
        smoothed_mask: Mask with smoothed boundaries
    """
    # Apply Gaussian smoothing
    smoothed = ndimage.gaussian_filter(mask.astype(float), sigma=sigma)
    
    # Re-threshold
    smoothed_mask = (smoothed > 0.5).astype(mask.dtype)
    
    return smoothed_mask


# Test post-processing
if __name__ == "__main__":
    print("Testing post-processing utilities...")
    
    # Create dummy mask with noise
    mask = np.zeros((64, 128, 128), dtype=np.uint8)
    
    # Add main tumor
    mask[20:40, 50:80, 50:80] = 1
    
    # Add small noise
    mask[10:12, 10:12, 10:12] = 1
    mask[50:52, 100:102, 100:102] = 1
    
    # Add hole
    mask[28:32, 63:67, 63:67] = 0
    
    print(f"Original volume: {mask.sum()} voxels")
    
    # Apply post-processing
    cleaned, stats = postprocess_segmentation(
        mask,
        min_size=100,
        fill_holes_size=500,
        apply_morphology=True
    )
    
    print(f"\nPost-processing stats:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    print(f"\nFinal volume: {cleaned.sum()} voxels")
    print("[OK] Post-processing test passed!")
