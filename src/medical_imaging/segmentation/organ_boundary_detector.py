"""
Organ Boundary Detection for CT images.
Extracts and analyzes organ boundaries for tumor localization and staging.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from scipy import ndimage
from skimage import measure, morphology
import cv2


class OrganBoundaryDetector:
    """
    Detect and analyze organ boundaries from segmentation masks.
    
    Provides:
    - Boundary extraction (2D and 3D)
    - Distance maps for tumor-organ proximity analysis
    - Multi-organ boundary overlap detection
    - Organ surface area calculation
    """
    
    def __init__(self, connectivity: int = 1):
        """
        Initialize boundary detector.
        
        Args:
            connectivity: Connectivity for boundary detection (1 or 2 for 3D)
        """
        self.connectivity = connectivity
    
    def extract_boundaries_3d(
        self,
        mask: np.ndarray,
        thickness: int = 1
    ) -> np.ndarray:
        """
        Extract 3D organ boundary.
        
        Args:
            mask: 3D binary organ mask (D, H, W)
            thickness: Boundary thickness in pixels
            
        Returns:
            Binary boundary mask
        """
        # Erode mask
        eroded = ndimage.binary_erosion(
            mask,
            iterations=thickness,
            structure=ndimage.generate_binary_structure(3, self.connectivity)
        )
        
        # Boundary = original - eroded
        boundary = mask.astype(bool) & ~eroded
        
        return boundary.astype(np.uint8)
    
    def extract_boundaries_2d_per_slice(
        self,
        mask_3d: np.ndarray,
        thickness: int = 2
    ) -> np.ndarray:
        """
        Extract 2D boundaries for each axial slice.
        
        Args:
            mask_3d: 3D organ mask (D, H, W)
            thickness: Boundary thickness in pixels
            
        Returns:
            3D boundary mask (same shape as input)
        """
        boundary_3d = np.zeros_like(mask_3d, dtype=np.uint8)
        
        for slice_idx in range(mask_3d.shape[0]):
            slice_mask = mask_3d[slice_idx]
            
            if np.any(slice_mask):
                # Find contours
                contours, _ = cv2.findContours(
                    slice_mask.astype(np.uint8),
                    cv2.RETR_EXTERNAL,
                    cv2.CHAIN_APPROX_SIMPLE
                )
                
                # Draw boundaries
                cv2.drawContours(
                    boundary_3d[slice_idx],
                    contours,
                    -1,
                    1,
                    thickness=thickness
                )
        
        return boundary_3d
    
    def compute_distance_map(
        self,
        mask: np.ndarray,
        spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0)
    ) -> np.ndarray:
        """
        Compute Euclidean distance transform from organ boundary.
        
        Useful for:
        - Measuring tumor distance to organs
        - Detecting organ invasion (T4 staging)
        
        Args:
            mask: 3D binary organ mask
            spacing: Voxel spacing in mm (z, y, x)
            
        Returns:
            Distance map in mm (negative inside organ, positive outside)
        """
        # Distance from outside to boundary (positive outside)
        dist_outside = ndimage.distance_transform_edt(
            ~mask.astype(bool),
            sampling=spacing
        )
        
        # Distance from inside to boundary (negative inside)
        dist_inside = ndimage.distance_transform_edt(
            mask.astype(bool),
            sampling=spacing
        )
        
        # Combine: negative inside, positive outside
        distance_map = np.where(mask, -dist_inside, dist_outside)
        
        return distance_map
    
    def detect_tumor_organ_proximity(
        self,
        tumor_mask: np.ndarray,
        organ_mask: np.ndarray,
        spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
        invasion_threshold_mm: float = 5.0
    ) -> Dict[str, any]:
        """
        Analyze tumor proximity to organ.
        
        Args:
            tumor_mask: 3D binary tumor mask
            organ_mask: 3D binary organ mask
            spacing: Voxel spacing in mm
            invasion_threshold_mm: Distance threshold for invasion detection
            
        Returns:
            Dictionary with proximity metrics:
            - min_distance_mm: Minimum distance tumor to organ
            - max_distance_mm: Maximum distance
            - mean_distance_mm: Mean distance
            - is_invading: Boolean, True if tumor invades organ
            - contact_fraction: Fraction of tumor in contact with organ
        """
        # Compute distance map from organ
        dist_map = self.compute_distance_map(organ_mask, spacing)
        
        # Sample distances at tumor locations
        tumor_distances = dist_map[tumor_mask > 0]
        
        if len(tumor_distances) == 0:
            return {
                'min_distance_mm': np.inf,
                'max_distance_mm': np.inf,
                'mean_distance_mm': np.inf,
                'is_invading': False,
                'contact_fraction': 0.0
            }
        
        min_dist = float(np.min(tumor_distances))
        max_dist = float(np.max(tumor_distances))
        mean_dist = float(np.mean(tumor_distances))
        
        # Check if tumor invades organ (negative distance = inside organ)
        is_invading = min_dist < -invasion_threshold_mm
        
        # Fraction of tumor voxels in contact with organ
        contact_fraction = float(np.sum(np.abs(tumor_distances) < invasion_threshold_mm) / len(tumor_distances))
        
        return {
            'min_distance_mm': min_dist,
            'max_distance_mm': max_dist,
            'mean_distance_mm': mean_dist,
            'is_invading': is_invading,
            'contact_fraction': contact_fraction
        }
    
    def find_multi_organ_overlaps(
        self,
        masks: Dict[str, np.ndarray]
    ) -> Dict[str, List[str]]:
        """
        Find overlapping regions between organs (useful for detecting errors).
        
        Args:
            masks: Dictionary mapping organ names to masks
            
        Returns:
            Dictionary mapping organ names to list of overlapping organs
        """
        overlaps = {organ: [] for organ in masks.keys()}
        
        organ_names = list(masks.keys())
        
        for i, organ1 in enumerate(organ_names):
            for organ2 in organ_names[i+1:]:
                overlap = masks[organ1] & masks[organ2]
                if np.any(overlap):
                    overlaps[organ1].append(organ2)
                    overlaps[organ2].append(organ1)
        
        return overlaps
    
    def compute_organ_surface_area(
        self,
        mask: np.ndarray,
        spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0)
    ) -> float:
        """
        Compute organ surface area using marching cubes.
        
        Args:
            mask: 3D binary organ mask
            spacing: Voxel spacing in mm (z, y, x)
            
        Returns:
            Surface area in mm²
        """
        try:
            # Use marching cubes to generate mesh
            verts, faces, normals, values = measure.marching_cubes(
                mask,
                level=0.5,
                spacing=spacing
            )
            
            # Compute surface area from triangular mesh
            surface_area = 0.0
            for face in faces:
                # Get triangle vertices
                v0, v1, v2 = verts[face]
                
                # Compute triangle area using cross product
                edge1 = v1 - v0
                edge2 = v2 - v0
                cross = np.cross(edge1, edge2)
                area = 0.5 * np.linalg.norm(cross)
                surface_area += area
            
            return surface_area
        
        except Exception as e:
            # Fallback: approximate using boundary voxels
            boundary = self.extract_boundaries_3d(mask, thickness=1)
            voxel_surface = spacing[1] * spacing[2]  # Approximate voxel face area
            surface_area = np.sum(boundary) * voxel_surface
            return surface_area
    
    def get_organ_bounding_box(
        self,
        mask: np.ndarray,
        padding: int = 5
    ) -> Tuple[slice, slice, slice]:
        """
        Get 3D bounding box around organ.
        
        Args:
            mask: 3D binary organ mask
            padding: Padding in voxels
            
        Returns:
            Tuple of slices (z_slice, y_slice, x_slice)
        """
        coords = np.argwhere(mask > 0)
        
        if len(coords) == 0:
            return (slice(0, 1), slice(0, 1), slice(0, 1))
        
        z_min, y_min, x_min = coords.min(axis=0)
        z_max, y_max, x_max = coords.max(axis=0)
        
        # Add padding
        z_min = max(0, z_min - padding)
        y_min = max(0, y_min - padding)
        x_min = max(0, x_min - padding)
        z_max = min(mask.shape[0], z_max + padding + 1)
        y_max = min(mask.shape[1], y_max + padding + 1)
        x_max = min(mask.shape[2], x_max + padding + 1)
        
        return (
            slice(z_min, z_max),
            slice(y_min, y_max),
            slice(x_min, x_max)
        )
    
    def visualize_boundaries_on_slice(
        self,
        ct_slice: np.ndarray,
        boundary_slice: np.ndarray,
        color: Tuple[int, int, int] = (255, 0, 0)
    ) -> np.ndarray:
        """
        Overlay organ boundary on CT slice for visualization.
        
        Args:
            ct_slice: 2D CT slice (H, W)
            boundary_slice: 2D binary boundary mask
            color: RGB color for boundary
            
        Returns:
            RGB image with boundary overlay
        """
        # Normalize CT to 0-255
        ct_norm = ((ct_slice - ct_slice.min()) / (ct_slice.max() - ct_slice.min()) * 255).astype(np.uint8)
        
        # Convert to RGB
        rgb = cv2.cvtColor(ct_norm, cv2.COLOR_GRAY2RGB)
        
        # Overlay boundary
        rgb[boundary_slice > 0] = color
        
        return rgb
