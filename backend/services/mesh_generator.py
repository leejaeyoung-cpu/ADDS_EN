"""
3D Mesh Generator Service

Converts medical image segmentation masks (NIfTI format) to 3D meshes
suitable for web-based visualization using AMI.js or VTK.js.
"""

import numpy as np
import nibabel as nib
from pathlib import Path
from skimage import measure
from scipy import ndimage
import json
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class MeshGenerator:
    """Generate 3D meshes from segmentation masks for web visualization"""
    
    def __init__(self, cache_dir: Optional[Path] = None):
        """
        Initialize mesh generator
        
        Args:
            cache_dir: Directory to cache generated meshes (optional)
        """
        self.cache_dir = cache_dir or Path("backend/outputs/mesh_cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def generate_organ_mesh(
        self, 
        mask_path: Path, 
        spacing: Tuple[float, float, float],
        organ_name: str = "organ",
        simplify: bool = True,
        target_faces: int = 10000
    ) -> Dict:
        """
        Generate 3D mesh from organ segmentation mask
        
        Args:
            mask_path: Path to NIfTI mask file
            spacing: Voxel spacing (x, y, z) in mm
            organ_name: Name of the organ for labeling
            simplify: Whether to decimate mesh for web performance
            target_faces: Target number of faces after decimation
            
        Returns:
            Dictionary with mesh data:
            {
                'vertices': [[x, y, z], ...],
                'faces': [[v1, v2, v3], ...],
                'normals': [[nx, ny, nz], ...],
                'name': str,
                'bounds': {'xmin': float, 'xmax': float, ...}
            }
        """
        logger.info(f"Generating mesh for {organ_name} from {mask_path}")
        
        # Load mask
        if not mask_path.exists():
            raise FileNotFoundError(f"Mask file not found: {mask_path}")
        
        nii = nib.load(mask_path)
        mask = nii.get_fdata().astype(bool)
        
        logger.info(f"Mask shape: {mask.shape}, voxels: {mask.sum():,}")
        
        # Generate mesh using marching cubes
        vertices, faces, normals, values = self._marching_cubes(
            mask, spacing, smooth=True
        )
        
        if vertices is None:
            logger.error(f"Failed to generate mesh for {organ_name}")
            return None
        
        logger.info(f"Initial mesh: {len(vertices)} vertices, {len(faces)} faces")
        
        # Simplify mesh if requested
        if simplify and len(faces) > target_faces:
            vertices, faces, normals = self._simplify_mesh(
                vertices, faces, normals, target_faces
            )
            logger.info(f"Simplified mesh: {len(vertices)} vertices, {len(faces)} faces")
        
        # Calculate bounds
        bounds = self._calculate_bounds(vertices)
        
        # Convert to JSON-serializable format
        mesh_data = {
            'vertices': vertices.tolist(),
            'faces': faces.tolist(),
            'normals': normals.tolist(),
            'name': organ_name,
            'bounds': bounds,
            'num_vertices': len(vertices),
            'num_faces': len(faces)
        }
        
        return mesh_data
    
    def generate_tumor_meshes(
        self,
        labeled_volume_path: Path,
        spacing: Tuple[float, float, float],
        min_volume_mm3: float = 100.0,
        max_tumors: int = 20
    ) -> List[Dict]:
        """
        Generate meshes for all detected tumors
        
        Args:
            labeled_volume_path: Path to labeled tumor volume (NIfTI)
            spacing: Voxel spacing (x, y, z) in mm
            min_volume_mm3: Minimum tumor volume to include
            max_tumors: Maximum number of tumors to generate meshes for
            
        Returns:
            List of mesh dictionaries, sorted by volume (largest first)
        """
        logger.info(f"Generating tumor meshes from {labeled_volume_path}")
        
        # Load labeled volume
        nii = nib.load(labeled_volume_path)
        labeled_volume = nii.get_fdata().astype(int)
        
        num_tumors = int(labeled_volume.max())
        logger.info(f"Found {num_tumors} labeled tumors")
        
        voxel_volume_mm3 = spacing[0] * spacing[1] * spacing[2]
        
        tumor_meshes = []
        
        for label in range(1, num_tumors + 1):
            tumor_mask = (labeled_volume == label)
            volume_mm3 = tumor_mask.sum() * voxel_volume_mm3
            
            # Skip small tumors
            if volume_mm3 < min_volume_mm3:
                continue
            
            # Generate mesh
            vertices, faces, normals, _ = self._marching_cubes(
                tumor_mask, spacing, smooth=True
            )
            
            if vertices is None:
                continue
            
            # Calculate centroid
            centroid = vertices.mean(axis=0)
            
            # Simplify tumor mesh (smaller target for performance)
            if len(faces) > 2000:
                vertices, faces, normals = self._simplify_mesh(
                    vertices, faces, normals, target_faces=2000
                )
            
            mesh_data = {
                'vertices': vertices.tolist(),
                'faces': faces.tolist(),
                'normals': normals.tolist(),
                'label': int(label),
                'volume_mm3': float(volume_mm3),
                'volume_ml': float(volume_mm3 / 1000),
                'centroid': centroid.tolist(),
                'num_vertices': len(vertices),
                'num_faces': len(faces)
            }
            
            tumor_meshes.append(mesh_data)
        
        # Sort by volume (largest first)
        tumor_meshes.sort(key=lambda x: x['volume_mm3'], reverse=True)
        
        # Limit to max_tumors
        tumor_meshes = tumor_meshes[:max_tumors]
        
        logger.info(f"Generated {len(tumor_meshes)} tumor meshes")
        
        return tumor_meshes
    
    def generate_tumor_meshes_perfect(
        self,
        tumor_masks_npz: Path,
        tumor_metadata_json: Path,
        spacing: Tuple[float, float, float],
        max_tumors: int = 20
    ) -> List[Dict]:
        """
        Generate meshes from perfect reconstruction masks with enhanced metadata
        
        Args:
            tumor_masks_npz: Path to NPZ file with perfect 3D masks
            tumor_metadata_json: Path to enhanced measurements JSON
            spacing: Voxel spacing (x, y, z) in mm
            max_tumors: Maximum number of tumors to generate meshes for
            
        Returns:
            List of mesh dictionaries with enhanced metadata:
            - vertices, faces, normals (mesh data)
            - tumor_id, volume, diameter (basic metrics)
            - surface_area, sphericity, elongation (shape metrics)
            - inside_colon, distance_to_colon (location metrics)
        """
        logger.info(f"Generating PERFECT tumor meshes from {tumor_masks_npz}")
        
        # Load perfect masks
        masks_npz = np.load(tumor_masks_npz)
        logger.info(f"Loaded {len(masks_npz.files)} perfect masks")
        
        # Load enhanced metadata
        with open(tumor_metadata_json) as f:
            metadata = json.load(f)
        
        tumors_metadata = {t['tumor_id']: t for t in metadata['tumors']}
        logger.info(f"Loaded metadata for {len(tumors_metadata)} tumors")
        
        tumor_meshes = []
        
        # Generate mesh for each tumor
        for tumor_id, tumor_meta in tumors_metadata.items():
            mask_key = f'tumor_{tumor_id}'
            
            if mask_key not in masks_npz:
                logger.warning(f"Mask not found for tumor {tumor_id}")
                continue
            
            tumor_mask = masks_npz[mask_key]
            
            # Generate mesh
            vertices, faces, normals, _ = self._marching_cubes(
                tumor_mask, spacing, smooth=True
            )
            
            if vertices is None:
                logger.warning(f"Failed to generate mesh for tumor {tumor_id}")
                continue
            
            # Calculate centroid from mesh vertices
            centroid = vertices.mean(axis=0)
            
            # Simplify mesh for web performance
            if len(faces) > 2000:
                vertices, faces, normals = self._simplify_mesh(
                    vertices, faces, normals, target_faces=2000
                )
            
            # Build enhanced mesh data
            distance_info = tumor_meta.get('distance_to_colon', {})
            
            mesh_data = {
                # Mesh geometry
                'vertices': vertices.tolist(),
                'faces': faces.tolist(),
                'normals': normals.tolist(),
                'centroid': centroid.tolist(),
                'num_vertices': len(vertices),
                'num_faces': len(faces),
                
                # Tumor identification
                'tumor_id': int(tumor_id),
                
                # Volume metrics
                'volume_mm3': float(tumor_meta['volume_mm3']),
                'volume_ml': float(tumor_meta['volume_mm3'] / 1000),
                'diameter_mm': float(tumor_meta.get('max_diameter_mm', 0)),
                
                # Shape metrics (NEW!)
                'surface_area_mm2': float(tumor_meta.get('surface_area_mm2', 0)),
                'sphericity': float(tumor_meta.get('sphericity', 0)),
               'elongation': float(tumor_meta.get('elongation', 0)),
                'flatness': float(tumor_meta.get('flatness', 0)),
                
                # Location metrics (NEW!)
                'inside_colon': bool(distance_info.get('inside_colon', True)),
                'distance_to_colon_mm': float(distance_info.get('min_distance_mm', 0))
            }
            
            tumor_meshes.append(mesh_data)
        
        # Sort by volume (largest first)
        tumor_meshes.sort(key=lambda x: x['volume_mm3'], reverse=True)
        
        # Limit to max_tumors
        tumor_meshes = tumor_meshes[:max_tumors]
        
        logger.info(f"Generated {len(tumor_meshes)} PERFECT tumor meshes with enhanced metadata")
        
        return tumor_meshes

    
    def _marching_cubes(
        self,
        mask: np.ndarray,
        spacing: Tuple[float, float, float],
        smooth: bool = True
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Apply marching cubes algorithm to generate mesh
        
        Args:
            mask: Binary 3D mask
            spacing: Voxel spacing
            smooth: Whether to smooth mask before extraction
            
        Returns:
            Tuple of (vertices, faces, normals, values) or (None, None, None, None) on failure
        """
        try:
            if smooth:
                # Smooth the mask slightly for better surface
                mask_smooth = ndimage.gaussian_filter(mask.astype(float), sigma=1.0)
                level = 0.5
            else:
                mask_smooth = mask.astype(float)
                level = 0.5
            
            # Apply marching cubes
            vertices, faces, normals, values = measure.marching_cubes(
                mask_smooth,
                level=level,
                spacing=spacing,
                step_size=1  # Full resolution
            )
            
            return vertices, faces, normals, values
            
        except Exception as e:
            logger.error(f"Marching cubes failed: {e}")
            return None, None, None, None
    
    def _simplify_mesh(
        self,
        vertices: np.ndarray,
        faces: np.ndarray,
        normals: np.ndarray,
        target_faces: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Simplify mesh by reducing face count
        
        Uses uniform face sampling to reduce mesh complexity
        while preserving overall shape.
        
        Args:
            vertices: Nx3 array of vertices
            faces: Mx3 array of face indices
            normals: Mx3 array of face normals (ignored, will be recomputed)
            target_faces: Desired number of faces
            
        Returns:
            Tuple of (simplified_vertices, simplified_faces, simplified_normals)
        """
        current_faces = len(faces)
        if current_faces <= target_faces:
            return vertices, faces, normals
        
        # Simple decimation: keep every nth face
        decimation_ratio = target_faces / current_faces
        step = max(1, int(1 / decimation_ratio))
        
        # Select faces uniformly
        selected_indices = np.arange(0, len(faces), step)[:target_faces]
        new_faces = faces[selected_indices]
        
        # Remove unused vertices
        used_vertices = np.unique(new_faces.flatten())
        vertex_map = {old_idx: new_idx for new_idx, old_idx in enumerate(used_vertices)}
        
        new_vertices = vertices[used_vertices]
        new_faces = np.array([[vertex_map[v] for v in face] for face in new_faces])
        
        # Recompute normals for the new faces
        new_normals = self._compute_face_normals(new_vertices, new_faces)
        
        return new_vertices, new_faces, new_normals
    
    def _compute_face_normals(self, vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
        """
        Compute face normals from vertices and faces
        
        Args:
            vertices: Nx3 array of vertices
            faces: Mx3 array of face indices
            
        Returns:
            Mx3 array of face normals
        """
        # Get vertices for each face
        v0 = vertices[faces[:, 0]]
        v1 = vertices[faces[:, 1]]
        v2 = vertices[faces[:, 2]]
        
        # Compute face normals using cross product
        edge1 = v1 - v0
        edge2 = v2 - v0
        normals = np.cross(edge1, edge2)
        
        # Normalize
        norms = np.linalg.norm(normals, axis=1, keepdims=True)
        normals = normals / (norms + 1e-10)  # Avoid division by zero
        
        return normals
    
    def _calculate_bounds(self, vertices: np.ndarray) -> Dict[str, float]:
        """Calculate bounding box of mesh"""
        return {
            'xmin': float(vertices[:, 0].min()),
            'xmax': float(vertices[:, 0].max()),
            'ymin': float(vertices[:, 1].min()),
            'ymax': float(vertices[:, 1].max()),
            'zmin': float(vertices[:, 2].min()),
            'zmax': float(vertices[:, 2].max())
        }
    
    def save_mesh_json(self, mesh_data: Dict, output_path: Path):
        """Save mesh data as JSON file"""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(mesh_data, f, indent=2)
        logger.info(f"Saved mesh to {output_path}")
    
    def load_mesh_json(self, mesh_path: Path) -> Optional[Dict]:
        """Load mesh data from JSON file"""
        if not mesh_path.exists():
            return None
        with open(mesh_path, 'r') as f:
            return json.load(f)


# Convenience function
def generate_meshes_for_patient(
    ct_output_dir: Path,
    patient_id: str = "default"
) -> Dict[str, any]:
    """
    Generate all meshes for a patient's CT analysis
    
    Args:
        ct_output_dir: Directory containing CT analysis outputs
                      (e.g., outputs/inha_ct_detection/3d_segmentation/)
        patient_id: Patient identifier
        
    Returns:
        Dictionary with all mesh data:
        {
            'organs': {'colon': {...}, 'soft_tissue': {...}, ...},
            'tumors': [{...}, {...}, ...],
            'metadata': {...}
        }
    """
    generator = MeshGenerator()
    
    # Expected file paths
    colon_mask = ct_output_dir / "colon_mask_3d.nii.gz"
    soft_tissue_mask = ct_output_dir / "soft_tissue_mask_3d.nii.gz"
    body_mask = ct_output_dir / "body_mask_3d.nii.gz"
    tumors_labeled = ct_output_dir / "tumors_3d_labeled.nii.gz"
    
    # Get spacing from one of the files
    if colon_mask.exists():
        nii = nib.load(colon_mask)
        spacing = nii.header.get_zooms()
    else:
        spacing = (1.0, 1.0, 1.0)  # Default
    
    result = {
        'organs': {},
        'tumors': [],
        'metadata': {
            'patient_id': patient_id,
            'spacing': spacing
        }
    }
    
    # Generate organ meshes
    organ_files = [
        (colon_mask, 'colon', 8000),
        (soft_tissue_mask, 'soft_tissue', 5000),
        (body_mask, 'body', 3000)
    ]
    
    for mask_path, organ_name, target_faces in organ_files:
        if mask_path.exists():
            logger.info(f"Generating {organ_name} mesh...")
            mesh_data = generator.generate_organ_mesh(
                mask_path, spacing, organ_name, 
                simplify=True, target_faces=target_faces
            )
            if mesh_data:
                result['organs'][organ_name] = mesh_data
    
    # Generate tumor meshes
    if tumors_labeled.exists():
        logger.info("Generating tumor meshes...")
        result['tumors'] = generator.generate_tumor_meshes(
            tumors_labeled, spacing, min_volume_mm3=50.0, max_tumors=15
        )
    
    return result
