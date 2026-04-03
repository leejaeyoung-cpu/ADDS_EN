"""
PyVista 3D Viewer for ADDS CT Analysis
Interactive 3D visualization of CT scans with tumor detection results
"""

import numpy as np
import pyvista as pv
import nibabel as nib
from pathlib import Path
from typing import Optional, Tuple, Dict
import tempfile
from skimage import measure


class PyVista3DViewer:
    """Interactive 3D medical image viewer using PyVista"""
    
    def __init__(self, theme='document'):
        """
        Initialize 3D viewer
        
        Args:
            theme: PyVista theme ('document', 'dark', 'paraview')
        """
        pv.set_plot_theme(theme)
        self.plotter = None
        
    def load_nifti_volume(self, nifti_path: Path) -> Tuple[np.ndarray, Tuple]:
        """
        Load NIfTI volume
        
        Args:
            nifti_path: Path to .nii or .nii.gz file
            
        Returns:
            volume: 3D numpy array
            spacing: Voxel spacing (x, y, z)
        """
        nii = nib.load(nifti_path)
        volume = nii.get_fdata()
        spacing = nii.header.get_zooms()
        
        return volume, spacing
    
    def create_surface_mesh(self, 
                           mask: np.ndarray, 
                           spacing: Tuple[float, float, float],
                           downsample: int = 2,
                           smoothing_iterations: int = 10) -> pv.PolyData:
        """
        Create 3D surface mesh from binary mask using marching cubes
        
        Args:
            mask: Binary 3D mask
            spacing: Voxel spacing (x, y, z)
            downsample: Downsample factor for faster rendering
            smoothing_iterations: Number of smoothing iterations
            
        Returns:
            mesh: PyVista PolyData mesh
        """
        # Downsample for performance
        if downsample > 1:
            mask_ds = mask[::downsample, ::downsample, ::downsample]
            spacing_ds = tuple(s * downsample for s in spacing)
        else:
            mask_ds = mask
            spacing_ds = spacing
        
        # Marching cubes
        try:
            verts, faces, normals, values = measure.marching_cubes(
                mask_ds.astype(float),
                level=0.5,
                spacing=spacing_ds,
                allow_degenerate=False
            )
        except Exception as e:
            print(f"[WARNING] Marching cubes failed: {e}")
            return None
        
        # Convert to PyVista mesh
        faces_pv = np.hstack([[3] + list(face) for face in faces])
        mesh = pv.PolyData(verts, faces_pv)
        
        # Smooth mesh
        if smoothing_iterations > 0:
            mesh = mesh.smooth(n_iter=smoothing_iterations, 
                              relaxation_factor=0.1)
        
        return mesh
    
    def create_ct_visualization(self,
                               ct_volume_path: Path,
                               colon_mask_path: Optional[Path] = None,
                               tumor_mask_path: Optional[Path] = None,
                               output_html: Optional[Path] = None,
                               downsample: int = 2) -> str:
        """
        Create comprehensive 3D visualization
        
        Args:
            ct_volume_path: Path to CT volume NIfTI
            colon_mask_path: Path to colon segmentation mask
            tumor_mask_path: Path to tumor detection mask
            output_html: Output HTML file path
            downsample: Downsample factor (2 = half resolution)
            
        Returns:
            html_path: Path to generated HTML file
        """
        print("[INFO] Loading CT data...")
        volume, spacing = self.load_nifti_volume(ct_volume_path)
        
        # Create plotter
        self.plotter = pv.Plotter(off_screen=True, window_size=[1200, 800])
        
        # Add colon surface
        if colon_mask_path and colon_mask_path.exists():
            print("[INFO] Creating colon surface mesh...")
            colon_mask, _ = self.load_nifti_volume(colon_mask_path)
            colon_mesh = self.create_surface_mesh(
                colon_mask.astype(bool), 
                spacing, 
                downsample=downsample,
                smoothing_iterations=15
            )
            
            if colon_mesh:
                self.plotter.add_mesh(
                    colon_mesh,
                    color='pink',
                    opacity=0.4,
                    smooth_shading=True,
                    name='Colon'
                )
                print(f"  [+] Colon mesh: {colon_mesh.n_points} points, {colon_mesh.n_cells} cells")
        
        # Add tumor surface
        if tumor_mask_path and tumor_mask_path.exists():
            print("[INFO] Creating tumor surface mesh...")
            tumor_mask, _ = self.load_nifti_volume(tumor_mask_path)
            
            if tumor_mask.max() > 0:
                tumor_mesh = self.create_surface_mesh(
                    tumor_mask.astype(bool),
                    spacing,
                    downsample=1,  # Higher resolution for tumors
                    smoothing_iterations=5
                )
                
                if tumor_mesh:
                    self.plotter.add_mesh(
                        tumor_mesh,
                        color='red',
                        opacity=0.8,
                        smooth_shading=True,
                        name='Tumors'
                    )
                    print(f"  [+] Tumor mesh: {tumor_mesh.n_points} points, {tumor_mesh.n_cells} cells")
        
        # Configure camera
        self.plotter.camera_position = 'xy'
        self.plotter.camera.azimuth = 45
        self.plotter.camera.elevation = 30
        
        # Add axes and orientation marker
        self.plotter.show_axes()
        self.plotter.add_orientation_widget()
        
        # Add title
        self.plotter.add_text(
            "ADDS CT 3D Visualization",
            position='upper_left',
            font_size=14,
            color='black',
            font='arial'
        )
        
        # Export to HTML
        if output_html is None:
            output_html = Path(tempfile.mktemp(suffix='.html'))
        
        print(f"[INFO] Exporting to HTML: {output_html}")
        self.plotter.export_html(str(output_html))
        
        print("[OK] 3D visualization created successfully!")
        
        return str(output_html)
    
    def create_quick_tumor_view(self,
                                tumor_mask_path: Path,
                                output_html: Optional[Path] = None) -> str:
        """
        Create quick tumor-only 3D view
        
        Args:
            tumor_mask_path: Path to tumor mask
            output_html: Output HTML path
            
        Returns:
            html_path: Path to HTML file
        """
        print("[INFO] Creating tumor-only 3D view...")
        
        tumor_mask, spacing = self.load_nifti_volume(tumor_mask_path)
        
        if tumor_mask.max() == 0:
            print("[WARNING] No tumors detected in mask")
            return None
        
        # Create mesh
        tumor_mesh = self.create_surface_mesh(
            tumor_mask.astype(bool),
            spacing,
            downsample=1,
            smoothing_iterations=10
        )
        
        if not tumor_mesh:
            return None
        
        # Create plotter
        self.plotter = pv.Plotter(off_screen=True, window_size=[1000, 800])
        
        self.plotter.add_mesh(
            tumor_mesh,
            color='red',
            opacity=0.9,
            smooth_shading=True,
            show_edges=False
        )
        
        self.plotter.camera_position = 'xy'
        self.plotter.show_axes()
        self.plotter.add_text(
            "Detected Tumors - 3D View",
            position='upper_left',
            font_size=12,
            color='black'
        )
        
        # Export
        if output_html is None:
            output_html = Path(tempfile.mktemp(suffix='.html'))
        
        self.plotter.export_html(str(output_html))
        print(f"[OK] Tumor view created: {output_html}")
        
        return str(output_html)


def generate_3d_visualization(
    patient_output_dir: Path,
    ct_volume_filename: str = "ct_volume.nii.gz",
    colon_mask_filename: str = "segmentation/colon_mask.nii.gz",
    tumor_mask_filename: str = "segmentation/tumor_mask.nii.gz",
    output_filename: str = "3d_viewer.html"
) -> Path:
    """
    Convenience function to generate 3D visualization for a patient
    
    Args:
        patient_output_dir: Patient output directory
        ct_volume_filename: CT volume filename
        colon_mask_filename: Colon mask filename
        tumor_mask_filename: Tumor mask filename
        output_filename: Output HTML filename
        
    Returns:
        html_path: Path to generated HTML file
    """
    viewer = PyVista3DViewer(theme='document')
    
    ct_path = patient_output_dir / ct_volume_filename
    colon_path = patient_output_dir / colon_mask_filename
    tumor_path = patient_output_dir / tumor_mask_filename
    output_path = patient_output_dir / "visualization" / output_filename
    
    # Create visualization directory
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Generate
    html_path = viewer.create_ct_visualization(
        ct_volume_path=ct_path,
        colon_mask_path=colon_path if colon_path.exists() else None,
        tumor_mask_path=tumor_path if tumor_path.exists() else None,
        output_html=output_path,
        downsample=2
    )
    
    return Path(html_path)


if __name__ == "__main__":
    # Example usage
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python pyvista_3d_viewer.py <patient_output_dir>")
        sys.exit(1)
    
    patient_dir = Path(sys.argv[1])
    html_path = generate_3d_visualization(patient_dir)
    
    print(f"\n✓ 3D visualization ready: {html_path}")
    print(f"  Open in browser to view interactive 3D model")
