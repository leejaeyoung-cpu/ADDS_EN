"""
Step 3 & 4: 3D Measurements + Surface Rendering

Accurate 3D tumor measurements and realistic surface visualization
"""
import numpy as np
import nibabel as nib
from pathlib import Path
from skimage import measure
import json
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy import ndimage

def measure_tumors_3d_accurate(labeled_volume: np.ndarray, spacing: tuple) -> list:
    """
    Accurate 3D measurements for each tumor
    
    Args:
        labeled_volume: 3D volume with labeled tumors
        spacing: Voxel spacing (x, y, z)
    
    Returns:
        List of tumor measurements
    """
    
    voxel_volume_mm3 = spacing[0] * spacing[1] * spacing[2]
    
    measurements = []
    num_tumors = int(labeled_volume.max())
    
    print(f"\n[*] Measuring {num_tumors} tumors...")
    
    for label in range(1, num_tumors + 1):
        tumor_mask = (labeled_volume == label)
        
        # Basic measurements
        voxel_count = tumor_mask.sum()
        volume_mm3 = voxel_count * voxel_volume_mm3
        
        # Get coordinates
        coords = np.where(tumor_mask)
        z_coords = coords[0] * spacing[2]
        y_coords = coords[1] * spacing[1]
        x_coords = coords[2] * spacing[0]
        
        # Centroid
        centroid_mm = (x_coords.mean(), y_coords.mean(), z_coords.mean())
        
        # Bounding box
        bbox = {
            'x': (x_coords.min(), x_coords.max()),
            'y': (y_coords.min(), y_coords.max()),
            'z': (z_coords.min(), z_coords.max())
        }
        
        # Maximum diameter (longest distance between any two points)
        # For efficiency, approximate using bounding box diagonal
        bbox_diag = np.sqrt(
            (bbox['x'][1] - bbox['x'][0])**2 +
            (bbox['y'][1] - bbox['y'][0])**2 +
            (bbox['z'][1] - bbox['z'][0])**2
        )
        
        # Equivalent sphere diameter
        radius_sphere = (3 * volume_mm3 / (4 * np.pi)) ** (1/3)
        diameter_sphere = 2 * radius_sphere
        
        # Use regionprops for shape features
        props = measure.regionprops(tumor_mask.astype(int))[0]
        
        # Sphericity (1.0 = perfect sphere)
        surface_area_approx = 4 * np.pi * radius_sphere**2  # sphere surface
        sphericity = (np.pi ** (1/3) * (6 * volume_mm3) ** (2/3)) / surface_area_approx if surface_area_approx > 0 else 0
        
        measurements.append({
            'label': int(label),
            'volume_mm3': float(volume_mm3),
            'volume_ml': float(volume_mm3 / 1000),
            'diameter_sphere_mm': float(diameter_sphere),
            'diameter_max_mm': float(bbox_diag),
            'centroid_mm': {
                'x': float(centroid_mm[0]),
                'y': float(centroid_mm[1]),
                'z': float(centroid_mm[2])
            },
            'bounding_box_mm': {
                'x_range': [float(bbox['x'][0]), float(bbox['x'][1])],
                'y_range': [float(bbox['y'][0]), float(bbox['y'][1])],
                'z_range': [float(bbox['z'][0]), float(bbox['z'][1])]
            },
            'sphericity': float(sphericity),
            'extent': float(props.extent)
        })
    
    print(f"[+] Measurements complete!")
    
    return measurements


def extract_surface_mesh(mask_3d: np.ndarray, spacing: tuple, smooth: bool = True):
    """
    Extract surface mesh using Marching Cubes
    
    Args:
        mask_3d: Binary 3D mask
        spacing: Voxel spacing
        smooth: Whether to smooth the surface
    
    Returns:
        verts, faces: Vertices and faces of the mesh
    """
    
    if smooth:
        # Smooth the mask slightly before extracting surface
        mask_smooth = ndimage.gaussian_filter(mask_3d.astype(float), sigma=1.0)
        level = 0.5
    else:
        mask_smooth = mask_3d.astype(float)
        level = 0.5
    
    # Marching Cubes
    try:
        verts, faces, normals, values = measure.marching_cubes(
            mask_smooth,
            level=level,
            spacing=spacing,
            step_size=2  # Reduce mesh complexity
        )
        return verts, faces
    except:
        return None, None


def visualize_3d_surface(
    volume: np.ndarray,
    masks: dict,
    spacing: tuple,
    output_dir: Path,
    tumors_labeled: np.ndarray = None
):
    """
    Create 3D surface renderings
    
    Args:
        volume: CT volume
        masks: Dictionary of masks {name: mask}
        spacing: Voxel spacing
        output_dir: Output directory
        tumors_labeled: Labeled tumor volume
    """
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n[*] Creating 3D surface visualizations...")
    
    # 1. Individual organ surfaces
    for name, mask in masks.items():
        print(f"[*] Extracting surface: {name}")
        
        verts, faces = extract_surface_mesh(mask, spacing, smooth=True)
        
        if verts is None:
            print(f"[!] Failed to extract surface for {name}")
            continue
        
        # Create visualization
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Create mesh
        mesh = Poly3DCollection(verts[faces], alpha=0.7, linewidth=0)
        
        # Color based on type
        if 'colon' in name.lower():
            color = 'orange'
        elif 'soft' in name.lower():
            color = 'lightcoral'
        elif 'body' in name.lower():
            color = 'lightblue'
        else:
            color = 'gray'
        
        mesh.set_facecolor(color)
        mesh.set_edgecolor('none')
        ax.add_collection3d(mesh)
        
        # Set limits
        ax.set_xlim(0, volume.shape[2] * spacing[0])
        ax.set_ylim(0, volume.shape[1] * spacing[1])
        ax.set_zlim(0, volume.shape[0] * spacing[2])
        
        ax.set_xlabel('X (mm)')
        ax.set_ylabel('Y (mm)')
        ax.set_zlabel('Z (mm)')
        ax.set_title(f'3D Surface: {name}', fontsize=14, fontweight='bold')
        
        # Multiple viewing angles
        for angle in [45, 135, 225, 315]:
            ax.view_init(elev=20, azim=angle)
            plt.savefig(
                output_dir / f'{name.replace(" ", "_")}_angle_{angle}.png',
                dpi=150,
                bbox_inches='tight',
                facecolor='white'
           )
        
        plt.close()
        print(f"[+] Saved: {name}")
    
    # 2. Combined visualization (colon + tumors)
    if tumors_labeled is not None:
        print(f"\n[*] Creating combined colon + tumors visualization...")
        
        fig = plt.figure(figsize=(16, 12))
        ax = fig.add_subplot(111, projection='3d')
        
        # Extract colon surface
        if 'Colon' in masks:
            verts_colon, faces_colon = extract_surface_mesh(masks['Colon'], spacing, smooth=True)
            
            if verts_colon is not None:
                mesh_colon = Poly3DCollection(verts_colon[faces_colon], alpha=0.3, linewidth=0)
                mesh_colon.set_facecolor('orange')
                mesh_colon.set_edgecolor('none')
                ax.add_collection3d(mesh_colon)
        
        # Extract tumor surfaces
        num_tumors = int(tumors_labeled.max())
        for tumor_id in range(1, min(num_tumors + 1, 11)):  # Top 10 tumors
            tumor_mask = (tumors_labeled == tumor_id)
            
            verts_tumor, faces_tumor = extract_surface_mesh(tumor_mask, spacing, smooth=True)
            
            if verts_tumor is not None:
                mesh_tumor = Poly3DCollection(verts_tumor[faces_tumor], alpha=0.9, linewidth=0)
                mesh_tumor.set_facecolor('red')
                mesh_tumor.set_edgecolor('darkred')
                ax.add_collection3d(mesh_tumor)
        
        # Set limits
        ax.set_xlim(0, volume.shape[2] * spacing[0])
        ax.set_ylim(0, volume.shape[1] * spacing[1])
        ax.set_zlim(0, volume.shape[0] * spacing[2])
        
        ax.set_xlabel('X (mm)', fontsize=12)
        ax.set_ylabel('Y (mm)', fontsize=12)
        ax.set_zlabel('Z (mm)', fontsize=12)
        ax.set_title('3D View: Colon (Orange) + Tumors (Red)', 
                     fontsize=16, fontweight='bold')
        
        # Save multiple angles
        for angle in [45, 90, 135, 180, 225, 270, 315, 360]:
            ax.view_init(elev=20, azim=angle)
            plt.savefig(
                output_dir / f'combined_colon_tumors_angle_{angle:03d}.png',
                dpi=200,
                bbox_inches='tight',
                facecolor='white'
            )
        
        plt.close()
        print(f"[+] Combined visualization saved!")
    
    print(f"\n[+] All 3D visualizations saved to: {output_dir}")


if __name__ == "__main__":
    print("="*80)
    print("STEP 3 & 4: 3D MEASUREMENTS + SURFACE RENDERING")
    print("="*80)
    
    # Load data
    ct_path = Path("outputs/inha_ct_analysis/inha_ct_volume.nii.gz")
    tumors_path = Path("outputs/inha_ct_detection/3d_segmentation/tumors_3d_labeled.nii.gz")
    
    print(f"\n[*] Loading CT volume...")
    ct_nii = nib.load(ct_path)
    volume = ct_nii.get_fdata()
    spacing = ct_nii.header.get_zooms()
    
    print(f"[*] Loading tumor labels...")
    tumors_nii = nib.load(tumors_path)
    tumors_labeled = tumors_nii.get_fdata().astype(int)
    
    # Step 3: Accurate measurements
    measurements = measure_tumors_3d_accurate(tumors_labeled, spacing)
    
    # Save measurements
    output_dir = Path("outputs/inha_ct_detection/3d_analysis")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    measurements_path = output_dir / "tumor_measurements_3d_accurate.json"
    with open(measurements_path, 'w') as f:
        json.dump({
            'tumor_count': len(measurements),
            'total_volume_ml': sum(m['volume_ml'] for m in measurements),
            'tumors': sorted(measurements, key=lambda x: x['volume_mm3'], reverse=True)
        }, f, indent=2)
    
    print(f"\n[+] Measurements saved: {measurements_path}")
    
    # Load masks
    masks_3d = {}
    mask_dir = Path("outputs/inha_ct_detection/3d_segmentation")
    for mask_name in ['body_mask_3d', 'soft_tissue_mask_3d', 'colon_mask_3d']:
        mask_path = mask_dir / f"{mask_name}.nii.gz"
        if mask_path.exists():
            mask_nii = nib.load(mask_path)
            masks_3d[mask_name.replace('_mask_3d', '').replace('_', ' ').title()] = mask_nii.get_fdata().astype(bool)
    
    # Step 4: Surface rendering
    visualize_3d_surface(
        volume,
        masks_3d,
        spacing,
        output_dir / '3d_surfaces',
        tumors_labeled=tumors_labeled
    )
    
    # Summary
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    print(f"Total tumors: {len(measurements)}")
    print(f"Total tumor volume: {sum(m['volume_ml'] for m in measurements):.2f} mL")
    
    print(f"\nTop 5 tumors by volume:")
    sorted_tumors = sorted(measurements, key=lambda x: x['volume_mm3'], reverse=True)
    for i, t in enumerate(sorted_tumors[:5], 1):
        print(f"  {i}. Tumor #{t['label']}: "
              f"{t['diameter_sphere_mm']:.1f}mm diameter, "
              f"{t['volume_ml']:.2f}mL, "
              f"sphericity={t['sphericity']:.2f}")
    
    print(f"\n{'='*80}")
    print("STEP 3 & 4 COMPLETE - 3D PIPELINE FINISHED!")
    print(f"{'='*80}")
