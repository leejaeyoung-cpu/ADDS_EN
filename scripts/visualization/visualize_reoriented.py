"""
CT 축 재정렬 후 시각화
Orientation을 표준 ('L', 'P', 'I')로 변환
"""

import nibabel as nib
import numpy as np
import pyvista as pv

# 데이터 로드
organ_path = "F:/ADDS/output/organs_simple/organs_multilabel_hu.nii.gz"
tumor_path = "F:/ADDS/output/tumor_organ_mapping/tumors_unique_3d.nii.gz"

print("데이터 로딩 중...")
organ_img = nib.load(organ_path)
organ_data_orig = organ_img.get_fdata()

print(f"Original shape: {organ_data_orig.shape}")
print(f"Original orientation: {nib.aff2axcodes(organ_img.affine)}")

# **축 재정렬: (L,I,A) -> (L,P,I)**
# Y와 Z를 교환
organ_data = np.transpose(organ_data_orig, (0, 2, 1))

print(f"Reoriented shape: {organ_data.shape}")
print(f"New orientation: (L, A->P, I) = 표준 axial")

# Voxel spacing도 재정렬
spacing_orig = organ_img.header.get_zooms()
spacing = (spacing_orig[0], spacing_orig[2], spacing_orig[1])

print(f"Spacing: {spacing} mm")

# PyVista 시각화
plotter = pv.Plotter()
plotter.set_background('white')

organ_labels = {
    2: {"name": "fat", "color": "gold"},
    3: {"name": "lung", "color": "lightblue"},
    4: {"name": "muscle", "color": "indianred"},
    5: {"name": "liver", "color": "saddlebrown"},
    6: {"name": "soft_tissue", "color": "pink"},
    7: {"name": "bone", "color": "white"}
}

print("\n표면 생성 중 (재정렬된 축, smoothing)...")
for label_id, info in organ_labels.items():
    mask = (organ_data == label_id).astype(np.uint8)
    
    if mask.sum() == 0:
        continue
    
    grid = pv.ImageData()
    grid.dimensions = mask.shape
    grid.spacing = spacing  # 재정렬된 spacing
    grid.point_data["values"] = mask.flatten(order="F")
    
    surface = grid.contour([0.5], scalars="values")
    
    if surface.n_points == 0:
        continue
    
    # Smoothing
    smoothed = surface.smooth(n_iter=50, relaxation_factor=0.1)
    decimated = smoothed.decimate(0.5)
    
    print(f"  {info['name']}: {decimated.n_points} vertices")
    
    plotter.add_mesh(
        decimated,
        color=info['color'],
        opacity=0.6,
        show_edges=False,
        smooth_shading=True,
        label=info['name']
    )

# 종양
try:
    tumor_img = nib.load(tumor_path)
    tumor_data_orig = tumor_img.get_fdata()
    tumor_data = np.transpose(tumor_data_orig, (0, 2, 1))  # 같은 재정렬
    
    tumor_mask = (tumor_data > 0.5).astype(np.uint8)
    
    if tumor_mask.sum() > 0:
        grid = pv.ImageData()
        grid.dimensions = tumor_mask.shape
        grid.spacing = spacing
        grid.point_data["values"] = tumor_mask.flatten(order="F")
        
        tumor_surface = grid.contour([0.5], scalars="values")
        tumor_smoothed = tumor_surface.smooth(n_iter=30, relaxation_factor=0.1)
        
        print(f"  tumors: {tumor_smoothed.n_points} vertices")
        
        plotter.add_mesh(
            tumor_smoothed,
            color='red',
            opacity=0.9,
            show_edges=False,
            smooth_shading=True,
            label='Tumors'
        )
except Exception as e:
    print(f"  tumors: Error - {e}")

# 시각화
plotter.add_axes()
plotter.add_legend()
plotter.camera_position = 'iso'

print("\n3D 시각화 (축 재정렬 + Smoothing)...")
print("줄무늬가 사라지고 부드러운 표면이 보일 것입니다!")
plotter.show()
