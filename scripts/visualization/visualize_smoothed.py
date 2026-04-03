"""
표면 매끄럽게 처리 (Smoothing)
"""

import nibabel as nib
import numpy as np
import pyvista as pv

# 데이터 로드
organ_seg_path = "F:/ADDS/output/organs_simple/organs_multilabel_hu.nii.gz"
tumor_seg_path = "F:/ADDS/output/tumor_organ_mapping/tumors_unique_3d.nii.gz"

print("데이터 로딩 중...")
organ_img = nib.load(organ_seg_path)
organ_data = organ_img.get_fdata()
voxel_spacing = organ_img.header.get_zooms()

print(f"Shape: {organ_data.shape}")
print(f"Voxel spacing: {voxel_spacing} mm")
print(f"Z/X ratio: {voxel_spacing[2] / voxel_spacing[0]:.2f}x (슬라이스 간격이 픽셀 간격의 몇 배)")

# PyVista 설정
plotter = pv.Plotter()
plotter.set_background('white')

# 장기 정의
organ_labels = {
    2: {"name": "fat", "color": "gold"},
    3: {"name": "lung", "color": "lightblue"},
    4: {"name": "muscle", "color": "indianred"},
    5: {"name": "liver", "color": "saddlebrown"},
    6: {"name": "soft_tissue", "color": "pink"},
    7: {"name": "bone", "color": "white"}
}

print("\n표면 생성 중 (Smoothing 적용)...")
for label_id, info in organ_labels.items():
    mask = (organ_data == label_id).astype(np.uint8)
    
    if mask.sum() == 0:
        print(f"  {info['name']}: 데이터 없음")
        continue
    
    # PyVista grid - SPACING 적용이 핵심!
    grid = pv.ImageData()
    grid.dimensions = mask.shape
    grid.spacing = voxel_spacing  # 이게 중요! 실제 mm 단위 간격
    grid.point_data["values"] = mask.flatten(order="F")
    
    # Isosurface 추출
    surface = grid.contour([0.5], scalars="values")
    
    if surface.n_points == 0:
        continue
    
    # **핵심: Smoothing 적용**
    # Laplacian smoothing으로 표면을 부드럽게
    smoothed = surface.smooth(n_iter=50, relaxation_factor=0.1)
    
    # 메시 단순화 (너무 많은 삼각형 제거)
    decimated = smoothed.decimate(0.5)  # 50% 감소
    
    print(f"  {info['name']}: {decimated.n_points} vertices (smoothed)")
    
    plotter.add_mesh(
        decimated,
        color=info['color'],
        opacity=0.6,
        show_edges=False,
        smooth_shading=True,  # 부드러운 쉐이딩
        label=info['name']
    )

# 종양
try:
    tumor_img = nib.load(tumor_seg_path)
    tumor_data = tumor_img.get_fdata()
    tumor_mask = (tumor_data > 0.5).astype(np.uint8)
    
    if tumor_mask.sum() > 0:
        grid = pv.ImageData()
        grid.dimensions = tumor_mask.shape
        grid.spacing = voxel_spacing  # spacing 적용
        grid.point_data["values"] = tumor_mask.flatten(order="F")
        
        tumor_surface = grid.contour([0.5], scalars="values")
        tumor_smoothed = tumor_surface.smooth(n_iter=30, relaxation_factor=0.1)
        
        print(f"  tumors: {tumor_smoothed.n_points} vertices (smoothed)")
        
        plotter.add_mesh(
            tumor_smoothed,
            color='red',
            opacity=0.9,
            show_edges=False,
            smooth_shading=True,
            label='Tumors'
        )
except Exception as e:
    print(f"  tumors: {e}")

# 시각화
plotter.add_axes()
plotter.add_legend()
plotter.camera_position = 'iso'

print("\n3D 시각화 (Smoothed)...")
plotter.show()
