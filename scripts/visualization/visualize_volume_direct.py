"""
CT volume을 직접 시각화 - PyVista 사용
슬라이스를 쌓아서 윤곽을 표면 처리
"""

import nibabel as nib
import numpy as np
import pyvista as pv

# CT 데이터 로드 - 실제 파일 사용
organ_seg_path = "F:/ADDS/output/organs_simple/organs_multilabel_hu.nii.gz"
tumor_seg_path = "F:/ADDS/output/tumor_organ_mapping/tumors_unique_3d.nii.gz"

print("장기 분할 데이터 로딩 중...")
organ_img = nib.load(organ_seg_path)
organ_data = organ_img.get_fdata()

print(f"Organ shape: {organ_data.shape}")

# PyVista로 직접 3D 시각화
plotter = pv.Plotter()
plotter.set_background('white')

# 각 장기별로 isosurface 생성
organ_labels = {
    2: {"name": "fat", "color": "gold"},
    3: {"name": "lung", "color": "lightblue"},
    4: {"name": "muscle", "color": "indianred"},
    5: {"name": "liver", "color": "saddlebrown"},
    6: {"name": "soft_tissue", "color": "pink"},
    7: {"name": "bone", "color": "white"}
}

print("\n장기별 표면 생성 중...")
for label_id, info in organ_labels.items():
    # 해당 장기 마스크
    mask = (organ_data == label_id).astype(np.uint8)
    
    if mask.sum() == 0:
        print(f"  {info['name']}: 데이터 없음")
        continue
    
    # PyVista grid 생성
    grid = pv.ImageData()
    grid.dimensions = mask.shape
    grid.point_data["values"] = mask.flatten(order="F")
    
    # Isosurface 추출 (contour)
    # 이것이 바로 "윤곽을 표면 처리"하는 것
    surface = grid.contour([0.5], scalars="values")
    
    if surface.n_points > 0:
        print(f"  {info['name']}: {surface.n_points} vertices, {surface.n_cells} faces")
        
        # 시각화 추가
        plotter.add_mesh(
            surface,
            color=info['color'],
            opacity=0.6,
            show_edges=False,
            label=info['name']
        )

# 종양도 추가 (경로는 위에서 정의됨)
try:
    tumor_img = nib.load(tumor_seg_path)
    tumor_data = tumor_img.get_fdata()
    
    # 모든 종양 (value > 0)
    tumor_mask = (tumor_data > 0.5).astype(np.uint8)
    
    if tumor_mask.sum() > 0:
        grid = pv.ImageData()
        grid.dimensions = tumor_mask.shape
        grid.point_data["values"] = tumor_mask.flatten(order="F")
        
        tumor_surface = grid.contour([0.5], scalars="values")
        
        print(f"  tumors: {tumor_surface.n_points} vertices, {tumor_surface.n_cells} faces")
        
        plotter.add_mesh(
            tumor_surface,
            color='red',
            opacity=0.9,
            show_edges=False,
            label='Tumors'
        )
except:
    print("  tumors: 파일 없음")

# 시각화 설정
plotter.add_axes()
plotter.add_legend()
plotter.camera_position = 'iso'

print("\n3D 시각화 창을 여는 중...")
print("마우스 드래그: 회전")
print("마우스 휠: 줌")
print("Q 키: 종료")

plotter.show()
