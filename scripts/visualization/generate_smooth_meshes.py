"""
부드러운 표면 메시 생성
PyVista smoothing + decimation → JSON export
"""

import nibabel as nib
import numpy as np
import pyvista as pv
import json
from pathlib import Path

# 입력/출력 경로
organ_seg_path = "F:/ADDS/output/organs_simple/organs_multilabel_hu.nii.gz"
tumor_seg_path = "F:/ADDS/output/tumor_organ_mapping/tumors_unique_3d.nii.gz"
output_dir = Path("F:/ADDS/output/meshes_smoothed")
output_dir.mkdir(exist_ok=True)

print("="*60)
print("부드러운 3D 메시 생성 시작")
print("="*60)

# 데이터 로드
print("\n데이터 로딩 중...")
organ_img = nib.load(organ_seg_path)
organ_data = organ_img.get_fdata()
voxel_spacing = organ_img.header.get_zooms()

print(f"Shape: {organ_data.shape}")
print(f"Voxel spacing: {voxel_spacing} mm")

# 장기 정의
organ_labels = {
    2: {"name": "fat", "color": "#FFD700"},
    3: {"name": "lung_tissue", "color": "#87CEEB"},
    4: {"name": "muscle", "color": "#CD5C5C"},
    5: {"name": "liver", "color": "#8B4513"},
    6: {"name": "soft_tissue", "color": "#FFB6C1"},
    7: {"name": "bone", "color": "#FFFFFF"}
}

def create_smooth_mesh(data, label_id, voxel_spacing, smooth_iterations=100, decimate_ratio=0.7):
    """
    부드러운 메시 생성
    
    Args:
        data: 3D segmentation array
        label_id: 장기 label
        voxel_spacing: voxel 간격 (mm)
        smooth_iterations: smoothing 반복 횟수 (높을수록 부드러움)
        decimate_ratio: 메시 간략화 비율 (0.5 = 50% 감소)
    
    Returns:
        smoothed mesh (PyVista PolyData)
    """
    # 1) 마스크 생성
    mask = (data == label_id).astype(np.uint8)
    
    if mask.sum() == 0:
        return None
    
    # 2) PyVista grid 생성
    grid = pv.ImageData()
    grid.dimensions = mask.shape
    grid.spacing = voxel_spacing
    grid.point_data["values"] = mask.flatten(order="F")
    
    # 3) Isosurface 추출 (marching cubes)
    surface = grid.contour([0.5], scalars="values")
    
    if surface.n_points == 0:
        return None
    
    # 4) **Smoothing (핵심!)**
    # Laplacian smoothing으로 표면을 부드럽게
    smoothed = surface.smooth(
        n_iter=smooth_iterations,
        relaxation_factor=0.1,
        feature_angle=120.0,
        boundary_smoothing=True,
        feature_smoothing=True
    )
    
    # 5) **Decimation (메시 단순화)**
    # 너무 많은 삼각형 제거하여 성능 향상
    decimated = smoothed.decimate(decimate_ratio)
    
    # 6) Normal 재계산
    decimated.compute_normals(inplace=True)
    
    return decimated

# 장기별 메시 생성
print("\n장기 메시 생성 중...")
for label_id, info in organ_labels.items():
    print(f"\n{info['name']} 처리 중...")
    
    mesh = create_smooth_mesh(
        organ_data, 
        label_id, 
        voxel_spacing,
        smooth_iterations=100,  # 많이 smooth
        decimate_ratio=0.7      # 70% 유지
    )
    
    if mesh is None:
        print(f"  [!] 데이터 없음")
        continue
    
    print(f"  [OK] Vertices: {mesh.n_points}")
    print(f"  [OK] Faces: {mesh.n_cells}")
    
    # JSON 형식으로 저장
    vertices = mesh.points.tolist()
    faces = mesh.faces.reshape(-1, 4)[:, 1:4].tolist()  # VTK format에서 Plotly format으로
    normals = mesh.point_data['Normals'].tolist() if 'Normals' in mesh.point_data else []
    
    mesh_json = {
        "name": info['name'],
        "color": info['color'],
        "num_vertices": len(vertices),
        "num_faces": len(faces),
        "vertices": vertices,
        "faces": faces,
        "normals": normals
    }
    
    output_file = output_dir / f"{info['name']}_mesh.json"
    with open(output_file, 'w') as f:
        json.dump(mesh_json, f)
    
    print(f"  [SAVE] 저장: {output_file.name}")

# 종양 메시 생성
print("\n종양 메시 생성 중...")

# 종양 분할별
tumor_organs = {
    4: "muscle",
    5: "liver", 
    6: "soft_tissue"
}

try:
    tumor_img = nib.load(tumor_seg_path)
    tumor_data = tumor_img.get_fdata()
    
    for organ_label, organ_name in tumor_organs.items():
        # 해당 장기 영역의 종양만 추출
        organ_mask = (organ_data == organ_label)
        tumor_mask = (tumor_data > 0.5) & organ_mask
        
        if tumor_mask.sum() == 0:
            print(f"  [!] {organ_name} 종양 없음")
            continue
        
        print(f"\n{organ_name} 종양 처리 중...")
        
        # 종양은 더 부드럽게
        grid = pv.ImageData()
        grid.dimensions = tumor_mask.astype(np.uint8).shape
        grid.spacing = voxel_spacing
        grid.point_data["values"] = tumor_mask.astype(np.uint8).flatten(order="F")
        
        surface = grid.contour([0.5], scalars="values")
        
        if surface.n_points == 0:
            continue
        
        # 종양은 더 많이 smooth (detail 유지하면서)
        smoothed = surface.smooth(
            n_iter=50,
            relaxation_factor=0.05,
            feature_angle=100.0
        )
        
        decimated = smoothed.decimate(0.8)  # 80% 유지
        decimated.compute_normals(inplace=True)
        
        print(f"  [OK] Vertices: {decimated.n_points}")
        print(f"  [OK] Faces: {decimated.n_cells}")
        
        # JSON 저장
        vertices = decimated.points.tolist()
        faces = decimated.faces.reshape(-1, 4)[:, 1:4].tolist()
        normals = decimated.point_data['Normals'].tolist() if 'Normals' in decimated.point_data else []
        
        mesh_json = {
            "name": f"{organ_name}_tumors",
            "color": "#FF0000",
            "num_vertices": len(vertices),
            "num_faces": len(faces),
            "vertices": vertices,
            "faces": faces,
            "normals": normals
        }
        
        output_file = output_dir / f"{organ_name}_tumors_mesh.json"
        with open(output_file, 'w') as f:
            json.dump(mesh_json, f)
        
        print(f"  [SAVE] 저장: {output_file.name}")

except Exception as e:
    print(f"종양 처리 오류: {e}")

print("\n" + "="*60)
print("[OK] 부드러운 메시 생성 완료!")
print("="*60)
print(f"저장 위치: {output_dir}")
print("\n다음 단계:")
print("1. streamlit_3d_viewer.py에서 경로를 meshes_smoothed로 변경")
print("2. Streamlit 앱 재실행")
