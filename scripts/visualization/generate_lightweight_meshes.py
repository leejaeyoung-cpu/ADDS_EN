"""
경량화된 메시 생성 - 웹 브라우저 최적화
Decimation을 더 강하게 적용
"""

import nibabel as nib
import numpy as np
import pyvista as pv
import json
from pathlib import Path

# 입력/출력 경로
organ_seg_path = "F:/ADDS/output/organs_simple/organs_multilabel_hu.nii.gz"
tumor_seg_path = "F:/ADDS/output/tumor_organ_mapping/tumors_unique_3d.nii.gz"
output_dir = Path("F:/ADDS/output/meshes_lightweight")
output_dir.mkdir(exist_ok=True)

print("="*60)
print("경량화된 웹용 메시 생성")
print("="*60)

# 데이터 로드
print("\n데이터 로딩 중...")
organ_img = nib.load(organ_seg_path)
organ_data = organ_img.get_fdata()
voxel_spacing = organ_img.header.get_zooms()

# 장기 정의
organ_labels = {
    2: {"name": "fat", "color": "#FFD700"},
    3: {"name": "lung_tissue", "color": "#87CEEB"},
    4: {"name": "muscle", "color": "#CD5C5C"},
    5: {"name": "liver", "color": "#8B4513"},
    6: {"name": "soft_tissue", "color": "#FFB6C1"},
    7: {"name": "bone", "color": "#FFFFFF"}
}

def create_lightweight_mesh(data, label_id, voxel_spacing):
    """
    웹 브라우저용 경량 메시 생성
    목표: 각 메시 < 10MB
    """
    mask = (data == label_id).astype(np.uint8)
    
    if mask.sum() == 0:
        return None
    
    # PyVista grid
    grid = pv.ImageData()
    grid.dimensions = mask.shape
    grid.spacing = voxel_spacing
    grid.point_data["values"] = mask.flatten(order="F")
    
    # Isosurface
    surface = grid.contour([0.5], scalars="values")
    
    if surface.n_points == 0:
        return None
    
    # **강력한 Decimation (95% 제거!)**
    # 0.05 = 5%만 남김
    decimated_heavy = surface.decimate(0.05)
    
    # 그 다음 Smoothing (decimation 후 smoothing이 더 효율적)
    smoothed = decimated_heavy.smooth(
        n_iter=50,
        relaxation_factor=0.15
    )
    
    # Normal 계산
    smoothed.compute_normals(inplace=True)
    
    return smoothed

# 장기 메시 생성
print("\n경량 메시 생성 중...")
for label_id, info in organ_labels.items():
    print(f"\n{info['name']} 처리 중...")
    
    mesh = create_lightweight_mesh(organ_data, label_id, voxel_spacing)
    
    if mesh is None:
        print(f"  [!] 데이터 없음")
        continue
    
    print(f"  [OK] Vertices: {mesh.n_points:,}")
    print(f"  [OK] Faces: {mesh.n_cells:,}")
    
    # JSON 저장
    vertices = mesh.points.tolist()
    faces = mesh.faces.reshape(-1, 4)[:, 1:4].tolist()
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
    
    file_size_mb = output_file.stat().st_size / 1024 / 1024
    print(f"  [SAVE] {output_file.name} ({file_size_mb:.1f} MB)")

# 종양 메시
print("\n종양 메시 생성 중...")

tumor_organs = {
    4: "muscle",
    5: "liver", 
    6: "soft_tissue"
}

try:
    tumor_img = nib.load(tumor_seg_path)
    tumor_data = tumor_img.get_fdata()
    
    for organ_label, organ_name in tumor_organs.items():
        organ_mask = (organ_data == organ_label)
        tumor_mask = (tumor_data > 0.5) & organ_mask
        
        if tumor_mask.sum() == 0:
            print(f"  [!] {organ_name} 종양 없음")
            continue
        
        print(f"\n{organ_name} 종양 처리 중...")
        
        grid = pv.ImageData()
        grid.dimensions = tumor_mask.astype(np.uint8).shape
        grid.spacing = voxel_spacing
        grid.point_data["values"] = tumor_mask.astype(np.uint8).flatten(order="F")
        
        surface = grid.contour([0.5], scalars="values")
        
        if surface.n_points == 0:
            continue
        
        # 종양은 약간 덜 decimation (디테일 유지)
        decimated = surface.decimate(0.1)  # 10% 유지
        smoothed = decimated.smooth(n_iter=30, relaxation_factor=0.1)
        smoothed.compute_normals(inplace=True)
        
        print(f"  [OK] Vertices: {smoothed.n_points:,}")
        print(f"  [OK] Faces: {smoothed.n_cells:,}")
        
        # JSON 저장
        vertices = smoothed.points.tolist()
        faces = smoothed.faces.reshape(-1, 4)[:, 1:4].tolist()
        normals = smoothed.point_data['Normals'].tolist() if 'Normals' in smoothed.point_data else []
        
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
        
        file_size_mb = output_file.stat().st_size / 1024 / 1024
        print(f"  [SAVE] {output_file.name} ({file_size_mb:.1f} MB)")

except Exception as e:
    print(f"종양 처리 오류: {e}")

print("\n" + "="*60)
print("[OK] 경량 메시 생성 완료!")
print("="*60)
print(f"저장 위치: {output_dir}")

# 파일 크기 합계
total_size = sum(f.stat().st_size for f in output_dir.glob("*.json"))
print(f"\n총 파일 크기: {total_size / 1024 / 1024:.1f} MB")
print("(브라우저 로딩 가능한 크기)")
