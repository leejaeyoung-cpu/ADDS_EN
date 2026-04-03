"""
메시 데이터 무결성 검증
"""

import json
import numpy as np
from pathlib import Path

def validate_mesh(json_path):
    """메시 데이터의 기본적인 무결성 검사"""
    
    print(f"\n{'='*60}")
    print(f"검증 중: {json_path.name}")
    print('='*60)
    
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    vertices = np.array(data['vertices'])
    faces = np.array(data['faces'])
    
    print(f"\n1. 기본 정보:")
    print(f"   Vertices: {len(vertices)}")
    print(f"   Faces: {len(faces)}")
    
    print(f"\n2. Vertices 범위:")
    print(f"   X: {vertices[:, 0].min():.2f} ~ {vertices[:, 0].max():.2f}")
    print(f"   Y: {vertices[:, 1].min():.2f} ~ {vertices[:, 1].max():.2f}")
    print(f"   Z: {vertices[:, 2].min():.2f} ~ {vertices[:, 2].max():.2f}")
    
    print(f"\n3. Face indices 범위:")
    print(f"   Min index: {faces.min()}")
    print(f"   Max index: {faces.max()}")
    print(f"   Total vertices: {len(vertices)}")
    
    # 치명적 오류 검사
    issues = []
    
    # 1) Faces가 vertices 범위 밖을 참조?
    if faces.max() >= len(vertices):
        issues.append(f"❌ CRITICAL: Face index {faces.max()} exceeds vertex count {len(vertices)}")
    
    # 2) Negative indices?
    if faces.min() < 0:
        issues.append(f"❌ CRITICAL: Negative face index found: {faces.min()}")
    
    # 3) Degenerate triangles (같은 vertex를 2번 이상 참조)
    degenerate_count = 0
    for i, face in enumerate(faces[:100]):  # 처음 100개만 체크
        if face[0] == face[1] or face[1] == face[2] or face[0] == face[2]:
            degenerate_count += 1
    
    if degenerate_count > 0:
        issues.append(f"⚠️ WARNING: {degenerate_count}/100 faces are degenerate")
    
    # 4) Zero-area triangles
    zero_area_count = 0
    for i in range(min(100, len(faces))):
        face = faces[i]
        v0, v1, v2 = vertices[face[0]], vertices[face[1]], vertices[face[2]]
        
        # Cross product to get area
        edge1 = v1 - v0
        edge2 = v2 - v0
        cross = np.cross(edge1, edge2)
        area = np.linalg.norm(cross) / 2
        
        if area < 1e-10:
            zero_area_count += 1
    
    if zero_area_count > 0:
        issues.append(f"⚠️ WARNING: {zero_area_count}/100 faces have near-zero area")
    
    # 5) 모든 vertices가 같은 위치?
    if np.allclose(vertices, vertices[0]):
        issues.append("❌ CRITICAL: All vertices are at the same location!")
    
    print(f"\n4. 무결성 검사:")
    if issues:
        for issue in issues:
            print(f"   {issue}")
    else:
        print("   ✅ All basic checks passed")
    
    # 샘플 데이터 출력
    print(f"\n5. 샘플 데이터:")
    print(f"   First 3 vertices:")
    for i in range(min(3, len(vertices))):
        print(f"      v{i}: {vertices[i]}")
    
    print(f"   First 3 faces:")
    for i in range(min(3, len(faces))):
        print(f"      f{i}: {faces[i]} -> vertices at indices {faces[i]}")
    
    return len(issues) == 0

# 검증 실행
mesh_dir = Path("F:/ADDS/output/meshes_multilayer")

test_meshes = [
    "fat_mesh.json",
    "liver_mesh.json",
    "muscle_tumors_mesh.json"
]

all_valid = True
for mesh_file in test_meshes:
    json_path = mesh_dir / mesh_file
    if json_path.exists():
        is_valid = validate_mesh(json_path)
        all_valid = all_valid and is_valid
    else:
        print(f"\n❌ Not found: {json_path}")
        all_valid = False

print("\n" + "="*60)
if all_valid:
    print("✅ 모든 메시가 기본 검증 통과")
    print("문제는 뷰어 설정일 가능성")
else:
    print("❌ 메시 데이터에 심각한 문제 발견")
    print("multilayer_mesh_generator.py 수정 필요")
print("="*60)
