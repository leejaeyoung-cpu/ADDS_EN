"""
가감없는 시스템 검증
Critical System Validation
"""

import json
import numpy as np
from pathlib import Path

print("=" * 80)
print("CRITICAL SYSTEM VALIDATION - 가감없는 검증")
print("=" * 80)
print()

issues = []
successes = []

# 1. 파일 존재 확인
print("[1] File Existence Check")
print("-" * 80)

files_to_check = {
    'Mesh Catalog': 'F:/ADDS/output/meshes_multilayer/mesh_catalog.json',
    'Tumor Mapping': 'F:/ADDS/output/tumor_organ_mapping/tumor_organ_mapping_improved.json',
    'Organ Volume': 'F:/ADDS/output/organs_simple/organs_multilabel_hu.nii_refined.nii.gz',
    'React Viewer': 'F:/ADDS/frontend/src/components/MultiLayerOrgan3DViewer.jsx',
    'API Endpoint': 'F:/ADDS/backend/api/organ_mesh_api.py',
}

for name, path in files_to_check.items():
    exists = Path(path).exists()
    status = "✓ EXISTS" if exists else "✗ MISSING"
    print(f"  {name:20s}: {status}")
    
    if exists:
        successes.append(f"{name} file exists")
    else:
        issues.append(f"CRITICAL: {name} file missing")

# 2. 메시 데이터 품질 검증
print("\n[2] Mesh Data Quality")
print("-" * 80)

catalog_path = Path('F:/ADDS/output/meshes_multilayer/mesh_catalog.json')
if catalog_path.exists():
    with open(catalog_path, 'r') as f:
        catalog = json.load(f)
    
    print(f"  Organ meshes: {len(catalog['organs'])}")
    print(f"  Tumor meshes: {len(catalog['tumors'])}")
    
    if len(catalog['tumors']) == 0:
        issues.append("WARNING: No tumor meshes generated (known issue)")
    
    # 각 메시 검증
    for organ_name, mesh_path in catalog['organs'].items():
        if Path(mesh_path).exists():
            with open(mesh_path, 'r') as f:
                mesh_data = json.load(f)
            
            num_verts = len(mesh_data['vertices'])
            num_faces = len(mesh_data['faces'])
            
            print(f"  {organ_name:15s}: {num_verts:,} verts, {num_faces:,} faces", end='')
            
            # 유효성 검사
            if num_verts == 0 or num_faces == 0:
                print(" ✗ INVALID")
                issues.append(f"{organ_name} mesh is empty")
            elif num_faces > 10000:
                print(f" ⚠ TOO LARGE for web")
                issues.append(f"{organ_name} mesh not optimized ({num_faces} faces)")
            else:
                print(" ✓ VALID")
                successes.append(f"{organ_name} mesh is valid")
        else:
            print(f"  {organ_name:15s}: ✗ FILE NOT FOUND")
            issues.append(f"{organ_name} mesh file missing")
else:
    issues.append("CRITICAL: Mesh catalog not found")

# 3. 종양 매핑 검증
print("\n[3] Tumor-Organ Mapping Validation")
print("-" * 80)

mapping_path = Path('F:/ADDS/output/tumor_organ_mapping/tumor_organ_mapping_improved.json')
if mapping_path.exists():
    with open(mapping_path, 'r') as f:
        mapping = json.load(f)
    
    total_tumors = mapping['summary']['total_tumors']
    organs_with_tumors = mapping['summary']['organs_with_tumors']
    
    print(f"  Total tumors: {total_tumors}")
    print(f"  Organs with tumors: {organs_with_tumors}")
    
    mapped_count = 0
    for organ, data in mapping['summary']['tumors_by_organ'].items():
        count = data['count']
        mapped_count += count
        print(f"    {organ:15s}: {count:3d} tumors ({data['total_volume_cm3']:.1f} cm³)")
    
    mapping_rate = (mapped_count / total_tumors * 100) if total_tumors > 0 else 0
    print(f"\n  Mapping success rate: {mapping_rate:.1f}% ({mapped_count}/{total_tumors})")
    
    if mapping_rate < 50:
        issues.append(f"LOW mapping rate: only {mapping_rate:.1f}%")
    elif mapping_rate < 80:
        issues.append(f"MODERATE mapping rate: {mapping_rate:.1f}%")
    else:
        successes.append(f"Good mapping rate: {mapping_rate:.1f}%")
else:
    issues.append("CRITICAL: Tumor mapping file not found")

# 4. Three.js 코드 검증
print("\n[4] Three.js Viewer Code Validation")
print("-" * 80)

viewer_path = Path('F:/ADDS/frontend/src/components/MultiLayerOrgan3DViewer.jsx')
if viewer_path.exists():
    with open(viewer_path, 'r', encoding='utf-8') as f:
        code = f.read()
    
    checks = {
        'Three.js import': 'import * as THREE',
        'OrbitControls': 'OrbitControls',
        'useState hook': 'useState',
        'Mesh loading': 'BufferGeometry',
        'Layer toggle': 'toggleLayer',
        'Opacity control': 'opacity',
    }
    
    for name, pattern in checks.items():
        found = pattern in code
        status = "✓" if found else "✗"
        print(f"  {name:25s}: {status}")
        
        if not found:
            issues.append(f"Viewer missing: {name}")
        else:
            successes.append(f"Viewer has: {name}")
else:
    issues.append("CRITICAL: Viewer component not found")

# 5. API 엔드포인트 검증
print("\n[5] API Endpoint Validation")
print("-" * 80)

api_path = Path('F:/ADDS/backend/api/organ_mesh_api.py')
if api_path.exists():
    with open(api_path, 'r', encoding='utf-8') as f:
        api_code = f.read()
    
    checks = {
        'FastAPI router': '@router',
        'Catalog endpoint': '/catalog',
        'Organ endpoint': '/organ/',
        'Stats endpoint': '/stats',
        'Error handling': 'HTTPException',
    }
    
    for name, pattern in checks.items():
        found = pattern in api_code
        status = "✓" if found else "✗"
        print(f"  {name:25s}: {status}")
        
        if not found:
            issues.append(f"API missing: {name}")
else:
    issues.append("CRITICAL: API endpoint file not found")
    print("  ✗ API file not found")

# 최종 평가
print("\n" + "=" * 80)
print("FINAL ASSESSMENT")
print("=" * 80)

print("\n✓ SUCCESSES:")
for success in successes[:10]:  # 상위 10개만
    print(f"  - {success}")

if len(successes) > 10:
    print(f"  ... and {len(successes) - 10} more")

print(f"\n✗ ISSUES ({len(issues)} total):")
for issue in issues:
    print(f"  - {issue}")

# 점수 계산
total_checks = len(successes) + len(issues)
success_rate = (len(successes) / total_checks * 100) if total_checks > 0 else 0

print(f"\n{'=' * 80}")
print(f"Overall Score: {success_rate:.1f}% ({len(successes)}/{total_checks} checks passed)")
print(f"{'=' * 80}")

if success_rate >= 90:
    print("VERDICT: ✓ PRODUCTION READY")
elif success_rate >= 70:
    print("VERDICT: ⚠ FUNCTIONAL WITH ISSUES")
elif success_rate >= 50:
    print("VERDICT: ⚠ PARTIALLY WORKING")
else:
    print("VERDICT: ✗ NOT FUNCTIONAL")

print()
