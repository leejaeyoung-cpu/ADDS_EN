"""
API endpoint for serving multi-layer organ meshes
"""

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from pathlib import Path
import json

router = APIRouter(prefix="/api/organ-meshes", tags=["organ_meshes"])

MESH_DIR = Path(BASE_DIR / "output/meshes_multilayer")


@router.get("/catalog")
async def get_mesh_catalog():
    """메시 카탈로그 반환"""
    catalog_file = MESH_DIR / "mesh_catalog.json"
    
    if not catalog_file.exists():
        raise HTTPException(status_code=404, detail="Mesh catalog not found")
    
    with open(catalog_file, 'r') as f:
        catalog = json.load(f)
    
    return catalog


@router.get("/organ/{organ_name}")
async def get_organ_mesh(organ_name: str):
    """특정 장기의 메시 데이터 반환"""
    mesh_file = MESH_DIR / f"{organ_name}_mesh.json"
    
    if not mesh_file.exists():
        raise HTTPException(status_code=404, detail=f"Mesh for {organ_name} not found")
    
    with open(mesh_file, 'r') as f:
        mesh_data = json.load(f)
    
    return mesh_data


@router.get("/tumor/{organ_name}")
async def get_tumor_mesh(organ_name: str):
    """특정 장기의 종양 메시 데이터 반환"""
    tumor_file = MESH_DIR / f"{organ_name}_tumors_mesh.json"
    
    if not tumor_file.exists():
        raise HTTPException(status_code=404, detail=f"Tumor mesh for {organ_name} not found")
    
    with open(tumor_file, 'r') as f:
        tumor_data = json.load(f)
    
    return tumor_data


@router.get("/stats")
async def get_mesh_stats():
    """모든 메시의 통계 정보"""


    catalog_file = MESH_DIR / "mesh_catalog.json"
    
    if not catalog_file.exists():
        raise HTTPException(status_code=404, detail="Mesh catalog not found")
    
    with open(catalog_file, 'r') as f:
        catalog = json.load(f)
    
    stats = {
        'total_organs': len(catalog['organs']),
        'total_tumors': len(catalog['tumors']),
        'organs': {}
    }
    
    # 각 장기의 통계
    for organ_name, mesh_path in catalog['organs'].items():
        try:
            with open(mesh_path, 'r') as f:
                mesh_data = json.load(f)
            
            stats['organs'][organ_name] = {
                'vertices': mesh_data['num_vertices'],
                'faces': mesh_data['num_faces'],
                'bounds': mesh_data['bounds']
            }
        except Exception as e:
            stats['organs'][organ_name] = {'error': str(e)}
    
    return stats
