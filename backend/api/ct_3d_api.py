"""
3D CT Visualization API
종양 3D 시각화를 위한 REST API 엔드포인트
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import List, Dict, Optional
from pathlib import Path
import logging
import json

from backend.services.ct_3d_reconstruction import CT3DReconstructor
from backend.services.mesh_generator import MeshGenerator

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/ct/3d", tags=["3D Visualization"])


# ============================================================================
# Request/Response Models
# ============================================================================

class Reconstruct3DRequest(BaseModel):
    """3D 재구성 요청"""
    scan_id: str
    dicom_directory: str
    include_organs: bool = True
    include_tumors: bool = True
    target_spacing: float = 1.0


class TumorCoordinates3D(BaseModel):
    """종양 3D 좌표"""
    tumor_id: str
    centroid_mm: List[float]
    bbox_min_mm: List[float]
    bbox_max_mm: List[float]
    volume_cm3: float
    longest_diameter_mm: float
    containing_organ: Optional[str] = None


class MeshInfo(BaseModel):
    """메시 정보"""
    name: str
    mesh_url: str
    color: List[int]
    volume_cm3: Optional[float] = None
    num_vertices: int
    num_faces: int


class CT3DVisualizationResponse(BaseModel):
    """3D 시각화 응답"""
    scan_id: str
    skin_mesh: Optional[MeshInfo] = None
    organ_meshes: List[MeshInfo] = []
    tumor_meshes: List[Dict] = []
    reconstruction_status: str


# ============================================================================
# Endpoints
# ============================================================================

@router.post("/reconstruct", response_model=Dict)
async def reconstruct_3d_volume(
    request: Reconstruct3DRequest,
    background_tasks: BackgroundTasks
):
    """
    3D 볼륨 재구성 및 메시 생성 (백그라운드 작업)
    
    Returns:
        {
            "task_id": "uuid",
            "status": "processing",
            "message": "3D reconstruction started"
        }
    """
    logger.info(f"Starting 3D reconstruction for scan {request.scan_id}")
    
    # 백그라운드 작업 시작
    task_id = f"recon_3d_{request.scan_id}"
    
    background_tasks.add_task(
        _process_3d_reconstruction,
        task_id=task_id,
        request=request
    )
    
    return {
        "task_id": task_id,
        "status": "processing",
        "message": "3D reconstruction started in background"
    }


@router.get("/mesh/{scan_id}", response_model=CT3DVisualizationResponse)
async def get_3d_meshes(scan_id: str):
    """
    스캔의 모든 3D 메시 정보 조회
    
    Returns:
        CT3DVisualizationResponse with URLs to mesh files
    """
    logger.info(f"Fetching 3D meshes for scan {scan_id}")
    
    # 메시 파일 경로
    mesh_dir = Path(f"data/3d_meshes/{scan_id}")
    
    if not mesh_dir.exists():
        raise HTTPException(
            status_code=404,
            detail=f"No 3D meshes found for scan {scan_id}"
        )
    
    # 메타데이터 로드
    metadata_file = mesh_dir / "metadata.json"
    
    if not metadata_file.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Metadata not found for scan {scan_id}"
        )
    
    with open(metadata_file) as f:
        metadata = json.load(f)
    
    # 응답 생성
    response = CT3DVisualizationResponse(
        scan_id=scan_id,
        reconstruction_status=metadata.get("status", "unknown"),
        skin_mesh=metadata.get("skin_mesh"),
        organ_meshes=metadata.get("organ_meshes", []),
        tumor_meshes=metadata.get("tumor_meshes", [])
    )
    
    return response


@router.get("/coordinates/{tumor_id}", response_model=TumorCoordinates3D)
async def get_tumor_coordinates(tumor_id: str):
    """
    특정 종양의 3D 좌표 조회
    """
    logger.info(f"Fetching coordinates for tumor {tumor_id}")
    
    # TODO: 데이터베이스에서 조회
    # 현재는 더미 데이터
    
    return TumorCoordinates3D(
        tumor_id=tumor_id,
        centroid_mm=[120.5, 45.2, 200.1],
        bbox_min_mm=[110.0, 40.0, 195.0],
        bbox_max_mm=[131.0, 50.0, 205.0],
        volume_cm3=12.5,
        longest_diameter_mm=25.3,
        containing_organ="colon_sigmoid"
    )


# ============================================================================
# Background Task
# ============================================================================

async def _process_3d_reconstruction(
    task_id: str,
    request: Reconstruct3DRequest
):
    """3D 재구성 백그라운드 작업"""
    try:
        logger.info(f"Processing 3D reconstruction task {task_id}")
        
        # 출력 디렉토리
        output_dir = Path(f"data/3d_meshes/{request.scan_id}")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 1. DICOM → 3D Volume
        reconstructor = CT3DReconstructor()
        volume = reconstructor.load_dicom_series(request.dicom_directory)
        
        # 2. 등방성 리샘플링
        volume_iso = reconstructor.resample_to_isotropic(
            target_spacing=request.target_spacing
        )
        
        # 3. NIfTI 저장
        nifti_path = output_dir / "ct_volume.nii.gz"
        reconstructor.save_nifti(str(nifti_path))
        
        # 4. 피부 분할 및 메시 생성
        skin_mesh_info = None
        if request.include_organs:
            skin_mask = reconstructor.segment_skin()
            skin_mask_path = output_dir / "skin_mask.nii.gz"
            reconstructor.save_mask_nifti(skin_mask, str(skin_mask_path))
            
            # 메시 생성
            generator = MeshGenerator()
            skin_mesh_data = generator.generate_organ_mesh(
                mask_path=skin_mask_path,
                spacing=reconstructor.metadata.spacing,
                organ_name="skin",
                simplify=True,
                target_faces=5000
            )
            
            # JSON 저장
            skin_mesh_file = output_dir / "skin.json"
            generator.save_mesh_json(skin_mesh_data, skin_mesh_file)
            
            skin_mesh_info = {
                "name": "skin",
                "mesh_url": f"/static/3d_meshes/{request.scan_id}/skin.json",
                "color": [255, 218, 168],
                "num_vertices": skin_mesh_data["num_vertices"],
                "num_faces": skin_mesh_data["num_faces"]
            }
        
        # 5. 메타데이터 저장
        metadata = {
            "scan_id": request.scan_id,
            "status": "completed",
            "spacing": list(reconstructor.metadata.spacing),
            "dimensions": list(reconstructor.metadata.dimensions),
            "skin_mesh": skin_mesh_info,
            "organ_meshes": [],
            "tumor_meshes": []
        }
        
        with open(output_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Completed 3D reconstruction task {task_id}")
        
    except Exception as e:
        logger.error(f"3D reconstruction failed for task {task_id}: {e}")
        raise
