"""
API Router: 3D Medical Viewer
환자 3D 메시 데이터 제공
"""

from fastapi import APIRouter, Depends, HTTPException
from typing import List, Dict, Optional
import numpy as np
import nibabel as nib
from pathlib import Path
from skimage import measure
import psycopg2

router = APIRouter(prefix="/patients", tags=["3d_viewer"])

# DB 연결 (의존성 주입)
def get_db():
    conn = psycopg2.connect(
        host="postgres",
        port=5432,
        database="adds_clinical",
        user="adds",
        password="adds_secure_2026"
    )
    try:
        yield conn
    finally:
        conn.close()


@router.get("/{patient_id}/3d-meshes")
async def get_3d_meshes(
    patient_id: str,
    timepoint: str = "T0",
    db = Depends(get_db)
):
    """
    환자의 3D 메시 데이터 반환
    
    Returns:
        {
            "patient_id": "PT001",
            "timepoint": "T0",
            "scan_date": "2026-01-15",
            "organs": [
                {
                    "name": "colon",
                    "vertices": [[x, y, z], ...],
                    "faces": [[v1, v2, v3], ...]
                }
            ],
            "tumors": [
                {
                    "tumor_id": "lesion_1",
                    "vertices": [...],
                    "faces": [...],
                    "centroid": [x, y, z],
                    "volume_mm3": 15234
                }
            ],
            "tumor_stats": {
                "count": 2,
                "total_volume": 24155,
                "max_diameter": 42.3,
                "change_percent": -20.5
            }
        }
    """
    
    # 1. CT 스캔 정보 조회
    scan_info = get_scan_info(patient_id, timepoint, db)
    
    if not scan_info:
        raise HTTPException(status_code=404, detail=f"Scan not found for {patient_id} at {timepoint}")
    
    # 2. 장기 메시 로드
    organs = []
    organ_files = ['colon', 'liver', 'kidneys']
    
    for organ_name in organ_files:
        organ_path = Path(scan_info['nifti_path']).parent / 'segmentation' / f'{organ_name}_mask.nii.gz'
        if organ_path.exists():
            organ_mesh = create_mesh_from_mask(organ_path, downsample=2)
            if organ_mesh:
                organs.append({
                    'name': organ_name,
                    'vertices': organ_mesh['vertices'],
                    'faces': organ_mesh['faces']
                })
    
    # 3. 종양 메시 로드
    tumors = []
    tumor_measurements = get_tumor_measurements(scan_info['scan_id'], db)
    
    tumor_mask_path = Path(scan_info['nifti_path']).parent / 'tumor_mask_3d.nii.gz'
    if tumor_mask_path.exists():
        tumor_mesh = create_mesh_from_mask(tumor_mask_path, downsample=1)
        
        if tumor_mesh and tumor_measurements:
            # 각 종양별로 분리
            for i, tm in enumerate(tumor_measurements):
                tumors.append({
                    'tumor_id': tm['tumor_id'],
                    'vertices': tumor_mesh['vertices'],  # TODO: 개별 종양 분리
                    'faces': tumor_mesh['faces'],
                    'centroid': [tm['centroid_x'], tm['centroid_y'], tm['centroid_z']],
                    'volume_mm3': tm['volume_mm3']
                })
    
    # 4. 종양 통계
    tumor_stats = calculate_tumor_stats(patient_id, timepoint, tumor_measurements, db)
    
    return {
        'patient_id': patient_id,
        'timepoint': timepoint,
        'scan_date': str(scan_info['scan_date']),
        'organs': organs,
        'tumors': tumors,
        'tumor_stats': tumor_stats
    }


def get_scan_info(patient_id: str, timepoint: str, db) -> Optional[Dict]:
    """스캔 정보 조회"""
    with db.cursor() as cur:
        cur.execute("""
            SELECT scan_id, scan_date, nifti_path
            FROM ct_scans
            WHERE patient_id = %s AND timepoint = %s
        """, (patient_id, timepoint))
        row = cur.fetchone()
    
    if not row:
        return None
    
    return {
        'scan_id': row[0],
        'scan_date': row[1],
        'nifti_path': row[2]
    }


def get_tumor_measurements(scan_id: int, db) -> List[Dict]:
    """종양 측정 데이터 조회"""
    with db.cursor() as cur:
        cur.execute("""
            SELECT tumor_id, volume_mm3, max_diameter_mm,
                   centroid_x, centroid_y, centroid_z
            FROM tumor_measurements
            WHERE scan_id = %s
        """, (scan_id,))
        rows = cur.fetchall()
    
    return [
        {
            'tumor_id': row[0],
            'volume_mm3': row[1],
            'max_diameter_mm': row[2],
            'centroid_x': row[3],
            'centroid_y': row[4],
            'centroid_z': row[5]
        }
        for row in rows
    ]


def create_mesh_from_mask(mask_path: Path, downsample: int = 2) -> Optional[Dict]:
    """
    NIfTI 마스크에서 3D 메시 생성 (Marching Cubes)
    
    Args:
        mask_path: NIfTI 마스크 파일 경로
        downsample: 다운샘플링 비율 (성능 최적화)
    
    Returns:
        {'vertices': [[x,y,z], ...], 'faces': [[v1,v2,v3], ...]}
    """
    try:
        # NIfTI 로드
        nifti = nib.load(str(mask_path))
        volume = nifti.get_fdata()
        spacing = nifti.header.get_zooms()
        
        # 다운샘플링
        if downsample > 1:
            volume = volume[::downsample, ::downsample, ::downsample]
            spacing = tuple(s * downsample for s in spacing)
        
        # Marching cubes
        verts, faces, normals, values = measure.marching_cubes(
            volume,
            level=0.5,
            spacing=spacing
        )
        
        # 중심으로 이동
        center = verts.mean(axis=0)
        verts -= center
        
        return {
            'vertices': verts.tolist(),
            'faces': faces.tolist()
        }
    
    except Exception as e:
        print(f"Failed to create mesh from {mask_path}: {e}")
        return None


def calculate_tumor_stats(
    patient_id: str,
    timepoint: str,
    tumor_measurements: List[Dict],
    db
) -> Dict:
    """종양 통계 계산"""
    
    if not tumor_measurements:
        return {
            'count': 0,
            'total_volume': 0,
            'max_diameter': 0,
            'change_percent': None
        }
    
    total_volume = sum(tm['volume_mm3'] for tm in tumor_measurements)
    max_diameter = max(tm['max_diameter_mm'] or 0 for tm in tumor_measurements)
    
    # Baseline과 비교
    change_percent = None
    if timepoint != 'T0':
        baseline_volume = get_baseline_volume(patient_id, db)
        if baseline_volume:
            change_percent = ((total_volume - baseline_volume) / baseline_volume) * 100
    
    return {
        'count': len(tumor_measurements),
        'total_volume': round(total_volume, 2),
        'max_diameter': round(max_diameter, 2),
        'change_percent': round(change_percent, 1) if change_percent is not None else None
    }


def get_baseline_volume(patient_id: str, db) -> Optional[float]:
    """Baseline (T0) 종양 부피 조회"""
    with db.cursor() as cur:
        cur.execute("""
            SELECT SUM(tm.volume_mm3)
            FROM tumor_measurements tm
            JOIN ct_scans ct ON tm.scan_id = ct.scan_id
            WHERE ct.patient_id = %s AND ct.timepoint = 'T0'
        """, (patient_id,))
        row = cur.fetchone()
    
    return row[0] if row and row[0] else None
