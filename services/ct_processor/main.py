"""
CT Processor Service
DICOM → 3D NIfTI → 종양 검출 → 3D Connected Components → 부피 계산
"""

import os
import sys
import logging
from pathlib import Path
from typing import Dict, List, Tuple
import time

import numpy as np
import nibabel as nib
import pydicom
from scipy import ndimage
import SimpleITK as sitk
import psycopg2
from minio import Minio
import json

# 기존 ADDS 모듈 import
sys.path.append('/app/utils')
sys.path.append('/app/detection')

from step1_3d_connected_components import (
    create_3d_mask_from_2d_detections,
    segment_tumors_3d
)
from pyvista_3d_viewer import PyVista3DViewer
from detect_tumors_inha_corrected import (
    load_nifti_volume,
    detect_tumors_in_slice
)

# 로깅
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 환경 변수
DB_HOST = os.getenv('DB_HOST', 'postgres')
DB_PORT = os.getenv('DB_PORT', '5432')
DB_NAME = os.getenv('DB_NAME', 'adds_clinical')
DB_USER = os.getenv('DB_USER', 'adds')
DB_PASSWORD = os.getenv('DB_PASSWORD')

MINIO_HOST = os.getenv('MINIO_HOST', 'minio')
MINIO_PORT = os.getenv('MINIO_PORT', '9000')
MINIO_USER = os.getenv('MINIO_USER')
MINIO_PASSWORD = os.getenv('MINIO_PASSWORD')


class CTProcessor:
    """CT 3D 처리 파이프라인"""
    
    def __init__(self):
        self.db = self.connect_db()
        self.minio = Minio(
            f"{MINIO_HOST}:{MINIO_PORT}",
            access_key=MINIO_USER,
            secret_key=MINIO_PASSWORD,
            secure=False
        )
        self.viewer_3d = PyVista3DViewer()
    
    def connect_db(self):
        """데이터베이스 연결"""
        return psycopg2.connect(
            host=DB_HOST,
            port=DB_PORT,
            database=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD
        )
    
    def process_scan(self, scan_id: int):
        """
        CT 스캔 전체 처리 파이프라인
        
        Steps:
        1. DICOM → 3D NIfTI 변환
        2. 슬라이스별 종양 검출 (Rule-based)
        3. 3D Connected Components (중복 제거)
        4. Centroid 및 부피 계산
        5. DB 저장
        6. 3D Visualization 생성
        """
        logger.info(f"Processing CT scan: {scan_id}")
        
        # 1. 스캔 정보 가져오기
        scan_info = self.get_scan_info(scan_id)
        dicom_dir = Path(scan_info['dicom_path'])
        
        # 2. DICOM → NIfTI
        nifti_path, spacing = self.dicom_to_nifti(dicom_dir, scan_id)
        logger.info(f"NIfTI created: {nifti_path}")
        
        # 3. 3D 볼륨 로드
        volume, _ = load_nifti_volume(nifti_path)
        logger.info(f"Volume shape: {volume.shape}, spacing: {spacing}")
        
        # 4. 슬라이스별 종양 검출
        detection_results = self.detect_tumors_2d(volume, spacing)
        logger.info(f"2D detections: {len(detection_results)} slices with findings")
        
        # 5. 3D Connected Components
        tumors_3d = self.convert_to_3d(detection_results, volume.shape, spacing)
        logger.info(f"3D segmentation: {len(tumors_3d)} tumors identified")
        
        # 6. DB 저장
        for tumor in tumors_3d:
            self.save_tumor_measurement(scan_id, tumor)
        
        # 7. 3D Visualization
        viz_path = self.create_3d_visualization(
            scan_id, 
            nifti_path, 
            tumors_3d
        )
        
        # 8. 스캔 상태 업데이트
        self.update_scan_status(scan_id, processed=True)
        
        logger.info(f"Scan {scan_id} processing complete")
        return tumors_3d
    
    def dicom_to_nifti(self, dicom_dir: Path, scan_id: int) -> Tuple[Path, Tuple]:
        """
        DICOM → NIfTI 변환 (3D 볼륨 재구성)
        """
        logger.info(f"Converting DICOM to NIfTI: {dicom_dir}")
        
        # SimpleITK로 DICOM 시리즈 읽기
        reader = sitk.ImageSeriesReader()
        dicom_files = reader.GetGDCMSeriesFileNames(str(dicom_dir))
        
        if not dicom_files:
            raise ValueError(f"No DICOM files found in {dicom_dir}")
        
        reader.SetFileNames(dicom_files)
        image = reader.Execute()
        
        # NumPy 배열로 변환
        volume = sitk.GetArrayFromImage(image)
        spacing = image.GetSpacing()  # (x, y, z)
        
        # NIfTI로 저장
        output_dir = dicom_dir.parent
        nifti_path = output_dir / 'volume.nii.gz'
        
        nifti_img = nib.Nifti1Image(volume, np.eye(4))
        nifti_img.header.set_zooms(spacing)
        nib.save(nifti_img, nifti_path)
        
        # DB 업데이트
        with self.db.cursor() as cur:
            cur.execute("""
                UPDATE ct_scans 
                SET nifti_path = %s 
                WHERE scan_id = %s
            """, (str(nifti_path), scan_id))
        self.db.commit()
        
        return nifti_path, spacing
    
    def detect_tumors_2d(self, volume: np.ndarray, spacing: Tuple) -> List[Dict]:
        """
        슬라이스별 2D 종양 검출 (기존 Rule-based 시스템)
        """
        results = []
        
        for z in range(volume.shape[0]):
            slice_data = volume[z, :, :]
            
            # 기존 검출 알고리즘 사용
            detections = detect_tumors_in_slice(
                slice_data,
                z_position=z * spacing[2],
                confidence_threshold=0.7
            )
            
            if detections:
                results.append({
                    'slice_index': z,
                    'detections': detections
                })
        
        return results
    
    def convert_to_3d(
        self, 
        detection_results: List[Dict],
        volume_shape: Tuple,
        spacing: Tuple
    ) -> List[Dict]:
        """
        2D 검출 결과 → 3D Connected Components
        
        Returns:
            List of {tumor_id, volume_mm3, centroid, bbox, ...}
        """
        logger.info("Converting 2D detections to 3D tumors...")
        
        # 1. 3D 마스크 생성
        mask_3d = create_3d_mask_from_2d_detections(
            detection_results,
            volume_shape,
            spacing
        )
        
        # 2. 3D Connected Components 분석
        tumors = segment_tumors_3d(
            mask_3d,
            spacing,
            min_volume_mm3=50.0,  # 50 mm³ = 0.05 cm³
            max_volume_mm3=500000.0  # 500 cm³
        )
        
        logger.info(f"Identified {len(tumors)} 3D tumors")
        
        return tumors
    
    def save_tumor_measurement(self, scan_id: int, tumor: Dict):
        """종양 측정값 DB 저장"""
        try:
            with self.db.cursor() as cur:
                cur.execute("""
                    INSERT INTO tumor_measurements (
                        scan_id, tumor_id, volume_mm3, max_diameter_mm,
                        centroid_x, centroid_y, centroid_z,
                        segmentation_method, confidence_score
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                """, (
                    scan_id,
                    tumor['tumor_id'],
                    tumor['volume_mm3'],
                    tumor.get('max_diameter_mm'),
                    tumor['centroid'][0],
                    tumor['centroid'][1],
                    tumor['centroid'][2],
                    'rule_based_3d_connected',
                    tumor.get('confidence', 0.85)
                ))
            self.db.commit()
            logger.info(f"Tumor measurement saved: {tumor['tumor_id']}")
        except Exception as e:
            self.db.rollback()
            logger.error(f"Failed to save tumor: {e}")
    
    def create_3d_visualization(
        self,
        scan_id: int,
        nifti_path: Path,
        tumors: List[Dict]
    ) -> Path:
        """
        3D 시각화 생성 (PyVista HTML)
        """
        logger.info("Creating 3D visualization...")
        
        # 종양 마스크 생성
        tumor_mask_path = nifti_path.parent / 'tumor_mask_3d.nii.gz'
        
        # TODO: 종양 마스크를 NIfTI로 저장하는 로직 추가
        
        # 3D HTML 뷰어 생성
        output_html = nifti_path.parent / '3d_viewer.html'
        
        self.viewer_3d.create_quick_tumor_view(
            tumor_mask_path=tumor_mask_path,
            output_html=output_html
        )
        
        logger.info(f"3D visualization created: {output_html}")
        return output_html
    
    def get_scan_info(self, scan_id: int) -> Dict:
        """스캔 정보 조회"""
        with self.db.cursor() as cur:
            cur.execute("""
                SELECT patient_id, timepoint, scan_date, dicom_path, nifti_path
                FROM ct_scans
                WHERE scan_id = %s
            """, (scan_id,))
            row = cur.fetchone()
        
        if not row:
            raise ValueError(f"Scan {scan_id} not found")
        
        return {
            'patient_id': row[0],
            'timepoint': row[1],
            'scan_date': row[2],
            'dicom_path': row[3],
            'nifti_path': row[4]
        }
    
    def update_scan_status(self, scan_id: int, processed: bool):
        """스캔 처리 상태 업데이트"""
        with self.db.cursor() as cur:
            cur.execute("""
                UPDATE ct_scans
                SET processed = %s, processing_method = %s
                WHERE scan_id = %s
            """, (processed, 'rule_based_3d_connected', scan_id))
        self.db.commit()


def main():
    """메인 서비스 루프"""
    logger.info("Starting CT Processor Service...")
    
    processor = CTProcessor()
    
    # Redis 큐에서 작업 받기 (TODO: Celery 통합)
    # 임시: pending 스캔 폴링
    
    while True:
        try:
            # Pending 스캔 조회
            with processor.db.cursor() as cur:
                cur.execute("""
                    SELECT scan_id 
                    FROM ct_scans 
                    WHERE processed = FALSE 
                    AND dicom_path IS NOT NULL
                    LIMIT 1
                """)
                row = cur.fetchone()
            
            if row:
                scan_id = row[0]
                logger.info(f"Processing scan: {scan_id}")
                
                try:
                    processor.process_scan(scan_id)
                except Exception as e:
                    logger.error(f"Scan {scan_id} processing failed: {e}")
            else:
                logger.debug("No pending scans, waiting...")
                time.sleep(30)
        
        except KeyboardInterrupt:
            logger.info("Service stopped")
            break
        except Exception as e:
            logger.error(f"Service error: {e}")
            time.sleep(10)


if __name__ == '__main__':
    main()
