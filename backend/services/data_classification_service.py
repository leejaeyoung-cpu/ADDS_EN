"""
ADDS 데이터 분류 서비스
목적: 인하대병원에서 수집되는 DICOM, 병리 이미지, 의사 소견 데이터를 자동 분류
"""

import os
import re
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import pydicom
import json
import hashlib
import psycopg2
from psycopg2.extras import Json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataClassificationService:
    """
    데이터 자동 분류 및 검증 서비스
    """
    
    def __init__(self, db_config: Dict, data_root: str):
        """
        Args:
            db_config: PostgreSQL 연결 설정
            data_root: 데이터 루트 디렉토리
        """
        self.db_config = db_config
        self.data_root = Path(data_root)
        self.conn = None
        
    def connect_db(self):
        """데이터베이스 연결"""
        try:
            self.conn = psycopg2.connect(**self.db_config)
            logger.info("Database connected successfully")
        except Exception as e:
            logger.error(f"Database connection failed: {e}")
            raise
    
    def close_db(self):
        """데이터베이스 연결 종료"""
        if self.conn:
            self.conn.close()
            logger.info("Database connection closed")
    
    # ========================================================================
    # DICOM 파일 분류
    # ========================================================================
    
    def classify_dicom_file(self, dicom_path: str) -> Dict:
        """
        DICOM 파일에서 메타데이터 추출 및 분류
        
        Returns:
            {
                'patient_id': str,
                'study_date': datetime,
                'modality': str,
                'study_uid': str,
                'series_uid': str,
                'slice_thickness': float,
                'pixel_spacing': list,
                'file_hash': str
            }
        """
        try:
            dcm = pydicom.dcmread(dicom_path)
            
            # 환자 ID 추출 (인하대병원 형식 고려)
            patient_id = self._extract_patient_id(dcm)
            
            # 촬영 날짜
            study_date = self._parse_dicom_date(dcm.StudyDate, dcm.StudyTime if hasattr(dcm, 'StudyTime') else None)
            
            # 파일 해시 (중복 방지)
            file_hash = self._calculate_file_hash(dicom_path)
            
            metadata = {
                'patient_id': patient_id,
                'study_date': study_date,
                'modality': dcm.Modality if hasattr(dcm, 'Modality') else 'CT',
                'study_uid': dcm.StudyInstanceUID,
                'series_uid': dcm.SeriesInstanceUID,
                'slice_thickness': float(dcm.SliceThickness) if hasattr(dcm, 'SliceThickness') else None,
                'pixel_spacing': [float(x) for x in dcm.PixelSpacing] if hasattr(dcm, 'PixelSpacing') else None,
                'file_path': str(dicom_path),
                'file_hash': file_hash
            }
            
            logger.info(f"DICOM classified: Patient {patient_id}, Date {study_date}")
            return metadata
            
        except Exception as e:
            logger.error(f"Failed to classify DICOM {dicom_path}: {e}")
            return None
    
    def _extract_patient_id(self, dcm) -> str:
        """
        환자 ID 추출 (인하대병원 형식 처리)
        예: INHA-2024-001, 2024001 등
        """
        if hasattr(dcm, 'PatientID'):
            raw_id = str(dcm.PatientID).strip()
            
            # 인하대병원 형식으로 정규화
            if not raw_id.startswith('INHA-'):
                # 숫자만 있는 경우
                if raw_id.isdigit():
                    year = datetime.now().year
                    raw_id = f"INHA-{year}-{raw_id.zfill(3)}"
            
            return raw_id
        
        raise ValueError("Patient ID not found in DICOM")
    
    def _parse_dicom_date(self, date_str: str, time_str: Optional[str] = None) -> datetime:
        """
        DICOM 날짜 형식 파싱
        Args:
            date_str: YYYYMMDD
            time_str: HHMMSS.ffffff (optional)
        """
        if not date_str:
            return None
        
        # 날짜 파싱
        date_obj = datetime.strptime(date_str, '%Y%m%d')
        
        # 시간 추가
        if time_str:
            time_clean = time_str.split('.')[0]  # microseconds 제거
            if len(time_clean) == 6:
                time_obj = datetime.strptime(time_clean, '%H%M%S').time()
                date_obj = datetime.combine(date_obj.date(), time_obj)
        
        return date_obj
    
    def _calculate_file_hash(self, file_path: str) -> str:
        """파일 해시 계산 (중복 체크용)"""
        hasher = hashlib.md5()
        with open(file_path, 'rb') as f:
            buf = f.read(65536)  # 64KB chunks
            while len(buf) > 0:
                hasher.update(buf)
                buf = f.read(65536)
        return hasher.hexdigest()
    
    # ========================================================================
    # 데이터베이스 저장
    # ========================================================================
    
    def save_ct_scan(self, metadata: Dict) -> Optional[str]:
        """
        CT 스캔 데이터를 데이터베이스에 저장
        
        Returns:
            scan_id (UUID) or None if failed
        """
        try:
            # 환자 ID로 patient UUID 조회
            patient_uuid = self._get_patient_uuid(metadata['patient_id'])
            
            if not patient_uuid:
                logger.warning(f"Patient {metadata['patient_id']} not found in database")
                return None
            
            cursor = self.conn.cursor()
            
            # 중복 체크
            cursor.execute("""
                SELECT scan_id FROM ct_scans 
                WHERE study_instance_uid = %s
            """, (metadata['study_uid'],))
            
            existing = cursor.fetchone()
            if existing:
                logger.info(f"CT scan {metadata['study_uid']} already exists")
                return existing[0]
            
            # 삽입
            cursor.execute("""
                INSERT INTO ct_scans (
                    patient_id, scan_date, scan_type, dicom_path,
                    study_instance_uid, series_instance_uid, modality,
                    slice_thickness, pixel_spacing
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                RETURNING scan_id
            """, (
                patient_uuid,
                metadata['study_date'],
                self._infer_scan_type(metadata),
                metadata['file_path'],
                metadata['study_uid'],
                metadata['series_uid'],
                metadata['modality'],
                metadata['slice_thickness'],
                metadata['pixel_spacing']
            ))
            
            scan_id = cursor.fetchone()[0]
            self.conn.commit()
            cursor.close()
            
            logger.info(f"CT scan saved: {scan_id}")
            return scan_id
            
        except Exception as e:
            logger.error(f"Failed to save CT scan: {e}")
            if self.conn:
                self.conn.rollback()
            return None
    
    def _get_patient_uuid(self, hospital_id: str) -> Optional[str]:
        """환자 hospital_id로 UUID 조회"""
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT patient_id FROM patients WHERE hospital_id = %s
        """, (hospital_id,))
        
        result = cursor.fetchone()
        cursor.close()
        
        return result[0] if result else None
    
    def _infer_scan_type(self, metadata: Dict) -> str:
        """
        스캔 타입 추론 (baseline, follow-up, restaging)
        환자의 이전 스캔과 비교하여 결정
        """
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT COUNT(*) FROM ct_scans 
            WHERE patient_id = (
                SELECT patient_id FROM patients WHERE hospital_id = %s
            )
        """, (metadata['patient_id'],))
        
        scan_count = cursor.fetchone()[0]
        cursor.close()
        
        if scan_count == 0:
            return 'baseline'
        else:
            return 'follow-up'
    
    # ========================================================================
    # 치료 세션 매칭
    # ========================================================================
    
    def match_scan_to_treatment(self, scan_id: str, tolerance_days: int = 30):
        """
        CT 스캔을 가장 가까운 치료 세션과 매칭
        
        Args:
            scan_id: CT 스캔 UUID
            tolerance_days: 매칭 허용 일수
        """
        try:
            cursor = self.conn.cursor()
            
            # 스캔 정보 조회
            cursor.execute("""
                SELECT patient_id, scan_date FROM ct_scans WHERE scan_id = %s
            """, (scan_id,))
            
            scan_info = cursor.fetchone()
            if not scan_info:
                return
            
            patient_id, scan_date = scan_info
            
            # 가장 가까운 치료 세션 찾기
            cursor.execute("""
                SELECT session_id, ABS(EXTRACT(DAY FROM (treatment_date - %s::date))) as day_diff
                FROM treatment_sessions
                WHERE patient_id = %s
                  AND ABS(EXTRACT(DAY FROM (treatment_date - %s::date))) <= %s
                ORDER BY day_diff ASC
                LIMIT 1
            """, (scan_date, patient_id, scan_date, tolerance_days))
            
            session = cursor.fetchone()
            
            if session:
                session_id, day_diff = session
                
                # CT 스캔에 session_id 업데이트
                cursor.execute("""
                    UPDATE ct_scans SET session_id = %s WHERE scan_id = %s
                """, (session_id, scan_id))
                
                self.conn.commit()
                logger.info(f"Matched scan {scan_id} to session {session_id} (±{day_diff} days)")
            else:
                logger.warning(f"No treatment session found for scan {scan_id} within {tolerance_days} days")
            
            cursor.close()
            
        except Exception as e:
            logger.error(f"Failed to match scan to treatment: {e}")
            if self.conn:
                self.conn.rollback()
    
    # ========================================================================
    # 데이터 검증
    # ========================================================================
    
    def validate_patient_data(self, patient_id: str) -> List[Dict]:
        """
        환자 데이터 검증 (누락, 중복, 이상치)
        
        Returns:
            List of validation issues
        """
        issues = []
        
        try:
            cursor = self.conn.cursor()
            
            # 1. CT 스캔이 있는데 종양 측정이 없는 경우
            cursor.execute("""
                SELECT ct.scan_id, ct.scan_date
                FROM ct_scans ct
                LEFT JOIN tumor_measurements tm ON ct.scan_id = tm.scan_id
                WHERE ct.patient_id = (SELECT patient_id FROM patients WHERE hospital_id = %s)
                  AND ct.analyzed = TRUE
                  AND tm.measurement_id IS NULL
            """, (patient_id,))
            
            for scan_id, scan_date in cursor.fetchall():
                issues.append({
                    'type': 'missing_data',
                    'severity': 'warning',
                    'table': 'tumor_measurements',
                    'description': f"CT scan {scan_date} analyzed but no tumor measurements found",
                    'scan_id': scan_id
                })
            
            # 2. 치료 세션이 연결되지 않은 CT 스캔
            cursor.execute("""
                SELECT scan_id, scan_date
                FROM ct_scans
                WHERE patient_id = (SELECT patient_id FROM patients WHERE hospital_id = %s)
                  AND session_id IS NULL
                  AND scan_type = 'follow-up'
            """, (patient_id,))
            
            for scan_id, scan_date in cursor.fetchall():
                issues.append({
                    'type': 'missing_link',
                    'severity': 'warning',
                    'table': 'ct_scans',
                    'description': f"Follow-up CT scan {scan_date} not linked to treatment session",
                    'scan_id': scan_id
                })
            
            # 3. 비정상적인 종양 크기 변화 (하루에 100% 이상 변화)
            cursor.execute("""
                SELECT tt.tracking_id, tt.volume_change_percent, tt.days_between_scans
                FROM tumor_tracking tt
                JOIN ct_scans ct ON ct.scan_id = (
                    SELECT scan_id FROM tumor_measurements WHERE measurement_id = tt.followup_measurement_id
                )
                WHERE ct.patient_id = (SELECT patient_id FROM patients WHERE hospital_id = %s)
                  AND ABS(tt.volume_change_percent / tt.days_between_scans) > 100
            """, (patient_id,))
            
            for tracking_id, change_pct, days in cursor.fetchall():
                issues.append({
                    'type': 'anomaly',
                    'severity': 'error',
                    'table': 'tumor_tracking',
                    'description': f"Abnormal tumor volume change: {change_pct:.1f}% in {days} days",
                    'tracking_id': tracking_id
                })
            
            cursor.close()
            
            # 데이터베이스에 기록
            self._log_validation_issues(patient_id, issues)
            
            return issues
            
        except Exception as e:
            logger.error(f"Validation failed for patient {patient_id}: {e}")
            return []
    
    def _log_validation_issues(self, patient_id: str, issues: List[Dict]):
        """검증 이슈를 데이터베이스에 기록"""
        if not issues:
            return
        
        try:
            cursor = self.conn.cursor()
            patient_uuid = self._get_patient_uuid(patient_id)
            
            for issue in issues:
                cursor.execute("""
                    INSERT INTO data_validation_log (
                        patient_id, validation_type, severity, table_name, 
                        record_id, issue_description
                    ) VALUES (%s, %s, %s, %s, %s, %s)
                """, (
                    patient_uuid,
                    issue['type'],
                    issue['severity'],
                    issue['table'],
                    issue.get('scan_id') or issue.get('tracking_id'),
                    issue['description']
                ))
            
            self.conn.commit()
            cursor.close()
            logger.info(f"Logged {len(issues)} validation issues for patient {patient_id}")
            
        except Exception as e:
            logger.error(f"Failed to log validation issues: {e}")
            if self.conn:
                self.conn.rollback()
    
    # ========================================================================
    # 배치 처리
    # ========================================================================
    
    def scan_and_classify_directory(self, directory: str, file_type: str = 'dicom'):
        """
        디렉토리를 스캔하여 모든 파일 분류
        
        Args:
            directory: 스캔할 디렉토리
            file_type: 'dicom', 'pathology', 'notes'
        """
        dir_path = Path(directory)
        
        if not dir_path.exists():
            logger.error(f"Directory not found: {directory}")
            return
        
        logger.info(f"Scanning directory: {directory}")
        
        if file_type == 'dicom':
            # DICOM 파일 찾기 (.dcm)
            dicom_files = list(dir_path.rglob('*.dcm'))
            logger.info(f"Found {len(dicom_files)} DICOM files")
            
            for dcm_file in dicom_files:
                metadata = self.classify_dicom_file(str(dcm_file))
                if metadata:
                    scan_id = self.save_ct_scan(metadata)
                    if scan_id:
                        self.match_scan_to_treatment(scan_id)
        
        logger.info("Directory scan complete")


# ============================================================================
# 사용 예시
# ============================================================================

if __name__ == "__main__":
    # 데이터베이스 설정
    db_config = {
        'host': 'localhost',
        'port': 5432,
        'database': 'adds_db',
        'user': 'adds_user',
        'password': 'your_password'
    }
    
    # 서비스 생성
    service = DataClassificationService(
        db_config=db_config,
        data_root=r"F:\ADDS\patient_data"
    )
    
    # 연결
    service.connect_db()
    
    try:
        # 디렉토리 스캔 및 분류
        service.scan_and_classify_directory(r"F:\ADDS\patient_data\incoming\dicom")
        
        # 환자 데이터 검증
        issues = service.validate_patient_data("INHA-2024-001")
        
        if issues:
            print(f"\n발견된 이슈: {len(issues)}개")
            for issue in issues:
                print(f"  [{issue['severity']}] {issue['description']}")
    
    finally:
        # 연결 종료
        service.close_db()
