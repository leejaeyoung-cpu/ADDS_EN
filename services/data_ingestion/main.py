"""
Data Ingestion Service
인하대 데이터 수신 및 자동 분류
"""

import os
import time
import shutil
import logging
from pathlib import Path
from datetime import datetime
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

import pandas as pd
import psycopg2
from psycopg2.extras import Json
from minio import Minio

# 로깅 설정
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
DB_PASSWORD = os.getenv('DB_PASSWORD', 'adds_secure_2026')

MINIO_HOST = os.getenv('MINIO_HOST', 'minio')
MINIO_PORT = os.getenv('MINIO_PORT', '9000')
MINIO_USER = os.getenv('MINIO_USER', 'adds_admin')
MINIO_PASSWORD = os.getenv('MINIO_PASSWORD', 'adds_minio_2026')

INCOMING_DIR = Path('/incoming')
PROCESSED_DIR = Path('/processed/patients')


class DatabaseManager:
    """PostgreSQL 데이터베이스 관리"""
    
    def __init__(self):
        self.conn = None
        self.connect()
    
    def connect(self):
        """데이터베이스 연결"""
        try:
            self.conn = psycopg2.connect(
                host=DB_HOST,
                port=DB_PORT,
                database=DB_NAME,
                user=DB_USER,
                password=DB_PASSWORD
            )
            logger.info(f"Connected to database: {DB_NAME}")
        except Exception as e:
            logger.error(f"Database connection failed: {e}")
            raise
    
    def register_patient(self, patient_data):
        """환자 등록"""
        try:
            with self.conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO patients (
                        patient_id, age, sex, weight_kg, height_cm,
                        cancer_type, cancer_stage, biomarkers, enrollment_date
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (patient_id) DO UPDATE SET
                        updated_at = NOW()
                """, (
                    patient_data['patient_id'],
                    patient_data.get('age'),
                    patient_data.get('sex'),
                    patient_data.get('weight_kg'),
                    patient_data.get('height_cm'),
                    patient_data.get('cancer_type'),
                    patient_data.get('cancer_stage'),
                    Json(patient_data.get('biomarkers', {})),
                    patient_data.get('enrollment_date', datetime.now().date())
                ))
            self.conn.commit()
            logger.info(f"Patient registered: {patient_data['patient_id']}")
        except Exception as e:
            self.conn.rollback()
            logger.error(f"Failed to register patient: {e}")
            raise
    
    def register_ct_scan(self, scan_data):
        """CT 스캔 등록"""
        try:
            with self.conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO ct_scans (
                        patient_id, timepoint, scan_date, days_from_baseline,
                        scan_phase, dicom_path, minio_bucket, minio_object_key
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                    RETURNING scan_id
                """, (
                    scan_data['patient_id'],
                    scan_data['timepoint'],
                    scan_data['scan_date'],
                    scan_data.get('days_from_baseline', 0),
                    scan_data.get('scan_phase', 'unknown'),
                    scan_data['dicom_path'],
                    scan_data.get('minio_bucket'),
                    scan_data.get('minio_object_key')
                ))
                scan_id = cur.fetchone()[0]
            self.conn.commit()
            logger.info(f"CT scan registered: {scan_id} for {scan_data['patient_id']}")
            return scan_id
        except Exception as e:
            self.conn.rollback()
            logger.error(f"Failed to register CT scan: {e}")
            raise


class MinIOManager:
    """MinIO 오브젝트 스토리지 관리"""
    
    def __init__(self):
        self.client = Minio(
            f"{MINIO_HOST}:{MINIO_PORT}",
            access_key=MINIO_USER,
            secret_key=MINIO_PASSWORD,
            secure=False
        )
        self.ensure_buckets()
    
    def ensure_buckets(self):
        """버킷 생성"""
        buckets = ['ct-scans', 'cell-images', 'processed-data']
        for bucket in buckets:
            if not self.client.bucket_exists(bucket):
                self.client.make_bucket(bucket)
                logger.info(f"Created bucket: {bucket}")


class InhaDataHandler(FileSystemEventHandler):
    """인하대 데이터 감시 및 처리"""
    
    def __init__(self):
        self.db = DatabaseManager()
        self.minio = MinIOManager()
        self.baseline_dates = {}  # {patient_id: baseline_date}
    
    def on_created(self, event):
        """새 폴더/파일 감지"""
        if event.is_directory and 'batch_' in event.src_path:
            logger.info(f"New batch detected: {event.src_path}")
            time.sleep(2)  # 파일 복사 완료 대기
            self.process_batch(Path(event.src_path))
    
    def process_batch(self, batch_path):
        """배치 데이터 처리"""
        logger.info(f"Processing batch: {batch_path}")
        
        # 메타데이터 파일 읽기
        metadata_file = batch_path / 'patient_files.csv'
        if not metadata_file.exists():
            logger.warning(f"No metadata file found: {metadata_file}")
            return
        
        try:
            df = pd.read_csv(metadata_file)
            logger.info(f"Found {len(df)} patients in batch")
            
            for _, row in df.iterrows():
                patient_id = row['patient_id']
                self.process_patient(batch_path, patient_id, row)
        
        except Exception as e:
            logger.error(f"Batch processing failed: {e}")
    
    def process_patient(self, batch_path, patient_id, metadata):
        """환자별 데이터 처리"""
        logger.info(f"Processing patient: {patient_id}")
        
        src_dir = batch_path / patient_id
        if not src_dir.exists():
            logger.warning(f"Patient directory not found: {src_dir}")
            return
        
        # 환자 등록
        self.register_patient(patient_id, metadata)
        
        # DICOM 분류
        dicom_dir = src_dir / 'DICOM'
        if dicom_dir.exists():
            self.classify_dicoms(patient_id, dicom_dir)
        
        # 세포 이미지
        cell_dir = src_dir / 'cell_images'
        if cell_dir.exists():
            self.classify_cell_images(patient_id, cell_dir)
        
        # 임상 노트
        notes_dir = src_dir / 'clinical_notes'
        if notes_dir.exists():
            self.classify_notes(patient_id, notes_dir)
    
    def register_patient(self, patient_id, metadata):
        """환자 DB 등록"""
        patient_data = {
            'patient_id': patient_id,
            'age': metadata.get('age'),
            'sex': metadata.get('sex'),
            'weight_kg': metadata.get('weight_kg'),
            'height_cm': metadata.get('height_cm'),
            'cancer_type': metadata.get('cancer_type'),
            'cancer_stage': metadata.get('cancer_stage'),
            'biomarkers': {},
            'enrollment_date': datetime.now().date()
        }
        self.db.register_patient(patient_data)
    
    def classify_dicoms(self, patient_id, dicom_dir):
        """DICOM 시점별 분류"""
        logger.info(f"Classifying DICOMs for {patient_id}")
        
        for folder in dicom_dir.iterdir():
            if not folder.is_dir():
                continue
            
            # 날짜 및 라벨 추출 (예: '2026-01-15_baseline')
            parts = folder.name.split('_')
            if len(parts) < 1:
                continue
            
            date_str = parts[0]
            label = parts[1] if len(parts) > 1 else 'unknown'
            
            try:
                scan_date = datetime.strptime(date_str, '%Y-%m-%d').date()
            except ValueError:
                logger.warning(f"Invalid date format: {date_str}")
                continue
            
            # Timepoint 결정
            timepoint = self.determine_timepoint(patient_id, scan_date, label)
            
            # 목적지 폴더
            dst_dir = PROCESSED_DIR / patient_id / 'imaging' / 'CT' / timepoint
            dst_dir.mkdir(parents=True, exist_ok=True)
            
            # 복사
            dicom_dst = dst_dir / 'dicom'
            if dicom_dst.exists():
                shutil.rmtree(dicom_dst)
            shutil.copytree(folder, dicom_dst)
            
            # DB 등록
            scan_data = {
                'patient_id': patient_id,
                'timepoint': timepoint,
                'scan_date': scan_date,
                'days_from_baseline': self.days_from_baseline(patient_id, scan_date),
                'dicom_path': str(dicom_dst),
                'minio_bucket': 'ct-scans',
                'minio_object_key': f"{patient_id}/{timepoint}/dicom"
            }
            self.db.register_ct_scan(scan_data)
            
            logger.info(f"Classified: {patient_id}/{timepoint}")
    
    def determine_timepoint(self, patient_id, scan_date, label):
        """Timepoint 결정"""
        # Baseline 설정
        if 'baseline' in label.lower() or patient_id not in self.baseline_dates:
            self.baseline_dates[patient_id] = scan_date
            return 'T0_baseline'
        
        # Days from baseline
        baseline = self.baseline_dates[patient_id]
        days = (scan_date - baseline).days
        
        # Timepoint 라벨링
        if days <= 7:
            return 'T0_baseline'
        elif days <= 21:
            return 'T1_week1-3'
        elif days <= 42:
            return 'T2_week4-6'
        elif days <= 84:
            return 'T3_week7-12'
        else:
            week = days // 7
            return f'T{week//4}_week{week}'
    
    def days_from_baseline(self, patient_id, scan_date):
        """Baseline으로부터 경과일"""
        if patient_id not in self.baseline_dates:
            return 0
        return (scan_date - self.baseline_dates[patient_id]).days
    
    def classify_cell_images(self, patient_id, cell_dir):
        """세포 이미지 분류"""
        logger.info(f"Classifying cell images for {patient_id}")
        # TODO: 세포 이미지 처리 로직
        pass
    
    def classify_notes(self, patient_id, notes_dir):
        """임상 노트 분류"""
        logger.info(f"Classifying clinical notes for {patient_id}")
        # TODO: 노트 처리 로직
        pass


def main():
    """메인 서비스 시작"""
    logger.info("Starting Data Ingestion Service...")
    
    # 디렉토리 생성
    INCOMING_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    
    # 파일 감시 시작
    event_handler = InhaDataHandler()
    observer = Observer()
    observer.schedule(event_handler, str(INCOMING_DIR), recursive=True)
    observer.start()
    
    logger.info(f"Watching directory: {INCOMING_DIR}")
    
    try:
        while True:
            time.sleep(10)
    except KeyboardInterrupt:
        observer.stop()
        logger.info("Service stopped")
    
    observer.join()


if __name__ == '__main__':
    main()
