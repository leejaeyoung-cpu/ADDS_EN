"""
Cell Image Analyzer Service
세포 이미지 분석 및 조직학적 특징 추출
"""

import os
import logging
from pathlib import Path
from typing import Dict, List
import time

import cv2
import numpy as np
from PIL import Image
import psycopg2
from psycopg2.extras import Json

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


class CellImageAnalyzer:
    """세포 이미지 분석"""
    
    def __init__(self):
        self.db = self.connect_db()
        logger.info("Cell Analyzer initialized")
    
    def connect_db(self):
        return psycopg2.connect(
            host=DB_HOST,
            port=DB_PORT,
            database=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD
        )
    
    def analyze_cell_image(self, image_id: int):
        """
        세포 이미지 분석
        
        추출 특징:
        1. 세포 밀도
        2. 핵 크기 분포
        3. 조직학적 등급 (Grade)
        4. Apoptosis 징후
        5. Necrosis 영역
        """
        logger.info(f"Analyzing cell image: {image_id}")
        
        # 이미지 정보 조회
        image_info = self.get_image_info(image_id)
        image_path = Path(image_info['file_path'])
        
        if not image_path.exists():
            logger.error(f"Image not found: {image_path}")
            return None
        
        # 이미지 로드
        img = cv2.imread(str(image_path))
        
        if img is None:
            logger.error(f"Failed to load image: {image_path}")
            return None
        
        # 특징 추출
        features = self.extract_features(img)
        
        # DB 저장
        self.save_features(image_id, features)
        
        logger.info(f"Cell image analyzed: {image_id}")
        
        return features
    
    def extract_features(self, img: np.ndarray) -> Dict:
        """
        조직학적 특징 추출
        
        간단한 버전:
        - 색상 통계
        - 텍스처 분석
        - 세포 카운트 (간단한 blob detection)
        """
        # Grayscale 변환
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # 1. 색상 통계
        mean_intensity = np.mean(gray)
        std_intensity = np.std(gray)
        
        # 2. 텍스처 (GLCM 간단 버전)
        texture_score = self.calculate_texture_score(gray)
        
        # 3. 세포 밀도 (blob detection)
        cell_count = self.count_cells(gray)
        
        # 4. 핵 크기 (평균)
        nucleus_size_avg = self.estimate_nucleus_size(gray)
        
        features = {
            'mean_intensity': float(mean_intensity),
            'std_intensity': float(std_intensity),
            'texture_score': float(texture_score),
            'cell_count_estimate': int(cell_count),
            'nucleus_size_avg_px': float(nucleus_size_avg),
            'analysis_version': 'v1.0_basic'
        }
        
        return features
    
    def calculate_texture_score(self, gray: np.ndarray) -> float:
        """텍스처 점수 (간단한 Laplacian variance)"""
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        variance = laplacian.var()
        return variance
    
    def count_cells(self, gray: np.ndarray) -> int:
        """세포 카운트 (SimpleBlobDetector)"""
        # 이진화
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # 크기 필터링 (너무 작거나 큰 것 제외)
        min_area = 50
        max_area = 5000
        
        valid_cells = [c for c in contours if min_area < cv2.contourArea(c) < max_area]
        
        return len(valid_cells)
    
    def estimate_nucleus_size(self, gray: np.ndarray) -> float:
        """핵 크기 추정"""
        # 간단한 방법: blob의 평균 크기
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return 0.0
        
        areas = [cv2.contourArea(c) for c in contours if cv2.contourArea(c) > 50]
        
        if not areas:
            return 0.0
        
        return np.mean(areas)
    
    def get_image_info(self, image_id: int) -> Dict:
        """이미지 정보 조회"""
        with self.db.cursor() as cur:
            cur.execute("""
                SELECT patient_id, image_type, file_path
                FROM cell_images
                WHERE image_id = %s
            """, (image_id,))
            row = cur.fetchone()
        
        if not row:
            raise ValueError(f"Image {image_id} not found")
        
        return {
            'patient_id': row[0],
            'image_type': row[1],
            'file_path': row[2]
        }
    
    def save_features(self, image_id: int, features: Dict):
        """특징 DB 저장"""
        try:
            with self.db.cursor() as cur:
                cur.execute("""
                    UPDATE cell_images
                    SET features = %s, analysis_complete = TRUE
                    WHERE image_id = %s
                """, (Json(features), image_id))
            self.db.commit()
            logger.info(f"Features saved for image {image_id}")
        except Exception as e:
            self.db.rollback()
            logger.error(f"Failed to save features: {e}")


def main():
    """메인 서비스"""
    logger.info("Starting Cell Analyzer Service...")
    
    analyzer = CellImageAnalyzer()
    
    while True:
        try:
            # Pending 이미지 조회
            with analyzer.db.cursor() as cur:
                cur.execute("""
                    SELECT image_id
                    FROM cell_images
                    WHERE analysis_complete = FALSE
                    LIMIT 1
                """)
                row = cur.fetchone()
            
            if row:
                image_id = row[0]
                analyzer.analyze_cell_image(image_id)
            else:
                logger.debug("No pending images")
                time.sleep(30)
        
        except KeyboardInterrupt:
            logger.info("Service stopped")
            break
        except Exception as e:
            logger.error(f"Service error: {e}")
            time.sleep(10)


if __name__ == '__main__':
    main()
