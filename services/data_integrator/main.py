"""
Data Integrator Service
환자 Timeline 생성 및 물리학 기반 에너지 추정

핵심 기능:
1. CT 시계열 데이터 통합
2. 종양 부피 변화 추적
3. 물리학 기반 에너지 추정
4. 항암제 농도 역추론
"""

import os
import sys
import logging
import time
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import numpy as np
import psycopg2
from psycopg2.extras import Json
import redis
import json

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

REDIS_HOST = os.getenv('REDIS_HOST', 'redis')
REDIS_PORT = os.getenv('REDIS_PORT', '6379')

# 물리 상수
ENERGY_MODEL_VERSION = 'v1.0_linear'
CALIBRATION_CONSTANT_K = 0.01  # J per mm³ (임시, 실험 데이터로 보정 필요)


class DataIntegrator:
    """환자 데이터 통합 및 에너지 추정 엔진"""
    
    def __init__(self):
        self.db = self.connect_db()
        self.redis = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)
        logger.info("Data Integrator initialized")
    
    def connect_db(self):
        """데이터베이스 연결"""
        return psycopg2.connect(
            host=DB_HOST,
            port=DB_PORT,
            database=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD
        )
    
    def build_patient_timeline(self, patient_id: str) -> Dict:
        """
        환자 Timeline 생성
        
        Returns:
            {
                'patient_id': 'PT001',
                'timepoints': [
                    {
                        'timepoint': 'T0',
                        'scan_date': '2026-01-15',
                        'days_from_baseline': 0,
                        'total_volume_mm3': 24155,
                        'tumor_count': 2,
                        'tumors': [...]
                    },
                    ...
                ]
            }
        """
        logger.info(f"Building timeline for patient: {patient_id}")
        
        # 1. 모든 CT 스캔 조회
        scans = self.get_ct_scans(patient_id)
        
        if not scans:
            logger.warning(f"No CT scans found for {patient_id}")
            return {'patient_id': patient_id, 'timepoints': []}
        
        # 2. 각 스캔의 종양 측정값
        timeline = []
        
        for scan in scans:
            measurements = self.get_tumor_measurements(scan['scan_id'])
            
            if not measurements:
                continue
            
            total_volume = sum(m['volume_mm3'] for m in measurements)
            
            timepoint_data = {
                'scan_id': scan['scan_id'],
                'timepoint': scan['timepoint'],
                'scan_date': str(scan['scan_date']),
                'days_from_baseline': scan['days_from_baseline'],
                'total_volume_mm3': round(total_volume, 2),
                'tumor_count': len(measurements),
                'tumors': measurements
            }
            
            timeline.append(timepoint_data)
        
        # 3. 시간순 정렬
        timeline.sort(key=lambda x: x['days_from_baseline'])
        
        logger.info(f"Timeline built: {len(timeline)} timepoints")
        
        return {
            'patient_id': patient_id,
            'timepoints': timeline
        }
    
    def calculate_energy_estimates(self, patient_id: str) -> List[Dict]:
        """
        에너지 추정 계산
        
        물리학 기반 모델:
        1. 종양 부피 변화 측정 (ΔV)
        2. 변화율 계산 (dV/dt)
        3. 에너지 추정 (E = f(ΔV, dV/dt))
        4. 농도 역추론 (C = g(E))
        
        Returns:
            List of energy estimates for each timepoint
        """
        logger.info(f"Calculating energy estimates for {patient_id}")
        
        # Timeline 가져오기
        timeline_data = self.build_patient_timeline(patient_id)
        timepoints = timeline_data['timepoints']
        
        if len(timepoints) < 2:
            logger.warning(f"Not enough timepoints for energy estimation (need ≥2)")
            return []
        
        # Baseline (T0)
        baseline = timepoints[0]
        baseline_volume = baseline['total_volume_mm3']
        
        energy_estimates = []
        
        # T1, T2, ... 각각에 대해 에너지 추정
        for i, current in enumerate(timepoints[1:], 1):
            # 1. 부피 변화
            current_volume = current['total_volume_mm3']
            delta_v = baseline_volume - current_volume  # 양수 = 감소
            
            # 2. 경과 시간
            delta_t = current['days_from_baseline']
            
            if delta_t == 0:
                continue
            
            # 3. 변화율
            volume_change_rate = delta_v / delta_t  # mm³/day
            
            # 4. 에너지 추정 (v1.0: 단순 선형 모델)
            estimated_energy = self.volume_change_to_energy(
                delta_v=delta_v,
                rate=volume_change_rate,
                baseline_volume=baseline_volume
            )
            
            # 5. 농도 역추론 (v1.0: 경험적 관계식)
            inferred_concentration = self.energy_to_concentration(
                energy=estimated_energy,
                baseline_volume=baseline_volume
            )
            
            # 6. 신뢰도 계산
            confidence = self.calculate_confidence(
                delta_v=delta_v,
                baseline_volume=baseline_volume,
                tumor_count=current['tumor_count']
            )
            
            estimate = {
                'patient_id': patient_id,
                'timepoint': current['timepoint'],
                'baseline_volume_mm3': baseline_volume,
                'current_volume_mm3': current_volume,
                'volume_change_mm3': delta_v,
                'days_elapsed': delta_t,
                'change_rate_mm3_per_day': round(volume_change_rate, 4),
                'estimated_energy_J': round(estimated_energy, 6),
                'inferred_concentration_uM': round(inferred_concentration, 4),
                'model_version': ENERGY_MODEL_VERSION,
                'confidence_score': round(confidence, 3)
            }
            
            # DB 저장
            self.save_energy_estimate(estimate)
            
            energy_estimates.append(estimate)
        
        logger.info(f"Energy estimates calculated: {len(energy_estimates)}")
        
        return energy_estimates
    
    def volume_change_to_energy(
        self, 
        delta_v: float, 
        rate: float, 
        baseline_volume: float
    ) -> float:
        """
        부피 변화 → 에너지 추정
        
        v1.0 모델 (단순 선형):
        E = k * |ΔV| * (1 + rate_factor)
        
        향후 개선:
        - 시그널 패스웨이 에너지 장벽 고려
        - 종양 특성 (암 종류, 단계) 반영
        - 비선형 모델 (Gompertz, Logistic)
        """
        # 정규화된 변화율 (baseline 대비)
        normalized_rate = abs(rate) / (baseline_volume + 1e-6)
        
        # 에너지 = 부피 변화 × (1 + 속도 기여도)
        energy = CALIBRATION_CONSTANT_K * abs(delta_v) * (1 + normalized_rate * 100)
        
        return energy
    
    def energy_to_concentration(self, energy: float, baseline_volume: float) -> float:
        """
        에너지 → 항암제 농도 역추론
        
        v1.0 모델:
        C = (E / E_ref) * C_ref
        
        가정:
        - 기준 에너지 (E_ref) = 1.0 J
        - 기준 농도 (C_ref) = 10 μM (FOLFOX 평균)
        """
        E_ref = 1.0  # J
        C_ref = 10.0  # μM
        
        # 선형 비례 (임시)
        concentration = (energy / E_ref) * C_ref
        
        # 부피 스케일링 (큰 종양 = 높은 농도 필요)
        volume_factor = 1 + (baseline_volume / 50000)  # 50,000 mm³ 기준
        
        return concentration * volume_factor
    
    def calculate_confidence(
        self, 
        delta_v: float, 
        baseline_volume: float,
        tumor_count: int
    ) -> float:
        """
        추정 신뢰도 계산
        
        기준:
        - 부피 변화가 클수록 신뢰도 높음
        - 종양 개수가 많을수록 통계적 신뢰도 높음
        - Baseline 부피가 충분히 클수록 측정 정확도 높음
        """
        # 1. 변화율 기반 (5% 이상 변화 = 유의미)
        change_ratio = abs(delta_v) / (baseline_volume + 1e-6)
        change_confidence = min(change_ratio / 0.05, 1.0)  # 5% = 1.0
        
        # 2. 종양 개수 기반 (많을수록 통계적으로 신뢰)
        count_confidence = min(tumor_count / 3.0, 1.0)  # 3개 = 1.0
        
        # 3. Baseline 부피 기반 (충분히 큰가?)
        volume_confidence = min(baseline_volume / 10000, 1.0)  # 10,000 mm³ = 1.0
        
        # 종합 신뢰도
        confidence = (change_confidence * 0.5 + 
                     count_confidence * 0.3 + 
                     volume_confidence * 0.2)
        
        return confidence
    
    def get_ct_scans(self, patient_id: str) -> List[Dict]:
        """CT 스캔 조회"""
        with self.db.cursor() as cur:
            cur.execute("""
                SELECT scan_id, timepoint, scan_date, days_from_baseline
                FROM ct_scans
                WHERE patient_id = %s
                ORDER BY days_from_baseline
            """, (patient_id,))
            rows = cur.fetchall()
        
        return [
            {
                'scan_id': row[0],
                'timepoint': row[1],
                'scan_date': row[2],
                'days_from_baseline': row[3]
            }
            for row in rows
        ]
    
    def get_tumor_measurements(self, scan_id: int) -> List[Dict]:
        """종양 측정값 조회"""
        with self.db.cursor() as cur:
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
                'centroid': [row[3], row[4], row[5]]
            }
            for row in rows
        ]
    
    def save_energy_estimate(self, estimate: Dict):
        """에너지 추정값 DB 저장"""
        try:
            with self.db.cursor() as cur:
                cur.execute("""
                    INSERT INTO energy_estimates (
                        patient_id, timepoint,
                        baseline_volume_mm3, current_volume_mm3,
                        volume_change_mm3, days_elapsed, change_rate_mm3_per_day,
                        estimated_energy_J, inferred_concentration_uM,
                        model_version, confidence_score
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (patient_id, timepoint) DO UPDATE SET
                        estimated_energy_J = EXCLUDED.estimated_energy_J,
                        inferred_concentration_uM = EXCLUDED.inferred_concentration_uM,
                        confidence_score = EXCLUDED.confidence_score
                """, (
                    estimate['patient_id'],
                    estimate['timepoint'],
                    estimate['baseline_volume_mm3'],
                    estimate['current_volume_mm3'],
                    estimate['volume_change_mm3'],
                    estimate['days_elapsed'],
                    estimate['change_rate_mm3_per_day'],
                    estimate['estimated_energy_J'],
                    estimate['inferred_concentration_uM'],
                    estimate['model_version'],
                    estimate['confidence_score']
                ))
            self.db.commit()
            logger.info(f"Energy estimate saved: {estimate['patient_id']} {estimate['timepoint']}")
        except Exception as e:
            self.db.rollback()
            logger.error(f"Failed to save energy estimate: {e}")
    
    def process_patient(self, patient_id: str):
        """환자 전체 처리 (Timeline + Energy)"""
        logger.info(f"Processing patient: {patient_id}")
        
        try:
            # 1. Timeline 생성
            timeline = self.build_patient_timeline(patient_id)
            
            # 2. 에너지 추정
            energy_estimates = self.calculate_energy_estimates(patient_id)
            
            # 3. Redis에 캐시 (빠른 조회용)
            cache_data = {
                'timeline': timeline,
                'energy_estimates': energy_estimates,
                'updated_at': datetime.now().isoformat()
            }
            self.redis.setex(
                f"patient:{patient_id}:integrated",
                3600,  # 1시간 TTL
                json.dumps(cache_data, default=str)
            )
            
            logger.info(f"Patient processed: {patient_id}")
            
            return {
                'status': 'success',
                'patient_id': patient_id,
                'timepoints': len(timeline['timepoints']),
                'energy_estimates': len(energy_estimates)
            }
        
        except Exception as e:
            logger.error(f"Patient processing failed: {e}")
            return {
                'status': 'error',
                'patient_id': patient_id,
                'error': str(e)
            }


def main():
    """메인 서비스 루프"""
    logger.info("Starting Data Integrator Service...")
    
    integrator = DataIntegrator()
    
    while True:
        try:
            # Pending 환자 조회
            with integrator.db.cursor() as cur:
                cur.execute("""
                    SELECT DISTINCT ct.patient_id
                    FROM ct_scans ct
                    LEFT JOIN energy_estimates ee 
                        ON ct.patient_id = ee.patient_id 
                        AND ct.timepoint = ee.timepoint
                    WHERE ct.processed = TRUE
                    AND ee.estimate_id IS NULL
                    LIMIT 10
                """)
                rows = cur.fetchall()
            
            if rows:
                for row in rows:
                    patient_id = row[0]
                    logger.info(f"Processing patient: {patient_id}")
                    integrator.process_patient(patient_id)
            else:
                logger.debug("No pending patients, waiting...")
                time.sleep(30)
        
        except KeyboardInterrupt:
            logger.info("Service stopped")
            break
        except Exception as e:
            logger.error(f"Service error: {e}")
            time.sleep(10)


if __name__ == '__main__':
    main()
