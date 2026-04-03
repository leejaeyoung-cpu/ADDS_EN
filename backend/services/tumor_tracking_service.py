"""
Time-Series Tumor Tracking Service
시계열 종양 추적 및 비교 분석

기능:
1. 3D 정합 (Registration) - Baseline과 Follow-up CT 정렬
2. 종양 매칭 - 동일 종양 식별
3. 변화 분석 - 부피, 위치 변화
4. RECIST 자동 평가
"""

import numpy as np
import SimpleITK as sitk
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RECISTResponse(Enum):
    """RECIST 1.1 반응 평가"""
    CR = "Complete Response"       # 완전 소실
    PR = "Partial Response"        # ≥30% 감소
    SD = "Stable Disease"          # -30% ~ +20%
    PD = "Progressive Disease"     # ≥20% 증가


@dataclass
class Tumor3DCoordinate:
    """종양 3D 좌표"""
    tumor_id: str
    centroid_mm: Tuple[float, float, float]
    bbox_min_mm: Tuple[float, float, float]
    bbox_max_mm: Tuple[float, float, float]
    volume_cm3: float
    longest_diameter_mm: float


@dataclass
class TumorMatch:
    """종양 매칭 결과"""
    baseline_tumor: Tumor3DCoordinate
    followup_tumor: Tumor3DCoordinate
    distance_moved_mm: float
    volume_change_percent: float
    diameter_change_percent: float
    recist_response: RECISTResponse
    is_same_tumor: bool  # 신뢰도 기반 판단


@dataclass
class RegistrationResult:
    """정합 결과"""
    transform_matrix: np.ndarray
    registration_metric: float
    registered_volume: np.ndarray
    success: bool


class TumorTrackingService:
    """시계열 종양 추적 서비스"""
    
    def __init__(self):
        self.registration_result = None
    
    # ========================================================================
    # 1. 3D 정합 (Registration)
    # ========================================================================
    
    def register_ct_scans(
        self,
        baseline_nifti: str,
        followup_nifti: str,
        registration_type: str = 'rigid'
    ) -> RegistrationResult:
        """
        두 CT 스캔을 정합하여 같은 좌표계로 변환
        
        Args:
            baseline_nifti: Baseline CT (고정 이미지)
            followup_nifti: Follow-up CT (이동 이미지)
            registration_type: 'rigid', 'affine', 'bspline'
            
        Returns:
            RegistrationResult
        """
        logger.info(f"Registering CT scans: {registration_type}")
        
        try:
            # 이미지 로드
            fixed_image = sitk.ReadImage(baseline_nifti, sitk.sitkFloat32)
            moving_image = sitk.ReadImage(followup_nifti, sitk.sitkFloat32)
            
            logger.info(f"Fixed image size: {fixed_image.GetSize()}")
            logger.info(f"Moving image size: {moving_image.GetSize()}")
            
            # Registration method
            registration_method = sitk.ImageRegistrationMethod()
            
            # Metric (유사도 측정)
            registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
            registration_method.SetMetricSamplingStrategy(registration_method.RANDOM)
            registration_method.SetMetricSamplingPercentage(0.01)
            
            # Interpolator
            registration_method.SetInterpolator(sitk.sitkLinear)
            
            # Optimizer
            if registration_type == 'rigid':
                # Rigid transform (회전 + 이동)
                initial_transform = sitk.CenteredTransformInitializer(
                    fixed_image,
                    moving_image,
                    sitk.Euler3DTransform(),
                    sitk.CenteredTransformInitializerFilter.GEOMETRY
                )
                
                registration_method.SetInitialTransform(initial_transform, inPlace=False)
                registration_method.SetOptimizerAsGradientDescent(
                    learningRate=1.0,
                    numberOfIterations=100,
                    convergenceMinimumValue=1e-6,
                    convergenceWindowSize=10
                )
                
            elif registration_type == 'affine':
                # Affine transform (회전 + 이동 + 스케일 + 전단)
                initial_transform = sitk.CenteredTransformInitializer(
                    fixed_image,
                    moving_image,
                    sitk.AffineTransform(3),
                    sitk.CenteredTransformInitializerFilter.GEOMETRY
                )
                
                registration_method.SetInitialTransform(initial_transform, inPlace=False)
                registration_method.SetOptimizerAsGradientDescent(
                    learningRate=1.0,
                    numberOfIterations=200,
                    convergenceMinimumValue=1e-6,
                    convergenceWindowSize=10
                )
            
            else:
                raise ValueError(f"Unsupported registration type: {registration_type}")
            
            registration_method.SetOptimizerScalesFromPhysicalShift()
            
            # Multi-resolution strategy
            registration_method.SetShrinkFactorsPerLevel(shrinkFactors=[4, 2, 1])
            registration_method.SetSmoothingSigmasPerLevel(smoothingSigmas=[2, 1, 0])
            registration_method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
            
            # Execute registration
            logger.info("Starting registration...")
            final_transform = registration_method.Execute(fixed_image, moving_image)
            
            # Registration metric
            final_metric = registration_method.GetMetricValue()
            logger.info(f"Final metric value: {final_metric}")
            
            # Apply transform to moving image
            registered_image = sitk.Resample(
                moving_image,
                fixed_image,
                final_transform,
                sitk.sitkLinear,
                0.0,
                moving_image.GetPixelID()
            )
            
            # Transform matrix (4x4)
            transform_matrix = self._get_transform_matrix(final_transform)
            
            # NumPy array
            registered_volume = sitk.GetArrayFromImage(registered_image)
            
            self.registration_result = RegistrationResult(
                transform_matrix=transform_matrix,
                registration_metric=final_metric,
                registered_volume=registered_volume,
                success=True
            )
            
            logger.info("Registration completed successfully")
            return self.registration_result
            
        except Exception as e:
            logger.error(f"Registration failed: {e}")
            return RegistrationResult(
                transform_matrix=np.eye(4),
                registration_metric=float('inf'),
                registered_volume=None,
                success=False
            )
    
    def _get_transform_matrix(self, transform: sitk.Transform) -> np.ndarray:
        """SimpleITK Transform → 4x4 행렬"""
        # Affine/Rigid transform의 경우
        if hasattr(transform, 'GetMatrix'):
            # 3x3 rotation matrix
            rotation = np.array(transform.GetMatrix()).reshape(3, 3)
            # 3x1 translation vector
            translation = np.array(transform.GetTranslation()).reshape(3, 1)
            
            # 4x4 homogeneous matrix
            matrix = np.eye(4)
            matrix[:3, :3] = rotation
            matrix[:3, 3] = translation.flatten()
            
            return matrix
        else:
            return np.eye(4)
    
    # ========================================================================
    # 2. 종양 매칭
    # ========================================================================
    
    def match_tumors_across_scans(
        self,
        baseline_tumors: List[Tumor3DCoordinate],
        followup_tumors: List[Tumor3DCoordinate],
        max_distance_mm: float = 30.0,
        min_volume_ratio: float = 0.2
    ) -> List[TumorMatch]:
        """
        Baseline과 Follow-up 스캔의 종양 매칭
        
        Args:
            baseline_tumors: Baseline 종양 리스트
            followup_tumors: Follow-up 종양 리스트
            max_distance_mm: 최대 허용 거리 (mm)
            min_volume_ratio: 최소 부피 비율 (작은 것 / 큰 것)
            
        Returns:
            매칭된 종양 쌍 리스트
        """
        logger.info(f"Matching {len(baseline_tumors)} baseline tumors with {len(followup_tumors)} followup tumors")
        
        matches = []
        matched_followup_ids = set()
        
        # Baseline 종양마다 가장 가까운 Follow-up 종양 찾기
        for b_tumor in baseline_tumors:
            best_match = None
            best_distance = float('inf')
            
            for f_tumor in followup_tumors:
                if f_tumor.tumor_id in matched_followup_ids:
                    continue
                
                # 중심점 거리 계산
                distance = self._calculate_distance(
                    b_tumor.centroid_mm,
                    f_tumor.centroid_mm
                )
                
                # 거리 조건
                if distance > max_distance_mm:
                    continue
                
                # 부피 비율 조건 (너무 다르면 다른 종양으로 판단)
                volume_ratio = min(b_tumor.volume_cm3, f_tumor.volume_cm3) / max(b_tumor.volume_cm3, f_tumor.volume_cm3)
                if volume_ratio < min_volume_ratio:
                    continue
                
                # 가장 가까운 것 선택
                if distance < best_distance:
                    best_distance = distance
                    best_match = f_tumor
            
            # 매칭 성공
            if best_match:
                # 변화율 계산
                volume_change_pct = (best_match.volume_cm3 - b_tumor.volume_cm3) / b_tumor.volume_cm3 * 100
                diameter_change_pct = (best_match.longest_diameter_mm - b_tumor.longest_diameter_mm) / b_tumor.longest_diameter_mm * 100
                
                # RECIST 평가
                recist = self._evaluate_recist(volume_change_pct)
                
                match = TumorMatch(
                    baseline_tumor=b_tumor,
                    followup_tumor=best_match,
                    distance_moved_mm=best_distance,
                    volume_change_percent=volume_change_pct,
                    diameter_change_percent=diameter_change_pct,
                    recist_response=recist,
                    is_same_tumor=True
                )
                
                matches.append(match)
                matched_followup_ids.add(best_match.tumor_id)
                
                logger.info(f"Matched: {b_tumor.tumor_id} ↔ {best_match.tumor_id} "
                           f"(distance: {best_distance:.1f} mm, volume change: {volume_change_pct:+.1f}%)")
        
        # 매칭 안 된 종양들 처리
        unmatched_baseline = [t for t in baseline_tumors if not any(m.baseline_tumor.tumor_id == t.tumor_id for m in matches)]
        unmatched_followup = [t for t in followup_tumors if t.tumor_id not in matched_followup_ids]
        
        if unmatched_baseline:
            logger.warning(f"{len(unmatched_baseline)} baseline tumors disappeared (CR or measurement error)")
        
        if unmatched_followup:
            logger.warning(f"{len(unmatched_followup)} new tumors appeared (metastasis or new detection)")
        
        logger.info(f"Successfully matched {len(matches)} tumor pairs")
        
        return matches
    
    def _calculate_distance(
        self,
        point1: Tuple[float, float, float],
        point2: Tuple[float, float, float]
    ) -> float:
        """3D 유클리드 거리"""
        return np.linalg.norm(np.array(point1) - np.array(point2))
    
    def _evaluate_recist(self, volume_change_percent: float) -> RECISTResponse:
        """
        RECIST 1.1 평가
        
        실제로는 longest diameter 변화를 사용하지만,
        여기서는 부피 변화로 근사
        """
        if volume_change_percent < -90:  # 거의 사라짐
            return RECISTResponse.CR
        elif volume_change_percent <= -30:
            return RECISTResponse.PR
        elif volume_change_percent < 20:
            return RECISTResponse.SD
        else:
            return RECISTResponse.PD
    
    # ========================================================================
    # 3. 전체 환자 평가
    # ========================================================================
    
    def evaluate_patient_response(
        self,
        tumor_matches: List[TumorMatch]
    ) -> Dict:
        """
        환자 전체 반응 평가 (모든 종양 종합)
        
        Returns:
            {
                'overall_response': RECISTResponse,
                'total_tumor_volume_baseline': float,
                'total_tumor_volume_followup': float,
                'volume_change_percent': float,
                'num_tumors_baseline': int,
                'num_tumors_followup': int,
                'num_cr': int,
                'num_pr': int,
                'num_sd': int,
                'num_pd': int
            }
        """
        if not tumor_matches:
            return {
                'overall_response': RECISTResponse.SD,
                'total_tumor_volume_baseline': 0,
                'total_tumor_volume_followup': 0,
                'volume_change_percent': 0,
                'num_tumors_baseline': 0,
                'num_tumors_followup': 0,
                'num_cr': 0,
                'num_pr': 0,
                'num_sd': 0,
                'num_pd': 0
            }
        
        # 전체 부피 합계
        total_baseline = sum(m.baseline_tumor.volume_cm3 for m in tumor_matches)
        total_followup = sum(m.followup_tumor.volume_cm3 for m in tumor_matches)
        
        volume_change_pct = (total_followup - total_baseline) / total_baseline * 100
        
        # 반응별 개수
        response_counts = {
            RECISTResponse.CR: 0,
            RECISTResponse.PR: 0,
            RECISTResponse.SD: 0,
            RECISTResponse.PD: 0
        }
        
        for match in tumor_matches:
            response_counts[match.recist_response] += 1
        
        # 전체 평가 (최악의 경우 우선)
        if response_counts[RECISTResponse.PD] > 0:
            overall = RECISTResponse.PD
        elif response_counts[RECISTResponse.SD] > 0:
            overall = RECISTResponse.SD
        elif response_counts[RECISTResponse.PR] > 0:
            overall = RECISTResponse.PR
        else:
            overall = RECISTResponse.CR
        
        return {
            'overall_response': overall,
            'total_tumor_volume_baseline': total_baseline,
            'total_tumor_volume_followup': total_followup,
            'volume_change_percent': volume_change_pct,
            'num_tumors_baseline': len(tumor_matches),
            'num_tumors_followup': len(tumor_matches),
            'num_cr': response_counts[RECISTResponse.CR],
            'num_pr': response_counts[RECISTResponse.PR],
            'num_sd': response_counts[RECISTResponse.SD],
            'num_pd': response_counts[RECISTResponse.PD]
        }
    
    # ========================================================================
    # 4. 시각화 데이터 생성
    # ========================================================================
    
    def generate_comparison_data(
        self,
        tumor_matches: List[TumorMatch]
    ) -> Dict:
        """
        Before/After 비교 시각화용 데이터 생성
        
        Returns:
            {
                'arrows': [  # 종양 이동 화살표
                    {
                        'start': [x, y, z],
                        'end': [x, y, z],
                        'color': 'green' | 'yellow' | 'red'
                    }
                ],
                'changes': [  # 종양별 변화
                    {
                        'tumor_id': str,
                        'volume_change': float,
                        'position_change': float,
                        'response': str
                    }
                ]
            }
        """
        arrows = []
        changes = []
        
        for match in tumor_matches:
            # 화살표 (baseline → followup)
            color = self._get_response_color(match.recist_response)
            
            arrows.append({
                'start': list(match.baseline_tumor.centroid_mm),
                'end': list(match.followup_tumor.centroid_mm),
                'color': color,
                'thickness': min(5, max(1, match.baseline_tumor.volume_cm3 / 5))  # 부피에 비례
            })
            
            # 변화 요약
            changes.append({
                'tumor_id': match.baseline_tumor.tumor_id,
                'volume_change': match.volume_change_percent,
                'position_change': match.distance_moved_mm,
                'response': match.recist_response.value,
                'response_code': match.recist_response.name
            })
        
        return {
            'arrows': arrows,
            'changes': changes
        }
    
    def _get_response_color(self, response: RECISTResponse) -> str:
        """RECIST 반응에 따른 컬러"""
        color_map = {
            RECISTResponse.CR: '#00ff00',  # 녹색 (완전 관해)
            RECISTResponse.PR: '#90ee90',  # 연두 (부분 관해)
            RECISTResponse.SD: '#ffff00',  # 노랑 (안정)
            RECISTResponse.PD: '#ff0000'   # 빨강 (진행)
        }
        return color_map.get(response, '#808080')


# ============================================================================
# 사용 예시
# ============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("Time-Series Tumor Tracking Test")
    print("=" * 80)
    
    # 서비스 생성
    service = TumorTrackingService()
    
    # 테스트 종양 데이터
    baseline_tumors = [
        Tumor3DCoordinate(
            tumor_id="tumor_1",
            centroid_mm=(100.0, 50.0, 200.0),
            bbox_min_mm=(90.0, 45.0, 195.0),
            bbox_max_mm=(110.0, 55.0, 205.0),
            volume_cm3=10.5,
            longest_diameter_mm=28.3
        ),
        Tumor3DCoordinate(
            tumor_id="tumor_2",
            centroid_mm=(150.0, 80.0, 180.0),
            bbox_min_mm=(140.0, 75.0, 175.0),
            bbox_max_mm=(160.0, 85.0, 185.0),
            volume_cm3=5.2,
            longest_diameter_mm=18.7
        )
    ]
    
    followup_tumors = [
        Tumor3DCoordinate(
            tumor_id="tumor_1_fu",
            centroid_mm=(102.0, 51.0, 201.0),  # 약간 이동
            bbox_min_mm=(95.0, 47.0, 197.0),
            bbox_max_mm=(109.0, 55.0, 205.0),
            volume_cm3=7.3,  # 30% 감소
            longest_diameter_mm=22.5
        ),
        Tumor3DCoordinate(
            tumor_id="tumor_2_fu",
            centroid_mm=(151.0, 81.0, 179.0),
            bbox_min_mm=(145.0, 78.0, 176.0),
            bbox_max_mm=(157.0, 84.0, 182.0),
            volume_cm3=5.0,  # 약간 감소
            longest_diameter_mm=18.0
        )
    ]
    
    # 종양 매칭
    print("\n[1] Tumor Matching")
    print("-" * 80)
    matches = service.match_tumors_across_scans(baseline_tumors, followup_tumors)
    
    for match in matches:
        print(f"\nMatch: {match.baseline_tumor.tumor_id} → {match.followup_tumor.tumor_id}")
        print(f"  Distance moved: {match.distance_moved_mm:.1f} mm")
        print(f"  Volume change: {match.volume_change_percent:+.1f}%")
        print(f"  RECIST: {match.recist_response.value}")
    
    # 전체 평가
    print("\n[2] Overall Patient Response")
    print("-" * 80)
    evaluation = service.evaluate_patient_response(matches)
    
    print(f"Overall response: {evaluation['overall_response'].value}")
    print(f"Total volume: {evaluation['total_tumor_volume_baseline']:.1f} → {evaluation['total_tumor_volume_followup']:.1f} cm³")
    print(f"Volume change: {evaluation['volume_change_percent']:+.1f}%")
    print(f"CR: {evaluation['num_cr']}, PR: {evaluation['num_pr']}, SD: {evaluation['num_sd']}, PD: {evaluation['num_pd']}")
    
    # 비교 데이터
    print("\n[3] Comparison Visualization Data")
    print("-" * 80)
    comparison = service.generate_comparison_data(matches)
    
    print(f"Generated {len(comparison['arrows'])} arrows")
    print(f"Generated {len(comparison['changes'])} change summaries")
    
    print("\n" + "=" * 80)
