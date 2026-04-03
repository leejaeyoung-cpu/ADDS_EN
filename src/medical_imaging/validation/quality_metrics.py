# -*- coding: utf-8 -*-
"""
종양 검출 품질 자동 평가 시스템
"""
import numpy as np
from typing import Dict, List, Tuple, Any
from pathlib import Path
import json


class DetectionQualityMetrics:
    """종양 검출 품질 자동 평가"""
    
    def __init__(self):
        # 임상 기준값 (의료 문헌 기반)
        self.normal_ranges = {
            'volume_ml': (0.1, 100),       # 정상 종양 부피 범위
            'num_regions': (1, 50),         # 정상 영역 개수
            'affected_ratio': (0.05, 0.4),  # 영향받은 슬라이스 비율
            'avg_size_mm2': (10, 5000),     # 영역당 평균 크기
            'regions_per_slice': (1, 5)     # 슬라이스당 평균 영역 수
        }
    
    def _in_range(self, value: float, metric_name: str) -> bool:
        """값이 정상 범위 내인지 확인"""
        min_val, max_val = self.normal_ranges[metric_name]
        return min_val <= value <= max_val
    
    def _is_too_scattered(self, centers: List[Tuple[int, int]]) -> bool:
        """검출 영역이 과도하게 분산되었는지 확인"""
        if len(centers) < 5:
            return False
        
        centers_array = np.array(centers)
        
        # 중심점들의 표준편차
        std_x = np.std(centers_array[:, 0])
        std_y = np.std(centers_array[:, 1])
        
        # 표준편차가 너무 크면 과도하게 분산
        return std_x > 150 or std_y > 150
    
    def _compute_clustering_score(self, centers: List[Tuple[int, int]]) -> float:
        """공간적 클러스터링 점수 계산 (0-1, 높을수록 밀집)"""
        if len(centers) < 2:
            return 1.0
        
        centers_array = np.array(centers)
        
        # 중심점 간 평균 거리
        distances = []
        for i in range(len(centers_array)):
            for j in range(i + 1, len(centers_array)):
                dist = np.linalg.norm(centers_array[i] - centers_array[j])
                distances.append(dist)
        
        avg_distance = np.mean(distances)
        
        # 정규화 (0-500 픽셀 범위를 1-0으로 변환)
        clustering_score = max(0, 1 - avg_distance / 500)
        
        return clustering_score
    
    def assess_results(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """
        검출 결과 자동 평가
        
        Args:
            results: analyze_tumor_location.py의 출력 결과
            
        Returns:
            품질 평가 리포트
        """
        score = 100  # 만점에서 시작
        warnings = []
        recommendations = []
        
        summary = results.get('summary', {})
        detections = results.get('tumor_detections', [])
        total_slices = results.get('total_slices', 1)
        
        # ===== 1. 부피 체크 =====
        volume = summary.get('total_volume_ml', 0)
        if volume == 0:
            warnings.append("검출된 종양 없음")
            score = 0
        elif not self._in_range(volume, 'volume_ml'):
            if volume > 100:
                warnings.append(f"⚠️ 비정상적으로 큰 부피: {volume:.1f} mL (정상: 0.1-100 mL)")
                score -= 30
                recommendations.append("임계값을 높이거나 형상 필터를 강화하세요")
            else:
                warnings.append(f"⚠️ 매우 작은 부피: {volume:.1f} mL")
                score -= 10
        
        # ===== 2. 검출 개수 체크 =====
        num_regions = summary.get('total_tumor_regions', 0)
        if num_regions > 100:
            warnings.append(f"⚠️ 과도한 검출: {num_regions}개 (정상: 1-50개)")
            score -= 25
            recommendations.append("크기 필터링을 강화하고 블러 적용을 고려하세요")
        elif num_regions > 50:
            warnings.append(f"⚠️ 많은 검출: {num_regions}개")
            score -= 15
        
        # ===== 3. 슬라이스당 영역 수 =====
        affected_slices = summary.get('affected_slices', 1)
        regions_per_slice = num_regions / max(affected_slices, 1)
        
        if regions_per_slice > 5:
            warnings.append(f"⚠️ 슬라이스당 평균 {regions_per_slice:.1f}개 영역 (과다 검출)")
            score -= 15
            recommendations.append("모폴로지 연산을 강화하여 작은 노이즈를 제거하세요")
        
        # ===== 4. 영역 크기 분포 =====
        if detections:
            areas = [d['area_pixels'] for d in detections]
            avg_area = np.mean(areas)
            std_area = np.std(areas)
            coef_var = std_area / avg_area if avg_area > 0 else 0
            
            if coef_var > 3.0:
                warnings.append("⚠️ 영역 크기 분포 불균등 (변동계수 > 3.0)")
                score -= 10
                recommendations.append("크기 범위를 좁게 설정하세요")
            
            # 너무 작은 영역이 많은지 확인
            small_regions = sum(1 for a in areas if a < 100)
            if small_regions / len(areas) > 0.5:
                warnings.append(f"⚠️ 작은 영역이 많음: {small_regions}/{len(areas)}")
                score -= 10
                recommendations.append("min_area를 100 이상으로 설정하세요")
        
        # ===== 5. 공간 분포 (clustering) =====
        if detections:
            centers = [(d['center'][0], d['center'][1]) for d in detections]
            
            if self._is_too_scattered(centers):
                warnings.append("⚠️ 검출 영역이 과도하게 분산됨")
                score -= 10
                recommendations.append("종양은 보통 국소적으로 발생합니다. 전역 검출이 의심스럽습니다")
            
            clustering_score = self._compute_clustering_score(centers)
            if clustering_score < 0.3:
                warnings.append("⚠️ 공간적 클러스터링이 낮음 (산발적 검출)")
                score -= 10
        
        # ===== 6. 형상 분석 (circularity) =====
        if detections:
            circularities = []
            for d in detections:
                area = d.get('area_pixels', 0)
                perimeter = d.get('perimeter', 0)
                if perimeter > 0:
                    circ = 4 * np.pi * area / (perimeter ** 2)
                    circularities.append(circ)
            
            if circularities:
                avg_circ = np.mean(circularities)
                if avg_circ < 0.3:
                    warnings.append(f"⚠️ 비정형 영역이 많음 (평균 circularity: {avg_circ:.2f})")
                    score -= 10
                    recommendations.append("형상 필터(circularity > 0.3)를 추가하세요")
        
        # ===== 7. 영향받은 슬라이스 비율 =====
        affected_ratio = affected_slices / total_slices if total_slices > 0 else 0
        if affected_ratio > 0.5:
            warnings.append(f"⚠️ 슬라이스의 {affected_ratio*100:.1f}%에서 검출 (과다)")
            score -= 15
            recommendations.append("대부분의 슬라이스에서 검출은 비정상적입니다")
        
        # 총점 보정
        final_score = max(0, min(100, score))
        
        # 품질 등급
        quality_level = self._get_quality_level(final_score)
        
        # 상세 메트릭
        metrics = {
            'volume_ml': volume,
            'num_regions': num_regions,
            'affected_slices': affected_slices,
            'regions_per_slice': regions_per_slice,
            'affected_ratio': affected_ratio
        }
        
        if detections:
            areas = [d['area_pixels'] for d in detections]
            circularities_computed = []
            for d in detections:
                area = d.get('area_pixels', 0)
                perimeter = d.get('perimeter', 0)
                if perimeter > 0:
                    circularities_computed.append(4 * np.pi * area / (perimeter ** 2))
            
            metrics.update({
                'avg_area_pixels': np.mean(areas),
                'std_area_pixels': np.std(areas),
                'avg_circularity': np.mean(circularities_computed) if circularities_computed else 0,
                'spatial_clustering': self._compute_clustering_score(
                    [(d['center'][0], d['center'][1]) for d in detections]
                )
            })
        
        return {
            'score': round(final_score, 1),
            'quality': quality_level,
            'warnings': warnings,
            'recommendations': recommendations,
            'metrics': metrics,
            'status': 'pass' if final_score >= 60 else 'fail'
        }
    
    def _get_quality_level(self, score: float) -> str:
        """점수에 따른 품질 등급"""
        if score >= 80:
            return "🟢 고품질"
        elif score >= 60:
            return "🟡 중간"
        elif score >= 40:
            return "🟠 낮음"
        else:
            return "🔴 매우 낮음"
    
    def save_report(self, report: Dict[str, Any], output_path: str):
        """품질 리포트 저장"""
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        print(f"✅ 품질 리포트 저장: {output_path}")
    
    def print_report(self, report: Dict[str, Any]):
        """품질 리포트 출력"""
        print("\n" + "=" * 70)
        print("🔍 종양 검출 품질 평가 리포트")
        print("=" * 70)
        
        print(f"\n📊 종합 점수: {report['score']}/100")
        print(f"   품질 등급: {report['quality']}")
        print(f"   평가 결과: {'✅ PASS' if report['status'] == 'pass' else '❌ FAIL'}")
        
        if report['warnings']:
            print(f"\n⚠️  경고 사항 ({len(report['warnings'])}개):")
            for warning in report['warnings']:
                print(f"   - {warning}")
        
        if report['recommendations']:
            print(f"\n💡 개선 권장사항:")
            for rec in report['recommendations']:
                print(f"   - {rec}")
        
        print(f"\n📈 상세 메트릭:")
        for key, value in report['metrics'].items():
            if isinstance(value, float):
                print(f"   {key}: {value:.2f}")
            else:
                print(f"   {key}: {value}")
        
        print("=" * 70 + "\n")


def evaluate_detection_results(result_json_path: str) -> Dict[str, Any]:
    """
    종양 검출 결과 JSON 파일을 평가
    
    Args:
        result_json_path: 검출 결과 JSON 파일 경로
        
    Returns:
        품질 평가 리포트
    """
    with open(result_json_path, 'r', encoding='utf-8') as f:
        results = json.load(f)
    
    evaluator = DetectionQualityMetrics()
    report = evaluator.assess_results(results)
    evaluator.print_report(report)
    
    return report


if __name__ == "__main__":
    # 테스트
    import sys
    import io
    
    # Windows 인코딩 설정
    if sys.platform == 'win32':
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    
    if len(sys.argv) > 1:
        result_path = sys.argv[1]
    else:
        result_path = "tumor_analysis_results/tumor_analysis_report.json"
    
    if Path(result_path).exists():
        print(f"[분석] 파일: {result_path}\n")
        report = evaluate_detection_results(result_path)
        
        # 리포트 저장
        output_path = str(Path(result_path).parent / "quality_report.json")
        evaluator = DetectionQualityMetrics()
        evaluator.save_report(report, output_path)
    else:
        print(f"[오류] 파일을 찾을 수 없습니다: {result_path}")
