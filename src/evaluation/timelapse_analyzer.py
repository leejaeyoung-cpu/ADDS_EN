"""
Time-lapse Analysis Module
===========================
시계열 세포 데이터 분석 및 약물 효과 정량화

주요 분석:
- 세포 증식 곡선 (Proliferation curve)
- 세포 이동 분석 (Migration analysis)
- 형태학적 동역학 (Morphology dynamics)
- 이벤트 분석 (Division/Death events)
- 약물 반응성 (Drug response)
- 칵테일 시너지 (Synergy analysis)
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from scipy.stats import linregress
from scipy.optimize import curve_fit
import logging

logger = logging.getLogger(__name__)


class TimeLapseAnalyzer:
    """Time-lapse 데이터 종합 분석기"""
    
    def __init__(self, frame_interval_minutes: float = 5.0):
        """
        Args:
            frame_interval_minutes: 프레임 간 시간 간격 (분 단위)
        """
        self.frame_interval_minutes = frame_interval_minutes
        self.frame_interval_hours = frame_interval_minutes / 60.0
    
    def calculate_proliferation_curve(self, 
                                     tracks: Dict,
                                     max_frame: Optional[int] = None) -> Dict:
        """
        세포 증식 곡선 계산
        
        Args:
            tracks: CellTracker에서 반환된 트랙 딕셔너리
            max_frame: 분석할 최대 프레임 (None이면 전체)
        
        Returns:
            {
                'frames': List[int],
                'cell_counts': List[int],
                'time_points_hours': List[float],
                'doubling_time_hours': float,
                'growth_rate': float,
                '생존 세포 수': List[int],
                'cumulative_births': List[int],
                'cumulative_deaths': List[int]
            }
        """
        logger.info("증식 곡선 계산 중...")
        
        if not tracks:
            return self._empty_proliferation_result()
        
        # 최대 프레임 결정
        all_frames = []
        for track in tracks.values():
            all_frames.extend(track.frames)
        
        if not all_frames:
            return self._empty_proliferation_result()
        
        max_f = max_frame if max_frame else max(all_frames)
        frames = list(range(max_f + 1))
        
        # 프레임별 세포 수 계산
        cell_counts = []
        alive_counts = []
        cumulative_births = []
        cumulative_deaths = []
        
        total_births = 0
        total_deaths = 0
        
        for frame in frames:
            # 해당 프레임에서 활성 세포 수
            count = 0
            alive = 0
            
            for track in tracks.values():
                if frame in track.frames:
                    count += 1
                    if track.death_frame is None or track.death_frame > frame:
                        alive += 1
                
                # 출생/사망 이벤트
                if track.birth_frame == frame:
                    total_births += 1
                if track.death_frame == frame:
                    total_deaths += 1
            
            cell_counts.append(count)
            alive_counts.append(alive)
            cumulative_births.append(total_births)
            cumulative_deaths.append(total_deaths)
        
        # 시간 변환
        time_points_hours = [f * self.frame_interval_hours for f in frames]
        
        # Doubling time 계산 (지수 성장 구간에서)
        doubling_time = self._calculate_doubling_time(time_points_hours, cell_counts)
        
        # Growth rate 계산
        growth_rate = self._calculate_growth_rate(time_points_hours, cell_counts)
        
        result = {
            'frames': frames,
            'cell_counts': cell_counts,
            'alive_counts': alive_counts,
            'time_points_hours': time_points_hours,
            'doubling_time_hours': doubling_time,
            'growth_rate_per_hour': growth_rate,
            'cumulative_births': cumulative_births,
            'cumulative_deaths': cumulative_deaths,
            'final_count': cell_counts[-1] if cell_counts else 0,
            'max_count': max(cell_counts) if cell_counts else 0
        }
        
        logger.info(f"증식 분석 완료: Doubling time = {doubling_time:.2f} hours")
        
        return result
    
    def _empty_proliferation_result(self) -> Dict:
        """빈 증식 결과"""
        return {
            'frames': [],
            'cell_counts': [],
            'alive_counts': [],
            'time_points_hours': [],
            'doubling_time_hours': None,
            'growth_rate_per_hour': 0.0,
            'cumulative_births': [],
            'cumulative_deaths': [],
            'final_count': 0,
            'max_count': 0
        }
    
    def _calculate_doubling_time(self, 
                                 time_points: List[float], 
                                 cell_counts: List[int]) -> Optional[float]:
        """
        Doubling time 계산 (지수 성장 모델)
        N(t) = N0 * e^(k*t)
        Doubling time = ln(2) / k
        """
        if len(time_points) < 10:
            return None
        
        # 로그 변환으로 선형화
        counts_array = np.array(cell_counts)
        time_array = np.array(time_points)
        
        # 0 제거
        valid_idx = counts_array > 0
        if valid_idx.sum() < 10:
            return None
        
        log_counts = np.log(counts_array[valid_idx])
        time_valid = time_array[valid_idx]
        
        # 선형 회귀
        try:
            slope, intercept, r_value, p_value, std_err = linregress(time_valid, log_counts)
            
            if slope > 0:
                doubling_time = np.log(2) / slope
                return doubling_time
            else:
                return None
        except:
            return None
    
    def _calculate_growth_rate(self, 
                               time_points: List[float],
                               cell_counts: List[int]) -> float:
        """평균 성장 속도 (세포/시간)"""
        if len(time_points) < 2:
            return 0.0
        
        time_array = np.array(time_points)
        counts_array = np.array(cell_counts)
        
        # 선형 회귀
        try:
            slope, _, _, _, _ = linregress(time_array, counts_array)
            return max(0.0, slope)
        except:
            return 0.0
    
    def analyze_migration(self, tracks: Dict) -> Dict:
        """
        세포 이동 분석
        
        Returns:
            {
                'total_displacements': List[float],
                'velocities': List[float],
                'directionalities': List[float],
                'mean_velocity': float,
                'mean_displacement': float,
                'migration_index': float
            }
        """
        logger.info("세포 이동 분석 중...")
        
        displacements = []
        velocities = []
        directionalities = []
        
        for track in tracks.values():
            if len(track.frames) < 2:
                continue
            
            # Total displacement
            displacement = track.total_displacement
            displacements.append(displacement)
            
            # Velocity (픽셀/프레임 → 픽셀/시간)
            velocity = track.average_velocity / self.frame_interval_hours
            velocities.append(velocity)
            
            # Directionality (직선 거리 / 실제 경로)
            if len(track.centroids) >= 2:
                start = np.array(track.centroids[0])
                end = np.array(track.centroids[-1])
                straight_distance = np.linalg.norm(end - start)
                
                if displacement > 0:
                    directionality = straight_distance / displacement
                else:
                    directionality = 0.0
                
                directionalities.append(directionality)
        
        result = {
            'total_displacements': displacements,
            'velocities': velocities,
            'directionalities': directionalities,
            'mean_velocity_pixels_per_hour': np.mean(velocities) if velocities else 0.0,
            'std_velocity': np.std(velocities) if velocities else 0.0,
            'mean_displacement_pixels': np.mean(displacements) if displacements else 0.0,
            'mean_directionality': np.mean(directionalities) if directionalities else 0.0,
            'migration_index': np.mean(displacements) if displacements else 0.0
        }
        
        logger.info(f"이동 분석 완료: 평균 속도 = {result['mean_velocity_pixels_per_hour']:.2f} px/h")
        
        return result
    
    def analyze_morphology_dynamics(self, tracks: Dict) -> Dict:
        """
        형태학적 변화 분석
        
        Returns:
            {
                'mean_area_over_time': List[float],
                'std_area_over_time': List[float],
                'area_change_rates': List[float]
            }
        """
        logger.info("형태 동역학 분석 중...")
        
        # 프레임별 면적 수집
        all_frames = []
        for track in tracks.values():
            all_frames.extend(track.frames)
        
        if not all_frames:
            return {
                'frames': [],
                'mean_area_over_time': [],
                'std_area_over_time': [],
                'median_area_over_time': []
            }
        
        max_frame = max(all_frames)
        frames = list(range(max_frame + 1))
        
        mean_areas = []
        std_areas = []
        median_areas = []
        
        for frame in frames:
            frame_areas = []
            
            for track in tracks.values():
                if frame in track.frames:
                    idx = track.frames.index(frame)
                    frame_areas.append(track.areas[idx])
            
            if frame_areas:
                mean_areas.append(np.mean(frame_areas))
                std_areas.append(np.std(frame_areas))
                median_areas.append(np.median(frame_areas))
            else:
                mean_areas.append(0.0)
                std_areas.append(0.0)
                median_areas.append(0.0)
        
        result = {
            'frames': frames,
            'mean_area_over_time': mean_areas,
            'std_area_over_time': std_areas,
            'median_area_over_time': median_areas
        }
        
        return result
    
    def detect_events(self, tracks: Dict) -> Dict:
        """
        이벤트 감지 및 정량화
        
        Returns:
            {
                'division_times': List[int],
                'death_times': List[int],
                'division_rate': float,
                'death_rate': float
            }
        """
        logger.info("이벤트 분석 중...")
        
        division_times = []
        death_times = []
        birth_times = []
        
        for track in tracks.values():
            if track.division_frame is not None:
                division_times.append(track.division_frame)
            if track.death_frame is not None:
                death_times.append(track.death_frame)
            if track.birth_frame is not None:
                birth_times.append(track.birth_frame)
        
        # 전체 시간
        all_frames = []
        for track in tracks.values():
            all_frames.extend(track.frames)
        
        total_time_hours = max(all_frames) * self.frame_interval_hours if all_frames else 1.0
        
        # 이벤트 비율 (events per hour)
        division_rate = len(division_times) / total_time_hours if total_time_hours > 0 else 0.0
        death_rate = len(death_times) / total_time_hours if total_time_hours > 0 else 0.0
        
        result = {
            'division_times_frames': division_times,
            'death_times_frames': death_times,
            'birth_times_frames': birth_times,
            'total_divisions': len(division_times),
            'total_deaths': len(death_times),
            'total_births': len(birth_times),
            'division_rate_per_hour': division_rate,
            'death_rate_per_hour': death_rate
        }
        
        logger.info(f"이벤트 분석 완료: 분열 {len(division_times)}회, 사멸 {len(death_times)}회")
        
        return result
    
    def calculate_drug_response(self,
                               control_proliferation: Dict,
                               treatment_proliferation: Dict) -> Dict:
        """
        약물 반응성 정량화
        
        Args:
            control_proliferation: 대조군 증식 데이터
            treatment_proliferation: 처리군 증식 데이터
        
        Returns:
            {
                'growth_inhibition_percent': float,
                'relative_cell_count': float,
                'time_to_half_effect_hours': Optional[float]
            }
        """
        logger.info("약물 반응 분석 중...")
        
        control_final = control_proliferation.get('final_count', 0)
        treatment_final = treatment_proliferation.get('final_count', 0)
        
        if control_final == 0:
            return {
                'growth_inhibition_percent': 0.0,
                'relative_cell_count': 0.0,
                'time_to_half_effect_hours': None,
                'effect_detected': False
            }
        
        # Growth inhibition %
        inhibition = ((control_final - treatment_final) / control_final) * 100
        inhibition = max(0.0, min(100.0, inhibition))  # 0-100% 범위
        
        # Relative cell count
        relative_count = treatment_final / control_final
        
        # Time to half-maximal effect
        time_to_half = self._calculate_time_to_half_effect(
            control_proliferation, 
            treatment_proliferation
        )
        
        result = {
            'growth_inhibition_percent': inhibition,
            'relative_cell_count': relative_count,
            'time_to_half_effect_hours': time_to_half,
            'effect_detected': inhibition > 10.0,  # >10% inhibition
            'control_final_count': control_final,
            'treatment_final_count': treatment_final
        }
        
        logger.info(f"약물 효과: {inhibition:.1f}% growth inhibition")
        
        return result
    
    def _calculate_time_to_half_effect(self,
                                      control_data: Dict,
                                      treatment_data: Dict) -> Optional[float]:
        """반효과 시간 계산"""
        control_counts = control_data.get('cell_counts', [])
        treatment_counts = treatment_data.get('cell_counts', [])
        time_points = control_data.get('time_points_hours', [])
        
        if len(control_counts) < 2 or len(treatment_counts) < 2:
            return None
        
        # 최대 효과의 50%에 도달하는 시점
        max_diff = max([abs(c - t) for c, t in zip(control_counts, treatment_counts)])
        half_effect = max_diff / 2.0
        
        for i, (c, t) in enumerate(zip(control_counts, treatment_counts)):
            if abs(c - t) >= half_effect:
                return time_points[i]
        
        return None
    
    def calculate_synergy_timelapse(self,
                                   drug_a_proliferation: Dict,
                                   drug_b_proliferation: Dict,
                                   combo_proliferation: Dict,
                                   control_proliferation: Dict,
                                   model: str = 'bliss') -> Dict:
        """
        시간에 따른 칵테일 시너지 분석
        
        Args:
            drug_a_proliferation: Drug A 단독 데이터
            drug_b_proliferation: Drug B 단독 데이터
            combo_proliferation: 조합 데이터
            control_proliferation: 대조군 데이터
            model: 'bliss' or 'loewe'
        
        Returns:
            {
                'synergy_scores_over_time': List[float],
                'mean_synergy_score': float,
                'peak_synergy_time_hours': float,
                'is_synergistic': bool
            }
        """
        logger.info(f"칵테일 시너지 분석 중 ({model} 모델)...")
        
        control_counts = np.array(control_proliferation.get('cell_counts', []))
        drug_a_counts = np.array(drug_a_proliferation.get('cell_counts', []))
        drug_b_counts = np.array(drug_b_proliferation.get('cell_counts', []))
        combo_counts = np.array(combo_proliferation.get('cell_counts', []))
        time_points = control_proliferation.get('time_points_hours', [])
        
        if len(control_counts) == 0:
            return self._empty_synergy_result()
        
        # 길이 맞추기
        min_len = min(len(control_counts), len(drug_a_counts), 
                     len(drug_b_counts), len(combo_counts))
        
        control_counts = control_counts[:min_len]
        drug_a_counts = drug_a_counts[:min_len]
        drug_b_counts = drug_b_counts[:min_len]
        combo_counts = combo_counts[:min_len]
        time_points = time_points[:min_len]
        
        # Normalize to control
        control_counts = np.maximum(control_counts, 1)  # 0 방지
        
        fa = 1.0 - (drug_a_counts / control_counts)  # Fractional inhibition A
        fb = 1.0 - (drug_b_counts / control_counts)  # Fractional inhibition B
        f_combo = 1.0 - (combo_counts / control_counts)  # Combo inhibition
        
        # Bliss synergy score
        synergy_scores = []
        
        for i in range(len(fa)):
            if model == 'bliss':
                # Bliss independence: E_expected = Ea + Eb - Ea*Eb
                expected = fa[i] + fb[i] - (fa[i] * fb[i])
                synergy = f_combo[i] - expected
            else:  # loewe (simplified)
                expected = max(fa[i], fb[i])
                synergy = f_combo[i] - expected
            
            synergy_scores.append(synergy)
        
        synergy_scores = np.array(synergy_scores)
        
        # Peak synergy
        if len(synergy_scores) > 0:
            peak_idx = np.argmax(np.abs(synergy_scores))
            peak_time = time_points[peak_idx]
            mean_synergy = np.mean(synergy_scores)
        else:
            peak_time = 0.0
            mean_synergy = 0.0
        
        result = {
            'time_points_hours': time_points,
            'synergy_scores_over_time': synergy_scores.tolist(),
            'mean_synergy_score': float(mean_synergy),
            'peak_synergy_score': float(synergy_scores[peak_idx]) if len(synergy_scores) > 0 else 0.0,
            'peak_synergy_time_hours': float(peak_time),
            'is_synergistic': mean_synergy > 0.1,  # >0.1 시너지로 판단
            'is_antagonistic': mean_synergy < -0.1
        }
        
        logger.info(f"시너지 분석 완료: 평균 = {mean_synergy:.3f}, Peak at {peak_time:.1f}h")
        
        return result
    
    def _empty_synergy_result(self) -> Dict:
        """빈 시너지 결과"""
        return {
            'time_points_hours': [],
            'synergy_scores_over_time': [],
            'mean_synergy_score': 0.0,
            'peak_synergy_score': 0.0,
            'peak_synergy_time_hours': 0.0,
            'is_synergistic': False,
            'is_antagonistic': False
        }
    
    def generate_comprehensive_report(self,
                                     tracks: Dict,
                                     experiment_name: str = "Experiment") -> Dict:
        """
        종합 분석 리포트 생성
        
        Returns:
            모든 분석 결과를 포함한 딕셔너리
        """
        logger.info(f"종합 리포트 생성 중: {experiment_name}")
        
        report = {
            'experiment_name': experiment_name,
            'total_tracks': len(tracks),
            'proliferation': self.calculate_proliferation_curve(tracks),
            'migration': self.analyze_migration(tracks),
            'morphology': self.analyze_morphology_dynamics(tracks),
            'events': self.detect_events(tracks)
        }
        
        logger.info("종합 리포트 생성 완료")
        
        return report
