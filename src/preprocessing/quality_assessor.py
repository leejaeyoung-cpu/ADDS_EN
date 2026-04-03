"""
Image Quality Assessment for ADDS
이미지 품질을 자동으로 평가하여 분석 신뢰도를 보장
"""

import numpy as np
import cv2
from typing import Dict, Tuple
from pathlib import Path

from utils import get_logger

logger = get_logger(__name__)


class ImageQualityAssessor:
    """
    이미지 품질을 다각도로 평가하는 클래스
    """
    
    def __init__(self):
        """Initialize quality assessor"""
        logger.info("ImageQualityAssessor initialized")
        
        # 품질 기준 임계값
        self.thresholds = {
            'focus_min': 100.0,      # Laplacian variance 최소값
            'snr_min': 10.0,         # SNR 최소값
            'brightness_range': (30, 225),  # 적정 밝기 범위
            'contrast_min': 50.0     # 최소 대비
        }
    
    def assess_focus(self, image: np.ndarray) -> Tuple[float, str]:
        """
        초점 품질 평가 (Laplacian variance 사용)
        
        Args:
            image: 이미지 배열
            
        Returns:
            (점수, 평가) 튜플
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        
        # Laplacian 계산
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        focus_measure = laplacian.var()
        
        # 평가
        if focus_measure > 500:
            quality = "Excellent"
        elif focus_measure > 200:
            quality = "Good"
        elif focus_measure > self.thresholds['focus_min']:
            quality = "Acceptable"
        else:
            quality = "Poor"
        
        return float(focus_measure), quality
    
    def assess_noise(self, image: np.ndarray) -> Tuple[float, str]:
        """
        노이즈 수준 평가 (Signal-to-Noise Ratio)
        
        Args:
            image: 이미지 배열
            
        Returns:
            (SNR, 평가) 튜플
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(float)
        else:
            gray = image.astype(float)
        
        # 신호: 평균 강도
        signal = np.mean(gray)
        
        # 노이즈: 표준편차 (간단한 추정)
        noise = np.std(gray)
        
        # SNR 계산
        if noise > 0:
            snr = signal / noise
        else:
            snr = float('inf')
        
        # 평가
        if snr > 20:
            quality = "Excellent"
        elif snr > 15:
            quality = "Good"
        elif snr > self.thresholds['snr_min']:
            quality = "Acceptable"
        else:
            quality = "Poor"
        
        return float(snr), quality
    
    def assess_brightness(self, image: np.ndarray) -> Tuple[float, str]:
        """
        밝기 적정성 평가
        
        Args:
            image: 이미지 배열
            
        Returns:
            (평균 밝기, 평가) 튜플
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        
        mean_brightness = float(np.mean(gray))
        
        min_val, max_val = self.thresholds['brightness_range']
        
        if min_val <= mean_brightness <= max_val:
            quality = "Good"
        elif mean_brightness < min_val:
            quality = "Too Dark"
        else:
            quality = "Too Bright"
        
        return mean_brightness, quality
    
    def assess_contrast(self, image: np.ndarray) -> Tuple[float, str]:
        """
        대비 평가
        
        Args:
            image: 이미지 배열
            
        Returns:
            (대비값, 평가) 튜플
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        
        contrast = float(gray.max() - gray.min())
        
        if contrast > 150:
            quality = "Excellent"
        elif contrast > 100:
            quality = "Good"
        elif contrast > self.thresholds['contrast_min']:
            quality = "Acceptable"
        else:
            quality = "Poor"
        
        return contrast, quality
    
    def assess_illumination_uniformity(self, image: np.ndarray) -> Tuple[float, str]:
        """
        조명 균일성 평가
        
        Args:
            image: 이미지 배열
            
        Returns:
            (균일성 점수, 평가) 튜플
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        
        # 이미지를 그리드로 나누어 평균 밝기 계산
        h, w = gray.shape
        grid_size = 4
        grid_h, grid_w = h // grid_size, w // grid_size
        
        grid_means = []
        for i in range(grid_size):
            for j in range(grid_size):
                region = gray[i*grid_h:(i+1)*grid_h, j*grid_w:(j+1)*grid_w]
                grid_means.append(np.mean(region))
        
        # 균일성: 그리드 간 표준편차가 작을수록 좋음
        uniformity_std = np.std(grid_means)
        
        # 정규화된 점수 (0-1, 높을수록 균일)
        uniformity_score = 1.0 / (1.0 + uniformity_std / 50.0)
        
        if uniformity_score > 0.9:
            quality = "Excellent"
        elif uniformity_score > 0.8:
            quality = "Good"
        elif uniformity_score > 0.7:
            quality = "Acceptable"
        else:
            quality = "Uneven"
        
        return float(uniformity_score), quality
    
    def assess_overall_quality(self, image: np.ndarray) -> Dict:
        """
        종합 품질 평가
        
        Args:
            image: 이미지 배열
            
        Returns:
            종합 평가 결과 딕셔너리
        """
        # 각 항목 평가
        focus_score, focus_quality = self.assess_focus(image)
        snr, snr_quality = self.assess_noise(image)
        brightness, brightness_quality = self.assess_brightness(image)
        contrast, contrast_quality = self.assess_contrast(image)
        uniformity, uniformity_quality = self.assess_illumination_uniformity(image)
        
        # 종합 점수 계산 (0-1 정규화)
        focus_norm = min(focus_score / 500.0, 1.0)
        snr_norm = min(snr / 20.0, 1.0)
        brightness_norm = 1.0 if brightness_quality == "Good" else 0.5
        contrast_norm = min(contrast / 200.0, 1.0)
        
        overall_score = (focus_norm * 0.3 + 
                        snr_norm * 0.2 + 
                        brightness_norm * 0.2 + 
                        contrast_norm * 0.15 + 
                        uniformity * 0.15)
        
        # 종합 판정
        if overall_score > 0.85:
            overall_quality = "Excellent"
        elif overall_score > 0.70:
            overall_quality = "Good"
        elif overall_score > 0.55:
            overall_quality = "Acceptable"
        else:
            overall_quality = "Poor"
        
        # 권장사항
        recommendations = []
        if focus_quality == "Poor":
            recommendations.append("⚠️ 초점이 불량합니다. 재촬영을 권장합니다.")
        if snr_quality == "Poor":
            recommendations.append("⚠️ 노이즈가 높습니다. 노출 시간 증가를 고려하세요.")
        if brightness_quality == "Too Dark":
            recommendations.append("⚠️ 이미지가 너무 어둡습니다.")
        elif brightness_quality == "Too Bright":
            recommendations.append("⚠️ 이미지가 너무 밝습니다.")
        if uniformity_quality == "Uneven":
            recommendations.append("⚠️ 조명이 불균일합니다.")
        if not recommendations:
            recommendations.append("✓ 이미지 품질이 양호합니다.")
        
        return {
            'overall_score': round(overall_score, 3),
            'overall_quality': overall_quality,
            'detailed_assessment': {
                'focus': {
                    'score': round(focus_score, 2),
                    'quality': focus_quality
                },
                'noise': {
                    'snr': round(snr, 2),
                    'quality': snr_quality
                },
                'brightness': {
                    'value': round(brightness, 2),
                    'quality': brightness_quality
                },
                'contrast': {
                    'value': round(contrast, 2),
                    'quality': contrast_quality
                },
                'illumination_uniformity': {
                    'score': round(uniformity, 3),
                    'quality': uniformity_quality
                }
            },
            'recommendations': recommendations
        }
