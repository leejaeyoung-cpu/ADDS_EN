"""
Smart Region Selection for Accurate Tumor Detection
크기, 위치, 형태 기반으로 실제 종양만 선택
"""

import numpy as np
import cv2
from typing import List, Dict, Tuple

class TumorRegionSelector:
    """
    여러 검출된 region에서 실제 종양일 가능성이 높은 것만 선택
    """
    
    def __init__(
        self,
        min_size=1000,
        max_size=10000,
        rectum_roi=None,
        circularity_range=(0.3, 0.7),
        min_intensity_variance=50
    ):
        """
        Args:
            min_size: 최소 종양 크기 (pixels)
            max_size: 최대 종양 크기 (pixels)
            rectum_roi: Rectum 영역 {'x': [x_min, x_max], 'y': [y_min, y_max]}
            circularity_range: 원형도 범위 (min, max)
            min_intensity_variance: 최소 강도 분산
        """
        self.min_size = min_size
        self.max_size = max_size
        self.rectum_roi = rectum_roi
        self.circularity_range = circularity_range
        self.min_intensity_variance = min_intensity_variance
    
    def select_tumor_regions(
        self,
        binary_mask: np.ndarray,
        original_image: np.ndarray,
        verbose=True
    ) -> Tuple[np.ndarray, List[Dict]]:
        """
        Binary mask에서 실제 종양 region만 선택
        
        Args:
            binary_mask: Segmentation mask (0/1)
            original_image: 원본 이미지
            verbose: 로그 출력
        
        Returns:
            filtered_mask: 필터링된 mask
            region_info: 선택된 region 정보 리스트
        """
        # Contour 추출
        contours, _ = cv2.findContours(
            binary_mask.astype(np.uint8),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )
        
        if verbose:
            print(f"\n[Region Selection] Total contours: {len(contours)}")
        
        # 각 contour 평가
        candidates = []
        
        for idx, contour in enumerate(contours):
            region_info = self._evaluate_region(
                contour,
                binary_mask,
                original_image,
                idx
            )
            
            if region_info['is_tumor_candidate']:
                candidates.append(region_info)
                if verbose:
                    print(f"  ✓ Region {idx}: PASS (score={region_info['tumor_score']:.2f})")
            else:
                if verbose:
                    print(f"  ✗ Region {idx}: FAIL ({region_info['fail_reason']})")
        
        # 필터링된 mask 생성
        filtered_mask = np.zeros_like(binary_mask)
        for candidate in candidates:
            cv2.drawContours(
                filtered_mask,
                [candidate['contour']],
                -1,
                1,
                -1  # Filled
            )
        
        if verbose:
            print(f"\n[Result] {len(candidates)}/{len(contours)} regions selected")
        
        return filtered_mask, candidates
    
    def _evaluate_region(
        self,
        contour: np.ndarray,
        mask: np.ndarray,
        image: np.ndarray,
        idx: int
    ) -> Dict:
        """개별 region 평가"""
        
        # 기본 특징 추출
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        
        # Bounding box
        x, y, w, h = cv2.boundingRect(contour)
        centroid_x = x + w // 2
        centroid_y = y + h // 2
        
        # Circularity (원형도)
        if perimeter > 0:
            circularity = 4 * np.pi * area / (perimeter ** 2)
        else:
            circularity = 0
        
        # Intensity variance (region 내 강도 분산)
        region_mask = np.zeros_like(mask)
        cv2.drawContours(region_mask, [contour], -1, 1, -1)
        region_pixels = image[region_mask > 0]
        intensity_var = np.var(region_pixels) if len(region_pixels) > 0 else 0
        
        # Image shape
        h_img, w_img = image.shape[:2]
        
        # 각 기준 평가
        checks = {}
        fail_reason = []
        
        # 1. 크기 필터
        size_ok = self.min_size <= area <= self.max_size
        checks['size'] = size_ok
        if not size_ok:
            fail_reason.append(f"size={area:.0f}")
        
        # 2. 위치 필터 (Rectum ROI)
        if self.rectum_roi is None:
            # Default ROI: 중앙 하단
            roi = {
                'x': [w_img * 0.3, w_img * 0.7],
                'y': [h_img * 0.5, h_img * 0.9]
            }
        else:
            roi = self.rectum_roi
        
        location_ok = (
            roi['x'][0] <= centroid_x <= roi['x'][1] and
            roi['y'][0] <= centroid_y <= roi['y'][1]
        )
        checks['location'] = location_ok
        if not location_ok:
            fail_reason.append(f"location=({centroid_x},{centroid_y})")
        
        # 3. 형태 필터
        shape_ok = self.circularity_range[0] <= circularity <= self.circularity_range[1]
        checks['shape'] = shape_ok
        if not shape_ok:
            fail_reason.append(f"circ={circularity:.2f}")
        
        # 4. Intensity variance 필터
        variance_ok = intensity_var >= self.min_intensity_variance
        checks['variance'] = variance_ok
        if not variance_ok:
            fail_reason.append(f"var={intensity_var:.1f}")
        
        # 종합 판정
        is_tumor = all(checks.values())
        
        # Tumor score (0-1)
        scores = []
        
        # Size score (closer to mid-range = better)
        mid_size = (self.min_size + self.max_size) / 2
        size_score = 1 - abs(area - mid_size) / mid_size
        scores.append(max(0, size_score))
        
        # Location score (closer to center of ROI = better)
        roi_center_x = (roi['x'][0] + roi['x'][1]) / 2
        roi_center_y = (roi['y'][0] + roi['y'][1]) / 2
        dist_x = abs(centroid_x - roi_center_x) / w_img
        dist_y = abs(centroid_y - roi_center_y) / h_img
        location_score = 1 - (dist_x + dist_y) / 2
        scores.append(max(0, location_score))
        
        # Circularity score
        circ_mid = (self.circularity_range[0] + self.circularity_range[1]) / 2
        circ_score = 1 - abs(circularity - circ_mid) / circ_mid
        scores.append(max(0, circ_score))
        
        # Variance score (normalized)
        var_score = min(1.0, intensity_var / 200)
        scores.append(var_score)
        
        tumor_score = np.mean(scores) if is_tumor else 0
        
        return {
            'region_id': idx,
            'contour': contour,
            'area': area,
            'perimeter': perimeter,
            'centroid': (centroid_x, centroid_y),
            'circularity': circularity,
            'intensity_variance': intensity_var,
            'bounding_box': {'x': x, 'y': y, 'width': w, 'height': h},
            'checks': checks,
            'is_tumor_candidate': is_tumor,
            'tumor_score': tumor_score,
            'fail_reason': ' | '.join(fail_reason) if fail_reason else 'none'
        }
