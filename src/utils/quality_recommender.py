"""
Quality Recommender
Automatic quality improvement recommendations based on analysis
"""

from typing import Dict, List, Tuple
import numpy as np


class QualityRecommender:
    """Generate automatic quality improvement recommendations"""
    
    def __init__(self):
        self.recommendations = {}
    
    def analyze_quality_issues(
        self,
        quality_metrics: Dict,
        cell_stats: Dict = None
    ) -> Dict:
        """
        Analyze quality metrics and identify issues
        
        Args:
            quality_metrics: Quality metrics from ImageQualityEnhancer
            cell_stats: Optional cell statistics
        
        Returns:
            Dictionary of identified issues with priorities
        """
        issues = {}
        
        # Focus issues
        focus_score = quality_metrics.get('focus_score', 0)
        focus_grade = self._get_focus_grade(focus_score)
        
        if focus_grade in ['Acceptable', 'Poor']:
            issues['focus'] = {
                'metric': 'Focus Quality',
                'current_value': focus_score,
                'current_grade': focus_grade,
                'target_grade': 'Good',
                'priority': 'High' if focus_grade == 'Poor' else 'Medium',
                'impact': 'Affects cell boundary detection accuracy'
            }
        
        # SNR issues
        snr = quality_metrics.get('snr', 0)
        snr_grade = self._get_snr_grade(snr)
        
        if snr_grade in ['Acceptable', 'Poor']:
            issues['noise'] = {
                'metric': 'Signal-to-Noise Ratio',
                'current_value': snr,
                'current_grade': snr_grade,
                'target_value': 15.0,
                'priority': 'High' if snr < 8 else 'Medium',
                'impact': 'Reduces segmentation reliability'
            }
        
        # Brightness issues
        brightness = quality_metrics.get('brightness', 128)
        if brightness < 50:
            issues['brightness'] = {
                'metric': 'Brightness',
                'current_value': brightness,
                'current_grade': 'Too Dark',
                'priority': 'Medium',
                'impact': 'May miss darker cells'
            }
        elif brightness > 200:
            issues['brightness'] = {
                'metric': 'Brightness',
                'current_value': brightness,
                'current_grade': 'Too Bright',
                'priority': 'Medium',
                'impact': 'May cause saturation, loss of detail'
            }
        
        # Contrast issues
        contrast = quality_metrics.get('contrast', 0)
        if contrast < 20:
            issues['contrast'] = {
                'metric': 'Contrast',
                'current_value': contrast,
                'current_grade': 'Low',
                'priority': 'Low',
                'impact': 'Reduces feature visibility'
            }
        
        # Sharpness issues
        sharpness = quality_metrics.get('sharpness', 0)
        if sharpness < 15:
            issues['sharpness'] = {
                'metric': 'Sharpness',
                'current_value': sharpness,
                'current_grade': 'Blurry',
                'priority': 'Medium',
                'impact': 'Affects edge detection'
            }
        
        # Cell statistics issues (if provided)
        if cell_stats:
            cell_count = cell_stats.get('num_cells', 0)
            if cell_count < 10:
                issues['cell_count'] = {
                    'metric': 'Cell Count',
                    'current_value': cell_count,
                    'current_grade': 'Very Low',
                    'priority': 'High',
                    'impact': 'Insufficient data for reliable statistics'
                }
        
        return issues
    
    def generate_recommendations(
        self,
        issues: Dict
    ) -> Dict:
        """
        Generate actionable recommendations for each issue
        
        Args:
            issues: Dictionary of identified issues
        
        Returns:
            Recommendations dictionary
        """
        recommendations = {}
        
        for issue_key, issue_data in issues.items():
            recommendations[issue_key] = self._generate_specific_recommendations(
                issue_key,
                issue_data
            )
        
        self.recommendations = recommendations
        return recommendations
    
    def _generate_specific_recommendations(
        self,
        issue_type: str,
        issue_data: Dict
    ) -> Dict:
        """Generate recommendations for specific issue type"""
        
        if issue_type == 'focus':
            return {
                'issue': f"Focus quality is {issue_data['current_grade']}",
                'recommendations': [
                    '🔧 이미지 획득 개선:',
                    '  - 현미경 초점을 수동으로 조정',
                    '  - 가능한 경우 자동 초점 기능 사용',
                    '  - 샘플과 대물 렌즈 거리 확인',
                    '',
                    '💻 소프트웨어 처리:',
                    '  - "포커스 향상" 전처리 활성화',
                    '  - Unsharp masking 필터 적용 (강도 1.0-1.5)',
                    '  - Laplacian sharpening 고려'
                ],
                'priority': issue_data['priority'],
                'expected_improvement': '+30-50% focus score',
                'processing_cost': '+0.3초 처리 시간'
            }
        
        elif issue_type == 'noise':
            current_snr = issue_data['current_value']
            target_snr = issue_data.get('target_value', 15.0)
            improvement = ((target_snr - current_snr) / current_snr * 100)
            
            return {
                'issue': f"SNR {current_snr:.2f} (목표: >{target_snr})",
                'recommendations': [
                    '🔧 이미지 획득 개선:',
                    '  - 카메라 노출 시간 증가 (더 많은 광자 수집)',
                    '  - ISO/Gain 설정 감소',
                    '  - 조명 강도 증가',
                    '  - 냉각 CCD 카메라 사용 (가능한 경우)',
                    '',
                    '💻 소프트웨어 처리:',
                    '  - "노이즈 제거" 전처리 활성화',
                    '  - Bilateral 필터 사용 (에지 보존)',
                    '  - 여러 이미지 평균화 (가능한 경우)'
                ],
                'priority': issue_data['priority'],
                'expected_improvement': f'SNR {current_snr:.1f} → {target_snr}+ (+{improvement:.0f}%)',
                'processing_cost': '+0.5-2초 처리 시간 (방법에 따라)'
            }
        
        elif issue_type == 'brightness':
            current = issue_data['current_value']
            is_dark = current < 50
            
            return {
                'issue': f"Brightness {current:.0f} ({'Too Dark' if is_dark else 'Too Bright'})",
                'recommendations': [
                    '🔧 이미지 획득 개선:',
                    f"  - 조명 강도 {'증가' if is_dark else '감소'}",
                    f"  - 노출 시간 {'증가' if is_dark else '감소'}",
                    '  - 자동 노출 설정 확인',
                    '',
                    '💻 소프트웨어 처리:',
                    '  - Histogram equalization 적용',
                    '  - Gamma correction 사용',
                    f"  - {'어두운' if is_dark else '밝은'} 영역 클리핑 주의"
                ],
                'priority': issue_data['priority'],
                'expected_improvement': '최적 밝기 범위 (80-180)',
                'processing_cost': '무시할 수 있음'
            }
        
        elif issue_type == 'contrast':
            return {
                'issue': f"Contrast {issue_data['current_value']:.1f} (낮음)",
                'recommendations': [
                    '💻 소프트웨어 처리:',
                    '  - "대비 최적화" 전처리 활성화',
                    '  - CLAHE (Adaptive Histogram Eq.) 사용',
                    '  - 로컬 대비 향상 적용',
                    '',
                    '🔧 이미지 획득 개선:',
                    '  - 명암비가 높은 염색 사용',
                    '  - 배경 조명 조정'
                ],
                'priority': issue_data['priority'],
                'expected_improvement': '+40-60% 대비 개선',
                'processing_cost': '+0.2초'
            }
        
        elif issue_type == 'sharpness':
            return {
                'issue': f"Sharpness {issue_data['current_value']:.1f} (흐림)",
                'recommendations': [
                    '🔧 이미지 획득 개선:',
                    '  - 초점 재조정',
                    '  - 렌즈 청소 확인',
                    '  - 진동 최소화',
                    '',
                    '💻 소프트웨어 처리:',
                    '  - "포커스 향상" 옵션 사용',
                    '  - Unsharp masking 적용',
                    '  - 고주파 필터 사용'
                ],
                'priority': issue_data['priority'],
                'expected_improvement': '+50% 선명도',
                'processing_cost': '+0.3초'
            }
        
        elif issue_type == 'cell_count':
            return {
                'issue': f"Cell count {issue_data['current_value']} (매우 적음)",
                'recommendations': [
                    '🔬 실험 조건:',
                    '  - 세포 밀도 확인 (너무 낮을 수 있음)',
                    '  - 더 넓은 시야 촬영',
                    '  - 여러 시야의 이미지 병합',
                    '',
                    '⚙️ 분석 파라미터:',
                    '  - Cellpose diameter 조정',
                    '  - Cell probability threshold 낮춤',
                    '  - Flow threshold 조정'
                ],
                'priority': issue_data['priority'],
                'expected_improvement': '더 많은 세포 탐지',
                'processing_cost': '파라미터 조정만 필요'
            }
        
        return {
            'issue': 'Unknown issue type',
            'recommendations': ['Manual inspection required'],
            'priority': 'Low',
            'expected_improvement': 'Unknown',
            'processing_cost': 'Unknown'
        }
    
    def _get_focus_grade(self, focus_score: float) -> str:
        """Get focus quality grade"""
        if focus_score >= 1000:
            return 'Excellent'
        elif focus_score >= 500:
            return 'Good'
        elif focus_score >= 100:
            return 'Acceptable'
        else:
            return 'Poor'
    
    def _get_snr_grade(self, snr: float) -> str:
        """Get SNR quality grade"""
        if snr >= 20:
            return 'Excellent'
        elif snr >= 15:
            return 'Good'
        elif snr >= 10:
            return 'Acceptable'
        else:
            return 'Poor'
    
    def get_priority_summary(self) -> Dict:
        """
        Get summary of recommendations by priority
        
        Returns:
            Dictionary with counts by priority
        """
        if not self.recommendations:
            return {'High': 0, 'Medium': 0, 'Low': 0}
        
        counts = {'High': 0, 'Medium': 0, 'Low': 0}
        
        for rec in self.recommendations.values():
            priority = rec.get('priority', 'Low')
            counts[priority] += 1
        
        return counts
    
    def generate_action_plan(self) -> List[Dict]:
        """
        Generate prioritized action plan
        
        Returns:
            List of actions sorted by priority
        """
        actions = []
        
        priority_order = {'High': 0, 'Medium': 1, 'Low': 2}
        
        for issue_type, rec in self.recommendations.items():
            actions.append({
                'issue_type': issue_type,
                'priority': rec['priority'],
                'issue': rec['issue'],
                'quickest_fix': rec['recommendations'][1] if len(rec['recommendations']) > 1 else rec['recommendations'][0],
                'expected_improvement': rec['expected_improvement'],
                'cost': rec['processing_cost']
            })
        
        # Sort by priority
        actions.sort(key=lambda x: priority_order[x['priority']])
        
        return actions
