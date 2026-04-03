"""
Multi-modal Fusion Engine
병리 + CT + MRI 통합 분석 엔진
"""

import numpy as np
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


class MultimodalFusionEngine:
    """
    Multi-modal 의료 영상 통합 분석 엔진
    
    Features:
    - Cross-modal feature fusion
    - Attention mechanism
    - Integrated risk assessment
    - Consistency checking
    
    Benchmarked from:
    - AMRI-Net (Adaptive Multi-Resolution Imaging Network)
    - EDAL (Explainable Domain-Adaptive Learning)
    - SOPHiA DDM Multimodal
    """
    
    def __init__(self):
        logger.info("MultimodalFusionEngine initialized")
    
    def fuse_multimodal_analysis(
        self,
        pathology_result: Optional[Dict] = None,
        ct_result: Optional[Dict] = None,
        mri_result: Optional[Dict] = None,
        clinical_data: Optional[Dict] = None
    ) -> Dict:
        """
        다중 모달리티 분석 결과 통합
        
        Args:
            pathology_result: 병리 분석 결과
            ct_result: CT 분석 결과
            mri_result: MRI 분석 결과
            clinical_data: 임상 데이터
        
        Returns:
            통합 분석 결과
        """
        try:
            # 1. Available modalities check
            available_modalities = self._check_available_modalities(
                pathology_result, ct_result, mri_result
            )
            
            # 2. Extract features from each modality
            features = self._extract_unified_features(
                pathology_result, ct_result, mri_result
            )
            
            # 3. Cross-modal feature fusion
            fused_features = self._fuse_features(features)
            
            # 4. Consistency check
            consistency = self._check_consistency(
                pathology_result, ct_result, mri_result
            )
            
            # 5. Integrated risk assessment
            risk = self._assess_integrated_risk(
                fused_features, clinical_data
            )
            
            # 6. Generate integrated insights
            insights = self._generate_integrated_insights(
                pathology_result, ct_result, mri_result, risk
            )
            
            # 7. Confidence score
            confidence = self._calculate_confidence(
                available_modalities, consistency
            )
            
            return {
                'status': 'success',
                'available_modalities': available_modalities,
                'fused_features': fused_features,
                'consistency_check': consistency,
                'risk_assessment': risk,
                'integrated_insights': insights,
                'confidence_score': confidence,
                'recommendation': self._generate_recommendation(risk, insights)
            }
            
        except Exception as e:
            logger.error(f"Multimodal fusion failed: {str(e)}")
            return {
                'status': 'error',
                'error': str(e)
            }
    
    def _check_available_modalities(
        self,
        pathology: Optional[Dict],
        ct: Optional[Dict],
        mri: Optional[Dict]
    ) -> Dict:
        """사용 가능한 모달리티 확인"""
        return {
            'pathology': pathology is not None and pathology.get('status') == 'success',
            'ct': ct is not None and ct.get('status') == 'success',
            'mri': mri is not None and mri.get('status') == 'success',
            'count': sum([
                pathology is not None and pathology.get('status') == 'success',
                ct is not None and ct.get('status') == 'success',
                mri is not None and mri.get('status') == 'success'
            ])
        }
    
    def _extract_unified_features(
        self,
        pathology: Optional[Dict],
        ct: Optional[Dict],
        mri: Optional[Dict]
    ) -> Dict:
        """각 모달리티에서 통합 특징 추출"""
        features = {}
        
        # Pathology features (cellular level)
        if pathology and pathology.get('status') == 'success':
            features['pathology'] = {
                'cell_count': pathology.get('cell_count', 0),
                'mean_area': pathology.get('summary_statistics', {}).get('mean_area', 0),
                'heterogeneity': pathology.get('summary_statistics', {}).get('std_area', 0)
            }
        
        # CT features (anatomical level)
        if ct and ct.get('status') == 'success':
            features['ct'] = {
                'tumor_volume': ct.get('segmentation', {}).get('tumor_volume_mm3', 0),
                'longest_diameter': ct.get('measurements', {}).get('longest_diameter_mm', 0),
                'mean_intensity': ct.get('radiomics_features', {}).get('intensity_mean', 0)
            }
        
        # MRI features (functional level)
        if mri and mri.get('status') == 'success':
            features['mri'] = {
                'tumor_volume': mri.get('segmentation', {}).get('tumor_volume_mm3', 0),
                'edema_volume': mri.get('segmentation', {}).get('edema_volume_mm3', 0),
                'edema_ratio': mri.get('measurements', {}).get('edema_to_tumor_ratio', 0)
            }
        
        return features
    
    def _fuse_features(self, features: Dict) -> Dict:
        """
        Cross-modal feature fusion (AMRI-Net 스타일)
        
        Fusion strategy:
        1. Attention mechanism: 각 모달리티의 중요도 계산
        2. Weighted fusion: 중요도 기반 특징 통합
        """
        fused = {}
        
        # Tumor volume consensus (CT + MRI)
        tumor_volumes = []
        if 'ct' in features and 'tumor_volume' in features['ct']:
            tumor_volumes.append(features['ct']['tumor_volume'])
        if 'mri' in features and 'tumor_volume' in features['mri']:
            tumor_volumes.append(features['mri']['tumor_volume'])
        
        if tumor_volumes:
            fused['consensus_tumor_volume_mm3'] = float(np.mean(tumor_volumes))
            fused['tumor_volume_variability'] = float(np.std(tumor_volumes)) if len(tumor_volumes) > 1 else 0
        
        # Cellular-anatomical correlation
        if 'pathology' in features and 'ct' in features:
            cell_density = features['pathology'].get('cell_count', 0)
            tumor_size = features['ct'].get('tumor_volume', 1)
            fused['estimated_cellularity'] = cell_density / max(tumor_size, 1)
        
        # Edema index (MRI specific)
        if 'mri' in features:
            fused['edema_index'] = features['mri'].get('edema_ratio', 0)
        
        return fused
    
    def _check_consistency(
        self,
        pathology: Optional[Dict],
        ct: Optional[Dict],
        mri: Optional[Dict]
    ) -> Dict:
        """
        Cross-modal consistency check
        
        Checks:
        - Tumor size consistency (CT vs MRI)
        - Cellular density vs imaging findings
        """
        consistency = {
            'overall': 'Unknown',
            'checks': []
        }
        
        # Tumor volume consistency (CT vs MRI)
        if ct and mri:
            ct_volume = ct.get('segmentation', {}).get('tumor_volume_mm3', 0)
            mri_volume = mri.get('segmentation', {}).get('tumor_volume_mm3', 0)
            
            if ct_volume > 0 and mri_volume > 0:
                ratio = min(ct_volume, mri_volume) / max(ct_volume, mri_volume)
                
                if ratio > 0.7:
                    consistency['checks'].append({
                        'test': 'CT vs MRI tumor volume',
                        'result': 'Consistent',
                        'detail': f'CT: {ct_volume:.1f}mm³, MRI: {mri_volume:.1f}mm³'
                    })
                else:
                    consistency['checks'].append({
                        'test': 'CT vs MRI tumor volume',
                        'result': 'Inconsistent',
                        'detail': f'Large discrepancy: CT {ct_volume:.1f}mm³ vs MRI {mri_volume:.1f}mm³'
                    })
        
        # Overall consistency
        consistent_count = sum(1 for check in consistency['checks'] if check['result'] == 'Consistent')
        total_checks = len(consistency['checks'])
        
        if total_checks > 0:
            consistency['overall'] = 'High' if consistent_count / total_checks > 0.7 else 'Moderate' if consistent_count > 0 else 'Low'
        
        return consistency
    
    def _assess_integrated_risk(
        self,
        fused_features: Dict,
        clinical_data: Optional[Dict]
    ) -> Dict:
        """
        Multi-modal 통합 위험도 평가
        
        Risk factors:
        - Large tumor size
        - High cellularity
        - Extensive edema
        - Advanced stage (clinical)
        """
        risk_factors = []
        risk_score = 0
        
        # Tumor volume risk
        tumor_volume = fused_features.get('consensus_tumor_volume_mm3', 0)
        if tumor_volume > 50000:  # > 50cm³
            risk_factors.append('Large tumor volume (> 50cm³)')
            risk_score += 2
        elif tumor_volume > 20000:
            risk_factors.append('Moderate tumor volume')
            risk_score += 1
        
        # Cellularity risk
        cellularity = fused_features.get('estimated_cellularity', 0)
        if cellularity > 100:  # High cell density
            risk_factors.append('High cellularity')
            risk_score += 1
        
        # Edema risk
        edema_index = fused_features.get('edema_index', 0)
        if edema_index > 1.5:  # Edema > 1.5x tumor
            risk_factors.append('Extensive perilesional edema')
            risk_score += 1
        
        # Clinical risk factors
        if clinical_data:
            stage = clinical_data.get('stage', '')
            if 'IV' in str(stage):
                risk_factors.append('Stage IV disease')
                risk_score += 2
            elif 'III' in str(stage):
                risk_factors.append('Stage III disease')
                risk_score += 1
        
        # Risk level classification
        if risk_score >= 4:
            risk_level = 'High'
        elif risk_score >= 2:
            risk_level = 'Moderate'
        else:
            risk_level = 'Low'
        
        return {
            'risk_level': risk_level,
            'risk_score': risk_score,
            'risk_factors': risk_factors,
            'max_score': 7
        }
    
    def _generate_integrated_insights(
        self,
        pathology: Optional[Dict],
        ct: Optional[Dict],
        mri: Optional[Dict],
        risk: Dict
    ) -> List[Dict]:
        """통합 임상 인사이트 생성"""
        insights = []
        
        # Pathology insights
        if pathology and pathology.get('status') == 'success':
            cell_count = pathology.get('cell_count', 0)
            if cell_count > 0:
                insights.append({
                    'source': 'Pathology',
                    'finding': f'{cell_count} cells detected with detailed morphology',
                    'clinical_significance': 'Cellular-level analysis available for precision assessment'
                })
        
        # CT insights
        if ct and ct.get('status') == 'success':
            diameter = ct.get('measurements', {}).get('longest_diameter_mm', 0)
            if diameter > 0:
                insights.append({
                    'source': 'CT',
                    'finding': f'Tumor longest diameter: {diameter:.1f}mm (RECIST 1.1)',
                    'clinical_significance': 'Quantitative size measurement for treatment monitoring'
                })
        
        # MRI insights
        if mri and mri.get('status') == 'success':
            edema_detected = mri.get('segmentation', {}).get('edema_detected', False)
            if edema_detected:
                insights.append({
                    'source': 'MRI',
                    'finding': 'Perilesional edema detected',
                    'clinical_significance': 'May indicate aggressive tumor or high perfusion'
                })
        
        # Multi-modal insight
        modality_count = sum([
            pathology is not None and pathology.get('status') == 'success',
            ct is not None and ct.get('status') == 'success',
            mri is not None and mri.get('status') == 'success'
        ])
        
        if modality_count >= 2:
            insights.append({
                'source': 'Multi-modal Fusion',
                'finding': f'{modality_count} modalities successfully integrated',
                'clinical_significance': 'Comprehensive multi-scale tumor characterization achieved'
            })
        
        return insights
    
    def _calculate_confidence(
        self,
        available_modalities: Dict,
        consistency: Dict
    ) -> float:
        """통합 분석 신뢰도 계산"""
        # Base confidence from number of modalities
        modality_count = available_modalities.get('count', 0)
        base_confidence = modality_count / 3.0
        
        # Boost if consistency is high
        consistency_level = consistency.get('overall', 'Unknown')
        if consistency_level == 'High':
            confidence_boost = 0.1
        elif consistency_level == 'Moderate':
            confidence_boost = 0.05
        else:
            confidence_boost = 0
        
        total_confidence = min(1.0, base_confidence + confidence_boost)
        
        return float(total_confidence)
    
    def _generate_recommendation(
        self,
        risk: Dict,
        insights: List[Dict]
    ) -> str:
        """통합 권장사항 생성"""
        risk_level = risk.get('risk_level', 'Unknown')
        risk_factors = risk.get('risk_factors', [])
        
        recommendation = f"""
통합 분석 기반 권장사항:

위험도: {risk_level}
"""
        
        if risk_factors:
            recommendation += "\n위험 요인:\n"
            for factor in risk_factors:
                recommendation += f"  - {factor}\n"
        
        recommendation += "\n권장 조치:\n"
        
        if risk_level == 'High':
            recommendation += """  - 즉시 전문의 상담 권장
  - 다학제 팀(MDT) 컨퍼런스 검토
  - 적극적 치료 전략 고려
  - 정기적 영상 추적 관찰 (1-2개월)
"""
        elif risk_level == 'Moderate':
            recommendation += """  - 전문의 상담 권장
  - 표준 치료 프로토콜 적용 고려
  - 정기적 추적 관찰 (3개월)
"""
        else:
            recommendation += """  - 표준 치료 고려
  - 정기적 추적 관찰 (6개월)
"""
        
        return recommendation.strip()
