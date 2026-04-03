"""
Patient cohort classifier
Integrates quantitative, clinical, and genomic data for patient stratification
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from typing import Dict, List, Tuple, Optional
import joblib
from pathlib import Path


class CohortClassifier:
    """정량 분석 + 임상 정보 기반 환자군 분류 (IP Module 2)"""
    
    def __init__(self, model_path: Optional[str] = None):
        self.scaler = StandardScaler()
        self.label_encoder_stage = LabelEncoder()
        self.label_encoder_grade = LabelEncoder()
        
        if model_path and Path(model_path).exists():
            self.classifier = joblib.load(model_path)
        else:
            self.classifier = GradientBoostingClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=5,
                random_state=42
            )
        
        self.cohort_definitions = {
            0: {
                'name': 'High-Risk Aggressive',
                'description': '고위험 공격적 암 - 다제 강력 요법 필요',
                'treatment_intensity': 'High'
            },
            1: {
                'name': 'Intermediate-Risk Standard',
                'description': '중등도 위험 - 표준 병용 요법',
                'treatment_intensity': 'Moderate'
            },
            2: {
                'name': 'Low-Risk Indolent',
                'description': '저위험 느린 진행 - 보존적 치료',
                'treatment_intensity': 'Low'
            },
            3: {
                'name': 'Targeted-Therapy Candidate',
                'description': '표적 치료 적합 - 분자 기반 정밀 치료',
                'treatment_intensity': 'Targeted'
            }
        }
    
    def prepare_integrated_features(
        self,
        quantitative_data: Dict[str, float],
        clinical_data: Dict[str, any],
        genomic_data: List[Dict[str, str]]
    ) -> pd.DataFrame:
        """
        특징 통합 및 전처리
        
        Args:
            quantitative_data: 병리 이미지 정량 분석 결과
            clinical_data: 임상 메타데이터
            genomic_data: 유전자 변이 정보 리스트
            
        Returns:
            통합 특징 DataFrame
        """
        features = {}
        
        # === 1. 정량 특징 (Quantitative) ===
        features['cell_count'] = quantitative_data.get('num_cells', 0)
        features['mean_area'] = quantitative_data.get('mean_area', 0)
        features['cv_area'] = quantitative_data.get('cv_area', 0)
        features['heterogeneity_score'] = quantitative_data.get('overall_heterogeneity', 0)
        features['spatial_clustering'] = quantitative_data.get('clustered_ratio', 0)
        features['clark_evans_index'] = quantitative_data.get('clark_evans_index', 1.0)
        features['density_variance'] = quantitative_data.get('density_variance', 0)
        
        # === 2. 임상 특징 (Clinical) ===
        features['age'] = clinical_data.get('age', 60)
        features['stage_encoded'] = self._encode_stage(clinical_data.get('stage', 'I'))
        features['grade_encoded'] = self._encode_grade(clinical_data.get('grade', 'well'))
        features['ecog_score'] = clinical_data.get('ecog_score', 0)
        
        # Biomarkers
        features['ki67_index'] = clinical_data.get('ki67_index', 0)
        features['tumor_size_mm'] = clinical_data.get('tumor_size_mm', 20)
        
        # === 3. 유전자 특징 (Genomic) ===
        genomic_features = self._extract_genomic_features(genomic_data)
        features.update(genomic_features)
        
        return pd.DataFrame([features])
    
    def _encode_stage(self, stage: str) -> int:
        """암 병기 인코딩"""
        stage_mapping = {
            'I': 1, 'IA': 1, 'IB': 1,
            'II': 2, 'IIA': 2, 'IIB': 2,
            'III': 3, 'IIIA': 3, 'IIIB': 3, 'IIIC': 3,
            'IV': 4, 'IVA': 4, 'IVB': 4
        }
        return stage_mapping.get(stage, 2)
    
    def _encode_grade(self, grade: str) -> int:
        """종양 등급 인코딩"""
        grade_mapping = {
            'well': 1, 'well differentiated': 1, 'G1': 1,
            'moderate': 2, 'moderately differentiated': 2, 'G2': 2,
            'poor': 3, 'poorly differentiated': 3, 'G3': 3,
            'undifferentiated': 4, 'G4': 4
        }
        return grade_mapping.get(grade.lower(), 2)
    
    def _extract_genomic_features(self, genomic_data: List[Dict]) -> Dict[str, int]:
        """유전자 변이 특징 추출"""
        features = {}
        
        # Common actionable mutations
        target_genes = [
            'KRAS', 'NRAS', 'BRAF', 'EGFR', 'ALK', 'ROS1',
            'PIK3CA', 'PTEN', 'TP53', 'APC', 'BRCA1', 'BRCA2',
            'HER2', 'MET', 'RET'
        ]
        
        for gene in target_genes:
            features[f'has_{gene}_mutation'] = int(
                any(v['gene_name'] == gene for v in genomic_data)
            )
        
        # Count total pathogenic variants
        pathogenic_count = sum(
            1 for v in genomic_data 
            if v.get('pathogenicity', '').lower() in ['pathogenic', 'likely pathogenic']
        )
        features['pathogenic_variant_count'] = pathogenic_count
        
        return features
    
    def classify_patient(
        self,
        quantitative_data: Dict[str, float],
        clinical_data: Dict[str, any],
        genomic_data: List[Dict[str, str]]
    ) -> Dict[str, any]:
        """
        환자 분류 및 코호트 할당
        
        Returns:
            분류 결과 딕셔너리
        """
        # Prepare features
        features = self.prepare_integrated_features(
            quantitative_data, clinical_data, genomic_data
        )
        
        # Check if model is trained
        try:
            # Try to check if scaler is fitted
            from sklearn.exceptions import NotFittedError
            try:
                self.scaler.check_is_fitted()
                model_fitted = True
            except (NotFittedError, AttributeError):
                model_fitted = False
        except:
            model_fitted = False
        
        # If model not trained, use rule-based classification
        if not model_fitted or not hasattr(self.classifier, 'predict'):
            return self._rule_based_classification(
                quantitative_data, clinical_data, genomic_data
            )
        
        # Scale features
        feature_cols = [c for c in features.columns if c not in ['patient_id']]
        features_scaled = self.scaler.transform(features[feature_cols])
        
        # Predict cohort
        cohort_id = int(self.classifier.predict(features_scaled)[0])
        probabilities = self.classifier.predict_proba(features_scaled)[0]
        confidence = float(np.max(probabilities))
        
        cohort_info = self.cohort_definitions[cohort_id]
        
        return {
            'cohort_id': cohort_id,
            'cohort_name': cohort_info['name'],
            'cohort_description': cohort_info['description'],
            'treatment_intensity': cohort_info['treatment_intensity'],
            'confidence_score': confidence,
            'alternative_cohorts': self._get_alternative_cohorts(probabilities, cohort_id),
            'classification_rationale': self._generate_rationale(
                features, cohort_id, quantitative_data, clinical_data, genomic_data
            )
        }
    
    def _rule_based_classification(
        self,
        quantitative_data: Dict[str, float],
        clinical_data: Dict[str, any],
        genomic_data: List[Dict]
    ) -> Dict[str, any]:
        """규칙 기반 분류 (모델 미학습 시)"""
        
        # Risk scoring
        risk_score = 0
        
        # Clinical factors
        stage = clinical_data.get('stage', 'I')
        if stage in ['III', 'IIIA', 'IIIB', 'IIIC', 'IV', 'IVA', 'IVB']:
            risk_score += 3
        elif stage in ['II', 'IIA', 'IIB']:
            risk_score += 1
        
        # Quantitative factors
        if quantitative_data.get('overall_heterogeneity', 0) > 0.7:
            risk_score += 2
        
        # Ki-67
        if clinical_data.get('ki67_index', 0) > 50:
            risk_score += 2
        
        # Genomic factors
        has_actionable_target = any(
            v['gene_name'] in ['EGFR', 'ALK', 'ROS1', 'BRAF', 'HER2']
            for v in genomic_data
        )
        
        # Assign cohort
        if has_actionable_target:
            cohort_id = 3  # Targeted therapy candidate
        elif risk_score >= 5:
            cohort_id = 0  # High-risk
        elif risk_score >= 3:
            cohort_id = 1  # Intermediate-risk
        else:
            cohort_id = 2  # Low-risk
        
        cohort_info = self.cohort_definitions[cohort_id]
        
        return {
            'cohort_id': cohort_id,
            'cohort_name': cohort_info['name'],
            'cohort_description': cohort_info['description'],
            'treatment_intensity': cohort_info['treatment_intensity'],
            'confidence_score': 0.85,  # Rule-based confidence
            'alternative_cohorts': [],
            'classification_rationale': self._generate_rationale(
                None, cohort_id, quantitative_data, clinical_data, genomic_data
            )
        }
    
    def _get_alternative_cohorts(
        self, 
        probabilities: np.ndarray, 
        primary_cohort: int
    ) -> List[Dict]:
        """대안 코호트 제시"""
        alternatives = []
        
        # Sort by probability
        sorted_indices = np.argsort(probabilities)[::-1]
        
        for idx in sorted_indices[1:3]:  # Top 2 alternatives
            if probabilities[idx] > 0.1:  # Only if probability > 10%
                alt_info = self.cohort_definitions[idx]
                alternatives.append({
                    'cohort_id': int(idx),
                    'cohort_name': alt_info['name'],
                    'probability': float(probabilities[idx])
                })
        
        return alternatives
    
    def _generate_rationale(
        self,
        features: Optional[pd.DataFrame],
        cohort_id: int,
        quantitative_data: Dict,
        clinical_data: Dict,
        genomic_data: List[Dict]
    ) -> List[str]:
        """분류 근거 생성"""
        rationale = []
        
        cohort_name = self.cohort_definitions[cohort_id]['name']
        rationale.append(f"환자군: {cohort_name}")
        
        # Quantitative factors
        if quantitative_data.get('overall_heterogeneity', 0) > 0.7:
            rationale.append(
                f"종양 이질성 {quantitative_data['overall_heterogeneity']:.2f} (매우 높음) "
                "- 다양한 세포 아형 존재"
            )
        
        # Clinical factors
        stage = clinical_data.get('stage', 'Unknown')
        rationale.append(f"병기: {stage}")
        
        if clinical_data.get('ki67_index', 0) > 30:
            rationale.append(
                f"Ki-67 지수 {clinical_data['ki67_index']}% - 높은 증식 활성"
            )
        
        # Genomic factors
        if genomic_data:
            genes = [v['gene_name'] for v in genomic_data]
            rationale.append(f"유전자 변이: {', '.join(genes[:5])}")
        
        return rationale
