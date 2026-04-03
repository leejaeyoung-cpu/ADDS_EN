"""
Integrated dataset builder
Combines Cellpose quantitative analysis + OpenAI interpretation
"""

import json
from typing import Dict, Optional, List
from pathlib import Path
import pandas as pd


class IntegratedDatasetBuilder:
    """통합 데이터셋 생성기 (Cellpose + OpenAI + Clinical)"""
    
    def __init__(self):
        self.dataset_dir = Path('data/integrated_datasets')
        self.dataset_dir.mkdir(parents=True, exist_ok=True)
    
    def build_comprehensive_patient_record(
        self,
        patient_id: str,
        patient_info: Dict,
        cellpose_analysis: Optional[Dict] = None,
        openai_image_analysis: Optional[Dict] = None,
        openai_report_analysis: Optional[Dict] = None,
        openai_radiology_analysis: Optional[Dict] = None,
        pathology_results: Optional[List[Dict]] = None,  # 다중 병리 이미지 결과
        ct_results: Optional[List[Dict]] = None,  # 다중 CT 결과
        mri_results: Optional[List[Dict]] = None,  # 다중 MRI 결과
        multimodal_fusion: Optional[Dict] = None,  # Multi-modal fusion 결과
        clinical_files: Optional[Dict] = None
    ) -> Dict:
        """
        종합 환자 레코드 생성 (다중 이미지 지원)
        
        Args:
            patient_id: 환자 ID
            patient_info: 기본 환자 정보
            cellpose_analysis: Cellpose 정량 분석 결과 (단일, 레거시)
            openai_image_analysis: OpenAI 병리 이미지 분석 결과 (단일, 레거시)
            openai_report_analysis: OpenAI 소견서 분석 결과
            openai_radiology_analysis: OpenAI CT/MRI 영상 분석 결과 (단일, 레거시)
            pathology_results: 다중 병리 이미지 분석 결과 리스트
            ct_results: 다중 CT 분석 결과 리스트
            mri_results: 다중 MRI 분석 결과 리스트
            multimodal_fusion: Multi-modal fusion 통합 결과
            clinical_files: 업로드된 파일 정보
            
        Returns:
            통합 데이터셋 레코드
        """
        record = {
            'patient_id': patient_id,
            'timestamp': pd.Timestamp.now().isoformat(),
            
            # 1. 기본 환자 정보
            'patient_demographics': {
                'age': patient_info.get('age'),
                'gender': patient_info.get('gender'),
                'cancer_type': patient_info.get('cancer_type'),
                'stage': patient_info.get('stage'),
                'grade': patient_info.get('grade'),
                'diagnosis_date': patient_info.get('diagnosis_date')
            },
            
            # 2. Cellpose 정량 분석
            'cellpose_quantitative': self._extract_cellpose_features(cellpose_analysis),
            
            # 3. OpenAI 병리 이미지 해석
            'ai_image_interpretation': self._extract_ai_image_features(openai_image_analysis),
            
            # 4. OpenAI CT/MRI 영상 해석 (신규)
            'ai_radiology_interpretation': self._extract_ai_image_features(openai_radiology_analysis),
            
            # 5. OpenAI 소견서 해석
            'ai_report_interpretation': self._extract_ai_report_features(openai_report_analysis),
            
            # 6. 통합 분석 (Cellpose + OpenAI)
            'integrated_analysis': self._generate_integrated_analysis(
                cellpose_analysis,
                openai_image_analysis,
                openai_report_analysis,
                openai_radiology_analysis  # CT/MRI 포함
            ),
            
            # 7. 파일 메타데이터
            'files': clinical_files or {},
            
            # 8. 다중 이미지 분석 결과
            'multi_image_analysis': {
                'pathology': pathology_results or [],
                'ct': ct_results or [],
                'mri': mri_results or [],
                'counts': {
                    'pathology_images': len(pathology_results) if pathology_results else 0,
                    'ct_images': len(ct_results) if ct_results else 0,
                    'mri_images': len(mri_results) if mri_results else 0
                }
            },
            
            # 9. Multi-modal fusion 결과
            'multimodal_fusion': multimodal_fusion or {},
            
            # 10. 데이터 품질 점수
            'data_quality': self._assess_data_quality(
                cellpose_analysis,
                openai_image_analysis,
                openai_report_analysis,
                openai_radiology_analysis  # CT/MRI 포함
            )
        }
        
        return record
    
    def _extract_cellpose_features(self, analysis: Optional[Dict]) -> Dict:
        """Cellpose 정량 특징 추출"""
        if not analysis:
            return {}
        
        return {
            'cell_count': analysis.get('num_cells', 0),
            'morphology': {
                'mean_area': analysis.get('mean_area', 0),
                'std_area': analysis.get('std_area', 0),
                'cv_area': analysis.get('cv_area', 0),
                'mean_circularity': analysis.get('mean_circularity', 0),
                'mean_eccentricity': analysis.get('mean_eccentricity', 0)
            },
            'spatial_distribution': {
                'clark_evans_index': analysis.get('clark_evans_index', 1.0),
                'clustered_ratio': analysis.get('clustered_ratio', 0),
                'num_clusters': analysis.get('num_clusters', 0),
                'mean_nnd': analysis.get('mean_nnd', 0)
            },
            'heterogeneity': {
                'overall_score': analysis.get('overall_heterogeneity', 0),
                'grade': analysis.get('heterogeneity_grade', 'Unknown'),
                'size_entropy': analysis.get('size_entropy', 0),
                'shape_diversity': analysis.get('shape_diversity', 0)
            }
        }
    
    def _extract_ai_image_features(self, analysis: Optional[Dict]) -> Dict:
        """OpenAI 이미지 분석 특징 추출"""
        if not analysis or analysis.get('status') != 'success':
            return {}
        
        ai_result = analysis.get('analysis', {})
        
        return {
            'morphology': ai_result.get('morphology', {}),
            'histology': ai_result.get('histology', {}),
            'grade': ai_result.get('grade', ''),
            'quantitative_estimation': ai_result.get('quantitative', {}),
            'diagnosis': ai_result.get('diagnosis', ''),
            'recommendations': ai_result.get('recommendations', [])
        }
    
    def _extract_ai_report_features(self, analysis: Optional[Dict]) -> Dict:
        """OpenAI 소견서 분석 특징 추출"""
        if not analysis or analysis.get('status') != 'success':
            return {}
        
        ai_result = analysis.get('analysis', {})
        
        return {
            'extracted_patient_info': ai_result.get('patient_info', {}),
            'tumor_characteristics': ai_result.get('tumor_info', {}),
            'pathology_details': ai_result.get('pathology', {}),
            'biomarkers': ai_result.get('biomarkers', {}),
            'genomic_variants': ai_result.get('genomic_variants', []),
            'treatment_recommendations': ai_result.get('recommendations', [])
        }
    
    def _generate_integrated_analysis(
        self,
        cellpose: Optional[Dict],
        ai_image: Optional[Dict],
        ai_report: Optional[Dict],
        ai_radiology: Optional[Dict] = None  # CT/MRI 추가
    ) -> Dict:
        """통합 분석 생성 (핵심!)"""
        integrated = {
            'consistency_check': {},
            'combined_insights': [],
            'risk_assessment': {},
            'confidence_score': 0.0
        }
        
        # Consistency check: Cellpose vs AI
        if cellpose and ai_image:
            integrated['consistency_check'] = self._check_consistency(
                cellpose, ai_image.get('analysis', {})
            )
        
        # Combined insights
        insights = []
        
        # From Cellpose
        if cellpose:
            if cellpose.get('overall_heterogeneity', 0) > 0.7:
                insights.append({
                    'source': 'Cellpose',
                    'finding': 'High tumor heterogeneity',
                    'quantitative_value': cellpose.get('overall_heterogeneity'),
                    'clinical_significance': 'Increased treatment complexity'
                })
        
        # From AI Image
        if ai_image and ai_image.get('status') == 'success':
            ai_grade = ai_image.get('analysis', {}).get('grade', '')
            if 'poor' in ai_grade.lower():
                insights.append({
                    'source': 'AI Image Analysis',
                    'finding': f'Poor differentiation ({ai_grade})',
                    'clinical_significance': 'Aggressive tumor behavior'
                })
        
        # From AI Radiology (CT/MRI) - 신규
        if ai_radiology and ai_radiology.get('status') == 'success':
            radiology_findings = ai_radiology.get('analysis', {}).get('interpretation', '')
            if radiology_findings:
                insights.append({
                    'source': 'AI Radiology (CT/MRI)',
                    'finding': radiology_findings[:200],  # 첫 200자만
                    'clinical_significance': 'Imaging-based assessment'
                })
        
        # From AI Report
        if ai_report and ai_report.get('status') == 'success':
            biomarkers = ai_report.get('analysis', {}).get('biomarkers', {})
            ki67 = biomarkers.get('ki67')
            if ki67 and ki67 > 50:
                insights.append({
                    'source': 'AI Report Analysis',
                    'finding': f'High Ki-67 index ({ki67}%)',
                    'clinical_significance': 'High proliferative activity'
                })
        
        integrated['combined_insights'] = insights
        
        # Risk assessment
        risk_factors = []
        if cellpose and cellpose.get('overall_heterogeneity', 0) > 0.7:
            risk_factors.append('High heterogeneity')
        if ai_report:
            stage = ai_report.get('analysis', {}).get('tumor_info', {}).get('stage', '')
            if 'IV' in stage or 'III' in stage:
                risk_factors.append(f'Advanced stage ({stage})')
        
        integrated['risk_assessment'] = {
            'risk_factors': risk_factors,
            'risk_level': 'High' if len(risk_factors) >= 2 else 'Moderate' if len(risk_factors) == 1 else 'Low'
        }
        
        # Confidence score (4개 소스로 업데이트)
        data_sources = sum([
            1 if cellpose else 0,
            1 if ai_image and ai_image.get('status') == 'success' else 0,
            1 if ai_radiology and ai_radiology.get('status') == 'success' else 0,  # CT/MRI 추가
            1 if ai_report and ai_report.get('status') == 'success' else 0
        ])
        integrated['confidence_score'] = data_sources / 4.0  # 3.0 -> 4.0
        
        return integrated
    
    def _check_consistency(self, cellpose: Dict, ai_analysis: Dict) -> Dict:
        """Cellpose와 AI 분석 일관성 체크"""
        consistency = {
            'cell_density': 'Unknown',
            'heterogeneity': 'Unknown',
            'overall': 'Unknown'
        }
        
        # Cell density comparison
        cellpose_count = cellpose.get('num_cells', 0)
        ai_density = ai_analysis.get('quantitative', {}).get('cell_density', '').lower()
        
        if cellpose_count > 700 and 'high' in ai_density:
            consistency['cell_density'] = 'Consistent'
        elif cellpose_count < 300 and 'low' in ai_density:
            consistency['cell_density'] = 'Consistent'
        elif ai_density:
            consistency['cell_density'] = 'Inconsistent'
        
        # Heterogeneity comparison
        cellpose_het = cellpose.get('overall_heterogeneity', 0)
        ai_het = ai_analysis.get('quantitative', {}).get('heterogeneity', '').lower()
        
        if cellpose_het > 0.7 and 'high' in ai_het:
            consistency['heterogeneity'] = 'Consistent'
        elif cellpose_het < 0.3 and 'low' in ai_het:
            consistency['heterogeneity'] = 'Consistent'
        elif ai_het:
            consistency['heterogeneity'] = 'Inconsistent'
        
        # Overall
        consistent_count = sum(1 for v in consistency.values() if v == 'Consistent')
        if consistent_count >= 2:
            consistency['overall'] = 'High agreement'
        elif consistent_count == 1:
            consistency['overall'] = 'Moderate agreement'
        else:
            consistency['overall'] = 'Low agreement'
        
        return consistency
    
    def _assess_data_quality(
        self,
        cellpose: Optional[Dict],
        ai_image: Optional[Dict],
        ai_report: Optional[Dict],
        ai_radiology: Optional[Dict] = None  # CT/MRI 추가
    ) -> Dict:
        """데이터 품질 평가"""
        quality = {
            'completeness': 0.0,
            'sources': [],
            'missing': []
        }
        
        # Check available sources
        if cellpose:
            quality['sources'].append('Cellpose quantitative analysis')
        else:
            quality['missing'].append('Cellpose analysis')
        
        if ai_image and ai_image.get('status') == 'success':
            quality['sources'].append('AI image interpretation')
        else:
            quality['missing'].append('AI image analysis')
        
        if ai_radiology and ai_radiology.get('status') == 'success':
            quality['sources'].append('AI radiology (CT/MRI) interpretation')
        else:
            quality['missing'].append('AI radiology analysis')
        
        if ai_report and ai_report.get('status') == 'success':
            quality['sources'].append('AI report interpretation')
        else:
            quality['missing'].append('AI report analysis')
        
        # 4개 소스로 업데이트
        quality['completeness'] = len(quality['sources']) / 4.0  # 3.0 -> 4.0
        
        return quality
    
    def save_to_dataset(self, record: Dict) -> str:
        """데이터셋에 레코드 저장"""
        patient_id = record['patient_id']
        
        # Save individual record
        record_path = self.dataset_dir / f"{patient_id}_integrated.json"
        with open(record_path, 'w', encoding='utf-8') as f:
            json.dump(record, f, indent=2, ensure_ascii=False)
        
        # Append to master dataset
        master_path = self.dataset_dir / 'master_dataset.jsonl'
        with open(master_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')
        
        return str(record_path)
    
    def export_to_dataframe(self) -> pd.DataFrame:
        """데이터셋을 DataFrame으로 변환"""
        master_path = self.dataset_dir / 'master_dataset.jsonl'
        
        if not master_path.exists():
            return pd.DataFrame()
        
        records = []
        with open(master_path, 'r', encoding='utf-8') as f:
            for line in f:
                records.append(json.loads(line))
        
        # Flatten nested structure for DataFrame
        flattened = []
        for record in records:
            flat = {
                'patient_id': record['patient_id'],
                'timestamp': record['timestamp'],
                'age': record['patient_demographics'].get('age'),
                'gender': record['patient_demographics'].get('gender'),
                'cancer_type': record['patient_demographics'].get('cancer_type'),
                'stage': record['patient_demographics'].get('stage'),
                
                # Cellpose
                'cell_count': record['cellpose_quantitative'].get('cell_count'),
                'heterogeneity': record['cellpose_quantitative'].get('heterogeneity', {}).get('overall_score'),
                'clark_evans_index': record['cellpose_quantitative'].get('spatial_distribution', {}).get('clark_evans_index'),
                
                # AI grade
                'ai_grade': record['ai_image_interpretation'].get('grade'),
                
                # Risk
                'risk_level': record['integrated_analysis'].get('risk_assessment', {}).get('risk_level'),
                'confidence': record['integrated_analysis'].get('confidence_score'),
                
                # Quality
                'data_completeness': record['data_quality'].get('completeness')
            }
            flattened.append(flat)
        
        return pd.DataFrame(flattened)
