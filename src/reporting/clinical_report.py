"""
Clinical report generator for precision oncology recommendations
Generates comprehensive treatment recommendation reports
"""

from typing import Dict, List
from datetime import datetime
import pandas as pd


class ClinicalReportGenerator:
    """병원용 임상 리포트 생성기 (IP Module 3 & Reporting)"""
    
    def generate_treatment_recommendation_report(
        self,
        patient_data: Dict,
        quantitative_analysis: Dict,
        clinical_metadata: Dict,
        genomic_data: List[Dict],
        cohort_classification: Dict,
        recommendation: Dict,
        dosage_plan: Dict,
        schedule: Dict
    ) -> Dict:
        """
        치료 추천 종합 리포트 생성
        
        Returns:
            구조화된 리포트 딕셔너리
        """
        report = {
            'report_id': self._generate_report_id(),
            'generation_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'report_version': '1.0.0',
            
            # Sections
            'header': self._create_header(patient_data),
            'patient_summary': self._create_patient_summary(patient_data, clinical_metadata),
            'quantitative_analysis_summary': self._format_quantitative_results(quantitative_analysis),
            'genomic_profile': self._format_genomic_profile(genomic_data),
            'cohort_classification': self._format_cohort_classification(cohort_classification),
            'treatment_recommendation': self._format_primary_recommendation(
                recommendation, dosage_plan, schedule
            ),
            'alternative_options': self._format_alternatives(recommendation.get('alternative_regimens', [])),
            'evidence_rationale': self._format_comprehensive_evidence(recommendation['evidence_summary']),
            'safety_considerations': self._format_warnings(recommendation.get('warnings', [])),
            'monitoring_plan': self._generate_monitoring_plan(recommendation),
            'references': self._add_references(recommendation)
        }
        
        return report
    
    def _generate_report_id(self) -> str:
        """리포트 ID 생성"""
        timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
        return f"ADDS-REP-{timestamp}"
    
    def _create_header(self, patient: Dict) -> Dict:
        """리포트 헤더"""
        return {
            'title': 'ADDS 정밀 항암 치료 추천 리포트',
            'subtitle': 'AI 기반 종양 병리 분석 및 맞춤형 약물 조합 추천',
            'patient_id': patient.get('patient_id', 'N/A'),
            'report_date': datetime.now().strftime('%Y년 %m월 %d일')
        }
    
    def _create_patient_summary(self, patient: Dict, clinical: Dict) -> Dict:
        """환자 요약"""
        return {
            'demographics': {
                '나이': f"{patient.get('age', 'N/A')}세",
                '성별': patient.get('gender', 'N/A'),
                '진단명': patient.get('cancer_type', 'N/A'),
                '병기': patient.get('stage', 'N/A'),
                '등급': patient.get('grade', 'N/A')
            },
            'clinical_status': {
                'ECOG': patient.get('ecog_score', 'N/A'),
                '진단일': patient.get('diagnosis_date', 'N/A'),
                '발생 부위': patient.get('primary_site', 'N/A')
            },
            'biomarkers': {
                'Ki-67 지수': f"{clinical.get('ki67_index', 'N/A')}%",
                'PD-L1 TPS': f"{clinical.get('pdl1_tps', 'N/A')}%",
                'MSI 상태': clinical.get('microsatellite_status', 'N/A')
            }
        }
    
    def _format_quantitative_results(self, quant: Dict) -> Dict:
        """정량 분석 결과 포맷팅"""
        return {
            '세포 분석': {
                '총 세포 수': f"{quant.get('num_cells', 0):,}개",
                '평균 세포 면적': f"{quant.get('mean_area', 0):.1f} px²",
                '세포 크기 변이도': f"CV = {quant.get('cv_area', 0):.2f}"
            },
            '종양 이질성': {
                '이질성 점수': f"{quant.get('overall_heterogeneity', 0):.2f}",
                '이질성 등급': quant.get('heterogeneity_grade', 'N/A'),
                '형태 다양성': f"{quant.get('shape_diversity', 0):.2f}"
            },
            '공간 분포': {
                'Clark-Evans 지수': f"{quant.get('clark_evans_index', 1.0):.2f}",
                '군집화 비율': f"{quant.get('clustered_ratio', 0)*100:.0f}%",
                '군집 수': f"{quant.get('num_clusters', 0)}개"
            },
            '임상적 의미': self._interpret_quantitative(quant)
        }
    
    def _interpret_quantitative(self, quant: Dict) -> str:
        """정량 분석 임상적 해석"""
        interpretations = []
        
        het_score = quant.get('overall_heterogeneity', 0)
        if het_score > 0.7:
            interpretations.append("매우 높은 종양 이질성으로 다양한 세포 아형 존재 추정")
        
        R = quant.get('clark_evans_index', 1.0)
        if R < 0.8:
            interpretations.append("세포 군집 분포 양상은 활발한 미세환경 상호작용 시사")
        
        return '; '.join(interpretations) if interpretations else "정상 범위"
    
    def _format_genomic_profile(self, genomic: List[Dict]) -> Dict:
        """유전자 프로파일 포맷팅"""
        if not genomic:
            return {'status': '유전자 검사 미실시'}
        
        actionable = [v for v in genomic if v.get('pathogenicity') in ['Pathogenic', 'Likely pathogenic']]
        
        return {
            '변이 개수': len(genomic),
            '병원성 변이': len(actionable),
            '주요 변이': [
                {
                    '유전자': v['gene_name'],
                    '변이': v.get('variant_detail', 'N/A'),
                    '병원성': v.get('pathogenicity', 'N/A'),
                    'VAF': f"{v.get('allele_frequency', 0)*100:.1f}%"
                }
                for v in genomic[:5]  # Top 5
            ],
            '표적 치료 가능성': 'Yes' if any(
                v['gene_name'] in ['EGFR', 'KRAS', 'BRAF', 'HER2', 'ALK', 'ROS1']
                for v in genomic
            ) else 'No'
        }
    
    def _format_cohort_classification(self, cohort: Dict) -> Dict:
        """환자군 분류 포맷팅"""
        return {
            '환자군': cohort['cohort_name'],
            '설명': cohort['cohort_description'],
            '치료 강도': cohort['treatment_intensity'],
            '분류 신뢰도': f"{cohort['confidence_score']*100:.0f}%",
            '분류 근거': cohort.get('classification_rationale', [])
        }
    
    def _format_primary_recommendation(
        self,
        recommendation: Dict,
        dosage: Dict,
        schedule: Dict
    ) -> Dict:
        """주 추천 조합 포맷팅"""
        primary = recommendation['primary_regimen']
        
        return {
            '추천 프로토콜': primary.get('name', 'Custom Regimen'),
            '약물 조합': primary['drugs'],
            '치료 강도': primary.get('intensity', 'N/A'),
            '근거 수준': primary.get('evidence_level', 'N/A'),
            '신뢰도': recommendation['confidence_level'],
            '용량 계획': [
                {
                    '약물': d['drug_name'],
                    '용량': f"{d['final_dose_mg']} mg",
                    '투여 경로': d['administration_route'],
                    '조정 인자': d.get('adjustments', {})
                }
                for d in dosage['drugs']
            ],
            '투여 스케줄': {
                '시작일': schedule['start_date'],
                '총 사이클': schedule['num_cycles'],
                '예상 완료일': schedule['estimated_completion'],
                '사이클 길이': f"{schedule['total_weeks'] // schedule['num_cycles']}주"
            }
        }
    
    def _format_alternatives(self, alternatives: List[Dict]) -> List[Dict]:
        """대안 조합 포맷팅"""
        return [
            {
                '프로토콜': alt.get('name', 'Alternative'),
                '약물': alt['drugs'],
                '적용 시나리오': self._when_to_consider_alternative(alt)
            }
            for alt in alternatives
        ]
    
    def _when_to_consider_alternative(self, alt: Dict) -> str:
        """대안 고려 시점"""
        if alt.get('intensity') == 'Low':
            return "환자 상태 악화 또는 부작용 발생 시"
        elif len(alt['drugs']) < 3:
            return "간/신기능 저하 시 고려"
        else:
            return "반응 불충분 시 치료 강화"
    
    def _format_comprehensive_evidence(self, evidence: Dict) -> Dict:
        """종합 근거 포맷팅 - 핵심!"""
        formatted = {
            '환자군 기반 근거': evidence.get('cohort_based_rationale', 'N/A'),
            '정량 분석 근거': [],
            '임상 가이드라인': evidence.get('clinical_guidelines', []),
            '유전자 기반 근거': [],
            '참고 문헌': evidence.get('supporting_literature', [])
        }
        
        # 정량 지표 근거
        for indicator in evidence.get('quantitative_indicators', []):
            formatted['정량 분석 근거'].append({
                '지표': indicator['metric'],
                '측정값': indicator['value'],
                '해석': indicator['interpretation'],
                '치료 결정 영향': indicator['decision_impact']
            })
        
        # 유전자 근거
        for genomic in evidence.get('genomic_rationale', []):
            formatted['유전자 기반 근거'].append({
                '표적 유전자': genomic.get('target_gene', 'N/A'),
                '변이': genomic.get('variant', 'N/A'),
                '선택 약물': genomic.get('drug', 'N/A'),
                '기전': genomic.get('mechanism', 'N/A')
            })
        
        return formatted
    
    def _format_warnings(self, warnings: List[str]) -> Dict:
        """주의사항 포맷팅"""
        return {
            '주의사항': warnings,
            '모니터링 필요': [
                '혈액 검사 (CBC, LFT, RFT) - 매 사이클 전',
                '심전도 (특정 약물 사용 시)',
                '부작용 평가 - 매 투여 전'
            ]
        }
    
    def _generate_monitoring_plan(self, recommendation: Dict) -> Dict:
        """모니터링 계획"""
        return {
            '치료 전': [
                'Baseline 혈액 검사 (CBC, CMP)',
                '심기능 평가 (LVEF)',
                '간/신기능 평가'
            ],
            '치료 중': [
                '매 사이클 전 혈액 검사',
                '반응 평가 (2사이클마다 영상 검사)',
                '부작용 모니터링 (CTCAE 기준)'
            ],
            '추적 관찰': [
                '치료 종료 후 3개월마다 영상 검사',
                '재발 모니터링',
                '장기 부작용 평가'
            ]
        }
    
    def _add_references(self, recommendation: Dict) -> List[str]:
        """참고 문헌"""
        return [
            "NCCN Clinical Practice Guidelines in Oncology (최신판)",
            "ESMO Clinical Practice Guidelines (최신판)",
            "특허: AI 기반 암세포 이미지 분석 및 정밀 항암제 조합 추천 시스템 (10-2025-0207756)"
        ]
    
    def export_to_markdown(self, report: Dict) -> str:
        """마크다운 형식 내보내기"""
        md = []
        
        # Header
        md.append(f"# {report['header']['title']}")
        md.append(f"## {report['header']['subtitle']}\n")
        md.append(f"**리포트 ID:** {report['report_id']}")
        md.append(f"**생성일:** {report['generation_date']}\n")
        md.append("---\n")
        
        # Patient Summary
        md.append("## 1. 환자 정보\n")
        summary = report['patient_summary']
        for category, items in summary.items():
            md.append(f"### {category}\n")
            for key, value in items.items():
                md.append(f"- **{key}:** {value}")
            md.append("")
        
        # Quantitative Analysis
        md.append("## 2. 정량 병리 분석 결과\n")
        quant = report['quantitative_analysis_summary']
        for section, data in quant.items():
            md.append(f"### {section}\n")
            if isinstance(data, dict):
                for key, value in data.items():
                    md.append(f"- **{key}:** {value}")
            else:
                md.append(f"{data}")
            md.append("")
        
        # Treatment Recommendation
        md.append("## 3. 치료 추천\n")
        rec = report['treatment_recommendation']
        md.append(f"### 주 추천 프로토콜: {rec['추천 프로토콜']}\n")
        md.append(f"**약물 조합:** {', '.join(rec['약물 조합'])}\n")
        md.append(f"**신뢰도:** {rec['신뢰도']}\n")
        
        # Evidence
        md.append("## 4. 추천 근거\n")
        evidence = report['evidence_rationale']
        md.append(f"**환자군 매칭:** {evidence['환자군 기반 근거']}\n")
        
        if evidence['정량 분석 근거']:
            md.append("### 정량 지표 근거:\n")
            for indicator in evidence['정량 분석 근거']:
                md.append(f"- **{indicator['지표']}** = {indicator['측정값']}")
                md.append(f"  - {indicator['해석']}")
                md.append(f"  - ➡️ {indicator['치료 결정 영향']}\n")
        
        return '\n'.join(md)
