"""
ADDS CDSS Integration Engine
=============================
Multi-modal data integration for Clinical Decision Support System

Integrates:
- Cellpose cell analysis results
- CT tumor detection results
- Clinical data (genomic, lab results)
- AI therapy selection
- OpenAI medical interpretation
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import numpy as np
from datetime import datetime
import json


@dataclass
class CellposeResults:
    """Cellpose cell analysis results"""
    cell_count: int
    mean_area_um2: float
    mean_circularity: float
    morphology_score: float  # 0-10
    ki67_index: float  # 0-1 (proliferation rate)
    image_path: Optional[str] = None
    
    def to_dict(self) -> Dict:
        return {
            'cell_count': self.cell_count,
            'mean_area_um2': self.mean_area_um2,
            'mean_circularity': self.mean_circularity,
            'morphology_score': self.morphology_score,
            'ki67_index': self.ki67_index,
            'image_path': self.image_path
        }


@dataclass
class CTDetectionResults:
    """CT tumor detection results"""
    tumor_detected: bool
    total_candidates: int
    high_conf_candidates: int
    max_confidence: float
    tumor_size_mm: Optional[float] = None
    tumor_location: Optional[str] = None
    tnm_stage: Optional[str] = None  # e.g., "T2N1M0"
    
    def to_dict(self) -> Dict:
        return {
            'tumor_detected': self.tumor_detected,
            'total_candidates': self.total_candidates,
            'high_conf_candidates': self.high_conf_candidates,
            'max_confidence': self.max_confidence,
            'tumor_size_mm': self.tumor_size_mm,
            'tumor_location': self.tumor_location,
            'tnm_stage': self.tnm_stage
        }


@dataclass
class ClinicalData:
    """Patient clinical data"""
    patient_id: str
    age: int
    gender: str
    # Genomic data
    kras_status: Optional[str] = None  # "Wild-type" or "Mutant"
    tp53_status: Optional[str] = None
    msi_status: Optional[str] = None  # "MSI-High", "MSI-Low", "MSS"
    # Lab results
    liver_function: Optional[str] = None  # "Normal", "Impaired"
    kidney_function: Optional[str] = None
    ecog_performance: Optional[int] = None  # 0-4
    comorbidities: Optional[List[str]] = None
    
    def to_dict(self) -> Dict:
        return {
            'patient_id': self.patient_id,
            'age': self.age,
            'gender': self.gender,
            'kras_status': self.kras_status,
            'tp53_status': self.tp53_status,
            'msi_status': self.msi_status,
            'liver_function': self.liver_function,
            'kidney_function': self.kidney_function,
            'ecog_performance': self.ecog_performance,
            'comorbidities': self.comorbidities or []
        }


@dataclass
class TherapyRecommendation:
    """AI therapy recommendation"""
    therapy_name: str
    drug_combination: List[str]
    predicted_efficacy: float  # 0-1  (literature-based estimate, not patient-specific)
    confidence: Optional[float]  # None = rule-based; set only when a real model provides it
    side_effect_risk: str  # "Low", "Moderate", "High"
    recommendation_basis: str = "rule_based"  # "rule_based" | "model"
    duration_weeks: Optional[int] = None
    contraindications: Optional[List[str]] = None
    
    def to_dict(self) -> Dict:
        return {
            'therapy_name': self.therapy_name,
            'drug_combination': self.drug_combination,
            'predicted_efficacy': self.predicted_efficacy,
            'confidence': self.confidence,
            'recommendation_basis': self.recommendation_basis,
            'side_effect_risk': self.side_effect_risk,
            'duration_weeks': self.duration_weeks,
            'contraindications': self.contraindications or []
        }


@dataclass
class IntegratedPatientProfile:
    """Complete integrated patient analysis"""
    patient_id: str
    timestamp: datetime
    
    # Input data
    cellpose_results: CellposeResults
    ct_results: CTDetectionResults
    clinical_data: ClinicalData
    
    # Integrated analysis
    cancer_stage: str  # e.g., "IIB"
    risk_level: str  # "Low", "Medium", "Medium-High", "High"
    prognosis_5yr_survival: float  # 0-1
    
    # AI recommendations
    recommended_therapies: List[TherapyRecommendation]
    
    # Medical interpretation
    doctor_interpretation: Optional[str] = None
    patient_interpretation: Optional[str] = None
    
    def to_dict(self) -> Dict:
        return {
            'patient_id': self.patient_id,
            'timestamp': self.timestamp.isoformat(),
            'cellpose_results': self.cellpose_results.to_dict(),
            'ct_results': self.ct_results.to_dict(),
            'clinical_data': self.clinical_data.to_dict(),
            'cancer_stage': self.cancer_stage,
            'risk_level': self.risk_level,
            'prognosis_5yr_survival': self.prognosis_5yr_survival,
            'recommended_therapies': [t.to_dict() for t in self.recommended_therapies],
            'doctor_interpretation': self.doctor_interpretation,
            'patient_interpretation': self.patient_interpretation
        }


class CDSSIntegrationEngine:
    """
    ADDS Clinical Decision Support System - Integration Engine
    
    Integrates multi-modal data and provides AI-driven recommendations
    """
    
    def __init__(self, openai_client=None):
        """
        Initialize CDSS Integration Engine
        
        Args:
            openai_client: Optional OpenAI client for medical interpretation
        """
        self.openai_client = openai_client
        
    def determine_cancer_stage(self, 
                               ct_results: CTDetectionResults,
                               cellpose_results: CellposeResults) -> str:
        """
        Determine cancer stage from integrated data
        
        Args:
            ct_results: CT detection results with TNM staging
            cellpose_results: Cell analysis results
        
        Returns:
            Cancer stage string (e.g., "IIB")
        """
        # Use TNM stage if available
        tnm = ct_results.tnm_stage
        
        if tnm:
            # Map TNM to stage
            if tnm.startswith('T1'):
                return "I"
            elif tnm.startswith('T2'):
                if 'N0' in tnm:
                    return "IIA"
                elif 'N1' in tnm:
                    return "IIB"
                else:
                    return "IIC"
            elif tnm.startswith('T3'):
                if 'N0' in tnm:
                    return "IIA"
                elif 'N1' in tnm or 'N2' in tnm:
                    return "IIIB"
                else:
                    return "IIIC"
            elif tnm.startswith('T4'):
                return "IIIC" if 'M0' in tnm else "IV"
            elif 'M1' in tnm:
                return "IV"
        
        # Fallback: estimate from tumor size
        if ct_results.tumor_size_mm:
            if ct_results.tumor_size_mm < 10:
                return "I"
            elif ct_results.tumor_size_mm < 20:
                return "IIA"
            else:
                return "IIB"
        
        return "Unknown"
    
    def calculate_risk_level(self,
                            cancer_stage: str,
                            cellpose_results: CellposeResults,
                            clinical_data: ClinicalData) -> str:
        """
        Calculate patient risk level
        
        Returns:
            "Low", "Medium", "Medium-High", or "High"
        """
        risk_score = 0
        
        # Stage contribution
        stage_risk = {
            "I": 0,
            "IIA": 1,
            "IIB": 2,
            "IIC": 3,
            "IIIA": 3,
            "IIIB": 4,
            "IIIC": 5,
            "IV": 6
        }
        risk_score += stage_risk.get(cancer_stage, 2)
        
        # Cell proliferation (Ki-67)
        if cellpose_results.ki67_index > 0.4:  # High proliferation
            risk_score += 2
        elif cellpose_results.ki67_index > 0.2:
            risk_score += 1
        
        # Age
        if clinical_data.age > 70:
            risk_score += 1
        
        # ECOG performance status
        if clinical_data.ecog_performance and clinical_data.ecog_performance >= 2:
            risk_score += 1
        
        # Comorbidities
        if clinical_data.comorbidities and len(clinical_data.comorbidities) >= 2:
            risk_score += 1
        
        # Map score to risk level
        if risk_score <= 2:
            return "Low"
        elif risk_score <= 4:
            return "Medium"
        elif risk_score <= 6:
            return "Medium-High"
        else:
            return "High"
    
    def estimate_prognosis(self, cancer_stage: str, risk_level: str) -> float:
        """
        Estimate 5-year survival rate
        
        Returns:
            Survival probability (0-1)
        """
        # Base survival by stage (simplified)
        stage_survival = {
            "I": 0.92,
            "IIA": 0.87,
            "IIB": 0.78,
            "IIC": 0.73,
            "IIIA": 0.69,
            "IIIB": 0.65,
            "IIIC": 0.58,
            "IV": 0.14
        }
        
        base_survival = stage_survival.get(cancer_stage, 0.65)
        
        # Adjust by risk level
        risk_adjustment = {
            "Low": 1.0,
            "Medium": 0.95,
            "Medium-High": 0.90,
            "High": 0.85
        }
        
        return base_survival * risk_adjustment.get(risk_level, 0.9)
    
    def select_therapy(self,
                      cancer_stage: str,
                      clinical_data: ClinicalData,
                      ct_results: CTDetectionResults) -> List[TherapyRecommendation]:
        """
        AI-based therapy selection
        
        Returns:
            List of recommended therapies, ranked by confidence
        """
        therapies = []
        
        # FOLFOX - Standard first-line
        folfox_contraindicated = (
            clinical_data.liver_function == "Impaired" or
            clinical_data.kidney_function == "Impaired"
        )
        
        if not folfox_contraindicated:
            therapies.append(TherapyRecommendation(
                therapy_name="FOLFOX Protocol",
                drug_combination=["5-Fluorouracil", "Leucovorin", "Oxaliplatin"],
                predicted_efficacy=0.78,
                confidence=None,             # rule-based; not from a trained model
                recommendation_basis="rule_based",
                side_effect_risk="Moderate",
                duration_weeks=24,
                contraindications=["Severe neuropathy", "Liver dysfunction"]
            ))
        
        # CAPOX + Bevacizumab
        therapies.append(TherapyRecommendation(
            therapy_name="CAPOX + Bevacizumab",
            drug_combination=["Capecitabine", "Oxaliplatin", "Bevacizumab"],
            predicted_efficacy=0.82,
            confidence=None,                 # rule-based; not from a trained model
            recommendation_basis="rule_based",
            side_effect_risk="Moderate-High",
            duration_weeks=24,
            contraindications=["Recent surgery", "Bleeding risk"]
        ))
        
        # Immunotherapy (only if MSI-High)
        if clinical_data.msi_status == "MSI-High":
            therapies.append(TherapyRecommendation(
                therapy_name="Immunotherapy Combination",
                drug_combination=["Pembrolizumab", "Chemotherapy backbone"],
                predicted_efficacy=0.85,
                confidence=None,             # rule-based; not from a trained model
                recommendation_basis="rule_based",
                side_effect_risk="Variable",
                duration_weeks=48,
                contraindications=["Autoimmune disease"]
            ))
        
        # Sort by confidence
        therapies.sort(key=lambda t: t.confidence, reverse=True)
        
        return therapies
    
    def generate_medical_interpretation(self,
                                       profile: IntegratedPatientProfile,
                                       for_patient: bool = False) -> Optional[str]:
        """
        Generate medical interpretation using OpenAI
        
        Args:
            profile: Integrated patient profile
            for_patient: If True, generate patient-friendly text
        
        Returns:
            Medical interpretation text or None if OpenAI unavailable
        """
        if not self.openai_client:
            return None
        
        try:
            if for_patient:
                # Patient-friendly explanation
                ki67_patient_str = (
                    f"증식률 {profile.cellpose_results.ki67_index*100:.0f}%"
                    if profile.cellpose_results.ki67_index is not None
                    else "증식률: 별도 IHC 검사 필요 (이미지 분석만으로 측정 불가)"
                )
                prompt = f"""
다음 의료 분석 결과를 환자가 이해하기 쉽게 한국어로 설명하세요:

세포 검사: 세포 수 {profile.cellpose_results.cell_count}개, {ki67_patient_str}
CT 검사: 종양 {'발견' if profile.ct_results.tumor_detected else '발견되지 않음'}
암 병기: {profile.cancer_stage}
위험도: {profile.risk_level}
5년 생존율: {profile.prognosis_5yr_survival*100:.0f}%

추천 치료 (NCCN 가이드라인 기반 규칙 추천): {profile.recommended_therapies[0].therapy_name if profile.recommended_therapies else '없음'}
(⚠️ 이 추천은 AI 학습 모델 예측이 아닌 임상 규칙 기반입니다)

다음 형식으로 작성:
1. 검사 결과가 의미하는 것 (2-3문장)
2. 암 병기 설명 (쉬운 용어)
3. 치료 계획 설명 (부작용 포함)
4. 긍정적인 요인 강조


전문 용어를 피하고, 희망적이고 명확한 톤으로 작성하세요.
"""
            else:
                # Doctor professional interpretation
                ki67_str = (
                    f"Ki-67 index: {profile.cellpose_results.ki67_index*100:.0f}%"
                    if profile.cellpose_results.ki67_index is not None
                    else "Ki-67 index: not available (IHC required)"
                )
                prompt = f"""
다음 통합 분석 결과에 대한 전문적인 임상 해석을 한국어로 작성하세요.
⚠️ 주의: 이 추천은 NCCN 가이드라인 기반 규칙 추체이기 때문에,
'AI 학습 데이터 규모' 또는 '학습된 파일 수'를 근거로 제시하지 마세요.

Cellpose 세포 분석:
- Cell count: {profile.cellpose_results.cell_count}
- {ki67_str}
- Morphology score: {profile.cellpose_results.morphology_score}/10

CT 종양 검출:
- Tumor detected: {profile.ct_results.tumor_detected}
- TNM stage: {profile.ct_results.tnm_stage}
- Max confidence: {profile.ct_results.max_confidence*100:.0f}%

통합 분석:
- Cancer stage: {profile.cancer_stage}
- Risk level: {profile.risk_level}
- 5-year survival: {profile.prognosis_5yr_survival*100:.0f}%

AI 추천 치료 (규칙 기반): {profile.recommended_therapies[0].therapy_name if profile.recommended_therapies else 'N/A'}

다음을 포함한 임상 해석을 작성하세요:
1. 종양 생물학적 특성 평가
2. 병기 및 예후 인자 분석
3. 치료 선택 근거
4. 추가 검사 필요성
5. 다학제 논의 포인트

NCCN 가이드라인을 참고하여 전문적으로 작성하세요.
"""
            
            response = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are an expert oncologist providing clinical interpretations."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=800,
                temperature=0.4
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            print(f"OpenAI interpretation error: {e}")
            return None
    
    def integrate_patient_data(self,
                              cellpose_results: CellposeResults,
                              ct_results: CTDetectionResults,
                              clinical_data: ClinicalData) -> IntegratedPatientProfile:
        """
        Main integration function - combines all data into patient profile
        
        Args:
            cellpose_results: Cell analysis results
            ct_results: CT tumor detection results
            clinical_data: Clinical and genomic data
        
        Returns:
            Complete integrated patient profile with recommendations
        """
        # Determine cancer stage
        cancer_stage = self.determine_cancer_stage(ct_results, cellpose_results)
        
        # Calculate risk level
        risk_level = self.calculate_risk_level(cancer_stage, cellpose_results, clinical_data)
        
        # Estimate prognosis
        prognosis = self.estimate_prognosis(cancer_stage, risk_level)
        
        # AI therapy selection
        therapies = self.select_therapy(cancer_stage, clinical_data, ct_results)
        
        # Create integrated profile
        profile = IntegratedPatientProfile(
            patient_id=clinical_data.patient_id,
            timestamp=datetime.now(),
            cellpose_results=cellpose_results,
            ct_results=ct_results,
            clinical_data=clinical_data,
            cancer_stage=cancer_stage,
            risk_level=risk_level,
            prognosis_5yr_survival=prognosis,
            recommended_therapies=therapies
        )
        
        # Generate medical interpretations
        profile.doctor_interpretation = self.generate_medical_interpretation(profile, for_patient=False)
        profile.patient_interpretation = self.generate_medical_interpretation(profile, for_patient=True)
        
        return profile


# Example usage
if __name__ == "__main__":
    # Mock data for testing
    cellpose = CellposeResults(
        cell_count=2450,
        mean_area_um2=185.0,
        mean_circularity=0.78,
        morphology_score=9.1,
        ki67_index=0.45
    )
    
    ct = CTDetectionResults(
        tumor_detected=True,
        total_candidates=33,
        high_conf_candidates=7,
        max_confidence=0.963,
        tumor_size_mm=15.2,
        tumor_location="Sigmoid colon",
        tnm_stage="T2N1M0"
    )
    
    clinical = ClinicalData(
        patient_id="P12345",
        age=58,
        gender="M",
        kras_status="Wild-type",
        msi_status="MSS",
        liver_function="Normal",
        kidney_function="Normal",
        ecog_performance=0
    )
    
    # Integrate
    engine = CDSSIntegrationEngine()
    profile = engine.integrate_patient_data(cellpose, ct, clinical)
    
    print("=== CDSS Integration Result ===")
    print(f"Cancer Stage: {profile.cancer_stage}")
    print(f"Risk Level: {profile.risk_level}")
    print(f"5-year Survival: {profile.prognosis_5yr_survival*100:.1f}%")
    print(f"\nRecommended Therapy: {profile.recommended_therapies[0].therapy_name}")
    print(f"Confidence: {profile.recommended_therapies[0].confidence*100:.0f}%")
