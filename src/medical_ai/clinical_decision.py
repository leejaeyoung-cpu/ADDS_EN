"""
Clinical Decision Engine
Generates treatment plans and clinical recommendations
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Optional
from enum import Enum


class TreatmentPhaseType(Enum):
    """Type of treatment phase"""
    SURGERY = "surgery"
    CHEMOTHERAPY = "chemotherapy"
    RADIATION = "radiation"
    TARGETED_THERAPY = "targeted_therapy"
    IMMUNOTHERAPY = "immunotherapy"
    MONITORING = "monitoring"


@dataclass
class TreatmentPhase:
    """Single phase of treatment"""
    name: str
    type: TreatmentPhaseType
    duration: str
    goal: str
    procedure: Optional[str] = None
    regimen: Optional[str] = None
    response_rate: Optional[float] = None
    drugs: Optional[List[Dict]] = None
    rationale: str = ""
    start_week: int = 0
    end_week: int = 0


@dataclass
class MonitoringProtocol:
    """Follow-up monitoring protocol"""
    frequency: str
    items: List[str]
    schedule: List[Dict]


@dataclass
class TreatmentPlan:
    """Complete treatment plan"""
    phases: List[TreatmentPhase]
    monitoring: MonitoringProtocol
    expected_duration_weeks: int
    success_probability: float
    rationale: str


class ClinicalDecisionEngine:
    """
    AI-powered clinical decision support
    Generates evidence-based treatment plans based on:
    - TNM staging
    - Patient biomarkers
    - Performance status
    - Predicted treatment response
    """
    
    def __init__(self):
        """Initialize clinical decision engine"""
        # Treatment regimens database
        self.chemotherapy_regimens = {
            'FOLFOX': {
                'drugs': [
                    {'name': 'Oxaliplatin', 'dose': '85 mg/m²'},
                    {'name': '5-Fluorouracil', 'dose': '400 mg/m² IV + 2400 mg/m² 46h'},
                    {'name': 'Leucovorin', 'dose': '400 mg/m²'}
                ],
                'response_rate': 0.50,
                'indication': 'Stage III/IV colon cancer'
            },
            'XELOX': {
                'drugs': [
                    {'name': 'Capecitabine', 'dose': '1000 mg/m² BID'},
                    {'name': 'Oxaliplatin', 'dose': '130 mg/m²'}
                ],
                'response_rate': 0.48,
                'indication': 'Stage III/IV, oral alternative'
            },
            'FOLFIRI': {
                'drugs': [
                    {'name': 'Irinotecan', 'dose': '180 mg/m²'},
                    {'name': '5-Fluorouracil', 'dose': '400 mg/m² IV + 2400 mg/m² 46h'},
                    {'name': 'Leucovorin', 'dose': '400 mg/m²'}
                ],
                'response_rate': 0.43,
                'indication': 'Second-line or alternative'
            }
        }
    
    def recommend_treatment(
        self,
        tnm_stage: str,
        prognosis: Dict,
        patient_profile: Dict,
        predicted_response: Optional[Dict] = None
    ) -> TreatmentPlan:
        """
        Generate comprehensive treatment plan
        
        Args:
            tnm_stage: TNM staging
            prognosis: Prognosis prediction results
            patient_profile: Patient clinical data
            predicted_response: Treatment response predictions
        
        Returns:
            Complete TreatmentPlan
        """
        # Extract stage components
        t_stage = self._extract_t_stage(tnm_stage)
        n_stage = self._extract_n_stage(tnm_stage)
        m_stage = self._extract_m_stage(tnm_stage)
        
        phases = []
        current_week = 0
        
        # Determine overall stage
        if m_stage >= 1:
            overall_stage = "IV"
        elif n_stage >= 1:
            overall_stage = "III"
        elif t_stage >= 3:
            overall_stage = "II"
        else:
            overall_stage = "I"
        
        # ========== Phase 1: Surgery ==========
        if m_stage == 0:  # No distant metastasis
            surgery_phase = self._recommend_surgery(
                t_stage, n_stage, overall_stage, current_week
            )
            phases.append(surgery_phase)
            current_week = surgery_phase.end_week
        
        # ========== Phase 2: Adjuvant Chemotherapy ==========
        if overall_stage in ["II", "III", "IV"]:
            chemo_phase = self._recommend_chemotherapy(
                overall_stage, patient_profile, predicted_response, current_week
            )
            if chemo_phase:
                phases.append(chemo_phase)
                current_week = chemo_phase.end_week
        
        # ========== Phase 3: Targeted/Immunotherapy ==========
        if overall_stage in ["III", "IV"]:
            targeted_phase = self._recommend_targeted_therapy(
                patient_profile, current_week
            )
            if targeted_phase:
                phases.append(targeted_phase)
                current_week = targeted_phase.end_week
        
        # ========== Monitoring Protocol ==========
        monitoring = self._generate_monitoring_protocol(
            overall_stage, prognosis.get('risk_group', 'Intermediate')
        )
        
        # Calculate success probability
        success_prob = self._calculate_success_probability(
            overall_stage, prognosis, patient_profile
        )
        
        # Generate rationale
        rationale = self._generate_rationale(
            overall_stage, tnm_stage, patient_profile, phases
        )
        
        return TreatmentPlan(
            phases=phases,
            monitoring=monitoring,
            expected_duration_weeks=current_week,
            success_probability=success_prob,
            rationale=rationale
        )
    
    def _recommend_surgery(
        self, t: int, n: int, overall_stage: str, start_week: int
    ) -> TreatmentPhase:
        """Recommend surgical intervention"""
        if overall_stage == "I":
            procedure = "Local excision or polypectomy"
            duration = "1-2 weeks recovery"
            goal = "Complete tumor removal"
        elif t <= 2:
            procedure = "Right or left hemicolectomy"
            duration = "4-6 weeks recovery"
            goal = "Remove primary tumor and regional lymph nodes"
        else:
            procedure = "Extended colectomy with lymphadenectomy"
            duration = "6-8 weeks recovery"
            goal = "En bloc resection of tumor and involved structures"
        
        return TreatmentPhase(
            name="Surgical Resection",
            type=TreatmentPhaseType.SURGERY,
            duration=duration,
            goal=goal,
            procedure=procedure,
            rationale=f"Standard of care for Stage {overall_stage} colon cancer",
            start_week=start_week,
            end_week=start_week + 6
        )
    
    def _recommend_chemotherapy(
        self,
        stage: str,
        patient_profile: Dict,
        predicted_response: Optional[Dict],
        start_week: int
    ) -> Optional[TreatmentPhase]:
        """Recommend chemotherapy regimen"""
        if stage == "I":
            return None  # Usually not needed for Stage I
        
        # Select regimen based on patient factors
        age = patient_profile.get('age', 65)
        performance = patient_profile.get('performance_status', 1)
        
        if age > 75 or performance >= 2:
            # Less intensive regimen for elderly/poor performance
            regimen_name = "XELOX"
            duration = "18 weeks (6 cycles)"
        else:
            # Standard regimen
            regimen_name = "FOLFOX"
            duration = "24 weeks (12 cycles)"
        
        regimen = self.chemotherapy_regimens[regimen_name]
        
        # Adjust response rate based on biomarkers
        response_rate = regimen['response_rate']
        kras = patient_profile.get('kras_status', 'Unknown')
        if kras == "Wild-type":
            response_rate += 0.10  # Better response
        
        # Use predicted response if available
        if predicted_response and regimen_name in predicted_response:
            response_rate = predicted_response[regimen_name]
        
        weeks = 24 if regimen_name == "FOLFOX" else 18
        
        return TreatmentPhase(
            name=f"Adjuvant Chemotherapy ({regimen_name})",
            type=TreatmentPhaseType.CHEMOTHERAPY,
            duration=duration,
            goal="Eliminate micrometastases and reduce recurrence",
            regimen=regimen_name,
            response_rate=response_rate,
            drugs=regimen['drugs'],
            rationale=f"NCCN guidelines for Stage {stage} colon cancer",
            start_week=start_week,
            end_week=start_week + weeks
        )
    
    def _recommend_targeted_therapy(
        self, patient_profile: Dict, start_week: int
    ) -> Optional[TreatmentPhase]:
        """Recommend targeted therapy if appropriate"""
        kras = patient_profile.get('kras_status', 'Unknown')
        
        if kras == "Wild-type":
            # EGFR inhibitors for KRAS wild-type
            return TreatmentPhase(
                name="Targeted Therapy (Cetuximab)",
                type=TreatmentPhaseType.TARGETED_THERAPY,
                duration="12 weeks",
                goal="Target EGFR pathway",
                regimen="Cetuximab",
                response_rate=0.65,
                drugs=[{'name': 'Cetuximab', 'dose': '400 mg/m² loading, then 250 mg/m² weekly'}],
                rationale="KRAS wild-type indicates EGFR inhibitor sensitivity",
                start_week=start_week,
                end_week=start_week + 12
            )
        
        return None
    
    def _generate_monitoring_protocol(
        self, stage: str, risk_group: str
    ) -> MonitoringProtocol:
        """Generate follow-up monitoring schedule"""
        if stage in ["I", "II"] and risk_group == "Low":
            frequency = "Every 6 months for 2 years, then annually"
            schedule = [
                {'time': '6 months', 'tests': 'CEA, CT scan'},
                {'time': '12 months', 'tests': 'CEA, CT scan, Colonoscopy'},
                {'time': '24 months', 'tests': 'CEA, CT scan'},
                {'time': 'Annual', 'tests': 'CEA, Colonoscopy every 3-5 years'}
            ]
        else:
            frequency = "Every 3-6 months for 2 years, then every 6 months for 3 years"
            schedule = [
                {'time': '3 months', 'tests': 'CEA, CT chest/abdomen/pelvis'},
                {'time': '6 months', 'tests': 'CEA, CT scan'},
                {'time': '9 months', 'tests': 'CEA, CT scan'},
                {'time': '12 months', 'tests': 'CEA, CT scan, Colonoscopy'},
                {'time': '18 months', 'tests': 'CEA, CT scan'},
                {'time': '24 months', 'tests': 'CEA, CT scan, Colonoscopy'}
            ]
        
        items = [
            "CEA (carcinoembryonic antigen) tumor marker",
            "CT chest/abdomen/pelvis",
            "Colonoscopy",
            "Physical examination",
            "Symptom assessment"
        ]
        
        return MonitoringProtocol(
            frequency=frequency,
            items=items,
            schedule=schedule
        )
    
    def _calculate_success_probability(
        self, stage: str, prognosis: Dict, patient_profile: Dict
    ) -> float:
        """Calculate overall treatment success probability"""
        # Base on survival rate
        base_prob = prognosis.get('survival_5yr', 0.7)
        
        # Adjust for performance status
        perf = patient_profile.get('performance_status', 1)
        if perf == 0:
            base_prob += 0.05
        elif perf >= 2:
            base_prob -= 0.10
        
        # Adjust for age
        age = patient_profile.get('age', 65)
        if age < 60:
            base_prob += 0.05
        elif age > 75:
            base_prob -= 0.05
        
        return min(0.95, max(0.10, base_prob))
    
    def _generate_rationale(
        self, stage: str, tnm: str, profile: Dict, phases: List
    ) -> str:
        """Generate treatment plan rationale"""
        rationale_parts = [
            f"Treatment plan for Stage {stage} colon cancer ({tnm}).",
            f"Based on NCCN Clinical Practice Guidelines in Oncology.",
        ]
        
        if len(phases) > 0:
            phase_names = [p.name for p in phases]
            rationale_parts.append(f"Recommended approach: {' → '.join(phase_names)}.")
        
        kras = profile.get('kras_status', 'Unknown')
        if kras == "Wild-type":
            rationale_parts.append("KRAS wild-type status supports EGFR inhibitor use.")
        
        return " ".join(rationale_parts)
    
    def _extract_t_stage(self, tnm: str) -> int:
        """Extract T stage"""
        try:
            return int(tnm.split('N')[0].replace('T', '')[0])
        except:
            return 2
    
    def _extract_n_stage(self, tnm: str) -> int:
        """Extract N stage"""
        try:
            return int(tnm.split('N')[1].split('M')[0][0])
        except:
            return 0
    
    def _extract_m_stage(self, tnm: str) -> int:
        """Extract M stage"""
        try:
            return int(tnm.split('M')[1][0])
        except:
            return 0
