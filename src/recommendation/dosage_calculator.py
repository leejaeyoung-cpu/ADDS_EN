"""
Dosage calculator for personalized drug dosing
Calculates optimal dosage based on patient characteristics
"""

import numpy as np
from typing import Dict, Optional


class DosageCalculator:
    """용량 최적화 계산기 (IP Module 3)"""
    
    def __init__(self):
        # Standard dosage database (mg/m² or mg/kg)
        self.dosage_standards = {
            '5-FU': {'base': 400, 'unit': 'mg/m2', 'max': 600},
            'Leucovorin': {'base': 400, 'unit': 'mg/m2', 'max': 600},
            'Oxaliplatin': {'base': 85, 'unit': 'mg/m2', 'max': 130},
            'Irinotecan': {'base': 180, 'unit': 'mg/m2', 'max': 250},
            'Bevacizumab': {'base': 5, 'unit': 'mg/kg', 'max': 10},
            'Cetuximab': {'base': 400, 'unit': 'mg/m2', 'initial': True, 'maintenance': 250},
            'Panitumumab': {'base': 6, 'unit': 'mg/kg', 'max': 6},
            'Pembrolizumab': {'base': 200, 'unit': 'mg', 'fixed': True},
            'Capecitabine': {'base': 1000, 'unit': 'mg/m2', 'max': 1250, 'po': True}
        }
    
    def calculate_optimal_dosage(
        self,
        drug_name: str,
        patient_weight: float,
        patient_height: float,
        age: int,
        hepatic_function: str = 'Normal',
        renal_function: float = 90,  # eGFR
        ecog_score: int = 0,
        is_maintenance: bool = False
    ) -> Dict:
        """
        최적 용량 계산
        
        Args:
            drug_name: 약물명
            patient_weight: 체중 (kg)
            patient_height: 키 (cm)
            age: 나이
            hepatic_function: 간 기능 (Normal/Mild/Moderate/Severe)
            renal_function: 신장 기능 (eGFR, mL/min/1.73m²)
            ecog_score: ECOG 수행 상태 (0-4)
            is_maintenance: 유지 요법 여부
            
        Returns:
            용량 정보 딕셔너리
        """
        if drug_name not in self.dosage_standards:
            return {'error': f'Drug {drug_name} not in database'}
        
        standard = self.dosage_standards[drug_name]
        
        # 1. Calculate body surface area (BSA) if needed
        bsa = self._calculate_bsa(patient_weight, patient_height)
        
        # 2. Base dosage
        if standard.get('fixed'):
            base_dose = standard['base']
        elif standard['unit'] == 'mg/m2':
            base_dose = standard['base'] * bsa
        elif standard['unit'] == 'mg/kg':
            base_dose = standard['base'] * patient_weight
        elif standard['unit'] == 'mg' and standard.get('po'):
            base_dose = standard['base'] * bsa
        else:
            base_dose = standard['base']
        
        # Maintenance dose for certain drugs
        if is_maintenance and 'maintenance' in standard:
            if standard['unit'] == 'mg/m2':
                base_dose = standard['maintenance'] * bsa
            else:
                base_dose = standard['maintenance']
        
        # 3. Apply adjustment factors
        adjustments = {}
        adjusted_dose = base_dose
        
        # Age adjustment
        if age > 75:
            age_factor = 0.85  # 15% reduction
            adjusted_dose *= age_factor
            adjustments['age'] = age_factor
        elif age > 70:
            age_factor = 0.9  # 10% reduction
            adjusted_dose *= age_factor
            adjustments['age'] = age_factor
        
        # Hepatic adjustment
        hepatic_factor = self._get_hepatic_adjustment(hepatic_function, drug_name)
        if hepatic_factor < 1.0:
            adjusted_dose *= hepatic_factor
            adjustments['hepatic'] = hepatic_factor
        
        # Renal adjustment
        renal_factor = self._get_renal_adjustment(renal_function, drug_name)
        if renal_factor < 1.0:
            adjusted_dose *= renal_factor
            adjustments['renal'] = renal_factor
        
        # ECOG adjustment
        if ecog_score >= 2:
            ecog_factor = 0.8  # 20% reduction for poor performance status
            adjusted_dose *= ecog_factor
            adjustments['ecog'] = ecog_factor
        
        # 4. Cap at maximum
        max_dose = standard.get('max', float('inf'))
        if standard['unit'] == 'mg/m2':
            max_dose_absolute = max_dose * bsa
        elif standard['unit'] == 'mg/kg':
            max_dose_absolute = max_dose * patient_weight
        else:
            max_dose_absolute = max_dose
        
        final_dose = min(adjusted_dose, max_dose_absolute)
        capped = final_dose < adjusted_dose
        
        return {
            'drug_name': drug_name,
            'base_dose_mg': round(base_dose, 1),
            'adjusted_dose_mg': round(adjusted_dose, 1),
            'final_dose_mg': round(final_dose, 1),
            'dose_unit': standard['unit'],
            'bsa_m2': round(bsa, 2),
            'adjustments': adjustments,
            'capped_at_max': capped,
            'administration_route': 'PO' if standard.get('po') else 'IV',
            'is_maintenance': is_maintenance
        }
    
    def _calculate_bsa(self, weight_kg: float, height_cm: float) -> float:
        """
        체표면적 계산 (DuBois formula)
        BSA (m²) = 0.007184 × W^0.425 × H^0.725
        """
        bsa = 0.007184 * (weight_kg ** 0.425) * (height_cm ** 0.725)
        return bsa
    
    def _get_hepatic_adjustment(self, function: str, drug: str) -> float:
        """간 기능 기반 용량 조정"""
        # Drugs primarily metabolized by liver need adjustment
        hepatotoxic_drugs = ['Irinotecan', 'Oxaliplatin', 'Capecitabine']
        
        if drug not in hepatotoxic_drugs:
            return 1.0
        
        adjustment_map = {
            'Normal': 1.0,
            'Mild': 0.9,
            'Moderate': 0.75,
            'Severe': 0.5
        }
        
        return adjustment_map.get(function, 1.0)
    
    def _get_renal_adjustment(self, egfr: float, drug: str) -> float:
        """신장 기능 기반 용량 조정"""
        # Drugs primarily excreted by kidney
        nephrotoxic_drugs = ['Oxaliplatin', 'Capecitabine', 'Carboplatin']
        
        if drug not in nephrotoxic_drugs:
            return 1.0
        
        if egfr >= 60:
            return 1.0
        elif egfr >= 30:
            return 0.75  # 25% reduction
        elif egfr >= 15:
            return 0.5   # 50% reduction
        else:
            return 0.25  # Severe reduction or contraindicated
    
    def calculate_regimen_dosages(
        self,
        regimen: Dict,
        patient_profile: Dict
    ) -> Dict:
        """
        전체 조합의 용량 계산
        
        Args:
            regimen: 약물 조합 정보
            patient_profile: 환자 프로파일
            
        Returns:
            각 약물별 용량 정보
        """
        dosage_plan = {
            'drugs': [],
            'cycle_number': 1,
            'total_drugs': len(regimen['drugs'])
        }
        
        for drug_name in regimen['drugs']:
            dosage = self.calculate_optimal_dosage(
                drug_name=drug_name,
                patient_weight=patient_profile.get('weight', 70),
                patient_height=patient_profile.get('height', 170),
                age=patient_profile.get('age', 60),
                hepatic_function=patient_profile.get('hepatic_function', 'Normal'),
                renal_function=patient_profile.get('egfr', 90),
                ecog_score=patient_profile.get('ecog_score', 0)
            )
            
            dosage_plan['drugs'].append(dosage)
        
        return dosage_plan
