"""
Prognosis Prediction Model
Predicts patient survival rates and risk stratification
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from enum import Enum


class RiskGroup(Enum):
    """Patient risk stratification"""
    LOW = "Low"
    INTERMEDIATE = "Intermediate"
    HIGH = "High"


@dataclass
class PrognosisResult:
    """Prognosis prediction results"""
    survival_1yr: float  # 1-year survival probability
    survival_3yr: float  # 3-year survival probability
    survival_5yr: float  # 5-year survival probability
    risk_group: RiskGroup
    recurrence_risk: float  # Probability of recurrence
    metastasis_risk: float  # Probability of metastasis
    risk_factors: List[str]  # List of identified risk factors
    confidence: float  # Model confidence


class PrognosisPredictor:
    """
    Survival and prognosis prediction using deep learning
    
    Based on DeepSurv architecture (Cox proportional hazards)
    Predicts 1yr/3yr/5yr survival rates and risk stratification
    
    Note: This is a framework. Replace with trained model for production.
    """
    
    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize prognosis predictor
        
        Args:
            model_path: Path to trained DeepSurv model (optional)
        """
        self.model_path = model_path
        self.model = None
        
        if model_path:
            try:
                # Load trained model
                import torch
                self.model = torch.load(model_path)
                print(f"[PrognosisPredictor] Loaded model: {model_path}")
            except Exception as e:
                print(f"[PrognosisPredictor] Could not load model: {e}")
                print("[PrognosisPredictor] Using rule-based fallback")
    
    def predict(
        self,
        tnm_stage: str,
        tumor_size_mm: float,
        histology: Optional[str] = None,
        differentiation: Optional[str] = None,
        lymphovascular_invasion: bool = False,
        perineural_invasion: bool = False,
        lymph_nodes_positive: int = 0,
        lymph_nodes_examined: int = 0,
        age: int = 65,
        kras_status: str = "Unknown",
        tp53_status: str = "Unknown",
        performance_status: int = 1
    ) -> PrognosisResult:
        """
        Predict patient prognosis
        
        Args:
            tnm_stage: TNM staging (e.g., "T2N1M0")
            tumor_size_mm: Tumor size in mm
            histology: Histological type
            differentiation: Tumor differentiation
            lymphovascular_invasion: LVI present
            perineural_invasion: PNI present
            lymph_nodes_positive: Number of positive lymph nodes
            lymph_nodes_examined: Total lymph nodes examined
            age: Patient age
            kras_status: KRAS mutation status
            tp53_status: TP53 mutation status
            performance_status: ECOG performance status (0-4)
        
        Returns:
            PrognosisResult with survival predictions
        """
        if self.model:
            # Use trained deep learning model
            return self._predict_with_model(
                tnm_stage, tumor_size_mm, histology, differentiation,
                lymphovascular_invasion, perineural_invasion,
                lymph_nodes_positive, lymph_nodes_examined,
                age, kras_status, tp53_status, performance_status
            )
        else:
            # Use rule-based approach (fallback)
            return self._predict_rule_based(
                tnm_stage, tumor_size_mm, histology, differentiation,
                lymphovascular_invasion, perineural_invasion,
                lymph_nodes_positive, lymph_nodes_examined,
                age, kras_status, tp53_status, performance_status
            )
    
    def _predict_with_model(self, *args, **kwargs) -> PrognosisResult:
        """Predict using trained deep learning model"""
        # TODO: Implement when model is available
        # For now, fallback to rule-based
        return self._predict_rule_based(*args, **kwargs)
    
    def _predict_rule_based(
        self,
        tnm_stage: str,
        tumor_size_mm: float,
        histology: Optional[str],
        differentiation: Optional[str],
        lymphovascular_invasion: bool,
        perineural_invasion: bool,
        lymph_nodes_positive: int,
        lymph_nodes_examined: int,
        age: int,
        kras_status: str,
        tp53_status: str,
        performance_status: int
    ) -> PrognosisResult:
        """
        Rule-based prognosis prediction
        Based on clinical guidelines (NCCN, AJCC)
        """
        # Extract T, N, M from stage
        t_stage = self._extract_t_stage(tnm_stage)
        n_stage = self._extract_n_stage(tnm_stage)
        m_stage = self._extract_m_stage(tnm_stage)
        
        # Calculate base survival rates
        base_5yr = self._get_base_survival(t_stage, n_stage, m_stage)
        
        # Risk factors
        risk_factors = []
        risk_score = 0
        
        # T stage risk
        if t_stage >= 3:
            risk_factors.append("Advanced T stage (T3/T4)")
            risk_score += 2
        
        # N stage risk
        if n_stage >= 1:
            risk_factors.append(f"Lymph node metastasis (N{n_stage})")
            risk_score += n_stage * 2
        
        # M stage risk
        if m_stage >= 1:
            risk_factors.append("Distant metastasis")
            risk_score += 5
        
        # Lymph node ratio
        if lymph_nodes_examined > 0:
            ln_ratio = lymph_nodes_positive / lymph_nodes_examined
            if ln_ratio > 0.2:
                risk_factors.append(f"High lymph node ratio ({ln_ratio:.1%})")
                risk_score += 2
        
        # Tumor size
        if tumor_size_mm > 50:
            risk_factors.append(f"Large tumor size ({tumor_size_mm:.0f}mm)")
            risk_score += 1
        
        # Vascular invasion
        if lymphovascular_invasion:
            risk_factors.append("Lymphovascular invasion")
            risk_score += 1
        
        if perineural_invasion:
            risk_factors.append("Perineural invasion")
            risk_score += 1
        
        # Differentiation
        if differentiation in ["Poorly differentiated", "Undifferentiated"]:
            risk_factors.append(f"Poor differentiation")
            risk_score += 2
        
        # Molecular markers
        if kras_status == "Mutant":
            risk_factors.append("KRAS mutation")
            risk_score += 1
        
        if tp53_status == "Mutant":
            risk_factors.append("TP53 mutation")
            risk_score += 1
        
        # Age
        if age > 75:
            risk_factors.append(f"Advanced age ({age})")
            risk_score += 1
        
        # Performance status
        if performance_status >= 2:
            risk_factors.append(f"Poor performance status (ECOG {performance_status})")
            risk_score += 2
        
        # Adjust survival based on risk score
        adjustment = risk_score * 0.05  # 5% per risk point
        survival_5yr = max(0.1, min(0.95, base_5yr - adjustment))
        survival_3yr = survival_5yr + 0.1  # Typically higher
        survival_1yr = survival_5yr + 0.15  # Even higher
        
        # Ensure logical progression
        survival_1yr = min(0.98, survival_1yr)
        survival_3yr = min(survival_1yr - 0.05, survival_3yr)
        survival_5yr = min(survival_3yr - 0.05, survival_5yr)
        
        # Risk stratification
        if risk_score <= 3:
            risk_group = RiskGroup.LOW
        elif risk_score <= 7:
            risk_group = RiskGroup.INTERMEDIATE
        else:
            risk_group = RiskGroup.HIGH
        
        # Recurrence and metastasis risk
        recurrence_risk = min(0.9, risk_score * 0.08)
        metastasis_risk = min(0.8, risk_score * 0.06)
        
        # If already has metastasis
        if m_stage >= 1:
            metastasis_risk = 1.0
        
        return PrognosisResult(
            survival_1yr=survival_1yr,
            survival_3yr=survival_3yr,
            survival_5yr=survival_5yr,
            risk_group=risk_group,
            recurrence_risk=recurrence_risk,
            metastasis_risk=metastasis_risk,
            risk_factors=risk_factors,
            confidence=0.75  # Rule-based has lower confidence
        )
    
    def _extract_t_stage(self, tnm: str) -> int:
        """Extract T stage from TNM string"""
        try:
            t_part = tnm.split('N')[0].replace('T', '')
            if 'is' in t_part.lower():
                return 0  # Tis
            return int(t_part[0])
        except:
            return 2  # Default
    
    def _extract_n_stage(self, tnm: str) -> int:
        """Extract N stage from TNM string"""
        try:
            n_part = tnm.split('N')[1].split('M')[0]
            return int(n_part[0])
        except:
            return 0  # Default
    
    def _extract_m_stage(self, tnm: str) -> int:
        """Extract M stage from TNM string"""
        try:
            m_part = tnm.split('M')[1]
            return int(m_part[0])
        except:
            return 0  # Default
    
    def _get_base_survival(self, t: int, n: int, m: int) -> float:
        """
        Get base 5-year survival rate from stage
        Based on SEER/AJCC statistics for colorectal cancer
        """
        if m >= 1:
            # Stage IV (metastatic)
            return 0.14
        elif n >= 2:
            # Stage III (N2)
            return 0.53
        elif n >= 1:
            # Stage III (N1)
            return 0.64
        elif t >= 3:
            # Stage II (T3/T4, N0)
            return 0.87
        else:
            # Stage I (T1/T2, N0)
            return 0.92
