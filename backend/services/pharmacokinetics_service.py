"""
Pharmacokinetics Service
Proper compartmental PK modeling for anticancer drug optimization

Models:
- 1-compartment: Simple IV bolus / oral
- 2-compartment: Distribution + elimination phases
- Population PK adjustments for tumor burden and organ function
"""

import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from scipy.integrate import solve_ivp
from scipy.optimize import minimize_scalar
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


# ─── Drug-specific PK Parameters (literature-based) ──────────────────────

DRUG_PK_PARAMS = {
    "5-fluorouracil": {
        "mw": 130.08,
        "route": "iv_infusion",
        "typical_cl": 50.0,       # L/h (population typical)
        "typical_vd": 20.0,       # L (central volume)
        "typical_vd2": 15.0,      # L (peripheral volume)
        "typical_q": 12.0,        # L/h (intercompartmental clearance)
        "typical_dose": 400,      # mg/m² bolus, or 2400 mg/m² 46h infusion
        "therapeutic_range": (0.3, 5.0),  # µg/mL
        "hill_coefficient": 1.2,
        "ec50": 1.5,              # µg/mL
    },
    "oxaliplatin": {
        "mw": 397.29,
        "route": "iv_infusion",
        "typical_cl": 17.4,       # L/h
        "typical_vd": 440.0,      # L (central)
        "typical_vd2": 150.0,     # L (peripheral)
        "typical_q": 30.0,        # L/h
        "typical_dose": 85,       # mg/m²
        "therapeutic_range": (0.1, 2.0),  # µg/mL
        "hill_coefficient": 1.5,
        "ec50": 0.8,
    },
    "irinotecan": {
        "mw": 586.68,
        "route": "iv_infusion",
        "typical_cl": 33.0,       # L/h
        "typical_vd": 110.0,      # L
        "typical_vd2": 200.0,     # L
        "typical_q": 25.0,        # L/h
        "typical_dose": 180,      # mg/m²
        "therapeutic_range": (0.2, 3.0),
        "hill_coefficient": 1.3,
        "ec50": 1.0,
    },
    "bevacizumab": {
        "mw": 149000,
        "route": "iv_infusion",
        "typical_cl": 0.207,      # L/day → ~0.00863 L/h
        "typical_vd": 2.73,       # L (central)
        "typical_vd2": 1.53,      # L
        "typical_q": 0.244,       # L/day → ~0.01 L/h
        "typical_dose": 5,        # mg/kg
        "therapeutic_range": (50, 250),  # µg/mL
        "hill_coefficient": 1.0,
        "ec50": 100,
    },
    "cetuximab": {
        "mw": 152000,
        "route": "iv_infusion",
        "typical_cl": 0.022,      # L/h
        "typical_vd": 3.28,       # L
        "typical_vd2": 2.0,       # L
        "typical_q": 0.08,        # L/h
        "typical_dose": 400,      # mg/m² initial, 250 mg/m² weekly
        "therapeutic_range": (40, 200),
        "hill_coefficient": 1.1,
        "ec50": 80,
    },
    "pembrolizumab": {
        "mw": 149000,
        "route": "iv_infusion",
        "typical_cl": 0.202,      # L/day
        "typical_vd": 6.0,        # L
        "typical_vd2": 3.43,      # L
        "typical_q": 0.70,        # L/day
        "typical_dose": 200,      # mg flat dose or 2 mg/kg
        "therapeutic_range": (10, 100),
        "hill_coefficient": 1.0,
        "ec50": 25,
    },
    "default": {
        "mw": 500,
        "route": "iv_infusion",
        "typical_cl": 30.0,
        "typical_vd": 50.0,
        "typical_vd2": 30.0,
        "typical_q": 15.0,
        "typical_dose": 200,
        "therapeutic_range": (0.5, 5.0),
        "hill_coefficient": 1.0,
        "ec50": 2.0,
    }
}


@dataclass
class PKResult:
    """Pharmacokinetic analysis result"""
    drug_name: str
    clearance: float          # L/h
    volume_central: float     # L
    volume_peripheral: float  # L
    half_life_alpha: float    # hours (distribution)
    half_life_beta: float     # hours (elimination)
    half_life_effective: float  # hours
    auc: float                # µg·h/mL
    cmax: float               # µg/mL
    trough: float             # µg/mL (at end of dosing interval)
    optimal_dose: float       # mg/m²
    dosing_interval: int      # hours
    predicted_efficacy: float # 0-1
    toxicity_risk: str
    therapeutic_window: Tuple[float, float]
    concentration_profile: Optional[Dict] = None


class PharmacokineticEngine:
    """
    Compartmental PK engine for anticancer drug optimization
    
    Models:
    - 2-compartment IV infusion (default for chemo)
    - Population PK covariate adjustments
    - Dose optimization to target therapeutic window
    """
    
    def analyze(
        self,
        drug_name: str,
        tumor_volume_cm3: float = 0.0,
        ki67_index: float = 0.0,
        body_surface_area: float = 1.7,  # m²
        body_weight: float = 70.0,       # kg
        renal_function: float = 100.0,   # eGFR mL/min/1.73m²
        hepatic_function: str = "normal",  # normal, mild, moderate, severe
        age: float = 60.0,
        infusion_duration: float = 2.0,  # hours
        dose_mg_m2: Optional[float] = None
    ) -> PKResult:
        """
        Run 2-compartment PK analysis for a specific drug
        """
        # Get drug parameters
        drug_key = drug_name.lower()
        params = DRUG_PK_PARAMS.get(drug_key, DRUG_PK_PARAMS["default"])
        
        # Population PK covariate adjustments
        cl, v1, v2, q = self._adjust_covariates(
            params, tumor_volume_cm3, body_weight, body_surface_area,
            renal_function, hepatic_function, age
        )
        
        # Use specified or typical dose
        dose = dose_mg_m2 or params["typical_dose"]
        dose_mg = dose * body_surface_area  # Convert to absolute mg
        
        # Run 2-compartment model
        t_span = (0, 72)  # 72 hours simulation
        t_eval = np.linspace(0, 72, 721)  # Every 0.1 hour
        
        conc_profile = self._solve_two_compartment(
            dose_mg, cl, v1, v2, q, infusion_duration, t_eval
        )
        
        # Extract PK parameters
        cmax = float(np.max(conc_profile))
        
        # Half-lives from macro-constants
        alpha, beta = self._calculate_hybrid_constants(cl, v1, v2, q)
        t_half_alpha = 0.693 / alpha if alpha > 0 else 0.0
        t_half_beta = 0.693 / beta if beta > 0 else 0.0
        t_half_eff = t_half_beta  # Elimination phase dominates
        
        # AUC by trapezoidal rule
        auc = float(np.trapz(conc_profile, t_eval))
        
        # Optimal dosing interval: when concentration drops below Cmin
        therapeutic = params["therapeutic_range"]
        dosing_interval = self._calculate_dosing_interval(
            conc_profile, t_eval, therapeutic[0], t_half_eff
        )
        
        # Trough at dosing interval
        idx_trough = min(int(dosing_interval * 10), len(conc_profile) - 1)
        trough = float(conc_profile[idx_trough])
        
        # Dose optimization
        optimal_dose = self._optimize_dose(
            dose, cl, v1, v2, q, body_surface_area,
            infusion_duration, therapeutic, params
        )
        
        # Predicted efficacy (Emax model)
        hill = params["hill_coefficient"]
        ec50 = params["ec50"]
        avg_conc = auc / max(dosing_interval, 1)
        efficacy = (avg_conc ** hill) / (ec50 ** hill + avg_conc ** hill)
        
        # Ki-67 adjustment: higher proliferation may improve chemo response
        if ki67_index > 0:
            ki67_factor = 1.0 + 0.003 * min(ki67_index, 80)
            efficacy = min(0.95, efficacy * ki67_factor)
        
        # Toxicity assessment
        toxicity = self._assess_toxicity(cmax, therapeutic, cl, renal_function)
        
        # Concentration profile for visualization (downsample)
        step = max(1, len(t_eval) // 100)
        profile = {
            "time_hours": [float(t) for t in t_eval[::step]],
            "concentration_ug_ml": [float(c) for c in conc_profile[::step]],
            "therapeutic_min": therapeutic[0],
            "therapeutic_max": therapeutic[1]
        }
        
        return PKResult(
            drug_name=drug_name,
            clearance=round(cl, 3),
            volume_central=round(v1, 2),
            volume_peripheral=round(v2, 2),
            half_life_alpha=round(t_half_alpha, 2),
            half_life_beta=round(t_half_beta, 2),
            half_life_effective=round(t_half_eff, 2),
            auc=round(auc, 2),
            cmax=round(cmax, 3),
            trough=round(trough, 3),
            optimal_dose=round(optimal_dose, 1),
            dosing_interval=dosing_interval,
            predicted_efficacy=round(float(efficacy), 3),
            toxicity_risk=toxicity,
            therapeutic_window=therapeutic,
            concentration_profile=profile
        )
    
    def _adjust_covariates(
        self, params: Dict, tumor_vol: float, bw: float, bsa: float,
        egfr: float, hepatic: str, age: float
    ) -> Tuple[float, float, float, float]:
        """Adjust PK parameters for patient covariates (Population PK approach)"""
        cl = params["typical_cl"]
        v1 = params["typical_vd"]
        v2 = params["typical_vd2"]
        q = params["typical_q"]
        
        # Body weight (allometric scaling, exponent 0.75 for CL, 1.0 for V)
        bw_ref = 70.0
        cl *= (bw / bw_ref) ** 0.75
        v1 *= (bw / bw_ref)
        v2 *= (bw / bw_ref)
        q *= (bw / bw_ref) ** 0.75
        
        # Renal function (for renally cleared drugs)
        if egfr < 90:
            renal_factor = max(0.3, egfr / 120.0)
            cl *= renal_factor
        
        # Hepatic function
        hepatic_factors = {
            "normal": 1.0,
            "mild": 0.85,
            "moderate": 0.65,
            "severe": 0.40
        }
        cl *= hepatic_factors.get(hepatic, 1.0)
        
        # Tumor burden: large tumors can act as drug sinks
        if tumor_vol > 50:
            v1 *= 1.0 + (tumor_vol / 1000.0) * 0.3
        
        # Age: clearance decreases ~1% per year after 40
        if age > 40:
            cl *= max(0.7, 1.0 - (age - 40) * 0.005)
        
        return cl, v1, v2, q
    
    def _solve_two_compartment(
        self, dose_mg: float, cl: float, v1: float, v2: float, q: float,
        infusion_dur: float, t_eval: np.ndarray
    ) -> np.ndarray:
        """
        Solve 2-compartment ODE for IV infusion
        
        dA1/dt = R(t) - (CL/V1 + Q/V1) * A1 + (Q/V2) * A2
        dA2/dt = (Q/V1) * A1 - (Q/V2) * A2
        
        where R(t) = dose/infusion_dur during infusion, 0 after
        """
        rate = dose_mg / max(infusion_dur, 0.01)  # mg/h infusion rate
        
        k10 = cl / v1
        k12 = q / v1
        k21 = q / v2
        
        def odes(t, y):
            a1, a2 = y
            r = rate if t <= infusion_dur else 0.0
            da1 = r - (k10 + k12) * a1 + k21 * a2
            da2 = k12 * a1 - k21 * a2
            return [da1, da2]
        
        sol = solve_ivp(
            odes, (t_eval[0], t_eval[-1]), [0.0, 0.0],
            t_eval=t_eval, method='RK45', max_step=0.1
        )
        
        # Convert amount to concentration (µg/mL = mg/L)
        concentration = sol.y[0] / v1  # mg/L = µg/mL
        concentration = np.maximum(concentration, 0)
        
        return concentration
    
    def _calculate_hybrid_constants(
        self, cl: float, v1: float, v2: float, q: float
    ) -> Tuple[float, float]:
        """Calculate macro-constants alpha and beta"""
        k10 = cl / v1
        k12 = q / v1
        k21 = q / v2
        
        sum_k = k10 + k12 + k21
        product_k = k10 * k21
        
        discriminant = sum_k ** 2 - 4 * product_k
        if discriminant < 0:
            # Degenerate case
            return sum_k / 2, sum_k / 2
        
        sqrt_disc = np.sqrt(discriminant)
        alpha = (sum_k + sqrt_disc) / 2  # Faster (distribution)
        beta = (sum_k - sqrt_disc) / 2   # Slower (elimination)
        
        return float(alpha), float(beta)
    
    def _calculate_dosing_interval(
        self, conc: np.ndarray, times: np.ndarray, 
        cmin: float, t_half: float
    ) -> int:
        """Determine dosing interval to maintain therapeutic concentration"""
        # Find when concentration drops below Cmin after Cmax
        cmax_idx = np.argmax(conc)
        
        below_min = np.where(conc[cmax_idx:] < cmin)[0]
        if len(below_min) > 0:
            interval_hours = float(times[cmax_idx + below_min[0]])
        else:
            # Never drops below — use 3× half-life
            interval_hours = t_half * 3
        
        # Round to clinical intervals: 6, 8, 12, 24, 48, 168 (weekly), 336 (biweekly)
        standard_intervals = [6, 8, 12, 24, 48, 168, 336, 504]
        interval = min(standard_intervals, key=lambda x: abs(x - interval_hours))
        
        return interval
    
    def _optimize_dose(
        self, current_dose: float, cl: float, v1: float, v2: float, q: float,
        bsa: float, infusion_dur: float, therapeutic: Tuple, params: Dict
    ) -> float:
        """Optimize dose to achieve target AUC within therapeutic window"""
        target_auc_per_interval = (therapeutic[0] + therapeutic[1]) / 2 * 24
        
        t_eval = np.linspace(0, 48, 481)
        
        def objective(dose_factor):
            test_dose = current_dose * dose_factor * bsa
            conc = self._solve_two_compartment(
                test_dose, cl, v1, v2, q, infusion_dur, t_eval
            )
            auc = np.trapz(conc, t_eval)
            cmax = np.max(conc)
            
            # Penalize exceeding Cmax
            penalty = 0
            if cmax > therapeutic[1] * 1.5:
                penalty = (cmax - therapeutic[1] * 1.5) ** 2 * 100
            
            return (auc - target_auc_per_interval) ** 2 + penalty
        
        result = minimize_scalar(objective, bounds=(0.3, 3.0), method='bounded')
        optimal_dose = current_dose * result.x
        
        return optimal_dose
    
    def _assess_toxicity(
        self, cmax: float, therapeutic: Tuple, cl: float, egfr: float
    ) -> str:
        """Assess toxicity risk based on PK parameters"""
        risk_score = 0
        
        # Cmax exceeds upper therapeutic limit
        if cmax > therapeutic[1] * 2:
            risk_score += 3
        elif cmax > therapeutic[1] * 1.5:
            risk_score += 2
        elif cmax > therapeutic[1]:
            risk_score += 1
        
        # Impaired clearance
        if egfr < 30:
            risk_score += 3
        elif egfr < 60:
            risk_score += 2
        elif egfr < 90:
            risk_score += 1
        
        if risk_score >= 4:
            return "High"
        elif risk_score >= 2:
            return "Moderate"
        else:
            return "Low"
