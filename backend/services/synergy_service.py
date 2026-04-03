"""
Synergy Service
Drug combination synergy calculation using multiple models:
- Bliss Independence
- Loewe Additivity
- Highest Single Agent (HSA)
- Zero Interaction Potency (ZIP)
- Combination Index (CI)
"""

import numpy as np
from typing import Dict, List, Optional, Any
from scipy.optimize import brentq
import logging

logger = logging.getLogger(__name__)


class SynergyService:
    """Service for drug synergy calculations"""
    
    SUPPORTED_MODELS = ["bliss", "loewe", "hsa", "zip"]
    
    async def calculate(
        self,
        drug_a_effect: float,
        drug_b_effect: float,
        combination_effect: float,
        model: str = "bliss",
        drug_a_dose: Optional[float] = None,
        drug_b_dose: Optional[float] = None,
        drug_a_ic50: Optional[float] = None,
        drug_b_ic50: Optional[float] = None,
        drug_a_hill: float = 1.0,
        drug_b_hill: float = 1.0
    ) -> Dict[str, Any]:
        """
        Calculate drug synergy score
        
        Args:
            drug_a_effect: Effect of drug A alone (0-1, fraction of viability inhibition)
            drug_b_effect: Effect of drug B alone (0-1)
            combination_effect: Observed effect of A+B combination (0-1)
            model: Synergy model to use
            drug_a_dose: Dose of drug A (required for Loewe/ZIP)
            drug_b_dose: Dose of drug B (required for Loewe/ZIP)
            drug_a_ic50: IC50 of drug A (required for Loewe/ZIP)
            drug_b_ic50: IC50 of drug B (required for Loewe/ZIP)
            drug_a_hill: Hill slope for drug A
            drug_b_hill: Hill slope for drug B
            
        Returns:
            Dict with synergy_score, interpretation, model details
        """
        # Validate input ranges
        for name, val in [("drug_a_effect", drug_a_effect), 
                          ("drug_b_effect", drug_b_effect),
                          ("combination_effect", combination_effect)]:
            if not (0 <= val <= 1):
                raise ValueError(f"{name} must be between 0 and 1, got {val}")
        
        if model == "bliss":
            result = self._bliss(drug_a_effect, drug_b_effect, combination_effect)
        elif model == "loewe":
            result = self._loewe(
                drug_a_effect, drug_b_effect, combination_effect,
                drug_a_dose, drug_b_dose, drug_a_ic50, drug_b_ic50,
                drug_a_hill, drug_b_hill
            )
        elif model == "hsa":
            result = self._hsa(drug_a_effect, drug_b_effect, combination_effect)
        elif model == "zip":
            result = self._zip(
                drug_a_effect, drug_b_effect, combination_effect,
                drug_a_dose, drug_b_dose, drug_a_ic50, drug_b_ic50,
                drug_a_hill, drug_b_hill
            )
        else:
            raise ValueError(
                f"Unsupported model: {model}. Use one of: {self.SUPPORTED_MODELS}"
            )
        
        return result
    
    async def calculate_all_models(
        self, 
        drug_a_effect: float, drug_b_effect: float, combination_effect: float,
        **kwargs
    ) -> Dict[str, Any]:
        """Calculate synergy using all available models and return consensus"""
        results = {}
        
        for model in self.SUPPORTED_MODELS:
            try:
                r = await self.calculate(
                    drug_a_effect, drug_b_effect, combination_effect,
                    model=model, **kwargs
                )
                results[model] = r
            except (ValueError, NotImplementedError) as e:
                results[model] = {"error": str(e)}
        
        # Consensus
        scores = [r["synergy_score"] for r in results.values() 
                  if "synergy_score" in r]
        
        consensus = {
            "models": results,
            "consensus_score": float(np.mean(scores)) if scores else 0.0,
            "agreement": self._assess_agreement(results)
        }
        
        return consensus
    
    def _bliss(
        self, ea: float, eb: float, eab: float
    ) -> Dict[str, Any]:
        """
        Bliss Independence Model
        Expected: E_exp = E_A + E_B - (E_A × E_B)
        Synergy = E_obs - E_exp
        Positive = synergistic, Negative = antagonistic
        """
        expected = ea + eb - (ea * eb)
        synergy = eab - expected
        
        return {
            "synergy_score": float(synergy),
            "expected_effect": float(expected),
            "observed_effect": float(eab),
            "model": "bliss",
            "model_full_name": "Bliss Independence",
            "interpretation": self._interpret_score(synergy),
            "details": {
                "formula": "E_exp = E_A + E_B - (E_A × E_B)",
                "drug_a_effect": float(ea),
                "drug_b_effect": float(eb),
                "excess_over_bliss": float(synergy)
            }
        }
    
    def _hsa(
        self, ea: float, eb: float, eab: float
    ) -> Dict[str, Any]:
        """
        Highest Single Agent (HSA) Model
        Expected: E_exp = max(E_A, E_B)
        Synergy = E_obs - E_exp
        Most conservative synergy model
        """
        expected = max(ea, eb)
        synergy = eab - expected
        
        return {
            "synergy_score": float(synergy),
            "expected_effect": float(expected),
            "observed_effect": float(eab),
            "model": "hsa",
            "model_full_name": "Highest Single Agent",
            "interpretation": self._interpret_score(synergy),
            "details": {
                "formula": "E_exp = max(E_A, E_B)",
                "highest_single": "drug_a" if ea >= eb else "drug_b",
                "drug_a_effect": float(ea),
                "drug_b_effect": float(eb),
                "excess_over_hsa": float(synergy)
            }
        }
    
    def _loewe(
        self, ea: float, eb: float, eab: float,
        dose_a: Optional[float] = None, dose_b: Optional[float] = None,
        ic50_a: Optional[float] = None, ic50_b: Optional[float] = None,
        hill_a: float = 1.0, hill_b: float = 1.0
    ) -> Dict[str, Any]:
        """
        Loewe Additivity Model
        CI = d_A/D_A(E) + d_B/D_B(E) where D_x(E) is the dose of drug x alone 
        that produces effect E
        CI < 1: synergistic, CI = 1: additive, CI > 1: antagonistic
        """
        if any(v is None for v in [dose_a, dose_b, ic50_a, ic50_b]):
            # Fall back to effect-based approximation
            return self._loewe_effect_based(ea, eb, eab)
        
        if eab <= 0 or eab >= 1:
            return {
                "synergy_score": 0.0,
                "combination_index": 1.0,
                "model": "loewe",
                "model_full_name": "Loewe Additivity",
                "interpretation": "additive",
                "details": {"note": "Effect at boundary, CI not calculable"}
            }
        
        # Calculate equivalent doses: what dose of each drug alone 
        # would produce the combination effect?
        # Using Hill equation: E = D^h / (IC50^h + D^h)
        # Solving for D: D = IC50 * (E / (1 - E))^(1/h)
        
        try:
            D_a_equiv = ic50_a * (eab / (1 - eab)) ** (1 / hill_a)
            D_b_equiv = ic50_b * (eab / (1 - eab)) ** (1 / hill_b)
            
            ci = (dose_a / D_a_equiv) + (dose_b / D_b_equiv)
        except (ZeroDivisionError, ValueError):
            ci = 1.0
        
        # Convert CI to synergy score (positive = synergistic)
        synergy = 1.0 - ci
        
        return {
            "synergy_score": float(synergy),
            "combination_index": float(ci),
            "observed_effect": float(eab),
            "model": "loewe",
            "model_full_name": "Loewe Additivity",
            "interpretation": self._interpret_ci(ci),
            "details": {
                "formula": "CI = d_A/D_A(E) + d_B/D_B(E)",
                "dose_a": float(dose_a),
                "dose_b": float(dose_b),
                "ic50_a": float(ic50_a),
                "ic50_b": float(ic50_b),
                "D_a_equivalent": float(D_a_equiv) if 'D_a_equiv' in dir() else None,
                "D_b_equivalent": float(D_b_equiv) if 'D_b_equiv' in dir() else None,
                "ci_interpretation": {
                    "< 0.3": "strong synergy",
                    "0.3-0.7": "synergy", 
                    "0.7-0.9": "moderate synergy",
                    "0.9-1.1": "additive",
                    "1.1-1.5": "moderate antagonism",
                    "> 1.5": "strong antagonism"
                }
            }
        }
    
    def _loewe_effect_based(
        self, ea: float, eb: float, eab: float
    ) -> Dict[str, Any]:
        """Loewe approximation when dose-response data is not available"""
        # Use isobolographic approach approximation
        if ea + eb > 0:
            ci_approx = (ea / max(eab, 0.001)) + (eb / max(eab, 0.001))
            ci_approx = min(ci_approx, 5.0)  # Cap extreme values
        else:
            ci_approx = 1.0
        
        synergy = 1.0 - ci_approx
        
        return {
            "synergy_score": float(synergy),
            "combination_index": float(ci_approx),
            "observed_effect": float(eab),
            "model": "loewe",
            "model_full_name": "Loewe Additivity (effect-based approximation)",
            "interpretation": self._interpret_ci(ci_approx),
            "details": {
                "note": "Approximation used (dose-response data not provided)",
                "drug_a_effect": float(ea),
                "drug_b_effect": float(eb),
                "provide_dose_ic50": "For accurate CI, provide dose_a, dose_b, ic50_a, ic50_b"
            }
        }
    
    def _zip(
        self, ea: float, eb: float, eab: float,
        dose_a: Optional[float] = None, dose_b: Optional[float] = None,
        ic50_a: Optional[float] = None, ic50_b: Optional[float] = None,
        hill_a: float = 1.0, hill_b: float = 1.0
    ) -> Dict[str, Any]:
        """
        Zero Interaction Potency (ZIP) Model
        Combines Bliss and Loewe concepts.
        ZIP score = observed - expected (ZIP reference)
        The ZIP reference accounts for both drugs' dose-response independently.
        """
        # ZIP expected effect: each drug acts independently without 
        # changing the other's potency
        # E_zip = E_A(d_A) + E_B(d_B) - E_A(d_A) * E_B(d_B)
        # This is effectively Bliss but derived from the potency principle
        
        # If dose-response parameters available, use full ZIP
        if all(v is not None for v in [dose_a, dose_b, ic50_a, ic50_b]):
            # Hill equation for each drug
            ea_pred = (dose_a ** hill_a) / (ic50_a ** hill_a + dose_a ** hill_a)
            eb_pred = (dose_b ** hill_b) / (ic50_b ** hill_b + dose_b ** hill_b)
            
            # ZIP reference: assume drugs don't affect each other's potency
            zip_expected = ea_pred + eb_pred - ea_pred * eb_pred
            
            zip_score = eab - zip_expected
            
            details = {
                "formula": "ZIP = E_obs - (E_A + E_B - E_A×E_B) with dose-response",
                "ea_predicted": float(ea_pred),
                "eb_predicted": float(eb_pred),
                "dose_a": float(dose_a),
                "dose_b": float(dose_b),
                "ic50_a": float(ic50_a),
                "ic50_b": float(ic50_b)
            }
        else:
            # Simplified ZIP (equivalent to Bliss in this case)
            zip_expected = ea + eb - ea * eb
            zip_score = eab - zip_expected
            
            details = {
                "formula": "ZIP ≈ Bliss (simplified, no dose-response data)",
                "note": "For full ZIP, provide dose and IC50 parameters"
            }
        
        return {
            "synergy_score": float(zip_score),
            "expected_effect": float(zip_expected),
            "observed_effect": float(eab),
            "model": "zip",
            "model_full_name": "Zero Interaction Potency",
            "interpretation": self._interpret_zip(zip_score),
            "details": details
        }
    
    @staticmethod
    def _interpret_score(score: float) -> str:
        """Interpret Bliss/HSA synergy score"""
        if score > 0.15:
            return "strong synergy"
        elif score > 0.05:
            return "moderate synergy"
        elif score > -0.05:
            return "additive"
        elif score > -0.15:
            return "moderate antagonism"
        else:
            return "strong antagonism"
    
    @staticmethod
    def _interpret_ci(ci: float) -> str:
        """Interpret Combination Index"""
        if ci < 0.3:
            return "strong synergy"
        elif ci < 0.7:
            return "synergy"
        elif ci < 0.9:
            return "moderate synergy"
        elif ci <= 1.1:
            return "additive"
        elif ci <= 1.5:
            return "moderate antagonism"
        else:
            return "strong antagonism"
    
    @staticmethod
    def _interpret_zip(score: float) -> str:
        """Interpret ZIP delta score"""
        if score > 0.10:
            return "strong synergy"
        elif score > 0.0:
            return "synergy"
        elif score > -0.10:
            return "additive"
        else:
            return "antagonism"
    
    @staticmethod
    def _assess_agreement(results: Dict) -> str:
        """Assess agreement across models"""
        interpretations = []
        for r in results.values():
            if isinstance(r, dict) and "interpretation" in r:
                interp = r["interpretation"].lower()
                if "synergy" in interp:
                    interpretations.append("synergy")
                elif "antagoni" in interp:
                    interpretations.append("antagonism")
                else:
                    interpretations.append("additive")
        
        if not interpretations:
            return "no data"
        
        unique = set(interpretations)
        if len(unique) == 1:
            return f"full agreement: {interpretations[0]}"
        
        from collections import Counter
        counts = Counter(interpretations)
        majority = counts.most_common(1)[0]
        return f"majority ({majority[1]}/{len(interpretations)}): {majority[0]}"
