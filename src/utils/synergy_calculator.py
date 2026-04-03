"""
Drug synergy score calculation for ADDS
Implements Bliss, Loewe, HSA, and ZIP methods
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
from scipy import optimize

from utils import get_logger

logger = get_logger(__name__)


class SynergyCalculator:
    """
    Calculate drug synergy scores using various methods
    """
    
    def __init__(self):
        """Initialize synergy calculator"""
        logger.info("✓ Synergy calculator initialized")
    
    def calculate_bliss(
        self,
        effect_a: float,
        effect_b: float,
        effect_combination: float
    ) -> float:
        """
        Calculate Bliss Independence synergy score
        
        Bliss Independence assumes drugs act independently.
        Expected combined effect = E_A + E_B - E_A * E_B
        
        Args:
            effect_a: Effect of drug A (0-1, where 1 is complete inhibition)
            effect_b: Effect of drug B (0-1)
            effect_combination: Observed combined effect (0-1)
        
        Returns:
            Synergy score (positive = synergy, negative = antagonism)
        """
        expected_effect = effect_a + effect_b - (effect_a * effect_b)
        synergy = effect_combination - expected_effect
        
        return synergy
    
    def calculate_loewe(
        self,
        dose_a: float,
        dose_b: float,
        ic50_a: float,
        ic50_b: float,
        effect_combination: float,
        target_effect: float = 0.5
    ) -> float:
        """
        Calculate Loewe Additivity synergy score
        
        Loewe assumes drugs act on the same target.
        Combination Index (CI) = D_A/IC50_A + D_B/IC50_B
        CI < 1: synergistic, CI = 1: additive, CI > 1: antagonistic
        
        Args:
            dose_a: Dose of drug A
            dose_b: Dose of drug B
            ic50_a: IC50 of drug A
            ic50_b: IC50 of drug B
            effect_combination: Observed combined effect
            target_effect: Target effect level (default 0.5 for IC50)
        
        Returns:
            Combination Index (CI)
        """
        if ic50_a == 0 or ic50_b == 0:
            logger.warning("IC50 cannot be zero for Loewe calculation")
            return np.nan
        
        ci = (dose_a / ic50_a) + (dose_b / ic50_b)
        
        # Convert CI to synergy score (negative CI for consistency with other metrics)
        # CI < 1 is synergy, so synergy_score = 1 - CI
        synergy_score = 1 - ci
        
        return synergy_score
    
    def calculate_hsa(
        self,
        effect_a: float,
        effect_b: float,
        effect_combination: float
    ) -> float:
        """
        Calculate Highest Single Agent (HSA) synergy score
        
        HSA compares combination to the better single agent.
        Expected effect = max(E_A, E_B)
        
        Args:
            effect_a: Effect of drug A (0-1)
            effect_b: Effect of drug B (0-1)
            effect_combination: Observed combined effect (0-1)
        
        Returns:
            Synergy score
        """
        expected_effect = max(effect_a, effect_b)
        synergy = effect_combination - expected_effect
        
        return synergy
    
    def calculate_zip(
        self,
        dose_a: float,
        dose_b: float,
        effect_a: float,
        effect_b: float,
        effect_combination: float,
        ic50_a: float,
        ic50_b: float
    ) -> float:
        """
        Calculate Zero Interaction Potency (ZIP) synergy score
        
        ZIP combines Bliss and Loewe approaches.
        
        Args:
            dose_a: Dose of drug A
            dose_b: Dose of drug B
            effect_a: Effect of drug A alone
            effect_b: Effect of drug B alone
            effect_combination: Observed combined effect
            ic50_a: IC50 of drug A
            ic50_b: IC50 of drug B
        
        Returns:
            ZIP synergy score
        """
        # Simplified ZIP calculation
        # Full ZIP requires fitting dose-response curves
        
        # Calculate Bliss and Loewe components
        bliss = self.calculate_bliss(effect_a, effect_b, effect_combination)
        loewe = self.calculate_loewe(dose_a, dose_b, ic50_a, ic50_b, effect_combination)
        
        # ZIP is a weighted combination
        zip_score = (bliss + loewe) / 2
        
        return zip_score
    
    def calculate_all_synergies(
        self,
        dose_a: float,
        dose_b: float,
        effect_a: float,
        effect_b: float,
        effect_combination: float,
        ic50_a: Optional[float] = None,
        ic50_b: Optional[float] = None
    ) -> Dict[str, float]:
        """
        Calculate all synergy metrics
        
        Args:
            dose_a: Dose of drug A
            dose_b: Dose of drug B
            effect_a: Effect of drug A alone (0-1)
            effect_b: Effect of drug B alone (0-1)
            effect_combination: Observed combined effect (0-1)
            ic50_a: IC50 of drug A (optional, for Loewe/ZIP)
            ic50_b: IC50 of drug B (optional, for Loewe/ZIP)
        
        Returns:
            Dictionary of synergy scores
        """
        results = {}
        
        # Always calculate Bliss and HSA
        results['bliss'] = self.calculate_bliss(effect_a, effect_b, effect_combination)
        results['hsa'] = self.calculate_hsa(effect_a, effect_b, effect_combination)
        
        # Calculate Loewe and ZIP if IC50 values provided
        if ic50_a is not None and ic50_b is not None:
            results['loewe'] = self.calculate_loewe(
                dose_a, dose_b, ic50_a, ic50_b, effect_combination
            )
            results['zip'] = self.calculate_zip(
                dose_a, dose_b, effect_a, effect_b,
                effect_combination, ic50_a, ic50_b
            )
        
        # Determine if synergistic (majority vote)
        synergy_values = [v for v in results.values() if not np.isnan(v)]
        if synergy_values:
            results['is_synergistic'] = sum(v > 0 for v in synergy_values) > len(synergy_values) / 2
            results['mean_synergy'] = np.mean(synergy_values)
        
        return results
    
    def analyze_dose_response_matrix(
        self,
        doses_a: np.ndarray,
        doses_b: np.ndarray,
        effects_matrix: np.ndarray,
        effects_a_alone: np.ndarray,
        effects_b_alone: np.ndarray,
        method: str = 'bliss'
    ) -> np.ndarray:
        """
        Calculate synergy across a dose-response matrix
        
        Args:
            doses_a: Array of doses for drug A
            doses_b: Array of doses for drug B
            effects_matrix: 2D array of combined effects (shape: len(doses_a) x len(doses_b))
            effects_a_alone: Effects of drug A alone at doses_a
            effects_b_alone: Effects of drug B alone at doses_b
            method: Synergy method ('bliss', 'hsa', 'loewe', 'zip')
        
        Returns:
            2D array of synergy scores
        """
        synergy_matrix = np.zeros_like(effects_matrix)
        
        for i, dose_a in enumerate(doses_a):
            for j, dose_b in enumerate(doses_b):
                effect_comb = effects_matrix[i, j]
                effect_a = effects_a_alone[i]
                effect_b = effects_b_alone[j]
                
                if method == 'bliss':
                    synergy_matrix[i, j] = self.calculate_bliss(
                        effect_a, effect_b, effect_comb
                    )
                elif method == 'hsa':
                    synergy_matrix[i, j] = self.calculate_hsa(
                        effect_a, effect_b, effect_comb
                    )
                # Add Loewe and ZIP if needed
        
        return synergy_matrix
    
    def fit_ic50(
        self,
        doses: np.ndarray,
        effects: np.ndarray
    ) -> Tuple[float, Dict[str, float]]:
        """
        Fit dose-response curve and extract IC50
        
        Uses 4-parameter logistic (Hill) equation:
        E = E_min + (E_max - E_min) / (1 + (dose/IC50)^hill_slope)
        
        Args:
            doses: Array of drug doses
            effects: Array of observed effects (0-1)
        
        Returns:
            IC50 value and fitted parameters
        """
        def hill_equation(dose, ic50, hill_slope, e_min, e_max):
            return e_min + (e_max - e_min) / (1 + (dose / ic50) ** hill_slope)
        
        # Initial parameter guess
        p0 = [
            np.median(doses),  # IC50
            1.0,               # Hill slope
            0.0,               # E_min
            1.0                # E_max
        ]
        
        try:
            # Fit curve
            popt, _ = optimize.curve_fit(
                hill_equation,
                doses,
                effects,
                p0=p0,
                bounds=([0, 0, 0, 0], [np.inf, 10, 1, 1]),
                maxfev=10000
            )
            
            ic50, hill_slope, e_min, e_max = popt
            
            params = {
                'ic50': ic50,
                'hill_slope': hill_slope,
                'e_min': e_min,
                'e_max': e_max
            }
            
            logger.info(f"✓ Fitted IC50: {ic50:.3f}")
            return ic50, params
            
        except Exception as e:
            logger.warning(f"IC50 fitting failed: {e}")
            return np.nan, {}
    
    def create_synergy_report(
        self,
        drug_a_name: str,
        drug_b_name: str,
        synergy_scores: Dict[str, float],
        dose_a: float,
        dose_b: float
    ) -> pd.DataFrame:
        """
        Create a formatted synergy report
        
        Args:
            drug_a_name: Name of drug A
            drug_b_name: Name of drug B
            synergy_scores: Dictionary of synergy scores
            dose_a: Dose of drug A
            dose_b: Dose of drug B
        
        Returns:
            DataFrame report
        """
        report_data = {
            'Drug A': [drug_a_name],
            'Drug B': [drug_b_name],
            'Dose A': [dose_a],
            'Dose B': [dose_b],
        }
        
        for method, score in synergy_scores.items():
            if method != 'is_synergistic':
                report_data[f'{method.upper()} Score'] = [score]
        
        if 'is_synergistic' in synergy_scores:
            report_data['Synergistic'] = [synergy_scores['is_synergistic']]
        
        df = pd.DataFrame(report_data)
        return df
