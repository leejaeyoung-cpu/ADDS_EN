"""
Drug Synergy Calculator
Implements standard synergy models: Bliss, Loewe, HSA, and ZIP
Based on latest research from Nature Methods and pharmacology literature
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from scipy.optimize import fsolve


class SynergyCalculator:
    """
    Calculate drug combination synergy using multiple reference models
    
    Models implemented:
    - Bliss Independence: Assumes drugs act independently
    - Loewe Additivity: Based on dose equivalence
    - HSA (Highest Single Agent): Conservative baseline
    - ZIP (Zero Interaction Potency): Integrates Bliss and Loewe
    """
    
    def __init__(self):
        pass
    
    def calculate_all_scores(
        self,
        drug_a_effect: float,
        drug_b_effect: float,
        combined_effect: float,
        dose_a: Optional[float] = None,
        dose_b: Optional[float] = None,
        ec50_a: Optional[float] = None,
        ec50_b: Optional[float] = None
    ) -> Dict[str, float]:
        """
        Calculate all synergy scores
        
        Args:
            drug_a_effect: Effect of drug A alone (0-1, where 1 is max effect)
            drug_b_effect: Effect of drug B alone (0-1)
            combined_effect: Effect of combination (0-1)
            dose_a: Dose of drug A (required for Loewe)
            dose_b: Dose of drug B (required for Loewe)
            ec50_a: EC50 of drug A (required for Loewe)
            ec50_b: EC50 of drug B (required for Loewe)
            
        Returns:
            Dictionary with synergy scores from all models
        """
        results = {
            'bliss': self.bliss_independence(drug_a_effect, drug_b_effect, combined_effect),
            'hsa': self.hsa(drug_a_effect, drug_b_effect, combined_effect)
        }
        
        # Loewe requires dose and EC50 information
        if all(v is not None for v in [dose_a, dose_b, ec50_a, ec50_b]):
            results['loewe'] = self.loewe_additivity(dose_a, dose_b, ec50_a, ec50_b)
        
        # ZIP requires dose-response curves (simplified version)
        if all(v is not None for v in [dose_a, dose_b, ec50_a, ec50_b]):
            results['zip'] = self.zip_score(
                drug_a_effect, drug_b_effect, combined_effect,
                dose_a, dose_b, ec50_a, ec50_b
            )
        
        return results
    
    def bliss_independence(
        self,
        effect_a: float,
        effect_b: float,
        effect_combined: float
    ) -> float:
        """
        Bliss Independence Model
        
        Assumes drugs act independently (probabilistic model)
        Expected combined effect: E(A+B) = EA + EB - (EA × EB)
        
        Synergy score:
        - Positive: Synergy (observed > expected)
        - Zero: Additive
        - Negative: Antagonism (observed < expected)
        
        Args:
            effect_a: Drug A effect (0-1)
            effect_b: Drug B effect (0-1)
            effect_combined: Observed combination effect (0-1)
            
        Returns:
            Bliss synergy score (delta from expected)
        """
        expected = effect_a + effect_b - (effect_a * effect_b)
        synergy = effect_combined - expected
        
        return synergy
    
    def loewe_additivity(
        self,
        dose_a: float,
        dose_b: float,
        ec50_a: float,
        ec50_b: float
    ) -> float:
        """
        Loewe Additivity Model
        
        Based on dose equivalence principle
        Combination Index (CI) = (dose_a / EC50_a) + (dose_b / EC50_b)
        
        CI interpretation:
        - CI < 1: Synergy
        - CI = 1: Additive
        - CI > 1: Antagonism
        
        Args:
            dose_a: Dose of drug A
            dose_b: Dose of drug B
            ec50_a: EC50 of drug A (dose for 50% effect)
            ec50_b: EC50 of drug B
            
        Returns:
            Combination Index (CI)
        """
        ci = (dose_a / ec50_a) + (dose_b / ec50_b)
        
        return ci
    
    def hsa(
        self,
        effect_a: float,
        effect_b: float,
        effect_combined: float
    ) -> float:
        """
        Highest Single Agent (HSA) Model
        
        Conservative synergy model
        Expected effect = max(effect_a, effect_b)
        
        Synergy score:
        - Positive: Synergy
        - Zero or Negative: No synergy
        
        Args:
            effect_a: Drug A effect (0-1)
            effect_b: Drug B effect (0-1)
            effect_combined: Observed combination effect (0-1)
            
        Returns:
            HSA synergy score
        """
        max_single = max(effect_a, effect_b)
        synergy = effect_combined - max_single
        
        return synergy
    
    def zip_score(
        self,
        effect_a: float,
        effect_b: float,
        effect_combined: float,
        dose_a: float,
        dose_b: float,
        ec50_a: float,
        ec50_b: float
    ) -> float:
        """
        Zero Interaction Potency (ZIP) Model
        
        Integrates Bliss and Loewe approaches
        Accounts for both potency changes and efficacy changes
        
        Simplified delta score calculation
        
        Args:
            effect_a: Drug A effect  
            effect_b: Drug B effect
            effect_combined: Observed combination effect
            dose_a: Dose of drug A
            dose_b: Dose of drug B
            ec50_a: EC50 of drug A
            ec50_b: EC50 of drug B
            
        Returns:
            ZIP delta score (positive = synergy)
        """
        # Simplified ZIP implementation
        # Full ZIP requires entire dose-response surface
        
        # Potency component (Loewe-like)
        potency_ratio_a = dose_a / ec50_a
        potency_ratio_b = dose_b / ec50_b
        
        # Efficacy component (Bliss-like)
        bliss_expected = effect_a + effect_b - (effect_a * effect_b)
        efficacy_delta = effect_combined - bliss_expected
        
        # Combined ZIP score (simplified)
        zip_delta = efficacy_delta * (1 + potency_ratio_a + potency_ratio_b)
        
        return zip_delta
    
    def classify_interaction(self, synergy_scores: Dict[str, float]) -> str:
        """
        Classify drug interaction based on consensus of models
        
        Args:
            synergy_scores: Dictionary with scores from different models
            
        Returns:
            Classification: "Strong Synergy", "Synergy", "Additive", "Antagonism"
        """
        # Count synergistic signals
        synergy_count = 0
        total_models = 0
        
        if 'bliss' in synergy_scores:
            total_models += 1
            if synergy_scores['bliss'] > 0.1:  # Threshold for meaningful synergy
                synergy_count += 1
        
        if 'hsa' in synergy_scores:
            total_models += 1
            if synergy_scores['hsa'] > 0.05:
                synergy_count += 1
        
        if 'loewe' in synergy_scores:
            total_models += 1
            if synergy_scores['loewe'] < 0.9:  # CI < 1 indicates synergy
                synergy_count += 1
        
        if 'zip' in synergy_scores:
            total_models += 1
            if synergy_scores['zip'] > 0.1:
                synergy_count += 1
        
        # Consensus classification
        if synergy_count == total_models and synergy_count > 0:
            return "Strong Synergy"
        elif synergy_count >= total_models * 0.6:
            return "Synergy"
        elif synergy_count >= total_models * 0.4:
            return "Additive"
        else:
            return "Antagonism"
    
    def generate_synergy_report(
        self,
        drug_a_name: str,
        drug_b_name: str,
        synergy_scores: Dict[str, float]
    ) -> Dict:
        """
        Generate comprehensive synergy report
        
        Args:
            drug_a_name: Name of drug A
            drug_b_name: Name of drug B
            synergy_scores: Calculated synergy scores
            
        Returns:
            Detailed report dictionary
        """
        classification = self.classify_interaction(synergy_scores)
        
        report = {
            'drug_a': drug_a_name,
            'drug_b': drug_b_name,
            'interaction_type': classification,
            'scores': synergy_scores,
            'interpretation': self._interpret_scores(synergy_scores, classification)
        }
        
        return report
    
    def _interpret_scores(
        self,
        scores: Dict[str, float],
        classification: str
    ) -> str:
        """Generate human-readable interpretation"""
        
        interpretations = []
        
        if 'bliss' in scores:
            if scores['bliss'] > 0.1:
                interpretations.append(
                    f"Bliss 모델: {scores['bliss']:.3f} (조합이 기대보다 {scores['bliss']*100:.1f}% 더 효과적)"
                )
            elif scores['bliss'] < -0.1:
                interpretations.append(
                    f"Bliss 모델: {scores['bliss']:.3f} (조합이 기대보다 덜 효과적)"
                )
        
        if 'loewe' in scores:
            if scores['loewe'] < 0.9:
                interpretations.append(
                    f"Loewe CI: {scores['loewe']:.2f} (시너지, CI < 1)"
                )
            elif scores['loewe'] > 1.1:
                interpretations.append(
                    f"Loewe CI: {scores['loewe']:.2f} (길항작용, CI > 1)"
                )
        
        if 'hsa' in scores:
            if scores['hsa'] > 0.05:
                interpretations.append(
                    f"HSA 모델: {scores['hsa']:.3f} (단일 약물 대비 추가 효과)"
                )
        
        summary = f"**종합: {classification}**\n" + "\n".join(interpretations)
        
        return summary


# Example usage
if __name__ == "__main__":
    calculator = SynergyCalculator()
    
    # Example 1: Bliss calculation
    scores = calculator.calculate_all_scores(
        drug_a_effect=0.4,
        drug_b_effect=0.3,
        combined_effect=0.8,  # Strong synergy
        dose_a=10,
        dose_b=5,
        ec50_a=15,
        ec50_b=10
    )
    
    report = calculator.generate_synergy_report(
        "Encorafenib",
        "Cetuximab",
        scores
    )
    
    print("Synergy Report:")
    print(f"Interaction: {report['interaction_type']}")
    print(f"Scores: {report['scores']}")
    print(f"\n{report['interpretation']}")
