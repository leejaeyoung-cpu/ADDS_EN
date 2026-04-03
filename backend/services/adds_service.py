"""ADDS Service - Wrapper for existing ADDS system"""

from pathlib import Path
from typing import Dict, Any, Optional
import sys

# Add ADDS src to path
ADDS_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ADDS_ROOT))
sys.path.insert(0, str(ADDS_ROOT / "src"))

from medical_imaging.adds_integrator import ADDSIntegrator
from utils.adds_recommender import ADDSRecommender


class ADDSService:
    """
    Wrapper service for ADDS system
    Integrates pathway-based drug recommendation
    """
    
    def __init__(self):
        """Initialize ADDS components"""
        self.integrator = ADDSIntegrator()
        self.recommender = ADDSRecommender()
    
    def run_inference(
        self,
        radiomics: Dict[str, Any],
        tumor_characteristics: Dict[str, Any],
        clinical_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Run ADDS pathway-based inference
        
        Args:
            radiomics: Radiomics features
            tumor_characteristics: Tumor morphology and intensity features
            clinical_data: Clinical metadata (TNM, MSI, KRAS, etc.)
            
        Returns:
            ADDS inference result with drug recommendations
        """
        # Prepare classification dict for integrator
        classification = {
            "tumor_location": clinical_data.get("tumor_location", "Unknown"),
            "tnm_stage": clinical_data.get("tnm_stage", "Unknown"),
            "msi_status": clinical_data.get("msi_status", "Unknown"),
            "kras_mutation": clinical_data.get("kras_mutation", "Unknown")
        }
        
        # Prepare tumor analysis from characteristics
        tumor_analysis = {
            "volume_mm3": tumor_characteristics.get("morphology", {}).get("area_mm2", 1000) * 5,  # Approximate volume
            "centroid": (
                tumor_characteristics.get("location", {}).get("centroid_x", 0),
                tumor_characteristics.get("location", {}).get("centroid_y", 0),
                tumor_characteristics.get("location", {}).get("slice_index", 0)
            )
        }
        
        # Use ADDS integrator to prepare input
        adds_input = self.integrator.prepare_adds_input(
            volume=None,  # Already processed
            tumor_analysis=tumor_analysis,
            radiomics=radiomics,
            classification=classification
        )
        
        # Get drug recommendations via ADDS recommender
        recommendation = self.recommender.recommend_combination(
            cancer_type="colorectal",
            num_drugs=3,
            patient_data=clinical_data
        )
        
        # Format result — normalize recommender output keys
        raw_drugs = recommendation.get("recommended_drugs", [])
        drug_names = [d.get("drug", d.get("name", "Unknown")) for d in raw_drugs]
        
        from datetime import datetime
        result = {
            "system": "ADDS",
            "version": "2.0",
            "recommended_drugs": drug_names,
            "drug_details": raw_drugs,
            "rationale": recommendation.get("rationale", ""),
            "targeted_pathways": recommendation.get("pathways_covered", []),
            "synergy_score": recommendation.get("synergy_score", 0.0),
            "confidence": adds_input.get("drug_predictions", {}).get("confidence", 0.85),
            "estimated_survival": adds_input.get("prognosis", {}).get("survival_5_year", 0.7),
            "prognosis": adds_input.get("prognosis", {}),
            "mechanism_details": {
                "primary_mechanisms": recommendation.get("mechanisms", []),
                "pathway_interactions": recommendation.get("pathway_interactions", {})
            },
            "generated_at": datetime.now().isoformat()
        }
        
        return result
