"""
ADDS Recommender - Pathway-based drug combination recommendation
"""

from typing import Dict, List, Tuple, Any
import sys
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent.parent
sys.path.insert(0, str(src_path))

from knowledge.signal_pathways import (
    SIGNAL_PATHWAYS,
    PATHWAY_CROSSTALK,
    get_pathways_for_cancer,
    get_drugs_for_pathway,
    get_pathway_crosstalk
)
from utils.drug_database import DRUG_DATABASE


class ADDSRecommender:
    """
    ADDS (AI-based Anticancer Drug Discovery System) Recommender
    Pathway-based drug combination recommendation engine
    """
    
    def __init__(self):
        self.pathways = SIGNAL_PATHWAYS
        self.crosstalk = PATHWAY_CROSSTALK
    
    def recommend_combination(
        self,
        cancer_type: str,
        num_drugs: int = 3,
        patient_data: Dict = None
    ) -> Dict[str, Any]:
        """
        Recommend drug combination based on pathway analysis
        
        Args:
            cancer_type: Type of cancer
            num_drugs: Number of drugs in combination
            patient_data: Optional patient-specific data
            
        Returns:
            Recommendation with rationale
        """
        # Step 1: Identify activated pathways
        activated_pathways = get_pathways_for_cancer(cancer_type, min_activation=0.3)
        
        if not activated_pathways:
            return {
                "success": False,
                "message": f"No major pathways identified for {cancer_type} cancer"
            }
        
        # Step 2: Rank pathways by activation
        pathway_scores = []
        for pathway_id in activated_pathways:
            pathway = self.pathways[pathway_id]
            activation = pathway["activation_in_cancer"].get(cancer_type, 0)
            pathway_scores.append((pathway_id, activation))
        
        pathway_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Step 3: Apply patient-specific contraindications
        excluded_drugs = set()
        patient_notes = []
        
        if patient_data:
            # KRAS mutation → anti-EGFR contraindicated (NCCN CRC guideline)
            kras = patient_data.get("kras_mutation", "").lower()
            if kras in ("mutant", "mut", "positive", "g12d", "g12v", "g13d"):
                excluded_drugs.update(["Cetuximab", "Panitumumab"])
                patient_notes.append("KRAS mutant → anti-EGFR (Cetuximab, Panitumumab) excluded")
            
            # MSI-H / dMMR → prioritize immunotherapy
            msi = patient_data.get("microsatellite_status", "").upper()
            if msi in ("MSI-H", "MSI_H", "DMMR"):
                # Boost NF_kB pathway (where immunotherapy-related drugs may be)
                for i, (pid, score) in enumerate(pathway_scores):
                    if pid == "NF_kB":
                        pathway_scores[i] = (pid, min(score + 0.3, 1.0))
                pathway_scores.sort(key=lambda x: x[1], reverse=True)
                patient_notes.append("MSI-H/dMMR → immunotherapy pathway prioritized")
            
            # Renal impairment → avoid nephrotoxic agents
            egfr_renal = patient_data.get("egfr", None)
            if egfr_renal is not None and float(egfr_renal) < 50:
                excluded_drugs.update(["Cisplatin"])  # Cisplatin: GFR>50 required
                patient_notes.append(f"eGFR {egfr_renal} < 50 → Cisplatin excluded")
            
            # Hepatic impairment → avoid hepatotoxic agents
            bilirubin = patient_data.get("bilirubin", None)
            if bilirubin is not None and float(bilirubin) > 3.0:
                excluded_drugs.update(["Irinotecan", "Doxorubicin"])
                patient_notes.append(f"Bilirubin {bilirubin} > 3.0 → Irinotecan, Doxorubicin excluded")
            
            # Age/ECOG → restrict aggressive regimens
            ecog = patient_data.get("ecog", None)
            age = patient_data.get("age", None)
            if (ecog is not None and int(ecog) >= 3) or (age is not None and int(age) >= 80):
                excluded_drugs.update(["Cisplatin", "Doxorubicin"])  # Avoid high-toxicity
                patient_notes.append("ECOG≥3 or age≥80 → high-toxicity agents excluded")
        
        # Step 4: Select drugs targeting top pathways (excluding contraindicated)
        drug_recommendations = []
        covered_pathways = set()
        
        for pathway_id, activation_score in pathway_scores:
            if len(drug_recommendations) >= num_drugs:
                break
            
            pathway_drugs = get_drugs_for_pathway(pathway_id)
            
            for drug_name, drug_info in pathway_drugs.items():
                if drug_name in excluded_drugs:
                    continue
                if drug_name not in [d["drug"] for d in drug_recommendations]:
                    drug_recommendations.append({
                        "drug": drug_name,
                        "pathway_targeted": pathway_id,
                        "pathway_name": self.pathways[pathway_id]["name"],
                        "target": drug_info["target"],
                        "mechanism": drug_info["mechanism"],
                        "efficacy": drug_info["efficacy"],
                        "activation_score": activation_score
                    })
                    covered_pathways.add(pathway_id)
                    
                    if len(drug_recommendations) >= num_drugs:
                        break
        
        # Step 5: Calculate synergy potential
        synergy_score = self._calculate_pathway_synergy(covered_pathways)
        
        # Step 6: Generate rationale
        rationale = self._generate_rationale(
            cancer_type,
            drug_recommendations,
            pathway_scores,
            synergy_score
        )
        
        return {
            "success": True,
            "cancer_type": cancer_type,
            "recommended_drugs": drug_recommendations,
            "pathways_covered": list(covered_pathways),
            "synergy_score": synergy_score,
            "rationale": rationale,
            "patient_contraindications": patient_notes,
            "excluded_drugs": list(excluded_drugs),
            "pathway_details": {
                pid: {
                    "name": self.pathways[pid]["name"],
                    "activation": self.pathways[pid]["activation_in_cancer"].get(cancer_type, 0),
                    "function": self.pathways[pid]["function"]
                }
                for pid in covered_pathways
            }
        }
    
    def _calculate_pathway_synergy(self, pathways: set) -> float:
        """Calculate synergy potential between pathways"""
        if len(pathways) < 2:
            return 0.5
        
        pathway_list = list(pathways)
        synergy_scores = []
        
        for i, p1 in enumerate(pathway_list):
            for p2 in pathway_list[i+1:]:
                crosstalk = get_pathway_crosstalk(p1, p2)
                if crosstalk:
                    synergy_scores.append(crosstalk.get("synergy_potential", 0.5))
        
        return sum(synergy_scores) / len(synergy_scores) if synergy_scores else 0.5
    
    def _generate_rationale(
        self,
        cancer_type: str,
        drugs: List[Dict],
        pathway_scores: List[Tuple],
        synergy: float
    ) -> str:
        """Generate human-readable rationale"""
        
        rationale_parts = []
        
        # Introduction
        rationale_parts.append(
            f"**{cancer_type} Cancer Pathway Analysis**\n"
        )
        
        # Pathway activation
        top_pathways = pathway_scores[:3]
        rationale_parts.append(
            f"**Key Activated Pathways** (activation rate):\n"
        )
        for pathway_id, activation in top_pathways:
            pathway_name = self.pathways[pathway_id]["name"]
            rationale_parts.append(
                f"- {pathway_name}: {activation*100:.0f}% activation\n"
            )
        
        # Drug recommendations
        rationale_parts.append(
            f"\n**Recommended Combination** ({len(drugs)} drugs):\n"
        )
        for i, drug_rec in enumerate(drugs, 1):
            rationale_parts.append(
                f"{i}. **{drug_rec['drug']}**\n"
                f"   - Target: {drug_rec['target']}\n"
                f"   - Pathway: {drug_rec['pathway_name']}\n"
                f"   - Mechanism: {drug_rec['mechanism']}\n"
            )
        
        # Synergy explanation
        rationale_parts.append(
            f"\n**Synergy Potential**: {synergy:.2f}/1.0\n"
        )
        
        if synergy > 0.7:
            rationale_parts.append(
                "✅ High synergy - these pathways show cooperative interactions\n"
            )
        elif synergy > 0.5:
            rationale_parts.append(
                "⚠️ Moderate synergy - complementary pathway targeting\n"
            )
        else:
            rationale_parts.append(
                "ℹ️ Independent pathways - reduced crosstalk\n"
            )
        
        # Clinical notes
        rationale_parts.append(
            f"\n**Clinical Considerations**:\n"
        )
        for pathway_id, _ in top_pathways:
            clinical_note = self.pathways[pathway_id].get("clinical_notes", "")
            if clinical_note:
                rationale_parts.append(f"- {clinical_note}\n")
        
        return "".join(rationale_parts)
    
    def analyze_custom_combination(
        self,
        drug_names: List[str],
        cancer_type: str
    ) -> Dict[str, Any]:
        """
        Analyze a user-provided drug combination
        
        Args:
            drug_names: List of drug names
            cancer_type: Type of cancer
            
        Returns:
            Analysis of the combination
        """
        # Map drugs to pathways
        drug_pathway_map = {}
        
        for drug in drug_names:
            # Find which pathways this drug targets
            targeting_pathways = []
            
            for pathway_id, pathway_data in self.pathways.items():
                pathway_drugs = pathway_data.get("drugs_targeting", {})
                if drug in pathway_drugs:
                    targeting_pathways.append({
                        "pathway_id": pathway_id,
                        "pathway_name": pathway_data["name"],
                        "activation": pathway_data["activation_in_cancer"].get(cancer_type, 0),
                        "drug_info": pathway_drugs[drug]
                    })
            
            drug_pathway_map[drug] = targeting_pathways
        
        # Calculate coverage
        covered_pathways = set()
        for pathways in drug_pathway_map.values():
            for p in pathways:
                covered_pathways.add(p["pathway_id"])
        
        synergy = self._calculate_pathway_synergy(covered_pathways)
        
        return {
            "drug_pathway_map": drug_pathway_map,
            "pathways_covered": list(covered_pathways),
            "synergy_score": synergy,
            "coverage_analysis": f"Covers {len(covered_pathways)} pathways with {len(drug_names)} drugs"
        }
