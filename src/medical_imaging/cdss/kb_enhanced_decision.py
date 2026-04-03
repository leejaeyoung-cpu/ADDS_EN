"""
ADDS CDSS - Knowledge Base Enhanced Decision Module
===================================================
Enhances existing CDSS with literature-based evidence.

Integrates:
- Literature knowledge base query
- Evidence-based drug recommendations
- Drug combination insights
- Biomarker matching
- Mechanism exploration

Author: ADDS Team
Date: 2026-01-31
"""

import sys
from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import dataclass

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Import knowledge base
try:
    from knowledge_base.kb_query import KnowledgeBaseQuery
    KB_AVAILABLE = True
except ImportError:
    KB_AVAILABLE = False
    print("[WARN] Knowledge base not available")


@dataclass
class EvidenceBasedRecommendation:
    """Evidence-based treatment recommendation"""
    drug_name: str
    drug_class: str
    mechanism: str
    target: str
    evidence_level: int  # Number of supporting papers
    pmids: List[str]
    cancer_specific: bool


@dataclass
class CombinationInsight:
    """Drug combination insight from literature"""
    drugs: List[str]
    synergy_type: str
    evidence: str
    pmid: str
    cancer_type: str


class KBEnhancedDecision:
    """
    Enhances CDSS decision with knowledge base evidence
    """
    
    def __init__(self):
        self.kb = None
        if KB_AVAILABLE:
            try:
                self.kb = KnowledgeBaseQuery()
                print("[KB-CDSS] Knowledge base loaded successfully")
            except Exception as e:
                print(f"[KB-CDSS] Failed to load KB: {e}")
                self.kb = None
    
    def is_available(self) -> bool:
        """Check if knowledge base is available"""
        return self.kb is not None
    
    def get_evidence_based_drugs(
        self,
        cancer_type: str,
        mutations: Optional[List[str]] = None,
        min_evidence: int = 1
    ) -> List[EvidenceBasedRecommendation]:
        """
        Get evidence-based drug recommendations
        
        Args:
            cancer_type: Type of cancer (e.g., "colorectal", "breast")
            mutations: List of known mutations (e.g., ["KRAS", "BRAF"])
            min_evidence: Minimum number of supporting papers
        
        Returns:
            List of evidence-based recommendations
        """
        if not self.kb:
            return []
        
        try:
            # Query KB for drugs
            kb_drugs = self.kb.search_drugs_by_cancer_type(
                cancer_type=cancer_type,
                min_papers=min_evidence
            )
            
            recommendations = []
            
            for drug in kb_drugs:
                # Check if drug targets relevant mutations
                cancer_specific = False
                if mutations:
                    target_lower = drug.target.lower()
                    cancer_specific = any(
                        mut.lower() in target_lower 
                        for mut in mutations
                    )
                
                recommendations.append(EvidenceBasedRecommendation(
                    drug_name=drug.drug_name,
                    drug_class=drug.drug_class,
                    mechanism=drug.mechanism,
                    target=drug.target,
                    evidence_level=drug.paper_count,
                    pmids=drug.pmids,
                    cancer_specific=cancer_specific
                ))
            
            # Sort: cancer-specific first, then by evidence level
            recommendations.sort(
                key=lambda x: (not x.cancer_specific, -x.evidence_level)
            )
            
            return recommendations
            
        except Exception as e:
            print(f"[KB-CDSS] Error getting drug recommendations: {e}")
            return []
    
    def get_combination_insights(
        self,
        cancer_type: str,
        current_drugs: Optional[List[str]] = None
    ) -> List[CombinationInsight]:
        """
        Get drug combination insights from literature
        
        Args:
            cancer_type: Type of cancer
            current_drugs: Currently considered drugs
        
        Returns:
            List of combination insights
        """
        if not self.kb:
            return []
        
        try:
            # Query for combinations
            combos = self.kb.find_drug_combinations(
                cancer_type=cancer_type,
                include_drug=current_drugs[0] if current_drugs else None
            )
            
            insights = []
            for combo in combos:
                insights.append(CombinationInsight(
                    drugs=combo.drugs,
                    synergy_type=combo.synergy_type,
                    evidence=combo.evidence,
                    pmid=combo.pmid,
                    cancer_type=combo.cancer_type
                ))
            
            return insights
            
        except Exception as e:
            print(f"[KB-CDSS] Error getting combinations: {e}")
            return []
    
    def get_biomarker_insights(
        self,
        cancer_type: str
    ) -> List[Dict]:
        """
        Get relevant biomarkers for cancer type
        
        Args:
            cancer_type: Type of cancer
        
        Returns:
            List of biomarker information
        """
        if not self.kb:
            return []
        
        try:
            biomarkers = self.kb.find_biomarkers_for_cancer(cancer_type)
            return biomarkers
        except Exception as e:
            print(f"[KB-CDSS] Error getting biomarkers: {e}")
            return []
    
    def get_mechanism_insights(
        self,
        target_proteins: List[str]
    ) -> List[Dict]:
        """
        Get mechanism insights for target proteins
        
        Args:
            target_proteins: List of proteins (e.g., ["KRAS", "BRAF"])
        
        Returns:
            List of mechanism information
        """
        if not self.kb:
            return []
        
        try:
            all_mechanisms = []
            for protein in target_proteins:
                mechs = self.kb.search_mechanisms_by_target(protein)
                all_mechanisms.extend(mechs)
            
            # Remove duplicates
            seen = set()
            unique_mechs = []
            for mech in all_mechanisms:
                if mech['pathway'] not in seen:
                    seen.add(mech['pathway'])
                    unique_mechs.append(mech)
            
            return unique_mechs
            
        except Exception as e:
            print(f"[KB-CDSS] Error getting mechanisms: {e}")
            return []
    
    def enhance_treatment_recommendation(
        self,
        cancer_type: str,
        mutations: List[str],
        current_therapies: List[str]
    ) -> Dict:
        """
        Comprehensive treatment enhancement with literature evidence
        
        Args:
            cancer_type: Type of cancer
            mutations: Known genetic mutations
            current_therapies: Current therapy recommendations from CDSS
        
        Returns:
            Enhanced recommendation with evidence
        """
        if not self.kb:
            return {
                'kb_available': False,
                'message': 'Knowledge base not available'
            }
        
        try:
            # Get comprehensive recommendation
            kb_recommendation = self.kb.generate_treatment_recommendation(
                cancer_type=cancer_type,
                known_mutations=mutations,
                avoid_drugs=[]  # Could add contraindications
            )
            
            return {
                'kb_available': True,
                'evidence_drugs': kb_recommendation['primary_drugs'][:10],
                'combination_therapies': kb_recommendation['combination_therapies'][:5],
                'relevant_biomarkers': kb_recommendation['relevant_biomarkers'],
                'evidence_base': kb_recommendation['evidence_base'],
                'cancer_type': cancer_type,
                'mutations': mutations
            }
            
        except Exception as e:
            print(f"[KB-CDSS] Error enhancing recommendation: {e}")
            return {
                'kb_available': False,
                'error': str(e)
            }


# Global instance (lazy loading)
_kb_decision = None

def get_kb_decision() -> KBEnhancedDecision:
    """Get or create KB decision enhancer"""
    global _kb_decision
    if _kb_decision is None:
        _kb_decision = KBEnhancedDecision()
    return _kb_decision


def format_evidence_summary(recommendations: List[EvidenceBasedRecommendation]) -> str:
    """Format evidence-based recommendations as markdown"""
    if not recommendations:
        return "No literature evidence available."
    
    md = "### Literature-Based Evidence\n\n"
    
    for i, rec in enumerate(recommendations[:5], 1):  # Top 5
        mutation_tag = "[MATCHED] " if rec.cancer_specific else ""
        md += f"**{i}. {mutation_tag}{rec.drug_name}** ({rec.drug_class})\n"
        md += f"   - Target: {rec.target}\n"
        md += f"   - Mechanism: {rec.mechanism[:80]}...\n" if len(rec.mechanism) > 80 else f"   - Mechanism: {rec.mechanism}\n"
        md += f"   - Evidence: {rec.evidence_level} papers\n"
        md += f"   - PMIDs: {', '.join(rec.pmids[:3])}\n"
        if rec.cancer_specific:
            md += f"   - **Mutation-matched**: Targets known mutations\n"
        md += "\n"
    
    return md


def format_combination_insights(insights: List[CombinationInsight]) -> str:
    """Format combination insights as markdown"""
    if not insights:
        return "No combination insights available."
    
    md = "### Drug Combination Insights\n\n"
    
    for i, insight in enumerate(insights[:3], 1):  # Top 3
        md += f"**{i}. {' + '.join(insight.drugs)}**\n"
        md += f"   - Synergy: {insight.synergy_type}\n"
        md += f"   - Cancer: {insight.cancer_type}\n"
        md += f"   - Evidence: {insight.evidence[:100]}...\n" if len(insight.evidence) > 100 else f"   - Evidence: {insight.evidence}\n"
        md += f"   - Reference: PMID {insight.pmid}\n\n"
    
    return md


if __name__ == "__main__":
    print("Testing KB-Enhanced CDSS Decision Module...")
    
    kb_decision = get_kb_decision()
    
    if kb_decision.is_available():
        print("[OK] Knowledge base available")
        
        # Test drug recommendations
        drugs = kb_decision.get_evidence_based_drugs(
            cancer_type="colorectal",
            mutations=["KRAS", "BRAF"],
            min_evidence=2
        )
        
        print(f"\nFound {len(drugs)} evidence-based drugs")
        print(format_evidence_summary(drugs))
        
        # Test combinations
        combos = kb_decision.get_combination_insights(
            cancer_type="colorectal"
        )
        
        print(f"\nFound {len(combos)} combination insights")
        print(format_combination_insights(combos))
        
    else:
        print("[WARN] Knowledge base not available")
