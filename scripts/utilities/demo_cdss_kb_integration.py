"""
CDSS KB Integration Demo
========================
Demonstrates the KB-enhanced CDSS with sample patient data

This script shows how the knowledge base integrates with CDSS
to provide evidence-based treatment recommendations.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from medical_imaging.cdss.kb_enhanced_decision import (
    get_kb_decision,
    format_evidence_summary,
    format_combination_insights
)

def demo_colorectal_kras_patient():
    """Demo: Colorectal cancer patient with KRAS mutation"""
    
    print("=" * 80)
    print("CDSS KB Integration Demo")
    print("=" * 80)
    print()
    
    print("Patient Profile:")
    print("-" * 40)
    print("  Diagnosis: Colorectal Cancer")
    print("  Stage: T2N1M0")
    print("  KRAS: Mutant")
    print("  TP53: Wild-type")
    print("  MSI: MSS")
    print()
    
    # Initialize KB
    kb_decision = get_kb_decision()
    
    if not kb_decision.is_available():
        print("[ERROR] Knowledge base not available!")
        return
    
    print("[OK] Knowledge base loaded\n")
    
    # Query 1: Evidence-based drugs
    print("Query 1: Evidence-Based Drug Recommendations")
    print("=" * 80)
    
    drugs = kb_decision.get_evidence_based_drugs(
        cancer_type="colorectal",
        mutations=["KRAS"],
        min_evidence=1
    )
    
    print(f"\nFound {len(drugs)} evidence-based drugs:\n")
    
    for i, drug in enumerate(drugs[:5], 1):
        marker = "[MUTATION-MATCHED]" if drug.cancer_specific else "             "
        print(f"{i}. {marker} {drug.drug_name} ({drug.drug_class})")
        print(f"   Target: {drug.target}")
        print(f"   Mechanism: {drug.mechanism[:80]}...")
        print(f"   Evidence: {drug.evidence_level} papers")
        print(f"   PMIDs: {', '.join(drug.pmids[:3])}")
        print()
    
    print("\n" + "=" * 80)
    
    # Query 2: Drug combinations
    print("\nQuery 2: Drug Combination Insights")
    print("=" * 80)
    
    combos = kb_decision.get_combination_insights(
        cancer_type="colorectal"
    )
    
    print(f"\nFound {len(combos)} drug combinations:\n")
    
    for i, combo in enumerate(combos[:5], 1):
        print(f"{i}. {' + '.join(combo.drugs)}")
        print(f"   Synergy: {combo.synergy_type}")
        print(f"   Cancer: {combo.cancer_type}")
        print(f"   Evidence: {combo.evidence[:100]}...")
        print(f"   PMID: {combo.pmid}")
        print()
    
    print("\n" + "=" * 80)
    
    # Query 3: Comprehensive recommendation
    print("\nQuery 3: Comprehensive Treatment Recommendation")
    print("=" * 80)
    
    recommendation = kb_decision.enhance_treatment_recommendation(
        cancer_type="colorectal",
        mutations=["KRAS", "TP53"],
        current_therapies=["FOLFOX"]
    )
    
    if recommendation['kb_available']:
        print(f"\nCancer Type: {recommendation['cancer_type']}")
        print(f"Known Mutations: {', '.join(recommendation['mutations'])}")
        print()
        
        print("Primary Drug Recommendations:")
        for i, drug in enumerate(recommendation['evidence_drugs'][:3], 1):
            print(f"  {i}. {drug.drug_name} (Target: {drug.target}) - {drug.paper_count} papers")
        print()
        
        print("Combination Therapies:")
        for i, combo in enumerate(recommendation['combination_therapies'][:3], 1):
            print(f"  {i}. {' + '.join(combo.drugs)} ({combo.synergy_type})")
        print()
        
        print(f"Evidence Base: {recommendation['evidence_base']['total_papers']} papers")
        print(f"Cancer-Specific Papers: {recommendation['evidence_base']['cancer_specific_papers']}")
    
    print("\n" + "=" * 80)
    print("Demo Complete!")
    print("=" * 80)


def demo_breast_her2_patient():
    """Demo: Breast cancer patient with HER2+"""
    
    print("\n\n")
    print("=" * 80)
    print("Demo 2: Breast Cancer (HER2+)")
    print("=" * 80)
    print()
    
    print("Patient Profile:")
    print("-" * 40)
    print("  Diagnosis: Breast Cancer")
    print("  Stage: T2N0M0")
    print("  HER2: Positive")
    print("  ER/PR: Positive")
    print()
    
    kb_decision = get_kb_decision()
    
    # Query for breast cancer drugs
    drugs = kb_decision.get_evidence_based_drugs(
        cancer_type="breast",
        mutations=[],
        min_evidence=1
    )
    
    print(f"Found {len(drugs)} evidence-based drugs for breast cancer:\n")
    
    for i, drug in enumerate(drugs[:5], 1):
        print(f"{i}. {drug.drug_name} ({drug.drug_class})")
        print(f"   Target: {drug.target}")
        print(f"   Evidence: {drug.evidence_level} papers")
        print()
    
    print("=" * 80)


if __name__ == "__main__":
    # Run demos
    demo_colorectal_kras_patient()
    demo_breast_her2_patient()
    
    print("\n[SUCCESS] All demos completed successfully!")
