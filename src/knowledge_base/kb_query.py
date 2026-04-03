"""
ADDS Knowledge Base Integration Module
=======================================
Integrate consolidated cancer knowledge base with ADDS CDSS system.

Features:
- Query drugs by cancer type
- Find drug combinations
- Search mechanisms and biomarkers
- Recommend treatment strategies
- Clinical decision support

Author: ADDS Team
Date: 2026-01-31
"""

import json
from pathlib import Path
from typing import List, Dict, Optional, Set
from dataclasses import dataclass


# Paths
KB_DIR = Path("data/knowledge_base")
CONSOLIDATED_KB = KB_DIR / "cancer_knowledge_base.json"
DRUG_INDEX = KB_DIR / "drug_index.json"
MECHANISM_INDEX = KB_DIR / "mechanism_index.json"
BIOMARKER_INDEX = KB_DIR / "biomarker_index.json"


@dataclass
class DrugRecommendation:
    """Drug recommendation with evidence"""
    drug_name: str
    drug_class: str
    mechanism: str
    target: str
    cancer_types: List[str]
    paper_count: int
    pmids: List[str]


@dataclass
class CombinationRecommendation:
    """Drug combination recommendation"""
    drugs: List[str]
    synergy_type: str
    cancer_type: str
    evidence: str
    pmid: str


class KnowledgeBaseQuery:
    """Query interface for cancer knowledge base"""
    
    def __init__(self):
        self.kb = None
        self.drug_index = None
        self.mechanism_index = None
        self.biomarker_index = None
        self.load_knowledge_base()
    
    def load_knowledge_base(self):
        """Load all knowledge base files"""
        print("[KB] Loading knowledge base...")
        
        try:
            with open(CONSOLIDATED_KB, 'r', encoding='utf-8') as f:
                self.kb = json.load(f)
            
            with open(DRUG_INDEX, 'r', encoding='utf-8') as f:
                self.drug_index = json.load(f)
            
            with open(MECHANISM_INDEX, 'r', encoding='utf-8') as f:
                self.mechanism_index = json.load(f)
            
            with open(BIOMARKER_INDEX, 'r', encoding='utf-8') as f:
                self.biomarker_index = json.load(f)
            
            print(f"[KB] Loaded {len(self.drug_index)} drugs, {len(self.mechanism_index)} mechanisms")
            
        except Exception as e:
            print(f"[ERROR] Failed to load knowledge base: {e}")
            raise
    
    def search_drugs_by_cancer_type(
        self,
        cancer_type: str,
        min_papers: int = 1
    ) -> List[DrugRecommendation]:
        """Find drugs effective for specific cancer type"""
        
        cancer_type_lower = cancer_type.lower()
        results = []
        
        for drug_key, drug_info in self.drug_index.items():
            # Check if cancer type matches
            matching_types = [
                ct for ct in drug_info.get('cancer_types', [])
                if cancer_type_lower in ct.lower() or 'pan-cancer' in ct.lower()
            ]
            
            if not matching_types:
                continue
            
            # Get PMIDs for this cancer type
            relevant_pmids = [
                p['pmid'] for p in drug_info.get('papers', [])
                if any(cancer_type_lower in p['cancer_type'].lower() 
                       or 'pan-cancer' in p['cancer_type'].lower()
                       for p in drug_info.get('papers', []))
            ]
            
            if len(relevant_pmids) < min_papers:
                continue
            
            results.append(DrugRecommendation(
                drug_name=drug_info['name'],
                drug_class=drug_info.get('class', 'Unknown'),
                mechanism=drug_info.get('mechanism', 'Unknown'),
                target=drug_info.get('target', 'Unknown'),
                cancer_types=matching_types,
                paper_count=len(relevant_pmids),
                pmids=relevant_pmids
            ))
        
        # Sort by paper count (evidence strength)
        results.sort(key=lambda x: x.paper_count, reverse=True)
        
        return results
    
    def find_drug_combinations(
        self,
        cancer_type: Optional[str] = None,
        include_drug: Optional[str] = None
    ) -> List[CombinationRecommendation]:
        """Find synergistic drug combinations"""
        
        combinations = []
        
        for paper in self.kb.get('papers', []):
            paper_cancer = paper.get('paper_summary', {}).get('cancer_type', '')
            
            # Filter by cancer type if specified
            if cancer_type:
                if cancer_type.lower() not in paper_cancer.lower():
                    continue
            
            # Extract combinations
            for combo in paper.get('drug_combinations', []):
                drugs = combo.get('combination', [])
                
                # Filter by specific drug if specified
                if include_drug:
                    if not any(include_drug.lower() in drug.lower() for drug in drugs):
                        continue
                
                combinations.append(CombinationRecommendation(
                    drugs=drugs,
                    synergy_type=combo.get('synergy_type', 'unknown'),
                    cancer_type=paper_cancer,
                    evidence=combo.get('evidence', ''),
                    pmid=paper['source']['pmid']
                ))
        
        return combinations
    
    def search_mechanisms_by_target(
        self,
        target_protein: str
    ) -> List[Dict]:
        """Find mechanisms involving specific protein"""
        
        results = []
        target_lower = target_protein.lower()
        
        for pathway, mech_info in self.mechanism_index.items():
            proteins = [p.lower() for p in mech_info.get('proteins', [])]
            
            if target_lower in proteins or any(target_lower in p for p in proteins):
                results.append({
                    'pathway': pathway,
                    'categories': mech_info.get('categories', []),
                    'proteins': mech_info.get('proteins', []),
                    'descriptions': mech_info.get('descriptions', []),
                    'cancer_types': mech_info.get('cancer_types', []),
                    'paper_count': mech_info.get('paper_count', 0),
                    'pmids': [p['pmid'] for p in mech_info.get('papers', [])]
                })
        
        results.sort(key=lambda x: x['paper_count'], reverse=True)
        
        return results
    
    def find_biomarkers_for_cancer(
        self,
        cancer_type: str
    ) -> List[Dict]:
        """Find relevant biomarkers for cancer type"""
        
        results = []
        cancer_lower = cancer_type.lower()
        
        for biomarker, bio_info in self.biomarker_index.items():
            cancer_types = bio_info.get('cancer_types', [])
            
            if any(cancer_lower in ct.lower() or 'pan-cancer' in ct.lower() 
                   for ct in cancer_types):
                results.append({
                    'name': biomarker,
                    'types': bio_info.get('types', []),
                    'predictive_values': bio_info.get('predictive_values', []),
                    'cancer_types': cancer_types,
                    'paper_count': bio_info.get('paper_count', 0)
                })
        
        results.sort(key=lambda x: x['paper_count'], reverse=True)
        
        return results
    
    def generate_treatment_recommendation(
        self,
        cancer_type: str,
        known_mutations: Optional[List[str]] = None,
        avoid_drugs: Optional[List[str]] = None
    ) -> Dict:
        """
        Generate comprehensive treatment recommendation
        
        Args:
            cancer_type: Type of cancer (e.g., "colorectal", "breast")
            known_mutations: List of known mutations (e.g., ["KRAS", "BRAF V600"])
            avoid_drugs: List of drugs to avoid (contraindications)
        
        Returns:
            Comprehensive treatment recommendation with evidence
        """
        
        avoid_drugs = avoid_drugs or []
        known_mutations = known_mutations or []
        
        # Find relevant drugs
        drug_recommendations = self.search_drugs_by_cancer_type(cancer_type)
        
        # Filter out contraindicated drugs
        if avoid_drugs:
            avoid_lower = [d.lower() for d in avoid_drugs]
            drug_recommendations = [
                d for d in drug_recommendations
                if not any(avoid.lower() in d.drug_name.lower() for avoid in avoid_lower)
            ]
        
        # Find combinations
        combinations = self.find_drug_combinations(cancer_type)
        
        # Find relevant biomarkers
        biomarkers = self.find_biomarkers_for_cancer(cancer_type)
        
        # Match biomarkers with known mutations
        relevant_biomarkers = []
        if known_mutations:
            for bio in biomarkers:
                if any(mut.lower() in bio['name'].lower() for mut in known_mutations):
                    relevant_biomarkers.append(bio)
        
        return {
            'cancer_type': cancer_type,
            'known_mutations': known_mutations,
            'primary_drugs': drug_recommendations[:10],  # Top 10
            'combination_therapies': combinations[:5],    # Top 5
            'relevant_biomarkers': relevant_biomarkers,
            'evidence_base': {
                'total_papers': len(self.kb.get('papers', [])),
                'cancer_specific_papers': len([
                    p for p in self.kb.get('papers', [])
                    if cancer_type.lower() in p.get('paper_summary', {}).get('cancer_type', '').lower()
                ])
            }
        }


def demo_query_interface():
    """Demonstrate query capabilities"""
    
    print("="*70)
    print(" ADDS KNOWLEDGE BASE - DEMO QUERY INTERFACE")
    print("="*70)
    
    # Initialize query interface
    kb_query = KnowledgeBaseQuery()
    
    # Example 1: Search drugs for colorectal cancer
    print("\n[QUERY 1] Drugs for Colorectal Cancer")
    print("-"*70)
    drugs = kb_query.search_drugs_by_cancer_type("colorectal", min_papers=2)
    for i, drug in enumerate(drugs[:5], 1):
        print(f"{i}. {drug.drug_name} ({drug.drug_class})")
        print(f"   Target: {drug.target}")
        print(f"   Evidence: {drug.paper_count} papers")
    
    # Example 2: Find drug combinations
    print("\n[QUERY 2] Drug Combinations for Colorectal Cancer")
    print("-"*70)
    combos = kb_query.find_drug_combinations("colorectal")
    for i, combo in enumerate(combos[:3], 1):
        print(f"{i}. {' + '.join(combo.drugs)}")
        print(f"   Synergy: {combo.synergy_type}")
        print(f"   Evidence: {combo.evidence[:100]}...")
    
    # Example 3: Search mechanisms
    print("\n[QUERY 3] Mechanisms Involving KRAS")
    print("-"*70)
    mechs = kb_query.search_mechanisms_by_target("KRAS")
    for i, mech in enumerate(mechs[:3], 1):
        print(f"{i}. {mech['pathway']}")
        print(f"   Proteins: {', '.join(mech['proteins'][:5])}")
        print(f"   Papers: {mech['paper_count']}")
    
    # Example 4: Comprehensive recommendation
    print("\n[QUERY 4] Comprehensive Treatment Recommendation")
    print("-"*70)
    recommendation = kb_query.generate_treatment_recommendation(
        cancer_type="colorectal",
        known_mutations=["KRAS", "BRAF"],
        avoid_drugs=[]
    )
    
    print(f"Cancer Type: {recommendation['cancer_type']}")
    print(f"Known Mutations: {', '.join(recommendation['known_mutations'])}")
    print(f"\nTop Recommended Drugs:")
    for i, drug in enumerate(recommendation['primary_drugs'][:3], 1):
        print(f"  {i}. {drug.drug_name} - {drug.drug_class}")
    
    print(f"\nTop Combination Therapies:")
    for i, combo in enumerate(recommendation['combination_therapies'][:2], 1):
        print(f"  {i}. {' + '.join(combo.drugs)}")
    
    print(f"\nEvidence Base:")
    print(f"  Total Papers: {recommendation['evidence_base']['total_papers']}")
    print(f"  Cancer-Specific: {recommendation['evidence_base']['cancer_specific_papers']}")
    
    print("\n" + "="*70)
    print(" DEMO COMPLETE")
    print("="*70)
    print("\nKnowledge base is ready for CDSS integration!")
    print()


if __name__ == "__main__":
    demo_query_interface()
