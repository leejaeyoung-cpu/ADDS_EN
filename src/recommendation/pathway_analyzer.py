"""
Signal Pathway Analyzer
Identifies active pathways from genomic and expression data
Designs multi-target combination strategies
"""

import json
from typing import Dict, List, Set, Tuple
from pathlib import Path


class PathwayAnalyzer:
    """
    Analyzes patient genomic/expression data to identify active pathways
    and recommend multi-target combination strategies
    """
    
    def __init__(self, pathway_db_path: str = None):
        if pathway_db_path is None:
            # Default path
            base_path = Path(__file__).parent.parent / "knowledge" / "pathways"
            pathway_db_path = str(base_path / "pathway_database.json")
        
        with open(pathway_db_path, 'r', encoding='utf-8') as f:
            self.pathway_db = json.load(f)
        
        self.pathways = self.pathway_db['pathways']
        self.crosstalk = self.pathway_db.get('pathway_crosstalk', {})
    
    def identify_active_pathways(
        self,
        genomic_variants: List[Dict],
        expression_data: Dict[str, float] = None
    ) -> Dict[str, Dict]:
        """
        Identify which signaling pathways are likely activated
        
        Args:
            genomic_variants: List of genomic alterations
                Format: [{'gene_name': 'KRAS', 'variant_detail': 'G12C'}, ...]
            expression_data: Gene expression data (optional)
                Format: {'EGFR': 0.8, 'HER2': 1.2, ...}
        
        Returns:
            Dictionary of active pathways with evidence
        """
        active_pathways = {}
        
        for pathway_id, pathway_info in self.pathways.items():
            evidence = self._assess_pathway_activation(
                pathway_id,
                pathway_info,
                genomic_variants,
                expression_data
            )
            
            if evidence['activation_score'] > 0:
                active_pathways[pathway_id] = evidence
        
        return active_pathways
    
    def _assess_pathway_activation(
        self,
        pathway_id: str,
        pathway_info: Dict,
        genomic_variants: List[Dict],
        expression_data: Dict[str, float] = None
    ) -> Dict:
        """
        Assess evidence for pathway activation
        """
        evidence = {
            'activation_score': 0.0,
            'genomic_evidence': [],
            'expression_evidence': [],
            'confidence': 'Low'
        }
        
        # Check genomic evidence
        pathway_proteins = pathway_info.get('key_proteins', [])
        known_alterations = pathway_info.get('alterations_in_cancer', [])
        
        for variant in genomic_variants:
            gene = variant.get('gene_name', '')
            
            # Direct match with pathway proteins
            if gene in pathway_proteins:
                evidence['genomic_evidence'].append({
                    'gene': gene,
                    'variant': variant.get('variant_detail', ''),
                    'impact': 'Direct pathway member'
                })
                evidence['activation_score'] += 0.5
            
            # Check if variant is known driver for this pathway
            for alteration in known_alterations:
                if gene in alteration:
                    evidence['genomic_evidence'].append({
                        'gene': gene,
                        'known_alteration': alteration
                    })
                    evidence['activation_score'] += 0.8
        
        # Check expression evidence (if available)
        if expression_data:
            for protein in pathway_proteins:
                if protein in expression_data:
                    expr_level = expression_data[protein]
                    if expr_level > 1.5:  # Overexpression
                        evidence['expression_evidence'].append({
                            'gene': protein,
                            'level': expr_level,
                            'status': 'Overexpressed'
                        })
                        evidence['activation_score'] += 0.3
        
        # Determine confidence
        if evidence['activation_score'] >= 1.0:
            evidence['confidence'] = 'High'
        elif evidence['activation_score'] >= 0.5:
            evidence['confidence'] = 'Moderate'
        
        return evidence
    
    def design_multi_target_strategy(
        self,
        active_pathways: Dict[str, Dict],
        drug_target_map: Dict = None
    ) -> Dict:
        """
        Design optimal multi-pathway inhibition strategy
        
        Args:
            active_pathways: Output from identify_active_pathways()
            drug_target_map: Optional drug-target mapping
        
        Returns:
            Recommended combination strategy
        """
        # Load drug-target mapping if not provided
        if drug_target_map is None:
            base_path = Path(__file__).parent.parent / "knowledge" / "pathways"
            map_path = str(base_path / "drug_target_mapping.json")
            with open(map_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                drug_target_map = data['drug_target_mapping']
        
        # Priority 1: Target primary driver pathway
        primary_pathway = self._identify_primary_driver(active_pathways)
        
        # Priority 2: Block bypass/resistance pathways
        resistance_pathways = self._identify_resistance_pathways(
            primary_pathway,
            active_pathways
        )
        
        # Priority 3: Consider pathway crosstalk
        crosstalk_pathways = self._identify_crosstalk(
            primary_pathway,
            list(resistance_pathways.keys())
        )
        
        # Select drugs
        recommended_drugs = self._select_optimal_drugs(
            primary_pathway,
            resistance_pathways,
            drug_target_map
        )
        
        strategy = {
            'primary_driver': primary_pathway,
            'secondary_targets': list(resistance_pathways.keys()),
            'crosstalk_considerations': crosstalk_pathways,
            'recommended_combination': recommended_drugs,
            'rationale': self._generate_strategy_rationale(
                primary_pathway,
                resistance_pathways,
                crosstalk_pathways
            )
        }
        
        return strategy
    
    def _identify_primary_driver(
        self,
        active_pathways: Dict[str, Dict]
    ) -> str:
        """Identify the most activated pathway as primary driver"""
        if not active_pathways:
            return None
        
        # Sort by activation score
        sorted_pathways = sorted(
            active_pathways.items(),
            key=lambda x: x[1]['activation_score'],
            reverse=True
        )
        
        return sorted_pathways[0][0]
    
    def _identify_resistance_pathways(
        self,
        primary_pathway: str,
        active_pathways: Dict[str, Dict]
    ) -> Dict:
        """
        Identify pathways that may serve as resistance mechanisms
        """
        if not primary_pathway:
            return {}
        
        resistance_pathways = {}
        
        # Known bypass pathways
        bypass_map = {
            'RTK_RAS_MAPK': ['PI3K_AKT_mTOR'],
            'PI3K_AKT_mTOR': ['RTK_RAS_MAPK'],
            'VEGF_Angiogenesis': ['Hippo_YAP_TAZ']
        }
        
        potential_bypass = bypass_map.get(primary_pathway, [])
        
        for pathway_id in potential_bypass:
            if pathway_id in active_pathways:
                resistance_pathways[pathway_id] = active_pathways[pathway_id]
        
        return resistance_pathways
    
    def _identify_crosstalk(
        self,
        primary_pathway: str,
        secondary_pathways: List[str]
    ) -> List[Dict]:
        """Identify relevant pathway crosstalk"""
        relevant_crosstalk = []
        
        for crosstalk_id, crosstalk_info in self.crosstalk.items():
            # Check if crosstalk involves primary or secondary pathways
            pathways_involved = crosstalk_id.split('_')
            
            if primary_pathway in crosstalk_id or any(p in crosstalk_id for p in secondary_pathways):
                relevant_crosstalk.append({
                    'crosstalk_id': crosstalk_id,
                    'description': crosstalk_info.get('description', ''),
                    'clinical_significance': crosstalk_info.get('clinical_significance', ''),
                    'combination_rationale': crosstalk_info.get('combination_rationale', '')
                })
        
        return relevant_crosstalk
    
    def _select_optimal_drugs(
        self,
        primary_pathway: str,
        resistance_pathways: Dict,
        drug_target_map: Dict
    ) -> List[Dict]:
        """Select optimal drug combination"""
        selected_drugs = []
        
        # Get therapeutic targets for primary pathway
        if primary_pathway and primary_pathway in self.pathways:
            targets = self.pathways[primary_pathway].get('therapeutic_targets', {})
            
            # Select first-line drug for primary target
            for target_name, target_info in targets.items():
                drugs = target_info.get('drugs', [])
                if drugs:
                    selected_drugs.append({
                        'drug': drugs[0],
                        'target': target_name,
                        'pathway': primary_pathway,
                        'role': 'Primary driver inhibition'
                    })
                    break  # Select only one for primary
        
        # Add drugs for resistance pathways
        for pathway_id in resistance_pathways.keys():
            if pathway_id in self.pathways:
                targets = self.pathways[pathway_id].get('therapeutic_targets', {})
                for target_name, target_info in targets.items():
                    drugs = target_info.get('drugs', [])
                    if drugs:
                        selected_drugs.append({
                            'drug': drugs[0],
                            'target': target_name,
                            'pathway': pathway_id,
                            'role': 'Resistance prevention'
                        })
                        break
        
        return selected_drugs
    
    def _generate_strategy_rationale(
        self,
        primary_pathway: str,
        resistance_pathways: Dict,
        crosstalk: List[Dict]
    ) -> str:
        """Generate human-readable rationale"""
        rationale_parts = []
        
        if primary_pathway:
            pathway_name = self.pathways[primary_pathway]['name']
            rationale_parts.append(
                f"**Primary Target**: {pathway_name} pathway shows strongest activation."
            )
        
        if resistance_pathways:
            pathway_names = [
                self.pathways[p]['name'] for p in resistance_pathways.keys()
            ]
            rationale_parts.append(
                f"**Resistance Prevention**: {', '.join(pathway_names)} pathway(s) "
                "targeted to prevent bypass resistance."
            )
        
        if crosstalk:
            rationale_parts.append(
                f"**Crosstalk Considerations**: {len(crosstalk)} pathway interactions identified."
            )
        
        return "\n\n".join(rationale_parts)
    
    def calculate_pathway_coverage(
        self,
        drug_combination: List[str]
    ) -> Dict:
        """
        Calculate how well a drug combination covers active pathways
        
        Args:
            drug_combination: List of drug names
        
        Returns:
            Coverage analysis
        """
        covered_pathways = set()
        drug_details = []
        
        # Load drug-target mapping
        base_path = Path(__file__).parent.parent / "knowledge" / "pathways"
        map_path = str(base_path / "drug_target_mapping.json")
        with open(map_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            drug_target_map = data['drug_target_mapping']
        
        for drug in drug_combination:
            if drug in drug_target_map:
                drug_info = drug_target_map[drug]
                pathways = drug_info.get('pathways', [])
                covered_pathways.update(pathways)
                
                drug_details.append({
                    'drug': drug,
                    'targets': drug_info.get('targets', []),
                    'pathways': pathways
                })
        
        coverage = {
            'total_drugs': len(drug_combination),
            'pathways_covered': list(covered_pathways),
            'coverage_count': len(covered_pathways),
            'drug_details': drug_details
        }
        
        return coverage


# Example usage
if __name__ == "__main__":
    analyzer = PathwayAnalyzer()
    
    # Example patient with KRAS mutation
    genomic_variants = [
        {'gene_name': 'KRAS', 'variant_detail': 'G12C'},
        {'gene_name': 'PIK3CA', 'variant_detail': 'H1047R'}
    ]
    
    # Identify active pathways
    active = analyzer.identify_active_pathways(genomic_variants)
    print("Active Pathways:")
    for pathway, evidence in active.items():
        print(f"  {pathway}: {evidence['activation_score']:.2f} ({evidence['confidence']})")
    
    # Design strategy
    strategy = analyzer.design_multi_target_strategy(active)
    print("\nRecommended Strategy:")
    print(f"Primary: {strategy['primary_driver']}")
    print(f"Combination: {[d['drug'] for d in strategy['recommended_combination']]}")
