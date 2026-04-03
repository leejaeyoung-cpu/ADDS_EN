"""
Enhanced Evidence-based Drug Combination Optimizer
Integrates signal pathway analysis and synergy calculation
"""

from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
import json
from pathlib import Path

# Import new modules
from .pathway_analyzer import PathwayAnalyzer
from .synergy_calculator import SynergyCalculator


class EnhancedDrugCombinationOptimizer:
    """
    Next-generation drug combination optimizer
    
    Enhancements over legacy version:
    - Signal pathway-based drug selection
    - Quantitative synergy scoring (Bliss, Loewe, HSA, ZIP)
    - Multi-pathway inhibition strategies
    - Evidence from Nature-level publications
    """
    
    def __init__(self):
        # Initialize new components
        self.pathway_analyzer = PathwayAnalyzer()
        self.synergy_calculator = SynergyCalculator()
        
        # Load knowledge bases
        self._load_knowledge_bases()
        
        # Load literature features (NEW) - always initialize has_literature first
        self.has_literature = False
        self.literature_loader = None
        
        try:
            from ..knowledge.literature_loader import LiteratureFeatureLoader
            self.literature_loader = LiteratureFeatureLoader()
            self.has_literature = True
        except Exception as e:
            # Silently fail if literature not available
            pass
        
        # Legacy components (for compatibility)
        self.protocol_database = self._initialize_protocols()
        self.targeted_therapy_map = self._initialize_targeted_therapies()
    
    def _load_knowledge_bases(self):
        """Load pathway and drug databases"""
        base_path = Path(__file__).parent.parent / "knowledge" / "pathways"
        
        # Load pathway database
        with open(base_path / "pathway_database.json", 'r', encoding='utf-8') as f:
            pathway_data = json.load(f)
            self.pathways = pathway_data['pathways']
            self.pathway_crosstalk = pathway_data.get('pathway_crosstalk', {})
        
        # Load drug-target mapping
        with open(base_path / "drug_target_mapping.json", 'r', encoding='utf-8') as f:
            drug_data = json.load(f)
            self.drug_target_map = drug_data['drug_target_mapping']
            self.known_combinations = drug_data.get('synergistic_combinations', [])
    
    def recommend_regimen(
        self,
        cohort_classification: Dict,
        quantitative_results: Dict,
        clinical_profile: Dict,
        genomic_variants: List[Dict]
    ) -> Dict:
        """
        Enhanced regimen recommendation with pathway analysis
        
        Args:
            cohort_classification: Patient cohort info
            quantitative_results: Quantitative pathology results
            clinical_profile: Clinical data
            genomic_variants: Genomic alterations
            
        Returns:
            Comprehensive recommendation with pathway-based rationale
        """
        # Step 1: Identify active signaling pathways
        active_pathways = self.pathway_analyzer.identify_active_pathways(
            genomic_variants
        )
        
        # Step 2: Design multi-pathway inhibition strategy
        pathway_strategy = self.pathway_analyzer.design_multi_target_strategy(
            active_pathways,
            self.drug_target_map
        )
        
        # Step 3: Get guideline-based protocols (legacy compatibility)
        cohort_name = cohort_classification.get('cohort_name', '')
        cancer_type = clinical_profile.get('cancer_type', 'Colorectal')
        base_protocols = self._get_cohort_protocols(cohort_name, cancer_type)
        
        # Step 4: Integrate pathway-based drugs with protocols
        enhanced_regimen = self._integrate_pathway_strategy(
            base_protocols,
            pathway_strategy,
            quantitative_results
        )
        
        # Step 5: Calculate synergy scores if applicable
        synergy_analysis = self._analyze_combination_synergy(
            enhanced_regimen
        )
        
        # Step 6: Generate comprehensive evidence
        evidence = self._generate_enhanced_evidence(
            enhanced_regimen,
            active_pathways,
            pathway_strategy,
            synergy_analysis,
            cohort_classification,
            quantitative_results,
            clinical_profile,
            genomic_variants
        )
        
        # Step 7: Build alternatives
        alternatives = self._build_enhanced_alternatives(
            base_protocols,
            pathway_strategy,
            genomic_variants
        )
        
        # Step 8: Generate warnings
        warnings = self._generate_warnings(enhanced_regimen, clinical_profile)
        
        return {
            'primary_regimen': enhanced_regimen,
            'alternative_regimens': alternatives,
            'active_pathways': active_pathways,
            'pathway_strategy': pathway_strategy,
            'synergy_analysis': synergy_analysis,
            'evidence_summary': evidence,
            'confidence_level': self._assess_confidence(evidence),
            'warnings': warnings
        }
    
    def _integrate_pathway_strategy(
        self,
        base_protocols: List[Dict],
        pathway_strategy: Dict,
        quant_results: Dict
    ) -> Dict:
        """
        Integrate pathway-based drug selection with clinical protocols
        """
        # Start with pathway-recommended drugs
        recommended_drugs = pathway_strategy.get('recommended_combination', [])
        
        # Get base protocol
        if base_protocols:
            base_regimen = base_protocols[0].copy()
        else:
            base_regimen = {
                'name': 'Pathway-Guided Precision Therapy',
                'drugs': [],
                'intensity': 'Targeted',
                'evidence_level': 'Pathway-Based'
            }
        
        # Add pathway-recommended drugs
        for drug_info in recommended_drugs:
            drug_name = drug_info['drug']
            if drug_name not in base_regimen['drugs']:
                base_regimen['drugs'].append(drug_name)
        
        # Adjust based on quantitative features
        if quant_results.get('overall_heterogeneity', 0) > 0.7:
            # High heterogeneity → prefer multi-drug combinations
            base_regimen['heterogeneity_adjusted'] = True
        
        if quant_results.get('clustered_ratio', 0) > 0.6:
            # High clustering → consider anti-angiogenic
            if 'Bevacizumab' not in base_regimen['drugs']:
                base_regimen['drugs'].append('Bevacizumab')
                base_regimen['anti_angiogenic_added'] = True
        
        # Add pathway information
        base_regimen['pathway_targets'] = recommended_drugs
        base_regimen['primary_pathway'] = pathway_strategy.get('primary_driver', 'Unknown')
        
        return base_regimen
    
    def _analyze_combination_synergy(self, regimen: Dict) -> Dict:
        """
        Analyze synergy for drug combinations using known data
        """
        drugs = regimen.get('drugs', [])
        
        if len(drugs) < 2:
            return {'synergy_applicable': False}
        
        synergy_results = []
        
        # Check against known synergistic combinations
        for combo in self.known_combinations:
            combo_drugs = combo['combination']
            
            # Check if this combination is in our regimen
            if all(drug in drugs for drug in combo_drugs):
                synergy_results.append({
                    'drugs': combo_drugs,
                    'mechanism': combo['synergy_mechanism'],
                    'evidence_level': combo['evidence_level'],
                    'indication': combo['indications'],
                    'references': combo.get('references', []),
                    'type': 'Known Synergy (FDA-approved or Clinical Trial)'
                })
        
        # For novel combinations, provide theoretical synergy based on pathways
        pathway_targets = regimen.get('pathway_targets', [])
        if len(pathway_targets) >= 2:
            theoretical_synergy = self._assess_theoretical_synergy(pathway_targets)
            if theoretical_synergy:
                synergy_results.append(theoretical_synergy)
        
        return {
            'synergy_applicable': True,
            'synergy_count': len(synergy_results),
            'synergies': synergy_results
        }
    
    def _assess_theoretical_synergy(self, pathway_targets: List[Dict]) -> Dict:
        """
        Assess theoretical synergy based on pathway interactions
        """
        if len(pathway_targets) < 2:
            return None
        
        pathways_involved = [pt['pathway'] for pt in pathway_targets]
        
        # Check pathway crosstalk
        for crosstalk_id, crosstalk_info in self.pathway_crosstalk.items():
            if any(pathway in crosstalk_id for pathway in pathways_involved):
                return {
                    'type': 'Theoretical (Pathway Crosstalk)',
                    'pathways': pathways_involved,
                    'mechanism': crosstalk_info.get('description', ''),
                    'rationale': crosstalk_info.get('combination_rationale', ''),
                    'evidence_level': 'Preclinical/Theoretical'
                }
        
        return None
    
    def _generate_enhanced_evidence(
        self,
        regimen: Dict,
        active_pathways: Dict,
        pathway_strategy: Dict,
        synergy_analysis: Dict,
        cohort: Dict,
        quant_results: Dict,
        clinical: Dict,
        genomic: List[Dict]
    ) -> Dict:
        """
        Generate comprehensive evidence summary
        """
        evidence = {
            # Pathway-based evidence (NEW)
            'pathway_analysis': {
                'active_pathways': list(active_pathways.keys()),
                'primary_driver': pathway_strategy.get('primary_driver', 'Unknown'),
                'strategy_rationale': pathway_strategy.get('rationale', ''),
                'pathway_coverage': self.pathway_analyzer.calculate_pathway_coverage(
                    regimen.get('drugs', [])
                )
            },
            
            # Synergy evidence (NEW)
            'synergy_evidence': synergy_analysis,
            
            # Legacy evidence (enhanced)
            'cohort_based_rationale': self._explain_cohort_match(cohort, regimen),
            'quantitative_indicators': self._explain_quantitative_rationale(quant_results, regimen),
            'clinical_guidelines': self._cite_clinical_guidelines(regimen, clinical),
            'genomic_rationale': self._explain_genomic_basis(genomic, regimen),
            'supporting_literature': self._add_literature_references(regimen)
        }
        
        return evidence
    
    def _build_enhanced_alternatives(
        self,
        base_protocols: List[Dict],
        pathway_strategy: Dict,
        genomic_variants: List[Dict]
    ) -> List[Dict]:
        """Build alternative regimens"""
        alternatives = []
        
        #  Alternative 1: Different pathway targets
        secondary_pathways = pathway_strategy.get('secondary_targets', [])
        if secondary_pathways:
            for pathway_id in secondary_pathways[:2]:  # Top 2 alternatives
                if pathway_id in self.pathways:
                    targets = self.pathways[pathway_id].get('therapeutic_targets', {})
                    for target_name, target_info in targets.items():
                        drugs = target_info.get('drugs', [])
                        if drugs:
                            alternatives.append({
                                'name': f'Alternative: {pathway_id} Targeting',
                                'drugs': drugs[:1],  # First drug
                                'rationale': f'Alternative pathway inhibition strategy',
                                'pathway': pathway_id
                            })
                            break
        
        # Alternative 2: From base protocols
        for protocol in base_protocols[1:3]:
            alt = self._integrate_targeted_therapies(protocol, genomic_variants)
            alternatives.append(alt)
        
        return alternatives[:3]  # Return top 3
    
    # ===== Legacy methods (kept for compatibility) =====
    
    def _initialize_protocols(self) -> Dict:
        """Legacy protocol database"""
        return {
            'Colorectal': {
                'High-Risk Aggressive': [
                    {
                        'name': 'FOLFOXIRI + Bevacizumab',
                        'drugs': ['5-FU', 'Leucovorin', 'Oxaliplatin', 'Irinotecan', 'Bevacizumab'],
                        'intensity': 'High',
                        'evidence_level': 'Level I'
                    }
                ],
                'Intermediate-Risk Standard': [
                    {
                        'name': 'FOLFOX',
                        'drugs': ['5-FU', 'Leucovorin', 'Oxaliplatin'],
                        'intensity': 'Moderate',
                        'evidence_level': 'Level I'
                    }
                ]
            },
            'Prostate': {
                'High-Risk Aggressive': [
                    {
                        'name': 'Docetaxel + ADT + Abiraterone',
                        'drugs': ['Docetaxel', 'Leuprolide', 'Abiraterone', 'Prednisone'],
                        'intensity': 'High',
                        'evidence_level': 'Level I'
                    }
                ],
                'Intermediate-Risk Standard': [
                    {
                        'name': 'ADT + Abiraterone',
                        'drugs': ['Leuprolide', 'Abiraterone', 'Prednisone'],
                        'intensity': 'Moderate',
                        'evidence_level': 'Level I'
                    }
                ],
                'Low-Risk Hormone-Sensitive': [
                    {
                        'name': 'ADT Monotherapy',
                        'drugs': ['Leuprolide'],
                        'intensity': 'Low',
                        'evidence_level': 'Level I'
                    }
                ]
            },
            'Gastric': {
                'High-Risk Aggressive': [
                    {
                        'name': 'FLOT + Trastuzumab (HER2+)',
                        'drugs': ['5-FU', 'Leucovorin', 'Oxaliplatin', 'Docetaxel', 'Trastuzumab'],
                        'intensity': 'High',
                        'evidence_level': 'Level I',
                        'indication': 'HER2-positive metastatic gastric cancer'
                    },
                    {
                        'name': 'FOLFOX + Nivolumab',
                        'drugs': ['5-FU', 'Leucovorin', 'Oxaliplatin', 'Nivolumab'],
                        'intensity': 'High',
                        'evidence_level': 'Level I',
                        'indication': 'PD-L1 positive or MSI-H'
                    }
                ],
                'Intermediate-Risk Standard': [
                    {
                        'name': 'XELOX',
                        'drugs': ['Capecitabine', 'Oxaliplatin'],
                        'intensity': 'Moderate',
                        'evidence_level': 'Level I'
                    }
                ]
            },
            'Ovarian': {
                'High-Risk Aggressive': [
                    {
                        'name': 'Carboplatin + Paclitaxel + Bevacizumab',
                        'drugs': ['Carboplatin', 'Paclitaxel', 'Bevacizumab'],
                        'intensity': 'High',
                        'evidence_level': 'Level I'
                    }
                ],
                'Intermediate-Risk Standard': [
                    {
                        'name': 'Carboplatin + Paclitaxel',
                        'drugs': ['Carboplatin', 'Paclitaxel'],
                        'intensity': 'Moderate',
                        'evidence_level': 'Level I'
                    }
                ],
                'BRCA-Mutated Maintenance': [
                    {
                        'name': 'Olaparib Maintenance',
                        'drugs': ['Olaparib'],
                        'intensity': 'Low',
                        'evidence_level': 'Level I',
                        'indication': 'BRCA1/2 mutation, platinum-responsive'
                    }
                ]
            },
            'Liver': {
                'High-Risk Aggressive': [
                    {
                        'name': 'Atezolizumab + Bevacizumab',
                        'drugs': ['Atezolizumab', 'Bevacizumab'],
                        'intensity': 'High',
                        'evidence_level': 'Level I',
                        'indication': 'Unresectable HCC, first-line'
                    }
                ],
                'Intermediate-Risk Standard': [
                    {
                        'name': 'Lenvatinib',
                        'drugs': ['Lenvatinib'],
                        'intensity': 'Moderate',
                        'evidence_level': 'Level I'
                    },
                    {
                        'name': 'Sorafenib',
                        'drugs': ['Sorafenib'],
                        'intensity': 'Moderate',
                        'evidence_level': 'Level I'
                    }
                ]
            },
            'Bladder': {
                'High-Risk Aggressive': [
                    {
                        'name': 'Cisplatin + Gemcitabine',
                        'drugs': ['Cisplatin', 'Gemcitabine'],
                        'intensity': 'High',
                        'evidence_level': 'Level I',
                        'indication': 'Cisplatin-eligible metastatic'
                    }
                ],
                'Intermediate-Risk Standard': [
                    {
                        'name': 'Carboplatin + Gemcitabine',
                        'drugs': ['Carboplatin', 'Gemcitabine'],
                        'intensity': 'Moderate',
                        'evidence_level': 'Level I',
                        'indication': 'Cisplatin-ineligible'
                    }
                ],
                'Immunotherapy': [
                    {
                        'name': 'Pembrolizumab',
                        'drugs': ['Pembrolizumab'],
                        'intensity': 'Moderate',
                        'evidence_level': 'Level I',
                        'indication': 'PD-L1 positive or platinum-refractory'
                    }
                ]
            },
            'Renal': {
                'High-Risk Aggressive': [
                    {
                        'name': 'Nivolumab + Ipilimumab',
                        'drugs': ['Nivolumab', 'Ipilimumab'],
                        'intensity': 'High',
                        'evidence_level': 'Level I',
                        'indication': 'Intermediate/poor-risk clear cell RCC'
                    },
                    {
                        'name': 'Pembrolizumab + Lenvatinib',
                        'drugs': ['Pembrolizumab', 'Lenvatinib'],
                        'intensity': 'High',
                        'evidence_level': 'Level I'
                    }
                ],
                'Intermediate-Risk Standard': [
                    {
                        'name': 'Sunitinib',
                        'drugs': ['Sunitinib'],
                        'intensity': 'Moderate',
                        'evidence_level': 'Level I'
                    },
                    {
                        'name': 'Pazopanib',
                        'drugs': ['Pazopanib'],
                        'intensity': 'Moderate',
                        'evidence_level': 'Level I'
                    }
                ]
            },
            'Esophageal': {
                'High-Risk Aggressive': [
                    {
                        'name': 'FLOT + Nivolumab',
                        'drugs': ['5-FU', 'Leucovorin', 'Oxaliplatin', 'Docetaxel', 'Nivolumab'],
                        'intensity': 'High',
                        'evidence_level': 'Level I',
                        'indication': 'PD-L1 positive or MSI-H'
                    },
                    {
                        'name': 'Carboplatin + Paclitaxel + Trastuzumab',
                        'drugs': ['Carboplatin', 'Paclitaxel', 'Trastuzumab'],
                        'intensity': 'High',
                        'evidence_level': 'Level I',
                        'indication': 'HER2-positive adenocarcinoma'
                    }
                ],
                'Intermediate-Risk Standard': [
                    {
                        'name': 'Carboplatin + Paclitaxel',
                        'drugs': ['Carboplatin', 'Paclitaxel'],
                        'intensity': 'Moderate',
                        'evidence_level': 'Level I'
                    }
                ]
            },
            'Head_Neck': {
                'High-Risk Aggressive': [
                    {
                        'name': 'Cisplatin + Radiation + Cetuximab',
                        'drugs': ['Cisplatin', 'Cetuximab'],
                        'intensity': 'High',
                        'evidence_level': 'Level I',
                        'indication': 'Locally advanced disease'
                    },
                    {
                        'name': 'Pembrolizumab + Chemotherapy',
                        'drugs': ['Pembrolizumab', 'Cisplatin', '5-FU'],
                        'intensity': 'High',
                        'evidence_level': 'Level I',
                        'indication': 'Recurrent/metastatic HNSCC'
                    }
                ],
                'Intermediate-Risk Standard': [
                    {
                        'name': 'Cisplatin + Radiation',
                        'drugs': ['Cisplatin'],
                        'intensity': 'Moderate',
                        'evidence_level': 'Level I'
                    }
                ]
            }
        }
    
    def _initialize_targeted_therapies(self) -> Dict:
        """Legacy targeted therapy map"""
        return {
            # Pan-cancer biomarkers
            'EGFR': ['Cetuximab', 'Panitumumab', 'Erlotinib', 'Gefitinib'],
            'HER2': ['Trastuzumab', 'Pertuzumab', 'Trastuzumab-Deruxtecan'],
            'BRAF V600E': ['Encorafenib + Cetuximab', 'Dabrafenib + Trametinib'],
            'AR': ['Enzalutamide', 'Apalutamide', 'Darolutamide'],
            'BRCA1': ['Olaparib', 'Rucaparib', 'Niraparib'],
            'BRCA2': ['Olaparib', 'Rucaparib', 'Niraparib'],
            
            # Lung/NSCLC-specific
            'ALK': ['Alectinib', 'Brigatinib', 'Lorlatinib'],
            'ROS1': ['Crizotinib', 'Entrectinib'],
            'NTRK': ['Larotrectinib', 'Entrectinib'],
            
            # Bladder-specific
            'FGFR2': ['Erdafitinib'],
            'FGFR3': ['Erdafitinib'],
            
            # Renal-specific  
            'VHL': ['Sunitinib', 'Pazopanib', 'Cabozantinib'],
            'MET': ['Cabozantinib', 'Crizotinib'],
            
            # Immunotherapy markers (pan-cancer)
            'PD-L1_High': ['Pembrolizumab', 'Nivolumab', 'Atezolizumab'],
            'MSI-High': ['Pembrolizumab', 'Nivolumab'],
            'TMB-High': ['Pembrolizumab', 'Nivolumab'],
            
            # Gastric/Esophageal
            'Claudin18.2': ['Zolbetuximab'],
            
            # Liver-specific
            'AFP_High': ['Ramucirumab'],
            
            # Multi-kinase targets
            'VEGF': ['Bevacizumab', 'Ramucirumab'],
            'VEGFR': ['Sunitinib', 'Pazopanib', 'Lenvatinib', 'Cabozantinib']
        }
    
    def _get_cohort_protocols(self, cohort_name: str, cancer_type: str) -> List[Dict]:
        """Get protocols for cohort"""
        cancer_protocols = self.protocol_database.get(cancer_type, {})
        return cancer_protocols.get(cohort_name, [])
    
    def _integrate_targeted_therapies(self, base_regimen: Dict, genomic_variants: List[Dict]) -> Dict:
        """Add targeted therapies based on genomics"""
        regimen = base_regimen.copy()
        added_targeted = []
        
        for variant in genomic_variants:
            gene = variant['gene_name']
            if gene in self.targeted_therapy_map:
                targeted_drugs = self.targeted_therapy_map[gene]
                if targeted_drugs and targeted_drugs[0] not in regimen.get('drugs', []):
                    regimen.setdefault('drugs', []).append(targeted_drugs[0])
                    added_targeted.append({
                        'drug': targeted_drugs[0],
                        'target': gene,
                        'variant': variant.get('variant_detail', '')
                    })
        
        regimen['targeted_additions'] = added_targeted
        return regimen
    
    def _explain_cohort_match(self, cohort: Dict, regimen: Dict) -> str:
        cohort_name = cohort.get('cohort_name', 'Unknown')
        regimen_name = regimen.get('name', 'Custom')
        intensity = regimen.get('intensity', 'Moderate')
        evidence = regimen.get('evidence_level', 'Clinical')
        
        return f"환자군 '{cohort_name}'에 최적화된 {regimen_name}. 치료 강도: {intensity}, 근거 수준: {evidence}"
    
    def _explain_quantitative_rationale(self, quant: Dict, regimen: Dict) -> List[Dict]:
        rationale = []
        
        het_score = quant.get('overall_heterogeneity', 0)
        if het_score > 0.7:
            rationale.append({
                'metric': '종양 이질성',
                'value': f'{het_score:.2f}',
                'interpretation': '매우 높음',
                'decision_impact': f"{len(regimen.get('drugs', []))}제 병용 요법 필요"
            })
        
        return rationale
    
    def _cite_clinical_guidelines(self, regimen: Dict, clinical: Dict) -> List[str]:
        guidelines = []
        cancer_type = clinical.get('cancer_type', 'Colorectal')
        
        if cancer_type == 'Colorectal':
            guidelines.append("NCCN Guidelines for Colon Cancer v.3.2024")
            guidelines.append("ESMO Clinical Practice Guidelines 2023")
        elif cancer_type == 'Prostate':
            guidelines.append("NCCN Guidelines for Prostate Cancer v.4.2024")
            guidelines.append("AUA/ASTRO/SUO Guidelines 2024")
            guidelines.append("EAU Guidelines on Prostate Cancer 2024")
        elif cancer_type == 'Gastric':
            guidelines.append("NCCN Guidelines for Gastric Cancer v.2.2024")
            guidelines.append("ESMO Clinical Practice Guidelines for Gastric Cancer 2024")
            guidelines.append("Korean Practice Guideline for Gastric Cancer 2024")
        elif cancer_type == 'Ovarian':
            guidelines.append("NCCN Guidelines for Ovarian Cancer v.1.2024")
            guidelines.append("ESMO Clinical Practice Guidelines for Ovarian Cancer 2023")
            guidelines.append("SGO Clinical Practice Statement on PARP Inhibitors")
        elif cancer_type == 'Liver':
            guidelines.append("NCCN Guidelines for Hepatocellular Carcinoma v.2.2024")
            guidelines.append("EASL Clinical Practice Guidelines 2024")
            guidelines.append("AASLD Practice Guidance on HCC 2023")
        elif cancer_type == 'Bladder':
            guidelines.append("NCCN Guidelines for Bladder Cancer v.3.2024")
            guidelines.append("EAU Guidelines on Muscle-Invasive Bladder Cancer 2024")
            guidelines.append("AUA/SUO Guidelines for Non-Muscle Invasive Bladder Cancer")
        elif cancer_type == 'Renal':
            guidelines.append("NCCN Guidelines for Kidney Cancer v.2.2024")
            guidelines.append("EAU Guidelines on Renal Cell Carcinoma 2024")
            guidelines.append("ESMO Clinical Practice Guidelines for RCC 2024")
        elif cancer_type == 'Esophageal':
            guidelines.append("NCCN Guidelines for Esophageal Cancer v.2.2024")
            guidelines.append("ESMO Clinical Practice Guidelines for Esophageal Cancer 2024")
        elif cancer_type == 'Head_Neck':
            guidelines.append("NCCN Guidelines for Head and Neck Cancers v.1.2024")
            guidelines.append("ESMO Clinical Practice Guidelines for HNSCC 2024")
        
        return guidelines
    
    def _explain_genomic_basis(self, genomic: List[Dict], regimen: Dict) -> List[Dict]:
        genomic_rationale = []
        
        for addition in regimen.get('targeted_additions', []):
            genomic_rationale.append({
                'target_gene': addition['target'],
                'variant': addition['variant'],
                'drug': addition['drug'],
                'mechanism': f"{addition['target']} 경로 억제"
            })
        
        # Add pathway-based rationale
        for pathway_target in regimen.get('pathway_targets', []):
            genomic_rationale.append({
                'target_gene': pathway_target.get('target', ''),
                'pathway': pathway_target.get('pathway', ''),
                'drug': pathway_target.get('drug', ''),
                'mechanism': f"{pathway_target.get('role', 'Pathway targeting')}"
            })
        
        return genomic_rationale
    
    def _add_literature_references(self, regimen: Dict) -> List[str]:
        references = []
        
        # Add synergy references if available
        for synergy in regimen.get('synergy_analysis', {}).get('synergies', []):
            refs = synergy.get('references', [])
            references.extend(refs)
        
        # Add general references
        references.extend([
            "Cremolini et al., NEJM 2015 (FOLFOXIRI)",
            "Kopetz et al., NEJM 2019 (BRAF + EGFR combo)",
           "André et al., NEJM 2019 (PI3K + ER therapy)"
        ])
        
        return list(set(references))  # Remove duplicates
    
    def _assess_confidence(self, evidence: Dict) -> str:
        num_pathways = len(evidence.get('pathway_analysis', {}).get('active_pathways', []))
        has_synergy = evidence.get('synergy_evidence', {}).get('synergy_count', 0) > 0
        num_quant = len(evidence.get('quantitative_indicators', []))
        
        if num_pathways >= 2 and has_synergy and num_quant >= 2:
            return 'Very High'
        elif num_pathways >= 1 and (has_synergy or num_quant >= 2):
            return 'High'
        elif num_pathways >= 1 or num_quant >= 1:
            return 'Moderate'
        else:
            return 'Low'
    
    def _generate_warnings(self, regimen: Dict, clinical: Dict) -> List[str]:
        warnings = []
        
        age = clinical.get('age', 60)
        if age > 75:
            warnings.append("고령 환자 (>75세) - 용량 감량 고려")
        
        ecog = clinical.get('ecog_score', 0)
        if ecog >= 2:
            warnings.append(f"ECOG {ecog} - 치료 강도 조정 필요")
        
        num_drugs = len(regimen.get('drugs', []))
        if num_drugs >= 4:
            warnings.append(f"{num_drugs}제 병용 - 독성 모니터링 강화 필요")
        
        return warnings
    
    # ==================== LITERATURE INTEGRATION METHODS ====================
    
    def enhance_recommendation_with_literature(
        self,
        base_recommendation: Dict,
        clinical: Dict,
        genomic_variants: List[Dict]
    ) -> Dict:
        """Enhance treatment recommendation with literature-based evidence"""
        if not self.has_literature:
            return base_recommendation
        
        cancer_type = clinical.get('cancer_type', '').lower()
        patient_biomarkers = self._extract_biomarker_status(genomic_variants, cancer_type)
        
        lit_recommendations = self.literature_loader.match_biomarkers_to_treatment(
            cancer_type,
            patient_biomarkers
        )
        
        if not lit_recommendations:
            return base_recommendation
        
        enhanced = base_recommendation.copy()
        enhanced['literature_evidence'] = []
        
        for lit_rec in lit_recommendations:
            evidence = {
                'biomarker': lit_rec['biomarker'],
                'recommended_treatment': lit_rec['treatment'],
                'evidence_level': lit_rec['evidence'],
                'expected_outcomes': lit_rec['expected_outcome'],
                'prevalence': self.literature_loader.get_biomarker_prevalence(
                    cancer_type, lit_rec['biomarker']
                )
            }
            enhanced['literature_evidence'].append(evidence)
        
        if 'genomic_alterations' in enhanced:
            for alteration in enhanced['genomic_alterations']:
                gene = alteration.get('gene', '')
                freq = self.literature_loader.get_mutation_frequency(cancer_type, gene)
                if freq:
                    alteration['literature_frequency'] = freq
                    alteration['interpretation'] = self._interpret_mutation_frequency(freq)
        
        return enhanced
    
    def _extract_biomarker_status(self, genomic_variants: List[Dict], cancer_type: str) -> Dict[str, bool]:
        """Extract biomarker status from genomic variants"""
        biomarkers = {}
        for variant in genomic_variants:
            gene = variant.get('gene', '').upper()
            if gene in ['HER2', 'ERBB2']:
                biomarkers['HER2'] = True
            elif gene in ['MSI', 'MLH1', 'MSH2', 'MSH6', 'PMS2']:
                biomarkers['MSI_high'] = True
            elif gene == 'PD-L1' or gene == 'CD274':
                biomarkers['PD_L1_positive'] = True
            elif gene in ['BRCA1', 'BRCA2']:
                biomarkers[gene] = True
            elif 'CLDN18' in gene:
                biomarkers['Claudin18_2'] = True
            elif gene in ['ALK', 'ROS1', 'NTRK', 'FGFR2', 'FGFR3', 'MET']:
                biomarkers[gene] = True
        return biomarkers
    
    def _interpret_mutation_frequency(self, frequency: float) -> str:
        """Interpret mutation frequency from literature"""
        if frequency >= 0.50:
            return f"Very common ({frequency*100:.0f}% in literature)"
        elif frequency >= 0.20:
            return f"Common ({frequency*100:.0f}% in literature)"
        elif frequency >= 0.10:
            return f"Moderately frequent ({frequency*100:.0f}% in literature)"
        elif frequency >= 0.05:
            return f"Uncommon ({frequency*100:.0f}% in literature)"
        else:
            return f"Rare ({frequency*100:.1f}% in literature)"
    
    def get_literature_summary(self, cancer_type: str) -> Dict:
        """Get summary of literature data for a cancer type"""
        if not self.has_literature:
            return {}
        
        features = self.literature_loader.cancer_features.get(cancer_type.lower())
        if not features:
            return {'available': False}
        
        summary = {'available': True}
        if 'actionable_biomarkers' in features:
            summary['biomarkers'] = {
                name: {'prevalence': data.get('prevalence'), 'treatment': data.get('treatment')}
                for name, data in features['actionable_biomarkers'].items()
            }
        if 'mutation_landscape' in features:
            mutations = features['mutation_landscape'].get('driver_mutations', {})
            summary['common_mutations'] = {
                gene: data.get('prevalence')
                for gene, data in mutations.items()
                if data.get('prevalence', 0) >= 0.10
            }
        if 'quality_metrics' in features:
            summary['evidence_quality'] = features['quality_metrics']
        return summary
    
    def cite_literature_sources(self, cancer_type: str, biomarker: str = None) -> List[Dict]:
        """Get literature citations for cancer type and biomarker"""
        if not self.has_literature:
            return []
        papers = self.literature_loader.search_papers(
            cancer_type=cancer_type,
            biomarker=biomarker,
            min_evidence_level="Level_I"
        )
        return [{
            'title': paper.get('title', ''),
            'journal': paper.get('journal', ''),
            'year': paper.get('year', ''),
            'pmid': paper.get('pmid', ''),
            'doi': paper.get('doi', ''),
            'evidence_level': paper.get('evidence_level', '')
        } for paper in papers]


# Make it backward compatible with old name
DrugCombinationOptimizer = EnhancedDrugCombinationOptimizer
