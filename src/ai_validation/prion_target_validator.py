"""
PrPc Target Validation Framework

AI-powered multi-criteria validation of PrPc/PRNP as cancer therapeutic target
using knowledge base evidence and GPT-4 reasoning.
"""

import os
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import openai
from dotenv import load_dotenv

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")


class PrionTargetValidator:
    """AI-powered validation of PrPc as therapeutic target"""
    
    VALIDATION_CRITERIA = {
        'biological_rationale': {
            'weight': 0.30,
            'description': 'Expression, disease correlation, mechanistic understanding'
        },
        'therapeutic_tractability': {
            'weight': 0.25,
            'description': 'Druggability, existing approaches, safety potential'
        },
        'clinical_evidence': {
            'weight': 0.25,
            'description': 'Preclinical quality, clinical trial data, real-world evidence'
        },
        'kras_synergy': {
            'weight': 0.20,
            'description': 'Mechanistic crosstalk, combination rationale, resistance bypass'
        }
    }
    
    def __init__(self, data_dir: str = "C:/Users/brook/Desktop/ADDS/data"):
        self.data_dir = Path(data_dir)
        self.output_dir = self.data_dir / "analysis"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load PrPc expression and correlation data
        self.prpc_data = self._load_prpc_data()
        self.extracted_knowledge = self._load_extracted_knowledge()
    
    def _load_prpc_data(self) -> Dict:
        """Load PrPc-KRAS correlation analysis results"""
        try:
            results_file = self.data_dir / "analysis" / "prion_kras_correlation_results.json"
            if results_file.exists():
                with open(results_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
        except Exception as e:
            print(f"Warning: Could not load PrPc correlation data: {e}")
        return {}
    
    def _load_extracted_knowledge(self) -> List[Dict]:
        """Load extracted literature knowledge including curated web search"""
        extractions = []
        
        # Load from GPT-4 extractions (if any exist)
        extraction_dir = self.data_dir / "extracted" / "prion"
        
        if extraction_dir.exists():
            # First load curated web search evidence
            curated_file = extraction_dir / "web_search_curated_evidence.json"
            if curated_file.exists():
                try:
                    with open(curated_file, 'r', encoding='utf-8') as f:
                        curated_data = json.load(f)
                        if 'papers' in curated_data:
                            # Convert curated format to extraction format
                            for paper in curated_data['papers']:
                                extraction = {
                                    'prion_mechanisms': paper.get('prion_mechanisms', {}),
                                    'kras_crosstalk': paper.get('kras_crosstalk', {}),
                                    'therapeutic_validation': paper.get('therapeutic_validation', {}),
                                    'biomarker_potential': paper.get('biomarker_potential', {}),
                                    'key_findings': paper.get('key_findings', []),
                                    'metadata': {
                                        'pmid': paper.get('source', 'web_search'),
                                        'title': paper.get('title', ''),
                                        'cancer_type': paper.get('cancer_type', 'unknown')
                                    }
                                }
                                extractions.append(extraction)
                            print(f"Loaded {len(curated_data['papers'])} curated papers from web search")
                except Exception as e:
                    print(f"Warning: Could not load curated evidence: {e}")
            
            # Then load any GPT-4 extraction files
            for file in extraction_dir.glob("prion_extraction_*.json"):
                try:
                    with open(file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        if 'extractions' in data:
                            extractions.extend(data['extractions'])
                            print(f"Loaded {len(data['extractions'])} GPT-4 extracted papers")
                except Exception as e:
                    print(f"Warning: Could not load {file}: {e}")
        
        print(f"Total papers loaded: {len(extractions)}")
        return extractions
    
    def aggregate_evidence(self) -> Dict:
        """
        Aggregate all available evidence for validation
        
        Returns:
            Comprehensive evidence dictionary
        """
        evidence = {
            'expression_data': {},
            'kras_correlation': {},
            'mechanisms': {
                'receptor_interactions': set(),
                'signaling_pathways': set(),
                'cellular_processes': set()
            },
            'kras_crosstalk': {
                'direct_evidence': [],
                'indirect_evidence': [],
                'molecular_complexes': set()
            },
            'therapeutic_data': {
                'preclinical': [],
                'clinical': [],
                'combinations': set()
            },
            'cancer_specific': {},
            'literature_count': len(self.extracted_knowledge)
        }
        
        # Expression data
        if 'prpc_expression' in self.prpc_data:
            evidence['expression_data'] = self.prpc_data['prpc_expression']
        
        # Correlation data
        if 'correlation_analysis' in self.prpc_data and self.prpc_data['correlation_analysis']:
            corr = self.prpc_data['correlation_analysis'][0]
            evidence['kras_correlation'] = {
                'pearson_r': corr.get('pearson_r'),
                'pearson_p': corr.get('pearson_p'),
                'spearman_r': corr.get('spearman_r'),
                'spearman_p': corr.get('spearman_p')
            }
        
        # Aggregate from extracted literature
        for ext in self.extracted_knowledge:
            # Mechanisms
            if 'prion_mechanisms' in ext:
                mech = ext['prion_mechanisms']
                evidence['mechanisms']['receptor_interactions'].update(
                    mech.get('receptor_interactions', [])
                )
                evidence['mechanisms']['signaling_pathways'].update(
                    mech.get('signaling_pathways', [])
                )
                evidence['mechanisms']['cellular_processes'].update(
                    mech.get('cellular_processes', [])
                )
            
            # KRAS crosstalk
            if 'kras_crosstalk' in ext:
                crosstalk = ext['kras_crosstalk']
                interaction_type = crosstalk.get('interaction_type', 'none')
                
                if interaction_type == 'direct':
                    evidence['kras_crosstalk']['direct_evidence'].append({
                        'pmid': ext.get('metadata', {}).get('pmid'),
                        'title': ext.get('metadata', {}).get('title'),
                        'complex': crosstalk.get('molecular_complex', []),
                        'effects': crosstalk.get('downstream_effects', [])
                    })
                    evidence['kras_crosstalk']['molecular_complexes'].update(
                        crosstalk.get('molecular_complex', [])
                    )
                elif interaction_type == 'indirect':
                    evidence['kras_crosstalk']['indirect_evidence'].append(
                        ext.get('metadata', {}).get('pmid')
                    )
            
            # Therapeutic evidence
            if 'therapeutic_validation' in ext:
                therapeutic = ext['therapeutic_validation']
                evidence['therapeutic_data']['preclinical'].extend(
                    therapeutic.get('preclinical_evidence', [])
                )
                evidence['therapeutic_data']['clinical'].extend(
                    therapeutic.get('clinical_trials', [])
                )
                evidence['therapeutic_data']['combinations'].update(
                    therapeutic.get('combination_strategies', [])
                )
        
        # Convert sets to lists for JSON serialization
        evidence['mechanisms']['receptor_interactions'] = list(
            evidence['mechanisms']['receptor_interactions']
        )
        evidence['mechanisms']['signaling_pathways'] = list(
            evidence['mechanisms']['signaling_pathways']
        )
        evidence['mechanisms']['cellular_processes'] = list(
            evidence['mechanisms']['cellular_processes']
        )
        evidence['kras_crosstalk']['molecular_complexes'] = list(
            evidence['kras_crosstalk']['molecular_complexes']
        )
        evidence['therapeutic_data']['combinations'] = list(
            evidence['therapeutic_data']['combinations']
        )
        
        return evidence
    
    def validate_with_ai(self, evidence: Dict) -> Dict:
        """
        Perform AI-powered validation analysis using GPT-4
        
        Args:
            evidence: Aggregated evidence dictionary
            
        Returns:
            Validation results with scores and reasoning
        """
        prompt = f"""You are an expert oncology drug discovery scientist evaluating PrPc/PRNP as a therapeutic target for cancer treatment.

Based on the following comprehensive evidence, score PrPc as a therapeutic target across 4 criteria (0-100 scale).

## Evidence Summary

### Expression Data
{json.dumps(evidence['expression_data'], indent=2)}

### KRAS Correlation
{json.dumps(evidence['kras_correlation'], indent=2)}

### Mechanisms Identified
- Receptor Interactions: {', '.join(evidence['mechanisms']['receptor_interactions'][:10])}
- Signaling Pathways: {', '.join(evidence['mechanisms']['signaling_pathways'][:10])}
- Cellular Processes: {', '.join(evidence['mechanisms']['cellular_processes'][:10])}

### KRAS Crosstalk Evidence
- Direct interaction papers: {len(evidence['kras_crosstalk']['direct_evidence'])}
- Molecular complexes: {', '.join(evidence['kras_crosstalk']['molecular_complexes'])}
- Key findings: {json.dumps(evidence['kras_crosstalk']['direct_evidence'][:2], indent=2)}

### Therapeutic Validation
- Preclinical studies: {len(evidence['therapeutic_data']['preclinical'])} findings
- Clinical trials: {len(evidence['therapeutic_data']['clinical'])} references
- Combination strategies: {', '.join(list(evidence['therapeutic_data']['combinations'])[:5])}

### Literature Base
- Total papers analyzed: {evidence['literature_count']}

## Scoring Criteria

Score each criterion from 0-100:

### 1. Biological Rationale (0-100)
Assess:
- Expression levels in target cancers (higher = more relevant)
- Correlation with KRAS mutations (stronger correlation = higher score)
- Depth of mechanistic understanding (more pathways/receptors = higher)
- Role in cancer hallmarks (proliferation, metastasis, etc.)

### 2. Therapeutic Tractability (0-100)
Assess:
- Druggability (extracellular protein = favorable)
- Existing therapeutic approaches (antibodies, small molecules)
- Selectivity potential (low in normal tissue vs cancer)
- Safety concerns based on PrPc's known biology

### 3. Clinical Evidence (0-100)
Assess:
- Quality and quantity of preclinical data
- Existence of clinical trials
- Real-world biomarker validation
- Strength of therapeutic evidence

### 4. KRAS Synergy Potential (0-100)
Assess:
- Strength of mechanistic crosstalk evidence
- Rationale for combination with KRAS inhibitors
- Potential to overcome resistance
- Non-overlapping mechanisms

## Output Format

Provide your analysis in this JSON format:

{{
  "scores": {{
    "biological_rationale": <0-100>,
    "therapeutic_tractability": <0-100>,
    "clinical_evidence": <0-100>,
    "kras_synergy": <0-100>,
    "overall_weighted": <calculated weighted average>
  }},
  "confidence": "high|medium|low",
  "reasoning": {{
    "biological_rationale": "2-3 sentences explaining the score with specific evidence",
    "therapeutic_tractability": "2-3 sentences explaining the score with specific evidence",
    "clinical_evidence": "2-3 sentences explaining the score with specific evidence",
    "kras_synergy": "2-3 sentences explaining the score with specific evidence"
  }},
  "key_strengths": ["3-5 bullet points of strongest evidence for PrPc as target"],
  "key_limitations": ["3-5 bullet points of main concerns or evidence gaps"],
  "recommendation": "strong_pursue|moderate_pursue|exploratory|deprioritize",
  "next_steps": ["3-5 specific recommended actions based on this analysis"]
}}

IMPORTANT: Be evidence-based and cite specific data points. Be honest about limitations."""

        try:
            response = openai.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are an expert drug discovery scientist. Provide rigorous, evidence-based target validation assessments."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,
                max_tokens=2500
            )
            
            content = response.choices[0].message.content.strip()
            
            # Parse JSON
            if content.startswith("```json"):
                content = content.split("```json")[1].split("```")[0].strip()
            elif content.startswith("```"):
                content = content.split("```")[1].split("```")[0].strip()
            
            validation_result = json.loads(content)
            
            # Add metadata
            validation_result['metadata'] = {
                'validation_date': datetime.now().isoformat(),
                'model': 'gpt-4o',
                'evidence_papers': evidence['literature_count']
            }
            
            # Calculate weighted score if not provided
            if 'overall_weighted' not in validation_result['scores']:
                weighted = sum(
                    validation_result['scores'].get(criterion, 0) * self.VALIDATION_CRITERIA[criterion]['weight']
                    for criterion in self.VALIDATION_CRITERIA.keys()
                )
                validation_result['scores']['overall_weighted'] = round(weighted, 1)
            
            return validation_result
            
        except Exception as e:
            print(f"AI validation error: {e}")
            return None
    
    def generate_validation_report(self, validation: Dict, evidence: Dict) -> str:
        """
        Generate comprehensive markdown validation report
        
        Args:
            validation: Validation results from AI
            evidence: Aggregated evidence
            
        Returns:
            Markdown report string
        """
        scores = validation['scores']
        reasoning = validation['reasoning']
        
        report = f"""# PrPc/PRNP Target Validation Report

**Validation Date**: {datetime.now().strftime("%Y-%m-%d")}  
**Evidence Base**: {evidence['literature_count']} research papers

---

## Executive Summary

**Overall Validation Score**: {scores['overall_weighted']:.1f}/100

**Recommendation**: **{validation['recommendation'].upper().replace('_', ' ')}**

**Confidence Level**: {validation['confidence'].upper()}

### Quick Scores

| Criterion | Score | Weight |
|-----------|-------|--------|
| **Biological Rationale** | {scores['biological_rationale']}/100 | 30% |
| **Therapeutic Tractability** | {scores['therapeutic_tractability']}/100 | 25% |
| **Clinical Evidence** | {scores['clinical_evidence']}/100 | 25% |
| **KRAS Synergy Potential** | {scores['kras_synergy']}/100 | 20% |

---

## Detailed Validation Analysis

### 1. Biological Rationale ({scores['biological_rationale']}/100)

{reasoning['biological_rationale']}

**Supporting Evidence**:
- **Expression Data**: 
  - Pancreatic cancer: 76%
  - Colorectal cancer: 74.5%
  - Gastric cancer: 68%
  - Breast cancer: 24%

- **KRAS Correlation**: 
  - Spearman ρ = {evidence['kras_correlation'].get('spearman_r', 'N/A')}
  - p-value = {evidence['kras_correlation'].get('spearman_p', 'N/A')}

- **Mechanisms Identified**: {len(evidence['mechanisms']['signaling_pathways'])} signaling pathways, {len(evidence['mechanisms']['receptor_interactions'])} receptor interactions

---

### 2. Therapeutic Tractability ({scores['therapeutic_tractability']}/100)

{reasoning['therapeutic_tractability']}

**Key Druggability Features**:
- Extracellular protein (antibody accessible)
- Known receptor interactions: {', '.join(evidence['mechanisms']['receptor_interactions'][:5])}
- Combination strategies identified: {len(evidence['therapeutic_data']['combinations'])}

---

### 3. Clinical Evidence ({scores['clinical_evidence']}/100)

{reasoning['clinical_evidence']}

**Evidence Summary**:
- Preclinical findings: {len(evidence['therapeutic_data']['preclinical'])}
- Clinical trial references: {len(evidence['therapeutic_data']['clinical'])}
- Literature base: {evidence['literature_count']} papers

---

### 4. KRAS Synergy Potential ({scores['kras_synergy']}/100)

{reasoning['kras_synergy']}

**KRAS Crosstalk Evidence**:
- **Direct interaction papers**: {len(evidence['kras_crosstalk']['direct_evidence'])}
- **Molecular complexes**: {', '.join(evidence['kras_crosstalk']['molecular_complexes'])}

"""
        
        # Add direct evidence details
        if evidence['kras_crosstalk']['direct_evidence']:
            report += "\n**Key Direct Interaction Studies**:\n"
            for study in evidence['kras_crosstalk']['direct_evidence'][:3]:
                report += f"\n- **PMID {study['pmid']}**: {study['title'][:80]}...\n"
                report += f"  - Complex: {', '.join(study['complex'])}\n"
                report += f"  - Effects: {', '.join(study['effects'][:2])}\n"
        
        report += f"""

---

## Key Strengths

"""
        for strength in validation['key_strengths']:
            report += f"- {strength}\n"
        
        report += f"""

---

## Key Limitations

"""
        for limitation in validation['key_limitations']:
            report += f"- {limitation}\n"
        
        report += f"""

---

## Recommended Next Steps

"""
        for step in validation['next_steps']:
            report += f"{validation['next_steps'].index(step) + 1}. {step}\n"
        
        report += f"""

---

## Appendix: Complete Evidence Summary

### Expression by Cancer Type
```json
{json.dumps(evidence['expression_data'], indent=2)}
```

### Identified Mechanisms
- **Receptors**: {', '.join(evidence['mechanisms']['receptor_interactions'])}
- **Pathways**: {', '.join(evidence['mechanisms']['signaling_pathways'])}
- **Processes**: {', '.join(evidence['mechanisms']['cellular_processes'])}

### Therapeutic Combinations Identified
{', '.join(list(evidence['therapeutic_data']['combinations']))}

---

**Report Generated**: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}  
**Validation Framework**: AI-powered multi-criteria assessment v1.0
"""
        
        return report
    
    def run_complete_validation(self) -> Tuple[Dict, str]:
        """
        Run complete validation pipeline
        
        Returns:
            Tuple of (validation_results, report_path)
        """
        print("="*70)
        print("PrPc Target Validation Analysis")
        print("="*70)
        
        # Step 1: Aggregate evidence
        print("\nStep 1: Aggregating evidence from all sources...")
        evidence = self.aggregate_evidence()
        print(f"  [OK] Loaded {evidence['literature_count']} papers")
        print(f"  [OK] Identified {len(evidence['mechanisms']['signaling_pathways'])} signaling pathways")
        print(f"  [OK] Found {len(evidence['kras_crosstalk']['direct_evidence'])} direct KRAS interaction studies")
        
        # Step 2: AI validation
        print("\nStep 2: Performing AI-powered validation analysis...")
        validation = self.validate_with_ai(evidence)
        
        if validation:
            print(f"  [OK] Overall validation score: {validation['scores']['overall_weighted']:.1f}/100")
            print(f"  [OK] Recommendation: {validation['recommendation']}")
            print(f"  [OK] Confidence: {validation['confidence']}")
        else:
            print("  [FAIL] Validation failed")
            return None, None
        
        # Step 3: Generate report
        print("\nStep 3: Generating validation report...")
        report = self.generate_validation_report(validation, evidence)
        
        # Save results
        validation_file = self.output_dir / "prpc_target_validation.json"
        with open(validation_file, 'w', encoding='utf-8') as f:
            json.dump({
                'validation': validation,
                'evidence': {k: v for k, v in evidence.items() if k != 'literature_count'}
            }, f, indent=2, ensure_ascii=False, default=str)
        
        report_file = self.output_dir / "prpc_target_validation_report.md"
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report)
        
        print(f"\n  [OK] Results saved to: {validation_file}")
        print(f"  [OK] Report saved to: {report_file}")
        
        print("\n" + "="*70)
        print("VALIDATION COMPLETE")
        print("="*70)
        
        return validation, str(report_file)


def main():
    """Run PrPc target validation"""
    validator = PrionTargetValidator()
    validation, report_path = validator.run_complete_validation()
    
    if validation:
        print(f"\n{'='*70}")
        print("VALIDATION SUMMARY")
        print(f"{'='*70}")
        print(f"Overall Score: {validation['scores']['overall_weighted']:.1f}/100")
        print(f"Recommendation: {validation['recommendation']}")
        print(f"\nFull report available at: {report_path}")


if __name__ == "__main__":
    main()
