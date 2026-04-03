"""
PrPc/PRNP Literature Knowledge Extraction

Specialized extraction pipeline for PrPc-related cancer mechanisms,
focusing on KRAS crosstalk and therapeutic validation.
"""

import os
import json
import time
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime
import openai
from dotenv import load_dotenv

load_dotenv()

# Configure OpenAI
openai.api_key = os.getenv("OPENAI_API_KEY")

class PrionLiteratureExtractor:
    """Extract PrPc mechanistic knowledge from research papers"""
    
    def __init__(self, output_dir: str = "C:/Users/brook/Desktop/ADDS/data/extracted/prion"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.extraction_schema = {
            "prion_mechanisms": {
                "receptor_interactions": [],
                "signaling_pathways": [],
                "cellular_processes": [],
                "cancer_specific_roles": {}
            },
            "kras_crosstalk": {
                "interaction_type": "",
                "molecular_complex": [],
                "downstream_effects": [],
                "functional_outcomes": []
            },
            "therapeutic_validation": {
                "preclinical_evidence": [],
                "clinical_trials": [],
                "combination_strategies": []
            },
            "biomarker_potential": {
                "prognostic_value": "",
                "predictive_value": "",
                "expression_correlation": {}
            }
        }
    
    def extract_from_abstract(self, paper_data: Dict) -> Dict:
        """
        Extract PrPc-specific knowledge from paper abstract using GPT-4
        
        Args:
            paper_data: Dictionary containing title, abstract, PMID, cancer_type
            
        Returns:
            Extracted structured knowledge
        """
        title = paper_data.get('title', '')
        abstract = paper_data.get('abstract', '')
        pmid = paper_data.get('pmid', '')
        cancer_type = paper_data.get('cancer_type', 'unknown')
        
        prompt = f"""You are an expert cancer biologist analyzing research on the cellular prion protein (PrPc/PRNP) in cancer.

Extract structured information from this research paper abstract about PrPc's role in cancer, with special focus on any KRAS crosstalk.

Title: {title}

Abstract: {abstract}

Extract the following information in valid JSON format:

{{
  "prion_mechanisms": {{
    "receptor_interactions": ["list specific receptors PrPc interacts with, e.g., RPSA/37LRP, integrins"],
    "signaling_pathways": ["list signaling pathways affected, e.g., RAS-AKT, ERK1/2, WNT"],
    "cellular_processes": ["list cellular processes, e.g., proliferation, migration, EMT, angiogenesis"],
    "cancer_specific_roles": {{
      "{cancer_type}": ["specific roles in this cancer type"]
    }}
  }},
  "kras_crosstalk": {{
    "interaction_type": "direct|indirect|none",
    "molecular_complex": ["components if direct interaction exists, e.g., PrPc-RPSA-KRAS"],
    "downstream_effects": ["effects on RAS signaling, e.g., RAS-GTP levels, AKT/ERK phosphorylation"],
    "functional_outcomes": ["functional consequences, e.g., proliferation, tumor growth"]
  }},
  "therapeutic_validation": {{
    "preclinical_evidence": ["experimental models showing therapeutic potential"],
    "clinical_trials": ["any clinical trial information"],
    "combination_strategies": ["drugs PrPc targeting could combine with"]
  }},
  "biomarker_potential": {{
    "prognostic_value": "high|medium|low|none - based on survival/outcome associations",
    "predictive_value": "high|medium|low|none - based on treatment response prediction",
    "expression_correlation": {{
      "disease_stage": "correlation with stage if mentioned",
      "mutation_status": "correlation with mutations if mentioned"
    }}
  }},
  "key_findings": ["3-5 most important findings from this paper"],
  "confidence": "high|medium|low - your confidence in these extractions"
}}

Rules:
1. Only extract information explicitly stated or strongly implied in the abstract
2. Use empty arrays [] if no information found for a field
3. Use "none" or "unknown" for scalar fields with no information
4. Be specific - include exact molecular names, pathway components, experimental details
5. For KRAS crosstalk, be conservative - only mark as "direct" if molecular interaction is described

Return ONLY valid JSON, no additional text."""

        try:
            response = openai.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are an expert cancer biologist specializing in prion protein research. Extract information accurately and return only valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=2000
            )
            
            content = response.choices[0].message.content.strip()
            
            # Try to parse JSON
            if content.startswith("```json"):
                content = content.split("```json")[1].split("```")[0].strip()
            elif content.startswith("```"):
                content = content.split("```")[1].split("```")[0].strip()
            
            extracted = json.loads(content)
            
            # Add metadata
            extracted['metadata'] = {
                'pmid': pmid,
                'title': title,
                'cancer_type': cancer_type,
                'extraction_date': datetime.now().isoformat(),
                'model': 'gpt-4o'
            }
            
            return extracted
            
        except json.JSONDecodeError as e:
            print(f"JSON parsing error for PMID {pmid}: {e}")
            print(f"Response content: {content[:200]}...")
            return None
        except Exception as e:
            print(f"Extraction error for PMID {pmid}: {e}")
            return None
    
    def search_prion_papers(self, max_papers: int = 30) -> List[Dict]:
        """
        Search PubMed for PrPc + cancer papers
        
        Args:
            max_papers: Maximum number of papers to retrieve
            
        Returns:
            List of paper metadata dictionaries
        """
        from Bio import Entrez
        
        Entrez.email = os.getenv("ENTREZ_EMAIL", "research@example.com")
        
        # Search queries for different aspects
        queries = [
            # Core PrPc cancer papers
            '("prion protein"[Title/Abstract] OR "PRNP"[Title/Abstract] OR "PrPc"[Title/Abstract]) AND (cancer[Title/Abstract] OR carcinoma[Title/Abstract] OR tumor[Title/Abstract])',
            
            # KRAS-specific
            '("prion protein"[Title/Abstract] OR "PRNP"[Title/Abstract]) AND "KRAS"[Title/Abstract]',
            
            # Specific cancer types
            '("prion protein"[Title/Abstract] OR "PRNP"[Title/Abstract]) AND ("colorectal cancer"[Title/Abstract] OR "gastric cancer"[Title/Abstract] OR "pancreatic cancer"[Title/Abstract])',
            
            # Mechanistic
            '("prion protein"[Title/Abstract] OR "PRNP"[Title/Abstract]) AND cancer AND ("signaling"[Title/Abstract] OR "pathway"[Title/Abstract] OR "mechanism"[Title/Abstract])',
            
            # Therapeutic
            '("prion protein"[Title/Abstract] OR "PRNP"[Title/Abstract]) AND cancer AND ("therapeutic"[Title/Abstract] OR "treatment"[Title/Abstract] OR "antibody"[Title/Abstract])'
        ]
        
        all_papers = []
        seen_pmids = set()
        
        for query in queries:
            try:
                print(f"\nSearching: {query[:80]}...")
                
                # Search
                handle = Entrez.esearch(
                    db="pubmed",
                    term=query,
                    retmax=max_papers // len(queries),
                    sort="relevance",
                    mindate="2015",
                    maxdate="2026"
                )
                record = Entrez.read(handle)
                handle.close()
                
                pmids = record["IdList"]
                print(f"  Found {len(pmids)} papers")
                
                # Fetch details
                if pmids:
                    time.sleep(0.5)  # Rate limiting
                    handle = Entrez.efetch(
                        db="pubmed",
                        id=pmids,
                        rettype="abstract",
                        retmode="xml"
                    )
                    records = Entrez.read(handle)
                    handle.close()
                    
                    for record in records['PubmedArticle']:
                        try:
                            pmid = str(record['MedlineCitation']['PMID'])
                            
                            if pmid in seen_pmids:
                                continue
                            seen_pmids.add(pmid)
                            
                            article = record['MedlineCitation']['Article']
                            
                            # Extract title
                            title = article.get('ArticleTitle', '')
                            
                            # Extract abstract
                            abstract_parts = article.get('Abstract', {}).get('AbstractText', [])
                            if isinstance(abstract_parts, list):
                                abstract = ' '.join([str(part) for part in abstract_parts])
                            else:
                                abstract = str(abstract_parts)
                            
                            # Determine cancer type from title/abstract
                            text = f"{title} {abstract}".lower()
                            cancer_type = 'pan-cancer'
                            if 'colorectal' in text or 'colon' in text:
                                cancer_type = 'colorectal'
                            elif 'gastric' in text or 'stomach' in text:
                                cancer_type = 'gastric'
                            elif 'pancreatic' in text or 'pancreas' in text:
                                cancer_type = 'pancreatic'
                            elif 'breast' in text:
                                cancer_type = 'breast'
                            elif 'lung' in text:
                                cancer_type = 'lung'
                            
                            paper = {
                                'pmid': pmid,
                                'title': title,
                                'abstract': abstract,
                                'cancer_type': cancer_type,
                                'year': record['MedlineCitation'].get('DateCompleted', {}).get('Year', 'unknown')
                            }
                            
                            all_papers.append(paper)
                            
                        except Exception as e:
                            print(f"  Error parsing record: {e}")
                            continue
                
                time.sleep(0.5)  # Rate limiting between queries
                
            except Exception as e:
                print(f"Search error: {e}")
                continue
        
        print(f"\nTotal unique papers found: {len(all_papers)}")
        return all_papers[:max_papers]
    
    def batch_extract(self, papers: List[Dict], delay: float = 1.0) -> Dict:
        """
        Extract knowledge from multiple papers with rate limiting
        
        Args:
            papers: List of paper metadata dictionaries
            delay: Seconds to wait between API calls
            
        Returns:
            Dictionary with extraction results and statistics
        """
        results = {
            'extractions': [],
            'failed': [],
            'statistics': {
                'total': len(papers),
                'successful': 0,
                'failed': 0,
                'cost_estimate': 0.0
            }
        }
        
        print(f"\nExtracting knowledge from {len(papers)} papers...")
        print("="*70)
        
        for i, paper in enumerate(papers, 1):
            print(f"\n[{i}/{len(papers)}] Processing PMID {paper.get('pmid', 'unknown')}")
            print(f"  Title: {paper.get('title', '')[:60]}...")
            
            extracted = self.extract_from_abstract(paper)
            
            if extracted:
                results['extractions'].append(extracted)
                results['statistics']['successful'] += 1
                
                # Print key findings
                if 'key_findings' in extracted:
                    print(f"  ??Extracted {len(extracted['key_findings'])} key findings")
                if extracted.get('kras_crosstalk', {}).get('interaction_type') == 'direct':
                    print(f"  тн?DIRECT KRAS CROSSTALK FOUND!")
            else:
                results['failed'].append(paper)
                results['statistics']['failed'] += 1
                print(f"  ??Extraction failed")
            
            # Rate limiting
            if i < len(papers):
                time.sleep(delay)
        
        # Estimate cost (rough: ~1000 tokens per paper * $0.01/1K tokens for GPT-4)
        results['statistics']['cost_estimate'] = results['statistics']['successful'] * 0.015
        
        print("\n" + "="*70)
        print("EXTRACTION COMPLETE")
        print("="*70)
        print(f"Successful: {results['statistics']['successful']}/{results['statistics']['total']}")
        print(f"Failed: {results['statistics']['failed']}")
        print(f"Estimated cost: ${results['statistics']['cost_estimate']:.2f}")
        
        return results
    
    def save_results(self, results: Dict, filename: str = None):
        """Save extraction results to file"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"prion_extraction_{timestamp}.json"
        
        output_path = self.output_dir / filename
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"\nResults saved to: {output_path}")
        return str(output_path)
    
    def generate_summary_report(self, results: Dict) -> str:
        """Generate markdown summary of extraction results"""
        
        report = f"""# PrPc Literature Extraction Summary

**Extraction Date**: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## Statistics

- **Total Papers**: {results['statistics']['total']}
- **Successful Extractions**: {results['statistics']['successful']}
- **Failed**: {results['statistics']['failed']}
- **Success Rate**: {results['statistics']['successful']/results['statistics']['total']*100:.1f}%
- **Estimated Cost**: ${results['statistics']['cost_estimate']:.2f}

---

## Key Findings Across Papers

### KRAS Crosstalk Evidence

"""
        
        # Analyze KRAS crosstalk
        direct_kras = []
        indirect_kras = []
        
        for ext in results['extractions']:
            crosstalk = ext.get('kras_crosstalk', {})
            interaction_type = crosstalk.get('interaction_type', 'none')
            
            if interaction_type == 'direct':
                direct_kras.append({
                    'pmid': ext['metadata']['pmid'],
                    'title': ext['metadata']['title'],
                    'complex': crosstalk.get('molecular_complex', []),
                    'effects': crosstalk.get('downstream_effects', [])
                })
            elif interaction_type == 'indirect':
                indirect_kras.append(ext['metadata']['pmid'])
        
        report += f"**Direct KRAS Interaction**: {len(direct_kras)} papers\n\n"
        
        for paper in direct_kras:
            report += f"- **PMID {paper['pmid']}**: {paper['title'][:80]}\n"
            report += f"  - Complex: {', '.join(paper['complex'])}\n"
            report += f"  - Effects: {', '.join(paper['effects'][:3])}\n\n"
        
        report += f"\n**Indirect KRAS Association**: {len(indirect_kras)} papers\n\n"
        
        # Mechanisms summary
        report += "### PrPc Mechanisms Identified\n\n"
        
        all_receptors = set()
        all_pathways = set()
        all_processes = set()
        
        for ext in results['extractions']:
            mech = ext.get('prion_mechanisms', {})
            all_receptors.update(mech.get('receptor_interactions', []))
            all_pathways.update(mech.get('signaling_pathways', []))
            all_processes.update(mech.get('cellular_processes', []))
        
        report += f"**Receptor Interactions** ({len(all_receptors)}):\n"
        for rec in sorted(all_receptors):
            if rec:
                report += f"- {rec}\n"
        
        report += f"\n**Signaling Pathways** ({len(all_pathways)}):\n"
        for path in sorted(all_pathways):
            if path:
                report += f"- {path}\n"
        
        report += f"\n**Cellular Processes** ({len(all_processes)}):\n"
        for proc in sorted(all_processes):
            if proc:
                report += f"- {proc}\n"
        
        # Therapeutic potential
        report += "\n### Therapeutic Validation\n\n"
        
        preclinical_count = sum(1 for e in results['extractions'] 
                               if e.get('therapeutic_validation', {}).get('preclinical_evidence'))
        clinical_count = sum(1 for e in results['extractions'] 
                            if e.get('therapeutic_validation', {}).get('clinical_trials'))
        
        report += f"- Papers with preclinical evidence: {preclinical_count}\n"
        report += f"- Papers with clinical trial data: {clinical_count}\n"
        
        # Save report
        report_path = self.output_dir / "extraction_summary.md"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)
        
        print(f"Summary report saved to: {report_path}")
        return report


def main():
    """Run PrPc literature extraction pipeline"""
    
    print("="*70)
    print("PrPc/PRNP Literature Knowledge Extraction Pipeline")
    print("="*70)
    
    extractor = PrionLiteratureExtractor()
    
    # Step 1: Search for papers
    print("\nSTEP 1: Searching PubMed for PrPc cancer papers")
    print("-"*70)
    papers = extractor.search_prion_papers(max_papers=30)
    
    if not papers:
        print("No papers found. Exiting.")
        return
    
    # Step 2: Extract knowledge
    print("\nSTEP 2: Extracting knowledge with GPT-4")
    print("-"*70)
    results = extractor.batch_extract(papers, delay=1.0)
    
    # Step 3: Save results
    print("\nSTEP 3: Saving results")
    print("-"*70)
    extractor.save_results(results)
    
    # Step 4: Generate summary
    print("\nSTEP 4: Generating summary report")
    print("-"*70)
    extractor.generate_summary_report(results)
    
    print("\n" + "="*70)
    print("PIPELINE COMPLETE")
    print("="*70)


if __name__ == "__main__":
    main()

