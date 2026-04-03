"""
PrPc Systematic Literature Expansion
=====================================
Expands PrPc validation evidence base from 5 to 30+ papers
and from n=4 to n=10+ cancer types.

Comprehensive PubMed search for:
1. PrPc expression across multiple cancer types
2. KRAS mutation prevalence data
3. PrPc-KRAS interaction evidence
4. Preclinical and clinical studies

Usage:
    python scripts/prpc_systematic_literature_expansion.py
"""

import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional
import pandas as pd

from Bio import Entrez

# Fix Windows encoding for print statements
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')

# Configuration
Entrez.email = "adds.research@university.edu"
OUTPUT_DIR = Path("data/analysis/prpc_validation")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Target: 30+ papers across 10+ cancer types
SEARCH_QUERIES = {
    "prpc_expression": [
        # Core PrPc cancer expression studies
        "(PrPc OR PRNP OR \"prion protein\") AND (cancer OR tumor OR carcinoma) AND expression",
        "(cellular prion protein) AND (malignancy OR neoplasm) AND immunohistochemistry",
        
        # Specific cancer types (expanding from 4 to 10+)
        "PRNP AND (lung cancer OR NSCLC OR adenocarcinoma lung)",
        "PrPc AND (hepatocellular carcinoma OR liver cancer OR HCC)",
        "prion protein AND (ovarian cancer OR ovarian carcinoma)",
        "PRNP AND (prostate cancer OR prostate carcinoma)",
        "PrPc AND (esophageal cancer OR esophageal carcinoma)",
        "prion protein AND (renal cell carcinoma OR kidney cancer)",
    ],
    
    "prpc_kras": [
        # PrPc-KRAS interaction studies
        "(PrPc OR PRNP) AND KRAS AND (interaction OR signaling OR pathway)",
        "(prion protein) AND (RAS OR KRAS) AND (GTPase OR activation)",
        "PRNP AND KRAS AND (mutation OR mutant)",
        "(PrPc OR PRNP) AND RPSA AND KRAS",
    ],
    
    "prpc_therapy": [
        # Therapeutic targeting studies
        "(PrPc OR PRNP) AND (therapeutic target OR drug target OR antibody)",
        "(prion protein) AND (inhibitor OR antagonist) AND cancer",
        "PRNP AND (combination therapy OR synergy) AND chemotherapy",
    ]
}

# Additional cancer types to add (n=4 → n=10+)
ADDITIONAL_CANCER_TYPES = [
    "lung_cancer_nsclc",
    "liver_cancer_hcc",
    "ovarian_cancer",
    "prostate_cancer",
    "esophageal_cancer",
    "renal_cell_carcinoma",
]


class PrPcLiteratureExpander:
    """Systematic literature expansion for PrPc validation"""
    
    def __init__(self):
        self.results = {
            "search_date": datetime.now().isoformat(),
            "papers": [],
            "cancer_type_data": {},
            "summary_stats": {}
        }
    
    def search_pubmed_comprehensive(
        self, 
        query: str, 
        max_results: int = 50,
        min_year: int = 2015
    ) -> List[str]:
        """
        Comprehensive PubMed search with quality filters
        
        Args:
            query: Search query
            max_results: Max number of results
            min_year: Minimum publication year
            
        Returns:
            List of PMIDs
        """
        print(f"\n[SEARCH] Searching: {query[:80]}...")
        
        # Add quality filters
        search_query = f"{query} AND {min_year}:3000[PDAT]"
        
        try:
            handle = Entrez.esearch(
                db="pubmed",
                term=search_query,
                retmax=max_results,
                sort="relevance"
            )
            record = Entrez.read(handle)
            handle.close()
            
            pmids = record['IdList']
            print(f"   [OK] Found {len(pmids)} papers")
            
            time.sleep(0.4)  # Rate limiting
            return pmids
            
        except Exception as e:
            print(f"   [ERROR] Search failed: {e}")
            return []
    
    def fetch_article_metadata(self, pmids: List[str]) -> List[Dict]:
        """Fetch detailed metadata for articles"""
        if not pmids:
            return []
        
        print(f"\n[FETCH] Fetching metadata for {len(pmids)} articles...")
        
        metadata_list = []
        batch_size = 20
        
        for i in range(0, len(pmids), batch_size):
            batch = pmids[i:i+batch_size]
            
            try:
                handle = Entrez.efetch(
                    db="pubmed",
                    id=",".join(batch),
                    rettype="medline",
                    retmode="xml"
                )
                records = Entrez.read(handle)
                handle.close()
                
                for record in records.get('PubmedArticle', []):
                    metadata = self._extract_metadata(record)
                    if metadata:
                        metadata_list.append(metadata)
                
                print(f"   Processed {min(i+batch_size, len(pmids))}/{len(pmids)}")
                time.sleep(0.5)
                
            except Exception as e:
                print(f"   [ERROR] Batch failed: {e}")
                continue
        
        return metadata_list
    
    def _extract_metadata(self, record) -> Optional[Dict]:
        """Extract metadata from PubMed record"""
        try:
            article = record['MedlineCitation']['Article']
            pmid = str(record['MedlineCitation']['PMID'])
            
            # Basic info
            title = article.get('ArticleTitle', '')
            
            # Authors
            authors = []
            if 'AuthorList' in article:
                for author in article['AuthorList'][:5]:  # First 5 authors
                    if 'LastName' in author:
                        name = author['LastName']
                        if 'Initials' in author:
                            name += f" {author['Initials']}"
                        authors.append(name)
            
            # Journal and year
            journal = article.get('Journal', {}).get('Title', 'N/A')
            pub_date = article.get('Journal', {}).get('JournalIssue', {}).get('PubDate', {})
            year = pub_date.get('Year', 'N/A')
            
            # Abstract
            abstract = ""
            if 'Abstract' in article and 'AbstractText' in article['Abstract']:
                abstract_texts = article['Abstract']['AbstractText']
                if isinstance(abstract_texts, list):
                    abstract = " ".join([str(t) for t in abstract_texts])
                else:
                    abstract = str(abstract_texts)
            
            # DOI
            doi = None
            if 'ELocationID' in article:
                for eloc in article['ELocationID']:
                    if eloc.attributes.get('EIdType') == 'doi':
                        doi = str(eloc)
            
            # MeSH terms for classification
            mesh_terms = []
            if 'MeshHeadingList' in record['MedlineCitation']:
                for mesh in record['MedlineCitation']['MeshHeadingList'][:10]:
                    mesh_terms.append(str(mesh['DescriptorName']))
            
            return {
                'pmid': pmid,
                'title': title,
                'authors': authors,
                'journal': journal,
                'year': year,
                'doi': doi,
                'abstract': abstract,
                'mesh_terms': mesh_terms,
                'relevance_score': None,
                'cancer_types_mentioned': self._identify_cancer_types(title + " " + abstract),
                'has_expression_data': self._check_expression_data(abstract),
                'has_kras_mention': 'KRAS' in abstract.upper() or 'RAS' in abstract.upper(),
            }
            
        except Exception as e:
            print(f"   [WARNING] Metadata extraction error: {e}")
            return None
    
    def _identify_cancer_types(self, text: str) -> List[str]:
        """Identify cancer types mentioned in text"""
        text_lower = text.lower()
        cancer_types = []
        
        cancer_keywords = {
            'pancreatic': ['pancreatic', 'pancreas', 'pdac'],
            'colorectal': ['colorectal', 'colon', 'rectal', 'crc'],
            'gastric': ['gastric', 'stomach'],
            'breast': ['breast'],
            'lung': ['lung', 'nsclc', 'adenocarcinoma lung'],
            'liver': ['liver', 'hepatocellular', 'hcc'],
            'ovarian': ['ovarian', 'ovary'],
            'prostate': ['prostate'],
            'esophageal': ['esophageal', 'esophagus'],
            'renal': ['renal', 'kidney'],
        }
        
        for cancer_type, keywords in cancer_keywords.items():
            if any(kw in text_lower for kw in keywords):
                cancer_types.append(cancer_type)
        
        return cancer_types
    
    def _check_expression_data(self, abstract: str) -> bool:
        """Check if abstract contains expression data"""
        expression_keywords = [
            'expression', 'immunohistochemistry', 'ihc', 'protein level',
            'overexpression', 'upregulation', 'downregulation',
            'protein expression', 'mrna expression'
        ]
        abstract_lower = abstract.lower()
        return any(kw in abstract_lower for kw in expression_keywords)
    
    def run_comprehensive_search(self):
        """Execute comprehensive literature search"""
        print("=" * 70)
        print("PrPc SYSTEMATIC LITERATURE EXPANSION")
        print("=" * 70)
        print(f"Target: 30+ papers across 10+ cancer types")
        print(f"Previous: 5 papers across 4 cancer types")
        print()
        
        all_pmids = set()
        
        # Execute all search queries
        for category, queries in SEARCH_QUERIES.items():
            print(f"\n[CATEGORY] {category}")
            for query in queries:
                pmids = self.search_pubmed_comprehensive(query, max_results=30)
                all_pmids.update(pmids)
        
        print(f"\n\n=== SEARCH SUMMARY ===")
        print(f"   Total unique papers found: {len(all_pmids)}")
        
        # Fetch metadata
        papers = self.fetch_article_metadata(list(all_pmids))
        self.results['papers'] = papers
        
        # Analyze cancer type coverage
        self._analyze_cancer_coverage(papers)
        
        # Generate summary statistics
        self._generate_summary_stats(papers)
        
        # Save results
        self._save_results()
        
        return self.results
    
    def _analyze_cancer_coverage(self, papers: List[Dict]):
        """Analyze cancer type coverage"""
        print(f"\n\n=== CANCER TYPE COVERAGE ANALYSIS ===")
        
        cancer_coverage = {}
        for paper in papers:
            for cancer_type in paper.get('cancer_types_mentioned', []):
                if cancer_type not in cancer_coverage:
                    cancer_coverage[cancer_type] = []
                cancer_coverage[cancer_type].append(paper['pmid'])
        
        print(f"\n   Cancer types covered: {len(cancer_coverage)}")
        for cancer_type, pmids in sorted(cancer_coverage.items(), key=lambda x: len(x[1]), reverse=True):
            print(f"   • {cancer_type}: {len(pmids)} papers")
        
        self.results['cancer_type_data'] = cancer_coverage
    
    def _generate_summary_stats(self, papers: List[Dict]):
        """Generate summary statistics"""
        stats = {
            'total_papers': len(papers),
            'papers_with_expression_data': sum(1 for p in papers if p.get('has_expression_data')),
            'papers_with_kras_mention': sum(1 for p in papers if p.get('has_kras_mention')),
            'cancer_types_covered': len(self.results['cancer_type_data']),
            'year_distribution': {},
        }
        
        # Year distribution
        for paper in papers:
            year = paper.get('year', 'N/A')
            stats['year_distribution'][year] = stats['year_distribution'].get(year, 0) + 1
        
        self.results['summary_stats'] = stats
        
        print(f"\n\n=== SUMMARY STATISTICS ===")
        print(f"   Total papers: {stats['total_papers']}")
        print(f"   With expression data: {stats['papers_with_expression_data']}")
        print(f"   With KRAS mention: {stats['papers_with_kras_mention']}")
        print(f"   Cancer types covered: {stats['cancer_types_covered']}")
    
    def _save_results(self):
        """Save results to JSON and Excel"""
        # Save JSON
        json_path = OUTPUT_DIR / "prpc_expanded_literature.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)
        print(f"\n[SAVED] JSON: {json_path}")
        
        # Save Excel summary
        if self.results['papers']:
            df = pd.DataFrame(self.results['papers'])
            excel_path = OUTPUT_DIR / "prpc_expanded_literature.xlsx"
            df.to_excel(excel_path, index=False, engine='openpyxl')
            print(f"[SAVED] Excel: {excel_path}")


def main():
    """Main execution"""
    expander = PrPcLiteratureExpander()
    results = expander.run_comprehensive_search()
    
    print("\n" + "=" * 70)
    print("[COMPLETE] LITERATURE EXPANSION COMPLETE")
    print("=" * 70)
    print(f"\nFound {results['summary_stats']['total_papers']} papers")
    print(f"Covering {results['summary_stats']['cancer_types_covered']} cancer types")
    print(f"\nOriginal: 5 papers, 4 cancer types")
    print(f"Updated:  {results['summary_stats']['total_papers']} papers, {results['summary_stats']['cancer_types_covered']} cancer types")
    
    increase_papers = results['summary_stats']['total_papers'] - 5
    increase_types = results['summary_stats']['cancer_types_covered'] - 4
    print(f"\nIncrease: +{increase_papers} papers, +{increase_types} cancer types")


if __name__ == "__main__":
    main()
