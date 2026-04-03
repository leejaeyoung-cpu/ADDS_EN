"""
Automated Literature Collection System
========================================
Comprehensive system for collecting 200+ high-quality cancer research papers
from Nature-grade and SCI journals.

Author: ADDS Team
Date: 2026-01-31
"""

import argparse
import json
import time
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from scripts.pubmed_literature_search import PubMedSearcher, PDFDownloader

# Configuration
OUTPUT_DIR = Path("data/literature")
METADATA_FILE = OUTPUT_DIR / "comprehensive_metadata.json"

# Journal tiers
TIER_S_JOURNALS = [
    "Nature", "Nature Medicine", "Nature Cancer",
    "Cell", "Science", 
    "New England Journal of Medicine", "NEJM",
    "Lancet", "Lancet Oncology"
]

TIER_A_JOURNALS = [
    "Clinical Cancer Research",
    "Cancer Research",
    "Journal of Clinical Oncology",
    "Molecular Cancer Therapeutics",
    "Cancer Cell"
]

# Search queries by cancer type
SEARCH_QUERIES = {
    'colorectal': [
        "colorectal cancer KRAS mutation EGFR inhibitor resistance",
        "FOLFOX combination therapy synergy molecular mechanism",
        "MSI-H colorectal cancer immunotherapy PD-1 pembrolizumab",
        "bevacizumab VEGF angiogenesis colorectal cancer",
        "colorectal cancer liver metastasis treatment strategy",
        "RAS RAF MEK ERK pathway colorectal cancer targeted therapy",
        "PI3K AKT mTOR pathway colorectal cancer inhibitor",
        "colorectal cancer stem cells drug resistance mechanism",
        "cetuximab panitumumab anti-EGFR colorectal cancer",
        "regorafenib multi-kinase inhibitor colorectal cancer"
    ],
    'gastric': [
        "gastric cancer HER2 trastuzumab targeted therapy",
        "gastric cancer PD-L1 nivolumab pembrolizumab immunotherapy",
        "gastric cancer claudin-18.2 zolbetuximab therapy",
        "gastric adenocarcinoma molecular subtype classification",
        "gastric cancer MSI-H immunotherapy response",
        "ramucirumab VEGFR2 inhibitor gastric cancer",
        "gastric cancer chemotherapy resistance mechanism"
    ],
    'lung': [
        "non-small cell lung cancer EGFR mutation targeted therapy",
        "ALK rearrangement lung cancer targeted therapy",
        "lung cancer PD-1 PD-L1 checkpoint inhibitor",
        "lung cancer KRAS G12C inhibitor sotorasib",
        "lung adenocarcinoma molecular profiling",
        "small cell lung cancer immunotherapy combination"
    ],
    'breast': [
        "breast cancer HER2 positive trastuzumab pertuzumab",
        "triple negative breast cancer immunotherapy",
        "breast cancer CDK4/6 inhibitor palbociclib",
        "breast cancer PI3K mutation targeted therapy",
        "breast cancer PARP inhibitor BRCA mutation"
    ],
    'pancreatic': [
        "pancreatic cancer FOLFIRINOX therapy",
        "pancreatic adenocarcinoma gemcitabine resistance",
        "pancreatic cancer KRAS mutation targeted therapy",
        "pancreatic cancer immunotherapy strategies"
    ],
    'liver': [
        "hepatocellular carcinoma sorafenib lenvatinib",
        "liver cancer atezolizumab bevacizumab immunotherapy",
        "HCC molecular classification targeted therapy"
    ],
    'pan_cancer': [
        "cancer signaling pathways targeted therapy",
        "tumor microenvironment immune checkpoint",
        "cancer drug resistance mechanisms molecular",
        "precision oncology biomarker guided therapy",
        "cancer metabolism targeted therapy",
        "DNA damage response cancer therapy"
    ]
}


class LiteratureCollector:
    """Automated collection of high-quality cancer research papers"""
    
    def __init__(self, output_dir: Path = OUTPUT_DIR):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.searcher = PubMedSearcher(output_dir)
        self.downloader = PDFDownloader(output_dir / "pdfs")
        self.collected_papers = []
        
    def collect_by_cancer_type(
        self, 
        cancer_type: str, 
        target_count: int,
        tier: str = 'S',
        min_year: int = 2018
    ) -> List[Dict]:
        """
        Collect papers for specific cancer type
        
        Args:
            cancer_type: Cancer type key (colorectal, gastric, etc.)
            target_count: Number of papers to collect
            tier: Journal tier ('S' or 'A')
            min_year: Minimum publication year
            
        Returns:
            List of paper metadata
        """
        print(f"\n{'='*70}")
        print(f"[*] Collecting {target_count} Tier-{tier} papers for {cancer_type.upper()} cancer")
        print('='*70)
        
        queries = SEARCH_QUERIES.get(cancer_type, [])
        if not queries:
            print(f"[WARN]  No queries defined for {cancer_type}")
            return []
        
        journal_filter = TIER_S_JOURNALS if tier == 'S' else TIER_A_JOURNALS
        papers_per_query = max(target_count // len(queries), 10)
        
        all_pmids = set()
        all_metadata = []
        
        for i, query in enumerate(queries, 1):
            print(f"\n[QUERY] Query {i}/{len(queries)}: {query[:60]}...")
            
            try:
                pmids = self.searcher.search_pubmed(
                    query=query,
                    max_results=papers_per_query,
                    min_year=min_year,
                    journal_filter=journal_filter
                )
                
                # Remove duplicates
                new_pmids = [p for p in pmids if p not in all_pmids]
                all_pmids.update(new_pmids)
                
                print(f"   Found {len(new_pmids)} new papers (total: {len(all_pmids)})")
                
                # Rate limiting
                time.sleep(0.5)
                
            except Exception as e:
                print(f"   [ERROR] {e}")
                continue
        
        # Fetch metadata in batches
        if all_pmids:
            print(f"\nFetching metadata for {len(all_pmids)} papers...")
            metadata = self.searcher.fetch_metadata(list(all_pmids))
            
            # Add cancer type label
            for meta in metadata:
                meta['cancer_type'] = cancer_type
                meta['tier'] = tier
            
            all_metadata.extend(metadata)
        
        # Trim to target count (keep highest quality)
        if len(all_metadata) > target_count:
            all_metadata = self._rank_and_filter(all_metadata, target_count)
        
        print(f"\n[OK] Collected {len(all_metadata)} papers for {cancer_type}")
        return all_metadata
    
    def _rank_and_filter(self, papers: List[Dict], target: int) -> List[Dict]:
        """Rank papers by quality and filter to target count"""
        
        # Score based on journal impact and relevance
        journal_scores = {
            "Nature": 10, "Nature Medicine": 9, "Nature Cancer": 9,
            "Cell": 10, "Science": 10,
            "New England Journal of Medicine": 10, "NEJM": 10,
            "Lancet": 9, "Lancet Oncology": 8,
            "Journal of Clinical Oncology": 8,
            "Clinical Cancer Research": 7,
            "Cancer Research": 7,
            "Cancer Cell": 8,
            "Molecular Cancer Therapeutics": 6
        }
        
        for paper in papers:
            journal = paper.get('journal', '')
            base_score = journal_scores.get(journal, 5)
            
            # Boost recent papers
            year = int(paper.get('publication_year', 2018))
            recency_boost = (year - 2018) * 0.1
            
            # Boost papers with abstracts
            abstract_boost = 1.0 if paper.get('abstract') else 0.0
            
            paper['quality_score'] = base_score + recency_boost + abstract_boost
        
        # Sort by quality and take top N
        papers.sort(key=lambda x: x.get('quality_score', 0), reverse=True)
        return papers[:target]
    
    def run_full_collection(
        self,
        tier_s_count: int = 100,
        tier_a_count: int = 100
    ) -> Dict[str, List[Dict]]:
        """
        Run complete collection for all cancer types
        
        Args:
            tier_s_count: Total Tier S papers to collect
            tier_a_count: Total Tier A papers to collect
            
        Returns:
            Dictionary of collected papers by cancer type
        """
        
        # Distribution by cancer type
        cancer_distribution = {
            'colorectal': 40,
            'gastric': 30,
            'lung': 30,
            'breast': 30,
            'pancreatic': 20,
            'liver': 20,
            'pan_cancer': 30
        }
        
        all_collections = {
            'tier_s': {},
            'tier_a': {}
        }
        
        # Collect Tier S papers
        print("\n" + "="*70)
        print("[PHASE] PHASE 1: Collecting Tier S Papers (Nature, Cell, NEJM, etc.)")
        print("="*70)
        
        for cancer_type, base_count in cancer_distribution.items():
            tier_s_target = int(base_count * tier_s_count / 200)
            papers = self.collect_by_cancer_type(
                cancer_type,
                tier_s_target,
                tier='S'
            )
            all_collections['tier_s'][cancer_type] = papers
            self.collected_papers.extend(papers)
            
            # Save intermediate results
            self.save_progress()
            time.sleep(1)
        
        # Collect Tier A papers
        print("\n" + "="*70)
        print("[PHASE] PHASE 2: Collecting Tier A Papers (JCO, Cancer Research, etc.)")
        print("="*70)
        
        for cancer_type, base_count in cancer_distribution.items():
            tier_a_target = int(base_count * tier_a_count / 200)
            papers = self.collect_by_cancer_type(
                cancer_type,
                tier_a_target,
                tier='A'
            )
            all_collections['tier_a'][cancer_type] = papers
            self.collected_papers.extend(papers)
            
            # Save intermediate results
            self.save_progress()
            time.sleep(1)
        
        # Final save
        self.save_final_metadata(all_collections)
        
        return all_collections
    
    def save_progress(self):
        """Save progress to intermediate file"""
        progress_file = self.output_dir / "collection_progress.json"
        with open(progress_file, 'w', encoding='utf-8') as f:
            json.dump({
                'timestamp': datetime.now().isoformat(),
                'total_papers': len(self.collected_papers),
                'papers': self.collected_papers
            }, f, indent=2, ensure_ascii=False)
    
    def save_final_metadata(self, collections: Dict):
        """Save final metadata with statistics"""
        
        # Calculate statistics
        stats = {
            'total_papers': len(self.collected_papers),
            'tier_s_count': sum(len(p) for p in collections['tier_s'].values()),
            'tier_a_count': sum(len(p) for p in collections['tier_a'].values()),
            'by_cancer_type': {},
            'by_journal': {},
            'year_distribution': {}
        }
        
        for paper in self.collected_papers:
            # By cancer type
            ct = paper.get('cancer_type', 'unknown')
            stats['by_cancer_type'][ct] = stats['by_cancer_type'].get(ct, 0) + 1
            
            # By journal
            journal = paper.get('journal', 'unknown')
            stats['by_journal'][journal] = stats['by_journal'].get(journal, 0) + 1
            
            # By year
            year = paper.get('publication_year', 'unknown')
            stats['year_distribution'][str(year)] = stats['year_distribution'].get(str(year), 0) + 1
        
        output = {
            'metadata': {
                'collection_date': datetime.now().isoformat(),
                'total_papers': len(self.collected_papers),
                'version': '1.0'
            },
            'statistics': stats,
            'papers': self.collected_papers
        }
        
        with open(METADATA_FILE, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        
        print(f"\n[SAVE] Final metadata saved to: {METADATA_FILE}")
        print(f"\n[STATS] Collection Statistics:")
        print(f"   Total papers: {stats['total_papers']}")
        print(f"   Tier S: {stats['tier_s_count']}")
        print(f"   Tier A: {stats['tier_a_count']}")
        print(f"\n   By Cancer Type:")
        for cancer, count in sorted(stats['by_cancer_type'].items()):
            print(f"      {cancer}: {count}")
    
    def generate_collection_report(self):
        """Generate detailed collection report"""
        report_file = self.output_dir / "collection_report.md"
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("# Literature Collection Report\n\n")
            f.write(f"**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n")
            f.write(f"**Total Papers Collected**: {len(self.collected_papers)}\n\n")
            
            f.write("## Summary by Cancer Type\n\n")
            f.write("| Cancer Type | Tier S | Tier A | Total |\n")
            f.write("|-------------|--------|--------|-------|\n")
            
            cancer_summary = {}
            for paper in self.collected_papers:
                ct = paper.get('cancer_type', 'unknown')
                tier = paper.get('tier', 'unknown')
                if ct not in cancer_summary:
                    cancer_summary[ct] = {'S': 0, 'A': 0}
                cancer_summary[ct][tier] = cancer_summary[ct].get(tier, 0) + 1
            
            for ct, counts in sorted(cancer_summary.items()):
                total = counts.get('S', 0) + counts.get('A', 0)
                f.write(f"| {ct.title()} | {counts.get('S', 0)} | {counts.get('A', 0)} | {total} |\n")
        
        print(f"\n[REPORT] Collection report saved to: {report_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Automated Literature Collection for ADDS Knowledge Base"
    )
    parser.add_argument(
        '--tier_s', 
        type=int, 
        default=100, 
        help='Number of Tier S papers to collect'
    )
    parser.add_argument(
        '--tier_a', 
        type=int, 
        default=100, 
        help='Number of Tier A papers to collect'
    )
    parser.add_argument(
        '--cancer_type',
        type=str,
        choices=['colorectal', 'gastric', 'lung', 'breast', 'pancreatic', 'liver', 'pan_cancer', 'all'],
        default='all',
        help='Specific cancer type to collect (default: all)'
    )
    parser.add_argument(
        '--dry_run',
        action='store_true',
        help='Test run without actual collection'
    )
    
    args = parser.parse_args()
    
    if args.dry_run:
        print("[TEST] DRY RUN MODE - No actual collection")
        return
    
    collector = LiteratureCollector()
    
    if args.cancer_type == 'all':
        # Full collection
        collections = collector.run_full_collection(
            tier_s_count=args.tier_s,
            tier_a_count=args.tier_a
        )
    else:
        # Single cancer type
        papers_s = collector.collect_by_cancer_type(
            args.cancer_type, 
            args.tier_s, 
            tier='S'
        )
        papers_a = collector.collect_by_cancer_type(
            args.cancer_type, 
            args.tier_a, 
            tier='A'
        )
        collector.collected_papers.extend(papers_s + papers_a)
        collector.save_progress()
    
    # Generate report
    collector.generate_collection_report()
    
    print("\n[SUCCESS] Collection Complete!")
    print(f"[OUTPUT] Output directory: {OUTPUT_DIR}")
    print(f"[TOTAL] Total papers: {len(collector.collected_papers)}")


if __name__ == "__main__":
    main()

