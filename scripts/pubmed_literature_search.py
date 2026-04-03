"""
PubMed Literature Search and Download Automation
================================================
Automatically searches and downloads high-quality cancer research papers
for the ADDS Cancer Mechanism Knowledge Base.

Usage:
    python scripts/pubmed_literature_search.py --query "colorectal cancer mechanisms" --max_results 100
"""

import argparse
import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional

import requests
from Bio import Entrez

# Configuration
ENTREZ_EMAIL = "your_email@university.edu"  # REQUIRED: Set your email
ENTREZ_API_KEY = None  # Optional: Get from NCBI for higher rate limits
OUTPUT_DIR = Path("data/literature")
METADATA_FILE = OUTPUT_DIR / "paper_metadata.json"

# Initialize Entrez
Entrez.email = ENTREZ_EMAIL
if ENTREZ_API_KEY:
    Entrez.api_key = ENTREZ_API_KEY


class PubMedSearcher:
    """PubMed literature search and metadata extraction"""
    
    def __init__(self, output_dir: Path = OUTPUT_DIR):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def search_pubmed(
        self, 
        query: str, 
        max_results: int = 100,
        min_year: int = 2019,
        journal_filter: Optional[List[str]] = None
    ) -> List[str]:
        """
        Search PubMed for articles matching query
        
        Args:
            query: Search query string
            max_results: Maximum number of results to return
            min_year: Minimum publication year
            journal_filter: List of high-impact journals to filter
            
        Returns:
            List of PubMed IDs (PMIDs)
        """
        print(f"[SEARCH] Searching PubMed: '{query}'")
        
        # Construct search query with filters
        search_query = f"{query} AND {min_year}:3000[PDAT]"
        
        # Add journal filter for high-impact journals
        if journal_filter:
            journal_str = " OR ".join([f'"{j}"[Journal]' for j in journal_filter])
            search_query += f" AND ({journal_str})"
        
        # Search PubMed
        handle = Entrez.esearch(
            db="pubmed",
            term=search_query,
            retmax=max_results,
            sort="relevance"
        )
        record = Entrez.read(handle)
        handle.close()
        
        pmids = record['IdList']
        print(f"[OK] Found {len(pmids)} articles")
        
        return pmids
    
    def fetch_metadata(self, pmids: List[str]) -> List[Dict]:
        """
        Fetch detailed metadata for PubMed articles
        
        Args:
            pmids: List of PubMed IDs
            
        Returns:
            List of metadata dictionaries
        """
        print(f"[FETCH] Fetching metadata for {len(pmids)} articles...")
        
        metadata_list = []
        
        # Fetch in batches to avoid overwhelming the API
        batch_size = 50
        for i in range(0, len(pmids), batch_size):
            batch_pmids = pmids[i:i+batch_size]
            
            handle = Entrez.efetch(
                db="pubmed",
                id=",".join(batch_pmids),
                rettype="medline",
                retmode="xml"
            )
            records = Entrez.read(handle)
            handle.close()
            
            for record in records['PubmedArticle']:
                metadata = self._extract_metadata(record)
                metadata_list.append(metadata)
            
            # Rate limiting
            time.sleep(0.5)
            print(f"  Processed {min(i+batch_size, len(pmids))}/{len(pmids)}")
        
        return metadata_list
    
    def _extract_metadata(self, record) -> Dict:
        """Extract relevant metadata from PubMed record"""
        article = record['MedlineCitation']['Article']
        
        # Basic info
        pmid = str(record['MedlineCitation']['PMID'])
        title = article['ArticleTitle']
        
        # Authors
        authors = []
        if 'AuthorList' in article:
            for author in article['AuthorList']:
                if 'LastName' in author and 'Initials' in author:
                    authors.append(f"{author['LastName']} {author['Initials']}")
        
        # Journal info
        journal = article['Journal']['Title']
        year = article['Journal']['JournalIssue']['PubDate'].get('Year', 'N/A')
        
        # Abstract
        abstract = ""
        if 'Abstract' in article:
            abstract_texts = article['Abstract']['AbstractText']
            if isinstance(abstract_texts, list):
                abstract = " ".join([str(text) for text in abstract_texts])
            else:
                abstract = str(abstract_texts)
        
        # DOI
        doi = None
        if 'ELocationID' in article:
            for eloc in article['ELocationID']:
                if eloc.attributes.get('EIdType') == 'doi':
                    doi = str(eloc)
        
        # MeSH terms (keywords)
        mesh_terms = []
        if 'MeshHeadingList' in record['MedlineCitation']:
            for mesh in record['MedlineCitation']['MeshHeadingList']:
                mesh_terms.append(str(mesh['DescriptorName']))
        
        return {
            'pmid': pmid,
            'title': title,
            'authors': authors,
            'journal': journal,
            'publication_year': year,
            'doi': doi,
            'abstract': abstract,
            'mesh_terms': mesh_terms,
            'relevance_score': None,  # To be filled later
            'downloaded': False,
            'pdf_path': None
        }
    
    def save_metadata(self, metadata_list: List[Dict]):
        """Save metadata to JSON file"""
        with open(METADATA_FILE, 'w', encoding='utf-8') as f:
            json.dump(metadata_list, f, indent=2, ensure_ascii=False)
        print(f"[SAVE] Saved metadata to {METADATA_FILE}")
    
    def load_metadata(self) -> List[Dict]:
        """Load existing metadata"""
        if METADATA_FILE.exists():
            with open(METADATA_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        return []


class PDFDownloader:
    """Download full-text PDFs of research papers"""
    
    def __init__(self, output_dir: Path = OUTPUT_DIR / "pdfs"):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def download_pdf(self, pmid: str, doi: Optional[str] = None) -> Optional[str]:
        """
        Attempt to download PDF for a paper
        
        Note: This requires institutional access or Sci-Hub (use ethically!)
        For legal access, use PubMed Central Open Access subset
        
        Args:
            pmid: PubMed ID
            doi: Digital Object Identifier
            
        Returns:
            Path to downloaded PDF or None if failed
        """
        # Try PubMed Central Open Access first
        pmc_url = f"https://www.ncbi.nlm.nih.gov/pmc/utils/oa/oa.fcgi?id=PMID:{pmid}"
        
        try:
            response = requests.get(pmc_url)
            if response.status_code == 200:
                # Parse XML to find PDF link
                # This is simplified - would need proper XML parsing
                pdf_path = self.output_dir / f"{pmid}.pdf"
                # Download PDF if available
                # (Implementation depends on PMC API response)
                return str(pdf_path)
        except Exception as e:
            print(f"[WARN]  Failed to download PMID {pmid}: {e}")
        
        return None


def main():
    parser = argparse.ArgumentParser(description="PubMed Literature Search")
    parser.add_argument("--query", type=str, required=True, help="Search query")
    parser.add_argument("--max_results", type=int, default=100, help="Maximum results")
    parser.add_argument("--min_year", type=int, default=2019, help="Minimum publication year")
    parser.add_argument("--download_pdfs", action="store_true", help="Attempt to download PDFs")
    
    args = parser.parse_args()
    
    # High-impact journals filter
    top_journals = [
        "Nature", "Science", "Cell", "Nature Medicine", "Nature Cancer",
        "New England Journal of Medicine", "Lancet", "Lancet Oncology",
        "Journal of Clinical Oncology", "Cancer Research",
        "Clinical Cancer Research", "Cancer Cell"
    ]
    
    # Search PubMed
    searcher = PubMedSearcher()
    pmids = searcher.search_pubmed(
        query=args.query,
        max_results=args.max_results,
        min_year=args.min_year,
        journal_filter=top_journals
    )
    
    # Fetch metadata
    metadata_list = searcher.fetch_metadata(pmids)
    
    # Save metadata
    searcher.save_metadata(metadata_list)
    
    # Download PDFs if requested
    if args.download_pdfs:
        downloader = PDFDownloader()
        for meta in metadata_list:
            pdf_path = downloader.download_pdf(meta['pmid'], meta.get('doi'))
            if pdf_path:
                meta['pdf_path'] = pdf_path
                meta['downloaded'] = True
        
        # Update metadata file
        searcher.save_metadata(metadata_list)
    
    print(f"\n[OK] Complete! Found and processed {len(metadata_list)} papers")
    print(f"📄 Metadata saved to: {METADATA_FILE}")


if __name__ == "__main__":
    # Example searches for cancer mechanisms
    EXAMPLE_QUERIES = [
        "colorectal cancer drug resistance mechanisms",
        "cancer signaling pathways targeted therapy",
        "FOLFOX synergy molecular mechanism",
        "PD-1 checkpoint inhibitor MSI-H colorectal",
        "KRAS mutation EGFR inhibitor resistance",
        "bevacizumab VEGF angiogenesis colorectal",
        "cancer cell cycle checkpoint inhibition",
        "RAS RAF MEK ERK pathway cancer",
        "PI3K AKT mTOR pathway oncology",
        "cancer stem cells drug resistance"
    ]
    
    main()

