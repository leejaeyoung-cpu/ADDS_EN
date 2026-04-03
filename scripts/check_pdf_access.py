"""
PDF Accessibility Checker and Downloader
=========================================
Checks which collected papers are available as open-access PDFs
and downloads them for knowledge extraction.

Author: ADDS Team
Date: 2026-01-31
"""

import json
import requests
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from urllib.parse import quote
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration
METADATA_FILES = [
    "data/literature/collection_progress.json",  # Tier S
    "data/literature/comprehensive_metadata.json"  # Tier A
]
PDF_OUTPUT_DIR = Path("data/literature/pdfs")
PDF_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# PubMed Central Open Access base URL
PMC_OA_BASE = "https://www.ncbi.nlm.nih.gov/pmc/utils/oa/oa.fcgi"


class PDFAccessChecker:
    """Check PDF accessibility for collected papers"""
    
    def __init__(self):
        self.open_access_papers = []
        self.paywall_papers = []
        self.unknown_papers = []
    
    def check_pmc_oa(self, pmid: str) -> Tuple[bool, Optional[str]]:
        """
        Check if paper is available via PubMed Central Open Access
        
        Returns:
            (is_available, pdf_url)
        """
        try:
            url = f"{PMC_OA_BASE}?id=PMID:{pmid}"
            response = requests.get(url, timeout=10)
            
            if response.status_code == 200:
                # Check if XML contains PDF link
                if 'error' not in response.text.lower():
                    # Try to extract PDF URL from XML
                    # This is simplified - would need proper XML parsing
                    if '.pdf' in response.text:
                        logger.info(f"PMID {pmid}: Open Access available")
                        return True, None  # URL extraction would go here
                    
            return False, None
            
        except Exception as e:
            logger.warning(f"Error checking PMID {pmid}: {e}")
            return False, None
    
    def check_doi_resolver(self, doi: str) -> Tuple[bool, Optional[str]]:
        """
        Check if DOI resolves to open access version
        
        Returns:
            (is_available, pdf_url)
        """
        if not doi:
            return False, None
        
        try:
            # Try Unpaywall API (free, no key needed for basic use)
            url = f"https://api.unpaywall.org/v2/{doi}?email=researcher@university.edu"
            response = requests.get(url, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                
                # Check if there's an open access location
                if data.get('is_oa', False):
                    oa_locations = data.get('oa_locations', [])
                    if oa_locations:
                        pdf_url = oa_locations[0].get('url_for_pdf')
                        if pdf_url:
                            logger.info(f"DOI {doi}: Open Access PDF found")
                            return True, pdf_url
                        
            return False, None
            
        except Exception as e:
            logger.warning(f"Error checking DOI {doi}: {e}")
            return False, None
    
    def assess_paper(self, paper: Dict) -> Dict:
        """
        Assess PDF availability for a single paper
        
        Returns:
            Updated paper dict with accessibility info
        """
        pmid = paper.get('pmid')
        doi = paper.get('doi')
        title = paper.get('title', 'Unknown')[:60]
        
        logger.info(f"Checking: {title}...")
        
        # Try PMC first
        pmc_available, pmc_url = self.check_pmc_oa(pmid)
        if pmc_available:
            paper['pdf_available'] = True
            paper['pdf_source'] = 'PMC'
            paper['pdf_url'] = pmc_url
            self.open_access_papers.append(paper)
            return paper
        
        # Try DOI resolver (Unpaywall)
        time.sleep(0.5)  # Rate limiting
        
        doi_available, doi_url = self.check_doi_resolver(doi)
        if doi_available:
            paper['pdf_available'] = True
            paper['pdf_source'] = 'Unpaywall'
            paper['pdf_url'] = doi_url
            self.open_access_papers.append(paper)
            return paper
        
        # Not available
        paper['pdf_available'] = False
        paper['pdf_source'] = 'Paywall'
        paper['pdf_url'] = None
        self.paywall_papers.append(paper)
        
        return paper
    
    def assess_all_papers(self) -> Dict:
        """
        Assess all collected papers
        
        Returns:
            Dictionary with assessment results
        """
        all_papers = []
        
        # Load Tier S papers
        logger.info("Loading Tier S papers...")
        try:
            with open(METADATA_FILES[0], 'r', encoding='utf-8') as f:
                tier_s_data = json.load(f)
                tier_s_papers = tier_s_data.get('papers', [])
                all_papers.extend(tier_s_papers)
                logger.info(f"Loaded {len(tier_s_papers)} Tier S papers")
        except Exception as e:
            logger.error(f"Error loading Tier S: {e}")
        
        # Load Tier A papers
        logger.info("Loading Tier A papers...")
        try:
            with open(METADATA_FILES[1], 'r', encoding='utf-8') as f:
                tier_a_data = json.load(f)
                tier_a_papers = tier_a_data.get('papers', [])
                all_papers.extend(tier_a_papers)
                logger.info(f"Loaded {len(tier_a_papers)} Tier A papers")
        except Exception as e:
            logger.error(f"Error loading Tier A: {e}")
        
        logger.info(f"\nTotal papers to assess: {len(all_papers)}")
        logger.info("="*70)
        
        # Assess each paper
        for i, paper in enumerate(all_papers, 1):
            logger.info(f"\n[{i}/{len(all_papers)}]")
            assessed = self.assess_paper(paper)
            
            # Rate limiting
            if i % 10 == 0:
                logger.info(f"\nProgress: {i}/{len(all_papers)} assessed")
                logger.info(f"Open Access so far: {len(self.open_access_papers)}")
                time.sleep(2)
        
        # Generate summary
        summary = {
            'total_papers': len(all_papers),
            'open_access': len(self.open_access_papers),
            'paywall': len(self.paywall_papers),
            'percentage_oa': len(self.open_access_papers) / len(all_papers) * 100 if all_papers else 0,
            'open_access_papers': self.open_access_papers,
            'paywall_papers': self.paywall_papers
        }
        
        return summary


class PDFDownloader:
    """Download open-access PDFs"""
    
    def __init__(self, output_dir: Path = PDF_OUTPUT_DIR):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.downloaded = []
        self.failed = []
    
    def download_pdf(self, paper: Dict) -> bool:
        """
        Download PDF for a paper
        
        Returns:
            Success status
        """
        pmid = paper.get('pmid')
        pdf_url = paper.get('pdf_url')
        title = paper.get('title', 'Unknown')[:40]
        
        if not pdf_url:
            logger.warning(f"No PDF URL for PMID {pmid}")
            return False
        
        output_path = self.output_dir / f"{pmid}.pdf"
        
        # Skip if already downloaded
        if output_path.exists():
            logger.info(f"Already exists: {pmid}")
            self.downloaded.append(str(output_path))
            return True
        
        try:
            logger.info(f"Downloading PMID {pmid}: {title}...")
            
            response = requests.get(pdf_url, timeout=30, stream=True)
            response.raise_for_status()
            
            # Check if response is actually a PDF
            content_type = response.headers.get('content-type', '').lower()
            if 'pdf' not in content_type and 'application/pdf' not in content_type:
                logger.warning(f"Not a PDF: {content_type}")
                self.failed.append(pmid)
                return False
            
            # Download
            with open(output_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            logger.info(f"[SUCCESS] Downloaded: {output_path}")
            self.downloaded.append(str(output_path))
            return True
            
        except Exception as e:
            logger.error(f"[FAILED] PMID {pmid}: {e}")
            self.failed.append(pmid)
            return False
    
    def download_all(self, papers: List[Dict]) -> Dict:
        """
        Download all available PDFs
        
        Returns:
            Download summary
        """
        logger.info(f"\nDownloading {len(papers)} PDFs...")
        logger.info("="*70)
        
        for i, paper in enumerate(papers, 1):
            logger.info(f"\n[{i}/{len(papers)}]")
            self.download_pdf(paper)
            
            # Rate limiting
            time.sleep(1)
            
            if i % 5 == 0:
                logger.info(f"\nProgress: {i}/{len(papers)} attempted")
                logger.info(f"Downloaded: {len(self.downloaded)}")
                logger.info(f"Failed: {len(self.failed)}")
        
        return {
            'attempted': len(papers),
            'downloaded': len(self.downloaded),
            'failed': len(self.failed),
            'success_rate': len(self.downloaded) / len(papers) * 100 if papers else 0,
            'pdf_paths': self.downloaded
        }


def main():
    """Main execution"""
    
    logger.info("="*70)
    logger.info("PDF ACCESSIBILITY ASSESSMENT")
    logger.info("="*70)
    
    # Step 1: Assess availability
    checker = PDFAccessChecker()
    assessment = checker.assess_all_papers()
    
    # Print summary
    logger.info("\n" + "="*70)
    logger.info("ASSESSMENT SUMMARY")
    logger.info("="*70)
    logger.info(f"Total papers: {assessment['total_papers']}")
    logger.info(f"Open Access: {assessment['open_access']} ({assessment['percentage_oa']:.1f}%)")
    logger.info(f"Paywall: {assessment['paywall']}")
    
    # Save assessment results
    assessment_file = Path("data/literature/pdf_assessment.json")
    with open(assessment_file, 'w', encoding='utf-8') as f:
        json.dump(assessment, f, indent=2, ensure_ascii=False)
    logger.info(f"\nAssessment saved to: {assessment_file}")
    
    # Step 2: Download available PDFs
    if assessment['open_access'] > 0:
        logger.info("\n" + "="*70)
        logger.info("PDF DOWNLOAD")
        logger.info("="*70)
        
        downloader = PDFDownloader()
        download_results = downloader.download_all(assessment['open_access_papers'])
        
        # Print download summary
        logger.info("\n" + "="*70)
        logger.info("DOWNLOAD SUMMARY")
        logger.info("="*70)
        logger.info(f"Attempted: {download_results['attempted']}")
        logger.info(f"Downloaded: {download_results['downloaded']} ({download_results['success_rate']:.1f}%)")
        logger.info(f"Failed: {download_results['failed']}")
        logger.info(f"Output directory: {PDF_OUTPUT_DIR}")
        
        # Save download results
        download_file = Path("data/literature/pdf_downloads.json")
        with open(download_file, 'w', encoding='utf-8') as f:
            json.dump(download_results, f, indent=2, ensure_ascii=False)
        logger.info(f"\nDownload results saved to: {download_file}")
    
    else:
        logger.warning("\nNo open-access PDFs found!")
        logger.info("Recommendation: Use institutional access or focus on abstract-only extraction")
    
    logger.info("\n" + "="*70)
    logger.info("COMPLETE")
    logger.info("="*70)


if __name__ == "__main__":
    main()
