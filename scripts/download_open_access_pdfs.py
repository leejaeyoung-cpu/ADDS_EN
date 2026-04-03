"""
Download Open-Access PDFs
==========================
Download all open-access PDFs from the assessed papers

Author: ADDS Team
Date: 2026-01-31
"""

import json
import time
import requests
from pathlib import Path
from tqdm import tqdm
from typing import Dict, List
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

# Configuration
ASSESSMENT_FILE = Path("data/literature/pdf_assessment.json")
PDF_DIR = Path("data/literature/pdfs")
PROGRESS_FILE = Path("data/literature/pdf_downloads.json")

# Success tracking
DOWNLOAD_LOG = {
    "timestamp": "",
    "total_attempted": 0,
    "successful": 0,
    "failed": 0,
    "failed_pmids": []
}


def download_pdf(url: str, output_path: Path, timeout: int = 30) -> bool:
    """
    Download PDF from URL with retry logic
    
    Args:
        url: PDF URL
        output_path: Save path
        timeout: Request timeout in seconds
        
    Returns:
        True if successful, False otherwise
    """
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
    }
    
    try:
        response = requests.get(url, headers=headers, timeout=timeout, stream=True)
        response.raise_for_status()
        
        # Check if response is actually a PDF
        content_type = response.headers.get('Content-Type', '')
        if 'pdf' not in content_type.lower() and 'application/octet-stream' not in content_type.lower():
            # Try anyway - some servers don't set content-type correctly
            pass
        
        # Download with progress
        with open(output_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        # Verify file size
        if output_path.stat().st_size < 1000:  # Less than 1KB is suspicious
            print(f"   [WARN] Very small file ({output_path.stat().st_size} bytes), might not be valid PDF")
            return False
        
        return True
        
    except requests.exceptions.RequestException as e:
        print(f"   [ERROR] Download failed: {e}")
        return False
    except Exception as e:
        print(f"   [ERROR] Unexpected error: {e}")
        return False


def download_all_open_access_pdfs(max_downloads: int = None, delay: float = 1.0):
    """
    Download all open-access PDFs
    
    Args:
        max_downloads: Maximum number of PDFs to download (None for all)
        delay: Delay between downloads in seconds
    """
    from datetime import datetime
    
    # Load assessment data
    print(f"\n{'='*70}")
    print("[*] Loading PDF Assessment Data")
    print('='*70)
    
    with open(ASSESSMENT_FILE, 'r', encoding='utf-8') as f:
        assessment = json.load(f)
    
    oa_papers = assessment['open_access_papers']
    total_oa = len(oa_papers)
    
    print(f"\n[INFO] Total Open-Access Papers: {total_oa}")
    print(f"[INFO] Total Papers Assessed: {assessment['total_papers']}")
    print(f"[INFO] Open-Access Percentage: {assessment['percentage_oa']:.1f}%")
    
    if max_downloads:
        oa_papers = oa_papers[:max_downloads]
        print(f"\n[LIMIT] Downloading only first {max_downloads} papers")
    
    # Create PDF directory
    PDF_DIR.mkdir(parents=True, exist_ok=True)
    print(f"\n[DIR] PDF directory: {PDF_DIR}")
    
    # Download PDFs
    print(f"\n{'='*70}")
    print("[*] Starting PDF Downloads")
    print('='*70)
    
    successful = 0
    failed = 0
    failed_pmids = []
    
    for i, paper in enumerate(tqdm(oa_papers, desc="Downloading PDFs"), 1):
        pmid = paper['pmid']
        title = paper['title'][:60] + "..." if len(paper['title']) > 60 else paper['title']
        pdf_url = paper.get('pdf_url', '')
        
        if not pdf_url:
            print(f"\n[{i}/{len(oa_papers)}] SKIP: {pmid} - No PDF URL")
            failed += 1
            failed_pmids.append(pmid)
            continue
        
        # PDF filename
        pdf_filename = f"{pmid}.pdf"
        pdf_path = PDF_DIR / pdf_filename
        
        # Skip if already downloaded
        if pdf_path.exists() and pdf_path.stat().st_size > 1000:
            print(f"\n[{i}/{len(oa_papers)}] EXISTS: {pmid}")
            successful += 1
            continue
        
        print(f"\n[{i}/{len(oa_papers)}] Downloading: {pmid}")
        print(f"   Title: {title}")
        print(f"   URL: {pdf_url}")
        
        # Download
        if download_pdf(pdf_url, pdf_path):
            print(f"   [OK] Downloaded successfully ({pdf_path.stat().st_size / 1024:.1f} KB)")
            successful += 1
            
            # Update paper metadata
            paper['downloaded'] = True
            paper['pdf_path'] = str(pdf_path)
        else:
            print(f"   [FAIL] Download failed")
            failed += 1
            failed_pmids.append(pmid)
            
            # Remove partial file
            if pdf_path.exists():
                pdf_path.unlink()
        
        # Rate limiting
        if i < len(oa_papers):  # Don't sleep after last download
            time.sleep(delay)
    
    # Final statistics
    print(f"\n{'='*70}")
    print("[*] Download Complete!")
    print('='*70)
    print(f"\n[STATS] Summary:")
    print(f"   Total attempted: {len(oa_papers)}")
    print(f"   Successful: {successful} ({successful/len(oa_papers)*100:.1f}%)")
    print(f"   Failed: {failed} ({failed/len(oa_papers)*100:.1f}%)")
    
    if failed_pmids:
        print(f"\n[FAIL] Failed PMIDs ({len(failed_pmids)}):")
        for pmid in failed_pmids[:10]:  # Show first 10
            print(f"   - {pmid}")
        if len(failed_pmids) > 10:
            print(f"   ... and {len(failed_pmids) - 10} more")
    
    # Save progress
    DOWNLOAD_LOG['timestamp'] = datetime.now().isoformat()
    DOWNLOAD_LOG['total_attempted'] = len(oa_papers)
    DOWNLOAD_LOG['successful'] = successful
    DOWNLOAD_LOG['failed'] = failed
    DOWNLOAD_LOG['failed_pmids'] = failed_pmids
    
    with open(PROGRESS_FILE, 'w', encoding='utf-8') as f:
        json.dump(DOWNLOAD_LOG, f, indent=2, ensure_ascii=False)
    
    # Update assessment file with download status
    with open(ASSESSMENT_FILE, 'w', encoding='utf-8') as f:
        json.dump(assessment, f, indent=2, ensure_ascii=False)
    
    print(f"\n[SAVE] Progress saved to: {PROGRESS_FILE}")
    print(f"[SAVE] Assessment updated: {ASSESSMENT_FILE}")
    
    return successful, failed


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Download Open-Access PDFs")
    parser.add_argument(
        '--max', 
        type=int, 
        default=None, 
        help='Maximum number of PDFs to download (default: all)'
    )
    parser.add_argument(
        '--delay', 
        type=float, 
        default=1.0, 
        help='Delay between downloads in seconds (default: 1.0)'
    )
    
    args = parser.parse_args()
    
    success, fail = download_all_open_access_pdfs(
        max_downloads=args.max,
        delay=args.delay
    )
    
    print(f"\n[DONE] Downloaded {success} PDFs successfully!")
    if fail > 0:
        print(f"[WARN] {fail} downloads failed - check logs for details")
