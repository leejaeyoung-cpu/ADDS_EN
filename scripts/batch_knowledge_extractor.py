"""
Batch Knowledge Extraction with GPT-4
======================================
Parallel processing system for extracting structured knowledge from 200+ papers.

Features:
- Parallel processing (4 workers)
- Automatic retry with exponential backoff
- Quality validation
- Progress tracking
- Error recovery

Author: ADDS Team
Date: 2026-01-31
"""

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import dataclass, asdict
import logging
import sys

sys.path.append(str(Path(__file__).parent.parent))

from scripts.knowledge_extractor import KnowledgeExtractor

# Configuration
OUTPUT_DIR = Path("data/extracted")
LOG_FILE = OUTPUT_DIR / "extraction.log"
PROGRESS_FILE = OUTPUT_DIR / "progress.json"

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


@dataclass
class ExtractionResult:
    """Result of knowledge extraction"""
    pdf_path: str
    success: bool
    quality_score: float
    output_path: Optional[str] = None
    error: Optional[str] = None
    retry_count: int = 0
    processing_time: float = 0.0


class QualityValidator:
    """Validate extracted knowledge quality"""
    
    def __init__(self):
        self.min_mechanisms = 1
        self.min_drugs = 1
        self.required_fields = ['mechanisms', 'drugs']
    
    def validate(self, extracted_data: Dict) -> Tuple[bool, float, List[str]]:
        """
        Validate extraction quality
        
        Returns:
            (is_valid, quality_score, issues)
        """
        issues = []
        scores = []
        
        # Check required top-level fields
        for field in self.required_fields:
            if field not in extracted_data:
                issues.append(f"Missing required field: {field}")
                scores.append(0.0)
            else:
                scores.append(1.0)
        
        # Check mechanisms
        mechanisms = extracted_data.get('mechanisms', [])
        if len(mechanisms) < self.min_mechanisms:
            issues.append(f"Too few mechanisms: {len(mechanisms)} < {self.min_mechanisms}")
            mech_score = 0.0
        else:
            # Score based on completeness
            complete_count = sum(
                1 for m in mechanisms 
                if m.get('pathway_name') and m.get('description')
            )
            mech_score = complete_count / len(mechanisms) if mechanisms else 0.0
        scores.append(mech_score)
        
        # Check drugs
        drugs = extracted_data.get('drugs', [])
        if len(drugs) < self.min_drugs:
            issues.append(f"Too few drugs: {len(drugs)} < {self.min_drugs}")
            drug_score = 0.0
        else:
            complete_count = sum(
                1 for d in drugs 
                if d.get('drug_name') and d.get('mechanism_of_action')
            )
            drug_score = complete_count / len(drugs) if drugs else 0.0
        scores.append(drug_score)
        
        # Check drug interactions
        interactions = extracted_data.get('drug_interactions', [])
        if interactions:
            complete_count = sum(
                1 for i in interactions
                if i.get('drug1') and i.get('drug2') and i.get('interaction_type')
            )
            int_score = complete_count / len(interactions)
        else:
            int_score = 0.5  # Neutral score if no interactions
        scores.append(int_score)
        
        # Check biomarkers
        biomarkers = extracted_data.get('biomarkers', [])
        if biomarkers:
            complete_count = sum(
                1 for b in biomarkers
                if b.get('name') and b.get('type')
            )
            bio_score = complete_count / len(biomarkers)
        else:
            bio_score = 0.5
        scores.append(bio_score)
        
        # Calculate overall score
        quality_score = sum(scores) / len(scores) if scores else 0.0
        is_valid = quality_score >= 0.6 and len(issues) == 0
        
        return is_valid, quality_score, issues


class BatchKnowledgeExtractor:
    """Batch processor for knowledge extraction"""
    
    def __init__(
        self,
        output_dir: Path = OUTPUT_DIR,
        max_workers: int = 4,
        max_retries: int = 3
    ):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.extractor = KnowledgeExtractor()
        self.validator = QualityValidator()
        self.max_workers = max_workers
        self.max_retries = max_retries
        
        self.results: List[ExtractionResult] = []
        self.progress = self.load_progress()
    
    def load_progress(self) -> Dict:
        """Load previous progress"""
        if PROGRESS_FILE.exists():
            with open(PROGRESS_FILE) as f:
                return json.load(f)
        return {'completed': [], 'failed': [], 'pending': []}
    
    def save_progress(self):
        """Save current progress"""
        with open(PROGRESS_FILE, 'w') as f:
            json.dump(self.progress, f, indent=2)
    
    def extract_with_retry(
        self, 
        pdf_path: Path
    ) -> ExtractionResult:
        """
        Extract knowledge with automatic retry
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            ExtractionResult with status and data
        """
        start_time = time.time()
        
        for attempt in range(self.max_retries):
            try:
                logger.info(f"Processing {pdf_path.name} (attempt {attempt + 1}/{self.max_retries})")
                
                # Extract knowledge
                extracted = self.extractor.process_paper(pdf_path, self.output_dir)
                
                # Validate quality
                is_valid, quality_score, issues = self.validator.validate(extracted)
                
                processing_time = time.time() - start_time
                
                if is_valid:
                    # Save successful extraction
                    output_path = self.output_dir / f"{pdf_path.stem}_extracted.json"
                    
                    # Add source information
                    extracted['source'] = {
                        'pdf_path': str(pdf_path),
                        'extraction_date': time.strftime('%Y-%m-%d %H:%M:%S'),
                        'quality_score': quality_score,
                        'processing_time': processing_time
                    }
                    
                    with open(output_path, 'w', encoding='utf-8') as f:
                        json.dump(extracted, f, indent=2, ensure_ascii=False)
                    
                    logger.info(f"✅ Success: {pdf_path.name} (quality: {quality_score:.2f})")
                    
                    return ExtractionResult(
                        pdf_path=str(pdf_path),
                        success=True,
                        quality_score=quality_score,
                        output_path=str(output_path),
                        retry_count=attempt,
                        processing_time=processing_time
                    )
                else:
                    logger.warning(f"Quality check failed for {pdf_path.name}: {issues}")
                    if attempt < self.max_retries - 1:
                        time.sleep(2 ** attempt)  # Exponential backoff
                    
            except Exception as e:
                logger.error(f"Error processing {pdf_path.name}: {e}")
                if attempt == self.max_retries - 1:
                    return ExtractionResult(
                        pdf_path=str(pdf_path),
                        success=False,
                        quality_score=0.0,
                        error=str(e),
                        retry_count=attempt,
                        processing_time=time.time() - start_time
                    )
                time.sleep(2 ** attempt)
        
        # All retries failed
        return ExtractionResult(
            pdf_path=str(pdf_path),
            success=False,
            quality_score=0.0,
            error="Max retries exceeded",
            retry_count=self.max_retries,
            processing_time=time.time() - start_time
        )
    
    def process_batch(
        self,
        pdf_dir: Path,
        resume: bool = True
    ) -> List[ExtractionResult]:
        """
        Process all PDFs in directory using parallel workers
        
        Args:
            pdf_dir: Directory containing PDF files
            resume: Resume from previous progress
            
        Returns:
            List of extraction results
        """
        pdfs = list(pdf_dir.glob("*.pdf"))
        
        if resume:
            # Skip already completed
            completed = set(self.progress.get('completed', []))
            pdfs = [p for p in pdfs if str(p) not in completed]
        
        total = len(pdfs)
        if total == 0:
            logger.info("No PDFs to process")
            return []
        
        logger.info(f"Starting batch processing: {total} PDFs with {self.max_workers} workers")
        
        # Process in parallel
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all jobs
            futures = {
                executor.submit(self.extract_with_retry, pdf): pdf 
                for pdf in pdfs
            }
            
            # Collect results as they complete
            completed_count = 0
            for future in as_completed(futures):
                result = future.result()
                self.results.append(result)
                
                completed_count += 1
                pdf = futures[future]
                
                # Update progress
                if result.success:
                    self.progress['completed'].append(str(pdf))
                    logger.info(f"Progress: {completed_count}/{total} (✅ {result.quality_score:.2f})")
                else:
                    self.progress['failed'].append(str(pdf))
                    logger.error(f"Progress: {completed_count}/{total} (❌ {result.error})")
                
                # Save progress every 10 papers
                if completed_count % 10 == 0:
                    self.save_progress()
                    self.generate_progress_report()
        
        # Final save
        self.save_progress()
        self.generate_final_report()
        
        return self.results
    
    def generate_progress_report(self):
        """Generate progress report"""
        completed = len([r for r in self.results if r.success])
        failed = len([r for r in self.results if not r.success])
        total = len(self.results)
        
        if total == 0:
            return
        
        avg_quality = sum(r.quality_score for r in self.results if r.success) / max(completed, 1)
        avg_time = sum(r.processing_time for r in self.results) / total
        
        report = f"""
        === EXTRACTION PROGRESS ===
        Total Processed: {total}
        Successful: {completed} ({completed/total*100:.1f}%)
        Failed: {failed} ({failed/total*100:.1f}%)
        Avg Quality: {avg_quality:.3f}
        Avg Time: {avg_time:.1f}s
        """
        
        logger.info(report)
    
    def generate_final_report(self):
        """Generate final detailed report"""
        report_file = self.output_dir / "extraction_report.md"
        
        successful = [r for r in self.results if r.success]
        failed = [r for r in self.results if not r.success]
        
        with open(report_file, 'w') as f:
            f.write("# Knowledge Extraction Report\n\n")
            f.write(f"**Generated**: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("## Summary\n\n")
            f.write(f"- **Total Papers**: {len(self.results)}\n")
            f.write(f"- **Successful**: {len(successful)} ({len(successful)/len(self.results)*100:.1f}%)\n")
            f.write(f"- **Failed**: {len(failed)} ({len(failed)/len(self.results)*100:.1f}%)\n\n")
            
            if successful:
                avg_quality = sum(r.quality_score for r in successful) / len(successful)
                avg_time = sum(r.processing_time for r in successful) / len(successful)
                f.write(f"- **Average Quality Score**: {avg_quality:.3f}\n")
                f.write(f"- **Average Processing Time**: {avg_time:.1f}s\n\n")
            
            f.write("## Quality Distribution\n\n")
            f.write("| Range | Count |\n")
            f.write("|-------|-------|\n")
            
            ranges = [(0.9, 1.0), (0.8, 0.9), (0.7, 0.8), (0.6, 0.7), (0.0, 0.6)]
            for low, high in ranges:
                count = sum(1 for r in successful if low <= r.quality_score < high)
                f.write(f"| {low:.1f}-{high:.1f} | {count} |\n")
            
            if failed:
                f.write("\n## Failed Extractions\n\n")
                f.write("| PDF | Error | Retries |\n")
                f.write("|-----|-------|----------|\n")
                for r in failed:
                    pdf_name = Path(r.pdf_path).name
                    f.write(f"| {pdf_name} | {r.error} | {r.retry_count} |\n")
        
        logger.info(f"📄 Final report saved to: {report_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Batch Knowledge Extraction with GPT-4"
    )
    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='Input directory containing PDFs'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=str(OUTPUT_DIR),
        help='Output directory for extracted knowledge'
    )
    parser.add_argument(
        '--workers',
        type=int,
        default=4,
        help='Number of parallel workers'
    )
    parser.add_argument(
        '--max_retries',
        type=int,
        default=3,
        help='Maximum retry attempts per paper'
    )
    parser.add_argument(
        '--resume',
        action='store_true',
        help='Resume from previous progress'
    )
    
    args = parser.parse_args()
    
    pdf_dir = Path(args.input)
    output_dir = Path(args.output)
    
    if not pdf_dir.exists():
        logger.error(f"Input directory not found: {pdf_dir}")
        return
    
    processor = BatchKnowledgeExtractor(
        output_dir=output_dir,
        max_workers=args.workers,
        max_retries=args.max_retries
    )
    
    logger.info("🚀 Starting batch extraction...")
    results = processor.process_batch(pdf_dir, resume=args.resume)
    
    successful = sum(1 for r in results if r.success)
    logger.info(f"\n🎉 Extraction complete!")
    logger.info(f"   Successful: {successful}/{len(results)}")
    logger.info(f"   Output: {output_dir}")


if __name__ == "__main__":
    main()
