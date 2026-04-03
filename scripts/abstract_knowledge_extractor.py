"""
Abstract-Based Knowledge Extraction with GPT-4
==============================================
Extract structured cancer knowledge from paper abstracts using GPT-4.

Features:
- Abstract-only processing (no PDF needed)
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
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, asdict
import logging

from openai import OpenAI

# Configuration
OUTPUT_DIR = Path("data/extracted/abstracts")
LOG_FILE = OUTPUT_DIR / "extraction.log"
PROGRESS_FILE = OUTPUT_DIR / "progress.json"

# Ensure output directory exists
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE, encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))



ABSTRACT_EXTRACTION_PROMPT = """Extract structured cancer knowledge from this abstract. RESPOND ONLY WITH A VALID JSON OBJECT.

Return JSON with this structure:
{{
  "mechanisms": [
    {{
      "pathway_name": "e.g., PI3K/AKT",
      "category": "Growth Signaling|Apoptosis|DNA Repair|Metabolism|Cell Cycle|Immune|Metastasis",
      "description": "brief description",
      "key_proteins": ["protein names"],
      "evidence_level": "in vitro|preclinical|clinical"
    }}
  ],
  "drugs": [
    {{
      "drug_name": "name",
      "drug_class": "Chemotherapy|Targeted|Immunotherapy|Hormone",
      "mechanism_of_action": "description",
      "molecular_target": "target"
    }}
  ],
  "drug_combinations": [
    {{
      "combination": ["Drug A", "Drug B"],
      "synergy_type": "additive|synergistic|antagonistic",
      "evidence": "quote from abstract"
    }}
  ],
  "biomarkers": [
    {{
      "name": "biomarker name",
      "type": "Genetic|Protein|Metabolic|Immune",
      "predictive_value": "what it predicts"
    }}
  ],
  "clinical_findings": {{
    "study_type": "Phase 1/2/3|Preclinical|Retrospective",
    "patient_count": null,
    "key_result": "main finding"
  }},
  "paper_summary": {{
    "cancer_type": "Colorectal|Gastric|Lung|Breast|Pancreatic|Pan-cancer",
    "novelty": "what's new (1 sentence)"
  }}
}}

RULES:
1. Extract ONLY explicitly stated information
2. Leave arrays empty [] if not mentioned
3. Use null for missing numbers
4. NO additional text, ONLY valid JSON

PAPER:
Title: {title}
Journal: {journal}
Year: {year}

ABSTRACT:
{abstract}

"""


@dataclass
class ExtractionResult:
    """Result of knowledge extraction"""
    pmid: str
    success: bool
    quality_score: float
    output_path: Optional[str] = None
    error: Optional[str] = None
    retry_count: int = 0
    processing_time: float = 0.0


class QualityValidator:
    """Validate extracted knowledge quality"""
    
    def __init__(self):
        self.min_quality = 0.5  # Lower threshold for abstracts
    
    def validate(self, extracted_data: Dict) -> Tuple[bool, float, List[str]]:
        """
        Validate extraction quality
        
        Returns:
            (is_valid, quality_score, issues)
        """
        issues = []
        scores = []
        
        # Check for valid JSON structure
        if not isinstance(extracted_data, dict):
            issues.append("Invalid JSON structure")
            return False, 0.0, issues
        
        # Check mechanisms
        mechanisms = extracted_data.get('mechanisms', [])
        if mechanisms:
            complete = sum(
                1 for m in mechanisms
                if m.get('pathway_name') and m.get('description')
            )
            mech_score = complete / len(mechanisms)
        else:
            mech_score = 0.3  # Abstracts might not mention mechanisms
        scores.append(mech_score)
        
        # Check drugs
        drugs = extracted_data.get('drugs', [])
        if drugs:
            complete = sum(
                1 for d in drugs
                if d.get('drug_name')
            )
            drug_score = complete / len(drugs)
        else:
            drug_score = 0.3  # Abstracts might not mention specific drugs
        scores.append(drug_score)
        
        # Check clinical findings
        clinical = extracted_data.get('clinical_findings', {})
        if clinical and clinical.get('key_result'):
            scores.append(1.0)
        else:
            scores.append(0.5)
        
        # Check paper metadata
        metadata = extracted_data.get('paper_metadata', {})
        if metadata.get('cancer_type') and metadata.get('key_conclusion'):
            scores.append(1.0)
        else:
            scores.append(0.5)
        
        # Calculate overall score
        quality_score = sum(scores) / len(scores) if scores else 0.0
        is_valid = quality_score >= self.min_quality
        
        if not is_valid:
            issues.append(f"Quality score {quality_score:.2f} below threshold {self.min_quality}")
        
        return is_valid, quality_score, issues


class AbstractKnowledgeExtractor:
    """Extract knowledge from paper abstracts"""
    
    def __init__(
        self,
        output_dir: Path = OUTPUT_DIR,
        max_workers: int = 4,
        max_retries: int = 3
    ):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.validator = QualityValidator()
        self.max_workers = max_workers
        self.max_retries = max_retries
        
        self.results: List[ExtractionResult] = []
        self.progress = self.load_progress()
    
    def load_progress(self) -> Dict:
        """Load previous progress"""
        if PROGRESS_FILE.exists():
            with open(PROGRESS_FILE, encoding='utf-8') as f:
                return json.load(f)
        return {'completed': [], 'failed': [], 'pending': []}
    
    def save_progress(self):
        """Save current progress"""
        with open(PROGRESS_FILE, 'w', encoding='utf-8') as f:
            json.dump(self.progress, f, indent=2, ensure_ascii=False)
    
    def extract_from_abstract(self, paper: Dict) -> Dict:
        """
        Extract knowledge from abstract using GPT-4
        
        Args:
            paper: Paper metadata including abstract
            
        Returns:
            Extracted knowledge structure
        """
        prompt = ABSTRACT_EXTRACTION_PROMPT.format(
            title=paper.get('title', 'Unknown'),
            journal=paper.get('journal', 'Unknown'),
            year=paper.get('publication_year', 'Unknown'),
            abstract=paper.get('abstract', '')
        )
        
        response = client.chat.completions.create(
            model="gpt-4-turbo-preview",
            messages=[
                {"role": "system", "content": "You are an expert oncology knowledge extractor. Return only valid JSON."},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"},
            temperature=0.1,
            max_tokens=2000
        )
        
        # Get response content
        content = response.choices[0].message.content
        
        # Debug: Log the raw response
        logger.debug(f"[{paper.get('pmid', 'unknown')}] Raw GPT response: {content[:200]}...")
        
        # Remove markdown code blocks if present
        if content.startswith('```'):
            # Remove ```json or ``` at start
            content = content.split('\n', 1)[1] if '\n' in content else content[3:]
            # Remove ``` at end
            if content.endswith('```'):
                content = content.rsplit('\n', 1)[0] if '\n' in content else content[:-3]
            content = content.strip()
        
        try:
            extracted = json.loads(content)
        except json.JSONDecodeError as e:
            logger.error(f"[{paper.get('pmid')}] JSON decode error: {e}")
            logger.error(f"[{paper.get('pmid')}] Problematic content: {content[:500]}")
            raise

        
        # Add source information
        extracted['source'] = {
            'pmid': paper.get('pmid'),
            'doi': paper.get('doi'),
            'title': paper.get('title'),
            'journal': paper.get('journal'),
            'year': paper.get('publication_year'),
            'tier': paper.get('tier'),
            'extraction_method': 'abstract_only',
            'extraction_date': time.strftime('%Y-%m-%d %H:%M:%S')
        }
        
        return extracted
    
    def extract_with_retry(self, paper: Dict) -> ExtractionResult:
        """
        Extract with automatic retry
        
        Args:
            paper: Paper metadata with abstract
            
        Returns:
            ExtractionResult with status
        """
        pmid = paper.get('pmid', 'unknown')
        title = paper.get('title', 'Unknown')[:50]
        start_time = time.time()
        
        for attempt in range(self.max_retries):
            try:
                logger.info(f"[{pmid}] Processing (attempt {attempt + 1}/{self.max_retries}): {title}...")
                
                # Extract knowledge
                extracted = self.extract_from_abstract(paper)
                
                # Validate quality
                is_valid, quality_score, issues = self.validator.validate(extracted)
                
                processing_time = time.time() - start_time
                
                if is_valid:
                    # Save successful extraction
                    output_path = self.output_dir / f"{pmid}_extracted.json"
                    
                    # Add quality info
                    extracted['quality'] = {
                        'score': quality_score,
                        'processing_time': processing_time,
                        'retry_count': attempt
                    }
                    
                    with open(output_path, 'w', encoding='utf-8') as f:
                        json.dump(extracted, f, indent=2, ensure_ascii=False)
                    
                    logger.info(f"[OK] {pmid}: Quality {quality_score:.2f}")
                    
                    return ExtractionResult(
                        pmid=pmid,
                        success=True,
                        quality_score=quality_score,
                        output_path=str(output_path),
                        retry_count=attempt,
                        processing_time=processing_time
                    )
                else:
                    logger.warning(f"[LOW QUALITY] {pmid}: {issues}")
                    if attempt < self.max_retries - 1:
                        time.sleep(2 ** attempt)
                    
            except Exception as e:
                logger.error(f"[ERROR] {pmid}: {e}")
                if attempt == self.max_retries - 1:
                    return ExtractionResult(
                        pmid=pmid,
                        success=False,
                        quality_score=0.0,
                        error=str(e),
                        retry_count=attempt,
                        processing_time=time.time() - start_time
                    )
                time.sleep(2 ** attempt)
        
        # All retries failed
        return ExtractionResult(
            pmid=pmid,
            success=False,
            quality_score=0.0,
            error="Max retries exceeded",
            retry_count=self.max_retries,
            processing_time=time.time() - start_time
        )
    
    def process_batch(
        self,
        papers: List[Dict],
        resume: bool = True
    ) -> List[ExtractionResult]:
        """
        Process all papers using parallel workers
        
        Args:
            papers: List of paper metadata dicts
            resume: Resume from previous progress
            
        Returns:
            List of extraction results
        """
        if resume:
            # Skip already completed
            completed = set(self.progress.get('completed', []))
            papers = [p for p in papers if p.get('pmid') not in completed]
        
        total = len(papers)
        if total == 0:
            logger.info("No papers to process")
            return []
        
        logger.info(f"Starting batch processing: {total} papers with {self.max_workers} workers")
        logger.info("="*70)
        
        # Process in parallel
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all jobs
            futures = {
                executor.submit(self.extract_with_retry, paper): paper
                for paper in papers
            }
            
            # Collect results as they complete
            completed_count = 0
            for future in as_completed(futures):
                result = future.result()
                self.results.append(result)
                
                completed_count += 1
                paper = futures[future]
                
                # Update progress
                if result.success:
                    self.progress['completed'].append(result.pmid)
                    logger.info(f"Progress: {completed_count}/{total} [SUCCESS] {result.pmid}")
                else:
                    self.progress['failed'].append(result.pmid)
                    logger.error(f"Progress: {completed_count}/{total} [FAILED] {result.pmid}: {result.error}")
                
                # Save progress every 10 papers
                if completed_count % 10 == 0:
                    self.save_progress()
                    self.print_progress_stats()
                
                # Rate limiting
                time.sleep(0.5)
        
        # Final save
        self.save_progress()
        self.generate_final_report()
        
        return self.results
    
    def print_progress_stats(self):
        """Print progress statistics"""
        completed = len([r for r in self.results if r.success])
        failed = len([r for r in self.results if not r.success])
        total = len(self.results)
        
        if total == 0:
            return
        
        avg_quality = sum(r.quality_score for r in self.results if r.success) / max(completed, 1)
        
        logger.info("")
        logger.info("="*70)
        logger.info(f"STATS: {completed} success / {failed} failed / {total} total")
        logger.info(f"SUCCESS RATE: {completed/total*100:.1f}%")
        logger.info(f"AVG QUALITY: {avg_quality:.3f}")
        logger.info("="*70)
        logger.info("")
    
    def generate_final_report(self):
        """Generate final detailed report"""
        report_file = self.output_dir / "extraction_report.md"
        
        successful = [r for r in self.results if r.success]
        failed = [r for r in self.results if not r.success]
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("# Abstract Knowledge Extraction Report\n\n")
            f.write(f"**Generated**: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("## Summary\n\n")
            f.write(f"- **Total Papers**: {len(self.results)}\n")
            f.write(f"- **Successful**: {len(successful)} ({len(successful)/len(self.results)*100:.1f}%)\n")
            f.write(f"- **Failed**: {len(failed)} ({len(failed)/len(self.results)*100:.1f}%)\n\n")
            
            if successful:
                avg_quality = sum(r.quality_score for r in successful) / len(successful)
                avg_time = sum(r.processing_time for r in successful) / len(successful)
                total_time = sum(r.processing_time for r in self.results)
                
                f.write(f"- **Average Quality Score**: {avg_quality:.3f}\n")
                f.write(f"- **Average Processing Time**: {avg_time:.1f}s\n")
                f.write(f"- **Total Processing Time**: {total_time/60:.1f} minutes\n\n")
            
            f.write("## Quality Distribution\n\n")
            f.write("| Range | Count | Percentage |\n")
            f.write("|-------|-------|------------|\n")
            
            ranges = [(0.9, 1.0), (0.8, 0.9), (0.7, 0.8), (0.6, 0.7), (0.5, 0.6), (0.0, 0.5)]
            for low, high in ranges:
                count = sum(1 for r in successful if low <= r.quality_score < high)
                pct = count / len(successful) * 100 if successful else 0
                f.write(f"| {low:.1f}-{high:.1f} | {count} | {pct:.1f}% |\n")
            
            if failed:
                f.write("\n## Failed Extractions\n\n")
                f.write("| PMID | Error | Retries |\n")
                f.write("|------|-------|----------|\n")
                for r in failed:
                    f.write(f"| {r.pmid} | {r.error} | {r.retry_count} |\n")
            
            f.write(f"\n## Output Files\n\n")
            f.write(f"- Extracted knowledge: `{self.output_dir}/`\n")
            f.write(f"- Progress tracking: `{PROGRESS_FILE}`\n")
            f.write(f"- Log file: `{LOG_FILE}`\n")
        
        logger.info(f"\n{'='*70}")
        logger.info(f"[REPORT] Final report saved to: {report_file}")
        logger.info(f"{'='*70}")


def main():
    parser = argparse.ArgumentParser(
        description="Abstract-Based Knowledge Extraction with GPT-4"
    )
    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='Input JSON file with paper metadata (including abstracts)'
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
    parser.add_argument(
        '--limit',
        type=int,
        help='Limit number of papers to process (for testing)'
    )
    
    args = parser.parse_args()
    
    input_file = Path(args.input)
    output_dir = Path(args.output)
    
    if not input_file.exists():
        logger.error(f"Input file not found: {input_file}")
        return
    
    # Load papers
    logger.info(f"Loading papers from: {input_file}")
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
        # Handle different JSON structures
        if 'papers' in data:
            papers = data['papers']
        elif isinstance(data, list):
            papers = data
        else:
            logger.error("Unrecognized JSON structure")
            return
    
    # Filter papers with abstracts
    papers_with_abstracts = [
        p for p in papers 
        if p.get('abstract') and len(p.get('abstract', '')) > 100
    ]
    
    logger.info(f"Found {len(papers)} total papers")
    logger.info(f"Papers with abstracts (>100 chars): {len(papers_with_abstracts)}")
    
    if args.limit:
        papers_with_abstracts = papers_with_abstracts[:args.limit]
        logger.info(f"Limited to {args.limit} papers for testing")
    
    # Create extractor
    extractor = AbstractKnowledgeExtractor(
        output_dir=output_dir,
        max_workers=args.workers,
        max_retries=args.max_retries
    )
    
    logger.info("\n" + "="*70)
    logger.info("STARTING ABSTRACT EXTRACTION")
    logger.info("="*70 + "\n")
    
    # Process batch
    results = extractor.process_batch(papers_with_abstracts, resume=args.resume)
    
    # Final statistics
    successful = sum(1 for r in results if r.success)
    logger.info(f"\n{'='*70}")
    logger.info(f"[COMPLETE] Extraction finished!")
    logger.info(f"[COMPLETE] Successful: {successful}/{len(results)} ({successful/len(results)*100:.1f}%)")
    logger.info(f"[COMPLETE] Output: {output_dir}")
    logger.info(f"{'='*70}\n")


if __name__ == "__main__":
    main()
