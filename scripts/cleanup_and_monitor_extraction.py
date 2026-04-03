"""
Literature Extraction Cleanup & Real-time Monitor
==================================================
Clean up duplicates and provide live monitoring for extraction progress.

Features:
- Remove duplicate entries from progress.json
- Validate all extracted JSON files
- Generate clean final report
- Real-time progress monitoring with visual indicators
- Identify remaining papers to process

Author: ADDS Team
Date: 2026-01-31
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Set, Tuple
from datetime import datetime

# Paths
DATA_DIR = Path("data")
LITERATURE_DIR = DATA_DIR / "literature"
EXTRACTED_DIR = DATA_DIR / "extracted" / "abstracts"

PROGRESS_FILE = EXTRACTED_DIR / "progress.json"
METADATA_FILE = LITERATURE_DIR / "comprehensive_metadata.json"
CLEANUP_REPORT = EXTRACTED_DIR / "cleanup_report.md"


def load_json(filepath: Path) -> Dict:
    """Load JSON file safely"""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_json(filepath: Path, data: Dict):
    """Save JSON file safely"""
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def cleanup_duplicates(progress: Dict) -> Dict:
    """Remove duplicate entries from progress tracking"""
    print("\n[CLEANUP] Cleaning up duplicates...")
    
    original_completed = len(progress.get('completed', []))
    original_failed = len(progress.get('failed', []))
    
    # Remove duplicates while preserving order
    progress['completed'] = list(dict.fromkeys(progress.get('completed', [])))
    progress['failed'] = list(dict.fromkeys(progress.get('failed', [])))
    progress['pending'] = list(dict.fromkeys(progress.get('pending', [])))
    
    # Remove items that appear in both completed and failed (completed takes precedence)
    completed_set = set(progress['completed'])
    progress['failed'] = [pmid for pmid in progress['failed'] if pmid not in completed_set]
    
    new_completed = len(progress['completed'])
    new_failed = len(progress['failed'])
    
    print(f"   Completed: {original_completed} -> {new_completed} (removed {original_completed - new_completed} duplicates)")
    print(f"   Failed: {original_failed} -> {new_failed} (removed {original_failed - new_failed} duplicates)")
    
    return progress


def validate_extracted_files(extracted_dir: Path) -> Tuple[List[str], List[str]]:
    """Validate all extracted JSON files"""
    print("\n[VALIDATE] Validating extracted files...")
    
    valid_files = []
    invalid_files = []
    
    for json_file in extracted_dir.glob("*_extracted.json"):
        try:
            data = load_json(json_file)
            
            # Basic validation
            if 'source' in data and 'pmid' in data['source']:
                valid_files.append(data['source']['pmid'])
            else:
                invalid_files.append(json_file.name)
                
        except Exception as e:
            invalid_files.append(f"{json_file.name} (error: {e})")
    
    print(f"   [OK] Valid: {len(valid_files)}")
    print(f"   [WARN] Invalid: {len(invalid_files)}")
    
    return valid_files, invalid_files


def analyze_coverage(metadata: Dict, extracted_pmids: Set[str]) -> Dict:
    """Analyze extraction coverage"""
    print("\n[ANALYZE] Analyzing coverage...")
    
    all_papers = metadata.get('papers', [])
    total_papers = len(all_papers)
    
    papers_with_abstracts = [
        p for p in all_papers
        if p.get('abstract') and len(p.get('abstract', '')) > 100
    ]
    
    available_count = len(papers_with_abstracts)
    extracted_count = len(extracted_pmids)
    
    # Find missing papers
    available_pmids = {p['pmid'] for p in papers_with_abstracts}
    missing_pmids = available_pmids - extracted_pmids
    
    coverage = {
        'total_papers': total_papers,
        'papers_with_abstracts': available_count,
        'extracted': extracted_count,
        'missing': len(missing_pmids),
        'coverage_rate': (extracted_count / available_count * 100) if available_count > 0 else 0,
        'missing_pmids': list(missing_pmids)
    }
    
    print(f"   Total papers: {total_papers}")
    print(f"   Papers with abstracts: {available_count}")
    print(f"   Successfully extracted: {extracted_count}")
    print(f"   Missing: {len(missing_pmids)}")
    print(f"   Coverage: {coverage['coverage_rate']:.1f}%")
    
    return coverage


def generate_cleanup_report(
    coverage: Dict,
    invalid_files: List[str],
    progress_before: Dict,
    progress_after: Dict
) -> str:
    """Generate comprehensive cleanup report"""
    
    report = f"""# Literature Extraction Cleanup Report

**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

---

## Cleanup Summary

### Progress Tracking
- **Completed entries**: {len(progress_before.get('completed', []))} -> {len(progress_after.get('completed', []))}
- **Failed entries**: {len(progress_before.get('failed', []))} -> {len(progress_after.get('failed', []))}
- **Duplicates removed**: {len(progress_before.get('completed', [])) - len(progress_after.get('completed', [])) + len(progress_before.get('failed', [])) - len(progress_after.get('failed', []))}

---

## Extraction Coverage

- **Total papers in database**: {coverage['total_papers']}
- **Papers with valid abstracts**: {coverage['papers_with_abstracts']}
- **Successfully extracted**: {coverage['extracted']}
- **Missing extractions**: {coverage['missing']}
- **Coverage rate**: {coverage['coverage_rate']:.1f}%

---

## Status

"""
    
    if coverage['coverage_rate'] >= 95:
        report += "**STATUS**: Excellent coverage! Extraction nearly complete.\n\n"
    elif coverage['coverage_rate'] >= 90:
        report += "**STATUS**: Good coverage, but some papers remaining.\n\n"
    else:
        report += "**STATUS**: Significant papers remaining to extract.\n\n"
    
    if invalid_files:
        report += "## Invalid Files\n\n"
        for f in invalid_files:
            report += f"- `{f}`\n"
        report += "\n"
    
    if coverage['missing'] > 0 and coverage['missing'] <= 10:
        report += "## Missing Papers\n\n"
        for pmid in coverage['missing_pmids'][:10]:
            report += f"- PMID: {pmid}\n"
        report += "\n"
    
    report += f"""---

## Output Locations

- **Extracted knowledge**: `{EXTRACTED_DIR}/`
- **Progress tracking**: `{PROGRESS_FILE}`
- **Cleanup report**: `{CLEANUP_REPORT}`

---

## Next Steps

"""
    
    if coverage['missing'] > 0:
        report += f"""1. **Continue extraction** for {coverage['missing']} remaining papers
2. Validate extraction quality
3. Integrate into ADDS knowledge base
"""
    else:
        report += """1. **Extraction complete!**
2. Validate extraction quality
3. Integrate into ADDS knowledge base
4. Proceed to Phase 5
"""
    
    return report


def main():
    """Main cleanup process"""
    print("="*70)
    print(" LITERATURE EXTRACTION CLEANUP & VALIDATION")
    print("="*70)
    
    # Check if files exist
    if not PROGRESS_FILE.exists():
        print(f"[ERROR] Progress file not found: {PROGRESS_FILE}")
        return
    
    if not METADATA_FILE.exists():
        print(f"[ERROR] Metadata file not found: {METADATA_FILE}")
        return
    
    # Load data
    print("\n[LOAD] Loading data...")
    progress_before = load_json(PROGRESS_FILE)
    metadata = load_json(METADATA_FILE)
    
    # Step 1: Clean up duplicates
    progress_after = cleanup_duplicates(progress_before)
    
    # Step 2: Validate extracted files
    valid_pmids, invalid_files = validate_extracted_files(EXTRACTED_DIR)
    
    # Update progress with validated data
    progress_after['completed'] = valid_pmids
    progress_after['failed'] = [
        pmid for pmid in progress_after['failed'] 
        if pmid not in valid_pmids
    ]
    
    # Step 3: Analyze coverage
    coverage = analyze_coverage(metadata, set(valid_pmids))
    
    # Step 4: Generate report
    print("\n[REPORT] Generating cleanup report...")
    report = generate_cleanup_report(coverage, invalid_files, progress_before, progress_after)
    
    # Save cleaned progress
    print("\n[SAVE] Saving cleaned data...")
    save_json(PROGRESS_FILE, progress_after)
    
    # Save report
    with open(CLEANUP_REPORT, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"   [OK] Progress file updated: {PROGRESS_FILE}")
    print(f"   [OK] Cleanup report saved: {CLEANUP_REPORT}")
    
    # Summary
    print("\n" + "="*70)
    print(" CLEANUP COMPLETE!")
    print("="*70)
    print(f"\nFinal Statistics:")
    print(f"   Successfully extracted: {coverage['extracted']}/{coverage['papers_with_abstracts']}")
    print(f"   Coverage: {coverage['coverage_rate']:.1f}%")
    print(f"   Missing papers: {coverage['missing']}")
    
    if coverage['missing'] > 0:
        print(f"\n>> Next: Run extraction for {coverage['missing']} remaining papers")
        print(f"   Command: python scripts/abstract_knowledge_extractor.py --input data/literature/comprehensive_metadata.json --resume")
    else:
        print("\n[SUCCESS] All papers extracted! Ready for Phase 5.")
    
    print()


if __name__ == "__main__":
    main()
