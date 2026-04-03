"""
Merge Literature Metadata - Use PDF Assessment
===============================================
Use pdf_assessment.json as primary source (has all 184 papers)

Author: ADDS Team
Date: 2026-01-31
"""

import json
from pathlib import Path

# Primary source: PDF assessment has all papers
PDF_ASSESSMENT = Path("data/literature/pdf_assessment.json")
OUTPUT_FILE = Path("data/literature/unified_literature_metadata.json")

print("\n" + "="*70)
print("PREPARING LITERATURE METADATA FOR EXTRACTION")
print("="*70 + "\n")

# Load PDF assessment (has all papers + OA status)
print(f"[*] Loading: {PDF_ASSESSMENT}")
with open(PDF_ASSESSMENT, 'r', encoding='utf-8') as f:
    assessment = json.load(f)

# Get both OA and paywall papers
oa_papers = assessment.get('open_access_papers', [])
paywall_papers = assessment.get('paywall_papers', [])
all_papers = oa_papers + paywall_papers

print(f"    Open Access: {len(oa_papers)} papers")
print(f"    Paywall: {len(paywall_papers)} papers")
print(f"    Total: {len(all_papers)} papers")

# Filter papers with good abstracts
papers_with_abstracts = [
    p for p in all_papers 
    if p.get('abstract') and len(p.get('abstract', '')) > 100
]

print(f"\n[FILTER] Papers with abstracts (>100 chars): {len(papers_with_abstracts)}")

# Count by tier
tier_counts = {}
cancer_counts = {}
for paper in papers_with_abstracts:
    tier = paper.get('tier', 'unknown')
    tier_counts[tier] = tier_counts.get(tier, 0) + 1
    
    cancer = paper.get('cancer_type', 'unknown')
    cancer_counts[cancer] = cancer_counts.get(cancer, 0) + 1

print(f"\n[BY TIER]")
for tier, count in sorted(tier_counts.items()):
    print(f"  Tier {tier}: {count} papers")

print(f"\n[BY CANCER TYPE]")
for cancer, count in sorted(cancer_counts.items(), key=lambda x: -x[1])[:7]:
    print(f"  {cancer.capitalize()}: {count} papers")

# Create output structure
output = {
    "papers": papers_with_abstracts,
    "metadata": {
        "total_papers": len(papers_with_abstracts),
        "tier_counts": tier_counts,
        "cancer_counts": cancer_counts,
        "collection_date": "2026-01-31",
        "source": str(PDF_ASSESSMENT)
    }
}

# Save
print(f"\n[SAVE] Writing to: {OUTPUT_FILE}")
with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
    json.dump(output, f, indent=2, ensure_ascii=False)

print(f"[OK] Saved {len(papers_with_abstracts)} papers to {OUTPUT_FILE.name}")

# Final summary
print("\n" + "="*70)
print(f"[OK] Ready for extraction: {len(papers_with_abstracts)} papers with abstracts")
print("="*70 + "\n")
