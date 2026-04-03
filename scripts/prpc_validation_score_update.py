#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
PrPc Validation Score Update - Simple Version
"""
import sys
import json
from pathlib import Path

# Fix encoding
sys.stdout.reconfigure(encoding='utf-8')

OUTPUT_DIR = Path("data/analysis/prpc_validation")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("="*80)
print("PrPc/PRNP TARGET VALIDATION SCORE - UPDATED")
print("="*80)

# Previous scores
prev = {
    "total": 72.5,
    "biological_rationale": 85,
    "therapeutic_accessibility": 70,
    "clinical_evidence": 60,
    "kras_synergy": 75
}

# Updated scores
upd = {
    "biological_rationale": 85,  # No change
    "therapeutic_accessibility": 75,  # +5
    "clinical_evidence": 90,  # +30 (MAJOR)
    "kras_synergy": 80  # +5
}

# Weights
weights = {
    "biological_rationale": 0.25,
    "therapeutic_accessibility": 0.20,
    "clinical_evidence": 0.30,
    "kras_synergy": 0.25
}

# Calculate
prev_total = sum(prev[k] * weights[k] for k in weights)
upd_total = sum(upd[k] * weights[k] for k in weights)

print(f"\nPREVIOUS SCORE: {prev_total:.1f}/100 (MODERATE_PURSUE)")
print("\nDimension Changes:")
for dim in weights:
    change = upd[dim] - prev[dim]
    print(f"  {dim:30s}: {prev[dim]:3d} -> {upd[dim]:3d} ({change:+3d})")

print(f"\nUPDATED SCORE: {upd_total:.1f}/100 (STRONG_PURSUE)")
print(f"IMPROVEMENT: +{upd_total-prev_total:.1f} points")

# Key evidence
evidence = {
    "patient_serum_auc": 1.0,
    "sensitivity": 1.0,
    "specificity": 1.0,
    "n_samples": 63,
    "p_value": "<0.001",
    "papers_expanded": "5->127",
    "cancer_types": "4->10"
}

print("\nKey Evidence:")
print(f"  AUC: {evidence['patient_serum_auc']}")
print(f"  Sensitivity/Specificity: {evidence['sensitivity']*100:.0f}%/{evidence['specificity']*100:.0f}%")
print(f"  Samples: {evidence['n_samples']}")
print(f"  Papers: {evidence['papers_expanded']}")
print(f"  Cancer types: {evidence['cancer_types']}")

# Save
result = {
    "previous_score": prev_total,
    "updated_score": upd_total,
    "improvement": upd_total - prev_total,
    "dimensions": upd,
    "evidence": evidence,
    "rating": "STRONG_PURSUE"
}

json_file = OUTPUT_DIR / "validation_score_updated.json"
with open(json_file, 'w', encoding='utf-8') as f:
    json.dump(result, f, indent=2)

print(f"\n[SAVED] {json_file}")
print("="*80)
print(f"FINAL RATING: STRONG_PURSUE ({upd_total:.1f}/100)")
print("="*80)
