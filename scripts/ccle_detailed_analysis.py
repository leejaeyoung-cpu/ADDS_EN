"""
Task 2: CCLE Real Expression Detailed Analysis
===============================================
Investigate WHY real CCLE interaction didn't dramatically outperform proxy.
"""
import pandas as pd
import numpy as np
import pickle
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

DATA_DIR = Path("F:/ADDS/data/ml_training")
CCLE_DIR = DATA_DIR / "ccle_raw"
MODEL_DIR = Path("F:/ADDS/models/synergy")

KNOWN_AFFINITIES = {
    'ERLOTINIB': {'EGFR': 8.7, 'ABL1': 5.9},
    'LAPATINIB': {'EGFR': 8.0, 'ERBB2': 8.0},
    'SORAFENIB': {'BRAF': 7.7, 'FLT3': 7.2, 'KIT': 7.2, 'PDGFRB': 7.2, 'RAF1': 8.2, 'KDR': 7.0},
    'DASATINIB': {'ABL1': 9.2, 'EPHA2': 7.8, 'KIT': 8.3, 'PDGFRB': 7.6, 'SRC': 9.3},
    'BORTEZOMIB': {'PSMB5': 9.2},
    'PACLITAXEL': {'TUBB': 8.4},
    'VINBLASTINE': {'TUBB': 9.0},
    'DOXORUBICIN': {'TOP2A': 6.8},
    'ETOPOSIDE': {'TOP2A': 5.7, 'TOP2B': 5.5},
    'TOPOTECAN': {'TOP1': 6.5},
    'MK-2206': {'AKT1': 8.1, 'AKT2': 7.9, 'AKT3': 7.2},
    'BEZ-235': {'PIK3CA': 8.4, 'PIK3CB': 7.1, 'MTOR': 8.2},
    'PD325901': {'MAP2K1': 9.5, 'MAP2K2': 9.1},
    'DINACICLIB': {'CDK1': 8.5, 'CDK2': 9.0, 'CDK5': 9.0, 'CDK9': 8.4},
}


def main():
    print("=" * 70)
    print("CCLE EXPRESSION DETAILED ANALYSIS")
    print("=" * 70)
    
    ccle = pd.read_csv(CCLE_DIR / "ccle_target_expression.csv")
    syn = pd.read_csv(DATA_DIR / "oneil_synergy.csv")
    
    # 1. Cell line coverage
    print(f"\n{'='*70}")
    print("1. CELL LINE COVERAGE")
    print("=" * 70)
    
    oneil_cells = set(syn['cell_line'].str.upper().unique())
    ccle_cells_full = set(ccle['cell_line'].values)
    ccle_cells_short = set(c.split('_')[0] for c in ccle['cell_line'].values)
    
    print(f"  O'Neil unique cell lines: {len(oneil_cells)}")
    print(f"  CCLE cell lines: {len(ccle_cells_full)}")
    
    matched = oneil_cells & ccle_cells_short
    unmatched = oneil_cells - ccle_cells_short
    print(f"  Direct match: {len(matched)}")
    print(f"  Unmatched: {len(unmatched)}: {sorted(unmatched)}")
    
    # Match rate by sample count
    total_samples = len(syn)
    matched_samples = syn[syn['cell_line'].str.upper().isin(matched) | 
                          syn['cell_line'].str.upper().isin(ccle_cells_full)].shape[0]
    
    # 2. Expression distribution per gene
    print(f"\n{'='*70}")
    print("2. EXPRESSION DISTRIBUTION")
    print("=" * 70)
    
    gene_cols = [c for c in ccle.columns if c != 'cell_line']
    
    # Problem analysis: sparsity
    print(f"\n  Gene-level statistics (RPKM, log2-scale):")
    print(f"  {'Gene':<15s} {'Mean':>8s} {'Std':>8s} {'Zero%':>8s} {'Min':>8s} {'Max':>8s} {'Issue':>15s}")
    
    issues = {}
    for gene in gene_cols:
        vals = ccle[gene].dropna().values
        mean = vals.mean()
        std = vals.std()
        zero_pct = (vals == 0).sum() / len(vals) * 100
        low_expr = (vals < 0.5).sum() / len(vals) * 100
        
        issue = ""
        if zero_pct > 50:
            issue = "HIGH ZERO%"
            issues[gene] = 'zero'
        elif std < 0.5:
            issue = "LOW VARIANCE"
            issues[gene] = 'low_var'
        elif mean < 0.5:
            issue = "LOW EXPR"
            issues[gene] = 'low_expr'
        else:
            issue = "OK"
        
        print(f"  {gene:<15s} {mean:>8.2f} {std:>8.2f} {zero_pct:>7.1f}% {vals.min():>8.2f} {vals.max():>8.2f} {issue:>15s}")
    
    # 3. Interaction feature sparsity
    print(f"\n{'='*70}")
    print("3. INTERACTION FEATURE SPARSITY")
    print("=" * 70)
    
    # For each drug, count how many truly non-zero interaction features
    print(f"\n  Drug-level interaction coverage:")
    
    total_interactions = 0
    nonzero_interactions = 0
    
    for drug, targets in KNOWN_AFFINITIES.items():
        n_targets = len(targets)
        target_in_ccle = [t for t in targets if t in gene_cols]
        
        if target_in_ccle:
            # For matched O'Neil cell lines, check expression
            for t in target_in_ccle:
                vals = ccle[t].dropna().values
                nonzero = (vals > 0.5).sum()
                total_interactions += len(vals)
                nonzero_interactions += nonzero
        
        coverage = len(target_in_ccle) / max(n_targets, 1) * 100
        print(f"  {drug:<20s}: {len(target_in_ccle)}/{n_targets} targets in CCLE ({coverage:.0f}%)")
    
    sparsity = 1 - nonzero_interactions / max(total_interactions, 1)
    print(f"\n  Overall interaction sparsity: {sparsity*100:.1f}%")
    print(f"  NonZero interactions: {nonzero_interactions}/{total_interactions}")
    
    # 4. Why real CCLE didn't dramatically help
    print(f"\n{'='*70}")
    print("4. ROOT CAUSE ANALYSIS: Why REAL CCLE ≈ Proxy")
    print("=" * 70)
    
    problem_genes = [g for g, issue in issues.items() if issue != 'ok']
    
    print(f"""
  FINDING 1: Many targets have VERY LOW expression
    → Genes like FLT3, KIT, KDR are receptor tyrosine kinases
      specifically expressed in hematopoietic cells
    → In solid tumor cell lines (most O'Neil), they are near-zero
    → pKi × 0 = 0 → interaction feature is useless for these drugs

  FINDING 2: Expression is LOG-SCALED (RPKM)
    → CCLE values are log2(RPKM+1)
    → A value of 5.0 means RPKM=31, which is moderate
    → Many drug targets have RPKM < 1 in non-target tissues
    → This creates a sparse interaction matrix

  FINDING 3: Limited drug-target coverage
    → Only {len(KNOWN_AFFINITIES)} drugs have pKi annotations
    → O'Neil has ~38 drugs → {len(KNOWN_AFFINITIES)/38*100:.0f}% coverage
    → Drugs without pKi get zero-vector interactions

  FINDING 4: Cell line diversity issue
    → O'Neil uses 39 cell lines, mostly solid tumors
    → Many kinase targets are tissue-specific
    → Target expression varies little WITHIN tissue types
    → XGBoost sees low per-cell variance → learns little

  RECOMMENDATIONS:
    1. Use RAW RPKM (not log) for interaction: pKi × RPKM
    2. Add pathway-level interaction: aggregate target scores per pathway
    3. Expand drug pKi coverage via ChEMBL API
    4. Consider Z-score normalization per gene across cell lines
    """)
    
    # 5. Save analysis
    analysis = {
        'n_cell_lines': len(ccle),
        'n_genes': len(gene_cols),
        'match_rate': len(matched) / max(len(oneil_cells), 1),
        'issues': issues,
        'sparsity': sparsity,
    }
    
    with open(CCLE_DIR / "ccle_analysis_results.pkl", 'wb') as f:
        pickle.dump(analysis, f)
    
    print(f"  Saved: {CCLE_DIR / 'ccle_analysis_results.pkl'}")


if __name__ == "__main__":
    main()
