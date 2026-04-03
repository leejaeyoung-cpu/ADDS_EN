"""
Task 5: Missing Cell Line Mapping
==================================
Map 10 unmatched O'Neil cell lines to CCLE names and re-extract expression.
"""
import pandas as pd
import numpy as np
import gzip
import csv
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

CCLE_DIR = Path("F:/ADDS/data/ml_training/ccle_raw")

# Manual mapping: O'Neil name → CCLE name
# Verified against DepMap cell line registry
CELL_LINE_MAP = {
    'COLO320DM': 'COLO320DM_LARGE_INTESTINE',
    'DLD1': 'DLD1_LARGE_INTESTINE',
    'EFM192B': 'EFM192B_BREAST',
    'LNCAP': 'LNCAPCLONEFGC_PROSTATE',
    'MSTO': 'MSTO211H_PLEURA',
    'OCUBM': 'OCUBM_BREAST',
    'OVCAR3': 'NIHOVCAR3_OVARY',
    'PA1': 'PA1_OVARY',
    'UWB1289': 'UWB1289_OVARY',
    'UWB1289BRCA1': 'UWB1289BRCA1_OVARY',
}

TARGET_GENES = [
    'ABL1', 'AKT1', 'AKT2', 'AKT3', 'AURKA', 'BRAF', 'CDK1', 'CDK2',
    'CDK5', 'CDK9', 'CHEK1', 'DHFR', 'EGFR', 'EPHA2', 'ERBB2', 'FLT3',
    'HDAC1', 'HDAC2', 'HDAC3', 'HSP90AA1', 'KIT', 'MAP2K1', 'MAP2K2',
    'MGMT', 'MTOR', 'NOTCH1', 'NR3C1', 'PARP1', 'PARP2', 'PDGFRA',
    'PDGFRB', 'PIK3CA', 'PIK3CB', 'PRKAA1', 'PRKAA2', 'PSMB5', 'RAF1',
    'RET', 'RRM1', 'SRC', 'TOP1', 'TOP2A', 'TOP2B', 'TUBB', 'TYMS',
    'KDR', 'TUBB1'
]


def main():
    print("=" * 70)
    print("TASK 5: Missing Cell Line Mapping")
    print("=" * 70)
    
    ccle_path = CCLE_DIR / "CCLE_expression_tpm.csv"
    target_path = CCLE_DIR / "ccle_target_expression.csv"
    
    # Read all CCLE cell lines
    all_ccle_names = set()
    with gzip.open(ccle_path, 'rt') as f:
        f.readline()  # #1.2
        f.readline()  # dims
        header = f.readline().strip().split('\t')
        all_ccle_names = set(header[2:])
    
    print(f"\nCCLE total cell lines: {len(all_ccle_names)}")
    
    # Check mappings
    print(f"\nMapping verification:")
    matched = {}
    for oneil, ccle in CELL_LINE_MAP.items():
        found = ccle in all_ccle_names
        if not found:
            # Fuzzy search
            candidates = [n for n in all_ccle_names if oneil.upper() in n.upper()]
            if candidates:
                ccle = candidates[0]
                found = True
                CELL_LINE_MAP[oneil] = ccle
                print(f"  {oneil:20s} → {ccle:35s} FUZZY MATCH ✓")
            else:
                print(f"  {oneil:20s} → {ccle:35s} NOT FOUND ✗")
        else:
            print(f"  {oneil:20s} → {ccle:35s} EXACT MATCH ✓")
        
        if found:
            matched[oneil] = ccle
    
    print(f"\nMatched: {len(matched)}/10")
    
    # Read existing target expression
    existing = pd.read_csv(target_path)
    existing_cells = set(existing['cell_line'].values)
    print(f"Existing entries: {len(existing_cells)}")
    
    # Read CCLE for matched cell lines
    gene_data = {}  # gene -> {ccle_name: value}
    
    with gzip.open(ccle_path, 'rt') as f:
        f.readline()  # #1.2
        f.readline()  # dims
        header = f.readline().strip().split('\t')
        
        target_ccle_names = set(matched.values())
        col_indices = {name: i for i, name in enumerate(header) if name in target_ccle_names}
        
        for line in f:
            parts = line.strip().split('\t')
            gene = parts[1]
            if gene in TARGET_GENES:
                for ccle_name, col_idx in col_indices.items():
                    try:
                        val = float(parts[col_idx])
                    except:
                        val = 0.0
                    if gene not in gene_data:
                        gene_data[gene] = {}
                    gene_data[gene][ccle_name] = val
    
    # Add new entries to target expression
    target_genes_found = sorted(gene_data.keys())
    print(f"Target genes extracted: {len(target_genes_found)}")
    
    new_rows = []
    for oneil, ccle in matched.items():
        if ccle not in existing_cells:
            row = {'cell_line': ccle}
            for gene in target_genes_found:
                row[gene] = gene_data.get(gene, {}).get(ccle, 0.0)
            new_rows.append(row)
            
            # Also add with short name for matching
            row2 = row.copy()
            row2['cell_line'] = oneil + '_MAPPED'
            new_rows.append(row2)
    
    if new_rows:
        new_df = pd.DataFrame(new_rows)
        # Align columns
        for col in existing.columns:
            if col not in new_df.columns and col != 'cell_line':
                new_df[col] = 0.0
        new_df = new_df[existing.columns]
        
        combined = pd.concat([existing, new_df], ignore_index=True)
        combined.to_csv(target_path, index=False)
        print(f"\nAdded {len(new_rows)} new cell line entries")
        print(f"Updated: {target_path} ({len(combined)} total)")
    
    # Save mapping file
    map_df = pd.DataFrame([
        {'oneil_name': k, 'ccle_name': v} for k, v in matched.items()
    ])
    map_df.to_csv(CCLE_DIR / "cell_line_mapping.csv", index=False)
    print(f"Saved mapping: {CCLE_DIR / 'cell_line_mapping.csv'}")


if __name__ == "__main__":
    main()
