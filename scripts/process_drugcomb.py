"""Process DrugComb v1.4 and merge with O'Neil."""
import pandas as pd
import numpy as np
from pathlib import Path

DATA_DIR = Path("F:/ADDS/data/ml_training")
dc = pd.read_csv(DATA_DIR / "drugcomb/drugcomb_summary_v1.4.csv", low_memory=False)

print("=== DrugComb v1.4 Processing ===")
print(f"Records: {len(dc):,}")
print(f"Columns: {list(dc.columns)}")

dc = dc.dropna(subset=['synergy_loewe'])
dc = dc[dc['drug_row'] != dc['drug_col']]
print(f"After cleaning: {len(dc):,}")

n_drugs_r = dc["drug_row"].nunique()
n_drugs_c = dc["drug_col"].nunique()
n_cells = dc["cell_line_name"].nunique()
print(f"\nUnique drugs (row): {n_drugs_r}")
print(f"Unique drugs (col): {n_drugs_c}")
print(f"Unique cell lines: {n_cells}")
print(f"Synergy Loewe: mean={dc['synergy_loewe'].mean():.2f}, std={dc['synergy_loewe'].std():.2f}")

result = pd.DataFrame({
    'drug_a': dc['drug_row'].astype(str).str.upper().str.strip(),
    'drug_b': dc['drug_col'].astype(str).str.upper().str.strip(),
    'cell_line': dc['cell_line_name'].astype(str).str.upper().str.strip(),
    'synergy_loewe': dc['synergy_loewe'].astype(float),
    'source': 'drugcomb',
})

np.random.seed(42)
result['fold'] = np.random.randint(0, 5, len(result))

out = DATA_DIR / "drugcomb/drugcomb_processed.csv"
result.to_csv(out, index=False)
print(f"\nProcessed: {out} ({len(result):,} records)")

# Merge with O'Neil
oneil = pd.read_csv(DATA_DIR / "oneil_synergy.csv")
oneil['drug_a'] = oneil['drug_a'].str.upper().str.strip()
oneil['drug_b'] = oneil['drug_b'].str.upper().str.strip()
oneil['cell_line'] = oneil['cell_line'].str.upper().str.strip()
oneil['source'] = 'oneil'

combined = pd.concat([oneil, result], ignore_index=True)
combined = combined.drop_duplicates(subset=['drug_a', 'drug_b', 'cell_line'], keep='first')

merged_path = DATA_DIR / "synergy_combined.csv"
combined.to_csv(merged_path, index=False)

print(f"\n=== FINAL COMBINED DATASET ===")
print(f"  O'Neil:     {len(oneil):>10,} records")
print(f"  DrugComb:   {len(result):>10,} records")
print(f"  Combined:   {len(combined):>10,} records (deduped)")
n_all_drugs = combined['drug_a'].nunique() + combined['drug_b'].nunique()
n_all_cells = combined['cell_line'].nunique()
print(f"  Unique drugs: {n_all_drugs}")
print(f"  Unique cells: {n_all_cells}")
print(f"  Saved: {merged_path}")
