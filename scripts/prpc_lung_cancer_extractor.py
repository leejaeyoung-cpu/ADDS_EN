"""
PrPc Lung Cancer Data Extractor
================================
Extracts and analyzes lung cancer PrPc expression data.
"""

import sys
import json
import pandas as pd
from pathlib import Path

# Fix Windows encoding
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

# Paths
LUNG_FILE = Path("C:/Users/brook/Desktop/ADDS/prpc/prion 폐암 결과 (1).xlsx")
OUTPUT_DIR = Path("data/analysis/prpc_clinical_data")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("PrPc LUNG CANCER DATA EXTRACTION")
print("=" * 80)
print(f"Source: {LUNG_FILE.name}")
print()

# Load Excel file
excel_data = pd.ExcelFile(LUNG_FILE)
print(f"Sheets found: {excel_data.sheet_names}")
print(f"Total sheets: {len(excel_data.sheet_names)}")
print()

lung_data = {}
all_sheets_summary = []

for idx, sheet in enumerate(excel_data.sheet_names, 1):
    print(f"\n[{idx}/{len(excel_data.sheet_names)}] Processing sheet: {sheet}")
    print("-" * 80)
    
    df = pd.read_excel(LUNG_FILE, sheet_name=sheet)
    lung_data[sheet] = df
    
    print(f"  Shape: {df.shape[0]} rows x {df.shape[1]} cols")
    print(f"  Columns: {list(df.columns)}")
    
    # Preview data
    print("\n  Data Preview:")
    if df.shape[0] > 0:
        # Show first few rows
        preview = df.head(10)
        print(preview.to_string(max_cols=10, max_rows=10))
    else:
        print("  (Empty sheet)")
    
    # Look for key information
    # Check for PrPc expression data
    sheet_lower = sheet.lower()
    has_expression = any('expression' in str(col).lower() or 'prpc' in str(col).lower() or 
                         '발현' in str(col) for col in df.columns)
    
    # Count non-empty cells
    non_empty = df.notna().sum().sum()
    total_cells = df.shape[0] * df.shape[1]
    
    all_sheets_summary.append({
        'sheet_name': sheet,
        'rows': df.shape[0],
        'columns': df.shape[1],
        'non_empty_cells': int(non_empty),
        'total_cells': int(total_cells),
        'fill_rate': float(non_empty / total_cells if total_cells > 0 else 0),
        'has_expression_data': has_expression
    })
    
    print(f"\n  Non-empty cells: {non_empty}/{total_cells} ({non_empty/total_cells*100:.1f}%)")

# Summary table
print("\n" + "=" * 80)
print("SUMMARY OF ALL SHEETS")
print("=" * 80)

summary_df = pd.DataFrame(all_sheets_summary)
print(summary_df.to_string(index=False))

# Save summary
summary_file = OUTPUT_DIR / "lung_cancer_data_summary.xlsx"
summary_df.to_excel(summary_file, index=False, engine='openpyxl')
print(f"\n[SAVED] {summary_file}")

# Save all sheet data as individual Excel files (for detailed review)
for sheet, df in lung_data.items():
    if df.shape[0] > 0:  # Only save non-empty sheets
        safe_name = sheet.replace('/', '_').replace('\\', '_')[:50]
        sheet_file = OUTPUT_DIR / f"lung_cancer_sheet_{safe_name}.xlsx"
        df.to_excel(sheet_file, index=False, engine='openpyxl')
        print(f"[SAVED] {sheet_file}")

# Create comprehensive JSON
lung_json = {
    "source_file": LUNG_FILE.name,
    "total_sheets": len(excel_data.sheet_names),
    "sheets_summary": all_sheets_summary,
    "extraction_date": pd.Timestamp.now().isoformat()
}

json_file = OUTPUT_DIR / "lung_cancer_data_full.json"
with open(json_file, 'w', encoding='utf-8') as f:
    json.dump(lung_json, f, indent=2, ensure_ascii=False)
print(f"\n[SAVED] {json_file}")

print("\n" + "=" * 80)
print("EXTRACTION COMPLETE")
print("=" * 80)
print(f"\nTotal sheets processed: {len(lung_data)}")
print(f"Output files generated: {len(lung_data) + 2}")  # +2 for summary and JSON
print("\nNext step: Manual review of extracted lung cancer data")
