"""
PrPC Expression Data Analyzer
==============================
Analyzes PrPC/PRNP expression across cancer types from Excel data

Author: ADDS Research Team
Date: 2026-01-31
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json

class PrPCExpressionAnalyzer:
    """Analyze PrPC/PRNP expression data across cancer types"""
    
    def __init__(self, excel_path):
        """
        Initialize analyzer
        
        Args:
            excel_path: Path to Excel file with PrPC expression data
        """
        self.excel_path = Path(excel_path)
        self.data = {}
        self.summary = {}
        
    def load_data(self):
        """Load all sheets from Excel file"""
        print(f"Loading data from {self.excel_path}...")
        
        try:
            xls = pd.ExcelFile(self.excel_path, engine='openpyxl')
            print(f"Found {len(xls.sheet_names)} sheets: {xls.sheet_names}")
            
            for sheet_name in xls.sheet_names:
                df = pd.read_excel(xls, sheet_name=sheet_name)
                self.data[sheet_name] = df
                print(f"  - {sheet_name}: {df.shape[0]} rows x {df.shape[1]} cols")
            
            print("[OK] Data loaded successfully\n")
            return True
            
        except Exception as e:
            print(f"[ERROR] Failed to load data: {e}")
            return False
    
    def analyze_expression_by_cancer(self):
        """Analyze PrPC expression for each cancer type"""
        print("Analyzing expression by cancer type...\n")
        
        results = {}
        
        for sheet_name, df in self.data.items():
            print(f"=== {sheet_name} ===")
            print(f"Shape: {df.shape}")
            print(f"Columns: {df.columns.tolist()}")
            print("\nData preview:")
            print(df.head(10).to_string())
            print("\n" + "="*80 + "\n")
            
            results[sheet_name] = {
                'shape': df.shape,
                'columns': df.columns.tolist(),
                'preview': df.head().to_dict()
            }
        
        self.summary['by_cancer'] = results
        return results
    
    def extract_expression_rates(self):
        """
        Extract expression rates from data
        Looks for percentage values and categorizes by cancer type
        """
        print("Extracting expression rates...\n")
        
        rates = {}
        
        for sheet_name, df in self.data.items():
            # Try to find percentage data
            # This will need to be customized based on actual Excel structure
            sheet_rates = []
            
            # Scan all cells for percentage patterns
            for col in df.columns:
                for val in df[col]:
                    if pd.notna(val) and isinstance(val, str):
                        # Look for patterns like "15%", "15-33%", etc.
                        if '%' in val:
                            sheet_rates.append(val)
            
            rates[sheet_name] = sheet_rates
            print(f"{sheet_name}: {len(sheet_rates)} percentage values found")
            if sheet_rates:
                print(f"  Examples: {sheet_rates[:5]}")
            print()
        
        self.summary['expression_rates'] = rates
        return rates
    
    def save_summary(self, output_path="data/prpc_analysis_summary.json"):
        """Save analysis summary to JSON"""
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(self.summary, f, indent=2, ensure_ascii=False, default=str)
        
        print(f"[OK] Summary saved to {output_file}")
    
    def generate_report(self, output_path="data/prpc_expression_report.md"):
        """Generate markdown report"""
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        md = "# PrPC/PRNP Expression Analysis Report\n\n"
        md += f"**Analysis Date**: 2026-01-31\n"
        md += f"**Data Source**: {self.excel_path.name}\n\n"
        md += "---\n\n"
        
        md += "## Cancer Types Analyzed\n\n"
        for sheet_name in self.data.keys():
            md += f"- {sheet_name}\n"
        md += "\n---\n\n"
        
        md += "## Expression Data by Cancer Type\n\n"
        for sheet_name, df in self.data.items():
            md += f"### {sheet_name}\n\n"
            md += f"- **Rows**: {df.shape[0]}\n"
            md += f"- **Columns**: {df.shape[1]}\n"
            md += f"- **Column Names**: {', '.join([str(c) for c in df.columns])}\n\n"
            
            # Add table preview
            md += "**Data Preview**:\n\n"
            md += df.head(5).to_markdown() + "\n\n"
        
        md += "---\n\n"
        md += f"**Analysis Complete**: {len(self.data)} cancer types processed\n"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(md)
        
        print(f"[OK] Report saved to {output_file}")
        return output_file


def main():
    """Main analysis pipeline"""
    print("="*80)
    print("PrPC/PRNP Cancer Expression Analysis")
    print("="*80)
    print()
    
    # Initialize analyzer
    analyzer = PrPCExpressionAnalyzer("PrPc, PRNP 암항원 발현 암종별 비율표.xlsx")
    
    # Load data
    if not analyzer.load_data():
        return
    
    # Analyze by cancer type
    analyzer.analyze_expression_by_cancer()
    
    # Extract rates
    analyzer.extract_expression_rates()
    
    # Save outputs
    analyzer.save_summary()
    analyzer.generate_report()
    
    print("="*80)
    print("[SUCCESS] Analysis complete!")
    print("="*80)


if __name__ == "__main__":
    main()
