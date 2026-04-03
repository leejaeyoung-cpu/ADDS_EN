"""
PrPc/PRNP Expression and KRAS Mutation Correlation Analysis

This module analyzes PrPc expression data across cancer types and correlates
with KRAS mutation prevalence using the ADDS knowledge base.
"""

import pandas as pd
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

class PrionKRASAnalyzer:
    """Analyze PrPc expression and KRAS mutation correlation"""
    
    def __init__(self, data_dir: str = "C:/Users/brook/Desktop/ADDS/data"):
        self.data_dir = Path(data_dir)
        self.prpc_file = Path("C:/Users/brook/Desktop/ADDS/PrPc, PRNP 암항원 발현 암종별 비율표.xlsx")
        self.kb_file = self.data_dir / "knowledge_base" / "cancer_knowledge_base.json"
        
        self.prpc_data = None
        self.kras_data = None
        self.correlation_results = None
        
    def load_prpc_expression_data(self) -> Dict[str, pd.DataFrame]:
        """
        Load PrPc expression data from Excel file with proper encoding handling
        
        Returns:
            Dictionary of cancer type -> DataFrame with expression data
        """
        print(f"Loading PrPc expression data from {self.prpc_file}")
        
        try:
            # Read all sheets
            all_sheets = pd.read_excel(
                self.prpc_file, 
                sheet_name=None,
                engine='openpyxl'
            )
            
            # Process main summary sheet
            summary_sheet = all_sheets['Sheet1']
            
            # Extract clean data from summary (skip header rows)
            prpc_expression = {}
            
            # Parse the main table - columns should be: 암항원, 암종, 발현비율
            if len(summary_sheet) > 1:
                # Find the actual data rows (skip headers)
                for idx in range(len(summary_sheet)):
                    row = summary_sheet.iloc[idx]
                    
                    # Check if this is a data row (has cancer type in second column)
                    cancer_type = row.iloc[1] if len(row) > 1 else None
                    expression_rate = row.iloc[2] if len(row) > 2 else None
                    
                    if pd.notna(cancer_type) and cancer_type not in ['암종', ''] and pd.notna(expression_rate):
                        # Normalize cancer type names
                        cancer_mapping = {
                            '유방암': 'breast',
                            '위암': 'gastric',
                            '췌장암': 'pancreatic',
                            '대장암': 'colorectal'
                        }
                        
                        cancer_eng = cancer_mapping.get(cancer_type, cancer_type)
                        prpc_expression[cancer_eng] = expression_rate
            
            self.prpc_data = prpc_expression
            
            print(f"Loaded PrPc expression data for {len(prpc_expression)} cancer types:")
            for cancer, rate in prpc_expression.items():
                print(f"  - {cancer}: {rate}")
            
            return all_sheets
            
        except Exception as e:
            print(f"Error loading PrPc data: {e}")
            raise
    
    def parse_expression_rate(self, rate_str) -> Tuple[float, float]:
        """
        Parse expression rate string to min/max values
        
        Args:
            rate_str: String like "15%~33%" or "66~70%" or 0.76
            
        Returns:
            Tuple of (min_rate, max_rate) as percentages
        """
        if isinstance(rate_str, (int, float)):
            # If it's a decimal like 0.76, assume it's a proportion
            if rate_str < 1:
                return (rate_str * 100, rate_str * 100)
            else:
                return (rate_str, rate_str)
        
        rate_str = str(rate_str).replace('%', '').replace(' ', '')
        
        if '~' in rate_str:
            parts = rate_str.split('~')
            min_rate = float(parts[0])
            max_rate = float(parts[1])
        else:
            min_rate = max_rate = float(rate_str)
        
        return (min_rate, max_rate)
    
    def query_kras_mutations(self) -> Dict[str, Dict]:
        """
        Query knowledge base for KRAS mutation data by cancer type
        
        Returns:
            Dictionary with KRAS mutation information per cancer type
        """
        print(f"\nQuerying KRAS mutations from knowledge base: {self.kb_file}")
        
        kras_info = {
            'colorectal': {'prevalence': 40, 'mutations': ['G12D', 'G12V', 'G13D'], 'source': 'literature'},
            'pancreatic': {'prevalence': 90, 'mutations': ['G12D', 'G12V', 'G12R'], 'source': 'literature'},
            'gastric': {'prevalence': 15, 'mutations': ['G12D', 'G12V'], 'source': 'literature'},
            'breast': {'prevalence': 5, 'mutations': ['G12D'], 'source': 'literature'},
            'lung': {'prevalence': 30, 'mutations': ['G12C', 'G12D', 'G12V'], 'source': 'literature'}
        }
        
        # Try to load from knowledge base
        if self.kb_file.exists():
            try:
                with open(self.kb_file, 'r', encoding='utf-8') as f:
                    kb_data = json.load(f)
                    
                # Extract KRAS information from knowledge base
                # The KB structure needs to be explored
                print(f"Knowledge base loaded with {len(kb_data)} entries")
                
                # For now, use literature-based estimates
                # TODO: Parse actual KB for KRAS mutation data
                
            except Exception as e:
                print(f"Warning: Could not load KB: {e}")
        
        self.kras_data = kras_info
        
        print(f"\nKRAS mutation prevalence by cancer type:")
        for cancer, info in kras_info.items():
            print(f"  - {cancer}: {info['prevalence']}% (mutations: {', '.join(info['mutations'])})")
        
        return kras_info
    
    def correlate_prpc_kras(self) -> pd.DataFrame:
        """
        Perform statistical correlation analysis between PrPc expression and KRAS prevalence
        
        Returns:
            DataFrame with correlation results
        """
        print("\n" + "="*60)
        print("CORRELATION ANALYSIS: PrPc Expression vs KRAS Mutation")
        print("="*60)
        
        if self.prpc_data is None:
            self.load_prpc_expression_data()
        if self.kras_data is None:
            self.query_kras_mutations()
        
        # Build correlation dataset
        cancer_types = []
        prpc_expression = []
        kras_prevalence = []
        
        for cancer in self.prpc_data.keys():
            if cancer in self.kras_data:
                cancer_types.append(cancer)
                
                # Get PrPc expression (use midpoint for ranges)
                rate_str = self.prpc_data[cancer]
                min_rate, max_rate = self.parse_expression_rate(rate_str)
                avg_rate = (min_rate + max_rate) / 2
                prpc_expression.append(avg_rate)
                
                # Get KRAS prevalence
                kras_prevalence.append(self.kras_data[cancer]['prevalence'])
        
        # Create DataFrame
        corr_df = pd.DataFrame({
            'cancer_type': cancer_types,
            'prpc_expression': prpc_expression,
            'kras_prevalence': kras_prevalence
        })
        
        # Calculate correlation
        if len(prpc_expression) > 2:
            pearson_r, pearson_p = stats.pearsonr(prpc_expression, kras_prevalence)
            spearman_r, spearman_p = stats.spearmanr(prpc_expression, kras_prevalence)
            
            print(f"\nCorrelation Statistics ({len(cancer_types)} cancer types):")
            print(f"  Pearson correlation: r = {pearson_r:.3f}, p = {pearson_p:.4f}")
            print(f"  Spearman correlation: ρ = {spearman_r:.3f}, p = {spearman_p:.4f}")
            
            # Add to dataframe
            corr_df['pearson_r'] = pearson_r
            corr_df['pearson_p'] = pearson_p
            corr_df['spearman_r'] = spearman_r
            corr_df['spearman_p'] = spearman_p
        
        print("\nData Summary:")
        print(corr_df[['cancer_type', 'prpc_expression', 'kras_prevalence']].to_string(index=False))
        
        self.correlation_results = corr_df
        return corr_df
    
    def generate_expression_heatmap(self, output_path: Optional[str] = None):
        """
        Generate visualization heatmap of PrPc expression by cancer type
        
        Args:
            output_path: Path to save figure (optional)
        """
        if self.correlation_results is None:
            self.correlate_prpc_kras()
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Plot 1: PrPc Expression by Cancer Type
        ax1 = axes[0]
        cancer_types = self.correlation_results['cancer_type'].values
        prpc_exp = self.correlation_results['prpc_expression'].values
        
        colors = plt.cm.RdYlGn(prpc_exp / 100)
        ax1.barh(cancer_types, prpc_exp, color=colors)
        ax1.set_xlabel('PrPc Expression (%)', fontsize=12)
        ax1.set_title('PrPc/PRNP Expression by Cancer Type', fontsize=14, fontweight='bold')
        ax1.grid(axis='x', alpha=0.3)
        
        # Add value labels
        for i, (ct, exp) in enumerate(zip(cancer_types, prpc_exp)):
            ax1.text(exp + 2, i, f'{exp:.1f}%', va='center', fontsize=10)
        
        # Plot 2: Correlation scatter
        ax2 = axes[1]
        ax2.scatter(
            self.correlation_results['kras_prevalence'],
            self.correlation_results['prpc_expression'],
            s=200, alpha=0.6, c=colors
        )
        
        # Add labels for each point
        for _, row in self.correlation_results.iterrows():
            ax2.annotate(
                row['cancer_type'],
                (row['kras_prevalence'], row['prpc_expression']),
                xytext=(5, 5), textcoords='offset points',
                fontsize=10
            )
        
        ax2.set_xlabel('KRAS Mutation Prevalence (%)', fontsize=12)
        ax2.set_ylabel('PrPc Expression (%)', fontsize=12)
        ax2.set_title('PrPc Expression vs KRAS Prevalence', fontsize=14, fontweight='bold')
        ax2.grid(alpha=0.3)
        
        # Add correlation annotation
        if 'pearson_r' in self.correlation_results.columns:
            r = self.correlation_results['pearson_r'].iloc[0]
            p = self.correlation_results['pearson_p'].iloc[0]
            ax2.text(
                0.05, 0.95, f'r = {r:.3f}\np = {p:.4f}',
                transform=ax2.transAxes,
                fontsize=11, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
            )
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"\nVisualization saved to: {output_path}")
        
        return fig
    
    def export_results(self, output_dir: Optional[str] = None) -> str:
        """
        Export analysis results to JSON
        
        Args:
            output_dir: Directory to save results
            
        Returns:
            Path to saved file
        """
        if output_dir is None:
            output_dir = self.data_dir / "analysis"
        else:
            output_dir = Path(output_dir)
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        results = {
            'prpc_expression': self.prpc_data,
            'kras_mutations': self.kras_data,
            'correlation_analysis': self.correlation_results.to_dict('records') if self.correlation_results is not None else None,
            'metadata': {
                'analysis_date': pd.Timestamp.now().isoformat(),
                'n_cancer_types': len(self.prpc_data) if self.prpc_data else 0
            }
        }
        
        output_file = output_dir / "prion_kras_correlation_results.json"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"\nResults exported to: {output_file}")
        return str(output_file)


def main():
    """Run complete PrPc-KRAS correlation analysis"""
    
    print("="*70)
    print("PrPc/PRNP Expression and KRAS Mutation Correlation Analysis")
    print("="*70)
    
    analyzer = PrionKRASAnalyzer()
    
    # Step 1: Load PrPc expression data
    print("\nSTEP 1: Loading PrPc Expression Data")
    print("-" * 70)
    analyzer.load_prpc_expression_data()
    
    # Step 2: Query KRAS mutation data
    print("\nSTEP 2: Querying KRAS Mutation Data")
    print("-" * 70)
    analyzer.query_kras_mutations()
    
    # Step 3: Correlation analysis
    print("\nSTEP 3: Correlation Analysis")
    print("-" * 70)
    results = analyzer.correlate_prpc_kras()
    
    # Step 4: Generate visualizations
    print("\nSTEP 4: Generating Visualizations")
    print("-" * 70)
    output_path = "C:/Users/brook/Desktop/ADDS/data/analysis/prpc_kras_correlation.png"
    analyzer.generate_expression_heatmap(output_path)
    
    # Step 5: Export results
    print("\nSTEP 5: Exporting Results")
    print("-" * 70)
    analyzer.export_results()
    
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)
    print(f"\nKey Findings:")
    print(f"  - {len(analyzer.prpc_data)} cancer types analyzed")
    print(f"  - PrPc expression ranges from {min(results['prpc_expression']):.1f}% to {max(results['prpc_expression']):.1f}%")
    print(f"  - KRAS prevalence ranges from {min(results['kras_prevalence'])}% to {max(results['kras_prevalence'])}%")
    
    if 'pearson_r' in results.columns:
        r = results['pearson_r'].iloc[0]
        p = results['pearson_p'].iloc[0]
        significance = "significant" if p < 0.05 else "not significant"
        print(f"  - Correlation: r = {r:.3f}, p = {p:.4f} ({significance})")


if __name__ == "__main__":
    main()
