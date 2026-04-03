#!/usr/bin/env python3
"""
Data Harmonization Script
Combines TCGA, GEO, and cBioPortal data into unified 10K cohort

Phase 1, Week 1, Day 3
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
from datetime import datetime
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Paths
BASE_DIR = Path("data/analysis/prpc_validation/open_data")
OUTPUT_DIR = Path("data/analysis/prpc_validation")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

class DataHarmonizer:
    """Harmonize multi-source PRNP expression data"""
    
    def __init__(self):
        self.metadata = {
            "harmonization_date": datetime.now().isoformat(),
            "sources": {},
            "total_samples": 0,
            "excluded_samples": 0,
            "final_samples": 0
        }
    
    def load_tcga(self):
        """Load TCGA data"""
        print("\n📂 Loading TCGA data...")
        
        file = BASE_DIR / "real" / "tcga_all_cancers_prnp_real.csv"
        
        if not file.exists():
            print(f"   ⚠️  TCGA file not found: {file}")
            return None
        
        df = pd.read_csv(file)
        print(f"   ✓ Loaded {len(df)} TCGA samples")
        
        # Standardize columns
        df_std = pd.DataFrame({
            'sample_id': df['sample_id'],
            'PRNP_value': df['PRNP_log2'],  # Use log2 transformed
            'cancer_type': df['cancer_type'],
            'sample_type': df['sample_type'],
            'data_source': 'TCGA'
        })
        
        self.metadata['sources']['TCGA'] = {
            'file': str(file),
            'n_samples': len(df_std),
            'cancer_types': df_std['cancer_type'].unique().tolist()
        }
        
        return df_std
    
    def load_geo(self):
        """Load GEO data"""
        print("\n📂 Loading GEO data...")
        
        file = BASE_DIR / "geo" / "geo_all_prnp_combined.csv"
        
        if not file.exists():
            print(f"   ⚠️  GEO file not found: {file}")
            print(f"   Skipping GEO data...")
            return None
        
        df = pd.read_csv(file)
        print(f"   ✓ Loaded {len(df)} GEO samples")
        
        # Standardize columns
        df_std = pd.DataFrame({
            'sample_id': df['sample_id'],
            'PRNP_value': df['PRNP_expression'],
            'cancer_type': df.get('cancer_type', 'Unknown'),
            'sample_type': df.get('sample_type', 'Tumor'),
            'data_source': 'GEO',
            'GSE_id': df.get('GSE_id', 'Unknown')
        })
        
        self.metadata['sources']['GEO'] = {
            'file': str(file),
            'n_samples': len(df_std),
            'datasets': df_std['GSE_id'].unique().tolist() if 'GSE_id' in df_std else []
        }
        
        return df_std
    
    def load_cbioportal(self):
        """Load cBioPortal data"""
        print("\n📂 Loading cBioPortal data...")
        
        file = BASE_DIR / "cbioportal" / "cbioportal_prnp_with_clinical.csv"
        
        if not file.exists():
            print(f"   ⚠️  cBioPortal file not found: {file}")
            print(f"   Skipping cBioPortal data...")
            return None
        
        df = pd.read_csv(file)
        print(f"   ✓ Loaded {len(df)} cBioPortal samples")
        
        # Standardize columns
        df_std = pd.DataFrame({
            'sample_id': df['sampleId'],
            'PRNP_value': df['value'],
            'cancer_type': df.get('studyId', 'Unknown'),
            'sample_type': df.get('sampleType', 'Tumor'),
            'data_source': 'cBioPortal',
            'study_id': df.get('study_id', 'Unknown')
        })
        
        self.metadata['sources']['cBioPortal'] = {
            'file': str(file),
            'n_samples': len(df_std),
            'studies': df_std['study_id'].unique().tolist() if 'study_id' in df_std else []
        }
        
        return df_std
    
    def normalize_expression(self, df):
        """Normalize expression values across sources"""
        print("\n🔧 Normalizing expression values...")
        
        # Group by data source and normalize
        normalized = []
        
        for source in df['data_source'].unique():
            source_df = df[df['data_source'] == source].copy()
            
            # Z-score normalization within source
            source_df['PRNP_zscore'] = stats.zscore(source_df['PRNP_value'].dropna())
            
            # Log2 scale (if not already)
            if source == 'TCGA':
                source_df['PRNP_log2'] = source_df['PRNP_value']  # Already log2
            else:
                source_df['PRNP_log2'] = np.log2(source_df['PRNP_value'] + 1)
            
            normalized.append(source_df)
        
        result = pd.concat(normalized, ignore_index=True)
        
        print(f"   ✓ Normalized {len(result)} samples")
        
        return result
    
    def quality_control(self, df):
        """Quality control - remove outliers and invalid data"""
        print("\n🔍 Quality control...")
        
        initial_n = len(df)
        
        # Remove missing PRNP values
        df = df.dropna(subset=['PRNP_value'])
        print(f"   Removed {initial_n - len(df)} samples with missing PRNP")
        
        # Remove extreme outliers (>5 SD)
        df = df[np.abs(df['PRNP_zscore']) < 5]
        excluded = initial_n - len(df)
        print(f"   Removed {excluded} extreme outliers (>5 SD)")
        
        self.metadata['excluded_samples'] = excluded
        self.metadata['final_samples'] = len(df)
        
        return df
    
    def add_features(self, df):
        """Add derived features"""
        print("\n➕ Adding features...")
        
        # Binary classification: High vs Low PRNP
        median_prnp = df['PRNP_zscore'].median()
        df['PRNP_high'] = (df['PRNP_zscore'] > median_prnp).astype(int)
        
        # Quartiles
        df['PRNP_quartile'] = pd.qcut(df['PRNP_zscore'], q=4, labels=['Q1', 'Q2', 'Q3', 'Q4'])
        
        # Cancer type grouping
        df['cancer_group'] = df['cancer_type'].apply(self.group_cancer_type)
        
        print(f"   ✓ Added 3 derived features")
        
        return df
    
    def group_cancer_type(self, cancer_type):
        """Group cancer types into major categories"""
        cancer_type_lower = str(cancer_type).lower()
        
        if 'coad' in cancer_type_lower or 'read' in cancer_type_lower or 'colorectal' in cancer_type_lower:
            return 'Colorectal'
        elif 'paad' in cancer_type_lower or 'pancrea' in cancer_type_lower:
            return 'Pancreatic'
        elif 'brca' in cancer_type_lower or 'breast' in cancer_type_lower:
            return 'Breast'
        elif 'stad' in cancer_type_lower or 'stomach' in cancer_type_lower or 'gastric' in cancer_type_lower:
            return 'Gastric'
        else:
            return 'Other'
    
    def harmonize_all(self):
        """Main harmonization pipeline"""
        print("=" * 80)
        print("🚀 Data Harmonization - Starting")
        print("=" * 80)
        
        # Load all sources
        tcga_df = self.load_tcga()
        geo_df = self.load_geo()
        cbio_df = self.load_cbioportal()
        
        # Combine
        dfs = [df for df in [tcga_df, geo_df, cbio_df] if df is not None]
        
        if not dfs:
            print("\n❌ No data loaded!")
            return None
        
        print(f"\n📊 Combining {len(dfs)} data sources...")
        combined = pd.concat(dfs, ignore_index=True)
        print(f"   ✓ Combined: {len(combined)} samples")
        
        self.metadata['total_samples'] = len(combined)
        
        # Normalize
        combined = self.normalize_expression(combined)
        
        # Quality control
        combined = self.quality_control(combined)
        
        # Add features
        combined = self.add_features(combined)
        
        # Save
        output_file = OUTPUT_DIR / "unified_cohort_harmonized.csv"
        combined.to_csv(output_file, index=False)
        
        print("\n" + "=" * 80)
        print("✅ Harmonization Complete!")
        print("=" * 80)
        print(f"📊 Summary:")
        print(f"   Total samples: {self.metadata['total_samples']}")
        print(f"   Final samples: {self.metadata['final_samples']}")
        print(f"   Excluded: {self.metadata['excluded_samples']}")
        print(f"   Data sources: {len(self.metadata['sources'])}")
        print(f"   Output file: {output_file}")
        
        # Save metadata
        metadata_file = OUTPUT_DIR / "harmonization_metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(self.metadata, f, indent=2)
        
        # Summary statistics
        print(f"\n📈 Data Distribution:")
        print(f"   By source:")
        print(combined['data_source'].value_counts())
        print(f"\n   By cancer group:")
        print(combined['cancer_group'].value_counts())
        print(f"\n   By sample type:")
        print(combined['sample_type'].value_counts())
        
        return combined


def main():
    """Main execution"""
    harmonizer = DataHarmonizer()
    result = harmonizer.harmonize_all()
    
    if result is not None:
        print(f"\n✅ Success! Harmonized {len(result)} samples")
        print(f"\nPreview:")
        print(result.head())
        print(f"\nColumns: {list(result.columns)}")
    else:
        print("\n⚠️  No data harmonized. Check if source files exist.")
    
    return result


if __name__ == "__main__":
    result = main()
