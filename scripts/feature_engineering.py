#!/usr/bin/env python3
"""
Advanced Feature Engineering for PrPc Prediction
Week 2: Feature extraction and engineering

Creates comprehensive feature matrix from:
1. Genomic features (PRNP + cancer genes)
2. Clinical features
3. Pathway features
4. Network features
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
from scipy import stats
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

# Configuration
DATA_DIR = Path("data/analysis/prpc_validation")
OUTPUT_DIR = DATA_DIR / "features"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("Feature Engineering Pipeline")
print("=" * 80)

class FeatureEngineer:
    """Advanced feature engineering for PrPc prediction"""
    
    def __init__(self, df):
        self.df = df
        self.features = {}
        self.metadata = {
            "n_samples": len(df),
            "feature_groups": {}
        }
    
    def extract_genomic_features(self):
        """Extract genomic features"""
        print("\n1. Genomic Features")
        print("-" * 40)
        
        # Primary feature: PRNP expression
        prnp_values = self.df['PRNP_value'].values
        
        features = {
            'PRNP_raw': prnp_values,
            'PRNP_log2': np.log2(prnp_values + 1),
            'PRNP_zscore': stats.zscore(prnp_values),
            'PRNP_quantile': pd.qcut(prnp_values, q=100, labels=False, duplicates='drop')
        }
        
        # In production: add correlate genes from co-expression analysis
        # For now: simulate cancer-related genes
        cancer_genes = ['TP53', 'KRAS', 'APC', 'EGFR', 'MYC', 'PIK3CA', 
                       'BRAF', 'PTEN', 'RB1', 'BRCA1']
        
        for gene in cancer_genes:
            # Simulate (in production: extract from expression data)
            features[f'{gene}_expr'] = np.random.randn(len(self.df))
        
        # Gene ratios (biomarker patterns)
        features['PRNP_TP53_ratio'] = features['PRNP_raw'] / (features['TP53_expr'] + 1)
        features['PRNP_KRAS_ratio'] = features['PRNP_raw'] / (features['KRAS_expr'] + 1)
        
        # Polynomial features (interactions)
        features['PRNP_squared'] = features['PRNP_raw'] ** 2
        features['PRNP_cubed'] = features['PRNP_raw'] ** 3
        
        genomic_df = pd.DataFrame(features)
        
        print(f"   Created {len(genomic_df.columns)} genomic features")
        
        self.features['genomic'] = genomic_df
        self.metadata['feature_groups']['genomic'] = {
            'n_features': len(genomic_df.columns),
            'features': list(genomic_df.columns)
        }
        
        return genomic_df
    
    def extract_clinical_features(self):
        """Extract clinical features"""
        print("\n2. Clinical Features")
        print("-" * 40)
        
        features = {}
        
        # Cancer type (one-hot encoding)
        if 'cancer_type' in self.df.columns:
            cancer_dummies = pd.get_dummies(self.df['cancer_type'], prefix='cancer')
            for col in cancer_dummies.columns:
                features[col] = cancer_dummies[col].values
        
        # Sample type
        if 'sample_type' in self.df.columns:
            features['is_tumor'] = (self.df['sample_type'] == 'Tumor').astype(int).values
        
        # Data source
        if 'data_source' in self.df.columns:
            source_dummies = pd.get_dummies(self.df['data_source'], prefix='source')
            for col in source_dummies.columns:
                features[col] = source_dummies[col].values
        
        clinical_df = pd.DataFrame(features)
        
        print(f"   Created {len(clinical_df.columns)} clinical features")
        
        self.features['clinical'] = clinical_df
        self.metadata['feature_groups']['clinical'] = {
            'n_features': len(clinical_df.columns),
            'features': list(clinical_df.columns)
        }
        
        return clinical_df
    
    def extract_statistical_features(self):
        """Extract statistical features from PRNP"""
        print("\n3. Statistical Features")
        print("-" * 40)
        
        prnp = self.df['PRNP_value'].values
        
        features = {
            # Distribution features
            'PRNP_skewness': pd.Series(prnp).rolling(100, min_periods=1).skew().fillna(0).values,
            'PRNP_kurtosis': pd.Series(prnp).rolling(100, min_periods=1).kurt().fillna(0).values,
            
            # Percentile features
            'PRNP_pct_10': np.percentile(prnp, 10),
            'PRNP_pct_25': np.percentile(prnp, 25),
            'PRNP_pct_50': np.percentile(prnp, 50),
            'PRNP_pct_75': np.percentile(prnp, 75),
            'PRNP_pct_90': np.percentile(prnp, 90),
            
            # Deviation features
            'PRNP_mad': stats.median_abs_deviation(prnp),
            'PRNP_iqr': stats.iqr(prnp),
        }
        
        # Expand to full array
        for key in ['PRNP_pct_10', 'PRNP_pct_25', 'PRNP_pct_50', 
                   'PRNP_pct_75', 'PRNP_pct_90', 'PRNP_mad', 'PRNP_iqr']:
            features[key] = np.full(len(self.df), features[key])
        
        stat_df = pd.DataFrame(features)
        
        print(f"   Created {len(stat_df.columns)} statistical features")
        
        self.features['statistical'] = stat_df
        self.metadata['feature_groups']['statistical'] = {
            'n_features': len(stat_df.columns),
            'features': list(stat_df.columns)
        }
        
        return stat_df
    
    def extract_pathway_features(self):
        """Extract pathway-based features (simulated)"""
        print("\n4. Pathway Features")
        print("-" * 40)
        
        # In production: KEGG/Reactome pathway enrichment scores
        pathways = [
            'cell_adhesion', 'immune_response', 'apoptosis',
            'cell_cycle', 'DNA_repair', 'metabolism',
            'signaling', 'proliferation'
        ]
        
        features = {}
        for pathway in pathways:
            # Simulate pathway activity (in production: GSEA scores)
            features[f'pathway_{pathway}'] = np.random.randn(len(self.df))
        
        pathway_df = pd.DataFrame(features)
        
        print(f"   Created {len(pathway_df.columns)} pathway features")
        
        self.features['pathway'] = pathway_df
        self.metadata['feature_groups']['pathway'] = {
            'n_features': len(pathway_df.columns),
            'features': list(pathway_df.columns)
        }
        
        return pathway_df
    
    def create_interaction_features(self):
        """Create interaction features"""
        print("\n5. Interaction Features")
        print("-" * 40)
        
        genomic = self.features['genomic']
        
        features = {}
        
        # PRNP × cancer type interactions
        if 'clinical' in self.features:
            clinical = self.features['clinical']
            cancer_cols = [c for c in clinical.columns if c.startswith('cancer_')]
            
            for cancer_col in cancer_cols[:3]:  # Limit to top 3
                features[f'PRNP_x_{cancer_col}'] = (
                    genomic['PRNP_raw'].values * clinical[cancer_col].values
                )
        
        # PRNP × gene interactions
        for gene in ['TP53_expr', 'KRAS_expr', 'APC_expr']:
            if gene in genomic.columns:
                features[f'PRNP_x_{gene}'] = (
                    genomic['PRNP_raw'].values * genomic[gene].values
                )
        
        interaction_df = pd.DataFrame(features)
        
        print(f"   Created {len(interaction_df.columns)} interaction features")
        
        self.features['interaction'] = interaction_df
        self.metadata['feature_groups']['interaction'] = {
            'n_features': len(interaction_df.columns),
            'features': list(interaction_df.columns)
        }
        
        return interaction_df
    
    def apply_dimensionality_reduction(self, n_components=50):
        """Apply PCA for dimensionality reduction"""
        print("\n6. Dimensionality Reduction (PCA)")
        print("-" * 40)
        
        # Combine all features
        all_features = pd.concat(self.features.values(), axis=1)
        
        # Normalize
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(all_features)
        
        # Adjust n_components to not exceed n_features
        n_components = min(n_components, all_features.shape[1])
        
        # PCA
        pca = PCA(n_components=n_components)
        features_pca = pca.fit_transform(features_scaled)
        
        print(f"   Original features: {all_features.shape[1]}")
        print(f"   PCA components: {n_components}")
        print(f"   Explained variance: {pca.explained_variance_ratio_.sum():.2%}")
        
        # Create PCA dataframe
        pca_df = pd.DataFrame(
            features_pca,
            columns=[f'PC{i+1}' for i in range(n_components)]
        )
        
        self.features['pca'] = pca_df
        self.metadata['pca'] = {
            'n_components': n_components,
            'explained_variance': float(pca.explained_variance_ratio_.sum()),
            'variance_per_component': pca.explained_variance_ratio_.tolist()
        }
        
        return pca_df, pca, scaler
    
    def create_targets(self):
        """Create prediction targets"""
        print("\n7. Creating Targets")
        print("-" * 40)
        
        prnp = self.df['PRNP_value'].values
        
        targets = {
            # Regression: PRNP level
            'prnp_value': prnp,
            'prnp_log2': np.log2(prnp + 1),
            
            # Binary: High vs Low
            'prnp_high': (prnp > np.median(prnp)).astype(int),
            
            # Multi-class: Quartiles
            'prnp_quartile': pd.qcut(prnp, q=4, labels=[0, 1, 2, 3]).astype(int),
        }
        
        # Cancer type classification
        if 'cancer_group' in self.df.columns:
            cancer_mapping = {ct: i for i, ct in enumerate(self.df['cancer_group'].unique())}
            targets['cancer_class'] = self.df['cancer_group'].map(cancer_mapping).values
        
        targets_df = pd.DataFrame(targets)
        
        print(f"   Created {len(targets_df.columns)} target variables")
        
        self.metadata['targets'] = {
            'n_targets': len(targets_df.columns),
            'targets': list(targets_df.columns)
        }
        
        return targets_df
    
    def save_all(self):
        """Save all features and metadata"""
        print("\n8. Saving Features")
        print("-" * 40)
        
        # Combine all features
        all_features = pd.concat(self.features.values(), axis=1)
        
        # Save full feature matrix
        full_file = OUTPUT_DIR / "feature_matrix_full.csv"
        all_features.to_csv(full_file, index=False)
        print(f"   Full features: {full_file}")
        print(f"   Shape: {all_features.shape}")
        
        # Save feature groups separately
        for group_name, group_df in self.features.items():
            group_file = OUTPUT_DIR / f"features_{group_name}.csv"
            group_df.to_csv(group_file, index=False)
            print(f"   {group_name}: {group_file}")
        
        # Save metadata
        self.metadata['total_features'] = all_features.shape[1]
        metadata_file = OUTPUT_DIR / "feature_metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(self.metadata, f, indent=2)
        print(f"   Metadata: {metadata_file}")
        
        return all_features


def main():
    """Main execution"""
    
    # Load unified cohort
    cohort_file = DATA_DIR / "unified_cohort_harmonized.csv"
    
    if not cohort_file.exists():
        print(f"⚠️  Unified cohort not found: {cohort_file}")
        print(f"   Run harmonize_datasets.py first")
        
        # For testing, use TCGA data
        print(f"\n   Using TCGA data for demonstration...")
        cohort_file = DATA_DIR / "open_data/real/tcga_all_cancers_prnp_real.csv"
        
        if not cohort_file.exists():
            print(f"❌ No data found")
            return None
        
        df = pd.read_csv(cohort_file)
        df = df.rename(columns={'PRNP_log2': 'PRNP_value'})
        df['cancer_group'] = df['cancer_type']
        df['data_source'] = 'TCGA'
    else:
        df = pd.read_csv(cohort_file)
    
    print(f"\n✓ Loaded {len(df)} samples")
    
    # Initialize feature engineer
    engineer = FeatureEngineer(df)
    
    # Extract all feature groups
    engineer.extract_genomic_features()
    engineer.extract_clinical_features()
    engineer.extract_statistical_features()
    engineer.extract_pathway_features()
    engineer.create_interaction_features()
    
    # PCA
    pca_features, pca_model, scaler = engineer.apply_dimensionality_reduction(n_components=50)
    
    # Create targets
    targets = engineer.create_targets()
    targets_file = OUTPUT_DIR / "targets.csv"
    targets.to_csv(targets_file, index=False)
    print(f"   Targets: {targets_file}")
    
    # Save all
    all_features = engineer.save_all()
    
    print("\n" + "=" * 80)
    print("✅ FEATURE ENGINEERING COMPLETE!")
    print("=" * 80)
    print(f"Total samples: {len(df)}")
    print(f"Total features: {all_features.shape[1]}")
    print(f"Feature groups: {len(engineer.features)}")
    print(f"Output directory: {OUTPUT_DIR}")
    
    return all_features, targets


if __name__ == "__main__":
    features, targets = main()
