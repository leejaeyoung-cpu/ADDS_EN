"""
TCGA PRNP Real Data Download - Alternative Method
==================================================
Uses direct UCSC Xena dataset access instead of cohort queries

Method: Download full gene expression matrices and filter for PRNP
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path
import json
from datetime import datetime
import requests
import gzip
import io

# Fix encoding
sys.stdout.reconfigure(encoding='utf-8')

# Setup  
OUTPUT_DIR = Path("data/analysis/prpc_validation/open_data/real")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("TCGA REAL DATA DOWNLOAD - PRNP EXPRESSION (Alternative Method)")
print("=" * 80)
print()

# ============================================================================
# Direct Download URLs from UCSC Xena
# ============================================================================

# These URLs point to the actual data files hosted on UCSC Xena
DATASETS = {
    'COAD': {
        'name': 'TCGA Colon Adenocarcinoma',
        'url': 'https://tcga-xena-hub.s3.us-east-1.amazonaws.com/download/TCGA.COAD.sampleMap%2FHiSeqV2.gz',
        'clinical_url': 'https://tcga-xena-hub.s3.us-east-1.amazonaws.com/download/TCGA.COAD.sampleMap%2FCOAD_clinicalMatrix.gz'
    },
    'READ': {
        'name': 'TCGA Rectal Adenocarcinoma',
        'url': 'https://tcga-xena-hub.s3.us-east-1.amazonaws.com/download/TCGA.READ.sampleMap%2FHiSeqV2.gz',
        'clinical_url': 'https://tcga-xena-hub.s3.us-east-1.amazonaws.com/download/TCGA.READ.sampleMap%2FREAD_clinicalMatrix.gz'
    },
    'PAAD': {
        'name': 'TCGA Pancreatic Adenocarcinoma',
        'url': 'https://tcga-xena-hub.s3.us-east-1.amazonaws.com/download/TCGA.PAAD.sampleMap%2FHiSeqV2.gz',
        'clinical_url': 'https://tcga-xena-hub.s3.us-east-1.amazonaws.com/download/TCGA.PAAD.sampleMap%2FPAAD_clinicalMatrix.gz'
    },
    'STAD': {
        'name': 'TCGA Stomach Adenocarcinoma',
        'url': 'https://tcga-xena-hub.s3.us-east-1.amazonaws.com/download/TCGA.STAD.sampleMap%2FHiSeqV2.gz',
        'clinical_url': 'https://tcga-xena-hub.s3.us-east-1.amazonaws.com/download/TCGA.STAD.sampleMap%2FSTAD_clinicalMatrix.gz'
    },
    'BRCA': {
        'name': 'TCGA Breast Invasive Carcinoma',
        'url': 'https://tcga-xena-hub.s3.us-east-1.amazonaws.com/download/TCGA.BRCA.sampleMap%2FHiSeqV2.gz',
        'clinical_url': 'https://tcga-xena-hub.s3.us-east-1.amazonaws.com/download/TCGA.BRCA.sampleMap%2FBRCA_clinicalMatrix.gz'
    }
}

GENE = "PRNP"

# ============================================================================
# Download Function
# ============================================================================

def download_and_extract_prnp(cancer_type, config):
    """Download gene expression data and extract PRNP"""
    
    print(f"\n{'='*60}")
    print(f"Downloading {cancer_type}: {config['name']}")
    print(f"{'='*60}")
    
    try:
        # Download expression data
        print(f"  Downloading gene expression matrix...")
        print(f"  URL: {config['url']}")
        
        response = requests.get(config['url'], timeout=300)
        response.raise_for_status()
        
        # Decompress and read
        print(f"  Decompressing and parsing...")
        with gzip.open(io.BytesIO(response.content), 'rt') as f:
            # Read the full matrix
            expr_df = pd.read_csv(f, sep='\t', index_col=0)
        
        print(f"  Full matrix downloaded: {expr_df.shape[0]} genes × {expr_df.shape[1]} samples")
        
        # Extract PRNP row
        if GENE not in expr_df.index:
            print(f"  ✗ ERROR: {GENE} not found in gene list")
            return None
        
        prnp_expr = expr_df.loc[GENE]
        print(f"  ✓ Found {GENE} expression for {len(prnp_expr)} samples")
        
        # Create DataFrame
        df = pd.DataFrame({
            'sample_id': prnp_expr.index,
            'cancer_type': cancer_type,
            'PRNP_rsem': prnp_expr.values,
            'PRNP_log2': np.log2(prnp_expr.values + 1)
        })
        
        # Classify sample types based on TCGA barcode
        # TCGA sample codes: 01-09 = Tumor, 10-19 = Normal, 20-29 = Control
        def classify_sample(sample_id):
            parts = str(sample_id).split('-')
            if len(parts) >= 4:
                sample_code = parts[3][:2]
                if sample_code in ['01', '02', '03', '04', '05', '06', '07', '08', '09']:
                    return 'Tumor'
                elif sample_code in ['10', '11', '12', '13', '14']:
                    return 'Normal'
            return 'Unknown'
        
        df['sample_type'] = df['sample_id'].apply(classify_sample)
        
        # Try to download clinical data
        print(f"  Downloading clinical data...")
        try:
            clin_response = requests.get(config['clinical_url'], timeout=120)
            clin_response.raise_for_status()
            
            with gzip.open(io.BytesIO(clin_response.content), 'rt') as f:
                clin_df = pd.read_csv(f, sep='\t', index_col=0)
            
            # Merge with clinical data
            if 'pathologic_stage' in clin_df.columns:
                stage_map = clin_df['pathologic_stage'].to_dict()
                df['stage'] = df['sample_id'].map(stage_map)
            else:
                df['stage'] = 'Unknown'
            
            print(f"  ✓ Clinical data merged")
            
        except Exception as e:
            print(f"  ! Clinical data unavailable: {e}")
            df['stage'] = 'Unknown'
        
        # Clean stage names
        def clean_stage(stage):
            if pd.isna(stage) or stage == 'Unknown':
                return 'Unknown'
            stage_str = str(stage).upper()
            if 'STAGE I' in stage_str and not any(x in stage_str for x in ['II', 'III', 'IV']):
                return 'Stage I'
            elif 'STAGE II' in stage_str and not any(x in stage_str for x in ['III', 'IV']):
                return 'Stage II'
            elif 'STAGE III' in stage_str and 'IV' not in stage_str:
                return 'Stage III'
            elif 'STAGE IV' in stage_str:
                return 'Stage IV'
            return 'Unknown'
        
        df['stage_clean'] = df['stage'].apply(clean_stage)
        
        # Summary
        print(f"  ✓ Processed {len(df)} samples")
        print(f"    - Tumor: {(df['sample_type']=='Tumor').sum()}")
        print(f"    - Normal: {(df['sample_type']=='Normal').sum()}")
        print(f"    - PRNP range: {df['PRNP_rsem'].min():.2f} - {df['PRNP_rsem'].max():.2f}")
        
        return df
        
    except requests.RequestException as e:
        print(f"  ✗ Download ERROR: {e}")
        return None
    except Exception as e:
        print(f"  ✗ Processing ERROR: {e}")
        import traceback
        traceback.print_exc()
        return None

# ============================================================================
# Main Download Process
# ============================================================================

print("\nStarting TCGA real data download...")
print(f"Target gene: {GENE}")
print(f"Cancer types: {', '.join(DATASETS.keys())}")
print()
print("NOTE: This will download large gene expression matrices.")
print("Total download size: ~500MB - 2GB (compressed)")
print("Time estimate: 5-20 minutes depending on connection speed")
print()

all_datasets = {}
download_summary = {
    'download_date': datetime.now().isoformat(),
    'gene': GENE,
    'source': 'UCSC Xena TCGA Hub (Direct Download)',
    'method': 'Full matrix download + PRNP extraction',
    'cohorts': {},
    'errors': []
}

# Download each cancer type
for cancer_type, config in DATASETS.items():
    df = download_and_extract_prnp(cancer_type, config)
    
    if df is not None and len(df) > 0:
        all_datasets[cancer_type] = df
        
        # Save individual file
        outfile = OUTPUT_DIR / f"tcga_{cancer_type.lower()}_prnp_real.csv"
        df.to_csv(outfile, index=False)
        print(f"  [SAVED] {outfile}")
        
        # Update summary
        download_summary['cohorts'][cancer_type] = {
            'total_samples': int(len(df)),
            'tumor_samples': int((df['sample_type']=='Tumor').sum()),
            'normal_samples': int((df['sample_type']=='Normal').sum()),
            'prnp_mean': float(df['PRNP_rsem'].mean()),
            'prnp_std': float(df['PRNP_rsem'].std()),
            'prnp_min': float(df['PRNP_rsem'].min()),
            'prnp_max': float(df['PRNP_rsem'].max())
        }
    else:
        download_summary['errors'].append(cancer_type)

# ============================================================================
# Combine and Analyze
# ============================================================================

if all_datasets:
    print(f"\n{'='*80}")
    print("COMBINING DATASETS")
    print(f"{'='*80}")
    
    # Combine all
    combined_df = pd.concat(all_datasets.values(), ignore_index=True)
    
    # Save combined
    combined_file = OUTPUT_DIR / "tcga_all_cancers_prnp_real.csv"
    combined_df.to_csv(combined_file, index=False)
    print(f"\n[SAVED] Combined dataset: {combined_file}")
    print(f"  Total samples: {len(combined_df):,}")
    print(f"  Tumor: {(combined_df['sample_type']=='Tumor').sum():,}")
    print(f"  Normal: {(combined_df['sample_type']=='Normal').sum():,}")
    
    # Summary statistics
    print(f"\n{'='*80}")
    print("SUMMARY STATISTICS")
    print(f"{'='*80}")
    
    summary_by_cancer = combined_df.groupby(['cancer_type', 'sample_type'])['PRNP_rsem'].agg([
        ('n', 'count'),
        ('mean', 'mean'),
        ('std', 'std'),
        ('median', 'median')
    ]).round(3)
    
    print("\nPRNP Expression by Cancer Type and Sample Type:")
    print(summary_by_cancer)
    
    # By stage
    tumor_only = combined_df[combined_df['sample_type'] == 'Tumor']
    if len(tumor_only) > 0:
        summary_by_stage = tumor_only.groupby('stage_clean')['PRNP_rsem'].agg([
            ('n', 'count'),
            ('mean', 'mean'),('std', 'std')
        ]).round(3)
        
        print("\n\nPRNP Expression by Stage (Tumor only):")
        print(summary_by_stage)
    
    # Save summaries
    summary_file = OUTPUT_DIR / "tcga_prnp_real_summary.xlsx"
    with pd.ExcelWriter(summary_file, engine='openpyxl') as writer:
        summary_by_cancer.to_excel(writer, sheet_name='By_Cancer_Type')
        if len(tumor_only) > 0:
            summary_by_stage.to_excel(writer, sheet_name='By_Stage')
        combined_df.describe().to_excel(writer, sheet_name='Overall_Stats')
    
    print(f"\n[SAVED] Summary statistics: {summary_file}")
    
    # Save metadata
    download_summary['total_samples'] = int(len(combined_df))
    download_summary['tumor_samples'] = int((combined_df['sample_type']=='Tumor').sum())
    download_summary['normal_samples'] = int((combined_df['sample_type']=='Normal').sum())
    
    metadata_file = OUTPUT_DIR / "tcga_real_download_metadata.json"
    with open(metadata_file, 'w') as f:
        json.dump(download_summary, f, indent=2)
    
    print(f"[SAVED] Metadata: {metadata_file}")

# ============================================================================
# Final Summary
# ============================================================================

print(f"\n{'='*80}")
print("DOWNLOAD COMPLETE")
print(f"{'='*80}")

print(f"""
Summary:
- Cancer types attempted: {len(DATASETS)}
- Successful downloads: {len(all_datasets)}
- Failed downloads: {len(download_summary.get('errors', []))}
- Total samples: {download_summary.get('total_samples', 0):,}
  - Tumor: {download_summary.get('tumor_samples', 0):,}
  - Normal: {download_summary.get('normal_samples', 0):,}

Files created in {OUTPUT_DIR}:
- {len(all_datasets)} individual cancer type CSV files
- 1 combined dataset CSV  
- 1 summary statistics Excel file
- 1 metadata JSON file

Next steps:
1. Review downloaded data quality
2. Proceed to mRNA→Protein conversion modeling
3. Integrate with existing PrPc validation analysis
4. Update PrPc_Hybrid_Validation_Final_Report.md
""")

print("=" * 80)
