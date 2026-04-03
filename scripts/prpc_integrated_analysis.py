"""
PrPc Biomarker - Data Integration and Analysis
==============================================
Combines:
1. Our 63 serum protein samples
2. TCGA 2,551 mRNA samples (simulated)
3. mRNA→Protein conversion model
4. Bayesian integration
"""

import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats
from sklearn.metrics import roc_curve, auc
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score

# Fix encoding
sys.stdout.reconfigure(encoding='utf-8')
if sys.platform == 'win32':
    import matplotlib
    matplotlib.rc('font', family='Malgun Gothic')
    plt.rcParams['axes.unicode_minus'] = False

# Setup
OUTPUT_DIR = Path("data/analysis/prpc_validation/integrated")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("PrPc BIOMARKER - INTEGRATED DATA ANALYSIS")
print("=" * 80)
print()

# ============================================================================
# Part 1: Load All Data
# ============================================================================

print("PART 1: LOADING DATA")
print("-" * 80)

# Our serum data (n=63)
print("\nLoading our serum protein data...")
our_normal = np.array([1.4275, 1.2995, 1.4333, 1.5962, 1.2181, 1.8288, 
                       1.5671, 1.8579, 1.3984, 1.6136, 1.4624, 1.4798,
                       1.8405, 1.5438, 1.759, 1.6377, 1.7005, 1.7633,
                       1.8888, 1.6796, 1.6168])

our_patient = np.array([2.1817, 2.3019, 3.7189, 3.1252, 2.8307, 2.8631,
                        2.8968, 2.1396, 2.7406, 1.9232, 1.9833, 2.0494,
                        2.1937, 2.0014, 2.9833, 2.2237, 2.1216, 2.6348,
                        2.0819, 2.1156, 1.9893, 2.4641, 2.1696, 2.422,
                        2.0014, 2.8788, 2.1216, 1.9112, 2.993, 2.6264,
                        1.9516, 2.3282, 2.1399, 2.4328, 2.0981, 2.4956,
                        2.2445, 2.0771, 2.4956, 2.1399, 2.2236, 2.826])

our_data = pd.DataFrame({
    'sample_id': [f'SERUM_N_{i:02d}' for i in range(len(our_normal))] +
                 [f'SERUM_P_{i:02d}' for i in range(len(our_patient))],
    'sample_type': ['Normal']*len(our_normal) + ['Cancer']*len(our_patient),
    'PrPc_protein': np.concatenate([our_normal, our_patient]),
    'source': 'Direct_Measurement',
    'data_weight': 1.0  # Highest confidence
})

print(f"  Our data: n={len(our_data)} (21 normal, 42 cancer)")

# TCGA data (real data)
print("\nLoading TCGA PRNP data...")

# Try to load real data first, fall back to simulated if not available
real_tcga_file = Path("data/analysis/prpc_validation/open_data/real/tcga_all_cancers_prnp_real.csv")
simulated_tcga_file = Path("data/analysis/prpc_validation/open_data/tcga_all_cancers_prnp_simulated.csv")

if real_tcga_file.exists():
    print(f"  ✓ Using REAL TCGA data: {real_tcga_file}")
    tcga_data = pd.read_csv(real_tcga_file)
    
    # Prepare TCGA data: rename columns to match expected format
    # Real TCGA has: PRNP_rsem, PRNP_log2_fpkm
    # We need: PRNP_linear, PRNP_log2
    if 'PRNP_rsem' in tcga_data.columns:
        tcga_data['PRNP_linear'] = tcga_data['PRNP_rsem']
        # Convert FPKM to log2 scale if not already
        if 'PRNP_log2_fpkm' in tcga_data.columns:
            tcga_data['PRNP_log2'] = tcga_data['PRNP_log2_fpkm']
        else:
            tcga_data['PRNP_log2'] = np.log2(tcga_data['PRNP_rsem'] + 1)
    
    # Standardize sample_type: 'Tumor' -> 'Tumor', keep 'Normal'
    if 'sample_type_clean' in tcga_data.columns:
        tcga_data['sample_type'] = tcga_data['sample_type_clean']
    
    # Add stage_clean if exists, otherwise use 'Unknown'
    if 'stage_clean' not in tcga_data.columns:
        tcga_data['stage_clean'] = 'Unknown'
    tcga_data['stage'] = tcga_data['stage_clean']
    
    print(f"  Using REAL TCGA RNA-seq data (PRNP expression)")
elif simulated_tcga_file.exists():
    print(f"  ! Real data not found, using SIMULATED data: {simulated_tcga_file}")
    tcga_data = pd.read_csv(simulated_tcga_file)
    print(f"  WARNING: This is simulated data. Download real data for final analysis!")
else:
    raise FileNotFoundError("No TCGA data found (neither real nor simulated)")

print(f"  TCGA data: n={len(tcga_data)}")
print(f"    Tumor: {(tcga_data['sample_type']=='Tumor').sum()}")
print(f"    Normal: {(tcga_data['sample_type']=='Normal').sum()}")

# ============================================================================
# Part 2: mRNA→Protein Conversion Model
# ============================================================================

print("\n\nPART 2: mRNA-TO-PROTEIN CONVERSION MODEL")
print("-" * 80)

print("""
Challenge: TCGA has mRNA (PRNP gene expression)  
          We need protein (PrPc) to match our serum data
          
Solution: Build conversion model using correlation assumption
""")

# For simulation, we'll create a conversion model
# In reality, you'd measure both mRNA and protein in calibration samples

print("\nBuilding conversion model...")

# Assume log-linear relationship
# PrPc_protein ≈ α + β × log2(PRNP_mRNA) + noise

# Calibration: Use literature-based correlation
# Typical mRNA-protein correlation: r ≈ 0.4-0.7 for most genes
# We'll use r = 0.60 (moderate correlation)

correlation_coefficient = 0.60

# Our protein data stats
protein_mean = our_data['PrPc_protein'].mean()
protein_std = our_data['PrPc_protein'].std()

print(f"\nOur protein data:")
print(f"  Mean: {protein_mean:.3f}")
print(f"  SD: {protein_std:.3f}")

# TCGA mRNA stats (tumor only for comparison)
tcga_tumor = tcga_data[tcga_data['sample_type']=='Tumor'].copy()
mrna_mean = tcga_tumor['PRNP_log2'].mean()
mrna_std = tcga_tumor['PRNP_log2'].std()

print(f"\nTCGA mRNA data (tumor):")
print(f"  Mean (log2): {mrna_mean:.3f}")
print(f"  SD (log2): {mrna_std:.3f}")

# Conversion formula with specified correlation
# protein = α + β × mRNA + ε
# where β is chosen to achieve desired correlation

beta = correlation_coefficient * (protein_std / mrna_std)
alpha = protein_mean - beta * mrna_mean

print(f"\nConversion model:")
print(f"  PrPc_protein = {alpha:.3f} + {beta:.3f} × PRNP_log2 + noise")
print(f"  Target correlation: r = {correlation_coefficient:.2f}")

# Apply conversion to TCGA data
tcga_data['PrPc_protein_predicted'] = (
    alpha + beta * tcga_data['PRNP_log2'] +
    np.random.normal(0, protein_std * np.sqrt(1 - correlation_coefficient**2), len(tcga_data))
)

# Ensure realistic range
tcga_data['PrPc_protein_predicted'] = np.clip(
    tcga_data['PrPc_protein_predicted'],
    0.5,  # Minimum
    5.0   # Maximum
)

# Calculate prediction uncertainty
tcga_data['prediction_se'] = protein_std * np.sqrt(1 - correlation_coefficient**2)
tcga_data['source'] = 'TCGA_Converted'
tcga_data['data_weight'] = correlation_coefficient  # Weight by correlation

print(f"\nTCGA protein predictions:")
print(f"  Median: {tcga_data['PrPc_protein_predicted'].median():.3f}")
print(f"  Range: [{tcga_data['PrPc_protein_predicted'].min():.3f}, {tcga_data['PrPc_protein_predicted'].max():.3f}]")

# ============================================================================
# Part 3: Data Integration with Weighting
# ============================================================================

print("\n\nPART 3: WEIGHTED DATA INTEGRATION")
print("-" * 80)

# Prepare TCGA data for integration
tcga_for_integration = tcga_data[['sample_id', 'sample_type', 'PrPc_protein_predicted', 
                                   'source', 'data_weight', 'stage', 'cancer_type']].copy()
tcga_for_integration.rename(columns={'PrPc_protein_predicted': 'PrPc_protein'}, inplace=True)
tcga_for_integration['stage'] = tcga_for_integration['stage'].fillna('Unknown')
tcga_for_integration['cancer_type'] = tcga_for_integration['cancer_type'].fillna('Unknown')

# Add missing columns to our data
our_data['stage'] = ['Normal']*21 + ['Stage III']*42  # Our data was stage 3
our_data['cancer_type'] = ['Normal']*21 + ['Mixed']*42

# Combine
integrated_data = pd.concat([
    our_data,
    tcga_for_integration
], ignore_index=True)

# Recode sample_type for consistency
integrated_data['sample_type'] = integrated_data['sample_type'].replace({
    'Tumor': 'Cancer',
    'Normal': 'Normal'
})

print(f"\nIntegrated dataset:")
print(f"  Total samples: {len(integrated_data)}")
print(f"  Normal: {(integrated_data['sample_type']=='Normal').sum()}")
print(f"  Cancer: {(integrated_data['sample_type']=='Cancer').sum()}")
print(f"\nBy source:")
print(integrated_data.groupby('source')['sample_id'].count())
print(f"\nBy weight:")
print(integrated_data.groupby('data_weight')['sample_id'].count())

# Calculate  effective sample size
n_effective = integrated_data.groupby('source').apply(
    lambda x: (x['data_weight'].iloc[0] * len(x))
).sum()

print(f"\n Effective sample size:")
print(f"  Raw N = {len(integrated_data)}")
print(f"  Weighted N_eff = {n_effective:.0f}")

# Save integrated data
integrated_file = OUTPUT_DIR / "prpc_integrated_dataset.csv"
integrated_data.to_csv(integrated_file, index=False)
print(f"\n[SAVED] {integrated_file}")

# ============================================================================
# Part 4: Weighted ROC Analysis
# ============================================================================

print("\n\nPART 4: WEIGHTED ROC ANALYSIS")
print("-" * 80)

def weighted_roc_auc(data, weight_col='data_weight'):
    """Calculate weighted ROC AUC."""
    
    # Separate normal and cancer
    normal = data[data['sample_type']=='Normal'].copy()
    cancer = data[data['sample_type']=='Cancer'].copy()
    
    # Prepare for ROC
    y_true = np.concatenate([
        np.zeros(len(normal)),
        np.ones(len(cancer))
    ])
    
    y_score = np.concatenate([
        normal['PrPc_protein'].values,
        cancer['PrPc_protein'].values
    ])
    
    sample_weight = np.concatenate([
        normal[weight_col].values,
        cancer[weight_col].values
    ])
    
    # Calculate ROC
    fpr, tpr, thresholds = roc_curve(y_true, y_score, sample_weight=sample_weight)
    roc_auc = auc(fpr, tpr)
    
    return fpr, tpr, thresholds, roc_auc

# Analysis 1: Our data only
our_only = integrated_data[integrated_data['source']=='Direct_Measurement'].copy()
fpr1, tpr1, _, auc1 = weighted_roc_auc(our_only)

print(f"\nAnalysis 1 - Our data only (n={len(our_only)}):")
print(f"  AUC = {auc1:.4f}")

# Analysis 2: Integrated (weighted)
fpr2, tpr2, thresh2, auc2 = weighted_roc_auc(integrated_data)

print(f"\nAnalysis 2 - Integrated weighted (n_eff={n_effective:.0f}):")
print(f"  AUC = {auc2:.4f}")

# Analysis 3: By stage (TCGA data)
tcga_tumor_only = integrated_data[
    (integrated_data['source']=='TCGA_Converted') &
    (integrated_data['sample_type']=='Cancer')
].copy()

print(f"\nAnalysis 3 - By stage (TCGA cancer samples):")
for stage in ['Stage I', 'Stage II', 'Stage III', 'Stage IV']:
    stage_data = tcga_tumor_only[tcga_tumor_only['stage']==stage]
    if len(stage_data) > 0:
        mean_val = stage_data['PrPc_protein'].mean()
        print(f"  {stage:12s}: n={len(stage_data):4d}, mean={mean_val:.3f}")

# ============================================================================
# Part 5: Visualizations
# ============================================================================

print("\n\nPART 5: CREATING VISUALIZATIONS")
print("-" * 80)

fig, axes = plt.subplots(2, 2, figsize=(14, 12))
fig.suptitle('PrPc 통합 데이터 분석 (우리 데이터 + TCGA)', fontsize=16, fontweight='bold')

# Plot 1: ROC comparison
ax1 = axes[0, 0]
ax1.plot(fpr1, tpr1, 'b-', linewidth=2, label=f'우리 데이터만 (AUC={auc1:.3f})')
ax1.plot(fpr2, tpr2, 'r-', linewidth=2, label=f'통합 데이터 (AUC={auc2:.3f})')
ax1.plot([0,1], [0,1], 'k--', alpha=0.3)
ax1.set_xlabel('False Positive Rate')
ax1.set_ylabel('True Positive Rate')
ax1.set_title('ROC Curve Comparison')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: Distribution by source
ax2 = axes[0, 1]
sources = ['Direct_Measurement', 'TCGA_Converted']
for i, source in enumerate(sources):
    subset = integrated_data[integrated_data['source']==source]
    normal_vals = subset[subset['sample_type']=='Normal']['PrPc_protein']
    cancer_vals = subset[subset['sample_type']=='Cancer']['PrPc_protein']
    
    offset = i * 0.4
    ax2.violinplot([normal_vals, cancer_vals], 
                   positions=[1+offset, 2+offset],
                   widths=0.3,
                   showmeans=True)

ax2.set_xticks([1.2, 2.2])
ax2.set_xticklabels(['Normal', 'Cancer'])
ax2.set_ylabel('PrPc Concentration')
ax2.set_title('Distribution by Data Source')
ax2.grid(True, alpha=0.3, axis='y')

# Plot 3: Stage-wise trend
ax3 = axes[1, 0]
stage_order = ['Normal', 'Stage I', 'Stage II', 'Stage III', 'Stage IV']
stage_data_for_plot = []
stage_labels = []

for stage in stage_order:
    if stage == 'Normal':
        vals = integrated_data[integrated_data['sample_type']=='Normal']['PrPc_protein']
    else:
        vals = integrated_data[
            (integrated_data['sample_type']=='Cancer') &
            (integrated_data['stage']==stage)
        ]['PrPc_protein']
    
    if len(vals) > 0:
        stage_data_for_plot.append(vals)
        stage_labels.append(f'{stage}\n(n={len(vals)})')

bp = ax3.boxplot(stage_data_for_plot, labels=stage_labels, patch_artist=True)
for patch in bp['boxes']:
    patch.set_facecolor('lightblue')
ax3.set_ylabel('PrPc Concentration')
ax3.set_title('PrPc by Disease Stage (Integrated Data)')
ax3.grid(True, alpha=0.3, axis='y')

# Plot 4: Sample composition
ax4 = axes[1, 1]
composition = integrated_data.groupby(['source', 'sample_type']).size().unstack(fill_value=0)
composition.plot(kind='bar', stacked=True, ax=ax4, color=['lightblue', 'lightcoral'])
ax4.set_xlabel('Data Source')
ax4.set_ylabel('Number of Samples')
ax4.set_title('Dataset Composition')
ax4.legend(title='Sample Type')
ax4.set_xticklabels(ax4.get_xticklabels(), rotation=45, ha='right')
ax4.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
fig_file = OUTPUT_DIR / "integrated_analysis_plots.png"
plt.savefig(fig_file, dpi=300, bbox_inches='tight')
print(f"\n[SAVED] {fig_file}")
plt.close()

# ============================================================================
# Part 6: Summary Report
# ============================================================================

print("\n\n" + "=" * 80)
print("INTEGRATED ANALYSIS - SUMMARY")
print("=" * 80)

summary = {
    "data_sources": {
        "direct_measurement": {
            "n": len(our_data),
            "weight": 1.0,
            "auc": float(auc1)
        },
        "tcga_converted": {
            "n": len(tcga_data),
            "weight": correlation_coefficient,
            "auc": float(auc2)
        }
    },
    "integrated_analysis": {
        "total_samples": len(integrated_data),
        "effective_n": float(n_effective),
        "auc_weighted": float(auc2),
        "improvement_vs_direct": f"{((auc2-auc1)/auc1*100):.1f}%"
    },
    "by_stage": {},
    "conversion_model": {
        "alpha": float(alpha),
        "beta": float(beta),
        "correlation": correlation_coefficient,
        "formula": f"PrPc_protein = {alpha:.3f} + {beta:.3f} × PRNP_log2"
    }
}

# Add stage statistics
for stage in ['Normal', 'Stage I', 'Stage II', 'Stage III', 'Stage IV']:
    if stage == 'Normal':
        stage_samples = integrated_data[integrated_data['sample_type']=='Normal']
    else:
        stage_samples = integrated_data[
            (integrated_data['sample_type']=='Cancer') &
            (integrated_data['stage']==stage)
        ]
    
    if len(stage_samples) > 0:
        summary['by_stage'][stage] = {
            "n": len(stage_samples),
            "mean": float(stage_samples['PrPc_protein'].mean()),
            "sd": float(stage_samples['PrPc_protein'].std())
        }

# Save summary
summary_file = OUTPUT_DIR / "integrated_analysis_summary.json"
import json
with open(summary_file, 'w', encoding='utf-8') as f:
    json.dump(summary, f, indent=2, ensure_ascii=False)

print(f"\n{summary_file.name} saved")

print(f"""
KEY FINDINGS:

1. Data Integration:
   - Our data: n={len(our_data)} (weight=1.0)
   - TCGA data: n={len(tcga_data)} (weight={correlation_coefficient})
   - Effective N: {n_effective:.0f} (vs original 63)
   - Increase: {n_effective/63:.1f}x

2. Performance:
   - Our data AUC: {auc1:.4f}
   - Integrated AUC: {auc2:.4f}
   - Change: {(auc2-auc1):.4f}

3. Stage-wise Pattern:
   {chr(10).join([f'   {k}: mean={v["mean"]:.3f} (n={v["n"]})' for k,v in summary['by_stage'].items()])}

4. Next Steps:
   - Download real TCGA data (replace simulated)
   - Add GEO datasets
   - Add CPTAC protein data
   - Bayesian modeling for uncertainty
   - Prepare manuscript

Files created:
- {integrated_file.name}
- {fig_file.name}
- {summary_file.name}
""")

print("=" * 80)
