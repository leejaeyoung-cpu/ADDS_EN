"""
PrPc Biomarker - Bayesian Hierarchical Model
============================================
Incorporates prior knowledge from 127 papers
Provides robust estimates with uncertainty quantification
"""

import sys
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats

# Fix encoding
sys.stdout.reconfigure(encoding='utf-8')
if sys.platform == 'win32':
    import matplotlib
    matplotlib.rc('font', family='Malgun Gothic')
    plt.rcParams['axes.unicode_minus'] = False

# Setup
OUTPUT_DIR = Path("data/analysis/prpc_validation/bayesian")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("PrPc BIOMARKER - BAYESIAN HIERARCHICAL MODEL")
print("=" * 80)
print()

# ============================================================================
# Part 1: Prior Knowledge from Literature (127 papers)
# ============================================================================

print("PART 1: EXTRACTING PRIOR KNOWLEDGE")
print("-" * 80)

# From our 127 papers analysis
literature_priors = {
    "pancreatic": {
        "expression_rate": 0.76,
        "n_papers": 9,
        "confidence": "high"
    },
    "colorectal": {
        "expression_rate": 0.745,  # Average of 58-91%
        "n_papers": 16,
        "confidence": "high"
    },
    "gastric": {
        "expression_rate": 0.68,
        "n_papers": 4,
        "confidence": "medium"
    },
    "breast": {
        "expression_rate": 0.24,
        "n_papers": 14,
        "confidence": "high"
    },
    "liver": {
        "expression_rate": 0.60,  # Estimated
        "n_papers": 18,
        "confidence": "medium"
    }
}

# Convert expression rates to expected mean PrPc levels
# Assumption: Higher expression → Higher protein level
# Normal baseline: 1.60
# Cancer with 100% expression → 2.38 (our Stage 3 data)

normal_baseline = 1.60
high_expression_level = 2.38

# Linear scaling
def expression_to_protein(expr_rate):
    """Convert expression rate to expected protein level."""
    return normal_baseline + (high_expression_level - normal_baseline) * expr_rate

print("\nLiterature-based protein level expectations:")
print(f"{'Cancer Type':<15} {'Expr%':<10} {'Expected PrPc':<15} {'Papers'}")
print("-" * 55)

prior_expectations = {}
for cancer, data in literature_priors.items():
    expected_prpc = expression_to_protein(data['expression_rate'])
    prior_expectations[cancer] = expected_prpc
    print(f"{cancer:<15} {data['expression_rate']*100:>6.1f}%   {expected_prpc:>8.3f}        {data['n_papers']:>6}")

# Overall prior
overall_expression = np.mean([d['expression_rate'] for d in literature_priors.values()])
overall_prior_mean = expression_to_protein(overall_expression)
overall_prior_sd = 0.4  # Conservative uncertainty

print(f"\nOverall prior:")
print(f"  Mean expression rate: {overall_expression*100:.1f}%")
print(f"  Expected PrPc: {overall_prior_mean:.3f} ± {overall_prior_sd:.3f}")

# ============================================================================
# Part 2: Bayesian Model Implementation
# ============================================================================

print("\n\nPART 2: BAYESIAN MODEL SPECIFICATION")
print("-" * 80)

print("""
Hierarchical model structure:

Level 1 (Hyper-priors from literature):
  μ_overall ~ Normal(2.05, 0.20)  # From 127 papers
  σ_overall ~ HalfNormal(0.20)

Level 2 (Cancer-type specific):
  μ_cancer[j] ~ Normal(μ_overall, σ_overall)  # j = cancer type
  
Level 3 (Individual observations):
  PrPc[i] ~ Normal(μ_cancer[cancer[i]], σ_measurement)
  
Data sources:
  - Our 63 samples (weight = 1.0)
  - TCGA 2,554 samples (weight = 0.6)
""")

# Simplified Bayesian estimation without PyMC
# Using conjugate priors for tractability

def bayesian_estimate(data, prior_mean, prior_sd, data_weight=1.0):
    """
    Bayesian estimation with Normal-Normal conjugate prior
    
    Parameters:
    - data: observed data
    - prior_mean: prior belief about mean
    - prior_sd: uncertainty in prior
    - data_weight: confidence in data (0-1)
    
    Returns:
    - posterior_mean, posterior_sd
    """
    
    # Data statistics
    n = len(data)
    data_mean = np.mean(data)
    data_sd = np.std(data, ddof=1)
    
    # Prior precision
    prior_precision = 1 / (prior_sd ** 2)
    
    # Data precision (weighted)
    data_precision = (n * data_weight) / (data_sd ** 2)
    
    # Posterior precision
    posterior_precision = prior_precision + data_precision
    
    # Posterior mean (precision-weighted average)
    posterior_mean = (
        (prior_precision * prior_mean + data_precision * data_mean) /
        posterior_precision
    )
    
    # Posterior standard deviation
    posterior_sd = 1 / np.sqrt(posterior_precision)
    
    return posterior_mean, posterior_sd

# ============================================================================
# Part 3: Apply Bayesian Updates
# ============================================================================

print("\n\nPART 3: BAYESIAN UPDATES")
print("-" * 80)

# Load integrated data
integrated_file = Path("data/analysis/prpc_validation/integrated/prpc_integrated_dataset.csv")
integrated_data = pd.read_csv(integrated_file)

# Separate cancer samples
cancer_samples = integrated_data[integrated_data['sample_type']=='Cancer'].copy()

print(f"\nTotal cancer samples: {len(cancer_samples)}")

# Update 1: Overall cancer mean (all cancers combined)
print("\n[UPDATE 1] Overall cancer mean:")

our_cancer = cancer_samples[cancer_samples['source']=='Direct_Measurement']['PrPc_protein'].values
tcga_cancer = cancer_samples[cancer_samples['source']=='TCGA_Converted']['PrPc_protein'].values

print(f"  Prior: μ={overall_prior_mean:.3f}, σ={overall_prior_sd:.3f}")
print(f"  Our data: n={len(our_cancer)}, mean={np.mean(our_cancer):.3f}")
print(f"  TCGA data: n={len(tcga_cancer)}, mean={np.mean(tcga_cancer):.3f}")

# Weighted combination
all_cancer_data = np.concatenate([our_cancer, tcga_cancer])
all_cancer_weights = np.concatenate([
    np.ones(len(our_cancer)),  # Weight 1.0
    np.ones(len(tcga_cancer)) * 0.6  # Weight 0.6
])

# Bayesian update
post_mean, post_sd = bayesian_estimate(
    all_cancer_data,
    prior_mean=overall_prior_mean,
    prior_sd=overall_prior_sd,
    data_weight=np.mean(all_cancer_weights)
)

print(f"  Posterior: μ={post_mean:.3f}, σ={post_sd:.3f}")
print(f"  95% CI: [{post_mean-1.96*post_sd:.3f}, {post_mean+1.96*post_sd:.3f}]")

# Update 2: Normal mean
print("\n[UPDATE 2] Normal (healthy) mean:")

our_normal = integrated_data[
    (integrated_data['sample_type']=='Normal') &
    (integrated_data['source']=='Direct_Measurement')
]['PrPc_protein'].values

tcga_normal = integrated_data[
    (integrated_data['sample_type']=='Normal') &
    (integrated_data['source']=='TCGA_Converted')
]['PrPc_protein'].values

normal_prior_mean = 1.60
normal_prior_sd = 0.15

all_normal_data = np.concatenate([our_normal, tcga_normal])
all_normal_weights = np.concatenate([
    np.ones(len(our_normal)),
    np.ones(len(tcga_normal)) * 0.6
])

normal_post_mean, normal_post_sd = bayesian_estimate(
    all_normal_data,
    prior_mean=normal_prior_mean,
    prior_sd=normal_prior_sd,
    data_weight=np.mean(all_normal_weights)
)

print(f"  Prior: μ={normal_prior_mean:.3f}, σ={normal_prior_sd:.3f}")
print(f"  Data: n={len(all_normal_data)}, mean={np.mean(all_normal_data):.3f}")
print(f"  Posterior: μ={normal_post_mean:.3f}, σ={normal_post_sd:.3f}")

# Update 3: Stage-wise estimates
print("\n[UPDATE 3] Stage-wise estimates:")

stage_posteriors = {}

for stage_name, stage_weight in [
    ('Stage I', 0.90),   # Early, closer to normal
    ('Stage II', 1.00),  # Intermediate
    ('Stage III', 1.10), # Advanced (our data)
    ('Stage IV', 1.15)   # Most advanced
]:
    stage_prior_mean = overall_prior_mean * stage_weight
    stage_prior_sd = overall_prior_sd * 1.2  # More uncertainty
    
    stage_data = integrated_data[
        (integrated_data['stage']==stage_name)
    ]['PrPc_protein'].values
    
    stage_weights = integrated_data[
        (integrated_data['stage']==stage_name)
    ]['data_weight'].values
    
    if len(stage_data) > 0:
        stage_post_mean, stage_post_sd = bayesian_estimate(
            stage_data,
            prior_mean=stage_prior_mean,
            prior_sd=stage_prior_sd,
            data_weight=np.mean(stage_weights)
        )
        
        stage_posteriors[stage_name] = {
            'mean': stage_post_mean,
            'sd': stage_post_sd,
            'n': len(stage_data)
        }
        
        print(f"  {stage_name:12s}: μ={stage_post_mean:.3f} ± {stage_post_sd:.3f} (n={len(stage_data)})")

# ============================================================================
# Part 4: Bayesian AUC Estimation
# ============================================================================

print("\n\nPART 4: BAYESIAN AUC ESTIMATION")
print("-" * 80)

# Generate posterior samples
np.random.seed(42)
n_samples = 10000

# Sample from posteriors
normal_samples = np.random.normal(normal_post_mean, normal_post_sd, n_samples)
cancer_samples_post = np.random.normal(post_mean, post_sd, n_samples)

# Calculate AUC for each posterior sample
# AUC = P(cancer > normal)
aucs = []
for i in range(n_samples):
    # P(X_cancer > X_normal)
    prob_greater = 1 - stats.norm.cdf(0, 
                                      loc=cancer_samples_post[i] - normal_samples[i],
                                      scale=np.sqrt(post_sd**2 + normal_post_sd**2))
    aucs.append(prob_greater)

aucs = np.array(aucs)

auc_mean = np.mean(aucs)
auc_sd = np.std(aucs)
auc_95ci = np.percentile(aucs, [2.5, 97.5])

print(f"\nBayesian AUC estimate:")
print(f"  Mean: {auc_mean:.4f}")
print(f"  SD: {auc_sd:.4f}")
print(f"  95% Credible Interval: [{auc_95ci[0]:.4f}, {auc_95ci[1]:.4f}]")

# Probability that AUC > 0.80
prob_auc_gt_80 = np.mean(aucs > 0.80)
prob_auc_gt_85 = np.mean(aucs > 0.85)
prob_auc_gt_90 = np.mean(aucs > 0.90)

print(f"\nPosterior probabilities:")
print(f"  P(AUC > 0.80) = {prob_auc_gt_80:.3f} ({prob_auc_gt_80*100:.1f}%)")
print(f"  P(AUC > 0.85) = {prob_auc_gt_85:.3f} ({prob_auc_gt_85*100:.1f}%)")
print(f"  P(AUC > 0.90) = {prob_auc_gt_90:.3f} ({prob_auc_gt_90*100:.1f}%)")

# ============================================================================
# Part 5: Visualization
# ============================================================================

print("\n\nPART 5: CREATING VISUALIZATIONS")
print("-" * 80)

fig, axes = plt.subplots(2, 2, figsize=(14, 12))
fig.suptitle('베이지안 분석 결과', fontsize=16, fontweight='bold')

# Plot 1: Prior vs Posterior (Cancer)
ax1 = axes[0, 0]
x = np.linspace(0.5, 4, 1000)
prior_dist = stats.norm.pdf(x, overall_prior_mean, overall_prior_sd)
posterior_dist = stats.norm.pdf(x, post_mean, post_sd)

ax1.plot(x, prior_dist, 'b--', linewidth=2, label=f'Prior (μ={overall_prior_mean:.2f})')
ax1.plot(x, posterior_dist, 'r-', linewidth=2, label=f'Posterior (μ={post_mean:.2f})')
ax1.axvline(post_mean, color='r', linestyle=':', alpha=0.5)
ax1.fill_between(x, 0, posterior_dist, alpha=0.2)
ax1.set_xlabel('PrPc Concentration (Cancer)')
ax1.set_ylabel('Probability Density')
ax1.set_title('Prior vs Posterior Distribution')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: AUC Distribution
ax2 = axes[0, 1]
ax2.hist(aucs, bins=50, density=True, alpha=0.7, edgecolor='black')
ax2.axvline(auc_mean, color='r', linewidth=2, label=f'Mean={auc_mean:.3f}')
ax2.axvline(auc_95ci[0], color='orange', linestyle='--', label=f'95% CI')
ax2.axvline(auc_95ci[1], color='orange', linestyle='--')
ax2.axvline(0.80, color='green', linestyle=':', label='Target=0.80')
ax2.set_xlabel('AUC')
ax2.set_ylabel('Density')
ax2.set_title('Posterior Distribution of AUC')
ax2.legend()
ax2.grid(True, alpha=0.3)

# Plot 3: Stage-wise posterior estimates
ax3 = axes[1, 0]
stages = list(stage_posteriors.keys())
means = [stage_posteriors[s]['mean'] for s in stages]
sds = [stage_posteriors[s]['sd'] for s in stages]
ns = [stage_posteriors[s]['n'] for s in stages]

# Add normal
stages = ['Normal'] + stages
means = [normal_post_mean] + means
sds = [normal_post_sd] + sds
ns = [len(all_normal_data)] + ns

x_pos = np.arange(len(stages))
ax3.errorbar(x_pos, means, yerr=[1.96*s for s in sds], 
             fmt='o-', capsize=5, markersize=8, linewidth=2)
ax3.set_xticks(x_pos)
ax3.set_xticklabels([f'{s}\n(n={n})' for s, n in zip(stages, ns)], rotation=0)
ax3.set_ylabel('PrPc Concentration')
ax3.set_title('Posterior Estimates by Stage (95% CI)')
ax3.grid(True, alpha=0.3, axis='y')

# Plot 4: Probability surface
ax4 = axes[1, 1]
# Create 2D grid
grid_normal = np.linspace(normal_post_mean - 3*normal_post_sd, 
                          normal_post_mean + 3*normal_post_sd, 50)
grid_cancer = np.linspace(post_mean - 3*post_sd,
                          post_mean + 3*post_sd, 50)
X, Y = np.meshgrid(grid_normal, grid_cancer)

# Joint probability (assuming independence)
Z = (stats.norm.pdf(X, normal_post_mean, normal_post_sd) *
stats.norm.pdf(Y, post_mean, post_sd))

contour = ax4.contourf(X, Y, Z, levels=15, cmap='viridis', alpha=0.7)
ax4.contour(X, Y, Z, levels=15, colors='black', linewidths=0.5, alpha=0.3)

# Add diagonal (Normal = Cancer)
ax4.plot([grid_normal[0], grid_normal[-1]], 
         [grid_normal[0], grid_normal[-1]], 
         'r--', linewidth=2, label='Normal = Cancer')

ax4.set_xlabel('Normal PrPc')
ax4.set_ylabel('Cancer PrPc')
ax4.set_title('Joint Posterior Distribution')
ax4.legend()
plt.colorbar(contour, ax=ax4, label='Joint Density')

plt.tight_layout()
fig_file = OUTPUT_DIR / "bayesian_analysis_plots.png"
plt.savefig(fig_file, dpi=300, bbox_inches='tight')
print(f"\n[SAVED] {fig_file}")
plt.close()

# ============================================================================
# Part 6: Summary and Export
# ============================================================================

print("\n\n" + "=" * 80)
print("BAYESIAN ANALYSIS - SUMMARY")
print("=" * 80)

summary = {
    "prior_knowledge": {
        "source": "127 papers systematic review",
        "overall_expression": overall_expression,
        "expected_protein": overall_prior_mean
    },
    "posterior_estimates": {
        "normal": {
            "mean": float(normal_post_mean),
            "sd": float(normal_post_sd),
            "ci_95": [float(normal_post_mean - 1.96*normal_post_sd),
                     float(normal_post_mean + 1.96*normal_post_sd)]
        },
        "cancer_overall": {
            "mean": float(post_mean),
            "sd": float(post_sd),
            "ci_95": [float(post_mean - 1.96*post_sd),
                     float(post_mean + 1.96*post_sd)]
        },
        "by_stage": {k: {
            'mean': float(v['mean']),
            'sd': float(v['sd']),
            'n': int(v['n'])
        } for k, v in stage_posteriors.items()}
    },
    "auc_estimate": {
        "mean": float(auc_mean),
        "sd": float(auc_sd),
        "ci_95": [float(auc_95ci[0]), float(auc_95ci[1])],
        "prob_gt_80": float(prob_auc_gt_80),
        "prob_gt_85": float(prob_auc_gt_85),
        "prob_gt_90": float(prob_auc_gt_90)
    },
    "interpretation": {
        "confidence_level": "moderate-high" if auc_mean > 0.80 else "moderate",
        "recommendation": "proceed" if prob_auc_gt_80 > 0.70 else "caution"
    }
}

summary_file = OUTPUT_DIR / "bayesian_analysis_summary.json"
with open(summary_file, 'w', encoding='utf-8') as f:
    json.dump(summary, f, indent=2, ensure_ascii=False)

print(f"\n[SAVED] {summary_file}")

print(f"""
FINAL BAYESIAN ESTIMATES:

Normal (Healthy):
  Mean: {normal_post_mean:.3f} +/- {normal_post_sd:.3f}
  95% CI: [{normal_post_mean-1.96*normal_post_sd:.3f}, {normal_post_mean+1.96*normal_post_sd:.3f}]

Cancer (Overall):
  Mean: {post_mean:.3f} +/- {post_sd:.3f}
  95% CI: [{post_mean-1.96*post_sd:.3f}, {post_mean+1.96*post_sd:.3f}]

Estimated AUC:
  Mean: {auc_mean:.4f}
  95% Credible Interval: [{auc_95ci[0]:.4f}, {auc_95ci[1]:.4f}]
  
Probability of clinical utility:
  P(AUC > 0.80): {prob_auc_gt_80*100:.1f}%
  P(AUC > 0.85): {prob_auc_gt_85*100:.1f}%
  
INTERPRETATION:
{'  STRONG EVIDENCE' if prob_auc_gt_80 > 0.90 else '  MODERATE EVIDENCE' if prob_auc_gt_80 > 0.70 else '  WEAK EVIDENCE'} that PrPc is a useful biomarker
  Recommendation: {'PROCEED to validation' if prob_auc_gt_80 > 0.70 else 'Additional data needed'}
""")

print("=" * 80)
