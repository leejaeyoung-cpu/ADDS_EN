#!/usr/bin/env python3
"""
Feature Ablation Study
Tests which feature groups are critical for performance
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
import json

# Configuration
DATA_DIR = Path("data/analysis/prpc_validation")
RESULTS_DIR = Path("results/ablation")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("Feature Ablation Study")
print("=" * 80)

# Load data
features_file = DATA_DIR / "features/feature_matrix_full.csv"
targets_file = DATA_DIR / "features/targets.csv"

features = pd.read_csv(features_file)
targets = pd.read_csv(targets_file)

X = features.values
y = targets['prnp_high'].values
feature_names = features.columns.tolist()

print(f"\nData loaded:")
print(f"   Samples: {len(X)}")
print(f"   Features: {len(feature_names)}")

# Categorize features
def categorize_feature(name):
    if name.startswith('PRNP'):
        return 'PRNP_features'
    elif name.startswith('cancer_'):
        return 'Cancer_type'
    elif name.startswith('source_'):
        return 'Data_source'
    elif name.startswith('pathway_'):
        return 'Pathway'
    elif '_expr' in name:
        return 'Gene_expression'
    elif '_ratio' in name:
        return 'Gene_ratio'
    elif name.startswith('PC'):
        return 'PCA'
    elif 'is_tumor' in name:
        return 'Sample_type'
    else:
        return 'Other'

# Group features by category
feature_categories = {name: categorize_feature(name) for name in feature_names}
categories = sorted(set(feature_categories.values()))

print(f"\nFeature categories: {categories}")

# ==============================================================================
# 1. Baseline Performance (All Features)
# ==============================================================================

print("\n" + "=" * 80)
print("1. Baseline Performance (All Features)")
print("=" * 80)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model = RandomForestClassifier(n_estimators=100, random_state=42)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
baseline_scores = cross_val_score(model, X_scaled, y, cv=cv, scoring='roc_auc')
baseline_mean = baseline_scores.mean()
baseline_std = baseline_scores.std()

print(f"\nBaseline (all {len(feature_names)} features):")
print(f"   Mean AUC: {baseline_mean:.4f} ± {baseline_std:.4f}")
print(f"   Fold scores: {[f'{s:.4f}' for s in baseline_scores]}")

# ==============================================================================
# 2. Leave-One-Group-Out Ablation
# ==============================================================================

print("\n" + "=" * 80)
print("2. Leave-One-Group-Out Ablation")
print("=" * 80)

ablation_results = {}

for category in categories:
    # Remove this category
    mask = [feature_categories[name] != category for name in feature_names]
    X_ablated = X_scaled[:, mask]
    
    n_removed = sum([not m for m in mask])
    n_remaining = sum(mask)
    
    # Cross-validate
    scores = cross_val_score(model, X_ablated, y, cv=cv, scoring='roc_auc')
    mean_score = scores.mean()
    std_score = scores.std()
    
    # Calculate performance drop
    drop = baseline_mean - mean_score
    drop_pct = (drop / baseline_mean) * 100
    
    ablation_results[category] = {
        'mean_auc': mean_score,
        'std_auc': std_score,
        'drop_from_baseline': drop,
        'drop_percentage': drop_pct,
        'n_features_removed': n_removed,
        'n_features_remaining': n_remaining
    }
    
    print(f"\nWithout {category}:")
    print(f"   Features removed: {n_removed}")
    print(f"   Mean AUC: {mean_score:.4f} ± {std_score:.4f}")
    print(f"   Drop from baseline: {drop:.4f} ({drop_pct:.2f}%)")

# ==============================================================================
# 3. Use-Only-Group Ablation
# ==============================================================================

print("\n" + "=" * 80)
print("3. Use-Only-Group Ablation")
print("=" * 80)

only_results = {}

for category in categories:
    # Use only this category
    mask = [feature_categories[name] == category for name in feature_names]
    X_only = X_scaled[:, mask]
    
    n_used = sum(mask)
    
    if n_used == 0:
        print(f"\n{category}: No features in this category")
        continue
    
    # Cross-validate
    scores = cross_val_score(model, X_only, y, cv=cv, scoring='roc_auc')
    mean_score = scores.mean()
    std_score = scores.std()
    
    # Calculate vs baseline
    diff = mean_score - baseline_mean
    diff_pct = (diff / baseline_mean) * 100
    
    only_results[category] = {
        'mean_auc': mean_score,
        'std_auc': std_score,
        'diff_from_baseline': diff,
        'diff_percentage': diff_pct,
        'n_features_used': n_used
    }
    
    print(f"\nOnly {category}:")
    print(f"   Features used: {n_used}")
    print(f"   Mean AUC: {mean_score:.4f} ± {std_score:.4f}")
    print(f"   vs Baseline: {diff:+.4f} ({diff_pct:+.2f}%)")

# ==============================================================================
# 4. Top-N Feature Ablation
# ==============================================================================

print("\n" + "=" * 80)
print("4. Top-N Feature Ablation")
print("=" * 80)

# Use RF feature importance to rank
model_full = RandomForestClassifier(n_estimators=100, random_state=42)
model_full.fit(X_scaled, y)
importance = model_full.feature_importances_

# Sort features by importance
sorted_idx = np.argsort(importance)[::-1]

# Test with different numbers of top features
n_features_to_test = [1, 2, 3, 5, 10, 20, 30, 50, 96]
topn_results = {}

for n in n_features_to_test:
    if n > len(feature_names):
        continue
    
    # Use top N features
    top_idx = sorted_idx[:n]
    X_topn = X_scaled[:, top_idx]
    
    # Cross-validate
    scores = cross_val_score(model, X_topn, y, cv=cv, scoring='roc_auc')
    mean_score = scores.mean()
    std_score = scores.std()
    
    topn_results[n] = {
        'mean_auc': mean_score,
        'std_auc': std_score,
        'top_features': [feature_names[i] for i in top_idx]
    }
    
    print(f"\nTop {n} features:")
    print(f"   Mean AUC: {mean_score:.4f} ± {std_score:.4f}")
    if n <= 10:
        print(f"   Features: {topn_results[n]['top_features']}")

# ==============================================================================
# 5. Save Results
# ==============================================================================

print("\n" + "=" * 80)
print("5. Saving Results")
print("=" * 80)

all_results = {
    'baseline': {
        'mean_auc': baseline_mean,
        'std_auc': baseline_std,
        'n_features': len(feature_names)
    },
    'leave_one_out': ablation_results,
    'use_only': only_results,
    'top_n': topn_results
}

results_file = RESULTS_DIR / 'ablation_results.json'
with open(results_file, 'w') as f:
    json.dump(all_results, f, indent=2, default=str)
print(f"   Results: {results_file}")

# ==============================================================================
# 6. Visualizations
# ==============================================================================

print("\n" + "=" * 80)
print("6. Creating Visualizations")
print("=" * 80)

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# 6.1 Leave-one-out impact
cats = list(ablation_results.keys())
drops = [ablation_results[c]['drop_percentage'] for c in cats]
colors = ['red' if d > 5 else 'orange' if d > 1 else 'green' for d in drops]

axes[0, 0].barh(cats, drops, color=colors, alpha=0.7, edgecolor='black')
axes[0, 0].set_xlabel('Performance Drop (%)', fontsize=12, fontweight='bold')
axes[0, 0].set_title('A. Impact of Removing Each Feature Group', fontsize=14, fontweight='bold')
axes[0, 0].axvline(1, color='orange', linestyle='--', label='1% threshold')
axes[0, 0].axvline(5, color='red', linestyle='--', label='5% threshold')
axes[0, 0].legend()
axes[0, 0].grid(axis='x', alpha=0.3)
axes[0, 0].invert_yaxis()

# 6.2 Use-only performance
only_cats = list(only_results.keys())
only_aucs = [only_results[c]['mean_auc'] for c in only_cats]

axes[0, 1].barh(only_cats, only_aucs, color='#3498db', alpha=0.7, edgecolor='black')
axes[0, 1].axvline(baseline_mean, color='red', linestyle='--', linewidth=2, label='Baseline')
axes[0, 1].axvline(0.75, color='green', linestyle='--', label='Target (0.75)')
axes[0, 1].set_xlabel('AUC', fontsize=12, fontweight='bold')
axes[0, 1].set_title('B. Performance Using Only Each Group', fontsize=14, fontweight='bold')
axes[0, 1].legend()
axes[0, 1].grid(axis='x', alpha=0.3)
axes[0, 1].invert_yaxis()
axes[0, 1].set_xlim([0, 1.05])

# 6.3 Top-N curves
ns = list(topn_results.keys())
aucs = [topn_results[n]['mean_auc'] for n in ns]
stds = [topn_results[n]['std_auc'] for n in ns]

axes[1, 0].plot(ns, aucs, 'o-', linewidth=2, markersize=8, color='#2ecc71')
axes[1, 0].fill_between(ns, 
                        [a - s for a, s in zip(aucs, stds)],
                        [a + s for a, s in zip(aucs, stds)],
                        alpha=0.3, color='#2ecc71')
axes[1, 0].axhline(baseline_mean, color='red', linestyle='--', label='Baseline (all features)')
axes[1, 0].axhline(0.75, color='orange', linestyle='--', label='Target (0.75)')
axes[1, 0].set_xlabel('Number of Top Features', fontsize=12, fontweight='bold')
axes[1, 0].set_ylabel('Cross-Val AUC', fontsize=12, fontweight='bold')
axes[1, 0].set_title('C. Performance vs Number of Features', fontsize=14, fontweight='bold')
axes[1, 0].legend()
axes[1, 0].grid(alpha=0.3)
axes[1, 0].set_ylim([0.5, 1.05])

# 6.4 Critical features summary
critical_info = []
for cat, res in ablation_results.items():
    if res['drop_percentage'] > 1:  # >1% drop = critical
        critical_info.append((cat, res['drop_percentage'], res['n_features_removed']))

critical_info.sort(key=lambda x: x[1], reverse=True)

if critical_info:
    cats, drops, counts = zip(*critical_info)
    
    axes[1, 1].barh(range(len(cats)), drops, color='#e74c3c', alpha=0.7, edgecolor='black')
    axes[1, 1].set_yticks(range(len(cats)))
    axes[1, 1].set_yticklabels([f"{c} (n={n})" for c, _, n in critical_info])
    axes[1, 1].set_xlabel('Performance Drop (%)', fontsize=12, fontweight='bold')
    axes[1, 1].set_title('D. Critical Feature Groups (>1% impact)', fontsize=14, fontweight='bold')
    axes[1, 1].grid(axis='x', alpha=0.3)
    axes[1, 1].invert_yaxis()
else:
    axes[1, 1].text(0.5, 0.5, 'No feature groups have >1% impact', 
                   ha='center', va='center', fontsize=14)
    axes[1, 1].axis('off')

plt.tight_layout()
fig_file = RESULTS_DIR / 'ablation_analysis.png'
plt.savefig(fig_file, dpi=300, bbox_inches='tight')
print(f"   Figure: {fig_file}")
plt.close()

# ==============================================================================
# 7. Summary
# ==============================================================================

print("\n" + "=" * 80)
print("✅ ABLATION STUDY COMPLETE!")
print("=" * 80)

print(f"\n🔍 Key Findings:")

# Most critical group
most_critical = max(ablation_results.items(), key=lambda x: x[1]['drop_percentage'])
print(f"\n   Most critical group: {most_critical[0]}")
print(f"   Drop when removed: {most_critical[1]['drop_percentage']:.2f}%")

# Minimal feature set
for n in [1, 5, 10]:
    if n in topn_results:
        print(f"\n   Top {n} features → AUC {topn_results[n]['mean_auc']:.4f}")

# Best standalone group
if only_results:
    best_standalone = max(only_results.items(), key=lambda x: x[1]['mean_auc'])
    print(f"\n   Best standalone group: {best_standalone[0]}")
    print(f"   AUC with only this group: {best_standalone[1]['mean_auc']:.4f}")

print(f"\n📁 All results saved to: {RESULTS_DIR}")
