#!/usr/bin/env python3
"""
Cross-Validation Analysis on TCGA Data
Provides robust performance estimates with confidence intervals
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import roc_auc_score, accuracy_score, precision_recall_fscore_support
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import json
from scipy import stats

# Paths
DATA_DIR = Path("data/analysis/prpc_validation")
RESULTS_DIR = Path("results/cross_validation")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("Cross-Validation Analysis - TCGA Data")
print("=" * 80)

# Load features and targets
features_file = DATA_DIR / "features/feature_matrix_full.csv"
targets_file = DATA_DIR / "features/targets.csv"

if not features_file.exists():
    print("⚠️  Features not found. Creating from TCGA...")
    # Run feature engineering first
    import subprocess
    subprocess.run(["python", "scripts/feature_engineering.py"], check=True)

features = pd.read_csv(features_file)
targets = pd.read_csv(targets_file)

# Use binary classification target
X = features.values
y = targets['prnp_high'].values

print(f"\nData loaded:")
print(f"   Samples: {len(X)}")
print(f"   Features: {X.shape[1]}")
print(f"   Positive class: {y.sum()} ({y.sum()/len(y)*100:.1f}%)")

# Normalize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ==============================================================================
# 1. K-Fold Cross-Validation with Multiple Models
# ==============================================================================

print("\n" + "=" * 80)
print("1. K-Fold Cross-Validation (k=5)")
print("=" * 80)

models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
    'SVM': SVC(probability=True, random_state=42)
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

results = {}

for name, model in models.items():
    print(f"\n{name}:")
    
    fold_aucs = []
    fold_accs = []
    
    for fold, (train_idx, val_idx) in enumerate(cv.split(X_scaled, y), 1):
        X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        # Train
        model.fit(X_train, y_train)
        
        # Predict
        y_pred_proba = model.predict_proba(X_val)[:, 1]
        y_pred = model.predict(X_val)
        
        # Metrics
        auc = roc_auc_score(y_val, y_pred_proba)
        acc = accuracy_score(y_val, y_pred)
        
        fold_aucs.append(auc)
        fold_accs.append(acc)
        
        print(f"   Fold {fold}: AUC={auc:.4f}, Acc={acc:.4f}")
    
    # Calculate mean and CI
    mean_auc = np.mean(fold_aucs)
    std_auc = np.std(fold_aucs)
    ci_auc = stats.t.interval(0.95, len(fold_aucs)-1, 
                              loc=mean_auc, 
                              scale=stats.sem(fold_aucs))
    
    mean_acc = np.mean(fold_accs)
    std_acc = np.std(fold_accs)
    
    results[name] = {
        'mean_auc': mean_auc,
        'std_auc': std_auc,
        'ci_auc_lower': ci_auc[0],
        'ci_auc_upper': ci_auc[1],
        'mean_acc': mean_acc,
        'std_acc': std_acc,
        'fold_aucs': fold_aucs,
        'fold_accs': fold_accs
    }
    
    print(f"\n   Summary:")
    print(f"   AUC: {mean_auc:.4f} ± {std_auc:.4f}")
    print(f"   95% CI: [{ci_auc[0]:.4f}, {ci_auc[1]:.4f}]")
    print(f"   Accuracy: {mean_acc:.4f} ± {std_acc:.4f}")

# ==============================================================================
# 2. Bootstrap Confidence Intervals
# ==============================================================================

print("\n" + "=" * 80)
print("2. Bootstrap Analysis (n=1000)")
print("=" * 80)

# Use best model (RF typically best)
best_model = RandomForestClassifier(n_estimators=200, random_state=42)
best_model.fit(X_scaled, y)

n_bootstrap = 1000
bootstrap_aucs = []

np.random.seed(42)
for i in range(n_bootstrap):
    # Bootstrap sample
    indices = np.random.choice(len(X_scaled), size=len(X_scaled), replace=True)
    oob_indices = np.array([i for i in range(len(X_scaled)) if i not in indices])
    
    if len(oob_indices) == 0:
        continue
    
    # Predict on out-of-bag samples
    X_oob = X_scaled[oob_indices]
    y_oob = y[oob_indices]
    
    y_pred_proba = best_model.predict_proba(X_oob)[:, 1]
    
    try:
        auc = roc_auc_score(y_oob, y_pred_proba)
        bootstrap_aucs.append(auc)
    except:
        continue

bootstrap_mean = np.mean(bootstrap_aucs)
bootstrap_ci = np.percentile(bootstrap_aucs, [2.5, 97.5])

print(f"\nBootstrap Results:")
print(f"   Mean AUC: {bootstrap_mean:.4f}")
print(f"   95% CI: [{bootstrap_ci[0]:.4f}, {bootstrap_ci[1]:.4f}]")
print(f"   Samples: {len(bootstrap_aucs)}")

# ==============================================================================
# 3. Leave-One-Cancer-Out CV
# ==============================================================================

print("\n" + "=" * 80)
print("3. Leave-One-Cancer-Out Cross-Validation")
print("=" * 80)

# Load cancer types
tcga_file = DATA_DIR / "open_data/real/tcga_all_cancers_prnp_real.csv"
tcga_df = pd.read_csv(tcga_file)
cancer_types = tcga_df['cancer_type'].unique()

loco_results = {}

for leave_out_cancer in cancer_types:
    # Split by cancer type
    train_mask = tcga_df['cancer_type'] != leave_out_cancer
    test_mask = tcga_df['cancer_type'] == leave_out_cancer
    
    X_train = X_scaled[train_mask]
    y_train = y[train_mask]
    X_test = X_scaled[test_mask]
    y_test = y[test_mask]
    
    # Train model
    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)
    
    # Predict
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)
    
    # Metrics
    auc = roc_auc_score(y_test, y_pred_proba)
    acc = accuracy_score(y_test, y_pred)
    
    loco_results[leave_out_cancer] = {
        'auc': auc,
        'acc': acc,
        'n_samples': len(y_test)
    }
    
    print(f"\nLeave-out {leave_out_cancer}:")
    print(f"   Test samples: {len(y_test)}")
    print(f"   AUC: {auc:.4f}")
    print(f"   Accuracy: {acc:.4f}")

# ==============================================================================
# 4. Save Results
# ==============================================================================

print("\n" + "=" * 80)
print("4. Saving Results")
print("=" * 80)

# Save numerical results
all_results = {
    'k_fold_cv': results,
    'bootstrap': {
        'mean_auc': float(bootstrap_mean),
        'ci_lower': float(bootstrap_ci[0]),
        'ci_upper': float(bootstrap_ci[1]),
        'n_iterations': len(bootstrap_aucs)
    },
    'leave_one_cancer_out': loco_results
}

results_file = RESULTS_DIR / 'cross_validation_results.json'
with open(results_file, 'w') as f:
    json.dump(all_results, f, indent=2)
print(f"   Results: {results_file}")

# ==============================================================================
# 5. Visualization
# ==============================================================================

print("\n5. Creating Visualizations")
print("-" * 40)

fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# 1. Model comparison (K-fold)
model_names = list(results.keys())
mean_aucs = [results[m]['mean_auc'] for m in model_names]
std_aucs = [results[m]['std_auc'] for m in model_names]

axes[0, 0].barh(model_names, mean_aucs, xerr=std_aucs, color='#3498db', alpha=0.7, capsize=5)
axes[0, 0].axvline(0.75, color='red', linestyle='--', label='Target (0.75)')
axes[0, 0].set_xlabel('AUC')
axes[0, 0].set_title('A. Model Comparison (5-Fold CV)')
axes[0, 0].legend()
axes[0, 0].grid(axis='x', alpha=0.3)
axes[0, 0].set_xlim([0.5, 1.0])

# 2. Bootstrap distribution
axes[0, 1].hist(bootstrap_aucs, bins=50, color='#2ecc71', alpha=0.7, edgecolor='black')
axes[0, 1].axvline(bootstrap_mean, color='red', linestyle='--', linewidth=2, label='Mean')
axes[0, 1].axvline(bootstrap_ci[0], color='orange', linestyle='--', label='95% CI')
axes[0, 1].axvline(bootstrap_ci[1], color='orange', linestyle='--')
axes[0, 1].set_xlabel('AUC')
axes[0, 1].set_ylabel('Frequency')
axes[0, 1].set_title('B. Bootstrap Distribution (n=1000)')
axes[0, 1].legend()
axes[0, 1].grid(alpha=0.3)

# 3. Leave-one-cancer-out
loco_cancers = list(loco_results.keys())
loco_aucs = [loco_results[c]['auc'] for c in loco_cancers]

axes[1, 0].barh(loco_cancers, loco_aucs, color='#e74c3c', alpha=0.7, edgecolor='black')
axes[1, 0].axvline(0.75, color='red', linestyle='--', linewidth=2, label='Target')
axes[1, 0].set_xlabel('AUC')
axes[1, 0].set_title('C. Leave-One-Cancer-Out CV')
axes[1, 0].legend()
axes[1, 0].grid(axis='x', alpha=0.3)

# 4. Fold-wise performance (best model)
best_model_name = max(results.keys(), key=lambda k: results[k]['mean_auc'])
fold_aucs = results[best_model_name]['fold_aucs']

axes[1, 1].plot(range(1, 6), fold_aucs, 'o-', linewidth=2, markersize=10, color='#9b59b6')
axes[1, 1].axhline(np.mean(fold_aucs), color='red', linestyle='--', label='Mean')
axes[1, 1].set_xlabel('Fold')
axes[1, 1].set_ylabel('AUC')
axes[1, 1].set_title(f'D. Fold Performance ({best_model_name})')
axes[1, 1].legend()
axes[1, 1].grid(alpha=0.3)
axes[1, 1].set_ylim([0.5, 1.0])

plt.tight_layout()
fig_file = RESULTS_DIR / 'cross_validation_analysis.png'
plt.savefig(fig_file, dpi=300, bbox_inches='tight')
print(f"   Figure: {fig_file}")

# ==============================================================================
# Final Summary
# ==============================================================================

print("\n" + "=" * 80)
print("✅ CROSS-VALIDATION COMPLETE!")
print("=" * 80)

best_cv_auc = results[best_model_name]['mean_auc']
best_cv_ci = (results[best_model_name]['ci_auc_lower'], 
              results[best_model_name]['ci_auc_upper'])

print(f"\n📊 Final Results:")
print(f"   Best Model: {best_model_name}")
print(f"   Cross-Val AUC: {best_cv_auc:.4f} (95% CI: [{best_cv_ci[0]:.4f}, {best_cv_ci[1]:.4f}])")
print(f"   Bootstrap AUC: {bootstrap_mean:.4f} (95% CI: [{bootstrap_ci[0]:.4f}, {bootstrap_ci[1]:.4f}])")
print(f"   LOCO Mean AUC: {np.mean(loco_aucs):.4f}")

if best_cv_auc > 0.85:
    print(f"\n🎯 EXCELLENT! AUC > 0.85")
    print(f"   Publication-ready performance!")
elif best_cv_auc > 0.75:
    print(f"\n✓ GOOD! AUC > 0.75")
    print(f"   Meets target threshold!")
else:
    print(f"\n⚠️  AUC < 0.75")
    print(f"   Consider feature engineering or model tuning")

print(f"\n📁 Output files:")
print(f"   {results_file}")
print(f"   {fig_file}")
