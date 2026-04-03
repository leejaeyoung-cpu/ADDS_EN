#!/usr/bin/env python3
"""
SHAP Analysis for Model Interpretability
Week 2: Understanding what drives the predictions
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import shap
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import json

# Configuration
DATA_DIR = Path("data/analysis/prpc_validation")
RESULTS_DIR = Path("results/interpretability")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("SHAP Analysis - Model Interpretability")
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

# Normalize
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train best model
print("\nTraining Random Forest...")
model = RandomForestClassifier(n_estimators=200, random_state=42, max_depth=10)
model.fit(X_scaled, y)
print(f"   Training accuracy: {model.score(X_scaled, y):.4f}")

# ==============================================================================
# 1. SHAP TreeExplainer
# ==============================================================================

print("\n" + "=" * 80)
print("1. Computing SHAP Values")
print("=" * 80)

# Use TreeExplainer for Random Forest
explainer = shap.TreeExplainer(model)

# Calculate SHAP values (use subset for speed)
sample_size = min(500, len(X_scaled))
sample_idx = np.random.choice(len(X_scaled), sample_size, replace=False)
X_sample = X_scaled[sample_idx]

print(f"\nCalculating SHAP values for {sample_size} samples...")
shap_values = explainer.shap_values(X_sample)

print(f"   Raw SHAP values shape: {shap_values.shape if not isinstance(shap_values, list) else [s.shape for s in shap_values]}")

# Handle different SHAP output formats
if isinstance(shap_values, list):
    # List of arrays (one per class)
    shap_values = shap_values[1]  # Use positive class
elif len(shap_values.shape) == 3:
    # Shape is (samples, features, classes)
    shap_values = shap_values[:, :, 1]  # Use positive class

print(f"   Final SHAP values shape: {shap_values.shape}")

# ==============================================================================
# 2. Feature Importance
# ==============================================================================

print("\n" + "=" * 80)
print("2. Feature Importance Analysis")
print("=" * 80)

# Mean absolute SHAP values
mean_abs_shap = np.abs(shap_values).mean(axis=0)

# Sort features by importance
importance_df = pd.DataFrame({
    'feature': feature_names,
    'importance': mean_abs_shap,
    'rf_importance': model.feature_importances_
})
importance_df = importance_df.sort_values('importance', ascending=False)

print(f"\nTop 20 Most Important Features:")
print(importance_df.head(20).to_string(index=False))

# Save full results
importance_file = RESULTS_DIR / 'feature_importance_shap.csv'
importance_df.to_csv(importance_file, index=False)
print(f"\n   Saved: {importance_file}")

# ==============================================================================
# 3. Visualizations
# ==============================================================================

print("\n" + "=" * 80)
print("3. Creating SHAP Visualizations")
print("=" * 80)

# 3.1 Summary Plot
fig, ax = plt.subplots(figsize=(10, 8))
shap.summary_plot(shap_values, X_sample, feature_names=feature_names, 
                  show=False, max_display=20)
plt.tight_layout()
summary_file = RESULTS_DIR / 'shap_summary_plot.png'
plt.savefig(summary_file, dpi=300, bbox_inches='tight')
print(f"   Saved: {summary_file}")
plt.close()

# 3.2 Bar Plot (Top 20)
fig, ax = plt.subplots(figsize=(10, 8))
top_20 = importance_df.head(20)
ax.barh(range(len(top_20)), top_20['importance'], color='#3498db', edgecolor='black')
ax.set_yticks(range(len(top_20)))
ax.set_yticklabels(top_20['feature'])
ax.set_xlabel('Mean |SHAP Value|', fontsize=12, fontweight='bold')
ax.set_title('Top 20 Features by SHAP Importance', fontsize=14, fontweight='bold')
ax.invert_yaxis()
ax.grid(axis='x', alpha=0.3)
plt.tight_layout()
bar_file = RESULTS_DIR / 'shap_importance_bar.png'
plt.savefig(bar_file, dpi=300, bbox_inches='tight')
print(f"   Saved: {bar_file}")
plt.close()

# 3.3 Comparison: SHAP vs RF importance
fig, ax = plt.subplots(figsize=(10, 8))
top_features = importance_df.head(20)

x = np.arange(len(top_features))
width = 0.35

ax.barh(x - width/2, top_features['importance'], width, 
        label='SHAP', color='#3498db', alpha=0.8, edgecolor='black')
ax.barh(x + width/2, top_features['rf_importance'], width,
        label='RF Gini', color='#e74c3c', alpha=0.8, edgecolor='black')

ax.set_yticks(x)
ax.set_yticklabels(top_features['feature'])
ax.set_xlabel('Importance Score', fontsize=12, fontweight='bold')
ax.set_title('SHAP vs Random Forest Feature Importance', fontsize=14, fontweight='bold')
ax.legend()
ax.invert_yaxis()
ax.grid(axis='x', alpha=0.3)
plt.tight_layout()
comparison_file = RESULTS_DIR / 'importance_comparison.png'
plt.savefig(comparison_file, dpi=300, bbox_inches='tight')
print(f"   Saved: {comparison_file}")
plt.close()

# ==============================================================================
# 4. Feature Groups Analysis
# ==============================================================================

print("\n" + "=" * 80)
print("4. Feature Group Analysis")
print("=" * 80)

# Categorize features by type
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

importance_df['category'] = importance_df['feature'].apply(categorize_feature)

# Group by category
category_importance = importance_df.groupby('category')['importance'].agg(['sum', 'mean', 'count'])
category_importance = category_importance.sort_values('sum', ascending=False)

print(f"\nImportance by Feature Category:")
print(category_importance.to_string())

# Visualize
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Total importance
axes[0].barh(category_importance.index, category_importance['sum'], 
            color='#2ecc71', alpha=0.8, edgecolor='black')
axes[0].set_xlabel('Total SHAP Importance', fontsize=12, fontweight='bold')
axes[0].set_title('A. Total Importance by Category', fontsize=14, fontweight='bold')
axes[0].invert_yaxis()
axes[0].grid(axis='x', alpha=0.3)

# Average importance
axes[1].barh(category_importance.index, category_importance['mean'],
            color='#f39c12', alpha=0.8, edgecolor='black')
axes[1].set_xlabel('Mean SHAP Importance', fontsize=12, fontweight='bold')
axes[1].set_title('B. Mean Importance by Category', fontsize=14, fontweight='bold')
axes[1].invert_yaxis()
axes[1].grid(axis='x', alpha=0.3)

plt.tight_layout()
category_file = RESULTS_DIR / 'category_importance.png'
plt.savefig(category_file, dpi=300, bbox_inches='tight')
print(f"   Saved: {category_file}")
plt.close()

# ==============================================================================
# 5. Individual Predictions (Examples)
# ==============================================================================

print("\n" + "=" * 80)
print("5. Example Prediction Explanations")
print("=" * 80)

# Select a few interesting examples
high_prnp_idx = np.where(y[sample_idx] == 1)[0][:3]
low_prnp_idx = np.where(y[sample_idx] == 0)[0][:3]

for i, (idx_list, label) in enumerate([(high_prnp_idx, 'High PRNP'), 
                                        (low_prnp_idx, 'Low PRNP')]):
    for j, idx in enumerate(idx_list):
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Get SHAP values for this instance
        instance_shap = shap_values[idx]
        
        # Get top contributing features
        top_idx = np.argsort(np.abs(instance_shap))[-15:][::-1]
        
        # Plot
        colors = ['red' if v > 0 else 'blue' for v in instance_shap[top_idx]]
        ax.barh(range(len(top_idx)), instance_shap[top_idx], color=colors, alpha=0.7, edgecolor='black')
        ax.set_yticks(range(len(top_idx)))
        ax.set_yticklabels([feature_names[i] for i in top_idx])
        ax.set_xlabel('SHAP Value (Impact on Prediction)', fontsize=12, fontweight='bold')
        ax.set_title(f'{label} Example {j+1}: Top 15 Feature Contributions', 
                    fontsize=14, fontweight='bold')
        ax.axvline(0, color='black', linestyle='--', linewidth=2)
        ax.invert_yaxis()
        ax.grid(axis='x', alpha=0.3)
        
        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor='red', alpha=0.7, label='Pushes towards High'),
                          Patch(facecolor='blue', alpha=0.7, label='Pushes towards Low')]
        ax.legend(handles=legend_elements, loc='best')
        
        plt.tight_layout()
        example_file = RESULTS_DIR / f'example_{label.replace(" ", "_").lower()}_{j+1}.png'
        plt.savefig(example_file, dpi=300, bbox_inches='tight')
        plt.close()

print(f"   Saved 6 example explanations")

# ==============================================================================
# 6. Summary Statistics
# ==============================================================================

print("\n" + "=" * 80)
print("6. Summary Statistics")
print("=" * 80)

summary = {
    'total_features': len(feature_names),
    'top_20_cumulative_importance': float(importance_df.head(20)['importance'].sum()),
    'top_10_cumulative_importance': float(importance_df.head(10)['importance'].sum()),
    'top_5_cumulative_importance': float(importance_df.head(5)['importance'].sum()),
    'most_important_feature': importance_df.iloc[0]['feature'],
    'most_important_value': float(importance_df.iloc[0]['importance']),
    'category_distribution': category_importance['sum'].to_dict()
}

summary_file = RESULTS_DIR / 'shap_summary.json'
with open(summary_file, 'w') as f:
    json.dump(summary, f, indent=2)

print(f"\nKey Insights:")
print(f"   Most important feature: {summary['most_important_feature']}")
print(f"   Top 5 features explain: {summary['top_5_cumulative_importance']:.1%} of predictions")
print(f"   Top 10 features explain: {summary['top_10_cumulative_importance']:.1%} of predictions")
print(f"   Top 20 features explain: {summary['top_20_cumulative_importance']:.1%} of predictions")

print(f"\n   Saved: {summary_file}")

# ==============================================================================
# Final Summary
# ==============================================================================

print("\n" + "=" * 80)
print("✅ SHAP ANALYSIS COMPLETE!")
print("=" * 80)

print(f"\n📊 Generated outputs:")
print(f"   1. Feature importance table: {importance_file}")
print(f"   2. SHAP summary plot: {summary_file}")
print(f"   3. Importance bar chart: {bar_file}")
print(f"   4. SHAP vs RF comparison: {comparison_file}")
print(f"   5. Category analysis: {category_file}")
print(f"   6. Individual examples: 6 files")
print(f"   7. Summary statistics: {summary_file}")

print(f"\n🔍 Key Findings:")
print(f"   ✓ Identified most important features")
print(f"   ✓ Compared SHAP vs RF importance")
print(f"   ✓ Analyzed feature categories")
print(f"   ✓ Explained individual predictions")

print(f"\n📁 All results in: {RESULTS_DIR}")
print("\n🚀 Ready for biological interpretation!")
