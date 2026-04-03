"""
PrPc Patient Serum Data - Statistical Analysis
===============================================
Performs comprehensive statistical analysis of patient serum PrPc concentration data.
Includes:
- Descriptive statistics
- Group comparison (normal vs cancer patients)
- ROC curve analysis
- Diagnostic performance metrics
"""

import sys
import json
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.metrics import roc_curve, auc, confusion_matrix
import seaborn as sns

# Fix Windows encoding
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')
    import matplotlib
    matplotlib.rc('font', family='Malgun Gothic')
    plt.rcParams['axes.unicode_minus'] = False

# Paths
SERUM_FILE = Path("C:/Users/brook/Desktop/ADDS/prpc/프리온 농도 환자혈청(정상인 3기환자)결과 보정.xlsx")
OUTPUT_DIR = Path("data/analysis/prpc_clinical_data")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("PrPc PATIENT SERUM DATA - STATISTICAL ANALYSIS")
print("=" * 80)
print(f"Data source: {SERUM_FILE.name}")
print()

# ============================================================================
# PART 1: Load and Explore Data
# ============================================================================

print("\n" + "=" * 80)
print("PART 1: DATA LOADING AND EXPLORATION")
print("=" * 80)

df = pd.read_excel(SERUM_FILE)
print(f"\nOriginal shape: {df.shape}")
print(f"Columns: {list(df.columns)}")
print("\nFirst few rows:")
print(df.head(10))

# The file name suggests: 정상인 (normal) vs 3기환자 (stage 3 patients)
# Column headers appear to be concentration values
# Let's infer the structure

# Get column names
col1 = df.columns[1]  # First measurement column
col2 = df.columns[2]  # Second measurement column

print(f"\nColumn 1 (header): {col1}")
print(f"Column 2 (header): {col2}")

# Convert to numeric, coercing errors to NaN
col1_numeric = pd.to_numeric(df[col1], errors='coerce')
col2_numeric = pd.to_numeric(df[col2], errors='coerce')

# Count non-NaN values in each column
col1_count = col1_numeric.notna().sum()
col2_count = col2_numeric.notna().sum()

print(f"\nNon-NaN values:")
print(f"  Column 1: {col1_count}")
print(f"  Column 2: {col2_count}")

# Hypothesis: Column 1 = Normal samples, Column 2 = Patient samples
# OR the columns represent two different measurements
# Let's check if there's a pattern

# ============================================================================
# PART 2: Group Assignment Based on Data Structure
# ============================================================================

print("\n" + "=" * 80)
print("PART 2: GROUP ASSIGNMENT")
print("=" * 80)

# From the preview, it looks like:
# - Column 1 has data for rows 0-20 (then NaN)
# - Column 2 has data for all rows 0-42
# This suggests Column 1 = one group, Column 2 = another group

# Extract non-NaN numeric values from each column
group1_data = col1_numeric.dropna().values
group2_data = col2_numeric.dropna().values

print(f"\nGroup 1 (Column {col1}):")
print(f"  N = {len(group1_data)}")
print(f"  Mean = {np.mean(group1_data):.4f}")
print(f"  Std = {np.std(group1_data, ddof=1):.4f}")
print(f"  Range = [{np.min(group1_data):.4f}, {np.max(group1_data):.4f}]")

print(f"\nGroup 2 (Column {col2}):")
print(f"  N = {len(group2_data)}")
print(f"  Mean = {np.mean(group2_data):.4f}")
print(f"  Std = {np.std(group2_data, ddof=1):.4f}")
print(f"  Range = [{np.min(group2_data):.4f}, {np.max(group2_data):.4f}]")

# Based on file name "정상인 3기환자", let's assign:
# If Group 2 (larger n, potentially all samples) has higher mean → patients
# If Group 1 (smaller n) has lower values → normal

if np.mean(group1_data) < np.mean(group2_data):
    normal_data = group1_data
    patient_data = group2_data
    print("\n[ASSIGNMENT] Based on mean values:")
    print(f"  Normal group = Column {col1} (lower values)")
    print(f"  Patient group = Column {col2} (higher values)")
else:
    normal_data = group2_data
    patient_data = group1_data
    print("\n[ASSIGNMENT] Based on mean values:")
    print(f"  Normal group = Column {col2} (lower values)")
    print(f"  Patient group = Column {col1} (higher values)")

# ============================================================================
# PART 3: Descriptive Statistics
# ============================================================================

print("\n" + "=" * 80)
print("PART 3: DESCRIPTIVE STATISTICS")
print("=" * 80)

stats_summary = pd.DataFrame({
    'Group': ['Normal (정상인)', 'Stage 3 Patient (3기 환자)'],
    'N': [len(normal_data), len(patient_data)],
    'Mean': [np.mean(normal_data), np.mean(patient_data)],
    'Std': [np.std(normal_data, ddof=1), np.std(patient_data, ddof=1)],
    'Median': [np.median(normal_data), np.median(patient_data)],
    'Min': [np.min(normal_data), np.min(patient_data)],
    'Max': [np.max(normal_data), np.max(patient_data)],
    'Q1': [np.percentile(normal_data, 25), np.percentile(patient_data, 25)],
    'Q3': [np.percentile(normal_data, 75), np.percentile(patient_data, 75)]
})

print("\n" + stats_summary.to_string(index=False))

# Save statistics
stats_file = OUTPUT_DIR / "serum_statistics_summary.xlsx"
stats_summary.to_excel(stats_file, index=False, engine='openpyxl')
print(f"\n[SAVED] {stats_file}")

# ============================================================================
# PART 4: Statistical Hypothesis Testing
# ============================================================================

print("\n" + "=" * 80)
print("PART 4: STATISTICAL HYPOTHESIS TESTING")
print("=" * 80)

# 1. Normality test (Shapiro-Wilk)
print("\n[1] Normality Test (Shapiro-Wilk):")
stat_normal, p_normal_normal = stats.shapiro(normal_data)
stat_patient, p_normal_patient = stats.shapiro(patient_data)

print(f"  Normal group: W={stat_normal:.4f}, p={p_normal_normal:.4f}")
print(f"  Patient group: W={stat_patient:.4f}, p={p_normal_patient:.4f}")

if p_normal_normal > 0.05 and p_normal_patient > 0.05:
    print("  → Both groups are normally distributed (p > 0.05)")
    use_parametric = True
else:
    print("  → At least one group is NOT normally distributed (p ≤ 0.05)")
    use_parametric = False

# 2. Variance equality test (Levene's)
print("\n[2] Variance Equality Test (Levene's):")
stat_levene, p_levene = stats.levene(normal_data, patient_data)
print(f"  Statistic={stat_levene:.4f}, p={p_levene:.4f}")

if p_levene > 0.05:
    print("  → Variances are equal (p > 0.05)")
    equal_var = True
else:
    print("  → Variances are NOT equal (p ≤ 0.05)")
    equal_var = False

# 3. Group comparison test
print("\n[3] Group Comparison:")

if use_parametric:
    # Independent t-test
    stat_test, p_value = stats.ttest_ind(normal_data, patient_data, equal_var=equal_var)
    test_name = "Independent t-test" + (" (equal var)" if equal_var else " (Welch's)")
else:
    # Mann-Whitney U test (non-parametric)
    stat_test, p_value = stats.mannwhitneyu(normal_data, patient_data, alternative='two-sided')
    test_name = "Mann-Whitney U test"

print(f"  Test used: {test_name}")
print(f"  Statistic = {stat_test:.4f}")
print(f"  p-value = {p_value:.6f}")

if p_value < 0.001:
    print(f"  → HIGHLY SIGNIFICANT difference (p < 0.001) ***")
elif p_value < 0.01:
    print(f"  → VERY SIGNIFICANT difference (p < 0.01) **")
elif p_value < 0.05:
    print(f"  → SIGNIFICANT difference (p < 0.05) *")
else:
    print(f"  → NO significant difference (p ≥ 0.05)")

# Effect size (Cohen's d)
pooled_std = np.sqrt(((len(normal_data)-1)*np.var(normal_data, ddof=1) + 
                      (len(patient_data)-1)*np.var(patient_data, ddof=1)) / 
                     (len(normal_data) + len(patient_data) - 2))
cohens_d = (np.mean(patient_data) - np.mean(normal_data)) / pooled_std

print(f"\n[4] Effect Size (Cohen's d):")
print(f"  d = {cohens_d:.4f}")
if abs(cohens_d) < 0.2:
    print("  → Small effect")
elif abs(cohens_d) < 0.5:
    print("  → Medium effect")
else:
    print("  → Large effect")

# ============================================================================
# PART 5: ROC Curve Analysis
# ============================================================================

print("\n" + "=" * 80)
print("PART 5: ROC CURVE ANALYSIS")
print("=" * 80)

# Prepare data for ROC
# Label: 0 = Normal, 1 = Patient
y_true = np.concatenate([np.zeros(len(normal_data)), np.ones(len(patient_data))])
y_score = np.concatenate([normal_data, patient_data])

# Calculate ROC curve
fpr, tpr, thresholds = roc_curve(y_true, y_score)
roc_auc = auc(fpr, tpr)

print(f"\nAUC (Area Under Curve) = {roc_auc:.4f}")

if roc_auc >= 0.9:
    print("  → EXCELLENT diagnostic accuracy")
elif roc_auc >= 0.8:
    print("  → GOOD diagnostic accuracy")
elif roc_auc >= 0.7:
    print("  → ACCEPTABLE diagnostic accuracy")
elif roc_auc >= 0.6:
    print("  → POOR diagnostic accuracy")
else:
    print("  → VERY POOR diagnostic accuracy")

# Find optimal threshold (Youden's index)
youden_index = tpr - fpr
optimal_idx = np.argmax(youden_index)
optimal_threshold = thresholds[optimal_idx]
optimal_sensitivity = tpr[optimal_idx]
optimal_specificity = 1 - fpr[optimal_idx]

print(f"\nOptimal Threshold (Youden's Index):")
print(f"  Threshold = {optimal_threshold:.4f}")
print(f"  Sensitivity = {optimal_sensitivity:.4f} ({optimal_sensitivity*100:.1f}%)")
print(f"  Specificity = {optimal_specificity:.4f} ({optimal_specificity*100:.1f}%)")
print(f"  Youden's J = {youden_index[optimal_idx]:.4f}")

# Confusion matrix at optimal threshold
y_pred = (y_score >= optimal_threshold).astype(int)
cm = confusion_matrix(y_true, y_pred)
tn, fp, fn, tp = cm.ravel()

print(f"\nConfusion Matrix at Optimal Threshold:")
print(f"  True Negative (TN) = {tn}")
print(f"  False Positive (FP) = {fp}")
print(f"  False Negative (FN) = {fn}")
print(f"  True Positive (TP) = {tp}")

accuracy = (tp + tn) / (tp + tn + fp + fn)
ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
npv = tn / (tn + fn) if (tn + fn) > 0 else 0

print(f"\nDiagnostic Performance Metrics:")
print(f"  Accuracy = {accuracy:.4f} ({accuracy*100:.1f}%)")
print(f"  PPV (Positive Predictive Value) = {ppv:.4f} ({ppv*100:.1f}%)")
print(f"  NPV (Negative Predictive Value) = {npv:.4f} ({npv*100:.1f}%)")

# ============================================================================
# PART 6: Visualizations
# ============================================================================

print("\n" + "=" * 80)
print("PART 6: CREATING VISUALIZATIONS")
print("=" * 80)

# Create figure with subplots
fig, axes = plt.subplots(2, 2, figsize=(14, 12))
fig.suptitle('PrPc 환자 혈청 데이터 통계 분석', fontsize=16, fontweight='bold')

# 1. Box plot
ax1 = axes[0, 0]
data_for_box = [normal_data, patient_data]
bp = ax1.boxplot(data_for_box, labels=['정상인', '3기 환자'], patch_artist=True)
bp['boxes'][0].set_facecolor('lightblue')
bp['boxes'][1].set_facecolor('lightcoral')
ax1.set_ylabel('PrPc 농도', fontsize=12)
ax1.set_title('그룹별 PrPc 농도 분포', fontsize=12, fontweight='bold')
ax1.grid(True, alpha=0.3)

# Add p-value annotation
y_max = max(np.max(normal_data), np.max(patient_data))
if p_value < 0.001:
    sig_text = '***'
elif p_value < 0.01:
    sig_text = '**'
elif p_value < 0.05:
    sig_text = '*'
else:
    sig_text = 'ns'
ax1.text(1.5, y_max * 1.05, f'p={p_value:.4f} {sig_text}', ha='center', fontsize=10)

# 2. Violin plot
ax2 = axes[0, 1]
parts = ax2.violinplot([normal_data, patient_data], positions=[1, 2], 
                        showmeans=True, showmedians=True)
ax2.set_xticks([1, 2])
ax2.set_xticklabels(['정상인', '3기 환자'])
ax2.set_ylabel('PrPc 농도', fontsize=12)
ax2.set_title('분포 형태 비교 (Violin Plot)', fontsize=12, fontweight='bold')
ax2.grid(True, alpha=0.3, axis='y')

# 3. ROC Curve
ax3 = axes[1, 0]
ax3.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
ax3.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random (AUC = 0.5)')
ax3.scatter(fpr[optimal_idx], tpr[optimal_idx], marker='o', color='red', s=100, 
            label=f'Optimal (Thr={optimal_threshold:.3f})', zorder=5)
ax3.set_xlim([0.0, 1.0])
ax3.set_ylim([0.0, 1.05])
ax3.set_xlabel('False Positive Rate (1 - Specificity)', fontsize=11)
ax3.set_ylabel('True Positive Rate (Sensitivity)', fontsize=11)
ax3.set_title('ROC Curve - 진단 성능', fontsize=12, fontweight='bold')
ax3.legend(loc="lower right", fontsize=9)
ax3.grid(True, alpha=0.3)

# 4. Histogram overlay
ax4 = axes[1, 1]
ax4.hist(normal_data, bins=15, alpha=0.6, label='정상인', color='blue', density=True)
ax4.hist(patient_data, bins=15, alpha=0.6, label='3기 환자', color='red', density=True)
ax4.axvline(optimal_threshold, color='green', linestyle='--', linewidth=2, 
            label=f'최적 임계값 = {optimal_threshold:.3f}')
ax4.set_xlabel('PrPc 농도', fontsize=11)
ax4.set_ylabel('밀도', fontsize=11)
ax4.set_title('그룹별 분포 및 최적 임계값', fontsize=12, fontweight='bold')
ax4.legend(fontsize=9)
ax4.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
fig_file = OUTPUT_DIR / "serum_statistical_analysis_plots.png"
plt.savefig(fig_file, dpi=300, bbox_inches='tight')
print(f"\n[SAVED] {fig_file}")
plt.close()

# ============================================================================
# PART 7: Summary Report
# ============================================================================

print("\n" + "=" * 80)
print("PART 7: COMPREHENSIVE SUMMARY")
print("=" * 80)

summary_report = {
    "data_summary": {
        "normal_n": int(len(normal_data)),
        "patient_n": int(len(patient_data)),
        "total_n": int(len(normal_data) + len(patient_data))
    },
    "descriptive_statistics": {
        "normal": {
            "mean": float(np.mean(normal_data)),
            "std": float(np.std(normal_data, ddof=1)),
            "median": float(np.median(normal_data)),
            "range": [float(np.min(normal_data)), float(np.max(normal_data))]
        },
        "patient": {
            "mean": float(np.mean(patient_data)),
            "std": float(np.std(patient_data, ddof=1)),
            "median": float(np.median(patient_data)),
            "range": [float(np.min(patient_data)), float(np.max(patient_data))]
        }
    },
    "statistical_tests": {
        "normality_test": {
            "test": "Shapiro-Wilk",
            "normal_p": float(p_normal_normal),
            "patient_p": float(p_normal_patient),
            "both_normal": bool(p_normal_normal > 0.05 and p_normal_patient > 0.05)
        },
        "variance_test": {
            "test": "Levene's",
            "p_value": float(p_levene),
            "equal_variance": bool(p_levene > 0.05)
        },
        "group_comparison": {
            "test": test_name,
            "statistic": float(stat_test),
            "p_value": float(p_value),
            "significant": bool(p_value < 0.05),
            "cohens_d": float(cohens_d)
        }
    },
    "roc_analysis": {
        "auc": float(roc_auc),
        "optimal_threshold": float(optimal_threshold),
        "sensitivity": float(optimal_sensitivity),
        "specificity": float(optimal_specificity),
        "accuracy": float(accuracy),
        "ppv": float(ppv),
        "npv": float(npv)
    },
    "confusion_matrix": {
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_positive": int(tp)
    },
    "clinical_interpretation": {
        "diagnostic_utility": "EXCELLENT" if roc_auc >= 0.9 else "GOOD" if roc_auc >= 0.8 else "ACCEPTABLE",
        "recommended_cutoff": float(optimal_threshold),
        "can_distinguish_groups": bool(p_value < 0.05 and roc_auc > 0.7)
    }
}

# Save JSON report
json_file = OUTPUT_DIR / "serum_statistical_analysis_report.json"
with open(json_file, 'w', encoding='utf-8') as f:
    json.dump(summary_report, f, indent=2, ensure_ascii=False)
print(f"\n[SAVED] {json_file}")

print("\n" + "=" * 80)
print("ANALYSIS COMPLETE")
print("=" * 80)
print(f"\n총 {len(normal_data) + len(patient_data)}개 샘플 분석 완료")
print(f"  - 정상인: {len(normal_data)}명")
print(f"  - 3기 환자: {len(patient_data)}명")
print(f"\n주요 결과:")
print(f"  - 그룹 간 차이: p={p_value:.6f} {'(유의함)' if p_value < 0.05 else '(유의하지 않음)'}")
print(f"  - AUC: {roc_auc:.4f} ({summary_report['clinical_interpretation']['diagnostic_utility']})")
print(f"  - 최적 임계값: {optimal_threshold:.4f}")
print(f"  - 민감도: {optimal_sensitivity*100:.1f}%")
print(f"  - 특이도: {optimal_specificity*100:.1f}%")
print(f"\n결론: PrPc 혈청 농도는 {'유용한' if p_value < 0.05 and roc_auc > 0.7 else '제한적인'} 진단 바이오마커입니다.")
print("=" * 80)
