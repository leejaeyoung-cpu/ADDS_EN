"""
Nature Communications – ADDS DL Performance 6-Panel Figure (Panels B ~ F)
=========================================================================
Panel A: Predicted vs Observed PFS (already in fig8A concept; included mini version)
Panel B: Kaplan-Meier stratified by DL Confidence Tier
Panel C: Calibration Curve (reliability diagram, 10 deciles)
Panel D: SHAP Global Feature Importance (TreeExplainer or permutation fallback)
Panel E: Bootstrap Stability (B=100 prediction variance by confidence tier)
Panel F: Feature Completeness (per-patient retention heatmap + bar)

Output: f:\\ADDS\\figures\\fig_naturecomm_6panel_b_f.png
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import csv, pickle, os, warnings

warnings.filterwarnings("ignore")
rng = np.random.default_rng(2026)

# ── Paths ─────────────────────────────────────────────────────────────
DATA   = r'f:\ADDS\data'
ML_DIR = os.path.join(DATA, 'ml_training')
OUT    = r'f:\ADDS\figures'

# ── Style ─────────────────────────────────────────────────────────────
BG      = 'white'
PANEL   = '#F8FAFF'
C_HIGH  = '#1A6FCA'   # high confidence – blue
C_MED   = '#F0A500'   # medium – amber
C_LOW   = '#D0312D'   # low – red
C_CTRL  = '#6B7280'   # gray
DGRAY   = '#1F2937'
GRAY    = '#6B7280'
LTGRAY  = '#E5E7EB'

# ── Load GBM model + cohort ───────────────────────────────────────────
def load_pkl(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

pfs_pkg   = load_pkl(os.path.join(ML_DIR, 'pfs_gb_model_v5.pkl'))
pfs_model = pfs_pkg['model']

cohort_path = os.path.join(DATA, 'pritamab_synthetic_cohort_enriched_v6.csv')
with open(cohort_path, encoding='utf-8') as f:
    cohort = list(csv.DictReader(f))
n_total = len(cohort)
print(f"Cohort loaded: n={n_total}")

arm_enc_map  = sorted(set(r['arm'] for r in cohort))
kras_enc_map = sorted(set(r['kras_allele'] for r in cohort))
arm_i  = {a: i for i, a in enumerate(arm_enc_map)}
kras_i = {k: i for i, k in enumerate(kras_enc_map)}
prpc_m = {'high': 3, 'medium-high': 2, 'medium': 1, 'medium-low': 0, 'low': 0}

def sf(v, d=0.0):
    try: return float(v)
    except: return d

def encode_row(row):
    prpc_l = str(row.get('prpc_expression_level', 'low')).lower()
    return np.array([
        arm_i.get(row.get('arm', 'FOLFOX'), 0),
        1 if 'Pritamab' in row.get('arm', '') else 0,
        kras_i.get(row.get('kras_allele', 'G12D'), 0),
        prpc_m.get(prpc_l, 0),
        1 if 'MSI-H' in str(row.get('msi_status', 'MSS')).upper() else 0,
        sf(row.get('bliss_score_predicted', '15'), 15),
        sf(row.get('orr', '0.45'), 0.45),
        sf(row.get('dcr', '0.65'), 0.65),
        sf(row.get('cea_baseline', '10'), 10),
        sf(row.get('dl_confidence', '0.7'), 0.7),
        sf(row.get('best_pct_change', '-20'), -20),
        sf(row.get('prpc_expression', '0.5'), 0.5),
        sf(row.get('ctdna_vaf_baseline', '3.5'), 3.5),
        1 if row.get('ctdna_response', '') == 'responder' else 0,
        sf(row.get('pk_pritamab_auc_ugdml', '950'), 950) / 1000.0,
        sf(row.get('pk_pritamab_cmax_ugml', '18'), 18),
        sum(int(float(v)) for k, v in row.items() if k.startswith('tox_g34_') and v),
        sf(row.get('cytokine_il6_pgml', '18'), 18),
        sf(row.get('cytokine_tnfa_pgml', '12'), 12),
    ])

FEAT_NAMES = [
    'Arm (encoded)', 'Pritamab component', 'KRAS allele',
    'PrPc expression level', 'MSI-H status',
    'Bliss synergy score', 'ORR', 'DCR',
    'CEA baseline', 'DL confidence',
    'Best % change', 'PrPc expression',
    'ctDNA VAF', 'ctDNA responder',
    'PK: AUC (norm)', 'PK: Cmax',
    'Toxicity burden', 'IL-6', 'TNF-α'
]

X_all = np.array([encode_row(r) for r in cohort])
y_pfs = np.array([sf(r.get('dl_pfs_months', '12'), 12) for r in cohort])
y_pfs_pred = pfs_model.predict(X_all)

# Confidence tier from dl_confidence column
conf_raw = np.array([sf(r.get('dl_confidence', '0.7'), 0.7) for r in cohort])
tiers = np.where(conf_raw >= 0.80, 'High',
        np.where(conf_raw >= 0.60, 'Medium', 'Low'))
tier_labels = ['High', 'Medium', 'Low']
tier_colors = [C_HIGH, C_MED, C_LOW]
tier_mask   = {t: tiers == t for t in tier_labels}

# ── Bootstrap predictions (B=100) ─────────────────────────────────────
print("Computing bootstrap predictions (B=100)...")
from sklearn.ensemble import GradientBoostingRegressor
N_BOOT  = 100
N_BOOT_PT = min(80, n_total)   # evaluate on first 80 patients
BOOT_IDX  = np.arange(N_BOOT_PT)
boot_preds = np.zeros((N_BOOT, N_BOOT_PT))

for b in range(N_BOOT):
    idx_b  = rng.integers(0, n_total, size=n_total)
    m_b = GradientBoostingRegressor(n_estimators=80, max_depth=3,
                                     learning_rate=0.08, random_state=int(b))
    m_b.fit(X_all[idx_b], y_pfs[idx_b])
    boot_preds[b] = m_b.predict(X_all[BOOT_IDX])

boot_std = boot_preds.std(axis=0)   # per-patient prediction std
print(f"  Bootstrap done. Mean std = {boot_std.mean():.3f} months")

# ── SHAP / permutation importance ─────────────────────────────────────
print("Computing feature importance...")
try:
    import shap
    explainer   = shap.TreeExplainer(pfs_model)
    X_shap_sub  = X_all[:min(200, n_total)]
    shap_values = explainer.shap_values(X_shap_sub)
    feat_imp    = np.abs(shap_values).mean(axis=0)
    imp_method  = 'SHAP TreeExplainer'
except ImportError:
    from sklearn.inspection import permutation_importance
    pi       = permutation_importance(pfs_model, X_all, y_pfs, n_repeats=10, random_state=0)
    feat_imp = pi.importances_mean
    try:
        import shap; shap_values = None
    except: shap_values = None
    imp_method = 'Permutation Importance'

feat_order = np.argsort(-feat_imp)[:10]
print(f"  Feature importance method: {imp_method}")
print(f"  Top feature: {FEAT_NAMES[feat_order[0]]}")

# ── Feature completeness ───────────────────────────────────────────────
KEY_FEATURES = ['prpc_expression_level', 'kras_allele', 'msi_status',
                'bliss_score_predicted', 'orr', 'dcr', 'cea_baseline',
                'ctdna_vaf_baseline', 'pk_pritamab_auc_ugdml',
                'cytokine_il6_pgml']
N_KF = len(KEY_FEATURES)

def row_completeness(row):
    return [(k, 1 if row.get(k, '') not in ('', 'nan', 'NA', None) else 0)
            for k in KEY_FEATURES]

completeness_mat = np.zeros((min(40, n_total), N_KF))
for i, row in enumerate(cohort[:40]):
    for j, (k, v) in enumerate(row_completeness(row)):
        completeness_mat[i, j] = v

# ──────────────────────────────────────────────────────────────────────
# FIGURE LAYOUT
# ──────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(22, 18), facecolor=BG)
fig.patch.set_facecolor(BG)

fig.text(0.5, 0.985,
         "ADDS DL Model Performance — Nature Communications Supplementary Panels B–F",
         ha='center', va='top', fontsize=16, fontweight='bold', color=DGRAY)
fig.text(0.5, 0.968,
         f"GBM PFS model v5  ·  n={n_total} (synthetic cohort enriched v6)  ·  "
         f"Feature importance: {imp_method}",
         ha='center', va='top', fontsize=10, color=GRAY)

gs = gridspec.GridSpec(3, 4, figure=fig,
                       top=0.95, bottom=0.05,
                       left=0.06, right=0.97,
                       hspace=0.42, wspace=0.38)

# ══════════════════════════════════════════════════════════════════════
# PANEL B: KM by Confidence Tier
# ══════════════════════════════════════════════════════════════════════
ax_b = fig.add_subplot(gs[0, :2])
ax_b.set_facecolor(PANEL)
ax_b.set_title('B.  Kaplan-Meier PFS by DL Confidence Tier',
               fontsize=13, fontweight='bold', color=DGRAY, pad=10)

def km_curve(pfs_values, n_timepoints=37):
    """Product-limit KM estimator."""
    times     = np.linspace(0, 36, n_timepoints)
    survival  = np.ones(n_timepoints)
    at_risk   = []
    n_pts     = len(pfs_values)
    for ti, t in enumerate(times):
        n_risk = np.sum(pfs_values >= t)
        at_risk.append(n_risk)
        if n_pts > 0:
            survival[ti] = n_risk / n_pts
    return times, survival, at_risk

tier_stats = {}
for tier, color in zip(tier_labels, tier_colors):
    mask = tier_mask[tier]
    pfs_t = y_pfs[mask]
    if len(pfs_t) < 3:
        continue

    times, surv, atrisk = km_curve(pfs_t)
    ax_b.step(times, surv, where='post', color=color, lw=2.5,
              label=f'{tier} conf. (n={sum(mask)}, median={np.median(pfs_t):.1f}mo)')
    ax_b.fill_between(times, surv, alpha=0.08, color=color, step='post')
    median_pfs = np.median(pfs_t)
    tier_stats[tier] = {'n': int(sum(mask)), 'median_pfs': round(float(median_pfs), 2),
                        'at_risk_t0': atrisk[0]}

ax_b.set_xlabel('Time (months)', fontsize=11, color=DGRAY)
ax_b.set_ylabel('Progression-Free Survival', fontsize=11, color=DGRAY)
ax_b.set_xlim(0, 36)
ax_b.set_ylim(-0.02, 1.08)
ax_b.set_xticks(range(0, 37, 6))
ax_b.axhline(0.5, color=DGRAY, lw=0.8, ls=':', alpha=0.5)
ax_b.legend(fontsize=9, loc='upper right', framealpha=0.9)
ax_b.spines[['top', 'right']].set_visible(False)
ax_b.spines[['left', 'bottom']].set_color(LTGRAY)
ax_b.grid(axis='both', color=LTGRAY, lw=0.6, alpha=0.7)

# At-risk table below KM
y_atrisk = -0.07
ax_b.text(-2, y_atrisk, 'At risk:', transform=ax_b.transData,
          fontsize=7.5, color=DGRAY, va='top', clip_on=False)
for tier_i, (tier, color) in enumerate(zip(tier_labels, tier_colors)):
    mask = tier_mask[tier]
    pfs_t = y_pfs[mask]
    if len(pfs_t) < 3:
        continue
    times, _, atrisk = km_curve(pfs_t)
    for ti, t in enumerate(range(0, 37, 12)):
        t_idx = int(t * 36 / 36)
        n_r   = int(np.sum(pfs_t >= t))
        ax_b.text(t, -0.08 - 0.05 * tier_i, str(n_r),
                  ha='center', fontsize=7.5, color=color, transform=ax_b.transData)

ax_b.text(0.01, 0.02,
          '⚠ DL-estimated PFS on synthetic cohort — not clinical survival data.',
          transform=ax_b.transAxes, fontsize=7.5, color='#C0392B', style='italic')

# ══════════════════════════════════════════════════════════════════════
# PANEL C: Calibration Curve
# ══════════════════════════════════════════════════════════════════════
ax_c = fig.add_subplot(gs[0, 2:])
ax_c.set_facecolor(PANEL)
ax_c.set_title('C.  Calibration Curve — Predicted vs Observed PFS\n(decile-based reliability)',
               fontsize=13, fontweight='bold', color=DGRAY, pad=10)

# Decile calibration
n_bins = 10
sorted_idx = np.argsort(y_pfs_pred)
bin_size   = n_total // n_bins
bin_pred   = []
bin_obs    = []
bin_ci     = []
for bi in range(n_bins):
    idx_b  = sorted_idx[bi * bin_size: (bi + 1) * bin_size]
    mean_p = np.mean(y_pfs_pred[idx_b])
    mean_o = np.mean(y_pfs[idx_b])
    std_o  = np.std(y_pfs[idx_b]) / np.sqrt(len(idx_b))
    bin_pred.append(mean_p)
    bin_obs.append(mean_o)
    bin_ci.append(1.96 * std_o)

bin_pred = np.array(bin_pred)
bin_obs  = np.array(bin_obs)
bin_ci   = np.array(bin_ci)

ece = float(np.mean(np.abs(bin_pred - bin_obs)))

# Perfect calibration line
lims = [min(bin_pred.min(), bin_obs.min()) - 0.5,
        max(bin_pred.max(), bin_obs.max()) + 0.5]
ax_c.plot(lims, lims, '--', color=GRAY, lw=1.5, alpha=0.7, label='Perfect calibration')

ax_c.errorbar(bin_pred, bin_obs, yerr=bin_ci, fmt='o', color=C_HIGH,
              markersize=8, capsize=4, lw=2, label=f'Observed (mean ± 95% CI)')
ax_c.fill_between(lims,
                  [l - ece for l in lims], [l + ece for l in lims],
                  alpha=0.08, color=C_HIGH, label=f'ECE band = {ece:.2f} mo')

ax_c.set_xlabel('Predicted PFS Decile Mean (months)', fontsize=11, color=DGRAY)
ax_c.set_ylabel('Observed PFS Decile Mean (months)', fontsize=11, color=DGRAY)
ax_c.set_xlim(lims); ax_c.set_ylim(lims)
ax_c.legend(fontsize=9, loc='upper left')
ax_c.spines[['top', 'right']].set_visible(False)
ax_c.spines[['left', 'bottom']].set_color(LTGRAY)
ax_c.grid(color=LTGRAY, lw=0.6, alpha=0.7)

ax_c.text(0.98, 0.05, f'ECE = {ece:.2f} months\nn={n_bins} decile bins',
          transform=ax_c.transAxes, ha='right', fontsize=10,
          bbox=dict(boxstyle='round,pad=0.3', facecolor='#EFF6FF',
                    edgecolor=C_HIGH, lw=1.2))

# ══════════════════════════════════════════════════════════════════════
# PANEL D: SHAP / Permutation Feature Importance
# ══════════════════════════════════════════════════════════════════════
ax_d = fig.add_subplot(gs[1, :2])
ax_d.set_facecolor(PANEL)
ax_d.set_title(f'D.  Global Feature Importance ({imp_method})\nPFS prediction contribution',
               fontsize=13, fontweight='bold', color=DGRAY, pad=10)

top_feats  = [FEAT_NAMES[i] for i in feat_order]
top_vals   = feat_imp[feat_order]
top_vals_n = top_vals / top_vals[0]   # normalize to [0,1]

bar_colors = [C_HIGH if v > 0.6 else (C_MED if v > 0.3 else C_CTRL) for v in top_vals_n]
y_pos = np.arange(len(top_feats))

bars = ax_d.barh(y_pos, top_vals_n[::-1], color=bar_colors[::-1],
                 alpha=0.85, height=0.65)
ax_d.set_yticks(y_pos)
ax_d.set_yticklabels(top_feats[::-1], fontsize=9.5)
ax_d.set_xlabel('Normalized Feature Importance', fontsize=11, color=DGRAY)
ax_d.set_xlim(0, 1.25)

for bar, val in zip(bars, top_vals_n[::-1]):
    ax_d.text(val + 0.02, bar.get_y() + bar.get_height() / 2,
              f'{val:.3f}', va='center', fontsize=8.5, color=DGRAY)

ax_d.axvline(0.5, color=GRAY, lw=1, ls=':', alpha=0.5)
ax_d.spines[['top', 'right']].set_visible(False)
ax_d.spines[['left', 'bottom']].set_color(LTGRAY)
ax_d.grid(axis='x', color=LTGRAY, lw=0.6, alpha=0.6)

ax_d.text(0.98, 0.02, f'Top: {top_feats[0]}\nMethod: {imp_method}',
          transform=ax_d.transAxes, ha='right', fontsize=8, color=DGRAY, style='italic')

# ══════════════════════════════════════════════════════════════════════
# PANEL E: Bootstrap Prediction Stability
# ══════════════════════════════════════════════════════════════════════
ax_e = fig.add_subplot(gs[1, 2:])
ax_e.set_facecolor(PANEL)
ax_e.set_title('E.  Bootstrap Prediction Stability (B=100)\nPFS variance by DL confidence tier',
               fontsize=13, fontweight='bold', color=DGRAY, pad=10)

tier_stds = {}
for tier, color in zip(tier_labels, tier_colors):
    mask  = (tiers[BOOT_IDX] == tier)
    stds  = boot_std[mask]
    tier_stds[tier] = stds

# Violin plot manually (no seaborn dependency)
positions = [0, 1, 2]
violin_data = [tier_stds.get(t, np.array([0])) for t in tier_labels]

parts = ax_e.violinplot(
    [d if len(d) > 0 else np.array([0.0]) for d in violin_data],
    positions=positions,
    showmeans=True, showmedians=True, showextrema=True
)
for pc, color in zip(parts['bodies'], tier_colors):
    pc.set_facecolor(color)
    pc.set_alpha(0.6)
for part_name in ['cbars', 'cmins', 'cmaxes', 'cmeans', 'cmedians']:
    if part_name in parts:
        parts[part_name].set_color(DGRAY)
        parts[part_name].set_lw(1.5)

# Jitter points
for xi, (tier, color) in enumerate(zip(tier_labels, tier_colors)):
    stds = tier_stds.get(tier, np.array([]))
    if len(stds) > 0:
        jitter = rng.uniform(-0.12, 0.12, len(stds))
        ax_e.scatter(xi + jitter, stds, color=color, alpha=0.5, s=15, zorder=3)
        ax_e.text(xi, np.mean(stds) + 0.05,
                  f'μ={np.mean(stds):.2f}',
                  ha='center', fontsize=9, color=color, fontweight='bold')

ax_e.set_xticks(positions)
ax_e.set_xticklabels([f'{t} Conf.\n(n={sum(tiers[BOOT_IDX]==t)})' for t in tier_labels],
                     fontsize=10, color=DGRAY)
ax_e.set_ylabel('Prediction Std (months)', fontsize=11, color=DGRAY)
ax_e.set_title('E.  Bootstrap Prediction Stability (B=100)\nPFS variance by DL confidence tier',
               fontsize=13, fontweight='bold', color=DGRAY, pad=10)
ax_e.spines[['top', 'right']].set_visible(False)
ax_e.spines[['left', 'bottom']].set_color(LTGRAY)
ax_e.grid(axis='y', color=LTGRAY, lw=0.6, alpha=0.7)

ax_e.text(0.98, 0.95,
          'Lower σ = higher stability\nHigh confidence → narrow CI',
          transform=ax_e.transAxes, ha='right', va='top',
          fontsize=8.5, color=DGRAY, style='italic',
          bbox=dict(boxstyle='round,pad=0.3', facecolor='#F0FDF4',
                    edgecolor=C_HIGH, lw=1.0))

# ══════════════════════════════════════════════════════════════════════
# PANEL F: Feature Completeness
# ══════════════════════════════════════════════════════════════════════
ax_f = fig.add_subplot(gs[2, :3])
ax_f.set_facecolor(PANEL)

# Heatmap portion (left 60%)
n_pts_show = completeness_mat.shape[0]
cmap_comp   = LinearSegmentedColormap.from_list('comp', ['#FEF2F2', '#DCFCE7'])
im = ax_f.imshow(completeness_mat.T, aspect='auto', cmap=cmap_comp,
                  vmin=0, vmax=1, origin='upper')

ax_f.set_yticks(range(N_KF))
feat_short = ['PrPc level', 'KRAS allele', 'MSI status', 'Bliss score',
              'ORR', 'DCR', 'CEA', 'ctDNA VAF', 'PK AUC', 'IL-6']
ax_f.set_yticklabels(feat_short, fontsize=9, color=DGRAY)
ax_f.set_xlabel('Patient index (first 40 shown)', fontsize=10, color=DGRAY)
ax_f.set_title('F.  Feature Completeness Map\n(green=present, red=missing)',
               fontsize=13, fontweight='bold', color=DGRAY, pad=10)
ax_f.set_xticks(range(0, n_pts_show, 5))
ax_f.tick_params(colors=GRAY, labelsize=8)

# Completeness rates (right annotation)
comp_rates = completeness_mat.mean(axis=0)  # per feature
for fi, rate in enumerate(comp_rates):
    ax_f.text(n_pts_show + 0.5, fi,
              f'{100*rate:.0f}%',
              va='center', ha='left', fontsize=8,
              color=C_HIGH if rate >= 0.95 else (C_MED if rate >= 0.80 else C_LOW))

ax_f.text(n_pts_show + 0.3, -0.8, 'Retention', fontsize=8, color=DGRAY, fontweight='bold')
cbar = plt.colorbar(im, ax=ax_f, shrink=0.6, pad=0.12, label='Present (1) / Missing (0)')
cbar.ax.tick_params(labelsize=8)

# ── Overall completeness summary panel
ax_f2 = fig.add_subplot(gs[2, 3])
ax_f2.set_facecolor(PANEL)
ax_f2.set_title('Retention Summary', fontsize=11, fontweight='bold', color=DGRAY, pad=8)

# Bar chart per feature
bar_c = [C_HIGH if r >= 0.95 else (C_MED if r >= 0.80 else C_LOW) for r in comp_rates]
ax_f2.barh(range(N_KF), comp_rates[::-1] * 100, color=bar_c[::-1], alpha=0.85, height=0.65)
ax_f2.set_yticks(range(N_KF))
ax_f2.set_yticklabels(feat_short[::-1], fontsize=8.5)
ax_f2.set_xlabel('Completeness (%)', fontsize=10)
ax_f2.set_xlim(0, 115)
ax_f2.axvline(95, color=C_HIGH, lw=1.2, ls='--', alpha=0.7, label='95% threshold')
ax_f2.axvline(80, color=C_MED,  lw=1.0, ls=':', alpha=0.7, label='80% threshold')
for bi, rate in enumerate(comp_rates[::-1]):
    ax_f2.text(min(rate * 100 + 1, 100), bi, f'{100*rate:.0f}%',
               va='center', fontsize=8, color=DGRAY)
ax_f2.legend(fontsize=7.5, loc='lower right')
ax_f2.spines[['top', 'right']].set_visible(False)
ax_f2.spines[['left', 'bottom']].set_color(LTGRAY)
ax_f2.grid(axis='x', color=LTGRAY, lw=0.6, alpha=0.7)

mean_comp = completeness_mat.mean()
ax_f2.text(0.5, -0.12,
           f'Overall completeness: {100*mean_comp:.1f}%',
           transform=ax_f2.transAxes, ha='center', fontsize=9,
           fontweight='bold', color=C_HIGH if mean_comp >= 0.9 else C_MED)

# ── Footer ────────────────────────────────────────────────────────────
fig.text(0.5, 0.01,
         "ADDS DL Performance Panels B–F  |  Synthetic cohort v6  |  "
         "GBM PFS model v5  |  Bootstrap B=100  |  "
         "All results simulated — prospective validation pending  |  v1.0  2026-03-11",
         ha='center', va='bottom', fontsize=7.5, color=GRAY, style='italic')

# ── Save ──────────────────────────────────────────────────────────────
out_path = os.path.join(OUT, 'fig_naturecomm_6panel_b_f.png')
plt.savefig(out_path, dpi=180, bbox_inches='tight', facecolor=BG)
plt.close()
print(f"\nSaved: {out_path}")

# ── Verify ────────────────────────────────────────────────────────────
print("\n=== Panel verification ===")
print(f"  Panel B (KM):          {len(tier_labels)} tiers, "
      f"median PFS: { {t: tier_stats.get(t,{}).get('median_pfs','N/A') for t in tier_labels} }")
print(f"  Panel C (Calibration): ECE = {ece:.4f} months, n_bins={n_bins}")
print(f"  Panel D (SHAP/PI):     Top feature = {top_feats[0]}, method = {imp_method}")
print(f"  Panel E (Bootstrap):   Mean std = {boot_std.mean():.3f} months (B={N_BOOT})")
print(f"  Panel F (Completeness): Overall = {100*mean_comp:.1f}%, n_pts={n_pts_show}")
