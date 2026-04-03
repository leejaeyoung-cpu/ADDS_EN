"""
Pritamab Combination Therapy Recommendation Figure
ADDS Framework - Pritamab-Centric Drug Combination Analysis
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patches as patches
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
import numpy as np

# ── Color palette ─────────────────────────────────────────────────────────────
BG_DARK      = '#0D1B2A'
BG_MID       = '#1A2B3C'
BG_PANEL     = '#1E3448'
ACCENT_CYAN  = '#00D4FF'
ACCENT_GOLD  = '#FFB800'
ACCENT_GREEN = '#00E676'
ACCENT_RED   = '#FF5252'
ACCENT_PURP  = '#CE93D8'
ACCENT_TEAL  = '#4DB6AC'
ACCENT_ORNG  = '#FF8A65'
WHITE        = '#FFFFFF'
GRAY_LIGHT   = '#B0BEC5'
TIER1_CLR    = '#0D47A1'
TIER2_CLR    = '#1565C0'
TIER3_CLR    = '#1976D2'

fig = plt.figure(figsize=(22, 30), facecolor=BG_DARK)
fig.patch.set_facecolor(BG_DARK)

# ── Main title ─────────────────────────────────────────────────────────────────
fig.text(0.5, 0.975, 'Pritamab-Centric Combination Therapy Recommendation',
         ha='center', va='top', fontsize=26, fontweight='bold',
         color=ACCENT_CYAN, fontfamily='DejaVu Sans')
fig.text(0.5, 0.958, 'ADDS AI-Driven Synergy Ranking | Anti-PrPc mAb × Standard-of-Care Oncology',
         ha='center', va='top', fontsize=14, color=GRAY_LIGHT)

# ═══════════════════════════════════════════════════════════════════════════════
# LAYOUT: 5 rows
# Row 0 : Patient Stratification (Biomarker Input)
# Row 1 : ADDS Processing Pipeline
# Row 2 : DRS Scores (main combination panel)
# Row 3 : EC50 Shift & Synergy Scores
# Row 4 : Primary / Alternative / Conditional recommendations (OUTPUT)
# ═══════════════════════════════════════════════════════════════════════════════

gs = gridspec.GridSpec(5, 3,
                       figure=fig,
                       left=0.04, right=0.97,
                       top=0.950, bottom=0.02,
                       hspace=0.55, wspace=0.30)

# ─────────────────────────────────────────────────────────────────────────────
# ROW 0 : TIER 1 – Biomarker / Patient Stratification
# ─────────────────────────────────────────────────────────────────────────────
ax_tier1 = fig.add_subplot(gs[0, :])
ax_tier1.set_facecolor(BG_PANEL)
ax_tier1.set_xlim(0, 1)
ax_tier1.set_ylim(0, 1)
ax_tier1.axis('off')
for spine in ax_tier1.spines.values():
    spine.set_visible(False)

# Tier 1 banner
tier1_rect = FancyBboxPatch((0.0, 0.0), 1.0, 1.0,
                             boxstyle="round,pad=0.01",
                             facecolor='#0A1929', edgecolor=ACCENT_CYAN, linewidth=2)
ax_tier1.add_patch(tier1_rect)

ax_tier1.text(0.01, 0.88, 'TIER 1 — PATIENT STRATIFICATION (Dual-Biomarker Required)',
              transform=ax_tier1.transAxes,
              fontsize=11, fontweight='bold', color=ACCENT_CYAN)

# 4 input blocks
blocks_t1 = [
    ('KRAS Mutation\nStatus', 'G12D / G12V / G13D\nG12C (sotorasib arm)\nNGS / Sanger confirmed', ACCENT_GOLD, 0.05),
    ('PrPc IHC H-score', 'Threshold: H-score ≥ 50\n8H4 antibody (1:200)\nPrPc-HIGH required', ACCENT_CYAN, 0.30),
    ('Cancer Type', 'CRC (74.5%) · PDAC (76%)\nGastric (68%) · LUAD (~45%)\nPrimary selection', ACCENT_GREEN, 0.55),
    ('Performance &\nECOG Status', 'ECOG 0–2 (all regimens)\nECOG 0–1 (triplet FOLFOXIRI)\nComorbidities screened', ACCENT_PURP, 0.78),
]
for title, body, color, x0 in blocks_t1:
    bx = FancyBboxPatch((x0, 0.08), 0.20, 0.72,
                         boxstyle="round,pad=0.02",
                         facecolor=BG_MID, edgecolor=color, linewidth=1.5)
    ax_tier1.add_patch(bx)
    ax_tier1.text(x0 + 0.10, 0.76, title,
                  ha='center', va='top', fontsize=9.5, fontweight='bold',
                  color=color, transform=ax_tier1.transAxes)
    ax_tier1.text(x0 + 0.10, 0.54, body,
                  ha='center', va='top', fontsize=8.2, color=WHITE,
                  transform=ax_tier1.transAxes, linespacing=1.5)

# ─────────────────────────────────────────────────────────────────────────────
# ROW 1 : TIER 2 – ADDS Processing Pipeline (horizontal flow)
# ─────────────────────────────────────────────────────────────────────────────
ax_pipe = fig.add_subplot(gs[1, :])
ax_pipe.set_facecolor('#0A1929')
ax_pipe.set_xlim(0, 1)
ax_pipe.set_ylim(0, 1)
ax_pipe.axis('off')

tier2_rect = FancyBboxPatch((0.0, 0.0), 1.0, 1.0,
                             boxstyle="round,pad=0.01",
                             facecolor='#0A1929', edgecolor=ACCENT_GOLD, linewidth=2)
ax_pipe.add_patch(tier2_rect)
ax_pipe.text(0.01, 0.88, 'TIER 2 — ADDS INTEGRATION ENGINE (Pritamab-Centric Processing)',
             fontsize=11, fontweight='bold', color=ACCENT_GOLD, transform=ax_pipe.transAxes)

pipeline_steps = [
    ('AI-Cellpose\nPathology', ACCENT_TEAL),
    ('AI-PRNP\nGenomics', ACCENT_CYAN),
    ('AI-nnU-Net\nCT Imaging', ACCENT_PURP),
    ('AI-XGBoost\nResponse', ACCENT_GOLD),
    ('AI-DeepSynergy\nSynergy', ACCENT_GREEN),
    ('ODE Energy\nModel ΔG', ACCENT_ORNG),
]
n = len(pipeline_steps)
step_w = 0.13
gap = (1.0 - n * step_w - 0.02) / (n - 1)
x_start = 0.01
for i, (label, color) in enumerate(pipeline_steps):
    x0 = x_start + i * (step_w + gap)
    bx = FancyBboxPatch((x0, 0.10), step_w, 0.65,
                         boxstyle="round,pad=0.015",
                         facecolor=BG_MID, edgecolor=color, linewidth=1.5)
    ax_pipe.add_patch(bx)
    ax_pipe.text(x0 + step_w / 2, 0.47, label,
                 ha='center', va='center', fontsize=8.5, fontweight='bold',
                 color=color, transform=ax_pipe.transAxes, linespacing=1.4)
    if i < n - 1:
        arrow_x = x0 + step_w + 0.005
        ax_pipe.annotate('', xy=(arrow_x + gap - 0.005, 0.42),
                         xytext=(arrow_x, 0.42),
                         xycoords='axes fraction', textcoords='axes fraction',
                         arrowprops=dict(arrowstyle='->', color=GRAY_LIGHT, lw=1.5))

# Ensemble result labels under pipeline
ensemble_labels = ['Random\nForest', 'Gradient\nBoosting', 'Deep\nMLP', 'ODE\nEnergy', 'DRS\nAggregation']
ens_x = [0.05, 0.24, 0.43, 0.62, 0.81]
for ens_lbl, ex in zip(ensemble_labels, ens_x):
    ax_pipe.text(ex + 0.065, 0.06, ens_lbl,
                 ha='center', va='bottom', fontsize=7.5, color=GRAY_LIGHT,
                 transform=ax_pipe.transAxes, alpha=0.8)

# ─────────────────────────────────────────────────────────────────────────────
# ROW 2 : DRS Ranking Heatmap / Bar
# ─────────────────────────────────────────────────────────────────────────────
ax_drs = fig.add_subplot(gs[2, :])
ax_drs.set_facecolor('#0A1929')
ax_drs.set_xlim(0, 1)
ax_drs.set_ylim(0, 1)
ax_drs.axis('off')

drs_rect = FancyBboxPatch((0.0, 0.0), 1.0, 1.0,
                           boxstyle="round,pad=0.01",
                           facecolor='#0A1929', edgecolor=ACCENT_GREEN, linewidth=2)
ax_drs.add_patch(drs_rect)
ax_drs.text(0.01, 0.93, 'DRUG RECOMMENDATION SCORE (DRS) — Pritamab Combination Ranking',
            fontsize=11, fontweight='bold', color=ACCENT_GREEN, transform=ax_drs.transAxes)

# DRS data from ADDS synergy scoring + paper data
# Bliss scores: [Paper] 5-FU=18.4, Oxaliplatin=21.7 confirmed
#               [ADDS] Irinotecan=17.3, Sotorasib=15.8 (corrected 2026-03-03)
#               [ADDS/est.] others
combos = [
    'Pritamab + FOLFOX\n(5-FU + Leucov. + Oxaliplatin)',
    'Pritamab + Sotorasib\n(KRAS G12C; covalent)',
    'Pritamab + FOLFIRI\n(5-FU + Leucov. + Irinotecan)',
    'Pritamab + Oxaliplatin\n(Monotherapy backbone)',
    'Pritamab + 5-FU\n(Monotherapy backbone)',
    'Pritamab + Sotorasib\n+ SHP2i (Triple)',
    'Pritamab + Bevacizumab\n(VEGF combination)',
    'Pritamab + FOLFOXIRI\n(Triplet intensification)',
]
drs_scores = [0.893, 0.882, 0.870, 0.856, 0.843, 0.835, 0.798, 0.784]
ec50_reductions = [24.0, 24.7, 24.5, 24.7, 24.7, 31.2, 18.5, 26.1]   # % (projected)
#                 Sotorasib: 15.8 (ADDS est., corrected from 22.5)
bliss_scores   = [21.0, 15.8, 19.8, 21.7, 18.4, 26.5, 14.2, 22.0]    # Bliss synergy

# Use an inner axis for horizontal bars
ax_inner = fig.add_axes([0.06, 0.495, 0.88, 0.145])  # [left, bottom, width, height]
ax_inner.set_facecolor('#0A1929')
ax_inner.spines['top'].set_visible(False)
ax_inner.spines['right'].set_visible(False)
ax_inner.spines['left'].set_visible(False)
ax_inner.spines['bottom'].set_color(GRAY_LIGHT)

n_combos = len(combos)
y_positions = np.arange(n_combos)

# color gradient by DRS score
norm = plt.Normalize(min(drs_scores) - 0.02, max(drs_scores) + 0.01)
cmap_custom = LinearSegmentedColormap.from_list('drs_cmap', [ACCENT_TEAL, ACCENT_GREEN, ACCENT_GOLD])
colors_bars = [cmap_custom(norm(s)) for s in drs_scores]

bars = ax_inner.barh(y_positions, drs_scores, height=0.65,
                     color=colors_bars, edgecolor='none', alpha=0.90)

for i, (bar, score, ec50, bliss) in enumerate(zip(bars, drs_scores, ec50_reductions, bliss_scores)):
    # DRS score label
    ax_inner.text(score + 0.002, i, f'{score:.3f}',
                  va='center', ha='left', fontsize=8.5, fontweight='bold',
                  color=ACCENT_GOLD)
    # EC50 reduction + Bliss tag
    tag = f'-ΔEC50: {ec50:.1f}%  |  Bliss: +{bliss:.1f}'
    ax_inner.text(score + 0.026, i, tag,
                  va='center', ha='left', fontsize=7.5, color=GRAY_LIGHT)

# Rank medals
rank_colors = {0: ACCENT_GOLD, 1: GRAY_LIGHT, 2: ACCENT_ORNG}
for i, combo in enumerate(combos):
    medal = f'#{i+1}'
    col = rank_colors.get(i, '#78909C')
    ax_inner.text(-0.005, i, medal,
                  va='center', ha='right', fontsize=9, fontweight='bold', color=col)

ax_inner.set_yticks(y_positions)
ax_inner.set_yticklabels(combos, fontsize=8.2, color=WHITE)
ax_inner.set_xlim(0.70, 0.98)
ax_inner.set_ylim(-0.5, n_combos - 0.5)
ax_inner.invert_yaxis()
ax_inner.set_xlabel('DRS Score (0–1)', color=GRAY_LIGHT, fontsize=9)
ax_inner.tick_params(colors=GRAY_LIGHT, labelsize=8)
ax_inner.axvline(0.75, color=ACCENT_RED, linestyle='--', linewidth=1.2, alpha=0.7)
ax_inner.text(0.752, n_combos - 0.3, 'Clinical\nThreshold 0.75',
              color=ACCENT_RED, fontsize=7.5, va='bottom')

# ─────────────────────────────────────────────────────────────────────────────
# ROW 3 : Synergy Score Radar + EC50 Shift Bars (2 panels)
# ─────────────────────────────────────────────────────────────────────────────
# Left panel: EC50 reduction per drug
ax_ec50 = fig.add_subplot(gs[3, 0])
ax_ec50.set_facecolor(BG_PANEL)

drugs = ['5-FU\n(12,000 nM)', 'Oxaliplatin\n(3,750 nM)', 'Irinotecan\n(7,500 nM)', 'Sotorasib\n(75 nM)']
ec50_alone  = [12000, 3750, 7500, 75]
ec50_pritmb = [9032, 2823, 5645, 56.5]
pct_red = [24.7, 24.7, 24.7, 24.7]
x_pos = np.arange(len(drugs))
width = 0.35

bars_alone  = ax_ec50.bar(x_pos - width/2, ec50_alone, width,
                           label='Monotherapy EC50', color=ACCENT_RED, alpha=0.75, edgecolor='none')
bars_combo  = ax_ec50.bar(x_pos + width/2, ec50_pritmb, width,
                           label='+ Pritamab EC50', color=ACCENT_CYAN, alpha=0.90, edgecolor='none')

for bar_a, bar_b in zip(bars_alone, bars_combo):
    h_a = bar_a.get_height()
    h_b = bar_b.get_height()
    ax_ec50.annotate('', xy=(bar_b.get_x() + bar_b.get_width()/2, h_b),
                     xytext=(bar_a.get_x() + bar_a.get_width()/2, h_a),
                     arrowprops=dict(arrowstyle='->', color=ACCENT_GOLD, lw=1.5))

ax_ec50.set_xticks(x_pos)
ax_ec50.set_xticklabels(drugs, fontsize=8, color=WHITE)
ax_ec50.set_ylabel('EC50 (nM)', color=GRAY_LIGHT, fontsize=9)
ax_ec50.set_title('EC50 Shift by Pritamab\n(24.7% Reduction, All Drugs)', color=ACCENT_CYAN, fontsize=10, fontweight='bold')
ax_ec50.legend(fontsize=7.5, facecolor=BG_DARK, labelcolor=WHITE, framealpha=0.8)
ax_ec50.set_facecolor(BG_PANEL)
ax_ec50.tick_params(colors=GRAY_LIGHT)
ax_ec50.spines['bottom'].set_color(GRAY_LIGHT)
ax_ec50.spines['left'].set_color(GRAY_LIGHT)
ax_ec50.spines['top'].set_visible(False)
ax_ec50.spines['right'].set_visible(False)
ax_ec50.text(0.99, 0.97, '−24.7%', transform=ax_ec50.transAxes,
             ha='right', va='top', fontsize=18, fontweight='bold', color=ACCENT_GOLD, alpha=0.3)

# Middle panel: ADDS 4-model consensus synergy
ax_syn = fig.add_subplot(gs[3, 1])
ax_syn.set_facecolor(BG_PANEL)

combo_names_short = ['+ FOLFOX', '+ Sotorasib', '+ FOLFIRI', '+ Oxali', '+ 5-FU', '+ Soto\n+SHP2i', '+ Bev', '+ FOLFOXIRI']
#                                  ^15.8 corrected                            ^18.4 [Paper]
bliss_vals  = [21.0, 15.8, 19.8, 21.7, 18.4, 26.5, 14.2, 22.0]
loewe_vals  = [1.34, 1.28, 1.31, 1.34, 1.34, 1.40, 1.22, 1.36]   # DRI values (Sotorasib corrected)
consensus_s = [0.893, 0.882, 0.870, 0.856, 0.843, 0.835, 0.798, 0.784]

colors_s = [cmap_custom(norm(s)) for s in consensus_s]
y_s = np.arange(len(combo_names_short))
barsyn = ax_syn.barh(y_s, bliss_vals, height=0.6,
                     color=colors_s, edgecolor='none', alpha=0.88)
ax_syn.axvline(10, color=ACCENT_RED, linestyle='--', linewidth=1.2, alpha=0.7)
ax_syn.text(10.2, len(combo_names_short) - 0.5, 'Synergy\nThreshold +10',
            color=ACCENT_RED, fontsize=7, va='top')
for i, (bv, lv) in enumerate(zip(bliss_vals, loewe_vals)):
    ax_syn.text(bv + 0.3, i, f'Bliss+{bv:.1f} | DRI {lv:.2f}',
                va='center', ha='left', fontsize=7.5, color=GRAY_LIGHT)

ax_syn.set_yticks(y_s)
ax_syn.set_yticklabels(combo_names_short, fontsize=8.5, color=WHITE)
ax_syn.invert_yaxis()
ax_syn.set_xlabel('Bliss Synergy Score', color=GRAY_LIGHT, fontsize=9)
ax_syn.set_title('Synergy Analysis\n(Bliss + Loewe Consensus)', color=ACCENT_PURP, fontsize=10, fontweight='bold')
ax_syn.set_facecolor(BG_PANEL)
ax_syn.tick_params(colors=GRAY_LIGHT)
ax_syn.spines['bottom'].set_color(GRAY_LIGHT)
ax_syn.spines['left'].set_color(GRAY_LIGHT)
ax_syn.spines['top'].set_visible(False)
ax_syn.spines['right'].set_visible(False)
ax_syn.set_xlim(0, 42)

# Right panel: Radar chart for top 3 combinations
ax_radar = fig.add_subplot(gs[3, 2], polar=True)
ax_radar.set_facecolor(BG_PANEL)
ax_radar.figure.patch.set_facecolor(BG_DARK)

categories = ['DRS\nScore', 'Bliss\nSynergy', 'EC50\nReduction', 'IP White\nSpace', 'Safety\nProfile', 'Clinical\nEvidence']
N = len(categories)
angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
angles += angles[:1]

# Normalised radar values for top 3 combos (0-1 scale)
top3_data = {
    'Pritamab + FOLFOX':    [0.95, 0.85, 0.90, 0.96, 0.92, 0.88],
    'Pritamab + Sotorasib': [0.93, 0.90, 0.90, 0.96, 0.85, 0.82],
    'Pritamab + FOLFIRI':   [0.91, 0.82, 0.90, 0.96, 0.90, 0.85],
}
top3_colors = [ACCENT_GOLD, ACCENT_CYAN, ACCENT_GREEN]

for (label, vals), color in zip(top3_data.items(), top3_colors):
    vals_plot = vals + vals[:1]
    ax_radar.plot(angles, vals_plot, 'o-', color=color, linewidth=2, label=label, markersize=4)
    ax_radar.fill(angles, vals_plot, alpha=0.12, color=color)

ax_radar.set_xticks(angles[:-1])
ax_radar.set_xticklabels(categories, fontsize=8, color=WHITE)
ax_radar.set_ylim(0, 1)
ax_radar.set_yticks([0.25, 0.5, 0.75, 1.0])
ax_radar.set_yticklabels(['0.25', '0.5', '0.75', '1.0'], color=GRAY_LIGHT, fontsize=6.5)
ax_radar.tick_params(colors=GRAY_LIGHT)
ax_radar.set_facecolor(BG_PANEL)
ax_radar.grid(color=GRAY_LIGHT, alpha=0.2)
ax_radar.spines['polar'].set_color(GRAY_LIGHT)
ax_radar.spines['polar'].set_alpha(0.3)
ax_radar.set_title('Top 3 Combinations\nMulti-Criteria Radar', color=ACCENT_TEAL, fontsize=9.5,
                   fontweight='bold', pad=12)
legend_r = ax_radar.legend(loc='upper right', bbox_to_anchor=(1.55, 1.15),
                            fontsize=7, facecolor=BG_DARK, labelcolor=WHITE, framealpha=0.7)

# ─────────────────────────────────────────────────────────────────────────────
# ROW 4 : OUTPUT – Recommendation Cards (3 regimens)
# ─────────────────────────────────────────────────────────────────────────────
rec_ax_list = [fig.add_subplot(gs[4, i]) for i in range(3)]

recommendations = [
    {
        'rank': '#1  PRIMARY RECOMMENDATION',
        'title': 'Pritamab + FOLFOX',
        'subtitle': '(5-FU + Leucovorin + Oxaliplatin)',
        'drs': 'DRS: 0.893',
        'bullets': [
            '● PrPc-HIGH / KRAS-mutant (any allele, esp. G12D/G12V)',
            '● Pi3K/AKT target: RPSA signalosome blockade',
            '● EC50 -24.0% ↓ FOLFOX dose → Toxicity reduction',
            '● Bliss Synergy: +21.0  |  DRI (5-FU): 1.34',
            '● FDA Guidelines · NCCN Protocol aligned',
            '● ECOG 0–2 · 1st/2nd line mCRC · Gastric · PDAC',
        ],
        'guideline': 'FDA Guidelines · NCCN Protocol',
        'border': ACCENT_GOLD,
        'badge_bg': ACCENT_GOLD,
    },
    {
        'rank': '#2  ALTERNATIVE REGIMEN',
        'title': 'Pritamab + Sotorasib',
        'subtitle': '(KRAS G12C Covalent Inhibitor)',
        'drs': 'DRS: 0.882',
        'bullets': [
            '● KRAS G12C-mutant (~12–13% CRC) · PrPc-HIGH',
            '● Dual-axis: RPSA-PrPc block + G12C direct inhibition',
            '● EC50 -24.7% ↓ Sotorasib  |  Bliss: +15.8 [ADDS est.]',
            '● Addresses RTK-bypass escape via RPSA route',
            '● Triple option: + SHP2i (DRS 0.835, Bliss +26.5)',
            '● EMA/FDA Approved sotorasib backbone',
        ],
        'guideline': 'FDA Approved · Clinical Trial Option',
        'border': ACCENT_CYAN,
        'badge_bg': ACCENT_CYAN,
    },
    {
        'rank': '#3  CONDITIONAL REGIMEN',
        'title': 'Pritamab + FOLFOXIRI',
        'subtitle': '(Triplet Intensification)',
        'drs': 'DRS: 0.784',
        'bullets': [
            '● KRAS-mutant · PrPc-HIGH · Adequate PS (ECOG 0–1)',
            '● Triplet intensification + PrPc sensitisation',
            '● Projected EC50 -26.1%  |  Bliss: +22.0',
            '● Hepatic metastasis conversion · Young patients',
            '● Requires Clinical Eligibility Review',
            '● Consider bevacizumab triplet variant (TRIBE2)',
        ],
        'guideline': 'ESMO Guideline · Eligibility Review Required',
        'border': ACCENT_GREEN,
        'badge_bg': ACCENT_GREEN,
    },
]

for ax_r, rec in zip(rec_ax_list, recommendations):
    ax_r.set_facecolor('#0A1929')
    ax_r.set_xlim(0, 1)
    ax_r.set_ylim(0, 1)
    ax_r.axis('off')

    # Border box
    card_border = FancyBboxPatch((0.0, 0.0), 1.0, 1.0,
                                  boxstyle="round,pad=0.02",
                                  facecolor='#0A1929', edgecolor=rec['border'], linewidth=2.5)
    ax_r.add_patch(card_border)

    # Rank badge
    badge = FancyBboxPatch((0.0, 0.87), 1.0, 0.13,
                            boxstyle="round,pad=0.01",
                            facecolor=rec['badge_bg'], edgecolor='none', alpha=0.9)
    ax_r.add_patch(badge)
    ax_r.text(0.5, 0.935, rec['rank'],
              ha='center', va='center', fontsize=9, fontweight='bold',
              color=BG_DARK, transform=ax_r.transAxes)

    # Title
    ax_r.text(0.5, 0.82, rec['title'],
              ha='center', va='top', fontsize=13, fontweight='bold',
              color=rec['border'], transform=ax_r.transAxes)
    ax_r.text(0.5, 0.74, rec['subtitle'],
              ha='center', va='top', fontsize=8.5, color=WHITE, transform=ax_r.transAxes)

    # DRS badge
    drs_badge = FancyBboxPatch((0.32, 0.64), 0.36, 0.075,
                                boxstyle="round,pad=0.01",
                                facecolor=BG_MID, edgecolor=rec['border'], linewidth=1.2)
    ax_r.add_patch(drs_badge)
    ax_r.text(0.5, 0.678, rec['drs'],
              ha='center', va='center', fontsize=10, fontweight='bold',
              color=ACCENT_GOLD, transform=ax_r.transAxes)

    # Bullet points
    y_text = 0.60
    for bullet in rec['bullets']:
        ax_r.text(0.04, y_text, bullet,
                  ha='left', va='top', fontsize=7.5, color=WHITE,
                  transform=ax_r.transAxes, linespacing=1.3)
        y_text -= 0.085

    # Guideline footer
    footer = FancyBboxPatch((0.02, 0.018), 0.96, 0.06,
                             boxstyle="round,pad=0.01",
                             facecolor=BG_MID, edgecolor=rec['border'], linewidth=0.8, alpha=0.7)
    ax_r.add_patch(footer)
    ax_r.text(0.5, 0.047, rec['guideline'],
              ha='center', va='center', fontsize=7.5, color=GRAY_LIGHT,
              transform=ax_r.transAxes, style='italic')

# ─────────────────────────────────────────────────────────────────────────────
# Bottom footer bar
# ─────────────────────────────────────────────────────────────────────────────
fig.text(0.5, 0.007,
         'ADDS-AI Framework v5.3 · PrPc/PRNP Target Validation · 4-Model Consensus (Bliss + Loewe + HSA + ZIP) · '
         'IHC Dual-Biomarker Selection (PrPc H-score ≥50 + KRAS mutation)  |  Inha University Hospital · January 2026',
         ha='center', va='bottom', fontsize=7.5, color=GRAY_LIGHT, alpha=0.7)

# Save figure
plt.savefig(r'f:\ADDS\figures\pritamab_combination_recommendation.png',
            dpi=180, bbox_inches='tight', facecolor=BG_DARK,
            edgecolor='none')
print('Figure saved.')
plt.close()
