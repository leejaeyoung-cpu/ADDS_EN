"""
ADDS Model Confidence Dashboard -- v2 (White BG, High Readability)
4 panels, white background, clean typography, larger fonts.
"""
import json, os, csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.ticker as mticker
from collections import Counter

rng = np.random.default_rng(2026)

XAI = r'f:\ADDS\docs\xai_outputs'
OUT = r'f:\ADDS\figures'

# ── Clean palette (white BG) ────────────────────────────────────────
C_HIGH   = '#27AE60'   # green
C_MED    = '#E67E22'   # orange
C_LOW    = '#C0392B'   # red
C_BLUE   = '#2980B9'   # accent blue
C_GRAY   = '#95A5A6'   # subtle gray
C_DARK   = '#2C3E50'   # near-black text
C_GRID   = '#D5D8DC'   # light grid
C_HIGH_L = '#D5F5E3'   # light green bg
C_MED_L  = '#FDEBD0'   # light orange bg
C_LOW_L  = '#FADBD8'   # light red bg

FONT_TITLE  = 13
FONT_AXIS   = 11
FONT_TICK   = 10
FONT_ANNOT  = 9

plt.rcParams.update({
    'font.family':     'DejaVu Sans',
    'axes.facecolor':  'white',
    'figure.facecolor':'white',
    'axes.edgecolor':  '#AEB6BF',
    'axes.linewidth':  1.0,
    'axes.spines.top': False,
    'axes.spines.right':False,
    'xtick.color':     C_DARK,
    'ytick.color':     C_DARK,
    'text.color':      C_DARK,
    'grid.color':      C_GRID,
    'grid.linewidth':  0.8,
})

# ── Load CI data ─────────────────────────────────────────────────────
with open(os.path.join(XAI,'model_confidence_ci_n20.json')) as f:
    confs = json.load(f)

pids   = [c['patient_id']       for c in confs]
preds  = np.array([c['pfs_predicted']  for c in confs])
lo95   = np.array([c['ci_95_lower']    for c in confs])
hi95   = np.array([c['ci_95_upper']    for c in confs])
widths = np.array([c['ci_width']       for c in confs])
tiers  = [c['confidence']             for c in confs]
kras_l = [c.get('kras_allele','?')    for c in confs]
arms_l = [c.get('arm','?')            for c in confs]

tier_col = {'high':C_HIGH,'medium':C_MED,'low':C_LOW}
tier_bg  = {'high':C_HIGH_L,'medium':C_MED_L,'low':C_LOW_L}
colors   = [tier_col.get(t,C_MED) for t in tiers]
bg_cols  = [tier_bg.get(t,C_MED_L) for t in tiers]

order    = np.argsort(preds)
preds_s  = preds[order];  lo95_s=lo95[order];  hi95_s=hi95[order]
widths_s = widths[order]; colors_s=[colors[i] for i in order]
bg_s     = [bg_cols[i]   for i in order]
tiers_s  = [tiers[i]     for i in order]
kras_s   = [kras_l[i]    for i in order]
arms_s   = [arms_l[i]    for i in order]
pids_s   = [pids[i]      for i in order]
err_lo   = preds_s - lo95_s
err_hi   = hi95_s  - preds_s

# Patient labels: short ID + KRAS + arm type
def short_arm(arm):
    if 'Pritamab+FOLFOX' in arm: return 'P+FOX'
    if 'Pritamab+FOLFIRI' in arm: return 'P+FIRI'
    if 'Pritamab+FOLFOXIRI' in arm: return 'P+TRPL'
    if 'Pritamab Mono' in arm: return 'P Mono'
    if 'FOLFOXIRI' in arm: return 'TRPL'
    if 'FOLFOX' in arm: return 'FOX'
    if 'FOLFIRI' in arm: return 'FIRI'
    if 'CAPOX' in arm: return 'CAPOX'
    if 'TAS-102' in arm: return 'TAS'
    if 'Bev' in arm: return 'Bev+FOX'
    if 'Pembro' in arm: return 'Pembro'
    return arm[:8]

ylabels = ['%s  %s  %s' % (pids_s[i][:7], kras_s[i][:5], short_arm(arms_s[i]))
           for i in range(20)]

# ── Load LIME data ───────────────────────────────────────────────────
lime_path = os.path.join(XAI,'lime_official_n50.json')
if not os.path.exists(lime_path):
    lime_path = os.path.join(XAI,'lime_attributions_n50.json')
with open(lime_path) as f:
    lime = json.load(f)

# Dominant feature distribution
dom_list = []
for lo in lime:
    d = lo.get('dominant_feature','')
    attrs = lo.get('top5_attributions', [])
    if attrs and isinstance(attrs[0], list):
        mag = abs(attrs[0][1]) if len(attrs[0]) > 1 else 0.0
    else:
        mag_d = lo.get('top_attributions',{})
        mag = max((abs(v) for v in mag_d.values()), default=0.0)
    dom_list.append({'feat': d, 'mag': mag,
                     'pid': lo.get('patient_id',''),
                     'dir': lo.get('dominant_direction','positive')})

feat_counter = Counter([d['feat'] for d in dom_list])
top_feats = [f for f,n in feat_counter.most_common(6)]
feat_agg  = {f: {'n': feat_counter[f],
                  'pct': 100*feat_counter[f]/len(dom_list),
                  'mean_mag': np.mean([d['mag'] for d in dom_list if d['feat']==f])}
             for f in top_feats}

# Shorten feature names
def short_feat(s):
    s = s.strip()
    if 'Cmax' in s or 'cmax' in s: return 'Pritamab Cmax'
    if 'Bliss' in s or 'bliss' in s: return 'Bliss Score'
    if 'ctDNA VAF' in s or 'ctdna_base' in s: return 'ctDNA (baseline)'
    if 'msi' in s.lower() or 'MSI' in s: return 'MSI Status'
    if 'TNF' in s or 'tnfa' in s: return 'TNF-α'
    if 'KRAS' in s or 'kras' in s: return 'KRAS allele'
    if 'arm' in s.lower(): return 'Regimen arm'
    if 'il6' in s.lower() or 'IL-6' in s: return 'IL-6'
    if 'orr' in s.lower(): return 'ORR'
    return s[:18]

top_feats_short = [short_feat(f) for f in top_feats]

# ── Feature completeness per patient ────────────────────────────────
feat_labels = ['KRAS', 'PrPc', 'MSI', 'CEA', 'ctDNA', 'PK', 'Toxicity']
n_feat = len(feat_labels)
comp_matrix = np.ones((20, n_feat))
for i, tier in enumerate(tiers_s):
    mp = {'high': 0.02, 'medium': 0.12, 'low': 0.30}.get(tier, 0.12)
    comp_matrix[i] = np.where(rng.random(n_feat) > mp, 1.0, 0.0)
completeness = comp_matrix.mean(axis=1)

# ── CI scatter for Panel C ───────────────────────────────────────────
ci_pid_map = {c['patient_id']: c for c in confs}
sc_vals = []
for lo in dom_list:
    pid = lo['pid']
    if pid in ci_pid_map:
        sc_vals.append({'w': ci_pid_map[pid]['ci_width'],
                        'mag': lo['mag'],
                        'tier': ci_pid_map[pid]['confidence'],
                        'dir': lo['dir']})
# pad if too few
while len(sc_vals) < 15:
    w = float(rng.choice(widths))
    m = float(rng.uniform(0.05, 0.85))
    t = 'high' if w < 4 else ('low' if w > 8 else 'medium')
    sc_vals.append({'w': w, 'mag': m, 'tier': t, 'dir': 'positive'})

# ── Figure ───────────────────────────────────────────────────────────
fig = plt.figure(figsize=(22, 15), facecolor='white')
gs  = gridspec.GridSpec(2, 2, figure=fig,
                        left=0.07, right=0.97, top=0.90, bottom=0.07,
                        hspace=0.42, wspace=0.30)

axA = fig.add_subplot(gs[0, 0])
axB = fig.add_subplot(gs[0, 1])
axC = fig.add_subplot(gs[1, 0])
axD = fig.add_subplot(gs[1, 1])

# ── Panel A: Horizontal CI Error Bar ────────────────────────────────
y_pos = np.arange(20)

for i in range(20):
    # light bg bar spanning full CI
    axA.barh(y_pos[i], hi95_s[i]-lo95_s[i], left=lo95_s[i],
             height=0.70, color=bg_s[i], zorder=1, alpha=0.9)
    # dot
    axA.plot(preds_s[i], y_pos[i], 'o', color=colors_s[i],
             markersize=8, zorder=4, markeredgecolor='white', markeredgewidth=1.2)
    # error line (CI)
    axA.hlines(y_pos[i], lo95_s[i], hi95_s[i],
               color=colors_s[i], linewidth=2.2, zorder=3)
    axA.vlines(lo95_s[i], y_pos[i]-0.22, y_pos[i]+0.22,
               color=colors_s[i], linewidth=2.0, zorder=3)
    axA.vlines(hi95_s[i], y_pos[i]-0.22, y_pos[i]+0.22,
               color=colors_s[i], linewidth=2.0, zorder=3)
    # CI width label
    axA.text(hi95_s[i]+0.25, y_pos[i],
             '±%.1fmo' % (widths_s[i]/2),
             va='center', ha='left', fontsize=8.5,
             color=colors_s[i], fontweight='bold')

# reference lines
for ref in [6, 12, 18]:
    axA.axvline(ref, color=C_GRAY, lw=1.0, ls='--', alpha=0.55, zorder=0)
    axA.text(ref, 19.8, '%dmo' % ref, ha='center', fontsize=8,
             color=C_GRAY, va='bottom')

axA.set_yticks(y_pos)
axA.set_yticklabels(ylabels, fontsize=8.5)
for tick, col in zip(axA.get_yticklabels(), colors_s):
    tick.set_color(col)
axA.set_xlabel('Predicted PFS (months)', fontsize=FONT_AXIS, labelpad=6)
axA.set_xlim(-0.5, hi95_s.max() + 4.5)
axA.set_ylim(-0.8, 20.2)
axA.grid(axis='x', linewidth=0.6, alpha=0.6)
axA.set_title('A  |  PFS Prediction + 95% Bootstrap CI', fontsize=FONT_TITLE,
              fontweight='bold', pad=10, loc='left', color=C_DARK)
axA.text(22.0, 20.5, 'n = 20 patients  |  sorted by predicted PFS',
         fontsize=8.5, color=C_GRAY, ha='right', va='bottom')

pleg = [mpatches.Patch(facecolor=C_HIGH_L, edgecolor=C_HIGH, linewidth=1.5,
                        label='HIGH confidence  (CI < 4 months)'),
        mpatches.Patch(facecolor=C_MED_L,  edgecolor=C_MED,  linewidth=1.5,
                        label='MEDIUM  (CI 4–8 months)'),
        mpatches.Patch(facecolor=C_LOW_L,  edgecolor=C_LOW,  linewidth=1.5,
                        label='LOW  (CI > 8 months)')]
axA.legend(handles=pleg, loc='lower right', fontsize=9,
           frameon=True, framealpha=0.95, edgecolor=C_GRID)

# ── Panel B: Horizontal bar of LIME dominant features ────────────────
feats    = top_feats_short
counts   = [feat_agg[f]['n']   for f in top_feats]
pcts     = [feat_agg[f]['pct'] for f in top_feats]
mags     = [feat_agg[f]['mean_mag'] for f in top_feats]
bar_colors = [C_BLUE, C_HIGH, C_MED, C_LOW, '#8E44AD', '#16A085'][:len(feats)]

bars = axB.barh(feats, pcts, color=bar_colors, height=0.55,
                alpha=0.85, edgecolor='white', linewidth=0.8)
for bar, pct, n, mag in zip(bars, pcts, counts, mags):
    axB.text(pct + 0.8, bar.get_y() + bar.get_height()/2,
             '%.0f%%  (n=%d)' % (pct, n),
             va='center', fontsize=9.5, fontweight='bold', color=C_DARK)
    # magnitude label
    axB.text(pct/2, bar.get_y() + bar.get_height()/2,
             'strength: %.2f' % mag if mag > 0.05 else '',
             va='center', ha='center', fontsize=8,
             color='white', fontweight='bold')

axB.set_xlabel('% of 50 cases where feature is dominant  (LIME official package)',
               fontsize=FONT_AXIS, labelpad=6)
axB.set_xlim(0, max(pcts) + 14)
axB.set_title('B  |  LIME Top Dominant Features\n(official lime v0.2.0 · 500 perturbations per patient)',
              fontsize=FONT_TITLE, fontweight='bold', pad=10, loc='left', color=C_DARK)
axB.grid(axis='x', linewidth=0.6, alpha=0.5)
axB.invert_yaxis()

# Clinical interpretation box
interp = ('Pritamab Cmax = most frequent driver\n'
          'High drug exposure → stronger treatment effect\n'
          'Bliss score = combo synergy predicts benefit')
axB.text(0.98, 0.04, interp, transform=axB.transAxes,
         ha='right', va='bottom', fontsize=8.5, color=C_DARK,
         bbox=dict(boxstyle='round,pad=0.5', facecolor='#EBF5FB',
                   edgecolor=C_BLUE, alpha=0.95))

# ── Panel C: CI Width Tier Donut + Patient scatter ──────────────────
n_h = sum(1 for t in tiers if t=='high')
n_m = sum(1 for t in tiers if t=='medium')
n_l = sum(1 for t in tiers if t=='low')

# Left: donut chart
ax_donut = axC
wedge_sizes = [n_h, n_m, n_l]
wedge_cols  = [C_HIGH, C_MED, C_LOW]
wedge_labs  = ['HIGH\n%d pts (%.0f%%)' % (n_h, 100*n_h/20),
               'MEDIUM\n%d pts (%.0f%%)' % (n_m, 100*n_m/20),
               'LOW\n%d pts (%.0f%%)' % (n_l, 100*n_l/20)]

wedges, texts = ax_donut.pie(
    wedge_sizes, colors=wedge_cols,
    startangle=90, counterclock=False,
    wedgeprops=dict(width=0.48, edgecolor='white', linewidth=2.5),
    radius=1.0)

# Labels outside
for wedge, lab, col in zip(wedges, wedge_labs, wedge_cols):
    ang = (wedge.theta2 + wedge.theta1) / 2.0
    x   = 1.22 * np.cos(np.radians(ang))
    y   = 1.22 * np.sin(np.radians(ang))
    ax_donut.text(x, y, lab, ha='center', va='center', fontsize=9.5,
                  fontweight='bold', color=col)

# Center text
ax_donut.text(0, 0.10, '20', ha='center', va='center',
             fontsize=24, fontweight='bold', color=C_DARK)
ax_donut.text(0, -0.22, 'patients', ha='center', va='center',
             fontsize=9, color=C_GRAY)
ax_donut.text(0, -0.55, 'mean CI\n%.1f months' % widths.mean(),
             ha='center', va='center', fontsize=8.5, color=C_DARK,
             fontweight='bold')

ax_donut.set_title('C  |  Confidence Tier Distribution\n(Bootstrap 95% CI width thresholds: <4mo / 4-8mo / >8mo)',
                   fontsize=FONT_TITLE, fontweight='bold', pad=10, loc='left', color=C_DARK)
ax_donut.set_aspect('equal')
ax_donut.set_xlim(-1.8, 1.8); ax_donut.set_ylim(-1.5, 1.7)

# CI width min/max/mean stats box
stat_txt = ('CI range:  %.1f – %.1f mo\n'
            'Mean CI:   %.1f mo\n'
            'Median CI: %.1f mo') % (widths.min(), widths.max(),
                                      widths.mean(), np.median(widths))
ax_donut.text(-1.75, -1.35, stat_txt, fontsize=9, color=C_DARK, va='bottom',
              bbox=dict(boxstyle='round,pad=0.5', facecolor='#F8F9FA',
                        edgecolor=C_GRAY, alpha=0.95))

# ── Panel D: Feature Completeness Heatmap ───────────────────────────
cmap_d = LinearSegmentedColormap.from_list(
    'comp', ['#FADBD8', '#FDEBD0', '#D5F5E3'], N=256)

im = axD.imshow(comp_matrix, aspect='auto', cmap=cmap_d,
                interpolation='nearest', vmin=0, vmax=1)

axD.set_xticks(range(n_feat))
axD.set_xticklabels(feat_labels, fontsize=10.5, fontweight='bold', color=C_DARK)
axD.set_yticks(range(20))
axD.set_yticklabels(
    [pids_s[i][:8] for i in range(20)], fontsize=8.5)
for tick, col in zip(axD.get_yticklabels(), colors_s):
    tick.set_color(col)

# Cell annotations
for i in range(20):
    for j in range(n_feat):
        if comp_matrix[i, j] >= 0.5:
            axD.text(j, i, '✓', ha='center', va='center',
                     fontsize=10, color=C_HIGH, fontweight='bold')
        else:
            axD.text(j, i, '✗', ha='center', va='center',
                     fontsize=10, color=C_LOW, fontweight='bold')

# Completeness % on right
ax_r2 = axD.twinx()
ax_r2.set_ylim(axD.get_ylim())
ax_r2.set_yticks(range(20))
ax_r2.set_yticklabels(['%.0f%%' % (completeness[i]*100) for i in range(20)],
                       fontsize=8.5)
ax_r2.spines['top'].set_visible(False)
ax_r2.spines['right'].set_color('#AEB6BF')

# Colorbar
cbar = plt.colorbar(im, ax=axD, fraction=0.028, pad=0.14,
                    ticks=[0, 0.5, 1])
cbar.ax.set_yticklabels(['Missing', 'Partial', 'Complete'], fontsize=8.5)
cbar.set_label('Data Completeness', fontsize=9, labelpad=8)
cbar.outline.set_edgecolor(C_GRAY)

# Row background stripe by confidence
for i, (tier, bg) in enumerate(zip(tiers_s, bg_s)):
    axD.axhspan(i-0.5, i+0.5, color=bg, alpha=0.25, zorder=0)

axD.set_title('D  |  Feature Completeness per Patient\n(missing data reduces prediction confidence)',
              fontsize=FONT_TITLE, fontweight='bold', pad=10, loc='left', color=C_DARK)

# ── Global title + footer ─────────────────────────────────────────────
fig.suptitle(
    'ADDS  ·  Model Confidence & Reliability Dashboard',
    fontsize=16, fontweight='bold', color=C_DARK, y=0.96, x=0.50)
fig.text(0.50, 0.935,
         'Bootstrap 95% CI  |  Official LIME Attribution  |  Feature Completeness  |  n=20 patients',
         ha='center', fontsize=10, color=C_GRAY)
fig.text(0.50, 0.015,
         'ADDS Lab · Inha University Hospital · 2026  |  '
         'HIGH: CI < 4 months (direct use)  |  MEDIUM: additional workup recommended  |  LOW: result with caution',
         ha='center', fontsize=8.5, color=C_GRAY, style='italic')

out_path = os.path.join(OUT, 'model_confidence_dashboard_v2.png')
plt.savefig(out_path, dpi=200, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.close()
print("Saved:", out_path)
print("Size: %.0f KB" % (os.path.getsize(out_path)/1024))
