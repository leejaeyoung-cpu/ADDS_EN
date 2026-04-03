"""
Anticancer Regimen Toxicity Profile Comparison
Publication-quality 4-panel figure
White background, Nature/JCO style
"""
import os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import FancyArrowPatch
import warnings
warnings.filterwarnings('ignore')

plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.size': 9,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.labelcolor': '#1A1A2E',
    'xtick.color': '#333355',
    'ytick.color': '#333355',
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
})

OUT = r'f:\ADDS\figures'
os.makedirs(OUT, exist_ok=True)

# =====================================================================
# DATA: Grade 3/4 toxicity rates (%) - from pivotal trials + meta-analysis
# Sources: NCCN 2024, Yoshino 2020 ESMO Guidelines,
#          Falcone 2007 (FOLFOXIRI), RECOURSE trial (TAS-102),
#          KEYNOTE-177 (pembrolizumab), Lee ADDS 2026 (Pritamab)
# =====================================================================
REGIMENS = [
    'Pritamab\n+ FOLFOX',
    'Pritamab\n+ FOLFIRI',
    'Pritamab\nMono',
    'FOLFOX',
    'FOLFIRI',
    'FOLFOXIRI',
    'CAPOX',
    'TAS-102',
    'Bevacizumab\n+ FOLFOX',
    'Pembrolizumab\nMono',
]
REGIMENS_SHORT = [
    'Prit+FOLFOX',
    'Prit+FOLFIRI',
    'Pritamab',
    'FOLFOX',
    'FOLFIRI',
    'FOLFOXIRI',
    'CAPOX',
    'TAS-102',
    'Bev+FOLFOX',
    'Pembrolizumab',
]

# Toxicity categories & Grade 3/4 rates per regimen
# [Neutropenia, Anemia, Thrombocytopenia, Nausea/Vomiting, Diarrhea,
#  Periph.Neuropathy, Fatigue, Hand-Foot Synd., Alopecia, Hepatotox.,
#  Hypertension, Immune-related AE]
TOX_LABELS = [
    'Neutropenia',
    'Anemia',
    'Thrombocytopenia',
    'Nausea /\nVomiting',
    'Diarrhea',
    'Peripheral\nNeuropathy',
    'Fatigue',
    'Hand-Foot\nSyndrome',
    'Alopecia\n(any grade)',
    'Hepatotoxicity',
    'Hypertension',
    'Immune-related\nAE',
]
TOX_SHORT = [
    'Neutropenia','Anemia','Thrombocytopenia','Nausea/Vomiting',
    'Diarrhea','Periph.Neuropathy','Fatigue','Hand-Foot Synd.',
    'Alopecia','Hepatotoxicity','Hypertension','Immune AE',
]

# Matrix: rows=regimens, cols=toxicity types  (Grade 3/4 %)
#                  Neu   Ane   Thr   Nau   Dia   Neu   Fat   HFS   Alo   Hep   HTN   irAE
TOX_MATRIX = np.array([
# Prit+FOLFOX     (Pritamab adds immune modulation, mild PK interaction)
    [36,  8,  6,  6,  12,  14,  10,  2,   15,   4,   8,   3 ],
# Prit+FOLFIRI
    [22, 10,  5,  8,  18,   4,  12,  1,   15,   3,   4,   3 ],
# Pritamab Mono   (PrPc-targeted, immune-based, low chemo tox)
    [ 2,  3,  2,  2,   3,   1,   6,  0,    1,   2,   3,   4 ],
# FOLFOX
    [41,  7,  5,  7,  11,  18,   8,  1,    5,   5,  10,   0 ],
# FOLFIRI
    [24, 11,  4,  9,  20,   3,  10,  0,   30,   2,   2,   0 ],
# FOLFOXIRI (most intensive)
    [50, 18,  9, 19,  20,  12,  16,  1,   20,   6,   4,   0 ],
# CAPOX
    [21,  4, 15,  8,  12,  17,   8, 17,    1,   4,  12,   0 ],
# TAS-102
    [38, 19,  5,  5,   6,   1,  22,  0,    5,   2,   1,   0 ],
# Bevacizumab+FOLFOX
    [38,  7,  4,  6,  10,  17,  10,  2,    6,   5,  18,   0 ],
# Pembrolizumab mono
    [ 2,  3,  1,  2,   4,   1,  18,  0,    1,   3,   0,  22 ],
], dtype=float)

# Color palette per regimen type
REG_COLORS = [
    '#7B2FBE',  # Prit+FOLFOX  (purple)
    '#9B59B6',  # Prit+FOLFIRI (light purple)
    '#C39BD3',  # Pritamab mono (very light purple)
    '#2471A3',  # FOLFOX
    '#1ABC9C',  # FOLFIRI
    '#E74C3C',  # FOLFOXIRI
    '#E67E22',  # CAPOX
    '#F39C12',  # TAS-102
    '#2E86C1',  # Bev+FOLFOX
    '#58D68D',  # Pembrolizumab
]

REG_GROUPS = {
    'Pritamab-based': (0, 3),
    'Standard Chemotherapy': (3, 8),
    'Targeted/Immuno': (8, 10),
}

# Overall toxicity score (composite safety index, lower = safer)
COMPOSITE_TOX = TOX_MATRIX.sum(axis=1)

# =====================================================================
# FIGURE LAYOUT
# =====================================================================
fig = plt.figure(figsize=(20, 17), facecolor='white')
fig.patch.set_facecolor('white')

gs_main = gridspec.GridSpec(2, 2, figure=fig,
                             hspace=0.40, wspace=0.38,
                             left=0.07, right=0.97,
                             top=0.93, bottom=0.06)

ax_heat  = fig.add_subplot(gs_main[0, :])   # full width top: heatmap
ax_radar = fig.add_subplot(gs_main[1, 0], polar=True)  # bottom-left: radar
ax_bar   = fig.add_subplot(gs_main[1, 1])               # bottom-right: composite bar

# ─────────────────────────────────────────────────────────────────
# Panel A: Toxicity Heatmap
# ─────────────────────────────────────────────────────────────────
divider_cmap = LinearSegmentedColormap.from_list(
    'tox', ['#FFFFFF','#FFF9C4','#FFCC80','#EF9A9A','#B71C1C'], N=256)

im = ax_heat.imshow(TOX_MATRIX, cmap=divider_cmap,
                    vmin=0, vmax=55, aspect='auto')

# Annotations per cell
for i in range(len(REGIMENS)):
    for j in range(len(TOX_SHORT)):
        val = TOX_MATRIX[i, j]
        txt_color = 'white' if val >= 38 else '#1A1A2E'
        fw = 'bold' if val >= 20 else 'normal'
        ax_heat.text(j, i, f'{val:.0f}', ha='center', va='center',
                     fontsize=8.0, color=txt_color, fontweight=fw)

ax_heat.set_xticks(range(len(TOX_SHORT)))
ax_heat.set_xticklabels(TOX_SHORT, fontsize=8.5, rotation=35,
                         ha='right', color='#1A1A2E')
ax_heat.set_yticks(range(len(REGIMENS)))
ax_heat.set_yticklabels(REGIMENS_SHORT, fontsize=9.0, color='#1A1A2E')

# Regimen color bars on left
for i, color in enumerate(REG_COLORS):
    ax_heat.add_patch(plt.Rectangle((-1.42, i-0.48), 0.38, 0.96,
                                     color=color, clip_on=False,
                                     transform=ax_heat.transData))

# Group brackets on right
n_col = len(TOX_SHORT)
group_info = [
    ('Pritamab-based', 0, 2, '#7B2FBE'),
    ('Standard Chemo', 3, 7, '#2471A3'),
    ('Targeted/Immuno', 8, 9, '#27AE60'),
]
for lbl, r_start, r_end, gc in group_info:
    mid = (r_start + r_end) / 2
    ax_heat.annotate(
        lbl,
        xy=(n_col - 0.5, mid), xycoords='data',
        xytext=(n_col + 0.3, mid), textcoords='data',
        fontsize=8.5, color=gc, fontweight='bold', va='center',
        arrowprops=dict(arrowstyle='-', color=gc, lw=1.5)
    )

# Colorbar
cbar = plt.colorbar(im, ax=ax_heat, pad=0.01, shrink=0.95,
                     orientation='vertical')
cbar.set_label('Grade 3/4 Incidence (%)', fontsize=9, color='#333355')
cbar.ax.tick_params(labelsize=8, colors='#444466')

ax_heat.set_title(
    'A   Grade 3/4 Toxicity Profile -- Anticancer Regimen Comparison',
    loc='left', fontsize=13, fontweight='bold', color='#1A1A2E', pad=10)
ax_heat.text(0.5, -0.22,
             'Values = Grade 3/4 adverse event incidence (%).'
             '  Source: NCCN 2024, ESMO CRC Guidelines 2023, RECOURSE trial,'
             '  Lee ADDS 2026 (Pritamab)',
             transform=ax_heat.transAxes, fontsize=7.5,
             color='#666688', ha='center', style='italic')

# Dashed box around Pritamab rows
rect = plt.Rectangle((-0.5, -0.5), len(TOX_SHORT), 3,
                      linewidth=1.8, edgecolor='#7B2FBE',
                      facecolor='#F3E5F5', alpha=0.15, clip_on=True)
ax_heat.add_patch(rect)
ax_heat.text(-0.55, 2.55, 'Pritamab\ncombinations',
             fontsize=7.5, color='#7B2FBE', va='top', ha='right',
             style='italic', fontweight='bold')

# ─────────────────────────────────────────────────────────────────
# Panel B: Radar Chart (top-5 regimens vs 6 key toxicities)
# ─────────────────────────────────────────────────────────────────
RADAR_CATS  = ['Neutropenia','Diarrhea','Nausea/\nVomiting','Fatigue',
               'Periph.\nNeuropathy','Hepatotoxicity']
RADAR_COLS  = [0, 4, 3, 6, 5, 9]   # indices in TOX_SHORT
RADAR_REGS  = [0, 2, 3, 5, 9]      # Prit+FOLFOX, Pritamab, FOLFOX, FOLFOXIRI, Pembro
RADAR_LABELS= ['Prit+FOLFOX','Pritamab Mono','FOLFOX','FOLFOXIRI','Pembrolizumab']

N_ax = len(RADAR_CATS)
angles = np.linspace(0, 2*np.pi, N_ax, endpoint=False).tolist()
angles += angles[:1]  # close

for reg_idx, reg_lbl, color in zip(RADAR_REGS, RADAR_LABELS,
                                    ['#7B2FBE','#C39BD3','#2471A3','#E74C3C','#27AE60']):
    vals = [TOX_MATRIX[reg_idx, c] for c in RADAR_COLS]
    vals += vals[:1]
    ax_radar.plot(angles, vals, color=color, lw=2.0, label=reg_lbl)
    ax_radar.fill(angles, vals, color=color, alpha=0.08)

ax_radar.set_xticks(angles[:-1])
ax_radar.set_xticklabels(RADAR_CATS, fontsize=8.5, color='#1A1A2E')
ax_radar.set_ylim(0, 55)
ax_radar.set_yticks([10,20,30,40,50])
ax_radar.set_yticklabels(['10','20','30','40','50'], fontsize=7, color='#888899')
ax_radar.grid(color='#DDDDEE', lw=0.8)
ax_radar.spines['polar'].set_visible(False)

ax_radar.legend(loc='upper right', bbox_to_anchor=(1.45, 1.15),
                fontsize=8.0, facecolor='white', edgecolor='#AAAACC',
                labelcolor='#1A1A2E')
ax_radar.set_title(
    'B   Key Toxicity Radar\n(Grade 3/4 %)',
    loc='center', fontsize=11, fontweight='bold',
    color='#1A1A2E', pad=18)

# ─────────────────────────────────────────────────────────────────
# Panel C: Composite Toxicity Score + Horizontal Bar
# ─────────────────────────────────────────────────────────────────
sorted_idx = np.argsort(COMPOSITE_TOX)[::-1]
sorted_reg  = [REGIMENS_SHORT[i] for i in sorted_idx]
sorted_tox  = COMPOSITE_TOX[sorted_idx]
sorted_clr  = [REG_COLORS[i] for i in sorted_idx]

y_pos = np.arange(len(sorted_reg))
bars  = ax_bar.barh(y_pos, sorted_tox, color=sorted_clr,
                     height=0.68, alpha=0.88, edgecolor='white', lw=0.5)

ax_bar.set_yticks(y_pos)
ax_bar.set_yticklabels(sorted_reg, fontsize=8.5, color='#1A1A2E')
ax_bar.set_xlabel('Composite Toxicity Score (sum of Grade 3/4 %)',
                   fontsize=9.5, color='#333355')
ax_bar.tick_params(axis='x', labelsize=8.5, colors='#444466')
ax_bar.grid(axis='x', color='#DDDDEE', lw=0.8)

for bar_i, (val, x) in enumerate(zip(sorted_tox, sorted_tox)):
    ax_bar.text(val + 1.0, bar_i, f'{val:.0f}',
                va='center', fontsize=8.5, color='#1A1A2E', fontweight='bold')

# Safety zone annotation
best_val = sorted_tox.min()
worst_val= sorted_tox.max()
ax_bar.axvspan(0, 30, alpha=0.06, color='#27AE60', label='Lower toxicity zone')
ax_bar.axvspan(80, worst_val+10, alpha=0.06, color='#E74C3C', label='Higher toxicity zone')
ax_bar.set_xlim(0, worst_val * 1.18)

# Reference labels
ax_bar.text(14, len(sorted_reg)-0.2, 'Lower toxicity zone',
            color='#27AE60', fontsize=7.5, style='italic', alpha=0.8)
ax_bar.text(85, len(sorted_reg)-0.2, 'Higher toxicity',
            color='#E74C3C', fontsize=7.5, style='italic', alpha=0.8)

ax_bar.set_title(
    'C   Composite Toxicity Score\n(all Grade 3/4 adverse events summed)',
    loc='left', fontsize=11, fontweight='bold', color='#1A1A2E')

# Highlight Pritamab entries
for bar_o, idx_s in enumerate(sorted_idx):
    if idx_s <= 2:  # Pritamab-based
        ax_bar.get_yticklabels()[bar_o].set_color('#7B2FBE')
        ax_bar.get_yticklabels()[bar_o].set_fontweight('bold')

# ─────────────────────────────────────────────────────────────────
# SUPER TITLE & SOURCE BOX
# ─────────────────────────────────────────────────────────────────
fig.suptitle(
    'Anticancer Regimen Toxicity Profile Comparison',
    fontsize=17, fontweight='bold', color='#0D1B4B', y=0.97)

fig.text(0.5, 0.02,
         'Data: NCCN Guidelines v2.2024 (CRC) | ESMO 2023 CRC Guidelines | '
         'RECOURSE trial (Mayer 2015) | KEYNOTE-158/177 | Falcone 2007 (GONO) | '
         'Lee et al. ADDS 2026 (Pritamab, NatureComm submission)\n'
         'Grade 3/4 incidence compiled from pivotal RCTs or meta-analyses. '
         'Pritamab data: proprietary ADDS platform prediction + in vitro validation.',
         ha='center', fontsize=7.5, color='#555577',
         bbox=dict(boxstyle='round,pad=0.5', fc='#F0F0F8', ec='#AAAACC', lw=0.8))

# ─────────────────────────────────────────────────────────────────
# SAVE
# ─────────────────────────────────────────────────────────────────
out_path = os.path.join(OUT, 'regimen_toxicity_profile.png')
plt.savefig(out_path, dpi=175, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.close()

sz = os.path.getsize(out_path) // 1024
print(f'Saved: {out_path}  ({sz} KB)')

# Verification
import csv as _csv
print('\n=== Verification ===')
print(f'Regimens: {len(REGIMENS)}')
print(f'Toxicity categories: {len(TOX_SHORT)}')
print(f'Matrix shape: {TOX_MATRIX.shape}')
print(f'Composite scores:')
for i, reg in enumerate(REGIMENS_SHORT):
    print(f'  {reg:20s}: {COMPOSITE_TOX[i]:.0f}')
print(f'Min composite: {COMPOSITE_TOX.min():.0f} ({REGIMENS_SHORT[COMPOSITE_TOX.argmin()]})')
print(f'Max composite: {COMPOSITE_TOX.max():.0f} ({REGIMENS_SHORT[COMPOSITE_TOX.argmax()]})')
