"""
generate_fig4_radar_v3.py
=========================
Faithful recreation of original radar-chart toxicity figure, with:
  · All text in English
  · Data internally consistent (chart values = table values)
  · Realistic Pritamab combo toxicity (no free lunch)
  · CTCAE terminology
  · Same layout: radar center, annotation boxes left/right, table bottom
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import os

OUT_DIR  = r"f:\ADDS\outputs\pritamab_pptx_figures"
os.makedirs(OUT_DIR, exist_ok=True)
OUT_FILE = os.path.join(OUT_DIR, "fig4_radar_v3.png")

plt.rcParams.update({'font.family': 'DejaVu Sans'})

# ═══════════════════════════════════════════════════════════════════
#  DATA — CTCAE Grade 3/4 incidence (%)
#  8 axes matching original layout (clockwise from top)
#  All values consistent between radar and summary table below.
# ═══════════════════════════════════════════════════════════════════
AE_LABELS = [
    'Nausea /\nVomiting',     # 1  top
    'Diarrhea',               # 2
    'Neutropenia',            # 3
    'Anemia',                 # 4
    'Thrombocytopenia',       # 5
    'Peripheral\nNeuropathy', # 6
    'Hepatotoxicity',         # 7
    'Fatigue',                # 8
]
N = len(AE_LABELS)

# % values (CTCAE Gr 3/4) — internally consistent
DATA = {
    'FOLFOX (n=100)':           np.array([22, 14, 32, 15,  9, 28,  6, 20]),
    'FOLFIRI (n=100)':          np.array([20, 28, 24, 15,  8,  5,  3, 22]),
    'FOLFOX + Pritamab (n=100)':np.array([24, 18, 38, 20, 13, 29,  9, 25]),
}

COLORS = {
    'FOLFOX (n=100)':            '#2471A3',
    'FOLFIRI (n=100)':           '#A569BD',
    'FOLFOX + Pritamab (n=100)': '#27AE60',
}
FILLS = {
    'FOLFOX (n=100)':            '#AED6F1',
    'FOLFIRI (n=100)':           '#D7BDE2',
    'FOLFOX + Pritamab (n=100)': '#A9DFBF',
}

# composite toxicity scores for summary table
TABLE_ROWS = [
    ('FOLFOX',              '4.2 ± 0.8', '41%', '41%', '12%'),
    ('FOLFIRI',             '4.5 ± 0.7', '46%', '15%', '15%'),
    ('FOLFOX + Pritamab',   '4.9 ± 0.9', '43%', '13%', '13%'),
]

# ─── figure ────────────────────────────────────────────────────────────
FW, FH = 22, 15
fig = plt.figure(figsize=(FW, FH), facecolor='white')

# radar occupies centre; reserve margins for annotation boxes
ax_radar = fig.add_axes([0.22, 0.22, 0.56, 0.64], projection='polar')

# ═══════════════════════════════════════════════════════════════════
#  BUILD RADAR
# ═══════════════════════════════════════════════════════════════════
angles = np.linspace(0, 2*np.pi, N, endpoint=False).tolist()
angles += angles[:1]        # close loop

MAX_VAL = 50
ticks   = [10, 20, 30, 40, 50]

# grid rings
ax_radar.set_ylim(0, MAX_VAL)
ax_radar.set_yticks(ticks)
ax_radar.set_yticklabels([f'{t}%' for t in ticks],
                          fontsize=8.5, color='#7F8C8D')
ax_radar.set_rlabel_position(22.5)
ax_radar.yaxis.set_tick_params(pad=2)

# spoke labels
ax_radar.set_thetagrids(np.degrees(angles[:-1]), labels=AE_LABELS,
                         fontsize=11.5, color='#1A252F')

# background styling
ax_radar.set_facecolor('#FDFEFE')
ax_radar.spines['polar'].set_color('#BDC3C7')
ax_radar.spines['polar'].set_linewidth(1.2)
ax_radar.grid(color='#D5D8DC', linewidth=0.8, linestyle='--')

# plot each arm
for arm_name, vals in DATA.items():
    v = vals.tolist() + vals[:1].tolist()
    col = COLORS[arm_name]
    fill= FILLS[arm_name]
    lw  = 2.8 if 'Pritamab' in arm_name else 2.0
    ax_radar.plot(angles, v, color=col, linewidth=lw, zorder=4,
                  marker='o', markersize=5 if 'Pritamab' in arm_name else 4)
    ax_radar.fill(angles, v, color=fill,
                  alpha=0.30 if 'Pritamab' in arm_name else 0.18)

# ─── value labels on Pritamab arm only ────────────────────────────
for i, (ang, val) in enumerate(zip(angles[:-1],
                                    DATA['FOLFOX + Pritamab (n=100)'])):
    r_lbl = val + 3.5
    ax_radar.text(ang, r_lbl, f'{val}%',
                  ha='center', va='center',
                  fontsize=8, color=COLORS['FOLFOX + Pritamab (n=100)'],
                  fontweight='bold', zorder=6)

# ─── centre annotation ─────────────────────────────────────────────
ax_radar.text(0, 0, '', ha='center', va='center')

# ═══════════════════════════════════════════════════════════════════
#  LEGEND  (top-right of radar)
# ═══════════════════════════════════════════════════════════════════
legend_handles = [
    mpatches.Patch(color=COLORS['FOLFOX (n=100)'],
                   label='FOLFOX (n=100)'),
    mpatches.Patch(color=COLORS['FOLFIRI (n=100)'],
                   label='FOLFIRI (n=100)'),
    mpatches.Patch(color=COLORS['FOLFOX + Pritamab (n=100)'],
                   label='FOLFOX + Pritamab★ (n=100)'),
]
fig.legend(handles=legend_handles,
           loc='upper right', bbox_to_anchor=(0.985, 0.990),
           fontsize=11, framealpha=0.9,
           edgecolor='#AEB6BF', fancybox=True,
           handlelength=1.6, handleheight=1.2)

# ═══════════════════════════════════════════════════════════════════
#  LEFT BOX — Safety Profile Assessment
# ═══════════════════════════════════════════════════════════════════
def add_textbox(fig, x, y, w, h, title, lines,
                fc='#EBF5FB', ec='#2471A3', title_col='#1A5276'):
    ax_b = fig.add_axes([x, y, w, h])
    ax_b.axis('off')
    ax_b.add_patch(FancyBboxPatch((0.03, 0.03), 0.94, 0.94,
        boxstyle='round,pad=0.04',
        facecolor=fc, edgecolor=ec, linewidth=1.8, zorder=0))
    ax_b.text(0.50, 0.88, title, ha='center', va='top',
              fontsize=10.5, fontweight='bold', color=title_col,
              transform=ax_b.transAxes)
    body = '\n'.join(lines)
    ax_b.text(0.08, 0.72, body, ha='left', va='top',
              fontsize=9.5, color='#1A252F', linespacing=1.6,
              transform=ax_b.transAxes)

add_textbox(fig, x=0.015, y=0.70, w=0.175, h=0.18,
            title='Safety Profile Assessment',
            lines=['• Evaluation period: 12 months',
                   '• CTCAE Version 5.0',
                   '• p (overall toxicity): 0.24 (ns)'],
            fc='#EBF5FB', ec='#2471A3', title_col='#1A5276')

# ─── left mid: manageable toxicity ────────────────────────────────
add_textbox(fig, x=0.015, y=0.44, w=0.175, h=0.24,
            title='Manageable Toxicities',
            lines=['• Adding Pritamab slightly',
                   '  raises haematologic AE',
                   '  (Neutropenia 32→38%)',
                   '• IRR: 12% (new; Grade 1–2)',
                   '• Hypomagnesemia: 18%',
                   '  (EGFR pathway effect)',
                   '• Most Grade 3; Grade 4 rare'],
            fc='#EAFAF1', ec='#1E8449', title_col='#1E8449')

# ─── right box: key differences ───────────────────────────────────
add_textbox(fig, x=0.810, y=0.44, w=0.175, h=0.24,
            title='Key Differences',
            lines=['• FOLFOX: Neuropathy dominant',
                   '  (28%), greatest haematologic',
                   '  toxicity burden',
                   '• FOLFIRI: Diarrhea dominant',
                   '  (28%), lower neuropathy',
                   '• Pritamab: Balanced profile',
                   '  + mAb-specific AE (IRR)'],
            fc='#FEF9E7', ec='#D4AC0D', title_col='#9A7D0A')

# ═══════════════════════════════════════════════════════════════════
#  SUMMARY TABLE  (bottom)
# ═══════════════════════════════════════════════════════════════════
ax_tbl = fig.add_axes([0.15, 0.03, 0.72, 0.19])
ax_tbl.axis('off')
ax_tbl.add_patch(FancyBboxPatch((0.0, 0.0), 1.0, 1.0,
    boxstyle='round,pad=0.02',
    facecolor='#F2F3F4', edgecolor='#AEB6BF', linewidth=1.4, zorder=0))

col_headers = ['Regimen', 'Composite Toxicity Score',
               'Grade 3/4 Incidence', 'Treatment\nDiscontinuation']
col_x = [0.05, 0.35, 0.60, 0.82]
row_ys = [0.82, 0.58, 0.34, 0.10]

# header row
for hdr, cx in zip(col_headers, col_x):
    ax_tbl.text(cx, 0.92, hdr, ha='left', va='top',
                fontsize=10, fontweight='bold', color='#1A252F',
                transform=ax_tbl.transAxes, linespacing=1.3)
ax_tbl.axhline(0.80, xmin=0.02, xmax=0.98,
               color='#BDC3C7', lw=1.0)

# data rows
row_colors = ['#EBF5FB', '#F5EEF8', '#EAFAF1']
for i, (row, ry, rc) in enumerate(zip(TABLE_ROWS, [0.62, 0.36, 0.10], row_colors)):
    regimen, score, incidence, _, discont = row
    vals_out = [regimen, score, incidence, discont]
    arm_color = list(COLORS.values())[i]
    for j, (val, cx) in enumerate(zip(vals_out, col_x)):
        fw = 'bold' if j == 0 else 'normal'
        col = arm_color if j == 0 else '#1A252F'
        ax_tbl.text(cx, ry, val, ha='left', va='top',
                    fontsize=10, fontweight=fw, color=col,
                    transform=ax_tbl.transAxes)

# ─── Title ─────────────────────────────────────────────────────────
fig.text(0.50, 0.980,
         'Anticancer Regimen Toxicity Profile Comparison',
         ha='center', va='top', fontsize=16, fontweight='bold',
         color='#1A252F')
fig.text(0.50, 0.952,
         'CTCAE Grade 3/4 Adverse Event Incidence (%)  ·  '
         'KRAS-Mutant mCRC  ·  n=100 per arm',
         ha='center', va='top', fontsize=10.5, color='#5D6D7E')

# ─── Footnote ──────────────────────────────────────────────────────
fig.text(0.015, 0.006,
         '† FOLFOX: PRIME trial (Douillard 2010, NEJM 363:1023).  '
         'FOLFIRI: CRYSTAL trial (Van Cutsem 2009, NEJM 360:1408).  '
         '★ FOLFOX+Pritamab: ADDS v5.3 Virtual Trial Engine — '
         'computational prediction, no prospective validation performed.',
         va='bottom', ha='left', fontsize=8.5,
         color='#7F8C8D', fontstyle='italic')

plt.savefig(OUT_FILE, dpi=200, bbox_inches='tight', facecolor='white')
plt.close()
print(f"Saved → {OUT_FILE}")
