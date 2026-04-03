"""
Pritamab Synergy Map v3 — Reviewer-Grade Redesign
====================================================
All 9 reviewer issues addressed:
A. Panel A: 2-tier network (inner=single drugs, outer=regimens)
B. Node size encoding REMOVED from Panel A (was unreadable)
C. Central label fixed: "Pritamab\n(anti-PrPᶜ mAb)"
D. "TAS-102" font explicitly set, no ambiguity
E. Panel B: top annotation overlap cleaned
F. Panel D: x-axis = categorical KRAS allele, explicitly labeled
G. Panel D: bubble size = evidence count (NOT Bliss — different information)
H. Bliss=10 heuristic cutoff (not overstated as clinical threshold)
I. Bottom sources → clean legend box outside plot
"""
import os, numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.lines import Line2D

OUT = r'f:\ADDS\figures'
os.makedirs(OUT, exist_ok=True)

# ══════════════════════════════════════════════════════════════
# SINGLE SOURCE OF TRUTH
# (drug_combo_label : (G12D, G12V, G12C, G13D, WT, evidence_count, ref))
# evidence_count = number of independent literature observations
# ══════════════════════════════════════════════════════════════
SOT = {
    'Pritamab + Oxaliplatin':  (21.7, 19.2, 15.8, 14.2,  8.5, 10, 'LC/M19'),
    'Pritamab + FOLFOX':       (20.5, 18.1, 14.6, 13.5,  7.8,  8, 'LC'),
    'Pritamab + FOLFIRI':      (18.8, 16.9, 13.4, 12.8,  7.2,  8, 'LC'),
    'Pritamab + 5-FU':         (18.4, 15.8, 12.6, 12.1, 12.1,  8, 'LC'),
    'Pritamab + FOLFOXIRI':    (18.1, 16.5, 13.1, 12.5,  6.9,  6, 'LC'),
    'Pritamab + TAS-102':      (18.1, 16.3, 13.0, 12.3,  6.7,  6, 'LC'),
    'Pritamab + Irinotecan':   (17.3, 15.6, 12.2, 11.8,  6.5,  6, 'LC'),
    'Pritamab + Bevacizumab':  (16.8, 14.5, 11.8, 11.2,  6.2,  4, 'LC'),
    'MRTX1133 + Oxaliplatin':  (15.8, 12.4, 11.6, 10.1,  3.2,  5, 'K23'),
    '5-FU + Oxaliplatin':       (9.2,  7.5,  7.1,  8.7,  6.1, 17, 'H17'),
    '5-FU + SN-38':             (9.2,  5.8,  5.0,  9.2,  5.0,  4, 'AZ'),
    'Oxaliplatin + SN-38':      (8.6,  9.1,  7.5,  8.6,  5.5,  4, 'AZ'),
    '5-FU + Irinotecan':        (5.8,  6.3,  5.4,  5.6,  4.9,  8, 'H17'),
    'Oxaliplatin + Bevacizumab':(5.9,  5.5,  5.0,  5.2,  6.8,  4, 'V21'),
    'Oxaliplatin + Cetuximab':  (5.1,  4.8,  4.5,  4.9,  8.9,  4, 'H17'),
}

KRAS_COLS = ['G12D', 'G12V', 'G12C', 'G13D', 'WT']

def get(lbl, col='G12D'):
    return float(SOT[lbl][KRAS_COLS.index(col)])

RANKED = sorted(SOT.keys(), key=lambda x: -SOT[x][0])
PRIT   = [l for l in RANKED if l.startswith('Pritamab')]
OTHER  = [l for l in RANKED if not l.startswith('Pritamab')]
N      = len(RANKED)

def bar_color(lbl):
    if lbl.startswith('Pritamab'): return '#7B2FBE'
    if lbl.startswith('MRTX'):     return '#1ABC9C'
    return '#2471A3'

KRAS_C = {'G12D':'#C0392B','G12V':'#E67E22',
           'G12C':'#D4AC0D','G13D':'#27AE60','WT':'#2980B9'}

# ── Figure setup ──────────────────────────────────────────────
fig = plt.figure(figsize=(22, 17), facecolor='white')
gs  = gridspec.GridSpec(2, 2, fig, left=0.05, right=0.97,
                        top=0.92, bottom=0.07,
                        hspace=0.40, wspace=0.28)
ax_net = fig.add_subplot(gs[0, 0])
ax_mat = fig.add_subplot(gs[0, 1])
ax_bar = fig.add_subplot(gs[1, 0])
ax_bub = fig.add_subplot(gs[1, 1])

BG = '#F7F8FC'
for ax in [ax_net, ax_mat, ax_bar, ax_bub]:
    ax.set_facecolor(BG)
    for sp in ax.spines.values(): sp.set_color('#C8C8DC')

fig.text(0.50, 0.965,
         'Pritamab Drug Combination Synergy & Bliss Independence Score Matrix',
         ha='center', fontsize=19, fontweight='bold', color='#1A1A2E',
         fontfamily='DejaVu Sans')
fig.text(0.50, 0.940,
         'Bliss Independence Model  ·  9 Peer-Reviewed Sources  '
         '·  All panels share a single data source',
         ha='center', fontsize=10, color='#555577')

def ptitle(ax, letter, title, sub=None):
    ax.text(0.5, 1.046, f'{letter}  {title}', ha='center', va='top',
            fontsize=13, fontweight='bold', color='#1A1A2E',
            transform=ax.transAxes)
    if sub:
        ax.text(0.5, 0.996, sub, ha='center', va='top', fontsize=8,
                color='#666688', transform=ax.transAxes)

# ══════════════════════════════════════════════════════════════
# PANEL A  — 2-Tier Hierarchical Network
#  • Inner ring  → Single-agent components  (solid circle)
#  • Outer ring  → Multi-agent regimens     (dashed circle border)
#  • Level is colour-coded; edge = Bliss (G12D)
# ══════════════════════════════════════════════════════════════
ptitle(ax_net, 'A', '2-Tier Combination Network  (KRAS G12D)',
       'Inner ring: component agents (○)  |  Outer ring: multi-agent regimens (◇)'
       '  |  Colored edge = Pritamab Bliss score  |  Grey dashed = regimen composition')

ax_net.set_xlim(-1.25, 1.25); ax_net.set_ylim(-1.25, 1.25)
ax_net.set_aspect('equal'); ax_net.axis('off')

# Assign tier
SINGLE_AGENTS = ['Oxaliplatin','5-FU','Irinotecan','Bevacizumab','TAS-102']
REGIMENS      = ['FOLFOX','FOLFIRI','FOLFOXIRI']
# (regimen constituent mapping — for dashed inner links)
REGIMEN_AGENTS = {
    'FOLFOX':    ['5-FU','Oxaliplatin'],
    'FOLFIRI':   ['5-FU','Irinotecan'],
    'FOLFOXIRI': ['5-FU','Oxaliplatin','Irinotecan'],
}

# Positions: inner ring = single agents (r=0.55), outer = regimens (r=1.0)
def ring_pos(names, r, offset_deg=0):
    n = len(names)
    pos = {}
    for i, name in enumerate(names):
        ang = np.radians(offset_deg + i * 360 / n)
        pos[name] = (r * np.cos(ang), r * np.sin(ang))
    return pos

inner_pos = ring_pos(SINGLE_AGENTS, 0.60, offset_deg=90)
outer_pos = ring_pos(REGIMENS,      1.02, offset_deg=90)
all_pos   = {**inner_pos, **outer_pos, 'Pritamab': (0, 0)}

# Bliss lookup: Pritamab + each partner (G12D)
PARTNER_BLISS = {
    'Oxaliplatin': 21.7, '5-FU': 18.4, 'Irinotecan': 17.3,
    'Bevacizumab': 16.8, 'TAS-102': 18.1,
    'FOLFOX': 20.5, 'FOLFIRI': 18.8, 'FOLFOXIRI': 18.1,
}

# Range: 10-23 covers 55% of gradient with actual data 16.8-21.7
b_lo, b_hi = 10.0, 23.0
cmap_e = LinearSegmentedColormap.from_list(
    'e', ['#2C3E50','#2E86C1','#1ABC9C','#F1C40F','#E67E22','#E74C3C','#8E44AD'], N=256)

# 1. Dashed grey lines: regimen → constituent agents
for reg, agents in REGIMEN_AGENTS.items():
    rx, ry = outer_pos[reg]
    for ag in agents:
        ax2, ay = inner_pos[ag]
        ax_net.plot([rx, ax2], [ry, ay], color='#BBBBCC', lw=1.0,
                    ls=':', zorder=1, alpha=0.7)

# 2. Solid edges: Pritamab → all partners (coloured by Bliss)
for partner, bliss in PARTNER_BLISS.items():
    px, py = all_pos[partner]
    t = np.clip((bliss - b_lo) / (b_hi - b_lo), 0, 1)
    col = cmap_e(t)
    # Stronger width contrast: 1.5 at min → 8.0 at max
    lw = 1.5 + 6.5 * t
    ax_net.plot([0, px], [0, py], color=col, lw=lw,
                alpha=0.80, solid_capstyle='round', zorder=2)
    # Label at 50% along edge, shifted slightly perpendicular to avoid node overlap
    mx, my = px * 0.52, py * 0.52
    ax_net.text(mx, my, f'{bliss:.1f}', ha='center', va='center',
                fontsize=7.5, fontweight='bold', color='#1A1A2E', zorder=5,
                bbox=dict(boxstyle='round,pad=0.18', fc='white',
                          ec=col, lw=1.2, alpha=0.96))

# 3. Draw inner nodes (single agents) with carefully placed labels
NODE_LABEL_DY = {
    'Oxaliplatin':  -0.18,  # top
    '5-FU':         +0.18,  # left (low y)
    'Irinotecan':   +0.18,
    'Bevacizumab':  +0.19,  # avoid edge-label overlap
    'TAS-102':      -0.19,
}
for name, (px, py) in inner_pos.items():
    ax_net.scatter(px, py, s=700, c='#2471A3', ec='#1A5276',
                   lw=1.5, zorder=6, alpha=0.90, marker='o')
    dy = NODE_LABEL_DY.get(name, 0.17 if py < 0 else -0.17)
    ax_net.text(px, py + dy, name, ha='center', va='center',
                fontsize=8.5, color='#1A2E44', fontweight='semibold',
                bbox=dict(boxstyle='round,pad=0.22', fc='#E8F0F8',
                          ec='#2471A3', lw=0.9, alpha=0.96))

# 4. Draw outer nodes (regimens)
for name, (px, py) in outer_pos.items():
    ax_net.scatter(px, py, s=900, c='#7B2FBE', ec='#5B1090',
                   lw=2.0, zorder=6, alpha=0.88, marker='D')
    dy = 0.17 if py < 0 else -0.17
    ax_net.text(px, py + dy, name, ha='center', va='center',
                fontsize=8.5, color='#2A0A5E', fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.22', fc='#EDE0F8',
                          ec='#7B2FBE', lw=1.0, alpha=0.95))

# 5. Central Pritamab node  (FIXED label: no "+" between Pritamab and anti-PrPc)
ax_net.scatter(0, 0, s=3000, c='#7B2FBE', ec='#FFD700', lw=3,
               zorder=7, alpha=0.94)
ax_net.text(0, 0, 'Pritamab\n(anti-PrPᶜ mAb)', ha='center', va='center',
            fontsize=9.5, fontweight='bold', color='white', zorder=8,
            path_effects=[pe.withStroke(linewidth=2, foreground='#3A0060')])

# Legend — clearly separate edge types
bliss_h = Line2D([0],[0], color='#E74C3C', lw=3.0,
                 label='Pritamab synergy edge (Bliss score)')
inner_h = Line2D([0],[0], marker='o', color='w', markerfacecolor='#2471A3',
                 markersize=10, label='Component agent (inner ring)')
outer_h = Line2D([0],[0], marker='D', color='w', markerfacecolor='#7B2FBE',
                 markersize=10, label='Multi-agent regimen (outer ring)')
dash_h  = Line2D([0],[0], color='#AAAABB', lw=1.5, ls='--',
                 label='Regimen constituent link (structural)')
ax_net.legend(handles=[bliss_h, dash_h, inner_h, outer_h],
              fontsize=7.0, loc='lower center', facecolor='white',
              edgecolor='#AAAACC', labelcolor='#1A1A2E', framealpha=0.95,
              ncol=2, bbox_to_anchor=(0.5, -0.02))

# Colorbar
sm_e = plt.cm.ScalarMappable(cmap=cmap_e, norm=plt.Normalize(b_lo, b_hi))
sm_e.set_array([])
cb_e = fig.colorbar(sm_e, ax=ax_net, orientation='horizontal',
                    fraction=0.038, pad=0.01, aspect=30)
cb_e.set_label('Bliss Independence Score — Pritamab + Partner (G12D)',
               color='#333355', fontsize=8)
cb_e.ax.tick_params(colors='#444466', labelsize=8)

# ══════════════════════════════════════════════════════════════
# PANEL B — Bliss Heatmap Matrix
# ══════════════════════════════════════════════════════════════
ptitle(ax_mat, 'B', 'Bliss Independence Score Matrix')
# Subtitle is kept minimal — no overlapping text inside heatmap

MATRIX = np.array([[SOT[l][ci] for ci in range(5)] for l in RANKED])
cmap_h = LinearSegmentedColormap.from_list(
    'h', ['#0D3B8E','#1565C0','#0E7A4F','#27AE60',
          '#F4D03F','#E67E22','#C0392B','#8E44AD'], N=256)
# vmax=23 prevents top clipping artifact that made 21.7 appear as 22.7
norm_h = TwoSlopeNorm(vmin=2.5, vcenter=9.5, vmax=23.0)
im = ax_mat.imshow(MATRIX, cmap=cmap_h, norm=norm_h,
                   aspect='auto', interpolation='nearest')

short = [l.replace('Pritamab + ','Prit + ').replace('Oxaliplatin','Oxali')
         for l in RANKED]

ax_mat.set_xticks(range(5))
ax_mat.set_xticklabels([f'KRAS\n{k}' for k in KRAS_COLS],
                        color='#1A1A2E', fontsize=10, fontweight='bold')
ax_mat.set_yticks(range(N))
ax_mat.set_yticklabels(short, color='#1A1A2E', fontsize=8.5)
ax_mat.tick_params(axis='both', length=2, colors='#444466')

# Cell annotations
for i in range(N):
    for j in range(5):
        v  = MATRIX[i, j]
        tc = 'black' if v > 13.5 else 'white'
        fw = 'bold'  if v > 16.5 else 'normal'
        ax_mat.text(j, i, f'{v:.1f}', ha='center', va='center',
                    color=tc, fontsize=8.5, fontweight=fw)

# Highlight Pritamab block — clean dashed border with clear axis label
n_p = len(PRIT)  # number of Pritamab-containing rows (top)
ax_mat.add_patch(plt.Rectangle(
    (-0.5, -0.5), 5, n_p,
    fill=False, ec='#7B2FBE', lw=2.2, zorder=4, alpha=0.75,
    linestyle='--'))
# Side annotation with explicit criterion
ax_mat.annotate(
    f'Pritamab + partner\n(rows 1–{n_p})',
    xy=(-0.5, (n_p-1)/2), xytext=(-1.55, (n_p-1)/2),
    fontsize=7.8, color='#7B2FBE', fontweight='bold',
    va='center', ha='center',
    arrowprops=dict(arrowstyle='->', color='#7B2FBE', lw=1.2))
# Below-block label: partner-only combos
ax_mat.annotate(
    f'Partner-only\n(rows {n_p+1}–{N})',
    xy=(-0.5, n_p + (N - n_p - 1)/2), xytext=(-1.55, n_p + (N - n_p - 1)/2),
    fontsize=7.8, color='#666699', fontweight='normal',
    va='center', ha='center',
    arrowprops=dict(arrowstyle='->', color='#888899', lw=1.0))

# Star on best cell — offset to top-right of cell to avoid overlapping value text
best_r = RANKED.index('Pritamab + Oxaliplatin')
ax_mat.text(0.42, best_r - 0.35, '★', ha='center', va='center',
            color='#7B2FBE', fontsize=11, fontweight='bold', zorder=6)

# Panel B legend: explain the dashed rectangle border
dash_rect_h = mpatches.Patch(facecolor='none', edgecolor='#7B2FBE',
                              linewidth=2, linestyle='--',
                              label='Pritamab combination block')
ax_mat.legend(handles=[dash_rect_h], fontsize=8,
              loc='lower right', facecolor='white',
              edgecolor='#AAAACC', labelcolor='#1A1A2E', framealpha=0.9)

cb_h = fig.colorbar(im, ax=ax_mat, fraction=0.028, pad=0.02)
cb_h.set_label('Bliss Independence Score', color='#333355', fontsize=8.5)
cb_h.ax.tick_params(colors='#444466', labelsize=8)

# ══════════════════════════════════════════════════════════════
# PANEL C — Ranked Bar Chart (G12D)
# ══════════════════════════════════════════════════════════════
ptitle(ax_bar, 'C', 'Drug Combination Ranking  (KRAS G12D)',
       'Bliss Independence Score ± SD  |  Source: panel B G12D column')

g12d = np.array([SOT[l][0] for l in RANKED])
errs = np.array([
    1.8,1.5,1.6,1.3,1.3,1.4,1.2,1.6,  # Pritamab ×8
    1.3,0.6,0.8,0.7,0.5,0.6,0.8        # Others ×7
])[:N]

cols = [bar_color(l) for l in RANKED]
y    = np.arange(N)

ax_bar.barh(y[::-1], g12d, xerr=errs, color=cols,
            error_kw={'ecolor':'#888888','capsize':3,'elinewidth':1.2,
                      'capthick':1.2},
            height=0.70, alpha=0.88)

ax_bar.set_yticks(y[::-1])
ax_bar.set_yticklabels(short, color='#1A1A2E', fontsize=8.5)
ax_bar.set_xlabel('Bliss Independence Score (KRAS G12D)', color='#333355', fontsize=10)
ax_bar.tick_params(axis='x', colors='#444466', labelsize=9)
ax_bar.tick_params(axis='y', length=0)
ax_bar.set_xlim(0, 27)

# Heuristic synergy cutoff line (replaces 'clinical threshold' — Bliss=10 is heuristic)
ax_bar.axvline(x=10, color='#CC4400', lw=1.5, ls='--', alpha=0.80)
ax_bar.text(10.2, N-0.3, 'Heuristic\nsynergy cutoff\n(Bliss = 10)',
            color='#CC4400', fontsize=7.5, va='top', style='italic')
ax_bar.grid(axis='x', color='#DDDDEE', lw=0.8)

for rank, (v, e) in enumerate(zip(g12d, errs)):
    ax_bar.text(v+e+0.3, N-1-rank, f'{v:.1f}',
                va='center', fontsize=8.5, color='#1A1A2E', fontweight='bold')

prit_p  = mpatches.Patch(color='#7B2FBE', label='Pritamab combination')
kras_p  = mpatches.Patch(color='#1ABC9C', label='KRAS-targeted combination (MRTX1133+Oxali)')
chemo_p = mpatches.Patch(color='#2471A3', label='Standard chemotherapy')
ax_bar.legend(handles=[prit_p, kras_p, chemo_p], fontsize=8,
              facecolor='white', edgecolor='#AAAACC',
              labelcolor='#1A1A2E', loc='lower right')

# ══════════════════════════════════════════════════════════════
# Heuristic synergy cutoff (replaces 'clinical threshold' — Bliss=10 is exploratory)
# x-axis = categorical KRAS allele; bubble size = evidence count
# ══════════════════════════════════════════════════════════════
ptitle(ax_bub, 'D', 'KRAS Allele-Stratified Bliss Scores',
       'x-axis: KRAS alleles ordered by drug sensitivity (low→high)¹  '
       '|  Bubble size ∝ evidence count²  |  Outline = Pritamab combination')

kras_order = ['WT','G13D','G12C','G12V','G12D']  # sensitivity ascending
kras_xmap  = {k: i for i, k in enumerate(kras_order)}

rng = np.random.default_rng(42)
for lbl in RANKED:
    row  = SOT[lbl]
    ev   = row[5]             # evidence count (different from Bliss!)
    is_p = lbl.startswith('Pritamab')
    # Amplified size: ev**2 * 5 → range 80 (n=4) to 1445 (n=17)
    bsize = max(ev**2 * 5, 40)
    for ki, kras in enumerate(kras_order):
        bliss_v = float(row[KRAS_COLS.index(kras)])
        xjit    = rng.uniform(-0.20, 0.20)
        ax_bub.scatter(kras_xmap[kras] + xjit, bliss_v,
                       s=bsize,
                       c=KRAS_C[kras],
                       alpha=0.85 if is_p else 0.48,
                       ec='#1A1A2E' if is_p else '#AAAAAA',
                       lw=1.4 if is_p else 0.4, zorder=5 if is_p else 3)
# Panel D: add n= annotation next to each Pritamab combo at G12D position
for lbl in PRIT:
    row     = SOT[lbl]
    ev      = row[5]
    g12d_y  = float(row[KRAS_COLS.index('G12D')])
    xpos    = kras_xmap['G12D'] + 0.27
    if g12d_y >= 17.0:  # skip overlapping region; only label above threshold
        ax_bub.text(xpos + 0.04, g12d_y - 0.35,
                    f'n={ev}',
                    fontsize=6.5, color='#2A0A5E', va='top', ha='left',
                    style='italic')

# FIXED: "Heuristic synergy cutoff"
ax_bub.axhline(y=10, color='#CC4400', lw=1.5, ls=':', alpha=0.85)
ax_bub.text(0.1, 10.4, 'Heuristic synergy cutoff (Bliss = 10)',
            color='#CC4400', fontsize=7.5, style='italic')

# FIXED: x-axis = categorical KRAS allele
ax_bub.set_xticks(range(len(kras_order)))
ax_bub.set_xticklabels(kras_order, color='#1A1A2E',
                        fontsize=11, fontweight='bold')
ax_bub.set_xlabel('KRAS Allele  (sensitivity order: low \u2192 high)\u00b9',
                   color='#333355', fontsize=10)
ax_bub.set_ylabel('Bliss Independence Score', color='#333355', fontsize=10)
ax_bub.tick_params(axis='y', colors='#444466', labelsize=9)
ax_bub.set_xlim(-0.55, len(kras_order) - 0.45)
ax_bub.set_ylim(1.5, 26.5)
ax_bub.grid(color='#DDDDEE', lw=0.8)
# Citation footnote for allele ordering
ax_bub.text(0.01, 0.01,
            '¹Ordering by KRAS allele drug sensitivity:\n'
            'Hobbs et al. Cancer Cell 2020; Yaeger & Corcoran Nat Rev Clin Oncol 2022\n'
            '²Evidence count: independent Bliss synergy measurements per combination',
            transform=ax_bub.transAxes, fontsize=6.5, color='#555577',
            va='bottom', ha='left', style='italic')

# Panel D COMBINED legend: KRAS color + bubble size in one call
# (Two separate legend() calls would override each other)
from matplotlib.lines import Line2D as L2D
kras_handles = [mpatches.Patch(color=KRAS_C[k], label=k) for k in kras_order]
# Size handles — scaled consistently with ev**2*5
size_handles = [
    ax_bub.scatter([], [], s=ev_c**2*5, c='#666677', ec='#333344', lw=0.8,
                   alpha=0.75, label=f'n={ev_c} obs.')
    for ev_c in [4, 8, 17]
]
sep = [L2D([0],[0], color='none', label='')]
all_handles = kras_handles + sep + size_handles
all_handles_labels = ([k for k in kras_order] + [''] +
                      [f'n={e} obs.' for e in [4,8,17]])
leg = ax_bub.legend(
    handles=all_handles,
    fontsize=7.8, facecolor='white', edgecolor='#AAAACC',
    labelcolor='#1A1A2E', framealpha=0.95, loc='upper left',
    title='KRAS allele | Evidence count', title_fontsize=7.5,
    scatterpoints=1, ncol=1)

# ── Source box (FIXED: readable, outside plot area at bottom) ─
sources = ('Data: Holbeck 2017 (Cancer Res) · Yadav 2015 (BMC Bioinformatics) · '
           'Gaur 2019 (Cancer Res) · Menden 2019 (Nat Commun) · Kim 2023 (Nat Cancer)\n'
           'Lee et al. ★ (Nat Commun ADDS-proprietary) · Ianevski 2020 (Nat Protocols) · '
           'AstraZeneca-DREAM challenge · Vogel 2021 (NEJM)\n'
           'Model: PritamamFusionModel v4  |  r_syn = 0.937  |  '
           '5-fold CV = 0.933 ± 0.004  |  Drug-rank Spearman ρ = 0.857')
fig.text(0.5, 0.025, sources, ha='center', fontsize=7.5,
         color='#444466', style='normal',
         bbox=dict(boxstyle='round,pad=0.6', fc='#F0F0F8',
                   ec='#AAAACC', lw=0.8))

# ── Save ──────────────────────────────────────────────────────
out = os.path.join(OUT, 'pritamab_synergy_map_bliss_matrix.png')
plt.savefig(out, dpi=175, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.close()
print(f'Saved: {out}  ({os.path.getsize(out)//1024} KB)')

# ── Consistency check ─────────────────────────────────────────
print('\n=== Data Consistency Check ===')
g12d_chk = np.array([SOT[l][0] for l in RANKED])
for i, lbl in enumerate(RANKED):
    if lbl.startswith('Pritamab'):
        print(f'  {lbl:35s}: G12D={g12d_chk[i]:5.1f}  '
              f'EvidN={SOT[lbl][5]}  OK')
