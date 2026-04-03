"""
Fig.2  MM/PBSA Binding Energy Landscape — Pritamab·PrPᶜ
========================================================
Style: white background, scientific illustration
  - Protein surface well (3D-style energy funnel diagram)
  - Energy component breakdown bars (horizontal)
  - Residue contribution heatstrip
  - Key annotation boxes
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.colors import LinearSegmentedColormap, Normalize
import matplotlib.cm as mcm
import matplotlib.gridspec as gridspec
import os

OUT = r'f:\ADDS\pritamab\figures'
os.makedirs(OUT, exist_ok=True)

NAVY   = '#1E3A5F'; BLUE   = '#2563EB'; LTBLUE = '#DBEAFE'
TEAL   = '#0D9488'; LTEAL  = '#CCFBF1'
RED    = '#DC2626'; LRED   = '#FEE2E2'
PURPLE = '#7C3AED'; LPUR   = '#EDE9FE'
GOLD   = '#D97706'; LGOLD  = '#FEF3C7'
GRAY   = '#6B7280'; LGRAY  = '#F1F5F9'
BG     = 'white'

fig = plt.figure(figsize=(17/2.54, 20/2.54), facecolor=BG)
gs  = gridspec.GridSpec(3, 2, figure=fig,
                        left=0.08, right=0.96, top=0.93, bottom=0.06,
                        hspace=0.45, wspace=0.40)

# ══════════════════════════════════════════════════════════════════════════════
# PANEL A (top-left, 2 cols wide): Energy funnel landscape
# ══════════════════════════════════════════════════════════════════════════════
ax_f = fig.add_subplot(gs[0, :])
ax_f.set_xlim(-3, 3); ax_f.set_ylim(-75, 25)
ax_f.set_facecolor(BG)
ax_f.spines[['top','right']].set_visible(False)
ax_f.spines[['left','bottom']].set_color('#CBD5E1')
ax_f.tick_params(left=True, bottom=True, labelsize=7)

# Funnel surface (parabolic energy well)
x_well = np.linspace(-3, 3, 400)
# Unbound baseline ~0
y_unbound = np.zeros_like(x_well)
# Bound state well (Morse-like)
y_bound = -70 * np.exp(-0.5*x_well**2) + 7.0*x_well**2 - 2.5

ax_f.fill_between(x_well, y_unbound, y_bound, where=(y_bound < y_unbound),
                  alpha=0.18, color=BLUE)
ax_f.plot(x_well, y_unbound, color='#94A3B8', lw=1.5, ls='--', label='Unbound')
ax_f.plot(x_well, y_bound,   color=BLUE,      lw=2.5, label='Bound state')

# Minimum point
x_min = 0.0
y_min = -70 * np.exp(-0.5*x_min**2) + 7.0*x_min**2 - 2.5
ax_f.plot(x_min, y_min, 'o', ms=10, color=TEAL,
          mec='white', mew=2, zorder=8)
ax_f.text(0.0, y_min - 5, f'ΔG = {y_min:.1f} kcal·mol⁻¹',
          ha='center', va='top', fontsize=8.5, color=TEAL, fontweight='bold')

# Energy component arrows on funnel wall
components = [
    (-2.0, -30.3, 'van der Waals\n−30.3', BLUE),
    (-1.1, -20.6, 'Electrostatic\n−20.6', PURPLE),
    ( 0.9, -18.8, 'GBSA Solvation\n−18.8', TEAL),
    ( 2.1,  +7.3, 'Entropy (+TΔS)\n+7.3',  RED),
]
for xc, yc, lbl, col in components:
    ax_f.annotate('',
        xy=(xc, y_min), xytext=(xc, yc),
        arrowprops=dict(arrowstyle='->', color=col, lw=1.5,
                        linestyle='dashed', connectionstyle='arc3,rad=0'))
    ax_f.text(xc, yc + (3 if yc > 0 else -3), lbl,
              ha='center', va='bottom' if yc > 0 else 'top',
              fontsize=6.5, color=col, fontweight='bold',
              bbox=dict(fc='white', ec=col, pad=2, boxstyle='round,pad=0.2'))

# Total ΔG bracket
ax_f.annotate('', xy=(-2.9, y_min), xytext=(-2.9, 0),
              arrowprops=dict(arrowstyle='<->', color=NAVY, lw=2.0))
ax_f.text(-3.1, y_min/2, 'ΔG_total\n−61.8\nkcal·mol⁻¹',
          ha='right', va='center', fontsize=7.5, color=NAVY, fontweight='bold')

ax_f.axhline(0, color='#CBD5E1', lw=0.8)
ax_f.set_xlabel('Binding Reaction Coordinate', fontsize=8, color=GRAY)
ax_f.set_ylabel('ΔG (kcal·mol⁻¹)', fontsize=8, color=GRAY)
ax_f.set_title('(A)  MM/PBSA Binding Energy Landscape',
               fontsize=9, fontweight='bold', pad=6)
ax_f.legend(fontsize=7, framealpha=0.8, loc='upper right')

# ══════════════════════════════════════════════════════════════════════════════
# PANEL B (mid-left): Horizontal energy component bars
# ══════════════════════════════════════════════════════════════════════════════
ax_b = fig.add_subplot(gs[1, 0])
ax_b.set_facecolor(BG)
ax_b.spines[['top','right']].set_visible(False)
ax_b.spines[['left','bottom']].set_color('#CBD5E1')

comps  = ['ΔG total', 'vdW', 'Electrostatic', 'GBSA Solv.', 'Entropy\n(−TΔS)']
vals   = [-61.8, -30.3, -20.6, -18.8, +7.3]
cols   = [NAVY, BLUE, PURPLE, TEAL, RED]
y_pos  = list(range(len(comps)))[::-1]

for y, val, c, lbl in zip(y_pos, vals, cols, comps):
    ax_b.barh(y, val, color=c, alpha=0.88, height=0.6, zorder=3,
              left=0 if val > 0 else val)
    # Bar border
    ax_b.barh(y, val, color='none', edgecolor=c, height=0.6,
              lw=0.8, zorder=4, left=0 if val > 0 else val)
    sign = '+' if val > 0 else ''
    ha = 'left' if val > 0 else 'right'
    ax_b.text(val + (1 if val > 0 else -1), y,
              f'{sign}{val}', ha=ha, va='center',
              fontsize=7.5, color=c, fontweight='bold')

ax_b.axvline(0, color='#94A3B8', lw=1.2)
ax_b.set_yticks(y_pos)
ax_b.set_yticklabels(comps, fontsize=7)
ax_b.set_xlabel('Energy (kcal·mol⁻¹)', fontsize=7.5, color=GRAY)
ax_b.set_title('(B)  Energy Components', fontsize=8.5, fontweight='bold')
ax_b.grid(axis='x', alpha=0.25, zorder=0)

# ══════════════════════════════════════════════════════════════════════════════
# PANEL C (mid-right): Per-residue ΔG heatstrip
# ══════════════════════════════════════════════════════════════════════════════
ax_c = fig.add_subplot(gs[1, 1])
ax_c.set_facecolor(BG)
ax_c.spines[['top','right']].set_visible(False)
ax_c.spines[['left','bottom']].set_color('#CBD5E1')

seq  = list('GGNRYPWNKPSK')
dG_r = np.array([-2.1,-1.8,-0.9,-4.3,-3.7,-2.6,-6.8,-1.2,-3.5,-2.9,-1.4,-4.1])
core = {6,7,8,9,10,11}

cmap_bar = LinearSegmentedColormap.from_list('rg',
    ['#1D4ED8','#3B82F6','#93C5FD','#BFDBFE'], N=128)
norm = Normalize(vmin=dG_r.min()-1, vmax=0)

for i, (aa, dg) in enumerate(zip(seq, dG_r)):
    col = cmap_bar(norm(dg))
    ec  = PURPLE if i in core else '#94A3B8'
    lw  = 2.0    if i in core else 0.8
    ax_c.bar(i, abs(dg), color=col, edgecolor=ec, linewidth=lw,
             zorder=3, bottom=0)
    ax_c.text(i, -0.55, aa, ha='center', fontsize=7.5,
              color=NAVY, fontweight='bold' if i in core else 'normal')
    ax_c.text(i, abs(dg)+0.15, f'{dg:.1f}', ha='center', fontsize=5.5,
              color='#374151')

# WNKPSK bracket
ax_c.annotate('', xy=(5.6, 7.8), xytext=(11.4, 7.8),
              arrowprops=dict(arrowstyle='<->', color=PURPLE, lw=1.5))
ax_c.text(8.5, 8.1, 'WNKPSK (epitope core)',
          ha='center', fontsize=6.5, color=PURPLE, fontweight='bold')

ax_c.set_xticks([]); ax_c.set_xlim(-0.7, 12.7)
ax_c.set_ylim(-0.8, 9.0)
ax_c.set_ylabel('|ΔG_res| (kcal·mol⁻¹)', fontsize=7, color=GRAY)
ax_c.set_title('(C)  Per-residue ΔG Contributions', fontsize=8.5, fontweight='bold')
ax_c.grid(axis='y', alpha=0.25, zorder=0)

# Colorbar inside ax_c
sm = mcm.ScalarMappable(cmap=cmap_bar, norm=norm)
sm.set_array([])
cbar = fig.colorbar(sm, ax=ax_c, fraction=0.040, pad=0.02)
cbar.set_label('ΔG (kcal/mol)', fontsize=5.5)
cbar.ax.tick_params(labelsize=5)

# ══════════════════════════════════════════════════════════════════════════════
# PANEL D (bottom, 2 cols wide): Peptide properties + KD summary box
# ══════════════════════════════════════════════════════════════════════════════
ax_d = fig.add_subplot(gs[2, :])
ax_d.set_xlim(0, 10); ax_d.set_ylim(0, 4.5)
ax_d.axis('off')

# Left: peptide property table
props = [
    ('Peptide', 'GGNRYPWNKPSK', TEAL),
    ('Molecular weight', '1402.7 Da', NAVY),
    ('Isoelectric point (pI)', '10.76', NAVY),
    ('Net charge', '+3', BLUE),
    ('Hydrophobicity (GRAVY)', '+17.25 kcal·mol⁻¹', RED),
    ('Extinction coeff.', '6990 M⁻¹·cm⁻¹', GRAY),
    ('Epitope core', 'WNKPSK (aa 99–104)', PURPLE),
]
title_box = FancyBboxPatch((0.1, 0.2), 4.6, 4.0,
    boxstyle='round,pad=0.15', fc=LGRAY, ec='#CBD5E1', lw=1.2)
ax_d.add_patch(title_box)
ax_d.text(2.4, 4.0, 'Binding Peptide Properties', ha='center',
          fontsize=8, fontweight='bold', color=NAVY)
for row_i, (key, val, col) in enumerate(props):
    y = 3.5 - row_i*0.45
    ax_d.text(0.3, y, key + ':', fontsize=6.8, color=GRAY, va='center')
    ax_d.text(2.5, y, val, fontsize=6.8, color=col, fontweight='bold', va='center')

# Right: KD & PBSA summary
kd_box = FancyBboxPatch((5.2, 0.2), 4.6, 4.0,
    boxstyle='round,pad=0.15', fc=LTEAL, ec=TEAL, lw=1.5)
ax_d.add_patch(kd_box)
ax_d.text(7.5, 4.0, 'Binding Summary', ha='center',
          fontsize=8, fontweight='bold', color=NAVY)
highlights = [
    ('K_D (affinity)',         '0.1 – 0.5 nM',     TEAL),
    ('ΔG_total (MM/PBSA)',     '−61.8 kcal·mol⁻¹', NAVY),
    ('van der Waals',          '−30.3 kcal·mol⁻¹', BLUE),
    ('Electrostatic',          '−20.6 kcal·mol⁻¹', PURPLE),
    ('GBSA Solvation',         '−18.8 kcal·mol⁻¹', TEAL),
    ('Entropy (−TΔS)',         '+7.3 kcal·mol⁻¹',  RED),
    ('Key contact residue',    'W (aa 97): −6.8',   NAVY),
]
for row_i, (key, val, col) in enumerate(highlights):
    y = 3.5 - row_i*0.45
    ax_d.text(5.4, y, key + ':', fontsize=6.8, color=GRAY, va='center')
    ax_d.text(7.8, y, val, fontsize=6.8, color=col, fontweight='bold', va='center')

# ══════════════════════════════════════════════════════════════════════════════
fig.suptitle('Figure 2.  Pritamab–PrPᶜ Binding Energy Decomposition',
             fontsize=10, fontweight='bold', color=NAVY, y=0.97)

for ext in ['png', 'pdf']:
    fpath = os.path.join(OUT, f'fig2_energy_v4.{ext}')
    fig.savefig(fpath, dpi=300, bbox_inches='tight',
                facecolor=BG, edgecolor='none')
    print(f'Saved: {fpath}')
plt.close(fig)
print('Done.')
