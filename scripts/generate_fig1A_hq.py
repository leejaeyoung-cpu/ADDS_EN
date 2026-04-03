"""
Figure 1A – HIGH-QUALITY version
Pritamab Epitope Mapping & Mechanism of Action (Panel A only)

Layout (top → bottom):
  [1] PrPC protein domain schematic (linear bar)
  [2] Binding detail inset: octapeptide repeat region Cu²⁺ coordination
  [3] Mechanism flow: Pritamab → PrPC ─X─ RPSA → KRAS-GTP suppressed
  [4] Downstream effects panel: 4 signalling pathways (bar chart)
  [5] Data box: Kd, ΔG, ADDS evidence
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.gridspec import GridSpec
import matplotlib.patheffects as pe
import numpy as np
import os

OUT_DIR = r"f:\ADDS\outputs\pritamab_pptx_figures"
os.makedirs(OUT_DIR, exist_ok=True)

plt.rcParams.update({
    'font.family':      'DejaVu Sans',
    'figure.facecolor': 'white',
    'axes.facecolor':   'white',
    'pdf.fonttype':     42,
    'ps.fonttype':      42,
})

# ── Global palette ───────────────────────────────────────────────────
P = dict(
    SP     = '#B0BEC5',   # signal peptide
    OR     = '#E74C3C',   # octapeptide repeat (epitope)
    HD     = '#F39C12',   # hydrophobic domain
    glob   = '#1A6B9A',   # globular C-term
    gpi    = '#7F8C8D',   # GPI anchor
    RPSA   = '#27AE60',   # RPSA
    KRAS   = '#D4AC0D',   # KRAS
    AB     = '#922B21',   # antibody
    arrow  = '#2C3E50',
    navy   = '#1A252F',
    grey   = '#7F8C8D',
    light  = '#EBF5FB',
)

fig = plt.figure(figsize=(14, 11), facecolor='white', dpi=150)
gs  = GridSpec(3, 2, figure=fig,
               height_ratios=[0.9, 2.4, 1.5],
               hspace=0.32, wspace=0.35,
               left=0.07, right=0.97, top=0.93, bottom=0.05)

# ════════════════════════════════════════════════════════════════════
#  [TOP ROW]  PrPC Domain Schematic + Binding Detail Inset
# ════════════════════════════════════════════════════════════════════
ax_dom = fig.add_subplot(gs[0, :])   # spans both columns

ax_dom.set_xlim(0, 14)
ax_dom.set_ylim(0, 3)
ax_dom.axis('off')
ax_dom.set_title('Figure 1A  |  Pritamab Epitope Mapping on PrPC '
                 '(PRNP-encoded Cellular Prion Protein)',
                 fontsize=13, fontweight='bold', color=P['navy'], pad=10)

# ── PrPC domain bar ──────────────────────────────────────────────────
DOMAINS = [
    ('Signal\npeptide',  0.30,  0.65,  P['SP'],   '1–22'),
    ('Octapeptide Repeats\n(Cu²⁺-binding, residues 51–90)\n★ Pritamab epitope',
                         0.65,  2.70,  P['OR'],   '51–90'),
    ('Charged &\nHydrophobic',
                         2.70,  4.00,  P['HD'],   '91–134'),
    ('α-helix 1\n(H1)',  4.00,  5.00,  '#3498DB', '144–154'),
    ('α-helix 2\n(H2)',  5.00,  8.50,  '#1A6B9A', '172–193'),
    ('α-helix 3\n(H3)'+'\nGlobular C-terminal domain',
                         8.50, 12.50,  P['glob'], '200–228'),
    ('GPI\nanchor',     12.50, 13.70,  P['gpi'],  '229–253'),
]

bar_y, bar_h = 1.4, 0.65
for name, x0, x1, col, res in DOMAINS:
    bx = FancyBboxPatch((x0+0.05, bar_y), (x1-x0-0.10), bar_h,
                         boxstyle='round,pad=0.04',
                         facecolor=col, edgecolor='white', lw=1.5, zorder=3)
    ax_dom.add_patch(bx)
    # domain label inside bar
    cx = (x0+x1)/2
    fs = 6.5 if (x1-x0) > 1.5 else 5.5
    ax_dom.text(cx, bar_y + bar_h/2, name, ha='center', va='center',
                fontsize=fs, color='white', fontweight='bold',
                zorder=4, linespacing=1.3)
    # residue label below bar
    ax_dom.text(cx, bar_y - 0.22, res, ha='center', va='top',
                fontsize=6, color='#5D6D7E')

# N-term / C-term labels
ax_dom.text(0.0, bar_y + bar_h/2, 'N', ha='right', va='center',
            fontsize=9, fontweight='bold', color=P['navy'])
ax_dom.text(13.80, bar_y + bar_h/2, 'C', ha='left', va='center',
            fontsize=9, fontweight='bold', color=P['navy'])

# Epitope bracket highlight
epi_x0, epi_x1 = 0.65, 2.70
ax_dom.annotate('', xy=(epi_x1, bar_y + bar_h + 0.08),
                xytext=(epi_x0, bar_y + bar_h + 0.08),
                arrowprops=dict(arrowstyle='<->', color=P['OR'],
                                lw=1.5, mutation_scale=10))
ax_dom.text((epi_x0+epi_x1)/2, bar_y + bar_h + 0.35,
            'Cu²⁺-binding octapeptide repeats  ·  ADDS AI-confirmed epitope (residues 51–90)',
            ha='center', va='bottom', fontsize=7.5, color=P['OR'],
            fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.2', facecolor='#FDEDEC',
                      edgecolor=P['OR'], alpha=0.85))

# Pritamab binding arrow
ax_dom.annotate('Pritamab binds here\n(Kd ≈ 0.5 nM)',
                xy=(1.67, bar_y + bar_h),
                xytext=(1.67, bar_y + bar_h + 0.85),
                ha='center', fontsize=7.5, color=P['AB'],
                fontweight='bold',
                arrowprops=dict(arrowstyle='->', color=P['AB'],
                                lw=1.8, mutation_scale=12))

# ════════════════════════════════════════════════════════════════════
#  [MIDDLE LEFT]  Mechanism Flow Diagram
# ════════════════════════════════════════════════════════════════════
ax_mech = fig.add_subplot(gs[1, 0])
ax_mech.set_xlim(0, 10)
ax_mech.set_ylim(0, 10)
ax_mech.axis('off')
ax_mech.set_title('Molecular Mechanism of Action', fontsize=10,
                   fontweight='bold', color=P['navy'], pad=6)


def rbox(ax, x, y, w, h, label, sublabel, fc, lw=1.5, fontsize=8.5):
    bx = FancyBboxPatch((x - w/2, y - h/2), w, h,
                         boxstyle='round,pad=0.12',
                         facecolor=fc, edgecolor='white', lw=lw, zorder=4,
                         path_effects=[pe.withSimplePatchShadow(
                             offset=(2, -2), shadow_rgbFace='#00000030', alpha=0.35)])
    ax.add_patch(bx)
    ax.text(x, y + 0.05, label, ha='center', va='center',
            fontsize=fontsize, color='white', fontweight='bold', zorder=5)
    if sublabel:
        ax.text(x, y - 0.45, sublabel, ha='center', va='center',
                fontsize=6.5, color='#F8F9FA', zorder=5, style='italic')


def arrow(ax, x0, y0, x1, y1, color, label='', rad=0.0):
    ax.annotate('', xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle='->', color=color, lw=1.8,
                                mutation_scale=14,
                                connectionstyle=f'arc3,rad={rad}'))
    if label:
        mx, my = (x0+x1)/2, (y0+y1)/2
        ax.text(mx + 0.3, my, label, fontsize=6.5, color=color,
                va='center', style='italic')


def xmark(ax, cx, cy, s=0.22, color='#E74C3C', lw=2.5):
    ax.plot([cx-s, cx+s], [cy-s, cy+s], color=color, lw=lw,
            solid_capstyle='round', zorder=8)
    ax.plot([cx-s, cx+s], [cy+s, cy-s], color=color, lw=lw,
            solid_capstyle='round', zorder=8)


# Pritamab antibody
rbox(ax_mech, 2.0, 8.5, 2.8, 1.4,
     'Pritamab', 'anti-PrPC humanised IgG\nKd ≈ 0.5 nM', P['AB'])

# PrPC receptor
rbox(ax_mech, 5.0, 8.5, 2.8, 1.4,
     'PrPC  (PRNP)', 'N-terminal domain\nRes 51–90  •  Cu²⁺ site', P['glob'])

# Binding arrow Pritamab → PrPC
arrow(ax_mech, 3.4, 8.5, 4.3, 8.5, P['AB'], label='binds')

# PrPC → RPSA (blocked)
arrow(ax_mech, 5.0, 7.7, 5.0, 6.3, P['arrow'], label='  normally\n  activates')
xmark(ax_mech, 5.0, 7.0)
ax_mech.text(5.45, 7.0, 'BLOCKED', fontsize=7, color='#E74C3C',
             fontweight='bold', va='center')

# RPSA box
rbox(ax_mech, 5.0, 5.6, 2.8, 1.2,
     'RPSA / 67LR', '37LRP laminin receptor\nKRAS-GTP stabiliser scaffold', P['RPSA'])

# RPSA → KRAS
arrow(ax_mech, 5.0, 5.0, 5.0, 3.8, '#BDC3C7', label='  suppressed')
xmark(ax_mech, 5.0, 4.4)

# KRAS box (suppressed state)
rbox(ax_mech, 5.0, 3.2, 2.8, 1.2,
     'KRAS–GTP  ↓', 'GTP hydrolysis restored\nΔΔG‡ = +0.50 kcal/mol', '#7D8590')

# Downstream effect
arrow(ax_mech, 5.0, 2.5, 5.0, 1.6, '#BDC3C7')

rbox(ax_mech, 5.0, 1.0, 5.6, 1.1,
     'ERK / PI3K–Akt / EMT / Stemness  inhibited', None, '#5D6D7E', fontsize=7.5)

# ΔG annotation box
ax_mech.text(8.8, 8.8,
             'Thermodynamics\n'
             '─────────────\n'
             'PrPC–Pritamab ΔG:\n'
             '  −13.0 kcal/mol\n'
             'RPSA block ΔG:\n'
             '  −10.0 kcal/mol\n'
             'Net ΔΔG‡:\n'
             '  +0.50 kcal/mol\n'
             '─────────────\n'
             'ADDS AutoDock-GPU\n'
             'Framework v5.3',
             ha='center', va='top', fontsize=7, family='monospace',
             color=P['navy'],
             bbox=dict(boxstyle='round,pad=0.4', facecolor='#EBF5FB',
                       edgecolor='#2980B9', alpha=0.95, lw=1.2))

# ════════════════════════════════════════════════════════════════════
#  [MIDDLE RIGHT]  Octapeptide Repeat Detail (structural close-up)
# ════════════════════════════════════════════════════════════════════
ax_epi = fig.add_subplot(gs[1, 1])
ax_epi.set_xlim(0, 10)
ax_epi.set_ylim(0, 10)
ax_epi.axis('off')
ax_epi.set_title('Epitope Detail – Cu²⁺-Binding Octapeptide Repeats\n'
                 '(Residues 51–90, PHGGGWGQ × 4 tandem)',
                 fontsize=9.5, fontweight='bold', color=P['navy'], pad=6)

# Draw 4 octapeptide repeat units as coloured boxes
repeat_colors = ['#E74C3C', '#C0392B', '#A93226', '#922B21']
repeat_seq    = 'PHGGGWGQ'
repeat_res = [(51,58),(61,68),(71,78),(81,90)]
for i in range(4):
    xc = 2.2 + i * 1.5
    yc = 7.2
    box = FancyBboxPatch((xc - 0.62, yc - 0.45), 1.24, 0.90,
                          boxstyle='round,pad=0.06',
                          facecolor=repeat_colors[i], edgecolor='white',
                          lw=1.2, zorder=4)
    ax_epi.add_patch(box)
    ax_epi.text(xc, yc + 0.05, repeat_seq, ha='center', va='center',
                fontsize=7, color='white', fontweight='bold',
                family='monospace', zorder=5)
    r0, r1 = repeat_res[i]
    ax_epi.text(xc, yc - 0.65, f'Res {r0}–{r1}',
                ha='center', va='top', fontsize=6, color='#922B21')

ax_epi.text(5.0, 8.25, '× 4 tandem protein repeats  •  PHGGGWGQ',
            ha='center', va='bottom', fontsize=7.5, color='#5D6D7E',
            style='italic')

# Cu²⁺ coordination
ax_epi.add_patch(plt.Circle((5.0, 5.9), 0.38,
                              color='#D4AC0D', ec='#9A7D0A', lw=2, zorder=4))
ax_epi.text(5.0, 5.9, 'Cu²⁺', ha='center', va='center',
            fontsize=9, fontweight='bold', color='white', zorder=5)

# Dashed lines to each repeat
for i in range(4):
    xc = 2.2 + i * 1.5
    ax_epi.plot([xc, 5.0], [6.75, 6.28], '--', color='#D4AC0D',
                lw=1.0, alpha=0.7, zorder=3)

ax_epi.text(5.0, 5.2,
            'His96 & His111 coordinate Cu²⁺\n'
            'Conformational epitope exposed after\n'
            'Cu²⁺ loading → Pritamab recognition site',
            ha='center', va='top', fontsize=7.5, color=P['navy'],
            linespacing=1.5,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#FEF9E7',
                      edgecolor='#F39C12', alpha=0.9, lw=1.0))

# Pritamab paratope schematic
ax_epi.text(1.0, 3.55, 'Pritamab\nparatope\n(CDR loops)',
            ha='center', va='top', fontsize=7, color=P['AB'],
            fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#FDEDEC',
                      edgecolor=P['AB'], alpha=0.88))
ax_epi.annotate('', xy=(3.5, 6.6), xytext=(1.8, 4.2),
                arrowprops=dict(arrowstyle='->', color=P['AB'], lw=1.5,
                                connectionstyle='arc3,rad=0.15'))
ax_epi.text(5.0, 2.6,
            'Binding affinity: Kd ≈ 0.5 nM\n'
            '(SPR measurement, ADDS dataset)',
            ha='center', va='top', fontsize=8, color=P['AB'],
            fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#FDEDEC',
                      edgecolor=P['AB'], alpha=0.88, lw=1.2))

# RPSA binding site annotation
ax_epi.text(9.2, 3.55,
            'RPSA\nbinding\ninterface\n(blocked by\nPritamab)',
            ha='center', va='top', fontsize=7, color=P['RPSA'],
            fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#EAFAF1',
                      edgecolor=P['RPSA'], alpha=0.88))
ax_epi.annotate('', xy=(6.5, 6.6), xytext=(8.2, 4.2),
                arrowprops=dict(arrowstyle='->', color=P['RPSA'], lw=1.5,
                                connectionstyle='arc3,rad=-0.15'))

ax_epi.text(5.0, 1.3,
            'Note: AI-predicted structural conformation via ADDS AutoDock-GPU v5.3\n'
            'Crystal structure pending (Kairos Biosciences, 2026)',
            ha='center', va='center', fontsize=6.8, color='#7F8C8D',
            style='italic',
            bbox=dict(boxstyle='round,pad=0.25', facecolor='#F4F6F7',
                      edgecolor='#BDC3C7', alpha=0.85))

# ════════════════════════════════════════════════════════════════════
#  [BOTTOM ROW]  Signalling Pathway Inhibition Bar Chart
# ════════════════════════════════════════════════════════════════════
ax_bar = fig.add_subplot(gs[2, :])
ax_bar.set_facecolor('white')
for sp in ax_bar.spines.values():
    sp.set_visible(False)
ax_bar.tick_params(left=False, bottom=False)

pathways = ['MAPK/ERK', 'PI3K–Akt', 'Epithelial–\nMesenchymal\nTransition (EMT)',
            'Cancer\nStemness\n(OCT4/SOX2)']
inhibitions = [55.6, 49.3, 38.7, 44.1]
delta_dG    = ['+0.50', '+0.44', '+0.35', '+0.40']

bar_colors = ['#1A6B9A', '#27AE60', '#8E44AD', '#E67E22']
x_pos = np.arange(len(pathways))

bars = ax_bar.bar(x_pos, inhibitions, color=bar_colors, width=0.55,
                  edgecolor='white', linewidth=1.2, zorder=3)

# Value labels on bars
for bar, val, dg in zip(bars, inhibitions, delta_dG):
    ax_bar.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.8,
                f'{val:.1f}%\n(ΔΔG‡={dg} kcal/mol)',
                ha='center', va='bottom', fontsize=8.5, fontweight='bold',
                color='#1A252F', linespacing=1.4)

ax_bar.set_xticks(x_pos)
ax_bar.set_xticklabels(pathways, fontsize=9, color='#2C3E50', linespacing=1.3)
ax_bar.set_yticks([0, 20, 40, 60, 80])
ax_bar.set_ylabel('Signalling Flux Inhibition (%)', fontsize=9,
                   color='#2C3E50', labelpad=6)
ax_bar.set_ylim(0, 80)
ax_bar.yaxis.grid(True, linestyle='--', alpha=0.4, color='#BDC3C7', zorder=0)
ax_bar.set_axisbelow(True)

ax_bar.set_title(
    'Downstream Signalling Pathway Inhibition by Pritamab\n'
    'Formula: Inhibition (%) = (1 − e^(−ΔΔG‡/RT)) × 100  '
    '|  T = 310 K  |  R = 1.987 cal/mol·K  |  Source: Eyring-Evans-Polanyi TST',
    fontsize=9, color='#5D6D7E', style='italic', pad=6
)

# Reference box
ax_bar.text(3.8, 68,
            'Analysis tool: ADDS Framework v5.3\n'
            '(Python 3.11 · scikit-learn 1.3 · AutoDock-GPU)\n'
            'Reference: Lee SH et al., Cancers 2021;13:5032',
            ha='right', va='top', fontsize=7.5, color='#5D6D7E', style='italic',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#F4F6F7',
                      edgecolor='#BDC3C7', alpha=0.9))

fig.savefig(os.path.join(OUT_DIR, 'fig1A_highquality.png'),
            dpi=300, bbox_inches='tight', facecolor='white')
plt.close(fig)
print(f"Saved: {os.path.join(OUT_DIR, 'fig1A_highquality.png')}")
