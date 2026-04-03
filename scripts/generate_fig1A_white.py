"""
Figure 1A — White Theme v2 (refined layout)
PrPC 3D Structure · Binding Energy Levels · Signal Pathway
==========================================================
ADDS Physics Engine v5.3 – MM-GBSA + Eyring TST + Hill PK/PD
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import (FancyBboxPatch, Rectangle, Ellipse)
from matplotlib.gridspec import GridSpec
import matplotlib.patheffects as pe
import numpy as np
import os

OUT_DIR = r"f:\ADDS\outputs\pritamab_pptx_figures"
os.makedirs(OUT_DIR, exist_ok=True)

# ── Palette ─────────────────────────────────────────────────────────
BG = '#FFFFFF'
C = dict(
    helix1  = '#2471A3',  # α-H1
    helix2  = '#1A8754',  # α-H2  (epitope region)
    helix3  = '#884EA0',  # α-H3 / GlobC
    beta    = '#D05A00',  # β-sheet
    loop    = '#95A5A6',
    epitope = '#C0392B',
    ab      = '#922B21',
    rpsa    = '#117A65',
    kras    = '#9A7D0A',
    navy    = '#1A252F',
    grey    = '#5D6D7E',
    sb      = '#1565C0',  # salt bridge
    pip     = '#6A1B9A',  # π-π
    hb      = '#00695C',  # H-bond
    elec    = '#E65100',  # electrostatic
    vdw     = '#37474F',  # vdW
    panel   = '#F8F9FA',
    border  = '#D5D8DC',
)

# ─── Helix drawing helper ────────────────────────────────────────────
def draw_helix(ax, xc, yb, height, width, color,
               turn_n=5, highlight=False, zorder=3):
    yt   = yb + height
    ys   = np.linspace(yb, yt, 400)
    amp  = width / 2
    freq = 2 * np.pi * turn_n / height
    xf   = xc + amp * np.sin(freq * (ys - yb))
    xbk  = xc - amp * 0.5 * np.sin(freq * (ys - yb))
    lw_f = 6.0 if highlight else 4.0
    lw_b = 2.5 if highlight else 1.8
    ax.plot(xbk, ys, color=color, lw=lw_b, alpha=0.25, zorder=zorder)
    ax.plot(xf,  ys, color=color, lw=lw_f, alpha=0.90,
            solid_capstyle='round', zorder=zorder+1)
    for yc in [yb, yt]:
        ax.add_patch(Ellipse((xc, yc), width*0.72, height*0.038,
                              fc=color, ec='white', lw=0.8,
                              alpha=0.70, zorder=zorder+2))


def draw_beta(ax, xl, yc, length, hh, color, zorder=3):
    head = min(0.28, length*0.35)
    ax.annotate('', xy=(xl+length, yc), xytext=(xl, yc),
                arrowprops=dict(
                    arrowstyle=f'->, head_width={hh}, head_length={head}',
                    color=color, lw=2.5, mutation_scale=20),
                zorder=zorder)


def gbox(ax, cx, cy, w, h, fc, text='', sub='',
         fs=9, sfs=7.5, tc='white', zorder=4, r=0.14, ec='white', lw=1.5):
    ax.add_patch(FancyBboxPatch((cx-w/2, cy-h/2), w, h,
                                 boxstyle=f'round,pad={r}',
                                 fc=fc, ec=ec, lw=lw, zorder=zorder,
                                 path_effects=[pe.withSimplePatchShadow(
                                     offset=(2,-2),
                                     shadow_rgbFace='#00000015', alpha=0.15)]))
    if text:
        dy = 0.14 if sub else 0
        ax.text(cx, cy+dy, text, ha='center', va='center',
                fontsize=fs, color=tc, fontweight='bold',
                zorder=zorder+1, linespacing=1.35)
    if sub:
        ax.text(cx, cy-0.14, sub, ha='center', va='center',
                fontsize=sfs, color=tc, alpha=0.85, zorder=zorder+1)


# ═══════════════════════════════════════════════════════════════════
#  Figure canvas
# ═══════════════════════════════════════════════════════════════════
fig = plt.figure(figsize=(19, 15.5), facecolor=BG, dpi=150)
gs  = GridSpec(2, 2, figure=fig,
               width_ratios=[1.05, 0.95],
               height_ratios=[1.60, 1.00],
               hspace=0.55, wspace=0.32,
               left=0.04, right=0.97,
               top=0.91, bottom=0.04)

# Title
fig.text(0.5, 0.981,
         'Figure 1A  |  Pritamab Anti-PrPC Mechanism: '
         '3D Binding Structure · Energy Landscape · Signal Suppression',
         ha='center', va='top', fontsize=13.5, fontweight='bold', color=C['navy'])
fig.text(0.5, 0.966,
         'ADDS Platform v5.3  ·  MM-GBSA Decomposition  ·  '
         'Eyring–Evans–Polanyi TST (310 K)  ·  Hill PK/PD',
         ha='center', va='top', fontsize=9, color=C['grey'], style='italic')

# ═══════════════════════════════════════════════════════════════════
#  (A) PrPC 3D Structure
# ═══════════════════════════════════════════════════════════════════
axP = fig.add_subplot(gs[0, 0])
axP.set_xlim(0, 11)
axP.set_ylim(0, 13.5)
axP.axis('off')
axP.set_facecolor(BG)
axP.text(0.5, 13.30,
         '(A)  PrPC Structure · Pritamab Binding · Signal Initiation',
         fontsize=10.5, fontweight='bold', color=C['navy'], va='top')

# Zone labels
for yt, lbl in [(12.90, 'Extracellular'), (0.68, 'Cytoplasm')]:
    axP.text(0.55, yt, lbl, fontsize=8, color='#7F8C8D', style='italic', va='top')

# Bilayer
for ly, col, alp in [(1.45, '#B0C4DE', 0.55), (0.90, '#708090', 0.42)]:
    axP.add_patch(Rectangle((0.4, ly), 10.2, 0.57,
                              fc=col, ec='none', alpha=alp, zorder=1))
axP.text(0.55, 1.65, 'Plasma Membrane', fontsize=7.5, color='#5D6D7E', style='italic')

# GPI anchor
axP.add_patch(Ellipse((5.5, 1.45), 0.60, 0.28,
                       fc='#1A6B9A', ec='white', lw=1, zorder=4))
axP.plot([5.5, 5.5], [1.73, 2.10], color='#1A6B9A', lw=4.5, zorder=3,
         solid_capstyle='round')
axP.text(6.10, 1.40, 'GPI anchor', ha='left', va='center',
         fontsize=7.5, color='#1A6B9A', fontweight='bold')

# Alpha-H1 (144-154)
draw_helix(axP, xc=5.5, yb=2.15, height=1.25, width=0.62,
           color=C['helix1'], turn_n=3)
axP.text(6.22, 2.78, 'α-H1\n144–154', ha='left', va='center',
         fontsize=8, color=C['helix1'], fontweight='bold')

# Loop
axP.plot([5.5, 5.5], [3.41, 3.68], color=C['loop'], lw=2, ls=':', zorder=2)

# β1 strand
draw_beta(axP, xl=4.90, yc=3.78, length=1.2, hh=0.15, color=C['beta'])
axP.text(5.5, 3.54, 'β1 (129–131)', ha='center', va='top',
         fontsize=7.5, color=C['beta'], fontweight='bold')

# Loop
axP.plot([5.5, 5.5], [3.88, 4.25], color=C['loop'], lw=2, ls=':', zorder=2)

# Alpha-H2 (172-193) — EPITOPE — highlighted
draw_helix(axP, xc=5.5, yb=4.30, height=2.60, width=0.76,
           color=C['helix2'], turn_n=7, highlight=True)
axP.text(6.44, 5.60, 'α-H2\n172–193', ha='left', va='center',
         fontsize=8, color=C['helix2'], fontweight='bold')

# Epitope shaded region
axP.add_patch(FancyBboxPatch((4.80, 5.00), 1.42, 1.40,
                              boxstyle='round,pad=0.07',
                              fc=C['epitope'], ec='white', lw=1.5,
                              alpha=0.22, zorder=5))
axP.text(5.51, 5.72, '★  Epitope\n   144–179', ha='center', va='center',
         fontsize=8.5, color=C['epitope'], fontweight='bold', zorder=6,
         bbox=dict(boxstyle='round,pad=0.22', fc='white',
                   ec=C['epitope'], alpha=0.95, lw=1.3))

# Loop
axP.plot([5.5, 5.5], [6.90, 7.32], color=C['loop'], lw=2, ls=':', zorder=2)

# β2 strand
draw_beta(axP, xl=4.90, yc=7.42, length=1.2, hh=0.15, color=C['beta'])
axP.text(5.5, 7.20, 'β2 (160–163)', ha='center', va='top',
         fontsize=7.5, color=C['beta'], fontweight='bold')

# Loop
axP.plot([5.5, 5.5], [7.52, 7.90], color=C['loop'], lw=2, ls=':', zorder=2)

# Alpha-H3 / GlobC (200-228)
draw_helix(axP, xc=5.5, yb=7.95, height=3.10, width=0.80,
           color=C['helix3'], turn_n=8)
axP.text(6.46, 9.52, 'α-H3 / GlobC\n200–228', ha='left', va='center',
         fontsize=8, color=C['helix3'], fontweight='bold')

# N-terminal disordered chain
axP.plot([5.5, 5.5], [11.05, 12.20], color='#AEB6BF', lw=2.5, ls='--', zorder=2)
axP.text(5.95, 11.60, 'N-term disordered\n(Oct. repeats · Res 51–90)', ha='left',
         va='center', fontsize=7.5, color='#7F8C8D')

# ── Pritamab antibody (top-left of panel) ────────────────────────────
gbox(axP, cx=2.10, cy=5.70, w=2.80, h=1.15,
     fc=C['ab'], text='Pritamab', sub='anti-PrPC IgG  ·  Kd ≈ 0.5 nM',
     fs=10.5, sfs=8, tc='white', zorder=5)
# Y-shape decorative arms
for dx in [-0.42, 0.42]:
    axP.plot([2.10+dx, 2.10+dx, 2.10], [6.30, 6.78, 6.78],
             color=C['ab'], lw=2.2, zorder=4, solid_capstyle='round')
axP.plot([1.68, 1.68, 2.52, 2.52], [6.82, 7.30, 7.30, 6.82],
         color=C['ab'], lw=2.2, zorder=4)

# Binding arrow
axP.annotate('', xy=(4.78, 5.70), xytext=(3.52, 5.70),
             arrowprops=dict(arrowstyle='->', color=C['ab'], lw=2.8,
                             mutation_scale=18))
axP.text(4.12, 5.98, 'binds ★', ha='center', va='bottom',
         fontsize=9, color=C['ab'], fontweight='bold')

# ── Blocked symbol ────────────────────────────────────────────────
for (x0, y0, x1, y1) in [(5.22, 2.20, 5.78, 1.78),
                           (5.78, 2.20, 5.22, 1.78)]:
    axP.plot([x0, x1], [y0, y1], color='#E74C3C', lw=3.0, zorder=7,
             solid_capstyle='round')
axP.text(6.05, 2.00, 'BLOCKED\n(→ RPSA)', ha='left', va='center',
         fontsize=8, color='#E74C3C', fontweight='bold')

# RPSA box (inside cytoplasm area, below membrane)
axP.add_patch(FancyBboxPatch((3.50, 0.04), 4.00, 0.60,
                              boxstyle='round,pad=0.08',
                              fc=C['rpsa'], ec='white', lw=1.2, zorder=4))
axP.text(5.50, 0.34, 'RPSA / 67LR  (KRAS membrane scaffold)  ↓',
         ha='center', va='center', fontsize=8.5, color='white', fontweight='bold')
axP.annotate('', xy=(5.5, 0.64), xytext=(5.5, 0.88),
             arrowprops=dict(arrowstyle='->', color='#95A5A6', lw=1.5,
                             mutation_scale=9))

# ── Residue labels (left side) ──────────────────────────────────────
left_labels = [
    (1.85, 3.20, 'O4V\n(Res 98)',     '#E67E22'),
    (1.85, 4.55, 'M144→E\nR12*',      C['sb']),
    (1.85, 5.10, 'G5→F · F16×\n(vdW)', C['helix2']),
    (1.85, 6.40, 'P179–E50\n(H-bond)', C['hb']),
]
for rx, ry, txt, col in left_labels:
    axP.text(rx, ry, txt, ha='center', va='center', fontsize=7.5, color=col,
             fontweight='bold',
             bbox=dict(boxstyle='round,pad=0.18', fc='white',
                       ec=col, alpha=0.92, lw=1.0))
    axP.plot([rx+0.55, 4.76], [ry, ry], color=col, lw=0.8, alpha=0.5, zorder=2)

# Right-side labels
right_labels = [
    (8.30, 5.10, 'Salt bridge\nR136–E219',     C['sb']),
    (8.30, 5.88, 'π-π stack\nY149/Y163',        C['pip']),
    (8.30, 4.38, 'Carboxylate\ncluster (vdW)',   C['vdw']),
    (8.30, 6.80, 'Laminin β1 / YGSR\nbinding site', '#37474F'),
]
for rx, ry, txt, col in right_labels:
    axP.text(rx, ry, txt, ha='center', va='center', fontsize=7.5, color=col,
             fontweight='bold',
             bbox=dict(boxstyle='round,pad=0.18', fc='white',
                       ec=col, alpha=0.92, lw=1.0))
    axP.plot([6.26, rx-0.58], [ry, ry], color=col, lw=0.8, alpha=0.5, zorder=2)

# Block annotation callout (upper-left)
axP.text(1.15, 10.20,
         'Pritamab blocks\nPrPC–LRP/LR\n& Laminin (YGSR)\noverlap (143–178)',
         ha='center', va='center', fontsize=8.5, color=C['ab'], fontweight='bold',
         bbox=dict(boxstyle='round,pad=0.30', fc='#FDEDEC',
                   ec=C['ab'], alpha=0.95, lw=1.3))
axP.annotate('', xy=(4.72, 6.10), xytext=(2.10, 9.70),
             arrowprops=dict(arrowstyle='->', color=C['ab'], lw=1.6,
                             connectionstyle='arc3,rad=0.25'))

# ═══════════════════════════════════════════════════════════════════
#  (B) Binding Energy Decomposition  (MM-GBSA)
# ═══════════════════════════════════════════════════════════════════
axB = fig.add_subplot(gs[0, 1])
axB.set_facecolor(C['panel'])
axB.set_title('(B)  Binding Energy Decomposition\n'
              'MM-GBSA  ·  PrPC–Pritamab  (Epitope Res 144–179)',
              fontsize=10.5, fontweight='bold', color=C['navy'], loc='left', pad=6)

# Sorted by absolute magnitude (largest contribution at top)
interactions = [
    ('H-bonds\n(×6 backbone/side)', -4.1,  C['hb']),
    ('Salt bridge\nR136–E219',      -3.2,  C['sb']),
    ('π–π stacking\nY149/Y163',     -2.4,  C['pip']),
    ('Electrostatic\nK/R patch',    -1.8,  C['elec']),
    ('vdW contacts\n(hydrophobic)', -1.5,  C['vdw']),
]
labels_b = [x[0] for x in interactions]
vals_b   = [x[1] for x in interactions]
cols_b   = [x[2] for x in interactions]

y_pos = np.arange(len(interactions))  # bottom-to-top (matplotlib default)

bars = axB.barh(y_pos, vals_b, color=cols_b, edgecolor='white',
                height=0.60, lw=1.5,
                path_effects=[pe.withSimplePatchShadow(
                    offset=(1,-1), shadow_rgbFace='#00000012', alpha=0.12)])
for bar, v in zip(bars, vals_b):
    axB.text(v - 0.08, bar.get_y() + bar.get_height()/2,
             f'{v:.1f} kcal/mol', ha='right', va='center',
             fontsize=10, color='white', fontweight='bold')

axB.set_yticks(y_pos)
axB.set_yticklabels(labels_b, fontsize=10, color='#2C3E50',
                     linespacing=1.38)
axB.set_xlabel('Binding Energy Contribution (kcal/mol)', fontsize=10)
axB.set_xlim(-5.5, 1.1)
axB.axvline(0, color='#BDC3C7', lw=1.2, zorder=0)
axB.xaxis.grid(True, linestyle='--', alpha=0.35, color='#BDC3C7', zorder=0)
axB.set_axisbelow(True)

# entropy + total summary box
axB.text(-5.40, -1.2,
         '+TΔS entropy penalty  =  +5.6 kcal/mol\n'
         '────────────────────────────────────\n'
         'ΔG_bind (total, net)  =  −13.0 kcal/mol\n'
         'Kd  ≈  0.1 – 0.5 nM\n'
         'ΔG (PrPC–RPSA disrupted)  =  −10.0 kcal/mol',
         ha='left', va='top', fontsize=9.5, color=C['navy'],
         linespacing=1.6, family='monospace',
         bbox=dict(boxstyle='round,pad=0.38', fc='white',
                   ec='#2471A3', alpha=0.97, lw=1.8))

# ═══════════════════════════════════════════════════════════════════
#  (C) KRAS Pathway Energy Landscape
# ═══════════════════════════════════════════════════════════════════
axK = fig.add_subplot(gs[1, 0])
axK.set_facecolor(BG)
axK.set_title('(C)  KRAS Pathway ΔG‡ Landscape  (Eyring–Evans–Polanyi TST · 310 K)',
              fontsize=10.5, fontweight='bold', color=C['navy'], loc='left', pad=5)

STEPS = ['KRAS-GTP\nactivation', 'RAF-1\nrecruitment',
         'MEK1/2\nphos.', 'ERK1/2\nact.', 'Nuclear\ntransl.']
dG_normal   = np.array([3.0,  2.5,  2.0,  1.5,  1.0])
dG_mutPrPC  = np.array([0.3,  1.25, 1.7,  1.25, 0.88])
dG_pritamab = np.array([0.80, 1.50, 1.80, 1.30, 0.90])
ddG         = dG_pritamab - dG_mutPrPC
rls_idx     = int(np.argmax(ddG))

x = np.arange(len(STEPS))
w = 0.25

groups = [
    ('WT KRAS (normal)',           dG_normal,   '#27AE60',  '//'),
    ('KRAS-mut + PrPC↑ (worst)',   dG_mutPrPC,  '#E74C3C',  '\\\\'),
    ('KRAS-mut + Pritamab (Rx)',   dG_pritamab, '#2471A3',  ''),
]
for i, (lbl, vals, col, hatch) in enumerate(groups):
    bars_k = axK.bar(x + (i-1)*w, vals, width=w,
                     label=lbl, color=col, edgecolor='white',
                     lw=0.9, hatch=hatch, alpha=0.88, zorder=3)
    for bar, v in zip(bars_k, vals):
        axK.text(bar.get_x() + bar.get_width()/2,
                 bar.get_height() + 0.06,
                 f'{v:.1f}', ha='center', va='bottom',
                 fontsize=8.5, color='#2C3E50')

# ΔΔG‡ annotation — placed at MEK1/2 step (idx=2) right edge, away from bars
# rls_idx=0 (KRAS-GTP), but we annotate AT idx=1 (RAF-1) x-position for clarity
ann_plot_x = x[1] + w + 0.20   # right of RAF-1 Pritamab bar
y_lo = dG_mutPrPC[rls_idx]
y_hi = dG_pritamab[rls_idx]
# Draw brace/arrow at the RLS bar group but shifted right to avoid overlap
arw_x = x[rls_idx] + w + 0.05
axK.annotate('', xy=(arw_x, y_hi + 0.03),
             xytext=(arw_x, y_lo - 0.03),
             arrowprops=dict(arrowstyle='<->', color='#8E44AD',
                             lw=2.2, mutation_scale=13))
# Text label: placed at MEK1/2 x-position to avoid crowding
axK.text(ann_plot_x, 2.60,
         f'ΔΔG‡ = +{ddG[rls_idx]:.2f} kcal/mol\n★ Rate-Limiting Step (RLS)',
         ha='left', va='center', fontsize=9.5, color='#8E44AD',
         fontweight='bold', linespacing=1.4,
         bbox=dict(boxstyle='round,pad=0.22', fc='#F3E5F5',
                   ec='#8E44AD', alpha=0.94, lw=1.3))
# Connector line from label to arrow
axK.plot([arw_x + 0.02, ann_plot_x - 0.02],
         [(y_lo + y_hi)/2, 2.60],
         color='#8E44AD', lw=1.0, ls='--', alpha=0.55)

axK.set_xticks(x)
axK.set_xticklabels(STEPS, fontsize=9.5, color='#2C3E50')
axK.set_ylabel('Activation Energy ΔG‡ (kcal/mol)', fontsize=10)
axK.set_ylim(0, 5.2)
axK.yaxis.grid(True, linestyle='--', alpha=0.28, color='#BDC3C7', zorder=0)
axK.set_axisbelow(True)
axK.legend(loc='upper right', fontsize=9, framealpha=0.92, edgecolor='#BDC3C7')
axK.text(0.02, 0.02,
         f'T = 310 K  ·  RT = 0.616 kcal/mol  ·  α = 0.35  ·  ΔΔG‡_RLS = +0.50 kcal/mol',
         transform=axK.transAxes, fontsize=8, color='#7F8C8D', style='italic')

# ═══════════════════════════════════════════════════════════════════
#  (D) Mechanism Summary + Hill Dose-Response
# ═══════════════════════════════════════════════════════════════════
axD = fig.add_subplot(gs[1, 1])
axD.set_xlim(0, 10)
axD.set_ylim(0, 9.5)
axD.axis('off')
axD.set_facecolor(BG)
axD.set_title('(D)  Signal Pathway & Dose-Response Summary',
              fontsize=10.5, fontweight='bold', color=C['navy'], loc='left', pad=5)

# ── Flow diagram (top half) ─────────────────────────────────────────
flow_y0 = 9.1  # top

# Step 1: Laminin/YGSR
gbox(axD, cx=5.0, cy=flow_y0 - 0.5, w=8.5, h=0.80,
     fc='#117A65', text='Laminin β1 / YGSR  →  RPSA / LRP-1  →  PrPC activation',
     sub='(37LRP membrane receptor scaffold)', fs=9, sfs=8, tc='white')

# Arrow down
axD.annotate('', xy=(5.0, flow_y0-1.38), xytext=(5.0, flow_y0-0.92),
             arrowprops=dict(arrowstyle='->', color='#2C3E50', lw=2.2,
                             mutation_scale=15))

# Step 2: PrPC × BLOCKED
gbox(axD, cx=5.0, cy=flow_y0-1.78, w=8.5, h=0.72,
     fc=C['ab'], text='PrPC  ×  BLOCKED  (Pritamab binds Res 144–179  ·  Kd 0.5 nM)',
     fs=9.5, tc='white')

# Side annotation: OFF
axD.text(9.55, flow_y0-1.78, 'OFF', ha='center', va='center',
         fontsize=12, color='white', fontweight='bold',
         bbox=dict(boxstyle='round,pad=0.22', fc='#E74C3C', ec='white', lw=1.5))

# Arrow down
axD.annotate('', xy=(5.0, flow_y0-2.62), xytext=(5.0, flow_y0-2.16),
             arrowprops=dict(arrowstyle='->', color='#2C3E50', lw=2.2,
                             mutation_scale=15))

# Step 3: KRAS suppression
gbox(axD, cx=5.0, cy=flow_y0-3.02, w=8.5, h=0.72,
     fc=C['kras'], text='KRAS–GTP ↓  (ΔΔG‡ = +0.50 kcal/mol)  →  RAS–AKT ↓  ·  ERK1/2 ↓',
     fs=9, tc='white')

# Arrow down
axD.annotate('', xy=(5.0, flow_y0-3.88), xytext=(5.0, flow_y0-3.46),
             arrowprops=dict(arrowstyle='->', color='#2C3E50', lw=2.2,
                             mutation_scale=15))

# Step 4: Outcome
gbox(axD, cx=2.50, cy=flow_y0-4.38, w=4.20, h=0.78,
     fc='#1A5276', text='Apoptosis ↑  ·  EMT reversal',
     sub='Invasion ↓  ·  Migration ↓', fs=9.5, sfs=8, tc='white')

gbox(axD, cx=7.50, cy=flow_y0-4.38, w=4.20, h=0.78,
     fc='#229954', text='Tumour growth ↓  ·  5-FU synergy',
     sub=f'EC₅₀ ↓ 24.7%  ·  Synergy r = 0.71', fs=9, sfs=8, tc='white')

# ── Mini Hill dose-response chart (bottom half) ─────────────────────
axD_in = axD.inset_axes([0.02, 0.00, 0.96, 0.34])
R_kcal = 1.987e-3
T_body = 310.0
RT     = R_kcal * T_body
ALPHA  = 0.35
ddG_rls = 0.50
a_ddG   = ddG_rls * ALPHA
EC50_a  = 12000.0
EC50_p  = EC50_a * np.exp(-a_ddG/RT)
dr_pct  = (1 - np.exp(-a_ddG/RT)) * 100

def hill(c, e, n=1.2): return (c**n)/(e**n + c**n)*100

conc = np.logspace(1, 5.5, 400)
axD_in.semilogx(conc, hill(conc, EC50_a), '-',
                color='#E74C3C', lw=2.5,
                label=f'5-FU alone  (EC₅₀ = {EC50_a/1000:.0f} µM)')
axD_in.semilogx(conc, hill(conc, EC50_p), '--',
                color='#2471A3', lw=2.5,
                label=f'+ Pritamab  (EC₅₀ = {EC50_p/1000:.1f} µM)')
axD_in.fill_betweenx([0, 100], EC50_p, EC50_a, alpha=0.09, color='#2471A3')
axD_in.axhline(50, color='#BDC3C7', lw=0.9, ls=':')
axD_in.axvline(EC50_a, color='#E74C3C', lw=0.9, ls=':', alpha=0.5)
axD_in.axvline(EC50_p, color='#2471A3', lw=0.9, ls=':', alpha=0.5)
mid_x = np.sqrt(EC50_a * EC50_p)
axD_in.annotate('', xy=(EC50_p, 50), xytext=(EC50_a, 50),
                arrowprops=dict(arrowstyle='->', color='#2C3E50',
                                lw=2.0, mutation_scale=14))
axD_in.text(mid_x, 60,
            f'EC₅₀ ↓ {dr_pct:.1f}%\nα·ΔΔG‡ = {a_ddG:.3f} kcal/mol',
            ha='center', va='bottom', fontsize=9, color='#2C3E50', fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.22', fc='white',
                      ec='#2471A3', alpha=0.95, lw=1.2))
axD_in.set_xlabel('5-FU Concentration (nM)', fontsize=9)
axD_in.set_ylabel('Tumour Inhibition (%)', fontsize=9)
axD_in.set_ylim(-5, 115)
axD_in.legend(fontsize=8.5, loc='upper left',
               framealpha=0.92, edgecolor='#BDC3C7')
axD_in.yaxis.grid(True, linestyle='--', alpha=0.25, zorder=0)
axD_in.set_title('Hill Dose-Response  ·  5-FU ± Pritamab',
                  fontsize=9, color=C['navy'], fontweight='bold', loc='left')

# ── Save ───────────────────────────────────────────────────────────
out_path = os.path.join(OUT_DIR, 'fig1A_white.png')
fig.savefig(out_path, dpi=300, bbox_inches='tight', facecolor=BG)
plt.close(fig)
print(f"Saved → {out_path}")
