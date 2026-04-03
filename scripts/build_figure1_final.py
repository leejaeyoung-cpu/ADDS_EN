"""
Figure 1 — Pritamab·PrPᶜ Binding & Mechanism
==============================================
Exact style match to reference image:
  Panel A (left 40%):
    - PrPC molecular ribbon cartoon (helices as flat ribbons, β-sheets as arrows,
      loops as thin coil lines)  — white background
    - Binding Energies table (right of ribbon)
    - Mechanism Summary bar chart (below table)

  Panel B (right 60%):
    - 2×2 grid of drug combo panels
    - Each cell: YIGSR oval + 67LR teal ribbon + ERK/PI3K X-marks + statistics
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['font.family']        = 'DejaVu Sans'  # clean sans-serif
plt.rcParams['axes.unicode_minus'] = False

import matplotlib.patches as mpatches
from matplotlib.patches import (FancyArrowPatch, FancyBboxPatch,
                                 Ellipse, Arc, Polygon, PathPatch)
from matplotlib.path import Path
import matplotlib.gridspec as gridspec
import os

OUT = r'f:\ADDS\pritamab\figures'
os.makedirs(OUT, exist_ok=True)

# ── palette ─────────────────────────────────────────────────────────────────
TEAL   = '#0D9488'
DTEAL  = '#065F46'
LTEAL  = '#A7F3D0'
GREEN  = '#15803D'
YELLOW = '#CA8A04'
GOLD   = '#F59E0B'
PURPLE = '#6D28D9'
LPURP  = '#DDD6FE'
NAVY   = '#1E3A5F'
ORANGE = '#EA580C'
RED    = '#DC2626'
GRAY   = '#6B7280'
LGRAY  = '#F3F4F6'
WHITE  = 'white'

# ═══════════════════════════════════════════════════════════════════════════
#  MOLECULAR CARTOON HELPERS
# ═══════════════════════════════════════════════════════════════════════════
def ribbon_helix(ax, x0, y0, length, angle_deg=90,
                 width=0.22, n_turns=3.5,
                 fc=TEAL, ec=DTEAL, lw=0.8, alpha=1.0, zorder=5):
    """
    Draw a ribbon α-helix as a flat sinusoidal polygon.
    angle_deg: orientation of the helix axis (degrees from +x).
    """
    from matplotlib.patches import PathPatch
    ang   = np.radians(angle_deg)
    ux, uy = np.cos(ang), np.sin(ang)       # along helix
    px, py = -uy, ux                         # perpendicular

    t = np.linspace(0, 1, 200)
    # backbone centre
    cx = x0 + t*length*ux
    cy = y0 + t*length*uy
    # ribbon half-width oscillates
    hw = width * np.abs(np.cos(n_turns*2*np.pi*t))

    top_x = cx + hw*px; top_y = cy + hw*py
    bot_x = cx - hw*px; bot_y = cy - hw*py

    xs = np.concatenate([top_x, bot_x[::-1]])
    ys = np.concatenate([top_y, bot_y[::-1]])

    verts = list(zip(xs, ys))
    verts.append(verts[0])
    codes = [Path.MOVETO] + [Path.LINETO]*(len(verts)-2) + [Path.CLOSEPOLY]
    patch = PathPatch(Path(verts, codes),
                      fc=fc, ec=ec, lw=lw, alpha=alpha, zorder=zorder)
    ax.add_patch(patch)
    # Highlight stripe
    stripe_x = cx + 0.04*px
    stripe_y = cy + 0.04*py
    ax.plot(stripe_x, stripe_y, color='white', lw=0.6, alpha=0.45,
            zorder=zorder+1)


def beta_arrow(ax, x0, y0, length, angle_deg=0,
               width=0.18, head_w=0.28, fc=GOLD, ec='#92400E',
               lw=0.8, zorder=5):
    """Draw a β-strand as a filled arrow."""
    ang = np.radians(angle_deg)
    ux, uy = np.cos(ang), np.sin(ang)
    px, py = -uy, ux

    shaft_l = length * 0.7
    head_l  = length * 0.3

    # shaft rectangle
    s_pts = [
        (x0 + shaft_l*ux + width*px, y0 + shaft_l*uy + width*py),
        (x0 + shaft_l*ux - width*px, y0 + shaft_l*uy - width*py),
        (x0             - width*px,  y0             - width*py),
        (x0             + width*px,  y0             + width*py),
    ]
    ax.add_patch(Polygon(s_pts, fc=fc, ec=ec, lw=lw, zorder=zorder))

    # arrowhead triangle
    tip_x = x0 + length*ux; tip_y = y0 + length*uy
    h_pts = [
        (x0 + shaft_l*ux + head_w*px, y0 + shaft_l*uy + head_w*py),
        (tip_x, tip_y),
        (x0 + shaft_l*ux - head_w*px, y0 + shaft_l*uy - head_w*py),
    ]
    ax.add_patch(Polygon(h_pts, fc=fc, ec=ec, lw=lw, zorder=zorder))


def coil_loop(ax, pts, color='#374151', lw=1.0, zorder=4):
    """Thin smoothed coil loop through pts."""
    from scipy.interpolate import splprep, splev
    import warnings; warnings.filterwarnings('ignore')
    xs, ys = [p[0] for p in pts], [p[1] for p in pts]
    if len(pts) >= 4:
        tck, _ = splprep([xs, ys], s=0, k=3)
        t_new  = np.linspace(0, 1, 300)
        xi, yi = splev(t_new, tck)
        ax.plot(xi, yi, color=color, lw=lw, zorder=zorder)
    else:
        ax.plot(xs, ys, color=color, lw=lw, zorder=zorder)


# ═══════════════════════════════════════════════════════════════════════════
#  PANEL A — PrPC ribbon + tables
# ═══════════════════════════════════════════════════════════════════════════
def draw_panel_A(ax):
    ax.set_xlim(0, 10.5)
    ax.set_ylim(0, 18)
    ax.axis('off')
    ax.set_facecolor(WHITE)

    # ── PrPC ribbon structure (left half, x=0.5–5, y=3–17) ──────────────
    # Proportions roughly match Figure 1A reference

    # Label
    ax.text(2.3, 17.3, 'PrPC', fontsize=9, fontweight='bold',
            color=NAVY, ha='center')

    # N-term disordered tail (coil, top-right area → curves down-left)
    coil_loop(ax,
        [(2.8,16.8),(3.5,16.4),(3.8,15.8),(3.4,15.2),(2.8,14.8),(2.2,14.5)],
        color='#9CA3AF', lw=1.2)

    # H1 helix (short, lower-middle left, ~45°)
    ribbon_helix(ax, 1.5, 11.5, 1.8, angle_deg=75, n_turns=2.5,
                 fc='#059669', ec='#065F46', width=0.20, zorder=6)

    # β1 strand
    beta_arrow(ax, 2.2, 10.8, 1.0, angle_deg=-15,
               width=0.13, head_w=0.22, fc=GOLD, ec='#92400E', zorder=6)

    # Connecting loop
    coil_loop(ax, [(2.2,10.8),(1.9,10.4),(1.7,9.9),(2.0,9.5)],
              color='#9CA3AF', lw=1.0)

    # H2 helix (main long central helix, ~80° nearly vertical)
    ribbon_helix(ax, 1.8, 7.0, 3.8, angle_deg=82, n_turns=4.5,
                 fc=TEAL, ec=DTEAL, width=0.26, zorder=7)

    # β2 strand (below H2)
    beta_arrow(ax, 1.5, 6.5, 1.0, angle_deg=5,
               width=0.13, head_w=0.22, fc=GOLD, ec='#92400E', zorder=6)

    # Loop H2→H3
    coil_loop(ax, [(2.1,6.4),(2.5,6.0),(2.9,5.7),(3.2,5.5),(3.4,5.2)],
              color='#9CA3AF', lw=1.0)

    # H3 helix (right side, ~100° tilted)
    ribbon_helix(ax, 2.8, 4.0, 3.5, angle_deg=95, n_turns=4.0,
                 fc='#047857', ec='#064E3B', width=0.28, zorder=7)

    # C-terminal tail
    coil_loop(ax, [(2.9,4.0),(2.4,3.5),(2.0,3.2),(1.6,3.0)],
              color='#9CA3AF', lw=1.0)

    # ── YIGSR / Laminin contact label lines ──────────────────────────────
    def ann_line(ax, x1, y1, x2, y2, txt, col, fs=5.8, ha='left'):
        ax.annotate('', xy=(x1, y1), xytext=(x2, y2),
                    arrowprops=dict(arrowstyle='-', color=col, lw=0.9))
        ax.text(x2 + (0.05 if ha=='left' else -0.05), y2, txt,
                fontsize=fs, color=col, ha=ha, va='center')

    ann_line(ax, 2.1,14.9, 0.5,15.5,
             'Pritamab blocks\nPrPC-LVV/LR\n& Laminin (YIGSR) overlap',
             NAVY, fs=5.2, ha='left')

    ann_line(ax, 2.9,13.0, 4.2,13.5,
             'Laminin\n\u03b21',  GRAY, fs=5.5)

    ann_line(ax, 2.0,10.8, 0.4,9.8,
             'Pritamab\nblocks\nPiPC-LAP/LR\n& Laminin\nING-1TE overlap',
             NAVY, fs=4.8, ha='left')

    # Residue markers (dashed lines matching reference)
    residues = [
        (1.95, 12.8, 'Gf7'),   (2.05, 12.2, 'O51-Sa'),
        (2.15, 11.6, 'pit'),   (1.85, 11.0, 'Pit4-C5'),
        (2.35, 9.6,  'GRAMERER'),
        (2.50, 9.0,  'Gt+P'),  (2.65, 8.4,  'Pit4-CG'),
    ]
    for rx, ry, rlbl in residues:
        ax.plot(rx, ry, 'o', ms=2.5, color='#EAB308', zorder=8)
        ax.text(rx+0.12, ry, rlbl, fontsize=4.2, color='#374151', va='center')

    # Laminin β1 / P(YIGSR) callout arc on right of helix
    ann_line(ax, 3.5, 6.0, 4.8, 6.2, 'Laminin β1', GRAY, fs=5.5)
    ax.add_patch(Ellipse((4.4,5.0), 1.0, 0.45,
                         fc=PURPLE, ec='white', lw=1.0, zorder=9))
    ax.text(4.4, 5.0, 'P(YIGSR)', ha='center', va='center',
            fontsize=5.0, color='white', fontweight='bold', zorder=10)

    ann_line(ax, 3.0, 4.5, 4.6, 4.0,
             'PrPC-CB', NAVY, fs=5.2)
    ann_line(ax, 2.0, 3.5, 0.4, 3.2,
             'PrPC-LRP/LR\n2 Laminin\n(MS-79) onhap',
             NAVY, fs=5.0, ha='left')

    # ── Binding Energies table (right side) ──────────────────────────────
    tx0, ty0 = 5.2, 17.2   # table top-left
    tw, th    = 5.1, 5.5

    # Table box
    ax.add_patch(FancyBboxPatch((tx0, ty0-th), tw, th,
        boxstyle='square,pad=0.0', fc=LGRAY, ec='#CBD5E1', lw=1.2))

    # Header bar
    ax.add_patch(FancyBboxPatch((tx0, ty0-0.55), tw, 0.55,
        boxstyle='square,pad=0', fc=NAVY, ec='none'))
    ax.text(tx0+tw/2, ty0-0.28, 'Binding Energies',
            ha='center', va='center', fontsize=8.5,
            fontweight='bold', color='white')

    ax.text(tx0+0.15, ty0-0.85,
            'PrPC - Pritamab  (epitope 144\u2013179)',
            fontsize=6.8, color=NAVY, fontweight='bold')

    rows = [
        ('Salt-bridge',    '− 30  kcal/mol'),
        ('\u03c0\u2013\u03c0 stacking',   '− S.S.  kcal/mol'),
        ('H-bond',         '− 8    kcal/mol'),
        ('Electrostatic',  '− 12.5 kcal/mol'),
        ('vdW',            '− 4.3  kcal/mol'),
    ]
    for ri, (k, v) in enumerate(rows):
        y = ty0 - 1.55 - ri*0.72
        # alternating row shade
        if ri % 2 == 0:
            ax.add_patch(FancyBboxPatch((tx0, y-0.30), tw, 0.62,
                boxstyle='square,pad=0', fc='white', ec='none', alpha=0.55))
        ax.text(tx0+0.20, y, k, fontsize=7.0, color='#374151', va='center')
        ax.text(tx0+tw-0.15, y, v, fontsize=7.0, color=NAVY,
                va='center', ha='right', fontweight='bold')

    # Divider
    div_y = ty0 - 5.10
    ax.plot([tx0+0.1, tx0+tw-0.1], [div_y, div_y],
            color='#94A3B8', lw=0.9, ls='--')

    # Totals
    ax.text(tx0+0.20, div_y-0.50,
            'Total  \u0394G  :  − 61.8  kcal/mol',
            fontsize=7.5, fontweight='bold', color=NAVY)
    ax.text(tx0+0.20, div_y-1.05,
            'K\u2082  :    0.1 \u2013 0.5 nM',
            fontsize=7.5, fontweight='bold', color=NAVY)

    # ── Mechanism Summary (below table) ──────────────────────────────────
    mx0, my0 = 5.2, 11.5
    mw        = 5.1

    ax.add_patch(FancyBboxPatch((mx0, my0-3.8), mw, 3.8,
        boxstyle='square,pad=0', fc=LGRAY, ec='#CBD5E1', lw=1.2))
    ax.add_patch(FancyBboxPatch((mx0, my0-0.55), mw, 0.55,
        boxstyle='square,pad=0', fc=NAVY, ec='none'))
    ax.text(mx0+mw/2, my0-0.28, 'Mechanism Summary',
            ha='center', va='center', fontsize=8.5,
            fontweight='bold', color='white')

    # Row 1: Laminin β1/YGSR → S1/S/ LR (teal fill bar)
    bar_y1 = my0 - 1.50
    ax.add_patch(FancyBboxPatch((mx0+0.15, bar_y1-0.30), mw-0.30, 0.60,
        boxstyle='round,pad=0.05', fc=TEAL, ec='none', alpha=0.90))
    ax.text(mx0+mw/2, bar_y1,
            'Laminin \u03b21/YGSR  \u2192  S1/S/ LR',
            ha='center', va='center', fontsize=7.0,
            color='white', fontweight='bold')

    # Arrow ↓
    ax.annotate('', xy=(mx0+mw/2, my0-2.15),
                xytext=(mx0+mw/2, my0-1.85),
                arrowprops=dict(arrowstyle='->', color=GRAY, lw=1.2))

    # Row 2: PrPC→OFF + Invasion↓ bar
    bar_y2 = my0 - 2.55
    ax.add_patch(FancyBboxPatch((mx0+0.15, bar_y2-0.30), mw-0.30, 0.60,
        boxstyle='round,pad=0.05', fc=ORANGE, ec='none', alpha=0.90))
    ax.text(mx0+mw/2, bar_y2,
            'PrPC \u2192 4OFF   Invasion \u25b2  OFF',
            ha='center', va='center', fontsize=7.0,
            color='white', fontweight='bold')

    # Row 3: outcome arrows
    out_y = my0 - 3.35
    ax.annotate('', xy=(mx0+1.5, out_y-0.25), xytext=(mx0+1.5, out_y+0.25),
                arrowprops=dict(arrowstyle='->', color=GREEN, lw=1.8))
    ax.text(mx0+1.5, out_y-0.0, 'Apoptosis \u2191',
            ha='center', fontsize=7.2, color=GREEN, fontweight='bold')
    ax.annotate('', xy=(mx0+3.6, out_y+0.25), xytext=(mx0+3.6, out_y-0.25),
                arrowprops=dict(arrowstyle='->', color=RED, lw=1.8))
    ax.text(mx0+3.6, out_y-0.0, 'Invasion \u2191',
            ha='center', fontsize=7.2, color=RED, fontweight='bold')

    ax.set_title('A', fontsize=12, fontweight='bold', loc='left', pad=6)


# ═══════════════════════════════════════════════════════════════════════════
#  67LR PROTEIN ICON  (reusable for Panel B)
# ═══════════════════════════════════════════════════════════════════════════
def draw_67LR(ax, cx, cy, scale=1.0, alpha=1.0):
    """
    Draw a simplified 67LR-like protein cartoon:
    - Two helices flanking a central stem
    - Transmembrane base (rectangle)
    Coordinates centered at (cx, cy).
    """
    s = scale
    # Transmembrane base
    ax.add_patch(FancyBboxPatch((cx-0.5*s, cy-1.4*s), 1.0*s, 0.55*s,
        boxstyle='round,pad=0.03',
        fc='#374151', ec='#1E293B', lw=0.8*s, alpha=alpha))

    # Central stem
    ax.add_patch(FancyBboxPatch((cx-0.12*s, cy-0.85*s), 0.24*s, 1.2*s,
        boxstyle='round,pad=0.03',
        fc=TEAL, ec=DTEAL, lw=0.6*s, alpha=alpha))

    # Left helix (tilted)
    ribbon_helix(ax, cx-0.8*s, cy+0.15*s, 1.3*s, angle_deg=115,
                 n_turns=2, width=0.16*s, fc='#059669', ec='#065F46',
                 lw=0.5, zorder=6)

    # Right helix (tilted other way)
    ribbon_helix(ax, cx+0.1*s, cy+0.30*s, 1.3*s, angle_deg=65,
                 n_turns=2, width=0.16*s, fc=TEAL, ec=DTEAL,
                 lw=0.5, zorder=6)

    # YIGSR oval on top
    ax.add_patch(Ellipse((cx, cy+1.6*s), 1.0*s, 0.42*s,
                         fc=PURPLE, ec='white', lw=0.8, alpha=0.95, zorder=8))
    ax.text(cx, cy+1.6*s, 'YIGSR',
            ha='center', va='center',
            fontsize=max(4.0, 6.5*s), color='white',
            fontweight='bold', zorder=9)

    # Lipid bilayer dots
    for side in [-1, 1]:
        for k in range(3):
            bx = cx + side*(0.65+k*0.25)*s
            ax.plot(bx, cy-1.12*s, 'o', ms=max(2, 4*s),
                    color='#D97706', mec='#92400E', mew=0.4*s, zorder=7)


def x_mark(ax, x, y, size=0.30, col=RED, lw=2.0, zorder=10):
    """Draw bold X inhibition mark."""
    ax.plot([x-size, x+size], [y-size, y+size],
            color=col, lw=lw, solid_capstyle='round', zorder=zorder)
    ax.plot([x-size, x+size], [y+size, y-size],
            color=col, lw=lw, solid_capstyle='round', zorder=zorder)


def erk_arrow(ax, x0, y0, length, label='ERK', col=TEAL,
              blocked=True, direction='right', fs=7.0):
    """Draw ERK activity arrow with optional X."""
    if direction == 'right':
        dx, dy = length, 0
    else:
        dx, dy = 0, -length

    ax.annotate('', xy=(x0+dx, y0+dy), xytext=(x0, y0),
                arrowprops=dict(arrowstyle='->', color=col,
                                lw=1.6, mutation_scale=12))
    ax.text(x0+dx/2, y0 + (0.20 if direction=='right' else 0),
            label, ha='center', fontsize=fs, color=col,
            fontstyle='italic', fontweight='bold')
    if blocked:
        mid_x = x0 + dx/2 + (0 if direction=='right' else 0.1)
        mid_y = y0 + dy/2 + (0 if direction=='right' else 0.1)
        x_mark(ax, mid_x, mid_y, size=0.18, lw=2.2)


# ═══════════════════════════════════════════════════════════════════════════
#  PANEL B cell — one drug combination
# ═══════════════════════════════════════════════════════════════════════════
def draw_drug_cell(ax, title, subtitle,
                   erk1_label, erk2_label,
                   pi3k_blocked, erk1_blocked, erk2_blocked,
                   invasion_pct, apop_fold, apop_pct,
                   dG1, dG2,
                   show_irinotecan_icon=False):
    """
    Draw one drug-combination cell inside ax.
    Layout matches reference image proportions.
    """
    ax.set_xlim(0, 6); ax.set_ylim(0, 9.5)
    ax.axis('off')
    ax.set_facecolor(WHITE)

    # ── Title ──────────────────────────────────────────────────────────
    ax.text(3.0, 9.1, title, ha='center', fontsize=8.5,
            fontweight='bold', color=NAVY)
    if subtitle:
        ax.text(3.0, 8.65, subtitle, ha='center', fontsize=7.0,
                color=TEAL, fontweight='bold')

    # ── 67LR protein icon (center-right) ──────────────────────────────
    draw_67LR(ax, 3.8, 5.8, scale=0.85)

    # ── Lipid bilayer horizontal line ──────────────────────────────────
    ax.plot([0.3, 5.7], [4.35, 4.35], color='#92400E', lw=5.5,
            solid_capstyle='round', alpha=0.55, zorder=3)
    ax.plot([0.3, 5.7], [4.10, 4.10], color='#D97706', lw=3.5,
            solid_capstyle='round', alpha=0.50, zorder=3)
    # Lipid heads along membrane
    for xdot in np.linspace(0.4, 5.6, 22):
        ax.plot(xdot, 4.47, 'o', ms=4, color='#F59E0B',
                mec='#92400E', mew=0.4, zorder=4)
        ax.plot(xdot, 3.97, 'o', ms=4, color='#F59E0B',
                mec='#92400E', mew=0.4, zorder=4)

    # ── PI3K-Akt label (left of 67LR) ─────────────────────────────────
    ax.text(1.2, 6.8, 'PI3K-\nAkt', ha='center', fontsize=6.5,
            color=NAVY, fontweight='bold')
    ax.text(1.2, 6.0, 'activity', ha='center', fontsize=5.8, color=GRAY)
    if pi3k_blocked:
        x_mark(ax, 1.2, 5.5, size=0.25, lw=2.2)

    # erk1 row  (e.g. "Invasion")
    erk_arrow(ax, 1.0, 4.85, 2.0, label=erk1_label,
              col=TEAL, blocked=erk1_blocked, direction='right', fs=6.5)

    # ERK label
    ax.text(3.2, 4.85, 'ERK', ha='center', fontsize=6.5,
            color=TEAL, fontweight='bold')

    # erk2 row (e.g. second ERK below)
    erk_arrow(ax, 1.0, 3.70, 2.0, label=erk2_label,
              col='#64748B', blocked=erk2_blocked, direction='right', fs=6.5)

    if show_irinotecan_icon:
        # Small bottle icon (simplified)
        ax.add_patch(FancyBboxPatch((4.6, 5.4), 0.40, 0.70,
            boxstyle='round,pad=0.04', fc=TEAL, ec=DTEAL, lw=0.7))
        ax.add_patch(FancyBboxPatch((4.72, 6.08), 0.16, 0.20,
            boxstyle='round,pad=0.02', fc=DTEAL, ec='none'))
        ax.text(4.80, 5.1, 'iri', ha='center', fontsize=4.5, color=DTEAL)

    # ── Stats block (bottom) ───────────────────────────────────────────
    stats_y = 3.15
    ax.plot([0.2, 5.8], [stats_y, stats_y], color='#E5E7EB', lw=0.8)

    if invasion_pct is not None:
        ax.text(1.5, stats_y-0.45, f'Invasion  {invasion_pct}%',
                ha='center', fontsize=7.0, color=NAVY)
    ax.text(3.5, stats_y-0.45,
            f'Apoptosis  {apop_fold}',
            ha='center', fontsize=7.0, color=NAVY)
    ax.text(3.5, stats_y-0.95,
            f'efficiency  {apop_pct}%',
            ha='center', fontsize=7.0, color=NAVY, fontweight='bold')
    ax.text(3.5, stats_y-1.45,
            f'G_end  {dG1}  kcal/mol',
            ha='center', fontsize=6.5, color=GRAY)
    ax.text(3.5, stats_y-1.90,
            f'{dG2}  kcal/mol',
            ha='center', fontsize=6.5, color=GRAY)

    # Outer box
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color('#CBD5E1')
        spine.set_linewidth(0.8)


# ═══════════════════════════════════════════════════════════════════════════
#  MAIN ASSEMBLY
# ═══════════════════════════════════════════════════════════════════════════
def build_figure1():
    fig = plt.figure(figsize=(22/2.54, 17/2.54), facecolor=WHITE)

    # 1 row × 2 col: A (40%), B (60%)
    gs_top = gridspec.GridSpec(1, 2, figure=fig,
                               left=0.01, right=0.99,
                               top=0.96, bottom=0.03,
                               wspace=0.05,
                               width_ratios=[4, 6])

    # Panel A
    axA = fig.add_subplot(gs_top[0])
    draw_panel_A(axA)

    # Panel B: 2×2 grid inside right half
    gs_B = gridspec.GridSpecFromSubplotSpec(
        2, 2, subplot_spec=gs_top[1],
        hspace=0.10, wspace=0.08)

    # 4 drug panels
    drug_cells = [
        # (row, col, title, subtitle,
        #  erk1_label, erk2_label,
        #  pi3k_blocked, erk1_blocked, erk2_blocked,
        #  invasion_pct, apop_fold_str, apop_pct,
        #  dG1_str, dG2_str, show_iri)
        (0, 0,
         'pritamab', None,
         'ERK', 'activity',
         True, True, True,
         None, 'Inhibition\nefficiency', 65,
         '\u221210 kcal/mol', '\u221213.0 kcal/mol', False),

        (0, 1,
         'pritamab +', 'irinotecan',
         'Invasion', 'ERK',
         True, True, True,
         None, '3.0-fold', 75,
         '\u221213.5 kcal/mol', '\u221213.5 kcal/mol', True),

        (1, 0,
         'pritamab +', 'oxaliplatin',
         'ERK\nactivity', 'Invasion',
         True, True, True,
         20, '3.0 fold', 75,
         '\u221214.0 kalmol/mol', '\u221213.0 kcal/mol', False),

        (1, 1,
         'pritamab +', 'TAS-102',
         'ERK-Akt\nactivity', 'Invasion',
         True, True, True,
         20, '3.3-fold', 80,
         '\u221214.3 kcal/mol', '\u221214.3 kcal/mol', False),
    ]

    for row, col, ttl, sub, e1, e2, pi3k, b1, b2, inv, afold, apct, dg1, dg2, iri in drug_cells:
        ax_d = fig.add_subplot(gs_B[row, col])
        draw_drug_cell(ax_d, ttl, sub, e1, e2, pi3k, b1, b2,
                       inv, afold, apct, dg1, dg2, iri)

    # Panel B label
    fig.text(0.41, 0.96, 'B',
             fontsize=12, fontweight='bold', color='black')

    # Figure label bottom-right
    fig.text(0.97, 0.01, 'Figure 1',
             ha='right', fontsize=8.5, color=GRAY)

    # Save
    for ext in ['png', 'pdf']:
        fpath = os.path.join(OUT, f'figure1_final.{ext}')
        fig.savefig(fpath, dpi=300, bbox_inches='tight',
                    facecolor=WHITE, edgecolor='none')
        print(f'Saved: {fpath}')
    plt.close(fig)
    print('Done.')


if __name__ == '__main__':
    print('Building Figure 1 (reference style)...')
    build_figure1()
