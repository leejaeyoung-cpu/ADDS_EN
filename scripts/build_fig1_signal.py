"""
Fig.1  Signal Pathway — Pritamab·PrPᶜ / 67LR Mechanism
=======================================================
Style: white background, biomedical illustration
  - Cell cross-section (warm gradient circle)
  - PrPC teal ribbon threaded through membrane
  - 67LR / YIGSR callout (top-right oval)
  - CDR3 binding callout (top-left oval)
  - c-MET / EGFR downstream
  - ERK suppression cascade with X marks
  - Bottom: cell proliferation / migration inhibited
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False
import matplotlib.patches as mpatches
from matplotlib.patches import (FancyArrowPatch, FancyBboxPatch,
                                 Circle, Ellipse, Arc)
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.patheffects as pe
import os

OUT = r'f:\ADDS\pritamab\figures'
os.makedirs(OUT, exist_ok=True)

# ── Palette ─────────────────────────────────────────────────────────────────
TEAL    = '#0D9488'
LTEAL   = '#5EEAD4'
ORANGE  = '#EA580C'
LORANGE = '#FED7AA'
PURPLE  = '#7C3AED'
LPURP   = '#DDD6FE'
NAVY    = '#1E3A5F'
RED     = '#DC2626'
GREEN   = '#16A34A'
LGREEN  = '#DCFCE7'
GOLD    = '#D97706'
BLUE    = '#2563EB'
GRAY    = '#6B7280'
LGRAY   = '#F3F4F6'
BG      = 'white'

fig, ax = plt.subplots(figsize=(16/2.54, 20/2.54), facecolor=BG)
ax.set_xlim(0, 16); ax.set_ylim(0, 20)
ax.set_aspect('equal')
ax.axis('off')
ax.set_facecolor(BG)

# ══════════════════════════════════════════════════════════════════════════════
# 1.  CELL – warm gradient circle (centre of figure)
# ══════════════════════════════════════════════════════════════════════════════
cx, cy, cr = 8.0, 11.5, 4.2   # cell centre, radius

# Gradient fill via many concentric circles
cmap_cell = LinearSegmentedColormap.from_list('cell',
    ['#FFF7ED', '#FDBA74', '#EA580C'], N=60)
for i in range(60, 0, -1):
    r_i = cr * i / 60
    alpha_i = 0.04 + 0.01*(60-i)/60
    col_i = cmap_cell(1 - i/60)
    circ_i = Circle((cx, cy), r_i, fc=col_i, ec='none', alpha=0.85, zorder=1)
    ax.add_patch(circ_i)

# Cell border
cell_outline = Circle((cx, cy), cr, fc='none', ec=ORANGE, lw=2.0, zorder=5)
ax.add_patch(cell_outline)

# Cell membrane dots (lipid bilayer look)
n_dots = 48
for k in range(n_dots):
    ang = 2*np.pi*k/n_dots
    # outer leaflet
    mx = cx + (cr + 0.25)*np.cos(ang)
    my = cy + (cr + 0.25)*np.sin(ang)
    ax.plot(mx, my, 'o', ms=4.5, color='#D97706',
            mec='#92400E', mew=0.5, zorder=6)
    # inner leaflet
    mx2 = cx + (cr - 0.60)*np.cos(ang)
    my2 = cy + (cr - 0.60)*np.sin(ang)
    ax.plot(mx2, my2, 'o', ms=3.5, color='#FB923C',
            mec='#C2410C', mew=0.4, zorder=6)

# ══════════════════════════════════════════════════════════════════════════════
# 2.  PrPC RIBBON inside/spanning cell
# ══════════════════════════════════════════════════════════════════════════════
def draw_helix(ax, x0, y0, dx, dy, n_coils=3.5, amp=0.28,
               color=TEAL, lw=2.5, alpha=1.0, zorder=8):
    """Draw a 2D ribbon helix along direction (dx,dy)."""
    length = np.sqrt(dx**2 + dy**2)
    ux, uy = dx/length, dy/length  # unit vector along helix axis
    px, py = -uy, ux               # perpendicular unit vector
    t = np.linspace(0, 1, 200)
    xs = x0 + t*dx + amp*np.sin(n_coils*2*np.pi*t)*px
    ys = y0 + t*dy + amp*np.sin(n_coils*2*np.pi*t)*py
    ax.plot(xs, ys, color=color, lw=lw, alpha=alpha,
            solid_capstyle='round', zorder=zorder)
    # depth shade: second line offset for 3D feel
    xs2 = x0 + t*dx + amp*np.cos(n_coils*2*np.pi*t)*px*0.3
    ys2 = y0 + t*dy + amp*np.cos(n_coils*2*np.pi*t)*py*0.3
    ax.plot(xs2, ys2, color=LTEAL, lw=lw*0.55, alpha=alpha*0.5,
            solid_capstyle='round', zorder=zorder)

def draw_loop(ax, pts, color=TEAL, lw=2.0, zorder=8):
    xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
    from scipy.interpolate import splprep, splev
    import warnings
    warnings.filterwarnings('ignore')
    try:
        tck, u = splprep([xs, ys], s=0, k=min(3, len(pts)-1))
        t_new = np.linspace(0, 1, 300)
        xn, yn = splev(t_new, tck)
        ax.plot(xn, yn, color=color, lw=lw, zorder=zorder)
    except Exception:
        ax.plot(xs, ys, color=color, lw=lw, zorder=zorder)

# PrPC central stem (transmembrane-like vertical)
ax.plot([cx-0.3, cx-0.3], [cy-cr+0.8, cy+1.2],
        color=TEAL, lw=5.5, solid_capstyle='round', zorder=8)
ax.plot([cx+0.3, cx+0.3], [cy-cr+0.8, cy+1.2],
        color=LTEAL, lw=3.0, solid_capstyle='round', zorder=8)

# H2 helix — upper left arm
draw_helix(ax, cx-0.2, cy+1.0, -2.2, 1.8, n_coils=3, amp=0.30,
           color=TEAL, lw=2.8, zorder=9)
# H3 helix — upper right arm
draw_helix(ax, cx+0.2, cy+1.0, 2.0, 1.8, n_coils=3, amp=0.30,
           color='#0F766E', lw=2.8, zorder=9)
# lower coil
draw_helix(ax, cx-0.5, cy-cr+0.8, 0.0, -1.0, n_coils=1.5, amp=0.25,
           color=TEAL, lw=2.2, zorder=7)

# YIGSR peptide label on right arm
ax.add_patch(Ellipse((cx+2.8, cy+3.0), 1.40, 0.65,
             fc=PURPLE, ec='white', lw=1.5, zorder=10))
ax.text(cx+2.8, cy+3.0, 'YIGSR', ha='center', va='center',
        fontsize=7, color='white', fontweight='bold', zorder=11)

# PrPC label
ax.text(cx+0.1, cy+2.55, 'PrPᶜ',
        ha='center', va='center', fontsize=10,
        color=TEAL, fontweight='bold', zorder=12,
        path_effects=[pe.withStroke(linewidth=2.5, foreground='white')])

# ══════════════════════════════════════════════════════════════════════════════
# 3.  TOP-LEFT CALLOUT: CDR3 binding (VH/VL)
# ══════════════════════════════════════════════════════════════════════════════
cl_x, cl_y, cl_r = 3.5, 17.5, 2.3   # circle center & radius
callout_L = Circle((cl_x, cl_y), cl_r,
                   fc='#F8FAFC', ec=NAVY, lw=1.8, zorder=15)
ax.add_patch(callout_L)

# Mini protein inside callout (grey ribbon)
draw_helix(ax, cl_x-1.6, cl_y-0.4, 3.2, 0.0, n_coils=2, amp=0.22,
           color='#94A3B8', lw=2.0, zorder=16)
# Dashed yellow binding lines
for yi in [-0.15, 0.10, 0.35]:
    ax.plot([cl_x-0.8, cl_x+0.2], [cl_y+yi, cl_y+yi],
            color='#EAB308', lw=1.2, ls='--', zorder=17)
# Labels
ax.text(cl_x, cl_y+1.65, 'VH CDR3: WNKPSK',
        ha='center', fontsize=6.2, color=NAVY, fontweight='bold', zorder=18)
ax.text(cl_x, cl_y+1.25, 'VL CDR: QQYYST',
        ha='center', fontsize=5.8, color=GRAY, zorder=18)
# residue labels inside
for txt, (tx, ty) in [('Tyr32',(cl_x-1.5, cl_y+0.7)),
                       ('Tyr150',(cl_x-1.3, cl_y-0.6)),
                       ('Arp95',(cl_x-0.5, cl_y-0.8)),
                       ('Gln33',(cl_x+0.8, cl_y+0.6)),
                       ('Ser33',(cl_x+0.9, cl_y-0.4)),
                       ('Fop93',(cl_x+0.5, cl_y-0.9))]:
    ax.text(tx, ty, txt, fontsize=4.8, color='#475569', ha='center', zorder=18)

# Arrow from callout to PrPC H2 tip
ax.annotate('', xy=(cx-2.0, cy+2.6), xytext=(cl_x+0.6, cl_y-1.8),
            arrowprops=dict(arrowstyle='->', color=NAVY, lw=1.4,
                            connectionstyle='arc3,rad=-0.2'), zorder=14)

# ══════════════════════════════════════════════════════════════════════════════
# 4.  TOP-RIGHT CALLOUT: 67LR / Laminin binding
# ══════════════════════════════════════════════════════════════════════════════
cr_x, cr_y, cr_r = 12.5, 17.5, 2.3
callout_R = Circle((cr_x, cr_y), cr_r,
                   fc='#F0FDF4', ec=TEAL, lw=1.8, zorder=15)
ax.add_patch(callout_R)
draw_helix(ax, cr_x-1.5, cr_y, 3.0, 0.0, n_coils=2, amp=0.25,
           color=TEAL, lw=2.2, zorder=16)
draw_helix(ax, cr_x-1.0, cr_y-0.6, 2.0, 0.8, n_coils=1.5, amp=0.18,
           color='#0F766E', lw=1.8, zorder=16)

ax.add_patch(Ellipse((cr_x+0.2, cr_y-1.1), 1.3, 0.60,
             fc=PURPLE, ec='white', lw=1.2, zorder=17))
ax.text(cr_x+0.2, cr_y-1.1, 'YIGSR', ha='center', va='center',
        fontsize=6.5, color='white', fontweight='bold', zorder=18)
ax.text(cr_x, cr_y+1.65, '엔디리: Lys(−)−E206',
        ha='center', fontsize=5.8, color=NAVY, fontweight='bold', zorder=18)
ax.text(cr_x, cr_y+1.25, '수노람: N/S−E213',
        ha='center', fontsize=5.5, color=GRAY, zorder=18)
ax.text(cr_x-0.5, cr_y, 'Laminin\nbinding\n(205–229)',
        ha='center', va='center', fontsize=5.2, color=TEAL, zorder=18)

# X mark on 67LR callout (blocked)
ax.text(cr_x+1.3, cr_y+1.3, '✕', fontsize=20, color=RED,
        fontweight='bold', ha='center', va='center', zorder=20)

# Arrow from callout to PrPC right arm
ax.annotate('', xy=(cx+2.4, cy+2.8), xytext=(cr_x-0.8, cr_y-1.8),
            arrowprops=dict(arrowstyle='->', color=TEAL, lw=1.4,
                            connectionstyle='arc3,rad=0.2'), zorder=14)

# ══════════════════════════════════════════════════════════════════════════════
# 5.  c-MET  (left side, outside cell)
# ══════════════════════════════════════════════════════════════════════════════
# c-MET protein shape
met_x, met_y = 2.0, 11.5
draw_helix(ax, met_x-0.3, met_y-1.5, 0.6, 3.0, n_coils=3, amp=0.25,
           color='#64748B', lw=2.5, zorder=8)
ax.add_patch(Circle((met_x, met_y+1.8), 0.55, fc='#94A3B8', ec='#475569',
             lw=1.2, zorder=9))
ax.text(met_x, met_y-2.0, 'c-MET', ha='center', fontsize=7,
        color=NAVY, fontweight='bold')
# X mark (blocked by Pritamab)
ax.text(met_x+0.1, met_y+0.8, '✕', fontsize=22, color=RED,
        fontweight='bold', ha='center', va='center', zorder=12)
# connector to cell
ax.annotate('', xy=(cx-cr+0.1, cy+0.3),
            xytext=(met_x+0.7, met_y+0.6),
            arrowprops=dict(arrowstyle='->', color='#475569', lw=1.2,
                            connectionstyle='arc3,rad=0.1'), zorder=7)

# ══════════════════════════════════════════════════════════════════════════════
# 6.  67LR label on right side of cell
# ══════════════════════════════════════════════════════════════════════════════
lr_x, lr_y = 14.2, 12.2
# 67LR protein stick
draw_helix(ax, lr_x-0.3, lr_y-2.0, 0.6, 4.0, n_coils=3, amp=0.28,
           color=TEAL, lw=2.5, zorder=8)
ax.add_patch(Ellipse((lr_x+0.2, lr_y+0.6), 1.3, 0.60,
             fc=PURPLE, ec='white', lw=1.2, zorder=9))
ax.text(lr_x+0.2, lr_y+0.6, 'YIGSR', ha='center', va='center',
        fontsize=6, color='white', fontweight='bold', zorder=10)
ax.text(lr_x, lr_y-2.5, '67 LR', ha='center', fontsize=7.5,
        color=TEAL, fontweight='bold')
# X mark
ax.text(lr_x-0.8, lr_y-0.5, '✕', fontsize=22, color=RED,
        fontweight='bold', ha='center', va='center', zorder=12)
ax.annotate('Laminin binding site (205–229)',
            xy=(cx+cr-0.2, cy+0.5), xytext=(lr_x-1.8, lr_y+2.2),
            fontsize=5.5, color=GRAY,
            arrowprops=dict(arrowstyle='->', color=GRAY, lw=0.9), zorder=12)
# X mark on direct binding to 67LR
ax.text(cx+cr+0.3, cy-0.5, '✕', fontsize=18, color=RED,
        fontweight='bold', zorder=12)
ax.text(cx+cr+0.5, cy-1.2, 'Direct\nBinding',
        fontsize=6, color=GRAY, ha='left')

# ══════════════════════════════════════════════════════════════════════════════
# 7.  EGFR / c-MET icons (lower left)
# ══════════════════════════════════════════════════════════════════════════════
# Diamond (c-MET inhibitor icon)
from matplotlib.patches import RegularPolygon
diamond = plt.Polygon([[3.0, 7.2],[3.6,7.7],[3.0,8.2],[2.4,7.7]],
                      fc='#3B82F6', ec=NAVY, lw=1.2, zorder=8)
ax.add_patch(diamond)
ax.text(3.0, 7.7, '', ha='center', va='center', fontsize=5, color='white')
# Gold sphere (kinase)
ax.add_patch(Circle((4.5, 7.7), 0.50, fc='#EAB308', ec='#92400E',
             lw=1.2, zorder=8))
ax.text(3.0, 6.8, 'c-MET', ha='center', fontsize=6.5,
        color=BLUE, fontweight='bold')
ax.text(4.5, 6.8, 'EGFR', ha='center', fontsize=6.5,
        color=GOLD, fontweight='bold')

# Dashed arrow to cell bottom
ax.annotate('', xy=(cx-cr+0.5, cy-cr+0.5),
            xytext=(4.2, 8.2),
            arrowprops=dict(arrowstyle='->', color=NAVY, lw=1.1,
                            linestyle='dashed',
                            connectionstyle='arc3,rad=-0.3'), zorder=7)

# ══════════════════════════════════════════════════════════════════════════════
# 8.  PrPC / 67LR activation label
# ══════════════════════════════════════════════════════════════════════════════
ax.text(cx, cy-cr-0.65, 'PrPᶜ / 67 LR activation',
        ha='center', fontsize=8, color=TEAL, fontweight='bold')
ax.annotate('', xy=(cx-1.5, cy-cr-1.1), xytext=(cx+1.5, cy-cr-1.1),
            arrowprops=dict(arrowstyle='<->', color=TEAL, lw=1.5))

# ══════════════════════════════════════════════════════════════════════════════
# 9.  ERK SUPPRESSION CASCADE (bottom)
# ══════════════════════════════════════════════════════════════════════════════
def erk_row(ax, y, label_left, arrow_col=GREEN, blocked=False, x0=3.0, x1=13.0):
    """Draw one ERK activity arrow row."""
    # Left label
    ax.text(x0-0.1, y, label_left, ha='right', va='center',
            fontsize=7, color=NAVY, fontweight='bold')
    # Arrow
    ax.annotate('', xy=(x1-0.5, y), xytext=(x0+0.1, y),
                arrowprops=dict(arrowstyle='->', color=arrow_col, lw=2.0))
    ax.text((x0+x1)/2, y+0.25, 'ERK activity',
            ha='center', fontsize=6.5, color=arrow_col, fontstyle='italic')
    if blocked:
        # X cross
        ax.text((x0+x1)/2 + 0.5, y, '✕', fontsize=26, color=RED,
                fontweight='bold', ha='center', va='center', zorder=10)

# Divider line
ax.plot([1.0, 15.0], [7.3, 7.3], color='#CBD5E1', lw=1.0, ls='--')

# Row 1: FOLFOX+Pritamab — ERK blocked
erk_row(ax, 5.8, 'FOLFOX\n+Pritamab', arrow_col=GREEN, blocked=True)
# sub-label
ax.text(2.85, 5.3, '하위 신호 억제\n(Suppression Signaling)',
        ha='right', fontsize=6.2, color=GRAY, style='italic')

# Row 2: FOLFOX — ERK attenuated
erk_row(ax, 4.5, 'FOLFOX', arrow_col='#86EFAC', blocked=True, x1=11.0)

# ══════════════════════════════════════════════════════════════════════════════
# 10.  OUTCOME BOXES (very bottom)
# ══════════════════════════════════════════════════════════════════════════════
def outcome_box(ax, x, y, title1, title2, sub, icon_color):
    """Draw cell outcome box with X circle."""
    rect = FancyBboxPatch((x-2.0, y-1.1), 4.0, 2.6,
                          boxstyle='round,pad=0.15',
                          fc=LGRAY, ec='#CBD5E1', lw=1.2, zorder=5)
    ax.add_patch(rect)
    # Big X circle
    circ_x = Circle((x, y+0.6), 0.75, fc='white', ec=RED, lw=2.0, zorder=6)
    ax.add_patch(circ_x)
    ax.text(x, y+0.6, '✕', fontsize=22, color=RED, fontweight='bold',
            ha='center', va='center', zorder=7)
    ax.text(x, y-0.3, title1, ha='center', fontsize=8,
            color=NAVY, fontweight='bold', zorder=7)
    ax.text(x, y-0.7, title2, ha='center', fontsize=6.5,
            color=NAVY, zorder=7)
    ax.text(x, y-1.05, sub, ha='center', fontsize=6,
            color=GRAY, fontstyle='italic', zorder=7)

outcome_box(ax, 5.5, 2.4,  '암세보 증식', '억제', '(암세보 다움)', RED)
outcome_box(ax, 10.5, 2.4, '암세보 미동', '억제', '(항암 효과)', RED)

# Arrows from ERK rows to outcome boxes
ax.annotate('', xy=(5.5, 3.5), xytext=(5.5, 4.2),
            arrowprops=dict(arrowstyle='->', color=NAVY, lw=1.3))
ax.annotate('', xy=(10.5, 3.5), xytext=(10.5, 4.2),
            arrowprops=dict(arrowstyle='->', color=NAVY, lw=1.3))

# ══════════════════════════════════════════════════════════════════════════════
# 11.  Title & overall label
# ══════════════════════════════════════════════════════════════════════════════
ax.text(8.0, 19.7,
        'Figure 1.  Pritamab Signal Pathway: PrPᶜ / 67LR Inhibition Mechanism',
        ha='center', va='top', fontsize=9, fontweight='bold', color=NAVY)

# Save
for ext in ['png', 'pdf']:
    fpath = os.path.join(OUT, f'fig1_signal_v4.{ext}')
    fig.savefig(fpath, dpi=300, bbox_inches='tight',
                facecolor=BG, edgecolor='none')
    print(f'Saved: {fpath}')
plt.close(fig)
print('Done.')
