"""
generate_fig3A_v3.py  ── v11 (LAYOUT-FIXED)
============================================
Key changes vs v10:
  · Input box titles now inside boxes (top, bold), features below
  · Boxes tighter with no title overflow → no overlap with each other
  · Feature Fusion label moved to y=0.30 (below network; no overlap)
  · Layer labels placed at y = Y_LOW - 0.68 (below all nodes)
  · Score panel formula lines aligned; entry arrows at 0.82/0.54/0.26
  · Canvas: 26 × 13 in, 200 dpi
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import os

FW, FH  = 26.0, 13.0
DPI     = 200
NODE_R  = 0.22

OUT_DIR  = r"f:\ADDS\outputs\pritamab_pptx_figures"
os.makedirs(OUT_DIR, exist_ok=True)
OUT_FILE = os.path.join(OUT_DIR, "fig3A_v3.png")

plt.rcParams.update({'font.family': 'DejaVu Sans'})
fig = plt.figure(figsize=(FW, FH), facecolor='white')

# ─── Axes: network left 74 %, score panel right 23 % ────────────────
ax = fig.add_axes([0.01, 0.12, 0.74, 0.82])
ax.set_aspect('equal')
ax.axis('off')
AX_W, AX_H = 20.5, 12.0          # data space
ax.set_xlim(0, AX_W)
ax.set_ylim(0, AX_H)

axS = fig.add_axes([0.77, 0.10, 0.225, 0.85])
axS.set_xlim(0, 1); axS.set_ylim(0, 1)
axS.axis('off')
axS.add_patch(FancyBboxPatch((0.03, 0.015), 0.94, 0.97,
              boxstyle='round,pad=0.025',
              facecolor='#EDE7F6', edgecolor='#6C3483',
              linewidth=2.8, zorder=0, clip_on=False))

# ─── Colours ─────────────────────────────────────────────────────────
C = dict(
    cf='#AED6F1', ce='#1A5276',       # clinical input
    mf='#A9DFBF', me='#196F3D',       # morphology input
    ff='#F9E79F', fe='#7D6608',       # fusion
    h1f='#48C9B0', h1e='#0E6655',
    h2f='#48C9B0', h2e='#0E6655',
    h3f='#F0B27A', h3e='#7E5109',
    wir='#BDC3C7', arr='#5D6D7E',
    ef='#F1948A',  ee='#922B21',
    sf_='#5DADE2', se='#1A5276',
    tf='#85C1E9',  te='#154360',
)

# ─── Helpers ──────────────────────────────────────────────────────────
def rbox(cx, cy, w, h, fc, ec, lw=1.7, r=0.25, z=2):
    ax.add_patch(FancyBboxPatch(
        (cx-w/2, cy-h/2), w, h,
        boxstyle=f'round,pad={r}',
        facecolor=fc, edgecolor=ec, linewidth=lw, zorder=z))

def circ(cx, cy, r=NODE_R, fc='#48C9B0', ec='#0E6655'):
    ax.add_patch(plt.Circle((cx, cy), r,
                             color=fc, ec=ec, lw=1.8, zorder=4))

def wire(x1, y1, x2, y2, a=0.22):
    ax.plot([x1,x2],[y1,y2], color=C['wir'], lw=0.4, alpha=a, zorder=1)

def barrow(x1, y1, x2, y2, col=C['arr'], lw=2.4):
    ax.annotate('', xy=(x2,y2), xytext=(x1,y1),
                arrowprops=dict(arrowstyle='->', color=col,
                                lw=lw, mutation_scale=18))

def lbl(cx, cy, txt, fs=10.5, col='#1A252F',
        bold=False, ha='center', va='center', clip=False):
    ax.text(cx, cy, txt, ha=ha, va=va, fontsize=fs, color=col,
            fontweight='bold' if bold else 'normal',
            linespacing=1.35, zorder=6, clip_on=clip)

def slbl(cx, cy, txt, fs=10.5, col='#4A235A',
         bold=False, ha='center', va='center'):
    axS.text(cx, cy, txt, ha=ha, va=va, fontsize=fs, color=col,
             fontweight='bold' if bold else 'normal',
             linespacing=1.4, zorder=5)

# ── Column positions ──────────────────────────────────────────────────
INP_X  = 2.10
FUS_X  = 5.90
H1_X   = 8.20
H2_X   = 10.90
H3_X   = 13.50
OUT_X  = 16.80

# Network y-range (nodes span 2.0 … 9.8)
Y_LOW, Y_HIGH = 2.0, 9.8

# ═══════════════════════════════════════════════════════════════════════
#  INPUT BOXES  (self-contained: title inside at top, features below)
# ═══════════════════════════════════════════════════════════════════════
# Clinical:  box centre y=7.15, height 4.4  →  spans 4.95 … 9.35
CLIN_CY, CLIN_H  = 7.15, 4.40
# Morphology: box centre y=2.72, height 3.20 → spans 1.12 … 4.32
MORPH_CY, MORPH_H = 2.72, 3.20

# Draw boxes
rbox(INP_X, CLIN_CY,  3.10, CLIN_H,  C['cf'], C['ce'], lw=2.2, r=0.3)
rbox(INP_X, MORPH_CY, 3.10, MORPH_H, C['mf'], C['me'], lw=2.2, r=0.3)

# Titles at box top (inside)
lbl(INP_X, CLIN_CY  + CLIN_H/2  - 0.35, 'Clinical Data Input',      fs=12.5, col=C['ce'], bold=True)
lbl(INP_X, CLIN_CY  + CLIN_H/2  - 0.80, '(6 features)',              fs=10,   col=C['ce'])
lbl(INP_X, MORPH_CY + MORPH_H/2 - 0.35, 'Cell Morphology Features',  fs=12.0, col=C['me'], bold=True)
lbl(INP_X, MORPH_CY + MORPH_H/2 - 0.78, '(5 features)',              fs=10,   col=C['me'])

# Clinical feature rows
clin_features = ['Genomic Profile', 'PrPC Serum Level', 'KRAS Status',
                 'Drug History', 'CT / Path Data', 'Clinical Features']
clin_ys = np.linspace(CLIN_CY + CLIN_H/2 - 1.20, CLIN_CY - CLIN_H/2 + 0.32, 6)
for feat, cy in zip(clin_features, clin_ys):
    circ(0.75, cy, r=0.19, fc=C['cf'], ec=C['ce'])
    lbl(1.12, cy, feat, fs=9.5, col=C['ce'], ha='left')

# Morphology feature rows
morph_features = ['Area (μm²)', 'Perimeter (P)', 'N/C Ratio',
                  'Circularity (Ci)', 'Atypia Index']
morph_ys = np.linspace(MORPH_CY + MORPH_H/2 - 1.08, MORPH_CY - MORPH_H/2 + 0.36, 5)
for feat, my in zip(morph_features, morph_ys):
    circ(0.75, my, r=0.19, fc=C['mf'], ec=C['me'])
    lbl(1.12, my, feat, fs=9.5, col=C['me'], ha='left')

# ═══════════════════════════════════════════════════════════════════════
#  FEATURE FUSION
# ═══════════════════════════════════════════════════════════════════════
fus_ys = np.linspace(Y_LOW, Y_HIGH, 6)
for fy in fus_ys:
    wire(INP_X+1.57, CLIN_CY,  FUS_X-NODE_R, fy)
    wire(INP_X+1.57, MORPH_CY, FUS_X-NODE_R, fy)
    circ(FUS_X, fy, fc=C['ff'], ec=C['fe'])

# Feature Fusion label — below nodes, centred on FUS_X (now well clear of morph box)
lbl(FUS_X, Y_LOW - 0.70, 'Feature\nFusion Layer',  fs=10.5, col=C['fe'], bold=True)
lbl(FUS_X, Y_LOW - 1.38, '32 units · ReLU',         fs=9.5,  col=C['fe'])

# ═══════════════════════════════════════════════════════════════════════
#  HIDDEN LAYERS
# ═══════════════════════════════════════════════════════════════════════
layer_cfg = [
    (H1_X, 7, 'Hidden Layer 1', '128 units', None,          C['h1f'], C['h1e']),
    (H2_X, 5, 'Hidden Layer 2', '64 units',  'Batch Norm',  C['h2f'], C['h2e']),
    (H3_X, 4, 'Hidden Layer 3', '64 units',  'Dropout 0.3', C['h3f'], C['h3e']),
]
layer_ys_map = {}
prev_x, prev_ys = FUS_X, fus_ys

for lx, n, ltitle, units, reg, lfc, lec in layer_cfg:
    ys = np.linspace(Y_LOW, Y_HIGH, n)
    layer_ys_map[lx] = ys
    for y1 in prev_ys:
        for y2 in ys:
            wire(prev_x+NODE_R, y1, lx-NODE_R, y2, a=0.18)
    barrow(prev_x+NODE_R+0.05, 5.9, lx-NODE_R-0.05, 5.9, lw=2.6)
    for y in ys:
        circ(lx, y, fc=lfc, ec=lec)
    lbl(lx, Y_LOW - 0.70, ltitle, fs=10.5, col=lec, bold=True)
    lbl(lx, Y_LOW - 1.38, units,  fs=9.5,  col=lec)
    # Reg tag at Y_HIGH + 0.90 (inside AX_H=12)
    if reg:
        tag_y = Y_HIGH + 0.90
        ax.add_patch(FancyBboxPatch(
            (lx-0.90, tag_y-0.36), 1.80, 0.72,
            boxstyle='round,pad=0.12',
            facecolor='#FDFEFE', edgecolor=lec, lw=1.6,
            zorder=5, clip_on=False))
        lbl(lx, tag_y, reg, fs=9.5, col=lec, bold=True, clip=False)
        ax.annotate('',
                    xy     =(lx, Y_HIGH+NODE_R+0.05),
                    xytext =(lx, tag_y-0.36),
                    arrowprops=dict(arrowstyle='->', color=lec,
                                    lw=1.5, mutation_scale=14),
                    annotation_clip=False)
    prev_x, prev_ys = lx, ys

# ═══════════════════════════════════════════════════════════════════════
#  OUTPUT HEADS
# ═══════════════════════════════════════════════════════════════════════
output_defs = [
    (8.60, 'Efficacy Prediction',  'E_pred ∈ [0, 1]',  C['ef'],  C['ee']),
    (5.60, 'Synergy Prediction',   'S_pred ∈ [0, 2]',  C['sf_'], C['se']),
    (2.60, 'Toxicity Prediction',  'T_tox  ∈ [1, 10]', C['tf'],  C['te']),
]
h3_ys = layer_ys_map[H3_X]
for oy, otitle, orange, ofc, oec in output_defs:
    rbox(OUT_X, oy, 3.9, 1.88, ofc, oec, lw=2.0, r=0.28, z=3)
    lbl(OUT_X, oy+0.44, otitle, fs=12.5, col=oec, bold=True)
    lbl(OUT_X, oy-0.47, orange, fs=10.5, col=oec)
    for hy in h3_ys:
        wire(H3_X+NODE_R, hy, OUT_X-1.97, oy, a=0.24)
    barrow(H3_X+NODE_R+0.04, oy, OUT_X-1.98, oy, col=oec, lw=2.2)
    ax.annotate('',
                xy    =(OUT_X+1.99, oy),
                xytext=(OUT_X+1.74, oy),
                arrowprops=dict(arrowstyle='->', color='#8E44AD',
                                lw=2.0, mutation_scale=16))

# ═══════════════════════════════════════════════════════════════════════
#  SCORE PANEL
# ═══════════════════════════════════════════════════════════════════════
slbl(0.53, 0.958, 'Overall Score\nCalculation', fs=13.5, bold=True)
axS.axhline(0.885, xmin=0.07, xmax=0.93, color='#8E44AD', lw=1.6)

# Formula + entry arrows
formula_rows = [
    (0.833, 'Score  ='),
    (0.775, '  w₁ · E_pred'),
    (0.720, '+ w₂ · S_pred'),
    (0.663, '− w₃ · (T_tox / 10)'),
]
entry_arrow_ys = (0.833, 0.720, 0.663)   # Score= / +w₂·S / −w₃·T lines
for i, (fy, ftxt) in enumerate(formula_rows):
    slbl(0.13, fy, ftxt, fs=11.5, ha='left')
# Entry arrows on left edge of score panel
for ey in entry_arrow_ys:
    axS.annotate('', xy=(0.11, ey), xytext=(0.03, ey),
                 arrowprops=dict(arrowstyle='->', color='#8E44AD',
                                 lw=1.8, mutation_scale=14))

axS.axhline(0.620, xmin=0.07, xmax=0.93, color='#8E44AD', lw=1.2)
slbl(0.53, 0.580, 'Final Ranking', fs=12.5, bold=True)
for ry, rl, rs, bold_s in [
    (0.524, '#1  Drug A + Drug B', '0.87 ✦', True),
    (0.468, '#2  Drug C + Drug D', '0.76',   False),
    (0.412, '#3  Drug E + Drug F', '0.72',   False),
]:
    slbl(0.09, ry, rl, fs=11,   ha='left')
    slbl(0.91, ry, rs, fs=11,   ha='right', bold=bold_s)

axS.axhline(0.368, xmin=0.07, xmax=0.93, color='#8E44AD', lw=1.2)
slbl(0.53, 0.328, 'Architecture', fs=12.5, bold=True)
for stxt, sy in [
    ('Parameters:  12,544',        0.280),
    ('Optimiser:  Adam  lr=0.001', 0.232),
    ('Loss:  Multi-task wMSE',     0.182),
    ('Framework:  PyTorch 2.x',    0.133),
    ('GPU:  NVIDIA RTX 5070',      0.083),
]:
    slbl(0.53, sy, stxt, fs=10.5)

# ─── Figure title & footnote ──────────────────────────────────────────
fig.text(0.445, 0.978,
         'Figure 3A  |  ADDS v5.3 Neural Network Architecture  ·  '
         'Multi-Task Deep Learning for Drug Recommendation — KRAS-Mutant mCRC / PAAD',
         ha='center', va='top', fontsize=15, fontweight='bold', color='#1A252F')

fig.text(0.015, 0.004,
         '† All Pritamab-related predictions are COMPUTATIONAL OUTPUTS of ADDS v5.3 '
         'Virtual Binding Engine — no clinical validation has been performed.',
         ha='left', va='bottom', fontsize=9,
         color='#7F8C8D', fontstyle='italic')

fig.savefig(OUT_FILE, dpi=DPI, bbox_inches='tight', facecolor='white')
plt.close()
print(f"Saved → {OUT_FILE}")
print(f"Resolution: {int(FW*DPI)} × {int(FH*DPI)} px at {DPI} dpi")
