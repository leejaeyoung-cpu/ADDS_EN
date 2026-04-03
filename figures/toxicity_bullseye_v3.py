"""
Toxicity Bullseye Chart v3
Key changes vs v2:
  [1] Right legend DRASTICALLY reduced -- ring reference removed (already on chart);
      only color encoding + regimen groups remain
  [2] 0% vs 1-15% contrast INCREASED: 0% = solid medium grey (#C8C8C8),
      1-15% = warm cream (#FFFBF0), cmap starts at 16%+ for clear separation
  [3] Sector labels redesigned: external label block (no rotation within +-60 deg),
      larger font, high contrast against outer arc
  [4] Center redesigned: "Hematologic (center) -> Immune (outer)" key message
  [5] Annotation threshold lowered to >= 6% for more information density
  [6] Title simplified and self-explanatory
"""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.cm import ScalarMappable
from matplotlib.patches import FancyBboxPatch
import warnings
warnings.filterwarnings('ignore')

plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.size': 9,
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
})

OUT = r'f:\ADDS\figures'
os.makedirs(OUT, exist_ok=True)

# ── DATA ──────────────────────────────────────────────────────────
REG_SHORT = [
    'Pritamab', 'Prit+FOLFOX', 'Prit+FOLFIRI',
    'FOLFOX', 'FOLFIRI', 'FOLFOXIRI',
    'CAPOX', 'TAS-102', 'Bev+FOLFOX', 'Pembrolizumab',
]
TOX_SHORT = [
    'Neutropenia','Anemia','Thrombocytopenia',
    'Nausea/Vomiting','Diarrhea','Periph.Neuropathy',
    'Fatigue','Hand-Foot','Alopecia',
    'Hepatotoxicity','Hypertension','Immune AE',
]
# Ring grouping annotations (for center key)
TOX_GROUPS = ['Hematologic','Hematologic','Hematologic',
              'GI','GI','Neurologic',
              'Constitutional','Dermatologic','Dermatologic',
              'Hepatic','Vascular','Immune']

TOX_MATRIX = np.array([
    [ 2,  3,  2,  2,  3,   1,  6,  0,   1,  2,  3,  4],  # Pritamab Mono
    [36,  8,  6,  6, 12,  14, 10,  2,  15,  4,  8,  3],  # Prit+FOLFOX
    [22, 10,  5,  8, 18,   4, 12,  1,  15,  3,  4,  3],  # Prit+FOLFIRI
    [41,  7,  5,  7, 11,  18,  8,  1,   5,  5, 10,  0],  # FOLFOX
    [24, 11,  4,  9, 20,   3, 10,  0,  30,  2,  2,  0],  # FOLFIRI
    [50, 18,  9, 19, 20,  12, 16,  1,  20,  6,  4,  0],  # FOLFOXIRI
    [21,  4, 15,  8, 12,  17,  8, 17,   1,  4, 12,  0],  # CAPOX
    [38, 19,  5,  5,  6,   1, 22,  0,   5,  2,  1,  0],  # TAS-102
    [38,  7,  4,  6, 10,  17, 10,  2,   6,  5, 18,  0],  # Bev+FOLFOX
    [ 2,  3,  1,  2,  4,   1, 18,  0,   1,  3,  0, 22],  # Pembrolizumab
], dtype=float)

N_REG = 10
N_TOX = 12

# [2] Color scheme with CLEAR 3-tier separation:
ZERO_COLOR   = '#C0C0C0'   # 0% = solid medium grey (clearly not data)
LOW_COLOR    = '#FFF8E1'   # 1-15% = warm pale cream (clearly distinct from grey)
# 16%+ uses the main heatmap colormap
CMAP_MAIN    = LinearSegmentedColormap.from_list(
    'tox_v3', ['#FFE082','#FF8C00','#C62828'], N=512)
NORM_MAIN    = Normalize(vmin=16, vmax=52)

REG_COLORS = [
    '#7B2FBE','#6A1B9A','#9C27B0',
    '#1565C0','#0097A7','#B71C1C',
    '#E65100','#F57F17',
    '#1976D2','#2E7D32',
]

def cell_color(val):
    """[2] 3-tier color: grey(0) / cream(<16) / heatmap(>=16)"""
    if val == 0:
        return ZERO_COLOR
    elif val < 16:
        return LOW_COLOR
    else:
        return CMAP_MAIN(NORM_MAIN(val))

# ── FIGURE ────────────────────────────────────────────────────────
# [1] Wider main chart, minimal right legend
fig = plt.figure(figsize=(22, 17), facecolor='white')
fig.patch.set_facecolor('white')

# 3-col: bullseye | cbar | compact legend
from matplotlib.gridspec import GridSpec
gs = GridSpec(1, 3, figure=fig,
              width_ratios=[4.2, 0.06, 0.96],
              left=0.01, right=0.99,
              top=0.91, bottom=0.06, wspace=0.03)

ax_bull = fig.add_subplot(gs[0, 0], polar=True)
ax_cbar = fig.add_subplot(gs[0, 1])
ax_leg  = fig.add_subplot(gs[0, 2])
ax_leg.axis('off'); ax_leg.set_facecolor('white')

# ── BULLSEYE ──────────────────────────────────────────────────────
sector_w = 2 * np.pi / N_REG
gap      = 0.024
r_core   = 0.28
ring_w   = (4.2 - r_core) / N_TOX
r_total  = r_core + N_TOX * ring_w

for i_reg in range(N_REG):
    theta_s = i_reg * sector_w + gap
    theta_e = (i_reg + 1) * sector_w - gap
    thetas  = np.linspace(theta_s, theta_e, 50)

    for i_tox in range(N_TOX):
        r_in  = r_core + i_tox * ring_w
        r_out = r_in + ring_w - 0.006

        val   = TOX_MATRIX[i_reg, i_tox]
        color = cell_color(val)

        r_arr = np.concatenate([np.full(50, r_in), np.full(50, r_out)[::-1]])
        t_arr = np.concatenate([thetas, thetas[::-1]])
        ax_bull.fill(t_arr, r_arr, color=color, zorder=2)
        ax_bull.plot(thetas, np.full(50, r_in),  color='white', lw=0.55, zorder=3)
        ax_bull.plot(thetas, np.full(50, r_out), color='white', lw=0.55, zorder=3)

        # [5] Annotate >= 6%
        if val >= 6:
            t_mid = (theta_s + theta_e) / 2
            r_mid = (r_in + r_out) / 2
            txt_c = 'white' if val >= 28 else '#1A1A1A'
            fw    = 'bold' if val >= 15 else 'normal'
            ax_bull.text(t_mid, r_mid, f'{val:.0f}',
                         ha='center', va='center', fontsize=5.8,
                         color=txt_c, fontweight=fw, zorder=5)

    # Sector spokes
    for te in [theta_s, theta_e]:
        ax_bull.plot([te]*2, [r_core, r_total], color='white', lw=1.5, zorder=4)

    # Outer color arc badge
    t_mid   = (theta_s + theta_e) / 2
    badge_r = r_total + 0.18
    badge_ts= np.linspace(theta_s + gap, theta_e - gap, 20)
    ax_bull.fill_between(badge_ts,
                          np.full(20, badge_r - 0.12),
                          np.full(20, badge_r + 0.12),
                          color=REG_COLORS[i_reg], alpha=0.92, zorder=6)

    # [3] Sector labels: constrain rotation to [-60, 60] for readability
    r_label   = r_total + 0.50
    angle_deg = np.degrees(t_mid) % 360
    if 90 < angle_deg <= 270:
        rot = angle_deg - 180
    else:
        rot = angle_deg
    # Clamp rotation
    rot_clamped = max(-65, min(65, rot - 90))
    ax_bull.text(t_mid, r_label, REG_SHORT[i_reg],
                 ha='center', va='center',
                 fontsize=9.2, fontweight='bold',
                 color=REG_COLORS[i_reg],
                 rotation=rot_clamped,
                 rotation_mode='anchor', zorder=7)

# Ring labels (right side of each ring, inside chart)
ring_theta_lbl = 2 * np.pi * (0.5 / N_REG) + 0.03
for i_tox, lbl in enumerate(TOX_SHORT):
    r_mid = r_core + i_tox * ring_w + ring_w * 0.5
    col_lbl = '#2C3E50' if i_tox < 3 else \
              '#1A5276' if i_tox < 5 else \
              '#145A32' if i_tox < 6 else \
              '#6E2F8B' if i_tox >= 11 else '#444455'
    ax_bull.text(ring_theta_lbl, r_mid, lbl,
                 ha='left', va='center',
                 fontsize=7.4, color=col_lbl,
                 fontweight='bold', zorder=8)

# [4] Center redesign: useful message instead of "n=10"
theta_full = np.linspace(0, 2*np.pi, 200)
ax_bull.fill(theta_full, np.full(200, r_core),
             color='#F0F2F5', zorder=9)
ax_bull.fill(theta_full, np.full(200, r_core * 0.52),
             color='white', zorder=10)
center_lines = [
    (0,    0.22, 'RINGS', '#1A1A2E', 7.5, 'bold'),
    (0,    0.07, 'Center', '#2C3E50', 7.0, 'normal'),
    (0,   -0.06, 'Hematologic', '#2C3E50', 6.5, 'normal'),
    (0,   -0.18, 'to Immune', '#6E2F8B', 6.5, 'normal'),
    (0,   -0.30, '(outer)', '#6E2F8B', 6.0, 'normal'),
]
for (x, y, txt, col, fs, fw) in center_lines:
    ax_bull.text(x, y, txt, ha='center', va='center',
                 fontsize=fs, fontweight=fw, color=col, zorder=11)

# Bull styling
ax_bull.set_facecolor('white')
ax_bull.set_theta_zero_location('N')
ax_bull.set_theta_direction(-1)
ax_bull.set_xticks([]); ax_bull.set_yticks([])
ax_bull.spines['polar'].set_visible(False)
ax_bull.set_ylim(0, r_total + 1.1)

# ── COLORBAR (16%+) ───────────────────────────────────────────────
sm = ScalarMappable(cmap=CMAP_MAIN, norm=NORM_MAIN)
sm.set_array([])
cbar = plt.colorbar(sm, cax=ax_cbar, orientation='vertical')
cbar.set_label('G3/4 >= 16%  incidence (%)', fontsize=8.5, color='#1A1A2E')
cbar.set_ticks([16, 24, 32, 40, 48])
cbar.ax.tick_params(labelsize=8, colors='#444466')

# ── [1] MINIMAL RIGHT LEGEND ──────────────────────────────────────
lx = 0.808
ly = 0.880

# Color encoding (3 tiers only)
fig.text(lx, ly, 'Color Key', ha='left', fontsize=11, fontweight='bold',
         color='#1A1A2E', transform=fig.transFigure)
color_items = [
    (ZERO_COLOR,  '#1A1A2E', '0%  (confirmed zero)'),
    (LOW_COLOR,   '#1A1A2E', '1-15%  (mild/moderate, see colorbar)'),
    ('#FF8C00',   'white',   '16-35%  (significant)'),
    ('#C62828',   'white',   '36-52%  (high severity)'),
]
for m, (bg, tc, lbl) in enumerate(color_items):
    ry = ly - 0.040 * (m + 1)
    fig.add_artist(plt.Rectangle(
        (lx, ry - 0.012), 0.024, 0.020, color=bg,
        ec='#AAAAAA', lw=0.7, transform=fig.transFigure, clip_on=False))
    fig.text(lx + 0.028, ry, lbl, ha='left', va='center',
             fontsize=8.5, color='#1A1A2E', transform=fig.transFigure)

# Annotation note
ly2 = ly - 0.040 * 6
fig.text(lx, ly2, 'Annotations', ha='left', fontsize=11, fontweight='bold',
         color='#1A1A2E', transform=fig.transFigure)
fig.text(lx, ly2 - 0.035,
         'Numbers shown for G3/4\n'
         '>= 6%.  Cells < 6% are\n'
         'color-coded only.',
         ha='left', va='top', fontsize=8.5, color='#555577',
         transform=fig.transFigure, style='italic')

# Regimen groups (compact)
ly3 = ly2 - 0.120
fig.text(lx, ly3, 'Regimen Groups', ha='left', fontsize=11, fontweight='bold',
         color='#1A1A2E', transform=fig.transFigure)
groups = [
    ('#7B2FBE', 'Pritamab-based'),
    ('#1565C0', 'Standard Chemo'),
    ('#E65100', 'Oral-based (CAPOX/TAS)'),
    ('#1976D2', 'Anti-VEGF combo'),
    ('#2E7D32', 'Immunotherapy (Pembro)'),
]
for gk, (gc, gl) in enumerate(groups):
    gy = ly3 - 0.034 * (gk + 1)
    fig.add_artist(plt.Rectangle(
        (lx, gy - 0.010), 0.018, 0.016, color=gc,
        transform=fig.transFigure, clip_on=False))
    fig.text(lx + 0.022, gy, gl, ha='left', va='center',
             fontsize=8.5, color='#1A1A2E', transform=fig.transFigure)

# ── TITLE + FOOTER ────────────────────────────────────────────────
fig.suptitle(
    'Anticancer Regimen Toxicity Fingerprint  --  Bullseye Radial Heatmap\n'
    'Sector = Regimen (n=10)  |  Ring = AE Category (center: Hematologic \u2192 outer: Immune AE)  '
    '|  Color = G3/4 Incidence  |  Numbers: >= 6%',
    fontsize=14, fontweight='bold', color='#0D1B4B', y=0.97, ha='center')

fig.text(0.43, 0.022,
    'Grey (#C0C0C0) = confirmed 0% incidence (not missing);  Cream = 1-15%;  '
    'Orange-Red gradient = 16-52%.\n'
    'Outer arc color denotes regimen group category (separate from incidence heatmap).  '
    'Source: NCCN 2024 | ESMO CRC 2023 | Pivotal CRC trials | Lee ADDS 2026.',
    ha='center', fontsize=7.8, color='#555577', style='italic',
    transform=fig.transFigure)

out_path = os.path.join(OUT, 'toxicity_bullseye_v3.png')
plt.savefig(out_path, dpi=175, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.close()
print('Saved: %s (%d KB)' % (out_path, os.path.getsize(out_path)//1024))
print('Zero cells: %d  |  Low(1-15): %d  |  High(>=16): %d' % (
    (TOX_MATRIX==0).sum(), ((TOX_MATRIX>0)&(TOX_MATRIX<16)).sum(),
    (TOX_MATRIX>=16).sum()))
print('Annotated (>=6%%): %d / %d' % ((TOX_MATRIX>=6).sum(), TOX_MATRIX.size))
