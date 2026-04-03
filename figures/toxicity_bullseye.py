"""
Anticancer Regimen Toxicity Bullseye Chart
Radial heatmap (polar): sector = regimen, ring = toxicity type
Color = Grade 3/4 incidence intensity
White background, Nature/oncology style
"""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
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
REGIMENS = [
    'Pritamab\nMono', 'Pritamab\n+FOLFOX', 'Pritamab\n+FOLFIRI',
    'FOLFOX', 'FOLFIRI', 'FOLFOXIRI',
    'CAPOX', 'TAS-102', 'Bev\n+FOLFOX', 'Pembro\nMono',
]
REG_SHORT = [
    'Prit.Mono','Prit+FOX','Prit+FIRI',
    'FOLFOX','FOLFIRI','FOLFOXIRI',
    'CAPOX','TAS-102','Bev+FOX','Pembro',
]

TOX_LABELS = [
    'Neutropenia',
    'Anemia',
    'Thrombocytopenia',
    'Nausea /\nVomiting',
    'Diarrhea',
    'Periph.\nNeuropathy',
    'Fatigue',
    'Hand-Foot\nSynd.',
    'Alopecia',
    'Hepatotox.',
    'Hypertension',
    'Immune AE',
]

# rows = regimens, cols = toxicity types  (G3/4 %)
TOX_MATRIX = np.array([
    [ 2,  3,  2,  2,  3,   1,  6,  0,   1,  2,  3,  4],   # Pritamab Mono
    [36,  8,  6,  6, 12,  14, 10,  2,  15,  4,  8,  3],   # Prit+FOLFOX
    [22, 10,  5,  8, 18,   4, 12,  1,  15,  3,  4,  3],   # Prit+FOLFIRI
    [41,  7,  5,  7, 11,  18,  8,  1,   5,  5, 10,  0],   # FOLFOX
    [24, 11,  4,  9, 20,   3, 10,  0,  30,  2,  2,  0],   # FOLFIRI
    [50, 18,  9, 19, 20,  12, 16,  1,  20,  6,  4,  0],   # FOLFOXIRI
    [21,  4, 15,  8, 12,  17,  8, 17,   1,  4, 12,  0],   # CAPOX
    [38, 19,  5,  5,  6,   1, 22,  0,   5,  2,  1,  0],   # TAS-102
    [38,  7,  4,  6, 10,  17, 10,  2,   6,  5, 18,  0],   # Bev+FOLFOX
    [ 2,  3,  1,  2,  4,   1, 18,  0,   1,  3,  0, 22],   # Pembrolizumab
], dtype=float)

N_REG = len(REGIMENS)     # 10 sectors
N_TOX = len(TOX_LABELS)   # 12 rings

# Color map: white -> pale yellow -> orange -> deep red
CMAP = LinearSegmentedColormap.from_list(
    'tox_bull',
    ['#FFFFFF', '#FFFDE7', '#FFE082', '#FF8C00', '#C62828'],
    N=512
)
NORM = Normalize(vmin=0, vmax=52)

# Colors to mark regimen groups
REG_GROUP_COLORS = [
    '#7B2FBE',  # Pritamab Mono
    '#6A1B9A',  # Prit+FOLFOX
    '#9C27B0',  # Prit+FOLFIRI
    '#1565C0',  # FOLFOX
    '#0097A7',  # FOLFIRI
    '#B71C1C',  # FOLFOXIRI
    '#E65100',  # CAPOX
    '#F57F17',  # TAS-102
    '#1976D2',  # Bev+FOLFOX
    '#2E7D32',  # Pembrolizumab
]

# ── FIGURE LAYOUT ─────────────────────────────────────────────────
fig = plt.figure(figsize=(22, 18), facecolor='white')
fig.patch.set_facecolor('white')

gs = gridspec.GridSpec(1, 2, figure=fig,
                        width_ratios=[3.5, 1.0],
                        left=0.02, right=0.97,
                        top=0.91, bottom=0.05,
                        wspace=0.02)

ax_bull  = fig.add_subplot(gs[0, 0], polar=True)
ax_legend= fig.add_subplot(gs[0, 1])
ax_legend.axis('off')
ax_legend.set_facecolor('white')

# ── BULLSEYE CONSTRUCTION ─────────────────────────────────────────
# Sector angle per regimen
sector_w = 2 * np.pi / N_REG           # each regimen gets equal sector
gap      = 0.018                         # small gap between sectors

# Ring radii: innermost ring = TOX[0], outermost = TOX[N_TOX-1]
r_inner_base = 0.30                      # inner dead-zone radius
ring_w       = (4.2 - r_inner_base) / N_TOX

for i_reg in range(N_REG):
    theta_start = i_reg * sector_w + gap
    theta_end   = (i_reg + 1) * sector_w - gap
    thetas      = np.linspace(theta_start, theta_end, 40)

    for i_tox in range(N_TOX):
        r_in  = r_inner_base + i_tox * ring_w
        r_out = r_in + ring_w - 0.01

        val   = TOX_MATRIX[i_reg, i_tox]
        color = CMAP(NORM(val))

        # Fill the wedge
        r_fill = np.concatenate([
            np.full(40, r_in),
            np.full(40, r_out)[::-1],
        ])
        t_fill = np.concatenate([thetas, thetas[::-1]])
        ax_bull.fill(t_fill, r_fill, color=color, zorder=2)

        # Border
        ax_bull.plot(thetas, np.full(40, r_in),  color='white', lw=0.5, zorder=3)
        ax_bull.plot(thetas, np.full(40, r_out), color='white', lw=0.5, zorder=3)

        # Value annotation (only if >= 10 to avoid clutter)
        if val >= 10:
            t_mid = (theta_start + theta_end) / 2
            r_mid = (r_in + r_out) / 2
            txt_color = 'white' if val >= 30 else '#1A1A1A'
            ax_bull.text(t_mid, r_mid, f'{val:.0f}',
                         ha='center', va='center', fontsize=6.2,
                         color=txt_color, fontweight='bold', zorder=5)

    # Sector border spokes
    for theta_edge in [theta_start, theta_end]:
        ax_bull.plot([theta_edge, theta_edge],
                     [r_inner_base, r_inner_base + N_TOX * ring_w],
                     color='white', lw=1.2, zorder=4)

    # Regimen label (outside outermost ring)
    r_label = r_inner_base + N_TOX * ring_w + 0.30
    t_mid   = (theta_start + theta_end) / 2

    # Colored arc badge behind label
    badge_r = r_inner_base + N_TOX * ring_w + 0.14
    badge_t = np.linspace(theta_start + gap*2, theta_end - gap*2, 15)
    ax_bull.fill_between(badge_t,
                          np.full(15, badge_r - 0.10),
                          np.full(15, badge_r + 0.10),
                          color=REG_GROUP_COLORS[i_reg], alpha=0.85, zorder=3)

    # Text label
    angle_deg = np.degrees(t_mid)
    if 90 < angle_deg <= 270:
        label_rot = angle_deg - 180
    else:
        label_rot = angle_deg
    ax_bull.text(t_mid, r_label + 0.12, REGIMENS[i_reg].replace('\n', '\n'),
                 ha='center', va='center',
                 fontsize=8.0, fontweight='bold',
                 color=REG_GROUP_COLORS[i_reg],
                 rotation=label_rot - 90,
                 rotation_mode='anchor', zorder=6)

# ── RING LABELS (toxicity names) ─────────────────────────────────
# Place at 0 angle (12-o-clock position = angle 0)
label_theta = -0.08  # slightly to the left of the 12-o-clock spoke
for i_tox in range(N_TOX):
    r_mid = r_inner_base + i_tox * ring_w + ring_w / 2
    # Draw ring label on the right side (theta = small angle)
    label_theta_r = 2 * np.pi * (0.5 / N_REG)  # halfway through first sector
    ax_bull.text(label_theta_r + 0.02, r_mid + 0.03,
                 TOX_LABELS[i_tox],
                 ha='left', va='center',
                 fontsize=6.8, color='#333355',
                 style='italic', zorder=7)

# Center circle (logo zone)
theta_full = np.linspace(0, 2 * np.pi, 200)
ax_bull.fill(theta_full, np.full(200, r_inner_base - 0.01),
             color='#F5F5FA', zorder=6)
ax_bull.fill(theta_full, np.full(200, r_inner_base * 0.6),
             color='white', zorder=7)

# Center text
ax_bull.text(0, 0, 'TOXICITY\nTARGET', ha='center', va='center',
             fontsize=9.5, fontweight='bold', color='#1A1A2E', zorder=8)

# Styling
ax_bull.set_facecolor('white')
ax_bull.set_theta_zero_location('N')
ax_bull.set_theta_direction(-1)
ax_bull.set_xticks([])
ax_bull.set_yticks([])
ax_bull.spines['polar'].set_visible(False)
ax_bull.set_ylim(0, r_inner_base + N_TOX * ring_w + 0.85)

# ── LEGEND PANEL (right side) ─────────────────────────────────────
# 1. Colorbar
cbar_ax = fig.add_axes([0.80, 0.55, 0.025, 0.32])
sm = ScalarMappable(cmap=CMAP, norm=NORM)
sm.set_array([])
cbar = plt.colorbar(sm, cax=cbar_ax)
cbar.set_label('Grade 3/4 Incidence (%)', fontsize=9.5, color='#1A1A2E')
cbar.set_ticks([0, 10, 20, 30, 40, 50])
cbar.ax.tick_params(labelsize=8.5, colors='#444466')

# 2. Ring legend (toxicity order)
leg_x_start = 0.795
leg_y_start = 0.52
ax_legend.set_xlim(0, 10)
ax_legend.set_ylim(0, 13)

# Ring legend header
fig.text(leg_x_start - 0.01, leg_y_start,
         'Ring Order\n(center -> outer)',
         ha='left', va='top', fontsize=9.5, fontweight='bold',
         color='#1A1A2E', transform=fig.transFigure)

tox_simple = [
    'Neutropenia','Anemia','Thrombocytopenia',
    'Nausea/Vomiting','Diarrhea','Periph. Neuropathy',
    'Fatigue','Hand-Foot Synd.','Alopecia',
    'Hepatotoxicity','Hypertension','Immune-related AE',
]
for k, tox_name in enumerate(tox_simple):
    r_frac = (k + 0.5) / N_TOX
    stripe_color = CMAP(r_frac * 0.7)
    fy = leg_y_start - 0.036 * (k + 2.2)
    fig.text(leg_x_start - 0.005, fy,
             f'Ring {k+1:02d}:',
             ha='left', va='center', fontsize=8.0,
             color='#666688', fontweight='bold',
             transform=fig.transFigure)
    fig.text(leg_x_start + 0.040, fy,
             tox_name,
             ha='left', va='center', fontsize=8.0,
             color='#1A1A2E',
             transform=fig.transFigure)

# 3. Regimen color swatches
fig.text(leg_x_start - 0.01, 0.22,
         'Regimen Groups',
         ha='left', va='top', fontsize=9.5, fontweight='bold',
         color='#1A1A2E', transform=fig.transFigure)

group_info = [
    ('Pritamab-based',       ['Prit.Mono','Prit+FOX','Prit+FIRI'],
     ['#7B2FBE','#6A1B9A','#9C27B0']),
    ('Standard Chemo',       ['FOLFOX','FOLFIRI','FOLFOXIRI','CAPOX','TAS-102'],
     ['#1565C0','#0097A7','#B71C1C','#E65100','#F57F17']),
    ('Targeted / Immuno',    ['Bev+FOLFOX','Pembrolizumab'],
     ['#1976D2','#2E7D32']),
]
gy = 0.205
for g_label, g_names, g_colors in group_info:
    gy -= 0.020
    fig.text(leg_x_start - 0.005, gy, g_label + ':',
             ha='left', va='center', fontsize=8.5, fontweight='bold',
             color='#333355', transform=fig.transFigure)
    for name, col in zip(g_names, g_colors):
        gy -= 0.022
        fig.add_artist(plt.Rectangle(
            (leg_x_start - 0.003, gy - 0.008), 0.020, 0.016,
            color=col, transform=fig.transFigure, clip_on=False, zorder=6))
        fig.text(leg_x_start + 0.020, gy, name,
                 ha='left', va='center', fontsize=8.0,
                 color='#1A1A2E', transform=fig.transFigure)

# ── SUPER TITLE & FOOTNOTES ───────────────────────────────────────
fig.suptitle(
    'Anticancer Regimen Toxicity Bullseye Chart\n'
    'Each sector = regimen  |  Each ring (center->outer) = toxicity category  |  Color = Grade 3/4 incidence (%)',
    fontsize=15, fontweight='bold', color='#0D1B4B', y=0.97, ha='center')

fig.text(0.42, 0.030,
         'Source: NCCN 2024 | ESMO CRC 2023 | MOSAIC 2004 | Falcone 2007 (JCO) | '
         'RECOURSE 2015 (NEJM) | KEYNOTE-177 2021 | NO16966 2008 | '
         'Lee et al. ADDS 2026 (Pritamab)\n'
         'Numbers shown only for Grade 3/4 >= 10%. '
         'Rings ordered from innermost (Neutropenia) to outermost (Immune-related AE).',
         ha='center', fontsize=7.8, color='#555577', style='italic',
         transform=fig.transFigure)

# ── SAVE ──────────────────────────────────────────────────────────
out_path = os.path.join(OUT, 'toxicity_bullseye.png')
plt.savefig(out_path, dpi=180, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.close()

sz = os.path.getsize(out_path) // 1024
print(f'Saved: {out_path}  ({sz} KB)')

# Quick sanity check
print('\nSanity check:')
print(f'  Sectors (regimens): {N_REG}')
print(f'  Rings (toxicities): {N_TOX}')
print(f'  Matrix cells total: {N_REG * N_TOX}')
print(f'  Max value: {TOX_MATRIX.max():.0f}% (FOLFOXIRI/Neutropenia)')
print(f'  Cells >= 10%% shown with numbers: {(TOX_MATRIX >= 10).sum()}')
