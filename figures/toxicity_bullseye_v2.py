"""
Anticancer Regimen Toxicity Bullseye Chart v2
Revised per rigorous reviewer critique:

FIXES APPLIED:
  [1] 0 vs missing value 시각적 분리:
      - 0% = very light grey (#F0F0F0)
      - data exists = CMAP(val)
      (현재 데이터에 true missing 없음, 모두 0 또는 positive)
  [2] Right legend 정리 -- colorbar + ring list + regimen swatches 분리
      compact 배치로 오버랩 제거
  [3] Sector/ring label 가독성 향상:
      - 바깥 regimen label: 폰트 확대, bold, 회전 최소화
      - ring label: 안쪽 arc 대신 오른쪽 정형화된 박스에 배치
  [4] 숫자 annotation 규칙 명확화 -- 제목에 명시
  [5] 중앙 텍스트 실용적 개선 -- 범례 key 정보로 교체
  [6] 바깥 sector badge/arc 색상과 내부 heatmap 색상 구분 명확화 주석 추가

White background, Nature oncology figure style
"""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
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
    'CAPOX', 'TAS-102', 'Bev+\nFOLFOX', 'Pembro\nMono',
]
REG_SHORT = [
    'Prit. Mono', 'Prit+FOLFOX', 'Prit+FOLFIRI',
    'FOLFOX', 'FOLFIRI', 'FOLFOXIRI',
    'CAPOX', 'TAS-102', 'Bev+FOLFOX', 'Pembrolizumab',
]
TOX_LABELS = [
    'Ring 1: Neutropenia',
    'Ring 2: Anemia',
    'Ring 3: Thrombocytopenia',
    'Ring 4: Nausea/Vomiting',
    'Ring 5: Diarrhea',
    'Ring 6: Periph.Neuropathy',
    'Ring 7: Fatigue',
    'Ring 8: Hand-Foot Synd.',
    'Ring 9: Alopecia',
    'Ring 10: Hepatotoxicity',
    'Ring 11: Hypertension',
    'Ring 12: Immune-related AE',
]
TOX_SHORT = [
    'Neutropenia', 'Anemia', 'Thrombocytopenia',
    'Nausea/Vomiting', 'Diarrhea', 'Periph.Neuropathy',
    'Fatigue', 'Hand-Foot Synd.', 'Alopecia',
    'Hepatotoxicity', 'Hypertension', 'Immune AE',
]

# FIX [1]: 0 is a value, not missing -- distinctly styled
# rows=regimens, cols=toxicity (G3/4 %)
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

N_REG = len(REGIMENS)
N_TOX = len(TOX_LABELS)

# Two-stage colormap:
# 0 = very light grey (NOT white, to distinguish from true missing)
# >0 = pale yellow through deep red
ZERO_COLOR  = '#EBEBEB'   # FIX [1]: 0% -- light grey (not white)
CMAP_HEAT   = LinearSegmentedColormap.from_list(
    'tox_v2',
    ['#FFF9E6', '#FFE082', '#FF8C00', '#C62828'],
    N=512)
NORM_HEAT   = Normalize(vmin=0.5, vmax=52)

REG_GROUP_COLORS = [
    '#7B2FBE', '#6A1B9A', '#9C27B0',  # Pritamab trio
    '#1565C0', '#0097A7', '#B71C1C',  # Standard chemo
    '#E65100', '#F57F17',             # CAPOX, TAS-102
    '#1976D2', '#2E7D32',             # Bev, Pembrolizumab
]

# ── FIGURE ────────────────────────────────────────────────────────
# FIX [2]: wider left for bullseye, tight right column for legend
fig = plt.figure(figsize=(24, 17), facecolor='white')
fig.patch.set_facecolor('white')

# 3-column layout: bullseye | colorbar | ring+regimen legend
gs = gridspec.GridSpec(1, 3, figure=fig,
                        width_ratios=[3.8, 0.08, 1.0],
                        left=0.01, right=0.99,
                        top=0.91, bottom=0.06, wspace=0.04)

ax_bull  = fig.add_subplot(gs[0, 0], polar=True)
ax_cbar  = fig.add_subplot(gs[0, 1])
ax_leg   = fig.add_subplot(gs[0, 2])
ax_leg.axis('off'); ax_leg.set_facecolor('white')

# ── BULLSEYE ──────────────────────────────────────────────────────
sector_w = 2 * np.pi / N_REG
gap      = 0.022
r_core   = 0.26
ring_w   = (4.1 - r_core) / N_TOX
r_total  = r_core + N_TOX * ring_w

for i_reg in range(N_REG):
    theta_s = i_reg * sector_w + gap
    theta_e = (i_reg + 1) * sector_w - gap
    thetas  = np.linspace(theta_s, theta_e, 50)

    for i_tox in range(N_TOX):
        r_in  = r_core + i_tox * ring_w
        r_out = r_in + ring_w - 0.008

        val = TOX_MATRIX[i_reg, i_tox]

        # FIX [1]: explicit 0 vs positive
        if val == 0:
            color = ZERO_COLOR
        else:
            color = CMAP_HEAT(NORM_HEAT(val))

        r_arr = np.concatenate([np.full(50, r_in), np.full(50, r_out)[::-1]])
        t_arr = np.concatenate([thetas, thetas[::-1]])
        ax_bull.fill(t_arr, r_arr, color=color, zorder=2)

        # White cell border
        for r_bord in [r_in, r_out]:
            ax_bull.plot(thetas, np.full(50, r_bord),
                         color='white', lw=0.6, zorder=3)

        # FIX [4]: annotate >=8% (slightly lower threshold for more numbers)
        if val >= 8:
            t_mid = (theta_s + theta_e) / 2
            r_mid = (r_in + r_out) / 2
            txt_c = 'white' if val >= 28 else '#1A1A1A'
            ax_bull.text(t_mid, r_mid, f'{val:.0f}',
                         ha='center', va='center', fontsize=5.8,
                         color=txt_c, fontweight='bold', zorder=5)

    # Sector dividers (white spokes)
    for theta_edge in [theta_s, theta_e]:
        ax_bull.plot([theta_edge]*2, [r_core, r_total],
                     color='white', lw=1.4, zorder=4)

    # FIX [6]: regimen color arc badge (clearly = regimen group color)
    badge_r   = r_total + 0.16
    badge_t   = np.linspace(theta_s + gap, theta_e - gap, 20)
    ax_bull.fill_between(badge_t,
                          np.full(20, badge_r - 0.11),
                          np.full(20, badge_r + 0.11),
                          color=REG_GROUP_COLORS[i_reg], alpha=0.90, zorder=6)

    # FIX [3]: regimen label -- larger font, min rotation for readability
    t_mid     = (theta_s + theta_e) / 2
    r_label   = r_total + 0.44
    angle_deg = np.degrees(t_mid) % 360
    # Normalize rotation so text always reads left-to-right within +-90 deg
    if 90 < angle_deg <= 270:
        rot = angle_deg - 180
    else:
        rot = angle_deg
    ax_bull.text(t_mid, r_label, REG_SHORT[i_reg],
                 ha='center', va='center',
                 fontsize=8.8, fontweight='bold',
                 color=REG_GROUP_COLORS[i_reg],
                 rotation=rot - 90,
                 rotation_mode='anchor', zorder=7)

# FIX [3]: Ring labels -- place inside at consistent angle (3-o-clock position)
ring_label_theta = 2 * np.pi * (0.5 / N_REG) + 0.05  # inside sector 1
for i_tox, lbl in enumerate(TOX_SHORT):
    r_mid = r_core + i_tox * ring_w + ring_w * 0.5
    ax_bull.text(ring_label_theta, r_mid, lbl,
                 ha='left', va='center', fontsize=7.2,
                 color='#333355', style='italic', zorder=8)

# Center: FIX [5] -- practical info instead of decorative text
theta_full = np.linspace(0, 2*np.pi, 200)
ax_bull.fill(theta_full, np.full(200, r_core),
             color='#F0F0F5', zorder=9)
ax_bull.fill(theta_full, np.full(200, r_core * 0.55),
             color='white', zorder=10)

# Center practical key
ax_bull.text(0,  0.07, 'n=10', ha='center', va='center',
             fontsize=9.5, fontweight='bold', color='#1A1A2E', zorder=11)
ax_bull.text(0, -0.07, 'regimens', ha='center', va='center',
             fontsize=7.5, color='#555577', zorder=11)
ax_bull.text(0, -0.19, '12 rings', ha='center', va='center',
             fontsize=7.5, color='#555577', zorder=11)

# Bullseye styling
ax_bull.set_facecolor('white')
ax_bull.set_theta_zero_location('N')
ax_bull.set_theta_direction(-1)
ax_bull.set_xticks([]); ax_bull.set_yticks([])
ax_bull.spines['polar'].set_visible(False)
ax_bull.set_ylim(0, r_total + 1.0)

# ── COLORBAR ──────────────────────────────────────────────────────
# FIX [2]: standalone colorbar axis, cleaner
sm = ScalarMappable(cmap=CMAP_HEAT, norm=NORM_HEAT)
sm.set_array([])
cbar = plt.colorbar(sm, cax=ax_cbar, orientation='vertical')
cbar.set_label('G3/4 Incidence (%)  [>0]', fontsize=9, color='#1A1A2E')
cbar.set_ticks([1, 10, 20, 30, 40, 50])
cbar.ax.tick_params(labelsize=8, colors='#444466')

# Add 0% swatch below colorbar
ax_cbar.add_patch(plt.Rectangle(
    (0, -0.06), 1, 0.04, color=ZERO_COLOR,
    transform=ax_cbar.transAxes, clip_on=False))
ax_cbar.text(1.05, -0.04, '0% (confirmed zero)',
             va='center', ha='left', fontsize=7.5, color='#888888',
             transform=ax_cbar.transAxes)

# ── FIX [2]: Compact right-side legend ────────────────────────────
# Composed with fig.text to avoid overlapping axes

# ---- Section 1: Ring legend (compact table) ----
lx = 0.785  # fig x (0-1)
ly = 0.895
fig.text(lx, ly, 'Ring Reference',
         ha='left', fontsize=10, fontweight='bold', color='#1A1A2E',
         transform=fig.transFigure)
figiy = ly - 0.028
col_step = 0.055
ring_label_short = [
    '1: Neutropenia',   '2: Anemia',         '3: Thrombocytopenia',
    '4: Nausea/Vomit.', '5: Diarrhea',        '6: Periph.Neuropathy',
    '7: Fatigue',        '8: Hand-Foot',       '9: Alopecia',
    '10: Hepatotox.',   '11: Hypertension',   '12: Immune AE',
]
for k, rl in enumerate(ring_label_short):
    col_x = lx if k % 2 == 0 else lx + 0.105
    row_y = figiy - (k // 2) * 0.028
    ring_num = k + 1
    ring_frac = (k + 0.5) / N_TOX
    # Small color swatch
    fig.add_artist(plt.Rectangle(
        (col_x - 0.002, row_y - 0.009), 0.014, 0.016,
        color=CMAP_HEAT(ring_frac * 0.75),
        transform=fig.transFigure, clip_on=False))
    fig.text(col_x + 0.016, row_y, rl,
             ha='left', va='center', fontsize=7.2, color='#1A1A2E',
             transform=fig.transFigure)

# ---- Section 2: Legend for color encoding ----
ly2 = figiy - 7 * 0.028 - 0.018
fig.text(lx, ly2, 'Color Encoding',
         ha='left', fontsize=10, fontweight='bold', color='#1A1A2E',
         transform=fig.transFigure)
color_legend_items = [
    (ZERO_COLOR,   '0% (confirmed zero -- light grey)'),
    (CMAP_HEAT(0.15), '1-15% (mild)'),
    (CMAP_HEAT(0.42), '16-30% (moderate)'),
    (CMAP_HEAT(0.75), '31-50% (severe)'),
]
for m, (col, lbl) in enumerate(color_legend_items):
    ry = ly2 - 0.028 * (m + 1)
    fig.add_artist(plt.Rectangle(
        (lx, ry - 0.010), 0.022, 0.018, color=col, ec='#AAAAAA', lw=0.5,
        transform=fig.transFigure, clip_on=False))
    fig.text(lx + 0.026, ry, lbl, ha='left', va='center',
             fontsize=7.8, color='#1A1A2E', transform=fig.transFigure)

# FIX [4]: annotation rule explicit
ly3 = ly2 - 0.028 * 5.5 - 0.010
fig.text(lx, ly3, 'Annotations',
         ha='left', fontsize=10, fontweight='bold', color='#1A1A2E',
         transform=fig.transFigure)
fig.text(lx, ly3 - 0.025,
         'Numbers shown for cells >= 8%.\n'
         'Cells < 8% are color-only (no annotation).\n'
         'All displayed values = G3/4 incidence %.',
         ha='left', va='top', fontsize=7.5, color='#555577',
         transform=fig.transFigure, style='italic')

# ---- Section 3: Regimen groups ----
ly4 = ly3 - 0.108
fig.text(lx, ly4, 'Regimen Groups',
         ha='left', fontsize=10, fontweight='bold', color='#1A1A2E',
         transform=fig.transFigure)

groups = [
    ('Pritamab-based (purple)', ['Prit. Mono','Prit+FOLFOX','Prit+FOLFIRI'],
     ['#7B2FBE','#6A1B9A','#9C27B0']),
    ('Standard Chemo (blue/red)', ['FOLFOX','FOLFIRI','FOLFOXIRI','CAPOX','TAS-102'],
     ['#1565C0','#0097A7','#B71C1C','#E65100','#F57F17']),
    ('Targeted/Immuno (teal/green)', ['Bev+FOLFOX','Pembrolizumab'],
     ['#1976D2','#2E7D32']),
]
gy = ly4
for g_lbl, g_names, g_cols in groups:
    gy -= 0.026
    fig.text(lx, gy, g_lbl, ha='left', va='center',
             fontsize=8.0, fontweight='bold', color='#333355',
             transform=fig.transFigure)
    for name, col in zip(g_names, g_cols):
        gy -= 0.023
        fig.add_artist(plt.Rectangle(
            (lx, gy - 0.008), 0.018, 0.014, color=col,
            transform=fig.transFigure, clip_on=False))
        fig.text(lx + 0.022, gy, name, ha='left', va='center',
                 fontsize=7.5, color='#1A1A2E',
                 transform=fig.transFigure)

# ── SUPER TITLE & FOOTER ──────────────────────────────────────────
fig.suptitle(
    'Anticancer Regimen Toxicity Bullseye Chart v2\n'
    'Sector = Regimen (n=10)  |  Ring = Toxicity Category (center: Neutropenia -> outer: Immune AE)  '
    '|  Color = Grade 3/4 Incidence (%)  |  Numbers shown for >= 8%',
    fontsize=14, fontweight='bold', color='#0D1B4B', y=0.97, ha='center')

fig.text(0.43, 0.025,
    'Light grey (#EBEBEB) = confirmed 0% incidence (not missing data). '
    'All 120 cells (10 regimens x 12 AE categories) populated.\n'
    'Source: NCCN 2024 | ESMO CRC 2023 | MOSAIC 2004 | Falcone 2007 (JCO) | '
    'RECOURSE 2015 (NEJM) | KEYNOTE-177 2021 (NEJM) | Lee ADDS 2026.'
    '  Outer arc color = regimen group (distinct from incidence heatmap).',
    ha='center', fontsize=7.5, color='#555577', style='italic',
    transform=fig.transFigure)

# ── SAVE ──────────────────────────────────────────────────────────
out_path = os.path.join(OUT, 'toxicity_bullseye_v2.png')
plt.savefig(out_path, dpi=175, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.close()
print('Saved:', out_path, '(%d KB)' % (os.path.getsize(out_path)//1024))
print('Matrix: %d regimens x %d tox = %d cells' % (N_REG, N_TOX, N_REG*N_TOX))
print('Zero cells (confirmed 0%%): %d' % (TOX_MATRIX==0).sum())
print('Annotated cells (>=8%%): %d' % (TOX_MATRIX>=8).sum())
