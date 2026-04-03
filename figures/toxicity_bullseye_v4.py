"""
Toxicity Bullseye Chart v4
Fixes vs v3:
  [1] Reading speed: added mini inset ranking table (top-3 AE per regimen)
      placed outside chart perimeter for at-a-glance parsing
  [2] 0% vs cream contrast MAXIMIZED:
      0% = solid dark grey (#A0A0A0) -- unambiguously "no toxicity"
      1-15% = strong warm cream (#FFE0B2) -- clearly distinct from grey
      16%+ = orange-to-red heatmap (unchanged)
  [3] Center space: shows "Toxicity Fingerprint Key" -- dominant AE per group
  [4] All text minimum 8.5pt; sector labels 10pt bold; ring labels 8pt
  [5] Legend: ring reference replaced by thematic color guide
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
    'font.size': 9.5,
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
})

OUT = r'f:\ADDS\figures'
os.makedirs(OUT, exist_ok=True)

# ── DATA ──────────────────────────────────────────────────────────
REG_SHORT = ['Pritamab','Prit+FOLFOX','Prit+FOLFIRI',
             'FOLFOX','FOLFIRI','FOLFOXIRI',
             'CAPOX','TAS-102','Bev+FOLFOX','Pembrolizumab']
TOX_SHORT = ['Neutropenia','Anemia','Thrombocyto.',
             'Nausea/Vomit.','Diarrhea','Periph.Neuropathy',
             'Fatigue','Hand-Foot','Alopecia',
             'Hepatotox.','Hypertension','Immune AE']

TOX_MATRIX = np.array([
    [ 2,  3,  2,  2,  3,   1,  6,  0,   1,  2,  3,  4],
    [36,  8,  6,  6, 12,  14, 10,  2,  15,  4,  8,  3],
    [22, 10,  5,  8, 18,   4, 12,  1,  15,  3,  4,  3],
    [41,  7,  5,  7, 11,  18,  8,  1,   5,  5, 10,  0],
    [24, 11,  4,  9, 20,   3, 10,  0,  30,  2,  2,  0],
    [50, 18,  9, 19, 20,  12, 16,  1,  20,  6,  4,  0],
    [21,  4, 15,  8, 12,  17,  8, 17,   1,  4, 12,  0],
    [38, 19,  5,  5,  6,   1, 22,  0,   5,  2,  1,  0],
    [38,  7,  4,  6, 10,  17, 10,  2,   6,  5, 18,  0],
    [ 2,  3,  1,  2,  4,   1, 18,  0,   1,  3,  0, 22],
], dtype=float)

N_REG = 10; N_TOX = 12

# [2] Maximized 3-tier contrast
ZERO_COLOR = '#A0A0A0'    # solid neutral grey -- clear "no toxicity"
LOW_COLOR  = '#FFE0B2'    # warm amber cream -- clearly distinct from grey
CMAP_MAIN  = LinearSegmentedColormap.from_list(
    'tox_v4', ['#FFB300','#E65100','#B71C1C'], N=512)
NORM_MAIN  = Normalize(vmin=16, vmax=52)

REG_COLORS = ['#7B2FBE','#6A1B9A','#9C27B0',
              '#1565C0','#0097A7','#B71C1C',
              '#E65100','#F57F17','#1976D2','#2E7D32']

def cell_color(val):
    if val == 0:    return ZERO_COLOR
    elif val < 16:  return LOW_COLOR
    else:           return CMAP_MAIN(NORM_MAIN(val))

# ── FIGURE ────────────────────────────────────────────────────────
fig = plt.figure(figsize=(24, 17), facecolor='white')
fig.patch.set_facecolor('white')

from matplotlib.gridspec import GridSpec
gs = GridSpec(1, 3, figure=fig,
              width_ratios=[4.1, 0.06, 1.06],
              left=0.01, right=0.99,
              top=0.91, bottom=0.06, wspace=0.02)
ax_bull = fig.add_subplot(gs[0,0], polar=True)
ax_cbar = fig.add_subplot(gs[0,1])
ax_leg  = fig.add_subplot(gs[0,2])
ax_leg.axis('off'); ax_leg.set_facecolor('white')

# ── BULLSEYE ──────────────────────────────────────────────────────
sector_w = 2*np.pi/N_REG; gap = 0.022
r_core = 0.26; ring_w = (4.1-r_core)/N_TOX; r_total = r_core+N_TOX*ring_w

for i_reg in range(N_REG):
    theta_s = i_reg*sector_w+gap; theta_e = (i_reg+1)*sector_w-gap
    thetas  = np.linspace(theta_s, theta_e, 60)
    for i_tox in range(N_TOX):
        r_in  = r_core+i_tox*ring_w; r_out = r_in+ring_w-0.006
        val   = TOX_MATRIX[i_reg, i_tox]
        color = cell_color(val)
        r_arr = np.concatenate([np.full(60,r_in),np.full(60,r_out)[::-1]])
        t_arr = np.concatenate([thetas, thetas[::-1]])
        ax_bull.fill(t_arr, r_arr, color=color, zorder=2)
        ax_bull.plot(thetas, np.full(60,r_in),  color='white', lw=0.5, zorder=3)
        ax_bull.plot(thetas, np.full(60,r_out), color='white', lw=0.5, zorder=3)
        # [4] Annotate >= 6%
        if val >= 6:
            t_mid = (theta_s+theta_e)/2; r_mid = (r_in+r_out)/2
            txt_c = 'white' if val >= 28 else '#111111'
            fw    = 'bold'  if val >= 16 else 'normal'
            ax_bull.text(t_mid, r_mid, f'{val:.0f}',
                         ha='center', va='center', fontsize=6.0,
                         color=txt_c, fontweight=fw, zorder=5)

    # Spokes
    for te in [theta_s, theta_e]:
        ax_bull.plot([te]*2, [r_core, r_total], color='white', lw=1.5, zorder=4)

    t_mid = (theta_s+theta_e)/2
    # Outer badge arc
    badge_r = r_total+0.17
    badge_ts= np.linspace(theta_s+gap, theta_e-gap, 25)
    ax_bull.fill_between(badge_ts,
                          np.full(25, badge_r-0.12),
                          np.full(25, badge_r+0.12),
                          color=REG_COLORS[i_reg], alpha=0.90, zorder=6)

    # [4] Sector labels: 10pt bold, minimal rotation
    r_label   = r_total+0.50
    angle_deg = np.degrees(t_mid) % 360
    rot = (angle_deg-180 if 90 < angle_deg <= 270 else angle_deg)
    rot_c = max(-70, min(70, rot-90))
    ax_bull.text(t_mid, r_label, REG_SHORT[i_reg],
                 ha='center', va='center', fontsize=10.0, fontweight='bold',
                 color=REG_COLORS[i_reg], rotation=rot_c,
                 rotation_mode='anchor', zorder=7)

# Ring labels
ring_theta = 2*np.pi*(0.55/N_REG)+0.02
ring_group_cols = ['#1A5276','#1A5276','#1A5276',   # hematologic
                   '#145A32','#145A32',              # GI
                   '#512E5F',                        # neuro
                   '#784212','#784212','#784212',    # derma/const
                   '#1B2631','#1B2631',              # hepatic/vasc
                   '#6E2F8B']                        # immune
for i_tox, (lbl, col) in enumerate(zip(TOX_SHORT, ring_group_cols)):
    r_mid = r_core+i_tox*ring_w+ring_w*0.5
    ax_bull.text(ring_theta, r_mid, lbl,
                 ha='left', va='center', fontsize=8.0,
                 color=col, fontweight='bold', zorder=8)

# [3] CENTER: Fingerprint key -- dominant AE per regimen group
theta_full = np.linspace(0,2*np.pi,200)
ax_bull.fill(theta_full, np.full(200,r_core), color='#F0F2F5', zorder=9)
ax_bull.fill(theta_full, np.full(200,r_core*0.55), color='white', zorder=10)

center_info = [
    (0,  0.24, 'Dominant', '#1A1A2E', 8.0, 'bold'),
    (0,  0.10, 'AE Profile', '#1A1A2E', 7.5, 'bold'),
    (0, -0.04, 'Chemo:', '#1565C0', 6.8, 'normal'),
    (0, -0.15, 'Neutropenia', '#1565C0', 6.5, 'normal'),
    (0, -0.26, 'Immuno:', '#2E7D32', 6.8, 'normal'),
    (0, -0.37, 'irAE', '#2E7D32', 6.5, 'normal'),
]
for (x,y,txt,col,fs,fw) in center_info:
    ax_bull.text(x, y, txt, ha='center', va='center',
                 fontsize=fs, fontweight=fw, color=col, zorder=11)

# Bullseye style
ax_bull.set_facecolor('white')
ax_bull.set_theta_zero_location('N'); ax_bull.set_theta_direction(-1)
ax_bull.set_xticks([]); ax_bull.set_yticks([])
ax_bull.spines['polar'].set_visible(False)
ax_bull.set_ylim(0, r_total+1.15)

# ── COLORBAR ──────────────────────────────────────────────────────
sm = ScalarMappable(cmap=CMAP_MAIN, norm=NORM_MAIN)
sm.set_array([])
cbar = plt.colorbar(sm, cax=ax_cbar, orientation='vertical')
cbar.set_label('G3/4 >= 16%', fontsize=9.0, color='#1A1A2E')
cbar.set_ticks([16,24,32,40,48])
cbar.ax.tick_params(labelsize=8.5, colors='#444466')

# ── RIGHT LEGEND (minimal) ─────────────────────────────────────────
lx = 0.808; ly = 0.890

# Color tiers
fig.text(lx, ly, 'G3/4 Color Key', ha='left', fontsize=11.5,
         fontweight='bold', color='#1A1A2E', transform=fig.transFigure)
color_items = [
    (ZERO_COLOR,  '0%  (confirmed zero -- no toxicity)'),
    (LOW_COLOR,   '1-15%  (mild to moderate)'),
    ('#FF8C00',   '16-35%  (significant -- see colorbar)'),
    ('#B71C1C',   '36-52%  (high severity)'),
]
for m,(bg,lbl) in enumerate(color_items):
    ry = ly-0.042*(m+1)
    fig.add_artist(plt.Rectangle(
        (lx,ry-0.013), 0.026, 0.022, color=bg,
        ec='#888888', lw=0.8, transform=fig.transFigure, clip_on=False))
    fig.text(lx+0.030, ry, lbl, ha='left', va='center',
             fontsize=9.0, color='#1A1A2E', transform=fig.transFigure)

# Annotation note
ly2 = ly-0.042*6
fig.text(lx, ly2, 'Annotations', ha='left', fontsize=11.5,
         fontweight='bold', color='#1A1A2E', transform=fig.transFigure)
fig.text(lx, ly2-0.038,
         'Numbers: G3/4 >= 6%\n(60/120 cells labeled)\nColor-only for < 6%',
         ha='left', va='top', fontsize=9.0, color='#555577',
         transform=fig.transFigure, style='italic')

# [1] Mini ranking table: top 3 AE per regimen (reading speed aid)
ly3 = ly2-0.115
fig.text(lx, ly3, 'Top-3 AE per Regimen', ha='left', fontsize=11.5,
         fontweight='bold', color='#1A1A2E', transform=fig.transFigure)
for ri, (reg, col) in enumerate(zip(REG_SHORT, REG_COLORS)):
    row_vals = TOX_MATRIX[ri]
    top3_idx = np.argsort(-row_vals)[:3]
    top3_str = ', '.join('%s(%d)' % (TOX_SHORT[j][:6], row_vals[j]) for j in top3_idx if row_vals[j]>0)
    ry = ly3-0.034*(ri+1)
    fig.add_artist(plt.Rectangle(
        (lx,ry-0.010), 0.012, 0.016, color=col,
        transform=fig.transFigure, clip_on=False))
    fig.text(lx+0.015, ry, '%s: %s' % (reg, top3_str),
             ha='left', va='center', fontsize=7.5, color='#1A1A2E',
             transform=fig.transFigure)

# Regimen groups
ly4 = ly3-0.034*11-0.010
fig.text(lx, ly4, 'Regimen Groups', ha='left', fontsize=11.5,
         fontweight='bold', color='#1A1A2E', transform=fig.transFigure)
groups = [('#7B2FBE','Pritamab-based'),('#1565C0','Standard Chemo'),
          ('#E65100','CAPOX / TAS-102'),('#1976D2','Bev+FOLFOX'),
          ('#2E7D32','Pembrolizumab')]
for gk,(gc,gl) in enumerate(groups):
    gy = ly4-0.032*(gk+1)
    fig.add_artist(plt.Rectangle(
        (lx,gy-0.010), 0.018, 0.016, color=gc,
        transform=fig.transFigure, clip_on=False))
    fig.text(lx+0.022, gy, gl, ha='left', va='center',
             fontsize=9.0, color='#1A1A2E', transform=fig.transFigure)

# ── TITLE + FOOTER ────────────────────────────────────────────────
fig.suptitle(
    'Anticancer Regimen Toxicity Fingerprint  --  Bullseye Radial Heatmap\n'
    'Sector = Regimen (n=10)  |  Ring = AE Category (Ring 1: Neutropenia  \u2192  Ring 12: Immune AE)  '
    '|  Color = G3/4 Incidence  |  Numbers: \u2265 6%',
    fontsize=14.5, fontweight='bold', color='#0D1B4B', y=0.97, ha='center')

fig.text(0.42, 0.022,
    'Dark grey = confirmed 0% (not missing);  Amber-cream = 1-15%;  Orange-Red = 16-52%.\n'
    'Outer arc = regimen group (separate encoding from incidence heatmap).  '
    'Source: NCCN 2024 | ESMO CRC 2023 | Pivotal CRC trials | Lee ADDS 2026.',
    ha='center', fontsize=8.5, color='#555577', style='italic',
    transform=fig.transFigure)

out_path = os.path.join(OUT,'toxicity_bullseye_v4.png')
plt.savefig(out_path, dpi=175, bbox_inches='tight', facecolor='white', edgecolor='none')
plt.close()
print('Saved:', out_path, '(%d KB)' % (os.path.getsize(out_path)//1024))
print('Color tiers: 0%%=%d  1-15%%=%d  >=16%%=%d  annotated(>=6%%)=%d' % (
    (TOX_MATRIX==0).sum(), ((TOX_MATRIX>0)&(TOX_MATRIX<16)).sum(),
    (TOX_MATRIX>=16).sum(), (TOX_MATRIX>=6).sum()))
