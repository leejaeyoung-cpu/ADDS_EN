"""
Fig.3  Anticancer Drug Response — Pritamab Combination Regimens
===============================================================
Style: white background, clinical illustration
  - Drug icon row (pill/bottle icons): Irinotecan, Oxaliplatin, TAS-102, Pritamab
  - Efficacy comparison bars (FOLFOX vs FOLFOX+Pritamab; FOLFIRI vs FOLFIRI+Pritamab)
  - Neutralisation efficiency gauge (55% → >75%)
  - Toxicity comparison radar inset
  - PRITAMAB efficacy summary box (styled like reference image)
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False
import matplotlib.patches as mpatches
from matplotlib.patches import (FancyBboxPatch, FancyArrowPatch,
                                 Circle, Arc, Wedge)
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.gridspec as gridspec
import os

OUT = r'f:\ADDS\pritamab\figures'
os.makedirs(OUT, exist_ok=True)

NAVY   = '#1E3A5F'; BLUE   = '#2563EB'; LTBLUE = '#DBEAFE'
TEAL   = '#0D9488'; LTEAL  = '#CCFBF1'
RED    = '#DC2626'; LRED   = '#FEE2E2'
PURPLE = '#7C3AED'; LPUR   = '#EDE9FE'
GOLD   = '#D97706'; LGOLD  = '#FEF3C7'
GREEN  = '#16A34A'; LGREEN = '#DCFCE7'
ORANGE = '#EA580C'; LORANGE= '#FED7AA'
GRAY   = '#6B7280'; LGRAY  = '#F1F5F9'
BG     = 'white'

fig = plt.figure(figsize=(17/2.54, 22/2.54), facecolor=BG)
gs  = gridspec.GridSpec(4, 3, figure=fig,
                        left=0.07, right=0.96, top=0.93, bottom=0.05,
                        hspace=0.55, wspace=0.42)

# ══════════════════════════════════════════════════════════════════════════════
# ROW 0: Drug Icons (across full width)
# ══════════════════════════════════════════════════════════════════════════════
ax_drugs = fig.add_subplot(gs[0, :])
ax_drugs.set_xlim(0, 10); ax_drugs.set_ylim(0, 3.5)
ax_drugs.axis('off')

drugs = [
    ('Irinotecan\n(CPT-11)', TEAL,   '#0F766E', 1.2),
    ('Oxaliplatin',           BLUE,   '#1D4ED8', 3.2),
    ('TAS-102\n(Tipiracil)', PURPLE,  '#5B21B6', 5.2),
    ('FOLFOX',               ORANGE,  '#C2410C', 7.2),
    ('Pritamab',             NAVY,    '#0F172A', 9.0),
]

def draw_bottle(ax, x, y, fc, ec=None, h=1.2, w=0.5):
    """Draw a simple medicine bottle."""
    ec = ec or fc
    # Bottle body
    body = FancyBboxPatch((x-w/2, y-h/2), w, h,
        boxstyle='round,pad=0.07',
        fc=fc, ec=ec, lw=1.5, alpha=0.90)
    ax.add_patch(body)
    # Cap
    cap = FancyBboxPatch((x-w*0.3, y+h/2-0.02), w*0.6, h*0.2,
        boxstyle='round,pad=0.03',
        fc=ec, ec='white', lw=0.8)
    ax.add_patch(cap)
    # Label stripe
    stripe = FancyBboxPatch((x-w/2+0.04, y-0.05), w-0.08, 0.28,
        boxstyle='square,pad=0', fc='white', ec='none', alpha=0.40)
    ax.add_patch(stripe)

for name, fc, ec, xp in drugs:
    draw_bottle(ax_drugs, xp, 1.7, fc, ec)
    ax_drugs.text(xp, 0.4, name, ha='center', fontsize=6.2,
                  color=NAVY, fontweight='bold')

ax_drugs.set_title('Drug Icons: Combination Regimens',
                   fontsize=8.5, fontweight='bold', color=NAVY)

# ══════════════════════════════════════════════════════════════════════════════
# ROW 1-left: Efficacy bar chart
# ══════════════════════════════════════════════════════════════════════════════
ax_bar = fig.add_subplot(gs[1, :2])
ax_bar.set_facecolor(BG)
ax_bar.spines[['top','right']].set_visible(False)
ax_bar.spines[['left','bottom']].set_color('#CBD5E1')

arms   = ['FOLFOX', 'FOLFOX\n+Pritamab', 'FOLFIRI', 'FOLFIRI\n+Pritamab']
cols   = [ORANGE, NAVY, TEAL, BLUE]
means  = [58.3, 74.1, 55.8, 71.4]
sds    = [4.2, 3.8, 4.5, 3.6]

x_pos = np.arange(len(arms))
bars  = ax_bar.bar(x_pos, means, yerr=sds, color=cols, alpha=0.88,
                   width=0.55, capsize=4, zorder=3,
                   error_kw=dict(ecolor='#374151', lw=1.5, capthick=1.5))
for b, mean in zip(bars, means):
    ax_bar.text(b.get_x()+b.get_width()/2, mean+1.8,
                f'{mean:.1f}%', ha='center', fontsize=7.5,
                fontweight='bold', color='#1E293B')

# Significance brackets
def sig_bracket(ax, x1, x2, y, p_text, col='#374151'):
    ax.plot([x1, x1, x2, x2], [y, y+1.0, y+1.0, y], color=col, lw=1.3)
    ax.text((x1+x2)/2, y+1.2, p_text, ha='center', fontsize=7, color=col)

sig_bracket(ax_bar, 0, 1, 80, 'p < 0.001 ***', RED)
sig_bracket(ax_bar, 2, 3, 80, 'p = 0.002 **', RED)

ax_bar.set_xticks(x_pos); ax_bar.set_xticklabels(arms, fontsize=7.5)
ax_bar.set_ylim(0, 93)
ax_bar.set_ylabel('Predicted Efficacy Score (%)', fontsize=7.5, color=GRAY)
ax_bar.set_title('(A)  Predicted Efficacy — Phase II Virtual Trial (n=400)',
                 fontsize=8.5, fontweight='bold')
ax_bar.grid(axis='y', alpha=0.25, zorder=0)

# ══════════════════════════════════════════════════════════════════════════════
# ROW 1-right: Neutralisation efficiency gauge
# ══════════════════════════════════════════════════════════════════════════════
ax_g = fig.add_subplot(gs[1, 2])
ax_g.set_xlim(0, 4); ax_g.set_ylim(0, 4)
ax_g.axis('off')
ax_g.set_facecolor(BG)

# Background arc (full semicircle, grey)
theta_bg = np.linspace(np.pi, 0, 200)
ax_g.plot(2+1.5*np.cos(theta_bg), 1.8+1.5*np.sin(theta_bg),
          color='#E5E7EB', lw=12, solid_capstyle='round', zorder=2)

# Alone arm: 75% → 135° in the semicircle
for pct, col, lbl, yoff in [(0.55, '#94A3B8', 'Alone 55%', 0.4),
                              (0.75, GREEN,      '>75% +Chemo', -0.1)]:
    theta_pct = np.pi - (np.pi * pct)
    th_arc = np.linspace(np.pi, theta_pct, 200)
    ax_g.plot(2+1.5*np.cos(th_arc), 1.8+1.5*np.sin(th_arc),
              color=col, lw=12, solid_capstyle='round', zorder=3)

# Needle (for 75%)
needle_ang = np.pi - np.pi*0.75
ax_g.annotate('', xy=(2+1.3*np.cos(needle_ang), 1.8+1.3*np.sin(needle_ang)),
              xytext=(2, 1.8),
              arrowprops=dict(arrowstyle='->', color=NAVY, lw=2.5))

ax_g.text(2, 1.0, '>75%', ha='center', fontsize=14,
          color=GREEN, fontweight='bold')
ax_g.text(2, 0.55, 'Neutralisation\nEfficiency', ha='center',
          fontsize=6.5, color=GRAY)
ax_g.text(0.35, 1.8, '0%', fontsize=6.5, color=GRAY, ha='center')
ax_g.text(3.65, 1.8, '100%', fontsize=6.5, color=GRAY, ha='center')
ax_g.set_title('(B)  Pritamab\nNeutralisation Eff.',
               fontsize=7.5, fontweight='bold', pad=3)

# ══════════════════════════════════════════════════════════════════════════════
# ROW 2: Toxicity radar chart (polar)
# ══════════════════════════════════════════════════════════════════════════════
ax_rad = fig.add_subplot(gs[2, :2], projection='polar')
categories = ['Neutropenia', 'Nausea', 'Fatigue', 'Diarrhoea',
              'Peripheral\nNeuropathy', 'Anaemia', 'Alopecia', 'Mucositis']
N = len(categories)
angles = [n/float(N)*2*np.pi for n in range(N)]
angles += angles[:1]   # close the loop

data = {
    'FOLFOX':          [48, 35, 42, 22, 52, 28, 8,  18],
    'FOLFIRI':         [38, 58, 40, 48, 12, 32, 72, 28],
    'FOLFOX+Pritamab': [35, 28, 35, 18, 44, 24, 6,  14],
}
arm_cols = {'FOLFOX': ORANGE, 'FOLFIRI': TEAL, 'FOLFOX+Pritamab': NAVY}

ax_rad.set_theta_offset(np.pi/2); ax_rad.set_theta_direction(-1)
ax_rad.set_rlabel_position(0)
plt.xticks(angles[:-1], categories, size=6.0, color=GRAY)
ax_rad.set_ylim(0, 80)
ax_rad.set_yticks([20, 40, 60, 80])
ax_rad.set_yticklabels(['20','40','60','80'], fontsize=5.5, color='#9CA3AF')
ax_rad.set_facecolor(BG)
ax_rad.spines['polar'].set_color('#E5E7EB')

for arm, vals in data.items():
    v = vals + vals[:1]
    ax_rad.plot(angles, v, 'o-', lw=1.8, ms=4,
                color=arm_cols[arm], label=arm)
    ax_rad.fill(angles, v, alpha=0.10, color=arm_cols[arm])

ax_rad.legend(loc='upper right', bbox_to_anchor=(1.45, 1.15),
              fontsize=6.5, framealpha=0.85)
ax_rad.set_title('(C)  Grade 3/4 Adverse Event Profile (%)',
                 fontsize=8.5, fontweight='bold', pad=22)

# ══════════════════════════════════════════════════════════════════════════════
# ROW 2-right + ROW 3-right: PRITAMAB Efficacy Summary box
#  (styled like the dark box in the reference image, but white bg)
# ══════════════════════════════════════════════════════════════════════════════
ax_sum = fig.add_subplot(gs[2:, 2])
ax_sum.set_xlim(0, 5); ax_sum.set_ylim(0, 7.5)
ax_sum.axis('off')

# Box background
eff_box = FancyBboxPatch((0.05, 0.05), 4.90, 7.35,
    boxstyle='round,pad=0.15',
    fc=NAVY, ec='white', lw=0)
ax_sum.add_patch(eff_box)

ax_sum.text(2.5, 7.0, 'PRITAMAB efficacy',
            ha='center', fontsize=9, fontweight='bold',
            color='white')

# Bullet items
bullets = [
    ('중화 효율',          'Neutralisation Efficiency: 55%', 'white'),
    ('',                   '>75% alone  75% with chemo',    GREEN),
    ('세보지열자 도 (ΔG)',  '결합 올고 Ehog도 촉키가-경확',   'white'),
    ('',                   '경리현기미 획식는 숲이 문입니다', '#CBD5E1'),
]
y_b = 6.3
for hdr, body, col in bullets:
    if hdr:
        ax_sum.text(0.3, y_b, f'• {hdr}', fontsize=6.8,
                    color='white', fontweight='bold')
        y_b -= 0.45
    ax_sum.text(0.5, y_b, body, fontsize=6.2, color=col)
    y_b -= 0.52

# Efficacy gauge: simple bar inside box
ax_sum.add_patch(FancyBboxPatch((0.3, 3.5), 4.4, 0.45,
    boxstyle='round,pad=0.05', fc='#374151', ec='none'))
ax_sum.add_patch(FancyBboxPatch((0.3, 3.5), 4.4*0.75, 0.45,
    boxstyle='round,pad=0.05', fc=GREEN, ec='none'))
ax_sum.text(2.5, 3.3, '>75%  efficacy with chemo',
            ha='center', fontsize=6, color='#CBD5E1')

# Mini drug bottles inside summary box
mini_drugs = ['Irinotecan', 'Oxaliplatin', 'TAS-102']
mini_cols  = [TEAL, BLUE, PURPLE]
for di, (dn, dc) in enumerate(zip(mini_drugs, mini_cols)):
    bx = 0.5 + di*1.5
    ax_sum.add_patch(FancyBboxPatch((bx-0.3, 0.5), 0.6, 1.1,
        boxstyle='round,pad=0.05', fc=dc, ec='white', lw=1.0, alpha=0.90))
    ax_sum.add_patch(FancyBboxPatch((bx-0.18, 1.58), 0.36, 0.28,
        boxstyle='round,pad=0.02', fc='white', ec='none', alpha=0.25))
    ax_sum.text(bx, 0.28, dn, ha='center', fontsize=5.0,
                color='#94A3B8')

ax_sum.set_title('Efficacy Summary', fontsize=8, fontweight='bold',
                 color=NAVY, pad=4)

# ══════════════════════════════════════════════════════════════════════════════
# ROW 3-left: ORR / DCR comparison table
# ══════════════════════════════════════════════════════════════════════════════
ax_tbl = fig.add_subplot(gs[3, :2])
ax_tbl.set_xlim(0, 10); ax_tbl.set_ylim(0, 5.5)
ax_tbl.axis('off')

headers = ['Regimen', 'ORR (%)', 'DCR (%)', 'mPFS (mo)', 'mOS (mo)']
rows = [
    ('FOLFOX',            '22.4', '62.1', '6.8', '18.3'),
    ('FOLFOX+Pritamab',   '31.7', '74.8', '9.2', '24.1'),
    ('FOLFIRI',           '19.8', '58.4', '6.1', '17.2'),
    ('FOLFIRI+Pritamab',  '28.5', '70.2', '8.7', '22.8'),
]
col_xs = [0.1, 2.7, 4.6, 6.4, 8.2]
head_y = 5.0
# Header
header_bg = FancyBboxPatch((0, 4.55), 9.9, 0.65,
    boxstyle='square,pad=0', fc=NAVY, ec='none')
ax_tbl.add_patch(header_bg)
for hdr, hx in zip(headers, col_xs):
    ax_tbl.text(hx, head_y, hdr, fontsize=7.5, color='white',
                fontweight='bold', va='center')

for ri, row in enumerate(rows):
    row_y = 4.0 - ri*0.88
    bg_col = LTBLUE if '+Pritamab' in row[0] else BG
    ax_tbl.add_patch(FancyBboxPatch((0, row_y-0.4), 9.9, 0.82,
        boxstyle='square,pad=0', fc=bg_col, ec='#E5E7EB', lw=0.6))
    tc = NAVY if '+Pritamab' in row[0] else GRAY
    for val, hx in zip(row, col_xs):
        fw = 'bold' if '+Pritamab' in row[0] else 'normal'
        ax_tbl.text(hx, row_y, val, fontsize=7, color=tc,
                    fontweight=fw, va='center')

ax_tbl.set_title('(D)  Simulated Clinical Outcomes', fontsize=8.5,
                 fontweight='bold', pad=4)

# ══════════════════════════════════════════════════════════════════════════════
fig.suptitle('Figure 3.  Pritamab Combination — Anticancer Drug Response',
             fontsize=10, fontweight='bold', color=NAVY, y=0.97)

for ext in ['png', 'pdf']:
    fpath = os.path.join(OUT, f'fig3_drug_response_v4.{ext}')
    fig.savefig(fpath, dpi=300, bbox_inches='tight',
                facecolor=BG, edgecolor='none')
    print(f'Saved: {fpath}')
plt.close(fig)
print('Done.')
