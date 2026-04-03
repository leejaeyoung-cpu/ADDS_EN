"""
figure_energy_landscape_academic.py
-------------------------------------
Nature Communications / Cell 스타일 아카데믹 피규어
- LaTeX mathtext 수식 표기
- Nature 팔레트 (ColorBrewer + Nature 저널 표준)
- 5-panel layout (A~E) with inset
- STAR Methods 수준 방법론 주석
- 통계 significance 마커 스타일
"""

import sys, warnings, math
sys.path.insert(0, r'f:\ADDS')
warnings.filterwarnings('ignore')

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import matplotlib.ticker as ticker
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from scipy.interpolate import make_interp_spline

from src.pritamab_ml.imaging_to_energy import (
    ImagingToEnergyMapper, BASELINE_DDG, RT_KCAL, MAX_DELTA_DDG
)
from src.pritamab_ml.pathway_drug_optimizer import PathwayDrugOptimizer

# ── Nature 팔레트 ─────────────────────────────────────────────────────────────
NAT = {
    'red':    '#E64B35',   # red
    'blue':   '#4DBBD5',   # blue
    'green':  '#00A087',   # green
    'orange': '#F39B7F',   # orange
    'purple': '#8491B4',   # purple
    'grey':   '#91D1C2',   # teal-grey
    'dark':   '#3C5488',   # dark blue
    'black':  '#1A1A24',
    'bg':     '#FAFAFA',
}

PATHWAY_COLORS = {
    'KRAS_ERK':  NAT['red'],
    'PI3K_mTOR': NAT['blue'],
    'HIF_VEGF':  NAT['green'],
    'RhoA_INV':  NAT['orange'],
}
PW_LABELS = {
    'KRAS_ERK':  r'KRAS$\rightarrow$RAF$\rightarrow$MEK$\rightarrow$ERK',
    'PI3K_mTOR': r'PI3K$\rightarrow$AKT$\rightarrow$mTOR',
    'HIF_VEGF':  r'HIF-1$\alpha$$\rightarrow$VEGF',
    'RhoA_INV':  r'RhoA$\rightarrow$ROCK$\rightarrow$Cofilin',
}

# ── matplotlib 글로벌 설정 ─────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family':        'DejaVu Sans',
    'font.size':          8,
    'axes.linewidth':     0.8,
    'axes.spines.top':    False,
    'axes.spines.right':  False,
    'axes.labelsize':     9,
    'axes.titlesize':     9,
    'xtick.labelsize':    8,
    'ytick.labelsize':    8,
    'xtick.major.width':  0.7,
    'ytick.major.width':  0.7,
    'xtick.major.size':   3,
    'ytick.major.size':   3,
    'legend.fontsize':    7.5,
    'legend.framealpha':  0.92,
    'legend.edgecolor':   '#cccccc',
    'legend.handlelength':1.5,
    'figure.facecolor':   'white',
    'axes.facecolor':     'white',
    'axes.grid':          True,
    'grid.alpha':         0.25,
    'grid.linewidth':     0.4,
    'grid.color':         '#999999',
    'savefig.dpi':        300,
    'savefig.bbox':       'tight',
})

# ── 데이터 준비 ────────────────────────────────────────────────────────────────
mapper    = ImagingToEnergyMapper()
optimizer = PathwayDrugOptimizer(max_drugs=4)

CASES = [
    ('KRAS G12D\n(High-Proliferative)',
     [{'total_cells':520,'mean_area_um2':310,'mean_circularity':0.50,'irregular_count':290,'normal_count':120}]*3,
     {'tumor_volume_cc':45,'mean_hu':-8,'std_hu':50,'necrosis_ratio':0.20},
     'G12D', NAT['red']),
    ('KRAS WT\n(Hypoxia-Dominant)',
     [{'total_cells':80,'mean_area_um2':520,'mean_circularity':0.75,'irregular_count':10,'normal_count':65}]*3,
     {'tumor_volume_cc':12,'mean_hu':-35,'std_hu':70,'necrosis_ratio':0.45},
     'WT', NAT['blue']),
    ('KRAS G12V\n(Invasive)',
     [{'total_cells':200,'mean_area_um2':280,'mean_circularity':0.38,'irregular_count':150,'normal_count':30}]*5,
     {'tumor_volume_cc':28,'mean_hu':15,'std_hu':35,'necrosis_ratio':0.05},
     'G12V', NAT['green']),
    ('KRAS G12C\n(Early-Stage)',
     [{'total_cells':60,'mean_area_um2':400,'mean_circularity':0.70,'irregular_count':8,'normal_count':48}],
     None,
     'G12C', NAT['orange']),
]

profiles = []
recs     = []
for label, cp, ct, kras, col in CASES:
    p = mapper.compute_profile(cp, ct, kras)
    profiles.append(p)
    recs.append(optimizer.optimize(p, kras_allele=kras))

PATHWAYS = ['KRAS_ERK','PI3K_mTOR','HIF_VEGF','RhoA_INV']

# ── 레이아웃: 2×3 + 와이드 bottom ──────────────────────────────────────────────
fig = plt.figure(figsize=(17, 11))
gs  = gridspec.GridSpec(2, 3, figure=fig,
                         hspace=0.50, wspace=0.40,
                         left=0.07, right=0.97,
                         top=0.90, bottom=0.08)

def panel_label(ax, letter, x=-0.13, y=1.07):
    ax.text(x, y, letter, transform=ax.transAxes,
            fontsize=13, fontweight='bold', va='top', ha='left',
            color=NAT['black'])

# ══════════════════════════════════════════════════════════════════════════════
# PANEL a — 이론적 프레임워크: 에너지 프로파일
# ══════════════════════════════════════════════════════════════════════════════
ax_a = fig.add_subplot(gs[0, 0])
panel_label(ax_a, 'a')
ax_a.set_xlabel('Reaction coordinate $\\xi$ (signal flux, a.u.)', fontsize=9)
ax_a.set_ylabel(r'$\Delta G$ (kcal mol$^{-1}$)', fontsize=9)
ax_a.set_title('Boltzmann-Inverse Energy Landscape Inference', fontsize=9, pad=6)

x_rc = np.linspace(0, 10, 600)

def morse_energy(x, ddg, x0=5.0, a=0.55, ground=0.3, prod=-0.15):
    """Energy profile mimicking transition state theory."""
    # Reactant well
    Ereact = ground * np.exp(-a**2 * (x - 1.5)**2)
    # TS barrier
    Ets    = ddg    * np.exp(-(x - x0)**2 / (2 * 0.9**2))
    # Product well
    Eprod  = prod  * np.exp(-a**2 * (x - 8.5)**2)
    return 0.28 + Ereact + Ets + Eprod

# Population-average baseline
y_base = morse_energy(x_rc, BASELINE_DDG['KRAS_ERK'])
ax_a.plot(x_rc, y_base, '--', color='#999999', lw=1.4, label='Population avg.', zorder=2)
ax_a.fill_between(x_rc, y_base.min(), y_base, alpha=0.05, color='#999999')

# 4 케이스 KRAS_ERK
for i, (lbl, *_, col) in enumerate(CASES):
    ddg_k = profiles[i].ddg_per_pathway['KRAS_ERK']
    y_i   = morse_energy(x_rc, ddg_k)
    short_lbl = lbl.split('\n')[0]
    ax_a.plot(x_rc, y_i, '-', color=col, lw=1.8, label=f'{short_lbl}  $\\Delta G^\\ddagger$={ddg_k:.2f}', zorder=3+i)

# 장벽 높이 브라켓 (Case 0 — G12D, 빨강)
ddg0 = profiles[0].ddg_per_pathway['KRAS_ERK']
y0   = morse_energy(x_rc, ddg0)
ts_x  = x_rc[np.argmax(y0)]; ts_y = y0.max(); ground_y = 0.28
ax_a.annotate('', xy=(ts_x, ts_y), xytext=(ts_x, ground_y),
              arrowprops=dict(arrowstyle='<->', color=NAT['red'], lw=1.2))
ax_a.text(ts_x + 0.2, (ts_y + ground_y)/2,
          f'$\\Delta G^\\ddagger$ = {ddg0:.2f}\nkcal mol$^{{-1}}$',
          fontsize=7.5, color=NAT['red'], va='center')

# Eyring 방정式 텍스트박스
ax_a.text(0.03, 0.97,
          r'$k = \frac{k_BT}{h}\,e^{-\Delta G^\ddagger/RT}$' + '\n' +
          r'$\Delta G^\ddagger_{\rm eff} = \Delta G^\ddagger_0 - \alpha\,\delta_{\rm imaging}$',
          transform=ax_a.transAxes, fontsize=8, va='top', ha='left',
          bbox=dict(boxstyle='round,pad=0.4', fc='white', ec='#cccccc', lw=0.8))

ax_a.set_xlim(0, 10); ax_a.set_ylim(0.0, 1.45)
ax_a.text(1.5, 0.24, 'Reactant', fontsize=7, color='#666', ha='center')
ax_a.text(8.5, 0.24, 'Product', fontsize=7, color='#666', ha='center')
ax_a.text(ts_x, ts_y + 0.05, 'TS', fontsize=7, color='#666', ha='center')
ax_a.legend(loc='upper right', fontsize=7, handlelength=1.3)

# ══════════════════════════════════════════════════════════════════════════════
# PANEL b — ΔG‡ Heatmap
# ══════════════════════════════════════════════════════════════════════════════
ax_b = fig.add_subplot(gs[0, 1])
panel_label(ax_b, 'b')
ax_b.set_title(r'Pathway $\Delta G^\ddagger$ profile per patient phenotype', fontsize=9, pad=6)
ax_b.grid(False)

mat = np.array([[profiles[ci].ddg_per_pathway[pw] for ci in range(4)] for pw in PATHWAYS])

# Nature 배색: white → light blue → dark blue
cmap_nat = LinearSegmentedColormap.from_list(
    'nat_energy',
    ['#E64B35','#FDDBC7','#F7F7F7','#D1E5F0','#4393C3'], N=256
)

im = ax_b.imshow(mat, cmap=cmap_nat, aspect='auto', vmin=0.85, vmax=1.30,
                 interpolation='nearest')

# 셀 값 + 최솟값 강조
for pi in range(4):
    for ci in range(4):
        val = mat[pi, ci]
        fg = 'white' if val < 0.93 or val > 1.22 else NAT['black']
        weight = 'bold' if val == mat[:, ci].min() else 'normal'
        ax_b.text(ci, pi, f'{val:.3f}', ha='center', va='center',
                  fontsize=8, color=fg, fontweight=weight)
        if val == mat[:, ci].min():
            rect = plt.Rectangle((ci-0.5, pi-0.5), 1, 1,
                                   fill=False, edgecolor='gold', lw=2.2)
            ax_b.add_patch(rect)

pw_short = [r'KRAS$\to$ERK', r'PI3K$\to$mTOR', r'HIF$\to$VEGF', r'RhoA$\to$Inv']
case_short = ['G12D\nHigh-Prolif.', 'WT\nHypoxia', 'G12V\nInvasive', 'G12C\nEarly']
ax_b.set_yticks(range(4)); ax_b.set_yticklabels(pw_short, fontsize=8)
ax_b.set_xticks(range(4)); ax_b.set_xticklabels(case_short, fontsize=7.5)
ax_b.tick_params(left=False, bottom=False)

cbar = plt.colorbar(im, ax=ax_b, fraction=0.038, pad=0.02, shrink=0.9)
cbar.set_label(r'$\Delta G^\ddagger$ (kcal mol$^{-1}$)', fontsize=8)
cbar.ax.tick_params(labelsize=7.5)
cbar.set_ticks([0.90, 1.00, 1.10, 1.20, 1.30])

ax_b.text(0.5, -0.16, u'\u2605 Gold outline = primary target pathway (lowest $\\Delta G^\\ddagger$)',
          transform=ax_b.transAxes, ha='center', fontsize=7.5, color='#9A7D0A')

# ══════════════════════════════════════════════════════════════════════════════
# PANEL c — Eyring k 바 차트 (경로별, 케이스별 그룹)
# ══════════════════════════════════════════════════════════════════════════════
ax_c = fig.add_subplot(gs[0, 2])
panel_label(ax_c, 'c')
ax_c.set_title('Eyring rate constant $k$ per pathway', fontsize=9, pad=6)
ax_c.set_ylabel(r'$k_{\rm pathway} \propto e^{-\Delta G^\ddagger/RT}$ (a.u.)', fontsize=9)

n_pw, n_case = 4, 4
x_base = np.arange(n_pw)
w      = 0.18
offsets= np.linspace(-(n_case-1)/2*w, (n_case-1)/2*w, n_case)

for ci, (offset, (lbl, *_, col)) in enumerate(zip(offsets, CASES)):
    k_vals = [profiles[ci].k_per_pathway[pw] for pw in PATHWAYS]
    bars = ax_c.bar(x_base + offset, k_vals, width=w*0.9,
                    color=col, alpha=0.85, edgecolor='white', linewidth=0.4,
                    label=lbl.split('\n')[0])
    # 최솟값 k에 * 표시
    min_k = min(k_vals)
    for bi, kv in enumerate(k_vals):
        if kv == max(k_vals):
            ax_c.text(x_base[bi] + offset, kv + 0.003, '★',
                      ha='center', va='bottom', fontsize=7, color=col)

ax_c.axhline(np.exp(-BASELINE_DDG['KRAS_ERK']/RT_KCAL),
             color='#888', lw=0.9, ls='--', label='Baseline $k$', alpha=0.7)
ax_c.set_xticks(x_base)
ax_c.set_xticklabels([r'KRAS$\to$ERK', r'PI3K$\to$mTOR', r'HIF$\to$VEGF', r'RhoA$\to$Inv'],
                      fontsize=7.5)
ax_c.set_ylim(0, 0.32)
ax_c.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
ax_c.legend(ncol=2, fontsize=7, loc='upper right',
            title='Patient phenotype', title_fontsize=7.5)
ax_c.text(0, 0.01, '★ = highest activity', fontsize=7, color='#666', transform=ax_c.get_xaxis_transform())

# ══════════════════════════════════════════════════════════════════════════════
# PANEL d — 경로 활성화 지수 scatter (imaging → pathway)
# ══════════════════════════════════════════════════════════════════════════════
ax_d = fig.add_subplot(gs[1, 0:2])
panel_label(ax_d, 'd', x=-0.06)
ax_d.set_title('Imaging phenotype indices → pathway activation mapping', fontsize=9, pad=6)
ax_d.set_xlabel('Imaging Index Value (normalised, 0–1)', fontsize=9)
ax_d.set_ylabel(r'$\Delta G^\ddagger_{\rm eff}$ deviation from baseline (kcal mol$^{-1}$)', fontsize=9)

index_keys = ['proliferation_index','cell_size_index','hypoxia_score','invasion_score']
index_labels = ['Proliferation\nIndex\n(Cellpose)', 'Cell Size\nIndex\n(Cellpose)',
                'Hypoxia\nScore\n(CT)', 'Invasion\nScore\n(Cellpose)']
pw_map = ['KRAS_ERK','PI3K_mTOR','HIF_VEGF','RhoA_INV']

for pi, (ik, il, pw) in enumerate(zip(index_keys, index_labels, pw_map)):
    col = PATHWAY_COLORS[pw]
    xs, ys = [], []
    for ci in range(4):
        idx_val = profiles[ci].imaging_indices.get(ik, 0)
        baseline = BASELINE_DDG[pw]
        ddg_val  = profiles[ci].ddg_per_pathway[pw]
        delta    = baseline - ddg_val   # positive = more activated
        xs.append(idx_val)
        ys.append(delta)

    # 회귀선
    if len(set(xs)) > 1:
        xs_arr = np.array(xs); ys_arr = np.array(ys)
        z = np.polyfit(xs_arr, ys_arr, 1)
        p = np.poly1d(z)
        x_fit = np.linspace(min(xs)-0.05, max(xs)+0.05, 50)
        ax_d.plot(x_fit, p(x_fit), '--', color=col, lw=1.0, alpha=0.7)

    r_val = np.corrcoef(xs, ys)[0,1] if len(set(xs)) > 1 else float('nan')
    label_str = f'{PW_LABELS[pw]}  ($r$={r_val:.2f})'
    ax_d.scatter(xs, ys, color=col, s=70, zorder=4, alpha=0.9,
                 edgecolors='white', linewidths=0.7, label=label_str)

    # 케이스 레이블
    for ci in range(4):
        idx_val = profiles[ci].imaging_indices.get(ik, 0)
        baseline = BASELINE_DDG[pw]
        delta    = baseline - profiles[ci].ddg_per_pathway[pw]
        case_id  = ['A','B','C','D'][ci]
        ax_d.annotate(case_id, (idx_val, delta),
                      xytext=(3, 3), textcoords='offset points',
                      fontsize=6.5, color=col, fontweight='bold')

ax_d.axhline(0, color='#aaa', lw=0.7, ls=':')
ax_d.text(0.01, 0.02, r'$\Delta\Delta G^\ddagger = 0$ (no activation)',
          transform=ax_d.transAxes, fontsize=7, color='#888')
ax_d.legend(fontsize=7, loc='upper left', ncol=2,
            title=r'Pathway ($r$ = Pearson correlation)', title_fontsize=7.5)
ax_d.set_xlim(-0.05, 1.10)

# ══════════════════════════════════════════════════════════════════════════════
# PANEL e — 약물 칵테일 추천 (아카데믹 버블 + 용량 배율)
# ══════════════════════════════════════════════════════════════════════════════
ax_e = fig.add_subplot(gs[1, 2])
panel_label(ax_e, 'e', x=-0.14)
ax_e.set_title('Precision drug cocktail recommendation\n& patient-adjusted dose ratio', fontsize=9, pad=6)
ax_e.grid(False)
ax_e.set_facecolor('white')

drug_cat_col = {
    'Pritamab':    ('#7B2D8B','PrPc-RPSA Ab.'),
    'Trametinib':  ('#C62828','MEK inhibitor'),
    'Sotorasib':   ('#B71C1C','KRAS G12C inh.'),
    'Everolimus':  ('#1565C0','mTOR inhibitor'),
    'Buparlisib':  ('#1976D2','PI3K inhibitor'),
    'Bevacizumab': ('#00695C','anti-VEGF Ab.'),
    'Lenvatinib':  ('#00796B','VEGFR inhibitor'),
}

all_drugs_ord = []
for r in recs:
    for d in r.recommended_drugs:
        if d not in all_drugs_ord:
            all_drugs_ord.append(d)

n_drugs = len(all_drugs_ord)
n_cases = 4

# 배경 줄무늬
for yi in range(n_drugs):
    ax_e.axhspan(yi-0.5, yi+0.5, alpha=0.04 if yi%2==0 else 0,
                 color='#1565C0', zorder=0)

for ci, (rec, (lbl, *_, col)) in enumerate(zip(recs, CASES)):
    for rank, drug in enumerate(rec.recommended_drugs):
        yi = all_drugs_ord.index(drug)
        dose = rec.doses_relative.get(drug, 1.0)
        size = 350 * (1 - rank * 0.18)
        drug_col = drug_cat_col.get(drug, ('#888888',))[0]
        alpha = 0.95 - rank * 0.12

        ax_e.scatter(ci, yi, s=size, color=drug_col, alpha=alpha,
                     edgecolors='white', linewidths=1.0, zorder=3)
        ax_e.text(ci, yi, str(rank+1), ha='center', va='center',
                  fontsize=7.5, fontweight='bold', color='white', zorder=4)
        # 용량 배율
        ax_e.text(ci + 0.30, yi - 0.28, f'{dose:.2f}',
                  fontsize=6, color='#555', va='center', ha='left')

# y축: 약물 + 카테고리
ax_e.set_yticks(range(n_drugs))
ytick_labels = []
for d in all_drugs_ord:
    cat = drug_cat_col.get(d, ('#888',''))[1]
    ytick_labels.append(f'{d}\n({cat})')
ax_e.set_yticklabels(ytick_labels, fontsize=7)
for i, d in enumerate(all_drugs_ord):
    col_d = drug_cat_col.get(d, ('#888',))[0]
    ax_e.get_yticklabels()[i].set_color(col_d)
    ax_e.get_yticklabels()[i].set_fontweight('bold')

case_xlabels = ['G12D\nHigh-Prolif.','WT\nHypoxia','G12V\nInvasive','G12C\nEarly']
ax_e.set_xticks(range(n_cases))
ax_e.set_xticklabels(case_xlabels, fontsize=8)
ax_e.set_xlim(-0.65, n_cases + 0.2)
ax_e.set_ylim(-0.7, n_drugs - 0.3)
ax_e.tick_params(left=False)
for sp in ax_e.spines.values():
    sp.set_visible(False)

ax_e.text(0.02, -0.12,
          'Bubble: rank order  |  Number: priority  |  Value: dose ratio (×EC₅₀)',
          transform=ax_e.transAxes, fontsize=7, color='#666')

# ── 전체 제목 ──────────────────────────────────────────────────────────────────
fig.text(0.5, 0.965,
         'Patient-Specific Signaling Energy Landscape Enables Precision Anticancer Cocktail Design',
         ha='center', fontsize=12, fontweight='bold', color=NAT['black'])
fig.text(0.5, 0.950,
         r'ADDS Platform: Cellpose morphology + CT phenotyping $\rightarrow$'
         r' pathway $\Delta G^\ddagger$ inference (Boltzmann-inverse + Eyring) $\rightarrow$ Bliss synergy optimization',
         ha='center', fontsize=8, color='#555')

# ── 저장 ────────────────────────────────────────────────────────────────────────
out = r'f:\ADDS\docs\energy_landscape_academic.png'
plt.savefig(out, dpi=250, bbox_inches='tight', facecolor='white')
plt.close()
print(f"Saved: {out}")
