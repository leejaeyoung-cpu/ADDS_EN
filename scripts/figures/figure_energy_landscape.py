"""
figure_energy_landscape.py
--------------------------
Nature Communications 스타일 환자-특이적 에너지 지형 피규어
4-panel layout:
  A. 에너지 장벽 개념도 (반응 좌표 다이어그램)
  B. 4개 케이스 경로별 ΔG‡ heatmap
  C. k_pathway 레이더 차트 (4-케이스 오버레이)
  D. 약물 칵테일 추천 Sankey-style dot chart
"""

import sys, warnings
sys.path.insert(0, r'f:\ADDS')
warnings.filterwarnings('ignore')

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyArrowPatch
from matplotlib.colors import LinearSegmentedColormap
import math

from src.pritamab_ml.imaging_to_energy import ImagingToEnergyMapper, BASELINE_DDG, RT_KCAL
from src.pritamab_ml.pathway_drug_optimizer import PathwayDrugOptimizer

# ── 데이터 준비 ────────────────────────────────────────────────────────────────
mapper    = ImagingToEnergyMapper()
optimizer = PathwayDrugOptimizer(max_drugs=4)

CASES = {
    'A\nKRAS G12D\n고증식형': {
        'cellpose': [{'total_cells':520,'mean_area_um2':310,'mean_circularity':0.50,
                      'irregular_count':290,'normal_count':120}]*3,
        'ct':  {'tumor_volume_cc':45,'mean_hu':-8,'std_hu':50,'necrosis_ratio':0.20},
        'kras':'G12D', 'color':'#e63946'
    },
    'B\nKRAS WT\n저산소형': {
        'cellpose': [{'total_cells':80,'mean_area_um2':520,'mean_circularity':0.75,
                      'irregular_count':10,'normal_count':65}]*3,
        'ct':  {'tumor_volume_cc':12,'mean_hu':-35,'std_hu':70,'necrosis_ratio':0.45},
        'kras':'WT', 'color':'#2196f3'
    },
    'C\nKRAS G12V\n고침윤형': {
        'cellpose': [{'total_cells':200,'mean_area_um2':280,'mean_circularity':0.38,
                      'irregular_count':150,'normal_count':30}]*5,
        'ct':  {'tumor_volume_cc':28,'mean_hu':15,'std_hu':35,'necrosis_ratio':0.05},
        'kras':'G12V', 'color':'#ff9800'
    },
    'D\nKRAS G12C\n초기형': {
        'cellpose': [{'total_cells':60,'mean_area_um2':400,'mean_circularity':0.70,
                      'irregular_count':8,'normal_count':48}],
        'ct':  None,
        'kras':'G12C', 'color':'#4caf50'
    },
}

profiles = {}
recs     = {}
for label, c in CASES.items():
    p = mapper.compute_profile(c['cellpose'], c['ct'], c['kras'])
    profiles[label] = p
    recs[label]     = optimizer.optimize(p, kras_allele=c['kras'])

PATHWAYS   = ['KRAS_ERK','PI3K_mTOR','HIF_VEGF','RhoA_INV']
PW_LABELS  = ['KRAS→ERK','PI3K→mTOR','HIF→VEGF','RhoA→Inv']
CASE_KEYS  = list(CASES.keys())
CASE_COLS  = [CASES[k]['color'] for k in CASE_KEYS]

# ── 피규어 설정 ────────────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family':       'DejaVu Sans',
    'font.size':         9,
    'axes.spines.top':   False,
    'axes.spines.right': False,
    'figure.facecolor':  '#ffffff',
    'axes.facecolor':    '#fafafa',
    'axes.grid':         True,
    'grid.alpha':        0.3,
    'grid.linewidth':    0.5,
})

fig = plt.figure(figsize=(18, 13))
fig.patch.set_facecolor('#ffffff')

gs_main = gridspec.GridSpec(2, 2, figure=fig,
                             hspace=0.42, wspace=0.35,
                             left=0.06, right=0.97,
                             top=0.92, bottom=0.07)

# ══════════════════════════════════════════════════════════════════════════════
# PANEL A — 에너지 장벽 개념 다이어그램 (반응 좌표)
# ══════════════════════════════════════════════════════════════════════════════
ax_a = fig.add_subplot(gs_main[0, 0])
ax_a.set_title('A   Energy Barrier Landscape (Concept)', fontweight='bold', loc='left', fontsize=10, pad=8)
ax_a.set_facecolor('#f8f9fa')

x = np.linspace(0, 10, 500)

def reaction_curve(x, ddg, base_energy=0.3, width=2.8, center=5.0):
    return (base_energy
            + ddg * np.exp(-((x - center + width*0.4)**2) / (2*(width*0.45)**2))
            - (ddg*0.65) * np.exp(-((x - center - width*0.7)**2) / (2*(width*0.5)**2)))

# 기준선 (treatment naive)
baseline_ddg = 1.20
y_naive  = reaction_curve(x, baseline_ddg)
ax_a.plot(x, y_naive, '--', color='#888888', lw=1.5, label='Population avg ΔG‡=1.20', zorder=2)
ax_a.fill_between(x, 0, y_naive, alpha=0.06, color='#888888')

# KRAS G12D (낮은 장벽)
ddg_g12d = profiles[CASE_KEYS[0]].ddg_per_pathway['KRAS_ERK']
y_g12d   = reaction_curve(x, ddg_g12d)
ax_a.plot(x, y_g12d, '-', color='#e63946', lw=2.2, label=f'KRAS G12D  ΔG‡={ddg_g12d:.2f}', zorder=3)
ax_a.fill_between(x, 0, y_g12d, alpha=0.12, color='#e63946')

# WT 저산소 (중간 장벽)
ddg_wt = profiles[CASE_KEYS[1]].ddg_per_pathway['KRAS_ERK']
y_wt   = reaction_curve(x, ddg_wt)
ax_a.plot(x, y_wt, '-', color='#2196f3', lw=2.2, label=f'KRAS WT    ΔG‡={ddg_wt:.2f}', zorder=3)

# 장벽 높이 화살표 (G12D)
peak_x_idx = np.argmax(y_g12d)
peak_x = x[peak_x_idx]; peak_y = y_g12d[peak_x_idx]
ax_a.annotate('', xy=(peak_x, peak_y), xytext=(peak_x, 0.3),
              arrowprops=dict(arrowstyle='<->', color='#e63946', lw=1.5))
ax_a.text(peak_x + 0.15, (peak_y + 0.3)/2, f'ΔG‡\n={ddg_g12d:.2f}', fontsize=8,
          color='#e63946', va='center')

# Eyring k 표시
k_g12d  = round(np.exp(-ddg_g12d / RT_KCAL), 3)
k_naive = round(np.exp(-baseline_ddg / RT_KCAL), 3)
ax_a.text(8, 0.9, f'k(G12D) = {k_g12d}\nk(naive) = {k_naive}\nRatio = {k_g12d/k_naive:.1f}×',
          fontsize=8, color='#333', va='top',
          bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='#ddd', lw=0.8))

ax_a.set_xlabel('Reaction Coordinate (Signal Flux)', fontsize=9)
ax_a.set_ylabel('Free Energy (kcal/mol)', fontsize=9)
ax_a.set_xlim(0, 10); ax_a.set_ylim(-0.1, 1.5)
ax_a.axhline(0.3, color='#aaa', lw=0.7, ls=':')
ax_a.text(0.3, 0.27, 'Ground state', fontsize=7.5, color='#888')
ax_a.legend(fontsize=8, loc='upper right', framealpha=0.85, edgecolor='#ddd')

# ══════════════════════════════════════════════════════════════════════════════
# PANEL B — ΔG‡ Heatmap (경로 × 케이스)
# ══════════════════════════════════════════════════════════════════════════════
ax_b = fig.add_subplot(gs_main[0, 1])
ax_b.set_title('B   Pathway ΔG‡ per Patient Case (kcal/mol)', fontweight='bold', loc='left', fontsize=10, pad=8)
ax_b.set_facecolor('white')
ax_b.grid(False)

# 데이터 행렬
ddg_matrix = np.array([
    [profiles[k].ddg_per_pathway[pw] for k in CASE_KEYS]
    for pw in PATHWAYS
])  # shape: (4 pathways, 4 cases)

cmap = LinearSegmentedColormap.from_list('energy',
    ['#d32f2f', '#ff8f00', '#f9f9f9', '#1565c0'], N=256)

im = ax_b.imshow(ddg_matrix, cmap=cmap, aspect='auto',
                 vmin=0.85, vmax=1.25, interpolation='nearest')

# 셀 값 표시
for i in range(4):
    for j in range(4):
        val = ddg_matrix[i, j]
        ax_b.text(j, i, f'{val:.3f}', ha='center', va='center',
                  fontsize=9, fontweight='bold',
                  color='white' if val < 0.95 else '#222')

# 최솟값 경로 표시 (우선 표적)
for j, key in enumerate(CASE_KEYS):
    min_pw_idx = np.argmin([profiles[key].ddg_per_pathway[pw] for pw in PATHWAYS])
    rect = mpatches.FancyBboxPatch((j-0.5, min_pw_idx-0.5), 1, 1,
                                    boxstyle='round,pad=0.07', fill=False,
                                    edgecolor='gold', linewidth=2.5)
    ax_b.add_patch(rect)

# 축
case_short = ['G12D\n고증식', 'WT\n저산소', 'G12V\n고침윤', 'G12C\n초기']
ax_b.set_xticks(range(4)); ax_b.set_xticklabels(case_short, fontsize=8.5)
ax_b.set_yticks(range(4)); ax_b.set_yticklabels(PW_LABELS, fontsize=9)
ax_b.tick_params(left=False, bottom=False)

cbar = plt.colorbar(im, ax=ax_b, fraction=0.035, pad=0.02)
cbar.set_label('ΔG‡ (kcal/mol)\n← activated   baseline →', fontsize=8)
cbar.ax.tick_params(labelsize=8)

ax_b.text(0.5, -0.14, '★ Gold box = primary target pathway (lowest ΔG‡)',
          transform=ax_b.transAxes, ha='center', fontsize=8, color='#b8860b')

# ══════════════════════════════════════════════════════════════════════════════
# PANEL C — k_pathway 레이더 차트
# ══════════════════════════════════════════════════════════════════════════════
ax_c = fig.add_subplot(gs_main[1, 0], polar=True)
ax_c.set_title('C   Pathway Activity Radar (Eyring k)',
               fontweight='bold', loc='left', fontsize=10, pad=18)

N = 4
angles = [n / N * 2 * math.pi for n in range(N)]
angles += angles[:1]  # close

k_max_global = max(
    max(profiles[k].k_per_pathway.values()) for k in CASE_KEYS
)

for key, col in zip(CASE_KEYS, CASE_COLS):
    k_vals = [profiles[key].k_per_pathway[pw] / k_max_global for pw in PATHWAYS]
    k_vals += k_vals[:1]
    ax_c.plot(angles, k_vals, '-o', color=col, lw=2, markersize=5, label=key.replace('\n', ' | ').strip())
    ax_c.fill(angles, k_vals, alpha=0.08, color=col)

ax_c.set_xticks(angles[:-1])
ax_c.set_xticklabels(PW_LABELS, fontsize=9)
ax_c.set_ylim(0, 1)
ax_c.set_yticks([0.25, 0.50, 0.75, 1.0])
ax_c.set_yticklabels(['25%','50%','75%','100%'], fontsize=7, color='#888')
ax_c.grid(color='#dddddd', linewidth=0.8)
ax_c.spines['polar'].set_visible(False)

legend = ax_c.legend(fontsize=8, loc='lower left', bbox_to_anchor=(-0.12, -0.18),
                     ncol=2, framealpha=0.85, edgecolor='#ddd')

# ══════════════════════════════════════════════════════════════════════════════
# PANEL D — 약물 추천 요약 (Dot matrix / bubble chart)
# ══════════════════════════════════════════════════════════════════════════════
ax_d = fig.add_subplot(gs_main[1, 1])
ax_d.set_title('D   Recommended Drug Cocktail per Case', fontweight='bold', loc='left', fontsize=10, pad=8)
ax_d.set_facecolor('white')
ax_d.grid(False)

# 전체 약물 목록 수집
all_drugs_set = []
for k in CASE_KEYS:
    for d in recs[k].recommended_drugs:
        if d not in all_drugs_set:
            all_drugs_set.append(d)
all_drugs = sorted(all_drugs_set)  # y축

# 약물 카테고리 색
drug_cat = {
    'Pritamab':    '#7b2d8b',  # 보라 — ADDS 자체 항체
    'Trametinib':  '#c62828',  # 빨강 — MEK 억제제
    'Sotorasib':   '#e53935',  # 진빨강 — KRAS G12C
    'Everolimus':  '#1565c0',  # 파랑 — mTOR
    'Buparlisib':  '#1976d2',  # 파랑 — PI3K
    'Bevacizumab': '#00796b',  # 청록 — anti-VEGF
    'Lenvatinib':  '#0097a7',  # 청록 — VEGFR
    'Fasudil':     '#f57c00',  # 주황 — ROCK
    'Y-27632':     '#ff9800',  # 주황 — ROCK
}

case_x = {k: i for i, k in enumerate(CASE_KEYS)}
drug_y = {d: i for i, d in enumerate(all_drugs)}

for ci, key in enumerate(CASE_KEYS):
    rec = recs[key]
    n_drugs = len(rec.recommended_drugs)
    for rank, drug in enumerate(rec.recommended_drugs):
        size  = 320 * (1 - rank * 0.15)   # 1순위가 가장 큼
        alpha = 0.9 - rank * 0.12
        col   = drug_cat.get(drug, '#888888')
        ax_d.scatter(ci, drug_y[drug], s=size, color=col, alpha=alpha,
                     edgecolors='white', linewidths=1.2, zorder=3)
        # 순위 숫자
        ax_d.text(ci, drug_y[drug], str(rank+1), ha='center', va='center',
                  fontsize=8, fontweight='bold', color='white', zorder=4)
        # 용량 조정 표시 (선택)
        dose = rec.doses_relative.get(drug, 1.0)
        ax_d.text(ci + 0.33, drug_y[drug], f'{dose:.2f}×',
                  fontsize=6.5, color='#555', va='center')

# y축: 약물 이름 + 카테고리 색
ax_d.set_yticks(range(len(all_drugs)))
ax_d.set_yticklabels(all_drugs, fontsize=9)
for i, drug in enumerate(all_drugs):
    col = drug_cat.get(drug, '#888888')
    ax_d.get_yticklabels()[i].set_color(col)
    ax_d.get_yticklabels()[i].set_fontweight('bold')

# x축: 케이스 레이블
case_short_labels = ['G12D\n고증식','WT\n저산소','G12V\n고침윤','G12C\n초기']
ax_d.set_xticks(range(4))
ax_d.set_xticklabels(case_short_labels, fontsize=9)
ax_d.set_xlim(-0.6, 4.1)
ax_d.set_ylim(-0.7, len(all_drugs)-0.3)
ax_d.tick_params(left=False)
for sp in ax_d.spines.values():
    sp.set_visible(False)

# 가이드 라인
for yi in range(len(all_drugs)):
    ax_d.axhline(yi, color='#eeeeee', lw=0.8, zorder=1)

# 범례
legend_patches = [
    mpatches.Patch(color='#7b2d8b', label='Pritamab (PrPc)'),
    mpatches.Patch(color='#c62828', label='MEK/KRAS inh.'),
    mpatches.Patch(color='#1565c0', label='mTOR/PI3K inh.'),
    mpatches.Patch(color='#00796b', label='Anti-VEGF'),
    mpatches.Patch(color='#f57c00', label='ROCK inh.'),
]
ax_d.legend(handles=legend_patches, fontsize=7.5, loc='lower right',
            bbox_to_anchor=(1.01, -0.02), framealpha=0.85, edgecolor='#ddd')
ax_d.text(0.02, -0.07, 'Bubble size ∝ priority rank  |  Number = rank  |  ×value = dose ratio',
          transform=ax_d.transAxes, fontsize=7.5, color='#888')

# ── 전체 제목 ──────────────────────────────────────────────────────────────
fig.text(0.5, 0.965,
         'Patient-Specific Signaling Energy Landscape → Precision Drug Cocktail Optimization',
         ha='center', va='top', fontsize=13, fontweight='bold', color='#1a2f5e')
fig.text(0.5, 0.945,
         'ADDS Platform | Cellpose + CT Imaging → ΔG‡ Inference (Boltzmann-Inverse + Eyring) → Bliss Synergy Optimization',
         ha='center', va='top', fontsize=8.5, color='#555')

# ── 저장 ──────────────────────────────────────────────────────────────────────
out_path = r'f:\ADDS\docs\energy_landscape_figure.png'
plt.savefig(out_path, dpi=200, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.close()
print(f"Saved: {out_path}")
