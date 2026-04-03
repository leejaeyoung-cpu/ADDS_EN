"""
Pritamab 전체 종합 보고서
Complete Comprehensive Report — Pritamab PK/PD, Mechanism, Synergy, Evidence Base
===========================================================================================
데이터 검증 등급:
  ★ = 논문 원문 직접 확인 (Pritamab_NatureComm_Paper.txt)
  ◆ = ADDS 시스템 계산 (paper3_pritamab_kras.py)
  ● = 문헌 근거 확인 (SCI 논문)
  ▲ = ADDS 에너지 모델 투영값
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Wedge, Circle
import matplotlib.patheffects as pe
from matplotlib.lines import Line2D
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# ════════════════════════════════════════════════════════
# 색상 팔레트
# ════════════════════════════════════════════════════════
BG     = "#070A14"
PANEL  = "#0F1726"
PANEL2 = "#141E30"
WHITE  = "#EEF2FF"
GRAY   = "#8B9CC4"
BLUE   = "#60A5FA"
GREEN  = "#34D399"
RED    = "#F87171"
AMBER  = "#FBBF24"
PURPLE = "#A78BFA"
CYAN   = "#67E8F9"
TEAL   = "#2DD4BF"
ORANGE = "#FB923C"
NAVY   = "#1E3A8A"
DARK   = "#1F2937"

def panel_bg(ax, color=PANEL):
    ax.set_facecolor(color)
    for spine in ax.spines.values():
        spine.set_visible(False)

def ptitle(ax, text, color=WHITE, fs=11):
    ax.set_title(text, fontsize=fs, fontweight='bold', color=color, pad=8)

# ════════════════════════════════════════════════════════
# 검증된 데이터 상수 (모두 ★ 논문 원문 확인)
# ════════════════════════════════════════════════════════
SIGNAL_DATA = {  # ★ 논문 Line 219-223
    'RAS-GTP loading':  -42,
    'ERK1/2 (pERK)':    -38,
    'AKT (pAKT S473)':  -31,
    'Notch1-NICD':      -55,
    'Cleaved Caspase-3': +280,  # +2.8-fold = +180%
}
ENERGY_NODES = {  # ★ 논문 Line 256-262
    'Survival init.':      (3.0, 0.30, 0.80),
    'Prolif. gate':        (2.5, 1.25, 1.50),
    'Resistance peak':     (2.0, 1.70, 1.80),
    'Apoptosis entry':     (1.5, 1.25, 1.30),
    'Apoptotic commit.':   (1.0, 0.88, 0.90),
}
EC50_DATA = {  # ★ 논문 Line 283-286
    '5-FU':        (12000, 9032),
    'Oxaliplatin': (3750,  2823),
    'Irinotecan':  (7500,  5645),
    'Sotorasib':   (75,    56.5),
}
SYNERGY = {  # ★ 논문 Line 305-310
    'Bliss_5FU':    18.4,
    'Bliss_Oxali':  21.7,
    'Loewe_DRI':    1.34,
    'ADDS_5FU':     0.87,
    'ADDS_Oxali':   0.89,
    'ADDS_Soto':    0.82,
    'ADDS_FOLFOX':  0.84,
}
PK = {  # ★ 논문 Line 394-407
    'CL': 0.18, 'Vd': 4.3, 't12': '21-25d',
    'IC50': 12.3, 'Cmin': 50, 'KD': 0.84,
    'dose': '10-15 mg/kg Q3W',
}
ENERGY_PARAMS = {  # ★ 논문 + ◆ ADDS
    'ddG_RLS': 0.50, 'ddG_EC50': 0.175,
    'alpha': 0.35, 'RT': 0.593,
    'rate_reduction': 55.6, 'EC50_reduction': 24.7,
}
BLISS_ADDS_ALL = {  # ★/◆ 혼합 — Irinotecan/FOLFOX/FOLFIRI/TAS-102는 모두 ADDS DL 추정
    'Pritamab\n+5-FU':      (18.4, 0.87, '★'),   # 논문 원문
    'Pritamab\n+Oxali':     (21.7, 0.89, '★'),   # 논문 원문
    'Pritamab\n+Irino':     (17.3, 0.84, '◆'),   # ❌수정: ADDS DL 추정 (이전에 ●로 잘못 표기)
    'Pritamab\n+Sotorasib': (15.8, 0.82, '◆'),   # ❌2차수정: ADDS consensus 0.82는 ★(L375), Bliss 15.8은 논문에 없음→◆
    'Pritamab\n+FOLFOX':    (20.5, 0.84, '◆'),   # ADDS 추론
    'Pritamab\n+FOLFIRI':   (18.8, 0.87, '◆'),   # ADDS 추론
    'Pritamab\n+TAS-102':   (18.1, 0.87, '◆'),   # ❌수정: ADDS 추론 (이전에 ●로 잘못 표기)
}
CS_SCORE = {  # ◆+▲ ADDS 계산 + Apoptosis 투영값 기반 — 이중 추론값 (임상 미검증)
    # CS = 0.35×DRS + 0.25×(Bliss/25) + 0.20×ADDS + 0.20×(Apo/100)
    'FOLFOX':   (0.893, 21.0, 0.84, 85, 0.889),  # Apo 85% = ADDS 투영
    'FOLFIRI':  (0.870, 19.8, 0.87, 82, 0.867),  # Apo 82% = ADDS 투영
    'TAS-102':  (0.880, 18.1, 0.87, 80, 0.858),  # Apo 80% = Oncotarget 범위
    'Oxali':    (0.850, 21.7, 0.89, 75, 0.848),  # Apo 75% = ADDS 투영
    '5-FU':     (0.820, 18.4, 0.87, 70, 0.830),  # Apo 70% = ADDS 투영
    'Sotorasib':(0.780, 15.8, 0.82, 68, 0.796),  # Apo 68% = ADDS 투영
    'Irinotecan':(0.760,17.3, 0.84, 75, 0.810),  # Apo 75% = ADDS 투영
}
DG_LIT = {  # ● 문헌 근거값
    'SN-38\n(Irino AM)\nTopo-I': (-13.0, -10.7, -14.0, BLUE),
    'Oxali\nDNA\n(QM/MM)':        (-14.0, -13.0, -14.5, RED),
    'FTD\n(TAS-102)\nDNA':        (-14.3, -12.0, -15.0, PURPLE),
    'FdUMP\n(5-FU AM)\nTS':       (-11.2, -10.0, -13.0, GREEN),
}
APO_LIT = {  # ● 문헌 + ▲ ADDS
    'KRAS-mut\nbaseline': (22, 15, 30, GRAY, '●'),
    'PrPc siRNA\n(CRC)': (40, 25, 50, BLUE, '●'),
    'Pritamab\nalone': (55, 50, 60, CYAN, '▲+●'),
    '+Irinotecan': (75, 60, 75, GREEN, '▲'),
    '+Oxaliplatin': (75, 60, 75, TEAL, '▲'),
    '+TAS-102': (80, 75, 82, PURPLE, '▲+●'),
    'FOLFOX\n+Bev': (70, 60, 75, ORANGE, '●'),
}

# ════════════════════════════════════════════════════════
# FIGURE 생성 (대형 A2급)
# ════════════════════════════════════════════════════════
fig = plt.figure(figsize=(36, 28), facecolor=BG)

# ── 메인 타이틀 ──────────────────────────────────────────
fig.text(0.5, 0.982,
         "PRITAMAB  ·  Comprehensive Research Report",
         ha='center', va='top', fontsize=28, fontweight='bold',
         color=WHITE, fontfamily='DejaVu Sans')
fig.text(0.5, 0.968,
         "PrPc-Targeting Anti-Cancer Antibody  |  Mechanism · PK/PD · Synergy · Evidence Base · Clinical Development",
         ha='center', va='top', fontsize=13, color=GRAY)

# 검증 등급 범례
legend_str = ("★ NatureComm Paper Direct   ◆ ADDS System Calculated   "
              "● SCI Literature Confirmed   ▲ ADDS Energy Model Projection")
fig.text(0.5, 0.956, legend_str,
         ha='center', va='top', fontsize=10, color=AMBER,
         bbox=dict(boxstyle='round,pad=0.4', facecolor='#1A1F35',
                   edgecolor=AMBER, alpha=0.9, linewidth=1))

# ──────────────────────────────────────────────────────────
# 메인 그리드: 4행 × 4열
# Row 1: 메커니즘 | 에너지 장벽 | 신호 차단 | PK 파라미터
# Row 2: EC50 | Bliss/ADDS 히트맵 | CS 랭킹 | Apoptosis%
# Row 3: ΔG 문헌 | Apo% 문헌 | PK 시뮬 | 독성 요약
# Row 4: 환자선택 | 임상 로드맵 | 검증 요약표
# ──────────────────────────────────────────────────────────
gs = gridspec.GridSpec(
    4, 4, figure=fig,
    left=0.03, right=0.985,
    top=0.945, bottom=0.03,
    hspace=0.42, wspace=0.30,
)

# ════════════════════════ ROW 1 ═══════════════════════════

# ── [R1C0] 메커니즘 다이어그램 ─────────────────────────
ax_mech = fig.add_subplot(gs[0, 0])
panel_bg(ax_mech)
ax_mech.set_xlim(0, 10)
ax_mech.set_ylim(0, 10)
ax_mech.axis('off')
ptitle(ax_mech, "① PrPc-RPSA Signalosome Mechanism  ★", color=CYAN)

mech_nodes = [
    (5, 9.0, "Surface PrPc\n(Octarepeat 51-90)", BLUE, 0.7),
    (5, 7.3, "RPSA / 37LRP", RED, 0.6),
    (5, 5.7, "SRC/FYN Kinase", AMBER, 0.5),
    (2.5, 3.8, "RAS-GTP\nLoading ↑", RED, 0.55),
    (5, 3.8, "KRAS G12D/V/C\nConstitutive Act.", ORANGE, 0.55),
    (7.5, 3.8, "Filamin A\nEMT/Invasion", PURPLE, 0.5),
    (2.5, 1.8, "RAF-MEK\nERK ↑", RED, 0.45),
    (5, 1.8, "PI3K-AKT\nSurvival ↑", ORANGE, 0.45),
    (7.5, 1.8, "Notch1-NICD\nCSC ↑", PURPLE, 0.45),
]
for (x, y, txt, c, s) in mech_nodes:
    circle = Circle((x, y), s, facecolor=c, alpha=0.22, edgecolor=c, linewidth=1.5,
                    transform=ax_mech.transData, clip_on=False)
    ax_mech.add_patch(circle)
    ax_mech.text(x, y, txt, ha='center', va='center', fontsize=7,
                 color=WHITE, fontweight='bold',
                 path_effects=[pe.withStroke(linewidth=1, foreground=BG)])

# 화살표 그리기
arrows = [(5,8.3,5,7.9),(5,6.7,5,6.3),(5,5.2,2.5,4.35),(5,5.2,5,4.35),(5,5.2,7.5,4.35),
          (2.5,3.25,2.5,2.25),(5,3.25,5,2.25),(7.5,3.25,7.5,2.25)]
# SOS1/2 중간체 (논문 Line443 명시 — 이전 다이어그램에서 누락)
ax_mech.text(2.5, 6.1, 'SOS1/2\n(RAS-GEF)\n★L443', ha='center', va='center', fontsize=6,
             color=AMBER, style='italic',
             path_effects=[pe.withStroke(linewidth=0.8, foreground=BG)])
for (x1,y1,x2,y2) in arrows:
    ax_mech.annotate('', xy=(x2,y2), xytext=(x1,y1),
                     arrowprops=dict(arrowstyle='->', color=GRAY, lw=1.2))

# Pritamab 차단 표시
ax_mech.text(5, 8.65, '[X] PRITAMAB\nIC₅₀=12.3nM ★',
             ha='center', va='center', fontsize=8, fontweight='bold',
             color=GREEN,
             bbox=dict(boxstyle='round,pad=0.3', facecolor='#064E3B',
                       edgecolor=GREEN, linewidth=1.5))
ax_mech.text(0.5, 9.2, 'KD = 0.84 nM ★', fontsize=7.5,
             color=AMBER, fontweight='bold')

# ── [R1C1] 에너지 장벽 프로파일 ─────────────────────────
ax_en = fig.add_subplot(gs[0, 1])
panel_bg(ax_en)
ptitle(ax_en, "② Energy Barrier Profile  ★", color=AMBER)

nodes = list(ENERGY_NODES.keys())
vals_norm = [v[0] for v in ENERGY_NODES.values()]
vals_mut  = [v[1] for v in ENERGY_NODES.values()]
vals_prit = [v[2] for v in ENERGY_NODES.values()]
x_n = range(len(nodes))

ax_en.plot(x_n, vals_norm, 'o-', color=GREEN, lw=2, markersize=6, label='Normal (WT)')
ax_en.plot(x_n, vals_mut,  's--', color=RED, lw=2, markersize=6, label='KRAS-mut+PrPc↑')
ax_en.plot(x_n, vals_prit, 'D-', color=BLUE, lw=2.5, markersize=6, label='+ Pritamab')
ax_en.fill_between(x_n, vals_mut, vals_prit, alpha=0.15, color=BLUE)

ax_en.set_xticks(list(x_n))
ax_en.set_xticklabels([n.replace(' ', '\n') for n in nodes], fontsize=7, color=WHITE)
ax_en.set_ylabel('Energy Barrier (rel. units)', color=GRAY, fontsize=9)
ax_en.tick_params(colors=GRAY, labelsize=8)
ax_en.grid(alpha=0.12, color=GRAY)
ax_en.legend(fontsize=8, facecolor=DARK, edgecolor=GRAY, labelcolor=WHITE, loc='upper right')
ax_en.text(0.05, 0.85, 'Rate reduction:\n55.6% ★\n(Arrhenius)', transform=ax_en.transAxes,
           fontsize=8, color=BLUE, fontweight='bold',
           bbox=dict(boxstyle='round,pad=0.3', facecolor=PANEL2, edgecolor=BLUE))
ax_en.annotate('10× collapse ★', xy=(0, 0.30), fontsize=7.5, color=RED,
               xytext=(0.5, 0.15),
               arrowprops=dict(arrowstyle='->', color=RED, lw=1))

# ── [R1C2] 신호 차단 효과 ─────────────────────────────────
ax_sig = fig.add_subplot(gs[0, 2])
panel_bg(ax_sig)
ptitle(ax_sig, "③ Signalling Inhibition (10nM, 24h)  ★", color=GREEN)

markers = list(SIGNAL_DATA.keys())
vals_raw = list(SIGNAL_DATA.values())
# CC-3는 +280%이지만 다른 스케일 — 별도 표시
vals_sig = vals_raw[:4]
labels_sig = markers[:4]

colors_s = [RED if v < 0 else GREEN for v in vals_sig]
bars_s = ax_sig.barh(labels_sig, vals_sig, color=colors_s, alpha=0.82,
                     edgecolor=BG, linewidth=0.5)
for bar, v in zip(bars_s, vals_sig):
    ax_sig.text(v - 2 if v < 0 else v + 1, bar.get_y() + bar.get_height()/2,
                f'{v}%', ha='right' if v < 0 else 'left', va='center',
                fontsize=9, color=WHITE, fontweight='bold')

ax_sig.axvline(0, color=GRAY, lw=1)
ax_sig.set_xlim(-70, 20)
ax_sig.tick_params(colors=GRAY, labelsize=8.5)
ax_sig.set_xlabel('Change vs Control (%)', color=GRAY, fontsize=9)
ax_sig.grid(axis='x', alpha=0.12, color=GRAY)

# CC-3 별도
ax_sig.text(0.98, 0.08,
            'Cleaved\nCaspase-3:\n+2.8-fold ★',
            ha='right', va='bottom', transform=ax_sig.transAxes,
            fontsize=9, color=CYAN, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#0C3547', edgecolor=CYAN))

# p값
pvals = ['<0.001', '0.001', '0.004', '<0.001']
for i, pv in enumerate(pvals):
    ax_sig.text(2, i, f'p={pv}', va='center', fontsize=7, color=GRAY)

# ── [R1C3] PK 파라미터 ───────────────────────────────────
ax_pk = fig.add_subplot(gs[0, 3])
panel_bg(ax_pk)
ptitle(ax_pk, "④ PK/PD Parameters  ★", color=PURPLE)
ax_pk.axis('off')

pk_rows = [
    ("KD (SPR)",        "0.84 nM",          "★"),
    ("IC₅₀ (RPSA inh)","12.3 nM",          "★"),
    ("IC₅₀ (cytotox)", ">500 nM",           "★"),
    ("Clearance",       "0.18 L/day",       "★"),
    ("Volume (Vd)",     "4.3 L",            "★"),
    ("t½ (terminal)",   "21-25 days",       "★"),
    ("Cmin target",     "≥50 nM",           "★"),
    ("EC50 Reduction",  "−24.7%",           "★"),
    ("Rate Reduction",  "−55.6%",          "★◆"),  # ★논문L252 + ◆Arrhenius계산
    ("ddG_RLS",         "0.50 kcal/mol",    "★"),
    ("α coupling",      "0.35",             "★"),
    ("ADCC fold",       "10-15× WT IgG1",   "★"),
    ("Dose",            "10-15 mg/kg Q3W",  "★"),
    ("Accum. ratio",    "1.4-1.6×",         "★"),
]
y0 = 0.97
for param, val, grade in pk_rows:
    gc = GREEN if grade == '★' else BLUE
    ax_pk.text(0.02, y0, f"{param}:", fontsize=8, color=GRAY,
               transform=ax_pk.transAxes, va='top')
    ax_pk.text(0.55, y0, val, fontsize=8, color=WHITE, fontweight='bold',
               transform=ax_pk.transAxes, va='top')
    ax_pk.text(0.92, y0, grade, fontsize=8, color=gc, fontweight='bold',
               transform=ax_pk.transAxes, va='top')
    y0 -= 0.068
    ax_pk.plot([0.02, 0.98], [y0 + 0.005, y0 + 0.005],
               color=PANEL2, lw=0.5, transform=ax_pk.transAxes, clip_on=False)

# ════════════════════════ ROW 2 ═══════════════════════════

# ── [R2C0] EC50 감작 ──────────────────────────────────────
ax_ec = fig.add_subplot(gs[1, 0])
panel_bg(ax_ec)
ptitle(ax_ec, "⑤ EC50 Sensitisation (−24.7%)  ★", color=AMBER)

drugs = list(EC50_DATA.keys())
alone = [v[0] for v in EC50_DATA.values()]
combo = [v[1] for v in EC50_DATA.values()]
x_ec = np.arange(len(drugs))
w = 0.35

b1 = ax_ec.bar(x_ec - w/2, alone, w, color=RED, alpha=0.8, label='Alone', edgecolor=BG)
b2 = ax_ec.bar(x_ec + w/2, combo, w, color=BLUE, alpha=0.8, label='+Pritamab', edgecolor=BG)
ax_ec.set_yscale('log')
ax_ec.set_xticks(x_ec)
ax_ec.set_xticklabels(drugs, color=WHITE, fontsize=9)
ax_ec.set_ylabel('EC50 (nM) — log scale', color=GRAY, fontsize=9)
ax_ec.tick_params(colors=GRAY, labelsize=8)
ax_ec.grid(axis='y', alpha=0.12, color=GRAY)
ax_ec.legend(fontsize=8.5, facecolor=DARK, edgecolor=GRAY, labelcolor=WHITE)
for bar in b2:
    ax_ec.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.2,
               '−24.7%', ha='center', va='bottom', fontsize=7.5,
               color=AMBER, fontweight='bold')

# ── [R2C1] 시너지 히트맵 ───────────────────────────────────
ax_syn = fig.add_subplot(gs[1, 1])
panel_bg(ax_syn)
ptitle(ax_syn, "⑥ 4-Model Synergy Heat Map  ★/◆/●", color=GREEN)

combos = list(BLISS_ADDS_ALL.keys())
bliss_v  = [BLISS_ADDS_ALL[c][0] for c in combos]
adds_v   = [BLISS_ADDS_ALL[c][1] for c in combos]
grades_s = [BLISS_ADDS_ALL[c][2] for c in combos]
# combos 순서: 5-FU, Oxali, Irino, Sotorasib, FOLFOX, FOLFIRI, TAS-102
# CS_SCORE: FOLFOX=0.893, FOLFIRI=0.870, TAS=0.880, Oxali=0.850, 5-FU=0.820, Soto=0.780, Irino=0.760
drs_v    = [0.820, 0.850, 0.760, 0.780, 0.893, 0.870, 0.880]  # ❌2차수정: combos 순서에 맞게 재정렬
apo_v    = [85, 82, 75, 68, 80, 75, 80]

metrics = ['Bliss\n(0-25)', 'ADDS\nConsensus', 'DRS\nScore', 'Apoptosis\n(÷100)']
data_heat = np.array([
    [b/25 for b in bliss_v],
    adds_v,
    drs_v,
    [a/100 for a in apo_v],
])

import matplotlib.colors as mcolors
cmap = mcolors.LinearSegmentedColormap.from_list(
    'synergy', ['#1E3A5F','#1E6B8A','#26A69A','#66BB6A','#FFD54F','#EF5350'])
im = ax_syn.imshow(data_heat, cmap=cmap, aspect='auto', vmin=0.5, vmax=1.0)
ax_syn.set_xticks(range(len(combos)))
ax_syn.set_xticklabels(combos, fontsize=7, color=WHITE, rotation=0)
ax_syn.set_yticks(range(4))
ax_syn.set_yticklabels(metrics, fontsize=8.5, color=WHITE)
ax_syn.tick_params(colors=GRAY, labelsize=8)

for i in range(4):
    for j in range(len(combos)):
        ax_syn.text(j, i, f'{data_heat[i,j]:.2f}',
                    ha='center', va='center', fontsize=7.5,
                    color='white' if data_heat[i,j] < 0.85 else BG,
                    fontweight='bold')
for j, g in enumerate(grades_s):
    ax_syn.text(j, -0.7, g, ha='center', va='center', fontsize=8, color=AMBER)

plt.colorbar(im, ax=ax_syn, shrink=0.7, pad=0.02).ax.tick_params(colors=GRAY, labelsize=8)

# ── [R2C2] CS 종합 점수 랭킹 ─────────────────────────────
ax_cs = fig.add_subplot(gs[1, 2])
panel_bg(ax_cs)
ptitle(ax_cs, "⑦ Comprehensive Score Ranking  ◆+▲", color=BLUE)  # 수정: ◆→◆+▲
ax_cs.text(0.5, -0.12,
           'CS = 0.35×DRS + 0.25×(Bliss/25) + 0.20×ADDS + 0.20×(Apo/100)\n'
           '⚠ ADDS 계산+Apoptosis 투영값 이중 추론 — 임상 미검증',
           ha='center', transform=ax_cs.transAxes, fontsize=7, color=AMBER, style='italic')

cs_names = list(CS_SCORE.keys())
cs_vals = [CS_SCORE[c][4] for c in cs_names]
cs_sorted = sorted(zip(cs_vals, cs_names), reverse=True)
cs_v_s, cs_n_s = zip(*cs_sorted)

rank_colors = [AMBER, WHITE, TEAL, BLUE, GREEN, PURPLE, GRAY]
bars_cs = ax_cs.barh(range(len(cs_n_s)), cs_v_s,
                     color=rank_colors[:len(cs_n_s)], alpha=0.85, edgecolor=BG)
ax_cs.set_yticks(range(len(cs_n_s)))
ax_cs.set_yticklabels([f'Pritamab+{n}' for n in cs_n_s], fontsize=8, color=WHITE)
ax_cs.set_xlim(0.70, 0.93)
ax_cs.axvline(0.86, color=RED, lw=1.5, linestyle='--', alpha=0.7)
ax_cs.text(0.863, -0.5, 'CS≥0.86\nTop Tier', fontsize=7.5, color=RED, va='top')
ax_cs.tick_params(colors=GRAY, labelsize=8)
ax_cs.grid(axis='x', alpha=0.12, color=GRAY)
ax_cs.set_xlabel('Comprehensive Score', color=GRAY, fontsize=9)
ranks = ['#1 ★', '#2  ', '#3  ', '#4  ', '#5  ', '#6  ', '#7  ']
for i, (bar, v) in enumerate(zip(bars_cs, cs_v_s)):
    ax_cs.text(-0.005, i, ranks[i], ha='right', va='center', fontsize=8.5, color=AMBER)
    ax_cs.text(v + 0.002, i, f'{v:.3f}', ha='left', va='center', fontsize=8, color=WHITE)

# ── [R2C3] Apoptosis% 비교 ────────────────────────────────
ax_apo = fig.add_subplot(gs[1, 3])
panel_bg(ax_apo)
ptitle(ax_apo, "⑧ Apoptosis Efficiency (%)  ▲/●", color=PURPLE)

combos_apo = ['Baseline\n(KRAS-mut)', 'PrPc\nsiRNA', 'Pritamab\nalone',
              '+Irino', '+Oxali', '+TAS-102',
              '+FOLFOX*', '+FOLFIRI*', '+FOLFOXIRI*']  # * = 미검증 추론
apo_pct = [22, 40, 55, 75, 75, 80, 85, 82, 88]
grades_a = ['●', '●', '▲+●', '▲', '▲', '▲+●',
            '▲(!)', '▲(!)', '▲(!)']  # (!) = 논문·SCI 근거 없음, 순수 추론
cmap_apo = plt.cm.get_cmap('YlOrRd')
colors_apo = [cmap_apo(v/100) for v in apo_pct]

bars_apo = ax_apo.barh(combos_apo, apo_pct, color=colors_apo, alpha=0.85, edgecolor=BG)
ax_apo.axvline(55, color=CYAN, lw=1.5, linestyle=':', alpha=0.7, label='Pritamab alone 55%')
ax_apo.axvline(25, color=GRAY, lw=1.2, linestyle='--', alpha=0.5, label='Baseline ~25%')
ax_apo.set_xlim(0, 110)
ax_apo.tick_params(colors=GRAY, labelsize=8)
ax_apo.set_xlabel('Apoptosis Rate (%)', color=GRAY, fontsize=9)
ax_apo.grid(axis='x', alpha=0.12, color=GRAY)
ax_apo.legend(fontsize=7.5, facecolor=DARK, edgecolor=GRAY, labelcolor=WHITE, loc='lower right')
for bar, v, g in zip(bars_apo, apo_pct, grades_a):
    ax_apo.text(v + 1, bar.get_y() + bar.get_height()/2,
                f'{v}% {g}', ha='left', va='center', fontsize=7.5, color=WHITE)
# 경고 박스: FOLFOX/FOLFIRI/FOLFOXIRI는 미검증
ax_apo.text(0.01, 0.06,
            '* FOLFOX/FOLFIRI/FOLFOXIRI Apo%:\nADDS 모델 추론 — SCI 문헌 직접 근거 없음\n임상 검증 전 미사용 권고',
            transform=ax_apo.transAxes, fontsize=6.5, color=AMBER,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#2A1A00', edgecolor=AMBER, alpha=0.9))

# ════════════════════════ ROW 3 ═══════════════════════════

# ── [R3C0] ΔG_bind 문헌 근거 ──────────────────────────────
ax_dg = fig.add_subplot(gs[2, 0])
panel_bg(ax_dg)
ptitle(ax_dg, "⑨ ΔG_bind Literature Evidence  ●", color=TEAL)

dg_labels_p = ['SN-38\nTopo-I-DNA',
                'Edotecarin\nTopo-I-DNA',
                'Naphtho.\nTopo-I',
                'Oxali\nDNA(QM/MM)',
                'FTD\nDNA(cov.)',
                '5-FU\nTS(native)\n[≠ADDS−11.2]',   # 수정: ΔG_bind 차이 명시
                'FdUMP\nTS(active)\n[≈ADDS−11.2]',   # 수정: ADDS 기준은 FdUMP
                'ADDS:\nIrino−13.0',
                'ADDS:\nOxali−14.0',
                'ADDS:\nFTD−14.3',
                'ADDS: 5-FU\n(FdUMP AM)\n−11.2']
dg_vals_p = [-12.0, -10.7, -11.94, -14.0, -13.5, -3.44, -11.5,
             -13.0, -14.0, -14.3, -11.2]
dg_colors_p = [BLUE]*3 + [RED]*2 + [GREEN]*2 + \
              [CYAN, CYAN, PURPLE, TEAL]

bars_dg = ax_dg.barh(dg_labels_p, dg_vals_p, color=dg_colors_p, alpha=0.8, edgecolor=BG)
ax_dg.axvline(-10.7, color=GRAY, lw=0.8, linestyle=':', alpha=0.5)

# ADDS 값 참조선
for v, c, lbl in [(-13.0, CYAN, ''), (-14.0, CYAN, ''), (-14.3, PURPLE, ''), (-11.2, TEAL, '')]:
    ax_dg.axvline(v, color=c, lw=1.5, linestyle='--', alpha=0.6)

for bar, v in zip(bars_dg, dg_vals_p):
    ax_dg.text(v - 0.4, bar.get_y() + bar.get_height()/2,
               f'{v}', ha='right', va='center', fontsize=7.5, color=WHITE)

ax_dg.set_xlim(-18, 0)
ax_dg.tick_params(colors=GRAY, labelsize=7.5)
ax_dg.set_xlabel('ΔG_bind (kcal/mol)', color=GRAY, fontsize=9)
ax_dg.grid(axis='x', alpha=0.12, color=GRAY)
# 5-FU 주의 표시
ax_dg.text(-3.44, 5.5, '← 5-FU native\n  (≠FdUMP AM)', fontsize=6.5, color=AMBER,
           ha='left', va='center')

# 범례
from matplotlib.patches import Patch
leg_dg = [Patch(fc=BLUE, label='Topo-I inhibitors'),
           Patch(fc=RED,  label='Oxaliplatin-DNA'),
           Patch(fc=GREEN,label='5-FU / TS'),
           Patch(fc=PURPLE,label='TAS-102 (FTD)'),
           Patch(fc=CYAN, label='ADDS inferred')]
ax_dg.legend(handles=leg_dg, fontsize=7, facecolor=DARK, edgecolor=GRAY,
             labelcolor=WHITE, loc='lower right')

# ── [R3C1] Apoptosis% 문헌 상세 ──────────────────────────
ax_apo2 = fig.add_subplot(gs[2, 1])
panel_bg(ax_apo2)
ptitle(ax_apo2, "⑩ Apoptosis Evidence (Literature)  ●", color=GREEN)

apo_cats  = ['KRAS-mut\nbaseline (flow)', 'PrPc siRNA\nHCT116/SW480',
             'FOLFOX+Bev\nHCT116 (PMID)', 'Irino+metformin\nSW480 (PMID)',
             '5-FU+diosmetin\nHCT116 (PMID)', 'TAS-102+target\n(Oncotarget 2021)',
             'ADDS Pritamab\nalone (▲)']
apo_mean2 = [22, 40, 70, 68, 45, 78, 55]
apo_lo2   = [15, 25, 60, 60, 45, 75, 50]
apo_hi2   = [30, 50, 75, 75, 45, 82, 60]
apo_err2  = [[m-l for m,l in zip(apo_mean2,apo_lo2)],
             [h-m for m,h in zip(apo_mean2,apo_hi2)]]
apo_c2 = [GRAY, BLUE, ORANGE, ORANGE, GREEN, PURPLE, CYAN]

ax_apo2.barh(apo_cats, apo_mean2, xerr=apo_err2, color=apo_c2, alpha=0.80,
             edgecolor=BG, error_kw=dict(ecolor=GRAY, lw=1.2))
ax_apo2.axvline(55, color=CYAN, lw=2, linestyle='--', alpha=0.8, label='ADDS 55%')
ax_apo2.set_xlim(0, 95)
ax_apo2.tick_params(colors=GRAY, labelsize=7.5)
ax_apo2.set_xlabel('Apoptosis Rate (%)', color=GRAY, fontsize=9)
ax_apo2.grid(axis='x', alpha=0.12, color=GRAY)
ax_apo2.legend(fontsize=8, facecolor=DARK, edgecolor=GRAY, labelcolor=WHITE)

# ── [R3C2] PK 시뮬레이션 ──────────────────────────────────
ax_pksim = fig.add_subplot(gs[2, 2])
panel_bg(ax_pksim)
ptitle(ax_pksim, "⑪ PK Simulation (10 mg/kg Q3W)  ★", color=AMBER)

CL = 0.18; Vd = 4.3; t = np.linspace(0, 100, 1000)
dose_mg = 70 * 10  # 70kg × 10 mg/kg
MW_IgG1 = 148000  # MW 가정: 표준 IgG1 ~148kDa (논문에 명시 없음 — 참고용)
C0 = dose_mg / Vd * 1e6 / MW_IgG1 * 1e9  # 개략 nM (MW=148kDa 가정)
k_el = CL / Vd
conc = np.zeros_like(t)
for dose_time in [0, 21, 42, 63]:
    mask = t >= dose_time
    conc[mask] += C0 * np.exp(-k_el * (t[mask] - dose_time))

ax_pksim.plot(t, conc, color=BLUE, lw=2.5)
ax_pksim.fill_between(t, conc, alpha=0.12, color=BLUE)
ax_pksim.axhline(50, color=GREEN, lw=1.8, linestyle='--', label='Cmin=50nM target ★ (Line403)')
ax_pksim.axhline(12.3 * 4, color=AMBER, lw=1.2, linestyle=':',
                 label=f'4×IC₅₀={12.3*4:.1f}nM safety margin ★ (Line404)')
for dt, i in zip([0, 21, 42, 63], range(1, 5)):
    ax_pksim.axvline(dt, color=GRAY, lw=0.8, linestyle=':', alpha=0.5)
    ax_pksim.text(dt + 0.5, max(conc) * 0.92, f'Dose {i}', fontsize=7,
                  color=GRAY, rotation=90)
ax_pksim.set_xlabel('Time (days)', color=GRAY, fontsize=9)
ax_pksim.set_ylabel('Serum Conc. (nM, MW=148kDa assumed)', color=GRAY, fontsize=9)
ax_pksim.tick_params(colors=GRAY, labelsize=8)
ax_pksim.grid(alpha=0.12, color=GRAY)
ax_pksim.legend(fontsize=8, facecolor=DARK, edgecolor=GRAY, labelcolor=WHITE)
ax_pksim.set_ylim(0, max(conc) * 1.15)
# 주의: 용량 Q3W(21일) 사용 — 논문 제안 용량(Line398)
# 논문 Line405: Q2W ≥10mg/kg를 Cmin 유지 기준으로 기재하나, 제안 임상용량은 Q3W
ax_pksim.text(0.01, 0.97,
              'Q3W 시뮬레이션 (논문 제안 Q3W, Line398)\n'
              '유지 Cmin기준: 논문 Q2W 기재(Line405)\n'
              'MW=148kDa 가정 — 절대값 참고용',
              transform=ax_pksim.transAxes, fontsize=6.5, color=AMBER, va='top',
              bbox=dict(boxstyle='round,pad=0.3', facecolor='#1A1200', edgecolor=AMBER, alpha=0.85))

# ── [R3C3] 독성 프로파일 요약 ─────────────────────────────
ax_tox = fig.add_subplot(gs[2, 3])
panel_bg(ax_tox)
ptitle(ax_tox, "⑫ Toxicity Profile (G3/4 %)  ●", color=RED)

tox_items = ['Neutropenia', 'Anemia', 'Diarrhea', 'Nausea/Vom', 'Neuropathy', 'Fatigue']
tox_folfox = [32, 8, 10, 14, 8, 22]
tox_pritfox = [24, 6, 8, 10, 6, 17]
tox_folfiri = [28, 6, 28, 18, 3, 20]

xp = np.arange(len(tox_items))
wt = 0.25
ax_tox.bar(xp - wt, tox_folfox, wt, label='FOLFOX alone ●', color=RED, alpha=0.75, edgecolor=BG)
ax_tox.bar(xp,      tox_pritfox, wt, label='Prit+FOLFOX ▲', color=BLUE, alpha=0.75, edgecolor=BG)
ax_tox.bar(xp + wt, tox_folfiri, wt, label='FOLFIRI alone ●', color=ORANGE, alpha=0.75, edgecolor=BG)
ax_tox.axhline(30, color=RED, lw=1.5, linestyle='--', alpha=0.6)
ax_tox.axhline(15, color=AMBER, lw=1.0, linestyle=':', alpha=0.6)
ax_tox.set_xticks(xp)
ax_tox.set_xticklabels(tox_items, rotation=30, ha='right', fontsize=7.5, color=WHITE)
ax_tox.set_ylabel('G3/4 Incidence (%)', color=GRAY, fontsize=9)
ax_tox.tick_params(colors=GRAY, labelsize=8)
ax_tox.grid(axis='y', alpha=0.12, color=GRAY)
ax_tox.legend(fontsize=7.5, facecolor=DARK, edgecolor=GRAY, labelcolor=WHITE)
ax_tox.text(0.01, 0.95, 'Horizontal dashed: 30% high-risk', transform=ax_tox.transAxes,
            fontsize=7, color=RED, va='top')

# ════════════════════════ ROW 4 ═══════════════════════════

# ── [R4C0] 환자 선택 전략 ─────────────────────────────────
ax_pts = fig.add_subplot(gs[3, 0])
panel_bg(ax_pts)
ptitle(ax_pts, "⑬ Patient Selection Strategy  ★", color=CYAN)
ax_pts.axis('off')

pat_data = [
    ('Biomarker',          'Criterion',                   'Value',       'Grade'),
    ('PrPc IHC (8H4)',     'H-score ≥ 50',               '85.7% KRAS+', '★'),
    ('KRAS mutation',      'Any allele (NGS)',             '40% CRC',     '★'),
    ('Dual positive',      'PrPc+/KRAS+',                 '34.5% CRC',   '★'),
    ('KRAS G12D H-score',  '142 ± 28',                    'Highest',     '★'),
    ('KRAS G12V H-score',  '138 ± 31',                    '2nd',         '★'),
    ('G13D H-score',       '124 ± 34',                    '3rd',         '★'),  # 추가: Line326
    ('CRC annual (US)',     'PrPc+/KRAS+',                '~52,500',     '★'),
    ('Global total (US)',   'All KRAS indications',        '~120,000+/yr','★'),
    # ❌제거: 'ORR 31%⬇' — 논문에 없는 수치
]  # 이전 버전 오류: 'Excl.: PrPc-low → ORR 31%⬇' 논문 미확인 수치 제거
col_x = [0.01, 0.33, 0.67, 0.93]
row_y = np.linspace(0.93, 0.08, len(pat_data))
for ri, row in enumerate(pat_data):
    for ci, (cell, x) in enumerate(zip(row, col_x)):
        fw = 'bold' if ri == 0 else 'normal'
        fc_bg = '#1E3A5F' if ri == 0 else (PANEL2 if ri % 2 == 0 else PANEL)
        tc = AMBER if ri == 0 else (GREEN if cell == '★' else (BLUE if cell == '◆' else WHITE))
        ax_pts.text(x, row_y[ri], cell, fontsize=7.5, color=tc, fontweight=fw,
                    transform=ax_pts.transAxes, va='center')
    if ri < len(pat_data) - 1:
        ax_pts.plot([0.0, 1.0], [row_y[ri] - 0.045, row_y[ri] - 0.045],
                    color=DARK, lw=0.5, transform=ax_pts.transAxes, clip_on=False)

# ── [R4C1] 임상 개발 로드맵 ──────────────────────────────
ax_clin = fig.add_subplot(gs[3, 1])
panel_bg(ax_clin)
ptitle(ax_clin, "⑭ Clinical Development Roadmap  ★", color=GREEN)
ax_clin.axis('off')

phases = [
    ('Phase I\n(12-18 mo)', '1→15 mg/kg Q3W\n3+3 dose escalation\nMTD/RP2D\nPK + RPSA occupancy', BLUE),
    ('Phase II\n(18-36 mo)', 'n=120 (2:1 Prit+FOLFOX vs FOLFOX)\nmCRC KRAS-mut PrPc-high\nmPFS: 5.5→8.25m ★\nHR=0.667 (α=0.10 power=80%)', GREEN),
    ('Phase III\n(3-5 yr)',  'FOLFOX+Bev ± Pritamab\nDouble-blind RCT\n1st-line mCRC\nOS HR target=0.75 ★', ORANGE),
]
y_ph = [0.88, 0.55, 0.22]
for (ph_lbl, ph_txt, c), y in zip(phases, y_ph):
    rect = FancyBboxPatch((0.03, y-0.12), 0.94, 0.28,
                           boxstyle='round,pad=0.02',
                           facecolor=f'{c}22', edgecolor=c,
                           linewidth=1.5, transform=ax_clin.transAxes)
    ax_clin.add_patch(rect)
    ax_clin.text(0.08, y+0.10, ph_lbl, transform=ax_clin.transAxes,
                 fontsize=9, fontweight='bold', color=c, va='top')
    ax_clin.text(0.08, y+0.02, ph_txt, transform=ax_clin.transAxes,
                 fontsize=7.5, color=WHITE, va='top', linespacing=1.5)
    if y != y_ph[-1]:
        ax_clin.annotate('', xy=(0.5, y-0.12), xytext=(0.5, y-0.12 + 0.04),
                         xycoords='axes fraction',
                         arrowprops=dict(arrowstyle='->', color=GRAY, lw=1.5))

# ── [R4C2-3] 검증 요약 표 ─────────────────────────────────
ax_ver = fig.add_subplot(gs[3, 2:])
panel_bg(ax_ver)
ptitle(ax_ver, "⑮ Data Validation Summary — All Report Values", color=WHITE, fs=12)
ax_ver.axis('off')

ver_rows = [
    ('Category', 'Value', 'Source', 'Grade', 'Status'),
    ('KD (SPR)', '0.84 nM', 'NatureComm Paper §Results', '★', 'VERIFIED'),
    ('IC50 PrPc-RPSA', '12.3 nM', 'NatureComm Paper §Results', '★', 'VERIFIED'),
    ('EC50 reduction', '−24.7% (all 4 drugs)', 'NatureComm Paper Table', '★', 'VERIFIED'),
    ('Bliss 5-FU', '+18.4', 'NatureComm Paper Line 305', '★', 'VERIFIED'),
    ('Bliss Oxaliplatin', '+21.7', 'NatureComm Paper Line 306', '★', 'VERIFIED'),
    ('Loewe DRI', '1.34 (both)', 'NatureComm Paper Line 309', '★', 'VERIFIED'),
    ('ADDS consensus', '0.87/0.89/0.82/0.84', 'NatureComm Paper Line 375-378', '★', 'VERIFIED'),
    ('ddG_RLS', '0.50 kcal/mol', 'NatureComm + paper3_kras.py', '★◆', 'VERIFIED'),
    ('Rate reduction', '55.6%', 'NatureComm + ADDS Arrhenius', '★◆', 'VERIFIED'),
    ('ΔG_bind (Irinotecan)', '−13.0 kcal/mol', 'Lit: −10.7~−14 (Topo-I inhibitors)', '●', 'SUPPORTED'),
    ('ΔG_bind (Oxaliplatin)', '−14.0 kcal/mol', 'Lit: RSC Dalton Trans 2019 QM/MM', '●', 'SUPPORTED'),
    ('ΔG_bind (TAS-102)', '−14.3 kcal/mol', 'Lit: AACR Cancer Res 2022', '●', 'SUPPORTED'),
    ('ΔG_bind (5-FU/FdUMP)', '−11.2 kcal/mol', 'Lit: FdUMP active form ~−11.5', '●', 'SUPPORTED'),
    ('Apoptosis 55%', 'Pritamab alone', 'CC-3+2.8× ★ + siRNA lit ●', '▲+●', 'PROJ+LIT'),
    ('Apoptosis 75%', '+Irinotecan/+Oxaliplatin', 'FOLFOX+sensitizer lit 60-75%', '▲+●', 'PROJ+LIT'),
    ('Apoptosis 80%', '+TAS-102', 'Oncotarget 2021: 75-82%', '▲+●', 'PROJ+LIT'),
    ('Bliss Irinotecan', '+17.3', 'ADDS DL mean 17.10 (synth. cohort)', '◆', 'ADDS EST'),
    ('mPFS target', '5.5→8.25m, HR=0.667', 'NatureComm §Clinical Development', '★', 'VERIFIED'),
]
col_x2 = [0.01, 0.18, 0.38, 0.62, 0.81]
row_y2 = np.linspace(0.97, 0.035, len(ver_rows))
for ri, row in enumerate(ver_rows):
    for ci, (cell, x) in enumerate(zip(row, col_x2)):
        fw = 'bold'
        if ri == 0:
            tc = AMBER
        elif 'VERIFIED' in cell:
            tc = GREEN
        elif 'SUPPORTED' in cell:
            tc = TEAL
        elif 'PROJ' in cell:
            tc = BLUE
        elif 'ADDS' in cell and ri > 0 and ci == 4:
            tc = PURPLE
        elif cell in ['★', '★◆', '◆', '●', '▲+●', '▲']:
            grade_c = {'★': GREEN, '★◆': GREEN, '◆': BLUE, '●': TEAL, '▲+●': PURPLE, '▲': AMBER}
            tc = grade_c.get(cell, WHITE)
        else:
            tc = WHITE
            fw = 'normal' if ri > 0 else 'bold'
        ax_ver.text(x, row_y2[ri], cell, fontsize=6.8, color=tc, fontweight=fw,
                    transform=ax_ver.transAxes, va='center')
    if ri < len(ver_rows)-1:
        ax_ver.plot([0.0, 1.0], [row_y2[ri]-0.028, row_y2[ri]-0.028],
                    color=DARK, lw=0.4, transform=ax_ver.transAxes, clip_on=False)

# ════════════════════════════════════════════════════════
# Footer
# ════════════════════════════════════════════════════════
fig.text(0.5, 0.012,
         "★ Pritamab Nature Communications Paper (2026)  |  "
         "◆ ADDS System (paper3_pritamab_kras.py + DL Pipeline)  |  "
         "● SCI Literature (PubMed-indexed: RSC / AACR / Oncotarget / J Med Chem)  |  "
         "▲ ADDS Energy Model Projection (wet-lab validation required)  |  "
         "Generated: 2026-03-04",
         ha='center', va='bottom', fontsize=8, color=GRAY, style='italic')

plt.savefig(r"f:\ADDS\figures\pritamab_comprehensive_full_report.png",
            dpi=180, bbox_inches='tight', facecolor=BG)
print("Saved: pritamab_comprehensive_full_report.png")
