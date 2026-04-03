"""
Figure 1. KRAS 변이 대장암의 치료 한계와 PrPᶜ 기반 고위험 진행 양상 연구 개념도
================================================================================
Layout (가로 3단 구조):

Panel A (Left):    임상적 문제 — KRAS 변이 CRC의 이질성 + 치료 한계
Panel B (Center):  PrPᶜ 병태생리 축 — 줄기세포성 / 다약제내성 / 침윤전이
Panel C (Right):   연구 전략 — 데이터 연계 → ADDS 통합분석 → 임상 활용

흐름: A → B → C  (문제 → 메커니즘 → 해결전략)
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Arc, Circle
from matplotlib.lines import Line2D
import numpy as np
import os

# ── Korean font setup ──────────────────────────────────────────────────────────
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

SAVE = r'F:\ADDS\figures'
OUT  = os.path.join(SAVE, 'Fig1_KRAS_PrPc_Concept.png')

# ── Color Palette ──────────────────────────────────────────────────────────────
BG       = '#0D1117'      # figure background
PANEL_BG = '#161B22'      # panel background
GRAY     = '#21262D'
BORDER   = '#30363D'

# Panel A — Clinical problem
RED_DARK  = '#FF4444'
RED_MED   = '#FF6B6B'
RED_LIGHT = '#FFB3B3'
ORANGE    = '#FF9500'

# Panel B — Mechanism
PURPLE_D  = '#7B2FBE'
PURPLE_M  = '#9B59B6'
PURPLE_L  = '#C39BD3'
GOLD      = '#F1C40F'
TEAL      = '#1ABC9C'
BLUE_M    = '#3498DB'

# Panel C — Research strategy
GREEN_D   = '#196F3D'
GREEN_M   = '#27AE60'
GREEN_L   = '#A9DFBF'
CYAN      = '#17A589'
STEEL     = '#2E86C1'

TEXT_MAIN = '#E6EDF3'
TEXT_SUB  = '#8B949E'
TEXT_EM   = '#FFFFFF'

# ── Figure Setup ───────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(24, 14), facecolor=BG, dpi=180)

# Three main columns + narrow connectors
# Proportions: [0.30, 0.03, 0.35, 0.03, 0.295]  (A, arrow, B, arrow, C)
left_pad = 0.03; right_pad = 0.03; top = 0.90; bottom = 0.04
col_w = (1 - left_pad - right_pad - 0.06) / 3   # width of each panel
gap   = 0.03

ax_A = fig.add_axes([left_pad,            bottom, col_w,        top-bottom])
ax_B = fig.add_axes([left_pad+col_w+gap,  bottom, col_w,        top-bottom])
ax_C = fig.add_axes([left_pad+2*col_w+2*gap, bottom, col_w,     top-bottom])

for ax in [ax_A, ax_B, ax_C]:
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.set_facecolor(PANEL_BG)
    ax.axis('off')
    for sp in ax.spines.values():
        sp.set_visible(False)

# Helper functions
def rounded_box(ax, x, y, w, h, fc, ec, lw=1.2, radius=0.3, alpha=1.0, zorder=2):
    box = FancyBboxPatch((x, y), w, h,
                         boxstyle=f'round,pad={radius}',
                         facecolor=fc, edgecolor=ec, linewidth=lw,
                         alpha=alpha, zorder=zorder)
    ax.add_patch(box)
    return box

def arrow(ax, x0, y0, x1, y1, color='#8B949E', lw=1.5, style='->', zorder=3):
    ax.annotate('', xy=(x1,y1), xytext=(x0,y0),
                arrowprops=dict(arrowstyle=style, color=color, lw=lw,
                                connectionstyle='arc3,rad=0'), zorder=zorder)

def txt(ax, x, y, s, size=7.5, color=TEXT_MAIN, ha='center', va='center',
        bold=False, italic=False, wrap=False, zorder=5):
    weight = 'bold' if bold else 'normal'
    style  = 'italic' if italic else 'normal'
    ax.text(x, y, s, fontsize=size, color=color, ha=ha, va=va,
            fontweight=weight, fontstyle=style, zorder=zorder,
            wrap=wrap, multialignment=ha)

def divider(ax, y, color=BORDER):
    ax.axhline(y, color=color, lw=0.8, xmin=0.02, xmax=0.98, zorder=1)

# ══════════════════════════════════════════════════════════════════════════════
# PANEL A: 임상적 문제 — KRAS 변이 CRC 이질성
# ══════════════════════════════════════════════════════════════════════════════
ax = ax_A

# Header
rounded_box(ax, 0.3, 9.0, 9.4, 0.85, RED_DARK, RED_MED, lw=2, radius=0.25)
txt(ax, 5.0, 9.42, 'KRAS 변이 대장암의 임상적 이질성', 9.5, TEXT_EM, bold=True)

# KRAS allele subtypes
rounded_box(ax, 0.3, 7.55, 9.4, 1.28, '#1A0A0A', RED_DARK+'55', radius=0.2, lw=1)
txt(ax, 5.0, 9.0,  'KRAS 아형',        7.5, RED_LIGHT, bold=True)
subtypes = [
    ('G12C\n(~13%)',  '#E74C3C', 1.2,  7.95, '유일 표적치료\n(Sotorasib)'),
    ('G12D\n(~36%)', '#E67E22', 3.3,  7.95, '표적치료 미존재\n임상시험 중'),
    ('G12V\n(~24%)', '#D35400', 5.4,  7.95, '표적치료 미존재\n예후 불량'),
    ('G12A/S/R\n외\n(~27%)', '#C0392B', 7.5, 7.95, '제한적 옵션\n대부분 non-G12C'),
]
for label, ec, cx, cy, note in subtypes:
    rounded_box(ax, cx-0.85, cy-0.35, 1.7, 0.95, '#2A0A0A', ec, lw=1.5, radius=0.2)
    txt(ax, cx, cy+0.18, label, 6.8, ec, bold=True)
    txt(ax, cx, cy-0.2, note, 5.5, TEXT_SUB, italic=True)

divider(ax, 7.5)

# Clinical heterogeneity — 4 dimensions
txt(ax, 5.0, 7.28, '임상적 이질성 4가지 차원', 7.5, RED_LIGHT, bold=True)
dims = [
    (1.5, 6.65, '전이 양상\n편차', '간·폐·복막\n부위별 차이', '#E74C3C'),
    (3.8, 6.65, '재발 위험\n편차', 'R0 절제 후\n조기 재발 다양', '#E67E22'),
    (6.2, 6.65, '항암반응\n편차', 'FOLFOX 반응률\n15–45% 편차', '#D35400'),
    (8.5, 6.65, '내성 획득\n시점 편차', '중앙 내성 획득\n4–16개월 차이', '#E74C3C'),
]
for cx, cy, title, desc, c in dims:
    rounded_box(ax, cx-1.1, cy-0.72, 2.2, 1.10, '#1C0808', c+'88', radius=0.2, lw=1.2)
    txt(ax, cx, cy+0.12, title, 7.0, c, bold=True)
    txt(ax, cx, cy-0.38, desc, 5.6, TEXT_SUB, italic=True)

divider(ax, 5.75)

# Problem statement
rounded_box(ax, 0.4, 4.72, 9.2, 0.88, '#110606', ORANGE, lw=1.5, radius=0.2)
txt(ax, 5.0, 5.12, '⚠  기존 KRAS 분류만으로는 임상적 이질성 설명 불충분', 7.5, ORANGE, bold=True)

# Non-G12C limitation
rounded_box(ax, 0.3, 3.30, 9.4, 1.25, '#0D0808', RED_DARK+'33', radius=0.2)
txt(ax, 5.0, 4.2, 'Non-G12C 아형의 치료 공백 (전체 KRAS 변이 CRC의 ~87%)', 7, RED_LIGHT, bold=True)
gap_items = [
    (1.8, 3.6, 'EGFR 항체\n（RAS 변이 → 불응）'),
    (5.0, 3.6, 'MEK/ERK 억제제\n（단독 효과 제한적）'),
    (8.2, 3.6, 'PI3K/AKT 경로\n（병용만 유효）'),
]
for cx, cy, s in gap_items:
    rounded_box(ax, cx-1.2, cy-0.35, 2.4, 0.82, '#1A0505', RED_DARK+'55', radius=0.15)
    txt(ax, cx, cy+0.03, s, 5.8, RED_MED)

divider(ax, 3.2)

# Research question
rounded_box(ax, 0.3, 1.92, 9.4, 1.15, '#120010', PURPLE_M+'66', radius=0.25, lw=1.5)
txt(ax, 5.0, 2.76, '핵심 연구 질문', 8, PURPLE_L, bold=True)
txt(ax, 5.0, 2.25, 'KRAS 변이 아형 이외의 추가 분자 마커가 이질성을 설명하는가?', 7, TEXT_MAIN)
txt(ax, 5.0, 1.90, '→ PrPᶜ (Cellular Prion Protein) 가설', 7.5, GOLD, bold=True)

# Panel label
txt(ax, 0.55, 9.72, 'A', 14, TEXT_EM, bold=True)

# ══════════════════════════════════════════════════════════════════════════════
# PANEL B: PrPᶜ 병태생리 축
# ══════════════════════════════════════════════════════════════════════════════
ax = ax_B

# Header
rounded_box(ax, 0.3, 9.0, 9.4, 0.85, PURPLE_D, PURPLE_L, lw=2, radius=0.25)
txt(ax, 5.0, 9.42, 'PrPᶜ 기반 병태생리 연결 축', 9.5, TEXT_EM, bold=True)

# Central PrPc node
cx, cy, r = 5.0, 6.50, 1.0
circle = plt.Circle((cx, cy), r, color=PURPLE_D, zorder=4)
ax.add_patch(circle)
circle2 = plt.Circle((cx, cy), r*0.88, color=PURPLE_M, zorder=5)
ax.add_patch(circle2)
txt(ax, cx, cy+0.25, 'PrPᶜ',  11, TEXT_EM, bold=True, zorder=6)
txt(ax, cx, cy-0.15, '(PRNP)', 7, PURPLE_L, zorder=6)
txt(ax, cx, cy-0.45, '과발현', 7, PURPLE_L, zorder=6)

# Three output nodes
node_data = [
    # (x, y, title, sub1, sub2, sub3, color)
    (2.0, 8.50, '암줄기세포성\n유지', 'CD44⁺/CD133⁺ 발현',   'Wnt/β-catenin 활성',     'LGR5 상향조절',      TEAL),
    (8.0, 8.50, '다약제 내성\n획득', 'ABCB1/MRP 상향',       'Akt/mTOR 생존신호',       'EMT 연계 내성',       BLUE_M),
    (5.0, 4.25, '침윤·전이\n촉진',   'MMP-9/uPA 활성화',     'E-cadherin 소실',         '혈관신생 (VEGF)',     RED_MED),
]
for nx, ny, title, s1, s2, s3, nc in node_data:
    rounded_box(ax, nx-1.55, ny-0.82, 3.1, 1.5, GRAY, nc, lw=1.8, radius=0.25)
    txt(ax, nx, ny+0.35, title, 7.5, nc, bold=True)
    for i, s in enumerate([s1, s2, s3]):
        txt(ax, nx, ny-0.05-i*0.35, f'• {s}', 5.8, TEXT_SUB)

# Arrows: PrPc → 3 nodes
for nx, ny, _, _2, _3, _4, nc in node_data:
    dx = nx - cx; dy = ny - cy
    dist = (dx**2+dy**2)**0.5
    ux, uy = dx/dist, dy/dist
    x0 = cx + ux*r*1.05; y0 = cy + uy*r*1.05
    ax.annotate('', xy=(x0+(nx-cx)*0.3, y0+(ny-cy)*0.3),
                xytext=(x0, y0),
                arrowprops=dict(arrowstyle='->', color=nc, lw=2.0), zorder=3)

divider(ax, 3.8)

# Upstream: KRAS → PrPc link
rounded_box(ax, 0.4, 2.80, 9.2, 0.85, '#0D0816', PURPLE_M+'88', radius=0.2)
txt(ax, 5.0, 3.22, 'KRAS 변이 → RAS/MAPK 신호 → PrPᶜ 전사 증폭 (NFκB, c-Myc)', 7, PURPLE_L)

# Evidence boxes
rounded_box(ax, 0.3, 1.40, 9.4, 1.25, '#08080F', GOLD+'44', radius=0.2)
txt(ax, 5.0, 2.30, '기존 근거', 8, GOLD, bold=True)
evid = [
    (1.8, 1.85, 'Llorens et al.\n2013 Nat Neurosci'),
    (5.0, 1.85, 'Meslin et al.\n2007 Cancer Res'),
    (8.2, 1.85, 'Cimini et al.\n2020 Oncotarget'),
]
for ex, ey, ev in evid:
    rounded_box(ax, ex-1.3, ey-0.38, 2.6, 0.82, GRAY, GOLD+'66', radius=0.15)
    txt(ax, ex, ey+0.03, ev, 5.8, GOLD)

# Bidirectional relationship annotation
rounded_box(ax, 0.3, 0.15, 9.4, 1.1, '#050510', STEEL+'44', radius=0.2)
txt(ax, 5.0, 0.90, '↗ PrPᶜ 과발현 = KRAS 변이 CRC 고위험 표현형 예측인자', 7, STEEL, bold=True)
txt(ax, 5.0, 0.55, 'PrPᶜ 발현↑ → 암줄기세포성↑ → 내성↑ → 전이↑ → 불량 예후', 6.8, TEXT_MAIN)
txt(ax, 0.55, 9.72, 'B', 14, TEXT_EM, bold=True, zorder=10)

# ══════════════════════════════════════════════════════════════════════════════
# PANEL C: 연구 전략 — 데이터 연계 → ADDS → 임상 활용
# ══════════════════════════════════════════════════════════════════════════════
ax = ax_C

# Header
rounded_box(ax, 0.3, 9.0, 9.4, 0.85, GREEN_D, GREEN_L, lw=2, radius=0.25)
txt(ax, 5.0, 9.42, '연구 전략 및 데이터 통합 파이프라인', 9.5, TEXT_EM, bold=True)

# --- Data sources (top)
txt(ax, 5.0, 8.70, '① 데이터 연계 (Multi-modal)', 7.5, GREEN_L, bold=True)
srcs = [
    (1.5, 8.15, '대장내시경\n생검', '병리·임상\nFFPE 정량', '#1A5276'),
    (3.8, 8.15, '수술 검체\nPDO', '약물반응\n오가노이드', '#145A32'),
    (6.2, 8.15, 'IHC/WB\n정량', 'PrPᶜ 단백\n발현 수준', '#4A235A'),
    (8.5, 8.15, 'RNA-seq\nWES', 'KRAS+PrPc\n공동분석', '#1B4F72'),
]
for sx, sy, t1, t2, c in srcs:
    rounded_box(ax, sx-1.05, sy-0.65, 2.1, 1.05, '#080C14', c, lw=1.5, radius=0.2)
    txt(ax, sx, sy+0.12, t1, 6.8, TEXT_MAIN, bold=True)
    txt(ax, sx, sy-0.35, t2, 5.5, TEXT_SUB, italic=True)

divider(ax, 7.35)

# --- ADDS integration
rounded_box(ax, 0.3, 6.20, 9.4, 0.98, '#0A0F1A', STEEL, lw=2.0, radius=0.25)
txt(ax, 5.0, 6.90, '② ADDS 기반 통합 분석 (AI-Driven Data Science)', 7.5, STEEL, bold=True)
adds_items = [
    '멀티오믹스 융합', 'PDO 예측 모델', 'PrPᶜ 발현 클러스터링', 'KRAS×PrPᶜ 상호작용'
]
for i, itm in enumerate(adds_items):
    xi = 1.25 + i * 2.3
    rounded_box(ax, xi-0.92, 6.22, 1.84, 0.50, GRAY, STEEL+'88', radius=0.12)
    txt(ax, xi, 6.47, itm, 5.5, TEXT_MAIN)

divider(ax, 6.0)

# --- Analysis outputs
txt(ax, 5.0, 5.82, '③ 분석 출력 (Endpoints)', 7.5, CYAN, bold=True)
outs = [
    (2.0, 5.30, 'PrPᶜ 발현\n위험도 층화', 'High / Mid / Low'),
    (5.0, 5.30, 'KRAS 아형별\nPDO 약물반응', 'IC₅₀ 상관 예측'),
    (8.0, 5.30, '전이·재발\n조기 예측', 'Time-to-event 모델'),
]
for ox, oy, t1, t2 in outs:
    rounded_box(ax, ox-1.3, oy-0.62, 2.6, 1.1, '#06100A', CYAN+'88', radius=0.2)
    txt(ax, ox, oy+0.18, t1, 7, CYAN, bold=True)
    txt(ax, ox, oy-0.28, t2, 5.8, TEXT_SUB, italic=True)

divider(ax, 4.45)

# --- Clinical translation
txt(ax, 5.0, 4.28, '④ 임상 적용 목표', 7.5, GREEN_M, bold=True)
clin = [
    (1.8, 3.85, '환자별 위험도 층화\n(Risk Stratification)'),
    (5.0, 3.85, 'PrPᶜ 표적\n치료전략 수립'),
    (8.2, 3.85, '병용 요법\n후보 우선순위화'),
]
for cl_x, cl_y, cl_t in clin:
    rounded_box(ax, cl_x-1.25, cl_y-0.52, 2.5, 0.95, '#050E07', GREEN_M+'77', radius=0.18)
    txt(ax, cl_x, cl_y+0.1, cl_t, 6.2, GREEN_M, bold=True)

divider(ax, 3.15)

# --- Integration with KRAS × PrPc matrix
rounded_box(ax, 0.3, 1.80, 9.4, 1.22, '#070C10', GOLD+'44', radius=0.2)
txt(ax, 5.0, 2.68, '⑤ KRAS × PrPᶜ 이중 마커 분류 체계', 8, GOLD, bold=True)
mat_labels = [
    (2.2, 2.10, 'KRAS↑×PrPᶜ↑',  '최고위험 (Grade 4)', '#E74C3C'),
    (5.5, 2.10, 'KRAS↑×PrPᶜ↓',  '중등위험 (Grade 2)', '#E67E22'),
    (8.5, 2.10, 'KRAS↓×PrPᶜ↑',  '주의 (Grade 3)',      '#F1C40F'),
]
for mx, my, mk, mv, mc in mat_labels:
    rounded_box(ax, mx-1.3, my-0.32, 2.6, 0.65, GRAY, mc+'99', radius=0.12)
    txt(ax, mx, my+0.06, mk, 6.2, mc, bold=True)
    txt(ax, mx, my-0.22, mv, 5.5, TEXT_SUB)

# Final goal
rounded_box(ax, 0.3, 0.15, 9.4, 1.5, '#050F0A', GREEN_M, lw=2.5, radius=0.3)
txt(ax, 5.0, 1.05, '🎯  최종 목표', 9, TEXT_EM, bold=True)
txt(ax, 5.0, 0.65, 'KRAS 변이 CRC 고위험 진행 양상 정의 + 환자별 위험도 층화', 7, TEXT_MAIN)
txt(ax, 5.0, 0.35, '→ PrPᶜ 기반 정밀의료 치료전략 근거 마련', 7, GREEN_L, bold=True)
txt(ax, 0.55, 9.72, 'C', 14, TEXT_EM, bold=True)

# ── Inter-panel arrows (A→B, B→C) ─────────────────────────────────────────────
# Uses figure coordinates
for x_pos in [left_pad + col_w + gap/2, left_pad + 2*col_w + 1.5*gap]:
    ax_arrow = fig.add_axes([x_pos-0.005, 0.50, 0.012, 0.04])
    ax_arrow.set_xlim(0, 1); ax_arrow.set_ylim(0, 1)
    ax_arrow.axis('off')
    ax_arrow.set_facecolor(BG)
    ax_arrow.annotate('', xy=(0.5, 0.05), xytext=(0.5, 0.95),
                      arrowprops=dict(arrowstyle='->', color=GOLD, lw=3, mutation_scale=22))

# ── Title ─────────────────────────────────────────────────────────────────────
fig.text(0.5, 0.96,
         'Figure 1.  KRAS 변이 대장암의 치료 한계와 PrPᶜ 기반 고위험 진행 양상 연구 개념도',
         ha='center', va='top', fontsize=12, color=TEXT_EM, fontweight='bold')
fig.text(0.5, 0.925,
         'KRAS-mutant CRC: Clinical Heterogeneity  ▶  PrPᶜ Pathophysiological Axis  ▶  ADDS-integrated Research Strategy',
         ha='center', va='top', fontsize=8.5, color=TEXT_SUB, fontstyle='italic')

# ── Save ──────────────────────────────────────────────────────────────────────
plt.savefig(OUT, dpi=200, bbox_inches='tight', facecolor=BG)
print(f'Saved: {OUT}')
plt.close()
print('Done.')
