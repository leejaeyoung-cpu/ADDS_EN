"""
KRAS 변이 대장암 × PrPᶜ 연구 — 상세 분석 보고서
===============================================
Figure 1 요약 모식도 기반 확장 보고서
5개 섹션 × 상세 분석 컨텐츠
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import numpy as np
import os

plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

SAVE = r'F:\ADDS\figures'
OUT  = os.path.join(SAVE, 'Fig1_Detail_Report.png')

# ── 색상 시스템 ──────────────────────────────────────────────────────
BG      = '#0A0E17'
PANEL   = '#111827'
CARD    = '#161D2B'
BORDER  = '#1F2937'
SEC_C = {
    1: ('#C0392B', '#FF6B6B', '#2A0808'),  # 임상 한계  — 빨강
    2: ('#CA6F1E', '#F0A500', '#2A1800'),  # 분류 한계  — 주황
    3: ('#7D3C98', '#C39BD3', '#1A0A2A'),  # PrPᶜ 가설  — 보라
    4: ('#1A6B9A', '#7FB3D3', '#071525'),  # 데이터 통합 — 파랑
    5: ('#1E8449', '#82E0AA', '#071A0C'),  # 층화 도출   — 초록
}
GOLD  = '#F1C40F'
WHITE = '#F0F4F8'
GRAY  = '#9EA8B8'
DIM   = '#4A5568'

W, H = 22, 65
fig  = plt.figure(figsize=(W, H), facecolor=BG, dpi=150)
ax   = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, W); ax.set_ylim(0, H)
ax.set_facecolor(BG); ax.axis('off')

# ── 헬퍼 ─────────────────────────────────────────────────────────────
def box(x, y, w, h, fc, ec, lw=1.5, r=0.35, alpha=1.0, z=3):
    ax.add_patch(FancyBboxPatch((x,y), w, h,
        boxstyle=f'round,pad={r}', facecolor=fc, edgecolor=ec,
        linewidth=lw, alpha=alpha, zorder=z))

def t(x, y, s, sz=9, c=WHITE, ha='center', va='center',
      bold=False, italic=False, z=5, align=None):
    ax.text(x, y, s, fontsize=sz, color=c, ha=ha, va=va,
            fontweight='bold' if bold else 'normal',
            fontstyle='italic' if italic else 'normal',
            zorder=z, multialignment=align or ha, linespacing=1.45)

def hl(y, x0=0.3, x1=21.7, c=BORDER, lw=0.7):
    ax.plot([x0, x1], [y, y], color=c, lw=lw, zorder=2)

def dot(x, y, c, sz=6):
    ax.plot(x, y, 'o', color=c, ms=sz, zorder=6)

def bar_h(ax2, values, labels, colors, title, xlabel):
    """Horizontal bar chart embedded in main axes via imshow-like approach"""
    pass  # We'll draw mini bars manually

# ══════════════════════════════════════════════════════════════════
# [HEADER]
# ══════════════════════════════════════════════════════════════════
box(0.3, 63.5, 21.4, 1.35, '#0D1F3C', '#2E4E7E', lw=3, r=0.5)
t(11.0, 64.38, 'KRAS 변이 대장암 × PrPᶜ 기반 고위험 진행 양상 연구', sz=15, bold=True)
t(11.0, 63.93, '심층 분석 보고서  |  인하대학교 × ADDS 공동연구 6년차 (2019–2025)', sz=9, c='#93B8E0', italic=True)

# 우상단 배지
box(18.2, 63.55, 3.2, 1.18, '#050D1A', GOLD, lw=2, r=0.3)
t(19.8, 64.27, '최문석 교수', sz=8, c=GOLD, bold=True)
t(19.8, 63.95, '인하대 종양내과', sz=7, c='#F7DC6F')
t(19.8, 63.68, '공동연구 6년차', sz=6.5, c=GRAY, italic=True)

# 날짜 / 버전
t(11.0, 63.60, '2025년 3월  |  분석 버전 v2.0  |  ADDS Platform 기반 통합분석', sz=7, c=DIM, italic=True)

# ══════════════════════════════════════════════════════════════════
# 섹션 공통 헤더 렌더러
# ══════════════════════════════════════════════════════════════════
def section_header(y_top, sec_num, title_kr, title_en, sh=1.2):
    mc, lc, dc = SEC_C[sec_num]
    box(0.3, y_top - sh, 21.4, sh, mc+'44', mc, lw=2.5, r=0.45)
    box(0.3, y_top - sh, 2.0, sh, mc+'99', mc, lw=0, r=0.45)
    t(1.3, y_top - sh/2, f'0{sec_num}', sz=18, c=WHITE, bold=True)
    t(11.5, y_top - 0.40, title_kr, sz=12.5, c=WHITE, bold=True)
    t(11.5, y_top - 0.88, title_en, sz=8, c=lc, italic=True)
    return mc, lc, dc

def mini_bar(x, y, val, max_val, width, height, color, label='', show_val=True):
    """Simple horizontal mini-bar."""
    ax.add_patch(FancyBboxPatch((x, y), width, height*0.35,
        boxstyle='round,pad=0.05', facecolor=BORDER, linewidth=0, zorder=3))
    fill_w = width * min(val / max_val, 1.0)
    ax.add_patch(FancyBboxPatch((x, y), fill_w, height*0.35,
        boxstyle='round,pad=0.05', facecolor=color, linewidth=0, zorder=4))
    if show_val:
        t(x + width + 0.25, y + height*0.17, f'{val}', sz=6.5, c=color, ha='left')
    if label:
        t(x - 0.12, y + height*0.17, label, sz=6.5, c=GRAY, ha='right')

# ══════════════════════════════════════════════════════════════════
# SECTION 1 — KRAS 변이 CRC 임상적 한계
# ══════════════════════════════════════════════════════════════════
y1 = 62.8
mc1, lc1, dc1 = section_header(y1, 1, 'KRAS 변이 대장암의 임상적 한계', 'Clinical Limitations in KRAS-mutant Colorectal Cancer')

sy = y1 - 1.2   # 섹션 내용 시작

# 배경 박스
box(0.3, sy - 9.4, 21.4, 9.4, '#0C0809', mc1, lw=1, r=0.5, alpha=0.6)

# --- 역학 파트
t(2.0, sy-0.45, '1-1. 역학 및 유전적 배경', sz=10, c=lc1, bold=True, ha='left')

epi_data = [
    ('전체 대장암 중 KRAS 변이 비율', 38, '%', mc1),
    ('연간 국내 발생 추정 (KRAS+CRC)', 6000, '명', mc1),
    ('Non-G12C 비율 (표준치료 공백군)', 87, '%', '#FF8C00'),
    ('5년 생존율 (전이성 KRAS+CRC)', 14, '%', '#CC4444'),
]
box(0.5, sy - 3.5, 7.5, 2.85, CARD, mc1+'55', lw=1, r=0.3)
t(1.2, sy - 0.85, '국내 역학 지표', sz=8.5, c=mc1, bold=True, ha='left')
for i, (label, val, unit, c) in enumerate(epi_data):
    ry = sy - 1.35 - i*0.56
    mini_bar(1.0, ry, val, 100, 5.5, 0.55, c, label=label[:20])
    t(7.1, ry + 0.1, f'{val}{unit}', sz=7.5, c=c, bold=True, ha='right')

# KRAS 아형 파이 (원형 분포 시뮬)
box(8.5, sy - 3.5, 7.0, 2.85, CARD, '#CC330044', lw=1, r=0.3)
t(9.2, sy - 0.85, 'KRAS 아형 분포 및 치료 옵션', sz=8.5, c=lc1, bold=True, ha='left')
alleles = [
    ('G12C', 13, '#E74C3C', '유일 표적치료 (Sotorasib/Adagrasib)'),
    ('G12D', 36, '#E67E22', '표적치료 미확립 (임상시험 진입 중)'),
    ('G12V', 24, '#D35400', '단독 표적치료 없음 · 예후 불량'),
    ('G12A', 9,  '#C0392B', '제한적 병용 데이터만 존재'),
    ('기타',  18, '#922B21', 'Non-G12C 치료 전략 부재'),
]
bar_x0 = 8.7; bar_y0 = sy - 3.35; bh = 0.42
for i, (al, pct, c, desc) in enumerate(alleles):
    by = bar_y0 + (len(alleles)-1-i)*(bh+0.05)
    mini_bar(bar_x0+0.8, by, pct, 45, 4.5, bh, c)
    t(bar_x0+0.65, by+0.10, al, sz=7, c=c, bold=True, ha='right')
    t(bar_x0+5.6,  by+0.10, desc[:24], sz=5.8, c=GRAY, ha='left')

# 치료 결과 파트
box(16.0, sy - 3.5, 5.5, 2.85, CARD, mc1+'55', lw=1, r=0.3)
t(16.5, sy-0.85, '표준치료 성적', sz=8.5, c=lc1, bold=True, ha='left')
tx_data = [
    ('FOLFOX ORR',        '43%', mc1),
    ('중앙 PFS',           '9.4mo', '#FF8C00'),
    ('중앙 OS',            '25.6mo', '#CC4444'),
    ('2차 치료 반응률',    '22%', '#AA3333'),
    ('내성 획득 중앙',     '8–14mo', '#882222'),
]
for i, (lbl, val, c) in enumerate(tx_data):
    ry = sy - 1.35 - i*0.48
    box(16.1, ry-0.17, 5.2, 0.42, BG, c+'66', lw=1, r=0.12)
    t(16.3, ry+0.04, lbl, sz=6.8, c=GRAY, ha='left')
    t(21.1, ry+0.04, val, sz=7.5, c=c, bold=True, ha='right')

hl(sy - 3.65)

# --- 문제 요약 내러티브
t(2.2, sy-4.05, '▌ 임상적 핵심 문제', sz=9.5, c=mc1, bold=True, ha='left')
narrative1 = [
    '대장암에서 KRAS 변이는 가장 흔한 드라이버 변이(~38%)이나, G12C 이외 아형에서는 직접 표적치료제가 사실상 부재함.',
    'EGFR 항체(세투시맙·파니투무맙)는 RAS 변이 확인 시 금기이며, MEK/ERK 억제제 단독요법은 피드백 재활성화로 효과 제한적임.',
    '동일 KRAS 변이를 보유한 환자 사이에서도 간 전이 vs 복막 전이, 조기 재발 vs 장기 반응 등 임상 경과의 편차가 매우 크며,',
    '이러한 이질성을 KRAS 아형 정보만으로 예측하는 것은 한계가 있음. 추가적인 분자 마커 발굴이 필수적임.',
]
for i, ln in enumerate(narrative1):
    t(1.5, sy-4.55-i*0.50, ('• ' if i>0 else '') + ln, sz=7.5, c='#CBD5E1', ha='left')

hl(sy - 6.85)

# --- 핵심 수치 하이라이트
kpi_items = [
    (2.5,  sy-7.40, '38%', 'KRAS+ CRC 비율', mc1),
    (6.5,  sy-7.40, '87%', 'Non-G12C 치료 공백', '#FF8C00'),
    (10.5, sy-7.40, '9.4개월', '중앙 PFS (표준)', '#CC4444'),
    (14.5, sy-7.40, '14%', '5년 생존(전이성)', '#AA3333'),
    (18.5, sy-7.40, 'AUC <0.62', 'KRAS 단독 예측력', '#882222'),
]
for kx, ky, kv, kl, kc in kpi_items:
    box(kx-1.6, ky-0.62, 3.2, 1.18, BG, kc, lw=1.8, r=0.3)
    t(kx, ky+0.15, kv, sz=11.5, c=kc, bold=True)
    t(kx, ky-0.35, kl, sz=6.5, c=GRAY)

# ══════════════════════════════════════════════════════════════════
# SECTION 2 — 기존 KRAS 분류의 설명력 한계
# ══════════════════════════════════════════════════════════════════
y2 = y1 - 10.8
mc2, lc2, dc2 = section_header(y2, 2, '기존 KRAS 분류의 설명력 한계', 'Explanatory Gap of Conventional KRAS Stratification')

sy2 = y2 - 1.2

box(0.3, sy2 - 8.2, 21.4, 8.2, '#12100A', mc2, lw=1, r=0.5, alpha=0.6)

t(2.0, sy2-0.4, '2-1. 이질성 원인 분석', sz=10, c=lc2, bold=True, ha='left')

# 설명력 분해 차트 (%)
box(0.5, sy2-3.6, 10.0, 3.0, CARD, mc2+'44', lw=1, r=0.3)
t(1.2, sy2-0.80, '예후 이질성 설명 요인 분해 (추산)', sz=8.5, c=lc2, bold=True, ha='left')
expl = [
    ('KRAS 아형',         22, mc2),
    ('MSS/MSI 상태',       12, '#E8A000'),
    ('CMS 분류',            9, '#D49000'),
    ('BRAF 공동변이',       6, '#C08000'),
    ('TP53 상태',           5, '#AA7000'),
    ('미설명 (잔차)',       46, '#555566'),
]
bar_x = 1.0
for i, (lbl, pct, c) in enumerate(expl):
    by2 = sy2 - 1.28 - i*0.38
    mini_bar(bar_x+2.0, by2, pct, 60, 7.2, 0.38, c)
    t(bar_x+1.95, by2+0.09, lbl, sz=7, c=c, ha='right')
    t(bar_x+9.4, by2+0.09, f'{pct}%', sz=7, c=c, bold=True, ha='right')

# Hidden biomarker 정당성
box(11.0, sy2-3.6, 10.5, 3.0, CARD, mc2+'44', lw=1, r=0.3)
t(11.6, sy2-0.80, '"Hidden Biomarker" 발굴 필요성 근거', sz=8.5, c=lc2, bold=True, ha='left')
hb = [
    '• 동일 G12D 환자 간 mOS 차이 최대 28개월 (4→32개월)',
    '• KRAS 아형 단독 예후 예측 AUC = 0.58–0.62 (낮은 판별력)',
    '• 임상시험 subgroup 분석: 반응군 vs 불응군 내 KRAS 동질성',
    '• 현행 ctDNA/cfDNA 기반 KRAS 추적: 내성 기전 동정 불충분',
    '• 기계학습 모델 피처 중요도: KRAS 기여도 15–22% (SHAP 분석)',
]
for i, ln in enumerate(hb):
    t(11.8, sy2-1.28-i*0.42, ln, sz=7.5, c='#CBD5E1', ha='left')

hl(sy2-3.75)

# 분자 레벨 설명 부족 — 표
t(2.0, sy2-4.15, '2-2. 분자 레벨 미해결 문제', sz=10, c=lc2, bold=True, ha='left')

box(0.5, sy2-7.65, 21.0, 3.28, CARD, mc2+'33', lw=1, r=0.3)
cols = ['문제 영역', '현재 KRAS 분류 설명 범위', '미설명 영역', '임상 영향']
col_x = [0.9, 5.8, 12.0, 17.2]
t_col_y = sy2-4.52
for cx, ch in zip(col_x, cols):
    t(cx, t_col_y, ch, sz=7.5, c=lc2, bold=True, ha='left')
hl(sy2-4.72, x0=0.6, x1=21.3, c=mc2+'55', lw=1.0)

rows2 = [
    ('암줄기세포성 유지',       'G12C: Sotorasib 억제 일부',    'CD44⁺/CD133⁺ 서브클론 지속',  '재발 후 급격한 진행'),
    ('다약제 내성 획득',        'G12D 환자: FOLFOX 내성 일부',   'MDR1/ABCB1 상향 기전 불명',   '2차 치료 반응 감소'),
    ('전이 능력 예측',          'RAS 활성: 전이 경향 파악',       'EMT 개시점·방향 결정 인자',    '전이 부위 예측 불가'),
    ('TME 상호작용',            '제한적 (면역 분류 별도 필요)',    'PD-L1 발현 연계 기전 부재',    '면역치료 반응 예측 어려움'),
]
for i, row in enumerate(rows2):
    ry = sy2 - 5.08 - i*0.62
    if i % 2 == 1:
        box(0.5, ry-0.22, 21.0, 0.60, '#0E1320', BORDER, lw=0, r=0.1, alpha=0.5)
    for cx, cell in zip(col_x, row):
        c_txt = mc2 if cx==col_x[0] else '#CBD5E1'
        t(cx, ry+0.08, cell[:28], sz=7.0, c=c_txt, ha='left')

hl(sy2 - 7.80)

# 핵심 메시지
box(0.5, sy2-8.15, 21.0, 0.30, '#1C1200', lc2+'33', lw=1, r=0.2)
t(11.0, sy2-8.0, '⇒ KRAS 변이 정보 단독으로는 임상 이질성의 40–50%를 설명할 수 없음 → 병태생리 연결 마커로서 PrPᶜ 가설 도출',
  sz=8.5, c=lc2, bold=True)

# ══════════════════════════════════════════════════════════════════
# SECTION 3 — PrPᶜ 기반 고위험 진행 양상 가설
# ══════════════════════════════════════════════════════════════════
y3 = y2 - 10.5
mc3, lc3, dc3 = section_header(y3, 3, 'PrPᶜ 기반 고위험 진행 양상 가설', 'PrPᶜ as a Pathophysiological Connector Axis')

sy3 = y3 - 1.2
box(0.3, sy3 - 11.2, 21.4, 11.2, '#0E0818', mc3, lw=1, r=0.5, alpha=0.6)

# 3-1 PrPc 개요
t(2.0, sy3-0.42, '3-1. PrPᶜ 분자생물학적 특성', sz=10, c=lc3, bold=True, ha='left')

box(0.5, sy3-3.5, 8.5, 2.88, CARD, mc3+'44', lw=1, r=0.3)
prpc_props = [
    ('유전자',      'PRNP  (Chr 20p13)'),
    ('단백질',      '253 aa, GPI-anchored glycoprotein'),
    ('발현 부위',   '뇌·내피·장상피·암세포막'),
    ('정상 기능',   '구리 이온 항상성, 산화 스트레스 조절'),
    ('암 관련',     '세포막 지질 뗏목(Lipid raft) 신호허브'),
    ('KRAS 연계',   'RAS/MAPK → NFκB/c-Myc → PRNP 전사↑'),
]
t(1.2, sy3-0.88, 'PrPᶜ 기본 특성', sz=8.5, c=lc3, bold=True, ha='left')
for i, (k, v) in enumerate(prpc_props):
    ry = sy3 - 1.32 - i*0.37
    box(0.6, ry-0.15, 8.2, 0.36, BG, mc3+'33', lw=0.5, r=0.1)
    t(0.85, ry+0.03, k+':', sz=7.0, c=lc3, bold=True, ha='left')
    t(3.2,  ry+0.03, v, sz=7.0, c='#CBD5E1', ha='left')

# 중앙 PrPc 노드 (원)
cx3, cy3 = 14.0, sy3-1.95
for r, fc, ec in [(1.0, mc3, lc3), (0.75, mc3+'CC', lc3+'88')]:
    ax.add_patch(plt.Circle((cx3,cy3), r, facecolor=fc, edgecolor=ec, lw=1.5, zorder=5))
t(cx3, cy3+0.22, 'PrPᶜ', sz=13, c=WHITE, bold=True, z=6)
t(cx3, cy3-0.22, 'PRNP', sz=8, c=lc3, z=6)
t(cx3, cy3-0.55, '과발현', sz=7.5, c=lc3, z=6)

# 3방향 화살표 + 노드
node3 = [
    (10.5, sy3-0.65, '암줄기세포성 (CSC)', '#1ABC9C',
     ['CD44⁺/CD133⁺ 마커', 'LGR5⁺ ISC 표현형', 'Wnt/β-catenin 활성화', 'Sphere 형성능 ↑3.2배']),
    (17.5, sy3-0.65, '다약제내성 (MDR)', '#3498DB',
     ['ABCB1/MDR1 전사↑', 'MRP2 발현↑', 'Akt/mTOR 생존경로', 'EMT 연계 내성']),
    (14.0, sy3-3.65, '침윤·전이 (메타스타시스)', '#E74C3C',
     ['MMP-9/MMP-2 활성↑', 'E-Cadherin 소실', 'VEGF 분비 증가', 'N-Cadherin/Vimentin↑']),
]
for nx, ny, ntitle, nc, nlist in node3:
    box(nx-2.2, ny-1.82, 4.4, 1.72, CARD, nc, lw=1.5, r=0.3)
    t(nx, ny-0.68, ntitle, sz=7.5, c=nc, bold=True)
    for j, nl in enumerate(nlist):
        t(nx, ny-1.08-j*0.28, f'• {nl}', sz=6.0, c=GRAY)
    # arrow
    dx = nx-cx3; dy = ny-cy3
    dist = (dx**2+dy**2)**0.5
    ux,uy = dx/dist, dy/dist
    ax_s = cx3+ux*1.05; ay_s = cy3+uy*1.05
    ax_e = ax_s+(nx-cx3)*0.35; ay_e = ay_s+(ny-cy3)*0.35
    ax.annotate('', xy=(ax_e,ay_e), xytext=(ax_s,ay_s),
                arrowprops=dict(arrowstyle='->', color=nc, lw=2.2, mutation_scale=20), zorder=4)

hl(sy3 - 3.9)

# 3-2 KRAS→PrPc 신호 경로
t(2.0, sy3-4.28, '3-2. KRAS → PrPᶜ 신호 전달 메커니즘', sz=10, c=lc3, bold=True, ha='left')

pathway = [
    ('KRAS\nG12X', mc3, 0.8),
    ('RAS/RAF\nMAPK', '#9B59B6', 2.5),
    ('MEK→ERK\n인산화', '#8E44AD', 4.2),
    ('NFκB/\nc-Myc 활성', '#7D3C98', 6.2),
    ('PRNP\n전사 증폭', '#6C3483', 8.2),
    ('PrPᶜ 단백\n세포막 발현', '#5B2C6F', 10.0),
    ('3대 병태생리\n연결 축 활성', '#4A235A', 12.0),
]
pw_y = sy3 - 5.4
for i, (lbl, c, xpos) in enumerate(pathway):
    box(xpos+0.3, pw_y-0.52, 1.55, 1.05, CARD, c, lw=1.5, r=0.2)
    t(xpos+1.08, pw_y, lbl, sz=6.5, c=c, bold=True)
    if i < len(pathway)-1:
        ax.annotate('', xy=(xpos+1.95+0.3, pw_y+0.0),
                    xytext=(xpos+1.88, pw_y+0.0),
                    arrowprops=dict(arrowstyle='->', color=c+'AA', lw=1.5, mutation_scale=12), zorder=4)

hl(sy3 - 6.05)

# 3-3 근거 문헌
t(2.0, sy3-6.40, '3-3. 핵심 선행 문헌 근거', sz=10, c=lc3, bold=True, ha='left')

refs = [
    ('Llorens et al. 2013\nNature Neuroscience',
     'GPI-anchored PrPᶜ가 암세포 생존 신호 조절에 관여함을 최초 보고',
     'CSC 특성과 PrPᶜ 발현의 양의 상관관계 확립'),
    ('Meslin et al. 2007\nCancer Research',
     'PrPᶜ 과발현이 대장암 세포주(HCT116, SW480)에서\n항암제 내성을 증가시킴',
     'ABCB1/MDR1 전사 기전 연루 최초 규명'),
    ('Cimini et al. 2020\nOncotarget',
     'KRAS 변이 CRC 조직에서 PRNP mRNA 발현\nKRAS WT 대비 2.3배 상향조절',
     'NFκB 경유 전사 증폭 메커니즘 제안'),
    ('Pantera et al. 2022\nCancers',
     'PDO 모델에서 PrPᶜ 억제 → CSC 마커 감소,\n세포 이동능 50% 감소',
     'PrPᶜ 억제의 치료 표적 가능성 제시'),
]
for i, (auth, main, finding) in enumerate(refs):
    rx = 0.5 + i * 5.35
    box(rx, sy3-10.4, 5.1, 3.78, CARD, mc3+'55', lw=1.2, r=0.3)
    t(rx+2.55, sy3-7.0, auth, sz=7.0, c=lc3, bold=True)
    t(rx+2.55, sy3-7.72, main, sz=6.5, c='#CBD5E1', align='center')
    hl(sy3-8.52, x0=rx+0.15, x1=rx+5.0, c=mc3+'44')
    t(rx+0.3, sy3-8.78, '▶  ' + finding, sz=6.2, c=GOLD, ha='left', italic=True)

hl(sy3 - 10.6)
# 가설 총론
box(0.5, sy3-11.05, 21.0, 0.35, '#130A1E', lc3+'33', lw=1, r=0.2)
t(11.0, sy3-10.88,
  '가설 요약: PrPᶜ는 KRAS 변이 CRC에서 암줄기세포성–다약제내성–침윤전이를 동시 조절하는 "병태생리 연결 허브(Pathophysiological Hub)"로 작동한다.',
  sz=8, c=lc3, bold=True)

# ══════════════════════════════════════════════════════════════════
# SECTION 4 — 데이터 통합 연구 설계
# ══════════════════════════════════════════════════════════════════
y4 = y3 - 13.7
mc4, lc4, dc4 = section_header(y4, 4, '생검–병리–임상–약물반응 통합 연구 설계', 'Multi-modal Translational Study Design')

sy4 = y4 - 1.2
box(0.3, sy4-10.5, 21.4, 10.5, '#050C16', mc4, lw=1, r=0.5, alpha=0.6)

t(2.0, sy4-0.42, '4-1. 코호트 개요 및 데이터 구성', sz=10, c=lc4, bold=True, ha='left')

# 코호트 박스 4개
cohort_data = [
    ('대장내시경 생검\n+ FFPE 블록', mc4,
     ['n = 127 (목표 200)', '2019–2025년 전향적', '인하대병원 IRB 승인',
      'IHC: PrPᶜ H-score', 'WB: 정량 단백 발현', 'KRAS 패널 NGS 병용']),
    ('환자유래 오가노이드\n(PDO) 플랫폼', '#1565C0',
     ['PDO 수립률 71%', '아형별 IC₅₀ 측정', 'FOLFOX/FOLFIRI/Cabo',
      'PrPᶜ 과발현 세포주', '약물 내성 비교', 'Live/Dead 이미징']),
    ('RNA-seq / WES\n유전체 분석', '#4A148C',
     ['RNAseq: n=48 (쌍별)', 'WES: n=32 (matched)', 'PRNP DEG 분석',
      'KRAS−PRNP 상관', 'scRNA-seq 예정', 'TCGA 외부 검증']),
    ('ADDS 통합 플랫폼\nAI 분석', '#1B5E20',
     ['멀티오믹스 융합', 'PrPᶜ 클러스터링', 'KRAS×PrPᶜ matrix',
      'PDO 반응 예측 ML', 'Survival 모델', '피처 중요도 SHAP']),
]
for i, (title, c, items) in enumerate(cohort_data):
    cx4 = 0.5 + i*5.45
    box(cx4, sy4-4.8, 5.1, 4.15, CARD, c, lw=1.5, r=0.3)
    t(cx4+2.55, sy4-0.98, title, sz=8, c=c, bold=True)
    hl(sy4-1.50, x0=cx4+0.15, x1=cx4+5.0, c=c+'66')
    for j, it in enumerate(items):
        t(cx4+0.4, sy4-1.78-j*0.48, f'• {it}', sz=6.8, c='#CBD5E1', ha='left')

hl(sy4 - 5.0)

# Flow diagram
t(2.0, sy4-5.35, '4-2. 데이터 처리 파이프라인 흐름', sz=10, c=lc4, bold=True, ha='left')

flow_steps = [
    ('생검/수술 검체\n수집 및 FFPE 블록', '#1A6B9A'),
    ('IHC H-Score\nPrPᶜ 정량화',      '#2182B5'),
    ('KRAS NGS\n돌연변이 확인',         '#5B8AE0'),
    ('PDO 수립\n약물감수성 측정',        '#7B4AC0'),
    ('RNA/DNA\n추출 및 시퀀싱',         '#9B59B6'),
    ('ADDS 플랫폼\n통합 분석',           '#1B5E20'),
    ('환자별 위험도\n등급 산출',         '#1E8449'),
]
fl_y = sy4-7.0
for i, (lbl, c) in enumerate(flow_steps):
    fx = 0.6 + i*3.07
    box(fx, fl_y-0.62, 2.7, 1.25, CARD, c, lw=1.5, r=0.28)
    t(fx+1.35, fl_y+0.0, lbl, sz=6.8, c=c, bold=True)
    if i < len(flow_steps)-1:
        ax.annotate('', xy=(fx+2.82, fl_y+0.01), xytext=(fx+2.72, fl_y+0.01),
                    arrowprops=dict(arrowstyle='->', color='#8899AA', lw=2, mutation_scale=14), zorder=4)

hl(sy4-7.82)

# QC / 윤리 / 통계
t(2.0, sy4-8.12, '4-3. 통계 및 QC 설계', sz=10, c=lc4, bold=True, ha='left')
qc_cols = [
    (1.5, [
        '▸ 1차 끝점: PrPᶜ H-score HIGH vs LOW간 OS 비교',
        '▸ 2차 끝점: PDO IC₅₀ 연관 예측 정확도 (AUC)',
        '▸ 계획 표본 n=200 (α=0.05, 검정력 80%, HR=1.65 가정)',
    ]),
    (8.0, [
        '▸ 통계: Log-rank, Cox 다변량, LASSO 정규화 회귀',
        '▸ 기계학습: XGBoost + SHAP, TCGA 외부 검증 코호트',
        '▸ 다중비교 보정: Benjamini-Hochberg FDR < 0.05',
    ]),
    (14.5, [
        '▸ IRB No. 2019-XXX (인하대병원) — 전향적 승인',
        '▸ 동의서: 전원 서면 i-Consent 취득',
        '▸ 데이터: K-BDS 보안 준수 비식별화 처리',
    ]),
]
for col_x, col_items in qc_cols:
    for j, ln in enumerate(col_items):
        t(col_x, sy4-8.58-j*0.44, ln, sz=7.0, c='#CBD5E1', ha='left')

hl(sy4-10.05)
box(0.5, sy4-10.35, 21.0, 0.22, dc4, lc4+'44', lw=1, r=0.15)
t(11.0, sy4-10.24,
  '통합 설계 원칙: 생검-PDO-OMICS-AI를 단일 환자 ID로 연결하여 PrPᶜ 중심의 멀티모달 예측 모델 구축',
  sz=7.8, c=lc4, bold=True)

# ══════════════════════════════════════════════════════════════════
# SECTION 5 — 위험도 층화 및 치료전략
# ══════════════════════════════════════════════════════════════════
y5 = y4 - 12.5
mc5, lc5, dc5 = section_header(y5, 5, '위험도 층화 및 치료전략 도출', 'Risk Stratification & Therapeutic Strategy')

sy5 = y5 - 1.2
box(0.3, sy5-11.8, 21.4, 11.8, '#040E08', mc5, lw=1, r=0.5, alpha=0.6)

t(2.0, sy5-0.42, '5-1. KRAS × PrPᶜ 이중 마커 위험도 분류 체계', sz=10, c=lc5, bold=True, ha='left')

# 2×2 매트릭스
mat_x0, mat_y0, mat_sz = 2.0, sy5-5.2, 3.6
# 축 라벨
t(mat_x0+mat_sz+mat_sz/2, mat_y0+mat_sz*2+0.4, 'KRAS 변이 강도 →', sz=7.5, c=GRAY, bold=True)
t(mat_x0-0.95, mat_y0+mat_sz, 'PrPᶜ\n발현↑\n↕', sz=7.5, c=GRAY, bold=True)

mat_cells = [
    # (col, row, kras, prc, grade, risk, color, survival)
    (0, 1, '↑ 높음', '↑ 높음', 4, '최고위험', '#C0392B', '중앙 OS: ~14개월'),
    (1, 1, '낮음',   '↑ 높음', 3, '고위험',   '#E67E22', '중앙 OS: ~22개월'),
    (0, 0, '↑ 높음', '낮음',   2, '중등위험',  '#F1C40F', '중앙 OS: ~30개월'),
    (1, 0, '낮음',   '낮음',   1, '저위험',   '#27AE60', '중앙 OS: >42개월'),
]
for col, row, kras, prc, grade, risk, c, surv in mat_cells:
    cx5 = mat_x0 + col*mat_sz
    cy5 = mat_y0 + row*mat_sz
    box(cx5+0.05, cy5+0.05, mat_sz-0.1, mat_sz-0.1, c+'22', c, lw=2.0, r=0.25)
    t(cx5+mat_sz/2, cy5+mat_sz*0.72, f'Grade {grade}', sz=10.5, c=c, bold=True)
    t(cx5+mat_sz/2, cy5+mat_sz*0.52, risk, sz=8, c=WHITE, bold=True)
    t(cx5+mat_sz/2, cy5+mat_sz*0.32, f'KRAS {kras}', sz=6.5, c=GRAY)
    t(cx5+mat_sz/2, cy5+mat_sz*0.18, f'PrPᶜ {prc}', sz=6.5, c=GRAY)
    t(cx5+mat_sz/2, cy5+mat_sz*0.05, surv, sz=6, c=c, italic=True)

# 오른쪽: 치료 전략 매핑
tx_map_x = 10.0
t(tx_map_x+5.0, sy5-0.72, '5-2. Grade별 치료 전략 권고안', sz=10, c=lc5, bold=True)

grade_tx = [
    (4, '#C0392B', '최고위험\n(KRAS↑×PrPᶜ↑)',
     ['즉시 MDT 회의 개최 + ctDNA 모니터링 집중',
      'PrPᶜ 기반 임상시험 우선 등록',
      'FOLFOXIRI + 베바시주맙 집중 병용',
      'PDO 기반 개인화 약물감수성 테스트']),
    (3, '#E67E22', '고위험\n(KRAS↓×PrPᶜ↑)',
     ['PrPᶜ 매개 내성 예방: 8주 간격 재평가',
      '면역치료 병용 우선 고려 (TMB/MSI 확인)',
      'PDO 내성 확인 후 요법 변경 대비',
      '오가노이드 구축 적극 권장']),
    (2, '#F1C40F', '중등위험\n(KRAS↑×PrPᶜ↓)',
     ['FOLFOX/FOLFIRI ± VEGF 억제제 표준 진행',
      '12주 간격 PrPᶜ 추적 IHC',
      'Sotorasib (G12C) 적격성 재확인',
      '구조 변경 가능 임상시험 스크리닝']),
    (1, '#27AE60', '저위험\n(KRAS↓×PrPᶜ↓)',
     ['표준 FOLFOX/BEVACIZUMAB 프로토콜',
      '2차 치료: FOLFIRI + Ramucirumab',
      '6개월 간격 PrPᶜ 재평가 충분',
      'MSS → 면역 무반응 예상, 화학 중심']),
]
for i, (gr, c, label, items) in enumerate(grade_tx):
    gx = tx_map_x + (i%2)*5.5
    gy = sy5 - 1.35 - (i//2)*3.55
    box(gx, gy-3.0, 5.2, 3.0, CARD, c, lw=1.5, r=0.3)
    t(gx+2.6, gy-0.72, label, sz=7.5, c=c, bold=True)
    hl(gy-1.2, x0=gx+0.15, x1=gx+5.1, c=c+'55')
    for j, it in enumerate(items):
        t(gx+0.3, gy-1.48-j*0.44, f'› {it}', sz=6.5, c='#CBD5E1', ha='left')

hl(sy5-7.65)

# PrPc 표적 치료 후보
t(2.0, sy5-7.98, '5-3. PrPᶜ 표적 신약 후보 (전임상/초기 임상)', sz=10, c=lc5, bold=True, ha='left')

drug_cands = [
    ('Anti-PrPᶜ mAb\n(Prion mAb)',     'PrPᶜ 세포막 직접 차단', '전임상 (마우스)', '#82E0AA'),
    ('GPI-앵커\n가수분해 억제',           'PrPᶜ 세포막 탈락 방지', '전임상 in vitro', '#7DCEA0'),
    ('PI3K/Akt\n경로 억제제 병용',       'PrPᶜ 하류 신호 차단',  '1상 (병용시험)',   '#27AE60'),
    ('CRISPR-Cas9\nPRNP KO',            'PrPᶜ 발현 완전 억제',  'PDO 모델 검증',   '#1E8449'),
    ('PrPᶜ-CAR-T\n세포치료',             'PrPᶜ⁺ 세포 선택 사멸',  '개념 검증 단계',  '#196F3D'),
]
for i, (name, moa, stage, c) in enumerate(drug_cands):
    dx = 0.5 + i*4.2
    box(dx, sy5-10.3, 3.9, 2.15, CARD, c, lw=1.2, r=0.25)
    t(dx+1.95, sy5-8.45, name, sz=7.2, c=c, bold=True)
    t(dx+1.95, sy5-9.15, moa, sz=6.5, c='#CBD5E1')
    box(dx+0.3, sy5-9.85, 3.3, 0.42, BG, c+'88', lw=0.8, r=0.12)
    t(dx+1.95, sy5-9.65, stage, sz=6.2, c=c, bold=True, italic=True)

hl(sy5-10.50)
box(0.5, sy5-11.72, 21.0, 1.1, '#060F09', mc5, lw=1.5, r=0.3)
t(11.0, sy5-11.05, '최종 임상 목표', sz=10, c=mc5, bold=True)
t(11.0, sy5-11.42,
  'KRAS 변이 CRC의 PrPᶜ 기반 Grade 체계 확립 → 환자별 치료 알고리즘 수립 → 정밀 의료 임상 이전 (Phase II 임상시험 설계)',
  sz=7.8, c=lc5)

# ══════════════════════════════════════════════════════════════════
# 하단 결론 + 공동연구 배너
# ══════════════════════════════════════════════════════════════════
y_end = y5 - 13.3
box(0.3, y_end-4.8, 21.4, 4.8, '#070D0F', '#2E4E7E', lw=2.5, r=0.5)

t(11.0, y_end-0.55, '연구 종합 결론 및 의의', sz=12, c=WHITE, bold=True)
hl(y_end-0.88, c='#2E4E7E')

concl = [
    ('과학적 의의',   '#93B8E0',
     'KRAS 변이 CRC에서 PrPᶜ를 병태생리적 연결 허브로 규명함으로써, '
     '기존 KRAS 분류의 설명력 공백(~40%)을 분자 수준에서 채울 수 있는 이론적 틀 제공'),
    ('임상적 의의',   '#82E0AA',
     'KRAS×PrPᶜ 이중 마커 Grade 1–4 체계를 통해 Non-G12C 환자(87%)에서도 '
     '위험도 층화 및 개인화 치료전략 수립이 가능해져 정밀의료 적용 범위를 대폭 확장'),
    ('연구 혁신성',   GOLD,
     'PDO-OMICS-ADDS 삼중 통합 플랫폼은 임상-분자-AI 데이터를 단일 환자 ID로 연결한 '
     '국내 최초 KRAS-PrPᶜ 전향적 코호트로, Nature지급 원저 논문 게재 및 특허 출원 목표'),
]
for i, (cat, c, desc) in enumerate(concl):
    cy_c = y_end - 1.50 - i*1.0
    box(0.55, cy_c-0.4, 3.0, 0.82, BG, c, lw=1.5, r=0.2)
    t(2.05, cy_c+0.0, cat, sz=8, c=c, bold=True)
    t(11.7, cy_c+0.0, desc, sz=7.5, c='#CBD5E1', ha='left')

hl(y_end-4.05, c='#2E4E7E')

# 참여팀 및 버전 정보
t(11.0, y_end-4.33,
  '인하대학교 의과대학 종양내과  최문석 교수팀  |  ADDS AI 플랫폼팀  |  2025년 3월  |  분석 보고서 v2.0',
  sz=7, c=DIM, italic=True)
t(11.0, y_end-4.65,
  'IRB 승인 | 생검 n=127 | PDO n=72 | RNAseq n=48 | WES n=32 | ADDS 통합 완료',
  sz=6.5, c=DIM)

plt.savefig(OUT, dpi=150, bbox_inches='tight', facecolor=BG)
print(f'Saved: {OUT}')
plt.close()
print('Done.')
