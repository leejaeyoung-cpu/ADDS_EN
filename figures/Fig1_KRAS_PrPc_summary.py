"""
Figure 1. KRAS 변이 대장암 PrPᶜ 기반 고위험 진행 연구 요약 모식도
- 5단계 수직 흐름도 (Linear narrative flowchart)
- 인하대학교 / 최문석 교수 공동연구 6년차 맥락 포함
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.lines import Line2D
import matplotlib.patheffects as pe
import numpy as np
import os

plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

SAVE = r'F:\ADDS\figures'
OUT  = os.path.join(SAVE, 'Fig1_KRAS_PrPc_Summary.png')

# ── 색상 ──────────────────────────────────────────────────────────
BG      = '#0A0E17'
PANEL   = '#111827'
BORDER  = '#1F2937'

C = [          # 단계별 색상 (5단계)
    '#C0392B',  # 1 — 임상 한계  (짙은 빨강)
    '#E67E22',  # 2 — 설명력 한계 (오렌지)
    '#8E44AD',  # 3 — PrPc 가설  (보라)
    '#1A6B9A',  # 4 — 데이터 통합 (파랑)
    '#1E8449',  # 5 — 층화 도출   (초록)
]
GOLD  = '#F1C40F'
WHITE = '#F0F4F8'
GRAY  = '#A0AEC0'
DIM   = '#4A5568'

# ── 레이아웃 ──────────────────────────────────────────────────────
W, H = 18, 26      # 세로 긴 형식 (모식도)
fig  = plt.figure(figsize=(W, H), facecolor=BG, dpi=180)
ax   = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, W); ax.set_ylim(0, H)
ax.set_facecolor(BG); ax.axis('off')

# ── 헬퍼 ──────────────────────────────────────────────────────────
def box(x, y, w, h, fc, ec, lw=1.5, radius=0.4, alpha=1.0, zorder=3):
    p = FancyBboxPatch((x, y), w, h,
                        boxstyle=f'round,pad={radius}',
                        facecolor=fc, edgecolor=ec,
                        linewidth=lw, alpha=alpha, zorder=zorder)
    ax.add_patch(p)

def txt(x, y, s, size=9, color=WHITE, ha='center', va='center',
        bold=False, italic=False, zorder=5, wrap=True):
    ax.text(x, y, s, fontsize=size, color=color, ha=ha, va=va,
            fontweight='bold' if bold else 'normal',
            fontstyle='italic' if italic else 'normal',
            zorder=zorder, multialignment=ha, wrap=wrap,
            linespacing=1.4)

def arrow_down(x, y1, y2, color, lw=3):
    ax.annotate('', xy=(x, y2+0.15), xytext=(x, y1-0.12),
                arrowprops=dict(arrowstyle='->', color=color, lw=lw, mutation_scale=28),
                zorder=4)

def hline(y, color=BORDER):
    ax.axhline(y, color=color, lw=0.6, xmin=0.04, xmax=0.96)

# ══════════════════════════════════════════════════════════════════
# 상단 헤더: 연구 배경 배너
# ══════════════════════════════════════════════════════════════════
box(0.5, 24.4, 17, 1.35, '#0D1F3C', '#2E4E7E', lw=2.5, radius=0.5)
txt(9.0, 25.32,
    'Figure 1.  KRAS 변이 대장암의 치료 한계와 PrPᶜ 기반 고위험 진행 양상 연구 개념도',
    size=13.5, color=WHITE, bold=True)
txt(9.0, 24.88,
    'KRASᵐᵘᵗ Colorectal Cancer: From Clinical Heterogeneity to PrPᶜ-Integrated Risk Stratification',
    size=9, color='#93B8E0', italic=True)

# 공동연구 태그 (우상단)
box(13.3, 24.55, 4.0, 1.05, '#0A1A2E', GOLD, lw=2.0, radius=0.3)
txt(15.3, 25.32, '인하대학교 의대', size=7.5, color=GOLD, bold=True)
txt(15.3, 25.00, '최문석 교수 공동연구팀', size=7.0, color=GOLD)
txt(15.3, 24.70, '☞ 연구 6년차  (2019–2025)', size=6.8, color='#F7DC6F', italic=True)

# ══════════════════════════════════════════════════════════════════
# 5단계 블록 정의
# ══════════════════════════════════════════════════════════════════
BX = 0.7   # 블록 시작 x
BW = 16.6  # 블록 너비
steps = [
    # (y_top, color, step_num, title_kr, title_en,
    #  body_lines,          side_data)
    (
        23.0, C[0], '①',
        'KRAS 변이 대장암의 임상적 한계',
        'Clinical Limitations of KRAS-mutant CRC',
        [
            '• KRAS 변이 CRC: 전체 대장암의 35–45%, 국내 연간 ~6,000명',
            '• Non-G12C 아형 (~87%): EGFR 항체 불응 · 표적치료 미확립',
            '• 동일 KRAS 변이라도 전이 양상, 재발 위험, 항암반응 편차 매우 큼',
            '• 기존 1차 치료 (FOLFOX/FOLFIRI ± 베바시주맙): 중앙 PFS 8–12개월',
        ],
        [
            ('G12C ~13%', '유일 표적치료 (Sotorasib)', C[0]),
            ('G12D ~36%', '표적치료 미확립', C[0]),
            ('G12V ~24%', '예후 불량군', C[0]),
            ('G12A/S/R ~27%', 'Non-G12C 공백', C[0]),
        ]
    ),
    (
        19.3, C[1], '②',
        '기존 KRAS 분류의 설명력 한계',
        'Explanatory Gap of Conventional KRAS Classification',
        [
            '• KRAS 아형만으로는 생존 예후 이분화 설명 불충분 (AUC < 0.62)',
            '• 동일 G12D라도 TME·면역 상태·줄기세포성에 따라 임상경과 분기',
            '• MSI, CMS 아형, POLE 변이 조합 → 부분 설명에 그침',
            '• "Hidden biomarker" 필요성: 분자 기전을 연결하는 통합 인자',
        ],
        [
            ('AUC <0.62', 'KRAS 단독 예후예측', C[1]),
            ('편차 원인', '미설명 ~40%', C[1]),
        ]
    ),
    (
        15.6, C[2], '③',
        'PrPᶜ 기반 고위험 진행 양상 가설',
        'PrPᶜ-Axis Hypothesis for High-Risk Progression',
        [
            '• PrPᶜ (Cellular Prion Protein, PRNP): 세포막 GPI-anchored glycoprotein',
            '• KRAS → RAS/MAPK → NFκB/c-Myc 경유 PrPᶜ 전사 증폭',
            '• PrPᶜ ↑ → [암줄기세포성 CD44⁺/CD133⁺] + [MDR: ABCB1/MRP] + [침윤: MMP-9/E-Cad↓]',
            '• 3대 병태생리 축 연결자: 줄기세포성 ↔ 내성 ↔ 전이 동시 조절',
        ],
        [
            ('암줄기세포성', 'Wnt/β-cat·LGR5', C[2]),
            ('다약제내성', 'Akt/mTOR·EMT', C[2]),
            ('침윤·전이', 'MMP9·VEGF', C[2]),
        ]
    ),
    (
        11.9, C[3], '④',
        '생검–병리–임상–약물반응 통합 연구 설계',
        'Integrated Multi-modal Translational Study Design',
        [
            '• 대장내시경 생검 + 수술 검체: FFPE 기반 IHC/WB PrPᶜ 정량',
            '• 환자유래 오가노이드 (PDO): KRAS 아형별 약물반응 IC₅₀ 측정',
            '• RNA-seq / WES: PRNP 발현 ↔ KRAS 공동변이 통합분석',
            '• ADDS 플랫폼: 멀티오믹스 융합 · 클러스터링 · 예측모델 학습',
        ],
        [
            ('생검/FFPE', 'PrPᶜ IHC·WB', C[3]),
            ('PDO', '약물반응 IC₅₀', C[3]),
            ('RNA-seq/WES', 'PRNP 공동분석', C[3]),
            ('ADDS AI', '통합 예측 모델', C[3]),
        ]
    ),
    (
        8.2, C[4], '⑤',
        '위험도 층화 및 치료전략 도출',
        'Risk Stratification and Therapeutic Strategy',
        [
            '• KRAS×PrPᶜ 이중마커 분류: Grade 1–4 위험도 체계 확립',
            '• Grade 4 (KRAS↑×PrPᶜ↑): 집중 모니터링 + 병용 임상시험 우선 등록',
            '• PrPᶜ 지향 치료 후보: anti-PrPᶜ 항체 / GPI-신호 차단 / PDO 검증',
            '• 임상 활용 목표: 환자별 정밀의료 실현, 불필요한 독성 최소화',
        ],
        [
            ('Grade 4', 'KRAS↑×PrPᶜ↑  최고위험', '#E74C3C'),
            ('Grade 3', 'KRAS↓×PrPᶜ↑  주의', '#E67E22'),
            ('Grade 1–2', 'KRAS↑×PrPᶜ↓  중등', '#F1C40F'),
        ]
    ),
]

STEP_H = [3.4, 3.4, 3.4, 3.4, 3.4]   # 각 블록 높이
ARROW_X = 9.0

for idx, (y_top, color, num, title_kr, title_en, bullets, side) in enumerate(steps):
    sh = STEP_H[idx]

    # ── 외곽 박스
    box(BX, y_top - sh, BW, sh, PANEL, color, lw=2.0, radius=0.5)

    # ── 번호 + 제목 헤더 영역
    box(BX, y_top - 1.05, BW, 1.05, color+'55', color, lw=0, radius=0.5)
    txt(1.55, y_top - 0.52, num,       size=16, color=WHITE,  bold=True, ha='center')
    txt(9.0,  y_top - 0.37, title_kr,  size=11, color=WHITE,  bold=True, ha='center')
    txt(9.0,  y_top - 0.78, title_en,  size=7.5, color=color+'DD', italic=True, ha='center')

    # ── 본문 bullet (좌측)
    body_x = 2.0
    n = len(bullets)
    for i, bul in enumerate(bullets):
        by = y_top - 1.40 - i * ((sh - 1.2) / max(n, 1))
        txt(body_x, by, bul, size=8.2, color='#D1D9E6', ha='left', va='top')

    # ── 우측 사이드 태그
    tag_x0 = 12.85
    for j, (tag_title, tag_sub, tc) in enumerate(side):
        ty = y_top - 1.45 - j * 0.78
        box(tag_x0, ty - 0.28, 4.0, 0.64, '#0F1B27', tc, lw=1.2, radius=0.18)
        txt(tag_x0 + 2.0, ty + 0.04, tag_title, size=7.2, color=tc,  bold=True, ha='center')
        txt(tag_x0 + 2.0, ty - 0.22, tag_sub,   size=6.2, color=GRAY, ha='center')

    # ── 화살표 (마지막 제외)
    if idx < len(steps) - 1:
        next_top = steps[idx+1][0]
        arrow_down(ARROW_X, y_top - sh, next_top, color)

# ══════════════════════════════════════════════════════════════════
# 하단: 최문석 교수 공동연구 맥락 + 연구 타임라인
# ══════════════════════════════════════════════════════════════════
y_bot = 7.6
box(BX, y_bot - 2.65, BW, 2.65, '#09130A', C[4], lw=2.5, radius=0.5)

txt(9.0, y_bot - 0.55,
    '공동연구 맥락 : 인하대학교 의과대학 종양내과  ×  ADDS 플랫폼  (연구 6년차)',
    size=9.5, color=GOLD, bold=True)

# 타임라인
tl_items = [
    (1.6,  '2019–2020\n연구 1–2년차', '기초 발굴\nPrPᶜ-CRC 상관 확인'),
    (5.0,  '2021–2022\n연구 3–4년차', 'PDO 구축\n약물반응 DB 확립'),
    (8.4,  '2023–2024\n연구 5–6년차', 'ADDS 통합\n멀티오믹스 분석'),
    (11.8, '2025\n(현재)', 'Grade 체계 완성\n임상 적용 프로토콜'),
]
for tx_x, yr, desc in tl_items:
    box(BX + tx_x - 0.05, y_bot - 2.45, 3.0, 1.72, '#0E1A10', C[4]+'99', lw=1.2, radius=0.25)
    txt(BX + tx_x + 1.45, y_bot - 1.38, yr,   size=7.5, color=C[4], bold=True)
    txt(BX + tx_x + 1.45, y_bot - 1.95, desc, size=7.0, color=GRAY)
    # connector dots
    ax.plot(BX + tx_x + 1.5, y_bot - 1.0, 'o', color=C[4], ms=6, zorder=6)
# timeline line
ax.plot([BX+1.6+1.45, BX+11.8+1.45], [y_bot-1.0, y_bot-1.0],
        color=C[4]+'88', lw=1.5, ls='--', zorder=3)

txt(9.0, y_bot - 2.42,
    '최문석 교수 (인하대 종양내과)  |  1저자 공동연구  |  IRB 승인 No. 2019-XXX  |  등록 케이스 n=127 (목표 n=200)',
    size=7.0, color=DIM, italic=True)

# ══════════════════════════════════════════════════════════════════
# 맨 하단: 범례 및 목표 강조
# ══════════════════════════════════════════════════════════════════
box(BX, 0.25, BW, 4.55, '#050C0F', '#2E4E7E', lw=1.8, radius=0.5)
txt(9.0, 4.45, '연구 최종 목표 : KRAS 변이 대장암 환자별 위험도 층화 → PrPᶜ 기반 정밀의료 치료전략 근거 마련',
    size=10, color=WHITE, bold=True)

goals = [
    (2.8,  3.75, C[0], '임상 문제 정의', 'Non-G12C 공백 해소'),
    (6.35, 3.75, C[2], 'PrPᶜ 메커니즘', '병태생리 연결 축 확립'),
    (9.9,  3.75, C[3], '데이터 통합',   'ADDS 기반 예측 모델'),
    (13.4, 3.75, C[4], '위험도 층화',  'Grade 체계 → 임상 이전'),
]
for gx, gy, gc, gt, gs in goals:
    box(gx-1.7, gy-0.72, 3.4, 1.2, '#0A1318', gc, lw=1.5, radius=0.22)
    txt(gx, gy+0.15, gt, size=8, color=gc, bold=True)
    txt(gx, gy-0.30, gs, size=7, color=GRAY)
ax.plot([1.2, 16.8], [3.02, 3.02], color=BORDER, lw=0.8)

# 범례 (하단)
leg_items = [
    (1.5,  C[0], '① 임상 한계'),
    (4.0,  C[1], '② 분류 한계'),
    (6.5,  C[2], '③ PrPᶜ 가설'),
    (9.0,  C[3], '④ 데이터 통합'),
    (11.5, C[4], '⑤ 위험도 층화'),
    (14.2, GOLD, '공동연구 6년차'),
]
for lx, lc, lt in leg_items:
    ax.plot(lx, 2.45, 's', color=lc, ms=9, zorder=5)
    txt(lx + 0.55, 2.45, lt, size=7.2, color=lc, ha='left')

txt(9.0, 1.72, 'ADDS : AI-Driven Data Science Platform  |  PDO : Patient-Derived Organoid',
    size=7, color=DIM, italic=True)
txt(9.0, 1.25,
    '본 연구는 인하대학교 최문석 교수팀과의 6년간 전향적 코호트 연구로, 대장내시경 생검 및 수술 검체 기반 PrPᶜ 발현 정량화,',
    size=7, color=DIM)
txt(9.0, 0.82,
    'KRAS 변이 아형별 PDO 약물반응 데이터, 그리고 ADDS 기반 통합분석을 결합하여 KRAS 변이 CRC의 고위험 진행 양상을 규명함.',
    size=7, color=DIM)

# ── 저장 ─────────────────────────────────────────────────────────
plt.savefig(OUT, dpi=200, bbox_inches='tight', facecolor=BG)
print(f'Saved: {OUT}')
plt.close()
print('Done.')
