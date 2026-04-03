#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
KRAS × PrPᶜ 연구 계획서 — 전체 개념도 파이프라인 v2 (Compact Layout)
=====================================================================
공동연구: [연구책임자] × 인하대학교병원 외과 최문석 교수
연구기간: 6개년 장기 종단 연구
플랫폼:   ADDS (AI-Driven Drug Selection) v6.2.0
작성일:   2026-03-16

v2 변경사항:
  - 전체 높이 48→32인치로 축소, 카드 높이 최적화
  - 배경 텍스트 완전 표시 (wrap width 조정 + 영역 확대)
  - 카드 내 좌측 영역 축소, 3컬럼 폰트 확대
  - inter-card 화살표 크기 개선
  - 역할 분담 항목 전체 렌더링 확인
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import matplotlib.font_manager as fm
import textwrap
import warnings
warnings.filterwarnings('ignore')

# ═══════════════════════════════════════════════════════════════════
# 1. 한글 폰트 탐색
# ═══════════════════════════════════════════════════════════════════
def get_korean_font():
    for c in ['Malgun Gothic', 'NanumGothic', 'AppleGothic', 'Noto Sans CJK KR']:
        if c in {f.name for f in fm.fontManager.ttflist}:
            return c
    for f in fm.fontManager.ttflist:
        if any(k in f.name.lower() for k in ['malgun', 'nanum', 'gothic']):
            return f.name
    return None

KF = get_korean_font()
if KF:
    plt.rcParams['font.family'] = KF
plt.rcParams['axes.unicode_minus'] = False

# ═══════════════════════════════════════════════════════════════════
# 2. 색상 팔레트
# ═══════════════════════════════════════════════════════════════════
C = {
    'bg':      '#FFFFFF',
    'navy':    '#1A2E4A',
    'navy_l':  '#2C4A6E',
    'p1':      '#1D4ED8',   # PART 1  Blue
    'p2':      '#6D28D9',   # PART 2  Purple
    'p3':      '#047857',   # PART 3  Green
    'p4':      '#B45309',   # PART 4  Amber
    'p5':      '#B91C1C',   # PART 5  Red
    'bd':      '#CBD5E1',
    'tl':      '#94A3B8',
    'td':      '#1E293B',
    'tm':      '#334155',
    'line':    '#E2E8F0',
    'line_d':  '#E8EDF3',
}

# ═══════════════════════════════════════════════════════════════════
# 3. 데이터 정의
# ═══════════════════════════════════════════════════════════════════
PARTS = [
    {
        'n': 1, 'c': C['p1'], 'lc': '#DBEAFE',
        'title': 'KRAS 변이 대장암의 임상적 한계',
        'en': 'Clinical Limitations of KRAS-Mutant CRC',
        'cols': [
            ('임상 이질성 4축',
             '• 전이 양상 / 재발 위험\n• 항암제 반응 / 내성 시점\n  환자마다 현격히 다름'),
            ('비-G12C 치료 공백',
             '• KRAS 변이 CRC의 ~80%\n  G12D/G12V — 승인 표적치료 없음\n• Sotorasib은 G12C만 적용'),
            ('ADDS 코호트 수치 (n=1,000)',
             '• G12D ORR 58%  mPFS 13.86mo\n• G12V ORR 55%  mPFS 14.45mo\n• G12C ORR 49%  (9%p 편차)'),
        ],
    },
    {
        'n': 2, 'c': C['p2'], 'lc': '#EDE9FE',
        'title': '기존 KRAS 분류의 설명력 한계',
        'en': 'Explanatory Gap of Current KRAS Classification',
        'cols': [
            ('공백 A — 암줄기세포성',
             '• CD44⁺/CD133⁺ 비율 차이\n  KRAS 분류에 미반영\n• 재발 시점·구형 형성 상관 분석'),
            ('공백 B — 내성 획득 속도',
             '• 동일 레지멘(FOLFOX) 환자 간\n  1차 내성 시점 분산 확인\n• PDO IC50 변화 + ctDNA VAF'),
            ('공백 C — 전이 경로 선택',
             '• 간·폐·복막 전이 초발 경로\n  예측 불가\n• CT 패턴 + 분자 상관성 분석'),
        ],
    },
    {
        'n': 3, 'c': C['p3'], 'lc': '#D1FAE5',
        'title': 'PrPᶜ 기반 고위험 진행 양상 가설',
        'en': 'PrPᶜ-Based High-Risk Progression Hypothesis',
        'cols': [
            ('PrPᶜ-RPSA 시그날로솜 기전',
             '• PrPᶜ–RPSA 복합체 활성\n• RAS-GTP 로딩 −42%\n• pERK −38% / Caspase-3 +280%'),
            ('TCGA 조직-혈청 역설',
             '• 5개 암종 2,285건 분석\n• 조직 mRNA↓ ↔ 혈청 단백질↑\n• ADAM10/17 shedding 규명'),
            ('바이오마커 전략 전환',
             '• 조직 mRNA 대리지표 → 폐기\n• 혈청 PrPᶜ 직접 ELISA 정량\n• 비침습적·동적 추적 가능'),
        ],
    },
    {
        'n': 4, 'c': C['p4'], 'lc': '#FEF3C7',
        'title': '생검–병리–임상–약물반응 통합',
        'en': '6-Layer Integrated Data Pipeline',
        'cols': [
            ('검체 레이어 (L1–L3)',
             '• L1 대장내시경 생검 (기저값)\n• L2 수술 검체 (공간 이질성)\n• L3 FFPE IHC H-score 정량'),
            ('CT·ADDS 레이어 (L4)',
             '• TotalSegmentator 117장기\n• 처리 15.67초/환자  ±7mm\n• RECIST 1.1 자동 평가'),
            ('ADDS 융합 레이어 (L5–L6)',
             '• L5 PDO 약물반응 IC50\n• L6 4모달 MLP  PFS R²=0.812\n• 이중 추론 엔진 MDT 보고서'),
        ],
    },
    {
        'n': 5, 'c': C['p5'], 'lc': '#FEE2E2',
        'title': '위험도 층화 및 치료전략 도출',
        'en': 'Risk Stratification & Therapeutic Strategy',
        'cols': [
            ('PrPᶜ × KRAS 매트릭스',
             '🔴 고위험: 비-G12C + 고발현\n🟡 중위험: 어느 한쪽\n🟢 저위험: G12C + 저발현'),
            ('Pritamab 병용 효과',
             '• ORR 51.5% vs 대조 24.0%\n   (+27.5%p 절대 개선)\n• mOS +2.87개월 (17.01 vs 14.14)'),
            ('동적 재층화 알고리즘',
             '• 혈청 PrPᶜ 3개월 간격 재측정\n• ctDNA VAF 2주 간격 추적\n• 위험 등급 자동 갱신'),
        ],
    },
]

PIPE_LAYERS = [
    ('L1', '대장내시경\n생검',       '#3B82F6'),
    ('L2', '수술\n검체',             '#6366F1'),
    ('L3', 'FFPE\nH-score',         '#8B5CF6'),
    ('L4', 'CT / ADDS\n파이프라인',  '#D97706'),
    ('L5', 'PDO\n약물반응',          '#10B981'),
    ('L6', 'ADDS 4모달\n융합 MLP',   '#DC2626'),
]

CELLS_MAT = [
    (0.12, 0.48, '#FEE2E2', '🔴 고위험',   '비-G12C + PrPᶜ 고발현\nPritamab 병용 / ctDNA 집중'),
    (0.57, 0.48, '#FEF9C3', '🟡 중위험-B',  'G12C + PrPᶜ 고발현\nSoto/Adagra + 항-PrPᶜ 검토'),
    (0.12, 0.08, '#FEF9C3', '🟡 중위험-A',  '비-G12C + PrPᶜ 저발현\nFOLFOX 우선 3개월 재측정'),
    (0.57, 0.08, '#DCFCE7', '🟢 저위험',    'G12C + PrPᶜ 저발현\nSotorasib 단독'),
]

ROADMAP = [
    ('Year 1–2', C['p1'], '기반 구축',
     'IRB 승인 · 레트로 코호트 N=200\n절단값 결정 · PDO 인프라 구축'),
    ('Year 3–4', C['p2'], '기전 검증',
     'PrPᶜ-RPSA 논문 (NatComm)\n전향 파일럿 N=100 · AUC≥0.75'),
    ('Year 5–6', C['p5'], '임상 적용',
     'Nature Medicine 최종 제출\n다기관 N=300 · PCT 특허 출원'),
]

ROLES = [
    ('최문석 교수 (인하대병원 외과)', C['p1'], [
        '환자 코호트 구축 및 임상 총괄',
        '대장내시경/수술 검체 수집',
        'FFPE 병리 판독 및 IHC 총괄',
        'PDO 배양 및 약물반응 실험',
        'IRB 임상 책임자',
    ]),
    ('ADDS 연구팀', C['p3'], [
        'ADDS 플랫폼 개발 및 유지',
        'CT 파이프라인 · 4모달 MLP 학습',
        '이중 추론 엔진 운영',
        '위험 점수 알고리즘 개발',
        '특허 출원 및 기술이전 담당',
    ]),
]

# 연구 배경 핵심 텍스트
BG_TEXT = (
    "KRAS 변이 대장암은 동일한 유전적 배경을 공유하더라도 전이 양상, 재발 위험, "
    "항암제 반응 및 내성 획득 시점의 편차가 크게 나타나며, 특히 비-G12C 아형에서는 "
    "적용 가능한 표적치료 전략이 제한적임. 본 연구는 이러한 임상적 이질성이 기존 "
    "KRAS 분류만으로는 충분히 설명되지 않는다는 문제의식에서 출발하여, PrPᶜ가 "
    "암줄기세포성 유지, 다약제 내성 획득 및 침윤·전이를 연결하는 병태생리 축으로 "
    "작동할 가능성을 검증하고자 함. 이를 위해 대장내시경 생검 및 수술 검체 기반 "
    "병리·임상 정보, FFPE 정량지표, 환자유래 오가노이드/PDO 약물반응 정보 및 "
    "ADDS 기반 통합분석을 연계하여, KRAS 변이 대장암의 고위험 진행 양상을 정의하고 "
    "환자별 위험도 층화와 치료전략 수립의 근거를 마련할 것이다."
)

# ═══════════════════════════════════════════════════════════════════
# 4. Figure Canvas  — 가로 26 × 세로 34 (v1 48 → v2 34)
# ═══════════════════════════════════════════════════════════════════
FIG_W, FIG_H = 26, 34
fig = plt.figure(figsize=(FIG_W, FIG_H), facecolor='white')


# ─── Helper functions ─────────────────────────────────────────────
def ax_(x, y, w, h):
    """Create an axes at figure-fraction coords."""
    a = fig.add_axes([x, y, w, h])
    a.set_xlim(0, 1)
    a.set_ylim(0, 1)
    a.axis('off')
    return a


def box_(ax, x, y, w, h, fc, ec, lw=1.5, r=0.015):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f'round,pad={r}',
        facecolor=fc, edgecolor=ec, linewidth=lw,
    ))


def t_(ax, x, y, s, **kw):
    ax.text(x, y, s, **kw)


# ═══════════════════════════════════════════════════════════════════
# 5. HEADER  (상단 3.5%)
# ═══════════════════════════════════════════════════════════════════
ah = ax_(0, 0.965, 1, 0.035)
box_(ah, 0, 0, 1, 1, C['navy'], 'none', r=0)
t_(ah, 0.5, 0.68,
   'KRAS 변이 대장암의 치료 한계와 PrPᶜ 기반 고위험 진행 양상 연구',
   ha='center', va='center', color='white', fontsize=26, fontweight='black')
t_(ah, 0.5, 0.28,
   '연구 계획서 전체 개념도  |  인하대학교병원 외과 최문석 교수 × ADDS 연구팀  |  6개년 장기 종단 연구  |  ADDS v6.2.0',
   ha='center', va='center', color='#93C5FD', fontsize=13)

# ═══════════════════════════════════════════════════════════════════
# 6. 연구 배경 배너 — 전체 텍스트 표시 (높이 5.5%)
# ═══════════════════════════════════════════════════════════════════
BG_H = 0.068
BG_Y = 0.965 - BG_H - 0.004
ab = ax_(0.02, BG_Y, 0.96, BG_H)
box_(ab, 0, 0, 1, 1, '#EFF6FF', C['p1'], lw=2)
t_(ab, 0.015, 0.92, '연구 배경  |  Research Background',
   ha='left', va='center', color=C['p1'], fontsize=14, fontweight='bold')

# wrap text into lines
lines = textwrap.wrap(BG_TEXT, width=100)
for li_idx, line_text in enumerate(lines):
    t_(ab, 0.015, 0.78 - li_idx * 0.12, line_text,
       ha='left', va='center', color=C['td'], fontsize=11, linespacing=1.5)

# ═══════════════════════════════════════════════════════════════════
# 7. 전체 흐름 배너 (2%)
# ═══════════════════════════════════════════════════════════════════
FLOW_H = 0.022
FLOW_Y = BG_Y - FLOW_H - 0.004
af_flow = ax_(0.02, FLOW_Y, 0.96, FLOW_H)
box_(af_flow, 0, 0, 1, 1, C['navy'], 'none', lw=0, r=0.008)
t_(af_flow, 0.5, 0.50,
   '임상 문제  ▶  분류 한계  ▶  PrPᶜ 가설  ▶  다층 데이터 통합  ▶  위험도 층화 · 치료전략',
   ha='center', va='center', color='white', fontsize=14, fontweight='bold')

# ═══════════════════════════════════════════════════════════════════
# 8. PART 1–5 카드  (each ~9% height, total ~48%)
# ═══════════════════════════════════════════════════════════════════
CARD_TOP = FLOW_Y - 0.004
CARD_H = 0.093
GAP_C = 0.010

for idx, P in enumerate(PARTS):
    top = CARD_TOP - (idx + 1) * (CARD_H + GAP_C) + CARD_H
    ac = ax_(0.02, top, 0.96, CARD_H)

    # card background
    box_(ac, 0, 0, 1, 1, '#FAFCFF', P['c'], lw=2.5)

    # left sidebar: 15% width
    box_(ac, 0, 0, 0.15, 1, P['lc'], 'none', r=0)
    # color accent edge
    box_(ac, 0, 0, 0.008, 1, P['c'], 'none', r=0)

    # PART badge (centered in sidebar)
    badge_cx = 0.079
    badge_cy = 0.72
    cc = plt.Circle((badge_cx, badge_cy), 0.08, color=P['c'], zorder=5)
    ac.add_patch(cc)
    t_(ac, badge_cx, badge_cy, str(P['n']),
       ha='center', va='center', color='white', fontsize=22, fontweight='black', zorder=6)

    # PART label below badge
    t_(ac, badge_cx, 0.52, f"PART {P['n']}",
       ha='center', va='center', color=P['c'], fontsize=11, fontweight='bold')

    # title + english subtitle in sidebar area
    t_(ac, badge_cx, 0.34, P['title'],
       ha='center', va='center', color=P['c'], fontsize=11.5, fontweight='bold',
       wrap=True)
    t_(ac, badge_cx, 0.15, P['en'],
       ha='center', va='center', color=C['tl'], fontsize=8.5, style='italic',
       wrap=True)

    # vertical separator
    ac.plot([0.155, 0.155], [0.05, 0.95], color=P['c'], lw=1.5, alpha=0.3)

    # 3-column bullets (right 85%)
    COL_START = 0.17
    COL_WIDTH = 0.27
    COL_GAP = 0.005
    for bi, (bttl, btxt) in enumerate(P['cols']):
        bx = COL_START + bi * (COL_WIDTH + COL_GAP)
        # header strip
        box_(ac, bx, 0.80, COL_WIDTH - 0.005, 0.16, P['lc'], P['c'], lw=1, r=0.005)
        t_(ac, bx + (COL_WIDTH - 0.005) / 2, 0.88, bttl,
           ha='center', va='center', color=P['c'], fontsize=11, fontweight='bold')
        # content
        t_(ac, bx + 0.015, 0.73, btxt,
           ha='left', va='top', color=C['tm'], fontsize=10.5, linespacing=1.65)
        # dotted separator between columns
        if bi < 2:
            sep_x = bx + COL_WIDTH - 0.002
            ac.plot([sep_x, sep_x], [0.05, 0.95],
                    color=C['line_d'], lw=0.8, linestyle='--')

    # ─ inter-card arrow ─
    if idx < len(PARTS) - 1:
        arr_y = top - GAP_C
        aa = fig.add_axes([0.485, arr_y, 0.03, GAP_C])
        aa.set_xlim(0, 1); aa.set_ylim(0, 1); aa.axis('off')
        aa.annotate('', xy=(0.5, 0.0), xytext=(0.5, 1.0),
                    arrowprops=dict(arrowstyle='-|>', color=C['tl'],
                                   lw=3, mutation_scale=18))

# ═══════════════════════════════════════════════════════════════════
# 9. 6-Layer 통합 데이터 파이프라인
# ═══════════════════════════════════════════════════════════════════
BOTTOM_5 = CARD_TOP - 5 * (CARD_H + GAP_C)
PIPE_TITLE_H = 0.015
PIPE_BODY_H = 0.055
PIPE_GAP = 0.003

# section title bar
pt_y = BOTTOM_5 - 0.008
ap_title = ax_(0.02, pt_y, 0.96, PIPE_TITLE_H)
box_(ap_title, 0, 0, 1, 1, C['p4'], 'none', r=0.008)
t_(ap_title, 0.5, 0.50,
   '6-Layer 통합 데이터 파이프라인  |  Integrated Data Pipeline',
   ha='center', va='center', color='white', fontsize=14, fontweight='bold')

# pipeline body
pb_y = pt_y - PIPE_BODY_H - PIPE_GAP
ap = ax_(0.02, pb_y, 0.96, PIPE_BODY_H)
box_(ap, 0, 0, 1, 1, '#FFFBEB', C['p4'], lw=1.5)

LW6 = 1 / 6
for li, (code, name, lc) in enumerate(PIPE_LAYERS):
    lx = li * LW6
    # layer box
    box_(ap, lx + 0.006, 0.08, LW6 - 0.012, 0.84, lc + '20', lc, lw=2.5, r=0.012)
    # code badge
    box_(ap, lx + 0.012, 0.72, 0.055, 0.16, lc, 'none', r=0.005)
    t_(ap, lx + 0.039, 0.80, code,
       ha='center', va='center', color='white', fontsize=10, fontweight='bold')
    # label
    t_(ap, lx + LW6 / 2, 0.40, name,
       ha='center', va='center', color=C['td'], fontsize=11.5,
       fontweight='bold', linespacing=1.35)
    # arrow to next
    if li < 5:
        arr_x = lx + LW6 - 0.003
        ap.annotate('', xy=(arr_x + 0.006, 0.42), xytext=(arr_x - 0.008, 0.42),
                    arrowprops=dict(arrowstyle='-|>', color=lc, lw=2.5,
                                   mutation_scale=14))

# ═══════════════════════════════════════════════════════════════════
# 10. PrPᶜ × KRAS 매트릭스(left) + 6개년 로드맵(right)
# ═══════════════════════════════════════════════════════════════════
SEC2_Y = pb_y - 0.008
SEC2_H = 0.115

# ── 매트릭스 ──
am = ax_(0.02, SEC2_Y - SEC2_H, 0.57, SEC2_H)
box_(am, 0, 0.88, 1, 0.12, C['p5'], 'none', r=0.008)
t_(am, 0.5, 0.94,
   'PrPᶜ × KRAS 위험도 층화 매트릭스  |  Risk Stratification Matrix',
   ha='center', va='center', color='white', fontsize=13, fontweight='bold')

# axis labels
t_(am, 0.28, 0.84, '비-G12C (G12D / G12V 등)',
   ha='center', va='center', color=C['td'], fontsize=11, fontweight='bold')
t_(am, 0.79, 0.84, 'G12C',
   ha='center', va='center', color=C['td'], fontsize=11, fontweight='bold')
t_(am, 0.035, 0.62, 'PrPᶜ\n고발현',
   ha='center', va='center', color=C['td'], fontsize=10.5, fontweight='bold')
t_(am, 0.035, 0.24, 'PrPᶜ\n저발현',
   ha='center', va='center', color=C['td'], fontsize=10.5, fontweight='bold')

for (cx, cy, bg, risk, detail) in CELLS_MAT:
    box_(am, cx, cy, 0.39, 0.35, bg, C['bd'], lw=1.5, r=0.012)
    t_(am, cx + 0.02, cy + 0.27, risk,
       ha='left', va='center', color=C['td'], fontsize=12, fontweight='bold')
    t_(am, cx + 0.02, cy + 0.10, detail,
       ha='left', va='center', color=C['tm'], fontsize=10, linespacing=1.4)

# ── 6개년 로드맵 ──
ao = ax_(0.62, SEC2_Y - SEC2_H, 0.36, SEC2_H)
box_(ao, 0, 0.88, 1, 0.12, C['navy'], 'none', r=0.008)
t_(ao, 0.5, 0.94, '6개년 연구 로드맵  |  6-Year Roadmap',
   ha='center', va='center', color='white', fontsize=13, fontweight='bold')

for ri, (yr, rc, phase, detail) in enumerate(ROADMAP):
    yy = 0.72 - ri * 0.27
    cc2 = plt.Circle((0.07, yy), 0.04, color=rc)
    ao.add_patch(cc2)
    t_(ao, 0.07, yy, str(ri + 1),
       ha='center', va='center', color='white', fontsize=11, fontweight='bold')
    t_(ao, 0.15, yy + 0.06, yr,
       ha='left', va='center', color=rc, fontsize=10, fontweight='bold')
    t_(ao, 0.28, yy + 0.06, phase,
       ha='left', va='center', color=C['td'], fontsize=12, fontweight='bold')
    t_(ao, 0.15, yy - 0.07, detail,
       ha='left', va='center', color=C['tm'], fontsize=9.5, linespacing=1.4)
    # connecting dotted line
    if ri < 2:
        ao.plot([0.07, 0.07], [yy - 0.08, yy - 0.18],
                color=C['bd'], lw=2, linestyle=':')

# ═══════════════════════════════════════════════════════════════════
# 11. 역할 분담
# ═══════════════════════════════════════════════════════════════════
ROLE_TITLE_Y = SEC2_Y - SEC2_H - 0.008
ROLE_BODY_H = 0.075

# section title
ar_title = ax_(0.02, ROLE_TITLE_Y, 0.96, 0.015)
box_(ar_title, 0, 0, 1, 1, C['navy_l'], 'none', r=0.008)
t_(ar_title, 0.5, 0.50,
   '공동연구 역할 분담  |  Collaborative Role Assignment',
   ha='center', va='center', color='white', fontsize=13, fontweight='bold')

role_body_y = ROLE_TITLE_Y - ROLE_BODY_H - 0.003
for ri, (role_name, role_c, items) in enumerate(ROLES):
    rx = 0.02 + ri * 0.49
    rw = 0.47
    ar = ax_(rx, role_body_y, rw, ROLE_BODY_H)
    box_(ar, 0, 0, 1, 1, '#F8FAFC', role_c, lw=2.5)
    # sidebar accent
    box_(ar, 0, 0, 0.018, 1, role_c, 'none', r=0)
    # title
    t_(ar, 0.04, 0.92, role_name,
       ha='left', va='center', color=role_c, fontsize=12, fontweight='bold')
    # items — 5 items, evenly spaced
    for ii, itm in enumerate(items):
        t_(ar, 0.055, 0.78 - ii * 0.155, f'• {itm}',
           ha='left', va='center', color=C['tm'], fontsize=10.5)

# ═══════════════════════════════════════════════════════════════════
# 12. 기대성과 배너
# ═══════════════════════════════════════════════════════════════════
OUT_Y = role_body_y - 0.010
OUT_H = 0.048

ao_out = ax_(0.02, OUT_Y, 0.96, OUT_H)
box_(ao_out, 0, 0, 1, 1, '#F0FDF4', C['p3'], lw=2)
t_(ao_out, 0.5, 0.90, '기대 성과  |  Expected Outcomes',
   ha='center', va='center', color=C['p3'], fontsize=14, fontweight='bold')

outcomes = [
    ('학술',  'SCI 논문 6편+  (Nature Comms · Nature Med · JCO)  |  ASCO · ESMO 학회 발표'),
    ('임상',  'KRAS 고위험군 조기 식별 기준  |  비-G12C PrPᶜ 기반 치료전략 세계 최초 제시  |  혈청 바이오마커 키트 개발'),
    ('산업',  'ADDS CDSS 상용화  |  Pritamab IND 기반  |  PCT 특허 2027 출원  (국내 우선권 2026.01.29)'),
]
for oi, (cat, desc) in enumerate(outcomes):
    yy = 0.72 - oi * 0.22
    t_(ao_out, 0.035, yy, f'[{cat}]',
       ha='left', va='center', color=C['p3'], fontsize=11, fontweight='bold')
    t_(ao_out, 0.09, yy, desc,
       ha='left', va='center', color=C['td'], fontsize=10.5)

# ═══════════════════════════════════════════════════════════════════
# 13. 풋터
# ═══════════════════════════════════════════════════════════════════
FOOTER_Y = OUT_Y - 0.005
aft = ax_(0, FOOTER_Y, 1, 0.010)
box_(aft, 0, 0, 1, 1, '#F1F5F9', 'none', r=0)
aft.plot([0, 1], [0.92, 0.92], color=C['bd'], lw=0.8)
t_(aft, 0.015, 0.42,
   '인하대학교병원 외과 최문석 교수 × ADDS 연구팀  |  ADDS Platform v6.2.0  |  2026. 03. 16. 작성',
   ha='left', va='center', color=C['tl'], fontsize=10)
t_(aft, 0.985, 0.42,
   'Inha University Hospital × AI-Driven Drug Selection (ADDS)',
   ha='right', va='center', color=C['tl'], fontsize=10)

# ═══════════════════════════════════════════════════════════════════
# 14. 저장
# ═══════════════════════════════════════════════════════════════════
OUT_PATH = r'F:\ADDS\figures\KRAS_PrPc_Research_Pipeline_Diagram.png'
plt.savefig(OUT_PATH, dpi=200, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.close()
print(f'[OK] Saved: {OUT_PATH}')
