"""
figure_research_framework.py
------------------------------
미래도전연구지원사업 추진체계 피규어
- 상단: 6개년 연차별 흐름 (타임라인)
- 중단: 양 팀 역할 분담 (이상훈팀 ↔ 최문석팀)
- 하단: ADDS 연계 흐름 (기초기전→환자군분류→환자유래검증→최적조합)
Nature / Cell 아카데믹 스타일
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.lines import Line2D
import numpy as np

plt.rcParams.update({
    'font.family': 'Malgun Gothic',  # 한글 지원
    'axes.unicode_minus': False,
    'font.size': 9,
    'figure.facecolor': 'white',
})

# ── 컬러 팔레트 ────────────────────────────────────────────────────────────────
C = {
    'navy':    '#1A2F5E',
    'blue':    '#2C5F9E',
    'lblue':   '#4A90D9',
    'teal':    '#1B7A6B',
    'lteal':   '#3BAA94',
    'orange':  '#D4541A',
    'lorange': '#F07840',
    'purple':  '#5B3F8A',
    'lpurple': '#8A6FC0',
    'gold':    '#C8960A',
    'lgold':   '#E8BC2A',
    'red':     '#B33030',
    'grey':    '#6B7280',
    'lgrey':   '#E8ECEF',
    'white':   '#FFFFFF',
    'bg':      '#F7F8FA',
}

# ════════════════════════════════════════════════════════════════════════════
# Figure 설정: 단일 axes (전체를 patches로 구성)
# ════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(18, 13))
ax.set_xlim(0, 18)
ax.set_ylim(0, 13)
ax.axis('off')
ax.set_facecolor(C['bg'])
fig.patch.set_facecolor(C['bg'])

# ── 헬퍼 함수 ─────────────────────────────────────────────────────────────────
def box(x, y, w, h, fc, ec, lw=1.2, alpha=1.0, radius=0.15):
    r = FancyBboxPatch((x, y), w, h,
                        boxstyle=f'round,pad={radius}',
                        facecolor=fc, edgecolor=ec,
                        linewidth=lw, alpha=alpha, zorder=3)
    ax.add_patch(r)
    return r

def text(x, y, s, size=9, color='#1A1A2A', ha='center', va='center',
         bold=False, wrap=False):
    fw = 'bold' if bold else 'normal'
    ax.text(x, y, s, fontsize=size, color=color, ha=ha, va=va,
            fontweight=fw, zorder=4,
            multialignment='center')

def arrow(x1, y1, x2, y2, color='#555', lw=1.5, style='->', head=8):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle=f'->', color=color,
                                lw=lw, mutation_scale=head),
                zorder=5)

def biArrow(x1, y1, x2, y2, color='#555', lw=1.5):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='<->', color=color,
                                lw=lw, mutation_scale=10),
                zorder=5)

def hline(x1, x2, y, color='#ccc', lw=0.8, ls='--'):
    ax.plot([x1, x2], [y, y], color=color, lw=lw, ls=ls, zorder=2)

# ════════════════════════════════════════════════════════════════════════════
# SECTION 0 — 전체 제목
# ════════════════════════════════════════════════════════════════════════════
box(0.2, 12.0, 17.6, 0.80, C['navy'], C['navy'], radius=0.12)
text(9.0, 12.40,
     '연구개발과제 추진체계',
     size=15, color='white', bold=True)
text(9.0, 12.05,
     'KRAS 변이 대장암 × PrPᶜ 기반 정밀치료 표적 발굴 및 병용치료 전략 제시 (6개년)',
     size=9, color='#C7D2FE')

# ════════════════════════════════════════════════════════════════════════════
# SECTION 1 — 연차별 단계 흐름 (상단, y=9.8~11.7)
# ════════════════════════════════════════════════════════════════════════════
# 배경 패널
box(0.2, 9.55, 17.6, 2.25, '#F0F4FF', C['blue'], lw=0.8, radius=0.1)
text(1.55, 11.55, '단계별 연차 흐름', size=9.5, color=C['blue'], bold=True)

PHASES = [
    ('1–3차년도', '세포주 기반\n기전 규명',
     ['PrPc 종양증식·암줄기세포성·침윤 기전\nKRAS 하위 신호전달·저산소 연계 기전\n항암제 내성 획득 기전 규명 (siRNA)\nADDS 기반 병용 조합 우선순위화'],
     C['blue'], C['lblue']),
    ('4차년도', '환자 검체\n기반 검증',
     ['KRAS 변이형 분석 (ddPCR/NGS)\nPrPc 발현 기반 환자군 분류 (IHC)\n환자 유래 PDO 구축\n세포주 결과의 재현성 검증'],
     C['teal'], C['lteal']),
    ('5–6차년도', 'ADDS 기반\n최적화·검증',
     ['분자·약물반응 정보 -> ADDS 입력\n최적 항암제 병용 조합 도출\nPDO 기반 항암 작용 기전 검증\nPrPc 치료표적 타당성 최종 평가'],
     C['purple'], C['lpurple']),
]

phase_xs = [0.45, 6.35, 12.25]
phase_w  = 5.60

for xi, (yr, phase_title, details, fc, lfc) in zip(phase_xs, PHASES):
    # 헤더 박스
    box(xi, 10.40, phase_w, 0.95, fc, fc, radius=0.1)
    text(xi + phase_w/2, 11.05, yr,    size=9, color='white', bold=True)
    text(xi + phase_w/2, 10.65, phase_title, size=8.5, color='#E0E8FF')

    # 내용 박스
    box(xi, 9.70, phase_w, 0.68, 'white', lfc, lw=0.9, radius=0.08)
    text(xi + phase_w/2, 10.04, details[0],
         size=7.8, color='#2D3748')

# 연결 화살표
for xi in [phase_xs[0] + phase_w, phase_xs[1] + phase_w]:
    arrow(xi + 0.02, 10.88, xi + 0.28, 10.88, color=C['grey'], lw=2, head=10)

# ════════════════════════════════════════════════════════════════════════════
# SECTION 2 — 양 팀 역할 분담 (중단, y=5.6~9.3)
# ════════════════════════════════════════════════════════════════════════════
# 배경 패널
box(0.2, 5.35, 17.6, 4.00, '#FFF8F5', C['orange'], lw=0.8, radius=0.1)
text(1.55, 9.15, '연구팀 역할 분담', size=9.5, color=C['orange'], bold=True)

# ── 이상훈팀 (왼쪽) ──────────────────────────────────────────────────────────
box(0.40, 5.55, 7.60, 3.40, '#EFF6FF', C['blue'], lw=1.4, radius=0.12)
text(4.20, 8.70, '연구책임자팀  |  이상훈 부교수', size=10, color=C['navy'], bold=True)
text(4.20, 8.40, '인하대학교 의과대학 의생명학교실 · 세포생리학', size=8, color=C['grey'])

tasks_L = [
    ('[A]', '기초 기전·전임상 실험',
     'PrPc 기능·신호전달 기전 분석\n(세포주·오가노이드)'),
    ('[B]', '약물반응 평가',
     '병용 조합 세포 독성·IC50·AUC\n(내성세포군 유도 포함)'),
    ('[C]', 'ADDS 적용·검증',
     '약물반응 데이터 -> ADDS 입력\n최적 조합 결과의 실험적 검증'),
]
ty = 7.95
for icon, title, detail in tasks_L:
    box(0.55, ty - 0.52, 7.30, 0.60, 'white', C['lblue'], lw=0.8, radius=0.08)
    text(1.25, ty - 0.22, icon + ' ' + title, size=8.5, color=C['blue'],
         bold=True, ha='left')
    text(5.10, ty - 0.22, detail, size=7.8, color='#374151', ha='center')
    ty -= 0.72

# ── 최문석팀 (오른쪽) ─────────────────────────────────────────────────────────
box(10.0, 5.55, 7.60, 3.40, '#F0FBF7', C['teal'], lw=1.4, radius=0.12)
text(13.80, 8.70, '공동연구원팀  |  최문석 조교수', size=10, color=C['navy'], bold=True)
text(13.80, 8.40, '인하대학교 의과대학 외과학교실 · 외과학', size=8, color=C['grey'])

tasks_R = [
    ('[D]', '임상 검체 확보',
     '내시경·수술 검체 확보\n원발암·전이성 병변 모두 포함'),
    ('[E]', '병리·유전체 분석',
     'KRAS 변이형 분석 (ddPCR/NGS)\nPrPc IHC 평가'),
    ('[F]', '임상정보 관리',
     '임상정보 정리·환자군 분류\n추적자료 구축·표준화'),
]
ty = 7.95
for icon, title, detail in tasks_R:
    box(10.15, ty - 0.52, 7.30, 0.60, 'white', C['lteal'], lw=0.8, radius=0.08)
    text(10.95, ty - 0.22, icon + ' ' + title, size=8.5, color=C['teal'],
         bold=True, ha='left')
    text(14.80, ty - 0.22, detail, size=7.8, color='#374151', ha='center')
    ty -= 0.72

# ── 중앙 교류 화살표 + 협력 박스 ────────────────────────────────────────────────
box(8.15, 5.95, 1.70, 2.60, '#FEF3C7', C['gold'], lw=1.4, radius=0.12)
text(9.00, 7.60, '정기\n공유', size=8.5, color=C['gold'], bold=True)
text(9.00, 7.10, '> 검체 현황\n> 병리 결과\n> 실험 결과\n> 조합 도출', size=7.2, color='#374151')

biArrow(8.01, 7.25, 8.13, 7.25, color=C['blue'], lw=2)
biArrow(9.87, 7.25, 10.00, 7.25, color=C['teal'], lw=2)

# ════════════════════════════════════════════════════════════════════════════
# SECTION 3 — ADDS 플랫폼 중심 통합 흐름 (하단, y=1.3~5.1)
# ════════════════════════════════════════════════════════════════════════════
# 배경 패널
box(0.2, 1.10, 17.6, 4.05, '#F5F0FF', C['purple'], lw=0.8, radius=0.1)
text(1.55, 4.90, 'ADDS 기반 통합 연구 흐름', size=9.5, color=C['purple'], bold=True)

# ADDS 중앙 박스
box(7.30, 2.40, 3.40, 1.80, C['purple'], C['purple'], radius=0.15)
text(9.00, 3.58, 'ADDS', size=13, color='white', bold=True)
text(9.00, 3.22, 'AI 기반 다중모달\n통합 분석 플랫폼', size=8.5, color='#E0D8FF')
text(9.00, 2.72, '약물반응 예측\n병용 조합 최적화', size=7.8, color='#C4B8FF')

# 4개 흐름 박스 (왼쪽 2 + 오른쪽 2)
FLOW = [
    (1.00, 2.60, '[1]\n기초 기전\n규명',
     '세포주 PrPc 발현·기능\n신호전달 기전\n내성 기전',
     C['blue'], C['lblue'], 'left'),
    (4.00, 2.60, '[2]\n환자군\n분류',
     'KRAS 변이형 분석\nPrPc IHC 평가\n병리·임상 연계',
     C['teal'], C['lteal'], 'left'),
    (12.10, 2.60, '[3]\n최적 조합\n도출',
     'ADDS 입력·분석\n병용 우선순위화\n농도·간격 최적화',
     C['orange'], C['lorange'], 'right'),
    (15.10, 2.60, '[4]\nPDO 기반\n검증',
     'PDO 약물반응 평가\n치료표적 타당성\n전임상 근거 확보',
     C['purple'], C['lpurple'], 'right'),
]

for fx, fy, title, detail, fc, lfc, side in FLOW:
    box(fx, fy, 2.80, 1.80, 'white', lfc, lw=1.2, radius=0.12)
    # 헤더 스트립
    box(fx, fy + 1.15, 2.80, 0.65, lfc, lfc, radius=0.08)
    text(fx + 1.40, fy + 1.47, title, size=9, color='white', bold=True)
    text(fx + 1.40, fy + 0.58, detail, size=7.8, color='#374151')

    # 화살표
    if side == 'left':
        arrow(fx + 2.83, fy + 1.06, 7.26, fy + 1.06,
              color=fc, lw=1.8, head=9)
    else:
        arrow(10.74, fy + 1.06, fx - 0.03, fy + 1.06,
              color=fc, lw=1.8, head=9)

# 흐름 연결 (1→2, 3→4) 상단 브릿지
ax.annotate('', xy=(3.97, 3.50), xytext=(3.83, 3.50),
            arrowprops=dict(arrowstyle='->', color=C['grey'], lw=1.3, mutation_scale=9))

# 수직 통합 화살표 (섹션2 → 섹션3 ADDS 흐름)
arrow(4.20, 5.37, 4.20, 4.43, color=C['teal'], lw=1.5, head=8)
text(4.60, 4.90, '환자 검체\n데이터', size=7.5, color=C['teal'], ha='left')

arrow(13.80, 5.37, 13.80, 4.43, color=C['blue'], lw=1.5, head=8)
text(14.20, 4.90, '세포주·약물\n반응 데이터', size=7.5, color=C['blue'], ha='left')

# ── 데이터 표준화 공유 흐름 (하단 띠) ──────────────────────────────────────────
box(0.40, 1.20, 17.20, 0.60, '#EDE9FF', C['purple'], lw=0.8, radius=0.08)
text(9.00, 1.50,
     '[연결]  검체-병리-분자-약물반응 정보를 표준화된 형식으로 축적·공유  -->  기초기전 규명 > 환자군 분류 > 환자 유래 검증 > 최적 약물조합 도출이 하나의 연속된 공동연구 체계로 운영',
     size=8, color=C['purple'], bold=False)

# ════════════════════════════════════════════════════════════════════════════
# 범례
# ════════════════════════════════════════════════════════════════════════════
legend_items = [
    mpatches.Patch(color=C['blue'],   label='이상훈팀 (기초·전임상)'),
    mpatches.Patch(color=C['teal'],   label='최문석팀 (임상·병리)'),
    mpatches.Patch(color=C['purple'], label='ADDS 플랫폼'),
    mpatches.Patch(color=C['orange'], label='병용 조합 최적화'),
    mpatches.Patch(color=C['gold'],   label='정기 공유·협력'),
]
ax.legend(handles=legend_items, loc='lower right',
          bbox_to_anchor=(0.995, 0.005),
          fontsize=8, framealpha=0.92, edgecolor='#cccccc',
          ncol=5, handlelength=1.2, columnspacing=0.8)

# ── 출처 ──────────────────────────────────────────────────────────────────────
ax.text(0.35, 0.02,
        '2026년도 미래도전연구지원사업 신규과제 연구계획서 기반  |  인하대학교 의과대학 의생명학교실 이상훈',
        fontsize=7.5, color='#888', transform=ax.transAxes, va='bottom')

# ── 저장 ──────────────────────────────────────────────────────────────────────
out = r'f:\ADDS\docs\research_framework_figure.png'
plt.savefig(out, dpi=250, bbox_inches='tight', facecolor=C['bg'])
plt.close()
print(f"Saved: {out}")
