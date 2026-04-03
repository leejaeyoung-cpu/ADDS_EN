#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
KRAS × PrPc 연구 파이프라인 인포그래픽 v3 (최종)
- 세로 확장 (FIG_H=36), 카드 높이 충분히 확보
- 3컬럼 불릿 텍스트 완전 표시
- 흰 배경, 전문 의학 인포그래픽 스타일
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import warnings
warnings.filterwarnings('ignore')

import matplotlib.font_manager as fm

def get_korean_font():
    for c in ['Malgun Gothic','NanumGothic','AppleGothic','Noto Sans CJK KR']:
        if c in {f.name for f in fm.fontManager.ttflist}:
            return c
    for f in fm.fontManager.ttflist:
        if any(k in f.name.lower() for k in ['malgun','nanum','gothic']):
            return f.name
    return None

KF = get_korean_font()
if KF:
    plt.rcParams['font.family'] = KF
plt.rcParams['axes.unicode_minus'] = False

C = {
    'bg':  '#FFFFFF', 'navy':'#1A2E4A',
    'p1':  '#1D4ED8', 'p2':'#6D28D9', 'p3':'#047857',
    'p4':  '#B45309', 'p5':'#B91C1C',
    'bd':  '#CBD5E1', 'tl':'#94A3B8',
    'td':  '#1E293B', 'tm':'#334155',
}

PARTS = [
  {'n':1,'c':C['p1'],'lc':'#DBEAFE',
   'title':'KRAS 변이 대장암의 임상적 한계','en':'Clinical Limitations',
   'cols':[
     ('임상 이질성 4축',
      '• 전이 양상 / 재발 위험\n• 항암제 반응 / 내성 시점\n  환자마다 현격히 다름'),
     ('비-G12C 치료 공백',
      '• KRAS 변이 대장암의 ~80%\n  G12D/G12V — 승인 표적치료 不在\n• Sotorasib은 G12C만 적용'),
     ('ADDS 코호트 수치',
      '• G12D ORR 58%\n• G12V ORR 55%\n• G12C ORR 49%  (9%p 편차)'),
   ]},
  {'n':2,'c':C['p2'],'lc':'#EDE9FE',
   'title':'기존 KRAS 분류의 설명력 한계','en':'Classification Gap',
   'cols':[
     ('공백 A — 암줄기세포성',
      '• CD44+/CD133+ 비율 차이\n  분류에 미반영\n• 재발 시점과 무관'),
     ('공백 B — 내성 획득 속도',
      '• 동일 레지멘(FOLFOX) 환자 간\n  1차 내성 시점 분산\n• 결정 인자 불명'),
     ('공백 C — 전이 경로',
      '• 간·폐·복막 전이 초발 경로\n  선택 예측 불가\n• CT 패턴 분류도 부재'),
   ]},
  {'n':3,'c':C['p3'],'lc':'#D1FAE5',
   'title':'PrPᶜ 기반 고위험 진행 양상 가설','en':'PrPᶜ Hypothesis',
   'cols':[
     ('시그날로솜 기전',
      '• PrPᶜ–RPSA 복합체 활성\n• RAS-GTP 로딩 −42%\n• pERK −38% / Caspase-3 +280%'),
     ('TCGA 조직-혈청 역설',
      '• 5개 암종 2,285건 분석\n• 조직 mRNA↓ ↔ 혈청 단백질↑\n• ADAM10/17 shedding 규명'),
     ('바이오마커 전략 전환',
      '• 조직 mRNA 대리지표 폐기\n• 혈청 PrPᶜ 직접 ELISA 정량\n• 비침습·동적 추적 가능'),
   ]},
  {'n':4,'c':C['p4'],'lc':'#FEF3C7',
   'title':'생검–병리–임상–약물반응 통합','en':'6-Layer Data Pipeline',
   'cols':[
     ('검체 레이어 (L1–L3)',
      '• 대장내시경 생검 (기저값)\n• 수술 검체 (공간 이질성)\n• FFPE IHC H-score 정량'),
     ('CT 방사선 레이어 (L4)',
      '• TotalSegmentator 117장기\n• 처리 15.67초/환자  ±7mm\n• RECIST 1.1 자동 평가'),
     ('ADDS 융합 레이어 (L5–L6)',
      '• PDO 약물반응 IC50\n• 4모달 MLP  PFS R²=0.812\n• 이중 추론 엔진 MDT 보고서'),
   ]},
  {'n':5,'c':C['p5'],'lc':'#FEE2E2',
   'title':'위험도 층화 및 치료전략 도출','en':'Risk Stratification & Strategy',
   'cols':[
     ('PrPᶜ × KRAS 층화 매트릭스',
      '🔴 고위험: 비-G12C + 고발현\n🟡 중위험: 어느 한쪽\n🟢 저위험: G12C + 저발현'),
     ('Pritamab 병용 효과',
      '• ORR 51.5% vs 대조 24.0%\n    (+27.5%p 절대 개선)\n• mOS +2.87개월'),
     ('동적 재층화 알고리즘',
      '• 혈청 PrPᶜ 3개월 간격 재측정\n• ctDNA VAF 2주 간격 추적\n• 위험 등급 자동 갱신'),
   ]},
]

PIPE_LAYERS = [
    ('L1','대장내시경\n생검', '#3B82F6'),
    ('L2','수술\n검체',    '#6366F1'),
    ('L3','FFPE\nH-score','#8B5CF6'),
    ('L4','CT / ADDS\n파이프라인','#D97706'),
    ('L5','PDO\n약물반응', '#10B981'),
    ('L6','ADDS 4모달\n융합 MLP', '#DC2626'),
]

CELLS_MAT = [
    (0.12,0.50,'#FEE2E2','🔴 고위험',  '비-G12C + PrPᶜ 고발현\nPritamab 병용 / ctDNA 집중'),
    (0.57,0.50,'#FEF9C3','🟡 중위험-B','G12C + PrPᶜ 고발현\nSoto/Adagra + 항-PrPᶜ 검토'),
    (0.12,0.10,'#FEF9C3','🟡 중위험-A','비-G12C + PrPᶜ 저발현\nFOLFOX 우선 3개월 재측정'),
    (0.57,0.10,'#DCFCE7','🟢 저위험',  'G12C + PrPᶜ 저발현\nSotorasib 단독'),
]

ROADMAP = [
    ('Year 1–2', C['p1'], '기반 구축', 'IRB 승인  ·  레트로 코호트 N=200\n절단값 결정  ·  PDO 인프라'),
    ('Year 3–4', C['p2'], '기전 검증', 'PrPᶜ-RPSA 논문 (NatComm)\n전향 파일럿 N=100'),
    ('Year 5–6', C['p5'], '임상 적용', 'Nature Medicine 최종 제출\n다기관 N=300  ·  PCT 특허'),
]

# ═══════════════════════════════════════
FIG_W, FIG_H = 26, 40
fig = plt.figure(figsize=(FIG_W, FIG_H), facecolor='white')

def ax_(x,y,w,h):
    a = fig.add_axes([x,y,w,h])
    a.set_xlim(0,1); a.set_ylim(0,1); a.axis('off')
    return a

def box_(ax, x,y,w,h, fc,ec, lw=1.5, r=0.015):
    ax.add_patch(FancyBboxPatch((x,y),w,h,
        boxstyle=f'round,pad={r}', facecolor=fc, edgecolor=ec, linewidth=lw))

def t_(ax,x,y,s,**kw):
    ax.text(x,y,s,**kw)

# ─── HEADER ───────────────────────────
ah = ax_(0, 0.953, 1, 0.047)
box_(ah,0,0,1,1, C['navy'],'none',r=0)
t_(ah,0.5,0.67,'KRAS 변이 대장암의 치료 한계와 PrPᶜ 기반 고위험 진행 양상 연구',
   ha='center',va='center',color='white',fontsize=22,fontweight='black')
t_(ah,0.5,0.22,
   'Therapeutic Limitations of KRAS-Mutant CRC & PrPᶜ-Based High-Risk Progression Study    |    인하대학교병원 외과 최문석 교수  ×  ADDS 연구팀',
   ha='center',va='center',color='#93C5FD',fontsize=11)

# ─── 배경 배너 ────────────────────────
ab = ax_(0.02, 0.915, 0.96, 0.034)
box_(ab,0,0,1,1, '#EFF6FF',C['p1'],lw=1.8)
t_(ab,0.012,0.55,'연구 배경 |',ha='left',va='center',
   color=C['p1'],fontsize=12,fontweight='bold')
t_(ab,0.105,0.55,
   'KRAS 변이 대장암의 임상적 이질성이 기존 분류만으로는 충분히 설명되지 않으며, '
   'PrPᶜ가 암줄기세포성·다약제 내성·침윤·전이를 연결하는 병태생리 축으로 작동할 가능성을 검증한다.',
   ha='left',va='center',color=C['td'],fontsize=10.5)

# ─── PART 카드 ────────────────────────
CARD_TOP = 0.908
CARD_H   = 0.098   # 높이 충분히
GAP_C    = 0.006

for idx,P in enumerate(PARTS):
    top = CARD_TOP - (idx+1)*(CARD_H+GAP_C)
    ac = ax_(0.02, top, 0.96, CARD_H)

    box_(ac,0,0,1,1, '#FAFCFF',P['c'],lw=2.2)
    # 사이드바
    box_(ac,0,0,0.036,1, P['c'],'none',r=0)
    # 번호 배지
    cc = plt.Circle((0.018,0.50),0.15,color='white',zorder=5)
    ac.add_patch(cc)
    t_(ac,0.018,0.50,str(P['n']),
       ha='center',va='center',color=P['c'],fontsize=18,fontweight='black',zorder=6)

    # PART 태그
    box_(ac,0.044,0.70,0.075,0.22, P['lc'],P['c'],lw=1,r=0.006)
    t_(ac,0.081,0.81,f"PART {P['n']}",
       ha='center',va='center',color=P['c'],fontsize=9,fontweight='bold')

    # 제목 / 영문
    t_(ac,0.044,0.52,P['title'],
       ha='left',va='center',color=P['c'],fontsize=14,fontweight='bold')
    t_(ac,0.044,0.17,P['en'],
       ha='left',va='center',color=C['tl'],fontsize=10,style='italic')

    # 수직선
    ac.plot([0.28,0.28],[0.04,0.96],color='#E2E8F0',lw=1.2)

    # 3컬럼 불릿
    XS  = [0.295, 0.555, 0.815]
    DXS = [0.24,  0.24,  0.17 ]
    for bi,(bx,(bttl,btxt)) in enumerate(zip(XS,P['cols'])):
        # 헤더 띠
        box_(ac, bx, 0.76, DXS[bi], 0.19, P['lc'],'none',r=0.004)
        t_(ac, bx+DXS[bi]/2, 0.855, bttl,
           ha='center',va='center',color=P['c'],fontsize=9,fontweight='bold')
        # 내용
        t_(ac, bx+0.012, 0.70, btxt,
           ha='left',va='top',color=C['tm'],fontsize=9.2,linespacing=1.5)
        # 구분 점선
        if bi < 2:
            ax_x = bx + DXS[bi] + 0.01
            ac.plot([ax_x,ax_x],[0.04,0.96],color='#E8EDF3',lw=0.8,linestyle='--')

    # PART 간 화살표 공간 (별도 ax)
    if idx < len(PARTS)-1:
        arr_y = top - GAP_C
        aa = fig.add_axes([0.495, arr_y, 0.01, GAP_C])
        aa.set_xlim(0,1); aa.set_ylim(0,1); aa.axis('off')
        aa.annotate('',xy=(0.5,0.05),xytext=(0.5,0.95),
            arrowprops=dict(arrowstyle='-|>',color='#94A3B8',lw=2.2,mutation_scale=14))

# ─── 6-LAYER 파이프라인 ───────────────
# 5개 카드 맨 아래 위치 계산
BOTTOM_5 = CARD_TOP - 5*(CARD_H+GAP_C)
PIPE_Y = BOTTOM_5 - 0.005
PIPE_H = 0.135

ap = ax_(0.02, PIPE_Y - PIPE_H, 0.96, PIPE_H)
box_(ap,0,0.855,1,0.145, C['p4'],'none',r=0.008)
t_(ap,0.5,0.927,'6-Layer 통합 데이터 파이프라인',
   ha='center',va='center',color='white',fontsize=13,fontweight='bold')

LW6 = 1/6
for li,(code,name,lc) in enumerate(PIPE_LAYERS):
    lx = li*LW6
    box_(ap, lx+0.006, 0.04, LW6-0.013, 0.77, lc+'18', lc, lw=2, r=0.01)
    box_(ap, lx+0.022, 0.74, 0.055, 0.09, lc,'none',r=0.004)
    t_(ap, lx+0.049, 0.785, code,
       ha='center',va='center',color='white',fontsize=9,fontweight='bold')
    t_(ap, lx+LW6/2, 0.44, name,
       ha='center',va='center',color=C['td'],fontsize=10,fontweight='bold',linespacing=1.35)
    if li < 5:
        ap.annotate('',xy=(lx+LW6-0.004,0.44),xytext=(lx+LW6-0.016,0.44),
            arrowprops=dict(arrowstyle='-|>',color=lc,lw=2.2,mutation_scale=11))

# ─── 위험도 매트릭스 ──────────────────
MAT_Y = PIPE_Y - PIPE_H - 0.01
MAT_H = 0.13

am = ax_(0.02, MAT_Y - MAT_H, 0.57, MAT_H)
box_(am,0,0.87,1,0.13, C['p5'],'none',r=0.008)
t_(am,0.5,0.935,'PrPᶜ × KRAS 위험도 층화 매트릭스',
   ha='center',va='center',color='white',fontsize=12,fontweight='bold')

t_(am,0.28,0.85,'비-G12C (G12D / G12V 등)',ha='center',va='center',color=C['td'],fontsize=10,fontweight='bold')
t_(am,0.79,0.85,'G12C',ha='center',va='center',color=C['td'],fontsize=10,fontweight='bold')
t_(am,0.03,0.65,'PrPᶜ\n고발현',ha='center',va='center',color=C['td'],fontsize=9.5,fontweight='bold')
t_(am,0.03,0.25,'PrPᶜ\n저발현',ha='center',va='center',color=C['td'],fontsize=9.5,fontweight='bold')

for (cx,cy,bg,risk,detail) in CELLS_MAT:
    box_(am,cx,cy,0.39,0.33, bg,'#CBD5E1',lw=1.2,r=0.012)
    t_(am,cx+0.02,cy+0.25,risk,ha='left',va='center',color=C['td'],fontsize=10,fontweight='bold')
    t_(am,cx+0.02,cy+0.10,detail,ha='left',va='center',color=C['tm'],fontsize=8.5,linespacing=1.3)

# ─── 6개년 로드맵 ─────────────────────
ao = ax_(0.62, MAT_Y - MAT_H, 0.36, MAT_H)
box_(ao,0,0.87,1,0.13, C['navy'],'none',r=0.008)
t_(ao,0.5,0.935,'6개년 연구 로드맵',
   ha='center',va='center',color='white',fontsize=12,fontweight='bold')

for ri,(yr,rc,phase,detail) in enumerate(ROADMAP):
    yy = 0.72 - ri*0.28
    cc2 = plt.Circle((0.07,yy+0.07),0.04,color=rc)
    ao.add_patch(cc2)
    t_(ao,0.07,yy+0.07,str(ri+1),
       ha='center',va='center',color='white',fontsize=9.5,fontweight='bold')
    t_(ao,0.15,yy+0.14,yr,ha='left',va='center',color=rc,fontsize=9,fontweight='bold')
    t_(ao,0.15,yy+0.05,phase,ha='left',va='center',color=C['td'],fontsize=10,fontweight='bold')
    t_(ao,0.15,yy-0.06,detail,ha='left',va='center',color=C['tm'],fontsize=8.5,linespacing=1.3)
    if ri < 2:
        ao.plot([0.07,0.07],[yy-0.015,yy+0.027],color='#CBD5E1',lw=1.5,linestyle=':')

# ─── 풋터 ─────────────────────────────
af = ax_(0,0,1,0.008)
box_(af,0,0,1,1,'#F1F5F9','none',r=0)
af.plot([0,1],[0.92,0.92],color='#CBD5E1',lw=0.7)
t_(af,0.012,0.42,
   '인하대학교병원 외과 최문석 교수  ×  ADDS 연구팀  |  ADDS Platform v6.2.0  |  2026. 03. 16',
   ha='left',va='center',color=C['tl'],fontsize=9)
t_(af,0.988,0.42,
   'Inha University Hospital  ×  AI-Driven Drug Selection (ADDS)',
   ha='right',va='center',color=C['tl'],fontsize=9)

# ─── 저장 ─────────────────────────────
OUT = r'f:\ADDS\Fig1_KRAS_PrPc_Infographic.png'
plt.savefig(OUT, dpi=200, bbox_inches='tight', facecolor='white', edgecolor='none')
plt.close()
print(f'[OK] {OUT}')
