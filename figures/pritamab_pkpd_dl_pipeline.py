"""
Pritamab PK/PD Deep Learning Pipeline — Visual Figure
임상 데이터 + 세포 형태 → DL → 효능/시너지/독성 예측 → 종합 점수
출력: f:\ADDS\figures\pritamab_pkpd_dl_pipeline.png
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from matplotlib.patheffects import withStroke
import numpy as np

# ── 색상 팔레트 ────────────────────────────────────────────────────
BG      = '#0A0F1E'
C_INPUT = '#1E3A5F'
C_ENC   = '#0E4D6E'
C_FUSE  = '#0D3D4A'
C_HEAD  = '#1A3A1A'
C_OUT   = '#2D1A4A'
C_SCORE = '#4A1A0A'

GOLD  = '#F59E0B'
CYAN  = '#22D3EE'
GREEN = '#34D399'
PURP  = '#A78BFA'
RED   = '#F87171'
BLUE  = '#60A5FA'
WHITE = '#F0F4FF'
GRAY  = '#64748B'

fig = plt.figure(figsize=(22, 14), facecolor=BG)
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 22); ax.set_ylim(0, 14)
ax.axis('off')
ax.set_facecolor(BG)

# ══════════════════════════════════════════════════════════════════
# 헬퍼 함수
# ══════════════════════════════════════════════════════════════════
def box(ax, x, y, w, h, color, alpha=0.85, radius=0.3, lw=1.5, edgecolor=None):
    ec = edgecolor if edgecolor else color
    p = FancyBboxPatch((x, y), w, h,
                       boxstyle=f"round,pad=0.05,rounding_size={radius}",
                       facecolor=color, edgecolor=ec, linewidth=lw, alpha=alpha, zorder=3)
    ax.add_patch(p)
    return p

def txt(ax, x, y, s, size=9, color=WHITE, bold=False, ha='center', va='center', zorder=5):
    weight = 'bold' if bold else 'normal'
    t = ax.text(x, y, s, fontsize=size, color=color, fontweight=weight,
                ha=ha, va=va, zorder=zorder, fontfamily='DejaVu Sans')
    return t

def arrow(ax, x1, y1, x2, y2, color=CYAN, lw=2, style='->', alpha=0.9):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle=style, color=color, lw=lw,
                                connectionstyle='arc3,rad=0.0'),
                zorder=4, alpha=alpha)

def glow_txt(ax, x, y, s, size=11, color=GOLD, bold=True):
    t = ax.text(x, y, s, fontsize=size, color=color, fontweight='bold',
                ha='center', va='center', zorder=6)
    t.set_path_effects([withStroke(linewidth=3, foreground='#00000066')])
    return t

# ══ 배경 그리드 라인 ══
for xg in np.arange(0, 22, 2):
    ax.axvline(xg, color='#1E2A3A', lw=0.4, alpha=0.5)
for yg in np.arange(0, 14, 1):
    ax.axhline(yg, color='#1E2A3A', lw=0.4, alpha=0.5)

# ══════════════════════════════════════════════════════════════════
# 제목
# ══════════════════════════════════════════════════════════════════
glow_txt(ax, 11, 13.4, 'PRITAMAB  Multimodal PK/PD Deep Learning Pipeline', size=16, color=GOLD)
txt(ax, 11, 12.95,
    'Clinical Data + Cell Morphology  →  DL Prediction  →  Efficacy · Synergy · Toxicity · Comprehensive Score',
    size=9.5, color=CYAN)

# ══════════════════════════════════════════════════════════════════
# ZONE A — INPUT (왼쪽)
# ══════════════════════════════════════════════════════════════════
zone_inputs = [
    # (y_center, label, sublabel, color, icon)
    (11.2, 'Clinical PK/PD', 'KD·IC₅₀·CL·t½·Cmin\nddG·α coupling·Rate Red.', '#1A3B5C', '[PK]'),
    (9.0,  'Cell Morphology\n(Cellpose)', 'Cell density · Circularity\nNuclear fragmentation\nPrPc surrogate intensity', '#1A2E3B', '[IMG]'),
    (6.8,  'RNA-seq', 'PRNP↓ · CASP3/9↑ · BCL2↓\n27 Pritamab signature genes\nPCA 200 components', '#1A3B2E', '[RNA]'),
    (4.6,  'CT Imaging', 'Tumour volume · HU density\nShrinkage rate (nnUNet)\nBaseline diameter', '#2E1A3B', '[CT]'),
    (2.4,  'Synergy Data\n(Drug Matrix)', 'Bliss · Loewe · HSA · ZIP\n6×6 conc. grid\nEC50 reduction −24.7%', '#3B2E1A', '[SYN]'),
]

for (yc, lbl, sub, col, icon) in zone_inputs:
    bh = 1.6
    box(ax, 0.3, yc-bh/2, 3.2, bh, col, alpha=0.9, radius=0.25,
        edgecolor='#4A6A8A' if 'PK' in lbl else
                  '#3A8A6A' if 'RNA' in lbl else
                  '#6A4A8A' if 'CT' in lbl else
                  '#8A6A3A' if 'Syn' in lbl else '#3A6A8A')
    txt(ax, 0.9, yc+0.35, icon, size=14, ha='center')
    txt(ax, 1.8, yc+0.35, lbl, size=8.5, color=GOLD if 'PK' in lbl else CYAN, bold=True, ha='left')
    txt(ax, 1.8, yc-0.25, sub, size=7.2, color='#A0B4C8', ha='left', va='center')

txt(ax, 1.9, 12.55, 'INPUT MODALITIES', size=10, color=WHITE, bold=True)
box(ax, 0.15, 1.5, 3.5, 11.3, '#0D1E2E', alpha=0.3, radius=0.4, lw=0.8,
    edgecolor='#2A4A6A')

# ══════════════════════════════════════════════════════════════════
# ZONE B — ENCODER (왼쪽→중앙)
# ══════════════════════════════════════════════════════════════════
enc_data = [
    (11.2, '32-dim', 'Physics-based\nFeature Module', BLUE),
    (9.0,  '128-dim', 'CNN +\nMorphology Enc.', CYAN),
    (6.8,  '256-dim', 'PCA(200) +\nSignature Enc.', GREEN),
    (4.6,  '64-dim',  'nnUNet\nExtractor', PURP),
    (2.4,  '32-dim',  'SynergyFinder\nEncoder', GOLD),
]

for (yc, dim, desc, col) in enc_data:
    box(ax, 4.1, yc-0.65, 2.0, 1.3, C_ENC, alpha=0.9, radius=0.2, lw=1.2, edgecolor=col)
    txt(ax, 5.1, yc+0.2, dim, size=10, color=col, bold=True)
    txt(ax, 5.1, yc-0.2, desc, size=7.2, color='#B0C8D8')
    # 화살표: input → encoder
    arrow(ax, 3.5, yc, 4.1, yc, color=col, lw=1.8)

txt(ax, 5.1, 12.55, 'ENCODER', size=10, color=WHITE, bold=True)
box(ax, 3.95, 1.5, 2.3, 11.3, '#0A1E2E', alpha=0.25, radius=0.3, lw=0.8, edgecolor='#2A4A6A')

# 차원 레이블
txt(ax, 5.1, 0.9, '= 512-dim total input', size=8, color=GRAY)

# ══════════════════════════════════════════════════════════════════
# ZONE C — FUSION MLP (중앙)
# ══════════════════════════════════════════════════════════════════
# 모든 encoder → fusion 화살표
for (yc, dim, desc, col) in enc_data:
    arrow(ax, 6.1, yc, 7.2, 7.0, color=col, lw=1.5, alpha=0.7)

# Fusion MLP 블록
fusion_layers = [
    (7.3, 10.5, 2.8, 1.1, '512-dim  →  Dense Layer', '#1A4A5A', CYAN,  'Concatenated Input\n5 modalities merged'),
    (7.3,  8.8, 2.8, 1.1, 'Dense (256, ReLU + LayerNorm)', '#0E3A4A', BLUE,  'Feature Integration\nCross-modal learning'),
    (7.3,  7.1, 2.8, 1.1, 'Dense (128, ReLU + Dropout 0.3)', '#0E2A3A', PURP, 'Representation\nCompression'),
    (7.3,  5.4, 2.8, 1.1, 'Dense (64, ReLU)', '#1A1A3A', PURP,  'Final Hidden\nRepresentation'),
    (7.3,  3.7, 2.8, 1.1, 'Dense (32) → Output Heads', '#2A1A3A', GOLD,  'Task-specific\nBranching'),
]

for (x, y, w, h, lbl, col, ec, sub) in fusion_layers:
    box(ax, x, y - h/2, w, h, col, alpha=0.92, radius=0.2, lw=1.5, edgecolor=ec)
    txt(ax, x+w/2, y+0.18, lbl, size=8.5, color=ec, bold=True)
    txt(ax, x+w/2, y-0.2, sub, size=7, color='#A0B8C8')

# 레이어 간 화살표
for i in range(len(fusion_layers)-1):
    _, y1, _, h1, *_ = fusion_layers[i]
    _, y2, _, h2, *_ = fusion_layers[i+1]
    arrow(ax, 8.7, y1-h1/2, 8.7, y2+h2/2, color='#4A6A8A', lw=1.5)

txt(ax, 8.7, 12.55, 'FUSION MLP', size=10, color=WHITE, bold=True)
box(ax, 7.1, 1.5, 3.2, 11.3, '#08141E', alpha=0.3, radius=0.35, lw=0.8, edgecolor='#2A4A6A')

# Xavier init 표시
txt(ax, 8.7, 2.8, 'Xavier Init  |  No supervised training\nCalibration-based output scaling', size=7.5, color=GRAY)

# ══════════════════════════════════════════════════════════════════
# ZONE D — OUTPUT HEADS (오른쪽 중앙)
# ══════════════════════════════════════════════════════════════════
# DL branches
arrow(ax, 10.1, 3.7, 11.2, 10.2, color=GREEN,  lw=2, alpha=0.8)
arrow(ax, 10.1, 3.7, 11.2,  7.2, color=GOLD,   lw=2, alpha=0.8)
arrow(ax, 10.1, 3.7, 11.2,  4.2, color=RED,    lw=2, alpha=0.8)

heads = [
    # (x, yc, w, h, label, sublabel, color, edgecolor, activation)
    (11.2, 10.2, 3.4, 2.8,
     '[ EFFICACY HEAD ]', 'Softplus activation',
     '#0D3A1A', GREEN,
     ['mPFS: 14.21 mo  ◆◇', 'mOS:  17.01 mo  ◆◇',
      'ORR:  51.5%     ◆◇', 'DCR:  99.2%     ◆◇',
      'DL HR G12D: 0.965', 'DL HR G12V: 0.888']),
    (11.2, 7.2,  3.4, 2.8,
     '[ SYNERGY HEAD ]', 'Sigmoid activation (0→1)',
     '#2D2A0A', GOLD,
     ['Bliss Score: 17.10±1.32 ◆◇', 'ADDS Consensus: 0.87 ★',
      '5-FU: 0.87  Oxali: 0.89 ★', 'Irino: 0.84  Soto: 0.82 ◆',
      'FOLFOX: 0.84  FOLFIRI: 0.87', 'TAS-102: 0.87  ◆']),
    (11.2, 4.2,  3.4, 2.8,
     '[ TOXICITY HEAD ]', 'Softplus activation',
     '#3A1A1A', RED,
     ['Neutropenia G3/4: 24% ▲', 'Anemia G3/4: 6% ▲',
      'Diarrhea G3/4: 8% ▲', 'Nausea/Vomiting: 10% ▲',
      'Neuropathy: 6% ▲', '(Prit+FOLFOX vs FOLFOX alone)']),
]

for (x, yc, w, h, lbl, act, col, ec, items) in heads:
    box(ax, x, yc-h/2, w, h, col, alpha=0.92, radius=0.25, lw=2.0, edgecolor=ec)
    txt(ax, x+w/2, yc+h/2-0.25, lbl, size=9, color=ec, bold=True)
    txt(ax, x+w/2, yc+h/2-0.55, act, size=7, color=GRAY)
    for j, item in enumerate(items):
        ypos = yc + h/2 - 0.95 - j*0.35
        col_item = WHITE if j < 2 else '#A0B8A8' if 'Efficacy' in lbl or 'SYNERGY' in lbl else '#C8A0A0'
        txt(ax, x+0.25, ypos, f'• {item}', size=7.5, color=col_item, ha='left')

txt(ax, 12.9, 12.55, 'OUTPUT HEADS', size=10, color=WHITE, bold=True)
box(ax, 11.0, 1.5, 3.8, 11.3, '#08100E', alpha=0.25, radius=0.35, lw=0.8, edgecolor='#2A5A3A')

# ══════════════════════════════════════════════════════════════════
# ZONE E — AGGREGATION → COMPREHENSIVE SCORE
# ══════════════════════════════════════════════════════════════════
# 각 head → aggregation
for yc in [10.2, 7.2, 4.2]:
    arrow(ax, 14.6, yc, 15.3, 7.3, color='#5A7A9A', lw=1.8, alpha=0.8)

# Aggregation box
box(ax, 15.3, 6.0, 3.0, 2.6, '#1A2A1A', alpha=0.95, radius=0.3, lw=2, edgecolor='#4ACA80')
txt(ax, 16.8, 7.9, '[ AGGREGATION ]', size=9, color=GREEN, bold=True)
agg_items = [
    ('w₁ × Efficacy score', BLUE),
    ('w₂ × Synergy score', GOLD),
    ('w₃ × (1 − Toxicity)', RED),
    ('─────────────────', GRAY),
]
for j, (item, col) in enumerate(agg_items):
    txt(ax, 16.8, 7.55 - j*0.42, item, size=8, color=col)
txt(ax, 16.8, 6.2, 'Weighted sum + Normalisation', size=7.5, color=GRAY)

# aggregation → final score
arrow(ax, 18.3, 7.3, 19.0, 7.3, color=GOLD, lw=3)

# ── COMPREHENSIVE SCORE ──────────────────────────────────────────
box(ax, 19.0, 5.0, 2.8, 4.6, '#2A1A0A', alpha=0.95, radius=0.35, lw=2.5, edgecolor=GOLD)

# 황금 그라디언트 제목
glow_txt(ax, 20.4, 9.25, '[CS]', size=18, color=GOLD)
glow_txt(ax, 20.4, 8.65, 'Comprehensive', size=10, color=GOLD)
glow_txt(ax, 20.4, 8.3,  'Score (CS)', size=12, color=GOLD)

# 반원 게이지처럼 보이는 CS 시각화
theta = np.linspace(np.pi, 0, 100)
r = 0.85
cx, cy = 20.4, 6.8
# 배경 반원
xs = cx + r * np.cos(theta); ys = cy + r * np.sin(theta)
ax.plot(xs, ys, color='#3A3A3A', lw=6, zorder=5)
# 채워진 부분 (DRS=0.893 → FOLFOX 기준)
frac = 0.893
theta_fill = np.linspace(np.pi, np.pi + frac * np.pi, 100)
xf = cx + r * np.cos(theta_fill); yf = cy + r * np.sin(theta_fill)
ax.plot(xf, yf, color=GOLD, lw=6, zorder=6, solid_capstyle='round')
# 포인터
angle = np.pi + frac * np.pi
ax.annotate('', xy=(cx + r*0.6*np.cos(angle), cy + r*0.6*np.sin(angle)),
            xytext=(cx, cy),
            arrowprops=dict(arrowstyle='->', color=GOLD, lw=2.5), zorder=7)
txt(ax, cx, cy, f'{frac:.3f}', size=13, color=GOLD, bold=True)
txt(ax, cx, cy-0.42, 'DRS (FOLFOX)', size=7, color=GRAY)

# 눈금 레이블
txt(ax, cx-r-0.05, cy, '0.0', size=7, color=GRAY, ha='right')
txt(ax, cx+r+0.05, cy, '1.0', size=7, color=GRAY, ha='left')
txt(ax, cx, cy+r+0.1, '0.5', size=7, color=GRAY)

# CS 상세
box(ax, 19.05, 5.05, 2.7, 0.8, '#1A0E0A', alpha=0.9, radius=0.15, lw=1, edgecolor='#4A3A2A')
txt(ax, 20.4, 5.5, 'Best: FOLFOX  CS=0.893 ◆', size=7.5, color=GOLD, bold=True)
txt(ax, 20.4, 5.2, 'Threshold ≥ 0.75  →  All 7 pass', size=7, color=GREEN)

txt(ax, 20.4, 12.55, 'CS OUTPUT', size=10, color=WHITE, bold=True)
box(ax, 18.8, 1.5, 3.0, 11.3, '#0E0A08', alpha=0.25, radius=0.35, lw=0.8, edgecolor='#4A3A1A')

# ══════════════════════════════════════════════════════════════════
# ZONE F — 하단 피드백 루프
# ══════════════════════════════════════════════════════════════════
box(ax, 3.5, 0.15, 15.0, 0.95, '#0D1E0D', alpha=0.9, radius=0.2, lw=1.5, edgecolor=GREEN)
txt(ax, 11.0, 0.62, '[ FEEDBACK LOOP ]  Clinical Update', size=9, color=GREEN, bold=True)
feedback_steps = ['Phase I PK data', '→  Model recalibration', '→  Phase II design',
                  '→  Real patient NGS/IHC', '→  DL retraining', '→  CS update']
for k, s in enumerate(feedback_steps):
    col = GREEN if '→' not in s else CYAN
    txt(ax, 3.8 + k*2.55, 0.28, s, size=8, color=col)

# 출력에서 피드백으로
ax.annotate('', xy=(18.5, 0.62), xytext=(20.4, 5.0),
            arrowprops=dict(arrowstyle='->', color=GREEN, lw=1.5,
                            connectionstyle='arc3,rad=0.3'),
            zorder=4, alpha=0.7)
ax.annotate('', xy=(3.5, 0.62), xytext=(1.9, 1.5),
            arrowprops=dict(arrowstyle='<-', color=GREEN, lw=1.5,
                            connectionstyle='arc3,rad=-0.3'),
            zorder=4, alpha=0.7)

# ══════════════════════════════════════════════════════════════════
# 범례 / 데이터 등급
# ══════════════════════════════════════════════════════════════════
legend_items = [
    (GOLD, '★  NatureComm Confirmed'),
    (CYAN, '◆  ADDS Calculated'),
    (GREEN,'●  SCI Literature'),
    (PURP, '▲  Energy Model Proj.'),
    (RED,  '◇  DL Synthetic Cohort'),
]
for k, (c, lbl) in enumerate(legend_items):
    box(ax, 0.3 + k*3.4, 12.6, 3.0, 0.38, '#0D1A2A', alpha=0.8, radius=0.1, lw=1, edgecolor=c)
    txt(ax, 1.8 + k*3.4, 12.79, lbl, size=8, color=c, bold=True)

# ══════════════════════════════════════════════════════════════════
# 세로 구분선
# ══════════════════════════════════════════════════════════════════
for xd in [3.7, 6.3, 10.4, 14.7, 18.75]:
    ax.axvline(xd, color='#1E3050', lw=0.8, alpha=0.7, ls='--', zorder=2)

# ── 저장 ──────────────────────────────────────────────────────────
plt.savefig(r'f:\ADDS\figures\pritamab_pkpd_dl_pipeline.png',
            dpi=150, bbox_inches='tight', facecolor=BG,
            edgecolor='none')
plt.close()
print("Saved: f:\\ADDS\\figures\\pritamab_pkpd_dl_pipeline.png")
