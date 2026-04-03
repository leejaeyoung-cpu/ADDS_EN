"""
prpc_cell_organic.py
=====================
논문용 PrPC 시그널 패스웨이 — 세포 단면 유기 일러스트레이션
- 세포 자체를 타원형으로 형상화 (세포막 / 세포질 / 핵 구획)
- 단백질 = 텍스트 레이블만
- 신호 = 두꺼운 유기 Bezier 곡선 (세포 내부를 흐르듯)
- 흰 배경, 논문 품질 300 dpi
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Ellipse, FancyArrowPatch, PathPatch
from matplotlib.path import Path
import matplotlib.patheffects as pe
from matplotlib.colors import to_rgba

OUT = r"f:\ADDS\outputs\pritamab_pptx_figures\prpc_cell_organic.png"

# ══════════════════════════════════════════════════════════════════════
# ①  캔버스
# ══════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(20, 16), dpi=150)
fig.patch.set_facecolor('white')
ax.set_facecolor('white')
ax.set_xlim(-10.5, 10.5)
ax.set_ylim(-9.5, 8.5)
ax.set_aspect('equal')
ax.axis('off')

# ══════════════════════════════════════════════════════════════════════
# ②  세포 구획 — 유기 타원 레이어 (뒤에서 앞으로 적층)
# ══════════════════════════════════════════════════════════════════════

# ─ 세포외 공간 배경 ─ (흰색)
ax.add_patch(Ellipse((0, -0.5), 18.5, 15.5,
                      facecolor='white', edgecolor='none', alpha=1.0, zorder=0))

# ─ 세포막 (두꺼운 이중 테두리) ─
ax.add_patch(Ellipse((0, -0.5), 17.8, 14.8,
                      facecolor='white', edgecolor='#1565C0', linewidth=3.0,
                      linestyle='-', alpha=1.0, zorder=1))
ax.add_patch(Ellipse((0, -0.5), 16.8, 13.8,
                      facecolor='white', edgecolor='#1565C0', linewidth=1.2,
                      linestyle='--', alpha=0.50, zorder=2))

# ─ 세포질 ─
ax.add_patch(Ellipse((0, -0.8), 15.8, 12.8,
                      facecolor='white', edgecolor='none', alpha=1.0, zorder=3))

# ─ 소포체/골지 힌트 (중간 유기 blob) ─
from matplotlib.patches import FancyBboxPatch
for cx, cy, rx, ry, col in [
    (-4.5, 1.0, 2.0, 0.7, '#E1F5FE'),
    ( 4.0, 0.5, 1.5, 0.6, '#F3E5F5'),
]:
    ax.add_patch(Ellipse((cx, cy), rx*2, ry*2,
                          facecolor=col, edgecolor='#B0BEC5',
                          linewidth=0.5, alpha=0.50, zorder=4))

# ─ 핵막 (이중 타원) ─
ax.add_patch(Ellipse((0, -4.8), 9.0, 4.8,
                      facecolor='#FFF8E1', edgecolor='#EF6C00',
                      linewidth=2.5, linestyle='-', alpha=0.70, zorder=5))
ax.add_patch(Ellipse((0, -4.8), 8.0, 4.0,
                      facecolor='#FFFDE7', edgecolor='#EF6C00',
                      linewidth=1.0, linestyle='--', alpha=0.50, zorder=6))

# 핵공 힌트
for ang in np.linspace(0, 2*np.pi, 10, endpoint=False):
    nx = 4.0*np.cos(ang); ny = 2.0*np.sin(ang) - 4.8
    ax.add_patch(plt.Circle((nx, ny), 0.15,
                              facecolor='#EF6C00', alpha=0.25, zorder=7))

# ══════════════════════════════════════════════════════════════════════
# ③  단백질 위치 (x, y, label, color, fontsize, fontweight)
# ══════════════════════════════════════════════════════════════════════
PROT = [
    # ── 세포 외부 (리간드) ──────────────────────────────────────────
    (-5.5,  7.0, 'Cu²⁺',              '#BF360C', 10, 'bold'),
    ( 0.5,  7.4, 'PrPSc\n(aggregates)','#B71C1C', 11, 'bold'),
    ( 5.5,  7.0, 'Laminin /\nHeparin', '#4527A0',  9, 'normal'),
    (-8.2,  4.0, 'Pritamab\n(ICSM18)', '#E65100', 12, 'bold'),

    # ── 세포막 ──────────────────────────────────────────────────────
    ( 0.0,  6.3, 'PrPC',              '#0D47A1', 16, 'bold'),
    (-4.0,  5.2, 'mGluR5',            '#6A1B9A', 11, 'normal'),
    ( 4.0,  5.2, 'NCAM1',             '#00695C', 11, 'normal'),
    (-6.5,  3.5, 'LRP1/2',            '#1B5E20', 10, 'normal'),
    ( 6.2,  3.5, 'Lipid Raft\n/ Caveolae','#0277BD',10, 'normal'),

    # ── 세포질 I: Kinases ──────────────────────────────────────────
    (-6.0,  2.0, 'Fyn',               '#1B5E20', 13, 'bold'),
    (-3.5,  2.5, 'Src',               '#2E7D32', 11, 'normal'),
    ( 0.0,  3.0, 'PI3K / Akt',        '#1B5E20', 13, 'bold'),
    ( 3.5,  2.5, 'ERK 1/2',           '#33691E', 12, 'bold'),
    ( 6.5,  2.0, 'PTEN',              '#827717', 11, 'normal'),
    (-5.5, -0.2, 'NMDAR\n(NR2B)',     '#4527A0', 12, 'bold'),
    ( 5.0, -0.2, 'Caveolin-1',        '#004D40', 11, 'normal'),
    ( 7.0,  0.8, 'ROS ↑',            '#BF360C', 12, 'bold'),
    (-7.5,  0.8, 'SOD1/2',            '#1A237E', 11, 'normal'),

    # ── 세포질 II: 2nd Messengers ──────────────────────────────────
    (-5.5,  1.2, 'mTOR',              '#1A237E', 13, 'bold'),
    (-2.0,  0.5, 'p38 MAPK',          '#880E4F', 12, 'bold'),
    ( 2.0,  0.5, 'GSK-3β',            '#4A148C', 13, 'bold'),
    ( 6.0, -1.0, 'Ca²⁺ Influx',      '#C62828', 12, 'bold'),
    (-6.5, -1.5, 'Autophagy / UPS',   '#2E7D32', 10, 'normal'),
    ( 5.5, -2.5, 'Mito ΔΨm ↓',       '#BF360C', 10, 'normal'),

    # ── 핵: Transcription Factors ─────────────────────────────────
    (-3.0, -4.0, 'NF-κB',             '#0D47A1', 13, 'bold'),
    (-1.0, -5.2, 'CREB',              '#1B5E20', 13, 'bold'),
    ( 1.5, -4.0, 'p53',               '#B71C1C', 13, 'bold'),
    ( 3.0, -5.5, 'Tau-P\n(tangle)',   '#E65100', 12, 'bold'),
    (-3.8, -6.0, 'BDNF / NGF',        '#006064', 11, 'normal'),
    ( 3.8, -3.5, 'Bcl-2 / Casp-3',   '#880E4F', 11, 'normal'),

    # ── 세포 하단 Outcomes ─────────────────────────────────────────
    (-5.5, -8.2, 'Neuroprotection',     '#00695C', 13, 'bold'),
    ( 0.0, -8.3, 'Synaptic Plasticity\n& Memory', '#1B5E20', 12, 'bold'),
    ( 6.0, -8.2, 'Neurodegeneration\n(prion / AD / PD)', '#C62828', 13, 'bold'),
]

# 레이블 → 좌표 검색
def fp(lbl):
    lbl_low = lbl.strip().lower()
    for p in PROT:
        if lbl_low in p[2].lower():
            return (p[0], p[1])
    return None

# ══════════════════════════════════════════════════════════════════════
# ④  유기 Bezier 곡선 함수
# ══════════════════════════════════════════════════════════════════════
def organic_curve(ax, p0, p3, color, lw=1.8, alpha=0.75,
                  bow=0.35, arrowhead=True, zorder=10):
    """두 점 사이를 S자 유기 Bezier로 연결"""
    x0,y0 = p0; x3,y3 = p3
    dx = x3-x0; dy = y3-y0
    L  = max(np.hypot(dx,dy), 0.1)
    # 제어점: 흐름 방향 수직 bow
    perp_x = -dy/L * bow * L * 0.5
    perp_y =  dx/L * bow * L * 0.5
    # cubic bezier 제어점
    cx1 = x0 + dx*0.30 + perp_x
    cy1 = y0 + dy*0.30 + perp_y
    cx2 = x0 + dx*0.70 - perp_x*0.4
    cy2 = y0 + dy*0.70 - perp_y*0.4
    t   = np.linspace(0, 1, 120)
    bx  = (1-t)**3*x0 + 3*(1-t)**2*t*cx1 + 3*(1-t)*t**2*cx2 + t**3*x3
    by  = (1-t)**3*y0 + 3*(1-t)**2*t*cy1 + 3*(1-t)*t**2*cy2 + t**3*y3
    # 곡선
    ax.plot(bx, by, color=color, lw=lw, alpha=alpha,
            solid_capstyle='round', zorder=zorder)
    # 화살표 (끝 방향)
    if arrowhead:
        dx2 = bx[-1]-bx[-6]; dy2 = by[-1]-by[-6]
        L2  = max(np.hypot(dx2,dy2), 1e-9)
        ax.annotate('', xy=(bx[-1], by[-1]),
                    xytext=(bx[-1]-dx2/L2*0.01, by[-1]-dy2/L2*0.01),
                    arrowprops=dict(
                        arrowstyle='->', color=color,
                        lw=lw*0.85,
                        mutation_scale=14,
                    ), zorder=zorder+1)

def block_mark(ax, p0, p3, color='#E65100', lw=2.5, alpha=0.85, zorder=12):
    """차단 (Pritamab): 중앙에 × 표시"""
    x0,y0 = p0; x3,y3 = p3
    mx = (x0+x3)/2; my = (y0+y3)/2
    organic_curve(ax, p0, (mx,my), color=color, lw=lw, alpha=alpha,
                  arrowhead=False, zorder=zorder)
    # × 심볼
    sz = 0.35
    ax.plot([mx-sz, mx+sz],[my-sz, my+sz], color=color, lw=lw*1.1,
            solid_capstyle='round', zorder=zorder+1)
    ax.plot([mx-sz, mx+sz],[my+sz, my-sz], color=color, lw=lw*1.1,
            solid_capstyle='round', zorder=zorder+1)

# ══════════════════════════════════════════════════════════════════════
# ⑤  시그널 엣지 정의 (src_partial, dst_partial, type, lw, bow)
# ══════════════════════════════════════════════════════════════════════
# colors: activation='#455A64', inhibition='#C62828', block='#E65100'
# pathway color families
C_ACT = '#546E7A'   # activation (gray-blue)
C_PRO = '#2E7D32'   # neuroprotective pathway (green)
C_PAT = '#B71C1C'   # pathological pathway (red)
C_BLK = '#E65100'   # Pritamab block (orange)
C_SYN = '#1565C0'   # synaptic plasticity (blue)

EDGES = [
    # ── Extracellular → Membrane ────────────────────────────────
    ('Cu²⁺',     'PrPC',       'act', C_ACT, 1.5, 0.2),
    ('Laminin',  'LRP1',       'act', C_ACT, 1.2, 0.1),
    ('PrPSc',    'PrPC',       'inh', C_PAT, 2.0, 0.25),
    ('Pritamab', 'PrPC',       'blk', C_BLK, 2.5, 0.15),

    # ── PrPC → Membrane lateral ─────────────────────────────────
    ('PrPC',     'mGluR5',     'act', C_SYN, 1.3, 0.3),
    ('PrPC',     'NCAM1',      'act', C_SYN, 1.3, 0.3),
    ('PrPC',     'LRP1',       'act', C_ACT, 1.0, 0.2),
    ('PrPC',     'Lipid Raft', 'act', C_ACT, 1.0, 0.2),

    # ── Membrane → Cytoplasm I ──────────────────────────────────
    ('PrPC',     'Fyn',        'act', C_PRO, 2.2, 0.30),
    ('PrPC',     'Src',        'act', C_ACT, 1.5, 0.20),
    ('PrPC',     'PI3K',       'act', C_PRO, 2.5, 0.15),
    ('PrPC',     'ERK 1/2',    'act', C_SYN, 2.0, 0.20),
    ('mGluR5',   'NMDAR',      'act', C_SYN, 1.8, 0.35),
    ('Lipid Raft','Caveolin',  'act', C_ACT, 1.0, 0.25),
    ('PrPSc',    'ROS',        'inh', C_PAT, 2.2, 0.35),
    ('Cu²⁺',     'SOD1',       'act', C_PRO, 1.0, 0.20),

    # ── CYT I → CYT I ───────────────────────────────────────────
    ('Fyn',      'PI3K',       'act', C_PRO, 1.4, 0.40),
    ('SOD1',     'ROS',        'inh', C_PRO, 1.2, 0.30),
    ('Src',      'ERK 1/2',    'act', C_ACT, 1.2, 0.30),

    # ── CYT I → CYT II ──────────────────────────────────────────
    ('PI3K',     'mTOR',       'act', C_PRO, 2.0, 0.25),
    ('ERK 1/2',  'p38 MAPK',   'act', C_PAT, 1.4, 0.30),
    ('PTEN',     'GSK-3β',     'act', C_PAT, 1.2, 0.25),
    ('NMDAR',    'Ca²⁺',       'act', C_PAT, 2.2, 0.40),
    ('ROS',      'p38 MAPK',   'inh', C_PAT, 1.5, 0.30),
    ('ROS',      'Mito',       'inh', C_PAT, 1.5, 0.30),
    ('Caveolin', 'Autophagy',  'act', C_PRO, 1.0, 0.30),

    # ── CYT II → Nucleus ────────────────────────────────────────
    ('mTOR',     'NF-κB',      'act', C_PRO, 2.0, 0.30),
    ('mTOR',     'CREB',       'act', C_SYN, 1.5, 0.20),
    ('p38 MAPK', 'p53',        'act', C_PAT, 1.5, 0.25),
    ('GSK-3β',   'Tau-P',      'act', C_PAT, 2.2, 0.30),
    ('GSK-3β',   'CREB',       'inh', C_PAT, 1.2, 0.35),
    ('Ca²⁺',     'Bcl-2',      'act', C_PAT, 1.5, 0.30),
    ('Ca²⁺',     'Tau-P',      'act', C_PAT, 1.3, 0.25),
    ('Autophagy','BDNF',       'act', C_PRO, 1.0, 0.20),
    ('Mito',     'Bcl-2',      'act', C_PAT, 1.2, 0.30),

    # ── Nucleus → Outcomes ──────────────────────────────────────
    ('NF-κB',    'Neuroprotection','act', C_PRO, 2.0, 0.30),
    ('CREB',     'Synaptic',   'act', C_SYN, 2.2, 0.25),
    ('BDNF',     'Synaptic',   'act', C_SYN, 1.5, 0.25),
    ('BDNF',     'Neuroprotection','act', C_PRO, 1.2, 0.30),
    ('p53',      'Neurodegeneration','act', C_PAT, 2.0, 0.30),
    ('Tau-P',    'Neurodegeneration','act', C_PAT, 2.5, 0.25),
    ('Bcl-2',    'Neurodegeneration','act', C_PAT, 1.5, 0.30),
    ('NF-κB',    'Neurodegeneration','act', C_PAT, 0.9, 0.35),
    ('Pritamab', 'Neurodegeneration','blk', C_BLK, 1.8, 0.10),
]

# ── 엣지 렌더링 (뒤→앞) ────────────────────────────────────────────
for row in EDGES:
    src, dst, etype, col, lw, bow = row
    p0 = fp(src); p3 = fp(dst)
    if p0 is None or p3 is None:
        continue
    if etype == 'blk':
        block_mark(ax, p0, p3, color=col, lw=lw, alpha=0.90, zorder=10)
    else:
        alpha = 0.75 if etype == 'act' else 0.85
        organic_curve(ax, p0, p3, color=col, lw=lw*1.6, alpha=alpha,
                      bow=bow, arrowhead=True, zorder=10)

# ══════════════════════════════════════════════════════════════════════
# ⑥  단백질 텍스트 렌더링
# ══════════════════════════════════════════════════════════════════════
for (x, y, lbl, col, fsize, fw) in PROT:
    # 얇은 그림자
    ax.text(x+0.06, y-0.06, lbl, ha='center', va='center',
            fontsize=fsize, color='#BDBDBD', fontweight=fw,
            fontfamily='DejaVu Sans', alpha=0.5, zorder=14)
    # 메인 텍스트
    ax.text(x, y, lbl, ha='center', va='center',
            fontsize=fsize, color=col, fontweight=fw,
            fontfamily='DejaVu Sans', zorder=15)

# ══════════════════════════════════════════════════════════════════════
# ⑦  구획 라벨
# ══════════════════════════════════════════════════════════════════════
compartment_labels = [
    ( 7.8,  6.5, 'EXTRACELLULAR\nSPACE',    '#78909C',  8),
    (-8.2,  6.0, 'PLASMA\nMEMBRANE',        '#1565C0',  8),
    (-8.5,  1.5, 'CYTOPLASM\n(Kinases)',     '#2E7D32',  8),
    (-8.5, -1.5, 'CYTOPLASM\n(2nd Msg)',     '#6A1B9A',  8),
    (-4.8, -7.8, 'NUCLEUS',                 '#EF6C00',  8),
    ( 0.0, -9.0, 'FUNCTIONAL OUTCOMES',     '#37474F',  9),
]
for (x, y, lbl, col, fsize) in compartment_labels:
    ax.text(x, y, lbl, ha='center', va='center',
            fontsize=fsize, color=col, alpha=0.80,
            fontweight='bold', fontfamily='DejaVu Sans',
            style='italic', zorder=16)

# ── 세포막 라벨 화살표 ──
ax.annotate('', xy=(0.5, 6.0), xytext=(-8.2+1.5, 5.5),
            arrowprops=dict(arrowstyle='->', color='#1565C0',
                            connectionstyle='arc3,rad=0.2', lw=1.0),
            zorder=16)

# ══════════════════════════════════════════════════════════════════════
# ⑧  범례
# ══════════════════════════════════════════════════════════════════════
legend_items = [
    mpatches.Patch(color=C_ACT, label='Activation'),
    mpatches.Patch(color=C_PRO, label='Neuroprotective pathway'),
    mpatches.Patch(color=C_SYN, label='Synaptic / plasticity'),
    mpatches.Patch(color=C_PAT, label='Pathological / pro-death'),
    mpatches.Patch(color=C_BLK, label='Pritamab (ICSM18) — block'),
]
ax.legend(handles=legend_items,
          loc='lower right', bbox_to_anchor=(1.0, 0.0),
          fontsize=9, framealpha=0.95,
          edgecolor='#CCCCCC', facecolor='white',
          handlelength=1.5, ncol=1, labelspacing=0.6)

# ══════════════════════════════════════════════════════════════════════
# ⑨  타이틀
# ══════════════════════════════════════════════════════════════════════
fig.text(0.50, 0.98,
         'PrPC Signaling Pathway in Living Cell',
         ha='center', va='top', fontsize=22,
         fontweight='bold', color='#1A237E',
         fontfamily='DejaVu Sans')
fig.text(0.50, 0.957,
         'A cell cross-section view of prion protein-mediated signal transduction',
         ha='center', va='top', fontsize=10.5,
         color='#555555', style='italic',
         fontfamily='DejaVu Sans')

plt.tight_layout(pad=0.5)
fig.savefig(OUT, dpi=300, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.close()
print(f"Saved: {OUT}")
