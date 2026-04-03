"""
prpc_pathway_organic.py
========================
논문용 PrPC 시그널 패스웨이 — 유기적 3D 스타일
- 단백질: 텍스트 레이블만 (박스/구체 없음)
- 연결선: 두꺼운 3D Bezier 곡선
- 레이어 경계: 물결형 유기 표면 (세포막처럼)
- 흰 배경, 논문 품질 (300 dpi)
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import Axes3D, proj3d
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.path import Path
import matplotlib.patheffects as pe

OUT = r"f:\ADDS\outputs\pritamab_pptx_figures\prpc_pathway_organic.png"

# ══════════════════════════════════════════════════════════════════════
# 레이어 Z-레벨
# ══════════════════════════════════════════════════════════════════════
ZL = {
    'MEM':  9.0,
    'CYT1': 6.0,
    'CYT2': 3.5,
    'NUC':  1.2,
    'OUT':  -1.0,
}

# ══════════════════════════════════════════════════════════════════════
# 단백질 노드 (x, y, z, label, color, fontsize, style)
# ══════════════════════════════════════════════════════════════════════
NODES = [
    # ── Membrane ────────────────────────────────────────────────────
    ( 0.0,  0.0, ZL['MEM'],  'PrPC',            '#1565C0', 15, 'bold'),
    (-3.8,  0.0, ZL['MEM'],  'Pritamab\n(ICSM18)', '#E65100', 11, 'bold'),
    ( 3.8,  0.0, ZL['MEM'],  'PrPSc',           '#B71C1C', 12, 'bold'),
    (-2.0,  2.2, ZL['MEM'],  'Cu²⁺',            '#BF360C', 10, 'normal'),
    ( 2.0,  2.2, ZL['MEM'],  'Lipid Raft',      '#0277BD', 10, 'normal'),
    (-2.0, -2.2, ZL['MEM'],  'mGluR5',          '#6A1B9A', 10, 'normal'),
    ( 2.0, -2.2, ZL['MEM'],  'NCAM1',           '#0277BD', 10, 'normal'),
    (-3.5,  2.5, ZL['MEM'],  'LRP1/2',          '#00695C', 10, 'normal'),

    # ── Cytoplasm I — Kinases ───────────────────────────────────────
    (-4.0,  1.5, ZL['CYT1'], 'Fyn',             '#2E7D32', 12, 'bold'),
    (-2.2,  1.5, ZL['CYT1'], 'Src',             '#388E3C', 11, 'normal'),
    ( 0.0,  1.5, ZL['CYT1'], 'PI3K\n/Akt',      '#1B5E20', 12, 'bold'),
    ( 2.2,  1.5, ZL['CYT1'], 'ERK 1/2',         '#33691E', 11, 'normal'),
    ( 4.0,  1.5, ZL['CYT1'], 'PTEN',            '#827717', 11, 'normal'),
    (-2.5, -1.5, ZL['CYT1'], 'NMDAR\n(NR2B)',   '#6A1B9A', 12, 'bold'),
    ( 0.0, -2.0, ZL['CYT1'], 'Caveolin-1',      '#004D40', 10, 'normal'),
    ( 3.2, -1.5, ZL['CYT1'], 'ROS↑',            '#BF360C', 12, 'bold'),
    (-4.2, -1.5, ZL['CYT1'], 'SOD1/2',          '#1A237E', 11, 'normal'),

    # ── Cytoplasm II — 2nd Messenger ───────────────────────────────
    (-3.5,  0.0, ZL['CYT2'], 'mTOR',            '#1A237E', 12, 'bold'),
    (-1.5,  0.0, ZL['CYT2'], 'p38 MAPK',        '#880E4F', 11, 'normal'),
    ( 0.5,  0.0, ZL['CYT2'], 'GSK-3β',          '#4A148C', 12, 'bold'),
    ( 3.0,  0.0, ZL['CYT2'], 'Ca²⁺\nInflux',    '#B71C1C', 12, 'bold'),
    (-3.5, -2.0, ZL['CYT2'], 'Autophagy\n/UPS', '#2E7D32', 11, 'normal'),
    ( 3.5, -2.0, ZL['CYT2'], 'Mito ΔΨ↓',       '#BF360C', 11, 'normal'),

    # ── Nucleus ─────────────────────────────────────────────────────
    (-4.0,  0.0, ZL['NUC'],  'NF-κB',           '#0D47A1', 12, 'bold'),
    (-2.0,  0.0, ZL['NUC'],  'CREB',            '#1B5E20', 12, 'bold'),
    ( 0.0,  0.0, ZL['NUC'],  'p53',             '#B71C1C', 12, 'bold'),
    ( 2.0,  0.5, ZL['NUC'],  'Tau-P',           '#E65100', 12, 'bold'),
    ( 3.8,  0.0, ZL['NUC'],  'Bcl-2\n/Casp-3',  '#880E4F', 11, 'normal'),
    (-4.0, -2.0, ZL['NUC'],  'BDNF/NGF',        '#006064', 11, 'normal'),

    # ── Outcomes ────────────────────────────────────────────────────
    (-3.5,  0.0, ZL['OUT'],  'Neuroprotection',    '#00695C', 13, 'bold'),
    ( 0.0,  0.0, ZL['OUT'],  'Synaptic Plasticity\n& Memory', '#2E7D32', 12, 'bold'),
    ( 4.0,  0.0, ZL['OUT'],  'Neurodegeneration\n(prion/AD/PD)', '#C62828', 13, 'bold'),
]

# ══════════════════════════════════════════════════════════════════════
# 엣지 (src_label_partial, dst_label_partial, type, linewidth)
# type: 'act'=activate '#555', 'inh'=inhibit '#D32F2F', 'block'=block '#E65100'
# ══════════════════════════════════════════════════════════════════════
EDGES = [
    # Pritamab block
    ('Pritamab', 'PrPC',      'block', 2.5),
    ('Pritamab', 'Neurodegeneration', 'block', 1.8),

    # EC → MEM
    ('Cu²⁺',    'PrPC',      'act', 1.2),
    ('LRP1',    'PrPC',      'act', 1.0),
    ('PrPSc',   'PrPC',      'inh', 2.0),   # misfolding

    # PrPC → others @ MEM
    ('PrPC',    'mGluR5',   'act', 1.2),
    ('PrPC',    'NCAM1',    'act', 1.0),
    ('PrPC',    'Lipid Raft','act', 1.0),
    ('PrPC',    'Cu²⁺',     'act', 1.0),

    # PrPC → CYT1
    ('PrPC',    'Fyn',      'act', 2.0),
    ('PrPC',    'Src',      'act', 1.5),
    ('PrPC',    'PI3K',     'act', 2.0),
    ('PrPC',    'ERK 1/2',  'act', 1.5),
    ('mGluR5',  'NMDAR',    'act', 1.5),
    ('Lipid Raft','Caveolin','act', 1.0),
    ('PrPSc',   'ROS',      'act', 2.0),
    ('Cu²⁺',    'SOD1',     'act', 1.0),

    # CYT1 → CYT1
    ('Fyn',     'PI3K',     'act', 1.2),
    ('SOD1',    'ROS',      'inh', 1.2),

    # CYT1 → CYT2
    ('PI3K',    'mTOR',     'act', 1.8),
    ('ERK 1/2', 'p38 MAPK', 'act', 1.2),
    ('PTEN',    'GSK-3β',   'act', 1.2),
    ('NMDAR',   'Ca²⁺',     'act', 2.0),
    ('Caveolin','Autophagy','act', 1.0),
    ('ROS',     'p38 MAPK', 'act', 1.5),
    ('ROS',     'Mito',     'act', 1.5),

    # CYT2 → NUC
    ('mTOR',    'NF-κB',    'act', 1.8),
    ('mTOR',    'CREB',     'act', 1.2),
    ('p38 MAPK','p53',      'act', 1.2),
    ('GSK-3β',  'Tau-P',    'act', 2.0),
    ('GSK-3β',  'CREB',     'inh', 1.2),
    ('Ca²⁺',    'Bcl-2',    'act', 1.5),
    ('Ca²⁺',    'Tau-P',    'act', 1.0),
    ('Autophagy','BDNF',    'act', 1.0),
    ('Mito',    'Bcl-2',    'act', 1.2),

    # NUC → Outcomes
    ('NF-κB',   'Neuroprotection','act', 1.5),
    ('CREB',    'Synaptic', 'act', 2.0),
    ('BDNF',    'Synaptic', 'act', 1.5),
    ('BDNF',    'Neuroprotection','act',1.2),
    ('p53',     'Neurodegeneration','act',1.5),
    ('Tau-P',   'Neurodegeneration','act',2.0),
    ('Bcl-2',   'Neurodegeneration','act',1.5),
    ('NF-κB',   'Neurodegeneration','act',0.8),
]

# 레이블 → 노드 위치 검색
def find_node(partial):
    for n in NODES:
        if partial.strip().lower() in n[3].lower():
            return (n[0], n[1], n[2])
    return None

# ══════════════════════════════════════════════════════════════════════
# 3D Bezier 곡선 (부드럽고 유기적)
# ══════════════════════════════════════════════════════════════════════
def bezier3d(p0, p3, n=80, bow=0.18):
    x0,y0,z0 = p0; x3,y3,z3 = p3
    dz = z0 - z3
    # 계층 통과 시 Z방향 bow, XY 약간 side bow
    side = bow * (y3 - y0)
    x1 = x0 + (x3-x0)*0.3 + side * 0.5
    y1 = y0 + (y3-y0)*0.3 + dz * 0.05
    z1 = z0 - dz * 0.33
    x2 = x0 + (x3-x0)*0.7 - side * 0.5
    y2 = y0 + (y3-y0)*0.7 - dz * 0.05
    z2 = z3 + dz * 0.33
    t  = np.linspace(0,1,n)
    B  = lambda a,b,c,d: (1-t)**3*a + 3*(1-t)**2*t*b + 3*(1-t)*t**2*c + t**3*d
    return B(x0,x1,x2,x3), B(y0,y1,y2,y3), B(z0,z1,z2,z3)

# ══════════════════════════════════════════════════════════════════════
# 유기적 물결 레이어 표면 (세포막처럼)
# ══════════════════════════════════════════════════════════════════════
def wave_surface(ax, z_base, x_range=(-5.5,5.5), y_range=(-3.2,3.2),
                 amp=0.10, freq=1.2, color='#90CAF9', alpha=0.10):
    xs = np.linspace(x_range[0], x_range[1], 40)
    ys = np.linspace(y_range[0], y_range[1], 30)
    X, Y = np.meshgrid(xs, ys)
    Z = z_base + amp * (np.sin(freq*X) + 0.5*np.cos(freq*1.3*Y))
    ax.plot_surface(X, Y, Z, color=color, alpha=alpha,
                    linewidth=0, antialiased=True, shade=False, zorder=1)
    return Z

def wave_edge(ax, z_base, x_range=(-5.5,5.5), y_val=3.2,
              amp=0.10, freq=1.2, color='#555', lw=0.8, alpha=0.35):
    xs = np.linspace(x_range[0], x_range[1], 200)
    zs = z_base + amp * np.sin(freq * xs)
    ys = np.full_like(xs, y_val)
    ax.plot(xs, ys, zs, color=color, lw=lw, alpha=alpha, zorder=4)

# ══════════════════════════════════════════════════════════════════════
# Figure 초기화
# ══════════════════════════════════════════════════════════════════════
fig = plt.figure(figsize=(22, 15), dpi=150)
fig.patch.set_facecolor('white')
ax  = fig.add_subplot(111, projection='3d', facecolor='white')

# 배경 패널 제거
ax.xaxis.pane.fill = False; ax.yaxis.pane.fill = False; ax.zaxis.pane.fill = False
ax.xaxis.pane.set_edgecolor('#E0E0E0')
ax.yaxis.pane.set_edgecolor('#E0E0E0')
ax.zaxis.pane.set_edgecolor('#E0E0E0')
ax.grid(False)

# ══════════════════════════════════════════════════════════════════════
# 유기적 레이어 표면
# ══════════════════════════════════════════════════════════════════════
layer_defs = [
    (ZL['MEM'],  '#90CAF9', 0.10, 'Membrane',     0.9, 1.1),
    (ZL['CYT1'], '#A5D6A7', 0.09, 'Cytoplasm I',  1.1, 0.9),
    (ZL['CYT2'], '#CE93D8', 0.09, 'Cytoplasm II', 0.8, 1.3),
    (ZL['NUC'],  '#FFCC80', 0.09, 'Nucleus',      1.3, 0.8),
    (ZL['OUT'],  '#B0BEC5', 0.07, 'Outcomes',     1.0, 1.0),
]
for z_base, col, alp, lname, amp_x, amp_y in layer_defs:
    wave_surface(ax, z_base, amp=0.12, freq=amp_x, color=col, alpha=alp*0.8)
    # 레이어 전면 경계선 (물결형)
    wave_edge(ax, z_base, color=col.replace('#90','#42').replace('#A5','#2E')
              .replace('#CE','#9C').replace('#FF','#EF').replace('#B0','#78'),
              amp=0.12, freq=amp_x, lw=1.5, alpha=0.5)

# ══════════════════════════════════════════════════════════════════════
# 엣지 — 두꺼운 유기 Bezier 곡선
# ══════════════════════════════════════════════════════════════════════
edge_colors = {
    'act':   '#546E7A',   # 회색-파랑 (활성화)
    'inh':   '#C62828',   # 빨강 (억제)
    'block': '#E65100',   # 주황 (차단)
}

for (src, dst, etype, lw) in EDGES:
    p0 = find_node(src); p3 = find_node(dst)
    if p0 is None or p3 is None:
        continue
    xs, ys, zs = bezier3d(p0, p3, bow=0.15)
    col = edge_colors[etype]
    # 차단/억제는 구별되는 두께/투명도
    alpha = 0.80 if etype=='act' else 0.90
    ax.plot(xs, ys, zs, color=col, lw=lw*1.8, alpha=alpha,
            solid_capstyle='round', zorder=3)
    # 화살표 방향 (끝 2포인트로 quiver)
    dx = xs[-1]-xs[-3]; dy = ys[-1]-ys[-3]; dz = zs[-1]-zs[-3]
    L = max(np.sqrt(dx**2+dy**2+dz**2), 1e-9)
    tip = 0.30
    ax.quiver(xs[-1], ys[-1], zs[-1],
              dx/L*tip, dy/L*tip, dz/L*tip,
              length=tip, color=col,
              arrow_length_ratio=1.0, lw=lw*1.0, alpha=alpha)

# ══════════════════════════════════════════════════════════════════════
# 노드 — 텍스트만 (그림자 효과로 부각)
# ══════════════════════════════════════════════════════════════════════
for (x, y, z, label, col, fsize, style) in NODES:
    # 그림자 (약간 offset)
    ax.text(x+0.04, y-0.04, z-0.04, label,
            ha='center', va='center',
            fontsize=fsize, color='#CCCCCC',
            fontweight=style, alpha=0.4,
            fontfamily='DejaVu Sans', zorder=8)
    # 메인 텍스트
    ax.text(x, y, z, label,
            ha='center', va='center',
            fontsize=fsize, color=col,
            fontweight=style,
            fontfamily='DejaVu Sans',
            zorder=10)

# ══════════════════════════════════════════════════════════════════════
# 레이어 라벨 (좌측 Z축 옆)
# ══════════════════════════════════════════════════════════════════════
layer_labels = [
    (ZL['MEM']+0.3,  'MEMBRANE',      '#1565C0'),
    (ZL['CYT1']+0.3, 'CYTOPLASM I',   '#2E7D32'),
    (ZL['CYT2']+0.3, 'CYTOPLASM II',  '#6A1B9A'),
    (ZL['NUC']+0.3,  'NUCLEUS',       '#BF360C'),
    (ZL['OUT']+0.2,  'OUTCOMES',      '#37474F'),
]
for z_lev, lbl, col in layer_labels:
    ax.text(-6.0, -3.5, z_lev, lbl,
            fontsize=9, color=col, alpha=0.85,
            fontweight='bold', fontfamily='DejaVu Sans', zorder=12)

# ══════════════════════════════════════════════════════════════════════
# 범례
# ══════════════════════════════════════════════════════════════════════
legend_items = [
    mpatches.Patch(color='#546E7A', label='Activation'),
    mpatches.Patch(color='#C62828', label='Inhibition'),
    mpatches.Patch(color='#E65100', label='Pritamab block'),
    mpatches.Patch(color='#1565C0', label='PrPC (central)'),
    mpatches.Patch(color='#2E7D32', label='Neuroprotective'),
    mpatches.Patch(color='#B71C1C', label='Pathological'),
]
ax.legend(handles=legend_items,
          loc='lower left',
          fontsize=9, framealpha=0.92,
          edgecolor='#CCCCCC',
          facecolor='white',
          ncol=2, handlelength=1.2,
          bbox_to_anchor=(-0.02, -0.04))

# ══════════════════════════════════════════════════════════════════════
# 축 / 뷰
# ══════════════════════════════════════════════════════════════════════
ax.set_xlim(-6.0, 6.0); ax.set_ylim(-3.5, 3.5); ax.set_zlim(-2.5, 11.5)
ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
ax.set_xlabel(''); ax.set_ylabel(''); ax.set_zlabel('')
ax.view_init(elev=22, azim=-50)

# ══════════════════════════════════════════════════════════════════════
# 타이틀
# ══════════════════════════════════════════════════════════════════════
fig.text(0.50, 0.97,
         'PrPC Signaling Pathway',
         ha='center', va='top', fontsize=20,
         fontweight='bold', color='#1A1A1A',
         fontfamily='DejaVu Sans')
fig.text(0.50, 0.945,
         'Membrane  →  Cytoplasm (Kinase / 2nd Messenger)  →  Nucleus  →  Outcomes',
         ha='center', va='top', fontsize=10, color='#555555',
         style='italic', fontfamily='DejaVu Sans')

plt.tight_layout()
fig.savefig(OUT, dpi=300, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.close()
print(f"Saved: {OUT}")
