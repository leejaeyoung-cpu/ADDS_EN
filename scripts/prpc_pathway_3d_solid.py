"""
prpc_pathway_3d_solid.py  v2
============================
PrPC 시그널 패스웨이 진짜 3D 렌더링
- Poly3DCollection 3D 박스
- 층별 Z-depth 분리 5단계
- 레이블: 3D 좌표 → 2D 투영 후 ax.annotate로 항상 전면 표시
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from mpl_toolkits.mplot3d import proj3d
import numpy as np

OUT = r"f:\ADDS\outputs\pritamab_pptx_figures\prpc_pathway_3d_solid.png"

# ── 전역 레이블 큐 (렌더 후 한번에 그림) ──────────────────────────────
_label_queue = []   # list of (x3,y3,z3, text, fontsize, color)

BG = '#030509'
fig = plt.figure(figsize=(24, 15), facecolor=BG)
ax  = fig.add_subplot(111, projection='3d', facecolor=BG)

# Z 레벨 (높을수록 Extracellular = 위)
Z_MEM  = 9.0
Z_CYT  = 5.5
Z_CYT2 = 3.5
Z_NUC  = 1.0
Z_OUT  = -1.0

# ─────────────────────────── 헬퍼 ────────────────────────────────────

def box3d(ax, cx, cy, cz, w, d, h=0.6, fc='#112233', ec='#4FC3F7',
          alpha=0.88):
    x0,x1 = cx-w/2, cx+w/2
    y0,y1 = cy-d/2, cy+d/2
    z0,z1 = cz, cz+h
    verts = [
        [(x0,y0,z0),(x1,y0,z0),(x1,y1,z0),(x0,y1,z0)],  # bottom
        [(x0,y0,z1),(x1,y0,z1),(x1,y1,z1),(x0,y1,z1)],  # top
        [(x0,y0,z0),(x1,y0,z0),(x1,y0,z1),(x0,y0,z1)],  # front
        [(x0,y1,z0),(x1,y1,z0),(x1,y1,z1),(x0,y1,z1)],  # back
        [(x0,y0,z0),(x0,y1,z0),(x0,y1,z1),(x0,y0,z1)],  # left
        [(x1,y0,z0),(x1,y1,z0),(x1,y1,z1),(x1,y0,z1)],  # right
    ]
    poly = Poly3DCollection(verts, alpha=alpha)
    poly.set_facecolor(fc); poly.set_edgecolor(ec); poly.set_linewidth(0.9)
    ax.add_collection3d(poly)

def node3d(ax, cx, cy, cz, w, d, label,
           fc='#112233', ec='#4FC3F7', fontsize=8.5, color='white'):
    """3D 박스 + 레이블 큐에 등록 (레이블은 나중에 2D로 찍음)"""
    box3d(ax, cx, cy, cz, w, d, fc=fc, ec=ec)
    _label_queue.append((cx, cy, cz+0.30, label, fontsize, color))

def layer_plate(ax, x0, x1, y0, y1, z, color, alpha=0.05, label=None):
    verts = [[(x0,y0,z),(x1,y0,z),(x1,y1,z),(x0,y1,z)]]
    poly = Poly3DCollection(verts, alpha=alpha)
    poly.set_facecolor(color); poly.set_edgecolor(color); poly.set_linewidth(1.2)
    ax.add_collection3d(poly)
    if label:
        ax.text(x0+0.2, y0+0.1, z, label, fontsize=8,
                color=color, alpha=0.75, fontweight='bold', zorder=5)

def arr(ax, p1, p2, color='#5577AA', lw=1.8, ls='-', alpha=0.85):
    x1,y1,z1 = p1; x2,y2,z2 = p2
    ax.plot([x1,x2],[y1,y2],[z1,z2],
            color=color, lw=lw, linestyle=ls, alpha=alpha)
    dx,dy,dz = x2-x1, y2-y1, z2-z1
    L = max(np.sqrt(dx**2+dy**2+dz**2), 1e-9)
    tl = min(0.35, L*0.22)
    ax.quiver(x2,y2,z2,-dx/L*tl,-dy/L*tl,-dz/L*tl,
              length=tl, color=color, arrow_length_ratio=1.0,
              lw=lw*0.6, alpha=alpha)

# ══════════════════════════════════════════════════════════════════════
# 레이어 플레이트
# ══════════════════════════════════════════════════════════════════════
layer_plate(ax,-7,7,-5,5, Z_MEM,  '#1E90FF', alpha=0.055,
            label='① EXTRACELLULAR / MEMBRANE')
layer_plate(ax,-7,7,-5,5, Z_CYT,  '#39FF14', alpha=0.045,
            label='② CYTOPLASM — Kinase Cascades')
layer_plate(ax,-7,7,-5,5, Z_CYT2, '#FF8C00', alpha=0.045,
            label='③ CYTOPLASM — 2nd Messengers')
layer_plate(ax,-7,7,-5,5, Z_NUC,  '#FF69B4', alpha=0.045,
            label='④ NUCLEUS')
layer_plate(ax,-7,7,-5,5, Z_OUT,  '#00FFFF', alpha=0.045,
            label='⑤ OUTCOMES')

# ══════════════════════════════════════════════════════════════════════
# TIER 1 : MEMBRANE  (Z=9)
# ══════════════════════════════════════════════════════════════════════
node3d(ax,  0.0,  0.0, Z_MEM, 2.5, 1.2,
       'PrPC\nGPI-anchored (119-231)',
       fc='#0D1F3C', ec='#4FC3F7', fontsize=10, color='#90CAF9')

node3d(ax, -5.0,  0.0, Z_MEM, 2.3, 1.0,
       'Pritamab\n(ICSM18)',
       fc='#1A1000', ec='#FFD700', fontsize=9, color='#FFD700')

node3d(ax,  5.0,  0.0, Z_MEM, 2.3, 1.0,
       'PrPSc\n/ Aggregates',
       fc='#2D0000', ec='#FF3333', fontsize=9, color='#FF6666')

node3d(ax, -3.0,  2.8, Z_MEM, 2.1, 0.9,
       'Cu² Binding\nHis 96, 111',
       fc='#0F2200', ec='#76FF03', fontsize=8, color='#CCFF90')

node3d(ax,  3.0,  2.8, Z_MEM, 2.0, 0.9,
       'Lipid Raft\nGPI Anchor',
       fc='#0A0F25', ec='#90CAF9', fontsize=8)

node3d(ax, -3.0, -2.8, Z_MEM, 2.1, 0.9,
       'Laminin-Rec\nLRP1/LRP2',
       fc='#0D1E20', ec='#26C6DA', fontsize=8)

node3d(ax,  3.0, -2.8, Z_MEM, 1.9, 0.9,
       'mGluR5\nreceptor',
       fc='#1C0D25', ec='#CE93D8', fontsize=8)

# ══════════════════════════════════════════════════════════════════════
# TIER 2 : CYTOPLASM Kinase  (Z=5.5)
# ══════════════════════════════════════════════════════════════════════
node3d(ax, -5.5, -2.5, Z_CYT, 1.9, 0.9,
       'Fyn\nKinase',
       fc='#152500', ec='#AEEA00', fontsize=9)

node3d(ax, -3.0, -2.5, Z_CYT, 1.9, 0.9,
       'PI3K\n/ Akt',
       fc='#002818', ec='#00E5FF', fontsize=9)

node3d(ax, -0.5, -2.5, Z_CYT, 1.8, 0.9,
       'ERK 1/2',
       fc='#1A2600', ec='#CCFF90', fontsize=9)

node3d(ax,  1.8, -2.5, Z_CYT, 1.7, 0.9,
       'PTEN',
       fc='#250A00', ec='#FFAB40', fontsize=9)

node3d(ax,  4.0,  0.0, Z_CYT, 1.9, 0.9,
       'NMDAR\nNR2B',
       fc='#180025', ec='#EA80FC', fontsize=9)

node3d(ax,  1.8,  2.5, Z_CYT, 1.9, 0.9,
       'Caveolin-1',
       fc='#001A20', ec='#84FFFF', fontsize=9)

node3d(ax, -0.5,  2.5, Z_CYT, 1.9, 0.9,
       'Src\nKinase',
       fc='#1A1A00', ec='#FFD740', fontsize=9)

node3d(ax, -3.0,  2.5, Z_CYT, 1.9, 0.9,
       'SOD1/SOD2',
       fc='#002008', ec='#00E676', fontsize=9)

node3d(ax,  5.8, -1.2, Z_CYT, 2.3, 0.9,
       'ROS\nOxidative Stress',
       fc='#2D0000', ec='#FF6D00', fontsize=9, color='#FFAB40')

# ══════════════════════════════════════════════════════════════════════
# TIER 3 : 2nd Messenger  (Z=3.5)
# ══════════════════════════════════════════════════════════════════════
node3d(ax, -4.5,  0.0, Z_CYT2, 1.7, 0.85,
       'mTOR',
       fc='#001F10', ec='#69F0AE', fontsize=9)

node3d(ax, -2.0,  0.0, Z_CYT2, 1.7, 0.85,
       'p38 MAPK',
       fc='#200020', ec='#F48FB1', fontsize=9)

node3d(ax,  0.5,  0.0, Z_CYT2, 1.7, 0.85,
       'GSK-3β',
       fc='#261500', ec='#FFD180', fontsize=9)

node3d(ax,  3.0, -1.2, Z_CYT2, 2.1, 0.9,
       'Ca² Influx\nexcitotoxicity',
       fc='#2D0D00', ec='#FF6E40', fontsize=8.5, color='#FFAB91')

node3d(ax,  5.5,  1.5, Z_CYT2, 1.9, 0.85,
       'Autophagy\nUPS',
       fc='#101A00', ec='#B2FF59', fontsize=9)

# ══════════════════════════════════════════════════════════════════════
# TIER 4 : NUCLEUS  (Z=1)
# ══════════════════════════════════════════════════════════════════════
node3d(ax, -5.5,  0.0, Z_NUC, 1.9, 0.9,
       'NF-κB\nsurvival',
       fc='#001A38', ec='#40C4FF', fontsize=9)

node3d(ax, -3.0,  0.0, Z_NUC, 1.7, 0.9,
       'p53\napoptosis',
       fc='#200010', ec='#FF4081', fontsize=9)

node3d(ax, -0.5,  0.0, Z_NUC, 1.7, 0.9,
       'CREB\nmemory',
       fc='#001F10', ec='#69F0AE', fontsize=9)

node3d(ax,  2.0,  1.2, Z_NUC, 2.1, 0.9,
       'Tau-P\ntangle',
       fc='#260800', ec='#FF6D00', fontsize=9, color='#FFAB40')

node3d(ax,  2.0, -1.2, Z_NUC, 2.1, 0.9,
       'Bcl-2/Casp-3\napoptosis',
       fc='#2D0000', ec='#FF3333', fontsize=8.5, color='#FF8A80')

node3d(ax,  5.0,  0.0, Z_NUC, 2.1, 0.9,
       'BDNF/NGF\nneuroprotection',
       fc='#001A38', ec='#18FFFF', fontsize=8.5, color='#80DEEA')

# ══════════════════════════════════════════════════════════════════════
# TIER 5 : OUTCOMES  (Z=-1)
# ══════════════════════════════════════════════════════════════════════
node3d(ax, -5.0,  0.0, Z_OUT, 2.8, 1.2,
       'Neuroprotection\n(normal PrPC fn)',
       fc='#001A20', ec='#00E5FF', fontsize=10, color='#80DEEA')

node3d(ax,  0.0,  0.0, Z_OUT, 3.0, 1.2,
       'Synaptic Plasticity\n& Memory (LTP/LTD)',
       fc='#001F10', ec='#69F0AE', fontsize=10, color='#B9F6CA')

node3d(ax,  5.0, -0.5, Z_OUT, 2.8, 1.2,
       'Neurodegeneration\n(prion/AD/PD)',
       fc='#2D0000', ec='#FF1744', fontsize=10, color='#FF8A80')

# ══════════════════════════════════════════════════════════════════════
# 3D 화살표 연결
# ══════════════════════════════════════════════════════════════════════
ZM = Z_MEM+0.30

# Membrane 내
arr(ax,(-3.9,0,ZM),(-1.3,0,ZM),'#FFD700',lw=2.2,ls='--')   # Pritamab→PrPC(block)
arr(ax,(1.3,0,ZM),(3.8,0,ZM),'#FF3333',lw=2.0)              # PrPC→PrPSc
arr(ax,(-0.5,0.7,ZM),(-2.2,2.5,ZM),'#76FF03',lw=1.4)       # PrPC→Cu2+
arr(ax,(0.5,0.7,ZM),(2.2,2.5,ZM),'#90CAF9',lw=1.4)         # PrPC→LipidRaft
arr(ax,(-0.5,-0.7,ZM),(-2.2,-2.5,ZM),'#26C6DA',lw=1.4)     # PrPC→Laminin
arr(ax,(0.5,-0.7,ZM),(2.2,-2.5,ZM),'#CE93D8',lw=1.4)       # PrPC→mGluR5

# Membrane→Cytoplasm (Z 관통)
ZC = Z_CYT+0.55
arr(ax,(0,0,Z_MEM),(-5.0,-2.0,ZC),'#AEEA00',lw=1.5)        # PrPC→Fyn
arr(ax,(0,0,Z_MEM),(-2.6,-2.0,ZC),'#00E5FF',lw=1.5)        # PrPC→PI3K
arr(ax,(0,0,Z_MEM),(-0.2,-2.0,ZC),'#CCFF90',lw=1.3)        # PrPC→ERK
arr(ax,(3.0,-2.5,Z_MEM),(4.0,-0.5,ZC),'#EA80FC',lw=1.5)    # mGluR5→NMDAR
arr(ax,(3.0,2.5,Z_MEM),(1.8,2.0,ZC),'#84FFFF',lw=1.3)      # LipidRaft→Caveolin
arr(ax,(-3.0,-2.5,Z_MEM),(-5.5,-2.0,ZC),'#AEEA00',lw=1.3) # Laminin→Fyn
arr(ax,(-3.0,2.5,Z_MEM),(-3.0,2.0,ZC),'#00E676',lw=1.3)   # Cu2+→SOD
arr(ax,(5.0,0,Z_MEM),(5.8,-0.8,ZC),'#FF6D00',lw=2.0)       # PrPSc→ROS

# Cytoplasm→2nd Messenger
ZC2 = Z_CYT2+0.4
arr(ax,(-2.5,-2.0,Z_CYT),(-4.0,0,ZC2),'#69F0AE',lw=1.3)   # PI3K→mTOR
arr(ax,(-0.2,-2.0,Z_CYT),(-2.0,0,ZC2),'#F48FB1',lw=1.2)   # ERK→p38
arr(ax,(1.5,-2.0,Z_CYT),(0.5,0,ZC2),'#FFD180',lw=1.3)      # PTEN→GSK3b
arr(ax,(4.0,-0.5,Z_CYT),(3.0,-0.8,ZC2),'#FF6E40',lw=1.5)   # NMDAR→Ca2+
arr(ax,(1.8,2.0,Z_CYT),(5.5,1.2,ZC2),'#B2FF59',lw=1.2)    # Caveolin→Autophagy
arr(ax,(5.8,-0.8,Z_CYT),(5.5,1.0,ZC2),'#FF6D00',lw=1.5)   # ROS→Autophagy

# 2nd Messenger→Nucleus
ZN = Z_NUC+0.4
arr(ax,(-4.5,0,Z_CYT2),(-5.5,0,ZN),'#40C4FF',lw=1.4)      # mTOR→NF-κB
arr(ax,(-2.0,0,Z_CYT2),(-3.0,0,ZN),'#FF4081',lw=1.3)      # p38→p53
arr(ax,(0.5,0,Z_CYT2),(-0.5,0,ZN),'#69F0AE',lw=1.3)       # GSK3b→CREB
arr(ax,(0.5,0,Z_CYT2),(2.0,1.2,ZN),'#FFD180',lw=1.4)      # GSK3b→Tau
arr(ax,(3.0,-0.8,Z_CYT2),(2.0,-1.2,ZN),'#FF3333',lw=1.5)  # Ca2+→Bcl2
arr(ax,(5.5,1.5,Z_CYT2),(5.0,0,ZN),'#18FFFF',lw=1.3)      # Autophagy→BDNF

# Nucleus→Outcomes
ZO = Z_OUT+0.55
arr(ax,(-5.5,0,Z_NUC),(-5.0,0,ZO),'#00E5FF',lw=1.8)       # NF-κB→Neuro
arr(ax,(-0.5,0,Z_NUC),(0.0,0,ZO),'#69F0AE',lw=1.6)        # CREB→Synaptic
arr(ax,(5.0,0,Z_NUC),(1.5,0,ZO),'#18FFFF',lw=1.3)         # BDNF→Synaptic
arr(ax,(2.0,1.2,Z_NUC),(4.5,-0.3,ZO),'#FF6D00',lw=1.5)    # Tau→Neurodegen
arr(ax,(2.0,-1.2,Z_NUC),(4.5,-0.5,ZO),'#FF3333',lw=1.8)   # Bcl2→Neurodegen

# Pritamab cross-layer block line
ax.plot([-4.2,4.8],[0,0],[Z_MEM+0.3,Z_OUT+0.8],
        color='#FFD700',lw=1.3,ls='--',alpha=0.35)
ax.text(0.5, 0, (Z_MEM+Z_OUT)/2+0.8,
        'Pritamab\nblocks\nconversion',
        ha='center', fontsize=8, color='#FFD700',
        alpha=0.8, fontweight='bold')

# ══════════════════════════════════════════════════════════════════════
# 축 / 뷰
# ══════════════════════════════════════════════════════════════════════
ax.set_xlim(-7.5, 7.5)
ax.set_ylim(-5.0, 5.0)
ax.set_zlim(-2.5, 11.5)
ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
ax.set_xlabel(''); ax.set_ylabel(''); ax.set_zlabel('')
ax.xaxis.pane.fill = False; ax.yaxis.pane.fill = False; ax.zaxis.pane.fill = False
ax.xaxis.pane.set_edgecolor('#0C1625')
ax.yaxis.pane.set_edgecolor('#0C1625')
ax.zaxis.pane.set_edgecolor('#0C1625')
ax.grid(False)

# layer 라벨 (Z축 옆)
for z_lev, lbl, col in [
    (Z_MEM+0.6, '① Membrane',      '#4FC3F7'),
    (Z_CYT+0.6, '② Cytoplasm-Kin', '#39FF14'),
    (Z_CYT2+0.5,'③ Cyto-2ndMsg',   '#FF8C00'),
    (Z_NUC+0.5, '④ Nucleus',       '#FF69B4'),
    (Z_OUT+0.6, '⑤ Outcomes',      '#00FFFF'),
]:
    ax.text(-7.3, -4.8, z_lev, lbl, fontsize=8,
            color=col, alpha=0.80, fontweight='bold')

ax.view_init(elev=28, azim=-48)

# ══════════════════════════════════════════════════════════════════════
# 타이틀 / 범례
# ══════════════════════════════════════════════════════════════════════
fig.text(0.50, 0.975,
         'PrPC Signaling Pathway — 3D Layered Network',
         ha='center', va='top', fontsize=19,
         fontweight='bold', color='white')
fig.text(0.50, 0.950,
         'Membrane  →  Cytoplasm (Kinase / 2nd Messenger)  →  Nucleus  →  Outcomes',
         ha='center', va='top', fontsize=10, color='#8899BB', style='italic')

legend_items = [
    mpatches.Patch(color='#4FC3F7', label='PrPC  (core node)'),
    mpatches.Patch(color='#FFD700', label='Pritamab  (blocks epitope 142-170)'),
    mpatches.Patch(color='#FF3333', label='PrPSc / pathological'),
    mpatches.Patch(color='#AEEA00', label='Fyn / Src kinase'),
    mpatches.Patch(color='#00E5FF', label='PI3K / Akt / NF-κB'),
    mpatches.Patch(color='#EA80FC', label='NMDAR / Ca²⁺'),
    mpatches.Patch(color='#FFD180', label='Tau-P / GSK-3β'),
    mpatches.Patch(color='#69F0AE', label='CREB / mTOR (survival)'),
    mpatches.Patch(color='#FF1744', label='Neurodegeneration'),
    mpatches.Patch(color='#00FFFF', label='Neuroprotection'),
]
ax.legend(handles=legend_items, loc='lower left',
          fontsize=8.5, facecolor='#070A14',
          edgecolor='#2A3A55', labelcolor='#CCDDEE',
          framealpha=0.92, ncol=2, handlelength=1.3,
          bbox_to_anchor=(-0.04, -0.06))

plt.tight_layout()

# ── 3D→2D 투영 레이블 (항상 전면) ─────────────────────────────────────
fig.canvas.draw()   # 먼저 3D 장면 렌더링
for (x3,y3,z3, text, fsize, col) in _label_queue:
    # 3D 좌표 → 화면 2D 좌표
    x2d, y2d, _ = proj3d.proj_transform(x3, y3, z3, ax.get_proj())
    ax.annotate(text,
                xy=(x2d, y2d), xycoords='axes fraction',
                xytext=(0, 0), textcoords='offset points',
                ha='center', va='center',
                fontsize=fsize, color=col,
                fontweight='bold',
                annotation_clip=False)

fig.savefig(OUT, dpi=220, bbox_inches='tight',
            facecolor=BG, edgecolor='none')
plt.close()
print(f"Saved: {OUT}")
