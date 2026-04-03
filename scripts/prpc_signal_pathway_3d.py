"""
prpc_signal_pathway_3d.py
==========================
PrPC 전체 시그널 패스웨이를 2.5D 스타일로 시각화
- 평면 기반이지만 3D 그림자·원근감·레이어 효과 적용
- Pritamab 차단/활성화 경로 모두 포함
- Dark theme, publication quality
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from matplotlib.colors import LinearSegmentedColormap
import numpy as np

OUT = r"f:\ADDS\outputs\pritamab_pptx_figures\prpc_signal_pathway_3d.png"

# ── 캔버스 ─────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(22, 14), facecolor='#04060F')
ax.set_facecolor('#04060F')
ax.set_xlim(0, 22); ax.set_ylim(0, 14)
ax.set_aspect('equal'); ax.axis('off')

# ── 헬퍼 함수 ──────────────────────────────────────────────────────────

def shadow_box(ax, x, y, w, h, fc, ec, lw=1.5, radius=0.35,
               shadow_offset=0.15, alpha=0.95, zorder=5):
    """3D 그림자 효과 박스"""
    # Shadow
    shadow = FancyBboxPatch(
        (x + shadow_offset, y - shadow_offset), w, h,
        boxstyle=f"round,pad=0", linewidth=0,
        facecolor='#000000', alpha=0.45, zorder=zorder-1
    )
    ax.add_patch(shadow)
    # Body
    box = FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad={radius}", linewidth=lw,
        facecolor=fc, edgecolor=ec, alpha=alpha, zorder=zorder
    )
    ax.add_patch(box)

def node(ax, x, y, w, h, label, fc, ec, fontsize=9, fontcolor='white',
         sublabel=None, zorder=6, bold=True):
    """노드: 그림자 박스 + 레이블"""
    shadow_box(ax, x, y, w, h, fc=fc, ec=ec, zorder=zorder)
    ty = y + h/2 + (0.10 if sublabel else 0)
    ax.text(x + w/2, ty, label,
            ha='center', va='center', fontsize=fontsize,
            fontweight='bold' if bold else 'normal',
            color=fontcolor, zorder=zorder+1)
    if sublabel:
        ax.text(x + w/2, y + h/2 - 0.22, sublabel,
                ha='center', va='center', fontsize=fontsize - 2,
                color=fontcolor, alpha=0.75, zorder=zorder+1)

def arrow(ax, x1, y1, x2, y2, color='#8899BB', lw=2,
          style='simple', label=None, label_color=None,
          dashed=False, zorder=4, arrowsize=14):
    """화살표"""
    ls = (0, (5, 4)) if dashed else 'solid'
    ax.annotate('',
        xy=(x2, y2), xytext=(x1, y1),
        arrowprops=dict(
            arrowstyle=f'-|>', color=color,
            lw=lw, mutation_scale=arrowsize,
            linestyle=ls,
            connectionstyle='arc3,rad=0.0'
        ), zorder=zorder
    )
    if label:
        mx, my = (x1 + x2) / 2, (y1 + y2) / 2
        ax.text(mx, my + 0.18, label,
                ha='center', va='bottom',
                fontsize=7.5, color=label_color or color,
                style='italic', zorder=zorder+1)

def block_arrow(ax, x1, y1, x2, y2, color='#FF4444', lw=2.5, label='blocks'):
    """차단 선 (T-bar)"""
    mx, my = (x1+x2)/2, (y1+y2)/2
    ax.plot([x1, x2], [y1, y2], color=color, lw=lw, zorder=4,
            linestyle='--')
    # T-bar
    dx = y2 - y1; dy = x1 - x2
    norm = np.sqrt(dx**2 + dy**2) + 1e-9
    dx /= norm; dy /= norm
    ax.plot([x2 - dx*0.3, x2 + dx*0.3],
            [y2 - dy*0.3, y2 + dy*0.3],
            color=color, lw=3.5, zorder=5)
    ax.text(mx + 0.15, my + 0.15, label,
            ha='center', fontsize=7.5, color=color,
            fontweight='bold', zorder=5)

def section_bg(ax, x, y, w, h, color, label=None, alpha=0.06, zorder=1):
    """섹션 배경"""
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.1", linewidth=1.5,
        facecolor=color, edgecolor=color,
        alpha=alpha, zorder=zorder
    ))
    if label:
        ax.text(x + 0.2, y + h - 0.28, label,
                ha='left', va='top', fontsize=8,
                color=color, alpha=0.6, fontweight='bold', zorder=zorder+1)

# ═══════════════════════════════════════════════════════════════════════
# 섹션 배경 (구역 구분)
# ═══════════════════════════════════════════════════════════════════════
section_bg(ax, 0.3,  9.8, 21.4, 4.0, '#00BFFF', 'EXTRACELLULAR / MEMBRANE',  alpha=0.05)
section_bg(ax, 0.3,  5.5, 21.4, 4.1, '#7CFC00', 'CYTOPLASM / SIGNALING CASCADES', alpha=0.04)
section_bg(ax, 0.3,  0.4, 21.4, 5.0, '#FF8C00', 'NUCLEUS / DOWNSTREAM OUTCOMES', alpha=0.04)

# ── 섹션 레이블 라인
for y_line, col in [(9.8, '#1E90FF'), (5.5, '#39FF14'), (0.4, '#FF8C00')]:
    ax.plot([0.3, 21.7], [y_line, y_line], color=col, lw=0.8, alpha=0.3, zorder=2)

# ═══════════════════════════════════════════════════════════════════════
# TIER 1: 세포 표면 / GPI 앵커
# ═══════════════════════════════════════════════════════════════════════

# PrPC 메인 노드 (중앙)
node(ax, 9.7, 11.0, 2.6, 0.95,
     label='PrPC', sublabel='GPI-anchored (Res 119-231)',
     fc='#0D1F3C', ec='#4FC3F7', fontsize=12, zorder=10)

# Pritamab (항체)
node(ax, 0.6, 11.1, 2.2, 0.75,
     label='Pritamab\n(ICSM18-like)', fc='#1A0D2E', ec='#FFD700',
     fontsize=9, fontcolor='#FFD700', zorder=10)

# PrPSc (병리)
node(ax, 18.2, 11.1, 3.0, 0.75,
     label='PrPSc / Aggregates', fc='#2D0000', ec='#FF3333',
     fontsize=9, fontcolor='#FF6666', zorder=10)

# Cu2+ binding
node(ax, 5.2, 11.8, 1.8, 0.60,
     label='Cu²⁺ Binding\n(His 96,111)', fc='#102800', ec='#76FF03',
     fontsize=7.5, zorder=8)

# GPI raft
node(ax, 14.2, 11.8, 2.0, 0.60,
     label='Lipid Raft\n(GPI Anchor)', fc='#0F1525', ec='#90CAF9',
     fontsize=7.5, zorder=8)

# Laminin receptor
node(ax, 5.2, 10.0, 2.0, 0.65,
     label='Laminin-Rec\n(LRP1/LRP2)', fc='#0D1E20', ec='#26C6DA',
     fontsize=7.5, zorder=8)

# mGluR5
node(ax, 14.2, 10.0, 1.8, 0.65,
     label='mGluR5\n(receptor)', fc='#1C0D25', ec='#CE93D8',
     fontsize=7.5, zorder=8)

# NCAM
node(ax, 9.7, 9.85, 1.6, 0.60,
     label='NCAM1\n(co-receptor)', fc='#0D2025', ec='#80DEEA',
     fontsize=7.5, zorder=8)

# ═══════════════════════════════════════════════════════════════════════
# TIER 2: Cytoplasmic 신호 전달
# ═══════════════════════════════════════════════════════════════════════

# Fyn Kinase
node(ax, 1.2, 7.7, 1.8, 0.70,
     label='Fyn Kinase', fc='#1A2500', ec='#AEEA00',
     fontsize=9, zorder=8)

# PI3K/Akt
node(ax, 3.8, 7.7, 1.8, 0.70,
     label='PI3K / Akt', fc='#002820', ec='#00E5FF',
     fontsize=9, zorder=8)

# ERK 1/2
node(ax, 6.5, 7.7, 1.6, 0.70,
     label='ERK 1/2', fc='#1C2800', ec='#CCFF90',
     fontsize=9, zorder=8)

# PTEN
node(ax, 8.8, 7.7, 1.5, 0.70,
     label='PTEN', fc='#260A00', ec='#FFAB40',
     fontsize=9, zorder=8)

# NMDAR
node(ax, 11.0, 7.7, 1.8, 0.70,
     label='NMDAR\n(NR2B)', fc='#1A0025', ec='#EA80FC',
     fontsize=9, zorder=8)

# Caveolin-1
node(ax, 13.5, 7.7, 1.9, 0.70,
     label='Caveolin-1', fc='#001A20', ec='#84FFFF',
     fontsize=9, zorder=8)

# Src Kinase
node(ax, 16.0, 7.7, 1.8, 0.70,
     label='Src Kinase', fc='#1A1A00', ec='#FFD740',
     fontsize=9, zorder=8)

# ROS / Oxidative
node(ax, 18.5, 7.7, 2.5, 0.70,
     label='ROS ↑\nOxidative Stress', fc='#2D0000', ec='#FF6D00',
     fontsize=8, fontcolor='#FFAB40', zorder=8)

# mTOR
node(ax, 3.8, 6.3, 1.8, 0.70,
     label='mTOR', fc='#001F10', ec='#69F0AE',
     fontsize=9, zorder=8)

# p38 MAPK
node(ax, 6.5, 6.3, 1.6, 0.70,
     label='p38 MAPK', fc='#200020', ec='#F48FB1',
     fontsize=9, zorder=8)

# GSK-3β
node(ax, 8.8, 6.3, 1.6, 0.70,
     label='GSK-3β', fc='#261500', ec='#FFD180',
     fontsize=9, zorder=8)

# Ca²⁺ influx
node(ax, 11.0, 6.2, 1.8, 0.75,
     label='Ca²⁺ Influx\n(excitotoxicity)', fc='#2D0D00', ec='#FF6E40',
     fontsize=8, fontcolor='#FFAB91', zorder=8)

# SOD1/SOD2
node(ax, 1.2, 6.3, 1.8, 0.70,
     label='SOD1 / SOD2\n(antioxidant)', fc='#00200A', ec='#00E676',
     fontsize=7.5, zorder=8)

# ═══════════════════════════════════════════════════════════════════════
# TIER 3: 핵 / 최종 아웃컴
# ═══════════════════════════════════════════════════════════════════════

# NF-κB
node(ax, 1.0, 3.5, 1.9, 0.80,
     label='NF-κB\n(survival)', fc='#001A38', ec='#40C4FF',
     fontsize=9, zorder=8)

# p53
node(ax, 3.5, 3.5, 1.6, 0.80,
     label='p53\n(apoptosis)', fc='#200010', ec='#FF4081',
     fontsize=9, zorder=8)

# CREB
node(ax, 5.8, 3.5, 1.6, 0.80,
     label='CREB\n(memory)', fc='#001F10', ec='#69F0AE',
     fontsize=9, zorder=8)

# Tau (hyperphosphorylated)
node(ax, 8.1, 3.5, 2.1, 0.80,
     label='Tau phospho\n(tangle)', fc='#260800', ec='#FF6D00',
     fontsize=9, fontcolor='#FFAB40', zorder=8)

# Bcl-2 / caspase
node(ax, 11.0, 3.5, 2.0, 0.80,
     label='Bcl-2 / Casp-3\n(apoptosis)', fc='#2D0000', ec='#FF3333',
     fontsize=9, fontcolor='#FF8A80', zorder=8)

# BDNF
node(ax, 13.5, 3.5, 1.8, 0.80,
     label='BDNF / NGF\n(neuroprotection)', fc='#001A38', ec='#18FFFF',
     fontsize=9, fontcolor='#80DEEA', zorder=8)

# Autophagy
node(ax, 16.0, 3.5, 2.0, 0.80,
     label='Autophagy\n(UPS clearance)', fc='#101A00', ec='#B2FF59',
     fontsize=9, zorder=8)

# Prion propagation
node(ax, 18.5, 3.5, 3.0, 0.80,
     label='Prion Propagation\nNeuronal Death', fc='#2D0000', ec='#FF1744',
     fontsize=9, fontcolor='#FF5252', zorder=8)

# ── OUTCOMES 최하단
node(ax, 1.4, 1.5, 3.5, 0.85,
     label='Neuroprotection\n(normal PrPC fn)', fc='#001A20', ec='#00E5FF',
     fontsize=10, fontcolor='#80DEEA', zorder=9)

node(ax, 6.4, 1.5, 4.0, 0.85,
     label='Synaptic Plasticity\n& Memory (LTP/LTD)', fc='#001F10', ec='#69F0AE',
     fontsize=10, fontcolor='#B9F6CA', zorder=9)

node(ax, 11.4, 1.5, 4.0, 0.85,
     label='Neurodegeneration\n(prion / AD / PD)', fc='#2D0000', ec='#FF1744',
     fontsize=10, fontcolor='#FF8A80', zorder=9)

node(ax, 16.8, 1.5, 4.5, 0.85,
     label='Pritamab Target:\nBlock PrPC→PrPSc conversion',
     fc='#1A100A', ec='#FFD700',
     fontsize=10, fontcolor='#FFD700', zorder=9)

# ═══════════════════════════════════════════════════════════════════════
# 화살표 연결 (TIER 1)
# ═══════════════════════════════════════════════════════════════════════

# Pritamab → PrPC (block)
block_arrow(ax, 2.8, 11.48, 9.7, 11.48, color='#FFD700', lw=2.5,
            label='Blocks Epitope\n142-170')

# PrPC → PrPSc
arrow(ax, 12.3, 11.48, 18.2, 11.48, color='#FF3333', lw=2.2,
      label='misfolding', label_color='#FF6666')

# PrPC → Cu2+
arrow(ax, 9.7, 11.6, 7.0, 11.95, color='#76FF03', lw=1.5,
      label='binds')

# PrPC → Lipid Raft
arrow(ax, 12.3, 11.7, 14.2, 11.95, color='#90CAF9', lw=1.5)

# PrPC → Laminin
arrow(ax, 10.3, 11.0, 6.2, 10.45, color='#26C6DA', lw=1.5)

# PrPC → NCAM
arrow(ax, 11.0, 11.0, 10.5, 10.45, color='#80DEEA', lw=1.5)

# PrPC → mGluR5
arrow(ax, 11.5, 11.0, 15.2, 10.45, color='#CE93D8', lw=1.5)

# ═══════════════════════════════════════════════════════════════════════
# 화살표 연결 (TIER 1 → TIER 2)
# ═══════════════════════════════════════════════════════════════════════

# Cu2+ → SOD
arrow(ax, 5.8, 11.8, 2.1, 7.9, color='#76FF03', lw=1.4,
      label='antioxidant\ndefense')

# Laminin/LRP → Fyn
arrow(ax, 6.0, 10.0, 2.1, 8.05, color='#26C6DA', lw=1.4)

# LRP → PI3K
arrow(ax, 6.5, 10.0, 4.7, 8.05, color='#00E5FF', lw=1.4)

# NCAM → ERK
arrow(ax, 10.5, 9.85, 7.3, 8.05, color='#CCFF90', lw=1.4)

# mGluR5 → Ca2+
arrow(ax, 15.0, 10.0, 11.9, 7.8, color='#EA80FC', lw=1.4,
      label='Ca²⁺ → excit.')

# mGluR5 → Caveolin
arrow(ax, 15.3, 10.0, 14.4, 8.05, color='#84FFFF', lw=1.4)

# GPI raft → Src
arrow(ax, 15.5, 11.8, 16.9, 8.05, color='#FFD740', lw=1.4)

# PrPSc → ROS
arrow(ax, 19.7, 11.1, 19.7, 8.05, color='#FF6D00', lw=2.0,
      label='oxidative', label_color='#FF6D00')

# ═══════════════════════════════════════════════════════════════════════
# 화살표 연결 (TIER 2 ↔ TIER 2)
# ═══════════════════════════════════════════════════════════════════════
arrow(ax, 2.1, 7.7, 3.8, 7.7, color='#AEEA00', lw=1.5)   # Fyn → PI3K
arrow(ax, 5.6, 7.7, 6.5, 7.7, color='#00E5FF', lw=1.5)   # PI3K → ERK
arrow(ax, 4.7, 7.7, 3.8, 6.3, color='#69F0AE', lw=1.3)   # PI3K → mTOR
arrow(ax, 7.3, 7.7, 6.5, 6.3, color='#F48FB1', lw=1.3)   # ERK → p38
arrow(ax, 8.8, 7.7, 8.8, 6.3, color='#FFD180', lw=1.3)   # PTEN → GSK3b
arrow(ax, 11.9, 7.7, 11.9, 6.85, color='#FF6E40', lw=1.5) # NMDAR → Ca2+
arrow(ax, 17.0, 7.7, 19.0, 7.7, color='#FF6D00', lw=1.3) # Src → ROS
arrow(ax, 2.1, 7.7, 2.1, 6.65, color='#00E676', lw=1.3)  # Fyn → SOD

# ═══════════════════════════════════════════════════════════════════════
# 화살표 연결 (TIER 2 → TIER 3)
# ═══════════════════════════════════════════════════════════════════════
arrow(ax, 2.1, 7.7, 1.9, 4.05, color='#40C4FF', lw=1.5,
      label='NF-κB act.', label_color='#40C4FF')

arrow(ax, 4.7, 6.3, 4.3, 4.05, color='#FF4081', lw=1.4)  # mTOR → p53

arrow(ax, 5.3, 6.3, 6.6, 4.05, color='#69F0AE', lw=1.4)  # p38 ↔ CREB

arrow(ax, 9.6, 6.3, 9.2, 4.05, color='#FFD180', lw=1.5,
      label='Tau hyper-P', label_color='#FFD180')  # GSK3b → Tau

arrow(ax, 11.9, 6.3, 12.0, 4.05, color='#FF3333', lw=1.5) # Ca2+ → Bcl-2

arrow(ax, 14.4, 7.7, 14.4, 4.05, color='#18FFFF', lw=1.4,
      label='BDNF↑', label_color='#18FFFF')         # Caveolin → BDNF

arrow(ax, 17.0, 7.7, 17.0, 4.05, color='#B2FF59', lw=1.4) # Src → Autophagy

arrow(ax, 19.7, 7.7, 20.0, 4.05, color='#FF1744', lw=2.0) # ROS → Prion

# ═══════════════════════════════════════════════════════════════════════
# 화살표 연결 (TIER 3 → OUTCOMES)
# ═══════════════════════════════════════════════════════════════════════
arrow(ax, 1.9, 3.5, 3.0, 2.15, color='#00E5FF', lw=1.8)  # NF-κB → Neuro
arrow(ax, 4.3, 3.5, 6.5, 2.15, color='#69F0AE', lw=1.6)  # p53 → Synaptic
arrow(ax, 6.6, 3.5, 7.5, 2.15, color='#69F0AE', lw=1.6)  # CREB → Synaptic
arrow(ax, 9.2, 3.5, 12.5, 2.15, color='#FF6D00', lw=1.6) # Tau → Neurodegen
arrow(ax, 12.0, 3.5, 13.0, 2.15, color='#FF3333', lw=1.8) # Bcl-2 → Neurodegen
arrow(ax, 14.4, 3.5, 7.5, 2.15, color='#18FFFF', lw=1.4) # BDNF → Synaptic
arrow(ax, 17.0, 3.5, 18.0, 2.15, color='#B2FF59', lw=1.4) # Autophagy → Pritamab
arrow(ax, 20.0, 3.5, 20.5, 2.15, color='#FF1744', lw=2.0) # Prion → Neurodegen

# Pritamab → 최종 차단
block_arrow(ax, 16.8, 11.1, 19.8, 2.4, color='#FFD700', lw=2.0,
            label='Pritamab\nblocks')

# ═══════════════════════════════════════════════════════════════════════
# 그라디언트 막 라인 (세포막 표현)
# ═══════════════════════════════════════════════════════════════════════
for y_mem, col, alpha_val in [(9.8, '#4FC3F7', 0.25), (9.65, '#1565C0', 0.15)]:
    ax.plot([0.3, 21.7], [y_mem, y_mem], color=col, lw=2.0,
            alpha=alpha_val, linestyle='-', zorder=3)
ax.text(21.3, 9.9, 'Plasma\nMembrane', ha='right', va='bottom',
        fontsize=7.5, color='#4FC3F7', alpha=0.7, zorder=5)

# Nuclear envelope
for y_nuc, col, alpha_val in [(5.5, '#FF8C00', 0.20), (5.37, '#E65100', 0.12)]:
    ax.plot([0.3, 21.7], [y_nuc, y_nuc], color=col, lw=1.5,
            alpha=alpha_val, linestyle='-.', zorder=3)
ax.text(21.3, 5.6, 'Nuclear\nEnvelope', ha='right', va='bottom',
        fontsize=7.5, color='#FF8C00', alpha=0.7, zorder=5)

# ═══════════════════════════════════════════════════════════════════════
# 타이틀 / 범례
# ═══════════════════════════════════════════════════════════════════════
fig.text(0.50, 0.985, 'PrPC Signaling Pathway — Complete Network',
         ha='center', va='top', fontsize=18, fontweight='bold', color='white')
fig.text(0.50, 0.962,
         'GPI-anchored PrPC | Copper homeostasis  ·  Fyn/PI3K/mTOR  ·  NMDAR/Ca²⁺  ·  '
         'Tau/Bcl-2/Caspase  ·  Prion conversion  |  Pritamab epitope 142-170',
         ha='center', va='top', fontsize=9.5, color='#8899BB', style='italic')

# 범례
legend_items = [
    (mpatches.Patch(color='#4FC3F7'),  'PrPC core node'),
    (mpatches.Patch(color='#FFD700'),  'Pritamab antibody / target'),
    (mpatches.Patch(color='#FF3333'),  'Pathological / PrPSc'),
    (mpatches.Patch(color='#00E5FF'),  'Neuroprotective pathway'),
    (mpatches.Patch(color='#69F0AE'),  'Synaptic plasticity'),
    (mpatches.Patch(color='#CCFF90'),  'MAPK/ERK cascade'),
    (mpatches.Patch(color='#FF6E40'),  'Ca²⁺ / excitotoxicity'),
    (mpatches.Patch(color='#FFD180'),  'Tau phosphorylation'),
    (mpatches.Patch(color='#FF1744'),  'Neurodegeneration / death'),
]
leg = ax.legend(
    [h for h, _ in legend_items],
    [l for _, l in legend_items],
    loc='lower right', fontsize=8.0,
    framealpha=0.88, facecolor='#080C1A',
    edgecolor='#2A3A55', labelcolor='#CCDDEE',
    bbox_to_anchor=(1.0, 0.01), ncol=3,
    handlelength=1.2, handleheight=0.9
)

plt.tight_layout(rect=[0, 0, 1, 0.955])
fig.savefig(OUT, dpi=220, bbox_inches='tight', facecolor='#04060F')
plt.close()
print(f"Saved: {OUT}")
