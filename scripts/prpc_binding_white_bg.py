"""
prpc_binding_white_bg.py  ── v2
참조 이미지 그대로 재현, 흰색 배경
  Left : PrPC 나선 코일(핑크) + Laminin루프(초록) + Pritamab 삼각형(노랑) + 분자 라벨
  Right-top : Binding Energies 테이블
  Right-bot : Mechanism Summary
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Polygon
from matplotlib.gridspec import GridSpec

OUT = r"f:\ADDS\outputs\pritamab_pptx_figures\prpc_binding_white_bg.png"

fig = plt.figure(figsize=(16, 9), dpi=200, facecolor='white')
gs  = GridSpec(2, 2, figure=fig,
               width_ratios=[1.4, 1.0],
               height_ratios=[1.0, 0.82],
               hspace=0.06, wspace=0.04,
               left=0.02, right=0.98, top=0.97, bottom=0.03)

# ═══════════════════════════════════════════════
# LEFT PANEL (2행 span)
# ═══════════════════════════════════════════════
ax = fig.add_subplot(gs[:, 0])
ax.set_xlim(0, 10)
ax.set_ylim(0, 22)
ax.axis('off')
ax.set_facecolor('white')

# ── 나선형 코일: 사인파로 구불구불 ────────────────
# 세로 중심선 x=3.8, 위→아래 (y: 20 → 1)
t = np.linspace(0, 1, 600)
y_spine = 20 - t * 19        # 20 → 1
x_spine = 3.8 + np.sin(t * 2 * np.pi * 9) * 0.55   # ~9 cycle 구불

# 두꺼운 외곽 (하이라이트)
ax.plot(x_spine, y_spine, color='#FF69B4', lw=22, alpha=0.35,
        solid_capstyle='round', solid_joinstyle='round', zorder=2)
# 메인 핑크
ax.plot(x_spine, y_spine, color='#E91E8C', lw=16, alpha=0.95,
        solid_capstyle='round', solid_joinstyle='round', zorder=3)
# 밝은 하이라이트
ax.plot(x_spine, y_spine, color='#FF80D0', lw=5, alpha=0.4,
        solid_capstyle='round', solid_joinstyle='round', zorder=4)

# ── PrPC 레이블 ───────────────────────────────────
ax.text(3.5, 21.0, 'PrPC', ha='center', va='center',
        fontsize=26, fontweight='bold', color='#111111', zorder=10)

# ── 상단 Laminin β1 YGSR 루프 ─────────────────────
# y≈15 부근에 오른쪽으로 큰 루프
theta_u = np.linspace(np.pi * 0.05, np.pi * 1.05, 120)
lu_x = 5.5 + 2.0 * np.cos(theta_u)
lu_y = 14.8 + 3.0 * np.sin(theta_u)
ax.plot(lu_x, lu_y, color='#00C86E', lw=13, alpha=0.92,
        solid_capstyle='round', zorder=5)
ax.plot(lu_x, lu_y, color='#7FFFC4', lw=4, alpha=0.4,
        solid_capstyle='round', zorder=6)

# Laminin β1 (YGSR) 레이블 — 오른쪽
ax.text(8.3, 16.0, 'Laminin\nβ1 (YGSR)', ha='left', va='center',
        fontsize=12, fontweight='bold', color='#006B3C', zorder=10,
        bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                  edgecolor='#00C86E', lw=1.5, alpha=0.95))

# 화살표 루프→ 레이블
ax.annotate('', xy=(8.1, 15.5), xytext=(7.4, 15.2),
            arrowprops=dict(arrowstyle='->', color='#006B3C', lw=1.2,
                            mutation_scale=12), zorder=8)

# ── 하단 Laminin β1 루프 (작은) ───────────────────
theta_d = np.linspace(np.pi * 0.0, np.pi * 0.85, 80)
ld_x = 5.0 + 1.2 * np.cos(theta_d)
ld_y = 7.8 + 2.0 * np.sin(theta_d)
ax.plot(ld_x, ld_y, color='#00C86E', lw=10, alpha=0.88,
        solid_capstyle='round', zorder=5)
ax.plot(ld_x, ld_y, color='#7FFFC4', lw=3, alpha=0.4,
        solid_capstyle='round', zorder=6)

# Laminin β1 레이블 (하단)
ax.text(7.3, 10.2, 'Laminin β1', ha='left', va='center',
        fontsize=12, fontweight='bold', color='#006B3C', zorder=10)
ax.annotate('', xy=(6.1, 9.5), xytext=(7.1, 10.0),
            arrowprops=dict(arrowstyle='->', color='#006B3C', lw=1.2,
                            mutation_scale=10), zorder=8)

# ── Pritamab 노란 삼각형 (M144 영역) ─────────────
tri = Polygon([[4.7, 16.4], [6.5, 15.0], [5.6, 13.4]],
              closed=True,
              facecolor='#FFD600', edgecolor='#FF8C00',
              linewidth=2.5, zorder=7)
ax.add_patch(tri)
ax.text(5.6, 15.1, 'M 144', ha='center', va='center',
        fontsize=9.5, fontweight='bold', color='#1A1A1A', zorder=9)
# 아래방향 화살표 (삼각형 포인터)
ax.annotate('', xy=(5.6, 13.6), xytext=(5.6, 16.2),
            arrowprops=dict(arrowstyle='->', color='#FF8C00',
                            lw=4, mutation_scale=22), zorder=8)

# ── 우상단 텍스트: Pritamab blocks… ──────────────
ax.text(5.5, 19.5,
        'Pritamab blocks\nPrPC–LRP/LR\n& Laminin (YGSR) overlap',
        ha='left', va='center', fontsize=10, color='#111111', zorder=10)
ax.annotate('', xy=(5.7, 16.8), xytext=(6.8, 18.4),
            arrowprops=dict(arrowstyle='->', color='#333333',
                            lw=1.0, mutation_scale=10,
                            connectionstyle='arc3,rad=-0.2'), zorder=8)

# ── 좌측 텍스트 박스: Pritamab blocks (143–178) ──
ax.text(0.1, 16.2,
        'Pritamab\nblocks\nPrPC–LRP/LR\n& Laminin\n(143–178)\noverlap',
        ha='left', va='center', fontsize=8.5, color='#111111', zorder=10)

# ── 분자 상호작용 초록 라벨 (코일 오른쪽)──────────
green_labels = [
    (5.3, 12.7, 'Sail bridge'),
    (5.3, 12.1, 'R12*'),
    (5.3, 11.5, 'G5–F'),
    (5.3, 10.9, 'Fl6x'),
    (5.3, 10.3, 'vdW'),
    (5.0,  9.1, 'P179–E50'),
    (5.1,  8.4, 'Carborate'),
]
for gx, gy, gt in green_labels:
    ax.text(gx, gy, gt, ha='left', va='center',
            fontsize=8.5, color='#00A050', zorder=10)

# 화살표: 'Sail bridge' → 코일
ax.annotate('', xy=(4.3, 12.4), xytext=(5.1, 12.7),
            arrowprops=dict(arrowstyle='->', color='#00A050', lw=1.0,
                            mutation_scale=8), zorder=8)

# ── 코일 왼쪽 라벨들 ────────────────────────────
left_labels = [
    (2.5, 17.2, '○4V'),
    (1.8, 14.2, 'G154–Rix\nfvdW'),
    (2.2, 11.2, 'P179–E50'),
    (2.2,  8.5, 'P179–E50'),
]
for lx, ly, lt in left_labels:
    ax.text(lx, ly, lt, ha='center', va='center',
            fontsize=8.5, color='#00A050', zorder=10)

# ── 하단 PrPC–LRP/LR 레이블 ───────────────────────
ax.text(2.8, 2.5,
        'PrPC–LRP/LR\n& Laminin\n(143–79) onbap',
        ha='center', va='center', fontsize=9, color='#111111',
        fontweight='bold', zorder=10)


# ═══════════════════════════════════════════════
# RIGHT TOP — Binding Energies
# ═══════════════════════════════════════════════
ax_rt = fig.add_subplot(gs[0, 1])
ax_rt.set_xlim(0, 10); ax_rt.set_ylim(0, 10)
ax_rt.axis('off'); ax_rt.set_facecolor('white')

# 테두리
ax_rt.add_patch(FancyBboxPatch((0.15, 0.15), 9.7, 9.7,
                boxstyle='round,pad=0.1',
                facecolor='white', edgecolor='#BBBBBB', lw=1.5, zorder=1))

ax_rt.text(5.0, 9.2, 'Binding Energies', ha='center', va='center',
           fontsize=18, fontweight='bold', color='#111111', zorder=5)
ax_rt.text(5.0, 8.35, 'PrPC – Pritamab (epitope 144 179)',
           ha='center', va='center', fontsize=11.5, color='#333333', zorder=5)
ax_rt.plot([0.5, 9.5], [7.95, 7.95], color='#AAAAAA', lw=1.2, zorder=4)

rows = [
    ('Sali-bridge',   '– 30/kcal/mol'),
    ('π–π stacking',  '– 8.5 cal/mol'),
    ('H-bond',        '– 8 kcal/mol'),
    ('Electrostatic', '– 12.5 kal/mol'),
    ('vdW',           '– 4.3 kcal/mol'),
]
ys = [7.15, 6.25, 5.35, 4.45, 3.55]
for (name, val), y in zip(rows, ys):
    ax_rt.text(0.8, y, name, ha='left', va='center',
               fontsize=12, color='#222222', zorder=5)
    ax_rt.text(9.3, y, val, ha='right', va='center',
               fontsize=12, color='#222222', zorder=5)
    ax_rt.plot([0.5, 9.5], [y - 0.45, y - 0.45],
               color='#EEEEEE', lw=0.8, zorder=3)

ax_rt.plot([0.5, 9.5], [3.0, 3.0], color='#AAAAAA', lw=1.2, zorder=4)
ax_rt.text(0.8, 2.3, 'Total AG  :', ha='left', va='center',
           fontsize=13, fontweight='bold', color='#111111', zorder=5)
ax_rt.text(9.3, 2.3, '– 61.8 kcal/mol', ha='right', va='center',
           fontsize=13, fontweight='bold', color='#111111', zorder=5)
ax_rt.text(0.8, 1.35, 'KD  :', ha='left', va='center',
           fontsize=13, fontweight='bold', color='#111111', zorder=5)
ax_rt.text(9.3, 1.35, '– 0.1 – 0.5 nM', ha='right', va='center',
           fontsize=13, fontweight='bold', color='#111111', zorder=5)


# ═══════════════════════════════════════════════
# RIGHT BOTTOM — Mechanism Summary
# ═══════════════════════════════════════════════
ax_rb = fig.add_subplot(gs[1, 1])
ax_rb.set_xlim(0, 10); ax_rb.set_ylim(0, 8)
ax_rb.axis('off'); ax_rb.set_facecolor('white')

ax_rb.add_patch(FancyBboxPatch((0.15, 0.15), 9.7, 7.7,
                boxstyle='round,pad=0.1',
                facecolor='white', edgecolor='#BBBBBB', lw=1.5, zorder=1))

ax_rb.text(5.0, 7.45, 'Mechanism Summary', ha='center', va='center',
           fontsize=17, fontweight='bold', color='#111111', zorder=5)

# 행1: Laminin 81/YGSR → S1/5/ LR (청록 배경)
ax_rb.add_patch(FancyBboxPatch((0.4, 5.6), 9.2, 1.2,
                boxstyle='round,pad=0.08',
                facecolor='#00BFBF', edgecolor='#009999', lw=1.5, zorder=4))
ax_rb.text(5.0, 6.2, 'Laminin 81/YGSR  →  S1/5/ LR',
           ha='center', va='center',
           fontsize=12, fontweight='bold', color='white', zorder=5)

# 행2: PcPC –4OFF → (위쪽 화살표) Invasion OFF
ax_rb.text(1.5, 4.8, 'PcPC –4OFF', ha='left', va='center',
           fontsize=11, color='#111111', zorder=5)
ax_rb.annotate('', xy=(4.8, 4.8), xytext=(3.6, 4.8),
               arrowprops=dict(arrowstyle='->', color='#111111',
                               lw=2.0, mutation_scale=16), zorder=6)
# Invasion 박스
ax_rb.add_patch(FancyBboxPatch((5.0, 4.2), 2.2, 1.2,
                boxstyle='round,pad=0.06',
                facecolor='#0099CC', edgecolor='#006699', lw=1.2, zorder=4))
ax_rb.text(6.1, 4.8, 'Invasion', ha='center', va='center',
           fontsize=10, fontweight='bold', color='white', zorder=5)
# 위쪽 화살표 (↑)
ax_rb.annotate('', xy=(6.1, 5.5), xytext=(6.1, 4.3),
               arrowprops=dict(arrowstyle='->', color='white',
                               lw=1.5, mutation_scale=10), zorder=7)
# OFF 텍스트
ax_rb.text(8.0, 5.1, 'OFF', ha='left', va='center',
           fontsize=14, fontweight='bold', color='#111111', zorder=5)
ax_rb.annotate('', xy=(7.8, 4.8), xytext=(7.4, 4.8),
               arrowprops=dict(arrowstyle='->', color='#111111',
                               lw=1.5, mutation_scale=12), zorder=6)

# 행3: Apoptosis ↑ → Invesion ↑
ax_rb.add_patch(FancyBboxPatch((0.4, 2.3), 4.0, 1.5,
                boxstyle='round,pad=0.08',
                facecolor='#FF8C00', edgecolor='#CC6600', lw=1.5, zorder=4))
ax_rb.text(2.4, 3.05, 'Apoptosis ↑', ha='center', va='center',
           fontsize=13, fontweight='bold', color='white', zorder=5)

ax_rb.annotate('', xy=(5.1, 3.05), xytext=(4.6, 3.05),
               arrowprops=dict(arrowstyle='->', color='#222222',
                               lw=3, mutation_scale=20), zorder=6)

ax_rb.add_patch(FancyBboxPatch((5.3, 2.3), 4.3, 1.5,
                boxstyle='round,pad=0.08',
                facecolor='#CC4400', edgecolor='#993300', lw=1.5, zorder=4))
ax_rb.text(7.45, 3.05, 'Invesion ↑', ha='center', va='center',
           fontsize=13, fontweight='bold', color='white', zorder=5)


# ── 저장 ──────────────────────────────────────────
fig.savefig(OUT, dpi=300, bbox_inches='tight', facecolor='white')
plt.close()
print(f"Saved: {OUT}")
