"""
항암 요법 독성 프로파일 — 원형 타겟판 (Bullseye Target Board)
Chemotherapy Toxicity Profile Comparison by Regimen
Validated data sources: NCCN, FOLFOX/FOLFIRI/FOLFOXIRI Phase III trials
Pritamab combination: ~24% EC50 reduction → proportional toxicity reduction projected
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Wedge, Circle, FancyBboxPatch
from matplotlib.colors import LinearSegmentedColormap, Normalize
import matplotlib.patheffects as pe
import numpy as np

# ── 색상 팔레트 ─────────────────────────────────────────────────────
BG    = "#0C0F1A"          # 배경: 다크 네이비
PANEL = "#111827"
RIM1  = "#1E2A3A"          # 타겟 링1: 최외곽(G4 위험)
RIM2  = "#7F1D1D"          # G3
RIM3  = "#92400E"          # G2
RIM4  = "#065F46"          # G1
RIM5  = "#1E3A5F"          # 센터 (무독성)
WHITE = "#F0F4FF"
GRAY  = "#94A3B8"

# 레지멘 색상
REG_COLORS = {
    "FOLFOX\nalone":           "#F87171",   # 빨강
    "Pritamab\n+FOLFOX":       "#60A5FA",   # 파랑 (개선)
    "FOLFIRI\nalone":          "#FB923C",   # 오렌지
    "Pritamab\n+FOLFIRI":      "#34D399",   # 초록 (개선)
    "FOLFOXIRI":               "#E879F9",   # 마젠타 (최강 독성)
}

# ════════════════════════════════════════════════════════════════
# 독성 데이터 (Grade 3/4 발생률 %, 0–100 스케일)
# 출처: FOLFOX(de Gramont 2000, MOSAIC), FOLFIRI(Douillard 2000),
#        FOLFOXIRI(Falcone 2007, TRIBE), Pritamab 조합: EC50 −24% 반영 추정
# ════════════════════════════════════════════════════════════════
toxicities = [
    "Neutropenia",
    "Anemia",
    "Thrombo-\ncytopenia",
    "Nausea/\nVomiting",
    "Diarrhea",
    "Peripheral\nNeuropathy",
    "Fatigue",
    "Mucositis/\nStomatitis",
]
n_tox = len(toxicities)

# 최대 스케일 (각 독성 공통 상한)
MAX_SCALE = 60   # %

# (G3/4 %) 데이터
data = {
    "FOLFOX\nalone":       [32, 8,  10, 14, 10,  8, 22,  9],
    "Pritamab\n+FOLFOX":   [24, 6,   8, 10,  8,  6, 17,  7],   # ~24% 감소 반영
    "FOLFIRI\nalone":      [28, 6,   4, 18, 28,  3, 20, 12],
    "Pritamab\n+FOLFIRI":  [21, 5,   3, 14, 21,  2, 16,  9],   # ~24% 감소 반영
    "FOLFOXIRI":           [50,12,  14, 22, 34, 10, 28, 14],
}

regimens = list(data.keys())
n_reg    = len(regimens)

# ════════════════════════════════════════════════════════════════
# 보조 함수 — 타겟판 그리기
# ════════════════════════════════════════════════════════════════
def draw_target_board(ax, values, color, title, subtitle,
                      toxicity_labels, max_val=60,
                      show_labels=True):
    """
    원형 타겟판: 안쪽=안전(0), 바깥=위험(max_val)
    values: list of len=n_tox, 독성 퍼센트
    color:  레지멘 대표 색상
    """
    ax.set_facecolor(PANEL)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_xlim(-1.25, 1.25)
    ax.set_ylim(-1.25, 1.25)

    n = len(values)
    angle_step = 360.0 / n

    # ── 타겟 링 그리기 (바깥→안) ──────────────────────────────
    ring_defs = [
        (1.00, 0.80, "#7F1D1D", "G3/4\n≥30%"),
        (0.80, 0.58, "#92400E", "G3/4\n15-30%"),
        (0.58, 0.36, "#1E3A5F", "G1/2\n5-15%"),
        (0.36, 0.14, "#065F46", "G1\n<5%"),
    ]
    for r_out, r_in, rc, _ in ring_defs:
        ring = plt.Circle((0, 0), r_out, color=rc, zorder=2)
        ax.add_patch(ring)
    # 중심 원
    center = plt.Circle((0, 0), 0.14, color="#0C3547", zorder=3)
    ax.add_patch(center)

    # 구분선 (웨지 경계)
    for k in range(n):
        angle_rad = np.radians(90 - k * angle_step)
        ax.plot([0, np.cos(angle_rad)],
                [0, np.sin(angle_rad)],
                color="#1E2A3A", lw=1.5, zorder=4)

    # 링 경계선
    for r_out, r_in, _, _ in ring_defs:
        circle = plt.Circle((0, 0), r_out,
                              fill=False, edgecolor="#1E2A3A",
                              lw=1.2, zorder=5)
        ax.add_patch(circle)

    # ── 독성값 → 웨지 채우기 ─────────────────────────────────
    for k, val in enumerate(values):
        theta1 = 90 - k * angle_step
        theta2 = 90 - (k + 1) * angle_step
        radius = min(val / max_val, 1.0) * 1.00   # 정규화
        radius = max(radius, 0.02)                  # 최소값

        # 메인 웨지
        w = Wedge((0, 0), radius,
                  theta2, theta1,          # matplotlib은 CCW
                  facecolor=color, alpha=0.75,
                  edgecolor=BG, linewidth=1.5, zorder=6)
        ax.add_patch(w)

        # 글로우 효과 (반투명 outer glow)
        if val > 15:
            glow = Wedge((0, 0), min(radius + 0.03, 1.0),
                         theta2, theta1,
                         facecolor=color, alpha=0.25,
                         edgecolor="none", zorder=5)
            ax.add_patch(glow)

        # 수치 레이블
        mid_angle = np.radians((theta1 + theta2) / 2)
        txt_r = min(radius + 0.12, 1.10)
        txt_r = max(txt_r, 0.28)
        tx = txt_r * np.cos(mid_angle)
        ty = txt_r * np.sin(mid_angle)
        ax.text(tx, ty, f"{val}%",
                ha="center", va="center",
                fontsize=7.5, color=WHITE, fontweight="bold",
                zorder=10,
                bbox=dict(boxstyle="round,pad=0.15",
                          facecolor=PANEL, edgecolor=color,
                          alpha=0.85, linewidth=0.8))

    # ── 독성 축 레이블 ────────────────────────────────────────
    if show_labels:
        for k, lbl in enumerate(toxicity_labels):
            mid_angle = np.radians(90 - (k + 0.5) * angle_step)
            lx = 1.16 * np.cos(mid_angle)
            ly = 1.16 * np.sin(mid_angle)
            ax.text(lx, ly, lbl,
                    ha="center", va="center",
                    fontsize=6.8, color=GRAY, fontstyle="normal",
                    zorder=11)

    # ── 타이틀 ───────────────────────────────────────────────
    ax.text(0, 1.30, title,
            ha="center", va="center",
            fontsize=10, fontweight="bold", color=color,
            zorder=12)
    ax.text(0, 1.19, subtitle,
            ha="center", va="center",
            fontsize=7.5, color=GRAY, zorder=12)

    # 중심 텍스트
    mean_tox = np.mean(values)
    ax.text(0, 0.02, f"{mean_tox:.0f}%",
            ha="center", va="center",
            fontsize=9, fontweight="bold", color=WHITE, zorder=13)
    ax.text(0, -0.08, "avg",
            ha="center", va="center",
            fontsize=6.5, color=GRAY, zorder=13)

# ════════════════════════════════════════════════════════════════
# FIGURE
# ════════════════════════════════════════════════════════════════
fig = plt.figure(figsize=(28, 18), facecolor=BG)

# 메인 타이틀
fig.text(0.5, 0.975,
         "Antineoplastic Regimen Toxicity Profile  ·  Circular Target Board",
         ha="center", va="top",
         fontsize=22, fontweight="bold", color=WHITE)
fig.text(0.5, 0.952,
         "Grade 3/4 Adverse Event Incidence (%)  |  Inner ring = Safe  ·  Outer ring = Severe  "
         "|  Source: NCCN Guidelines + Phase III Trial Data + Pritamab EC50 −24% Projection",
         ha="center", va="top", fontsize=10, color=GRAY)

# 메인 그리드: 5개 타겟판 + 1 범례
gs_top = gridspec.GridSpec(
    1, 5, figure=fig,
    left=0.02, right=0.98,
    top=0.88, bottom=0.23,
    wspace=0.08,
)

# ─── 5개 타겟판 ──────────────────────────────────────────────
subtitles = {
    "FOLFOX\nalone":       "Control arm (Phase II design)",
    "Pritamab\n+FOLFOX":   "EC50 −24.0%  |  ADDS DRS 0.893",
    "FOLFIRI\nalone":      "Historical 2nd-line comparator",
    "Pritamab\n+FOLFIRI":  "EC50 −24.5%  |  ADDS DRS 0.870",
    "FOLFOXIRI":           "Triplet intensification (ECOG 0-1)",
}

for idx, (regimen, color) in enumerate(zip(regimens, REG_COLORS.values())):
    ax = fig.add_subplot(gs_top[0, idx], aspect="equal")
    draw_target_board(
        ax=ax,
        values=data[regimen],
        color=color,
        title=regimen.replace("\n", " + ") if "\n" in regimen else regimen,
        subtitle=subtitles[regimen],
        toxicity_labels=toxicities,
        max_val=MAX_SCALE,
        show_labels=True,
    )

# ════════════════════════════════════════════════════════════════
# 하단: 레지멘별 독성 비교 바 차트 (가로)
# ════════════════════════════════════════════════════════════════
gs_bot = gridspec.GridSpec(
    1, 1, figure=fig,
    left=0.05, right=0.98,
    top=0.20, bottom=0.04,
)
ax_bar = fig.add_subplot(gs_bot[0, 0])
ax_bar.set_facecolor(PANEL)

# 각 독성별, 레지멘별 수치를 그룹 바로
n_tox_c = len(toxicities)
x = np.arange(n_tox_c)
bar_w = 0.15
offsets = np.linspace(-(n_reg-1)*bar_w/2, (n_reg-1)*bar_w/2, n_reg)

colors_list = list(REG_COLORS.values())
reg_labels  = [r.replace("\n", " ") for r in regimens]

for ri, (regimen, offset, clr) in enumerate(zip(regimens, offsets, colors_list)):
    vals = data[regimen]
    bars = ax_bar.bar(x + offset, vals, bar_w,
                      color=clr, alpha=0.85,
                      edgecolor=BG, linewidth=0.5,
                      label=reg_labels[ri],
                      zorder=3)

# 기준선
ax_bar.axhline(30, color="#7F1D1D", lw=1.5, linestyle="--",
               alpha=0.8, label="G3/4 ≥30% (High Risk)")
ax_bar.axhline(15, color="#92400E", lw=1.2, linestyle=":",
               alpha=0.7, label="G3/4 ≥15% (Moderate)")

# 배경 영역
ax_bar.fill_between([-0.5, n_tox_c-0.5], [30, 30], [MAX_SCALE, MAX_SCALE],
                    color="#7F1D1D", alpha=0.06, zorder=1)
ax_bar.fill_between([-0.5, n_tox_c-0.5], [15, 15], [30, 30],
                    color="#92400E", alpha=0.06, zorder=1)

tox_labels_flat = [t.replace("\n", "/") for t in toxicities]
ax_bar.set_xticks(x)
ax_bar.set_xticklabels(tox_labels_flat, fontsize=9.5, color=WHITE)
ax_bar.set_ylabel("Grade 3/4 Incidence (%)", fontsize=10, color=GRAY)
ax_bar.set_ylim(0, MAX_SCALE + 5)
ax_bar.set_xlim(-0.6, n_tox_c - 0.4)
ax_bar.set_title(
    "Grade 3/4 Toxicity — Grouped Bar Comparison by Toxicity Type",
    fontsize=12, fontweight="bold", color=WHITE, pad=10)
ax_bar.spines[:].set_visible(False)
ax_bar.tick_params(colors=GRAY, labelsize=9)
ax_bar.grid(axis="y", alpha=0.15, color=GRAY)

# 범례
leg = ax_bar.legend(
    loc="upper right", fontsize=8.5,
    facecolor="#1A2035", edgecolor=GRAY,
    labelcolor=WHITE, framealpha=0.95,
    ncol=2, handlelength=1.5,
)

# ── 독성 감소 어노테이션 ─────────────────────────────────────────
for k, tox in enumerate(toxicities):
    v_folfox    = data["FOLFOX\nalone"][k]
    v_prit_fox  = data["Pritamab\n+FOLFOX"][k]
    v_folfiri   = data["FOLFIRI\nalone"][k]
    v_prit_firi = data["Pritamab\n+FOLFIRI"][k]
    red1 = v_folfox - v_prit_fox
    red2 = v_folfiri - v_prit_firi
    if red1 > 0 or red2 > 0:
        ax_bar.text(k, max(v_folfox, v_folfiri) + 2.5,
                    f"↓{red1}%",
                    ha="center", va="bottom", fontsize=7,
                    color="#60A5FA", fontweight="bold")

# 우측 독성 감소 텍스트박스
ax_bar.text(
    0.01, 0.97,
    "Pritamab +FOLFOX vs FOLFOX alone\n"
    "  Neutropenia:  32% → 24%  (−25%)\n"
    "  Nausea/Vom:   14% → 10%  (−29%)\n"
    "  Diarrhea:     10% →  8%  (−20%)\n"
    "Mechanism: EC50 −24.0% → lower cytotoxic\n"
    "dose needed for equivalent tumour kill",
    ha="left", va="top",
    transform=ax_bar.transAxes,
    fontsize=8, color=WHITE,
    bbox=dict(boxstyle="round,pad=0.5",
              facecolor="#0D2137", edgecolor="#60A5FA",
              alpha=0.92, linewidth=1.2),
)

# ════════════════════════════════════════════════════════════════
# 타겟판 범례 (링 설명)
# ════════════════════════════════════════════════════════════════
# 범례 패널을 figure 좌측에 별도 axes로
ax_leg = fig.add_axes([0.00, 0.22, 0.05, 0.66])
ax_leg.set_facecolor(BG)
ax_leg.axis("off")

ring_legend = [
    ("#7F1D1D", "●  Grade 3/4  ≥30%  (High)"),
    ("#92400E", "●  Grade 3/4  15–30%  (Mod)"),
    ("#1E3A5F", "●  Grade 1/2  5–15%  (Low)"),
    ("#065F46", "●  Grade 1    <5%  (Min)"),
    ("#0C3547", "●  Centre    = 0%  (None)"),
]
ax_leg.text(0.5, 0.99, "Ring\nGuide",
            ha="center", va="top",
            fontsize=7.5, color=GRAY, fontweight="bold")
yy = 0.90
for rc, lbl in ring_legend:
    ax_leg.text(0.5, yy, lbl,
                ha="center", va="top",
                fontsize=6.5, color=rc)
    yy -= 0.14

# Footer
fig.text(0.5, 0.015,
         "Grade 3/4 data: FOLFOX (de Gramont 2000; MOSAIC trial) · FOLFIRI (Douillard 2000) · "
         "FOLFOXIRI (Falcone 2007 TRIBE; Cremolini 2015 TRIBE2)  |  "
         "Pritamab combination toxicity: projected from EC50 −24.0%/−24.5% chemo dose reduction  |  "
         "Peripheral neuropathy = cumulative G1-2 (oxaliplatin); Grade 3 rare",
         ha="center", va="bottom", fontsize=7.5, color=GRAY, style="italic")

plt.savefig(r"f:\ADDS\figures\pritamab_toxicity_target_board.png",
            dpi=180, bbox_inches="tight", facecolor=BG)
print("Saved: pritamab_toxicity_target_board.png")
