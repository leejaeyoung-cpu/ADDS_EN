"""
figure_v3_final.py
───────────────────────────────────────────────────────
Complete rebuild with strict grid layout.
All boxes placed on an explicit grid → no overlaps.
Connectors routed through safe waypoints → no box crossings.
Bilingual (Korean + English) suitable for NRF proposal.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import matplotlib.patheffects as pe
import numpy as np

# ── Font ─────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":       "Malgun Gothic",
    "axes.unicode_minus": False,
    "figure.facecolor":  "white",
    "savefig.facecolor": "white",
})

# ── Canvas ───────────────────────────────────────────────────────────────────
FW, FH = 22, 17          # figure width / height in inches (at 300 dpi)
fig, ax = plt.subplots(figsize=(FW, FH), dpi=300)
ax.set_xlim(0, FW)
ax.set_ylim(0, FH)
ax.axis("off")

# ── Palette ──────────────────────────────────────────────────────────────────
P = {
    "bg":       "#F8FAFC",
    "border":   "#CBD5E1",
    "txt":      "#0F172A",
    "txt2":     "#475569",
    "txt3":     "#94A3B8",
    # Blue: basic/preclinical team
    "b0": "#1E3A8A", "b1": "#2563EB", "b2": "#DBEAFE", "b3": "#EFF6FF",
    # Green: clinical team
    "g0": "#064E3B", "g1": "#059669", "g2": "#D1FAE5", "g3": "#ECFDF5",
    # Purple: ADDS
    "v0": "#4C1D95", "v1": "#7C3AED", "v2": "#EDE9FE", "v3": "#F5F3FF",
    # Amber: output/validation
    "a0": "#78350F", "a1": "#D97706", "a2": "#FEF3C7",
    # Neutral white card
    "card": "#FFFFFF",
}

fig.patch.set_facecolor(P["bg"])
ax.set_facecolor(P["bg"])

# ─────────────────────────────────────────────────────────────────────────────
# HELPER FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

def rb(x, y, w, h, fc, ec, lw=1.5, r=0.12, z=2, alpha=1.0):
    """Draw a Rounded Box at (x,y) with width w and height h."""
    bp = FancyBboxPatch((x, y), w, h,
                        boxstyle=f"round,pad={r}",
                        facecolor=fc, edgecolor=ec,
                        linewidth=lw, alpha=alpha, zorder=z)
    ax.add_patch(bp)

def txt(x, y, s, sz=9, c=None, bold=False, ha="center", va="center", z=4,
        wrap_width=None, ls=1.5):
    c = c or P["txt"]
    fw = "bold" if bold else "normal"
    ax.text(x, y, s, fontsize=sz, color=c, fontweight=fw,
            ha=ha, va=va, zorder=z, linespacing=ls)

def arr_v(x, y_start, y_end, color="#94A3B8", lw=2.5, z=3):
    """Vertical arrow from y_start to y_end at x."""
    ax.annotate("", xy=(x, y_end), xytext=(x, y_start),
                arrowprops=dict(arrowstyle="-|>,head_width=0.45,head_length=0.55",
                                color=color, lw=lw),
                zorder=z)

def arr_h(y, x_start, x_end, color="#94A3B8", lw=2.5, z=3):
    """Horizontal arrow from x_start to x_end at y."""
    ax.annotate("", xy=(x_end, y), xytext=(x_start, y),
                arrowprops=dict(arrowstyle="-|>,head_width=0.45,head_length=0.55",
                                color=color, lw=lw),
                zorder=z)

def elbow(x1, y1, x2, y2, color="#94A3B8", lw=2.5, z=3, corner="h-v"):
    """L-shaped connector: 'h-v' goes horizontal first then vertical."""
    if corner == "h-v":
        ax.plot([x1, x2], [y1, y1], color=color, lw=lw, zorder=z)
        ax.annotate("", xy=(x2, y2), xytext=(x2, y1),
                    arrowprops=dict(arrowstyle="-|>,head_width=0.45,head_length=0.55",
                                    color=color, lw=lw),
                    zorder=z)
    else:  # v-h
        ax.plot([x1, x1], [y1, y2], color=color, lw=lw, zorder=z)
        ax.annotate("", xy=(x2, y2), xytext=(x1, y2),
                    arrowprops=dict(arrowstyle="-|>,head_width=0.45,head_length=0.55",
                                    color=color, lw=lw),
                    zorder=z)

def divider(x, y, w, c="#E2E8F0", lw=1):
    ax.plot([x, x+w], [y, y], color=c, lw=lw, zorder=3)

# ─────────────────────────────────────────────────────────────────────────────
# GRID DEFINITION
#
#  Row layout (y from top to bottom):
#  [Title] 16.3 – 17.0
#  [Timeline] 14.8 – 15.8
#  [Team Header Row] 13.3 – 14.4
#  [Module Row 1] 11.4 – 12.9
#  [Module Row 2] 9.5 – 11.0
#  [Module Row 3] 7.6 – 9.1
#  [ADDS core] 3.5 – 7.2   (spans all 3 modules vertically)
#  [Output] 1.2 – 3.1
#  [Footer] 0.5
#
#  Column layout (x):
#  Left Team:  0.8 – 8.0   (width 7.2)
#  Gap Left:   8.0 – 8.8
#  ADDS:       8.8 – 13.2  (width 4.4)  — placed right side of ADDS box
#  Gap Right: 13.2 – 14.0
#  Right Team: 14.0 – 21.2 (width 7.2)
# ─────────────────────────────────────────────────────────────────────────────

LX, LW = 0.8, 7.2        # Left team x, width
RX, RW = 14.0, 7.2       # Right team x, width
AX, AW = 8.8, 4.4        # ADDS x, width

LC = LX + LW/2           # Left column center x
RC = RX + RW/2           # Right column center x
AC = AX + AW/2           # ADDS column center x

# ═══════════════════════════════════════════════════════════════
# TITLE
# ═══════════════════════════════════════════════════════════════
rb(0.4, 16.3, FW-0.8, 0.65, P["v0"], "none", lw=0, r=0.1, z=2)
txt(FW/2, 16.65,
    "연구개발과제 추진체계 | Joint Research Framework: PrPc-Based Precision Therapy in KRAS-Mutant CRC",
    sz=14, c="white", bold=True)

# ═══════════════════════════════════════════════════════════════
# TIMELINE ROW
# ═══════════════════════════════════════════════════════════════
TY, TH = 14.8, 0.90
tblocks = [
    (LX,   6.0, P["b1"], "1–3차년도  |  Phase 1–3",
     "세포주 기반 기초 기전 규명  ·  In-Vitro Mechanistic Studies"),
    (LX+6.3, 4.5, P["v1"], "4차년도  |  Phase 4",
     "임상 검체 수집 + PDO 구축  ·  Clinical Biospecimen & Organoid"),
    (LX+11.1, 9.3, P["g1"], "5–6차년도  |  Phase 5–6",
     "ADDS 기반 최적화 + 전임상 검증  ·  ADDS Validation"),
]

for tx, tw, tc, tlabel, tdesc in tblocks:
    rb(tx, TY, tw, TH, tc, "none", lw=0, r=0.1)
    txt(tx + tw/2, TY + TH*0.68, tlabel, sz=10, c="white", bold=True)
    txt(tx + tw/2, TY + TH*0.28, tdesc, sz=8.5, c="white")
    # Chevron clip: notch on right side (if not last)
    if tx < LX + 11:
        tri = mpatches.Polygon([[tx+tw, TY], [tx+tw+0.25, TY+TH/2], [tx+tw, TY+TH]],
                               facecolor=P["bg"], edgecolor="none", zorder=3)
        ax.add_patch(tri)

# ═══════════════════════════════════════════════════════════════
# TEAM HEADER BOXES
# ═══════════════════════════════════════════════════════════════
THEAD_Y, THEAD_H = 13.1, 1.45

# LEFT: basic/preclinical team
rb(LX, THEAD_Y, LW, THEAD_H, P["b1"], P["b1"], lw=0, r=0.15, z=2)
txt(LC, THEAD_Y + THEAD_H - 0.45, "이상훈 교수 연구팀", sz=13, c="white", bold=True)
txt(LC, THEAD_Y + THEAD_H - 0.88, "PI: Prof. Sang-Hoon Lee", sz=10, c="#BFDBFE")
divider(LX+0.4, THEAD_Y + THEAD_H - 1.05, LW-0.8, c="#93C5FD")
txt(LC, THEAD_Y + 0.25,
    "인하대학교 의과대학 의생명학교실  ·  Molecular & Cell Physiology",
    sz=9, c="#DBEAFE")

# RIGHT: clinical team
rb(RX, THEAD_Y, RW, THEAD_H, P["g1"], P["g1"], lw=0, r=0.15, z=2)
txt(RC, THEAD_Y + THEAD_H - 0.45, "최문석 교수 연구팀", sz=13, c="white", bold=True)
txt(RC, THEAD_Y + THEAD_H - 0.88, "Co-PI: Prof. Moon-Seok Choi", sz=10, c="#A7F3D0")
divider(RX+0.4, THEAD_Y + THEAD_H - 1.05, RW-0.8, c="#6EE7B7")
txt(RC, THEAD_Y + 0.25,
    "인하대학교 의과대학 외과학교실  ·  Surgical Oncology Dept.",
    sz=9, c="#D1FAE5")

# ═══════════════════════════════════════════════════════════════
# MODULE CARDS — 3 per team, strictly gridded
# ═══════════════════════════════════════════════════════════════
MH      = 1.35    # Module height
M_GAP   = 0.22   # Gap between modules
M_TOP_Y = THEAD_Y - 0.25   # First module starts just below header

left_modules = [
    ("PrPc 발현 및 기능 분석",
     "In-Vitro PrPc Functional Phenotyping",
     "Colony formation, wound healing, sphere assay\n"
     "siRNA knockdown → stemness & EMT markers\n"
     "(CD44, LGR5, OCT4, Snail, ZEB1, N-cadherin)"),
    ("항암제 내성 기전 규명",
     "Therapeutic Resistance Mechanism",
     "5-FU / Oxaliplatin / Irinotecan repeated exposure\n"
     "IC50, AUC; Resistant cell-line induction\n"
     "PrPc-dependent pathway profiling (p-AKT, p-ERK)"),
    ("KRAS 하위 신호·저산소 기전",
     "KRAS Downstream & Hypoxia Pathway",
     "KRAS G12D/V/G13D subtype-specific expression analysis\n"
     "HIF-1α-mediated PrPc stabilization under hypoxia\n"
     "Invasion / Matrigel transwell quantification"),
]

right_modules = [
    ("임상 검체 확보 및 병리 판독",
     "Biospecimen Bank & Pathologic Scoring",
     "Endoscopic biopsy + surgical resection specimens\n"
     "Primary tumor & metastatic lesion collection\n"
     "Blinded IHC PrPc scoring (H-score, 0–300)"),
    ("KRAS 변이형 분석 및 환자군 분류",
     "KRAS Mutation Typing & Stratification",
     "ddPCR + targeted NGS for KRAS codon 12/13\n"
     "PrPc-High vs. PrPc-Low patient cohort split\n"
     "Correlation with stage, grade, lymphovascular invasion"),
    ("환자 유래 오가노이드(PDO) 구축",
     "Patient-Derived Organoid (PDO) Construction",
     "Fresh tumor → Matrigel 3D culture; passage up to P5\n"
     "WGS / RNA-seq congruency confirmation\n"
     "Biobank for ADDS-driven drug response evaluation"),
]

left_module_centers  = []
right_module_centers = []

my = M_TOP_Y
for i, (kor, eng, desc) in enumerate(left_modules):
    box_y = my - MH
    rb(LX, box_y, LW, MH, P["card"], P["b2"], lw=1.8, r=0.12, z=2)
    # Color strip left edge
    rb(LX, box_y, 0.18, MH, P["b1"], "none", lw=0, r=0.04, z=3)
    txt(LX + 0.35 + (LW-0.35)/2, box_y + MH - 0.32, kor,
        sz=10.5, bold=True, c=P["b0"], ha="center")
    txt(LX + 0.35 + (LW-0.35)/2, box_y + MH - 0.62, eng,
        sz=8.5, c=P["b1"])
    divider(LX+0.35, box_y + MH - 0.75, LW-0.5, c=P["b2"])
    txt(LX + 0.35 + (LW-0.35)/2, box_y + 0.42, desc,
        sz=8.5, c=P["txt2"], ls=1.55)
    left_module_centers.append(box_y + MH/2)
    my = box_y - M_GAP

my = M_TOP_Y
for i, (kor, eng, desc) in enumerate(right_modules):
    box_y = my - MH
    rb(RX, box_y, RW, MH, P["card"], P["g2"], lw=1.8, r=0.12, z=2)
    # Color strip right edge
    rb(RX + RW - 0.18, box_y, 0.18, MH, P["g1"], "none", lw=0, r=0.04, z=3)
    txt(RX + (RW-0.35)/2, box_y + MH - 0.32, kor,
        sz=10.5, bold=True, c=P["g0"])
    txt(RX + (RW-0.35)/2, box_y + MH - 0.62, eng,
        sz=8.5, c=P["g1"])
    divider(RX+0.2, box_y + MH - 0.75, RW-0.5, c=P["g2"])
    txt(RX + (RW-0.35)/2, box_y + 0.42, desc,
        sz=8.5, c=P["txt2"], ls=1.55)
    right_module_centers.append(box_y + MH/2)
    my = box_y - M_GAP

# ─── Bottom Y of the lowest module row ───────────────────────────────────────
lowest_module_y = my   # = bottom of last module

# ═══════════════════════════════════════════════════════════════
# ADDS CORE PANEL (vertical center, spans module row)
# ═══════════════════════════════════════════════════════════════
# Place ADDS so its top aligns ~with team headers
# and its bottom ends just above the output box

ADDS_TOP    = THEAD_Y + THEAD_H - 0.1   # flush with top of team headers
ADDS_BOTTOM = lowest_module_y - 0.10    # flush with bottom of last module
ADDS_H = ADDS_TOP - ADDS_BOTTOM

# Main outer panel (subtle)
rb(AX - 0.15, ADDS_BOTTOM, AW + 0.3, ADDS_H, P["v3"], P["v1"], lw=2, r=0.2, z=2)

# ADDS Title Ribbon
rb(AX - 0.15, ADDS_TOP - 0.75, AW + 0.3, 0.75, P["v0"], "none", lw=0, r=0.15, z=3)
txt(AC, ADDS_TOP - 0.38, "ADDS  플랫폼", sz=14, c="white", bold=True)

# Subtitle
txt(AC, ADDS_TOP - 0.62, "AI-Driven Drug Discovery & Synergy System", sz=9, c=P["v2"])

divider(AX+0.2, ADDS_TOP - 0.78, AW-0.2, c=P["v2"], lw=1.2)

# 3 inner ADDS process steps
adds_steps = [
    ("① 데이터 통합 | Data Harmonization",
     "In-Vitro assay + IHC + KRAS subtype\n→ unified multimodal feature vector"),
    ("② 에너지 지형 연산 | Energy Landscape ΔG",
     "Eyring & Boltzmann mapping of PrPc\nexpression → pathway activation barrier"),
    ("③ 병용 최적화 | Combination Optimization",
     "Synergy scoring, IC50 grid search\n→ ranked drug pair, dose & schedule"),
]

step_h    = 1.10
step_gap  = 0.20
step_pool_h = len(adds_steps) * step_h + (len(adds_steps)-1) * step_gap
step_start_y = ADDS_TOP - 0.85 - 0.10   # just below divider

sy = step_start_y
for j, (step_title, step_desc) in enumerate(adds_steps):
    by = sy - step_h
    rb(AX+0.05, by, AW-0.1, step_h, P["card"], P["v2"], lw=1.5, r=0.10, z=3)
    # Number badge
    rb(AX+0.12, by + step_h/2 - 0.22, 0.44, 0.44, P["v1"], "none", lw=0, r=0.08, z=4)
    txt(AX+0.34, by + step_h/2, str(j+1), sz=11, c="white", bold=True)
    # Text
    txt(AX + 0.65 + (AW-0.65)/2, by + step_h - 0.30, step_title,
        sz=10, bold=True, c=P["v0"], ha="center")
    txt(AX + 0.65 + (AW-0.65)/2, by + 0.38, step_desc,
        sz=8.5, c=P["txt2"])
    # Down arrow between steps
    if j < len(adds_steps) - 1:
        arr_v(AC, by-0.01, by - step_gap + 0.01, color=P["v1"], lw=2)
    sy = by - step_gap

# ═══════════════════════════════════════════════════════════════
# OUTPUT BOX — Preclinical Validation & Translation
# ═══════════════════════════════════════════════════════════════
OUT_H = 1.75
OUT_Y = 0.55
OUT_W = FW - 1.6

rb(0.8, OUT_Y, OUT_W, OUT_H, P["v0"], "none", lw=0, r=0.18, z=2)
txt(FW/2, OUT_Y + OUT_H - 0.38,
    "전임상 검증 및 임상 전환  |  Preclinical Validation & Translational Output",
    sz=12, c="white", bold=True)
divider(1.2, OUT_Y + OUT_H - 0.62, OUT_W - 0.8, c=P["v2"], lw=1)

outputs = [
    ("In-Vivo 이식 모델 검증\nXenograft tumor suppression\n& safety panel"),
    ("PDO 병용 약물 반응 평가\nOrganoid drug response\n& PrPc inhibition readout"),
    ("환자군 선별 기준 확립\nKRAS subtype × PrPc expression\nstratification criteria"),
    ("ADDS 기반 최적 조합 도출\nTop-ranked combination\ndose & schedule regimen"),
]

ox_start = 1.5
ow_each  = (OUT_W - 1.2) / len(outputs) - 0.2
for k, otext in enumerate(outputs):
    ox = ox_start + k * (ow_each + 0.2)
    rb(ox, OUT_Y + 0.10, ow_each, OUT_H - 0.75, P["v1"], "none", lw=0, r=0.1, z=3)
    txt(ox + ow_each/2, OUT_Y + 0.10 + (OUT_H-0.75)/2, otext,
        sz=8.5, c="white", ls=1.5)

# ═══════════════════════════════════════════════════════════════
# CONNECTORS — fully explicit, no crossings
# ═══════════════════════════════════════════════════════════════

# ─ Team headers → each module: short vertical line on left/right edge
for my_center in left_module_centers:
    arr_v(LX+0.09, THEAD_Y - 0.01, my_center + MH/2 + 0.02, color=P["b2"], lw=1.5)

for my_center in right_module_centers:
    arr_v(RX + RW - 0.09, THEAD_Y - 0.01, my_center + MH/2 + 0.02, color=P["g2"], lw=1.5)

# ─ Data flows INTO ADDS from left column:  right edge of each left module → ADDS left wall
conn_y_left  = [lmc for lmc in left_module_centers]   # center-y of each left module
conn_y_right = [rmc for rmc in right_module_centers]

# ADDS entry ports on the left side
adds_port_y_left  = [ADDS_TOP - 1.10, ADDS_TOP - 2.55, ADDS_TOP - 4.0]
adds_port_y_right = [ADDS_TOP - 1.10, ADDS_TOP - 2.55, ADDS_TOP - 4.0]

for i, (src_y, port_y) in enumerate(zip(conn_y_left, adds_port_y_left)):
    # Horizontal segment: left module right edge → bend x
    bend_x = AX - 0.10
    ax.plot([LX + LW, bend_x], [src_y, src_y], color=P["b1"], lw=2, zorder=3, alpha=0.7)
    # Vertical drop to port height
    ax.plot([bend_x, bend_x], [src_y, port_y], color=P["b1"], lw=2, zorder=3, alpha=0.7)
    # Arrow into ADDS
    arr_h(port_y, bend_x, AX - 0.01, color=P["b1"], lw=2)

# Right modules → ADDS right wall
for i, (src_y, port_y) in enumerate(zip(conn_y_right, adds_port_y_right)):
    bend_x = AX + AW + 0.10
    ax.plot([RX, bend_x], [src_y, src_y], color=P["g1"], lw=2, zorder=3, alpha=0.7)
    ax.plot([bend_x, bend_x], [src_y, port_y], color=P["g1"], lw=2, zorder=3, alpha=0.7)
    arr_h(port_y, bend_x, AX + AW + 0.01, color=P["g1"], lw=2)

# ─ ADDS bottom → Output box
adds_out_y_top = OUT_Y + OUT_H
arr_v(AC, ADDS_BOTTOM - 0.01, adds_out_y_top + 0.01, color=P["v1"], lw=4)

# ─ Feedback loop: right side of output → right team (Clinical Feedback)
feedback_x = FW - 0.55
ax.plot([0.8 + OUT_W, feedback_x], [OUT_Y + OUT_H/2, OUT_Y + OUT_H/2], color=P["g1"], lw=2.5, alpha=0.7, linestyle="--", zorder=3)
ax.plot([feedback_x, feedback_x], [OUT_Y + OUT_H/2, RC], color=P["g1"], lw=2.5, alpha=0.7, linestyle="--", zorder=3)
ax.annotate("", xy=(RC, THEAD_Y + 0.1), xytext=(RC, RC - THEAD_Y + 0.5),
            arrowprops=dict(arrowstyle="-|>,head_width=0.4,head_length=0.6",
                            color=P["g1"], lw=2, linestyle="dashed"),
            zorder=3)

txt(feedback_x - 0.35, OUT_Y + OUT_H + 0.18,
    "임상 피드백 루프\nClinical Feedback Loop",
    sz=8.5, c=P["g0"], bold=True, ha="right")

# ─ Small connector label annotations
txt(AX - 1.1, adds_port_y_left[0] + 0.18, "In-Vitro\nData →", sz=8, c=P["b1"], bold=True, ha="center")
txt(AX + AW + 1.1, adds_port_y_right[0] + 0.18, "← Clinical\nData",  sz=8, c=P["g1"], bold=True, ha="center")

# ═══════════════════════════════════════════════════════════════
# LEGEND
# ═══════════════════════════════════════════════════════════════
legend_items = [
    mpatches.Patch(facecolor=P["b1"], label="기초·전임상팀 (이상훈 교수)  ·  Basic/Preclinical"),
    mpatches.Patch(facecolor=P["g1"], label="임상·병리팀 (최문석 교수)  ·  Clinical/Pathology"),
    mpatches.Patch(facecolor=P["v1"], label="ADDS 통합 플랫폼  ·  ADDS Integration Core"),
    mpatches.Patch(facecolor=P["v0"], label="전임상 번역 및 출력  ·  Translational Output"),
]
ax.legend(handles=legend_items, loc="lower left", bbox_to_anchor=(0.01, 0.005),
          fontsize=9, framealpha=0.95, edgecolor=P["border"],
          ncol=4, handleheight=1.2, handlelength=1.5, columnspacing=1.2,
          fancybox=True)

# Footer
txt(FW/2, 0.3,
    "2026년도 미래도전연구지원사업 신규과제  |  인하대학교 의과대학  |  이상훈·최문석 공동연구팀",
    sz=8.5, c=P["txt3"])

# ─────────────────────────────────────────────────────────────
out = r"f:\ADDS\docs\figure_v3_final.png"
plt.savefig(out, dpi=300, bbox_inches="tight")
plt.close()
print(f"Saved → {out}")
