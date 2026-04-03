"""
figure_v4_clean.py
───────────────────────────────────────────────────────
v4: All coordinates locked. Column widths tightened.
Connectors use safe fixed waypoints. Feedback loop cleaned up.
ADDS panel height auto-computed from module layout.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import numpy as np

plt.rcParams.update({
    "font.family":        "Malgun Gothic",
    "axes.unicode_minus": False,
    "figure.facecolor":   "white",
    "savefig.facecolor":  "white",
})

# ── Canvas ────────────────────────────────────────────────────────────────────
FW, FH = 22, 16
fig, ax = plt.subplots(figsize=(FW, FH), dpi=300)
ax.set_xlim(0, FW)
ax.set_ylim(0, FH)
ax.axis("off")

P = {
    "bg":   "#F8FAFC", "border": "#CBD5E1",
    "txt":  "#0F172A", "txt2":   "#374151", "txt3": "#94A3B8",
    "b0": "#1E3A8A", "b1": "#2563EB", "b2": "#BFDBFE", "b3": "#EFF6FF",
    "g0": "#064E3B", "g1": "#059669", "g2": "#A7F3D0", "g3": "#ECFDF5",
    "v0": "#4C1D95", "v1": "#7C3AED", "v2": "#C4B5FD", "v3": "#F5F3FF",
    "card": "#FFFFFF",
}
fig.patch.set_facecolor(P["bg"])
ax.set_facecolor(P["bg"])

# ─────────────────────────────────────────────────────────────────────────────
# STRICT GRID
# ─────────────────────────────────────────────────────────────────────────────
# Column boundaries (x):
#  Left team      : 0.5  → 7.7    width=7.2
#  Gap            : 7.7  → 8.1
#  ADDS           : 8.1  → 13.9   width=5.8
#  Gap            : 13.9 → 14.3
#  Right team     : 14.3 → 21.5   width=7.2
LX, LW = 0.5, 7.2
RX, RW = 14.3, 7.2
AX, AW = 8.1, 5.8
LC = LX + LW/2
RC = RX + RW/2
AC = AX + AW/2

# Row boundaries (y, top to bottom):
TITLE_Y  = 15.35    # Single title band top
TLINE_Y  = 14.10    # Timeline top   (height=0.85)
TLINE_H  = 0.85
THEAD_Y  = 13.15    # Team header top (height=1.25)
THEAD_H  = 1.25

M_H   = 1.38        # Module height
M_GAP = 0.20        # Gap between modules
N_MOD = 3           # Number of modules per team

# Compute module bottoms  (stacked downward from THEAD_Y)
mod_y = []   # list of [y_bottom, y_center] for each module (top→bottom order)
cur = THEAD_Y - 0.10
for _ in range(N_MOD):
    by = cur - M_H
    mod_y.append((by, by + M_H/2))
    cur = by - M_GAP

LOWEST_MOD = mod_y[-1][0]   # bottom y of last (3rd) module

OUT_H = 1.60
OUT_Y = 0.45
FOOTER_Y = 0.20

# ADDS panel spans from just below timeline to just above output
ADDS_TOP    = TLINE_Y - 0.0
ADDS_BOTTOM = OUT_Y + OUT_H + 0.12
ADDS_H = ADDS_TOP - ADDS_BOTTOM

# ─────────────────────────────────────────────────────────────────────────────
# HELPER FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────
def rb(x, y, w, h, fc, ec, lw=1.5, r=0.12, z=2, alpha=1.0):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad={r}",
                                facecolor=fc, edgecolor=ec, linewidth=lw,
                                alpha=alpha, zorder=z))

def T(x, y, s, sz=9, c=None, bold=False, ha="center", va="center", z=5, ls=1.5):
    ax.text(x, y, s, fontsize=sz, color=c or P["txt"], ha=ha, va=va,
            fontweight="bold" if bold else "normal", zorder=z, linespacing=ls)

def hline(y, x1, x2, c=P["border"], lw=1.0, z=3):
    ax.plot([x1, x2], [y, y], color=c, lw=lw, zorder=z)

def av(x, y1, y2, c, lw=2.5, z=4):
    """Vertical arrow from y1→y2."""
    ax.annotate("", xy=(x, y2), xytext=(x, y1),
                arrowprops=dict(arrowstyle="-|>,head_width=0.42,head_length=0.52",
                                color=c, lw=lw), zorder=z)

def ah(y, x1, x2, c, lw=2.5, z=4):
    """Horizontal arrow from x1→x2 at height y."""
    ax.annotate("", xy=(x2, y), xytext=(x1, y),
                arrowprops=dict(arrowstyle="-|>,head_width=0.42,head_length=0.52",
                                color=c, lw=lw), zorder=z)

def elbow_right(x_src_right, y_src, x_dst_left, y_dst, c, lw=2.5, z=4):
    """Go right → bend down → arrive from left."""
    bx = x_dst_left - 0.12
    ax.plot([x_src_right, bx], [y_src, y_src], color=c, lw=lw, zorder=z)
    ax.plot([bx, bx], [y_src, y_dst], color=c, lw=lw, zorder=z)
    ah(y_dst, bx, x_dst_left, c=c, lw=lw, z=z)

def elbow_left(x_src_left, y_src, x_dst_right, y_dst, c, lw=2.5, z=4):
    """Go left → bend down → arrive from right."""
    bx = x_dst_right + 0.12
    ax.plot([x_src_left, bx], [y_src, y_src], color=c, lw=lw, zorder=z)
    ax.plot([bx, bx], [y_src, y_dst], color=c, lw=lw, zorder=z)
    ah(y_dst, bx, x_dst_right, c=c, lw=lw, z=z)

# ─────────────────────────────────────────────────────────────────────────────
# 1. TITLE BAND
# ─────────────────────────────────────────────────────────────────────────────
rb(0.4, TITLE_Y, FW-0.8, 0.58, P["v0"], "none", lw=0, r=0.10)
T(FW/2, TITLE_Y+0.30,
  "연구개발과제 추진체계  |  Joint Research Framework: PrPc-Based Precision Therapy in KRAS-Mutant CRC",
  sz=13, c="white", bold=True)

# ─────────────────────────────────────────────────────────────────────────────
# 2. TIMELINE CHEVRONS
# ─────────────────────────────────────────────────────────────────────────────
phases = [
    (LX,        7.0,  P["b1"],
     "1–3차년도  |  Phase 1–3",
     "세포주 기반 기초 기전 규명  ·  In-Vitro Mechanistic Studies"),
    (LX+7.3,    6.9,  P["v1"],
     "4차년도  |  Phase 4",
     "임상 검체 + PDO 구축  ·  Biospecimen & Organoid Construction"),
    (LX+14.5,   0.0,  P["g1"],
     "5–6차년도  |  Phase 5–6",
     "ADDS 최적화 + 전임상 검증  ·  ADDS-Driven Validation"),
]
for i, (px, pw_override, pc, plabel, pdesc) in enumerate(phases):
    pw = pw_override if pw_override else (FW-LX-0.5)
    if i < 2:
        pw = 7.1
    else:
        pw = FW - px - 0.4
    rb(px, TLINE_Y, pw, TLINE_H, pc, "none", lw=0, r=0.08)
    T(px+pw/2, TLINE_Y+TLINE_H*0.70, plabel, sz=10, c="white", bold=True)
    T(px+pw/2, TLINE_Y+TLINE_H*0.28, pdesc, sz=8.5, c="white")
    if i < 2:
        nt = mpatches.Polygon([[px+pw, TLINE_Y], [px+pw+0.22, TLINE_Y+TLINE_H/2],
                               [px+pw, TLINE_Y+TLINE_H]],
                              facecolor=P["bg"], edgecolor="none", zorder=3)
        ax.add_patch(nt)

# ─────────────────────────────────────────────────────────────────────────────
# 3. TEAM HEADER BOXES
# ─────────────────────────────────────────────────────────────────────────────
for (tx, tw, tc, name, pi, affil) in [
    (LX, LW, P["b1"], "이상훈 교수 연구팀",
     "PI: Prof. Sang-Hoon Lee",
     "인하대학교 의과대학 의생명학교실  ·  Molecular & Cell Physiology"),
    (RX, RW, P["g1"], "최문석 교수 연구팀",
     "Co-PI: Prof. Moon-Seok Choi",
     "인하대학교 의과대학 외과학교실  ·  Surgical Oncology Dept."),
]:
    rb(tx, THEAD_Y, tw, THEAD_H, tc, "none", lw=0, r=0.14)
    T(tx+tw/2, THEAD_Y+THEAD_H-0.38, name, sz=13, c="white", bold=True)
    T(tx+tw/2, THEAD_Y+THEAD_H-0.72, pi, sz=9.5, c="white")
    hline(THEAD_Y+0.52, tx+0.4, tx+tw-0.4, c=(1.0, 1.0, 1.0, 0.35), lw=1.0)
    T(tx+tw/2, THEAD_Y+0.25, affil, sz=8.5, c="white")

# ─────────────────────────────────────────────────────────────────────────────
# 4. MODULE CARDS
# ─────────────────────────────────────────────────────────────────────────────
LEFT_MODS = [
    ("PrPc 발현 및 기능 분석 | In-Vitro Functional Phenotyping",
     "Colony formation·Wound healing·Sphere assay\n"
     "siRNA knockdown → stemness & EMT markers\n"
     "(CD44, LGR5, OCT4, Snail, ZEB1, N-cadherin)"),
    ("항암제 내성 기전 규명 | Drug Resistance Mechanism",
     "5-FU / Oxaliplatin / Irinotecan · IC50 & AUC analysis\n"
     "Repeated exposure → Resistant cell-line induction\n"
     "PrPc-dependent profiling: p-AKT, p-ERK, Bcl-2"),
    ("KRAS 하위 신호·저산소 기전 | KRAS & Hypoxia Pathway",
     "KRAS G12D/V/G13D subtype-specific expression\n"
     "HIF-1α-mediated PrPc stabilization under hypoxia\n"
     "Matrigel invasion · transwell quantification"),
]

RIGHT_MODS = [
    ("임상 검체 확보 및 병리 판독 | Biospecimen Bank",
     "Endoscopic biopsy + surgical resection specimens\n"
     "Primary tumor & metastatic lesion collection\n"
     "Blinded IHC PrPc scoring (H-score 0–300)"),
    ("KRAS 변이형 분석 및 환자군 분류 | Mutation Typing",
     "ddPCR + targeted NGS · KRAS codon 12/13\n"
     "PrPc-High vs. PrPc-Low patient cohort stratification\n"
     "Correlation: stage, grade, lymphovascular invasion"),
    ("환자 유래 오가노이드(PDO) 구축 | PDO Construction",
     "Fresh tumor → Matrigel 3D culture (passage ≤ P5)\n"
     "WGS / RNA-seq molecular congruency confirmation\n"
     "Biobank for ADDS-driven drug response evaluation"),
]

left_cy  = []   # mid-y of each left module (for connectors)
right_cy = []

for i, ((ltitle, ldesc), (rtitle, rdesc)) in enumerate(zip(LEFT_MODS, RIGHT_MODS)):
    by, bcy = mod_y[i]

    # ── Left module
    rb(LX, by, LW, M_H, P["card"], P["b2"], lw=1.8, r=0.12)
    rb(LX, by, 0.15, M_H, P["b1"], "none", lw=0, r=0.05, z=3)   # color stripe
    T(LX+0.25+(LW-0.25)/2, by+M_H-0.32, ltitle, sz=9.5, bold=True, c=P["b0"])
    hline(by+M_H-0.52, LX+0.20, LX+LW-0.15, c=P["b2"])
    T(LX+0.25+(LW-0.25)/2, by+0.46, ldesc, sz=8.5, c=P["txt2"], ls=1.52)
    left_cy.append(bcy)

    # ── Right module
    rb(RX, by, RW, M_H, P["card"], P["g2"], lw=1.8, r=0.12)
    rb(RX+RW-0.15, by, 0.15, M_H, P["g1"], "none", lw=0, r=0.05, z=3)  # color stripe
    T(RX+(RW-0.25)/2, by+M_H-0.32, rtitle, sz=9.5, bold=True, c=P["g0"])
    hline(by+M_H-0.52, RX+0.15, RX+RW-0.20, c=P["g2"])
    T(RX+(RW-0.25)/2, by+0.46, rdesc, sz=8.5, c=P["txt2"], ls=1.52)
    right_cy.append(bcy)

# ─────────────────────────────────────────────────────────────────────────────
# 5. ADDS CORE PANEL
# ─────────────────────────────────────────────────────────────────────────────
rb(AX, ADDS_BOTTOM, AW, ADDS_H, P["v3"], P["v1"], lw=2.0, r=0.18, z=2)

# Title ribbon
RIB_H = 0.72
rb(AX, ADDS_TOP - RIB_H, AW, RIB_H, P["v0"], "none", lw=0, r=0.15, z=3)
# Squared bottom of ribbon
ax.add_patch(mpatches.Rectangle((AX, ADDS_TOP-RIB_H), AW, RIB_H*0.4,
                                  facecolor=P["v0"], edgecolor="none", zorder=3))
T(AC, ADDS_TOP - RIB_H/2, "ADDS  플랫폼  |  AI-Driven Drug Discovery & Synergy System",
  sz=12, c="white", bold=True)

# Internal step boxes
STEPS = [
    ("① 데이터 통합  |  Data Harmonization",
     "In-Vitro assay + IHC + KRAS subtype\n→ unified multimodal feature vector"),
    ("② 에너지 지형 연산  |  Energy Landscape ΔG",
     "Eyring & Boltzmann mapping of PrPc\nexpression → pathway activation barrier"),
    ("③ 병용 최적화  |  Combination Optimization",
     "Synergy scoring & IC50 grid search\n→ ranked drug pair, dose & schedule"),
]

STEP_H   = 1.08
STEP_GAP = 0.22

# Position 3 step boxes evenly in available height
available = ADDS_H - RIB_H - 0.25
total_steps_h = len(STEPS)*STEP_H + (len(STEPS)-1)*STEP_GAP
padding_top = (available - total_steps_h) / 2

sy = ADDS_TOP - RIB_H - padding_top
for j, (stitle, sdesc) in enumerate(STEPS):
    box_y = sy - STEP_H
    rb(AX+0.25, box_y, AW-0.5, STEP_H, P["card"], P["v2"], lw=1.5, r=0.10, z=3)
    # Number badge
    rb(AX+0.32, box_y+STEP_H/2-0.24, 0.48, 0.48, P["v1"], "none", lw=0, r=0.08, z=4)
    T(AX+0.56, box_y+STEP_H/2, str(j+1), sz=12, c="white", bold=True)
    # Text
    T(AX+0.90+(AW-0.90)/2, box_y+STEP_H-0.30, stitle, sz=10, bold=True, c=P["v0"])
    T(AX+0.90+(AW-0.90)/2, box_y+0.38, sdesc, sz=8.5, c=P["txt2"])
    # Downward arrow between steps
    if j < len(STEPS)-1:
        av(AC, box_y-0.02, box_y - STEP_GAP + 0.02, c=P["v1"], lw=2)
    sy = box_y - STEP_GAP

adds_bottom_of_steps = sy + STEP_GAP  # actual bottom of last step box

# ─────────────────────────────────────────────────────────────────────────────
# 6. ADDS → OUTPUT ARROW
# ─────────────────────────────────────────────────────────────────────────────
av(AC, ADDS_BOTTOM, OUT_Y + OUT_H + 0.01, c=P["v1"], lw=4, z=4)

# ─────────────────────────────────────────────────────────────────────────────
# 7. OUTPUT BOX
# ─────────────────────────────────────────────────────────────────────────────
OX = 0.5
OW = FW - 1.0
rb(OX, OUT_Y, OW, OUT_H, P["v0"], "none", lw=0, r=0.14, z=2)
T(FW/2, OUT_Y+OUT_H-0.35,
  "전임상 검증 및 임상 전환  |  Preclinical Validation & Translational Output",
  sz=12, c="white", bold=True)
hline(OUT_Y+OUT_H-0.56, OX+0.4, OX+OW-0.4, c=P["v2"], lw=1)

outs = [
    "In-Vivo 이식 모델 검증\nXenograft tumor suppression\n& safety assessment",
    "PDO 병용 약물 반응 평가\nOrganoid drug response\n& PrPc inhibition readout",
    "환자군 선별 기준 확립\nKRAS subtype × PrPc expression\nstratification criteria",
    "ADDS 기반 최적 조합 도출\nTop-ranked combination\ndose & schedule regimen",
]
n_out = len(outs)
ow_each = (OW - 0.8) / n_out - 0.15
ox_s = OX + 0.4
for k, otxt in enumerate(outs):
    ox = ox_s + k * (ow_each + 0.15)
    rb(ox, OUT_Y+0.08, ow_each, OUT_H-0.66, P["v1"], "none", lw=0, r=0.10, z=3)
    T(ox+ow_each/2, OUT_Y+0.08+(OUT_H-0.66)/2, otxt, sz=8.5, c="white", ls=1.5)

# ─────────────────────────────────────────────────────────────────────────────
# 8. CONNECTORS: Left/Right modules → ADDS
# ─────────────────────────────────────────────────────────────────────────────
# 3 equispaced entry ports on each side of ADDS
# Vertical range for ports: just inside the ADDS steps area
port_range_top    = ADDS_TOP - RIB_H - padding_top - STEP_H/2
port_range_bottom = ADDS_TOP - RIB_H - padding_top - (2.5*STEP_H + 2.0*STEP_GAP)
port_ys = [
    port_range_top,
    (port_range_top + port_range_bottom) / 2,
    port_range_bottom,
]

# Use a separate bend x col so stacked lines don't overlap
BEND_L = AX - 0.40   # bend column for left team
BEND_R = AX + AW + 0.40

for i, (ly, ry, py) in enumerate(zip(left_cy, right_cy, port_ys)):
    # Left module right edge → bend x → port left
    col_alpha = 1.0 - i * 0.0
    ax.plot([LX+LW, BEND_L], [ly, ly], color=P["b1"], lw=2.2, zorder=4, alpha=0.75)
    ax.plot([BEND_L, BEND_L], [ly, py], color=P["b1"], lw=2.2, zorder=4, alpha=0.75)
    ah(py, BEND_L, AX-0.01, c=P["b1"], lw=2.2, z=4)

    # Right module left edge → bend x → port right
    ax.plot([RX, BEND_R], [ry, ry], color=P["g1"], lw=2.2, zorder=4, alpha=0.75)
    ax.plot([BEND_R, BEND_R], [ry, py], color=P["g1"], lw=2.2, zorder=4, alpha=0.75)
    ah(py, BEND_R, AX+AW+0.01, c=P["g1"], lw=2.2, z=4)

# ─────────────────────────────────────────────────────────────────────────────
# 9. DATA FLOW LABELS
# ─────────────────────────────────────────────────────────────────────────────
T(BEND_L - 0.2, port_ys[0] + 0.22, "In-Vitro\nData", sz=8, c=P["b1"], bold=True, ha="right")
T(BEND_R + 0.2, port_ys[0] + 0.22, "Clinical\nData", sz=8, c=P["g1"], bold=True, ha="left")

# ─────────────────────────────────────────────────────────────────────────────
# 10. FEEDBACK LOOP (right side, dashed)
# ─────────────────────────────────────────────────────────────────────────────
fb_x = FW - 0.42
ax.plot([OX+OW, fb_x], [OUT_Y+OUT_H-0.40, OUT_Y+OUT_H-0.40], color=P["g1"], lw=2, ls="--", zorder=3)
ax.plot([fb_x, fb_x], [OUT_Y+OUT_H-0.40, THEAD_Y+THEAD_H/2], color=P["g1"], lw=2, ls="--",  zorder=3)
ax.annotate("", xy=(RX+RW, THEAD_Y+THEAD_H/2), xytext=(fb_x, THEAD_Y+THEAD_H/2),
            arrowprops=dict(arrowstyle="-|>,head_width=0.4,head_length=0.55",
                            color=P["g1"], lw=2, linestyle="dashed"), zorder=4)
T(fb_x-0.2, OUT_Y+OUT_H+0.15, "임상 피드백 루프\nClinical Feedback Loop",
  sz=8.5, c=P["g0"], bold=True, ha="right")

# ─────────────────────────────────────────────────────────────────────────────
# 11. LEGEND & FOOTER
# ─────────────────────────────────────────────────────────────────────────────
legend_items = [
    mpatches.Patch(facecolor=P["b1"], label=" 기초·전임상팀 (이상훈 교수)  Basic/Preclinical Team"),
    mpatches.Patch(facecolor=P["g1"], label=" 임상·병리팀 (최문석 교수)  Clinical/Pathology Team"),
    mpatches.Patch(facecolor=P["v1"], label=" ADDS 통합 플랫폼  ADDS Integration Core"),
    mpatches.Patch(facecolor=P["v0"], label=" 전임상 번역 출력  Translational Output"),
]
ax.legend(handles=legend_items, loc="lower left", bbox_to_anchor=(0.015, 0.003),
          fontsize=9, framealpha=0.96, edgecolor=P["border"], ncol=4,
          handleheight=1.0, handlelength=1.6, columnspacing=1.0, fancybox=True)

T(FW/2, FOOTER_Y,
  "2026년도 미래도전연구지원사업 신규과제  |  인하대학교 의과대학  |  이상훈·최문석 공동연구팀",
  sz=8.5, c=P["txt3"])

# ─────────────────────────────────────────────────────────────────────────────
out = r"f:\ADDS\docs\figure_v4_clean.png"
plt.savefig(out, dpi=300, bbox_inches="tight")
plt.close()
print(f"Saved → {out}")
