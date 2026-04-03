"""
figure_v5.py
───────────────────────────────────────────────────────
v5 Final:
- ADDS top aligned with team header top (NOT timeline)
- ADDS bottom aligned with lowest module bottom
- Steps distributed inside that vertical range
- Timeline chevrons stay above everything
- No overlaps, no gaps
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch

plt.rcParams.update({
    "font.family":        "Malgun Gothic",
    "axes.unicode_minus": False,
})

FW, FH = 22, 16
fig, ax = plt.subplots(figsize=(FW, FH), dpi=300)
ax.set_xlim(0, FW)
ax.set_ylim(0, FH)
ax.axis("off")

P = {
    "bg":   "#F8FAFC", "txt2": "#374151", "txt3": "#94A3B8", "border": "#CBD5E1",
    "b0": "#1E3A8A", "b1": "#2563EB", "b2": "#BFDBFE",
    "g0": "#064E3B", "g1": "#059669", "g2": "#A7F3D0",
    "v0": "#4C1D95", "v1": "#7C3AED", "v2": "#C4B5FD", "v3": "#F5F3FF",
    "card": "#FFFFFF",
}
fig.patch.set_facecolor(P["bg"])
ax.set_facecolor(P["bg"])

# ─── Helpers ─────────────────────────────────────────
def rb(x, y, w, h, fc, ec, lw=1.5, r=0.10, z=2, alpha=1.0):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad={r}",
                                facecolor=fc, edgecolor=ec, linewidth=lw,
                                alpha=alpha, zorder=z))

def T(x, y, s, sz=9, c="#0F172A", bold=False, ha="center", va="center", z=5, ls=1.5):
    ax.text(x, y, s, fontsize=sz, color=c, ha=ha, va=va,
            fontweight="bold" if bold else "normal", zorder=z, linespacing=ls)

def hl(y, x1, x2, c=P["border"], lw=1.0, ls="-", z=3):
    ax.plot([x1, x2], [y, y], color=c, lw=lw, zorder=z, linestyle=ls)

def vl(x, y1, y2, c=P["border"], lw=1.0, ls="-", z=3):
    ax.plot([x, x], [y1, y2], color=c, lw=lw, zorder=z, linestyle=ls)

def av(x, y0, y1, c, lw=2.5, z=4):
    ax.annotate("", xy=(x, y1), xytext=(x, y0),
                arrowprops=dict(arrowstyle="-|>,head_width=0.42,head_length=0.52",
                                color=c, lw=lw), zorder=z)

def ah(y, x0, x1, c, lw=2.5, z=4):
    ax.annotate("", xy=(x1, y), xytext=(x0, y),
                arrowprops=dict(arrowstyle="-|>,head_width=0.42,head_length=0.52",
                                color=c, lw=lw), zorder=z)

# ═══════════════════════════════════════════════════════
# GRID
# Columns:  Left 0.5-7.7 (7.2)  |  ADDS 8.0-13.8 (5.8)  |  Right 14.1-21.3 (7.2)
# ═══════════════════════════════════════════════════════
LX, LW = 0.5, 7.2
RX, RW = 14.1, 7.2
AX, AW = 8.0, 5.8
LC = LX+LW/2;  RC = RX+RW/2;  AC = AX+AW/2

# Row heights (all pinned)
TITLE_TOP = FH - 0.20
TITLE_H   = 0.58
TLINE_H   = 0.85
THEAD_H   = 1.22
M_H       = 1.38
M_GAP     = 0.22
N_MOD     = 3
OUT_H     = 1.55
FOOTER_H  = 0.25
GAP_OUT   = 0.18   # gap ADDS→output arrow zone

# Compute Y positions top-down
TITLE_Y = TITLE_TOP - TITLE_H         # 15.22
TLINE_Y = TITLE_Y - 0.10 - TLINE_H   # just below title
THEAD_Y = TLINE_Y - 0.08 - THEAD_H

# Modules start just below team headers
mod_tops = []   # (y_bottom, y_center) per module
cur = THEAD_Y - 0.08
for _ in range(N_MOD):
    by = cur - M_H
    mod_tops.append((by, by + M_H/2))
    cur = by - M_GAP

LOWEST_MOD_BOTTOM = mod_tops[-1][0]    # ← bottom y of 3rd module

OUT_Y  = FOOTER_H
OUT_TOP = OUT_Y + OUT_H

# ADDS panel: top = team header top, bottom = lowest module bottom
ADDS_TOP    = THEAD_Y + THEAD_H        # same as team header top
ADDS_BOTTOM = LOWEST_MOD_BOTTOM - 0.05
ADDS_H      = ADDS_TOP - ADDS_BOTTOM

# ═══════════════════════════════════════════════════════
# 1. BACKGROUND
# ═══════════════════════════════════════════════════════
fig.patch.set_facecolor(P["bg"])

# ═══════════════════════════════════════════════════════
# 2. TITLE
# ═══════════════════════════════════════════════════════
rb(0.4, TITLE_Y, FW-0.8, TITLE_H, P["v0"], "none", lw=0, r=0.09)
T(FW/2, TITLE_Y+TITLE_H*0.50,
  "연구개발과제 추진체계  |  Joint Research Framework: PrPc-Based Precision Therapy in KRAS-Mutant CRC",
  sz=12.5, c="white", bold=True)

# ═══════════════════════════════════════════════════════
# 3. TIMELINE CHEVRONS
# ═══════════════════════════════════════════════════════
CHEVRON_NOTCH = 0.26
phases = [
    (LX,       7.15, P["b1"],
     "1–3차년도  |  Phase 1–3",
     "세포주 기반 기초 기전 규명  ·  In-Vitro Mechanistic Studies"),
    (LX+7.38,  7.02, P["v1"],
     "4차년도  |  Phase 4",
     "임상 검체 + PDO 구축  ·  Biospecimen & Organoid"),
    (LX+14.73, FW-LX-14.73-0.4, P["g1"],
     "5–6차년도  |  Phase 5–6",
     "ADDS 최적화 + 전임상 검증  ·  Validation"),
]
for i, (px, pw, pc, plabel, pdesc) in enumerate(phases):
    rb(px, TLINE_Y, pw, TLINE_H, pc, "none", lw=0, r=0.07)
    T(px+pw/2, TLINE_Y+TLINE_H*0.70, plabel, sz=9.5, c="white", bold=True)
    T(px+pw/2, TLINE_Y+TLINE_H*0.28, pdesc, sz=8.5, c="white")
    if i < 2:  # right notch
        nt = mpatches.Polygon(
            [[px+pw, TLINE_Y], [px+pw+CHEVRON_NOTCH, TLINE_Y+TLINE_H/2], [px+pw, TLINE_Y+TLINE_H]],
            facecolor=P["bg"], edgecolor="none", zorder=3)
        ax.add_patch(nt)

# ═══════════════════════════════════════════════════════
# 4. TEAM HEADER BOXES
# ═══════════════════════════════════════════════════════
teams = [
    (LX, LW, P["b1"], "이상훈 교수 연구팀", "PI: Prof. Sang-Hoon Lee",
     "인하대학교 의과대학 의생명학교실  ·  Molecular & Cell Physiology"),
    (RX, RW, P["g1"], "최문석 교수 연구팀", "Co-PI: Prof. Moon-Seok Choi",
     "인하대학교 의과대학 외과학교실  ·  Surgical Oncology Dept."),
]
for tx, tw, tc, nm, pi, aff in teams:
    rb(tx, THEAD_Y, tw, THEAD_H, tc, "none", lw=0, r=0.12)
    T(tx+tw/2, THEAD_Y+THEAD_H-0.35, nm, sz=13, c="white", bold=True)
    T(tx+tw/2, THEAD_Y+THEAD_H-0.68, pi, sz=9.5, c="white")
    hl(THEAD_Y+0.50, tx+0.4, tx+tw-0.4, c=(1,1,1,0.35), lw=0.8)
    T(tx+tw/2, THEAD_Y+0.24, aff, sz=8.5, c="white")

# ═══════════════════════════════════════════════════════
# 5. MODULE CARDS
# ═══════════════════════════════════════════════════════
LEFT_MODS = [
    ("PrPc 발현 및 기능 분석 | In-Vitro Functional Phenotyping",
     "Colony formation·Wound healing·Sphere assay\n"
     "siRNA knockdown → stemness & EMT markers\n"
     "(CD44, LGR5, OCT4, Snail, ZEB1, N-cadherin)"),
    ("항암제 내성 기전 규명 | Drug Resistance Mechanism",
     "5-FU / Oxaliplatin / Irinotecan · IC50 & AUC\n"
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
     "PrPc-High vs. PrPc-Low patient cohort split\n"
     "Correlation: stage, grade, lymphovascular invasion"),
    ("환자 유래 오가노이드(PDO) 구축 | PDO Construction",
     "Fresh tumor → Matrigel 3D culture (passage ≤ P5)\n"
     "WGS / RNA-seq molecular congruency confirmation\n"
     "Biobank for ADDS-driven drug response evaluation"),
]

left_cy  = []
right_cy = []

for i, ((lt, ld), (rt, rd)) in enumerate(zip(LEFT_MODS, RIGHT_MODS)):
    by, bcy = mod_tops[i]

    # Left
    rb(LX, by, LW, M_H, P["card"], P["b2"], lw=1.8, r=0.10)
    rb(LX, by, 0.14, M_H, P["b1"], "none", lw=0, r=0.04, z=3)
    T(LX+0.22+(LW-0.22)/2, by+M_H-0.30, lt, sz=9.5, bold=True, c=P["b0"])
    hl(by+M_H-0.50, LX+0.18, LX+LW-0.12, c=P["b2"])
    T(LX+0.22+(LW-0.22)/2, by+0.46, ld, sz=8.5, c=P["txt2"], ls=1.52)
    left_cy.append(bcy)

    # Right
    rb(RX, by, RW, M_H, P["card"], P["g2"], lw=1.8, r=0.10)
    rb(RX+RW-0.14, by, 0.14, M_H, P["g1"], "none", lw=0, r=0.04, z=3)
    T(RX+(RW-0.22)/2, by+M_H-0.30, rt, sz=9.5, bold=True, c=P["g0"])
    hl(by+M_H-0.50, RX+0.12, RX+RW-0.18, c=P["g2"])
    T(RX+(RW-0.22)/2, by+0.46, rd, sz=8.5, c=P["txt2"], ls=1.52)
    right_cy.append(bcy)

# ═══════════════════════════════════════════════════════
# 6. ADDS PANEL  (THEAD_Y top ↕ LOWEST_MOD_BOTTOM bottom)
# ═══════════════════════════════════════════════════════
rb(AX, ADDS_BOTTOM, AW, ADDS_H, P["v3"], P["v1"], lw=2.0, r=0.14, z=2)

# Title ribbon
RIB_H = 0.70
rb(AX, ADDS_TOP - RIB_H, AW, RIB_H, P["v0"], "none", lw=0, r=0.12, z=3)
# Square bottom corners of ribbon so it blends into panel
ax.add_patch(mpatches.Rectangle((AX, ADDS_TOP-RIB_H), AW, RIB_H*0.45,
                                  facecolor=P["v0"], edgecolor="none", zorder=3))
T(AC, ADDS_TOP - RIB_H*0.50,
  "ADDS  플랫폼  |  AI-Driven Drug Discovery & Synergy System",
  sz=11.5, c="white", bold=True)

# 3 step boxes inside ADDS
STEPS = [
    ("① 데이터 통합  |  Data Harmonization",
     "In-Vitro assay + IHC + KRAS subtype\n→ unified multimodal feature vector"),
    ("② 에너지 지형 연산  |  Energy Landscape ΔG",
     "Eyring & Boltzmann mapping of PrPc expression\n→ pathway activation barrier ΔG"),
    ("③ 병용 최적화  |  Combination Optimization",
     "Synergy scoring & IC50 grid search\n→ ranked drug pair, dose & schedule"),
]
STEP_H   = 1.08
STEP_GAP = 0.20

inner_h   = ADDS_H - RIB_H - 0.30
total_s   = len(STEPS)*STEP_H + (len(STEPS)-1)*STEP_GAP
pad_top   = (inner_h - total_s) / 2

sy = ADDS_TOP - RIB_H - pad_top
step_ports = []   # mid-y of each step (for connectors)
for j, (stitle, sdesc) in enumerate(STEPS):
    bby = sy - STEP_H
    step_ports.append(bby + STEP_H/2)
    rb(AX+0.22, bby, AW-0.44, STEP_H, P["card"], P["v2"], lw=1.4, r=0.09, z=3)
    # Badge
    rb(AX+0.30, bby+STEP_H/2-0.23, 0.46, 0.46, P["v1"], "none", lw=0, r=0.07, z=4)
    T(AX+0.53, bby+STEP_H/2, str(j+1), sz=11, c="white", bold=True)
    T(AX+0.85+(AW-0.85)/2, bby+STEP_H-0.28, stitle, sz=10, bold=True, c=P["v0"])
    T(AX+0.85+(AW-0.85)/2, bby+0.37, sdesc, sz=8.5, c=P["txt2"])
    if j < len(STEPS)-1:
        av(AC, bby-0.01, bby-STEP_GAP+0.01, c=P["v1"], lw=2)
    sy = bby - STEP_GAP

# ═══════════════════════════════════════════════════════
# 7. OUTPUT BOX
# ═══════════════════════════════════════════════════════
rb(0.5, OUT_Y, FW-1.0, OUT_H, P["v0"], "none", lw=0, r=0.12, z=2)
T(FW/2, OUT_Y+OUT_H-0.32,
  "전임상 검증 및 임상 전환  |  Preclinical Validation & Translational Output",
  sz=12, c="white", bold=True)
hl(OUT_Y+OUT_H-0.54, 0.9, FW-0.9, c=P["v2"], lw=0.9)

outs = [
    "In-Vivo 이식 모델 검증\nXenograft tumor suppression\n& safety assessment",
    "PDO 병용 약물 반응 평가\nOrganoid drug response\n& PrPc inhibition readout",
    "환자군 선별 기준 확립\nKRAS subtype × PrPc expression\nstratification criteria",
    "ADDS 기반 최적 조합 도출\nTop-ranked combination\ndose & schedule regimen",
]
OX_S = 0.9
OW_E = (FW-1.8)/4 - 0.14
for k, otxt in enumerate(outs):
    ox = OX_S + k*(OW_E+0.14)
    rb(ox, OUT_Y+0.06, OW_E, OUT_H-0.62, P["v1"], "none", lw=0, r=0.09, z=3)
    T(ox+OW_E/2, OUT_Y+0.06+(OUT_H-0.62)/2, otxt, sz=8.5, c="white", ls=1.5)

# ═══════════════════════════════════════════════════════
# 8. CONNECTORS
# ═══════════════════════════════════════════════════════
# ADDS → Output arrow
av(AC, ADDS_BOTTOM - 0.01, OUT_Y+OUT_H+0.01, c=P["v1"], lw=4, z=4)

# Left / Right modules → ADDS  (3 per side, routed via bend columns)
BL = AX - 0.35    # bend x left
BR = AX + AW + 0.35

for i, (ly, ry, py) in enumerate(zip(left_cy, right_cy, step_ports)):
    # Left module right edge → BL → step port (left wall of ADDS)
    ax.plot([LX+LW, BL], [ly, ly], color=P["b1"], lw=2.1, zorder=4, alpha=0.8)
    ax.plot([BL, BL], [ly, py],    color=P["b1"], lw=2.1, zorder=4, alpha=0.8)
    ah(py, BL, AX-0.01, c=P["b1"], lw=2.1, z=4)

    # Right module left edge → BR → step port (right wall of ADDS)
    ax.plot([RX, BR],              [ry, ry], color=P["g1"], lw=2.1, zorder=4, alpha=0.8)
    ax.plot([BR, BR],              [ry, py], color=P["g1"], lw=2.1, zorder=4, alpha=0.8)
    ah(py, BR, AX+AW+0.01, c=P["g1"], lw=2.1, z=4)

# Flow labels
T(BL-0.18, step_ports[0]+0.22, "In-Vitro\nData →", sz=8, c=P["b1"], bold=True, ha="right")
T(BR+0.18, step_ports[0]+0.22, "← Clinical\nData",  sz=8, c=P["g1"], bold=True, ha="left")

# Feedback loop (right side, dashed)
FB_X = FW - 0.38
hl(OUT_Y+OUT_H-0.35, FW-0.5, FB_X, c=P["g1"], lw=2, ls="--")
vl(FB_X, OUT_Y+OUT_H-0.35, THEAD_Y+THEAD_H/2, c=P["g1"], lw=2, ls="--")
ah(THEAD_Y+THEAD_H/2, FB_X, RX+RW, c=P["g1"], lw=2)
T(FB_X-0.2, OUT_Y+OUT_H+0.13, "임상 피드백 루프 / Clinical Feedback Loop",
  sz=8, c=P["g0"], bold=True, ha="right")

# ═══════════════════════════════════════════════════════
# 9. LEGEND & FOOTER
# ═══════════════════════════════════════════════════════
leg = [
    mpatches.Patch(facecolor=P["b1"], label=" 기초·전임상팀 (이상훈 교수)  Basic/Preclinical"),
    mpatches.Patch(facecolor=P["g1"], label=" 임상·병리팀 (최문석 교수)  Clinical/Pathology"),
    mpatches.Patch(facecolor=P["v1"], label=" ADDS 통합 플랫폼  ADDS Core"),
    mpatches.Patch(facecolor=P["v0"], label=" 전임상 번역 출력  Translational Output"),
]
ax.legend(handles=leg, loc="lower left", bbox_to_anchor=(0.012, 0.003),
          fontsize=9, framealpha=0.96, edgecolor=P["border"], ncol=4,
          handleheight=1.0, handlelength=1.5, columnspacing=0.9, fancybox=True)

T(FW/2, FOOTER_H*0.40,
  "2026년도 미래도전연구지원사업 신규과제  |  인하대학교 의과대학  |  이상훈·최문석 공동연구팀",
  sz=8.5, c=P["txt3"])

# ─────────────────────────────────────────────────────────────────────────────
out = r"f:\ADDS\docs\figure_v5.png"
plt.savefig(out, dpi=300, bbox_inches="tight")
plt.close()
print(f"Saved → {out}")
