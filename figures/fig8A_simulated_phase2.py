"""
Figure 8A – Simulated Phase II Design & Results
================================================
ADDS Pritamab Phase II 시뮬레이션 전용 패널
모집단: KRAS-mutant mCRC, N=280 일관 유지
지표: mPFS (months) 단일 사용 — HR / mPFS 혼재 없음
KRAS-WT 완전 제외

Panels:
  A. Phase II Trial Schema & Assumptions
  B. Simulated mPFS by arm (biomarker-stratified)
  C. Biomarker-enrichment effect on mPFS
  D. Waterfall: individual PFS gain projection

Output: f:\\ADDS\\figures\\fig8A_simulated_phase2.png
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch, FancyArrow
import numpy as np
import warnings
warnings.filterwarnings("ignore")

# ── Consistent style ─────────────────────────────────────────────────
BG      = "white"
PANEL   = "#F8FAFF"
C_TREAT = "#1A6FCA"   # Pritamab + FOLFOX — blue
C_CTRL  = "#D0312D"   # FOLFOX control    — red
C_HIGH  = "#1E8A4A"   # PrPc-high enriched — green
C_LOW   = "#F0A500"   # PrPc-low          — amber
GRAY    = "#6B7280"
DGRAY   = "#374151"
LTGRAY  = "#D1D5DB"

np.random.seed(2026)

# ── Simulation assumptions (all from NatureComm ★ / ADDS estimate ◆) ─
# Consistent N=280 throughout (2:1 randomization)
N_TOTAL  = 280
N_TREAT  = 187   # rounded 2:1
N_CTRL   = 93

# Arms — mPFS (months)
# Source: NatureComm §Clinical target ★, ADDS DL estimate ◆
ARMS = {
    "FOLFOX (Control)\nn=93":                   {"mPFS": 5.5,  "color": C_CTRL,  "source": "★"},
    "Pritamab + FOLFOX\n(All, n=187)":          {"mPFS": 8.25, "color": C_TREAT, "source": "★◆"},
    "Pritamab + FOLFOX\n(PrPc-high, n=131)":    {"mPFS": 9.8,  "color": C_HIGH,  "source": "◆"},
    "Pritamab + FOLFOX\n(PrPc-low, n=56)":      {"mPFS": 6.7,  "color": C_LOW,   "source": "◆"},
}

# Biomarker subgroups (KRAS-mutant only, consistently)
BM_GROUPS = [
    ("G12D (n=56)",  9.2, C_TREAT, "◆"),
    ("G12V (n=45)",  9.7, C_TREAT, "◆"),
    ("G12C (n=29)",  9.5, C_TREAT, "◆"),
    ("G13D (n=23)",  7.9, C_TREAT, "◆"),
    ("Other (n=34)", 7.5, C_TREAT, "◆"),
]

# ─────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(20, 13), facecolor=BG)
fig.patch.set_facecolor(BG)

# Header
fig.text(0.5, 0.985,
         "Figure 8A  |  Pritamab Phase II – Simulation Design & Projected Outcomes"
         "  [Target mPFS from NatureComm -- NOT same as Fig8B DL Cohort estimate]",
         ha="center", va="top", fontsize=15, fontweight="bold", color=DGRAY)
fig.text(0.5, 0.965,
         "KRAS-mutant mCRC (all subtypes) · N=280 · [★] NatureComm target  [◆] ADDS in silico estimate  "
         "— Data source labels shown on each panel",
         ha="center", va="top", fontsize=9.5, color=GRAY,
         bbox=dict(boxstyle="round,pad=0.35", facecolor="#FFF9E6",
                   edgecolor="#F0A500", lw=1.2))

fig.text(0.5, 0.945,
         "⚠  ALL RESULTS ARE SIMULATED — NOT EXPERIMENTAL DATA  ⚠",
         ha="center", va="top", fontsize=10.5, fontweight="bold", color="#C0392B",
         bbox=dict(boxstyle="round,pad=0.3", facecolor="#FEF2F2", edgecolor="#C0392B", lw=1.5))

gs = gridspec.GridSpec(2, 2, figure=fig, top=0.93, bottom=0.06,
                       left=0.05, right=0.97, hspace=0.40, wspace=0.32)

# ── Panel A: Trial Schema ──────────────────────────────────────────
ax_a = fig.add_subplot(gs[0, 0])
ax_a.set_facecolor(PANEL)
ax_a.set_xlim(0, 10); ax_a.set_ylim(0, 10); ax_a.axis("off")
ax_a.set_title("A.  Phase II Trial Schema & Key Assumptions",
               fontsize=12, fontweight="bold", color=DGRAY, pad=10)

# Eligibility box
ax_a.add_patch(FancyBboxPatch((0.3, 7.8), 9.4, 1.8,
               boxstyle="round,pad=0.15", facecolor="#EFF6FF", edgecolor="#1A6FCA", lw=1.5))
ax_a.text(5.0, 9.3, "Eligibility Criteria  ★◆", ha="center", fontsize=9.5,
          fontweight="bold", color="#1A6FCA")
ax_a.text(5.0, 8.75,
          "mCRC · 2nd-line · KRAS-mut (any subtype: G12D/V/C, G13D, other)\n"
          "PrPc IHC H-score ≥ 50 (85.7% of KRAS-mut patients ★) · ECOG 0-1",
          ha="center", fontsize=8, color=DGRAY, linespacing=1.5)

# Randomization
ax_a.text(5.0, 7.3, "2 : 1 Randomization  N = 280", ha="center",
          fontsize=9, fontweight="bold", color=DGRAY)
ax_a.annotate("", xy=(3.0, 7.0), xytext=(5.0, 7.1),
              arrowprops=dict(arrowstyle="-|>", color=C_TREAT, lw=1.8))
ax_a.annotate("", xy=(7.0, 7.0), xytext=(5.0, 7.1),
              arrowprops=dict(arrowstyle="-|>", color=C_CTRL, lw=1.8))

# Arm A
ax_a.add_patch(FancyBboxPatch((0.3, 5.5), 4.5, 1.35,
               boxstyle="round,pad=0.15", facecolor="#EFF6FF", edgecolor=C_TREAT, lw=1.5))
ax_a.text(2.55, 6.6, "Arm A  (n = 187)", ha="center", fontsize=9.5,
          fontweight="bold", color=C_TREAT)
ax_a.text(2.55, 6.1, "Pritamab 10 mg/kg Q3W\n+ FOLFOX (mFOLFOX6)",
          ha="center", fontsize=8, color=DGRAY, linespacing=1.5)

# Arm B
ax_a.add_patch(FancyBboxPatch((5.2, 5.5), 4.5, 1.35,
               boxstyle="round,pad=0.15", facecolor="#FFF1F1", edgecolor=C_CTRL, lw=1.5))
ax_a.text(7.45, 6.6, "Arm B  (n = 93)", ha="center", fontsize=9.5,
          fontweight="bold", color=C_CTRL)
ax_a.text(7.45, 6.1, "Placebo Q3W\n+ FOLFOX (mFOLFOX6)",
          ha="center", fontsize=8, color=DGRAY, linespacing=1.5)

# Primary Endpoint
ax_a.add_patch(FancyBboxPatch((0.3, 3.9), 9.4, 1.3,
               boxstyle="round,pad=0.15", facecolor="#F0FDF4", edgecolor=C_HIGH, lw=1.5))
ax_a.text(5.0, 4.95, "Primary Endpoint: mPFS  (Target mPFS [★] -- see Fig8A caption)",
          ha="center", fontsize=9.5,
          fontweight="bold", color=C_HIGH)
ax_a.text(5.0, 4.45,
          "Target mPFS: 5.5 -> 8.25 months (HR = 0.667)  [★ NatureComm Phase II target]"
          "  |  DL cohort estimate is DIFFERENT (see Fig8B)",
          ha="center", fontsize=7.8, color=DGRAY)

# Key Assumptions
ax_a.add_patch(FancyBboxPatch((0.3, 2.3), 9.4, 1.35,
               boxstyle="round,pad=0.15", facecolor="#FEFCE8", edgecolor=C_LOW, lw=1.5))
ax_a.text(5.0, 3.40, "Key Assumptions  ★◆", ha="center", fontsize=9,
          fontweight="bold", color=C_LOW)
ax_a.text(5.0, 3.00,
          "α = 0.10 (one-sided) · Power = 80% · PrPc-enriched superiority analysis  ★",
          ha="center", fontsize=7.5, color=DGRAY)
ax_a.text(5.0, 2.65,
          "Biomarker: PrPc IHC H-score ≥ 50  ·  KRAS subtype as stratification factor  ★",
          ha="center", fontsize=7.5, color=DGRAY)

# Disclaimer
ax_a.text(5.0, 1.7,
          "ALL NUMBERS ARE SIMULATED PROJECTIONS — NOT FROM AN ACTUAL TRIAL",
          ha="center", fontsize=8, color="#C0392B", fontweight="bold",
          style="italic")

# Source legend
ax_a.text(5.0, 1.2,
          "[★] NatureComm 2026 §Clinical target  |  [◆] ADDS in silico DL estimate",
          ha="center", fontsize=7.5, color=GRAY)
for sp in ax_a.spines.values(): sp.set_visible(False)

# ── Panel B: Simulated mPFS by Arm ────────────────────────────────
ax_b = fig.add_subplot(gs[0, 1])
ax_b.set_facecolor(PANEL)
ax_b.set_title("B.  Simulated mPFS by Treatment Arm  [★ target / ◆ estimate]",
               fontsize=12, fontweight="bold", color=DGRAY, pad=10)

labels  = list(ARMS.keys())
mPFS_v  = [v["mPFS"] for v in ARMS.values()]
colors  = [v["color"] for v in ARMS.values()]
sources = [v["source"] for v in ARMS.values()]
y_pos   = np.arange(len(labels))

bars = ax_b.barh(y_pos, mPFS_v, color=colors, alpha=0.82,
                  edgecolor="white", lw=0.8, height=0.6)

# Reference line: control arm
ax_b.axvline(5.5, color=C_CTRL, lw=1.5, ls="--", alpha=0.6, label="Control mPFS = 5.5 mo")

for bar, val, src in zip(bars, mPFS_v, sources):
    ax_b.text(val + 0.15, bar.get_y() + bar.get_height()/2,
              f"{val:.1f} mo  [{src}]",
              va="center", fontsize=9.5, fontweight="bold", color=DGRAY)

ax_b.set_yticks(y_pos)
ax_b.set_yticklabels(labels, fontsize=9, color=DGRAY)
ax_b.set_xlabel("Median PFS (months)", fontsize=11, color=DGRAY)
ax_b.set_xlim(0, 14)
ax_b.set_xticks(range(0, 15, 2))
ax_b.tick_params(colors=GRAY, labelsize=9)
ax_b.legend(fontsize=8, loc="lower right")
ax_b.spines[["top", "right"]].set_visible(False)
ax_b.spines[["left", "bottom"]].set_color(LTGRAY)
ax_b.grid(axis="x", color=LTGRAY, lw=0.7, alpha=0.7)

ax_b.text(0.01, -0.12,
          "⚠ Simulated results — not from actual trial. "
          "All projections require prospective clinical validation.",
          transform=ax_b.transAxes, fontsize=7.5, color="#C0392B",
          style="italic")

# ── Panel C: Biomarker Enrichment (KRAS-mutant only) ─────────────
ax_c = fig.add_subplot(gs[1, 0])
ax_c.set_facecolor(PANEL)
ax_c.set_title("C.  Biomarker Enrichment: mPFS by KRAS Subtype  [◆ ADDS estimate]\n"
               "Population: KRAS-mutant mCRC only (N=187, Arm A)",
               fontsize=11, fontweight="bold", color=DGRAY, pad=8)

bm_labels = [b[0] for b in BM_GROUPS]
bm_vals   = [b[1] for b in BM_GROUPS]
bm_colors = [b[2] for b in BM_GROUPS]
bm_src    = [b[3] for b in BM_GROUPS]
x_pos     = np.arange(len(bm_labels))

# Show control mPFS as reference
ax_c.axhline(5.5, color=C_CTRL, lw=2, ls="--", alpha=0.7, label="Control mPFS = 5.5 mo  ★")
ax_c.axhline(8.25, color=C_TREAT, lw=1.5, ls=":", alpha=0.7, label="Overall Prit+FOL = 8.25 mo  ★")

b_c = ax_c.bar(x_pos, bm_vals, color=bm_colors, alpha=0.80,
                edgecolor="white", lw=0.8, width=0.6)

for bar, val, src in zip(b_c, bm_vals, bm_src):
    ax_c.text(bar.get_x() + bar.get_width()/2, val + 0.18,
              f"{val:.1f} mo\n[{src}]",
              ha="center", va="bottom", fontsize=9, fontweight="bold", color=DGRAY)
    # Delta vs control
    delta = val - 5.5
    ax_c.text(bar.get_x() + bar.get_width()/2, 2.3,
              f"Δ+{delta:.1f}m",
              ha="center", va="bottom", fontsize=8, color=C_HIGH, fontweight="bold")

ax_c.set_xticks(x_pos)
ax_c.set_xticklabels(bm_labels, fontsize=9, color=DGRAY)
ax_c.set_ylabel("Median PFS (months)", fontsize=11, color=DGRAY)
ax_c.set_ylim(0, 13)
ax_c.tick_params(colors=GRAY, labelsize=9)
ax_c.legend(fontsize=8.5, loc="upper right")
ax_c.spines[["top", "right"]].set_visible(False)
ax_c.spines[["left", "bottom"]].set_color(LTGRAY)
ax_c.grid(axis="y", color=LTGRAY, lw=0.7, alpha=0.7)

ax_c.text(0.5, -0.15,
          "KRAS-WT patients excluded from this analysis — separate cohort required.",
          transform=ax_c.transAxes, ha="center", fontsize=8,
          color="#1A6FCA", style="italic", fontweight="bold")

ax_c.text(0.01, -0.20,
          "⚠ ADDS in silico estimate [◆] — awaiting prospective biomarker-stratified validation.",
          transform=ax_c.transAxes, fontsize=7.5, color="#C0392B", style="italic")

# ── Panel D: Individual PFS Gain Projection (waterfall) ──────────
ax_d = fig.add_subplot(gs[1, 1])
ax_d.set_facecolor(PANEL)
ax_d.set_title("D.  Individual PFS Gain Projection (Waterfall, Arm A, n=187)  [◆]",
               fontsize=11, fontweight="bold", color=DGRAY, pad=8)

# Simulate individual PFS gains (Weibull, KRAS-mut distribution)
rng = np.random.default_rng(2026)
pfs_treat  = rng.weibull(1.4, N_TREAT) * (8.25 / (np.log(2)**(1/1.4))) * 0.95
pfs_ctrl_m = rng.weibull(1.4, N_TREAT) * (5.5  / (np.log(2)**(1/1.4))) * 0.95
pfs_delta  = pfs_treat - pfs_ctrl_m
pfs_delta  = np.sort(pfs_delta)[::-1]  # descending

colors_w = [C_HIGH if d >= 0 else C_CTRL for d in pfs_delta]
x_w      = np.arange(len(pfs_delta))

ax_d.bar(x_w, pfs_delta, color=colors_w, alpha=0.75, width=1.0, edgecolor="none")
ax_d.axhline(0, color=DGRAY, lw=1.2)
ax_d.axhline(np.mean(pfs_delta), color=C_TREAT, lw=2, ls="--",
             label=f"Mean ΔPFS = +{np.mean(pfs_delta):.2f} mo")

# Annotations
pct_pos = (pfs_delta >= 0).mean() * 100
ax_d.text(0.02, 0.92, f"PFS gain ≥ 0: {pct_pos:.0f}% of patients  [◆]",
          transform=ax_d.transAxes, fontsize=9, color=C_HIGH, fontweight="bold")
ax_d.text(0.02, 0.85, f"Mean ΔPFS:  +{np.mean(pfs_delta[pfs_delta>=0]):.2f} mo (responders)  [◆]",
          transform=ax_d.transAxes, fontsize=8.5, color=DGRAY)

ax_d.set_xlabel("Simulated Patient (rank ordered by ΔPFS)", fontsize=10, color=DGRAY)
ax_d.set_ylabel("ΔPFS vs Matched Control (months)", fontsize=10, color=DGRAY)
ax_d.legend(fontsize=9, loc="upper right")
ax_d.tick_params(colors=GRAY, labelsize=8)
ax_d.set_xticks([])  # individual patient axis — no individual IDs
ax_d.spines[["top", "right"]].set_visible(False)
ax_d.spines[["left", "bottom"]].set_color(LTGRAY)
ax_d.grid(axis="y", color=LTGRAY, lw=0.7, alpha=0.7)

ax_d.text(0.01, -0.12,
          "⚠ ADDS DL simulation [◆] — Weibull-based individual projection. "
          "Not from actual patient data.",
          transform=ax_d.transAxes, fontsize=7.5, color="#C0392B", style="italic")

# ── Footer ────────────────────────────────────────────────────────
fig.text(0.5, 0.01,
         "[★] NatureComm 2026 — §Clinical Development  |  [◆] ADDS in silico estimate  |  "
         "Cohort: KRAS-mutant mCRC, N=280 (2:1), consistent throughout  |  "
         "Generated: ADDS System  |  v1.0  —  ALL RESULTS SIMULATED",
         ha="center", va="bottom", fontsize=7.5, color=GRAY, style="italic")

plt.savefig(r"f:\ADDS\figures\fig8A_simulated_phase2.png",
            dpi=180, bbox_inches="tight", facecolor=BG)
plt.close()
print("Saved: f:\\ADDS\\figures\\fig8A_simulated_phase2.png")
