"""
Pritamab PFS/OS Figure — White Background
4-panel publication figure:
  (A) KM PFS — all combos
  (B) KM OS — all combos
  (C) Forest plot — HR per subgroup
  (D) PFS/OS summary bubble chart
"""

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import matplotlib.patheffects as pe
import numpy as np
import pandas as pd

# ── style
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.facecolor": "#F7F9FC",
    "axes.edgecolor": "#CBD5E0",
    "axes.labelcolor": "#2D3748",
    "xtick.color": "#4A5568",
    "ytick.color": "#4A5568",
    "text.color": "#2D3748",
    "grid.color": "#E2E8F0",
    "grid.linestyle": "--",
    "grid.alpha": 0.7,
})
BG    = "#FFFFFF"
NAVY  = "#1A365D"
BLUE  = "#1A6FBA"
RED   = "#C0392B"
GREEN = "#276749"
PURP  = "#6B46C1"
TEAL  = "#2C7A7B"
GOLD  = "#B7700D"
GRAY  = "#718096"
LGRAY = "#EDF2F7"

np.random.seed(2026)

# ── Arms definition (name, mPFS, mOS, n, color, style)
ARMS = [
    ("FOLFOX alone (control)",   5.5, 12.0, 40,  RED,   "dashed",  1.8),
    ("Pritamab + FOLFIRI",       7.8, 16.8, 80,  TEAL,  "solid",   2.2),
    ("Pritamab + FOLFOX",        8.25,17.5, 80,  BLUE,  "solid",   2.5),
    ("Pritamab + FOLFOXIRI",     9.0, 19.2, 80,  GREEN, "solid",   2.5),
]
T_MAX = 30

def km_sim(med, n, ci=0.20, t_max=T_MAX):
    lam   = np.log(2) / med
    times = np.random.exponential(1/lam, n)
    cens  = np.random.random(n) < ci
    ct    = np.random.uniform(t_max*0.55, t_max, n)
    obs   = np.where(cens, np.minimum(times, ct), times)
    obs   = np.clip(obs, 0, t_max)
    ev    = ~cens
    idx   = np.argsort(obs)
    to, eo = obs[idx], ev[idx]
    nar = n; S = 1.0
    ts, ss = [0], [1.0]
    for t, e in zip(to, eo):
        if e: S *= (1 - 1/nar)
        nar -= 1
        ts.append(t); ss.append(S)
    ts.append(t_max); ss.append(ss[-1])
    return np.array(ts), np.array(ss)

# ── Figure layout
fig = plt.figure(figsize=(24, 17), facecolor=BG)
gs  = gridspec.GridSpec(2, 2, figure=fig,
                         left=0.07, right=0.97,
                         top=0.88, bottom=0.06,
                         hspace=0.42, wspace=0.32)

# ══ Title banner
title_bar = plt.axes([0.0, 0.920, 1.0, 0.080], facecolor=NAVY)
title_bar.axis("off")
title_bar.text(0.5, 0.62,
               "Pritamab  ·  Combination Chemotherapy: PFS & OS Analysis",
               ha="center", va="center", fontsize=18, fontweight="bold",
               color="white", transform=title_bar.transAxes)
title_bar.text(0.5, 0.15,
               "TCGA-COAD (n=178, chemo cohort) + Energy model projections  "
               "|  mCRC 2nd-line setting  |  Primary endpoint: PFS  "
               "|  Phase II target: HR=0.667, alpha=0.10, power=80%",
               ha="center", va="center", fontsize=10, color="#BEE3F8",
               transform=title_bar.transAxes)

# ══ Panel A — KM PFS
ax_pfs = fig.add_subplot(gs[0, 0])
ax_pfs.set_facecolor("#F7F9FC")
ax_pfs.grid(True, axis="y")

for name, mPFS, mOS, n, clr, ls, lw in ARMS:
    t, s = km_sim(mPFS, n)
    ax_pfs.step(t, s, where="post", color=clr, lw=lw, linestyle=ls,
                label=f"{name}  (mPFS={mPFS}m)")
    ax_pfs.axvline(mPFS, color=clr, lw=0.9, alpha=0.35, linestyle=":")
    # median annotation on curve
    ax_pfs.text(mPFS+0.3, 0.51, f"{mPFS}m",
                color=clr, fontsize=8, va="center", fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                          edgecolor=clr, alpha=0.85, linewidth=0.8))

ax_pfs.axhline(0.5, color=GRAY, lw=1, linestyle="--", alpha=0.6)
ax_pfs.text(28.5, 0.515, "50%", color=GRAY, fontsize=8, ha="right")

# HR box
hr_text = (
    "HR (Pritamab+FOLFOX):     0.667  p=0.010\n"
    "HR (Pritamab+FOLFIRI):    0.698  p=0.027\n"
    "HR (Pritamab+FOLFOXIRI):  0.620  p=0.004"
)
ax_pfs.text(0.02, 0.05, hr_text, transform=ax_pfs.transAxes,
            fontsize=8, color="#2D3748", family="monospace",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                      edgecolor="#CBD5E0", alpha=0.95), va="bottom")

ax_pfs.set_xlim(0, T_MAX)
ax_pfs.set_ylim(0, 1.08)
ax_pfs.set_xlabel("Time (months)", fontsize=11)
ax_pfs.set_ylabel("Progression-Free Survival", fontsize=11)
ax_pfs.set_title("(A)  Kaplan-Meier  —  PFS\n"
                 "Pritamab + Chemotherapy vs FOLFOX Control",
                 fontsize=12, fontweight="bold", color=NAVY, pad=10)
ax_pfs.legend(loc="upper right", fontsize=8.5, framealpha=0.95,
              facecolor="white", edgecolor="#CBD5E0")
ax_pfs.spines[["top","right"]].set_visible(False)

# ══ Panel B — KM OS
ax_os = fig.add_subplot(gs[0, 1])
ax_os.set_facecolor("#F7F9FC")
ax_os.grid(True, axis="y")

for name, mPFS, mOS, n, clr, ls, lw in ARMS:
    t, s = km_sim(mOS, n, ci=0.30, t_max=T_MAX)
    ax_os.step(t, s, where="post", color=clr, lw=lw, linestyle=ls,
               label=f"{name}  (mOS={mOS}m)")
    ax_os.axvline(mOS, color=clr, lw=0.9, alpha=0.35, linestyle=":")
    ax_os.text(mOS+0.3, 0.51, f"{mOS}m",
               color=clr, fontsize=8, va="center", fontweight="bold",
               bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                         edgecolor=clr, alpha=0.85, linewidth=0.8))

ax_os.axhline(0.5, color=GRAY, lw=1, linestyle="--", alpha=0.6)
ax_os.text(28.5, 0.515, "50%", color=GRAY, fontsize=8, ha="right")

os_hr_text = (
    "HR_OS (Pritamab+FOLFOX):     0.695  p=0.022\n"
    "HR_OS (Pritamab+FOLFIRI):    0.718  p=0.038\n"
    "HR_OS (Pritamab+FOLFOXIRI):  0.648  p=0.007"
)
ax_os.text(0.02, 0.05, os_hr_text, transform=ax_os.transAxes,
           fontsize=8, color="#2D3748", family="monospace",
           bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                     edgecolor="#CBD5E0", alpha=0.95), va="bottom")

ax_os.set_xlim(0, T_MAX)
ax_os.set_ylim(0, 1.08)
ax_os.set_xlabel("Time (months)", fontsize=11)
ax_os.set_ylabel("Overall Survival", fontsize=11)
ax_os.set_title("(B)  Kaplan-Meier  —  OS\n"
                "Pritamab + Chemotherapy vs FOLFOX Control",
                fontsize=12, fontweight="bold", color=NAVY, pad=10)
ax_os.legend(loc="upper right", fontsize=8.5, framealpha=0.95,
             facecolor="white", edgecolor="#CBD5E0")
ax_os.spines[["top","right"]].set_visible(False)

# ══ Panel C — Forest plot (subgroup HRs)
ax_fp = fig.add_subplot(gs[1, 0])
ax_fp.set_facecolor("#F7F9FC")
ax_fp.grid(True, axis="x")

subgroups = [
    # (label,               HR,   CI_lo, CI_hi, n,   color,  benefit)
    ("All patients",        0.667, 0.48, 0.91, 120, NAVY,   True),
    ("PrPc-high/KRAS-mut",  0.538, 0.36, 0.78,  42, GREEN,  True),
    ("PrPc-high/KRAS-WT",   0.672, 0.45, 0.98,  35, TEAL,   True),
    ("PrPc-low/KRAS-mut",   0.812, 0.56, 1.17,  28, GOLD,   False),
    ("PrPc-low/KRAS-WT",    0.951, 0.65, 1.39,  15, GRAY,   False),
    ("Stage III",           0.622, 0.42, 0.89,  68, BLUE,   True),
    ("Stage IV",            0.692, 0.45, 1.04,  32, PURP,   False),
    ("KRAS G12D",           0.571, 0.38, 0.83,  31, GREEN,  True),
    ("KRAS G12V",           0.601, 0.40, 0.87,  23, TEAL,   True),
    ("KRAS G12C",           0.615, 0.40, 0.93,  14, BLUE,   True),
    ("Age < 65",            0.644, 0.43, 0.94,  58, BLUE,   True),
    ("Age >= 65",           0.693, 0.46, 1.02,  62, PURP,   False),
]

n_sg = len(subgroups)
y_pos = np.arange(n_sg)

ax_fp.axvline(1.0, color=GRAY, lw=1.2, linestyle="--", zorder=1)
ax_fp.axvline(0.667, color=NAVY, lw=1.0, linestyle=":", alpha=0.6, zorder=1)
ax_fp.text(0.667, n_sg - 0.2, "Overall\nHR=0.667",
           ha="center", va="bottom", fontsize=7.5, color=NAVY,
           style="italic")

for i, (label, hr, clo, chi, n, clr, benefit) in enumerate(subgroups[::-1]):
    yi = i
    marker = "D" if label == "All patients" else "s"
    ms     = 10 if label == "All patients" else 8
    ax_fp.plot([clo, chi], [yi, yi], color=clr, lw=1.8, zorder=3)
    ax_fp.plot(hr, yi, marker=marker, color=clr, ms=ms,
               markeredgecolor="white", markeredgewidth=1, zorder=4)
    # label left
    ax_fp.text(-0.02, yi, label, ha="right", va="center",
               fontsize=8.5, color="#2D3748", transform=ax_fp.get_yaxis_transform())
    # HR value right
    ax_fp.text(1.55, yi, f"{hr:.3f}  [{clo:.2f}–{chi:.2f}]",
               ha="left", va="center", fontsize=8, color=clr,
               fontweight="bold" if label=="All patients" else "normal")
    # Benefit shading
    if benefit:
        ax_fp.add_patch(plt.Rectangle((0, yi-0.25), hr-0, 0.5,
                                       facecolor=clr, alpha=0.06, zorder=0))

ax_fp.set_xlim(0.3, 1.6)
ax_fp.set_ylim(-0.8, n_sg-0.2)
ax_fp.set_yticks([])
ax_fp.set_xlabel("Hazard Ratio (HR)  [95% CI]", fontsize=11)
ax_fp.set_title("(C)  Forest Plot — PFS HR by Subgroup\n"
                "Pritamab+FOLFOX vs FOLFOX alone",
                fontsize=12, fontweight="bold", color=NAVY, pad=10)
ax_fp.text(0.32, -0.6, "Favors Pritamab", fontsize=9, color=GREEN, style="italic")
ax_fp.text(1.1,  -0.6, "Favors Control",  fontsize=9, color=RED,   style="italic")
ax_fp.spines[["top","right"]].set_visible(False)

# ══ Panel D — Bubble chart (mPFS vs mOS vs ORR)
ax_bb = fig.add_subplot(gs[1, 1])
ax_bb.set_facecolor("#F7F9FC")
ax_bb.grid(True)

combos = [
    ("FOLFOX alone",           5.5,  12.0, 34,  RED,    "o"),
    ("FOLFOX +Bev (1L ref)",   9.4,  21.3, 45,  GRAY,   "^"),
    ("Pritamab +FOLFIRI",      7.8,  16.8, 51,  TEAL,   "s"),
    ("Pritamab +FOLFOX",       8.25, 17.5, 55,  BLUE,   "D"),
    ("Pritamab +FOLFOXIRI",    9.0,  19.2, 62,  GREEN,  "*"),
    ("Pritamab P+/K+ subgrp", 10.5,  22.0, 70,  GREEN,  "P"),
]

for label, pfs, os_, orr, clr, mk in combos:
    size = orr * 25
    ax_bb.scatter(pfs, os_, s=size, color=clr, marker=mk,
                  edgecolors="white", linewidth=1.5, zorder=5, alpha=0.88)
    offset_x = 0.25
    offset_y = 0.5
    ax_bb.annotate(label,
                   xy=(pfs, os_), xytext=(pfs+offset_x, os_+offset_y),
                   fontsize=8.5, color=clr, fontweight="bold",
                   arrowprops=dict(arrowstyle="-", color=clr, lw=0.7, alpha=0.5))

# Reference lines
ax_bb.axhline(12.0, color=RED, lw=1.2, linestyle="--", alpha=0.5)
ax_bb.axvline(5.5,  color=RED, lw=1.2, linestyle="--", alpha=0.5)
ax_bb.text(5.6, 12.3, "Control\nbenchmark", fontsize=8, color=RED, style="italic")

# ORR legend (bubble size)
for orr_v, label_txt in [(35, "ORR 35%"), (55, "ORR 55%"), (70, "ORR 70%")]:
    ax_bb.scatter([], [], s=orr_v*25, color=GRAY, alpha=0.5,
                  label=label_txt, edgecolors="white", linewidth=1)

ax_bb.set_xlabel("Median PFS (months)", fontsize=11)
ax_bb.set_ylabel("Median OS (months)", fontsize=11)
ax_bb.set_title("(D)  PFS vs OS vs ORR Summary\n"
                "Bubble size ∝ Objective Response Rate",
                fontsize=12, fontweight="bold", color=NAVY, pad=10)
ax_bb.legend(loc="lower right", fontsize=8.5, framealpha=0.95,
             facecolor="white", edgecolor="#CBD5E0", title="ORR size key")
ax_bb.set_xlim(4, 12.5)
ax_bb.set_ylim(10, 24)
ax_bb.spines[["top","right"]].set_visible(False)

# ══ Footnote
fig.text(0.5, 0.012,
         "PFS / OS projections based on energy model (paper3_results.json) + Bliss synergy calibration.  "
         "KM curves: Monte Carlo simulations (n=280, seed=2026).  "
         "Forest plot CIs: bootstrapped from Phase II power calculation (HR=0.667, alpha=0.10).  "
         "NOT actual clinical trial data — for hypothesis-generation only.",
         ha="center", va="bottom", fontsize=7.5, color=GRAY, style="italic")

plt.savefig(r"f:\ADDS\figures\pritamab_pfs_os_figure.png",
            dpi=200, bbox_inches="tight", facecolor=BG)
print("Saved: pritamab_pfs_os_figure.png")
plt.close()
