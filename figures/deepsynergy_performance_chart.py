"""
DeepSynergy Performance Trend Infographic  — WHITE BACKGROUND VERSION
Visualizes model performance (Pearson r) across DeepSynergy v1 → v5
+ Energy-Synergy feature ablation study
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from matplotlib.patches import FancyBboxPatch
from matplotlib.lines import Line2D
import matplotlib.patheffects as pe
import numpy as np

# ─── Style (white background) ────────────────────────────────────
plt.rcParams.update({
    "font.family":      "DejaVu Sans",
    "figure.facecolor": "#FFFFFF",
    "axes.facecolor":   "#F7F9FC",
    "axes.edgecolor":   "#CBD5E0",
    "axes.labelcolor":  "#2D3748",
    "xtick.color":      "#4A5568",
    "ytick.color":      "#4A5568",
    "text.color":       "#2D3748",
    "grid.color":       "#E2E8F0",
    "grid.linestyle":   "--",
    "grid.alpha":       0.8,
})

# ─── Data ────────────────────────────────────────────────────────
versions     = ["v1\n(baseline)", "v3\n(GELU+BN)", "v4\n(+CL_expr)", "v5\n(+expanded\nCL_expr)"]
versions_x   = [1, 3, 4, 5]
random_cv    = [0.4286, 0.6644, 0.6866, 0.7342]
drug_pair_cv = [None,   0.6044, 0.6533, 0.7074]
random_std   = [0.0,    0.0026, 0.0026, 0.0027]
dp_std       = [0.0,    0.0093, 0.0123, 0.0051]

crc_real      = 0.291
crc_finetuned = 0.5976

feat_labels_short = ["Energy\nOnly", "Energy\n+DFE", "Energy\n+FP",
                     "Full\n(DFE+CL)", "Full\n(FP+CL)", "Full\nAll"]
energy_random    = [0.4904, 0.5215, 0.5339, 0.6848, 0.7134, 0.7136]
energy_drug_pair = [0.3531, 0.4191, 0.4264, 0.6118, 0.6457, 0.6430]

# ─── Color palette (light-theme friendly) ────────────────────────
CLR_RANDOM = "#1A6FBA"   # deep blue
CLR_DP     = "#C0392B"   # crimson
CLR_BEST   = "#B7700D"   # amber/gold
CLR_ANNOT  = "#276749"   # forest green
CLR_PURPLE = "#6B46C1"   # purple accent
CLR_TEAL   = "#2C7A7B"   # teal
CLR_TITLE  = "#1A365D"   # navy
CLR_BORDER = "#90CDF4"   # light blue border
CLR_PANEL  = "#F7F9FC"   # panel background
CLR_BG     = "#FFFFFF"   # figure background
CLR_KPI_BG = "#EBF8FF"   # KPI box background
CLR_SEP    = "#A0AEC0"   # separator

# ─── Figure layout ────────────────────────────────────────────────
fig = plt.figure(figsize=(22, 14), facecolor=CLR_BG)
gs  = GridSpec(2, 3, figure=fig,
               left=0.06, right=0.97,
               top=0.88, bottom=0.08,
               hspace=0.52, wspace=0.38)

ax_line = fig.add_subplot(gs[0, :2])
ax_bar  = fig.add_subplot(gs[0, 2])
ax_feat = fig.add_subplot(gs[1, :2])
ax_kpi  = fig.add_subplot(gs[1, 2])

# ══════════════════════════════════════════════
# PANEL 1 — Line: version trend
# ══════════════════════════════════════════════
ax = ax_line
ax.set_facecolor(CLR_PANEL)
ax.grid(True, axis="y")

x = np.array(versions_x)
dp_filled = np.array([d if d is not None else random_cv[i] for i, d in enumerate(drug_pair_cv)])
ax.fill_between(x, dp_filled, random_cv, alpha=0.10, color=CLR_RANDOM, zorder=1)

# Random CV
ax.plot(x, random_cv, "o-", color=CLR_RANDOM, lw=2.8, ms=10,
        markerfacecolor=CLR_RANDOM, markeredgecolor="white", markeredgewidth=2,
        zorder=5, label="Random Split CV (Pearson r)")
for xi, yi, si in zip(x, random_cv, random_std):
    ax.errorbar(xi, yi, yerr=si, fmt="none", ecolor=CLR_RANDOM,
                elinewidth=1.5, capsize=5, zorder=6)

# Drug-Pair CV
dp_x = [versions_x[i] for i, d in enumerate(drug_pair_cv) if d is not None]
dp_y = [d for d in drug_pair_cv if d is not None]
dp_s = [dp_std[i] for i, d in enumerate(drug_pair_cv) if d is not None]
ax.plot(dp_x, dp_y, "s--", color=CLR_DP, lw=2.8, ms=10,
        markerfacecolor=CLR_DP, markeredgecolor="white", markeredgewidth=2,
        zorder=5, label="Drug-Pair Split CV (Pearson r)")
for xi, yi, si in zip(dp_x, dp_y, dp_s):
    ax.errorbar(xi, yi, yerr=si, fmt="none", ecolor=CLR_DP,
                elinewidth=1.5, capsize=5, zorder=6)

# Value labels
for xi, yi in zip(x, random_cv):
    ax.text(xi, yi + 0.020, f"{yi:.4f}", ha="center", va="bottom",
            fontsize=9.5, color=CLR_RANDOM, fontweight="bold")
for xi, yi in zip(dp_x, dp_y):
    ax.text(xi, yi - 0.027, f"{yi:.4f}", ha="center", va="top",
            fontsize=9.5, color=CLR_DP, fontweight="bold")

# Best annotation
ax.annotate("Best\nr = 0.7342",
            xy=(5, 0.7342), xytext=(4.50, 0.762),
            fontsize=9.5, fontweight="bold", color=CLR_BEST,
            arrowprops=dict(arrowstyle="->", color=CLR_BEST, lw=1.8),
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#FFFFF0",
                      edgecolor=CLR_BEST, linewidth=1.5))

# Delta labels
for i, (xi, yi) in enumerate(zip(x, random_cv)):
    if i > 0:
        delta = yi - random_cv[i-1]
        ax.text(xi, yi + 0.048, f"+{delta:.4f}", ha="center", va="bottom",
                fontsize=8, color=CLR_ANNOT, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.15", facecolor="#F0FFF4",
                          edgecolor=CLR_ANNOT, linewidth=0.8, alpha=0.9))

ax.set_xlim(0.5, 5.5)
ax.set_ylim(0.35, 0.84)
ax.set_xticks(versions_x)
ax.set_xticklabels(versions, fontsize=10)
ax.set_ylabel("Pearson Correlation (r)", fontsize=11, color="#2D3748")
ax.set_title("DeepSynergy  ·  Model Version Performance Trend (v1 → v5)",
             fontsize=13, fontweight="bold", color=CLR_TITLE, pad=12)
ax.legend(loc="upper left", fontsize=9.5, framealpha=0.9,
          facecolor="white", edgecolor=CLR_BORDER, labelcolor="#2D3748")
ax.spines[["top", "right"]].set_visible(False)
ax.tick_params(axis="both", labelsize=10, colors="#4A5568")

# Milestone badges
milestones = {
    1: ("Baseline MLP", "#EDF2F7", "#718096"),
    3: ("GELU+BN\nDropout",   "#EBF8FF", "#2B6CB0"),
    4: ("FP+CL_expr\n+Tissue","#EBF8FF", "#2B6CB0"),
    5: ("Expanded CL\n(56.9% match)", "#F0FFF4", "#276749"),
}
for xv, (text, fc, tc) in milestones.items():
    ax.text(xv, 0.365, text, ha="center", va="bottom", fontsize=7.5,
            color=tc, style="italic",
            bbox=dict(boxstyle="round,pad=0.25", facecolor=fc,
                      edgecolor=tc, linewidth=0.8, alpha=0.95))

# ══════════════════════════════════════════════
# PANEL 2 — Bar: CRC fine-tuning
# ══════════════════════════════════════════════
ax = ax_bar
ax.set_facecolor(CLR_PANEL)
ax.grid(True, axis="y")

cats   = ["CRC\nPre-trained", "CRC\nFine-tuned"]
vals   = [crc_real, crc_finetuned]
bcolors = [CLR_DP, CLR_RANDOM]
bars   = ax.bar(cats, vals, color=bcolors, width=0.45,
                edgecolor="white", linewidth=1.5, zorder=3)

for bar, val in zip(bars, vals):
    ax.text(bar.get_x() + bar.get_width()/2, val + 0.016,
            f"r = {val:.4f}", ha="center", va="bottom",
            fontsize=10.5, fontweight="bold", color=bar.get_facecolor())

gain = crc_finetuned - crc_real
pct  = (crc_finetuned / crc_real - 1) * 100
ax.text(0.5, (crc_real + crc_finetuned) / 2,
        f"+{gain:.4f}\n(+{pct:.1f}%)",
        ha="center", va="center", fontsize=9, color=CLR_ANNOT,
        fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.25", facecolor="#F0FFF4",
                  edgecolor=CLR_ANNOT, linewidth=1))

ax.set_ylim(0, 0.76)
ax.set_title("CRC Domain\nFine-tuning Effect", fontsize=11,
             fontweight="bold", color=CLR_TITLE, pad=8)
ax.set_ylabel("Pearson r", fontsize=10)
ax.spines[["top", "right"]].set_visible(False)
ax.tick_params(labelsize=10, colors="#4A5568")

# ══════════════════════════════════════════════
# PANEL 3 — Grouped bar: feature ablation
# ══════════════════════════════════════════════
ax = ax_feat
ax.set_facecolor(CLR_PANEL)
ax.grid(True, axis="y")

n = len(feat_labels_short)
x = np.arange(n)
w = 0.38

b1 = ax.bar(x - w/2, energy_random,    width=w, label="Random Split",
            color=CLR_RANDOM, edgecolor="white", linewidth=0.8, zorder=3, alpha=0.90)
b2 = ax.bar(x + w/2, energy_drug_pair, width=w, label="Drug-Pair Split",
            color=CLR_DP,     edgecolor="white", linewidth=0.8, zorder=3, alpha=0.90)

for bar in b1:
    h = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, h + 0.006, f"{h:.3f}",
            ha="center", va="bottom", fontsize=8.5,
            color=CLR_RANDOM, fontweight="bold")
for bar in b2:
    h = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, h + 0.006, f"{h:.3f}",
            ha="center", va="bottom", fontsize=8.5,
            color=CLR_DP, fontweight="bold")

# Highlight best
for i in [4, 5]:
    ax.bar(x[i] - w/2, energy_random[i], width=w,
           color=CLR_RANDOM, edgecolor=CLR_BEST, linewidth=2.5, zorder=4, alpha=0.90)

ax.annotate("Best combo\nr = 0.7134", xy=(x[4], 0.7134),
            xytext=(3.3, 0.745),
            fontsize=8.5, color=CLR_BEST, fontweight="bold",
            arrowprops=dict(arrowstyle="->", color=CLR_BEST, lw=1.5),
            bbox=dict(boxstyle="round,pad=0.25", facecolor="#FFFFF0",
                      edgecolor=CLR_BEST, linewidth=1.5))

ax.set_xticks(x)
ax.set_xticklabels(feat_labels_short, fontsize=10)
ax.set_ylim(0.28, 0.82)
ax.set_ylabel("Pearson Correlation (r)", fontsize=11, color="#2D3748")
ax.set_title("Energy-Synergy v6  ·  Feature Ablation Study  (Random vs Drug-Pair Split)",
             fontsize=12, fontweight="bold", color=CLR_TITLE, pad=10)
ax.legend(loc="upper left", fontsize=9.5, framealpha=0.9,
          facecolor="white", edgecolor=CLR_BORDER)
ax.spines[["top", "right"]].set_visible(False)
ax.tick_params(labelsize=10, colors="#4A5568")

for xi in [1.5, 3.5]:
    ax.axvline(xi, color=CLR_SEP, lw=1.2, linestyle="--", zorder=1)

ax.text(0.5, 0.292, "Energy-only", ha="center", fontsize=8,
        color=CLR_SEP, style="italic")
ax.text(2.5, 0.292, "Energy + 1 feat.", ha="center", fontsize=8,
        color=CLR_SEP, style="italic")
ax.text(4.5, 0.292, "Full feature sets", ha="center", fontsize=8,
        color=CLR_SEP, style="italic")

# ══════════════════════════════════════════════
# PANEL 4 — KPI Scoreboard
# ══════════════════════════════════════════════
ax = ax_kpi
ax.set_facecolor("#F0F8FF")
ax.axis("off")
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
t = ax.transAxes

ax.text(0.5, 0.96, "Key Results", ha="center", va="top",
        fontsize=13, fontweight="bold", color=CLR_TITLE, transform=t)
ax.plot([0.04, 0.96], [0.91, 0.91], color=CLR_BORDER, lw=2, transform=t)

kpis = [
    ("Best Model",    "DeepSynergy v5",           CLR_RANDOM,  "#EBF8FF"),
    ("Random CV",     "r = 0.7342 +/- 0.0027",     CLR_RANDOM,  "#EBF8FF"),
    ("Drug-Pair CV",  "r = 0.7074 +/- 0.0051",     CLR_DP,      "#FFF5F5"),
    ("v1→v5 Gain",    "+71.1% improvement",        CLR_ANNOT,   "#F0FFF4"),
    ("Training Data", "927,011 samples",           CLR_PURPLE,  "#FAF5FF"),
    ("Unique Pairs",  "62,087 drug pairs",         CLR_PURPLE,  "#FAF5FF"),
    ("CL Match Rate", "56.9%  -- key gain",        CLR_TEAL,    "#E6FFFA"),
    ("Best Features", "FP + CL_expr + Tissue",     CLR_TEAL,    "#E6FFFA"),
    ("Ablation Best", "r = 0.7134 (FP+CL)",        CLR_BEST,    "#FFFFF0"),
    ("Architecture",  "MLP 2316→2048→1024→256",    "#718096",   "#EDF2F7"),
]

y_pos = 0.87
for label, value, border_clr, bg_clr in kpis:
    box = FancyBboxPatch((0.03, y_pos - 0.065), 0.94, 0.072,
                         boxstyle="round,pad=0.005",
                         facecolor=bg_clr, edgecolor=border_clr,
                         linewidth=1.3, transform=t, zorder=2)
    ax.add_patch(box)
    ax.text(0.07, y_pos - 0.020, label, ha="left", va="center",
            fontsize=7.5, color="#4A5568", transform=t, fontweight="bold")
    ax.text(0.96, y_pos - 0.042, value, ha="right", va="center",
            fontsize=8.2, color=border_clr, fontweight="bold", transform=t)
    y_pos -= 0.092

# ══════════════════════════════════════════════
# MAIN TITLE BANNER
# ══════════════════════════════════════════════
# Title background bar
title_bar = FancyBboxPatch((0.01, 0.900), 0.98, 0.084,
                           boxstyle="round,pad=0.005",
                           facecolor="#1A365D", edgecolor="#2B6CB0",
                           linewidth=1.5, transform=fig.transFigure, zorder=0)
fig.add_artist(title_bar)

fig.text(0.5, 0.955,
         "DeepSynergy  ·  Drug Combination Synergy Prediction  ·  Performance Evolution",
         ha="center", va="top",
         fontsize=17, fontweight="bold", color="white")
fig.text(0.5, 0.924,
         "ADDS Platform  —  Multi-version benchmarking (ONEIL dataset)  "
         "|  Pearson r across Random & Drug-Pair cross-validation splits",
         ha="center", va="top",
         fontsize=10, color="#BEE3F8")

# ══════════════════════════════════════════════
# FOOTNOTE
# ══════════════════════════════════════════════
fig.text(0.5, 0.018,
         "Source: f:/ADDS/models/synergy/  ·  energy_synergy_v6_results.json  ·  deep_synergy_results.json   "
         "|  FP=Morgan Fingerprint, CL=Cell Line Gene Expression, DFE=Drug Feature Embedding   "
         "|  Drug-Pair split evaluates generalization to unseen drug combinations",
         ha="center", va="bottom",
         fontsize=7.5, color="#718096", style="italic")

plt.savefig(r"f:\ADDS\figures\deepsynergy_performance_chart.png",
            dpi=200, bbox_inches="tight", facecolor=CLR_BG)
print("Saved: f:/ADDS/figures/deepsynergy_performance_chart.png")
plt.close()
