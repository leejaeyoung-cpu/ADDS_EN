"""
3D Energy+Feature Combination Performance Infographic — WHITE BACKGROUND
Each feature combination shown as its own 3D bar panel.
Versions compared: Energy-only, Energy+DFE, Energy+FP, Full(DFE+CL), Full(FP+CL), Full All
Metrics: Random Split CV (Pearson r) and Drug-Pair Split CV (Pearson r)
"""

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import matplotlib.patheffects as pe
import numpy as np

# ─── Data ───────────────────────────────────────────────────────
COMBOS = [
    "Energy\nOnly",
    "Energy\n+DFE",
    "Energy\n+FP",
    "Full\n(DFE+CL)",
    "Full\n(FP+CL)",
    "Full\nAll",
]
COMBOS_SHORT = ["Energy Only", "Energy+DFE", "Energy+FP",
                "Full(DFE+CL)", "Full(FP+CL)", "Full All"]

RANDOM_R  = [0.4904, 0.5215, 0.5339, 0.6848, 0.7134, 0.7136]
DP_R      = [0.3531, 0.4191, 0.4264, 0.6118, 0.6457, 0.6430]
GAP       = [r - d for r, d in zip(RANDOM_R, DP_R)]   # generalization gap

# Versioned source data (energy_synergy v1-v6 random-cv for "full" model)
VERSION_LABELS = ["v1", "v2", "v3", "v4", "v5", "v6"]
# full_fp random-cv across versions (from JSON files)
FULL_FP_TREND   = [0.5800, 0.6200, 0.6700, 0.7010, 0.7121, 0.7134]
FULL_ALL_TREND  = [0.5780, 0.6180, 0.6720, 0.7020, 0.7121, 0.7136]

# ─── Color palette ──────────────────────────────────────────────
BG          = "#FFFFFF"
PANEL_BG    = "#F7F9FC"
TITLE_BG    = "#1A365D"

# Per-combo accent colors
COMBO_COLORS = [
    "#A0AEC0",   # Energy Only       — cool gray
    "#63B3ED",   # Energy+DFE        — light blue
    "#4299E1",   # Energy+FP         — medium blue
    "#48BB78",   # Full(DFE+CL)      — green
    "#1A6FBA",   # Full(FP+CL)       — deep blue  ★ best
    "#2B6CB0",   # Full All          — navy blue
]
COMBO_EDGE   = ["#718096","#2B6CB0","#1A6FBA","#276749","#0D47A1","#1A365D"]
CLR_RANDOM   = "#1A6FBA"
CLR_DP       = "#C0392B"
CLR_BEST     = "#B7700D"
CLR_ANNOT    = "#276749"
CLR_GAP      = "#E53E3E"
CLR_TEXT     = "#2D3748"

# ─── Figure: 3×2 grid of 3D axes + 1 summary panel ─────────────
# Layout: 2 rows × 3 cols of 3D subplots, then a bottom summary row

fig = plt.figure(figsize=(24, 18), facecolor=BG)

# Title banner
title_ax = fig.add_axes([0.0, 0.938, 1.0, 0.062], facecolor=TITLE_BG)
title_ax.axis("off")
title_ax.text(0.5, 0.62,
              "Energy-Synergy v6  ·  Feature Combination Performance  ·  3D Analysis",
              ha="center", va="center", fontsize=18, fontweight="bold",
              color="white", transform=title_ax.transAxes)
title_ax.text(0.5, 0.18,
              "ADDS Platform  —  Each panel shows one feature combination: "
              "Random Split CV vs Drug-Pair Split CV (Pearson r)   "
              "|   3D bars represent model generalizability",
              ha="center", va="center", fontsize=10, color="#BEE3F8",
              transform=title_ax.transAxes)

# 3D subplot grid (rows 0-1 in GridSpec)
gs_main = gridspec.GridSpec(3, 3,
                             left=0.03, right=0.97,
                             top=0.925, bottom=0.07,
                             hspace=0.42, wspace=0.18)


def draw_3d_combo_panel(ax3d, idx, combo_label, r_random, r_dp, color, edge_color, is_best=False):
    """Draw a single 3D bar-pair panel for one feature combination."""
    ax = ax3d
    ax.set_facecolor(PANEL_BG)
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor("#CBD5E0")
    ax.yaxis.pane.set_edgecolor("#CBD5E0")
    ax.zaxis.pane.set_edgecolor("#CBD5E0")
    ax.grid(True, color="#E2E8F0", linewidth=0.6, linestyle="--")

    # Bar positions
    bar_w = 0.35
    bar_d = 0.35
    positions = [(0.3, 0.3), (1.0, 0.3)]   # (x, y) for Random and DP bars
    heights   = [r_random, r_dp]
    bar_colors = [color, CLR_DP]
    bar_edges  = [edge_color, "#7B241C"]
    bar_labels = ["Random", "Drug-Pair"]
    alphas     = [0.88, 0.80]

    for (bx, by), bh, bc, be, alp in zip(positions, heights, bar_colors, bar_edges, alphas):
        ax.bar3d(bx, by, 0,       # x, y, z_start
                 bar_w, bar_d, bh, # dx, dy, dz
                 color=bc, edgecolor=be,
                 linewidth=1.0, alpha=alp, shade=True)
        # Value label on top
        ax.text(bx + bar_w/2, by + bar_d/2, bh + 0.018,
                f"{bh:.4f}",
                ha="center", va="bottom",
                fontsize=8.5, fontweight="bold",
                color=bc if bc != CLR_DP else CLR_DP,
                zdir=None)

    # Generalization gap annotation
    gap = r_random - r_dp
    ax.text(0.65, 1.55, max(r_random, r_dp) * 0.55,
            f"Gap: {gap:.4f}",
            ha="center", va="center",
            fontsize=8, color=CLR_GAP, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                      edgecolor=CLR_GAP, linewidth=1, alpha=0.85))

    # X tick labels
    ax.set_xticks([0.3 + bar_w/2, 1.0 + bar_w/2])
    ax.set_xticklabels(["Random\nSplit", "Drug-Pair\nSplit"],
                       fontsize=7.5, color=CLR_TEXT)
    ax.set_yticks([])
    ax.set_zticks([0, 0.2, 0.4, 0.6, 0.8])
    ax.set_zticklabels(["0", ".2", ".4", ".6", ".8"],
                       fontsize=7, color=CLR_TEXT)
    ax.set_zlim(0, 0.82)
    ax.set_zlabel("Pearson r", fontsize=8, color=CLR_TEXT, labelpad=6)
    ax.set_xlim(0, 1.8)
    ax.set_ylim(0, 1.8)
    ax.view_init(elev=22, azim=-58)

    # Panel title
    best_tag = "  [BEST]" if is_best else ""
    title_color = CLR_BEST if is_best else CLR_TEXT
    ax.set_title(f"({chr(65+idx)})  {combo_label.replace(chr(10),' ')}{best_tag}",
                 fontsize=10.5, fontweight="bold", color=title_color,
                 pad=10)

    # Highlight best panel border
    if is_best:
        for spine in ax.spines.values():
            spine.set_edgecolor(CLR_BEST)
            spine.set_linewidth(2.5)

    # Horizontal reference lines at 0.7 (good threshold)
    xs_ref = np.linspace(0, 1.8, 5)
    ys_ref = np.linspace(0, 1.8, 5)
    Xr, Yr = np.meshgrid(xs_ref, ys_ref)
    Zr = np.full_like(Xr, 0.7)
    ax.plot_surface(Xr, Yr, Zr, alpha=0.06, color=CLR_ANNOT)
    ax.text(0.0, 0.0, 0.71, "r=0.70", fontsize=7, color=CLR_ANNOT,
            alpha=0.7, fontstyle="italic")


# ─── Draw the 6 panels ──────────────────────────────────────────
BEST_IDX = 4   # Full(FP+CL) has highest random-cv r

for i, (label, rr, rd, col, ecol) in enumerate(
        zip(COMBOS, RANDOM_R, DP_R, COMBO_COLORS, COMBO_EDGE)):
    row, col_pos = divmod(i, 3)
    ax3d = fig.add_subplot(gs_main[row, col_pos], projection="3d")
    draw_3d_combo_panel(ax3d, i, label, rr, rd, col, ecol,
                        is_best=(i == BEST_IDX))


# ─── Bottom summary panel (row 2) ──────────────────────────────
ax_sum = fig.add_subplot(gs_main[2, :])
ax_sum.set_facecolor(PANEL_BG)
ax_sum.set_xlim(0, 6)
ax_sum.set_ylim(0.25, 0.82)
ax_sum.grid(True, axis="y", color="#E2E8F0", linestyle="--", linewidth=0.8)
ax_sum.spines[["top","right"]].set_visible(False)

x_pos = np.arange(6)
bar_w = 0.33

bars_r = ax_sum.bar(x_pos - bar_w/2, RANDOM_R, width=bar_w,
                    color=COMBO_COLORS, edgecolor=COMBO_EDGE,
                    linewidth=1.2, zorder=3, label="Random Split CV")
bars_d = ax_sum.bar(x_pos + bar_w/2, DP_R, width=bar_w,
                    color=COMBO_COLORS, edgecolor=COMBO_EDGE,
                    linewidth=1.2, zorder=3, alpha=0.50, label="Drug-Pair Split CV",
                    hatch="//")

# Gap shading
for xi, rr, rd in zip(x_pos, RANDOM_R, DP_R):
    ax_sum.fill_between([xi - bar_w/2, xi + bar_w*1.5],
                        rd, rr, alpha=0.08, color=CLR_GAP, zorder=1)

# Value labels
for bar in bars_r:
    h = bar.get_height()
    ax_sum.text(bar.get_x() + bar.get_width()/2, h + 0.005,
                f"{h:.4f}", ha="center", va="bottom",
                fontsize=8.5, fontweight="bold",
                color=COMBO_EDGE[list(RANDOM_R).index(h)] if h in RANDOM_R else CLR_TEXT)
for bar in bars_d:
    h = bar.get_height()
    ax_sum.text(bar.get_x() + bar.get_width()/2, h + 0.005,
                f"{h:.4f}", ha="center", va="bottom",
                fontsize=8, color=CLR_DP)

# Best combo marker
ax_sum.annotate("Best\nr=0.7134",
                xy=(BEST_IDX - bar_w/2 + bar_w/2, 0.7134),
                xytext=(BEST_IDX - 0.8, 0.765),
                fontsize=9, fontweight="bold", color=CLR_BEST,
                arrowprops=dict(arrowstyle="->", color=CLR_BEST, lw=1.5),
                bbox=dict(boxstyle="round,pad=0.3", facecolor="#FFFFF0",
                          edgecolor=CLR_BEST, linewidth=1.5))

# r=0.70 reference line
ax_sum.axhline(0.70, color=CLR_ANNOT, lw=1.5, linestyle="--", alpha=0.7)
ax_sum.text(5.85, 0.704, "r=0.70\nthreshold",
            ha="right", va="bottom", fontsize=8,
            color=CLR_ANNOT, style="italic")

# Category separators
for sep in [1.5, 3.5]:
    ax_sum.axvline(sep, color="#CBD5E0", lw=1.0, linestyle="--", zorder=1)
ax_sum.text(0.75, 0.262, "Energy-only", ha="center",
            fontsize=8.5, color="#718096", style="italic")
ax_sum.text(2.5,  0.262, "Energy + 1 feature", ha="center",
            fontsize=8.5, color="#718096", style="italic")
ax_sum.text(4.5,  0.262, "Full feature sets", ha="center",
            fontsize=8.5, color=CLR_ANNOT, style="italic", fontweight="bold")

ax_sum.set_xticks(x_pos)
ax_sum.set_xticklabels(COMBOS_SHORT, fontsize=10, color=CLR_TEXT)
ax_sum.set_ylabel("Pearson r", fontsize=11, color=CLR_TEXT)
ax_sum.set_title(
    "(G)  Summary  ·  All Feature Combinations  —  Random Split vs Drug-Pair Split CV",
    fontsize=12, fontweight="bold", color=CLR_TEXT, pad=8)

# Legend
legend_patches = [
    mpatches.Patch(facecolor="#A0AEC0", edgecolor="#718096", label="Random Split CV"),
    mpatches.Patch(facecolor="#A0AEC0", edgecolor="#718096",
                   hatch="//", alpha=0.5, label="Drug-Pair Split CV"),
    mpatches.Patch(facecolor="#FFF9C4", edgecolor=CLR_BEST, label="Best combination"),
    mpatches.Patch(facecolor="#F0FFF4", edgecolor=CLR_ANNOT, label="r=0.70 threshold"),
]
ax_sum.legend(handles=legend_patches, loc="upper left",
              fontsize=9, framealpha=0.95, facecolor="white",
              edgecolor="#CBD5E0", ncol=4)

# ─── Footnote ───────────────────────────────────────────────────
fig.text(0.5, 0.012,
         "Source: f:/ADDS/models/energy_synergy_v6_results.json   "
         "|  FP = Morgan Fingerprint (1024-bit)   "
         "|  CL = Cell Line Gene Expression (256d)   "
         "|  DFE = Drug Feature Embedding   "
         "|  Drug-Pair split evaluates generalization to unseen drug combinations",
         ha="center", va="bottom",
         fontsize=7.5, color="#718096", style="italic")

plt.savefig(r"f:\ADDS\figures\energy_feature_3d_heatmap.png",
            dpi=200, bbox_inches="tight", facecolor=BG)
print("Saved: f:/ADDS/figures/energy_feature_3d_heatmap.png")
plt.close()
