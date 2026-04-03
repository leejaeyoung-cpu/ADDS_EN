"""
Pritamab Drug Synergy Map — Standalone Publication Figure (v2)
Validated 2026-03-03: Corrected Irinotecan 18.4→17.3, TAS-102 19.2→18.1
Source labels added: [Paper] / [ADDS] / [est.]
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
import matplotlib.patheffects as pe
import numpy as np

# ── Global style ────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":      "DejaVu Sans",
    "axes.facecolor":   "#F7F9FC",
    "axes.edgecolor":   "#CBD5E0",
    "axes.labelcolor":  "#2D3748",
    "xtick.color":      "#4A5568",
    "ytick.color":      "#4A5568",
    "text.color":       "#2D3748",
    "grid.color":       "#E2E8F0",
    "grid.linestyle":   "--",
    "grid.alpha":       0.7,
})

BG     = "#FFFFFF"
NAVY   = "#1A365D"
BLUE   = "#1A6FBA"
RED    = "#C0392B"
GREEN  = "#276749"
GOLD   = "#B7700D"
PURPLE = "#6B46C1"
TEAL   = "#2C7A7B"
ORANGE = "#C05621"
GRAY   = "#718096"
LGRAY  = "#EDF2F7"

# ════════════════════════════════════════════════════════════════
# DATA — Validated 2026-03-03 against Pritamab_NatureComm_Paper.txt
# ════════════════════════════════════════════════════════════════
drugs = ["Pritamab", "5-FU", "Oxaliplatin", "Irinotecan",
         "Sotorasib", "TAS-102", "Bevacizumab", "Cetuximab"]
n = len(drugs)

# Bliss synergy matrix
# [Paper] = Directly from Pritamab_NatureComm_Paper.txt (ground truth)
# [ADDS]  = ADDS 4-model consensus → Bliss conversion (ratio-scaled)
# [est.]  = Literature estimate (not from Pritamab paper)
raw = {
    (0,1): 18.4,  # Pritamab + 5-FU         [Paper] ★
    (0,2): 21.7,  # Pritamab + Oxaliplatin   [Paper] ★
    (0,3): 17.3,  # Pritamab + Irinotecan    [ADDS] (corrected from 18.4)
    (0,4): 15.8,  # Pritamab + Sotorasib     [ADDS]
    (0,5): 18.1,  # Pritamab + TAS-102       [ADDS] (corrected from 19.2)
    (0,6): 12.1,  # Pritamab + Bevacizumab   [est.]
    (0,7): 10.5,  # Pritamab + Cetuximab     [est.]
    (1,2): 16.8,  # 5-FU + Oxaliplatin       [est.] FOLFOX base
    (1,3): 15.2,  # 5-FU + Irinotecan        [est.] FOLFIRI base
    (1,4):  8.3,  # 5-FU + Sotorasib         [est.]
    (1,5): 14.1,  # 5-FU + TAS-102           [est.]
    (1,6): 13.5,  # 5-FU + Bevacizumab       [est.]
    (1,7):  6.2,  # 5-FU + Cetuximab         [est.]
    (2,3): 11.4,  # Oxaliplatin + Irinotecan  [est.]
    (2,4):  7.8,  # Oxaliplatin + Sotorasib   [est.]
    (2,5): 10.9,  # Oxaliplatin + TAS-102     [est.]
    (2,6): 12.2,  # Oxaliplatin + Bevacizumab [est.]
    (2,7):  9.1,  # Oxaliplatin + Cetuximab   [est.]
    (3,4):  6.5,  # Irinotecan + Sotorasib    [est.]
    (3,5): 13.7,  # Irinotecan + TAS-102      [est.]
    (3,6): 14.8,  # Irinotecan + Bevacizumab  [est.]
    (3,7):  5.8,  # Irinotecan + Cetuximab    [est.]
    (4,5):  8.9,  # Sotorasib + TAS-102       [est.]
    (4,6):  9.3,  # Sotorasib + Bevacizumab   [est.]
    (4,7): 12.4,  # Sotorasib + Cetuximab     [est.]
    (5,6): 10.2,  # TAS-102 + Bevacizumab     [est.]
    (5,7):  7.6,  # TAS-102 + Cetuximab       [est.]
    (6,7): 13.9,  # Bevacizumab + Cetuximab   [est.]
}

# Paper-confirmed pairs (for star annotation)
PAPER_CONFIRMED = {(0,1), (1,0), (0,2), (2,0)}

mat = np.zeros((n, n))
for (i, j), v in raw.items():
    mat[i, j] = v
    mat[j, i] = v

# ════════════════════════════════════════════════════════════════
# FIGURE: 3-panel layout
#   Left (wide):   Heatmap
#   Top-right:     Network
#   Bot-right:     Bar chart (Pritamab avg synergy per drug)
# ════════════════════════════════════════════════════════════════
fig = plt.figure(figsize=(26, 14), facecolor=BG)
gs = gridspec.GridSpec(
    2, 2, figure=fig,
    left=0.05, right=0.97,
    top=0.88, bottom=0.10,
    hspace=0.42, wspace=0.32,
    width_ratios=[1.45, 1],
    height_ratios=[1.3, 1],
)

drug_colors = [NAVY, BLUE, PURPLE, RED, ORANGE, TEAL, GREEN, GOLD]

# ── CMAP ────────────────────────────────────────────────────────
cmap = LinearSegmentedColormap.from_list(
    "syn", ["#FFFFFF", "#FEF9C3", "#FDE68A", "#F59E0B",
            "#EF4444", "#991B1B"], N=256)

# ════════════════════════════════════════════════════════════════
# PANEL A — Heatmap (left, spanning both rows)
# ════════════════════════════════════════════════════════════════
ax_hm = fig.add_subplot(gs[:, 0])
ax_hm.set_facecolor(BG)

mask = mat.copy()
np.fill_diagonal(mask, np.nan)

im = ax_hm.imshow(mask, cmap=cmap, vmin=0, vmax=25, aspect="equal")

# Cell text
for i in range(n):
    for j in range(n):
        if i == j:
            ax_hm.text(j, i, "—", ha="center", va="center",
                       fontsize=10, color=GRAY)
        else:
            v = mat[i, j]
            clr   = "white" if v > 17 else "#2D3748"
            wt    = "bold"  if v >= 18 else "normal"
            # star for paper-confirmed
            star  = "★" if (i, j) in PAPER_CONFIRMED else ""
            ax_hm.text(j, i, f"{v:.1f}{star}",
                       ha="center", va="center",
                       fontsize=10, color=clr, fontweight=wt)

# Pritamab row/col highlight border
for k in range(n):
    ax_hm.add_patch(plt.Rectangle(
        (k - 0.5, -0.5), 1, 1, fill=False,
        edgecolor=NAVY, linewidth=2.5 if k == 0 else 0))
    ax_hm.add_patch(plt.Rectangle(
        (-0.5, k - 0.5), 1, 1, fill=False,
        edgecolor=NAVY, linewidth=2.5 if k == 0 else 0))

# Source label overlay (top-right of cell for Pritamab row)
src_labels = ["", "[Paper]★", "[Paper]★", "[ADDS]", "[ADDS]", "[ADDS]", "[est.]", "[est.]"]
for j, lbl in enumerate(src_labels):
    if j == 0 or not lbl:
        continue
    color = RED if "Paper" in lbl else (PURPLE if "ADDS" in lbl else GRAY)
    ax_hm.text(j + 0.48, 0 - 0.46, lbl,
               ha="right", va="bottom", fontsize=6.5,
               color=color, fontstyle="italic", zorder=5)

ax_hm.set_xticks(range(n))
ax_hm.set_yticks(range(n))
ax_hm.set_xticklabels(drugs, rotation=35, ha="right", fontsize=11)
ax_hm.set_yticklabels(drugs, fontsize=11)

# Colorbar
cbar = fig.colorbar(im, ax=ax_hm, fraction=0.038, pad=0.03)
cbar.set_label("Bliss Synergy Score", fontsize=10, color="#2D3748")
cbar.ax.tick_params(labelcolor="#2D3748", labelsize=9)
cbar.ax.axhline(10, color=GOLD, lw=2.2, linestyle="--")
cbar.ax.text(1.6, 10, " Clinical\nthreshold\n(≥10)",
             fontsize=8, color=GOLD, va="center")
cbar.ax.axhline(18, color=RED, lw=2.0, linestyle=":")
cbar.ax.text(1.6, 18, " Strong\nsynergy\n(≥18)",
             fontsize=8, color=RED, va="center")

ax_hm.set_title(
    "(A)  Pritamab Drug Combination Synergy Map\nBliss Independence Model — Score Matrix",
    fontsize=13, fontweight="bold", color=NAVY, pad=14)

legend_src = [
    mpatches.Patch(facecolor="white", edgecolor=RED,   label="★ Paper-confirmed (5-FU: +18.4, Oxaliplatin: +21.7)"),
    mpatches.Patch(facecolor="white", edgecolor=PURPLE, label="[ADDS] 4-model consensus → Bliss (Irinotecan: 17.3, Sotorasib: 15.8)"),
    mpatches.Patch(facecolor="white", edgecolor=GRAY,   label="[est.] Literature estimate"),
]
ax_hm.legend(handles=legend_src, loc="lower center",
             bbox_to_anchor=(0.5, -0.15), ncol=1,
             fontsize=9, framealpha=0.95, facecolor="white",
             edgecolor="#CBD5E0", handlelength=0.8)

# ════════════════════════════════════════════════════════════════
# PANEL B — Network (top-right)
# ════════════════════════════════════════════════════════════════
ax_net = fig.add_subplot(gs[0, 1])
ax_net.set_facecolor(BG)
ax_net.set_aspect("equal")
ax_net.axis("off")
ax_net.set_xlim(-1.6, 1.6)
ax_net.set_ylim(-1.6, 1.6)

theta = np.linspace(0, 2 * np.pi, n, endpoint=False) + np.pi / 2
cx = np.cos(theta)
cy = np.sin(theta)

synergy_thr = 10.0
strong_thr  = 18.0

# Edges
for i in range(n):
    for j in range(i + 1, n):
        v = mat[i, j]
        if v < synergy_thr:
            continue
        lw    = 1.0 + (v - synergy_thr) / 4.5
        alpha = 0.35 + (v - synergy_thr) / 22
        color = RED if v >= strong_thr else GOLD
        ax_net.plot([cx[i], cx[j]], [cy[i], cy[j]],
                    lw=lw, alpha=min(alpha, 0.88), color=color, zorder=2)
        if v >= strong_thr:
            mx, my = (cx[i] + cx[j]) / 2, (cy[i] + cy[j]) / 2
            ax_net.text(mx * 1.08, my * 1.08, f"{v:.1f}",
                        ha="center", va="center", fontsize=7.5,
                        color=RED, fontweight="bold",
                        bbox=dict(boxstyle="round,pad=0.15", facecolor="white",
                                  edgecolor=RED, alpha=0.88, linewidth=0.8))

# Nodes
from matplotlib.patches import Circle
for i in range(n):
    r = 0.21 if i == 0 else 0.15
    ax_net.add_patch(Circle((cx[i], cy[i]), r,
                             facecolor=drug_colors[i], edgecolor="white",
                             linewidth=2.5, zorder=5))
    avg = np.mean([mat[i, j] for j in range(n) if j != i])
    ax_net.text(cx[i], cy[i], f"{avg:.1f}",
                ha="center", va="center",
                fontsize=8 if i == 0 else 7,
                color="white", fontweight="bold", zorder=6)
    ax_net.text(cx[i] * 1.42, cy[i] * 1.32, drugs[i],
                ha="center", va="center",
                fontsize=10 if i == 0 else 9,
                fontweight="bold" if i == 0 else "normal",
                color=drug_colors[i])

legend_net = [
    Line2D([0], [0], color=RED,  lw=3, label=f"Strong synergy (≥{strong_thr:.0f})"),
    Line2D([0], [0], color=GOLD, lw=2, label=f"Synergy (≥{synergy_thr:.0f})"),
    mpatches.Patch(facecolor=NAVY, label="Pritamab (central node)"),
]
ax_net.legend(handles=legend_net, loc="lower center",
              bbox_to_anchor=(0.5, -0.12), ncol=1,
              fontsize=8.5, framealpha=0.95, facecolor="white",
              edgecolor="#CBD5E0")
ax_net.set_title("(B)  Synergy Network\nNode = avg Bliss score",
                 fontsize=12, fontweight="bold", color=NAVY, pad=12)

# ════════════════════════════════════════════════════════════════
# PANEL C — Pritamab pair bar chart (bottom-right)
# ════════════════════════════════════════════════════════════════
ax_bar = fig.add_subplot(gs[1, 1])
ax_bar.set_facecolor("#F7F9FC")

# Values for Pritamab pairs (row 0)
pair_drugs  = drugs[1:]           # exclude self
pair_vals   = [mat[0, j] for j in range(1, n)]
pair_colors = drug_colors[1:]
pair_src    = ["[Paper]★", "[Paper]★", "[ADDS]", "[ADDS]", "[ADDS]", "[est.]", "[est.]"]

y_pos = np.arange(len(pair_drugs))
bars  = ax_bar.barh(y_pos, pair_vals, height=0.62,
                    color=pair_colors, edgecolor="white",
                    linewidth=1.2, zorder=3)

ax_bar.axvline(10, color=GOLD, lw=1.8, linestyle="--", alpha=0.8, zorder=4)
ax_bar.axvline(18, color=RED,  lw=1.5, linestyle=":",  alpha=0.7, zorder=4)
ax_bar.text(10.3, -0.7, "≥10\n(synergy)", fontsize=7.5, color=GOLD, va="top")
ax_bar.text(18.3, -0.7, "≥18\n(strong)",  fontsize=7.5, color=RED,  va="top")

for bar, val, src, clr in zip(bars, pair_vals, pair_src, pair_colors):
    ax_bar.text(val + 0.3, bar.get_y() + bar.get_height() / 2,
                f"+{val:.1f}  {src}",
                va="center", fontsize=8.5,
                color=RED if "Paper" in src else (PURPLE if "ADDS" in src else GRAY),
                fontweight="bold" if "Paper" in src else "normal")

ax_bar.set_yticks(y_pos)
ax_bar.set_yticklabels(pair_drugs, fontsize=10)
ax_bar.set_xlim(0, 30)
ax_bar.set_xlabel("Bliss Synergy Score (+)", fontsize=10)
ax_bar.set_title("(C)  Pritamab Pair Synergy Scores\n(Sorted by drug class)",
                 fontsize=12, fontweight="bold", color=NAVY, pad=10)
ax_bar.spines[["top", "right"]].set_visible(False)
ax_bar.grid(True, axis="x", alpha=0.4)

# ── Main title / subtitle ────────────────────────────────────────
fig.text(0.5, 0.965,
         "Pritamab  ·  Drug Combination Synergy Map",
         ha="center", va="top",
         fontsize=18, fontweight="bold", color=NAVY)
fig.text(0.5, 0.935,
         "Bliss Independence Model  |  Score > 10 = clinical synergy threshold  |  "
         "Source: ADDS 4-model consensus + Pritamab paper (★) + literature [est.]  |  "
         "Validated 2026-03-03",
         ha="center", va="top", fontsize=10, color=GRAY)

fig.savefig(r"f:\ADDS\figures\pritamab_synergy_map_v2.png",
            dpi=200, bbox_inches="tight", facecolor=BG)
print("Saved: pritamab_synergy_map_v2.png")
