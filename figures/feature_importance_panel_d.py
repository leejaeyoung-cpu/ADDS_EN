"""
feature_importance_panel_d.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ADDS Model — Feature Importance (Permutation, n=15 repeats)
정확히 Top-8 feature만 표시. 레이블 줄바꿈 없음.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

plt.rcParams.update({
    "font.family":       "DejaVu Sans",
    "font.size":         10,
    "axes.facecolor":    "white",
    "figure.facecolor":  "white",
    "axes.edgecolor":    "#BDC3C7",
    "axes.linewidth":    0.9,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "xtick.color":       "#1A252F",
    "ytick.color":       "#1A252F",
    "text.color":        "#1A252F",
    "grid.color":        "#D5D8DC",
    "grid.linewidth":    0.7,
    "grid.alpha":        0.6,
})

# ── Data: Top-8 from permutation_importance_global.json ──────────────
# Sorted descending by mean importance
TOP8 = [
    ("Bliss score",     0.305, 0.009),
    ("PK AUC (norm)",   0.109, 0.004),
    ("DL confidence",   0.097, 0.004),
    ("IL-6",            0.097, 0.006),
    ("Pritamab Cmax",   0.096, 0.003),
    ("DCR",             0.095, 0.003),
    ("ORR",             0.091, 0.005),
    ("Best % change",   0.081, 0.003),
]

labels  = [r[0] for r in TOP8]
means   = np.array([r[1] for r in TOP8])
stds    = np.array([r[2] for r in TOP8])

# Publication-quality color palette (8 distinct)
COLORS8 = [
    "#2471A3",  # Bliss — Blue
    "#27AE60",  # PK AUC — Green
    "#E67E22",  # DL conf — Orange
    "#C0392B",  # IL-6 — Red
    "#8E44AD",  # Cmax — Purple
    "#16A085",  # DCR — Teal
    "#D4AC0D",  # ORR — Gold
    "#717D7E",  # Best % change — Gray
]

# ── Figure ────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 6), dpi=300)

y_pos = np.arange(len(labels))[::-1]   # top item at position 7 (drawn first)

bars = ax.barh(
    y_pos, means, xerr=stds,
    color=COLORS8, height=0.58, alpha=0.85,
    edgecolor="white", linewidth=0.8,
    capsize=5, error_kw=dict(color="#555555", lw=1.3),
    zorder=3,
)

# Value labels (right of error bar)
for bar, v, e in zip(bars, means, stds):
    ax.text(
        v + e + 0.004,
        bar.get_y() + bar.get_height() / 2,
        f"{v:.3f} ± {e:.3f}",
        va="center", ha="left",
        fontsize=9, color="#1A252F", fontweight="bold",
    )

# Y-axis ticks
ax.set_yticks(y_pos)
ax.set_yticklabels(labels, fontsize=11)

# X-axis
ax.set_xlim(0, 0.36)
ax.set_xlabel("Permutation Importance (mean decrease in R²)",
               fontsize=11, fontweight="bold")
ax.grid(axis="x", alpha=0.4, zorder=0)

# Title
ax.set_title(
    "D  |  Feature Importance (Permutation, n=15 repeats)\n"
    "[Each bar = mean ± SD across 15 permutation repeats  |  Model: GBM v5  |  n=1,000]",
    fontsize=12, fontweight="bold", loc="left", pad=12,
)

# Note
ax.text(
    0.98, 0.03,
    "Note: Top-8 of 19 total features shown\n(remaining 11 features: importance < 0.08)",
    transform=ax.transAxes,
    ha="right", va="bottom",
    fontsize=8.5, color="#808B96", style="italic",
)

# Footnote
fig.text(
    0.5, -0.04,
    "Permutation importance computed on synthetic cohort v5 (n=1,000). "
    "Each feature shuffled independently; score = mean R² decrease over n=15 repeats. "
    "ADDS Lab, Inha University Hospital, 2026.",
    ha="center", fontsize=7.5, color="#E74C3C", style="italic",
)

plt.tight_layout(rect=[0.05, 0.02, 1.0, 1.0])

out = r"f:\ADDS\figures\feature_importance_panel_d.png"
plt.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
plt.close()
print(f"Saved → {out}")
