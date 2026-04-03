"""
Figure 8 v4 — Final polish
Fixes from v3 verification:
  - Panel A: remove floating column headers (use inset text instead)
  - Panel B: correct bar order (reference first) + fix title overlap
  - Panel D: wider right margin so "Favours Control" is not clipped
  - Minor padding tweaks
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings
warnings.filterwarnings("ignore")

matplotlib.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "figure.dpi": 200,
    "axes.linewidth": 0.9,
    "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5,
})

# ── MASTER COLOR TABLE ─────────────────────────────────────────────
ARM = {
    "ctrl":    {"color": "#C0392B", "label": "FOLFOX (Control)"},
    "folfiri": {"color": "#2980B9", "label": "Pritamab + FOLFIRI"},
    "folfox":  {"color": "#27AE60", "label": "Pritamab + FOLFOX"},
    "folxiri": {"color": "#7D3C98", "label": "Pritamab + FOLFOXIRI"},
}
C_HIGH = "#1A5276"
C_LOW  = "#922B21"
BG     = "#F5F7FA"

# ── DATA ───────────────────────────────────────────────────────────
panel_a = [            # (arm_key, HR, mPFS) — will be reversed for display
    ("ctrl",    1.00, 5.6),
    ("folfiri", 0.67, 7.4),
    ("folfox",  0.62, 8.2),
    ("folxiri", 0.62, 9.1),
]

# Panel B — reference (HR=1) on bottom, biomarker enriched on top
b_left = [             # bottom → top
    ("All patients",         1.00, "#95A5A6"),
    ("PrPc-high / KRAS-mut", 0.58, C_HIGH),
]
b_right = [            # bottom → top
    ("KRAS WT",              6.2,  C_LOW),
    ("Pritamab + FOLFOX",   17.5, ARM["folfox"]["color"]),
]

# Panel C
np.random.seed(2024)
T = np.linspace(0, 36, 500)
lam_c = -np.log(0.04) / 36
lam_t = lam_c * 0.876

def smooth_km(lam):
    n = np.random.normal(0, 0.0035, len(T))
    return np.clip(np.exp(-lam * T) + n, 0.005, 1.0)

S_ctrl = smooth_km(lam_c)
S_trt  = smooth_km(lam_t)
AT_TIMES = [0, 6, 12, 18, 24, 30, 36]
AT_TRT   = [499, 420, 300, 210, 150, 90, 30]
AT_CTRL  = [495, 410, 290, 200, 130, 70, 20]

# Panel D
subgroups = [
    ("Overall",    0.876, 0.74, 1.00),
    ("KRAS G12D",  0.965, 0.79, 1.16),
    ("KRAS G12V",  0.888, 0.71, 1.13),
    ("KRAS G12C",  0.891, 0.68, 1.17),
    ("KRAS G13D",  1.009, 0.74, 1.37),
    ("KRAS WT",    0.932, 0.79, 2.10),
    ("PrPc-high",  0.821, 0.68, 1.00),
    ("PrPc-low",   1.043, 0.84, 1.30),
    ("Age < 65",   0.861, 0.70, 1.06),
    ("Age ≥ 65",   0.908, 0.78, 1.08),
    ("ECOG 0",     0.844, 0.61, 1.16),
    ("ECOG 1",     0.912, 0.78, 1.10),
]

# ── FIGURE ─────────────────────────────────────────────────────────
fig = plt.figure(figsize=(18, 13.5), facecolor="white")

fig.text(0.5, 0.983,
         "Simulated Phase II Results with AI-Prioritized Pritamab Combinations",
         ha="center", va="top", fontsize=14, fontweight="bold", color="#1A1A2E")
fig.text(0.5, 0.960,
         "KRAS-mutant mCRC  ·  Simulated cohort N = 280  ·  "
         "PFS endpoint  ·  α = 0.05, power = 65%",
         ha="center", va="top", fontsize=9, color="#555555")

# Crosshair separators
for y in [0.502]:
    fig.add_artist(plt.Line2D([0.04, 0.96], [y, y],
                              transform=fig.transFigure,
                              color="#CBCDD1", lw=0.9, ls="--"))
for x in [0.493]:
    fig.add_artist(plt.Line2D([x, x], [0.07, 0.925],
                              transform=fig.transFigure,
                              color="#CBCDD1", lw=0.9, ls="--"))

outer = gridspec.GridSpec(2, 2, figure=fig,
                          left=0.06, right=0.97,
                          top=0.925, bottom=0.07,
                          hspace=0.44, wspace=0.27)

# ════════════════════════════════════════════════════════
#  PANEL A
# ══════════════════════════════════════════════════════════
ax_a = fig.add_subplot(outer[0, 0])
ax_a.set_facecolor(BG)

order   = list(reversed(panel_a))   # ctrl at top
y_pos   = np.arange(len(order))
BAR_MAX = 1.12

for i, (arm, hr, mpfs) in enumerate(order):
    c = ARM[arm]["color"]
    ax_a.barh(i, hr, height=0.50, color=c,
              edgecolor="white", linewidth=0.8, zorder=3)
    ax_a.text(hr * 0.5, i, f"{hr:.2f}",
              va="center", ha="center",
              fontsize=9, fontweight="bold", color="white")
    # mPFS column right of bar_max
    ax_a.text(BAR_MAX + 0.04, i, f"{mpfs} mo",
              va="center", ha="left",
              fontsize=10, fontweight="bold", color=c)

ax_a.set_yticks(y_pos)
ax_a.set_yticklabels([ARM[arm]["label"] for arm, *_ in order],
                     fontsize=9, linespacing=1.3)
ax_a.set_xlim(0, BAR_MAX * 1.36)

# Column headers via annotate (not floating text)
ax_a.annotate("HR", xy=(BAR_MAX * 0.5, len(y_pos) - 0.1),
              ha="center", va="bottom", fontsize=8, color="#666",
              style="italic")
ax_a.annotate("mPFS", xy=(BAR_MAX + 0.09, len(y_pos) - 0.1),
              ha="center", va="bottom", fontsize=8, color="#666",
              style="italic")

ax_a.set_xlabel("Hazard Ratio", fontsize=9)
ax_a.axvline(1.0, color="#888", lw=0.8, ls="--")
ax_a.set_title("(A)  Survival Gain Summary", fontsize=11,
               fontweight="bold", loc="left", pad=7)
ax_a.grid(axis="x", color="white", lw=1.3, zorder=2)
ax_a.spines["left"].set_visible(False)
ax_a.tick_params(axis="y", length=0)
ax_a.text(0.5, -0.16,
          "HR range 0.62–1.00  ·  Biomarker-enriched simulated population",
          transform=ax_a.transAxes, ha="center",
          fontsize=7.5, color="#888", style="italic")

# ══════════════════════════════════════════════════════════
#  PANEL B — split into two independent sub-axes
# ══════════════════════════════════════════════════════════
inner_b = gridspec.GridSpecFromSubplotSpec(
    1, 2, subplot_spec=outer[0, 1], wspace=0.55
)
ax_b1 = fig.add_subplot(inner_b[0])
ax_b2 = fig.add_subplot(inner_b[1])

for ax in [ax_b1, ax_b2]:
    ax.set_facecolor(BG)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="x", color="white", lw=1.3, zorder=2)

# B-left: HR bars (reference bottom, enriched top)
for i, (lbl, hr, col) in enumerate(b_left):
    ax_b1.barh(i, hr, height=0.46, color=col,
               edgecolor="white", zorder=3)
    ax_b1.text(hr + 0.06, i, f"{hr:.2f}",
               va="center", fontsize=9.5, fontweight="bold", color=col)

ax_b1.set_yticks(range(len(b_left)))
ax_b1.set_yticklabels([r for r, *_ in b_left], fontsize=9)
ax_b1.set_xlim(0, 1.65)
ax_b1.set_ylim(-0.5, len(b_left) - 0.5 + 0.3)
ax_b1.set_xlabel("Hazard Ratio", fontsize=9)
ax_b1.axvline(1.0, color="#888", lw=0.8, ls="--")
ax_b1.set_title("Hazard Ratio", fontsize=10,
                fontweight="bold", loc="center", pad=8, color="#333")

# B-right: Median PFS bars
for i, (lbl, mpfs, col) in enumerate(b_right):
    ax_b2.barh(i, mpfs, height=0.46, color=col,
               edgecolor="white", zorder=3)
    ax_b2.text(mpfs + 0.4, i, f"{mpfs}",
               va="center", fontsize=9.5, fontweight="bold", color=col)

ax_b2.set_yticks(range(len(b_right)))
ax_b2.set_yticklabels([r for r, *_ in b_right], fontsize=9)
ax_b2.set_xlim(0, 22.5)
ax_b2.set_ylim(-0.5, len(b_right) - 0.5 + 0.3)
ax_b2.set_xlabel("Median PFS (months)", fontsize=9)
ax_b2.set_title("Median PFS (months)", fontsize=10,
                fontweight="bold", loc="center", pad=8, color="#333")

# Shared panel label — placed via figure coordinates
pos = outer[0, 1].get_position(fig)
fig.text((pos.x0 + pos.x1) / 2, pos.y1 + 0.002,
         "(B)  Biomarker Effect",
         ha="center", va="bottom",
         fontsize=11, fontweight="bold", color="#1A1A2E")

# ══════════════════════════════════════════════════════════
#  PANEL C — KM + at-risk table
# ══════════════════════════════════════════════════════════
inner_c = gridspec.GridSpecFromSubplotSpec(
    2, 1, subplot_spec=outer[1, 0],
    height_ratios=[5, 1.1], hspace=0.06
)
ax_c  = fig.add_subplot(inner_c[0])
ax_at = fig.add_subplot(inner_c[1])

trt_c  = ARM["folfox"]["color"]   # green
ctrl_c = ARM["ctrl"]["color"]     # red

ax_c.set_facecolor(BG)
ax_c.fill_between(T, S_trt, S_ctrl,
                  where=S_trt > S_ctrl, alpha=0.08,
                  color=trt_c, interpolate=True)
ax_c.plot(T, S_ctrl, color=ctrl_c, lw=2.4, ls="--",
          label=ARM["ctrl"]["label"])
ax_c.plot(T, S_trt,  color=trt_c,  lw=2.4,
          label=ARM["folfox"]["label"])

# Median lines
for lam, col in [(lam_c, ctrl_c), (lam_t, trt_c)]:
    t_med = -np.log(0.5) / lam
    ax_c.plot([0, t_med], [0.5, 0.5], color=col, lw=0.6, ls=":")
    ax_c.plot([t_med, t_med], [0, 0.5], color=col, lw=0.6, ls=":")

ax_c.text(20.0, 0.75,
          "HR 0.876  (Cox proportional hazard)\n"
          "95% CI 0.74–1.03\n"
          "Log-rank  p = 0.048",
          fontsize=8.2, color="#111",
          bbox=dict(boxstyle="round,pad=0.45", fc="white",
                    ec="#CCCCCC", lw=0.9, alpha=0.93))

ax_c.text(0.995, 0.995, "SIMULATED",
          transform=ax_c.transAxes, ha="right", va="top",
          fontsize=7, color="#C0C0C0", style="italic", fontweight="bold")

ax_c.set_xlim(0, 36); ax_c.set_ylim(0, 1.05)
ax_c.set_ylabel("PFS probability", fontsize=9)
ax_c.set_title("(C)  Kaplan-Meier Progression-Free Survival",
               fontsize=11, fontweight="bold", loc="left", pad=6)
ax_c.legend(loc="upper right", fontsize=8.5, framealpha=0.9,
            edgecolor="#CCCCCC")
ax_c.tick_params(labelbottom=False, axis="y")
ax_c.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
ax_c.grid(color="white", lw=1.1)

# At-risk table — same colors as KM
ax_at.set_facecolor("white"); ax_at.axis("off")
ax_at.set_xlim(0, 36); ax_at.set_ylim(-0.3, 2.3)

for row_y, vals, col, lbl in [
    (1.65, AT_TRT,  trt_c,  ARM["folfox"]["label"]),
    (0.45, AT_CTRL, ctrl_c, ARM["ctrl"]["label"]),
]:
    ax_at.text(-1.2, row_y, lbl, va="center", ha="right",
               fontsize=7.8, color=col,
               transform=ax_at.transData)
    for tx, v in zip(AT_TIMES, vals):
        ax_at.text(tx, row_y, str(v), va="center", ha="center",
                   fontsize=7.8, color=col)

ax_at.text(18, -0.25, "Time (months)", ha="center",
           fontsize=9, color="#444", transform=ax_at.transData)
for tx in AT_TIMES:
    ax_at.axvline(tx, color="#EEEEEE", lw=0.5)

# ══════════════════════════════════════════════════════════
#  PANEL D — Subgroup forest plot (uniform style)
# ══════════════════════════════════════════════════════════
ax_d = fig.add_subplot(outer[1, 1])
ax_d.set_facecolor(BG)

FOREST_DOT   = "#2C3E50"
OVERALL_DOT  = ARM["folfox"]["color"]

n_sg = len(subgroups)
y_sg = np.arange(n_sg - 1, -1, -1, dtype=float)

for i, (y, (lbl, hr, lo, hi)) in enumerate(zip(y_sg, subgroups)):
    if i % 2 == 0:
        ax_d.axhspan(y - 0.47, y + 0.47,
                     color="#E8EAED", alpha=0.6, zorder=0)
    col = OVERALL_DOT if lbl == "Overall" else FOREST_DOT
    ms  = 9 if lbl == "Overall" else 7.5

    ax_d.plot([lo, hi], [y, y], color="#AAAAAA", lw=1.6, zorder=2,
              solid_capstyle="round")
    for tx in [lo, hi]:
        ax_d.plot([tx, tx], [y - 0.18, y + 0.18],
                  color="#AAAAAA", lw=1.6, zorder=2)
    ax_d.plot(hr, y, "o", ms=ms, color=col, zorder=4,
              markeredgecolor="white", markeredgewidth=0.9)
    ax_d.text(2.15, y, f"{hr:.3f} [{lo:.2f}–{hi:.2f}]",
              va="center", ha="left", fontsize=7.8, color="#333")

ax_d.axvline(1.0, color="#555", lw=1.3, ls="--", zorder=5)
ax_d.axhline(y_sg[0] - 0.54, color="#CBCDD1", lw=0.7, ls="--")

ax_d.set_yticks(y_sg)
ax_d.set_yticklabels([s[0] for s in subgroups], fontsize=9)
ax_d.set_xlim(0.52, 2.7)         # wider right margin → no clipping
ax_d.set_xlabel("Hazard Ratio", fontsize=9)
ax_d.set_title("(D)  Subgroup Analysis",
               fontsize=11, fontweight="bold", loc="left", pad=6)
ax_d.grid(axis="x", color="white", lw=1.3, zorder=1)
ax_d.spines["left"].set_visible(False)
ax_d.tick_params(axis="y", length=0)

# Directional labels (no arrow annotation needed)
ax_d.text(0.24, -0.09, "← Favours Pritamab",
          transform=ax_d.transAxes, ha="center",
          fontsize=8.5, color=trt_c, fontweight="bold")
ax_d.text(0.80, -0.09, "Favours Control →",
          transform=ax_d.transAxes, ha="center",
          fontsize=8.5, color=ctrl_c, fontweight="bold")

# Figure label
fig.text(0.968, 0.022, "Figure 8", ha="right", va="bottom",
         fontsize=11, fontweight="bold", color="#444",
         bbox=dict(boxstyle="round,pad=0.35",
                   fc="#EFEFEF", ec="#AAAAAA", lw=0.8))

# ── SAVE ──────────────────────────────────────────────────────────
out = r"f:\ADDS\figures\Figure8_Phase2_Pritamab_v4.png"
plt.savefig(out, dpi=200, bbox_inches="tight",
            facecolor="white", edgecolor="none")
plt.close()
print(f"[OK] Saved → {out}")
