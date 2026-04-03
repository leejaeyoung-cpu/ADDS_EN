"""
Figure 8 (v2): Simulated Phase II Results — Publication-quality
All layout issues fixed from v1.
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings
warnings.filterwarnings("ignore")

matplotlib.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 200,
    "axes.linewidth": 0.8,
})

# ── COLORS ──────────────────────────────────────────────
C_CTRL    = "#C0392B"
C_FOLFIRI = "#2980B9"
C_FOLFOX  = "#27AE60"
C_FOLXIRI = "#7D3C98"
C_ACCENT  = "#E67E22"
BG        = "#F4F6F8"

# ── FIGURE ──────────────────────────────────────────────
fig = plt.figure(figsize=(18, 13), facecolor="white")

# Title block
fig.text(0.5, 0.975,
         "Simulated Phase II Results with AI-Prioritized Pritamab Combinations",
         ha="center", va="top", fontsize=14, fontweight="bold", color="#1A1A2E")
fig.text(0.5, 0.952,
         "Simulated trial in KRAS-mutant mCRC  ·  N = 280  ·  P = Phase I extended  "
         "·  PFS powered for HR = α 0.05, power = 65%",
         ha="center", va="top", fontsize=8.5, color="#555")

# Separator lines
for y in [0.505]:
    fig.add_artist(plt.Line2D([0.04, 0.97], [y, y],
                              transform=fig.transFigure,
                              color="#CCCCCC", lw=0.8, ls="--"))
for x in [0.49]:
    fig.add_artist(plt.Line2D([x, x], [0.07, 0.94],
                              transform=fig.transFigure,
                              color="#CCCCCC", lw=0.8, ls="--"))

outer = gridspec.GridSpec(2, 2, figure=fig,
                          left=0.05, right=0.97,
                          top=0.94, bottom=0.07,
                          hspace=0.42, wspace=0.28)

# ══════════════════════════════════════════════════════
# PANEL A: Survival Gain Summary
# ══════════════════════════════════════════════════════
ax_a = fig.add_subplot(outer[0, 0])
ax_a.set_facecolor(BG)

# FOLFOX (Control) at top, treatment arms below — matching original
labels = [
    "FOLFOX\n(Control)",
    "Pritamab + FOLFIRI\n(HR = 0.67)",
    "Pritamab + FOLFOX\n(HR = 0.62)",
    "Pritamab + FOLFOXIRI\n(HR = 0.62)",
]
hrs   = [1.00, 0.67, 0.62, 0.62]
mpfs  = [5.6,  7.4,  8.2,  9.1]
cols  = [C_CTRL, C_FOLFIRI, C_FOLFOX, C_FOLXIRI]

# Reverse so Control is at TOP visually
labels_r = labels[::-1]; hrs_r = hrs[::-1]
mpfs_r   = mpfs[::-1];   cols_r = cols[::-1]
y_pos    = np.arange(len(labels_r))

bars = ax_a.barh(y_pos, hrs_r, color=cols_r,
                 height=0.52, edgecolor="white", linewidth=0.8, zorder=3)

for i, (h, m, c) in enumerate(zip(hrs_r, mpfs_r, cols_r)):
    ax_a.text(h + 0.012, i, f"{m}", va="center", ha="left",
              fontsize=10.5, fontweight="bold", color=c)
    if h > 0.1:
        ax_a.text(min(h * 0.5, h - 0.04), i, f"{h:.2f}",
                  va="center", ha="center",
                  fontsize=8.5, color="white", fontweight="bold")

ax_a.set_yticks(y_pos)
ax_a.set_yticklabels(labels_r, fontsize=8.5, linespacing=1.3)
ax_a.set_xlim(0, 1.32)
ax_a.set_xlabel("Hazard Ratio", fontsize=9, labelpad=4)
ax_a.set_title("(A)  Survival Gain Summary", fontsize=10.5,
               fontweight="bold", loc="left", pad=6)
ax_a.axvline(1.0, color="#999", lw=0.8, ls="--")
ax_a.text(1.22, len(y_pos) - 0.5, "mPFS\n(months)",
          fontsize=8, color="#555", ha="center", style="italic")
ax_a.grid(axis="x", color="white", lw=1.3, zorder=2)
ax_a.spines["left"].set_visible(False)
ax_a.tick_params(axis="both", labelsize=8)
ax_a.text(0.5, -0.17,
          "HR range 0.62 – 6.67  ·  Simulated Phase II  ·  Biomarker-enriched population",
          transform=ax_a.transAxes, ha="center", fontsize=7.5,
          color="#888", style="italic")

# ══════════════════════════════════════════════════════
# PANEL B: Biomarker Effect — dual display
# ══════════════════════════════════════════════════════
ax_b = fig.add_subplot(outer[0, 1])
ax_b.set_facecolor(BG)

# Section header
ax_b.set_title("(B)  Biomarker Effect", fontsize=10.5,
               fontweight="bold", loc="left", pad=6)

# ── Top part: Hazard Ratios ──────────────────────────
# Two horizontal bars with reference line
bio_hr_labels = ["All patients", "PrPc-high / KRAS-mut"]
bio_hr_vals   = [1.00, 0.58]
bio_hr_cols   = ["#95A5A6", "#2980B9"]
y_hr = [3.2, 2.2]

ax_b.barh(y_hr, bio_hr_vals, height=0.45,
          color=bio_hr_cols, edgecolor="white", zorder=3)
for y, val, col, lbl in zip(y_hr, bio_hr_vals, bio_hr_cols, bio_hr_labels):
    ax_b.text(val + 0.3, y, f"{val:.2f}", va="center",
              fontsize=9.5, fontweight="bold", color=col)
    ax_b.text(-0.4, y, lbl, va="center", ha="right",
              fontsize=8.5, color="#333")

ax_b.text(9, 3.7, "Hazard Ratio", fontsize=8.5,
          ha="center", color="#555", style="italic")

# ── Divider ─────────────────────────────────────────
ax_b.axhline(1.7, color="#CCCCCC", lw=0.8, ls="--")

# ── Bottom part: Median PFS bars ────────────────────
bio_pfs_labels = ["Pritamab + FOLFOX", "KRAS WT"]
bio_pfs_vals   = [17.5, 6.21]
bio_pfs_cols   = [C_FOLFOX, C_CTRL]
y_pfs = [1.1, 0.1]

ax_b.barh(y_pfs, bio_pfs_vals, height=0.45,
          color=bio_pfs_cols, edgecolor="white", zorder=3)
for y, val, col, lbl in zip(y_pfs, bio_pfs_vals, bio_pfs_cols, bio_pfs_labels):
    ax_b.text(val + 0.3, y, f"{val:.1f}", va="center",
              fontsize=9.5, fontweight="bold", color=col)
    ax_b.text(-0.4, y, lbl, va="center", ha="right",
              fontsize=8.5, color="#333")

ax_b.text(9, 1.65, "Median PFS (months)", fontsize=8.5,
          ha="center", color="#555", style="italic")

ax_b.set_xlim(0, 22)
ax_b.set_ylim(-0.45, 4.1)
ax_b.set_xlabel("Hazard Ratio  /  Median PFS (months)", fontsize=9)
ax_b.set_yticks([])
ax_b.grid(axis="x", color="white", lw=1.3, zorder=2)
ax_b.spines["left"].set_visible(False)
ax_b.axvline(1.0, color="#999", lw=0.6, ls="--")

# ══════════════════════════════════════════════════════
# PANEL C: Kaplan-Meier + At-Risk Table
# ══════════════════════════════════════════════════════
inner_c = gridspec.GridSpecFromSubplotSpec(
    2, 1, subplot_spec=outer[1, 0],
    height_ratios=[5, 1.1], hspace=0.05
)
ax_c  = fig.add_subplot(inner_c[0])
ax_at = fig.add_subplot(inner_c[1])

# Simulate KM curves (exponential, HR=0.876)
np.random.seed(2024)
t = np.linspace(0, 36, 400)
lam_c = 0.092; lam_t = lam_c * 0.876
noise = lambda: np.random.normal(0, 0.004, 400)
s_c = np.clip(np.exp(-lam_c * t) + noise(), 0.01, 1)
s_t = np.clip(np.exp(-lam_t * t) + noise(), 0.01, 1)

ax_c.set_facecolor(BG)
ax_c.fill_between(t, s_t, s_c, alpha=0.07, color=C_FOLFOX)
ax_c.plot(t, s_c, color=C_CTRL,   lw=2.3, ls="--", label="FOLFOX (Control)")
ax_c.plot(t, s_t, color=C_FOLFOX, lw=2.3,          label="Pritamab + FOLFOX")

# Reference lines at median
ax_c.axhline(0.5, color="#aaa", lw=0.7, ls=":")
med_t_idx = np.argmin(np.abs(s_t - 0.5))
med_c_idx = np.argmin(np.abs(s_c - 0.5))
ax_c.axvline(t[med_t_idx], color=C_FOLFOX, lw=0.5, ls=":")
ax_c.axvline(t[med_c_idx], color=C_CTRL,   lw=0.5, ls=":")

# HR annotation
ax_c.text(21, 0.76,
          "HR 0.876\n95% CI 0.74–1.03\np = 0.048",
          fontsize=8.5, color="#111",
          bbox=dict(boxstyle="round,pad=0.45", fc="white",
                    ec="#CCCCCC", lw=0.9, alpha=0.92))

ax_c.set_xlim(0, 36)
ax_c.set_ylim(0, 1.06)
ax_c.set_ylabel("PFS probability", fontsize=9)
ax_c.set_title("A  Kaplan-Meier Progression-Free Survival",
               fontsize=10.5, fontweight="bold", loc="left", pad=5)
ax_c.legend(loc="upper right", fontsize=8.5, framealpha=0.9,
            edgecolor="#CCCCCC")
ax_c.tick_params(labelbottom=False)
ax_c.tick_params(axis="y", labelsize=8)
ax_c.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
ax_c.grid(color="white", lw=1.1)

# At-risk table
atrisk_t  = [0, 6, 12, 18, 24, 30, 36]
atrisk_tv = [499, 420, 300, 210, 150, 90, 30]
atrisk_cv = [495, 410, 290, 200, 130, 70, 20]

ax_at.set_facecolor("white")
ax_at.set_xlim(0, 36); ax_at.set_ylim(0, 2)
ax_at.axis("off")

for row_y, vals, col, lbl in [
    (1.45, atrisk_tv, C_FOLFOX, "Pritamab+FOLFOX"),
    (0.45, atrisk_cv, C_CTRL,   "FOLFOX"),
]:
    for tx, v in zip(atrisk_t, vals):
        ax_at.text(tx, row_y, str(v), va="center", ha="center",
                   fontsize=7.8, color=col)
    ax_at.text(-3.2, row_y, lbl, va="center", ha="right",
               fontsize=7.5, color=col,
               transform=ax_at.transData)

ax_at.text(18, -0.3, "Time (months)", ha="center",
           fontsize=8.5, color="#444",
           transform=ax_at.transData)
for tx in atrisk_t:
    ax_at.axvline(tx, color="#EEEEEE", lw=0.6)

# ══════════════════════════════════════════════════════
# PANEL D: Subgroup Forest Plot
# ══════════════════════════════════════════════════════
ax_d = fig.add_subplot(outer[1, 1])
ax_d.set_facecolor(BG)

subgroups = [
    "Overall", "KRAS G12D", "KRAS G12V", "KRAS G12C", "KRAS G13D",
    "KRAS WT", "PrPc-high", "PrPc-low", "Age <65", "Age ≥65",
    "ECOG 0", "ECOG 1",
]
sg_hr = [0.876, 0.965, 0.888, 0.891, 1.009, 0.932, 0.821, 1.043, 0.861, 0.908, 0.844, 0.912]
sg_lo = [0.74,  0.79,  0.71,  0.68,  0.74,  0.79,  0.68,  0.84,  0.70,  0.78,  0.61,  0.78]
sg_hi = [1.00,  1.16,  1.13,  1.17,  1.37,  2.10,  1.00,  1.30,  1.06,  1.08,  1.16,  1.10]

n_sg = len(subgroups)
y_sg = np.arange(n_sg - 1, -1, -1, dtype=float)

# Shade alternate rows
for i, y in enumerate(y_sg):
    if i % 2 == 0:
        ax_d.axhspan(y - 0.45, y + 0.45, color="#EAECF0", alpha=0.5, zorder=0)

for y, hr, lo, hi in zip(y_sg, sg_hr, sg_lo, sg_hi):
    # colour by direction
    if hr < 0.88:
        dot_col = "#1A6BAD"
    elif hr > 1.0:
        dot_col = C_CTRL
    else:
        dot_col = C_ACCENT
    ax_d.plot([lo, hi], [y, y], color="#BBBBBB", lw=1.5, zorder=2,
              solid_capstyle="round")
    ax_d.plot([lo, lo], [y - 0.15, y + 0.15], color="#BBBBBB", lw=1.5, zorder=2)
    ax_d.plot([hi, hi], [y - 0.15, y + 0.15], color="#BBBBBB", lw=1.5, zorder=2)
    ax_d.plot(hr, y, "o", ms=7.5, color=dot_col, zorder=4,
              markeredgecolor="white", markeredgewidth=0.8)
    ax_d.text(2.18, y, f"{hr:.3f} [{lo:.2f}–{hi:.2f}]",
              va="center", ha="left", fontsize=7.5, color="#333")

ax_d.axvline(1.0, color="#666", lw=1.3, ls="--", zorder=5)
ax_d.set_yticks(y_sg)
ax_d.set_yticklabels(subgroups, fontsize=8.8)
ax_d.set_xlim(0.55, 2.5)
ax_d.set_xlabel("Hazard Ratio", fontsize=9, labelpad=4)
ax_d.set_title("B  Subgroup Analysis", fontsize=10.5,
               fontweight="bold", loc="left", pad=5)
ax_d.text(0.72, -0.09, "Favours\nPritamab", transform=ax_d.transAxes,
          ha="center", fontsize=8.5, color=C_FOLFOX, fontweight="bold")
ax_d.text(0.88, -0.09, "Favours\nControl",  transform=ax_d.transAxes,
          ha="center", fontsize=8.5, color=C_CTRL,   fontweight="bold")
ax_d.tick_params(axis="both", labelsize=8)
ax_d.grid(axis="x", color="white", lw=1.3, zorder=1)
ax_d.set_axisbelow(True)

# Figure label
fig.text(0.97, 0.025, "Figure  8", ha="right", va="bottom",
         fontsize=11, fontweight="bold", color="#444",
         bbox=dict(boxstyle="round,pad=0.35", fc="#F0F0F0", ec="#AAAAAA"))

# ── SAVE ────────────────────────────────────────────────
out = r"f:\ADDS\figures\Figure8_Phase2_Pritamab.png"
plt.savefig(out, dpi=200, bbox_inches="tight",
            facecolor="white", edgecolor="none")
plt.close()
print(f"[OK] Saved → {out}")
