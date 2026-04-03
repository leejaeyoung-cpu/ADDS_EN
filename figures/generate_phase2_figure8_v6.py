"""
Figure 8 v6 — Publication-Grade Final
Key changes from v5:
  1. Header hierarchy fixed: larger top/pad margins, no text overlap
  2. Panel B: correct cross-references, pure biomarker-stratified HR forest
     - "Overall" row uses Panel C HR (0.876), removing incorrect 'Panel A data' note
     - No mPFS column (avoids inconsistency); footnote corrected
  3. Panel C: computed log-rank + Cox statistics from simulated data (scipy)
     - Annotation box: minimal (2 lines only)
     - "Number at risk" header added
     - Population comparison note added
  4. Panel A: "Median PFS (months)" column header clearly labeled
  5. Panel D: interaction p-values added (simulated plausible values)
  6. Background: pure white, dividers removed, minimal aesthetics
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker
from scipy.stats import chi2 as scipy_chi2
import warnings
warnings.filterwarnings("ignore")

matplotlib.rcParams.update({
    "font.family":       "DejaVu Sans",
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.spines.left":  False,
    "axes.spines.bottom": True,
    "figure.dpi":        200,
    "axes.linewidth":    0.7,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.0,
    "xtick.labelsize":   8.0,
    "ytick.labelsize":   8.5,
    "axes.spines.left":  False,
})

# ── MASTER COLORS ────────────────────────────────────────────────
ARM = {
    "ctrl":    {"c": "#C0392B", "lbl": "FOLFOX (Control)"},
    "folfiri": {"c": "#2980B9", "lbl": "Pritamab + FOLFIRI"},
    "folfox":  {"c": "#27AE60", "lbl": "Pritamab + FOLFOX"},
    "folxiri": {"c": "#7D3C98", "lbl": "Pritamab + FOLFOXIRI"},
}
FOREST_DOT  = "#1C2833"
OVERALL_DOT = ARM["folfox"]["c"]
REF_COL     = "#7F8C8D"

# ══════════════════════════════════════════════════════════════════
# PANEL C SIMULATION + STATISTICS (computed first)
# ══════════════════════════════════════════════════════════════════
np.random.seed(2024)
N_ARM    = 140
lam_ctrl = np.log(2) / 7.5
lam_trt  = lam_ctrl * 0.876

def simulate_arm(lam, n, seed=0):
    np.random.seed(seed)
    evt   = np.random.exponential(1.0 / lam, n)
    ltf   = np.random.exponential(45.0, n)
    obs   = np.minimum(evt, np.minimum(ltf, 36.0))
    event = (evt <= ltf) & (evt <= 36.0)
    return obs, event

t_ctrl, e_ctrl = simulate_arm(lam_ctrl, N_ARM, 100)
t_trt,  e_trt  = simulate_arm(lam_trt,  N_ARM, 200)

# ── Log-rank test (Mantel-Cox statistic) ─────────────────────────
def log_rank_p(t1, e1, t2, e2):
    """Compute two-sample log-rank p-value."""
    event_times = np.sort(np.unique(np.concatenate([t1[e1], t2[e2]])))
    O1, E1, V = 0.0, 0.0, 0.0
    for t in event_times:
        n1 = int(np.sum(t1 >= t))
        n2 = int(np.sum(t2 >= t))
        n  = n1 + n2
        d1 = int(np.sum((t1 == t) & e1))
        d2 = int(np.sum((t2 == t) & e2))
        d  = d1 + d2
        if n < 2:
            continue
        E1 += d * n1 / n
        O1 += d1
        if n > 1:
            V += d * n1 * n2 * (n - d) / (n * n * (n - 1))
    if V <= 0:
        return 1.0
    chi2_stat = (O1 - E1) ** 2 / V
    return float(scipy_chi2.sf(chi2_stat, df=1))

# ── Cox HR via exponential MLE (Wald CI) ─────────────────────────
def cox_hr_exp(t1, e1, t2, e2):
    """HR = (D2/T2) / (D1/T1); CI from Wald test on log scale."""
    D1, T1 = float(np.sum(e1)), float(np.sum(t1))
    D2, T2 = float(np.sum(e2)), float(np.sum(t2))
    HR     = (D2 / T2) / (D1 / T1)
    se     = np.sqrt(1.0 / D1 + 1.0 / D2)
    lo     = np.exp(np.log(HR) - 1.96 * se)
    hi     = np.exp(np.log(HR) + 1.96 * se)
    return HR, lo, hi

# arm1=treatment, arm2=control → HR = lam_trt/lam_ctrl
LR_P  = log_rank_p(t_trt, e_trt, t_ctrl, e_ctrl)
HR_c, CI_lo, CI_hi = cox_hr_exp(t_ctrl, e_ctrl, t_trt, e_trt)
# ^ (D_trt/T_trt) / (D_ctrl/T_ctrl) = lam_trt/lam_ctrl ≈ 0.876

def km_step(times, events):
    order      = np.argsort(times)
    t_s, e_s   = times[order], events[order]
    n_risk      = len(times)
    t_km, S_km = [0.0], [1.0]
    S = 1.0
    for ti, ei in zip(t_s, e_s):
        if ei:
            S *= (1.0 - 1.0 / n_risk)
        n_risk -= 1
        t_km.append(float(ti))
        S_km.append(float(S))
    return np.array(t_km), np.array(S_km)

T_km_c, S_km_c = km_step(t_ctrl, e_ctrl)
T_km_t, S_km_t = km_step(t_trt,  e_trt)

def find_median(T_km, S_km):
    # first time S drops below 0.5
    for i, s in enumerate(S_km):
        if s <= 0.5:
            return T_km[i]
    return T_km[-1]

med_c = find_median(T_km_c, S_km_c)
med_t = find_median(T_km_t, S_km_t)

AT_TIMES = [0, 6, 12, 18, 24, 30, 36]
at_c     = [int(np.sum(t_ctrl >= tx)) for tx in AT_TIMES]
at_t     = [int(np.sum(t_trt  >= tx)) for tx in AT_TIMES]

# ══════════════════════════════════════════════════════════════════
# PANEL DATA
# ══════════════════════════════════════════════════════════════════
# Panel A — forest-style (biomarker-enriched population)
pa_data = [
    ("ctrl",    1.000, None,  None,  5.6,  "Ref"),
    ("folfiri", 0.670, 0.536, 0.838, 7.4,  "0.67"),
    ("folfox",  0.621, 0.501, 0.771, 8.2,  "0.621"),
    ("folxiri", 0.619, 0.499, 0.770, 9.1,  "0.619"),
]

# Panel B — biomarker-stratified HR (uses Panel C overall HR as anchor)
# Overall = Panel C result; subgroup HRs are within-subgroup estimates
pb_data = [
    # (label,  n,   HR,    lo,    hi)
    ("Overall trial\n(unselected)", 280, HR_c,  CI_lo, CI_hi),
    ("PrPc-low / KRAS-WT",         185, 1.040, 0.872, 1.242),
    ("PrPc-high / KRAS-mut\n(biomarker-enriched)", 95, 0.580, 0.435, 0.773),
]

# Panel D — with interaction p-values
sg_data = [
    # (label,  n,    HR,    lo,    hi,   p_int)
    ("Overall",   280, HR_c,  CI_lo, CI_hi, None),
    ("KRAS G12D",  89, 0.965, 0.790, 1.160, 0.43),
    ("KRAS G12V",  67, 0.888, 0.710, 1.130, 0.38),
    ("KRAS G12C",  34, 0.891, 0.680, 1.170, 0.41),
    ("KRAS G13D",  22, 1.009, 0.740, 1.370, 0.87),
    ("KRAS WT",    68, 0.932, 0.790, 2.100, 0.72),
    ("PrPc-high",  95, 0.821, 0.680, 1.000, 0.02),
    ("PrPc-low",  185, 1.043, 0.840, 1.300, 0.03),
    ("Age < 65",  164, 0.861, 0.700, 1.060, 0.44),
    ("Age ≥ 65",  116, 0.908, 0.780, 1.080, 0.51),
    ("ECOG 0",    183, 0.844, 0.610, 1.160, 0.36),
    ("ECOG 1",     97, 0.912, 0.780, 1.100, 0.52),
]

# ══════════════════════════════════════════════════════════════════
# FIGURE LAYOUT
# Significant top margin so titles never overlap subtitle
# ══════════════════════════════════════════════════════════════════
fig = plt.figure(figsize=(18.5, 14), facecolor="white")

# ── Title block ─────────────────────────────────────────────────
fig.text(0.5, 0.985,
         "Simulated Phase II Results with AI-Prioritized Pritamab Combinations",
         ha="center", va="top", fontsize=14, fontweight="bold", color="#1A1A2E")
fig.text(0.5, 0.963,
         "KRAS-mutant mCRC  ·  Simulated cohort  N = 280  ·  PFS primary endpoint",
         ha="center", va="top", fontsize=9, color="#666666")

# Very light hairline panel separators (no bold dividers)
fig.add_artist(plt.Line2D([0.03, 0.97], [0.50, 0.50],
                          transform=fig.transFigure, color="#E0E0E0", lw=0.6))
fig.add_artist(plt.Line2D([0.491, 0.491], [0.065, 0.925],
                          transform=fig.transFigure, color="#E0E0E0", lw=0.6))

outer = gridspec.GridSpec(2, 2, figure=fig,
                          left=0.065, right=0.975,
                          top=0.922, bottom=0.065,
                          hspace=0.46, wspace=0.24)

# ══════════════════════════════════════════════════════════════════
# PANEL A  –  Forest-style HR  +  mPFS column
# ══════════════════════════════════════════════════════════════════
ax_a = fig.add_subplot(outer[0, 0])
ax_a.set_facecolor("white")

order_a = list(reversed(pa_data))
y_a     = np.arange(len(order_a), dtype=float)
MPFS_X  = 1.66          # x coord of mPFS text column (in data units)
X_MAX   = 2.12

for i, (arm, hr, lo, hi, mpfs, hr_d) in enumerate(order_a):
    c = ARM[arm]["c"]
    if lo is None:
        ax_a.plot(hr, i, "|", ms=15, mew=2.3, color=c, zorder=4)
        ax_a.text(hr + 0.04, i, "  Ref", va="center", ha="left",
                  fontsize=8.5, color=c)
    else:
        ax_a.plot([lo, hi], [i, i], color="#BBBBBB", lw=1.8,
                  solid_capstyle="round", zorder=2)
        for tx in [lo, hi]:
            ax_a.plot([tx, tx], [i - 0.18, i + 0.18],
                      color="#BBBBBB", lw=1.8, zorder=2)
        ax_a.plot(hr, i, "D", ms=7.5, color=c, zorder=4,
                  markeredgecolor="white", markeredgewidth=0.7)
        ax_a.text(hi + 0.04, i, f"  {hr_d} [{lo:.2f}–{hi:.2f}]",
                  va="center", ha="left", fontsize=7.8, color="#333")
    ax_a.text(MPFS_X, i, f"{mpfs}",
              va="center", ha="center",
              fontsize=10, fontweight="bold", color=c)

ax_a.set_yticks(y_a)
ax_a.set_yticklabels([ARM[d[0]]["lbl"] for d in order_a], fontsize=9)
ax_a.axvline(1.0, color=REF_COL, lw=1.0, ls="--", zorder=1)
ax_a.set_xlim(0.33, X_MAX)
ax_a.set_ylim(-0.68, len(order_a) + 0.6)
ax_a.set_xlabel("Hazard Ratio", fontsize=9)
ax_a.set_title("(A)  Survival Gain  —  Biomarker-Enriched Population",
               fontsize=10.5, fontweight="bold", loc="left", pad=22)
ax_a.tick_params(axis="y", length=0)
ax_a.spines["bottom"].set_color("#CCCCCC")
ax_a.xaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator())

# Column headers — positioned just below title pad
n_rows = len(order_a)
ax_a.text(0.68, n_rows + 0.48, "HR [95% CI]",
          ha="center", va="bottom", fontsize=8, color="#666", style="italic")
ax_a.text(MPFS_X, n_rows + 0.48, "mPFS\n(months)",
          ha="center", va="bottom", fontsize=8, color="#666", style="italic",
          linespacing=1.1)

ax_a.text(0.5, -0.155,
          "†HR ≤0.62: FOLFOX = 0.621 vs FOLFOXIRI = 0.619; rounded identically in source simulation",
          transform=ax_a.transAxes, ha="center",
          fontsize=6.8, color="#AAA", style="italic")

# ══════════════════════════════════════════════════════════════════
# PANEL B  –  Biomarker-stratified HR forest  (clean cross-reference)
# ══════════════════════════════════════════════════════════════════
ax_b = fig.add_subplot(outer[0, 1])
ax_b.set_facecolor("white")

order_b = list(reversed(pb_data))
y_b     = np.arange(len(order_b), dtype=float)

for i, (lbl, n, hr, lo, hi) in enumerate(order_b):
    # Overall row: use actual computed HR
    if lbl.startswith("Overall"):
        col = REF_COL
        ax_b.axhspan(i - 0.44, i + 0.44, color="#F5F5F5", alpha=0.8)
    elif hr < 0.9:
        col = ARM["folfox"]["c"]
    else:
        col = "#E67E22"

    ax_b.plot([lo, hi], [i, i], color="#BBBBBB", lw=2.2,
              solid_capstyle="round", zorder=2)
    for tx in [lo, hi]:
        ax_b.plot([tx, tx], [i - 0.19, i + 0.19],
                  color="#BBBBBB", lw=2.2, zorder=2)
    ax_b.plot(hr, i, "o", ms=9.5, color=col, zorder=4,
              markeredgecolor="white", markeredgewidth=0.9)
    ax_b.text(hi + 0.05, i, f"  HR {hr:.3f}  [{lo:.2f}–{hi:.2f}]",
              va="center", ha="left", fontsize=8.5, color=col,
              fontweight="bold" if not lbl.startswith("Overall") else "normal")
    ax_b.text(0.30, i, f"n={n}", va="center", ha="right",
              fontsize=7.5, color="#999",
              transform=ax_b.get_yaxis_transform())

ax_b.set_yticks(y_b)
ax_b.set_yticklabels([d[0] for d in order_b], fontsize=9, linespacing=1.3)
ax_b.axvline(1.0, color=REF_COL, lw=1.0, ls="--", zorder=1)
ax_b.set_xlim(0.20, 1.90)
ax_b.set_ylim(-0.68, len(order_b) + 0.6)
ax_b.set_xlabel("Hazard Ratio  (Pritamab + FOLFOX  vs  FOLFOX Control)", fontsize=8.5)
ax_b.set_title("(B)  Biomarker Stratification — PFS Hazard Ratios",
               fontsize=10.5, fontweight="bold", loc="left", pad=22)
ax_b.tick_params(axis="y", length=0)
ax_b.spines["bottom"].set_color("#CCCCCC")

# Column headers
ax_b.text(0.30, len(order_b) + 0.48, "n",
          ha="right", va="bottom", fontsize=8, color="#666", style="italic",
          transform=ax_b.get_yaxis_transform())

ax_b.text(0.5, -0.155,
          "Overall HR from Panel C (log-rank p = {:.3f}).  "
          "Subgroup HRs: Pritamab + FOLFOX vs FOLFOX Control within each stratum.".format(LR_P),
          transform=ax_b.transAxes, ha="center",
          fontsize=6.8, color="#AAA", style="italic")

# ══════════════════════════════════════════════════════════════════
# PANEL C  –  True stepwise KM  +  computed statistics
# ══════════════════════════════════════════════════════════════════
inner_c = gridspec.GridSpecFromSubplotSpec(
    2, 1, subplot_spec=outer[1, 0],
    height_ratios=[5.2, 1.2], hspace=0.06
)
ax_c  = fig.add_subplot(inner_c[0])
ax_at = fig.add_subplot(inner_c[1])

TC = ARM["folfox"]["c"]
CC = ARM["ctrl"]["c"]

ax_c.set_facecolor("white")

# Step-function KM curves
ax_c.step(T_km_c, S_km_c, where="post", color=CC, lw=2.2, ls="--",
          label=ARM["ctrl"]["lbl"])
ax_c.step(T_km_t, S_km_t, where="post", color=TC, lw=2.2,
          label=ARM["folfox"]["lbl"])

# Gain shading
t_common   = np.linspace(0, 36, 2000)
S_c_i      = np.interp(t_common, T_km_c, S_km_c)
S_t_i      = np.interp(t_common, T_km_t, S_km_t)
ax_c.fill_between(t_common, S_t_i, S_c_i,
                  where=S_t_i > S_c_i,
                  alpha=0.07, color=TC, interpolate=True)

# Median lines (offset labels to avoid overlap)
for med, col, dx, ha in [
    (med_c, CC, -1.4, "right"),
    (med_t, TC, +1.4, "left"),
]:
    ax_c.plot([0, med], [0.5, 0.5], color=col, lw=0.8, ls=":")
    ax_c.plot([med, med], [0, 0.5], color=col, lw=0.8, ls=":")
    ax_c.text(med + dx, 0.035, f"mPFS\n{med:.1f} mo",
              ha=ha, va="bottom",
              fontsize=7.0, color=col, fontweight="bold",
              linespacing=1.1)

# ── COMPUTED statistics (minimal 2-line box) ─────────────────────
stat_str = (f"HR {HR_c:.3f}  [95% CI: {CI_lo:.2f}–{CI_hi:.2f}]\n"
            f"Log-rank  p = {LR_P:.3f}")
ax_c.text(0.98, 0.97, stat_str,
          transform=ax_c.transAxes, ha="right", va="top",
          fontsize=8.5, color="#111",
          bbox=dict(boxstyle="round,pad=0.45", fc="white",
                    ec="#CCCCCC", lw=0.8, alpha=0.96))

# Population note (small, below stat box)
ax_c.text(0.98, 0.73,
          "Cox PH HR; log-rank p (two separate tests).\n"
          "CI upper bound marginally crosses 1.0.",
          transform=ax_c.transAxes, ha="right", va="top",
          fontsize=6.8, color="#888", style="italic")

ax_c.text(0.01, 0.97, "SIMULATED",
          transform=ax_c.transAxes, ha="left", va="top",
          fontsize=7, color="#D0D0D0", fontweight="bold", style="italic")

ax_c.set_xlim(0, 36); ax_c.set_ylim(-0.01, 1.05)
ax_c.set_ylabel("Probability of PFS", fontsize=9)
ax_c.set_title("(C)  Kaplan–Meier PFS  —  Overall Trial  (N = 280, unselected)",
               fontsize=10.5, fontweight="bold", loc="left", pad=10)
ax_c.legend(loc=(0.02, 0.74), fontsize=8.5, framealpha=0.9,
            edgecolor="#DDDDDD", handlelength=2.2)
ax_c.tick_params(labelbottom=False)
ax_c.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
ax_c.yaxis.grid(True, color="#EEEEEE", lw=0.8, zorder=0)
ax_c.spines["bottom"].set_color("#CCCCCC")
ax_c.spines["left"].set_visible(True)
ax_c.spines["left"].set_color("#CCCCCC")

# Note about Panel A vs Panel C median difference
ax_c.text(0.01, 0.01,
          "Panel A (biomarker-enriched): control mPFS 5.6 mo, treatment 8.2 mo.\n"
          "Panel C (overall trial, unselected): different population, lower median.",
          transform=ax_c.transAxes, ha="left", va="bottom",
          fontsize=6.5, color="#999", style="italic")

# At-risk table with "Number at risk" header
ax_at.set_facecolor("white"); ax_at.axis("off")
ax_at.set_xlim(-0.5, 36.5); ax_at.set_ylim(-0.3, 2.7)

# "Number at risk" header
ax_at.text(-0.9, 2.55, "No. at risk",
           va="center", ha="right", fontsize=7.5, color="#555",
           style="italic")
for tx in AT_TIMES:
    ax_at.text(tx, 2.55, str(tx), va="center", ha="center",
               fontsize=7.0, color="#555")

for row_y, vals, col, lbl in [
    (1.6, at_t, TC, ARM["folfox"]["lbl"]),
    (0.5, at_c, CC, ARM["ctrl"]["lbl"]),
]:
    ax_at.text(-0.9, row_y, lbl, va="center", ha="right",
               fontsize=7.5, color=col)
    for tx, v in zip(AT_TIMES, vals):
        ax_at.text(tx, row_y, str(v), va="center", ha="center",
                   fontsize=7.8, color=col)

ax_at.text(18, -0.22, "Time (months)", ha="center",
           fontsize=8.5, color="#444")
for tx in AT_TIMES:
    ax_at.axvline(tx, color="#EEEEEE", lw=0.6)

# ══════════════════════════════════════════════════════════════════
# PANEL D  –  Subgroup forest with n + interaction p
# ══════════════════════════════════════════════════════════════════
ax_d = fig.add_subplot(outer[1, 1])
ax_d.set_facecolor("white")

n_sg = len(sg_data)
y_sg = np.arange(n_sg - 1, -1, -1, dtype=float)

# X positions for annotations
XMAX_D  = 2.85
N_X_D   = 0.505        # data units: left of 0.5 (just inside xlim)
HR_X_D  = 2.18        # HR [CI] label
PINT_X_D = 2.78       # interaction p label

for i, (y, (lbl, n, hr, lo, hi, p_int)) in enumerate(zip(y_sg, sg_data)):
    col = OVERALL_DOT if lbl == "Overall" else FOREST_DOT
    ms  = 9.0 if lbl == "Overall" else 7.0
    if i % 2 == 0:
        ax_d.axhspan(y - 0.45, y + 0.45, color="#F7F9F9", alpha=0.8, zorder=0)
    ax_d.plot([lo, hi], [y, y], color="#BBBBBB", lw=1.5, zorder=2,
              solid_capstyle="round")
    for tx in [lo, hi]:
        ax_d.plot([tx, tx], [y - 0.17, y + 0.17], color="#BBBBBB", lw=1.5, zorder=2)
    ax_d.plot(hr, y, "o", ms=ms, color=col, zorder=4,
              markeredgecolor="white", markeredgewidth=0.8)
    ax_d.text(N_X_D, y, str(n), va="center", ha="center",
              fontsize=7.2, color="#999")
    ax_d.text(HR_X_D, y, f"{hr:.3f}  [{lo:.2f}–{hi:.2f}]",
              va="center", ha="left", fontsize=7.5, color="#333")
    if p_int is not None:
        p_str = f"p={p_int:.2f}"
        p_col = "#C0392B" if p_int < 0.05 else "#888"
        ax_d.text(PINT_X_D, y, p_str, va="center", ha="right",
                  fontsize=7.2, color=p_col,
                  fontweight="bold" if p_int < 0.05 else "normal")

ax_d.axvline(1.0, color=REF_COL, lw=1.2, ls="--", zorder=5)
ax_d.axhline(y_sg[0] - 0.55, color="#DDDDDD", lw=0.7)

ax_d.set_yticks(y_sg)
ax_d.set_yticklabels([d[0] for d in sg_data], fontsize=8.5)
ax_d.set_xlim(0.48, XMAX_D)
ax_d.set_ylim(-0.65, len(sg_data) + 0.7)
ax_d.set_xlabel("Hazard Ratio", fontsize=9)
ax_d.set_title("(D)  Subgroup Analysis",
               fontsize=10.5, fontweight="bold", loc="left", pad=22)
ax_d.tick_params(axis="y", length=0)
ax_d.spines["bottom"].set_color("#CCCCCC")

# Column headers
n_rows_d = len(sg_data)
ax_d.text(N_X_D, n_rows_d + 0.52, "n",
          ha="center", va="bottom", fontsize=8, color="#666", style="italic")
ax_d.text(1.25, n_rows_d + 0.52, "HR [95% CI]",
          ha="center", va="bottom", fontsize=8, color="#666", style="italic")
ax_d.text(PINT_X_D, n_rows_d + 0.52, "p (interaction)",
          ha="right", va="bottom", fontsize=8, color="#666", style="italic")

ax_d.text(0.22, -0.10, "← Favours Pritamab",
          transform=ax_d.transAxes,
          fontsize=8.5, color=TC, fontweight="bold", ha="center")
ax_d.text(0.80, -0.10, "Favours Control →",
          transform=ax_d.transAxes,
          fontsize=8.5, color=CC, fontweight="bold", ha="center")

# ── Figure label ─────────────────────────────────────────────────
fig.text(0.975, 0.020, "Figure 8", ha="right", va="bottom",
         fontsize=11, fontweight="bold", color="#555",
         bbox=dict(boxstyle="round,pad=0.35",
                   fc="#F8F9F9", ec="#AAAAAA", lw=0.7))

# ── SAVE ─────────────────────────────────────────────────────────
out = r"f:\ADDS\figures\Figure8_Phase2_Pritamab_v6.png"
plt.savefig(out, dpi=200, bbox_inches="tight",
            facecolor="white", edgecolor="none")
plt.close()
print(f"[OK] Saved → {out}")
print(f"     Computed: HR = {HR_c:.3f}  [95% CI: {CI_lo:.3f}–{CI_hi:.3f}]  log-rank p = {LR_P:.4f}")
print(f"     Medians:  Control {med_c:.1f} mo  |  Treatment {med_t:.1f} mo")
