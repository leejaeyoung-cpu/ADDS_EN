"""
Figure 8 v13 — Readable Layout
Key changes from v12:
  - Canvas 21 × 16 in (wider, taller)
  - hspace / wspace increased for breathing room
  - Fonts bumped: titles 12pt, body 9.5pt, annotations 9pt
  - Panel C KM area taller (height ratio 6.8 : 1.3)
  - More pad on all panel titles
  - Risk table row spacing improved
  - All column headers bolder / larger
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
    "axes.linewidth":    0.8,
    "xtick.major.width": 0.7,
    "ytick.major.width": 0.0,
    "xtick.labelsize":   9.0,
    "ytick.labelsize":   9.5,
})

# ── MASTER COLORS ─────────────────────────────────────────────────
ARM = {
    "ctrl":    {"c": "#C0392B", "lbl": "FOLFOX (Control)"},
    "folfiri": {"c": "#2980B9", "lbl": "Pritamab + FOLFIRI"},
    "folfox":  {"c": "#27AE60", "lbl": "Pritamab + FOLFOX"},
    "folxiri": {"c": "#7D3C98", "lbl": "Pritamab + FOLFOXIRI"},
}
FOREST_DOT  = "#1C2833"
OVERALL_DOT = "#555555"   # neutral gray: CI crosses 1
REF_COL     = "#7F8C8D"
GRP_COL     = "#444444"

def fmt(x):
    return f"{x:.2f}"

# ══════════════════════════════════════════════════════════════════
# STATISTICS
# ══════════════════════════════════════════════════════════════════
np.random.seed(2024)
N_ARM    = 140
lam_ctrl = np.log(2) / 7.5
lam_trt  = lam_ctrl * 0.876

def sim_arm(lam, n, seed=0):
    np.random.seed(seed)
    evt = np.random.exponential(1.0 / lam, n)
    ltf = np.random.exponential(45.0, n)
    obs = np.minimum(evt, np.minimum(ltf, 36.0))
    ev  = (evt <= ltf) & (evt <= 36.0)
    return obs, ev

t_ctrl, e_ctrl = sim_arm(lam_ctrl, N_ARM, 100)
t_trt,  e_trt  = sim_arm(lam_trt,  N_ARM, 200)

def log_rank_p(t1, e1, t2, e2):
    event_times = np.sort(np.unique(np.concatenate([t1[e1], t2[e2]])))
    O1 = E1 = V = 0.0
    for t in event_times:
        n1 = int(np.sum(t1 >= t)); n2 = int(np.sum(t2 >= t)); n = n1 + n2
        d1 = int(np.sum((t1 == t) & e1)); d2 = int(np.sum((t2 == t) & e2)); d = d1 + d2
        if n < 2: continue
        E1 += d * n1 / n; O1 += d1
        if n > 1: V += d * n1 * n2 * (n - d) / (n * n * (n - 1))
    return float(scipy_chi2.sf((O1 - E1) ** 2 / V, df=1)) if V > 0 else 1.0

def cox_hr_exp(t1, e1, t2, e2):
    D1, T1 = float(np.sum(e1)), float(np.sum(t1))
    D2, T2 = float(np.sum(e2)), float(np.sum(t2))
    HR = (D2 / T2) / (D1 / T1)
    se = np.sqrt(1.0 / D1 + 1.0 / D2)
    return HR, np.exp(np.log(HR) - 1.96 * se), np.exp(np.log(HR) + 1.96 * se)

LR_P             = log_rank_p(t_trt, e_trt, t_ctrl, e_ctrl)
HR_c, CI_lo, CI_hi = cox_hr_exp(t_ctrl, e_ctrl, t_trt, e_trt)

# PrPc interaction p via Wald test
prop_high = 95 / 280
n_c_high  = round(140 * prop_high)
n_c_low   = 140 - n_c_high
t_ch, e_ch = sim_arm(lam_ctrl,            n_c_high, 300)
t_th, e_th = sim_arm(lam_ctrl * 0.82,     n_c_high, 301)
t_cl, e_cl = sim_arm(lam_ctrl * 0.95,     n_c_low,  302)
t_tl, e_tl = sim_arm(lam_ctrl * 0.95 * 1.04, n_c_low, 303)
HR_h, _, _ = cox_hr_exp(t_ch, e_ch, t_th, e_th)
HR_l, _, _ = cox_hr_exp(t_cl, e_cl, t_tl, e_tl)
D_ch = float(np.sum(e_ch)); D_th = float(np.sum(e_th))
D_cl = float(np.sum(e_cl)); D_tl = float(np.sum(e_tl))
log_int = np.log(max(HR_h, 1e-9)) - np.log(max(HR_l, 1e-9))
se_int  = np.sqrt(1/max(D_ch,1) + 1/max(D_th,1) + 1/max(D_cl,1) + 1/max(D_tl,1))
from scipy.stats import norm as scipy_norm
P_INT_PRPC = float(2 * (1 - scipy_norm.cdf(abs(log_int / se_int))))

def km_step(times, events):
    order = np.argsort(times)
    t_s, e_s = times[order], events[order]
    n_risk = len(times)
    t_km, S_km = [0.0], [1.0]; S = 1.0
    for ti, ei in zip(t_s, e_s):
        if ei: S *= (1.0 - 1.0 / n_risk)
        n_risk -= 1
        t_km.append(float(ti)); S_km.append(float(S))
    return np.array(t_km), np.array(S_km)

T_km_c, S_km_c = km_step(t_ctrl, e_ctrl)
T_km_t, S_km_t = km_step(t_trt,  e_trt)

def find_median(T, S):
    for i, s in enumerate(S):
        if s <= 0.5: return T[i]
    return T[-1]

med_c = find_median(T_km_c, S_km_c)
med_t = find_median(T_km_t, S_km_t)

AT_TIMES = [0, 6, 12, 18, 24, 30, 36]
at_c = [int(np.sum(t_ctrl >= tx)) for tx in AT_TIMES]
at_t = [int(np.sum(t_trt  >= tx)) for tx in AT_TIMES]

# ══════════════════════════════════════════════════════════════════
# PANEL DATA
# ══════════════════════════════════════════════════════════════════
pa_data = [  # (arm, HR, lo, hi, mPFS, hr_lbl, ci_lo_lbl, ci_hi_lbl)
    ("ctrl",    1.000, None,  None,  5.6, "Ref",   None,    None),
    ("folfiri", 0.670, 0.540, 0.840, 7.4, "0.670", "0.540", "0.840"),
    ("folfox",  0.621, 0.501, 0.771, 8.2, "0.621", "0.501", "0.771"),
    ("folxiri", 0.619, 0.498, 0.769, 9.1, "0.619", "0.498", "0.769"),
]

pb_data = [  # (label, n, HR, lo, hi)
    ("All patients\n(overall trial, N=280)",              280, HR_c, CI_lo, CI_hi),
    ("PrPc-low / KRAS-WT",                                185, 1.04, 0.87,  1.24),
    ("PrPc-high / KRAS-mut\n(biomarker-enriched)",         95, 0.58, 0.44,  0.77),
]

sg_groups = [
    ("", None, [("Overall", 280, HR_c, CI_lo, CI_hi)]),
    ("KRAS subtype", 0.54, [
        ("KRAS G12D",  89, 0.97, 0.79, 1.16),
        ("KRAS G12V",  67, 0.89, 0.71, 1.13),
        ("KRAS G12C",  34, 0.89, 0.68, 1.17),
        ("KRAS G13D",  22, 1.01, 0.74, 1.37),
        ("KRAS WT",    68, 0.93, 0.79, 2.10),
    ]),
    ("PrPc expression", None, [
        ("PrPc-high",  95, 0.82, 0.68, 1.00),
        ("PrPc-low",  185, 1.04, 0.84, 1.30),
    ]),
    ("Age", 0.48, [
        ("< 65 years",  164, 0.86, 0.70, 1.06),
        (">= 65 years", 116, 0.91, 0.78, 1.08),
    ]),
    ("ECOG PS", 0.49, [
        ("ECOG 0", 183, 0.84, 0.61, 1.16),
        ("ECOG 1",  97, 0.91, 0.78, 1.10),
    ]),
]
sg_groups = [(gl, (P_INT_PRPC if gl == "PrPc expression" else pi), rows)
             for gl, pi, rows in sg_groups]

flat_rows = []
for grp_lbl, p_int, rows in sg_groups:
    if grp_lbl:
        flat_rows.append(("header", grp_lbl, None, None, None, None, p_int))
    for row in rows:
        flat_rows.append(("overall" if row[0] == "Overall" else "data",) + row + (None,))

y_positions = []
y = len(flat_rows) - 1
for row in flat_rows:
    y_positions.append(float(y))
    y -= 0.65 if row[0] == "header" else 1.0

# ══════════════════════════════════════════════════════════════════
# FIGURE — wider, taller, more room
# ══════════════════════════════════════════════════════════════════
fig = plt.figure(figsize=(21, 16), facecolor="white")

fig.text(0.5, 0.984,
         "Simulated Phase II Results with AI-Prioritized Pritamab Combinations",
         ha="center", va="top", fontsize=15, fontweight="bold", color="#1A1A2E")
fig.text(0.5, 0.964,
         "mCRC simulated cohort (KRAS-mutant enriched)  \u00b7  N = 280  \u00b7  PFS primary endpoint",
         ha="center", va="top", fontsize=10, color="#666666")

outer = gridspec.GridSpec(2, 2, figure=fig,
                          left=0.07, right=0.975,
                          top=0.926, bottom=0.06,
                          hspace=0.52, wspace=0.28)

# ══════════════════════════════════════════════════════════════════
# PANEL A
# ══════════════════════════════════════════════════════════════════
ax_a = fig.add_subplot(outer[0, 0])
ax_a.set_facecolor("white")

order_a = list(reversed(pa_data))
y_a     = np.arange(len(order_a), dtype=float)
MPFS_X  = 1.72
X_MAX_A = 2.20

for i, (arm, hr, lo, hi, mpfs, hr_lbl, ci_lo_lbl, ci_hi_lbl) in enumerate(order_a):
    c = ARM[arm]["c"]
    if lo is None:
        ax_a.plot(hr, i, "|", ms=18, mew=2.5, color=c, zorder=4)
        ax_a.text(hr + 0.04, i, "  Ref", va="center", ha="left", fontsize=10, color=c)
    else:
        ax_a.plot([lo, hi], [i, i], color="#CCCCCC", lw=2.0, solid_capstyle="round", zorder=2)
        for tx in [lo, hi]:
            ax_a.plot([tx, tx], [i - 0.18, i + 0.18], color="#CCCCCC", lw=2.0, zorder=2)
        ax_a.plot(hr, i, "D", ms=9, color=c, zorder=4,
                  markeredgecolor="white", markeredgewidth=0.8)
        ax_a.text(hi + 0.04, i, f"  {hr_lbl} [{ci_lo_lbl}\u2013{ci_hi_lbl}]",
                  va="center", ha="left", fontsize=9, color="#333")
    ax_a.text(MPFS_X, i, f"{mpfs}",
              va="center", ha="right", fontsize=11, fontweight="bold", color=c)

ax_a.set_yticks(y_a)
ax_a.set_yticklabels([ARM[d[0]]["lbl"] for d in order_a], fontsize=10)
ax_a.axvline(1.0, color=REF_COL, lw=1.0, ls="--", zorder=1)
ax_a.set_xlim(0.30, X_MAX_A)
ax_a.set_ylim(-0.8, len(order_a) + 0.95)
ax_a.set_xlabel("Hazard Ratio", fontsize=10)
ax_a.set_title("(A)  Survival Gain  \u2014  Biomarker-Enriched Population  (n\u202f=\u202f95)",
               fontsize=12, fontweight="bold", loc="left", pad=26)
ax_a.tick_params(axis="y", length=0)
ax_a.spines["bottom"].set_color("#CCCCCC")
ax_a.xaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator())

n_rows_a = len(order_a)
HY = n_rows_a + 0.72
ax_a.text(0.68, HY, "HR [95% CI]",
          ha="center", va="bottom", fontsize=9.5, color="#444", style="italic")
ax_a.text(MPFS_X, HY, "mPFS (mo)",
          ha="right",  va="bottom", fontsize=9.5, color="#444", style="italic")
ax_a.axhline(n_rows_a + 0.28, color="#DDDDDD", lw=0.6)

# ══════════════════════════════════════════════════════════════════
# PANEL B
# ══════════════════════════════════════════════════════════════════
ax_b = fig.add_subplot(outer[0, 1])
ax_b.set_facecolor("white")

order_b = list(reversed(pb_data))
y_b     = np.arange(len(order_b), dtype=float)
n_rows_b = len(order_b)

for i, (lbl, n, hr, lo, hi) in enumerate(order_b):
    is_overall = lbl.startswith("All patients")
    if is_overall:
        col = OVERALL_DOT
        ax_b.axhspan(i - 0.46, i + 0.46, color="#F2F2F2", alpha=0.95, zorder=0)
    elif hr < 0.9:
        col = ARM["folfox"]["c"]
    else:
        col = "#E67E22"

    ax_b.plot([lo, hi], [i, i], color="#CCCCCC", lw=2.4,
              solid_capstyle="round", zorder=2)
    for tx in [lo, hi]:
        ax_b.plot([tx, tx], [i - 0.20, i + 0.20], color="#CCCCCC", lw=2.4, zorder=2)
    ax_b.plot(hr, i, "o", ms=12, color=col, zorder=4,
              markeredgecolor="white", markeredgewidth=1.0)
    ax_b.text(hi + 0.05, i,
              f"  HR {fmt(hr)}  [{fmt(lo)}\u2013{fmt(hi)}]",
              va="center", ha="left", fontsize=9.5, color=col,
              fontweight="bold" if not is_overall else "normal")
    if not is_overall:
        ax_b.text(0.27, i, f"n={n}", va="center", ha="right",
                  fontsize=8.5, color="#999",
                  transform=ax_b.get_yaxis_transform())

# Separator between overall and strata
ax_b.axhline(n_rows_b - 1 - 0.54, color="#999999", lw=1.1, ls="-", alpha=0.8)

ax_b.set_yticks(y_b)
ax_b.set_yticklabels([d[0] for d in order_b], fontsize=10, linespacing=1.4)
ax_b.axvline(1.0, color=REF_COL, lw=1.0, ls="--", zorder=1)
ax_b.set_xlim(0.18, 1.95)
ax_b.set_ylim(-0.8, n_rows_b + 0.95)
ax_b.set_xlabel("Hazard Ratio  (Pritamab + FOLFOX  vs  FOLFOX Control)", fontsize=9.5)
ax_b.set_title("(B)  Pre-specified Composite Biomarker Strata",
               fontsize=12, fontweight="bold", loc="left", pad=26)
ax_b.tick_params(axis="y", length=0)
ax_b.spines["bottom"].set_color("#CCCCCC")

HY_B = n_rows_b + 0.72
ax_b.text(0.27, HY_B, "n",
          ha="right", va="bottom", fontsize=9.5, color="#444", style="italic",
          transform=ax_b.get_yaxis_transform())
ax_b.axhline(n_rows_b + 0.28, color="#DDDDDD", lw=0.6)

# ══════════════════════════════════════════════════════════════════
# PANEL C — KM
# ══════════════════════════════════════════════════════════════════
inner_c = gridspec.GridSpecFromSubplotSpec(
    2, 1, subplot_spec=outer[1, 0],
    height_ratios=[6.8, 1.3], hspace=0.06
)
ax_c  = fig.add_subplot(inner_c[0])
ax_at = fig.add_subplot(inner_c[1])

TC = ARM["folfox"]["c"]
CC = ARM["ctrl"]["c"]
ax_c.set_facecolor("white")

ax_c.step(T_km_c, S_km_c, where="post", color=CC, lw=2.4, ls="--",
          label=ARM["ctrl"]["lbl"])
ax_c.step(T_km_t, S_km_t, where="post", color=TC, lw=2.4,
          label=ARM["folfox"]["lbl"])

# Censor ticks
def add_censor_ticks(ax, T_km, S_km, t_obs, e_obs, color):
    cens_t = t_obs[(~e_obs) & (t_obs < 36.0)]
    for ct in cens_t:
        sv = np.interp(ct, T_km, S_km)
        ax.plot(ct, sv, "|", ms=7.5, mew=1.5, color=color, alpha=0.65, zorder=5)

add_censor_ticks(ax_c, T_km_c, S_km_c, t_ctrl, e_ctrl, CC)
add_censor_ticks(ax_c, T_km_t, S_km_t, t_trt,  e_trt,  TC)

t_common = np.linspace(0, 36, 2000)
S_c_i = np.interp(t_common, T_km_c, S_km_c)
S_t_i = np.interp(t_common, T_km_t, S_km_t)
ax_c.fill_between(t_common, S_t_i, S_c_i, where=S_t_i > S_c_i,
                  alpha=0.07, color=TC, interpolate=True)

for med, col, dx, ha in [(med_c, CC, -1.8, "right"), (med_t, TC, +1.8, "left")]:
    ax_c.plot([0, med], [0.5, 0.5], color=col, lw=0.9, ls=":")
    ax_c.plot([med, med], [0, 0.5], color=col, lw=0.9, ls=":")
    ax_c.text(med + dx, 0.05, f"mPFS = {med:.1f} mo",
              ha=ha, va="bottom", fontsize=8.5, color=col, fontweight="bold")

stat_str = (f"HR {fmt(HR_c)}  [95% CI: {fmt(CI_lo)}\u2013{fmt(CI_hi)}]\n"
            f"Log-rank  p = {LR_P:.3f}")
ax_c.text(0.98, 0.97, stat_str,
          transform=ax_c.transAxes, ha="right", va="top",
          fontsize=9.5, color="#111",
          bbox=dict(boxstyle="round,pad=0.5", fc="white", ec="#CCCCCC", lw=0.8, alpha=0.97))

ax_c.text(0.02, 0.97, "Simulated data",
          transform=ax_c.transAxes, ha="left", va="top",
          fontsize=8, color="#BBBBBB", style="italic")

ax_c.set_xlim(0, 36); ax_c.set_ylim(-0.01, 1.05)
ax_c.set_ylabel("Probability of PFS", fontsize=10)
ax_c.set_title("(C)  Kaplan\u2013Meier PFS  \u2014  Overall Trial  (N = 280, unselected)",
               fontsize=12, fontweight="bold", loc="left", pad=12)
ax_c.legend(loc=(0.02, 0.76), fontsize=9.5, framealpha=0.9,
            edgecolor="#DDDDDD", handlelength=2.6)
ax_c.tick_params(labelbottom=False)
ax_c.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
ax_c.yaxis.grid(True, color="#EEEEEE", lw=0.8, zorder=0)
ax_c.spines["bottom"].set_color("#CCCCCC")
ax_c.spines["left"].set_visible(True); ax_c.spines["left"].set_color("#CCCCCC")

# At-risk table
ax_at.set_facecolor("white"); ax_at.axis("off")
ax_at.set_xlim(-0.5, 36.5); ax_at.set_ylim(-0.4, 3.3)

ax_at.text(-0.9, 3.05, "No. at risk",
           va="center", ha="right", fontsize=8.5, color="#333", style="italic",
           fontweight="semibold")
for tx in AT_TIMES:
    ax_at.text(tx, 3.05, str(tx), va="center", ha="center",
               fontsize=8, color="#666")

for row_y, vals, col, lbl in [
    (2.05, at_t, TC, ARM["folfox"]["lbl"]),
    (0.95, at_c, CC, ARM["ctrl"]["lbl"]),
]:
    ax_at.text(-0.9, row_y, lbl, va="center", ha="right",
               fontsize=8.5, color=col, fontweight="semibold")
    for tx, v in zip(AT_TIMES, vals):
        ax_at.text(tx, row_y, str(v), va="center", ha="center",
                   fontsize=9, color=col)

ax_at.text(18, -0.32, "Time (months)", ha="center", fontsize=9, color="#444")
for tx in AT_TIMES:
    ax_at.axvline(tx, color="#EEEEEE", lw=0.6)

# ══════════════════════════════════════════════════════════════════
# PANEL D — Grouped subgroup forest
# ══════════════════════════════════════════════════════════════════
ax_d = fig.add_subplot(outer[1, 1])
ax_d.set_facecolor("white")

Y_MAX  = y_positions[0] + 0.9
XMAX_D = 2.90
HR_X_D = 2.20
PINT_X = 2.87

for row, y in zip(flat_rows, y_positions):
    rtype = row[0]

    if rtype == "header":
        _, grp_lbl, *_, p_int = row
        ax_d.text(0.545, y, grp_lbl,
                  va="center", ha="left", fontsize=9,
                  color=GRP_COL, fontweight="bold",
                  transform=ax_d.get_yaxis_transform())
        if p_int is not None:
            p_str = f"p\u2009=\u2009{p_int:.2f}"
            p_col = "#C0392B" if p_int < 0.05 else "#888"
            ax_d.text(PINT_X, y, p_str,
                      va="center", ha="right", fontsize=9,
                      color=p_col,
                      fontweight="bold" if p_int < 0.05 else "normal")
        ax_d.axhline(y + 0.38, color="#DDDDDD", lw=0.7)
        continue

    _, lbl, n, hr, lo, hi, _ = row
    col = OVERALL_DOT if rtype == "overall" else FOREST_DOT
    ms  = 10.5 if rtype == "overall" else 7.5

    if rtype == "overall":
        ax_d.axhspan(y - 0.44, y + 0.44, color="#F0F0F0", alpha=0.9, zorder=0)

    ax_d.plot([lo, hi], [y, y], color="#CCCCCC", lw=1.6, zorder=2,
              solid_capstyle="round")
    for tx in [lo, hi]:
        ax_d.plot([tx, tx], [y - 0.17, y + 0.17], color="#CCCCCC", lw=1.6, zorder=2)
    ax_d.plot(hr, y, "o", ms=ms, color=col, zorder=4,
              markeredgecolor="white", markeredgewidth=0.9)

    ax_d.text(0.52, y, f"n={n}", va="center", ha="right",
              fontsize=8, color="#999",
              transform=ax_d.get_yaxis_transform())
    ax_d.text(HR_X_D, y,
              f"{fmt(hr)}  [{fmt(lo)}\u2013{fmt(hi)}]",
              va="center", ha="left", fontsize=8.5, color="#333")

ax_d.axvline(1.0, color=REF_COL, lw=1.3, ls="--", zorder=5)
ax_d.axhline(y_positions[0] - 0.58, color="#DDDDDD", lw=0.7)

data_y   = [y for row, y in zip(flat_rows, y_positions) if row[0] != "header"]
data_lbl = [row[1] for row in flat_rows if row[0] != "header"]
ax_d.set_yticks(data_y)
ax_d.set_yticklabels(data_lbl, fontsize=9.5)

ax_d.set_xlim(0.44, XMAX_D)
ax_d.set_ylim(min(y_positions) - 0.7, Y_MAX)
ax_d.set_xlabel("Hazard Ratio", fontsize=10)
ax_d.set_title("(D)  Exploratory Subgroup Analysis  (individual covariates)",
               fontsize=12, fontweight="bold", loc="left", pad=26)
ax_d.tick_params(axis="y", length=0)
ax_d.spines["bottom"].set_color("#CCCCCC")

HEADER_Y_D = Y_MAX - 0.18
ax_d.text(0.52, HEADER_Y_D, "n",
          ha="right", va="bottom", fontsize=9.5, color="#444", style="italic",
          transform=ax_d.get_yaxis_transform())
ax_d.text(1.24, HEADER_Y_D, "HR [95% CI]",
          ha="center", va="bottom", fontsize=9.5, color="#444", style="italic")
ax_d.text(PINT_X, HEADER_Y_D, "p\u202f(interaction)",
          ha="right", va="bottom", fontsize=9.5, color="#444", style="italic")
ax_d.axhline(Y_MAX - 0.52, color="#DDDDDD", lw=0.6)

ax_d.text(0.22, -0.09, "\u2190 Favours Pritamab",
          transform=ax_d.transAxes, fontsize=9, color=TC,
          fontweight="bold", ha="center")
ax_d.text(0.80, -0.09, "Favours Control \u2192",
          transform=ax_d.transAxes, fontsize=9, color=CC,
          fontweight="bold", ha="center")

# ── Figure label ──────────────────────────────────────────────────
fig.text(0.976, 0.018, "Figure 8", ha="right", va="bottom",
         fontsize=12, fontweight="bold", color="#555",
         bbox=dict(boxstyle="round,pad=0.4", fc="#F8F9F9", ec="#AAAAAA", lw=0.7))

out = r"f:\ADDS\figures\Figure8_Phase2_Pritamab_v13.png"
plt.savefig(out, dpi=200, bbox_inches="tight", facecolor="white", edgecolor="none")
plt.close()
print(f"[SAVED] {out}")
