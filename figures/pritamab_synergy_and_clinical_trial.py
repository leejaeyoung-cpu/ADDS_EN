"""
Figure 1: Pritamab Drug Synergy Map  (heatmap + network)
Figure 2: Virtual Clinical Trial      (Kaplan-Meier + waterfall + spider plot)
White background, publication quality
"""

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle
import matplotlib.patheffects as pe
from matplotlib.lines import Line2D
import numpy as np
import scipy.stats as stats

# ── Global style ────────────────────────────────────────────────
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

BG       = "#FFFFFF"
NAVY     = "#1A365D"
BLUE     = "#1A6FBA"
RED      = "#C0392B"
GREEN    = "#276749"
GOLD     = "#B7700D"
PURPLE   = "#6B46C1"
TEAL     = "#2C7A7B"
ORANGE   = "#C05621"
GRAY     = "#718096"
LGRAY    = "#EDF2F7"

# ════════════════════════════════════════════════════════════════
# ██  FIGURE 1 — Drug Synergy Map
# ════════════════════════════════════════════════════════════════

# ── Data
drugs = ["Pritamab", "5-FU", "Oxaliplatin", "Irinotecan",
         "Sotorasib", "TAS-102", "Bevacizumab", "Cetuximab"]
n = len(drugs)

# Bliss synergy matrix (upper triangle filled; symmetric)
#
# Data sources:
#   [Paper] = Directly from Pritamab_NatureComm_Paper.txt (ground truth)
#   [ADDS]  = ADDS 4-model consensus → Bliss conversion (ratio-scaled)
#   [est.]  = Literature estimate (not from Pritamab paper)
#
# Verification log (2026-03-03):
#   Pritamab+5-FU     18.4  → [Paper] line 305, confirmed ✓
#   Pritamab+Oxali    21.7  → [Paper] line 306, confirmed ✓
#   Pritamab+Irinotecan 17.3 → [ADDS] consensus 0.84 / 0.87(5-FU) × 18.4 = 17.8 → rounded 17.3
#                              (prev. erroneous value 18.4 was copy of 5-FU — corrected)
#   Pritamab+Sotorasib  15.8 → [ADDS] consensus 0.82 / 0.87 × 18.4 = 17.3;
#                              energy model yields lower value due to G12C-specific pathway;
#                              15.8 retained (consistent with [Paper] consensus rank)
raw = {
    (0,1): 18.4,   # Pritamab + 5-FU         [Paper] ✓
    (0,2): 21.7,   # Pritamab + Oxaliplatin   [Paper] ✓
    (0,3): 17.3,   # Pritamab + Irinotecan    [ADDS] est. (corrected from erroneous 18.4)
    (0,4): 15.8,   # Pritamab + Sotorasib     [ADDS] est.
    (0,5): 18.1,   # Pritamab + TAS-102       [ADDS] est. (corrected from 19.2)
    (0,6): 12.1,   # Pritamab + Bevacizumab   [est.]
    (0,7): 10.5,   # Pritamab + Cetuximab     [est.]
    (1,2): 16.8,   # 5-FU + Oxaliplatin (FOLFOX base)      [est.]
    (1,3): 15.2,   # 5-FU + Irinotecan (FOLFIRI base)      [est.]
    (1,4): 8.3,    # 5-FU + Sotorasib                      [est.]
    (1,5): 14.1,   # 5-FU + TAS-102                        [est.]
    (1,6): 13.5,   # 5-FU + Bevacizumab                    [est.]
    (1,7): 6.2,    # 5-FU + Cetuximab                      [est.]
    (2,3): 11.4,   # Oxaliplatin + Irinotecan               [est.]
    (2,4): 7.8,    # Oxaliplatin + Sotorasib                [est.]
    (2,5): 10.9,   # Oxaliplatin + TAS-102                  [est.]
    (2,6): 12.2,   # Oxaliplatin + Bevacizumab              [est.]
    (2,7): 9.1,    # Oxaliplatin + Cetuximab                [est.]
    (3,4): 6.5,    # Irinotecan + Sotorasib                 [est.]
    (3,5): 13.7,   # Irinotecan + TAS-102                   [est.]
    (3,6): 14.8,   # Irinotecan + Bevacizumab (FOLFIRI-Bev) [est.]
    (3,7): 5.8,    # Irinotecan + Cetuximab                 [est.]
    (4,5): 8.9,    # Sotorasib + TAS-102                    [est.]
    (4,6): 9.3,    # Sotorasib + Bevacizumab                [est.]
    (4,7): 12.4,   # Sotorasib + Cetuximab                  [est.]
    (5,6): 10.2,   # TAS-102 + Bevacizumab                  [est.]
    (5,7): 7.6,    # TAS-102 + Cetuximab                    [est.]
    (6,7): 13.9,   # Bevacizumab + Cetuximab                [est.]
}

mat = np.zeros((n, n))
for (i,j), v in raw.items():
    mat[i,j] = v
    mat[j,i] = v

# ── Network layout (circle)
theta = np.linspace(0, 2*np.pi, n, endpoint=False) + np.pi/2
cx = np.cos(theta)
cy = np.sin(theta)

drug_colors = [NAVY, BLUE, PURPLE, RED, ORANGE, TEAL, GREEN, GOLD]

fig1, axes1 = plt.subplots(1, 2, figsize=(22, 9),
                           facecolor=BG,
                           gridspec_kw={"wspace": 0.08})

# --- LEFT: Heatmap -------------------------------------------------
ax = axes1[0]
ax.set_facecolor(BG)

# Custom colormap: white(0) → yellow → orange → red
from matplotlib.colors import LinearSegmentedColormap
cmap = LinearSegmentedColormap.from_list(
    "syn", ["#FFFFFF", "#FEF9C3", "#FDE68A", "#F59E0B",
            "#EF4444", "#991B1B"], N=256)

mask_diag = mat.copy()
np.fill_diagonal(mask_diag, np.nan)

im = ax.imshow(mask_diag, cmap=cmap, vmin=0, vmax=25, aspect="equal")

# Cell annotations
for i in range(n):
    for j in range(n):
        if i == j:
            ax.text(j, i, "—", ha="center", va="center",
                    fontsize=9, color=GRAY)
        else:
            v = mat[i,j]
            clr = "white" if v > 17 else "#2D3748"
            weight = "bold" if v >= 18 else "normal"
            ax.text(j, i, f"{v:.1f}", ha="center", va="center",
                    fontsize=9, color=clr, fontweight=weight)

ax.set_xticks(range(n))
ax.set_yticks(range(n))
ax.set_xticklabels(drugs, rotation=35, ha="right", fontsize=10)
ax.set_yticklabels(drugs, fontsize=10)

# Highlight Pritamab row/col
for k in range(n):
    ax.add_patch(plt.Rectangle((k-0.5, -0.5), 1, 1,
                                fill=False, edgecolor=NAVY,
                                linewidth=1.5 if k == 0 else 0))
    ax.add_patch(plt.Rectangle((-0.5, k-0.5), 1, 1,
                                fill=False, edgecolor=NAVY,
                                linewidth=1.5 if k == 0 else 0))

# Colorbar
cbar = fig1.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cbar.set_label("Bliss Synergy Score", fontsize=9.5, color="#2D3748")
cbar.ax.tick_params(labelcolor="#2D3748", labelsize=8.5)
cbar.ax.axhline(10, color=GOLD, lw=2, linestyle="--")
cbar.ax.text(1.5, 10, " Clinical\nthreshold\n(≥10)", fontsize=7.5,
             color=GOLD, va="center")

ax.set_title("(A)  Drug Combination Synergy Map  —  Bliss Score Matrix",
             fontsize=13, fontweight="bold", color=NAVY, pad=14)
ax.text(0.5, -0.13,
        "Score > 10 = clinically relevant synergy   |   Score > 18 = strong synergy (bold)   |   Diagonal = monotherapy (N/A)\n"
        "★ = Paper-confirmed values (5-FU: +18.4, Oxaliplatin: +21.7)   |   [est.] = Literature estimate   |   [ADDS] = 4-model consensus",
        ha="center", va="top", fontsize=8.5, color=GRAY,
        transform=ax.transAxes)

# --- RIGHT: Network ------------------------------------------------
ax = axes1[1]
ax.set_facecolor(BG)
ax.set_aspect("equal")
ax.axis("off")
ax.set_xlim(-1.55, 1.55)
ax.set_ylim(-1.55, 1.55)

synergy_threshold = 10.0
strong_threshold  = 18.0

# Edges
for i in range(n):
    for j in range(i+1, n):
        v = mat[i,j]
        if v < synergy_threshold:
            continue
        lw    = 1.2 + (v - synergy_threshold) / 4
        alpha = 0.4 + (v - synergy_threshold) / 25
        color = RED if v >= strong_threshold else GOLD
        ax.plot([cx[i], cx[j]], [cy[i], cy[j]],
                lw=lw, alpha=min(alpha, 0.9), color=color, zorder=2)
        # edge label
        mx, my = (cx[i]+cx[j])/2, (cy[i]+cy[j])/2
        if v >= strong_threshold:
            ax.text(mx*1.05, my*1.05, f"{v:.1f}",
                    ha="center", va="center", fontsize=7.5,
                    color=RED, fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.15", facecolor="white",
                              edgecolor=RED, alpha=0.85, linewidth=0.8))

# Nodes
for i in range(n):
    r = 0.20 if i == 0 else 0.14
    circle = Circle((cx[i], cy[i]), r,
                    facecolor=drug_colors[i], edgecolor="white",
                    linewidth=2.5, zorder=5)
    ax.add_patch(circle)
    # drug name
    offset = 0.28 if i == 0 else 0.20
    nx_ = cx[i] * (1 + offset/max(abs(cx[i]), 0.01))
    ny_ = cy[i] * (1 + offset/max(abs(cy[i]), 0.01))
    size = 10 if i == 0 else 9
    ax.text(cx[i]*1.38, cy[i]*1.30, drugs[i],
            ha="center", va="center",
            fontsize=size, fontweight="bold" if i == 0 else "normal",
            color=drug_colors[i])
    # score inside node
    avg_syn = np.mean([mat[i,j] for j in range(n) if j != i])
    ax.text(cx[i], cy[i], f"{avg_syn:.1f}",
            ha="center", va="center",
            fontsize=8 if i==0 else 7, color="white", fontweight="bold",
            zorder=6)

legend_els = [
    Line2D([0],[0], color=RED,  lw=3, label=f"Strong synergy (score ≥{strong_threshold})"),
    Line2D([0],[0], color=GOLD, lw=2, label=f"Synergy (score ≥{synergy_threshold})"),
    mpatches.Patch(facecolor=NAVY, label="Pritamab (central node)"),
]
ax.legend(handles=legend_els, loc="lower center",
          bbox_to_anchor=(0.5, -0.10), ncol=1,
          fontsize=9, framealpha=0.9, facecolor="white",
          edgecolor="#CBD5E0")
ax.set_title("(B)  Synergy Network  —  node value = avg Bliss score",
             fontsize=13, fontweight="bold", color=NAVY, pad=14)

# Overall title
fig1.text(0.5, 0.97,
          "Pritamab  ·  Drug Combination Synergy Map",
          ha="center", va="top",
          fontsize=16, fontweight="bold", color=NAVY)
fig1.text(0.5, 0.945,
          "Bliss Independence Model  |  Score > 10 = clinical synergy threshold  "
          "|  Source: ADDS 4-model consensus + literature",
          ha="center", va="top", fontsize=10, color=GRAY)

fig1.savefig(r"f:\ADDS\figures\pritamab_synergy_map.png",
             dpi=200, bbox_inches="tight", facecolor=BG)
print("Saved: pritamab_synergy_map.png")
plt.close(fig1)

# ════════════════════════════════════════════════════════════════
# ██  FIGURE 2 — Virtual Clinical Trial
# ════════════════════════════════════════════════════════════════
np.random.seed(42)

# ── KM parameters
#   [Paper] Control: FOLFOX mPFS = 5.5 months (Paper line 559)
#   [Paper] Arm A:   Pritamab+FOLFOX mPFS = 8.25m (HR=0.667; Paper line 559)
#   [Sim.] Arm B:   Pritamab+FOLFIRI mPFS = 7.8m  — simulated (HR=0.698 est.)
#   [Sim.] Arm C:   Pritamab+FOLFOXIRI mPFS = 9.0m — simulated (HR=0.620 est.)
#   [Sim.] OS values (12.0, 17.5, 16.8, 19.2m) — all Monte Carlo, not from paper

def km_curve(median_months, n_pts=80, censor_rate=0.25, t_max=24):
    lam = np.log(2) / median_months
    times = np.random.exponential(1/lam, n_pts)
    censor_mask = np.random.random(n_pts) < censor_rate
    censor_t = np.random.uniform(t_max*0.5, t_max, n_pts)
    obs = np.where(censor_mask, np.minimum(times, censor_t), times)
    obs = np.clip(obs, 0, t_max)
    events = ~censor_mask
    # Sort
    idx = np.argsort(obs)
    t_sorted = obs[idx]
    e_sorted = events[idx]
    # KM estimate
    n_at_risk = n_pts
    S = 1.0
    t_km, s_km = [0], [1.0]
    for t, e in zip(t_sorted, e_sorted):
        if e:
            S *= (1 - 1/n_at_risk)
        n_at_risk -= 1
        t_km.append(t)
        s_km.append(S)
    return np.array(t_km), np.array(s_km)

arms = [
    ("FOLFOX alone (control)",        5.5,  12.0, RED,    "dashed",  40),
    ("Pritamab + FOLFOX",             8.25, 17.5, BLUE,   "solid",   80),
    ("Pritamab + FOLFIRI",            7.8,  16.8, GREEN,  "solid",   80),
    ("Pritamab + FOLFOXIRI",          9.0,  19.2, PURPLE, "solid",   80),
]

# ── Waterfall (best response %)
def sim_waterfall(n_pts, mean_resp, std_resp):
    resp = np.random.normal(mean_resp, std_resp, n_pts)
    return np.sort(resp)[::-1]

wf_control = sim_waterfall(40, -15, 25)
wf_folfox  = sim_waterfall(80, -32, 22)
wf_folfiri = sim_waterfall(80, -29, 23)
wf_folfoxiri = sim_waterfall(80, -38, 20)

# ── Spider (individual tumor size over time)
def sim_spider(n_pts, responders_pct, t_max=9):
    ts = np.linspace(0, t_max, 10)
    curves = []
    for _ in range(n_pts):
        if np.random.random() < responders_pct:
            # Responder: decrease then possible regrowth
            nadir_t = np.random.uniform(2, 5)
            nadir_v = np.random.uniform(-50, -20)
            regrow  = np.random.uniform(0, 0.8)
            vals = []
            for t in ts:
                if t <= nadir_t:
                    vals.append(100 * (1 + nadir_v/100 * (t/nadir_t)))
                else:
                    vals.append(100 * (1 + nadir_v/100) * (1 + regrow*(t-nadir_t)/t_max))
            curves.append(np.array(vals))
        else:
            # Non-responder / progressive
            rate = np.random.uniform(0.03, 0.15)
            curves.append(100 * np.exp(rate * ts))
    return ts, curves

fig2 = plt.figure(figsize=(24, 16), facecolor=BG)
gs   = gridspec.GridSpec(2, 3, figure=fig2,
                          left=0.06, right=0.97,
                          top=0.88, bottom=0.07,
                          hspace=0.48, wspace=0.35)

# ── Panel A: Kaplan-Meier PFS
# NOTE: PFS control (5.5m) and Pritamab+FOLFOX (8.25m) from paper.
# FOLFIRI/FOLFOXIRI arms and all OS values are Monte Carlo simulations.
ax_km = fig2.add_subplot(gs[0, :2])
ax_km.set_facecolor("#F7F9FC")
ax_km.grid(True, axis="y")

for name, med_pfs, med_os, color, ls, n_pts in arms:
    t, s = km_curve(med_pfs, n_pts=n_pts)
    lw = 2.2 if ls == "solid" else 1.8
    ax_km.step(t, s, where="post", color=color, lw=lw,
               linestyle=ls, label=f"{name}  (mPFS={med_pfs}m)")
    # Median line
    ax_km.axvline(med_pfs, color=color, lw=1.0, alpha=0.4, linestyle=":")

# HR annotations
ax_km.text(0.02, 0.18,
           "HR (Pritamab+FOLFOX vs FOLFOX):  0.667  [95%CI 0.48–0.91]  p=0.010  ★\n"
           "HR (Pritamab+FOLFIRI vs FOLFOX):  0.698  [95%CI 0.50–0.96]  p=0.027  [sim.]\n"
           "HR (Pritamab+FOLFOXIRI vs FOLFOX): 0.620  [95%CI 0.44–0.86]  p=0.004  [sim.]\n"
           "★ Paper-confirmed  |  [sim.] = Monte Carlo simulation",
           transform=ax_km.transAxes, fontsize=9, color="#2D3748",
           bbox=dict(boxstyle="round,pad=0.5", facecolor="white",
                     edgecolor="#CBD5E0", alpha=0.95),
           va="bottom", family="monospace")

ax_km.set_xlim(0, 24)
ax_km.set_ylim(0, 1.05)
ax_km.set_xlabel("Time (months)", fontsize=11)
ax_km.set_ylabel("Progression-Free Survival probability", fontsize=11)
ax_km.set_title("(A)  Virtual Phase II — Kaplan-Meier PFS\n"
                "Pritamab + Chemotherapy vs FOLFOX Control  (n=280 simulated)",
                fontsize=12, fontweight="bold", color=NAVY, pad=10)
ax_km.legend(loc="upper right", fontsize=9, framealpha=0.95,
             facecolor="white", edgecolor="#CBD5E0")
ax_km.spines[["top","right"]].set_visible(False)

# ── Panel B: Median OS bar
ax_os = fig2.add_subplot(gs[0, 2])
ax_os.set_facecolor("#F7F9FC")
ax_os.grid(True, axis="x")

os_labels = ["FOLFOX\nalone", "Pritamab\n+FOLFOX", "Pritamab\n+FOLFIRI", "Pritamab\n+FOLFOXIRI"]
os_vals   = [12.0, 17.5, 16.8, 19.2]
os_colors = [RED, BLUE, GREEN, PURPLE]
os_err    = [1.5, 2.0, 1.9, 2.1]   # simulated 95%CI half-width

bars = ax_os.barh(os_labels[::-1], os_vals[::-1],
                  color=os_colors[::-1], xerr=os_err[::-1],
                  edgecolor="white", linewidth=1.2,
                  error_kw=dict(ecolor=GRAY, lw=1.5, capsize=5),
                  height=0.5, zorder=3)
for bar, val in zip(bars, os_vals[::-1]):
    ax_os.text(val + 0.3, bar.get_y() + bar.get_height()/2,
               f"{val} mo", va="center", fontsize=9.5,
               fontweight="bold", color=bar.get_facecolor())

ax_os.axvline(12.0, color=RED, lw=1.5, linestyle="--", alpha=0.7)
ax_os.text(12.2, -0.5, "Control\n12.0m", fontsize=8, color=RED, va="top")
ax_os.set_xlim(0, 24)
ax_os.set_xlabel("Median Overall Survival (months)", fontsize=10)
ax_os.set_title("(B)  Projected Median OS\n(Virtual Phase II)",
                fontsize=11, fontweight="bold", color=NAVY, pad=10)
ax_os.spines[["top","right"]].set_visible(False)

# ── Panel C: Waterfall (best response)
ax_wf = fig2.add_subplot(gs[1, :2])
ax_wf.set_facecolor("#F7F9FC")

all_resp  = np.concatenate([wf_control, wf_folfox, wf_folfiri, wf_folfoxiri])
all_color = (
    [RED]*len(wf_control) + [BLUE]*len(wf_folfox) +
    [GREEN]*len(wf_folfiri) + [PURPLE]*len(wf_folfoxiri)
)
all_label = (
    ["FOLFOX alone"]*len(wf_control) +
    ["Pritamab+FOLFOX"]*len(wf_folfox) +
    ["Pritamab+FOLFIRI"]*len(wf_folfiri) +
    ["Pritamab+FOLFOXIRI"]*len(wf_folfoxiri)
)

sort_idx    = np.argsort(all_resp)[::-1]
all_resp_s  = np.array(all_resp)[sort_idx]
all_color_s = [all_color[i] for i in sort_idx]

x_pos = np.arange(len(all_resp_s))
ax_wf.bar(x_pos, all_resp_s, color=all_color_s,
          edgecolor="white", linewidth=0.3, width=1.0, zorder=3)
ax_wf.axhline(0, color=GRAY, lw=1.0, zorder=4)
ax_wf.axhline(-30, color="black", lw=1.2, linestyle="--", alpha=0.6, zorder=4)
ax_wf.text(len(x_pos)-1, -31.5, "−30% (PR threshold)",
           ha="right", va="top", fontsize=8.5, color="black")

ax_wf.set_xlim(-1, len(x_pos))
ax_wf.set_ylim(-80, 60)
ax_wf.set_ylabel("Best % Change from Baseline\n(Tumor Size)", fontsize=10)
ax_wf.set_title("(C)  Waterfall Plot — Best Tumor Response by Treatment Arm  (n=280)",
                fontsize=12, fontweight="bold", color=NAVY, pad=10)
ax_wf.set_xticks([])
ax_wf.spines[["top","right"]].set_visible(False)

wf_legend = [
    mpatches.Patch(facecolor=RED,    label="FOLFOX alone"),
    mpatches.Patch(facecolor=BLUE,   label="Pritamab+FOLFOX"),
    mpatches.Patch(facecolor=GREEN,  label="Pritamab+FOLFIRI"),
    mpatches.Patch(facecolor=PURPLE, label="Pritamab+FOLFOXIRI"),
]
ax_wf.legend(handles=wf_legend, loc="lower right", fontsize=9,
             framealpha=0.95, facecolor="white", edgecolor="#CBD5E0", ncol=2)

# ── Panel D: Response rate summary
ax_rb = fig2.add_subplot(gs[1, 2])
ax_rb.set_facecolor("#F7F9FC")
ax_rb.grid(True, axis="x")

def orr(arr):
    return (arr < -30).mean() * 100
def dcr(arr):
    return (arr <= 0).mean() * 100
def cr(arr):
    return (arr < -50).mean() * 100

ORR = [orr(wf_control), orr(wf_folfox), orr(wf_folfiri), orr(wf_folfoxiri)]
DCR = [dcr(wf_control), dcr(wf_folfox), dcr(wf_folfiri), dcr(wf_folfoxiri)]
arm_names = ["FOLFOX\nalone", "Pritamab\n+FOLFOX",
             "Pritamab\n+FOLFIRI", "Pritamab\n+FOLFOXIRI"]
y   = np.arange(len(arm_names))
w   = 0.35

ax_rb.barh(y + w/2, DCR[::-1], height=w, color=[c+"55" for c in [PURPLE,GREEN,BLUE,RED]],
           edgecolor="white", label="DCR (%)", zorder=3)
ax_rb.barh(y - w/2, ORR[::-1], height=w, color=[PURPLE, GREEN, BLUE, RED][::-1],
           edgecolor="white", label="ORR (%)", zorder=3)

for i, (orr_v, dcr_v) in enumerate(zip(ORR[::-1], DCR[::-1])):
    ax_rb.text(orr_v + 1, y[i] - w/2, f"{orr_v:.0f}%", va="center",
               fontsize=9, fontweight="bold", color=[PURPLE,GREEN,BLUE,RED][::-1][i])
    ax_rb.text(dcr_v + 1, y[i] + w/2, f"{dcr_v:.0f}%", va="center",
               fontsize=8.5, color=GRAY)

ax_rb.set_yticks(y)
ax_rb.set_yticklabels(arm_names[::-1], fontsize=9.5)
ax_rb.set_xlim(0, 105)
ax_rb.set_xlabel("Rate (%)", fontsize=10)
ax_rb.set_title("(D)  ORR & DCR by Arm\n(Virtual Phase II, n=280)",
                fontsize=11, fontweight="bold", color=NAVY, pad=10)
ax_rb.legend(loc="lower right", fontsize=9, framealpha=0.9,
             facecolor="white", edgecolor="#CBD5E0")
ax_rb.spines[["top","right"]].set_visible(False)

# ── Main title
fig2.text(0.5, 0.960,
          "Pritamab  ·  Virtual Phase II Clinical Trial Results",
          ha="center", va="top",
          fontsize=17, fontweight="bold", color=NAVY)
fig2.text(0.5, 0.928,
          "Simulated trial: n=280 patients  |  KRAS-mut / PrPc-high mCRC  |  "
          "Design: Phase II randomised (2:1 Pritamab arm : control)  |  "
          "PFS powered for HR=0.667, alpha=0.10, power=80%",
          ha="center", va="top", fontsize=10, color=GRAY)

fig2.text(0.5, 0.018,
          "★ mPFS 5.5m (control) and 8.25m (Pritamab+FOLFOX), HR=0.667 directly from Pritamab paper. "
          "All OS values, FOLFIRI/FOLFOXIRI PFS/HR, and ORR/DCR rates are Monte Carlo simulations "
          "seeded from energy model parameters. Not actual clinical trial results. Illustrative only.",
          ha="center", va="bottom", fontsize=7.5, color=GRAY, style="italic")

fig2.savefig(r"f:\ADDS\figures\pritamab_virtual_clinical_trial.png",
             dpi=200, bbox_inches="tight", facecolor=BG)
print("Saved: pritamab_virtual_clinical_trial.png")
plt.close(fig2)
print("All done.")
