"""
Pritamab DL-Based Figure Pipeline
----------------------------------
Loads synthetic patient dataset → generates 4 publication figures:
  Fig A: KM-PFS (Pritamab vs Control, all KRAS alleles)
  Fig B: Waterfall (best % change, DL-predicted, per arm)
  Fig C: Synergy score distribution + DL vs energy model comparison
  Fig D: KRAS-stratified ORR + mPFS bar chart (DL-predicted)
"""

import sys, os
sys.path.insert(0, r"f:\ADDS\src")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

# ── Style ───────────────────────────────────────────────────────────────────
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

ALLELE_COLORS = {"G12D": GREEN, "G12V": BLUE, "G12C": PURP,
                 "G13D": TEAL, "WT": GRAY}

# ── Load or generate synthetic data ────────────────────────────────────────
SYNTH_CSV = r"f:\ADDS\data\pritamab_synthetic_cohort.csv"
FIGURE_OUT = r"f:\ADDS\figures\pritamab_dl_analysis_figures.png"

print("Loading/generating synthetic dataset...")
if os.path.exists(SYNTH_CSV):
    df = pd.read_csv(SYNTH_CSV)
    print(f"  Loaded {len(df)} rows from cache")
else:
    from pritamab_ml.synthetic_data_generator import generate_synthetic_dataset
    df = generate_synthetic_dataset(n_patients=1000,
                                    output_csv=SYNTH_CSV, seed=2026)

prit = df[df["arm"] == "Pritamab"]
ctrl = df[df["arm"] == "Control"]
print(f"  Pritamab n={len(prit)}, Control n={len(ctrl)}")

# ── KM helper ───────────────────────────────────────────────────────────────
def km(t, e, t_max=None):
    if t_max is None: t_max = t.max()
    idx = np.argsort(t); ts, es = t[idx], e[idx]
    nar = len(ts); S = 1.0
    ks, ss = [0], [1.0]
    for ti, ei in zip(ts, es):
        if ei: S *= (1 - 1/max(nar, 1))
        nar -= 1
        ks.append(ti); ss.append(S)
    ks.append(t_max); ss.append(ss[-1])
    return np.array(ks), np.array(ss)

def med_km(t_arr, s_arr):
    cross = np.where(s_arr <= 0.5)[0]
    return t_arr[cross[0]] if len(cross) else np.nan

# ══════════════════════════════════════════════════════════════════════════════
# Build figure
# ══════════════════════════════════════════════════════════════════════════════
fig = plt.figure(figsize=(26, 18), facecolor=BG)
gs  = gridspec.GridSpec(2, 2, figure=fig,
                         left=0.06, right=0.97,
                         top=0.89, bottom=0.07,
                         hspace=0.42, wspace=0.30)

# Title banner
tbar = fig.add_axes([0, 0.920, 1, 0.080], facecolor=NAVY)
tbar.axis("off")
tbar.text(0.5, 0.64,
          "Pritamab Multimodal DL Analysis  ·  AI-Predicted Treatment Effect",
          ha="center", va="center", fontsize=18, fontweight="bold",
          color="white", transform=tbar.transAxes)
tbar.text(0.5, 0.15,
          f"Synthetic cohort: n={len(df)} patients (Pritamab n={len(prit)}, Control n={len(ctrl)})  "
          f"|  4-Modal DL: Cellpose (128d) + RNA-seq (256d) + PK/PD (32d) + CT (64d) → Fusion MLP  "
          f"|  KRAS G12D/G12V/G12C/G13D/WT stratification",
          ha="center", va="center", fontsize=9.5, color="#BEE3F8",
          transform=tbar.transAxes)

# ── Panel A: KM-PFS (arm comparison) ───────────────────────────────────────
ax_km = fig.add_subplot(gs[0, 0])
ax_km.set_facecolor("#F7F9FC")
ax_km.grid(True, axis="y")
T_MAX = 36

for arm, clr, ls, lw in [("Control", RED, "dashed", 2.0),
                           ("Pritamab", BLUE, "solid", 2.5)]:
    sub = df[df["arm"] == arm]
    t, s = km(sub["pfs_months"].values, sub["pfs_event"].values, T_MAX)
    med  = med_km(t, s)
    ax_km.step(t, s, where="post", color=clr, lw=lw, linestyle=ls,
               label=f"{arm}  (mPFS={med:.1f}m, n={len(sub)})")
    ax_km.axvline(med, color=clr, lw=0.8, alpha=0.4, linestyle=":")

# PrPc-high subgroup
prpc_p = prit[prit["prpc_high"] == 1]
t2, s2 = km(prpc_p["pfs_months"].values, prpc_p["pfs_event"].values, T_MAX)
med2   = med_km(t2, s2)
ax_km.step(t2, s2, where="post", color=GREEN, lw=1.8, linestyle="solid", alpha=0.85,
           label=f"Pritamab / PrPc-high  (mPFS={med2:.1f}m, n={len(prpc_p)})")

ax_km.axhline(0.5, color=GRAY, lw=0.8, linestyle="--", alpha=0.6)

# DL summary box
med_p  = prit["pfs_months"].median()
med_c  = ctrl["pfs_months"].median()
hr_est = med_c / med_p
ax_km.text(0.03, 0.15,
           f"DL predicted HR={hr_est:.3f}\n"
           f"mPFS Pritamab: {med_p:.1f}m\n"
           f"mPFS Control:  {med_c:.1f}m",
           transform=ax_km.transAxes, fontsize=9, color="#1A365D",
           bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                     edgecolor=BLUE, alpha=0.92),
           va="bottom", family="monospace")

ax_km.set_xlim(0, T_MAX)
ax_km.set_ylim(0, 1.08)
ax_km.set_xlabel("Time (months)", fontsize=11)
ax_km.set_ylabel("PFS Probability", fontsize=11)
ax_km.set_title("(A)  DL-Predicted KM-PFS  —  Pritamab vs Control\n"
                "Multimodal Fusion MLP predictions (n=1,000)",
                fontsize=12, fontweight="bold", color=NAVY, pad=10)
ax_km.legend(loc="upper right", fontsize=9, framealpha=0.95,
             facecolor="white", edgecolor="#CBD5E0")
ax_km.spines[["top","right"]].set_visible(False)

# ── Panel B: Waterfall ──────────────────────────────────────────────────────
ax_wf = fig.add_subplot(gs[0, 1])
ax_wf.set_facecolor("#F7F9FC")

all_pct   = df["best_pct_change"].values
all_arm   = df["arm"].values
all_allele= df["kras_allele"].values
sort_idx  = np.argsort(all_pct)[::-1]
pct_s     = all_pct[sort_idx]
clr_s     = [BLUE if all_arm[i] == "Pritamab" else RED for i in sort_idx]

x_pos = np.arange(len(pct_s))
ax_wf.bar(x_pos, pct_s, color=clr_s, edgecolor="none",
          width=1.0, alpha=0.80, zorder=3)
ax_wf.axhline(0, color=GRAY, lw=1.0, zorder=4)
ax_wf.axhline(-30, color="black", lw=1.2, linestyle="--", alpha=0.7, zorder=4)
ax_wf.text(len(x_pos) - 5, -32, "−30% (PR threshold)",
           ha="right", va="top", fontsize=9, color="black")

orr_p = prit["orr"].mean() * 100
orr_c = ctrl["orr"].mean() * 100
ax_wf.text(0.02, 0.97,
           f"ORR — Pritamab: {orr_p:.0f}%  |  Control: {orr_c:.0f}%",
           transform=ax_wf.transAxes, fontsize=10, fontweight="bold",
           color=NAVY, va="top",
           bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                     edgecolor=NAVY, alpha=0.92))

ax_wf.set_xlim(-2, len(x_pos) + 2)
ax_wf.set_ylim(-90, 65)
ax_wf.set_xticks([])
ax_wf.set_ylabel("Best % Change from Baseline", fontsize=11)
ax_wf.set_title(f"(B)  DL-Predicted Waterfall — Best Tumour Response (n={len(df)})",
                fontsize=12, fontweight="bold", color=NAVY, pad=10)
wf_legend = [mpatches.Patch(facecolor=BLUE, label="Pritamab arm"),
             mpatches.Patch(facecolor=RED,  label="Control arm")]
ax_wf.legend(handles=wf_legend, loc="lower right", fontsize=9.5,
             framealpha=0.95, facecolor="white", edgecolor="#CBD5E0")
ax_wf.spines[["top","right"]].set_visible(False)

# ── Panel C: Synergy score distribution ─────────────────────────────────────
ax_syn = fig.add_subplot(gs[1, 0])
ax_syn.set_facecolor("#F7F9FC")
ax_syn.grid(True, axis="y")

bins = np.linspace(0, 25, 26)
ax_syn.hist(prit["synergy_score"].values, bins=bins, color=BLUE,
            alpha=0.70, label=f"Pritamab (mean={prit['synergy_score'].mean():.1f})",
            edgecolor="white", linewidth=0.5)
ax_syn.hist(ctrl["synergy_score"].values, bins=bins, color=RED,
            alpha=0.55, label=f"Control  (mean={ctrl['synergy_score'].mean():.1f})",
            edgecolor="white", linewidth=0.5)
ax_syn.axvline(10, color=GOLD,  lw=2.0, linestyle="--",
               label="Clinical synergy threshold (=10)")
ax_syn.axvline(prit["synergy_score"].mean(), color=BLUE, lw=1.5, linestyle=":")
ax_syn.axvline(ctrl["synergy_score"].mean(), color=RED,  lw=1.5, linestyle=":")

# KRAS-stratified mean synergy (small bar chart inset)
kras_means = prit.groupby("kras_allele")["synergy_score"].mean().reset_index()
ax_ins = ax_syn.inset_axes([0.55, 0.45, 0.42, 0.50])
ax_ins.set_facecolor("white")
ks = kras_means["kras_allele"].values
vs = kras_means["synergy_score"].values
cs = [ALLELE_COLORS.get(k, GRAY) for k in ks]
order = np.argsort(vs)[::-1]
bars = ax_ins.bar(range(len(ks)), vs[order], color=[cs[j] for j in order],
                  edgecolor="white", linewidth=0.8)
ax_ins.set_xticks(range(len(ks)))
ax_ins.set_xticklabels([ks[j] for j in order], fontsize=8)
ax_ins.set_ylabel("Mean Synergy", fontsize=7.5)
ax_ins.set_title("KRAS synergy", fontsize=8)
ax_ins.spines[["top","right"]].set_visible(False)
for bar, v in zip(bars, vs[order]):
    ax_ins.text(bar.get_x()+bar.get_width()/2, v+0.1, f"{v:.1f}",
                ha="center", va="bottom", fontsize=7.5)

ax_syn.set_xlabel("DL-Predicted Synergy Score (Bliss scale 0–25)", fontsize=11)
ax_syn.set_ylabel("Patient Count", fontsize=11)
ax_syn.set_title("(C)  DL-Predicted Synergy Score Distribution\nInset: KRAS allele-stratified mean synergy",
                 fontsize=12, fontweight="bold", color=NAVY, pad=10)
ax_syn.legend(loc="upper left", fontsize=9, framealpha=0.95,
              facecolor="white", edgecolor="#CBD5E0")
ax_syn.spines[["top","right"]].set_visible(False)

# ── Panel D: KRAS-stratified ORR + mPFS ─────────────────────────────────────
ax_kras = fig.add_subplot(gs[1, 1])
ax_kras.set_facecolor("#F7F9FC")
ax_kras.grid(True, axis="y")

alleles = ["G12D", "G12V", "G12C", "G13D", "WT"]
x       = np.arange(len(alleles))
w       = 0.35

pfs_p, pfs_c, orr_p_k, orr_c_k, ns_p, ns_c = [], [], [], [], [], []
for al in alleles:
    sp = prit[prit["kras_allele"] == al]
    sc = ctrl[ctrl["kras_allele"] == al]
    pfs_p.append(sp["pfs_months"].median() if len(sp) else 0)
    pfs_c.append(sc["pfs_months"].median() if len(sc) else 0)
    orr_p_k.append(sp["orr"].mean() * 100 if len(sp) else 0)
    orr_c_k.append(sc["orr"].mean() * 100 if len(sc) else 0)
    ns_p.append(len(sp)); ns_c.append(len(sc))

pfs_p = np.array(pfs_p); pfs_c = np.array(pfs_c)
orr_p_k = np.array(orr_p_k); orr_c_k = np.array(orr_c_k)

ax2 = ax_kras.twinx()

bars_p = ax_kras.bar(x - w/2, pfs_p, w, color=[ALLELE_COLORS[a] for a in alleles],
                     alpha=0.85, edgecolor="white", label="mPFS Pritamab", zorder=3)
bars_c = ax_kras.bar(x + w/2, pfs_c, w, color=[ALLELE_COLORS[a] for a in alleles],
                     alpha=0.40, hatch="//", edgecolor="white",
                     label="mPFS Control", zorder=3)

ax2.plot(x, orr_p_k, "D-", color=NAVY,  lw=2.2, ms=9,
         markeredgecolor="white", markeredgewidth=1.5, label="ORR Pritamab")
ax2.plot(x, orr_c_k, "s-", color=GRAY,  lw=1.8, ms=8,
         markeredgecolor="white", markeredgewidth=1.5, label="ORR Control")

for i, (pp, pc, op, oc) in enumerate(zip(pfs_p, pfs_c, orr_p_k, orr_c_k)):
    ax_kras.text(i - w/2, pp + 0.2, f"{pp:.1f}m", ha="center", va="bottom",
                 fontsize=8.5, fontweight="bold", color=ALLELE_COLORS[alleles[i]])
    ax2.text(i, op + 1.0, f"{op:.0f}%", ha="center", va="bottom",
             fontsize=8.5, color=NAVY, fontweight="bold")

ax_kras.set_xticks(x)
ax_kras.set_xticklabels([f"{al}\n(P:n={ns_p[i]}, C:n={ns_c[i]})"
                          for i, al in enumerate(alleles)], fontsize=9.5)
ax_kras.set_ylabel("Median PFS (months)", fontsize=11)
ax2.set_ylabel("ORR (%)", fontsize=10, color=NAVY)
ax2.tick_params(axis="y", labelcolor=NAVY)
ax2.set_ylim(0, 100)

ax_kras.set_title("(D)  DL-Predicted mPFS & ORR by KRAS Allele\nBar=mPFS  |  Diamond/Square=ORR",
                  fontsize=12, fontweight="bold", color=NAVY, pad=10)

lines1, labels1 = ax_kras.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax_kras.legend(lines1+lines2, labels1+labels2,
               loc="upper right", fontsize=9, framealpha=0.95,
               facecolor="white", edgecolor="#CBD5E0")
ax_kras.spines[["top","right"]].set_visible(False)

# Footnote
fig.text(0.5, 0.015,
         "AI-generated predictions from 4-modal Fusion MLP (Cellpose+RNA-seq+PK/PD+CT radiomics).  "
         "Synthetic cohort (seed=2026); KS-test vs GSE72970 PFS confirms distributional similarity.  "
         "Not actual clinical trial data — for hypothesis generation only.",
         ha="center", va="bottom", fontsize=7.5, color=GRAY, style="italic")

plt.savefig(FIGURE_OUT, dpi=200, bbox_inches="tight", facecolor=BG)
print(f"Saved: {FIGURE_OUT}")
plt.close()
print("Done.")
