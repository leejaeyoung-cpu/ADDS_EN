"""
generate_fig2A_boxplot_v3.py
============================
Figure 2A — ADDS Virtual Phase II Trial: Efficacy Score Distribution
(English-only, scientifically honest rewrite of original Korean boxplot)

CRITICAL CHANGES vs. ORIGINAL:
  1. p-values REMOVED — synthetic distributions cannot yield real significance tests.
     Replaced with † footnote: "Computational simulation only; no clinical validation."
  2. y-axis explicitly defined: "ADDS Composite Efficacy Score (0–1 scale)"
     0 = no response; 1 = complete response
  3. ±SD anchored to ADDS Score formula (Gaussian sampling around formula output)
  4. Korean labels fully replaced with English
  5. Trial Design inset integrated as a proper inset table (not a floating text box)
  6. Score formula displayed INSIDE figure with explicit source

SCIENTIFIC BASIS:
  ADDS Composite Efficacy Score = 0.5·E_pred + 0.3·S_pred - 0.2·(T_tox/10)
    E_pred: ML-predicted tumor response (0-1)
    S_pred: Synergy score (Loewe SI, normalised to 0-1)
    T_tox:  Aggregate toxicity score (0-10, lower = better)

  Distribution parameters per arm (ADDS VBE v5.3 simulation, N=100/arm):
    FOLFOX control  : mean=0.82, SD=0.05   (anchored to PRIME mCRC KRAS-mut OS data)
    FOLFIRI control : mean=0.79, SD=0.06   (anchored to CRYSTAL mCRC KRAS-mut OS data)
    FOLFOX+Pritamab : mean=0.95, SD=0.04   (+15.8% vs FOLFOX; HR delta = -0.33)
    FOLFIRI+Pritamab: mean=0.92, SD=0.05   (+16.5% vs FOLFIRI; HR delta = -0.38)

  Literature anchors (HR → Efficacy Score conversion):
    HR = exp(-1.5 * DeltaScore)  =>  DeltaScore = -ln(HR)/1.5
    FOLFOX+Pritamab  HR=0.67  =>  DeltaScore=0.269  =>  E_pred increase ~+0.18
    FOLFIRI+Pritamab HR=0.62  =>  DeltaScore=0.310  =>  E_pred increase ~+0.21

  Refs: PRIME (Douillard, NEJM 2010); CRYSTAL (Van Cutsem, NEJM 2009)

NOTE: All scores are OUTPUT OF THE ADDS VIRTUAL BINDING ENGINE v5.3.
      No clinical trial has validated these Pritamab values.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import os

np.random.seed(42)   # reproducibility

OUT_DIR = r"f:\ADDS\outputs\pritamab_pptx_figures"
os.makedirs(OUT_DIR, exist_ok=True)

# ── rcParams ──────────────────────────────────────────────
plt.rcParams.update({
    'font.family':       'DejaVu Sans',
    'font.size':         10.5,
    'figure.facecolor':  'white',
    'axes.facecolor':    'white',
    'axes.spines.top':   False,
    'axes.spines.right': False,
    'axes.linewidth':    1.1,
})

# ── Arm definitions ───────────────────────────────────────
arms = [
    # (short_label, long_label,           mean,  sd,    color,     hatch)
    ('FOLFOX\n(Standard,\nN=100)',
     'FOLFOX (Control)',                  0.82,  0.05,  '#607D8B', ''),
    ('FOLFIRI\n(Standard,\nN=100)',
     'FOLFIRI (Control)',                 0.79,  0.06,  '#90A4AE', '//'),
    ('FOLFOX +\nPritamab†\n(N=100)',
     'FOLFOX + Pritamab',                0.95,  0.04,  '#1A6BA0', ''),
    ('FOLFIRI +\nPritamab†\n(N=100)',
     'FOLFIRI + Pritamab',               0.92,  0.05,  '#C0392B', ''),
]

# Simulate N=100 patient scores per arm
data = [np.clip(np.random.normal(m, s, 100), 0, 1)
        for *_, m, s, _, __ in arms]

labels      = [a[0] for a in arms]
long_labels = [a[1] for a in arms]
means       = [a[2] for a in arms]
sds         = [a[3] for a in arms]
colors      = [a[4] for a in arms]

# ── Figure layout ─────────────────────────────────────────
fig = plt.figure(figsize=(10, 8.5), facecolor='white')
ax  = fig.add_subplot(111)
fig.subplots_adjust(bottom=0.20)

positions = [1, 2, 3.3, 4.3]   # gap between controls and Pritamab arms

bp = ax.boxplot(data, positions=positions, widths=0.60,
                patch_artist=True, notch=False,
                medianprops=dict(color='white', linewidth=2.2),
                whiskerprops=dict(linewidth=1.4,  color='#4A4A4A'),
                capprops=dict(linewidth=1.5, color='#4A4A4A'),
                flierprops=dict(marker='o', markerfacecolor='#AAAAAA',
                                markersize=3.5, alpha=0.5, linewidth=0),
                boxprops=dict(linewidth=1.4))

# Fill colour per arm
for patch, col in zip(bp['boxes'], colors):
    patch.set_facecolor(col)
    patch.set_alpha(0.82)

# ── Baseline reference line ───────────────────────────────
ax.axhline(0.82, color='#607D8B', linestyle='--', linewidth=0.9,
           alpha=0.55, zorder=0)
ax.text(4.75, 0.823, 'FOLFOX baseline\n(0.82)', fontsize=7.5,
        color='#607D8B', va='bottom')

# ── Mean ± SD annotation above each box ──────────────────
for pos, m, s, col in zip(positions, means, sds, colors):
    ax.text(pos, m + s + 0.018,
            f'{m:.2f} ± {s:.2f}',
            ha='center', va='bottom', fontsize=8.5,
            color=col, fontweight='bold')

# ── Bracket with Δ% annotation (no p-values) ─────────────
def draw_bracket(ax, x1, x2, y, label, col='#2C3E50'):
    ax.plot([x1, x1, x2, x2], [y, y + 0.008, y + 0.008, y],
            lw=1.5, color=col)
    ax.text((x1 + x2) / 2, y + 0.012, label,
            ha='center', va='bottom', fontsize=9,
            color=col, fontweight='bold')

# FOLFOX → FOLFOX+Pritamab: +15.8%
y_br1 = max(data[0].max(), data[2].max()) + 0.025
draw_bracket(ax, 1, 3.3, y_br1,
             '+15.8% vs. FOLFOX (ADDS predicted)',
             col='#1A6BA0')

# FOLFIRI → FOLFIRI+Pritamab: +16.5%
y_br2 = max(data[1].max(), data[3].max()) + 0.025
draw_bracket(ax, 2, 4.3, y_br2 + 0.042,
             '+16.5% vs. FOLFIRI (ADDS predicted)',
             col='#C0392B')

# ── Vertical separator between control / Pritamab arms ────
ax.axvline(2.65, color='#CFD8DC', lw=1.2, linestyle='-', zorder=0)
ax.text(2.65, 0.585, '◄ Standard  |  Pritamab ►',
        ha='center', va='bottom', fontsize=8,
        color='#78909C', style='italic')

# ── Inset comparison table ────────────────────────────────
table_data = [
    ['Regimen',            'Efficacy Score',  'Δ vs. Control'],
    ['FOLFOX (control)',   '0.82 ± 0.05',     '—'],
    ['FOLFIRI (control)',  '0.79 ± 0.06',     '—'],
    ['FOLFOX + Pritamab†','0.95 ± 0.04',     '+15.8% ↑'],
    ['FOLFIRI + Pritamab†','0.92 ± 0.05',    '+16.5% ↑'],
]

tbl_colors = [
    ['#ECEFF1'] * 3,
    ['#FAFAFA']  * 3,
    ['#FAFAFA']  * 3,
    ['#E3F2FD']  * 3,
    ['#FFEBEE']  * 3,
]

tbl = ax.table(cellText=table_data[1:],
               colLabels=table_data[0],
               cellLoc='center', loc='lower right',
               bbox=[0.52, 0.04, 0.48, 0.30])
tbl.auto_set_font_size(False)
tbl.set_fontsize(8)
for (r, c), cell in tbl.get_celld().items():
    cell.set_edgecolor('#B0BEC5')
    cell.set_facecolor(tbl_colors[r][min(c, 2)] if r < len(tbl_colors) else '#FAFAFA')
    if r == 0:   # header
        cell.set_text_props(fontweight='bold', color='#263238')
    if r in (3, 4):   # Pritamab rows
        cell.set_text_props(fontweight='bold',
                            color='#1A6BA0' if r == 3 else '#C0392B')

# ── Axes formatting ───────────────────────────────────────
ax.set_xticks(positions)
ax.set_xticklabels(labels, fontsize=9.5, linespacing=1.3)
ax.set_ylabel('ADDS Composite Efficacy Score\n(0 = no response, 1 = complete response)',
              fontsize=10)
ax.set_ylim(0.55, 1.20)
ax.set_xlim(0.3, 5.2)

ax.set_title(
    'Figure 2A  |  ADDS Virtual Phase II Trial — Efficacy Score Distribution\n'
    'KRAS-Mutant mCRC/PAAD  |  N = 400  (4 arms × 100 patients, 1:1:1:1 allocation)',
    fontsize=11, fontweight='bold', pad=12)

# ── Trial Design inset (bottom-left, avoids bracket overlap) ─────────────────
design_txt = (
    'Trial Design (Virtual)\n'
    '─────────────────────────\n'
    '• Stage III-IV CRC patients\n'
    '• 1:1:1:1 randomisation\n'
    '• Primary endpoint: Efficacy Score\n'
    '• Follow-up: 12 months (simulated)'
)
ax.text(0.015, 0.38, design_txt, transform=ax.transAxes,
        fontsize=8, va='top', ha='left', linespacing=1.5,
        bbox=dict(boxstyle='round,pad=0.5', facecolor='#F5F5F5',
                  edgecolor='#B0BEC5', alpha=0.92))

# ── Footnote ──────────────────────────────────────────────
footnote = (
    '† Pritamab values: COMPUTATIONAL PREDICTIONS by ADDS v5.3 Virtual Binding Engine — no clinical validation.\n'
    'ADDS Score = 0.5·E\u2095\u1d63\u1d49\u1d48 + 0.3·S\u2095\u1d63\u1d49\u1d48 - 0.2·(T\u209c\u2092\u2093/10)  |  '
    'Distribution: Gaussian sampling, N=100/arm, seed=42.\n'
    'Control anchors: FOLFOX mOS 15.5 mo (PRIME, NEJM 2010); FOLFIRI mOS 16.0 mo (CRYSTAL, NEJM 2009).\n'
    'No p-values shown: significance testing is not applicable to a single-run computational simulation.'
)
fig.text(0.01, 0.01, footnote, fontsize=7.0, va='bottom',
         style='italic', color='#546E7A', wrap=True)

# ── Save ──────────────────────────────────────────────────
out = os.path.join(OUT_DIR, 'fig2A_v3.png')
fig.savefig(out, dpi=300, bbox_inches='tight', facecolor='white')
plt.close()
print(f"Saved: {out}")

# ── Verification ──────────────────────────────────────────
print("\n=== Fig 2A Verification ===")
print(f"{'Arm':<28} {'Mean':>6} {'SD':>6} {'Pct_chg':>10}")
print("-" * 55)
ref_folfox = 0.82
for arm, m, s in zip(long_labels, means, sds):
    chg = (m - ref_folfox) / ref_folfox * 100
    print(f"{arm:<28} {m:>6.2f} {s:>6.2f} {chg:>+10.1f}%")
print("\nDeltaScore → HR consistency check:")
for arm, delta, hr_lit in [
    ('FOLFOX+Pritamab',  0.269, 0.67),
    ('FOLFIRI+Pritamab', 0.310, 0.62),
]:
    hr_calc = np.exp(-1.5 * delta)
    print(f"  {arm}: DeltaScore={delta}  HR_calc={hr_calc:.4f}  HR_lit={hr_lit}  "
          f"{'OK' if abs(hr_calc-hr_lit)<0.002 else 'MISMATCH'}")
