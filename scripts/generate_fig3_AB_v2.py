"""
generate_fig3_AB_v2.py
======================
Figure 3: Patient-Tailored AI Drug Recommendation System & Dose-Response Analysis
  Panel A: Neural Network Architecture (Input->Hidden->Output, left to right)
  Panel B: Dose-Response EC50 shift curves with Pritamab

Scientific foundation:
  EC50 values (baseline, no Pritamab):
    5-FU         EC50 = 12,000 nM  (Longley DB et al., Nat Rev Cancer 2003; CRC cell lines)
    Oxaliplatin  EC50 =  3,750 nM  (Ahmad S, Oncologist 2010; mCRC clinical)
    Irinotecan   EC50 =  7,500 nM  (Xu Y et al., Int J Cancer 2002; CRC cells)
    Sotorasib    EC50 =     75 nM  (Canon J et al., Nature 2019; KRAS G12C cells)
    MRTX1133     EC50 =     30 nM  (Fell JB et al., JACS 2020; KRAS G12D cells)

  Pritamab EC50 shift (thermodynamic coupling):
    EC50(+Pritamab) = EC50(alone) * exp(-alpha * DeltaDeltaG / RT)
    alpha   = 0.35  (phenomenological coupling factor, Lee SH et al. 2021)
    DeltaDeltaG = +0.50 kcal/mol
    RT at 310K  = 0.616 kcal/mol
    shift factor = exp(-0.35 * 0.50 / 0.616) = exp(-0.2841) = 0.7525 ~ 0.753  (-24.7%)

  Hill equation for dose-response:
    Inhibition(%) = C^n / (EC50^n + C^n) * 100
    n = 1.2  (Hill cooperativity coefficient for CRC cell models)
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import FancyArrowPatch
import matplotlib.patches as mpatches
import os

OUT_DIR = r"f:\ADDS\outputs\pritamab_pptx_figures"
os.makedirs(OUT_DIR, exist_ok=True)

plt.rcParams.update({
    'font.family':       'DejaVu Sans',
    'font.size':         10,
    'figure.facecolor':  'white',
    'axes.facecolor':    'white',
    'axes.spines.top':   False,
    'axes.spines.right': False,
    'axes.linewidth':    1.0,
})

# ── Verified EC50 values ─────────────────────────────────
ec50_alone = {
    '5-FU':        12000,
    'Oxaliplatin':  3750,
    'Irinotecan':   7500,
    'Sotorasib':      75,
    'MRTX1133':       30,
}
# Shift factor: exp(-alpha * DDG / RT)
alpha   = 0.35
DDG     = 0.50   # kcal/mol
RT      = 0.616  # kcal/mol at T=310K
shift   = np.exp(-alpha * DDG / RT)   # = 0.7525
ec50_prit = {k: v * shift for k, v in ec50_alone.items()}

drug_colors = {
    '5-FU':        '#E74C3C',
    'Oxaliplatin': '#3498DB',
    'Irinotecan':  '#2ECC71',
    'Sotorasib':   '#9B59B6',
    'MRTX1133':    '#F39C12',
}

def hill(c, ec50, n=1.2):
    return c**n / (ec50**n + c**n) * 100

# ════════════════════════════════════════════════════════
fig = plt.figure(figsize=(15, 6.8), facecolor='white')
gs  = GridSpec(1, 2, figure=fig, wspace=0.42,
               left=0.03, right=0.97, top=0.87, bottom=0.10)

# ═══════════════════════════════════════════════════════
# PANEL A  —  Neural Network Architecture
# ═══════════════════════════════════════════════════════
ax_a = fig.add_subplot(gs[0])
ax_a.set_xlim(0, 11)
ax_a.set_ylim(0, 10)
ax_a.axis('off')
ax_a.set_facecolor('white')
ax_a.set_title(
    'Panel A  |  Patient-Tailored Drug Recommendation\n'
    'Neural Network Architecture (ADDS v5.3)',
    fontsize=10, fontweight='bold')

# Layer definitions (left to right: input → hidden1 → hidden2 → hidden3 → output)
layers = [
    {'x': 1.3,  'label': 'INPUT\nLAYER',                   'nodes': 6,
     'color': '#AED6F1', 'ec': '#1A5276',
     'inputs': ['Genomic\nProfile', 'PrPC Serum\nLevel', 'KRAS\nStatus',
                'Drug\nHistory', 'CT/Path\nData', 'Clinical\nFeatures']},
    {'x': 3.8,  'label': 'HIDDEN\nLAYER 1\n(256 units)',   'nodes': 5,
     'color': '#A9DFBF', 'ec': '#196F3D'},
    {'x': 6.0,  'label': 'HIDDEN\nLAYER 2\n(128 units)',   'nodes': 4,
     'color': '#A9DFBF', 'ec': '#196F3D'},
    {'x': 8.0,  'label': 'HIDDEN\nLAYER 3\n(64 units)',    'nodes': 3,
     'color': '#FAD7A0', 'ec': '#7E5109'},
    {'x': 10.0, 'label': 'OUTPUT\nLAYER',                  'nodes': 3,
     'color': '#F1948A', 'ec': '#922B21',
     'outputs': ['Efficacy\nScore (E)', 'Synergy\nScore (S)', 'Toxicity\nRisk (T)']},
]

node_pos = {}   # x -> list of y positions
r = 0.28

for layer in layers:
    n = layer['nodes']
    ys = np.linspace(2.0, 8.0, n)
    node_pos[layer['x']] = ys
    for y in ys:
        circle = plt.Circle((layer['x'], y), r,
                             color=layer['color'], ec=layer['ec'],
                             linewidth=1.5, zorder=3)
        ax_a.add_patch(circle)
    # Layer label below
    ax_a.text(layer['x'], 1.15, layer['label'],
              ha='center', va='top', fontsize=7, fontweight='bold', color='#1A252F')

# Connections between adjacent layers
for i in range(len(layers) - 1):
    x1 = layers[i]['x'] + r
    x2 = layers[i+1]['x'] - r
    for y1 in node_pos[layers[i]['x']]:
        for y2 in node_pos[layers[i+1]['x']]:
            ax_a.plot([x1, x2], [y1, y2],
                      color='#BDC3C7', lw=0.35, alpha=0.45, zorder=1)

# Input feature labels (left of input layer)
for label, y in zip(layers[0]['inputs'], node_pos[layers[0]['x']]):
    ax_a.text(layers[0]['x'] - r - 0.1, y, label,
              ha='right', va='center', fontsize=6.5, color='#2C3E50')

# Output labels (right of output layer)
out_colors = ['#C0392B', '#1A5276', '#784212']   # E, S, T
for label, y, col in zip(layers[-1]['outputs'], node_pos[layers[-1]['x']], out_colors):
    ax_a.text(layers[-1]['x'] + r + 0.12, y, label,
              ha='left', va='center', fontsize=8, fontweight='bold', color=col)

# Framework footnote
ax_a.text(5.5, 0.25,
          'Framework: Multi-task Deep Learning  |  ADDS v5.3 (PyTorch 2.x, RTX 5070)\n'
          'Training data: 18,532 drug-pair samples  |  Loss function: joint BCE + MSE',
          ha='center', va='center', fontsize=7, color='#5D6D7E', style='italic')

# ═══════════════════════════════════════════════════════
# PANEL B  —  Dose-Response Curves
# ═══════════════════════════════════════════════════════
ax_b = fig.add_subplot(gs[1])
c_range = np.logspace(0, 5, 600)

for drug, col in drug_colors.items():
    ec_a = ec50_alone[drug]
    ec_p = ec50_prit[drug]
    # Solid = alone, dashed = +Pritamab
    ax_b.semilogx(c_range, hill(c_range, ec_a), color=col, lw=1.8, ls='-',  alpha=0.70)
    ax_b.semilogx(c_range, hill(c_range, ec_p), color=col, lw=2.5, ls='--',
                  label=drug)

ax_b.axhline(50, color='#95A5A6', linestyle=':', linewidth=0.8)
ax_b.text(1.5, 52, '50% inhibition', fontsize=7.5, color='#95A5A6')

ax_b.set_xlabel('Drug Concentration (nM, log scale)', fontsize=10)
ax_b.set_ylabel('Tumour Cell Inhibition (%)', fontsize=10)
ax_b.set_title(
    'Panel B  |  Dose-Response EC\u2085\u2080 Shift with Pritamab\n'
    '(Solid: alone  |  Dashed: +Pritamab,  -24.7% EC\u2085\u2080)',
    fontsize=10, fontweight='bold')
ax_b.set_ylim(0, 108)

# Legend
legend = ax_b.legend(loc='upper left', fontsize=8.5, frameon=True,
                     framealpha=0.9, title='Drug  (dashed = +Pritamab)',
                     title_fontsize=8)

# Thermodynamic formula box
ax_b.text(0.97, 0.08,
          'EC\u2085\u2080(+Pritamab) = EC\u2085\u2080(alone) x exp(-\u03b1\u00b7\u0394\u0394G\u2021/RT)\n'
          '\u03b1 = 0.35  (thermodynamic coupling factor)\n'
          '\u0394\u0394G\u2021 = +0.50 kcal/mol  |  RT = 0.616 kcal/mol (310 K)\n'
          'shift = exp(-0.284) = 0.753  \u2192  EC\u2085\u2080 reduced by 24.7%',
          transform=ax_b.transAxes, fontsize=7.8, va='bottom', ha='right',
          bbox=dict(boxstyle='round,pad=0.45', facecolor='#EBF5FB',
                    edgecolor='#AED6F1', alpha=0.95))

# EC50 footnote
ax_b.text(0.02, 0.02,
          'EC\u2085\u2080 sources: 5-FU (Longley 2003), Oxaliplatin (Ahmad 2010),\n'
          'Irinotecan (Xu 2002), Sotorasib (Canon 2019), MRTX1133 (Fell 2020)\n'
          'Hill eq: Inhib(%) = C^n/(EC\u2085\u2080^n+C^n)*100, n=1.2',
          transform=ax_b.transAxes, fontsize=6.8, va='bottom',
          style='italic', color='#626567')

ax_b.grid(True, which='both', alpha=0.12, linewidth=0.5)

# ── Overall title ────────────────────────────────────────
fig.suptitle(
    'Figure 3  |  Patient-Tailored AI Drug Recommendation System & Dose-Response Analysis',
    fontsize=13, fontweight='bold', y=0.97)

# ── Save ────────────────────────────────────────────────
out = os.path.join(OUT_DIR, 'fig3_AB_v2.png')
fig.savefig(out, dpi=300, bbox_inches='tight', facecolor='white')
plt.close()
print(f"Saved: {out}")

# ── Verification ────────────────────────────────────────
print(f"\n=== EC50 Shift Verification ===")
print(f"alpha={alpha}, DDG={DDG} kcal/mol, RT={RT} kcal/mol")
print(f"shift = exp(-{alpha}*{DDG}/{RT}) = exp(-{alpha*DDG/RT:.4f}) = {shift:.4f}")
print(f"EC50 reduction = {(1-shift)*100:.1f}%")
print(f"\n{'Drug':<14} {'EC50_alone':>12} {'EC50_+Prit':>12} {'Ratio':>7}")
print("-" * 50)
for drug in ec50_alone:
    ea = ec50_alone[drug]
    ep = ec50_prit[drug]
    print(f"{drug:<14} {ea:>12.0f} {ep:>12.1f} {ep/ea:>7.4f}")
