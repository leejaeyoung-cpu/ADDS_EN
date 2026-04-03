"""
Figure 1 – Revised v2 (clean layout, no text overlap)
Pritamab mechanism of action, 4 panels A/B/C/D
White background, academic style, full English
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import numpy as np
import os

OUT_DIR = r"f:\ADDS\outputs\pritamab_pptx_figures"
os.makedirs(OUT_DIR, exist_ok=True)

plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
})

# ── Colors ───────────────────────────────────────────────────────────
C_PrPC   = '#1A6B9A'
C_RPSA   = '#217A4E'
C_AB     = '#922B21'
C_KRAS   = '#D4AC0D'
C_INH    = '#6C3483'
C_X_RED  = '#CB4335'
C_ARROW  = '#2C3E50'


def draw_panel(ax, label, drug_str,
               pathways, apop_fold, apop_pct, dg_list, bg_color):
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.set_facecolor(bg_color)
    for sp in ax.spines.values():
        sp.set_edgecolor('#BDC3C7')
        sp.set_linewidth(0.8)
    ax.set_xticks([])
    ax.set_yticks([])

    # ── Panel header ──────────────────────────────────────────────
    ax.text(5, 9.65, label, ha='center', va='top',
            fontsize=10, fontweight='bold', color='#1A252F')
    ax.text(5, 9.10, drug_str, ha='center', va='top',
            fontsize=8, color=C_AB, fontweight='bold')

    # ── PrPC receptor (centre-left) ───────────────────────────────
    # Stem
    stem = FancyBboxPatch((3.2, 3.5), 0.6, 3.2,
                           boxstyle='round,pad=0.05',
                           facecolor=C_PrPC, edgecolor='white', lw=1.5, zorder=4)
    ax.add_patch(stem)
    # Left arm
    ax.annotate('', xy=(2.4, 7.8), xytext=(3.5, 6.7),
                arrowprops=dict(arrowstyle='-', color=C_PrPC, lw=7,
                                connectionstyle='arc3,rad=-0.30'))
    # Right arm
    ax.annotate('', xy=(4.6, 7.8), xytext=(3.5, 6.7),
                arrowprops=dict(arrowstyle='-', color=C_PrPC, lw=7,
                                connectionstyle='arc3,rad=0.30'))
    # Epitope domain (red dot on N-terminal)
    ax.add_patch(plt.Circle((3.5, 8.2), 0.30,
                             color='#E74C3C', zorder=5, ec='white', lw=1.5))
    ax.text(3.5, 8.2, 'Epi', ha='center', va='center',
            fontsize=5.5, color='white', fontweight='bold', zorder=6)
    # PrPC label below
    ax.text(3.5, 3.0, 'PrPC\n(PRNP)', ha='center', va='top',
            fontsize=7, color=C_PrPC, fontweight='bold')

    # Epitope annotation (top-left)
    ax.annotate(
        'Epitope: PrPC\nN-terminal\nRes 51–90\n(Cu²⁺-octapeptide)',
        xy=(3.2, 8.2), xytext=(0.3, 8.8),
        fontsize=6, color='#922B21', va='top',
        arrowprops=dict(arrowstyle='->', color='#922B21', lw=0.8),
        bbox=dict(boxstyle='round,pad=0.2', facecolor='#FDEDEC',
                  edgecolor='#F1948A', alpha=0.9)
    )

    # ── Pritamab antibody (left of PrPC) ──────────────────────────
    ab_box = FancyBboxPatch((0.4, 6.5), 1.8, 1.4,
                             boxstyle='round,pad=0.08',
                             facecolor=C_AB, edgecolor='white', lw=1.5, zorder=5)
    ax.add_patch(ab_box)
    ax.text(1.3, 7.2, 'Pritamab\n(anti-PrPC\nhumanised IgG)',
            ha='center', va='center', fontsize=6.2, color='white',
            fontweight='bold', zorder=6)
    ax.text(1.3, 6.3, 'Kd ≈ 0.5 nM', ha='center', va='top',
            fontsize=6.5, color='#6C3483', fontstyle='italic')
    # Binding arrow: Pritamab → PrPC left arm
    ax.annotate('', xy=(2.6, 7.6), xytext=(2.2, 7.2),
                arrowprops=dict(arrowstyle='->', color=C_AB, lw=1.8))

    # ── RPSA (67LR) centre ───────────────────────────────────────
    rpsa_box = FancyBboxPatch((5.2, 4.8), 2.0, 2.4,
                               boxstyle='round,pad=0.08',
                               facecolor=C_RPSA, edgecolor='white', lw=1.5, zorder=3)
    ax.add_patch(rpsa_box)
    ax.text(6.2, 6.0, 'RPSA\n(67LR / 37LRP)\nLaminin Receptor',
            ha='center', va='center', fontsize=6.2, color='white',
            fontweight='bold', zorder=4)

    # Blocking connection: PrPC → RPSA (X)
    ax.annotate('', xy=(5.2, 5.8), xytext=(3.8, 5.8),
                arrowprops=dict(arrowstyle='->', color='#BDC3C7', lw=1.2,
                                connectionstyle='arc3,rad=0'))
    # X mark between PrPC and RPSA (blocking)
    cx, cy = 4.45, 5.8
    d = 0.28
    ax.plot([cx-d, cx+d], [cy-d, cy+d], color=C_X_RED, lw=2.5,
            solid_capstyle='round', zorder=7)
    ax.plot([cx-d, cx+d], [cy+d, cy-d], color=C_X_RED, lw=2.5,
            solid_capstyle='round', zorder=7)
    ax.text(4.45, 5.1, 'Blocked', ha='center', va='top',
            fontsize=5.5, color=C_X_RED, fontweight='bold')

    # ── KRAS-GTP spheres (right, inactive after blocking) ────────
    for i in range(5):
        ox = 8.4 + (i - 2) * 0.30
        ax.add_patch(plt.Circle((ox, 5.8), 0.12,
                                 color='#BDC3C7', ec='#95A5A6', lw=0.8, zorder=3))
    ax.text(8.4, 5.35, 'KRAS-GTP ↓\n(ADDS-suppressed)',
            ha='center', va='top', fontsize=6, color='#7F8C8D', style='italic')
    # Arrow RPSA → KRAS (blocked)
    ax.annotate('', xy=(7.7, 5.8), xytext=(7.2, 5.8),
                arrowprops=dict(arrowstyle='->', color='#BDC3C7', lw=1.0))

    # ── Inhibited pathways ────────────────────────────────────────
    py = 4.0
    for pname, pct in pathways:
        ax.text(3.0, py, pname, ha='right', va='center',
                fontsize=7, color='#2C3E50')
        # mini X
        mx, my = 3.15, py
        md = 0.18
        ax.plot([mx-md, mx+md], [my-md, my+md], color=C_X_RED, lw=2.0,
                solid_capstyle='round', zorder=7)
        ax.plot([mx-md, mx+md], [my+md, my-md], color=C_X_RED, lw=2.0,
                solid_capstyle='round', zorder=7)
        ax.text(3.40, py, f'{pct}% inhibition',
                ha='left', va='center', fontsize=7, color=C_X_RED,
                fontweight='bold')
        py -= 0.70

    # ── Apoptosis box ─────────────────────────────────────────────
    apop_box = FancyBboxPatch((0.3, 0.5), 9.4, 1.0,
                               boxstyle='round,pad=0.08',
                               facecolor='#FDEDEC', edgecolor='#E74C3C',
                               lw=1.0, zorder=3)
    ax.add_patch(apop_box)
    ax.text(5.0, 1.0,
            f'Apoptosis induction: {apop_fold}×-fold   |   '
            f'Overall inhibition efficiency: {apop_pct}%',
            ha='center', va='center', fontsize=7.5,
            fontweight='bold', color='#922B21', zorder=4)

    # ── Binding energy box ────────────────────────────────────────
    lines = ['ΔG (ADDS AI docking):']
    for lbl, val in dg_list:
        lines.append(f'  {lbl}: {val} kcal/mol')
    ax.text(9.7, 4.0, '\n'.join(lines), ha='right', va='top',
            fontsize=6, family='monospace', color='#1A252F',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#EBF5FB',
                      edgecolor='#2980B9', alpha=0.92, lw=0.8))


# ── Main figure ────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(16, 12), facecolor='white',
                          gridspec_kw={'hspace': 0.12, 'wspace': 0.08})
axes = axes.flatten()

BG = ['#FFFCFC', '#F9FFFC', '#FAFAFE', '#FFFDF5']

PANELS = [
    {
        'label':     '(A)  Pritamab alone',
        'drug_str':  'Pritamab  |  Anti-PrPC humanised monoclonal antibody',
        'pathways':  [('PI3K–Akt activity', 55), ('ERK1/2 activity', 55)],
        'apop_fold': 1.5, 'apop_pct': 55,
        'dg_list':   [('PrPC–Pritamab Kd', '−13.0'), ('PrPC–RPSA block ΔG', '−10.0')],
    },
    {
        'label':     '(B)  Pritamab + Irinotecan',
        'drug_str':  'Pritamab + Irinotecan  |  TOP1 inhibitor',
        'pathways':  [('PI3K–Akt activity', 75), ('Invasion', 75)],
        'apop_fold': 3.0, 'apop_pct': 75,
        'dg_list':   [('PrPC–Pritamab', '−13.5'), ('Irinotecan ΔG', '−13.5')],
    },
    {
        'label':     '(C)  Pritamab + Oxaliplatin',
        'drug_str':  'Pritamab + Oxaliplatin  |  Pt cross-linking agent',
        'pathways':  [('ERK1/2 activity', 75), ('Invasion', 20)],
        'apop_fold': 3.0, 'apop_pct': 75,
        'dg_list':   [('PrPC–Pritamab', '−14.0'), ('Oxaliplatin ΔG', '−13.0')],
    },
    {
        'label':     '(D)  Pritamab + TAS-102',
        'drug_str':  'Pritamab + TAS-102  |  FTD/TPI oral fluoropyrimidine',
        'pathways':  [('ERK–Akt activity', 20), ('Invasion', 20)],
        'apop_fold': 3.3, 'apop_pct': 80,
        'dg_list':   [('PrPC–Pritamab', '−14.3'), ('TAS-102 ΔG', '−14.3')],
    },
]

for ax, pd_, bg in zip(axes, PANELS, BG):
    draw_panel(ax, pd_['label'], pd_['drug_str'],
               pd_['pathways'], pd_['apop_fold'], pd_['apop_pct'],
               pd_['dg_list'], bg_color=bg)

fig.suptitle(
    'Figure 1  |  Pritamab Mechanism of Action: PrPC–RPSA–KRAS Axis Inhibition\n'
    'Epitope: PrPC N-terminal Cu²⁺-binding octapeptide repeats (residues 51–90, Kd ≈ 0.5 nM)',
    fontsize=13, fontweight='bold', y=1.01, color='#1A252F'
)

legend_str = (
    'PrPC (blue): cellular prion protein encoded by PRNP – oncogenic membrane scaffold in KRAS-mutant cancer  '
    '|  RPSA/67LR (green): ribosomal protein SA / 37-kDa laminin receptor – bridge for KRAS-GTP stabilisation\n'
    'Epi (red): Cu²⁺-binding octapeptide repeat region, residues 51–90, the actual Pritamab binding epitope  '
    '|  ΔG: ADDS AI AutoDock-GPU simulated binding free energies  '
    '|  Analysis: ADDS Framework v5.3 (Python 3.11, scikit-learn 1.3)'
)
fig.text(0.5, -0.01, legend_str, ha='center', va='top', fontsize=7.5,
         color='#5D6D7E', style='italic',
         bbox=dict(boxstyle='round,pad=0.4', facecolor='#F4F6F7',
                   edgecolor='#BDC3C7', alpha=0.9))

out_path = os.path.join(OUT_DIR, 'fig1_revised.png')
fig.savefig(out_path, dpi=300, bbox_inches='tight', facecolor='white')
plt.close(fig)
print(f"Saved: {out_path}")
