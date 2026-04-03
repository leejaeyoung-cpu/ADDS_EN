"""
Figure 1A — Physics-Engine Based, High-Quality Academic Figure (v2)
===================================================================
Data sources:
  • paper3_pritamab_kras.py  — KRAS pathway ΔG‡ (Eyring-Evans-Polanyi TST)
  • table_energy_model.csv   — per-step ΔG values
  • PK/PD: Hill dose-response, EC50 shift via thermodynamic coupling (α=0.35)
  • Synergy: Bliss independence  r=0.71 (ADDS energy_synergy_v6)

Layout:
  Row-0 col-0  (a) Membrane schematic (wide)
  Row-0 col-1  (b) KRAS energy landscape bar
  Row-1 col-0  (c) Hill dose-response curve
  Row-1 col-1  (d) Per-step inhibition bar
  Row-2 full   (e) Integrated mechanism flow
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, Rectangle
from matplotlib.gridspec import GridSpec
import matplotlib.patheffects as pe
import numpy as np
import os

# ── Output ─────────────────────────────────────────────────────────
OUT_DIR = r"f:\ADDS\outputs\pritamab_pptx_figures"
os.makedirs(OUT_DIR, exist_ok=True)

# ── Global style ────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family':       'DejaVu Sans',
    'figure.facecolor':  'white',
    'axes.facecolor':    'white',
    'pdf.fonttype':      42,
    'ps.fonttype':       42,
    'axes.spines.top':   False,
    'axes.spines.right': False,
})

# ── Physics constants ───────────────────────────────────────────────
R_kcal  = 1.987e-3   # kcal/(mol·K)
T_body  = 310.0      # K (37 °C)
RT      = R_kcal * T_body  # ≈ 0.616 kcal/mol
ALPHA   = 0.35

# KRAS pathway activation energies (kcal/mol)
STEPS       = ['KRAS-GTP\nactivation', 'RAF-1\nrecruitment',
               'MEK1/2\nphosphorylation', 'ERK1/2\nactivation',
               'Nuclear\ntranslocation']
dG_normal   = np.array([3.0, 2.5, 2.0, 1.5, 1.0])
dG_mutPrPC  = np.array([0.3, 1.25, 1.7, 1.25, 0.88])
dG_pritamab = np.array([0.8, 1.5,  1.8, 1.3,  0.9 ])
ddG         = dG_pritamab - dG_mutPrPC

rls_idx     = int(np.argmax(ddG))
ddG_rls     = ddG[rls_idx]           # +0.50 kcal/mol
ddG_ec50    = ddG_rls * ALPHA        # +0.175 kcal/mol

# PK/PD
def hill(c, EC50, n=1.2):
    return (c**n) / (EC50**n + c**n) * 100
EC50_alone     = 12000.0
EC50_shift_f   = np.exp(-ddG_ec50 / RT)
EC50_pritamab_ = EC50_alone * EC50_shift_f
dose_red_pct   = (1 - EC50_shift_f) * 100

# Per-step inhibition
inh = (1 - np.exp(-ddG / RT)) * 100

# ── Colour palette ──────────────────────────────────────────────────
C = dict(
    normal   = '#27AE60',
    mutPrPC  = '#E74C3C',
    pritamab = '#2980B9',
    PrPC     = '#1A6B9A',
    RPSA     = '#27AE60',
    KRAS_mut = '#D4AC0D',
    AB       = '#922B21',
    navy     = '#1A252F',
    grey     = '#7F8C8D',
    purple   = '#8E44AD',
    mem_ext  = '#B0C4DE',
    mem_int  = '#708090',
)

# ═══════════════════════════════════════════════════════════
#  Canvas
# ═══════════════════════════════════════════════════════════
fig = plt.figure(figsize=(17, 14), facecolor='white', dpi=150)
gs  = GridSpec(3, 2, figure=fig,
               height_ratios=[1.9, 1.6, 0.9],
               hspace=0.55, wspace=0.40,
               left=0.05, right=0.97, top=0.94, bottom=0.04)

fig.text(0.5, 0.976,
         'Figure 1A  |  Pritamab: Membrane Binding → Downstream Signal Suppression '
         '(Physics-Based Energy Model  ·  ADDS v5.3)',
         ha='center', va='top', fontsize=12.5, fontweight='bold', color=C['navy'])

# ═══════════════════════════════════════════════════════════
#  (a) Membrane schematic
# ═══════════════════════════════════════════════════════════
axM = fig.add_subplot(gs[0, 0])
axM.set_xlim(0, 10)
axM.set_ylim(0, 10)
axM.axis('off')
axM.set_title('(a)  PrPC–Pritamab Binding at Cell Membrane',
              fontsize=10.5, fontweight='bold', color=C['navy'], loc='left', pad=5)

# --- Extracellular space label
axM.text(9.8, 8.2, 'Extra-\ncellular', ha='center', va='center',
         fontsize=7, color='#5D6D7E', rotation=90)
axM.text(9.8, 2.2, 'Intra-\ncellular', ha='center', va='center',
         fontsize=7, color='#5D6D7E', rotation=90)

# --- Bilayer
for ly, col in [(3.85, C['mem_ext']), (3.20, C['mem_int'])]:
    axM.add_patch(Rectangle((0.2, ly), 9.2, 0.65,
                             facecolor=col, edgecolor='none', alpha=0.55, zorder=1))
axM.text(0.5, 4.28, 'Plasma Membrane', fontsize=7, color='#5D6D7E', style='italic')

# --- GPI anchor
axM.add_patch(plt.Circle((5.0, 3.85), 0.22, color=C['PrPC'],
                           ec='white', lw=1, zorder=3))
axM.plot([5.0, 5.0], [4.07, 4.70], color=C['PrPC'], lw=3.5, zorder=3,
         solid_capstyle='round')

# --- PrPC globular domain box (extracellular)
axM.add_patch(FancyBboxPatch((4.30, 4.70), 1.40, 2.30,
                              boxstyle='round,pad=0.12',
                              facecolor=C['PrPC'], edgecolor='white', lw=1.5, zorder=3))
for yy, lbl in [(4.92, 'GlobC  200–228'),
                 (5.55, 'α-H2   172–193'),
                 (6.15, 'α-H1   144–154')]:
    axM.text(5.0, yy, lbl, ha='center', va='center',
             fontsize=5.8, color='white', fontweight='bold')

# --- Octapeptide repeat region (N-term disordered)
axM.plot([5.0, 5.0], [7.00, 8.20], color=C['AB'], lw=4,
         linestyle='-.', zorder=4, solid_capstyle='round')
axM.add_patch(plt.Circle((5.0, 7.60), 0.36, color=C['AB'],
                           ec='#922B21', lw=1.5, zorder=5, alpha=0.9))
axM.text(5.0, 7.60, 'Epi', ha='center', va='center',
         fontsize=7.5, color='white', fontweight='bold', zorder=6)
axM.text(5.0, 8.68,
         'N-terminal Octapeptide Repeats\nRes 51–90  ·  Cu²⁺ binding site',
         ha='center', va='center', fontsize=7.5, color=C['AB'], fontweight='bold',
         bbox=dict(boxstyle='round,pad=0.25', facecolor='#FDEDEC',
                   edgecolor=C['AB'], alpha=0.92, lw=1.2))

# --- Pritamab antibody (dark red box, left side)
axM.add_patch(FancyBboxPatch((0.5, 7.10), 2.50, 1.05,
                              boxstyle='round,pad=0.12',
                              facecolor=C['AB'], edgecolor='white', lw=1.5, zorder=4))
axM.text(1.75, 7.75, 'Pritamab', ha='center', va='center',
         fontsize=9.5, color='white', fontweight='bold')
axM.text(1.75, 7.28, 'anti-PrPC IgG  ·  Kd ≈ 0.5 nM',
         ha='center', va='center', fontsize=6.5, color='#FADBD8')

# binding arrow
axM.annotate('', xy=(4.62, 7.60), xytext=(3.02, 7.60),
             arrowprops=dict(arrowstyle='->', color=C['AB'], lw=2.2,
                             mutation_scale=16))
axM.text(3.80, 7.90, 'binds  ★', ha='center', va='bottom',
         fontsize=8, color=C['AB'], fontweight='bold')

# --- PrPC → RPSA block annotation (✕)
for dx in [-0.18, 0.18]:
    axM.plot([5.0+dx, 5.0-dx+0.36], [4.35, 3.65],
             color='#E74C3C', lw=2.8, zorder=7, solid_capstyle='round')
    axM.plot([5.0+dx, 5.0-dx+0.36], [3.65, 4.35],
             color='#E74C3C', lw=2.8, zorder=7, solid_capstyle='round')
axM.text(5.55, 4.00, 'BLOCKED\nby Pritamab',
         ha='left', va='center', fontsize=7, color='#E74C3C', fontweight='bold')

# --- Downstream nodes (inside cell)
for yy, lbl, col in [(2.80, 'RPSA / 67LR\n(KRAS membrane scaffold)', C['RPSA']),
                      (1.70, 'KRAS–GTP  ↓   (activation suppressed)', C['KRAS_mut'])]:
    axM.add_patch(FancyBboxPatch((2.7, yy-0.42), 4.60, 0.82,
                                  boxstyle='round,pad=0.12',
                                  facecolor=col, edgecolor='white', lw=1.2, zorder=3))
    axM.text(5.0, yy, lbl, ha='center', va='center',
             fontsize=8, color='white', fontweight='bold')

# arrows inside cell
for y0, y1 in [(3.18, 3.00), (2.38, 2.15)]:
    axM.annotate('', xy=(5.0, y1), xytext=(5.0, y0),
                 arrowprops=dict(arrowstyle='->', color='#AEB6BF', lw=1.8,
                                 mutation_scale=12))

# --- Thermodynamics info box (bottom-left, clear of schematic)
axM.text(0.3, 0.15,
         'ADDS AutoDock-GPU  v5.3  (Eyring–Evans–Polanyi TST)\n'
         '─────────────────────────────────────────────\n'
         'ΔG_bind (Pritamab–PrPC)  =  −13.0 kcal/mol\n'
         'ΔG (PrPC–RPSA disrupted) =  −10.0 kcal/mol\n'
         'ΔΔG‡_RLS (KRAS-GTP step) =  +0.50 kcal/mol\n'
         'α-coupled ΔΔG‡            =  +0.175 kcal/mol\n'
         'EC₅₀ shift factor          =  exp(−0.175/0.616)',
         ha='left', va='bottom', fontsize=7, family='monospace', color=C['navy'],
         bbox=dict(boxstyle='round,pad=0.4', facecolor='#EBF5FB',
                   edgecolor='#2980B9', alpha=0.93, lw=1.2))

# ═══════════════════════════════════════════════════════════
#  (b) KRAS Energy Landscape
# ═══════════════════════════════════════════════════════════
axE = fig.add_subplot(gs[0, 1])
axE.set_title('(b)  KRAS Pathway Activation Energy Landscape\n'
              '(Eyring–Evans–Polanyi TST  ·  ADDS Physics Engine v5.3)',
              fontsize=9.5, fontweight='bold', color=C['navy'], loc='left', pad=5)

x = np.arange(len(STEPS))
w = 0.24
bar_groups = [
    ('WT KRAS (normal)',           dG_normal,   C['normal'],   '//'),
    ('KRAS-mut + PrPC↑ (worst)',   dG_mutPrPC,  C['mutPrPC'],  '\\\\'),
    ('KRAS-mut + Pritamab (Rx)',   dG_pritamab, C['pritamab'], ''),
]
for i, (lbl, vals, col, hatch) in enumerate(bar_groups):
    bars = axE.bar(x + (i-1)*w, vals, width=w, label=lbl,
                   color=col, edgecolor='white', lw=0.8,
                   hatch=hatch, alpha=0.88, zorder=3)
    for bar, v in zip(bars, vals):
        axE.text(bar.get_x() + bar.get_width()/2,
                 bar.get_height() + 0.05,
                 f'{v:.1f}', ha='center', va='bottom', fontsize=7, color='#2C3E50')

# ΔΔG‡ brace annotation on rate-limiting step
# KRAS-GTP is step index 0, so x[0]=0; bars are at x-w, x, x+w → use right bar at x+w
bx_bar  = x[rls_idx] + w   # rightmost bar of RLS group (Pritamab bar)
y_lo = dG_mutPrPC[rls_idx]
y_hi = dG_pritamab[rls_idx]
anno_x = bx_bar + 0.10
axE.annotate('', xy=(anno_x, y_hi + 0.02),
             xytext=(anno_x, y_lo - 0.02),
             arrowprops=dict(arrowstyle='<->', color=C['purple'],
                             lw=2.2, mutation_scale=12))
axE.text(anno_x + 0.12, (y_lo + y_hi)/2,
         f'ΔΔG‡ = +{ddG_rls:.2f}\nkcal/mol\n★ RLS',
         ha='left', va='center', fontsize=9, color=C['purple'], fontweight='bold')

axE.set_xticks(x)
axE.set_xticklabels(STEPS, fontsize=8.5, linespacing=1.3, color='#2C3E50')
axE.set_ylabel('Activation Energy ΔG‡ (kcal/mol)', fontsize=9)
axE.set_ylim(0, 4.5)
axE.yaxis.grid(True, linestyle='--', alpha=0.35, color='#BDC3C7', zorder=0)
axE.set_axisbelow(True)
axE.legend(loc='upper right', fontsize=8, framealpha=0.92, edgecolor='#BDC3C7')
axE.text(0.02, 0.02,
         f'T = {T_body:.0f} K  ·  RT = {RT:.3f} kcal/mol  ·  α = {ALPHA}',
         transform=axE.transAxes, fontsize=7, color='#7F8C8D', style='italic')

# ═══════════════════════════════════════════════════════════
#  (c) Hill Dose-Response
# ═══════════════════════════════════════════════════════════
axD = fig.add_subplot(gs[1, 0])
axD.set_title('(c)  Hill Dose–Response: 5-FU ± Pritamab\n'
              r'$f(C)=C^n/(EC_{50}^n+C^n)$  ·  '
              r'$EC_{50}^{shift}=\exp(-\alpha\Delta\Delta G^{\ddagger}/RT)$',
              fontsize=9.5, fontweight='bold', color=C['navy'], loc='left', pad=5)

conc       = np.logspace(np.log10(EC50_alone*0.01), np.log10(EC50_alone*50), 500)
f_a        = hill(conc, EC50_alone)
f_p        = hill(conc, EC50_pritamab_)

axD.semilogx(conc, f_a, '-',  color=C['mutPrPC'], lw=2.8,
             label='5-FU  alone  (KRAS-mut + PrPC↑)')
axD.semilogx(conc, f_p, '--', color=C['pritamab'], lw=2.8,
             label='5-FU  + Pritamab  (PrPC neutralised)')
axD.fill_betweenx([0, 100], EC50_pritamab_, EC50_alone,
                  alpha=0.09, color=C['pritamab'], label='Reduced-dose window')
axD.axhline(50, color='#7F8C8D', lw=0.9, ls=':', alpha=0.7)
axD.axvline(EC50_alone,     color=C['mutPrPC'], lw=0.9, ls=':', alpha=0.5)
axD.axvline(EC50_pritamab_, color=C['pritamab'],lw=0.9, ls=':', alpha=0.5)

# EC50 shift arrow
mid_x = np.sqrt(EC50_alone * EC50_pritamab_)
axD.annotate('', xy=(EC50_pritamab_, 50), xytext=(EC50_alone, 50),
             arrowprops=dict(arrowstyle='->', color='#2C3E50', lw=2.0,
                             mutation_scale=14))
axD.text(mid_x, 57,
         f'EC₅₀ ↓ {dose_red_pct:.1f}%\nα·ΔΔG‡ = {ddG_ec50:.3f} kcal/mol',
         ha='center', va='bottom', fontsize=9, color='#2C3E50', fontweight='bold',
         bbox=dict(boxstyle='round,pad=0.25', facecolor='#EBF5FB',
                   edgecolor='#2980B9', alpha=0.90))

axD.set_xlabel('5-Fluorouracil Concentration (nM)', fontsize=9)
axD.set_ylabel('Tumour Cell Inhibition (%)', fontsize=9)
axD.set_ylim(-5, 113)
axD.legend(fontsize=8.5, loc='upper left', framealpha=0.92, edgecolor='#BDC3C7')
axD.yaxis.grid(True, linestyle='--', alpha=0.3, color='#BDC3C7', zorder=0)
axD.set_axisbelow(True)

# ═══════════════════════════════════════════════════════════
#  (d) Per-step inhibition bar
# ═══════════════════════════════════════════════════════════
axI = fig.add_subplot(gs[1, 1])
axI.set_title('(d)  Per-Step Signalling Inhibition by Pritamab\n'
              r'Inhibition (%) = $(1 - e^{-\Delta\Delta G^{\ddagger}/RT})\times 100$  ·  T = 310 K',
              fontsize=9.5, fontweight='bold', color=C['navy'], loc='left', pad=5)

step_lbl = ['KRAS-GTP\nact.', 'RAF-1\nrecruit.', 'MEK1/2\nphos.',
            'ERK1/2\nact.', 'Nuclear\ntransl.']

bar_cols = [C['purple'] if i == rls_idx else '#5D8AA8' for i in range(len(STEPS))]
bars_i = axI.bar(step_lbl, inh, color=bar_cols,
                  edgecolor='white', lw=1.0, width=0.52, zorder=3)

for b, v, dg in zip(bars_i, inh, ddG):
    axI.text(b.get_x() + b.get_width()/2, b.get_height() + 0.8,
             f'{v:.1f}%\nΔΔG‡=+{dg:.2f}',
             ha='center', va='bottom', fontsize=8, color='#1A252F',
             fontweight='bold', linespacing=1.4)

# RLS annotation
axI.annotate('★ Rate-limiting\nstep (RLS)',
             xy=(rls_idx, inh[rls_idx] + 1.0),
             xytext=(rls_idx + 1.3, inh[rls_idx] * 0.80),
             fontsize=8.5, color=C['purple'], fontweight='bold',
             arrowprops=dict(arrowstyle='->', color=C['purple'], lw=1.5,
                             connectionstyle='arc3,rad=0.20'))

axI.set_ylabel('Signalling Inhibition (%)', fontsize=9)
axI.set_ylim(0, inh.max() * 1.60)
axI.yaxis.grid(True, linestyle='--', alpha=0.3, color='#BDC3C7', zorder=0)
axI.set_axisbelow(True)
tls = axI.get_xticklabels()
tls[rls_idx].set_color(C['purple'])
tls[rls_idx].set_fontweight('bold')

# ═══════════════════════════════════════════════════════════
#  (e) Integrated mechanism flow (full-width)
# ═══════════════════════════════════════════════════════════
axF = fig.add_subplot(gs[2, :])
axF.set_xlim(0, 17)
axF.set_ylim(0, 3.2)
axF.axis('off')
axF.set_title('(e)  Integrated Mechanism Summary  —  '
              'PrPC Neutralisation → KRAS Signal Suppression',
              fontsize=10, fontweight='bold', color=C['navy'], loc='left', pad=5)

# Nodes: (cx, cy, label, bg_color, width, height)
NODES = [
    (1.40,  1.70, 'Pritamab\n(IgG)\nKd 0.5 nM',           C['AB'],       1.90, 0.90),
    (3.95,  1.70, 'PrPC\n(Res 51–90\nGPI-anchored)',        C['PrPC'],     2.00, 0.90),
    (6.70,  1.70, 'RPSA / 67LR\n(KRAS\nscaffold)',          C['RPSA'],     2.00, 0.90),
    (9.55,  1.70, 'KRAS–GTP\n(ΔΔG‡ +0.50\nkcal/mol ↑)',   '#7D8590',     2.10, 0.90),
    (12.50, 1.70, 'ERK / PI3K\nAkt / EMT\n(suppressed)',   '#5D6D7E',     2.10, 0.90),
    (15.40, 1.70, 'Apoptosis ↑\nGrowth ↓\nTumour control', '#229954',     1.90, 0.90),
]

for cx, cy, lbl, col, bw, bh in NODES:
    axF.add_patch(FancyBboxPatch((cx-bw/2, cy-bh/2), bw, bh,
                                  boxstyle='round,pad=0.12',
                                  facecolor=col, edgecolor='white', lw=1.5, zorder=3,
                                  path_effects=[pe.withSimplePatchShadow(
                                      offset=(2, -2), shadow_rgbFace='#00000020', alpha=0.2)]))
    axF.text(cx, cy, lbl, ha='center', va='center',
             fontsize=8, color='white', fontweight='bold', zorder=4,
             linespacing=1.35)

# Arrows between nodes
ARROWS = [
    #  x0    y0     x1     y1    color       label         rad    fc_label
    (2.37, 1.70, 2.97, 1.70, C['AB'],      'binds ★\nblocks',  0.0, '#FDEDEC'),
    (4.97, 1.70, 5.72, 1.70, '#C0392B',   '✕ disrupts',       0.0, None     ),
    (7.72, 1.70, 8.52, 1.70, '#C0392B',   '✕ reduces',        0.0, None     ),
    (10.62,1.70, 11.47,1.70, '#7F8C8D',   'suppresses',       0.0, None     ),
    (13.57,1.70, 14.47,1.70, '#229954',   'induces',          0.0, None     ),
]

for x0, y0, x1, y1, acol, albl, rad, fc in ARROWS:
    axF.annotate('', xy=(x1, y1), xytext=(x0, y0),
                 arrowprops=dict(arrowstyle='->', color=acol, lw=2.0,
                                 mutation_scale=13,
                                 connectionstyle=f'arc3,rad={rad}'))
    kw = dict(ha='center', va='bottom', fontsize=8, color=acol, fontweight='bold')
    if fc:
        kw['bbox'] = dict(boxstyle='round,pad=0.18', facecolor=fc,
                          edgecolor=acol, alpha=0.88, lw=0.9)
    axF.text((x0+x1)/2, y1 + 0.32, albl, **kw)

# Bottom citation bar
axF.text(8.5, 0.12,
         'ΔΔG‡_RLS = +0.50 kcal/mol  (KRAS-GTP act.)  ·  '
         'α-coupled ΔΔG‡ = +0.175 kcal/mol  ·  '
         'EC₅₀ shift = exp(−0.175/0.616) = −24.9%  ·  '
         'Bliss synergy r = 0.71 (ADDS energy_synergy_v6)',
         ha='center', va='bottom', fontsize=8, color='#5D6D7E', style='italic',
         bbox=dict(boxstyle='round,pad=0.32', facecolor='#F4F6F7',
                   edgecolor='#BDC3C7', alpha=0.92))

# ── Save ───────────────────────────────────────────────────
out_path = os.path.join(OUT_DIR, 'fig1A_physics_engine_v2.png')
fig.savefig(out_path, dpi=300, bbox_inches='tight', facecolor='white')
plt.close(fig)
print(f"Saved → {out_path}")
