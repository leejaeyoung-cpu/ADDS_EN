"""
Pritamab AI-Driven Combination Therapy Pipeline
ADDS Framework Style — Nature Communications Figure
Replicates the ADDS Framework figure style with Pritamab-specific content
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.patheffects as pe
import numpy as np

# ── Color palette ──────────────────────────────────────────────────────────────
BG          = '#F0F4F8'      # Light blue-grey background (like the reference figure)
BG_WHITE    = '#FFFFFF'
TIER1_BG    = '#EBF5FB'
TIER2_BG    = '#EAF6EA'
TIER3_BG    = '#FFF3E0'

# Block colors
COL_GENOMIC = '#7E57C2'     # Purple
COL_CELL    = '#26A69A'     # Teal
COL_IMAGING = '#EF5350'     # Red-pink
COL_CLINICAL= '#42A5F5'     # Blue

COL_AI      = '#5C6BC0'     # Indigo (AI modules)
COL_PIPE    = '#0288D1'     # Pipeline steps
COL_ENS     = '#388E3C'     # Ensemble

COL_PRIM    = '#1565C0'     # Primary rec (blue)
COL_ALT     = '#2E7D32'     # Alternative (green)
COL_COND    = '#6A1B9A'     # Conditional (purple)

DARK        = '#1A237E'
TEXT_DARK   = '#212121'
TEXT_MED    = '#424242'
TEXT_LIGHT  = '#757575'
ARROW_CLR   = '#546E7A'

fig = plt.figure(figsize=(20, 26), facecolor=BG)
fig.patch.set_facecolor(BG)

# ── Main Title ─────────────────────────────────────────────────────────────────
fig.text(0.5, 0.975,
         'AI-Driven Pritamab Combination Therapy Selection: The ADDS Framework',
         ha='center', va='top', fontsize=22, fontweight='bold', color=DARK,
         fontfamily='DejaVu Sans')
fig.text(0.5, 0.960,
         'Anti-PrPc Monoclonal Antibody Synergy Pipeline for KRAS-Mutant Precision Oncology',
         ha='center', va='top', fontsize=13, color=TEXT_MED, style='italic')

# Helper: draw a rounded box with title + bullets
def draw_box(ax, x, y, w, h, title, lines, color, title_color=None,
             fontsize=8.0, title_fontsize=9.0, ax_coords=True, alpha_bg=0.18):
    if title_color is None:
        title_color = color
    box = FancyBboxPatch((x, y), w, h,
                          boxstyle="round,pad=0.015",
                          facecolor=color, edgecolor=color,
                          linewidth=1.8, alpha=alpha_bg,
                          transform=ax.transAxes if ax_coords else ax.transData)
    box.set_facecolor(BG_WHITE)
    box.set_alpha(1.0)
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                                boxstyle="round,pad=0.015",
                                facecolor=BG_WHITE, edgecolor=color,
                                linewidth=2.0,
                                transform=ax.transAxes if ax_coords else ax.transData))
    ax.text(x + w/2, y + h - 0.025, title,
            transform=ax.transAxes if ax_coords else ax.transData,
            ha='center', va='top', fontsize=title_fontsize, fontweight='bold',
            color=title_color)
    for i, line in enumerate(lines):
        ax.text(x + w/2, y + h - 0.055 - i * 0.030, line,
                transform=ax.transAxes if ax_coords else ax.transData,
                ha='center', va='top', fontsize=fontsize, color=TEXT_MED,
                linespacing=1.2)

def draw_arrow_fig(fig, x1, y1, x2, y2, color=ARROW_CLR, style='->', lw=2.0):
    """Arrow in figure coordinates"""
    fig.add_artist(FancyArrowPatch(
        (x1, y1), (x2, y2),
        transform=fig.transFigure,
        arrowstyle=f'{style},head_width=0.005,head_length=0.008',
        color=color, linewidth=lw, zorder=10
    ))

# ═══════════════════════════════════════════════════════════════════════════════
# TIER 1 — SOURCE DATA STREAMS
# ═══════════════════════════════════════════════════════════════════════════════
tier1_ax = fig.add_axes([0.03, 0.775, 0.94, 0.175])
tier1_ax.set_xlim(0, 1); tier1_ax.set_ylim(0, 1)
tier1_ax.axis('off')

# Tier1 background
t1_bg = FancyBboxPatch((0, 0), 1, 1,
                        boxstyle="round,pad=0.01",
                        facecolor=TIER1_BG, edgecolor='#1565C0',
                        linewidth=2.5, transform=tier1_ax.transAxes)
tier1_ax.add_patch(t1_bg)

tier1_ax.text(0.01, 0.96, 'TIER 1', fontsize=11, fontweight='bold',
              color='#1565C0', transform=tier1_ax.transAxes, va='top')
tier1_ax.text(0.20, 0.96, 'SOURCE DATA STREAMS', fontsize=14, fontweight='bold',
              color=TEXT_DARK, transform=tier1_ax.transAxes, va='top')

# 4 source blocks
sources = [
    ('[DNA]  Genomic Data', ['• KRAS G12D/V/C/G13D', '• PRNP/PrPc expression', '• MSI status, TMB', '• EGFR/HER2'], COL_GENOMIC, 0.03),
    ('[LAB]  Cellular Analysis', ['• H&E via AI-Cellpose', '• Nuclear density', '• PrPc IHC H-score', '• Stromal fraction'], COL_CELL, 0.27),
    ('[IMG]  Imaging Data', ['• CT tumour detection', '  (nnU-Net)', '• Anatomical saliency', '• TNM staging'], COL_IMAGING, 0.51),
    ('[Rx]  Clinical Data', ['• Demographics', '• ECOG 0–2', '• Prior treatments', '• Comorbidities'], COL_CLINICAL, 0.75),
]

for title, bullets, color, x0 in sources:
    tier1_ax.add_patch(FancyBboxPatch(
        (x0, 0.05), 0.22, 0.82,
        boxstyle="round,pad=0.015",
        facecolor=BG_WHITE, edgecolor=color, linewidth=2.0,
        transform=tier1_ax.transAxes))
    tier1_ax.text(x0 + 0.11, 0.84, title,
                  ha='center', va='top', fontsize=9.5, fontweight='bold',
                  color=color, transform=tier1_ax.transAxes)
    for i, b in enumerate(bullets):
        tier1_ax.text(x0 + 0.02, 0.70 - i * 0.145, b,
                      ha='left', va='top', fontsize=8.0, color=TEXT_MED,
                      transform=tier1_ax.transAxes)

# Down arrows from Tier1 to Tier2
for xpos in [0.145, 0.385, 0.620, 0.860]:
    draw_arrow_fig(fig, xpos, 0.775, xpos, 0.738, color='#1565C0', lw=2.5)

# ═══════════════════════════════════════════════════════════════════════════════
# TIER 2 — ADDS INTEGRATION ENGINE
# ═══════════════════════════════════════════════════════════════════════════════
tier2_ax = fig.add_axes([0.03, 0.415, 0.94, 0.310])
tier2_ax.set_xlim(0, 1); tier2_ax.set_ylim(0, 1)
tier2_ax.axis('off')

t2_bg = FancyBboxPatch((0, 0), 1, 1,
                        boxstyle="round,pad=0.01",
                        facecolor=TIER2_BG, edgecolor='#2E7D32',
                        linewidth=2.5, transform=tier2_ax.transAxes)
tier2_ax.add_patch(t2_bg)
tier2_ax.text(0.01, 0.97, 'TIER 2', fontsize=11, fontweight='bold',
              color='#2E7D32', transform=tier2_ax.transAxes, va='top')
tier2_ax.text(0.12, 0.97, 'ADDS INTEGRATION ENGINE',
              fontsize=14, fontweight='bold', color=TEXT_DARK,
              transform=tier2_ax.transAxes, va='top')

# ── AI Modules (left block) ────────────────────────────────────────────────────
tier2_ax.add_patch(FancyBboxPatch(
    (0.01, 0.04), 0.25, 0.88,
    boxstyle="round,pad=0.015",
    facecolor='#E8EAF6', edgecolor=COL_AI, linewidth=2.0,
    transform=tier2_ax.transAxes))
tier2_ax.text(0.135, 0.90, 'AI Modules',
              ha='center', va='top', fontsize=11, fontweight='bold',
              color=COL_AI, transform=tier2_ax.transAxes)

ai_modules = [
    ('Pathology\nAI-Cellpose', COL_CELL,    0.01, 0.62, 0.115),
    ('CT Imaging\nAI-nnU-Net', COL_IMAGING, 0.135, 0.62, 0.115),
    ('Genomic\nAI-PRNP',      COL_GENOMIC, 0.01, 0.35, 0.115),
    ('Biomarker\nAI-Pr/Pc',   '#F4511E',   0.135, 0.35, 0.115),
    ('Response\nAI-XGBoost',  '#039BE5',   0.01, 0.08, 0.115),
    ('Synergy\nAI-DeepSynergy','#00897B',  0.135, 0.076, 0.115),
]

for label, color, bx, by, bw in ai_modules:
    tier2_ax.add_patch(FancyBboxPatch(
        (bx+0.003, by), bw, 0.22,
        boxstyle="round,pad=0.01",
        facecolor=BG_WHITE, edgecolor=color, linewidth=1.8,
        transform=tier2_ax.transAxes))
    tier2_ax.text(bx + 0.003 + bw/2, by + 0.11, label,
                  ha='center', va='center', fontsize=7.5, fontweight='bold',
                  color=color, transform=tier2_ax.transAxes, linespacing=1.4)

# Arrow: AI modules → Pipeline
draw_arrow_fig(fig, 0.285, 0.570, 0.340, 0.570, color='#2E7D32', lw=2.5)

# ── Processing Pipeline (center block) ────────────────────────────────────────
tier2_ax.add_patch(FancyBboxPatch(
    (0.29, 0.04), 0.26, 0.88,
    boxstyle="round,pad=0.015",
    facecolor='#E3F2FD', edgecolor=COL_PIPE, linewidth=2.0,
    transform=tier2_ax.transAxes))
tier2_ax.text(0.42, 0.90, 'Processing Pipeline',
              ha='center', va='top', fontsize=11, fontweight='bold',
              color=COL_PIPE, transform=tier2_ax.transAxes)

pipe_steps = [
    'Data Normalisation',
    'Patient Stratification\n(PrPc H-score + KRAS)',
    'Feature Extraction',
    'Energy Landscape\nModelling (ΔG)',
    'Drug Interaction\nModelling',
    'Efficacy & Synergy\nScoring (4-Model)',
]
step_colors_pipe = ['#0288D1','#1565C0','#0288D1','#EF6C00','#0288D1','#2E7D32']
py_positions = [0.78, 0.63, 0.50, 0.37, 0.24, 0.09]

for i, (step, stepcolor, py) in enumerate(zip(pipe_steps, step_colors_pipe, py_positions)):
    tier2_ax.add_patch(FancyBboxPatch(
        (0.30, py-0.005), 0.24, 0.115,
        boxstyle="round,pad=0.01",
        facecolor=BG_WHITE, edgecolor=stepcolor, linewidth=1.5,
        transform=tier2_ax.transAxes))
    tier2_ax.text(0.42, py + 0.055, step,
                  ha='center', va='center', fontsize=7.8, color=stepcolor,
                  fontweight='bold', transform=tier2_ax.transAxes, linespacing=1.3)
    if i < len(pipe_steps) - 1:
        tier2_ax.annotate('', xy=(0.42, py_positions[i+1] + 0.115),
                          xytext=(0.42, py - 0.005),
                          xycoords='axes fraction', textcoords='axes fraction',
                          arrowprops=dict(arrowstyle='->', color='#546E7A', lw=1.5))

# Arrow: Pipeline → Ensemble
draw_arrow_fig(fig, 0.610, 0.570, 0.660, 0.570, color='#2E7D32', lw=2.5)

# ── Ensemble Methods (right block) ────────────────────────────────────────────
tier2_ax.add_patch(FancyBboxPatch(
    (0.67, 0.04), 0.31, 0.88,
    boxstyle="round,pad=0.015",
    facecolor='#E8F5E9', edgecolor=COL_ENS, linewidth=2.0,
    transform=tier2_ax.transAxes))
tier2_ax.text(0.825, 0.90, 'Ensemble Methods',
              ha='center', va='top', fontsize=11, fontweight='bold',
              color=COL_ENS, transform=tier2_ax.transAxes)

ensemble_items = [
    ('Random Forest', '(Backbone eligibility)', '#43A047'),
    ('Gradient Boosting', '(Response probability)', '#1E88E5'),
    ('Deep Learning MLP', '(Drug synergy)', '#8E24AA'),
    ('ODE Energy Model', '(ΔG filter, 55.6% rate ↓)', '#FB8C00'),
    ('4-Model Consensus', '(Bliss+Loewe+HSA+ZIP)', '#00897B'),
    ('DRS Aggregation', '(Drug Recommendation Score)', '#E53935'),
]
ey_positions = [0.77, 0.64, 0.51, 0.38, 0.25, 0.10]

for (label, sublabel, color), ey in zip(ensemble_items, ey_positions):
    tier2_ax.add_patch(FancyBboxPatch(
        (0.68, ey-0.008), 0.29, 0.115,
        boxstyle="round,pad=0.01",
        facecolor=BG_WHITE, edgecolor=color, linewidth=1.5,
        transform=tier2_ax.transAxes))
    tier2_ax.text(0.825, ey + 0.082, label,
                  ha='center', va='top', fontsize=8.5, fontweight='bold',
                  color=color, transform=tier2_ax.transAxes)
    tier2_ax.text(0.825, ey + 0.038, sublabel,
                  ha='center', va='top', fontsize=7.5, color=TEXT_MED,
                  transform=tier2_ax.transAxes)

# Down arrows from Tier2 to Tier3
for xpos in [0.20, 0.50, 0.80]:
    draw_arrow_fig(fig, xpos, 0.415, xpos, 0.378, color='#2E7D32', lw=2.5)

# ═══════════════════════════════════════════════════════════════════════════════
# TIER 3 — OUTPUT: COCKTAIL RECOMMENDATIONS (DRS Ranking)
# ═══════════════════════════════════════════════════════════════════════════════
tier3_ax = fig.add_axes([0.03, 0.090, 0.94, 0.270])
tier3_ax.set_xlim(0, 1); tier3_ax.set_ylim(0, 1)
tier3_ax.axis('off')

t3_bg = FancyBboxPatch((0, 0), 1, 1,
                        boxstyle="round,pad=0.01",
                        facecolor=TIER3_BG, edgecolor='#E65100',
                        linewidth=2.5, transform=tier3_ax.transAxes)
tier3_ax.add_patch(t3_bg)
tier3_ax.text(0.01, 0.97, 'TIER 3', fontsize=11, fontweight='bold',
              color='#E65100', transform=tier3_ax.transAxes, va='top')
tier3_ax.text(0.13, 0.97,
              'OUTPUT — PRITAMAB COMBINATION RECOMMENDATIONS (DRS Ranking)',
              fontsize=13, fontweight='bold', color=TEXT_DARK,
              transform=tier3_ax.transAxes, va='top')

# ── Three recommendation cards ─────────────────────────────────────────────────
cards = [
    {
        'header': 'PRIMARY RECOMMENDATION',
        'title': 'Pritamab + FOLFOX',
        'subtitle': '(5-FU + Leucovorin + Oxaliplatin)',
        'drs': 'DRS: 0.893',
        'bliss': 'Bliss Synergy: +21.0',
        'ec50': 'EC50 ↓ 24.0% (FOLFOX)',
        'points': [
            '>> Pi3K/AKT target; RPSA signalosome blockade',
            '>> PrPc-HIGH (H-score ≥50) + KRAS-mutant (any allele)',
            '     especially G12D (H-score 142) / G12V (H-score 138)',
            '>> 24.0% dose reduction → Toxicity ↓',
            '>> Loewe DRI 5-FU: 1.34 | Oxaliplatin: 1.34',
        ],
        'footer': 'FDA Guidelines · NCCN Protocol',
        'hdr_color': COL_PRIM,
        'border': COL_PRIM,
        'x0': 0.01,
    },
    {
        'header': 'ALTERNATIVE REGIMEN',
        'title': 'Pritamab + Sotorasib',
        'subtitle': '(KRAS G12C Covalent Inhibitor)',
        'drs': 'DRS: 0.882',
        'bliss': 'Bliss Synergy: +22.5',
        'ec50': 'EC50 ↓ 24.7% (Sotorasib)',
        'points': [
            '>> Dual-axis: RPSA-PrPc + G12C direct block',
            '>> KRAS G12C-mutant (~12-13% CRC) + PrPc-HIGH',
            '     VEGF anti-angiogenic (Bevacizumab) alt. available',
            '>> Addresses RTK-bypass via RPSA escape route',
            '>> Triple option: +SHP2i → DRS 0.835, Bliss +28.0',
        ],
        'footer': 'FDA Approved · ESMO Guideline',
        'hdr_color': COL_ALT,
        'border': COL_ALT,
        'x0': 0.345,
    },
    {
        'header': 'CONDITIONAL REGIMEN',
        'title': 'Pritamab + FOLFOXIRI',
        'subtitle': '(Triplet Intensification)',
        'drs': 'DRS: 0.784',
        'bliss': 'Bliss Synergy: +22.0',
        'ec50': 'EC50 ↓ 26.1% (Triplet)',
        'points': [
            '>> Triplet intensification + PrPc sensitisation',
            '>> KRAS-mutant · PrPc-HIGH · ECOG 0-1 (PS required)',
            '     Hepatic met. conversion · Younger patients',
            '[!] Requires clinical eligibility review',
            '>> Consider +Bevacizumab variant (TRIBE2 protocol)',
        ],
        'footer': 'Requires: Clinical Eligibility Review',
        'hdr_color': COL_COND,
        'border': COL_COND,
        'x0': 0.68,
    },
]

for card in cards:
    x0 = card['x0']
    cw = 0.305
    border = card['border']
    hdr_color = card['hdr_color']

    # Main border
    tier3_ax.add_patch(FancyBboxPatch(
        (x0, 0.03), cw, 0.90,
        boxstyle="round,pad=0.015",
        facecolor=BG_WHITE, edgecolor=border, linewidth=2.5,
        transform=tier3_ax.transAxes))

    # Header band
    tier3_ax.add_patch(FancyBboxPatch(
        (x0, 0.80), cw, 0.13,
        boxstyle="round,pad=0.015",
        facecolor=hdr_color, edgecolor=hdr_color, linewidth=2.0,
        transform=tier3_ax.transAxes))
    tier3_ax.text(x0 + cw/2, 0.875, card['header'],
                  ha='center', va='center', fontsize=9.5, fontweight='bold',
                  color='white', transform=tier3_ax.transAxes)

    # Combination title
    tier3_ax.text(x0 + cw/2, 0.785, card['title'],
                  ha='center', va='top', fontsize=13, fontweight='bold',
                  color=border, transform=tier3_ax.transAxes)
    tier3_ax.text(x0 + cw/2, 0.738, card['subtitle'],
                  ha='center', va='top', fontsize=8.5, color=TEXT_MED,
                  transform=tier3_ax.transAxes)

    # DRS badge
    tier3_ax.add_patch(FancyBboxPatch(
        (x0 + 0.03, 0.67), cw - 0.06, 0.055,
        boxstyle="round,pad=0.01",
        facecolor='#FFF8E1', edgecolor='#F9A825', linewidth=1.5,
        transform=tier3_ax.transAxes))
    tier3_ax.text(x0 + cw/2, 0.697, card['drs'],
                  ha='center', va='center', fontsize=11, fontweight='bold',
                  color='#E65100', transform=tier3_ax.transAxes)

    # Synergy metrics row
    tier3_ax.text(x0 + cw/2, 0.650, card['bliss'],
                  ha='center', va='top', fontsize=8.5, fontweight='bold',
                  color='#2E7D32', transform=tier3_ax.transAxes)
    tier3_ax.text(x0 + cw/2, 0.618, card['ec50'],
                  ha='center', va='top', fontsize=8.5, fontweight='bold',
                  color='#1565C0', transform=tier3_ax.transAxes)

    # Horizontal divider
    tier3_ax.plot([x0 + 0.02, x0 + cw - 0.02], [0.595, 0.595], '-',
                  color='#E0E0E0', lw=1.0, transform=tier3_ax.transAxes)

    # Bullet points
    for i, pt in enumerate(card['points']):
        tier3_ax.text(x0 + 0.02, 0.575 - i * 0.090, pt,
                      ha='left', va='top', fontsize=7.8, color=TEXT_MED,
                      transform=tier3_ax.transAxes, linespacing=1.3)

    # Footer
    tier3_ax.add_patch(FancyBboxPatch(
        (x0 + 0.01, 0.038), cw - 0.02, 0.055,
        boxstyle="round,pad=0.01",
        facecolor='#F5F5F5', edgecolor=border, linewidth=1.0,
        transform=tier3_ax.transAxes))
    tier3_ax.text(x0 + cw/2, 0.063, card['footer'],
                  ha='center', va='center', fontsize=7.5, color=TEXT_LIGHT,
                  style='italic', transform=tier3_ax.transAxes)

# ═══════════════════════════════════════════════════════════════════════════════
# BOTTOM — Supporting Infrastructure Banner
# ═══════════════════════════════════════════════════════════════════════════════
infra_ax = fig.add_axes([0.03, 0.030, 0.94, 0.045])
infra_ax.set_xlim(0, 1); infra_ax.set_ylim(0, 1)
infra_ax.axis('off')

infra_ax.add_patch(FancyBboxPatch((0, 0), 1, 1,
                                   boxstyle="round,pad=0.01",
                                   facecolor='#ECEFF1', edgecolor='#90A4AE',
                                   linewidth=1.5, transform=infra_ax.transAxes))

infra_items = [
    ('Serum PrPc Liquid Biopsy (AUC=0.777)', 0.07),
    ('IHC Dual-Biomarker Selection (H-score ≥50 + KRAS)', 0.32),
    ('Pharmacogenomics Library (113 drugs · 59 synergy pairs)', 0.60),
    ('ADDS v5.3 · TCGA n=2,285 · ClinTrialDB Integration', 0.83),
]
for label, xpos in infra_items:
    infra_ax.text(xpos, 0.50, label,
                  ha='left', va='center', fontsize=8.0, color='#455A64',
                  transform=infra_ax.transAxes)
    if xpos < 0.83:
        infra_ax.plot([xpos - 0.025, xpos - 0.025], [0.15, 0.85], '-',
                      color='#B0BEC5', lw=1.0, transform=infra_ax.transAxes)

# ── Key data annotations ───────────────────────────────────────────────────────
# Small data box overlaying right side of figure
stats_ax = fig.add_axes([0.73, 0.015, 0.24, 0.060])
stats_ax.set_xlim(0, 1); stats_ax.set_ylim(0, 1)
stats_ax.axis('off')

# ── Figure caption ─────────────────────────────────────────────────────────────
fig.text(0.5, 0.008,
         'Figure. ADDS-AI Pritamab Pipeline: PrPc IHC expression (CRC 74.5% · PDAC 76% · Gastric 68%) | EC50 ↓24.7% (4 drugs) | '
         'Bliss synergy: FOLFOX+21.0, Sotorasib+22.5 | Energy barrier restoration 55.6% | Patent KR Jan 2026',
         ha='center', va='bottom', fontsize=7.5, color=TEXT_LIGHT,
         style='italic')

plt.savefig(r'f:\ADDS\figures\pritamab_pipeline_figure.png',
            dpi=180, bbox_inches='tight', facecolor=BG, edgecolor='none')
print('Figure saved: f:\\ADDS\\figures\\pritamab_pipeline_figure.png')
plt.close()
