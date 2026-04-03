"""
ADDS AI Inference Report — Publication Figure
==============================================
4-panel figure:
  A) AI Analysis Pipeline (flowchart)
  B) Cell Morphology Metrics (simulated output distributions)
  C) Clinical Decision Logic (rule-based therapy selection)
  D) Output Reliability Matrix (what can / cannot be inferred from images alone)
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
from matplotlib import gridspec
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import matplotlib.patheffects as PathEffects

# ─── Global style ───────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'axes.spines.top': False,
    'axes.spines.right': False,
})

# Colour palette
C = {
    'blue':    '#2563EB',
    'indigo':  '#4F46E5',
    'teal':    '#0D9488',
    'green':   '#16A34A',
    'amber':   '#D97706',
    'red':     '#DC2626',
    'gray':    '#64748B',
    'light':   '#F1F5F9',
    'white':   '#FFFFFF',
    'dark':    '#1E293B',
    'orange':  '#EA580C',
    'purple':  '#7C3AED',
    'ok':      '#22C55E',
    'warn':    '#F59E0B',
    'bad':     '#EF4444',
}

np.random.seed(42)

# ============================================================
fig = plt.figure(figsize=(24, 19), facecolor='white')
fig.text(0.5, 0.988,
         'ADDS AI-Based Inference Pipeline — Analysis Components and Output Reliability',
         ha='center', va='top', fontsize=15, fontweight='bold', color=C['dark'])
fig.text(0.5, 0.974,
         'AI-Based Anticancer Drug Discovery System (ADDS) v4.0  ·  Inha University Hospital  ·  Simulated demonstration data',
         ha='center', va='top', fontsize=9, color=C['gray'])

outer = gridspec.GridSpec(2, 2, figure=fig,
                          left=0.04, right=0.978,
                          top=0.958, bottom=0.045,
                          hspace=0.48, wspace=0.28)


# ============================================================
# Panel A — Pipeline Flowchart
# ============================================================
ax_a = fig.add_subplot(outer[0, 0])
ax_a.set_xlim(0, 10)
ax_a.set_ylim(0, 10)
ax_a.set_aspect('equal')
ax_a.axis('off')
ax_a.text(-0.02, 1.03, '(A)', transform=ax_a.transAxes,
          fontsize=12, fontweight='bold', color=C['dark'])
ax_a.set_title('ADDS AI Inference Pipeline', fontsize=11, fontweight='bold',
               color=C['dark'], pad=6, loc='left')

def draw_box(ax, x, y, w, h, label, sublabel='', color=C['blue'], fontsize=9):
    box = FancyBboxPatch((x - w/2, y - h/2), w, h,
                         boxstyle='round,pad=0.15',
                         facecolor=color, edgecolor='none', zorder=3)
    ax.add_patch(box)
    ax.text(x, y + (0.18 if sublabel else 0), label,
            ha='center', va='center', fontsize=fontsize,
            fontweight='bold', color='white', zorder=4)
    if sublabel:
        ax.text(x, y - 0.22, sublabel,
                ha='center', va='center', fontsize=6.5,
                color='white', alpha=0.88, zorder=4)

def draw_arrow(ax, x1, y1, x2, y2):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color=C['gray'],
                                lw=2.0, mutation_scale=14),
                zorder=2)

# Nodes (x, y, label, sublabel, colour)
nodes = [
    (5.0, 9.2, 'Cell Image Input',      'Microscopy TIFF/PNG',      C['indigo']),
    (5.0, 7.7, 'Preprocessing',         'CLAHE + Normalisation',     C['teal']),
    (5.0, 6.2, 'Cellpose Segmentation', 'cyto2 model (GPU)',         C['blue']),
    (5.0, 4.7, 'Feature Extraction',    'Area · Circularity · Count', C['blue']),
    (2.2, 3.1, 'Morphology\nReport',    'Cell count, Shape',         C['green']),
    (7.8, 3.1, 'Integration Engine',    'CT + Clinical + Cellpose',  C['orange']),
    (5.0, 1.5, 'CDSS Output',           'Rule-based Recommendation', C['purple']),
]

for x, y, lbl, sub, col in nodes:
    draw_box(ax_a, x, y, 3.2, 0.9, lbl, sub, col)

# Arrows
arrow_pairs = [
    (5.0,8.75, 5.0,8.15),
    (5.0,7.25, 5.0,6.65),
    (5.0,5.75, 5.0,5.15),
    (3.6,4.48, 2.5,3.56),
    (6.4,4.48, 7.5,3.56),
    (2.2,2.66, 3.6,1.96),
    (7.8,2.66, 6.4,1.96),
]
for x1,y1,x2,y2 in arrow_pairs:
    draw_arrow(ax_a, x1,y1,x2,y2)

# Caution callout
ax_a.text(0.5, 0.58, '⚠ Ki-67 not derivable\nfrom image alone\n(IHC required)',
          transform=ax_a.transAxes,
          ha='center', va='center', fontsize=7.5, color=C['red'],
          bbox=dict(boxstyle='round,pad=0.35', facecolor='#FEF2F2',
                    edgecolor=C['red'], lw=1.2))

# ============================================================
# Panel B — Cell Morphology Distributions
# ============================================================
ax_b = fig.add_subplot(outer[0, 1])
ax_b.axis('off')  # hide container; sub-panels drawn via b_gs below
ax_b.text(-0.08, 1.03, '(B)', transform=ax_b.transAxes,
          fontsize=12, fontweight='bold', color=C['dark'])
ax_b.set_title('Cell Morphology Metrics (N = 295 cells)', fontsize=11,
               fontweight='bold', color=C['dark'], pad=6, loc='left')

# Simulate 295-cell dataset (replicating b.png demo data)
n = 295
areas   = np.random.lognormal(np.log(4520), 0.35, n)   # px²
circs   = np.random.beta(6, 2.5, n)                     # 0–1
counts  = np.random.poisson(295, 40)                    # per image

b_gs = gridspec.GridSpecFromSubplotSpec(3, 1, subplot_spec=outer[0, 1],
                                        hspace=0.90)

# B1 — Cell area
ax_b1 = fig.add_subplot(b_gs[0])
ax_b1.hist(areas, bins=28, color=C['blue'], alpha=0.80, edgecolor='white', lw=0.4)
ax_b1.axvline(np.mean(areas), color=C['red'], lw=1.8, ls='--',
              label=f'Mean = {np.mean(areas):.0f} px²')
ax_b1.set_xlabel('Cell area (px²)', fontsize=8)
ax_b1.set_ylabel('Frequency', fontsize=8)
ax_b1.set_title('Cell Area Distribution', fontsize=8.5, fontweight='bold')
ax_b1.legend(fontsize=7.5, frameon=False)
ax_b1.tick_params(labelsize=7.5)
ax_b1.spines['top'].set_visible(False); ax_b1.spines['right'].set_visible(False)

# B2 — Circularity
ax_b2 = fig.add_subplot(b_gs[1])
ax_b2.hist(circs, bins=22, color=C['teal'], alpha=0.80, edgecolor='white', lw=0.4)
ax_b2.axvline(np.mean(circs), color=C['red'], lw=1.8, ls='--',
              label=f'Mean = {np.mean(circs):.3f}')
ax_b2.set_xlabel('Circularity (4πA/P²)', fontsize=8)
ax_b2.set_ylabel('Frequency', fontsize=8)
ax_b2.set_title('Cell Circularity Distribution', fontsize=8.5, fontweight='bold')
ax_b2.legend(fontsize=7.5, frameon=False)
ax_b2.tick_params(labelsize=7.5)
ax_b2.spines['top'].set_visible(False); ax_b2.spines['right'].set_visible(False)

# B3 — Cells per image across 40 images
ax_b3 = fig.add_subplot(b_gs[2])
ax_b3.bar(range(len(counts)), counts, color=C['indigo'], alpha=0.75, width=0.7)
ax_b3.axhline(np.mean(counts), color=C['amber'], lw=1.8, ls='--',
              label=f'Mean = {np.mean(counts):.0f}')
ax_b3.set_xlabel('Image index', fontsize=8)
ax_b3.set_ylabel('Cell count', fontsize=8)
ax_b3.set_title('Cell Count per Image (simulated batch)', fontsize=8.5,
                fontweight='bold')
ax_b3.legend(fontsize=7.5, frameon=False)
ax_b3.tick_params(labelsize=7.5)
ax_b3.spines['top'].set_visible(False); ax_b3.spines['right'].set_visible(False)

# ============================================================
# Panel C — Rule-based Therapy Decision Logic
# ============================================================
ax_c = fig.add_subplot(outer[1, 0])
ax_c.set_xlim(0, 10)
ax_c.set_ylim(0, 10)
ax_c.axis('off')
ax_c.text(-0.02, 1.03, '(C)', transform=ax_c.transAxes,
          fontsize=12, fontweight='bold', color=C['dark'])
ax_c.set_title('Rule-Based Therapy Recommendation Logic (NCCN-aligned)',
               fontsize=11, fontweight='bold', color=C['dark'], pad=6, loc='left')

# Decision nodes
def draw_diamond(ax, x, y, w, h, label, color=C['amber']):
    dx, dy = w/2, h/2
    pts = np.array([[x, y+dy],[x+dx, y],[x, y-dy],[x-dx, y]])
    poly = plt.Polygon(pts, closed=True, facecolor=color,
                       edgecolor='none', zorder=3)
    ax.add_patch(poly)
    ax.text(x, y, label, ha='center', va='center',
            fontsize=7.5, fontweight='bold', color='white', zorder=4,
            multialignment='center')

def draw_rect(ax, x, y, w, h, label, color=C['green']):
    box = FancyBboxPatch((x-w/2, y-h/2), w, h,
                         boxstyle='round,pad=0.12',
                         facecolor=color, edgecolor='none', zorder=3)
    ax.add_patch(box)
    ax.text(x, y, label, ha='center', va='center',
            fontsize=8, fontweight='bold', color='white', zorder=4,
            multialignment='center')

def carr(ax, x1,y1,x2,y2, label='', color=C['gray']):
    ax.annotate('', xy=(x2,y2), xytext=(x1,y1),
                arrowprops=dict(arrowstyle='->', color=color,
                                lw=1.8, mutation_scale=13), zorder=2)
    if label:
        mx, my = (x1+x2)/2, (y1+y2)/2
        ax.text(mx+0.15, my, label, fontsize=7, color=color, va='center')

# Top start
draw_rect(ax_c, 5, 9.3, 4.0, 0.8, 'Patient Data Integration\n(Cellpose + CT + Clinical)',
          C['indigo'])
carr(ax_c, 5,8.9, 5,8.3)

draw_diamond(ax_c, 5, 7.7, 3.0, 1.0, 'Liver &\nKidney\nNormal?', C['amber'])
carr(ax_c, 5,7.2, 5,6.5, 'Yes')
carr(ax_c, 3.5,7.7, 2.2,7.7)                               # No branch left
ax_c.text(2.9,7.85, 'No', fontsize=7, color=C['gray'])
draw_rect(ax_c, 1.3, 7.7, 1.9, 0.7, 'Exclude\nFOLFOX', C['red'])

draw_rect(ax_c, 5, 6.0, 3.2, 0.8, 'FOLFOX\n(5-FU + Leu + Oxali)', C['green'])
carr(ax_c, 5,5.6, 5,4.9)

draw_diamond(ax_c, 5, 4.3, 3.0, 1.0, 'MSI\nStatus\n= High?', C['amber'])
carr(ax_c, 5,3.8, 5,3.1, 'No')
carr(ax_c, 6.5,4.3, 8.0,4.3)
ax_c.text(6.6,4.45, 'Yes', fontsize=7, color=C['gray'])
draw_rect(ax_c, 8.8, 4.3, 1.7, 0.8, 'Add\nPembro-\nlizumab', C['purple'])

draw_rect(ax_c, 5, 2.5, 3.6, 0.8, 'CAPOX + Bevacizumab\n(Alternative)', C['orange'])
carr(ax_c, 5,2.1, 5,1.4)

draw_rect(ax_c, 5, 0.9, 4.0, 0.8,
          'Recommendation Output\n(rule_based, confidence = N/A)', C['dark'])

# Note
ax_c.text(0.5, 0.04,
          '⚠ confidence values are rule-based; no trained model output',
          transform=ax_c.transAxes, ha='center', va='bottom',
          fontsize=7.5, color=C['red'],
          bbox=dict(boxstyle='round,pad=0.3', facecolor='#FEF2F2',
                    edgecolor=C['red'], lw=1.0))

# ============================================================
# Panel D — Output Reliability Matrix
# ============================================================
ax_d = fig.add_subplot(outer[1, 1])
ax_d.text(-0.08, 1.03, '(D)', transform=ax_d.transAxes,
          fontsize=12, fontweight='bold', color=C['dark'])
ax_d.set_title('Output Metric Reliability Classification',
               fontsize=11, fontweight='bold', color=C['dark'], pad=6, loc='left')

metrics = [
    # (metric, source, reliability, note)
    ('Cell count',            'Cellpose segmentation',       'High',        'Direct mask count'),
    ('Cell area (px²)',       'skimage regionprops',         'High',        'Calibration required for μm²'),
    ('Circularity (4πA/P²)',  'skimage regionprops',         'High',        'Standard formula'),
    ('Morphology score',      'Area CV heuristic',           'Moderate',    'Threshold-dependent'),
    ('Cell density',          'Count ÷ image area',          'Moderate',    'Depends on magnification'),
    ('Ki-67 index',           'IHC assay (NOT image)',        'Not from img','Requires separate IHC test'),
    ('Cancer stage',          'TNM from CT module',          'Moderate',    'Combined CT + clinical rule'),
    ('5-yr survival',         'Stage × risk lookup table',   'Indicative',  'Population-level; not personalised'),
    ('Therapy confidence',    'Rule-based (NCCN)',           'Rule-based',  'No probabilistic model'),
]

col_labels = ['Metric', 'Data Source', 'Reliability', 'Notes']
col_widths = [0.22, 0.28, 0.16, 0.34]

colors_rel = {
    'High':            ('#DCFCE7', '#16A34A'),
    'Moderate':        ('#FEF9C3', '#CA8A04'),
    'Not from img':    ('#FEE2E2', '#DC2626'),
    'Indicative':      ('#FEF3C7', '#D97706'),
    'Rule-based':      ('#EDE9FE', '#7C3AED'),
}

row_h = 0.085
header_y = 0.93

# Header row
ax_d.set_xlim(0, 1); ax_d.set_ylim(0, 1); ax_d.axis('off')

x = 0
for i, (lbl, w) in enumerate(zip(col_labels, col_widths)):
    ax_d.add_patch(mpatches.FancyBboxPatch(
        (x, header_y - 0.01), w - 0.005, 0.07,
        boxstyle='round,pad=0.005', facecolor=C['dark'], edgecolor='none'))
    ax_d.text(x + w/2, header_y + 0.025, lbl,
              ha='center', va='center', fontsize=8.5,
              fontweight='bold', color='white')
    x += w

# Data rows
for r, (metric, source, rel, note) in enumerate(metrics):
    y = header_y - 0.015 - (r+1) * row_h
    x = 0
    row_data = [metric, source, rel, note]
    bg_face, text_col = colors_rel.get(rel, ('#F8FAFC', C['dark']))

    for i, (cell, w) in enumerate(zip(row_data, col_widths)):
        face = bg_face if i == 2 else ('#F8FAFC' if r % 2 == 0 else 'white')
        tc   = text_col if i == 2 else C['dark']
        ax_d.add_patch(mpatches.FancyBboxPatch(
            (x + 0.001, y), w - 0.006, row_h - 0.008,
            boxstyle='square,pad=0.0', facecolor=face, edgecolor='#E2E8F0', lw=0.5))
        ax_d.text(x + 0.008 + (w-0.016)/2, y + (row_h-0.008)/2,
                  cell, ha='center', va='center',
                  fontsize=7.2, color=tc,
                  fontweight='bold' if i==2 else 'normal',
                  wrap=True)
        x += w

# Legend — two rows of 3 items
legend_items = [
    (colors_rel['High'][0],         colors_rel['High'][1],         'High reliability'),
    (colors_rel['Moderate'][0],     colors_rel['Moderate'][1],     'Moderate / context-dependent'),
    (colors_rel['Not from img'][0], colors_rel['Not from img'][1], 'Not from image alone'),
    (colors_rel['Indicative'][0],   colors_rel['Indicative'][1],   'Indicative / population-level'),
    (colors_rel['Rule-based'][0],   colors_rel['Rule-based'][1],   'Rule-based (no model confidence)'),
]
for i, (fc, ec, lbl) in enumerate(legend_items):
    lx = 0.01 + (i % 3) * 0.33
    ly = 0.06 - (i // 3) * 0.045
    ax_d.add_patch(mpatches.Rectangle((lx, ly - 0.012), 0.025, 0.025,
                                      facecolor=fc, edgecolor=ec, lw=1))
    ax_d.text(lx + 0.032, ly + 0.001, lbl,
              va='center', fontsize=7, color=C['dark'])

# ─── Save ───────────────────────────────────────────────────────────────────
OUT = r'f:\ADDS\figures\ADDS_AI_Report_Figure_v2.png'
plt.savefig(OUT, dpi=200, bbox_inches='tight', facecolor='white')
print(f'Saved: {OUT}')
plt.close()
