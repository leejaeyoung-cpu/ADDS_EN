"""
Pritamab AI-Driven Combination Therapy Pipeline — PUBLICATION-QUALITY INFOGRAPHIC
Uses matplotlib vector backend → saves PDF (vector) + PNG (300 DPI rasterized from vector)
Box/shadow/gradient via FancyBboxPatch + LinearSegmentedColormap
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, PathPatch
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.patheffects as pe
import matplotlib.gridspec as gridspec
import numpy as np
import os

# ═══════════════════════════════════════════════════════════════════════════════
# COLOR PALETTE
# ═══════════════════════════════════════════════════════════════════════════════
P = {
    # Tier brand
    'T1': '#1565C0',   # blue
    'T2': '#2E7D32',   # green
    'T3': '#E65100',   # deep orange
    # Source blocks
    'genomic': '#7B1FA2',
    'cell':    '#00695C',
    'imaging': '#C62828',
    'clinic':  '#1565C0',
    # AI modules
    'ai1': '#5C6BC0',
    'ai2': '#00897B',
    'ai3': '#E53935',
    'ai4': '#F4511E',
    'ai5': '#039BE5',
    'ai6': '#00897B',
    # Ensemble
    'ens1': '#43A047',
    'ens2': '#1E88E5',
    'ens3': '#8E24AA',
    'ens4': '#FB8C00',
    'ens5': '#00897B',
    'ens6': '#E53935',
    # Rec cards
    'prim': '#1565C0',
    'alt':  '#2E7D32',
    'cond': '#6A1B9A',
    'gold': '#F57F17',
    # Neutral
    'bg':       '#EDF2F7',
    'white':    '#FFFFFF',
    'dark':     '#1A237E',
    'text_d':   '#212121',
    'text_m':   '#424242',
    'text_l':   '#757575',
    'arrow':    '#546E7A',
    'shadow':   '#90A4AE',
}

# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE SETUP — vector-quality
# ═══════════════════════════════════════════════════════════════════════════════
W_IN, H_IN = 22, 30    # inches
DPI = 300
fig = plt.figure(figsize=(W_IN, H_IN), facecolor=P['bg'])

# ── Gradient background via image ──────────────────────────────────────────────
bg_ax = fig.add_axes([0, 0, 1, 1], zorder=0)
bg_ax.set_xlim(0,1); bg_ax.set_ylim(0,1); bg_ax.axis('off')
grad = np.linspace(1, 0, 512).reshape(512, 1)
bg_ax.imshow(grad, aspect='auto', extent=[0,1,0,1],
             cmap=LinearSegmentedColormap.from_list('bg', ['#DDE8F5','#EDF2F7']),
             alpha=0.6, origin='upper', zorder=0)

# ═══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════════════
def fbox(ax, x, y, w, h, fill, edge, lw=2.5, r='round,pad=0.015', alpha=1.0,
         shadow=True, shadow_offset=(0.002,-0.002)):
    """Draw a FancyBboxPatch with optional drop-shadow."""
    if shadow:
        sh = FancyBboxPatch((x+shadow_offset[0], y+shadow_offset[1]), w, h,
                             boxstyle=r, facecolor='#90A4AE', edgecolor='none',
                             alpha=0.22, transform=ax.transAxes, zorder=2,
                             clip_on=False)
        ax.add_patch(sh)
    p = FancyBboxPatch((x, y), w, h, boxstyle=r,
                        facecolor=fill, edgecolor=edge, linewidth=lw,
                        transform=ax.transAxes, zorder=3, alpha=alpha, clip_on=False)
    ax.add_patch(p)
    return p

def fbox_data(ax, x, y, w, h, fill, edge, lw=2.5, r='round,pad=0.015',
              shadow=True, shadow_offset=(0.003,-0.004)):
    """FancyBboxPatch in data coordinates."""
    if shadow:
        sh = FancyBboxPatch((x+shadow_offset[0], y+shadow_offset[1]), w, h,
                             boxstyle=r, facecolor='#90A4AE', edgecolor='none',
                             alpha=0.22, zorder=2, clip_on=False)
        ax.add_patch(sh)
    p = FancyBboxPatch((x, y), w, h, boxstyle=r,
                        facecolor=fill, edgecolor=edge, linewidth=lw,
                        zorder=3, clip_on=False)
    ax.add_patch(p)
    return p

def txt(ax, x, y, s, fs=10, bold=False, color='#212121', ha='center', va='center',
        coord='axes', alpha=1.0, lsp=1.4):
    fw = 'bold' if bold else 'normal'
    tr = ax.transAxes if coord == 'axes' else ax.transData
    t = ax.text(x, y, s, fontsize=fs, fontweight=fw, color=color,
                ha=ha, va=va, transform=tr, alpha=alpha,
                linespacing=lsp, zorder=5)
    return t

def txt_outline(ax, x, y, s, fs=12, bold=True, color='white',
                outline='#1A237E', lw=3):
    """Text with outline (path effect)."""
    fw = 'bold' if bold else 'normal'
    t = ax.text(x, y, s, fontsize=fs, fontweight=fw, color=color,
                ha='center', va='center', transform=ax.transAxes, zorder=6)
    t.set_path_effects([pe.withStroke(linewidth=lw, foreground=outline)])
    return t

def arrow_down(fig, ax, x_fig, y_top_fig, y_bot_fig, color, lw=3):
    """Arrow in figure-fraction coords bridging two axes."""
    fig.add_artist(FancyArrowPatch(
        (x_fig, y_top_fig), (x_fig, y_bot_fig),
        transform=fig.transFigure,
        arrowstyle='simple,head_width=0.008,head_length=0.012',
        color=color, linewidth=lw, zorder=10))

def arrow_right(ax, x0, x1, cy, color='#546E7A', lw=3):
    ax.annotate('', xy=(x1, cy), xytext=(x0, cy),
                xycoords='axes fraction', textcoords='axes fraction',
                arrowprops=dict(arrowstyle='simple,head_width=0.38,head_length=0.015',
                                color=color, lw=lw), zorder=8)

def tier_header(ax, tier_label, title, color):
    """Draw a tier header band."""
    fbox(ax, 0, 0.90, 1.0, 0.10, fill=color, edge=color, lw=0, r='round,pad=0.0',
         shadow=False)
    txt(ax, 0.06, 0.950, tier_label, fs=13, bold=True,
        color=_lighten(color, 0.7), ha='left')
    txt(ax, 0.22, 0.950, title, fs=14, bold=True, color='white', ha='left')

def _lighten(hex_color, factor=0.5):
    h = hex_color.lstrip('#')
    r, g, b = int(h[0:2],16), int(h[2:4],16), int(h[4:6],16)
    r2 = int(r + (255-r)*factor)
    g2 = int(g + (255-g)*factor)
    b2 = int(b + (255-b)*factor)
    return f'#{r2:02X}{g2:02X}{b2:02X}'

def pill(ax, x, y, label, color, fs=8):
    """Small pill badge."""
    fbox(ax, x-0.005, y-0.014, 0.16, 0.028, fill=color, edge=color,
         lw=0, r='round,pad=0.008', shadow=False)
    txt(ax, x+0.075, y, label, fs=fs, bold=True, color='white', ha='center')

# ═══════════════════════════════════════════════════════════════════════════════
# MAIN LAYOUT  (figure-fraction positioning)
# ═══════════════════════════════════════════════════════════════════════════════
MARGIN_L = 0.025
MARGIN_R = 0.025
MID_W    = 1.0 - MARGIN_L - MARGIN_R   # 0.95

# Y positions (top-down in figure fraction)
HEADER_BOT = 0.942
T1_TOP     = 0.930
T1_BOT     = 0.720
T2_TOP     = 0.700
T2_BOT     = 0.360
T3_TOP     = 0.340
T3_BOT     = 0.040
INFRA_TOP  = 0.035
INFRA_BOT  = 0.010

# ─────────────────────────────────────────────────────────────────────────────
# HEADER BAR
# ─────────────────────────────────────────────────────────────────────────────
hdr_ax = fig.add_axes([0, HEADER_BOT, 1, 1 - HEADER_BOT], facecolor='none')
hdr_ax.set_xlim(0,1); hdr_ax.set_ylim(0,1); hdr_ax.axis('off')

# Gradient header
cmap_hdr = LinearSegmentedColormap.from_list('hdr', ['#0D47A1','#1565C0'])
hdr_img = np.linspace(0,1,256).reshape(1,256)
hdr_ax.imshow(hdr_img, aspect='auto', extent=[0,1,0,1],
              cmap=cmap_hdr, zorder=0, origin='lower')
hdr_ax.text(0.5, 0.70,
            'AI-Driven Pritamab Combination Therapy Selection: The ADDS Framework',
            ha='center', va='center', fontsize=19, fontweight='bold',
            color='white', transform=hdr_ax.transAxes, zorder=5)
hdr_ax.text(0.5, 0.22,
            'Anti-PrPc Monoclonal Antibody Synergy Pipeline for KRAS-Mutant Precision Oncology',
            ha='center', va='center', fontsize=12, color='#BBDEFB',
            style='italic', transform=hdr_ax.transAxes, zorder=5)
hdr_ax.axhline(0.0, color='#42A5F5', linewidth=3, alpha=0.7)

# ═══════════════════════════════════════════════════════════════════════════════
# TIER 1 — SOURCE DATA STREAMS
# ═══════════════════════════════════════════════════════════════════════════════
t1_h = T1_TOP - T1_BOT
t1_ax = fig.add_axes([MARGIN_L, T1_BOT, MID_W, t1_h], facecolor='none')
t1_ax.set_xlim(0,1); t1_ax.set_ylim(0,1); t1_ax.axis('off')

# Tier background
fbox(t1_ax, 0, 0, 1, 1, fill='#EBF5FB', edge=P['T1'], lw=4,
     r='round,pad=0.012', shadow=True, shadow_offset=(0.004,-0.006))

# Header band
fbox(t1_ax, 0, 0.86, 1, 0.14, fill=P['T1'], edge=P['T1'], lw=0,
     r='round,pad=0.0', shadow=False)
txt(t1_ax, 0.04, 0.930, 'TIER 1', fs=13, bold=True,
    color='#BBDEFB', ha='left')
txt(t1_ax, 0.145, 0.930, '—  SOURCE DATA STREAMS', fs=14, bold=True,
    color='white', ha='left')

SOURCES = [
    ('Genomic Data',
     ['Gene mutations', 'KRAS G12D / G12V / G12C', 'PRNP / PrPc expression', 'MSI · TMB · EGFR/HER2'],
     P['genomic']),
    ('Cellular Analysis',
     ['H&E segmentation', 'via AI-Cellpose', 'Nuclear density', 'PrPc IHC H-score'],
     P['cell']),
    ('Imaging Data',
     ['CT tumour detection', 'via nnU-Net', 'Anatomical saliency', 'TNM staging'],
     P['imaging']),
    ('Clinical Data',
     ['Patient demographics', 'Comorbidities', 'Prior treatments', 'Performance status'],
     P['clinic']),
]

src_bw = 0.215
src_bh = 0.72
src_xs = [0.025, 0.265, 0.505, 0.745]

for (title, bullets, color), sx in zip(SOURCES, src_xs):
    # Card
    fbox(t1_ax, sx, 0.06, src_bw, src_bh, fill='white', edge=color, lw=4,
         r='round,pad=0.012', shadow=True, shadow_offset=(0.003,-0.005))
    # Color header strip
    fbox(t1_ax, sx, 0.06+src_bh-0.18, src_bw, 0.18, fill=color, edge=color,
         lw=0, r='round,pad=0.0', shadow=False)
    txt(t1_ax, sx+src_bw/2, 0.06+src_bh-0.09, title,
        fs=11, bold=True, color='white')
    # Bullets
    for bi, b in enumerate(bullets):
        txt(t1_ax, sx+0.015, 0.06+src_bh-0.24-bi*0.135, f'• {b}',
            fs=9.2, color=P['text_m'], ha='left')

# Arrows source blocks → Tier2 (in figure coords)
arr_xs_t1 = [MARGIN_L + sx + src_bw/2 for sx in src_xs]
for ax_ in arr_xs_t1:
    arrow_down(fig, None, ax_, T1_BOT, T2_TOP, P['T1'], lw=3)

# ═══════════════════════════════════════════════════════════════════════════════
# TIER 2 — ADDS INTEGRATION ENGINE
# ═══════════════════════════════════════════════════════════════════════════════
t2_h = T2_TOP - T2_BOT
t2_ax = fig.add_axes([MARGIN_L, T2_BOT, MID_W, t2_h], facecolor='none')
t2_ax.set_xlim(0,1); t2_ax.set_ylim(0,1); t2_ax.axis('off')

fbox(t2_ax, 0, 0, 1, 1, fill='#E8F5E9', edge=P['T2'], lw=4,
     r='round,pad=0.012', shadow=True, shadow_offset=(0.004,-0.006))

fbox(t2_ax, 0, 0.88, 1, 0.12, fill=P['T2'], edge=P['T2'], lw=0,
     r='round,pad=0.0', shadow=False)
txt(t2_ax, 0.04, 0.940, 'TIER 2', fs=13, bold=True, color='#C8E6C9', ha='left')
txt(t2_ax, 0.145, 0.940, '—  ADDS INTEGRATION ENGINE', fs=14, bold=True,
    color='white', ha='left')

# ── AI Modules column ─────────────────────────────────────────────────────────
AI_COL_X, AI_COL_W = 0.01, 0.29

fbox(t2_ax, AI_COL_X, 0.02, AI_COL_W, 0.82,
     fill='#E8EAF6', edge=P['ai1'], lw=3,
     r='round,pad=0.012', shadow=False)
txt(t2_ax, AI_COL_X+AI_COL_W/2, 0.875, 'AI Modules',
    fs=12, bold=True, color=P['ai1'])

AI_MODULES = [
    ('Pathology\nAI-Cellpose',    P['ai2']),
    ('CT Imaging\nAI-nnU-Net',    P['ai3']),
    ('Genomic\nAI-PRNP',          P['ai1']),
    ('Biomarker\nAI-PrPc',        P['ai4']),
    ('Response\nAI-XGBoost',      P['ai5']),
    ('Synergy\nAI-DeepSynergy',   P['ai6']),
]

grid_positions = [(0,0),(1,0),(0,1),(1,1),(0,2),(1,2)]
ai_bw, ai_bh = 0.128, 0.215
ai_x_starts = [AI_COL_X+0.014, AI_COL_X+0.014+ai_bw+0.018]
ai_y_start = 0.56

for (col_, row_), (label, color) in zip(grid_positions, AI_MODULES):
    bx = ai_x_starts[col_]
    by = ai_y_start - row_ * (ai_bh + 0.025)
    fbox(t2_ax, bx, by, ai_bw, ai_bh, fill='white', edge=color, lw=3.5,
         r='round,pad=0.01', shadow=True, shadow_offset=(0.002,-0.003))
    # Top accent strip
    fbox(t2_ax, bx, by+ai_bh-0.058, ai_bw, 0.058, fill=color, edge=color,
         lw=0, r='round,pad=0.0', shadow=False)
    txt(t2_ax, bx+ai_bw/2, by+ai_bh/2-0.014, label, fs=9.5, bold=True, color=color)

# Arrow AI→Pipe
arrow_right(t2_ax, AI_COL_X+AI_COL_W+0.003, AI_COL_X+AI_COL_W+0.040,
            0.50, color=P['T2'], lw=2.5)

# ── Processing Pipeline column ────────────────────────────────────────────────
PIPE_COL_X, PIPE_COL_W = 0.325, 0.295

fbox(t2_ax, PIPE_COL_X, 0.02, PIPE_COL_W, 0.82,
     fill='#E3F2FD', edge=P['T1'], lw=3,
     r='round,pad=0.012', shadow=False)
txt(t2_ax, PIPE_COL_X+PIPE_COL_W/2, 0.875, 'Processing Pipeline',
    fs=12, bold=True, color=P['T1'])

PIPE_STEPS = [
    ('Data Normalisation',        (2,136,209)),
    ('Patient Stratification\nPrPc IHC + KRAS', (21,101,192)),
    ('Feature Extraction',        (2,136,209)),
    ('Energy Landscape Modelling\n\u0394G / ODE System',    (230,81,0)),
    ('Drug Interaction Modelling',  (2,136,209)),
    ('Efficacy & Synergy Scoring\n4-Model Consensus',     (46,125,50)),
]

pipe_bw = PIPE_COL_W - 0.040
pipe_bh_each = 0.107
pipe_gap = 0.018
pipe_y_start = 0.755

for i, (label, rgb) in enumerate(PIPE_STEPS):
    color_hex = '#{:02X}{:02X}{:02X}'.format(*rgb)
    py = pipe_y_start - i*(pipe_bh_each+pipe_gap)
    fbox(t2_ax, PIPE_COL_X+0.018, py, pipe_bw, pipe_bh_each,
         fill='white', edge=color_hex, lw=3.5,
         r='round,pad=0.01', shadow=True, shadow_offset=(0.002,-0.003))
    # Left accent bar
    fbox(t2_ax, PIPE_COL_X+0.018, py, 0.012, pipe_bh_each,
         fill=color_hex, edge=color_hex, lw=0,
         r='round,pad=0.0', shadow=False)
    txt(t2_ax, PIPE_COL_X+0.018+pipe_bw/2+0.006, py+pipe_bh_each/2,
        label, fs=9.5, bold=True, color=color_hex, lsp=1.35)
    # Down connector dot
    if i < len(PIPE_STEPS)-1:
        arr_y = py - pipe_gap/2
        t2_ax.annotate('', xy=(PIPE_COL_X+0.018+pipe_bw/2, arr_y),
                        xytext=(PIPE_COL_X+0.018+pipe_bw/2, py),
                        xycoords='axes fraction', textcoords='axes fraction',
                        arrowprops=dict(arrowstyle='->', color=P['arrow'],
                                        lw=2.0, mutation_scale=14), zorder=8)

# Arrow Pipe→Ensemble
arrow_right(t2_ax, PIPE_COL_X+PIPE_COL_W+0.003, PIPE_COL_X+PIPE_COL_W+0.040,
            0.50, color=P['T2'], lw=2.5)

# ── Ensemble Methods column ───────────────────────────────────────────────────
ENS_COL_X = 0.655
ENS_COL_W = 0.335

fbox(t2_ax, ENS_COL_X, 0.02, ENS_COL_W, 0.82,
     fill='#E8F5E9', edge=P['T2'], lw=3,
     r='round,pad=0.012', shadow=False)
txt(t2_ax, ENS_COL_X+ENS_COL_W/2, 0.875, 'Ensemble Methods',
    fs=12, bold=True, color=P['T2'])

ENSEMBLE = [
    ('Random Forest',      'Backbone eligibility scoring',         P['ens1']),
    ('Gradient Boosting',  'Response probability estimation',       P['ens2']),
    ('Deep Learning MLP',  'Drug-drug synergy modelling',           P['ens3']),
    ('ODE Energy Model',   '\u0394G filter \u2014 55.6% oncogenic rate \u2193', P['ens4']),
    ('4-Model Consensus',  'Bliss + Loewe + HSA + ZIP (≥0.75)',    P['ens5']),
    ('DRS Aggregation',    'Drug Recommendation Score output',      P['ens6']),
]

ens_bw = ENS_COL_W - 0.030
ens_bh = 0.107
ens_gap = 0.018
ens_y_start = 0.755

for i, (label, sub, color) in enumerate(ENSEMBLE):
    ey = ens_y_start - i*(ens_bh+ens_gap)
    fbox(t2_ax, ENS_COL_X+0.015, ey, ens_bw, ens_bh,
         fill='white', edge=color, lw=3.5,
         r='round,pad=0.01', shadow=True, shadow_offset=(0.002,-0.003))
    fbox(t2_ax, ENS_COL_X+0.015, ey, 0.012, ens_bh,
         fill=color, edge=color, lw=0, r='round,pad=0.0', shadow=False)
    txt(t2_ax, ENS_COL_X+0.015+0.025, ey+ens_bh*0.72, label,
        fs=10, bold=True, color=color, ha='left')
    txt(t2_ax, ENS_COL_X+0.015+0.025, ey+ens_bh*0.28, sub,
        fs=8.8, color=P['text_l'], ha='left')

# Arrows Tier2 → Tier3
for fx in [MARGIN_L+0.13, MARGIN_L+0.50, MARGIN_L+0.82]:
    arrow_down(fig, None, fx, T2_BOT, T3_TOP, P['T2'], lw=3)

# ═══════════════════════════════════════════════════════════════════════════════
# TIER 3 — OUTPUT: PRITAMAB COMBINATION RECOMMENDATIONS
# ═══════════════════════════════════════════════════════════════════════════════
t3_h = T3_TOP - T3_BOT
t3_ax = fig.add_axes([MARGIN_L, T3_BOT, MID_W, t3_h], facecolor='none')
t3_ax.set_xlim(0,1); t3_ax.set_ylim(0,1); t3_ax.axis('off')

fbox(t3_ax, 0, 0, 1, 1, fill='#FFF3E0', edge=P['T3'], lw=4,
     r='round,pad=0.012', shadow=True, shadow_offset=(0.004,-0.006))
fbox(t3_ax, 0, 0.89, 1, 0.11, fill=P['T3'], edge=P['T3'], lw=0,
     r='round,pad=0.0', shadow=False)
txt(t3_ax, 0.04, 0.945, 'TIER 3', fs=13, bold=True, color='#FFE0B2', ha='left')
txt(t3_ax, 0.145, 0.945,
    '—  OUTPUT — PRITAMAB COMBINATION RECOMMENDATIONS  (DRS Ranking)',
    fs=13, bold=True, color='white', ha='left')

# ── Three recommendation cards ─────────────────────────────────────────────────
CARDS = [
    {
        'rank':     '#1  PRIMARY RECOMMENDATION',
        'title':    'Pritamab + FOLFOX',
        'sub':      '(5-FU + Leucovorin + Oxaliplatin)',
        'drs':      'DRS: 0.893',
        'bliss':    'Bliss Synergy: +21.0',
        'ec50':     'EC50 \u2212 24.0%',
        'points': [
            '• Pi3K/AKT target: RPSA signalosome blockade',
            '• PrPc-HIGH (H-score \u226550) + KRAS-mutant (any allele)',
            '    G12D (H-score 142)  \u00b7  G12V (138)  \u00b7  G13D (124)',
            '• 24.0% FOLFOX dose reduction \u2192 Cumulative toxicity \u2193',
            '• Loewe DRI  5-FU: 1.34  \u00b7  Oxaliplatin: 1.34',
        ],
        'pills':   ['FDA Guidelines', 'NCCN Protocol'],
        'color':   P['prim'],
        'light':   '#DBEAFE',
        'x0':      0.015,
    },
    {
        'rank':     '#2  ALTERNATIVE REGIMEN',
        'title':    'Pritamab + Sotorasib',
        'sub':      '(KRAS G12C Covalent Inhibitor)',
        'drs':      'DRS: 0.882',
        'bliss':    'Bliss Synergy: +22.5',
        'ec50':     'EC50 \u2212 24.7%',
        'points': [
            '• Dual-axis: RPSA-PrPc block + G12C direct inhibition',
            '• KRAS G12C-mutant (~12\u201313% CRC) + PrPc-HIGH',
            '    VEGF anti-angiogenic target for metastatic CRC',
            '• Addresses RTK-bypass escape via RPSA route',
            '• Triple option: + SHP2i \u2192 DRS 0.835 \u00b7 Bliss +28.0',
        ],
        'pills':   ['FDA Approved', 'ESMO Guideline'],
        'color':   P['alt'],
        'light':   '#DCFCE7',
        'x0':      0.348,
    },
    {
        'rank':     '#3  CONDITIONAL REGIMEN',
        'title':    'Pritamab + FOLFOXIRI',
        'sub':      '(Triplet Intensification)',
        'drs':      'DRS: 0.784',
        'bliss':    'Bliss Synergy: +22.0',
        'ec50':     'EC50 \u2212 26.1%',
        'points': [
            '• Triplet intensification + PrPc chemosensitisation',
            '• KRAS-mutant \u00b7 PrPc-HIGH \u00b7 Adequate PS (ECOG 0\u20131)',
            '    Hepatic metastasis conversion \u00b7 Younger patients',
            '• Clinical eligibility review REQUIRED before use',
            '• Bevacizumab triplet variant (TRIBE2) as option',
        ],
        'pills':   ['ESMO Guideline', 'Eligibility Review'],
        'color':   P['cond'],
        'light':   '#F3E8FF',
        'x0':      0.681,
    },
]

card_w = 0.296
card_y0 = 0.035
card_h  = 0.840

for card in CARDS:
    cx = card['x0']
    color = card['color']
    light = card['light']

    # Card shadow + white fill
    fbox(t3_ax, cx, card_y0, card_w, card_h,
         fill='white', edge=color, lw=5,
         r='round,pad=0.014', shadow=True, shadow_offset=(0.004,-0.006))

    # Gradient header
    cmap_card = LinearSegmentedColormap.from_list('card',
        [color, '#{:02X}{:02X}{:02X}'.format(
            *[max(0, int(int(color.lstrip('#')[i*2:i*2+2], 16)*0.7)) for i in range(3)])])
    card_img = np.linspace(0, 1, 128).reshape(1, 128)
    fake_ax = t3_ax.inset_axes(
        [cx, card_y0+card_h-0.175, card_w, 0.175],
        transform=t3_ax.transAxes)
    fake_ax.imshow(card_img, aspect='auto', extent=[0,1,0,1],
                   cmap=cmap_card, origin='lower', zorder=4)
    fake_ax.axis('off')

    # Rank badge (white circle)
    circ = plt.Circle((cx+0.058, card_y0+card_h-0.088), 0.040,
                       transform=t3_ax.transAxes, color='white', zorder=7)
    t3_ax.add_patch(circ)
    rank_num = card['rank'].split()[0]
    txt(t3_ax, cx+0.058, card_y0+card_h-0.088, rank_num,
        fs=10.5, bold=True, color=color)
    # Rank label
    rank_label = ' '.join(card['rank'].split()[1:])
    txt(t3_ax, cx+card_w/2+0.026, card_y0+card_h-0.088, rank_label,
        fs=9.5, bold=True, color='white')

    # Combination title
    txt(t3_ax, cx+card_w/2, card_y0+card_h-0.220, card['title'],
        fs=14, bold=True, color=color)
    txt(t3_ax, cx+card_w/2, card_y0+card_h-0.270, card['sub'],
        fs=9.5, color=P['text_m'])

    # DRS badge
    fbox(t3_ax, cx+0.028, card_y0+card_h-0.345, card_w-0.056, 0.062,
         fill='#FFFDE7', edge=P['gold'], lw=3,
         r='round,pad=0.008', shadow=False)
    txt(t3_ax, cx+card_w/2, card_y0+card_h-0.314, card['drs'],
        fs=15, bold=True, color='#E65100')

    # Synergy row
    half_w = (card_w-0.056)/2 - 0.008
    fbox(t3_ax, cx+0.028, card_y0+card_h-0.420, half_w, 0.060,
         fill=light, edge=P['T2'], lw=2.5, r='round,pad=0.008', shadow=False)
    txt(t3_ax, cx+0.028+half_w/2, card_y0+card_h-0.390,
        card['bliss'], fs=9.5, bold=True, color=P['T2'])

    fbox(t3_ax, cx+0.028+half_w+0.016, card_y0+card_h-0.420, half_w, 0.060,
         fill=light, edge=P['prim'], lw=2.5, r='round,pad=0.008', shadow=False)
    txt(t3_ax, cx+0.028+half_w+0.016+half_w/2, card_y0+card_h-0.390,
        card['ec50'], fs=9.5, bold=True, color=P['prim'])

    # Divider
    t3_ax.plot([cx+0.02, cx+card_w-0.02], [card_y0+card_h-0.432, card_y0+card_h-0.432],
               color='#E0E0E0', lw=1.5, transform=t3_ax.transAxes, zorder=4)

    # Bullet points
    for bi, pt in enumerate(card['points']):
        txt(t3_ax, cx+0.016, card_y0+card_h-0.482-bi*0.110, pt,
            fs=9.0, color=P['text_m'], ha='left')

    # Pills at bottom
    pill_x = cx + 0.022
    for ptext in card['pills']:
        pill(t3_ax, pill_x, card_y0+0.058, ptext, color, fs=8.0)
        pill_x += 0.148

# ═══════════════════════════════════════════════════════════════════════════════
# INFRASTRUCTURE FOOTER
# ═══════════════════════════════════════════════════════════════════════════════
inf_ax = fig.add_axes([MARGIN_L, INFRA_BOT+0.005, MID_W, INFRA_TOP-INFRA_BOT], facecolor='none')
inf_ax.set_xlim(0,1); inf_ax.set_ylim(0,1); inf_ax.axis('off')

fbox(inf_ax, 0, 0, 1, 1, fill='#ECEFF1', edge='#90A4AE', lw=2.5,
     r='round,pad=0.012', shadow=True, shadow_offset=(0.002,-0.003))

infra = [
    ('Serum PrPc Liquid Biopsy  AUC=0.777',                 0.02),
    ('IHC Biomarker: PrPc H-score \u226550 + KRAS mut.',    0.27),
    ('Pharmacogenomics: 113 drugs | 59 synergy pairs',       0.55),
    ('ADDS v5.3  |  TCGA n=2,285  |  KR Patent Jan 2026',   0.76),
]
for label, xi in infra:
    txt(inf_ax, xi, 0.50, label, fs=9, color=P['text_m'], ha='left')
    if xi > 0.02:
        inf_ax.plot([xi-0.01, xi-0.01], [0.1, 0.9], color='#B0BEC5', lw=1.5,
                    transform=inf_ax.transAxes, zorder=4)

# Caption
fig.text(0.5, 0.004,
         'Figure: AI-Driven Pritamab Pipeline  |  Inha University Hospital × ADDS Precision Oncology Framework  |  Nature Communications 2026',
         ha='center', va='bottom', fontsize=8.5, color=P['text_l'], style='italic')

for fx in [MARGIN_L+0.12, MARGIN_L+0.49, MARGIN_L+0.82]:
    arrow_down(fig, None, fx, T2_BOT, T3_TOP, P['T2'], lw=3)

# ── Save ───────────────────────────────────────────────────────────────────────
OUT_DIR = r'f:\ADDS\figures'
PNG_PATH = os.path.join(OUT_DIR, 'pritamab_infographic_v2.png')
PDF_PATH = os.path.join(OUT_DIR, 'pritamab_infographic_v2.pdf')

# Save as PDF (true vector)
fig.savefig(PDF_PATH, format='pdf', bbox_inches='tight', facecolor=P['bg'])
print(f'PDF saved: {PDF_PATH}')

# Save as PNG (300 DPI, rasterized from vector)
fig.savefig(PNG_PATH, format='png', dpi=DPI, bbox_inches='tight',
            facecolor=P['bg'], edgecolor='none')
print(f'PNG saved: {PNG_PATH}  ({W_IN*DPI:.0f} x {H_IN*DPI:.0f} px @ {DPI} dpi)')
plt.close()
