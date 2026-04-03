"""
Pritamab AI Pipeline — 3D Photoshop-Style Infographic
Dark glossy theme: multilayer shadows, gradient fills, glass highlights, neon glow
Output: PDF (vector) + PNG 300DPI
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Ellipse, Circle
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.patheffects as pe
import numpy as np
import os

# ═══════════════════════════════════════════════════════════════════════════════
# DARK NEON PALETTE
# ═══════════════════════════════════════════════════════════════════════════════
BG_TOP    = '#050D1A'   # near-black deep blue top
BG_BOT    = '#0A1628'   # slightly lighter bottom
PANEL_BG  = '#0D1F35'   # card interiors
PANEL_BDR = '#1C3A5C'   # subtle panel border

# Tier neon accents
T1_NE  = '#00B4D8'  # cyan
T2_NE  = '#06D6A0'  # teal-green
T3_NE  = '#FF9F1C'  # warm gold

# Source block neons
C_DNA  = '#BB86FC'  # purple
C_CELL = '#03DAC6'  # teal
C_IMG  = '#FF6B9D'  # pink
C_CLI  = '#4FC3F7'  # light blue

# Pipeline/ensemble neons
C_P    = '#00B4D8'
C_E    = '#06D6A0'
C_AI   = [('#BB86FC','#7C4DFF'),('#03DAC6','#00838F'),('#FF6B9D','#C51162'),
          ('#FF8C42','#E65100'),('#4FC3F7','#0277BD'),('#06D6A0','#00695C')]

# Card colors
C_PRIM = ('#2979FF','#1A237E')   # (neon, dark)
C_ALT  = ('#00E676','#1B5E20')
C_COND = ('#E040FB','#4A148C')

WHITE  = '#FFFFFF'
SILVER = '#B0BEC5'
DIM    = '#546E7A'

# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE
# ═══════════════════════════════════════════════════════════════════════════════
W_IN, H_IN = 22, 32
DPI = 300
fig = plt.figure(figsize=(W_IN, H_IN), facecolor=BG_TOP)

# ── Full background gradient ────────────────────────────────────────────────────
bg_ax = fig.add_axes([0,0,1,1], zorder=0, facecolor=BG_TOP)
bg_ax.axis('off')
bg_img = np.linspace(0, 1, 512).reshape(512, 1)
cmap_bg = LinearSegmentedColormap.from_list('bg', [BG_BOT, BG_TOP])
bg_ax.imshow(bg_img, aspect='auto', extent=[0,1,0,1],
             cmap=cmap_bg, origin='lower', zorder=0, alpha=1.0)
# faint grid lines for depth
for gx in np.linspace(0.05, 0.95, 18):
    bg_ax.axvline(gx, color='#0D2137', lw=0.5, alpha=0.5)
for gy in np.linspace(0.05, 0.95, 24):
    bg_ax.axhline(gy, color='#0D2137', lw=0.5, alpha=0.5)

# ═══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════════════
def hex_darken(h, f=0.5):
    h = h.lstrip('#')
    r,g,b = int(h[0:2],16),int(h[2:4],16),int(h[4:6],16)
    return '#{:02X}{:02X}{:02X}'.format(int(r*f),int(g*f),int(b*f))

def hex_lighten(h, f=0.5):
    h = h.lstrip('#')
    r,g,b = int(h[0:2],16),int(h[2:4],16),int(h[4:6],16)
    return '#{:02X}{:02X}{:02X}'.format(
        min(255,int(r+(255-r)*f)),min(255,int(g+(255-g)*f)),min(255,int(b+(255-b)*f)))

def glow_box(ax, x, y, w, h, neon, depth=4, r='round,pad=0.015'):
    """Multi-layer glow shadow + dark fill + gradient top highlight."""
    alphas = [0.06, 0.10, 0.14, 0.18]
    offsets = [depth*3.5, depth*2.2, depth*1.3, depth*0.6]
    for alpha, off in zip(alphas, offsets):
        off_n = off * 0.001
        sh = FancyBboxPatch((x-off_n, y-off_n*1.5), w+off_n*2, h+off_n*2,
                             boxstyle=r, facecolor=neon, edgecolor='none',
                             alpha=alpha, transform=ax.transAxes,
                             zorder=2, clip_on=False)
        ax.add_patch(sh)
    # Main box dark fill
    box = FancyBboxPatch((x, y), w, h, boxstyle=r,
                          facecolor=PANEL_BG, edgecolor=neon,
                          linewidth=1.8, transform=ax.transAxes,
                          zorder=3, clip_on=False)
    ax.add_patch(box)
    # Highlight strip at top (glass effect)
    hi_h = min(h*0.18, 0.06)
    hi = FancyBboxPatch((x+0.004, y+h-hi_h-0.003), w-0.008, hi_h,
                         boxstyle='round,pad=0.006',
                         facecolor='white', edgecolor='none',
                         alpha=0.07, transform=ax.transAxes,
                         zorder=4, clip_on=False)
    ax.add_patch(hi)

def gradient_band(ax, x, y, w, h, color_l, color_r, alpha=1.0, zorder=4):
    """Horizontal gradient fill using inset_axes imshow."""
    ia = ax.inset_axes([x, y, w, h], transform=ax.transAxes)
    ia.imshow(np.linspace(0,1,128).reshape(1,128), aspect='auto', extent=[0,1,0,1],
              cmap=LinearSegmentedColormap.from_list('g',[color_l,color_r]),
              origin='lower', alpha=alpha)
    ia.axis('off')
    return ia

def neon_text(ax, x, y, s, fs=11, bold=True, color='#FFFFFF', neon=None,
              ha='center', va='center', lsp=1.4):
    fw = 'bold' if bold else 'normal'
    effects = []
    if neon:
        effects = [pe.withStroke(linewidth=4, foreground=neon)]
    t = ax.text(x, y, s, fontsize=fs, fontweight=fw, color=color,
                ha=ha, va=va, transform=ax.transAxes,
                linespacing=lsp, zorder=8)
    if effects:
        t.set_path_effects(effects)
    return t

def neon_arrow_down(fig, xf, y0, y1, color, lw=3.0):
    fig.add_artist(FancyArrowPatch(
        (xf, y0), (xf, y1),
        transform=fig.transFigure,
        arrowstyle='simple,head_width=0.010,head_length=0.014',
        color=color, linewidth=lw, zorder=12,
        path_effects=[pe.withStroke(linewidth=lw*2.5, foreground=color, alpha=0.25)]))

def neon_arrow_right(ax, x0, x1, cy, color, lw=2.5):
    ax.annotate('', xy=(x1, cy), xytext=(x0, cy),
                xycoords='axes fraction', textcoords='axes fraction',
                arrowprops=dict(
                    arrowstyle='simple,head_width=0.5,head_length=0.025',
                    color=color, lw=lw), zorder=9)

def circle_icon(ax, cx, cy, r, neon, label, fs=9):
    """Neon-glow circle icon."""
    for a, scale in [(0.12, 2.6),(0.20, 1.8),(0.35, 1.3)]:
        ec = Ellipse((cx, cy), r*scale*2, r*scale*2,
                     transform=ax.transAxes,
                     facecolor=neon, edgecolor='none', alpha=a, zorder=3)
        ax.add_patch(ec)
    ci = Ellipse((cx, cy), r*2, r*2,
                  transform=ax.transAxes,
                  facecolor=PANEL_BG, edgecolor=neon, linewidth=2.0,
                  alpha=1.0, zorder=4)
    ax.add_patch(ci)
    ax.text(cx, cy, label, ha='center', va='center',
            fontsize=fs, fontweight='bold', color=neon,
            transform=ax.transAxes, zorder=5)

# ═══════════════════════════════════════════════════════════════════════════════
# LAYOUT Y positions (figure fraction, 0=bottom, 1=top)
# ═══════════════════════════════════════════════════════════════════════════════
ML = 0.025
MW = 0.950

HDR_T = 1.000; HDR_B = 0.953
T1_T  = 0.940; T1_B  = 0.730
T2_T  = 0.716; T2_B  = 0.365
T3_T  = 0.350; T3_B  = 0.038
INF_T = 0.032; INF_B = 0.005

# ═══════════════════════════════════════════════════════════════════════════════
# HEADER
# ═══════════════════════════════════════════════════════════════════════════════
hdr_ax = fig.add_axes([0, HDR_B, 1, HDR_T-HDR_B], facecolor='none')
hdr_ax.set_xlim(0,1); hdr_ax.set_ylim(0,1); hdr_ax.axis('off')

gradient_band(hdr_ax, 0, 0, 1, 1, '#0D2845', '#071A30', alpha=1.0)

# Decorative neon lines
for y_l, col, lw_ in [(0.95, T1_NE, 1.5),(0.05, T1_NE, 0.8)]:
    hdr_ax.plot([0, 1], [y_l, y_l], color=col, lw=lw_, alpha=0.6,
                transform=hdr_ax.transAxes, zorder=4)

t = hdr_ax.text(0.5, 0.68, 'AI-Driven Pritamab Combination Therapy:  The ADDS Framework',
                ha='center', va='center', fontsize=20, fontweight='bold',
                color=WHITE, transform=hdr_ax.transAxes, zorder=5)
t.set_path_effects([pe.withStroke(linewidth=8, foreground=T1_NE, alpha=0.4)])

hdr_ax.text(0.5, 0.20,
            'Anti-PrPc mAb Synergy Pipeline  ·  KRAS-Mutant Precision Oncology  ·  ADDS v5.3',
            ha='center', va='center', fontsize=12, color=T1_NE,
            style='italic', transform=hdr_ax.transAxes, zorder=5)

# ═══════════════════════════════════════════════════════════════════════════════
# TIER 1 — SOURCE DATA STREAMS
# ═══════════════════════════════════════════════════════════════════════════════
t1_ax = fig.add_axes([ML, T1_B, MW, T1_T-T1_B], facecolor='none')
t1_ax.set_xlim(0,1); t1_ax.set_ylim(0,1); t1_ax.axis('off')

# Tier background
glow_box(t1_ax, 0, 0, 1, 1, T1_NE, depth=6, r='round,pad=0.010')
gradient_band(t1_ax, 0.001, 0.860, 0.998, 0.130, T1_NE, hex_darken(T1_NE, 0.55), alpha=0.92)

t = t1_ax.text(0.04, 0.925, 'TIER 1', fontsize=12, fontweight='bold',
               color=hex_darken(T1_NE, 0.3), transform=t1_ax.transAxes, zorder=8)
t1_ax.text(0.155, 0.925, 'SOURCE DATA STREAMS', fontsize=13, fontweight='bold',
           color=WHITE, transform=t1_ax.transAxes, zorder=8)

SOURCES = [
    ('Genomic\nData',      ['KRAS G12D/V/C','PRNP / PrPc','MSI · TMB','EGFR / HER2'],    C_DNA,  'DNA'),
    ('Cellular\nAnalysis', ['H&E AI-Cellpose','PrPc IHC H-score','Nuclear density','Stromal frac.'], C_CELL, 'LAB'),
    ('Imaging\nData',      ['CT detection','AI nnU-Net','Saliency filter','TNM staging'],  C_IMG,  'IMG'),
    ('Clinical\nData',     ['Demographics','ECOG 0–2 status','Prior treatments','Comorbidities'],   C_CLI,  ' Rx '),
]

src_xs = [0.024, 0.265, 0.506, 0.747]
src_bw, src_bh = 0.218, 0.740

for (title, bullets, neon, icon), sx in zip(SOURCES, src_xs):
    glow_box(t1_ax, sx, 0.060, src_bw, src_bh, neon, depth=3)

    # Gradient top header
    gradient_band(t1_ax, sx+0.002, 0.060+src_bh-0.230,
                  src_bw-0.004, 0.228, neon, hex_darken(neon, 0.4), alpha=0.90)

    # Icon circle
    circle_icon(t1_ax, sx+0.055, 0.060+src_bh-0.115,
                0.038, neon, icon, fs=8)

    # Title
    t1_ax.text(sx+src_bw/2+0.030, 0.060+src_bh-0.108, title,
               ha='center', va='center', fontsize=10.5, fontweight='bold',
               color=WHITE, transform=t1_ax.transAxes, zorder=8, linespacing=1.3)

    # Bullets
    for bi, b in enumerate(bullets):
        t1_ax.text(sx+0.018, 0.060+src_bh-0.275-bi*0.128, f'• {b}',
                   ha='left', va='top', fontsize=8.8, color=SILVER,
                   transform=t1_ax.transAxes, zorder=8)

# Tier label badge
glow_box(t1_ax, 0.005, 0.862, 0.110, 0.090, T1_NE, depth=2, r='round,pad=0.008')
t1_ax.text(0.060, 0.907, 'INPUT', ha='center', va='center', fontsize=9,
           fontweight='bold', color=T1_NE, transform=t1_ax.transAxes, zorder=8)

# Arrows T1 → T2
for sx in src_xs:
    xf = ML + sx + src_bw/2
    neon_arrow_down(fig, xf, T1_B, T2_T, T1_NE, lw=2.5)

# ═══════════════════════════════════════════════════════════════════════════════
# TIER 2 — ADDS INTEGRATION ENGINE
# ═══════════════════════════════════════════════════════════════════════════════
t2_ax = fig.add_axes([ML, T2_B, MW, T2_T-T2_B], facecolor='none')
t2_ax.set_xlim(0,1); t2_ax.set_ylim(0,1); t2_ax.axis('off')

glow_box(t2_ax, 0, 0, 1, 1, T2_NE, depth=6, r='round,pad=0.010')
gradient_band(t2_ax, 0.001, 0.885, 0.998, 0.106, T2_NE, hex_darken(T2_NE, 0.45), alpha=0.92)

t2_ax.text(0.04, 0.935, 'TIER 2', fontsize=12, fontweight='bold',
           color=hex_darken(T2_NE, 0.3), transform=t2_ax.transAxes, zorder=8)
t2_ax.text(0.155, 0.935, 'ADDS INTEGRATION ENGINE', fontsize=13, fontweight='bold',
           color=WHITE, transform=t2_ax.transAxes, zorder=8)

# ── Column 1: AI Modules ───────────────────────────────────────────────────────
AI_X, AI_W = 0.010, 0.278
glow_box(t2_ax, AI_X, 0.025, AI_W, 0.845, C_DNA, depth=2, r='round,pad=0.010')
gradient_band(t2_ax, AI_X+0.002, 0.025+0.845-0.082,
              AI_W-0.004, 0.080, C_DNA, hex_darken(C_DNA, 0.5), alpha=0.85)
t2_ax.text(AI_X+AI_W/2, 0.025+0.845-0.040, 'AI Modules',
           ha='center', va='center', fontsize=11, fontweight='bold',
           color=WHITE, transform=t2_ax.transAxes, zorder=8)

AI_MODS = [
    ('Pathology\nAI-Cellpose',   C_AI[0]),
    ('CT Imaging\nAI-nnU-Net',   C_AI[1]),
    ('Genomic\nAI-PRNP',         C_AI[2]),
    ('Biomarker\nAI-PrPc',       C_AI[3]),
    ('Response\nAI-XGBoost',     C_AI[4]),
    ('Synergy\nDeepSynergy',     C_AI[5]),
]

ai_grid = [(0,0),(1,0),(0,1),(1,1),(0,2),(1,2)]
ai_bw, ai_bh = 0.122, 0.195
ai_x0 = [AI_X+0.012, AI_X+0.012+ai_bw+0.018]
ai_y0 = 0.025+0.845-0.140

for (col_, row_), (label, (neon, dark)) in zip(ai_grid, AI_MODS):
    bx = ai_x0[col_]
    by = ai_y0 - row_*(ai_bh+0.022)
    glow_box(t2_ax, bx, by, ai_bw, ai_bh, neon, depth=2)
    gradient_band(t2_ax, bx+0.002, by+ai_bh-0.055, ai_bw-0.004, 0.053,
                  neon, dark, alpha=0.85)
    t2_ax.text(bx+ai_bw/2, by+ai_bh/2-0.012, label,
               ha='center', va='center', fontsize=9, fontweight='bold',
               color=neon, transform=t2_ax.transAxes, zorder=8, linespacing=1.35)

# Arrow AI→Pipe
neon_arrow_right(t2_ax, AI_X+AI_W+0.002, AI_X+AI_W+0.038, 0.50, T2_NE, lw=2.5)

# ── Column 2: Processing Pipeline ─────────────────────────────────────────────
PIPE_X, PIPE_W = 0.300, 0.284
glow_box(t2_ax, PIPE_X, 0.025, PIPE_W, 0.845, C_P, depth=2, r='round,pad=0.010')
gradient_band(t2_ax, PIPE_X+0.002, 0.025+0.845-0.082,
              PIPE_W-0.004, 0.080, C_P, hex_darken(C_P, 0.45), alpha=0.85)
t2_ax.text(PIPE_X+PIPE_W/2, 0.025+0.845-0.040, 'Processing Pipeline',
           ha='center', va='center', fontsize=11, fontweight='bold',
           color=WHITE, transform=t2_ax.transAxes, zorder=8)

PIPE_STEPS = [
    ('Data Normalisation',                    C_P),
    ('Patient Stratification\nPrPc + KRAS',  '#1565C0'),
    ('Feature Extraction',                    C_P),
    ('Energy Landscape ΔG\nODE Modelling',   T3_NE),
    ('Drug Interaction\nModelling',           C_P),
    ('Synergy Scoring\n4-Model Consensus',    T2_NE),
]

pbw = PIPE_W - 0.040
pbh = 0.098
pgap = 0.018
py0 = 0.025+0.845-0.135

for i, (label, neon) in enumerate(PIPE_STEPS):
    py = py0 - i*(pbh+pgap)
    glow_box(t2_ax, PIPE_X+0.020, py, pbw, pbh, neon, depth=2)
    # Left accent bar
    acc = FancyBboxPatch((PIPE_X+0.020, py), 0.010, pbh,
                          boxstyle='round,pad=0.0',
                          facecolor=neon, edgecolor='none',
                          transform=t2_ax.transAxes, zorder=4, alpha=0.9)
    t2_ax.add_patch(acc)
    t2_ax.text(PIPE_X+0.020+pbw/2+0.005, py+pbh/2, label,
               ha='center', va='center', fontsize=9.5, fontweight='bold',
               color=neon, transform=t2_ax.transAxes, zorder=8, linespacing=1.3)
    if i < len(PIPE_STEPS)-1:
        mid_x = PIPE_X+0.020+pbw/2
        t2_ax.annotate('', xy=(mid_x, py), xytext=(mid_x, py-pgap),
                        xycoords='axes fraction', textcoords='axes fraction',
                        arrowprops=dict(arrowstyle='->',
                                        color=SILVER, lw=1.8,
                                        mutation_scale=12), zorder=8)

# Arrow Pipe→Ensemble
neon_arrow_right(t2_ax, PIPE_X+PIPE_W+0.002, PIPE_X+PIPE_W+0.038, 0.50, T2_NE, lw=2.5)

# ── Column 3: Ensemble Methods ────────────────────────────────────────────────
ENS_X, ENS_W = 0.618, 0.370
glow_box(t2_ax, ENS_X, 0.025, ENS_W, 0.845, T2_NE, depth=2, r='round,pad=0.010')
gradient_band(t2_ax, ENS_X+0.002, 0.025+0.845-0.082,
              ENS_W-0.004, 0.080, T2_NE, hex_darken(T2_NE, 0.45), alpha=0.85)
t2_ax.text(ENS_X+ENS_W/2, 0.025+0.845-0.040, 'Ensemble Methods',
           ha='center', va='center', fontsize=11, fontweight='bold',
           color=WHITE, transform=t2_ax.transAxes, zorder=8)

ENSEMBLE = [
    ('Random Forest',      'Backbone eligibility',         '#43A047'),
    ('Gradient Boosting',  'Response probability',          '#1E88E5'),
    ('Deep Learning MLP',  'Drug synergy modelling',        '#AB47BC'),
    ('ODE Energy Model',   'ΔG filter · 55.6% rate ↓',     T3_NE),
    ('4-Model Consensus',  'Bliss+Loewe+HSA+ZIP ≥0.75',   T2_NE),
    ('DRS Aggregation',    'Drug Recommendation Score',     '#EF5350'),
]

ebw = ENS_W-0.030
ebh = 0.098
egap = 0.018
ey0 = 0.025+0.845-0.135

for i, (label, sub, neon) in enumerate(ENSEMBLE):
    ey = ey0 - i*(ebh+egap)
    glow_box(t2_ax, ENS_X+0.014, ey, ebw, ebh, neon, depth=2)
    acc = FancyBboxPatch((ENS_X+0.014, ey), 0.010, ebh,
                          boxstyle='round,pad=0.0',
                          facecolor=neon, edgecolor='none',
                          transform=t2_ax.transAxes, zorder=4, alpha=0.9)
    t2_ax.add_patch(acc)
    t2_ax.text(ENS_X+0.034, ey+ebh*0.72, label,
               ha='left', va='center', fontsize=10, fontweight='bold',
               color=neon, transform=t2_ax.transAxes, zorder=8)
    t2_ax.text(ENS_X+0.034, ey+ebh*0.28, sub,
               ha='left', va='center', fontsize=8.8, color=SILVER,
               transform=t2_ax.transAxes, zorder=8)

# Arrows T2 → T3
for xf in [ML+0.12, ML+0.49, ML+0.82]:
    neon_arrow_down(fig, xf, T2_B, T3_T, T2_NE, lw=2.5)

# ═══════════════════════════════════════════════════════════════════════════════
# TIER 3 — OUTPUT: PRITAMAB COMBINATION RECOMMENDATIONS
# ═══════════════════════════════════════════════════════════════════════════════
t3_ax = fig.add_axes([ML, T3_B, MW, T3_T-T3_B], facecolor='none')
t3_ax.set_xlim(0,1); t3_ax.set_ylim(0,1); t3_ax.axis('off')

glow_box(t3_ax, 0, 0, 1, 1, T3_NE, depth=6, r='round,pad=0.010')
gradient_band(t3_ax, 0.001, 0.905, 0.998, 0.086, T3_NE, hex_darken(T3_NE, 0.45), alpha=0.92)

t3_ax.text(0.04, 0.947, 'TIER 3', fontsize=12, fontweight='bold',
           color=hex_darken(T3_NE, 0.3), transform=t3_ax.transAxes, zorder=8)
t3_ax.text(0.155, 0.947,
           'OUTPUT — PRITAMAB COMBINATION RECOMMENDATIONS  (DRS Ranking)',
           fontsize=12, fontweight='bold', color=WHITE, transform=t3_ax.transAxes, zorder=8)

# ── Three recommendation cards ─────────────────────────────────────────────────
CARDS = [
    {
        'rank': '#1',
        'hdr':  'PRIMARY RECOMMENDATION',
        'title':'Pritamab + FOLFOX',
        'sub':  '5-FU + Leucovorin + Oxaliplatin',
        'drs':  'DRS: 0.893',
        'm1':   'Bliss Synergy: +21.0',
        'm2':   'EC50 −24.0%',
        'pts': ['>> Pi3K/AKT target · RPSA signalosome blockade',
                '>> PrPc-HIGH (H-score ≥50) + KRAS-mutant G12D/V/G13D',
                '>> 24.0% FOLFOX dose reduction → Toxicity ↓',
                '>> Loewe DRI 5-FU: 1.34  ·  Oxaliplatin: 1.34'],
        'foot':['FDA Guidelines','NCCN Protocol'],
        'neon': C_PRIM[0],
        'dark': C_PRIM[1],
        'x0':  0.012,
    },
    {
        'rank': '#2',
        'hdr':  'ALTERNATIVE REGIMEN',
        'title':'Pritamab + Sotorasib',
        'sub':  'KRAS G12C Covalent Inhibitor',
        'drs':  'DRS: 0.882',
        'm1':   'Bliss Synergy: +22.5',
        'm2':   'EC50 −24.7%',
        'pts': ['>> Dual-axis: RPSA-PrPc + G12C covalent inhibition',
                '>> KRAS G12C-mutant (~12–13% CRC) + PrPc-HIGH',
                '>> Addresses RTK-bypass escape via RPSA route',
                '>> Triple option: +SHP2i → DRS 0.835 · Bliss +28.0'],
        'foot':['FDA Approved','ESMO Guideline'],
        'neon': C_ALT[0],
        'dark': C_ALT[1],
        'x0':  0.344,
    },
    {
        'rank': '#3',
        'hdr':  'CONDITIONAL REGIMEN',
        'title':'Pritamab + FOLFOXIRI',
        'sub':  'Triplet Intensification',
        'drs':  'DRS: 0.784',
        'm1':   'Bliss Synergy: +22.0',
        'm2':   'EC50 −26.1%',
        'pts': ['>> Triplet intensification + PrPc chemosensitisation',
                '>> KRAS-mutant · PrPc-HIGH · ECOG 0–1 required',
                '>> Hepatic met. conversion · Younger patients',
                '>> Clinical eligibility review REQUIRED'],
        'foot':['ESMO Guideline','Eligibility Review'],
        'neon': C_COND[0],
        'dark': C_COND[1],
        'x0':  0.676,
    },
]

card_w = 0.302
card_y0 = 0.030
card_h  = 0.855

for card in CARDS:
    cx   = card['x0']
    neon = card['neon']
    dark = card['dark']

    # Multi-layer neon glow card
    glow_box(t3_ax, cx, card_y0, card_w, card_h, neon, depth=4, r='round,pad=0.012')

    # Gradient header band
    gradient_band(t3_ax, cx+0.002, card_y0+card_h-0.198, card_w-0.004, 0.195,
                  neon, dark, alpha=0.95)

    # Rank badge circle
    for a, sc in [(0.18, 2.2),(0.30, 1.6),(0.50, 1.2),(1.0, 1.0)]:
        bg_c = WHITE if a == 1.0 else neon
        circ = Circle((cx+0.060, card_y0+card_h-0.098), 0.038*sc,
                       transform=t3_ax.transAxes,
                       facecolor=bg_c, edgecolor='none', alpha=a, zorder=5+int(a))
        t3_ax.add_patch(circ)
    t3_ax.text(cx+0.060, card_y0+card_h-0.098, card['rank'],
               ha='center', va='center', fontsize=11, fontweight='bold',
               color=neon, transform=t3_ax.transAxes, zorder=9)

    # Header label
    t3_ax.text(cx+card_w/2+0.024, card_y0+card_h-0.098, card['hdr'],
               ha='center', va='center', fontsize=9.5, fontweight='bold',
               color=WHITE, transform=t3_ax.transAxes, zorder=8)

    # Combination title
    t = t3_ax.text(cx+card_w/2, card_y0+card_h-0.248, card['title'],
                   ha='center', va='center', fontsize=13.5, fontweight='bold',
                   color=WHITE, transform=t3_ax.transAxes, zorder=8)
    t.set_path_effects([pe.withStroke(linewidth=5, foreground=neon, alpha=0.45)])
    t3_ax.text(cx+card_w/2, card_y0+card_h-0.290, card['sub'],
               ha='center', va='center', fontsize=9.5, color=SILVER,
               transform=t3_ax.transAxes, zorder=8)

    # SRS Score  — gold metallic badge
    gradient_band(t3_ax, cx+0.030, card_y0+card_h-0.360, card_w-0.060, 0.058,
                  '#F57F17', '#E65100', alpha=0.95)
    # border
    bd = FancyBboxPatch((cx+0.030, card_y0+card_h-0.360), card_w-0.060, 0.058,
                         boxstyle='round,pad=0.008',
                         facecolor='none', edgecolor='#FFD740', linewidth=2.0,
                         transform=t3_ax.transAxes, zorder=7)
    t3_ax.add_patch(bd)
    td = t3_ax.text(cx+card_w/2, card_y0+card_h-0.331, card['drs'],
                    ha='center', va='center', fontsize=15, fontweight='bold',
                    color=WHITE, transform=t3_ax.transAxes, zorder=9)
    td.set_path_effects([pe.withStroke(linewidth=4, foreground='#FFD740', alpha=0.5)])

    # Metrics row
    hw = (card_w - 0.060)/2 - 0.006
    for mi, (metric, mcol) in enumerate([(card['m1'], T2_NE),(card['m2'], T1_NE)]):
        mx0 = cx+0.030 + mi*(hw+0.012)
        glow_box(t3_ax, mx0, card_y0+card_h-0.428, hw, 0.052, mcol, depth=1)
        t3_ax.text(mx0+hw/2, card_y0+card_h-0.402, metric,
                   ha='center', va='center', fontsize=9, fontweight='bold',
                   color=mcol, transform=t3_ax.transAxes, zorder=8)

    # Subtle divider line
    t3_ax.plot([cx+0.020, cx+card_w-0.020],
               [card_y0+card_h-0.442, card_y0+card_h-0.442],
               color=neon, lw=0.8, alpha=0.35,
               transform=t3_ax.transAxes, zorder=4)

    # Bullet points
    for bi, pt in enumerate(card['pts']):
        t3_ax.text(cx+0.016, card_y0+card_h-0.468-bi*0.100, pt,
                   ha='left', va='top', fontsize=8.8, color=SILVER,
                   transform=t3_ax.transAxes, zorder=8, linespacing=1.3)

    # Footer pills
    px = cx + 0.020
    for pill_txt in card['foot']:
        pw = 0.135
        gradient_band(t3_ax, px, card_y0+0.038, pw, 0.048, neon, dark, alpha=0.90)
        bd2 = FancyBboxPatch((px, card_y0+0.038), pw, 0.048,
                              boxstyle='round,pad=0.006',
                              facecolor='none', edgecolor=neon, linewidth=1.5,
                              transform=t3_ax.transAxes, zorder=6)
        t3_ax.add_patch(bd2)
        t3_ax.text(px+pw/2, card_y0+0.062, pill_txt,
                   ha='center', va='center', fontsize=7.8, fontweight='bold',
                   color=WHITE, transform=t3_ax.transAxes, zorder=8)
        px += pw + 0.018

# ═══════════════════════════════════════════════════════════════════════════════
# INFRA FOOTER
# ═══════════════════════════════════════════════════════════════════════════════
inf_ax = fig.add_axes([ML, INF_B+0.003, MW, INF_T-INF_B], facecolor='none')
inf_ax.set_xlim(0,1); inf_ax.set_ylim(0,1); inf_ax.axis('off')
glow_box(inf_ax, 0, 0, 1, 1, DIM, depth=1, r='round,pad=0.010')

items = [
    ('Serum PrPc Liquid Biopsy  AUC=0.777',       0.020),
    ('IHC: PrPc H-score ≥50 + KRAS mutation',      0.280),
    ('Pharmacogenomics: 113 drugs · 59 synergy',    0.560),
    ('ADDS v5.3  |  TCGA n=2,285  |  KR Patent 2026', 0.780),
]
for label, xi in items:
    inf_ax.text(xi, 0.50, label, ha='left', va='center',
                fontsize=8.5, color=SILVER, transform=inf_ax.transAxes, zorder=8)
    if xi > 0.02:
        inf_ax.plot([xi-0.015, xi-0.015],[0.15,0.85], color=DIM, lw=1.2,
                    transform=inf_ax.transAxes, alpha=0.7, zorder=5)

fig.text(0.5, 0.001,
         'Figure: AI-Driven Pritamab Pipeline  |  Inha University Hospital × ADDS Framework  |  Nature Communications 2026',
         ha='center', va='bottom', fontsize=8, color=DIM, style='italic')

# ═══════════════════════════════════════════════════════════════════════════════
# SAVE
# ═══════════════════════════════════════════════════════════════════════════════
OUT = r'f:\ADDS\figures'
pdf_path = os.path.join(OUT, 'pritamab_3d_infographic.pdf')
png_path = os.path.join(OUT, 'pritamab_3d_infographic.png')

fig.savefig(pdf_path, format='pdf', bbox_inches='tight', facecolor=BG_TOP)
print(f'PDF saved: {pdf_path}')

fig.savefig(png_path, format='png', dpi=DPI, bbox_inches='tight',
            facecolor=BG_TOP, edgecolor='none')
print(f'PNG saved: {png_path}  ({int(W_IN*DPI)} x {int(H_IN*DPI)} px @ {DPI} dpi)')
plt.close()
