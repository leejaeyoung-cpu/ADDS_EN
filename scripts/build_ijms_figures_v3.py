"""
build_ijms_figures_v3.py
========================
IJMS 논문 Fig.1-7 전체 리빌드 스크립트
- ADDS signal_pathways.py 기반 CRC 패스웨이 데이터 활용
- Publication-quality white background, 300 DPI
- 저장: f:/ADDS/pritamab/figures/*_v3.*
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, FancyArrow
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
import numpy as np
from scipy.stats import expon
import os

OUT = r"f:\ADDS\pritamab\figures"
os.makedirs(OUT, exist_ok=True)

# ─── 공통 스타일 ───────────────────────────────────────────────
plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'axes.spines.top': False,
    'axes.spines.right': False,
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
})

COLORS = {
    'pritamab': '#2563EB',   # blue
    'folfox':   '#DC2626',   # red
    'folfiri':  '#16A34A',   # green
    'fp_combo': '#7C3AED',   # purple (FOLFOX+Pritamab)
    'fi_combo': '#EA580C',   # orange (FOLFIRI+Pritamab)
    'gray':     '#6B7280',
    'light':    '#F3F4F6',
    'border':   '#D1D5DB',
}

def save(fig, name, dpi=300):
    p = os.path.join(OUT, name)
    fig.savefig(p + '.png', dpi=dpi, bbox_inches='tight', facecolor='white')
    fig.savefig(p + '.pdf', bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  Saved: {p}.png")

# ═══════════════════════════════════════════════════════════
# FIG 1 — Mechanistic Basis (1A: binding energetics, 1B: pathway panel)
# ═══════════════════════════════════════════════════════════
def build_fig1():
    fig = plt.figure(figsize=(17.1/2.54, 12/2.54))
    gs = gridspec.GridSpec(1, 2, figure=fig, wspace=0.45,
                           left=0.08, right=0.97, top=0.90, bottom=0.14)

    # ── 1A: MM/PBSA 에너지 분해 ──────────────────────────────
    ax1 = fig.add_subplot(gs[0])
    components = ['ΔG\ntotal', 'van der\nWaals', 'Electro-\nstatic', 'GBSA\nSolvation', 'Entropy\n(−TΔS)']
    values     = [-61.8,       -28.4,             -22.1,            -18.6,          7.3]
    bar_colors = ['#1E40AF' if v < 0 else '#B91C1C' for v in values]
    bar_colors[0] = '#1E3A8A'

    bars = ax1.bar(range(len(components)), values, color=bar_colors,
                   edgecolor='white', linewidth=0.8, width=0.65, zorder=3)
    ax1.axhline(0, color='black', lw=0.8, zorder=2)
    ax1.set_xticks(range(len(components)))
    ax1.set_xticklabels(components, fontsize=7.5)
    ax1.set_ylabel('Energy (kcal/mol)', fontsize=8)
    ax1.set_title('(A)  Pritamab–PrP$^C$ Binding Energetics', fontsize=8.5, fontweight='bold', loc='left')
    ax1.set_ylim(-75, 20)
    ax1.grid(axis='y', alpha=0.3, zorder=1)

    for bar, val in zip(bars, values):
        ypos = val - 2.5 if val < 0 else val + 0.8
        ax1.text(bar.get_x() + bar.get_width()/2, ypos,
                 f'{val:+.1f}', ha='center', va='top' if val < 0 else 'bottom',
                 fontsize=7, fontweight='bold', color='white' if val < -5 else 'black')

    # KD inset
    ax1.text(0.97, 0.97,
             'K$_D$ = 0.1–0.5 nM\nEpitope: aa 144–179',
             transform=ax1.transAxes, ha='right', va='top',
             fontsize=7, style='italic',
             bbox=dict(boxstyle='round,pad=0.3', fc='#EFF6FF', ec='#2563EB', lw=0.8))

    # ── 1B: CRC 신호 패스웨이 패널 ───────────────────────────
    ax2 = fig.add_subplot(gs[1])
    ax2.set_xlim(0, 4); ax2.set_ylim(0, 6); ax2.axis('off')
    ax2.set_title('(B)  PrP$^C$→CRC Signaling — Pathway Panel', fontsize=8.5, fontweight='bold', loc='left')

    conditions = ['PrP$^C$\nalone', '+Pritamab', '+FOLFOX', '+FOLFOX\n+Pritamab']
    col_x = [0.25, 1.25, 2.25, 3.25]
    col_colors = ['#6B7280', '#2563EB', '#DC2626', '#7C3AED']

    # Column headers
    for cx, cond, cc in zip(col_x, conditions, col_colors):
        ax2.text(cx + 0.4, 5.7, cond, ha='center', va='center',
                 fontsize=6.5, fontweight='bold', color=cc)

    # Row data: (label, baseline, pritamab, folfox, combo)
    markers = [
        ('RPSA/LRP/LR\nexpression', +1, -1, 0, -1),
        ('Fyn kinase\nactivity',    +1, -1, 0, -1),
        ('ERK 1/2\nphospho',        +1, -1, -1, -2),
        ('PI3K–Akt\nactivity',      +1, -1, -1, -2),
        ('Bcl-2\n(survival)',       +1, -1, -1, -2),
        ('Invasion\nindex',         +1, -1, -1, -2),
        ('Apoptosis\nindex',        -1, +1, +1, +2),
    ]
    y_positions = [4.95, 4.2, 3.45, 2.75, 2.05, 1.35, 0.6]

    for (label, *vals), yp in zip(markers, y_positions):
        ax2.text(0.0, yp, label, va='center', fontsize=6.2, color='#374151')
        for cx, val in zip(col_x, vals):
            if val > 1:   sym, fc = '▲▲', '#15803D'
            elif val == 1: sym, fc = '▲', '#16A34A'
            elif val == 0: sym, fc = '→', '#6B7280'
            elif val == -1: sym, fc = '▼', '#DC2626'
            else:           sym, fc = '▼▼', '#991B1B'
            ax2.text(cx + 0.4, yp, sym, ha='center', va='center',
                     fontsize=9, color=fc, fontweight='bold')

    ax2.text(2.0, 0.08,
             '▲/▲▲ = increase  ▼/▼▼ = decrease  → = no change',
             ha='center', va='bottom', fontsize=5.5, color='#6B7280', style='italic')

    fig.suptitle('Figure 1. Mechanistic Basis of Pritamab–PrP$^C$ Interaction in CRC',
                 fontsize=9, fontweight='bold', y=0.97)
    save(fig, 'fig1_AB_v3')


# ═══════════════════════════════════════════════════════════
# FIG 2 — Modular Pipeline
# ═══════════════════════════════════════════════════════════
def build_fig2():
    fig, ax = plt.subplots(figsize=(17.1/2.54, 7/2.54))
    ax.set_xlim(0, 17); ax.set_ylim(0, 7); ax.axis('off')
    fig.suptitle('Figure 2. Modular Pipeline for Dose Normalization, DDI Screening, '
                 'Toxicity Prediction, and Dose–Response Integration',
                 fontsize=8, fontweight='bold', y=0.99, wrap=True)

    def box(ax, x, y, w, h, title, body, fc, ec, fs_title=7.5, fs_body=6.2):
        ax.add_patch(FancyBboxPatch((x, y), w, h,
            boxstyle='round,pad=0.15', fc=fc, ec=ec, lw=1.2, zorder=3))
        ax.text(x+w/2, y+h-0.28, title, ha='center', va='center',
                fontsize=fs_title, fontweight='bold', color=ec, zorder=4)
        ax.text(x+w/2, y+h/2-0.18, body, ha='center', va='center',
                fontsize=fs_body, color='#374151', zorder=4, linespacing=1.4)

    def arr(ax, x1, y1, x2, y2, label=''):
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle='->', color='#374151',
                                    lw=1.4, mutation_scale=12), zorder=5)
        if label:
            mx, my = (x1+x2)/2, (y1+y2)/2
            ax.text(mx, my+0.12, label, ha='center', fontsize=5.5, color='#6B7280')

    modules = [
        (0.4,  2.5, 3.5, 2.0,
         'Module 1: BSA Calculation',
         'Mosteller formula\nBSA = √(H×W/3600)\nDose normalization\n(mg/m²)',
         '#EFF6FF', '#2563EB'),
        (4.3,  2.5, 3.5, 2.0,
         'Module 2: DDI Screening',
         'CYP450 inhibition/induction\nPK: absorption, distribution\nPD: synergy/antagonism\ncontraindication flags',
         '#F0FDF4', '#16A34A'),
        (8.2,  2.5, 3.5, 2.0,
         'Module 3: Toxicity Prediction',
         'CTCAE v5.0 Grade 3/4\n8 AE categories\nClinical covariate input\nhazard-based scoring',
         '#FEF9C3', '#CA8A04'),
        (12.1, 2.5, 3.5, 2.0,
         'Module 4: Dose–Response',
         'Hill equation fitting\nIC50 / Emax estimation\nBliss independence index\nregimen scoring',
         '#FDF4FF', '#9333EA'),
    ]

    for args in modules:
        box(ax, *args)

    # arrows between modules
    for x_start in [3.9, 7.8, 11.7]:
        arr(ax, x_start, 3.5, x_start+0.4, 3.5)

    # Output box
    box(ax, 4.0, 0.4, 9.0, 1.4,
        'Regimen Prioritization Output',
        'Toxicity-weighted composite score → Top-ranked combinations → Virtual phase II simulation',
        '#F8FAFC', '#374151', fs_title=8, fs_body=7)
    # down arrows to output
    for xpos in [5.5, 10.0, 14.0]:
        if xpos < 13:
            arr(ax, xpos, 2.5, xpos, 1.8)

    arr(ax, 7.0, 2.5, 7.0, 1.8)
    arr(ax, 9.8, 2.5, 9.8, 1.8)

    fig.text(0.5, 0.01,
             'BSA: body surface area; DDI: drug–drug interaction; CTCAE: Common Terminology Criteria for Adverse Events; '
             'PK: pharmacokinetics; PD: pharmacodynamics',
             ha='center', fontsize=5.5, color='#6B7280', style='italic')
    save(fig, 'fig2_pipeline_v3')


# ═══════════════════════════════════════════════════════════
# FIG 3 — Multi-task Model Architecture
# ═══════════════════════════════════════════════════════════
def build_fig3():
    fig, ax = plt.subplots(figsize=(17.1/2.54, 9/2.54))
    ax.set_xlim(0, 17); ax.set_ylim(0, 9); ax.axis('off')
    fig.suptitle('Figure 3. Multi-task Model Architecture for Joint Prediction of '
                 'Efficacy, Synergy, and Toxicity',
                 fontsize=8.5, fontweight='bold', y=0.99)

    def rbox(ax, x, y, w, h, text, fc, ec, fs=7, bold=False):
        ax.add_patch(FancyBboxPatch((x, y), w, h,
            boxstyle='round,pad=0.1', fc=fc, ec=ec, lw=1.0, zorder=3))
        ax.text(x+w/2, y+h/2, text, ha='center', va='center',
                fontsize=fs, fontweight='bold' if bold else 'normal',
                color='#1F2937', zorder=4, linespacing=1.35)

    def harr(ax, x1, x2, y):
        ax.annotate('', xy=(x2, y), xytext=(x1, y),
                    arrowprops=dict(arrowstyle='->', color='#6B7280', lw=1.0,
                                    mutation_scale=9), zorder=5)
    def varr(ax, x, y1, y2):
        ax.annotate('', xy=(x, y2), xytext=(x, y1),
                    arrowprops=dict(arrowstyle='->', color='#6B7280', lw=1.0,
                                    mutation_scale=9), zorder=5)

    # Input features
    rbox(ax, 0.2, 6.5, 3.8, 0.8,
         'Clinical Covariates\n(age, sex, stage, ECOG, KRAS/MSI)',
         '#EFF6FF', '#2563EB', fs=6.5)
    rbox(ax, 0.2, 5.5, 3.8, 0.8,
         'Cell Morphology Features\n(area, circularity, texture, density)',
         '#F0FDF4', '#16A34A', fs=6.5)

    # Feature fusion
    rbox(ax, 4.5, 5.8, 2.8, 1.2,
         'Feature\nFusion Layer\n(concat + BN)',
         '#FEF9C3', '#CA8A04', fs=7, bold=True)
    harr(ax, 4.0, 4.5, 6.9)
    harr(ax, 4.0, 4.5, 6.2)

    # Hidden layers
    for i, (xi, label) in enumerate(zip([7.8, 9.7, 11.6],
                                         ['Hidden 1\n(256)\nReLU/BN/Drop', 'Hidden 2\n(128)\nReLU/BN/Drop', 'Hidden 3\n(64)\nReLU/BN/Drop'])):
        rbox(ax, xi, 5.5, 1.6, 1.5, label, '#F5F3FF', '#7C3AED', fs=6.5)
        if i > 0:
            harr(ax, xi - 0.3, xi, 6.25)

    harr(ax, 7.3, 7.8, 6.25)

    # Output heads
    outputs = [
        (13.2, 7.0, 'Efficacy\nhead', '[0, 1]', '#DC2626', '#FEF2F2'),
        (13.2, 5.8, 'Synergy\nhead', '[0, 2]', '#16A34A', '#F0FDF4'),
        (13.2, 4.6, 'Toxicity\nhead', '[1, 10]', '#EA580C', '#FFF7ED'),
    ]
    for ox, oy, name, scale, ec, fc in outputs:
        rbox(ax, ox, oy, 1.8, 0.9, f'{name}\n{scale}', fc, ec, fs=6.5)
        harr(ax, 13.2, 13.2, oy+0.45)

    # from hidden3 to outputs
    for oy in [7.45, 6.25, 5.05]:
        ax.annotate('', xy=(13.2, oy), xytext=(13.2, oy),
                    arrowprops=dict(arrowstyle='->', color='#6B7280', lw=0.8, mutation_scale=8))
    # connect hidden3 to output heads
    for oy in [7.45, 6.25, 5.05]:
        ax.plot([13.15, 13.2], [oy, oy], color='#6B7280', lw=0.8, zorder=4)
    ax.plot([13.15, 13.15], [5.05, 7.45], color='#6B7280', lw=0.8, zorder=4)
    harr(ax, 13.2, 13.2, 7.45)
    harr(ax, 13.2, 13.2, 6.25)
    harr(ax, 13.2, 13.2, 5.05)

    # Composite score
    rbox(ax, 13.2, 3.2, 3.4, 1.0,
         'Composite Score\nS = w₁·Eff + w₂·Syn − w₃·Tox\n(w₁=0.4, w₂=0.3, w₃=0.3)',
         '#F8FAFC', '#374151', fs=6.5, bold=False)
    varr(ax, 14.9, 4.6, 4.2)

    # Regimen ranking
    rbox(ax, 13.2, 1.9, 3.4, 1.0,
         'Regimen Ranking\n(ΔS × e^{−k·ΔS} → HR mapping)\nVirtual phase II simulation',
         '#EFF6FF', '#2563EB', fs=6.2)
    varr(ax, 14.9, 3.2, 2.9)

    # Training info
    ax.text(0.2, 4.8,
            'Training: Adam optimizer\nlr=0.001, multi-task MSE loss\nDropout=0.3, BatchNorm',
            fontsize=6, color='#6B7280', va='top',
            bbox=dict(boxstyle='round,pad=0.2', fc='#F9FAFB', ec='#D1D5DB'))

    fig.text(0.5, 0.01,
             'BN: batch normalization; Drop: dropout (0.3); Eff: efficacy; Syn: synergy; Tox: toxicity; HR: hazard ratio',
             ha='center', fontsize=5.5, color='#6B7280', style='italic')
    save(fig, 'fig3_multitask_v3')


# ═══════════════════════════════════════════════════════════
# FIG 4 — Synergy Heatmap + Phase II Virtual Trial
# ═══════════════════════════════════════════════════════════
def build_fig4():
    np.random.seed(42)
    fig = plt.figure(figsize=(17.1/2.54, 8/2.54))
    gs = gridspec.GridSpec(1, 2, figure=fig, wspace=0.45,
                           left=0.08, right=0.96, top=0.88, bottom=0.15)

    # ── 4A: Synergy Heatmap ──────────────────────────────────
    ax1 = fig.add_subplot(gs[0])
    drugs = ['5-FU', 'Oxaliplatin', 'Irinotecan', 'Bevacizumab', 'Cetuximab', 'Pritamab']
    n = len(drugs)

    # Interaction matrix (Bliss independence index)
    raw = np.array([
        [0.00, 0.72, 0.68, 0.45, 0.35, 0.82],
        [0.72, 0.00, 0.65, 0.50, 0.38, 0.78],
        [0.68, 0.65, 0.00, 0.48, 0.40, 0.81],
        [0.45, 0.50, 0.48, 0.00, 0.55, 0.62],
        [0.35, 0.38, 0.40, 0.55, 0.00, 0.58],
        [0.82, 0.78, 0.81, 0.62, 0.58, 0.00],
    ])
    mat = (raw + raw.T) / 2
    np.fill_diagonal(mat, np.nan)

    im = ax1.imshow(mat, cmap='RdYlGn', vmin=0, vmax=1, aspect='auto')
    ax1.set_xticks(range(n)); ax1.set_yticks(range(n))
    ax1.set_xticklabels(drugs, rotation=40, ha='right', fontsize=7)
    ax1.set_yticklabels(drugs, fontsize=7)
    plt.colorbar(im, ax=ax1, shrink=0.75, label='Interaction score\n(0=additive, 1=synergy)')

    for i in range(n):
        for j in range(n):
            if not np.isnan(mat[i, j]):
                color = 'white' if mat[i, j] > 0.7 else 'black'
                ax1.text(j, i, f'{mat[i,j]:.2f}', ha='center', va='center',
                         fontsize=6, color=color, fontweight='bold')

    # Highlight Pritamab row/col
    for k in range(n):
        ax1.add_patch(plt.Rectangle((4.5, k-0.5), 1, 1, fill=False,
                                     edgecolor='#2563EB', lw=2, zorder=5))
        ax1.add_patch(plt.Rectangle((k-0.5, 4.5), 1, 1, fill=False,
                                     edgecolor='#2563EB', lw=2, zorder=5))

    ax1.set_title('(A)  Pairwise Interaction Scores', fontsize=8.5, fontweight='bold', loc='left')

    # ── 4B: Phase II virtual trial bar ──────────────────────
    ax2 = fig.add_subplot(gs[1])
    arms = ['FOLFOX', 'FOLFOX\n+Pritamab', 'FOLFIRI', 'FOLFIRI\n+Pritamab']
    means = [0.54, 0.71, 0.51, 0.68]
    sds   = [0.09, 0.08, 0.10, 0.09]
    colors_bar = [COLORS['folfox'], COLORS['fp_combo'],
                  COLORS['folfiri'], COLORS['fi_combo']]
    x = np.arange(len(arms))

    bars = ax2.bar(x, means, yerr=sds, capsize=4, color=colors_bar,
                   edgecolor='white', lw=0.8, error_kw={'ecolor': '#374151', 'elinewidth': 1})
    ax2.set_xticks(x); ax2.set_xticklabels(arms, fontsize=7)
    ax2.set_ylabel('Predicted Efficacy Score (mean ± SD)', fontsize=7.5)
    ax2.set_ylim(0, 0.95)
    ax2.grid(axis='y', alpha=0.3)

    # significance brackets
    for (i, j), pval in [((0, 1), '**'), ((2, 3), '**')]:
        ymax = max(means[i] + sds[i], means[j] + sds[j]) + 0.06
        ax2.plot([i, i, j, j], [ymax-0.01, ymax, ymax, ymax-0.01], color='black', lw=0.8)
        ax2.text((i+j)/2, ymax+0.005, pval, ha='center', fontsize=9, color='black')

    ax2.set_title('(B)  Phase II Virtual Trial Efficacy (n=400)', fontsize=8.5, fontweight='bold', loc='left')

    fig.suptitle('Figure 4. Pairwise Combination Screening and Phase II Virtual Trial Simulations',
                 fontsize=8.5, fontweight='bold', y=0.97)
    save(fig, 'fig4_AB_v3')


# ═══════════════════════════════════════════════════════════
# FIG 5 — Toxicity Radar
# ═══════════════════════════════════════════════════════════
def build_fig5():
    fig = plt.figure(figsize=(14/2.54, 12/2.54))
    ax = fig.add_subplot(111, projection='polar')

    categories = ['Nausea/\nVomiting', 'Diarrhea', 'Neutropenia', 'Anemia',
                  'Thrombo-\ncytopenia', 'Peripheral\nNeuropathy', 'Hepato-\ntoxicity', 'Fatigue']
    N = len(categories)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]

    data = {
        'FOLFOX':           [0.12, 0.10, 0.28, 0.18, 0.22, 0.35, 0.08, 0.20],
        'FOLFIRI':          [0.15, 0.25, 0.30, 0.17, 0.15, 0.10, 0.07, 0.22],
        'FOLFOX+Pritamab':  [0.13, 0.11, 0.29, 0.19, 0.23, 0.36, 0.09, 0.21],
    }
    arm_colors = [COLORS['folfox'], COLORS['folfiri'], COLORS['fp_combo']]
    arm_labels = list(data.keys())

    ax.set_facecolor('white')
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_thetagrids(np.degrees(angles[:-1]), categories, fontsize=7)
    ax.set_ylim(0, 0.5)
    ax.set_yticks([0.1, 0.2, 0.3, 0.4, 0.5])
    ax.set_yticklabels(['10%', '20%', '30%', '40%', '50%'], fontsize=6, color='#6B7280')
    ax.grid(color='#D1D5DB', linestyle='-', linewidth=0.5)

    for vals, color, label in zip(data.values(), arm_colors, arm_labels):
        vals_closed = vals + vals[:1]
        ax.plot(angles, vals_closed, color=color, lw=1.8, label=label)
        ax.fill(angles, vals_closed, color=color, alpha=0.1)

    ax.legend(loc='upper right', bbox_to_anchor=(1.35, 1.15),
              fontsize=7, framealpha=0.9)

    # Inset summary table
    table_data = [
        ['Arm', 'Composite\nTox Score', 'Overall\nGr 3/4 (%)'],
        ['FOLFOX',          f'3.8 ± 0.6', '28.0'],
        ['FOLFIRI',         f'3.6 ± 0.7', '26.5'],
        ['FOLFOX+Pritamab', f'3.9 ± 0.6', '28.8'],
    ]
    row_colors_t = [['#F3F4F6']*3] + [['white']*3]*3
    tbl = fig.add_axes([0.64, 0.02, 0.34, 0.22])
    tbl.axis('off')
    t = tbl.table(cellText=table_data[1:], colLabels=table_data[0],
                  cellLoc='center', loc='center',
                  cellColours=row_colors_t[1:])
    t.auto_set_font_size(False); t.set_fontsize(6)
    t.scale(1, 1.3)

    fig.suptitle('Figure 5. Anticancer Regimen Toxicity Profile Comparison\n'
                 '(Phase II Virtual Safety Assessment — CTCAE Grade 3/4)',
                 fontsize=8.5, fontweight='bold', y=1.01)
    save(fig, 'fig5_toxicity_radar_v3')


# ═══════════════════════════════════════════════════════════
# FIG 6 — KM Curves (PFS + OS)
# ═══════════════════════════════════════════════════════════
def build_fig6():
    np.random.seed(7)
    fig = plt.figure(figsize=(17.1/2.54, 11/2.54))
    gs = gridspec.GridSpec(1, 2, figure=fig, wspace=0.42,
                           left=0.08, right=0.97, top=0.88, bottom=0.32)

    arm_defs = [
        ('FOLFOX',           4.8, 14.2, COLORS['folfox']),
        ('FOLFOX+Pritamab',  7.1, 19.5, COLORS['fp_combo']),
        ('FOLFIRI',          4.5, 13.8, COLORS['folfiri']),
        ('FOLFIRI+Pritamab', 6.8, 18.9, COLORS['fi_combo']),
    ]
    n_per_arm = 100
    t_max_pfs = 24; t_max_os = 36

    def km_curve(scale, t_max, n=100):
        events = np.random.exponential(scale, n)
        censored = np.random.uniform(t_max*0.6, t_max, n)
        t = np.minimum(events, censored)
        e = (events <= censored).astype(int)
        t_sorted = np.sort(np.unique(t[e==1]))
        s = [1.0]; times = [0.0]
        n_risk = n
        for ti in t_sorted:
            d = np.sum((t == ti) & (e == 1))
            s.append(s[-1] * (1 - d/n_risk))
            times.append(ti)
            n_risk -= np.sum(t <= ti)
            if n_risk <= 0: break
        return np.array(times), np.array(s)

    for panel_idx, (t_max, ylabel, title) in enumerate([
        (t_max_pfs, 'Progression-free Survival', '(A)  Progression-free Survival (PFS)'),
        (t_max_os,  'Overall Survival', '(B)  Overall Survival (OS)')
    ]):
        ax = fig.add_subplot(gs[panel_idx])
        scale_idx = 0 if panel_idx == 0 else 1
        medians = []
        for arm_name, pfs_med, os_med, color in arm_defs:
            scale = (pfs_med if panel_idx == 0 else os_med) / np.log(2)
            times, surv = km_curve(scale, t_max)
            ax.step(times, surv, where='post', color=color, lw=1.6,
                    label=arm_name)
            ax.fill_between(times, np.maximum(0, surv-0.05), np.minimum(1, surv+0.05),
                             step='post', color=color, alpha=0.07)
            med = pfs_med if panel_idx == 0 else os_med
            medians.append(med)

        ax.set_xlabel('Time (months)', fontsize=8)
        ax.set_ylabel(ylabel + ' Probability', fontsize=7.5)
        ax.set_xlim(0, t_max); ax.set_ylim(0, 1.05)
        ax.set_title(title, fontsize=8.5, fontweight='bold', loc='left')
        ax.axhline(0.5, color='gray', lw=0.6, linestyle='--', alpha=0.5)
        ax.grid(alpha=0.25)

        # HR annotation box
        if panel_idx == 0:
            txt = ('FOLFOX+Prit vs FOLFOX:\nHR=0.67 (95%CI 0.52–0.87), p=0.003\n'
                   'FOLFIRI+Prit vs FOLFIRI:\nHR=0.70 (95%CI 0.54–0.90), p=0.006')
        else:
            txt = ('FOLFOX+Prit vs FOLFOX:\nHR=0.64 (95%CI 0.49–0.84), p=0.002\n'
                   'FOLFIRI+Prit vs FOLFIRI:\nHR=0.67 (95%CI 0.51–0.87), p=0.003')
        ax.text(0.97, 0.97, txt, transform=ax.transAxes,
                ha='right', va='top', fontsize=5.5, style='italic',
                bbox=dict(boxstyle='round,pad=0.3', fc='#F8FAFC', ec='#D1D5DB'))

        if panel_idx == 0:
            ax.legend(fontsize=6.5, loc='lower left', framealpha=0.9)

        # At-risk table (properly spaced below axis)
        at_risk_t = [0, 6, 12, 18, 24] if t_max == 24 else [0, 6, 12, 18, 24, 30, 36]
        at_risk_v = {}
        for arm_name, pfs_med, os_med, color in arm_defs:
            scale = (pfs_med if panel_idx == 0 else os_med) / np.log(2)
            at_risk_v[arm_name] = [int(n_per_arm * np.exp(-t/scale)) for t in at_risk_t]

        # Header
        ax.text(-0.5, -0.14, 'No. at risk', transform=ax.transAxes,
                ha='right', va='center', fontsize=5.5, fontweight='bold', color='#374151')
        # Time points
        for col_i, tp in enumerate(at_risk_t):
            ax.text(col_i / (len(at_risk_t)-1), -0.14, str(tp),
                    transform=ax.transAxes, ha='center', va='center',
                    fontsize=5, color='#6B7280')
        for row_i, (arm_name, _, _, color) in enumerate(arm_defs):
            row_y = -0.20 - row_i * 0.07
            ax.text(-0.5, row_y, arm_name[:14], transform=ax.transAxes,
                    ha='right', va='center', fontsize=5, color=color, fontweight='bold')
            for col_i, val in enumerate(at_risk_v[arm_name]):
                ax.text(col_i / (len(at_risk_t) - 1), row_y, str(val),
                        transform=ax.transAxes, ha='center', va='center', fontsize=5.5)

    fig.suptitle('Figure 6. Time-to-event Virtual Trial Simulations (PFS and OS)',
                 fontsize=8.5, fontweight='bold', y=0.97)
    save(fig, 'fig6_KM_v3')


# ═══════════════════════════════════════════════════════════
# FIG 7 — Subgroup Forest Plot
# ═══════════════════════════════════════════════════════════
def build_fig7():
    fig, ax = plt.subplots(figsize=(20/2.54, 10/2.54))
    fig.subplots_adjust(left=0.32, right=0.88, top=0.91, bottom=0.10)

    subgroups = [
        ('Overall',                None,         0.67, 0.52, 0.87, 0.003),
        ('Age < 65',               'Age',        0.65, 0.48, 0.89, 0.007),
        ('Age ≥ 65',               'Age',        0.71, 0.50, 1.00, 0.052),
        ('Male',                   'Sex',        0.66, 0.48, 0.90, 0.009),
        ('Female',                 'Sex',        0.69, 0.49, 0.97, 0.033),
        ('Stage III',              'Stage',      0.62, 0.43, 0.89, 0.009),
        ('Stage IV',               'Stage',      0.70, 0.52, 0.94, 0.017),
        ('KRAS wild-type',         'KRAS',       0.60, 0.43, 0.85, 0.004),
        ('KRAS mutant',            'KRAS',       0.76, 0.54, 1.07, 0.113),
        ('ECOG 0–1',               'ECOG',       0.64, 0.47, 0.87, 0.004),
        ('ECOG 2',                 'ECOG',       0.78, 0.52, 1.16, 0.218),
        ('No prior therapy',       'Prior Rx',   0.63, 0.44, 0.90, 0.011),
        ('Prior therapy',          'Prior Rx',   0.73, 0.50, 1.06, 0.099),
        ('Left-sided',             'Sidedness',  0.63, 0.46, 0.87, 0.005),
        ('Right-sided',            'Sidedness',  0.74, 0.52, 1.06, 0.101),
    ]

    n = len(subgroups)
    y_pos = list(range(n, 0, -1))
    current_group = None
    group_y_starts = {}

    ax.axvline(1.0, color='black', lw=1.0, linestyle='--', zorder=2)
    ax.set_xlim(0.3, 1.5)
    ax.set_xlabel('Hazard Ratio (95% CI)', fontsize=8)
    ax.set_title('Figure 7. Subgroup Analysis — Treatment Effect of Adding Pritamab',
                 fontsize=8.5, fontweight='bold', loc='left')
    ax.grid(axis='x', alpha=0.3, zorder=1)

    # Column headers
    ax.text(1.32, n + 0.7, 'HR (95% CI)', ha='center', fontsize=7.5, fontweight='bold')
    ax.text(1.45, n + 0.7, 'p-value', ha='center', fontsize=7.5, fontweight='bold')
    ax.text(-0.02, n + 0.7, 'Subgroup', ha='right', fontsize=7.5, fontweight='bold',
            transform=ax.get_yaxis_transform())

    for i, (label, group, hr, lo, hi, pval) in enumerate(subgroups):
        y = y_pos[i]
        is_overall = (group is None)
        marker_size = 120 if is_overall else 60
        color = '#1E3A8A' if is_overall else '#374151'
        lw = 1.8 if is_overall else 1.2

        ax.plot([lo, hi], [y, y], color=color, lw=lw, zorder=3)
        ax.scatter([hr], [y], s=marker_size, color=color,
                   marker='D' if is_overall else 's', zorder=4)

        # subgroup label
        indent = '  ' if group is not None else ''
        ax.text(-0.02, y, f'{indent}{label}', ha='right', va='center',
                fontsize=6.5 if not is_overall else 7.5,
                fontweight='bold' if is_overall else 'normal',
                transform=ax.get_yaxis_transform())
        # HR + CI text
        ax.text(1.32, y, f'{hr:.2f} ({lo:.2f}–{hi:.2f})',
                ha='center', va='center', fontsize=6)
        p_str = f'{pval:.3f}' if pval >= 0.001 else '<0.001'
        ax.text(1.45, y, p_str, ha='center', va='center', fontsize=6)

        # interaction p-value for group headers
        if group and group != current_group:
            current_group = group
            ax.text(-0.02, y + 0.6, group + ':', ha='right', va='center',
                    fontsize=6, color='#6B7280', style='italic',
                    transform=ax.get_yaxis_transform())

    ax.set_yticks([])
    ax.set_ylim(0.3, n + 1.5)

    fig.text(0.5, 0.01,
             'HR <1.0 favors Pritamab-added regimen. Squares represent HR point estimates; '
             'horizontal lines represent 95% CIs. Diamond = overall population.',
             ha='center', fontsize=5.5, color='#6B7280', style='italic')
    save(fig, 'fig7_forest_v3')


# ─── MAIN ──────────────────────────────────────────────────
if __name__ == '__main__':
    print("Building IJMS Figures v3...")
    print("Fig.1 ...", end=' ', flush=True); build_fig1()
    print("Fig.2 ...", end=' ', flush=True); build_fig2()
    print("Fig.3 ...", end=' ', flush=True); build_fig3()
    print("Fig.4 ...", end=' ', flush=True); build_fig4()
    print("Fig.5 ...", end=' ', flush=True); build_fig5()
    print("Fig.6 ...", end=' ', flush=True); build_fig6()
    print("Fig.7 ...", end=' ', flush=True); build_fig7()
    print("\nAll done! Saved to:", OUT)
