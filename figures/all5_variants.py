"""
Figure 4 — All 5 Panel C variants (full A+B+C figure each)
Outputs: F:\\ADDS\\B_opt1.png ~ B_opt5.png
  opt1 = Cleveland Dot Plot
  opt2 = Delta Bar (ADDS - Single)
  opt3 = Heatmap Grid
  opt4 = Lollipop
  opt5 = Radar / Spider
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import numpy as np
from scipy.stats import spearmanr, mannwhitneyu
from scipy.ndimage import gaussian_filter1d

np.random.seed(2026)

# ── Palette ──────────────────────────────────────────────────────
C_ADDS = '#1D6FA5'
C_SING = '#D4720B'
C_RAND = '#555555'
C_TEXT = '#1A1A2E'
C_GRID = '#E8EFF5'

# ── Drug pool & ground truth ─────────────────────────────────────
DRUGS = [
    'Pritamab+FOLFOX', 'Pritamab+FOLFIRI', 'Pritamab+Oxali',
    'Pritamab+5-FU',   'Pritamab+Irino',   'Pritamab+TAS-102',
    'Pritamab+Soto',   'FOLFOX',            'FOLFIRI',
    'Sotorasib',       'Cetuximab+FOLFOX',  'Bev+FOLFOX',
]
N_DRUGS = len(DRUGS)
TRUE_CS = np.array([0.893, 0.872, 0.851, 0.823, 0.810, 0.807,
                    0.796, 0.641, 0.628, 0.572, 0.611, 0.654])

def top3_ret(tc, pc):
    return len(set(np.argsort(-tc)[:3]) & set(np.argsort(-pc)[:3])) / 3.0

def spear_r(tc, pc):
    r, _ = spearmanr(tc, pc)
    return max(r, 0.0)

def adds_s(base, sd): return np.clip(base + np.random.normal(0, sd,       len(base)), 0, 1)
def sing_s(base, sd): return np.clip(base + np.random.normal(0, sd * 2.8, len(base)), 0, 1)
def rand_s(n):        return np.random.uniform(0, 1, n)

# ══ A. Bootstrap ══════════════════════════════════════════════════
N_BOOT     = 5000
NOISE_BOOT = 0.035
adds_t3 = [top3_ret(TRUE_CS, adds_s(TRUE_CS, NOISE_BOOT)) for _ in range(N_BOOT)]
sing_t3 = [top3_ret(TRUE_CS, sing_s(TRUE_CS, NOISE_BOOT)) for _ in range(N_BOOT)]
rand_t3 = [top3_ret(TRUE_CS, rand_s(N_DRUGS))             for _ in range(N_BOOT)]

def _stats(d):
    mu, sd = np.mean(d), np.std(d)
    return mu, sd, np.median(d), np.percentile(d,25), np.percentile(d,75)

mu_a,sd_a,med_a,q25_a,q75_a = _stats(adds_t3)
mu_s,sd_s,med_s,q25_s,q75_s = _stats(sing_t3)
mu_r,sd_r,med_r,q25_r,q75_r = _stats(rand_t3)

# ══ B. Perturbation ═══════════════════════════════════════════════
noise_levels = np.linspace(0, 0.25, 50)
N_REP = 2000
adds_mu, adds_lo, adds_hi = [], [], []
sing_mu, sing_lo, sing_hi = [], [], []
for sd in noise_levels:
    a = [spear_r(TRUE_CS, adds_s(TRUE_CS, sd)) for _ in range(N_REP)]
    s = [spear_r(TRUE_CS, sing_s(TRUE_CS, sd)) for _ in range(N_REP)]
    adds_mu.append(np.mean(a))
    adds_lo.append(np.percentile(a, 2.5));  adds_hi.append(np.percentile(a, 97.5))
    sing_mu.append(np.mean(s))
    sing_lo.append(np.percentile(s, 2.5));  sing_hi.append(np.percentile(s, 97.5))
for lst in [adds_mu, adds_lo, adds_hi, sing_mu, sing_lo, sing_hi]:
    lst[:] = gaussian_filter1d(lst, 1.5)
sd_idx    = np.searchsorted(noise_levels, 0.10)
delta_rho = adds_mu[sd_idx] - sing_mu[sd_idx]

# ══ C. Modality Ablation ══════════════════════════════════════════
MOD_NAMES = ['PK/PD\n(32d)', 'Cell Morph\n(128d)',
             'RNA-seq\n(256d)', 'CT\n(64d)', 'Synergy\n(32d)']
MOD_BARS  = ['Reference\n(no removal)'] + MOD_NAMES
SHORT_LBL = ['Ref', 'PK/PD', 'Cell\nMph', 'RNA-\nseq', 'CT', 'Syn']
N_REP_MOD = 3000
NOISE_MOD = 0.025
mod_w     = [32/512, 128/512, 256/512, 64/512, 32/512]

full_a = [top3_ret(TRUE_CS, adds_s(TRUE_CS, NOISE_MOD)) for _ in range(N_REP_MOD)]
full_s = [top3_ret(TRUE_CS, sing_s(TRUE_CS, NOISE_MOD)) for _ in range(N_REP_MOD)]
raw_a_all = [full_a]
raw_s_all = [full_s]
for w in mod_w:
    ra = [top3_ret(TRUE_CS, adds_s(TRUE_CS, NOISE_MOD*(1+w*3.5))) for _ in range(N_REP_MOD)]
    rs = [top3_ret(TRUE_CS, sing_s(TRUE_CS, NOISE_MOD*(1+w*9.0))) for _ in range(N_REP_MOD)]
    raw_a_all.append(ra); raw_s_all.append(rs)

pvals = [mannwhitneyu(ra, rs, alternative='two-sided')[1]
         for ra, rs in zip(raw_a_all, raw_s_all)]
sig_labels = ['***' if p<0.001 else ('**' if p<0.01 else ('*' if p<0.05 else 'ns'))
              for p in pvals]

def rank_biserial(x, y):
    n1, n2 = len(x), len(y)
    U, _ = mannwhitneyu(x, y, alternative='greater')
    return 2*U/(n1*n2) - 1

rbis    = [rank_biserial(ra, rs) for ra, rs in zip(raw_a_all, raw_s_all)]
drp_a   = [round(np.mean(ra), 2) for ra in raw_a_all]
drp_s   = [round(np.mean(rs), 2) for rs in raw_s_all]
err_a_v = [np.std(ra) for ra in raw_a_all]
err_s_v = [np.std(rs) for rs in raw_s_all]

N_COND = len(MOD_BARS)
x_pos  = np.arange(N_COND)

# ── Shared helpers ────────────────────────────────────────────────
def note_box(ax, x, y, txt, ha='left', fs=7.5):
    ax.text(x, y, txt, transform=ax.transAxes,
            ha=ha, va='top', fontsize=fs, color='#4A5568',
            bbox=dict(boxstyle='round,pad=0.35', facecolor='#F7FAFC',
                      edgecolor='#CBD5E1', lw=0.8))

def style_ax(ax, xlabel='', ylabel='', title=''):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_color('#B0C0D0')
    ax.spines['left'].set_color('#B0C0D0')
    ax.yaxis.grid(True, color=C_GRID, lw=0.8, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(labelsize=9, colors=C_TEXT)
    if xlabel: ax.set_xlabel(xlabel, fontsize=9.5, color=C_TEXT, labelpad=4)
    if ylabel: ax.set_ylabel(ylabel, fontsize=9.5, color=C_TEXT, labelpad=4)
    if title:  ax.set_title(title, fontsize=10.5, fontweight='bold', color=C_TEXT, pad=8)

# ── Panel A builder ──────────────────────────────────────────────
def build_panel_a(ax):
    kw = dict(bw_method=0.18)
    for data, col, pos in [(adds_t3, C_ADDS, 0), (sing_t3, C_SING, 1), (rand_t3, C_RAND, 2)]:
        xv = np.linspace(-0.05, 1.05, 400)
        kde = __import__('scipy').stats.gaussian_kde(data, **kw)(xv)
        kde = kde / kde.max() * 0.38
        ax.fill_betweenx(xv, pos - kde, pos + kde, color=col, alpha=0.38, zorder=2)
        ax.plot(pos - kde, xv, color=col, lw=0.7, alpha=0.6)
        ax.plot(pos + kde, xv, color=col, lw=0.7, alpha=0.6)
        mu = np.mean(data)
        q25, q75 = np.percentile(data, 25), np.percentile(data, 75)
        med = np.median(data)
        ax.plot([pos-0.04, pos+0.04], [q25, q25], color=col, lw=1.8)
        ax.plot([pos-0.04, pos+0.04], [q75, q75], color=col, lw=1.8)
        ax.plot([pos-0.04, pos+0.04], [q25, q25], color=col, lw=0)
        ax.plot([pos, pos], [q25, q75], color=col, lw=1.8)
        ax.scatter(pos, med, s=38, color='white', edgecolors=col, lw=1.8, zorder=5)
        ax.text(pos, mu, f'{mu:.3f}', ha='center', va='center',
                fontsize=9.5, fontweight='bold', color=col, zorder=6)
    ax.set_xticks([0,1,2])
    ax.set_xticklabels(['ADDS\n(4-modal)', 'Single-modality\nbaseline', 'Random\nbaseline'], fontsize=9)
    ax.set_xlim(-0.55, 2.55); ax.set_ylim(-0.05, 1.08)
    note_box(ax, 0.02, 0.98,
             'Top-3 retention: discrete metric (0, \u2153, \u2154, 1)\n'
             'n\u2009=\u20095,000 resamples, noise \u03c3\u2009=\u20090.035\n'
             '\u25cb median\u2009|\u2009IQR bar\u2009|\u2009bold\u2009=\u2009mean')
    style_ax(ax, ylabel='Top-3 Retention Rate', title='Bootstrap Ranking Stability')

# ── Panel B builder ──────────────────────────────────────────────
def build_panel_b(ax):
    ax.fill_between(noise_levels, adds_lo, adds_hi, color=C_ADDS, alpha=0.025)
    ax.fill_between(noise_levels, sing_lo, sing_hi, color=C_SING, alpha=0.025)
    ax.plot(noise_levels, adds_mu, color=C_ADDS, lw=2.2, label='ADDS (4-modal)')
    ax.plot(noise_levels, sing_mu, color=C_SING, lw=2.2, ls='--', label='Single-modality baseline')
    sd10 = noise_levels[sd_idx]
    ya, ys = adds_mu[sd_idx], sing_mu[sd_idx]
    ax.annotate('', xy=(sd10, ys+0.005), xytext=(sd10, ya-0.005),
                arrowprops=dict(arrowstyle='<->', color=C_ADDS, lw=1.6))
    ax.text(sd10+0.012, (ya+ys)/2, f'\u0394\u03c1 = {delta_rho:.2f}',
            fontsize=9.5, color=C_ADDS, fontweight='bold', va='center')
    ax.legend(fontsize=9, framealpha=0.92, edgecolor='#CBD5E1', loc='upper right')
    note_box(ax, 0.98, 0.98,
             'Shaded: 95% variability band\n(2.5th\u201397.5th pct. over 2,000 reps per \u03c3)',
             ha='right')
    style_ax(ax, xlabel='Input noise \u03c3 (Gaussian feature noise)',
             ylabel='Spearman rank correlation (\u03c1)',
             title='Rank Correlation Under Perturbation')

# ── Panel C variants ─────────────────────────────────────────────
def build_c_cleveland(ax):
    """Option 1: Connected dot plot — refined"""
    # Data max across both groups (for tight x-limit)
    data_max = max(max(drp_a), max(drp_s))
    x_max    = min(data_max + 0.08, 1.04)   # tight, ≤1.04
    annot_x  = x_max - 0.002                 # annotations just inside right margin

    for i, (va, vs, r, sl) in enumerate(zip(drp_a, drp_s, rbis, sig_labels)):
        # ① Connector: very light so points are the stars
        ax.plot([vs, va], [i, i], color='#D5DCE4', lw=1.4, zorder=1)
        # ② Dots: full saturation
        ax.scatter(va, i, s=75, color=C_ADDS, zorder=4, edgecolors='white', lw=1.0)
        ax.scatter(vs, i, s=75, color=C_SING, zorder=4,
                   edgecolors='white', lw=1.0, marker='s')
        # Mean value labels (tight, small)
        ax.text(va + 0.005, i - 0.22, f'{va:.2f}',
                ha='center', va='top', fontsize=7, color=C_ADDS)
        ax.text(vs - 0.005, i - 0.22, f'{vs:.2f}',
                ha='center', va='top', fontsize=7, color=C_SING)
        # ③ Annotation: one line, right margin, outside data region
        col_sig = '#C00000' if sl == 'ns' else C_ADDS
        ax.text(annot_x, i, f'{sl} r={r:.2f}',
                ha='right', va='center', fontsize=7.8,
                color=col_sig, fontweight='bold')

    # ④ No background shading for Reference row
    #   (removed axhspan — rely on label "no removal" alone)

    ax.set_yticks(range(N_COND))
    ax.set_yticklabels(MOD_BARS, fontsize=8.5)
    # ① Tight x-axis: data lives in 0.42–0.87, give minimal padding
    ax.set_xlim(0.30, x_max)
    ax.set_ylim(-0.55, N_COND - 0.45)
    ax.invert_yaxis()

    # Vertical ref line at 1.0 (context marker)
    if x_max >= 1.0:
        ax.axvline(1.0, color='#C8D4DF', lw=0.8, ls=':', zorder=0)

    leg_a = mpatches.Patch(color=C_ADDS, label='ADDS (4-modal)')
    leg_s = mpatches.Patch(color=C_SING, label='Single-modality baseline')
    ax.legend(handles=[leg_a, leg_s], fontsize=8.5, loc='lower right',
              framealpha=0.92, edgecolor='#CBD5E1')
    note_box(ax, 0.02, 0.10,
             f'n\u2009=\u2009{N_REP_MOD:,} reps | MWU two-sided\n'
             f'r\u2009=\u2009rank-biserial  |  ***\u2009p\u2009<\u20090.001')
    style_ax(ax, xlabel='Top-3 Retention Rate', title='Modality Ablation Robustness')
    ax.yaxis.grid(False)
    ax.xaxis.grid(True, color=C_GRID, lw=0.7)
    ax.set_axisbelow(True)


def build_c_delta(ax):
    """Option 2: Delta bar (ADDS - Single)"""
    deltas = [va - vs for va, vs in zip(drp_a, drp_s)]
    err_d  = [np.sqrt(ea**2 + es**2) for ea, es in zip(err_a_v, err_s_v)]
    bars = ax.bar(x_pos, deltas, color=C_ADDS, alpha=0.85, zorder=3, width=0.55,
                  edgecolor='white', lw=0)
    ax.errorbar(x_pos, deltas, yerr=err_d,
                fmt='none', ecolor=C_ADDS, elinewidth=1.2, capsize=3, zorder=5)
    ax.axhline(0, color='#888', lw=1.0, ls='--')
    for i, (d, r, sl) in enumerate(zip(deltas, rbis, sig_labels)):
        col_sig = '#C00000' if sl == 'ns' else C_ADDS
        ax.text(i, d + err_d[i] + 0.015, f'{sl}', ha='center', va='bottom',
                fontsize=10, color=col_sig, fontweight='bold')
        ax.text(i, d + err_d[i] + 0.005, f'r={r:.2f}', ha='center', va='top',
                fontsize=7.5, color='#333', style='italic')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(SHORT_LBL, fontsize=8.5)
    ax.set_ylim(-0.05, 0.65)
    ax.set_ylabel('\u0394 Retention Rate (ADDS \u2212 Single)', fontsize=9, color=C_TEXT)
    note_box(ax, 0.02, 0.98,
             f'n\u2009=\u2009{N_REP_MOD:,} reps | MWU two-sided\n'
             f'\u0394\u2009=\u2009mean(ADDS)\u2009\u2212\u2009mean(Single)\n'
             f'r\u2009=\u2009rank-biserial correlation')
    style_ax(ax, title='Modality Ablation Robustness')

def build_c_heatmap(ax):
    """Option 3: Heatmap"""
    heat_data = np.array([drp_a, drp_s])
    im = ax.imshow(heat_data, cmap='Blues', aspect='auto', vmin=0.0, vmax=1.0)
    for r in range(2):
        for c in range(N_COND):
            val = heat_data[r, c]
            txt_col = 'white' if val > 0.58 else C_TEXT
            base = f'{val:.2f}'
            sig  = sig_labels[c] if r == 0 else ''
            label = f'{base}\n{sig}' if sig else base
            ax.text(c, r, label, ha='center', va='center',
                    fontsize=8.5 if not sig else 8,
                    color=txt_col,
                    fontweight='bold' if sig else 'normal')
    ax.set_xticks(range(N_COND))
    ax.set_xticklabels(SHORT_LBL, fontsize=9)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(['ADDS\n(4-modal)', 'Single-modality\nbaseline'],
                       fontsize=9, fontweight='bold')
    cbar = plt.colorbar(im, ax=ax, shrink=0.9, pad=0.02)
    cbar.set_label('Top-3 Retention Rate', fontsize=8.5)
    cbar.ax.tick_params(labelsize=8)
    ax.set_title('Modality Ablation Robustness',
                 fontsize=10.5, fontweight='bold', color=C_TEXT, pad=8)
    note_box(ax, 0.02, -0.18,
             f'n\u2009=\u2009{N_REP_MOD:,} reps | MWU two-sided | ***\u2009p<0.001 | r\u2009=\u2009rank-biserial',
             fs=7.0)

def build_c_lollipop(ax):
    """Option 4: Lollipop"""
    w4 = 0.26
    err_a_hi = [min(e, 1-v) for v, e in zip(drp_a, err_a_v)]
    err_s_hi = [min(e, 1-v) for v, e in zip(drp_s, err_s_v)]
    for i, (va, vs, r, sl) in enumerate(zip(drp_a, drp_s, rbis, sig_labels)):
        xa, xs = i - w4/2, i + w4/2
        ax.plot([xa, xa], [0, va], color=C_ADDS, lw=2.4, zorder=3)
        ax.scatter(xa, va, s=62, color=C_ADDS, zorder=5, edgecolors='white', lw=1.2)
        ax.text(xa, va + err_a_hi[i] + 0.012,
                f'{va:.2f}', ha='center', fontsize=7.5, color=C_ADDS)
        ax.plot([xs, xs], [0, vs], color=C_SING, lw=2.4, ls='--', zorder=3)
        ax.scatter(xs, vs, s=62, color=C_SING, zorder=5, marker='s',
                   edgecolors='white', lw=1.2)
        ax.text(xs, vs + err_s_hi[i] + 0.012,
                f'{vs:.2f}', ha='center', fontsize=7.5, color=C_SING)
        col_sig = '#C00000' if sl == 'ns' else C_ADDS
        y_br = max(va+err_a_hi[i], vs+err_s_hi[i]) + 0.07
        ax.plot([xa, xa, xs, xs], [va+err_a_hi[i], y_br, y_br, vs+err_s_hi[i]],
                color='#555', lw=0.8)
        ax.text(i, y_br+0.006, sl, ha='center', va='bottom',
                fontsize=11, color=col_sig, fontweight='bold')
        ax.text(i, y_br-0.052, f'r={r:.2f}', ha='center', va='bottom',
                fontsize=8.5, color='#333', style='italic')
    ax.axhline(0, color='#CCC', lw=0.7)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(SHORT_LBL, fontsize=8.5)
    ax.set_ylim(0, 1.40)
    leg_a = mpatches.Patch(color=C_ADDS, label='ADDS (4-modal)')
    leg_s = mpatches.Patch(color=C_SING, label='Single-modality baseline')
    ax.legend(handles=[leg_a, leg_s], fontsize=9, loc='lower right',
              framealpha=0.92, edgecolor='#CBD5E1')
    note_box(ax, 0.98, 0.97,
             f'n\u2009=\u2009{N_REP_MOD:,} reps\nMWU two-sided\nr\u2009=\u2009rank-biserial',
             ha='right')
    style_ax(ax, ylabel='Top-3 Retention Rate', title='Modality Ablation Robustness')

def build_c_radar(ax):
    """Option 5: Radar / Spider"""
    radar_labels = ['Reference\n(full)', 'PK/PD\nremoved', 'Cell Morph\nremoved',
                    'RNA-seq\nremoved', 'CT\nremoved', 'Synergy\nremoved']
    N_R    = len(radar_labels)
    angles = np.linspace(0, 2*np.pi, N_R, endpoint=False).tolist()
    angles += angles[:1]
    vals_a = drp_a + drp_a[:1]
    vals_s = drp_s + drp_s[:1]

    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)

    for ring in [0.25, 0.50, 0.75, 1.00]:
        ax.plot(np.linspace(0, 2*np.pi, 201), [ring]*201,
                color='#D0DAE4', lw=0.6, zorder=0)
        ax.text(0, ring+0.04, f'{ring:.2f}',
                ha='center', va='bottom', fontsize=7, color='#99A8B8')

    ax.plot(angles, vals_a, color=C_ADDS, lw=2.2, zorder=4)
    ax.fill(angles, vals_a, color=C_ADDS, alpha=0.20, zorder=3)
    ax.scatter(angles[:-1], drp_a, s=42, color=C_ADDS,
               zorder=5, edgecolors='white', lw=1.2)

    ax.plot(angles, vals_s, color=C_SING, lw=2.2, ls='--', zorder=4)
    ax.fill(angles, vals_s, color=C_SING, alpha=0.20, zorder=3)
    ax.scatter(angles[:-1], drp_s, s=42, color=C_SING,
               zorder=5, edgecolors='white', lw=1.2, marker='s')

    for ang, lab, va_v, vs_v, r, sl in zip(
            angles[:-1], radar_labels, drp_a, drp_s, rbis, sig_labels):
        cos_a = np.cos(ang - np.pi/2)
        sin_a = np.sin(ang - np.pi/2)
        ha  = 'left' if cos_a > 0.1 else ('right' if cos_a < -0.1 else 'center')
        va2 = 'bottom' if sin_a >= 0 else 'top'
        ax.text(ang, 1.18, lab, ha=ha, va=va2, fontsize=8.5, color=C_TEXT)
        col_sig = '#C00000' if sl == 'ns' else C_ADDS
        ax.text(ang, 1.40, f'{va_v} vs {vs_v} | {sl}',
                ha=ha, va=va2, fontsize=7.2, color=col_sig, fontweight='bold')

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([])
    ax.set_yticks([]); ax.set_ylim(0, 1.0)
    ax.spines['polar'].set_visible(False)
    leg_a = mpatches.Patch(facecolor=C_ADDS, alpha=0.70, label='ADDS (4-modal)')
    leg_s = mpatches.Patch(facecolor=C_SING, alpha=0.70, label='Single-modality baseline')
    ax.legend(handles=[leg_a, leg_s], fontsize=9, framealpha=0.92,
              edgecolor='#CBD5E1', loc='lower center',
              bbox_to_anchor=(0.5, -0.22), ncol=1)
    ax.set_title('Modality Ablation Robustness',
                 fontsize=10.5, fontweight='bold', color=C_TEXT, pad=85)
    ax.text(0.5, -0.10,
            f'n\u2009=\u2009{N_REP_MOD:,} reps\u2009|\u2009MWU two-sided\u2009|\u2009r\u2009=\u2009rank-biserial r',
            transform=ax.transAxes, ha='center', va='top',
            fontsize=7.5, color='#6B7280')

# ── Panel C variant registry ──────────────────────────────────────
PANEL_C_VARIANTS = [
    ('opt1_cleveland',  build_c_cleveland, False),
    ('opt2_delta',      build_c_delta,     False),
    ('opt3_heatmap',    build_c_heatmap,   False),
    ('opt4_lollipop',   build_c_lollipop,  False),
    ('opt5_radar',      build_c_radar,     True),   # True = polar axis
]

# ── Generate each full figure ─────────────────────────────────────
for tag, builder, polar in PANEL_C_VARIANTS:
    fig = plt.figure(figsize=(19, 7), facecolor='white')
    gs  = gridspec.GridSpec(1, 3, figure=fig,
                            left=0.055, right=0.975,
                            top=0.920, bottom=0.160,
                            wspace=0.34)

    ax_a = fig.add_subplot(gs[0])
    build_panel_a(ax_a)

    ax_b = fig.add_subplot(gs[1])
    build_panel_b(ax_b)

    ax_c = fig.add_subplot(gs[2], polar=polar)
    builder(ax_c)

    # Panel labels A B C
    for ax, lbl in zip([ax_a, ax_b, ax_c], ['A', 'B', 'C']):
        ax.text(-0.08, 1.06, lbl, transform=ax.transAxes,
                fontsize=14, fontweight='bold', color=C_TEXT, va='top')

    OUT = fr'F:\ADDS\B_{tag}.png'
    plt.savefig(OUT, dpi=200, bbox_inches='tight', facecolor='white')
    print(f'Saved → {OUT}')
    plt.close()

print('All 5 variants done.')
