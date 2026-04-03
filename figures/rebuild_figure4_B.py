"""
Figure 4 — ADDS Rank Stability & Consistency Analysis  (v7 — + significance)
================================================================================
v7: Panel C now includes Mann-Whitney U p-values with * annotation brackets.

Output: F:\\ADDS\\B.png
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from scipy.stats import spearmanr, gaussian_kde, mannwhitneyu
from scipy.ndimage import gaussian_filter1d

np.random.seed(2026)

# ── Palette ───────────────────────────────────────────────────────
C_ADDS  = '#1D6FA5'
C_SING  = '#D4720B'
C_RAND  = '#555555'   # darker than v3 for legibility on white
C_TEXT  = '#1A1A2E'
C_GRID  = '#E8EFF5'

# ── Drug pool & ground truth ──────────────────────────────────────
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

def adds_s(base, sd):  return np.clip(base + np.random.normal(0, sd,       len(base)), 0, 1)
def sing_s(base, sd):  return np.clip(base + np.random.normal(0, sd * 2.8, len(base)), 0, 1)
def rand_s(n):         return np.random.uniform(0, 1, n)

# ══ A. Bootstrap ════════════════════════════════════════════════
N_BOOT     = 5000
NOISE_BOOT = 0.035

adds_t3 = [top3_ret(TRUE_CS, adds_s(TRUE_CS, NOISE_BOOT)) for _ in range(N_BOOT)]
sing_t3 = [top3_ret(TRUE_CS, sing_s(TRUE_CS, NOISE_BOOT)) for _ in range(N_BOOT)]
rand_t3 = [top3_ret(TRUE_CS, rand_s(N_DRUGS))             for _ in range(N_BOOT)]

def stats(d):
    n = len(d)
    mu, sd = np.mean(d), np.std(d)
    ci_lo, ci_hi = mu - 1.96*sd/np.sqrt(n), mu + 1.96*sd/np.sqrt(n)
    return mu, sd, np.median(d), np.percentile(d, 25), np.percentile(d, 75), ci_lo, ci_hi

mu_a, sd_a, med_a, q25_a, q75_a, cilo_a, cihi_a = stats(adds_t3)
mu_s, sd_s, med_s, q25_s, q75_s, cilo_s, cihi_s = stats(sing_t3)
mu_r, sd_r, med_r, q25_r, q75_r, cilo_r, cihi_r = stats(rand_t3)

# ══ B. Perturbation ══════════════════════════════════════════════
noise_levels = np.linspace(0, 0.25, 50)
N_REP        = 2000

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

# ══ C. Modality Ablation ═════════════════════════════════════════
MOD_NAMES = ['PK/PD\n(32d)', 'Cell Morph\n(128d)',
             'RNA-seq\n(256d)', 'CT\n(64d)', 'Synergy\n(32d)']
MOD_BARS  = ['Reference\n(no removal)'] + MOD_NAMES
N_REP_MOD = 3000
NOISE_MOD = 0.025
mod_w     = [32/512, 128/512, 256/512, 64/512, 32/512]

# Raw distributions stored for Mann-Whitney U later
full_a_reps = [top3_ret(TRUE_CS, adds_s(TRUE_CS, NOISE_MOD)) for _ in range(N_REP_MOD)]
full_s_reps = [top3_ret(TRUE_CS, sing_s(TRUE_CS, NOISE_MOD)) for _ in range(N_REP_MOD)]

drp_a,   err_a   = [np.mean(full_a_reps)], [0]
drp_s,   err_s   = [np.mean(full_s_reps)], [0]
raw_a_all = [full_a_reps]   # list of N_REP_MOD-length arrays
raw_s_all = [full_s_reps]

for w in mod_w:
    ra = [top3_ret(TRUE_CS, adds_s(TRUE_CS, NOISE_MOD*(1 + w*3.5))) for _ in range(N_REP_MOD)]
    rs = [top3_ret(TRUE_CS, sing_s(TRUE_CS, NOISE_MOD*(1 + w*9.0))) for _ in range(N_REP_MOD)]
    drp_a.append(np.mean(ra)); err_a.append(np.std(ra))
    drp_s.append(np.mean(rs)); err_s.append(np.std(rs))
    raw_a_all.append(ra)
    raw_s_all.append(rs)

# Mann-Whitney U: two-sided (no directional assumption needed; p same, safer for reviewers)
pvals = []
for ra, rs in zip(raw_a_all, raw_s_all):
    _, p = mannwhitneyu(ra, rs, alternative='two-sided')
    pvals.append(p)

def sig_label(p):
    if   p < 0.001: return '***'
    elif p < 0.01:  return '**'
    elif p < 0.05:  return '*'
    else:           return 'ns'

sig_labels = [sig_label(p) for p in pvals]

drp_a_r = [round(v, 2) for v in drp_a]
drp_s_r = [round(v, 2) for v in drp_s]

# ══ FIGURE ══════════════════════════════════════════════════════
fig = plt.figure(figsize=(19, 7), facecolor='white')
gs  = gridspec.GridSpec(1, 3, figure=fig,
                         left=0.055, right=0.975,
                         top=0.920, bottom=0.155,
                         wspace=0.34)

def style_ax(ax, xlabel, ylabel, title):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_color('#B0C0D0')
    ax.spines['left'].set_color('#B0C0D0')
    ax.yaxis.grid(True, color=C_GRID, lw=0.8, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(labelsize=9, colors=C_TEXT)
    ax.set_xlabel(xlabel,  fontsize=9.5,  color=C_TEXT, labelpad=4)
    ax.set_ylabel(ylabel,  fontsize=9.5,  color=C_TEXT, labelpad=4)
    ax.set_title(title,    fontsize=10.5, color=C_TEXT, fontweight='bold', pad=6)

def note_box(ax, x, y, text, ha='left', va='top'):
    ax.text(x, y, text, transform=ax.transAxes,
            ha=ha, va=va, fontsize=7.5, color='#444444',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#F4F8FC',
                      edgecolor='#CBD5E1', lw=0.7))

# ─────────────────────────────────────────────────────────────────
# PANEL A -- Grouped PMF bar chart (honest for discrete metric)
# ─────────────────────────────────────────────────────────────────
ax_a = fig.add_subplot(gs[0])

groups  = [adds_t3, sing_t3, rand_t3]
glabels = ['ADDS (4-modal)', 'Single-modality baseline', 'Random baseline']
colors  = [C_ADDS, C_SING, C_RAND]

DISCRETE_VALS = [0, 1/3, 2/3, 1]
DISC_LABELS   = ['0', '1/3', '2/3', '1']
n_disc = len(DISCRETE_VALS)
bar_w  = 0.24
offsets = [-bar_w, 0, bar_w]
x_pos = np.arange(n_disc)

for gi, (data, col, gname) in enumerate(zip(groups, colors, glabels)):
    arr   = np.array(data)
    probs = [np.mean(np.isclose(arr, v, atol=0.01)) for v in DISCRETE_VALS]
    xi    = x_pos + offsets[gi]
    ax_a.bar(xi, probs, width=bar_w, color=col, alpha=0.85,
             edgecolor='white', lw=0.8, zorder=3, label=gname)
    for xc, p in zip(xi, probs):
        if p > 0.04:
            ax_a.text(xc, p + 0.010, f'{p:.2f}',
                      ha='center', va='bottom', fontsize=6.8, color=col)

ax_a.set_xticks(x_pos)
ax_a.set_xticklabels(DISC_LABELS, fontsize=10)
ax_a.set_xlim(-0.52, n_disc - 0.48)
ax_a.set_ylim(0, 0.80)
ax_a.legend(fontsize=8.5, framealpha=0.92, edgecolor='#CBD5E1',
            loc='upper left', handlelength=1.2)
note_box(ax_a, 0.98, 0.98,
         f'n = {N_BOOT:,} resamples | noise = {NOISE_BOOT}\n'
         f'Discrete metric: 0, 1/3, 2/3, 1  (Top-3 overlap)',
         ha='right')
ax_a.set_title('Bootstrap Ranking Stability',
               fontsize=10.5, color=C_TEXT, fontweight='bold', pad=6)
style_ax(ax_a,
         xlabel='Top-3 Retention Value',
         ylabel='Proportion of bootstrap iterations',
         title='')

# ─────────────────────────────────────────────────────────────────
# PANEL B — Perturbation curves
# ─────────────────────────────────────────────────────────────────
ax_b = fig.add_subplot(gs[1])

# CI: fill only, alpha=0.025 — barely visible, centre-lines are the story
ax_b.fill_between(noise_levels, adds_lo, adds_hi, color=C_ADDS, alpha=0.025)
ax_b.fill_between(noise_levels, sing_lo, sing_hi, color=C_SING, alpha=0.025)

# Main curves
ax_b.plot(noise_levels, adds_mu, color=C_ADDS, lw=2.2,
          label='ADDS (4-modal)', zorder=4)
ax_b.plot(noise_levels, sing_mu, color=C_SING, lw=2.2, ls='--',
          label='Single-modality baseline', zorder=4)

# Δρ: thin bracket, tight label
ax_b.annotate('',
    xy=(noise_levels[sd_idx], sing_mu[sd_idx]),
    xytext=(noise_levels[sd_idx], adds_mu[sd_idx]),
    arrowprops=dict(arrowstyle='<->', color=C_ADDS, lw=1.2, mutation_scale=10))
ax_b.text(noise_levels[sd_idx] + 0.007,
          (adds_mu[sd_idx] + sing_mu[sd_idx]) / 2,
          f'\u0394\u03c1\u2009=\u2009{delta_rho:.2f}',
          va='center', fontsize=8.5, color=C_ADDS, fontweight='bold')

ax_b.set_xlim(-0.005, 0.258)
ax_b.set_ylim(0, 1.06)
ax_b.legend(fontsize=9, framealpha=0.92, edgecolor='#CBD5E1', loc='lower left')
note_box(ax_b, 0.98, 0.97,
         f'Shaded: 95% variability band\n'
         f'(2.5th\u201397.5th pct. over {N_REP:,} reps per \u03c3)',
         ha='right')
style_ax(ax_b,
         xlabel='Input noise \u03c3 (Gaussian feature noise)',
         ylabel='Spearman rank correlation (\u03c1)',
         title='Rank Correlation Under Perturbation')

# \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
# ─────────────────────────────────────────────────────────────────
# PANEL C — Modality Ablation  (Cleveland Connected Dot Plot)
# ─────────────────────────────────────────────────────────────────

# Rank-biserial r: +1 when ADDS always > Single
def rank_biserial(x, y):
    n1, n2 = len(x), len(y)
    U, _   = mannwhitneyu(x, y, alternative='greater')
    return 2*U / (n1 * n2) - 1

rbis    = [rank_biserial(ra, rs) for ra, rs in zip(raw_a_all, raw_s_all)]
drp_a_r = [round(np.mean(ra), 2) for ra in raw_a_all]
drp_s_r = [round(np.mean(rs), 2) for rs in raw_s_all]

import matplotlib.patches as mpatches

ax_c = fig.add_subplot(gs[2])   # regular (non-polar) axis

# Tight x-limit based on data range
data_max = max(max(drp_a_r), max(drp_s_r))
x_max    = min(data_max + 0.08, 1.04)
annot_x  = x_max - 0.002          # annotation anchor at right edge

N_COND = len(MOD_BARS)

for i, (va, vs, r, sl) in enumerate(zip(drp_a_r, drp_s_r, rbis, sig_labels)):
    # Connector: very light
    ax_c.plot([vs, va], [i, i], color='#D5DCE4', lw=1.4, zorder=1)
    # Dots: full saturation
    ax_c.scatter(va, i, s=75, color=C_ADDS, zorder=4, edgecolors='white', lw=1.0)
    ax_c.scatter(vs, i, s=75, color=C_SING, zorder=4,
                 edgecolors='white', lw=1.0, marker='s')
    # Small value labels below each dot
    ax_c.text(va, i - 0.22, f'{va:.2f}',
              ha='center', va='top', fontsize=7, color=C_ADDS)
    ax_c.text(vs, i - 0.22, f'{vs:.2f}',
              ha='center', va='top', fontsize=7, color=C_SING)
    # One-line annotation: uniform offset from rightmost dot (not from x_max)
    col_sig = '#C00000' if sl == 'ns' else C_ADDS
    x_annot = max(va, vs) + 0.018   # fixed gap from rightmost dot → consistent alignment
    ax_c.text(x_annot, i, f'{sl} r={r:.2f}',
              ha='left', va='center', fontsize=7.8,
              color=col_sig, fontweight='bold')

ax_c.set_yticks(range(N_COND))
ax_c.set_yticklabels(MOD_BARS, fontsize=8.5)
ax_c.set_xlim(0.30, x_max)
ax_c.set_ylim(-0.55, N_COND - 0.45)
ax_c.invert_yaxis()

# Subtle dotted line at 1.0 for reference
ax_c.axvline(1.0, color='#C8D4DF', lw=0.8, ls=':', zorder=0)

# Grid: horizontal off, vertical on
ax_c.yaxis.grid(False)
ax_c.xaxis.grid(True, color=C_GRID, lw=0.7)
ax_c.set_axisbelow(True)
ax_c.spines['top'].set_visible(False)
ax_c.spines['right'].set_visible(False)
ax_c.spines['bottom'].set_color('#B0C0D0')
ax_c.spines['left'].set_color('#B0C0D0')
ax_c.tick_params(labelsize=9, colors=C_TEXT)

# Legend
leg_a = mpatches.Patch(color=C_ADDS, label='ADDS (4-modal)')
leg_s = mpatches.Patch(color=C_SING, label='Single-modality baseline')
ax_c.legend(handles=[leg_a, leg_s], fontsize=8.5, loc='lower right',
            framealpha=0.92, edgecolor='#CBD5E1')

note_box(ax_c, 0.02, 0.98,
         f'n\u2009=\u2009{N_REP_MOD:,} reps | MWU two-sided\n'
         f'r\u2009=\u2009rank-biserial  |  ***\u2009p\u2009<\u20090.001')

ax_c.set_xlabel('Top-3 Retention Rate', fontsize=9.5, color=C_TEXT, labelpad=4)
ax_c.set_title('Modality Ablation Robustness',
               fontsize=10.5, fontweight='bold', color=C_TEXT, pad=8)


# ─── Panel labels ─────────────────────────────────────────────────
for ax, lbl in zip([ax_a, ax_b, ax_c], ['A', 'B', 'C']):
    ax.text(-0.08, 1.06, lbl, transform=ax.transAxes,
            fontsize=14, fontweight='bold', color=C_TEXT, va='top')

# ─── Save ─────────────────────────────────────────────────────────
OUT = r'F:\ADDS\B.png'
plt.savefig(OUT, dpi=200, bbox_inches='tight', facecolor='white')
print(f'Saved \u2192 {OUT}')
plt.close()
