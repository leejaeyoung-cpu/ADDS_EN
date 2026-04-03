"""
Panel C — 4 variants comparison
Options 1–4 shown as 2×2 grid in F:\\ADDS\\panelC_compare.png
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import numpy as np
from scipy.stats import mannwhitneyu

np.random.seed(2026)

# ── Colour palette ──────────────────────────────────────────────
C_ADDS = '#2E6FA3'
C_SING = '#D97E14'
C_TEXT = '#1A2B3C'
C_GRID = '#DDE8F0'

# ── Simulated data (matching main script) ───────────────────────
N_DRUGS   = 9
TRUE_CS   = np.sort(np.random.dirichlet(np.ones(N_DRUGS)))[::-1]
MOD_NAMES = ['PK/PD\n(32d)', 'Cell Morph\n(128d)', 'RNA-seq\n(256d)',
             'CT\n(64d)', 'Synergy\n(32d)']
MOD_BARS  = ['Reference\n(no removal)'] + MOD_NAMES

def top3_ret(true, pred):
    t3 = set(np.argsort(true)[-3:])
    p3 = set(np.argsort(pred)[-3:])
    return len(t3 & p3) / 3

def adds_s(cs, noise):
    scores = cs + np.random.normal(0, noise, len(cs))
    return scores

def sing_s(cs, noise):
    scores = cs + np.random.normal(0, noise * 1.9, len(cs))
    return scores

N_REP_MOD = 3000
NOISE_MOD = 0.025
mod_w     = [32/512, 128/512, 256/512, 64/512, 32/512]

raw_a_all, raw_s_all = [], []
full_a = [top3_ret(TRUE_CS, adds_s(TRUE_CS, NOISE_MOD)) for _ in range(N_REP_MOD)]
full_s = [top3_ret(TRUE_CS, sing_s(TRUE_CS, NOISE_MOD)) for _ in range(N_REP_MOD)]
raw_a_all.append(full_a); raw_s_all.append(full_s)

for w in mod_w:
    ra = [top3_ret(TRUE_CS, adds_s(TRUE_CS, NOISE_MOD*(1 + w*3.5))) for _ in range(N_REP_MOD)]
    rs = [top3_ret(TRUE_CS, sing_s(TRUE_CS, NOISE_MOD*(1 + w*9.0))) for _ in range(N_REP_MOD)]
    raw_a_all.append(ra); raw_s_all.append(rs)

def rank_biserial(x, y):
    n1, n2 = len(x), len(y)
    U, _ = mannwhitneyu(x, y, alternative='greater')
    return 2*U / (n1*n2) - 1

pvals      = [mannwhitneyu(ra, rs, alternative='two-sided')[1]
              for ra, rs in zip(raw_a_all, raw_s_all)]
rbis       = [rank_biserial(ra, rs) for ra, rs in zip(raw_a_all, raw_s_all)]
sig_labels = ['***' if p < 0.001 else ('**' if p < 0.01 else ('*' if p < 0.05 else 'ns'))
              for p in pvals]

drp_a = [round(np.mean(ra), 2) for ra in raw_a_all]
drp_s = [round(np.mean(rs), 2) for rs in raw_s_all]
err_a = [np.std(ra) for ra in raw_a_all]
err_s = [np.std(rs) for rs in raw_s_all]
N_COND = len(MOD_BARS)
x      = np.arange(N_COND)

# Short labels for tight axes
short_labels = ['Ref', 'PK/PD', 'Cell\nMph', 'RNA-\nseq', 'CT', 'Syn']

def style_panel(ax, title):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_color('#B0C0D0')
    ax.spines['left'].set_color('#B0C0D0')
    ax.yaxis.grid(True, color=C_GRID, lw=0.8, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(labelsize=8.5, colors=C_TEXT)
    ax.set_title(title, fontsize=10.5, fontweight='bold', color=C_TEXT, pad=6)

# ═══════════════════════════════════════════════════════════════
fig = plt.figure(figsize=(18, 12), facecolor='white')
fig.suptitle('Panel C — Four Alternative Representations',
             fontsize=13, fontweight='bold', color=C_TEXT, y=0.98)
gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.35,
                       left=0.07, right=0.97, top=0.93, bottom=0.08)

# ── Option 1: Cleveland Connected Dot Plot ──────────────────────
ax1 = fig.add_subplot(gs[0, 0])
for i, (va, vs, r, sl) in enumerate(zip(drp_a, drp_s, rbis, sig_labels)):
    ax1.plot([va, vs], [i, i], color='#BFC9D4', lw=1.8, zorder=1)
    ax1.scatter(va, i, s=70, color=C_ADDS, zorder=4, edgecolors='white', lw=1)
    ax1.scatter(vs, i, s=70, color=C_SING, zorder=4, edgecolors='white', lw=1, marker='s')
    col_sig = '#C00000' if sl == 'ns' else C_ADDS
    ax1.text(max(va, vs) + 0.025, i, f'{sl}  r={r:.2f}',
             va='center', fontsize=7.5, color=col_sig, fontweight='bold')
ax1.set_yticks(range(N_COND))
ax1.set_yticklabels(short_labels, fontsize=9)
ax1.set_xlim(0, 1.22)
ax1.set_ylim(-0.5, N_COND - 0.5)
ax1.set_xlabel('Top-3 Retention Rate', fontsize=9, color=C_TEXT)
leg_a1 = mpatches.Patch(color=C_ADDS, label='ADDS')
leg_s1 = mpatches.Patch(color=C_SING, label='Single')
ax1.legend(handles=[leg_a1, leg_s1], fontsize=8.5, loc='lower right',
           framealpha=0.92, edgecolor='#CBD5E1')
style_panel(ax1, 'Option 1 — Cleveland Dot Plot')

# ── Option 2: Δ Bar Chart (ADDS − Single) ───────────────────────
ax2 = fig.add_subplot(gs[0, 1])
deltas = [va - vs for va, vs in zip(drp_a, drp_s)]
colors2 = [C_ADDS if d >= 0 else '#C00000' for d in deltas]
bars2 = ax2.bar(x, deltas, color=colors2, alpha=0.85, zorder=3, width=0.55)
ax2.axhline(0, color='#888', lw=1.0, ls='--')
for i, (d, r, sl) in enumerate(zip(deltas, rbis, sig_labels)):
    ax2.text(i, d + 0.005 if d >= 0 else d - 0.005,
             f'{sl}\nr={r:.2f}', ha='center',
             va='bottom' if d >= 0 else 'top',
             fontsize=7.5, color=C_ADDS, fontweight='bold')
ax2.set_xticks(x)
ax2.set_xticklabels(short_labels, fontsize=8.5)
ax2.set_ylabel('\u0394 Retention (ADDS \u2212 Single)', fontsize=9, color=C_TEXT)
ax2.set_ylim(-0.05, 0.55)
style_panel(ax2, 'Option 2 — \u0394 Bar (ADDS \u2212 Single)')

# ── Option 3: Heatmap ───────────────────────────────────────────
ax3 = fig.add_subplot(gs[1, 0])
heat_data = np.array([drp_a, drp_s])
im = ax3.imshow(heat_data, cmap='Blues', aspect='auto', vmin=0, vmax=1)
for r in range(2):
    for c in range(N_COND):
        val = heat_data[r, c]
        sig = sig_labels[c] if r == 0 else ''
        ax3.text(c, r, f'{val:.2f}\n{sig}',
                 ha='center', va='center', fontsize=8.5,
                 color='white' if val > 0.55 else C_TEXT,
                 fontweight='bold' if sig else 'normal')
ax3.set_xticks(range(N_COND))
ax3.set_xticklabels(short_labels, fontsize=9)
ax3.set_yticks([0, 1])
ax3.set_yticklabels(['ADDS', 'Single'], fontsize=9.5, fontweight='bold')
plt.colorbar(im, ax=ax3, shrink=0.85, label='Top-3 Retention', pad=0.01)
ax3.set_title('Option 3 — Heatmap Grid', fontsize=10.5,
              fontweight='bold', color=C_TEXT, pad=6)

# ── Option 4: Lollipop ──────────────────────────────────────────
ax4 = fig.add_subplot(gs[1, 1])
width4 = 0.28
for i, (va, vs, r, sl) in enumerate(zip(drp_a, drp_s, rbis, sig_labels)):
    xa, xs = i - width4/2, i + width4/2
    # ADDS lollipop
    ax4.plot([xa, xa], [0, va], color=C_ADDS, lw=2.2, zorder=3)
    ax4.scatter(xa, va, s=70, color=C_ADDS, zorder=5, edgecolors='white', lw=1.2)
    ax4.text(xa, va + 0.012, f'{va:.2f}', ha='center', fontsize=7.5, color=C_ADDS)
    # Single lollipop
    ax4.plot([xs, xs], [0, vs], color=C_SING, lw=2.2, ls='--', zorder=3)
    ax4.scatter(xs, vs, s=70, color=C_SING, zorder=5, edgecolors='white', lw=1.2, marker='s')
    ax4.text(xs, vs + 0.012, f'{vs:.2f}', ha='center', fontsize=7.5, color=C_SING)
    # bracket
    col_sig = '#C00000' if sl == 'ns' else C_ADDS
    y_br = max(va, vs) + 0.08
    ax4.plot([xa, xa, xs, xs], [va, y_br, y_br, vs], color='#555', lw=0.8)
    ax4.text(i, y_br + 0.004, sl, ha='center', va='bottom',
             fontsize=9, color=col_sig, fontweight='bold')
    ax4.text(i, y_br - 0.05, f'r={r:.2f}', ha='center', va='bottom',
             fontsize=7.5, color='#333', style='italic')
ax4.axhline(0, color='#BBB', lw=0.8)
ax4.set_xticks(x)
ax4.set_xticklabels(short_labels, fontsize=8.5)
ax4.set_ylim(0, 1.38)
ax4.set_ylabel('Top-3 Retention Rate', fontsize=9, color=C_TEXT)
leg_a4 = mpatches.Patch(color=C_ADDS, label='ADDS (4-modal)')
leg_s4 = mpatches.Patch(color=C_SING, label='Single-modality')
ax4.legend(handles=[leg_a4, leg_s4], fontsize=8.5, loc='lower right',
           framealpha=0.92, edgecolor='#CBD5E1')
style_panel(ax4, 'Option 4 — Lollipop')

# ── Save ────────────────────────────────────────────────────────
OUT = r'F:\ADDS\panelC_compare.png'
plt.savefig(OUT, dpi=180, bbox_inches='tight', facecolor='white')
print(f'Saved → {OUT}')
plt.close()
