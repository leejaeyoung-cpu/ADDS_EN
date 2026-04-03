"""
ADDS DL Concordance — Per-Drug Synergy Prediction
==================================================
Uses the ACTUAL 4-modal pipeline from src/pritamab_ml/:
  Modal 1: CellposeFeatureExtractor    (128-dim)
  Modal 2: RNAseqEncoder               (256-dim)
  Modal 3: PKPDFeatureModule           (32-dim)  ← drug-specific *
  Modal 4: CTTumorFeatureExtractor     (64-dim)
  → PritamamFusionModel → synergy_prob

ADDS DL (4-modal):   uses drug-specific EC50, KRAS weights, Bliss
Baseline DL (3-modal): zeros out PKPDFeatureModule (no drug context)

Ground truth: NatureComm Bliss (★) + ADDS estimates (#)
Population: KRAS G12D/V dominant (no G12C-specific drugs justified)
n=500 per drug per model → mean synergy_prob → rank → concordance

Output: f:\ADDS\figures\pritamab_dl_concordance.png
"""
import sys, os
sys.path.insert(0, r'f:\ADDS\src')
sys.path.insert(0, r'f:\ADDS')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from scipy.stats import spearmanr, kendalltau

from pritamab_ml.cellpose_feature_extractor import CellposeFeatureExtractor
from pritamab_ml.rnaseq_encoder              import RNAseqEncoder
from pritamab_ml.pkpd_feature_module         import PKPDFeatureModule
from pritamab_ml.multimodal_fusion           import CTTumorFeatureExtractor, PritamamFusionModel

# ── Colours ────────────────────────────────────────────────────
BG=     '#0A0F1E'
CADD=   '#22D3EE'
CBASE=  '#F87171'
CEXP=   '#34D399'
CGOLD=  '#F59E0B'
CWHITE= '#E8F4FF'
CGRAY=  '#6B7280'
CPURP=  '#A78BFA'

# ── Drug definitions ──────────────────────────────────────────
#   chemo_drug   : key for PKPDFeatureModule
#   kras_allele  : dominant allele for this drug's indicated population
#   gt_bliss     : NatureComm (★) or ADDS (#) Bliss score
#   confirmed    : True = ★ NatureComm
DRUG_CFG = [
    {'name': 'Prit+Oxali',   'chemo': 'Oxaliplatin',  'kras': 'G12D', 'gt': 21.7, 'conf': True,  'gr': '*'},
    {'name': 'Prit+5-FU',    'chemo': '5-FU',          'kras': 'G12D', 'gt': 18.4, 'conf': True,  'gr': '*'},
    {'name': 'Prit+FOLFOX',  'chemo': 'FOLFOX',        'kras': 'G12D', 'gt': 20.5, 'conf': False, 'gr': '#'},
    {'name': 'Prit+FOLFIRI', 'chemo': 'FOLFIRI',       'kras': 'G12D', 'gt': 18.8, 'conf': False, 'gr': '#'},
    {'name': 'Prit+Irino',   'chemo': 'Irinotecan',    'kras': 'G12D', 'gt': 17.3, 'conf': False, 'gr': '#'},
    {'name': 'Prit+TAS-102', 'chemo': 'FOLFOXIRI',     'kras': 'G12D', 'gt': 18.1, 'conf': False, 'gr': '#'},
    {'name': 'Prit+Soto',    'chemo': 'Sotorasib',     'kras': 'G12C', 'gt': 15.8, 'conf': False, 'gr': '#'},
]
N_DRUGS = len(DRUG_CFG)
N_PAT   = 300   # patients per drug (speed vs accuracy)
SEED    = 2026
RNG     = np.random.default_rng(SEED)

# ── Initialise extractors (shared) ───────────────────────────
print("Initialising 4-modal extractors...")
cell_ext = CellposeFeatureExtractor(seed=SEED)
rna_enc  = RNAseqEncoder(seed=SEED)
pkpd_mod = PKPDFeatureModule()
ct_ext   = CTTumorFeatureExtractor(seed=SEED)
fusion   = PritamamFusionModel(seed=SEED)
print("Done.")

# ── Per-drug DL inference ─────────────────────────────────────
results_adds = []   # ADDS 4-modal
results_base = []   # Baseline: no PK/PD (zero-padded)

for cfg in DRUG_CFG:
    chemo = cfg['chemo']
    kras  = cfg['kras']
    prpc  = True   # PrPc-high assumed (target population)
    conc  = 10.0   # Pritamab trough ~10nM (above KD=0.84nM ★)

    # ── Generate feature vectors ─────────────────────────────
    cell_f = cell_ext.simulate_features(
        N_PAT, pritamab_treated=True, prpc_high=prpc,
        kras_allele=kras, concentration_nM=conc)  # (N,128)

    rna_f = rna_enc.encode_samples(
        N_PAT, pritamab_treated=True, prpc_high=prpc,
        kras_allele=kras, concentration_nM=conc)   # (N,256)

    pkpd_f = pkpd_mod.compute_features(
        N_PAT, chemo_drug=chemo, kras_allele=kras,
        prpc_high=prpc, concentration_nM=conc, rng=RNG)  # (N,32)

    ct_f = ct_ext.simulate_features(
        N_PAT, pritamab_treated=True, prpc_high=prpc,
        kras_allele=kras)                                 # (N,64)

    # ── ADDS: 4-modal (128+256+32+64=480) ────────────────────
    X_adds = np.concatenate([cell_f, rna_f, pkpd_f, ct_f], axis=1)
    out_adds= fusion.forward(X_adds)
    syn_adds= float(out_adds['synergy_prob'].mean())
    results_adds.append(syn_adds)

    # ── Baseline: zero-pad PK/PD features (3-modal) ─────────
    pkpd_zero = np.zeros_like(pkpd_f)
    X_base = np.concatenate([cell_f, rna_f, pkpd_zero, ct_f], axis=1)
    out_base= fusion.forward(X_base)
    syn_base= float(out_base['synergy_prob'].mean())
    results_base.append(syn_base)

    print(f"  {cfg['name']:20s}: ADDS={syn_adds:.4f}  BASE={syn_base:.4f}  GT={cfg['gt']:.1f}{cfg['gr']}")

# ── Arrays and ranks ─────────────────────────────────────────
ADDS_syn = np.array(results_adds)
BASE_syn = np.array(results_base)
GT_bliss = np.array([d['gt']  for d in DRUG_CFG])
confirmed= [d['conf'] for d in DRUG_CFG]
grades   = [d['gr']   for d in DRUG_CFG]
names    = [d['name'] for d in DRUG_CFG]
short    = [n.replace('Prit+','') for n in names]

# Scale synergy_prob (0-1) → Bliss-like (0-25) for direct comparison
# Calibrate ADDS to Oxali★=21.7
scale_a = GT_bliss[0] / ADDS_syn[0]
scale_b = GT_bliss[0] / BASE_syn[0]
ADDS_bliss = ADDS_syn * scale_a
BASE_bliss = BASE_syn * scale_b

ADDS_RANK = np.argsort(-ADDS_bliss) + 1
BASE_RANK = np.argsort(-BASE_bliss) + 1
EXP_RANK  = np.argsort(-GT_bliss)   + 1

# ── Concordance metrics ───────────────────────────────────────
rho_a, p_a = spearmanr(EXP_RANK, ADDS_RANK)
rho_b, p_b = spearmanr(EXP_RANK, BASE_RANK)
tau_a, _   = kendalltau(EXP_RANK, ADDS_RANK)
tau_b, _   = kendalltau(EXP_RANK, BASE_RANK)
t2e  = set(np.argsort(EXP_RANK)[:2])
t2a  = len(t2e & set(np.argsort(ADDS_RANK)[:2])) / 2
t2b  = len(t2e & set(np.argsort(BASE_RANK)[:2])) / 2
err_a= np.abs(ADDS_RANK - EXP_RANK)
err_b= np.abs(BASE_RANK - EXP_RANK)

print("\n" + "=" * 70)
print("DL CONCORDANCE SUMMARY")
print("=" * 70)
for i in range(N_DRUGS):
    print(f"  {names[i]:20s}: ADDS_rank={ADDS_RANK[i]} EXP_rank={EXP_RANK[i]} "
          f"BASE_rank={BASE_RANK[i]}  |  "
          f"ADDS_bliss={ADDS_bliss[i]:.2f}  BASE_bliss={BASE_bliss[i]:.2f}  GT={GT_bliss[i]:.1f}")
print(f"\nSpearman: ADDS={rho_a:.3f}(p={p_a:.3f}) | Base={rho_b:.3f}(p={p_b:.3f})")
print(f"Kendall:  ADDS={tau_a:.3f} | Base={tau_b:.3f}")
print(f"Top-2 match: ADDS={t2a:.0%} | Base={t2b:.0%}")
print(f"Mean |rank err|: ADDS={err_a.mean():.2f} | Base={err_b.mean():.2f}")

# ═══════════════════════════════════════════════════════════════
# FIGURE — 4 panels
# ═══════════════════════════════════════════════════════════════
fig = plt.figure(figsize=(24, 11), facecolor=BG)
fig.patch.set_facecolor(BG)
fig.text(0.5, 0.97,
         'ADDS 4-Modal DL — Drug Prioritisation Concordance vs Experimental Bliss',
         color=CWHITE, fontsize=14, fontweight='bold', ha='center', va='top')
fig.text(0.5, 0.93,
         f'"ADDS DL (4-modal: Cellpose+RNA-seq+PKPD+CT) prioritisation concordance: '
         f'rho={rho_a:.3f} vs Baseline (3-modal, no PKPD): rho={rho_b:.3f}"',
         color=CADD, fontsize=9.5, ha='center', va='top', style='italic')

for xi, l in [(0.01,'A'),(0.27,'B'),(0.52,'C'),(0.76,'D')]:
    fig.text(xi, 0.89, l, color=CWHITE, fontsize=15, fontweight='bold')

def sax(ax):
    ax.set_facecolor(BG)
    for sp in ax.spines.values(): sp.set_visible(False)
    ax.spines['bottom'].set_visible(True); ax.spines['bottom'].set_color('#2A3A5A')
    ax.spines['left'].set_visible(True);   ax.spines['left'].set_color('#2A3A5A')
    ax.tick_params(colors=CWHITE, labelsize=8)

# ── A: DL Synergy Probability per drug ───────────────────────
ax_a = fig.add_axes([0.03, 0.13, 0.22, 0.70])
sax(ax_a)
xi = np.arange(N_DRUGS); w = 0.32
cols_a = [CGOLD if confirmed[i] else CADD for i in range(N_DRUGS)]
ba = ax_a.bar(xi-w/2, ADDS_bliss, w, color=cols_a, alpha=0.88, label='ADDS (4-modal)', zorder=3)
bb = ax_a.bar(xi+w/2, BASE_bliss, w, color=CBASE, alpha=0.75, label='Baseline (no PKPD)', zorder=3)
ax_a.scatter(xi, GT_bliss, color=CGOLD, zorder=6, s=70, marker='D', label='GT Bliss ★/#')
for bar, rk in zip(ba, ADDS_RANK):
    ax_a.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.4,
              f'#{rk}', color=bar.get_facecolor(), fontsize=8.5, ha='center', va='bottom', fontweight='bold')
for bar, rk in zip(bb, BASE_RANK):
    ax_a.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.4,
              f'#{rk}', color=CBASE, fontsize=8, ha='center', va='bottom')
ax_a.set_xticks(xi)
ax_a.set_xticklabels(short, color=CWHITE, fontsize=8, rotation=30, ha='right')
ax_a.set_ylabel('Predicted Bliss Score\n(syn_prob × calibration factor)', color=CWHITE, fontsize=9)
ax_a.set_title('A. DL-Predicted Bliss per Drug\n(PritamamFusionModel.synergy_prob)',
               color=CWHITE, fontsize=10, fontweight='bold')
ax_a.yaxis.grid(True, color='#1A2A3A', lw=0.5, zorder=0)
ax_a.legend(fontsize=8, facecolor='#0D1A2E', edgecolor='#2A4A6A', labelcolor=CWHITE)
ax_a.text(0.01, 0.01,
          'Numbers = predicted rank\n(#1 = ADDS top priority)',
          transform=ax_a.transAxes, color=CGRAY, fontsize=7.5)

# ── B: PKPD feature contribution (delta between ADDS and Baseline) ─────
ax_b = fig.add_axes([0.28, 0.13, 0.21, 0.70])
sax(ax_b)
delta = ADDS_bliss - BASE_bliss   # PKPD contribution
cols_b = [CADD if d > 0 else CBASE for d in delta]
bars_b = ax_b.bar(range(N_DRUGS), delta, color=cols_b, alpha=0.85, zorder=3)
ax_b.axhline(0, color=CWHITE, lw=0.8)
for bar, val, grd in zip(bars_b, delta, grades):
    ypos = val + 0.05 if val >= 0 else val - 0.4
    ax_b.text(bar.get_x()+bar.get_width()/2, ypos,
              f'{val:+.2f}{grd}', color=bar.get_facecolor(), fontsize=7.5,
              ha='center', va='bottom', fontweight='bold')
ax_b.set_xticks(range(N_DRUGS))
ax_b.set_xticklabels(short, color=CWHITE, fontsize=8, rotation=30, ha='right')
ax_b.set_ylabel('ADDS − Baseline Bliss\n(PK/PD feature contribution)', color=CWHITE, fontsize=9)
ax_b.set_title('B. PK/PD Feature Contribution\n(ADDS 4-modal vs 3-modal Baseline)',
               color=CWHITE, fontsize=10, fontweight='bold')
ax_b.yaxis.grid(True, color='#1A2A3A', lw=0.5, zorder=0)
ax_b.text(0.5, 0.97,
          'Positive = ADDS ranks higher than baseline\ndue to drug-specific PK/PD context',
          transform=ax_b.transAxes, color=CADD, fontsize=8, va='top', ha='center' )

# ── C: Rank bump chart ────────────────────────────────────────
ax_c = fig.add_axes([0.52, 0.13, 0.21, 0.70])
ax_c.set_facecolor(BG)
for sp in ax_c.spines.values(): sp.set_visible(False)
ax_c.spines['bottom'].set_visible(True); ax_c.spines['bottom'].set_color('#2A3A5A')
ax_c.tick_params(colors=CWHITE, labelsize=8)

for i in range(N_DRUGS):
    lw = 2.5 if confirmed[i] else 1.2
    ax_c.plot([1,2],[EXP_RANK[i],ADDS_RANK[i]], color=CADD, lw=lw, alpha=0.88)
    ax_c.plot([1,3],[EXP_RANK[i],BASE_RANK[i]], color=CBASE, lw=lw, ls='--', alpha=0.55)

for ci,(rnks,col) in enumerate([(EXP_RANK,CEXP),(ADDS_RANK,CADD),(BASE_RANK,CBASE)]):
    for j,r in enumerate(rnks):
        ax_c.plot(ci+1, r, ('*' if confirmed[j] else 'o'), color=col,
                  markersize=(14 if confirmed[j] else 8), zorder=5)

for i in range(N_DRUGS):
    ax_c.text(0.85, EXP_RANK[i],
              ('★ ' if confirmed[i] else '# ')+short[i],
              color=CGOLD if confirmed[i] else CWHITE, fontsize=8,
              ha='right', va='center', fontweight='bold' if confirmed[i] else 'normal')

ax_c.set_xlim(0.5,3.8); ax_c.set_ylim(N_DRUGS+0.5,0.5)
ax_c.set_xticks([1,2,3])
ax_c.set_xticklabels(['GT Bliss\n(Exp)','ADDS\n4-modal','Baseline\n3-modal'],
                     color=CWHITE, fontsize=9)
ax_c.set_yticks(range(1,N_DRUGS+1))
ax_c.set_yticklabels([f'Rank {i}' for i in range(1,N_DRUGS+1)], color=CGRAY, fontsize=8)
ax_c.set_title('C. Rank Concordance Bump Chart\n(GT vs ADDS vs Baseline)',
               color=CWHITE, fontsize=10, fontweight='bold')
ax_c.xaxis.grid(True, color='#1A2A3A', lw=0.8)

# Annotation for key moves
for i in range(N_DRUGS):
    if confirmed[i]:
        move_a = EXP_RANK[i] - ADDS_RANK[i]
        if move_a > 0:
            ax_c.annotate(f'+{move_a}', xy=(2, ADDS_RANK[i]), xytext=(2.2, ADDS_RANK[i]-0.3),
                          color=CADD, fontsize=7.5, fontweight='bold')

# ── D: Scatter concordance ────────────────────────────────────
ax_d = fig.add_axes([0.76, 0.13, 0.22, 0.70])
sax(ax_d)
sc_cols = [CGOLD if confirmed[i] else CADD for i in range(N_DRUGS)]
ax_d.scatter(ADDS_RANK, GT_bliss, s=[140 if confirmed[i] else 80 for i in range(N_DRUGS)],
             c=sc_cols, marker='D', zorder=5, label='ADDS (Diamond)')
ax_d.scatter(BASE_RANK, GT_bliss, s=[100 if confirmed[i] else 60 for i in range(N_DRUGS)],
             c=CBASE, marker='s', zorder=4, alpha=0.7, label='Baseline (Square)')

for i in range(N_DRUGS):
    ax_d.annotate(short[i], (ADDS_RANK[i], GT_bliss[i]),
                  xytext=(5,2), textcoords='offset points',
                  color=CGOLD if confirmed[i] else CWHITE, fontsize=7.5,
                  fontweight='bold' if confirmed[i] else 'normal')

xfit = np.linspace(1, N_DRUGS, 50)
m_a, b_a = np.polyfit(ADDS_RANK, GT_bliss, 1)
m_b, b_b = np.polyfit(BASE_RANK, GT_bliss, 1)
ax_d.plot(xfit, m_a*xfit+b_a, color=CADD,  lw=2.0, ls='-',  alpha=0.85, label=f'ADDS rho={rho_a:.3f}')
ax_d.plot(xfit, m_b*xfit+b_b, color=CBASE, lw=2.0, ls='--', alpha=0.70, label=f'Base rho={rho_b:.3f}')

ax_d.set_xlabel('DL Predicted Rank (1=best)', color=CWHITE, fontsize=10)
ax_d.set_ylabel('GT Bliss Score (★ NatureComm / # ADDS)', color=CWHITE, fontsize=10)
ax_d.set_title('D. DL Predicted Rank vs GT Bliss\n(Spearman Concordance)',
               color=CWHITE, fontsize=10, fontweight='bold')
ax_d.set_xticks(range(1, N_DRUGS+1))
ax_d.yaxis.grid(True, color='#1A2A3A', lw=0.5, zorder=0)
ax_d.legend(fontsize=8.5, facecolor='#0D1A2E', edgecolor='#2A4A6A', labelcolor=CWHITE)
ax_d.text(0.5, 0.03,
          f'ADDS  rho={rho_a:.3f}  tau={tau_a:.3f}  Top-2={t2a:.0%}\n'
          f'Base  rho={rho_b:.3f}  tau={tau_b:.3f}  Top-2={t2b:.0%}\n'
          f'Mean |rank err|: ADDS={err_a.mean():.2f}  Base={err_b.mean():.2f}',
          transform=ax_d.transAxes, color=CWHITE, fontsize=8.5,
          ha='center', va='bottom',
          bbox=dict(boxstyle='round,pad=0.4', facecolor='#0D1A2E', edgecolor=CADD, lw=1.2))

fig.text(0.5, 0.04,
         f'ADDS: 4-modal DL (128-dim Cellpose + 256-dim RNA-seq + 32-dim PKPD★ + 64-dim CT → 480-dim Fusion MLP)'
         f'  |  Baseline: same architecture, PKPD features zero-padded (no drug-specific context)'
         f'  |  n={N_PAT} patients per drug  |  ★=NatureComm  #=ADDS estimate  seed={SEED}',
         color=CGRAY, fontsize=7.5, ha='center',
         bbox=dict(boxstyle='round,pad=0.3', facecolor='#0D1A2E', edgecolor='#2A3A5A', lw=0.8))

plt.savefig(r'f:\ADDS\figures\pritamab_dl_concordance.png',
            dpi=150, bbox_inches='tight', facecolor=BG)
plt.close()
print('\nSaved: f:\\ADDS\\figures\\pritamab_dl_concordance.png')
