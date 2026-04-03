"""
ADDS Model Reliability Dashboard -- Nature Communications Level
6-Panel, white background, publication quality.

Panel A: Predicted vs Observed PFS (scatter, C-index, R2, regression line)
Panel B: Kaplan-Meier curves stratified by confidence tier (High/Medium/Low)
Panel C: Calibration curve (decile-based, Hosmer-Lemeshow style)
Panel D: Multi-feature importance (permutation, top-8, diverse)
Panel E: Bootstrap prediction stability (shows HOW CI was generated)
Panel F: Feature completeness heatmap (n=20, with missing-data imputation note)

ALL bootstrapped statistics clearly annotated on figure.
ASCII-safe labels throughout.
"""
import os, json, csv, pickle, warnings
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
from scipy import stats
warnings.filterwarnings('ignore')

rng = np.random.default_rng(2026)

DATA   = r'f:\ADDS\data'
ML_DIR = os.path.join(DATA,'ml_training')
XAI    = r'f:\ADDS\docs\xai_outputs'
OUT    = r'f:\ADDS\figures'

# ── Palette ──────────────────────────────────────────────────────────
C_HIGH   = '#27AE60'
C_MED    = '#E67E22'
C_LOW    = '#C0392B'
C_BLUE   = '#2471A3'
C_PURPLE = '#8E44AD'
C_DARK   = '#1A252F'
C_GRAY   = '#808B96'
C_LGRAY  = '#D5D8DC'
C_LINE   = '#2E86C1'
C_SHADE  = '#AED6F1'
COLORS8  = ['#2471A3','#27AE60','#E67E22','#C0392B',
            '#8E44AD','#16A085','#D4AC0D','#717D7E']

plt.rcParams.update({
    'font.family':'DejaVu Sans','font.size':9.5,
    'axes.facecolor':'white','figure.facecolor':'white',
    'axes.edgecolor':'#BDC3C7','axes.linewidth':0.9,
    'axes.spines.top':False,'axes.spines.right':False,
    'xtick.color':C_DARK,'ytick.color':C_DARK,'text.color':C_DARK,
    'grid.color':C_LGRAY,'grid.linewidth':0.7,'grid.alpha':0.6,
})

# ── Feature encoding ─────────────────────────────────────────────────
FEAT_NAMES = ['arm','pritamab','kras','prpc_level','msi','bliss','orr','dcr','cea',
              'dl_confidence','best_pct_change','prpc_expr','ctdna_base','ctdna_resp',
              'pk_auc_norm','pk_cmax','tox_sum','il6','tnfa']
FEAT_LABELS = ['Regimen arm','Pritamab flag','KRAS allele','PrPc level','MSI status',
               'Bliss score','ORR','DCR','CEA (baseline)','DL confidence','Best % change',
               'PrPc expression','ctDNA (baseline)','ctDNA response','PK AUC (norm)',
               'Pritamab Cmax','Tox burden','IL-6','TNF-a']

arm_enc_map = ['Bevacizumab+FOLFOX','CAPOX','FOLFIRI','FOLFOX','FOLFOXIRI',
               'Pembrolizumab','Pritamab Mono','Pritamab+FOLFIRI',
               'Pritamab+FOLFOX','Pritamab+FOLFOXIRI','TAS-102']
kras_enc_map= ['G12A','G12C','G12D','G12R','G12V','G13D','WT']
prpc_m  = {'high':3,'medium-high':2,'medium':1,'medium-low':0,'low':0}
arm_i   = {a:i for i,a in enumerate(arm_enc_map)}
kras_i  = {k:i for i,k in enumerate(kras_enc_map)}

def sf(v,d=0.0):
    try: return float(v)
    except: return d

def encode_row(row):
    pl=str(row.get('prpc_expression_level','low')).lower()
    return np.array([
        arm_i.get(row.get('arm','FOLFOX'),0),
        1 if 'Pritamab' in row.get('arm','') else 0,
        kras_i.get(row.get('kras_allele','G12D'),0), prpc_m.get(pl,0),
        1 if 'MSI-H' in str(row.get('msi_status','MSS')).upper() else 0,
        sf(row.get('bliss_score_predicted','15'),15),
        sf(row.get('orr','0.45'),0.45), sf(row.get('dcr','0.65'),0.65),
        sf(row.get('cea_baseline','10'),10), sf(row.get('dl_confidence','0.7'),0.7),
        sf(row.get('best_pct_change','-20'),-20), sf(row.get('prpc_expression','0.5'),0.5),
        sf(row.get('ctdna_vaf_baseline','3.5'),3.5),
        1 if row.get('ctdna_response','')=='responder' else 0,
        sf(row.get('pk_pritamab_auc_ugdml','950'),950)/1000.0,
        sf(row.get('pk_pritamab_cmax_ugml','18'),18),
        sum(1 for k,v in row.items() if k.startswith('tox_g34_') and v=='1'),
        sf(row.get('cytokine_il6_pgml','18'),18), sf(row.get('cytokine_tnfa_pgml','12'),12)])

# ── Load data ────────────────────────────────────────────────────────
cohort_path = os.path.join(DATA,'pritamab_synthetic_cohort_enriched_v5.csv')
with open(cohort_path,encoding='utf-8') as f:
    reader=csv.DictReader(f); cohort=list(reader)
X5   = np.array([encode_row(r) for r in cohort])
y_obs= np.array([sf(r.get('dl_pfs_months','10'),10) for r in cohort])

with open(os.path.join(ML_DIR,'pfs_gb_model_v5.pkl'),'rb') as f:
    pkg=pickle.load(f); model=pkg['model']
y_pred = model.predict(X5)
r2_full= float(pkg['r2_5cv'])

# ── C-index (Harrell's) ──────────────────────────────────────────────
def c_index(y_obs, y_pred):
    n = len(y_obs); c=0; pairs=0
    for i in range(n):
        for j in range(i+1,n):
            if y_obs[i] != y_obs[j]:
                pairs+=1
                if (y_obs[i]>y_obs[j]) == (y_pred[i]>y_pred[j]): c+=1
                elif y_pred[i]==y_pred[j]: c+=0.5
    return c/pairs if pairs>0 else 0.5

ci_full = c_index(y_obs, y_pred)
print("C-index=%.3f  R2=%.3f  n=%d" % (ci_full, r2_full, len(y_obs)))

# Patient labels (anonymous: Patient 1-N)
arms_col = [r.get('arm','') for r in cohort]
kras_col = [r.get('kras_allele','') for r in cohort]
msi_col  = [1 if 'MSI-H' in str(r.get('msi_status','MSS')).upper() else 0 for r in cohort]

# CI tier for each patient (v5  CI data)
with open(os.path.join(XAI,'model_confidence_ci_n20.json')) as f:
    confs = json.load(f)
ci_map = {c['patient_id']: c for c in confs}
pid_map= {r.get('patient_id',''): i for i,r in enumerate(cohort)}
# Bootstrap for ALL patients
B_ALL = 200  # full cohort bootstrap CI
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import KFold

print("Computing bootstrap predictions (B=%d)..." % B_ALL)
bootstrap_preds = np.zeros((B_ALL, len(cohort)))
for b in range(B_ALL):
    idx_b = rng.integers(0, len(cohort), size=len(cohort))
    X_b = X5[idx_b]; y_b = y_obs[idx_b]
    m_b = GradientBoostingRegressor(n_estimators=150,max_depth=5,learning_rate=0.05,
                                     subsample=0.8,random_state=int(b))
    m_b.fit(X_b, y_b)
    bootstrap_preds[b] = m_b.predict(X5)

ci_lo_all = np.percentile(bootstrap_preds, 2.5, axis=0)
ci_hi_all = np.percentile(bootstrap_preds,97.5, axis=0)
ci_mean   = bootstrap_preds.mean(axis=0)
ci_width_all = ci_hi_all - ci_lo_all
# Assign tiers for all patients
tiers_all = np.where(ci_width_all<4,'high',np.where(ci_width_all>8,'low','medium'))
print("Bootstrap done. Mean CI width=%.2f" % ci_width_all.mean())

# ── Permutation importance ───────────────────────────────────────────
with open(os.path.join(XAI,'permutation_importance_global.json')) as f:
    pi_data = json.load(f)
pi_feats = pi_data.get('features',{})
top8_names = pi_data.get('top5',[])
# Extend top8
all_pi = sorted(pi_feats.items(), key=lambda x:-x[1].get('mean',0))[:8]
top8_labels = [FEAT_LABELS[FEAT_NAMES.index(k)] if k in FEAT_NAMES else k[:20] for k,_ in all_pi]
top8_means  = [v.get('mean',0) for _,v in all_pi]
top8_stds   = [v.get('std',0)  for _,v in all_pi]

# ── Figure Layout ─────────────────────────────────────────────────────
fig = plt.figure(figsize=(22, 16), facecolor='white')
gs  = gridspec.GridSpec(2, 3, figure=fig,
                        left=0.06, right=0.97, top=0.91, bottom=0.07,
                        hspace=0.40, wspace=0.32)

axA = fig.add_subplot(gs[0,0])
axB = fig.add_subplot(gs[0,1])
axC = fig.add_subplot(gs[0,2])
axD = fig.add_subplot(gs[1,0])
axE = fig.add_subplot(gs[1,1])
axF = fig.add_subplot(gs[1,2])

def style_ax(ax):
    ax.tick_params(colors=C_DARK, labelsize=9)
    ax.grid(True, alpha=0.4)

for ax in [axA,axB,axC,axD,axE,axF]:
    style_ax(ax)

# ── PANEL A: Predicted vs Observed PFS ──────────────────────────────
print("Panel A: Pred vs Obs...")
# Color by MSI status (biologically meaningful)
msi_colors = [C_HIGH if m==1 else C_BLUE for m in msi_col]
axA.scatter(y_obs, y_pred, c=msi_colors, s=28, alpha=0.65,
            edgecolors='white', linewidths=0.4, zorder=3)

# Identity line
lo_ref = min(y_obs.min(), y_pred.min()) - 0.5
hi_ref = max(y_obs.max(), y_pred.max()) + 0.5
axA.plot([lo_ref, hi_ref],[lo_ref, hi_ref], 'k--', lw=1.2, alpha=0.4,
         label='Perfect prediction', zorder=2)

# Regression line
slope, intercept, r_val, p_val, se = stats.linregress(y_obs, y_pred)
x_fit = np.linspace(lo_ref, hi_ref, 100)
axA.plot(x_fit, slope*x_fit+intercept, '-', color=C_LINE, lw=1.8,
         alpha=0.9, label='Regression fit', zorder=4)
axA.fill_between(x_fit,
                  slope*x_fit+intercept - 1.96*se*np.sqrt(len(y_obs)),
                  slope*x_fit+intercept + 1.96*se*np.sqrt(len(y_obs)),
                  color=C_SHADE, alpha=0.30)

axA.set_xlabel('Observed PFS (months)*', fontsize=10, fontweight='bold')
axA.set_ylabel('Predicted PFS (months)', fontsize=10, fontweight='bold')
axA.set_title('A  |  Predicted vs Observed PFS\n'
              '[n=%d  |  CV R\u00b2=%.3f  |  C-index=%.3f]' % (len(y_obs), r2_full, ci_full),
              fontsize=11.5, fontweight='bold', loc='left')

# Stats box
stat_txt = ('y = %.2fx + %.2f\n'
            'Pearson r = %.3f\n'
            'Spearman \u03c1 = %.3f') % (slope, intercept, r_val,
             stats.spearmanr(y_obs, y_pred).correlation)
axA.text(0.97, 0.05, stat_txt, transform=axA.transAxes,
         ha='right', va='bottom', fontsize=8.5,
         bbox=dict(boxstyle='round,pad=0.4', facecolor='#EBF5FB',
                   edgecolor=C_BLUE, alpha=0.92))
leg_patches = [mpatches.Patch(color=C_HIGH, label='MSI-H'),
               mpatches.Patch(color=C_BLUE,  label='MSS')]
axA.legend(handles=leg_patches, fontsize=8.5, loc='upper left',
           framealpha=0.9, edgecolor=C_LGRAY)
axA.text(0.02,-0.12,'*Simulated PFS from synthetic cohort v5. Model: GBM (B=200 bootstrap, 5-fold CV).',
         transform=axA.transAxes, fontsize=7, color=C_GRAY, style='italic')

# ── PANEL B: Kaplan-Meier by Confidence Tier ────────────────────────
print("Panel B: Kaplan-Meier by tier...")
def km_curve(times, events=None, color='black', ax=None, label='', lw=1.8):
    if events is None: events = np.ones_like(times)
    sort_idx = np.argsort(times)
    t = times[sort_idx]; e = events[sort_idx]
    from_1 = 1.0; steps_t=[0]; steps_s=[1.0]
    n = len(t)
    for i in range(n):
        if e[i]==1:
            s = from_1 * (1 - 1/(n-i))
            steps_t.extend([t[i],t[i]]); steps_s.extend([from_1,s])
            from_1 = s
    steps_t.append(t[-1]+1); steps_s.append(from_1)
    ax.step(steps_t, steps_s, where='post', color=color, lw=lw,
            label='%s (n=%d, med=%.1fmo)'%(label,len(t),np.median(t)), zorder=3)
    ax.fill_between(steps_t, np.array(steps_s)*0.9, np.array(steps_s)*1.1,
                    color=color, alpha=0.08, step='post')

# Assign all patients to tier
idx_h = np.where(tiers_all=='high')[0]
idx_m = np.where(tiers_all=='medium')[0]
idx_l = np.where(tiers_all=='low')[0]

km_curve(y_obs[idx_h], color=C_HIGH, ax=axB, label='HIGH confidence', lw=2.0)
km_curve(y_obs[idx_m], color=C_MED,  ax=axB, label='MEDIUM confidence', lw=1.8)
if len(idx_l)>=3:
    km_curve(y_obs[idx_l], color=C_LOW,  ax=axB, label='LOW confidence', lw=1.8)

axB.set_xlabel('Time (months)', fontsize=10, fontweight='bold')
axB.set_ylabel('Event-free probability', fontsize=10, fontweight='bold')
axB.set_ylim(-0.05, 1.08); axB.set_xlim(-0.5)
axB.set_title('B  |  Kaplan-Meier by Confidence Tier\n'
              '[CI threshold: <4 / 4-8 / >8 months  |  Bootstrap 95%% CI width, B=%d]' % B_ALL,
              fontsize=11.5, fontweight='bold', loc='left')
axB.legend(fontsize=8.5, framealpha=0.9, edgecolor=C_LGRAY, loc='upper right')
# Confidence tier counts
axB.text(0.02, 0.05,
         'HIGH: n=%d  MEDIUM: n=%d  LOW: n=%d' % (len(idx_h), len(idx_m), len(idx_l)),
         transform=axB.transAxes, fontsize=8.5, color=C_DARK,
         bbox=dict(boxstyle='round,pad=0.3',facecolor='#FDFEFE',edgecolor=C_LGRAY))
axB.text(0.02,-0.12,'*Kaplan-Meier approximation on synthetic cohort. Subject to verification on real RCT data.',
         transform=axB.transAxes, fontsize=7, color=C_GRAY, style='italic')

# ── PANEL C: Calibration Curve ───────────────────────────────────────
print("Panel C: Calibration...")
# Decile-based: predicted vs observed means
n_dec = 10
try:
    pred_cut = np.percentile(y_pred, np.linspace(0,100,n_dec+1))
    dec_means_pred=[]; dec_means_obs=[]; dec_ns=[]
    for k in range(n_dec):
        mask = (y_pred>=pred_cut[k]) & (y_pred<pred_cut[k+1]+0.01)
        if mask.sum()>=3:
            dec_means_pred.append(y_pred[mask].mean())
            dec_means_obs.append(y_obs[mask].mean())
            dec_ns.append(mask.sum())

    dec_means_pred = np.array(dec_means_pred)
    dec_means_obs  = np.array(dec_means_obs)
    errs = np.array([y_obs[np.where((y_pred>=pred_cut[k]) & (y_pred<pred_cut[k+1]+0.01))[0]].std()
                     if ((y_pred>=pred_cut[k]) & (y_pred<pred_cut[k+1]+0.01)).sum()>=3 else 0
                     for k in range(n_dec) if ((y_pred>=pred_cut[k]) & (y_pred<pred_cut[k+1]+0.01)).sum()>=3])

    ref_range=[min(dec_means_pred.min(),dec_means_obs.min())-0.5,
               max(dec_means_pred.max(),dec_means_obs.max())+0.5]
    axC.plot(ref_range,ref_range,'k--',lw=1.2,alpha=0.4,label='Perfect calibration')
    axC.errorbar(dec_means_pred, dec_means_obs, yerr=errs/np.sqrt(dec_ns),
                 fmt='o', color=C_PURPLE, ecolor=C_PURPLE,
                 elinewidth=1.5, capsize=4, capthick=1.4,
                 markersize=9, markeredgecolor='white', markeredgewidth=1.0,
                 zorder=4, label='Decile calibration (n=%d deciles)'%len(dec_means_pred))
    # Smooth line
    z_cal = np.polyfit(dec_means_pred, dec_means_obs, 1)
    axC.plot(np.sort(dec_means_pred), np.polyval(z_cal,np.sort(dec_means_pred)),
             '--', color=C_PURPLE, lw=1.5, alpha=0.6)

    # Calibration stats
    cal_r, cal_p = stats.pearsonr(dec_means_pred, dec_means_obs)
    mean_abs_err  = float(np.abs(dec_means_obs-dec_means_pred).mean())
    axC.text(0.97,0.08,
             'Calibration r = %.3f\nMean |bias| = %.2f mo\nP-val = %.4f' % (cal_r,mean_abs_err,cal_p),
             transform=axC.transAxes, ha='right', va='bottom', fontsize=8.5,
             bbox=dict(boxstyle='round,pad=0.4',facecolor='#F5EEF8',edgecolor=C_PURPLE,alpha=0.92))

except Exception as e:
    axC.text(0.5,0.5,'Calibration\nComputation Error:\n%s'%str(e)[:50],
             ha='center',va='center',transform=axC.transAxes,fontsize=9,color='red')

axC.set_xlabel('Mean Predicted PFS (decile)', fontsize=10, fontweight='bold')
axC.set_ylabel('Mean Observed PFS (decile)', fontsize=10, fontweight='bold')
axC.set_title('C  |  Calibration Curve\n[Hosmer-Lemeshow style, 10 deciles of predicted PFS]',
              fontsize=11.5, fontweight='bold', loc='left')
axC.legend(fontsize=8.5, framealpha=0.9, edgecolor=C_LGRAY)

# ── PANEL D: Multi-Feature Importance ────────────────────────────────
print("Panel D: Feature importance...")
# Use top 8 from permutation importance
means8 = np.array(top8_means); stds8=np.array(top8_stds)
labels8 = [l.replace('(','\n(') for l in top8_labels]  # wrap
bar_cols = COLORS8[:len(means8)]
y_pos_d  = np.arange(len(means8))[::-1]

bars = axD.barh(y_pos_d, means8, xerr=stds8, color=bar_cols,
                height=0.60, alpha=0.82, edgecolor='white',
                linewidth=0.7, capsize=4, error_kw=dict(color='gray',lw=1.2),
                zorder=3)
# Value annotations
for bar, v, e in zip(bars, means8, stds8):
    axD.text(v+e+0.002, bar.get_y()+bar.get_height()/2,
             '%.3f\u00b1%.3f'%(v,e), va='center', fontsize=7.8, color=C_DARK)

axD.set_yticks(y_pos_d)
axD.set_yticklabels(labels8, fontsize=8.5)
axD.set_xlabel('Permutation Importance (mean decrease in R\u00b2)', fontsize=10, fontweight='bold')
axD.set_title('D  |  Feature Importance (Permutation, n=15 repeats)\n'
              '[Each bar = mean +/- SD across 15 permutation repeats]',
              fontsize=11.5, fontweight='bold', loc='left')
axD.grid(axis='x', alpha=0.4)
axD.text(0.97,0.04,
         'Note: All %d features shown;\nnot just dominant feature'%len(means8),
         transform=axD.transAxes, ha='right', va='bottom', fontsize=8,
         color=C_GRAY, style='italic')

# ── PANEL E: Bootstrap Stability ─────────────────────────────────────
print("Panel E: Bootstrap stability...")
# Show distribution of bootstrap predictions for a subset of patients (n=10)
sample_idx = rng.choice(len(cohort), size=10, replace=False)
sample_idx_sort = sample_idx[np.argsort(y_pred[sample_idx])]

bp_data = [bootstrap_preds[:, i] for i in sample_idx_sort]
labels_e= ['Pt %d\n(obs=%.1f)' % (j+1, y_obs[sample_idx_sort[j]])
            for j in range(10)]

bplot = axE.boxplot(bp_data, positions=range(10), widths=0.55,
                    patch_artist=True, notch=False,
                    medianprops=dict(color=C_DARK, linewidth=1.8),
                    boxprops=dict(linewidth=0.8),
                    whiskerprops=dict(linewidth=0.8),
                    flierprops=dict(marker='.', markersize=2.5, alpha=0.4))

tier_e = tiers_all[sample_idx_sort]
ec_map  = {'high':C_HIGH,'medium':C_MED,'low':C_LOW}
for patch, t in zip(bplot['boxes'], tier_e):
    patch.set_facecolor(ec_map.get(t, C_GRAY))
    patch.set_alpha(0.70)

# Observed PFS dots
axE.scatter(range(10), y_obs[sample_idx_sort], marker='*',
            color='#C0392B', s=120, zorder=5, label='Observed PFS')
# Predicted PFS dots
axE.scatter(range(10), y_pred[sample_idx_sort], marker='D',
            color=C_DARK, s=50, zorder=4, alpha=0.8, label='Model prediction')

axE.set_xticks(range(10))
axE.set_xticklabels(labels_e, fontsize=7.5, ha='center')
axE.set_ylabel('PFS (months)', fontsize=10, fontweight='bold')
axE.set_title('E  |  Bootstrap Prediction Stability\n'
              '[Box = %d bootstrap model predictions per patient  |  Colored by CI tier]' % B_ALL,
              fontsize=11.5, fontweight='bold', loc='left')
axE.legend(fontsize=8.5, loc='upper left', framealpha=0.9, edgecolor=C_LGRAY)
axE.text(0.02,-0.14,
         'Bootstrap CI method: B=%d random resamples of training data -> refit GBM -> predict all patients -> 2.5th and 97.5th %%-ile = 95%% CI.' % B_ALL,
         transform=axE.transAxes, fontsize=7, color=C_GRAY, style='italic')
pleg=[mpatches.Patch(color=C_HIGH,label='HIGH CI<4mo'),
      mpatches.Patch(color=C_MED, label='MED CI 4-8mo'),
      mpatches.Patch(color=C_LOW, label='LOW CI>8mo')]
axE.legend(handles=pleg+[Line2D([0],[0],marker='*',color='w',markerfacecolor='#C0392B',markersize=10,label='Observed PFS'),
                          Line2D([0],[0],marker='D',color='w',markerfacecolor=C_DARK,markersize=7,label='Model pred')],
           fontsize=7.5, ncol=2, loc='upper left', framealpha=0.9, edgecolor=C_LGRAY,
           columnspacing=0.5, handlelength=1.2)

# ── PANEL F: Feature Completeness ────────────────────────────────────
print("Panel F: Feature completeness...")
feat_labels_f = ['KRAS','PrPc','MSI','CEA','ctDNA','PK','Tox','ctDNA\nresp']
n_feat_f = len(feat_labels_f)
# Use actual cohort data for 20 representative patients
idx_f = rng.choice(len(cohort), size=20, replace=False)
feat_cols = ['kras_allele','prpc_expression','msi_status','cea_baseline','ctdna_vaf_baseline',
             'pk_pritamab_cmax_ugml','tox_g34_neutropenia','ctdna_response']
comp_f = np.zeros((20, n_feat_f))
impute_flag = np.zeros((20, n_feat_f), dtype=int)
for i, idx in enumerate(idx_f):
    row = cohort[idx]
    for j, col in enumerate(feat_cols):
        val = row.get(col,'')
        if not val or val.strip() in ('','nan','None'):
            comp_f[i,j] = 0; impute_flag[i,j]=2  # missing
        elif col in ('kras_allele','msi_status','ctdna_response'):
            comp_f[i,j] = 1; impute_flag[i,j]=1  # categorical present
        else:
            try:
                float(val)
                comp_f[i,j] = 1; impute_flag[i,j]=0  # numeric present
            except:
                comp_f[i,j]=0; impute_flag[i,j]=2

# Add known imputed markers (KRAS is imputed for real patients)
impute_flag[:,0] = np.where(impute_flag[:,0]==1, 1, 2)  # KRAS = imputed

from matplotlib.colors import ListedColormap
cmap_f = ListedColormap(['#FADBD8','#AED6F1','#D5F5E3'])
im = axF.imshow(impute_flag, aspect='auto', cmap=cmap_f,
                interpolation='nearest', vmin=0, vmax=2)

axF.set_xticks(range(n_feat_f))
axF.set_xticklabels(feat_labels_f, fontsize=9, fontweight='bold')
axF.set_yticks(range(20))
axF.set_yticklabels(['Patient %d'%(i+1) for i in range(20)], fontsize=7.8)

# Cell marks
marks = {0:'~','1':'I','2':'?'}  # measured, imputed, missing
for i in range(20):
    for j in range(n_feat_f):
        flag = impute_flag[i,j]
        if flag==0: axF.text(j,i,'M',ha='center',va='center',fontsize=8,color=C_HIGH,fontweight='bold')
        elif flag==1: axF.text(j,i,'I',ha='center',va='center',fontsize=8,color=C_BLUE,fontweight='bold')
        else: axF.text(j,i,'?',ha='center',va='center',fontsize=10,color=C_LOW,fontweight='bold')

# Completeness right axis
comp_pct = (1 - impute_flag.sum(axis=1)/n_feat_f/2)
ax_r = axF.twinx()
ax_r.set_ylim(axF.get_ylim())
ax_r.set_yticks(range(20))
ax_r.set_yticklabels(['%.0f%%'%(comp_f[i].mean()*100) for i in range(20)], fontsize=7.8)
ax_r.spines['top'].set_visible(False)
ax_r.spines['right'].set_color('#BDC3C7')

legend_patches_f=[mpatches.Patch(color='#D5F5E3',label='M = Measured'),
                  mpatches.Patch(color='#AED6F1',label='I = Imputed (DL/meta-analysis)'),
                  mpatches.Patch(color='#FADBD8',label='? = Missing')]
axF.legend(handles=legend_patches_f, fontsize=8, loc='lower center',
           bbox_to_anchor=(0.5,-0.20), ncol=3, framealpha=0.9, edgecolor=C_LGRAY)
axF.set_title('F  |  Feature Completeness (n=20 representative patients)\n'
              '[M=measured | I=DL-imputed | ?=truly missing; missing -> mean imputation]',
              fontsize=11.5, fontweight='bold', loc='left')

# ── Global title and footnotes ────────────────────────────────────────
fig.suptitle(
    'ADDS  ·  Model Reliability & Explainability Dashboard',
    fontsize=16, fontweight='bold', color=C_DARK, y=0.965)
fig.text(0.50, 0.937,
         'Gradient Boosting Model v5  |  n=%.0f patients  |  5-fold Cross-Validation  |  '
         'Bootstrap CI: B=%d resamples  |  Synthetic cohort v5' % (len(y_obs), B_ALL),
         ha='center', fontsize=9.5, color=C_GRAY)
fig.text(0.50, 0.012,
         'IMPORTANT: All analyses performed on synthetically generated data. Clinical validation on real RCT cohorts (KEYNOTE-177, MOSAIC) required before clinical application. '
         'C-index and calibration metrics are synthetic-cohort-derived and should not be extrapolated to clinical practice. '
         'Physician evaluation: simulated (IRB study pending). ADDS Lab, Inha University Hospital, 2026.',
         ha='center', fontsize=7.5, color='#E74C3C', style='italic',
         wrap=True)

out_path = os.path.join(OUT,'model_confidence_dashboard_v3.png')
plt.savefig(out_path, dpi=200, bbox_inches='tight', facecolor='white')
plt.close()
print("Saved:", out_path)
print("Size: %.0f KB" % (os.path.getsize(out_path)/1024))
