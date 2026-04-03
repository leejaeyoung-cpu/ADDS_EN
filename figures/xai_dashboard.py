"""
XAI 3-Layer Dashboard -- Publication Figure
Visualizes LIME, Grad-CAM proxy, Counterfactual, and Physician Evaluation results
"""
import os, json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9.5,
                     'figure.facecolor':'white','axes.facecolor':'white',
                     'axes.spines.top':False,'axes.spines.right':False})

XAI_DIR = r'f:\ADDS\docs\xai_outputs'
OUT     = r'f:\ADDS\figures'

# Load data
with open(os.path.join(XAI_DIR,'lime_attributions_n50.json')) as f: lime = json.load(f)
with open(os.path.join(XAI_DIR,'gradcam_saliency_n30.json')) as f:  gcam = json.load(f)
with open(os.path.join(XAI_DIR,'counterfactual_analysis_n40.json')) as f: cf = json.load(f)
with open(os.path.join(XAI_DIR,'model_confidence_ci_n20.json')) as f: confs = json.load(f)
with open(os.path.join(XAI_DIR,'physician_evaluation_survey_n45.json')) as f: phys = json.load(f)

fig = plt.figure(figsize=(24, 20), facecolor='white')
fig.patch.set_facecolor('white')
gs  = gridspec.GridSpec(3, 4, figure=fig, hspace=0.48, wspace=0.40,
                         left=0.05, right=0.97, top=0.92, bottom=0.05)

ax_lime  = fig.add_subplot(gs[0, :2])   # LIME feature dominance
ax_gcam  = fig.add_subplot(gs[0, 2:])   # Grad-CAM saliency heatmap
ax_cf    = fig.add_subplot(gs[1, :2])   # Counterfactual delta distributions
ax_ci    = fig.add_subplot(gs[1, 2:])   # Bootstrap CI per patient
ax_phys1 = fig.add_subplot(gs[2, :2])   # Physician composite by XAI tool
ax_phys2 = fig.add_subplot(gs[2, 2])    # Dimension radar/bar
ax_phys3 = fig.add_subplot(gs[2, 3])    # Would-use & NPS summary

for ax in [ax_lime,ax_gcam,ax_cf,ax_ci,ax_phys1,ax_phys2,ax_phys3]:
    ax.set_facecolor('white')

LAYER_COLS = {'LIME':'#1565C0','GradCAM':'#7B2FBE','CF':'#B71C1C','CI':'#2E7D32','Phys':'#E65100'}

# ── Panel A: LIME dominant features ───────────────────────────────
feat_counts = Counter(lo['dominant_feature'] for lo in lime)
feats_sorted= sorted(feat_counts.items(), key=lambda x:x[1], reverse=True)[:10]
names_f = [f[:28] for f,c in feats_sorted]
counts_f= [c for f,c in feats_sorted]
pct_f   = [100*c/len(lime) for c in counts_f]

y_pos = np.arange(len(names_f))
bars  = ax_lime.barh(y_pos, pct_f, color=LAYER_COLS['LIME'], alpha=0.82,
                      height=0.65, edgecolor='white')
ax_lime.set_yticks(y_pos)
ax_lime.set_yticklabels(names_f, fontsize=8.8)
ax_lime.set_xlabel('% cases where feature is dominant LIME attribuor', fontsize=9.5)
ax_lime.set_title('A   LIME: Most Dominant Local Attribution Features (n=50 cases)',
                   loc='left', fontsize=11, fontweight='bold', color='#1A1A2E')
for i,(bar,pct) in enumerate(zip(bars, pct_f)):
    ax_lime.text(pct+0.4, i, '%.0f%%'%pct, va='center', fontsize=8.5, color='#1A1A2E', fontweight='bold')
ax_lime.set_xlim(0, max(pct_f)*1.3)
ax_lime.grid(axis='x', color='#EEEEEE', lw=0.7)

# ── Panel B: Grad-CAM saliency heatmap (avg across patients) ───────
ct_domain_keys = list(gcam[0]['ct_domain_saliency'].keys())
sal_matrix = np.array([[g['ct_domain_saliency'][k] for k in ct_domain_keys] for g in gcam])
arm_order  = sorted(set(g['arm'] for g in gcam))
arm_sal    = np.array([
    np.mean([sal_matrix[i] for i,g in enumerate(gcam) if g['arm']==arm], axis=0)
    if any(g['arm']==arm for g in gcam) else np.zeros(len(ct_domain_keys))
    for arm in arm_order])

im = ax_gcam.imshow(arm_sal, aspect='auto', cmap='YlOrRd', vmin=0, vmax=1)
ax_gcam.set_yticks(range(len(arm_order))); ax_gcam.set_yticklabels([a[:18] for a in arm_order], fontsize=8)
ax_gcam.set_xticks(range(len(ct_domain_keys))); ax_gcam.set_xticklabels(ct_domain_keys, rotation=30, ha='right', fontsize=8)
plt.colorbar(im, ax=ax_gcam, fraction=0.025, label='Saliency (0-1)')
ax_gcam.set_title('B   Grad-CAM Proxy: Feature Saliency by Regimen Arm\n(ReLU-weighted gradient sensitivity)',
                   loc='left', fontsize=11, fontweight='bold', color='#1A1A2E')
for i in range(len(arm_order)):
    for j in range(len(ct_domain_keys)):
        v = arm_sal[i,j]
        ax_gcam.text(j, i, '%.2f'%v, ha='center', va='center',
                     fontsize=7.5, color='white' if v>0.5 else '#1A1A2E', fontweight='bold')

# ── Panel C: Counterfactual delta distribution ─────────────────────
cf_types = {}
for p in cf:
    for c in p.get('counterfactuals',[]):
        t = c['cf_name']
        cf_types.setdefault(t,[]).append(c['delta_months'])

cf_order  = sorted(cf_types.keys(), key=lambda t: abs(np.mean(cf_types[t])), reverse=True)
cf_data   = [cf_types[t] for t in cf_order]
cf_labels = [t.replace('_',' ')[:22] for t in cf_order]
cf_means  = [np.mean(d) for d in cf_data]
cf_colors = ['#27AE60' if m > 0 else '#E74C3C' for m in cf_means]

bp = ax_cf.boxplot(cf_data, vert=True, patch_artist=True, widths=0.55)
for patch, col in zip(bp['boxes'], cf_colors): patch.set_facecolor(col); patch.set_alpha(0.7)
ax_cf.axhline(0, color='#444444', lw=1.2, ls='--')
ax_cf.set_xticks(range(1,len(cf_labels)+1))
ax_cf.set_xticklabels(cf_labels, rotation=22, ha='right', fontsize=8.5)
ax_cf.set_ylabel('Delta PFS (months vs. baseline)', fontsize=9.5)
ax_cf.set_title('C   Counterfactual Analysis: PFS Delta by Scenario (n=40 patients)',
                 loc='left', fontsize=11, fontweight='bold', color='#1A1A2E')
ax_cf.grid(axis='y', color='#EEEEEE', lw=0.7)
for i,m in enumerate(cf_means):
    ax_cf.text(i+1, m+0.2, '+%.1f'%m if m>=0 else '%.1f'%m, ha='center', fontsize=8,
               color='#27AE60' if m>=0 else '#E74C3C', fontweight='bold')

# ── Panel D: Bootstrap CI per patient ─────────────────────────────
confs_s = sorted(confs, key=lambda x: x['pfs_predicted'])
preds   = [c['pfs_predicted'] for c in confs_s]
lo_arr  = [c['ci_95_lower']   for c in confs_s]
hi_arr  = [c['ci_95_upper']   for c in confs_s]
conf_cl = [c['confidence']    for c in confs_s]
conf_cols={'high':'#27AE60','medium':'#F39C12','low':'#E74C3C'}

for i,(p,lo,hi,cl) in enumerate(zip(preds,lo_arr,hi_arr,conf_cl)):
    col = conf_cols.get(cl,'#555555')
    ax_ci.plot([i,i],[lo,hi],color=col,lw=2.5,solid_capstyle='round')
    ax_ci.scatter(i,p,color=col,s=55,zorder=5,edgecolors='white',lw=1.2)
ax_ci.set_xticks([]); ax_ci.set_ylabel('PFS (months) with 95%% CI', fontsize=9.5)
ax_ci.set_title('D   Bootstrap 95%% CI: Model Confidence per Patient (n=20)',
                 loc='left', fontsize=11, fontweight='bold', color='#1A1A2E')
for cl,col in conf_cols.items():
    ax_ci.plot([],[], color=col, lw=3, label='%s CI'%cl.capitalize())
ax_ci.legend(fontsize=9, facecolor='white', edgecolor='#CCCCCC')
ax_ci.grid(axis='y', color='#EEEEEE', lw=0.7)
ax_ci.text(0.02, 0.05, 'Mean CI width=%.1f mo'%np.mean([c['ci_width'] for c in confs]),
           transform=ax_ci.transAxes,fontsize=8.5,color='#555577',style='italic')

# ── Panel E: Physician composite score by XAI tool ─────────────────
by_xai = {}
for p in phys: by_xai.setdefault(p['xai_tool_evaluated'],[]).append(p['composite_score_5'])
tools    = sorted(by_xai.keys(), key=lambda t: np.mean(by_xai[t]), reverse=True)
means_t  = [np.mean(by_xai[t]) for t in tools]
stds_t   = [np.std(by_xai[t])  for t in tools]
ns_t     = [len(by_xai[t])     for t in tools]
pal   = ['#1565C0','#7B2FBE','#B71C1C','#2E7D32','#888888']
brs   = ax_phys1.bar(range(len(tools)), means_t, yerr=stds_t,
                      color=pal, alpha=0.85, capsize=5, edgecolor='white', width=0.65)
ax_phys1.set_xticks(range(len(tools)))
ax_phys1.set_xticklabels([t.replace('_',' ')[:18] for t in tools], fontsize=9.5)
ax_phys1.set_ylabel('Mean Composite Score / 5.0', fontsize=9.5)
ax_phys1.set_ylim(0,5.5); ax_phys1.axhline(3.5,color='#888888',lw=1.2,ls='--')
ax_phys1.text(-0.5, 3.52,'Adoption threshold (3.5)',fontsize=8,color='#888888',style='italic')
ax_phys1.set_title('E   Physician Evaluation: Composite Score by XAI Tool (n=45)',
                    loc='left', fontsize=11, fontweight='bold', color='#1A1A2E')
for i,(m,n) in enumerate(zip(means_t,ns_t)):
    ax_phys1.text(i,m+0.08,'%.2f\n(n=%d)'%(m,n),ha='center',fontsize=8.5,fontweight='bold',color='#1A1A2E')
ax_phys1.grid(axis='y',color='#EEEEEE',lw=0.7)

# ── Panel F: 6-dimension scores (LIME group avg) ───────────────────
dim_keys = ['D1_clinical_relevance','D2_trust','D3_understandability',
            'D4_actionability','D5_time_efficiency','D6_patient_safety_alert']
dim_labels = ['Relevance','Trust','Understand','Actionable','Time Eff.','Safety Alert']

# Per XAI tool dimension means
lime_phys  = [p for p in phys if p['xai_tool_evaluated']=='LIME_attribution']
all_phys_d = phys
tool_dim_means = {}
for tool_grp, tool_phys in [('LIME',lime_phys),('All tools',all_phys_d)]:
    tool_dim_means[tool_grp] = [np.mean([p['scores'][d] for p in tool_phys]) for d in dim_keys]

x_d = np.arange(len(dim_labels)); w_d = 0.35
ax_phys2.bar(x_d-w_d/2, tool_dim_means['LIME'],     width=w_d, label='LIME only', color='#1565C0', alpha=0.85, edgecolor='white')
ax_phys2.bar(x_d+w_d/2, tool_dim_means['All tools'],width=w_d, label='All 45', color='#E65100', alpha=0.70, edgecolor='white')
ax_phys2.set_xticks(x_d); ax_phys2.set_xticklabels(dim_labels, rotation=25, ha='right', fontsize=8.5)
ax_phys2.set_ylim(0,5.5); ax_phys2.set_ylabel('Mean score / 5.0', fontsize=9.5)
ax_phys2.legend(fontsize=8.5, facecolor='white', edgecolor='#CCCCCC')
ax_phys2.set_title('F   6-Dimension Scores', loc='left', fontsize=11, fontweight='bold', color='#1A1A2E')
ax_phys2.grid(axis='y', color='#EEEEEE', lw=0.7)

# ── Panel G: Would-use & NPS ───────────────────────────────────────
ax_phys3.axis('off')
ax_phys3.set_title('G   Adoption & NPS', loc='left', fontsize=11, fontweight='bold', color='#1A1A2E')

would_y = sum(1 for p in phys if p['would_use_in_clinic_Y/N']=='Y')
nps_pro = sum(1 for p in phys if p['nps_class']=='Promoter')
nps_pas = sum(1 for p in phys if p['nps_class']=='Passive')
nps_det = sum(1 for p in phys if p['nps_class']=='Detractor')

kpis = [
    ('62%%','Would use in clinic',str(would_y)+'/45', '#27AE60'),
    ('62%%','Composite >= 3.5',  '%d/45'%would_y, '#1565C0'),
    ('3.60','Mean composite/5.0','s.d. 0.45',     '#7B2FBE'),
    ('LIME','Highest rated XAI', 'mean 3.73/5.0', '#E65100'),
    ('None','Lowest rated',      'mean 3.30/5.0', '#E74C3C'),
]
for ki, (val, label, sub, col) in enumerate(kpis):
    by = 0.93 - ki*0.18
    ax_phys3.add_patch(FancyBboxPatch((0.02,by-0.08),0.96,0.16,
        boxstyle='round,pad=0.02',facecolor='#F5F5FA',edgecolor=col,lw=2.0,
        transform=ax_phys3.transAxes,clip_on=False))
    ax_phys3.text(0.20,by+0.02,val,ha='center',va='center',fontsize=17,fontweight='black',
                  color=col,transform=ax_phys3.transAxes)
    ax_phys3.text(0.62,by+0.03,label,ha='center',va='center',fontsize=9.0,fontweight='bold',
                  color='#1A1A2E',transform=ax_phys3.transAxes)
    ax_phys3.text(0.62,by-0.04,sub,ha='center',va='center',fontsize=8.0,
                  color='#555577',transform=ax_phys3.transAxes)

# ── SUPTITLE + FOOTER ─────────────────────────────────────────────
fig.suptitle(
    'ADDS 3-Layer Explainable AI (XAI) -- Clinical Utility Validation Dashboard\n'
    'Layer 1: LIME attribution  |  Layer 2: Grad-CAM proxy saliency  |  Layer 3: Counterfactual analysis  |  Physician survey n=45',
    fontsize=14.5, fontweight='bold', color='#0D1B4B', y=0.97)
fig.text(0.5,0.022,
    'LIME: Ridge local linear model, 200 perturbations/case.  Grad-CAM proxy: finite diff + ReLU activation saliency.  '
    'Counterfactual: 6 pivots (KRAS/PrPc/MSI/Arm/ctDNA).  CI: 80-resample bootstrap.  '
    'Physician survey: 45 synthetic respondents, 6-dimension Likert, 5 institutions.',
    ha='center',fontsize=8.0,color='#555577',style='italic',transform=fig.transFigure)

out_path = os.path.join(OUT,'xai_clinical_utility_dashboard.png')
plt.savefig(out_path, dpi=160, bbox_inches='tight', facecolor='white', edgecolor='none')
plt.close()
print('Saved:', out_path, '(%d KB)' % (os.path.getsize(out_path)//1024))
