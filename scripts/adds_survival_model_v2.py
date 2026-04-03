"""
OS/PFS Model v2 -- improved features to fix R2 < 0
Uses cohort v3 full field set including toxicity/PK/ctDNA
"""
import os, csv, json, pickle
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.model_selection import cross_val_score, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

rng = np.random.default_rng(42)

DATA   = r'f:\ADDS\data'
ML_DIR = os.path.join(DATA, 'ml_training')

cohort_path = os.path.join(DATA, 'pritamab_synthetic_cohort_enriched_v3.csv')
with open(cohort_path, encoding='utf-8') as f:
    reader = csv.DictReader(f)
    cohort = list(reader)

def safe_float(v, default=0.0):
    try: return float(v)
    except: return default

def safe_int(v, default=0):
    try: return int(float(v))
    except: return default

arm_map = {'Pritamab Mono':0,'Pritamab+FOLFOX':1,'Pritamab+FOLFIRI':2,
           'Pritamab+FOLFOXIRI':3,'FOLFOX':4,'FOLFIRI':5,'FOLFOXIRI':6,
           'CAPOX':7,'TAS-102':8,'Bevacizumab+FOLFOX':9,'Pembrolizumab':10}
kras_map= {'G12D':0,'G12V':1,'G12C':2,'G13D':3,'WT':4,'G12A':5,'G12R':6}
prpc_m  = {'high':3,'medium-high':2,'medium':1,'medium-low':0,'low':0,'':0}

X_all=[]; y_pfs=[]; y_os=[]
for row in cohort:
    arm_e  = arm_map.get(row.get('arm','FOLFOX'),4)
    prit   = 1 if 'Pritamab' in row.get('arm','') else 0
    kras_e = kras_map.get(row.get('kras_allele','G12D'),0)
    prpc_e = prpc_m.get(str(row.get('prpc_expression_level','low')).lower(),0)
    msi_e  = 1 if 'MSI-H' in str(row.get('msi_status','MSS')).upper() else 0
    bliss_v= safe_float(row.get('bliss_score_predicted',15), 15)
    orr_v  = safe_float(row.get('orr',0.4), 0.4)
    dcr_v  = safe_float(row.get('dcr',0.6), 0.6)
    cea_v  = safe_float(row.get('cea_baseline',10), 10)
    conf_v = safe_float(row.get('dl_confidence',0.7), 0.7)
    bpc_v  = safe_float(row.get('best_pct_change',-20), -20)
    prpc_n = safe_float(row.get('prpc_expression',0.5), 0.5)
    # ctDNA (new)
    ctdna  = safe_float(row.get('ctdna_vaf_baseline',3.5), 3.5)
    ctdna8 = safe_float(row.get('ctdna_vaf_week8',1.5), 1.5)
    ctdna_r= 1 if row.get('ctdna_response','non-responder')=='responder' else 0
    # PK (new)
    pk_auc = safe_float(row.get('pk_pritamab_auc_ugdml',950), 950)
    pk_cmax= safe_float(row.get('pk_pritamab_cmax_ugml',18), 18)
    # Toxicity burden
    tox_sum= sum(safe_int(v) for k,v in row.items() if k.startswith('tox_g34_'))
    # Cytokines
    il6    = safe_float(row.get('cytokine_il6_pgml',18), 18)
    tnf    = safe_float(row.get('cytokine_tnfa_pgml',12), 12)

    feat = [arm_e, prit, kras_e, prpc_e, msi_e,
            bliss_v, orr_v, dcr_v, cea_v, conf_v, bpc_v, prpc_n,
            ctdna, ctdna8, ctdna_r, pk_auc, pk_cmax, tox_sum, il6, tnf]

    pfs_v = safe_float(row.get('dl_pfs_months', row.get('pfs_months',12)), 12)
    os_v  = safe_float(row.get('dl_os_months',  row.get('os_months',18)),  18)
    X_all.append(feat); y_pfs.append(pfs_v); y_os.append(os_v)

X = np.array(X_all); y_pfs = np.array(y_pfs); y_os = np.array(y_os)
# Add small noise to break degenerate patterns
X += rng.normal(0, 0.01, X.shape)

print("Features: %d x %d" % X.shape)
print("PFS range: %.1f - %.1f" % (y_pfs.min(), y_pfs.max()))
print("OS  range: %.1f - %.1f" % (y_os.min(),  y_os.max()))

feat_names = ['arm','pritamab','kras','prpc_level','msi','bliss','orr','dcr','cea',
              'dl_confidence','best_pct_change','prpc_expr','ctdna_base','ctdna_wk8',
              'ctdna_resp','pk_auc','pk_cmax','tox_sum','il6','tnfa']

best_pfs_r2 = -999.0; best_pfs_model = None
best_os_r2  = -999.0; best_os_model  = None

configs = [
    GradientBoostingRegressor(n_estimators=300, max_depth=4, learning_rate=0.03,
                               subsample=0.8, min_samples_leaf=10, random_state=2026),
    GradientBoostingRegressor(n_estimators=500, max_depth=3, learning_rate=0.02,
                               subsample=0.7, min_samples_leaf=15, random_state=42),
    RandomForestRegressor(n_estimators=300, max_depth=10, min_samples_leaf=8,
                           random_state=2026, n_jobs=-1),
]

kf = KFold(n_splits=5, shuffle=True, random_state=42)
for model in configs:
    cv_p = cross_val_score(model, X, y_pfs, cv=kf, scoring='r2')
    cv_o = cross_val_score(model, X, y_os,  cv=kf, scoring='r2')
    print("  %s | PFS=%.3f OS=%.3f" % (type(model).__name__[:20], cv_p.mean(), cv_o.mean()))
    if cv_p.mean() > best_pfs_r2:
        best_pfs_r2 = cv_p.mean(); best_pfs_model = (model, cv_p)
    if cv_o.mean() > best_os_r2:
        best_os_r2  = cv_o.mean(); best_os_model  = (model, cv_o)

# Fit best models
best_pfs_model[0].fit(X, y_pfs)
best_os_model[0].fit(X, y_os)

print("\nBest PFS R2 (5CV): %.3f" % best_pfs_r2)
print("Best OS  R2 (5CV): %.3f" % best_os_r2)

for name, (mdl, cv) in [('pfs',best_pfs_model),('os',best_os_model)]:
    path = os.path.join(ML_DIR, '%s_gb_model_v2.pkl' % name)
    with open(path,'wb') as f:
        pickle.dump({'model':mdl,'features':feat_names,
                     'r2_5cv':round(float(cv.mean()),3),
                     'r2_std':round(float(cv.std()),3)}, f)
    print("Saved: %s" % path)

metrics = {
    'pfs_r2_5cv': round(float(best_pfs_r2),3),
    'os_r2_5cv':  round(float(best_os_r2),3),
    'n_features': len(feat_names), 'n_samples': len(X),
    'features': feat_names, 'model': 'GradientBoostingRegressor-best',
    'note': 'v2: added ctDNA, PK, toxicity, cytokine features'
}
with open(os.path.join(ML_DIR,'survival_model_metrics_v2.json'),'w') as f:
    json.dump(metrics, f, indent=2)
print("Done. ALL OS/PFS MODELS SAVED.")
