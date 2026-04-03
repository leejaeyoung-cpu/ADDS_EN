"""
OS/PFS survival model -- standalone fix
Uses cohort v3 which has all fields
"""
import os, csv, json, pickle
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import cross_val_score

rng = np.random.default_rng(2026)

DATA   = r'f:\ADDS\data'
ML_DIR = os.path.join(DATA, 'ml_training')
DOCS   = r'f:\ADDS\docs'

# Load enriched cohort v3
cohort_path = os.path.join(DATA, 'pritamab_synthetic_cohort_enriched_v3.csv')
with open(cohort_path, encoding='utf-8') as f:
    reader = csv.DictReader(f)
    cohort = list(reader)

print("Cohort v3: %d rows, fields: %s" % (len(cohort), str(list(cohort[0].keys()))[:120]))

arm_catalog  = list(set(r.get('arm','FOLFOX') for r in cohort))
kras_catalog = list(set(r.get('kras_allele','G12D') for r in cohort))

arm_map  = {a:i for i,a in enumerate(sorted(arm_catalog))}
kras_map = {k:i for i,k in enumerate(sorted(kras_catalog))}
prpc_map = {'high':3,'medium-high':2,'medium':1,'medium-low':0,'low':0}

X_all = []; y_pfs_all = []; y_os_all = []
for row in cohort:
    try:
        arm_e  = arm_map.get(row.get('arm','FOLFOX'), 0)
        kras_e = kras_map.get(row.get('kras_allele','G12D'), 0)
        prpc_e = prpc_map.get(str(row.get('prpc_expression_level','low')).lower(), 0)
        msi_e  = 1 if str(row.get('msi_status','MSS')).upper()=='MSI-H' else 0
        bliss  = float(row.get('bliss_score_predicted','15') or 15)
        orr    = float(row.get('orr','0.4') or 0.4)
        cea    = float(row.get('cea_baseline','10') or 10)
        conf   = float(row.get('dl_confidence','0.7') or 0.7)
        pfs_v  = float(row.get('dl_pfs_months','12') or 12)
        os_v   = float(row.get('dl_os_months','18') or 18)
        X_all.append([arm_e, kras_e, prpc_e, msi_e, bliss, orr, cea, conf])
        y_pfs_all.append(pfs_v)
        y_os_all.append(os_v)
    except Exception as e:
        pass

print("Training samples: %d" % len(X_all))
X = np.array(X_all); y_pfs = np.array(y_pfs_all); y_os = np.array(y_os_all)

gb_pfs = GradientBoostingRegressor(n_estimators=200, max_depth=5, learning_rate=0.05, random_state=2026)
gb_os  = GradientBoostingRegressor(n_estimators=200, max_depth=5, learning_rate=0.05, random_state=2026)

cv_pfs = cross_val_score(gb_pfs, X, y_pfs, cv=5, scoring='r2')
cv_os  = cross_val_score(gb_os,  X, y_os,  cv=5, scoring='r2')
gb_pfs.fit(X, y_pfs); gb_os.fit(X, y_os)

print("PFS model R2 (5CV): %.3f +/- %.3f" % (cv_pfs.mean(), cv_pfs.std()))
print("OS  model R2 (5CV): %.3f +/- %.3f" % (cv_os.mean(),  cv_os.std()))

feat_names = ['arm','kras','prpc','msi','bliss','orr','cea','dl_confidence']
for name, model, cv in [('pfs', gb_pfs, cv_pfs), ('os', gb_os, cv_os)]:
    path = os.path.join(ML_DIR, '%s_gb_model_v1.pkl' % name)
    with open(path, 'wb') as f:
        pickle.dump({'model': model, 'features': feat_names,
                     'r2_5cv': round(float(cv.mean()),3),
                     'r2_std': round(float(cv.std()),3)}, f)
    print("Saved: %s" % path)

metrics = {'pfs_r2_5cv': round(float(cv_pfs.mean()),3),
           'pfs_r2_std': round(float(cv_pfs.std()),3),
           'os_r2_5cv':  round(float(cv_os.mean()),3),
           'os_r2_std':  round(float(cv_os.std()),3),
           'n_train': len(X), 'model': 'GradientBoostingRegressor'}
with open(os.path.join(ML_DIR, 'survival_model_metrics_v1.json'),'w') as f:
    json.dump(metrics, f, indent=2)
print("Metrics saved.")
print("ALL DONE -- OS/PFS models ready")
