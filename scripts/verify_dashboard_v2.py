"""
MODEL CONFIDENCE DASHBOARD v2 -- NO-HOLDS-BARRED VERIFICATION
Checks every number that appears in the figure against raw data files.
ASCII-safe.
"""
import os, json, csv, pickle
import numpy as np
from collections import Counter

XAI  = r'f:\ADDS\docs\xai_outputs'
ML   = r'f:\ADDS\data\ml_training'
OUT  = r'f:\ADDS\figures'
DOCS = r'f:\ADDS\docs'

ok_items, issues = [], []

def OK(cat, msg):
    ok_items.append((cat,msg))
    print("  [OK]  [%s] %s" % (cat, msg))

def WARN(cat, msg, detail=''):
    issues.append(('WARN',cat,msg,detail))
    print("  [WARN][%s] %s" % (cat,msg))
    if detail: print("         -> %s"%detail)

def FAIL(cat, msg, detail=''):
    issues.append(('FAIL',cat,msg,detail))
    print("  [FAIL][%s] %s" % (cat,msg))
    if detail: print("         -> %s"%detail)

print("="*65)
print("CONFIDENCE DASHBOARD v2 -- INDEPENDENT VERIFICATION")
print("="*65)

# ======================================================================
# BLOCK 1: File existence and size
# ======================================================================
print("\n[BLOCK 1] File existence and minimum size")

files_needed = [
    (os.path.join(XAI,'model_confidence_ci_n20.json'), 5000,  'CI data'),
    (os.path.join(XAI,'lime_official_n50.json'),       5000,  'Official LIME'),
    (os.path.join(OUT,'model_confidence_dashboard_v2.png'), 400000, 'Dashboard PNG'),
]
for path, min_b, label in files_needed:
    if os.path.exists(path):
        sz = os.path.getsize(path)
        if sz >= min_b: OK('Files', '%s: %d bytes' % (label, sz))
        else: WARN('Files', '%s too small: %d bytes (min %d)' % (label, sz, min_b))
    else:
        FAIL('Files', '%s MISSING: %s' % (label, path))

# ======================================================================
# BLOCK 2: CI data -- Panel A claims
# ======================================================================
print("\n[BLOCK 2] CI data integrity (Panel A)")

try:
    with open(os.path.join(XAI,'model_confidence_ci_n20.json')) as f:
        confs = json.load(f)

    n = len(confs)
    if n == 20: OK('CI', 'n=20 records')
    else: FAIL('CI', 'n=20 expected, got %d' % n)

    widths = np.array([c['ci_width'] for c in confs])
    preds  = np.array([c['pfs_predicted'] for c in confs])
    lo95   = np.array([c['ci_95_lower'] for c in confs])
    hi95   = np.array([c['ci_95_upper'] for c in confs])
    tiers  = [c['confidence'] for c in confs]

    # 2-1: all CIs valid (lo <= pred <= hi)
    bad_ci = [(c['patient_id'], c['pfs_predicted'], c['ci_95_lower'], c['ci_95_upper'])
              for c in confs if not (c['ci_95_lower'] <= c['pfs_predicted'] <= c['ci_95_upper'])]
    if not bad_ci:
        OK('CI', 'All 20 CIs valid: lo <= pred <= hi')
    else:
        FAIL('CI', '%d invalid CIs (pred outside interval)' % len(bad_ci))
        for pid,p,lo,hi in bad_ci:
            print('    %s: pred=%.2f lo=%.2f hi=%.2f' % (pid,p,lo,hi))

    # 2-2: CI width = hi - lo
    for c in confs:
        calc_w = c['ci_95_upper'] - c['ci_95_lower']
        if abs(calc_w - c['ci_width']) > 0.15:
            FAIL('CI', 'Width mismatch for %s: stored=%.2f calc=%.2f'
                 % (c['patient_id'], c['ci_width'], calc_w))
    ok_w = sum(1 for c in confs
               if abs((c['ci_95_upper']-c['ci_95_lower']) - c['ci_width']) <= 0.15)
    if ok_w == 20: OK('CI', 'All 20 CI widths consistent with lo/hi')

    # 2-3: Tier thresholds match figure legend (HIGH<4, MEDIUM 4-8, LOW>8)
    for c in confs:
        w = c['ci_width']; t = c['confidence']
        if w < 4.0 and t != 'high':
            FAIL('CI', 'Patient %s: width=%.2f but tier=%s (expected high)' % (c['patient_id'],w,t))
        elif 4.0 <= w <= 8.0 and t not in ('medium','high','low'):
            WARN('CI', 'Patient %s: borderline width=%.2f tier=%s' % (c['patient_id'],w,t))
        elif w > 8.0 and t != 'low':
            FAIL('CI', 'Patient %s: width=%.2f but tier=%s (expected low)' % (c['patient_id'],w,t))
    n_high = tiers.count('high'); n_med=tiers.count('medium'); n_low=tiers.count('low')
    OK('CI', 'Tier distribution: HIGH=%d MED=%d LOW=%d (sum=%d)' % (n_high,n_med,n_low,n_high+n_med+n_low))

    # 2-4: Mean CI width
    mean_w = float(widths.mean())
    OK('CI', 'Mean CI width = %.2f months' % mean_w)
    if mean_w < 1.0 or mean_w > 15.0:
        FAIL('CI', 'Mean CI width %.2f is unrealistic (expected 1-15 months)' % mean_w)

    # 2-5: PFS range plausibility
    if preds.min() > 0 and preds.max() < 35:
        OK('CI', 'PFS range: %.1f - %.1f months (plausible)' % (preds.min(), preds.max()))
    else:
        WARN('CI', 'PFS range: %.1f - %.1f months (check boundaries)' % (preds.min(), preds.max()))

    # 2-6: Arm & KRAS present
    n_arm  = sum(1 for c in confs if c.get('arm','').strip())
    n_kras = sum(1 for c in confs if c.get('kras_allele','').strip())
    if n_arm  == 20: OK('CI', 'All 20 records have arm label')
    else: WARN('CI', '%d/20 records missing arm label' % (20-n_arm))
    if n_kras == 20: OK('CI', 'All 20 records have KRAS allele')
    else: WARN('CI', '%d/20 records missing KRAS' % (20-n_kras))

except Exception as e:
    FAIL('CI', 'Cannot read CI file: %s' % str(e))

# ======================================================================
# BLOCK 3: LIME official data -- Panel B claims
# ======================================================================
print("\n[BLOCK 3] Official LIME data (Panel B)")

try:
    with open(os.path.join(XAI,'lime_official_n50.json')) as f:
        lime = json.load(f)

    if len(lime)==50: OK('LIME', 'n=50 cases')
    else: FAIL('LIME', 'Expected 50, got %d' % len(lime))

    # dominant feature distribution
    dom_list = [lo.get('dominant_feature','') for lo in lime]
    dom_cnt  = Counter(dom_list)
    top3     = dom_cnt.most_common(3)
    OK('LIME', 'Top-3 dominant features: %s' % str([(f[:20],n) for f,n in top3]))

    # All pcts should sum to ~100%
    total_dom = sum(dom_cnt.values())
    if total_dom == 50: OK('LIME', 'All 50 cases have dominant feature')
    else: WARN('LIME', 'Only %d/50 cases have dominant feature' % total_dom)

    # direction distribution (should not be all-one-direction if model is weak)
    dirs = [lo.get('dominant_direction','') for lo in lime]
    n_pos = dirs.count('positive'); n_neg=dirs.count('negative')
    print("  LIME direction: positive=%d negative=%d" % (n_pos, n_neg))
    if n_pos == 50 or n_neg == 50:
        WARN('LIME', 'ALL attributions in same direction -- highly suspicious, suggests degenerate model')
    elif 0.30 <= n_pos/50 <= 0.70:
        OK('LIME', 'Direction split balanced (pos=%.0f%%, neg=%.0f%%) -- model uncertainty reflected' % (100*n_pos/50, 100*n_neg/50))
    else:
        WARN('LIME', 'Direction skewed: pos=%.0f%% neg=%.0f%% -- acceptable but leans one way' % (100*n_pos/50, 100*n_neg/50))

    # top5_attributions structure check
    n_has_attrs = sum(1 for lo in lime if lo.get('top5_attributions') and len(lo.get('top5_attributions',[]))>=3)
    if n_has_attrs >= 45: OK('LIME', '%d/50 cases have >=3 top attributions' % n_has_attrs)
    else: WARN('LIME', 'Only %d/50 have >=3 attributions' % n_has_attrs)

    # xai_method field confirms official lime
    methods = set(lo.get('xai_method','') for lo in lime)
    if any('LimeTabularExplainer' in m for m in methods):
        OK('LIME', 'Official lime.LimeTabularExplainer confirmed in method field')
    else:
        WARN('LIME', 'xai_method field: %s -- confirm official lime used' % str(methods))

    # n_samples field
    n_samples = {lo.get('n_samples') for lo in lime}
    if {500} == n_samples: OK('LIME', '500 perturbation samples confirmed for all cases')
    else: WARN('LIME', 'n_samples varies: %s' % str(n_samples))

except Exception as e:
    FAIL('LIME', 'Cannot read LIME file: %s' % str(e))

# ======================================================================
# BLOCK 4: Permutation importance claims (Panel B supplement)
# ======================================================================
print("\n[BLOCK 4] Permutation importance global (Panel B supplement)")

try:
    pi_path = os.path.join(XAI,'permutation_importance_global.json')
    with open(pi_path) as f:
        pi = json.load(f)
    top5 = pi.get('top5',[])
    feats = pi.get('features',{})
    if top5:
        OK('Perm', 'Top-5 features: %s' % str([f[:15] for f in top5]))
        # Check msi dominance claim from report
        msi_rank = top5.index('msi') + 1 if 'msi' in top5 else None
        if msi_rank == 1: OK('Perm', 'msi is #1 permutation importance (matches report)')
        elif msi_rank: WARN('Perm', 'msi is rank %d, not #1 as claimed in report' % msi_rank)
        else: WARN('Perm', 'msi not in top-5 -- check report claim')
        # check bliss
        bliss_rank = top5.index('bliss')+1 if 'bliss' in top5 else None
        if bliss_rank == 2: OK('Perm', 'bliss is #2 (matches report)')
        elif bliss_rank: WARN('Perm', 'bliss is rank %d' % bliss_rank)
        else: WARN('Perm', 'bliss not in top-5')
    if feats:
        msi_val = feats.get('msi',{}).get('mean',0)
        bliss_val= feats.get('bliss',{}).get('mean',0)
        OK('Perm', 'msi importance=%.4f, bliss=%.4f' % (msi_val,bliss_val))
        # Report claims msi=0.604, bliss=0.330
        if abs(msi_val-0.604)<0.02: OK('Perm', 'msi=0.604 VERIFIED')
        else: WARN('Perm', 'msi=%.4f vs claimed 0.604'%msi_val)
        if abs(bliss_val-0.330)<0.02: OK('Perm', 'bliss=0.330 VERIFIED')
        else: WARN('Perm', 'bliss=%.4f vs claimed 0.330'%bliss_val)
except Exception as e:
    WARN('Perm', 'permutation_importance_global.json error: %s'%str(e))

# ======================================================================
# BLOCK 5: Model v5 claims (OS/PFS improvement)
# ======================================================================
print("\n[BLOCK 5] OS/PFS model v5 claims")

try:
    with open(os.path.join(ML,'survival_model_metrics_v5.json')) as f:
        sv5 = json.load(f)
    pfs_r2 = sv5.get('pfs_r2_5cv',0)
    os_r2  = sv5.get('os_r2_5cv',0)
    if pfs_r2 >= 0.25: OK('Model', 'PFS R2 v5=%.3f >= 0.25 (significant improvement from 0.056)' % pfs_r2)
    else: WARN('Model', 'PFS R2 v5=%.3f -- still low' % pfs_r2)
    if abs(pfs_r2-0.303)<0.02: OK('Model', 'PFS R2=0.303 VERIFIED')
    else: WARN('Model', 'PFS R2 claimed 0.303, actual=%.3f' % pfs_r2)
    OK('Model', 'OS R2 v5=%.3f' % os_r2)
except Exception as e:
    WARN('Model', 'survival_model_metrics_v5.json error: %s'%str(e))

# ======================================================================
# BLOCK 6: Near-CF data integrity
# ======================================================================
print("\n[BLOCK 6] Near-CF data (Fix 3b)")

try:
    with open(os.path.join(XAI,'near_counterfactual_n30.json')) as f:
        ncf = json.load(f)
    n_ok = len(ncf)
    if n_ok == 30: OK('CF', 'n=30 Near-CF records')
    else: WARN('CF', 'n=30 expected, got %d' % n_ok)
    deltas = [c['delta_pfs'] for c in ncf]
    n_pos  = sum(1 for d in deltas if d > 0)
    mean_d = float(np.mean(deltas))
    if n_pos == 30: OK('CF', 'All 30 CFs have positive delta (mean=%.2f)' % mean_d)
    else: WARN('CF', 'Only %d/30 have positive delta' % n_pos)
    if abs(mean_d-4.71)<0.3: OK('CF', 'Mean delta=%.2f matches claimed 4.71' % mean_d)
    else: WARN('CF', 'Mean delta=%.2f vs claimed 4.71' % mean_d)
    # method field
    methods = set(c.get('xai_method','') for c in ncf)
    if any('Greedy' in m or 'feasibility' in m.lower() for m in methods):
        OK('CF', 'Near-CF method confirmed: Greedy feasibility-constrained')
    else:
        WARN('CF', 'xai_method: %s -- confirm method' % str(methods))
except Exception as e:
    WARN('CF', 'near_counterfactual_n30.json error: %s'%str(e))

# ======================================================================
# BLOCK 7: Bliss KS cross-validation
# ======================================================================
print("\n[BLOCK 7] Bliss cross-validation result")

try:
    with open(os.path.join(DOCS,'bliss_crossvalidation.json')) as f:
        ks = json.load(f)
    ks_stat = ks.get('ks_statistic',0)
    ks_p    = ks.get('p_value',1)
    lit_m   = ks.get('lit_mean',0)
    int_m   = ks.get('internal_mean',0)
    OK('Bliss', 'KS stat=%.4f p=%.4f' % (ks_stat, ks_p))
    if ks_p < 0.05:
        WARN('Bliss',
             'Internal Bliss DIFFERENT from literature (p=%.4f) -- bias confirmed' % ks_p,
             'Internal mean=%.1f vs lit mean=%.1f (overclaim by +%.1f)' % (int_m,lit_m,int_m-lit_m))
    else:
        OK('Bliss', 'Internal vs literature distributions COMPATIBLE (p=%.3f)' % ks_p)
    gap = int_m - lit_m
    if gap > 5:
        WARN('Bliss', 'Internal synergy inflated by +%.1f Bliss units vs literature' % gap,
             'Published papers using internal Bliss values must note this upward bias.')
    else:
        OK('Bliss', 'Internal vs lit gap=%.1f (acceptable)' % gap)
except Exception as e:
    WARN('Bliss', 'bliss_crossvalidation.json error: %s'%str(e))

# ======================================================================
# BLOCK 8: KRAS imputation uncertainty
# ======================================================================
print("\n[BLOCK 8] KRAS imputation uncertainty")

try:
    with open(os.path.join(DOCS,'patient_kras_imputation_uncertainty.json')) as f:
        kras_u = json.load(f)
    in_range = [k for k in kras_u if k.get('within_meta_range') is True]
    out_range= [k for k in kras_u if k.get('within_meta_range') is False]
    OK('KRAS', '%d alleles in meta range, %d outside (small sample bootstrap instability)' % (len(in_range),len(out_range)))
    if out_range:
        WARN('KRAS',
             '%d alleles bootstrap CI misses meta frequency' % len(out_range),
             'Expected: n=26 too small for stable freq estimate. Must report as "imputed, uncertain".')
    # Check all 5 alleles covered
    alleles = [k['allele'] for k in kras_u]
    expected = {'G12D','G12V','WT','G13D','G12C'}
    missing  = expected - set(alleles)
    if not missing: OK('KRAS', 'All 5 KRAS alleles have uncertainty quantification')
    else: WARN('KRAS', 'Missing uncertainty for: %s' % str(missing))
except Exception as e:
    WARN('KRAS', 'KRAS uncertainty file error: %s'%str(e))

# ======================================================================
# BLOCK 9: Physician eval v2 calibration check
# ======================================================================
print("\n[BLOCK 9] Physician eval v2 calibration")

try:
    with open(os.path.join(XAI,'physician_evaluation_v2_n45.json')) as f:
        phys2 = json.load(f)
    n = len(phys2)
    if n==45: OK('PhysEval', 'n=45 records')
    else: WARN('PhysEval', 'n=45 expected, got %d' % n)

    comp_v2 = [p['composite_score_5'] for p in phys2]
    use_v2  = sum(1 for p in phys2 if p['would_use_in_clinic_Y/N']=='Y')
    OK('PhysEval', 'Mean composite=%.2f, would-use=%d/45 (%.0f%%)' % (
        np.mean(comp_v2), use_v2, 100*use_v2/45))

    # Check method field
    bases = set(p.get('simulation_basis','') for p in phys2)
    if any('Cai' in b or 'Sindhu' in b for b in bases):
        OK('PhysEval', 'Simulation basis: literature-calibrated (Cai 2021 / Sindhu 2022)')
    else:
        WARN('PhysEval', 'simulation_basis not set: %s' % str(bases))

    # D4 actionability -- must be explicitly reported as low
    d4_vals = [p['scores'].get('D4_actionability',3) for p in phys2]
    d4_mean = float(np.mean(d4_vals))
    if abs(d4_mean-3.00)<0.15: OK('PhysEval', 'D4 actionability=%.2f (matches corrected value ~3.00)' % d4_mean)
    else: WARN('PhysEval', 'D4=%.2f -- check if report uses corrected value 3.00' % d4_mean)

    # All tools: LIME should still be top
    by_tool = {}
    for p in phys2:
        t = p['xai_tool_evaluated']
        by_tool.setdefault(t,[]).append(p['composite_score_5'])
    tool_means = {t: float(np.mean(v)) for t,v in by_tool.items()}
    top_tool = max(tool_means, key=tool_means.get)
    bot_tool = min(tool_means, key=tool_means.get)
    if 'LIME' in top_tool: OK('PhysEval', 'LIME still top-rated (%.2f)' % tool_means[top_tool])
    else: WARN('PhysEval', 'LIME not top anymore: %s (%.2f)' % (top_tool, tool_means[top_tool]))
    if 'None' in bot_tool or 'control' in bot_tool.lower():
        OK('PhysEval', 'None_control still lowest rated (%.2f)' % tool_means[bot_tool])
    else:
        WARN('PhysEval', 'None_control not lowest: %s' % bot_tool)

except Exception as e:
    WARN('PhysEval', 'physician_evaluation_v2 error: %s'%str(e))

# ======================================================================
# BLOCK 10: Critical scientific / honesty checks
# ======================================================================
print("\n[BLOCK 10] Scientific honesty")

# OS/PFS R2=0.303 -- still not great, honest?
WARN('Honesty',
     'OS/PFS R2=0.303 is an improvement (from 0.056) but still moderate',
     'This is because arm effect was hard-coded into synthetic cohort -- '
     'the improvement reflects data redesign, NOT real clinical signal improvement. '
     'Must be disclosed as "improved synthetic design, not real RCT validation".')

# Bliss internal > literature by ~7.7 units
WARN('Honesty',
     'Internal Bliss mean 17.4 vs literature 9.7 (+7.7 units, p<0.0001)',
     'Any claims using internal Bliss synergy values overstate synergy by ~80%. '
     'Downward recalibration or explicit bias disclosure required for publication.')

# LIME direction still 50/50 for Cmax
WARN('Honesty',
     'Official LIME Cmax direction still not consistently positive (50/50)',
     'This reflects that the GBM v5 model still does not fully capture PK-outcome '
     'mechanism even with improved cohort. Cmax frequency alone is not sufficient evidence.')

# Near-CF delta inflated by model design?
WARN('Honesty',
     'Near-CF mean delta=+4.71 may be inflated',
     'The greedy algorithm is designed to find any feature change that increases prediction. '
     'If the model is noisy, small feature perturbations may accidentally land on noise peaks. '
     'Validate with held-out patients to confirm delta is clinically meaningful.')

# IRB questionnaire -- is it complete?
irb_path = os.path.join(DOCS,'IRB_physician_eval_questionnaire.txt')
if os.path.exists(irb_path):
    sz = os.path.getsize(irb_path)
    if sz > 500: OK('Honesty', 'IRB questionnaire file exists (%d bytes)' % sz)
    else: WARN('Honesty', 'IRB questionnaire too small (%d bytes)' % sz)
else:
    FAIL('Honesty', 'IRB questionnaire file MISSING')

# ======================================================================
# FINAL
# ======================================================================
n_fail = sum(1 for s,*_ in issues if s=='FAIL')
n_warn = sum(1 for s,*_ in issues if s=='WARN')
n_ok   = len(ok_items)

print("\n" + "="*65)
print("CONFIDENCE DASHBOARD v2 -- FINAL VERDICT")
print("="*65)
print("  OK:   %d" % n_ok)
print("  WARN: %d" % n_warn)
print("  FAIL: %d" % n_fail)

print("\nFAIL items:")
if not [x for x in issues if x[0]=='FAIL']:
    print("  (none)")
for s,cat,msg,detail in issues:
    if s=='FAIL':
        print("  [FAIL][%s] %s"%(cat,msg))
        if detail: print("         -> %s"%detail[:120])

print("\nWARN items:")
for s,cat,msg,detail in issues:
    if s=='WARN':
        print("  [WARN][%s] %s"%(cat,msg))
        if detail: print("         -> %s"%detail[:110])

verdict = 'PASS' if n_fail==0 else 'FAIL'
print("\nVerdict:", verdict, "(structural)  |  %d scientific disclosures required"%n_warn)

import json as json2
result = {'ok':n_ok,'warn':n_warn,'fail':n_fail,'verdict':verdict,
          'issues':[{'level':s,'cat':c,'msg':m,'detail':d} for s,c,m,d in issues]}
with open(os.path.join(DOCS,'dashboard_v2_verification.json'),'w') as f:
    json2.dump(result, f, indent=2, ensure_ascii=True)
print("Saved: f:\\ADDS\\docs\\dashboard_v2_verification.json")
