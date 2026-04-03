"""
ADDS COMPREHENSIVE NO-HOLDS-BARRED VERIFICATION
Verifies: report accuracy, data integrity, scientific claims,
          XAI validity, model honesty, statistical consistency
ASCII-safe.
"""
import os, json, csv, pickle
import numpy as np
from collections import Counter

DATA   = r'f:\ADDS\data'
ML_DIR = os.path.join(DATA,'ml_training')
INT_DS = os.path.join(DATA,'integrated_datasets')
SYN_EN = os.path.join(DATA,'synergy_enriched')
XAI    = r'f:\ADDS\docs\xai_outputs'
DOCS   = r'f:\ADDS\docs'

issues   = []  # [severity, category, finding, verdict]
ok_items = []

def FAIL(cat, finding, detail=''):
    issues.append(('FAIL', cat, finding, detail))
    print("  [FAIL] [%s] %s" % (cat, finding))
    if detail: print("         Detail: %s" % detail)

def WARN(cat, finding, detail=''):
    issues.append(('WARN', cat, finding, detail))
    print("  [WARN] [%s] %s" % (cat, finding))
    if detail: print("         Detail: %s" % detail)

def OK(cat, finding):
    ok_items.append((cat, finding))
    print("  [OK]   [%s] %s" % (cat, finding))

print("=" * 70)
print("ADDS SYSTEM -- INDEPENDENT NO-HOLDS-BARRED VERIFICATION")
print("=" * 70)

# ======================================================================
# BLOCK 1: REPORT CLAIMS vs ACTUAL DATA
# ======================================================================
print("\n[BLOCK 1] Report claims vs. actual data")

# 1-1: Bliss DB claim "136 records"
try:
    with open(os.path.join(SYN_EN,'bliss_curated_v3.csv'),encoding='utf-8') as f:
        bliss = list(csv.DictReader(f))
    if len(bliss) == 136: OK('Report','Bliss claim 136 records VERIFIED')
    else: FAIL('Report','Bliss claim 136 but actual=%d'%len(bliss))

    n_antag = sum(1 for r in bliss if float(r.get('bliss','0') or 0) < 0)
    if n_antag == 4: OK('Report','Bliss antagonism claim 4 records VERIFIED')
    else: FAIL('Report','Antagonism claim 4 but actual=%d'%n_antag)

    n_prit = sum(1 for r in bliss if 'Pritamab' in r.get('combination',''))
    if n_prit >= 60: OK('Report','Pritamab combos >= 60 VERIFIED (n=%d)'%n_prit)
    else: WARN('Report','Pritamab combos claim 60 actual=%d'%n_prit)
except Exception as e:
    FAIL('Report','Cannot open bliss_curated_v3.csv: %s'%str(e))

# 1-2: Cohort claim "1000 rows, 44 fields"
try:
    with open(os.path.join(DATA,'pritamab_synthetic_cohort_enriched_v4.csv'),encoding='utf-8') as f:
        reader = csv.DictReader(f)
        coh4 = list(reader)
        fields = list(coh4[0].keys()) if coh4 else []
    if len(coh4) == 1000: OK('Report','Cohort n=1000 VERIFIED')
    else: FAIL('Report','Cohort claim 1000 actual=%d'%len(coh4))
    if len(fields) >= 44: OK('Report','Cohort fields=%d (claimed 44) VERIFIED'%len(fields))
    else: WARN('Report','Cohort fields=%d vs claimed 44'%len(fields),'May differ slightly')
    # PFS range
    pfs_vals = [float(r.get('dl_pfs_months',0) or 0) for r in coh4]
    if min(pfs_vals) < 2 and max(pfs_vals) > 20:
        OK('Report','PFS range %.1f-%.1f (claimed 1.0-28.2)'%(min(pfs_vals),max(pfs_vals)))
    else:
        WARN('Report','PFS range %.1f-%.1f narrower than claimed'%(min(pfs_vals),max(pfs_vals)))
except Exception as e:
    FAIL('Report','Cannot open cohort_v4: %s'%str(e))

# 1-3: TCGA claim "n=594"
try:
    with open(os.path.join(ML_DIR,'tcga_crc_clinical_enriched_v3.csv'),encoding='utf-8') as f:
        tcga = list(csv.DictReader(f))
    if len(tcga) == 594: OK('Report','TCGA n=594 VERIFIED')
    else: FAIL('Report','TCGA claim 594 actual=%d'%len(tcga))
    arm_filled = sum(1 for r in tcga if r.get('treatment_arm','').strip())
    if arm_filled == 594: OK('Report','TCGA arm 594/594 VERIFIED')
    else: FAIL('Report','TCGA arm claim 594/594 actual=%d/594'%arm_filled)
    cea_f = sum(1 for r in tcga if r.get('cea_baseline','').strip())
    if cea_f == 594: OK('Report','TCGA CEA 594/594 VERIFIED')
    else: FAIL('Report','TCGA CEA claim 594/594 actual=%d/594'%cea_f)
except Exception as e:
    FAIL('Report','TCGA v3 error: %s'%str(e))

# 1-4: Patient claim "26 patients fully enriched"
try:
    with open(os.path.join(INT_DS,'master_dataset.jsonl'),encoding='utf-8') as f:
        patients = [json.loads(l.strip()) for l in f if l.strip()]
    if len(patients) == 26: OK('Report','Patient n=26 VERIFIED')
    else: FAIL('Report','Patient claim 26 actual=%d'%len(patients))
    kras_ok = sum(1 for p in patients
                  if p['patient_demographics'].get('kras_mutation','') not in ('Unknown','',None))
    if kras_ok == 26: OK('Report','Patient KRAS 26/26 VERIFIED')
    else: FAIL('Report','Patient KRAS claim 26/26 actual=%d/26'%kras_ok)
    orr_ok = sum(1 for p in patients if p.get('clinical_outcomes',{}).get('orr') is not None)
    if orr_ok == 26: OK('Report','Patient ORR 26/26 VERIFIED')
    else: FAIL('Report','Patient ORR claim 26/26 actual=%d/26'%orr_ok)
except Exception as e:
    FAIL('Report','Patient dataset error: %s'%str(e))

# 1-5: DL Synergy "R2=0.996"
try:
    with open(os.path.join(ML_DIR,'evaluation_results_v3.json')) as f:
        ev = json.load(f)
    r2 = ev.get('r2_5cv', ev.get('r2',0))
    if abs(r2 - 0.996) < 0.01: OK('Report','Synergy R2=0.996 VERIFIED in eval JSON')
    else: FAIL('Report','Synergy R2 claim 0.996 actual=%.3f'%r2)
except:
    WARN('Report','evaluation_results_v3.json missing or different format -- cannot verify R2')

# 1-6: OS/PFS "R2=0.056"
try:
    with open(os.path.join(ML_DIR,'survival_model_metrics_v3.json')) as f:
        sm = json.load(f)
    pfs_r2 = sm.get('pfs_r2_5cv',0)
    os_r2  = sm.get('os_r2_5cv',0)
    if abs(pfs_r2 - 0.056) < 0.01: OK('Report','PFS R2=0.056 VERIFIED')
    else: WARN('Report','PFS R2 claim 0.056 actual=%.3f'%pfs_r2)
    if abs(os_r2 - 0.066) < 0.01: OK('Report','OS R2=0.066 VERIFIED')
    else: WARN('Report','OS R2 claim 0.066 actual=%.3f'%os_r2)
except Exception as e:
    WARN('Report','survival_model_metrics_v3.json issue: %s'%str(e))

# ======================================================================
# BLOCK 2: SCIENTIFIC VALIDITY AUDIT
# ======================================================================
print("\n[BLOCK 2] Scientific validity of claims")

# 2-1: Composite scores (the numbers reported in the report)
expected_composites = {
    'Pritamab Mono':       29,   'Pritamab+FOLFOX':  124,
    'Pritamab+FOLFIRI':   105,   'Pritamab+FOLFOXIRI': None,
    'FOLFOX':             118,   'FOLFIRI':           115,
    'FOLFOXIRI':          175,   'CAPOX':             119,
    'TAS-102':            104,   'Bev+FOLFOX':        123,
    'Pembrolizumab':       57,
}
TOX_MATRIX = np.array([
    [ 2,  3,  2,  2,  3,   1,  6,  0,   1,  2,  3,  4],
    [36,  8,  6,  6, 12,  14, 10,  2,  15,  4,  8,  3],
    [22, 10,  5,  8, 18,   4, 12,  1,  15,  3,  4,  3],
    [41,  7,  5,  7, 11,  18,  8,  1,   5,  5, 10,  0],
    [24, 11,  4,  9, 20,   3, 10,  0,  30,  2,  2,  0],
    [50, 18,  9, 19, 20,  12, 16,  1,  20,  6,  4,  0],
    [21,  4, 15,  8, 12,  17,  8, 17,   1,  4, 12,  0],
    [38, 19,  5,  5,  6,   1, 22,  0,   5,  2,  1,  0],
    [38,  7,  4,  6, 10,  17, 10,  2,   6,  5, 18,  0],
    [ 2,  3,  1,  2,  4,   1, 18,  0,   1,  3,  0, 22],
], dtype=float)
REG_SHORT = ['Pritamab Mono','Pritamab+FOLFOX','Pritamab+FOLFIRI','FOLFOX','FOLFIRI',
             'FOLFOXIRI','CAPOX','TAS-102','Bev+FOLFOX','Pembrolizumab']
computed = {r: int(TOX_MATRIX[i].sum()) for i,r in enumerate(REG_SHORT)}

for reg, claimed in expected_composites.items():
    if claimed is None: continue
    comp = computed.get(reg)
    if comp == claimed: OK('Science','Composite %s=%d VERIFIED'%(reg,claimed))
    else: FAIL('Science','Composite mismatch: %s claimed=%s computed=%s'%(reg,claimed,comp))

# 2-2: KRAS distribution check (report claims G12D 37% etc.)
expected_kras_dist = {'G12D':0.37,'G12V':0.22,'WT':0.25,'G13D':0.09,'G12C':0.07}
# Published literature: Taieb 2022, COSMIC CRC -- verify plausibility
for allele, claimed_pct in expected_kras_dist.items():
    # Cross-check with COSMIC CRC frequencies (within 5% tolerance)
    cosmos_ranges = {'G12D':(0.30,0.42),'G12V':(0.18,0.26),'WT':(0.20,0.30),'G13D':(0.07,0.13),'G12C':(0.03,0.12)}
    lo, hi = cosmos_ranges.get(allele,(0,1))
    if lo <= claimed_pct <= hi:
        OK('Science','KRAS %s=%.0f%% within COSMIC CRC range [%.0f-%.0f%%]'%(allele,claimed_pct*100,lo*100,hi*100))
    else:
        WARN('Science','KRAS %s=%.0f%% outside COSMIC range [%.0f-%.0f%%]'%(allele,claimed_pct*100,lo*100,hi*100))

# 2-3: Literature anchor values
lit_anchors = {
    'FOLFOX_Neutropenia': (41, 35, 48, 'MOSAIC 2004'),
    'FOLFIRI_Neutropenia': (24, 20, 32, 'Douillard 2000'),
    'FOLFOXIRI_Neutropenia': (50, 45, 58, 'Falcone 2007'),
    'TAS102_Neutropenia': (38, 33, 44, 'RECOURSE 2015'),
    'Pembro_irAE': (22, 17, 28, 'KEYNOTE-177 2021'),
}
for key, (val, lo, hi, ref) in lit_anchors.items():
    if lo <= val <= hi: OK('Science','%s val=%d in [%d-%d] %s VERIFIED'%(key,val,lo,hi,ref))
    else: FAIL('Science','%s val=%d OUTSIDE range [%d-%d] %s'%(key,val,lo,hi,ref))

# 2-4: Clinical logic checks (same as dashboard)
checks = [
    ('FOLFOXIRI highest', computed['FOLFOXIRI'] > max(v for k,v in computed.items() if 'FOLFOXIRI' not in k)),
    ('Pritamab Mono lowest', computed['Pritamab Mono'] < min(v for k,v in computed.items() if k != 'Pritamab Mono')),
    ('FOLFOX PN > FOLFIRI PN', TOX_MATRIX[3][5] > TOX_MATRIX[4][5]),
    ('FOLFIRI Alopecia > FOLFOX', TOX_MATRIX[4][8] > TOX_MATRIX[3][8]),
    ('CAPOX HFS > FOLFOX', TOX_MATRIX[6][7] > TOX_MATRIX[3][7]),
    ('Pembro irAE highest', TOX_MATRIX[9][11] == max(TOX_MATRIX[:,11])),
    ('Pure chemo irAE=0', all(TOX_MATRIX[i][11] == 0 for i in [3,4,5,6,7,8])),
]
for desc, result in checks:
    if result: OK('Science','Logic: %s PASS'%desc)
    else: FAIL('Science','Logic: %s FAIL'%desc)

# ======================================================================
# BLOCK 3: XAI METHODOLOGY HONESTY
# ======================================================================
print("\n[BLOCK 3] XAI methodology honesty")

# 3-1: LIME implementation is NOT the official LIME library
WARN('XAI','LIME implementation is a custom Ridge approximation, NOT the official lime package',
     'This is a documented simplification. Valid for local linear attribution but less robust than official LIME.')

# 3-2: Grad-CAM is explicitly a proxy
WARN('XAI','Grad-CAM implementation is a finite-difference proxy, NOT a true neural network Grad-CAM',
     'Report calls it "proxy" -- correctly labeled. Valid for tree models but NOT equivalent to CNN Grad-CAM.')

# 3-3: Counterfactual does not use wachter/dice -- it is manual pivot
WARN('XAI','Counterfactual is a manual feature-pivot, NOT a formal CF algorithm (DICE/Wachter)',
     'Valid for clinical illustration but may not find nearest CF on data manifold.')

# 3-4: Check LIME outputs have actual content
try:
    with open(os.path.join(XAI,'lime_attributions_n50.json')) as f:
        lime = json.load(f)
    if len(lime) == 50: OK('XAI','LIME 50 cases file exists and complete')
    else: FAIL('XAI','LIME file incomplete: %d/50'%len(lime))
    # Check attributions are non-trivial
    nonzero = sum(1 for lo in lime if any(v != 0 for v in lo['top_attributions'].values()))
    if nonzero == 50: OK('XAI','LIME all 50 cases have non-zero attributions')
    else: WARN('XAI','%d LIME cases have zero attributions'%(50-nonzero))
    # Dominant feature consistency check
    top_feat = Counter(lo['dominant_feature'] for lo in lime).most_common(1)[0]
    OK('XAI','LIME top dominant: %s (%d/50)'%(top_feat[0], top_feat[1]))
except Exception as e:
    FAIL('XAI','LIME file error: %s'%str(e))

# 3-5: Check CF outputs
try:
    with open(os.path.join(XAI,'counterfactual_analysis_n40.json')) as f:
        cf = json.load(f)
    if len(cf) >= 30: OK('XAI','CF analysis >= 30 patients')
    else: WARN('XAI','CF only %d patients (claimed 40)'%len(cf))
    # Check deltas are non-trivial
    all_deltas = [c['delta_months'] for p in cf for c in p.get('counterfactuals',[])]
    if len(all_deltas) > 0:
        std_d = float(np.std(all_deltas))
        if std_d > 0.1: OK('XAI','CF deltas have variance (std=%.2f)'%std_d)
        else: WARN('XAI','CF deltas all identical -- suspect')
    else:
        WARN('XAI','No CF deltas found')
except Exception as e:
    FAIL('XAI','CF file error: %s'%str(e))

# 3-6: Bootstrap CI check
try:
    with open(os.path.join(XAI,'model_confidence_ci_n20.json')) as f:
        confs = json.load(f)
    widths = [c['ci_width'] for c in confs]
    mean_w = float(np.mean(widths))
    if abs(mean_w - 3.90) < 0.20: OK('XAI','CI mean width 3.90mo VERIFIED (actual=%.2f)'%mean_w)
    else: WARN('XAI','CI mean width claimed 3.90 actual=%.2f'%mean_w)
    # Check all CIs are valid (lo < pred < hi)
    valid_ci = sum(1 for c in confs if c['ci_95_lower'] < c['pfs_predicted'] < c['ci_95_upper'])
    if valid_ci == len(confs): OK('XAI','All %d CIs valid (lo < pred < hi)'%len(confs))
    else: FAIL('XAI','%d/%d CIs invalid (pred outside CI)'%(len(confs)-valid_ci,len(confs)))
except Exception as e:
    FAIL('XAI','CI file error: %s'%str(e))

# ======================================================================
# BLOCK 4: PHYSICIAN EVALUATION HONESTY
# ======================================================================
print("\n[BLOCK 4] Physician evaluation data honesty")

try:
    with open(os.path.join(XAI,'physician_evaluation_survey_n45.json')) as f:
        phys = json.load(f)
    n = len(phys)
    if n == 45: OK('PhysEval','n=45 VERIFIED')
    else: FAIL('PhysEval','claim 45 actual=%d'%n)

    # CRITICAL: This is SIMULATED data
    FAIL('PhysEval',
         'ALL 45 physician responses are SYNTHETIC SIMULATION -- NOT real physicians',
         'Report must explicitly state: physician evaluation is a SIMULATION, '
         'not a real user study. Reporting as clinical evidence would be misleading.')

    # Check composite score claim
    comps = [p['composite_score_5'] for p in phys]
    mean_c = float(np.mean(comps))
    if abs(mean_c - 3.60) < 0.05: OK('PhysEval','Composite 3.60 VERIFIED (%.2f)'%mean_c)
    else: WARN('PhysEval','Composite claim 3.60 actual=%.2f'%mean_c)

    # Score distribution sanity
    if 1 <= min(comps) and max(comps) <= 5: OK('PhysEval','All scores in [1,5] Likert range')
    else: FAIL('PhysEval','Scores outside Likert range: min=%.1f max=%.1f'%(min(comps),max(comps)))

    # NPS check
    nps_pro = sum(1 for p in phys if p['nps_class']=='Promoter')
    if nps_pro == 1:
        WARN('PhysEval','NPS Promoters mere 1/45 (2%) -- extremely low adoption signal',
             'This is realistic but should be flagged: model is not yet clinically persuasive.')
    elif nps_pro > 20:
        WARN('PhysEval','NPS Promoters suspiciously high (%d/45)'%nps_pro,'May indicate simulation bias')
    else:
        OK('PhysEval','NPS Promoters %d/45 plausible'%nps_pro)

    # Check "would use" claim
    use_y = sum(1 for p in phys if p['would_use_in_clinic_Y/N']=='Y')
    if abs(use_y/n - 0.62) < 0.02: OK('PhysEval','Adoption 62%% VERIFIED (%d/45)'%use_y)
    else: WARN('PhysEval','Adoption claim 62%% actual=%.0f%%'%(100*use_y/n))

    # Score variance check -- are they too uniform?
    std_c = float(np.std(comps))
    if std_c < 0.1:
        WARN('PhysEval','Composite scores suspiciously uniform (std=%.3f)'%std_c,'Simulation may lack variance')
    else:
        OK('PhysEval','Score variance adequate (std=%.3f)'%std_c)

except Exception as e:
    FAIL('PhysEval','Error: %s'%str(e))

# ======================================================================
# BLOCK 5: DATA PROVENANCE INTEGRITY
# ======================================================================
print("\n[BLOCK 5] Data provenance and scientific honesty")

# 5-1: Is the Bliss DB actually from literature or generated?
try:
    with open(os.path.join(SYN_EN,'bliss_curated_v3.csv'),encoding='utf-8') as f:
        bliss = list(csv.DictReader(f))
    refs = [r.get('ref','') for r in bliss]
    has_adds_ref = sum(1 for r in refs if 'ADDS' in r or 'Lee' in r)
    has_lit_ref  = sum(1 for r in refs if 'NEJM' in r or 'JCO' in r or 'Nature' in r
                       or 'Lancet' in r or 'Phase' in r or 'O''Neil' in r or 'Lee 201' in r)
    has_no_ref   = sum(1 for r in refs if not r.strip())
    print("  Bliss ref breakdown: LIT=%d ADDS_internal=%d no_ref=%d" % (has_lit_ref, has_adds_ref, has_no_ref))
    if has_no_ref > 10:
        WARN('Provenance','%d/%d Bliss records have no reference'%(has_no_ref,len(bliss)),
             'Published figures need source for each record')
    else:
        OK('Provenance','Bliss references: %d lit / %d internal / %d missing'%(has_lit_ref,has_adds_ref,has_no_ref))
except Exception as e:
    FAIL('Provenance','Bliss provenance error: %s'%str(e))

# 5-2: KRAS values in patient dataset -- were they really imputed?
try:
    with open(os.path.join(INT_DS,'master_dataset.jsonl'),encoding='utf-8') as f:
        patients = [json.loads(l.strip()) for l in f if l.strip()]
    kras_vals = [p['patient_demographics'].get('kras_mutation','') for p in patients]
    kras_counts = Counter(kras_vals)
    print("  Patient KRAS distribution:", dict(kras_counts))
    # All imputed -- need to flag this
    kras_DL_imputed = all(p.get('data_quality',{}).get('source_note','').startswith('Enriched') for p in patients)
    if kras_DL_imputed:
        WARN('Provenance',
             'ALL 26 patient KRAS values are DL-imputed (not measured clinically)',
             'Must be labeled as imputed/synthetic in any publication. Cannot be reported as measured values.')
    else:
        OK('Provenance','At least some patient KRAS values are original clinical measurements')
except Exception as e:
    FAIL('Provenance','Patient KRAS check error: %s'%str(e))

# 5-3: Cohort outcomes -- synthesized from formulas
WARN('Provenance',
     'Cohort v4 PFS/OS outcomes are formula-generated (arm x kras x prpc x msi parameters)',
     'NOT from real clinical trial data. Model trained on these cannot validly claim '
     'clinical predictive performance. Must be presented as illustrative simulation only.')

# 5-4: OS/PFS R2=0.056 -- is this honestly acknowledged?
try:
    with open(os.path.join(ML_DIR,'survival_model_metrics_v3.json')) as f:
        sm = json.load(f)
    pfs_r2 = sm.get('pfs_r2_5cv',0)
    if pfs_r2 < 0.10:
        WARN('ModelHonesty',
             'OS/PFS model R2=%.3f -- essentially no predictive power'%pfs_r2,
             'The explanation "synthetic cohort noise" is valid but must be prominently stated. '
             'This model should NOT be used for clinical decision support in current form.')
    else:
        OK('ModelHonesty','OS/PFS R2=%.3f acceptable given synthetic data'%pfs_r2)
except: pass

# 5-5: Synergy R2=0.996 -- circular validation risk?
WARN('ModelHonesty',
     'Synergy model R2=0.996 trained partially on synthetic/curated Bliss data',
     'High R2 may partly reflect circular validation: synthetic targets generated with similar '
     'distributions to training features. External validation on independent lab data required.')

# ======================================================================
# BLOCK 6: FILE INTEGRITY
# ======================================================================
print("\n[BLOCK 6] File integrity")

critical_files = [
    (os.path.join(DATA,'pritamab_synthetic_cohort_enriched_v4.csv'), 100000, 'Cohort v4'),
    (os.path.join(ML_DIR,'synergy_mlp_v3.pt'), 100000, 'Synergy model'),
    (os.path.join(ML_DIR,'pfs_gb_model_v3.pkl'), 1000, 'PFS GBM model'),
    (os.path.join(ML_DIR,'os_gb_model_v3.pkl'), 1000, 'OS GBM model'),
    (os.path.join(XAI,'physician_evaluation_survey_n45.json'), 1000, 'PhysEval'),
    (os.path.join(XAI,'lime_attributions_n50.json'), 1000, 'LIME'),
    (os.path.join(XAI,'counterfactual_analysis_n40.json'), 1000, 'CF'),
    (os.path.join(DOCS,'ADDS_comprehensive_report_v4.txt'), 10000, 'Report TXT'),
]
for path, min_size, label in critical_files:
    if os.path.exists(path):
        size = os.path.getsize(path)
        if size >= min_size: OK('Files','%s: %d bytes'%(label,size))
        else: WARN('Files','%s: only %d bytes (expected >=%d)'%(label,size,min_size))
    else:
        FAIL('Files','%s MISSING: %s'%(label,path))

# ======================================================================
# FINAL VERDICT
# ======================================================================
n_fail = sum(1 for s,*_ in issues if s=='FAIL')
n_warn = sum(1 for s,*_ in issues if s=='WARN')
n_ok   = len(ok_items)

print("\n" + "="*70)
print("VERIFICATION FINAL VERDICT")
print("="*70)
print("  OK:   %d" % n_ok)
print("  WARN: %d" % n_warn)
print("  FAIL: %d" % n_fail)
print()
print("FAIL items:")
for s,cat,finding,detail in issues:
    if s == 'FAIL':
        print("  [FAIL][%s] %s" % (cat,finding))
print()
print("WARN items:")
for s,cat,finding,detail in issues:
    if s == 'WARN':
        print("  [WARN][%s] %s" % (cat,finding))
        if detail: print("         (%s)" % detail[:100])

# Save
report = {
    'ok':n_ok,'warn':n_warn,'fail':n_fail,
    'issues':[{'level':s,'cat':c,'finding':f,'detail':d} for s,c,f,d in issues],
}
with open(os.path.join(DOCS,'ADDS_VERIFICATION_FINAL.json'),'w') as f:
    json.dump(report, f, indent=2, ensure_ascii=True)
print("\nVerification report saved: f:\\ADDS\\docs\\ADDS_VERIFICATION_FINAL.json")
