"""
ADDS Final Verification: Validate all enriched datasets
ASCII-only
"""
import os, json, csv, datetime
import numpy as np

ROOT   = r'f:\ADDS'
DATA   = os.path.join(ROOT, 'data')
INT_DS = os.path.join(DATA, 'integrated_datasets')
ML_DIR = os.path.join(DATA, 'ml_training')
SYN_DIR= os.path.join(DATA, 'synergy_enriched')
DOCS   = os.path.join(ROOT, 'docs')

rng = np.random.default_rng(42)

print("=" * 60)
print("ADDS FINAL VERIFICATION REPORT")
print("=" * 60)

PASS = 0; FAIL = 0; WARN = 0
results = []

def ok(cat, msg):
    global PASS; PASS += 1
    results.append(('OK',cat,msg)); print(f"  OK   [{cat}] {msg}")
def fail(cat, msg):
    global FAIL; FAIL += 1
    results.append(('FAIL',cat,msg)); print(f"  FAIL [{cat}] {msg}")
def warn(cat, msg):
    global WARN; WARN += 1
    results.append(('WARN',cat,msg)); print(f"  WARN [{cat}] {msg}")

# ---- Patient dataset ----
print("\n[1] Patient Dataset Verification")
master = os.path.join(INT_DS, 'master_dataset.jsonl')
patients = []
with open(master, encoding='utf-8') as f:
    for line in f:
        line = line.strip()
        if line: patients.append(json.loads(line))

ages  = [p['patient_demographics']['age'] for p in patients]
gends = [p['patient_demographics']['gender'] for p in patients]
comps = [p['data_quality']['completeness'] for p in patients]
has_report = [bool(p.get('ai_report_interpretation')) for p in patients]
stages= [p['patient_demographics']['stage'] for p in patients]

if len(set(ages)) >= 10: ok('Patient','Age diversity: %d unique values in [%d,%d]' % (len(set(ages)),min(ages),max(ages)))
else: fail('Patient','Age diversity insufficient: only %d unique values' % len(set(ages)))

if len(set(gends)) > 1: ok('Patient','Gender diversity: %s' % str(set(gends)))
else: fail('Patient','Gender single-valued: %s' % str(set(gends)))

if len(set(stages)) >= 3: ok('Patient','Stage diversity: %s' % str(sorted(set(stages))))
else: warn('Patient','Stage diversity limited: %s' % str(set(stages)))

mean_comp = sum(comps)/len(comps)
if mean_comp >= 0.90: ok('Patient','Completeness mean=%.3f (>= 0.90)' % mean_comp)
else: fail('Patient','Completeness mean=%.3f (< 0.90)' % mean_comp)

n_rep = sum(has_report)
if n_rep == len(patients): ok('Patient','ai_report_interpretation: all %d patients filled' % n_rep)
else: fail('Patient','ai_report_interpretation: only %d/%d filled' % (n_rep,len(patients)))

# Check KRAS in reports
n_kras = sum(1 for p in patients if p.get('ai_report_interpretation',{}).get('molecular_profile',{}).get('kras_status',''))
if n_kras >= len(patients)//2: ok('Patient','KRAS status in reports: %d/%d' % (n_kras,len(patients)))
else: warn('Patient','KRAS status sparse: %d/%d' % (n_kras,len(patients)))

# ---- Bliss database ----
print("\n[2] Bliss Synergy Database Verification")
bliss_path = os.path.join(SYN_DIR, 'bliss_curated_v2.csv')
if os.path.exists(bliss_path):
    with open(bliss_path, encoding='utf-8') as f:
        reader = csv.DictReader(f)
        bliss_recs = list(reader)
    n_prit   = sum(1 for r in bliss_recs if 'Pritamab' in r.get('combination',''))
    n_ctrl   = sum(1 for r in bliss_recs if 'Pritamab' not in r.get('combination',''))
    kras_set = set(r.get('kras','') for r in bliss_recs)
    bliss_vals= [float(r['bliss']) for r in bliss_recs]
    refs_set  = set(r.get('ref','') for r in bliss_recs)

    if len(bliss_recs) >= 100: ok('Bliss','Total records: %d (>= 100)' % len(bliss_recs))
    else: fail('Bliss','Records only %d (< 100)' % len(bliss_recs))

    if len(kras_set) >= 5: ok('Bliss','KRAS alleles covered: %s' % str(kras_set))
    else: warn('Bliss','KRAS coverage limited: %s' % str(kras_set))

    if n_prit >= 40: ok('Bliss','Pritamab combo records: %d' % n_prit)
    else: warn('Bliss','Pritamab records only %d' % n_prit)

    if len(refs_set) >= 4: ok('Bliss','References: %d different sources' % len(refs_set))
    else: warn('Bliss','Limited sources: %d' % len(refs_set))

    # Value sanity
    bad_vals = [v for v in bliss_vals if v < -5 or v > 50]
    if not bad_vals: ok('Bliss','All Bliss values in plausible range [-5,50]')
    else: fail('Bliss','%d Bliss values out of plausible range' % len(bad_vals))

    # SOT consistency: Pritamab+Oxaliplatin G12D primary should be ~21.7
    prit_ox_g12d = [float(r['bliss']) for r in bliss_recs
                    if r.get('combination','') == 'Pritamab+Oxaliplatin'
                    and r.get('kras','') == 'G12D'
                    and r.get('augmented','False') == 'False']
    if prit_ox_g12d:
        mean_v = sum(prit_ox_g12d)/len(prit_ox_g12d)
        if abs(mean_v - 21.7) <= 0.5: ok('Bliss','Pritamab+Oxali G12D primary mean=%.2f (near 21.7)' % mean_v)
        else: fail('Bliss','Pritamab+Oxali G12D drift: mean=%.2f (expected ~21.7)' % mean_v)
else:
    fail('Bliss','bliss_curated_v2.csv not found')

# ---- DL Model metrics ----
print("\n[3] DL Model Metrics Verification")
metrics_path = os.path.join(ML_DIR, 'evaluation_results_v3.json')
if os.path.exists(metrics_path):
    with open(metrics_path) as f:
        m = json.load(f)
    r2  = m.get('r2_5cv',0)
    rho = m.get('rho_5cv', m.get('drug_rank_rho',0))
    dr  = m.get('drug_rank_rho',0)
    t1  = m.get('top1_match',False)
    t2  = m.get('top2_match',False)

    if r2 >= 0.90: ok('DL','r2_5cv=%.3f (>= 0.70 target, EXCELLENT)' % r2)
    elif r2 >= 0.70: ok('DL','r2_5cv=%.3f (>= 0.70 target MET)' % r2)
    else: fail('DL','r2_5cv=%.3f (< 0.70 target)' % r2)

    if rho >= 0.90: ok('DL','rho_5cv=%.3f (EXCELLENT)' % rho)
    elif rho >= 0.70: ok('DL','rho_5cv=%.3f (target met)' % rho)
    else: fail('DL','rho_5cv=%.3f (< 0.70 target)' % rho)

    if dr >= 0.90: ok('DL','drug_rank_rho=%.3f (EXCELLENT)' % dr)
    elif dr >= 0.70: ok('DL','drug_rank_rho=%.3f (target met)' % dr)
    else: fail('DL','drug_rank_rho=%.3f (< 0.70)' % dr)

    if t1: ok('DL','Top-1 match: CORRECT (Oxaliplatin #1)')
    else: fail('DL','Top-1 match: WRONG')

    if t2: ok('DL','Top-2 match: CORRECT')
    else: warn('DL','Top-2 match: PARTIAL')

    model_path = os.path.join(ML_DIR,'synergy_mlp_v3.pt')
    if os.path.exists(model_path):
        sz = os.path.getsize(model_path)
        ok('DL','Model saved: synergy_mlp_v3.pt (%d KB)' % (sz//1024))
    else:
        fail('DL','synergy_mlp_v3.pt not found')
else:
    fail('DL','evaluation_results_v3.json not found')

# ---- TCGA clinical ----
print("\n[4] TCGA Clinical Verification")
tcga_enriched = os.path.join(ML_DIR,'tcga_crc_clinical_enriched_v2.csv')
if os.path.exists(tcga_enriched):
    with open(tcga_enriched, encoding='utf-8') as f:
        reader = csv.DictReader(f)
        tcga_e = list(reader)
    n_k = sum(1 for r in tcga_e if r.get('kras_mutation','').strip())
    n_m = sum(1 for r in tcga_e if r.get('msi_status','').strip())
    n_p = sum(1 for r in tcga_e if r.get('prpc_expression','').strip())
    ok('TCGA','Rows: %d' % len(tcga_e))
    if n_k == len(tcga_e): ok('TCGA','KRAS 100%% filled (%d/%d)' % (n_k,len(tcga_e)))
    else: fail('TCGA','KRAS only %d/%d' % (n_k,len(tcga_e)))
    if n_m >= len(tcga_e)*0.90: ok('TCGA','MSI >= 90%% filled (%d/%d)' % (n_m,len(tcga_e)))
    else: warn('TCGA','MSI partially filled: %d/%d' % (n_m,len(tcga_e)))
else:
    warn('TCGA','Enriched TCGA file not found')

# ---- Synthetic cohort ----
print("\n[5] Synthetic Cohort Verification")
cohort_enriched = os.path.join(DATA,'pritamab_synthetic_cohort_enriched_v2.csv')
if os.path.exists(cohort_enriched):
    with open(cohort_enriched, encoding='utf-8') as f:
        reader = csv.DictReader(f)
        coh = list(reader)
    key_fields = ['bliss_score_predicted','prpc_expression_level','cea_baseline','msi_status','dl_confidence']
    all_present = all(f in (coh[0].keys() if coh else {}) for f in key_fields)
    ok('Cohort','n=%d patients' % len(coh))
    if all_present: ok('Cohort','All key fields present: %s' % str(key_fields))
    else:
        missing_f = [f for f in key_fields if f not in (coh[0].keys() if coh else {})]
        fail('Cohort','Missing fields: %s' % str(missing_f))
    # Check bliss range
    bliss_pred = [float(r.get('bliss_score_predicted',0)) for r in coh if r.get('bliss_score_predicted')]
    if bliss_pred:
        b_min,b_max = min(bliss_pred),max(bliss_pred)
        if 4 < b_min and b_max < 30: ok('Cohort','bliss_predicted range [%.1f,%.1f] plausible' % (b_min,b_max))
        else: warn('Cohort','bliss_predicted range [%.1f,%.1f] may be out of bounds' % (b_min,b_max))
else:
    warn('Cohort','Enriched cohort CSV not found')

# ---- Summary report ----
print("\n" + "=" * 60)
print("FINAL SUMMARY: %d OK  %d WARN  %d FAIL" % (PASS, WARN, FAIL))
print("=" * 60)

if FAIL == 0:
    verdict = "PASS -- All critical checks passed"
elif FAIL <= 2:
    verdict = "CONDITIONAL PASS -- minor issues only"
else:
    verdict = "NEEDS WORK -- %d critical failures" % FAIL
print("Verdict: " + verdict)

# Write machine-readable final results
final_results = {
    'timestamp': '2026-03-08T23:30 KST',
    'verdict': verdict,
    'ok': PASS, 'warn': WARN, 'fail': FAIL,
    'enrichment_summary': {
        'patient_n': len(patients),
        'patient_age_range': [min(ages),max(ages)],
        'patient_completeness_mean': round(sum(comps)/len(comps),3),
        'bliss_records_total': len(bliss_recs) if 'bliss_recs' in dir() else 0,
        'dl_r2_5cv': m.get('r2_5cv',0) if 'm' in dir() else 0,
        'dl_drug_rank_rho': m.get('drug_rank_rho',0) if 'm' in dir() else 0,
        'tcga_rows': len(tcga_e) if 'tcga_e' in dir() else 0,
        'synthetic_cohort_rows': len(coh) if 'coh' in dir() else 0,
    },
    'checks': [{'status':s,'category':c,'message':msg} for s,c,msg in results]
}

result_path = os.path.join(DOCS, 'ADDS_ENRICHMENT_RESULTS.json')
os.makedirs(DOCS, exist_ok=True)
with open(result_path,'w',encoding='utf-8') as f:
    json.dump(final_results, f, indent=2, ensure_ascii=True)
print("\nResults written: " + result_path)
