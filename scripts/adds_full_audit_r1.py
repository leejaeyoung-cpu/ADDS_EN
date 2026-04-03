"""
ADDS Full System Audit + Iterative Enrichment Pipeline
Round 1: Comprehensive audit of all data domains
ASCII-safe, no unicode
"""
import os, json, csv, glob
import numpy as np

ROOT   = r'f:\ADDS'
DATA   = os.path.join(ROOT, 'data')
ML_DIR = os.path.join(DATA, 'ml_training')
INT_DS = os.path.join(DATA, 'integrated_datasets')
SYN_EN = os.path.join(DATA, 'synergy_enriched')
DOCS   = os.path.join(ROOT, 'docs')
os.makedirs(DOCS, exist_ok=True)

rng = np.random.default_rng(2026)

issues   = []  # list of (severity, domain, description, action)
results  = []

def crit(domain, msg, action):
    issues.append(('CRITICAL', domain, msg, action))
    print("  [CRIT] [%s] %s" % (domain, msg))

def warn(domain, msg, action):
    issues.append(('WARN', domain, msg, action))
    print("  [WARN] [%s] %s" % (domain, msg))

def ok(domain, msg):
    results.append(('OK', domain, msg))
    print("  [OK]   [%s] %s" % (domain, msg))

print("=" * 65)
print("ADDS SYSTEM FULL AUDIT -- Round 1")
print("=" * 65)

# ======================================================================
# A. PATIENT DATASET
# ======================================================================
print("\n[A] Patient Dataset (master_dataset.jsonl)")
master_path = os.path.join(INT_DS, 'master_dataset.jsonl')
patients = []
try:
    with open(master_path, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line: patients.append(json.loads(line))

    ages   = [p['patient_demographics']['age']    for p in patients]
    gends  = [p['patient_demographics']['gender'] for p in patients]
    comps  = [p['data_quality']['completeness']   for p in patients]
    stages = [p['patient_demographics'].get('stage','Unknown') for p in patients]
    kras_s = [p['patient_demographics'].get('kras_mutation','Unknown') for p in patients]
    orr_ls = [p.get('clinical_outcomes',{}).get('orr') for p in patients]
    pfs_ls = [p.get('clinical_outcomes',{}).get('pfs_months') for p in patients]
    os_ls  = [p.get('clinical_outcomes',{}).get('os_months')  for p in patients]
    ai_r   = [bool(p.get('ai_report_interpretation')) for p in patients]
    radiol = [bool(p.get('radiology_image'))       for p in patients]
    patho  = [bool(p.get('pathology_data'))        for p in patients]
    img_q  = [p.get('data_quality',{}).get('image_quality') for p in patients]

    print("  Patients: %d" % len(patients))

    # Age
    if len(set(ages)) >= 10: ok('Patient','Age diversity OK: [%d-%d]'%(min(ages),max(ages)))
    else: crit('Patient','Age diversity low (%d unique)'%len(set(ages)),'Diversify age distribution')

    # Gender
    if len(set(gends)) > 1: ok('Patient','Gender diverse: %s'%str(set(gends)))
    else: crit('Patient','Single gender','Add female patients')

    # Stage
    if len(set(stages)) >= 3: ok('Patient','Stage diversity: %s'%str(sorted(set(stages))))
    else: warn('Patient','Stage coverage limited: %s'%str(set(stages)),'Add Stage I-IV')

    # KRAS
    n_kras_un = sum(1 for k in kras_s if k in ('Unknown','','None',None))
    if n_kras_un == 0: ok('Patient','KRAS mutation all filled')
    else: crit('Patient','KRAS unknown in %d/%d patients'%(n_kras_un,len(patients)),'Fill KRAS from DL')

    # Completeness
    mean_c = sum(comps)/len(comps)
    if mean_c >= 0.90: ok('Patient','Completeness mean=%.3f'%mean_c)
    else: warn('Patient','Completeness mean=%.3f (target 0.90)'%mean_c,'Enrich missing fields')

    # AI report
    n_ai = sum(ai_r)
    if n_ai == len(patients): ok('Patient','ai_report: all %d filled'%n_ai)
    else: crit('Patient','ai_report missing in %d/%d'%(len(patients)-n_ai,len(patients)),'Fill reports')

    # Radiology
    n_rad = sum(radiol)
    if n_rad >= len(patients)*0.7: ok('Patient','Radiology available: %d/%d'%(n_rad,len(patients)))
    else: warn('Patient','Radiology sparse: %d/%d'%(n_rad,len(patients)),'Generate synthetic CT features')

    # Outcomes
    orr_fill = sum(1 for o in orr_ls if o is not None)
    pfs_fill = sum(1 for p in pfs_ls if p is not None)
    os_fill  = sum(1 for o in os_ls  if o is not None)
    if orr_fill == len(patients): ok('Patient','ORR all filled')
    else: crit('Patient','ORR missing %d/%d'%(len(patients)-orr_fill,len(patients)),'DL imputation')
    if pfs_fill == len(patients): ok('Patient','PFS all filled')
    else: crit('Patient','PFS missing %d/%d'%(len(patients)-pfs_fill,len(patients)),'DL imputation')
    if os_fill  == len(patients): ok('Patient','OS all filled')
    else: crit('Patient','OS missing %d/%d' %(len(patients)-os_fill, len(patients)),'DL imputation')

    # Image quality
    n_img_q = sum(1 for q in img_q if q and q != 'unavailable')
    if n_img_q >= len(patients)*0.6: ok('Patient','Image quality flag: %d/%d'%(n_img_q,len(patients)))
    else: warn('Patient','Image quality flag missing %d/%d'%(len(patients)-n_img_q,len(patients)),'Add quality flags')

except Exception as e:
    crit('Patient','Cannot read master_dataset: %s'%str(e),'Fix file')

# ======================================================================
# B. BLISS SYNERGY DATABASE
# ======================================================================
print("\n[B] Bliss Synergy Database")
bliss_path = os.path.join(SYN_EN, 'bliss_curated_v2.csv')
try:
    with open(bliss_path, encoding='utf-8') as f:
        reader = csv.DictReader(f)
        bliss  = list(reader)
    n_prit  = sum(1 for r in bliss if 'Pritamab' in r.get('combination',''))
    n_kras  = len(set(r.get('kras','') for r in bliss))
    n_refs  = len(set(r.get('ref','')  for r in bliss))
    vals    = [float(r['bliss']) for r in bliss]
    combos  = set(r['combination'] for r in bliss)
    cell_lines = set(r.get('cell_line','') for r in bliss)

    print("  Records: %d | Pritamab combos: %d | KRAS alleles: %d | cell lines: %d" % (
        len(bliss), n_prit, n_kras, len(cell_lines)))

    if len(bliss) >= 100: ok('Bliss','Total >= 100 records')
    else: crit('Bliss','Only %d records'%len(bliss),'Expand from literature')

    if n_kras >= 5: ok('Bliss','KRAS alleles covered: %d'%n_kras)
    else: warn('Bliss','KRAS coverage: %d/5'%n_kras,'Add G12A/G12R/G12S')

    if n_prit >= 40: ok('Bliss','Pritamab combos: %d'%n_prit)
    else: crit('Bliss','Pritamab combos only %d'%n_prit,'Add more combinations')

    # Check if FOLFOXIRI + Pritamab exists
    has_folfoxiri = any('FOLFOXIRI' in c and 'Pritamab' in c for c in combos)
    if has_folfoxiri: ok('Bliss','Pritamab+FOLFOXIRI exists')
    else: warn('Bliss','Pritamab+FOLFOXIRI missing','Add triplet combo Bliss')

    # Phase 2 clinical anchor
    has_ph2 = any('Phase' in r.get('ref','') or 'phase' in r.get('ref','') for r in bliss)
    if has_ph2: ok('Bliss','Clinical phase data referenced')
    else: warn('Bliss','No clinical phase anchor','Add Phase 1/2 SOT data')

    # Value diversity
    std_val = float(np.std(vals))
    if std_val >= 4.0: ok('Bliss','Bliss value spread (std=%.1f) adequate'%std_val)
    else: warn('Bliss','Bliss values too homogeneous (std=%.1f)'%std_val,'Add diverse values')

    # Cell line coverage
    if len(cell_lines) >= 5: ok('Bliss','Cell lines: %d'%len(cell_lines))
    else: warn('Bliss','Cell line coverage: %d'%len(cell_lines),'Add HCT116/LoVo/SW620')

    # antagonism records (negative Bliss)
    n_antag = sum(1 for v in vals if v < 0)
    if n_antag >= 2: ok('Bliss','Antagonism records: %d (realistic)'%n_antag)
    else: warn('Bliss','No antagonism records -- unrealistically all positive','Add known antagonistic pairs')

except Exception as e:
    crit('Bliss','Cannot read bliss_curated_v2.csv: %s'%str(e),'Rebuild Bliss DB')

# ======================================================================
# C. SYNTHETIC COHORT
# ======================================================================
print("\n[C] Synthetic Cohort")
cohort_path = os.path.join(DATA, 'pritamab_synthetic_cohort_enriched_v2.csv')
try:
    with open(cohort_path, encoding='utf-8') as f:
        reader = csv.DictReader(f)
        cohort = list(reader)
        fields = reader.fieldnames if reader.fieldnames else list(cohort[0].keys()) if cohort else []

    print("  n=%d | fields=%d" % (len(cohort), len(fields)))

    if len(cohort) >= 1000: ok('Cohort','n=%d (>= 1000)'%len(cohort))
    else: crit('Cohort','n=%d < 1000'%len(cohort),'Expand cohort')

    # Required fields audit
    required = ['arm','kras_allele','orr','dcr','dl_pfs_months','dl_os_months',
                'bliss_score_predicted','prpc_expression','prpc_expression_level',
                'cea_baseline','msi_status','dl_confidence','best_pct_change',
                'chemo_drug']
    for req in required:
        pct = 100*sum(1 for r in cohort if r.get(req,'').strip())/max(len(cohort),1)
        if pct >= 95: ok('Cohort','%s: %.0f%% filled'%(req,pct))
        else: warn('Cohort','%s: only %.0f%% filled'%(req,pct),'DL imputation')

    # Missing: toxicity profile per patient
    has_tox = any('toxicity' in f.lower() or 'ae_grade' in f.lower() for f in fields)
    if has_tox: ok('Cohort','Toxicity/AE fields present')
    else: crit('Cohort','No patient-level toxicity fields in cohort','Add per-patient AE profile')

    # Missing: biomarker panel
    has_cea  = 'cea_baseline' in fields
    has_cyto = any('cytokine' in f.lower() for f in fields)
    has_ctdna= any('ctdna' in f.lower() for f in fields)
    if has_cea:  ok('Cohort','CEA baseline present')
    else: warn('Cohort','CEA missing','Add CEA')
    if has_cyto: ok('Cohort','Cytokine data present')
    else: warn('Cohort','No cytokine features','Add IL-6/TNF/IFN-g')
    if has_ctdna:ok('Cohort','ctDNA data present')
    else: crit('Cohort','No ctDNA field','Add ctDNA VAF -- critical liquid biopsy marker')

    # Missing: pharmacokinetics
    has_pk = any('pk' in f.lower() or 'auc' in f.lower() or 'cmax' in f.lower() for f in fields)
    if has_pk: ok('Cohort','PK parameters present')
    else: crit('Cohort','No PK/PD parameters (AUC/Cmax/t_half)','Add PK from DL simulation')

    # KRAS diversity
    kras_vals = set(r.get('kras_allele','') for r in cohort)
    if len(kras_vals) >= 5: ok('Cohort','KRAS allele diversity: %s'%str(kras_vals))
    else: warn('Cohort','KRAS alleles limited: %s'%str(kras_vals),'Add G12A/G12R/G12S')

except Exception as e:
    crit('Cohort','Cannot read cohort: %s'%str(e),'Rebuild cohort')

# ======================================================================
# D. TCGA CLINICAL
# ======================================================================
print("\n[D] TCGA Clinical")
tcga_path = os.path.join(ML_DIR, 'tcga_crc_clinical_enriched_v2.csv')
try:
    with open(tcga_path, encoding='utf-8') as f:
        reader = csv.DictReader(f)
        tcga   = list(reader)
        tcga_fields = reader.fieldnames if reader.fieldnames else list(tcga[0].keys()) if tcga else []

    print("  n=%d | fields=%d" % (len(tcga), len(tcga_fields)))

    if len(tcga) >= 500: ok('TCGA','n=%d'%len(tcga))
    else: warn('TCGA','n=%d < 500'%len(tcga),'Expand')

    # KRAS
    n_kras = sum(1 for r in tcga if r.get('kras_mutation','').strip())
    if n_kras == len(tcga): ok('TCGA','KRAS 100%% filled')
    else: crit('TCGA','KRAS only %d/%d'%(n_kras,len(tcga)),'Fill KRAS')

    # MSI
    n_msi = sum(1 for r in tcga if r.get('msi_status','').strip())
    if n_msi >= len(tcga)*0.90: ok('TCGA','MSI >= 90%% filled')
    else: warn('TCGA','MSI only %d/%d'%(n_msi,len(tcga)),'Fill MSI')

    # PrPc
    n_prc = sum(1 for r in tcga if r.get('prpc_expression','').strip())
    if n_prc >= len(tcga)*0.90: ok('TCGA','PrPc >= 90%% filled')
    else: warn('TCGA','PrPc only %d/%d'%(n_prc,len(tcga)),'Fill PrPc')

    # Survival
    has_os  = any('os' in f.lower() or 'survival' in f.lower() for f in tcga_fields)
    has_pfs = any('pfs' in f.lower() or 'rfs' in f.lower() or 'dfs' in f.lower() for f in tcga_fields)
    if has_os:  ok('TCGA','OS/survival field present')
    else: warn('TCGA','No OS field','Add OS from TCGA clinical')
    if has_pfs: ok('TCGA','PFS/DFS/RFS field present')
    else: warn('TCGA','No PFS field -- needed for KM analysis','Add DFS from GSE39582')

    # Treatment arm
    has_arm = any('treatment' in f.lower() or 'arm' in f.lower() or 'therapy' in f.lower() for f in tcga_fields)
    if has_arm: ok('TCGA','Treatment/arm field present')
    else: crit('TCGA','No treatment arm field','Add treatment history')

    # CEA
    has_cea = any('cea' in f.lower() for f in tcga_fields)
    if has_cea: ok('TCGA','CEA present')
    else: warn('TCGA','No CEA in TCGA','Add CEA from literature median imputation')

except Exception as e:
    crit('TCGA','Cannot read TCGA: %s'%str(e),'Rebuild TCGA enriched')

# ======================================================================
# E. DL MODEL
# ======================================================================
print("\n[E] DL Synergy Model")
model_path  = os.path.join(ML_DIR, 'synergy_mlp_v3.pt')
metrics_path= os.path.join(ML_DIR, 'evaluation_results_v3.json')

if os.path.exists(model_path):
    sz = os.path.getsize(model_path)//1024
    ok('DL','Model exists: synergy_mlp_v3.pt (%d KB)'%sz)
else:
    crit('DL','Model file missing','Retrain DL model')

if os.path.exists(metrics_path):
    with open(metrics_path) as f: m = json.load(f)
    r2  = m.get('r2_5cv',0)
    rho = m.get('rho_5cv',0) or m.get('drug_rank_rho',0)
    if r2  >= 0.90: ok('DL','r2_5cv=%.3f (EXCELLENT)'%r2)
    elif r2 >= 0.70: ok('DL','r2_5cv=%.3f (target met)'%r2)
    else: crit('DL','r2_5cv=%.3f (< 0.70)'%r2,'Retrain with more data')
    if rho >= 0.90: ok('DL','rho=%.3f (EXCELLENT)'%rho)
    elif rho >= 0.70: ok('DL','rho=%.3f (target met)'%rho)
    else: crit('DL','rho=%.3f (< 0.70)'%rho,'Retrain')

    # Coverage: check if non-CRC cancer types covered
    feat_dim = m.get('feat_dim',0)
    if feat_dim >= 50: ok('DL','Feature dim=%d (adequate)'%feat_dim)
    else: warn('DL','Feature dim=%d (consider expansion)'%feat_dim,'Add mutation/expression features')

    # Missing: PK/PD integration
    has_pk_feat = 'pkpd' in str(m).lower() or 'auc' in str(m).lower()
    if has_pk_feat: ok('DL','PK/PD features integrated')
    else: warn('DL','No PK/PD features in model','Add AUC/Cmax to feature vector')

    # Missing: multimodal (CT/pathology)
    has_mm = 'multimodal' in str(m).lower() or 'image' in str(m).lower()
    if has_mm: ok('DL','Multimodal features present')
    else: warn('DL','Model is tabular-only, no image modality','Add CT/pathology embedding')
else:
    warn('DL','Metrics file missing','Run evaluation')

# Missing: OS/PFS prediction model
os_model = glob.glob(os.path.join(ML_DIR,'*os*model*')) + glob.glob(os.path.join(ML_DIR,'*pfs*model*'))
if os_model: ok('DL','OS/PFS prediction model found')
else: crit('DL','No OS/PFS prediction model','Build survival DL model')

# ======================================================================
# F. PK/PD DATA
# ======================================================================
print("\n[F] PK/PD Data")
pkpd_files = glob.glob(os.path.join(DATA,'**','*pk*'), recursive=True) + \
             glob.glob(os.path.join(DATA,'**','*pkpd*'), recursive=True)
pkpd_files = [f for f in pkpd_files if os.path.isfile(f)]
if pkpd_files:
    ok('PKPD','PK/PD files found: %d'%len(pkpd_files))
else:
    crit('PKPD','No PK/PD dataset found','Generate Pritamab PK parameters from published data')

# ======================================================================
# G. CELL VIABILITY / IC50
# ======================================================================
print("\n[G] Cell Viability / IC50 Data")
ic50_files = glob.glob(os.path.join(DATA,'**','*ic50*'), recursive=True) + \
             glob.glob(os.path.join(DATA,'**','*viability*'), recursive=True) + \
             glob.glob(os.path.join(DATA,'**','*dose_response*'), recursive=True)
ic50_files = [f for f in ic50_files if os.path.isfile(f)]
if ic50_files:
    ok('IC50','IC50/viability files found: %d'%len(ic50_files))
else:
    crit('IC50','No IC50/dose-response data','Generate from CTRPv2/PRISM databases')

# Check GDSC
gdsc_files = glob.glob(os.path.join(DATA,'**','*gdsc*'), recursive=True)
if gdsc_files: ok('IC50','GDSC data found: %d files'%len(gdsc_files))
else: warn('IC50','No GDSC data','Add GDSC IC50 for chemotherapy drugs')

# ======================================================================
# H. PROTEIN STRUCTURE / MOLECULAR
# ======================================================================
print("\n[H] Molecular / Protein Data")
prot_dir = os.path.join(DATA, 'protein_structures')
prot_files = glob.glob(os.path.join(prot_dir,'**','*.pdb')) + \
             glob.glob(os.path.join(prot_dir,'**','*.cif'))
if prot_files: ok('Molecular','Protein structures: %d files'%len(prot_files))
else: warn('Molecular','No protein structure files','Add PrPc/RPSA PDB structures')

smiles_path = os.path.join(ML_DIR,'verified_drug_smiles.json')
if os.path.exists(smiles_path):
    with open(smiles_path) as f: sml = json.load(f)
    if len(sml) >= 10: ok('Molecular','SMILES: %d drugs'%len(sml))
    else: warn('Molecular','SMILES only %d drugs'%len(sml),'Expand SMILES library')
else:
    crit('Molecular','No SMILES file','Add drug SMILES for Morgan fingerprints')

# ======================================================================
# I. DRUG COMBINATION COVERAGE
# ======================================================================
print("\n[I] Drug Combination Coverage")
if 'synergy_combined.csv' in os.listdir(ML_DIR):
    sz = os.path.getsize(os.path.join(ML_DIR,'synergy_combined.csv'))//1024//1024
    ok('DrugComb','synergy_combined.csv (%d MB) -- main large DB'%sz)
else:
    crit('DrugComb','Main synergy_combined.csv missing','Download from DrugComb DB')

# Check KRAS-targeted combos
kras_drugs = ['MRTX1133','MRTX849','AMG510','ARS-1620']
drug_targets_path = os.path.join(ML_DIR,'drug_targets.json')
if os.path.exists(drug_targets_path):
    with open(drug_targets_path) as f: dt = json.load(f)
    dt_keys = list(dt.keys()) if isinstance(dt, dict) else []
    kras_covered = [d for d in kras_drugs if any(d.lower() in k.lower() for k in dt_keys)]
    if len(kras_covered) >= 2: ok('DrugComb','KRAS inhibitors in drug_targets: %s'%str(kras_covered))
    else: warn('DrugComb','KRAS inhibitor coverage: %s'%str(kras_covered),'Add MRTX1133/AMG510')

# ======================================================================
# SUMMARY
# ======================================================================
n_crit = sum(1 for i in issues if i[0]=='CRITICAL')
n_warn = sum(1 for i in issues if i[0]=='WARN')
n_ok   = len(results)

print("\n" + "="*65)
print("AUDIT SUMMARY: %d OK | %d CRITICAL | %d WARN" % (n_ok, n_crit, n_warn))
print("="*65)

print("\nCRITICAL items:")
for sev, domain, msg, action in issues:
    if sev == 'CRITICAL':
        print("  [%s] %s  --> %s" % (domain, msg, action))

print("\nWARN items:")
for sev, domain, msg, action in issues:
    if sev == 'WARN':
        print("  [%s] %s  --> %s" % (domain, msg, action))

# Save audit report
audit_report = {
    'timestamp': '2026-03-10T00:30 KST',
    'ok': n_ok, 'critical': n_crit, 'warn': n_warn,
    'issues': [{'level':s,'domain':d,'msg':m,'action':a} for s,d,m,a in issues],
    'ok_items': [{'domain':d,'msg':m} for s,d,m in results],
}
rpt_path = os.path.join(DOCS,'ADDS_AUDIT_ROUND1.json')
with open(rpt_path,'w',encoding='utf-8') as f:
    json.dump(audit_report, f, indent=2, ensure_ascii=True)
print("\nAudit report saved: %s" % rpt_path)
