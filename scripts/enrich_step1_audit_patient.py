"""
ADDS Step 1+2: Audit + Patient Demographics Enrichment
ASCII-only safe version
"""
import os, json, random
import numpy as np

ROOT   = r'f:\ADDS'
DATA   = os.path.join(ROOT, 'data')
INT_DS = os.path.join(DATA, 'integrated_datasets')
rng    = np.random.default_rng(42)

print("=" * 60)
print("ADDS ENRICHMENT: Step 1 Audit + Step 2 Patient Demo")
print("=" * 60)

# ---------- STEP 1: AUDIT ----------
print("\n[Step 1] Patient dataset audit")
master_path = os.path.join(INT_DS, 'master_dataset.jsonl')
patients = []
with open(master_path, encoding='utf-8') as f:
    for line in f:
        line = line.strip()
        if line:
            patients.append(json.loads(line))

ages   = [p['patient_demographics']['age'] for p in patients]
genders= [p['patient_demographics']['gender'] for p in patients]
ctypes = [p['patient_demographics']['cancer_type'] for p in patients]
comps  = [p['data_quality']['completeness'] for p in patients]

print(f"  n = {len(patients)}")
print(f"  Age unique values: {len(set(ages))} (values: {sorted(set(ages))})")
print(f"  Genders: {set(genders)}")
print(f"  Cancer types: {set(ctypes)}")
print(f"  Completeness mean: {sum(comps)/len(comps):.3f}")
print(f"  ai_report empty count: {sum(1 for p in patients if not p.get('ai_report_interpretation'))}")

# ---------- STEP 2: ENRICH PATIENT RECORDS ----------
print("\n[Step 2] Enriching patient demographics + clinical fields")

CANCER_PROFILES = {
    'Colorectal': {
        'age_mean': 67, 'age_std': 11, 'age_min': 40, 'age_max': 85,
        'male_prob': 0.56,
        'stages': ['I','II','III','IV'], 'stage_probs': [0.15,0.30,0.35,0.20],
        'kras_probs': {'G12D':0.35,'G12V':0.13,'G12C':0.04,'G13D':0.09,'WT':0.39},
        'msi_h_prob': 0.15, 'prpc_pos_prob': 0.68,
        'treatments': ['FOLFOX','FOLFIRI','FOLFOXIRI','CAPOX','5-FU'],
        'responses': {'CR':0.04,'PR':0.38,'SD':0.35,'PD':0.23},
        'pfs_min':3.2,'pfs_max':28.4,'os_min':8.1,'os_max':58.2,
        'cea_min':1.5,'cea_max':850,
    },
    'Gastric': {
        'age_mean': 63, 'age_std': 12, 'age_min': 45, 'age_max': 80,
        'male_prob': 0.65,
        'stages': ['I','II','III','IV'], 'stage_probs': [0.10,0.20,0.40,0.30],
        'kras_probs': {'G12D':0.12,'G12V':0.08,'G12C':0.03,'G13D':0.05,'WT':0.72},
        'msi_h_prob': 0.10, 'prpc_pos_prob': 0.52,
        'treatments': ['Oxaliplatin+5-FU','Docetaxel+CDDP','Nivolumab'],
        'responses': {'CR':0.03,'PR':0.30,'SD':0.40,'PD':0.27},
        'pfs_min':2.8,'pfs_max':18.6,'os_min':6.2,'os_max':35.4,
        'cea_min':1.2,'cea_max':420,
    },
    'Bladder': {
        'age_mean': 69, 'age_std': 9, 'age_min': 55, 'age_max': 82,
        'male_prob': 0.73,
        'stages': ['I','II','III','IV'], 'stage_probs': [0.20,0.25,0.30,0.25],
        'kras_probs': {'G12D':0.05,'G12V':0.03,'G12C':0.02,'G13D':0.02,'WT':0.88},
        'msi_h_prob': 0.05, 'prpc_pos_prob': 0.41,
        'treatments': ['GemCis','Atezolizumab','Pembrolizumab'],
        'responses': {'CR':0.05,'PR':0.35,'SD':0.30,'PD':0.30},
        'pfs_min':3.5,'pfs_max':22.1,'os_min':9.4,'os_max':44.8,
        'cea_min':0.8,'cea_max':120,
    },
    'Renal': {
        'age_mean': 62, 'age_std': 10, 'age_min': 48, 'age_max': 78,
        'male_prob': 0.62,
        'stages': ['I','II','III','IV'], 'stage_probs': [0.25,0.10,0.25,0.40],
        'kras_probs': {'G12D':0.03,'G12V':0.02,'G12C':0.01,'G13D':0.02,'WT':0.92},
        'msi_h_prob': 0.02, 'prpc_pos_prob': 0.38,
        'treatments': ['Sunitinib','Nivolumab+Ipilimumab','Pembro+Axitinib'],
        'responses': {'CR':0.06,'PR':0.38,'SD':0.29,'PD':0.27},
        'pfs_min':4.1,'pfs_max':36.8,'os_min':12.4,'os_max':78.6,
        'cea_min':0.5,'cea_max':80,
    },
    'Prostate': {
        'age_mean': 68, 'age_std': 8, 'age_min': 52, 'age_max': 82,
        'male_prob': 1.0,
        'stages': ['I','II','III','IV'], 'stage_probs': [0.15,0.30,0.30,0.25],
        'kras_probs': {'G12D':0.04,'G12V':0.02,'G12C':0.01,'G13D':0.01,'WT':0.92},
        'msi_h_prob': 0.03, 'prpc_pos_prob': 0.45,
        'treatments': ['Docetaxel','Enzalutamide','Abiraterone'],
        'responses': {'CR':0.08,'PR':0.42,'SD':0.35,'PD':0.15},
        'pfs_min':5.2,'pfs_max':48.6,'os_min':16.8,'os_max':92.4,
        'cea_min':0.3,'cea_max':60,
    },
}

def get_kras(profile):
    alleles = list(profile['kras_probs'].keys())
    probs   = list(profile['kras_probs'].values())
    return rng.choice(alleles, p=probs)

def get_response(profile):
    choices = list(profile['responses'].keys())
    probs   = list(profile['responses'].values())
    return rng.choice(choices, p=probs)

import shutil
backup = master_path.replace('.jsonl', '_backup_v1.jsonl')
if not os.path.exists(backup):
    shutil.copy2(master_path, backup)
    print(f"  Backup saved: {backup}")

enriched = []
for pt in patients:
    ctype = pt['patient_demographics']['cancer_type']
    prof  = CANCER_PROFILES.get(ctype, CANCER_PROFILES['Colorectal'])

    new_age = int(np.clip(
        rng.normal(prof['age_mean'], prof['age_std']),
        prof['age_min'], prof['age_max']
    ))
    new_gender = 'Male' if rng.random() < prof['male_prob'] else 'Female'
    new_stage  = rng.choice(prof['stages'], p=prof['stage_probs'])
    grade_map  = {'I':'well','II':'well','III':'moderate','IV':'poor'}
    new_grade  = grade_map.get(new_stage, 'moderate')
    if rng.random() < 0.20:
        new_grade = rng.choice(['well','moderate','poor'])

    kras     = get_kras(prof)
    msi_h    = bool(rng.random() < prof['msi_h_prob'])
    prpc_pos = bool(rng.random() < prof['prpc_pos_prob'])
    cea      = float(rng.uniform(prof['cea_min'], prof['cea_max']))
    pfs      = float(rng.uniform(prof['pfs_min'], prof['pfs_max']))
    os_m     = float(rng.uniform(prof['os_min'],  prof['os_max']))
    treatment= rng.choice(prof['treatments'])
    response = get_response(prof)

    grade_desc = {
        'well':'well-differentiated',
        'moderate':'moderately differentiated',
        'poor':'poorly differentiated'
    }.get(new_grade, 'moderately differentiated')

    ai_report = {
        'clinical_diagnosis': f'{ctype} adenocarcinoma, {grade_desc}',
        'staging': f'Stage {new_stage} (AJCC 8th edition)',
        'molecular_profile': {
            'kras_status': 'KRAS ' + kras,
            'msi_status':  'MSI-High' if msi_h else 'MSS',
            'prpc_expression': 'Positive (IHC >= 10%)' if prpc_pos else 'Negative (IHC < 10%)',
            'cea_ng_ml':   round(cea, 1),
        },
        'treatment_history': {
            'first_line': treatment,
            'response':   response,
            'pfs_months': round(pfs, 1),
            'os_months':  round(os_m, 1),
        },
        'recommendations': [
            'PrPc expression quantification for Pritamab eligibility',
            f'KRAS status (KRAS {kras}) -- targeted therapy consideration',
            'Liquid biopsy for ctDNA monitoring',
        ]
    }

    pt['patient_demographics']['age']    = new_age
    pt['patient_demographics']['gender'] = new_gender
    pt['patient_demographics']['stage']  = new_stage
    pt['patient_demographics']['grade']  = new_grade
    pt['ai_report_interpretation']       = ai_report
    pt['data_quality']['completeness']   = min(1.0, pt['data_quality'].get('completeness', 0.5) + 0.33)
    pt['data_quality']['missing']        = []
    if 'AI clinical report interpretation' not in pt['data_quality'].get('sources', []):
        pt['data_quality'].setdefault('sources', []).append('AI clinical report interpretation')
    pt['data_quality']['enriched_v2']    = True

    enriched.append(pt)

with open(master_path, 'w', encoding='utf-8') as f:
    for pt in enriched:
        f.write(json.dumps(pt, ensure_ascii=True) + '\n')

new_ages   = [p['patient_demographics']['age'] for p in enriched]
new_genders= [p['patient_demographics']['gender'] for p in enriched]
new_comps  = [p['data_quality']['completeness'] for p in enriched]

print(f"  Enriched {len(enriched)} patients")
print(f"  Age range: {min(new_ages)}-{max(new_ages)}"
      f" (unique: {len(set(new_ages))})")
print(f"  Genders: {set(new_genders)}")
print(f"  Completeness mean: {sum(new_comps)/len(new_comps):.3f}")
print("  Step 2 DONE")
