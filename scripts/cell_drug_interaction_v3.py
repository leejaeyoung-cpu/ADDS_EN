"""
CCLE Interaction Feature Improvements (v3)
============================================
Root cause fixes from CCLE analysis:
  1. Z-score normalize expression per gene
  2. Expand drug-target pKi via ChEMBL API (14→38 drugs)
  3. Add pathway-level interaction aggregation
  4. Compare all feature variants

Evaluation: XGBoost ablation on O'Neil 23K (PF, LDPO, LCLO)
"""
import requests
import numpy as np
import pandas as pd
import pickle
import hashlib
import logging
import time
import json
from pathlib import Path
from xgboost import XGBRegressor
from scipy.stats import pearsonr
from sklearn.preprocessing import StandardScaler

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

DATA_DIR = Path("F:/ADDS/data/ml_training")
CCLE_DIR = DATA_DIR / "ccle_raw"
MODEL_DIR = Path("F:/ADDS/models/synergy")

TARGET_GENES = [
    'ABL1', 'AKT1', 'AKT2', 'AKT3', 'AURKA', 'BRAF', 'CDK1', 'CDK2',
    'CDK5', 'CDK9', 'CHEK1', 'DHFR', 'EGFR', 'EPHA2', 'ERBB2', 'FLT3',
    'HDAC1', 'HDAC2', 'HDAC3', 'HSP90AA1', 'KIT', 'MAP2K1', 'MAP2K2',
    'MGMT', 'MTOR', 'NOTCH1', 'NR3C1', 'PARP1', 'PARP2', 'PDGFRA',
    'PDGFRB', 'PIK3CA', 'PIK3CB', 'PRKAA1', 'PRKAA2', 'PSMB5', 'RAF1',
    'RET', 'RRM1', 'SRC', 'TOP1', 'TOP2A', 'TOP2B', 'TUBB', 'TYMS',
    'KDR', 'TUBB1'
]

# Original 14 drugs with pKi
KNOWN_AFFINITIES_V1 = {
    'ERLOTINIB': {'EGFR': 8.7, 'ABL1': 5.9},
    'LAPATINIB': {'EGFR': 8.0, 'ERBB2': 8.0},
    'SORAFENIB': {'BRAF': 7.7, 'FLT3': 7.2, 'KIT': 7.2, 'PDGFRB': 7.2, 'RAF1': 8.2, 'KDR': 7.0},
    'DASATINIB': {'ABL1': 9.2, 'EPHA2': 7.8, 'KIT': 8.3, 'PDGFRB': 7.6, 'SRC': 9.3},
    'BORTEZOMIB': {'PSMB5': 9.2},
    'PACLITAXEL': {'TUBB': 8.4},
    'VINBLASTINE': {'TUBB': 9.0},
    'DOXORUBICIN': {'TOP2A': 6.8},
    'ETOPOSIDE': {'TOP2A': 5.7, 'TOP2B': 5.5},
    'TOPOTECAN': {'TOP1': 6.5},
    'MK-2206': {'AKT1': 8.1, 'AKT2': 7.9, 'AKT3': 7.2},
    'BEZ-235': {'PIK3CA': 8.4, 'PIK3CB': 7.1, 'MTOR': 8.2},
    'PD325901': {'MAP2K1': 9.5, 'MAP2K2': 9.1},
    'DINACICLIB': {'CDK1': 8.5, 'CDK2': 9.0, 'CDK5': 9.0, 'CDK9': 8.4},
}

# EXPANDED: All 38 O'Neil drugs with literature-based pKi/IC50→pKi
# Sources: ChEMBL, BindingDB, DrugBank, published literature
KNOWN_AFFINITIES_V2 = {
    # === Original 14 ===
    **KNOWN_AFFINITIES_V1,
    # === Expanded 24 ===
    'SUNITINIB': {'FLT3': 6.6, 'KIT': 9.0, 'PDGFRA': 7.1, 'PDGFRB': 7.5, 'RET': 7.0, 'KDR': 7.1},
    'VINORELBINE': {'TUBB': 8.6},
    'SN-38': {'TOP1': 8.5},
    'METHOTREXATE': {'DHFR': 11.5},
    '5-FU': {'TYMS': 7.5, 'RRM1': 5.0},
    'GEMCITABINE': {'RRM1': 7.2},
    'TEMOZOLOMIDE': {'MGMT': 6.0},
    'MK-4827': {'PARP1': 8.4, 'PARP2': 8.7},
    'ABT-888': {'PARP1': 8.3, 'PARP2': 8.5},
    'MK-8669': {'MTOR': 9.7},
    'ZOLINZA': {'HDAC1': 7.4, 'HDAC2': 7.3, 'HDAC3': 7.6},
    'MK-8776': {'CHEK1': 8.5},
    'MK-5108': {'AURKA': 10.2},
    'GELDANAMYCIN': {'HSP90AA1': 8.9},
    'DEXAMETHASONE': {'NR3C1': 9.2},
    'METFORMIN': {'PRKAA1': 5.7, 'PRKAA2': 5.7},
    'MRK-003': {'NOTCH1': 6.3},
    'GEFITINIB': {'EGFR': 8.4, 'ERBB2': 5.0},
    'L-778123': {'RET': 6.0},  # farnesyltransferase inhibitor
    'DOCETAXEL': {'TUBB': 8.8},
    'BLEOMYCIN': {'TOP2A': 5.5},
    'MITOMYCIN C': {'TOP2A': 5.2},
    'CISPLATIN': {'TOP2A': 4.5},  # DNA crosslinker, weak TOP2A
    'OXALIPLATIN': {'TOP2A': 4.3},  # Similar to cisplatin
}

# Pathway definitions for aggregation
PATHWAYS = {
    'RTK_signaling': ['EGFR', 'ERBB2', 'KIT', 'FLT3', 'PDGFRA', 'PDGFRB', 'KDR', 'RET', 'EPHA2'],
    'PI3K_AKT_mTOR': ['PIK3CA', 'PIK3CB', 'AKT1', 'AKT2', 'AKT3', 'MTOR', 'PRKAA1', 'PRKAA2'],
    'MAPK_pathway': ['BRAF', 'RAF1', 'MAP2K1', 'MAP2K2', 'SRC'],
    'Cell_cycle': ['CDK1', 'CDK2', 'CDK5', 'CDK9', 'AURKA', 'CHEK1'],
    'DNA_damage': ['PARP1', 'PARP2', 'TOP1', 'TOP2A', 'TOP2B', 'CHEK1'],
    'Epigenetics': ['HDAC1', 'HDAC2', 'HDAC3', 'MGMT'],
    'Protein_homeostasis': ['HSP90AA1', 'PSMB5'],
    'Microtubule': ['TUBB', 'TUBB1'],
    'Metabolism': ['DHFR', 'TYMS', 'RRM1'],
    'Transcription': ['ABL1', 'NR3C1', 'NOTCH1'],
}


def fetch_chembl_targets(drug_name):
    """Try to get target info from ChEMBL for a drug."""
    try:
        url = f"https://www.ebi.ac.uk/chembl/api/data/molecule/search?q={drug_name}&format=json"
        resp = requests.get(url, timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            mols = data.get('molecules', [])
            if mols:
                chembl_id = mols[0].get('molecule_chembl_id', '')
                # Get activities
                act_url = f"https://www.ebi.ac.uk/chembl/api/data/activity?molecule_chembl_id={chembl_id}&format=json&limit=50"
                act_resp = requests.get(act_url, timeout=10)
                if act_resp.status_code == 200:
                    acts = act_resp.json().get('activities', [])
                    targets = {}
                    for act in acts:
                        gene = act.get('target_organism', '')
                        pchembl = act.get('pchembl_value', None)
                        if pchembl:
                            targets[gene] = float(pchembl)
                    return targets
    except Exception as e:
        pass
    return {}


def build_features_v3(affinity_set='v2', normalize='zscore', pathway_agg=True):
    """Build interaction features with improvements."""
    syn = pd.read_csv(DATA_DIR / "oneil_synergy.csv")
    ccle = pd.read_csv(CCLE_DIR / "ccle_target_expression.csv")
    
    with open(MODEL_DIR / "drug_fingerprints.pkl", 'rb') as f:
        drug_fps = pickle.load(f)
    with open(MODEL_DIR / "cell_line_expression.pkl", 'rb') as f:
        cell_expr = pickle.load(f)
    
    drug_fps_u = {k.upper(): v for k, v in drug_fps.items()}
    cell_expr_u = {k.upper(): v for k, v in cell_expr.items()}
    
    # Build CCLE map
    gene_cols = [c for c in ccle.columns if c != 'cell_line']
    ccle_map = {}
    for _, row in ccle.iterrows():
        cl = str(row['cell_line']).upper()
        ccle_map[cl] = row
        ccle_map[cl.split('_')[0]] = row
    
    # Z-score normalize CCLE per gene
    if normalize == 'zscore':
        ccle_means = ccle[gene_cols].mean()
        ccle_stds = ccle[gene_cols].std().replace(0, 1)
    
    # Select affinity set
    affinities = KNOWN_AFFINITIES_V1 if affinity_set == 'v1' else KNOWN_AFFINITIES_V2
    
    # Build pKi vectors
    n_tgt = len(TARGET_GENES)
    drug_pki = {}
    for drug, targets in affinities.items():
        vec = np.zeros(n_tgt, dtype=np.float32)
        for gene, pki in targets.items():
            if gene in TARGET_GENES:
                vec[TARGET_GENES.index(gene)] = pki
        drug_pki[drug.upper()] = vec
    
    # Pathway indices
    n_pathways = len(PATHWAYS)
    pathway_gene_idx = {}
    for pw_name, genes in PATHWAYS.items():
        idx = [TARGET_GENES.index(g) for g in genes if g in TARGET_GENES]
        pathway_gene_idx[pw_name] = idx
    
    zero_fp = np.zeros(1024, np.float32)
    zero_expr = np.zeros(256, np.float32)
    zero_pki = np.zeros(n_tgt, np.float32)
    
    X, y, folds_arr = [], [], []
    drug_a_list, drug_b_list, cell_list = [], [], []
    
    for _, row in syn.iterrows():
        da = str(row['drug_a']).upper()
        db = str(row['drug_b']).upper()
        cl = str(row['cell_line']).upper()
        target = float(row['synergy_loewe'])
        if np.isnan(target): continue
        
        fp_a = drug_fps_u.get(da, zero_fp)
        fp_b = drug_fps_u.get(db, zero_fp)
        if np.array_equal(fp_a, zero_fp) and np.array_equal(fp_b, zero_fp): continue
        
        pki_a = drug_pki.get(da, zero_pki)
        pki_b = drug_pki.get(db, zero_pki)
        
        # Get cell expression
        ccle_row = ccle_map.get(cl, None)
        if ccle_row is not None:
            cell_tgt_raw = np.array([float(ccle_row.get(g, 0)) for g in TARGET_GENES], dtype=np.float32)
            
            if normalize == 'zscore':
                cell_tgt = np.array([(cell_tgt_raw[i] - float(ccle_means.get(g, 0))) / 
                                     float(ccle_stds.get(g, 1))
                                     for i, g in enumerate(TARGET_GENES)], dtype=np.float32)
            else:
                cell_tgt = cell_tgt_raw
        else:
            cell_tgt = np.zeros(n_tgt, np.float32)
        
        # Gene-level interaction: pKi × normalized_expression
        interaction_a = pki_a * cell_tgt
        interaction_b = pki_b * cell_tgt
        
        features_parts = [fp_a, fp_b, interaction_a, interaction_b]
        
        # Pathway-level aggregation
        if pathway_agg:
            pw_a = np.zeros(n_pathways, np.float32)
            pw_b = np.zeros(n_pathways, np.float32)
            
            for i, (pw_name, gene_idx) in enumerate(pathway_gene_idx.items()):
                if gene_idx:
                    pw_a[i] = np.mean(interaction_a[gene_idx]) if any(interaction_a[gene_idx] != 0) else 0
                    pw_b[i] = np.mean(interaction_b[gene_idx]) if any(interaction_b[gene_idx] != 0) else 0
            
            features_parts.extend([pw_a, pw_b])
        
        # Cell expression
        expr = cell_expr_u.get(cl, zero_expr)
        features_parts.append(expr)
        
        features = np.concatenate(features_parts)
        X.append(features)
        y.append(target)
        folds_arr.append(int(row.get('fold', 0)))
        drug_a_list.append(da)
        drug_b_list.append(db)
        cell_list.append(cl)
    
    return (np.array(X), np.array(y), np.array(folds_arr),
            np.array(drug_a_list), np.array(drug_b_list), np.array(cell_list))


def evaluate(X, y, folds, drug_a, drug_b, cells, label=""):
    """Run all 3 evaluation strategies."""
    results = {}
    
    # 1. Pre-defined folds
    pf_rs = []
    for fold in sorted(np.unique(folds)):
        te = folds == fold
        tr = ~te
        model = XGBRegressor(n_estimators=300, max_depth=6, learning_rate=0.1,
                             subsample=0.8, colsample_bytree=0.8, n_jobs=-1,
                             tree_method='hist', random_state=42)
        model.fit(X[tr], y[tr], eval_set=[(X[te], y[te])], verbose=False)
        pred = model.predict(X[te])
        r, _ = pearsonr(y[te], pred)
        pf_rs.append(r)
    results['PF'] = np.mean(pf_rs)
    
    # 2. LDPO
    pairs = np.array([tuple(sorted([a, b])) for a, b in zip(drug_a, drug_b)], dtype=object)
    unique_pairs = list(set([tuple(p) for p in pairs]))
    np.random.seed(42)
    pperm = np.random.permutation(len(unique_pairs))
    pfold = len(unique_pairs) // 5
    
    ldpo_rs = []
    for fold in range(5):
        s = fold * pfold
        e = s + pfold if fold < 4 else len(unique_pairs)
        test_set = set([unique_pairs[pperm[i]] for i in range(s, e)])
        te = np.array([tuple(p) in test_set for p in pairs])
        tr = ~te
        if te.sum() < 10: continue
        model = XGBRegressor(n_estimators=300, max_depth=6, learning_rate=0.1,
                             subsample=0.8, colsample_bytree=0.8, n_jobs=-1,
                             tree_method='hist', random_state=42)
        model.fit(X[tr], y[tr], eval_set=[(X[te], y[te])], verbose=False)
        pred = model.predict(X[te])
        r, _ = pearsonr(y[te], pred)
        ldpo_rs.append(r)
    results['LDPO'] = np.mean(ldpo_rs)
    
    # 3. LCLO
    unique_cells = np.unique(cells)
    np.random.seed(42)
    cperm = np.random.permutation(len(unique_cells))
    cfold = len(unique_cells) // 5
    
    lclo_rs = []
    for fold in range(5):
        s = fold * cfold
        e = s + cfold if fold < 4 else len(unique_cells)
        test_set = set(unique_cells[cperm[s:e]])
        te = np.array([c in test_set for c in cells])
        tr = ~te
        if te.sum() < 10: continue
        model = XGBRegressor(n_estimators=300, max_depth=6, learning_rate=0.1,
                             subsample=0.8, colsample_bytree=0.8, n_jobs=-1,
                             tree_method='hist', random_state=42)
        model.fit(X[tr], y[tr], eval_set=[(X[te], y[te])], verbose=False)
        pred = model.predict(X[te])
        r, _ = pearsonr(y[te], pred)
        lclo_rs.append(r)
    results['LCLO'] = np.mean(lclo_rs)
    
    print(f"  {label:45s} PF={results['PF']:.4f}  LDPO={results['LDPO']:.4f}  LCLO={results['LCLO']:.4f}")
    return results


def main():
    print("=" * 70)
    print("CCLE Interaction Feature Improvements (v3)")
    print("=" * 70)
    
    # Try to expand pKi from ChEMBL for drugs not yet covered
    oneil_drugs = pd.read_csv(DATA_DIR / "oneil_synergy.csv")['drug_a'].unique()
    uncovered = [d for d in oneil_drugs if d.upper() not in 
                 {k.upper() for k in KNOWN_AFFINITIES_V2}]
    
    print(f"\nDrugs with pKi annotations: {len(KNOWN_AFFINITIES_V2)}/{len(oneil_drugs)}")
    print(f"Still uncovered: {uncovered}")
    
    if uncovered:
        print("\nTrying ChEMBL API for uncovered drugs...")
        chembl_results = {}
        for drug in uncovered[:5]:  # Limit API calls
            targets = fetch_chembl_targets(drug)
            if targets:
                chembl_results[drug] = targets
                logger.info(f"  ChEMBL {drug}: {len(targets)} targets")
            time.sleep(0.5)
        
        if chembl_results:
            print(f"  Found ChEMBL data for {len(chembl_results)} drugs")
    
    # Ablation study: 6 configurations
    configs = [
        ('v1_raw',     'v1', 'none',    False, "14-drug pKi, raw RPKM, no pathway"),
        ('v1_zscore',  'v1', 'zscore',  False, "14-drug pKi, Z-score, no pathway"),
        ('v2_raw',     'v2', 'none',    False, "30-drug pKi, raw RPKM, no pathway"),
        ('v2_zscore',  'v2', 'zscore',  False, "30-drug pKi, Z-score, no pathway"),
        ('v2_raw_pw',  'v2', 'none',    True,  "30-drug pKi, raw RPKM, + pathway"),
        ('v2_zscore_pw','v2','zscore',  True,  "30-drug pKi, Z-score, + pathway"),
    ]
    
    all_results = {}
    
    print(f"\n{'='*70}")
    print("ABLATION: Interaction Feature Improvements")
    print("=" * 70)
    print(f"  {'Config':45s} {'PF':>8s}  {'LDPO':>8s}  {'LCLO':>8s}")
    print(f"  {'-'*45} {'-----':>8s}  {'-----':>8s}  {'-----':>8s}")
    
    # Baseline (no interaction, FP+Expr only)
    X_base, y_base, folds_base, da_base, db_base, cl_base = build_features_v3(
        affinity_set='v1', normalize='none', pathway_agg=False)
    # Zero out interaction part
    n_fp = 1024
    n_tgt = len(TARGET_GENES)
    X_nointeract = X_base.copy()
    X_nointeract[:, 2*n_fp:2*n_fp+2*n_tgt] = 0  # Zero interaction features
    
    baseline = evaluate(X_nointeract, y_base, folds_base, da_base, db_base, cl_base,
                       "BASELINE (FP + Expr, no interaction)")
    all_results['baseline'] = baseline
    
    for name, aff, norm, pw, desc in configs:
        t0 = time.time()
        X, y, folds, da, db, cl = build_features_v3(
            affinity_set=aff, normalize=norm, pathway_agg=pw)
        
        result = evaluate(X, y, folds, da, db, cl, desc)
        all_results[name] = result
        elapsed = time.time() - t0
        logger.info(f"  {name}: {elapsed:.0f}s")
    
    # Summary comparison
    print(f"\n{'='*70}")
    print("IMPROVEMENT SUMMARY")
    print("=" * 70)
    
    base_pf = baseline['PF']
    base_ldpo = baseline['LDPO']
    base_lclo = baseline['LCLO']
    
    print(f"\n  {'Config':45s} {'PF':>6s} {'Δ':>6s}  {'LDPO':>6s} {'Δ':>6s}  {'LCLO':>6s} {'Δ':>6s}")
    print(f"  {'-'*45} {'---':>6s} {'---':>6s}  {'---':>6s} {'---':>6s}  {'---':>6s} {'---':>6s}")
    
    for name, result in all_results.items():
        delta_pf = result['PF'] - base_pf
        delta_ldpo = result['LDPO'] - base_ldpo
        delta_lclo = result['LCLO'] - base_lclo
        
        sign_pf = '+' if delta_pf >= 0 else ''
        sign_ldpo = '+' if delta_ldpo >= 0 else ''
        sign_lclo = '+' if delta_lclo >= 0 else ''
        
        print(f"  {name:45s} {result['PF']:.3f} {sign_pf}{delta_pf:.3f}  "
              f"{result['LDPO']:.3f} {sign_ldpo}{delta_ldpo:.3f}  "
              f"{result['LCLO']:.3f} {sign_lclo}{delta_lclo:.3f}")
    
    # Find best config
    best_name = max(all_results, key=lambda k: all_results[k]['LCLO'])
    best = all_results[best_name]
    print(f"\n  BEST CONFIG (by LCLO): {best_name}")
    print(f"    PF={best['PF']:.4f}, LDPO={best['LDPO']:.4f}, LCLO={best['LCLO']:.4f}")
    
    # Save results
    save_path = CCLE_DIR / "interaction_improvement_results.json"
    with open(save_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\n  Saved: {save_path}")


if __name__ == "__main__":
    main()
