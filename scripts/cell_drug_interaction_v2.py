"""
Cell-Drug Interaction v2 — Using REAL CCLE Expression
=====================================================
Replace proxy hash expression with actual CCLE RPKM for 46 target genes.
Re-run full ablation (pre-defined, LDPO, LCLO).
"""
import pandas as pd
import numpy as np
import pickle
import logging
from pathlib import Path
from scipy.stats import pearsonr
import xgboost as xgb

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

DATA_DIR = Path("F:/ADDS/data/ml_training")
CCLE_DIR = DATA_DIR / "ccle_raw"
MODEL_DIR = Path("F:/ADDS/models/synergy")

# Literature pKi values
KNOWN_AFFINITIES = {
    'ERLOTINIB': {'EGFR': 8.7, 'ABL1': 5.9},
    'LAPATINIB': {'EGFR': 8.0, 'ERBB2': 8.0},
    'SORAFENIB': {'BRAF': 7.7, 'FLT3': 7.2, 'KIT': 7.2, 'PDGFRB': 7.2, 'RAF1': 8.2, 'KDR': 7.0},
    'SUNITINIB': {'FLT3': 6.6, 'KIT': 9.0, 'PDGFRA': 7.1, 'RET': 7.0, 'KDR': 7.1},
    'DASATINIB': {'ABL1': 9.2, 'EPHA2': 7.8, 'KIT': 8.3, 'PDGFRB': 7.6, 'SRC': 9.3},
    'BORTEZOMIB': {'PSMB5': 9.2},
    'PACLITAXEL': {'TUBB': 8.4},
    'VINBLASTINE': {'TUBB': 9.0},
    'VINORELBINE': {'TUBB': 8.6},
    'DOXORUBICIN': {'TOP2A': 6.8},
    'ETOPOSIDE': {'TOP2A': 5.7, 'TOP2B': 5.5},
    'TOPOTECAN': {'TOP1': 6.5},
    'SN-38': {'TOP1': 8.5},
    'METHOTREXATE': {'DHFR': 11.5},
    '5-FU': {'TYMS': 7.5},
    'GEMCITABINE': {'RRM1': 7.2},
    'TEMOZOLOMIDE': {'MGMT': 6.0},
    'MK-4827': {'PARP1': 8.4, 'PARP2': 8.7},
    'ABT-888': {'PARP1': 8.3, 'PARP2': 8.5},
    'MK-2206': {'AKT1': 8.1, 'AKT2': 7.9, 'AKT3': 7.2},
    'BEZ-235': {'PIK3CA': 8.4, 'PIK3CB': 7.1, 'MTOR': 8.2},
    'MK-8669': {'MTOR': 9.7},
    'PD325901': {'MAP2K1': 9.5, 'MAP2K2': 9.1},
    'ZOLINZA': {'HDAC1': 7.4, 'HDAC2': 7.3, 'HDAC3': 7.6},
    'DINACICLIB': {'CDK1': 8.5, 'CDK2': 9.0, 'CDK5': 9.0, 'CDK9': 8.4},
    'MK-8776': {'CHEK1': 8.5},
    'MK-5108': {'AURKA': 10.2},
    'GELDANAMYCIN': {'HSP90AA1': 8.9},
    'DEXAMETHASONE': {'NR3C1': 9.2},
    'METFORMIN': {'PRKAA1': 5.7, 'PRKAA2': 5.7},
    'MRK-003': {'NOTCH1': 6.3},
}


def build_features():
    """Build features using real CCLE expression."""
    syn = pd.read_csv(DATA_DIR / "oneil_synergy.csv")
    with open(MODEL_DIR / "drug_fingerprints.pkl", 'rb') as f:
        drug_fps = pickle.load(f)
    with open(MODEL_DIR / "cell_line_expression.pkl", 'rb') as f:
        cell_expr = pickle.load(f)
    
    # Load REAL CCLE expression
    ccle_path = CCLE_DIR / "ccle_target_expression.csv"
    ccle_df = pd.read_csv(ccle_path)
    logger.info(f"CCLE data: {ccle_df.shape[0]} cell lines × {ccle_df.shape[1]-1} genes")
    
    # CCLE cell line name format: NAME_TISSUE
    # O'Neil cell lines may have different names — build mapping
    ccle_cell_map = {}
    for _, row in ccle_df.iterrows():
        cl_full = str(row['cell_line']).upper()
        cl_short = cl_full.split('_')[0]  # e.g., A549_LUNG → A549
        ccle_cell_map[cl_full] = row
        ccle_cell_map[cl_short] = row  # Also try short name
    
    logger.info(f"CCLE cell line mapping: {len(ccle_cell_map)} entries")
    
    # Get target gene list from CCLE columns
    target_genes = [c for c in ccle_df.columns if c != 'cell_line']
    n_tgt = len(target_genes)
    logger.info(f"Target genes: {n_tgt}")
    
    # Build pKi vectors
    drug_pki = {}
    for drug, targets in KNOWN_AFFINITIES.items():
        vec = np.zeros(n_tgt, dtype=np.float32)
        for gene, pki in targets.items():
            if gene in target_genes:
                vec[target_genes.index(gene)] = pki
        drug_pki[drug.upper()] = vec
    
    # Build dataset
    drug_fps_u = {k.upper(): v for k, v in drug_fps.items()}
    cell_expr_u = {k.upper(): v for k, v in cell_expr.items()}
    
    bio_df = pd.read_csv(DATA_DIR / "cell_line_features.csv")
    cell_bio = {str(row['cell_line']).upper(): row.drop('cell_line').values.astype(np.float32)
                for _, row in bio_df.iterrows()}
    
    zero_fp = np.zeros(1024, np.float32)
    zero_expr = np.zeros(256, np.float32)
    zero_pki = np.zeros(n_tgt, np.float32)
    zero_bio = np.zeros(len(next(iter(cell_bio.values()))), np.float32)
    zero_ccle = np.zeros(n_tgt, np.float32)
    
    matched_cells = 0
    unmatched_cells = set()
    
    X, y, fold_list, pair_list, cell_list = [], [], [], [], []
    
    for _, row in syn.iterrows():
        da, db = str(row['drug_a']).upper(), str(row['drug_b']).upper()
        cl = str(row['cell_line']).upper()
        target = float(row['synergy_loewe'])
        if np.isnan(target): continue
        
        fp_a = drug_fps_u.get(da, zero_fp)
        fp_b = drug_fps_u.get(db, zero_fp)
        if np.array_equal(fp_a, zero_fp) and np.array_equal(fp_b, zero_fp): continue
        
        pki_a = drug_pki.get(da, zero_pki)
        pki_b = drug_pki.get(db, zero_pki)
        
        # Get REAL CCLE target expression
        ccle_row = ccle_cell_map.get(cl, None)
        if ccle_row is None:
            # Try common name variations
            for suffix in ['_LARGE_INTESTINE', '_BREAST', '_LUNG', '_SKIN', '_BONE',
                          '_OVARY', '_PROSTATE', '_HAEMATOPOIETIC_AND_LYMPHOID_TISSUE',
                          '_STOMACH', '_KIDNEY', '_CENTRAL_NERVOUS_SYSTEM', '_LIVER',
                          '_PANCREAS', '_ENDOMETRIUM', '_SOFT_TISSUE', '_PLEURA']:
                ccle_row = ccle_cell_map.get(cl + suffix, None)
                if ccle_row is not None:
                    break
        
        if ccle_row is not None:
            cell_tgt = np.array([float(ccle_row.get(g, 0)) for g in target_genes], dtype=np.float32)
            matched_cells += 1
        else:
            cell_tgt = zero_ccle
            unmatched_cells.add(cl)
        
        # Interaction: pKi × cell_expression
        interaction_a = pki_a * cell_tgt
        interaction_b = pki_b * cell_tgt
        
        expr = cell_expr_u.get(cl, zero_expr)
        bio = cell_bio.get(cl, zero_bio)
        
        features = np.concatenate([
            fp_a, fp_b,
            interaction_a, interaction_b,
            expr, bio,
        ])
        
        X.append(features)
        y.append(target)
        fold_list.append(int(row.get('fold', 0)))
        pair_list.append(tuple(sorted([da, db])))
        cell_list.append(cl)
    
    X = np.array(X)
    y = np.array(y)
    folds = np.array(fold_list)
    pairs = np.array(pair_list, dtype=object)
    cells = np.array(cell_list)
    
    total = len(X)
    logger.info(f"CCLE match rate: {matched_cells}/{total} ({matched_cells/total*100:.1f}%)")
    logger.info(f"Unmatched cells: {sorted(unmatched_cells)}")
    logger.info(f"Features: {X.shape}")
    
    n_fp = 1024
    groups = {
        'fp_a': (0, n_fp),
        'fp_b': (n_fp, 2*n_fp),
        'interact_a': (2*n_fp, 2*n_fp + n_tgt),
        'interact_b': (2*n_fp + n_tgt, 2*n_fp + 2*n_tgt),
        'expr': (2*n_fp + 2*n_tgt, 2*n_fp + 2*n_tgt + 256),
        'bio': (2*n_fp + 2*n_tgt + 256, X.shape[1]),
    }
    
    return X, y, folds, pairs, cells, groups, n_tgt


def run_eval(X, y, idx_fn, name=""):
    params = {
        'n_estimators': 500, 'max_depth': 8, 'learning_rate': 0.05,
        'subsample': 0.8, 'colsample_bytree': 0.3, 'min_child_weight': 10,
        'reg_alpha': 0.1, 'reg_lambda': 1.0, 'random_state': 42, 'n_jobs': -1,
    }
    
    prs = []
    for tr, te in idx_fn():
        m = xgb.XGBRegressor(**params)
        m.fit(X[tr], y[tr], verbose=False)
        yp = m.predict(X[te])
        r, _ = pearsonr(y[te], yp)
        prs.append(r)
    
    mean_r = np.mean(prs)
    print(f"    {name:45s}: r={mean_r:.4f} ± {np.std(prs):.4f}")
    return mean_r


def main():
    print("=" * 70)
    print("Cell-Drug Interaction v2 — REAL CCLE Expression")
    print("=" * 70)
    
    X, y, folds, pairs, cells, groups, n_tgt = build_features()
    
    # Define CV generators
    def pf_gen():
        for f in sorted(np.unique(folds)):
            yield ~(folds == f), folds == f
    
    def ldpo_gen():
        unique_pairs = list(set([tuple(p) for p in pairs]))
        np.random.seed(42)
        perm = np.random.permutation(len(unique_pairs))
        fold_size = len(unique_pairs) // 5
        for fold in range(5):
            start = fold * fold_size
            end = start + fold_size if fold < 4 else len(unique_pairs)
            test_set = set([unique_pairs[perm[i]] for i in range(start, end)])
            te = np.array([tuple(p) in test_set for p in pairs])
            yield ~te, te
    
    def lclo_gen():
        unique_cells = np.unique(cells)
        np.random.seed(42)
        cperm = np.random.permutation(len(unique_cells))
        cfold_size = len(unique_cells) // 5
        for fold in range(5):
            start = fold * cfold_size
            end = start + cfold_size if fold < 4 else len(unique_cells)
            test_set = set(unique_cells[cperm[start:end]])
            te = np.array([c in test_set for c in cells])
            yield ~te, te
    
    # ===== ABLATION =====
    configs = [
        ("FP only", ['fp_a', 'fp_b']),
        ("FP + Expr", ['fp_a', 'fp_b', 'expr']),
        ("FP + Interaction (REAL CCLE)", ['fp_a', 'fp_b', 'interact_a', 'interact_b']),
        ("FP + Interaction + Expr (REAL CCLE)", ['fp_a', 'fp_b', 'interact_a', 'interact_b', 'expr']),
        ("Interaction only (REAL CCLE)", ['interact_a', 'interact_b']),
        ("ALL", ['fp_a', 'fp_b', 'interact_a', 'interact_b', 'expr', 'bio']),
    ]
    
    for eval_name, gen_fn in [("Pre-defined Folds", pf_gen), ("LDPO", ldpo_gen), ("LCLO", lclo_gen)]:
        print(f"\n{'='*70}")
        print(f"{eval_name}")
        print("=" * 70)
        
        for name, keys in configs:
            col_idx = []
            for k in keys:
                s, e = groups[k]
                col_idx.extend(range(s, e))
            X_sub = X[:, col_idx]
            run_eval(X_sub, y, gen_fn, name)
    
    # ===== COMPARISON with v1 (proxy) =====
    print(f"\n{'='*70}")
    print("COMPARISON: Real CCLE vs Proxy Expression")
    print("=" * 70)
    print("  Metric        | v1 (proxy hash) | v2 (REAL CCLE)")
    print("  Pre-defined   |   FP+Expr=0.5923 FP+I+E=0.6210 |  see above")
    print("  LDPO          |   FP+Expr=0.5860 FP+I+E=0.6160 |  see above")
    print("  LCLO          |   FP+Expr=0.5032 FP+I+E=0.4699 |  see above")
    print("  (proxy LCLO hurt by -0.033, real CCLE should fix this)")
    
    # Save
    with open(MODEL_DIR / "cell_drug_interaction_v2.pkl", 'wb') as f:
        pickle.dump({'groups': groups, 'n_targets': n_tgt}, f)
    
    print(f"\n  Saved: {MODEL_DIR / 'cell_drug_interaction_v2.pkl'}")


if __name__ == "__main__":
    main()
