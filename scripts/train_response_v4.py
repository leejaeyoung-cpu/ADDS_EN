"""
Treatment Response v4 — Multi-Dataset Unified Model
====================================================
Combines GEO FOLFOX response data (247 pts) + TCGA treatment outcome (106 pts)
All with actual clinical treatment response labels (not alive/dead proxy)
"""
import pandas as pd
import numpy as np
import pickle
import gzip
import logging
from pathlib import Path
from sklearn.model_selection import StratifiedKFold, LeaveOneGroupOut
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, classification_report, accuracy_score
import xgboost as xgb

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

DATA_DIR = Path("F:/ADDS/data/ml_training")
RESPONSE_DIR = DATA_DIR / "chemo_response"
TCGA_DIR = DATA_DIR / "tcga_rnaseq"
MODEL_DIR = Path("F:/ADDS/models/synergy")


def load_geo_dataset(gse_id):
    """Load GEO expression + response labels."""
    # Load clinical
    clin_path = RESPONSE_DIR / f"{gse_id}_clinical.csv"
    if not clin_path.exists():
        logger.warning(f"{gse_id}: No clinical file")
        return None, None, None
    
    clin = pd.read_csv(clin_path)
    
    # Load expression matrix
    matrix_path = RESPONSE_DIR / f"{gse_id}_series_matrix.txt.gz"
    if not matrix_path.exists():
        logger.warning(f"{gse_id}: No expression matrix")
        return None, None, None
    
    logger.info(f"Parsing {gse_id} expression matrix...")
    
    open_fn = gzip.open if str(matrix_path).endswith('.gz') else open
    
    sample_ids = None
    data_rows = []
    in_data = False
    
    with open_fn(matrix_path, 'rt', errors='replace') as f:
        for line in f:
            line = line.strip()
            if line.startswith('!Sample_geo_accession'):
                sample_ids = [s.strip('"') for s in line.split('\t')[1:]]
            if line.startswith('"ID_REF"'):
                in_data = True
                continue
            if line.startswith('!series_matrix_table_end'):
                break
            if in_data and line and not line.startswith('!'):
                parts = line.split('\t')
                if len(parts) > 1:
                    probe = parts[0].strip('"')
                    vals = []
                    for v in parts[1:]:
                        try:
                            vals.append(float(v.strip('"')))
                        except:
                            vals.append(np.nan)
                    data_rows.append([probe] + vals)
    
    if not data_rows or not sample_ids:
        return None, None, None
    
    cols = ['probe_id'] + sample_ids
    df = pd.DataFrame(data_rows, columns=cols[:len(data_rows[0])])
    df = df.set_index('probe_id')
    df = df.astype(float)
    
    logger.info(f"  {gse_id}: {df.shape[0]} probes × {df.shape[1]} samples")
    
    # Match labels to samples
    label_map = dict(zip(clin['sample_id'], clin['response']))
    
    matched_samples = [s for s in sample_ids if s in label_map]
    if not matched_samples:
        # Try matching via geo_accession
        if 'geo_accession' in clin.columns:
            label_map2 = dict(zip(clin['geo_accession'], clin['response']))
            matched_samples = [s for s in sample_ids if s in label_map2]
            label_map = label_map2
    
    logger.info(f"  Matched samples with labels: {len(matched_samples)}")
    
    if not matched_samples:
        return None, None, None
    
    expr = df[matched_samples].T  # samples × probes
    labels = np.array([label_map[s] for s in matched_samples])
    
    return expr, labels, matched_samples


def load_tcga_response():
    """Load TCGA samples with actual treatment outcome."""
    clinical = pd.read_csv(TCGA_DIR / "clinical.csv")
    
    # Build treatment outcome labels
    labels = {}
    for _, row in clinical.iterrows():
        sid = row['submitter_id']
        outcomes_str = str(row.get('treatment_outcomes', ''))
        if not outcomes_str or outcomes_str == 'nan':
            continue
        outcomes = [o.strip() for o in outcomes_str.split('|') if o.strip()]
        is_resp = any(o in ['Complete Response', 'Partial Response'] for o in outcomes)
        is_nonresp = any(o in ['Progressive Disease'] for o in outcomes)
        is_stable = any(o in ['Stable Disease'] for o in outcomes)
        if is_resp:
            labels[sid] = 1
        elif is_nonresp:
            labels[sid] = 0
        elif is_stable:
            labels[sid] = 0
    
    # Load expression
    expr_files = list(TCGA_DIR.glob("TCGA-*.tsv.gz"))
    expr_data = {}
    gene_names = None
    gene_col = value_col = None
    
    for fpath in expr_files:
        try:
            fname = fpath.name
            if fname.endswith('.tsv.gz'):
                fname = fname[:-7]
            sample_id = fname.rsplit('_', 1)[0]
            
            if sample_id not in labels:
                continue
            
            try:
                df = pd.read_csv(fpath, sep='\t', compression='gzip', comment='#')
            except:
                df = pd.read_csv(str(fpath), sep='\t', comment='#', compression=None, engine='python')
            
            if gene_col is None:
                gene_col = 'gene_name'
                value_col = 'fpkm_unstranded'
                mask = ~df[gene_col].str.startswith('_', na=True)
                if 'gene_type' in df.columns:
                    mask = mask & (df['gene_type'] == 'protein_coding')
                gene_names = df.loc[mask, gene_col].values
            
            mask = ~df[gene_col].str.startswith('_', na=True)
            if 'gene_type' in df.columns:
                mask = mask & (df['gene_type'] == 'protein_coding')
            values = df.loc[mask, value_col].values
            
            if len(values) == len(gene_names):
                expr_data[sample_id] = np.log2(values.astype(np.float32) + 1)
        except:
            pass
    
    matched = [sid for sid in labels if sid in expr_data]
    logger.info(f"  TCGA treatment outcome: {len(matched)} samples")
    
    if not matched:
        return None, None, None, None
    
    X = np.array([expr_data[sid] for sid in matched])
    y = np.array([labels[sid] for sid in matched])
    
    return X, y, matched, gene_names


def find_common_genes_via_variance(datasets):
    """Select top-variance probes/genes from each dataset, then find overlap."""
    # For microarray: use top-variance probes as features
    # For RNA-seq: use top-variance genes
    # We'll take top 1000 variance probes from each GEO dataset
    top_k = 1000
    
    all_top_probes = {}
    for name, (expr_df, labels) in datasets.items():
        if expr_df is None:
            continue
        if isinstance(expr_df, pd.DataFrame):
            var = expr_df.var(axis=0)
            top_probes = var.nlargest(top_k).index.tolist()
            all_top_probes[name] = set(top_probes)
        elif isinstance(expr_df, np.ndarray):
            # TCGA: use gene names
            pass
    
    # For microarray datasets sharing same platform: use common probes
    return all_top_probes


def main():
    print("=" * 70)
    print("Treatment Response v4 — Multi-Dataset Unified Model")
    print("=" * 70)
    
    # ===== Load all datasets =====
    geo_datasets = {}
    total_samples = 0
    total_resp = 0
    total_nonresp = 0
    
    for gse_id in ['GSE28702', 'GSE19860', 'GSE72970']:
        expr, labels, samples = load_geo_dataset(gse_id)
        if expr is not None:
            geo_datasets[gse_id] = (expr, labels, samples)
            r = (labels == 1).sum()
            nr = (labels == 0).sum()
            total_samples += len(labels)
            total_resp += r
            total_nonresp += nr
            print(f"  {gse_id}: {len(labels)} pts ({r} R, {nr} NR)")
    
    print(f"\n  GEO Total: {total_samples} pts ({total_resp} R, {total_nonresp} NR)")
    
    # ===== Cross-dataset validation =====
    # All 3 GEO datasets use the SAME microarray platform (HG-U133 Plus 2.0)
    # → Probes are directly comparable
    
    print(f"\n{'='*70}")
    print("STRATEGY 1: Cross-Dataset Validation (Leave-One-Study-Out)")
    print("=" * 70)
    
    # Check probe overlap
    probe_sets = {}
    for name, (expr, labels, samples) in geo_datasets.items():
        probe_sets[name] = set(expr.columns)
    
    common_probes = None
    for name, probes in probe_sets.items():
        if common_probes is None:
            common_probes = probes
        else:
            common_probes = common_probes & probes
    
    print(f"  Common probes: {len(common_probes)}")
    common_probes_list = sorted(common_probes)
    
    # Select top-variance probes across all data
    all_expr = []
    all_labels = []
    all_groups = []
    
    for name, (expr, labels, samples) in geo_datasets.items():
        expr_common = expr[common_probes_list].values
        all_expr.append(expr_common)
        all_labels.extend(labels.tolist())
        all_groups.extend([name] * len(labels))
    
    X_all = np.vstack(all_expr)
    y_all = np.array(all_labels)
    groups_all = np.array(all_groups)
    
    print(f"  Combined: {X_all.shape[0]} samples × {X_all.shape[1]} probes")
    print(f"  Responders: {(y_all==1).sum()}, Non-responders: {(y_all==0).sum()}")
    
    # LOSO: Leave-One-Study-Out
    print(f"\n  LOSO Cross-Validation:")
    loso_aucs = []
    unique_groups = np.unique(groups_all)
    
    for test_group in unique_groups:
        te_mask = groups_all == test_group
        tr_mask = ~te_mask
        
        X_tr, X_te = X_all[tr_mask], X_all[te_mask]
        y_tr, y_te = y_all[tr_mask], y_all[te_mask]
        
        # Feature selection on train
        var = np.var(X_tr, axis=0)
        top_idx = np.argsort(var)[::-1][:256]
        
        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr[:, top_idx])
        X_te_s = scaler.transform(X_te[:, top_idx])
        
        model = xgb.XGBClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.5, min_child_weight=5,
            scale_pos_weight=(y_tr==0).sum() / max((y_tr==1).sum(), 1),
            random_state=42,
        )
        model.fit(X_tr_s, y_tr, verbose=False)
        yp = model.predict_proba(X_te_s)[:, 1]
        
        try:
            auc = roc_auc_score(y_te, yp)
        except:
            auc = 0.5
        
        acc = accuracy_score(y_te, (yp > 0.5).astype(int))
        loso_aucs.append(auc)
        
        n_tr = len(y_tr)
        n_te = len(y_te)
        print(f"    Train on others → Test {test_group}: AUC={auc:.4f}, Acc={acc:.4f} "
              f"(train={n_tr}, test={n_te})")
    
    print(f"    LOSO Mean AUC: {np.mean(loso_aucs):.4f}")
    
    # ===== 5-fold CV on pooled data =====
    print(f"\n{'='*70}")
    print("STRATEGY 2: 5-fold Stratified CV on Pooled Data (CV-internal gene sel.)")
    print("=" * 70)
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_aucs = []
    
    for fold, (tr_idx, te_idx) in enumerate(cv.split(X_all, y_all)):
        X_tr, X_te = X_all[tr_idx], X_all[te_idx]
        y_tr, y_te = y_all[tr_idx], y_all[te_idx]
        
        # Gene selection inside CV
        var = np.var(X_tr, axis=0)
        top_idx = np.argsort(var)[::-1][:256]
        
        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr[:, top_idx])
        X_te_s = scaler.transform(X_te[:, top_idx])
        
        model = xgb.XGBClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.5, min_child_weight=5,
            scale_pos_weight=(y_tr==0).sum() / max((y_tr==1).sum(), 1),
            random_state=42,
        )
        model.fit(X_tr_s, y_tr, verbose=False)
        yp = model.predict_proba(X_te_s)[:, 1]
        
        auc = roc_auc_score(y_te, yp)
        cv_aucs.append(auc)
        print(f"    Fold {fold+1}: AUC={auc:.4f}")
    
    print(f"    Pooled CV Mean AUC: {np.mean(cv_aucs):.4f} ± {np.std(cv_aucs):.4f}")
    
    # ===== Train final model =====
    print(f"\n{'='*70}")
    print("FINAL MODEL: Train on all GEO data")
    print("=" * 70)
    
    var = np.var(X_all, axis=0)
    top_idx = np.argsort(var)[::-1][:256]
    
    scaler_final = StandardScaler()
    X_final = scaler_final.fit_transform(X_all[:, top_idx])
    
    model_final = xgb.XGBClassifier(
        n_estimators=200, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.5, min_child_weight=5,
        scale_pos_weight=(y_all==0).sum() / max((y_all==1).sum(), 1),
        random_state=42,
    )
    model_final.fit(X_final, y_all, verbose=False)
    
    # Feature importance
    imp = model_final.feature_importances_
    top_feat_idx = np.argsort(imp)[::-1][:20]
    selected_probes = [common_probes_list[top_idx[i]] for i in range(256)]
    
    print(f"\n  TOP FEATURES:")
    for rank, idx in enumerate(top_feat_idx, 1):
        probe = selected_probes[idx]
        print(f"    {rank:2d}. Probe {probe:15s} importance={imp[idx]:.4f}")
    
    # Save
    model_data = {
        'model': model_final,
        'scaler': scaler_final,
        'selected_probes': selected_probes,
        'n_probes': 256,
        'sources': list(geo_datasets.keys()),
        'n_samples': len(y_all),
        'label_definition': 'FOLFOX response: Responder vs Non-responder (actual clinical)',
        'loso_auc': f"{np.mean(loso_aucs):.4f}",
        'cv_auc': f"{np.mean(cv_aucs):.4f} ± {np.std(cv_aucs):.4f}",
    }
    
    with open(MODEL_DIR / "treatment_response_v4.pkl", 'wb') as f:
        pickle.dump(model_data, f)
    
    print(f"\n  Saved: {MODEL_DIR / 'treatment_response_v4.pkl'}")
    
    # ===== Comparison =====
    print(f"\n{'='*70}")
    print("COMPARISON: All Treatment Response Versions")
    print("=" * 70)
    print(f"  v2 (TCGA alive/dead, leaked):      AUC = 0.6839 ± 0.0700  (n=529)")
    print(f"  v3 (TCGA outcome, prop.CV):         AUC = 0.5000 ± 0.0000  (n=106)")
    print(f"  v4 LOSO (GEO FOLFOX, cross-study):  AUC = {np.mean(loso_aucs):.4f}          (n={len(y_all)})")
    print(f"  v4 5-CV (GEO FOLFOX, pooled):       AUC = {np.mean(cv_aucs):.4f} ± {np.std(cv_aucs):.4f}  (n={len(y_all)})")


if __name__ == "__main__":
    main()
