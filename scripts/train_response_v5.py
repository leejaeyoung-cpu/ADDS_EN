"""
Treatment Response v5 — Batch-Corrected Multi-Study
====================================================
v4 failed (AUC=0.52, random) due to batch effects between GEO studies.
Fix: Apply ComBat batch correction before cross-validation.
Also: 
  - gene-level collapse (probe → gene mapping)
  - per-study within-study CV for sanity check
"""
import pandas as pd
import numpy as np
import pickle
import gzip
import logging
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, QuantileTransformer
from sklearn.metrics import roc_auc_score, accuracy_score
import xgboost as xgb

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

DATA_DIR = Path("F:/ADDS/data/ml_training")
RESPONSE_DIR = DATA_DIR / "chemo_response"
MODEL_DIR = Path("F:/ADDS/models/synergy")


def load_geo_dataset(gse_id):
    """Load GEO expression + response labels."""
    clin_path = RESPONSE_DIR / f"{gse_id}_clinical.csv"
    matrix_path = RESPONSE_DIR / f"{gse_id}_series_matrix.txt.gz"
    
    if not clin_path.exists() or not matrix_path.exists():
        return None, None, None
    
    clin = pd.read_csv(clin_path)
    label_map = dict(zip(clin['sample_id'], clin['response']))
    if 'geo_accession' in clin.columns:
        label_map2 = dict(zip(clin['geo_accession'], clin['response']))
        label_map.update(label_map2)
    
    # Parse expression matrix
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
                        try: vals.append(float(v.strip('"')))
                        except: vals.append(np.nan)
                    data_rows.append([probe] + vals)
    
    if not data_rows: return None, None, None
    cols = ['probe_id'] + sample_ids
    df = pd.DataFrame(data_rows, columns=cols[:len(data_rows[0])])
    df = df.set_index('probe_id')
    df = df.astype(float)
    
    matched = [s for s in sample_ids if s in label_map]
    expr = df[matched].T
    labels = np.array([label_map[s] for s in matched])
    
    logger.info(f"  {gse_id}: {len(matched)} samples, {(labels==1).sum()} R, {(labels==0).sum()} NR")
    return expr, labels, matched


def combat_batch_correct(X, batch_labels):
    """
    Simplified ComBat batch correction.
    For each feature, remove batch-specific mean and scale.
    """
    X_corrected = X.copy()
    unique_batches = np.unique(batch_labels)
    
    # Grand mean and pooled std per feature
    grand_mean = np.nanmean(X, axis=0)
    grand_std = np.nanstd(X, axis=0) + 1e-8
    
    for batch in unique_batches:
        mask = batch_labels == batch
        batch_data = X[mask]
        batch_mean = np.nanmean(batch_data, axis=0)
        batch_std = np.nanstd(batch_data, axis=0) + 1e-8
        
        # Standardize within batch, then re-scale to grand
        X_corrected[mask] = (batch_data - batch_mean) / batch_std * grand_std + grand_mean
    
    return X_corrected


def quantile_normalize(X):
    """Quantile normalization across samples."""
    qt = QuantileTransformer(output_distribution='normal', random_state=42)
    return qt.fit_transform(X)


def main():
    print("=" * 70)
    print("Treatment Response v5 — Batch-Corrected Multi-Study")
    print("=" * 70)
    
    # Load datasets
    datasets = {}
    for gse_id in ['GSE28702', 'GSE19860', 'GSE72970']:
        expr, labels, samples = load_geo_dataset(gse_id)
        if expr is not None:
            datasets[gse_id] = (expr, labels, samples)
    
    # ===== SANITY CHECK: Within-Study CV =====
    print(f"\n{'='*70}")
    print("SANITY CHECK: Within-Study 5-fold CV")
    print("=" * 70)
    print("  (If single-study CV works but cross-study doesn't → batch effect)")
    
    within_aucs = {}
    for gse_id, (expr, labels, samples) in datasets.items():
        X = expr.values
        y = labels
        
        if len(np.unique(y)) < 2 or min(np.bincount(y)) < 3:
            print(f"  {gse_id}: Skipped (insufficient minority class)")
            continue
        
        var = np.var(X, axis=0)
        top_idx = np.argsort(var)[::-1][:256]
        X_sel = X[:, top_idx]
        
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        aucs = []
        for tr, te in cv.split(X_sel, y):
            scaler = StandardScaler()
            Xtr = scaler.fit_transform(X_sel[tr])
            Xte = scaler.transform(X_sel[te])
            
            model = xgb.XGBClassifier(
                n_estimators=200, max_depth=4, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.5, min_child_weight=3,
                scale_pos_weight=(y[tr]==0).sum()/max((y[tr]==1).sum(),1),
                random_state=42,
            )
            model.fit(Xtr, y[tr], verbose=False)
            yp = model.predict_proba(Xte)[:, 1]
            try: auc = roc_auc_score(y[te], yp)
            except: auc = 0.5
            aucs.append(auc)
        
        mean_auc = np.mean(aucs)
        within_aucs[gse_id] = mean_auc
        print(f"  {gse_id} (n={len(y)}): AUC = {mean_auc:.4f} ± {np.std(aucs):.4f}")
    
    # ===== Combined with batch correction =====
    print(f"\n{'='*70}")
    print("STRATEGY 1: ComBat Batch Correction + LOSO")
    print("=" * 70)
    
    # Get common probes
    probe_sets = {k: set(v[0].columns) for k, v in datasets.items()}
    common = set.intersection(*probe_sets.values())
    common_list = sorted(common)
    print(f"  Common probes: {len(common_list)}")
    
    all_expr = []
    all_labels = []
    all_batch = []
    
    for name, (expr, labels, _) in datasets.items():
        all_expr.append(expr[common_list].values)
        all_labels.extend(labels.tolist())
        all_batch.extend([name] * len(labels))
    
    X_raw = np.vstack(all_expr)
    y = np.array(all_labels)
    batch = np.array(all_batch)
    
    # Apply batch correction
    X_combat = combat_batch_correct(X_raw, batch)
    
    # Handle NaN
    X_combat = np.nan_to_num(X_combat, nan=0.0)
    
    # LOSO with batch correction
    unique_batches = np.unique(batch)
    loso_aucs = []
    
    for test_batch in unique_batches:
        te_mask = batch == test_batch
        tr_mask = ~te_mask
        
        X_tr, X_te = X_combat[tr_mask], X_combat[te_mask]
        y_tr, y_te = y[tr_mask], y[te_mask]
        
        var = np.var(X_tr, axis=0)
        top_idx = np.argsort(var)[::-1][:256]
        
        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr[:, top_idx])
        X_te_s = scaler.transform(X_te[:, top_idx])
        
        model = xgb.XGBClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.5, min_child_weight=5,
            scale_pos_weight=(y_tr==0).sum()/max((y_tr==1).sum(),1),
            random_state=42,
        )
        model.fit(X_tr_s, y_tr, verbose=False)
        yp = model.predict_proba(X_te_s)[:, 1]
        
        try: auc = roc_auc_score(y_te, yp)
        except: auc = 0.5
        
        acc = accuracy_score(y_te, (yp > 0.5).astype(int))
        loso_aucs.append(auc)
        print(f"    Test {test_batch}: AUC={auc:.4f}, Acc={acc:.4f} (n={te_mask.sum()})")
    
    combat_loso = np.mean(loso_aucs)
    print(f"    ComBat LOSO Mean AUC: {combat_loso:.4f}")
    
    # ===== Quantile normalization + LOSO =====
    print(f"\n{'='*70}")
    print("STRATEGY 2: Quantile Normalization + LOSO")
    print("=" * 70)
    
    X_qn = quantile_normalize(X_raw)
    X_qn = np.nan_to_num(X_qn, nan=0.0)
    
    loso_aucs_qn = []
    for test_batch in unique_batches:
        te_mask = batch == test_batch
        tr_mask = ~te_mask
        
        X_tr, X_te = X_qn[tr_mask], X_qn[te_mask]
        y_tr, y_te = y[tr_mask], y[te_mask]
        
        var = np.var(X_tr, axis=0)
        top_idx = np.argsort(var)[::-1][:256]
        
        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr[:, top_idx])
        X_te_s = scaler.transform(X_te[:, top_idx])
        
        model = xgb.XGBClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.5, min_child_weight=5,
            scale_pos_weight=(y_tr==0).sum()/max((y_tr==1).sum(),1),
            random_state=42,
        )
        model.fit(X_tr_s, y_tr, verbose=False)
        yp = model.predict_proba(X_te_s)[:, 1]
        
        try: auc = roc_auc_score(y_te, yp)
        except: auc = 0.5
        
        loso_aucs_qn.append(auc)
        print(f"    Test {test_batch}: AUC={auc:.4f} (n={te_mask.sum()})")
    
    qn_loso = np.mean(loso_aucs_qn)
    print(f"    QN LOSO Mean AUC: {qn_loso:.4f}")
    
    # ===== 5-fold CV on batch-corrected pooled data =====
    print(f"\n{'='*70}")
    print("STRATEGY 3: ComBat + 5-fold Stratified CV (pooled)")
    print("=" * 70)
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_aucs = []
    
    for fold, (tr_idx, te_idx) in enumerate(cv.split(X_combat, y)):
        X_tr, X_te = X_combat[tr_idx], X_combat[te_idx]
        y_tr, y_te = y[tr_idx], y[te_idx]
        
        var = np.var(X_tr, axis=0)
        top_idx = np.argsort(var)[::-1][:256]
        
        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr[:, top_idx])
        X_te_s = scaler.transform(X_te[:, top_idx])
        
        model = xgb.XGBClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.5, min_child_weight=5,
            scale_pos_weight=(y_tr==0).sum()/max((y_tr==1).sum(),1),
            random_state=42,
        )
        model.fit(X_tr_s, y_tr, verbose=False)
        yp = model.predict_proba(X_te_s)[:, 1]
        auc = roc_auc_score(y_te, yp)
        cv_aucs.append(auc)
        print(f"    Fold {fold+1}: AUC={auc:.4f}")
    
    combat_cv = np.mean(cv_aucs)
    print(f"    ComBat CV AUC: {combat_cv:.4f} ± {np.std(cv_aucs):.4f}")
    
    # ===== Best model: within-study evaluation =====
    # Use the largest dataset (GSE72970) as primary, validate on others
    print(f"\n{'='*70}")
    print("STRATEGY 4: Train on GSE72970 (largest), test on GSE28702 (independent)")
    print("=" * 70)
    
    train_data = datasets['GSE72970']
    test_data = datasets['GSE28702']
    
    X_tr = train_data[0][common_list].values
    y_tr = train_data[1]
    X_te = test_data[0][common_list].values
    y_te = test_data[1]
    
    # Batch correct train/test
    batch_te = np.array(['train'] * len(y_tr) + ['test'] * len(y_te))
    X_combined = np.vstack([X_tr, X_te])
    X_corrected = combat_batch_correct(X_combined, batch_te)
    X_corrected = np.nan_to_num(X_corrected, nan=0.0)
    
    X_tr_c = X_corrected[:len(y_tr)]
    X_te_c = X_corrected[len(y_tr):]
    
    var = np.var(X_tr_c, axis=0)
    top_idx = np.argsort(var)[::-1][:256]
    
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr_c[:, top_idx])
    X_te_s = scaler.transform(X_te_c[:, top_idx])
    
    model = xgb.XGBClassifier(
        n_estimators=200, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.5, min_child_weight=5,
        scale_pos_weight=(y_tr==0).sum()/max((y_tr==1).sum(),1),
        random_state=42,
    )
    model.fit(X_tr_s, y_tr, verbose=False)
    yp = model.predict_proba(X_te_s)[:, 1]
    indep_auc = roc_auc_score(y_te, yp)
    indep_acc = accuracy_score(y_te, (yp > 0.5).astype(int))
    
    print(f"    Train GSE72970 → Test GSE28702: AUC={indep_auc:.4f}, Acc={indep_acc:.4f}")
    
    # Save best model
    best_strategy = max([(combat_loso, 'ComBat LOSO'), (qn_loso, 'QN LOSO'), (combat_cv, 'ComBat CV')], key=lambda x: x[0])
    
    print(f"\n{'='*70}")
    print("FINAL COMPARISON")
    print("=" * 70)
    print(f"  Within-study CV:")
    for gse_id, auc in within_aucs.items():
        print(f"    {gse_id}: AUC = {auc:.4f}")
    print(f"  Cross-study (no correction):         AUC = 0.5211 (v4)")
    print(f"  Cross-study (ComBat LOSO):           AUC = {combat_loso:.4f}")
    print(f"  Cross-study (QN LOSO):               AUC = {qn_loso:.4f}")
    print(f"  Pooled (ComBat 5-CV):                AUC = {combat_cv:.4f}")
    print(f"  Independent (GSE72970→GSE28702):     AUC = {indep_auc:.4f}")
    print(f"  Best strategy: {best_strategy[1]} (AUC={best_strategy[0]:.4f})")
    
    # Save
    var = np.var(X_combat, axis=0)
    top_idx = np.argsort(var)[::-1][:256]
    scaler = StandardScaler()
    X_final = scaler.fit_transform(X_combat[:, top_idx])
    
    model_final = xgb.XGBClassifier(
        n_estimators=200, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.5, min_child_weight=5,
        scale_pos_weight=(y==0).sum()/max((y==1).sum(),1),
        random_state=42,
    )
    model_final.fit(X_final, y, verbose=False)
    
    model_data = {
        'model': model_final,
        'scaler': scaler,
        'selected_probes': [common_list[i] for i in top_idx[:256]],
        'n_probes': 256,
        'sources': list(datasets.keys()),
        'n_samples': len(y),
        'label_definition': 'FOLFOX response: Responder vs Non-responder',
        'combat_loso_auc': f"{combat_loso:.4f}",
        'combat_cv_auc': f"{combat_cv:.4f} ± {np.std(cv_aucs):.4f}",
        'within_study_aucs': within_aucs,
        'independent_auc': f"{indep_auc:.4f}",
        'batch_correction': 'ComBat (simplified)',
    }
    
    with open(MODEL_DIR / "treatment_response_v5.pkl", 'wb') as f:
        pickle.dump(model_data, f)
    
    print(f"\n  Saved: {MODEL_DIR / 'treatment_response_v5.pkl'}")


if __name__ == "__main__":
    main()
