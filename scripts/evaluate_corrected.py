"""
Corrected Model Evaluation
============================
Fixes bugs found in previous evaluation:
1. LDPO/LCLO index alignment: skipped rows caused mismatch
2. Should use pre-defined folds from dataset (no pair leakage guaranteed)
3. Properly track which rows were kept during feature construction
"""
import pandas as pd
import numpy as np
import pickle
import json
import logging
from pathlib import Path
from scipy.stats import pearsonr, spearmanr
from sklearn.model_selection import KFold
import xgboost as xgb

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

DATA_DIR = Path("F:/ADDS/data/ml_training")
MODEL_DIR = Path("F:/ADDS/models/synergy")


def build_features_with_metadata():
    """Build feature matrix AND keep metadata aligned."""
    syn = pd.read_csv(DATA_DIR / "oneil_synergy.csv")
    
    with open(MODEL_DIR / "drug_fingerprints.pkl", 'rb') as f:
        drug_fps = pickle.load(f)
    with open(MODEL_DIR / "cell_line_expression.pkl", 'rb') as f:
        cell_expr = pickle.load(f)
    bio_df = pd.read_csv(DATA_DIR / "cell_line_features.csv")
    cell_bio = {}
    for _, row in bio_df.iterrows():
        cell_bio[row['cell_line']] = row.drop('cell_line').values.astype(np.float32)
    
    drug_fps_upper = {k.upper(): v for k, v in drug_fps.items()}
    cell_expr_upper = {k.upper(): v for k, v in cell_expr.items()}
    cell_bio_upper = {k.upper(): v for k, v in cell_bio.items()}
    
    n_fp, n_expr, n_bio = 1024, 256, len(next(iter(cell_bio.values())))
    zero_fp = np.zeros(n_fp, dtype=np.float32)
    zero_expr = np.zeros(n_expr, dtype=np.float32)
    zero_bio = np.zeros(n_bio, dtype=np.float32)
    
    X_list, y_list = [], []
    meta_list = []  # Keep drug/cell/fold info aligned
    
    for idx, row in syn.iterrows():
        da = str(row['drug_a']).upper()
        db = str(row['drug_b']).upper()
        cl = str(row['cell_line']).upper()
        target = float(row['synergy_loewe'])
        fold = int(row.get('fold', 0)) if 'fold' in row.index else 0
        
        if np.isnan(target):
            continue
        
        fp_a = drug_fps_upper.get(da, zero_fp)
        fp_b = drug_fps_upper.get(db, zero_fp)
        
        if np.array_equal(fp_a, zero_fp) and np.array_equal(fp_b, zero_fp):
            continue
        
        expr = cell_expr_upper.get(cl, zero_expr)
        bio = cell_bio_upper.get(cl, zero_bio)
        
        features = np.concatenate([fp_a, fp_b, expr, bio])
        X_list.append(features)
        y_list.append(target)
        meta_list.append({
            'drug_a': da, 'drug_b': db, 'cell_line': cl,
            'pair': tuple(sorted([da, db])),
            'fold': fold,
            'has_expr': np.sum(np.abs(expr)) > 1e-6,
        })
    
    X = np.array(X_list)
    y = np.array(y_list)
    meta = pd.DataFrame(meta_list)
    
    return X, y, meta, drug_fps


def main():
    print("=" * 70)
    print("CORRECTED MODEL EVALUATION")
    print("=" * 70)
    
    X, y, meta, drug_fps = build_features_with_metadata()
    print(f"\nData: {len(X)} samples, {X.shape[1]} features")
    print(f"Drugs: {meta['drug_a'].nunique() + meta['drug_b'].nunique()} (combined)")
    print(f"Unique pairs: {meta['pair'].nunique()}")
    print(f"Cell lines: {meta['cell_line'].nunique()}")
    print(f"Folds in data: {sorted(meta['fold'].unique())}")
    
    # Check pair-fold distribution
    pair_folds = meta.groupby('pair')['fold'].nunique()
    cross_fold_pairs = (pair_folds > 1).sum()
    print(f"Pairs crossing folds: {cross_fold_pairs}/{len(pair_folds)}")
    
    params = {
        'n_estimators': 500, 'max_depth': 8, 'learning_rate': 0.05,
        'subsample': 0.8, 'colsample_bytree': 0.3, 'min_child_weight': 10,
        'reg_alpha': 0.1, 'reg_lambda': 1.0, 'random_state': 42, 'n_jobs': -1,
    }
    
    # ================================================================
    # TEST 1: Random 5-fold CV (our reported metric)
    # ================================================================
    print(f"\n{'='*70}")
    print("TEST 1: Random 5-fold CV")
    print("=" * 70)
    
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    t1_pr, t1_sr, t1_rmse = [], [], []
    
    for fold, (tr, te) in enumerate(kf.split(X)):
        m = xgb.XGBRegressor(**params)
        m.fit(X[tr], y[tr], verbose=False)
        yp = m.predict(X[te])
        pr, _ = pearsonr(y[te], yp)
        sr, _ = spearmanr(y[te], yp)
        rmse = np.sqrt(np.mean((y[te] - yp)**2))
        t1_pr.append(pr); t1_sr.append(sr); t1_rmse.append(rmse)
        print(f"  Fold {fold+1}: r={pr:.4f}, rho={sr:.4f}, RMSE={rmse:.2f}")
    print(f"  Mean: r={np.mean(t1_pr):.4f}+/-{np.std(t1_pr):.4f}")
    
    # ================================================================
    # TEST 2: Pre-defined folds from dataset (pair-aware, no leakage)
    # ================================================================
    print(f"\n{'='*70}")
    print("TEST 2: Pre-defined Folds (pair-aware, NO pair leakage)")
    print("=" * 70)
    
    folds = sorted(meta['fold'].unique())
    t2_pr, t2_sr, t2_rmse = [], [], []
    
    for test_fold in folds:
        te_mask = (meta['fold'] == test_fold).values
        tr_mask = ~te_mask
        
        # Verify: no test pairs in train
        test_pairs = set(meta.loc[te_mask, 'pair'])
        train_pairs = set(meta.loc[tr_mask, 'pair'])
        overlap = test_pairs & train_pairs
        
        m = xgb.XGBRegressor(**params)
        m.fit(X[tr_mask], y[tr_mask], verbose=False)
        yp = m.predict(X[te_mask])
        pr, _ = pearsonr(y[te_mask], yp)
        sr, _ = spearmanr(y[te_mask], yp)
        rmse = np.sqrt(np.mean((y[te_mask] - yp)**2))
        t2_pr.append(pr); t2_sr.append(sr); t2_rmse.append(rmse)
        print(f"  Fold {test_fold}: r={pr:.4f}, rho={sr:.4f}, RMSE={rmse:.2f}, "
              f"test={te_mask.sum()}, pair_overlap={len(overlap)}")
    
    print(f"  Mean: r={np.mean(t2_pr):.4f}+/-{np.std(t2_pr):.4f}")
    
    # ================================================================
    # TEST 3: Leave-Drug-Pair-Out (completely unseen pairs)
    # ================================================================
    print(f"\n{'='*70}")
    print("TEST 3: Leave-Drug-Pair-Out (completely unseen pairs)")
    print("=" * 70)
    
    unique_pairs = meta['pair'].unique()
    np.random.seed(42)
    indices = np.random.permutation(len(unique_pairs))
    fold_size = len(unique_pairs) // 5
    
    t3_pr, t3_sr = [], []
    for fold in range(5):
        start = fold * fold_size
        end = start + fold_size if fold < 4 else len(unique_pairs)
        test_pairs = set(unique_pairs[indices[start:end]])
        
        te_mask = meta['pair'].isin(test_pairs).values
        tr_mask = ~te_mask
        
        # Double-check: no pair overlap
        assert len(set(meta.loc[te_mask, 'pair']) & set(meta.loc[tr_mask, 'pair'])) == 0
        
        m = xgb.XGBRegressor(**params)
        m.fit(X[tr_mask], y[tr_mask], verbose=False)
        yp = m.predict(X[te_mask])
        pr, _ = pearsonr(y[te_mask], yp)
        sr, _ = spearmanr(y[te_mask], yp)
        t3_pr.append(pr); t3_sr.append(sr)
        print(f"  Fold {fold+1}: r={pr:.4f}, rho={sr:.4f}, pairs={len(test_pairs)}, samples={te_mask.sum()}")
    
    print(f"  Mean: r={np.mean(t3_pr):.4f}+/-{np.std(t3_pr):.4f}")
    
    # ================================================================
    # TEST 4: Leave-Cell-Line-Out (completely unseen cell lines)
    # ================================================================
    print(f"\n{'='*70}")
    print("TEST 4: Leave-Cell-Line-Out (unseen cell lines)")
    print("=" * 70)
    
    unique_cls = meta['cell_line'].unique()
    np.random.seed(42)
    cl_indices = np.random.permutation(len(unique_cls))
    cl_fold_size = len(unique_cls) // 5
    
    t4_pr, t4_sr = [], []
    for fold in range(5):
        start = fold * cl_fold_size
        end = start + cl_fold_size if fold < 4 else len(unique_cls)
        test_cls = set(unique_cls[cl_indices[start:end]])
        
        te_mask = meta['cell_line'].isin(test_cls).values
        tr_mask = ~te_mask
        
        m = xgb.XGBRegressor(**params)
        m.fit(X[tr_mask], y[tr_mask], verbose=False)
        yp = m.predict(X[te_mask])
        pr, _ = pearsonr(y[te_mask], yp)
        sr, _ = spearmanr(y[te_mask], yp)
        t4_pr.append(pr); t4_sr.append(sr)
        print(f"  Fold {fold+1}: r={pr:.4f}, rho={sr:.4f}, cls={test_cls}, samples={te_mask.sum()}")
    
    print(f"  Mean: r={np.mean(t4_pr):.4f}+/-{np.std(t4_pr):.4f}")
    
    # ================================================================
    # TEST 5: Feature importance sanity check
    # ================================================================
    print(f"\n{'='*70}")
    print("TEST 5: Feature Importance Sanity Check")
    print("=" * 70)
    
    final = xgb.XGBRegressor(**params)
    final.fit(X, y, verbose=False)
    imp = final.feature_importances_
    
    # Split by feature group
    fp_a_imp = imp[:1024].sum()
    fp_b_imp = imp[1024:2048].sum()
    expr_imp = imp[2048:2048+256].sum()
    bio_imp = imp[2048+256:].sum()
    total_imp = imp.sum()
    
    print(f"  Drug A FP (1024): {fp_a_imp/total_imp*100:.1f}%")
    print(f"  Drug B FP (1024): {fp_b_imp/total_imp*100:.1f}%")
    print(f"  CCLE Expression (256): {expr_imp/total_imp*100:.1f}%")
    print(f"  Bio features (15): {bio_imp/total_imp*100:.1f}%")
    
    # Top 20 individual features
    feat_names = [f"fp_a_{i}" for i in range(1024)] + \
                 [f"fp_b_{i}" for i in range(1024)] + \
                 [f"expr_{i}" for i in range(256)] + \
                 [f"bio_{i}" for i in range(15)]
    
    top_idx = np.argsort(imp)[::-1][:20]
    print(f"\n  Top 20 features:")
    for rank, idx in enumerate(top_idx, 1):
        print(f"    {rank:2d}. {feat_names[idx]:15s} importance={imp[idx]:.4f}")
    
    # ================================================================
    # TEST 6: Ablation - Drug FP only vs FP+Expr vs FP+Bio vs All
    # ================================================================
    print(f"\n{'='*70}")
    print("TEST 6: Feature Ablation (using pre-defined folds)")
    print("=" * 70)
    
    configs = [
        ("Drug FP only", list(range(2048))),
        ("FP + Expression", list(range(2048 + 256))),
        ("FP + Bio only", list(range(2048)) + list(range(2048+256, 2048+256+15))),
        ("All (FP+Expr+Bio)", list(range(X.shape[1]))),
    ]
    
    for name, col_idx in configs:
        X_sub = X[:, col_idx]
        abl_pr = []
        for test_fold in folds:
            te_mask = (meta['fold'] == test_fold).values
            tr_mask = ~te_mask
            m = xgb.XGBRegressor(**params)
            m.fit(X_sub[tr_mask], y[tr_mask], verbose=False)
            yp = m.predict(X_sub[te_mask])
            pr, _ = pearsonr(y[te_mask], yp)
            abl_pr.append(pr)
        print(f"  {name:25s}: r={np.mean(abl_pr):.4f}+/-{np.std(abl_pr):.4f}")
    
    # ================================================================
    # SUMMARY
    # ================================================================
    print(f"\n{'='*70}")
    print("SUMMARY COMPARISON")
    print("=" * 70)
    print(f"  Random CV:       r = {np.mean(t1_pr):.4f} +/- {np.std(t1_pr):.4f}")
    print(f"  Pre-defined:     r = {np.mean(t2_pr):.4f} +/- {np.std(t2_pr):.4f}")
    print(f"  LDPO:            r = {np.mean(t3_pr):.4f} +/- {np.std(t3_pr):.4f}")
    print(f"  LCLO:            r = {np.mean(t4_pr):.4f} +/- {np.std(t4_pr):.4f}")
    
    print(f"\n  Random vs Pre-defined gap: {(np.mean(t1_pr) - np.mean(t2_pr))/np.mean(t1_pr)*100:+.1f}%")
    print(f"  Random vs LDPO gap:        {(np.mean(t1_pr) - np.mean(t3_pr))/np.mean(t1_pr)*100:+.1f}%")
    print(f"  Random vs LCLO gap:        {(np.mean(t1_pr) - np.mean(t4_pr))/np.mean(t1_pr)*100:+.1f}%")
    
    # Save
    results = {
        'random_cv': {'pearson': float(np.mean(t1_pr)), 'std': float(np.std(t1_pr))},
        'predefined_folds': {'pearson': float(np.mean(t2_pr)), 'std': float(np.std(t2_pr))},
        'ldpo': {'pearson': float(np.mean(t3_pr)), 'std': float(np.std(t3_pr))},
        'lclo': {'pearson': float(np.mean(t4_pr)), 'std': float(np.std(t4_pr))},
    }
    with open(DATA_DIR / "corrected_evaluation.json", 'w') as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
