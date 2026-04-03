"""
Evaluate DeepSynergy with Mechanism Features
=============================================
Compare: FP only → FP+Mechanism → FP+Mechanism+Expr → All
"""
import pandas as pd
import numpy as np
import pickle
import json
import logging
from pathlib import Path
from scipy.stats import pearsonr, spearmanr
import xgboost as xgb

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

DATA_DIR = Path("F:/ADDS/data/ml_training")
MODEL_DIR = Path("F:/ADDS/models/synergy")


def load_data_with_mechanism():
    """Load all features including mechanism."""
    syn = pd.read_csv(DATA_DIR / "oneil_synergy.csv")
    
    with open(MODEL_DIR / "drug_fingerprints.pkl", 'rb') as f:
        drug_fps = pickle.load(f)
    with open(MODEL_DIR / "cell_line_expression.pkl", 'rb') as f:
        cell_expr = pickle.load(f)
    with open(MODEL_DIR / "drug_mechanism_features.pkl", 'rb') as f:
        mech_features = pickle.load(f)
    
    bio_df = pd.read_csv(DATA_DIR / "cell_line_features.csv")
    cell_bio = {}
    for _, row in bio_df.iterrows():
        cell_bio[row['cell_line']] = row.drop('cell_line').values.astype(np.float32)
    
    drug_fps_upper = {k.upper(): v for k, v in drug_fps.items()}
    cell_expr_upper = {k.upper(): v for k, v in cell_expr.items()}
    cell_bio_upper = {k.upper(): v for k, v in cell_bio.items()}
    mech_upper = {k.upper(): v for k, v in mech_features.items()}
    
    n_fp, n_expr = 1024, 256
    n_bio = len(next(iter(cell_bio.values())))
    n_mech = len(next(iter(mech_features.values())))
    
    zero_fp = np.zeros(n_fp, dtype=np.float32)
    zero_expr = np.zeros(n_expr, dtype=np.float32)
    zero_bio = np.zeros(n_bio, dtype=np.float32)
    zero_mech = np.zeros(n_mech, dtype=np.float32)
    
    # Feature indices
    feature_groups = {
        'fp_a': (0, n_fp),
        'fp_b': (n_fp, 2*n_fp),
        'mech_a': (2*n_fp, 2*n_fp + n_mech),
        'mech_b': (2*n_fp + n_mech, 2*n_fp + 2*n_mech),
        'expr': (2*n_fp + 2*n_mech, 2*n_fp + 2*n_mech + n_expr),
        'bio': (2*n_fp + 2*n_mech + n_expr, 2*n_fp + 2*n_mech + n_expr + n_bio),
    }
    
    X_list, y_list, folds_list = [], [], []
    pairs_list = []
    
    for _, row in syn.iterrows():
        da = str(row['drug_a']).upper()
        db = str(row['drug_b']).upper()
        cl = str(row['cell_line']).upper()
        target = float(row['synergy_loewe'])
        fold = int(row.get('fold', 0))
        
        if np.isnan(target):
            continue
        
        fp_a = drug_fps_upper.get(da, zero_fp)
        fp_b = drug_fps_upper.get(db, zero_fp)
        
        if np.array_equal(fp_a, zero_fp) and np.array_equal(fp_b, zero_fp):
            continue
        
        mech_a = mech_upper.get(da, zero_mech)
        mech_b = mech_upper.get(db, zero_mech)
        expr = cell_expr_upper.get(cl, zero_expr)
        bio = cell_bio_upper.get(cl, zero_bio)
        
        features = np.concatenate([fp_a, fp_b, mech_a, mech_b, expr, bio])
        X_list.append(features)
        y_list.append(target)
        folds_list.append(fold)
        pairs_list.append(tuple(sorted([da, db])))
    
    X = np.array(X_list)
    y = np.array(y_list)
    folds = np.array(folds_list)
    pairs = np.array(pairs_list, dtype=object)
    
    logger.info(f"Data: {X.shape[0]} samples, {X.shape[1]} features")
    logger.info(f"  Feature groups: {feature_groups}")
    
    return X, y, folds, pairs, feature_groups


def run_cv(X, y, folds, name=""):
    """Run pair-aware CV with XGBoost."""
    params = {
        'n_estimators': 500, 'max_depth': 8, 'learning_rate': 0.05,
        'subsample': 0.8, 'colsample_bytree': 0.3, 'min_child_weight': 10,
        'reg_alpha': 0.1, 'reg_lambda': 1.0, 'random_state': 42, 'n_jobs': -1,
    }
    
    unique_folds = sorted(np.unique(folds))
    prs = []
    
    for test_fold in unique_folds:
        te_mask = folds == test_fold
        tr_mask = ~te_mask
        
        m = xgb.XGBRegressor(**params)
        m.fit(X[tr_mask], y[tr_mask], verbose=False)
        yp = m.predict(X[te_mask])
        pr, _ = pearsonr(y[te_mask], yp)
        prs.append(pr)
    
    mean_r = np.mean(prs)
    std_r = np.std(prs)
    print(f"  {name:40s}: r={mean_r:.4f} +/- {std_r:.4f}")
    return mean_r, std_r


def main():
    print("=" * 70)
    print("MECHANISM FEATURE ABLATION")
    print("=" * 70)
    
    X, y, folds, pairs, groups = load_data_with_mechanism()
    
    # Define feature configs
    configs = [
        ("FP only (Drug A+B)", ['fp_a', 'fp_b']),
        ("FP + Mechanism (target+affinity+pathway)", ['fp_a', 'fp_b', 'mech_a', 'mech_b']),
        ("FP + Expression", ['fp_a', 'fp_b', 'expr']),
        ("FP + Mechanism + Expression", ['fp_a', 'fp_b', 'mech_a', 'mech_b', 'expr']),
        ("Mechanism only (no FP)", ['mech_a', 'mech_b']),
        ("Mechanism + Expression (no FP)", ['mech_a', 'mech_b', 'expr']),
        ("ALL (FP+Mech+Expr+Bio)", ['fp_a', 'fp_b', 'mech_a', 'mech_b', 'expr', 'bio']),
    ]
    
    print(f"\nPre-defined Pair-Aware Folds (XGBoost):")
    results = {}
    
    for name, group_keys in configs:
        # Build column indices
        col_idx = []
        for key in group_keys:
            start, end = groups[key]
            col_idx.extend(range(start, end))
        
        X_sub = X[:, col_idx]
        mean_r, std_r = run_cv(X_sub, y, folds, name)
        results[name] = {'pearson': mean_r, 'std': std_r}
    
    # LDPO with best config
    print(f"\n{'='*70}")
    print("LDPO: Best config vs baseline")
    print("=" * 70)
    
    # FP only - LDPO
    fp_cols = list(range(groups['fp_a'][0], groups['fp_b'][1]))
    all_cols = list(range(X.shape[1]))
    
    unique_pairs = np.unique(pairs)
    np.random.seed(42)
    indices = np.random.permutation(len(unique_pairs))
    fold_size = len(unique_pairs) // 5
    
    params = {
        'n_estimators': 500, 'max_depth': 8, 'learning_rate': 0.05,
        'subsample': 0.8, 'colsample_bytree': 0.3, 'min_child_weight': 10,
        'reg_alpha': 0.1, 'reg_lambda': 1.0, 'random_state': 42, 'n_jobs': -1,
    }
    
    for col_name, col_idx in [("FP only", fp_cols), ("ALL", all_cols)]:
        X_sub = X[:, col_idx]
        ldpo_prs = []
        
        for fold in range(5):
            start = fold * fold_size
            end = start + fold_size if fold < 4 else len(unique_pairs)
            test_pair_set = set(unique_pairs[indices[start:end]])
            
            te_mask = np.array([p in test_pair_set for p in pairs])
            tr_mask = ~te_mask
            
            m = xgb.XGBRegressor(**params)
            m.fit(X_sub[tr_mask], y[tr_mask], verbose=False)
            yp = m.predict(X_sub[te_mask])
            pr, _ = pearsonr(y[te_mask], yp)
            ldpo_prs.append(pr)
        
        print(f"  LDPO {col_name:40s}: r={np.mean(ldpo_prs):.4f} +/- {np.std(ldpo_prs):.4f}")
    
    # Save results
    with open(DATA_DIR / "mechanism_ablation_results.json", 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\nResults saved: {DATA_DIR / 'mechanism_ablation_results.json'}")


if __name__ == "__main__":
    main()
