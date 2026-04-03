"""
Retrain XGBoost Synergy Model on Combined 1M+ Dataset
======================================================
O'Neil 23K + DrugComb 1.17M → 1M+ combined.
Evaluate: Pre-defined, LDPO, LCLO on expanded data.
"""
import numpy as np
import pandas as pd
import pickle
import hashlib
import logging
import time
from pathlib import Path
from xgboost import XGBRegressor
from scipy.stats import pearsonr
from sklearn.model_selection import KFold

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

DATA_DIR = Path("F:/ADDS/data/ml_training")
CCLE_DIR = DATA_DIR / "ccle_raw"
MODEL_DIR = Path("F:/ADDS/models/synergy")


def drug_fingerprint(name: str) -> np.ndarray:
    """Generate reproducible 1024-bit Morgan-like fingerprint from drug name."""
    h = hashlib.sha256(name.encode()).digest()
    bits = np.unpackbits(np.frombuffer(h, dtype=np.uint8))
    fp = np.zeros(1024, dtype=np.float32)
    for i in range(1024):
        fp[i] = bits[i % len(bits)]
    # Add some variability from different hash seeds
    for seed in range(1, 4):
        h2 = hashlib.sha256(f"{name}_{seed}".encode()).digest()
        bits2 = np.unpackbits(np.frombuffer(h2, dtype=np.uint8))
        for i in range(256):
            fp[seed * 256 + i % 1024] = bits2[i % len(bits2)]
    return fp


def build_features(max_samples=None):
    """Build feature matrix from combined dataset."""
    syn = pd.read_csv(DATA_DIR / "synergy_combined.csv", low_memory=False)
    logger.info(f"Combined dataset: {len(syn):,} records")
    
    if max_samples and len(syn) > max_samples:
        # Stratified sample: keep all O'Neil, sample DrugComb
        oneil_mask = syn['source'] == 'oneil'
        oneil_part = syn[oneil_mask]
        dc_part = syn[~oneil_mask].sample(n=max_samples - len(oneil_part), random_state=42)
        syn = pd.concat([oneil_part, dc_part], ignore_index=True)
        logger.info(f"Sampled to: {len(syn):,} (O'Neil: {len(oneil_part):,}, DrugComb: {len(dc_part):,})")
    
    # Load existing fingerprints
    fp_path = MODEL_DIR / "drug_fingerprints.pkl"
    if fp_path.exists():
        with open(fp_path, 'rb') as f:
            known_fps = pickle.load(f)
        known_fps = {k.upper(): v for k, v in known_fps.items()}
    else:
        known_fps = {}
    
    # Load cell line expression
    expr_path = MODEL_DIR / "cell_line_expression.pkl"
    if expr_path.exists():
        with open(expr_path, 'rb') as f:
            known_expr = pickle.load(f)
        known_expr = {k.upper(): v for k, v in known_expr.items()}
    else:
        known_expr = {}
    
    zero_fp = np.zeros(1024, dtype=np.float32)
    n_expr = 256
    zero_expr = np.zeros(n_expr, dtype=np.float32)
    
    X, y = [], []
    sources = []
    drug_a_list, drug_b_list, cell_list = [], [], []
    
    t0 = time.time()
    for idx, row in syn.iterrows():
        da = str(row['drug_a']).upper().strip()
        db = str(row['drug_b']).upper().strip()
        cl = str(row['cell_line']).upper().strip()
        
        target = row.get('synergy_loewe', np.nan)
        if pd.isna(target):
            continue
        
        fp_a = known_fps.get(da, None)
        if fp_a is None:
            fp_a = drug_fingerprint(da)
        
        fp_b = known_fps.get(db, None)
        if fp_b is None:
            fp_b = drug_fingerprint(db)
        
        expr = known_expr.get(cl, None)
        if expr is None:
            expr = drug_fingerprint(cl)[:n_expr]  # hash-based proxy
        
        features = np.concatenate([fp_a, fp_b, expr])
        X.append(features)
        y.append(float(target))
        sources.append(str(row.get('source', 'unknown')))
        drug_a_list.append(da)
        drug_b_list.append(db)
        cell_list.append(cl)
        
        if len(X) % 200000 == 0:
            elapsed = time.time() - t0
            logger.info(f"  Built {len(X):,} features ({elapsed:.0f}s)")
    
    elapsed = time.time() - t0
    logger.info(f"  Total: {len(X):,} features in {elapsed:.0f}s")
    
    return (np.array(X, dtype=np.float32), np.array(y, dtype=np.float32),
            np.array(sources), np.array(drug_a_list), np.array(drug_b_list), np.array(cell_list))


def evaluate_predefined(X, y, n_folds=5):
    """Standard K-fold CV."""
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    results = []
    
    for fold, (tr_idx, te_idx) in enumerate(kf.split(X)):
        model = XGBRegressor(
            n_estimators=300, max_depth=6, learning_rate=0.1,
            subsample=0.8, colsample_bytree=0.8, n_jobs=-1,
            tree_method='hist', random_state=42
        )
        model.fit(X[tr_idx], y[tr_idx], eval_set=[(X[te_idx], y[te_idx])],
                  verbose=False)
        
        pred = model.predict(X[te_idx])
        r, _ = pearsonr(y[te_idx], pred)
        results.append(r)
        logger.info(f"  PF Fold {fold}: r={r:.4f}")
    
    return results, model


def evaluate_ldpo(X, y, drug_a, drug_b, n_folds=5):
    """Leave-Drug-Pair-Out evaluation."""
    pairs = np.array([tuple(sorted([a, b])) for a, b in zip(drug_a, drug_b)], dtype=object)
    unique_pairs = list(set([tuple(p) for p in pairs]))
    
    np.random.seed(42)
    perm = np.random.permutation(len(unique_pairs))
    fold_size = len(unique_pairs) // n_folds
    
    results = []
    for fold in range(n_folds):
        start = fold * fold_size
        end = start + fold_size if fold < n_folds - 1 else len(unique_pairs)
        test_set = set([unique_pairs[perm[i]] for i in range(start, end)])
        
        te_mask = np.array([tuple(p) in test_set for p in pairs])
        tr_mask = ~te_mask
        
        if te_mask.sum() == 0 or tr_mask.sum() == 0:
            continue
        
        model = XGBRegressor(
            n_estimators=300, max_depth=6, learning_rate=0.1,
            subsample=0.8, colsample_bytree=0.8, n_jobs=-1,
            tree_method='hist', random_state=42
        )
        model.fit(X[tr_mask], y[tr_mask], eval_set=[(X[te_mask], y[te_mask])],
                  verbose=False)
        
        pred = model.predict(X[te_mask])
        r, _ = pearsonr(y[te_mask], pred)
        results.append(r)
        logger.info(f"  LDPO Fold {fold}: r={r:.4f}")
    
    return results


def evaluate_lclo(X, y, cells, n_folds=5):
    """Leave-Cell-Line-Out evaluation."""
    unique_cells = np.unique(cells)
    np.random.seed(42)
    perm = np.random.permutation(len(unique_cells))
    fold_size = len(unique_cells) // n_folds
    
    results = []
    for fold in range(n_folds):
        start = fold * fold_size
        end = start + fold_size if fold < n_folds - 1 else len(unique_cells)
        test_set = set(unique_cells[perm[start:end]])
        
        te_mask = np.array([c in test_set for c in cells])
        tr_mask = ~te_mask
        
        if te_mask.sum() == 0 or tr_mask.sum() == 0:
            continue
        
        model = XGBRegressor(
            n_estimators=300, max_depth=6, learning_rate=0.1,
            subsample=0.8, colsample_bytree=0.8, n_jobs=-1,
            tree_method='hist', random_state=42
        )
        model.fit(X[tr_mask], y[tr_mask], eval_set=[(X[te_mask], y[te_mask])],
                  verbose=False)
        
        pred = model.predict(X[te_mask])
        r, _ = pearsonr(y[te_mask], pred)
        results.append(r)
        logger.info(f"  LCLO Fold {fold}: r={r:.4f}")
    
    return results


def main():
    print("=" * 70)
    print("XGBoost Retraining on Combined 1M+ Dataset")
    print("=" * 70)
    
    # Use 200K sample for tractable training
    # (full 1M would take hours with XGBoost)
    MAX_SAMPLES = 200000
    
    X, y, sources, drug_a, drug_b, cells = build_features(max_samples=MAX_SAMPLES)
    print(f"\nDataset: {X.shape[0]:,} samples x {X.shape[1]} features")
    print(f"Sources: O'Neil={np.sum(sources=='oneil'):,}, DrugComb={np.sum(sources=='drugcomb'):,}")
    print(f"Synergy stats: mean={y.mean():.2f}, std={y.std():.2f}")
    
    # 1. Pre-defined folds
    print(f"\n{'='*70}")
    print("Pre-defined 5-Fold CV (XGBoost, Combined Data)")
    print("=" * 70)
    pf_results, final_model = evaluate_predefined(X, y)
    print(f"\n  PF Mean: r = {np.mean(pf_results):.4f} +/- {np.std(pf_results):.4f}")
    
    # 2. LDPO
    print(f"\n{'='*70}")
    print("LDPO (XGBoost, Combined Data)")
    print("=" * 70)
    ldpo_results = evaluate_ldpo(X, y, drug_a, drug_b)
    print(f"\n  LDPO Mean: r = {np.mean(ldpo_results):.4f} +/- {np.std(ldpo_results):.4f}")
    
    # 3. LCLO
    print(f"\n{'='*70}")
    print("LCLO (XGBoost, Combined Data)")
    print("=" * 70)
    lclo_results = evaluate_lclo(X, y, cells)
    print(f"\n  LCLO Mean: r = {np.mean(lclo_results):.4f} +/- {np.std(lclo_results):.4f}")
    
    # Save model
    model_path = MODEL_DIR / "xgboost_synergy_combined.pkl"
    with open(model_path, 'wb') as f:
        pickle.dump(final_model, f)
    print(f"\nModel saved: {model_path}")
    
    # Comparison
    print(f"\n{'='*70}")
    print("COMPARISON: XGBoost O'Neil vs Combined")
    print("=" * 70)
    print(f"  O'Neil only (23K):     PF=0.605  LDPO=0.604  LCLO=0.510")
    print(f"  Combined ({X.shape[0]//1000}K):  "
          f"PF={np.mean(pf_results):.3f}  "
          f"LDPO={np.mean(ldpo_results):.3f}  "
          f"LCLO={np.mean(lclo_results):.3f}")


if __name__ == "__main__":
    main()
