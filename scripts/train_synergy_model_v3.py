"""
Train v3 Synergy Model: Drug FP + CCLE Expression + Biological Features
========================================================================
Features:
  - Drug A Morgan FP (1024 bits)
  - Drug B Morgan FP (1024 bits)
  - Cell line CCLE expression (256 top-variance genes)
  - Cell line biological features (7 tissue + 8 mutation = 15)
  Total: 2048 + 256 + 15 = 2319 features
"""
import pandas as pd
import numpy as np
import xgboost as xgb
import pickle
import json
import logging
from pathlib import Path
from sklearn.model_selection import KFold
from scipy.stats import pearsonr, spearmanr

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

DATA_DIR = Path("F:/ADDS/data/ml_training")
MODEL_DIR = Path("F:/ADDS/models/synergy")


def main():
    # Load data
    synergy = pd.read_csv(DATA_DIR / "oneil_synergy.csv")
    logger.info(f"Synergy: {len(synergy)} records")
    
    # Load drug fingerprints
    with open(MODEL_DIR / "drug_fingerprints.pkl", 'rb') as f:
        drug_fps = pickle.load(f)
    logger.info(f"Drug FPs: {len(drug_fps)} drugs x 1024 bits")
    
    # Load cell line expression
    expr_file = MODEL_DIR / "cell_line_expression.pkl"
    if expr_file.exists():
        with open(expr_file, 'rb') as f:
            cell_expr = pickle.load(f)
        logger.info(f"Cell expression: {len(cell_expr)} lines x 256 genes")
    else:
        cell_expr = {}
        logger.warning("No cell expression data found")
    
    # Load biological features
    bio_file = DATA_DIR / "cell_line_features.csv"
    cell_bio = {}
    if bio_file.exists():
        bio_df = pd.read_csv(bio_file)
        for _, row in bio_df.iterrows():
            cl = row['cell_line']
            feats = row.drop('cell_line').values.astype(np.float32)
            cell_bio[cl] = feats
        logger.info(f"Bio features: {len(cell_bio)} lines x {len(feats)} features")
    
    # Build feature matrix
    n_fp = 1024
    n_expr = 256
    n_bio = len(next(iter(cell_bio.values()))) if cell_bio else 0
    
    zero_fp = np.zeros(n_fp, dtype=np.float32)
    zero_expr = np.zeros(n_expr, dtype=np.float32)
    zero_bio = np.zeros(n_bio, dtype=np.float32) if n_bio > 0 else np.array([], dtype=np.float32)
    
    X_list = []
    y_list = []
    skipped = 0
    
    # Build case-insensitive lookups
    drug_fps_upper = {k.upper(): v for k, v in drug_fps.items()}
    cell_expr_upper = {k.upper(): v for k, v in cell_expr.items()}
    cell_bio_upper = {k.upper(): v for k, v in cell_bio.items()}
    
    for _, row in synergy.iterrows():
        drug_a = str(row['drug_a']).upper()
        drug_b = str(row['drug_b']).upper()
        cell = str(row['cell_line']).upper()
        target = float(row['synergy_loewe'])
        
        if np.isnan(target):
            skipped += 1
            continue
        
        fp_a = drug_fps_upper.get(drug_a, zero_fp)
        fp_b = drug_fps_upper.get(drug_b, zero_fp)
        expr = cell_expr_upper.get(cell, zero_expr)
        bio = cell_bio_upper.get(cell, zero_bio) if n_bio > 0 else np.array([], dtype=np.float32)
        
        # Only include if at least one drug is known
        if np.array_equal(fp_a, zero_fp) and np.array_equal(fp_b, zero_fp):
            skipped += 1
            continue
        
        features = np.concatenate([fp_a, fp_b, expr, bio])
        X_list.append(features)
        y_list.append(target)
    
    X = np.array(X_list)
    y = np.array(y_list)
    logger.info(f"Feature matrix: {X.shape}, skipped: {skipped}")
    logger.info(f"Target range: [{y.min():.1f}, {y.max():.1f}], mean={y.mean():.2f}")
    
    # 5-fold CV
    n_total_features = n_fp * 2 + n_expr + n_bio
    logger.info(f"Features breakdown: {n_fp}*2 drug FP + {n_expr} expr + {n_bio} bio = {n_total_features}")
    
    params = {
        'n_estimators': 500,
        'max_depth': 8,
        'learning_rate': 0.05,
        'subsample': 0.8,
        'colsample_bytree': 0.3,
        'min_child_weight': 10,
        'reg_alpha': 0.1,
        'reg_lambda': 1.0,
        'random_state': 42,
        'n_jobs': -1,
    }
    
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    results = {'pearson_r': [], 'spearman_r': [], 'rmse': [], 'r2': []}
    
    logger.info(f"Training XGBoost v3 with 5-fold CV...")
    
    for fold, (train_idx, test_idx) in enumerate(kf.split(X)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        model = xgb.XGBRegressor(**params)
        model.fit(X_train, y_train, verbose=False)
        
        y_pred = model.predict(X_test)
        
        pr, _ = pearsonr(y_test, y_pred)
        sr, _ = spearmanr(y_test, y_pred)
        rmse = np.sqrt(np.mean((y_test - y_pred) ** 2))
        ss_res = np.sum((y_test - y_pred) ** 2)
        ss_tot = np.sum((y_test - y_test.mean()) ** 2)
        r2 = 1 - ss_res / ss_tot
        
        results['pearson_r'].append(pr)
        results['spearman_r'].append(sr)
        results['rmse'].append(rmse)
        results['r2'].append(r2)
        
        logger.info(f"  Fold {fold+1}: Pearson r={pr:.4f}, Spearman={sr:.4f}, RMSE={rmse:.2f}, R2={r2:.4f}")
    
    # Train final model on all data
    final_model = xgb.XGBRegressor(**params)
    final_model.fit(X, y, verbose=False)
    
    # Save v3 model
    final_model.save_model(str(MODEL_DIR / "xgb_synergy_v3.json"))
    logger.info(f"Model saved: {MODEL_DIR / 'xgb_synergy_v3.json'}")
    
    # Save encoders and metadata
    meta = {
        'version': 'v3',
        'description': 'Drug Morgan FP (1024x2) + CCLE Expression (256) + Bio features (15)',
        'n_features': n_total_features,
        'n_fp_bits': n_fp,
        'n_expr_genes': n_expr,
        'n_bio_features': n_bio,
        'n_drugs': len(drug_fps),
        'n_cell_lines_expr': len(cell_expr),
        'n_cell_lines_bio': len(cell_bio),
        'n_samples': len(X),
        'cv_results': {k: f"{np.mean(v):.4f} +/- {np.std(v):.4f}" for k, v in results.items()},
        'params': params,
    }
    
    with open(MODEL_DIR / "model_metadata_v3.json", 'w') as f:
        json.dump(meta, f, indent=2)
    
    # Print summary
    print("=" * 60)
    print("Synergy Model v3: Drug FP + CCLE Expression + Bio Features")
    print("=" * 60)
    print(f"\n{'='*60}")
    print(f"5-Fold CV Results (Drug FP + CCLE + Bio, no leakage)")
    print(f"{'='*60}")
    for key, vals in results.items():
        print(f"  {key:15s}: {np.mean(vals):.4f} +/- {np.std(vals):.4f}")
    print(f"{'='*60}")
    
    # Compare with v2 (drug FP only)
    print(f"\nComparison with v2:")
    print(f"  v2 (drug FP only):  Pearson r=0.6920 +/- 0.0146")
    print(f"  v3 (FP+expr+bio):  Pearson r={np.mean(results['pearson_r']):.4f} +/- {np.std(results['pearson_r']):.4f}")
    improvement = ((np.mean(results['pearson_r']) - 0.6920) / 0.6920) * 100
    print(f"  Improvement: {improvement:+.1f}%")


if __name__ == "__main__":
    main()
