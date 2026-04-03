"""
Retrain Synergy Model v2 — Real Data + Morgan Fingerprints
============================================================
Uses O'Neil real synergy data (23,052 records) with drug Morgan FP features.
NO feature leakage: only drug chemical structure + cell line identity as inputs.

Reference: O'Neil J et al., Mol Cancer Ther 2016;15(6):1155-62
Method: XGBoost with 5-fold CV (matching DeepSynergy eval protocol)
"""

import pandas as pd
import numpy as np
import json
import pickle
import logging
from pathlib import Path
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from scipy.stats import pearsonr, spearmanr

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

DATA_DIR = Path("F:/ADDS/data/ml_training")
MODEL_DIR = Path("F:/ADDS/models/synergy")
MODEL_DIR.mkdir(parents=True, exist_ok=True)


def load_data():
    """Load synergy data + drug fingerprints."""
    synergy = pd.read_csv(DATA_DIR / "oneil_synergy.csv")
    fps = pd.read_csv(DATA_DIR / "drug_fingerprints.csv")
    
    logger.info(f"Synergy: {len(synergy)} records")
    logger.info(f"Drug FPs: {len(fps)} drugs x {len(fps.columns)-1} features")
    
    return synergy, fps


def build_features(synergy: pd.DataFrame, fps: pd.DataFrame) -> tuple:
    """
    Build feature matrix: Drug_A FP (1024) + Drug_B FP (1024) + cell_line_enc = 2049 features.
    
    NO synergy_bliss, synergy_loewe, synergy_hsa, IC50 — these are leaky features.
    """
    fp_cols = [c for c in fps.columns if c.startswith('fp_')]
    fp_lookup = {}
    for _, row in fps.iterrows():
        fp_lookup[row['drug_name']] = row[fp_cols].values.astype(np.float32)
    
    # Cell line encoding
    cell_le = LabelEncoder()
    synergy['cell_line_enc'] = cell_le.fit_transform(synergy['cell_line'])
    
    n = len(synergy)
    n_fp = len(fp_cols)
    
    # Feature matrix: [drug_a_fp (1024) | drug_b_fp (1024) | cell_line_enc (1)]
    X = np.zeros((n, 2 * n_fp + 1), dtype=np.float32)
    y = synergy['synergy_loewe'].values.astype(np.float32)
    
    valid_mask = np.ones(n, dtype=bool)
    
    for i in range(n):
        drug_a = synergy.iloc[i]['drug_a']
        drug_b = synergy.iloc[i]['drug_b']
        
        fp_a = fp_lookup.get(drug_a)
        fp_b = fp_lookup.get(drug_b)
        
        if fp_a is None or fp_b is None:
            valid_mask[i] = False
            continue
        
        X[i, :n_fp] = fp_a
        X[i, n_fp:2*n_fp] = fp_b
        X[i, 2*n_fp] = synergy.iloc[i]['cell_line_enc']
    
    X = X[valid_mask]
    y = y[valid_mask]
    
    feature_names = [f'fp_a_{j}' for j in range(n_fp)] + \
                    [f'fp_b_{j}' for j in range(n_fp)] + \
                    ['cell_line_enc']
    
    logger.info(f"Feature matrix: {X.shape[0]} samples x {X.shape[1]} features")
    logger.info(f"Target range: [{y.min():.1f}, {y.max():.1f}], mean={y.mean():.2f}")
    
    return X, y, feature_names, cell_le


def train_xgboost_cv(X, y, feature_names, n_folds=5):
    """Train XGBoost with k-fold CV, matching DeepSynergy evaluation."""
    import xgboost as xgb
    
    # Use pre-defined folds if available, otherwise create random folds
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    
    metrics = {'pearson_r': [], 'spearman_r': [], 'rmse': [], 'r2': []}
    
    best_model = None
    best_pearson = -1
    
    params = {
        'n_estimators': 500,
        'max_depth': 8,
        'learning_rate': 0.05,
        'subsample': 0.8,
        'colsample_bytree': 0.3,  # important: many sparse binary features
        'min_child_weight': 10,
        'reg_alpha': 0.1,
        'reg_lambda': 1.0,
        'random_state': 42,
        'n_jobs': -1,
    }
    
    logger.info(f"Training XGBoost with {n_folds}-fold CV...")
    logger.info(f"Params: {params}")
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        model = xgb.XGBRegressor(**params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )
        
        y_pred = model.predict(X_val)
        
        # Metrics
        r, _ = pearsonr(y_val, y_pred)
        rho, _ = spearmanr(y_val, y_pred)
        rmse = np.sqrt(np.mean((y_val - y_pred) ** 2))
        ss_res = np.sum((y_val - y_pred) ** 2)
        ss_tot = np.sum((y_val - np.mean(y_val)) ** 2)
        r2 = 1 - ss_res / ss_tot
        
        metrics['pearson_r'].append(r)
        metrics['spearman_r'].append(rho)
        metrics['rmse'].append(rmse)
        metrics['r2'].append(r2)
        
        logger.info(f"  Fold {fold+1}: Pearson r={r:.4f}, Spearman={rho:.4f}, RMSE={rmse:.2f}, R2={r2:.4f}")
        
        if r > best_pearson:
            best_pearson = r
            best_model = model
    
    # Summary
    print("\n" + "=" * 60)
    print("5-Fold Cross-Validation Results (Drug FP only, no leakage)")
    print("=" * 60)
    for metric_name, values in metrics.items():
        mean = np.mean(values)
        std = np.std(values)
        print(f"  {metric_name:15s}: {mean:.4f} +/- {std:.4f}")
    print("=" * 60)
    
    return best_model, metrics


def save_model(model, feature_names, cell_le, metrics):
    """Save v2 model + metadata."""
    # Model
    model_file = MODEL_DIR / "xgb_synergy_v2.json"
    model.save_model(str(model_file))
    logger.info(f"Model saved: {model_file}")
    
    # Encoders
    encoder_file = MODEL_DIR / "encoders_v2.pkl"
    with open(encoder_file, 'wb') as f:
        pickle.dump({'cell_line': cell_le}, f)
    logger.info(f"Encoders saved: {encoder_file}")
    
    # Metadata
    meta = {
        'version': 'v2',
        'data_source': "O'Neil et al. 2016 (DeepSynergy labels.csv)",
        'n_samples': 23052,
        'n_drugs': 38,
        'n_cell_lines': 39,
        'feature_type': 'Morgan FP 1024-bit ECFP4 (no leakage)',
        'feature_names': feature_names,
        'n_features': len(feature_names),
        'model_type': 'XGBoost Regressor',
        'cv_metrics': {k: {'mean': float(np.mean(v)), 'std': float(np.std(v))} 
                       for k, v in metrics.items()},
    }
    
    # Top feature importances
    importances = model.feature_importances_
    top_idx = np.argsort(importances)[-20:][::-1]
    meta['top_features'] = [
        {'name': feature_names[i], 'importance': float(importances[i])}
        for i in top_idx
    ]
    
    meta_file = MODEL_DIR / "model_metadata_v2.json"
    with open(meta_file, 'w') as f:
        json.dump(meta, f, indent=2)
    logger.info(f"Metadata saved: {meta_file}")
    
    # Drug fingerprint lookup (for integration)
    fps = pd.read_csv(DATA_DIR / "drug_fingerprints.csv")
    fp_file = MODEL_DIR / "drug_fingerprints.pkl"
    fp_cols = [c for c in fps.columns if c.startswith('fp_')]
    fp_dict = {row['drug_name']: row[fp_cols].values.astype(np.float32) for _, row in fps.iterrows()}
    with open(fp_file, 'wb') as f:
        pickle.dump(fp_dict, f)
    logger.info(f"Drug FP lookup saved: {fp_file}")


def main():
    print("=" * 80)
    print("Synergy Model v2: Real Data + Morgan FP (No Feature Leakage)")
    print("=" * 80)
    
    # Load data
    synergy, fps = load_data()
    
    # Build features
    X, y, feature_names, cell_le = build_features(synergy, fps)
    
    # Train
    best_model, metrics = train_xgboost_cv(X, y, feature_names)
    
    # Save
    save_model(best_model, feature_names, cell_le, metrics)
    
    print("\n[DONE] Synergy model v2 saved to", MODEL_DIR)


if __name__ == "__main__":
    main()
