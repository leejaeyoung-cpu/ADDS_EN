"""
Drug Synergy Prediction Model Trainer
======================================
DrugComb 데이터로 XGBoost 시너지 예측 모델을 학습합니다.

Features: drug pair encoding, cell line, IC50 values
Target: ZIP synergy score (continuous regression)
Evaluation: 5-fold CV, Pearson r, Spearman rho, RMSE

Output: models/synergy/xgb_synergy_v1.json
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
import logging
import pickle

from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import pearsonr, spearmanr

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    from sklearn.ensemble import GradientBoostingRegressor
    HAS_GBR = True
except ImportError:
    HAS_GBR = False

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

DATA_DIR = Path("F:/ADDS/data/ml_training")
MODEL_DIR = Path("F:/ADDS/models/synergy")
MODEL_DIR.mkdir(parents=True, exist_ok=True)


def load_synergy_data() -> pd.DataFrame:
    """Load DrugComb synergy data (API or synthetic fallback)."""
    primary_file = DATA_DIR / "drugcomb_synergy.csv"
    lit_file = DATA_DIR / "drugcomb_synergy_literature.csv"

    if primary_file.exists():
        df = pd.read_csv(primary_file)
        logger.info(f"Loaded primary synergy data: {len(df)} records from {primary_file}")
    elif lit_file.exists():
        df = pd.read_csv(lit_file)
        logger.info(f"Loaded literature synergy data: {len(df)} records from {lit_file}")
    else:
        raise FileNotFoundError(
            "No synergy data found. Run download_drugcomb.py first."
        )

    return df


def engineer_features(df: pd.DataFrame) -> tuple:
    """
    Feature engineering for synergy prediction.

    Encodes drug pairs and cell lines as categorical features.
    Uses IC50 values as continuous features.

    Returns: (X, y, feature_names, encoders)
    """
    logger.info(f"Engineering features from {len(df)} records...")

    # Target: ZIP synergy score
    target_col = None
    for col in ['synergy_zip', 'synergy_bliss', 'synergy_loewe']:
        if col in df.columns and df[col].notna().sum() > 50:
            target_col = col
            break

    if not target_col:
        raise ValueError("No synergy score column found with sufficient data")

    logger.info(f"Target column: {target_col}")

    # Drop rows with missing target
    df_clean = df.dropna(subset=[target_col]).copy()

    # Encode drug names
    le_drug_a = LabelEncoder()
    le_drug_b = LabelEncoder()
    le_cell_line = LabelEncoder()

    encoders = {}

    # Drug A encoding
    if 'drug_a' in df_clean.columns:
        df_clean['drug_a_enc'] = le_drug_a.fit_transform(df_clean['drug_a'].astype(str))
        encoders['drug_a'] = le_drug_a
    else:
        df_clean['drug_a_enc'] = 0

    # Drug B encoding
    if 'drug_b' in df_clean.columns:
        df_clean['drug_b_enc'] = le_drug_b.fit_transform(df_clean['drug_b'].astype(str))
        encoders['drug_b'] = le_drug_b
    else:
        df_clean['drug_b_enc'] = 0

    # Cell line encoding
    if 'cell_line' in df_clean.columns:
        df_clean['cell_line_enc'] = le_cell_line.fit_transform(df_clean['cell_line'].astype(str))
        encoders['cell_line'] = le_cell_line
    else:
        df_clean['cell_line_enc'] = 0

    # Feature matrix
    feature_cols = ['drug_a_enc', 'drug_b_enc', 'cell_line_enc']

    # Add IC50 values if available
    for ic50_col in ['ic50_a', 'ic50_b']:
        if ic50_col in df_clean.columns:
            df_clean[ic50_col] = pd.to_numeric(df_clean[ic50_col], errors='coerce')
            # Log-transform IC50 (common in pharmacology)
            df_clean[f'{ic50_col}_log'] = np.log1p(df_clean[ic50_col].fillna(1.0))
            feature_cols.append(f'{ic50_col}_log')

    # Add other synergy scores as features (if predicting one, others can help)
    other_synergy = [c for c in ['synergy_bliss', 'synergy_loewe', 'synergy_hsa']
                     if c in df_clean.columns and c != target_col]
    for sc in other_synergy:
        df_clean[sc] = pd.to_numeric(df_clean[sc], errors='coerce').fillna(0)
        feature_cols.append(sc)

    X = df_clean[feature_cols].values
    y = df_clean[target_col].values

    logger.info(f"Features: {feature_cols}")
    logger.info(f"X shape: {X.shape}, y shape: {y.shape}")
    logger.info(f"Target stats: mean={y.mean():.3f}, std={y.std():.3f}, "
                f"range=[{y.min():.3f}, {y.max():.3f}]")

    return X, y, feature_cols, encoders, df_clean


def train_with_cv(X: np.ndarray, y: np.ndarray, n_folds: int = 5):
    """
    Train XGBoost regressor with k-fold cross-validation.

    Returns: trained model, cv_results dict
    """
    logger.info(f"Starting {n_folds}-fold cross-validation...")

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)

    cv_pearson = []
    cv_spearman = []
    cv_rmse = []
    cv_r2 = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(X), 1):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        if HAS_XGB:
            model = xgb.XGBRegressor(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=0.1,
                reg_lambda=1.0,
                random_state=42,
                verbosity=0,
                n_jobs=-1,
            )
        elif HAS_GBR:
            model = GradientBoostingRegressor(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                random_state=42,
            )
        else:
            raise ImportError("Neither xgboost nor sklearn GradientBoostingRegressor available")

        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)

        # Metrics
        r_val, _ = pearsonr(y_val, y_pred)
        rho_val, _ = spearmanr(y_val, y_pred)
        rmse_val = np.sqrt(mean_squared_error(y_val, y_pred))
        r2_val = r2_score(y_val, y_pred)

        cv_pearson.append(r_val)
        cv_spearman.append(rho_val)
        cv_rmse.append(rmse_val)
        cv_r2.append(r2_val)

        logger.info(f"  Fold {fold}: Pearson r={r_val:.4f}, Spearman rho={rho_val:.4f}, "
                    f"RMSE={rmse_val:.4f}, R2={r2_val:.4f}")

    cv_results = {
        'pearson_r': {'mean': np.mean(cv_pearson), 'std': np.std(cv_pearson), 'folds': cv_pearson},
        'spearman_rho': {'mean': np.mean(cv_spearman), 'std': np.std(cv_spearman), 'folds': cv_spearman},
        'rmse': {'mean': np.mean(cv_rmse), 'std': np.std(cv_rmse), 'folds': cv_rmse},
        'r2': {'mean': np.mean(cv_r2), 'std': np.std(cv_r2), 'folds': cv_r2},
    }

    logger.info(f"\n{'='*60}")
    logger.info(f"CV Results ({n_folds}-fold):")
    logger.info(f"  Pearson r:     {cv_results['pearson_r']['mean']:.4f} +/- {cv_results['pearson_r']['std']:.4f}")
    logger.info(f"  Spearman rho:  {cv_results['spearman_rho']['mean']:.4f} +/- {cv_results['spearman_rho']['std']:.4f}")
    logger.info(f"  RMSE:          {cv_results['rmse']['mean']:.4f} +/- {cv_results['rmse']['std']:.4f}")
    logger.info(f"  R2:            {cv_results['r2']['mean']:.4f} +/- {cv_results['r2']['std']:.4f}")
    logger.info(f"{'='*60}")

    # Train final model on all data
    logger.info("Training final model on all data...")
    if HAS_XGB:
        final_model = xgb.XGBRegressor(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=42,
            verbosity=0,
            n_jobs=-1,
        )
    else:
        final_model = GradientBoostingRegressor(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            random_state=42,
        )
    final_model.fit(X, y)

    return final_model, cv_results


def save_model(model, feature_names, encoders, cv_results):
    """Save trained model + metadata."""
    # Save XGBoost model
    if HAS_XGB and isinstance(model, xgb.XGBRegressor):
        model_file = MODEL_DIR / "xgb_synergy_v1.json"
        model.save_model(str(model_file))
        logger.info(f"XGBoost model saved to {model_file}")
    else:
        model_file = MODEL_DIR / "gbr_synergy_v1.pkl"
        with open(model_file, 'wb') as f:
            pickle.dump(model, f)
        logger.info(f"GBR model saved to {model_file}")

    # Save encoders
    encoder_file = MODEL_DIR / "encoders.pkl"
    with open(encoder_file, 'wb') as f:
        pickle.dump(encoders, f)
    logger.info(f"Encoders saved to {encoder_file}")

    # Save metadata
    metadata = {
        'feature_names': feature_names,
        'cv_results': {
            k: {'mean': float(v['mean']), 'std': float(v['std'])}
            for k, v in cv_results.items()
        },
        'model_type': 'xgboost' if HAS_XGB else 'sklearn_gbr',
        'encoder_classes': {
            name: list(enc.classes_) for name, enc in encoders.items()
        },
        'version': 'v1',
    }

    meta_file = MODEL_DIR / "model_metadata.json"
    with open(meta_file, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    logger.info(f"Metadata saved to {meta_file}")


def print_feature_importance(model, feature_names):
    """Print feature importance ranking."""
    if HAS_XGB and isinstance(model, xgb.XGBRegressor):
        importances = model.feature_importances_
    elif hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    else:
        return

    sorted_idx = np.argsort(importances)[::-1]
    print("\n--- Feature Importance ---")
    for i in sorted_idx:
        print(f"  {feature_names[i]:25s}: {importances[i]:.4f}")


def main():
    print("=" * 80)
    print("Drug Synergy Prediction Model Training")
    print("=" * 80)

    # Load data
    df = load_synergy_data()
    print(f"\nData: {len(df)} records")
    print(f"Columns: {list(df.columns)}")

    # Feature engineering
    X, y, feature_names, encoders, df_processed = engineer_features(df)
    print(f"\nFeature matrix: {X.shape}")
    print(f"Target: synergy_zip, mean={y.mean():.3f}, std={y.std():.3f}")

    # Train with CV
    model, cv_results = train_with_cv(X, y, n_folds=5)

    # Feature importance
    print_feature_importance(model, feature_names)

    # Save
    save_model(model, feature_names, encoders, cv_results)

    # Summary
    r_mean = cv_results['pearson_r']['mean']
    print(f"\n{'='*60}")
    print(f"FINAL RESULTS")
    print(f"{'='*60}")
    print(f"  Pearson r:    {r_mean:.4f} (target >= 0.50)")
    print(f"  Spearman rho: {cv_results['spearman_rho']['mean']:.4f}")
    print(f"  RMSE:         {cv_results['rmse']['mean']:.4f}")
    print(f"  R2:           {cv_results['r2']['mean']:.4f}")

    if r_mean >= 0.5:
        print("\n  [PASS] Synergy model meets target performance!")
    else:
        print(f"\n  [INFO] Pearson r = {r_mean:.4f} < 0.50 target.")
        print("  This is expected with limited/synthetic data.")
        print("  Performance will improve with real DrugComb data or fine-tuning.")

    print(f"\nModel saved to: {MODEL_DIR}")
    print("Done!")


if __name__ == "__main__":
    main()
