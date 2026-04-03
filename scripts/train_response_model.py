"""
Treatment Response Prediction Model Trainer
=============================================
TCGA-COAD/READ 임상+변이 데이터로 치료 반응 예측 모델을 학습합니다.

Features: MSI status, KRAS/BRAF/TP53/PIK3CA/APC mutations, stage, age, sex
Target: Disease-free survival >= 36 months (binary classification)

Models: XGBoost (primary) + Logistic Regression (interpretable)
Output: models/response/xgb_response_v1.json, lr_response_v1.pkl
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
import logging
import pickle

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score, classification_report
from sklearn.calibration import calibration_curve

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    from sklearn.ensemble import GradientBoostingClassifier
    HAS_GBC = True
except ImportError:
    HAS_GBC = False

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

DATA_DIR = Path("F:/ADDS/data/ml_training")
MODEL_DIR = Path("F:/ADDS/models/response")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# DFS threshold for binary classification (months)
DFS_THRESHOLD = 36


def load_clinical_data() -> pd.DataFrame:
    """Load and merge TCGA clinical + mutation data."""
    clin_file = DATA_DIR / "tcga_crc_clinical.csv"
    mut_file = DATA_DIR / "tcga_crc_mutations.csv"

    if not clin_file.exists():
        raise FileNotFoundError("No clinical data found. Run download_tcga_crc.py first.")

    clinical = pd.read_csv(clin_file)
    logger.info(f"Clinical data: {len(clinical)} patients, columns: {list(clinical.columns)}")

    # Load mutations if available
    if mut_file.exists():
        mutations = pd.read_csv(mut_file)
        logger.info(f"Mutation data: {len(mutations)} patients, columns: {list(mutations.columns)}")

        # Merge on patient_id
        df = clinical.merge(mutations, on='patient_id', how='left')
        # Fill missing mutations as 0 (wildtype)
        mut_cols = [c for c in df.columns if c.startswith('mut_')]
        df[mut_cols] = df[mut_cols].fillna(0).astype(int)
        logger.info(f"Merged data: {len(df)} patients")
    else:
        df = clinical
        logger.warning("No mutation data file found, proceeding with clinical only")

    return df


def engineer_response_features(df: pd.DataFrame) -> tuple:
    """
    Engineer features for treatment response prediction.

    Returns: (X, y, feature_names, scaler)
    """
    logger.info(f"Engineering response features from {len(df)} patients...")

    # --- Target: DFS >= threshold ---
    if 'dfs_months' in df.columns and 'dfs_event' in df.columns:
        # Patients with DFS >= threshold AND alive/disease-free = positive
        # Patients with DFS < threshold AND event occurred = negative
        # Exclude censored before threshold (ambiguous)
        df_target = df.copy()
        df_target['dfs_months'] = pd.to_numeric(df_target['dfs_months'], errors='coerce')
        df_target['dfs_event'] = pd.to_numeric(df_target['dfs_event'], errors='coerce')

        # Create binary target
        mask_positive = df_target['dfs_months'] >= DFS_THRESHOLD
        mask_negative = (df_target['dfs_months'] < DFS_THRESHOLD) & (df_target['dfs_event'] == 1)
        mask_valid = mask_positive | mask_negative

        df_valid = df_target[mask_valid].copy()
        df_valid['target'] = mask_positive[mask_valid].astype(int)

        logger.info(f"After filtering: {len(df_valid)} patients "
                    f"(positive: {df_valid['target'].sum()}, "
                    f"negative: {(~df_valid['target'].astype(bool)).sum()})")
    elif 'os_months' in df.columns and 'os_event' in df.columns:
        # Fallback to OS
        logger.info("DFS not available, using OS as target")
        df_target = df.copy()
        df_target['os_months'] = pd.to_numeric(df_target['os_months'], errors='coerce')
        df_target['os_event'] = pd.to_numeric(df_target['os_event'], errors='coerce')

        mask_positive = df_target['os_months'] >= DFS_THRESHOLD
        mask_negative = (df_target['os_months'] < DFS_THRESHOLD) & (df_target['os_event'] == 1)
        mask_valid = mask_positive | mask_negative

        df_valid = df_target[mask_valid].copy()
        df_valid['target'] = mask_positive[mask_valid].astype(int)

        logger.info(f"After filtering: {len(df_valid)} patients")
    else:
        raise ValueError("No survival/DFS data available for target creation")

    # --- Features ---
    feature_cols = []

    # Mutation features
    mut_priority = ['mut_KRAS', 'mut_BRAF', 'mut_TP53', 'mut_PIK3CA', 'mut_APC',
                    'mut_SMAD4', 'mut_NRAS', 'mut_PTEN', 'mut_ERBB2',
                    'mut_FBXW7', 'mut_MLH1', 'mut_MSH2', 'mut_MSH6']
    for col in mut_priority:
        if col in df_valid.columns:
            df_valid[col] = pd.to_numeric(df_valid[col], errors='coerce').fillna(0).astype(int)
            feature_cols.append(col)

    # MSI status
    if 'msi_h' in df_valid.columns:
        df_valid['msi_h'] = pd.to_numeric(df_valid['msi_h'], errors='coerce').fillna(0).astype(int)
        feature_cols.append('msi_h')

    # Stage
    if 'stage_ordinal' in df_valid.columns:
        df_valid['stage_ordinal'] = pd.to_numeric(df_valid['stage_ordinal'], errors='coerce').fillna(2)
        feature_cols.append('stage_ordinal')

    # Age
    if 'age' in df_valid.columns:
        df_valid['age'] = pd.to_numeric(df_valid['age'], errors='coerce')
        df_valid['age'] = df_valid['age'].fillna(df_valid['age'].median())
        feature_cols.append('age')

    # Sex
    if 'sex_male' in df_valid.columns:
        df_valid['sex_male'] = pd.to_numeric(df_valid['sex_male'], errors='coerce').fillna(0).astype(int)
        feature_cols.append('sex_male')

    if not feature_cols:
        raise ValueError("No valid feature columns found")

    X = df_valid[feature_cols].values.astype(float)
    y = df_valid['target'].values.astype(int)

    # Handle any remaining NaN
    X = np.nan_to_num(X, nan=0.0)

    logger.info(f"Feature matrix: {X.shape}")
    logger.info(f"Features: {feature_cols}")
    logger.info(f"Target distribution: positive={y.sum()}/{len(y)} "
                f"({100*y.mean():.1f}%)")

    # Scale features for Logistic Regression
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X, X_scaled, y, feature_cols, scaler, df_valid


def train_xgboost_cv(X: np.ndarray, y: np.ndarray, n_folds: int = 5):
    """Train XGBoost classifier with stratified CV."""
    logger.info(f"\n{'='*60}")
    logger.info("Training XGBoost Classifier")
    logger.info(f"{'='*60}")

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

    cv_auc = []
    cv_acc = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        if HAS_XGB:
            model = xgb.XGBClassifier(
                n_estimators=200,
                max_depth=4,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=0.1,
                reg_lambda=1.0,
                scale_pos_weight=len(y_train[y_train == 0]) / max(len(y_train[y_train == 1]), 1),
                random_state=42,
                verbosity=0,
                n_jobs=-1,
                use_label_encoder=False,
                eval_metric='logloss',
            )
        elif HAS_GBC:
            model = GradientBoostingClassifier(
                n_estimators=200,
                max_depth=4,
                learning_rate=0.1,
                subsample=0.8,
                random_state=42,
            )
        else:
            raise ImportError("No gradient boosting library available")

        model.fit(X_train, y_train)
        y_proba = model.predict_proba(X_val)[:, 1]
        y_pred = model.predict(X_val)

        try:
            auc = roc_auc_score(y_val, y_proba)
        except ValueError:
            auc = 0.5

        acc = accuracy_score(y_val, y_pred)

        cv_auc.append(auc)
        cv_acc.append(acc)
        logger.info(f"  Fold {fold}: AUC={auc:.4f}, Accuracy={acc:.4f}")

    mean_auc = np.mean(cv_auc)
    mean_acc = np.mean(cv_acc)

    logger.info(f"\nXGBoost CV Results:")
    logger.info(f"  AUC:      {mean_auc:.4f} +/- {np.std(cv_auc):.4f}")
    logger.info(f"  Accuracy: {mean_acc:.4f} +/- {np.std(cv_acc):.4f}")

    # Train final model on all data
    if HAS_XGB:
        final_model = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            scale_pos_weight=len(y[y == 0]) / max(len(y[y == 1]), 1),
            random_state=42,
            verbosity=0,
            n_jobs=-1,
            use_label_encoder=False,
            eval_metric='logloss',
        )
    else:
        final_model = GradientBoostingClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.1,
            subsample=0.8, random_state=42,
        )

    final_model.fit(X, y)

    xgb_results = {
        'auc': {'mean': float(mean_auc), 'std': float(np.std(cv_auc)), 'folds': [float(x) for x in cv_auc]},
        'accuracy': {'mean': float(mean_acc), 'std': float(np.std(cv_acc)), 'folds': [float(x) for x in cv_acc]},
    }

    return final_model, xgb_results


def train_logistic_regression_cv(X_scaled: np.ndarray, y: np.ndarray,
                                  feature_names: list, n_folds: int = 5):
    """Train Logistic Regression with stratified CV."""
    logger.info(f"\n{'='*60}")
    logger.info("Training Logistic Regression (Interpretable Model)")
    logger.info(f"{'='*60}")

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

    cv_auc = []
    cv_acc = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_scaled, y), 1):
        X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        model = LogisticRegression(
            C=1.0,
            penalty='l2',
            max_iter=1000,
            solver='lbfgs',
            random_state=42,
            class_weight='balanced',
        )
        model.fit(X_train, y_train)
        y_proba = model.predict_proba(X_val)[:, 1]
        y_pred = model.predict(X_val)

        try:
            auc = roc_auc_score(y_val, y_proba)
        except ValueError:
            auc = 0.5

        acc = accuracy_score(y_val, y_pred)

        cv_auc.append(auc)
        cv_acc.append(acc)
        logger.info(f"  Fold {fold}: AUC={auc:.4f}, Accuracy={acc:.4f}")

    mean_auc = np.mean(cv_auc)
    mean_acc = np.mean(cv_acc)

    logger.info(f"\nLogistic Regression CV Results:")
    logger.info(f"  AUC:      {mean_auc:.4f} +/- {np.std(cv_auc):.4f}")
    logger.info(f"  Accuracy: {mean_acc:.4f} +/- {np.std(cv_acc):.4f}")

    # Final model
    final_model = LogisticRegression(
        C=1.0, penalty='l2', max_iter=1000,
        solver='lbfgs', random_state=42, class_weight='balanced',
    )
    final_model.fit(X_scaled, y)

    # Print coefficients
    print("\n--- Logistic Regression Coefficients ---")
    print(f"  Intercept: {final_model.intercept_[0]:.4f}")
    for name, coef in sorted(zip(feature_names, final_model.coef_[0]),
                              key=lambda x: abs(x[1]), reverse=True):
        direction = "(+) better prognosis" if coef > 0 else "(-) worse prognosis"
        print(f"  {name:25s}: {coef:+.4f}  {direction}")

    lr_results = {
        'auc': {'mean': float(mean_auc), 'std': float(np.std(cv_auc)), 'folds': [float(x) for x in cv_auc]},
        'accuracy': {'mean': float(mean_acc), 'std': float(np.std(cv_acc)), 'folds': [float(x) for x in cv_acc]},
        'coefficients': {name: float(coef) for name, coef in zip(feature_names, final_model.coef_[0])},
        'intercept': float(final_model.intercept_[0]),
    }

    return final_model, lr_results


def save_models(xgb_model, lr_model, scaler, feature_names,
                xgb_results, lr_results):
    """Save all trained models and metadata."""
    # XGBoost
    if HAS_XGB and isinstance(xgb_model, xgb.XGBClassifier):
        xgb_file = MODEL_DIR / "xgb_response_v1.json"
        xgb_model.save_model(str(xgb_file))
    else:
        xgb_file = MODEL_DIR / "gbc_response_v1.pkl"
        with open(xgb_file, 'wb') as f:
            pickle.dump(xgb_model, f)
    logger.info(f"XGBoost model saved to {xgb_file}")

    # Logistic Regression
    lr_file = MODEL_DIR / "lr_response_v1.pkl"
    with open(lr_file, 'wb') as f:
        pickle.dump(lr_model, f)
    logger.info(f"Logistic Regression saved to {lr_file}")

    # Scaler
    scaler_file = MODEL_DIR / "scaler.pkl"
    with open(scaler_file, 'wb') as f:
        pickle.dump(scaler, f)
    logger.info(f"Scaler saved to {scaler_file}")

    # Metadata
    metadata = {
        'feature_names': feature_names,
        'dfs_threshold_months': DFS_THRESHOLD,
        'xgb_results': xgb_results,
        'lr_results': lr_results,
        'model_type': {
            'primary': 'xgboost' if HAS_XGB else 'sklearn_gbc',
            'secondary': 'logistic_regression',
        },
        'version': 'v1',
    }

    meta_file = MODEL_DIR / "model_metadata.json"
    with open(meta_file, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    logger.info(f"Metadata saved to {meta_file}")


def main():
    print("=" * 80)
    print("Treatment Response Prediction Model Training")
    print("=" * 80)

    # Load data
    df = load_clinical_data()
    print(f"\nData: {len(df)} patients")

    # Feature engineering
    X, X_scaled, y, feature_names, scaler, df_processed = engineer_response_features(df)
    print(f"\nFeature matrix: {X.shape}")
    print(f"Target: DFS >= {DFS_THRESHOLD} months")
    print(f"  Positive (good response): {y.sum()}/{len(y)} ({100*y.mean():.1f}%)")
    print(f"  Negative (poor response): {(~y.astype(bool)).sum()}/{len(y)} ({100*(1-y.mean()):.1f}%)")

    # Train XGBoost
    xgb_model, xgb_results = train_xgboost_cv(X, y)

    # Train Logistic Regression
    lr_model, lr_results = train_logistic_regression_cv(X_scaled, y, feature_names)

    # Save
    save_models(xgb_model, lr_model, scaler, feature_names, xgb_results, lr_results)

    # Summary
    print(f"\n{'='*60}")
    print("FINAL RESULTS SUMMARY")
    print(f"{'='*60}")
    print(f"  XGBoost:    AUC = {xgb_results['auc']['mean']:.4f} +/- {xgb_results['auc']['std']:.4f}")
    print(f"  Log. Reg.:  AUC = {lr_results['auc']['mean']:.4f} +/- {lr_results['auc']['std']:.4f}")

    best_auc = max(xgb_results['auc']['mean'], lr_results['auc']['mean'])
    if best_auc >= 0.65:
        print(f"\n  [PASS] Best AUC = {best_auc:.4f} >= 0.65 target!")
    else:
        print(f"\n  [INFO] Best AUC = {best_auc:.4f} < 0.65 target.")
        print("  Performance will improve with clinical treatment data or fine-tuning.")

    print(f"\nModels saved to: {MODEL_DIR}")
    print("Done!")


if __name__ == "__main__":
    main()
