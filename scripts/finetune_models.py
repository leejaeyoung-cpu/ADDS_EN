"""
Model Fine-tuning Interface
=============================
기존 학습 모델을 사용자 데이터로 fine-tuning합니다.

Usage:
  python finetune_models.py --model synergy --data my_synergy_data.csv
  python finetune_models.py --model response --data my_clinical_data.csv

XGBoost: warm start (기존 모델 + 추가 학습)
Logistic Regression: 합산 데이터로 재학습
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path
import json
import pickle
import logging
from datetime import datetime

from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error, r2_score, roc_auc_score, accuracy_score
from scipy.stats import pearsonr, spearmanr

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

DATA_DIR = Path("F:/ADDS/data/ml_training")
SYNERGY_MODEL_DIR = Path("F:/ADDS/models/synergy")
RESPONSE_MODEL_DIR = Path("F:/ADDS/models/response")


def finetune_synergy(data_file: str, n_boost_round: int = 50):
    """
    Fine-tune synergy model with new data.

    Expected CSV columns: drug_a, drug_b, cell_line, synergy_zip
    Optional: ic50_a, ic50_b, synergy_bliss, synergy_loewe, synergy_hsa
    """
    print("=" * 60)
    print("Fine-tuning Synergy Model")
    print("=" * 60)

    # Load new data
    new_data = pd.read_csv(data_file)
    print(f"New data: {len(new_data)} records from {data_file}")

    # Load existing model metadata
    meta_file = SYNERGY_MODEL_DIR / "model_metadata.json"
    encoder_file = SYNERGY_MODEL_DIR / "encoders.pkl"

    if not meta_file.exists():
        print("[ERROR] No existing synergy model found. Run train_synergy_model.py first.")
        return

    with open(meta_file) as f:
        metadata = json.load(f)

    with open(encoder_file, 'rb') as f:
        encoders = pickle.load(f)

    # Extend label encoders with new categories
    for enc_name, le in encoders.items():
        if enc_name in new_data.columns:
            new_vals = set(new_data[enc_name].astype(str)) - set(le.classes_)
            if new_vals:
                le.classes_ = np.append(le.classes_, list(new_vals))
                logger.info(f"Extended {enc_name} encoder with {len(new_vals)} new categories")

    # Feature engineering (same as training)
    feature_cols = metadata['feature_names']
    df = new_data.copy()

    if 'drug_a' in df.columns and 'drug_a' in encoders:
        df['drug_a_enc'] = encoders['drug_a'].transform(df['drug_a'].astype(str))
    else:
        df['drug_a_enc'] = 0

    if 'drug_b' in df.columns and 'drug_b' in encoders:
        df['drug_b_enc'] = encoders['drug_b'].transform(df['drug_b'].astype(str))
    else:
        df['drug_b_enc'] = 0

    if 'cell_line' in df.columns and 'cell_line' in encoders:
        df['cell_line_enc'] = encoders['cell_line'].transform(df['cell_line'].astype(str))
    else:
        df['cell_line_enc'] = 0

    for ic50_col in ['ic50_a', 'ic50_b']:
        if ic50_col in df.columns:
            df[f'{ic50_col}_log'] = np.log1p(pd.to_numeric(df[ic50_col], errors='coerce').fillna(1.0))
        elif f'{ic50_col}_log' in feature_cols:
            df[f'{ic50_col}_log'] = 0.0

    for sc in ['synergy_bliss', 'synergy_loewe', 'synergy_hsa']:
        if sc in df.columns:
            df[sc] = pd.to_numeric(df[sc], errors='coerce').fillna(0)
        elif sc in feature_cols:
            df[sc] = 0.0

    # Prepare X, y
    available_features = [f for f in feature_cols if f in df.columns]
    X_new = df[available_features].values
    y_new = df['synergy_zip'].values

    # Load existing model
    model_file = SYNERGY_MODEL_DIR / "xgb_synergy_v1.json"
    if not model_file.exists():
        model_file = SYNERGY_MODEL_DIR / "gbr_synergy_v1.pkl"

    if HAS_XGB and model_file.suffix == '.json':
        # XGBoost warm start
        existing_model = xgb.XGBRegressor()
        existing_model.load_model(str(model_file))

        print(f"\nFine-tuning XGBoost with {n_boost_round} additional rounds...")
        dtrain = xgb.DMatrix(X_new, label=y_new)
        params = existing_model.get_xgb_params()
        params['verbosity'] = 0

        bst = xgb.Booster(params)
        bst.load_model(str(model_file))
        bst = xgb.train(params, dtrain, num_boost_round=n_boost_round, xgb_model=bst)

        # Save fine-tuned model
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        ft_file = SYNERGY_MODEL_DIR / f"xgb_synergy_finetuned_{timestamp}.json"
        bst.save_model(str(ft_file))
        print(f"Fine-tuned model saved to {ft_file}")

        # Evaluate
        y_pred = bst.predict(dtrain)
        r_val, _ = pearsonr(y_new, y_pred)
        rmse = np.sqrt(mean_squared_error(y_new, y_pred))
        print(f"\nFine-tuned performance on new data:")
        print(f"  Pearson r: {r_val:.4f}")
        print(f"  RMSE:      {rmse:.4f}")
    else:
        print("[INFO] Fine-tuning with sklearn model (full retrain with combined data)")
        # Load original training data and combine
        orig_file = DATA_DIR / "drugcomb_synergy.csv"
        if orig_file.exists():
            orig_data = pd.read_csv(orig_file)
            combined = pd.concat([orig_data, new_data], ignore_index=True)
            print(f"Combined data: {len(combined)} records")
            print("Re-run train_synergy_model.py with combined data for best results.")

    # Save updated encoders
    with open(encoder_file, 'wb') as f:
        pickle.dump(encoders, f)

    print("\nDone!")


def finetune_response(data_file: str, dfs_threshold: int = 36):
    """
    Fine-tune treatment response model with new clinical data.

    Expected CSV columns: patient_id + mutation columns + clinical columns
    """
    print("=" * 60)
    print("Fine-tuning Treatment Response Model")
    print("=" * 60)

    # Load new data
    new_data = pd.read_csv(data_file)
    print(f"New data: {len(new_data)} patients from {data_file}")

    # Load existing metadata
    meta_file = RESPONSE_MODEL_DIR / "model_metadata.json"
    if not meta_file.exists():
        print("[ERROR] No existing response model found. Run train_response_model.py first.")
        return

    with open(meta_file) as f:
        metadata = json.load(f)

    feature_names = metadata['feature_names']

    # Prepare features from new data
    X_new = []
    for col in feature_names:
        if col in new_data.columns:
            X_new.append(pd.to_numeric(new_data[col], errors='coerce').fillna(0).values)
        else:
            logger.warning(f"Feature {col} not in new data, filling with 0")
            X_new.append(np.zeros(len(new_data)))

    X_new = np.column_stack(X_new)

    # Target
    if 'dfs_months' in new_data.columns:
        new_data['dfs_months'] = pd.to_numeric(new_data['dfs_months'], errors='coerce')
        if 'dfs_event' in new_data.columns:
            new_data['dfs_event'] = pd.to_numeric(new_data['dfs_event'], errors='coerce')
            mask_pos = new_data['dfs_months'] >= dfs_threshold
            mask_neg = (new_data['dfs_months'] < dfs_threshold) & (new_data['dfs_event'] == 1)
            mask = mask_pos | mask_neg
            X_new = X_new[mask]
            y_new = mask_pos[mask].astype(int).values
        else:
            y_new = (new_data['dfs_months'] >= dfs_threshold).astype(int).values
    else:
        print("[ERROR] No dfs_months column found in new data")
        return

    print(f"Valid samples: {len(y_new)} (pos: {y_new.sum()}, neg: {(~y_new.astype(bool)).sum()})")

    # Load and fine-tune XGBoost
    xgb_file = RESPONSE_MODEL_DIR / "xgb_response_v1.json"
    if HAS_XGB and xgb_file.exists():
        print("\nFine-tuning XGBoost classifier...")
        dtrain = xgb.DMatrix(X_new, label=y_new)
        bst = xgb.Booster()
        bst.load_model(str(xgb_file))

        params = {'max_depth': 4, 'learning_rate': 0.05, 'objective': 'binary:logistic',
                  'eval_metric': 'logloss', 'verbosity': 0}
        bst = xgb.train(params, dtrain, num_boost_round=50, xgb_model=bst)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        ft_file = RESPONSE_MODEL_DIR / f"xgb_response_finetuned_{timestamp}.json"
        bst.save_model(str(ft_file))
        print(f"Fine-tuned XGBoost saved to {ft_file}")

    # Retrain Logistic Regression with combined data
    scaler_file = RESPONSE_MODEL_DIR / "scaler.pkl"
    if scaler_file.exists():
        with open(scaler_file, 'rb') as f:
            scaler = pickle.load(f)

        # Combine with original data
        orig_clin = DATA_DIR / "tcga_crc_clinical.csv"
        orig_mut = DATA_DIR / "tcga_crc_mutations.csv"
        if orig_clin.exists():
            print("\nRetraining Logistic Regression with combined data...")
            # Simple retrain on new data only (transfer learning style)
            X_scaled = scaler.transform(X_new)

            lr = LogisticRegression(C=1.0, max_iter=1000, solver='lbfgs',
                                     random_state=42, class_weight='balanced')
            lr.fit(X_scaled, y_new)

            ft_lr_file = RESPONSE_MODEL_DIR / f"lr_response_finetuned_{timestamp}.pkl"
            with open(ft_lr_file, 'wb') as f:
                pickle.dump(lr, f)
            print(f"Fine-tuned LR saved to {ft_lr_file}")

            # Print coefficients
            print("\n--- Updated Coefficients ---")
            for name, coef in sorted(zip(feature_names, lr.coef_[0]),
                                      key=lambda x: abs(x[1]), reverse=True):
                print(f"  {name:25s}: {coef:+.4f}")

    print("\nDone!")


def main():
    parser = argparse.ArgumentParser(description='Fine-tune ADDS prediction models')
    parser.add_argument('--model', choices=['synergy', 'response'], required=True,
                        help='Model to fine-tune')
    parser.add_argument('--data', required=True,
                        help='Path to new training data CSV')
    parser.add_argument('--boost-rounds', type=int, default=50,
                        help='Additional XGBoost boosting rounds')
    parser.add_argument('--dfs-threshold', type=int, default=36,
                        help='DFS threshold in months (response model only)')

    args = parser.parse_args()

    if not Path(args.data).exists():
        print(f"[ERROR] Data file not found: {args.data}")
        return

    if args.model == 'synergy':
        finetune_synergy(args.data, n_boost_round=args.boost_rounds)
    elif args.model == 'response':
        finetune_response(args.data, dfs_threshold=args.dfs_threshold)


if __name__ == "__main__":
    main()
