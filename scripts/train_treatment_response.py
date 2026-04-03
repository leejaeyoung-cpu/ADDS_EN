"""
Train Treatment Response Model from GSE39582
=============================================
Uses 240 CRC patients who received chemotherapy.
Response = Relapse-Free Survival (RFS) event at 3 years.

Features:
  - Top-variance gene expression (128 genes)
  - Clinical: stage, mutations (KRAS/BRAF/TP53), MMR status
  - Chemo type (5FU vs FOLFOX vs FOLFIRI)
"""
import pandas as pd
import numpy as np
import gzip
import json
import pickle
import logging
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

DATA_DIR = Path("F:/ADDS/data/ml_training")
MODEL_DIR = Path("F:/ADDS/models/treatment_response")
MODEL_DIR.mkdir(parents=True, exist_ok=True)


def load_clinical():
    """Load and process clinical data."""
    df = pd.read_csv(DATA_DIR / "gse39582_clinical_full.csv", index_col=0)
    logger.info(f"Clinical: {len(df)} patients")
    
    # Filter to chemo patients
    chemo = df[df['chemotherapy_adjuvant'] == 'Y'].copy()
    logger.info(f"Chemo patients: {len(chemo)}")
    
    # Create response label: RFS event within 36 months (3 years)
    # rfs_event=1 means relapse, rfs_delay is in months
    chemo['rfs_delay'] = pd.to_numeric(chemo['rfs_delay'], errors='coerce')
    chemo['rfs_event'] = pd.to_numeric(chemo['rfs_event'], errors='coerce')
    chemo['os_event'] = pd.to_numeric(chemo['os_event'], errors='coerce')
    
    # Binary label: responder (no relapse within 3 years) vs non-responder
    chemo['responded'] = 0  # default: non-responder
    # Responder: no RFS event AND followed > 36 months, OR no RFS event with any follow-up
    chemo.loc[(chemo['rfs_event'] == 0) & (chemo['rfs_delay'] >= 36), 'responded'] = 1
    # Non-responder: RFS event within observation period
    chemo.loc[chemo['rfs_event'] == 1, 'responded'] = 0
    # Censored early (no event, but < 36 months follow-up) -> exclude
    censored = (chemo['rfs_event'] == 0) & (chemo['rfs_delay'] < 36)
    
    logger.info(f"Response labels: {chemo['responded'].value_counts().to_dict()}")
    logger.info(f"Censored early (excluded): {censored.sum()}")
    
    chemo = chemo[~censored]
    logger.info(f"After exclusion: {len(chemo)} patients")
    
    return chemo


def build_clinical_features(chemo):
    """Build clinical feature matrix."""
    features = pd.DataFrame(index=chemo.index)
    
    # Stage
    features['stage'] = pd.to_numeric(chemo['tnm_stage'], errors='coerce').fillna(2)
    
    # Mutations
    features['kras_mut'] = (chemo['kras_mutation'] == 'M').astype(int)
    features['braf_mut'] = (chemo['braf_mutation'] == 'M').astype(int)
    features['tp53_mut'] = (chemo['tp53_mutation'] == 'M').astype(int)
    
    # MMR status
    features['mmr_deficient'] = (chemo['mmr_status'] == 'dMMR').astype(int)
    
    # Location
    features['proximal'] = (chemo['tumor_location'] == 'proximal').astype(int)
    
    # Sex
    features['male'] = (chemo['sex'] == 'Male').astype(int)
    
    # Age
    features['age'] = pd.to_numeric(chemo['age_at_diagnosis_(year)'], errors='coerce').fillna(65)
    
    # Chemo type
    features['chemo_5fu'] = (chemo['chemotherapy_adjuvant_type'].isin(['5FU', 'FUFOL'])).astype(int)
    features['chemo_folfox'] = (chemo['chemotherapy_adjuvant_type'] == 'FOLFOX').astype(int)
    features['chemo_folfiri'] = (chemo['chemotherapy_adjuvant_type'] == 'FOLFIRI').astype(int)
    
    # CIT molecular subtype
    for subtype in ['C1', 'C2', 'C3', 'C4', 'C5', 'C6']:
        features[f'subtype_{subtype}'] = (chemo['cit_molecularsubtype'] == subtype).astype(int)
    
    logger.info(f"Clinical features: {features.shape}")
    return features


def load_expression(sample_ids, n_top_genes=128):
    """Load expression data and select top-variance genes."""
    logger.info(f"Loading expression for {len(sample_ids)} samples...")
    
    expr_file = DATA_DIR / "gse39582_expression.csv.gz"
    if not expr_file.exists():
        logger.warning("Expression file not found, using clinical features only")
        return None
    
    # Read expression (genes x samples) - transpose to samples x genes
    expr = pd.read_csv(expr_file, compression='gzip', index_col=0)
    logger.info(f"Full expression: {expr.shape}")
    
    # Filter to our samples
    available = [s for s in sample_ids if s in expr.columns]
    logger.info(f"Available samples: {len(available)}/{len(sample_ids)}")
    
    expr_t = expr[available].T
    
    # Select top-variance genes
    gene_var = expr_t.var(axis=0)
    top_genes = gene_var.nlargest(n_top_genes).index.tolist()
    expr_selected = expr_t[top_genes]
    
    # Z-score normalize
    expr_norm = (expr_selected - expr_selected.mean()) / (expr_selected.std() + 1e-8)
    
    logger.info(f"Expression features: {expr_norm.shape}")
    return expr_norm


def main():
    print("=" * 60)
    print("Treatment Response Model (GSE39582)")
    print("=" * 60)
    
    # Load clinical data
    chemo = load_clinical()
    y = chemo['responded'].values
    
    # Build clinical features
    clinical_feats = build_clinical_features(chemo)
    
    # Load expression
    expr_feats = load_expression(chemo.index.tolist(), n_top_genes=128)
    
    # Combine features
    if expr_feats is not None:
        # Align indices
        common = clinical_feats.index.intersection(expr_feats.index)
        X_clinical = clinical_feats.loc[common]
        X_expr = expr_feats.loc[common]
        X = pd.concat([X_clinical, X_expr], axis=1)
        y = chemo.loc[common, 'responded'].values
    else:
        X = clinical_feats
    
    logger.info(f"Final feature matrix: {X.shape}")
    logger.info(f"Labels: 0 (non-responder)={sum(y==0)}, 1 (responder)={sum(y==1)}")
    
    # Handle NaN
    X = X.fillna(0)
    
    # 5-fold stratified CV
    params = {
        'n_estimators': 200,
        'max_depth': 4,
        'learning_rate': 0.05,
        'subsample': 0.8,
        'colsample_bytree': 0.5,
        'min_child_weight': 5,
        'scale_pos_weight': sum(y==0) / max(sum(y==1), 1),
        'random_state': 42,
        'n_jobs': -1,
        'eval_metric': 'auc',
    }
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    results = {'auc': [], 'accuracy': [], 'f1': []}
    
    logger.info(f"Training XGBoost with 5-fold stratified CV...")
    
    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        X_train, X_test = X.values[train_idx], X.values[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        model = xgb.XGBClassifier(**params)
        model.fit(X_train, y_train, verbose=False)
        
        y_prob = model.predict_proba(X_test)[:, 1]
        y_pred = model.predict(X_test)
        
        auc = roc_auc_score(y_test, y_prob) if len(np.unique(y_test)) > 1 else 0.5
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        
        results['auc'].append(auc)
        results['accuracy'].append(acc)
        results['f1'].append(f1)
        
        logger.info(f"  Fold {fold+1}: AUC={auc:.4f}, Acc={acc:.4f}, F1={f1:.4f}")
    
    # Train final model on all data
    final_model = xgb.XGBClassifier(**params)
    final_model.fit(X.values, y, verbose=False)
    
    # Save model
    final_model.save_model(str(MODEL_DIR / "xgb_treatment_response.json"))
    logger.info(f"Model saved: {MODEL_DIR / 'xgb_treatment_response.json'}")
    
    # Save feature names
    meta = {
        'version': 'v1',
        'description': 'CRC treatment response (GSE39582, chemo patients)',
        'n_samples': len(X),
        'n_features': X.shape[1],
        'feature_names': list(X.columns),
        'label': 'responded (1=no relapse at 3yr, 0=relapsed)',
        'cv_results': {k: f"{np.mean(v):.4f} +/- {np.std(v):.4f}" for k, v in results.items()},
        'params': params,
    }
    with open(MODEL_DIR / "model_metadata.json", 'w') as f:
        json.dump(meta, f, indent=2, default=str)
    
    # Feature importance
    importances = final_model.feature_importances_
    feat_imp = sorted(zip(X.columns, importances), key=lambda x: -x[1])[:20]
    
    print(f"\n{'='*60}")
    print(f"Treatment Response Model (GSE39582)")
    print(f"{'='*60}")
    print(f"Patients: {len(X)} (chemo, non-censored)")
    print(f"Features: {X.shape[1]} (clinical + expression)")
    print(f"\n5-Fold Stratified CV:")
    for k, v in results.items():
        print(f"  {k:10s}: {np.mean(v):.4f} +/- {np.std(v):.4f}")
    
    print(f"\nTop 20 Features:")
    for feat, imp in feat_imp:
        print(f"  {feat:40s}: {imp:.4f}")


if __name__ == "__main__":
    main()
