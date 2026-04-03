"""
치료 반응 예측 모델 v2 — Multi-study 통합 학습

4개 GEO 데이터셋:
- GSE39582: 585 CRC patients (chemo adjuvant subset)
- GSE19860: 40 CRC patients (FOLFOX response)
- GSE28702: 83 CRC patients (mFOLFOX6 response)
- GSE72970: 124 CRC patients (FOLFIRI/FOLFOX response)

개선:
1. Multi-study 통합 (832 samples total, chemo subset ~350+)
2. DEG feature selection (Wilcoxon rank-sum)
3. Ensemble: RF + XGBoost + Lasso
4. Leave-one-study-out CV (LOSO-CV) for external validation
"""

import json
import logging
import pickle
import warnings
import gzip
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.preprocessing import StandardScaler, LabelEncoder

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DATA_DIR = Path("F:/ADDS/data/ml_training")
MODEL_DIR = Path("F:/ADDS/models/treatment_response/v2")
MODEL_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 1. Extract Clinical Data from GSE39582 SOFT
# ============================================================

def parse_gse39582_soft() -> pd.DataFrame:
    """Parse GSE39582 SOFT file to extract clinical info per sample."""
    soft_path = DATA_DIR / "GSE39582_family.soft.gz"
    logger.info(f"Parsing GSE39582 SOFT: {soft_path}")

    samples = {}
    current_sample = None

    with gzip.open(soft_path, "rt", errors="replace") as f:
        for line in f:
            if line.startswith("^SAMPLE"):
                current_sample = line.strip().split("=")[-1].strip()
                samples[current_sample] = {}
            elif line.startswith("!Sample_characteristics_ch") and current_sample:
                if ":" in line:
                    raw = line.strip()
                    # Format: !Sample_characteristics_ch1 = key: value
                    eq_parts = raw.split("=", 1)
                    if len(eq_parts) == 2:
                        kv = eq_parts[1].strip()
                        colon_idx = kv.find(":")
                        if colon_idx > 0:
                            key = kv[:colon_idx].strip().lower().replace(" ", "_")
                            val = kv[colon_idx + 1:].strip()
                            samples[current_sample][key] = val

    df = pd.DataFrame.from_dict(samples, orient="index")
    logger.info(f"  Parsed {len(df)} samples, {len(df.columns)} characteristics")
    return df


def build_gse39582_labels(clinical: pd.DataFrame) -> Tuple[List[str], np.ndarray]:
    """
    Build binary response labels for GSE39582.
    Strategy: patients who received adjuvant chemo & rfs.event=0 → responder.
    """
    # Filter to chemo patients only
    chemo_col = [c for c in clinical.columns if "chemotherapy" in c and "adjuvant" in c and "type" not in c]
    if not chemo_col:
        logger.warning("No chemotherapy.adjuvant column found")
        return [], np.array([])

    chemo_col = chemo_col[0]
    chemo_patients = clinical[clinical[chemo_col].str.upper() == "Y"]
    logger.info(f"  Chemo patients: {len(chemo_patients)} / {len(clinical)}")

    # RFS event: 0 = no relapse (responder), 1 = relapse (non-responder)
    rfs_col = [c for c in clinical.columns if "rfs" in c.lower() and "event" in c.lower()]
    if not rfs_col:
        # Try os.event as fallback
        rfs_col = [c for c in clinical.columns if "os" in c.lower() and "event" in c.lower()]
    if not rfs_col:
        logger.warning("No rfs.event or survival event column found")
        return [], np.array([])

    rfs_col = rfs_col[0]
    labeled = chemo_patients.dropna(subset=[rfs_col])
    labeled_rfs = labeled[rfs_col].astype(str).str.strip()

    # rfs.event: 0 = censored (no event = good), 1 = relapse (bad)
    # Invert for "response": 1 = responder (no relapse), 0 = non-responder (relapse)
    y = np.where(labeled_rfs == "0", 1, 0)  # 0 event = responder
    sample_ids = list(labeled.index)

    n_resp = int(sum(y == 1))
    n_nonresp = int(sum(y == 0))
    logger.info(f"  Labels: {len(y)} total ({n_resp} resp, {n_nonresp} non-resp)")
    return sample_ids, y


# ============================================================
# 2. Load Expression + Clinical from all studies
# ============================================================

def load_expression(series_path: Path) -> pd.DataFrame:
    """Load expression matrix from GEO series_matrix.txt.gz."""
    expr = pd.read_csv(series_path, sep="\t", comment="!", index_col=0, low_memory=False)
    expr = expr.apply(pd.to_numeric, errors="coerce")
    expr = expr.dropna(axis=0, how="all").dropna(axis=1, how="all")
    return expr


def build_multi_study_data() -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Build unified expression + label dataset across all studies."""

    all_X = []  # (n_samples, n_probes) DataFrames
    all_y = []  # labels
    all_study = []  # study IDs for LOSO-CV

    # --- GSE39582 ---
    logger.info("=== GSE39582 ===")
    expr_39582 = load_expression(DATA_DIR / "GSE39582_series_matrix.txt.gz")
    logger.info(f"  Expression: {expr_39582.shape}")

    clinical_39582 = parse_gse39582_soft()
    sample_ids, y_39582 = build_gse39582_labels(clinical_39582)

    if len(sample_ids) > 0:
        # Filter expression to labeled samples only
        common_samples = [s for s in sample_ids if s in expr_39582.columns]
        expr_sub = expr_39582[common_samples].T
        y_sub = y_39582[:len(common_samples)]
        all_X.append(expr_sub)
        all_y.append(y_sub)
        all_study.append(np.full(len(y_sub), 0))  # study 0
        logger.info(f"  Added: {len(common_samples)} samples")

    # --- Additional GEO datasets ---
    for study_idx, geo_id in enumerate(["GSE19860", "GSE28702", "GSE72970"], start=1):
        logger.info(f"\n=== {geo_id} ===")

        series_path = DATA_DIR / "chemo_response" / f"{geo_id}_series_matrix.txt.gz"
        clinical_path = DATA_DIR / "chemo_response" / f"{geo_id}_clinical.csv"

        if not series_path.exists():
            logger.warning(f"  Series matrix not found")
            continue

        expr = load_expression(series_path)
        logger.info(f"  Expression: {expr.shape}")

        if clinical_path.exists():
            clinical = pd.read_csv(clinical_path)
            logger.info(f"  Clinical: {clinical.shape}")

            # Get response labels (already preprocessed with 'response' column)
            if "response" in clinical.columns:
                y = clinical["response"].values.astype(int)

                # Map sample_id to geo_accession for expression matching
                if "geo_accession" in clinical.columns:
                    sample_ids = clinical["geo_accession"].values
                elif "sample_id" in clinical.columns:
                    sample_ids = clinical["sample_id"].values
                else:
                    sample_ids = clinical.index.values

                # Match expression columns
                common = [s for s in sample_ids if s in expr.columns]
                if len(common) > 0:
                    expr_sub = expr[common].T
                    y_idx = [i for i, s in enumerate(sample_ids) if s in expr.columns]
                    y_sub = y[y_idx]
                    all_X.append(expr_sub)
                    all_y.append(y_sub)
                    all_study.append(np.full(len(y_sub), study_idx))
                    n_r = int(sum(y_sub == 1))
                    n_nr = int(sum(y_sub == 0))
                    logger.info(f"  Added: {len(common)} samples ({n_r} resp, {n_nr} non-resp)")
            else:
                logger.warning(f"  No 'response' column in clinical data")

    if not all_X:
        logger.error("No data loaded!")
        return pd.DataFrame(), np.array([]), np.array([])

    # --- Find common probes ---
    common_probes = set(all_X[0].columns)
    for x in all_X[1:]:
        common_probes &= set(x.columns)
    common_probes = sorted(common_probes)
    logger.info(f"\nCommon probes across studies: {len(common_probes)}")

    # --- Unify ---
    X_list = [x[common_probes] for x in all_X]
    X_merged = pd.concat(X_list, ignore_index=True)
    y_merged = np.concatenate(all_y)
    study_merged = np.concatenate(all_study)

    logger.info(f"Merged dataset: X={X_merged.shape}, y={y_merged.shape}")
    logger.info(f"  Studies: {dict(zip(*np.unique(study_merged, return_counts=True)))}")
    logger.info(f"  Labels: {int(sum(y_merged==1))} resp, {int(sum(y_merged==0))} non-resp")

    return X_merged, y_merged, study_merged


# ============================================================
# 3. Feature Selection
# ============================================================

def deg_selection(X: pd.DataFrame, y: np.ndarray, top_n: int = 100) -> List[str]:
    """DEG feature selection with Wilcoxon rank-sum test."""
    logger.info(f"DEG selection: {X.shape[1]} → top {top_n}")
    resp_mask = y == 1
    pvals = {}
    for col in X.columns:
        try:
            _, p = stats.mannwhitneyu(
                X.loc[resp_mask, col].dropna(),
                X.loc[~resp_mask, col].dropna(),
                alternative="two-sided"
            )
            pvals[col] = p
        except Exception:
            pvals[col] = 1.0

    sorted_feats = sorted(pvals.items(), key=lambda x: x[1])
    selected = [f for f, p in sorted_feats[:top_n]]
    logger.info(f"  Top 5 DEGs: {[(f, f'{p:.2e}') for f, p in sorted_feats[:5]]}")
    return selected


def variance_filter(X: pd.DataFrame, pct: float = 20) -> pd.DataFrame:
    """Remove low-variance probes."""
    variances = X.var()
    threshold = np.percentile(variances.dropna(), pct)
    kept = variances > threshold
    logger.info(f"Variance filter: {X.shape[1]} → {kept.sum()} probes")
    return X.loc[:, kept]


# ============================================================
# 4. Training
# ============================================================

def train_ensemble(X_train, y_train, feature_names):
    """Train RF + XGBoost + Lasso soft-voting ensemble."""
    from sklearn.ensemble import RandomForestClassifier, VotingClassifier
    from sklearn.linear_model import LogisticRegression
    import xgboost as xgb

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X_train)

    rf = RandomForestClassifier(
        n_estimators=500, max_depth=6, min_samples_leaf=5,
        max_features="sqrt", class_weight="balanced",
        random_state=42, n_jobs=-1
    )
    xgb_clf = xgb.XGBClassifier(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, min_child_weight=3,
        scale_pos_weight=max(sum(y_train==0), 1) / max(sum(y_train==1), 1),
        tree_method="hist", random_state=42,
        eval_metric="logloss"
    )
    lasso = LogisticRegression(
        penalty="l1", solver="saga", C=0.1,
        max_iter=5000, class_weight="balanced", random_state=42
    )
    ensemble = VotingClassifier(
        estimators=[("rf", rf), ("xgb", xgb_clf), ("lasso", lasso)],
        voting="soft"
    )
    ensemble.fit(Xs, y_train)

    # Get RF importances separately
    rf.fit(Xs, y_train)
    importances = dict(zip(feature_names, rf.feature_importances_))

    return {"ensemble": ensemble, "scaler": scaler, "importances": importances, "features": feature_names}


def evaluate(model_dict, X, y, label=""):
    """Evaluate model with AUC, accuracy, F1."""
    from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, precision_score, recall_score

    Xs = model_dict["scaler"].transform(X)
    y_pred = model_dict["ensemble"].predict(Xs)
    y_prob = model_dict["ensemble"].predict_proba(Xs)[:, 1]

    auc = roc_auc_score(y, y_prob) if len(np.unique(y)) > 1 else 0.5
    metrics = {
        "auc": round(float(auc), 4),
        "accuracy": round(float(accuracy_score(y, y_pred)), 4),
        "f1": round(float(f1_score(y, y_pred, zero_division=0)), 4),
        "precision": round(float(precision_score(y, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y, y_pred, zero_division=0)), 4),
        "n": len(y),
    }
    logger.info(f"{label}: AUC={metrics['auc']:.4f} Acc={metrics['accuracy']:.4f} F1={metrics['f1']:.4f} (n={metrics['n']})")
    return metrics


def stratified_cv(X, y, features, cv=5):
    """Stratified K-fold cross-validation."""
    from sklearn.model_selection import StratifiedKFold

    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    fold_results = []

    for fold, (ti, vi) in enumerate(skf.split(X, y)):
        model = train_ensemble(X[ti], y[ti], features)
        metrics = evaluate(model, X[vi], y[vi], f"Fold {fold+1}")
        fold_results.append(metrics)

    agg = {k: f"{np.mean([f[k] for f in fold_results]):.4f} +/- {np.std([f[k] for f in fold_results]):.4f}"
           for k in fold_results[0] if k != "n"}
    logger.info(f"\n{cv}-Fold CV: " + " | ".join(f"{k}={v}" for k, v in agg.items()))
    return {"per_fold": fold_results, "aggregated": agg}


def loso_cv(X, y, study, features):
    """Leave-One-Study-Out cross-validation."""
    studies = np.unique(study)
    if len(studies) < 2:
        logger.warning("LOSO-CV needs >= 2 studies")
        return {}

    fold_results = []
    study_names = ["GSE39582", "GSE19860", "GSE28702", "GSE72970"]

    for s in studies:
        train_mask = study != s
        test_mask = study == s

        if sum(test_mask) < 5:
            continue

        model = train_ensemble(X[train_mask], y[train_mask], features)
        name = study_names[int(s)] if int(s) < len(study_names) else f"Study {int(s)}"
        metrics = evaluate(model, X[test_mask], y[test_mask], f"LOSO [{name}]")
        metrics["study"] = name
        fold_results.append(metrics)

    if fold_results:
        agg = {k: f"{np.mean([f[k] for f in fold_results if k in f]):.4f}"
               for k in ["auc", "accuracy", "f1"] if any(k in f for f in fold_results)}
        logger.info(f"LOSO-CV avg: " + " | ".join(f"{k}={v}" for k, v in agg.items()))

    return {"per_study": fold_results, "aggregated": agg if fold_results else {}}


# ============================================================
# 5. Main
# ============================================================

def main():
    logger.info("=" * 60)
    logger.info("치료 반응 예측 v2 — Multi-study 앙상블")
    logger.info("=" * 60)

    # Build unified data
    X_df, y, study = build_multi_study_data()
    if len(X_df) == 0:
        logger.error("No data — exiting")
        return

    # Variance filter
    X_filtered = variance_filter(X_df, pct=20)

    # DEG feature selection
    deg_features = deg_selection(X_filtered, y, top_n=100)
    X_deg = X_filtered[deg_features].values
    feature_names = deg_features

    logger.info(f"\nFinal dataset: {X_deg.shape} ({len(feature_names)} DEG features)")

    # Handle NaN
    from sklearn.impute import SimpleImputer
    imputer = SimpleImputer(strategy="median")
    X_clean = imputer.fit_transform(X_deg)

    # --- Stratified 5-fold CV ---
    logger.info("\n--- Stratified 5-Fold CV ---")
    cv_results = stratified_cv(X_clean, y, feature_names, cv=5)

    # --- LOSO-CV ---
    logger.info("\n--- Leave-One-Study-Out CV ---")
    loso_results = loso_cv(X_clean, y, study, feature_names)

    # --- Train final model ---
    logger.info("\n--- Training Final Model ---")
    final_model = train_ensemble(X_clean, y, feature_names)
    train_metrics = evaluate(final_model, X_clean, y, "Train (full)")

    # --- Save ---
    with open(MODEL_DIR / "ensemble_model.pkl", "wb") as f:
        pickle.dump({
            "ensemble": final_model["ensemble"],
            "scaler": final_model["scaler"],
            "imputer": imputer,
            "features": feature_names,
        }, f)
    logger.info(f"Saved: {MODEL_DIR / 'ensemble_model.pkl'}")

    # Feature importance
    top_fi = sorted(final_model["importances"].items(), key=lambda x: x[1], reverse=True)[:30]
    with open(MODEL_DIR / "feature_importance.json", "w") as f:
        json.dump(top_fi, f, indent=2)

    # Metadata
    meta = {
        "version": "v2",
        "description": "CRC treatment response — multi-study ensemble (RF+XGBoost+Lasso)",
        "n_samples": int(len(y)),
        "n_responders": int(sum(y == 1)),
        "n_non_responders": int(sum(y == 0)),
        "n_features": len(feature_names),
        "feature_selection": "Variance filter (>20th pct) + Wilcoxon DEG (top 100)",
        "datasets": {
            "GSE39582": int(sum(study == 0)),
            "GSE19860": int(sum(study == 1)),
            "GSE28702": int(sum(study == 2)),
            "GSE72970": int(sum(study == 3)),
        },
        "cv_results_5fold": cv_results.get("aggregated", {}),
        "cv_results_loso": loso_results.get("per_study", []),
        "train_metrics": train_metrics,
        "improvement": {
            "v1_auc": "0.6419 +/- 0.0831",
            "v2_auc_5fold": cv_results.get("aggregated", {}).get("auc", "N/A"),
            "v2_auc_loso": loso_results.get("aggregated", {}).get("auc", "N/A"),
        },
        "feature_names": feature_names,
    }
    with open(MODEL_DIR / "model_metadata_v2.json", "w") as f:
        json.dump(meta, f, indent=2)

    logger.info(f"\n{'='*60}")
    logger.info("결과 요약")
    logger.info(f"  v1: AUC = 0.6419 +/- 0.0831 (211 samples, 145 features)")
    logger.info(f"  v2: AUC = {cv_results.get('aggregated', {}).get('auc', 'N/A')} ({len(y)} samples, {len(feature_names)} DEG features)")
    logger.info(f"  LOSO: AUC = {loso_results.get('aggregated', {}).get('auc', 'N/A')}")
    logger.info(f"{'='*60}")


if __name__ == "__main__":
    main()
