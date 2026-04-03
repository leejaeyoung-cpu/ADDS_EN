"""
치료 반응 v2b — Batch-corrected multi-study ensemble

Improvement over v2:
- Per-study z-score normalization (simple ComBat alternative)
- More aggressive DEG selection (top 50 most discriminative)
- Tuned class weights
"""

import json, logging, pickle, warnings, gzip
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DATA_DIR = Path("F:/ADDS/data/ml_training")
MODEL_DIR = Path("F:/ADDS/models/treatment_response/v2")
MODEL_DIR.mkdir(parents=True, exist_ok=True)


def parse_gse39582_soft():
    soft_path = DATA_DIR / "GSE39582_family.soft.gz"
    samples = {}
    current = None
    with gzip.open(soft_path, "rt", errors="replace") as f:
        for line in f:
            if line.startswith("^SAMPLE"):
                current = line.strip().split("=")[-1].strip()
                samples[current] = {}
            elif line.startswith("!Sample_characteristics_ch") and current and ":" in line:
                kv = line.strip().split("=", 1)[1].strip()
                ci = kv.find(":")
                if ci > 0:
                    samples[current][kv[:ci].strip().lower().replace(" ", "_")] = kv[ci+1:].strip()
    return pd.DataFrame.from_dict(samples, orient="index")


def load_expression(path):
    expr = pd.read_csv(path, sep="\t", comment="!", index_col=0, low_memory=False)
    expr = expr.apply(pd.to_numeric, errors="coerce").dropna(axis=0, how="all").dropna(axis=1, how="all")
    return expr


def build_data():
    """Build batch-corrected multi-study data."""
    studies_X = []
    studies_y = []
    studies_id = []

    # GSE39582
    logger.info("=== GSE39582 ===")
    e39 = load_expression(DATA_DIR / "GSE39582_series_matrix.txt.gz")
    c39 = parse_gse39582_soft()
    chemo_col = [c for c in c39.columns if "chemotherapy" in c and "adjuvant" in c and "type" not in c]
    rfs_col = [c for c in c39.columns if "rfs" in c.lower() and "event" in c.lower()]
    if chemo_col and rfs_col:
        mask = c39[chemo_col[0]].str.upper() == "Y"
        chemo = c39[mask].dropna(subset=[rfs_col[0]])
        ids = [s for s in chemo.index if s in e39.columns]
        y = np.array([0 if chemo.loc[s, rfs_col[0]].strip() == "0" else 1 for s in ids])
        # Invert: 0=no relapse=responder(1), 1=relapse=non-resp(0)
        y = 1 - y
        studies_X.append(("GSE39582", e39[ids].T))
        studies_y.append(y)
        studies_id.append(np.full(len(y), 0))
        logger.info(f"  {len(ids)} chemo patients ({sum(y==1)} resp, {sum(y==0)} non-resp)")

    # Additional datasets
    for idx, geo_id in enumerate(["GSE19860", "GSE28702", "GSE72970"], 1):
        sp = DATA_DIR / "chemo_response" / f"{geo_id}_series_matrix.txt.gz"
        cp = DATA_DIR / "chemo_response" / f"{geo_id}_clinical.csv"
        if not sp.exists(): continue

        logger.info(f"=== {geo_id} ===")
        expr = load_expression(sp)
        if cp.exists():
            clin = pd.read_csv(cp)
            if "response" in clin.columns:
                col = "geo_accession" if "geo_accession" in clin.columns else "sample_id"
                ids = [s for s in clin[col].values if s in expr.columns]
                y = np.array([clin[clin[col]==s]["response"].values[0] for s in ids]).astype(int)
                studies_X.append((geo_id, expr[ids].T))
                studies_y.append(y)
                studies_id.append(np.full(len(y), idx))
                logger.info(f"  {len(ids)} patients ({sum(y==1)} resp, {sum(y==0)} non-resp)")

    # Common probes
    common = set(studies_X[0][1].columns)
    for _, df in studies_X[1:]:
        common &= set(df.columns)
    common = sorted(common)
    logger.info(f"\nCommon probes: {len(common)}")

    # Per-study z-score normalization (batch correction)
    normalized = []
    for name, df in studies_X:
        sub = df[common]
        # z-score within each study
        mean = sub.mean(axis=0)
        std = sub.std(axis=0).replace(0, 1)
        z = (sub - mean) / std
        normalized.append(z)
        logger.info(f"  {name}: z-scored {z.shape}")

    X = pd.concat(normalized, ignore_index=True)
    y = np.concatenate(studies_y)
    study = np.concatenate(studies_id)

    logger.info(f"Merged: X={X.shape}, y={y.shape} ({sum(y==1)} resp, {sum(y==0)} non-resp)")
    return X, y, study, common


def deg_select(X, y, top_n=50):
    pvals = {}
    r_mask = y == 1
    for col in X.columns:
        try:
            _, p = stats.mannwhitneyu(X.loc[r_mask, col].dropna(), X.loc[~r_mask, col].dropna())
            pvals[col] = p
        except:
            pvals[col] = 1.0
    sorted_f = sorted(pvals.items(), key=lambda x: x[1])
    logger.info(f"DEG top 5: {[(f, f'{p:.2e}') for f, p in sorted_f[:5]]}")
    return [f for f, _ in sorted_f[:top_n]]


def train_ensemble(X, y, feats):
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
    from sklearn.linear_model import LogisticRegression

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    w = max(sum(y==0), 1) / max(sum(y==1), 1)

    rf = RandomForestClassifier(n_estimators=1000, max_depth=5, min_samples_leaf=8,
                                 max_features="sqrt", class_weight="balanced", random_state=42, n_jobs=-1)
    gb = GradientBoostingClassifier(n_estimators=300, max_depth=3, learning_rate=0.03,
                                     subsample=0.8, min_samples_leaf=10, random_state=42)
    lr = LogisticRegression(penalty="l1", solver="saga", C=0.05, max_iter=5000,
                            class_weight="balanced", random_state=42)

    ensemble = VotingClassifier([("rf", rf), ("gb", gb), ("lr", lr)], voting="soft")
    ensemble.fit(Xs, y)

    rf.fit(Xs, y)
    imp = dict(zip(feats, rf.feature_importances_))
    return {"ensemble": ensemble, "scaler": scaler, "importances": imp, "features": feats}


def evaluate(m, X, y, label=""):
    from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
    Xs = m["scaler"].transform(X)
    yp = m["ensemble"].predict(Xs)
    ypr = m["ensemble"].predict_proba(Xs)[:, 1]
    auc = roc_auc_score(y, ypr) if len(np.unique(y)) > 1 else 0.5
    acc = accuracy_score(y, yp)
    f1 = f1_score(y, yp, zero_division=0)
    logger.info(f"{label}: AUC={auc:.4f} Acc={acc:.4f} F1={f1:.4f} (n={len(y)})")
    return {"auc": round(auc, 4), "accuracy": round(acc, 4), "f1": round(f1, 4), "n": len(y)}


def main():
    logger.info("=" * 60)
    logger.info("치료반응 v2b — Batch-corrected Multi-study Ensemble")
    logger.info("=" * 60)

    X_df, y, study, probes = build_data()

    # Impute NaN
    imp = SimpleImputer(strategy="median")
    X_imp = pd.DataFrame(imp.fit_transform(X_df), columns=X_df.columns)

    # Variance filter
    var = X_imp.var()
    keep = var > np.percentile(var, 25)
    X_flt = X_imp.loc[:, keep]
    logger.info(f"Variance filter: {X_df.shape[1]} → {X_flt.shape[1]}")

    # DEG selection
    deg = deg_select(X_flt, y, top_n=50)
    X_deg = X_flt[deg].values
    feats = deg

    # 5-fold CV
    logger.info("\n--- 5-Fold Stratified CV ---")
    from sklearn.model_selection import StratifiedKFold
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_results = []
    for fold, (ti, vi) in enumerate(skf.split(X_deg, y)):
        m = train_ensemble(X_deg[ti], y[ti], feats)
        r = evaluate(m, X_deg[vi], y[vi], f"Fold {fold+1}")
        cv_results.append(r)

    cv_agg = {k: f"{np.mean([f[k] for f in cv_results]):.4f} +/- {np.std([f[k] for f in cv_results]):.4f}"
              for k in ["auc", "accuracy", "f1"]}
    logger.info(f"CV: {cv_agg}")

    # LOSO-CV
    logger.info("\n--- LOSO-CV ---")
    study_names = ["GSE39582", "GSE19860", "GSE28702", "GSE72970"]
    loso_results = []
    for s in np.unique(study):
        tr = study != s
        te = study == s
        if sum(te) < 5: continue
        m = train_ensemble(X_deg[tr], y[tr], feats)
        r = evaluate(m, X_deg[te], y[te], f"LOSO [{study_names[int(s)]}]")
        r["study"] = study_names[int(s)]
        loso_results.append(r)

    if loso_results:
        loso_agg = {k: f"{np.mean([f[k] for f in loso_results]):.4f}" for k in ["auc", "accuracy", "f1"]}
        logger.info(f"LOSO avg: {loso_agg}")
    else:
        loso_agg = {}

    # Final model
    logger.info("\n--- Final Model ---")
    final = train_ensemble(X_deg, y, feats)
    train_m = evaluate(final, X_deg, y, "Train")

    # Save
    with open(MODEL_DIR / "ensemble_model_v2b.pkl", "wb") as f:
        pickle.dump({"ensemble": final["ensemble"], "scaler": final["scaler"],
                      "imputer": imp, "features": feats}, f)

    top_fi = sorted(final["importances"].items(), key=lambda x: x[1], reverse=True)[:20]
    with open(MODEL_DIR / "feature_importance_v2b.json", "w") as f:
        json.dump(top_fi, f, indent=2)

    meta = {
        "version": "v2b",
        "description": "CRC treatment response — batch-corrected multi-study ensemble",
        "batch_correction": "per-study z-score normalization",
        "n_samples": int(len(y)),
        "n_features": len(feats),
        "feature_selection": "Variance >25th pct + Wilcoxon DEG top 50",
        "cv_5fold": cv_agg,
        "cv_loso": loso_results,
        "loso_avg": loso_agg,
        "train_metrics": train_m,
        "improvement": {"v1": "0.6419 +/- 0.0831", "v2": "0.6574 +/- 0.0439",
                         "v2b_5fold": cv_agg.get("auc", ""), "v2b_loso": loso_agg.get("auc", "")},
        "feature_names": feats,
    }
    with open(MODEL_DIR / "model_metadata_v2b.json", "w") as f:
        json.dump(meta, f, indent=2)

    logger.info(f"\n{'='*60}")
    logger.info(f"v1:  AUC = 0.6419 +/- 0.0831 (211 samples)")
    logger.info(f"v2:  AUC = 0.6574 +/- 0.0439 (487 samples, no batch correction)")
    logger.info(f"v2b: AUC = {cv_agg.get('auc', 'N/A')} (487 samples, z-score batch corrected)")
    logger.info(f"v2b LOSO: AUC = {loso_agg.get('auc', 'N/A')}")
    logger.info(f"{'='*60}")


if __name__ == "__main__":
    main()
