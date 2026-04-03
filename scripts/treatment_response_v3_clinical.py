"""
치료반응 v3: DEG + Clinical features

v2b 대비 개선:
1. Clinical features 추가: age, sex, stage, KRAS, BRAF, TP53, MMR, CIMP, CIN, location, subtype
2. GSE39582만 사용 (가장 풍부한 clinical annotation)
3. DEG + clinical 통합 모델

기대: AUC 0.68 → 0.73+
"""

import gzip
import json
import logging
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import rankdata, mannwhitneyu
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DATA_DIR = Path("F:/ADDS/data/ml_training")
MODEL_DIR = Path("F:/ADDS/models/treatment_response/v3")
MODEL_DIR.mkdir(parents=True, exist_ok=True)


def parse_gse39582_soft():
    """Parse GSE39582 SOFT file for clinical annotations."""
    soft_path = DATA_DIR / "GSE39582_family.soft.gz"
    
    samples = {}
    current_sample = None
    with gzip.open(soft_path, "rt", errors="replace") as f:
        for line in f:
            line = line.strip()
            if line.startswith("^SAMPLE = "):
                current_sample = line.split(" = ")[1]
                samples[current_sample] = {}
            elif line.startswith("!Sample_characteristics_ch1 = ") and current_sample:
                parts = line.split(" = ", 1)[1]
                if ": " in parts:
                    key, val = parts.split(": ", 1)
                    key = key.strip().lower().replace(" ", "_")
                    samples[current_sample][key] = val.strip()
    
    logger.info(f"GSE39582: {len(samples)} samples parsed")
    return samples


def extract_clinical_features(samples):
    """Extract clinical features from SOFT annotations."""
    records = []
    
    for gsm, clin in samples.items():
        rec = {"gsm": gsm}
        
        # Age (continuous)
        age = clin.get("age.at.diagnosis_(year)", "N/A")
        try:
            rec["age"] = float(age)
        except:
            rec["age"] = np.nan
        
        # Sex (binary)
        sex = clin.get("sex", "N/A")
        rec["sex_male"] = 1.0 if sex == "Male" else (0.0 if sex == "Female" else np.nan)
        
        # TNM Stage (ordinal)
        stage = clin.get("tnm.stage", "N/A")
        stage_map = {"0": 0, "1": 1, "2": 2, "3": 3, "4": 4}
        rec["stage"] = stage_map.get(stage, np.nan)
        
        # T stage
        t = clin.get("tnm.t", "N/A")
        t_map = {"T0": 0, "Tis": 0, "T1": 1, "T2": 2, "T3": 3, "T4": 4}
        rec["t_stage"] = t_map.get(t, np.nan)
        
        # N stage
        n = clin.get("tnm.n", "N/A")
        n_map = {"N0": 0, "N1": 1, "N+": 2, "N2": 2, "N3": 3}
        rec["n_stage"] = n_map.get(n, np.nan)
        
        # M stage  
        m = clin.get("tnm.m", "N/A")
        rec["m_distant"] = 1.0 if m == "M1" else (0.0 if m in ["M0", "MX"] else np.nan)
        
        # Mutations (binary)
        for gene in ["kras", "braf", "tp53"]:
            mut = clin.get(f"{gene}.mutation", "N/A")
            rec[f"{gene}_mut"] = 1.0 if mut == "M" else (0.0 if mut == "WT" else np.nan)
        
        # MMR status
        mmr = clin.get("mmr.status", "N/A")
        rec["mmr_deficient"] = 1.0 if mmr == "dMMR" else (0.0 if mmr == "pMMR" else np.nan)
        
        # CIMP / CIN
        cimp = clin.get("cimp.status", "N/A")
        rec["cimp_pos"] = 1.0 if cimp == "+" else (0.0 if cimp == "-" else np.nan)
        cin = clin.get("cin.status", "N/A")
        rec["cin_pos"] = 1.0 if cin == "+" else (0.0 if cin == "-" else np.nan)
        
        # Tumor location
        loc = clin.get("tumor.location", "N/A")
        rec["proximal"] = 1.0 if loc == "proximal" else (0.0 if loc == "distal" else np.nan)
        
        # Molecular subtype (one-hot)
        subtype = clin.get("cit.molecularsubtype", "N/A")
        for s in ["C1", "C2", "C3", "C4", "C5", "C6"]:
            rec[f"subtype_{s}"] = 1.0 if subtype == s else 0.0
        
        # Chemo (for label)
        chemo = clin.get("chemotherapy.adjuvant", "N/A")
        rec["chemo"] = chemo
        chemo_type = clin.get("chemotherapy.adjuvant.type", "N/A")
        rec["chemo_type"] = chemo_type
        
        # Survival (for label)
        try:
            rec["rfs_delay"] = float(clin.get("rfs.delay", "N/A"))
        except:
            rec["rfs_delay"] = np.nan
        try:
            rec["rfs_event"] = int(clin.get("rfs.event", "N/A"))
        except:
            rec["rfs_event"] = np.nan
        try:
            rec["os_delay"] = float(clin.get("os.delay_(months)", "N/A"))
        except:
            rec["os_delay"] = np.nan
        try:
            rec["os_event"] = int(clin.get("os.event", "N/A"))
        except:
            rec["os_event"] = np.nan
        
        # Dataset split
        rec["dataset"] = clin.get("dataset", "N/A")
        
        records.append(rec)
    
    df = pd.DataFrame(records)
    logger.info(f"Clinical features: {df.shape}")
    return df


def load_expression_data():
    """Load GSE39582 expression matrix."""
    matrix_path = DATA_DIR / "GSE39582_series_matrix.txt.gz"
    
    if matrix_path.exists():
        logger.info(f"Loading expression: {matrix_path}")
        # Parse series matrix
        data_lines = []
        header = None
        with gzip.open(matrix_path, "rt", errors="replace") as f:
            for line in f:
                if line.startswith("!series_matrix_table_begin"):
                    header = next(f).strip().split("\t")
                    for dline in f:
                        if dline.startswith("!series_matrix_table_end"):
                            break
                        data_lines.append(dline.strip().split("\t"))
        
        if data_lines and header:
            # Clean header (remove quotes)
            header = [h.strip('"') for h in header]
            
            df = pd.DataFrame(data_lines, columns=header)
            df.set_index(header[0], inplace=True)
            df = df.apply(pd.to_numeric, errors="coerce")
            df = df.dropna(axis=0, how="all")
            
            logger.info(f"Expression: {df.shape} (probes x samples)")
            return df
    
    logger.warning("Expression matrix not found")
    return pd.DataFrame()


def select_deg_features(expr, labels, n_top=100):
    """Select top DEG features by Wilcoxon rank-sum p-value."""
    resp_idx = labels[labels == 1].index
    nonr_idx = labels[labels == 0].index
    
    p_values = {}
    for probe in expr.index:
        r_vals = expr.loc[probe, resp_idx].dropna().values
        n_vals = expr.loc[probe, nonr_idx].dropna().values
        if len(r_vals) > 5 and len(n_vals) > 5:
            try:
                _, p = mannwhitneyu(r_vals, n_vals, alternative="two-sided")
                p_values[probe] = p
            except:
                pass
    
    sorted_probes = sorted(p_values, key=p_values.get)
    top_probes = sorted_probes[:n_top]
    logger.info(f"DEG selection: {len(p_values)} tested, top {n_top} selected")
    if top_probes:
        logger.info(f"  Best p-value: {p_values[top_probes[0]]:.2e}")
        logger.info(f"  Worst selected: {p_values[top_probes[-1]]:.2e}")
    return top_probes


def main():
    logger.info("=" * 60)
    logger.info("Treatment Response v3: DEG + Clinical Features")
    logger.info("=" * 60)
    
    # 1. Parse clinical
    samples = parse_gse39582_soft()
    clin_df = extract_clinical_features(samples)
    
    # 2. Filter chemo patients
    chemo_patients = clin_df[
        (clin_df["chemo"] == "Y") & 
        clin_df["rfs_event"].notna() & 
        clin_df["rfs_delay"].notna()
    ].copy()
    logger.info(f"Chemo patients with survival: {len(chemo_patients)}")
    
    # Label: responder = no recurrence within 3 years
    chemo_patients["label"] = ((chemo_patients["rfs_event"] == 0) | 
                                (chemo_patients["rfs_delay"] > 36)).astype(int)
    
    resp_count = chemo_patients["label"].sum()
    nonr_count = len(chemo_patients) - resp_count
    logger.info(f"  Responder: {resp_count}, Non-responder: {nonr_count}")
    
    # 3. Load expression
    expr_df = load_expression_data()
    
    if len(expr_df) == 0:
        logger.error("No expression data. Exiting.")
        return
    
    # Align samples
    common_gsm = sorted(set(chemo_patients["gsm"]) & set(expr_df.columns))
    logger.info(f"Common samples: {len(common_gsm)}")
    
    if len(common_gsm) < 50:
        logger.error(f"Too few common samples: {len(common_gsm)}")
        return
    
    expr_sub = expr_df[common_gsm]
    clin_sub = chemo_patients[chemo_patients["gsm"].isin(common_gsm)].set_index("gsm").loc[common_gsm]
    labels = clin_sub["label"]
    
    # 4. Variance filter
    variances = expr_sub.var(axis=1)
    var_thresh = variances.quantile(0.5)
    expr_filt = expr_sub[variances > var_thresh]
    logger.info(f"After variance filter: {expr_filt.shape[0]} probes")
    
    # 5. Select DEGs
    top_probes = select_deg_features(expr_filt, labels, n_top=100)
    expr_deg = expr_filt.loc[top_probes]
    
    # 6. Build feature matrix
    # Clinical features
    clinical_cols = ["age", "sex_male", "stage", "t_stage", "n_stage", "m_distant",
                     "kras_mut", "braf_mut", "tp53_mut", "mmr_deficient",
                     "cimp_pos", "cin_pos", "proximal",
                     "subtype_C1", "subtype_C2", "subtype_C3", "subtype_C4", "subtype_C5", "subtype_C6"]
    
    X_clin = clin_sub[clinical_cols].values.astype(np.float32)
    X_deg = expr_deg.T.values.astype(np.float32)
    
    # Fill NaN with median
    for j in range(X_clin.shape[1]):
        col = X_clin[:, j]
        mask = np.isnan(col)
        if mask.any():
            med = np.nanmedian(col)
            X_clin[mask, j] = med
    
    for j in range(X_deg.shape[1]):
        col = X_deg[:, j]
        mask = np.isnan(col)
        if mask.any():
            med = np.nanmedian(col)
            X_deg[mask, j] = med
    
    y = labels.values.astype(int)
    
    # Experiments
    experiments = {
        "A_deg_only": X_deg,
        "B_clinical_only": X_clin,
        "C_deg_clinical": np.hstack([X_deg, X_clin]),
    }
    
    results = {}
    
    for name, X in experiments.items():
        logger.info(f"\n=== {name}: X={X.shape} ===")
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # 5-fold stratified CV
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        aucs = []
        
        for fold, (ti, vi) in enumerate(skf.split(X_scaled, y)):
            clf = VotingClassifier(
                estimators=[
                    ("rf", RandomForestClassifier(n_estimators=500, max_depth=5, min_samples_leaf=5, random_state=42)),
                    ("gb", GradientBoostingClassifier(n_estimators=200, max_depth=3, learning_rate=0.05, min_samples_leaf=5, random_state=42)),
                    ("lr", LogisticRegression(C=0.05, penalty="l1", solver="liblinear", random_state=42)),
                ],
                voting="soft",
            )
            clf.fit(X_scaled[ti], y[ti])
            prob = clf.predict_proba(X_scaled[vi])[:, 1]
            auc = roc_auc_score(y[vi], prob)
            aucs.append(auc)
            logger.info(f"  {name} fold {fold+1}: AUC={auc:.4f}")
        
        avg = np.mean(aucs)
        std = np.std(aucs)
        logger.info(f"  {name} CV: AUC={avg:.4f} +/- {std:.4f}")
        results[name] = {"avg_auc": round(avg, 4), "std_auc": round(std, 4), "n_features": X.shape[1]}
    
    # Save best model (full retrain)
    best_name = max(results, key=lambda k: results[k]["avg_auc"])
    best_X = experiments[best_name]
    logger.info(f"\nBest: {best_name} with AUC={results[best_name]['avg_auc']:.4f}")
    
    scaler = StandardScaler()
    best_X_scaled = scaler.fit_transform(best_X)
    
    final_clf = VotingClassifier(
        estimators=[
            ("rf", RandomForestClassifier(n_estimators=500, max_depth=5, min_samples_leaf=5, random_state=42)),
            ("gb", GradientBoostingClassifier(n_estimators=200, max_depth=3, learning_rate=0.05, min_samples_leaf=5, random_state=42)),
            ("lr", LogisticRegression(C=0.05, penalty="l1", solver="liblinear", random_state=42)),
        ],
        voting="soft",
    )
    final_clf.fit(best_X_scaled, y)
    
    with open(MODEL_DIR / "ensemble_model_v3.pkl", "wb") as f:
        pickle.dump({"model": final_clf, "scaler": scaler, "top_probes": top_probes,
                      "clinical_cols": clinical_cols, "best_config": best_name}, f)
    
    # Feature importance (from RF)
    rf = final_clf.named_estimators_["rf"]
    importances = rf.feature_importances_
    if best_name == "C_deg_clinical":
        deg_imp = importances[:len(top_probes)].sum()
        clin_imp = importances[len(top_probes):].sum()
        logger.info(f"Feature importance: DEG={deg_imp:.3f}, Clinical={clin_imp:.3f}")
        results["feature_importance"] = {
            "deg_pct": round(deg_imp / (deg_imp + clin_imp) * 100, 1),
            "clin_pct": round(clin_imp / (deg_imp + clin_imp) * 100, 1),
        }
    
    # Summary
    logger.info(f"\n{'='*60}")
    logger.info("COMPARISON:")
    logger.info(f"  v2b (DEG 50, no clinical): AUC=0.6827 +/- 0.0240")
    for name, r in results.items():
        if isinstance(r, dict) and "avg_auc" in r:
            logger.info(f"  {name}: AUC={r['avg_auc']:.4f} +/- {r['std_auc']:.4f} (n_feat={r['n_features']})")
    logger.info(f"{'='*60}")
    
    results["v2b_baseline"] = {"avg_auc": 0.6827, "std_auc": 0.0240, "n_features": 50}
    results["n_samples"] = len(y)
    results["n_responder"] = int(resp_count)
    results["n_non_responder"] = int(nonr_count)
    
    with open(MODEL_DIR / "v3_results.json", "w") as f:
        json.dump(results, f, indent=2, default=float)
    
    logger.info(f"Saved: v3 model and results")


if __name__ == "__main__":
    main()
