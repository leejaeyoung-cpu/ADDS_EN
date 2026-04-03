"""
TCGA RNA-seq 처리 + 치료반응 통합 모델

1. 다운로드된 50개 STAR-Counts TSV 파일 파싱
2. DEG feature 추출 (Wilcoxon rank-sum)
3. GSE39582 + TCGA 통합 학습
"""

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu
from sklearn.ensemble import (
    GradientBoostingClassifier,
    RandomForestClassifier,
    VotingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DATA_DIR = Path("F:/ADDS/data/ml_training")
MODEL_DIR = Path("F:/ADDS/models/treatment_response")


def load_tcga_rnaseq():
    """Load and parse TCGA RNA-seq STAR-Counts files."""
    rnaseq_dir = DATA_DIR / "tcga" / "rnaseq"
    chemo_csv = DATA_DIR / "tcga" / "tcga_coad_chemo_survival.csv"

    # Load survival labels
    chemo_df = pd.read_csv(chemo_csv)
    labels = dict(zip(chemo_df["case_id"], chemo_df["label"]))
    logger.info("Chemo patients with labels: %d", len(labels))

    # Parse each TSV
    all_data = {}
    gene_names = None

    for fpath in sorted(rnaseq_dir.glob("*.tsv")):
        # Filename format: TCGA-XX-XXXX_*.rna_seq.*.tsv
        case_id = fpath.name.split("_")[0]
        if case_id not in labels:
            continue

        try:
            df = pd.read_csv(fpath, sep="\t", comment="#")

            # STAR-Counts format: gene_id, gene_name, gene_type, unstranded, ...
            # We want 'gene_name' and 'tpm_unstranded' or 'unstranded'
            if "gene_name" not in df.columns:
                continue

            # Use unstranded counts (or tpm_unstranded if available)
            count_col = None
            for col in ["tpm_unstranded", "fpkm_unstranded", "unstranded"]:
                if col in df.columns:
                    count_col = col
                    break

            if count_col is None:
                continue

            # Filter protein-coding genes
            if "gene_type" in df.columns:
                df = df[df["gene_type"] == "protein_coding"]

            # Remove duplicates
            df = df.drop_duplicates(subset=["gene_name"]).set_index("gene_name")
            values = df[count_col].astype(float)

            # Skip first few rows if they contain metadata
            values = values[~values.index.str.startswith("N_")]

            if case_id in all_data:
                # Average if multiple samples for same patient
                all_data[case_id] = (all_data[case_id] + values.reindex(all_data[case_id].index, fill_value=0)) / 2
            else:
                all_data[case_id] = values
                if gene_names is None:
                    gene_names = values.index.tolist()

        except Exception as e:
            logger.warning("  Failed %s: %s", fpath.name, e)

    logger.info("Parsed %d TCGA patients", len(all_data))

    if not all_data:
        return None, None, None

    # Build expression matrix
    common_genes = set(gene_names)
    for v in all_data.values():
        common_genes &= set(v.index)
    common_genes = sorted(common_genes)
    logger.info("Common genes: %d", len(common_genes))

    X_data = []
    y_data = []
    case_ids = []
    for case_id, values in all_data.items():
        X_data.append(values.reindex(common_genes, fill_value=0).values)
        y_data.append(labels[case_id])
        case_ids.append(case_id)

    X = np.array(X_data, dtype=np.float32)
    y = np.array(y_data, dtype=int)

    # Log transform
    X = np.log2(X + 1)

    logger.info("TCGA expression: X=%s, y=%d resp / %d nonresp",
                X.shape, y.sum(), len(y) - y.sum())

    return X, y, common_genes


def select_deg_features(X, y, genes, n_top=100):
    """Select top DEG features by Wilcoxon rank-sum test."""
    resp_idx = np.where(y == 1)[0]
    nonr_idx = np.where(y == 0)[0]

    pvals = []
    for j in range(X.shape[1]):
        try:
            _, p = mannwhitneyu(X[resp_idx, j], X[nonr_idx, j], alternative="two-sided")
        except Exception:
            p = 1.0
        pvals.append(p)

    top_idx = np.argsort(pvals)[:n_top]
    top_genes = [genes[i] for i in top_idx]
    top_pvals = [pvals[i] for i in top_idx]

    logger.info("Top DEG features (p-value):")
    for g, p in zip(top_genes[:10], top_pvals[:10]):
        logger.info("  %s: p=%.2e", g, p)

    return X[:, top_idx], top_genes


def train_tcga_treatment_model(X, y, title="TCGA DEG"):
    """Train treatment response model with DEG features."""
    n_resp = y.sum()
    n_nonr = len(y) - n_resp

    if n_nonr < 5 or n_resp < 5:
        logger.warning("Too few samples: resp=%d, nonresp=%d", n_resp, n_nonr)
        return {"error": "too few samples"}

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    n_splits = min(5, n_nonr)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    aucs = []

    for fold, (ti, vi) in enumerate(skf.split(X_scaled, y)):
        clf = VotingClassifier(
            estimators=[
                ("rf", RandomForestClassifier(n_estimators=100, random_state=42, max_depth=5)),
                ("gb", GradientBoostingClassifier(n_estimators=100, random_state=42, max_depth=3)),
                ("lr", LogisticRegression(C=0.1, random_state=42, max_iter=1000)),
            ],
            voting="soft",
        )
        clf.fit(X_scaled[ti], y[ti])
        prob = clf.predict_proba(X_scaled[vi])[:, 1]

        try:
            auc = roc_auc_score(y[vi], prob)
            aucs.append(auc)
            logger.info("  %s fold %d: AUC=%.4f", title, fold + 1, auc)
        except Exception:
            pass

    if aucs:
        avg = np.mean(aucs)
        std = np.std(aucs)
        logger.info("  %s CV AUC: %.4f +/- %.4f (%d-fold)", title, avg, std, len(aucs))
        return {"auc": round(avg, 4), "std": round(std, 4), "n_folds": len(aucs)}

    return {"error": "no valid folds"}


def main():
    logger.info("=" * 60)
    logger.info("TCGA RNA-seq Treatment Response Model")
    logger.info("=" * 60)

    # Load TCGA expression
    X_tcga, y_tcga, genes_tcga = load_tcga_rnaseq()

    results = {}

    if X_tcga is not None:
        # Select DEG features
        X_deg, deg_genes = select_deg_features(X_tcga, y_tcga, genes_tcga, n_top=100)

        # Train on TCGA DEG features
        logger.info("\n--- TCGA DEG 100 ---")
        r1 = train_tcga_treatment_model(X_deg, y_tcga, "TCGA DEG-100")
        results["tcga_deg100"] = r1

        # Also try with 50 DEG
        X_deg50, _ = select_deg_features(X_tcga, y_tcga, genes_tcga, n_top=50)
        logger.info("\n--- TCGA DEG 50 ---")
        r2 = train_tcga_treatment_model(X_deg50, y_tcga, "TCGA DEG-50")
        results["tcga_deg50"] = r2

    else:
        logger.warning("No TCGA expression data loaded")

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("RESULTS")
    logger.info("=" * 60)
    for name, r in results.items():
        if "auc" in r:
            logger.info("  %s: AUC=%.4f +/- %.4f", name, r["auc"], r["std"])
        else:
            logger.info("  %s: %s", name, r)

    # Save
    out_path = MODEL_DIR / "v4" / "tcga_results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Saved: %s", out_path)


if __name__ == "__main__":
    main()
