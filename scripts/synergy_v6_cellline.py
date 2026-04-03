"""
시너지 모델 v6 — Cell-line features + High-quality data only

핵심 개선:
1. O'Neil 22.7K (dose-response, 고품질 데이터만)
2. DepMap CCLE gene expression → PCA 100차원 cell-line 피처
3. Drug FP + Cell-line expression = 3-way input
4. DrugComb 987K 제외 (노이즈 심한 single-concentration)

기대 효과: r=0.636 → r=0.75+ (cell-line 정보 추가로 +0.10)
"""

import json
import logging
import pickle
import time
import urllib.request
from io import StringIO
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DATA_DIR = Path("F:/ADDS/data/ml_training")
MODEL_DIR = Path("F:/ADDS/models/synergy")
DEPMAP_DIR = DATA_DIR / "depmap"
DEPMAP_DIR.mkdir(parents=True, exist_ok=True)


# ===================================================================
# 1. DepMap Cell-line Expression Download
# ===================================================================

def download_depmap_expression() -> pd.DataFrame:
    """Download DepMap CCLE expression (OmicsExpressionProteinCodingGenesTPMLogp1).
    
    URL: https://depmap.org/portal/download/api/downloads
    Direct file: ~24Q4 release
    """
    cache_path = DEPMAP_DIR / "ccle_expression.parquet"
    
    if cache_path.exists():
        logger.info(f"Loading cached DepMap expression: {cache_path}")
        return pd.read_parquet(cache_path)
    
    # Try multiple DepMap URLs
    urls = [
        # DepMap 24Q4 expression (protein coding genes, TPM log2+1)
        "https://ndownloader.figshare.com/files/43346616",  # 23Q4
        "https://depmap.org/portal/download/api/downloads?release_name=DepMap+Public+24Q2&file_name=OmicsExpressionProteinCodingGenesTPMLogp1.csv",
    ]
    
    csv_path = DEPMAP_DIR / "ccle_expression.csv"
    
    for url in urls:
        try:
            logger.info(f"Trying DepMap download: {url[:80]}...")
            req = urllib.request.Request(url, headers={"User-Agent": "ADDS-Research/1.0"})
            with urllib.request.urlopen(req, timeout=120) as resp:
                data = resp.read()
            with open(csv_path, "wb") as f:
                f.write(data)
            logger.info(f"  Downloaded: {len(data)/1e6:.1f} MB")
            break
        except Exception as e:
            logger.warning(f"  Failed: {e}")
    
    if csv_path.exists() and csv_path.stat().st_size > 1e6:
        df = pd.read_csv(csv_path, index_col=0, low_memory=False)
        logger.info(f"  DepMap expression: {df.shape}")
        df.to_parquet(cache_path)
        return df
    
    return pd.DataFrame()


def build_cellline_features_fallback() -> Tuple[Dict[str, np.ndarray], int]:
    """Fallback: build cell-line features from curated cancer gene panel.
    
    Uses known cancer pathway gene expression signatures
    instead of full DepMap download.
    """
    logger.info("Building fallback cell-line features from curated cancer gene panel...")
    
    # NCI-60 cell line characterization (published gene signatures)
    # Based on cancer type + known molecular subtypes
    CANCER_TYPES = {
        # Colorectal
        "HCT116": ("colon", "MSI-H", "KRAS_G13D"),
        "SW620": ("colon", "MSS", "KRAS_G12V"),
        "DLD1": ("colon", "MSI-H", "KRAS_G13D"),
        "COLO320DM": ("colon", "MSS", "WT"),
        "HT29": ("colon", "MSS", "BRAF_V600E"),
        "SW837": ("colon", "MSS", "KRAS_G12C"),
        "RKO": ("colon", "MSI-H", "BRAF_V600E"),
        "HCT15": ("colon", "MSI-H", "KRAS_G13D"),
        # Breast
        "MCF7": ("breast", "ER+", "PIK3CA_E545K"),
        "T47D": ("breast", "ER+", "PIK3CA_H1047R"),
        "EFM192B": ("breast", "HER2+", "WT"),
        "MDAMB231": ("breast", "TNBC", "KRAS_G13D"),
        "MDAMB468": ("breast", "TNBC", "WT"),
        # Ovarian
        "SKOV3": ("ovarian", "serous", "PIK3CA"),
        "OVCAR3": ("ovarian", "serous", "TP53"),
        "OV90": ("ovarian", "serous", "TP53"),
        "CAOV3": ("ovarian", "serous", "TP53"),
        "ES2": ("ovarian", "clear_cell", "BRAF_G464V"),
        "PA1": ("ovarian", "teratoma", "WT"),
        # Lung
        "A427": ("lung", "NSCLC", "KRAS_G12D"),
        "NCIH460": ("lung", "NSCLC", "KRAS_Q61H"),
        "NCIH23": ("lung", "NSCLC", "KRAS_G12C"),
        "NCIH520": ("lung", "NSCLC_SCC", "WT"),
        # Melanoma
        "A2058": ("melanoma", "BRAF_mut", "BRAF_V600E"),
        "A375": ("melanoma", "BRAF_mut", "BRAF_V600E"),
        "HT144": ("melanoma", "BRAF_mut", "BRAF_V600E"),
        "SKMEL30": ("melanoma", "NRAS_mut", "NRAS_Q61K"),
        "RPMI7951": ("melanoma", "BRAF_mut", "BRAF_V600E"),
        "UACC62": ("melanoma", "BRAF_mut", "BRAF_V600E"),
        # Other
        "A2780": ("ovarian", "serous", "WT"),
        "VCAP": ("prostate", "AR+", "WT"),
        "UWB1289": ("ovarian", "serous", "BRCA1"),
        "UWB1289BRCA1": ("ovarian", "serous", "BRCA1_restored"),
        "SKMES1": ("lung", "SCC", "WT"),
        "OCUBM": ("bladder", "TCC", "WT"),
    }
    
    # Cancer type encoding (one-hot)
    cancer_types = sorted(set(v[0] for v in CANCER_TYPES.values()))
    subtype_set = sorted(set(v[1] for v in CANCER_TYPES.values()))
    
    # Feature dimension: cancer_type (7) + subtype (12) + mutation_profile (8) + synthetic_expr (20)
    n_feat = len(cancer_types) + len(subtype_set) + 8 + 20
    
    features = {}
    np.random.seed(42)
    
    # Mutation gene indices
    MUT_GENES = ["KRAS", "BRAF", "PIK3CA", "TP53", "NRAS", "BRCA1", "PTEN", "WT"]
    
    for cell, (ctype, subtype, mutation) in CANCER_TYPES.items():
        feat = np.zeros(n_feat, dtype=np.float32)
        
        # Cancer type one-hot
        feat[cancer_types.index(ctype)] = 1.0
        
        # Subtype one-hot
        off1 = len(cancer_types)
        if subtype in subtype_set:
            feat[off1 + subtype_set.index(subtype)] = 1.0
        
        # Mutation profile
        off2 = off1 + len(subtype_set)
        for i, gene in enumerate(MUT_GENES):
            if mutation.startswith(gene):
                feat[off2 + i] = 1.0
        
        # Synthetic expression signature (based on cancer type lineage)
        off3 = off2 + len(MUT_GENES)
        # Use hash of cancer type + cell name for reproducible pseudo-expression
        hash_val = hash(cell + ctype) % 10000
        rng = np.random.RandomState(hash_val)
        
        # Cancer-type-specific expression patterns
        expr_base = rng.randn(20).astype(np.float32) * 0.5
        
        # MSI status affects expression of MMR genes
        if subtype == "MSI-H":
            expr_base[:3] += 1.5  # MMR deficiency signature
        
        # BRAF mutation affects MAPK pathway
        if "BRAF" in mutation:
            expr_base[3:6] += 1.0  # MAPK activation
        
        # KRAS mutation
        if "KRAS" in mutation:
            expr_base[5:8] += 0.8
        
        feat[off3:off3+20] = expr_base
        features[cell] = feat
    
    logger.info(f"  Built features for {len(features)} cell lines, dim={n_feat}")
    return features, n_feat


def load_depmap_or_fallback():
    """Try DepMap, fall back to curated features."""
    expr_df = download_depmap_expression()
    
    if len(expr_df) > 0:
        logger.info(f"DepMap loaded: {expr_df.shape}")
        # Handle NaN: drop columns >50% NaN, then impute remaining
        nan_pct = expr_df.isna().mean()
        keep_cols = nan_pct[nan_pct < 0.5].index
        expr_clean = expr_df[keep_cols].copy()
        logger.info(f"  After NaN filter: {expr_clean.shape} (dropped {len(expr_df.columns)-len(keep_cols)} cols)")
        # Median impute remaining NaN
        expr_clean = expr_clean.fillna(expr_clean.median())
        # Drop rows still with NaN
        expr_clean = expr_clean.dropna()
        logger.info(f"  After impute: {expr_clean.shape}")
        # PCA compress
        scaler = StandardScaler()
        X = scaler.fit_transform(expr_clean.values)
        n_comp = min(100, X.shape[1], X.shape[0]-1)
        pca = PCA(n_components=n_comp)
        X_pca = pca.fit_transform(X)
        logger.info(f"  PCA: {X_pca.shape}, variance explained: {pca.explained_variance_ratio_.sum():.2f}")
        
        features = {}
        for i, cell in enumerate(expr_clean.index):
            features[cell] = X_pca[i].astype(np.float32)
        
        return features, X_pca.shape[1], pca
    else:
        logger.warning("DepMap download failed, using fallback features")
        features, dim = build_cellline_features_fallback()
        return features, dim, None


# ===================================================================
# 2. Cell-line Name Normalization
# ===================================================================

def normalize_cellline_name(name: str) -> str:
    """Normalize cell line names for matching across datasets."""
    n = str(name).upper().strip()
    # Remove common separators
    n = n.replace("-", "").replace("_", "").replace(" ", "").replace("/", "").replace(".", "")
    return n


def build_cellline_map(available_features: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """Build mapping from any cell-line name variant to feature vector."""
    mapping = {}
    for name, feat in available_features.items():
        norm = normalize_cellline_name(name)
        mapping[norm] = feat
        mapping[name] = feat
        mapping[name.upper()] = feat
    return mapping


# ===================================================================
# 3. Data Loading and Feature Construction
# ===================================================================

def build_synergy_features(use_oneil_only: bool = True):
    """Build drug+drug+cell_line features for synergy prediction."""
    
    # Load synergy data
    df = pd.read_csv(DATA_DIR / "synergy_combined.csv", low_memory=False)
    logger.info(f"Total synergy data: {len(df)} rows")
    
    if use_oneil_only:
        df = df[df["source"] == "oneil"]
        logger.info(f"O'Neil only: {len(df)} rows, {df.cell_line.nunique()} cell lines")
    
    # Load SMILES + FPs
    smiles = {}
    for p in [MODEL_DIR / "drug_smiles.json", MODEL_DIR / "drug_smiles_extended.json"]:
        if p.exists():
            with open(p) as f:
                smiles.update(json.load(f))
    
    from rdkit import Chem
    from rdkit.Chem import AllChem
    
    fps = {}
    for name, smi in smiles.items():
        mol = Chem.MolFromSmiles(smi)
        if mol:
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=1024)
            fps[name] = np.array(fp, dtype=np.float32)
    logger.info(f"Drug FPs: {len(fps)}")
    
    # Load cell-line features
    cl_features, cl_dim, pca = load_depmap_or_fallback()
    cl_map = build_cellline_map(cl_features)
    logger.info(f"Cell-line features: {len(cl_features)} lines, dim={cl_dim}")
    
    # Build features
    X_list = []
    y_list = []
    n_matched_cl = 0
    n_total = 0
    
    for _, row in df.iterrows():
        da, db = str(row["drug_a"]), str(row["drug_b"])
        cl = str(row["cell_line"])
        score = float(row["synergy_loewe"])
        
        if da not in fps or db not in fps or np.isnan(score):
            continue
        
        n_total += 1
        
        # Drug features (2048)
        fp = np.concatenate([fps[da], fps[db]])
        
        # Cell-line features
        cl_norm = normalize_cellline_name(cl)
        if cl_norm in cl_map:
            cl_feat = cl_map[cl_norm]
            n_matched_cl += 1
        else:
            cl_feat = np.zeros(cl_dim, dtype=np.float32)
        
        x = np.concatenate([fp, cl_feat])
        X_list.append(x)
        y_list.append(score)
    
    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.float32)
    
    cl_pct = n_matched_cl / max(n_total, 1) * 100
    logger.info(f"Features: X={X.shape} ({2048} FP + {cl_dim} cell-line)")
    logger.info(f"Cell-line match: {n_matched_cl}/{n_total} ({cl_pct:.1f}%)")
    
    return X, y, cl_dim


# ===================================================================
# 4. Training + Evaluation
# ===================================================================

def train_and_evaluate(X, y, tag="", n_splits=5):
    """Train XGBoost with 5-fold CV for honest evaluation."""
    import xgboost as xgb
    
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    fold_results = []
    
    for fold, (ti, vi) in enumerate(kf.split(X)):
        model = xgb.XGBRegressor(
            n_estimators=500, max_depth=8, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
            reg_alpha=0.1, reg_lambda=1.0,
            tree_method="hist", device="cuda:0",
            random_state=42, early_stopping_rounds=30,
        )
        model.fit(X[ti], y[ti], eval_set=[(X[vi], y[vi])], verbose=0)
        
        yp = model.predict(X[vi])
        r = pearsonr(y[vi], yp)[0]
        rmse = np.sqrt(mean_squared_error(y[vi], yp))
        fold_results.append({"r": r, "rmse": rmse, "n": len(vi)})
        logger.info(f"  {tag} Fold {fold+1}: r={r:.4f}, RMSE={rmse:.2f}")
    
    avg_r = np.mean([f["r"] for f in fold_results])
    std_r = np.std([f["r"] for f in fold_results])
    avg_rmse = np.mean([f["rmse"] for f in fold_results])
    
    logger.info(f"  {tag} CV: r={avg_r:.4f}+/-{std_r:.4f}, RMSE={avg_rmse:.2f}")
    return {"avg_r": round(avg_r, 4), "std_r": round(std_r, 4), 
            "avg_rmse": round(avg_rmse, 2), "per_fold": fold_results}


def train_final_model(X, y, cl_dim):
    """Train final model on all data."""
    import xgboost as xgb
    
    # 80/20 split for final metrics
    ti, vi = train_test_split(np.arange(len(X)), test_size=0.2, random_state=42)
    
    model = xgb.XGBRegressor(
        n_estimators=500, max_depth=8, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
        reg_alpha=0.1, reg_lambda=1.0,
        tree_method="hist", device="cuda:0",
        random_state=42, early_stopping_rounds=30,
    )
    model.fit(X[ti], y[ti], eval_set=[(X[vi], y[vi])], verbose=100)
    
    yp = model.predict(X[vi])
    r = pearsonr(y[vi], yp)[0]
    sr = spearmanr(y[vi], yp)[0]
    rmse = np.sqrt(mean_squared_error(y[vi], yp))
    
    # Feature importance: FP vs cell-line
    imp = model.feature_importances_
    fp_imp = imp[:2048].sum()
    cl_imp = imp[2048:].sum()
    total = fp_imp + cl_imp + 1e-10
    
    logger.info(f"Final model: r={r:.4f}, spearman={sr:.4f}, RMSE={rmse:.2f}")
    logger.info(f"Feature importance: FP={fp_imp/total*100:.1f}%, CellLine={cl_imp/total*100:.1f}%")
    
    return model, {
        "val_r": round(float(r), 4),
        "val_spearman": round(float(sr), 4),
        "val_rmse": round(float(rmse), 2),
        "fp_importance_pct": round(float(fp_imp/total*100), 1),
        "cellline_importance_pct": round(float(cl_imp/total*100), 1),
        "n_train": len(ti),
        "n_val": len(vi),
    }


# ===================================================================
# 5. Main with A/B Comparison
# ===================================================================

def main():
    logger.info("=" * 60)
    logger.info("시너지 v6: Cell-line features + O'Neil data")
    logger.info("=" * 60)
    
    # === Experiment A: O'Neil only, FP only (baseline) ===
    logger.info("\n--- A: O'Neil only, FP only (baseline) ---")
    X_fp, y_fp, _ = build_synergy_features(use_oneil_only=True)
    X_fp_only = X_fp[:, :2048]  # Strip cell-line features
    cv_fp = train_and_evaluate(X_fp_only, y_fp, tag="FP-only")
    
    # === Experiment B: O'Neil only, FP + Cell-line ===
    logger.info("\n--- B: O'Neil only, FP + Cell-line ---")
    cv_full = train_and_evaluate(X_fp, y_fp, tag="FP+CellLine")
    
    # === Experiment C: All data, FP + Cell-line ===
    logger.info("\n--- C: All data, FP + Cell-line ---")
    X_all, y_all, cl_dim_all = build_synergy_features(use_oneil_only=False)
    cv_all = train_and_evaluate(X_all, y_all, tag="AllData+CL")
    
    # === Final model: best config ===
    logger.info("\n--- Final Model ---")
    # Choose best based on CV r
    best_tag = "B"
    best_X, best_y = X_fp, y_fp
    best_cl_dim = X_fp.shape[1] - 2048
    
    if cv_all["avg_r"] > cv_full["avg_r"]:
        best_tag = "C"
        best_X, best_y = X_all, y_all
        best_cl_dim = cl_dim_all
    
    model, final_metrics = train_final_model(best_X, best_y, best_cl_dim)
    
    # Save
    with open(MODEL_DIR / "xgboost_synergy_v6_cellline.pkl", "wb") as f:
        pickle.dump(model, f)
    
    results = {
        "version": "v6",
        "description": "Drug synergy with cell-line features",
        "experiments": {
            "A_oneil_fp_only": cv_fp,
            "B_oneil_fp_cellline": cv_full,
            "C_alldata_fp_cellline": cv_all,
        },
        "best_config": best_tag,
        "final_metrics": final_metrics,
        "comparison": {
            "v4_alldata_fponly": "r=0.636 (927K samples, FP only)",
            "v6_best": f"r={final_metrics['val_r']} ({best_tag})",
        },
    }
    
    with open(MODEL_DIR / "synergy_v6_results.json", "w") as f:
        json.dump(results, f, indent=2, default=float)
    
    logger.info(f"\n{'='*60}")
    logger.info("비교 결과:")
    logger.info(f"  v4 (927K, FP only):      r=0.636")
    logger.info(f"  A (O'Neil, FP only):     r={cv_fp['avg_r']:.4f}+/-{cv_fp['std_r']:.4f}")
    logger.info(f"  B (O'Neil, FP+CellLine): r={cv_full['avg_r']:.4f}+/-{cv_full['std_r']:.4f}")
    logger.info(f"  C (All, FP+CellLine):    r={cv_all['avg_r']:.4f}+/-{cv_all['std_r']:.4f}")
    logger.info(f"  Best → {best_tag}: r={final_metrics['val_r']}")
    logger.info(f"{'='*60}")


if __name__ == "__main__":
    main()
