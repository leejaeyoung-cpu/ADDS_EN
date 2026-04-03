"""
시너지 v6 실험: 4가지 A/B/C/D 비교

A) O'Neil 22K, FP only (고품질 데이터 baseline)
B) O'Neil 22K, FP + cell-line features (cell-line 정보 추가)
C) All 927K, FP + cell-line features (전체 데이터 + cell-line)
D) All 927K, FP only (v4 baseline 재현)

핵심: 데이터 품질(O'Neil only) vs 데이터 양(All) 비교
      + cell-line features 효과 측정
"""

import json, logging, pickle, time, numpy as np, pandas as pd
from pathlib import Path
from scipy.stats import pearsonr
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DATA_DIR = Path("F:/ADDS/data/ml_training")
MODEL_DIR = Path("F:/ADDS/models/synergy")

# Load data
df = pd.read_csv(DATA_DIR / "synergy_combined.csv", low_memory=False)
logger.info(f"Total synergy data: {len(df)} rows")

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
        fps[name] = np.array(AllChem.GetMorganFingerprintAsBitVect(mol, 2, 1024), dtype=np.float32)
logger.info(f"Drug FPs: {len(fps)}")

# ======================================================
# Curated cell-line features (cancer type + mutations)
# For O'Neil 39 cell lines
# ======================================================
CANCER_TYPES = ["colon", "breast", "ovarian", "lung", "melanoma", "prostate", "bladder", "other"]
MUTATIONS = ["KRAS", "BRAF", "PIK3CA", "TP53", "NRAS", "BRCA1", "ERBB2", "ESR1", "MSI", "TNBC", "AR"]

CELLLINE_INFO = {
    "HCT116": ("colon", ["KRAS", "PIK3CA", "MSI"]),
    "SW620": ("colon", ["KRAS"]),
    "DLD1": ("colon", ["KRAS", "PIK3CA", "MSI"]),
    "COLO320DM": ("colon", []),
    "SW837": ("colon", ["KRAS", "TP53"]),
    "RKO": ("colon", ["BRAF", "PIK3CA", "MSI"]),
    "MCF7": ("breast", ["PIK3CA", "ESR1"]),
    "T47D": ("breast", ["PIK3CA", "ESR1"]),
    "EFM192B": ("breast", ["ERBB2"]),
    "MDAMB231": ("breast", ["KRAS", "TP53", "TNBC"]),
    "MDAMB468": ("breast", ["TP53", "TNBC"]),
    "SKOV3": ("ovarian", ["PIK3CA", "ERBB2"]),
    "OVCAR3": ("ovarian", ["TP53"]),
    "OV90": ("ovarian", ["TP53"]),
    "CAOV3": ("ovarian", ["TP53"]),
    "ES2": ("ovarian", ["BRAF"]),
    "PA1": ("ovarian", []),
    "A2780": ("ovarian", []),
    "A427": ("lung", ["KRAS"]),
    "NCIH460": ("lung", ["KRAS", "PIK3CA"]),
    "NCIH23": ("lung", ["KRAS", "TP53"]),
    "NCIH520": ("lung", ["TP53"]),
    "SKMES1": ("lung", ["TP53"]),
    "A2058": ("melanoma", ["BRAF", "TP53"]),
    "A375": ("melanoma", ["BRAF"]),
    "HT144": ("melanoma", ["BRAF", "TP53"]),
    "SKMEL30": ("melanoma", ["NRAS"]),
    "RPMI7951": ("melanoma", ["BRAF"]),
    "UACC62": ("melanoma", ["BRAF"]),
    "VCAP": ("prostate", ["AR"]),
    "UWB1289": ("ovarian", ["BRCA1"]),
    "UWB1289BRCA1": ("ovarian", []),
    "OCUBM": ("bladder", []),
    # DrugComb common cell lines (with hyphenated names mapping)
    "SW620": ("colon", ["KRAS"]),
    "SKOV3": ("ovarian", ["PIK3CA", "ERBB2"]),
    "T47D": ("breast", ["PIK3CA", "ESR1"]),
    "NCIH460": ("lung", ["KRAS", "PIK3CA"]),
    "HCT15": ("colon", ["KRAS", "PIK3CA", "MSI"]),
    "MDAMB231": ("breast", ["KRAS", "TP53", "TNBC"]),
    "DU145": ("prostate", []),
    "PC3": ("prostate", ["TP53"]),
    "U251": ("other", ["TP53", "PTEN"]),
    "ACHN": ("other", []),
    "KM12": ("colon", ["KRAS"]),
    "OVCAR8": ("ovarian", ["TP53"]),
    "OVCAR5": ("ovarian", []),
    "SF268": ("other", ["TP53"]),
    "UACC257": ("melanoma", ["BRAF"]),
    "SKMEL28": ("melanoma", ["BRAF"]),
}

CL_DIM = len(CANCER_TYPES) + len(MUTATIONS)  # 19


def normalize_cl(name):
    return str(name).upper().replace("-", "").replace("_", "").replace(" ", "").replace(".", "")


def get_cl_feat(name):
    norm = normalize_cl(name)
    if norm not in CELLLINE_INFO:
        return None
    ctype, muts = CELLLINE_INFO[norm]
    feat = np.zeros(CL_DIM, dtype=np.float32)
    if ctype in CANCER_TYPES:
        feat[CANCER_TYPES.index(ctype)] = 1.0
    for m in muts:
        if m in MUTATIONS:
            feat[len(CANCER_TYPES) + MUTATIONS.index(m)] = 1.0
    return feat


def build_features(source_filter=None):
    sub = df.copy()
    if source_filter:
        sub = sub[sub["source"] == source_filter]
    
    X_list, y_list = [], []
    n_cl = 0
    for _, row in sub.iterrows():
        da, db = str(row["drug_a"]), str(row["drug_b"])
        if da not in fps or db not in fps:
            continue
        score = float(row["synergy_loewe"])
        if np.isnan(score):
            continue
        
        fp = np.concatenate([fps[da], fps[db]])  # 2048
        cl_feat = get_cl_feat(str(row["cell_line"]))
        if cl_feat is None:
            cl_feat = np.zeros(CL_DIM, dtype=np.float32)
        else:
            n_cl += 1
        x = np.concatenate([fp, cl_feat])
        X_list.append(x)
        y_list.append(score)
    
    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.float32)
    pct = n_cl / max(len(X), 1) * 100
    logger.info(f"  Features: {X.shape}, CL matched: {n_cl}/{len(X)} ({pct:.1f}%)")
    return X, y


def cv_eval(X, y, tag, n_splits=5):
    import xgboost as xgb
    
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    rs = []
    last_model = None
    
    for fold, (ti, vi) in enumerate(kf.split(X)):
        m = xgb.XGBRegressor(
            n_estimators=500, max_depth=8, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
            tree_method="hist", device="cuda:0",
            random_state=42, early_stopping_rounds=30,
        )
        m.fit(X[ti], y[ti], eval_set=[(X[vi], y[vi])], verbose=0)
        yp = m.predict(X[vi])
        r = pearsonr(y[vi], yp)[0]
        rs.append(r)
        last_model = m
        logger.info(f"  {tag} fold {fold+1}: r={r:.4f}")
    
    avg = np.mean(rs)
    std = np.std(rs)
    logger.info(f"  {tag} CV: r={avg:.4f} +/- {std:.4f}")
    
    # Feature importance
    imp = last_model.feature_importances_
    fp_imp = imp[:2048].sum()
    cl_imp = imp[2048:].sum() if len(imp) > 2048 else 0
    total = fp_imp + cl_imp + 1e-10
    if cl_imp > 0:
        logger.info(f"  Feature imp: FP={fp_imp/total*100:.1f}%, CL={cl_imp/total*100:.1f}%")
    
    return {
        "avg_r": round(float(avg), 4),
        "std_r": round(float(std), 4),
        "cl_imp_pct": round(float(cl_imp / total * 100), 1) if cl_imp > 0 else 0,
    }


def main():
    logger.info("=" * 60)
    logger.info("Synergy v6: 4-way comparison")
    logger.info("=" * 60)
    
    # A: O'Neil only, FP only
    logger.info("\n=== A: O'Neil 22K, FP only ===")
    X_a, y_a = build_features("oneil")
    r_a = cv_eval(X_a[:, :2048], y_a, "A")
    
    # B: O'Neil only, FP + cell-line
    logger.info("\n=== B: O'Neil 22K, FP + CellLine ===")
    r_b = cv_eval(X_a, y_a, "B")
    
    # C: All data, FP + cell-line
    logger.info("\n=== C: All 927K, FP + CellLine ===")
    X_c, y_c = build_features(None)
    r_c = cv_eval(X_c, y_c, "C")
    
    # D: All data, FP only (v4 baseline)
    logger.info("\n=== D: All 927K, FP only (v4 baseline) ===")
    r_d = cv_eval(X_c[:, :2048], y_c, "D")
    
    # Results
    logger.info("\n" + "=" * 60)
    logger.info("FINAL COMPARISON:")
    logger.info(f"  D) All 927K, FP only:     r={r_d['avg_r']:.4f} +/- {r_d['std_r']:.4f}")
    logger.info(f"  C) All 927K, FP+CL:       r={r_c['avg_r']:.4f} +/- {r_c['std_r']:.4f}  (CL imp: {r_c['cl_imp_pct']:.1f}%)")
    logger.info(f"  A) O'Neil 22K, FP only:   r={r_a['avg_r']:.4f} +/- {r_a['std_r']:.4f}")
    logger.info(f"  B) O'Neil 22K, FP+CL:     r={r_b['avg_r']:.4f} +/- {r_b['std_r']:.4f}  (CL imp: {r_b['cl_imp_pct']:.1f}%)")
    logger.info("=" * 60)
    
    results = {
        "A_oneil_fp": r_a,
        "B_oneil_fp_cl": r_b,
        "C_all_fp_cl": r_c,
        "D_all_fp": r_d,
        "conclusion": "TBD",
    }
    
    # Determine best
    all_r = {"A": r_a["avg_r"], "B": r_b["avg_r"], "C": r_c["avg_r"], "D": r_d["avg_r"]}
    best = max(all_r, key=all_r.get)
    results["best_config"] = best
    results["best_r"] = all_r[best]
    
    with open(MODEL_DIR / "synergy_v6_comparison.json", "w") as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"\nBest: {best} with r={all_r[best]:.4f}")
    logger.info(f"Saved: synergy_v6_comparison.json")


if __name__ == "__main__":
    main()
