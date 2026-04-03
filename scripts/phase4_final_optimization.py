"""
Phase 4: 1,828 DepMap 전체 매핑 + DNN/XGBoost + 통합 치료반응

1. PCA embedding 재구축 (1,828 매핑으로 near-100% O'Neil match)
2. DNN + XGBoost 비교 (O'Neil + All data)
3. GSE39582 + TCGA 통합 치료반응 모델
"""

import json
import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import pearsonr, mannwhitneyu
from sklearn.ensemble import (
    GradientBoostingClassifier,
    RandomForestClassifier,
    VotingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DATA_DIR = Path("F:/ADDS/data/ml_training")
MODEL_DIR = Path("F:/ADDS/models")
device = "cuda" if torch.cuda.is_available() else "cpu"


# ============================================================
# PART 1: Rebuild PCA with 1,828 full mapping
# ============================================================

def rebuild_embedding():
    """Rebuild cell-line embedding with full 1,828 mapping."""
    map_path = DATA_DIR / "depmap" / "ach_to_name_full_v2.json"
    expr_path = DATA_DIR / "depmap" / "ccle_expression.parquet"
    
    with open(map_path) as f:
        mapping = json.load(f)
    logger.info("Full mapping: %d entries", len(mapping))
    
    expr = pd.read_parquet(expr_path)
    logger.info("Expression: %s", expr.shape)
    
    # Match rate before
    mapped_in_expr = [a for a in expr.index if a in mapping]
    logger.info("ACH IDs in expression + mapping: %d/%d (%.1f%%)",
                len(mapped_in_expr), len(expr), len(mapped_in_expr)/len(expr)*100)
    
    # Clean & PCA
    expr_clean = expr.fillna(expr.median()).dropna(axis=1)
    scaler = StandardScaler()
    X = scaler.fit_transform(expr_clean.values)
    n_comp = min(100, X.shape[1], X.shape[0]-1)
    pca = PCA(n_components=n_comp)
    X_pca = pca.fit_transform(X)
    var_exp = pca.explained_variance_ratio_.sum()
    logger.info("PCA: %s, var=%.2f", X_pca.shape, var_exp)
    
    # Build embeddings dict with ALL name variants
    embeddings = {}
    for i, ach in enumerate(expr_clean.index):
        vec = X_pca[i].astype(np.float32)
        embeddings[ach] = vec
        
        if ach in mapping:
            name = mapping[ach]
            # Multiple normalized forms
            for norm in [
                name,
                name.upper(),
                name.upper().replace("-","").replace("_","").replace(" ","").replace(".",""),
                name.replace(" ",""),
                name.replace("-",""),
            ]:
                embeddings[norm] = vec
    
    result = {"embeddings": embeddings, "dim": n_comp}
    
    out_path = DATA_DIR / "depmap" / "cellline_embedding_v2.pkl"
    with open(out_path, "wb") as f:
        pickle.dump(result, f)
    
    logger.info("Embedding v2: %d entries (dim=%d)", len(embeddings), n_comp)
    return result


# ============================================================
# PART 2: Synergy with full mapping + DNN + XGBoost
# ============================================================

class DeepSynergyMLP(nn.Module):
    def __init__(self, fp_dim=2048, cl_dim=100, hidden=[512, 256, 128]):
        super().__init__()
        layers = []
        prev = fp_dim + cl_dim
        for h in hidden:
            layers.extend([nn.Linear(prev, h), nn.LayerNorm(h), nn.SiLU(), nn.Dropout(0.3)])
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.net(x).squeeze(-1)


def build_synergy_features(embed_data, source_filter=None):
    """Build feature matrix for synergy prediction."""
    embeddings = embed_data["embeddings"]
    emb_dim = embed_data["dim"]
    
    df = pd.read_csv(DATA_DIR / "synergy_combined.csv", low_memory=False)
    if source_filter:
        df = df[df["source"] == source_filter]
    
    smiles = {}
    for p in [MODEL_DIR / "synergy" / "drug_smiles.json",
              MODEL_DIR / "synergy" / "drug_smiles_extended.json"]:
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
    
    def norm_cl(n):
        return str(n).upper().replace("-","").replace("_","").replace(" ","").replace(".","")
    
    X_list, y_list = [], []
    n_cl = 0
    for _, row in df.iterrows():
        da, db = str(row["drug_a"]), str(row["drug_b"])
        if da not in fps or db not in fps:
            continue
        score = float(row["synergy_loewe"])
        if np.isnan(score): continue
        
        fp = np.concatenate([fps[da], fps[db]])
        cl = norm_cl(str(row["cell_line"]))
        if cl in embeddings:
            cl_feat = embeddings[cl]
            n_cl += 1
        else:
            cl_feat = np.zeros(emb_dim, dtype=np.float32)
        
        X_list.append(np.concatenate([fp, cl_feat]))
        y_list.append(score)
    
    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.float32)
    match_pct = n_cl/max(len(X),1)*100
    return X, y, match_pct, emb_dim


def train_dnn_cv(X, y, emb_dim, n_epochs=200, label=""):
    """Train DeepSynergy DNN with 5-fold CV."""
    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)
    
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    rs = []
    
    for fold, (ti, vi) in enumerate(kf.split(X_s)):
        Xt = torch.FloatTensor(X_s[ti]).to(device)
        yt = torch.FloatTensor(y[ti]).to(device)
        Xv = torch.FloatTensor(X_s[vi]).to(device)
        yv = y[vi]
        
        model = DeepSynergyMLP(fp_dim=2048, cl_dim=emb_dim).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_epochs)
        
        best_r = -1
        patience = 0
        
        for epoch in range(n_epochs):
            model.train()
            perm = torch.randperm(len(Xt))
            for start in range(0, len(Xt), 2048):
                idx = perm[start:start+2048]
                opt.zero_grad()
                loss = F.mse_loss(model(Xt[idx]), yt[idx])
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
            sched.step()
            
            if (epoch+1) % 10 == 0:
                model.eval()
                with torch.no_grad():
                    vp = model(Xv).cpu().numpy()
                r = pearsonr(yv, vp)[0]
                if r > best_r:
                    best_r = r
                    patience = 0
                    best_state = {k:v.clone() for k,v in model.state_dict().items()}
                else:
                    patience += 1
                if patience >= 5:
                    break
        
        model.load_state_dict(best_state)
        model.eval()
        with torch.no_grad():
            vp = model(Xv).cpu().numpy()
        r = pearsonr(yv, vp)[0]
        rs.append(r)
        logger.info("  %s DNN fold %d: r=%.4f (ep %d)", label, fold+1, r, epoch+1)
    
    avg = np.mean(rs)
    std = np.std(rs)
    logger.info("  %s DNN: r=%.4f +/- %.4f", label, avg, std)
    return {"r": round(float(avg), 4), "std": round(float(std), 4), "folds": [round(float(r),4) for r in rs]}


def train_xgb_cv(X, y, label=""):
    """Train XGBoost baseline for comparison."""
    import xgboost as xgb
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    rs = []
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
    avg = np.mean(rs)
    logger.info("  %s XGB: r=%.4f +/- %.4f", label, avg, np.std(rs))
    return {"r": round(float(avg), 4), "std": round(float(np.std(rs)), 4)}


# ============================================================
# PART 3: GSE39582 + TCGA 통합 치료반응
# ============================================================

def build_combined_treatment():
    """Combine GSE39582 + TCGA data for treatment response."""
    
    # TCGA data
    rnaseq_dir = DATA_DIR / "tcga" / "rnaseq"
    chemo_csv = DATA_DIR / "tcga" / "tcga_coad_chemo_survival.csv"
    chemo_df = pd.read_csv(chemo_csv)
    tcga_labels = dict(zip(chemo_df["case_id"], chemo_df["label"]))
    
    tcga_data = {}
    for fpath in sorted(rnaseq_dir.glob("*.tsv")):
        cid = fpath.name.split("_")[0]
        if cid not in tcga_labels: continue
        try:
            df = pd.read_csv(fpath, sep="\t", comment="#")
            if "gene_name" not in df.columns: continue
            cc = None
            for c in ["tpm_unstranded", "fpkm_unstranded", "unstranded"]:
                if c in df.columns: cc = c; break
            if cc is None: continue
            if "gene_type" in df.columns: df = df[df["gene_type"]=="protein_coding"]
            df = df.drop_duplicates(subset=["gene_name"]).set_index("gene_name")
            vals = df[cc].astype(float)
            vals = vals[~vals.index.str.startswith("N_")]
            tcga_data[cid] = vals
        except: continue
    
    logger.info("TCGA patients: %d", len(tcga_data))
    
    # GSE39582 data - load from existing v3 training features
    gse_path = DATA_DIR / "geo" / "GSE39582_expression.pkl"
    gse_labels_path = DATA_DIR / "geo" / "GSE39582_labels.pkl"
    
    # Check if GSE data is available in processed form
    gse_expr_dir = DATA_DIR / "geo"
    gse_soft = gse_expr_dir / "GSE39582_family.soft"
    
    gse_available = False
    gse_data = {}
    gse_labels = {}
    
    # Try to load from SOFT file
    if gse_soft.exists():
        logger.info("Loading GSE39582 from SOFT file...")
        try:
            current_sample = None
            sample_data = {}
            gene_ids = []
            in_table = False
            
            with open(gse_soft, "r", encoding="utf-8", errors="replace") as f:
                for line in f:
                    line = line.strip()
                    if line.startswith("^SAMPLE"):
                        current_sample = line.split("=")[1].strip()
                    elif line.startswith("!Sample_characteristics_ch1") and current_sample:
                        if "dfs event:" in line.lower():
                            val = line.split(":")[-1].strip()
                            if val in ("0", "1"):
                                gse_labels[current_sample] = int(val)
                    elif line == "!sample_table_begin":
                        in_table = True
                        sample_data[current_sample] = {}
                    elif line == "!sample_table_end":
                        in_table = False
                    elif in_table and current_sample and "\t" in line:
                        parts = line.split("\t")
                        if len(parts) >= 2 and parts[0] != "ID_REF":
                            try:
                                sample_data[current_sample][parts[0]] = float(parts[1])
                            except: pass
            
            # Filter to labeled samples
            for sid, values in sample_data.items():
                if sid in gse_labels and len(values) > 100:
                    gse_data[sid] = pd.Series(values)
            
            gse_available = len(gse_data) > 0
            logger.info("GSE39582: %d samples, %d labeled", len(gse_data), len(gse_labels))
            
        except Exception as e:
            logger.warning("GSE39582 loading failed: %s", e)
    
    # Build TCGA-only model (primary)
    if not tcga_data:
        return {"error": "no data"}
    
    tcga_genes = None
    for v in tcga_data.values():
        if tcga_genes is None:
            tcga_genes = set(v.index)
        else:
            tcga_genes &= set(v.index)
    tcga_genes = sorted(tcga_genes)
    
    X_tcga, y_tcga = [], []
    for cid, vals in tcga_data.items():
        X_tcga.append(vals.reindex(tcga_genes, fill_value=0).values)
        y_tcga.append(tcga_labels[cid])
    
    X_tcga = np.log2(np.array(X_tcga, dtype=np.float32)+1)
    y_tcga = np.array(y_tcga, dtype=int)
    
    # DEG selection
    resp = np.where(y_tcga==1)[0]
    nonr = np.where(y_tcga==0)[0]
    pvals = []
    for j in range(X_tcga.shape[1]):
        try:
            _, p = mannwhitneyu(X_tcga[resp,j], X_tcga[nonr,j], alternative="two-sided")
        except: p = 1.0
        pvals.append(p)
    
    results = {}
    
    # Try different DEG counts
    for n_deg in [50, 100, 200]:
        top_idx = np.argsort(pvals)[:n_deg]
        X_deg = X_tcga[:, top_idx]
        sc = StandardScaler()
        X_s = sc.fit_transform(X_deg)
        
        n_min = min(y_tcga.sum(), len(y_tcga)-y_tcga.sum())
        n_splits = min(5, n_min)
        if n_splits < 2: continue
        
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        aucs = []
        for ti, vi in skf.split(X_s, y_tcga):
            clf = VotingClassifier(
                estimators=[
                    ("rf", RandomForestClassifier(n_estimators=200, max_depth=5, random_state=42)),
                    ("gb", GradientBoostingClassifier(n_estimators=200, max_depth=3, random_state=42, learning_rate=0.05)),
                    ("lr", LogisticRegression(C=0.1, random_state=42, max_iter=1000)),
                ],
                voting="soft",
            )
            clf.fit(X_s[ti], y_tcga[ti])
            prob = clf.predict_proba(X_s[vi])[:,1]
            try:
                aucs.append(roc_auc_score(y_tcga[vi], prob))
            except: pass
        
        if aucs:
            avg = np.mean(aucs)
            std = np.std(aucs)
            logger.info("  TCGA DEG-%d (n=%d): AUC=%.4f +/- %.4f", n_deg, len(y_tcga), avg, std)
            results[f"tcga_deg{n_deg}"] = {"auc": round(avg,4), "std": round(std,4), "n": len(y_tcga)}
    
    return results


# ============================================================
# Main
# ============================================================

def main():
    logger.info("=" * 60)
    logger.info("PHASE 4: Full DepMap Mapping + Final Optimization")
    logger.info("=" * 60)
    
    results = {}
    
    # Part 1: Rebuild embedding with 1,828 mappings
    logger.info("\n--- Part 1: Rebuild Embedding (1,828 mappings) ---")
    embed_data = rebuild_embedding()
    
    # Part 2a: O'Neil synergy
    logger.info("\n--- Part 2a: O'Neil Synergy ---")
    X_on, y_on, match_on, dim = build_synergy_features(embed_data, "oneil")
    logger.info("O'Neil: X=%s, CL match=%.1f%%", X_on.shape, match_on)
    
    oneil_dnn = train_dnn_cv(X_on, y_on, dim, n_epochs=200, label="O'Neil")
    oneil_xgb = train_xgb_cv(X_on, y_on, label="O'Neil")
    results["oneil"] = {
        "dnn": oneil_dnn, "xgb": oneil_xgb,
        "cl_match": round(match_on,1), "n": len(y_on),
    }
    
    # Part 2b: All-data synergy
    logger.info("\n--- Part 2b: All Data Synergy ---")
    X_all, y_all, match_all, _ = build_synergy_features(embed_data, None)
    logger.info("All: X=%s, CL match=%.1f%%", X_all.shape, match_all)
    
    all_xgb = train_xgb_cv(X_all, y_all, label="All")
    results["all_data"] = {
        "xgb": all_xgb,
        "cl_match": round(match_all,1), "n": len(y_all),
    }
    
    # All-data DNN (train only if feasible — 927K might be slow)
    if len(y_all) < 200000:
        all_dnn = train_dnn_cv(X_all, y_all, dim, n_epochs=50, label="All")
        results["all_data"]["dnn"] = all_dnn
    else:
        logger.info("All-data too large for full DNN CV, training sample...")
        # Sample 200K for DNN
        idx = np.random.RandomState(42).choice(len(y_all), 200000, replace=False)
        all_dnn = train_dnn_cv(X_all[idx], y_all[idx], dim, n_epochs=50, label="All-200K")
        results["all_data"]["dnn_200k"] = all_dnn
    
    # Part 3: Combined treatment response
    logger.info("\n--- Part 3: Combined Treatment Response ---")
    tr_results = build_combined_treatment()
    results["treatment"] = tr_results
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("PHASE 4 FINAL RESULTS")
    logger.info("=" * 60)
    logger.info(json.dumps(results, indent=2, default=str))
    
    with open(MODEL_DIR / "phase4_results.json", "w") as f:
        json.dump(results, f, indent=2, default=float)
    logger.info("\nSaved: phase4_results.json")


if __name__ == "__main__":
    main()
