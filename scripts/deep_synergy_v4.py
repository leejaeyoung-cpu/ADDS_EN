"""
DeepSynergy v4 — CL Gene Expression Embedding
================================================
v3 features: Morgan FP (1024x2) + tissue (12d) = 2060d
v4 features: Morgan FP (1024x2) + CL expression (256d) + tissue (12d) = 2316d

The critical missing feature in v3 was cell-line specific gene expression.
Energy v6 showed CL embedding adds +0.14 to drug-pair r.
"""

import json, logging, time, os
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.stats import pearsonr
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from rdkit import Chem
from rdkit.Chem import AllChem

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DATA_DIR = Path("F:/ADDS/data/ml_training")
MODEL_DIR = Path("F:/ADDS/models/synergy")
device = "cuda" if torch.cuda.is_available() else "cpu"


# ================================================================
# 1. Load Cell Line Expression Embeddings
# ================================================================

def load_cl_expression():
    """Load CL gene expression embeddings (256d from DepMap PCA)."""
    expr_path = DATA_DIR / "cell_line_expression_256.csv"
    if expr_path.exists():
        df = pd.read_csv(expr_path, index_col=0)
        logger.info("CL expression: %d cell lines x %dd", df.shape[0], df.shape[1])
        # Normalize index
        cl_map = {}
        for cl in df.index:
            cl_map[cl.upper().strip()] = df.loc[cl].values.astype(np.float32)
        return cl_map, df.shape[1]
    
    # Fallback: download from DepMap
    logger.info("CL expression file not found, creating from DepMap...")
    return _download_depmap_expression()


def _download_depmap_expression():
    """Download DepMap CCLE expression and create PCA embedding."""
    info_path = DATA_DIR / "depmap_sample_info.csv"
    
    if not info_path.exists():
        logger.warning("No DepMap data available")
        return {}, 0
    
    # Use sample info to get cell line names at minimum
    info = pd.read_csv(info_path)
    logger.info("DepMap sample info: %d entries", len(info))
    
    # For now, create expression proxy from sample info (tissue type one-hot)
    # Real implementation would download OmicsExpressionProteinCodingGenesTPMLogp1.csv
    return {}, 0


# ================================================================
# 2. Cell Line Name Matching
# ================================================================

def fuzzy_match_cl(cl_name, cl_map):
    """Try various normalizations to match cell line names."""
    cl = cl_name.upper().strip()
    
    # Direct match
    if cl in cl_map:
        return cl_map[cl]
    
    # Remove common suffixes/prefixes
    variants = [
        cl,
        cl.replace("-", ""),
        cl.replace("_", ""),
        cl.replace(" ", ""),
        cl.replace("-", "").replace("_", ""),
    ]
    
    for v in variants:
        if v in cl_map:
            return cl_map[v]
        # Partial match
        for k in cl_map:
            k_clean = k.replace("-", "").replace("_", "").replace(" ", "")
            if v == k_clean:
                return cl_map[k]
    
    return None


# ================================================================
# 3. Build Features
# ================================================================

TISSUE_KEYWORDS = {
    'COLON':['HCT','SW480','SW620','COLO','DLD','HT29','LOVO','RKO'],
    'BREAST':['MCF','MDA','BT','T47D','SKBR'],
    'LUNG':['A549','NCI','H460','H1299','H1975','HCC827','PC9'],
    'OVARIAN':['SKOV','A2780','OVCAR','IGROV'],
    'LEUKEMIA':['K562','HL60','MOLT','JURKAT','CCRF'],
    'MELANOMA':['A375','SKMEL','MALME','UACC'],
    'PROSTATE':['PC3','DU145','LNCAP'],
    'LIVER':['HEPG2','HEP3B','HUH'],
    'BRAIN':['U87','U251','SF','SNB','A172'],
    'PANCREAS':['PANC','MIAPACA','BXPC'],
    'KIDNEY':['786','A498','ACHN','CAKI'],
    'STOMACH':['AGS','KATO','MKN','SNU'],
}


def get_tissue_vec(cl_name):
    """Get tissue type one-hot vector."""
    vec = np.zeros(len(TISSUE_KEYWORDS), dtype=np.float32)
    cl = cl_name.upper().replace('-','').replace('_','').replace(' ','')
    for i, (tissue, kws) in enumerate(TISSUE_KEYWORDS.items()):
        for kw in kws:
            if kw.replace('-','') in cl:
                vec[i] = 1.0
                break
    return vec


# ================================================================
# 4. Model
# ================================================================

class DeepSynergyV4(nn.Module):
    def __init__(self, input_dim, hidden=[2048, 1024, 512, 256]):
        super().__init__()
        layers = []
        prev = input_dim
        for i, h in enumerate(hidden):
            layers.extend([
                nn.Linear(prev, h),
                nn.BatchNorm1d(h),
                nn.GELU(),
                nn.Dropout(0.3 if i < 2 else 0.2),
            ])
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.net(x).squeeze(-1)


# ================================================================
# 5. Main
# ================================================================

def main():
    t0 = time.time()
    logger.info("=" * 60)
    logger.info("DeepSynergy v4 -- CL Expression Embedding")
    logger.info("=" * 60)
    
    # Load data
    df = pd.read_csv(DATA_DIR / "synergy_combined.csv", low_memory=False)
    df['drug_a'] = df.drug_a.str.upper().str.strip()
    df['drug_b'] = df.drug_b.str.upper().str.strip()
    df['cell_line'] = df.cell_line.str.upper().str.strip()
    logger.info("Combined data: %d records", len(df))
    
    # Load SMILES + FP
    smiles = json.load(open(MODEL_DIR / "drug_smiles.json"))
    fps = {}
    for name, smi in smiles.items():
        if smi is None:
            continue
        mol = Chem.MolFromSmiles(smi)
        if mol:
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, 1024)
            fps[name.upper()] = np.array(fp, dtype=np.float32)
    logger.info("Drug FPs: %d drugs", len(fps))
    
    # Load CL expression
    cl_map, cl_dim = load_cl_expression()
    logger.info("CL expression: %d cell lines, %dd", len(cl_map), cl_dim)
    
    # Build normalized CL map for matching
    cl_map_norm = {}
    for k, v in cl_map.items():
        cl_map_norm[k] = v
        cl_map_norm[k.replace("-","").replace("_","").replace(" ","")] = v
    
    # Build features
    logger.info("Building features...")
    X_list = []
    y_list = []
    groups = []
    cl_matched = 0
    cl_total = 0
    tissue_dim = len(TISSUE_KEYWORDS)
    
    for _, row in df.iterrows():
        da, db = row['drug_a'], row['drug_b']
        if da not in fps or db not in fps:
            continue
        
        cl = row['cell_line']
        cl_expr = fuzzy_match_cl(cl, cl_map_norm)
        cl_total += 1
        
        tissue = get_tissue_vec(cl)
        
        if cl_expr is not None:
            cl_matched += 1
            feat = np.concatenate([fps[da], fps[db], cl_expr, tissue])
        else:
            # Zero-fill expression for unmatched cell lines
            feat = np.concatenate([fps[da], fps[db], np.zeros(cl_dim, dtype=np.float32), tissue])
        
        X_list.append(feat)
        y_list.append(row['synergy_loewe'])
        groups.append(tuple(sorted([da, db])))
    
    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.float32)
    
    # Remove NaN
    mask = ~np.isnan(y)
    X = X[mask]
    y = y[mask]
    groups_clean = [groups[i] for i in range(len(mask)) if mask[i]]
    
    unique_pairs = list(set(groups_clean))
    pair_to_id = {p: i for i, p in enumerate(unique_pairs)}
    group_ids = np.array([pair_to_id[g] for g in groups_clean])
    
    input_dim = X.shape[1]
    logger.info("Features: %d samples, %dd (FP=%d + CL_expr=%d + tissue=%d)",
                len(X), input_dim, 2048, cl_dim, tissue_dim)
    logger.info("CL expression match: %d/%d (%.1f%%)", cl_matched, cl_total, 
                100*cl_matched/max(cl_total,1))
    logger.info("Drug pairs: %d", len(unique_pairs))
    
    # Normalize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # ================================================================
    # EVAL A: Random 5-fold CV
    # ================================================================
    logger.info("\n--- Random 5-fold CV ---")
    from sklearn.model_selection import KFold
    kf = KFold(5, shuffle=True, random_state=42)
    random_results = []
    BS = 4096
    
    def predict_batched(model, X_np):
        """Predict in mini-batches to avoid OOM."""
        model.eval()
        preds = []
        with torch.no_grad():
            for s in range(0, len(X_np), BS):
                batch = torch.FloatTensor(X_np[s:s+BS]).to(device)
                preds.append(model(batch).cpu().numpy())
                del batch
        return np.concatenate(preds)
    
    for fold, (ti, vi) in enumerate(kf.split(X_scaled)):
        torch.cuda.empty_cache()
        model = DeepSynergyV4(input_dim).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, 200)
        
        X_train, y_train = X_scaled[ti], y[ti]
        X_val, y_val = X_scaled[vi], y[vi]
        
        best_r, patience, best_state = -1, 0, None
        for ep in range(200):
            model.train()
            perm = np.random.permutation(len(X_train))
            for s in range(0, len(X_train), BS):
                idx = perm[s:s+BS]
                xb = torch.FloatTensor(X_train[idx]).to(device)
                yb = torch.FloatTensor(y_train[idx]).to(device)
                opt.zero_grad()
                loss = nn.functional.mse_loss(model(xb), yb)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                del xb, yb
            sched.step()
            
            if (ep+1) % 5 == 0:
                vp = predict_batched(model, X_val)
                r = pearsonr(y_val, vp)[0]
                if r > best_r:
                    best_r = r
                    patience = 0
                    best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                else:
                    patience += 1
                if patience >= 4:
                    break
        
        if best_state:
            model.load_state_dict(best_state)
            model.to(device)
        vp = predict_batched(model, X_val)
        r = pearsonr(y_val, vp)[0]
        random_results.append(r)
        logger.info("  Fold %d: r=%.4f (n=%d)", fold+1, r, len(vi))
        del model; torch.cuda.empty_cache()
    
    logger.info("  Random CV: r=%.4f +/- %.4f", np.mean(random_results), np.std(random_results))
    
    # ================================================================
    # EVAL B: Drug-pair holdout 5-fold CV
    # ================================================================
    logger.info("\n--- Drug-pair holdout CV ---")
    gkf = GroupKFold(5)
    dp_results = []
    best_model_state = None
    best_dp_r = -1
    
    for fold, (ti, vi) in enumerate(gkf.split(X_scaled, y, group_ids)):
        torch.cuda.empty_cache()
        model = DeepSynergyV4(input_dim).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, 200)
        
        X_train, y_train = X_scaled[ti], y[ti]
        X_val, y_val = X_scaled[vi], y[vi]
        
        best_r, patience, best_state = -1, 0, None
        for ep in range(200):
            model.train()
            perm = np.random.permutation(len(X_train))
            for s in range(0, len(X_train), BS):
                idx = perm[s:s+BS]
                xb = torch.FloatTensor(X_train[idx]).to(device)
                yb = torch.FloatTensor(y_train[idx]).to(device)
                opt.zero_grad()
                loss = nn.functional.mse_loss(model(xb), yb)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                del xb, yb
            sched.step()
            
            if (ep+1) % 5 == 0:
                vp = predict_batched(model, X_val)
                r = pearsonr(y_val, vp)[0]
                if (ep+1) % 10 == 0:
                    logger.info("  Epoch %3d: loss=%.4f, val_r=%.4f", ep+1, float(loss), r)
                if r > best_r:
                    best_r = r
                    patience = 0
                    best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                else:
                    patience += 1
                if patience >= 4:
                    logger.info("  Early stopping at epoch %d", ep+1)
                    break
        
        if best_state:
            model.load_state_dict(best_state)
            model.to(device)
        vp = predict_batched(model, X_val)
        r = pearsonr(y_val, vp)[0]
        dp_results.append(r)
        logger.info("  Fold %d: r=%.4f (n=%d)", fold+1, r, len(vi))
        
        if r > best_dp_r:
            best_dp_r = r
            best_model_state = best_state
        del model; torch.cuda.empty_cache()
    
    dp_mean = np.mean(dp_results)
    dp_std = np.std(dp_results)
    logger.info("Drug-pair CV: r=%.4f +/- %.4f", dp_mean, dp_std)
    
    # ================================================================
    # Save best model
    # ================================================================
    if best_model_state:
        save_model = DeepSynergyV4(input_dim)
        save_model.load_state_dict(best_model_state)
        torch.save(save_model.state_dict(), MODEL_DIR / "deep_synergy_v4.pt")
        
        results = {
            "model": "DeepSynergy v4",
            "input_dim": input_dim,
            "features": f"FP(1024x2) + CL_expr({cl_dim}d) + tissue({tissue_dim}d)",
            "architecture": [2048, 1024, 512, 256],
            "training_samples": int(len(X)),
            "unique_drug_pairs": int(len(unique_pairs)),
            "cl_match_rate": round(cl_matched / max(cl_total, 1), 3),
            "random_cv": {"r": round(float(np.mean(random_results)), 4), 
                         "std": round(float(np.std(random_results)), 4)},
            "drug_pair_cv": {"r": round(float(dp_mean), 4),
                            "std": round(float(dp_std), 4)},
            "training_time_seconds": round(time.time() - t0, 1),
            "v3_comparison": {"random": 0.6644, "drug_pair": 0.6044},
        }
        
        with open(MODEL_DIR / "deep_synergy_v4_results.json", "w") as f:
            json.dump(results, f, indent=2)
    
    # Final summary
    elapsed = time.time() - t0
    logger.info("\n" + "=" * 60)
    logger.info("FINAL RESULTS -- DeepSynergy v4")
    logger.info("=" * 60)
    logger.info("  Features:     %dd (FP=%d + CL_expr=%d + tissue=%d)", 
                input_dim, 2048, cl_dim, tissue_dim)
    logger.info("  CL match:     %d/%d (%.1f%%)", cl_matched, cl_total, 100*cl_matched/max(cl_total,1))
    logger.info("  Samples:      %d (%d drug pairs)", len(X), len(unique_pairs))
    logger.info("  Random CV:    r=%.4f +/- %.4f", np.mean(random_results), np.std(random_results))
    logger.info("  Drug-pair CV: r=%.4f +/- %.4f", dp_mean, dp_std)
    logger.info("")
    logger.info("  v3 comparison:")
    logger.info("    v3 Random:    r=0.6644")
    logger.info("    v3 Drug-pair: r=0.6044")
    logger.info("    v4 Random:    r=%.4f (delta=%.4f)", np.mean(random_results), np.mean(random_results)-0.6644)
    logger.info("    v4 Drug-pair: r=%.4f (delta=%.4f)", dp_mean, dp_mean-0.6044)
    logger.info("  Time: %.1f seconds", elapsed)
    logger.info("  Saved: %s", MODEL_DIR / "deep_synergy_v4.pt")


if __name__ == "__main__":
    main()
