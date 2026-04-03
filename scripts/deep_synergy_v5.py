"""
DeepSynergy v5 — Expanded CL Expression Embedding
====================================================
v4: 38 CLs (9.3% match) → drug-pair r=0.653
v5: 752 CLs (38.5% match) → expecting r≥0.67

Uses cell_line_expression_256_expanded.pkl from full DepMap CCLE expression.
"""

import json, logging, time, pickle
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.stats import pearsonr
from sklearn.model_selection import GroupKFold, KFold
from sklearn.preprocessing import StandardScaler
from rdkit import Chem
from rdkit.Chem import AllChem

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                    handlers=[logging.StreamHandler()])
for h in logging.getLogger().handlers:
    h.flush = lambda: None  # force immediate flush
logger = logging.getLogger(__name__)

DATA_DIR = Path("F:/ADDS/data/ml_training")
MODEL_DIR = Path("F:/ADDS/models/synergy")
device = "cuda" if torch.cuda.is_available() else "cpu"
BS = 4096

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


class DeepSynergyV5(nn.Module):
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


def get_tissue_vec(cl_name):
    vec = np.zeros(len(TISSUE_KEYWORDS), dtype=np.float32)
    cl = cl_name.upper().replace('-','').replace('_','').replace(' ','')
    for i, (tissue, kws) in enumerate(TISSUE_KEYWORDS.items()):
        for kw in kws:
            if kw.replace('-','') in cl:
                vec[i] = 1.0
                break
    return vec


def predict_batched(model, X_np):
    model.eval()
    preds = []
    with torch.no_grad():
        for s in range(0, len(X_np), BS):
            batch = torch.FloatTensor(X_np[s:s+BS]).to(device)
            preds.append(model(batch).cpu().numpy())
            del batch
    return np.concatenate(preds)


def main():
    t0 = time.time()
    logger.info("=" * 60)
    logger.info("DeepSynergy v5 -- Expanded CL Expression")
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
        if smi is None: continue
        mol = Chem.MolFromSmiles(smi)
        if mol:
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, 1024)
            fps[name.upper()] = np.array(fp, dtype=np.float32)
    logger.info("Drug FPs: %d drugs", len(fps))
    
    # Load EXPANDED CL expression
    pkl_path = DATA_DIR / "cell_line_expression_256_expanded.pkl"
    with open(pkl_path, 'rb') as f:
        cl_map = pickle.load(f)
    cl_dim = 256
    logger.info("CL expression (expanded): %d name entries, %dd", len(cl_map), cl_dim)
    
    # Build features
    logger.info("Building features...")
    X_list, y_list, groups = [], [], []
    cl_matched = 0
    cl_total = 0
    tissue_dim = len(TISSUE_KEYWORDS)
    
    for _, row in df.iterrows():
        da, db = row['drug_a'], row['drug_b']
        if da not in fps or db not in fps:
            continue
        
        cl = row['cell_line']
        cl_upper = cl.upper().strip()
        cl_clean = cl_upper.replace('-','').replace('_','').replace(' ','')
        cl_total += 1
        
        # Try exact, then cleaned match
        cl_expr = cl_map.get(cl_upper)
        if cl_expr is None:
            cl_expr = cl_map.get(cl_clean)
        tissue = get_tissue_vec(cl)
        
        if cl_expr is not None:
            cl_matched += 1
            feat = np.concatenate([fps[da], fps[db], cl_expr, tissue])
        else:
            feat = np.concatenate([fps[da], fps[db], np.zeros(cl_dim, dtype=np.float32), tissue])
        
        X_list.append(feat)
        y_list.append(row['synergy_loewe'])
        groups.append(tuple(sorted([da, db])))
    
    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.float32)
    
    mask = ~np.isnan(y)
    X, y = X[mask], y[mask]
    groups_clean = [groups[i] for i in range(len(mask)) if mask[i]]
    
    unique_pairs = list(set(groups_clean))
    pair_to_id = {p: i for i, p in enumerate(unique_pairs)}
    group_ids = np.array([pair_to_id[g] for g in groups_clean])
    
    input_dim = X.shape[1]
    match_pct = 100*cl_matched/max(cl_total,1)
    logger.info("Features: %d samples, %dd", len(X), input_dim)
    logger.info("CL match: %d/%d (%.1f%%) [v4 was 9.3%%]", cl_matched, cl_total, match_pct)
    logger.info("Drug pairs: %d", len(unique_pairs))
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # ================================================================
    # Random 5-fold CV
    # ================================================================
    logger.info("\n--- Random 5-fold CV ---")
    kf = KFold(5, shuffle=True, random_state=42)
    random_results = []
    
    for fold, (ti, vi) in enumerate(kf.split(X_scaled)):
        torch.cuda.empty_cache()
        model = DeepSynergyV5(input_dim).to(device)
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
            model.load_state_dict(best_state); model.to(device)
        vp = predict_batched(model, X_val)
        r = pearsonr(y_val, vp)[0]
        random_results.append(r)
        logger.info("  Fold %d: r=%.4f (n=%d)", fold+1, r, len(vi))
        del model; torch.cuda.empty_cache()
    
    logger.info("  Random CV: r=%.4f +/- %.4f", np.mean(random_results), np.std(random_results))
    
    # ================================================================
    # Drug-pair holdout 5-fold CV
    # ================================================================
    logger.info("\n--- Drug-pair holdout CV ---")
    gkf = GroupKFold(5)
    dp_results = []
    best_model_state = None
    best_dp_r = -1
    
    for fold, (ti, vi) in enumerate(gkf.split(X_scaled, y, group_ids)):
        torch.cuda.empty_cache()
        model = DeepSynergyV5(input_dim).to(device)
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
                    logger.info("  Epoch %3d: val_r=%.4f", ep+1, r)
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
            model.load_state_dict(best_state); model.to(device)
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
    
    # Save
    if best_model_state:
        save_model = DeepSynergyV5(input_dim)
        save_model.load_state_dict(best_model_state)
        torch.save(save_model.state_dict(), MODEL_DIR / "deep_synergy_v5.pt")
        
        results = {
            "model": "DeepSynergy v5",
            "input_dim": input_dim,
            "features": f"FP(1024x2) + CL_expr({cl_dim}d expanded) + tissue({tissue_dim}d)",
            "cl_coverage": int(cl_matched),
            "cl_total": int(cl_total),
            "cl_match_pct": round(match_pct, 1),
            "training_samples": int(len(X)),
            "unique_drug_pairs": int(len(unique_pairs)),
            "random_cv": {"r": round(float(np.mean(random_results)), 4),
                         "std": round(float(np.std(random_results)), 4)},
            "drug_pair_cv": {"r": round(float(dp_mean), 4),
                            "std": round(float(dp_std), 4)},
            "training_time_seconds": round(time.time() - t0, 1),
            "comparison": {
                "v3_drug_pair": 0.6044,
                "v4_drug_pair": 0.6533,
                "v4_cl_match_pct": 9.3,
            },
        }
        
        with open(MODEL_DIR / "deep_synergy_v5_results.json", "w") as f:
            json.dump(results, f, indent=2)
    
    elapsed = time.time() - t0
    logger.info("\n" + "=" * 60)
    logger.info("FINAL RESULTS -- DeepSynergy v5")
    logger.info("=" * 60)
    logger.info("  CL match:     %d/%d (%.1f%%) [v4 was 9.3%%]", cl_matched, cl_total, match_pct)
    logger.info("  Random CV:    r=%.4f +/- %.4f [v4: 0.6866]", np.mean(random_results), np.std(random_results))
    logger.info("  Drug-pair CV: r=%.4f +/- %.4f [v4: 0.6533]", dp_mean, dp_std)
    logger.info("  Delta vs v4:  Random=%+.4f, Drug-pair=%+.4f",
                np.mean(random_results)-0.6866, dp_mean-0.6533)
    logger.info("  Delta vs v3:  Random=%+.4f, Drug-pair=%+.4f",
                np.mean(random_results)-0.6644, dp_mean-0.6044)
    logger.info("  Time: %.0f seconds", elapsed)
    logger.info("  Saved: %s", MODEL_DIR / "deep_synergy_v5.pt")

if __name__ == "__main__":
    main()
