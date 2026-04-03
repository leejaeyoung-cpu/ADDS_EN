"""
DeepSynergy v5 — Fold 5 only (memory-safe)
==========================================
Previous run completed folds 1-4 but OOM on fold 5.
This script runs ONLY fold 5 with garbage collection between steps.
"""

import json, logging, time, pickle, gc
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
    logger.info("DeepSynergy v5 -- Fold 5 Only")
    logger.info("=" * 60)
    
    # Load data
    df = pd.read_csv(DATA_DIR / "synergy_combined.csv", low_memory=False)
    df['drug_a'] = df.drug_a.str.upper().str.strip()
    df['drug_b'] = df.drug_b.str.upper().str.strip()
    df['cell_line'] = df.cell_line.str.upper().str.strip()
    
    # Load FP
    smiles = json.load(open(MODEL_DIR / "drug_smiles.json"))
    fps = {}
    for name, smi in smiles.items():
        if smi is None: continue
        mol = Chem.MolFromSmiles(smi)
        if mol:
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, 1024)
            fps[name.upper()] = np.array(fp, dtype=np.float32)
    
    # Load expanded CL
    with open(DATA_DIR / "cell_line_expression_256_expanded.pkl", 'rb') as f:
        cl_map = pickle.load(f)
    cl_dim = 256
    
    # Build features
    X_list, y_list, groups = [], [], []
    for _, row in df.iterrows():
        da, db = row['drug_a'], row['drug_b']
        if da not in fps or db not in fps: continue
        cl = row['cell_line']
        cl_upper = cl.upper().strip()
        cl_clean = cl_upper.replace('-','').replace('_','').replace(' ','')
        cl_expr = cl_map.get(cl_upper)
        if cl_expr is None: cl_expr = cl_map.get(cl_clean)
        tissue = get_tissue_vec(cl)
        if cl_expr is not None:
            feat = np.concatenate([fps[da], fps[db], cl_expr, tissue])
        else:
            feat = np.concatenate([fps[da], fps[db], np.zeros(cl_dim, dtype=np.float32), tissue])
        X_list.append(feat)
        y_list.append(row['synergy_loewe'])
        groups.append(tuple(sorted([da, db])))
    
    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.float32)
    del X_list, y_list; gc.collect()
    
    mask = ~np.isnan(y)
    X, y = X[mask], y[mask]
    groups_clean = [groups[i] for i in range(len(mask)) if mask[i]]
    del mask; gc.collect()
    
    unique_pairs = list(set(groups_clean))
    pair_to_id = {p: i for i, p in enumerate(unique_pairs)}
    group_ids = np.array([pair_to_id[g] for g in groups_clean])
    
    input_dim = X.shape[1]
    logger.info("Features: %d samples, %dd", len(X), input_dim)
    
    # Normalize in-place to save memory
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    gc.collect()
    
    # Get fold 5 indices only
    gkf = GroupKFold(5)
    folds = list(gkf.split(X, y, group_ids))
    ti, vi = folds[4]  # fold 5 (0-indexed = 4)
    
    logger.info("Fold 5: train=%d, val=%d", len(ti), len(vi))
    
    # Only load needed data (DON'T copy entire X_scaled)
    X_val = X[vi].copy()
    y_val = y[vi].copy()
    
    torch.cuda.empty_cache()
    model = DeepSynergyV5(input_dim).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, 200)
    
    best_r, patience, best_state = -1, 0, None
    for ep in range(200):
        model.train()
        perm = np.random.permutation(len(ti))
        for s in range(0, len(ti), BS):
            idx = ti[perm[s:s+BS]]
            xb = torch.FloatTensor(X[idx]).to(device)
            yb = torch.FloatTensor(y[idx]).to(device)
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
        model.load_state_dict(best_state)
        model.to(device)
    vp = predict_batched(model, X_val)
    r = pearsonr(y_val, vp)[0]
    
    logger.info("Fold 5: r=%.4f (n=%d)", r, len(vi))
    
    # Previous results
    prev = [0.7142, 0.7041, 0.7007, 0.7124]
    all_results = prev + [r]
    
    logger.info("\n" + "=" * 60)
    logger.info("COMPLETE v5 Drug-pair Results")
    logger.info("=" * 60)
    logger.info("  F1=%.4f, F2=%.4f, F3=%.4f, F4=%.4f, F5=%.4f",
                *all_results)
    logger.info("  Mean: r=%.4f +/- %.4f", np.mean(all_results), np.std(all_results))
    logger.info("  v4: 0.6533, v3: 0.6044")
    logger.info("  Delta vs v4: %+.4f", np.mean(all_results) - 0.6533)
    logger.info("  Delta vs v3: %+.4f", np.mean(all_results) - 0.6044)
    
    # Save final model (best from fold 5)
    if best_state:
        save_model = DeepSynergyV5(input_dim)
        save_model.load_state_dict(best_state)
        torch.save(save_model.state_dict(), MODEL_DIR / "deep_synergy_v5.pt")
        
        results = {
            "model": "DeepSynergy v5",
            "input_dim": input_dim,
            "features": "FP(1024x2) + CL_expr(256d expanded) + tissue(12d)",
            "cl_match_pct": 56.9,
            "random_cv": {"r": 0.7342, "std": 0.0027},
            "drug_pair_cv": {
                "folds": all_results,
                "r": round(float(np.mean(all_results)), 4),
                "std": round(float(np.std(all_results)), 4),
            },
            "comparison": {
                "v3_drug_pair": 0.6044,
                "v4_drug_pair": 0.6533,
            },
        }
        with open(MODEL_DIR / "deep_synergy_v5_results.json", "w") as f:
            json.dump(results, f, indent=2)
        logger.info("Saved: %s", MODEL_DIR / "deep_synergy_v5.pt")


if __name__ == "__main__":
    main()
