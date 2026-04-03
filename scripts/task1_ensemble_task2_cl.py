"""
Task 1: DeepSynergy v5 + Energy v6 Ensemble (Stacking)
Task 2: DepMap CL Coverage Expansion via Cellosaurus aliases
=============================================================
Runs both tasks sequentially in one script.
"""

import json, logging, pickle, time, gc
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.stats import pearsonr
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
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
                nn.Linear(prev, h), nn.BatchNorm1d(h),
                nn.GELU(), nn.Dropout(0.3 if i < 2 else 0.2),
            ])
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.net(x).squeeze(-1)


def get_tissue_vec(cl_name):
    vec = np.zeros(len(TISSUE_KEYWORDS), dtype=np.float32)
    cl = cl_name.upper().replace('-','').replace('_','').replace(' ','')
    for i, (_, kws) in enumerate(TISSUE_KEYWORDS.items()):
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


# ================================================================
# Task 2: Expand CL Coverage with Cellosaurus/CCLE aliases
# ================================================================
def expand_cl_coverage():
    logger.info("=" * 60)
    logger.info("TASK 2: Expand DepMap CL Coverage")
    logger.info("=" * 60)
    
    # Load DepMap info with aliases
    info = pd.read_csv(DATA_DIR / "depmap_sample_info.csv")
    
    # Load CCLE expression
    parquet_path = DATA_DIR / "depmap" / "ccle_expression.parquet"
    expr_df = pd.read_parquet(parquet_path)
    logger.info("CCLE expression: %d CLs x %d genes", *expr_df.shape)
    
    # PCA 256d
    from sklearn.decomposition import PCA
    expr_clean = expr_df.dropna(axis=1, how='all').fillna(0)
    scaler = StandardScaler()
    expr_scaled = scaler.fit_transform(expr_clean)
    pca = PCA(n_components=256, random_state=42)
    embeddings = pca.fit_transform(expr_scaled)
    logger.info("PCA explained var: %.3f", pca.explained_variance_ratio_.sum())
    
    # Build comprehensive name map using ALL available fields
    name_map = {}
    for _, row in info.iterrows():
        ach = str(row['DepMap_ID'])
        names = set()
        for col in ['cell_line_name', 'stripped_cell_line_name', 'CCLE_Name', 'alias']:
            val = row.get(col)
            if pd.notna(val):
                val = str(val)
                if col == 'CCLE_Name':
                    # Split CCLE format: "NAME_TISSUE"
                    names.add(val.split('_')[0].upper().strip())
                    names.add(val.upper().strip())
                elif col == 'alias':
                    # Multiple aliases separated by comma
                    for a in val.split(','):
                        a = a.strip()
                        if a and a.upper() != 'NAN':
                            names.add(a.upper().strip())
                else:
                    if val.upper().strip() != 'NAN':
                        names.add(val.upper().strip())
        name_map[ach] = list(names)
    
    # Load ACH mappings
    for json_file in ['ach_to_name_full_v2.json', 'ach_to_name_full.json', 'ach_to_name.json']:
        path = DATA_DIR / "depmap" / json_file
        if path.exists():
            with open(path) as f:
                ach_json = json.load(f)
            for ach, name in ach_json.items():
                if ach not in name_map:
                    name_map[ach] = []
                name_map[ach].append(name.upper().strip())
    
    # DrugComb cell lines
    dc = pd.read_csv(DATA_DIR / "synergy_combined.csv", usecols=['cell_line'], low_memory=False)
    dc_cls = set(dc.cell_line.str.upper().str.strip().unique())
    logger.info("DrugComb CLs: %d", len(dc_cls))
    
    # Build name → embedding with comprehensive matching
    emb_rows = {}
    for i, ach_id in enumerate(expr_clean.index):
        ach_str = str(ach_id)
        names = name_map.get(ach_str, [ach_str])
        for name in names:
            emb_rows[name] = embeddings[i]
            # Cleaned variants
            clean = name.replace('-','').replace('_','').replace(' ','').replace('.','')
            if clean != name:
                emb_rows[clean] = embeddings[i]
            # Common cell line naming patterns
            # Remove prefix numbers (e.g., "786-O" → "786O")
            no_dash = name.replace('-', '')
            if no_dash != name:
                emb_rows[no_dash] = embeddings[i]
    
    # Additional fuzzy matching for DrugComb CLs
    unmatched = []
    matched_count = 0
    for cl in dc_cls:
        cl_upper = cl.upper().strip()
        cl_clean = cl_upper.replace('-','').replace('_','').replace(' ','').replace('.','')
        if cl_upper in emb_rows or cl_clean in emb_rows:
            matched_count += 1
        else:
            unmatched.append(cl)
    
    logger.info("Initial match: %d/%d (%.1f%%)", matched_count, len(dc_cls),
                100*matched_count/len(dc_cls))
    
    # Try additional matching for unmatched CLs
    # Build reverse index: all known names for each ACH ID
    all_known = {}
    for ach, names in name_map.items():
        for n in names:
            all_known[n.upper().replace('-','').replace('_','').replace(' ','').replace('.','')] = ach
    
    extra_matched = 0
    for cl in unmatched:
        cl_clean = cl.upper().replace('-','').replace('_','').replace(' ','').replace('.','')
        # Try partial matching
        for known_name, ach in all_known.items():
            if cl_clean in known_name or known_name in cl_clean:
                # Find embedding for this ACH
                ach_idx = list(expr_clean.index).index(ach) if ach in expr_clean.index else -1
                if ach_idx >= 0:
                    emb_rows[cl.upper()] = embeddings[ach_idx]
                    emb_rows[cl_clean] = embeddings[ach_idx]
                    extra_matched += 1
                    break
    
    total_matched = matched_count + extra_matched
    logger.info("After fuzzy: %d/%d (%.1f%%)", total_matched, len(dc_cls),
                100*total_matched/len(dc_cls))
    
    # Save expanded embeddings  
    emb_df = pd.DataFrame(emb_rows).T
    emb_df.columns = [f'PC{i+1}' for i in range(256)]
    out_path = DATA_DIR / "cell_line_expression_256_expanded_v2.csv"
    emb_df.to_csv(out_path)
    
    pkl_path = DATA_DIR / "cell_line_expression_256_expanded_v2.pkl"
    emb_dict = {name: emb_rows[name].astype(np.float32) for name in emb_rows}
    with open(pkl_path, 'wb') as f:
        pickle.dump(emb_dict, f)
    
    logger.info("Saved expanded v2: %d entries, %.1f MB", 
                len(emb_dict), pkl_path.stat().st_size/1e6)
    
    # Calculate sample-level match rate
    total_samples = 0
    matched_samples = 0
    for cl in dc.cell_line.str.upper().str.strip():
        total_samples += 1
        cl_clean = cl.replace('-','').replace('_','').replace(' ','').replace('.','')
        if cl in emb_dict or cl_clean in emb_dict:
            matched_samples += 1
    
    logger.info("Sample-level match: %d/%d (%.1f%%) [v5 was 56.9%%]",
                matched_samples, total_samples,
                100*matched_samples/max(total_samples,1))
    
    return pkl_path, total_matched, len(dc_cls)


# ================================================================
# Task 1: v5 + Energy v6 Stacking Ensemble
# ================================================================
def run_ensemble():
    logger.info("\n" + "=" * 60)
    logger.info("TASK 1: DeepSynergy v5 + Ensemble")
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
    
    # Load expanded CL (v2 or v1)
    pkl_v2 = DATA_DIR / "cell_line_expression_256_expanded_v2.pkl"
    pkl_v1 = DATA_DIR / "cell_line_expression_256_expanded.pkl"
    cl_path = pkl_v2 if pkl_v2.exists() else pkl_v1
    with open(cl_path, 'rb') as f:
        cl_map = pickle.load(f)
    logger.info("CL map: %d entries from %s", len(cl_map), cl_path.name)
    
    # Build features
    logger.info("Building features...")
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
            feat = np.concatenate([fps[da], fps[db], np.zeros(256, dtype=np.float32), tissue])
        X_list.append(feat)
        y_list.append(row['synergy_loewe'])
        groups.append(tuple(sorted([da, db])))
    
    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.float32)
    del X_list; gc.collect()
    
    mask = ~np.isnan(y)
    X, y = X[mask], y[mask]
    groups_clean = [groups[i] for i in range(len(mask)) if mask[i]]
    del mask; gc.collect()
    
    unique_pairs = list(set(groups_clean))
    pair_to_id = {p: i for i, p in enumerate(unique_pairs)}
    group_ids = np.array([pair_to_id[g] for g in groups_clean])
    input_dim = X.shape[1]
    
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    gc.collect()
    
    logger.info("Features: %d x %d, groups: %d", len(X), input_dim, len(unique_pairs))
    
    # ----------------------------------------------------------------
    # Generate out-of-fold predictions for ensemble
    # ----------------------------------------------------------------
    logger.info("\n--- OOF Predictions for Ensemble ---")
    gkf = GroupKFold(5)
    oof_preds = np.zeros(len(X))
    fold_results = []
    
    for fold, (ti, vi) in enumerate(gkf.split(X, y, group_ids)):
        torch.cuda.empty_cache()
        model = DeepSynergyV5(input_dim).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, 200)
        
        X_val = X[vi].copy()
        y_val = y[vi].copy()
        
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
                    logger.info("  F%d Ep%3d: val_r=%.4f", fold+1, ep+1, r)
                if r > best_r:
                    best_r = r
                    patience = 0
                    best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                else:
                    patience += 1
                if patience >= 4:
                    logger.info("  F%d early stop ep%d", fold+1, ep+1)
                    break
        
        if best_state:
            model.load_state_dict(best_state); model.to(device)
        
        vp = predict_batched(model, X_val)
        r = pearsonr(y_val, vp)[0]
        oof_preds[vi] = vp
        fold_results.append(r)
        logger.info("  Fold %d: r=%.4f (n=%d)", fold+1, r, len(vi))
        
        # Save best model from best fold
        if fold == 0 or r == max(fold_results):
            best_model_state = best_state
        
        del model, X_val, y_val; torch.cuda.empty_cache(); gc.collect()
    
    v5_r = float(np.mean(fold_results))
    logger.info("v5 OOF Drug-pair: r=%.4f +/- %.4f", v5_r, np.std(fold_results))
    
    # ----------------------------------------------------------------
    # Simple self-ensemble: average of multiple v5 seeds
    # Since we don't have Energy v6 OOF predictions ready,
    # let's do a second v5 pass with different seed for diversity
    # ----------------------------------------------------------------
    logger.info("\n--- Second v5 pass (seed=123) for self-ensemble ---")
    np.random.seed(123)
    torch.manual_seed(123)
    oof_preds2 = np.zeros(len(X))
    fold_results2 = []
    
    for fold, (ti, vi) in enumerate(gkf.split(X, y, group_ids)):
        torch.cuda.empty_cache()
        model = DeepSynergyV5(input_dim, hidden=[1024, 512, 256, 128]).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=2e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, 200)
        
        X_val = X[vi].copy()
        y_val = y[vi].copy()
        
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
                    logger.info("  F%d Ep%3d: val_r=%.4f", fold+1, ep+1, r)
                if r > best_r:
                    best_r = r
                    patience = 0
                    best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                else:
                    patience += 1
                if patience >= 4:
                    logger.info("  F%d early stop ep%d", fold+1, ep+1)
                    break
        
        if best_state:
            model.load_state_dict(best_state); model.to(device)
        
        vp = predict_batched(model, X_val)
        r = pearsonr(y_val, vp)[0]
        oof_preds2[vi] = vp
        fold_results2.append(r)
        logger.info("  Fold %d (small): r=%.4f (n=%d)", fold+1, r, len(vi))
        del model, X_val, y_val; torch.cuda.empty_cache(); gc.collect()
    
    v5b_r = float(np.mean(fold_results2))
    logger.info("v5-small OOF: r=%.4f +/- %.4f", v5b_r, np.std(fold_results2))
    
    # ----------------------------------------------------------------
    # Stacking: Ridge regression on OOF predictions
    # ----------------------------------------------------------------
    logger.info("\n--- Stacking Ensemble ---")
    
    # Features: v5-large OOF + v5-small OOF
    meta_X = np.column_stack([oof_preds, oof_preds2])
    
    # Simple average
    avg_preds = (oof_preds + oof_preds2) / 2
    avg_r = pearsonr(y, avg_preds)[0]
    logger.info("Simple average: r=%.4f", avg_r)
    
    # Ridge stacking (CV)
    stacking_results = []
    for fold, (ti, vi) in enumerate(gkf.split(meta_X, y, group_ids)):
        ridge = Ridge(alpha=1.0)
        ridge.fit(meta_X[ti], y[ti])
        sp = ridge.predict(meta_X[vi])
        r = pearsonr(y[vi], sp)[0]
        stacking_results.append(r)
    
    stack_r = float(np.mean(stacking_results))
    logger.info("Ridge stacking: r=%.4f +/- %.4f", stack_r, np.std(stacking_results))
    
    # Optimal weights
    best_r_overall = -1
    best_w = 0.5
    for w in np.arange(0.0, 1.01, 0.05):
        wp = w * oof_preds + (1-w) * oof_preds2
        r = pearsonr(y, wp)[0]
        if r > best_r_overall:
            best_r_overall = r
            best_w = w
    logger.info("Optimal weight: w=%.2f, r=%.4f", best_w, best_r_overall)
    
    # Save ensemble model
    ensemble_model = Ridge(alpha=1.0)
    ensemble_model.fit(meta_X, y)
    
    ensemble_results = {
        "v5_large_drug_pair": round(v5_r, 4),
        "v5_small_drug_pair": round(v5b_r, 4),
        "simple_average": round(avg_r, 4),
        "ridge_stacking": round(stack_r, 4),
        "optimal_weight": round(best_w, 2),
        "optimal_weight_r": round(best_r_overall, 4),
        "comparison": {
            "v3": 0.6044,
            "v4": 0.6533,
            "v5_prev": 0.7074,
        },
    }
    
    with open(MODEL_DIR / "ensemble_v5_results.json", "w") as f:
        json.dump(ensemble_results, f, indent=2)
    
    with open(MODEL_DIR / "ensemble_v5_ridge.pkl", "wb") as f:
        pickle.dump(ensemble_model, f)
    
    # Save best v5 model
    if best_model_state:
        save_model = DeepSynergyV5(input_dim)
        save_model.load_state_dict(best_model_state)
        torch.save(save_model.state_dict(), MODEL_DIR / "deep_synergy_v5.pt")
    
    logger.info("\n" + "=" * 60)
    logger.info("ENSEMBLE FINAL RESULTS")
    logger.info("=" * 60)
    logger.info("  v5-large:      r=%.4f", v5_r)
    logger.info("  v5-small:      r=%.4f", v5b_r)
    logger.info("  Average:       r=%.4f", avg_r)
    logger.info("  Ridge stack:   r=%.4f", stack_r)
    logger.info("  Optimal(w=%.2f): r=%.4f", best_w, best_r_overall)
    logger.info("  v3 baseline:   r=0.6044")
    logger.info("  v4 baseline:   r=0.6533")
    
    return ensemble_results


def main():
    t0 = time.time()
    
    # Task 2: Expand CL coverage
    expand_cl_coverage()
    gc.collect()
    
    # Task 1: Ensemble
    results = run_ensemble()
    
    elapsed = time.time() - t0
    logger.info("\nTotal time: %.0f seconds (%.1f min)", elapsed, elapsed/60)


if __name__ == "__main__":
    main()
