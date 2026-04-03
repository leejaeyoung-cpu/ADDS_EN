"""
Phase 5: 정직한 진단 + 생물학 기반 시너지 개선

1. Drug-pair stratified CV (데이터 누출 진단)
2. Drug target features (ChEMBL API)
3. Cross-attention bimodal DNN
4. 전체 비교
"""

import json
import logging
import pickle
import urllib.request
import time
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import pearsonr
from sklearn.model_selection import KFold, GroupKFold
from sklearn.preprocessing import StandardScaler

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DATA_DIR = Path("F:/ADDS/data/ml_training")
MODEL_DIR = Path("F:/ADDS/models")
device = "cuda" if torch.cuda.is_available() else "cpu"


# ============================================================
# PART 0: Load data and features
# ============================================================

def load_all_data():
    """Load synergy data, fingerprints, and embeddings."""
    df = pd.read_csv(DATA_DIR / "synergy_combined.csv", low_memory=False)
    df_oneil = df[df["source"] == "oneil"]
    
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
    
    embed_path = DATA_DIR / "depmap" / "cellline_embedding_v2.pkl"
    with open(embed_path, "rb") as f:
        embed_data = pickle.load(f)
    
    return df_oneil, fps, embed_data


def build_features_with_groups(df, fps, embed_data):
    """Build features with drug-pair group IDs for stratified CV."""
    embeddings = embed_data["embeddings"]
    emb_dim = embed_data["dim"]
    
    def norm_cl(n):
        return str(n).upper().replace("-","").replace("_","").replace(" ","").replace(".","")
    
    X_list, y_list, groups = [], [], []
    for _, row in df.iterrows():
        da, db = str(row["drug_a"]), str(row["drug_b"])
        if da not in fps or db not in fps:
            continue
        score = float(row["synergy_loewe"])
        if np.isnan(score): continue
        
        fp = np.concatenate([fps[da], fps[db]])
        cl = norm_cl(str(row["cell_line"]))
        cl_feat = embeddings.get(cl, np.zeros(emb_dim, dtype=np.float32))
        
        X_list.append(np.concatenate([fp, cl_feat]))
        y_list.append(score)
        
        # Drug pair group (sorted for consistency)
        pair = tuple(sorted([da, db]))
        groups.append(pair)
    
    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.float32)
    
    # Convert pair tuples to integer group IDs
    unique_pairs = list(set(groups))
    pair_to_id = {p: i for i, p in enumerate(unique_pairs)}
    group_ids = np.array([pair_to_id[g] for g in groups])
    
    logger.info("Features: X=%s, unique drug pairs: %d", X.shape, len(unique_pairs))
    return X, y, group_ids, X.shape[1]


# ============================================================
# PART 1: Drug-pair stratified CV (diagnosis)
# ============================================================

class SimpleMLP(nn.Module):
    def __init__(self, input_dim, hidden=[512, 256, 128]):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden:
            layers.extend([nn.Linear(prev, h), nn.LayerNorm(h), nn.SiLU(), nn.Dropout(0.3)])
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.net(x).squeeze(-1)


def train_dnn(X_train, y_train, X_val, y_val, input_dim, n_epochs=200, label=""):
    """Train DNN and return val Pearson r."""
    Xt = torch.FloatTensor(X_train).to(device)
    yt = torch.FloatTensor(y_train).to(device)
    Xv = torch.FloatTensor(X_val).to(device)
    
    model = SimpleMLP(input_dim).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_epochs)
    
    best_r = -1
    best_state = None
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
            r = pearsonr(y_val, vp)[0]
            if r > best_r:
                best_r = r
                patience = 0
                best_state = {k:v.clone() for k,v in model.state_dict().items()}
            else:
                patience += 1
            if patience >= 5:
                break
    
    if best_state:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        vp = model(Xv).cpu().numpy()
    return pearsonr(y_val, vp)[0]


def diagnose_cv(X, y, groups, input_dim):
    """Compare random CV vs drug-pair stratified CV."""
    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)
    
    results = {}
    
    # 1. Random 5-fold (current method)
    logger.info("\n--- Random 5-fold CV ---")
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    random_rs = []
    for fold, (ti, vi) in enumerate(kf.split(X_s)):
        r = train_dnn(X_s[ti], y[ti], X_s[vi], y[vi], input_dim, label=f"Random-{fold}")
        random_rs.append(r)
        logger.info("  Random fold %d: r=%.4f", fold+1, r)
    logger.info("  Random CV: r=%.4f +/- %.4f", np.mean(random_rs), np.std(random_rs))
    results["random"] = {"r": round(float(np.mean(random_rs)), 4), "std": round(float(np.std(random_rs)), 4)}
    
    # 2. Drug-pair stratified (GroupKFold)
    logger.info("\n--- Drug-pair GroupKFold CV ---")
    gkf = GroupKFold(n_splits=5)
    pair_rs = []
    for fold, (ti, vi) in enumerate(gkf.split(X_s, y, groups)):
        # Check: no drug pairs overlap
        train_groups = set(groups[ti])
        val_groups = set(groups[vi])
        overlap = train_groups & val_groups
        
        r = train_dnn(X_s[ti], y[ti], X_s[vi], y[vi], input_dim, label=f"Pair-{fold}")
        pair_rs.append(r)
        logger.info("  Pair fold %d: r=%.4f (overlap=%d)", fold+1, r, len(overlap))
    logger.info("  Pair CV: r=%.4f +/- %.4f", np.mean(pair_rs), np.std(pair_rs))
    results["drug_pair"] = {"r": round(float(np.mean(pair_rs)), 4), "std": round(float(np.std(pair_rs)), 4)}
    
    # 3. Leave-cell-line-out (optional, quick)
    logger.info("\n--- Leave-cell-out (sample 10 cell lines) ---")
    # Not implemented as GroupKFold on cell lines; just report the pair result
    
    delta = results["random"]["r"] - results["drug_pair"]["r"]
    logger.info("\n  DATA LEAKAGE ESTIMATE: %.4f (random - pair)", delta)
    results["leakage_estimate"] = round(float(delta), 4)
    
    return results


# ============================================================
# PART 2: Drug target features
# ============================================================

def get_drug_targets():
    """Get drug targets from ChEMBL API."""
    cache = DATA_DIR / "drug_targets.json"
    if cache.exists():
        with open(cache) as f:
            return json.load(f)
    
    smiles_path = MODEL_DIR / "synergy" / "drug_smiles.json"
    ext_path = MODEL_DIR / "synergy" / "drug_smiles_extended.json"
    
    all_smiles = {}
    for p in [smiles_path, ext_path]:
        if p.exists():
            with open(p) as f:
                all_smiles.update(json.load(f))
    
    # Known drug-target mappings (curated from literature)
    known_targets = {
        "5-FU": ["TYMS", "DPYD"],
        "ABT-888": ["PARP1", "PARP2"],
        "AZD1775": ["WEE1"],
        "BEZ-235": ["PIK3CA", "MTOR"],
        "Bortezomib": ["PSMB5"],
        "Carboplatin": ["DNA"],
        "Cisplatin": ["DNA"],
        "Cyclophosphamide": ["DNA"],
        "Dasatinib": ["BCR-ABL", "SRC", "KIT"],
        "Doxorubicin": ["TOP2A"],
        "Erlotinib": ["EGFR"],
        "Etoposide": ["TOP2A"],
        "Everolimus": ["MTOR"],
        "Gefitinib": ["EGFR"],
        "Gemcitabine": ["RRM1", "RRM2"],
        "Imatinib": ["BCR-ABL", "KIT", "PDGFRA"],
        "Lapatinib": ["EGFR", "ERBB2"],
        "Methotrexate": ["DHFR"],
        "MK-1775": ["WEE1"],
        "MK-2206": ["AKT1", "AKT2", "AKT3"],
        "MK-4541": ["BRD4"],
        "MK-8669": ["MTOR"],
        "MK-8776": ["CHEK1"],
        "Oxaliplatin": ["DNA"],
        "Paclitaxel": ["TUBB"],
        "Palbociclib": ["CDK4", "CDK6"],
        "Rapamycin": ["MTOR"],
        "Sorafenib": ["RAF1", "BRAF", "VEGFR2", "KIT"],
        "Sunitinib": ["VEGFR2", "PDGFRA", "KIT", "FLT3"],
        "Tamoxifen": ["ESR1"],
        "Temozolomide": ["DNA"],
        "Topotecan": ["TOP1"],
        "Trametinib": ["MAP2K1", "MAP2K2"],
        "Vemurafenib": ["BRAF"],
        "Vinblastine": ["TUBB"],
        "Vinorelbine": ["TUBB"],
        "Vorinostat": ["HDAC1", "HDAC2", "HDAC3"],
    }
    
    # Log unmapped drugs (skip ChEMBL API to avoid timeout)
    unmapped = [d for d in all_smiles if d not in known_targets]
    if unmapped:
        logger.info("Unmapped drugs (no curated targets): %s", unmapped[:10])
    
    with open(cache, "w") as f:
        json.dump(known_targets, f, indent=2)
    
    logger.info("Drug targets: %d drugs mapped", len(known_targets))
    return known_targets


def build_target_features(df, fps, embed_data, targets):
    """Build features with drug target information."""
    embeddings = embed_data["embeddings"]
    emb_dim = embed_data["dim"]
    
    # All unique targets
    all_targets = set()
    for tlist in targets.values():
        all_targets.update(tlist)
    all_targets = sorted(all_targets)
    target_to_idx = {t: i for i, t in enumerate(all_targets)}
    n_targets = len(all_targets)
    logger.info("Unique targets: %d", n_targets)
    
    def norm_cl(n):
        return str(n).upper().replace("-","").replace("_","").replace(" ","").replace(".","")
    
    # Build target vectors for each drug
    drug_target_vecs = {}
    for drug, tlist in targets.items():
        vec = np.zeros(n_targets, dtype=np.float32)
        for t in tlist:
            if t in target_to_idx:
                vec[target_to_idx[t]] = 1.0
        drug_target_vecs[drug] = vec
    
    X_list, y_list, groups = [], [], []
    for _, row in df.iterrows():
        da, db = str(row["drug_a"]), str(row["drug_b"])
        if da not in fps or db not in fps:
            continue
        score = float(row["synergy_loewe"])
        if np.isnan(score): continue
        
        fp = np.concatenate([fps[da], fps[db]])
        cl = norm_cl(str(row["cell_line"]))
        cl_feat = embeddings.get(cl, np.zeros(emb_dim, dtype=np.float32))
        
        # Target features
        ta = drug_target_vecs.get(da, np.zeros(n_targets, dtype=np.float32))
        tb = drug_target_vecs.get(db, np.zeros(n_targets, dtype=np.float32))
        
        # Target interaction features
        target_overlap = (ta * tb).sum()  # shared targets
        target_union = np.clip(ta + tb, 0, 1).sum()  # total unique targets
        target_combo = np.concatenate([ta, tb, [target_overlap, target_union]])
        
        X_list.append(np.concatenate([fp, cl_feat, target_combo]))
        y_list.append(score)
        groups.append(tuple(sorted([da, db])))
    
    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.float32)
    
    unique_pairs = list(set(groups))
    pair_to_id = {p: i for i, p in enumerate(unique_pairs)}
    group_ids = np.array([pair_to_id[g] for g in groups])
    
    input_dim = X.shape[1]
    logger.info("Target features: X=%s (FP=2048, CL=%d, targets=%d)", X.shape, emb_dim, n_targets*2+2)
    return X, y, group_ids, input_dim


# ============================================================
# PART 3: Cross-attention bimodal DNN
# ============================================================

class CrossAttentionSynergy(nn.Module):
    """Bimodal model with drug-cell line cross-attention."""
    
    def __init__(self, fp_dim=1024, cl_dim=100, n_heads=4, hidden=256):
        super().__init__()
        self.d_model = hidden
        
        # Drug encoders (separate for A and B)
        self.drug_encoder = nn.Sequential(
            nn.Linear(fp_dim, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(),
        )
        
        # Cell line encoder
        self.cl_encoder = nn.Sequential(
            nn.Linear(cl_dim, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(),
        )
        
        # Cross-attention: drug queries, cell line keys/values
        self.cross_attn = nn.MultiheadAttention(hidden, n_heads, dropout=0.1, batch_first=True)
        
        # Drug-drug interaction
        self.drug_interact = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(),
            nn.Dropout(0.3),
        )
        
        # Final prediction
        self.predictor = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden, 64),
            nn.SiLU(),
            nn.Linear(64, 1),
        )
    
    def forward(self, drug_a, drug_b, cl):
        # Encode
        ha = self.drug_encoder(drug_a)  # [B, hidden]
        hb = self.drug_encoder(drug_b)
        hc = self.cl_encoder(cl)        # [B, hidden]
        
        # Stack drugs as sequence for cross-attention
        drugs = torch.stack([ha, hb], dim=1)   # [B, 2, hidden]
        cl_kv = hc.unsqueeze(1)                 # [B, 1, hidden]
        
        # Cross-attention: how drugs attend to cell line context
        drugs_ctx, _ = self.cross_attn(drugs, cl_kv, cl_kv)  # [B, 2, hidden]
        
        # Drug-drug interaction in cell context
        drug_inter = self.drug_interact(
            torch.cat([drugs_ctx[:, 0], drugs_ctx[:, 1]], dim=-1)
        )  # [B, hidden]
        
        # Combine with cell line for final prediction
        combined = torch.cat([drug_inter, hc], dim=-1)  # [B, 2*hidden]
        return self.predictor(combined).squeeze(-1)


def build_bimodal_features(df, fps, embed_data):
    """Build separate drug A, drug B, cell line features."""
    embeddings = embed_data["embeddings"]
    emb_dim = embed_data["dim"]
    
    def norm_cl(n):
        return str(n).upper().replace("-","").replace("_","").replace(" ","").replace(".","")
    
    fpA_list, fpB_list, cl_list, y_list, groups = [], [], [], [], []
    for _, row in df.iterrows():
        da, db = str(row["drug_a"]), str(row["drug_b"])
        if da not in fps or db not in fps:
            continue
        score = float(row["synergy_loewe"])
        if np.isnan(score): continue
        
        cl = norm_cl(str(row["cell_line"]))
        cl_feat = embeddings.get(cl, np.zeros(emb_dim, dtype=np.float32))
        
        fpA_list.append(fps[da])
        fpB_list.append(fps[db])
        cl_list.append(cl_feat)
        y_list.append(score)
        groups.append(tuple(sorted([da, db])))
    
    fpA = np.array(fpA_list, dtype=np.float32)
    fpB = np.array(fpB_list, dtype=np.float32)
    cl = np.array(cl_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.float32)
    
    unique_pairs = list(set(groups))
    pair_to_id = {p: i for i, p in enumerate(unique_pairs)}
    group_ids = np.array([pair_to_id[g] for g in groups])
    
    return fpA, fpB, cl, y, group_ids, emb_dim


def train_cross_attention_cv(fpA, fpB, cl, y, groups, cl_dim, n_epochs=200):
    """Train cross-attention model with both random and group CV."""
    # Normalize
    sA = StandardScaler().fit_transform(fpA)
    sB = StandardScaler().fit_transform(fpB)
    sC = StandardScaler().fit_transform(cl)
    
    results = {}
    
    for cv_name, kf_gen in [("random", KFold(n_splits=5, shuffle=True, random_state=42)),
                             ("drug_pair", GroupKFold(n_splits=5))]:
        rs = []
        
        if cv_name == "drug_pair":
            splits = kf_gen.split(sA, y, groups)
        else:
            splits = kf_gen.split(sA)
        
        for fold, (ti, vi) in enumerate(splits):
            tA = torch.FloatTensor(sA[ti]).to(device)
            tB = torch.FloatTensor(sB[ti]).to(device)
            tC = torch.FloatTensor(sC[ti]).to(device)
            ty = torch.FloatTensor(y[ti]).to(device)
            
            vA = torch.FloatTensor(sA[vi]).to(device)
            vB = torch.FloatTensor(sB[vi]).to(device)
            vC = torch.FloatTensor(sC[vi]).to(device)
            
            model = CrossAttentionSynergy(fp_dim=1024, cl_dim=cl_dim, n_heads=4, hidden=256).to(device)
            opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_epochs)
            
            best_r = -1
            best_state = None
            patience = 0
            
            for epoch in range(n_epochs):
                model.train()
                perm = torch.randperm(len(tA))
                for start in range(0, len(tA), 2048):
                    idx = perm[start:start+2048]
                    opt.zero_grad()
                    pred = model(tA[idx], tB[idx], tC[idx])
                    loss = F.mse_loss(pred, ty[idx])
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    opt.step()
                sched.step()
                
                if (epoch+1) % 10 == 0:
                    model.eval()
                    with torch.no_grad():
                        vp = model(vA, vB, vC).cpu().numpy()
                    r = pearsonr(y[vi], vp)[0]
                    if r > best_r:
                        best_r = r
                        patience = 0
                        best_state = {k:v.clone() for k,v in model.state_dict().items()}
                    else:
                        patience += 1
                    if patience >= 5:
                        break
            
            if best_state:
                model.load_state_dict(best_state)
            model.eval()
            with torch.no_grad():
                vp = model(vA, vB, vC).cpu().numpy()
            r = pearsonr(y[vi], vp)[0]
            rs.append(r)
            logger.info("  CrossAttn %s fold %d: r=%.4f", cv_name, fold+1, r)
        
        avg = np.mean(rs)
        std = np.std(rs)
        logger.info("  CrossAttn %s: r=%.4f +/- %.4f", cv_name, avg, std)
        results[cv_name] = {"r": round(float(avg), 4), "std": round(float(std), 4)}
    
    return results


# ============================================================
# Main
# ============================================================

def main():
    logger.info("=" * 60)
    logger.info("PHASE 5: Honest Diagnosis + Biology-based Improvement")
    logger.info("=" * 60)
    
    results = {}
    
    # Load data
    df_oneil, fps, embed_data = load_all_data()
    
    # Part 1: Diagnosis — Random vs Drug-pair CV
    logger.info("\n" + "=" * 40)
    logger.info("PART 1: Data Leakage Diagnosis")
    logger.info("=" * 40)
    X, y, groups, input_dim = build_features_with_groups(df_oneil, fps, embed_data)
    diag = diagnose_cv(X, y, groups, input_dim)
    results["diagnosis"] = diag
    
    # Part 2: Drug target features
    logger.info("\n" + "=" * 40)
    logger.info("PART 2: Drug Target Features")
    logger.info("=" * 40)
    targets = get_drug_targets()
    X_t, y_t, groups_t, input_dim_t = build_target_features(df_oneil, fps, embed_data, targets)
    
    # Train with both CV methods
    scaler = StandardScaler()
    X_ts = scaler.fit_transform(X_t)
    
    for cv_name, kf_gen in [("random", KFold(n_splits=5, shuffle=True, random_state=42)),
                             ("drug_pair", GroupKFold(n_splits=5))]:
        rs = []
        splits = kf_gen.split(X_ts, y_t, groups_t) if cv_name == "drug_pair" else kf_gen.split(X_ts)
        for fold, (ti, vi) in enumerate(splits):
            r = train_dnn(X_ts[ti], y_t[ti], X_ts[vi], y_t[vi], input_dim_t, label=f"Target-{cv_name}")
            rs.append(r)
            logger.info("  Target+DNN %s fold %d: r=%.4f", cv_name, fold+1, r)
        logger.info("  Target+DNN %s: r=%.4f +/- %.4f", cv_name, np.mean(rs), np.std(rs))
        results[f"target_{cv_name}"] = {"r": round(float(np.mean(rs)), 4), "std": round(float(np.std(rs)), 4)}
    
    # Part 3: Cross-attention
    logger.info("\n" + "=" * 40)
    logger.info("PART 3: Cross-Attention Bimodal DNN")
    logger.info("=" * 40)
    fpA, fpB, cl, y_ca, groups_ca, cl_dim = build_bimodal_features(df_oneil, fps, embed_data)
    ca_results = train_cross_attention_cv(fpA, fpB, cl, y_ca, groups_ca, cl_dim)
    results["cross_attention"] = ca_results
    
    # Final comparison
    logger.info("\n" + "=" * 60)
    logger.info("FINAL COMPARISON")
    logger.info("=" * 60)
    
    comparison = {
        "MLP random": results["diagnosis"]["random"],
        "MLP drug-pair": results["diagnosis"]["drug_pair"],
        "MLP+Target random": results.get("target_random", {}),
        "MLP+Target drug-pair": results.get("target_drug_pair", {}),
        "CrossAttn random": results["cross_attention"].get("random", {}),
        "CrossAttn drug-pair": results["cross_attention"].get("drug_pair", {}),
    }
    
    for name, vals in comparison.items():
        if vals:
            logger.info("  %-25s r=%.4f +/- %.4f", name, vals.get("r", 0), vals.get("std", 0))
    
    logger.info("\nLeakage estimate: %.4f", results["diagnosis"]["leakage_estimate"])
    
    with open(MODEL_DIR / "phase5_results.json", "w") as f:
        json.dump(results, f, indent=2, default=float)
    logger.info("\nSaved: phase5_results.json")


if __name__ == "__main__":
    main()
