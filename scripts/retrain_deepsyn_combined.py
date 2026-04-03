"""
Retrain DeepSynergy v3 on Combined 1M+ Dataset
================================================
Uses sampled combined data for tractable DNN training.
"""
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import pickle
import hashlib
import logging
import time
from pathlib import Path
from scipy.stats import pearsonr
from sklearn.preprocessing import StandardScaler

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

DATA_DIR = Path("F:/ADDS/data/ml_training")
CCLE_DIR = DATA_DIR / "ccle_raw"
MODEL_DIR = Path("F:/ADDS/models/synergy")

KNOWN_AFFINITIES = {
    'ERLOTINIB': {'EGFR': 8.7, 'ABL1': 5.9},
    'LAPATINIB': {'EGFR': 8.0, 'ERBB2': 8.0},
    'SORAFENIB': {'BRAF': 7.7, 'FLT3': 7.2, 'KIT': 7.2, 'PDGFRB': 7.2, 'RAF1': 8.2, 'KDR': 7.0},
    'SUNITINIB': {'FLT3': 6.6, 'KIT': 9.0, 'PDGFRA': 7.1, 'RET': 7.0, 'KDR': 7.1},
    'DASATINIB': {'ABL1': 9.2, 'EPHA2': 7.8, 'KIT': 8.3, 'PDGFRB': 7.6, 'SRC': 9.3},
    'BORTEZOMIB': {'PSMB5': 9.2},
    'PACLITAXEL': {'TUBB': 8.4},
    'VINBLASTINE': {'TUBB': 9.0},
    'DOXORUBICIN': {'TOP2A': 6.8},
    'ETOPOSIDE': {'TOP2A': 5.7, 'TOP2B': 5.5},
    'TOPOTECAN': {'TOP1': 6.5},
    'MK-2206': {'AKT1': 8.1, 'AKT2': 7.9, 'AKT3': 7.2},
    'BEZ-235': {'PIK3CA': 8.4, 'PIK3CB': 7.1, 'MTOR': 8.2},
    'PD325901': {'MAP2K1': 9.5, 'MAP2K2': 9.1},
    'DINACICLIB': {'CDK1': 8.5, 'CDK2': 9.0, 'CDK5': 9.0, 'CDK9': 8.4},
}


class DeepSynergyV3(nn.Module):
    def __init__(self, n_fp=1024, n_interact=47, n_cell=256,
                 h_drug=512, h_interact=64, h_cell=256, h_comb=512, dropout=0.3):
        super().__init__()
        
        self.drug_net = nn.Sequential(
            nn.Linear(n_fp, h_drug), nn.ReLU(), nn.BatchNorm1d(h_drug), nn.Dropout(dropout),
            nn.Linear(h_drug, h_drug // 2), nn.ReLU(), nn.Dropout(dropout),
        )
        self.interact_net = nn.Sequential(
            nn.Linear(n_interact, h_interact), nn.ReLU(), nn.BatchNorm1d(h_interact), nn.Dropout(dropout),
            nn.Linear(h_interact, h_interact), nn.ReLU(),
        )
        self.cell_net = nn.Sequential(
            nn.Linear(n_cell, h_cell), nn.ReLU(), nn.BatchNorm1d(h_cell), nn.Dropout(dropout),
            nn.Linear(h_cell, h_cell // 2), nn.ReLU(), nn.Dropout(dropout),
        )
        
        comb_in = h_drug // 2 * 2 + h_interact * 2 + h_cell // 2
        self.combined_net = nn.Sequential(
            nn.Linear(comb_in, h_comb), nn.ReLU(), nn.BatchNorm1d(h_comb), nn.Dropout(dropout),
            nn.Linear(h_comb, h_comb // 2), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(h_comb // 2, 1),
        )
        self.n_fp = n_fp
        self.n_interact = n_interact
        self.n_cell = n_cell
    
    def forward(self, x):
        idx = 0
        fp_a = self.drug_net(x[:, idx:idx + self.n_fp]); idx += self.n_fp
        fp_b = self.drug_net(x[:, idx:idx + self.n_fp]); idx += self.n_fp
        int_a = self.interact_net(x[:, idx:idx + self.n_interact]); idx += self.n_interact
        int_b = self.interact_net(x[:, idx:idx + self.n_interact]); idx += self.n_interact
        cell = self.cell_net(x[:, idx:idx + self.n_cell])
        return self.combined_net(torch.cat([fp_a, fp_b, int_a, int_b, cell], dim=1))


def drug_fingerprint(name):
    h = hashlib.sha256(name.encode()).digest()
    bits = np.unpackbits(np.frombuffer(h, dtype=np.uint8))
    fp = np.zeros(1024, dtype=np.float32)
    for i in range(1024):
        fp[i] = bits[i % len(bits)]
    for seed in range(1, 4):
        h2 = hashlib.sha256(f"{name}_{seed}".encode()).digest()
        bits2 = np.unpackbits(np.frombuffer(h2, dtype=np.uint8))
        for i in range(256):
            fp[seed * 256 + i % 1024] = bits2[i % len(bits2)]
    return fp


def build_features(max_samples=100000):
    """Build features with interaction for combined dataset."""
    syn = pd.read_csv(DATA_DIR / "synergy_combined.csv", low_memory=False)
    
    if max_samples and len(syn) > max_samples:
        oneil_mask = syn['source'] == 'oneil'
        oneil_part = syn[oneil_mask]
        dc_part = syn[~oneil_mask].sample(n=max_samples - len(oneil_part), random_state=42)
        syn = pd.concat([oneil_part, dc_part], ignore_index=True)
    
    # Load existing resources
    fp_path = MODEL_DIR / "drug_fingerprints.pkl"
    known_fps = {}
    if fp_path.exists():
        with open(fp_path, 'rb') as f:
            known_fps = {k.upper(): v for k, v in pickle.load(f).items()}
    
    expr_path = MODEL_DIR / "cell_line_expression.pkl"
    known_expr = {}
    if expr_path.exists():
        with open(expr_path, 'rb') as f:
            known_expr = {k.upper(): v for k, v in pickle.load(f).items()}
    
    # CCLE
    ccle = pd.read_csv(CCLE_DIR / "ccle_target_expression.csv")
    target_genes = [c for c in ccle.columns if c != 'cell_line']
    n_tgt = len(target_genes)
    
    ccle_map = {}
    for _, row in ccle.iterrows():
        cl = str(row['cell_line']).upper()
        ccle_map[cl] = row
        ccle_map[cl.split('_')[0]] = row
    
    drug_pki = {}
    for drug, targets in KNOWN_AFFINITIES.items():
        vec = np.zeros(n_tgt, dtype=np.float32)
        for gene, pki in targets.items():
            if gene in target_genes:
                vec[target_genes.index(gene)] = pki
        drug_pki[drug.upper()] = vec
    
    zero_fp = np.zeros(1024, np.float32)
    zero_expr = np.zeros(256, np.float32)
    zero_pki = np.zeros(n_tgt, np.float32)
    
    X, y = [], []
    sources, cells_arr, da_arr, db_arr = [], [], [], []
    
    t0 = time.time()
    for idx, row in syn.iterrows():
        da = str(row['drug_a']).upper().strip()
        db = str(row['drug_b']).upper().strip()
        cl = str(row['cell_line']).upper().strip()
        target = row.get('synergy_loewe', np.nan)
        if pd.isna(target): continue
        
        fp_a = known_fps.get(da, drug_fingerprint(da))
        fp_b = known_fps.get(db, drug_fingerprint(db))
        pki_a = drug_pki.get(da, zero_pki)
        pki_b = drug_pki.get(db, zero_pki)
        
        ccle_row = ccle_map.get(cl, None)
        if ccle_row is not None:
            cell_tgt = np.array([float(ccle_row.get(g, 0)) for g in target_genes], dtype=np.float32)
        else:
            cell_tgt = np.ones(n_tgt, np.float32)
        
        interaction_a = pki_a * cell_tgt
        interaction_b = pki_b * cell_tgt
        expr = known_expr.get(cl, drug_fingerprint(cl)[:256])
        
        features = np.concatenate([fp_a, fp_b, interaction_a, interaction_b, expr])
        X.append(features)
        y.append(float(target))
        sources.append(str(row.get('source', '')))
        cells_arr.append(cl)
        da_arr.append(da)
        db_arr.append(db)
        
        if len(X) % 50000 == 0:
            logger.info(f"  Built {len(X):,} features ({time.time()-t0:.0f}s)")
    
    logger.info(f"  Total: {len(X):,} in {time.time()-t0:.0f}s")
    return (np.array(X), np.array(y), np.array(sources),
            np.array(da_arr), np.array(db_arr), np.array(cells_arr), n_tgt)


def train_fold(X_tr, y_tr, X_te, y_te, n_interact, device, epochs=300, patience=30):
    model = DeepSynergyV3(n_interact=n_interact).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=15, factor=0.5)
    criterion = nn.MSELoss()
    
    X_t = torch.FloatTensor(X_tr).to(device)
    y_t = torch.FloatTensor(y_tr).to(device)
    X_v = torch.FloatTensor(X_te).to(device)
    
    best_r, best_state, wait = -1, None, 0
    
    for epoch in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(len(X_t))
        
        for i in range(0, len(X_t), 512):
            idx = perm[i:i+512]
            pred = model(X_t[idx]).squeeze()
            loss = criterion(pred, y_t[idx])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        model.eval()
        with torch.no_grad():
            pred_v = model(X_v).squeeze().cpu().numpy()
        
        r, _ = pearsonr(y_te, pred_v)
        scheduler.step(-r)
        
        if r > best_r:
            best_r = r
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
        
        if epoch % 30 == 0:
            logger.info(f"    Epoch {epoch}: r={r:.4f}, best={best_r:.4f}")
        
        if wait >= patience:
            logger.info(f"    Early stop at epoch {epoch}")
            break
    
    return best_r, best_state


def main():
    print("=" * 70)
    print("DeepSynergy v3 Retraining on Combined Dataset")
    print("=" * 70)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    MAX_SAMPLES = 100000
    X, y, sources, da, db, cells, n_interact = build_features(max_samples=MAX_SAMPLES)
    
    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)
    
    print(f"Dataset: {X.shape[0]:,} x {X.shape[1]}")
    n_oneil = np.sum(sources == 'oneil')
    n_dc = np.sum(sources == 'drugcomb')
    print(f"Sources: O'Neil={n_oneil:,}, DrugComb={n_dc:,}")
    
    # Pre-defined 5-fold
    print(f"\n{'='*70}")
    print("Pre-defined Folds (DeepSynergy v3, Combined)")
    print("=" * 70)
    
    from sklearn.model_selection import KFold
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    pf_results = []
    
    for fold, (tr_idx, te_idx) in enumerate(kf.split(X_s)):
        logger.info(f"  Fold {fold}: train={len(tr_idx):,}, test={len(te_idx):,}")
        r, _ = train_fold(X_s[tr_idx], y[tr_idx], X_s[te_idx], y[te_idx],
                          n_interact, device, epochs=300, patience=30)
        pf_results.append(r)
        print(f"  Fold {fold}: r = {r:.4f}")
    
    print(f"\n  PF Mean: r = {np.mean(pf_results):.4f} +/- {np.std(pf_results):.4f}")
    
    # LDPO
    print(f"\n{'='*70}")
    print("LDPO (DeepSynergy v3, Combined)")
    print("=" * 70)
    
    pairs = np.array([tuple(sorted([a, b])) for a, b in zip(da, db)], dtype=object)
    unique_pairs = list(set([tuple(p) for p in pairs]))
    np.random.seed(42)
    pperm = np.random.permutation(len(unique_pairs))
    pfold = len(unique_pairs) // 5
    
    ldpo_results = []
    for fold in range(5):
        s = fold * pfold
        e = s + pfold if fold < 4 else len(unique_pairs)
        test_set = set([unique_pairs[pperm[i]] for i in range(s, e)])
        te_mask = np.array([tuple(p) in test_set for p in pairs])
        tr_mask = ~te_mask
        
        if te_mask.sum() < 10: continue
        logger.info(f"  LDPO Fold {fold}: train={tr_mask.sum():,}, test={te_mask.sum():,}")
        r, _ = train_fold(X_s[tr_mask], y[tr_mask], X_s[te_mask], y[te_mask],
                          n_interact, device, epochs=300, patience=30)
        ldpo_results.append(r)
        print(f"  LDPO Fold {fold}: r = {r:.4f}")
    
    print(f"\n  LDPO Mean: r = {np.mean(ldpo_results):.4f} +/- {np.std(ldpo_results):.4f}")
    
    # LCLO
    print(f"\n{'='*70}")
    print("LCLO (DeepSynergy v3, Combined)")
    print("=" * 70)
    
    unique_cells = np.unique(cells)
    np.random.seed(42)
    cperm = np.random.permutation(len(unique_cells))
    cfold = len(unique_cells) // 5
    
    lclo_results = []
    for fold in range(5):
        s = fold * cfold
        e = s + cfold if fold < 4 else len(unique_cells)
        test_set = set(unique_cells[cperm[s:e]])
        te_mask = np.array([c in test_set for c in cells])
        tr_mask = ~te_mask
        
        if te_mask.sum() < 10: continue
        logger.info(f"  LCLO Fold {fold}: train={tr_mask.sum():,}, test={te_mask.sum():,}")
        r, _ = train_fold(X_s[tr_mask], y[tr_mask], X_s[te_mask], y[te_mask],
                          n_interact, device, epochs=300, patience=30)
        lclo_results.append(r)
        print(f"  LCLO Fold {fold}: r = {r:.4f}")
    
    print(f"\n  LCLO Mean: r = {np.mean(lclo_results):.4f} +/- {np.std(lclo_results):.4f}")
    
    # Summary
    print(f"\n{'='*70}")
    print("COMPARISON: DeepSynergy v3")
    print("=" * 70)
    print(f"  O'Neil only (23K):     PF=0.640  LDPO=0.623  LCLO=0.505")
    print(f"  Combined ({X.shape[0]//1000}K):  "
          f"PF={np.mean(pf_results):.3f}  "
          f"LDPO={np.mean(ldpo_results):.3f}  "
          f"LCLO={np.mean(lclo_results):.3f}")


if __name__ == "__main__":
    main()
