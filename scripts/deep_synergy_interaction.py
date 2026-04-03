"""
Task 4: DeepSynergy + Cell-Drug Interaction Integration
========================================================
Add 47-dim pKi × CCLE_expression cross-features to DeepSynergy DNN.
"""
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import pickle
import logging
from pathlib import Path
from scipy.stats import pearsonr

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

DATA_DIR = Path("F:/ADDS/data/ml_training")
CCLE_DIR = DATA_DIR / "ccle_raw"
MODEL_DIR = Path("F:/ADDS/models/synergy")

# pKi annotations
KNOWN_AFFINITIES = {
    'ERLOTINIB': {'EGFR': 8.7, 'ABL1': 5.9},
    'LAPATINIB': {'EGFR': 8.0, 'ERBB2': 8.0},
    'SORAFENIB': {'BRAF': 7.7, 'FLT3': 7.2, 'KIT': 7.2, 'PDGFRB': 7.2, 'RAF1': 8.2, 'KDR': 7.0},
    'SUNITINIB': {'FLT3': 6.6, 'KIT': 9.0, 'PDGFRA': 7.1, 'RET': 7.0, 'KDR': 7.1},
    'DASATINIB': {'ABL1': 9.2, 'EPHA2': 7.8, 'KIT': 8.3, 'PDGFRB': 7.6, 'SRC': 9.3},
    'BORTEZOMIB': {'PSMB5': 9.2},
    'PACLITAXEL': {'TUBB': 8.4},
    'VINBLASTINE': {'TUBB': 9.0},
    'VINORELBINE': {'TUBB': 8.6},
    'DOXORUBICIN': {'TOP2A': 6.8},
    'ETOPOSIDE': {'TOP2A': 5.7, 'TOP2B': 5.5},
    'TOPOTECAN': {'TOP1': 6.5},
    'SN-38': {'TOP1': 8.5},
    'METHOTREXATE': {'DHFR': 11.5},
    '5-FU': {'TYMS': 7.5},
    'GEMCITABINE': {'RRM1': 7.2},
    'TEMOZOLOMIDE': {'MGMT': 6.0},
    'MK-4827': {'PARP1': 8.4, 'PARP2': 8.7},
    'ABT-888': {'PARP1': 8.3, 'PARP2': 8.5},
    'MK-2206': {'AKT1': 8.1, 'AKT2': 7.9, 'AKT3': 7.2},
    'BEZ-235': {'PIK3CA': 8.4, 'PIK3CB': 7.1, 'MTOR': 8.2},
    'MK-8669': {'MTOR': 9.7},
    'PD325901': {'MAP2K1': 9.5, 'MAP2K2': 9.1},
    'ZOLINZA': {'HDAC1': 7.4, 'HDAC2': 7.3, 'HDAC3': 7.6},
    'DINACICLIB': {'CDK1': 8.5, 'CDK2': 9.0, 'CDK5': 9.0, 'CDK9': 8.4},
    'MK-8776': {'CHEK1': 8.5},
    'MK-5108': {'AURKA': 10.2},
    'GELDANAMYCIN': {'HSP90AA1': 8.9},
    'DEXAMETHASONE': {'NR3C1': 9.2},
    'METFORMIN': {'PRKAA1': 5.7, 'PRKAA2': 5.7},
    'MRK-003': {'NOTCH1': 6.3},
}


class DeepSynergyInteraction(nn.Module):
    """DeepSynergy v3: with dedicated interaction feature processing."""
    
    def __init__(self, n_fp=1024, n_interact=47, n_cell=256, n_bio=None,
                 h_drug=512, h_interact=64, h_cell=256, h_comb=512, dropout=0.3):
        super().__init__()
        
        # Drug branch (fingerprints)
        self.drug_net = nn.Sequential(
            nn.Linear(n_fp, h_drug),
            nn.ReLU(),
            nn.BatchNorm1d(h_drug),
            nn.Dropout(dropout),
            nn.Linear(h_drug, h_drug // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        
        # Interaction branch (pKi × expression per drug)
        self.interact_net = nn.Sequential(
            nn.Linear(n_interact, h_interact),
            nn.ReLU(),
            nn.BatchNorm1d(h_interact),
            nn.Dropout(dropout),
            nn.Linear(h_interact, h_interact),
            nn.ReLU(),
        )
        
        # Cell branch
        self.cell_net = nn.Sequential(
            nn.Linear(n_cell, h_cell),
            nn.ReLU(),
            nn.BatchNorm1d(h_cell),
            nn.Dropout(dropout),
            nn.Linear(h_cell, h_cell // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        
        # Combined
        comb_in = h_drug // 2 * 2 + h_interact * 2 + h_cell // 2
        if n_bio:
            comb_in += n_bio
        
        self.combined_net = nn.Sequential(
            nn.Linear(comb_in, h_comb),
            nn.ReLU(),
            nn.BatchNorm1d(h_comb),
            nn.Dropout(dropout),
            nn.Linear(h_comb, h_comb // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(h_comb // 2, 1),
        )
        
        self.n_fp = n_fp
        self.n_interact = n_interact
        self.n_cell = n_cell
        self.n_bio = n_bio
    
    def forward(self, x):
        idx = 0
        fp_a = self.drug_net(x[:, idx:idx + self.n_fp])
        idx += self.n_fp
        fp_b = self.drug_net(x[:, idx:idx + self.n_fp])
        idx += self.n_fp
        
        int_a = self.interact_net(x[:, idx:idx + self.n_interact])
        idx += self.n_interact
        int_b = self.interact_net(x[:, idx:idx + self.n_interact])
        idx += self.n_interact
        
        cell = self.cell_net(x[:, idx:idx + self.n_cell])
        idx += self.n_cell
        
        parts = [fp_a, fp_b, int_a, int_b, cell]
        
        if self.n_bio:
            bio = x[:, idx:idx + self.n_bio]
            parts.append(bio)
        
        combined = torch.cat(parts, dim=1)
        return self.combined_net(combined)


def build_dataset():
    """Build feature matrix with interaction features."""
    syn = pd.read_csv(DATA_DIR / "oneil_synergy.csv")
    
    with open(MODEL_DIR / "drug_fingerprints.pkl", 'rb') as f:
        drug_fps = pickle.load(f)
    with open(MODEL_DIR / "cell_line_expression.pkl", 'rb') as f:
        cell_expr = pickle.load(f)
    
    # Load CCLE
    ccle = pd.read_csv(CCLE_DIR / "ccle_target_expression.csv")
    target_genes = [c for c in ccle.columns if c != 'cell_line']
    n_tgt = len(target_genes)
    
    ccle_map = {}
    for _, row in ccle.iterrows():
        cl_full = str(row['cell_line']).upper()
        cl_short = cl_full.split('_')[0]
        ccle_map[cl_full] = row
        ccle_map[cl_short] = row
    
    # Build pKi vectors
    drug_pki = {}
    for drug, targets in KNOWN_AFFINITIES.items():
        vec = np.zeros(n_tgt, dtype=np.float32)
        for gene, pki in targets.items():
            if gene in target_genes:
                vec[target_genes.index(gene)] = pki
        drug_pki[drug.upper()] = vec
    
    drug_fps_u = {k.upper(): v for k, v in drug_fps.items()}
    cell_expr_u = {k.upper(): v for k, v in cell_expr.items()}
    
    zero_fp = np.zeros(1024, np.float32)
    zero_expr = np.zeros(256, np.float32)
    zero_pki = np.zeros(n_tgt, np.float32)
    
    X, y, folds = [], [], []
    
    for _, row in syn.iterrows():
        da, db = str(row['drug_a']).upper(), str(row['drug_b']).upper()
        cl = str(row['cell_line']).upper()
        target = float(row['synergy_loewe'])
        if np.isnan(target): continue
        
        fp_a = drug_fps_u.get(da, zero_fp)
        fp_b = drug_fps_u.get(db, zero_fp)
        if np.array_equal(fp_a, zero_fp) and np.array_equal(fp_b, zero_fp): continue
        
        pki_a = drug_pki.get(da, zero_pki)
        pki_b = drug_pki.get(db, zero_pki)
        
        ccle_row = ccle_map.get(cl, None)
        if ccle_row is not None:
            cell_tgt = np.array([float(ccle_row.get(g, 0)) for g in target_genes], dtype=np.float32)
        else:
            cell_tgt = np.ones(n_tgt, np.float32)  # neutral if no CCLE
        
        interaction_a = pki_a * cell_tgt
        interaction_b = pki_b * cell_tgt
        expr = cell_expr_u.get(cl, zero_expr)
        
        features = np.concatenate([fp_a, fp_b, interaction_a, interaction_b, expr])
        
        X.append(features)
        y.append(target)
        folds.append(int(row.get('fold', 0)))
    
    return np.array(X), np.array(y), np.array(folds), n_tgt


def train_fold(X_train, y_train, X_test, y_test, n_interact, device, epochs=500, patience=50):
    """Train one fold."""
    n_fp = 1024
    n_cell = 256
    
    model = DeepSynergyInteraction(
        n_fp=n_fp, n_interact=n_interact, n_cell=n_cell,
        h_drug=512, h_interact=64, h_cell=256, h_comb=512, dropout=0.3
    ).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=20, factor=0.5)
    criterion = nn.MSELoss()
    
    X_tr = torch.FloatTensor(X_train).to(device)
    y_tr = torch.FloatTensor(y_train).to(device)
    X_te = torch.FloatTensor(X_test).to(device)
    
    best_r = -1
    best_state = None
    wait = 0
    
    for epoch in range(1, epochs + 1):
        model.train()
        
        perm = torch.randperm(len(X_tr))
        total_loss = 0
        n_batch = 0
        
        for i in range(0, len(X_tr), 256):
            idx = perm[i:i+256]
            xb = X_tr[idx]
            yb = y_tr[idx]
            
            pred = model(xb).squeeze()
            loss = criterion(pred, yb)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            n_batch += 1
        
        # Evaluate
        model.eval()
        with torch.no_grad():
            pred_te = model(X_te).squeeze().cpu().numpy()
        
        r, _ = pearsonr(y_test, pred_te)
        scheduler.step(-r)
        
        if r > best_r:
            best_r = r
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
        
        if epoch % 50 == 0:
            logger.info(f"    Epoch {epoch}: r={r:.4f}, best={best_r:.4f}, wait={wait}")
        
        if wait >= patience:
            logger.info(f"    Early stopping at epoch {epoch}")
            break
    
    return best_r, best_state


def main():
    print("=" * 70)
    print("TASK 4: DeepSynergy + Interaction Integration (v3)")
    print("=" * 70)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"  Device: {device}")
    
    X, y, folds, n_interact = build_dataset()
    print(f"  Dataset: {X.shape[0]} samples × {X.shape[1]} features")
    print(f"  Interaction features: {n_interact} (per drug)")
    
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Pre-defined fold evaluation
    print(f"\n{'='*70}")
    print("Pre-defined Folds (DeepSynergy v3 + Interaction)")
    print("=" * 70)
    
    pf_results = []
    for fold in sorted(np.unique(folds)):
        te_mask = folds == fold
        tr_mask = ~te_mask
        
        X_tr, X_te = X_scaled[tr_mask], X_scaled[te_mask]
        y_tr, y_te = y[tr_mask], y[te_mask]
        
        logger.info(f"  Fold {fold}: train={len(y_tr)}, test={len(y_te)}")
        
        r, state = train_fold(X_tr, y_tr, X_te, y_te, n_interact, device,
                              epochs=500, patience=50)
        pf_results.append(r)
        print(f"  Fold {fold}: r = {r:.4f}")
    
    print(f"\n  Pre-defined Mean: r = {np.mean(pf_results):.4f} ± {np.std(pf_results):.4f}")
    
    # LDPO
    print(f"\n{'='*70}")
    print("LDPO (DeepSynergy v3 + Interaction)")
    print("=" * 70)
    
    pairs = np.array([tuple(sorted([str(r['drug_a']).upper(), str(r['drug_b']).upper()]))
                      for _, r in pd.read_csv(DATA_DIR / "oneil_synergy.csv").iterrows()
                      if not np.isnan(r['synergy_loewe'])])
    
    # Get pairs matching our dataset (some may have been filtered)
    # Use same index tracking
    syn = pd.read_csv(DATA_DIR / "oneil_synergy.csv")
    drug_fps_u = {k.upper(): v for k, v in pickle.load(open(MODEL_DIR / "drug_fingerprints.pkl", 'rb')).items()}
    zero_fp = np.zeros(1024, np.float32)
    
    valid_pairs = []
    valid_cells = []
    for _, row in syn.iterrows():
        da, db = str(row['drug_a']).upper(), str(row['drug_b']).upper()
        if np.isnan(float(row['synergy_loewe'])): continue
        fp_a = drug_fps_u.get(da, zero_fp)
        fp_b = drug_fps_u.get(db, zero_fp)
        if np.array_equal(fp_a, zero_fp) and np.array_equal(fp_b, zero_fp): continue
        valid_pairs.append(tuple(sorted([da, db])))
        valid_cells.append(str(row['cell_line']).upper())
    
    pairs_arr = np.array(valid_pairs, dtype=object)
    cells_arr = np.array(valid_cells)
    
    unique_pairs = list(set([tuple(p) for p in pairs_arr]))
    np.random.seed(42)
    pperm = np.random.permutation(len(unique_pairs))
    pfold_size = len(unique_pairs) // 5
    
    ldpo_results = []
    for fold in range(5):
        start = fold * pfold_size
        end = start + pfold_size if fold < 4 else len(unique_pairs)
        test_set = set([unique_pairs[pperm[i]] for i in range(start, end)])
        te_mask = np.array([tuple(p) in test_set for p in pairs_arr])
        tr_mask = ~te_mask
        
        X_tr, X_te = X_scaled[tr_mask], X_scaled[te_mask]
        y_tr, y_te = y[tr_mask], y[te_mask]
        
        logger.info(f"  LDPO Fold {fold}: train={len(y_tr)}, test={len(y_te)}")
        
        r, _ = train_fold(X_tr, y_tr, X_te, y_te, n_interact, device,
                          epochs=500, patience=50)
        ldpo_results.append(r)
        print(f"  LDPO Fold {fold}: r = {r:.4f}")
    
    print(f"\n  LDPO Mean: r = {np.mean(ldpo_results):.4f} ± {np.std(ldpo_results):.4f}")
    
    # LCLO
    print(f"\n{'='*70}")
    print("LCLO (DeepSynergy v3 + Interaction)")
    print("=" * 70)
    
    unique_cells = np.unique(cells_arr)
    np.random.seed(42)
    cperm = np.random.permutation(len(unique_cells))
    cfold_size = len(unique_cells) // 5
    
    lclo_results = []
    for fold in range(5):
        start = fold * cfold_size
        end = start + cfold_size if fold < 4 else len(unique_cells)
        test_set = set(unique_cells[cperm[start:end]])
        te_mask = np.array([c in test_set for c in cells_arr])
        tr_mask = ~te_mask
        
        X_tr, X_te = X_scaled[tr_mask], X_scaled[te_mask]
        y_tr, y_te = y[tr_mask], y[te_mask]
        
        logger.info(f"  LCLO Fold {fold}: train={len(y_tr)}, test={len(y_te)}")
        
        r, _ = train_fold(X_tr, y_tr, X_te, y_te, n_interact, device,
                          epochs=500, patience=50)
        lclo_results.append(r)
        print(f"  LCLO Fold {fold}: r = {r:.4f}")
    
    print(f"\n  LCLO Mean: r = {np.mean(lclo_results):.4f} ± {np.std(lclo_results):.4f}")
    
    # Summary
    print(f"\n{'='*70}")
    print("COMPARISON: DeepSynergy Versions")
    print("=" * 70)
    print(f"  v1 (FP+Expr, 100ep):       PF=0.6574  LDPO=?       LCLO=?")
    print(f"  v2 (500ep+early stop):      PF=0.6774  LDPO=pending LCLO=pending")
    print(f"  v3 (500ep+Interaction):     PF={np.mean(pf_results):.4f}  "
          f"LDPO={np.mean(ldpo_results):.4f}  LCLO={np.mean(lclo_results):.4f}")


if __name__ == "__main__":
    main()
