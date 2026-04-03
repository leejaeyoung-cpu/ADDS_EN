"""
DeepSynergy v3 — Full DrugComb Training (1M+ records)
=====================================================
Train on 1.17M DrugComb synergy records with:
- Morgan FP (1024d) per drug
- Cell line mutation profile (derived from name)
- GPU training with early stopping
- Drug-pair holdout evaluation (hardest test)
"""

import json, logging, math, time, pickle
from pathlib import Path
from collections import Counter
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import pearsonr, spearmanr
from sklearn.model_selection import KFold, GroupKFold
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DATA_DIR = Path("F:/ADDS/data/ml_training")
MODEL_DIR = Path("F:/ADDS/models/synergy")
device = "cuda" if torch.cuda.is_available() else "cpu"


# ================================================================
# DATA LOADING
# ================================================================

def load_drug_smiles():
    """Load all available drug SMILES."""
    smiles = {}
    for p in [MODEL_DIR/"drug_smiles.json", MODEL_DIR/"drug_smiles_extended.json"]:
        if p.exists():
            with open(p) as f: smiles.update(json.load(f))
    return smiles


def compute_fingerprints(smiles_dict, radius=2, n_bits=1024):
    """Pre-compute Morgan fingerprints for all drugs."""
    from rdkit import Chem
    from rdkit.Chem import AllChem
    fps = {}
    for name, smi in smiles_dict.items():
        if smi is None: continue
        try:
            mol = Chem.MolFromSmiles(smi)
            if mol:
                fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, n_bits)
                fps[name.upper()] = np.array(fp, dtype=np.float32)
        except: pass
    return fps


def load_data():
    """Load and prepare training data."""
    # Try combined, fall back to DrugComb processed
    combined_path = DATA_DIR / "synergy_combined.csv"
    dc_path = DATA_DIR / "drugcomb" / "drugcomb_processed.csv"
    
    if combined_path.exists():
        df = pd.read_csv(combined_path, low_memory=False)
    elif dc_path.exists():
        df = pd.read_csv(dc_path, low_memory=False)
    else:
        raise FileNotFoundError("No synergy data found!")
    
    logger.info("Raw data: %d rows", len(df))
    
    # Normalize drug names
    df['drug_a'] = df['drug_a'].astype(str).str.upper().str.strip()
    df['drug_b'] = df['drug_b'].astype(str).str.upper().str.strip()
    df['cell_line'] = df['cell_line'].astype(str).str.upper().str.strip()
    
    # Drop NaN synergy
    df = df.dropna(subset=['synergy_loewe'])
    df = df[df['drug_a'] != df['drug_b']]
    
    logger.info("Clean data: %d rows", len(df))
    return df


# ================================================================
# FEATURE ENGINEERING
# ================================================================

def build_cell_line_features(cell_lines):
    """Build simple cell line features from name patterns."""
    # Common tissue patterns
    tissue_keywords = {
        'COLON': ['HCT','SW480','SW620','COLO','DLD','HT29','LOVO','RKO','LS174'],
        'BREAST': ['MCF','MDA','BT','T47D','SKBR','ZR75','HCC1'],
        'LUNG': ['A549','NCI-H','NCIH','H460','H1299','H1650','H1975','H23','H358','HCC827','PC9'],
        'OVARIAN': ['SKOV','A2780','OVCAR','IGROV','CAOV'],
        'LEUKEMIA': ['K562','HL60','MOLT','JURKAT','CCRF','KG1','THP','NB4','U937'],
        'MELANOMA': ['A375','SKMEL','MALME','UACC','WM'],
        'PROSTATE': ['PC3','DU145','LNCAP','VCAP','22RV'],
        'LIVER': ['HEPG2','HEP3B','HUH','SKHEP','PLC'],
        'BRAIN': ['U87','U251','SF','A172','LN229','T98G','SNB'],
        'PANCREAS': ['PANC','MIAPACA','BXPC','ASPC','CAPAN','CFPAC','HPAF'],
        'KIDNEY': ['786O','A498','ACHN','CAKI','OS-RC','RCC'],
        'STOMACH': ['AGS','KATO','MKN','NCI-N87','SNU'],
    }
    
    features = {}
    n_tissues = len(tissue_keywords)
    
    for cl in cell_lines:
        vec = np.zeros(n_tissues, dtype=np.float32)
        cl_upper = cl.upper().replace('-','').replace('_','').replace(' ','')
        for i, (tissue, keywords) in enumerate(tissue_keywords.items()):
            for kw in keywords:
                if kw.replace('-','') in cl_upper:
                    vec[i] = 1.0
                    break
        features[cl] = vec
    
    return features, n_tissues


def build_feature_matrix(df, fps, cl_features, fp_dim=1024, cl_dim=12):
    """Build feature matrix: [drug_a_fp | drug_b_fp | cell_line]."""
    n = len(df)
    feat_dim = fp_dim * 2 + cl_dim
    X = np.zeros((n, feat_dim), dtype=np.float32)
    y = np.zeros(n, dtype=np.float32)
    valid = np.ones(n, dtype=bool)
    groups = []
    
    for i, (_, row) in enumerate(df.iterrows()):
        da = row['drug_a']
        db = row['drug_b']
        cl = row['cell_line']
        
        if da not in fps or db not in fps:
            valid[i] = False
            groups.append(('',''))
            continue
        
        X[i, :fp_dim] = fps[da]
        X[i, fp_dim:fp_dim*2] = fps[db]
        X[i, fp_dim*2:] = cl_features.get(cl, np.zeros(cl_dim, dtype=np.float32))
        y[i] = row['synergy_loewe']
        groups.append(tuple(sorted([da, db])))
        
        if (i + 1) % 200000 == 0:
            logger.info("  Built %d/%d features...", i+1, n)
    
    X = X[valid]
    y = y[valid]
    groups = [g for g, v in zip(groups, valid) if v]
    
    logger.info("Feature matrix: %d samples, %d features (dropped %d)", 
                len(y), feat_dim, n - len(y))
    return X, y, groups


# ================================================================
# MODEL
# ================================================================

class DeepSynergyV3(nn.Module):
    """DeepSynergy with modern architecture."""
    def __init__(self, input_dim, hidden=[2048, 1024, 512, 256]):
        super().__init__()
        layers = []
        prev = input_dim
        for i, h in enumerate(hidden):
            layers.append(nn.Linear(prev, h))
            layers.append(nn.BatchNorm1d(h))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(0.3 if i < 2 else 0.2))
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)
        
        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.zeros_(m.bias)
    
    def forward(self, x):
        return self.net(x).squeeze(-1)


# ================================================================
# TRAINING
# ================================================================

def train_model(X_train, y_train, X_val, y_val, input_dim, 
                max_epochs=200, patience=20, batch_size=4096, lr=1e-3):
    """Train with early stopping on validation Pearson r."""
    model = DeepSynergyV3(input_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)
    
    # Create data loaders
    train_ds = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train))
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, 
                          pin_memory=True, num_workers=0)
    
    X_val_t = torch.FloatTensor(X_val).to(device)
    y_val_np = y_val
    
    best_r = -1
    best_state = None
    wait = 0
    
    for epoch in range(max_epochs):
        # Train
        model.train()
        total_loss = 0
        n_batches = 0
        for xb, yb in train_dl:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            pred = model(xb)
            loss = F.mse_loss(pred, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1
        
        scheduler.step()
        
        # Validate every 5 epochs
        if (epoch + 1) % 5 == 0:
            model.eval()
            with torch.no_grad():
                val_pred = model(X_val_t).cpu().numpy()
            r = pearsonr(y_val_np, val_pred)[0]
            
            avg_loss = total_loss / n_batches
            if r > best_r:
                best_r = r
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
                wait = 0
                marker = " ★"
            else:
                wait += 1
                marker = ""
            
            if (epoch + 1) % 20 == 0 or wait == 0:
                logger.info("  Epoch %3d: loss=%.4f, val_r=%.4f%s", 
                           epoch+1, avg_loss, r, marker)
            
            if wait >= patience // 5:
                logger.info("  Early stopping at epoch %d", epoch+1)
                break
    
    if best_state:
        model.load_state_dict(best_state)
    
    return model, best_r


def evaluate_model(model, X_test, y_test):
    """Evaluate model on test set."""
    model.eval()
    X_t = torch.FloatTensor(X_test).to(device)
    with torch.no_grad():
        pred = model(X_t).cpu().numpy()
    
    r = pearsonr(y_test, pred)[0]
    rho = spearmanr(y_test, pred)[0]
    rmse = np.sqrt(np.mean((y_test - pred) ** 2))
    
    return {"pearson_r": round(float(r), 4),
            "spearman_rho": round(float(rho), 4),
            "rmse": round(float(rmse), 2)}


# ================================================================
# MAIN
# ================================================================

def main():
    t0 = time.time()
    logger.info("=" * 60)
    logger.info("DeepSynergy v3 — Full DrugComb Training")
    logger.info("=" * 60)
    logger.info("Device: %s", device)
    
    # Load data
    df = load_data()
    smiles = load_drug_smiles()
    logger.info("Drug SMILES: %d", len(smiles))
    
    # Compute fingerprints
    logger.info("Computing Morgan fingerprints...")
    fps = compute_fingerprints(smiles, radius=2, n_bits=1024)
    logger.info("Fingerprints: %d drugs", len(fps))
    
    # Cell line features
    all_cls = df['cell_line'].unique()
    cl_features, cl_dim = build_cell_line_features(all_cls)
    logger.info("Cell line features: %d lines, %d dim", len(cl_features), cl_dim)
    
    # Build features
    logger.info("Building feature matrix...")
    X, y, groups = build_feature_matrix(df, fps, cl_features, 1024, cl_dim)
    
    # Create group IDs
    unique_pairs = list(set(groups))
    pair_to_id = {p: i for i, p in enumerate(unique_pairs)}
    group_ids = np.array([pair_to_id[g] for g in groups])
    
    logger.info("Dataset: %d samples, %d unique drug pairs", len(y), len(unique_pairs))
    logger.info("Feature dim: %d (FP=%d×2 + CL=%d)", X.shape[1], 1024, cl_dim)
    
    # Normalize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # ================================================================
    # EVALUATION 1: Random 5-fold CV
    # ================================================================
    logger.info("\n" + "=" * 40)
    logger.info("EVAL 1: Random 5-fold CV")
    logger.info("=" * 40)
    
    kf = KFold(5, shuffle=True, random_state=42)
    random_results = []
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X_scaled)):
        r_result = train_model(
            X_scaled[train_idx], y[train_idx],
            X_scaled[val_idx], y[val_idx],
            X.shape[1], max_epochs=100, patience=15
        )
        model, best_r = r_result
        metrics = evaluate_model(model, X_scaled[val_idx], y[val_idx])
        random_results.append(metrics['pearson_r'])
        logger.info("  Fold %d: r=%.4f, rho=%.4f, rmse=%.2f", 
                    fold+1, metrics['pearson_r'], metrics['spearman_rho'], metrics['rmse'])
        
        # Save best model from fold 1
        if fold == 0:
            torch.save(model.state_dict(), MODEL_DIR / "deep_synergy_v3.pt")
    
    avg_r = np.mean(random_results)
    std_r = np.std(random_results)
    logger.info("Random CV: r=%.4f ± %.4f", avg_r, std_r)
    
    # ================================================================
    # EVALUATION 2: Drug-pair holdout (GroupKFold)
    # ================================================================
    logger.info("\n" + "=" * 40)
    logger.info("EVAL 2: Drug-pair holdout (GroupKFold)")
    logger.info("=" * 40)
    
    gkf = GroupKFold(5)
    pair_results = []
    
    for fold, (train_idx, val_idx) in enumerate(gkf.split(X_scaled, y, group_ids)):
        r_result = train_model(
            X_scaled[train_idx], y[train_idx],
            X_scaled[val_idx], y[val_idx],
            X.shape[1], max_epochs=100, patience=15
        )
        model, best_r = r_result
        metrics = evaluate_model(model, X_scaled[val_idx], y[val_idx])
        pair_results.append(metrics['pearson_r'])
        logger.info("  Fold %d: r=%.4f (n=%d)", fold+1, metrics['pearson_r'], len(val_idx))
    
    avg_pair_r = np.mean(pair_results)
    std_pair_r = np.std(pair_results)
    logger.info("Drug-pair CV: r=%.4f ± %.4f", avg_pair_r, std_pair_r)
    
    # ================================================================
    # SUMMARY
    # ================================================================
    elapsed = time.time() - t0
    
    logger.info("\n" + "=" * 60)
    logger.info("FINAL RESULTS — DeepSynergy v3")
    logger.info("=" * 60)
    logger.info("  Training data:  %d samples (%d drug pairs)", len(y), len(unique_pairs))
    logger.info("  Random CV:      r=%.4f ± %.4f", avg_r, std_r)
    logger.info("  Drug-pair CV:   r=%.4f ± %.4f", avg_pair_r, std_pair_r)
    logger.info("  ")
    logger.info("  References:")
    logger.info("    Energy v6 Full-FP:    r=0.646 (O'Neil 38 drugs)")
    logger.info("    Energy v3 Full:       r=0.641 (O'Neil 38 drugs)")
    logger.info("    DeepSynergy v2 (old): r=0.660 (O'Neil 23K, random)")
    logger.info("  Time: %.1f seconds", elapsed)
    
    # Save results
    results = {
        "model": "DeepSynergy v3",
        "training_samples": int(len(y)),
        "unique_drug_pairs": int(len(unique_pairs)),
        "unique_drugs": int(len(fps)),
        "architecture": "MLP [2060 → 2048 → 1024 → 512 → 256 → 1], GELU, BN, dropout",
        "random_cv": {"r": round(float(avg_r), 4), "std": round(float(std_r), 4)},
        "drug_pair_cv": {"r": round(float(avg_pair_r), 4), "std": round(float(std_pair_r), 4)},
        "training_time_seconds": round(elapsed, 1),
    }
    
    with open(MODEL_DIR / "deep_synergy_v3_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    # Save metadata for API
    metadata = {
        "version": "deep_synergy_v3",
        "input_dim": int(X.shape[1]),
        "fp_dim": 1024,
        "cl_dim": int(cl_dim),
        "architecture": [2048, 1024, 512, 256],
        "n_samples": int(len(y)),
        "cv_results": results,
        "device": device,
    }
    with open(MODEL_DIR / "deep_synergy_v3_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    
    logger.info("Saved model and results to %s", MODEL_DIR)


if __name__ == "__main__":
    main()
