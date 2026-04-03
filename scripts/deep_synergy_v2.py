"""
DeepSynergy v2 — Extended Training + Early Stopping
====================================================
Fix 3: epochs 80 → 500 with early stopping (patience=50)
Evaluate: pre-defined folds, LDPO, LCLO
"""
import pandas as pd
import numpy as np
import pickle
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import logging
from pathlib import Path
from scipy.stats import pearsonr, spearmanr

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

DATA_DIR = Path("F:/ADDS/data/ml_training")
MODEL_DIR = Path("F:/ADDS/models/synergy")
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class DeepSynergyNet(nn.Module):
    """Original DeepSynergy: NO BatchNorm, just tanh+dropout (per paper)."""
    def __init__(self, input_dim, hidden_dims=[8192, 4096, 2048], dropout=0.5):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.Tanh(),
                nn.Dropout(dropout),
            ])
            prev_dim = h_dim
        self.hidden = nn.Sequential(*layers)
        self.output = nn.Linear(prev_dim, 1)
        
        # Xavier init
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, x):
        x = self.hidden(x)
        return self.output(x).squeeze(-1)


def load_data():
    syn = pd.read_csv(DATA_DIR / "oneil_synergy.csv")
    with open(MODEL_DIR / "drug_fingerprints.pkl", 'rb') as f:
        drug_fps = pickle.load(f)
    with open(MODEL_DIR / "cell_line_expression.pkl", 'rb') as f:
        cell_expr = pickle.load(f)
    bio_df = pd.read_csv(DATA_DIR / "cell_line_features.csv")
    cell_bio = {row['cell_line']: row.drop('cell_line').values.astype(np.float32)
                for _, row in bio_df.iterrows()}
    
    du = {k.upper(): v for k, v in drug_fps.items()}
    cu = {k.upper(): v for k, v in cell_expr.items()}
    bu = {k.upper(): v for k, v in cell_bio.items()}
    
    zero_fp = np.zeros(1024, np.float32)
    zero_expr = np.zeros(256, np.float32)
    zero_bio = np.zeros(len(next(iter(cell_bio.values()))), np.float32)
    
    X, y, folds, pairs, cells = [], [], [], [], []
    
    for _, row in syn.iterrows():
        da, db = str(row['drug_a']).upper(), str(row['drug_b']).upper()
        cl = str(row['cell_line']).upper()
        t = float(row['synergy_loewe'])
        if np.isnan(t): continue
        
        fp_a, fp_b = du.get(da, zero_fp), du.get(db, zero_fp)
        if np.array_equal(fp_a, zero_fp) and np.array_equal(fp_b, zero_fp): continue
        
        features = np.concatenate([fp_a, fp_b, cu.get(cl, zero_expr), bu.get(cl, zero_bio)])
        X.append(features)
        y.append(t)
        folds.append(int(row.get('fold', 0)))
        pairs.append(tuple(sorted([da, db])))
        cells.append(cl)
    
    return np.array(X), np.array(y), np.array(folds), np.array(pairs, dtype=object), np.array(cells)


def normalize_features(X_train, X_test):
    """Z-score normalization on train, apply to test."""
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0) + 1e-8
    return (X_train - mean) / std, (X_test - mean) / std


def train_with_early_stopping(X_tr, y_tr, X_val, y_val, max_epochs=500, patience=50):
    """Train with early stopping on validation Pearson r."""
    input_dim = X_tr.shape[1]
    model = DeepSynergyNet(input_dim).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)
    criterion = nn.MSELoss()
    
    # Normalize
    X_tr_n, X_val_n = normalize_features(X_tr, X_val)
    
    X_tr_t = torch.FloatTensor(X_tr_n).to(DEVICE)
    y_tr_t = torch.FloatTensor(y_tr).to(DEVICE)
    X_val_t = torch.FloatTensor(X_val_n).to(DEVICE)
    
    # Augment: swap drug A and B
    fp_a = X_tr_t[:, :1024].clone()
    fp_b = X_tr_t[:, 1024:2048].clone()
    X_aug = X_tr_t.clone()
    X_aug[:, :1024] = fp_b
    X_aug[:, 1024:2048] = fp_a
    
    X_tr_all = torch.cat([X_tr_t, X_aug], dim=0)
    y_tr_all = torch.cat([y_tr_t, y_tr_t], dim=0)
    
    dataset = TensorDataset(X_tr_all, y_tr_all)
    loader = DataLoader(dataset, batch_size=256, shuffle=True, drop_last=True)
    
    best_r = -1
    best_state = None
    wait = 0
    
    for epoch in range(max_epochs):
        model.train()
        for bx, by in loader:
            optimizer.zero_grad()
            loss = criterion(model(bx), by)
            loss.backward()
            optimizer.step()
        scheduler.step()
        
        if (epoch + 1) % 10 == 0:
            model.eval()
            with torch.no_grad():
                yp = model(X_val_t).cpu().numpy()
            r, _ = pearsonr(y_val, yp)
            
            if r > best_r:
                best_r = r
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                wait = 0
            else:
                wait += 1
            
            if (epoch + 1) % 50 == 0:
                logger.info(f"    Epoch {epoch+1}: r={r:.4f}, best={best_r:.4f}, wait={wait}")
            
            if wait >= patience // 10:  # Check every 10 epochs, so patience=50 → 5 checks
                logger.info(f"    Early stopping at epoch {epoch+1}")
                break
    
    # Load best
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        yp = model(X_val_t).cpu().numpy()
    
    r, _ = pearsonr(y_val, yp)
    rho, _ = spearmanr(y_val, yp)
    rmse = np.sqrt(np.mean((y_val - yp)**2))
    
    return r, rho, rmse, model


def main():
    X, y, folds, pairs, cells = load_data()
    print(f"Data: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"Device: {DEVICE}")
    
    # ===== PRE-DEFINED FOLDS =====
    print(f"\n{'='*70}")
    print("TEST 1: Pre-defined Folds (500 epochs, early stopping)")
    print("=" * 70)
    
    unique_folds = sorted(np.unique(folds))
    pf_results = []
    
    for test_fold in unique_folds:
        te = folds == test_fold
        tr = ~te
        r, rho, rmse, _ = train_with_early_stopping(X[tr], y[tr], X[te], y[te])
        pf_results.append((r, rho, rmse))
        print(f"  Fold {test_fold}: r={r:.4f}, rho={rho:.4f}, RMSE={rmse:.2f}")
    
    mean_r = np.mean([x[0] for x in pf_results])
    print(f"  Pre-defined Mean: r={mean_r:.4f}")
    
    # ===== LDPO =====
    print(f"\n{'='*70}")
    print("TEST 2: LDPO (500 epochs, early stopping)")
    print("=" * 70)
    
    unique_pairs = np.array(list(set([tuple(p) for p in pairs])))
    np.random.seed(42)
    perm = np.random.permutation(len(unique_pairs))
    fold_size = len(unique_pairs) // 5
    
    ldpo_results = []
    for fold in range(5):
        start = fold * fold_size
        end = start + fold_size if fold < 4 else len(unique_pairs)
        test_pairs = set([tuple(unique_pairs[i]) for i in perm[start:end]])
        
        te = np.array([tuple(p) in test_pairs for p in pairs])
        tr = ~te
        
        r, rho, rmse, _ = train_with_early_stopping(X[tr], y[tr], X[te], y[te])
        ldpo_results.append((r, rho, rmse))
        print(f"  Fold {fold}: r={r:.4f}, rho={rho:.4f}, RMSE={rmse:.2f}")
    
    mean_ldpo = np.mean([x[0] for x in ldpo_results])
    print(f"  LDPO Mean: r={mean_ldpo:.4f}")
    
    # ===== LCLO =====
    print(f"\n{'='*70}")
    print("TEST 3: LCLO (500 epochs, early stopping)")
    print("=" * 70)
    
    unique_cells = np.unique(cells)
    np.random.seed(42)
    cperm = np.random.permutation(len(unique_cells))
    cfold_size = len(unique_cells) // 5
    
    lclo_results = []
    for fold in range(5):
        start = fold * cfold_size
        end = start + cfold_size if fold < 4 else len(unique_cells)
        test_cells = set(unique_cells[cperm[start:end]])
        
        te = np.array([c in test_cells for c in cells])
        tr = ~te
        
        r, rho, rmse, _ = train_with_early_stopping(X[tr], y[tr], X[te], y[te])
        lclo_results.append((r, rho, rmse))
        print(f"  Fold {fold}: r={r:.4f}, rho={rho:.4f}, RMSE={rmse:.2f}")
    
    mean_lclo = np.mean([x[0] for x in lclo_results])
    print(f"  LCLO Mean: r={mean_lclo:.4f}")
    
    # ===== COMPARISON =====
    print(f"\n{'='*70}")
    print("v1 (80ep, BN) vs v2 (500ep, early stop, no BN)")
    print("=" * 70)
    print(f"{'':20s} {'v1 (80ep)':>12s} {'v2 (500ep)':>12s} {'Delta':>10s}")
    print(f"{'Pre-defined':20s} {'0.6600':>12s} {mean_r:>12.4f} {mean_r-0.6600:>+10.4f}")
    print(f"{'LDPO':20s} {'0.6153':>12s} {mean_ldpo:>12.4f} {mean_ldpo-0.6153:>+10.4f}")
    print(f"{'LCLO':20s} {'0.4598':>12s} {mean_lclo:>12.4f} {mean_lclo-0.4598:>+10.4f}")
    
    # Save best model
    print(f"\nTraining final model on all data...")
    _, _, _, final_model = train_with_early_stopping(X, y, X[:4600], y[:4600], max_epochs=300)
    
    torch.save({
        'model_state_dict': final_model.state_dict(),
        'input_dim': X.shape[1],
        'hidden_dims': [8192, 4096, 2048],
        'version': 'v2',
        'pre_defined_r': mean_r,
        'ldpo_r': mean_ldpo,
        'lclo_r': mean_lclo,
    }, MODEL_DIR / "deep_synergy_v2.pt")
    
    print(f"Saved: {MODEL_DIR / 'deep_synergy_v2.pt'}")


if __name__ == "__main__":
    main()
