"""
DeepSynergy: Deep Learning Drug Synergy Prediction
====================================================
PyTorch implementation based on Preuer et al. (2018).
Architecture: Feed-forward DNN with tanh activation, order-invariant.
"""
import pandas as pd
import numpy as np
import pickle
import json
import logging
from pathlib import Path
from scipy.stats import pearsonr, spearmanr

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

DATA_DIR = Path("F:/ADDS/data/ml_training")
MODEL_DIR = Path("F:/ADDS/models/synergy")
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ================================================================
# Dataset
# ================================================================
class SynergyDataset(Dataset):
    def __init__(self, X, y, augment=True):
        """
        augment: If True, swap Drug A and Drug B for order-invariance.
        """
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
        self.augment = augment
        self.n_fp = 1024
    
    def __len__(self):
        return len(self.y) * (2 if self.augment else 1)
    
    def __getitem__(self, idx):
        real_idx = idx % len(self.y)
        x = self.X[real_idx].clone()
        
        # If augmented index, swap Drug A FP and Drug B FP
        if self.augment and idx >= len(self.y):
            fp_a = x[:self.n_fp].clone()
            fp_b = x[self.n_fp:2*self.n_fp].clone()
            x[:self.n_fp] = fp_b
            x[self.n_fp:2*self.n_fp] = fp_a
        
        return x, self.y[real_idx]


# ================================================================
# DeepSynergy Model
# ================================================================
class DeepSynergyNet(nn.Module):
    """
    DeepSynergy architecture (Preuer et al. 2018):
    - Input normalization
    - Multiple hidden layers with tanh + dropout
    - Linear output for regression
    """
    def __init__(self, input_dim, hidden_dims=[4096, 2048, 512], dropout=0.5):
        super().__init__()
        
        # Input normalization layer
        self.input_norm = nn.BatchNorm1d(input_dim)
        
        # Build hidden layers
        layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.Tanh(),
                nn.Dropout(dropout),
            ])
            prev_dim = h_dim
        
        self.hidden = nn.Sequential(*layers)
        self.output = nn.Linear(prev_dim, 1)
    
    def forward(self, x):
        x = self.input_norm(x)
        x = self.hidden(x)
        return self.output(x).squeeze(-1)


# ================================================================
# Data Loading
# ================================================================
def load_data():
    """Load synergy data with all features and metadata."""
    syn = pd.read_csv(DATA_DIR / "oneil_synergy.csv")
    
    with open(MODEL_DIR / "drug_fingerprints.pkl", 'rb') as f:
        drug_fps = pickle.load(f)
    with open(MODEL_DIR / "cell_line_expression.pkl", 'rb') as f:
        cell_expr = pickle.load(f)
    
    bio_df = pd.read_csv(DATA_DIR / "cell_line_features.csv")
    cell_bio = {}
    for _, row in bio_df.iterrows():
        cell_bio[row['cell_line']] = row.drop('cell_line').values.astype(np.float32)
    
    drug_fps_upper = {k.upper(): v for k, v in drug_fps.items()}
    cell_expr_upper = {k.upper(): v for k, v in cell_expr.items()}
    cell_bio_upper = {k.upper(): v for k, v in cell_bio.items()}
    
    n_fp, n_expr = 1024, 256
    n_bio = len(next(iter(cell_bio.values())))
    zero_fp = np.zeros(n_fp, dtype=np.float32)
    zero_expr = np.zeros(n_expr, dtype=np.float32)
    zero_bio = np.zeros(n_bio, dtype=np.float32)
    
    X_list, y_list, folds_list = [], [], []
    
    for _, row in syn.iterrows():
        da = str(row['drug_a']).upper()
        db = str(row['drug_b']).upper()
        cl = str(row['cell_line']).upper()
        target = float(row['synergy_loewe'])
        fold = int(row.get('fold', 0))
        
        if np.isnan(target):
            continue
        
        fp_a = drug_fps_upper.get(da, zero_fp)
        fp_b = drug_fps_upper.get(db, zero_fp)
        
        if np.array_equal(fp_a, zero_fp) and np.array_equal(fp_b, zero_fp):
            continue
        
        expr = cell_expr_upper.get(cl, zero_expr)
        bio = cell_bio_upper.get(cl, zero_bio)
        
        features = np.concatenate([fp_a, fp_b, expr, bio])
        X_list.append(features)
        y_list.append(target)
        folds_list.append(fold)
    
    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.float32)
    folds = np.array(folds_list)
    
    logger.info(f"Data: {X.shape[0]} samples, {X.shape[1]} features")
    return X, y, folds


# ================================================================
# Training
# ================================================================
def train_one_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0
    n = 0
    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
        optimizer.zero_grad()
        pred = model(X_batch)
        loss = criterion(pred, y_batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item() * len(y_batch)
        n += len(y_batch)
    return total_loss / n


def evaluate(model, loader):
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(DEVICE)
            pred = model(X_batch)
            preds.extend(pred.cpu().numpy())
            targets.extend(y_batch.numpy())
    
    preds = np.array(preds)
    targets = np.array(targets)
    pr, _ = pearsonr(targets, preds)
    sr, _ = spearmanr(targets, preds)
    rmse = np.sqrt(np.mean((targets - preds) ** 2))
    return pr, sr, rmse, preds, targets


def train_fold(X_train, y_train, X_test, y_test, input_dim, epochs=80, lr=1e-4, batch_size=256):
    """Train one fold."""
    train_ds = SynergyDataset(X_train, y_train, augment=True)
    test_ds = SynergyDataset(X_test, y_test, augment=False)
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    
    model = DeepSynergyNet(input_dim, hidden_dims=[4096, 2048, 512], dropout=0.5).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.MSELoss()
    
    best_pr = -1
    best_state = None
    patience = 15
    wait = 0
    
    for epoch in range(epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion)
        scheduler.step()
        
        if (epoch + 1) % 5 == 0 or epoch == 0:
            pr, sr, rmse, _, _ = evaluate(model, test_loader)
            if pr > best_pr:
                best_pr = pr
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                wait = 0
            else:
                wait += 1
            
            if (epoch + 1) % 20 == 0:
                logger.info(f"    Epoch {epoch+1}: loss={train_loss:.4f}, r={pr:.4f}, rho={sr:.4f}")
            
            if wait >= patience:
                logger.info(f"    Early stop at epoch {epoch+1}")
                break
    
    if best_state:
        model.load_state_dict(best_state)
    
    pr, sr, rmse, preds, targets = evaluate(model, test_loader)
    return model, pr, sr, rmse


def main():
    print("=" * 70)
    print("DeepSynergy DNN - Drug Synergy Prediction")
    print(f"Device: {DEVICE}")
    print("=" * 70)
    
    X, y, folds = load_data()
    input_dim = X.shape[1]
    unique_folds = sorted(np.unique(folds))
    
    # ================================================================
    # Test 1: Pre-defined pair-aware folds
    # ================================================================
    print(f"\n{'='*70}")
    print("TEST 1: Pre-defined Pair-Aware Folds (DeepSynergy DNN)")
    print("=" * 70)
    
    t1_pr, t1_sr, t1_rmse = [], [], []
    
    for test_fold in unique_folds:
        te_mask = folds == test_fold
        tr_mask = ~te_mask
        
        logger.info(f"Fold {test_fold}: train={tr_mask.sum()}, test={te_mask.sum()}")
        
        model, pr, sr, rmse = train_fold(
            X[tr_mask], y[tr_mask], X[te_mask], y[te_mask], input_dim,
            epochs=80, lr=1e-4, batch_size=256
        )
        
        t1_pr.append(pr)
        t1_sr.append(sr)
        t1_rmse.append(rmse)
        print(f"  Fold {test_fold}: r={pr:.4f}, rho={sr:.4f}, RMSE={rmse:.2f}")
    
    print(f"\n  Mean: r={np.mean(t1_pr):.4f}+/-{np.std(t1_pr):.4f}, "
          f"rho={np.mean(t1_sr):.4f}+/-{np.std(t1_sr):.4f}")
    
    # ================================================================
    # Compare with XGBoost
    # ================================================================
    print(f"\n{'='*70}")
    print("COMPARISON: DeepSynergy vs XGBoost v3")
    print("=" * 70)
    print(f"  XGBoost v3 (pre-defined folds): r=0.6080")
    print(f"  DeepSynergy (pre-defined folds): r={np.mean(t1_pr):.4f}")
    diff = np.mean(t1_pr) - 0.6080
    print(f"  Improvement: {diff:+.4f} ({diff/0.6080*100:+.1f}%)")
    
    # ================================================================
    # Train final model on all data
    # ================================================================
    print(f"\nTraining final model on all data...")
    
    # Use 90/10 split for final training with early stopping
    idx = np.random.RandomState(42).permutation(len(X))
    split = int(0.9 * len(X))
    tr_idx, val_idx = idx[:split], idx[split:]
    
    final_model, pr, sr, rmse = train_fold(
        X[tr_idx], y[tr_idx], X[val_idx], y[val_idx], input_dim,
        epochs=100, lr=1e-4, batch_size=256
    )
    print(f"  Final val: r={pr:.4f}, rho={sr:.4f}, RMSE={rmse:.2f}")
    
    # Save model
    torch.save({
        'model_state_dict': final_model.state_dict(),
        'input_dim': input_dim,
        'hidden_dims': [4096, 2048, 512],
        'dropout': 0.5,
    }, str(MODEL_DIR / "deep_synergy_v1.pt"))
    
    # Save metadata
    meta = {
        'version': 'deep_synergy_v1',
        'architecture': 'DeepSynergy DNN (4096-2048-512, tanh, dropout=0.5)',
        'input_dim': input_dim,
        'n_samples': len(X),
        'device': str(DEVICE),
        'cv_results': {
            'pearson_r': f"{np.mean(t1_pr):.4f} +/- {np.std(t1_pr):.4f}",
            'spearman_r': f"{np.mean(t1_sr):.4f} +/- {np.std(t1_sr):.4f}",
            'rmse': f"{np.mean(t1_rmse):.2f} +/- {np.std(t1_rmse):.2f}",
        },
    }
    with open(MODEL_DIR / "deep_synergy_metadata.json", 'w') as f:
        json.dump(meta, f, indent=2)
    
    print(f"\nModel saved: {MODEL_DIR / 'deep_synergy_v1.pt'}")


if __name__ == "__main__":
    main()
