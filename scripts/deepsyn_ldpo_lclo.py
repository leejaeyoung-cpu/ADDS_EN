"""
DeepSynergy LDPO + LCLO — The REAL generalization test
======================================================
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
    def __init__(self, input_dim, hidden_dims=[4096, 2048, 512], dropout=0.5):
        super().__init__()
        self.input_norm = nn.BatchNorm1d(input_dim)
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


def load_data():
    syn = pd.read_csv(DATA_DIR / "oneil_synergy.csv")
    with open(MODEL_DIR / "drug_fingerprints.pkl", 'rb') as f:
        drug_fps = pickle.load(f)
    with open(MODEL_DIR / "cell_line_expression.pkl", 'rb') as f:
        cell_expr = pickle.load(f)
    
    bio_df = pd.read_csv(DATA_DIR / "cell_line_features.csv")
    cell_bio = {row['cell_line']: row.drop('cell_line').values.astype(np.float32) for _, row in bio_df.iterrows()}
    
    drug_fps_u = {k.upper(): v for k, v in drug_fps.items()}
    cell_expr_u = {k.upper(): v for k, v in cell_expr.items()}
    cell_bio_u = {k.upper(): v for k, v in cell_bio.items()}
    
    zero_fp = np.zeros(1024, np.float32)
    zero_expr = np.zeros(256, np.float32)
    zero_bio = np.zeros(len(next(iter(cell_bio.values()))), np.float32)
    
    X_list, y_list, pairs_list, cells_list = [], [], [], []
    
    for _, row in syn.iterrows():
        da, db = str(row['drug_a']).upper(), str(row['drug_b']).upper()
        cl = str(row['cell_line']).upper()
        target = float(row['synergy_loewe'])
        if np.isnan(target):
            continue
        
        fp_a = drug_fps_u.get(da, zero_fp)
        fp_b = drug_fps_u.get(db, zero_fp)
        if np.array_equal(fp_a, zero_fp) and np.array_equal(fp_b, zero_fp):
            continue
        
        expr = cell_expr_u.get(cl, zero_expr)
        bio = cell_bio_u.get(cl, zero_bio)
        
        features = np.concatenate([fp_a, fp_b, expr, bio])
        X_list.append(features)
        y_list.append(target)
        pairs_list.append(tuple(sorted([da, db])))
        cells_list.append(cl)
    
    X = np.array(X_list)
    y = np.array(y_list)
    pairs = np.array(pairs_list, dtype=object)
    cells = np.array(cells_list)
    
    return X, y, pairs, cells


def train_eval_fold(X_tr, y_tr, X_te, y_te, epochs=80):
    """Train DeepSynergy on one fold and evaluate."""
    input_dim = X_tr.shape[1]
    model = DeepSynergyNet(input_dim).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
    criterion = nn.MSELoss()
    
    X_tr_t = torch.FloatTensor(X_tr).to(DEVICE)
    y_tr_t = torch.FloatTensor(y_tr).to(DEVICE)
    X_te_t = torch.FloatTensor(X_te).to(DEVICE)
    
    dataset = TensorDataset(X_tr_t, y_tr_t)
    loader = DataLoader(dataset, batch_size=256, shuffle=True, drop_last=True)
    
    model.train()
    for epoch in range(epochs):
        for batch_x, batch_y in loader:
            optimizer.zero_grad()
            pred = model(batch_x)
            loss = criterion(pred, batch_y)
            loss.backward()
            optimizer.step()
    
    model.eval()
    with torch.no_grad():
        y_pred = model(X_te_t).cpu().numpy()
    
    r, _ = pearsonr(y_te, y_pred)
    rho, _ = spearmanr(y_te, y_pred)
    rmse = np.sqrt(np.mean((y_te - y_pred)**2))
    
    return r, rho, rmse


def main():
    X, y, pairs, cells = load_data()
    print(f"Data: {X.shape[0]} samples, {X.shape[1]} features")
    
    # ==========================================
    # LDPO: Leave-Drug-Pair-Out
    # ==========================================
    print(f"\n{'='*70}")
    print("LDPO: Leave-Drug-Pair-Out (5-fold)")
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
        
        te_mask = np.array([tuple(p) in test_pairs for p in pairs])
        tr_mask = ~te_mask
        
        r, rho, rmse = train_eval_fold(X[tr_mask], y[tr_mask], X[te_mask], y[te_mask])
        ldpo_results.append((r, rho, rmse))
        print(f"  Fold {fold}: r={r:.4f}, rho={rho:.4f}, RMSE={rmse:.2f} (test={te_mask.sum()})")
    
    mean_r = np.mean([r[0] for r in ldpo_results])
    mean_rho = np.mean([r[1] for r in ldpo_results])
    print(f"  LDPO Mean: r={mean_r:.4f}, rho={mean_rho:.4f}")
    
    # ==========================================
    # LCLO: Leave-Cell-Line-Out
    # ==========================================
    print(f"\n{'='*70}")
    print("LCLO: Leave-Cell-Line-Out (5-fold)")
    print("=" * 70)
    
    unique_cells = np.unique(cells)
    np.random.seed(42)
    cell_perm = np.random.permutation(len(unique_cells))
    cell_fold_size = len(unique_cells) // 5
    
    lclo_results = []
    for fold in range(5):
        start = fold * cell_fold_size
        end = start + cell_fold_size if fold < 4 else len(unique_cells)
        test_cells = set(unique_cells[cell_perm[start:end]])
        
        te_mask = np.array([c in test_cells for c in cells])
        tr_mask = ~te_mask
        
        r, rho, rmse = train_eval_fold(X[tr_mask], y[tr_mask], X[te_mask], y[te_mask])
        lclo_results.append((r, rho, rmse))
        print(f"  Fold {fold}: r={r:.4f}, rho={rho:.4f}, RMSE={rmse:.2f} (test={te_mask.sum()}, cells={len(test_cells)})")
    
    mean_r = np.mean([r[0] for r in lclo_results])
    mean_rho = np.mean([r[1] for r in lclo_results])
    print(f"  LCLO Mean: r={mean_r:.4f}, rho={mean_rho:.4f}")
    
    # ==========================================
    # Summary comparison
    # ==========================================
    print(f"\n{'='*70}")
    print("COMPARISON: Pre-defined vs LDPO vs LCLO")
    print("=" * 70)
    print(f"  Pre-defined Folds:  r=0.6600 (same pairs can appear in train+test)")
    print(f"  LDPO (pair-out):    r={np.mean([r[0] for r in ldpo_results]):.4f}")
    print(f"  LCLO (cell-out):    r={np.mean([r[0] for r in lclo_results]):.4f}")
    print(f"\n  The GAP between pre-defined and LDPO reveals how much the model")
    print(f"  relies on memorizing drug pair effects vs truly predicting synergy")


if __name__ == "__main__":
    main()
