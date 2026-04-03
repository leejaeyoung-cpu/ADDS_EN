"""
Fix C (fix8): DeepSynergy DNN Architecture
============================================
Replace XGBoost with a drug-aware DNN that learns chemical structure
embeddings. Uses the expanded dataset from Fix A+B (725K records).
"""
import numpy as np
import pandas as pd
import pickle
import hashlib
import json
import time
import logging
from pathlib import Path
from scipy.stats import pearsonr

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

DATA_DIR = Path("F:/ADDS/data/ml_training")
MODEL_DIR = Path("F:/ADDS/models/synergy")

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


class DeepSynergyNet(nn.Module):
    """
    DeepSynergy-style DNN for drug synergy prediction.
    Separate branches for each drug's fingerprint → dense embedding,
    then concatenated with cell expression for final prediction.
    """
    def __init__(self, fp_dim=1024, expr_dim=256, drug_hidden=256, expr_hidden=128,
                 fusion_hidden=256, dropout=0.3):
        super().__init__()

        # Drug branch (shared weights = permutation invariant)
        self.drug_branch = nn.Sequential(
            nn.Linear(fp_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, drug_hidden),
            nn.BatchNorm1d(drug_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Cell expression branch
        self.expr_branch = nn.Sequential(
            nn.Linear(expr_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, expr_hidden),
            nn.BatchNorm1d(expr_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Fusion layers (drug_a + drug_b + expression)
        fusion_in = drug_hidden * 2 + expr_hidden
        self.fusion = nn.Sequential(
            nn.Linear(fusion_in, fusion_hidden),
            nn.BatchNorm1d(fusion_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_hidden, fusion_hidden // 2),
            nn.BatchNorm1d(fusion_hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout // 2),
            nn.Linear(fusion_hidden // 2, 1),
        )

    def forward(self, fp_a, fp_b, expr):
        da = self.drug_branch(fp_a)
        db = self.drug_branch(fp_b)
        ce = self.expr_branch(expr)
        combined = torch.cat([da, db, ce], dim=1)
        return self.fusion(combined).squeeze(-1)


def build_data():
    """Build dataset from Fix A+B results."""
    logger.info("Loading data...")

    combined = pd.read_csv(DATA_DIR / "synergy_combined.csv", low_memory=False)

    # Load Morgan FPs
    with open(MODEL_DIR / "drug_fingerprints_morgan_full.pkl", 'rb') as f:
        morgan_fps = {k.upper(): v for k, v in pickle.load(f).items()}

    # Load cell expression
    with open(MODEL_DIR / "cell_line_expression.pkl", 'rb') as f:
        oneil_expr = {k.upper(): v for k, v in pickle.load(f).items()}

    # Build CCLE index (same as fix7)
    import re
    from difflib import SequenceMatcher

    def normalize(name):
        n = name.upper().strip()
        return re.sub(r'[\s\-_/\\()]+', '', n)

    ccle = pd.read_csv(DATA_DIR / "ccle_raw" / "ccle_target_expression.csv")
    gene_cols = [c for c in ccle.columns if c != 'cell_line']
    ccle_idx = {}
    for _, row in ccle.iterrows():
        cl = str(row['cell_line']).strip()
        vec = np.array([float(row.get(g, 0)) for g in gene_cols], dtype=np.float32)
        ccle_idx[normalize(cl)] = vec
        if '_' in cl:
            ccle_idx[normalize(cl.split('_')[0])] = vec

    expr_dim = len(gene_cols)
    fp_dim = 1024

    def hash_fp(name, n=1024):
        h = hashlib.sha256(name.encode()).digest()
        bits = np.unpackbits(np.frombuffer(h, dtype=np.uint8))
        fp = np.zeros(n, dtype=np.float32)
        for i in range(n):
            fp[i] = bits[i % len(bits)]
        return fp

    # Build arrays
    fpa_list, fpb_list, expr_list, y_list = [], [], [], []
    da_list, db_list, cl_list = [], [], []

    for i, row in enumerate(combined.itertuples(index=False)):
        da = str(row.drug_a).upper().strip()
        db = str(row.drug_b).upper().strip()
        cl = str(row.cell_line).upper().strip()
        source = str(row.source)

        fp_a = morgan_fps.get(da, hash_fp(da))
        fp_b = morgan_fps.get(db, hash_fp(db))

        # Expression
        cl_norm = normalize(cl)
        expr = oneil_expr.get(cl)
        if expr is None:
            expr = ccle_idx.get(cl_norm)
        if expr is None:
            continue

        if len(expr) > expr_dim:
            expr = expr[:expr_dim]
        elif len(expr) < expr_dim:
            pad = np.zeros(expr_dim, dtype=np.float32)
            pad[:len(expr)] = expr
            expr = pad

        fpa_list.append(fp_a)
        fpb_list.append(fp_b)
        expr_list.append(expr)
        y_list.append(float(row.synergy_loewe))
        da_list.append(da)
        db_list.append(db)
        cl_list.append(cl)

        if (i + 1) % 200000 == 0:
            logger.info(f"  {i+1:,}/{len(combined):,}")

    FPA = np.array(fpa_list, dtype=np.float32)
    FPB = np.array(fpb_list, dtype=np.float32)
    EXPR = np.array(expr_list, dtype=np.float32)
    Y = np.array(y_list, dtype=np.float32)

    logger.info(f"  Dataset: {len(Y):,} samples, FP={fp_dim}, Expr={expr_dim}")

    return FPA, FPB, EXPR, Y, np.array(da_list), np.array(db_list), np.array(cl_list), fp_dim, expr_dim


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    n = 0
    for fpa, fpb, expr, y in loader:
        fpa, fpb, expr, y = fpa.to(device), fpb.to(device), expr.to(device), y.to(device)
        optimizer.zero_grad()
        pred = model(fpa, fpb, expr)
        loss = criterion(pred, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(y)
        n += len(y)
    return total_loss / n


def eval_model(model, loader, device):
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for fpa, fpb, expr, y in loader:
            fpa, fpb, expr, y = fpa.to(device), fpb.to(device), expr.to(device), y.to(device)
            pred = model(fpa, fpb, expr)
            preds.append(pred.cpu().numpy())
            targets.append(y.cpu().numpy())
    preds = np.concatenate(preds)
    targets = np.concatenate(targets)
    r, _ = pearsonr(targets, preds)
    return r


def run_cv(FPA, FPB, EXPR, Y, da, db, cl, fp_dim, expr_dim, cv_type='pf', device='cpu'):
    """Run 5-fold CV with specified split strategy."""
    np.random.seed(42)

    if cv_type == 'pf':
        # Random split
        idx = np.random.permutation(len(Y))
        folds = np.array_split(idx, 5)
    elif cv_type == 'ldpo':
        # Leave drug pairs out
        pairs = [tuple(sorted([a, b])) for a, b in zip(da, db)]
        up = list(set(pairs))
        np.random.shuffle(up)
        fold_pairs = np.array_split(up, 5)
        folds = []
        pair_arr = np.array(pairs, dtype=object)
        for fp in fold_pairs:
            fp_set = set(map(tuple, fp))
            mask = np.array([tuple(p) in fp_set for p in pair_arr])
            folds.append(np.where(mask)[0])
    elif cv_type == 'lclo':
        # Leave cell lines out
        uc = np.unique(cl)
        np.random.shuffle(uc)
        fold_cells = np.array_split(uc, 5)
        folds = []
        for fc in fold_cells:
            fc_set = set(fc)
            mask = np.array([c in fc_set for c in cl])
            folds.append(np.where(mask)[0])

    results = []
    for fold_i, test_idx in enumerate(folds):
        train_mask = np.ones(len(Y), dtype=bool)
        train_mask[test_idx] = False
        tr = np.where(train_mask)[0]
        te = test_idx

        if len(te) < 10:
            continue

        # Subsample training if too large
        if len(tr) > 150000:
            tr = np.random.choice(tr, 150000, replace=False)

        # Build dataloaders
        tr_ds = TensorDataset(
            torch.FloatTensor(FPA[tr]), torch.FloatTensor(FPB[tr]),
            torch.FloatTensor(EXPR[tr]), torch.FloatTensor(Y[tr])
        )
        te_ds = TensorDataset(
            torch.FloatTensor(FPA[te]), torch.FloatTensor(FPB[te]),
            torch.FloatTensor(EXPR[te]), torch.FloatTensor(Y[te])
        )
        tr_loader = DataLoader(tr_ds, batch_size=1024, shuffle=True, num_workers=0)
        te_loader = DataLoader(te_ds, batch_size=2048, shuffle=False, num_workers=0)

        # Model
        model = DeepSynergyNet(fp_dim=fp_dim, expr_dim=expr_dim).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)
        criterion = nn.MSELoss()

        best_r = -1
        patience_counter = 0
        for epoch in range(50):
            loss = train_epoch(model, tr_loader, optimizer, criterion, device)
            r = eval_model(model, te_loader, device)
            scheduler.step(loss)

            if r > best_r:
                best_r = r
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= 7:
                    break

        logger.info(f"  {cv_type} fold {fold_i}: r={best_r:.4f} (epochs={epoch+1})")
        results.append(best_r)

    return np.mean(results) if results else 0.0


def main():
    t0 = time.time()
    print("=" * 70)
    print("Fix C: DeepSynergy DNN Architecture")
    print("=" * 70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"  Device: {device}")

    FPA, FPB, EXPR, Y, da, db, cl, fp_dim, expr_dim = build_data()

    # Subsample for speed
    if len(Y) > 200000:
        np.random.seed(42)
        idx = np.random.choice(len(Y), 200000, replace=False)
        FPA, FPB, EXPR, Y = FPA[idx], FPB[idx], EXPR[idx], Y[idx]
        da, db, cl = da[idx], db[idx], cl[idx]
        print(f"  Subsampled to: {len(Y):,}")

    # Normalize expression (z-score per gene)
    expr_mean = EXPR.mean(axis=0)
    expr_std = EXPR.std(axis=0) + 1e-8
    EXPR = (EXPR - expr_mean) / expr_std

    # Run evaluations
    print(f"\n  Running PF (random 5-fold)...")
    pf = run_cv(FPA, FPB, EXPR, Y, da, db, cl, fp_dim, expr_dim, 'pf', device)

    print(f"\n  Running LDPO (leave drug pairs out)...")
    ldpo = run_cv(FPA, FPB, EXPR, Y, da, db, cl, fp_dim, expr_dim, 'ldpo', device)

    print(f"\n  Running LCLO (leave cell lines out)...")
    lclo = run_cv(FPA, FPB, EXPR, Y, da, db, cl, fp_dim, expr_dim, 'lclo', device)

    print(f"\n{'='*70}")
    print("RESULTS")
    print("=" * 70)
    print(f"  {'Model':50s} {'PF':>6} {'LDPO':>6} {'LCLO':>6}")
    print(f"  {'-'*50} {'-'*6} {'-'*6} {'-'*6}")
    print(f"  {'XGBoost (P2, 200K, 39cl)':50s} 0.708  0.629  0.610")
    print(f"  {'XGBoost (FixAB, 725K, 930cl)':50s} 0.665  0.576  0.571")
    print(f"  {'DeepSynergy DNN (FixAB, 200K sub, {0}cl)'.format(len(np.unique(cl))):50s} {pf:.3f}  {ldpo:.3f}  {lclo:.3f}")
    print(f"\n  Total time: {time.time()-t0:.0f}s")

    # Save
    results = {
        'model': 'DeepSynergy DNN',
        'samples': int(len(Y)),
        'fp_dim': fp_dim, 'expr_dim': expr_dim,
        'PF': float(pf), 'LDPO': float(ldpo), 'LCLO': float(lclo),
        'device': device,
    }
    with open(MODEL_DIR / "fix8_dnn_results.json", 'w') as f:
        json.dump(results, f, indent=2)
    print(f"  Saved: fix8_dnn_results.json")


if __name__ == "__main__":
    main()
