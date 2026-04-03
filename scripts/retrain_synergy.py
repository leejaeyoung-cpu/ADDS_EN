"""
DeepSynergy + XGBoost 재학습 스크립트 (확장 데이터)
기존 23K → 1M+ 확장 데이터로 시너지 예측 모델 재학습

사용법:
    python scripts/retrain_synergy.py [--xgb-only] [--deep-only] [--sample N]

출력:
    models/synergy/deep_synergy_v3.pt          - DeepSynergy v3 (확장 데이터)
    models/synergy/xgboost_synergy_v4_*.pkl    - XGBoost v4 (확장 데이터)
    models/synergy/retrain_results.json        - 학습 결과 비교
"""

import argparse
import json
import logging
import os
import pickle
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DATA_DIR = Path("F:/ADDS/data/ml_training")
MODEL_DIR = Path("F:/ADDS/models/synergy")
COMBINED_CSV = DATA_DIR / "synergy_combined.csv"


def load_data(sample_n: int = 0) -> pd.DataFrame:
    """Load combined synergy data."""
    logger.info(f"Loading data from {COMBINED_CSV}")
    df = pd.read_csv(COMBINED_CSV, low_memory=False)
    df = df.dropna(subset=["synergy_loewe"])

    if sample_n > 0 and len(df) > sample_n:
        logger.info(f"Sampling {sample_n:,} rows from {len(df):,}")
        df = df.sample(n=sample_n, random_state=42).reset_index(drop=True)

    logger.info(f"Loaded: {len(df):,} rows")
    return df


def load_drug_smiles() -> Dict[str, str]:
    """Load all available SMILES mappings."""
    smiles = {}
    for path in [
        MODEL_DIR / "drug_smiles.json",
        MODEL_DIR / "drug_smiles_extended.json",
        DATA_DIR / "drugcomb" / "drug_smiles_all.json",
    ]:
        if path.exists():
            with open(path) as f:
                new = json.load(f)
                smiles.update(new)
    logger.info(f"Loaded {len(smiles)} drug SMILES")
    return smiles


def compute_fingerprints(smiles_dict: Dict[str, str], radius: int = 2, nbits: int = 1024) -> Dict[str, np.ndarray]:
    """Compute Morgan fingerprints for all drugs."""
    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem
    except ImportError:
        logger.error("RDKit required! pip install rdkit-pypi")
        return {}

    fps = {}
    for name, smi in smiles_dict.items():
        mol = Chem.MolFromSmiles(smi)
        if mol:
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=nbits)
            fps[name] = np.array(fp, dtype=np.float32)

    logger.info(f"Computed fingerprints: {len(fps)}/{len(smiles_dict)} drugs")
    return fps


def build_features(df: pd.DataFrame, fps: Dict[str, np.ndarray]) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Build feature matrix from drug pair fingerprints."""
    X_list = []
    y_list = []
    meta_list = []

    known = set(fps.keys())

    for _, row in df.iterrows():
        da = str(row["drug_a"])
        db = str(row["drug_b"])

        if da not in known or db not in known:
            continue

        x = np.concatenate([fps[da], fps[db]])
        X_list.append(x)
        y_list.append(float(row["synergy_loewe"]))
        meta_list.append({"drug_a": da, "drug_b": db, "cell_line": str(row.get("cell_line", ""))})

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.float32)
    meta = pd.DataFrame(meta_list)

    logger.info(f"Features: X={X.shape}, y={y.shape} ({len(df)-len(X_list):,} rows dropped, no FP)")
    return X, y, meta


def train_xgboost(X_train, y_train, X_val, y_val) -> Dict:
    """Train XGBoost regressor."""
    import xgboost as xgb
    from scipy.stats import pearsonr, spearmanr
    from sklearn.metrics import mean_squared_error, mean_absolute_error

    logger.info("=== Training XGBoost ===")

    model = xgb.XGBRegressor(
        n_estimators=500,
        max_depth=8,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,
        reg_alpha=0.1,
        reg_lambda=1.0,
        tree_method="hist",
        device="cuda:0",
        n_jobs=-1,
        random_state=42,
        early_stopping_rounds=30,
    )

    t0 = time.time()
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=50,
    )
    train_time = time.time() - t0

    # Evaluate
    y_pred_train = model.predict(X_train)
    y_pred_val = model.predict(X_val)

    results = {
        "train_pearson_r": float(pearsonr(y_train, y_pred_train)[0]),
        "train_spearman_r": float(spearmanr(y_train, y_pred_train)[0]),
        "train_rmse": float(np.sqrt(mean_squared_error(y_train, y_pred_train))),
        "val_pearson_r": float(pearsonr(y_val, y_pred_val)[0]),
        "val_spearman_r": float(spearmanr(y_val, y_pred_val)[0]),
        "val_rmse": float(np.sqrt(mean_squared_error(y_val, y_pred_val))),
        "val_mae": float(mean_absolute_error(y_val, y_pred_val)),
        "train_time_sec": round(train_time, 1),
        "n_train": len(X_train),
        "n_val": len(X_val),
    }

    logger.info(f"XGBoost Results:")
    logger.info(f"  Train r={results['train_pearson_r']:.4f}, RMSE={results['train_rmse']:.2f}")
    logger.info(f"  Val   r={results['val_pearson_r']:.4f}, RMSE={results['val_rmse']:.2f}")

    return {"model": model, "results": results}


def train_deep_synergy(X_train, y_train, X_val, y_val, epochs: int = 100, batch_size: int = 4096) -> Dict:
    """Train DeepSynergy MLP model."""
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    logger.info("=== Training DeepSynergy v3 ===")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    # Convert to tensors
    X_train_t = torch.FloatTensor(X_train).to(device)
    y_train_t = torch.FloatTensor(y_train).unsqueeze(1).to(device)
    X_val_t = torch.FloatTensor(X_val).to(device)
    y_val_t = torch.FloatTensor(y_val).unsqueeze(1).to(device)

    train_ds = TensorDataset(X_train_t, y_train_t)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    input_dim = X_train.shape[1]

    # DeepSynergy architecture (matches original paper)
    class DeepSynergyV3(nn.Module):
        def __init__(self, input_dim):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(input_dim, 4096),
                nn.Tanh(),
                nn.Dropout(0.5),
                nn.Linear(4096, 2048),
                nn.Tanh(),
                nn.Dropout(0.5),
                nn.Linear(2048, 512),
                nn.Tanh(),
                nn.Dropout(0.3),
                nn.Linear(512, 128),
                nn.Tanh(),
                nn.Dropout(0.2),
                nn.Linear(128, 1),
            )

        def forward(self, x):
            return self.net(x)

    model = DeepSynergyV3(input_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.MSELoss()

    best_val_loss = float("inf")
    best_state = None
    history = []

    t0 = time.time()
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for xb, yb in train_loader:
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(xb)
        train_loss /= len(train_ds)

        model.eval()
        with torch.no_grad():
            val_pred = model(X_val_t)
            val_loss = criterion(val_pred, y_val_t).item()

        scheduler.step()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        if (epoch + 1) % 10 == 0 or epoch == 0:
            logger.info(f"  Epoch {epoch+1:3d}/{epochs}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")
            history.append({"epoch": epoch+1, "train_loss": train_loss, "val_loss": val_loss})

    train_time = time.time() - t0

    # Load best model and evaluate
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        y_pred_val = model(X_val_t).cpu().numpy().flatten()
        y_pred_train = model(X_train_t).cpu().numpy().flatten()

    from scipy.stats import pearsonr, spearmanr

    results = {
        "train_pearson_r": float(pearsonr(y_train, y_pred_train)[0]),
        "train_spearman_r": float(spearmanr(y_train, y_pred_train)[0]),
        "train_rmse": float(np.sqrt(np.mean((y_train - y_pred_train) ** 2))),
        "val_pearson_r": float(pearsonr(y_val, y_pred_val)[0]),
        "val_spearman_r": float(spearmanr(y_val, y_pred_val)[0]),
        "val_rmse": float(np.sqrt(np.mean((y_val - y_pred_val) ** 2))),
        "val_mae": float(np.mean(np.abs(y_val - y_pred_val))),
        "best_val_loss": float(best_val_loss),
        "train_time_sec": round(train_time, 1),
        "epochs": epochs,
        "n_train": len(X_train),
        "n_val": len(X_val),
    }

    logger.info(f"DeepSynergy v3 Results:")
    logger.info(f"  Train r={results['train_pearson_r']:.4f}, RMSE={results['train_rmse']:.2f}")
    logger.info(f"  Val   r={results['val_pearson_r']:.4f}, RMSE={results['val_rmse']:.2f}")

    return {"model": model, "results": results, "history": history}


def main():
    parser = argparse.ArgumentParser(description="Retrain synergy models with expanded data")
    parser.add_argument("--xgb-only", action="store_true", help="Only train XGBoost")
    parser.add_argument("--deep-only", action="store_true", help="Only train DeepSynergy")
    parser.add_argument("--sample", type=int, default=0, help="Sample N rows (0=all)")
    parser.add_argument("--epochs", type=int, default=100, help="DeepSynergy epochs")
    parser.add_argument("--batch-size", type=int, default=4096, help="Batch size")
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("시너지 모델 재학습 (확장 데이터)")
    logger.info("=" * 60)

    # Load data
    df = load_data(sample_n=args.sample)
    smiles = load_drug_smiles()
    fps = compute_fingerprints(smiles)

    if not fps:
        logger.error("No fingerprints computed — aborting")
        return

    # Build features
    X, y, meta = build_features(df, fps)

    if len(X) < 1000:
        logger.error(f"Too few samples with fingerprints: {len(X)}")
        return

    # Train/val split (80/20, stratified by source roughly)
    from sklearn.model_selection import train_test_split
    idx_train, idx_val = train_test_split(
        np.arange(len(X)), test_size=0.2, random_state=42
    )
    X_train, X_val = X[idx_train], X[idx_val]
    y_train, y_val = y[idx_train], y[idx_val]

    logger.info(f"Train: {len(X_train):,}, Val: {len(X_val):,}")

    all_results = {}

    # XGBoost training
    if not args.deep_only:
        xgb_out = train_xgboost(X_train, y_train, X_val, y_val)

        # Save
        xgb_path = MODEL_DIR / "xgboost_synergy_v4_expanded.pkl"
        with open(xgb_path, "wb") as f:
            pickle.dump(xgb_out["model"], f)
        logger.info(f"Saved: {xgb_path}")

        all_results["xgboost_v4"] = xgb_out["results"]

    # DeepSynergy training
    if not args.xgb_only:
        import torch

        deep_out = train_deep_synergy(
            X_train, y_train, X_val, y_val,
            epochs=args.epochs,
            batch_size=args.batch_size,
        )

        # Save
        ds_path = MODEL_DIR / "deep_synergy_v3.pt"
        torch.save({
            "model_state_dict": deep_out["model"].state_dict(),
            "input_dim": X.shape[1],
            "architecture": "DeepSynergy v3 (4096-2048-512-128-1, tanh, dropout)",
            "data": f"DrugComb+O'Neil expanded ({len(X):,} samples)",
            "results": deep_out["results"],
            "history": deep_out["history"],
        }, ds_path)
        logger.info(f"Saved: {ds_path}")

        all_results["deep_synergy_v3"] = deep_out["results"]

    # Load previous results for comparison
    prev_results = {}
    prev_meta = MODEL_DIR / "deep_synergy_metadata.json"
    if prev_meta.exists():
        with open(prev_meta) as f:
            meta_data = json.load(f)
            prev_results["deep_synergy_v2"] = meta_data.get("cv_results", {})

    # Summary
    all_results["previous_baseline"] = prev_results
    all_results["data_stats"] = {
        "total_rows": len(df),
        "rows_with_fp": len(X),
        "train_size": len(X_train),
        "val_size": len(X_val),
        "n_drugs_with_fp": len(fps),
    }

    results_path = MODEL_DIR / "retrain_results.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)

    logger.info(f"\n{'='*60}")
    logger.info(f"결과 저장: {results_path}")

    if "xgboost_v4" in all_results:
        r = all_results["xgboost_v4"]
        logger.info(f"XGBoost v4: val r={r['val_pearson_r']:.4f}, RMSE={r['val_rmse']:.2f}")

    if "deep_synergy_v3" in all_results:
        r = all_results["deep_synergy_v3"]
        logger.info(f"DeepSynergy v3: val r={r['val_pearson_r']:.4f}, RMSE={r['val_rmse']:.2f}")

    if prev_results.get("deep_synergy_v2"):
        p = prev_results["deep_synergy_v2"]
        logger.info(f"Previous v2 baseline: r={p.get('pearson_r', 'N/A')}")

    logger.info("=" * 60)


if __name__ == "__main__":
    main()
