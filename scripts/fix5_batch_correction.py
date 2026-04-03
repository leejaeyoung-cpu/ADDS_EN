"""
Priority 5: DrugComb Batch Correction (v2 — fully vectorized)
===============================================================
Problem: O'Neil mean=4.94, std=22.78 vs DrugComb mean=-3.16, std=10.25
Fix: Compare 4 batch correction strategies with XGBoost
"""
import numpy as np
import pandas as pd
import pickle
import hashlib
import json
import time
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

DATA_DIR = Path("F:/ADDS/data/ml_training")
MODEL_DIR = Path("F:/ADDS/models/synergy")


def hash_fp(name, n=1024):
    h = hashlib.sha256(name.encode()).digest()
    bits = np.unpackbits(np.frombuffer(h, dtype=np.uint8))
    fp = np.zeros(n, dtype=np.float32)
    for i in range(n):
        fp[i] = bits[i % len(bits)]
    return fp


def main():
    t0 = time.time()
    print("=" * 70)
    print("Priority 5: Batch Correction (v2 — vectorized)")
    print("=" * 70)

    from xgboost import XGBRegressor
    from scipy.stats import pearsonr
    from sklearn.model_selection import KFold

    # ── Load data ──
    combined = pd.read_csv(DATA_DIR / "synergy_combined.csv", low_memory=False)
    print(f"\n  Combined: {len(combined):,} records")

    # Distribution analysis
    print("\n  Synergy Score Distribution:")
    for src in ['oneil', 'drugcomb']:
        d = combined[combined['source'] == src]['synergy_loewe']
        print(f"    {src:10s}: n={len(d):>8,}  mean={d.mean():>7.2f}  std={d.std():>7.2f}")

    # ── Load FPs + expression ──
    with open(MODEL_DIR / "drug_fingerprints_morgan.pkl", 'rb') as f:
        morgan_fps = {k.upper(): v for k, v in pickle.load(f).items()}

    with open(MODEL_DIR / "cell_line_expression.pkl", 'rb') as f:
        oneil_expr = {k.upper(): v for k, v in pickle.load(f).items()}

    ccle = pd.read_csv(DATA_DIR / "ccle_raw" / "ccle_target_expression.csv")
    gene_cols = [c for c in ccle.columns if c != 'cell_line']
    ccle_expr_map = {}
    for _, row in ccle.iterrows():
        cl = str(row['cell_line']).upper()
        gv = np.array([float(row.get(g, 0)) for g in gene_cols], dtype=np.float32)
        e256 = np.zeros(256, dtype=np.float32)
        e256[:len(gv)] = gv
        ccle_expr_map[cl] = e256
        short = cl.split('_')[0]
        if short not in ccle_expr_map:
            ccle_expr_map[short] = e256

    # ── Pre-build feature lookup tables ──
    # Drug FP lookup (all unique drugs → numpy array)
    all_drugs = sorted(set(combined['drug_a'].str.upper()) | set(combined['drug_b'].str.upper()))
    drug_fp_map = {}
    for d in all_drugs:
        drug_fp_map[d] = morgan_fps.get(d, hash_fp(d))

    # Cell expression lookup (combine oneil + ccle)
    cell_expr_map = {}
    for cl, expr in oneil_expr.items():
        cell_expr_map[cl] = expr
    for cl, expr in ccle_expr_map.items():
        if cl not in cell_expr_map:
            cell_expr_map[cl] = expr

    # ── Pre-filter: only keep rows with resolvable expression ──
    combined['da_upper'] = combined['drug_a'].str.upper().str.strip()
    combined['db_upper'] = combined['drug_b'].str.upper().str.strip()
    combined['cl_upper'] = combined['cell_line'].str.upper().str.strip()
    combined['has_expr'] = combined['cl_upper'].isin(cell_expr_map)
    filtered = combined[combined['has_expr']].copy().reset_index(drop=True)
    print(f"\n  Filtered (real expression only): {len(filtered):,} records")
    print(f"  Dropped: {len(combined) - len(filtered):,}")

    # ── Vectorized feature building (once) ──
    logger.info("Building feature matrix (vectorized)...")
    N = len(filtered)
    X = np.zeros((N, 2304), dtype=np.float32)  # 1024 + 1024 + 256

    for i in range(N):
        da = filtered.iloc[i]['da_upper']
        db = filtered.iloc[i]['db_upper']
        cl = filtered.iloc[i]['cl_upper']
        X[i, :1024] = drug_fp_map.get(da, hash_fp(da))
        X[i, 1024:2048] = drug_fp_map.get(db, hash_fp(db))
        X[i, 2048:] = cell_expr_map.get(cl, np.zeros(256, dtype=np.float32))

        if (i + 1) % 100000 == 0:
            logger.info(f"  {i+1:,}/{N:,} rows ({time.time()-t0:.0f}s)")

    logger.info(f"  Feature matrix built: {X.shape} ({time.time()-t0:.0f}s)")

    da_arr = filtered['da_upper'].values
    db_arr = filtered['db_upper'].values
    cl_arr = filtered['cl_upper'].values
    y_raw = filtered['synergy_loewe'].values.astype(np.float32)

    # ── Prepare batch correction variants of y ──
    # 1. No correction
    y_none = y_raw.copy()

    # 2. Source z-score
    y_src_z = y_raw.copy()
    for src in ['oneil', 'drugcomb']:
        mask = filtered['source'].values == src
        m, s = y_raw[mask].mean(), max(y_raw[mask].std(), 1e-6)
        y_src_z[mask] = (y_raw[mask] - m) / s

    # 3. Pair z-score per source
    y_pair_z = y_raw.copy()
    pair_key = np.where(da_arr <= db_arr, da_arr + '||' + db_arr, db_arr + '||' + da_arr)
    src_arr = filtered['source'].values
    combo = np.array([f"{s}|{p}" for s, p in zip(src_arr, pair_key)])
    unique_combos = np.unique(combo)
    for c in unique_combos:
        mask = combo == c
        if mask.sum() < 2:
            continue
        m, s = y_raw[mask].mean(), max(y_raw[mask].std(), 1e-6)
        y_pair_z[mask] = (y_raw[mask] - m) / s

    # 4. Quantile matching (DrugComb → O'Neil)
    from scipy.stats import rankdata
    y_quant = y_raw.copy()
    oneil_mask = filtered['source'].values == 'oneil'
    dc_mask = ~oneil_mask
    if dc_mask.sum() > 0 and oneil_mask.sum() > 0:
        oneil_sorted = np.sort(y_raw[oneil_mask])
        dc_ranks = rankdata(y_raw[dc_mask]) / (dc_mask.sum() + 1)
        y_quant[dc_mask] = np.quantile(oneil_sorted, dc_ranks)

    print(f"\n  Corrected y stats:")
    for name, yv in [('none', y_none), ('source_z', y_src_z),
                      ('pair_z', y_pair_z), ('quantile', y_quant)]:
        print(f"    {name:12s}: mean={yv.mean():>7.3f}  std={yv.std():>7.3f}")

    # ── Evaluate each correction ──
    corrections = [
        ('none', 'No correction (baseline)', y_none),
        ('source_zscore', 'Z-score per source', y_src_z),
        ('pair_zscore', 'Z-score per drug pair', y_pair_z),
        ('quantile', "Quantile match DC→O'Neil", y_quant),
    ]

    print(f"\n{'='*70}")
    print("Batch Correction Comparison")
    print("=" * 70)

    all_results = {}
    for corr_name, label, y_corr in corrections:
        logger.info(f"Evaluating {corr_name}...")

        # Subsample if needed
        if len(X) > 200000:
            np.random.seed(42)
            idx = np.random.choice(len(X), 200000, replace=False)
            Xs, ys = X[idx], y_corr[idx]
            da_s, db_s, cl_s = da_arr[idx], db_arr[idx], cl_arr[idx]
        else:
            Xs, ys, da_s, db_s, cl_s = X, y_corr, da_arr, db_arr, cl_arr

        # PF
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        pf_r = []
        for tr, te in kf.split(Xs):
            m = XGBRegressor(n_estimators=300, max_depth=6, learning_rate=0.1,
                             subsample=0.8, colsample_bytree=0.8, n_jobs=-1,
                             tree_method='hist', random_state=42)
            m.fit(Xs[tr], ys[tr], eval_set=[(Xs[te], ys[te])], verbose=False)
            r, _ = pearsonr(ys[te], m.predict(Xs[te]))
            pf_r.append(r)

        # LDPO
        pairs = [tuple(sorted([a, b])) for a, b in zip(da_s, db_s)]
        pairs_np = np.array(pairs, dtype=object)
        up = list(set(pairs))
        np.random.seed(42)
        pp = np.random.permutation(len(up))
        pf_sz = len(up) // 5
        ldpo_r = []
        for fold in range(5):
            s, e = fold * pf_sz, (fold + 1) * pf_sz if fold < 4 else len(up)
            ts = set([up[pp[i]] for i in range(s, e)])
            te = np.array([tuple(p) in ts for p in pairs_np])
            tr = ~te
            if te.sum() < 10: continue
            m = XGBRegressor(n_estimators=300, max_depth=6, learning_rate=0.1,
                             subsample=0.8, colsample_bytree=0.8, n_jobs=-1,
                             tree_method='hist', random_state=42)
            m.fit(Xs[tr], ys[tr], eval_set=[(Xs[te], ys[te])], verbose=False)
            r, _ = pearsonr(ys[te], m.predict(Xs[te]))
            ldpo_r.append(r)

        # LCLO
        uc = np.unique(cl_s)
        np.random.seed(42)
        cp = np.random.permutation(len(uc))
        cf = len(uc) // 5
        lclo_r = []
        for fold in range(5):
            s, e = fold * cf, (fold + 1) * cf if fold < 4 else len(uc)
            ts = set(uc[cp[s:e]])
            te = np.array([c in ts for c in cl_s])
            tr = ~te
            if te.sum() < 10: continue
            m = XGBRegressor(n_estimators=300, max_depth=6, learning_rate=0.1,
                             subsample=0.8, colsample_bytree=0.8, n_jobs=-1,
                             tree_method='hist', random_state=42)
            m.fit(Xs[tr], ys[tr], eval_set=[(Xs[te], ys[te])], verbose=False)
            r, _ = pearsonr(ys[te], m.predict(Xs[te]))
            lclo_r.append(r)

        pf = np.mean(pf_r)
        ldpo = np.mean(ldpo_r) if ldpo_r else 0
        lclo = np.mean(lclo_r) if lclo_r else 0
        all_results[corr_name] = {'PF': pf, 'LDPO': ldpo, 'LCLO': lclo}
        print(f"  {label:30s} PF={pf:.4f}  LDPO={ldpo:.4f}  LCLO={lclo:.4f}")

    # Summary
    print(f"\n{'='*70}")
    print("FINAL COMPARISON (all models)")
    print("=" * 70)
    print(f"  {'Version':40s} {'PF':>6s} {'LDPO':>6s} {'LCLO':>6s}")
    print(f"  {'-'*40} {'-'*6} {'-'*6} {'-'*6}")
    print(f"  {'Hash FP+Hash Expr (O Neil 23K)':40s} 0.605  0.604  0.510")
    print(f"  {'Morgan FP+Real Expr (P2 200K)':40s} 0.708  0.629  0.610")
    for corr_name, label, _ in corrections:
        r = all_results[corr_name]
        print(f"  {f'P5: {label}':40s} {r['PF']:.3f}  {r['LDPO']:.3f}  {r['LCLO']:.3f}")

    best = max(all_results.items(), key=lambda x: x[1]['LCLO'])
    print(f"\n  Best batch correction: {best[0]} (LCLO={best[1]['LCLO']:.4f})")
    print(f"  Total time: {time.time()-t0:.0f}s")

    # Save
    out = {'results': {k: v for k, v in all_results.items()},
           'best': best[0]}
    with open(MODEL_DIR / "batch_correction_results.json", 'w') as f:
        json.dump(out, f, indent=2)
    print(f"  Saved: batch_correction_results.json")


if __name__ == "__main__":
    main()
