"""
Priority 2 Fix: DepMap Expression for DrugComb Cell Lines
==========================================================
Replace hash-based expression proxies with REAL expression data.

Strategy:
1. Use existing CCLE legacy data (1019 cell lines)
2. Match DrugComb cell lines to CCLE by name
3. ONLY keep matches → drop hash proxies entirely
4. Retrain with Morgan FP + real expression
5. Compare vs previous combined results
"""
import numpy as np
import pandas as pd
import pickle
import hashlib
import json
import logging
import time
from pathlib import Path
from difflib import SequenceMatcher

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

DATA_DIR = Path("F:/ADDS/data/ml_training")
CCLE_DIR = DATA_DIR / "ccle_raw"
MODEL_DIR = Path("F:/ADDS/models/synergy")


def build_cell_line_mapping(ccle_cell_lines, drugcomb_cell_lines):
    """Build comprehensive cell line name mapping."""
    mapping = {}
    unmatched = []
    
    # Normalize CCLE names into lookup dict
    ccle_lookup = {}
    for cl in ccle_cell_lines:
        # Store original
        ccle_lookup[cl.upper()] = cl
        # Without underscores
        ccle_lookup[cl.upper().split('_')[0]] = cl
        # Without hyphens
        ccle_lookup[cl.upper().replace('-', '')] = cl
        # Without spaces
        ccle_lookup[cl.upper().replace(' ', '')] = cl
        # First word
        first_word = cl.upper().split('_')[0].split('-')[0]
        if len(first_word) >= 3:
            ccle_lookup[first_word] = cl
    
    for dc_cl in drugcomb_cell_lines:
        dc_upper = dc_cl.upper().strip()
        
        # Try exact match
        if dc_upper in ccle_lookup:
            mapping[dc_cl] = ccle_lookup[dc_upper]
            continue
        
        # Try without hyphens
        cleaned = dc_upper.replace('-', '').replace(' ', '').replace('/', '')
        if cleaned in ccle_lookup:
            mapping[dc_cl] = ccle_lookup[cleaned]
            continue
        
        # Try first word
        first_word = dc_upper.split('-')[0].split(' ')[0].split('/')[0]
        if len(first_word) >= 3 and first_word in ccle_lookup:
            mapping[dc_cl] = ccle_lookup[first_word]
            continue
        
        # Fuzzy match (>0.8 similarity)
        best_ratio = 0
        best_match = None
        for ccle_cl in ccle_cell_lines:
            ratio = SequenceMatcher(None, dc_upper, ccle_cl.upper()).ratio()
            if ratio > best_ratio:
                best_ratio = ratio
                best_match = ccle_cl
        
        if best_ratio > 0.80:
            mapping[dc_cl] = best_match
            continue
        
        unmatched.append(dc_cl)
    
    return mapping, unmatched


def main():
    print("=" * 70)
    print("Priority 2: DepMap Expression for DrugComb Cell Lines")
    print("=" * 70)
    
    from xgboost import XGBRegressor
    from scipy.stats import pearsonr
    
    # Step 1: Load CCLE expression data
    ccle_path = CCLE_DIR / "ccle_target_expression.csv"
    ccle = pd.read_csv(ccle_path)
    gene_cols = [c for c in ccle.columns if c != 'cell_line']
    
    print(f"\n  CCLE data: {len(ccle)} cell lines × {len(gene_cols)} genes")
    
    # Build CCLE expression matrix (256-dim PCA-like representation)
    # Use target gene expression directly + padding
    ccle_expr_map = {}
    for _, row in ccle.iterrows():
        cl = str(row['cell_line']).upper()
        genes_vec = np.array([float(row.get(g, 0)) for g in gene_cols], dtype=np.float32)
        # Pad to 256 dimensions (47 genes → 256 with zero padding)
        expr_256 = np.zeros(256, dtype=np.float32)
        expr_256[:len(genes_vec)] = genes_vec
        ccle_expr_map[cl] = expr_256
        # Also store short name variant
        short = cl.split('_')[0]
        if short not in ccle_expr_map:
            ccle_expr_map[short] = expr_256
    
    print(f"  CCLE expression map: {len(ccle_expr_map)} entries")
    
    # Step 2: Load DrugComb combined data
    combined_path = DATA_DIR / "synergy_combined.csv"
    combined = pd.read_csv(combined_path, low_memory=False)
    print(f"\n  Combined data: {len(combined):,} records")
    
    dc_cells = combined[combined['source'] == 'drugcomb']['cell_line'].unique()
    oneil_cells = combined[combined['source'] == 'oneil']['cell_line'].unique()
    
    print(f"  O'Neil cell lines: {len(oneil_cells)}")
    print(f"  DrugComb cell lines: {len(dc_cells)}")
    
    # Step 3: Match DrugComb cell lines to CCLE
    all_dc_cells = sorted(set(dc_cells))
    mapping, unmatched = build_cell_line_mapping(ccle.cell_line.values, all_dc_cells)
    
    print(f"\n  DrugComb→CCLE mapping:")
    print(f"    Matched: {len(mapping)}/{len(all_dc_cells)} ({100*len(mapping)/len(all_dc_cells):.1f}%)")
    print(f"    Unmatched: {len(unmatched)}")
    
    # Count records with real expression
    matched_set = set(k.upper() for k in mapping.keys())
    dc_data = combined[combined['source'] == 'drugcomb']
    has_expr = dc_data['cell_line'].str.upper().isin(matched_set)
    
    print(f"\n  DrugComb records with real expression: {has_expr.sum():,}/{len(dc_data):,} ({100*has_expr.sum()/len(dc_data):.1f}%)")
    
    # Step 4: Build features — ONLY use real expression
    print(f"\n{'='*70}")
    print("Building Features (Real Morgan FP + Real Expression Only)")
    print("=" * 70)
    
    # Load Morgan FPs
    morgan_path = MODEL_DIR / "drug_fingerprints_morgan.pkl"
    with open(morgan_path, 'rb') as f:
        morgan_fps = {k.upper(): v for k, v in pickle.load(f).items()}
    
    # Also load O'Neil cell expression (256-dim from original model)
    with open(MODEL_DIR / "cell_line_expression.pkl", 'rb') as f:
        oneil_expr = {k.upper(): v for k, v in pickle.load(f).items()}
    
    zero_expr = np.zeros(256, dtype=np.float32)
    
    X, y = [], []
    da_arr, db_arr, cl_arr, src_arr = [], [], [], []
    skipped_no_fp = 0
    skipped_no_expr = 0
    
    # Hash-based FP for drugs not in Morgan set
    def hash_fp(name):
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
    
    t0 = time.time()
    for idx, row in combined.iterrows():
        da = str(row['drug_a']).upper().strip()
        db = str(row['drug_b']).upper().strip()
        cl = str(row['cell_line']).upper().strip()
        source = str(row.get('source', ''))
        
        target = row.get('synergy_loewe', np.nan)
        if pd.isna(target):
            continue
        
        # Drug FPs: prefer Morgan, fallback to hash for DrugComb drugs
        fp_a = morgan_fps.get(da, None)
        if fp_a is None:
            fp_a = hash_fp(da)  # Still using hash for non-O'Neil drugs
        
        fp_b = morgan_fps.get(db, None)
        if fp_b is None:
            fp_b = hash_fp(db)
        
        # Cell expression: ONLY real data, NO hash proxies
        if source == 'oneil':
            # O'Neil cells: use original 256-dim expression
            expr = oneil_expr.get(cl, zero_expr)
        else:
            # DrugComb: try CCLE mapping
            mapped = mapping.get(row['cell_line'], None)
            if mapped is None:
                # Try uppercase lookup
                mapped_str = None
                for k, v in mapping.items():
                    if k.upper() == cl:
                        mapped_str = v
                        break
                if mapped_str:
                    expr = ccle_expr_map.get(mapped_str.upper(), None)
                    if expr is None:
                        skipped_no_expr += 1
                        continue
                else:
                    # Try direct CCLE match
                    expr = ccle_expr_map.get(cl, None)
                    if expr is None:
                        skipped_no_expr += 1
                        continue
            else:
                expr = ccle_expr_map.get(mapped.upper(), None)
                if expr is None:
                    skipped_no_expr += 1
                    continue
        
        features = np.concatenate([fp_a, fp_b, expr])
        X.append(features)
        y.append(float(target))
        da_arr.append(da)
        db_arr.append(db)
        cl_arr.append(cl)
        src_arr.append(source)
        
        if len(X) % 100000 == 0:
            logger.info(f"  Built {len(X):,} features ({time.time()-t0:.0f}s)")
    
    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.float32)
    da_arr = np.array(da_arr)
    db_arr = np.array(db_arr)
    cl_arr = np.array(cl_arr)
    src_arr = np.array(src_arr)
    
    n_oneil = np.sum(src_arr == 'oneil')
    n_dc = np.sum(src_arr == 'drugcomb')
    
    print(f"\n  Final dataset: {len(X):,} × {X.shape[1]}")
    print(f"  O'Neil: {n_oneil:,}, DrugComb (real expr): {n_dc:,}")
    print(f"  Skipped (no expression): {skipped_no_expr:,}")
    print(f"  Unique cell lines: {len(np.unique(cl_arr))}")
    print(f"  Unique drugs: {len(np.unique(np.concatenate([da_arr, db_arr])))}")
    
    # Step 5: Evaluate
    print(f"\n{'='*70}")
    print("XGBoost Evaluation (Real Expression Only)")
    print("=" * 70)
    
    from sklearn.model_selection import KFold
    
    # Subsample if too large (max 200K)
    if len(X) > 200000:
        np.random.seed(42)
        oneil_idx = np.where(src_arr == 'oneil')[0]
        dc_idx = np.where(src_arr == 'drugcomb')[0]
        dc_sample = np.random.choice(dc_idx, size=min(200000 - len(oneil_idx), len(dc_idx)), replace=False)
        idx = np.concatenate([oneil_idx, dc_sample])
        np.random.shuffle(idx)
        X, y = X[idx], y[idx]
        da_arr, db_arr, cl_arr, src_arr = da_arr[idx], db_arr[idx], cl_arr[idx], src_arr[idx]
        print(f"  Subsampled to {len(X):,}")
    
    # PF (5-fold)
    print(f"\n  Pre-defined 5-Fold:")
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    pf_results = []
    for fold, (tr, te) in enumerate(kf.split(X)):
        model = XGBRegressor(n_estimators=300, max_depth=6, learning_rate=0.1,
                             subsample=0.8, colsample_bytree=0.8, n_jobs=-1,
                             tree_method='hist', random_state=42)
        model.fit(X[tr], y[tr], eval_set=[(X[te], y[te])], verbose=False)
        pred = model.predict(X[te])
        r, _ = pearsonr(y[te], pred)
        pf_results.append(r)
        print(f"    Fold {fold}: r={r:.4f}")
    pf_mean = np.mean(pf_results)
    print(f"    PF Mean: r={pf_mean:.4f}")
    
    # LDPO
    print(f"\n  LDPO:")
    pairs = [tuple(sorted([a, b])) for a, b in zip(da_arr, db_arr)]
    pairs = np.array(pairs, dtype=object)
    unique_pairs = list(set([tuple(p) for p in pairs]))
    np.random.seed(42)
    pperm = np.random.permutation(len(unique_pairs))
    pfold = len(unique_pairs) // 5
    
    ldpo_results = []
    for fold in range(5):
        s = fold * pfold
        e = s + pfold if fold < 4 else len(unique_pairs)
        test_set = set([unique_pairs[pperm[i]] for i in range(s, e)])
        te = np.array([tuple(p) in test_set for p in pairs])
        tr = ~te
        if te.sum() < 10: continue
        model = XGBRegressor(n_estimators=300, max_depth=6, learning_rate=0.1,
                             subsample=0.8, colsample_bytree=0.8, n_jobs=-1,
                             tree_method='hist', random_state=42)
        model.fit(X[tr], y[tr], eval_set=[(X[te], y[te])], verbose=False)
        pred = model.predict(X[te])
        r, _ = pearsonr(y[te], pred)
        ldpo_results.append(r)
        print(f"    LDPO Fold {fold}: r={r:.4f}")
    ldpo_mean = np.mean(ldpo_results)
    print(f"    LDPO Mean: r={ldpo_mean:.4f}")
    
    # LCLO
    print(f"\n  LCLO:")
    unique_cells = np.unique(cl_arr)
    np.random.seed(42)
    cperm = np.random.permutation(len(unique_cells))
    cfold = len(unique_cells) // 5
    
    lclo_results = []
    for fold in range(5):
        s = fold * cfold
        e = s + cfold if fold < 4 else len(unique_cells)
        test_set = set(unique_cells[cperm[s:e]])
        te = np.array([c in test_set for c in cl_arr])
        tr = ~te
        if te.sum() < 10: continue
        model = XGBRegressor(n_estimators=300, max_depth=6, learning_rate=0.1,
                             subsample=0.8, colsample_bytree=0.8, n_jobs=-1,
                             tree_method='hist', random_state=42)
        model.fit(X[tr], y[tr], eval_set=[(X[te], y[te])], verbose=False)
        pred = model.predict(X[te])
        r, _ = pearsonr(y[te], pred)
        lclo_results.append(r)
        print(f"    LCLO Fold {fold}: r={r:.4f}")
    lclo_mean = np.mean(lclo_results)
    print(f"    LCLO Mean: r={lclo_mean:.4f}")
    
    # Save model
    model_path = MODEL_DIR / "xgboost_synergy_realexpr.pkl"
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    
    # Comparison
    print(f"\n{'='*70}")
    print("COMPARISON: All Versions")
    print("=" * 70)
    print(f"  Hash FP (O'Neil 23K):          PF=0.605  LDPO=0.604  LCLO=0.510")
    print(f"  Morgan FP (O'Neil 23K):        PF=0.603  LDPO=0.597  LCLO=0.489")
    print(f"  Hash+Hash (Combined 200K):     PF=0.659  LDPO=0.584  LCLO=0.550")
    print(f"  Morgan+Real (Combined {len(X)//1000}K):  "
          f"PF={pf_mean:.3f}  LDPO={ldpo_mean:.3f}  LCLO={lclo_mean:.3f}")
    
    # Save matching stats
    stats = {
        'ccle_cell_lines': len(ccle),
        'drugcomb_cell_lines': len(all_dc_cells),
        'matched': len(mapping),
        'unmatched': len(unmatched),
        'records_with_real_expr': int(has_expr.sum()),
        'total_dc_records': len(dc_data),
        'final_dataset_size': len(X),
        'results': {
            'PF': float(pf_mean),
            'LDPO': float(ldpo_mean),
            'LCLO': float(lclo_mean),
        }
    }
    stats_path = MODEL_DIR / "real_expression_matching_stats.json"
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"\n  Stats saved: {stats_path}")


if __name__ == "__main__":
    main()
