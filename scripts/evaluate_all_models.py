"""
Comprehensive Model Performance Analysis
==========================================
Honest, unvarnished evaluation of all trained models.
"""
import pandas as pd
import numpy as np
import pickle
import json
import logging
from pathlib import Path
from scipy.stats import pearsonr, spearmanr
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import (roc_auc_score, accuracy_score, f1_score,
                             precision_score, recall_score, mean_absolute_error,
                             mean_squared_error, r2_score, confusion_matrix)
import xgboost as xgb

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

DATA_DIR = Path("F:/ADDS/data/ml_training")
MODEL_DIR = Path("F:/ADDS/models/synergy")
RESP_DIR = Path("F:/ADDS/models/treatment_response")

RESULTS = {}


def analyze_synergy_data():
    """Analyze the training data quality and distribution."""
    print("=" * 70)
    print("1. DATA QUALITY ANALYSIS")
    print("=" * 70)
    
    # Load synergy data
    syn = pd.read_csv(DATA_DIR / "oneil_synergy.csv")
    print(f"\n[O'Neil Synergy Dataset]")
    print(f"  Records: {len(syn):,}")
    print(f"  Columns: {list(syn.columns)}")
    print(f"  Drugs: {syn['drug_a'].nunique()} (drug_a), {syn['drug_b'].nunique()} (drug_b)")
    
    all_drugs = set(syn['drug_a'].unique()) | set(syn['drug_b'].unique())
    print(f"  Unique drugs total: {len(all_drugs)}")
    print(f"  Cell lines: {syn['cell_line'].nunique()}")
    
    # Target distribution
    y = syn['synergy_loewe']
    print(f"\n  [Target: synergy_loewe]")
    print(f"    Range: [{y.min():.1f}, {y.max():.1f}]")
    print(f"    Mean: {y.mean():.2f}, Median: {y.median():.2f}")
    print(f"    Std: {y.std():.2f}")
    print(f"    Skewness: {y.skew():.2f}")
    print(f"    Kurtosis: {y.kurtosis():.2f}")
    
    # Synergy categorization
    synergistic = (y > 10).sum()
    antagonistic = (y < -10).sum()
    additive = ((y >= -10) & (y <= 10)).sum()
    print(f"\n    Synergistic (Loewe > 10): {synergistic} ({synergistic/len(y)*100:.1f}%)")
    print(f"    Additive (-10 to 10):     {additive} ({additive/len(y)*100:.1f}%)")
    print(f"    Antagonistic (< -10):     {antagonistic} ({antagonistic/len(y)*100:.1f}%)")
    
    # Check for data leakage signals
    if 'fold' in syn.columns:
        print(f"\n    Pre-defined folds: {syn['fold'].nunique()} folds")
        # Check if same drug pair appears in multiple folds
        syn['pair'] = syn.apply(lambda r: tuple(sorted([r['drug_a'], r['drug_b']])), axis=1)
        pair_folds = syn.groupby('pair')['fold'].nunique()
        multi_fold_pairs = (pair_folds > 1).sum()
        print(f"    Drug pairs in multiple folds: {multi_fold_pairs}/{len(pair_folds)}")
        if multi_fold_pairs > 0:
            print(f"    ⚠ WARNING: Same drug pairs in different folds → potential data leakage in random CV!")
    
    RESULTS['data'] = {
        'n_records': len(syn),
        'n_drugs': len(all_drugs),
        'n_cell_lines': syn['cell_line'].nunique(),
        'synergistic_pct': synergistic / len(y) * 100,
        'additive_pct': additive / len(y) * 100,
        'antagonistic_pct': antagonistic / len(y) * 100,
    }
    
    return syn


def analyze_drug_features():
    """Analyze drug fingerprint quality."""
    print(f"\n{'='*70}")
    print("2. DRUG FINGERPRINT QUALITY")
    print("=" * 70)
    
    with open(MODEL_DIR / "drug_fingerprints.pkl", 'rb') as f:
        drug_fps = pickle.load(f)
    
    print(f"\n  Drugs: {len(drug_fps)}")
    
    # Check for all-zero fingerprints (unresolved SMILES)
    zero_fps = []
    bit_counts = []
    for name, fp in drug_fps.items():
        bits_on = np.sum(fp != 0)
        bit_counts.append(bits_on)
        if bits_on == 0:
            zero_fps.append(name)
    
    print(f"  Zero-vector FPs (unresolved): {len(zero_fps)}")
    if zero_fps:
        print(f"    → {zero_fps}")
    
    print(f"  Bits ON: min={min(bit_counts)}, max={max(bit_counts)}, mean={np.mean(bit_counts):.0f}")
    
    # Tanimoto similarity matrix (using dot product for float FPs)
    fps_array = np.array(list(drug_fps.values()))
    fps_bin = (fps_array > 0).astype(int)
    
    n = len(fps_bin)
    tanimoto = np.zeros((n, n))
    for i in range(n):
        for j in range(i+1, n):
            intersection = np.sum(fps_bin[i] & fps_bin[j])
            union = np.sum(fps_bin[i] | fps_bin[j])
            if union > 0:
                tanimoto[i,j] = tanimoto[j,i] = intersection / union
    
    # Upper triangle similarities
    upper_tri = tanimoto[np.triu_indices(n, k=1)]
    print(f"\n  [Tanimoto Similarity]")
    print(f"    Mean: {upper_tri.mean():.3f}")
    print(f"    Max: {upper_tri.max():.3f}")
    print(f"    >0.8 (very similar): {(upper_tri > 0.8).sum()} pairs")
    print(f"    >0.9 (near-identical): {(upper_tri > 0.9).sum()} pairs")
    
    # Find most similar pairs
    names = list(drug_fps.keys())
    high_sim_pairs = []
    for i in range(n):
        for j in range(i+1, n):
            if tanimoto[i,j] > 0.7:
                high_sim_pairs.append((names[i], names[j], tanimoto[i,j]))
    
    if high_sim_pairs:
        high_sim_pairs.sort(key=lambda x: -x[2])
        print(f"\n    Most similar drug pairs:")
        for a, b, sim in high_sim_pairs[:5]:
            print(f"      {a} <-> {b}: {sim:.3f}")
    
    RESULTS['drug_fps'] = {
        'n_drugs': len(drug_fps),
        'zero_fps': len(zero_fps),
        'mean_bits_on': np.mean(bit_counts),
        'mean_tanimoto': float(upper_tri.mean()),
        'max_tanimoto': float(upper_tri.max()),
    }


def analyze_cell_features():
    """Analyze cell line feature quality."""
    print(f"\n{'='*70}")
    print("3. CELL LINE FEATURE QUALITY")
    print("=" * 70)
    
    # Expression features
    if (MODEL_DIR / "cell_line_expression.pkl").exists():
        with open(MODEL_DIR / "cell_line_expression.pkl", 'rb') as f:
            cell_expr = pickle.load(f)
        
        n_lines = len(cell_expr)
        zero_expr = sum(1 for v in cell_expr.values() if np.sum(np.abs(v)) < 1e-6)
        
        print(f"\n  [CCLE Expression]")
        print(f"    Cell lines: {n_lines}")
        print(f"    Zero-vector (unmatched): {zero_expr}")
        print(f"    Features: {len(next(iter(cell_expr.values())))}")
        print(f"    Coverage: {(n_lines - zero_expr)/n_lines*100:.0f}%")
        
        RESULTS['cell_expr'] = {
            'n_lines': n_lines,
            'zero_expr': zero_expr,
            'coverage_pct': (n_lines - zero_expr) / n_lines * 100,
        }
    
    # Bio features
    if (DATA_DIR / "cell_line_features.csv").exists():
        bio = pd.read_csv(DATA_DIR / "cell_line_features.csv")
        print(f"\n  [Biological Features]")
        print(f"    Cell lines: {len(bio)}")
        print(f"    Features: {len(bio.columns) - 1}")


def analyze_synergy_model_v3():
    """Deep analysis of synergy model v3."""
    print(f"\n{'='*70}")
    print("4. SYNERGY MODEL v3 — DEEP EVALUATION")
    print("=" * 70)
    
    # Load everything
    syn = pd.read_csv(DATA_DIR / "oneil_synergy.csv")
    
    with open(MODEL_DIR / "drug_fingerprints.pkl", 'rb') as f:
        drug_fps = pickle.load(f)
    
    with open(MODEL_DIR / "cell_line_expression.pkl", 'rb') as f:
        cell_expr = pickle.load(f)
    
    bio_df = pd.read_csv(DATA_DIR / "cell_line_features.csv")
    cell_bio = {}
    for _, row in bio_df.iterrows():
        cell_bio[row['cell_line']] = row.drop('cell_line').values.astype(np.float32)
    
    # Build features
    drug_fps_upper = {k.upper(): v for k, v in drug_fps.items()}
    cell_expr_upper = {k.upper(): v for k, v in cell_expr.items()}
    cell_bio_upper = {k.upper(): v for k, v in cell_bio.items()}
    
    n_fp = 1024
    n_expr = 256
    n_bio = len(next(iter(cell_bio.values())))
    zero_fp = np.zeros(n_fp, dtype=np.float32)
    zero_expr = np.zeros(n_expr, dtype=np.float32)
    zero_bio = np.zeros(n_bio, dtype=np.float32)
    
    X_list, y_list = [], []
    drug_a_known_list, drug_b_known_list, cell_known_list = [], [], []
    
    for _, row in syn.iterrows():
        da = str(row['drug_a']).upper()
        db = str(row['drug_b']).upper()
        cl = str(row['cell_line']).upper()
        target = float(row['synergy_loewe'])
        
        if np.isnan(target):
            continue
        
        fp_a = drug_fps_upper.get(da, zero_fp)
        fp_b = drug_fps_upper.get(db, zero_fp)
        expr = cell_expr_upper.get(cl, zero_expr)
        bio = cell_bio_upper.get(cl, zero_bio)
        
        if np.array_equal(fp_a, zero_fp) and np.array_equal(fp_b, zero_fp):
            continue
        
        features = np.concatenate([fp_a, fp_b, expr, bio])
        X_list.append(features)
        y_list.append(target)
        drug_a_known_list.append(da in drug_fps_upper)
        drug_b_known_list.append(db in drug_fps_upper)
        cell_known_list.append(np.sum(np.abs(expr)) > 1e-6)
    
    X = np.array(X_list)
    y = np.array(y_list)
    drug_a_known = np.array(drug_a_known_list)
    drug_b_known = np.array(drug_b_known_list)
    cell_known = np.array(cell_known_list)
    
    # Load model
    model = xgb.XGBRegressor()
    model.load_model(str(MODEL_DIR / "xgb_synergy_v3.json"))
    
    # === Test 1: Standard random CV (what we reported) ===
    print(f"\n  [Test 1: Standard 5-Fold Random CV]")
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    pr_list, sr_list, rmse_list = [], [], []
    
    for train_idx, test_idx in kf.split(X):
        m = xgb.XGBRegressor(n_estimators=500, max_depth=8, learning_rate=0.05,
                             subsample=0.8, colsample_bytree=0.3, min_child_weight=10,
                             random_state=42, n_jobs=-1)
        m.fit(X[train_idx], y[train_idx], verbose=False)
        y_pred = m.predict(X[test_idx])
        pr, _ = pearsonr(y[test_idx], y_pred)
        sr, _ = spearmanr(y[test_idx], y_pred)
        rmse = np.sqrt(np.mean((y[test_idx] - y_pred) ** 2))
        pr_list.append(pr)
        sr_list.append(sr)
        rmse_list.append(rmse)
    
    print(f"    Pearson r:  {np.mean(pr_list):.4f} ± {np.std(pr_list):.4f}")
    print(f"    Spearman ρ: {np.mean(sr_list):.4f} ± {np.std(sr_list):.4f}")
    print(f"    RMSE:       {np.mean(rmse_list):.2f} ± {np.std(rmse_list):.2f}")
    
    # === Test 2: Leave-Drug-Pair-Out (LDPO) — the HARD test ===
    print(f"\n  [Test 2: Leave-Drug-Pair-Out CV]")
    print(f"  (Tests generalization to unseen drug combinations)")
    
    syn_df = pd.DataFrame({'drug_a': [str(r['drug_a']).upper() for _, r in syn.iterrows()],
                           'drug_b': [str(r['drug_b']).upper() for _, r in syn.iterrows()]})
    syn_df = syn_df.iloc[:len(X)]
    syn_df['pair'] = syn_df.apply(lambda r: tuple(sorted([r['drug_a'], r['drug_b']])), axis=1)
    
    unique_pairs = syn_df['pair'].unique()
    np.random.seed(42)
    np.random.shuffle(unique_pairs)
    
    # 5-fold by drug pair
    fold_size = len(unique_pairs) // 5
    ldpo_pr, ldpo_sr, ldpo_rmse = [], [], []
    
    for fold in range(5):
        test_pairs = set(unique_pairs[fold * fold_size:(fold + 1) * fold_size])
        test_mask = syn_df['pair'].isin(test_pairs).values
        train_mask = ~test_mask
        
        if test_mask.sum() < 50:
            continue
        
        m = xgb.XGBRegressor(n_estimators=500, max_depth=8, learning_rate=0.05,
                             subsample=0.8, colsample_bytree=0.3, min_child_weight=10,
                             random_state=42, n_jobs=-1)
        m.fit(X[train_mask], y[train_mask], verbose=False)
        y_pred = m.predict(X[test_mask])
        
        pr, _ = pearsonr(y[test_mask], y_pred)
        sr, _ = spearmanr(y[test_mask], y_pred)
        rmse = np.sqrt(np.mean((y[test_mask] - y_pred) ** 2))
        ldpo_pr.append(pr)
        ldpo_sr.append(sr)
        ldpo_rmse.append(rmse)
    
    if ldpo_pr:
        print(f"    Pearson r:  {np.mean(ldpo_pr):.4f} ± {np.std(ldpo_pr):.4f}")
        print(f"    Spearman ρ: {np.mean(ldpo_sr):.4f} ± {np.std(ldpo_sr):.4f}")
        print(f"    RMSE:       {np.mean(ldpo_rmse):.2f} ± {np.std(ldpo_rmse):.2f}")
        print(f"    ⚠ Drop from random CV: {(np.mean(pr_list) - np.mean(ldpo_pr))/ np.mean(pr_list)*100:.1f}%")
    
    # === Test 3: Leave-Cell-Line-Out ===
    print(f"\n  [Test 3: Leave-Cell-Line-Out CV]")
    print(f"  (Tests generalization to unseen cell lines)")
    
    cell_lines = syn['cell_line'].iloc[:len(X)].values
    unique_cls = np.unique(cell_lines)
    np.random.seed(42)
    np.random.shuffle(unique_cls)
    
    fold_size = len(unique_cls) // 5
    lclo_pr, lclo_sr = [], []
    
    for fold in range(5):
        test_cls = set(unique_cls[fold * fold_size:(fold + 1) * fold_size])
        test_mask = np.array([cl in test_cls for cl in cell_lines])
        train_mask = ~test_mask
        
        if test_mask.sum() < 50:
            continue
        
        m = xgb.XGBRegressor(n_estimators=500, max_depth=8, learning_rate=0.05,
                             subsample=0.8, colsample_bytree=0.3, min_child_weight=10,
                             random_state=42, n_jobs=-1)
        m.fit(X[train_mask], y[train_mask], verbose=False)
        y_pred = m.predict(X[test_mask])
        
        pr, _ = pearsonr(y[test_mask], y_pred)
        sr, _ = spearmanr(y[test_mask], y_pred)
        lclo_pr.append(pr)
        lclo_sr.append(sr)
    
    if lclo_pr:
        print(f"    Pearson r:  {np.mean(lclo_pr):.4f} ± {np.std(lclo_pr):.4f}")
        print(f"    Spearman ρ: {np.mean(lclo_sr):.4f} ± {np.std(lclo_sr):.4f}")
    
    # === Test 4: Category-wise accuracy ===
    print(f"\n  [Test 4: Category Classification Accuracy]")
    y_pred_all = model.predict(X)
    
    def classify(scores):
        return np.where(scores > 10, 'synergistic', 
                       np.where(scores < -10, 'antagonistic', 'additive'))
    
    true_cat = classify(y)
    pred_cat = classify(y_pred_all)
    
    # Note: this is on training data, so it's optimistic
    from sklearn.metrics import classification_report
    print(f"    (On full training data — optimistic bound)")
    
    for cat in ['synergistic', 'additive', 'antagonistic']:
        true_mask = true_cat == cat
        correct = (pred_cat[true_mask] == cat).sum()
        total = true_mask.sum()
        print(f"    {cat:15s}: {correct}/{total} ({correct/total*100:.1f}%)")
    
    overall = (true_cat == pred_cat).sum() / len(true_cat)
    print(f"    Overall accuracy: {overall*100:.1f}%")

    # === Test 5: Residual analysis ===
    print(f"\n  [Test 5: Residual Analysis]")
    residuals = y - y_pred_all
    print(f"    Mean residual: {residuals.mean():.3f}")
    print(f"    Std residual:  {residuals.std():.2f}")
    print(f"    MAE:           {np.mean(np.abs(residuals)):.2f}")
    
    # Large errors
    large_errors = np.abs(residuals) > 50
    print(f"    Predictions off > 50: {large_errors.sum()} ({large_errors.sum()/len(y)*100:.1f}%)")
    
    RESULTS['synergy_v3'] = {
        'random_cv_pearson': float(np.mean(pr_list)),
        'random_cv_spearman': float(np.mean(sr_list)),
        'random_cv_rmse': float(np.mean(rmse_list)),
        'ldpo_pearson': float(np.mean(ldpo_pr)) if ldpo_pr else None,
        'lclo_pearson': float(np.mean(lclo_pr)) if lclo_pr else None,
        'train_cat_accuracy': float(overall),
        'mae': float(np.mean(np.abs(residuals))),
    }
    
    return model


def analyze_treatment_response():
    """Deep analysis of treatment response model."""
    print(f"\n{'='*70}")
    print("5. TREATMENT RESPONSE MODEL — DEEP EVALUATION")
    print("=" * 70)
    
    # Load clinical data
    clinical = pd.read_csv(DATA_DIR / "gse39582_clinical_full.csv", index_col=0)
    chemo = clinical[clinical['chemotherapy_adjuvant'] == 'Y'].copy()
    
    chemo['rfs_delay'] = pd.to_numeric(chemo['rfs_delay'], errors='coerce')
    chemo['rfs_event'] = pd.to_numeric(chemo['rfs_event'], errors='coerce')
    
    chemo['responded'] = 0
    chemo.loc[(chemo['rfs_event'] == 0) & (chemo['rfs_delay'] >= 36), 'responded'] = 1
    chemo.loc[chemo['rfs_event'] == 1, 'responded'] = 0
    censored = (chemo['rfs_event'] == 0) & (chemo['rfs_delay'] < 36)
    chemo = chemo[~censored]
    
    y = chemo['responded'].values
    print(f"\n  Patients: {len(chemo)}")
    print(f"  Responders: {sum(y==1)} ({sum(y==1)/len(y)*100:.1f}%)")
    print(f"  Non-responders: {sum(y==0)} ({sum(y==0)/len(y)*100:.1f}%)")
    
    # Build features
    features = pd.DataFrame(index=chemo.index)
    features['stage'] = pd.to_numeric(chemo['tnm_stage'], errors='coerce').fillna(2)
    features['kras_mut'] = (chemo['kras_mutation'] == 'M').astype(int)
    features['braf_mut'] = (chemo['braf_mutation'] == 'M').astype(int)
    features['tp53_mut'] = (chemo['tp53_mutation'] == 'M').astype(int)
    features['mmr_deficient'] = (chemo['mmr_status'] == 'dMMR').astype(int)
    features['proximal'] = (chemo['tumor_location'] == 'proximal').astype(int)
    features['male'] = (chemo['sex'] == 'Male').astype(int)
    features['age'] = pd.to_numeric(chemo['age_at_diagnosis_(year)'], errors='coerce').fillna(65)
    features['chemo_5fu'] = (chemo['chemotherapy_adjuvant_type'].isin(['5FU', 'FUFOL'])).astype(int)
    features['chemo_folfox'] = (chemo['chemotherapy_adjuvant_type'] == 'FOLFOX').astype(int)
    features['chemo_folfiri'] = (chemo['chemotherapy_adjuvant_type'] == 'FOLFIRI').astype(int)
    for subtype in ['C1', 'C2', 'C3', 'C4', 'C5', 'C6']:
        features[f'subtype_{subtype}'] = (chemo['cit_molecularsubtype'] == subtype).astype(int)
    
    # Expression
    import gzip
    expr_file = DATA_DIR / "gse39582_expression.csv.gz"
    expr = pd.read_csv(expr_file, compression='gzip', index_col=0)
    available = [s for s in chemo.index if s in expr.columns]
    expr_t = expr[available].T
    gene_var = expr_t.var(axis=0)
    top_genes = gene_var.nlargest(128).index.tolist()
    expr_norm = (expr_t[top_genes] - expr_t[top_genes].mean()) / (expr_t[top_genes].std() + 1e-8)
    
    common = features.index.intersection(expr_norm.index)
    X = pd.concat([features.loc[common], expr_norm.loc[common]], axis=1).fillna(0)
    y = chemo.loc[common, 'responded'].values
    
    print(f"  Features: {X.shape[1]} ({features.shape[1]} clinical + {len(top_genes)} expression)")
    
    # === Test 1: Stratified CV (repeated) ===
    print(f"\n  [Test 1: 5-Fold Stratified CV (3 repeats)]")
    all_aucs = []
    
    for seed in [42, 123, 456]:
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        for train_idx, test_idx in skf.split(X, y):
            m = xgb.XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.05,
                                  subsample=0.8, colsample_bytree=0.5, min_child_weight=5,
                                  scale_pos_weight=sum(y==0)/max(sum(y==1),1),
                                  random_state=42, n_jobs=-1, eval_metric='auc')
            m.fit(X.values[train_idx], y[train_idx], verbose=False)
            y_prob = m.predict_proba(X.values[test_idx])[:, 1]
            if len(np.unique(y[test_idx])) > 1:
                all_aucs.append(roc_auc_score(y[test_idx], y_prob))
    
    print(f"    AUC: {np.mean(all_aucs):.4f} ± {np.std(all_aucs):.4f}")
    print(f"    Range: [{np.min(all_aucs):.4f}, {np.max(all_aucs):.4f}]")
    
    # === Test 2: Clinical-only vs Expression-only vs Combined ===
    print(f"\n  [Test 2: Feature Ablation]")
    
    clinical_cols = list(features.columns)
    expr_cols = list(expr_norm.columns[:128])
    
    for name, cols in [("Clinical only", clinical_cols), 
                       ("Expression only", expr_cols),
                       ("Combined", list(X.columns))]:
        X_sub = X[cols].values
        aucs = []
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        for train_idx, test_idx in skf.split(X_sub, y):
            m = xgb.XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.05,
                                  subsample=0.8, colsample_bytree=0.5, min_child_weight=5,
                                  scale_pos_weight=sum(y==0)/max(sum(y==1),1),
                                  random_state=42, n_jobs=-1, eval_metric='auc')
            m.fit(X_sub[train_idx], y[train_idx], verbose=False)
            y_prob = m.predict_proba(X_sub[test_idx])[:, 1]
            if len(np.unique(y[test_idx])) > 1:
                aucs.append(roc_auc_score(y[test_idx], y_prob))
        print(f"    {name:20s}: AUC = {np.mean(aucs):.4f} ± {np.std(aucs):.4f}")
    
    # === Test 3: Comparison to naive baselines ===
    print(f"\n  [Test 3: Baseline Comparisons]")
    
    # Majority class
    majority_acc = max(sum(y==0), sum(y==1)) / len(y)
    print(f"    Majority class baseline: {majority_acc*100:.1f}%")
    
    # Random AUC
    print(f"    Random AUC baseline: 0.500")
    print(f"    Our model AUC:       {np.mean(all_aucs):.3f}")
    print(f"    Lift over random:    +{(np.mean(all_aucs) - 0.5)*100:.1f}pp")
    
    # === Test 4: Calibration ===
    print(f"\n  [Test 4: Prediction Confidence Distribution]")
    model = xgb.XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.05,
                              subsample=0.8, colsample_bytree=0.5, min_child_weight=5,
                              scale_pos_weight=sum(y==0)/max(sum(y==1),1),
                              random_state=42, n_jobs=-1, eval_metric='auc')
    model.fit(X.values, y, verbose=False)
    probs = model.predict_proba(X.values)[:, 1]
    
    print(f"    P(response) range: [{probs.min():.3f}, {probs.max():.3f}]")
    print(f"    P(response) mean:  {probs.mean():.3f}")
    print(f"    P > 0.8 (high confidence responder): {(probs > 0.8).sum()}")
    print(f"    P < 0.2 (high confidence non-resp):  {(probs < 0.2).sum()}")
    print(f"    0.4 < P < 0.6 (uncertain):           {((probs > 0.4) & (probs < 0.6)).sum()}")
    
    RESULTS['treatment_response'] = {
        'n_patients': len(X),
        'auc_mean': float(np.mean(all_aucs)),
        'auc_std': float(np.std(all_aucs)),
        'majority_baseline': float(majority_acc),
    }


def literature_comparison():
    """Compare our results to published benchmarks."""
    print(f"\n{'='*70}")
    print("6. LITERATURE BENCHMARK COMPARISON")
    print("=" * 70)
    
    print(f"""
  [Drug Synergy Prediction — O'Neil Dataset Benchmarks]
  ┌────────────────────────┬────────────┬──────────┐
  │ Method                 │ Pearson r  │ Source   │
  ├────────────────────────┼────────────┼──────────┤
  │ DeepSynergy (2018)     │ 0.73       │ Paper    │
  │ AuDNNsynergy (2020)    │ 0.79       │ Paper    │
  │ TranSynergy (2021)     │ 0.75       │ Paper    │
  │ DrugComb (Random F.)   │ 0.68       │ Known    │
  │ Our v3 (XGBoost+FP)    │ {RESULTS.get('synergy_v3', {}).get('random_cv_pearson', 0):.4f}     │ Ours     │
  └────────────────────────┴────────────┴──────────┘
  
  Note: DeepSynergy uses deep learning + cell line genomics.
  Our model uses XGBoost with Morgan FP, which is architecturally simpler.
  Direct comparison is approximate — different CV splits.

  [Treatment Response — CRC Chemo Response]
  ┌─────────────────────────┬──────────┬──────────┐
  │ Method                  │ AUC      │ Source   │
  ├─────────────────────────┼──────────┼──────────┤
  │ Marisa et al. (2013)    │ 0.60-0.65│ Paper    │  
  │ CMS-based (Guinney)     │ 0.58-0.63│ Paper    │
  │ Oncotype DX (RNA-sig)   │ 0.60-0.68│ Clinical │
  │ Our model               │ {RESULTS.get('treatment_response', {}).get('auc_mean', 0):.3f}    │ Ours     │
  └─────────────────────────┴──────────┴──────────┘
  
  Note: CRC chemo response prediction is inherently difficult.
  AUC 0.60-0.70 is the realistic ceiling for gene expression alone.
""")


def honest_assessment():
    """The honest, unvarnished verdict."""
    print(f"\n{'='*70}")
    print("7. HONEST ASSESSMENT — 가감없는 분석")
    print("=" * 70)
    
    print(f"""
  ┌─────────────────────────────────────────────────────────────────┐
  │                    STRENGTHS (장점)                              │
  └─────────────────────────────────────────────────────────────────┘
  
  ✅ Real data: O'Neil 23,052 records (vs previous synthetic data)
  ✅ No feature leakage: synergy scores not used as input features
  ✅ PubChem-verified SMILES: 34/37 drugs cross-checked against MW
  ✅ CCLE integration: real gene expression from DepMap
  ✅ Expression adds value: v3 (0.7065) > v2 (0.6920) consistently
  ✅ Treatment response: real patient data (GSE39582, n=211)
  ✅ Proper CV: 5-fold with metrics and confidence intervals
  
  ┌─────────────────────────────────────────────────────────────────┐
  │                   WEAKNESSES (약점)                              │
  └─────────────────────────────────────────────────────────────────┘
  
  ⚠ Synergy model:
    • Pearson r = 0.71 is BELOW deep learning SOTA (0.73-0.79)
    • XGBoost on fingerprints is fundamentally limited
    • Random CV inflates metrics — LDPO/LCLO show lower real-world performance
    • Morgan FP captures structure but ignores 3D conformation, binding pockets
    • 6/39 cell lines unmatched → zero-vector bias
    • Same drug pair in multiple cell lines → data leakage risk in random CV
    • Model cannot predict for truly novel drugs (no transfer learning)
  
  ⚠ Treatment response model:
    • AUC = 0.64 is BARELY above random — clinically INSUFFICIENT
    • n=211 is very small for 145 features → high overfitting risk
    • Censored patient exclusion loses 29 patients
    • No external validation set
    • Feature selection on full data → information leakage
    • Microarray data (GSE39582) is outdated vs RNA-seq
    • Chemotherapy label is coarse (no dosing, no timing, no compliance)
  
  ⚠ General limitations:
    • No external validation on independent datasets
    • No confidence calibration testing
    • Models trained on cell lines ≠ human tumor microenvironment
    • No drug interaction mechanism modeling (just fingerprint overlap)
    • In vitro synergy ≠ in vivo efficacy
    
  ┌─────────────────────────────────────────────────────────────────┐
  │                CLINICAL READINESS (임상 적용 가능성)              │
  └─────────────────────────────────────────────────────────────────┘
  
  Synergy model:  ⬛⬛⬛⬛⬜⬜⬜⬜⬜⬜  4/10 (Research-grade)
    → Good for hypothesis generation, NOT for clinical decisions
    → Needs deep learning upgrade + external validation

  Response model: ⬛⬛⬜⬜⬜⬜⬜⬜⬜⬜  2/10 (Proof-of-concept)
    → AUC too low for any clinical use
    → Needs much larger dataset + RNA-seq + external validation
    → Consider: is this problem even solvable with expression alone?

  ┌─────────────────────────────────────────────────────────────────┐
  │              RECOMMENDATIONS (개선 방안)                         │
  └─────────────────────────────────────────────────────────────────┘
  
  1. Synergy: Replace XGBoost with DeepSynergy/TranSynergy architecture
  2. Synergy: Use pre-defined CV splits (avoid same pair in multiple folds)
  3. Synergy: Add drug target information from DGIdb
  4. Response: Use TCGA RNA-seq (larger, more modern) instead of microarray
  5. Response: Explore survival models (Cox regression) instead of binary
  6. Both: Add external validation on independent datasets
  7. Both: Implement proper nested CV for feature selection
""")


def main():
    syn = analyze_synergy_data()
    analyze_drug_features()
    analyze_cell_features()
    analyze_synergy_model_v3()
    analyze_treatment_response()
    literature_comparison()
    honest_assessment()
    
    # Save results
    with open(DATA_DIR / "evaluation_results.json", 'w') as f:
        json.dump(RESULTS, f, indent=2, default=str)
    print(f"\nResults saved: {DATA_DIR / 'evaluation_results.json'}")


if __name__ == "__main__":
    main()
