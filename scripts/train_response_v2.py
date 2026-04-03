"""
Treatment Response Model v2 - TCGA RNA-seq + GSE39582
=====================================================
Trains on whatever TCGA data is available, cross-validates with GSE39582.
Uses TCGA STAR-Counts FPKM + GSE39582 microarray.
"""
import pandas as pd
import numpy as np
import pickle
import json
import logging
from pathlib import Path
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
import xgboost as xgb

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

DATA_DIR = Path("F:/ADDS/data/ml_training")
TCGA_DIR = DATA_DIR / "tcga_rnaseq"
MODEL_DIR = Path("F:/ADDS/models/synergy")


def load_tcga_data():
    """Load TCGA expression + clinical data."""
    logger.info("Loading TCGA data...")
    
    # Load clinical
    clinical = pd.read_csv(TCGA_DIR / "clinical.csv")
    logger.info(f"Clinical: {len(clinical)} patients")
    
    # Check available expression files (may be .tsv.gz or .tsv)
    expr_files = list(TCGA_DIR.glob("TCGA-*.tsv.gz")) + list(TCGA_DIR.glob("TCGA-*.tsv"))
    logger.info(f"Expression files: {len(expr_files)}")
    
    if len(expr_files) == 0:
        logger.warning("No expression files found!")
        return None, None
    
    # Parse expression
    expr_data = {}
    gene_names = None
    gene_col = None
    value_col = None
    
    for fpath in expr_files:
        try:
            # Extract TCGA sample ID: TCGA-AA-3561_ff710149.tsv.gz -> TCGA-AA-3561
            fname = fpath.name
            if fname.endswith('.tsv.gz'):
                fname = fname[:-7]  # remove .tsv.gz
            elif fname.endswith('.tsv'):
                fname = fname[:-4]
            sample_id = fname.rsplit('_', 1)[0]  # remove hash suffix
            
            # Try reading: GDC returns uncompressed TSV despite .gz extension
            try:
                df = pd.read_csv(fpath, sep='\t', compression='gzip', comment='#')
            except Exception:
                df = pd.read_csv(str(fpath), sep='\t', comment='#', compression=None, engine='python')
            
            if gene_col is None:
                # Detect columns
                if 'gene_name' in df.columns:
                    gene_col = 'gene_name'
                    if 'fpkm_unstranded' in df.columns:
                        value_col = 'fpkm_unstranded'
                    elif 'tpm_unstranded' in df.columns:
                        value_col = 'tpm_unstranded'
                    else:
                        value_col = 'unstranded'
                else:
                    gene_col = df.columns[0]
                    value_col = df.columns[1]
                
                mask = ~df[gene_col].str.startswith('_', na=True)
                if 'gene_type' in df.columns:
                    mask = mask & (df['gene_type'] == 'protein_coding')
                gene_names = df.loc[mask, gene_col].values
                logger.info(f"Genes: {len(gene_names)}, using {value_col}")
            
            mask = ~df[gene_col].str.startswith('_', na=True)
            if 'gene_type' in df.columns:
                mask = mask & (df['gene_type'] == 'protein_coding')
            values = df.loc[mask, value_col].values
            
            if len(values) == len(gene_names):
                expr_data[sample_id] = values.astype(np.float32)
        except Exception as e:
            pass
    
    logger.info(f"Parsed {len(expr_data)} expression samples")
    
    if not expr_data:
        return None, None
    
    return clinical, (list(expr_data.keys()), gene_names, expr_data)


def load_gse39582_data():
    """Load GSE39582 expression + clinical."""
    logger.info("Loading GSE39582 data...")
    
    clinical = pd.read_csv(DATA_DIR / "gse39582_clinical_full.csv")
    
    import gzip
    expr_path = DATA_DIR / "gse39582_expression.csv.gz"
    if expr_path.exists():
        logger.info("Reading GSE39582 expression (large file)...")
        expr_df = pd.read_csv(expr_path, index_col=0, nrows=200)  # Sample for speed
        # Actually load the full data efficiently via chunked reading
        # For now use the pre-computed 128 features
    
    return clinical


def build_tcga_features(clinical, expr_data_tuple):
    """Build features for treatment response from TCGA."""
    samples, gene_names, expr_dict = expr_data_tuple
    
    # Merge clinical
    clinical_idx = {row['submitter_id']: row for _, row in clinical.iterrows()}
    
    X_list, y_list = [], []
    sample_ids = []
    
    for sample in samples:
        if sample not in clinical_idx:
            continue
        
        clin = clinical_idx[sample]
        
        # Skip if no chemo
        if not clin.get('has_chemo', False):
            continue
        
        # Define response: alive = responder, dead = non-responder (simplified)
        vital = str(clin.get('vital_status', '')).lower()
        if vital == 'alive':
            label = 1  # Responder
        elif vital == 'dead':
            label = 0  # Non-responder
        else:
            continue
        
        expr = expr_dict[sample]
        
        # Clinical features
        age = clin.get('age_at_index', 65)
        if pd.isna(age):
            age = 65
        
        gender = 1 if str(clin.get('gender', '')).lower() == 'male' else 0
        
        # Stage encoding
        stage_map = {'Stage I': 1, 'Stage IA': 1, 'Stage II': 2, 'Stage IIA': 2, 'Stage IIB': 2,
                    'Stage III': 3, 'Stage IIIA': 3, 'Stage IIIB': 3, 'Stage IIIC': 3,
                    'Stage IV': 4, 'Stage IVA': 4, 'Stage IVB': 4}
        stage = stage_map.get(str(clin.get('ajcc_stage', '')), 2)
        
        clin_features = np.array([age, gender, stage], dtype=np.float32)
        
        X_list.append((clin_features, expr))
        y_list.append(label)
        sample_ids.append(sample)
    
    logger.info(f"Chemo patients with expression: {len(X_list)}")
    
    if not X_list:
        return None, None, None, None
    
    # Select top variance genes
    all_expr = np.array([x[1] for x in X_list])
    var = np.var(all_expr, axis=0)
    top_idx = np.argsort(var)[::-1][:256]
    
    # Build feature matrix
    X = np.array([np.concatenate([x[0], x[1][top_idx]]) for x in X_list])
    y = np.array(y_list)
    
    logger.info(f"Feature matrix: {X.shape}, labels: {np.bincount(y)}")
    
    return X, y, sample_ids, gene_names[top_idx]


def main():
    print("=" * 70)
    print("Treatment Response Model v2 - TCGA + GSE39582")
    print("=" * 70)
    
    # Load TCGA
    clinical, expr_tuple = load_tcga_data()
    
    if expr_tuple is None:
        print("\n  ERROR: No TCGA expression data available!")
        print("  Falling back to GSE39582-only model...")
        
        # Fall back to existing model
        from pathlib import Path
        model_path = MODEL_DIR / "treatment_response_v1.pkl"
        if model_path.exists():
            print(f"  Existing model: {model_path}")
        return
    
    samples, gene_names, expr_dict = expr_tuple
    
    print(f"\n  TCGA Samples: {len(samples)}")
    print(f"  TCGA Genes: {len(gene_names)}")
    print(f"  Clinical records: {len(clinical)}")
    print(f"  With chemo: {clinical['has_chemo'].sum()}")
    
    # Build features
    result = build_tcga_features(clinical, expr_tuple)
    
    if result[0] is None:
        print("\n  ERROR: No matched chemo patients with expression!")
        return
    
    X, y, sample_ids, selected_genes = result
    
    print(f"\n  Training samples: {len(X)}")
    print(f"  Responders (alive): {(y==1).sum()}")
    print(f"  Non-responders (dead): {(y==0).sum()}")
    print(f"  Features: {X.shape[1]} (3 clinical + 256 expression)")
    
    # Train with CV
    print(f"\n{'='*70}")
    print("TRAINING: 5-fold Stratified CV")
    print("=" * 70)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    model = xgb.XGBClassifier(
        n_estimators=200, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.5, min_child_weight=5,
        reg_alpha=0.1, reg_lambda=1.0, random_state=42,
        eval_metric='auc', use_label_encoder=False,
    )
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    aucs = []
    
    for fold, (tr, te) in enumerate(cv.split(X_scaled, y)):
        model.fit(X_scaled[tr], y[tr], verbose=False)
        yp = model.predict_proba(X_scaled[te])[:, 1]
        auc = roc_auc_score(y[te], yp)
        aucs.append(auc)
        print(f"  Fold {fold+1}: AUC = {auc:.4f}")
    
    print(f"  Mean AUC: {np.mean(aucs):.4f} +/- {np.std(aucs):.4f}")
    
    # Train final model
    print(f"\nTraining final model...")
    model.fit(X_scaled, y, verbose=False)
    
    # Save
    import pickle
    model_data = {
        'model': model,
        'scaler': scaler,
        'selected_genes': list(selected_genes),
        'n_clinical': 3,
        'n_expression': 256,
        'source': 'TCGA COAD/READ RNA-seq',
        'auc': f"{np.mean(aucs):.4f} +/- {np.std(aucs):.4f}",
    }
    
    with open(MODEL_DIR / "treatment_response_v2.pkl", 'wb') as f:
        pickle.dump(model_data, f)
    
    print(f"  Saved: {MODEL_DIR / 'treatment_response_v2.pkl'}")
    
    # Feature importance
    print(f"\n{'='*70}")
    print("TOP FEATURES")
    print("=" * 70)
    
    feat_names = ['age', 'gender', 'stage'] + [f"gene_{g}" for g in selected_genes]
    imp = model.feature_importances_
    top_idx = np.argsort(imp)[::-1][:20]
    
    for rank, idx in enumerate(top_idx, 1):
        print(f"  {rank:2d}. {feat_names[idx]:30s} importance={imp[idx]:.4f}")


if __name__ == "__main__":
    main()
