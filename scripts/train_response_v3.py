"""
Treatment Response v3 — FIXED
==============================
Fix 1: Use TCGA treatment_outcome (CR+PR=Responder vs PD=Non-responder)
Fix 2: Gene selection INSIDE each CV fold (no data leakage)
"""
import pandas as pd
import numpy as np
import pickle
import json
import logging
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, classification_report
import xgboost as xgb

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

DATA_DIR = Path("F:/ADDS/data/ml_training")
TCGA_DIR = DATA_DIR / "tcga_rnaseq"
MODEL_DIR = Path("F:/ADDS/models/synergy")


def load_tcga_expression():
    """Load and parse all TCGA expression files."""
    logger.info("Loading TCGA expression...")
    
    expr_files = list(TCGA_DIR.glob("TCGA-*.tsv.gz"))
    logger.info(f"Expression files: {len(expr_files)}")
    
    expr_data = {}
    gene_names = None
    gene_col = value_col = None
    
    for fpath in expr_files:
        try:
            fname = fpath.name
            if fname.endswith('.tsv.gz'):
                fname = fname[:-7]
            sample_id = fname.rsplit('_', 1)[0]
            
            try:
                df = pd.read_csv(fpath, sep='\t', compression='gzip', comment='#')
            except Exception:
                df = pd.read_csv(str(fpath), sep='\t', comment='#', compression=None, engine='python')
            
            if gene_col is None:
                gene_col = 'gene_name'
                value_col = 'fpkm_unstranded'
                mask = ~df[gene_col].str.startswith('_', na=True)
                if 'gene_type' in df.columns:
                    mask = mask & (df['gene_type'] == 'protein_coding')
                gene_names = df.loc[mask, gene_col].values
                logger.info(f"Genes: {len(gene_names)}")
            
            mask = ~df[gene_col].str.startswith('_', na=True)
            if 'gene_type' in df.columns:
                mask = mask & (df['gene_type'] == 'protein_coding')
            values = df.loc[mask, value_col].values
            
            if len(values) == len(gene_names):
                expr_data[sample_id] = np.log2(values.astype(np.float32) + 1)
        except:
            pass
    
    logger.info(f"Parsed: {len(expr_data)} samples")
    return expr_data, gene_names


def build_labels_from_treatment_outcome(clinical):
    """
    Use TCGA treatment_outcome instead of alive/dead.
    Responder = Complete Response + Partial Response
    Non-responder = Progressive Disease + Stable Disease
    """
    from collections import Counter
    
    labels = {}
    outcome_counts = Counter()
    
    for _, row in clinical.iterrows():
        sid = row['submitter_id']
        outcomes_str = str(row.get('treatment_outcomes', ''))
        
        if not outcomes_str or outcomes_str == 'nan':
            continue
        
        outcomes = [o.strip() for o in outcomes_str.split('|') if o.strip()]
        
        for o in outcomes:
            outcome_counts[o] += 1
        
        # Classify: take the BEST response if multiple
        is_responder = any(o in ['Complete Response', 'Partial Response'] for o in outcomes)
        is_nonresponder = any(o in ['Progressive Disease'] for o in outcomes)
        is_stable = any(o in ['Stable Disease'] for o in outcomes)
        
        if is_responder:
            labels[sid] = 1  # Responder
        elif is_nonresponder:
            labels[sid] = 0  # Non-responder
        elif is_stable:
            labels[sid] = 0  # Stable Disease → conservative: non-responder
        # Skip: 'Treatment Ongoing', 'Unknown', 'Not Reported'
    
    logger.info(f"Treatment outcome distribution:")
    for outcome, count in outcome_counts.most_common():
        logger.info(f"  {outcome}: {count}")
    
    # Also build alive/dead labels for comparison
    alive_dead_labels = {}
    for _, row in clinical.iterrows():
        sid = row['submitter_id']
        vital = str(row.get('vital_status', '')).lower()
        has_chemo = row.get('has_chemo', False)
        if not has_chemo:
            continue
        if vital == 'alive':
            alive_dead_labels[sid] = 1
        elif vital == 'dead':
            alive_dead_labels[sid] = 0
    
    return labels, alive_dead_labels, outcome_counts


def evaluate_with_proper_cv(X_all_expr, y, sample_ids, gene_names, clinical, label_name=""):
    """
    Proper CV: gene selection INSIDE each fold.
    """
    n_genes_select = 256
    n_clinical_features = 3  # age, gender, stage
    
    # Build clinical features
    clinical_idx = {row['submitter_id']: row for _, row in clinical.iterrows()}
    
    stage_map = {
        'Stage I': 1, 'Stage IA': 1, 'Stage II': 2, 'Stage IIA': 2, 'Stage IIB': 2,
        'Stage IIC': 2, 'Stage III': 3, 'Stage IIIA': 3, 'Stage IIIB': 3, 'Stage IIIC': 3,
        'Stage IV': 4, 'Stage IVA': 4, 'Stage IVB': 4
    }
    
    clinical_features = []
    for sid in sample_ids:
        clin = clinical_idx.get(sid, {})
        age = clin.get('age_at_index', 65)
        if pd.isna(age):
            age = 65
        gender = 1 if str(clin.get('gender', '')).lower() == 'male' else 0
        stage = stage_map.get(str(clin.get('ajcc_stage', '')), 2)
        clinical_features.append([age, gender, stage])
    
    X_clin = np.array(clinical_features, dtype=np.float32)
    
    # CV with gene selection INSIDE each fold
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    aucs = []
    all_y_true, all_y_pred = [], []
    
    for fold, (tr_idx, te_idx) in enumerate(cv.split(X_all_expr, y)):
        X_expr_train = X_all_expr[tr_idx]
        X_expr_test = X_all_expr[te_idx]
        y_train, y_test = y[tr_idx], y[te_idx]
        
        # Gene selection on TRAIN ONLY
        gene_var = np.var(X_expr_train, axis=0)
        top_gene_idx = np.argsort(gene_var)[::-1][:n_genes_select]
        
        X_train = np.hstack([X_clin[tr_idx], X_expr_train[:, top_gene_idx]])
        X_test = np.hstack([X_clin[te_idx], X_expr_test[:, top_gene_idx]])
        
        # Scale
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        
        model = xgb.XGBClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.5, min_child_weight=5,
            reg_alpha=0.1, reg_lambda=1.0, random_state=42,
            scale_pos_weight=sum(y_train == 0) / max(sum(y_train == 1), 1),
        )
        
        model.fit(X_train, y_train, verbose=False)
        yp = model.predict_proba(X_test)[:, 1]
        
        try:
            auc = roc_auc_score(y_test, yp)
        except:
            auc = 0.5
        
        aucs.append(auc)
        all_y_true.extend(y_test.tolist())
        all_y_pred.extend(yp.tolist())
        
        print(f"    Fold {fold+1}: AUC={auc:.4f} (train: {len(y_train)}, test: {len(y_test)}, "
              f"pos_rate={y_test.mean():.2f})")
    
    mean_auc = np.mean(aucs)
    std_auc = np.std(aucs)
    print(f"    {label_name} Mean AUC: {mean_auc:.4f} +/- {std_auc:.4f}")
    
    return mean_auc, std_auc, aucs


def main():
    print("=" * 70)
    print("Treatment Response v3 — FIXED LABELS + FIXED CV")
    print("=" * 70)
    
    # Load clinical
    clinical = pd.read_csv(TCGA_DIR / "clinical.csv")
    print(f"\n  Clinical records: {len(clinical)}")
    
    # Build labels
    outcome_labels, alive_dead_labels, outcome_counts = build_labels_from_treatment_outcome(clinical)
    
    resp = sum(1 for v in outcome_labels.values() if v == 1)
    nonresp = sum(1 for v in outcome_labels.values() if v == 0)
    print(f"\n  Treatment Outcome Labels:")
    print(f"    Responder (CR+PR): {resp}")
    print(f"    Non-responder (PD+SD): {nonresp}")
    print(f"    Total: {resp + nonresp}")
    print(f"    Class ratio: {resp/max(nonresp,1):.1f}:1")
    
    ad_resp = sum(1 for v in alive_dead_labels.values() if v == 1)
    ad_nonresp = sum(1 for v in alive_dead_labels.values() if v == 0)
    print(f"\n  Alive/Dead Labels (old, for comparison):")
    print(f"    Alive: {ad_resp}")
    print(f"    Dead: {ad_nonresp}")
    print(f"    Class ratio: {ad_resp/max(ad_nonresp,1):.1f}:1")
    
    # Load expression
    expr_data, gene_names = load_tcga_expression()
    
    # ==========================================
    # TEST A: Treatment Outcome labels + proper CV
    # ==========================================
    print(f"\n{'='*70}")
    print("TEST A: Treatment Outcome Labels + CV-internal Gene Selection")
    print("=" * 70)
    
    # Match samples
    matched_a = [sid for sid in outcome_labels if sid in expr_data]
    print(f"  Matched samples: {len(matched_a)}")
    
    if len(matched_a) > 20:
        X_expr_a = np.array([expr_data[sid] for sid in matched_a])
        y_a = np.array([outcome_labels[sid] for sid in matched_a])
        
        print(f"  Responders: {(y_a==1).sum()}, Non-responders: {(y_a==0).sum()}")
        
        auc_a, std_a, folds_a = evaluate_with_proper_cv(
            X_expr_a, y_a, matched_a, gene_names, clinical,
            label_name="Treatment Outcome"
        )
    else:
        print("  Insufficient matched samples!")
        auc_a = 0
    
    # ==========================================
    # TEST B: Alive/Dead labels + proper CV (for comparison)
    # ==========================================
    print(f"\n{'='*70}")
    print("TEST B: Alive/Dead Labels + CV-internal Gene Selection")
    print("=" * 70)
    
    matched_b = [sid for sid in alive_dead_labels if sid in expr_data]
    print(f"  Matched samples: {len(matched_b)}")
    
    if len(matched_b) > 20:
        X_expr_b = np.array([expr_data[sid] for sid in matched_b])
        y_b = np.array([alive_dead_labels[sid] for sid in matched_b])
        
        print(f"  Alive: {(y_b==1).sum()}, Dead: {(y_b==0).sum()}")
        
        auc_b, std_b, folds_b = evaluate_with_proper_cv(
            X_expr_b, y_b, matched_b, gene_names, clinical,
            label_name="Alive/Dead"
        )
    else:
        print("  Insufficient matched samples!")
        auc_b = 0
    
    # ==========================================
    # TEST C: Alive/Dead + LEAKED CV (to show leakage effect)
    # ==========================================
    print(f"\n{'='*70}")
    print("TEST C: Alive/Dead + LEAKED Gene Selection (original v2 method)")
    print("=" * 70)
    
    if len(matched_b) > 20:
        # Select genes on ALL data (the leaky way)
        gene_var = np.var(X_expr_b, axis=0)
        top_gene_idx = np.argsort(gene_var)[::-1][:256]
        
        clinical_idx = {row['submitter_id']: row for _, row in clinical.iterrows()}
        stage_map = {
            'Stage I': 1, 'Stage IA': 1, 'Stage II': 2, 'Stage IIA': 2, 'Stage IIB': 2,
            'Stage IIC': 2, 'Stage III': 3, 'Stage IIIA': 3, 'Stage IIIB': 3, 'Stage IIIC': 3,
            'Stage IV': 4, 'Stage IVA': 4, 'Stage IVB': 4
        }
        
        X_clin = []
        for sid in matched_b:
            clin = clinical_idx.get(sid, {})
            age = clin.get('age_at_index', 65)
            if pd.isna(age): age = 65
            gender = 1 if str(clin.get('gender', '')).lower() == 'male' else 0
            stage = stage_map.get(str(clin.get('ajcc_stage', '')), 2)
            X_clin.append([age, gender, stage])
        X_clin = np.array(X_clin, dtype=np.float32)
        
        X_leaked = np.hstack([X_clin, X_expr_b[:, top_gene_idx]])
        scaler = StandardScaler()
        X_leaked_scaled = scaler.fit_transform(X_leaked)
        
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        aucs_c = []
        for fold, (tr_idx, te_idx) in enumerate(cv.split(X_leaked_scaled, y_b)):
            model = xgb.XGBClassifier(
                n_estimators=200, max_depth=4, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.5, min_child_weight=5,
                reg_alpha=0.1, reg_lambda=1.0, random_state=42,
            )
            model.fit(X_leaked_scaled[tr_idx], y_b[tr_idx], verbose=False)
            yp = model.predict_proba(X_leaked_scaled[te_idx])[:, 1]
            auc = roc_auc_score(y_b[te_idx], yp)
            aucs_c.append(auc)
            print(f"    Fold {fold+1}: AUC={auc:.4f}")
        
        print(f"    Leaked CV Mean AUC: {np.mean(aucs_c):.4f} +/- {np.std(aucs_c):.4f}")
    
    # ==========================================
    # SUMMARY
    # ==========================================
    print(f"\n{'='*70}")
    print("COMPARISON SUMMARY")
    print("=" * 70)
    if auc_a > 0:
        print(f"  A. Treatment Outcome + Proper CV:  AUC = {auc_a:.4f} +/- {std_a:.4f}")
    if auc_b > 0:
        print(f"  B. Alive/Dead + Proper CV:         AUC = {auc_b:.4f} +/- {std_b:.4f}")
    if len(matched_b) > 20:
        print(f"  C. Alive/Dead + Leaked CV:         AUC = {np.mean(aucs_c):.4f} +/- {np.std(aucs_c):.4f}")
    print(f"  D. Original v2 (reported):         AUC = 0.6839 +/- 0.0700")
    
    print(f"\n  Leakage effect: C - B = {np.mean(aucs_c) - auc_b:.4f}" if len(matched_b) > 20 else "")
    
    # Save best model
    if auc_a > 0 and len(matched_a) > 20:
        print(f"\n  Training final v3 model with treatment outcome labels...")
        
        gene_var = np.var(X_expr_a, axis=0)
        top_gene_idx = np.argsort(gene_var)[::-1][:256]
        
        X_clin_a = []
        for sid in matched_a:
            clin = clinical_idx.get(sid, {})
            age = clin.get('age_at_index', 65)
            if pd.isna(age): age = 65
            gender = 1 if str(clin.get('gender', '')).lower() == 'male' else 0
            stage = stage_map.get(str(clin.get('ajcc_stage', '')), 2)
            X_clin_a.append([age, gender, stage])
        X_clin_a = np.array(X_clin_a, dtype=np.float32)
        
        X_final = np.hstack([X_clin_a, X_expr_a[:, top_gene_idx]])
        scaler = StandardScaler()
        X_final_scaled = scaler.fit_transform(X_final)
        
        model = xgb.XGBClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.5, min_child_weight=5,
            reg_alpha=0.1, reg_lambda=1.0, random_state=42,
            scale_pos_weight=sum(y_a == 0) / max(sum(y_a == 1), 1),
        )
        model.fit(X_final_scaled, y_a, verbose=False)
        
        selected_genes = gene_names[top_gene_idx]
        
        model_data = {
            'model': model,
            'scaler': scaler,
            'selected_genes': list(selected_genes),
            'n_clinical': 3,
            'n_expression': 256,
            'source': 'TCGA COAD/READ RNA-seq',
            'label_definition': 'CR+PR=Responder, PD+SD=Non-responder',
            'auc': f"{auc_a:.4f} +/- {std_a:.4f}",
            'cv_method': 'CV-internal gene selection (no leakage)',
        }
        
        with open(MODEL_DIR / "treatment_response_v3.pkl", 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"  Saved: {MODEL_DIR / 'treatment_response_v3.pkl'}")
        
        # Feature importance
        feat_names = ['age', 'gender', 'stage'] + [str(g) for g in selected_genes]
        imp = model.feature_importances_
        top_idx = np.argsort(imp)[::-1][:15]
        print(f"\n  TOP FEATURES (v3):")
        for rank, idx in enumerate(top_idx, 1):
            print(f"    {rank:2d}. {feat_names[idx]:25s} importance={imp[idx]:.4f}")


if __name__ == "__main__":
    main()
