"""
Expand CL Expression Coverage
==============================
Current: 38 CLs (9.3% of DrugComb)
Goal: Use full DepMap CCLE expression data to create PCA embeddings for 1000+ CLs
Then retrain DeepSynergy v5 with expanded coverage.
"""

import pickle, json, logging, time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DATA_DIR = Path("F:/ADDS/data/ml_training")
DEPMAP_DIR = DATA_DIR / "depmap"

def main():
    t0 = time.time()
    
    # ================================================================
    # Step 1: Load existing embeddings
    # ================================================================
    logger.info("=== Step 1: Check existing embeddings ===")
    
    # Check v2 embedding
    emb_v2_path = DEPMAP_DIR / "cellline_embedding_v2.pkl"
    with open(emb_v2_path, 'rb') as f:
        emb_v2 = pickle.load(f)
    logger.info("cellline_embedding_v2: type=%s", type(emb_v2).__name__)
    
    if isinstance(emb_v2, dict):
        keys = list(emb_v2.keys())
        logger.info("  Entries: %d", len(emb_v2))
        logger.info("  Sample keys: %s", keys[:5])
        for k in keys[:3]:
            v = emb_v2[k]
            if hasattr(v, 'shape'):
                logger.info("  %s: shape=%s", k, v.shape)
            elif hasattr(v, '__len__'):
                logger.info("  %s: len=%d", k, len(v))
    elif isinstance(emb_v2, pd.DataFrame):
        logger.info("  Shape: %s", emb_v2.shape)
        logger.info("  Index: %s", list(emb_v2.index[:5]))
    elif hasattr(emb_v2, 'shape'):
        logger.info("  Shape: %s", emb_v2.shape)
    
    # Check v1 embedding
    emb_v1_path = DEPMAP_DIR / "cellline_embedding.pkl"
    with open(emb_v1_path, 'rb') as f:
        emb_v1 = pickle.load(f)
    logger.info("\ncellline_embedding: type=%s", type(emb_v1).__name__)
    
    if isinstance(emb_v1, dict):
        keys = list(emb_v1.keys())
        logger.info("  Entries: %d", len(emb_v1))
        logger.info("  Sample keys: %s", keys[:5])
        for k in keys[:3]:
            v = emb_v1[k]
            if hasattr(v, 'shape'):
                logger.info("  %s: shape=%s", k, v.shape)
    elif isinstance(emb_v1, pd.DataFrame):
        logger.info("  Shape: %s", emb_v1.shape)
        logger.info("  Index: %s", list(emb_v1.index[:5]))
    elif hasattr(emb_v1, 'shape'):
        logger.info("  Shape: %s", emb_v1.shape)

    # ================================================================
    # Step 2: Load ACH ID to name mapping
    # ================================================================
    logger.info("\n=== Step 2: ACH to name mapping ===")
    with open(DEPMAP_DIR / "ach_to_name_full_v2.json") as f:
        ach_map = json.load(f)
    logger.info("ACH map: %d entries", len(ach_map))
    
    # Sample info
    info = pd.read_csv(DATA_DIR / "depmap_sample_info.csv")
    logger.info("Sample info: %d CLs", len(info))
    
    # ================================================================
    # Step 3: Check CCLE expression data
    # ================================================================
    logger.info("\n=== Step 3: CCLE expression ===")
    
    parquet_path = DEPMAP_DIR / "ccle_expression.parquet"
    csv_path = DEPMAP_DIR / "ccle_expression.csv"
    
    if parquet_path.exists():
        logger.info("Loading from parquet (faster)...")
        expr_df = pd.read_parquet(parquet_path)
    else:
        logger.info("Loading from CSV...")
        expr_df = pd.read_csv(csv_path, index_col=0, low_memory=False)
    
    logger.info("Expression: %d CLs x %d genes", expr_df.shape[0], expr_df.shape[1])
    logger.info("Index sample: %s", list(expr_df.index[:5]))
    
    # ================================================================
    # Step 4: Create PCA embeddings for all CLs
    # ================================================================
    logger.info("\n=== Step 4: PCA embedding ===")
    
    # Clean expression data
    expr_clean = expr_df.dropna(axis=1, how='all').fillna(0)
    logger.info("After cleaning: %d CLs x %d genes", expr_clean.shape[0], expr_clean.shape[1])
    
    # Standardize
    scaler = StandardScaler()
    expr_scaled = scaler.fit_transform(expr_clean)
    
    # PCA to 256d
    n_components = 256
    pca = PCA(n_components=n_components, random_state=42)
    embeddings = pca.fit_transform(expr_scaled)
    
    explained_var = pca.explained_variance_ratio_.sum()
    logger.info("PCA %dd: explained variance=%.3f", n_components, explained_var)
    
    # ================================================================
    # Step 5: Map ACH IDs to cell line names
    # ================================================================
    logger.info("\n=== Step 5: Name mapping ===")
    
    # Build comprehensive name map
    name_map = {}
    
    # From sample info
    for _, row in info.iterrows():
        ach = str(row['DepMap_ID'])
        names = []
        if pd.notna(row.get('cell_line_name')): names.append(str(row['cell_line_name']))
        if pd.notna(row.get('stripped_cell_line_name')): names.append(str(row['stripped_cell_line_name']))
        if pd.notna(row.get('CCLE_Name')): names.append(str(row['CCLE_Name']).split('_')[0])
        if pd.notna(row.get('alias')): 
            for a in str(row['alias']).split(','):
                names.append(a.strip())
        name_map[ach] = list(set(n.upper().strip() for n in names if n and n != 'NAN'))
    
    # From ACH JSON  
    for ach, name in ach_map.items():
        if ach not in name_map:
            name_map[ach] = [name.upper().strip()]
        else:
            name_map[ach].append(name.upper().strip())
    
    logger.info("Name map: %d ACH IDs", len(name_map))
    
    # Create embedding DataFrame
    emb_rows = {}
    for i, ach_id in enumerate(expr_clean.index):
        ach_str = str(ach_id)
        names = name_map.get(ach_str, [ach_str])
        for name in names:
            emb_rows[name] = embeddings[i]
            # Also add cleaned variants
            clean = name.replace('-','').replace('_','').replace(' ','')
            if clean != name:
                emb_rows[clean] = embeddings[i]
    
    logger.info("Total name entries: %d (for %d unique CLs)", len(emb_rows), len(expr_clean))
    
    # ================================================================
    # Step 6: Match against DrugComb
    # ================================================================
    logger.info("\n=== Step 6: DrugComb matching ===")
    
    dc = pd.read_csv(DATA_DIR / "synergy_combined.csv", usecols=['cell_line'], low_memory=False)
    dc_cls = set(dc.cell_line.str.upper().str.strip().unique())
    logger.info("DrugComb CLs: %d", len(dc_cls))
    
    matched = set()
    unmatched = set()
    for cl in dc_cls:
        cl_upper = cl.upper().strip()
        cl_clean = cl_upper.replace('-','').replace('_','').replace(' ','')
        if cl_upper in emb_rows or cl_clean in emb_rows:
            matched.add(cl)
        else:
            unmatched.add(cl)
    
    logger.info("Matched: %d/%d (%.1f%%)", len(matched), len(dc_cls), 100*len(matched)/len(dc_cls))
    logger.info("Unmatched sample: %s", list(unmatched)[:20])
    
    # ================================================================  
    # Step 7: Save expanded embeddings
    # ================================================================
    logger.info("\n=== Step 7: Save ===")
    
    # Save as DataFrame for easy loading
    emb_df = pd.DataFrame(emb_rows).T
    emb_df.columns = [f'PC{i+1}' for i in range(n_components)]
    
    out_path = DATA_DIR / "cell_line_expression_256_expanded.csv"
    emb_df.to_csv(out_path)
    logger.info("Saved: %s (%d entries, %.1f MB)", 
                out_path.name, len(emb_df), out_path.stat().st_size/1e6)
    
    # Also save as pickle for faster loading
    emb_dict = {name: emb_rows[name].astype(np.float32) for name in emb_rows}
    pkl_path = DATA_DIR / "cell_line_expression_256_expanded.pkl"
    with open(pkl_path, 'wb') as f:
        pickle.dump(emb_dict, f)
    logger.info("Saved pickle: %s (%.1f MB)", pkl_path.name, pkl_path.stat().st_size/1e6)
    
    elapsed = time.time() - t0
    logger.info("\n=== Complete (%.1fs) ===", elapsed)
    logger.info("Coverage: %d/%d CLs (%.1f%%) vs previous 38 CLs (9.3%%)",
                len(matched), len(dc_cls), 100*len(matched)/len(dc_cls))


if __name__ == "__main__":
    main()
