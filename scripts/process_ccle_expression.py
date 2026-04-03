"""
Process CCLE Expression data for O'Neil cell lines.
Extracts top-variance gene expression features and maps to O'Neil names.
"""
import pandas as pd
import numpy as np
import json
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

DATA_DIR = Path("F:/ADDS/data/ml_training")

# DepMap cell line name -> O'Neil name mapping
# CCLE uses DepMap IDs (ACH-XXXXXX), but the CSV has cell line names as first column
# We need to match CCLE cell line names to O'Neil names


def main():
    # Step 1: Load raw CCLE expression
    logger.info("Loading CCLE expression (this may take a minute)...")
    ccle_raw = DATA_DIR / "ccle_expression_raw.csv"
    
    # Read first few rows to understand format
    df_head = pd.read_csv(ccle_raw, nrows=3, index_col=0)
    logger.info(f"Columns: {len(df_head.columns)}, Index type: {df_head.index.name}")
    logger.info(f"Sample indices: {list(df_head.index[:5])}")
    logger.info(f"Sample columns: {list(df_head.columns[:5])}")
    
    # The index is DepMap ID like ACH-001113
    # We need a mapping file or we can try to map based on CCLE metadata
    # First, let's try to download the DepMap cell line metadata (sample_info.csv)
    import requests
    
    meta_urls = [
        "https://ndownloader.figshare.com/files/35020903",  # sample_info.csv
    ]
    
    meta_file = DATA_DIR / "depmap_sample_info.csv"
    if not meta_file.exists():
        for url in meta_urls:
            try:
                logger.info(f"Downloading DepMap metadata...")
                resp = requests.get(url, timeout=30)
                if resp.status_code == 200 and len(resp.content) > 10000:
                    with open(meta_file, 'wb') as f:
                        f.write(resp.content)
                    logger.info(f"Saved: {meta_file} ({len(resp.content)} bytes)")
                    break
            except Exception as e:
                logger.warning(f"Failed: {e}")
    
    if not meta_file.exists():
        logger.error("Cannot download DepMap metadata. Cannot map cell lines.")
        return
    
    # Step 2: Load metadata and map cell line names
    meta = pd.read_csv(meta_file)
    logger.info(f"DepMap metadata: {len(meta)} cell lines")
    logger.info(f"Columns: {list(meta.columns[:10])}")
    
    # Find the right columns
    id_col = None
    name_col = None
    for col in meta.columns:
        if 'DepMap' in col or col == 'ModelID':
            id_col = col
        if 'stripped' in col.lower() or 'cell_line_name' in col.lower():
            name_col = col
    
    if id_col is None:
        # Look for the first column that contains "ACH-" values  
        for col in meta.columns:
            if meta[col].astype(str).str.contains('ACH-').any():
                id_col = col
                break
    
    logger.info(f"ID column: {id_col}")
    logger.info(f"Name column: {name_col}")
    
    # Build name -> DepMap ID mapping
    if id_col and name_col:
        name_to_id = dict(zip(meta[name_col].str.upper(), meta[id_col]))
    else:
        # Try alternative: use 'CCLE_Name' column
        for col in meta.columns:
            if 'ccle' in col.lower() and 'name' in col.lower():
                name_col = col
                break
        if name_col:
            # CCLE names are like "A549_LUNG"
            ccle_to_id = dict(zip(meta[name_col], meta[id_col] if id_col else meta.iloc[:, 0]))
            # Extract just the cell line part
            name_to_id = {}
            for ccle_name, dep_id in ccle_to_id.items():
                if isinstance(ccle_name, str) and '_' in ccle_name:
                    short_name = ccle_name.split('_')[0].upper()
                    name_to_id[short_name] = dep_id
        else:
            logger.error("Cannot find name columns in metadata")
            return
    
    logger.info(f"Name -> ID mapping: {len(name_to_id)} entries")
    
    # Load O'Neil cell lines
    synergy = pd.read_csv(DATA_DIR / "oneil_synergy.csv")
    oneil_lines = sorted(synergy['cell_line'].unique())
    logger.info(f"O'Neil cell lines: {len(oneil_lines)}")
    
    # Match
    matched = {}
    unmatched = []
    for cl in oneil_lines:
        cl_upper = cl.upper().replace('-', '').replace(' ', '')
        # Try exact
        if cl_upper in name_to_id:
            matched[cl] = name_to_id[cl_upper]
        else:
            # Try fuzzy: strip non-alpha chars
            cl_stripped = ''.join(c for c in cl_upper if c.isalnum())
            found = False
            for name, dep_id in name_to_id.items():
                if not isinstance(name, str):
                    continue
                name_stripped = ''.join(c for c in name if c.isalnum())
                if cl_stripped == name_stripped:
                    matched[cl] = dep_id
                    found = True
                    break
            if not found:
                unmatched.append(cl)
    
    logger.info(f"Matched: {len(matched)}/{len(oneil_lines)}")
    if unmatched:
        logger.warning(f"Unmatched: {unmatched}")
    
    # Step 3: Extract expression for matched cell lines
    logger.info("Loading full CCLE expression matrix...")
    df_full = pd.read_csv(ccle_raw, index_col=0)
    logger.info(f"Full matrix: {df_full.shape}")
    
    # Filter to matched cell lines
    dep_ids = list(matched.values())
    available_ids = [did for did in dep_ids if did in df_full.index]
    logger.info(f"Available in expression: {len(available_ids)}/{len(dep_ids)}")
    
    df_matched = df_full.loc[available_ids]
    
    # Step 4: Select top-variance genes (256 features)
    n_top_genes = 256
    gene_var = df_matched.var(axis=0)
    top_genes = gene_var.nlargest(n_top_genes).index.tolist()
    
    df_selected = df_matched[top_genes]
    logger.info(f"Selected top {n_top_genes} genes by variance")
    
    # Step 5: Normalize (z-score per gene)
    df_norm = (df_selected - df_selected.mean()) / (df_selected.std() + 1e-8)
    
    # Map back to O'Neil names
    id_to_name = {v: k for k, v in matched.items()}
    df_norm.index = [id_to_name.get(idx, idx) for idx in df_norm.index]
    
    # Add zero-vector for unmatched cell lines
    for cl in unmatched:
        df_norm.loc[cl] = 0.0
    
    # Save
    expr_file = DATA_DIR / "cell_line_expression_256.csv"
    df_norm.to_csv(expr_file)
    logger.info(f"Saved: {expr_file} ({len(df_norm)} lines x {n_top_genes} genes)")
    
    # Also save as a dict for model integration
    import pickle
    expr_dict = {}
    for cl in df_norm.index:
        expr_dict[cl] = df_norm.loc[cl].values.astype(np.float32)
    
    expr_pkl = Path("F:/ADDS/models/synergy/cell_line_expression.pkl")
    with open(expr_pkl, 'wb') as f:
        pickle.dump(expr_dict, f)
    logger.info(f"Saved: {expr_pkl}")
    
    # Summary
    print(f"\n{'='*60}")
    print(f"CCLE Expression Features for O'Neil Cell Lines")
    print(f"{'='*60}")
    print(f"Matched: {len(matched)}/{len(oneil_lines)}")
    print(f"Unmatched (zero-vector): {unmatched}")
    print(f"Top genes: {n_top_genes} (by variance)")
    print(f"Feature matrix: {df_norm.shape}")
    print(f"Sample genes: {top_genes[:10]}")
    
    # Save metadata
    meta_out = {
        'n_cell_lines': len(df_norm),
        'n_genes': n_top_genes,
        'matched': len(matched),
        'unmatched': unmatched,
        'top_genes': top_genes,
        'source': 'DepMap CCLE OmicsExpressionProteinCodingGenesTPMLogp1',
    }
    with open(DATA_DIR / "cell_line_expression_meta.json", 'w') as f:
        json.dump(meta_out, f, indent=2)


if __name__ == "__main__":
    main()
