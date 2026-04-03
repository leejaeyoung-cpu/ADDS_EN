"""
Download and process GSE39582 for treatment response prediction.
================================================================
GSE39582: 585 CRC patients with clinical follow-up and chemo data.
- Gene expression: Affymetrix microarray
- Clinical: chemotherapy (FOLFOX/FOLFIRI), DFS, OS, response

This is the largest publicly available CRC gene expression dataset
with documented chemotherapy response and survival outcomes.
"""
import requests
import gzip
import json
import logging
from pathlib import Path
from io import StringIO

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

DATA_DIR = Path("F:/ADDS/data/ml_training")
DATA_DIR.mkdir(parents=True, exist_ok=True)


def download_geo_series_matrix(gse_id="GSE39582"):
    """Download series matrix file from GEO (contains expression + clinical data)."""
    # GEO series matrix URLs
    urls = [
        f"https://ftp.ncbi.nlm.nih.gov/geo/series/{gse_id[:5]}nnn/{gse_id}/matrix/{gse_id}_series_matrix.txt.gz",
        f"https://ftp.ncbi.nlm.nih.gov/geo/series/{gse_id[:5]}nnn/{gse_id}/matrix/{gse_id}-GPL570_series_matrix.txt.gz",
    ]
    
    for url in urls:
        try:
            logger.info(f"Trying: {url}")
            resp = requests.get(url, timeout=60, stream=True)
            if resp.status_code == 200:
                content_len = int(resp.headers.get('content-length', 0))
                logger.info(f"Downloading {content_len/1e6:.1f} MB...")
                
                outfile = DATA_DIR / f"{gse_id}_series_matrix.txt.gz"
                with open(outfile, 'wb') as f:
                    for chunk in resp.iter_content(chunk_size=8192):
                        f.write(chunk)
                
                logger.info(f"Saved: {outfile}")
                return outfile
        except Exception as e:
            logger.warning(f"Failed: {e}")
    
    return None


def download_geo_soft(gse_id="GSE39582"):
    """Download SOFT format (clinical metadata) from GEO."""
    url = f"https://ftp.ncbi.nlm.nih.gov/geo/series/{gse_id[:5]}nnn/{gse_id}/soft/{gse_id}_family.soft.gz"
    
    try:
        logger.info(f"Downloading SOFT: {url}")
        resp = requests.get(url, timeout=60, stream=True)
        if resp.status_code == 200:
            outfile = DATA_DIR / f"{gse_id}_family.soft.gz"
            with open(outfile, 'wb') as f:
                for chunk in resp.iter_content(chunk_size=8192):
                    f.write(chunk)
            logger.info(f"Saved: {outfile}")
            return outfile
    except Exception as e:
        logger.warning(f"Failed: {e}")
    
    return None


def parse_series_matrix(gzfile):
    """Parse GEO series matrix to extract clinical metadata and expression."""
    import pandas as pd
    
    logger.info(f"Parsing: {gzfile}")
    
    clinical_lines = []
    expr_lines = []
    in_table = False
    
    with gzip.open(gzfile, 'rt', errors='replace') as f:
        for line in f:
            line = line.strip()
            if line.startswith('!Sample_'):
                clinical_lines.append(line)
            elif line == '!series_matrix_table_begin':
                in_table = True
                continue
            elif line == '!series_matrix_table_end':
                in_table = False
                continue
            elif in_table:
                expr_lines.append(line)
    
    # Parse clinical data
    clinical = {}
    for line in clinical_lines:
        parts = line.split('\t')
        key = parts[0].replace('!Sample_', '')
        values = [v.strip('"') for v in parts[1:]]
        clinical[key] = values
    
    clinical_df = pd.DataFrame(clinical)
    logger.info(f"Clinical metadata: {len(clinical_df)} samples x {len(clinical_df.columns)} features")
    logger.info(f"Clinical columns: {list(clinical_df.columns[:20])}")
    
    # Parse expression data (first line is header with sample IDs)
    if expr_lines:
        expr_text = '\n'.join(expr_lines)
        expr_df = pd.read_csv(StringIO(expr_text), sep='\t', index_col=0)
        logger.info(f"Expression: {expr_df.shape}")
    else:
        expr_df = None
    
    return clinical_df, expr_df


def extract_chemo_response(clinical_df):
    """Extract chemotherapy response labels from clinical metadata."""
    import pandas as pd
    
    logger.info(f"Looking for chemo/response columns...")
    
    # Print all clinical columns
    for col in clinical_df.columns:
        unique_vals = clinical_df[col].unique()[:5]
        logger.info(f"  {col}: {unique_vals}")
    
    # GSE39582 has characteristics columns containing key-value pairs
    # Look for chemotherapy, DFS, OS, response
    chemo_cols = [c for c in clinical_df.columns if 'characteristics' in c.lower()]
    logger.info(f"\nCharacteristics columns: {chemo_cols}")
    
    # Parse characteristics to find chemo/response data
    for col in chemo_cols:
        sample_vals = clinical_df[col].unique()[:10]
        for val in sample_vals:
            if any(kw in str(val).lower() for kw in ['chemo', 'response', 'dfs', 'event', 'treatment']):
                logger.info(f"  --> Found relevant data in {col}: {val}")
    
    return clinical_df


def main():
    print("=" * 60)
    print("GSE39582: CRC Treatment Response Dataset")
    print("=" * 60)
    
    # Step 1: Download series matrix
    gzfile = download_geo_series_matrix()
    
    if gzfile is None:
        logger.error("Failed to download GSE39582")
        return
    
    # Step 2: Parse clinical metadata and expression
    clinical_df, expr_df = parse_series_matrix(gzfile)
    
    # Step 3: Extract chemo response
    chemo_df = extract_chemo_response(clinical_df)
    
    # Save intermediate results
    clinical_df.to_csv(DATA_DIR / "gse39582_clinical.csv", index=False)
    logger.info(f"Saved clinical data: {DATA_DIR / 'gse39582_clinical.csv'}")
    
    if expr_df is not None:
        # Save expression (compressed to save space)
        expr_file = DATA_DIR / "gse39582_expression.csv.gz"
        expr_df.to_csv(expr_file, compression='gzip')
        logger.info(f"Saved expression: {expr_file} ({expr_df.shape})")
    
    print(f"\nDownloaded and parsed GSE39582:")
    print(f"  Clinical: {clinical_df.shape}")
    print(f"  Expression: {expr_df.shape if expr_df is not None else 'None'}")


if __name__ == "__main__":
    main()
