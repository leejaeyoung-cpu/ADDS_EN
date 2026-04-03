"""
Part A: Download CRC Treatment Response Datasets
=================================================
Sources:
  1. GSE28702: 83 CRC patients, FOLFOX response (R/NR), microarray
  2. GSE19860: CRC FOLFOX response prediction, microarray
  3. GSE72970: CRC chemo response (combined with above in literature)
  4. cBioPortal: TCGA CRC with treatment response annotations

Part B: Download CCLE Raw Gene Expression
==========================================
  DepMap OmicsExpressionAllGenesTPMLogp1Profile.csv
  For 46 drug target genes in our synergy model
"""
import requests
import json
import gzip
import csv
import io
import os
import time
import logging
import numpy as np
import pandas as pd
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

DATA_DIR = Path("F:/ADDS/data/ml_training")
RESPONSE_DIR = DATA_DIR / "chemo_response"
CCLE_DIR = DATA_DIR / "ccle_raw"

os.makedirs(RESPONSE_DIR, exist_ok=True)
os.makedirs(CCLE_DIR, exist_ok=True)


# ========================================
# PART A: GEO CRC Chemotherapy Response
# ========================================

def download_geo_series_matrix(gse_id, output_dir):
    """Download series matrix file from GEO."""
    url = f"https://ftp.ncbi.nlm.nih.gov/geo/series/{gse_id[:-3]}nnn/{gse_id}/matrix/{gse_id}_series_matrix.txt.gz"
    out_path = output_dir / f"{gse_id}_series_matrix.txt.gz"
    
    if out_path.exists():
        logger.info(f"{gse_id}: Already downloaded")
        return out_path
    
    logger.info(f"Downloading {gse_id} from GEO...")
    try:
        resp = requests.get(url, stream=True, timeout=120)
        if resp.status_code == 200:
            with open(out_path, 'wb') as f:
                for chunk in resp.iter_content(chunk_size=8192):
                    f.write(chunk)
            logger.info(f"  Saved: {out_path} ({out_path.stat().st_size / 1e6:.1f} MB)")
            return out_path
        else:
            logger.warning(f"  HTTP {resp.status_code}")
    except Exception as e:
        logger.error(f"  Error: {e}")
    
    return None


def download_geo_soft(gse_id, output_dir):
    """Download SOFT file for clinical/sample info."""
    url = f"https://ftp.ncbi.nlm.nih.gov/geo/series/{gse_id[:-3]}nnn/{gse_id}/soft/{gse_id}_family.soft.gz"
    out_path = output_dir / f"{gse_id}_family.soft.gz"
    
    if out_path.exists():
        logger.info(f"{gse_id} SOFT: Already downloaded")
        return out_path
    
    logger.info(f"Downloading {gse_id} SOFT...")
    try:
        resp = requests.get(url, stream=True, timeout=120)
        if resp.status_code == 200:
            with open(out_path, 'wb') as f:
                for chunk in resp.iter_content(chunk_size=8192):
                    f.write(chunk)
            logger.info(f"  Saved: {out_path}")
            return out_path
    except Exception as e:
        logger.error(f"  Error: {e}")
    
    return None


def parse_geo_soft_for_response(soft_path, gse_id):
    """Parse SOFT file to extract sample-level clinical data including chemo response."""
    samples = {}
    current_sample = None
    current_chars = {}
    
    open_fn = gzip.open if str(soft_path).endswith('.gz') else open
    
    with open_fn(soft_path, 'rt', errors='replace') as f:
        for line in f:
            line = line.strip()
            if line.startswith('^SAMPLE'):
                if current_sample and current_chars:
                    samples[current_sample] = current_chars
                current_sample = line.split('=')[-1].strip()
                current_chars = {}
            elif line.startswith('!Sample_characteristics_ch'):
                # Parse "key: value" format
                val = line.split('=', 1)[-1].strip().strip('"')
                if ':' in val:
                    key, v = val.split(':', 1)
                    current_chars[key.strip().lower()] = v.strip()
                else:
                    current_chars[f'char_{len(current_chars)}'] = val
            elif line.startswith('!Sample_title'):
                val = line.split('=', 1)[-1].strip().strip('"')
                current_chars['title'] = val
            elif line.startswith('!Sample_geo_accession'):
                val = line.split('=', 1)[-1].strip().strip('"')
                current_chars['geo_accession'] = val
    
    if current_sample and current_chars:
        samples[current_sample] = current_chars
    
    logger.info(f"  {gse_id}: Parsed {len(samples)} samples from SOFT")
    
    # Show example
    if samples:
        first = list(samples.values())[0]
        logger.info(f"  Sample keys: {list(first.keys())}")
    
    return samples


def parse_series_matrix(matrix_path):
    """Parse series matrix file to get expression data."""
    open_fn = gzip.open if str(matrix_path).endswith('.gz') else open
    
    header = None
    data_rows = []
    sample_ids = None
    in_data = False
    
    with open_fn(matrix_path, 'rt', errors='replace') as f:
        for line in f:
            line = line.strip()
            if line.startswith('!Series_') or line.startswith('!series_'):
                continue
            if line.startswith('!Sample_geo_accession'):
                sample_ids = line.split('\t')[1:]
                sample_ids = [s.strip('"') for s in sample_ids]
            if line.startswith('"ID_REF"'):
                header = line.split('\t')
                header = [h.strip('"') for h in header]
                in_data = True
                continue
            if line.startswith('!series_matrix_table_end'):
                break
            if in_data and line and not line.startswith('!'):
                parts = line.split('\t')
                if len(parts) > 1:
                    gene_id = parts[0].strip('"')
                    values = []
                    for v in parts[1:]:
                        v = v.strip('"').strip()
                        try:
                            values.append(float(v))
                        except:
                            values.append(np.nan)
                    data_rows.append([gene_id] + values)
    
    if not data_rows:
        logger.warning("  No data rows found in matrix!")
        return None, None
    
    # Build DataFrame
    cols = ['probe_id'] + (sample_ids if sample_ids else [f'sample_{i}' for i in range(len(data_rows[0])-1)])
    df = pd.DataFrame(data_rows, columns=cols[:len(data_rows[0])])
    df = df.set_index('probe_id')
    df = df.astype(float)
    
    logger.info(f"  Expression matrix: {df.shape[0]} probes × {df.shape[1]} samples")
    return df, sample_ids


def extract_response_labels(samples_dict, gse_id):
    """Extract treatment response labels from sample characteristics."""
    labels = {}
    
    for sample_name, chars in samples_dict.items():
        geo_acc = chars.get('geo_accession', sample_name)
        
        # Try different response field names
        response = None
        for key in chars:
            kl = key.lower()
            if any(x in kl for x in ['response', 'folfox', 'chemo', 'treatment', 'responder',
                                       'recist', 'outcome', 'sensitivity']):
                response = chars[key]
                break
        
        if response:
            response_lower = response.lower().strip()
            
            # Map to binary
            if any(x in response_lower for x in ['responder', 'response', 'sensitive', 'cr', 'pr',
                                                    'complete', 'partial', 'yes', 'r ']):
                if 'non' in response_lower or 'no' in response_lower:
                    labels[geo_acc] = 0  # Non-responder
                else:
                    labels[geo_acc] = 1  # Responder
            elif any(x in response_lower for x in ['non-responder', 'nonresponder', 'resistant',
                                                      'progressive', 'pd', 'sd', 'no', 'nr',
                                                      'insensitive']):
                labels[geo_acc] = 0  # Non-responder
            else:
                logger.info(f"    Unknown response: '{response}' for {geo_acc}")
    
    if labels:
        resp = sum(1 for v in labels.values() if v == 1)
        nonresp = sum(1 for v in labels.values() if v == 0)
        logger.info(f"  {gse_id} Labels: {resp} responders, {nonresp} non-responders")
    else:
        logger.warning(f"  {gse_id}: No response labels found!")
    
    return labels


# ========================================
# PART B: CCLE/DepMap Raw Expression
# ========================================

def download_ccle_expression():
    """Download CCLE gene expression from DepMap."""
    # Use DepMap Public 24Q2 - Expression TPM
    # Direct file URL
    url = "https://figshare.com/ndownloader/files/44623790"  # OmicsExpressionProteinCodingGenesTPMLogp1.csv
    
    out_path = CCLE_DIR / "CCLE_expression_tpm.csv"
    
    if out_path.exists():
        logger.info(f"CCLE expression: Already downloaded ({out_path.stat().st_size / 1e6:.1f} MB)")
        return out_path
    
    logger.info("Downloading CCLE expression from DepMap...")
    logger.info("  This is a large file (~200MB), please wait...")
    
    try:
        resp = requests.get(url, stream=True, timeout=300)
        if resp.status_code == 200:
            total = int(resp.headers.get('content-length', 0))
            downloaded = 0
            with open(out_path, 'wb') as f:
                for chunk in resp.iter_content(chunk_size=65536):
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total > 0 and downloaded % (10 * 1024 * 1024) == 0:
                        logger.info(f"  {downloaded / 1e6:.0f} / {total / 1e6:.0f} MB")
            
            logger.info(f"  Saved: {out_path} ({out_path.stat().st_size / 1e6:.1f} MB)")
            return out_path
        else:
            logger.warning(f"  HTTP {resp.status_code}")
    except Exception as e:
        logger.error(f"  Download error: {e}")
    
    # Fallback: try alternative URL
    logger.info("  Trying alternative download method...")
    alt_url = "https://ndownloader.figshare.com/files/44623790"
    try:
        resp = requests.get(alt_url, stream=True, timeout=300, allow_redirects=True)
        if resp.status_code == 200:
            with open(out_path, 'wb') as f:
                for chunk in resp.iter_content(chunk_size=65536):
                    f.write(chunk)
            logger.info(f"  Saved: {out_path}")
            return out_path
    except Exception as e:
        logger.error(f"  Alt download error: {e}")
    
    return None


def download_ccle_sample_info():
    """Download CCLE sample/cell line info for mapping."""
    url = "https://figshare.com/ndownloader/files/44623372"  # Model.csv
    out_path = CCLE_DIR / "CCLE_model_info.csv"
    
    if out_path.exists():
        logger.info(f"CCLE model info: Already downloaded")
        return out_path
    
    logger.info("Downloading CCLE model info...")
    try:
        resp = requests.get(url, timeout=60, allow_redirects=True)
        if resp.status_code == 200:
            with open(out_path, 'wb') as f:
                f.write(resp.content)
            logger.info(f"  Saved: {out_path}")
            return out_path
    except Exception as e:
        logger.error(f"  Error: {e}")
    
    return None


def main():
    print("=" * 70)
    print("DATA ACQUISITION: CRC Response + CCLE Expression")
    print("=" * 70)
    
    # ===== PART A: GEO Datasets =====
    print(f"\n{'='*70}")
    print("PART A: GEO CRC Chemotherapy Response Datasets")
    print("=" * 70)
    
    geo_datasets = ['GSE28702', 'GSE19860', 'GSE72970']
    all_labels = {}
    all_expr = {}
    
    for gse_id in geo_datasets:
        print(f"\n--- {gse_id} ---")
        
        # Download SOFT for clinical data
        soft_path = download_geo_soft(gse_id, RESPONSE_DIR)
        if soft_path:
            samples = parse_geo_soft_for_response(soft_path, gse_id)
            labels = extract_response_labels(samples, gse_id)
            
            if labels:
                all_labels[gse_id] = labels
                
                # Save clinical data
                clin_df = pd.DataFrame([
                    {'sample_id': k, 'response': v, 'dataset': gse_id, **samples.get(k, {})}
                    for k, v in labels.items()
                ])
                clin_df.to_csv(RESPONSE_DIR / f"{gse_id}_clinical.csv", index=False)
        
        # Download expression matrix
        matrix_path = download_geo_series_matrix(gse_id, RESPONSE_DIR)
        if matrix_path:
            expr_df, sample_ids = parse_series_matrix(matrix_path)
            if expr_df is not None:
                all_expr[gse_id] = expr_df
        
        time.sleep(2)
    
    # Summary
    print(f"\n{'='*70}")
    print("GEO SUMMARY")
    print("=" * 70)
    
    total_resp = 0
    total_nonresp = 0
    for gse_id, labels in all_labels.items():
        resp = sum(1 for v in labels.values() if v == 1)
        nonresp = sum(1 for v in labels.values() if v == 0)
        total_resp += resp
        total_nonresp += nonresp
        print(f"  {gse_id}: {resp} responders, {nonresp} non-responders")
    
    print(f"  TOTAL: {total_resp} responders, {total_nonresp} non-responders")
    print(f"  Combined with TCGA outcome (83 resp, 24 non-resp): {total_resp+83} resp, {total_nonresp+24} non-resp")
    
    # Save merged labels
    merged = []
    for gse_id, labels in all_labels.items():
        for sample, label in labels.items():
            merged.append({'sample_id': sample, 'response': label, 'source': gse_id})
    
    merged_df = pd.DataFrame(merged)
    merged_df.to_csv(RESPONSE_DIR / "merged_response_labels.csv", index=False)
    print(f"  Saved: {RESPONSE_DIR / 'merged_response_labels.csv'}")
    
    # ===== PART B: CCLE Expression =====
    print(f"\n{'='*70}")
    print("PART B: CCLE/DepMap Raw Gene Expression")
    print("=" * 70)
    
    ccle_path = download_ccle_expression()
    model_path = download_ccle_sample_info()
    
    if ccle_path and ccle_path.exists():
        # Parse for our target genes
        target_genes = [
            'ABL1', 'AKT1', 'AKT2', 'AKT3', 'AURKA', 'BRAF', 'CDK1', 'CDK2',
            'CDK5', 'CDK9', 'CHEK1', 'DHFR', 'EGFR', 'EPHA2', 'ERBB2', 'FLT3',
            'HDAC1', 'HDAC2', 'HDAC3', 'HSP90AA1', 'KIT', 'MAP2K1', 'MAP2K2',
            'MGMT', 'MTOR', 'NOTCH1', 'NR3C1', 'PARP1', 'PARP2', 'PDGFRA',
            'PDGFRB', 'PIK3CA', 'PIK3CB', 'PRKAA1', 'PRKAA2', 'PSMB5', 'RAF1',
            'RET', 'RRM1', 'SRC', 'TOP1', 'TOP2A', 'TOP2B', 'TUBB', 'TYMS', 'VEGFR2'
        ]
        # VEGFR2 gene name is KDR
        target_genes_alt = target_genes.copy()
        target_genes_alt.append('KDR')  # VEGFR2 = KDR gene
        target_genes_alt.append('TUBB1')  # Possible TUBB alias
        
        logger.info("Reading CCLE expression header...")
        header = pd.read_csv(ccle_path, nrows=0)
        cols = list(header.columns)
        
        # CCLE uses "GENE (ENTREZ_ID)" format for column names
        # Find matching columns
        gene_cols = {}
        for col in cols[1:]:  # Skip first col (cell line ID)
            gene_name = col.split(' ')[0] if ' ' in col else col
            if gene_name in target_genes_alt:
                gene_cols[gene_name] = col
        
        logger.info(f"Found {len(gene_cols)}/{len(target_genes)} target genes in CCLE data")
        logger.info(f"  Found: {sorted(gene_cols.keys())}")
        missing = set(target_genes_alt) - set(gene_cols.keys())
        if missing:
            logger.info(f"  Missing: {sorted(missing)}")
        
        if gene_cols:
            # Read only target gene columns
            usecols = [cols[0]] + list(gene_cols.values())
            logger.info(f"Reading CCLE expression for {len(usecols)-1} genes...")
            ccle_df = pd.read_csv(ccle_path, usecols=usecols)
            
            # Rename columns to gene names
            rename_map = {v: k for k, v in gene_cols.items()}
            ccle_df = ccle_df.rename(columns=rename_map)
            ccle_df = ccle_df.rename(columns={ccle_df.columns[0]: 'cell_line_id'})
            
            logger.info(f"CCLE expression: {ccle_df.shape[0]} cell lines × {ccle_df.shape[1]-1} genes")
            
            # Save
            ccle_df.to_csv(CCLE_DIR / "ccle_target_expression.csv", index=False)
            logger.info(f"Saved: {CCLE_DIR / 'ccle_target_expression.csv'}")
            
            # Show stats
            print(f"\n  Cell lines: {len(ccle_df)}")
            print(f"  Target genes: {ccle_df.shape[1]-1}")
            for gene in sorted(gene_cols.keys())[:10]:
                vals = ccle_df[gene].dropna()
                print(f"    {gene:15s}: mean={vals.mean():.2f}, std={vals.std():.2f}")
    
    # Cell line name mapping
    if model_path and model_path.exists():
        logger.info("Reading CCLE model info for cell line mapping...")
        model_df = pd.read_csv(model_path)
        
        # Find relevant columns
        name_cols = [c for c in model_df.columns if any(x in c.lower() for x in ['name', 'cell', 'stripped'])]
        logger.info(f"  Model info columns: {name_cols[:10]}")
        
        model_df.to_csv(CCLE_DIR / "ccle_model_info.csv", index=False)
        logger.info(f"  Saved cell line mapping")
    
    print(f"\n{'='*70}")
    print("DATA ACQUISITION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
