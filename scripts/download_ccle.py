"""
Download CCLE/DepMap gene expression for our 46 drug targets.
Uses DepMap portal API to find correct download URL.
"""
import requests
import os
import json
import csv
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

CCLE_DIR = Path("F:/ADDS/data/ml_training/ccle_raw")
os.makedirs(CCLE_DIR, exist_ok=True)

TARGET_GENES = [
    'ABL1', 'AKT1', 'AKT2', 'AKT3', 'AURKA', 'BRAF', 'CDK1', 'CDK2',
    'CDK5', 'CDK9', 'CHEK1', 'DHFR', 'EGFR', 'EPHA2', 'ERBB2', 'FLT3',
    'HDAC1', 'HDAC2', 'HDAC3', 'HSP90AA1', 'KIT', 'MAP2K1', 'MAP2K2',
    'MGMT', 'MTOR', 'NOTCH1', 'NR3C1', 'PARP1', 'PARP2', 'PDGFRA',
    'PDGFRB', 'PIK3CA', 'PIK3CB', 'PRKAA1', 'PRKAA2', 'PSMB5', 'RAF1',
    'RET', 'RRM1', 'SRC', 'TOP1', 'TOP2A', 'TOP2B', 'TUBB', 'TYMS',
    'KDR',  # = VEGFR2
]


def try_depmap_downloads():
    """Try various DepMap download endpoints."""
    
    # Method 1: DepMap taiga dataset (most reliable)
    # The expression data is available from multiple figshare articles
    figshare_ids = [
        '22765112',  # DepMap 23Q4
        '24667905',  # DepMap 24Q2
        '25880521',  # DepMap 24Q4 
    ]
    
    for article_id in figshare_ids:
        logger.info(f"Checking figshare article {article_id}...")
        try:
            resp = requests.get(f'https://api.figshare.com/v2/articles/{article_id}/files', timeout=30)
            if resp.status_code != 200:
                continue
            
            files = resp.json()
            for f in files:
                name = f.get('name', '')
                if 'Expression' in name and ('TPM' in name or 'expression' in name.lower()):
                    logger.info(f"  Found: {name}")
                    download_url = f.get('download_url', '')
                    if download_url:
                        return download_url, name
                    
        except Exception as e:
            logger.warning(f"  Error: {e}")
    
    # Method 2: Direct CCLE legacy URLs
    legacy_urls = [
        ('https://data.broadinstitute.org/ccle/CCLE_RNAseq_genes_rpkm_20180929.gct.gz', 'CCLE_expression_legacy.gct.gz'),
        ('https://ndownloader.figshare.com/files/40449021', 'OmicsExpression.csv'),  # 23Q4
    ]
    
    for url, name in legacy_urls:
        logger.info(f"Trying legacy URL: {name}...")
        try:
            resp = requests.head(url, timeout=30, allow_redirects=True)
            if resp.status_code == 200:
                logger.info(f"  Available! Size: {resp.headers.get('content-length', '?')} bytes")
                return url, name
        except:
            pass
    
    return None, None


def download_file(url, output_path, max_mb=500):
    """Download file with progress."""
    logger.info(f"Downloading {output_path.name}...")
    resp = requests.get(url, stream=True, timeout=600, allow_redirects=True)
    
    if resp.status_code != 200:
        logger.error(f"  HTTP {resp.status_code}")
        return False
    
    total = int(resp.headers.get('content-length', 0))
    downloaded = 0
    
    with open(output_path, 'wb') as f:
        for chunk in resp.iter_content(chunk_size=65536):
            f.write(chunk)
            downloaded += len(chunk)
            if total > 0 and downloaded % (20 * 1024 * 1024) < 65536:
                logger.info(f"  {downloaded / 1e6:.0f} / {total / 1e6:.0f} MB ({downloaded/total*100:.0f}%)")
    
    logger.info(f"  Done: {output_path} ({downloaded / 1e6:.1f} MB)")
    return True


def extract_target_genes(ccle_path, output_path):
    """Extract only target gene columns from full CCLE file."""
    logger.info(f"Extracting target genes from {ccle_path.name}...")
    
    # Read header to find gene columns
    with open(ccle_path, 'r') as f:
        header = f.readline().strip().split(',')
    
    # Find target gene columns
    # DepMap format: "GENE (ENTREZ_ID)" or just "GENE"
    gene_col_map = {}
    for i, col in enumerate(header):
        gene = col.split(' ')[0].strip('"')
        if gene in TARGET_GENES:
            gene_col_map[gene] = i
    
    logger.info(f"  Matched {len(gene_col_map)}/{len(TARGET_GENES)} genes")
    
    if not gene_col_map:
        logger.warning("  No genes matched! Showing first 10 columns:")
        for col in header[1:11]:
            logger.info(f"    {col}")
        return
    
    # Read only needed columns
    col_indices = [0] + sorted(gene_col_map.values())
    result_rows = []
    
    with open(ccle_path, 'r') as f:
        reader = csv.reader(f)
        for i, row in enumerate(reader):
            try:
                selected = [row[j] for j in col_indices]
                result_rows.append(selected)
            except IndexError:
                pass
    
    # Write output
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(result_rows)
    
    logger.info(f"  Saved: {output_path} ({len(result_rows)-1} cell lines)")


def main():
    print("=" * 70)
    print("CCLE/DepMap Target Gene Expression Download")
    print("=" * 70)
    
    out_path = CCLE_DIR / "CCLE_expression_tpm.csv"
    target_path = CCLE_DIR / "ccle_target_expression.csv"
    
    if target_path.exists() and target_path.stat().st_size > 1000:
        logger.info(f"Target expression already exists: {target_path}")
        return
    
    if not out_path.exists():
        url, name = try_depmap_downloads()
        
        if url:
            logger.info(f"Best URL: {url}")
            success = download_file(url, out_path)
            if not success:
                logger.error("Download failed!")
                return
        else:
            logger.error("No working download URL found!")
            logger.info("Manual download required:")
            logger.info("  1. Go to https://depmap.org/portal/download/all/")
            logger.info("  2. Search for 'Expression'")
            logger.info("  3. Download OmicsExpressionProteinCodingGenesTPMLogp1.csv")
            logger.info(f"  4. Save to: {out_path}")
            return
    
    # Extract target genes
    extract_target_genes(out_path, target_path)


if __name__ == "__main__":
    main()
