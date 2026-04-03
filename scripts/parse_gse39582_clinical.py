"""
Parse GSE39582 SOFT file for clinical annotations.
The series matrix lacks chemo/survival data - SOFT format has it.
"""
import gzip
import pandas as pd
import logging
from pathlib import Path
from collections import defaultdict
import requests

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

DATA_DIR = Path("F:/ADDS/data/ml_training")


def download_soft():
    """Download GSE39582 SOFT format."""
    url = "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE39nnn/GSE39582/soft/GSE39582_family.soft.gz"
    outfile = DATA_DIR / "GSE39582_family.soft.gz"
    
    if outfile.exists():
        logger.info(f"SOFT file exists: {outfile}")
        return outfile
    
    logger.info(f"Downloading SOFT: {url}")
    resp = requests.get(url, timeout=120, stream=True)
    if resp.status_code == 200:
        with open(outfile, 'wb') as f:
            for chunk in resp.iter_content(8192):
                f.write(chunk)
        logger.info(f"Saved: {outfile}")
        return outfile
    return None


def parse_soft_clinical(gzfile):
    """Parse SOFT file to extract per-sample clinical annotations."""
    logger.info(f"Parsing SOFT: {gzfile}")
    
    samples = {}
    current_sample = None
    
    with gzip.open(gzfile, 'rt', errors='replace') as f:
        for line in f:
            line = line.strip()
            
            if line.startswith('^SAMPLE = '):
                current_sample = line.split(' = ')[1]
                samples[current_sample] = {}
            
            elif line.startswith('!Sample_characteristics_ch1 = ') and current_sample:
                # Parse key: value pairs
                chars = line[len('!Sample_characteristics_ch1 = '):]
                if ': ' in chars:
                    key, val = chars.split(': ', 1)
                    key = key.strip().lower().replace(' ', '_').replace('.', '_')
                    samples[current_sample][key] = val.strip()
    
    df = pd.DataFrame.from_dict(samples, orient='index')
    logger.info(f"Parsed {len(df)} samples with {len(df.columns)} characteristics")
    logger.info(f"Columns: {list(df.columns)}")
    
    return df


def main():
    # Download SOFT
    soft_file = download_soft()
    if soft_file is None:
        logger.error("Failed to download SOFT file")
        return
    
    # Parse clinical
    clinical = parse_soft_clinical(soft_file)
    
    # Show sample values for each column
    for col in sorted(clinical.columns):
        unique = clinical[col].dropna().unique()
        print(f"  {col}: {len(unique)} unique values -> {list(unique[:5])}")
    
    # Identify chemo/response columns
    chemo_cols = [c for c in clinical.columns if any(kw in c.lower() for kw in
                  ['chemo', 'response', 'dfs', 'os_', 'event', 'treatment', 'relapse'])]
    print(f"\nChemo/response columns: {chemo_cols}")
    
    if chemo_cols:
        for col in chemo_cols:
            print(f"\n{col}:")
            print(clinical[col].value_counts())
    
    # Save
    outfile = DATA_DIR / "gse39582_clinical_full.csv"
    clinical.to_csv(outfile)
    logger.info(f"Saved: {outfile}")


if __name__ == "__main__":
    main()
