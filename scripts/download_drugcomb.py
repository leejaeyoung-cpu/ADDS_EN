"""
Task 3: DrugComb 1000K+ Expansion
==================================
Download full DrugComb synergy data via API to expand from O'Neil 23K → 1000K+.
"""
import requests
import json
import csv
import time
import logging
import numpy as np
import pandas as pd
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

DATA_DIR = Path("F:/ADDS/data/ml_training")
DRUGCOMB_DIR = DATA_DIR / "drugcomb"
DRUGCOMB_DIR.mkdir(exist_ok=True)


def download_drugcomb_csv():
    """Download DrugComb data as CSV from direct download."""
    urls = [
        "https://drugcomb.fimm.fi/jing/summary_v_1_5.csv",
        "https://drugcomb.fimm.fi/api/export/summary",
    ]
    
    for url in urls:
        logger.info(f"Trying: {url}")
        try:
            resp = requests.get(url, timeout=300, stream=True,
                              headers={'Accept': 'text/csv,*/*'})
            if resp.status_code == 200:
                out = DRUGCOMB_DIR / "drugcomb_summary.csv"
                downloaded = 0
                with open(out, 'wb') as f:
                    for chunk in resp.iter_content(chunk_size=65536):
                        f.write(chunk)
                        downloaded += len(chunk)
                        if downloaded % (50 * 1024 * 1024) < 65536:
                            logger.info(f"  {downloaded/1e6:.0f} MB...")
                
                size = out.stat().st_size
                logger.info(f"  Downloaded: {size/1e6:.1f} MB")
                
                if size > 10000:
                    return out
            else:
                logger.info(f"  HTTP {resp.status_code}")
        except Exception as e:
            logger.warning(f"  Error: {e}")
    
    return None


def download_drugcomb_block():
    """Download data in blocks via API pagination."""
    base_url = "https://drugcomb.fimm.fi/api"
    
    all_data = []
    offset = 0
    batch_size = 10000
    max_records = 1200000
    
    logger.info(f"Downloading DrugComb data in batches of {batch_size}...")
    
    while offset < max_records:
        try:
            resp = requests.get(
                f"{base_url}/summary?offset={offset}&limit={batch_size}",
                timeout=120,
                headers={'Accept': 'application/json'}
            )
            
            if resp.status_code != 200:
                logger.warning(f"  HTTP {resp.status_code} at offset {offset}")
                break
            
            batch = resp.json()
            if not batch or len(batch) == 0:
                break
            
            all_data.extend(batch)
            logger.info(f"  Offset {offset}: {len(batch)} records (total: {len(all_data)})")
            
            if len(batch) < batch_size:
                break
            
            offset += batch_size
            time.sleep(1)
            
        except Exception as e:
            logger.warning(f"  Error at offset {offset}: {e}")
            time.sleep(5)
            continue
    
    if all_data:
        df = pd.DataFrame(all_data)
        out = DRUGCOMB_DIR / "drugcomb_api.csv"
        df.to_csv(out, index=False)
        logger.info(f"  Saved: {out} ({len(df)} records)")
        return df
    
    return None


def process_drugcomb_data(data_path):
    """Process DrugComb data into training format."""
    logger.info(f"Processing DrugComb data from {data_path}...")
    
    df = pd.read_csv(data_path, low_memory=False)
    logger.info(f"  Raw records: {len(df)}")
    logger.info(f"  Columns: {list(df.columns[:20])}")
    
    col_map = {}
    for col in df.columns:
        cl = col.lower()
        if 'drug_row' in cl or 'druga' in cl or 'compound_row' in cl:
            col_map['drug_a'] = col
        elif 'drug_col' in cl or 'drugb' in cl or 'compound_col' in cl:
            col_map['drug_b'] = col
        elif 'cell_line' in cl or 'cellline' in cl:
            col_map['cell_line'] = col
        elif 'synergy_loewe' in cl or 'loewe' in cl:
            col_map['synergy_loewe'] = col
        elif 'synergy_bliss' in cl or 'bliss' in cl:
            col_map['synergy_bliss'] = col
        elif 'css' in cl and 'synergy' not in cl:
            col_map['css'] = col
    
    logger.info(f"  Column mapping: {col_map}")
    
    if 'drug_a' not in col_map or 'drug_b' not in col_map or 'cell_line' not in col_map:
        logger.error("  Missing required columns!")
        logger.info(f"  Available: {list(df.columns)}")
        return None
    
    synergy_col = col_map.get('synergy_loewe', col_map.get('synergy_bliss', col_map.get('css', None)))
    
    if synergy_col is None:
        logger.error("  No synergy score column found!")
        return None
    
    result = pd.DataFrame({
        'drug_a': df[col_map['drug_a']].astype(str),
        'drug_b': df[col_map['drug_b']].astype(str),
        'cell_line': df[col_map['cell_line']].astype(str),
        'synergy_loewe': pd.to_numeric(df[synergy_col], errors='coerce'),
    })
    
    result = result.dropna(subset=['synergy_loewe'])
    result = result[result['drug_a'] != result['drug_b']]
    
    logger.info(f"  Clean records: {len(result)}")
    logger.info(f"  Unique drugs: {result['drug_a'].nunique() + result['drug_b'].nunique()}")
    logger.info(f"  Unique cell lines: {result['cell_line'].nunique()}")
    
    return result


def merge_with_oneil(drugcomb_df):
    """Merge DrugComb with existing O'Neil data."""
    oneil = pd.read_csv(DATA_DIR / "oneil_synergy.csv")
    
    drugcomb_df['drug_a'] = drugcomb_df['drug_a'].str.upper().str.strip()
    drugcomb_df['drug_b'] = drugcomb_df['drug_b'].str.upper().str.strip()
    drugcomb_df['cell_line'] = drugcomb_df['cell_line'].str.upper().str.strip()
    
    oneil['drug_a'] = oneil['drug_a'].str.upper().str.strip()
    oneil['drug_b'] = oneil['drug_b'].str.upper().str.strip()
    oneil['cell_line'] = oneil['cell_line'].str.upper().str.strip()
    
    oneil['source'] = 'oneil'
    drugcomb_df['source'] = 'drugcomb'
    
    np.random.seed(42)
    drugcomb_df['fold'] = np.random.randint(0, 5, len(drugcomb_df))
    
    combined = pd.concat([oneil, drugcomb_df], ignore_index=True)
    combined = combined.drop_duplicates(subset=['drug_a', 'drug_b', 'cell_line'], keep='first')
    
    logger.info(f"  Combined: {len(combined)} records")
    return combined


def main():
    print("=" * 70)
    print("TASK 3: DrugComb 1000K+ Expansion")
    print("=" * 70)
    
    data_path = DRUGCOMB_DIR / "drugcomb_summary.csv"
    api_path = DRUGCOMB_DIR / "drugcomb_api.csv"
    
    if not data_path.exists() and not api_path.exists():
        result = download_drugcomb_csv()
        
        if result is None:
            logger.info("\nCSV download failed, trying API pagination...")
            api_df = download_drugcomb_block()
            if api_df is not None:
                data_path = api_path
    
    actual_path = data_path if data_path.exists() else (api_path if api_path.exists() else None)
    
    if actual_path and actual_path.stat().st_size > 10000:
        dc_df = process_drugcomb_data(actual_path)
        
        if dc_df is not None:
            out = DRUGCOMB_DIR / "drugcomb_processed.csv"
            dc_df.to_csv(out, index=False)
            
            merged = merge_with_oneil(dc_df)
            merged_path = DATA_DIR / "synergy_combined.csv"
            merged.to_csv(merged_path, index=False)
            
            print(f"\n{'='*70}")
            print("EXPANSION SUMMARY")
            print("=" * 70)
            print(f"  O'Neil original:    23,052 records")
            print(f"  DrugComb additions: {len(dc_df):,} records")
            print(f"  Combined total:     {len(merged):,} records")
            print(f"  Saved: {merged_path}")
    else:
        logger.error("Could not download DrugComb data!")
        logger.info("Creating augmented dataset from O'Neil...")
        
        oneil = pd.read_csv(DATA_DIR / "oneil_synergy.csv")
        swapped = oneil.copy()
        swapped['drug_a'], swapped['drug_b'] = oneil['drug_b'].values, oneil['drug_a'].values
        swapped['source'] = 'swap_augment'
        oneil['source'] = 'oneil'
        
        combined = pd.concat([oneil, swapped], ignore_index=True)
        combined = combined.drop_duplicates(subset=['drug_a', 'drug_b', 'cell_line'], keep='first')
        
        merged_path = DATA_DIR / "synergy_combined.csv"
        combined.to_csv(merged_path, index=False)
        print(f"  Augmented: {len(combined)} records → {merged_path}")


if __name__ == "__main__":
    main()
