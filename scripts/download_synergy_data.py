"""
ADDS 시너지 데이터 확장 스크립트
DrugComb + NCI-ALMANAC 데이터를 다운로드하고 전처리합니다.

사용법:
    python scripts/download_synergy_data.py

출력:
    data/synergy/drugcomb_synergy.csv      - DrugComb Loewe scores
    data/synergy/nci_almanac_synergy.csv   - NCI-ALMANAC ComboScores
    data/synergy/merged_synergy_data.csv   - 통합 데이터셋
    data/synergy/data_summary.json         - 데이터 요약 통계
"""

import os
import sys
import json
import time
import logging
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# === Configuration ===
DATA_DIR = Path("F:/ADDS/data/synergy")
DATA_DIR.mkdir(parents=True, exist_ok=True)

EXISTING_DATA = Path("F:/ADDS/models/synergy")

# DrugComb bulk download URLs
DRUGCOMB_URLS = [
    "https://drugcomb.org/download/?file=summary_v_1_5.csv",
    "https://drugcomb.fimm.fi/download/?file=summary_v_1_5.csv",
]

# NCI-ALMANAC download
NCI_ALMANAC_URL = "https://wiki.nci.nih.gov/download/attachments/338237347/ComboDrugGrowth_Nov2017.zip"
NCI_ALMANAC_ALT = "https://dtp.cancer.gov/ncialmanac/NCIalmanac_data.zip"


def download_file(url: str, dest: Path, timeout: int = 120) -> bool:
    """Download a file with progress reporting."""
    import urllib.request
    import urllib.error
    
    logger.info(f"Downloading: {url}")
    logger.info(f"Destination: {dest}")
    
    try:
        req = urllib.request.Request(url, headers={
            "User-Agent": "Mozilla/5.0 (ADDS Research Platform)"
        })
        
        with urllib.request.urlopen(req, timeout=timeout) as response:
            total = response.headers.get("Content-Length")
            total = int(total) if total else None
            
            downloaded = 0
            block_size = 1024 * 1024  # 1MB
            
            with open(dest, "wb") as f:
                while True:
                    chunk = response.read(block_size)
                    if not chunk:
                        break
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total:
                        pct = downloaded / total * 100
                        logger.info(f"  Progress: {downloaded / 1e6:.1f} / {total / 1e6:.1f} MB ({pct:.0f}%)")
                    else:
                        logger.info(f"  Downloaded: {downloaded / 1e6:.1f} MB")
        
        logger.info(f"✓ Download complete: {dest} ({dest.stat().st_size / 1e6:.1f} MB)")
        return True
        
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as e:
        logger.warning(f"✗ Download failed: {e}")
        if dest.exists():
            dest.unlink()
        return False


def download_drugcomb_api(max_pages: int = 500) -> Optional[pd.DataFrame]:
    """
    Download synergy data from DrugComb REST API (paginated).
    Endpoint: https://api.drugcomb.org/combination
    """
    import urllib.request
    import urllib.error
    
    logger.info("=== DrugComb API Download ===")
    all_records = []
    
    base_url = "https://api.drugcomb.org/combination"
    page_size = 1000
    
    for page in range(max_pages):
        url = f"{base_url}?offset={page * page_size}&limit={page_size}"
        try:
            req = urllib.request.Request(url, headers={
                "User-Agent": "ADDS-Research",
                "Accept": "application/json",
            })
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read())
            
            if not data:
                logger.info(f"No more data at page {page}")
                break
            
            all_records.extend(data)
            logger.info(f"  Page {page}: +{len(data)} records (total: {len(all_records)})")
            
            if len(data) < page_size:
                break
            
            time.sleep(0.5)  # Rate limiting
            
        except Exception as e:
            logger.warning(f"  API error at page {page}: {e}")
            break
    
    if all_records:
        df = pd.DataFrame(all_records)
        logger.info(f"✓ DrugComb API: {len(df)} records downloaded")
        return df
    return None


def download_drugcomb_bulk() -> Optional[pd.DataFrame]:
    """Download DrugComb bulk CSV."""
    logger.info("=== DrugComb Bulk Download ===")
    
    csv_path = DATA_DIR / "drugcomb_raw.csv"
    
    # Try each URL
    for url in DRUGCOMB_URLS:
        if download_file(url, csv_path, timeout=180):
            try:
                df = pd.read_csv(csv_path)
                logger.info(f"✓ DrugComb bulk: {len(df)} rows, columns: {list(df.columns[:10])}")
                return df
            except Exception as e:
                logger.warning(f"Failed to parse CSV: {e}")
    
    return None


def download_nci_almanac() -> Optional[pd.DataFrame]:
    """Download NCI-ALMANAC ComboScore data."""
    logger.info("=== NCI-ALMANAC Download ===")
    
    zip_path = DATA_DIR / "nci_almanac_raw.zip"
    csv_path = DATA_DIR / "nci_almanac_raw.csv"
    
    # Try downloading
    for url in [NCI_ALMANAC_URL, NCI_ALMANAC_ALT]:
        if download_file(url, zip_path, timeout=300):
            break
    
    # Extract if zip
    if zip_path.exists():
        try:
            import zipfile
            with zipfile.ZipFile(zip_path, "r") as z:
                csv_files = [f for f in z.namelist() if f.endswith(".csv")]
                if csv_files:
                    z.extract(csv_files[0], DATA_DIR)
                    extracted = DATA_DIR / csv_files[0]
                    if extracted != csv_path:
                        extracted.rename(csv_path)
                    logger.info(f"Extracted: {csv_path}")
        except Exception as e:
            logger.warning(f"Failed to extract zip: {e}")
    
    if csv_path.exists():
        try:
            df = pd.read_csv(csv_path)
            logger.info(f"✓ NCI-ALMANAC: {len(df)} rows")
            return df
        except Exception as e:
            logger.warning(f"Failed to parse CSV: {e}")
    
    return None


def load_existing_oneil_data() -> pd.DataFrame:
    """Load existing O'Neil 23K dataset."""
    labels_path = EXISTING_DATA / "labels.csv"
    if not labels_path.exists():
        # Search for it
        for p in EXISTING_DATA.rglob("labels*.csv"):
            labels_path = p
            break
    
    if labels_path.exists():
        df = pd.read_csv(labels_path)
        logger.info(f"✓ Existing O'Neil data: {len(df)} rows from {labels_path}")
        return df
    
    logger.warning("No existing O'Neil labels.csv found")
    return pd.DataFrame()


def standardize_drugcomb(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize DrugComb data to common format."""
    # Expected columns: drug_row, drug_col, cell_line_name, synergy_loewe, ...
    col_map = {}
    
    # Try to identify columns
    cols_lower = {c.lower(): c for c in df.columns}
    
    for target, candidates in {
        "drug_a": ["drug_row", "drug_a", "drug1", "compound_a"],
        "drug_b": ["drug_col", "drug_b", "drug2", "compound_b"],
        "cell_line": ["cell_line_name", "cell_line", "cellline", "tissue_cell_line"],
        "synergy_score": ["synergy_loewe", "loewe", "synergy_score", "css", "bliss"],
    }.items():
        for cand in candidates:
            if cand in cols_lower:
                col_map[cols_lower[cand]] = target
                break
    
    if len(col_map) < 3:
        logger.warning(f"Could not map DrugComb columns. Found: {list(df.columns[:15])}")
        # Use whatever is available
        if len(df.columns) >= 3:
            col_map = {df.columns[0]: "drug_a", df.columns[1]: "drug_b"}
            if "cell_line_name" in df.columns:
                col_map["cell_line_name"] = "cell_line"
    
    result = df.rename(columns=col_map)
    result["source"] = "drugcomb"
    
    logger.info(f"Standardized DrugComb: {len(result)} rows, {len(result.columns)} cols")
    return result


def standardize_nci_almanac(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize NCI-ALMANAC data to common format."""
    cols_lower = {c.lower(): c for c in df.columns}
    
    col_map = {}
    for target, candidates in {
        "drug_a": ["nsc1", "drug1", "agent1"],
        "drug_b": ["nsc2", "drug2", "agent2"],
        "cell_line": ["cellname", "cell_line", "cellline", "panel/cell line"],
        "synergy_score": ["comboscore", "combo_score", "score"],
    }.items():
        for cand in candidates:
            if cand in cols_lower:
                col_map[cols_lower[cand]] = target
                break
    
    result = df.rename(columns=col_map)
    result["source"] = "nci_almanac"
    
    logger.info(f"Standardized NCI-ALMANAC: {len(result)} rows")
    return result


def generate_morgan_fingerprints(drug_names: List[str], smiles_dict: Dict[str, str]) -> Dict[str, np.ndarray]:
    """Generate Morgan fingerprints for all drugs with known SMILES."""
    fps = {}
    
    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem
        
        for name in drug_names:
            smiles = smiles_dict.get(name)
            if smiles:
                mol = Chem.MolFromSmiles(smiles)
                if mol:
                    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=1024)
                    fps[name] = np.array(fp, dtype=np.float32)
        
        logger.info(f"Generated fingerprints for {len(fps)}/{len(drug_names)} drugs")
    except ImportError:
        logger.warning("RDKit not available — cannot generate fingerprints")
    
    return fps


def build_training_features(
    synergy_df: pd.DataFrame,
    fps: Dict[str, np.ndarray],
) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Build feature matrix (concatenated Morgan FPs) and target vector."""
    
    rows_X = []
    rows_y = []
    rows_meta = []
    
    for _, row in synergy_df.iterrows():
        da = str(row.get("drug_a", ""))
        db = str(row.get("drug_b", ""))
        score = row.get("synergy_score")
        
        if pd.isna(score) or da not in fps or db not in fps:
            continue
        
        fp_a = fps[da]
        fp_b = fps[db]
        x = np.concatenate([fp_a, fp_b])
        
        rows_X.append(x)
        rows_y.append(float(score))
        rows_meta.append({
            "drug_a": da,
            "drug_b": db,
            "cell_line": str(row.get("cell_line", "")),
            "source": str(row.get("source", "")),
        })
    
    X = np.array(rows_X, dtype=np.float32)
    y = np.array(rows_y, dtype=np.float32)
    meta = pd.DataFrame(rows_meta)
    
    logger.info(f"Features built: X={X.shape}, y={y.shape}")
    return X, y, meta


def main():
    logger.info("=" * 60)
    logger.info("ADDS 시너지 데이터 확장 스크립트")
    logger.info("=" * 60)
    
    # Step 1: Load existing data
    existing = load_existing_oneil_data()
    
    # Load existing drug SMILES
    smiles_path = EXISTING_DATA / "drug_smiles.json"
    drug_smiles = {}
    if smiles_path.exists():
        with open(smiles_path) as f:
            drug_smiles = json.load(f)
        logger.info(f"Loaded {len(drug_smiles)} existing drug SMILES")
    
    # Step 2: Download DrugComb
    drugcomb_df = download_drugcomb_bulk()
    if drugcomb_df is None:
        logger.info("Trying DrugComb API (paginated)...")
        drugcomb_df = download_drugcomb_api(max_pages=100)
    
    if drugcomb_df is not None:
        drugcomb_std = standardize_drugcomb(drugcomb_df)
        drugcomb_std.to_csv(DATA_DIR / "drugcomb_synergy.csv", index=False)
        logger.info(f"✓ DrugComb saved: {len(drugcomb_std)} rows")
    else:
        logger.warning("⚠ DrugComb download failed — create manual download instructions")
        drugcomb_std = pd.DataFrame()
        
        # Write manual instructions
        instructions = DATA_DIR / "MANUAL_DOWNLOAD_INSTRUCTIONS.md"
        instructions.write_text(
            "# 수동 다운로드 안내\n\n"
            "## DrugComb\n"
            "1. https://drugcomb.org/download/ 접속\n"
            "2. 'summary' CSV 파일 다운로드\n"
            "3. 이 폴더에 `drugcomb_raw.csv`로 저장\n"
            "4. 이 스크립트 다시 실행\n\n"
            "## NCI-ALMANAC\n"
            "1. https://wiki.nci.nih.gov/display/NCIDTPdata/NCI-ALMANAC 접속\n"
            "2. 'ComboDrugGrowth' CSV 다운로드\n"
            "3. 이 폴더에 `nci_almanac_raw.csv`로 저장\n"
            "4. 이 스크립트 다시 실행\n",
            encoding="utf-8",
        )
        logger.info(f"Manual instructions written to {instructions}")
    
    # Step 3: Download NCI-ALMANAC
    nci_df = download_nci_almanac()
    if nci_df is not None:
        nci_std = standardize_nci_almanac(nci_df)
        nci_std.to_csv(DATA_DIR / "nci_almanac_synergy.csv", index=False)
        logger.info(f"✓ NCI-ALMANAC saved: {len(nci_std)} rows")
    else:
        logger.warning("⚠ NCI-ALMANAC download failed — see manual instructions")
        nci_std = pd.DataFrame()
        
        # Check if user manually placed files
        manual_nci = DATA_DIR / "nci_almanac_raw.csv"
        if manual_nci.exists():
            nci_df = pd.read_csv(manual_nci)
            nci_std = standardize_nci_almanac(nci_df)
            nci_std.to_csv(DATA_DIR / "nci_almanac_synergy.csv", index=False)
            logger.info(f"✓ NCI-ALMANAC from manual file: {len(nci_std)} rows")
    
    # Step 4: Merge all data
    dfs = []
    if len(existing) > 0:
        existing["source"] = "oneil_2016"
        dfs.append(existing)
    if len(drugcomb_std) > 0:
        dfs.append(drugcomb_std)
    if len(nci_std) > 0:
        dfs.append(nci_std)
    
    if dfs:
        merged = pd.concat(dfs, ignore_index=True, sort=False)
        
        # Remove exact duplicates
        key_cols = [c for c in ["drug_a", "drug_b", "cell_line"] if c in merged.columns]
        if key_cols:
            before = len(merged)
            merged = merged.drop_duplicates(subset=key_cols, keep="first")
            logger.info(f"Deduplicated: {before} → {len(merged)} rows")
        
        merged.to_csv(DATA_DIR / "merged_synergy_data.csv", index=False)
        logger.info(f"✓ Merged dataset: {len(merged)} rows")
        
        # Summary statistics
        summary = {
            "total_rows": len(merged),
            "sources": {src: int(cnt) for src, cnt in merged["source"].value_counts().items()} if "source" in merged.columns else {},
            "n_unique_drugs_a": int(merged["drug_a"].nunique()) if "drug_a" in merged.columns else 0,
            "n_unique_drugs_b": int(merged["drug_b"].nunique()) if "drug_b" in merged.columns else 0,
            "n_cell_lines": int(merged["cell_line"].nunique()) if "cell_line" in merged.columns else 0,
        }
        
        if "synergy_score" in merged.columns:
            scores = merged["synergy_score"].dropna()
            summary["synergy_score_stats"] = {
                "mean": float(scores.mean()),
                "std": float(scores.std()),
                "min": float(scores.min()),
                "max": float(scores.max()),
                "n_synergistic": int((scores > 5).sum()),
                "n_antagonistic": int((scores < -5).sum()),
                "n_additive": int(((scores >= -5) & (scores <= 5)).sum()),
            }
        
        with open(DATA_DIR / "data_summary.json", "w") as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"\n{'='*60}")
        logger.info(f"데이터 요약:")
        logger.info(f"  총 행 수: {summary['total_rows']:,}")
        logger.info(f"  소스: {summary.get('sources', {})}")
        logger.info(f"{'='*60}")
        
    else:
        logger.error("No data loaded at all")
        sys.exit(1)
    
    # Step 5: Generate fingerprints for new drugs
    if "drug_a" in merged.columns and "drug_b" in merged.columns:
        all_drugs = set(merged["drug_a"].dropna().unique()) | set(merged["drug_b"].dropna().unique())
        known_drugs = set(drug_smiles.keys())
        new_drugs = all_drugs - known_drugs
        
        logger.info(f"\n약물 현황:")
        logger.info(f"  전체 약물: {len(all_drugs)}")
        logger.info(f"  SMILES 알려진 약물: {len(known_drugs & all_drugs)}")
        logger.info(f"  SMILES 미보유 약물: {len(new_drugs)}")
        
        if new_drugs:
            logger.info(f"  미보유 약물 목록 (상위 20): {sorted(new_drugs)[:20]}")
            
            # Try to get SMILES from PubChem for new drugs
            new_smiles = fetch_pubchem_smiles(list(new_drugs)[:200])
            if new_smiles:
                drug_smiles.update(new_smiles)
                with open(EXISTING_DATA / "drug_smiles_extended.json", "w") as f:
                    json.dump(drug_smiles, f, indent=2)
                logger.info(f"✓ Extended SMILES: {len(drug_smiles)} drugs total")
    
    logger.info("\n✓ 데이터 수집 완료!")
    logger.info(f"  출력 디렉토리: {DATA_DIR}")
    logger.info(f"  다음 단계: python scripts/retrain_synergy.py")


def fetch_pubchem_smiles(drug_names: List[str], batch_size: int = 10) -> Dict[str, str]:
    """Fetch SMILES from PubChem REST API for unknown drugs."""
    import urllib.request
    import urllib.error
    
    logger.info(f"Fetching SMILES from PubChem for {len(drug_names)} drugs...")
    smiles_dict = {}
    
    for i in range(0, len(drug_names), batch_size):
        batch = drug_names[i:i + batch_size]
        for name in batch:
            try:
                url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{name}/property/CanonicalSMILES/JSON"
                req = urllib.request.Request(url, headers={"User-Agent": "ADDS-Research"})
                with urllib.request.urlopen(req, timeout=10) as resp:
                    data = json.loads(resp.read())
                    props = data.get("PropertyTable", {}).get("Properties", [])
                    if props:
                        smiles = props[0].get("CanonicalSMILES")
                        if smiles:
                            smiles_dict[name] = smiles
            except Exception:
                pass  # Skip unknown drugs
        
        if smiles_dict:
            logger.info(f"  PubChem: {len(smiles_dict)} SMILES found so far...")
        time.sleep(0.5)
    
    logger.info(f"✓ PubChem: {len(smiles_dict)} SMILES retrieved")
    return smiles_dict


if __name__ == "__main__":
    main()
