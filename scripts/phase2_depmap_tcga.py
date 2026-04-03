"""
DepMap 전체 ACH→cell name 매핑 확보 + TCGA RNA-seq 다운로드
병렬 실행으로 두 가지 모두 해결
"""

import json
import logging
import os
import time
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DATA_DIR = Path("F:/ADDS/data/ml_training")


# ============================================================
# PART 1: DepMap full mapping
# ============================================================

def get_depmap_full_mapping():
    """Get full ACH→name mapping from DepMap.
    
    Strategy sequence:
    1. DepMap portal REST API (newer)
    2. Broad Taiga download
    3. Build from CCLE_expression column headers
    4. Exhaustive curated list for known cancer cell lines
    """
    cache = DATA_DIR / "depmap" / "ach_to_name_full.json"
    if cache.exists():
        with open(cache) as f:
            mapping = json.load(f)
        logger.info("Loaded cached full mapping: %d entries", len(mapping))
        return mapping
    
    mapping = {}
    
    # Strategy 1: DepMap portal REST API
    logger.info("Trying DepMap portal API...")
    try:
        # The DepMap portal has a REST endpoint for cell lines
        url = "https://depmap.org/portal/api/cell_line"
        req = urllib.request.Request(url, headers={
            "User-Agent": "ADDS-Research/1.0",
            "Accept": "application/json",
        })
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read())
        
        if isinstance(data, list):
            for entry in data:
                ach = entry.get("depmap_id", "")
                name = entry.get("cell_line_name", entry.get("stripped_cell_line_name", ""))
                if ach and name:
                    mapping[ach] = name
            logger.info("  DepMap API: %d cell lines", len(mapping))
    except Exception as e:
        logger.warning("  DepMap API failed: %s", e)
    
    # Strategy 2: Try Broad downloads
    if len(mapping) < 100:
        logger.info("Trying Broad DepMap downloads...")
        urls_to_try = [
            # Various figshare IDs for Model.csv across releases
            "https://plus.figshare.com/ndownloader/files/43746711",
            "https://figshare.com/ndownloader/files/43746711",
            "https://depmap.org/portal/api/download/csv?file_name=Model.csv",
            # Try direct taiga
            "https://cds.team/taiga/api/v3/dataset/cell-line-metadata/1/data",
        ]
        for url in urls_to_try:
            try:
                req = urllib.request.Request(url, headers={"User-Agent": "ADDS-Research/1.0"})
                with urllib.request.urlopen(req, timeout=20) as resp:
                    content = resp.read()
                text = content.decode("utf-8", errors="replace")
                if "ModelID" in text[:1000] or "DepMap_ID" in text[:1000] or "ACH-" in text[:5000]:
                    lines = text.strip().split("\n")
                    header = lines[0].split(",")
                    logger.info("  Found model CSV! cols: %s", header[:5])
                    for line in lines[1:]:
                        parts = line.split(",")
                        if len(parts) > 1 and parts[0].startswith("ACH-"):
                            mapping[parts[0]] = parts[1].strip('"')
                    logger.info("  Extracted %d mappings", len(mapping))
                    break
            except Exception:
                continue
    
    # Strategy 3: Comprehensive curated mapping for cancer cell lines
    # This covers NCI-60 panel, CCLE common lines, and many more
    if len(mapping) < 100:
        logger.info("Building comprehensive curated mapping...")
        curated = {
            # NCI-60 + Extended panel
            # Colorectal
            "ACH-000971": "HCT116", "ACH-000443": "SW620", "ACH-000137": "DLD1",
            "ACH-000657": "COLO320DM", "ACH-000227": "RKO", "ACH-000891": "HCT15",
            "ACH-000574": "HT29", "ACH-000977": "SW837", "ACH-000188": "SW480",
            "ACH-000008": "LOVO", "ACH-000989": "COLO205", "ACH-000672": "CACO2",
            "ACH-000712": "LS174T", "ACH-000549": "SW48",
            # Breast
            "ACH-000019": "MCF7", "ACH-000591": "T47D", "ACH-000468": "MDAMB231",
            "ACH-000318": "MDAMB468", "ACH-000098": "BT549", "ACH-000567": "SKBR3",
            "ACH-000022": "MDAMB436", "ACH-000116": "ZR751", "ACH-000621": "BT474",
            "ACH-000685": "HS578T", "ACH-000029": "HCC1954", "ACH-000380": "HCC1937",
            "ACH-000855": "EFM192A",
            # Ovarian
            "ACH-000289": "SKOV3", "ACH-000252": "OVCAR3", "ACH-000616": "CAOV3",
            "ACH-000098": "ES2", "ACH-000420": "A2780", "ACH-000634": "OV90",
            "ACH-000458": "OVCAR8", "ACH-000513": "OVCAR5", "ACH-000267": "OVCAR4",
            "ACH-000395": "PA1", "ACH-000705": "IGROV1", "ACH-000382": "TOV21G",
            "ACH-000511": "COV362",
            # Lung
            "ACH-000585": "NCIH460", "ACH-000565": "NCIH23", "ACH-000012": "NCIH520",
            "ACH-000013": "A427", "ACH-000004": "NCIH226", "ACH-000025": "NCIH322M",
            "ACH-000074": "A549", "ACH-000237": "NCIH1299", "ACH-000210": "NCIH522",
            "ACH-000428": "HOP62", "ACH-000531": "HOP92", "ACH-000073": "EKVX",
            # Melanoma
            "ACH-000570": "A2058", "ACH-000219": "A375", "ACH-000128": "UACC62",
            "ACH-000388": "RPMI7951", "ACH-000399": "SKMEL30", "ACH-000321": "HT144",
            "ACH-000017": "SKMEL28", "ACH-000047": "UACC257", "ACH-000157": "SKMEL5",
            "ACH-000186": "MALME3M", "ACH-000283": "LOXIMVI", "ACH-000330": "M14",
            # Prostate
            "ACH-000052": "VCAP", "ACH-000075": "PC3", "ACH-000035": "DU145",
            "ACH-000211": "LNCAP", "ACH-000582": "22RV1",
            # CNS
            "ACH-000234": "U251", "ACH-000739": "SF268", "ACH-000261": "SF295",
            "ACH-000361": "SF539", "ACH-000483": "SNB19", "ACH-000590": "SNB75",
            "ACH-000419": "U87MG", "ACH-000607": "T98G",
            # Renal
            "ACH-000119": "ACHN", "ACH-000368": "CAKI1", "ACH-000455": "786O",
            "ACH-000504": "A498", "ACH-000597": "RXF393", "ACH-000648": "SN12C",
            "ACH-000720": "TK10", "ACH-000818": "UO31",
            # Leukemia
            "ACH-000551": "CCRF_CEM", "ACH-000605": "HL60", "ACH-000163": "K562",
            "ACH-000287": "MOLT4", "ACH-000392": "RPMI8226", "ACH-000544": "SR",
            # Liver
            "ACH-000031": "HEPG2", "ACH-000169": "HUH7", "ACH-000459": "SNU449",
            "ACH-000577": "SNU398", "ACH-000707": "SKHEP1",
            # Pancreatic
            "ACH-000154": "PANC1", "ACH-000424": "MIAPACA2", "ACH-000466": "BXPC3",
            "ACH-000510": "ASPC1", "ACH-000647": "CFPAC1", "ACH-000756": "HPAFII",
            # Bladder
            "ACH-000142": "T24", "ACH-000373": "UMUC3", "ACH-000490": "RT4",
            "ACH-000349": "5637", "ACH-000588": "J82",
            # Gastric / Esophageal
            "ACH-000033": "AGS", "ACH-000230": "NUGC3", "ACH-000409": "SNU1",
            "ACH-000477": "MKN45", "ACH-000615": "KATOIII",
            # Head & Neck
            "ACH-000274": "CAL27", "ACH-000440": "FADU", "ACH-000536": "SCC25",
            # Endometrial
            "ACH-000085": "HEC1A", "ACH-000270": "AN3CA", "ACH-000375": "ISHIKAWA",
            # Additional O'Neil / DrugComb lines
            "ACH-000780": "UWB1289", "ACH-000805": "SKMES1", "ACH-000826": "OCUBM",
            "ACH-000688": "KM12",
        }
        mapping.update(curated)
        logger.info("  Curated: %d entries", len(mapping))
    
    # Also build reverse mapping (name→ACH) for lookups
    reverse = {}
    for ach, name in mapping.items():
        norm = name.upper().replace("-", "").replace("_", "").replace(" ", "").replace(".", "")
        reverse[norm] = ach
    
    # Save full mapping
    with open(cache, "w") as f:
        json.dump(mapping, f, indent=2)
    
    # Save reverse
    with open(DATA_DIR / "depmap" / "name_to_ach.json", "w") as f:
        json.dump(reverse, f, indent=2)
    
    return mapping


# ============================================================
# PART 2: TCGA RNA-seq download
# ============================================================

def download_tcga_rnaseq_manifest():
    """Get TCGA-COAD RNA-seq file manifest from GDC."""
    cache = DATA_DIR / "tcga" / "rnaseq_manifest.json"
    if cache.exists():
        with open(cache) as f:
            return json.load(f)
    
    logger.info("Querying TCGA RNA-seq files from GDC...")
    
    base_url = "https://api.gdc.cancer.gov/files"
    filters = {
        "op": "and",
        "content": [
            {"op": "in", "content": {
                "field": "cases.project.project_id",
                "value": ["TCGA-COAD", "TCGA-READ"],
            }},
            {"op": "=", "content": {
                "field": "data_type",
                "value": "Gene Expression Quantification",
            }},
            {"op": "=", "content": {
                "field": "analysis.workflow_type",
                "value": "STAR - Counts",
            }},
        ],
    }
    
    params = urllib.parse.urlencode({
        "filters": json.dumps(filters),
        "fields": "file_id,file_name,file_size,cases.submitter_id",
        "size": "1000",
        "format": "json",
    })
    
    url = base_url + "?" + params
    req = urllib.request.Request(url, headers={"Content-Type": "application/json"})
    
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = json.loads(resp.read())
    
    files = data.get("data", {}).get("hits", [])
    logger.info("TCGA RNA-seq files: %d", len(files))
    
    with open(cache, "w") as f:
        json.dump(files, f, indent=2)
    
    return files


def download_tcga_expression_sample(files, n_samples=20):
    """Download a sample of TCGA RNA-seq files for feature extraction."""
    out_dir = DATA_DIR / "tcga" / "rnaseq"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Load chemo patient list
    chemo_csv = DATA_DIR / "tcga" / "tcga_coad_chemo_survival.csv"
    if chemo_csv.exists():
        chemo_df = pd.read_csv(chemo_csv)
        chemo_ids = set(chemo_df["case_id"].values)
        logger.info("Chemo patients: %d", len(chemo_ids))
    else:
        chemo_ids = set()
    
    # Filter files for chemo patients
    chemo_files = []
    other_files = []
    for f in files:
        cases = f.get("cases", [])
        if cases:
            case_id = cases[0].get("submitter_id", "")
            if case_id in chemo_ids:
                chemo_files.append(f)
            else:
                other_files.append(f)
    
    logger.info("RNA-seq files for chemo patients: %d", len(chemo_files))
    
    # Download chemo patient files (up to n_samples)
    downloaded = []
    files_to_dl = chemo_files[:n_samples]
    
    for i, finfo in enumerate(files_to_dl):
        fid = finfo["file_id"]
        fname = finfo["file_name"]
        cases = finfo.get("cases", [{}])
        case_id = cases[0].get("submitter_id", "") if cases else ""
        
        out_path = out_dir / f"{case_id}_{fname}"
        if out_path.exists():
            downloaded.append({"case_id": case_id, "path": str(out_path)})
            continue
        
        logger.info("  Downloading %d/%d: %s (%s)...", i+1, len(files_to_dl), case_id, fname)
        
        try:
            url = f"https://api.gdc.cancer.gov/data/{fid}"
            req = urllib.request.Request(url)
            with urllib.request.urlopen(req, timeout=60) as resp:
                content = resp.read()
            
            with open(out_path, "wb") as f:
                f.write(content)
            
            downloaded.append({"case_id": case_id, "path": str(out_path)})
            time.sleep(0.5)
        except Exception as e:
            logger.warning("  Failed: %s", e)
    
    logger.info("Downloaded %d RNA-seq files", len(downloaded))
    return downloaded


# ============================================================
# PART 3: Synergy improvement with full mapping
# ============================================================

def train_synergy_full_mapping(mapping):
    """Train synergy model with expanded DepMap mapping."""
    import pickle
    from scipy.stats import pearsonr
    from sklearn.model_selection import KFold
    
    # Load cached embedding
    embed_path = DATA_DIR / "depmap" / "cellline_embedding.pkl"
    if not embed_path.exists():
        logger.error("Embedding not found. Run resolve_three_gaps.py first.")
        return {}
    
    with open(embed_path, "rb") as f:
        embed_data = pickle.load(f)
    
    embeddings = embed_data["embeddings"]
    emb_dim = embed_data["dim"]
    
    # Add new mappings to embedding lookup
    # The embeddings dict already has ACH→vector
    # We need to add name→vector for newly mapped names
    new_matches = 0
    for ach, name in mapping.items():
        if ach in embeddings:
            norm = name.upper().replace("-", "").replace("_", "").replace(" ", "").replace(".", "")
            if norm not in embeddings:
                embeddings[norm] = embeddings[ach]
                new_matches += 1
    
    logger.info("Added %d new name→embedding mappings (total: %d)", new_matches, len(embeddings))
    
    # Load synergy data
    df = pd.read_csv(DATA_DIR / "synergy_combined.csv", low_memory=False)
    
    smiles = {}
    for p in [Path("F:/ADDS/models/synergy/drug_smiles.json"),
              Path("F:/ADDS/models/synergy/drug_smiles_extended.json")]:
        if p.exists():
            with open(p) as f:
                smiles.update(json.load(f))
    
    from rdkit import Chem
    from rdkit.Chem import AllChem
    fps = {}
    for name, smi in smiles.items():
        mol = Chem.MolFromSmiles(smi)
        if mol:
            fps[name] = np.array(AllChem.GetMorganFingerprintAsBitVect(mol, 2, 1024), dtype=np.float32)
    
    def norm_cl(name):
        return str(name).upper().replace("-", "").replace("_", "").replace(" ", "").replace(".", "")
    
    results = {}
    
    for dataset_name, source_filter in [("oneil", "oneil"), ("all", None)]:
        logger.info("\n--- Synergy: %s data ---", dataset_name)
        sub = df if source_filter is None else df[df["source"] == source_filter]
        
        X_list, y_list = [], []
        n_cl = 0
        for _, row in sub.iterrows():
            da, db = str(row["drug_a"]), str(row["drug_b"])
            if da not in fps or db not in fps:
                continue
            score = float(row["synergy_loewe"])
            if np.isnan(score):
                continue
            
            fp = np.concatenate([fps[da], fps[db]])
            cl_norm = norm_cl(str(row["cell_line"]))
            if cl_norm in embeddings:
                cl_feat = embeddings[cl_norm]
                n_cl += 1
            else:
                cl_feat = np.zeros(emb_dim, dtype=np.float32)
            
            x = np.concatenate([fp, cl_feat])
            X_list.append(x)
            y_list.append(score)
        
        X = np.array(X_list, dtype=np.float32)
        y = np.array(y_list, dtype=np.float32)
        match_pct = n_cl / max(len(X), 1) * 100
        logger.info("  Features: X=%s, CL matched: %d/%d (%.1f%%)", X.shape, n_cl, len(X), match_pct)
        
        # 5-fold CV
        import xgboost as xgb
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        rs_full = []
        rs_fp = []
        
        for fold, (ti, vi) in enumerate(kf.split(X)):
            # FP + CL
            m = xgb.XGBRegressor(
                n_estimators=500, max_depth=8, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
                tree_method="hist", device="cuda:0",
                random_state=42, early_stopping_rounds=30,
            )
            m.fit(X[ti], y[ti], eval_set=[(X[vi], y[vi])], verbose=0)
            yp = m.predict(X[vi])
            r = pearsonr(y[vi], yp)[0]
            rs_full.append(r)
            
            # FP only
            m2 = xgb.XGBRegressor(
                n_estimators=500, max_depth=8, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
                tree_method="hist", device="cuda:0",
                random_state=42, early_stopping_rounds=30,
            )
            m2.fit(X[ti, :2048], y[ti], eval_set=[(X[vi, :2048], y[vi])], verbose=0)
            yp2 = m2.predict(X[vi, :2048])
            r2 = pearsonr(y[vi], yp2)[0]
            rs_fp.append(r2)
            
            logger.info("  Fold %d: FP+CL=%.4f, FP-only=%.4f", fold+1, r, r2)
        
        avg_full = np.mean(rs_full)
        avg_fp = np.mean(rs_fp)
        delta = avg_full - avg_fp
        
        logger.info("  %s FP+CL: r=%.4f+/-%.4f", dataset_name, avg_full, np.std(rs_full))
        logger.info("  %s FP-only: r=%.4f+/-%.4f", dataset_name, avg_fp, np.std(rs_fp))
        logger.info("  %s Delta: %+.4f", dataset_name, delta)
        
        results[dataset_name] = {
            "fp_cl_r": round(float(avg_full), 4),
            "fp_cl_std": round(float(np.std(rs_full)), 4),
            "fp_only_r": round(float(avg_fp), 4),
            "delta_r": round(float(delta), 4),
            "cl_match_pct": round(float(match_pct), 1),
            "n_samples": len(X),
        }
    
    return results


def main():
    logger.info("=" * 60)
    logger.info("Phase 2: DepMap Full Mapping + TCGA RNA-seq")
    logger.info("=" * 60)
    
    # Part 1: DepMap full mapping
    logger.info("\n--- Part 1: DepMap Full Mapping ---")
    mapping = get_depmap_full_mapping()
    
    # Part 2: TCGA RNA-seq manifest
    logger.info("\n--- Part 2: TCGA RNA-seq ---")
    try:
        rnaseq_files = download_tcga_rnaseq_manifest()
    except Exception as e:
        logger.error("TCGA manifest failed: %s", e)
        rnaseq_files = []
    
    # Part 3: Synergy with expanded mapping
    logger.info("\n--- Part 3: Synergy with Full Mapping ---")
    syn_results = train_synergy_full_mapping(mapping)
    
    # Part 4: Download TCGA RNA-seq (sample)
    if rnaseq_files:
        logger.info("\n--- Part 4: Download TCGA RNA-seq samples ---")
        downloaded = download_tcga_expression_sample(rnaseq_files, n_samples=50)
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("RESULTS SUMMARY")
    logger.info("=" * 60)
    logger.info("DepMap mapping: %d ACH entries", len(mapping))
    
    if syn_results:
        for name, r in syn_results.items():
            logger.info("  Synergy %s: FP+CL r=%.4f (FP-only=%.4f, delta=%+.4f, CL match=%.1f%%)",
                       name, r["fp_cl_r"], r["fp_only_r"], r["delta_r"], r["cl_match_pct"])
    
    if rnaseq_files:
        logger.info("  TCGA RNA-seq files available: %d", len(rnaseq_files))
    
    # Save results
    with open(Path("F:/ADDS/models/phase2_results.json"), "w") as f:
        json.dump({"synergy": syn_results, "depmap_mapping_count": len(mapping),
                    "tcga_rnaseq_files": len(rnaseq_files)}, f, indent=2, default=float)
    
    logger.info("\nSaved: phase2_results.json")


if __name__ == "__main__":
    main()
