"""
Phase 3: TCGA 전체 RNA-seq + DepMap 전체 매핑 + DeepSynergy DNN

1. TCGA 177명 전체 RNA-seq 다운로드 → 안정화된 AUC
2. DepMap Model.csv via 다양한 소스 → 전체 ACH 매핑
3. DeepSynergy MLP → XGBoost 대비 성능 비교
"""

import json
import logging
import os
import pickle
import time
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import pearsonr, mannwhitneyu
from sklearn.ensemble import (
    GradientBoostingClassifier,
    RandomForestClassifier,
    VotingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DATA_DIR = Path("F:/ADDS/data/ml_training")
MODEL_DIR = Path("F:/ADDS/models")


# ============================================================
# PART 1: TCGA 전체 RNA-seq 다운로드
# ============================================================

def download_all_tcga_rnaseq():
    """Download ALL TCGA RNA-seq for chemo patients."""
    manifest_path = DATA_DIR / "tcga" / "rnaseq_manifest.json"
    rnaseq_dir = DATA_DIR / "tcga" / "rnaseq"
    rnaseq_dir.mkdir(parents=True, exist_ok=True)
    
    with open(manifest_path) as f:
        files = json.load(f)
    
    chemo_csv = DATA_DIR / "tcga" / "tcga_coad_chemo_survival.csv"
    chemo_df = pd.read_csv(chemo_csv)
    chemo_ids = set(chemo_df["case_id"].values)
    
    # Filter for chemo patients
    chemo_files = []
    for fi in files:
        cases = fi.get("cases", [])
        if cases:
            cid = cases[0].get("submitter_id", "")
            if cid in chemo_ids:
                chemo_files.append(fi)
    
    logger.info("TCGA chemo RNA-seq files: %d (for %d patients)", len(chemo_files), len(chemo_ids))
    
    # Download all that we don't have yet
    downloaded = 0
    skipped = 0
    failed = 0
    
    for i, fi in enumerate(chemo_files):
        fid = fi["file_id"]
        fname = fi["file_name"]
        cases = fi.get("cases", [{}])
        cid = cases[0].get("submitter_id", "") if cases else ""
        
        out_path = rnaseq_dir / f"{cid}_{fname}"
        if out_path.exists():
            skipped += 1
            continue
        
        if (i + 1) % 10 == 0 or i == 0:
            logger.info("  Downloading %d/%d: %s...", i + 1, len(chemo_files), cid)
        
        try:
            url = f"https://api.gdc.cancer.gov/data/{fid}"
            req = urllib.request.Request(url)
            with urllib.request.urlopen(req, timeout=60) as resp:
                content = resp.read()
            with open(out_path, "wb") as f:
                f.write(content)
            downloaded += 1
            time.sleep(0.3)
        except Exception as e:
            failed += 1
            if failed <= 3:
                logger.warning("  Failed %s: %s", cid, e)
    
    logger.info("TCGA RNA-seq: downloaded=%d, skipped=%d, failed=%d", downloaded, skipped, failed)
    return downloaded + skipped


# ============================================================
# PART 2: DepMap 전체 매핑 확보
# ============================================================

def get_depmap_full_mapping_v2():
    """Try harder to get full DepMap ACH→name mapping."""
    cache = DATA_DIR / "depmap" / "ach_to_name_full_v2.json"
    if cache.exists():
        with open(cache) as f:
            return json.load(f)
    
    mapping = {}
    
    # Load existing curated mapping
    old_cache = DATA_DIR / "depmap" / "ach_to_name_full.json"
    if old_cache.exists():
        with open(old_cache) as f:
            mapping = json.load(f)
    
    # Strategy: Extract cell line names from expression data column analysis
    # The expression parquet has ACH IDs as index, gene names as columns
    # We can try to match ACH IDs to known cell lines by:
    # 1. Expression profile correlation with known cell lines
    # 2. DepMap sample info CSV download
    
    # Try multiple DepMap sample_info download URLs
    logger.info("Attempting DepMap sample info download...")
    urls = [
        # DepMap 24Q4 / 24Q2 / 23Q4 sample_info URLs
        "https://ndownloader.figshare.com/files/44906718",  # 24Q2 Model.csv
        "https://ndownloader.figshare.com/files/42067636",  # 23Q4
        "https://ndownloader.figshare.com/files/46489594",  # 24Q4
        "https://ndownloader.figshare.com/files/48277087",
        "https://ndownloader.figshare.com/files/39461822",  # 23Q2
        "https://ndownloader.figshare.com/files/35141766",  # 22Q4
        "https://ndownloader.figshare.com/files/34008503",  # 22Q2
        "https://ndownloader.figshare.com/files/30127639",  # 21Q4
        # CellModelPassport
        "https://cog.sanger.ac.uk/cmp/download/model_list_latest.csv",
    ]
    
    for url in urls:
        try:
            logger.info("  Trying %s...", url[:60])
            req = urllib.request.Request(url, headers={"User-Agent": "ADDS-Research/1.0"})
            with urllib.request.urlopen(req, timeout=15) as resp:
                data = resp.read(5_000_000)  # 5MB max
            
            text = data.decode("utf-8", errors="replace")
            lines = text.strip().split("\n")
            header = lines[0]
            
            # Check for known columns
            if any(x in header for x in ["ModelID", "DepMap_ID", "model_id", "ACH-"]):
                logger.info("  Found cell line metadata! Header: %s", header[:200])
                
                # Parse CSV (simple approach)
                cols = header.split(",")
                id_idx = None
                name_idx = None
                
                for ci, col in enumerate(cols):
                    col_clean = col.strip('"').strip()
                    if col_clean in ("ModelID", "DepMap_ID", "model_id"):
                        id_idx = ci
                    if col_clean in ("CellLineName", "cell_line_name", "stripped_cell_line_name",
                                     "CCLEName", "model_name"):
                        name_idx = ci
                
                if id_idx is not None and name_idx is not None:
                    for line in lines[1:]:
                        parts = line.split(",")
                        if len(parts) > max(id_idx, name_idx):
                            ach = parts[id_idx].strip('"').strip()
                            name = parts[name_idx].strip('"').strip()
                            if ach.startswith("ACH-") and name and name != "nan":
                                mapping[ach] = name
                    
                    logger.info("  Extracted %d mappings!", len(mapping))
                    if len(mapping) > 200:
                        break
                else:
                    # Try to find ACH- patterns anywhere
                    for line in lines[1:]:
                        parts = line.split(",")
                        for pi, part in enumerate(parts):
                            part = part.strip('"').strip()
                            if part.startswith("ACH-"):
                                # Next non-ACH field is likely the name
                                if pi + 1 < len(parts):
                                    name = parts[pi + 1].strip('"').strip()
                                    if name and not name.startswith("ACH-"):
                                        mapping[part] = name
                                break
                    if len(mapping) > 200:
                        logger.info("  Pattern-extracted %d mappings", len(mapping))
                        break
                    
        except Exception as e:
            logger.debug("  Failed: %s", e)
            continue
    
    # Approach 2: Build from expression file index + web scraping each ACH
    if len(mapping) < 200:
        logger.info("Using expression file ACH IDs + curated expansion...")
        expr = pd.read_parquet(DATA_DIR / "depmap" / "ccle_expression.parquet")
        
        # For unmapped ACH IDs, we'll use a different strategy:
        # Try to query DepMap API one-by-one for a sample
        unmapped = [ach for ach in expr.index if ach not in mapping][:50]
        
        for ach in unmapped:
            try:
                url = f"https://depmap.org/portal/cell_line/{ach}"
                req = urllib.request.Request(url, headers={"User-Agent": "ADDS-Research/1.0"})
                with urllib.request.urlopen(req, timeout=5) as resp:
                    html = resp.read().decode("utf-8", errors="replace")
                # Extract cell line name from HTML title
                if "<title>" in html:
                    title = html.split("<title>")[1].split("</title>")[0]
                    # Format: "CellLineName - DepMap" or similar
                    name = title.split(" - ")[0].split(" |")[0].strip()
                    if name and name != "DepMap":
                        mapping[ach] = name
            except Exception:
                break  # Rate limited or unavailable
            time.sleep(0.2)
        
        logger.info("  After web scraping: %d total mappings", len(mapping))
    
    # Save
    with open(cache, "w") as f:
        json.dump(mapping, f, indent=2)
    
    logger.info("Final DepMap mapping: %d entries", len(mapping))
    return mapping


# ============================================================
# PART 3: DeepSynergy DNN
# ============================================================

class DeepSynergyMLP(nn.Module):
    """DeepSynergy-style MLP for drug synergy prediction."""
    
    def __init__(self, fp_dim=2048, cl_dim=100, hidden_dims=[512, 256, 128]):
        super().__init__()
        input_dim = fp_dim + cl_dim
        
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers.extend([
                nn.Linear(prev, h),
                nn.LayerNorm(h),
                nn.SiLU(),
                nn.Dropout(0.3),
            ])
            prev = h
        layers.append(nn.Linear(prev, 1))
        
        self.net = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.net(x).squeeze(-1)


def train_deep_synergy(embed_data, dataset="oneil"):
    """Train DeepSynergy MLP and compare with XGBoost."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    embeddings = embed_data["embeddings"]
    emb_dim = embed_data["dim"]
    
    df = pd.read_csv(DATA_DIR / "synergy_combined.csv", low_memory=False)
    
    smiles = {}
    for p in [MODEL_DIR / "synergy" / "drug_smiles.json",
              MODEL_DIR / "synergy" / "drug_smiles_extended.json"]:
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
    
    def norm_cl(n):
        return str(n).upper().replace("-","").replace("_","").replace(" ","").replace(".","")
    
    sub = df[df["source"] == "oneil"] if dataset == "oneil" else df
    
    X_list, y_list = [], []
    for _, row in sub.iterrows():
        da, db = str(row["drug_a"]), str(row["drug_b"])
        if da not in fps or db not in fps:
            continue
        score = float(row["synergy_loewe"])
        if np.isnan(score):
            continue
        fp = np.concatenate([fps[da], fps[db]])
        cl_norm = norm_cl(str(row["cell_line"]))
        cl_feat = embeddings.get(cl_norm, np.zeros(emb_dim, dtype=np.float32))
        X_list.append(np.concatenate([fp, cl_feat]))
        y_list.append(score)
    
    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.float32)
    logger.info("DeepSynergy %s: X=%s", dataset, X.shape)
    
    # Normalize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    dnn_rs = []
    
    for fold, (ti, vi) in enumerate(kf.split(X_scaled)):
        Xt = torch.FloatTensor(X_scaled[ti]).to(device)
        yt = torch.FloatTensor(y[ti]).to(device)
        Xv = torch.FloatTensor(X_scaled[vi]).to(device)
        yv = y[vi]
        
        model = DeepSynergyMLP(fp_dim=2048, cl_dim=emb_dim).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=100)
        
        best_val_r = -1
        patience = 0
        
        for epoch in range(200):
            model.train()
            
            # Mini-batch training
            perm = torch.randperm(len(Xt))
            batch_size = 2048
            epoch_loss = 0
            n_batches = 0
            
            for start in range(0, len(Xt), batch_size):
                idx = perm[start:start + batch_size]
                opt.zero_grad()
                pred = model(Xt[idx])
                loss = F.mse_loss(pred, yt[idx])
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                epoch_loss += loss.item()
                n_batches += 1
            
            sched.step()
            
            # Validate
            if (epoch + 1) % 10 == 0:
                model.eval()
                with torch.no_grad():
                    val_pred = model(Xv).cpu().numpy()
                r = pearsonr(yv, val_pred)[0]
                if r > best_val_r:
                    best_val_r = r
                    patience = 0
                else:
                    patience += 1
                
                if patience >= 5:
                    break
        
        model.eval()
        with torch.no_grad():
            val_pred = model(Xv).cpu().numpy()
        r = pearsonr(yv, val_pred)[0]
        dnn_rs.append(r)
        logger.info("  DNN fold %d: r=%.4f (epoch %d)", fold + 1, r, epoch + 1)
    
    avg_r = np.mean(dnn_rs)
    std_r = np.std(dnn_rs)
    logger.info("  DNN %s: r=%.4f +/- %.4f", dataset, avg_r, std_r)
    
    return {"dnn_r": round(float(avg_r), 4), "dnn_std": round(float(std_r), 4)}


# ============================================================
# PART 4: TCGA 전체 치료반응 모델
# ============================================================

def train_full_tcga_model():
    """Train treatment response with all available TCGA RNA-seq."""
    rnaseq_dir = DATA_DIR / "tcga" / "rnaseq"
    chemo_csv = DATA_DIR / "tcga" / "tcga_coad_chemo_survival.csv"
    
    chemo_df = pd.read_csv(chemo_csv)
    labels = dict(zip(chemo_df["case_id"], chemo_df["label"]))
    
    all_data = {}
    gene_names = None
    
    for fpath in sorted(rnaseq_dir.glob("*.tsv")):
        case_id = fpath.name.split("_")[0]
        if case_id not in labels:
            continue
        try:
            df = pd.read_csv(fpath, sep="\t", comment="#")
            if "gene_name" not in df.columns:
                continue
            
            count_col = None
            for col in ["tpm_unstranded", "fpkm_unstranded", "unstranded"]:
                if col in df.columns:
                    count_col = col
                    break
            if count_col is None:
                continue
            
            if "gene_type" in df.columns:
                df = df[df["gene_type"] == "protein_coding"]
            
            df = df.drop_duplicates(subset=["gene_name"]).set_index("gene_name")
            values = df[count_col].astype(float)
            values = values[~values.index.str.startswith("N_")]
            
            if case_id in all_data:
                all_data[case_id] = (all_data[case_id] + values.reindex(all_data[case_id].index, fill_value=0)) / 2
            else:
                all_data[case_id] = values
                if gene_names is None:
                    gene_names = values.index.tolist()
        except Exception:
            continue
    
    logger.info("Full TCGA: %d patients", len(all_data))
    
    if len(all_data) < 30:
        return {"error": "too few patients", "n": len(all_data)}
    
    common = sorted(set(gene_names).intersection(*[set(v.index) for v in all_data.values()]))
    
    X_data, y_data = [], []
    for cid, vals in all_data.items():
        X_data.append(vals.reindex(common, fill_value=0).values)
        y_data.append(labels[cid])
    
    X = np.log2(np.array(X_data, dtype=np.float32) + 1)
    y = np.array(y_data, dtype=int)
    
    logger.info("Full TCGA: X=%s, resp=%d, nonresp=%d", X.shape, y.sum(), len(y) - y.sum())
    
    # DEG selection
    resp_idx = np.where(y == 1)[0]
    nonr_idx = np.where(y == 0)[0]
    
    pvals = []
    for j in range(X.shape[1]):
        try:
            _, p = mannwhitneyu(X[resp_idx, j], X[nonr_idx, j], alternative="two-sided")
        except Exception:
            p = 1.0
        pvals.append(p)
    
    results = {}
    for n_deg in [50, 100, 200]:
        top_idx = np.argsort(pvals)[:n_deg]
        X_deg = X[:, top_idx]
        
        scaler = StandardScaler()
        X_s = scaler.fit_transform(X_deg)
        
        n_min = min(y.sum(), len(y) - y.sum())
        n_splits = min(5, n_min)
        if n_splits < 2:
            continue
        
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        aucs = []
        for ti, vi in skf.split(X_s, y):
            clf = VotingClassifier(
                estimators=[
                    ("rf", RandomForestClassifier(n_estimators=100, random_state=42, max_depth=5)),
                    ("gb", GradientBoostingClassifier(n_estimators=100, random_state=42, max_depth=3)),
                    ("lr", LogisticRegression(C=0.1, random_state=42, max_iter=1000)),
                ],
                voting="soft",
            )
            clf.fit(X_s[ti], y[ti])
            prob = clf.predict_proba(X_s[vi])[:, 1]
            try:
                auc = roc_auc_score(y[vi], prob)
                aucs.append(auc)
            except Exception:
                pass
        
        if aucs:
            avg = np.mean(aucs)
            std = np.std(aucs)
            logger.info("  TCGA DEG-%d: AUC=%.4f +/- %.4f (n=%d)", n_deg, avg, std, len(all_data))
            results[f"deg{n_deg}"] = {"auc": round(avg, 4), "std": round(std, 4)}
    
    return results


# ============================================================
# Main
# ============================================================

def main():
    logger.info("=" * 60)
    logger.info("PHASE 3: Full TCGA + DepMap Mapping + DeepSynergy")
    logger.info("=" * 60)
    
    results = {}
    
    # Part 1: Download remaining TCGA RNA-seq
    logger.info("\n--- Part 1: TCGA Full RNA-seq Download ---")
    n_tcga = download_all_tcga_rnaseq()
    results["tcga_downloaded"] = n_tcga
    
    # Part 2: DepMap full mapping
    logger.info("\n--- Part 2: DepMap Full Mapping ---")
    mapping = get_depmap_full_mapping_v2()
    results["depmap_mapping"] = len(mapping)
    
    # Part 3: DeepSynergy DNN
    logger.info("\n--- Part 3: DeepSynergy DNN ---")
    embed_path = DATA_DIR / "depmap" / "cellline_embedding.pkl"
    if embed_path.exists():
        with open(embed_path, "rb") as f:
            embed_data = pickle.load(f)
        
        # Add new mappings
        embeddings = embed_data["embeddings"]
        for ach, name in mapping.items():
            if ach in embeddings:
                norm = name.upper().replace("-","").replace("_","").replace(" ","").replace(".","")
                embeddings[norm] = embeddings[ach]
        embed_data["embeddings"] = embeddings
        
        dnn_results = train_deep_synergy(embed_data, "oneil")
        results["deep_synergy_oneil"] = dnn_results
    
    # Part 4: Full TCGA treatment model
    logger.info("\n--- Part 4: Full TCGA Treatment Response ---")
    tcga_results = train_full_tcga_model()
    results["tcga_full"] = tcga_results
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("PHASE 3 RESULTS")
    logger.info("=" * 60)
    logger.info(json.dumps(results, indent=2, default=str))
    
    with open(MODEL_DIR / "phase3_results.json", "w") as f:
        json.dump(results, f, indent=2, default=float)
    logger.info("\nSaved: phase3_results.json")


if __name__ == "__main__":
    main()
