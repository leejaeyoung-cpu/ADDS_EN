"""
PPI 네트워크 토폴로지 피처 추가 + 시너지 모델 재학습

STRING DB에서 약물 타겟 PPI 네트워크를 다운로드하고,
네트워크 topological features를 추출하여 시너지 예측에 추가합니다.

참고 논문: DeepSynergyTF (2024) - PPI topology로 +11% 개선 보고

피처:
1. Betweenness centrality (drug target nodes)
2. Closeness centrality
3. Degree centrality
4. Clustering coefficient
5. PageRank
6. Shortest path length between drug A/B targets

사용법:
    python scripts/synergy_ppi_features.py
"""

import json
import logging
import pickle
import time
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DATA_DIR = Path("F:/ADDS/data/ml_training")
MODEL_DIR = Path("F:/ADDS/models/synergy")

# =======================================================================
# 1. Drug → Target mapping (from ChEMBL/DrugBank common targets)
# =======================================================================

# Curated drug-target mappings for common anticancer drugs
# This covers the most frequently used drugs in DrugComb CRC data
DRUG_TARGETS = {
    # EGFR inhibitors
    "Erlotinib": ["EGFR"], "Gefitinib": ["EGFR"], "Lapatinib": ["EGFR", "ERBB2"],
    "Cetuximab": ["EGFR"], "Panitumumab": ["EGFR"], "Afatinib": ["EGFR", "ERBB2", "ERBB4"],
    # VEGF/angiogenesis
    "Bevacizumab": ["VEGFA"], "Sorafenib": ["BRAF", "VEGFR2", "KIT", "FLT3"],
    "Sunitinib": ["VEGFR2", "PDGFRA", "KIT", "FLT3"], "Axitinib": ["VEGFR1", "VEGFR2", "VEGFR3"],
    # BRAF/MEK/RAS
    "Vemurafenib": ["BRAF"], "Dabrafenib": ["BRAF"], "Trametinib": ["MAP2K1", "MAP2K2"],
    "Cobimetinib": ["MAP2K1"], "Binimetinib": ["MAP2K1", "MAP2K2"],
    # mTOR/PI3K
    "Everolimus": ["MTOR"], "Temsirolimus": ["MTOR"], "Alpelisib": ["PIK3CA"],
    # CDK
    "Palbociclib": ["CDK4", "CDK6"], "Ribociclib": ["CDK4", "CDK6"],
    # Immunotherapy
    "Nivolumab": ["PDCD1"], "Pembrolizumab": ["PDCD1"], "Ipilimumab": ["CTLA4"],
    # Chemotherapy
    "5-Fluorouracil": ["TYMS"], "Oxaliplatin": ["DNA"], "Irinotecan": ["TOP1"],
    "Cisplatin": ["DNA"], "Carboplatin": ["DNA"], "Gemcitabine": ["RRM1"],
    "Doxorubicin": ["TOP2A"], "Paclitaxel": ["TUBB"], "Docetaxel": ["TUBB"],
    "Vincristine": ["TUBB"], "Etoposide": ["TOP2A"], "Topotecan": ["TOP1"],
    # PARP
    "Olaparib": ["PARP1", "PARP2"], "Niraparib": ["PARP1", "PARP2"],
    # HER2
    "Trastuzumab": ["ERBB2"], "Pertuzumab": ["ERBB2"],
    # Multi-kinase
    "Regorafenib": ["BRAF", "VEGFR2", "KIT", "TIE2", "PDGFRA"],
    "Lenvatinib": ["VEGFR1", "VEGFR2", "FGFR1", "KIT"],
    # HDAC
    "Vorinostat": ["HDAC1", "HDAC2", "HDAC3"], "Panobinostat": ["HDAC1", "HDAC2", "HDAC3"],
    # Proteasome
    "Bortezomib": ["PSMB5"], "Carfilzomib": ["PSMB5"],
    # BCL2
    "Venetoclax": ["BCL2"],
    # JAK
    "Ruxolitinib": ["JAK1", "JAK2"],
    # MET
    "Crizotinib": ["ALK", "MET"], "Capmatinib": ["MET"],
}

# All unique targets
ALL_TARGETS = sorted(set(t for targets in DRUG_TARGETS.values() for t in targets))
logger.info(f"Drug-target map: {len(DRUG_TARGETS)} drugs, {len(ALL_TARGETS)} unique targets")


# =======================================================================
# 2. STRING DB PPI Network
# =======================================================================

def download_string_ppi(species: int = 9606, score_threshold: int = 700) -> Optional[pd.DataFrame]:
    """Download human PPI from STRING DB API."""
    import urllib.request

    logger.info(f"Downloading STRING PPI (species={species}, score>={score_threshold})...")

    # Get PPI for all our targets
    targets_str = "%0d".join(ALL_TARGETS)

    # STRING API: network interactions
    url = f"https://string-db.org/api/tsv/network?identifiers={targets_str}&species={species}&required_score={score_threshold}&limit=0"

    ppi_path = DATA_DIR / "string_ppi.tsv"

    try:
        req = urllib.request.Request(url, headers={"User-Agent": "ADDS-Research"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = resp.read()
        with open(ppi_path, "wb") as f:
            f.write(data)

        df = pd.read_csv(ppi_path, sep="\t")
        logger.info(f"  STRING PPI: {len(df)} interactions")
        return df
    except Exception as e:
        logger.warning(f"  STRING API failed: {e}")
        return None


def build_ppi_graph(ppi_df: Optional[pd.DataFrame] = None) -> "nx.Graph":
    """Build PPI graph from STRING data or fallback curated network."""
    import networkx as nx

    G = nx.Graph()

    if ppi_df is not None and len(ppi_df) > 0:
        # Use STRING data
        for _, row in ppi_df.iterrows():
            a = str(row.get("preferredName_A", row.get("stringId_A", "")))
            b = str(row.get("preferredName_B", row.get("stringId_B", "")))
            score = float(row.get("score", row.get("combined_score", 0.5)))
            G.add_edge(a, b, weight=score)
        logger.info(f"  PPI graph from STRING: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    else:
        # Fallback: curated cancer signaling PPI
        logger.info("  Building curated cancer signaling PPI...")
        curated_edges = [
            # EGFR signaling
            ("EGFR", "ERBB2", 0.95), ("EGFR", "GRB2", 0.90), ("GRB2", "SOS1", 0.85),
            ("SOS1", "KRAS", 0.90), ("KRAS", "BRAF", 0.95), ("BRAF", "MAP2K1", 0.95),
            ("MAP2K1", "MAPK1", 0.95), ("MAP2K1", "MAPK3", 0.90),
            # PI3K/AKT/mTOR
            ("EGFR", "PIK3CA", 0.85), ("PIK3CA", "AKT1", 0.90), ("AKT1", "MTOR", 0.85),
            ("PTEN", "PIK3CA", 0.80), ("AKT1", "BAD", 0.75),
            # JAK/STAT
            ("JAK1", "STAT3", 0.90), ("JAK2", "STAT3", 0.85), ("STAT3", "BCL2", 0.70),
            # Apoptosis
            ("BCL2", "BAX", 0.80), ("TP53", "BAX", 0.85), ("TP53", "CDKN1A", 0.90),
            # Cell cycle
            ("CDK4", "RB1", 0.90), ("CDK6", "RB1", 0.85),
            ("CDKN1A", "CDK4", 0.80), ("CDKN1A", "CDK6", 0.75),
            # DNA repair
            ("PARP1", "BRCA1", 0.80), ("PARP2", "BRCA1", 0.75),
            # RTK → RAS connections
            ("MET", "GRB2", 0.85), ("ERBB2", "GRB2", 0.80),
            ("VEGFR2", "PIK3CA", 0.70), ("PDGFRA", "PIK3CA", 0.65),
            # WNT signaling
            ("APC", "CTNNB1", 0.90), ("CTNNB1", "TCF7L2", 0.80),
            # Additional interactions
            ("KRAS", "PIK3CA", 0.75), ("MTOR", "RPS6KB1", 0.80),
            ("FLT3", "STAT5A", 0.75), ("KIT", "PIK3CA", 0.65),
            ("TOP1", "TP53", 0.60), ("TOP2A", "TP53", 0.60),
            ("TYMS", "DHFR", 0.85), ("RRM1", "RRM2", 0.90),
            ("ALK", "GRB2", 0.70), ("PDCD1", "CD274", 0.95),
            ("CTLA4", "CD80", 0.90),
            ("HDAC1", "HDAC2", 0.90), ("HDAC2", "HDAC3", 0.85),
            ("PSMB5", "PSMA1", 0.80),
        ]
        for a, b, w in curated_edges:
            G.add_edge(a, b, weight=w)

        # Also add all targets as nodes (even if isolated)
        for t in ALL_TARGETS:
            if t not in G:
                G.add_node(t)

        logger.info(f"  Curated PPI: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    return G


# =======================================================================
# 3. Topological Feature Extraction
# =======================================================================

def compute_node_features(G) -> Dict[str, Dict[str, float]]:
    """Compute topological features for each node."""
    import networkx as nx

    logger.info("Computing node topological features...")

    bc = nx.betweenness_centrality(G, weight="weight")
    cc = nx.closeness_centrality(G)
    dc = nx.degree_centrality(G)
    pr = nx.pagerank(G, weight="weight")

    clustering = {}
    for n in G.nodes():
        try:
            clustering[n] = nx.clustering(G, n, weight="weight")
        except:
            clustering[n] = 0.0

    features = {}
    for node in G.nodes():
        features[node] = {
            "betweenness": bc.get(node, 0),
            "closeness": cc.get(node, 0),
            "degree": dc.get(node, 0),
            "pagerank": pr.get(node, 0),
            "clustering": clustering.get(node, 0),
        }

    return features


def get_drug_ppi_features(drug_name: str, node_features: Dict, G) -> np.ndarray:
    """Get PPI features for a drug based on its targets."""
    import networkx as nx

    targets = DRUG_TARGETS.get(drug_name, [])
    if not targets:
        return np.zeros(11, dtype=np.float32)  # 5 node feats + 5 agg + 1 path

    # Per-target features (aggregate over targets)
    feat_arrays = []
    for t in targets:
        if t in node_features:
            nf = node_features[t]
            feat_arrays.append([nf["betweenness"], nf["closeness"], nf["degree"],
                                nf["pagerank"], nf["clustering"]])

    if not feat_arrays:
        return np.zeros(11, dtype=np.float32)

    feats = np.array(feat_arrays, dtype=np.float32)

    # Mean and max pool
    mean_feats = feats.mean(axis=0)  # 5
    max_feats = feats.max(axis=0)    # 5

    # Number of targets (degree indicator)
    n_targets = len(targets)

    return np.concatenate([mean_feats, max_feats, [n_targets]])


def get_pair_ppi_features(drug_a: str, drug_b: str, node_features: Dict, G) -> np.ndarray:
    """Get PPI features for a drug pair."""
    import networkx as nx

    feat_a = get_drug_ppi_features(drug_a, node_features, G)
    feat_b = get_drug_ppi_features(drug_b, node_features, G)

    # Shortest path between any target pair
    targets_a = DRUG_TARGETS.get(drug_a, [])
    targets_b = DRUG_TARGETS.get(drug_b, [])

    min_path = 99
    for ta in targets_a:
        for tb in targets_b:
            if ta in G and tb in G:
                try:
                    sp = nx.shortest_path_length(G, ta, tb)
                    min_path = min(min_path, sp)
                except nx.NetworkXNoPath:
                    pass

    # Common neighbors between target sets
    common_neighbors = 0
    for ta in targets_a:
        for tb in targets_b:
            if ta in G and tb in G:
                common_neighbors += len(set(G.neighbors(ta)) & set(G.neighbors(tb)))

    pair_feats = np.array([
        min_path if min_path < 99 else -1,  # -1 = no path
        common_neighbors,
    ], dtype=np.float32)

    return np.concatenate([feat_a, feat_b, pair_feats])  # 11 + 11 + 2 = 24


# =======================================================================
# 4. Build Enhanced Features + Retrain
# =======================================================================

def build_enhanced_features():
    """Build Morgan FP + PPI topological features."""
    import networkx as nx

    # Load synergy data
    df = pd.read_csv(DATA_DIR / "synergy_combined.csv", low_memory=False)
    logger.info(f"Synergy data: {len(df)} rows")

    # Load SMILES
    smiles = {}
    for p in [MODEL_DIR / "drug_smiles.json", MODEL_DIR / "drug_smiles_extended.json"]:
        if p.exists():
            with open(p) as f:
                smiles.update(json.load(f))
    logger.info(f"SMILES: {len(smiles)} drugs")

    # Build PPI graph
    ppi_df = download_string_ppi()
    G = build_ppi_graph(ppi_df)
    node_features = compute_node_features(G)

    # Compute Morgan FPs
    from rdkit import Chem
    from rdkit.Chem import AllChem

    fps = {}
    for name, smi in smiles.items():
        mol = Chem.MolFromSmiles(smi)
        if mol:
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=1024)
            fps[name] = np.array(fp, dtype=np.float32)
    logger.info(f"Fingerprints: {len(fps)} drugs")

    # Build feature matrix
    X_list = []
    y_list = []
    meta_list = []
    ppi_known = set(DRUG_TARGETS.keys())

    for _, row in df.iterrows():
        da, db = str(row["drug_a"]), str(row["drug_b"])

        if da not in fps or db not in fps:
            continue

        score = float(row["synergy_loewe"])
        if np.isnan(score):
            continue

        # Morgan FP features (2048)
        fp_concat = np.concatenate([fps[da], fps[db]])

        # PPI features (24) — only if drug has known targets
        ppi_feats = get_pair_ppi_features(da, db, node_features, G)

        x = np.concatenate([fp_concat, ppi_feats])
        X_list.append(x)
        y_list.append(score)
        meta_list.append({"drug_a": da, "drug_b": db, "cell_line": str(row.get("cell_line", "")),
                          "has_ppi_a": da in ppi_known, "has_ppi_b": db in ppi_known})

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.float32)
    meta = pd.DataFrame(meta_list)

    logger.info(f"Enhanced features: X={X.shape} (2048 FP + 24 PPI), y={y.shape}")
    n_with_ppi = meta["has_ppi_a"].sum() + meta["has_ppi_b"].sum()
    logger.info(f"  Samples with PPI info: {n_with_ppi}/{len(meta)*2} drug instances")

    return X, y, meta, G, node_features


def train_xgboost_enhanced(X, y, test_size=0.2):
    """Train XGBoost with enhanced features."""
    import xgboost as xgb
    from sklearn.model_selection import train_test_split
    from scipy.stats import pearsonr, spearmanr
    from sklearn.metrics import mean_squared_error

    idx_train, idx_val = train_test_split(np.arange(len(X)), test_size=test_size, random_state=42)
    X_train, X_val = X[idx_train], X[idx_val]
    y_train, y_val = y[idx_train], y[idx_val]

    logger.info(f"Training XGBoost with PPI features: train={len(X_train)}, val={len(X_val)}")

    model = xgb.XGBRegressor(
        n_estimators=500, max_depth=8, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
        reg_alpha=0.1, reg_lambda=1.0,
        tree_method="hist", device="cuda:0",
        n_jobs=-1, random_state=42, early_stopping_rounds=30,
    )

    t0 = time.time()
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=50)
    train_time = time.time() - t0

    y_pred_val = model.predict(X_val)
    y_pred_train = model.predict(X_train)

    results = {
        "train_pearson_r": float(pearsonr(y_train, y_pred_train)[0]),
        "val_pearson_r": float(pearsonr(y_val, y_pred_val)[0]),
        "val_spearman_r": float(spearmanr(y_val, y_pred_val)[0]),
        "val_rmse": float(np.sqrt(mean_squared_error(y_val, y_pred_val))),
        "train_time_sec": round(train_time, 1),
        "n_train": len(X_train),
        "n_val": len(X_val),
        "n_features": X.shape[1],
        "ppi_features": 24,
        "fp_features": 2048,
    }

    logger.info(f"XGBoost+PPI: val r={results['val_pearson_r']:.4f}, RMSE={results['val_rmse']:.2f}")
    return model, results


def main():
    logger.info("=" * 60)
    logger.info("시너지 모델 + PPI 네트워크 피처")
    logger.info("=" * 60)

    X, y, meta, G, node_features = build_enhanced_features()

    # Train with all features (FP + PPI)
    model_full, results_full = train_xgboost_enhanced(X, y)

    # Train FP-only baseline for comparison
    X_fp_only = X[:, :2048]
    _, results_fp = train_xgboost_enhanced(X_fp_only, y)

    # Save
    with open(MODEL_DIR / "xgboost_synergy_v5_ppi.pkl", "wb") as f:
        pickle.dump(model_full, f)

    results_cmp = {
        "v4_fp_only": results_fp,
        "v5_fp_ppi": results_full,
        "improvement": {
            "val_r_delta": round(results_full["val_pearson_r"] - results_fp["val_pearson_r"], 4),
            "val_rmse_delta": round(results_full["val_rmse"] - results_fp["val_rmse"], 2),
        },
        "ppi_graph": {"n_nodes": G.number_of_nodes(), "n_edges": G.number_of_edges()},
    }

    with open(MODEL_DIR / "ppi_retrain_results.json", "w") as f:
        json.dump(results_cmp, f, indent=2)

    # Save PPI graph info
    with open(DATA_DIR / "ppi_node_features.json", "w") as f:
        json.dump(node_features, f, indent=2, default=float)

    logger.info(f"\n{'='*60}")
    logger.info(f"FP-only:  val r={results_fp['val_pearson_r']:.4f}, RMSE={results_fp['val_rmse']:.2f}")
    logger.info(f"FP+PPI:   val r={results_full['val_pearson_r']:.4f}, RMSE={results_full['val_rmse']:.2f}")
    logger.info(f"Δr = {results_cmp['improvement']['val_r_delta']:+.4f}")
    logger.info(f"{'='*60}")


if __name__ == "__main__":
    main()
