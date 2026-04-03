"""
Energy Landscape Drug Synergy Prediction System
=================================================

Signal pathway → chemical reaction network → Eyring ΔG‡ energy landscape
→ drug perturbation → Dijkstra shortest path → synergy prediction

Components:
1. Pathway Energy Graph (Eyring equation: kcat → ΔG‡)
2. Drug Target Perturbation Engine
3. Minimum Energy Path Synergy Predictor
4. O'Neil Drug-pair Holdout Evaluation
"""

import json
import logging
import math
import pickle
from pathlib import Path
from collections import defaultdict

import networkx as nx
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import pearsonr
from sklearn.model_selection import KFold, GroupKFold
from sklearn.preprocessing import StandardScaler

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DATA_DIR = Path("F:/ADDS/data/ml_training")
MODEL_DIR = Path("F:/ADDS/models")
device = "cuda" if torch.cuda.is_available() else "cpu"


# ============================================================
# Constants
# ============================================================

R = 1.987e-3  # kcal/mol/K (gas constant)
T = 310.15    # K (37°C body temperature)
RT = R * T    # ~0.616 kcal/mol

kB = 3.2996e-24  # kcal/K (Boltzmann constant)
h  = 1.5837e-34  # kcal·s (Planck constant)
kBT_over_h = kB * T / h  # ~6.45e12 s⁻¹ (Eyring prefactor)


def kcat_to_dG(kcat):
    """Eyring equation: kcat → ΔG‡ (kcal/mol)."""
    return -RT * math.log(kcat / kBT_over_h)


def dG_to_kcat(dG):
    """Inverse Eyring: ΔG‡ → kcat (s⁻¹)."""
    return kBT_over_h * math.exp(-dG / RT)


# ============================================================
# PART 1: Signaling Pathway Energy Graph
# ============================================================

def build_pathway_graph():
    """
    Build EGFR signaling pathway as directed weighted graph.
    
    Nodes = signaling proteins/states
    Edges = enzymatic reactions with kcat → ΔG‡ weights
    
    Three output phenotypes:
      - Proliferation (ERK → transcription)
      - Survival (mTOR → translation)
      - Migration (FAK → cytoskeleton)
    """
    
    # Literature-curated kcat values (s⁻¹) and their sources
    reactions = [
        # (source, target, kcat, reaction_name, reference)
        # MAPK cascade
        ("EGFR", "RAS",  1.0,   "SOS-catalyzed GTP exchange",     "Sondermann 2004"),
        ("RAS",  "RAF",  2.0,   "RAS-GTP → RAF recruitment",      "Fetics 2015"),
        ("RAF",  "MEK",  5.0,   "RAF phosphorylates MEK",         "Roskoski 2012"),
        ("MEK",  "ERK",  10.0,  "MEK phosphorylates ERK",         "Roskoski 2012"),
        ("ERK",  "PROLIFERATION", 1.0, "ERK → transcription factors", "Yoon 2006"),
        
        # PI3K/AKT/mTOR survival pathway
        ("EGFR", "PI3K",  1.5,   "EGFR activates PI3K",            "Burke 2012"),
        ("PI3K", "AKT",   3.0,   "PIP3 → AKT phosphorylation",    "Manning 2007"),
        ("AKT",  "MTOR",  2.0,   "AKT phosphorylates mTOR",       "Saxton 2017"),
        ("MTOR", "SURVIVAL", 1.0, "mTOR → translation/survival",  "Saxton 2017"),
        
        # FAK migration pathway (direct)
        ("EGFR", "FAK",  5.0,   "RTK → FAK direct activation",    "Schlaepfer 1999"),
        ("FAK",  "MIGRATION", 8.0, "FAK → paxillin/migration",   "Parsons 2003"),
        
        # Crosstalk edges
        ("ERK",  "FAK",  0.1,   "ERK → FAK crosstalk",            "Hunger-Glaser 2003"),
        ("AKT",  "RAF",  0.5,   "AKT -| RAF (negative crosstalk)", "Zimmermann 1999"),
        ("RAS",  "PI3K", 1.2,   "RAS-GTP → PI3K direct",           "Rodriguez-Viciana 1994"),
        
        # RAS inactivation (important for mutation modeling)
        ("RAS",  "RAS_GDP", 19.0, "GAP-mediated GTP hydrolysis",  "Scheffzek 1997"),
        
        # EGFR autophosphorylation (rate-limiting)
        ("STIMULUS", "EGFR", 0.16, "EGF → EGFR autophosphorylation", "Jura 2011"),
        
        # Additional regulatory edges
        ("ERK",  "SOS_FEEDBACK", 0.8, "ERK → SOS negative feedback", "Douville 1992"),
        ("MTOR", "PI3K_FEEDBACK", 0.3, "S6K → IRS1 negative feedback", "Harrington 2005"),
    ]
    
    G = nx.DiGraph()
    
    for src, tgt, kcat, rxn_name, ref in reactions:
        dG = kcat_to_dG(kcat)
        G.add_edge(src, tgt, 
                   kcat=kcat, 
                   dG=dG, 
                   weight=dG,  # Dijkstra uses this
                   rxn_name=rxn_name,
                   reference=ref)
    
    # Log the graph
    logger.info("Pathway graph: %d nodes, %d edges", G.number_of_nodes(), G.number_of_edges())
    for src, tgt, data in G.edges(data=True):
        logger.info("  %s → %s: kcat=%.2f s⁻¹, ΔG‡=%.2f kcal/mol (%s)", 
                     src, tgt, data['kcat'], data['dG'], data['rxn_name'])
    
    return G


# ============================================================
# PART 2: Drug Target Perturbation Engine  
# ============================================================

# O'Neil drug → pathway target mapping (curated from DrugBank + literature)
DRUG_TARGETS = {
    # EGFR pathway
    "Erlotinib":    [("EGFR",  50.0)],   # IC50-based potency factor
    "Gefitinib":    [("EGFR",  30.0)],
    "Lapatinib":    [("EGFR",  40.0)],   # dual EGFR/HER2
    
    # RAS pathway (indirect — no direct RAS inhibitors in O'Neil)
    
    # RAF inhibitors
    "Sorafenib":    [("RAF",   20.0), ("EGFR", 5.0)],  # multi-kinase
    "Vemurafenib":  [("RAF",   80.0)],  # selective BRAF
    
    # MEK inhibitors
    "Trametinib":   [("MEK",   100.0)],  # very potent MEK
    
    # PI3K/AKT/mTOR pathway
    "BEZ-235":      [("PI3K",  60.0), ("MTOR", 40.0)],  # dual PI3K/mTOR
    "MK-2206":      [("AKT",   70.0)],
    "Everolimus":   [("MTOR",  50.0)],
    "Rapamycin":    [("MTOR",  60.0)],
    "MK-8669":      [("MTOR",  40.0)],  # ridaforolimus
    
    # CDK/cell cycle (affects proliferation output)
    "Palbociclib":  [("ERK",   15.0)],  # CDK4/6 → downstream of ERK
    "MK-1775":      [("ERK",   10.0)],  # WEE1 → cell cycle checkpoint
    "AZD1775":      [("ERK",   10.0)],  # WEE1
    
    # DNA damage (increases "cost" of proliferation pathway)
    "5-FU":         [("PROLIFERATION", 25.0)],   # thymidylate synthase
    "Gemcitabine":  [("PROLIFERATION", 30.0)],   # ribonucleotide reductase
    "Cisplatin":    [("PROLIFERATION", 35.0)],
    "Carboplatin":  [("PROLIFERATION", 25.0)],
    "Oxaliplatin":  [("PROLIFERATION", 30.0)],
    "Doxorubicin":  [("PROLIFERATION", 35.0)],   # TOP2A
    "Etoposide":    [("PROLIFERATION", 20.0)],
    "Topotecan":    [("PROLIFERATION", 25.0)],   # TOP1
    "Temozolomide": [("PROLIFERATION", 20.0)],
    "Methotrexate": [("PROLIFERATION", 25.0)],   # DHFR
    
    # Tubulin (affects migration AND proliferation)
    "Paclitaxel":   [("MIGRATION",    30.0), ("PROLIFERATION", 15.0)],
    "Vinblastine":  [("MIGRATION",    25.0), ("PROLIFERATION", 10.0)],
    "Vinorelbine":  [("MIGRATION",    25.0), ("PROLIFERATION", 10.0)],
    
    # PARP inhibitors (synthetic lethality with DNA damage)
    "ABT-888":      [("PROLIFERATION", 15.0)],   # PARP → DNA repair
    
    # Proteasome (survival pathway disruption)
    "Bortezomib":   [("SURVIVAL",  40.0), ("PROLIFERATION", 20.0)],
    
    # HDAC inhibitors (epigenetic → multiple pathways)
    "Vorinostat":   [("PROLIFERATION", 15.0), ("SURVIVAL", 10.0)],
    
    # SRC/multi-kinase
    "Dasatinib":    [("FAK",   30.0), ("RAF", 10.0)],  # SRC/ABL affects FAK
    
    # Hormonal
    "Tamoxifen":    [("PROLIFERATION", 20.0)],  # ER → proliferation
    
    # CHK1 checkpoint
    "MK-8776":      [("PROLIFERATION", 20.0)],   # CHK1
    
    # BRD4
    "MK-4541":      [("PROLIFERATION", 15.0), ("SURVIVAL", 10.0)],
    
    # Others in O'Neil
    "Imatinib":     [("RAF", 15.0), ("EGFR", 5.0)],  # BCR-ABL/KIT
    "Sunitinib":    [("EGFR", 10.0), ("RAF", 10.0)],  # multi-kinase
    "Cyclophosphamide": [("PROLIFERATION", 20.0)],
}

# Phenotype output nodes
PHENOTYPES = ["PROLIFERATION", "SURVIVAL", "MIGRATION"]


def perturb_graph(G_base, drug_name, potency_scale=1.0):
    """
    Apply drug perturbation to pathway graph.
    
    Drug blocks target → increases ΔG‡ of outgoing edges from target node.
    
    ΔG‡_new = ΔG‡_base + RT × ln(1 + potency_factor × scale)
    """
    G = G_base.copy()
    
    if drug_name not in DRUG_TARGETS:
        return G
    
    for target_node, potency in DRUG_TARGETS[drug_name]:
        # Increase barrier of all edges FROM the target node
        # (blocking the node makes its downstream reactions harder)
        barrier_increase = RT * math.log(1 + potency * potency_scale)
        
        if target_node in G:
            for _, successor, data in G.edges(target_node, data=True):
                data['weight'] = data['dG'] + barrier_increase
                data['dG_perturbed'] = data['dG'] + barrier_increase
        
        # Also increase barrier of edges TO phenotype nodes (for DNA damage drugs)
        if target_node in PHENOTYPES:
            for pred, _, data in G.in_edges(target_node, data=True):
                data['weight'] = data['dG'] + barrier_increase
                data['dG_perturbed'] = data['dG'] + barrier_increase
    
    return G


def apply_cellline_modulation(G_base, gene_expression, gene_to_node):
    """
    Modulate kcat based on cell-line gene expression.
    
    Higher expression → higher effective kcat → lower ΔG‡
    kcat_eff = kcat_base × (expr / median_expr)
    """
    G = G_base.copy()
    
    for node in G.nodes():
        if node in gene_to_node:
            genes = gene_to_node[node]
            expr_values = []
            for g in genes:
                if g in gene_expression:
                    expr_values.append(gene_expression[g])
            
            if expr_values:
                # Average expression relative to 1.0 (already normalized)
                expr_factor = max(0.1, np.mean(expr_values))
                
                for _, succ, data in G.edges(node, data=True):
                    kcat_mod = data['kcat'] * expr_factor
                    dG_mod = kcat_to_dG(max(kcat_mod, 1e-6))
                    data['weight'] = dG_mod
                    data['kcat_mod'] = kcat_mod
    
    return G


# Node → gene mapping for expression modulation
NODE_TO_GENES = {
    "EGFR":  ["EGFR"],
    "RAS":   ["KRAS", "NRAS", "HRAS"],
    "RAF":   ["BRAF", "RAF1"],
    "MEK":   ["MAP2K1", "MAP2K2"],
    "ERK":   ["MAPK1", "MAPK3"],
    "PI3K":  ["PIK3CA", "PIK3CB"],
    "AKT":   ["AKT1", "AKT2"],
    "MTOR":  ["MTOR"],
    "FAK":   ["PTK2"],
}


# ============================================================
# PART 3: Synergy Predictor (Dijkstra-based)
# ============================================================

def compute_min_path_energy(G, source="STIMULUS", targets=None):
    """
    Compute minimum energy path from source to each phenotype target.
    Uses Dijkstra with ΔG‡ as edge weights.
    
    Returns dict: {phenotype: total_ΔG‡}
    """
    if targets is None:
        targets = PHENOTYPES
    
    results = {}
    for target in targets:
        try:
            path_length = nx.dijkstra_path_length(G, source, target, weight='weight')
            path_nodes = nx.dijkstra_path(G, source, target, weight='weight')
            results[target] = {
                'total_dG': path_length,
                'path': path_nodes,
                'n_steps': len(path_nodes) - 1,
            }
        except nx.NetworkXNoPath:
            results[target] = {
                'total_dG': 999.0,  # Effectively blocked
                'path': [],
                'n_steps': 0,
            }
    
    return results


def predict_synergy_energy(G_base, drug_a, drug_b, potency_a=1.0, potency_b=1.0):
    """
    Predict synergy score for drug combination using energy landscape.
    
    Synergy = (Δpath_AB) - (Δpath_A + Δpath_B - Δpath_none)
    
    Positive = synergistic (combination blocks more than expected)
    Negative = antagonistic
    """
    # Baseline: no drugs
    paths_none = compute_min_path_energy(G_base)
    
    # Drug A only
    G_a = perturb_graph(G_base, drug_a, potency_a)
    paths_a = compute_min_path_energy(G_a)
    
    # Drug B only
    G_b = perturb_graph(G_base, drug_b, potency_b)
    paths_b = compute_min_path_energy(G_b)
    
    # Combination A+B
    G_ab = perturb_graph(G_base, drug_a, potency_a)
    G_ab = perturb_graph(G_ab, drug_b, potency_b)
    paths_ab = compute_min_path_energy(G_ab)
    
    # Calculate synergy for each phenotype
    synergy_scores = {}
    for pheno in PHENOTYPES:
        dG_none = paths_none[pheno]['total_dG']
        dG_a = paths_a[pheno]['total_dG']
        dG_b = paths_b[pheno]['total_dG']
        dG_ab = paths_ab[pheno]['total_dG']
        
        # Energy increase from each drug alone
        delta_a = dG_a - dG_none
        delta_b = dG_b - dG_none
        delta_ab = dG_ab - dG_none
        
        # Synergy = supra-additive blocking
        # If AB blocks more than A+B individually → positive synergy
        synergy = delta_ab - (delta_a + delta_b)
        
        synergy_scores[pheno] = {
            'synergy': synergy,
            'delta_a': delta_a,
            'delta_b': delta_b,
            'delta_ab': delta_ab,
            'path_none': paths_none[pheno]['path'],
            'path_ab': paths_ab[pheno]['path'],
        }
    
    # Weighted average across phenotypes (proliferation is most important for Loewe)
    weights = {"PROLIFERATION": 0.5, "SURVIVAL": 0.3, "MIGRATION": 0.2}
    total_synergy = sum(
        synergy_scores[p]['synergy'] * weights.get(p, 0.33)
        for p in PHENOTYPES
    )
    
    return {
        'total_synergy': total_synergy,
        'per_phenotype': synergy_scores,
        'delta_ab_total': sum(synergy_scores[p]['delta_ab'] for p in PHENOTYPES),
    }


def extract_energy_features(G_base, drug_a, drug_b, potency_a=1.0, potency_b=1.0):
    """
    Extract rich feature vector from energy landscape for ML model.
    
    Returns: 1D feature vector capturing energy landscape perturbation.
    """
    result = predict_synergy_energy(G_base, drug_a, drug_b, potency_a, potency_b)
    
    features = []
    
    # Per-phenotype features (3 × 4 = 12 features)
    for pheno in PHENOTYPES:
        ps = result['per_phenotype'][pheno]
        features.extend([
            ps['synergy'],
            ps['delta_a'],
            ps['delta_b'],
            ps['delta_ab'],
        ])
    
    # Global features (3 features)
    features.append(result['total_synergy'])
    features.append(result['delta_ab_total'])
    
    # Pathway blocking indicators (3 features)
    for pheno in PHENOTYPES:
        ps = result['per_phenotype'][pheno]
        features.append(1.0 if ps['delta_ab'] > 5.0 else 0.0)  # Effectively blocked?
    
    # Drug interaction type features (2 features)
    # Same pathway? Complementary?
    targets_a = set(t for t, _ in DRUG_TARGETS.get(drug_a, []))
    targets_b = set(t for t, _ in DRUG_TARGETS.get(drug_b, []))
    overlap = len(targets_a & targets_b)
    total = len(targets_a | targets_b)
    features.append(float(overlap))
    features.append(float(total))
    
    return np.array(features, dtype=np.float32)


# ============================================================
# PART 4: O'Neil Integration & Evaluation
# ============================================================

class HybridSynergyMLP(nn.Module):
    """Hybrid model combining energy features + molecular fingerprints."""
    
    def __init__(self, energy_dim, fp_dim=0, cl_dim=0, hidden=[256, 128, 64]):
        super().__init__()
        total_dim = energy_dim + fp_dim + cl_dim
        layers = []
        prev = total_dim
        for h in hidden:
            layers.extend([nn.Linear(prev, h), nn.LayerNorm(h), nn.SiLU(), nn.Dropout(0.3)])
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.net(x).squeeze(-1)


def build_oneil_energy_dataset(G_base, embed_data=None):
    """Build O'Neil dataset with energy landscape features."""
    
    df = pd.read_csv(DATA_DIR / "synergy_combined.csv", low_memory=False)
    df = df[df["source"] == "oneil"]
    
    # Optional: molecular fingerprints
    fps = None
    smiles = {}
    for p in [MODEL_DIR / "synergy" / "drug_smiles.json",
              MODEL_DIR / "synergy" / "drug_smiles_extended.json"]:
        if p.exists():
            with open(p) as f:
                smiles.update(json.load(f))
    
    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem
        fps = {}
        for name, smi in smiles.items():
            mol = Chem.MolFromSmiles(smi)
            if mol:
                fps[name] = np.array(AllChem.GetMorganFingerprintAsBitVect(mol, 2, 1024), dtype=np.float32)
    except ImportError:
        logger.warning("RDKit not available, using energy features only")
    
    # Cell-line embeddings
    embeddings = None
    emb_dim = 0
    if embed_data:
        embeddings = embed_data["embeddings"]
        emb_dim = embed_data["dim"]
    
    def norm_cl(n):
        return str(n).upper().replace("-","").replace("_","").replace(" ","").replace(".","")
    
    # Build dataset
    X_energy, X_fp, X_cl = [], [], []
    y_list, groups = [], []
    n_mapped = 0
    
    # Cache energy features per drug pair (same across cell lines in base graph)
    energy_cache = {}
    
    for _, row in df.iterrows():
        da, db = str(row["drug_a"]), str(row["drug_b"])
        score = float(row["synergy_loewe"])
        if np.isnan(score):
            continue
        
        # Need both drugs mapped OR at least fingerprints
        has_energy = da in DRUG_TARGETS and db in DRUG_TARGETS
        has_fp = fps and da in fps and db in fps
        
        if not has_fp:
            continue
        
        # Energy features
        pair_key = tuple(sorted([da, db]))
        if pair_key not in energy_cache:
            if has_energy:
                energy_feat = extract_energy_features(G_base, da, db)
                energy_cache[pair_key] = energy_feat
                n_mapped += 1
            else:
                energy_cache[pair_key] = np.zeros(19, dtype=np.float32)
        
        energy_feat = energy_cache[pair_key]
        X_energy.append(energy_feat)
        
        # Fingerprint features
        if fps and da in fps and db in fps:
            X_fp.append(np.concatenate([fps[da], fps[db]]))
        else:
            X_fp.append(np.zeros(2048, dtype=np.float32))
        
        # Cell-line embedding
        if embeddings:
            cl = norm_cl(str(row["cell_line"]))
            cl_feat = embeddings.get(cl, np.zeros(emb_dim, dtype=np.float32))
            X_cl.append(cl_feat)
        
        y_list.append(score)
        groups.append(pair_key)
    
    X_energy = np.array(X_energy, dtype=np.float32)
    X_fp = np.array(X_fp, dtype=np.float32)
    y = np.array(y_list, dtype=np.float32)
    
    unique_pairs = list(set(groups))
    pair_to_id = {p: i for i, p in enumerate(unique_pairs)}
    group_ids = np.array([pair_to_id[g] for g in groups])
    
    energy_mapped = sum(1 for g in groups if g in energy_cache and energy_cache[g].sum() != 0)
    
    logger.info("Dataset: %d samples, %d unique pairs", len(y), len(unique_pairs))
    logger.info("Energy-mapped: %d/%d pairs (%.1f%%)", 
                len([k for k,v in energy_cache.items() if v.sum() != 0]),
                len(unique_pairs),
                len([k for k,v in energy_cache.items() if v.sum() != 0]) / max(len(unique_pairs), 1) * 100)
    
    # Combine features
    if embeddings:
        X_cl = np.array(X_cl, dtype=np.float32)
        X_combined = np.concatenate([X_energy, X_fp, X_cl], axis=1)
    else:
        X_combined = np.concatenate([X_energy, X_fp], axis=1)
    
    return {
        'X_combined': X_combined,
        'X_energy': X_energy,
        'X_fp': X_fp,
        'y': y,
        'groups': group_ids,
        'energy_dim': X_energy.shape[1],
        'fp_dim': X_fp.shape[1],
        'cl_dim': emb_dim,
    }


def train_and_evaluate(X, y, groups, input_dim, n_epochs=200, label=""):
    """Train DNN with both random and drug-pair CV."""
    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)
    
    results = {}
    
    for cv_name, kf_gen in [("random", KFold(n_splits=5, shuffle=True, random_state=42)),
                             ("drug_pair", GroupKFold(n_splits=5))]:
        rs = []
        splits = kf_gen.split(X_s, y, groups) if cv_name == "drug_pair" else kf_gen.split(X_s)
        
        for fold, (ti, vi) in enumerate(splits):
            Xt = torch.FloatTensor(X_s[ti]).to(device)
            yt = torch.FloatTensor(y[ti]).to(device)
            Xv = torch.FloatTensor(X_s[vi]).to(device)
            
            model = HybridSynergyMLP(input_dim).to(device)
            opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_epochs)
            
            best_r = -1
            best_state = None
            patience = 0
            
            for epoch in range(n_epochs):
                model.train()
                perm = torch.randperm(len(Xt))
                for start in range(0, len(Xt), 2048):
                    idx = perm[start:start+2048]
                    opt.zero_grad()
                    loss = F.mse_loss(model(Xt[idx]), yt[idx])
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    opt.step()
                sched.step()
                
                if (epoch+1) % 10 == 0:
                    model.eval()
                    with torch.no_grad():
                        vp = model(Xv).cpu().numpy()
                    r = pearsonr(y[vi], vp)[0]
                    if r > best_r:
                        best_r = r
                        patience = 0
                        best_state = {k:v.clone() for k,v in model.state_dict().items()}
                    else:
                        patience += 1
                    if patience >= 5:
                        break
            
            if best_state:
                model.load_state_dict(best_state)
            model.eval()
            with torch.no_grad():
                vp = model(Xv).cpu().numpy()
            r = pearsonr(y[vi], vp)[0]
            rs.append(r)
            logger.info("  %s %s fold %d: r=%.4f", label, cv_name, fold+1, r)
        
        avg = np.mean(rs)
        std = np.std(rs)
        logger.info("  %s %s: r=%.4f +/- %.4f", label, cv_name, avg, std)
        results[cv_name] = {"r": round(float(avg), 4), "std": round(float(std), 4)}
    
    return results


# ============================================================
# Main
# ============================================================

def main():
    logger.info("=" * 60)
    logger.info("ENERGY LANDSCAPE SYNERGY PREDICTION SYSTEM")
    logger.info("=" * 60)
    
    all_results = {}
    
    # === Part 1: Build pathway graph ===
    logger.info("\n" + "=" * 40)
    logger.info("PART 1: Pathway Energy Graph")
    logger.info("=" * 40)
    G = build_pathway_graph()
    
    # Show baseline minimum energy paths
    logger.info("\n--- Baseline Minimum Energy Paths ---")
    paths = compute_min_path_energy(G)
    for pheno, info in paths.items():
        logger.info("  %s: ΔG‡=%.1f kcal/mol, steps=%d, path=%s",
                     pheno, info['total_dG'], info['n_steps'], "→".join(info['path']))
    
    # === Part 2: Drug perturbation examples ===
    logger.info("\n" + "=" * 40)
    logger.info("PART 2: Drug Perturbation Examples")
    logger.info("=" * 40)
    
    # Example: RAF inhibitor (Vemurafenib)
    result = predict_synergy_energy(G, "Vemurafenib", "Trametinib")
    logger.info("\nVemurafenib + Trametinib (RAF+MEK, same pathway):")
    for p in PHENOTYPES:
        ps = result['per_phenotype'][p]
        logger.info("  %s: synergy=%.2f, Δa=%.2f, Δb=%.2f, Δab=%.2f",
                     p, ps['synergy'], ps['delta_a'], ps['delta_b'], ps['delta_ab'])
    logger.info("  Total synergy: %.3f", result['total_synergy'])
    
    # Example: Complementary pathway (ERK + PI3K)
    result2 = predict_synergy_energy(G, "Trametinib", "BEZ-235")
    logger.info("\nTrametinib + BEZ-235 (MEK + PI3K/mTOR, complementary):")
    for p in PHENOTYPES:
        ps = result2['per_phenotype'][p]
        logger.info("  %s: synergy=%.2f, Δa=%.2f, Δb=%.2f, Δab=%.2f",
                     p, ps['synergy'], ps['delta_a'], ps['delta_b'], ps['delta_ab'])
    logger.info("  Total synergy: %.3f", result2['total_synergy'])
    
    # Example: FAK + DNA damage
    result3 = predict_synergy_energy(G, "Dasatinib", "Cisplatin")
    logger.info("\nDasatinib + Cisplatin (FAK + DNA damage):")
    for p in PHENOTYPES:
        ps = result3['per_phenotype'][p]
        logger.info("  %s: synergy=%.2f, Δa=%.2f, Δb=%.2f, Δab=%.2f",
                     p, ps['synergy'], ps['delta_a'], ps['delta_b'], ps['delta_ab'])
    logger.info("  Total synergy: %.3f", result3['total_synergy'])
    
    # === Part 3: Build O'Neil dataset ===
    logger.info("\n" + "=" * 40)
    logger.info("PART 3: O'Neil Energy Features")
    logger.info("=" * 40)
    
    # Load embeddings
    embed_path = DATA_DIR / "depmap" / "cellline_embedding_v2.pkl"
    embed_data = None
    if embed_path.exists():
        with open(embed_path, "rb") as f:
            embed_data = pickle.load(f)
    
    data = build_oneil_energy_dataset(G, embed_data)
    
    # === Part 4: Evaluate different feature combinations ===
    logger.info("\n" + "=" * 40)
    logger.info("PART 4: Model Evaluation")
    logger.info("=" * 40)
    
    # 4a: Energy features ONLY
    logger.info("\n--- Energy features only ---")
    r_energy = train_and_evaluate(
        data['X_energy'], data['y'], data['groups'],
        data['energy_dim'], n_epochs=200, label="Energy-only"
    )
    all_results['energy_only'] = r_energy
    
    # 4b: FP features ONLY (baseline)
    logger.info("\n--- FP features only (baseline) ---")
    X_fp_cl = np.concatenate([data['X_fp'], 
                               np.zeros((len(data['y']), data['cl_dim']), dtype=np.float32)], axis=1) \
              if data['cl_dim'] > 0 else data['X_fp']
    r_fp = train_and_evaluate(
        data['X_fp'], data['y'], data['groups'],
        data['fp_dim'], n_epochs=200, label="FP-only"
    )
    all_results['fp_only'] = r_fp
    
    # 4c: Energy + FP (hybrid)
    logger.info("\n--- Energy + FP (hybrid) ---")
    X_efp = np.concatenate([data['X_energy'], data['X_fp']], axis=1)
    r_efp = train_and_evaluate(
        X_efp, data['y'], data['groups'],
        data['energy_dim'] + data['fp_dim'], n_epochs=200, label="Energy+FP"
    )
    all_results['energy_fp'] = r_efp
    
    # 4d: Energy + FP + CL embedding (full hybrid)
    logger.info("\n--- Energy + FP + CL (full) ---")
    r_full = train_and_evaluate(
        data['X_combined'], data['y'], data['groups'],
        data['X_combined'].shape[1], n_epochs=200, label="Full"
    )
    all_results['full_hybrid'] = r_full
    
    # === Final comparison ===
    logger.info("\n" + "=" * 60)
    logger.info("FINAL COMPARISON")
    logger.info("=" * 60)
    
    logger.info("\n  %-25s %-15s %-15s", "Model", "Random CV", "Drug-pair CV")
    logger.info("  " + "-" * 55)
    for name, res in all_results.items():
        r_rand = res.get('random', {}).get('r', 0)
        r_pair = res.get('drug_pair', {}).get('r', 0)
        logger.info("  %-25s r=%.4f        r=%.4f", name, r_rand, r_pair)
    
    logger.info("\n  Previous MLP baseline:   r=0.7030        r=0.6200")
    
    with open(MODEL_DIR / "energy_synergy_results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=float)
    logger.info("\nSaved: energy_synergy_results.json")


if __name__ == "__main__":
    main()
