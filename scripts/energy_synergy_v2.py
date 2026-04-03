"""
Energy Landscape Drug Synergy v2
=================================

Fixes from v1:
1. Case-insensitive drug matching (O'Neil uses UPPERCASE)
2. All 38 O'Neil drugs mapped to pathway targets
3. Synergy formula captures pathway convergence/bottleneck effects
4. Multi-path analysis with backup route detection
"""

import json, logging, math, pickle
from pathlib import Path
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

# === Physical Constants ===
R = 1.987e-3    # kcal/mol/K
T = 310.15      # K (37°C)
RT = R * T      # ~0.616 kcal/mol
kB = 3.2996e-24 # kcal/K
h  = 1.5837e-34 # kcal·s
kBT_h = kB * T / h  # ~6.45e12 s⁻¹

def kcat_to_dG(kcat):
    return -RT * math.log(kcat / kBT_h)

# ============================================================
# PART 1: Extended Signaling Pathway
# ============================================================

def build_pathway_graph():
    """
    Extended signaling network with multiple redundant paths 
    to enable pathway convergence-based synergy.
    
    Key addition: parallel/redundant routes so that blocking 
    one node forces traffic through alternative (higher energy) routes.
    """
    reactions = [
        # === MAPK cascade (primary proliferation) ===
        ("STIMULUS", "EGFR", 0.16, "EGFR activation"),
        ("EGFR", "RAS",  1.0,  "SOS-catalyzed exchange"),
        ("RAS",  "RAF",  2.0,  "RAS→RAF recruitment"),
        ("RAF",  "MEK",  5.0,  "RAF→MEK phosphorylation"),
        ("MEK",  "ERK",  10.0, "MEK→ERK phosphorylation"),
        ("ERK",  "PROLIFERATION", 1.0, "ERK→transcription"),
        
        # === PI3K/AKT/mTOR (survival) ===
        ("EGFR", "PI3K", 1.5,  "EGFR→PI3K"),
        ("RAS",  "PI3K", 1.2,  "RAS→PI3K direct"),
        ("PI3K", "AKT",  3.0,  "PI3K→AKT"),
        ("AKT",  "MTOR", 2.0,  "AKT→mTOR"),
        ("MTOR", "SURVIVAL", 1.0, "mTOR→survival"),
        
        # === FAK (migration) ===
        ("EGFR", "FAK",  5.0,  "EGFR→FAK direct"),
        ("FAK",  "MIGRATION", 8.0, "FAK→migration"),
        
        # === Crosstalk (creates convergence points) ===
        ("ERK",  "FAK",  0.1,  "ERK→FAK crosstalk"),
        ("AKT",  "PROLIFERATION", 0.3, "AKT→cell cycle (bypass)"),
        ("ERK",  "AKT",  0.15, "ERK→AKT crosstalk"),
        ("PI3K", "RAF",  0.2,  "PI3K→RAF crosstalk"),
        
        # === Alternative proliferation routes ===
        ("MTOR", "PROLIFERATION", 0.4, "mTOR→S6K→proliferation"),
        
        # === WEE1/CHK1 checkpoint ===
        ("ERK",  "CHECKPOINT", 0.8, "ERK→checkpoint activation"),
        ("CHECKPOINT", "PROLIFERATION", 2.0, "checkpoint→cell cycle"),
        
        # === HSP90/proteasome ===
        ("HSP90", "RAF",  3.0,  "HSP90 stabilizes RAF"),
        ("HSP90", "AKT",  2.0,  "HSP90 stabilizes AKT"),
        ("HSP90", "EGFR", 1.5,  "HSP90 stabilizes EGFR"),
        ("STIMULUS", "HSP90", 10.0, "constitutive HSP90"),
        
        # === CDK pathway ===
        ("ERK",  "CDK",  1.5,  "ERK→Cyclin D→CDK4/6"),
        ("CDK",  "PROLIFERATION", 3.0, "CDK→Rb→E2F→S phase"),
        
        # === RAS inactivation ===
        ("RAS",  "RAS_GDP", 19.0, "GAP hydrolysis"),
    ]
    
    G = nx.DiGraph()
    for src, tgt, kcat, name in reactions:
        dG = kcat_to_dG(kcat)
        G.add_edge(src, tgt, kcat=kcat, dG=dG, weight=dG, rxn_name=name)
    
    logger.info("Extended graph: %d nodes, %d edges", G.number_of_nodes(), G.number_of_edges())
    return G


# ============================================================
# PART 2: Drug Targets (ALL 38 O'Neil drugs, case-insensitive)
# ============================================================

# All targets specified as: (pathway_node, potency_factor)
# Potency: higher = stronger barrier increase
DRUG_TARGETS = {
    # EGFR inhibitors
    "ERLOTINIB":    [("EGFR", 40)],
    "LAPATINIB":    [("EGFR", 50)],
    
    # RAF/MAPK
    "SORAFENIB":    [("RAF", 25), ("EGFR", 5)],
    "PD325901":     [("MEK", 80)],   # MEK inhibitor (potent)
    
    # PI3K/AKT/mTOR
    "BEZ-235":      [("PI3K", 60), ("MTOR", 40)],
    "MK-2206":      [("AKT", 70)],
    "MK-8669":      [("MTOR", 40)],
    
    # CDK/Cell cycle
    "DINACICLIB":   [("CDK", 80)],    # pan-CDK inhibitor
    "MK-1775":      [("CHECKPOINT", 50)],  # WEE1
    "AZD1775":      [("CHECKPOINT", 50)],
    "MK-8776":      [("CHECKPOINT", 40)],  # CHK1
    "MK-5108":      [("PROLIFERATION", 25)], # Aurora A kinase
    
    # DNA damage agents (block proliferation output)
    "5-FU":         [("PROLIFERATION", 30)],
    "GEMCITABINE":  [("PROLIFERATION", 35)],
    "OXALIPLATIN":  [("PROLIFERATION", 35)],
    "DOXORUBICIN":  [("PROLIFERATION", 40)],
    "ETOPOSIDE":    [("PROLIFERATION", 25)],
    "TOPOTECAN":    [("PROLIFERATION", 30)],
    "SN-38":        [("PROLIFERATION", 35)],  # active metabolite of irinotecan (TOP1)
    "TEMOZOLOMIDE": [("PROLIFERATION", 20)],
    "METHOTREXATE": [("PROLIFERATION", 25)],
    "CARBOPLATIN":  [("PROLIFERATION", 30)],
    "MITOMYCINE":   [("PROLIFERATION", 30)],
    "CYCLOPHOSPHAMIDE": [("PROLIFERATION", 20)],
    
    # Tubulin (migration + proliferation)
    "PACLITAXEL":   [("MIGRATION", 30), ("PROLIFERATION", 20)],
    "VINBLASTINE":  [("MIGRATION", 25), ("PROLIFERATION", 15)],
    "VINORELBINE":  [("MIGRATION", 25), ("PROLIFERATION", 15)],
    
    # PARP
    "ABT-888":      [("PROLIFERATION", 15)],
    "MK-4827":      [("PROLIFERATION", 20)],  # niraparib
    
    # Proteasome
    "BORTEZOMIB":   [("SURVIVAL", 40), ("PROLIFERATION", 25)],
    
    # HDAC = ZOLINZA (vorinostat)
    "ZOLINZA":      [("PROLIFERATION", 15), ("SURVIVAL", 10)],
    
    # HSP90 inhibitor
    "GELDANAMYCIN": [("HSP90", 60)],  # destabilizes RAF, AKT, EGFR clients
    
    # Multi-kinase
    "DASATINIB":    [("FAK", 30), ("RAF", 10)],
    "SUNITINIB":    [("EGFR", 10), ("RAF", 10)],
    
    # Gamma-secretase (Notch pathway → proliferation)
    "MRK-003":      [("PROLIFERATION", 15)],
    
    # Farnesyltransferase (RAS membrane localization)
    "L778123":      [("RAS", 20)],   # FTI → blocks RAS membrane anchoring
    
    # BRD4
    "MK-4541":      [("PROLIFERATION", 15), ("SURVIVAL", 10)],
    
    # AMPK/metabolic
    "METFORMIN":    [("MTOR", 15)],   # AMPK activation → mTOR inhibition
    
    # Corticosteroid
    "DEXAMETHASONE": [("SURVIVAL", 15), ("PROLIFERATION", 10)],
}

PHENOTYPES = ["PROLIFERATION", "SURVIVAL", "MIGRATION"]


# ============================================================
# PART 3: Improved Perturbation + Synergy with Convergence
# ============================================================

def perturb_graph(G_base, drug_name, potency_scale=1.0):
    """Apply drug perturbation. Increases ΔG‡ of edges from/to target."""
    G = G_base.copy()
    drug_up = drug_name.upper().strip()
    
    if drug_up not in DRUG_TARGETS:
        return G
    
    for target_node, potency in DRUG_TARGETS[drug_up]:
        barrier_inc = RT * math.log(1 + potency * potency_scale)
        
        if target_node in G:
            # Block outgoing edges
            for _, succ, data in G.edges(target_node, data=True):
                data['weight'] = data['dG'] + barrier_inc
            # Block incoming edges (for phenotype nodes like PROLIFERATION)
            if target_node in PHENOTYPES:
                for pred, _, data in G.in_edges(target_node, data=True):
                    data['weight'] = data['dG'] + barrier_inc
    
    return G


def compute_all_paths_energy(G, source="STIMULUS"):
    """Compute min energy path to each phenotype."""
    results = {}
    for target in PHENOTYPES:
        try:
            length = nx.dijkstra_path_length(G, source, target, weight='weight')
            path = nx.dijkstra_path(G, source, target, weight='weight')
            
            # Also find number of simple paths (redundancy measure)
            try:
                n_paths = sum(1 for _ in nx.all_simple_paths(G, source, target, cutoff=10))
            except:
                n_paths = 1
            
            results[target] = {'total_dG': length, 'path': path, 'n_paths': n_paths}
        except nx.NetworkXNoPath:
            results[target] = {'total_dG': 999.0, 'path': [], 'n_paths': 0}
    return results


def compute_synergy_features(G_base, drug_a, drug_b):
    """
    Extract rich synergy features from energy landscape perturbation.
    
    Key insight for CONVERGENCE synergy:
    - Count how many alternative paths remain after drug combo
    - If combo removes more paths than expected → synergistic
    - If one drug's effect depends on the other → interaction
    """
    drug_a_up = drug_a.upper().strip()
    drug_b_up = drug_b.upper().strip()
    
    # Baseline
    paths_none = compute_all_paths_energy(G_base)
    
    # Drug A only
    G_a = perturb_graph(G_base, drug_a_up)
    paths_a = compute_all_paths_energy(G_a)
    
    # Drug B only
    G_b = perturb_graph(G_base, drug_b_up)
    paths_b = compute_all_paths_energy(G_b)
    
    # Combination
    G_ab = perturb_graph(G_base, drug_a_up)
    G_ab = perturb_graph(G_ab, drug_b_up)
    paths_ab = compute_all_paths_energy(G_ab)
    
    features = []
    
    for pheno in PHENOTYPES:
        dG_none = paths_none[pheno]['total_dG']
        dG_a = paths_a[pheno]['total_dG']
        dG_b = paths_b[pheno]['total_dG']
        dG_ab = paths_ab[pheno]['total_dG']
        
        # Energy shifts
        delta_a = dG_a - dG_none
        delta_b = dG_b - dG_none
        delta_ab = dG_ab - dG_none
        
        # Supra-additive synergy (Bliss-like on energy)
        synergy = delta_ab - (delta_a + delta_b)
        
        # Path redundancy loss (key convergence signal)
        n_none = paths_none[pheno]['n_paths']
        n_a = paths_a[pheno]['n_paths']
        n_b = paths_b[pheno]['n_paths']
        n_ab = paths_ab[pheno]['n_paths']
        
        # Fractional path loss
        path_loss_a = 1.0 - (n_a / max(n_none, 1))
        path_loss_b = 1.0 - (n_b / max(n_none, 1))
        path_loss_ab = 1.0 - (n_ab / max(n_none, 1))
        
        # Supra-additive path blocking
        path_synergy = path_loss_ab - (path_loss_a + path_loss_b)
        
        # Route switching indicator (did the optimal path change?)
        path_changed_a = 1.0 if paths_a[pheno]['path'] != paths_none[pheno]['path'] else 0.0
        path_changed_ab = 1.0 if paths_ab[pheno]['path'] != paths_none[pheno]['path'] else 0.0
        
        features.extend([
            delta_a,          # Drug A effect
            delta_b,          # Drug B effect
            delta_ab,         # Combo effect
            synergy,          # Supra-additive energy
            path_loss_ab,     # Fraction of paths blocked by combo
            path_synergy,     # Supra-additive path blocking
            path_changed_ab,  # Did optimal route switch?
            n_ab / max(n_none, 1),  # Remaining path fraction
        ])
    
    # === Cross-phenotype features ===
    # Total energy increase across all phenotypes
    total_delta_ab = sum(
        paths_ab[p]['total_dG'] - paths_none[p]['total_dG'] for p in PHENOTYPES
    )
    features.append(total_delta_ab)
    
    # Do drugs target same pathway or different?
    targets_a = set(t for t, _ in DRUG_TARGETS.get(drug_a_up, []))
    targets_b = set(t for t, _ in DRUG_TARGETS.get(drug_b_up, []))
    overlap = len(targets_a & targets_b)
    total_targets = len(targets_a | targets_b)
    features.extend([float(overlap), float(total_targets)])
    
    # Same-vs-complementary pathway score
    same_path = 1.0 if overlap > 0 else 0.0
    features.append(same_path)
    
    # Total potency
    pot_a = sum(p for _, p in DRUG_TARGETS.get(drug_a_up, [(None, 0)]))
    pot_b = sum(p for _, p in DRUG_TARGETS.get(drug_b_up, [(None, 0)]))
    features.extend([float(pot_a), float(pot_b), float(pot_a * pot_b / 1000)])
    
    return np.array(features, dtype=np.float32)


# ============================================================
# PART 4: O'Neil Integration + Evaluation
# ============================================================

def build_dataset(G_base):
    """Build O'Neil dataset with energy + FP + CL features."""
    df = pd.read_csv(DATA_DIR / "synergy_combined.csv", low_memory=False)
    df = df[df["source"] == "oneil"]
    
    # Fingerprints 
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
    
    # Cell-line embeddings
    embed_path = DATA_DIR / "depmap" / "cellline_embedding_v2.pkl"
    embed_data = None
    emb_dim = 0
    if embed_path.exists():
        with open(embed_path, "rb") as f:
            embed_data = pickle.load(f)
        emb_dim = embed_data["dim"]
    
    def norm_cl(n):
        return str(n).upper().replace("-","").replace("_","").replace(" ","").replace(".","")
    
    # Check coverage
    mapped_count = sum(1 for d in df.drug_a.unique() if d.upper() in DRUG_TARGETS)
    logger.info("Drug coverage: %d/%d drugs mapped", mapped_count, df.drug_a.nunique())
    
    # Energy feature cache (per drug pair — same across cell lines for base graph)
    energy_cache = {}
    
    X_energy, X_fp, X_cl, y_list, groups = [], [], [], [], []
    
    for _, row in df.iterrows():
        da, db = str(row["drug_a"]), str(row["drug_b"])
        score = float(row["synergy_loewe"])
        if np.isnan(score): continue
        if da not in fps or db not in fps: continue
        
        # Energy features (cached per pair)
        pair_key = tuple(sorted([da.upper(), db.upper()]))
        if pair_key not in energy_cache:
            has_a = da.upper() in DRUG_TARGETS
            has_b = db.upper() in DRUG_TARGETS
            if has_a and has_b:
                energy_cache[pair_key] = compute_synergy_features(G_base, da, db)
            elif has_a or has_b:
                energy_cache[pair_key] = compute_synergy_features(G_base, da, db)
            else:
                energy_cache[pair_key] = np.zeros(31, dtype=np.float32)
        
        X_energy.append(energy_cache[pair_key])
        X_fp.append(np.concatenate([fps[da], fps[db]]))
        
        if embed_data:
            cl = norm_cl(str(row["cell_line"]))
            cl_feat = embed_data["embeddings"].get(cl, np.zeros(emb_dim, dtype=np.float32))
            X_cl.append(cl_feat)
        
        y_list.append(score)
        groups.append(pair_key)
    
    X_energy = np.array(X_energy, dtype=np.float32)
    X_fp = np.array(X_fp, dtype=np.float32)
    y = np.array(y_list, dtype=np.float32)
    
    unique_pairs = list(set(groups))
    pair_to_id = {p: i for i, p in enumerate(unique_pairs)}
    group_ids = np.array([pair_to_id[g] for g in groups])
    
    mapped_pairs = sum(1 for k, v in energy_cache.items() if v.sum() != 0)
    logger.info("Dataset: %d samples, %d pairs, %d energy-mapped (%.1f%%)",
                len(y), len(unique_pairs), mapped_pairs, 
                mapped_pairs / max(len(unique_pairs), 1) * 100)
    
    result = {
        'X_energy': X_energy,
        'X_fp': X_fp,
        'y': y,
        'groups': group_ids,
        'energy_dim': X_energy.shape[1],
        'fp_dim': X_fp.shape[1],
    }
    
    if embed_data:
        X_cl = np.array(X_cl, dtype=np.float32)
        result['X_cl'] = X_cl
        result['cl_dim'] = emb_dim
    
    return result


class SynergyDNN(nn.Module):
    def __init__(self, dim, hidden=[512, 256, 128]):
        super().__init__()
        layers = []
        prev = dim
        for h in hidden:
            layers.extend([nn.Linear(prev, h), nn.LayerNorm(h), nn.SiLU(), nn.Dropout(0.3)])
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)
    def forward(self, x):
        return self.net(x).squeeze(-1)


def train_eval(X, y, groups, dim, n_epochs=250, label=""):
    """Train with both CV methods."""
    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)
    
    results = {}
    for cv_name, kf in [("random", KFold(5, shuffle=True, random_state=42)),
                         ("drug_pair", GroupKFold(5))]:
        rs = []
        splits = kf.split(X_s, y, groups) if cv_name == "drug_pair" else kf.split(X_s)
        
        for fold, (ti, vi) in enumerate(splits):
            Xt = torch.FloatTensor(X_s[ti]).to(device)
            yt = torch.FloatTensor(y[ti]).to(device)
            Xv = torch.FloatTensor(X_s[vi]).to(device)
            
            model = SynergyDNN(dim).to(device)
            opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_epochs)
            
            best_r, best_state, patience = -1, None, 0
            for epoch in range(n_epochs):
                model.train()
                perm = torch.randperm(len(Xt))
                for s in range(0, len(Xt), 2048):
                    idx = perm[s:s+2048]
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
        
        avg, std = np.mean(rs), np.std(rs)
        logger.info("  %s %s: r=%.4f +/- %.4f", label, cv_name, avg, std)
        results[cv_name] = {"r": round(float(avg), 4), "std": round(float(std), 4)}
    
    return results


def main():
    logger.info("=" * 60)
    logger.info("ENERGY LANDSCAPE SYNERGY v2")
    logger.info("=" * 60)
    
    all_results = {}
    
    # Part 1: Build graph
    logger.info("\n--- Building pathway graph ---")
    G = build_pathway_graph()
    
    # Show baseline paths
    paths = compute_all_paths_energy(G)
    for ph, info in paths.items():
        logger.info("  %s: ΔG‡=%.1f, steps=%d, n_paths=%d, route=%s",
                     ph, info['total_dG'], len(info['path'])-1, info['n_paths'],
                     "→".join(info['path']))
    
    # Part 2: Drug perturbation examples
    logger.info("\n--- Drug perturbation examples ---")
    
    examples = [
        ("SORAFENIB", "PD325901",  "RAF+MEK (vertical)"),
        ("PD325901",  "BEZ-235",   "MEK+PI3K (horizontal)"),
        ("GELDANAMYCIN", "MK-2206", "HSP90+AKT (client destabilization)"),
        ("5-FU",      "OXALIPLATIN", "FOLFOX (DNA damage × DNA damage)"),
        ("DASATINIB",  "GEMCITABINE", "FAK+DNA (migration + proliferation)"),
    ]
    
    for da, db, desc in examples:
        feats = compute_synergy_features(G, da, db)
        # Total energy synergy is at index 24
        total_synergy = feats[3] + feats[11] + feats[19]  # sum of per-phenotype synergies
        logger.info("  %s + %s (%s): energy_synergy=%.3f", da, db, desc, total_synergy)
    
    # Part 3: Build dataset
    logger.info("\n--- Building O'Neil dataset ---")
    data = build_dataset(G)
    
    # Part 4: Model comparison
    logger.info("\n" + "=" * 40)
    logger.info("MODEL EVALUATION")
    logger.info("=" * 40)
    
    # A: Energy only
    logger.info("\n--- A: Energy only (19d) ---")
    r_e = train_eval(data['X_energy'], data['y'], data['groups'], data['energy_dim'], label="Energy")
    all_results['energy_only'] = r_e
    
    # B: FP only (baseline)
    logger.info("\n--- B: FP only (2048d) ---")
    r_fp = train_eval(data['X_fp'], data['y'], data['groups'], data['fp_dim'], label="FP")
    all_results['fp_only'] = r_fp
    
    # C: Energy + FP
    logger.info("\n--- C: Energy + FP ---")
    X_efp = np.concatenate([data['X_energy'], data['X_fp']], axis=1)
    r_efp = train_eval(X_efp, data['y'], data['groups'], X_efp.shape[1], label="E+FP")
    all_results['energy_fp'] = r_efp
    
    # D: Energy + FP + CL
    if 'X_cl' in data:
        logger.info("\n--- D: Energy + FP + CL (full) ---")
        X_full = np.concatenate([data['X_energy'], data['X_fp'], data['X_cl']], axis=1)
        r_full = train_eval(X_full, data['y'], data['groups'], X_full.shape[1], label="Full")
        all_results['full'] = r_full
    
    # E: Energy + CL (no FP — test mechanistic generalization)
    if 'X_cl' in data:
        logger.info("\n--- E: Energy + CL (no FP) ---")
        X_ecl = np.concatenate([data['X_energy'], data['X_cl']], axis=1)
        r_ecl = train_eval(X_ecl, data['y'], data['groups'], X_ecl.shape[1], label="E+CL")
        all_results['energy_cl'] = r_ecl
    
    # Final summary
    logger.info("\n" + "=" * 60)
    logger.info("FINAL COMPARISON")
    logger.info("=" * 60)
    logger.info("  %-20s %-15s %-15s", "Model", "Random", "Drug-pair")
    logger.info("  " + "-" * 50)
    for name, res in all_results.items():
        rr = res.get('random', {}).get('r', 0)
        rp = res.get('drug_pair', {}).get('r', 0)
        logger.info("  %-20s r=%.4f        r=%.4f", name, rr, rp)
    logger.info("\n  Phase 5 MLP (ref):  r=0.7030        r=0.6200")
    
    with open(MODEL_DIR / "energy_synergy_v2_results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=float)
    logger.info("\nSaved: energy_synergy_v2_results.json")


if __name__ == "__main__":
    main()
