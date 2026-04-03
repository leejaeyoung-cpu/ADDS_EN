"""
Energy Landscape Synergy v5 — Cell-line Specific Graphs + Optimized Features
=============================================================================

Key improvements over v4:
1. Cell-line specific graph: DepMap expression → per-cell-line edge weights
2. Cell-line mutations: KRAS/BRAF/PIK3CA → constitutive activation
3. Feature optimization: reduce noise from large graphs
4. All v4 fixes retained: real IC50, feedback loops, Hill thresholds, SL
"""

import json, logging, math, pickle, time
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

R = 1.987e-3; T = 310.15; RT = R * T
kB = 3.2996e-24; h_p = 1.5837e-34
kBT_h = kB * T / h_p

def kcat_to_dG(kcat):
    return -RT * math.log(max(kcat, 1e-10) / kBT_h)

def ic50_to_potency(ic50_nM):
    return -math.log10(max(ic50_nM, 0.01) * 1e-9)

# ================================================================
# PATHWAY GRAPH (Focused: 25 key nodes, ~50 edges)
# Simplified from v4's 40 nodes to reduce all_simple_paths noise
# ================================================================

def build_pathway_graph():
    """Focused pathway with key nodes only — avoids combinatorial path explosion."""
    reactions = [
        # RTK layer
        ("STIMULUS", "EGFR",     0.16, 500),
        ("STIMULUS", "HER2",     0.12, 600),
        ("STIMULUS", "IGF1R",    0.20, 400),
        
        # Adaptor
        ("EGFR",    "SOS_RAS",   4.6, 120),
        ("HER2",    "SOS_RAS",   3.0, 150),
        ("IGF1R",   "IRS1",      2.0, 200),
        
        # RAS-MAPK cascade (merged KRAS/NRAS, BRAF/CRAF for cleaner graph)
        ("SOS_RAS", "RAS",       2.0, 80),
        ("RAS",     "RAF",       5.0, 60),
        ("RAF",     "MEK",       8.0, 15),
        ("MEK",     "ERK",      10.2, 8),
        ("RAS",     "RAS_GDP",  19.0, 45),
        
        # PI3K-AKT-mTOR
        ("EGFR",    "PI3K",      1.5, 200),
        ("IRS1",    "PI3K",      2.5, 100),
        ("RAS",     "PI3K",      1.2, 250),
        ("PI3K",    "AKT",       4.7, 30),
        ("AKT",     "MTORC1",    2.0, 50),
        ("MTORC1",  "S6K",       1.5, 80),
        
        # HSP90
        ("STIMULUS","HSP90",    10.0, 10),
        ("HSP90",   "RAF",       3.0, 30),
        ("HSP90",   "AKT",       2.0, 50),
        ("HSP90",   "EGFR",      1.5, 70),
        
        # FAK/SRC
        ("EGFR",    "FAK",       6.0, 40),
        ("FAK",     "SRC",       4.0, 50),
        
        # Cell cycle
        ("ERK",     "CCND1",     1.5, 80),
        ("CCND1",   "CDK46",     3.0, 25),
        ("CDK46",   "RB1",       5.0, 30),
        ("RB1",     "PROLIFERATION", 8.0, 15),
        
        # Checkpoint
        ("WEE1",    "CDK1",      1.0, 50),  # WEE1 active = CDK1 brake ON
        ("CHK1",    "CDK1",      1.0, 60),
        ("CDK1",    "PROLIFERATION", 3.0, 30),
        
        # Outputs
        ("ERK",     "PROLIFERATION", 1.0, 100),
        ("AKT",     "PROLIFERATION", 0.3, 350),
        ("AKT",     "SURVIVAL",      3.0, 40),
        ("S6K",     "SURVIVAL",      1.0, 100),
        ("MTORC1",  "SURVIVAL",      0.8, 120),
        ("SRC",     "MIGRATION",     5.0, 40),
        ("FAK",     "MIGRATION",     8.5, 20),
    ]
    
    G = nx.DiGraph()
    for src, tgt, kcat, Km in reactions:
        dG = kcat_to_dG(kcat)
        G.add_edge(src, tgt, kcat=kcat, Km=Km, dG=dG, weight=dG)
    
    logger.info("Focused graph: %d nodes, %d edges", G.number_of_nodes(), G.number_of_edges())
    return G


# ================================================================
# DRUG IC50 (Real literature values, nM)
# ================================================================

DRUG_IC50 = {
    "ERLOTINIB":    [("EGFR", 2.0)],
    "LAPATINIB":    [("EGFR", 10.8), ("HER2", 9.2)],
    "SORAFENIB":    [("RAF", 22.0)],
    "PD325901":     [("MEK", 0.33)],
    "BEZ-235":      [("PI3K", 4.0), ("MTORC1", 6.0)],
    "MK-2206":      [("AKT", 8.0)],
    "MK-8669":      [("MTORC1", 0.2)],
    "DINACICLIB":   [("CDK46", 1.0), ("CDK1", 3.0)],
    "MK-1775":      [("WEE1", 5.2)],
    "AZD1775":      [("WEE1", 5.2)],
    "MK-8776":      [("CHK1", 3.0)],
    "MK-5108":      [("CDK1", 13.0)],
    "5-FU":         [("PROLIFERATION", 5000.0)],
    "GEMCITABINE":  [("PROLIFERATION", 50.0)],
    "OXALIPLATIN":  [("PROLIFERATION", 1000.0)],
    "DOXORUBICIN":  [("PROLIFERATION", 100.0)],
    "ETOPOSIDE":    [("PROLIFERATION", 1400.0)],
    "TOPOTECAN":    [("PROLIFERATION", 6.0)],
    "SN-38":        [("PROLIFERATION", 1.4)],
    "TEMOZOLOMIDE": [("PROLIFERATION", 200000.0)],
    "METHOTREXATE": [("PROLIFERATION", 21.0)],
    "CARBOPLATIN":  [("PROLIFERATION", 10000.0)],
    "MITOMYCINE":   [("PROLIFERATION", 500.0)],
    "CYCLOPHOSPHAMIDE": [("PROLIFERATION", 1000000.0)],
    "PACLITAXEL":   [("MIGRATION", 4.0), ("CDK1", 100.0)],
    "VINBLASTINE":  [("MIGRATION", 2.0), ("CDK1", 50.0)],
    "VINORELBINE":  [("MIGRATION", 3.0), ("CDK1", 80.0)],
    "ABT-888":      [("PROLIFERATION", 5.2)],
    "MK-4827":      [("PROLIFERATION", 3.8)],
    "BORTEZOMIB":   [("SURVIVAL", 3.0), ("PROLIFERATION", 10.0)],
    "ZOLINZA":      [("PROLIFERATION", 1000.0), ("SURVIVAL", 2000.0)],
    "GELDANAMYCIN": [("HSP90", 1.2)],
    "DASATINIB":    [("SRC", 0.55), ("FAK", 100.0)],
    "SUNITINIB":    [("EGFR", 880.0)],
    "MRK-003":      [("PROLIFERATION", 1500.0)],
    "L778123":      [("RAS", 2000.0)],
    "MK-4541":      [("PROLIFERATION", 500.0), ("SURVIVAL", 800.0)],
    "METFORMIN":    [("MTORC1", 200000.0)],
    "DEXAMETHASONE":[("SURVIVAL", 10.0), ("PROLIFERATION", 100.0)],
}

PHENOTYPES = ["PROLIFERATION", "SURVIVAL", "MIGRATION"]

# Feedback loops
FEEDBACK_LOOPS = [
    ("ERK",    "SOS_RAS", 0.4),
    ("S6K",    "IRS1",    0.3),
    ("MTORC1", "PI3K",    0.2),
    ("AKT",    "RAF",     0.3),
]

# Synthetic lethality
SYNTHETIC_LETHALITY = [
    ({"RAF"},        {"MEK"},    2.0),
    ({"MEK"},        {"PI3K"},   3.0),
    ({"EGFR"},       {"PI3K"},   2.5),
    ({"RAF"},        {"PI3K"},   2.5),
    ({"PROLIFERATION"}, {"SURVIVAL"}, 4.0),
]


# ================================================================
# FIX #2 + #5: Cell-line Specific Graph
# ================================================================

# Node → gene symbols for expression lookup
NODE_GENES = {
    "EGFR": ["EGFR"], "HER2": ["ERBB2"], "IGF1R": ["IGF1R"],
    "SOS_RAS": ["SOS1","GRB2"], "IRS1": ["IRS1"],
    "RAS": ["KRAS","NRAS"], "RAF": ["BRAF","RAF1"],
    "MEK": ["MAP2K1","MAP2K2"], "ERK": ["MAPK1","MAPK3"],
    "PI3K": ["PIK3CA","PIK3CB"], "AKT": ["AKT1","AKT2"],
    "MTORC1": ["MTOR"], "S6K": ["RPS6KB1"],
    "FAK": ["PTK2"], "SRC": ["SRC"],
    "CDK46": ["CDK4","CDK6"], "CDK1": ["CDK1"],
    "CCND1": ["CCND1"], "RB1": ["RB1"],
    "WEE1": ["WEE1"], "CHK1": ["CHEK1"],
    "HSP90": ["HSP90AA1","HSP90AB1"],
}

ALL_GENES = sorted(set(g for gs in NODE_GENES.values() for g in gs))

# Known mutations per O'Neil cell line (from COSMIC/CCLE)
CELLLINE_MUTATIONS = {
    "HCT116":  ["KRAS_G13D", "PIK3CA_H1047R"],
    "DLD1":    ["KRAS_G13D", "PIK3CA_E545K"],
    "SW620":   ["KRAS_G12V", "TP53_MUT"],
    "SW837":   ["KRAS_G12V"],
    "LOVO":    ["KRAS_G13D"],
    "RKO":     ["BRAF_V600E", "PIK3CA_H1047R"],
    "HT29":    ["BRAF_V600E", "PIK3CA_P449T"],
    "COLO320DM": ["TP53_MUT"],
    "A375":    ["BRAF_V600E"],
    "HT144":   ["BRAF_V600E", "TP53_MUT"],
    "RPMI7951": ["BRAF_V600E"],
    "SKMEL30": ["NRAS_Q61K"],
    "UACC62":  ["BRAF_V600E"],
    "A2058":   ["BRAF_V600E"],
    "NCIH460": ["KRAS_Q61H", "PIK3CA_E545K"],
    "NCIH23":  ["KRAS_G12C", "TP53_MUT"],
    "NCIH2122":["KRAS_G12C"],
    "NCIH1650":["EGFR_DEL19"],  # exon 19 deletion
    "NCIH520": ["TP53_MUT"],
    "A427":    ["KRAS_G12D"],
    "LNCAP":   ["PTEN_LOSS"],
    "VCAP":    ["TP53_MUT"],
    "SKOV3":   ["PIK3CA_H1047R"],
    "OVCAR3":  ["TP53_MUT"],
    "OV90":    ["TP53_MUT"],
    "A2780":   [],
    "CAOV3":   ["TP53_MUT"],
    "ES2":     ["BRAF_G464V"],
    "MDAMB436":["BRCA1_MUT"],
    "T47D":    ["PIK3CA_H1047R", "TP53_MUT"],
    "KPL1":    [],
    "EFM192B": ["TP53_MUT"],
    "ZR751":   [],
    "PA1":     [],
    "UWB1289": ["BRCA1_MUT"],
    "UWB1289BRCA1": [],  # BRCA1 restored
    "MSTO":    [],
    "SKMES1":  ["TP53_MUT"],
    "OCUBM":   [],
}

# Mutation → graph effects
MUTATION_EFFECTS = {
    "KRAS_G12V":    {"RAS": 3.0, "RAS_GDP": 0.05},
    "KRAS_G12D":    {"RAS": 3.0, "RAS_GDP": 0.05},
    "KRAS_G12C":    {"RAS": 2.0, "RAS_GDP": 0.1},
    "KRAS_G13D":    {"RAS": 2.0, "RAS_GDP": 0.1},
    "KRAS_Q61H":    {"RAS": 4.0, "RAS_GDP": 0.02},
    "NRAS_Q61K":    {"RAS": 3.0, "RAS_GDP": 0.05},
    "BRAF_V600E":   {"RAF": 5.0},
    "BRAF_G464V":   {"RAF": 2.0},
    "PIK3CA_H1047R":{"PI3K": 3.0},
    "PIK3CA_E545K": {"PI3K": 2.5},
    "PIK3CA_P449T": {"PI3K": 1.5},
    "EGFR_DEL19":   {"EGFR": 5.0},  # sensitizing mutation
    "PTEN_LOSS":    {"AKT": 2.0},    # no PTEN → AKT overactive
    "TP53_MUT":     {"CHK1": 0.3},   # impaired checkpoint
    "BRCA1_MUT":    {},              # affects DNA repair, not pathway kinetics
}


def load_expression_data():
    """Load CCLE expression and create gene-level lookup per cell line."""
    expr_path = DATA_DIR / "depmap" / "ccle_expression.csv"
    sample_path = DATA_DIR / "depmap_sample_info.csv"
    
    logger.info("Loading CCLE expression data...")
    expr_df = pd.read_csv(expr_path, index_col=0, low_memory=False)
    sample_df = pd.read_csv(sample_path, low_memory=False)
    
    # Build ACH → cell_line_name mapping
    ach_to_name = {}
    for _, row in sample_df.iterrows():
        name = str(row.get('stripped_cell_line_name', '')).upper().replace("-","").replace("_","").replace(" ","")
        ach_to_name[row['DepMap_ID']] = name
    
    # Find columns for our pathway genes
    gene_cols = {}
    for col in expr_df.columns:
        gene = col.split(" (")[0].strip()
        if gene in ALL_GENES:
            gene_cols[gene] = col
    
    logger.info("Found %d/%d pathway genes in expression data", len(gene_cols), len(ALL_GENES))
    
    # Compute median expression per gene (across all cell lines)
    gene_medians = {}
    for gene, col in gene_cols.items():
        vals = expr_df[col].dropna().values
        gene_medians[gene] = float(np.median(vals)) if len(vals) > 0 else 0.0
    
    # Build per-cell-line expression dict
    cl_expression = {}
    for ach_id, row in expr_df.iterrows():
        cl_name = ach_to_name.get(ach_id, "")
        if not cl_name:
            continue
        expr = {}
        for gene, col in gene_cols.items():
            val = row.get(col, np.nan)
            if not np.isnan(val):
                med = gene_medians.get(gene, 0)
                # Expression ratio: how much above/below median
                # log2(TPM+1) is the scale; convert to fold-change
                if abs(med) > 0.01:
                    expr[gene] = 2.0 ** (val - med)  # fold change vs median
                else:
                    expr[gene] = 1.0
            else:
                expr[gene] = 1.0
        cl_expression[cl_name] = expr
    
    logger.info("Built expression profiles for %d cell lines", len(cl_expression))
    return cl_expression


def build_cellline_graph(G_base, cell_line_name, cl_expression):
    """
    Create cell-line specific graph by:
    1. Scaling edge weights by gene expression (Fix #5)
    2. Applying known mutations (Fix #2)
    """
    G = G_base.copy()
    cl_norm = cell_line_name.upper().replace("-","").replace("_","").replace(" ","").replace(".","")
    
    # Fix #5: Expression-based edge modulation
    expr = cl_expression.get(cl_norm, {})
    if expr:
        for node, genes in NODE_GENES.items():
            if node not in G:
                continue
            # Average fold-change for genes in this node
            fcs = [expr.get(g, 1.0) for g in genes]
            fc = np.mean(fcs)
            fc = max(0.1, min(10.0, fc))  # clamp
            
            for u, v, data in list(G.edges(node, data=True)):
                kcat_eff = data['kcat'] * fc
                data['weight'] = kcat_to_dG(max(kcat_eff, 0.001))
    
    # Fix #2: Mutation-based modulation
    mutations = CELLLINE_MUTATIONS.get(cl_norm, [])
    for mut in mutations:
        effects = MUTATION_EFFECTS.get(mut, {})
        for node, mult in effects.items():
            if node not in G:
                continue
            for u, v, data in list(G.edges(node, data=True)):
                kcat_eff = data['kcat'] * mult
                data['weight'] = kcat_to_dG(max(kcat_eff, 0.001))
    
    return G


# ================================================================
# Drug perturbation + Feedback + SL (same as v4)
# ================================================================

def perturb_graph(G_base, drug_name):
    G = G_base.copy()
    drug_up = drug_name.upper().strip()
    if drug_up not in DRUG_IC50:
        return G, set()
    blocked = set()
    for target_node, ic50_nM in DRUG_IC50[drug_up]:
        potency = ic50_to_potency(ic50_nM)
        barrier = RT * potency
        blocked.add(target_node)
        if target_node in G:
            for _, succ, data in G.edges(target_node, data=True):
                data['weight'] = data.get('dG', data['weight']) + barrier
            if target_node in PHENOTYPES:
                for pred, _, data in G.in_edges(target_node, data=True):
                    data['weight'] = data.get('dG', data['weight']) + barrier
    return G, blocked


def apply_feedback(G, blocked):
    for fb_src, fb_tgt, strength in FEEDBACK_LOOPS:
        if fb_src in blocked and fb_tgt in G:
            release = RT * strength * 10
            for _, succ, data in G.edges(fb_tgt, data=True):
                data['weight'] = max(data['weight'] - release, data.get('dG', 0) * 0.5)
    return G


def hill(x, K=0.5, n=3):
    return x**n / (K**n + x**n)


def sl_bonus(blocked_a, blocked_b):
    bonus = 0.0
    both = blocked_a | blocked_b
    for sa, sb, val in SYNTHETIC_LETHALITY:
        a1 = bool(sa & blocked_a); b1 = bool(sb & blocked_b)
        a2 = bool(sa & blocked_b); b2 = bool(sb & blocked_a)
        if (a1 and b1) or (a2 and b2):
            bonus += val
    return bonus


# ================================================================
# Feature Extraction
# ================================================================

def compute_flow(G, source="STIMULUS"):
    results = {}
    for target in PHENOTYPES:
        try:
            paths = list(nx.all_simple_paths(G, source, target, cutoff=8))
            if not paths:
                results[target] = {'dG': 999, 'n': 0, 'cap': 0, 'mean': 999, 'std': 0, 'path': []}
                continue
            energies = [sum(G[p[i]][p[i+1]]['weight'] for i in range(len(p)-1)) for p in paths]
            minE = min(energies)
            Z = sum(math.exp(-(E - minE)/RT) for E in energies)
            cap = math.exp(-minE/RT) * Z
            idx = int(np.argmin(energies))
            results[target] = {'dG': minE, 'n': len(paths), 'cap': cap,
                               'mean': np.mean(energies), 'std': np.std(energies) if len(energies) > 1 else 0,
                               'path': paths[idx]}
        except:
            results[target] = {'dG': 999, 'n': 0, 'cap': 0, 'mean': 999, 'std': 0, 'path': []}
    return results


def extract_features(G_cl, drug_a, drug_b):
    """Extract synergy features for a drug pair on a cell-line-specific graph."""
    da, db = drug_a.upper().strip(), drug_b.upper().strip()
    
    f0 = compute_flow(G_cl)
    
    Ga, ba = perturb_graph(G_cl, da); Ga = apply_feedback(Ga, ba)
    Gb, bb = perturb_graph(G_cl, db); Gb = apply_feedback(Gb, bb)
    
    Gab, bab1 = perturb_graph(G_cl, da)
    Gab, bab2 = perturb_graph(Gab, db)
    Gab = apply_feedback(Gab, bab1 | bab2)
    
    fa = compute_flow(Ga); fb = compute_flow(Gb); fab = compute_flow(Gab)
    
    feats = []
    for ph in PHENOTYPES:
        n = f0.get(ph, {}); a = fa.get(ph, {}); b = fb.get(ph, {}); ab = fab.get(ph, {})
        
        dA = a.get('dG',0) - n.get('dG',0)
        dB = b.get('dG',0) - n.get('dG',0)
        dAB = ab.get('dG',0) - n.get('dG',0)
        syn = dAB - (dA + dB)
        
        cn = max(n.get('cap',1), 1e-100)
        rA = a.get('cap',0)/cn; rB = b.get('cap',0)/cn; rAB = ab.get('cap',0)/cn
        bliss = rA * rB
        csyn = bliss - rAB
        
        hA = hill(rA); hB = hill(rB); hAB = hill(rAB)
        hsyn = hA * hB - hAB
        
        nb = max(n.get('n',1),1)
        ploss = 1.0 - ab.get('n',0)/nb
        rswitch = 1.0 if ab.get('path',[]) != n.get('path',[]) else 0.0
        
        feats.extend([dA, dB, dAB, syn, csyn, hsyn, ploss, rswitch, rAB,
                      ab.get('mean',0) - n.get('mean',0)])
    
    # Cross-phenotype
    total_loss = sum(1.0 - fab.get(p,{}).get('cap',0)/max(f0.get(p,{}).get('cap',1),1e-100)
                     for p in PHENOTYPES)
    feats.append(total_loss)
    
    # Synthetic lethality
    sbl = sl_bonus(ba, bb)
    feats.append(sbl)
    
    # Drug target features
    ta = {t for t,_ in DRUG_IC50.get(da, [])}
    tb = {t for t,_ in DRUG_IC50.get(db, [])}
    ov = len(ta & tb)
    tot = len(ta | tb)
    pa = sum(ic50_to_potency(ic) for _,ic in DRUG_IC50.get(da, [("",1000)]))
    pb = sum(ic50_to_potency(ic) for _,ic in DRUG_IC50.get(db, [("",1000)]))
    
    feats.extend([float(ov), float(tot), 1.0 if ov > 0 else 0.0,
                  pa, pb, pa*pb/100, float(sbl > 0)])
    
    return np.array(feats, dtype=np.float32)


# ================================================================
# Model
# ================================================================

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
    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)
    results = {}
    for cv, kf in [("random", KFold(5, shuffle=True, random_state=42)),
                    ("drug_pair", GroupKFold(5))]:
        rs = []
        splits = kf.split(X_s, y, groups) if cv == "drug_pair" else kf.split(X_s)
        for fold, (ti, vi) in enumerate(splits):
            Xt = torch.FloatTensor(X_s[ti]).to(device)
            yt = torch.FloatTensor(y[ti]).to(device)
            Xv = torch.FloatTensor(X_s[vi]).to(device)
            model = SynergyDNN(dim).to(device)
            opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_epochs)
            best_r, patience = -1, 0
            best_state = None
            for ep in range(n_epochs):
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
                if (ep+1) % 10 == 0:
                    model.eval()
                    with torch.no_grad(): vp = model(Xv).cpu().numpy()
                    r = pearsonr(y[vi], vp)[0]
                    if r > best_r:
                        best_r = r; patience = 0
                        best_state = {k:v.clone() for k,v in model.state_dict().items()}
                    else: patience += 1
                    if patience >= 5: break
            if best_state: model.load_state_dict(best_state)
            model.eval()
            with torch.no_grad(): vp = model(Xv).cpu().numpy()
            r = pearsonr(y[vi], vp)[0]
            rs.append(r)
            logger.info("  %s %s fold %d: r=%.4f", label, cv, fold+1, r)
        avg = np.mean(rs); std = np.std(rs)
        logger.info("  %s %s: r=%.4f +/- %.4f", label, cv, avg, std)
        results[cv] = {"r": round(float(avg),4), "std": round(float(std),4)}
    return results


# ================================================================
# Main
# ================================================================

def main():
    t0 = time.time()
    logger.info("=" * 60)
    logger.info("ENERGY LANDSCAPE SYNERGY v5 — Cell-line Specific Graphs")
    logger.info("=" * 60)
    
    G_base = build_pathway_graph()
    cl_expression = load_expression_data()
    
    # Show expression examples
    for cl_name in ["HCT116", "A375", "LNCAP"]:
        cl_norm = cl_name.upper().replace("-","")
        expr = cl_expression.get(cl_norm, {})
        if expr:
            logger.info("  %s expression: EGFR=%.2f, KRAS=%.2f, BRAF=%.2f, PIK3CA=%.2f, AKT1=%.2f",
                         cl_name, expr.get('EGFR',1), expr.get('KRAS',1),
                         expr.get('BRAF',1), expr.get('PIK3CA',1), expr.get('AKT1',1))
    
    # Show mutation effects
    logger.info("\n--- Mutation Effects ---")
    for cl_name in ["HCT116", "A375", "LNCAP", "A2780"]:
        muts = CELLLINE_MUTATIONS.get(cl_name, [])
        logger.info("  %s: %s", cl_name, muts if muts else "(wild-type)")
    
    # Build dataset
    df = pd.read_csv(DATA_DIR / "synergy_combined.csv", low_memory=False)
    df = df[df["source"] == "oneil"]
    
    smiles = {}
    for p in [MODEL_DIR / "synergy" / "drug_smiles.json",
              MODEL_DIR / "synergy" / "drug_smiles_extended.json"]:
        if p.exists():
            with open(p) as f: smiles.update(json.load(f))
    
    from rdkit import Chem
    from rdkit.Chem import AllChem
    fps = {}
    for name, smi in smiles.items():
        mol = Chem.MolFromSmiles(smi)
        if mol: fps[name] = np.array(AllChem.GetMorganFingerprintAsBitVect(mol, 2, 1024), dtype=np.float32)
    
    embed_data = None
    ep = DATA_DIR / "depmap" / "cellline_embedding_v2.pkl"
    if ep.exists():
        with open(ep, "rb") as f: embed_data = pickle.load(f)
    
    def norm_cl(n):
        return str(n).upper().replace("-","").replace("_","").replace(" ","").replace(".","")
    
    # Build cell-line specific graphs
    logger.info("\nBuilding cell-line specific graphs...")
    cl_graphs = {}
    for cl_name in df['cell_line'].unique():
        cl_norm = norm_cl(cl_name)
        cl_graphs[cl_norm] = build_cellline_graph(G_base, cl_name, cl_expression)
    logger.info("Built %d cell-line specific graphs", len(cl_graphs))
    
    # Compute features PER SAMPLE (not just per pair — because graph varies by cell line!)
    logger.info("\nComputing per-sample energy features (cell-line-specific)...")
    X_energy, X_fp, X_cl, y_list, groups = [], [], [], [], []
    
    feat_cache = {}  # (pair_key, cl_norm) → features
    
    for i, (_, row) in enumerate(df.iterrows()):
        da, db = str(row["drug_a"]), str(row["drug_b"])
        score = float(row["synergy_loewe"])
        cl = norm_cl(str(row["cell_line"]))
        if np.isnan(score) or da not in fps or db not in fps: continue
        
        pair_key = tuple(sorted([da.upper(), db.upper()]))
        cache_key = (pair_key, cl)
        
        if cache_key not in feat_cache:
            G_cl = cl_graphs.get(cl, G_base)
            has_a = da.upper() in DRUG_IC50
            has_b = db.upper() in DRUG_IC50
            if has_a or has_b:
                feat_cache[cache_key] = extract_features(G_cl, da, db)
            else:
                feat_cache[cache_key] = np.zeros(39, dtype=np.float32)
        
        X_energy.append(feat_cache[cache_key])
        X_fp.append(np.concatenate([fps[da], fps[db]]))
        if embed_data:
            X_cl.append(embed_data["embeddings"].get(cl, np.zeros(embed_data["dim"], dtype=np.float32)))
        y_list.append(score)
        groups.append(pair_key)
        
        if (i+1) % 5000 == 0:
            logger.info("  Processed %d/%d samples, %d unique (pair,cl) combos",
                         i+1, len(df), len(feat_cache))
    
    X_energy = np.array(X_energy, dtype=np.float32)
    X_fp = np.array(X_fp, dtype=np.float32)
    y = np.array(y_list, dtype=np.float32)
    
    unique_pairs = list(set(groups))
    pair_to_id = {p: i for i, p in enumerate(unique_pairs)}
    group_ids = np.array([pair_to_id[g] for g in groups])
    
    mapped = sum(1 for v in feat_cache.values() if v.sum() != 0)
    logger.info("Dataset: %d samples, %d pairs, %d unique (pair,cl), %d mapped",
                len(y), len(unique_pairs), len(feat_cache), mapped)
    logger.info("Energy dim: %d", X_energy.shape[1])
    
    # Evaluate
    all_results = {}
    
    logger.info("\n" + "=" * 40)
    logger.info("MODEL EVALUATION")
    logger.info("=" * 40)
    
    logger.info("\n--- A: Energy only (cell-line specific) ---")
    r_e = train_eval(X_energy, y, group_ids, X_energy.shape[1], label="Energy")
    all_results['energy_only'] = r_e
    
    logger.info("\n--- B: Energy + FP ---")
    X_efp = np.concatenate([X_energy, X_fp], axis=1)
    r_efp = train_eval(X_efp, y, group_ids, X_efp.shape[1], label="E+FP")
    all_results['energy_fp'] = r_efp
    
    if embed_data and X_cl:
        X_cl_arr = np.array(X_cl, dtype=np.float32)
        
        logger.info("\n--- C: Full (Energy + FP + CL) ---")
        X_full = np.concatenate([X_energy, X_fp, X_cl_arr], axis=1)
        r_full = train_eval(X_full, y, group_ids, X_full.shape[1], label="Full")
        all_results['full'] = r_full
        
        logger.info("\n--- D: Energy + CL (no FP) ---")
        X_ecl = np.concatenate([X_energy, X_cl_arr], axis=1)
        r_ecl = train_eval(X_ecl, y, group_ids, X_ecl.shape[1], label="E+CL")
        all_results['energy_cl'] = r_ecl
    
    elapsed = time.time() - t0
    
    logger.info("\n" + "=" * 60)
    logger.info("FINAL COMPARISON v5 (Cell-line Specific)")
    logger.info("=" * 60)
    logger.info("  %-20s %-15s %-15s", "Model", "Random", "Drug-pair")
    logger.info("  " + "-" * 50)
    for name, res in all_results.items():
        rr = res.get('random',{}).get('r',0)
        rp = res.get('drug_pair',{}).get('r',0)
        logger.info("  %-20s r=%.4f        r=%.4f", name, rr, rp)
    logger.info("\n  v3 Full (ref):     r=0.7164        r=0.6409")
    logger.info("  v4 Full (ref):     r=0.7140        r=0.6331")
    logger.info("  Phase5 MLP (ref):  r=0.7030        r=0.6200")
    logger.info("\n  Time: %.1f seconds", elapsed)
    
    with open(MODEL_DIR / "energy_synergy_v5_results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=float)
    logger.info("Saved: energy_synergy_v5_results.json")


if __name__ == "__main__":
    main()
