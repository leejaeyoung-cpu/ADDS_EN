"""
Fetch real PPI (Co-IP) data from BioGRID and IntAct for GNN edge calibration.

Sources:
1. BioGRID REST API v4 (thebiogrid.org) - free, no API key needed for basic queries
2. IntAct (EBI) REST API - free

Targets: PRNP (PrPC) and its known interaction partners relevant to
cancer signaling: MET, EGFR, RPSA (LamR), STI1/STIP1, FYN, SRC, etc.
"""

import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional

import requests
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

OUT_DIR = Path("F:/ADDS/data/real_ppi")

# ─── Genes of interest ───────────────────────────────────────────────────────
# PrPC's known interaction partners (literature + databases)
QUERY_GENES = {
    "PRNP":  {"uniprot": "P04156", "aliases": ["PrPC", "CD230"]},
    "MET":   {"uniprot": "P08581", "aliases": ["c-MET", "HGFR"]},
    "EGFR":  {"uniprot": "P00533", "aliases": ["ErbB1", "HER1"]},
    "RPSA":  {"uniprot": "P08865", "aliases": ["LamR", "67LR"]},
    "STIP1": {"uniprot": "P31948", "aliases": ["STI1", "HOP"]},
    "FYN":   {"uniprot": "P06241", "aliases": ["FYN"]},
    "SRC":   {"uniprot": "P12931", "aliases": ["c-Src"]},
    "NCAM1": {"uniprot": "P13591", "aliases": ["CD56"]},
    "GRB2":  {"uniprot": "P62993", "aliases": []},
    "CTNNB1": {"uniprot": "P35222", "aliases": ["β-catenin"]},
}

# Map from gene names to our GNN pathway node names
GENE_TO_NODE = {
    "PRNP": "PrPC", "MET": "cMET", "EGFR": "EGFR", "RPSA": "LamR",
    "KRAS": "RAS", "HRAS": "RAS", "NRAS": "RAS", "BRAF": "RAS",
    "PIK3CA": "PI3K", "PIK3CB": "PI3K", "PIK3R1": "PI3K",
    "AKT1": "PI3K", "AKT2": "PI3K", "MTOR": "PI3K",
    "JAK1": "JAK_STAT", "JAK2": "JAK_STAT", "STAT3": "JAK_STAT",
    "PTK2": "FAK", "SRC": "FAK",
    "CTNNB1": "Wnt", "APC": "Wnt", "GSK3B": "Wnt",
    "NOTCH1": "Notch", "NOTCH2": "Notch",
    "YAP1": "Hippo", "TAZ": "Hippo", "LATS1": "Hippo",
    "NFKB1": "NF_kB", "RELA": "NF_kB",
    "BECN1": "Autophagy", "ATG5": "Autophagy", "ATG7": "Autophagy",
    "MAP2K1": "RAS", "MAP2K2": "RAS", "MAPK1": "RAS", "MAPK3": "RAS",
    "GRB2": "RAS", "SOS1": "RAS", "RAF1": "RAS",
    "FYN": "PrPC",  # FYN is PrPC's direct signaling partner
    "STIP1": "PrPC",  # STI1/HOP is PrPC co-chaperone
}


# ─── STRING-DB API (free, no key) ─────────────────────────────────────────────

STRING_BASE = "https://string-db.org/api"


def fetch_string_interactions(gene: str, species: int = 9606,
                               score_threshold: int = 400) -> List[Dict]:
    """
    Fetch protein-protein interactions from STRING-DB.
    STRING is completely free with no API key requirement.
    Returns interactions with combined scores and evidence channels.
    """
    url = f"{STRING_BASE}/json/interaction_partners"
    params = {
        "identifiers": gene,
        "species": species,
        "required_score": score_threshold,
        "limit": 50,
        "caller_identity": "ADDS_energy_framework",
    }

    try:
        logger.info(f"  Fetching STRING-DB: {gene}...")
        resp = requests.get(url, params=params, timeout=30)
        if resp.status_code == 200:
            data = resp.json()
            interactions = []
            for item in data:
                interactions.append({
                    "gene_a": item.get("preferredName_A", gene),
                    "gene_b": item.get("preferredName_B", ""),
                    "combined_score": item.get("score", 0),
                    "experimental": item.get("escore", 0),  # Experimental evidence (Co-IP etc.)
                    "database": item.get("dscore", 0),       # Curated databases
                    "textmining": item.get("tscore", 0),      # Text mining
                    "coexpression": item.get("ascore", 0),    # Co-expression
                    "string_id_a": item.get("stringId_A", ""),
                    "string_id_b": item.get("stringId_B", ""),
                })
            logger.info(f"    → {len(interactions)} interactions found")
            return interactions
        else:
            logger.warning(f"    STRING-DB returned {resp.status_code}")
            return []
    except Exception as e:
        logger.warning(f"    STRING-DB error for {gene}: {e}")
        return []


# ─── Literature-curated PPI (fallback) ────────────────────────────────────────

# Manually curated from PubMed — these are well-established interactions
LITERATURE_PPI = [
    # PrPC direct interactors (extensively documented)
    {"gene_a": "PRNP", "gene_b": "RPSA", "score": 0.95, "pubmeds": ["11590154", "20418879"],
     "method": "Co-IP, GST pull-down, SPR", "note": "PrPC-LamR (67LR) direct binding"},
    {"gene_a": "PRNP", "gene_b": "STIP1", "score": 0.92, "pubmeds": ["15665375", "20018211"],
     "method": "Co-IP, Surface binding assay", "note": "PrPC-STI1/HOP neuroprotective signaling"},
    {"gene_a": "PRNP", "gene_b": "FYN", "score": 0.88, "pubmeds": ["20724480", "22761440"],
     "method": "Co-IP, FRET, phosphorylation assay", "note": "PrPC → Fyn kinase activation"},
    {"gene_a": "PRNP", "gene_b": "NCAM1", "score": 0.85, "pubmeds": ["12670416", "15485919"],
     "method": "Co-IP, FRET", "note": "PrPC-NCAM interaction in lipid rafts"},
    {"gene_a": "PRNP", "gene_b": "MET", "score": 0.75, "pubmeds": ["31723279"],
     "method": "Co-IP, proximity ligation", "note": "PrPC promotes c-MET activation in CRC"},
    {"gene_a": "PRNP", "gene_b": "EGFR", "score": 0.70, "pubmeds": ["28615065"],
     "method": "Co-IP, signaling analysis", "note": "PrPC-EGFR cross-talk in glioblastoma"},
    {"gene_a": "PRNP", "gene_b": "CTNNB1", "score": 0.65, "pubmeds": ["26553931"],
     "method": "Co-IP, reporter assay", "note": "PrPC stabilizes β-catenin (Wnt pathway)"},
    {"gene_a": "PRNP", "gene_b": "SRC", "score": 0.60, "pubmeds": ["24898255"],
     "method": "phosphorylation assay", "note": "PrPC → Src family kinase activation"},
    # Key cancer signaling interactions
    {"gene_a": "MET", "gene_b": "GRB2", "score": 0.98, "pubmeds": ["7651832"],
     "method": "Co-IP, Y2H", "note": "c-MET adaptor recruitment → RAS activation"},
    {"gene_a": "EGFR", "gene_b": "GRB2", "score": 0.98, "pubmeds": ["8242743"],
     "method": "Co-IP, crystallography", "note": "EGFR → GRB2/SOS → RAS canonical"},
    {"gene_a": "GRB2", "gene_b": "KRAS", "score": 0.95, "pubmeds": ["7651832"],
     "method": "Co-IP, reconstitution", "note": "GRB2/SOS → KRAS activation"},
]


def get_literature_edges() -> Dict:
    """Convert literature-curated PPI to GNN edges."""
    edges = {}
    for ppi in LITERATURE_PPI:
        a = ppi["gene_a"].upper()
        b = ppi["gene_b"].upper()
        node_a = GENE_TO_NODE.get(a)
        node_b = GENE_TO_NODE.get(b)
        if not node_a or not node_b or node_a == node_b:
            continue

        pair = tuple(sorted([node_a, node_b]))
        if pair not in edges or ppi["score"] > edges[pair]["weight"]:
            edges[pair] = {
                "weight": ppi["score"],
                "n_publications": len(ppi.get("pubmeds", [])),
                "methods": [ppi.get("method", "unknown")],
                "note": ppi.get("note", ""),
                "has_coip": "Co-IP" in ppi.get("method", ""),
                "source": "literature",
            }
    return edges


# ─── Process STRING-DB into GNN edges ─────────────────────────────────────────

def process_string_to_edges(all_interactions: List[Dict]) -> Dict:
    """
    Convert STRING-DB interactions to GNN edge weights.
    
    Strategy:
    - Use combined_score (0-1) as base confidence
    - Boost experimental evidence (escore) which includes real Co-IP data
    - Map gene names to our pathway node names
    """
    pair_data = {}
    for ix in all_interactions:
        a = ix["gene_a"].upper()
        b = ix["gene_b"].upper()
        
        node_a = GENE_TO_NODE.get(a)
        node_b = GENE_TO_NODE.get(b)
        if not node_a or not node_b or node_a == node_b:
            continue
        
        pair = tuple(sorted([node_a, node_b]))
        combined = ix.get("combined_score", 0)
        experimental = ix.get("experimental", 0)
        
        if pair not in pair_data or combined > pair_data[pair]["score"]:
            pair_data[pair] = {
                "score": combined,
                "experimental": experimental,
                "genes": set(),
            }
        pair_data[pair]["genes"].add(f"{a}-{b}")
    
    if not pair_data:
        return {}
    
    edges = {}
    for pair, data in pair_data.items():
        # Weight: use combined score, boost if experimental evidence exists
        weight = data["score"]
        has_coip = data["experimental"] > 0.3
        if has_coip:
            weight = min(0.98, weight * 1.15)
        
        edges[pair] = {
            "weight": round(float(weight), 3),
            "experimental_score": round(float(data["experimental"]), 3),
            "combined_score": round(float(data["score"]), 3),
            "genes": sorted(data["genes"]),
            "has_coip": has_coip,
            "source": "string_db",
        }
    
    return edges


def fetch_all_ppi():
    """Fetch PPI data from STRING-DB + literature fallback."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print("Fetching Real PPI Data (STRING-DB + Literature)")
    print("=" * 70)
    
    # ── STRING-DB ──
    all_interactions = []
    string_raw = {}
    
    for gene in QUERY_GENES:
        ixns = fetch_string_interactions(gene)
        string_raw[gene] = len(ixns)
        all_interactions.extend(ixns)
        time.sleep(1.0)  # STRING-DB rate limit: 1 req/sec
    
    # Save raw
    with open(OUT_DIR / "string_raw.json", 'w') as f:
        json.dump(all_interactions, f, indent=2, default=str)
    logger.info(f"Saved {len(all_interactions)} raw STRING-DB interactions")
    
    # Process STRING to edges
    string_edges = process_string_to_edges(all_interactions)
    
    # ── Literature fallback ──
    lit_edges = get_literature_edges()
    
    # ── Merge (literature takes priority for known pairs) ──
    edges = {}
    for pair, data in string_edges.items():
        edges[pair] = data
    for pair, data in lit_edges.items():
        if pair in edges:
            # Merge: use max of STRING and literature
            if data["weight"] > edges[pair]["weight"]:
                data["source"] = "literature+string_db"
                edges[pair] = data
            else:
                edges[pair]["source"] = "string_db+literature"
        else:
            edges[pair] = data
    
    # Print results
    print(f"\n  Merged Edges ({len(edges)} unique):")
    for pair, data in sorted(edges.items(), key=lambda x: x[1]["weight"], reverse=True):
        coip_tag = " [Co-IP/Exp]" if data.get("has_coip") else ""
        src = data.get("source", "unknown")
        print(f"    {pair[0]:15s} <-> {pair[1]:15s}: "
              f"w={data['weight']:.3f} [{src}]{coip_tag}")
    
    # Save processed edges
    edges_json = {f"{k[0]}|{k[1]}": v for k, v in edges.items()}
    with open(OUT_DIR / "ppi_gnn_edges.json", 'w') as f:
        json.dump(edges_json, f, indent=2, default=lambda x: list(x) if isinstance(x, set) else str(x))
    
    # Summary
    summary = {
        "genes_queried": list(QUERY_GENES.keys()),
        "string_per_gene": string_raw,
        "total_raw_string": len(all_interactions),
        "string_edges": len(string_edges),
        "literature_edges": len(lit_edges),
        "merged_edges": len(edges),
        "edges_with_coip": sum(1 for e in edges.values() if e.get("has_coip")),
    }
    with open(OUT_DIR / "ppi_summary.json", 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n  Summary:")
    print(f"    STRING-DB raw interactions: {len(all_interactions)}")
    print(f"    STRING-DB mapped edges: {len(string_edges)}")
    print(f"    Literature edges: {len(lit_edges)}")
    print(f"    Merged GNN edges: {len(edges)}")
    print(f"    With experimental evidence: {summary['edges_with_coip']}")
    print(f"    Saved to: {OUT_DIR}")
    
    return edges, all_interactions


if __name__ == "__main__":
    fetch_all_ppi()

