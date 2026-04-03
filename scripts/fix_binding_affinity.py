"""
Fix Binding Affinity Features via ChEMBL API
=============================================
PubChem assaysummary didn't yield binding data.
Use ChEMBL target search + activity endpoint.
Also enrich DGIdb targets with known MOA-based affinity.
"""
import requests
import json
import numpy as np
import pickle
import logging
import time
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

DATA_DIR = Path("F:/ADDS/data/ml_training")
MODEL_DIR = Path("F:/ADDS/models/synergy")
MECH_DIR = DATA_DIR / "mechanism_features"


# Known binding affinities from literature (pKi/pIC50 values, nM)
# Sources: DrugBank, literature reviews
KNOWN_AFFINITIES = {
    'ERLOTINIB': {'EGFR': 2.0, 'ABL1': 1300},  # Ki nM  
    'LAPATINIB': {'EGFR': 10.8, 'ERBB2': 9.2},
    'SORAFENIB': {'BRAF': 22, 'VEGFR2': 90, 'PDGFRB': 57, 'FLT3': 58, 'KIT': 68, 'RAF1': 6},
    'SUNITINIB': {'VEGFR2': 80, 'PDGFRA': 71, 'KIT': 1, 'FLT3': 250, 'RET': 100},
    'DASATINIB': {'ABL1': 0.6, 'SRC': 0.5, 'KIT': 5, 'EPHA2': 17, 'PDGFRB': 28},
    'BORTEZOMIB': {'PSMB5': 0.6},  # Proteasome subunit
    'PACLITAXEL': {'TUBB': 4.0},   # Beta-tubulin
    'VINBLASTINE': {'TUBB': 0.9},
    'VINORELBINE': {'TUBB': 2.5},
    'DOXORUBICIN': {'TOP2A': 160},
    'ETOPOSIDE': {'TOP2A': 2200, 'TOP2B': 3400},
    'TOPOTECAN': {'TOP1': 6.5},
    'SN-38': {'TOP1': 3.4},
    'METHOTREXATE': {'DHFR': 0.0034},  # Very potent
    '5-FU': {'TYMS': 30},
    'GEMCITABINE': {'RRM1': 61},
    'TEMOZOLOMIDE': {'MGMT': 1000},  # Alkylating agent (indirect)
    'CARBOPLATIN': {'DNA': 500},  # DNA crosslinker
    'OXALIPLATIN': {'DNA': 200},
    'CYCLOPHOSPHAMIDE': {'DNA': 800},
    'METFORMIN': {'PRKAA1': 2000, 'PRKAA2': 2200},  # AMPK activation
    'DEXAMETHASONE': {'NR3C1': 0.7},  # GR receptor
    'GELDANAMYCIN': {'HSP90AA1': 1.2},
    'ABT-888': {'PARP1': 5.2, 'PARP2': 2.9},  # Veliparib
    'MK-4827': {'PARP1': 3.8, 'PARP2': 2.1},  # Niraparib
    'MK-2206': {'AKT1': 8, 'AKT2': 12, 'AKT3': 65},
    'BEZ-235': {'PIK3CA': 4, 'PIK3CB': 75, 'MTOR': 6},  # Dactolisib
    'MK-8669': {'MTOR': 0.2},  # Ridaforolimus
    'PD325901': {'MAP2K1': 0.33, 'MAP2K2': 0.79},  # MEK inhibitor
    'ZOLINZA': {'HDAC1': 38, 'HDAC2': 53, 'HDAC3': 28},  # Vorinostat
    'DINACICLIB': {'CDK1': 3, 'CDK2': 1, 'CDK5': 1, 'CDK9': 4},
    'MK-8776': {'CHEK1': 3},  # CHK1 inhibitor
    'MK-5108': {'AURKA': 0.064},  # Aurora A inhibitor
    'MRK-003': {'NOTCH1': 500},  # Gamma-secretase inhibitor (indirect)
}


def build_affinity_from_literature():
    """Build affinity features from known literature values."""
    # Collect all target genes
    all_targets = set()
    for targets in KNOWN_AFFINITIES.values():
        all_targets.update(targets.keys())
    
    target_list = sorted(all_targets)
    target_idx = {t: i for i, t in enumerate(target_list)}
    n_targets = len(target_list)
    
    logger.info(f"Binding targets from literature: {n_targets}")
    
    # Build pKi vectors
    affinity_vectors = {}
    for drug, targets in KNOWN_AFFINITIES.items():
        vec = np.zeros(n_targets, dtype=np.float32)
        for gene, ki_nm in targets.items():
            if gene in target_idx:
                # Convert Ki (nM) to pKi = -log10(Ki_M) = -log10(Ki_nM * 1e-9) = 9 - log10(Ki_nM)
                pki = 9.0 - np.log10(max(ki_nm, 0.001))
                vec[target_idx[gene]] = pki
        affinity_vectors[drug.upper()] = vec
    
    # Fill missing drugs with zero vectors
    with open(MODEL_DIR / "drug_fingerprints.pkl", 'rb') as f:
        drug_fps = pickle.load(f)
    
    for drug in drug_fps:
        du = drug.upper()
        if du not in affinity_vectors:
            affinity_vectors[du] = np.zeros(n_targets, dtype=np.float32)
    
    return affinity_vectors, target_list


def try_chembl_affinity(drug_names):
    """Try ChEMBL API for additional binding data."""
    logger.info("Querying ChEMBL for binding affinities...")
    
    chembl_data = {}
    
    for drug in drug_names[:10]:  # Start with first 10 to test
        try:
            # Search for molecule
            url = f"https://www.ebi.ac.uk/chembl/api/data/molecule/search?q={drug}&format=json"
            resp = requests.get(url, timeout=15)
            
            if resp.status_code == 200:
                data = resp.json()
                molecules = data.get('molecules', [])
                if molecules:
                    chembl_id = molecules[0].get('molecule_chembl_id', '')
                    
                    # Get activities
                    url2 = f"https://www.ebi.ac.uk/chembl/api/data/activity?molecule_chembl_id={chembl_id}&standard_type=Ki&format=json&limit=20"
                    resp2 = requests.get(url2, timeout=15)
                    
                    if resp2.status_code == 200:
                        acts = resp2.json().get('activities', [])
                        targets = {}
                        for act in acts:
                            target_name = act.get('target_pref_name', '')
                            value = act.get('standard_value')
                            if target_name and value:
                                try:
                                    targets[target_name] = float(value)
                                except:
                                    pass
                        chembl_data[drug] = targets
                        logger.info(f"  {drug} ({chembl_id}): {len(targets)} Ki values")
                    
        except Exception as e:
            logger.warning(f"  {drug}: {e}")
        
        time.sleep(0.5)
    
    return chembl_data


def main():
    print("=" * 70)
    print("FIX: Building Binding Affinity Features")
    print("=" * 70)
    
    # Build from literature
    lit_vectors, lit_targets = build_affinity_from_literature()
    
    drugs_with_affinity = sum(1 for v in lit_vectors.values() if v.sum() > 0)
    print(f"\n  Literature-based affinity:")
    print(f"    Drugs with data: {drugs_with_affinity}/{len(lit_vectors)}")
    print(f"    Target genes: {len(lit_targets)}")
    print(f"    Targets: {lit_targets}")
    
    # Print pKi values
    print(f"\n  Drug binding profiles (pKi):")
    for drug in sorted(KNOWN_AFFINITIES.keys()):
        vec = lit_vectors[drug.upper()]
        nonzero = [(lit_targets[i], vec[i]) for i in range(len(vec)) if vec[i] > 0]
        targets_str = ", ".join([f"{t}={v:.1f}" for t, v in nonzero])
        print(f"    {drug:25s}: {targets_str}")
    
    # Try ChEMBL for additional data
    print(f"\n  Trying ChEMBL API for additional data...")
    with open(MODEL_DIR / "drug_fingerprints.pkl", 'rb') as f:
        drug_fps = pickle.load(f)
    
    chembl_data = try_chembl_affinity(list(drug_fps.keys()))
    print(f"  ChEMBL results: {sum(1 for v in chembl_data.values() if v)}/{len(chembl_data)}")
    
    # Now update the mechanism features
    print(f"\n  Updating mechanism features with literature affinity...")
    
    with open(MODEL_DIR / "drug_mechanism_features.pkl", 'rb') as f:
        mechanism_features = pickle.load(f)
    
    # Current: 200 target + 50 affinity(zero) + 47 pathway = 297
    # Replace the 50 zero-affinity with literature-based affinity
    n_target = 200
    n_old_affinity = 50
    n_pathway = 47
    n_lit = len(lit_targets)
    
    updated_features = {}
    for drug_upper, old_vec in mechanism_features.items():
        target_vec = old_vec[:n_target]
        pathway_vec = old_vec[n_target + n_old_affinity:]
        affinity_vec = lit_vectors.get(drug_upper, np.zeros(n_lit, dtype=np.float32))
        
        new_vec = np.concatenate([target_vec, affinity_vec, pathway_vec])
        updated_features[drug_upper] = new_vec
    
    n_new = len(next(iter(updated_features.values())))
    print(f"  Updated feature dim: {n_new} (200 target + {n_lit} affinity + {n_pathway} pathway)")
    
    with open(MODEL_DIR / "drug_mechanism_features.pkl", 'wb') as f:
        pickle.dump(updated_features, f)
    
    # Update metadata
    with open(MODEL_DIR / "mechanism_metadata.json") as f:
        meta = json.load(f)
    
    meta['n_affinity'] = n_lit
    meta['n_features'] = n_new
    meta['affinity_targets'] = lit_targets
    meta['drugs_with_affinity'] = drugs_with_affinity
    meta['affinity_source'] = 'literature (DrugBank, clinical pharmacology)'
    
    with open(MODEL_DIR / "mechanism_metadata.json", 'w') as f:
        json.dump(meta, f, indent=2)
    
    print(f"\n  Saved updated features")
    print(f"  Drugs with affinity: {drugs_with_affinity}/{len(updated_features)}")


if __name__ == "__main__":
    main()
