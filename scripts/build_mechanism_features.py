"""
Drug Mechanism Feature Engineering
====================================
Build mechanism-aware features for each drug:
1. Drug-Target Interaction Vector (DGIdb)
2. Binding Affinity Features (PubChem + ChEMBL)
3. Pathway Signaling Features (KEGG)
"""
import pandas as pd
import numpy as np
import requests
import json
import pickle
import time
import logging
from pathlib import Path
from collections import defaultdict

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

DATA_DIR = Path("F:/ADDS/data/ml_training")
MODEL_DIR = Path("F:/ADDS/models/synergy")
MECH_DIR = DATA_DIR / "mechanism_features"
MECH_DIR.mkdir(parents=True, exist_ok=True)


# ================================================================
# Step 1: Drug target genes from DGIdb
# ================================================================
def get_dgidb_targets(drug_names):
    """Query DGIdb GraphQL API for drug-gene interactions."""
    logger.info(f"Querying DGIdb for {len(drug_names)} drugs...")
    
    drug_targets = {}
    
    # DGIdb GraphQL endpoint
    url = "https://dgidb.org/api/graphql"
    
    for drug in drug_names:
        query = """
        {
          drugs(names: ["%s"]) {
            nodes {
              name
              conceptId
              interactions {
                gene {
                  name
                  conceptId
                }
                interactionScore
                interactionTypes {
                  type
                  directionality
                }
              }
            }
          }
        }
        """ % drug
        
        try:
            resp = requests.post(url, json={'query': query}, timeout=20)
            if resp.status_code == 200:
                data = resp.json()
                nodes = data.get('data', {}).get('drugs', {}).get('nodes', [])
                targets = []
                for node in nodes:
                    for interaction in node.get('interactions', []):
                        gene = interaction.get('gene', {})
                        gene_name = gene.get('name', '')
                        score = interaction.get('interactionScore')
                        types = [t.get('type', '') for t in interaction.get('interactionTypes', [])]
                        if gene_name:
                            targets.append({
                                'gene': gene_name,
                                'score': score,
                                'types': types,
                            })
                drug_targets[drug] = targets
                logger.info(f"  {drug}: {len(targets)} targets")
            else:
                logger.warning(f"  {drug}: HTTP {resp.status_code}")
                drug_targets[drug] = []
        except Exception as e:
            logger.warning(f"  {drug}: Error - {e}")
            drug_targets[drug] = []
        
        time.sleep(0.5)
    
    return drug_targets


def build_target_vectors(drug_targets, top_n=200):
    """Build binary target vectors from DGIdb results."""
    # Count all target genes across drugs
    gene_counts = defaultdict(int)
    for drug, targets in drug_targets.items():
        for t in targets:
            gene_counts[t['gene']] += 1
    
    # Select top N most common targets
    top_genes = sorted(gene_counts.items(), key=lambda x: -x[1])[:top_n]
    gene_list = [g[0] for g in top_genes]
    gene_idx = {g: i for i, g in enumerate(gene_list)}
    
    logger.info(f"Top {len(gene_list)} target genes selected")
    logger.info(f"  Most targeted: {top_genes[:10]}")
    
    # Build binary vectors
    vectors = {}
    for drug, targets in drug_targets.items():
        vec = np.zeros(len(gene_list), dtype=np.float32)
        for t in targets:
            if t['gene'] in gene_idx:
                vec[gene_idx[t['gene']]] = 1.0
        vectors[drug] = vec
    
    return vectors, gene_list


# ================================================================
# Step 2: Binding affinity from PubChem
# ================================================================
def get_pubchem_bioactivity(drug_names, smiles_dict):
    """Get binding affinity data from PubChem Bioassay."""
    logger.info(f"Querying PubChem bioactivity for {len(drug_names)} drugs...")
    
    drug_affinity = {}
    
    for drug in drug_names:
        try:
            # First get CID
            url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{drug}/cids/JSON"
            resp = requests.get(url, timeout=15)
            
            if resp.status_code != 200:
                drug_affinity[drug] = {}
                continue
            
            cids = resp.json().get('IdentifierList', {}).get('CID', [])
            if not cids:
                drug_affinity[drug] = {}
                continue
            
            cid = cids[0]
            
            # Get bioactivity data (Ki, Kd, IC50)
            url2 = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cid}/assaysummary/JSON"
            resp2 = requests.get(url2, timeout=20)
            
            affinities = {}
            if resp2.status_code == 200:
                data = resp2.json()
                table = data.get('Table', {})
                columns = table.get('Columns', {}).get('Column', [])
                rows = table.get('Row', [])
                
                # Find activity columns
                for row in rows[:100]:  # Limit to first 100 assays
                    cells = row.get('Cell', [])
                    if len(cells) >= len(columns):
                        # Look for target name and activity value
                        target_name = None
                        activity = None
                        activity_type = None
                        
                        for i, col in enumerate(columns):
                            val = cells[i] if i < len(cells) else ''
                            if col == 'TargetName':
                                target_name = str(val)
                            elif col == 'ActivityValue':
                                try:
                                    activity = float(val)
                                except:
                                    pass
                            elif col == 'ActivityName':
                                activity_type = str(val)
                        
                        if target_name and activity and activity > 0:
                            if target_name not in affinities or activity < affinities[target_name]:
                                affinities[target_name] = activity
            
            drug_affinity[drug] = affinities
            n_targets = len(affinities)
            logger.info(f"  {drug} (CID={cid}): {n_targets} targets with activity data")
            
        except Exception as e:
            logger.warning(f"  {drug}: Error - {e}")
            drug_affinity[drug] = {}
        
        time.sleep(0.5)
    
    return drug_affinity


def build_affinity_features(drug_affinity, drug_targets, top_genes, n_features=50):
    """Build affinity feature vector: -log10(Ki) for top target genes."""
    vectors = {}
    
    for drug in drug_affinity:
        vec = np.zeros(n_features, dtype=np.float32)
        affinities = drug_affinity[drug]
        
        # Map affinities to top gene positions
        for i, gene in enumerate(top_genes[:n_features]):
            if gene in affinities:
                ki = affinities[gene]
                if ki > 0:
                    vec[i] = -np.log10(ki * 1e-9) if ki > 1 else 9.0  # nM to pKi
            elif gene.upper() in {k.upper() for k in affinities}:
                for k, v in affinities.items():
                    if k.upper() == gene.upper() and v > 0:
                        vec[i] = -np.log10(v * 1e-9) if v > 1 else 9.0
        
        vectors[drug] = vec
    
    return vectors


# ================================================================
# Step 3: KEGG pathway features
# ================================================================
def get_kegg_pathways(gene_list):
    """Map genes to KEGG pathways using REST API."""
    logger.info(f"Mapping {len(gene_list)} genes to KEGG pathways...")
    
    gene_pathways = {}
    
    for gene in gene_list:
        try:
            # Convert gene symbol to KEGG ID
            url = f"https://rest.kegg.jp/find/hsa/{gene}"
            resp = requests.get(url, timeout=10)
            
            if resp.status_code == 200 and resp.text.strip():
                # Get first matching KEGG gene ID
                first_line = resp.text.strip().split('\n')[0]
                kegg_id = first_line.split('\t')[0]
                
                # Get pathways for this gene
                url2 = f"https://rest.kegg.jp/link/pathway/{kegg_id}"
                resp2 = requests.get(url2, timeout=10)
                
                if resp2.status_code == 200 and resp2.text.strip():
                    pathways = []
                    for line in resp2.text.strip().split('\n'):
                        parts = line.split('\t')
                        if len(parts) >= 2:
                            pw = parts[1].replace('path:', '')
                            if pw.startswith('hsa'):
                                pathways.append(pw)
                    gene_pathways[gene] = pathways
                else:
                    gene_pathways[gene] = []
            else:
                gene_pathways[gene] = []
                
        except Exception as e:
            gene_pathways[gene] = []
        
        time.sleep(0.35)  # KEGG rate limit: 3 requests/sec
    
    logger.info(f"  Mapped {sum(1 for v in gene_pathways.values() if v)} genes to pathways")
    return gene_pathways


# Cancer-related KEGG pathways (pre-defined list)
CANCER_PATHWAYS = {
    'hsa05200': 'Pathways_in_cancer',
    'hsa05210': 'Colorectal_cancer',
    'hsa05212': 'Pancreatic_cancer',
    'hsa05214': 'Glioma',
    'hsa05216': 'Thyroid_cancer',
    'hsa05220': 'Chronic_myeloid_leukemia',
    'hsa05221': 'Acute_myeloid_leukemia',
    'hsa05222': 'Small_cell_lung_cancer',
    'hsa05223': 'Non-small_cell_lung_cancer',
    'hsa05224': 'Breast_cancer',
    'hsa05225': 'Hepatocellular_carcinoma',
    'hsa05226': 'Gastric_cancer',
    'hsa05230': 'Central_carbon_metabolism_in_cancer',
    # Signaling pathways
    'hsa04010': 'MAPK_signaling',
    'hsa04012': 'ErbB_signaling',
    'hsa04014': 'Ras_signaling',
    'hsa04015': 'Rap1_signaling',
    'hsa04020': 'Calcium_signaling',
    'hsa04024': 'cAMP_signaling',
    'hsa04062': 'Chemokine_signaling',
    'hsa04064': 'NF-kB_signaling',
    'hsa04066': 'HIF-1_signaling',
    'hsa04068': 'FoxO_signaling',
    'hsa04110': 'Cell_cycle',
    'hsa04115': 'p53_signaling',
    'hsa04150': 'mTOR_signaling',
    'hsa04151': 'PI3K-Akt_signaling',
    'hsa04152': 'AMPK_signaling',
    'hsa04210': 'Apoptosis',
    'hsa04310': 'Wnt_signaling',
    'hsa04330': 'Notch_signaling',
    'hsa04340': 'Hedgehog_signaling',
    'hsa04350': 'TGF-beta_signaling',
    'hsa04370': 'VEGF_signaling',
    'hsa04390': 'Hippo_signaling',
    'hsa04510': 'Focal_adhesion',
    'hsa04520': 'Adherens_junction',
    'hsa04530': 'Tight_junction',
    # DNA repair
    'hsa03410': 'Base_excision_repair',
    'hsa03420': 'Nucleotide_excision_repair',
    'hsa03430': 'Mismatch_repair',
    'hsa03440': 'Homologous_recombination',
    'hsa03450': 'Non-homologous_end-joining',
    # Metabolism relevant to drugs
    'hsa00230': 'Purine_metabolism',
    'hsa00240': 'Pyrimidine_metabolism',
    'hsa00983': 'Drug_metabolism_other',
    'hsa01524': 'Platinum_drug_resistance',
}


def build_pathway_vectors(drug_targets, gene_pathways, pathway_list=None):
    """Build pathway activation vector for each drug."""
    if pathway_list is None:
        pathway_list = list(CANCER_PATHWAYS.keys())
    
    pw_idx = {pw: i for i, pw in enumerate(pathway_list)}
    n_pw = len(pathway_list)
    
    vectors = {}
    for drug, targets in drug_targets.items():
        vec = np.zeros(n_pw, dtype=np.float32)
        target_genes = [t['gene'] for t in targets]
        
        for gene in target_genes:
            if gene in gene_pathways:
                for pw in gene_pathways[gene]:
                    if pw in pw_idx:
                        vec[pw_idx[pw]] += 1.0
        
        # Normalize
        max_val = vec.max()
        if max_val > 0:
            vec = vec / max_val
        
        vectors[drug] = vec
    
    return vectors, pathway_list


def build_pathway_crosstalk(drug_a_targets, drug_b_targets, gene_pathways, pathway_list):
    """Compute pathway-level crosstalk between two drugs."""
    pw_idx = {pw: i for i, pw in enumerate(pathway_list)}
    n_pw = len(pathway_list)
    
    # Drug A pathway activation
    pw_a = np.zeros(n_pw)
    for gene in drug_a_targets:
        if gene in gene_pathways:
            for pw in gene_pathways[gene]:
                if pw in pw_idx:
                    pw_a[pw_idx[pw]] = 1
    
    # Drug B pathway activation
    pw_b = np.zeros(n_pw)
    for gene in drug_b_targets:
        if gene in gene_pathways:
            for pw in gene_pathways[gene]:
                if pw in pw_idx:
                    pw_b[pw_idx[pw]] = 1
    
    # Crosstalk: shared pathways
    shared = np.minimum(pw_a, pw_b)
    
    return {
        'shared_pathways': int(shared.sum()),
        'total_a': int(pw_a.sum()),
        'total_b': int(pw_b.sum()),
        'jaccard': float(shared.sum() / max(pw_a.sum() + pw_b.sum() - shared.sum(), 1)),
    }


def main():
    print("=" * 70)
    print("Drug Mechanism Feature Engineering")
    print("=" * 70)
    
    # Load drug names
    with open(MODEL_DIR / "drug_fingerprints.pkl", 'rb') as f:
        drug_fps = pickle.load(f)
    drug_names = list(drug_fps.keys())
    logger.info(f"Drugs: {len(drug_names)}")
    
    # Load verified SMILES
    smiles_file = DATA_DIR / "verified_drug_smiles.json"
    smiles_dict = {}
    if smiles_file.exists():
        with open(smiles_file) as f:
            smiles_dict = json.load(f)
    
    # ================================================================
    # Step 1: DGIdb Drug-Target Interactions
    # ================================================================
    print(f"\n{'='*70}")
    print("STEP 1: DGIdb Drug-Target Interactions")
    print("=" * 70)
    
    cache_file = MECH_DIR / "dgidb_targets.json"
    if cache_file.exists():
        logger.info("Loading cached DGIdb targets...")
        with open(cache_file) as f:
            drug_targets = json.load(f)
    else:
        drug_targets = get_dgidb_targets(drug_names)
        with open(cache_file, 'w') as f:
            json.dump(drug_targets, f, indent=2)
    
    # Build target vectors
    target_vectors, top_genes = build_target_vectors(drug_targets, top_n=200)
    
    drugs_with_targets = sum(1 for v in target_vectors.values() if v.sum() > 0)
    print(f"  Drugs with targets: {drugs_with_targets}/{len(drug_names)}")
    print(f"  Target genes: {len(top_genes)}")
    
    # ================================================================
    # Step 2: Binding Affinity (PubChem)
    # ================================================================
    print(f"\n{'='*70}")
    print("STEP 2: Binding Affinity (PubChem Bioassay)")
    print("=" * 70)
    
    cache_file2 = MECH_DIR / "pubchem_affinity.json"
    if cache_file2.exists():
        logger.info("Loading cached PubChem affinity...")
        with open(cache_file2) as f:
            drug_affinity = json.load(f)
    else:
        drug_affinity = get_pubchem_bioactivity(drug_names, smiles_dict)
        with open(cache_file2, 'w') as f:
            json.dump(drug_affinity, f, indent=2, default=str)
    
    affinity_vectors = build_affinity_features(drug_affinity, drug_targets, top_genes, n_features=50)
    drugs_with_affinity = sum(1 for v in affinity_vectors.values() if v.sum() > 0)
    print(f"  Drugs with affinity data: {drugs_with_affinity}/{len(drug_names)}")
    
    # ================================================================
    # Step 3: KEGG Pathway Features
    # ================================================================
    print(f"\n{'='*70}")
    print("STEP 3: KEGG Pathway Signaling")
    print("=" * 70)
    
    cache_file3 = MECH_DIR / "kegg_gene_pathways.json"
    if cache_file3.exists():
        logger.info("Loading cached KEGG pathway mappings...")
        with open(cache_file3) as f:
            gene_pathways = json.load(f)
    else:
        # Only query top target genes that appear in our drugs
        all_target_genes = set()
        for targets in drug_targets.values():
            for t in targets:
                all_target_genes.add(t['gene'])
        
        genes_to_query = list(all_target_genes)[:300]  # Limit API calls
        gene_pathways = get_kegg_pathways(genes_to_query)
        with open(cache_file3, 'w') as f:
            json.dump(gene_pathways, f, indent=2)
    
    pathway_list = list(CANCER_PATHWAYS.keys())
    pathway_vectors, _ = build_pathway_vectors(drug_targets, gene_pathways, pathway_list)
    
    drugs_with_pathways = sum(1 for v in pathway_vectors.values() if v.sum() > 0)
    print(f"  Cancer pathways tracked: {len(pathway_list)}")
    print(f"  Drugs with pathway data: {drugs_with_pathways}/{len(drug_names)}")
    
    # ================================================================
    # Combine and save
    # ================================================================
    print(f"\n{'='*70}")
    print("SAVING MECHANISM FEATURES")
    print("=" * 70)
    
    mechanism_features = {}
    for drug in drug_names:
        drug_upper = drug.upper()
        target_vec = target_vectors.get(drug, np.zeros(200, dtype=np.float32))
        affinity_vec = affinity_vectors.get(drug, np.zeros(50, dtype=np.float32))
        pathway_vec = pathway_vectors.get(drug, np.zeros(len(pathway_list), dtype=np.float32))
        
        combined = np.concatenate([target_vec, affinity_vec, pathway_vec])
        mechanism_features[drug_upper] = combined
    
    n_mech = len(next(iter(mechanism_features.values())))
    print(f"  Feature dimensions: {n_mech} (200 target + 50 affinity + {len(pathway_list)} pathway)")
    
    # Save
    with open(MODEL_DIR / "drug_mechanism_features.pkl", 'wb') as f:
        pickle.dump(mechanism_features, f)
    
    meta = {
        'n_drugs': len(mechanism_features),
        'n_features': n_mech,
        'n_target': 200,
        'n_affinity': 50,
        'n_pathway': len(pathway_list),
        'top_target_genes': top_genes[:20],
        'pathway_names': {k: v for k, v in CANCER_PATHWAYS.items()},
        'drugs_with_targets': drugs_with_targets,
        'drugs_with_affinity': drugs_with_affinity,
        'drugs_with_pathways': drugs_with_pathways,
    }
    with open(MODEL_DIR / "mechanism_metadata.json", 'w') as f:
        json.dump(meta, f, indent=2)
    
    print(f"\n  Saved: {MODEL_DIR / 'drug_mechanism_features.pkl'}")
    print(f"  Metadata: {MODEL_DIR / 'mechanism_metadata.json'}")
    
    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print("=" * 70)
    for drug in sorted(drug_names):
        du = drug.upper()
        tgt = target_vectors.get(drug, np.zeros(200)).sum()
        aff = affinity_vectors.get(drug, np.zeros(50)).sum()
        pw = pathway_vectors.get(drug, np.zeros(len(pathway_list))).sum()
        print(f"  {drug:25s}: targets={tgt:.0f}, affinity_score={aff:.1f}, pathway_coverage={pw:.2f}")


if __name__ == "__main__":
    main()
