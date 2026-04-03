"""
Download TCGA COAD/READ RNA-seq + Clinical Data
================================================
Uses GDC API to download:
- Gene Expression Quantification (STAR-Counts, FPKM)
- Clinical data including treatment and survival
"""
import requests
import json
import gzip
import os
import logging
import time
import pandas as pd
import numpy as np
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

DATA_DIR = Path("F:/ADDS/data/ml_training")
TCGA_DIR = DATA_DIR / "tcga_rnaseq"
TCGA_DIR.mkdir(parents=True, exist_ok=True)

GDC_API = "https://api.gdc.cancer.gov"


def get_tcga_file_ids(project_ids=["TCGA-COAD", "TCGA-READ"]):
    """Get file UUIDs for STAR-Counts gene expression files."""
    logger.info(f"Querying GDC for {project_ids}...")
    
    filters = {
        "op": "and",
        "content": [
            {"op": "in", "content": {"field": "cases.project.project_id", "value": project_ids}},
            {"op": "=", "content": {"field": "files.data_type", "value": "Gene Expression Quantification"}},
            {"op": "=", "content": {"field": "files.experimental_strategy", "value": "RNA-Seq"}},
            {"op": "=", "content": {"field": "files.analysis.workflow_type", "value": "STAR - Counts"}},
            {"op": "=", "content": {"field": "files.data_format", "value": "TSV"}},
        ]
    }
    
    params = {
        "filters": json.dumps(filters),
        "fields": "file_id,file_name,cases.case_id,cases.submitter_id,cases.project.project_id",
        "format": "JSON",
        "size": "1000",
    }
    
    resp = requests.get(f"{GDC_API}/files", params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    
    hits = data['data']['hits']
    logger.info(f"Found {len(hits)} files")
    
    file_info = []
    for hit in hits:
        cases = hit.get('cases', [{}])
        case = cases[0] if cases else {}
        file_info.append({
            'file_id': hit['file_id'],
            'file_name': hit['file_name'],
            'case_id': case.get('case_id', ''),
            'submitter_id': case.get('submitter_id', ''),
            'project': case.get('project', {}).get('project_id', ''),
        })
    
    return file_info


def download_expression_files(file_info, max_files=None):
    """Download gene expression TSV files from GDC."""
    if max_files:
        file_info = file_info[:max_files]
    
    logger.info(f"Downloading {len(file_info)} expression files...")
    
    downloaded = []
    for i, fi in enumerate(file_info):
        fid = fi['file_id']
        out_path = TCGA_DIR / f"{fi['submitter_id']}_{fid[:8]}.tsv.gz"
        
        if out_path.exists():
            downloaded.append(fi)
            continue
        
        try:
            resp = requests.get(f"{GDC_API}/data/{fid}", timeout=60,
                              headers={"Content-Type": "application/json"})
            
            if resp.status_code == 200:
                with open(out_path, 'wb') as f:
                    f.write(resp.content)
                downloaded.append(fi)
                
                if (i+1) % 20 == 0:
                    logger.info(f"  Downloaded {i+1}/{len(file_info)}")
            else:
                logger.warning(f"  {fi['submitter_id']}: HTTP {resp.status_code}")
            
            time.sleep(0.3)
            
        except Exception as e:
            logger.warning(f"  {fi['submitter_id']}: {e}")
    
    logger.info(f"Downloaded: {len(downloaded)}/{len(file_info)}")
    return downloaded


def get_clinical_data(project_ids=["TCGA-COAD", "TCGA-READ"]):
    """Get clinical and treatment data from GDC."""
    logger.info("Querying clinical data...")
    
    filters = {
        "op": "in",
        "content": {"field": "project.project_id", "value": project_ids}
    }
    
    fields = [
        "case_id", "submitter_id",
        "demographic.gender", "demographic.vital_status",
        "demographic.days_to_death", "demographic.age_at_index",
        "diagnoses.primary_diagnosis", "diagnoses.tumor_stage",
        "diagnoses.ajcc_pathologic_stage",
        "diagnoses.ajcc_pathologic_t",
        "diagnoses.ajcc_pathologic_n",
        "diagnoses.ajcc_pathologic_m",
        "diagnoses.days_to_last_follow_up",
        "diagnoses.days_to_recurrence",
        "diagnoses.treatments.treatment_type",
        "diagnoses.treatments.therapeutic_agents",
        "diagnoses.treatments.treatment_intent_type",
        "diagnoses.treatments.treatment_outcome",
    ]
    
    params = {
        "filters": json.dumps(filters),
        "fields": ",".join(fields),
        "format": "JSON",
        "size": "1000",
    }
    
    resp = requests.get(f"{GDC_API}/cases", params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    
    hits = data['data']['hits']
    logger.info(f"Clinical records: {len(hits)}")
    
    records = []
    for hit in hits:
        rec = {
            'case_id': hit.get('case_id', ''),
            'submitter_id': hit.get('submitter_id', ''),
        }
        
        demo = hit.get('demographic', {})
        rec['gender'] = demo.get('gender', '')
        rec['vital_status'] = demo.get('vital_status', '')
        rec['days_to_death'] = demo.get('days_to_death', None)
        rec['age_at_index'] = demo.get('age_at_index', None)
        
        diags = hit.get('diagnoses', [{}])
        diag = diags[0] if diags else {}
        rec['primary_diagnosis'] = diag.get('primary_diagnosis', '')
        rec['ajcc_stage'] = diag.get('ajcc_pathologic_stage', '')
        rec['ajcc_t'] = diag.get('ajcc_pathologic_t', '')
        rec['ajcc_n'] = diag.get('ajcc_pathologic_n', '')
        rec['ajcc_m'] = diag.get('ajcc_pathologic_m', '')
        rec['days_to_last_followup'] = diag.get('days_to_last_follow_up', None)
        rec['days_to_recurrence'] = diag.get('days_to_recurrence', None)
        
        treatments = diag.get('treatments', [])
        chemo_agents = []
        treatment_types = []
        treatment_outcomes = []
        
        for tx in treatments:
            tx_type = tx.get('treatment_type', '')
            treatment_types.append(tx_type)
            if tx.get('therapeutic_agents'):
                chemo_agents.append(tx['therapeutic_agents'])
            if tx.get('treatment_outcome'):
                treatment_outcomes.append(tx['treatment_outcome'])
        
        rec['treatment_types'] = '|'.join(treatment_types)
        rec['chemo_agents'] = '|'.join(chemo_agents)
        rec['treatment_outcomes'] = '|'.join(treatment_outcomes)
        rec['has_chemo'] = any('Chemother' in t or 'Pharmaceutical' in t for t in treatment_types)
        
        records.append(rec)
    
    return pd.DataFrame(records)


def parse_expression_matrix(file_info, clinical_df):
    """Parse downloaded expression files into a matrix."""
    logger.info("Parsing expression files...")
    
    expr_files = {}
    for fi in file_info:
        sid = fi['submitter_id']
        fid = fi['file_id']
        path = TCGA_DIR / f"{sid}_{fid[:8]}.tsv.gz"
        if path.exists():
            expr_files[sid] = path
    
    logger.info(f"Available expression files: {len(expr_files)}")
    
    if not expr_files:
        logger.warning("No expression files found!")
        return None
    
    first_path = list(expr_files.values())[0]
    try:
        if str(first_path).endswith('.gz'):
            first_df = pd.read_csv(first_path, sep='\t', compression='gzip', comment='#')
        else:
            first_df = pd.read_csv(first_path, sep='\t', comment='#')
        
        logger.info(f"First file columns: {list(first_df.columns)}")
        logger.info(f"First file shape: {first_df.shape}")
        logger.info(f"First 5 rows:\n{first_df.head()}")
        
        if 'gene_name' in first_df.columns:
            gene_col = 'gene_name'
            if 'fpkm_unstranded' in first_df.columns:
                value_col = 'fpkm_unstranded'
            elif 'tpm_unstranded' in first_df.columns:
                value_col = 'tpm_unstranded'
            elif 'unstranded' in first_df.columns:
                value_col = 'unstranded'
            else:
                value_col = first_df.columns[1]
        else:
            gene_col = first_df.columns[0]
            value_col = first_df.columns[1]
        
        logger.info(f"Using gene_col={gene_col}, value_col={value_col}")
        
        mask = ~first_df[gene_col].str.startswith('_', na=True)
        if 'gene_type' in first_df.columns:
            mask = mask & (first_df['gene_type'] == 'protein_coding')
        
        gene_names = first_df.loc[mask, gene_col].values
        logger.info(f"Protein-coding genes: {len(gene_names)}")
        
    except Exception as e:
        logger.error(f"Error reading first file: {e}")
        return None
    
    expr_data = {}
    for sid, path in expr_files.items():
        try:
            if str(path).endswith('.gz'):
                df = pd.read_csv(path, sep='\t', compression='gzip', comment='#')
            else:
                df = pd.read_csv(path, sep='\t', comment='#')
            
            if gene_col in df.columns and value_col in df.columns:
                mask = ~df[gene_col].str.startswith('_', na=True)
                if 'gene_type' in df.columns:
                    mask = mask & (df['gene_type'] == 'protein_coding')
                
                values = df.loc[mask, value_col].values
                if len(values) == len(gene_names):
                    expr_data[sid] = values.astype(np.float32)
        except Exception as e:
            logger.warning(f"Error reading {sid}: {e}")
    
    logger.info(f"Parsed expression for {len(expr_data)} samples")
    
    if expr_data:
        samples = sorted(expr_data.keys())
        expr_matrix = np.array([expr_data[s] for s in samples])
        expr_matrix = np.log2(expr_matrix + 1)
        
        logger.info(f"Expression matrix: {expr_matrix.shape}")
        return samples, gene_names, expr_matrix
    
    return None


def main():
    print("=" * 70)
    print("TCGA COAD/READ RNA-seq Download")
    print("=" * 70)
    
    # Step 1: Clinical data
    print(f"\n{'='*70}")
    print("STEP 1: Clinical Data")
    print("=" * 70)
    
    clinical_cache = TCGA_DIR / "clinical.csv"
    if clinical_cache.exists():
        clinical_df = pd.read_csv(clinical_cache)
        logger.info(f"Loaded cached clinical: {len(clinical_df)}")
    else:
        clinical_df = get_clinical_data()
        clinical_df.to_csv(clinical_cache, index=False)
    
    print(f"  Total patients: {len(clinical_df)}")
    print(f"  With chemo: {clinical_df['has_chemo'].sum()}")
    if 'vital_status' in clinical_df.columns:
        print(f"  Vital status:")
        for k, v in clinical_df['vital_status'].value_counts().items():
            print(f"    {k}: {v}")
    if 'chemo_agents' in clinical_df.columns:
        from collections import Counter
        agents = clinical_df['chemo_agents'].dropna()
        agents = agents[agents != '']
        if len(agents) > 0:
            print(f"  Chemo agents:")
            agent_counts = Counter()
            for a in agents:
                for x in a.split('|'):
                    x = x.strip()
                    if x:
                        agent_counts[x] += 1
            for agent, count in agent_counts.most_common(15):
                print(f"    {agent}: {count}")
    
    # Step 2: File IDs
    print(f"\n{'='*70}")
    print("STEP 2: Expression File IDs")
    print("=" * 70)
    
    file_cache = TCGA_DIR / "file_ids.json"
    if file_cache.exists():
        with open(file_cache) as f:
            file_info = json.load(f)
        logger.info(f"Loaded cached file IDs: {len(file_info)}")
    else:
        file_info = get_tcga_file_ids()
        with open(file_cache, 'w') as f:
            json.dump(file_info, f, indent=2)
    
    print(f"  Total expression files: {len(file_info)}")
    
    # Step 3: Download
    print(f"\n{'='*70}")
    print("STEP 3: Download Expression Files")
    print("=" * 70)
    
    clinical_sids = set(clinical_df['submitter_id'].values)
    matched_files = [fi for fi in file_info if fi['submitter_id'] in clinical_sids]
    print(f"  Files matching clinical data: {len(matched_files)}")
    
    downloaded = download_expression_files(matched_files)
    print(f"  Downloaded: {len(downloaded)}")
    
    # Step 4: Parse expression
    print(f"\n{'='*70}")
    print("STEP 4: Parse Expression Matrix")
    print("=" * 70)
    
    result = parse_expression_matrix(downloaded, clinical_df)
    if result:
        samples, genes, expr = result
        print(f"  Samples: {len(samples)}")
        print(f"  Genes: {len(genes)}")
        print(f"  Matrix: {expr.shape}")
        
        np.savez_compressed(
            TCGA_DIR / "tcga_crc_expression.npz",
            samples=samples,
            genes=genes,
            expression=expr,
        )
        print(f"  Saved: {TCGA_DIR / 'tcga_crc_expression.npz'}")
        
        clinical_matched = clinical_df[clinical_df['submitter_id'].isin(samples)]
        print(f"\n  Clinical-expression matched: {len(clinical_matched)}")
        print(f"  With chemo: {clinical_matched['has_chemo'].sum()}")
    else:
        print("  No expression data available")


if __name__ == "__main__":
    main()
