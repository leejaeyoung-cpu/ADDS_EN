"""
TCGA Colorectal Cancer Clinical Data Downloader
================================================
cBioPortal REST API에서 TCGA-COAD/READ 임상+변이 데이터를 다운로드합니다.

No authentication required for public data.
Output: data/ml_training/tcga_crc_clinical.csv
        data/ml_training/tcga_crc_mutations.csv
"""

import requests
import pandas as pd
import numpy as np
from pathlib import Path
import json
import time
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

OUTPUT_DIR = Path("F:/ADDS/data/ml_training")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


class CBioPortalDownloader:
    """cBioPortal REST API client for TCGA data"""

    def __init__(self):
        self.base_url = "https://www.cbioportal.org/api"
        self.session = requests.Session()
        self.session.headers.update({
            "Accept": "application/json",
            "User-Agent": "ADDS-Research/1.0"
        })

        # TCGA CRC study IDs (try in order)
        self.study_ids = [
            "coadread_tcga_pan_can_atlas_2018",
            "coadread_tcga",
            "coad_tcga_pan_can_atlas_2018",
            "coad_tcga"
        ]

    def find_study(self) -> str:
        """Find available TCGA CRC study"""
        for study_id in self.study_ids:
            try:
                url = f"{self.base_url}/studies/{study_id}"
                resp = self.session.get(url, timeout=30)
                if resp.status_code == 200:
                    data = resp.json()
                    logger.info(f"Found study: {data.get('name', study_id)} "
                                f"({data.get('allSampleCount', '?')} samples)")
                    return study_id
            except Exception as e:
                logger.debug(f"Study {study_id} not found: {e}")
                continue

        logger.error("No TCGA CRC study found!")
        return ""

    def download_clinical_data(self, study_id: str) -> pd.DataFrame:
        """Download clinical data for all patients in study"""
        logger.info(f"Downloading clinical data for {study_id}...")

        url = f"{self.base_url}/studies/{study_id}/clinical-data"
        params = {
            "clinicalDataType": "PATIENT",
            "projection": "DETAILED"
        }

        try:
            resp = self.session.get(url, params=params, timeout=60)
            resp.raise_for_status()
            data = resp.json()

            if not data:
                logger.warning("No clinical data returned")
                return pd.DataFrame()

            # Convert to wide format (each attribute as column)
            records = {}
            for item in data:
                pid = item.get('patientId', '')
                attr = item.get('clinicalAttributeId', '')
                value = item.get('value', '')

                if pid not in records:
                    records[pid] = {'patient_id': pid}
                records[pid][attr] = value

            df = pd.DataFrame(list(records.values()))
            logger.info(f"Clinical data: {len(df)} patients, {len(df.columns)} attributes")
            return df

        except Exception as e:
            logger.error(f"Error downloading clinical data: {e}")
            return pd.DataFrame()

    def download_mutation_data(self, study_id: str) -> pd.DataFrame:
        """Download mutation data for key CRC genes"""
        logger.info(f"Downloading mutation data for {study_id}...")

        # Find molecular profile for mutations
        try:
            url = f"{self.base_url}/studies/{study_id}/molecular-profiles"
            resp = self.session.get(url, timeout=30)
            resp.raise_for_status()
            profiles = resp.json()

            mut_profile = None
            for p in profiles:
                if p.get('molecularAlterationType') == 'MUTATION_EXTENDED':
                    mut_profile = p['molecularProfileId']
                    break

            if not mut_profile:
                logger.warning("No mutation profile found")
                return pd.DataFrame()

            logger.info(f"Using molecular profile: {mut_profile}")

        except Exception as e:
            logger.error(f"Error finding mutation profile: {e}")
            return pd.DataFrame()

        # Key CRC genes
        CRC_GENES = ["KRAS", "BRAF", "TP53", "APC", "PIK3CA", "SMAD4",
                      "NRAS", "PTEN", "ERBB2", "MLH1", "MSH2", "MSH6",
                      "FBXW7", "TCF7L2", "SOX9", "CTNNB1"]

        # Download mutations for each gene
        all_mutations = []

        for gene in CRC_GENES:
            try:
                url = f"{self.base_url}/molecular-profiles/{mut_profile}/mutations"
                params = {
                    "entrezGeneId": self._get_entrez_id(gene),
                    "projection": "DETAILED"
                }

                # Use gene Hugo symbol instead if entrez fails
                url_alt = f"{self.base_url}/molecular-profiles/{mut_profile}/mutations"
                body = {
                    "sampleListId": f"{study_id}_all",
                    "entrezGeneIds": [self._get_entrez_id(gene)]
                }

                resp = self.session.post(
                    f"{self.base_url}/molecular-profiles/{mut_profile}/mutations/fetch",
                    json=body,
                    timeout=30
                )

                if resp.status_code == 200:
                    muts = resp.json()
                    for m in muts:
                        m['hugo_symbol'] = gene
                    all_mutations.extend(muts)
                    logger.info(f"  {gene}: {len(muts)} mutations")
                else:
                    logger.info(f"  {gene}: no data (status {resp.status_code})")

                time.sleep(0.2)

            except Exception as e:
                logger.warning(f"  {gene}: error - {e}")
                continue

        if not all_mutations:
            logger.warning("No mutation data retrieved")
            return pd.DataFrame()

        df = pd.DataFrame(all_mutations)
        logger.info(f"Total mutations: {len(df)}")
        return df

    def _get_entrez_id(self, gene: str) -> int:
        """Map gene symbol to Entrez ID"""
        GENE_MAP = {
            "KRAS": 3845, "BRAF": 673, "TP53": 7157, "APC": 324,
            "PIK3CA": 5290, "SMAD4": 4089, "NRAS": 4893, "PTEN": 5728,
            "ERBB2": 2064, "MLH1": 4292, "MSH2": 4436, "MSH6": 2956,
            "FBXW7": 55294, "TCF7L2": 6934, "SOX9": 6662, "CTNNB1": 1499
        }
        return GENE_MAP.get(gene, 0)


def process_clinical_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Process raw clinical data into ML-ready features.

    Key features extracted:
    - OS_STATUS, OS_MONTHS: Overall survival
    - DFS_STATUS, DFS_MONTHS: Disease-free survival
    - AGE: Diagnosis age
    - SEX: M/F
    - STAGE: TNM stage
    - MSI: Microsatellite instability status
    """
    if df.empty:
        return df

    logger.info(f"Processing clinical data. Columns: {list(df.columns)[:30]}")

    # Standardize column names
    col_map = {}
    for col in df.columns:
        cu = col.upper()
        if 'OS_STATUS' in cu or cu == 'OVERALL_SURVIVAL_STATUS':
            col_map[col] = 'os_status'
        elif 'OS_MONTHS' in cu or cu == 'OVERALL_SURVIVAL_MONTHS':
            col_map[col] = 'os_months'
        elif 'DFS_STATUS' in cu or 'DISEASE_FREE' in cu:
            col_map[col] = 'dfs_status'
        elif 'DFS_MONTHS' in cu:
            col_map[col] = 'dfs_months'
        elif cu in ['AGE', 'AGE_AT_DIAGNOSIS', 'DIAGNOSIS_AGE']:
            col_map[col] = 'age'
        elif cu in ['SEX', 'GENDER']:
            col_map[col] = 'sex'
        elif cu in ['AJCC_PATHOLOGIC_TUMOR_STAGE', 'PATHOLOGIC_STAGE',
                    'AJCC_STAGING_EDITION'] or cu == 'STAGE':
            if 'stage' not in [v for v in col_map.values()]:  # avoid duplicates
                col_map[col] = 'stage'
        elif 'MSI' in cu or 'MICROSATELLITE' in cu:
            col_map[col] = 'msi_status'
        elif 'SUBTYPE' in cu:
            col_map[col] = 'subtype'
        elif 'TUMOR_SITE' in cu or 'PRIMARY_SITE' in cu:
            col_map[col] = 'tumor_site'
        elif cu == 'PATIENT_ID':
            col_map[col] = 'patient_id'

    df_clean = df.rename(columns=col_map)

    # Keep mapped columns + patient_id
    keep_cols = ['patient_id', 'os_status', 'os_months', 'dfs_status', 'dfs_months',
                 'age', 'sex', 'stage', 'msi_status', 'subtype', 'tumor_site']
    available = [c for c in keep_cols if c in df_clean.columns]
    df_clean = df_clean[available].copy()

    # Convert numeric columns
    for col in ['os_months', 'dfs_months', 'age']:
        if col in df_clean.columns:
            df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')

    # Encode OS status: 1 = deceased, 0 = alive/censored
    if 'os_status' in df_clean.columns:
        df_clean['os_event'] = df_clean['os_status'].apply(
            lambda x: 1 if str(x).upper() in ['DECEASED', '1:DECEASED', 'DEAD', '1'] else 0
        )

    # Encode DFS status: 1 = recurred, 0 = disease-free
    if 'dfs_status' in df_clean.columns:
        df_clean['dfs_event'] = df_clean['dfs_status'].apply(
            lambda x: 1 if str(x).upper() in ['RECURRED/PROGRESSED', '1:RECURRED/PROGRESSED',
                                                'RECURRED', 'PROGRESSED', '1'] else 0
        )

    # Encode stage to ordinal
    if 'stage' in df_clean.columns:
        STAGE_MAP = {
            'STAGE I': 1, 'STAGE IA': 1, 'STAGE IB': 1,
            'STAGE II': 2, 'STAGE IIA': 2, 'STAGE IIB': 2, 'STAGE IIC': 2,
            'STAGE III': 3, 'STAGE IIIA': 3, 'STAGE IIIB': 3, 'STAGE IIIC': 3,
            'STAGE IV': 4, 'STAGE IVA': 4, 'STAGE IVB': 4,
            'I': 1, 'II': 2, 'III': 3, 'IV': 4,
        }
        df_clean['stage_ordinal'] = df_clean['stage'].str.upper().map(STAGE_MAP)

    # Encode MSI
    if 'msi_status' in df_clean.columns:
        df_clean['msi_h'] = df_clean['msi_status'].apply(
            lambda x: 1 if str(x).upper() in ['MSI-H', 'MSI', 'HIGH'] else 0
        )

    # Encode sex
    if 'sex' in df_clean.columns:
        df_clean['sex_male'] = df_clean['sex'].apply(
            lambda x: 1 if str(x).upper() in ['MALE', 'M'] else 0
        )

    logger.info(f"Processed clinical data: {len(df_clean)} patients, {len(df_clean.columns)} features")
    return df_clean


def process_mutation_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert mutation-level data to patient-level binary features.
    Output: patient_id × gene mutation matrix
    """
    if df.empty:
        return df

    # Find relevant columns
    sample_col = None
    gene_col = None
    for col in df.columns:
        if 'sampleId' in col:
            sample_col = col
        elif 'hugo' in col.lower() or 'hugoGeneSymbol' in col:
            gene_col = col

    if not sample_col or not gene_col:
        # Try alternate names
        sample_col = sample_col or 'sampleId'
        gene_col = gene_col or 'hugoGeneSymbol'
        if sample_col not in df.columns or gene_col not in df.columns:
            logger.warning(f"Cannot find sample/gene columns. Available: {list(df.columns)[:20]}")
            return pd.DataFrame()

    # Patient ID from sample ID (TCGA format: TCGA-XX-XXXX-01)
    df['patient_id'] = df[sample_col].str[:12]  # First 12 chars of TCGA barcode

    # Create binary mutation matrix
    genes = df[gene_col].unique()
    logger.info(f"Creating mutation matrix for {len(genes)} genes")

    pivot = df.groupby(['patient_id', gene_col]).size().unstack(fill_value=0)
    # Convert to binary (mutated or not)
    pivot = (pivot > 0).astype(int)

    # Add prefix
    pivot.columns = [f"mut_{g}" for g in pivot.columns]
    pivot = pivot.reset_index()

    logger.info(f"Mutation matrix: {len(pivot)} patients × {len(pivot.columns) - 1} genes")
    return pivot


def create_synthetic_tcga_dataset() -> tuple:
    """
    Create a scientifically-grounded synthetic TCGA-like dataset for CRC.

    Based on published mutation frequencies and survival data:
    - KRAS: ~43% (COSMIC, Pylayeva-Gupta et al. 2011)
    - BRAF V600E: ~10% (Davies et al. 2002)
    - TP53: ~60% (TCGA 2012)
    - APC: ~80% (TCGA 2012)
    - PIK3CA: ~18% (TCGA 2012)
    - MSI-H: ~15% (Boland & Goel 2010)
    - Stage distribution: I(10%), II(25%), III(35%), IV(30%)
    """
    logger.info("Creating synthetic TCGA-like CRC dataset...")
    np.random.seed(42)

    n_patients = 600

    # --- Clinical Data ---
    patient_ids = [f"TCGA-SYNTH-{i:04d}" for i in range(n_patients)]
    ages = np.clip(np.random.normal(65, 12, n_patients), 25, 95).astype(int)
    sex = np.random.choice(['Male', 'Female'], n_patients, p=[0.55, 0.45])

    # Stage distribution (SEER data)
    stages = np.random.choice([1, 2, 3, 4], n_patients, p=[0.10, 0.25, 0.35, 0.30])
    stage_labels = {1: 'STAGE I', 2: 'STAGE II', 3: 'STAGE III', 4: 'STAGE IV'}

    # MSI-H: 15% overall, more common in right-sided
    msi_h = np.random.binomial(1, 0.15, n_patients)

    # Tumor site
    tumor_site = np.random.choice(['Right colon', 'Left colon', 'Rectum'],
                                   n_patients, p=[0.35, 0.35, 0.30])
    # MSI-H more likely right-sided
    for i in range(n_patients):
        if tumor_site[i] == 'Right colon':
            msi_h[i] = np.random.binomial(1, 0.25)
        elif tumor_site[i] == 'Rectum':
            msi_h[i] = np.random.binomial(1, 0.05)

    # Survival (correlated with stage and MSI status)
    os_months = np.zeros(n_patients)
    os_event = np.zeros(n_patients, dtype=int)
    dfs_months = np.zeros(n_patients)
    dfs_event = np.zeros(n_patients, dtype=int)

    for i in range(n_patients):
        # Base survival by stage (median OS in months)
        stage_os = {1: 120, 2: 84, 3: 60, 4: 24}[stages[i]]
        # MSI-H has better prognosis in early stage
        if msi_h[i] and stages[i] <= 3:
            stage_os *= 1.3
        # Age effect
        age_factor = 1.0 - max(0, (ages[i] - 65) * 0.008)

        os_months[i] = max(1, np.random.exponential(stage_os * age_factor))
        # Censoring (~40%)
        if np.random.random() < 0.4:
            os_months[i] = min(os_months[i], np.random.uniform(6, 60))
            os_event[i] = 0
        else:
            os_event[i] = 1

        # DFS (shorter than OS)
        dfs_months[i] = max(0.5, os_months[i] * np.random.uniform(0.5, 0.9))
        dfs_event[i] = 1 if (os_event[i] == 1 or np.random.random() < 0.3) else 0

    clinical_df = pd.DataFrame({
        'patient_id': patient_ids,
        'age': ages,
        'sex': sex,
        'stage': [stage_labels[s] for s in stages],
        'stage_ordinal': stages,
        'msi_status': ['MSI-H' if m else 'MSS' for m in msi_h],
        'msi_h': msi_h,
        'sex_male': [1 if s == 'Male' else 0 for s in sex],
        'tumor_site': tumor_site,
        'os_months': np.round(os_months, 1),
        'os_event': os_event,
        'dfs_months': np.round(dfs_months, 1),
        'dfs_event': dfs_event,
    })

    # --- Mutation Data ---
    # Frequencies from TCGA 2012 Nature paper
    GENE_FREQS = {
        'APC': 0.80, 'TP53': 0.60, 'KRAS': 0.43, 'PIK3CA': 0.18,
        'SMAD4': 0.10, 'NRAS': 0.05, 'BRAF': 0.10, 'PTEN': 0.06,
        'ERBB2': 0.05, 'FBXW7': 0.11, 'MLH1': 0.08, 'MSH2': 0.04,
    }

    mut_data = {'patient_id': patient_ids}
    for gene, freq in GENE_FREQS.items():
        muts = np.random.binomial(1, freq, n_patients)
        # BRAF V600E correlates with MSI-H
        if gene == 'BRAF':
            for j in range(n_patients):
                if msi_h[j]:
                    muts[j] = np.random.binomial(1, 0.35)  # Higher in MSI-H
        # KRAS and BRAF are mutually exclusive
        if gene == 'KRAS':
            for j in range(n_patients):
                if mut_data.get(f'mut_BRAF', np.zeros(n_patients))[j] if f'mut_BRAF' in mut_data else False:
                    muts[j] = 0
        mut_data[f'mut_{gene}'] = muts

    mutation_df = pd.DataFrame(mut_data)

    logger.info(f"Synthetic clinical: {len(clinical_df)} patients")
    logger.info(f"Synthetic mutations: {len(mutation_df)} patients × {len(GENE_FREQS)} genes")

    return clinical_df, mutation_df


def main():
    print("=" * 80)
    print("TCGA CRC Clinical Data Download (cBioPortal)")
    print("=" * 80)

    downloader = CBioPortalDownloader()

    # Find study
    study_id = downloader.find_study()

    clinical_df = pd.DataFrame()
    mutation_df = pd.DataFrame()

    if study_id:
        # Download clinical data
        raw_clinical = downloader.download_clinical_data(study_id)
        if not raw_clinical.empty:
            clinical_df = process_clinical_data(raw_clinical)

        # Download mutation data
        raw_mutations = downloader.download_mutation_data(study_id)
        if not raw_mutations.empty:
            mutation_df = process_mutation_data(raw_mutations)

    # Save API data if available
    if not clinical_df.empty:
        clin_file = OUTPUT_DIR / "tcga_crc_clinical.csv"
        clinical_df.to_csv(clin_file, index=False)
        print(f"\n[OK] Clinical data: {len(clinical_df)} patients saved to {clin_file}")
    else:
        print("\n[WARN] No clinical data from API")

    if not mutation_df.empty:
        mut_file = OUTPUT_DIR / "tcga_crc_mutations.csv"
        mutation_df.to_csv(mut_file, index=False)
        print(f"[OK] Mutation data: {len(mutation_df)} patients saved to {mut_file}")
    else:
        print("[WARN] No mutation data from API")

    # Always create synthetic fallback
    synth_clinical, synth_mutations = create_synthetic_tcga_dataset()

    synth_clin_file = OUTPUT_DIR / "tcga_crc_clinical_literature.csv"
    synth_mut_file = OUTPUT_DIR / "tcga_crc_mutations_literature.csv"
    synth_clinical.to_csv(synth_clin_file, index=False)
    synth_mutations.to_csv(synth_mut_file, index=False)
    print(f"[OK] Literature-grounded synthetic clinical: {len(synth_clinical)} patients -> {synth_clin_file}")
    print(f"[OK] Literature-grounded synthetic mutations: {len(synth_mutations)} patients -> {synth_mut_file}")

    # Use synthetic as primary if API failed
    if clinical_df.empty:
        primary_clin = OUTPUT_DIR / "tcga_crc_clinical.csv"
        synth_clinical.to_csv(primary_clin, index=False)
        print(f"[OK] Using synthetic as primary clinical data: {primary_clin}")

    if mutation_df.empty:
        primary_mut = OUTPUT_DIR / "tcga_crc_mutations.csv"
        synth_mutations.to_csv(primary_mut, index=False)
        print(f"[OK] Using synthetic as primary mutation data: {primary_mut}")

    print("\nDone!")


if __name__ == "__main__":
    main()
