"""
Priority 4 Fix: ChEMBL pKi Automation (v2 — robust timeout)
=============================================================
Fetch verified pKi from ChEMBL for O'Neil drugs.
Uses short timeouts + single-page queries to avoid hanging.
"""
import requests
import json
import time
import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

MODEL_DIR = Path("F:/ADDS/models/synergy")

TARGET_GENES = [
    'ABL1', 'AKT1', 'AKT2', 'AKT3', 'AURKA', 'BRAF', 'CDK1', 'CDK2',
    'CDK5', 'CDK9', 'CHEK1', 'DHFR', 'EGFR', 'EPHA2', 'ERBB2', 'FLT3',
    'HDAC1', 'HDAC2', 'HDAC3', 'HSP90AA1', 'KIT', 'MAP2K1', 'MAP2K2',
    'MTOR', 'PARP1', 'PARP2', 'PDGFRA', 'PDGFRB', 'PIK3CA', 'PIK3CB',
    'PSMB5', 'RAF1', 'RET', 'SRC', 'TOP1', 'TOP2A', 'TOP2B', 'TUBB',
    'TYMS', 'KDR',
]

# Drug → ChEMBL ID (pre-verified)
DRUGS = {
    'ERLOTINIB': 'CHEMBL553',
    'LAPATINIB': 'CHEMBL554',
    'SORAFENIB': 'CHEMBL1336',
    'SUNITINIB': 'CHEMBL535',
    'DASATINIB': 'CHEMBL1421',
    'BORTEZOMIB': 'CHEMBL325041',
    'PACLITAXEL': 'CHEMBL428647',
    'VINBLASTINE': 'CHEMBL119',
    'DOXORUBICIN': 'CHEMBL53463',
    'ETOPOSIDE': 'CHEMBL44657',
    'TOPOTECAN': 'CHEMBL84',
    'GEFITINIB': 'CHEMBL939',
    'GEMCITABINE': 'CHEMBL888',
    'METHOTREXATE': 'CHEMBL34259',
    'CISPLATIN': 'CHEMBL11359',
    'TEMOZOLOMIDE': 'CHEMBL810',
    '5-FU': 'CHEMBL185',
    'ZOLINZA': 'CHEMBL98',
    'ABT-888': 'CHEMBL501849',
    'MK-4827': 'CHEMBL1278959',
}

# Manual pKi for comparison
MANUAL_PKI = {
    'ERLOTINIB': {'EGFR': 8.7, 'ABL1': 5.9},
    'LAPATINIB': {'EGFR': 8.0, 'ERBB2': 8.0},
    'SORAFENIB': {'BRAF': 7.7, 'FLT3': 7.2, 'KIT': 7.2, 'PDGFRB': 7.2, 'RAF1': 8.2, 'KDR': 7.0},
    'DASATINIB': {'ABL1': 9.2, 'EPHA2': 7.8, 'KIT': 8.3, 'PDGFRB': 7.6, 'SRC': 9.3},
    'BORTEZOMIB': {'PSMB5': 9.2},
    'PACLITAXEL': {'TUBB': 8.4},
    'VINBLASTINE': {'TUBB': 9.0},
    'DOXORUBICIN': {'TOP2A': 6.8},
    'ETOPOSIDE': {'TOP2A': 5.7, 'TOP2B': 5.5},
    'TOPOTECAN': {'TOP1': 6.5},
    'MK-4827': {'PARP1': 8.4, 'PARP2': 8.7},
    'ABT-888': {'PARP1': 8.3, 'PARP2': 8.5},
}


def query_chembl_molecule(drug_name, chembl_id):
    """Query ChEMBL for a single drug with timeout protection."""
    url = (f"https://www.ebi.ac.uk/chembl/api/data/activity.json?"
           f"molecule_chembl_id={chembl_id}"
           f"&pchembl_value__isnull=false"
           f"&limit=100")

    try:
        resp = requests.get(url, timeout=8)
        if resp.status_code == 200:
            data = resp.json()
            activities = data.get('activities', [])
            return activities
    except requests.Timeout:
        logger.warning(f"  {drug_name}: timeout")
    except Exception as e:
        logger.warning(f"  {drug_name}: {e}")
    return []


def match_activities_to_genes(activities, target_genes):
    """Find best pKi for each target gene."""
    best = {}
    for act in activities:
        target = act.get('target_pref_name', '').upper()
        pchembl = act.get('pchembl_value')
        std_type = act.get('standard_type', '')

        if not pchembl:
            continue
        pchembl = float(pchembl)

        for gene in target_genes:
            if gene.upper() in target:
                if gene not in best or pchembl > best[gene]['pKi']:
                    best[gene] = {
                        'pKi': pchembl,
                        'type': std_type,
                        'target_full': act.get('target_pref_name', ''),
                    }
    return best


def main():
    print("=" * 70)
    print("Priority 4: ChEMBL pKi Automation (v2)")
    print("=" * 70)
    sys.stdout.flush()

    all_results = {}
    success = 0
    failed = 0

    for drug, chembl_id in DRUGS.items():
        print(f"  {drug:20s} ({chembl_id})... ", end='', flush=True)
        acts = query_chembl_molecule(drug, chembl_id)

        if acts:
            matched = match_activities_to_genes(acts, TARGET_GENES)
            all_results[drug] = {
                'chembl_id': chembl_id,
                'n_activities': len(acts),
                'targets': {g: info for g, info in matched.items()},
            }
            targets_str = ', '.join([f"{g}={v['pKi']:.1f}" for g, v in sorted(matched.items())])
            print(f"{len(acts)} acts → {len(matched)} targets: {targets_str}")
            success += 1
        else:
            all_results[drug] = {'chembl_id': chembl_id, 'n_activities': 0, 'targets': {}}
            print("NO DATA")
            failed += 1

        sys.stdout.flush()
        time.sleep(0.5)

    # Summary
    print(f"\n{'='*70}")
    print(f"RESULTS: {success}/{len(DRUGS)} drugs got data, {failed} failed")
    print("=" * 70)

    # Build clean affinities dict
    chembl_affinities = {}
    for drug, data in all_results.items():
        if data['targets']:
            chembl_affinities[drug] = {g: v['pKi'] for g, v in data['targets'].items()}

    # Compare with manual
    print(f"\n  Manual vs ChEMBL pKi Comparison:")
    print(f"  {'Drug':15s} {'Gene':10s} {'Manual':>8s} {'ChEMBL':>8s} {'Delta':>8s}")
    print(f"  {'-'*15} {'-'*10} {'-'*8} {'-'*8} {'-'*8}")

    n_match = 0
    n_mismatch = 0
    for drug, manual in MANUAL_PKI.items():
        chembl = chembl_affinities.get(drug, {})
        for gene, m_pki in manual.items():
            c_pki = chembl.get(gene, None)
            if c_pki is not None:
                delta = c_pki - m_pki
                flag = '✓' if abs(delta) < 1.5 else '⚠'
                if abs(delta) < 1.5:
                    n_match += 1
                else:
                    n_mismatch += 1
                print(f"  {drug:15s} {gene:10s} {m_pki:>8.1f} {c_pki:>8.1f} {delta:>+8.1f} {flag}")
            else:
                print(f"  {drug:15s} {gene:10s} {m_pki:>8.1f} {'N/A':>8s} {'---':>8s}")

    print(f"\n  Verified: {n_match}, Discrepant (>1.5): {n_mismatch}")

    # Save
    output = {
        'source': 'ChEMBL REST API',
        'date': '2026-02-16',
        'results': {},
        'affinities': chembl_affinities,
    }
    for drug, data in all_results.items():
        output['results'][drug] = {
            'chembl_id': data['chembl_id'],
            'n_activities': data['n_activities'],
            'targets': {g: {'pKi': v['pKi'], 'type': v.get('type', ''),
                           'full_target': v.get('target_full', '')}
                       for g, v in data.get('targets', {}).items()},
        }

    out_path = MODEL_DIR / "chembl_pki_verified.json"
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\n  Saved: {out_path}")


if __name__ == "__main__":
    main()
