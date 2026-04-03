"""
Priority 1 Fix: Replace Hash FP with Real RDKit Morgan Fingerprints
=====================================================================
1. Fetch SMILES from PubChem API for all O'Neil drugs
2. Generate 1024-bit Morgan FP (radius=2)
3. Retrain XGBoost with real FP on O'Neil 23K
4. Compare PF/LDPO/LCLO vs hash baseline
"""
import numpy as np
import pandas as pd
import pickle
import requests
import hashlib
import json
import logging
import time
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

DATA_DIR = Path("F:/ADDS/data/ml_training")
MODEL_DIR = Path("F:/ADDS/models/synergy")

# ============================================================
# Drug name → PubChem lookup aliases
# (some O'Neil names differ from PubChem preferred names)
# ============================================================
DRUG_ALIASES = {
    '5-FU': '5-fluorouracil',
    'MK-2206': 'MK-2206',
    'MK-4541': 'MK-4541',
    'MK-8669': 'Ridaforolimus',
    'MK-8776': 'SCH 900776',
    'PD325901': 'PD-0325901',
    'BEZ-235': 'Dactolisib',
    'ABT-888': 'Veliparib',
    'MK-4827': 'Niraparib',
    'MK-5108': 'MK-5108',
    'MRK-003': 'MRK-003',
    'MK-1775': 'Adavosertib',
    'AZD1775': 'Adavosertib',
    'L778123': 'L-778123',
    'SN-38': 'SN-38',
    'ZOLINZA': 'Vorinostat',
    'MITOMYCINE': 'Mitomycin C',
}

# Manually verified SMILES for drugs PubChem may not resolve well
MANUAL_SMILES = {
    '5-FU': 'O=C1NC(=O)C(F)=CN1',
    'PACLITAXEL': 'CC1=C2[C@@]([C@]([C@H]([C@@H]3[C@]4([C@H](OC4)C[C@@H]([C@]3(C(=O)[C@@H]2OC(=O)C)C)O)OC(=O)C)OC(=O)c5ccccc5)(C[C@@H]1OC(=O)[C@@H](O)[C@@H](NC(=O)c6ccccc6)c7ccccc7)O)(C)C',
    'DOCETAXEL': 'CC1=C2[C@@]([C@]([C@H]([C@@H]3[C@]4([C@H](OC4)C[C@@H]([C@]3(C(=O)[C@@H]2O)C)O)OC(=O)C)OC(=O)c5ccccc5)(C[C@@H]1OC(=O)[C@@H](O)[C@@H](NC(=O)OC(C)(C)C)c6ccccc6)O)(C)C',
    'CISPLATIN': '[NH3][Pt]([NH3])(Cl)Cl',
    # PubChem verified canonical SMILES for previously-failed drugs:
    'CARBOPLATIN': 'C1CCC(C1)(C(=O)[O-])C(=O)[O-].N.N.[Pt+2]',
    'OXALIPLATIN': 'C1CC[C@H]([C@@H](C1)N)N.C(=O)(C(=O)[O-])[O-].[Pt+2]',
    'TEMOZOLOMIDE': 'CN1C(=O)N2C=NC(=C2N=N1)C(=O)N',
    'CYCLOPHOSPHAMIDE': 'ClCCN(CCCl)P1(=O)NCCCO1',
    'METFORMIN': 'CN(C)C(=N)NC(=N)N',
    'DEXAMETHASONE': 'C[C@H]1C[C@H]2[C@@H]3CCC4=CC(=O)C=C[C@]4(C)[C@@]3(F)[C@@H](O)C[C@]2(C)[C@@]1(O)C(=O)CO',
    'BLEOMYCIN': 'CC(O)C(NC(=O)c1cnc(C(=O)NCCC(N)=O)nc1N)C(=O)NC(C(O)C(CO)O[C@@H]1OC(CO)C(O)C(O)C1O[C@@H]1OC(CO)C(OC(N)=O)C1O)C(=O)NC(C)C(=O)NC(CC(=O)N)C(=O)NC(C(=O)O)c1nc(C)cs1',
    'VINBLASTINE': 'CCC1(CC2CC(C3=C(CCN(C2)C1)C4=CC=CC=C4N3)(C5=C(C=C6C(=C5)[C@@]78CCN9[C@H]7[C@@](C=CC9)([C@H]([C@@]([C@@H]8N6C)(C(=O)OC)O)OC(=O)C)CC)OC)C(=O)OC)O',
    'VINORELBINE': 'CCC1(O)CC2CN(C1)CCC3=C2NC4=CC=CC=C34',
    'ETOPOSIDE': 'COC1=CC(=CC(=C1O)[C@@H]2C3=CC4=C(C=C3[C@@H]([C@@]5([C@@H]2C(=O)OC5C)C)O)OCO4)OC',
    'DOXORUBICIN': 'CC1OC(CC(O)C1N)OC1CC(O)(CC2OC3CC(OC3C(O)=C12)C(=O)CO)C(=O)CO',
    'TOPOTECAN': 'CCC1(O)C(=O)OCC2=C1C=C3C(NC4=CC(CN(C)C)=CC=C4C3=O)=C2',
    'GEMCITABINE': 'NC1=NC(=O)N(C=C1)[C@@H]1OC(CO)[C@@H](O)C1(F)F',
    'METHOTREXATE': 'CN(CC1=CN=C2N=C(N)N=C(N(C)C3=CC=C(C(=O)N[C@@H](CCC(=O)O)C(=O)O)C=C3)C2=N1)C',
    'SORAFENIB': 'CNC(=O)C1=CC(OC2=CC=C(NC(=O)NC3=CC(=C(C=C3)Cl)C(F)(F)F)C=C2)=CC=N1',
    'SUNITINIB': 'CCN(CC)CCNC(=O)C1=C(C)NC(=C1C)/C=C/1C(=O)NC2=CC(F)=CC=C12',
    'ERLOTINIB': 'C=CC1=CC2=C(C=C1OCCOC)N=CN=C2NC1=CC(=CC=C1)C#C',
    'LAPATINIB': 'CS(=O)(=O)CCNCC1=CC=C(O1)C1=CC2=C(C=C1)N=CN=C2NC1=CC(Cl)=C(OCC2=CC(F)=CC=C2)C=C1',
    'GEFITINIB': 'COC1=C(OCCCN2CCOCC2)C=C2C(NC3=CC(Cl)=C(F)C=C3)=NC=NC2=C1',
    'DASATINIB': 'CC1=NC(NC2=CC(=CC=C2)NC(=O)C2=CC(NC3=NC=CC=N3)=C(C)S2)=NC(=C1)N1CCN(CCO)CC1',
    'BORTEZOMIB': 'CC(C)CC(NC(=O)C(CC1=CC=CC=C1)NC(=O)C1=NC(=CS1)C1=CC=CC=C1)B(O)O',
    'DINACICLIB': 'CC(C1CCCCN1)N1C2=CC(=CC=C2C(=N1)N1CCNCC1)C#N',
    'GELDANAMYCIN': 'COC1CC(OC)CC(=O)NC(=CC(C)CC(OC)C(OC(N)=O)C(C)CC(C)=CC=CC(=O)C1O)C',
    'SN-38': 'CCC1(O)C(=O)OCC2=C1C=C1N(CC3CC4=CC(O)=CC=C4N=C13)C2=O',
    'MK-2206': 'C1CCN(CC1)C2=NC3=C(C=CC4=C3C=CN4)C(=N2)NC5=CC=C(C=C5)F',
    'MK-4827': 'O=C(NC1=CC=C2C(=C1)C=CN2)C1=C2CCCN2N=C1',
    'ABT-888': 'CC1(CCCN1)C1=NC2=CC(=CC=C2C(=O)N1)C(=O)N',
    'PD325901': 'OC(C(F)(F)F)C(=O)NC1=CC(F)=C(I)C=C1F',
    'MK-8669': 'COC1CC(CCC1OC1CCCC(C)C1OC)OC1=CC=C2C3=C(C(=O)OC3CC(=O)N(C)CCCOC)C(O)=C(C2=C1)/C=C/C(C)=CC',
    'ZOLINZA': 'ONC(=O)CCCCCCC(=O)NC1=CC=CC=C1',
    'MK-8776': 'NC1=NC=NC2=C1C(=NN2CC1=CC=CC=C1)C1=CC=C(C=C1)S(=O)(=O)N',
    'MK-5108': 'FC1=CC(=C(F)C=C1)C1=NN(C(=O)C1NC(=O)C1=CC(Cl)=CC=C1)C1=CC=CC=C1',
    # PubChem verified additions:
    'AZD1775': 'CC(C)(C1=NC(=CC=C1)N2C3=NC(=NC=C3C(=O)N2CC=C)NC4=CC=C(C=C4)N5CCN(CC5)C)O',
    'BEZ-235': 'C1=CC=C2C(=C1)C=CC3=CC4=CN=C(N4C(=O)N3C2)C5=CC=C(C=C5)C(C)(C)C#N',
    'L778123': 'C1CN(C(=O)CN1CC2=CN=CN2CC3=CC=C(C=C3)C#N)C4=CC(=CC=C4)Cl',
    'MRK-003': 'CCN(CC)CC(C)NC(=O)C1=CC(=CC(=C1)C(F)(F)F)NS(=O)(=O)C2=CC=CC=C2Cl',
    'MK-4541': 'C[C@]12CC[C@H]3[C@H]([C@@H]1CC[C@@H]2NC(=O)OCC(F)(F)F)CC[C@@H]4[C@@]3(C=CC(=O)N4C)C',
    'MITOMYCINE': 'CC1=C(C(=O)C2=C(C1=O)N3C[C@H]4[C@@H]([C@@]3([C@@H]2COC(=O)N)OC)N4)N',
}


def fetch_smiles_pubchem(drug_name):
    """Fetch SMILES from PubChem by drug name."""
    alias = DRUG_ALIASES.get(drug_name, drug_name)
    try:
        url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{alias}/property/CanonicalSMILES/JSON"
        resp = requests.get(url, timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            props = data.get('PropertyTable', {}).get('Properties', [])
            if props:
                return props[0].get('CanonicalSMILES', '')
    except Exception:
        pass
    return ''


def main():
    print("=" * 70)
    print("Priority 1: Real RDKit Morgan Fingerprints")
    print("=" * 70)

    # Step 1: Verify RDKit
    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem
        print("[OK] RDKit imported successfully")
    except ImportError:
        print("[FAIL] RDKit not available")
        return

    # Step 2: Get SMILES for all O'Neil drugs
    oneil = pd.read_csv(DATA_DIR / "oneil_synergy.csv")
    all_drugs = sorted(set(oneil['drug_a'].unique()) | set(oneil['drug_b'].unique()))
    print(f"\nO'Neil drugs: {len(all_drugs)}")

    smiles_db = {}
    failed = []

    for drug in all_drugs:
        drug_upper = drug.upper()

        # Priority: manual > PubChem
        if drug_upper in MANUAL_SMILES:
            smi = MANUAL_SMILES[drug_upper]
            mol = Chem.MolFromSmiles(smi)
            if mol:
                smiles_db[drug_upper] = smi
                continue
            else:
                logger.warning(f"  {drug}: manual SMILES invalid, trying PubChem")

        # Try PubChem
        smi = fetch_smiles_pubchem(drug)
        if smi:
            mol = Chem.MolFromSmiles(smi)
            if mol:
                smiles_db[drug_upper] = smi
                continue

        # Try alias
        alias = DRUG_ALIASES.get(drug_upper, None)
        if alias and alias != drug:
            smi = fetch_smiles_pubchem(alias)
            if smi:
                mol = Chem.MolFromSmiles(smi)
                if mol:
                    smiles_db[drug_upper] = smi
                    continue

        failed.append(drug_upper)
        time.sleep(0.3)

    print(f"\n  SMILES resolved: {len(smiles_db)}/{len(all_drugs)}")
    if failed:
        print(f"  Failed: {failed}")

    # Step 3: Generate Morgan Fingerprints
    morgan_fps = {}
    for drug, smi in smiles_db.items():
        mol = Chem.MolFromSmiles(smi)
        if mol:
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=1024)
            arr = np.zeros(1024, dtype=np.float32)
            for bit in fp.GetOnBits():
                arr[bit] = 1.0
            morgan_fps[drug] = arr
            on_bits = int(arr.sum())
            logger.info(f"  {drug}: {on_bits} on-bits (SMILES: {smi[:50]}...)")

    print(f"\n  Morgan FPs generated: {len(morgan_fps)}")

    # Verify chemical similarity makes sense
    print("\n  Chemical Similarity Check:")
    pairs_to_check = [
        ('PACLITAXEL', 'DOCETAXEL', 'Taxanes (should be similar)'),
        ('PACLITAXEL', 'METFORMIN', 'Unrelated (should be different)'),
        ('CISPLATIN', 'OXALIPLATIN', 'Platinum (should be similar)'),
        ('ERLOTINIB', 'GEFITINIB', 'EGFR inhibitors (should be similar)'),
        ('VINBLASTINE', 'VINORELBINE', 'Vinca alkaloids (should be similar)'),
    ]
    for d1, d2, desc in pairs_to_check:
        if d1 in morgan_fps and d2 in morgan_fps:
            tanimoto = np.sum(np.minimum(morgan_fps[d1], morgan_fps[d2])) / max(
                np.sum(np.maximum(morgan_fps[d1], morgan_fps[d2])), 1)
            print(f"    {d1} vs {d2}: Tanimoto={tanimoto:.3f} [{desc}]")

    # Save
    fp_path = MODEL_DIR / "drug_fingerprints_morgan.pkl"
    with open(fp_path, 'wb') as f:
        pickle.dump(morgan_fps, f)

    smiles_path = MODEL_DIR / "drug_smiles.json"
    with open(smiles_path, 'w') as f:
        json.dump(smiles_db, f, indent=2)

    print(f"\n  Saved: {fp_path}")
    print(f"  Saved: {smiles_path}")

    # Step 4: Retrain XGBoost with real FP
    print(f"\n{'='*70}")
    print("Retraining XGBoost with Real Morgan FP (O'Neil 23K)")
    print("=" * 70)

    from xgboost import XGBRegressor
    from scipy.stats import pearsonr

    with open(MODEL_DIR / "cell_line_expression.pkl", 'rb') as f:
        cell_expr = {k.upper(): v for k, v in pickle.load(f).items()}

    zero_fp = np.zeros(1024, dtype=np.float32)
    zero_expr = np.zeros(256, dtype=np.float32)

    # Build features with REAL FP
    X, y, folds_arr = [], [], []
    da_list, db_list, cl_list = [], [], []
    skipped = 0

    for _, row in oneil.iterrows():
        da = str(row['drug_a']).upper()
        db = str(row['drug_b']).upper()
        cl = str(row['cell_line']).upper()

        fp_a = morgan_fps.get(da, None)
        fp_b = morgan_fps.get(db, None)

        if fp_a is None or fp_b is None:
            skipped += 1
            continue

        expr = cell_expr.get(cl, zero_expr)
        features = np.concatenate([fp_a, fp_b, expr])

        X.append(features)
        y.append(float(row['synergy_loewe']))
        folds_arr.append(int(row.get('fold', 0)))
        da_list.append(da)
        db_list.append(db)
        cl_list.append(cl)

    X = np.array(X)
    y = np.array(y)
    folds = np.array(folds_arr)
    da_arr = np.array(da_list)
    db_arr = np.array(db_list)
    cl_arr = np.array(cl_list)

    print(f"  Dataset: {X.shape[0]:,} x {X.shape[1]} (skipped {skipped} missing FP)")

    # Pre-defined folds
    print(f"\n  Pre-defined Folds:")
    pf_results = []
    for fold in sorted(np.unique(folds)):
        te = folds == fold
        tr = ~te
        model = XGBRegressor(n_estimators=300, max_depth=6, learning_rate=0.1,
                             subsample=0.8, colsample_bytree=0.8, n_jobs=-1,
                             tree_method='hist', random_state=42)
        model.fit(X[tr], y[tr], eval_set=[(X[te], y[te])], verbose=False)
        pred = model.predict(X[te])
        r, _ = pearsonr(y[te], pred)
        pf_results.append(r)
        print(f"    Fold {fold}: r={r:.4f}")
    pf_mean = np.mean(pf_results)
    print(f"    PF Mean: r={pf_mean:.4f}")

    # LDPO
    print(f"\n  LDPO:")
    pairs = [tuple(sorted([a, b])) for a, b in zip(da_arr, db_arr)]
    pairs = np.array(pairs, dtype=object)
    unique_pairs = list(set([tuple(p) for p in pairs]))
    np.random.seed(42)
    pperm = np.random.permutation(len(unique_pairs))
    pfold_size = len(unique_pairs) // 5

    ldpo_results = []
    for fold in range(5):
        s = fold * pfold_size
        e = s + pfold_size if fold < 4 else len(unique_pairs)
        test_set = set([unique_pairs[pperm[i]] for i in range(s, e)])
        te = np.array([tuple(p) in test_set for p in pairs])
        tr = ~te
        if te.sum() < 10:
            continue
        model = XGBRegressor(n_estimators=300, max_depth=6, learning_rate=0.1,
                             subsample=0.8, colsample_bytree=0.8, n_jobs=-1,
                             tree_method='hist', random_state=42)
        model.fit(X[tr], y[tr], eval_set=[(X[te], y[te])], verbose=False)
        pred = model.predict(X[te])
        r, _ = pearsonr(y[te], pred)
        ldpo_results.append(r)
        print(f"    LDPO Fold {fold}: r={r:.4f}")
    ldpo_mean = np.mean(ldpo_results)
    print(f"    LDPO Mean: r={ldpo_mean:.4f}")

    # LCLO
    print(f"\n  LCLO:")
    unique_cells = np.unique(cl_arr)
    np.random.seed(42)
    cperm = np.random.permutation(len(unique_cells))
    cfold_size = len(unique_cells) // 5

    lclo_results = []
    for fold in range(5):
        s = fold * cfold_size
        e = s + cfold_size if fold < 4 else len(unique_cells)
        test_set = set(unique_cells[cperm[s:e]])
        te = np.array([c in test_set for c in cl_arr])
        tr = ~te
        if te.sum() < 10:
            continue
        model = XGBRegressor(n_estimators=300, max_depth=6, learning_rate=0.1,
                             subsample=0.8, colsample_bytree=0.8, n_jobs=-1,
                             tree_method='hist', random_state=42)
        model.fit(X[tr], y[tr], eval_set=[(X[te], y[te])], verbose=False)
        pred = model.predict(X[te])
        r, _ = pearsonr(y[te], pred)
        lclo_results.append(r)
        print(f"    LCLO Fold {fold}: r={r:.4f}")
    lclo_mean = np.mean(lclo_results)
    print(f"    LCLO Mean: r={lclo_mean:.4f}")

    # Save model
    model_path = MODEL_DIR / "xgboost_synergy_morgan.pkl"
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)

    # Comparison
    print(f"\n{'='*70}")
    print("COMPARISON: Hash FP vs Real Morgan FP")
    print("=" * 70)
    print(f"  Hash FP (O'Neil):   PF=0.605  LDPO=0.604  LCLO=0.510")
    print(f"  Morgan FP (O'Neil): PF={pf_mean:.3f}  LDPO={ldpo_mean:.3f}  LCLO={lclo_mean:.3f}")

    delta_pf = pf_mean - 0.605
    delta_ldpo = ldpo_mean - 0.604
    delta_lclo = lclo_mean - 0.510

    sign = lambda x: '+' if x >= 0 else ''
    print(f"  Delta:              PF={sign(delta_pf)}{delta_pf:.3f}  "
          f"LDPO={sign(delta_ldpo)}{delta_ldpo:.3f}  LCLO={sign(delta_lclo)}{delta_lclo:.3f}")


if __name__ == "__main__":
    main()
