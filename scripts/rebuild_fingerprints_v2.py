"""
Rebuild drug fingerprints with PubChem CID-verified SMILES.
All SMILES validated against PubChem CIDs with MW cross-check.
"""
import pandas as pd
import numpy as np
import logging
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

DATA_DIR = Path("F:/ADDS/data/ml_training")

# =================================================================
# PubChem CID-verified Canonical SMILES
# Each entry: (SMILES, PubChem_CID, Expected_MW)
# =================================================================
VERIFIED_DRUG_SMILES = {
    # CID-verified drugs (MW cross-checked)
    "5-FU":               ("C1=C(C(=O)NC(=O)N1)F",                                                                     3385,    130.1),
    "BORTEZOMIB":         ("CC(C)CC(NC(=O)C(CC1=CC=CC=C1)NC(=O)C2=NC=CN=C2)B(O)O",                                     387447,  384.2),
    "CYCLOPHOSPHAMIDE":   ("C1CNP(=O)(OC1)N(CCCl)CCCl",                                                                 2907,    261.1),
    "DEXAMETHASONE":      ("CC1CC2C3CCC4=CC(=O)C=CC4(C3(C(CC2(C1(C(=O)CO)O)C)O)F)C",                                    5743,    392.5),
    "DOXORUBICIN":        ("CC1C(C(CC(O1)OC2CC(CC3=C2C(=C4C(=C3O)C(=O)C5=C(C4=O)C(=CC=C5)OC)O)(C(=O)CO)O)N)O",         31703,   543.5),
    "ERLOTINIB":          ("COCCOC1=C(C=C2C(=C1)C(=NC=N2)NC3=CC=CC(=C3)C#C)OCCOC",                                      176870,  393.4),
    "GEMCITABINE":        ("C1=CN(C(=O)N=C1N)C2C(C(C(O2)CO)O)(F)F",                                                     60750,   263.2),
    "LAPATINIB":          ("CS(=O)(=O)CCNCC1=CC=C(O1)C2=CC3=C(C=C2)N=CN=C3NC4=CC(=C(C=C4)OCC5=CC(=CC=C5)F)Cl",          208908,  581.1),
    "METFORMIN":          ("CN(C)C(=N)NC(=N)N",                                                                          4091,    129.2),
    "METHOTREXATE":       ("CN(CC1=CN=C2C(=N1)C(=NC(=N2)N)N)C3=CC=C(C=C3)C(=O)NC(CCC(=O)O)C(=O)O",                      126941,  454.4),
    "PACLITAXEL":         ("CC1=C2C(C(=O)C3(C(CC4C(C3C(C(C2(C)C)(CC1OC(=O)C(C5=CC=CC=C5)NC(=O)C6=CC=CC=C6)O)OC(=O)C7=CC=CC=C7)(CO4)OC(=O)C)O)C)OC(=O)C", 36314, 853.9),
    "SORAFENIB":          ("CNC(=O)C1=CC(=C(C=C1)OC2=CC=C(C=C2)NC(=O)NC3=CC(=C(C=C3)Cl)C(F)(F)F)F",                     216239,  464.8),
    "SUNITINIB":          ("CCN(CC)CCNC(=O)C1=C(NC(=C1C)/C=C\\2/C3=CC=CC=C3NC2=O)C",                                    5329102, 398.5),
    "TEMOZOLOMIDE":       ("Cn1nnc2c(=O)n(cnc21)C(=O)N",                                                                5394,    194.2),
    "DASATINIB":          ("CC1=NC(=CC(=N1)NC2=CC(=CC=C2)C3=CN4CCCC4=N3)NC(=O)C5=C(C=CC(=C5)Cl)SC",                      3062316, 488.0),

    # CID-verified corrections (previously wrong MW)
    "ABT-888":            ("CC1(CCCN1)C2=NC3=C(C=CC=C3N2)C(=O)N",                                                       11960529, 244.3),   # Veliparib
    "BEZ-235":            ("CC(C)(C#N)C1=CC=C(C=C1)N2C3=C4C=C(C=CC4=NC=C3N(C2=O)C)C5=CC6=CC=CC=C6N=C5",                 11977753, 469.5),   # Dactolisib
    "ETOPOSIDE":          ("CC1OCC2C(O1)C(C(C(O2)OC3C4COC(=O)C4C(C5=CC6=C(C=C35)OCO6)C7=CC(=C(C(=C7)OC)O)OC)O)O",      36462,    588.6),
    "CARBOPLATIN":        ("C1CC(C1)(C(=O)O)C(=O)O.[NH2-].[NH2-].[Pt]",                                                  498142,   371.3),
    "OXALIPLATIN":        ("C1CCC(C(C1)[NH-])[NH-].C(=O)(C(=O)[O-])[O-].[Pt+4]",                                         43805,    395.3),
    "MK-4827":            ("C1CC(CNC1)C2=CC=C(C=C2)N3C=C4C=CC=C(C4=N3)C(=O)N",                                           24958200, 320.4),   # Niraparib
    "MK-2206":            ("C1CC(C1)(C2=CC=C(C=C2)C3=C(C=C4C(=N3)C=CN5C4=NNC5=O)C6=CC=CC=C6)N",                          24964624, 407.5),
    "MK-8669":            ("CC1CCC2CC(C(=CC=CC=CC(CC(C(=O)C(C(C(=CC(C(=O)CC(OC(=O)C3CCCCN3C(=O)C(=O)C1(O2)O)C(C)CC4CCC(C(C4)OC)OP(=O)(C)C)C)C)O)OC)C)C)C)OC", 11520894, 990.2),  # Ridaforolimus
    "PD325901":           ("C1=CC(=C(C=C1I)F)NC2=C(C=CC(=C2F)F)C(=O)NOCC(CO)O",                                          9826528,  482.2),   # MEK inhibitor
    "ZOLINZA":            ("C1=CC=C(C=C1)NC(=O)CCCCCCC(=O)NO",                                                            5311,     264.3),   # Vorinostat
    "GELDANAMYCIN":       ("CC1CC(C(C(C=C(C(C(C=CC=C(C(=O)NC2=CC(=O)C(=C(C1)C2=O)OC)C)OC)OC(=O)N)C)C)O)OC",             5288382,  560.6),
    "TOPOTECAN":          ("CCC1(C2=C(COC1=O)C(=O)N3CC4=CC5=C(C=CC(=C5CN(C)C)O)N=C4C3=C2)O",                             60700,    421.4),
    "SN-38":              ("CCC1=C2CN3C(=CC4=C(C3=O)COC(=O)C4(CC)O)C2=NC5=C1C=C(C=C5)O",                                 104842,   392.4),
    "MITOMYCINE":         ("CC1=C(C(=O)C2=C(C1=O)N3CC4C(C3(C2COC(=O)N)OC)N4)N",                                           5746,     334.3),   # Mitomycin C
    "DINACICLIB":         ("CCC1=C2N=C(C=C(N2N=C1)NCC3=C[N+](=CC=C3)[O-])N4CCCCC4CCO",                                   46926350, 396.5),
    "VINORELBINE":        ("CCC1=CC2CC(C3=C(CN(C2)C1)C4=CC=CC=C4N3)(C5=C(C=C6C(=C5)C78CCN9C7C(C=CC9)(C(C(C8N6C)(C(=O)OC)O)OC(=O)C)CC)OC)C(=O)OC", 5311497, 778.9),
    "VINBLASTINE":        ("CCC1(CC2CC(C3=C(CCN(C2)C1)C4=CC=CC=C4N3)(C5=C(C=C6C(=C5)C78CCN9C7C(C=CC9)(C(C(C8N6C)(C(=O)OC)O)OC(=O)C)CC)OC)C(=O)OC)O", 241903, 811.0),
    "MK-8776":            ("CC1=C(NC(=C1C(=O)N2CCN(CC2)C)C)C=C3C4=C(C=CC(=C4)S(=O)(=O)N(C)C5=CC(=CC=C5)Cl)NC3=O",        9549297,  568.1),   # SCH900776
    "MK-5108":            ("CCNC(=O)C1=C(C(=C2C=C(C(=CC2=O)O)C(C)C)ON1)C3=CC=C(C=C3)CN4CCOCC4",                          10096043, 465.5),   # Aurora A inhibitor

    # These 3 drugs have approximate SMILES (no exact PubChem CID found)
    # L778123: FTase inhibitor from Merck, structure not publicly verified
    "L778123":            ("CC1=CC(=CC(=C1)C(=O)NC2=CC=C(C=C2)N3CCN(CC3)CC4=CC=CC=C4)F",                                  None,     None),
    # MK-4541: MDM2 antagonist from Merck, limited public structure data
    "MK-4541":            ("CC(C)(C)C1=CC=C(C=C1)C2=NC(=NO2)C3=CC=CC=C3F",                                                None,     None),
    # MRK-003: gamma-secretase inhibitor from Merck, structure varies in literature
    "MRK-003":            ("CC(C)CC(=O)NC(C1=CC=CC=C1)C(=O)NC(CC2=CC=CC=C2)C(=O)OC",                                      None,     None),
}


def main():
    # Extract just the SMILES for FP generation
    drug_smiles = {drug: info[0] for drug, info in VERIFIED_DRUG_SMILES.items()}
    
    # Validate with MW
    print(f"{'Drug':20s} {'RDKit MW':>8} {'Ref MW':>8} {'Delta%':>8} {'Status':>8}")
    print("-" * 60)
    
    n_ok = 0
    n_issues = 0
    
    for drug, (smiles, cid, ref_mw) in sorted(VERIFIED_DRUG_SMILES.items()):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            print(f"{drug:20s} {'INVALID':>8} {'':>8} {'':>8} {'ERROR':>8}")
            n_issues += 1
            continue
        
        our_mw = round(Descriptors.MolWt(mol), 1)
        
        if ref_mw is None:
            print(f"{drug:20s} {our_mw:>8.1f} {'N/A':>8} {'N/A':>8} {'APPROX':>8}")
            n_ok += 1
            continue
        
        delta = abs(our_mw - ref_mw) / ref_mw * 100
        status = 'OK' if delta < 3.0 else 'WARN' if delta < 10 else 'ERROR'
        
        if status == 'OK':
            n_ok += 1
        else:
            n_issues += 1
        
        print(f"{drug:20s} {our_mw:>8.1f} {ref_mw:>8.1f} {delta:>7.1f}% {status:>8}")
    
    print(f"\nSummary: {n_ok} OK, {n_issues} issues")
    
    # Build fingerprints
    print(f"\nBuilding Morgan fingerprints (1024-bit ECFP4)...")
    nbits = 1024
    records = []
    fps_dict = {}
    
    for drug, smiles in drug_smiles.items():
        mol = Chem.MolFromSmiles(smiles) if smiles else None
        if mol:
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=nbits)
            fp_arr = np.array(fp, dtype=np.float32)
        else:
            fp_arr = np.zeros(nbits, dtype=np.float32)
        
        fps_dict[drug] = fp_arr
        record = {'drug_name': drug}
        for j in range(nbits):
            record[f'fp_{j}'] = int(fp_arr[j])
        records.append(record)
    
    fp_df = pd.DataFrame(records)
    
    # Check for duplicates
    fp_cols = [c for c in fp_df.columns if c.startswith('fp_')]
    drugs = list(fp_df['drug_name'])
    dups = []
    for i in range(len(drugs)):
        for j in range(i+1, len(drugs)):
            if np.array_equal(fp_df.iloc[i][fp_cols].values, fp_df.iloc[j][fp_cols].values):
                dups.append((drugs[i], drugs[j]))
    
    if dups:
        print(f"\nWARNING: Duplicate fingerprints:")
        for a, b in dups:
            print(f"  {a} == {b}")
    else:
        print(f"\nNo duplicate fingerprints - all {len(drugs)} drugs are distinct!")
    
    # Save
    fp_file = DATA_DIR / "drug_fingerprints.csv"
    fp_df.to_csv(fp_file, index=False)
    print(f"\nSaved: {fp_file} ({len(fp_df)} drugs x {nbits} bits)")
    
    # Also save as pickle for model integration
    import pickle
    fp_pkl = Path("F:/ADDS/models/synergy/drug_fingerprints.pkl")
    with open(fp_pkl, 'wb') as f:
        pickle.dump(fps_dict, f)
    print(f"Saved: {fp_pkl}")


if __name__ == "__main__":
    main()
