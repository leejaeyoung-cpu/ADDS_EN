"""
Offline SMILES verification using RDKit + literature molecular weights.
No PubChem API needed - validates against known MW from literature.
"""
from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem
import numpy as np

# Known MW from DrugBank / PubChem (literature values)
LITERATURE_MW = {
    '5-FU': 130.08,
    'ABT-888': 244.29,       # Veliparib
    'AZD1775': 381.43,       # Adavosertib  
    'BEZ-235': 469.54,       # Dactolisib
    'BORTEZOMIB': 384.24,
    'CARBOPLATIN': 371.25,
    'CYCLOPHOSPHAMIDE': 261.08,
    'DASATINIB': 488.01,
    'DEXAMETHASONE': 392.46,
    'DINACICLIB': 396.49,    # SCH727965
    'DOXORUBICIN': 543.52,
    'ERLOTINIB': 393.44,
    'ETOPOSIDE': 588.56,
    'GELDANAMYCIN': 560.64,
    'GEMCITABINE': 263.20,
    'L778123': 464.51,       # FTase inhibitor
    'LAPATINIB': 581.06,
    'METFORMIN': 129.16,
    'METHOTREXATE': 454.44,
    'MITOMYCIN_C': 334.33,
    'MK-2206': 407.51,
    'MK-4541': 309.31,       # MDM2 inhibitor (may vary)
    'MK-4827': 320.39,       # Niraparib
    'MK-5108': 296.37,       # Aurora A inhibitor (may vary)
    'MK-8669': 990.21,       # Ridaforolimus
    'MK-8776': 380.50,       # SCH900776
    'MRK-003': 556.69,       # gamma-secretase (may vary)
    'OXALIPLATIN': 397.29,
    'PACLITAXEL': 853.91,
    'PD325901': 482.19,      # MEK inhibitor
    'SN-38': 392.41,
    'SORAFENIB': 464.82,
    'SUNITINIB': 398.47,
    'TEMOZOLOMIDE': 194.15,
    'TOPOTECAN': 421.45,
    'VINBLASTINE': 810.97,
    'VINORELBINE': 778.93,   # Note: MW is different from vinblastine
    'ZOLINZA': 264.32,       # Vorinostat
}

# Current SMILES from our download_oneil_synergy.py
CURRENT_SMILES = {
    "5-FU": "C1=C(C(=O)NC(=O)N1)F",
    "ABT-888": "C1CC1C(=O)NC2=CC3=C(C=C2)N=C(N3)C4=CC=CC5=C4C(=O)N(C5=O)C",
    "AZD1775": "CC1=CC2=C(S1)C(=NC=N2)NC3=CC(=CC=C3)C(=O)N4CCCC(C4)N5CCOCC5",
    "BEZ-235": "C1CN(CCN1)C2=CC3=C(C=C2)C(=CC4=CC5=CC=CC=C5N=C34)C#N",
    "BORTEZOMIB": "CC(C)CC(NC(=O)C(CC1=CC=CC=C1)NC(=O)C2=NC=CN=C2)B(O)O",
    "CARBOPLATIN": "C(CC1CC1)(=O)[O-].[NH3][Pt+2]([NH3])OC(=O)C1CCC1",
    "CYCLOPHOSPHAMIDE": "C1CNP(=O)(OC1)N(CCCl)CCCl",
    "DASATINIB": "CC1=NC(=CC(=N1)NC2=CC(=CC=C2)C3=CN4CCCC4=N3)NC(=O)C4=C(C=CC(=C4)Cl)SC",
    "DEXAMETHASONE": "CC1CC2C3CCC4=CC(=O)C=CC4(C3(C(CC2(C1(C(=O)CO)O)C)O)F)C",
    "DINACICLIB": "CC(C1=NN=C(N1C2=CC=C(C=C2)C3CCNCC3)C)NC4=NC=C(N=C4)C(=O)N",
    "DOXORUBICIN": "CC1C(C(CC(O1)OC2CC(CC3=C2C(=C4C(=C3O)C(=O)C5=C(C4=O)C(=CC=C5)OC)O)(C(=O)CO)O)N)O",
    "ERLOTINIB": "COCCOC1=C(C=C2C(=C1)C(=NC=N2)NC3=CC=CC(=C3)C#C)OCCOC",
    "ETOPOSIDE": "CC1OCC2C(O1)C(C(C(O2)OC3=CC4=CC5=C(C=C5)OCO4)C(=O)C3)O",
    "GELDANAMYCIN": "COC1CC(OC)C(O)C(C=CC(=O)CC(OC)C(=CC2=C(C(=O)C=C(C2=O)N)OC)C)OC(=O)N1",
    "GEMCITABINE": "C1=CN(C(=O)N=C1N)C2C(C(C(O2)CO)O)(F)F",
    "L778123": "CC1=CC(=CC(=C1)C(=O)NC2=CC=C(C=C2)N3CCN(CC3)CC4=CC=CC=C4)F",
    "LAPATINIB": "CS(=O)(=O)CCNCC1=CC=C(O1)C2=CC3=C(C=C2)N=CN=C3NC4=CC(=C(C=C4)OCC5=CC(=CC=C5)F)Cl",
    "METFORMIN": "CN(C)C(=N)NC(=N)N",
    "METHOTREXATE": "CN(CC1=CN=C2C(=N1)C(=NC(=N2)N)N)C3=CC=C(C=C3)C(=O)NC(CCC(=O)O)C(=O)O",
    "MITOMYCINE": "COC1=C(C)C2=C(N1)C1=CC3=C(C(=O)C=C1N2CC1CO1)C(N)=O.OC3",
    "MK-2206": "C1CCC(CC1)NC2=NC3=CC=CC=C3N=C2C4=CC=C(C=C4)NC5=NC(=NC(=N5)N)N",
    "MK-4541": "CC(C)(C)C1=CC=C(C=C1)C2=NC(=NO2)C3=CC=CC=C3F",
    "MK-4827": "C1CCC2=C(C1)C(=CC=C2)C(=O)NC3=CC=C(C=C3)CC(=O)N4CCCC4=O",
    "MK-5108": "CC(C)OC1=CC=C(C=C1)NC(=O)NC2=CC=CC(=C2)C3=CNN=C3",
    "MK-8669": "CC1CCC2CC(C(=CC=CC(CC(C(=O)C(C(C(=CC(C(=O)CC(OC(=O)C3CCCCN3C(=O)C(=O)C1(O2)O)C(C)CC4CCC(C(C4)OC)O)C)C)O)OC)C)C)C)OC",
    "MK-8776": "CC1=C2C(=NC(=N1)NC3=CC=C(C=C3)S(=O)(=O)N(C)C)N=CN2C4CCCC4",
    "MRK-003": "CC(C)CC(=O)NC(C1=CC=CC=C1)C(=O)NC(CC2=CC=CC=C2)C(=O)OC",
    "OXALIPLATIN": "C1CCC(C1)[NH2][Pt]([NH2]C2CCCC2)(OC(=O)C(=O)O)OC(=O)C(=O)O",
    "PACLITAXEL": "CC1=C2C(C(=O)C3(C(CC4C(C3C(C(C2(C)C)(CC1OC(=O)C(C5=CC=CC=C5)NC(=O)C6=CC=CC=C6)O)OC(=O)C7=CC=CC=C7)(CO4)OC(=O)C)O)C)OC(=O)C",
    "PD325901": "OC(C(F)(F)F)C(=O)NC1=CC=C(I)C(=C1F)F",
    "SN-38": "CCC1(O)C(=O)OCC2=C1C=C3N(CC4=CC5=CC=CC=C5NC34)C2=O",
    "SORAFENIB": "CNC(=O)C1=CC(=C(C=C1)OC2=CC=C(C=C2)NC(=O)NC3=CC(=C(C=C3)Cl)C(F)(F)F)F",
    "SUNITINIB": "CCN(CC)CCNC(=O)C1=C(NC(=C1C)/C=C\\2/C3=CC=CC=C3NC2=O)C",
    "TEMOZOLOMIDE": "Cn1nnc2c(=O)n(cnc21)C(=O)N",
    "TOPOTECAN": "CCC1(C2=C(COC1=O)C(=O)N3CC4=C(C=C(C=C4C3=C2)O)CN(C)C)O",
    "VINBLASTINE": "CCC1(CC2CC(C3=C(CN(C2)C1)C4=CC=CC=C4N3)(C5=C(C=C6C(=C5)C78CCN9C7C(C=CC9)(C(C(C8N6C=O)(C(=O)OC)O)OC(=O)C)CC)OC)O)O",
    "VINORELBINE": "CCC1(CC2CC(C3=C(CN(C2)C1)C4=CC=CC=C4N3)(C5=C(C=C6C(=C5)C78CCN9C7C(C=CC9)(C(C(C8N6C=O)(C(=O)OC)O)OC(=O)C)CC)OC)O)O",
    "ZOLINZA": "OC(=O)CCCCCCC(=O)NO",
}


def main():
    print(f"{'Drug':20s} {'RDKit Valid':>10} {'Our MW':>8} {'Lit MW':>8} {'Delta%':>8} {'Status':>10}")
    print("-" * 76)
    
    issues = []
    
    for drug in sorted(CURRENT_SMILES.keys()):
        smiles = CURRENT_SMILES[drug]
        mol = Chem.MolFromSmiles(smiles) if smiles else None
        
        if mol is None:
            print(f"{drug:20s} {'INVALID':>10} {'---':>8} {'---':>8} {'---':>8} {'[ERROR]':>10}")
            issues.append((drug, 'INVALID_SMILES', smiles))
            continue
        
        our_mw = round(Descriptors.MolWt(mol), 1)
        
        # Lookup literature MW (handle both drug name and aliases)
        lit_key = drug.replace('MITOMYCINE', 'MITOMYCIN_C')
        lit_mw = LITERATURE_MW.get(lit_key, LITERATURE_MW.get(drug, None))
        
        if lit_mw is None:
            print(f"{drug:20s} {'OK':>10} {our_mw:>8.1f} {'???':>8} {'???':>8} {'[NO_REF]':>10}")
            issues.append((drug, 'NO_REFERENCE', smiles))
            continue
        
        delta_pct = abs(our_mw - lit_mw) / lit_mw * 100.0
        
        if delta_pct > 15.0:
            status = '[WRONG]'
            issues.append((drug, f'MW_MISMATCH ({our_mw:.0f} vs {lit_mw:.0f}, {delta_pct:.0f}%)', smiles))
        elif delta_pct > 5.0:
            status = '[CHECK]'
            issues.append((drug, f'MW_SUSPECT ({our_mw:.0f} vs {lit_mw:.0f}, {delta_pct:.0f}%)', smiles))
        else:
            status = 'OK'
        
        print(f"{drug:20s} {'OK':>10} {our_mw:>8.1f} {lit_mw:>8.1f} {delta_pct:>7.1f}% {status:>10}")
    
    # Check if Vinblastine == Vinorelbine
    if CURRENT_SMILES.get('VINBLASTINE') == CURRENT_SMILES.get('VINORELBINE'):
        issues.append(('VINORELBINE', 'DUPLICATE_OF_VINBLASTINE', 'Same SMILES as Vinblastine'))
    
    # Summary
    print(f"\n{'='*76}")
    print(f"Issues found: {len(issues)}")
    for drug, issue, detail in issues:
        print(f"  {drug:20s}: {issue}")
    
    # Fingerprint comparison for dup check  
    print(f"\n{'='*76}")
    print("Fingerprint uniqueness check:")
    fps_dict = {}
    for drug, smiles in CURRENT_SMILES.items():
        mol = Chem.MolFromSmiles(smiles) if smiles else None
        if mol:
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=1024)
            fp_arr = np.array(fp, dtype=int)
            fps_dict[drug] = fp_arr
    
    # Find duplicates
    drugs = list(fps_dict.keys())
    dups = []
    for i in range(len(drugs)):
        for j in range(i+1, len(drugs)):
            if np.array_equal(fps_dict[drugs[i]], fps_dict[drugs[j]]):
                dups.append((drugs[i], drugs[j]))
    
    if dups:
        print("  DUPLICATE fingerprints found:")
        for a, b in dups:
            print(f"    {a} == {b}")
    else:
        print("  No duplicate fingerprints")


if __name__ == "__main__":
    main()
