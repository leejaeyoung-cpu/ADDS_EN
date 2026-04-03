"""
Signal Pathway Database for ADDS Recommendation System
Based on major cancer signaling pathways
"""

from typing import Dict, List, Any

# Major signal pathways in cancer
# Activation frequencies sourced from COSMIC, TCGA, and cited publications
SIGNAL_PATHWAYS = {
    "MAPK_ERK": {
        "name": "MAPK/ERK Pathway",
        "full_name": "Mitogen-Activated Protein Kinase / Extracellular signal-Regulated Kinase",
        "description": "Controls cell proliferation, differentiation, and survival",
        "function": "Growth signal transduction from cell surface to nucleus",
        "components": [
            {"name": "EGFR", "type": "receptor", "role": "Growth factor receptor"},
            {"name": "RAS", "type": "GTPase", "role": "Signal transducer"},
            {"name": "RAF", "type": "kinase", "role": "MAPK kinase kinase"},
            {"name": "MEK", "type": "kinase", "role": "MAPK kinase"},
            {"name": "ERK", "type": "kinase", "role": "MAPK, transcription regulator"}
        ],
        "activation_in_cancer": {
            "Colorectal": 0.45,  # KRAS mut ~45% (PMID: 25964097, TCGA CRC)
            "Lung": 0.30,       # KRAS mut ~30% NSCLC (PMID: 25157968)
            "Breast": 0.20,     # KRAS/BRAF rare (PMID: 22797534)
            "Pancreatic": 0.90  # KRAS mut ~90% (PMID: 18794811)
        },
        "drugs_targeting": {
            "Cetuximab": {
                "target": "EGFR",
                "mechanism": "Monoclonal antibody blocking EGFR",
                "effect": "inhibit",
                "efficacy": "High in wild-type KRAS (PMID: 15269313)"
            },
            "Encorafenib": {
                "target": "BRAF",
                "mechanism": "BRAF V600E kinase inhibitor",
                "effect": "inhibit",
                "efficacy": "High in BRAF-mutant CRC with Cetuximab (BEACON CRC, PMID: 31566309)"
            }
        },
        "clinical_notes": "KRAS mutations make EGFR inhibitors ineffective. BRAF V600E treated with Encorafenib+Cetuximab."
    },
    
    "PI3K_AKT_mTOR": {
        "name": "PI3K/AKT/mTOR Pathway",
        "full_name": "Phosphatidylinositol 3-Kinase / Protein Kinase B / Mammalian Target of Rapamycin",
        "description": "Regulates cell survival, metabolism, and angiogenesis",
        "function": "Promotes cell survival and growth, inhibits apoptosis",
        "components": [
            {"name": "PI3K", "type": "kinase", "role": "Lipid kinase"},
            {"name": "AKT", "type": "kinase", "role": "Serine/threonine kinase"},
            {"name": "mTOR", "type": "kinase", "role": "Master growth regulator"},
            {"name": "PTEN", "type": "phosphatase", "role": "Tumor suppressor, PI3K inhibitor"}
        ],
        "activation_in_cancer": {
            "Breast": 0.70,      # PIK3CA mut + PTEN loss (PMID: 22722839)
            "Ovarian": 0.60,     # (PMID: 25281616)
            "Colorectal": 0.30,  # PIK3CA mut ~15-20%, PTEN loss ~30% (PMID: 20921465)
            "Lung": 0.25         # (PMID: 22282465)
        },
        "drugs_targeting": {
            "Doxorubicin": {
                "target": "DNA",
                "mechanism": "DNA intercalation, indirect pathway effects",
                "effect": "cytotoxic",
                "efficacy": "Broad spectrum"
            },
            "Paclitaxel": {
                "target": "Microtubules",
                "mechanism": "Stabilizes microtubules, indirect AKT inhibition",
                "effect": "cytotoxic",
                "efficacy": "High"
            }
        },
        "clinical_notes": "Often co-activated with MAPK, synergy with targeted therapies"
    },
    
    "p53": {
        "name": "p53 Pathway",
        "full_name": "Tumor Protein p53",
        "description": "Guardian of the genome - controls cell cycle and apoptosis",
        "function": "DNA damage response, apoptosis, cell cycle arrest",
        "components": [
            {"name": "p53", "type": "transcription_factor", "role": "Tumor suppressor"},
            {"name": "MDM2", "type": "E3_ligase", "role": "p53 degradation"},
            {"name": "p21", "type": "CDK_inhibitor", "role": "Cell cycle arrest"},
            {"name": "BAX", "type": "pro_apoptotic", "role": "Apoptosis execution"}
        ],
        "activation_in_cancer": {
            "Colorectal": 0.50,  # TP53 mut ~50% (PMID: 25964097, TCGA)
            "Lung": 0.50,       # TP53 mut ~50% NSCLC (PMID: 25079552)
            "Breast": 0.30,     # TP53 mut ~30% overall, ~80% TNBC (PMID: 23000897)
            "Ovarian": 0.96     # TP53 mut ~96% HGSOC (PMID: 21720365)
        },
        "drugs_targeting": {
            "Cisplatin": {
                "target": "DNA",
                "mechanism": "DNA crosslinking, triggers p53-mediated apoptosis",
                "effect": "activate_apoptosis",
                "efficacy": "High in p53 wild-type"
            },
            "5-FU": {
                "target": "Thymidylate synthase",
                "mechanism": "DNA/RNA synthesis inhibition",
                "effect": "cytotoxic",
                "efficacy": "Moderate"
            },
            "Doxorubicin": {
                "target": "DNA",
                "mechanism": "DNA damage, p53 activation",
                "effect": "activate_apoptosis",
                "efficacy": "High"
            }
        },
        "clinical_notes": "p53 mutations reduce efficacy of DNA-damaging agents"
    },
    
    "Wnt_beta_catenin": {
        "name": "Wnt/β-catenin Pathway",
        "full_name": "Wingless-related integration site / β-catenin",
        "description": "Controls stem cell renewal and differentiation",
        "function": "Embryonic development, tissue homeostasis, cancer stem cell maintenance",
        "components": [
            {"name": "Wnt", "type": "ligand", "role": "Signaling molecule"},
            {"name": "Frizzled", "type": "receptor", "role": "Wnt receptor"},
            {"name": "β-catenin", "type": "transcription_coactivator", "role": "Nuclear signaling"},
            {"name": "APC", "type": "tumor_suppressor", "role": "β-catenin degradation"}
        ],
        "activation_in_cancer": {
            "Colorectal": 0.80,  # APC mut ~80% CRC (PMID: 25964097, TCGA)
            "Breast": 0.20,     # Wnt dysreg ~20% (PMID: 25592788)
            "Lung": 0.10,       # (PMID: 22282465)
            "Pancreatic": 0.15  # (PMID: 18794811)
        },
        "drugs_targeting": {
            "5-FU": {
                "target": "Thymidylate synthase",
                "mechanism": "Reduces proliferation of stem-like cells",
                "effect": "cytotoxic",
                "efficacy": "Standard in colorectal"
            },
            "Oxaliplatin": {
                "target": "DNA",
                "mechanism": "DNA crosslinking",
                "effect": "cytotoxic",
                "efficacy": "High"
            },
            "Irinotecan": {
                "target": "Topoisomerase I",
                "mechanism": "DNA damage in rapidly dividing cells",
                "effect": "cytotoxic",
                "efficacy": "High"
            }
        },
        "clinical_notes": "Colorectal cancer hallmark, targets stem cell population"
    },
    
    "NF_kB": {
        "name": "NF-κB Pathway",
        "full_name": "Nuclear Factor kappa B",
        "description": "Master regulator of inflammation and immune response",
        "function": "Inflammation, immune response, cell survival",
        "components": [
            {"name": "NF-κB", "type": "transcription_factor", "role": "Inflammation regulator"},
            {"name": "IκB", "type": "inhibitor", "role": "NF-κB sequestration"},
            {"name": "IKK", "type": "kinase", "role": "IκB phosphorylation"},
            {"name": "TNF-α", "type": "cytokine", "role": "Inflammatory signal"}
        ],
        "activation_in_cancer": {
            "Colorectal": 0.40,  # NF-kB constitutive activation (PMID: 15064735)
            "Lung": 0.35,       # (PMID: 19185345)
            "Breast": 0.30,     # (PMID: 17679093)
            "Pancreatic": 0.60  # Strong NF-kB activation (PMID: 17510367)
        },
        "drugs_targeting": {
            "Bortezomib": {
                "target": "Proteasome/IκB",
                "mechanism": "Proteasome inhibition, blocks NF-κB activation via IκB stabilization",
                "effect": "inhibit",
                "efficacy": "Approved for multiple myeloma, investigated in solid tumors"
            },
            "Bevacizumab": {
                "target": "VEGF",
                "mechanism": "Anti-inflammatory effects via angiogenesis inhibition",
                "effect": "indirect_inhibit",
                "efficacy": "Moderate"
            }
        },
        "clinical_notes": "Linked to tumor microenvironment and drug resistance"
    },
    
    "VEGF_Angiogenesis": {
        "name": "VEGF/Angiogenesis Pathway",
        "full_name": "Vascular Endothelial Growth Factor Signaling",
        "description": "Controls tumor angiogenesis and vascular permeability",
        "function": "New blood vessel formation, tumor nutrient supply",
        "components": [
            {"name": "VEGF-A", "type": "ligand", "role": "Primary angiogenic factor"},
            {"name": "VEGFR-2", "type": "receptor", "role": "Main signaling receptor"},
            {"name": "HIF-1α", "type": "transcription_factor", "role": "Hypoxia sensor, VEGF inducer"},
            {"name": "Angiopoietin-2", "type": "ligand", "role": "Vascular destabilization"}
        ],
        "activation_in_cancer": {
            "Colorectal": 0.70,  # High VEGF expression (PMID: 15746999)
            "Lung": 0.65,       # (PMID: 17909108)
            "Breast": 0.60,     # (PMID: 15583023)
            "Glioblastoma": 0.90,  # Highly vascular tumor (PMID: 17363563)
            "Pancreatic": 0.50  # (PMID: 19033854)
        },
        "drugs_targeting": {
            "Bevacizumab": {
                "target": "VEGF-A",
                "mechanism": "Monoclonal antibody neutralizing VEGF-A",
                "effect": "inhibit",
                "efficacy": "High in CRC + chemo (PMID: 15175435, Hurwitz 2004)"
            },
            "Aflibercept": {
                "target": "VEGF-A/VEGF-B/PlGF",
                "mechanism": "Decoy receptor trapping VEGF ligands",
                "effect": "inhibit",
                "efficacy": "Approved 2nd-line mCRC (VELOUR, PMID: 23169662)"
            },
            "Ramucirumab": {
                "target": "VEGFR-2",
                "mechanism": "Monoclonal antibody blocking VEGFR-2",
                "effect": "inhibit",
                "efficacy": "2nd-line CRC (RAISE, PMID: 25877006)"
            }
        },
        "clinical_notes": "Anti-VEGF is backbone of CRC treatment. Combined with FOLFOX or FOLFIRI."
    },
    
    "TGF_beta": {
        "name": "TGF-β Pathway",
        "full_name": "Transforming Growth Factor Beta",
        "description": "Dual role: tumor suppressor early, tumor promoter late stage",
        "function": "EMT, immune evasion, metastasis in advanced cancer",
        "components": [
            {"name": "TGF-β", "type": "ligand", "role": "Growth factor"},
            {"name": "TGFBR1/2", "type": "receptor", "role": "Serine/threonine kinase receptor"},
            {"name": "SMAD2/3", "type": "transcription_factor", "role": "Signal transducer"},
            {"name": "SMAD4", "type": "transcription_factor", "role": "Co-SMAD, often lost in CRC"}
        ],
        "activation_in_cancer": {
            "Colorectal": 0.55,  # SMAD4 loss ~30%, TGFBR2 mut in MSI-H (PMID: 25964097)
            "Pancreatic": 0.75,  # SMAD4 loss ~55% (PMID: 18794811)
            "Breast": 0.35,     # (PMID: 18414484)
            "Lung": 0.30        # (PMID: 22282465)
        },
        "drugs_targeting": {
            "5-FU": {
                "target": "Thymidylate synthase",
                "mechanism": "Cytotoxic against TGF-β-driven EMT cancer cells",
                "effect": "cytotoxic",
                "efficacy": "Standard CRC backbone"
            }
        },
        "clinical_notes": "SMAD4 loss indicates poor prognosis. MSI-H CRCs often have TGFBR2 frameshift mutations."
    },
    
    "Notch": {
        "name": "Notch Pathway",
        "full_name": "Notch Signaling",
        "description": "Cell fate determination and stem cell maintenance",
        "function": "Stem cell self-renewal, differentiation, angiogenesis",
        "components": [
            {"name": "Notch1-4", "type": "receptor", "role": "Transmembrane receptors"},
            {"name": "DLL1/3/4", "type": "ligand", "role": "Delta-like ligands"},
            {"name": "Jagged1/2", "type": "ligand", "role": "Serrate-like ligands"},
            {"name": "γ-secretase", "type": "protease", "role": "Notch cleavage and activation"}
        ],
        "activation_in_cancer": {
            "Colorectal": 0.40,  # Notch1 overexpr (PMID: 22855530)
            "Breast": 0.50,     # (PMID: 20068183)
            "Lung": 0.35,       # (PMID: 22282465)
            "Pancreatic": 0.45  # (PMID: 18794811)
        },
        "drugs_targeting": {
            "Irinotecan": {
                "target": "Topoisomerase I",
                "mechanism": "DNA damage in Notch-active rapidly dividing CSCs",
                "effect": "cytotoxic",
                "efficacy": "Effective in CRC with high Notch (PMID: 22855530)"
            }
        },
        "clinical_notes": "Notch crosstalk with Wnt drives intestinal stem cell compartment in CRC."
    }
}


# Pathway crosstalk and interactions
PATHWAY_CROSSTALK = {
    ("MAPK_ERK", "PI3K_AKT_mTOR"): {
        "interaction": "cooperative",
        "description": "Often co-activated, drive proliferation and survival",
        "synergy_potential": 0.8,
        "clinical_relevance": "Dual inhibition shows strong synergy (PMID: 22289917)"
    },
    ("p53", "MAPK_ERK"): {
        "interaction": "antagonistic",
        "description": "p53 arrests cells, MAPK promotes proliferation",
        "synergy_potential": 0.6,
        "clinical_relevance": "DNA damage + growth inhibition"
    },
    ("Wnt_beta_catenin", "PI3K_AKT_mTOR"): {
        "interaction": "cooperative",
        "description": "Both promote stem cell renewal",
        "synergy_potential": 0.7,
        "clinical_relevance": "Target cancer stem cells"
    },
    ("NF_kB", "PI3K_AKT_mTOR"): {
        "interaction": "cooperative",
        "description": "Both promote survival and inflammation",
        "synergy_potential": 0.75,
        "clinical_relevance": "Anti-inflammatory + anti-survival"
    },
    ("VEGF_Angiogenesis", "MAPK_ERK"): {
        "interaction": "cooperative",
        "description": "RAS-MAPK upregulates VEGF expression",
        "synergy_potential": 0.85,
        "clinical_relevance": "Anti-VEGF + anti-EGFR combo (PMID: 24718886)"
    },
    ("VEGF_Angiogenesis", "PI3K_AKT_mTOR"): {
        "interaction": "cooperative",
        "description": "PI3K/AKT promotes VEGF transcription via HIF-1α",
        "synergy_potential": 0.75,
        "clinical_relevance": "Anti-VEGF + mTOR inhibition"
    },
    ("TGF_beta", "Wnt_beta_catenin"): {
        "interaction": "cooperative",
        "description": "TGF-β and Wnt jointly promote EMT and stemness",
        "synergy_potential": 0.65,
        "clinical_relevance": "Metastasis and CSC targeting"
    },
    ("Notch", "Wnt_beta_catenin"): {
        "interaction": "cooperative",
        "description": "Notch and Wnt co-regulate intestinal stem cell fate",
        "synergy_potential": 0.70,
        "clinical_relevance": "Critical in CRC stem cell compartment (PMID: 19812184)"
    }
}


def get_pathway_info(pathway_id: str) -> Dict:
    """Get detailed pathway information"""
    return SIGNAL_PATHWAYS.get(pathway_id, {})


def get_pathways_for_cancer(cancer_type: str, min_activation: float = 0.3) -> List[str]:
    """
    Get pathways activated in a given cancer type
    
    Args:
        cancer_type: Type of cancer
        min_activation: Minimum activation threshold
        
    Returns:
        List of pathway IDs
    """
    activated = []
    for pathway_id, pathway in SIGNAL_PATHWAYS.items():
        activation = pathway.get("activation_in_cancer", {}).get(cancer_type, 0)
        if activation >= min_activation:
            activated.append(pathway_id)
    
    return activated


def get_drugs_for_pathway(pathway_id: str) -> Dict:
    """Get drugs targeting a specific pathway"""
    pathway = SIGNAL_PATHWAYS.get(pathway_id, {})
    return pathway.get("drugs_targeting", {})


def get_pathway_crosstalk(pathway1: str, pathway2: str) -> Dict:
    """Get interaction between two pathways"""
    key1 = (pathway1, pathway2)
    key2 = (pathway2, pathway1)
    return PATHWAY_CROSSTALK.get(key1, PATHWAY_CROSSTALK.get(key2, {}))
