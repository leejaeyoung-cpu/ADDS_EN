"""
Drug Database for Anticancer Cocktail System
"""

# Comprehensive drug database with mechanism of action
DRUG_DATABASE = {
    "Doxorubicin": {
        "class": "Anthracycline",
        "mechanism": "DNA intercalation, Topoisomerase II inhibition",
        "targets": ["DNA", "Top2"],
        "cancer_types": ["Breast", "Lung", "Leukemia", "Lymphoma"],
        "typical_dose": "20-75 mg/m²",
        "half_life": "1-3 days"
    },
    "Paclitaxel": {
        "class": "Taxane",
        "mechanism": "Microtubule stabilization",
        "targets": ["Tubulin"],
        "cancer_types": ["Breast", "Ovarian", "Lung"],
        "typical_dose": "135-175 mg/m²",
        "half_life": "13-52 hours"
    },
    "Cisplatin": {
        "class": "Platinum compound",
        "mechanism": "DNA crosslinking",
        "targets": ["DNA"],
        "cancer_types": ["Testicular", "Ovarian", "Bladder", "Lung"],
        "typical_dose": "50-100 mg/m²",
        "half_life": "0.5-3 hours"
    },
    "5-FU": {
        "class": "Antimetabolite",
        "mechanism": "Thymidylate synthase inhibition",
        "targets": ["TS", "RNA/DNA"],
        "cancer_types": ["Colorectal", "Breast", "Head and Neck"],
        "typical_dose": "300-450 mg/m²/day",
        "half_life": "10-20 minutes"
    },
    "Irinotecan": {
        "class": "Topoisomerase inhibitor",
        "mechanism": "Topoisomerase I inhibition",
        "targets": ["Top1"],
        "cancer_types": ["Colorectal", "Lung"],
        "typical_dose": "125-350 mg/m²",
        "half_life": "6-12 hours"
    },
    "Bevacizumab": {
        "class": "Monoclonal antibody (VEGF inhibitor)",
        "mechanism": "VEGF neutralization, angiogenesis inhibition",
        "targets": ["VEGF"],
        "cancer_types": ["Colorectal", "Lung", "Breast", "Glioblastoma"],
        "typical_dose": "5-15 mg/kg",
        "half_life": "20 days"
    },
    "Cetuximab": {
        "class": "Monoclonal antibody (EGFR inhibitor)",
        "mechanism": "EGFR blockade",
        "targets": ["EGFR"],
        "cancer_types": ["Colorectal", "Head and Neck"],
        "typical_dose": "250-400 mg/m²",
        "half_life": "112 hours"
    },
    "Gemcitabine": {
        "class": "Antimetabolite",
        "mechanism": "DNA synthesis inhibition",
        "targets": ["Ribonucleotide reductase"],
        "cancer_types": ["Pancreatic", "Lung", "Bladder", "Breast"],
        "typical_dose": "1000-1250 mg/m²",
        "half_life": "0.7-1.5 hours"
    },
    "Oxaliplatin": {
        "class": "Platinum compound",
        "mechanism": "DNA crosslinking",
        "targets": ["DNA"],
        "cancer_types": ["Colorectal"],
        "typical_dose": "85-130 mg/m²",
        "half_life": "14 minutes-391 hours"
    },
    "Capecitabine": {
        "class": "Antimetabolite (oral fluoropyrimidine)",
        "mechanism": "5-FU prodrug, converted to 5-FU preferentially in tumor (TP upregulated)",
        "targets": ["TS", "RNA/DNA"],
        "cancer_types": ["Colorectal", "Breast", "Gastric"],
        "typical_dose": "1000-1250 mg/m² BID",
        "half_life": "0.5-1 hours"
    },
    "Aflibercept": {
        "class": "Recombinant fusion protein (VEGF trap)",
        "mechanism": "Decoy receptor binding VEGF-A, VEGF-B, PlGF",
        "targets": ["VEGF-A", "VEGF-B", "PlGF"],
        "cancer_types": ["Colorectal"],
        "typical_dose": "4 mg/kg",
        "half_life": "6 days"
    },
    "Ramucirumab": {
        "class": "Monoclonal antibody (VEGFR-2 inhibitor)",
        "mechanism": "VEGFR-2 blockade, prevents VEGF-mediated signaling",
        "targets": ["VEGFR-2"],
        "cancer_types": ["Colorectal", "Gastric", "Lung"],
        "typical_dose": "8 mg/kg",
        "half_life": "14 days"
    },
    "Pembrolizumab": {
        "class": "Immune checkpoint inhibitor (anti-PD-1)",
        "mechanism": "PD-1 blockade, restores T-cell anti-tumor immunity",
        "targets": ["PD-1"],
        "cancer_types": ["Colorectal (MSI-H/dMMR)", "Lung", "Melanoma"],
        "typical_dose": "200 mg Q3W or 400 mg Q6W",
        "half_life": "26 days"
    },
    "Nivolumab": {
        "class": "Immune checkpoint inhibitor (anti-PD-1)",
        "mechanism": "PD-1 blockade, anti-tumor immunity restoration",
        "targets": ["PD-1"],
        "cancer_types": ["Colorectal (MSI-H/dMMR)", "Lung", "Melanoma", "Renal"],
        "typical_dose": "240 mg Q2W or 480 mg Q4W",
        "half_life": "25 days"
    },
    "Encorafenib": {
        "class": "Kinase inhibitor (BRAF V600E)",
        "mechanism": "Selective BRAF V600E kinase inhibition",
        "targets": ["BRAF V600E"],
        "cancer_types": ["Colorectal (BRAF V600E)", "Melanoma"],
        "typical_dose": "300 mg/day",
        "half_life": "3.5 hours"
    },
    "Regorafenib": {
        "class": "Multi-kinase inhibitor",
        "mechanism": "Inhibits VEGFR, PDGFR, FGFR, RAF, RET, KIT",
        "targets": ["VEGFR", "PDGFR", "FGFR", "RAF"],
        "cancer_types": ["Colorectal", "GIST", "HCC"],
        "typical_dose": "160 mg/day (3 weeks on, 1 week off)",
        "half_life": "28 hours"
    },
    "Trifluridine/Tipiracil": {
        "class": "Antimetabolite (oral nucleoside analogue)",
        "mechanism": "Thymidine phosphorylase inhibition + DNA incorporation",
        "targets": ["TP", "DNA"],
        "cancer_types": ["Colorectal"],
        "typical_dose": "35 mg/m² BID",
        "half_life": "1.4 hours"
    }
}

# Common drug combinations (synergistic) — NCCN Guidelines-based
KNOWN_COMBINATIONS = {
    "FOLFOX": {
        "drugs": ["5-FU", "Oxaliplatin"],
        "indication": "Colorectal cancer",
        "evidence_level": "High",
        "reference": "MOSAIC trial, PMID: 15175435"
    },
    "FOLFIRI": {
        "drugs": ["5-FU", "Irinotecan"],
        "indication": "Colorectal cancer",
        "evidence_level": "High",
        "reference": "Tournigand 2004, PMID: 14966096"
    },
    "FOLFOX+Bevacizumab": {
        "drugs": ["5-FU", "Oxaliplatin", "Bevacizumab"],
        "indication": "Colorectal cancer",
        "evidence_level": "High",
        "reference": "NO16966, PMID: 18172188"
    },
    "FOLFIRI+Bevacizumab": {
        "drugs": ["5-FU", "Irinotecan", "Bevacizumab"],
        "indication": "Colorectal cancer",
        "evidence_level": "High",
        "reference": "Hurwitz 2004, PMID: 15175435"
    },
    "FOLFOXIRI+Bevacizumab": {
        "drugs": ["5-FU", "Oxaliplatin", "Irinotecan", "Bevacizumab"],
        "indication": "Colorectal cancer",
        "evidence_level": "High",
        "reference": "TRIBE trial, PMID: 25286638"
    },
    "CAPOX": {
        "drugs": ["Capecitabine", "Oxaliplatin"],
        "indication": "Colorectal cancer",
        "evidence_level": "High",
        "reference": "XELOXA trial, PMID: 21986492"
    },
    "BEACON-CRC": {
        "drugs": ["Encorafenib", "Cetuximab"],
        "indication": "Colorectal cancer (BRAF V600E)",
        "evidence_level": "High",
        "reference": "BEACON CRC, PMID: 31566309"
    },
    "Pembrolizumab-MSI-H": {
        "drugs": ["Pembrolizumab"],
        "indication": "Colorectal cancer (MSI-H/dMMR)",
        "evidence_level": "High",
        "reference": "KEYNOTE-177, PMID: 33501440"
    },
    "Trifluridine-mono": {
        "drugs": ["Trifluridine/Tipiracil"],
        "indication": "Colorectal cancer (3rd line+)",
        "evidence_level": "High",
        "reference": "RECOURSE trial, PMID: 25970009"
    },
    "AC": {
        "drugs": ["Doxorubicin", "Cisplatin"],
        "indication": "Breast cancer",
        "evidence_level": "High"
    },
    "Gem-Cis": {
        "drugs": ["Gemcitabine", "Cisplatin"],
        "indication": "Lung cancer",
        "evidence_level": "High"
    }
}

# Antagonistic combinations (to avoid)
ANTAGONISTIC_PAIRS = [
    ("5-FU", "Gemcitabine"),      # Same pathway antimetabolite competition
    ("5-FU", "Capecitabine"),     # Capecitabine IS a 5-FU prodrug — redundant
    ("Cetuximab", "Bevacizumab"), # CAIRO2/PACCE: worse outcomes when combined (PMID: 19826127)
]

def get_drug_info(drug_name: str) -> dict:
    """Get detailed drug information"""
    return DRUG_DATABASE.get(drug_name, {})

def get_compatible_drugs(drug_name: str) -> list:
    """Get list of drugs compatible with given drug"""
    all_drugs = list(DRUG_DATABASE.keys())
    compatible = []
    
    for drug in all_drugs:
        if drug != drug_name:
            # Check if pair is antagonistic
            if (drug_name, drug) not in ANTAGONISTIC_PAIRS and (drug, drug_name) not in ANTAGONISTIC_PAIRS:
                compatible.append(drug)
    
    return compatible

def suggest_combinations(cancer_type: str, num_drugs: int = 2) -> list:
    """Suggest drug combinations for specific cancer type"""
    suggestions = []
    
    # Find drugs effective for this cancer type
    effective_drugs = [
        drug for drug, info in DRUG_DATABASE.items()
        if cancer_type in info.get("cancer_types", [])
    ]
    
    # Return known combinations first
    for combo_name, combo_info in KNOWN_COMBINATIONS.items():
        if combo_info["indication"] == f"{cancer_type} cancer":
            suggestions.append({
                "name": combo_name,
                "drugs": combo_info["drugs"],
                "evidence": combo_info["evidence_level"]
            })
    
    return suggestions
