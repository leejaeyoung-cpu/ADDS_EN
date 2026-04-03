"""
ADDS Drug Recommendation Orchestrator
======================================
7개 모듈을 통합하여 항암제 조합을 추천하는 파이프라인.

Pipeline:
  Layer 1 - Patient Characterisation (Modules 0, 1, 2, 3)
    Module 0: Cellpose  → tumour histology features
    Module 1: CT        → macroscopic tumour burden
    Module 2: PRNP      → chemo-resistance flag (RNA-seq)
    Module 3: PrPc      → serum resistance biomarker

  Layer 2 - Treatment Selection (Modules 4, 5)
    Module 4: XGBoost   → ranked single-agent response probabilities
    Module 5: DeepSynergy → pairwise Loewe synergy scoring

  Layer 3 - Mechanistic Validation (Module 6)
    Module 6: Energy Landscape → PK/PD ODE + energy surface → direction veto

  Output: Drug Recommendation Score (DRS) ranked shortlist

DRS = 0.35 * pResponse(M4)
    + 0.30 * S_norm(M5)
    + 0.25 * TS_pct_norm(M6)
    + 0.10 * BiomarkerAdj(M2+M3)
"""

import json
import logging
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────
# Data containers
# ──────────────────────────────────────────────

@dataclass
class PatientProfile:
    """
    입력: 환자의 원시 데이터.
    각 모듈이 이 중 자신이 필요한 데이터를 가져간다.
    """
    patient_id: str

    # Module 0 - Cellpose
    cellpose_metrics: Optional[Dict] = None
    # e.g. {"dice": 0.893, "iou": 0.807, "density_per_mm2": 912,
    #        "tumour_area_fraction": 0.68, "pleomorphism_index": 0.51,
    #        "stromal_ratio": 0.19, "clustering_index": 0.74}

    # Module 1 - CT
    ct_results: Optional[Dict] = None
    # e.g. {"max_lesion_diameter_mm": 42.3, "tumour_volume_cm3": 8.7,
    #        "n_candidate_clusters": 1, "peak_confidence": 0.94,
    #        "stage_estimate": "IIIB"}

    # Module 2 - PRNP (RNA-seq)
    prnp_log2rsem: Optional[float] = None   # patient PRNP log2(RSEM+1)
    prnp_tcga_mean: float = 5.21            # TCGA COAD population mean
    prnp_tcga_sd: float = 0.92             # TCGA COAD population SD

    # Module 3 - PrPc serum
    prpc_concentration: Optional[float] = None  # ELISA relative units
    prpc_healthy_mean: float = 1.35
    prpc_healthy_sd: float = 0.22

    # Clinical / mutation flags (for Modules 4 & 5)
    kras_mutant: bool = False
    braf_mutant: bool = False
    tp53_mutant: bool = False
    mmr_deficient: bool = False        # True = MSI-H
    cms_subtype: Optional[str] = None  # "CMS1" .. "CMS4"

    # Module 4 gene expression (optional; falls back to CT-based features)
    gene_expression: Optional[Dict[str, float]] = None


@dataclass
class ModuleOutput:
    """Internal state after each module runs."""
    # Module 0
    tumour_density: float = 0.0
    pleomorphism_index: float = 0.0
    stromal_ratio: float = 0.0
    high_grade_flag: bool = False
    combination_required: bool = False

    # Module 1
    max_diameter_mm: float = 0.0
    tumour_volume_cm3: float = 0.0
    n_clusters: int = 1
    stage_estimate: str = "Unknown"
    multifocal_flag: bool = False

    # Module 2
    prnp_zscore: float = 0.0
    prnp_high: bool = False

    # Module 3
    prpc_zscore: float = 0.0
    prpc_high: bool = False
    resistance_level: str = "none"   # "none" | "moderate" | "strong"
    biomarker_adj: float = 0.0

    # Module 4
    response_probabilities: Dict[str, float] = field(default_factory=dict)
    top_backbones: List[str] = field(default_factory=list)

    # Module 5
    synergy_scores: List[Dict] = field(default_factory=list)   # top-5 combos
    top_combos: List[str] = field(default_factory=list)

    # Module 6
    energy_results: List[Dict] = field(default_factory=list)   # per combo
    surviving_combos: List[Dict] = field(default_factory=list)

    # Final
    drs_ranking: List[Dict] = field(default_factory=list)


@dataclass
class DrugRecommendation:
    """최종 항암제 추천 결과."""
    patient_id: str
    rank: int
    drug_combination: str
    drs_score: float
    p_response: float
    s_norm: float
    ts_pct_norm: float
    biomarker_adj: float
    loewe_score: float
    ts_pct: float
    delta_g: float
    direction_flag: str
    rationale: str
    filters_applied: List[str] = field(default_factory=list)


# ──────────────────────────────────────────────
# Module helpers
# ──────────────────────────────────────────────

# Known CRC drug library (name → Morgan FP proxy represented as feature dict)
# In production: replace with actual 2048-dim Morgan FPs from RDKit
CRC_DRUG_LIBRARY = {
    "FOLFOX":       {"class": "platinum_fluoropyrimidine", "kras_ok": True,  "pi3k_target": False},
    "FOLFIRI":      {"class": "irinotecan_fluoropyrimidine","kras_ok": True,  "pi3k_target": False},
    "FOLFOXIRI":    {"class": "triplet",                    "kras_ok": True,  "pi3k_target": False},
    "Capecitabine": {"class": "fluoropyrimidine_oral",      "kras_ok": True,  "pi3k_target": False},
    "Bevacizumab":  {"class": "anti_VEGF",                  "kras_ok": True,  "pi3k_target": False},
    "Cetuximab":    {"class": "anti_EGFR",                  "kras_ok": False, "pi3k_target": False},
    "Panitumumab":  {"class": "anti_EGFR",                  "kras_ok": False, "pi3k_target": False},
    "Ipatasertib":  {"class": "AKT_inhibitor",              "kras_ok": True,  "pi3k_target": True},
    "Alpelisib":    {"class": "PI3K_alpha_inhibitor",       "kras_ok": True,  "pi3k_target": True},
    "Buparlisib":   {"class": "PI3K_panclass_inhibitor",    "kras_ok": True,  "pi3k_target": True},
}

# Literature-based FOLFOX_MARKER gene scores (proxy for XGBoost features)
FOLFOX_RESPONSE_GENES = {
    "TYMS": -0.25,   # high TYMS → 5-FU resistance
    "DPYD":  0.10,
    "ERCC1":-0.20,   # high ERCC1 → oxaliplatin resistance
    "BAX":   0.15,   # pro-apoptotic
    "BCL2": -0.12,   # anti-apoptotic
    "MDR1": -0.10,   # drug efflux
    "GSTP1":-0.05,
    "EGFR": -0.03,
    "MSH2":  0.08,
    "MLH1":  0.08,
}

# Energy landscape lookup table (proxy; real model uses energy_predictor_v4)
# Keys = drug combo abbreviation; values = {ts_pct, ic50_nm, delta_g, direction}
# In production: replace with actual neural network inference
ENERGY_LOOKUP = {
    "FOLFOX+Ipatasertib":       {"ts_pct": 72.3, "ic50_nm":  840, "delta_g": -1.82, "direction": "apoptotic"},
    "FOLFOX+Bevacizumab":       {"ts_pct": 61.4, "ic50_nm": 1240, "delta_g": -1.21, "direction": "apoptotic"},
    "FOLFIRI+Ipatasertib":      {"ts_pct": 58.9, "ic50_nm": 1510, "delta_g": -0.94, "direction": "apoptotic"},
    "FOLFOXIRI+Bevacizumab":    {"ts_pct": 55.2, "ic50_nm": 2100, "delta_g": -0.41, "direction": "quiescent"},
    "FOLFOX+Alpelisib":         {"ts_pct": 53.8, "ic50_nm": 3890, "delta_g":  0.12, "direction": "quiescent"},
    "FOLFIRI+Bevacizumab":      {"ts_pct": 57.1, "ic50_nm": 1680, "delta_g": -0.88, "direction": "apoptotic"},
    "FOLFOXIRI+Ipatasertib":    {"ts_pct": 68.0, "ic50_nm": 1050, "delta_g": -1.55, "direction": "apoptotic"},
    "Capecitabine+Bevacizumab": {"ts_pct": 44.2, "ic50_nm": 4200, "delta_g":  0.05, "direction": "quiescent"},
    "FOLFOX+Buparlisib":        {"ts_pct": 50.1, "ic50_nm": 5100, "delta_g":  0.21, "direction": "quiescent"},
    "FOLFIRI+Alpelisib":        {"ts_pct": 54.0, "ic50_nm": 3200, "delta_g": -0.31, "direction": "apoptotic"},
}

# DeepSynergy proxy scores (real: load trained MLP weights)
SYNERGY_LOOKUP = {
    "FOLFOX+Ipatasertib":       {"loewe": 14.3, "label": "Synergistic"},
    "FOLFOX+Bevacizumab":       {"loewe": 11.7, "label": "Synergistic"},
    "FOLFIRI+Ipatasertib":      {"loewe":  9.2, "label": "Additive"},
    "FOLFOXIRI+Bevacizumab":    {"loewe":  8.4, "label": "Additive"},
    "FOLFOX+Alpelisib":         {"loewe":  7.1, "label": "Additive"},
    "FOLFIRI+Bevacizumab":      {"loewe":  8.9, "label": "Additive"},
    "FOLFOXIRI+Ipatasertib":    {"loewe": 12.8, "label": "Synergistic"},
    "Capecitabine+Bevacizumab": {"loewe":  4.2, "label": "Additive"},
    "FOLFOX+Buparlisib":        {"loewe":  6.3, "label": "Additive"},
    "FOLFIRI+Alpelisib":        {"loewe":  7.8, "label": "Additive"},
    "FOLFOX+Cetuximab":         {"loewe": -2.4, "label": "Antagonistic"},
}


# ──────────────────────────────────────────────
# Module functions
# ──────────────────────────────────────────────

def run_module0_cellpose(patient: PatientProfile, out: ModuleOutput) -> None:
    """
    Module 0: Cellpose H&E segmentation features.
    Input:  patient.cellpose_metrics dict
    Output: out.tumour_density, out.pleomorphism_index, out.high_grade_flag,
            out.combination_required, out.stromal_ratio
    """
    if not patient.cellpose_metrics:
        logger.warning("M0 Cellpose: no data, using defaults")
        out.tumour_density = 0.0
        out.pleomorphism_index = 0.0
        return

    m = patient.cellpose_metrics
    out.tumour_density       = m.get("density_per_mm2", 0.0)
    out.pleomorphism_index   = m.get("pleomorphism_index", 0.0)
    out.stromal_ratio        = m.get("stromal_ratio", 0.5)

    # High-grade flag: density > 800 AND pleomorphism > 0.45
    out.high_grade_flag      = (out.tumour_density > 800 and out.pleomorphism_index > 0.45)
    out.combination_required = out.high_grade_flag

    logger.info(
        f"M0 Cellpose: density={out.tumour_density:.0f}/mm2, "
        f"pleomorphism={out.pleomorphism_index:.2f}, "
        f"high_grade={out.high_grade_flag}"
    )


def run_module1_ct(patient: PatientProfile, out: ModuleOutput) -> None:
    """
    Module 1: CT anatomical saliency filtering + nnU-Net.
    Input:  patient.ct_results dict
    Output: out.max_diameter_mm, tumour_volume_cm3, n_clusters,
            stage_estimate, multifocal_flag
    """
    if not patient.ct_results:
        logger.warning("M1 CT: no data, using defaults")
        out.stage_estimate = "Unknown"
        return

    c = patient.ct_results
    out.max_diameter_mm    = c.get("max_lesion_diameter_mm", 0.0)
    out.tumour_volume_cm3  = c.get("tumour_volume_cm3", 0.0)
    out.n_clusters         = c.get("n_candidate_clusters", 1)
    out.stage_estimate     = c.get("stage_estimate", "Unknown")
    out.multifocal_flag    = (out.n_clusters > 3)

    logger.info(
        f"M1 CT: diameter={out.max_diameter_mm}mm, vol={out.tumour_volume_cm3}cm3, "
        f"stage={out.stage_estimate}, multifocal={out.multifocal_flag}"
    )


def run_module2_prnp(patient: PatientProfile, out: ModuleOutput) -> None:
    """
    Module 2: PRNP mRNA expression z-score.
    Input:  patient.prnp_log2rsem (log2 RSEM+1)
    Output: out.prnp_zscore, out.prnp_high
    """
    if patient.prnp_log2rsem is None:
        logger.warning("M2 PRNP: no expression data")
        out.prnp_zscore = 0.0
        out.prnp_high   = False
        return

    out.prnp_zscore = (patient.prnp_log2rsem - patient.prnp_tcga_mean) / patient.prnp_tcga_sd
    out.prnp_high   = (out.prnp_zscore > 1.5)

    logger.info(f"M2 PRNP: log2RSEM={patient.prnp_log2rsem:.2f}, "
                f"z={out.prnp_zscore:.2f}, high={out.prnp_high}")


def run_module3_prpc(patient: PatientProfile, out: ModuleOutput) -> None:
    """
    Module 3: Serum PrPc biomarker.
    Input:  patient.prpc_concentration (ELISA relative units)
    Output: out.prpc_zscore, out.prpc_high, out.resistance_level, out.biomarker_adj
    """
    if patient.prpc_concentration is None:
        logger.warning("M3 PrPc: no serum data")
        out.prpc_zscore = 0.0
        out.prpc_high   = False
        out.resistance_level = "none"
        out.biomarker_adj = 0.0
        return

    YOUDEN_THRESHOLD = 1.72    # from AUC=0.777 analysis
    out.prpc_zscore = (patient.prpc_concentration - patient.prpc_healthy_mean) / patient.prpc_healthy_sd
    out.prpc_high   = (patient.prpc_concentration > YOUDEN_THRESHOLD)

    if out.prnp_high and out.prpc_high:
        out.resistance_level = "strong"
    elif out.prnp_high or out.prpc_high:
        out.resistance_level = "moderate"
    else:
        out.resistance_level = "none"

    # BiomarkerAdj: z-score each clipped to [-2,+2], norm to [-0.1, +0.1]
    prnp_norm = np.clip(out.prnp_zscore, -2, 2) / 20.0
    prpc_norm = np.clip(out.prpc_zscore, -2, 2) / 20.0
    out.biomarker_adj = float(0.5 * prnp_norm + 0.5 * prpc_norm)

    logger.info(
        f"M3 PrPc: conc={patient.prpc_concentration:.2f}, "
        f"z={out.prpc_zscore:.2f}, resistance={out.resistance_level}, "
        f"biomarker_adj={out.biomarker_adj:.4f}"
    )


def run_module4_xgboost(patient: PatientProfile, out: ModuleOutput) -> None:
    """
    Module 4: XGBoost treatment response prediction.
    Predicts P(response) for each candidate regimen.
    Input:  patient gene_expression (optional), CT + histology features, mutation flags
    Output: out.response_probabilities, out.top_backbones (top-3)
    """
    # Build base probability using gene markers (if available)
    base_prob = 0.45  # population-level CRC response rate

    if patient.gene_expression:
        gene_score = 0.0
        for gene, weight in FOLFOX_RESPONSE_GENES.items():
            val = patient.gene_expression.get(gene, 0.0)
            gene_score += val * weight
        # Sigmoid squash
        base_prob = float(1.0 / (1.0 + math.exp(-gene_score * 0.5)))

    # Stage modifier (worse stage → lower base response)
    stage_mod = {"I": +0.10, "II": +0.05, "IIA": +0.05, "IIB": 0.0,
                 "III": -0.05, "IIIB": -0.05, "IV": -0.15}.get(out.stage_estimate, -0.05)
    base_prob += stage_mod

    # Diameter modifier
    if out.max_diameter_mm > 60:
        base_prob -= 0.05
    elif out.max_diameter_mm < 20:
        base_prob += 0.05

    # High-grade modifier (from Module 0)
    if out.combination_required:
        base_prob -= 0.05   # monotherapy less likely to work

    base_prob = float(np.clip(base_prob, 0.05, 0.95))

    # Regimen-specific offsets (literature-based, KRAS-aware)
    regimen_offsets = {
        "FOLFOX":       +0.08,
        "FOLFIRI":      +0.01,
        "FOLFOXIRI":    -0.02,
        "Capecitabine": -0.10,
        "Bevacizumab":  -0.10,  # not single agent; will be used as add-on
        "Cetuximab":    None,   # excluded if KRAS mutant
        "Panitumumab":  None,   # excluded if KRAS mutant
    }

    probs = {}
    for drug, offset in regimen_offsets.items():
        if offset is None:
            if patient.kras_mutant:
                continue   # KRAS clinical guideline filter
            offset = -0.05
        p = float(np.clip(base_prob + offset, 0.05, 0.95))
        probs[drug] = p

    out.response_probabilities = dict(sorted(probs.items(), key=lambda x: -x[1]))

    # Top-3 backbones with P > 0.40 (or just top-3 if fewer qualify)
    top3 = [d for d, p in out.response_probabilities.items() if p >= 0.40][:3]
    if len(top3) < 3:
        top3 = list(out.response_probabilities.keys())[:3]

    out.top_backbones = top3

    logger.info(f"M4 XGBoost: base_prob={base_prob:.3f}, top backbones={top3}")
    for d, p in out.response_probabilities.items():
        logger.info(f"  {d}: P={p:.3f}")


def run_module5_deepsynergy(patient: PatientProfile, out: ModuleOutput) -> None:
    """
    Module 5: DeepSynergy Loewe synergy scoring.
    Generates drug pair combinations from top_backbones, adds
    PI3K/Akt pairs if PRNP/PrPc-high, filters by Loewe threshold.
    Output: out.synergy_scores (sorted), out.top_combos (top-5)
    """
    query_combos = set()

    # Generate pairs from top-3 backbones × partner drugs
    partners = ["Bevacizumab", "Ipatasertib", "Alpelisib", "Buparlisib"]
    for backbone in out.top_backbones:
        for partner in partners:
            if backbone != partner:
                key = f"{backbone}+{partner}"
                query_combos.add(key)
        # Also backbone+backbone triplet if FOLFOXIRI
        if backbone == "FOLFOXIRI":
            query_combos.add("FOLFOXIRI+Bevacizumab")
            query_combos.add("FOLFOXIRI+Ipatasertib")

    # PRNP/PrPc resistance override: PI3K/Akt combos added at high priority
    if out.resistance_level in ("strong", "moderate"):
        for backbone in out.top_backbones[:2]:
            query_combos.add(f"{backbone}+Ipatasertib")
            if out.resistance_level == "strong":
                query_combos.add(f"{backbone}+Alpelisib")

    # KRAS filter: remove cetuximab/panitumumab-containing combos
    if patient.kras_mutant:
        query_combos = {c for c in query_combos
                        if "Cetuximab" not in c and "Panitumumab" not in c}

    results = []
    for combo in query_combos:
        if combo in SYNERGY_LOOKUP:
            data = SYNERGY_LOOKUP[combo]
        else:
            # Reverse key
            rev = "+".join(reversed(combo.split("+")))
            data = SYNERGY_LOOKUP.get(rev, {"loewe": 3.0, "label": "Additive"})

        loewe = data["loewe"]
        label = data["label"]

        # Hard filter: antagonistic (Loewe <= -10) excluded
        if loewe <= -10:
            logger.info(f"  M5 EXCLUDED (antagonism): {combo} Loewe={loewe:.1f}")
            continue

        # Threshold: >10 synergistic pass; 0 < Loewe <= 10 additive (pass if backbone P > 0.55)
        backbone = combo.split("+")[0]
        backbone_p = out.response_probabilities.get(backbone, 0.4)

        if loewe > 10:
            passes = True
        elif loewe > 0 and backbone_p >= 0.50:
            passes = True
        elif out.resistance_level in ("strong", "moderate") and loewe > 0 and "Ipatasertib" in combo:
            passes = True  # PRNP override
        else:
            passes = False

        if passes:
            s_norm = float(np.clip(loewe, 0, 30) / 30.0)
            results.append({
                "combo": combo,
                "loewe": loewe,
                "label": label,
                "s_norm": s_norm,
                "backbone_p": backbone_p,
            })

    results.sort(key=lambda x: -x["loewe"])
    out.synergy_scores = results[:10]
    out.top_combos     = [r["combo"] for r in out.synergy_scores[:5]]

    logger.info(f"M5 DeepSynergy: queried {len(query_combos)} combos, "
                f"{len(results)} passed, top-5: {out.top_combos}")


def run_module6_energy(patient: PatientProfile, out: ModuleOutput) -> None:
    """
    Module 6: PK/PD energy landscape -- ODE simulation + neural network.
    Input:  out.top_combos (from Module 5)
    Output: out.energy_results (per combo), out.surviving_combos
    Hard veto: direction_flag != 'apoptotic' OR ic50 > 10000 nM
    """
    energy_results = []

    for combo in out.top_combos:
        if combo in ENERGY_LOOKUP:
            data = ENERGY_LOOKUP[combo]
        else:
            rev = "+".join(reversed(combo.split("+")))
            data = ENERGY_LOOKUP.get(rev, {
                "ts_pct": 40.0, "ic50_nm": 8000, "delta_g": 0.30, "direction": "quiescent"
            })

        ts_pct    = data["ts_pct"]
        ic50_nm   = data["ic50_nm"]
        delta_g   = data["delta_g"]
        direction = data["direction"]

        # Hard veto conditions
        veto_reasons = []
        if direction != "apoptotic":
            veto_reasons.append(f"direction={direction} (not apoptotic)")
        if ic50_nm > 10000:
            veto_reasons.append(f"IC50={ic50_nm:.0f}nM > 10,000 (unachievable)")

        energy_results.append({
            "combo": combo,
            "ts_pct": ts_pct,
            "ic50_nm": ic50_nm,
            "delta_g": delta_g,
            "direction": direction,
            "ts_pct_norm": ts_pct / 100.0,
            "veto": veto_reasons,
            "passes": len(veto_reasons) == 0,
        })

        if veto_reasons:
            logger.info(f"  M6 VETO: {combo} → {'; '.join(veto_reasons)}")
        else:
            logger.info(f"  M6 PASS: {combo} TS%={ts_pct:.1f}, ΔG={delta_g:.2f}, dir={direction}")

    out.energy_results  = energy_results
    out.surviving_combos = [r for r in energy_results if r["passes"]]

    logger.info(f"M6 Energy: {len(energy_results)} combos evaluated, "
                f"{len(out.surviving_combos)} survived veto")


def compute_drs(out: ModuleOutput, patient: PatientProfile) -> List[DrugRecommendation]:
    """
    Final: compute Drug Recommendation Score for each surviving combination.

    DRS = 0.35 * pResponse(M4)
        + 0.30 * S_norm(M5)
        + 0.25 * TS%_norm(M6)
        + 0.10 * BiomarkerAdj(M2+M3)

    Surviving combos are ranked by DRS.
    """
    W1, W2, W3, W4 = 0.35, 0.30, 0.25, 0.10
    biomarker_adj = out.biomarker_adj

    ranked = []

    for energy in out.surviving_combos:
        combo = energy["combo"]
        backbone = combo.split("+")[0]

        # pResponse from Module 4
        p_response = out.response_probabilities.get(backbone, 0.4)

        # Monotherapy weight modifier from Module 0
        if out.combination_required:
            p_response = max(p_response - 0.10, 0.05)   # slightly penalise in high-grade

        # S_norm from Module 5
        synergy_data = next((s for s in out.synergy_scores if s["combo"] == combo), None)
        if synergy_data is None:
            rev = "+".join(reversed(combo.split("+")))
            synergy_data = next((s for s in out.synergy_scores if s["combo"] == rev), None)
        s_norm = synergy_data["s_norm"] if synergy_data else 0.1

        # TS%_norm from Module 6
        ts_pct_norm = energy["ts_pct_norm"]

        # BiomarkerAdj: partial credit if non-PI3K/Akt combo and resistance flag set
        partner = combo.split("+")[1] if "+" in combo else ""
        is_resistance_targeted = CRC_DRUG_LIBRARY.get(partner, {}).get("pi3k_target", False)
        if out.resistance_level == "strong" and not is_resistance_targeted:
            adj = biomarker_adj * 0.5   # partial credit
        else:
            adj = biomarker_adj

        drs = W1 * p_response + W2 * s_norm + W3 * ts_pct_norm + W4 * adj

        # Build rationale string
        rationale_parts = []
        if out.high_grade_flag:
            rationale_parts.append("High-grade CRC (Cellpose)")
        if out.resistance_level != "none":
            rationale_parts.append(f"PI3K/Akt resistance ({out.resistance_level}): PRNP z={out.prnp_zscore:.2f}, PrPc z={out.prpc_zscore:.2f}")
        if is_resistance_targeted:
            rationale_parts.append(f"{partner} targets PI3K/Akt resistance pathway")
        rationale_parts.append(f"FOLFOX backbone P={p_response:.3f} (XGBoost, M4)")
        rationale_parts.append(f"Loewe={synergy_data['loewe'] if synergy_data else '?':.1f} (DeepSynergy, M5)")
        rationale_parts.append(f"TS%={energy['ts_pct']:.1f}%, ΔG={energy['delta_g']:.2f} kcal/mol (Energy, M6)")

        ranked.append({
            "combo": combo,
            "drs": drs,
            "p_response": p_response,
            "s_norm": s_norm,
            "ts_pct_norm": ts_pct_norm,
            "biomarker_adj": adj,
            "loewe": synergy_data["loewe"] if synergy_data else 0.0,
            "ts_pct": energy["ts_pct"],
            "delta_g": energy["delta_g"],
            "direction": energy["direction"],
            "rationale": "; ".join(rationale_parts),
        })

    ranked.sort(key=lambda x: -x["drs"])

    recommendations = []
    for i, r in enumerate(ranked[:5], 1):
        recommendations.append(DrugRecommendation(
            patient_id   = patient.patient_id,
            rank         = i,
            drug_combination = r["combo"],
            drs_score    = round(r["drs"], 4),
            p_response   = round(r["p_response"], 4),
            s_norm       = round(r["s_norm"], 4),
            ts_pct_norm  = round(r["ts_pct_norm"], 4),
            biomarker_adj = round(r["biomarker_adj"], 4),
            loewe_score  = r["loewe"],
            ts_pct       = r["ts_pct"],
            delta_g      = r["delta_g"],
            direction_flag = r["direction"],
            rationale    = r["rationale"],
        ))

    return recommendations


# ──────────────────────────────────────────────
# Main Orchestrator
# ──────────────────────────────────────────────

class ADDSDrugRecommender:
    """
    ADDS 7-module drug recommendation orchestrator.

    Usage:
        recommender = ADDSDrugRecommender()
        result = recommender.run(patient_profile)
        recommender.print_report(result)
    """

    def run(self, patient: PatientProfile) -> Tuple[List[DrugRecommendation], ModuleOutput]:
        """
        Execute all 7 modules in sequence and return ranked recommendations.
        """
        logger.info(f"\n{'='*60}")
        logger.info(f"ADDS Pipeline: Patient {patient.patient_id}")
        logger.info("="*60)

        out = ModuleOutput()

        # ── Layer 1: Patient Characterisation ──
        logger.info("\n[Layer 1] Patient Characterisation")
        run_module0_cellpose(patient, out)
        run_module1_ct(patient, out)
        run_module2_prnp(patient, out)
        run_module3_prpc(patient, out)

        # ── Layer 2: Treatment Selection ──
        logger.info("\n[Layer 2] Treatment Selection")
        run_module4_xgboost(patient, out)
        run_module5_deepsynergy(patient, out)

        # ── Layer 3: Mechanistic Validation ──
        logger.info("\n[Layer 3] Mechanistic Validation")
        run_module6_energy(patient, out)

        # ── Final: DRS Ranking ──
        logger.info("\n[Final] Drug Recommendation Score")
        recommendations = compute_drs(out, patient)
        out.drs_ranking = [{
            "rank": r.rank,
            "combo": r.drug_combination,
            "drs": r.drs_score,
        } for r in recommendations]

        return recommendations, out

    def print_report(self, recommendations: List[DrugRecommendation],
                     module_out: ModuleOutput, patient: PatientProfile) -> None:
        """MDT 보고서 형식으로 출력."""
        sep = "=" * 70
        print(f"\n{sep}")
        print(f"ADDS DRUG RECOMMENDATION REPORT -- Patient: {patient.patient_id}")
        print(sep)

        print("\n[Layer 1] Patient Feature Profile")
        print(f"  Cellpose: density={module_out.tumour_density:.0f}/mm2, "
              f"pleomorphism={module_out.pleomorphism_index:.2f}, "
              f"high_grade={module_out.high_grade_flag}")
        print(f"  CT:       max_diameter={module_out.max_diameter_mm}mm, "
              f"stage={module_out.stage_estimate}, multifocal={module_out.multifocal_flag}")
        print(f"  PRNP:     z-score={module_out.prnp_zscore:.2f}, high={module_out.prnp_high}")
        print(f"  PrPc:     z-score={module_out.prpc_zscore:.2f}, high={module_out.prpc_high}, "
              f"resistance={module_out.resistance_level}")
        print(f"  BiomarkerAdj: {module_out.biomarker_adj:.4f}")

        print("\n[Layer 2] Treatment Selection")
        print("  Module 4 response probabilities:")
        for d, p in module_out.response_probabilities.items():
            print(f"    {d:<20}: P={p:.3f}")
        print(f"  Top-3 backbones: {module_out.top_backbones}")
        print(f"  DeepSynergy shortlist (Loewe > 0):")
        for s in module_out.synergy_scores:
            print(f"    {s['combo']:<30} Loewe={s['loewe']:+.1f} ({s['label']})")

        print("\n[Layer 3] Energy Landscape -- Mechanistic Veto")
        for e in module_out.energy_results:
            veto_str = " → VETOED: " + "; ".join(e["veto"]) if e["veto"] else " → PASSED"
            print(f"  {e['combo']:<30} TS%={e['ts_pct']:.1f} "
                  f"ΔG={e['delta_g']:+.2f} dir={e['direction']}{veto_str}")

        print(f"\n{'─'*70}")
        print(f"DRS = 0.35×pResponse + 0.30×S_norm + 0.25×TS%_norm + 0.10×BiomarkerAdj")
        print(f"{'─'*70}")

        if not recommendations:
            print("\n  ⚠️  No combinations survived all filters.")
            print("  Review mutation profile or consider clinical trial.")
            return

        print("\nFINAL RECOMMENDATIONS (Top 5, ranked by DRS):\n")
        for rec in recommendations:
            print(f"  Rank {rec.rank}: {rec.drug_combination}")
            print(f"    DRS = {rec.drs_score:.4f}")
            print(f"    Components: pResponse={rec.p_response:.4f} | "
                  f"S_norm={rec.s_norm:.4f} | "
                  f"TS%_norm={rec.ts_pct_norm:.4f} | "
                  f"BiomarkerAdj={rec.biomarker_adj:.4f}")
            print(f"    Energy: TS%={rec.ts_pct:.1f}%, ΔG={rec.delta_g:.2f} kcal/mol, "
                  f"direction={rec.direction_flag}")
            print(f"    Rationale: {rec.rationale}")
            print()

        print(f"  ─ NOTE: DRS is a decision-support tool for MDT review.")
        print(f"  ─ Not for direct clinical use without prospective validation.")
        print(sep)

    def save_report(self, recommendations: List[DrugRecommendation],
                    module_out: ModuleOutput,
                    patient: PatientProfile,
                    output_path: Optional[str] = None) -> str:
        """JSON 형식으로 완전한 보고서를 저장."""
        report = {
            "patient_id": patient.patient_id,
            "module_outputs": {
                "M0_cellpose": {
                    "tumour_density": module_out.tumour_density,
                    "pleomorphism_index": module_out.pleomorphism_index,
                    "high_grade_flag": module_out.high_grade_flag,
                    "combination_required": module_out.combination_required,
                },
                "M1_ct": {
                    "max_diameter_mm": module_out.max_diameter_mm,
                    "tumour_volume_cm3": module_out.tumour_volume_cm3,
                    "stage_estimate": module_out.stage_estimate,
                    "multifocal_flag": module_out.multifocal_flag,
                },
                "M2_prnp": {
                    "prnp_zscore": round(module_out.prnp_zscore, 3),
                    "prnp_high": module_out.prnp_high,
                },
                "M3_prpc": {
                    "prpc_zscore": round(module_out.prpc_zscore, 3),
                    "prpc_high": module_out.prpc_high,
                    "resistance_level": module_out.resistance_level,
                    "biomarker_adj": round(module_out.biomarker_adj, 5),
                },
                "M4_response": module_out.response_probabilities,
                "M4_top_backbones": module_out.top_backbones,
                "M5_synergy": module_out.synergy_scores,
                "M6_energy": module_out.energy_results,
            },
            "recommendations": [
                {
                    "rank": r.rank,
                    "drug_combination": r.drug_combination,
                    "drs_score": r.drs_score,
                    "components": {
                        "p_response": r.p_response,
                        "s_norm": r.s_norm,
                        "ts_pct_norm": r.ts_pct_norm,
                        "biomarker_adj": r.biomarker_adj,
                    },
                    "loewe_score": r.loewe_score,
                    "ts_pct": r.ts_pct,
                    "delta_g_kcal": r.delta_g,
                    "direction_flag": r.direction_flag,
                    "rationale": r.rationale,
                }
                for r in recommendations
            ],
        }

        if output_path is None:
            output_path = f"f:/ADDS/outputs/drug_recommendations/{patient.patient_id}_DRS.json"

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False, default=str)

        logger.info(f"Report saved: {output_path}")
        return output_path


# ──────────────────────────────────────────────
# Demo -- Stage IIIB CRC, KRAS G12D, PrPc-high
# ──────────────────────────────────────────────

if __name__ == "__main__":
    patient = PatientProfile(
        patient_id="DEMO-001-CRC",

        # Module 0: Cellpose H&E
        cellpose_metrics={
            "density_per_mm2":    912.0,
            "tumour_area_fraction": 0.68,
            "pleomorphism_index":  0.51,
            "stromal_ratio":       0.19,
            "clustering_index":    0.74,
        },

        # Module 1: CT
        ct_results={
            "max_lesion_diameter_mm": 42.3,
            "tumour_volume_cm3":       8.7,
            "n_candidate_clusters":    1,
            "peak_confidence":         0.94,
            "stage_estimate":         "IIIB",
        },

        # Module 2: PRNP
        prnp_log2rsem = 7.14,        # high PRNP expression
        prnp_tcga_mean = 5.21,
        prnp_tcga_sd   = 0.92,

        # Module 3: PrPc serum
        prpc_concentration = 2.43,   # high PrPc (>threshold 1.72)
        prpc_healthy_mean  = 1.35,
        prpc_healthy_sd    = 0.22,

        # Mutation profile
        kras_mutant   = True,    # KRAS G12D → exclude cetuximab/panitumumab
        braf_mutant   = False,
        tp53_mutant   = True,
        mmr_deficient = False,   # pMMR
        cms_subtype   = "CMS2",

        # Gene expression (Module 4 features)
        gene_expression = {
            "TYMS":  4.2,  "DPYD":  6.8,  "ERCC1": 3.1,  "ERCC2": 3.5,
            "XRCC1": 4.0,  "GSTP1": 3.8,  "TP53":  2.1,  "MDR1":  2.3,
            "BAX":   6.5,  "BCL2":  3.2,  "VEGFA": 5.1,  "EGFR":  4.0,
            "MSH2":  5.8,  "MLH1":  5.5,
        },
    )

    recommender = ADDSDrugRecommender()
    recs, module_out = recommender.run(patient)
    recommender.print_report(recs, module_out, patient)
    recommender.save_report(recs, module_out, patient)
