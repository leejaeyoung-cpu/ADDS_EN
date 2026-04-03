"""
Synthetic Data Generator
------------------------
Combines all 4 modality feature extractors → Fusion MLP → synthetic patient dataset.
Generates n_patients × [pfs, os, synergy_score, response_label, KRAS, PrPc, chemo_drug, ...]
Stratified by: KRAS allele, PrPc status, chemo regimen, Pritamab arm.
"""

import numpy as np
import pandas as pd
import sys, os
sys.path.insert(0, str(__file__).split("pritamab_ml")[0].rstrip("/\\"))

from pritamab_ml.cellpose_feature_extractor import CellposeFeatureExtractor
from pritamab_ml.rnaseq_encoder              import RNAseqEncoder
from pritamab_ml.pkpd_feature_module         import PKPDFeatureModule
from pritamab_ml.multimodal_fusion           import CTTumorFeatureExtractor, PritamamFusionModel

import logging
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ── Population parameters ──────────────────────────────────────────────────
KRAS_ALLELE_DIST = {
    "G12D": 0.26, "G12V": 0.19, "G12C": 0.12, "G13D": 0.09, "WT": 0.34
}
PRPC_HIGH_PROB   = {"G12D": 0.88, "G12V": 0.85, "G12C": 0.83,
                    "G13D": 0.80, "WT":   0.60}
CHEMO_DRUGS      = ["FOLFOX", "FOLFIRI", "FOLFOXIRI"]
CHEMO_PROBS      = [0.40,     0.45,      0.15]

# Known ground-truth HR targets per KRAS × arm (from energy model)
HR_PFS_TARGETS = {
    ("G12D", True):  0.52, ("G12D", False): 1.00,
    ("G12V", True):  0.55, ("G12V", False): 1.00,
    ("G12C", True):  0.53, ("G12C", False): 1.00,
    ("G13D", True):  0.58, ("G13D", False): 1.00,
    ("WT",   True):  0.67, ("WT",   False): 1.00,
}
CONTROL_MPFS = {"G12D": 5.5, "G12V": 5.8, "G12C": 5.6, "G13D": 5.9, "WT":  6.2}
CONTROL_MOS  = {"G12D": 11.5,"G12V": 12.0,"G12C": 11.8,"G13D": 12.3,"WT": 13.0}
PRPC_PFS_BOOST = 1.20   # PrPc-high adds 20% PFS gain on top of HR

def _generate_pfs_os(allele: str, pritamab: bool, prpc_high: bool,
                     chemo: str, rng: np.random.Generator) -> tuple:
    """Generate realistic PFS and OS in months."""
    hr_pfs = HR_PFS_TARGETS.get((allele, pritamab), 1.0)
    if pritamab and prpc_high:
        hr_pfs *= (1 / PRPC_PFS_BOOST)

    base_pfs  = CONTROL_MPFS.get(allele, 5.5)
    base_os   = CONTROL_MOS.get(allele, 12.0)
    chemo_fac = {"FOLFOX": 1.0, "FOLFIRI": 1.02, "FOLFOXIRI": 1.12}.get(chemo, 1.0)

    lam_pfs = np.log(2) / (base_pfs / hr_pfs * chemo_fac)
    pfs     = rng.exponential(1 / lam_pfs)

    # OS correlated with PFS but with additional tail
    hr_os   = hr_pfs * 1.06
    lam_os  = np.log(2) / (base_os / hr_os * chemo_fac)
    os_min  = pfs * rng.uniform(1.0, 1.5)
    os_tail = rng.exponential(1 / lam_os)
    os_     = max(os_min, os_tail)

    # Random censoring
    pfs_event = rng.random() > 0.25   # 75% events
    os_event  = rng.random() > 0.30   # 70% events

    # Clip to realistic range
    pfs = float(np.clip(pfs, 0.5, 60.0))
    os_ = float(np.clip(os_, pfs, 120.0))
    return pfs, pfs_event, os_, os_event


def generate_synthetic_dataset(n_patients: int = 1000,
                                n_pritamab: int = None,
                                output_csv: str = None,
                                seed: int = 2026) -> pd.DataFrame:
    """
    Main function: generate synthetic patient dataset.
    
    Args:
        n_patients: total number of virtual patients
        n_pritamab: number in Pritamab arm (default: 2/3 of total)
        output_csv:  path to save CSV
        seed:        random seed for reproducibility
    
    Returns:
        pd.DataFrame with columns:
        [patient_id, arm, kras_allele, prpc_high, chemo_drug,
         pfs, pfs_event, os, os_event, synergy_score, response_label,
         best_pct_change, cell_feat_*, rnaseq_feat_*, pkpd_feat_*, ct_feat_*]
    """
    rng = np.random.default_rng(seed)

    if n_pritamab is None:
        n_pritamab = int(n_patients * (2/3))
    n_control = n_patients - n_pritamab

    # ── Initialise feature extractors
    logger.info("Initialising feature extractors...")
    cell_ext  = CellposeFeatureExtractor(seed=seed)
    rna_enc   = RNAseqEncoder(seed=seed)
    pkpd_mod  = PKPDFeatureModule()
    ct_ext    = CTTumorFeatureExtractor(seed=seed)
    fusion    = PritamamFusionModel(seed=seed)

    records = []

    for arm, n_arm in [("Pritamab", n_pritamab), ("Control", n_control)]:
        pritamab = (arm == "Pritamab")
        logger.info(f"Generating {n_arm} patients for {arm} arm...")

        # Sample KRAS alleles
        alleles = rng.choice(list(KRAS_ALLELE_DIST.keys()), size=n_arm,
                             p=list(KRAS_ALLELE_DIST.values()))
        # Sample chemo drugs
        chemos  = rng.choice(CHEMO_DRUGS, size=n_arm, p=CHEMO_PROBS)

        for i in range(n_arm):
            allele = alleles[i]
            chemo  = chemos[i]
            prpc_high = rng.random() < PRPC_HIGH_PROB.get(allele, 0.75)
            conc_nM   = rng.uniform(5.0, 20.0)

            # ── Extract features per modality
            cell_f = cell_ext.simulate_features(
                1, pritamab_treated=pritamab, prpc_high=prpc_high,
                kras_allele=allele, chemo_drug=chemo,
                concentration_nM=conc_nM)[0]

            rna_f = rna_enc.encode_samples(
                1, pritamab_treated=pritamab, prpc_high=prpc_high,
                kras_allele=allele, concentration_nM=conc_nM)[0]

            pkpd_f = pkpd_mod.compute_features(
                1, pritamab_dose_mg=750.0, chemo_drug=chemo,
                kras_allele=allele, prpc_high=prpc_high,
                concentration_nM=conc_nM, rng=rng)[0]

            ct_f = ct_ext.simulate_features(
                1, pritamab_treated=pritamab, prpc_high=prpc_high,
                kras_allele=allele)[0]

            # ── Fusion model inference
            x = np.concatenate([cell_f, rna_f, pkpd_f, ct_f]).reshape(1, -1)
            out = fusion.forward(x)

            # ── Generate survival outcomes
            pfs, pfs_ev, os_, os_ev = _generate_pfs_os(
                allele, pritamab, prpc_high, chemo, rng)

            # Blend DL predictions with simulation (70% DL, 30% simulation)
            dl_pfs = float(out["pfs"][0])
            dl_os  = float(out["os"][0])
            final_pfs = 0.70 * dl_pfs + 0.30 * pfs
            final_os  = 0.70 * dl_os  + 0.30 * os_
            final_os  = max(final_os, final_pfs)

            # Synergy score (Bliss 0-25 scale)
            syn_prob  = float(out["synergy_prob"][0])
            syn_score = syn_prob * 25.0 if pritamab else rng.uniform(0, 8.0)

            # Response label (waterfall best % change)
            # Positive = tumour growth, Negative = shrinkage
            if pritamab and prpc_high:
                best_pct = rng.normal(-35, 18)
            elif pritamab:
                best_pct = rng.normal(-22, 22)
            else:
                best_pct = rng.normal(-12, 26)

            response_label = "CR" if best_pct < -50 else (
                             "PR" if best_pct < -30 else (
                             "SD" if best_pct <  20 else "PD"))

            record = {
                "patient_id":     f"{arm[0]}{i+1:04d}",
                "arm":            arm,
                "kras_allele":    allele,
                "prpc_high":      int(prpc_high),
                "chemo_drug":     chemo,
                "dl_pfs_months":  round(dl_pfs,  2),
                "dl_os_months":   round(dl_os,   2),
                "pfs_months":     round(final_pfs, 2),
                "pfs_event":      int(pfs_ev),
                "os_months":      round(final_os,  2),
                "os_event":       int(os_ev),
                "synergy_score":  round(syn_score, 3),
                "synergy_prob":   round(syn_prob,  4),
                "best_pct_change":round(best_pct,  1),
                "response_label": response_label,
                "orr":           int(response_label in ("CR", "PR")),
                "dcr":           int(response_label in ("CR", "PR", "SD")),
            }
            records.append(record)

    df = pd.DataFrame(records).reset_index(drop=True)

    # Validation: KS-test vs GSE72970
    try:
        from scipy.stats import ks_2samp
        import pandas as _pd
        gse = _pd.read_csv(r"f:\ADDS\data\ml_training\chemo_response\GSE72970_clinical.csv")
        gse["pfs"] = _pd.to_numeric(gse["pfs"], errors="coerce").dropna()
        stat, pval = ks_2samp(df["pfs_months"].values, gse["pfs"].dropna().values)
        logger.info(f"KS-test vs GSE72970 PFS: stat={stat:.3f}, p={pval:.4f} "
                    f"({'PASS' if pval > 0.05 else 'WARNING: distributions differ'})")
    except Exception as e:
        logger.warning(f"KS-test skipped: {e}")

    if output_csv:
        df.to_csv(output_csv, index=False)
        logger.info(f"Saved synthetic dataset: {output_csv}  ({len(df)} patients)")

    # Summary stats
    for arm in ["Pritamab", "Control"]:
        sub = df[df["arm"] == arm]
        logger.info(f"{arm}: n={len(sub)}, "
                    f"mPFS={sub['pfs_months'].median():.1f}m, "
                    f"mOS={sub['os_months'].median():.1f}m, "
                    f"ORR={sub['orr'].mean()*100:.0f}%, "
                    f"Syn={sub['synergy_score'].mean():.1f}")
    return df


if __name__ == "__main__":
    df = generate_synthetic_dataset(
        n_patients=500,
        output_csv=r"f:\ADDS\data\pritamab_synthetic_cohort.csv",
        seed=2026
    )
    print(df.head(10).to_string())
    print(f"\nTotal: {len(df)}, Columns: {list(df.columns)}")
