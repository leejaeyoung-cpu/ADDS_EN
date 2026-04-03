"""
Modal 3: PK/PD Feature Module  (v2 — patient-specific energy landscape)
------------------------------------------------------------------------
Generates 32-dim PK/PD feature vector from Pritamab pharmacokinetic
parameters and drug concentration inputs.

New in v2:
  compute_features() accepts an optional PatientEnergyProfile (from
  imaging_to_energy.py) to replace the static DDG_RLS_KCAL with
  patient-specific per-pathway ΔG‡ values.

Based on: paper3_results.json (ddG=0.50 kcal/mol, alpha=0.35, EC50 model)
          imaging_to_energy.py (Cellpose + CT → ΔG‡ inference)
"""

import numpy as np
import json
from pathlib import Path
import logging
from typing import Optional

# Lazy import — avoid circular dependency at module level
_PatientEnergyProfile = None
def _get_profile_class():
    global _PatientEnergyProfile
    if _PatientEnergyProfile is None:
        try:
            from src.pritamab_ml.imaging_to_energy import PatientEnergyProfile
            _PatientEnergyProfile = PatientEnergyProfile
        except ImportError:
            pass
    return _PatientEnergyProfile

logger = logging.getLogger(__name__)

PAPER3_JSON = r"f:\ADDS\pritamab\data\paper3_results.json"

# Pritamab PK constants (from Nature Comm paper)
PRITAMAB_KD_nM       = 0.84    # Binding affinity (nM)
PRITAMAB_T_HALF_DAYS = 23.0    # Mean half-life (days)
PRITAMAB_BIOAVAIL    = 1.00    # IV = 100%
PRITAMAB_VD_Lkg      = 0.055   # Volume of distribution (L/kg)

# Energy landscape constants
DDG_RLS_KCAL   = 0.50    # ddG for rate-limiting step
ALPHA_COUPLING  = 0.35   # Coupling coefficient
RT_KCAL        = 0.593   # RT at 37°C
RATE_REDUCTION  = 0.556  # 55.6% reduction in escape rate


class PKPDFeatureModule:
    """
    32-dim PK/PD feature vector:
    [0:8]   Pharmacokinetic features (Cmax, AUC, t1/2, clearance...)
    [8:16]  Pharmacodynamic features (EC50, Emax, Hill, Bliss synergy...)
    [16:24] Energy landscape features (ddG, rate reduction, barrier heights...)
    [24:32] Drug combination features (synergy scores, CI, dose-reduction...)
    """
    FEATURE_DIM = 32

    def __init__(self):
        self._load_energy_params()
        logger.info("PKPDFeatureModule initialized (feature_dim=32)")

    def _load_energy_params(self):
        try:
            with open(PAPER3_JSON, "r") as f:
                data = json.load(f)
            self.energy = data.get("energy_landscape", {})
            self.dose   = data.get("dose_response", {})
            logger.info("Loaded paper3_results.json")
        except Exception as e:
            logger.warning(f"paper3 load failed: {e}, using defaults")
            self.energy = {"ddG_rls_kcal_mol": DDG_RLS_KCAL,
                           "alpha_coupling": ALPHA_COUPLING,
                           "rate_reduction_pct": RATE_REDUCTION * 100}
            self.dose   = {
                "5-FU":        {"EC50_alone_nM": 12000, "EC50_pritamab_nM": 9032},
                "Oxaliplatin": {"EC50_alone_nM": 3750,  "EC50_pritamab_nM": 2823},
                "Irinotecan":  {"EC50_alone_nM": 7500,  "EC50_pritamab_nM": 5645},
                "Sotorasib":   {"EC50_alone_nM": 75,    "EC50_pritamab_nM": 56.5},
            }

    def pk_simulation(self, dose_mg: float, weight_kg: float = 70.0,
                      t_hours: float = 168.0) -> dict:
        """One-compartment PK model for Pritamab (IV)."""
        Vd = PRITAMAB_VD_Lkg * weight_kg
        k_el = np.log(2) / (PRITAMAB_T_HALF_DAYS * 24)
        Cmax = dose_mg / Vd                           # mg/L ≈ µg/mL
        Ct   = Cmax * np.exp(-k_el * t_hours)
        AUC  = Cmax / k_el                            # mg·h/L
        CL   = dose_mg / AUC
        return {"Cmax_mgL": Cmax, "Ct_mgL": Ct,
                "AUC_mghL": AUC, "CL_Lh": CL,
                "t_half_h": PRITAMAB_T_HALF_DAYS * 24}

    def pd_bliss(self, E_pritamab: float, E_chemo: float) -> float:
        """Bliss independence synergy score."""
        E_expected = E_pritamab + E_chemo - E_pritamab * E_chemo
        # Observed effect estimate (literature-calibrated +22% boost)
        E_observed = min(E_expected * 1.22, 0.98)
        return E_observed - E_expected

    def compute_features(self, n_samples: int,
                         pritamab_dose_mg: float = 750.0,
                         chemo_drug: str = "FOLFOX",
                         kras_allele: str = "G12D",
                         prpc_high: bool = True,
                         concentration_nM: float = 10.0,
                         rng: np.random.Generator = None,
                         patient_profile=None) -> np.ndarray:
        """
        patient_profile : PatientEnergyProfile (from imaging_to_energy.py) or None.
            None → backward-compatible static DDG_RLS_KCAL behaviour.
        """
        """
        Generate n_samples × 32 PK/PD feature matrices.
        """
        if rng is None:
            rng = np.random.default_rng(2026)

        feats = np.zeros((n_samples, self.FEATURE_DIM), dtype=np.float32)

        # Weight variation across patients
        weights = rng.normal(70.0, 10.0, n_samples).clip(45, 120)
        doses   = np.full(n_samples, pritamab_dose_mg) * rng.uniform(0.9, 1.1, n_samples)

        for i in range(n_samples):
            pk = self.pk_simulation(doses[i], weights[i])

            # [0:8] PK features
            feats[i, 0] = pk["Cmax_mgL"] / 20.0       # normalized
            feats[i, 1] = pk["Ct_mgL"]  / 10.0
            feats[i, 2] = pk["AUC_mghL"] / 2000.0
            feats[i, 3] = pk["CL_Lh"]   / 5.0
            feats[i, 4] = PRITAMAB_KD_nM / 10.0
            feats[i, 5] = np.log1p(concentration_nM / PRITAMAB_KD_nM)
            feats[i, 6] = weights[i] / 70.0
            feats[i, 7] = doses[i] / pritamab_dose_mg

            # [8:16] PD features
            # EC50 from paper3 (drug-specific)
            drug_map = {"FOLFOX": "Oxaliplatin", "FOLFIRI": "Irinotecan",
                        "FOLFOXIRI": "Oxaliplatin", "5-FU": "5-FU",
                        "Sotorasib": "Sotorasib"}
            dr = self.dose.get(drug_map.get(chemo_drug, "Oxaliplatin"),
                               {"EC50_alone_nM": 3750, "EC50_pritamab_nM": 2823})
            ec50_alone = dr["EC50_alone_nM"]
            ec50_combo = dr["EC50_pritamab_nM"]
            dose_red   = (ec50_alone - ec50_combo) / ec50_alone

            feats[i, 8]  = ec50_alone / 10000.0
            feats[i, 9]  = ec50_combo / 10000.0
            feats[i, 10] = dose_red
            # Bliss synergy
            E_p = 0.30 if prpc_high else 0.15   # Pritamab single-agent effect
            E_c = 0.40 + rng.normal(0, 0.05)    # Chemo single-agent effect
            bliss = self.pd_bliss(E_p, E_c)
            feats[i, 11] = bliss
            # Hill coefficient (cooperative binding)
            feats[i, 12] = rng.normal(1.5, 0.2)
            # Emax (combined)
            feats[i, 13] = min(E_p + E_c + bliss, 0.95)
            feats[i, 14] = (prpc_high) * 1.0
            feats[i, 15] = {"G12D": 1.0, "G12V": 0.85, "G12C": 0.78,
                            "G13D": 0.60, "WT": 0.30}.get(kras_allele, 0.5)

            # [16:24] Energy landscape features — patient-specific or static
            if patient_profile is not None:
                # Patient-specific: use per-pathway ΔG‡ from imaging
                ddg_kras = patient_profile.ddg_per_pathway.get("KRAS_ERK",  DDG_RLS_KCAL)
                ddg_pi3k = patient_profile.ddg_per_pathway.get("PI3K_mTOR", DDG_RLS_KCAL)
                ddg_hif  = patient_profile.ddg_per_pathway.get("HIF_VEGF",  DDG_RLS_KCAL)
                ddg_rho  = patient_profile.ddg_per_pathway.get("RhoA_INV",  DDG_RLS_KCAL)
                ddG = patient_profile.ddg_effective
                alpha = self.energy.get("alpha_coupling", ALPHA_COUPLING)
                k_kras = patient_profile.k_per_pathway.get("KRAS_ERK",  0.1)
                k_pi3k = patient_profile.k_per_pathway.get("PI3K_mTOR", 0.1)
                k_hif  = patient_profile.k_per_pathway.get("HIF_VEGF",  0.1)
                k_rho  = patient_profile.k_per_pathway.get("RhoA_INV",  0.1)
            else:
                # Backward compat: static values from paper
                ddG = self.energy.get("ddG_rls_kcal_mol", DDG_RLS_KCAL)
                alpha = self.energy.get("alpha_coupling", ALPHA_COUPLING)
                ddg_kras = ddg_pi3k = ddg_hif = ddg_rho = ddG
                k_kras = k_pi3k = k_hif = k_rho = np.exp(-ddG / RT_KCAL)

            rate_red = 1 - np.exp(-ddG / RT_KCAL)
            feats[i, 16] = ddg_kras / 2.0             # KRAS pathway ΔG‡ (normalized)
            feats[i, 17] = ddg_pi3k / 2.0             # PI3K pathway ΔG‡
            feats[i, 18] = ddg_hif  / 2.0             # HIF pathway ΔG‡
            feats[i, 19] = ddg_rho  / 2.0             # RhoA pathway ΔG‡
            feats[i, 20] = k_kras                      # Eyring k — KRAS
            feats[i, 21] = k_pi3k                      # Eyring k — PI3K
            feats[i, 22] = k_hif                       # Eyring k — HIF
            feats[i, 23] = k_rho                       # Eyring k — RhoA

            # [24:32] Drug combination synergy features
            feats[i, 24] = bliss * 25.0               # Bliss score (0-25 scale)
            feats[i, 25] = ec50_combo / ec50_alone     # CI (Loewe)
            feats[i, 26] = max(E_p, E_c)              # HSA
            feats[i, 27] = (bliss + (1 - feats[i, 25])) / 2  # ZIP proxy
            feats[i, 28] = 1 if bliss > 0.05 else 0   # Synergy flag
            feats[i, 29] = dose_red * 100              # % dose reduction
            feats[i, 30] = feats[i, 11] * feats[i, 15]  # PrPc-weighted synergy
            feats[i, 31] = rng.normal(0, 0.01)

        return feats


if __name__ == "__main__":
    mod = PKPDFeatureModule()
    f = mod.compute_features(5, prpc_high=True, kras_allele="G12D")
    print(f"Shape: {f.shape}, Bliss score: {f[:,24].mean():.3f}")
