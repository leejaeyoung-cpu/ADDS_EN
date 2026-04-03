"""
ChEMBL IC50 데이터 다운로드 + PINN v5 calibration

합성 시나리오를 실측 ChEMBL IC50 데이터로 교체합니다.
"""

import json
import logging
import time
import urllib.request
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DATA_DIR = Path("F:/ADDS/data/ml_training")
CHEMBL_DIR = DATA_DIR / "chembl"
CHEMBL_DIR.mkdir(parents=True, exist_ok=True)

# CRC-relevant targets with ChEMBL IDs
TARGETS = {
    "EGFR": "CHEMBL203",
    "BRAF": "CHEMBL5145",
    "MTOR": "CHEMBL2842",
    "PIK3CA": "CHEMBL3267",
    "VEGFR2": "CHEMBL279",
    "CDK4": "CHEMBL3116",
    "PARP1": "CHEMBL3105",
    "TOP1": "CHEMBL1781",
    "TYMS": "CHEMBL1952",
    "KRAS": "CHEMBL2093868",
}


def download_chembl_ic50(target_name: str, chembl_id: str, max_records: int = 500) -> pd.DataFrame:
    """Download IC50 data from ChEMBL API for a specific target."""
    cache_path = CHEMBL_DIR / f"ic50_{target_name}.json"

    if cache_path.exists():
        with open(cache_path) as f:
            records = json.load(f)
        logger.info(f"  {target_name}: loaded {len(records)} cached records")
        return pd.DataFrame(records)

    all_records = []
    offset = 0
    limit = 100  # ChEMBL max per page

    while len(all_records) < max_records:
        url = (
            f"https://www.ebi.ac.uk/chembl/api/data/activity.json?"
            f"target_chembl_id={chembl_id}&standard_type=IC50"
            f"&standard_relation=%3D&limit={limit}&offset={offset}"
        )
        try:
            req = urllib.request.Request(url, headers={
                "Accept": "application/json",
                "User-Agent": "ADDS-Research/1.0",
            })
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read())
        except Exception as e:
            logger.warning(f"  API error at offset {offset}: {e}")
            break

        activities = data.get("activities", [])
        if not activities:
            break

        for act in activities:
            try:
                rec = {
                    "target": target_name,
                    "molecule_chembl_id": act.get("molecule_chembl_id", ""),
                    "molecule_name": act.get("canonical_smiles", ""),
                    "ic50_nm": float(act.get("standard_value", 0)),
                    "standard_units": act.get("standard_units", ""),
                    "assay_type": act.get("assay_type", ""),
                    "pchembl_value": float(act["pchembl_value"]) if act.get("pchembl_value") else None,
                }
                if rec["ic50_nm"] > 0 and rec["standard_units"] == "nM":
                    all_records.append(rec)
            except (ValueError, TypeError):
                pass

        offset += limit
        if offset >= data.get("page_meta", {}).get("total_count", 0):
            break
        time.sleep(0.2)  # Rate limiting

    with open(cache_path, "w") as f:
        json.dump(all_records, f, indent=2)

    logger.info(f"  {target_name}: downloaded {len(all_records)} IC50 records")
    return pd.DataFrame(all_records)


def download_all_targets():
    """Download IC50 data for all CRC targets."""
    logger.info("Downloading ChEMBL IC50 data for CRC targets...")
    all_dfs = []

    for target_name, chembl_id in TARGETS.items():
        logger.info(f"  Fetching {target_name} ({chembl_id})...")
        df = download_chembl_ic50(target_name, chembl_id, max_records=500)
        all_dfs.append(df)

    combined = pd.concat(all_dfs, ignore_index=True)
    combined.to_csv(CHEMBL_DIR / "crc_ic50_combined.csv", index=False)
    logger.info(f"Combined: {len(combined)} records across {len(TARGETS)} targets")

    # Summary statistics
    for target in TARGETS:
        sub = combined[combined["target"] == target]
        if len(sub) > 0:
            median_ic50 = sub["ic50_nm"].median()
            q25, q75 = sub["ic50_nm"].quantile([0.25, 0.75])
            logger.info(f"  {target}: n={len(sub)}, median IC50={median_ic50:.1f} nM "
                        f"(IQR: {q25:.1f}-{q75:.1f})")

    return combined


def build_real_pinn_scenarios(ic50_df: pd.DataFrame):
    """Build PINN calibration scenarios from real ChEMBL IC50 data.

    Each scenario = (target, representative IC50, PK features)
    Groups by target and IC50 range (bins).
    """
    scenarios = []

    for target_name in TARGETS:
        sub = ic50_df[ic50_df["target"] == target_name].copy()
        if len(sub) < 10:
            continue

        # Bin IC50 into log-scale groups
        sub["log_ic50"] = np.log10(sub["ic50_nm"].clip(0.1))
        bins = np.linspace(sub["log_ic50"].min(), sub["log_ic50"].max(), 6)

        for i in range(len(bins) - 1):
            mask = (sub["log_ic50"] >= bins[i]) & (sub["log_ic50"] < bins[i + 1])
            group = sub[mask]
            if len(group) < 3:
                continue

            median_ic50 = group["ic50_nm"].median()
            mean_pchembl = group["pchembl_value"].dropna().mean() if group["pchembl_value"].notna().any() else 5.0

            # Build PK features based on real data
            # Feature vector: [binding_energy, Cmax, t_half, inhibition_efficiency,
            #                  clearance, volume_dist, bioavailability]
            scenario = {
                "target": target_name,
                "ic50_nm": float(median_ic50),
                "pchembl": float(mean_pchembl) if not np.isnan(mean_pchembl) else 5.0,
                "n_compounds": len(group),
                "ic50_range": [float(group["ic50_nm"].min()), float(group["ic50_nm"].max())],
                "pk": [
                    -mean_pchembl * 1.36,  # Binding energy (kcal/mol) from pChEMBL
                    median_ic50 * 10,       # Cmax (nM) ~ 10x IC50
                    12.0,                   # t_half (h) typical
                    max(0.1, 1 - median_ic50 / 10000),  # Inhibition efficiency
                    0.5,                    # Clearance (normalized)
                    0.5,                    # Volume of distribution (normalized)
                    0.6,                    # Bioavailability (typical oral)
                ],
            }
            scenarios.append(scenario)

    logger.info(f"Built {len(scenarios)} real PINN scenarios from ChEMBL data")
    return scenarios


def main():
    logger.info("=" * 60)
    logger.info("PINN v5: ChEMBL Real IC50 Calibration")
    logger.info("=" * 60)

    # Download
    ic50_df = download_all_targets()

    # Build scenarios
    scenarios = build_real_pinn_scenarios(ic50_df)

    # Save scenarios
    with open(CHEMBL_DIR / "pinn_real_scenarios.json", "w") as f:
        json.dump(scenarios, f, indent=2, default=float)

    logger.info(f"\nSaved {len(scenarios)} scenarios to pinn_real_scenarios.json")
    logger.info("Ready for PINN v5 training with real IC50 data")

    # Summary
    targets_used = set(s["target"] for s in scenarios)
    logger.info(f"\nTargets used: {sorted(targets_used)}")
    for t in sorted(targets_used):
        t_scenarios = [s for s in scenarios if s["target"] == t]
        ic50s = [s["ic50_nm"] for s in t_scenarios]
        logger.info(f"  {t}: {len(t_scenarios)} scenarios, IC50 range: {min(ic50s):.1f}-{max(ic50s):.1f} nM")


if __name__ == "__main__":
    main()
