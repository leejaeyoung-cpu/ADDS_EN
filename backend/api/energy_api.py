"""
Energy Framework API
Physics-informed drug energy prediction and DrugComb validation endpoints.
"""

import os as _os
from pathlib import Path as _Path
BASE_DIR = _Path(_os.environ.get("ADDS_BASE_DIR", str(_Path(__file__).resolve().parent.parent.parent)))

from fastapi import APIRouter, HTTPException
from typing import Optional
import json
import logging
from pathlib import Path

import numpy as np
import torch

from backend.schemas.energy_schemas import (
    EnergyPredictRequest, EnergyPredictResponse,
    DrugCombValidationRequest, DrugCombValidationResponse,
    PathwayGraphResponse, PathwayEdge,
    CalibrationStatusResponse,
)

logger = logging.getLogger(__name__)
router = APIRouter()

# ─── Constants ────────────────────────────────────────────────────────────────
R_cal = 1.987e-3
T_body = 310.15
RT = R_cal * T_body

MODEL_DIR = Path(BASE_DIR / "models/energy")
DATA_DIR = Path(BASE_DIR / "data")

# Pathway node names (must match v3 model)
PATHWAY_NODES = [
    "PrPC", "cMET", "EGFR", "LamR", "Wnt", "Autophagy",
    "RAS", "PI3K", "JAK_STAT", "FAK", "Notch", "Hippo", "NF_kB",
    "proliferation", "survival", "migration", "metabolism",
    "immune_evasion", "stemness", "inflammation",
]
N_NODES = len(PATHWAY_NODES)
NODE_IDX = {n: i for i, n in enumerate(PATHWAY_NODES)}


# ─── Model loader (lazy) ─────────────────────────────────────────────────────
_model_cache = {}


def _load_model():
    """Load the best available energy model."""
    if "model" in _model_cache:
        return _model_cache["model"], _model_cache["meta"]

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Try calibrated first, then base v3
    for fname, label in [
        ("energy_predictor_v3_calibrated.pt", "v3_calibrated"),
        ("energy_predictor_v3.pt", "v3_base"),
    ]:
        path = MODEL_DIR / fname
        if path.exists():
            try:
                # Import the model class
                import sys
                sys.path.insert(0, str(Path(BASE_DIR / "scripts")))
                from track2_energy_pinn_v3 import EnergyPredictorV3

                ckpt = torch.load(path, map_location=device, weights_only=False)
                model = EnergyPredictorV3(n_pk=7).to(device)
                model.load_state_dict(ckpt['model_state'])
                model.eval()

                meta = {
                    "version": label,
                    "pk_mean": ckpt.get("pk_mean"),
                    "pk_std": ckpt.get("pk_std"),
                    "pearson_r": ckpt.get("pearson_r"),
                    "device": device,
                }
                _model_cache["model"] = model
                _model_cache["meta"] = meta
                logger.info(f"Loaded energy model: {label}")
                return model, meta
            except Exception as e:
                logger.warning(f"Failed to load {fname}: {e}")

    return None, None


def kd_to_dg(kd_nm):
    return RT * np.log(kd_nm * 1e-9)


# ─── Endpoints ────────────────────────────────────────────────────────────────

@router.post("/predict", response_model=EnergyPredictResponse)
async def predict_energy(request: EnergyPredictRequest):
    """
    Predict drug efficacy using physics-informed energy framework.

    Converts binding affinity (KD/IC50) → Gibbs free energy (ΔG),
    propagates through learned GNN pathway graph,
    outputs tumor suppression %, IC50, and synergy CI.
    """
    model, meta = _load_model()

    # Calculate binding energy
    if request.kd_nm:
        dg = kd_to_dg(request.kd_nm)
    elif request.ic50_nm:
        dg = kd_to_dg(request.ic50_nm)
    else:
        dg = kd_to_dg(100)  # Default moderate affinity

    # Build PK features
    combo_dg = kd_to_dg(request.combination_ic50_nm) if request.combination_ic50_nm else 0
    combo_dose = 1.0 if request.combination_drug else 0.0
    pk = [dg, dg * 0.8, RT * 20, 0.96, 0.0, combo_dg, combo_dose]

    # Build pathway modulation
    mod = np.ones(N_NODES, dtype=np.float32)
    if request.mutations:
        from scripts.track2_energy_pinn_v3 import MUTATION_DDG
        for node, mut in request.mutations.items():
            if node in NODE_IDX:
                ddg = MUTATION_DDG.get(mut, 0)
                mod[NODE_IDX[node]] *= (1 + ddg / 5.0)

    if model is None:
        # Fallback: analytical prediction
        ts = min(95, max(5, 50 + abs(dg) * 3))
        ic50 = max(1, abs(1 / (abs(dg) + 0.01)) * 100)
        return EnergyPredictResponse(
            drug_name=request.drug_name,
            binding_energy_kcal=round(dg, 2),
            predicted_tumor_suppression_pct=round(ts, 1),
            predicted_ic50_nm=round(ic50, 1),
            model_version="analytical_fallback",
        )

    # GNN prediction
    device = meta["device"]
    pk_t = torch.FloatTensor([pk]).to(device)
    mod_t = torch.FloatTensor([mod]).to(device)

    # Normalize with training stats
    if meta.get("pk_mean") is not None:
        pk_t = (pk_t - meta["pk_mean"].to(device)) / (meta["pk_std"].to(device) + 1e-8)

    with torch.no_grad():
        pred_ts, pred_ic50, pred_ci, node_e, edge_w = model(pk_t, mod_t)

    # Pathway energies
    pw_energies = {PATHWAY_NODES[i]: round(float(node_e[0, i]), 3) for i in range(N_NODES)}

    ci_val = float(pred_ci[0])
    interpretation = ("strong synergy" if ci_val < 0.5 else
                     "synergy" if ci_val < 0.9 else
                     "additive" if ci_val < 1.1 else "antagonistic")

    return EnergyPredictResponse(
        drug_name=request.drug_name,
        binding_energy_kcal=round(dg, 2),
        predicted_tumor_suppression_pct=round(float(pred_ts[0]), 1),
        predicted_ic50_nm=round(float(pred_ic50[0]), 1),
        predicted_synergy_ci=round(ci_val, 3) if request.combination_drug else None,
        synergy_interpretation=interpretation if request.combination_drug else None,
        pathway_energies=pw_energies,
        model_version=meta["version"],
    )


@router.post("/validate", response_model=DrugCombValidationResponse)
async def validate_against_drugcomb(request: DrugCombValidationRequest):
    """
    Cross-validate energy model against real DrugComb synergy data.
    Uses 592 real Loewe scores from 20 CRC drug pairs.
    """
    results_file = MODEL_DIR / "drugcomb_validation.json"

    if not results_file.exists():
        raise HTTPException(
            status_code=404,
            detail="DrugComb validation not yet performed. "
                   "Run: python scripts/validate_with_drugcomb.py"
        )

    with open(results_file) as f:
        results = json.load(f)

    pairs = results.get("per_pair", [])
    if request.cell_line:
        # Filter not yet implemented (all data is HCT116 currently)
        pass
    if request.drug_pairs:
        filter_set = {(p["drug_a"], p["drug_b"]) for p in request.drug_pairs}
        pairs = [p for p in pairs if (p["drug_a"], p["drug_b"]) in filter_set]

    return DrugCombValidationResponse(
        n_pairs_validated=results["n_pairs"],
        n_datapoints=results["n_datapoints"],
        pearson_r_loewe=results["pearson_r"],
        rmse_loewe=results["rmse_loewe"],
        classification_accuracy=results["synergy_accuracy"],
        per_pair_results=pairs,
        model_version="v3_gnn_calibrated",
    )


@router.get("/pathway-graph", response_model=PathwayGraphResponse)
async def get_pathway_graph():
    """
    Return the learned GNN pathway graph with edge weights.
    Shows which biological pathways the model considers most important.
    """
    model, meta = _load_model()

    edges = []
    calibration_source = "simulation"

    if model is not None:
        with torch.no_grad():
            ew = model.gnn.get_edge_weights().cpu().numpy()

        for i in range(N_NODES):
            for j in range(N_NODES):
                if ew[i, j] > 0.05:
                    edges.append(PathwayEdge(
                        source=PATHWAY_NODES[i],
                        target=PATHWAY_NODES[j],
                        weight=round(float(ew[i, j]), 3),
                        source_type="learned",
                    ))

        if meta.get("version", "").endswith("calibrated"):
            calibration_source = "drugcomb"

    # Add BioGRID edges if available
    ppi_file = Path(BASE_DIR / "data/real_ppi/ppi_gnn_edges.json")
    if ppi_file.exists():
        with open(ppi_file) as f:
            ppi = json.load(f)
        for key, val in ppi.items():
            src, tgt = key.split("|")
            if src in PATHWAY_NODES and tgt in PATHWAY_NODES:
                edges.append(PathwayEdge(
                    source=src, target=tgt,
                    weight=val["weight"],
                    source_type="biogrid",
                ))
        calibration_source = "biogrid+drugcomb" if calibration_source == "drugcomb" else "biogrid"

    edges.sort(key=lambda x: x.weight, reverse=True)

    return PathwayGraphResponse(
        n_nodes=N_NODES,
        n_active_edges=len(edges),
        nodes=PATHWAY_NODES,
        edges=edges,
        calibration_source=calibration_source,
    )


@router.get("/calibration-status", response_model=CalibrationStatusResponse)
async def get_calibration_status():
    """Check the calibration status of the energy model."""
    # BioGRID
    ppi_summary = Path(BASE_DIR / "data/real_ppi/ppi_summary.json")
    biogrid_loaded = ppi_summary.exists()
    biogrid_count = 0
    if biogrid_loaded:
        with open(ppi_summary) as f:
            s = json.load(f)
        biogrid_count = s.get("total_raw_interactions", 0)

    # DrugComb
    val_file = MODEL_DIR / "drugcomb_validation.json"
    drugcomb_rows = 0
    val_r = None
    if val_file.exists():
        with open(val_file) as f:
            v = json.load(f)
        drugcomb_rows = v.get("n_datapoints", 0)
        val_r = v.get("pearson_r")

    # Model
    _, meta = _load_model()
    train_r_ts = meta.get("pearson_r") if meta else None

    return CalibrationStatusResponse(
        biogrid_ppi_loaded=biogrid_loaded,
        biogrid_ppi_count=biogrid_count,
        drugcomb_rows_used=drugcomb_rows,
        train_r_tumor_supp=train_r_ts,
        validation_r_loewe=val_r,
    )


# ─── DeepSynergy Endpoints ───────────────────────────────────────────────────

_deep_synergy_predictor = None


def _get_deep_synergy():
    global _deep_synergy_predictor
    if _deep_synergy_predictor is None:
        try:
            import sys
            sys.path.insert(0, str(Path(BASE_DIR / "scripts")))
            from deep_synergy_inference import DeepSynergyPredictor
            _deep_synergy_predictor = DeepSynergyPredictor()
        except Exception as e:
            logger.warning(f"DeepSynergy not available: {e}")
            return None
    return _deep_synergy_predictor


@router.post("/deep-synergy")
async def predict_deep_synergy(
    drug_a: str, drug_b: str, cell_line: str = "HCT116"
):
    """
    Predict drug combination synergy using DeepSynergy MLP model.
    Uses Morgan fingerprints (2048-bit) + molecular descriptors + cell mutation profile.
    Trained on 18K+ combinations, fine-tuned on 592 CRC real data (r=0.60).
    """
    predictor = _get_deep_synergy()
    if predictor is None:
        raise HTTPException(status_code=503, detail="DeepSynergy model not loaded")

    try:
        result = predictor.predict_synergy(drug_a, drug_b, cell_line)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/deep-synergy/batch")
async def predict_deep_synergy_batch(
    pairs: list, cell_line: str = "HCT116"
):
    """Batch predict synergy for multiple drug pairs."""
    predictor = _get_deep_synergy()
    if predictor is None:
        raise HTTPException(status_code=503, detail="DeepSynergy model not loaded")

    results = []
    for pair in pairs[:50]:  # Limit to 50
        a, b = pair if isinstance(pair, (list, tuple)) else (pair["drug_a"], pair["drug_b"])
        results.append(predictor.predict_synergy(a, b, cell_line))
    return results


@router.get("/deep-synergy/drugs")
async def get_deep_synergy_drugs():
    """Get available drugs grouped by mechanism class."""


    predictor = _get_deep_synergy()
    if predictor is None:
        raise HTTPException(status_code=503, detail="DeepSynergy model not loaded")

    return {
        "drugs_by_class": predictor.get_available_drugs(),
        "cell_lines": predictor.get_available_cell_lines(),
        "model_metrics": {
            "test_pearson_r": predictor.test_results.get("pearson_r"),
            "crc_pearson_r": predictor.crc_results.get("pearson_r"),
            "training_data": predictor.train_size,
            "timestamp": predictor.timestamp,
        },
    }

