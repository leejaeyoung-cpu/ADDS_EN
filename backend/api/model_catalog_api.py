"""
ML Model Catalog API
Provides inventory of all trained ML models in the ADDS system.
"""
import os as _os
from pathlib import Path as _Path
BASE_DIR = _Path(_os.environ.get("ADDS_BASE_DIR", str(_Path(__file__).resolve().parent.parent.parent)))

from fastapi import APIRouter
from pathlib import Path
from typing import Dict, List, Any
import json
import logging

logger = logging.getLogger(__name__)

router = APIRouter()

MODELS_ROOT = BASE_DIR / "models"


def _get_model_catalog() -> List[Dict[str, Any]]:
    """Scan models/ directory and build complete catalog."""
    catalog = []

    # nnU-Net Tumor Segmentation
    for variant in ["sota_balanced", "sota_100epoch", "sota_optimized_v1", "sota_combo_v2"]:
        variant_path = MODELS_ROOT / variant
        if variant_path.exists():
            folds = list(variant_path.iterdir()) if variant_path.is_dir() else []
            fold_dirs = [f for f in folds if f.is_dir() and f.name.startswith("fold")]
            best_model = any((variant_path / f.name / "best_model.pth").exists() for f in fold_dirs)
            catalog.append({
                "id": f"nnunet_{variant}",
                "name": f"nnU-Net ({variant})",
                "category": "segmentation",
                "task": "Colon tumor segmentation",
                "architecture": "nnU-Net 3D Full-Resolution",
                "data": "Dataset011_ColonMasked",
                "status": "trained" if best_model else "checkpoints_only",
                "folds": len(fold_dirs),
                "path": str(variant_path),
            })

    # DeepSynergy DNN
    for ds_file in ["deep_synergy_v1.pt", "deep_synergy_v2.pt"]:
        ds_path = MODELS_ROOT / "synergy" / ds_file
        if ds_path.exists():
            version = ds_file.replace(".pt", "")
            meta_path = MODELS_ROOT / "synergy" / "deep_synergy_metadata.json"
            meta = {}
            if meta_path.exists():
                with open(meta_path) as f:
                    meta = json.load(f)

            catalog.append({
                "id": f"deep_synergy_{version}",
                "name": f"DeepSynergy {version.upper()}",
                "category": "synergy",
                "task": "Drug combination synergy prediction",
                "architecture": meta.get("architecture", "MLP DNN"),
                "data": "O'Neil et al. 2016 (23,052 combos)",
                "performance": meta.get("cv_results", {}),
                "size_mb": round(ds_path.stat().st_size / 1e6, 1),
                "status": "trained",
                "path": str(ds_path),
            })

    # XGBoost Synergy Models
    for variant in ["morgan", "combined", "fixAB", "realexpr"]:
        pkl_path = MODELS_ROOT / "synergy" / f"xgboost_synergy_{variant}.pkl"
        if pkl_path.exists():
            catalog.append({
                "id": f"xgb_synergy_{variant}",
                "name": f"XGBoost Synergy ({variant})",
                "category": "synergy",
                "task": "Drug synergy regression",
                "architecture": "XGBoost",
                "data": "O'Neil 2016 + Morgan FP",
                "size_mb": round(pkl_path.stat().st_size / 1e6, 1),
                "status": "trained",
                "path": str(pkl_path),
            })

    # Treatment Response
    tr_path = MODELS_ROOT / "treatment_response" / "xgb_treatment_response.json"
    if tr_path.exists():
        meta_path = MODELS_ROOT / "treatment_response" / "model_metadata.json"
        meta = {}
        if meta_path.exists():
            with open(meta_path) as f:
                meta = json.load(f)
        catalog.append({
            "id": "xgb_treatment_response",
            "name": "XGBoost Treatment Response",
            "category": "treatment_response",
            "task": "CRC 3-year relapse prediction",
            "architecture": "XGBoost Classifier",
            "data": "GSE39582 (211 CRC patients)",
            "performance": meta.get("cv_results", {}),
            "n_features": meta.get("n_features", 145),
            "status": "trained",
            "path": str(tr_path),
        })

    # Energy PINN
    for ep_file in ["energy_predictor_v3_calibrated_v2.pt", "energy_predictor_v3_calibrated.pt",
                     "energy_predictor_v3.pt"]:
        ep_path = MODELS_ROOT / "energy" / ep_file
        if ep_path.exists():
            results_path = MODELS_ROOT / "energy" / "energy_predictor_v3_results.json"
            perf = {}
            if results_path.exists():
                with open(results_path) as f:
                    data = json.load(f)
                    perf = {
                        "train_pearson_r": data.get("train_pearson_r", {}),
                        "loo_mae": data.get("loo_mae", {}),
                        "n_scenarios": data.get("n_scenarios", 0),
                    }
            catalog.append({
                "id": "energy_pinn_v3",
                "name": "Energy Framework PINN v3",
                "category": "energy",
                "task": "Physics-informed drug efficacy prediction",
                "architecture": "GNN + Physics-Informed NN",
                "data": "67 drug-pathway scenarios",
                "performance": perf,
                "status": "trained",
                "path": str(ep_path),
            })
            break  # Only add best version

    # SAM (Segment Anything)
    sam_path = MODELS_ROOT / "sam_vit_h_4b8939.pth"
    if sam_path.exists():
        catalog.append({
            "id": "sam_vit_h",
            "name": "SAM ViT-H",
            "category": "segmentation",
            "task": "General-purpose image segmentation",
            "architecture": "ViT-H (Segment Anything)",
            "data": "SA-1B (pretrained)",
            "size_mb": round(sam_path.stat().st_size / 1e6, 1),
            "status": "pretrained",
            "path": str(sam_path),
        })

    return catalog


@router.get("/catalog")
async def get_model_catalog():
    """
    Return complete inventory of all trained/pretrained ML models.

    Categories: segmentation, synergy, treatment_response, energy
    """
    catalog = _get_model_catalog()
    summary = {}
    for item in catalog:
        cat = item["category"]
        summary[cat] = summary.get(cat, 0) + 1

    return {
        "total_models": len(catalog),
        "by_category": summary,
        "models": catalog,
    }


@router.get("/{model_id}/metrics")
async def get_model_metrics(model_id: str):
    """Get detailed performance metrics for a specific model."""
    catalog = _get_model_catalog()
    model = next((m for m in catalog if m["id"] == model_id), None)
    if not model:
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail=f"Model {model_id} not found")
    return model


@router.get("/health")
async def models_health_check():
    """Check which models are loadable."""
    catalog = _get_model_catalog()
    return {
        "total": len(catalog),
        "trained": sum(1 for m in catalog if m["status"] == "trained"),
        "pretrained": sum(1 for m in catalog if m["status"] == "pretrained"),
        "checkpoints_only": sum(1 for m in catalog if m["status"] == "checkpoints_only"),
        "models": [{
            "id": m["id"],
            "status": m["status"],
            "category": m["category"],
        } for m in catalog],
    }
