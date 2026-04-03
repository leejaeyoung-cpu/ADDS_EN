"""
ADDS FastAPI Backend
Medical-grade cell analysis API based on Cellpose segmentation
"""

# ==================== GPU DEVICE SELECTION ====================
# Force PyTorch and Cellpose to use NVIDIA GPU (GPU 0) instead of AMD (GPU 1)
import os
import sys

# Fix Windows console encoding for Unicode characters
if sys.platform == 'win32':
    try:
        # Set console to UTF-8 mode
        os.system('chcp 65001 > nul')
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except:
        pass  # If reconfigure fails, continue with ASCII-safe output

# Set environment variable BEFORE importing any PyTorch/Cellpose modules
os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # Only GPU 0 (NVIDIA RTX 5070)
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'  # Use consistent device ordering

# Verify GPU selection
try:
    import torch
    if torch.cuda.is_available():
        torch.cuda.set_device(0)
        device_name = torch.cuda.get_device_name(0)
        print("=" * 60)
        print("[Backend] GPU Device Selected")
        print(f"  Device: {device_name}")
        print(f"  Index: 0")
        print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        print("=" * 60)
    else:
        print("[Backend] CUDA not available, using CPU")
except ImportError:
    print("[Backend] PyTorch not found, GPU selection skipped")
# =============================================================

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional
import uvicorn

# Import routers
from backend.api import (
    segmentation, features, statistics, synergy, ct_analysis, patients, 
    adds_inference, openai_inference, pharmacokinetics, metadata,
    nnunet_inference, energy_api,
    treatment_response_api, ml_synergy_api, model_catalog_api, biomarker_api
)


# Create FastAPI app
app = FastAPI(
    title="ADDS API",
    description="AI Anticancer Drug Discovery System - Cell Analysis Backend + CT CRC Detection",
    version="2.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8501",  # Streamlit
        "http://localhost:3000",  # React (if needed)
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
    max_age=3600,  # Cache preflight requests for 1 hour
)

# Include routers
app.include_router(segmentation.router, prefix="/api/v1/segmentation", tags=["Segmentation"])
app.include_router(features.router, prefix="/api/v1/features", tags=["Features"])
app.include_router(statistics.router, prefix="/api/v1/statistics", tags=["Statistics"])
app.include_router(synergy.router, prefix="/api/v1/synergy", tags=["Drug Synergy"])
app.include_router(ct_analysis.router, prefix="/api/v1/ct", tags=["CT Analysis"])

# Patient Management System routers
app.include_router(patients.router, prefix="/api/v1/patients", tags=["Patient Management"])
app.include_router(adds_inference.router, prefix="/api/v1/adds", tags=["ADDS Inference"])
app.include_router(openai_inference.router, prefix="/api/v1/openai", tags=["OpenAI Inference"])
app.include_router(pharmacokinetics.router)  # Prefix already in router definition

# Metadata Learning System router
app.include_router(metadata.router, tags=["Metadata Learning"])

# nnU-Net Segmentation router
app.include_router(nnunet_inference.router)  # Prefix already in router definition

# Energy Framework router
app.include_router(energy_api.router, prefix="/api/v1/energy", tags=["Energy Framework"])

# Treatment Response Prediction router
app.include_router(treatment_response_api.router, prefix="/api/v1/treatment-response", tags=["Treatment Response"])

# ML Synergy Prediction router
app.include_router(ml_synergy_api.router, prefix="/api/v1/ml-synergy", tags=["ML Synergy"])

# Model Catalog router
app.include_router(model_catalog_api.router, prefix="/api/v1/models", tags=["Model Catalog"])

# Biomarker Prediction router
app.include_router(biomarker_api.router, prefix="/api/v1/biomarkers", tags=["Biomarkers"])

@app.get("/")
async def root():
    """API Root - Health check"""
    return {
        "status": "healthy",
        "service": "ADDS API - Integrated Patient Management System",
        "version": "3.0.0",
        "endpoints": {
            "docs": "/api/docs",
            "segmentation": "/api/v1/segmentation",
            "features": "/api/v1/features",
            "statistics": "/api/v1/statistics",
            "synergy": "/api/v1/synergy",
            "ct_analysis": "/api/v1/ct",
            "patients": "/api/v1/patients",
            "adds_inference": "/api/v1/adds",
            "openai_inference": "/api/v1/openai",
            "energy": "/api/v1/energy",
            "treatment_response": "/api/v1/treatment-response",
            "ml_synergy": "/api/v1/ml-synergy",
            "model_catalog": "/api/v1/models",
            "biomarkers": "/api/v1/biomarkers"
        }
    }

@app.get("/health")
async def health_check():
    """Detailed health check — actual component verification"""
    components = {"api": "ok"}

    # GPU check
    try:
        import torch
        if torch.cuda.is_available():
            components["gpu"] = torch.cuda.get_device_name(0)
        else:
            components["gpu"] = "unavailable"
    except ImportError:
        components["gpu"] = "pytorch_not_installed"

    # Cellpose check
    try:
        import cellpose
        components["cellpose"] = "ok"
    except ImportError:
        components["cellpose"] = "not_installed"

    # Database check
    try:
        from backend.database_init import get_db
        from sqlalchemy import text
        db = next(get_db())
        db.execute(text("SELECT 1"))
        components["database"] = "ok"
    except Exception as e:
        components["database"] = f"error: {str(e)[:60]}"

    # Model files check
    from pathlib import Path
    _root = Path(__file__).parent.parent  # Project root (ADDS/)
    model_checks = {
        "nnunet": (_root / "models" / "sota_balanced" / "fold_0" / "best_model.pth").exists(),
        "deep_synergy": (_root / "models" / "synergy" / "deep_synergy_v2.pt").exists(),
        "treatment_response": (_root / "models" / "treatment_response" / "xgb_treatment_response.json").exists(),
    }
    components["models"] = {k: "ok" if v else "missing" for k, v in model_checks.items()}

    all_ok = all(
        v == "ok" or (isinstance(v, dict) and all(sv == "ok" for sv in v.values()))
        for v in components.values()
    )

    return {
        "status": "healthy" if all_ok else "degraded",
        "components": components
    }

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
