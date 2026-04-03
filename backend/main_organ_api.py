"""
FastAPI Main Application with Organ Mesh Endpoints
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import sys
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent))

from api.organ_mesh_api import router as organ_mesh_router

app = FastAPI(title="ADDS 3D Organ Visualization API")

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 개발용, production에서는 특정 도메인만
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Organ Mesh API 라우터 등록
app.include_router(organ_mesh_router)

@app.get("/")
async def root():
    return {
        "message": "ADDS 3D Organ Visualization API",
        "endpoints": {
            "catalog": "/api/organ-meshes/catalog",
            "organ": "/api/organ-meshes/organ/{organ_name}",
            "tumor": "/api/organ-meshes/tumor/{organ_name}",
            "stats": "/api/organ-meshes/stats"
        }
    }

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
