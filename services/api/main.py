"""
FastAPI Main Application
3D Viewer API Integration
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from routers import viewer_3d

app = FastAPI(
    title="ADDS Clinical System API",
    description="물리학 기반 항암제 최적화 시스템",
    version="1.0.0"
)

# CORS (브라우저에서 API 호출 허용)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 프로덕션에서는 특정 도메인으로 제한
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 정적 파일 (3D Viewer HTML)
app.mount("/viewer", StaticFiles(directory="../../frontend"), name="viewer")

# Router 등록
app.include_router(viewer_3d.router, prefix="/api")

@app.get("/")
async def root():
    return {
        "message": "ADDS Clinical System API",
        "endpoints": {
            "3d_viewer": "/viewer/3d_viewer.html",
            "api_docs": "/docs",
            "patient_3d_meshes": "/api/patients/{patient_id}/3d-meshes"
        }
    }

@app.get("/health")
async def health_check():
    return {"status": "healthy"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
