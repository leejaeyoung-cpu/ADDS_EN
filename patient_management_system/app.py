"""
ADDS Patient Management System - Main Application
FastAPI application with patient management and treatment planning
"""

from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
import uvicorn

# Import routers
from .api.patients import router as patients_router
from .api.ct_analysis import router as ct_router
from .api.adds_inference import router as adds_router
from .api.openai_inference import router as openai_router
from .api.cell_culture import router as cell_culture_router
from .api.enhanced_endpoints import router as enhanced_router  # NEW: CDSS metadata system

# Import database initialization
from .database import init_database
from .database.db_enhanced import init_db as init_enhanced_db  # NEW: Enhanced DB


# Initialize FastAPI app
app = FastAPI(
    title="ADDS Patient Management System",
    description="AI-based Anticancer Drug Discovery System with CT + Cell Culture Analysis",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Get base directory
BASE_DIR = Path(__file__).parent

# Mount static files
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")

# Templates
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

# Include API routers
app.include_router(patients_router, prefix="/api/patients", tags=["Patients"])
app.include_router(ct_router, prefix="/api/ct", tags=["CT Analysis"])
app.include_router(cell_culture_router, prefix="/api/microscopy", tags=["Cell Culture"])
app.include_router(adds_router, prefix="/api/adds", tags=["ADDS Inference"])
app.include_router(openai_router, prefix="/api/openai", tags=["OpenAI Inference"])
app.include_router(enhanced_router, tags=["CDSS Metadata System"])  # NEW


@app.on_event("startup")
async def startup_event():
    """Initialize database on startup"""
    print("Starting ADDS Patient Management System...")
    
    # Initialize original database
    init_database()
    
    # Initialize enhanced metadata database
    print("Initializing CDSS metadata learning system...")
    init_enhanced_db()
    
    print("System ready!")



@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Homepage"""
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/patient/register", response_class=HTMLResponse)
async def patient_register_page(request: Request):
    """Patient registration page"""
    return templates.TemplateResponse("patient_register.html", {"request": request})


@app.get("/comparison", response_class=HTMLResponse)
async def comparison_page(request: Request):
    """Dual-inference comparison page"""
    return templates.TemplateResponse("comparison.html", {"request": request})


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "ADDS Patient Management System",
        "version": "1.0.0"
    }


if __name__ == "__main__":
    print("=" * 60)
    print("ADDS Patient Management System")
    print("=" * 60)
    print("\nStarting server at http://localhost:8000")
    print("API documentation at http://localhost:8000/docs")
    print("\nPress Ctrl+C to stop\n")
    
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
