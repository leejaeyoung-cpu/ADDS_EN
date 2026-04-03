"""
Features API
Morphological feature extraction endpoints
"""

from fastapi import APIRouter, HTTPException
from typing import List, Optional
import pandas as pd

from backend.services.feature_service import FeatureService
from backend.schemas.cell_schema import FeatureRequest, FeatureResponse

router = APIRouter()
service = FeatureService()

@router.post("/extract", response_model=FeatureResponse)
async def extract_features(request: FeatureRequest):
    """
    Extract morphological features from segmented cells
    
    Feature sets:
    - basic: 15 features (area, perimeter, circularity, etc.)
    - advanced: +spatial features (Voronoi, density)
    - all: +texture features (GLCM)
    
    Returns:
        DataFrame with features per cell + summary statistics
    """
    
    try:
        result = await service.extract(
            image_id=request.image_id,
            masks=request.masks,
            feature_set=request.feature_set
        )
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Feature extraction failed: {str(e)}")

@router.get("/available")
async def list_features():
    """List all available features"""
    return {
        "basic": [
            "area", "perimeter", "circularity", "eccentricity", 
            "solidity", "extent", "major_axis", "minor_axis",
            "orientation", "mean_intensity", "std_intensity",
            "integrated_intensity", "contrast", "homogeneity", "energy"
        ],
        "spatial": [
            "nearest_neighbor_dist", "cell_density", "uniformity",
            "entropy", "symmetry"
        ],
        "texture": [
            "contrast_glcm", "dissimilarity_glcm", "homogeneity_glcm",
            "energy_glcm", "correlation_glcm"
        ]
    }
