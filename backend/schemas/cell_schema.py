"""
Pydantic schemas for cell segmentation and features
"""

from pydantic import BaseModel, Field
from typing import Optional, List, Dict
import numpy as np

class SegmentationRequest(BaseModel):
    """Request schema for cell segmentation"""
    diameter: Optional[float] = Field(None, description="Cell diameter in pixels")
    flow_threshold: float = Field(0.6, ge=0.1, le=3.0)
    cellprob_threshold: float = Field(-1.0, ge=-6.0, le=6.0)
    batch_size: int = Field(8, ge=1, le=32)
    
    class Config:
        schema_extra = {
            "example": {
                "diameter": None,
                "flow_threshold": 0.6,
                "cellprob_threshold": -1.0,
                "batch_size": 8
            }
        }

class SegmentationResponse(BaseModel):
    """Response schema for segmentation results"""
    image_id: str
    cell_count: int
    masks_shape: List[int]
    metadata: Dict
    
    class Config:
        schema_extra = {
            "example": {
                "image_id": "img_12345",
                "cell_count": 792,
                "masks_shape": [1024, 1024],
                "metadata": {
                    "diameter_used": 28.5,
                    "processing_time": 12.3
                }
            }
        }

class FeatureRequest(BaseModel):
    """Request schema for feature extraction"""
    image_id: str
    masks: List[List[int]]  # 2D array
    feature_set: str = Field("basic", pattern="^(basic|advanced|all)$")
    
class FeatureResponse(BaseModel):
    """Response schema for extracted features"""
    image_id: str
    num_cells: int
    features: Dict  # Cell features dataframe as dict
    summary: Dict   # Summary statistics
