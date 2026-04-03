"""
Pydantic schemas for analysis and statistics
"""

from pydantic import BaseModel, Field
from typing import Dict, List, Optional

class CompareRequest(BaseModel):
    """Request for statistical group comparison"""
    groups: Dict[str, List[str]] = Field(..., description="Group name to image IDs mapping")
    features: List[str]
    test_type: str = Field("auto", pattern="^(auto|anova|kruskal|ttest)$")
    
    class Config:
        schema_extra = {
            "example": {
                "groups": {
                    "control": ["img_001", "img_002"],
                    "drug_a": ["img_003", "img_004"]
                },
                "features": ["area", "circularity"],
                "test_type": "auto"
            }
        }

class CompareResponse(BaseModel):
    """Response for group comparison"""
    test_used: str
    p_values: Dict[str, float]
    effect_sizes: Dict[str, float]
    post_hoc: Optional[Dict] = None

class SynergyRequest(BaseModel):
    """Request for synergy calculation"""
    drug_a_effect: float = Field(..., ge=0, le=1, description="Drug A effect (0-1)")
    drug_b_effect: float = Field(..., ge=0, le=1, description="Drug B effect (0-1)")
    combination_effect: float = Field(..., ge=0, le=1, description="Combination effect (0-1)")
    model: str = Field("bliss", pattern="^(bliss|loewe|hsa|zip)$")
    
class SynergyResponse(BaseModel):
    """Response for synergy calculation"""
    synergy_score: float
    expected_effect: float
    observed_effect: float
    model: str
    interpretation: str
