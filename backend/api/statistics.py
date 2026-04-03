"""
Statistics API  
Statistical comparison and analysis endpoints
"""

from fastapi import APIRouter, HTTPException
from typing import Dict, List
import pandas as pd
from scipy import stats

from backend.services.stats_service import StatsService
from backend.schemas.analysis_schema import CompareRequest, CompareResponse

router = APIRouter()
service = StatsService()

@router.post("/compare", response_model=CompareResponse)
async def compare_groups(request: CompareRequest):
    """
    Compare multiple experimental groups statistically
    
    Tests:
    - auto: Automatically select (Shapiro-Wilk → t-test/Mann-Whitney)
    - anova: One-way ANOVA + Tukey HSD
    - kruskal: Kruskal-Wallis + Dunn's test
    
    Returns:
        p-values, effect sizes, and post-hoc results
    """
    
    try:
        result = await service.compare(
            groups=request.groups,
            features=request.features,
            test_type=request.test_type
        )
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Statistical comparison failed: {str(e)}")

@router.get("/tests")
async def list_tests():
    """List available statistical tests"""
    return {
        "tests": {
            "parametric": ["t-test", "anova", "paired-t"],
            "non-parametric": ["mann-whitney", "kruskal-wallis", "wilcoxon"],
            "post-hoc": ["tukey-hsd", "dunns-test"]
        },
        "corrections": ["bonferroni", "fdr-bh", "holm"]
    }
