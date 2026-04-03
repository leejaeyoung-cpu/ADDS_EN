"""
Streamlit-FastAPI Integration
Connect frontend to backend API
"""

import streamlit as st
import requests
from typing import Dict, Any, Optional
import asyncio
import httpx

# Backend configuration
BACKEND_URL = st.secrets.get("BACKEND_URL", "http://localhost:8000")

class ADDSBackendClient:
    """Client for ADDS FastAPI backend"""
    
    def __init__(self, base_url: str = BACKEND_URL):
        self.base_url = base_url
        self.session = requests.Session()
    
    def health_check(self) -> Dict[str, Any]:
        """Check backend health"""
        try:
            response = self.session.get(f"{self.base_url}/health", timeout=5)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return {"status": "error", "message": str(e)}
    
    def segment_image(
        self,
        image_file,
        diameter: Optional[float] = None,
        flow_threshold: float = 0.6,
        cellprob_threshold: float = -1.0,
        batch_size: int = 8
    ) -> Dict[str, Any]:
        """
        Segment image via API
        
        Args:
            image_file: UploadedFile from Streamlit
            
        Returns:
            Segmentation result
        """
        try:
            files = {"file": (image_file.name, image_file.getvalue())}
            data = {
                "diameter": diameter,
                "flow_threshold": flow_threshold,
                "cellprob_threshold": cellprob_threshold,
                "batch_size": batch_size
            }
            
            response = self.session.post(
                f"{self.base_url}/api/v1/segmentation",
                files=files,
                data=data,
                timeout=60
            )
            response.raise_for_status()
            return response.json()
            
        except Exception as e:
            st.error(f"Segmentation API error: {str(e)}")
            return None
    
    def extract_features(
        self,
        image_id: str,
        masks: list,
        feature_set: str = "basic"
    ) -> Dict[str, Any]:
        """Extract morphological features"""
        try:
            response = self.session.post(
                f"{self.base_url}/api/v1/features/extract",
                json={
                    "image_id": image_id,
                    "masks": masks,
                    "feature_set": feature_set
                },
                timeout=30
            )
            response.raise_for_status()
            return response.json()
            
        except Exception as e:
            st.error(f"Feature extraction API error: {str(e)}")
            return None
    
    def calculate_synergy(
        self,
        drug_a_effect: float,
        drug_b_effect: float,
        combination_effect: float,
        model: str = "bliss"
    ) -> Dict[str, Any]:
        """Calculate drug synergy"""
        try:
            response = self.session.post(
                f"{self.base_url}/api/v1/synergy/calculate",
                json={
                    "drug_a_effect": drug_a_effect,
                    "drug_b_effect": drug_b_effect,
                    "combination_effect": combination_effect,
                    "model": model
                },
                timeout=10
            )
            response.raise_for_status()
            return response.json()
            
        except Exception as e:
            st.error(f"Synergy calculation API error: {str(e)}")
            return None
    
    def compare_groups(
        self,
        groups: Dict[str, list],
        features: list,
        test_type: str = "auto"
    ) -> Dict[str, Any]:
        """Compare experimental groups"""
        try:
            response = self.session.post(
                f"{self.base_url}/api/v1/statistics/compare",
                json={
                    "groups": groups,
                    "features": features,
                    "test_type": test_type
                },
                timeout=30
            )
            response.raise_for_status()
            return response.json()
            
        except Exception as e:
            st.error(f"Statistical comparison API error: {str(e)}")
            return None

# Singleton instance
@st.cache_resource
def get_backend_client():
    """Get or create backend client"""
    return ADDSBackendClient()
