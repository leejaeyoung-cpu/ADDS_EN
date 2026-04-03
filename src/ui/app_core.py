"""
Core utilities and shared resources for ADDS UI
Centralized GPU configuration, caching, and styling
"""

import streamlit as st
import os
import sys
import torch
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from preprocessing.image_processor import CellposeProcessor
from preprocessing.document_parser import DocumentParser
from utils.synergy_calculator import SynergyCalculator
import requests


# ==================== GPU DEVICE SELECTION ====================
# Force PyTorch and Cellpose to use NVIDIA GPU (GPU 0) instead of AMD (GPU 1)

def configure_gpu():
    """Configure GPU settings for optimal performance"""
    # Fix Windows console encoding for Unicode characters
    if sys.platform == 'win32':
        try:
            os.system('chcp 65001 > nul')
            sys.stdout.reconfigure(encoding='utf-8')
            sys.stderr.reconfigure(encoding='utf-8')
        except:
            pass
    
    # CRITICAL: Set environment variable BEFORE any CUDA initialization
    # This ensures only NVIDIA GPU (device 0) is visible to PyTorch
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # Only GPU 0 (NVIDIA RTX 5070)
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'  # Use consistent device ordering
    
    # If CUDA is available, set default device to GPU 0 and verify
    if torch.cuda.is_available():
        try:
            # Explicitly set device to 0
            torch.cuda.set_device(0)
            
            # Verify we're using the correct GPU
            current_device = torch.cuda.current_device()
            device_name = torch.cuda.get_device_name(current_device)
            
            print("=" * 60)
            print("[UI] GPU Device Selected")
            print(f"   Device: {device_name}")
            print(f"   Index: {current_device}")
            print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
            
            # Verify this is NVIDIA GPU (not AMD)
            if "NVIDIA" not in device_name.upper():
                print(f"[WARNING] Non-NVIDIA GPU detected: {device_name}")
                print("   To use NVIDIA GPU:")
                print("   1. Windows Graphics Settings: Set Python to 'High Performance'")
                print("   2. NVIDIA Control Panel: Set global to 'NVIDIA GPU'")
            else:
                print("[OK] NVIDIA GPU in use")
            print("=" * 60)
            
        except Exception as e:
            print(f"[WARNING] GPU setup error: {e}")
            print("   Falling back to CPU mode")
    else:
        print("=" * 60)
        print("[WARNING] CUDA not available - Running in CPU mode")
        print("=" * 60)


# Configure GPU on module load
configure_gpu()
# =============================================================


# Cached resource functions for performance optimization
@st.cache_resource(show_spinner=False)
def get_cellpose_processor(model_type: str, gpu: bool = False):
    """
    캐시된 Cellpose processor 반환
    모델은 첫 실행 시에만 로딩되고 이후에는 재사용됩니다.
    GPU/CPU 모드별로 별도로 캐싱됩니다.
    
    Args:
        model_type: Cellpose 모델 타입 ('cyto', 'cyto2', 'nuclei')
        gpu: GPU 사용 여부
    
    Returns:
        CellposeProcessor 인스턴스
    """
    import logging
    logger = logging.getLogger(__name__)
    device_mode = "GPU" if gpu else "CPU"
    logger.info(f"🔧 Creating NEW Cellpose processor: model={model_type}, device={device_mode}")
    return CellposeProcessor(model_type=model_type, gpu=gpu)


@st.cache_resource
def get_document_parser():
    """캐시된 Document parser 반환"""
    return DocumentParser()


@st.cache_resource
def get_synergy_calculator():
    """캐시된 Synergy calculator 반환"""
    return SynergyCalculator()


def check_backend_status():
    """백엔드 API 상태 확인"""
    try:
        response = requests.get("http://localhost:8000/", timeout=2)
        if response.status_code == 200:
            return True, response.json()
        return False, None
    except:
        return False, None


def setup_page_config():
    """Configure Streamlit page settings"""
    st.set_page_config(
        page_title="ADDS - AI Anticancer Drug Discovery System",
        page_icon="🧬",
        layout="wide",
        initial_sidebar_state="expanded"
    )


def apply_custom_css():
    """Apply custom CSS styling"""
    st.markdown("""
    <style>
        .main-header {
            font-size: 3.5rem;
            font-weight: 700;
            background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 1rem;
        }
        .sub-header {
            font-size: 1.8rem;
            color: #555;
            margin-bottom: 2rem;
        }
        .metric-card {
            padding: 1.5rem;
            border-radius: 10px;
            background: linear-gradient(135deg, #667eea20 0%, #764ba220 100%);
            border-left: 4px solid #667eea;
        }
    </style>
    """, unsafe_allow_html=True)
