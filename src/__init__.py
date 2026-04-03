"""
ADDS (AI Anticancer Drug Discovery System)
Main package initialization
"""

__version__ = "1.0.0"
__author__ = "인하대학병원 의생명공학과"
__description__ = "AI-powered anticancer drug cocktail discovery and optimization system"

from pathlib import Path

# Package root directory
PACKAGE_ROOT = Path(__file__).parent.parent
DATA_DIR = PACKAGE_ROOT / "data"
MODELS_DIR = PACKAGE_ROOT / "models"
CONFIGS_DIR = PACKAGE_ROOT / "configs"
