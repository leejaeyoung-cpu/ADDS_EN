"""Pathology module for advanced quantitative analysis"""

# Import all modules
from . import spatial_analyzer
from . import heterogeneity_metrics

# Alias for validation compatibility
feature_extractor = spatial_analyzer

# Import CellposeProcessor from preprocessing as cellpose_segmenter
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
try:
    from preprocessing.image_processor import CellposeProcessor
    cellpose_segmenter = CellposeProcessor
except ImportError:
    cellpose_segmenter = None
