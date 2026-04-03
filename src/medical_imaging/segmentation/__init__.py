"""
Multi-organ segmentation module for CT imaging.
Integrates TotalSegmentator for automatic organ boundary detection.
"""

from .organ_segmentation_engine import OrganSegmentationEngine
from .organ_boundary_detector import OrganBoundaryDetector

__all__ = [
    'OrganSegmentationEngine',
    'OrganBoundaryDetector'
]
