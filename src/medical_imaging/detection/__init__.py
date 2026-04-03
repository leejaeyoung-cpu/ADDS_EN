"""Detection module initialization"""

from .candidate_detector import (
    TumorCandidate,
    TumorDetector,
    CTPreprocessor,
    BodySegmentation,
    merge_candidates
)

from .optimized_detector import (
    OptimizedColonDetector,
    create_optimized_detector
)

__all__ = [
    'TumorCandidate',
    'TumorDetector',
    'CTPreprocessor',
    'BodySegmentation',
    'merge_candidates',
    'OptimizedColonDetector',
    'create_optimized_detector'
]
