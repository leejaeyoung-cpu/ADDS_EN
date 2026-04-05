"""Evaluation package"""

from .data_validator import DataQualityValidator

try:
    from .model_evaluator import ModelEvaluator, BiologicalValidator
    _has_model_evaluator = True
except ImportError:
    _has_model_evaluator = False
    ModelEvaluator = None
    BiologicalValidator = None

__all__ = [
    "DataQualityValidator",
    "ModelEvaluator",
    "BiologicalValidator",
]
