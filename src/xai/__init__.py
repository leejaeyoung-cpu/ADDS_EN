"""
XAI (Explainable AI) Module for ADDS
Provides interpretability for AI predictions using LIME, SHAP, Grad-CAM and Counterfactuals
"""

from .lime_explainer import LIMEExplainer
from .counterfactual_generator import CounterfactualGenerator
from .gradcam_visualizer import GradCAM, CellposeGradCAM

# SHAP analyzer to be implemented
# from .shap_analyzer import SHAPAnalyzer

__all__ = [
    'LIMEExplainer',
    'CounterfactualGenerator', 
    'GradCAM',
    'CellposeGradCAM'
]
