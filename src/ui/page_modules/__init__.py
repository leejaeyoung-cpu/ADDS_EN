"""
ADDS UI Pages Module
Page modules for ADDS UI
"""

from .home import show_home
from .image_analysis import show_image_analysis
from .document_processing import show_document_processing
from .drug_cocktail import show_drug_cocktail
from .synergy_calculation import show_synergy_calculation
from .ai_prediction import show_ai_prediction
from .data_management import show_data_management
from .dashboard import show_dashboard

# CDSS Metadata Learning Components
from .outcome_collection import show_outcome_collection, show_outcome_statistics
from .physician_notes_entry import show_notes_entry
from .data_management_cdss_enhanced import show_cdss_dashboard

__all__ = [
    'show_home',
    'show_image_analysis',
    'show_document_processing',
    'show_drug_cocktail',
    'show_synergy_calculation',
    'show_ai_prediction',
    'show_data_management',
    'show_dashboard',
    'show_outcome_collection',
    'show_outcome_statistics',
    'show_notes_entry',
    'show_cdss_dashboard'
]
