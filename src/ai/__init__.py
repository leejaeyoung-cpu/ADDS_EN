"""AI module for medical document analysis"""

# Import all modules
from . import openai_analyzer
from . import dataset_builder
from . import finetuning

# Alias for validation compatibility
document_parser = openai_analyzer

# Placeholder for future implementation
vision_analyzer = None  # Not yet implemented
