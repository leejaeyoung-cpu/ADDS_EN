"""Clinical module for precision oncology pipeline"""

# Import all modules
from . import clinical_database
from . import cohort_classifier

# Alias for validation compatibility
patient_profile = clinical_database
