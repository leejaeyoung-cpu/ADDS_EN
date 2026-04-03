"""
ADDS Precision Oncology Module - Init

⚠️ DEPRECATED: This package is a duplicate.
The canonical version is: src/ui/show_precision_oncology.py
This package exists for backward compatibility only.
"""

import warnings
warnings.warn(
    "src.ui.precision_oncology is deprecated. "
    "Use src.ui.show_precision_oncology instead.",
    DeprecationWarning,
    stacklevel=2
)

from .main import show_precision_oncology

__all__ = ['show_precision_oncology']

