"""
ADDS Home Page
Professional infographic design showing system overview and metrics
"""

import streamlit as st
import torch
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from ui.app_core import check_backend_status


def show_home():
    """
    Home page with dual dashboard modes
    - Clinical: For daily clinical work (EMR style)
    - Presentation: For patient consultations (modern minimalist)
    """
    
    # Import mode-specific functions
    from ui.page_modules.home_clinical import show_home_clinical
    from ui.page_modules.home_presentation import show_home_presentation
    
    # Get current dashboard mode from session state
    dashboard_mode = st.session_state.get('dashboard_mode', 'Clinical')
    
    # Dispatch to appropriate dashboard mode
    if dashboard_mode == 'Clinical':
        show_home_clinical()
    elif dashboard_mode == 'Presentation':
        show_home_presentation()
    else:
        # Fallback to clinical mode if unknown
        st.warning(f"Unknown dashboard mode: {dashboard_mode}. Falling back to Clinical mode.")
        show_home_clinical()
