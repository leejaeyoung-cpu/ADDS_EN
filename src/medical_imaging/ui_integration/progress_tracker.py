"""
Progress Tracker for UI
Real-time progress updates for long-running tasks
"""

import time
from typing import Callable, Optional


class ProgressTracker:
    """
    Simple progress tracker for Streamlit UI
    """
    
    def __init__(self, total_stages: int = 5):
        """
        Args:
            total_stages: Total number of stages
        """
        self.total_stages = total_stages
        self.current_stage = 0
        self.stage_progress = 0.0
        self.stage_names = [
            "Loading DICOM series",
            "Preprocessing volume",
            "Running inference",
            "Post-processing",
            "Calculating metrics"
        ]
        self.start_time = None
    
    def start(self):
        """Start tracking"""
        self.start_time = time.time()
        self.current_stage = 0
        self.stage_progress = 0.0
    
    def update(self, stage: int, progress: float, message: str = ""):
        """
        Update progress
        
        Args:
            stage: Stage number (0-indexed)
            progress: Progress within stage (0.0-1.0)
            message: Optional custom message
        """
        self.current_stage = stage
        self.stage_progress = progress
        
        # Calculate overall progress
        overall_progress = (stage + progress) / self.total_stages
        
        # Get stage name
        if message:
            status = message
        else:
            stage_name = self.stage_names[stage] if stage < len(self.stage_names) else f"Stage {stage + 1}"
            status = f"{stage_name}... {progress * 100:.0f}%"
        
        return overall_progress, status
    
    def complete(self):
        """Mark as complete"""
        elapsed = time.time() - self.start_time if self.start_time else 0
        return 1.0, f"Complete! ({elapsed:.1f}s)"
