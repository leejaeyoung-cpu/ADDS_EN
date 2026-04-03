"""
Real-time Process Monitor for 5-Stage Pipeline Tracking
Provides visual progress indicators and performance metrics for analysis workflows
"""

import time
from typing import Optional, List, Dict, Any
from dataclasses import dataclass
from datetime import datetime


@dataclass
class ProcessStage:
    """Represents a single stage in the processing pipeline"""
    name: str
    description: str
    progress_percent: int
    icon: str = "⏳"


class ProcessMonitor:
    """
    Monitor and track progress through a multi-stage processing pipeline
    Designed for integration with Streamlit UI components
    """
    
    # Predefined 5-stage pipeline for image analysis
    DEFAULT_STAGES = [
        ProcessStage("준비", "이미지 로딩 및 전처리", 10, "📁"),
        ProcessStage("모델 로딩", "Cellpose 모델 초기화", 20, "🤖"),
        ProcessStage("세그멘테이션", "세포 분할 실행 중", 40, "🔬"),
        ProcessStage("특징 추출", "세포 특징 계산 중", 70, "📊"),
        ProcessStage("완료", "결과 저장 및 시각화", 100, "✅")
    ]
    
    def __init__(self, stages: Optional[List[ProcessStage]] = None):
        """
        Initialize process monitor
        
        Args:
            stages: Custom stage definitions (uses DEFAULT_STAGES if None)
        """
        self.stages = stages if stages is not None else self.DEFAULT_STAGES
        self.current_stage_idx = -1
        self.start_time = None
        self.stage_times: Dict[int, float] = {}
        self.is_complete = False
        
    def start(self):
        """Start the monitoring process"""
        self.start_time = time.time()
        self.current_stage_idx = -1
        self.stage_times = {}
        self.is_complete = False
        
    def next_stage(self) -> Optional[ProcessStage]:
        """
        Move to the next stage in the pipeline
        
        Returns:
            The new current stage, or None if already at the end
        """
        # Record time for current stage
        if self.current_stage_idx >= 0:
            self.stage_times[self.current_stage_idx] = time.time()
        
        # Move to next stage
        self.current_stage_idx += 1
        
        if self.current_stage_idx >= len(self.stages):
            self.is_complete = True
            return None
            
        return self.stages[self.current_stage_idx]
    
    def get_current_stage(self) -> Optional[ProcessStage]:
        """Get the current stage"""
        if 0 <= self.current_stage_idx < len(self.stages):
            return self.stages[self.current_stage_idx]
        return None
    
    def get_progress_percent(self) -> int:
        """Get current overall progress percentage"""
        if self.current_stage_idx < 0:
            return 0
        if self.is_complete or self.current_stage_idx >= len(self.stages):
            return 100
        return self.stages[self.current_stage_idx].progress_percent
    
    def get_status_text(self, include_icon: bool = True) -> str:
        """
        Get formatted status text for current stage
        
        Args:
            include_icon: Whether to include emoji icon
            
        Returns:
            Formatted status string (e.g., "⏳ 2/5: 모델 로딩 - Cellpose 모델 초기화")
        """
        if self.current_stage_idx < 0:
            return "대기 중..."
        
        if self.is_complete:
            return "✅ 완료!"
        
        stage = self.get_current_stage()
        if stage is None:
            return "완료"
        
        stage_num = self.current_stage_idx + 1
        total_stages = len(self.stages)
        
        icon = f"{stage.icon} " if include_icon else ""
        return f"{icon}{stage_num}/{total_stages}: {stage.name} - {stage.description}"
    
    def get_elapsed_time(self) -> float:
        """Get elapsed time in seconds since start"""
        if self.start_time is None:
            return 0.0
        return time.time() - self.start_time
    
    def get_stage_duration(self, stage_idx: int) -> Optional[float]:
        """
        Get duration of a specific stage in seconds
        
        Args:
            stage_idx: Index of the stage
            
        Returns:
            Duration in seconds, or None if stage hasn't completed
        """
        if stage_idx not in self.stage_times:
            return None
        
        # Calculate duration
        if stage_idx == 0:
            start = self.start_time
        else:
            start = self.stage_times.get(stage_idx - 1, self.start_time)
        
        end = self.stage_times[stage_idx]
        return end - start if start else None
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """
        Get performance summary for all completed stages
        
        Returns:
            Dictionary with stage names and their durations
        """
        summary = {
            'total_time': self.get_elapsed_time(),
            'stages': []
        }
        
        for idx, stage in enumerate(self.stages):
            duration = self.get_stage_duration(idx)
            if duration is not None:
                summary['stages'].append({
                    'name': stage.name,
                    'duration': duration,
                    'percent_of_total': (duration / summary['total_time'] * 100) if summary['total_time'] > 0 else 0
                })
        
        return summary
    
    def format_time(self, seconds: float) -> str:
        """Format seconds into readable string"""
        if seconds < 60:
            return f"{seconds:.1f}초"
        elif seconds < 3600:
            minutes = int(seconds // 60)
            secs = int(seconds % 60)
            return f"{minutes}분 {secs}초"
        else:
            hours = int(seconds // 3600)
            minutes = int((seconds % 3600) // 60)
            return f"{hours}시간 {minutes}분"


class StreamlitProcessMonitor(ProcessMonitor):
    """
    Extended ProcessMonitor with Streamlit-specific UI integration
    Automatically manages progress bar and status text containers
    """
    
    def __init__(self, progress_bar=None, status_text=None, stages: Optional[List[ProcessStage]] = None):
        """
        Initialize with Streamlit UI containers
        
        Args:
            progress_bar: Streamlit progress bar widget (from st.progress())
            status_text: Streamlit empty container (from st.empty())
            stages: Custom stage definitions
        """
        super().__init__(stages)
        self.progress_bar = progress_bar
        self.status_text = status_text
    
    def update_ui(self):
        """Update Streamlit UI components with current progress"""
        if self.progress_bar is not None:
            self.progress_bar.progress(self.get_progress_percent())
        
        if self.status_text is not None:
            self.status_text.text(self.get_status_text())
    
    def next_stage(self) -> Optional[ProcessStage]:
        """Move to next stage and update UI"""
        stage = super().next_stage()
        self.update_ui()
        return stage
    
    def complete(self):
        """Mark process as complete and update UI"""
        self.is_complete = True
        if self.progress_bar is not None:
            self.progress_bar.progress(100)
        if self.status_text is not None:
            self.status_text.text("✅ 처리 완료!")
