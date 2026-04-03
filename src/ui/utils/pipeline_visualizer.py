"""
Pipeline Visualizer
Visualizes ML/DL processing pipeline stages for transparency and traceability
"""

import streamlit as st
import numpy as np
from PIL import Image
from datetime import datetime
from typing import Dict, Any, List, Optional
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path


class PipelineStage:
    """Represents a single stage in the processing pipeline"""
    
    def __init__(
        self,
        name: str,
        stage_number: int,
        image: Optional[np.ndarray] = None,
        metadata: Optional[Dict[str, Any]] = None,
        timestamp: Optional[datetime] = None
    ):
        self.name = name
        self.stage_number = stage_number
        self.image = image
        self.metadata = metadata or {}
        self.timestamp = timestamp or datetime.now()
        self.duration_ms = None
    
    def set_duration(self, duration_ms: float):
        """Set processing duration for this stage"""
        self.duration_ms = duration_ms


class PipelineVisualizer:
    """
    Visualizes ML/DL processing pipeline with multiple stages
    
    Usage:
        pipeline = PipelineVisualizer()
        pipeline.add_stage("원본 이미지", 1, original_img, {'size': img.shape})
        pipeline.add_stage("전처리", 2, preprocessed_img, {'method': 'CLAHE'})
        pipeline.display_pipeline()
    """
    
    def __init__(self, pipeline_name: str = "분석 파이프라인"):
        self.pipeline_name = pipeline_name
        self.stages: List[PipelineStage] = []
        self.start_time = datetime.now()
        self.total_duration_ms = None
    
    def add_stage(
        self,
        name: str,
        image: Optional[np.ndarray] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> PipelineStage:
        """
        Add a processing stage to the pipeline
        
        Args:
            name: Stage name (e.g., "원본 이미지", "전처리")
            image: Numpy array or PIL Image
            metadata: Dictionary with stage information
            
        Returns:
            Created PipelineStage object
        """
        stage_number = len(self.stages) + 1
        stage = PipelineStage(name, stage_number, image, metadata)
        self.stages.append(stage)
        return stage
    
    def finalize(self):
        """Mark pipeline as complete and calculate total duration"""
        end_time = datetime.now()
        self.total_duration_ms = (end_time - self.start_time).total_seconds() * 1000
    
    def display_pipeline(self, display_mode: str = "carousel"):
        """
        Display the complete pipeline in Streamlit
        
        Args:
            display_mode: "carousel", "tabs", or "grid"
        """
        if not self.stages:
            st.warning("파이프라인에 단계가 없습니다")
            return
        
        st.markdown(f"### 🔄 {self.pipeline_name}")
        
        # Pipeline overview
        self._display_overview()
        
        st.markdown("---")
        
        # Display stages based on mode
        if display_mode == "carousel":
            self._display_carousel()
        elif display_mode == "tabs":
            self._display_tabs()
        elif display_mode == "grid":
            self._display_grid()
    
    def _display_overview(self):
        """Display pipeline overview with metrics"""
        cols = st.columns(len(self.stages))
        
        for idx, stage in enumerate(self.stages):
            with cols[idx]:
                status_icon = "✅" if idx < len(self.stages) else "⏳"
                st.markdown(f"**{status_icon} {stage.name}**")
                
                if stage.duration_ms:
                    st.caption(f"⏱️ {stage.duration_ms:.0f}ms")
        
        # Total duration
        if self.total_duration_ms:
            st.info(f"⏱️ 전체 처리 시간: {self.total_duration_ms:.0f}ms ({self.total_duration_ms/1000:.2f}초)")
    
    def _display_carousel(self):
        """Display stages as a carousel (using markdown carousel syntax)"""
        
        # Create carousel markdown
        carousel_items = []
        
        for stage in self.stages:
            item_md = f"### {stage.stage_number}. {stage.name}\n\n"
            
            # Display image if available
            if stage.image is not None:
                # Convert to PIL if numpy
                if isinstance(stage.image, np.ndarray):
                    # Normalize to 0-255 if needed
                    img_display = stage.image.copy()
                    if img_display.dtype == np.float32 or img_display.dtype == np.float64:
                        img_display = (img_display * 255).astype(np.uint8)
                    
                    # Use st.image with unique key
                    st.image(img_display, caption=f"{stage.name}", use_container_width=True)
            
            # Display metadata
            if stage.metadata:
                item_md += "\n**📊 정보:**\n\n"
                for key, value in stage.metadata.items():
                    item_md += f"- **{key}**: {value}\n"
            
            carousel_items.append(item_md)
        
        # Display as tabs (simpler than actual carousel)
        tab_names = [f"{s.stage_number}. {s.name}" for s in self.stages]
        tabs = st.tabs(tab_names)
        
        for idx, (tab, stage) in enumerate(zip(tabs, self.stages)):
            with tab:
                # Display image
                if stage.image is not None:
                    img_display = stage.image.copy()
                    if isinstance(img_display, np.ndarray):
                        if img_display.dtype in [np.float32, np.float64]:
                            img_display = (img_display * 255).astype(np.uint8)
                    
                    st.image(img_display, caption=f"{stage.name}", use_container_width=True)
                
                # Display metadata in columns
                if stage.metadata:
                    st.markdown("#### 📊 처리 정보")
                    
                    # Create metrics for numeric values
                    numeric_metadata = {k: v for k, v in stage.metadata.items() 
                                       if isinstance(v, (int, float))}
                    
                    if numeric_metadata:
                        cols = st.columns(min(3, len(numeric_metadata)))
                        for idx, (key, value) in enumerate(numeric_metadata.items()):
                            with cols[idx % 3]:
                                if isinstance(value, float):
                                    st.metric(key, f"{value:.3f}")
                                else:
                                    st.metric(key, f"{value:,}")
                    
                    # Display other metadata
                    other_metadata = {k: v for k, v in stage.metadata.items() 
                                     if not isinstance(v, (int, float))}
                    
                    if other_metadata:
                        st.markdown("**상세 정보:**")
                        for key, value in other_metadata.items():
                            st.write(f"- **{key}**: {value}")
    
    def _display_tabs(self):
        """Display stages as tabs"""
        tab_names = [f"{s.stage_number}. {s.name}" for s in self.stages]
        tabs = st.tabs(tab_names)
        
        for tab, stage in zip(tabs, self.stages):
            with tab:
                self._display_stage_content(stage)
    
    def _display_grid(self):
        """Display stages in a grid layout"""
        cols_per_row = 3
        
        for i in range(0, len(self.stages), cols_per_row):
            cols = st.columns(cols_per_row)
            
            for j in range(cols_per_row):
                if i + j < len(self.stages):
                    with cols[j]:
                        stage = self.stages[i + j]
                        st.markdown(f"**{stage.stage_number}. {stage.name}**")
                        
                        if stage.image is not None:
                            st.image(stage.image, use_container_width=True)
                        
                        if stage.metadata:
                            with st.expander("상세 정보"):
                                st.json(stage.metadata)
    
    def _display_stage_content(self, stage: PipelineStage):
        """Display content for a single stage"""
        st.markdown(f"### {stage.name}")
        
        if stage.image is not None:
            st.image(stage.image, caption=f"{stage.name} 결과", use_container_width=True)
        
        if stage.metadata:
            st.markdown("#### 📊 처리 정보")
            
            # Organize metadata
            cols = st.columns(2)
            
            with cols[0]:
                for idx, (key, value) in enumerate(stage.metadata.items()):
                    if idx % 2 == 0:
                        st.write(f"**{key}**: {value}")
            
            with cols[1]:
                for idx, (key, value) in enumerate(stage.metadata.items()):
                    if idx % 2 == 1:
                        st.write(f"**{key}**: {value}")
    
    def get_timeline_data(self) -> Dict[str, Any]:
        """
        Get timeline data for visualization
        
        Returns:
            Dictionary with timeline information
        """
        timeline = {
            'pipeline_name': self.pipeline_name,
            'total_stages': len(self.stages),
            'total_duration_ms': self.total_duration_ms,
            'stages': []
        }
        
        for stage in self.stages:
            timeline['stages'].append({
                'name': stage.name,
                'stage_number': stage.stage_number,
                'timestamp': stage.timestamp.isoformat(),
                'duration_ms': stage.duration_ms,
                'metadata': stage.metadata
            })
        
        return timeline
    
    def display_timeline_chart(self):
        """Display pipeline timeline as a Gantt-style chart"""
        if not self.stages:
            return
        
        st.markdown("#### ⏱️ 파이프라인 타임라인")
        
        # Prepare data for Gantt chart
        start_time = self.start_time
        
        timeline_data = []
        current_time = 0
        
        for stage in self.stages:
            duration = stage.duration_ms if stage.duration_ms else 100  # Default 100ms
            
            timeline_data.append({
                'Stage': stage.name,
                'Start': current_time,
                'Duration': duration
            })
            
            current_time += duration
        
        # Create Gantt chart using plotly
        fig = go.Figure()
        
        for idx, data in enumerate(timeline_data):
            fig.add_trace(go.Bar(
                name=data['Stage'],
                x=[data['Duration']],
                y=[data['Stage']],
                orientation='h',
                text=f"{data['Duration']:.0f}ms",
                textposition='auto',
                marker=dict(color=px.colors.qualitative.Plotly[idx % len(px.colors.qualitative.Plotly)])
            ))
        
        fig.update_layout(
            title="처리 단계별 소요 시간",
            xaxis_title="시간 (ms)",
            yaxis_title="처리 단계",
            showlegend=False,
            height=300
        )
        
        st.plotly_chart(fig, use_container_width=True)


def example_usage():
    """Example of how to use PipelineVisualizer"""
    pipeline = PipelineVisualizer("Cellpose 세포 분석")
    
    # Stage 1: Original
    dummy_img = np.random.rand(256, 256, 3)
    pipeline.add_stage(
        "원본 이미지",
        dummy_img,
        {'크기': '256x256', '채널': 3, '타입': 'RGB'}
    )
    
    # Stage 2: Preprocessing
    pipeline.add_stage(
        "전처리 (CLAHE)",
        dummy_img * 1.2,
        {'방법': 'CLAHE', 'clip_limit': 2.0, 'tile_size': '8x8'}
    )
    
    # Stage 3: Inference
    pipeline.add_stage(
        "Cellpose 추론",
        dummy_img * 0.8,
        {'모델': 'cyto2', '검출된 세포': 1247, 'diameter': 30}
    )
    
    # Stage 4: Postprocessing
    pipeline.add_stage(
        "후처리 (마스크)",
        dummy_img * 0.6,
        {'컬러맵': 'jet', '마스크 수': 1247}
    )
    
    # Stage 5: Results
    pipeline.add_stage(
        "결과 추출",
        None,
        {
            '총 세포 수': 1247,
            '평균 면적': 185.3,
            'Ki-67 지표': 0.45,
            '형태학 점수': 9.1
        }
    )
    
    pipeline.finalize()
    pipeline.display_pipeline()
    pipeline.display_timeline_chart()
