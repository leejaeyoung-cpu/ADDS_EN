"""
Metadata Viewer Component
Displays data provenance and processing lineage information
"""

import streamlit as st
from typing import Dict, Any, List
import json
from datetime import datetime
import plotly.graph_objects as go


def show_metadata_timeline(metadata: Dict[str, Any]):
    """
    Display metadata as a visual timeline showing data flow
    
    Args:
        metadata: Complete metadata dictionary with pipeline stages
    """
    st.markdown("### 📊 데이터 생성 과정")
    
    # Extract pipeline information
    stages = metadata.get('pipeline', {}).get('stages', [])
    
    if not stages:
        st.info("파이프라인 정보가 없습니다")
        return
    
    # Create Mermaid flowchart
    mermaid_code = "graph LR\n"
    
    for i, stage in enumerate(stages):
        stage_name = stage.get('stage_name', f'Stage {i+1}')
        timestamp = stage.get('timestamp', '')
        
        # Format node
        node_id = f"S{i}"
        next_node_id = f"S{i+1}" if i < len(stages) - 1 else None
        
        # Clean stage name for mermaid
        clean_name = stage_name.replace('"', "'")
        
        if i == 0:
            mermaid_code += f'    {node_id}["{clean_name}"]\n'
        else:
            mermaid_code += f'    {node_id}["{clean_name}"]\n'
        
        if next_node_id and i < len(stages) - 1:
            # Add arrow with timestamp
            time_str = timestamp.split('T')[1][:8] if 'T' in timestamp else ''
            mermaid_code += f'    {node_id} -->|{time_str}| {next_node_id}\n'
    
    # Display mermaid diagram
    st.markdown(f"```mermaid\n{mermaid_code}\n```")


def show_data_provenance(metadata: Dict[str, Any]):
    """
    Display data provenance information
    
    Args:
        metadata: Metadata dictionary with provenance info
    """
    st.markdown("### 🔍 데이터 출처")
    
    provenance = metadata.get('provenance', {})
    
    if not provenance:
        st.info("출처 정보가 없습니다")
        return
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 📁 원본 파일")
        st.write(f"**파일명**: {provenance.get('source_file', 'Unknown')}")
        st.write(f"**업로드 시간**: {provenance.get('upload_timestamp', 'Unknown')}")
        
        uploader = provenance.get('uploader', {})
        if uploader:
            st.write(f"**업로드자**: {uploader.get('author_name', 'Unknown')}")
            st.write(f"**소속**: {uploader.get('author_department', 'Unknown')}")
    
    with col2:
        st.markdown("#### 🔐 파일 정보")
        st.write(f"**해시**: {provenance.get('file_hash', 'N/A')[:16]}...")
        
        original_meta = provenance.get('original_metadata', {})
        if original_meta:
            st.write("**원본 메타데이터**:")
            for key, value in list(original_meta.items())[:5]:
                st.write(f"- {key}: {value}")


def show_model_inference_info(metadata: Dict[str, Any]):
    """
    Display model inference information
    
    Args:
        metadata: Metadata with model inference details
    """
    st.markdown("### 🤖 모델 추론 정보")
    
    inference = metadata.get('inference', {})
    
    if not inference:
        st.info("추론 정보가 없습니다")
        return
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 모델 정보")
        model_info = inference.get('model', {})
        st.write(f"**모델명**: {model_info.get('name', 'Unknown')}")
        st.write(f"**버전**: {model_info.get('version', 'Unknown')}")
        st.write(f"**프레임워크**: {model_info.get('framework', 'Unknown')}")
        st.write(f"**디바이스**: {model_info.get('device', 'Unknown')}")
    
    with col2:
        st.markdown("#### 추론 파라미터")
        params = inference.get('inference_params', {})
        for key, value in params.items():
            st.write(f"**{key}**: {value}")
        
        executor = inference.get('executed_by', {})
        if executor:
            st.write(f"**실행자**: {executor.get('author_name', 'Unknown')}")


def show_complete_metadata_viewer(metadata: Dict[str, Any]):
    """
    Display complete metadata viewer with all sections
    
    Args:
        metadata: Complete metadata dictionary
    """
    st.markdown("## 📋 완전한 메타데이터")
    
    # Create tabs for different views
    tab1, tab2, tab3, tab4 = st.tabs([
        "🔄 처리 과정",
        "🔍 데이터 출처",
        "🤖 모델 정보",
        "📄 전체 JSON"
    ])
    
    with tab1:
        show_metadata_timeline(metadata)
    
    with tab2:
        show_data_provenance(metadata)
    
    with tab3:
        show_model_inference_info(metadata)
    
    with tab4:
        st.markdown("### 📄 전체 메타데이터 (JSON)")
        st.json(metadata)
        
        # Download button
        json_str = json.dumps(metadata, ensure_ascii=False, indent=2)
        st.download_button(
            label="📥 JSON 다운로드",
            data=json_str,
            file_name=f"metadata_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json"
        )


def show_compact_metadata_card(metadata: Dict[str, Any]):
    """
    Display compact metadata card for quick overview
    
    Args:
        metadata: Metadata dictionary
    """
    author = metadata.get('author', {})
    
    st.info(f"""
    📋 **작성자**: {author.get('author_name', 'Unknown')} ({author.get('author_role', '')})  
    🏢 **소속**: {author.get('author_department', 'Unknown')}  
    📅 **시간**: {author.get('timestamp', 'Unknown')}  
    🔄 **분석 유형**: {metadata.get('analysis_type', 'Unknown')}
    """)


def create_processing_statistics(metadata: Dict[str, Any]) -> go.Figure:
    """
    Create statistics visualization from metadata
    
    Args:
        metadata: Metadata with processing information
        
    Returns:
        Plotly figure with statistics
    """
    stages = metadata.get('pipeline', {}).get('stages', [])
    
    if not stages:
        return None
    
    # Extract timing data
    stage_names = [s.get('stage_name', f'Stage {i+1}') for i, s in enumerate(stages)]
    durations = [s.get('duration_ms', 0) for s in stages]
    
    # Create bar chart
    fig = go.Figure(data=[
        go.Bar(
            x=stage_names,
            y=durations,
            text=[f"{d:.0f}ms" for d in durations],
            textposition='auto',
        )
    ])
    
    fig.update_layout(
        title="단계별 처리 시간",
        xaxis_title="처리 단계",
        yaxis_title="시간 (ms)",
        height=400
    )
    
    return fig
