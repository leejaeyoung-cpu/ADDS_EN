"""
Metadata Editor Component
Editable form for experimental metadata with auto-parsing
"""

import streamlit as st
from pathlib import Path
from typing import Dict, Optional
from datetime import datetime


def show_metadata_editor(
    filename: str = None,
    auto_parse: bool = True,
    existing_metadata: Dict = None
) -> Optional[Dict]:
    """
    Display editable metadata form with auto-parsing
    
    Args:
        filename: Image filename for auto-parsing
        auto_parse: Enable automatic parsing from filename
        existing_metadata: Pre-fill with existing metadata
    
    Returns:
        Dictionary with metadata or None if not submitted
    """
    st.subheader("✏️ 실험 메타데이터 입력/편집")
    
    # Auto-parse from filename if available
    parsed = {}
    if filename and auto_parse:
        try:
            from utils.filename_parser import parse_filename_metadata, format_metadata_preview
            parsed = parse_filename_metadata(filename)
            
            if parsed:
                st.success(f"✅ 파일명에서 자동 감지: {format_metadata_preview(parsed)}")
        except Exception as e:
            st.warning(f"⚠️ 자동 파싱 실패: {str(e)}")
    
    # Merge parsed with existing
    initial_values = {**(existing_metadata or {}), **parsed}
    
    # Create editable form
    st.markdown("**기본 정보**")
    
    with st.form("metadata_form", clear_on_submit=False):
        col1, col2 = st.columns(2)
        
        with col1:
            experiment_name = st.text_input(
                "실험명 *",
                value=initial_values.get('experiment_name', ''),
                help="예: Exp-A, Drug-Screening-001",
                placeholder="실험을 식별할 이름"
            )
            
            cell_line = st.text_input(
                "세포주 *",
                value=initial_values.get('cell_line', ''),
                help="예: HUVEC, HEK293, MCF7",
                placeholder="사용된 세포주"
            )
            
            treatment = st.text_input(
                "처리/약물",
                value=initial_values.get('treatment', ''),
                help="예: Drug-A, TNF-α, Control",
                placeholder="처리 조건"
            )
            
            concentration = st.text_input(
                "농도",
                value=initial_values.get('concentration', ''),
                help="예: 10μM, 5mg/ml, N/A",
                placeholder="약물 농도"
            )
        
        with col2:
            time_point = st.text_input(
                "시간",
                value=initial_values.get('time_point', ''),
                help="예: 24hr, 48h, 72hours",
                placeholder="처리 시간"
            )
            
            replicate = st.number_input(
                "반복 번호",
                min_value=1,
                max_value=100,
                value=int(initial_values.get('replicate', 1)),
                help="생물학적 반복 번호"
            )
            
            condition = st.text_input(
                "조건",
                value=initial_values.get('condition', ''),
                help="예: Normoxia, Hypoxia, Serum-free",
                placeholder="실험 조건"
            )
            
            researcher = st.text_input(
                "연구자",
                value=initial_values.get('researcher', ''),
                placeholder="담당 연구자 이름"
            )
        
        st.markdown("---")
        st.markdown("**추가 정보**")
        
        notes = st.text_area(
            "메모",
            value=initial_values.get('notes', ''),
            height=100,
            help="실험 관련 추가 메모",
            placeholder="특이사항, 관찰 내용, 참고사항 등..."
        )
        
        # Submit buttons
        col1, col2, col3 = st.columns([1, 1, 2])
        
        with col1:
            submitted = st.form_submit_button(
                "💾 저장",
                type="primary",
                use_container_width=True
            )
        
        with col2:
            reset = st.form_submit_button(
                "🔄 초기화",
                use_container_width=True
            )
        
        if submitted:
            # Validate required fields
            if not experiment_name or not cell_line:
                st.error("❌ 실험명과 세포주는 필수 입력 사항입니다")
                return None
            
            # Create metadata dictionary
            metadata = {
                'experiment_name': experiment_name,
                'cell_line': cell_line,
                'treatment': treatment if treatment else None,
                'concentration': concentration if concentration else None,
                'time_point': time_point if time_point else None,
                'condition': condition if condition else None,
                'replicate': replicate,
                'researcher': researcher if researcher else None,
                'notes': notes if notes else None,
                'image_filename': filename,
                'created_at': datetime.now().isoformat()
            }
            
            st.success("✅ 메타데이터가 저장되었습니다")
            
            # Display summary
            with st.expander("📋 저장된 메타데이터 확인"):
                st.json(metadata)
            
            return metadata
        
        elif reset:
            st.info("🔄 폼이 초기화되었습니다")
            st.rerun()
    
    return None


def show_metadata_preview(metadata: Dict):
    """
    Display metadata in a nicely formatted way
    
    Args:
        metadata: Metadata dictionary
    """
    st.markdown("### 📋 메타데이터 요약")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("실험명", metadata.get('experiment_name', 'N/A'))
        st.metric("세포주", metadata.get('cell_line', 'N/A'))
    
    with col2:
        st.metric("처리", metadata.get('treatment', 'N/A'))
        st.metric("농도", metadata.get('concentration', 'N/A'))
    
    with col3:
        st.metric("시간", metadata.get('time_point', 'N/A'))
        st.metric("반복", f"#{metadata.get('replicate', 1)}")
    
    if metadata.get('notes'):
        st.info(f"💬 메모: {metadata['notes']}")


if __name__ == "__main__":
    # For testing
    st.set_page_config(page_title="Metadata Editor", layout="wide")
    
    st.title("Metadata Editor Test")
    
    # Test with sample filename
    test_filename = "HUVEC_TNFa_10uM_24hr_Rep2.tif"
    
    metadata = show_metadata_editor(
        filename=test_filename,
        auto_parse=True
    )
    
    if metadata:
        st.markdown("---")
        show_metadata_preview(metadata)
