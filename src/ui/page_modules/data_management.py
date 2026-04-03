"""
Show Data Management Page
"""

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import sys
from datetime import datetime

# Add src to path  
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from utils.analysis_db import AnalysisDatabase
from utils.system_utils import format_file_size, format_duration

# Import CDSS data management functions
try:
    from ui.page_modules.data_management_cdss import show_patient_ct_data, show_ml_performance_data
    CDSS_AVAILABLE = True
except ImportError:
    CDSS_AVAILABLE = False


def show_data_management():
    """Data management page - now with CDSS integration"""
    st.header("📊 데이터 관리")
    st.markdown("실험 데이터 및 CDSS 메타데이터 통합 관리")
    
    # Create tabs
    if CDSS_AVAILABLE:
        tab1, tab2, tab3, tab4 = st.tabs([
            "🔬 세포 실험 데이터",
            "🏥 환자 & CT 데이터",
            "🤖 ML 학습 성과",
            "📁 파일 시스템"
        ])
        
        with tab1:
            show_cell_experiment_data()
        
        with tab2:
            show_patient_ct_data()
        
        with tab3:
            show_ml_performance_data()
        
        with tab4:
            show_file_system_data()
    else:
        # Fallback to original single-view
        show_cell_experiment_data()


def show_cell_experiment_data():
    """Cell experiment data management (original functionality)"""
    # ===== ANALYSIS RESULTS FROM DATABASE =====
    st.markdown("### 📜 분석 결과 데이터베이스")
    
    try:
        db = AnalysisDatabase()
        stats = db.get_statistics()
        
        # Display statistics
        stat_cols = st.columns(3)
        with stat_cols[0]:
            st.metric("전체 분석 수", f"{stats['total_analyses']:,}")
        with stat_cols[1]:
            st.metric("평균 세포 수", f"{stats['avg_cells']:.0f}")
        with stat_cols[2]:
            st.metric("평균 품질 점수", f"{stats['avg_quality']:.2f}")
        
        st.markdown("---")
        
        # Search bar
        search_term = st.text_input("🔍 검색 (이미지명, 메모, 태그)", key="search_analysis")
        
        # Load analyses
        if search_term:
            analyses = db.search_analyses(search_term)
        else:
            analyses = db.get_all_analyses(limit=100)
        
        if analyses:
            st.success(f"✅ **{len(analyses)}개의 분석 결과 로드됨**")
            
            # Create DataFrame for display
            display_data = []
            for analysis in analyses:
                display_data.append({
                    'ID': analysis['id'],
                    '시각': pd.to_datetime(analysis['timestamp']).strftime('%Y-%m-%d %H:%M'),
                    '이미지': analysis['image_name'],
                    '세포 수': f"{analysis['num_cells']:,}",
                    '평균 크기': f"{analysis['mean_area']:.1f} px²",
                    '원형도': f"{analysis['mean_circularity']:.3f}",
                    '품질': analysis['quality_grade'],
                    '중요도': '⭐' * (analysis.get('importance', 0) or 0),
                    '태그': analysis.get('tags', '') or '',
                })
            
            df = pd.DataFrame(display_data)
            st.dataframe(df, use_container_width=True, hide_index=True)
            
            # ===== EDIT SECTION =====
            st.markdown("---")
            st.markdown("### ✏️ 분석 결과 편집")
            
            # Select record to edit
            selected_id = st.selectbox(
                "편집할 분석 선택",
                [a['id'] for a in analyses],
                format_func=lambda x: f"ID {x} - {next(a['image_name'] for a in analyses if a['id'] == x)} ({next(pd.to_datetime(a['timestamp']).strftime('%Y-%m-%d %H:%M') for a in analyses if a['id'] == x)})"
            )
            
            selected_analysis = db.get_analysis_by_id(selected_id)
            
            if selected_analysis:
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.markdown("#### 📝 메모")
                    notes = st.text_area(
                        "분석에 대한 메모 작성",
                        value=selected_analysis.get('notes', '') or '',
                        height=150,
                        key=f"notes_{selected_id}"
                    )
                    
                    st.markdown("#### 🏷️ 태그")
                    tags = st.text_input(
                        "태그 (쉼표로 구분)",
                        value=selected_analysis.get('tags', '') or '',
                        key=f"tags_{selected_id}",
                        help="예: 실험1, 대조군, 약물A"
                    )
                
                with col2:
                    st.markdown("#### ⭐ 중요도")
                    importance = st.slider(
                        "중요도 평가",
                        0, 5,
                        value=selected_analysis.get('importance', 0) or 0,
                        key=f"importance_{selected_id}"
                    )
                    
                    st.markdown("#### 📊 요약 정보")
                    st.metric("세포 수", f"{selected_analysis['num_cells']:,}")
                    st.metric("품질 점수", f"{selected_analysis['quality_score']:.2f}")
                
                # Save changes button
                col_save1, col_save2, col_save3 = st.columns([1, 1, 2])
                
                with col_save1:
                    if st.button("💾 변경사항 저장", type="primary", use_container_width=True):
                        try:
                            db.update_notes(selected_id, notes)
                            db.update_tags(selected_id, tags)
                            db.update_importance(selected_id, importance)
                            st.success("✅ 저장 완료!")
                            st.rerun()
                        except Exception as e:
                            st.error(f"저장 실패: {str(e)}")
                
                with col_save2:
                    if st.button("📥 JSON 내보내기", use_container_width=True):
                        import json
                        export_data = {k: str(v) if isinstance(v, (pd.Timestamp, datetime)) else v 
                                     for k, v in selected_analysis.items() if k != 'results_json'}
                        
                        json_str = json.dumps(export_data, indent=2, ensure_ascii=False)
                        st.download_button(
                            "다운로드",
                            json_str,
                            file_name=f"analysis_{selected_id}.json",
                            mime="application/json"
                        )
                
                with col_save3:
                    if st.button("🗑️ 이 분석 삭제", type="secondary", use_container_width=True):
                        if st.session_state.get('confirm_delete') == selected_id:
                            db.delete_analysis(selected_id)
                            st.success("✅ 삭제 완료!")
                            st.session_state.pop('confirm_delete', None)
                            st.rerun()
                        else:
                            st.session_state['confirm_delete'] = selected_id
                            st.warning("⚠️ 한 번 더 클릭하여 삭제를 확인하세요.")
        
        else:
            st.info("📌 아직 저장된 분석 결과가 없습니다. 이미지 분석을 실행하면 자동으로 저장됩니다.")
        
    except Exception as e:
        st.error(f"데이터베이스 로드 실패: {str(e)}")
        st.info("새로운 분석을 실행하면 데이터베이스가 자동으로 생성됩니다.")
    
    st.markdown("---")
    
    # 실제 데이터 폴더 확인
    data_dirs = {
        '원본 이미지': Path('data/raw'),
        '전처리된 데이터': Path('data/processed'),
        '분석 결과': Path('data/outputs')
    }
    
    st.markdown("### 📁 데이터 저장 현황")
    
    cols = st.columns(3)
    for idx, (name, path) in enumerate(data_dirs.items()):
        with cols[idx]:
            if path.exists():
                files = list(path.glob('*'))
                file_count = len([f for f in files if f.is_file()])
                total_size = sum(f.stat().st_size for f in files if f.is_file()) / (1024 * 1024)  # MB
                
                st.metric(
                    name,
                    f"{file_count} 파일",
                    delta=f"{total_size:.1f} MB"
                )
            else:
                st.metric(name, "0 파일", delta="폴더 없음")
    
    st.markdown("---")
    
    # 데이터 폴더 선택 및 파일 목록
    st.markdown("### 📋 파일 목록")
    selected_folder = st.selectbox(
        "폴더 선택",
        list(data_dirs.keys())
    )
    
    folder_path = data_dirs[selected_folder]
    if folder_path.exists():
        files = sorted([f for f in folder_path.glob('*') if f.is_file()], key=lambda x: x.stat().st_mtime, reverse=True)
        
        if files:
            file_data = []
            for f in files:
                stats = f.stat()
                file_data.append({
                    '파일명': f.name,
                    '크기 (MB)': f"{stats.st_size / (1024*1024):.2f}",
                    '수정일': pd.to_datetime(stats.st_mtime, unit='s').strftime('%Y-%m-%d %H:%M')
                })
            
            st.dataframe(pd.DataFrame(file_data), use_container_width=True)
            
            # 데이터 내보내기 옵션
            st.markdown("### 💾 데이터 내보내기")
            if st.button("🗜️ 전체 데이터 압축 다운로드"):
                st.info("압축 기능은 구현 예정입니다.")
        else:
            st.info(f"'{selected_folder}' 폴더에 파일이 없습니다.")
    else:
        st.info(f"'{selected_folder}' 폴더가 존재하지 않습니다.")


def show_file_system_data():
    """File system data management (original functionality)"""
    # 실제 데이터 폴더 확인
    data_dirs = {
        '원본 이미지': Path('data/raw'),
        '전처리된 데이터': Path('data/processed'),
        '분석 결과': Path('data/outputs')
    }
    
    st.markdown("### 📁 데이터 저장 현황")
    
    cols = st.columns(3)
    for idx, (name, path) in enumerate(data_dirs.items()):
        with cols[idx]:
            if path.exists():
                files = list(path.glob('*'))
                file_count = len([f for f in files if f.is_file()])
                total_size = sum(f.stat().st_size for f in files if f.is_file()) / (1024 * 1024)  # MB
                
                st.metric(
                    name,
                    f"{file_count} 파일",
                    delta=f"{total_size:.1f} MB"
                )
            else:
                st.metric(name, "0 파일", delta="폴더 없음")
    
    st.markdown("---")
    
    # 데이터 폴더 선택 및 파일 목록
    st.markdown("### 📋 파일 목록")
    selected_folder = st.selectbox(
        "폴더 선택",
        list(data_dirs.keys())
    )
    
    folder_path = data_dirs[selected_folder]
    if folder_path.exists():
        files = sorted([f for f in folder_path.glob('*') if f.is_file()], key=lambda x: x.stat().st_mtime, reverse=True)
        
        if files:
            file_data = []
            for f in files:
                stats = f.stat()
                file_data.append({
                    '파일명': f.name,
                    '크기 (MB)': f"{stats.st_size / (1024*1024):.2f}",
                    '수정일': pd.to_datetime(stats.st_mtime, unit='s').strftime('%Y-%m-%d %H:%M')
                })
            
            st.dataframe(pd.DataFrame(file_data), use_container_width=True)
            
            # 데이터 내보내기 옵션
            st.markdown("### 💾 데이터 내보내기")
            if st.button("🗜️ 전체 데이터 압축 다운로드"):
                st.info("압축 기능은 구현 예정입니다.")
        else:
            st.info(f"'{selected_folder}' 폴더에 파일이 없습니다.")
    else:
        st.info(f"'{selected_folder}' 폴더가 존재하지 않습니다.")
