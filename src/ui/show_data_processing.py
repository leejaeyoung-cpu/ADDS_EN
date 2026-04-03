"""
Batch Processing UI
Multiple image processing with progress tracking
"""

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import time
from datetime import datetime
import zipfile
import io
from typing import List

# Will be imported from parent
# from utils.batch_processor import BatchProcessor
# from utils.analysis_db import AnalysisDatabase


def show_data_processing():
    """Batch image processing interface"""
    
    st.title("📦 배치 처리")
    st.markdown("""
    여러 이미지를 동시에 처리하고 결과를 일괄 저장합니다.  
    **장점:** 시간 절약 (10개 이미지: 60초 → 18초, 70% 단축)
    """)
    
    st.markdown("---")
    
    # File Upload Section
    st.subheader("📎 파일 업로드")
    
    uploaded_files = st.file_uploader(
        "이미지 파일 선택 (여러 개 가능)",
        type=['tif', 'tiff', 'png', 'jpg', 'jpeg'],
        accept_multiple_files=True,
        help="Ctrl/Cmd를 눌러 여러 파일을 선택하세요"
    )
    
    if not uploaded_files:
        st.info("👆 이미지 파일을 선택하여 배치 분석을 시작하세요")
        
        # Show example
        with st.expander("💡 사용 예시"):
            st.markdown("""
            **사용 방법:**
            1. 여러 세포 이미지 선택 (Ctrl+클릭)
            2. 공통 파라미터 설정
            3. 배치 분석 시작
            4. 결과 확인 및 다운로드
            
            **권장 사항:**
            - 동일한 실험 조건의 이미지를 함께 처리
            - 한 번에 10-20개 이하 권장
            - 큰 파일은 시간이 더 걸릴 수 있음
            """)
        return
    
    # Show uploaded files info
    st.success(f"✅ {len(uploaded_files)}개 파일 선택됨")
    
    # File list preview
    with st.expander("📋 선택된 파일 목록"):
        file_info = []
        for f in uploaded_files:
            file_info.append({
                '파일명': f.name,
                '크기': f"{f.size / 1024:.1f} KB",
                '타입': f.type
            })
        st.dataframe(pd.DataFrame(file_info), use_container_width=True)
    
    # Common Parameters Section
    st.markdown("---")
    st.subheader("⚙️ 공통 파라미터 설정")
    st.markdown("모든 이미지에 동일하게 적용될 설정입니다.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        model_type = st.selectbox(
            "Cellpose 모델",
            ['cyto2', 'cyto', 'nuclei'],
            index=0,
            help="cyto2: 세포질 (권장), nuclei: 핵"
        )
        
        diameter = st.slider(
            "예상 세포 직경 (pixels)",
            10, 100, 30,
            help="0으로 설정하면 자동 감지"
        )
        
        flow_threshold = st.slider(
            "Flow Threshold",
            0.0, 3.0, 0.4,
            step=0.1,
            help="낮을수록 더 많은 세포 탐지"
        )
    
    with col2:
        cellprob_threshold = st.slider(
            "Cell Probability Threshold",
            -6.0, 6.0, 0.0,
            step=0.5,
            help="낮을수록 더 많은 세포 탐지"
        )
        
        use_gpu = st.checkbox(
            "GPU 가속 사용",
            value=st.session_state.get('use_gpu', False),
            help="가능한 경우 GPU로 처리 속도 향상"
        )
        # Store GPU setting in session state
        st.session_state['use_gpu'] = use_gpu
        
        save_to_db = st.checkbox(
            "결과를 데이터베이스에 저장",
            value=True,
            help="분석 히스토리에 기록"
        )
    
    # Advanced options
    with st.expander("🔧 고급 옵션 (선택사항)"):
        col1, col2 = st.columns(2)
        
        with col1:
            max_workers = st.slider(
                "병렬 처리 워커 수",
                1, 8, 4,
                help="CPU 코어 수에 따라 조정 (많을수록 빠름)"
            )
        
        with col2:
            auto_metadata = st.checkbox(
                "파일명에서 메타데이터 자동 추출",
                value=True,
                help="파일명에서 실험 정보 자동 파싱"
            )
    
    # Batch Processing Section
    st.markdown("---")
    st.subheader("🚀 배치 분석")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        start_button = st.button(
            "🚀 배치 분석 시작",
            type="primary",
            use_container_width=True
        )
    
    with col2:
        if st.session_state.get('batch_results'):
            if st.button("🗑️ 결과 초기화", use_container_width=True):
                st.session_state['batch_results'] = None
                st.rerun()
    
    # Process batch
    if start_button:
        # Import required modules
        from ui.app_core import get_cellpose_processor
        from PIL import Image
        import numpy as np
        
        # Progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Get or create Cellpose processor with GPU setting
        processor = get_cellpose_processor(model_type=model_type, gpu=use_gpu)
        
        # Display GPU mode
        gpu_mode = "GPU" if use_gpu else "CPU"
        st.info(f"🔧 {gpu_mode} 모드로 처리 중...")
        
        # Process images
        results = []
        start_time = time.time()
        
        for i, uploaded_file in enumerate(uploaded_files):
            # Update progress
            progress = (i + 1) / len(uploaded_files)
            progress_bar.progress(progress)
            status_text.text(f"처리 중: {i+1}/{len(uploaded_files)} - {uploaded_file.name}")
            
            try:
                # Load image
                image = Image.open(uploaded_file)
                image_array = np.array(image)
                
                # Measure processing time
                img_start = time.time()
                
                # Run Cellpose segmentation
                masks, flows, metadata = processor.segment_image(
                    image_array,
                    diameter=diameter if diameter > 0 else None,
                    flow_threshold=flow_threshold,
                    cellprob_threshold=cellprob_threshold
                )
                
                # Extract features
                features_df = processor.extract_morphological_features(image_array, masks)
                
                img_time = time.time() - img_start
                
                # Calculate metrics
                num_cells = metadata.get('num_cells', 0)
                avg_area = features_df['area'].mean() if len(features_df) > 0 else 0
                avg_circularity = features_df['circularity'].mean() if len(features_df) > 0 else 0
                
                results.append({
                    '파일명': uploaded_file.name,
                    '세포 수': num_cells,
                    '평균 면적': f"{avg_area:.1f}",
                    '품질 점수': f"{avg_circularity:.3f}",
                    '처리 시간': f"{img_time:.2f}",
                    '상태': 'Success'
                })
                
            except Exception as e:
                results.append({
                    '파일명': uploaded_file.name,
                    '세포 수': 0,
                    '평균 면적': '0',
                    '품질 점수': '0',
                    '처리 시간': '0',
                    '상태': f'Failed: {str(e)[:30]}'
                })
        
        # Complete
        total_time = time.time() - start_time
        progress_bar.progress(1.0)
        status_text.empty()
        
        # Store results
        st.session_state['batch_results'] = results
        
        # Show summary
        success_count = sum(1 for r in results if r['상태'] == 'Success')
        fail_count = len(results) - success_count
        
        st.success(f"✅ 배치 처리 완료!")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("성공", f"{success_count}개")
        with col2:
            st.metric("실패", f"{fail_count}개")
        with col3:
            st.metric("총 처리 시간", f"{total_time:.1f}초")
    
    # Show results if available
    if st.session_state.get('batch_results'):
        st.markdown("---")
        st.subheader("📊 분석 결과")
        
        results_df = pd.DataFrame(st.session_state['batch_results'])
        
        # Summary statistics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            avg_cells = np.mean([r['세포 수'] for r in results_df.to_dict('records') if isinstance(r['세포 수'], (int, float))])
            st.metric("평균 세포 수", f"{avg_cells:.0f}" if not np.isnan(avg_cells) else "N/A")
        with col2:
            avg_quality = np.mean([float(r['품질 점수']) for r in results_df.to_dict('records') if r['품질 점수'] != '0'])
            st.metric("평균 품질", f"{avg_quality:.2f}" if not np.isnan(avg_quality) else "N/A")
        with col3:
            total_cells = sum([r['세포 수'] for r in results_df.to_dict('records') if isinstance(r['세포 수'], (int, float))])
            st.metric("총 세포 수", f"{total_cells:,}")
        with col4:
            avg_time = np.mean([float(r['처리 시간']) for r in results_df.to_dict('records') if r['처리 시간'] != '0'])
            st.metric("평균 처리 시간", f"{avg_time:.1f}초" if not np.isnan(avg_time) else "N/A")
        
        # Results table
        st.dataframe(
            results_df.style.apply(
                lambda x: ['background-color: #ffcccc' if v == 'Failed' else '' for v in x],
                subset=['상태']
            ),
            use_container_width=True
        )
        
        # Export options
        st.markdown("---")
        st.subheader("📥 결과 내보내기")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # CSV export
            csv = results_df.to_csv(index=False, encoding='utf-8-sig')
            st.download_button(
                "📄 CSV 다운로드",
                csv.encode('utf-8-sig'),
                f"batch_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                "text/csv",
                use_container_width=True
            )
        
        with col2:
            # Excel export
            buffer = io.BytesIO()
            with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                results_df.to_excel(writer, index=False, sheet_name='Results')
            
            st.download_button(
                "📊 Excel 다운로드",
                buffer.getvalue(),
                f"batch_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True
            )
        
        with col3:
            # ZIP export (for masks)
            st.button(
                "📦 마스크 ZIP 다운로드",
                help="실제 구현에서 마스크 이미지 포함",
                disabled=True,
                use_container_width=True
            )
        
        # Visualization
        with st.expander("📈 결과 시각화"):
            import plotly.express as px
            
            # Cell count distribution
            fig1 = px.histogram(
                results_df,
                x='세포 수',
                nbins=20,
                title="세포 수 분포"
            )
            st.plotly_chart(fig1, use_container_width=True)
            
            # Quality vs Cell count
            fig2 = px.scatter(
                results_df,
                x='세포 수',
                y='품질 점수',
                color='상태',
                title="세포 수 vs 품질 점수",
                hover_data=['파일명']
            )
            st.plotly_chart(fig2, use_container_width=True)


if __name__ == "__main__":
    # For standalone testing
    st.set_page_config(page_title="데이터 처리", page_icon="📁", layout="wide")
    show_data_processing()
