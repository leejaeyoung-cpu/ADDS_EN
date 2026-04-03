"""
ADDS Image Analysis Page  
Cellpose-based cell segmentation with comprehensive visualization
"""

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import os
from datetime import datetime
from PIL import Image, ImageDraw
import json
import plotly.express as px

# Add src to path  
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from preprocessing.image_processor import CellposeProcessor
from ui.app_core import get_cellpose_processor
from utils.ai_analyzer import generate_comprehensive_insights
from utils.analysis_db import AnalysisDatabase
from utils.filename_parser import parse_filename_metadata, format_metadata_preview
from ui.components.interactive_cell_viewer import InteractiveCellViewer

def show_image_analysis():
    """Image analysis page"""
    st.header("🔬 이미지 분석 - 병리/CT/MRI 통합 분석")
    st.info("💡 병리 이미지는 Cellpose 기반 세포 분할, CT/MRI는 의료 영상 전문 분석을 제공합니다")
    
    tab1, tab2, tab3, tab4 = st.tabs([
        "🧫 병리 이미지 분석",
        "🏥 CT/MRI 영상 분석", 
        "📦 배치 처리",
        "📊 분석 인사이트"
    ])
    
    with tab1:
        st.markdown("### 이미지 업로드 및 분석")
        
        uploaded_file = st.file_uploader(
            "세포 이미지를 업로드하세요",
            type=['tif', 'tiff', 'png', 'jpg', 'jpeg'],
            help="지원 형식: TIFF, PNG, JPEG"
        )
        
        col1, col2 = st.columns(2)
        
        with col1:
            model_type = st.selectbox(
                "Cellpose 모델",
                ["cyto", "cyto2", "nuclei"],
                index=1
            )
        
        with col2:
            diameter = st.slider(
                "예상 세포 직경 (pixels)",
                min_value=10,
                max_value=200,
                value=30,
                help="자동 감지하려면 None"
            )
        
        st.markdown("---")
        
        # Experimental condition input
        st.markdown("### 📋 실험 정보 (선택사항)")
        
        # Auto-parse filename if file is uploaded
        if uploaded_file:
            parsed_meta = parse_filename_metadata(uploaded_file.name)
            st.info(f"📄 파일명에서 자동 인식: {format_metadata_preview(parsed_meta)}")
            
            # Initialize session state with parsed values if not already set
            if 'auto_parsed' not in st.session_state or st.session_state.get('last_filename') != uploaded_file.name:
                if parsed_meta.get('cell_line'):
                    st.session_state['cell_line'] = parsed_meta['cell_line']
                if parsed_meta.get('treatment'):
                    st.session_state['treatment'] = parsed_meta['treatment']
                if parsed_meta.get('condition'):
                    st.session_state['parsed_condition'] = parsed_meta['condition']
                if parsed_meta.get('replicate_number'):
                    st.session_state['replicate_number'] = parsed_meta['replicate_number']
                
                st.session_state['auto_parsed'] = True
                st.session_state['last_filename'] = uploaded_file.name
        
        with st.expander("⚙️ 실험 조건 입력/수정", expanded=False):
            exp_col1, exp_col2 = st.columns(2)
            
            with exp_col1:
                experiment_name = st.text_input(
                    "실험 이름",
                    value=st.session_state.get('experiment_name', ''),
                    placeholder="예: HUVEC_TNFa_TimeCourse",
                    key="exp_name_input"
                )
                
                cell_line = st.text_input(
                    "세포주",
                    value=st.session_state.get('cell_line', 'HUVEC'),
                    placeholder="예: HUVEC",
                    key="cell_line_input"
                )
                
                treatment = st.text_input(
                    "처리 약물",
                    value=st.session_state.get('treatment', ''),
                    placeholder="예: TNF-α",
                    key="treatment_input"
                )
            
            with exp_col2:
                concentration = st.text_input(
                    "농도",
                    value=st.session_state.get('concentration', ''),
                    placeholder="예: 10 ng/mL",
                    key="concentration_input"
                )
                
                # Get parsed condition if available
                parsed_cond = st.session_state.get('parsed_condition')
                time_conditions = ["None", "Control", "6hr", "12hr", "24hr", "48hr", "Custom"]
                
                # Set default index based on parsed condition
                default_idx = 0
                if parsed_cond and parsed_cond in time_conditions:
                    default_idx = time_conditions.index(parsed_cond)
                
                condition = st.selectbox(
                    "실험 조건",
                    options=time_conditions,
                    index=default_idx,
                    key="condition_select"
                )
                
                if condition == "Custom":
                    condition = st.text_input("조건 직접 입력", placeholder="예: 72hr", key="custom_condition")
                elif condition == "None":
                    condition = None
                
                # Ensure replicate number is within valid range
                parsed_rep = st.session_state.get('replicate_number', 1)
                if parsed_rep > 10:
                    parsed_rep = 1  # Reset to default if too large
                
                replicate_number = st.number_input(
                    "반복 번호",
                    min_value=1,
                    max_value=10,
                    value=parsed_rep,
                    key="replicate_input"
                )
        
        st.markdown("---")
        
        # 심층 분석 옵션
        deep_analysis = st.checkbox("🔬 심층 분석 활성화 (품질 평가 + PDF 리포트)", value=True)
        
        # AI 플랫폼 비교 옵션
        ai_comparison = st.checkbox(
            "🤖 AI 플랫폼 비교 분석 (GPT-4 Vision)", 
            value=False,
            help="Cellpose 결과를 GPT-4 Vision과 비교하여 정확도 검증 및 하이퍼파라미터 최적화 권장사항 생성 (OpenAI API 키 필요)"
        )
        
        if ai_comparison:
            api_key = os.getenv("OPENAI_API_KEY")
            if api_key:
                st.success("✅ OpenAI API 키가 설정되어 있습니다. AI 비교 분석이 활성화됩니다.")
                # Set environment variable to enable comparison in processor
                os.environ["ENABLE_AI_COMPARISON"] = "true"
            else:
                st.warning("⚠️ OpenAI API 키가 설정되지 않았습니다.")
                st.info("""
                AI 비교 분석을 사용하려면:
                1. `.env` 파일에 `OPENAI_API_KEY=your-api-key` 추가
                2. 또는 환경 변수로 설정
                
                AI 비교 없이 일반 Cellpose 분석만 진행됩니다.
                """)
                os.environ["ENABLE_AI_COMPARISON"] = "false"
        else:
            # Disable AI comparison if checkbox is unchecked
            os.environ["ENABLE_AI_COMPARISON"] = "false"
        
        if uploaded_file and st.button("🔍 분석 시작", type="primary"):
            # Process image
            try:
                # 원본 파일명 저장
                original_filename = uploaded_file.name
                file_stem = Path(original_filename).stem
                
                # 3D 프로세스 시각화
                st.markdown("### 🎯 이미지 분석 파이프라인")
                
                # CSS for arrows and styling
                st.markdown("""
                <style>
                .stage-status {
                    font-size: 11px;
                    margin-top: 5px;
                    font-weight: bold;
                }
                .status-waiting { color: #94a3b8; }
                .status-active { color: #3b82f6; }
                .status-complete { color: #10b981; }
                </style>
                """, unsafe_allow_html=True)
                
                # 5개의 단계 정의
                stages = [
                    {"name": "이미지\n로딩", "icon": "stage_1_loading"},
                    {"name": "전처리", "icon": "stage_2_preprocessing"},
                    {"name": "AI\n세그멘테이션", "icon": "stage_3_segmentation"},
                    {"name": "특징\n추출", "icon": "stage_4_features"},
                    {"name": "품질\n평가", "icon": "stage_5_quality"}
                ]
                
                # 파이프라인 시각화
                cols = st.columns([2, 1, 2, 1, 2, 1, 2, 1, 2])
                
                for idx, stage in enumerate(stages):
                    col_idx = idx * 2
                    
                    with cols[col_idx]:
                        icon_files = list(Path("assets/process_icons").glob(f"{stage['icon']}_*.png"))
                        if icon_files:
                            st.image(str(icon_files[0]), use_container_width=True)
                        
                        st.markdown(f"<p style='text-align: center; font-size: 12px; margin-top: 5px;'>{stage['name']}</p>", 
                                  unsafe_allow_html=True)
                        
                        status_placeholder = st.empty()
                        status_placeholder.markdown(
                            f"<p class='stage-status status-waiting' style='text-align: center;'>⏸ 대기</p>",
                            unsafe_allow_html=True
                        )
                        stages[idx]['status_placeholder'] = status_placeholder
                    
                    if idx < len(stages) - 1:
                        with cols[col_idx + 1]:
                            st.markdown("<p style='text-align: center; font-size: 32px; color: #6366f1; margin-top: 50px;'>→</p>", 
                                      unsafe_allow_html=True)
                
                st.markdown("---")
                
                # 전체 진행률 바
                progress_bar = st.progress(0)
                status_text = st.empty()
                current_stage_text = st.empty()
                
                # 헬퍼 함수
                def update_stage_status(stage_idx, status):
                    if status == "active":
                        stages[stage_idx]['status_placeholder'].markdown(
                            f"<p class='stage-status status-active' style='text-align: center;'>⏳ 진행중</p>",
                            unsafe_allow_html=True
                        )
                    elif status == "complete":
                        stages[stage_idx]['status_placeholder'].markdown(
                            f"<p class='stage-status status-complete' style='text-align: center;'>✓ 완료</p>",
                            unsafe_allow_html=True
                        )
                
                # 1단계 시작
                update_stage_status(0, "active")
                current_stage_text.markdown("**현재 단계:** 1/5 - 이미지 파일 준비")
                status_text.text("파일을 메모리에 로드하고 있습니다...")
                progress_bar.progress(10)
                
                # 원본 파일명 저장
                original_filename = uploaded_file.name
                file_stem = Path(original_filename).stem
                
                # 진행률 표시 컨테이너 생성
                
                # 1단계: 파일 준비 (10%)
                status_text.text("⏳ 1/5: 이미지 파일 준비 중...")
                progress_bar.progress(10)
                
                # 출력 디렉토리 - 원본 파일명 기반 서브폴더 생성
                output_dir = Path("data/outputs") / file_stem
                output_dir.mkdir(parents=True, exist_ok=True)
                
                # 업로드된 파일을 output 디렉토리에 직접 저장 (임시 파일 문제 방지)
                file_ext = Path(original_filename).suffix or '.png'
                saved_image_path = output_dir / f"uploaded_{file_stem}{file_ext}"
                with open(saved_image_path, 'wb') as f:
                    f.write(uploaded_file.getvalue())
                tmp_path = saved_image_path
                
                # 2단계: 모델 로딩 (20%)
                status_text.text("⏳ 2/5: Cellpose 모델 로딩 중...")
                progress_bar.progress(20)
                
                # GPU 설정 가져오기
                use_gpu = st.session_state.get('use_gpu', False)
                
                # 캐시된 processor 사용
                # 1단계 완료
                update_stage_status(0, "complete")
                
                # 2단계: 모델 로딩
                update_stage_status(1, "active")
                current_stage_text.markdown("**현재 단계:** 2/5 - Cellpose 모델 로딩")
                status_text.text("AI 세그멘테이션 모델을 초기화하고 있습니다...")
                progress_bar.progress(20)
                
                processor = get_cellpose_processor(model_type=model_type, gpu=use_gpu)
                
                # 3단계: 세포 세그멘테이션 (40%)
                status_text.text("⏳ 3/5: 세포 세그멘테이션 실행 중...")
                progress_bar.progress(40)
                
                # 심층 분석 실행
                # 2단계 완료
                update_stage_status(1, "complete")
                
                # 3단계: 세포 세그멘테이션
                update_stage_status(2, "active")
                current_stage_text.markdown("**현재 단계:** 3/5 - AI 세포 세그멘테이션")
                status_text.text("딥러닝 모델을 사용하여 세포를 식별하고 있습니다...")
                progress_bar.progress(40)
                
                results = processor.process_and_save(
                    tmp_path,
                    output_dir,
                    deep_analysis=deep_analysis,
                    save_metadata=deep_analysis,
                    save_report=deep_analysis,
                    save_visualization=True
                )
                
                # 4단계: 특징 추출 완료 (70%)
                status_text.text("⏳ 4/5: 세포 특징 추출 및 저장 중...")
                progress_bar.progress(70)
                
                # 5단계: 리포트 생성 완료 (90%)
                if deep_analysis:
                    status_text.text("⏳ 5/5: PDF 리포트 생성 중...")
                    progress_bar.progress(90)
                
                # 완료 (100%)
                progress_bar.progress(100)
                status_text.empty()  # 상태 텍스트 제거
                
                # Store results in session state for insights tab
                st.session_state['analysis_results'] = results
                
                # Auto-save to database
                try:
                    db = AnalysisDatabase()
                    
                    # Create a clean copy for database (remove large binary data)
                    db_results = {k: v for k, v in results.items() 
                                  if k not in ['image', 'masks', 'flows']}
                    
                    record_id = db.save_analysis(
                        db_results, 
                        uploaded_file.name if uploaded_file else "Unknown",
                        experiment_name=experiment_name if experiment_name else None,
                        cell_line=cell_line if cell_line else None,
                        treatment=treatment if treatment else None,
                        concentration=concentration if concentration else None,
                        condition=condition,
                        replicate_number=replicate_number
                    )
                    st.session_state['last_saved_id'] = record_id
                except Exception as e:
                    st.warning(f"데이터베이스 저장 실패: {str(e)}")
                
                # 세그멘테이션 메타데이터
                seg_metadata = results.get('segmentation_metadata', {})
                metrics = results.get('metrics', {})
                
                # 성공 메시지
                st.success(f"✓ 분석 완료: {seg_metadata.get('num_cells', 0)} 개의 세포 감지")
                st.info("💡 '📊 분석 인사이트' 탭에서 상세한 해석을 확인하세요!")
                if 'last_saved_id' in st.session_state:
                    st.info(f"💾 결과가 자동 저장되었습니다 (ID: {st.session_state['last_saved_id']})")

                    
                    
                # 시각화 갤러리 (6개 이미지: 2행 x 3열)
                st.markdown("---")
                st.markdown("### 🖼️ 분석 결과 시각화")
                
                # 이미지 경로 매핑
                image_configs = [
                    {"path": results.get('original_path'), "title": "① 원본 이미지", "desc": "업로드된 원본"},
                    {"path": results.get('preprocessed_path'), "title": "② 전처리", "desc": "CLAHE 적용"},
                    {"path": results.get('colored_mask_path'), "title": "③ 세그멘테이션 마스크", "desc": "컬러 라벨링"},
                    {"path": results.get('overlay_path'), "title": "④ 오버레이", "desc": "원본 + 마스크"},
                    {"path": results.get('contour_path'), "title": "⑤ 컨투어", "desc": "경계선 표시"},
                    {"path": results.get('heatmap_path'), "title": "⑥ 히트맵", "desc": "세포 크기 분포"}
                ]
                
                # 2행으로 구성
                for row in range(2):
                    cols = st.columns(3)
                    for col_idx in range(3):
                        img_idx = row * 3 + col_idx
                        if img_idx < len(image_configs):
                            config = image_configs[img_idx]
                            with cols[col_idx]:
                                if config['path'] and Path(config['path']).exists():
                                    st.markdown(f"**{config['title']}**")
                                    st.image(str(config['path']), use_container_width=True)
                                    st.caption(config['desc'])
                                else:
                                    st.info(f"{config['title']}: 생성 중...")
                
                    
                
                # === 주요 메트릭 - 가로 4개 배치 ===
                st.markdown("### 📊 분석 결과")
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    cell_count = seg_metadata.get('num_cells', 0)
                    st.metric(
                        "세포 수",
                        f"{cell_count:,}",
                        help="검출된 총 세포 수"
                    )
                
                with col2:
                    metrics = results.get('metrics', {})
                    avg_area = metrics.get('mean_area', 0)
                    st.metric(
                        "평균 면적",
                        f"{avg_area:.1f} px²",
                        help="세포 평균 면적"
                    )
                
                with col3:
                    circularity = metrics.get('mean_circularity', 0)
                    st.metric(
                        "평균 원형도",
                        f"{circularity:.2f}",
                        help="1.0에 가까울수록 원형"
                    )
                
                with col4:
                    quality = results.get("quality_assessment", {})
                    score = quality.get('overall_score', 0)
                    st.metric(
                        "품질 점수",
                        f"{score:.2f}",
                        delta=quality.get('overall_quality', 'N/A'),
                        help="이미지 품질 종합 평가"
                    )
                
                st.markdown("---")
                
                # === 상세 품질 평가 ===
                st.markdown("### 🎯 품질 평가 상세")
                quality = results.get("quality_assessment", {})
                        
                # 상세 평가
                st.markdown("#### 상세 품질 지표")
                detailed = quality.get('detailed_assessment', {})
                        
                quality_data = []
                for metric, values in detailed.items():
                    quality_data.append({
                        '항목': metric.replace('_', ' ').title(),
                        '값': str(values.get('score', values.get('value', values.get('snr', 'N/A')))),
                        '평가': values.get('quality', 'N/A')
                    })
                        
                if quality_data:
                    st.dataframe(quality_data, use_container_width=True)
                
                # 메타데이터 뷰어 추가 (화면에 표시)
                if 'metadata_path' in results:
                    metadata_path = Path(results['metadata_path'])
                    if metadata_path.exists():
                        st.markdown("---")
                        st.markdown("### 📋 메타데이터 뷰어")
                        
                        with st.expander("🔍 메타데이터 보기", expanded=False):
                            try:
                                import json
                                with open(metadata_path, 'r', encoding='utf-8') as f:
                                    metadata_content = json.load(f)
                                
                                # 메타데이터를 보기 좋게 표시
                                st.json(metadata_content)
                                
                                # 주요 정보 요약 표시
                                if isinstance(metadata_content, dict):
                                    st.markdown("#### 📊 주요 정보 요약")
                                    
                                    summary_cols = st.columns(3)
                                    
                                    # 작성자 정보
                                    if 'author' in metadata_content:
                                        with summary_cols[0]:
                                            author = metadata_content['author']
                                            st.write(f"**작성자**: {author.get('author_name', 'N/A')}")
                                            st.write(f"**소속**: {author.get('author_department', 'N/A')}")
                                    
                                    # 분석 정보
                                    if 'analysis_type' in metadata_content:
                                        with summary_cols[1]:
                                            st.write(f"**분석 유형**: {metadata_content.get('analysis_type', 'N/A')}")
                                            st.write(f"**생성 시간**: {metadata_content.get('author', {}).get('timestamp', 'N/A')}")
                                    
                                    # 세그멘테이션 정보
                                    if 'segmentation_metadata' in metadata_content:
                                        with summary_cols[2]:
                                            seg_meta = metadata_content['segmentation_metadata']
                                            st.write(f"**검출 세포 수**: {seg_meta.get('num_cells', 'N/A')}")
                                            st.write(f"**모델**: {seg_meta.get('model_type', 'N/A')}")
                                
                            except Exception as e:
                                st.error(f"메타데이터 로드 실패: {str(e)}")
                        
                # 다운로드 섹션 - 가로 배치
                if 'report_path' in results or 'metadata_path' in results:
                    st.markdown("---")
                    st.markdown("### 📥 다운로드")
                            
                    dl_col1, dl_col2, dl_col3 = st.columns(3)
                            
                    with dl_col1:
                        if 'report_path' in results:
                                    report_path = Path(results['report_path'])
                                    if report_path.exists():
                                        with open(report_path, 'rb') as f:
                                            st.download_button(
                                                label="📄 PDF 리포트",
                                                data=f,
                                                file_name=report_path.name,
                                                mime="application/pdf",
                                                use_container_width=True
                                            )
                            
                    with dl_col2:
                        if 'metadata_path' in results:
                                    metadata_path = Path(results['metadata_path'])
                                    if metadata_path.exists():
                                        with open(metadata_path, 'r') as f:
                                            st.download_button(
                                                label="📊 메타데이터 (JSON)",
                                                data=f.read(),
                                                file_name=metadata_path.name,
                                                mime="application/json",
                                                use_container_width=True
                                            )
                            
                    with dl_col3:
                        if 'features_path' in results:
                                    features_path = Path(results['features_path'])
                                    if features_path.exists():
                                        with open(features_path, 'r') as f:
                                            st.download_button(
                                                label="📈 특징 데이터 (CSV)",
                                                data=f.read(),
                                                file_name=features_path.name,
                                                mime="text/csv",
                                                use_container_width=True
                                            )
                        # 임시 파일 삭제
                # 임시 파일 삭제
                tmp_path.unlink()
            except Exception as e:
                st.error(f"오류 발생: {str(e)}")
                import traceback
                st.code(traceback.format_exc())
    
    
    with tab2:
        st.markdown("### 🏥 CT 종양 검출 및 영상 분석")
        
        # Sub-tabs for different analysis types
        ct_subtab1, ct_subtab2, ct_subtab3 = st.tabs([
            "📊 CT 데이터 분석",
            "🔍 종양 검출 (Batch)",
            "🖼️ 개별 영상 분석"
        ])
        
        # === SUB-TAB 1: CT 데이터 품질 분석 ===
        with ct_subtab1:
            st.markdown("#### CT 데이터 품질 분석")
            
            st.info("""
            **📁 데이터 위치**: `CTdata` 또는 `CTdata_cleaned`
            
            현재 분석된 데이터:
            - 총 슬라이스: 382개 (정제 후)
            - 해상도: 512×512
            - 슬라이스 범위: 10004 ~ 10425
            """)
            
            # 분석 리포트 로드
            report_path = Path("ct_data_analysis_report.json")
            if report_path.exists():
                with open(report_path, 'r', encoding='utf-8') as f:
                    report = json.load(f)
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric(
                        "총 파일",
                        f"{report.get('total_files', 0)} 개"
                    )
                
                with col2:
                    st.metric(
                        "정상 이미지",
                        f"{report.get('valid_images', 0)} 개"
                    )
                
                with col3:
                    avg_size = report.get('file_sizes', {}).get('mean', 0) / 1024
                    st.metric(
                        "평균 파일 크기",
                        f"{avg_size:.1f} KB"
                    )
                
                with st.expander("📄 상세 리포트 보기"):
                    st.json(report)
            else:
                st.warning("분석 리포트를 찾을 수 없습니다. 먼저 데이터 분석을 실행해주세요.")
                
                if st.button("데이터 분석 실행", key="ct_data_analysis_btn"):
                    with st.spinner("CT 데이터 분석 중..."):
                        import subprocess
                        result = subprocess.run(
                            ["python", "analyze_ct_data.py"],
                            capture_output=True,
                            text=True
                        )
                        if result.returncode == 0:
                            st.success("✅ 분석 완료!")
                            st.rerun()
                        else:
                            st.error(f"분석 실패: {result.stderr}")
        
        # === SUB-TAB 2: 종양 검출 (Batch Processing) ===
        with ct_subtab2:
            st.markdown("#### 🔍 CT 종양 검출 (배치 처리)")
            
            # 데이터 소스 선택
            data_source = st.radio(
                "데이터 소스 선택",
                ["📁 기존 데이터 폴더 사용", "📤 CT 이미지 업로드"],
                help="기존 데이터 또는 새로 업로드한 이미지를 분석할 수 있습니다",
                key="ct_data_source"
            )
            
            data_path = None
            uploaded_ct_files = None
            
            if data_source == "📁 기존 데이터 폴더 사용":
                # 데이터 경로 선택
                data_options = {
                    "정제된 데이터 (CTdata_cleaned)": "CTdata_cleaned",
                    "원본 데이터 (CTdata)": "CTdata"
                }
                
                selected_data = st.selectbox(
                    "분석할 데이터 선택",
                    list(data_options.keys()),
                    key="ct_data_select"
                )
                
                data_path = data_options[selected_data]
            
            else:  # 업로드
                st.markdown("**CT 이미지 업로드**")
                uploaded_ct_files = st.file_uploader(
                    "CT 슬라이스 이미지 업로드 (여러 파일 선택 가능)",
                    type=['jpg', 'jpeg', 'png', 'dcm'],
                    accept_multiple_files=True,
                    key='ct_batch_upload',
                    help="DICOM 또는 일반 이미지 형식 지원"
                )
                
                if uploaded_ct_files:
                    st.success(f"✓ {len(uploaded_ct_files)}개 파일 업로드됨")
                else:
                    st.info("⬆️ CT 이미지를 업로드하세요")
            
            # 검출 모드 선택
            detection_mode = st.radio(
                "검출 모드",
                ["Mock 모드 (임계값 기반)", "SOTA 모드 (딥러닝 - 학습 중)"],
                help="Mock 모드는 빠르지만 정확도가 제한적입니다.",
                key="tumor_detection_mode"
            )
            
            if "SOTA" in detection_mode:
                st.warning("⚠️ SOTA 모드는 현재 모델 학습 중입니다 (Epoch 8/100). Mock 모드를 사용해주세요.")
                st.info("💡 학습 진행 상황: `check_training_progress.py`로 확인 가능")
                # Don't return - allow continuing to view existing results
            
            # 검출 모드 선택
            with st.expander("⚙️ 고급 설정"):
                col1, col2 = st.columns(2)
                
                with col1:
                    slice_thickness = st.number_input(
                        "슬라이스 간격 (mm)",
                        min_value=0.1,
                        max_value=10.0,
                        value=1.0,
                        step=0.1,
                        key="slice_thickness_input"
                    )
                
                with col2:
                    pixel_spacing = st.number_input(
                        "픽셀 간격 (mm)",
                        min_value=0.1,
                        max_value=5.0,
                        value=1.0,
                        step=0.1,
                        key="pixel_spacing_input"
                    )
            
            # 검출 실행
            can_start = False
            if data_source == "📁 기존 데이터 폴더 사용":
                can_start = data_path and Path(data_path).exists()
                if not can_start and data_path:
                    st.error(f"❌ 데이터를 찾을 수 없습니다: {data_path}")
            else:
                can_start = uploaded_ct_files is not None and len(uploaded_ct_files) > 0
            
            if "SOTA" not in detection_mode and st.button("🚀 종양 검출 시작", type="primary", key="start_tumor_detection", disabled=not can_start):
                
                if data_source == "📁 기존 데이터 폴더 사용":
                    # 기존 폴더 데이터 처리
                    with st.spinner("종양 검출 중... (수 분 소요될 수 있습니다)"):
                        try:
                            import subprocess
                            
                            # 분석 실행
                            result = subprocess.run([
                                "python",
                                "analyze_tumor_location.py"
                            ], capture_output=True, text=True, timeout=600)
                            
                            if result.returncode == 0:
                                st.success("✅ 종양 검출 완료!")
                                
                                # 결과 로드
                                result_path = Path("tumor_analysis_results/tumor_analysis_report.json")
                                if result_path.exists():
                                    with open(result_path, 'r', encoding='utf-8') as f:
                                        results = json.load(f)
                                    
                                    # ===== 품질 자동 평가 =====
                                    try:
                                        import sys
                                        sys.path.insert(0, str(Path(__file__).parent.parent.parent))
                                        from medical_imaging.validation.quality_metrics import DetectionQualityMetrics
                                        
                                        metrics_evaluator = DetectionQualityMetrics()
                                        quality_report = metrics_evaluator.assess_results(results)
                                        
                                        # 품질 점수 표시
                                        st.markdown("#### 🎯 검출 품질 평가")
                                        qual_col1, qual_col2, qual_col3, qual_col4 = st.columns(4)
                                        
                                        with qual_col1:
                                            st.metric("품질 점수", f"{quality_report['score']}/100")
                                        
                                        with qual_col2:
                                            st.metric("품질 등급", quality_report['quality'])
                                        
                                        with qual_col3:
                                            status_emoji = "✅" if quality_report['status'] == 'pass' else "❌"
                                            st.metric("평가 결과", f"{status_emoji} {quality_report['status'].upper()}")
                                        
                                        with qual_col4:
                                            st.metric("경고 수", len(quality_report['warnings']))
                                        
                                        # 경고 메시지
                                        if quality_report['warnings']:
                                            st.warning("⚠️ 검출 품질 경고:")
                                            for warning in quality_report['warnings']:
                                                st.write(f"- {warning}")
                                            
                                            if quality_report['recommendations']:
                                                with st.expander("💡 개선 권장사항"):
                                                    for rec in quality_report['recommendations']:
                                                        st.write(f"- {rec}")
                                        
                                        st.session_state['quality_report'] = quality_report
                                    
                                    except Exception as e:
                                        st.warning(f"품질 평가 로드 중 오류 (검출은 성공): {str(e)}")
                                    
                                    st.markdown("---")
                                    
                                    # 결과 요약 표시
                                    summary = results.get('summary', {})
                                    
                                    st.subheader("📊 검출 결과 요약")
                                    
                                    col1, col2, col3, col4 = st.columns(4)
                                    
                                    with col1:
                                        st.metric(
                                            "총 슬라이스",
                                            f"{results.get('total_slices', 0)} 개"
                                        )
                                    
                                    with col2:
                                        st.metric(
                                            "종양 검출 슬라이스",
                                            f"{summary.get('affected_slices', 0)} 개"
                                        )
                                    
                                    with col3:
                                        st.metric(
                                            "검출된 영역",
                                            f"{summary.get('total_tumor_regions', 0)} 개"
                                        )
                                    
                                    with col4:
                                        volume = summary.get('total_volume_ml', 0)
                                        st.metric(
                                            "추정 부피",
                                            f"{volume:.2f} mL"
                                        )
                                    
                                    # 3D 바운딩 박스
                                    bbox = summary.get('bounding_box_3d')
                                    if bbox:
                                        st.subheader("📦 3D 바운딩 박스")
                                        
                                        col1, col2, col3 = st.columns(3)
                                        
                                        with col1:
                                            st.write(f"**X**: {bbox['x_min']} ~ {bbox['x_max']} ({bbox['x_max'] - bbox['x_min']} px)")
                                        
                                        with col2:
                                            st.write(f"**Y**: {bbox['y_min']} ~ {bbox['y_max']} ({bbox['y_max'] - bbox['y_min']} px)")
                                        
                                        with col3:
                                            st.write(f"**Z**: {bbox['z_min']} ~ {bbox['z_max']} ({bbox['z_max'] - bbox['z_min']} slices)")
                                    
                                    # 세션 상태에 저장
                                    st.session_state['tumor_detection_results'] = results
                                    
                                    st.info("💡 아래 '결과 시각화' 섹션에서 상세 결과를 확인하세요!")
                                else:
                                    st.warning("결과 파일을 찾을 수 없습니다.")
                            else:
                                st.error(f"검출 실패:\n{result.stderr}")
                                
                        except subprocess.TimeoutExpired:
                            st.error("⏱️ 분석 시간이 초과되었습니다. 데이터 크기를 확인해주세요.")
                        except Exception as e:
                            st.error(f"오류 발생: {str(e)}")
                
                else:  # 업로드된 파일 처리
                    st.info("📤 업로드된 이미지 간단 분석 모드 (Mock)")
                    st.warning("⚠️ 실제 종양 검출을 위해서는 전체 CT 시리즈와 SOTA 모델이 필요합니다.")
                    
                    # 간단한 통계만 제공
                    with st.spinner(f"{len(uploaded_ct_files)}개 이미지 분석 중..."):
                        import time
                        time.sleep(1)  # 시뮬레이션
                        
                        st.success(f"✅ {len(uploaded_ct_files)}개 이미지 확인 완료!")
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("업로드된 슬라이스", f"{len(uploaded_ct_files)} 개")
                        with col2:
                            st.metric("상태", "분석 준비 완료")
                        
                        st.info("""
                        💡 **업로드된 이미지 분석 안내**
                        
                        현재는 간단한 확인만 가능합니다. 정확한 종양 검출을 위해서는:
                        - 전체 CT 시리즈 (100+ 슬라이스)
                        - DICOM 형식 권장
                        - SOTA 모델 사용
                        
                        **개별 영상 상세 분석**은 "🖼️ 개별 영상 분석" 탭을 이용하세요.
                        """)
            
            # 결과 시각화 (세션에 저장된 결과가 있는 경우)
            if 'tumor_detection_results' in st.session_state:
                st.markdown("---")
                st.markdown("### 📈 종양 검출 결과 시각화")
                
                results = st.session_state['tumor_detection_results']
                
                # 슬라이스별 검출 결과 탐색
                st.subheader("🔍 슬라이스별 검출 결과")
                
                affected_slices = sorted(results.get('slices_with_tumors', []))
                
                if affected_slices:
                    # 슬라이스 선택
                    selected_slice = st.select_slider(
                        "슬라이스 선택",
                        options=affected_slices,
                        value=affected_slices[len(affected_slices)//2],
                        key="slice_selector"
                    )
                    
                    if selected_slice:
                        # 검출 이미지 표시
                        img_path = Path(f"tumor_analysis_results/detection_{selected_slice:05d}.jpg")
                        
                        if img_path.exists():
                            from PIL import Image as PILImage
                            img = PILImage.open(img_path)
                            st.image(img, caption=f"슬라이스 #{selected_slice} - 종양 검출 결과", use_container_width=True)
                        else:
                            st.warning("이미지를 찾을 수 없습니다.")
                        
                        # 해당 슬라이스 상세 정보
                        slice_detections = [
                            d for d in results.get('tumor_detections', [])
                            if d.get('slice_num') == selected_slice
                        ]
                        
                        if slice_detections:
                            st.write(f"**검출된 영역**: {len(slice_detections)} 개")
                            
                            with st.expander("상세 정보"):
                                for i, det in enumerate(slice_detections[:5], 1):
                                    st.write(f"**영역 #{i}**")
                                    st.write(f"- 중심: ({det['center'][0]}, {det['center'][1]})")
                                    st.write(f"- 면적: {det['area_pixels']:.0f} px")
                                    st.write(f"- 등가 직경: {det.get('equivalent_diameter', 0):.1f} px")
                    
                    # 3D 분포 시각화
                    st.subheader("🌐 3D 종양 분포")
                    
                    detections = results.get('tumor_detections', [])
                    
                    if detections:
                        # 데이터 준비
                        import pandas as pd
                        centers = [(d['center'][0], d['center'][1], d['slice_num']) for d in detections]
                        areas = [d['area_pixels'] for d in detections]
                        
                        df_3d = pd.DataFrame(centers, columns=['X', 'Y', 'Z'])
                        df_3d['Area'] = areas
                        
                        # 3D 산점도
                        import plotly.express as px
                        fig = px.scatter_3d(
                            df_3d,
                            x='X',
                            y='Y',
                            z='Z',
                            size='Area',
                            color='Area',
                            title='종양 영역 3D 분포',
                            labels={'X': 'X (픽셀)', 'Y': 'Y (픽셀)', 'Z': '슬라이스 번호', 'Area': '면적 (px)'},
                            color_continuous_scale='Reds'
                        )
                        
                        fig.update_layout(height=600)
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # 데이터 다운로드
                    st.subheader("📥 결과 다운로드")
                    
                    if detections:
                        df = pd.DataFrame(detections)
                        
                        # 좌표 분리
                        df['center_x'] = df['center'].apply(lambda x: x[0])
                        df['center_y'] = df['center'].apply(lambda x: x[1])
                        
                        # 다운로드 컬럼 선택
                        download_df = df[[
                            'slice_num', 'center_x', 'center_y',
                            'area_pixels', 'equivalent_diameter'
                        ]].copy()
                        
                        csv = download_df.to_csv(index=False, encoding='utf-8-sig')
                        
                        st.download_button(
                            label="📥 CSV 파일 다운로드",
                            data=csv,
                            file_name="tumor_detection_results.csv",
                            mime="text/csv"
                        )
                else:
                    st.info("검출된 종양이 없습니다.")
        
        # === SUB-TAB 3: 개별 영상 분석 (기존 로직) ===
        with ct_subtab3:
            st.markdown("#### 🖼️ 개별 CT/MRI 영상 분석")
            st.info("의료 영상 전문 분석: DICOM 지원, Tumor segmentation, RECIST 측정, Radiomics")
            
            # CT/MRI 업로드
            upload_col1, upload_col2 = st.columns(2)
            
            with upload_col1:
                st.markdown("**CT 영상**")
                ct_images = st.file_uploader(
                    "CT 이미지 업로드",
                    type=['jpg', 'jpeg', 'png', 'dcm'],
                    accept_multiple_files=True,
                    key='ct_upload_analysis',
                    help="DICOM 또는 일반 이미지 형식"
                )
                
                if ct_images:
                    st.success(f"✓ {len(ct_images)}장 업로드됨")
                    
            with upload_col2:
                st.markdown("**MRI 영상**")
                mri_images = st.file_uploader(
                    "MRI 이미지 업로드",
                    type=['jpg', 'jpeg', 'png', 'dcm'],
                    accept_multiple_files=True,
                    key='mri_upload_analysis',
                    help="T1, T2, FLAIR, DWI 등"
                )
                
                if mri_images:
                    st.success(f"✓ {len(mri_images)}장 업로드됨")
            
            
            # 분석 옵션
            st.markdown("---")
            st.markdown("**분석 옵션**")
            
            analysis_col1, analysis_col2, analysis_col3 = st.columns(3)
            
            with analysis_col1:
                cancer_type = st.selectbox(
                    "암 종류",
                    ["Colorectal", "Lung", "Breast", "Brain", "Liver", "Pancreatic", "Prostate"],
                    index=0,
                    key="individual_cancer_type"
                )
            
            with analysis_col2:
                sequence_type = st.selectbox(
                    "MRI 시퀀스 (MRI인 경우)",
                    ["T1", "T2", "FLAIR", "DWI"],
                    index=1,
                    key="individual_sequence_type"
                )
            
            with analysis_col3:
                use_hybrid = st.checkbox(
                    "🔬 Hybrid Mode",
                    value=False,
                    help="Use nnU-Net + TotalSegmentator for enhanced accuracy (requires models)",
                    key="individual_use_hybrid"
                )
            
            # 분석 실행
            button_label = "🔍 Hybrid 영상 분석" if use_hybrid else "🔍 영상 분석 시작"
            if (ct_images or mri_images) and st.button(button_label, type="primary", key="start_individual_analysis"):
                with st.spinner("의료 영상을 분석 중입니다..."):
                    try:
                        # Hybrid mode 또는 standard mode
                        if use_hybrid and ct_images:
                            try:
                                from src.medical_imaging.ct_analyzer_hybrid import HybridCTAnalyzer
                                
                                st.info("🔬 Hybrid Mode: nnU-Net + TotalSegmentator")
                                
                                # Initialize hybrid analyzer
                                hybrid_analyzer = HybridCTAnalyzer(
                                    use_gpu=st.session_state.get('use_gpu', False)
                                )
                                
                                # Initialize results
                                ct_results = []
                                mri_results = []  # Initialize even if not processing MRI
                                
                                # Process each CT image
                                for idx, ct_img in enumerate(ct_images):
                                    with st.spinner(f"CT {idx+1}/{len(ct_images)} 분석 중..."):
                                        result = hybrid_analyzer.analyze_ct_hybrid(
                                            ct_img,
                                            metadata={},
                                            cancer_type=cancer_type
                                        )
                                        result['filename'] = ct_img.name
                                        ct_results.append(result)
                                
                                st.success(f"✓ Hybrid CT 분석 완료: {len(ct_results)}/{len(ct_images)}")
                                
                            except ImportError:
                                st.error("⚠️ Hybrid mode requires nnU-Net and TotalSegmentator")
                                st.info("Install: pip install nnunetv2 TotalSegmentator")
                                st.stop()
                            except Exception as e:
                                st.error(f"Hybrid mode error: {e}")
                                st.info("Falling back to standard mode...")
                                use_hybrid = False
                        
                        # Standard mode
                        if not use_hybrid:
                            from src.medical_imaging.ct_analyzer import CTAnalyzer
                            from src.medical_imaging.mri_analyzer import MRIAnalyzer
                            from src.processing.batch_processor import BatchImageProcessor
                            from src.medical_imaging.multimodal_fusion import MultimodalFusionEngine
                            
                            batch_processor = BatchImageProcessor(max_workers=4)
                            ct_results = []
                            mri_results = []
                            
                            # CT 분석
                            if ct_images:
                                st.info(f"🏥 CT 이미지 {len(ct_images)}장 분석 중...")
                                ct_analyzer = CTAnalyzer(use_gpu=st.session_state.get('use_gpu', False))
                                
                                ct_results = batch_processor.process_multiple_images(
                                    ct_images,
                                    ct_analyzer.analyze_ct_image,
                                    'ct',
                                    cancer_type=cancer_type
                                )
                                
                                successful_ct = [r for r in ct_results if r.get('status') == 'success']
                                st.success(f"✓ CT 분석 완료: {len(successful_ct)}/{len(ct_images)} 성공")
                        
                        # MRI 분석
                        if mri_images:
                            st.info(f"🧲 MRI 이미지 {len(mri_images)}장 분석 중...")
                            mri_analyzer = MRIAnalyzer(use_gpu=st.session_state.get('use_gpu', False))
                            
                            mri_results = batch_processor.process_multiple_images(
                                mri_images,
                                mri_analyzer.analyze_mri_image,
                                'mri',
                                sequence_type=sequence_type,
                                cancer_type=cancer_type
                            )
                            
                            successful_mri = [r for r in mri_results if r.get('status') == 'success']
                            st.success(f"✓ MRI 분석 완료: {len(successful_mri)}/{len(mri_images)} 성공")
                        
                        # 결과 저장
                        st.session_state['ct_analysis_results'] = ct_results
                        st.session_state['mri_analysis_results'] = mri_results
                        
                        # Multi-modal fusion (둘 다 있을 때)
                        if ct_results and mri_results:
                            st.info("🔗 Multi-modal fusion 분석 중...")
                            fusion_engine = MultimodalFusionEngine()
                            
                            # 첫 번째 성공한 결과만 fusion
                            ct_first = next((r for r in ct_results if r.get('status') == 'success'), None)
                            mri_first = next((r for r in mri_results if r.get('status') == 'success'), None)
                            
                            if ct_first and mri_first:
                                fusion_result = fusion_engine.fuse_multimodal_analysis(
                                    ct_result=ct_first,
                                    mri_result=mri_first
                                )
                                
                                st.session_state['fusion_result'] = fusion_result
                                st.success("✓ Multi-modal fusion 완료")
                        
                        # 결과 표시
                        st.markdown("---")
                        st.markdown("### 📊 분석 결과")
                        
                        result_tab1, result_tab2, result_tab3 = st.tabs(["CT 결과", "MRI 결과", "통합 분석"])
                        
                    except Exception as e:
                        st.error(f"분석 중 오류 발생: {str(e)}")
                        import traceback
                        st.code(traceback.format_exc())
                    
                    # 결과 표시 (try 블록 밖으로 이동)
                    if 'ct_analysis_results' in st.session_state or 'mri_analysis_results' in st.session_state:
                        result_tab1, result_tab2, result_tab3 = st.tabs(["CT 결과", "MRI 결과", "통합 분석"])
                        
                        ct_results = st.session_state.get('ct_analysis_results', [])
                        mri_results = st.session_state.get('mri_analysis_results', [])
                    
                        with result_tab1:
                            if ct_results:
                                for idx, result in enumerate(ct_results):
                                    if result.get('status') == 'success':
                                        with st.expander(f"📄 CT #{idx+1} - {result.get('filename', 'Unknown')}", expanded=(idx==0)):
                                            # 이미지 시각화
                                            st.markdown("#### 🖼️ 종양 검출 시각화")
                                        
                                        img_col1, img_col2 = st.columns(2)
                                        
                                        with img_col1:
                                            st.markdown("**원본 이미지**")
                                            # 원본 이미지 표시 (업로드된 파일에서)
                                            try:
                                                if idx < len(ct_images):
                                                    st.image(ct_images[idx], use_container_width=True, caption="Original CT")
                                            except:
                                                st.info("원본 이미지 표시 불가")
                                        
                                        with img_col2:
                                            st.markdown("**종양 검출 결과**")
                                            # Segmentation overlay 생성 및 표시
                                            try:
                                                import numpy as np
                                                import io
                                                
                                                # 원본 이미지 로드
                                                if idx < len(ct_images):
                                                    ct_images[idx].seek(0)
                                                    original_img = Image.open(ct_images[idx]).convert('RGB')
                                                    
                                                    # Overlay 생성
                                                    overlay = original_img.copy()
                                                    draw = ImageDraw.Draw(overlay, 'RGBA')
                                                    
                                                    # Bounding box 그리기
                                                    bbox = result.get('segmentation', {}).get('tumor_bounding_box', {})
                                                    if bbox and bbox.get('width', 0) > 0:
                                                        x, y, w, h = bbox.get('x', 0), bbox.get('y', 0), bbox.get('width', 0), bbox.get('height', 0)
                                                        
                                                        # 빨간색 반투명 사각형
                                                        draw.rectangle([x, y, x+w, y+h], outline='red', width=3)
                                                        draw.rectangle([x, y, x+w, y+h], fill=(255, 0, 0, 50))
                                                        
                                                        # 라벨 추가
                                                        draw.text((x, y-15), "Tumor", fill='red')
                                                    
                                                    st.image(overlay, use_container_width=True, caption="Tumor Detection")
                                                else:
                                                    st.info("종양 시각화 생성 불가")
                                            except Exception as e:
                                                st.warning(f"시각화 오류: {str(e)}")
                                        
                                        st.markdown("---")
                                        
                                        # 메타데이터
                                        st.markdown("#### 📋 영상 정보")
                                        metadata = result.get('metadata', {})
                                        
                                        meta_col1, meta_col2, meta_col3 = st.columns(3)
                                        with meta_col1:
                                            st.metric("형식", metadata.get('format', 'Unknown'))
                                        with meta_col2:
                                            st.metric("영상 크기", str(metadata.get('image_shape', 'N/A')))
                                        with meta_col3:
                                            st.metric("슬라이스 두께", f"{metadata.get('slice_thickness', 'N/A')} mm")
                                        
                                        # Segmentation 결과
                                        st.markdown("---")
                                        st.markdown("#### 🎯 종양 검출 결과")
                                        
                                        segmentation = result.get('segmentation', {})
                                        seg_col1, seg_col2 = st.columns(2)
                                        
                                        with seg_col1:
                                            tumor_detected = segmentation.get('tumor_detected', False)
                                            if tumor_detected:
                                                st.success("✅ 종양 검출됨")
                                                volume = segmentation.get('tumor_volume_mm3', 0)
                                                st.metric("종양 부피", f"{volume:.2f} mm³")
                                            else:
                                                st.info("종양 미검출")
                                        
                                        with seg_col2:
                                            bbox = segmentation.get('tumor_bounding_box', {})
                                            if bbox:
                                                st.metric("위치 (X, Y)", f"({bbox.get('x', 0)}, {bbox.get('y', 0)})")
                                                st.metric("크기 (W × H)", f"{bbox.get('width', 0)} × {bbox.get('height', 0)} px")
                                        
                                        # RECIST 측정
                                        st.markdown("---")
                                        st.markdown("#### 📏 RECIST 1.1 측정")
                                        
                                        measurements = result.get('measurements', {})
                                        recist_col1, recist_col2, recist_col3 = st.columns(3)
                                        
                                        with recist_col1:
                                            longest = measurements.get('longest_diameter_mm', 0)
                                            st.metric("장축 직경", f"{longest:.2f} mm", 
                                                     help="RECIST 1.1 장축 측정값")
                                        
                                        with recist_col2:
                                            shortest = measurements.get('shortest_diameter_mm', 0)
                                            st.metric("단축 직경", f"{shortest:.2f} mm",
                                                     help="RECIST 1.1 단축 측정값")
                                        
                                        with recist_col3:
                                            quality = measurements.get('measurement_quality', 'Unknown')
                                            st.metric("측정 품질", quality)
                                        
                                        # Radiomics Features
                                        st.markdown("---")
                                        st.markdown("#### 🧬 Radiomics 특징")
                                        
                                        radiomics = result.get('radiomics_features', {})
                                        
                                        if radiomics:
                                            radio_tab1, radio_tab2, radio_tab3 = st.tabs(["형태", "강도", "텍스처"])
                                            
                                            with radio_tab1:
                                                st.markdown("**Shape Features**")
                                                shape_col1, shape_col2 = st.columns(2)
                                                with shape_col1:
                                                    st.metric("부피 (voxels)", radiomics.get('shape_volume_voxels', 0))
                                                with shape_col2:
                                                    st.metric("표면적", f"{radiomics.get('shape_surface_area', 0):.2f}")
                                            
                                            with radio_tab2:
                                                st.markdown("**Intensity Features**")
                                                int_col1, int_col2, int_col3 = st.columns(3)
                                                with int_col1:
                                                    st.metric("평균", f"{radiomics.get('intensity_mean', 0):.2f}")
                                                with int_col2:
                                                    st.metric("표준편차", f"{radiomics.get('intensity_std', 0):.2f}")
                                                with int_col3:
                                                    st.metric("범위", f"{radiomics.get('intensity_range', 0):.2f}")
                                            
                                            with radio_tab3:
                                                st.markdown("**Texture Features**")
                                                tex_col1, tex_col2 = st.columns(2)
                                                with tex_col1:
                                                    st.metric("분산", f"{radiomics.get('texture_variance', 0):.2f}")
                                                with tex_col2:
                                                    st.metric("엔트로피", f"{radiomics.get('texture_entropy', 0):.2f}")
                                        
                                        # AI 해석
                                        st.markdown("---")
                                        st.markdown("#### 🤖 AI 임상 해석")
                                        
                                        ai_interpretation = result.get('ai_interpretation', '')
                                        if ai_interpretation:
                                            st.markdown(ai_interpretation)
                                        else:
                                            st.info("AI 해석을 사용하려면 OpenAI API 키를 설정하세요")
                            else:
                                st.info("CT 이미지를 업로드하고 분석하세요")
                        
                        
                        with result_tab2:
                            if mri_results:
                                for idx, result in enumerate(mri_results):
                                    if result.get('status') == 'success':
                                        with st.expander(f"📄 MRI #{idx+1} - {result.get('filename', 'Unknown')}", expanded=(idx==0)):
                                            # 이미지 시각화
                                            st.markdown("#### 🖼️ 종양/부종 검출 시각화")
                                            
                                            img_col1, img_col2 = st.columns(2)
                                            
                                            with img_col1:
                                                st.markdown("**원본 이미지**")
                                                try:
                                                    if idx < len(mri_images):
                                                        st.image(mri_images[idx], use_container_width=True, caption="Original MRI")
                                                except:
                                                    st.info("원본 이미지 표시 불가")
                                        
                                        with img_col2:
                                            st.markdown("**종양/부종 검출 결과**")
                                            try:
                                                from PIL import Image, ImageDraw
                                                import numpy as np
                                                import io
                                                
                                                # 원본 이미지 로드
                                                if idx < len(mri_images):
                                                    mri_images[idx].seek(0)
                                                    original_img = Image.open(mri_images[idx]).convert('RGB')
                                                    
                                                    # Overlay 생성
                                                    overlay = original_img.copy()
                                                    draw = ImageDraw.Draw(overlay, 'RGBA')
                                                    
                                                    segmentation = result.get('segmentation', {})
                                                    
                                                    # 종양 bounding box 그리기 (빨간색)
                                                    tumor_bbox = segmentation.get('tumor_bounding_box', {})
                                                    if tumor_bbox and tumor_bbox.get('width', 0) > 0:
                                                        x, y, w, h = tumor_bbox.get('x', 0), tumor_bbox.get('y', 0), tumor_bbox.get('width', 0), tumor_bbox.get('height', 0)
                                                        draw.rectangle([x, y, x+w, y+h], outline='red', width=3)
                                                        draw.rectangle([x, y, x+w, y+h], fill=(255, 0, 0, 50))
                                                        draw.text((x, y-15), "Tumor", fill='red')
                                                    
                                                    # 부종 bounding box 그리기 (녹색)
                                                    edema_bbox = segmentation.get('edema_bounding_box', {})
                                                    if edema_bbox and edema_bbox.get('width', 0) > 0:
                                                        x, y, w, h = edema_bbox.get('x', 0), edema_bbox.get('y', 0), edema_bbox.get('width', 0), edema_bbox.get('height', 0)
                                                        draw.rectangle([x, y, x+w, y+h], outline='green', width=2)
                                                        draw.rectangle([x, y, x+w, y+h], fill=(0, 255, 0, 30))
                                                        draw.text((x, y-15), "Edema", fill='green')
                                                    
                                                    st.image(overlay, use_container_width=True, caption="Tumor & Edema Detection")
                                                else:
                                                    st.info("종양 시각화 생성 불가")
                                            except Exception as e:
                                                st.warning(f"시각화 오류: {str(e)}")
                                        
                                        st.markdown("---")
                                        
                                        # 시퀀스 정보
                                        st.markdown("#### 📋 MRI 시퀀스 정보")
                                        sequence = result.get('sequence', 'Unknown')
                                        st.info(f"시퀀스: **{sequence}**")
                                        
                                        # Segmentation 결과
                                        st.markdown("#### 🎯 종양/부종 검출 결과")
                                        
                                        segmentation = result.get('segmentation', {})
                                        seg_col1, seg_col2 = st.columns(2)
                                        
                                        with seg_col1:
                                            tumor_detected = segmentation.get('tumor_detected', False)
                                            if tumor_detected:
                                                st.success("✅ 종양 검출됨")
                                                volume = segmentation.get('tumor_volume_mm3', 0)
                                                st.metric("종양 부피", f"{volume:.2f} mm³")
                                        
                                        with seg_col2:
                                            edema_detected = segmentation.get('edema_detected', False)
                                            if edema_detected:
                                                st.warning("⚠️ 부종 검출됨")
                                                edema_vol = segmentation.get('edema_volume_mm3', 0)
                                                st.metric("부종 부피", f"{edema_vol:.2f} mm³")
                                        
                                        # Volume 측정
                                        st.markdown("---")
                                        st.markdown("#### 📏 부피 측정")
                                        
                                        measurements = result.get('measurements', {})
                                        vol_col1, vol_col2, vol_col3 = st.columns(3)
                                        
                                        with vol_col1:
                                            tumor_vol = measurements.get('tumor_volume_mm3', 0)
                                            st.metric("종양 부피", f"{tumor_vol:.2f} mm³")
                                        
                                        with vol_col2:
                                            edema_vol = measurements.get('edema_volume_mm3', 0)
                                            st.metric("부종 부피", f"{edema_vol:.2f} mm³")
                                        
                                        with vol_col3:
                                            ratio = measurements.get('edema_to_tumor_ratio', 0)
                                            st.metric("부종/종양 비율", f"{ratio:.2f}")
                                        
                                        # MRI Features
                                        st.markdown("---")
                                        st.markdown("#### 🧬 MRI 특징")
                                        
                                        mri_features = result.get('mri_features', {})
                                        if mri_features:
                                            feat_col1, feat_col2, feat_col3 = st.columns(3)
                                            with feat_col1:
                                                st.metric("평균 강도", f"{mri_features.get('intensity_mean', 0):.2f}")
                                            with feat_col2:
                                                st.metric("표준편차", f"{mri_features.get('intensity_std', 0):.2f}")
                                            with feat_col3:
                                                st.metric("엔트로피", f"{mri_features.get('texture_entropy', 0):.2f}")
                                        
                                        # AI 해석
                                        st.markdown("---")
                                        st.markdown("#### 🤖 AI 임상 해석")
                                        
                                        ai_interpretation = result.get('ai_interpretation', '')
                                        if ai_interpretation:
                                            st.markdown(ai_interpretation)
                                        else:
                                            st.info("AI 해석을 사용하려면 OpenAI API 키를 설정하세요")
                            else:
                                st.info("MRI 이미지를 업로드하고 분석하세요")
                        
                        with result_tab3:
                            if 'fusion_result' in st.session_state:
                                fusion = st.session_state['fusion_result']
                                
                                st.markdown("#### 🎯 통합 위험도 평가")
                                risk = fusion.get('risk_assessment', {})
                                
                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    st.metric("위험도", risk.get('risk_level', 'Unknown'))
                                with col2:
                                    st.metric("위험 점수", f"{risk.get('risk_score', 0)}/{risk.get('max_score', 7)}")
                                with col3:
                                    confidence = fusion.get('confidence_score', 0)
                                    st.metric("신뢰도", f"{confidence*100:.0f}%")
                                
                                st.markdown("#### 📋 통합 인사이트")
                                for insight in fusion.get('integrated_insights', []):
                                    st.info(f"**{insight.get('source')}:** {insight.get('finding')}")
                                
                                st.markdown("#### 💊 권장사항")
                                st.markdown(fusion.get('recommendation', ''))
                            else:
                                st.info("CT와 MRI를 모두 업로드하면 통합 분석을 볼 수 있습니다")
    
    with tab3:
        st.markdown("### 배치 이미지 처리")
        st.info("여러 이미지를 한 번에 처리합니다.")
        
        uploaded_files = st.file_uploader(
            "여러 이미지 선택",
            type=['tif', 'tiff', 'png', 'jpg', 'jpeg'],
            accept_multiple_files=True
        )
        
        if uploaded_files and st.button("배치 분석 시작"):
            st.write(f"총 {len(uploaded_files)}개의 이미지를 처리합니다...")
            # Batch processing logic here
    
    with tab4:
        st.markdown("### 📊 분석 인사이트")
        st.info("💡 이미지를 분석한 후 이 탭에서 종합적인 인사이트를 확인하세요.")
        
        # Check if results exist in session state
        if 'analysis_results' not in st.session_state:
            st.warning("⚠️ 먼저 '단일 이미지 처리' 탭에서 이미지를 분석해주세요.")
        else:
            results = st.session_state['analysis_results']
            
            # Generate insights
            with st.spinner("인사이트 생성 중..."):
                try:
                    insights = generate_comprehensive_insights(results)
                    
                    # Statistical Interpretation
                    with st.expander("📊 통계적 해석", expanded=True):
                        st.markdown(insights['statistical'])
                    
                    # Biological Meaning
                    with st.expander("🔬 생물학적 의미", expanded=True):
                        st.markdown(insights['biological'])
                    
                    # AI Insights (if available)
                    if insights['ai_insights']:
                        with st.expander("🤖 AI 기반 종합 분석", expanded=True):
                            st.markdown("#### OpenAI 분석 결과")
                            st.markdown(insights['ai_insights'])
                            st.caption("*이 분석은 GPT-4를 활용하여 생성되었습니다.")
                    else:
                        with st.expander("🤖 AI 기반 종합 분석", expanded=False):
                            st.warning("""
                            **AI 기반 분석을 사용하려면 OpenAI API 키가 필요합니다.**
                            
                            `.env` 파일에 다음을 추가하세요:
                            ```
                            OPENAI_API_KEY=your_api_key_here
                            ```
                            
                            API 키는 https://platform.openai.com/api-keys 에서 발급받을 수 있습니다.
                            """)
                    
                    # Key Findings
                    with st.expander("🎯 주요 발견사항", expanded=True):
                        st.markdown(insights['key_findings'])
                    
                    # Recommendations
                    with st.expander("💡 권장사항", expanded=True):
                        st.markdown(insights['recommendations'])
                    
                    # ===== DETAILED ANALYSIS VISUALIZATIONS =====
                    st.markdown("---")
                    st.markdown("### 📈 상세 분석 및 시각화")
                    st.caption("AI가 제안한 추가 분석 항목을 시각화합니다")
                    
                    # Get cell-level features if available
                    features_path = results.get('features_path')
                    
                    if features_path and Path(features_path).exists():
                        import pandas as pd
                        
                        # Load features
                        features_df = pd.read_csv(features_path)
                        
                        # Distribution visualizations
                        viz_col1, viz_col2 = st.columns(2)
                        
                        with viz_col1:
                            st.markdown("#### 세포 크기 분포")
                            if 'area' in features_df.columns:
                                fig_area = px.histogram(
                                    features_df, 
                                    x='area',
                                    nbins=30,
                                    title="세포 면적 분포",
                                    labels={'area': '면적 (px²)', 'count': '세포 수'}
                                )
                                fig_area.update_layout(showlegend=False, height=300)
                                
                                # Add mean and std lines
                                mean_area = features_df['area'].mean()
                                std_area = features_df['area'].std()
                                fig_area.add_vline(x=mean_area, line_dash="dash", line_color="red", 
                                                  annotation_text=f"평균")
                                
                                st.plotly_chart(fig_area, use_container_width=True)
                                
                                # Outlier detection
                                q1 = features_df['area'].quantile(0.25)
                                q3 = features_df['area'].quantile(0.75)
                                iqr = q3 - q1
                                lower_bound = q1 - 1.5 * iqr
                                upper_bound = q3 + 1.5 * iqr
                                outliers = features_df[(features_df['area'] < lower_bound) | (features_df['area'] > upper_bound)]
                                
                                if len(outliers) > 0:
                                    st.warning(f"⚠️ 이상치: {len(outliers)}개 ({len(outliers)/len(features_df)*100:.1f}%)")
                                else:
                                    st.success("✅ 균일한 크기 분포")
                        
                        with viz_col2:
                            st.markdown("#### 원형도 분포")
                            if 'circularity' in features_df.columns:
                                fig_circ = px.histogram(
                                    features_df,
                                    x='circularity',
                                    nbins=30,
                                    title="세포 원형도 분포",
                                    labels={'circularity': '원형도', 'count': '세포 수'}
                                )
                                fig_circ.update_layout(showlegend=False, height=300)
                                
                                mean_circ = features_df['circularity'].mean()
                                fig_circ.add_vline(x=mean_circ, line_dash="dash", line_color="red")
                                
                                st.plotly_chart(fig_circ, use_container_width=True)
                                
                                # Health assessment
                                healthy_cells = len(features_df[features_df['circularity'] > 0.7])
                                healthy_pct = healthy_cells / len(features_df) * 100
                                
                                if healthy_pct > 80:
                                    st.success(f"✅ 건강한 세포: {healthy_pct:.1f}%")
                                elif healthy_pct > 60:
                                    st.info(f"ℹ️ 중간 수준: {healthy_pct:.1f}%")
                                else:
                                    st.warning(f"⚠️ 주의 필요: {healthy_pct:.1f}%")
                    
                    else:
                        st.info("📊 상세 시각화를 위해 세포 특징 데이터가 필요합니다.")
                    
                    # ===== CELL CLUSTERING ANALYSIS =====
                    if features_path and Path(features_path).exists():
                        st.markdown("---")
                        st.markdown("### 🧬 세포 클러스터링 분석")
                        st.caption("형태적 유사성을 기반으로 세포를 그룹화합니다")
                        
                        try:
                            from sklearn.cluster import KMeans
                            from sklearn.preprocessing import StandardScaler
                            import pandas as pd
                            
                            features_df = pd.read_csv(features_path)
                            
                            # Prepare features for clustering
                            if 'area' in features_df.columns and 'circularity' in features_df.columns:
                                # Select features for clustering
                                X = features_df[['area', 'circularity']].copy()
                                
                                # Handle any NaN or inf values
                                X = X.replace([np.inf, -np.inf], np.nan).dropna()
                                
                                if len(X) > 10:  # Need minimum samples for clustering
                                    # Standardize features
                                    scaler = StandardScaler()
                                    X_scaled = scaler.fit_transform(X)
                                    
                                    # Determine optimal number of clusters (2-4)
                                    n_clusters = min(3, len(X) // 50 + 2)
                                    
                                    # Perform K-means clustering
                                    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                                    clusters = kmeans.fit_predict(X_scaled)
                                    
                                    # Add cluster labels to dataframe
                                    X['cluster'] = clusters
                                    
                                    # Visualize clusters
                                    fig_cluster = px.scatter(
                                        X,
                                        x='area',
                                        y='circularity',
                                        color='cluster',
                                        title=f"세포 클러스터링 결과 ({n_clusters}개 그룹)",
                                        labels={
                                            'area': '면적 (px²)',
                                            'circularity': '원형도',
                                            'cluster': '클러스터'
                                        },
                                        color_discrete_sequence=px.colors.qualitative.Set2
                                    )
                                    fig_cluster.add_hline(y=0.7, line_dash="dash", line_color="green",
                                                         annotation_text="건강 기준선")
                                    fig_cluster.update_layout(height=400)
                                    st.plotly_chart(fig_cluster, use_container_width=True)
                                    
                                    # Cluster characterization
                                    st.markdown("#### 📋 클러스터별 특성")
                                    
                                    cluster_stats = []
                                    for i in range(n_clusters):
                                        cluster_data = X[X['cluster'] == i]
                                        n_cells = len(cluster_data)
                                        mean_area = cluster_data['area'].mean()
                                        mean_circ = cluster_data['circularity'].mean()
                                        
                                        # Classify cluster health
                                        if mean_circ > 0.75:
                                            health = "✅ 매우 건강"
                                        elif mean_circ > 0.65:
                                            health = "✓ 건강"
                                        elif mean_circ > 0.55:
                                            health = "⚠️ 주의"
                                        else:
                                            health = "🔴 비정상"
                                        
                                        cluster_stats.append({
                                            '클러스터': f'그룹 {i+1}',
                                            '세포 수': n_cells,
                                            '비율': f'{n_cells/len(X)*100:.1f}%',
                                            '평균 크기': f'{mean_area:.1f} px²',
                                            '평균 원형도': f'{mean_circ:.3f}',
                                            '건강도': health
                                        })
                                    
                                    cluster_df = pd.DataFrame(cluster_stats)
                                    st.dataframe(cluster_df, use_container_width=True, hide_index=True)
                                    
                                    # Identify abnormal clusters
                                    st.markdown("#### 🔍 비정상 클러스터 분석")
                                    
                                    abnormal_clusters = []
                                    for i, stats in enumerate(cluster_stats):
                                        mean_circ = float(stats['평균 원형도'])
                                        if mean_circ < 0.6:
                                            abnormal_clusters.append((i, stats))
                                    
                                    if abnormal_clusters:
                                        st.warning(f"⚠️ **{len(abnormal_clusters)}개의 비정상 클러스터 감지**")
                                        
                                        for cluster_idx, stats in abnormal_clusters:
                                            with st.expander(f"🔴 {stats['클러스터']} - {stats['세포 수']}개 세포 ({stats['비율']})"):
                                                st.markdown(f"""
                                                **특징:**
                                                - 평균 크기: {stats['평균 크기']}
                                                - 평균 원형도: {stats['평균 원형도']} (낮음)
                                                - 상태: {stats['건강도']}
                                                
                                                **가능한 원인:**
                                                - 세포 스트레스 또는 약물 효과
                                                - 세포 사멸 과정 (apoptosis)
                                                - 실험 조건의 부적합성
                                                - 특정 세포 주기 단계
                                                
                                                **권장 조치:**
                                                - 이 그룹의 세포를 현미경으로 직접 확인
                                                - 실험 조건 재검토 (배지, 온도, CO₂)
                                                - Live/Dead assay로 생존율 확인
                                                - 다른 시간대에서 재분석
                                                """)
                                    else:
                                        st.success("✅ **모든 클러스터가 정상 범위**")
                                        st.info("모든 세포 그룹이 건강한 형태를 유지하고 있습니다.")
                                    
                                    # Cluster size distribution
                                    st.markdown("#### 📊 클러스터별 크기 분포")
                                    
                                    fig_cluster_box = px.box(
                                        X,
                                        x='cluster',
                                        y='area',
                                        color='cluster',
                                        title="클러스터별 세포 크기 분포",
                                        labels={
                                            'cluster': '클러스터',
                                            'area': '면적 (px²)'
                                        },
                                        color_discrete_sequence=px.colors.qualitative.Set2
                                    )
                                    fig_cluster_box.update_layout(showlegend=False, height=300)
                                    st.plotly_chart(fig_cluster_box, use_container_width=True)
                                    
                                else:
                                    st.warning("클러스터링을 위한 충분한 세포 데이터가 없습니다 (최소 10개 필요)")
                            
                        except Exception as e:
                            st.error(f"클러스터링 분석 중 오류: {str(e)}")
                            with st.expander("오류 상세"):
                                import traceback
                                st.code(traceback.format_exc())
                    
                    # ===== CELL PHYSIOLOGICAL ASSESSMENT =====
                    st.markdown("---")
                    st.markdown("### 🔬 세포 생리학적 평가")
                    st.caption("형태학적 특징을 기반으로 세포의 기능적 상태를 추정합니다")
                    
                    seg_metadata = results.get('segmentation_metadata', {})
                    metrics = results.get('metrics', {})
                    
                    if features_path and Path(features_path).exists():
                        try:
                            import pandas as pd
                            features_df = pd.read_csv(features_path)
                            
                            if 'area' in features_df.columns and 'circularity' in features_df.columns:
                                # Calculate physiological indicators
                                total_cells = len(features_df)
                                
                                # 1. Apoptosis rate estimation (low circularity + variable size)
                                apoptotic_cells = len(features_df[
                                    (features_df['circularity'] < 0.5) | 
                                    (features_df['area'] < features_df['area'].quantile(0.1))
                                ])
                                apoptosis_rate = (apoptotic_cells / total_cells) * 100
                                
                                # 2. Proliferation estimation (large, round cells)
                                proliferating_cells = len(features_df[
                                    (features_df['area'] > features_df['area'].quantile(0.75)) &
                                    (features_df['circularity'] > 0.7)
                                ])
                                proliferation_rate = (proliferating_cells / total_cells) * 100
                                
                                # 3. Healthy cells (normal size and shape)
                                healthy_cells = len(features_df[
                                    (features_df['circularity'] > 0.7) &
                                    (features_df['area'] >= features_df['area'].quantile(0.25)) &
                                    (features_df['area'] <= features_df['area'].quantile(0.75))
                                ])
                                health_rate = (healthy_cells / total_cells) * 100
                                
                                # 4. Cell health index (0-100)
                                mean_circ = features_df['circularity'].mean()
                                cv_area = features_df['area'].std() / features_df['area'].mean()
                                cell_health_index = (mean_circ * 70 + (1 - min(cv_area, 1)) * 30)
                                
                                # Display metrics
                                physio_cols = st.columns(4)
                                
                                with physio_cols[0]:
                                    st.metric("세포 사멸률", f"{apoptosis_rate:.1f}%", 
                                             delta="낮음" if apoptosis_rate < 10 else "높음",
                                             delta_color="inverse")
                                    if apoptosis_rate > 20:
                                        st.caption("⚠️ 높은 사멸률")
                                    else:
                                        st.caption("✅ 정상 범위")
                                
                                with physio_cols[1]:
                                    st.metric("증식 활성", f"{proliferation_rate:.1f}%",
                                             delta="활발" if proliferation_rate > 15 else "보통")
                                    if proliferation_rate > 25:
                                        st.caption("📈 높은 증식")
                                    else:
                                        st.caption("ℹ️ 정상 증식")
                                
                                with physio_cols[2]:
                                    st.metric("건강 세포 비율", f"{health_rate:.1f}%",
                                             delta="우수" if health_rate > 70 else "주의")
                                    if health_rate > 70:
                                        st.caption("✅ 양호")
                                    else:
                                        st.caption("⚠️ 개선 필요")
                                
                                with physio_cols[3]:
                                    st.metric("세포 건강 지수", f"{cell_health_index:.0f}/100")
                                    if cell_health_index > 70:
                                        st.caption("🟢 건강")
                                    elif cell_health_index > 50:
                                        st.caption("🟡 보통")
                                    else:
                                        st.caption("🔴 주의")
                                
                                # Pie chart for cell population distribution
                                st.markdown("#### 📊 세포 집단 분포")
                                
                                # Stressed cells (remaining)
                                stressed_cells = total_cells - (apoptotic_cells + proliferating_cells + healthy_cells)
                                stressed_cells = max(0, stressed_cells)  # Ensure non-negative
                                
                                population_data = pd.DataFrame({
                                    '상태': ['건강', '증식 중', '스트레스', '사멸'],
                                    '세포 수': [healthy_cells, proliferating_cells, stressed_cells, apoptotic_cells],
                                    '비율 (%)': [
                                        health_rate,
                                        proliferation_rate,
                                        (stressed_cells / total_cells) * 100,
                                        apoptosis_rate
                                    ]
                                })
                                
                                fig_pop = px.pie(
                                    population_data,
                                    values='세포 수',
                                    names='상태',
                                    title="세포 상태별 분포",
                                    color='상태',
                                    color_discrete_map={
                                        '건강': '#10b981',
                                        '증식 중': '#3b82f6',
                                        '스트레스': '#f59e0b',
                                        '사멸': '#ef4444'
                                    }
                                )
                                fig_pop.update_traces(textposition='inside', textinfo='percent+label')
                                st.plotly_chart(fig_pop, use_container_width=True)
                                
                                # Detailed interpretation
                                st.markdown("#### 🔍 생리학적 해석")
                                
                                interpretations = []
                                
                                if apoptosis_rate > 20:
                                    interpretations.append("🔴 **높은 세포 사멸률**: 약물 독성, 영양 부족 또는 부적절한 배양 조건 가능성")
                                elif apoptosis_rate < 5:
                                    interpretations.append("✅ **정상 사멸률**: 건강한 세포 집단")
                                
                                if proliferation_rate > 30:
                                    interpretations.append("📈 **높은 증식률**: 활발한 세포 분열 또는 약물 자극 반응")
                                elif proliferation_rate < 5:
                                    interpretations.append("⚠️ **낮은 증식률**: Confluence 도달 또는 성장 억제")
                                
                                if health_rate < 50:
                                    interpretations.append("⚠️ **건강 세포 부족**: 실험 조건 재검토 필요")
                                
                                if cell_health_index < 50:
                                    interpretations.append("🔴 **낮은 건강 지수**: 종합적인 세포 상태 개선 필요")
                                
                                if interpretations:
                                    for interp in interpretations:
                                        st.markdown(f"- {interp}")
                                else:
                                    st.success("✅ 모든 생리학적 지표가 정상 범위에 있습니다")
                                
                                # Recommendations based on physiological status
                                st.markdown("#### 💡 생리학적 기반 권장사항")
                                
                                recommendations = []
                                
                                if apoptosis_rate > 15:
                                    recommendations.append("**세포 사멸 감소 방안:**\n  - 배지 교체 주기 단축\n  - 혈청 농도 증가\n  - 약물 농도 감소 검토")
                                
                                if proliferation_rate < 10 and health_rate > 60:
                                    recommendations.append("**증식 촉진 방안:**\n  - 계대 배양 (sub-culture)\n  - 성장 인자 추가\n  - 배양 밀도 감소")
                                
                                if cell_health_index < 60:
                                    recommendations.append("**전반적 건강도 개선:**\n  - 배양 조건 전반 재검토\n  - 새로운 세포주 사용 고려\n  - 스트레스 요인 제거")
                                
                                if recommendations:
                                    for rec in recommendations:
                                        st.info(rec)
                                else:
                                    st.success("✅ 현재 배양 조건 유지 권장")
                        
                        except Exception as e:
                            st.error(f"생리학적 평가 중 오류: {str(e)}")
                            with st.expander("오류 상세"):
                                import traceback
                                st.code(traceback.format_exc())
                    else:
                        st.info("생리학적 평가를 위해서는 세포 특징 데이터가 필요합니다.")
                    
                    # Export insights button
                    st.markdown("---")
                    col1, col2 = st.columns([3, 1])
                    with col2:
                        # Combine all insights into one text
                        full_report = f"""
# 이미지 분석 종합 인사이트 리포트

{insights['statistical']}

---

{insights['biological']}

---

{insights['key_findings']}

---

{insights['recommendations']}
"""
                        if insights['ai_insights']:
                            full_report += f"""

---

## 🤖 AI 기반 종합 분석

{insights['ai_insights']}
"""
                        
                        st.download_button(
                            label="📥 인사이트 다운로드",
                            data=full_report,
                            file_name="analysis_insights.md",
                            mime="text/markdown",
                            use_container_width=True
                        )
                
                except Exception as e:
                    st.error(f"인사이트 생성 중 오류 발생: {str(e)}")
                    import traceback
                    with st.expander("오류 상세 정보"):
                        st.code(traceback.format_exc())


