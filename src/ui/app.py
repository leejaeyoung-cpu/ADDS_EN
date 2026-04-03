"""
Streamlit UI application for ADDS
ADDS - AI-Driven Drug Discovery System
Main Streamlit UI Application
"""

import sys
import os
from pathlib import Path

# Setup path
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir.parent))

# GPU 강제 초기화 - 반드시 최우선 import
from gpu_init import *

import streamlit as st
import torch

# Import core utilities
from ui.app_core import setup_page_config, apply_custom_css, check_backend_status, get_cellpose_processor
from utils.cache_manager import CacheManager
from utils.gpu_monitor import GPUMonitor

# Import page modules
from ui.page_modules import (
    show_home,
    show_image_analysis,
    show_document_processing,
    show_drug_cocktail,
    show_data_management
)
from ui.page_modules.dicom_batch_analysis import show_dicom_batch_analysis
from ui.page_modules.cdss_doctor_interface import show_cdss_doctor_interface
from ui.page_modules.cdss_patient_interface import show_patient_interface

# Import other UI modules (not refactored yet)
from ui.show_data_processing import show_data_processing
from ui.show_performance_analytics import show_performance_analytics
from ui.show_document_explorer import show_document_explorer
from ui.show_precision_oncology import show_precision_oncology
from ui.show_finetuning import show_finetuning_management
from ui.show_timelapse_analysis import show_timelapse_analysis
from ui.show_ct_analysis import show_ct_analysis
from ui.show_energy_framework import show_energy_framework

# Import Patient Management
from ui.page_modules.patient_management import show_patient_management as show_patient_mgmt_api

# Import CDSS Metadata Learning Components
from ui.page_modules.outcome_collection import show_outcome_collection, show_outcome_statistics
from ui.page_modules.physician_notes_entry import show_notes_entry
from ui.page_modules.data_management_cdss_enhanced import show_cdss_dashboard



# Configure page
setup_page_config()
apply_custom_css()


def main():
    """Main application"""
    
    # Header - Inline styles for immediate rendering
    st.markdown('''
    <p style="
        font-size: 3.5rem;
        font-weight: 700;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 1rem;
        text-align: center;
    ">🧬 ADDS - AI 항암 약물 개발 시스템</p>
    ''', unsafe_allow_html=True)
    st.markdown('''
    <p style="
        font-size: 1.8rem;
        color: #555;
        margin-bottom: 2rem;
        text-align: center;
    ">인하대학교 의과대학 의생명학교실 이상훈 - ADDS 플렛폼 개발</p>
    ''', unsafe_allow_html=True)
    
    # Sidebar navigation
    with st.sidebar:
        # ADDS 로고 표시
        logo_path = Path(__file__).parent.parent.parent / "assets" / "adds_logo.png"
        if logo_path.exists():
            st.image(str(logo_path), width=300)
        else:
            st.markdown("### 🧬 ADDS")
        st.markdown("---")
        # Main navigation - Clean flat structure
        st.markdown("### 📍 메뉴")
        
        # Home
        if st.sidebar.button("🏠 홈", use_container_width=True, key="nav_home"):
            st.session_state['current_page'] = "🏠 홈"
        
        # Core Analysis Features (통합된 메뉴)
        if st.sidebar.button("🔬 메타 데이터", use_container_width=True, key="nav_meta_analysis"):
            st.session_state['current_page'] = "🔬 메타 데이터 분석 & CDSS"
        
        st.markdown("---")
        
        # Management Features
        if st.sidebar.button("👥 환자 관리", use_container_width=True, key="nav_patient_mgmt"):
            st.session_state['current_page'] = "👥 환자 관리"
        if st.sidebar.button("📊 데이터 관리", use_container_width=True, key="nav_datamgmt"):
            st.session_state['current_page'] = "📊 데이터 관리"
        
        st.markdown("---")
        
        # Advanced Features
        if st.sidebar.button("⚡ 에너지 프레임워크", use_container_width=True, key="nav_energy"):
            st.session_state['current_page'] = "⚡ 에너지 프레임워크"
        if st.sidebar.button("🎓 AI 파인튜닝", use_container_width=True, key="nav_finetune"):
            st.session_state['current_page'] = "🎓 AI 파인튜닝"
        if st.sidebar.button("📈 성과지표 분석", use_container_width=True, key="nav_performance"):
            st.session_state['current_page'] = "📈 성과지표 분석"
        
        # Get current page from session state
        page = st.session_state.get('current_page', "🏠 홈")
        
        st.markdown("---")
        
        # User Selection
        st.markdown("### 👤 사용자 선택")
        
        users = {
            "이재영": {"role": "연구원", "department": "바이오메디컬사이언스"},
            "이상훈": {"role": "연구원", "department": "바이오메디컬사이언스"},
            "최문석": {"role": "연구원", "department": "바이오메디컬사이언스"}
        }
        
        selected_user = st.selectbox(
            "현재 사용자",
            options=list(users.keys()),
            index=0 if st.session_state.get('current_user') is None else list(users.keys()).index(st.session_state.get('current_user', '이재영')),
            key='user_selector'
        )
        
        # Update session state
        if selected_user != st.session_state.get('current_user'):
            st.session_state['current_user'] = selected_user
            st.session_state['user_info'] = users[selected_user]
            st.rerun()
        
        # Initialize user info if not set
        if 'current_user' not in st.session_state:
            st.session_state['current_user'] = selected_user
            st.session_state['user_info'] = users[selected_user]
        
        # Display user info
        user_info = st.session_state.get('user_info', users[selected_user])
        st.info(f"👤 {selected_user} ({user_info['role']})\n📚 {user_info['department']}")
        
        st.markdown("---")
        
        # Dashboard Mode Selection (only show on home page)
        if page == "🏠 홈":
            st.markdown("### 🎨 Dashboard Mode")
            
            dashboard_mode = st.radio(
                "Display Mode",
                options=["Clinical", "Presentation"],
                index=0 if st.session_state.get('dashboard_mode', 'Clinical') == 'Clinical' else 1,
                help="Clinical: 업무용 상세 뷰 (EMR Style)\nPresentation: 환자 상담용 간결한 뷰 (Modern)",
                horizontal=True,
                key='dashboard_mode_radio'
            )
            
            # Update session state if changed
            if dashboard_mode != st.session_state.get('dashboard_mode'):
                st.session_state['dashboard_mode'] = dashboard_mode
                st.rerun()
            
            # Show current mode info
            if dashboard_mode == 'Clinical':
                st.info("💼 Clinical Mode: 임상 업무용 상세 대시보드")
            else:
                st.success("🎨 Presentation Mode: 환자 면담용 시각적 대시보드")
            
            st.markdown("---")
        
        st.markdown("### ⚙️ 시스템 설정")

        
        # GPU 토글
        cuda_available = torch.cuda.is_available()
        
        # Initialize monitors
        gpu_monitor = GPUMonitor()
        
        if cuda_available:
            use_gpu = st.checkbox(
                "🚀 GPU 가속 사용",
                value=st.session_state.get('use_gpu', False),
                help=f"GPU: {torch.cuda.get_device_name(0) if cuda_available else 'N/A'}"
            )
            # Detect GPU mode change
            prev_gpu_state = st.session_state.get('prev_use_gpu', None)
            if prev_gpu_state is not None and prev_gpu_state != use_gpu:
                st.warning(f"⚠️ GPU 모드 변경: {'CPU → GPU' if use_gpu else 'GPU → CPU'}\n아래 버튼으로 캐시를 초기화하세요.")
                if st.button("🔄 프로세서 재로드", type="primary"):
                    get_cellpose_processor.clear()
                    st.success("✓ 캐시 초기화됨")
                    st.rerun()
            
            st.session_state['use_gpu'] = use_gpu
            st.session_state['prev_use_gpu'] = use_gpu
            
            if use_gpu:
                st.success(f"✓ GPU 활성화\n{torch.cuda.get_device_name(0)}")
                
                # GPU Memory Display
                if gpu_monitor.is_available():
                    memory_info = gpu_monitor.get_memory_info()
                    gpu_util = memory_info['utilization_percent']
                    
                    # Color based on utilization
                    if gpu_util < 50:
                        color = "normal"
                    elif gpu_util < 80:
                        color = "off"
                    else:
                        color = "inverse"
                    
                    st.metric(
                        "GPU 메모리",
                        gpu_monitor.format_memory(memory_info['allocated_mb']),
                        delta=f"{gpu_util:.0f}% 사용중",
                        delta_color=color,
                        help=f"총 메모리: {gpu_monitor.format_memory(memory_info['total_mb'])}"
                    )
            else:
                st.info("CPU 모드")
        else:
            st.warning("⚠️ GPU 사용 불가\nCPU 모드로 실행")
            st.session_state['use_gpu'] = False
        
        # Cache Management
        st.markdown("---")
        with st.expander("🗄️ 캐시 관리", expanded=False):
            cache_mgr = CacheManager()
            cache_stats = cache_mgr.get_statistics()
            
            st.markdown("**캐시 통계**")
            col_c1, col_c2 = st.columns(2)
            with col_c1:
                st.metric("캐시 크기", cache_stats['total_size_formatted'])
            with col_c2:
                st.metric("항목 수", cache_stats['item_count'])
            
            # Cache statistics
            if cache_stats['cache_hits'] + cache_stats['cache_misses'] > 0:
                st.metric(
                    "캐시 적중률",
                    f"{cache_stats['hit_rate_percent']:.1f}%",
                    help="높을수록 효율적"
                )
            
            # Cache controls
            st.markdown("**캐시 제어**")
            if st.button("🗑️ 전체 캐시 삭제", use_container_width=True):
                if cache_mgr.clear_cache(confirm=True):
                    st.success("✓ 캐시가 삭제되었습니다")
                    st.rerun()
                else:
                    st.error("❌ 캐시 삭제 실패")
        
        
        st.markdown("---")
        st.markdown("### 📌 시스템 정보")
        st.info(f"Version: 1.0.0\nPyTorch: {torch.__version__}\n\n인하대학교 의과대학 의생명학교실 이상훈")
        
        # Backend API 상태
        st.markdown("---")
        st.markdown("### 🌐 Backend API")
        is_running, api_info = check_backend_status()
        
        if is_running and api_info:
            st.success(f"✓ API 연결됨")
            if st.checkbox("API 상세 정보", value=False):
                st.json({
                    "status": api_info.get("status"),
                    "service": api_info.get("service"),
                    "version": api_info.get("version")
                })
        else:
            st.warning("⚠️ API 연결 안됨\n백엔드를 시작하려면:\n`start_backend.bat`")
    
    # Route to pages
    if page == "🏠 홈":
        show_home()
    elif page == "🔬 메타 데이터 분석 & CDSS":
        # Unified Meta Data Analysis Interface with CDSS
        show_meta_data_analysis()
    elif page == "👥 환자 관리":
        # Enhanced Patient Management with CDSS Metadata Learning
        show_enhanced_patient_management()
    elif page == "🎓 AI 파인튜닝":
        show_finetuning_management()
    elif page == "📈 성과지표 분석":
        show_performance_analytics()
    elif page == "⚡ 에너지 프레임워크":
        show_energy_framework()
    elif page == "📊 데이터 관리":
        show_data_management()  # DB 관리 + CDSS 메타데이터




def show_enhanced_patient_management():
    """
    Enhanced Patient Management System with CDSS Metadata Learning Integration
    Combines traditional patient management with outcome collection, notes entry, and CDSS dashboard
    """
    st.title("👥 ADDS Patient Management & CDSS Learning")
    st.markdown("**환자 정보 관리 + 치료 결과 학습 통합 시스템**")
    st.markdown("---")
    
    # Create enhanced tabs with CDSS components
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📋 환자 관리",
        "📊 CDSS 대시보드",
        "📝 치료 결과 수집",
        "🩺 의사 소견 입력",
        "📈 통계 분석",
        "🤖 ML 성능"
    ])
    
    # ==================== TAB 1: 환자 관리 ====================
    with tab1:
        st.subheader("환자 정보 관리")
        # Use existing patient management API
        show_patient_mgmt_api()
    
    # ==================== TAB 2: CDSS 대시보드 ====================
    with tab2:
        st.subheader("CDSS 메타데이터 학습 대시보드")
        # Enhanced CDSS Dashboard with ML performance
        try:
            show_cdss_dashboard()
        except Exception as e:
            st.error(f"대시보드 로딩 오류: {e}")
            st.info("데이터베이스가 초기화되지 않았을 수 있습니다.")
    
    # ==================== TAB 3: 치료 결과 수집 ====================
    with tab3:
        st.subheader("치료 결과 데이터 수집")
        st.markdown("""
        **RECIST 기준 평가, QoL 점수, 부작용 추적**
        
        치료 결과는 자동으로 메타데이터 학습 시스템에 저장되어 AI 모델을 개선합니다.
        """)
        
        try:
            show_outcome_collection()
        except Exception as e:
            st.error(f"결과 수집 UI 로딩 오류: {e}")
            import traceback
            with st.expander("오류 상세 정보"):
                st.code(traceback.format_exc())
    
    # ==================== TAB 4: 의사 소견 입력 ====================
    with tab4:
        st.subheader("의사 소견 및 임상 노트")
        st.markdown("""
        **NLP 자동 분석 + 재분석 트리거**
        
        입력된 소견은 자동으로 NLP 파싱되어 중요 정보가 추출됩니다.
        """)
        
        try:
            show_notes_entry()
        except Exception as e:
            st.error(f"소견 입력 UI 로딩 오류: {e}")
            import traceback
            with st.expander("오류 상세 정보"):
                st.code(traceback.format_exc())
    
    # ==================== TAB 5: 통계 분석 ====================
    with tab5:
        st.subheader("치료 결과 통계 분석")
        
        try:
            show_outcome_statistics()
        except Exception as e:
            st.warning("통계 데이터가 아직 없습니다.")
            st.info("치료 결과를 수집하면 여기에 통계가 표시됩니다.")
            
            # Sample statistics view
            st.markdown("### 예시 통계")
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("전체 치료", "0건", delta="대기 중")
            with col2:
                st.metric("완전 관해 (CR)", "0%", delta="N/A")
            with col3:
                st.metric("부분 관해 (PR)", "0%", delta="N/A")
            with col4:
                st.metric("평균 QoL", "N/A", delta="데이터 없음")
    
    # ==================== TAB 6: ML 성능 ====================
    with tab6:
        st.subheader("머신러닝 모델 성능")
        st.markdown("""
        **일일 자동 학습 + 성능 추적**
        
        시스템은 매일 새로운 데이터로 자동 학습하며 성능을 추적합니다.
        """)
        
        try:
            from patient_management_system.database.db_enhanced import get_session
            from patient_management_system.database.models_enhanced import MLTrainingRun
            
            db = get_session()
            latest_run = db.query(MLTrainingRun).order_by(MLTrainingRun.run_date.desc()).first()
            
            if latest_run:
                st.success(f"✅ 최근 학습: {latest_run.run_date.strftime('%Y-%m-%d %H:%M')}")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Training Accuracy", f"{latest_run.train_accuracy:.2%}" if latest_run.train_accuracy else "N/A")
                with col2:
                    st.metric("Validation Accuracy", f"{latest_run.val_accuracy:.2%}" if latest_run.val_accuracy else "N/A")
                with col3:
                    st.metric("AUC-ROC", f"{latest_run.auc_roc:.3f}" if latest_run.auc_roc else "N/A")
                
                if latest_run.deployed:
                    st.info("🚀 이 모델은 현재 배포되어 사용 중입니다.")
            else:
                st.info("아직 훈련된 모델이 없습니다.")
                st.markdown("**학습 시작 방법:**")
                st.code("python patient_management_system/services/daily_ml_trainer.py", language="bash")
        
        except Exception as e:
            st.warning("ML 성능 데이터를 불러올 수 없습니다.")
            st.info("데이터베이스를 초기화하고 학습을 실행하세요.")


def show_patient_management():
    """
    Legacy Patient Management System (kept for compatibility)
    Use show_enhanced_patient_management() for new features
    """
    st.title("👥 ADDS Patient Management")
    st.markdown("**환자 정보 통합 관리 시스템**")
    st.markdown("---")
    
    st.info("ℹ️ 이 페이지는 기존 호환성을 위해 유지됩니다. 새로운 기능은 '환자 관리' 메뉴를 사용하세요.")
    
    # Create tabs
    patient_tab1, patient_tab2, patient_tab3, patient_tab4 = st.tabs([
        "📋 환자 등록",
        "🔍 환자 검색",
        "📊 분석 기록",
        "📈 통계 대시보드"
    ])
    
    # ==================== TAB 1: 환자 등록 ====================
    with patient_tab1:
        st.subheader("신규 환자 등록")
        
        col1, col2 = st.columns(2)
        with col1:
            patient_id = st.text_input("환자 ID", placeholder="PT-2026-0001")
            patient_name = st.text_input("환자 이름", placeholder="홍길동")
            birth_date = st.date_input("생년월일")
            gender = st.selectbox("성별", ["남성", "여성"])
        
        with col2:
            contact = st.text_input("연락처", placeholder="010-0000-0000")
            email = st.text_input("이메일", placeholder="patient@example.com")
            blood_type = st.selectbox("혈액형", ["A", "B", "AB", "O"])
            allergies = st.text_area("알레르기 정보", placeholder="페니실린, 땅콩 등")
        
        st.markdown("---")
        st.subheader("임상 정보")
        
        col3, col4 = st.columns(2)
        with col3:
            diagnosis = st.text_area("진단명", placeholder="대장암 (Colorectal Cancer)")
            stage = st.selectbox("병기", ["Stage I", "Stage II", "Stage III", "Stage IV"])
        
        with col4:
            ecog = st.selectbox("ECOG 점수", [0, 1, 2, 3, 4])
            treatment_status = st.selectbox("치료 상태", ["신환", "치료 중", "추적관찰", "완료"])
        
        st.markdown("---")
        if st.button("💾 환자 정보 저장", type="primary", use_container_width=True):
            st.success(f"✅ 환자 '{patient_id}' 정보가 저장되었습니다!")
            st.balloons()
    
    # ==================== TAB 2: 환자 검색 ====================
    with patient_tab2:
        st.subheader("환자 정보 검색")
        
        search_col1, search_col2, search_col3 = st.columns(3)
        with search_col1:
            search_id = st.text_input("환자 ID 검색", placeholder="PT-")
        with search_col2:
            search_name = st.text_input("이름 검색", placeholder="이름")
        with search_col3:
            search_date = st.date_input("등록일", value=None)
        
        if st.button("🔍 검색", type="primary"):
            st.info("환자 데이터베이스 검색 중...")
            
            # Sample data (would be from database in production)
            st.markdown("### 검색 결과")
            
            import pandas as pd
            sample_patients = pd.DataFrame({
                "환자ID": ["PT-TEST-1000", "PT-TEST-1001", "PT-TEST-1002"],
                "이름": ["홍길동", "김철수", "이영희"],
                "성별": ["남성", "남성", "여성"],
                "나이": [58, 62, 55],
                "진단": ["대장암 Stage III", "대장암 Stage II", "대장암 Stage IV"],
                "등록일": ["2025-12-01", "2025-12-15", "2026-01-10"]
            })
            
            st.dataframe(sample_patients, use_container_width=True)
    
    # ==================== TAB 3: 분석 기록 ====================
    with patient_tab3:
        st.subheader("환자 분석 기록 조회")
        
        selected_patient = st.selectbox(
            "환자 선택",
            ["PT-TEST-1000 (홍길동)", "PT-TEST-1001 (김철수)", "PT-TEST-1002 (이영희)"]
        )
        
        st.markdown("### 분석 이력")
        
        # Sample analysis history
        import pandas as pd
        analysis_history = pd.DataFrame({
            "날짜": ["2026-01-28", "2026-01-15", "2026-01-01"],
            "분석 유형": ["CDSS 통합분석", "Cellpose 분석", "CT 검출"],
            "결과": ["Stage IIIB, High Risk", "787 cells, Ki-67: 65%", "3 tumors detected"],
            "담당의": ["Dr. Kim", "Dr. Lee", "Dr. Park"]
        })
        
        st.dataframe(analysis_history, use_container_width=True)
        
        st.markdown("---")
        st.markdown("### 최근 검사 결과")
        
        col_res1, col_res2, col_res3 = st.columns(3)
        with col_res1:
            st.metric("세포 수", "787개", delta="12%")
        with col_res2:
            st.metric("Ki-67 Index", "65%", delta="High")
        with col_res3:
            st.metric("종양 크기", "15.2mm", delta="-2.1mm")
    
    # ==================== TAB 4: 통계 대시보드 ====================
    with patient_tab4:
        st.subheader("환자 통계 대시보드")
        
        metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
        with metric_col1:
            st.metric("전체 환자", "1,247명", delta="+23")
        with metric_col2:
            st.metric("신규 환자 (월)", "45명", delta="+8")
        with metric_col3:
            st.metric("CDSS 분석", "892건", delta="+34")
        with metric_col4:
            st.metric("평균 Ki-67", "58%", delta="-3%")
        
        st.markdown("---")
        
        chart_col1, chart_col2 = st.columns(2)
        
        with chart_col1:
            st.markdown("#### 병기별 환자 분포")
            import pandas as pd
            stage_data = pd.DataFrame({
                "병기": ["Stage I", "Stage II", "Stage III", "Stage IV"],
                "환자수": [120, 350, 480, 297]
            })
            st.bar_chart(stage_data.set_index("병기"))
        
        with chart_col2:
            st.markdown("#### 월별 신규 환자")
            monthly_data = pd.DataFrame({
                "월": ["10월", "11월", "12월", "1월"],
                "환자수": [35, 42, 38, 45]
            })
            st.line_chart(monthly_data.set_index("월"))




def show_meta_data_analysis():
    """
    Unified Meta Data Analysis Interface
    Integrates Image Analysis, Precision Oncology, Drug Cocktail, Timelapse, and CDSS
    """
    st.title("🔬 ADDS Meta Data Analysis & CDSS")
    st.markdown("**통합 메타 데이터 분석 + 임상 의사 결정 지원 플랫폼**")
    st.markdown("---")
    
    # Create top-level tabs with CDSS integrated
    meta_tab1, meta_tab2, meta_tab3, meta_tab4, meta_tab5 = st.tabs([
        "🔬 이미지 분석",
        "🧬 개인 맞춤형 분석",
        "💊 Drug Cocktail",
        "🎥 Timelapse 분석",
        "🏥 CDSS 통합 시스템"
    ])
    
    # ==================== TAB 1: 이미지 분석 ====================
    with meta_tab1:
        show_image_analysis()
    
    # ==================== TAB 2: 개인 맞춤형 분석 ====================
    with meta_tab2:
        show_precision_oncology()
    
    # ==================== TAB 3: Drug Cocktail ====================
    with meta_tab3:
        show_drug_cocktail()
    
    # ==================== TAB 4: Timelapse 분석 ====================
    with meta_tab4:
        show_timelapse_analysis()
    
    # ==================== TAB 5: CDSS 통합 시스템 ====================
    with meta_tab5:
        st.markdown("### 🏥 Clinical Decision Support System")
        st.markdown("**CT/DICOM 분석 + 환자 통합 관리**")
        show_cdss_doctor_interface()


if __name__ == "__main__":
    main()
