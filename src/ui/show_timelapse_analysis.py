"""
Time-lapse Analysis UI (Streamlit)
==================================
실시간 영상 분석을 위한 Streamlit 인터페이스

기능:
- 동영상/이미지 시퀀스 업로드
- 실시간 분석 진행 상황
- 결과 시각화
- 약물 비교 분석
- 칵테일 시너지 분석
"""

import streamlit as st
import sys
from pathlib import Path
import tempfile
import shutil

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.evaluation.timelapse_pipeline import TimeLapsePipeline
import logging

logger = logging.getLogger(__name__)


def show_timelapse_analysis():
    """Time-lapse 분석 메인 UI"""
    
    st.header("🎥 Time-lapse 세포 분석")
    st.markdown("""
    3-4일 장기 촬영 데이터를 분석하여 다음을 제공합니다:
    - 📈 세포 증식 곡선 & Doubling time
    - 🔍 세포 추적 & 계보 분석
    - 🏃 세포 이동 분석
    - 💊 약물 반응성 정량화
    - 🧪 칵테일 시너지 분석
    """)
    
    # 탭 구성
    tab1, tab2, tab3 = st.tabs([
        "📁 기본 분석",
        "💊 약물 비교",
        "🧪 칵테일 시너지"
    ])
    
    with tab1:
        show_basic_analysis_tab()
    
    with tab2:
        show_drug_comparison_tab()
    
    with tab3:
        show_cocktail_synergy_tab()


def show_basic_analysis_tab():
    """기본 분석 탭"""
    st.subheader("단일 실험 분석")
    
    st.markdown("### 1️⃣ 데이터 업로드")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        input_type = st.radio(
            "입력 형식",
            ["동영상 파일", "이미지 시퀀스 (폴더)"],
            horizontal=True
        )
    
    with col2:
        max_frames = st.number_input(
            "최대 프레임 수 (0 = 전체)",
            min_value=0,
            value=0,
            help="테스트를 위해 일부만 분석할 수 있습니다"
        )
    
    uploaded_file = None
    input_path = None
    
    if input_type == "동영상 파일":
        uploaded_file = st.file_uploader(
            "동영상 파일 선택",
            type=['avi', 'mp4', 'mov', 'mkv'],
            help="Time-lapse 동영상을 업로드하세요"
        )
    else:
        st.info("이미지 시퀀스는 서버의 디렉토리 경로를 직접 입력하세요")
        input_path = st.text_input(
            "이미지 시퀀스 디렉토리 경로",
            placeholder="예: C:/data/timelapse/frames/"
        )
    
    st.markdown("### 2️⃣ 분석 설정")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        experiment_name = st.text_input(
            "실험 이름",
            value="Experiment_1",
            help="결과 파일명에 사용됩니다"
        )
    
    with col2:
        frame_interval = st.number_input(
            "프레임 간격 (분)",
            min_value=0.1,
            value=5.0,
            step=0.5,
            help="촬영 간격을 입력하세요"
        )
    
    with col3:
        cellpose_model = st.selectbox(
            "Cellpose 모델",
            ["cyto2", "cyto", "cyto3"],
            help="세포 타입에 맞는 모델 선택"
        )
    
    with st.expander("고급 설정"):
        col1, col2 = st.columns(2)
        
        with col1:
            diameter = st.number_input(
                "세포 지름 (픽셀, 0=자동)",
                min_value=0,
                value=0,
                help="0이면 Cellpose가 자동으로 추정합니다"
            )
        
        with col2:
            iou_threshold = st.slider(
                "추적 IoU 임계값",
                min_value=0.1,
                max_value=0.9,
                value=0.3,
                step=0.05,
                help="세포 매칭의 엄격도 (높을수록 엄격)"
            )
        
        generate_video = st.checkbox(
            "추적 동영상 생성",
            value=True,
            help="시간이 오래 걸릴 수 있습니다"
        )
    
    # 분석 실행
    if st.button("🚀 분석 시작", type="primary", use_container_width=True):
        
        # 입력 검증
        if uploaded_file is None and not input_path:
            st.error("파일을 업로드하거나 경로를 입력하세요!")
            return
        
        # 임시 파일 저장
        if uploaded_file:
            with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as tmp_file:
                tmp_file.write(uploaded_file.read())
                analysis_input = tmp_file.name
        else:
            if not Path(input_path).exists():
                st.error(f"경로를 찾을 수 없습니다: {input_path}")
                return
            analysis_input = input_path
        
        # 파이프라인 실행
        try:
            with st.spinner("분석 중... (수 분 소요될 수 있습니다)"):
                
                # Progress tracking
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                status_text.text("1/5: 데이터 로딩...")
                progress_bar.progress(0.1)
                
                # 파이프라인 초기화
                pipeline = TimeLapsePipeline(
                    frame_interval_minutes=frame_interval,
                    cellpose_model=cellpose_model,
                    cellpose_diameter=diameter if diameter > 0 else None,
                    iou_threshold=iou_threshold
                )
                
                status_text.text("2/5: Cellpose 세그멘테이션...")
                progress_bar.progress(0.3)
                
                # 분석 실행
                results = pipeline.run_full_analysis(
                    analysis_input,
                    output_dir=Path("data/timelapse/results") / experiment_name,
                    experiment_name=experiment_name,
                    max_frames=max_frames if max_frames > 0 else None,
                    generate_video=generate_video
                )
                
                status_text.text("5/5: 시각화 생성...")
                progress_bar.progress(1.0)
                
                # 결과 저장
                st.session_state['timelapse_results'] = results
                
            st.success("✅ 분석 완료!")
            
            # 결과 표시
            display_results(results)
            
        except Exception as e:
            st.error(f"분석 중 오류 발생: {e}")
            logger.exception(e)
        
        finally:
            # 임시 파일 정리
            if uploaded_file and Path(analysis_input).exists():
                Path(analysis_input).unlink()


def show_drug_comparison_tab():
    """약물 비교 탭"""
    st.subheader("약물 효과 비교 분석")
    
    st.markdown("### 실험 조건 설정")
    
    # 대조군
    st.markdown("#### 대조군 (Control)")
    control_path = st.text_input(
        "대조군 경로",
        key="control_path",
        placeholder="C:/data/control.avi"
    )
    
    # 처리군
    st.markdown("#### 처리군 (Treatments)")
    num_treatments = st.number_input(
        "처리군 개수",
        min_value=1,
        max_value=5,
        value=2
    )
    
    treatment_paths = {}
    for i in range(num_treatments):
        col1, col2 = st.columns([1, 2])
        with col1:
            drug_name = st.text_input(
                f"약물 {i+1} 이름",
                value=f"Drug_{chr(65+i)}",
                key=f"drug_name_{i}"
            )
        with col2:
            drug_path = st.text_input(
                f"약물 {i+1} 경로",
                key=f"drug_path_{i}",
                placeholder=f"C:/data/{drug_name.lower()}.avi"
            )
        
        if drug_name and drug_path:
            treatment_paths[drug_name] = drug_path
    
    # 분석 설정
    with st.expander("분석 설정"):
        frame_interval = st.number_input(
            "프레임 간격 (분)",
            min_value=0.1,
            value=5.0,
            key="drug_comp_interval"
        )
    
    # 실행
    if st.button("🔬 약물 비교 분석 시작", type="primary", use_container_width=True):
        if not control_path or not treatment_paths:
            st.error("모든 경로를 입력하세요!")
            return
        
        try:
            with st.spinner("약물 비교 분석 중..."):
                pipeline = TimeLapsePipeline(frame_interval_minutes=frame_interval)
                
                results = pipeline.run_drug_comparison(
                    control_path,
                    treatment_paths,
                    output_dir=Path("data/timelapse/results/drug_comparison"),
                    experiment_name="Drug_Comparison"
                )
                
                st.session_state['drug_comparison_results'] = results
            
            st.success("✅ 약물 비교 분석 완료!")
            
            # 결과 표시
            display_drug_comparison_results(results)
            
        except Exception as e:
            st.error(f"분석 중 오류: {e}")
            logger.exception(e)


def show_cocktail_synergy_tab():
    """칵테일 시너지 탭"""
    st.subheader("칵테일 시너지 분석")
    
    st.markdown("""
    항암제 + 엑소좀 조합의 시너지 효과를 분석합니다.
    Bliss independence model을 사용하여 시너지 스코어를 계산합니다.
    """)
    
    st.markdown("### 실험 조건")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 단독 실험")
        control_path = st.text_input("대조군", key="synergy_control")
        drug_a_name = st.text_input("Drug A 이름", value="Anticancer_Drug")
        drug_a_path = st.text_input("Drug A 경로", key="synergy_drug_a")
        drug_b_name = st.text_input("Drug B 이름", value="Exosome")
        drug_b_path = st.text_input("Drug B 경로", key="synergy_drug_b")
    
    with col2:
        st.markdown("#### 조합 실험")
        combo_path = st.text_input(
            f"{drug_a_name} + {drug_b_name} 경로",
            key="synergy_combo"
        )
    
    # 실행
    if st.button("🧪 시너지 분석 시작", type="primary", use_container_width=True):
        if not all([control_path, drug_a_path, drug_b_path, combo_path]):
            st.error("모든 경로를 입력하세요!")
            return
        
        try:
            with st.spinner("시너지 분석 중..."):
                pipeline = TimeLapsePipeline()
                
                results = pipeline.run_cocktail_synergy_analysis(
                    control_path,
                    drug_a_path,
                    drug_b_path,
                    combo_path,
                    output_dir=Path("data/timelapse/results/synergy"),
                    experiment_name="Cocktail_Synergy",
                    drug_a_name=drug_a_name,
                    drug_b_name=drug_b_name
                )
                
                st.session_state['synergy_results'] = results
            
            st.success("✅ 시너지 분석 완료!")
            
            # 결과 표시
            display_synergy_results(results)
            
        except Exception as e:
            st.error(f"분석 중 오류: {e}")
            logger.exception(e)


def display_results(results: dict):
    """기본 분석 결과 표시"""
    st.markdown("---")
    st.markdown("## 📊 분석 결과")
    
    # 통계 요약
    col1, col2, col3, col4 = st.columns(4)
    
    track_stats = results['track_statistics']
    prolif = results['analysis_report']['proliferation']
    
    with col1:
        st.metric("총 Track 수", track_stats['total_tracks'])
    
    with col2:
        doubling_time = prolif.get('doubling_time_hours', 0)
        st.metric("Doubling Time", f"{doubling_time:.1f}h" if doubling_time else "N/A")
    
    with col3:
        st.metric("분열 이벤트", track_stats['division_events'])
    
    with col4:
        final_count = prolif.get('final_count', 0)
        st.metric("최종 세포 수", final_count)
    
    # 다운로드 링크
    st.markdown("### 📥 결과 파일")
    
    output_files = results['output_files']
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if output_files.get('tracks_csv'):
            st.markdown(f"📄 [Tracks CSV]({output_files['tracks_csv']})")
    
    with col2:
        if output_files.get('tracking_video'):
            st.markdown(f"🎥 [Tracking Video]({output_files['tracking_video']})")
    
    with col3:
        if output_files.get('dashboard_html'):
            st.markdown(f"📊 [Dashboard HTML]({output_files['dashboard_html']})")
    
    # 대시보드 iframe 표시
    if output_files.get('dashboard_html'):
        with open(output_files['dashboard_html'], 'r', encoding='utf-8') as f:
            st.components.v1.html(f.read(), height=1200, scrolling=True)


def display_drug_comparison_results(results: dict):
    """약물 비교 결과 표시"""
    st.markdown("### 💊 약물 효과 비교")
    
    drug_responses = results['drug_responses']
    
    # 효과 요약
    for drug_name, response in drug_responses.items():
        with st.expander(f"{drug_name} - {response['growth_inhibition_percent']:.1f}% inhibition"):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Growth Inhibition", f"{response['growth_inhibition_percent']:.1f}%")
            
            with col2:
                st.metric("Relative Count", f"{response['relative_cell_count']:.2f}")
            
            with col3:
                effect = "✅ 효과 있음" if response['effect_detected'] else "❌ 효과 없음"
                st.metric("효과 판정", effect)
    
    # 비교 플롯
    if results.get('comparison_plot'):
        with open(results['comparison_plot'], 'r', encoding='utf-8') as f:
            st.components.v1.html(f.read(), height=700)


def display_synergy_results(results: dict):
    """시너지 분석 결과 표시"""
    st.markdown("### 🧪 칵테일 시너지 분석")
    
    synergy = results['synergy_analysis']
    
    # 시너지 요약
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("평균 시너지 스코어", f"{synergy['mean_synergy_score']:.3f}")
    
    with col2:
        st.metric("Peak 시너지 시간", f"{synergy['peak_synergy_time_hours']:.1f}h")
    
    with col3:
        judgment = "✅ Synergistic" if synergy['is_synergistic'] else "⚠️ Not synergistic"
        if synergy.get('is_antagonistic'):
            judgment = "❌ Antagonistic"
        st.metric("판정", judgment)
    
    # 해석
    if synergy['is_synergistic']:
        st.success("🎉 조합이 시너지 효과를 보입니다! 단독 사용보다 효과적입니다.")
    elif synergy.get('is_antagonistic'):
        st.warning("⚠️ 조합이 길항 효과를 보입니다. 단독 사용보다 덜 효과적입니다.")
    else:
        st.info("ℹ️ 조합이 additive 효과를 보입니다. 단독 효과의 합과 유사합니다.")
    
    # 시너지 플롯
    if results.get('synergy_plot'):
        with open(results['synergy_plot'], 'r', encoding='utf-8') as f:
            st.components.v1.html(f.read(), height=600)


if __name__ == "__main__":
    show_timelapse_analysis()
