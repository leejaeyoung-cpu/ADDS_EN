"""
Deep Analysis Options Component
UI for enabling advanced quality enhancement and morphological analysis
"""

import streamlit as st
from typing import Dict


def show_deep_analysis_options() -> Dict:
    """
    Display deep analysis options UI
    
    Returns:
        Dictionary with selected options
    """
    options = {}
    
    st.subheader("🔬 심층 분석 옵션")
    
    with st.expander("⚙️ 고급 전처리 및 분석", expanded=False):
        st.markdown("#### 이미지 품질 향상")
        
        col1, col2 = st.columns(2)
        
        with col1:
            enable_denoise = st.checkbox(
                "노이즈 제거",
                value=False,
                help="SNR이 낮거나 노이즈가 많은 이미지일 때 활성화",
                key="enable_denoise"
            )
            
            if enable_den잘oise:
                denoise_method = st.selectbox(
                    "방법",
                    ['bilateral', 'nlm', 'gaussian'],
                    help="bilateral: 에지 보존 (권장), nlm: 강력 (느림), gaussian: 빠름",
                    key="denoise_method"
                )
                options['denoise'] = True
                options['denoise_method'] = denoise_method
            else:
                options['denoise'] = False
        
        with col2:
            enable_sharpen = st.checkbox(
                "포커스 향상",
                value=False,
                help="흐릿한 이미지를 선명하게 (Unsharp masking)",
                key="enable_sharpen"
            )
            
            if enable_sharpen:
                sharpen_strength = st.slider(
                    "강도",
                    0.5, 2.0, 1.0,
                    step=0.1,
                    help="1.0 = 보통, 1.5-2.0 = 강함",
                    key="sharpen_strength"
                )
                options['sharpen'] = True
                options['sharpen_strength'] = sharpen_strength
            else:
                options['sharpen'] = False
        
        st.markdown("---")
        
        # Contrast enhancement
        enable_contrast = st.checkbox(
            "대비 최적화 (CLAHE)",
            value=False,
            help="대비가 낮은 이미지의 시각성 향상",
            key="enable_contrast"
        )
        
        if enable_contrast:
            options['enhance_contrast'] = True
        else:
            options['enhance_contrast'] = False
        
        st.markdown("---")
        st.markdown("#### 고급 형태학적 분석")
        
        enable_morphology = st.checkbox(
            "상세 형태 분석",
            value=False,
            help="세포 분포, 이상치 탐지, 형태 분류 분석",
            key="enable_morphology"
        )
        
        if enable_morphology:
            st.info("""
            **포함 사항:**
            - 📊 면적 분포 분석 (왜도, 첨도, 백분위수)
            - ⚠️ 비정상 세포 자동 탐지 (IQR, Z-score)
            - 🔬 형태별 분류 (Round/Elongated/Irregular/Normal)
            - 📦 크기 클러스터링 (K-means)
            """)
            
            morphology_method = st.radio(
                "이상치 탐지 방법",
                ['iqr', 'zscore'],
                format_func=lambda x: 'IQR (사분위수 범위)' if x == 'iqr' else 'Z-Score (표준편차)',
                help="IQR: 보수적, Z-score: 민감",
                key="morphology_method",
                horizontal=True
            )
            
            options['morphology'] = True
            options['morphology_method'] = morphology_method
        else:
            options['morphology'] = False
        
        st.markdown("---")
        st.markdown("#### 품질 평가")
        
        enable_quality_check = st.checkbox(
            "품질 자동 평가 및 권장사항",
            value=True,
            help="이미지 품질 메트릭 계산 및 개선 제안 생성",
            key="enable_quality"
        )
        
        if enable_quality_check:
            st.info("""
            **평가 항목:**
            - 🎯 Focus Quality (Laplacian variance)
            - 📶 Signal-to-Noise Ratio (SNR)
            - 💡 Brightness & Contrast
            - ✨ Sharpness (gradient magnitude)
            - 🌟 Illumination Uniformity
            """)
            
            options['quality_check'] = True
        else:
            options['quality_check'] = False
    
    return options


def show_quality_recommendations(quality_metrics: Dict, recommender):
    """
    Display quality recommendations
    
    Args:
        quality_metrics: Quality metrics dictionary
        recommender: QualityRecommender instance
    """
    issues = recommender.analyze_quality_issues(quality_metrics)
    
    if not issues:
        st.success("✅ 이미지 품질이 우수합니다!")
        return
    
    st.warning(f"⚠️ {len(issues)}개의 품질 이슈 발견")
    
    recommendations = recommender.generate_recommendations(issues)
    
    # Priority summary
    priority_summary = recommender.get_priority_summary()
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if priority_summary['High'] > 0:
            st.error(f"🔴 High Priority: {priority_summary['High']}개")
    
    with col2:
        if priority_summary['Medium'] > 0:
            st.warning(f"🟡 Medium Priority: {priority_summary['Medium']}개")
    
    with col3:
        if priority_summary['Low'] > 0:
            st.info(f"🟢 Low Priority: {priority_summary['Low']}개")
    
    st.markdown("---")
    
    # Show recommendations
    for issue_key, rec in recommendations.items():
        priority_icon = {
            'High': '🔴',
            'Medium': '🟡',
            'Low': '🟢'
        }[rec['priority']]
        
        with st.expander(f"{priority_icon} {rec['issue']} ({rec['priority']} Priority)", expanded=(rec['priority'] == 'High')):
            st.markdown("**권장사항:**")
            for r in rec['recommendations']:
                st.markdown(r)
            
            col1, col2 = st.columns(2)
            with col1:
                st.info(f"**예상 개선:** {rec['expected_improvement']}")
            with col2:
                st.caption(f"**처리 비용:** {rec['processing_cost']}")


def show_morphology_results(morphology_results: Dict):
    """
    Display morphology analysis results
    
    Args:
        morphology_results: Results from MorphologyAnalyzer
    """
    st.subheader("🔬 형태학적 분석 결과")
    
    # Abnormalities
    abnormalities = morphology_results['abnormalities']
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "비정상 세포",
            f"{abnormalities['count']}개",
            delta=f"{abnormalities['percentage']:.1f}%"
        )
    
    with col2:
        total_cells = sum(morphology_results['classification']['distribution'].values())
        st.metric("총 세포 수", f"{total_cells}개")
    
    with col3:
        dist_stats = morphology_results['distribution']['statistics']
        st.metric("CV (변동계수)", f"{dist_stats['cv']:.2f}")
    
    # Classification
    st.markdown("---")
    st.markdown("#### 형태 분류")
    
    classification = morphology_results['classification']
    
    import plotly.express as px
    import pandas as pd
    
    class_df = pd.DataFrame({
        'Category': list(classification['distribution'].keys()),
        'Count': list(classification['distribution'].values()),
        'Percentage': list(classification['percentages'].values())
    })
    
    fig = px.pie(
        class_df,
        values='Count',
        names='Category',
        title="세포 형태 분포",
        hover_data=['Percentage'],
        color_discrete_sequence=px.colors.qualitative.Set3
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Abnormal cells detail
    if abnormalities['abnormal_cells']:
        with st.expander(f"⚠️ 비정상 세포 상세 ({len(abnormalities['abnormal_cells'])}개)"):
            abnormal_df = pd.DataFrame(abnormalities['abnormal_cells'])
            st.dataframe(
                abnormal_df.style.background_gradient(
                    subset=['score'],
                    cmap='Reds'
                ),
                use_container_width=True
            )


if __name__ == "__main__":
    # For testing
    st.set_page_config(page_title="Deep Analysis Options", layout="wide")
    
    st.title("Deep Analysis Options Component")
    
    options = show_deep_analysis_options()
    
    st.markdown("---")
    st.subheader("Selected Options")
    st.json(options)
