"""
Performance Analytics Dashboard
Comprehensive analysis dashboard with KPIs, trends, and reports
"""

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Will be imported from parent
# from utils.analysis_db import AnalysisDatabase


def show_performance_analytics():
    """Performance analytics dashboard"""
    
    st.title("📈 성과지표 분석")
    st.markdown("""
    전체 실험 데이터를 통합 분석하고 트렌드를 시각화합니다.  
    **기능:** KPI 지표, 시간별 추이, 실험 비교, 통계 분석, 리포트 생성
    """)
    
    st.markdown("---")
    
    # Try to load real data from AnalysisDatabase first
    all_data = []
    using_real_data = False
    
    try:
        from utils.analysis_db import AnalysisDatabase
        db = AnalysisDatabase()
        all_data = db.get_all_analyses(limit=500)
        if len(all_data) > 0:
            using_real_data = True
            st.success(f"✅ 실제 분석 기록 {len(all_data)}건 로드됨 (from {db.db_path})")
    except Exception as e:
        pass  # DB not available, fall through to mock
    
    # Fallback to mock data if no real data available
    if not using_real_data:
        st.warning("⚠️ 실제 분석 기록이 없습니다. 데모 데이터를 표시합니다. 이미지 분석을 수행하면 실시간 데이터로 전환됩니다.")
        all_data = generate_mock_analysis_data(100)
    
    if len(all_data) == 0:
        st.warning("⚠️ 분석 데이터가 없습니다")
        st.info("먼저 이미지 분석을 수행하여 데이터를 생성하세요.")
        
        with st.expander("💡 사용 방법"):
            st.markdown("""
            1. **이미지 분석** 페이지에서 세포 이미지 분석 수행
            2. 결과가 자동으로 데이터베이스에 저장됨
            3. 여러 분석 수행 후 이 페이지에서 통합 분석
            4. 트렌드, 패턴, 인사이트 확인
            """)
        return
    
    # Convert to DataFrame
    df = pd.DataFrame(all_data)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['date'] = df['timestamp'].dt.date
    
    # 1. KPI Overview Section
    st.subheader("📊 핵심 성과 지표")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_analyses = len(df)
        recent_7days = len(df[df['timestamp'] > datetime.now() - timedelta(days=7)])
        st.metric(
            "총 분석 횟수",
            f"{total_analyses}",
            delta=f"+{recent_7days} (최근 7일)",
            help="전체 수행된 분석 횟수"
        )
    
    with col2:
        total_cells = df['num_cells'].sum()
        avg_cells = df['num_cells'].mean()
        st.metric(
            "분석된 총 세포",
            f"{total_cells:,}",
            delta=f"평균 {avg_cells:.0f}개/분석",
            help="전체 분석에서 탐지된 세포 수"
        )
    
    with col3:
        avg_quality = df['quality_score'].mean()
        quality_trend = df.tail(10)['quality_score'].mean() - df.head(10)['quality_score'].mean()
        st.metric(
            "평균 품질 점수",
            f"{avg_quality:.3f}",
            delta=f"{quality_trend:+.3f} (최근 트렌드)",
            delta_color="normal" if quality_trend >= 0 else "inverse",
            help="전체 이미지 품질 평균"
        )
    
    with col4:
        unique_experiments = df['experiment_name'].nunique()
        unique_cell_lines = df['cell_line'].nunique()
        st.metric(
            "실험 / 세포주",
            f"{unique_experiments} / {unique_cell_lines}",
            help="고유한 실험 및 세포주 개수"
        )
    
    # 2. Filters Section
    st.markdown("---")
    st.subheader("🔍 데이터 필터")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        experiment_options = ['전체'] + sorted(df['experiment_name'].dropna().unique().tolist())
        experiment_filter = st.selectbox(
            "실험 선택",
            experiment_options
        )
    
    with col2:
        cell_line_options = ['전체'] + sorted(df['cell_line'].dropna().unique().tolist())
        cell_line_filter = st.selectbox(
            "세포주 선택",
            cell_line_options
        )
    
    with col3:
        date_range = st.date_input(
            "날짜 범위",
            value=(
                (datetime.now() - timedelta(days=30)).date(),
                datetime.now().date()
            ),
            max_value=datetime.now().date()
        )
    
    with col4:
        quality_threshold = st.slider(
            "최소 품질 점수",
            0.0, 1.0, 0.0,
            step=0.05,
            help="이 값 이상의 품질만 표시"
        )
    
    # Apply filters
    filtered_df = df.copy()
    
    if experiment_filter != '전체':
        filtered_df = filtered_df[filtered_df['experiment_name'] == experiment_filter]
    
    if cell_line_filter != '전체':
        filtered_df = filtered_df[filtered_df['cell_line'] == cell_line_filter]
    
    if len(date_range) == 2:
        start_date, end_date = date_range
        filtered_df = filtered_df[
            (filtered_df['date'] >= start_date) &
            (filtered_df['date'] <= end_date)
        ]
    
    filtered_df = filtered_df[filtered_df['quality_score'] >= quality_threshold]
    
    # Show filter results
    if len(filtered_df) < len(df):
        st.info(f"ℹ️ 필터 적용: {len(df)}개 → {len(filtered_df)}개 레코드")
    
    if len(filtered_df) == 0:
        st.warning("⚠️ 필터 조건에 맞는 데이터가 없습니다")
        return
    
    # 3. Trend Analysis Section
    st.markdown("---")
    st.subheader("📊 트렌드 분석")
    
    tab1, tab2, tab3, tab4 = st.tabs([
        "📈 시간별 추이",
        "📊 실험별 비교",
        "🎯 품질 분석",
        "📋 상세 통계"
    ])
    
    with tab1:
        st.markdown("### 시간별 세포 수 추이")
        
        # Aggregate by date
        daily_stats = filtered_df.groupby('date').agg({
            'num_cells': ['sum', 'mean'],
            'quality_score': 'mean'
        }).reset_index()
        
        daily_stats.columns = ['date', 'total_cells', 'avg_cells', 'avg_quality']
        
        # Create dual-axis chart
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        
        fig.add_trace(
            go.Scatter(
                x=daily_stats['date'],
                y=daily_stats['total_cells'],
                name="총 세포 수",
                mode='lines+markers',
                line=dict(color='#3b82f6', width=2)
            ),
            secondary_y=False
        )
        
        fig.add_trace(
            go.Scatter(
                x=daily_stats['date'],
                y=daily_stats['avg_quality'],
                name="평균 품질",
                mode='lines+markers',
                line=dict(color='#10b981', width=2, dash='dash')
            ),
            secondary_y=True
        )
        
        fig.update_xaxes(title_text="날짜")
        fig.update_yaxes(title_text="총 세포 수", secondary_y=False)
        fig.update_yaxes(title_text="평균 품질 점수", secondary_y=True)
        
        fig.update_layout(
            height=400,
            hovermode='x unified',
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Insights
        trend = daily_stats['total_cells'].diff().mean()
        if trend > 0:
            st.success(f"📈 상승 트렌드: 평균 일일 +{trend:.0f}개 세포")
        elif trend < 0:
            st.warning(f"📉 하락 트렌드: 평균 일일 {trend:.0f}개 세포")
        else:
            st.info("➡️ 안정적 트렌드")
    
    with tab2:
        st.markdown("### 실험별 성과 비교")
        
        # Aggregate by experiment
        exp_stats = filtered_df.groupby('experiment_name').agg({
            'num_cells': ['mean', 'std', 'count'],
            'quality_score': 'mean',
            'mean_area': 'mean'
        }).reset_index()
        
        exp_stats.columns = ['experiment', 'avg_cells', 'std_cells', 'count', 'avg_quality', 'avg_area']
        exp_stats = exp_stats.sort_values('avg_cells', ascending=False)
        
        # Bar chart
        fig = px.bar(
            exp_stats,
            x='experiment',
            y='avg_cells',
            color='avg_quality',
            title="실험별 평균 세포 수",
            labels={
                'experiment': '실험',
                'avg_cells': '평균 세포 수',
                'avg_quality': '평균 품질'
            },
            color_continuous_scale='Viridis',
            text='count'
        )
        
        fig.update_traces(texttemplate='n=%{text}', textposition='outside')
        fig.update_layout(height=400)
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Summary table
        st.markdown("**요약 표:**")
        st.dataframe(
            exp_stats.style.format({
                'avg_cells': '{:.0f}',
                'std_cells': '{:.0f}',
                'avg_quality': '{:.3f}',
                'avg_area': '{:.1f}'
            }).background_gradient(subset=['avg_quality'], cmap='RdYlGn'),
            use_container_width=True
        )
    
    with tab3:
        st.markdown("### 품질 점수 분포")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Histogram
            fig = px.histogram(
                filtered_df,
                x='quality_score',
                nbins=30,
                title="품질 점수 분포",
                labels={'quality_score': '품질 점수'},
                color_discrete_sequence=['#3b82f6']
            )
            fig.update_layout(showlegend=False, height=350)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Box plot by experiment
            fig = px.box(
                filtered_df,
                x='experiment_name',
                y='quality_score',
                title="실험별 품질 분포",
                labels={
                    'experiment_name': '실험',
                    'quality_score': '품질 점수'
                },
                color='experiment_name'
            )
            fig.update_layout(showlegend=False, height=350)
            st.plotly_chart(fig, use_container_width=True)
        
        # Quality grades
        quality_grades = pd.cut(
            filtered_df['quality_score'],
            bins=[0, 0.6, 0.75, 0.85, 1.0],
            labels=['Poor', 'Acceptable', 'Good', 'Excellent']
        ).value_counts()
        
        col1, col2, col3, col4 = st.columns(4)
        for i, (grade, count) in enumerate(quality_grades.items()):
            with [col1, col2, col3, col4][i]:
                pct = count / len(filtered_df) * 100
                st.metric(grade, f"{count} ({pct:.1f}%)")
    
    with tab4:
        st.markdown("### 상세 통계")
        
        # Statistical summary
        stats_cols = ['num_cells', 'mean_area', 'mean_circularity', 'quality_score']
        stats_summary = filtered_df[stats_cols].describe()
        
        st.dataframe(
            stats_summary.style.format("{:.2f}"),
            use_container_width=True
        )
        
        # Correlation matrix
        st.markdown("**상관관계 분석:**")
        corr = filtered_df[stats_cols].corr()
        
        fig = px.imshow(
            corr,
            text_auto='.2f',
            aspect='auto',
            color_continuous_scale='RdBu_r',
            zmin=-1, zmax=1,
            labels=dict(color="Correlation")
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    # 4. Export and Reports
    st.markdown("---")
    st.subheader("📥 리포트 생성 및 내보내기")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # CSV export
        csv = filtered_df.to_csv(index=False, encoding='utf-8-sig')
        st.download_button(
            "📄 CSV 다운로드",
            csv.encode('utf-8-sig'),
            f"analytics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            "text/csv",
            use_container_width=True
        )
    
    with col2:
        # Excel export
        import io
        buffer = io.BytesIO()
        with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
            filtered_df.to_excel(writer, index=False, sheet_name='Data')
            exp_stats.to_excel(writer, index=False, sheet_name='Experiment Summary')
        
        st.download_button(
            "📊 Excel 다운로드",
            buffer.getvalue(),
            f"analytics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True
        )
    
    with col3:
        # PDF report (placeholder)
        st.button(
            "📑 PDF 리포트 생성",
            help="실제 구현에서 PDF 생성 (reportlab/weasyprint)",
            disabled=True,
            use_container_width=True
        )


def generate_mock_analysis_data(n=100):
    """Generate mock analysis data for demonstration"""
    np.random.seed(42)
    
    experiments = ['Exp-A', 'Exp-B', 'Exp-C', 'Control']
    cell_lines = ['HUVEC', 'HEK293', 'MCF7']
    
    data = []
    base_date = datetime.now() - timedelta(days=30)
    
    for i in range(n):
        data.append({
            'id': i + 1,
            'timestamp': base_date + timedelta(
                days=np.random.randint(0, 30),
                hours=np.random.randint(0, 24)
            ),
            'image_name': f"cell_image_{i+1:03d}.tif",
            'experiment_name': np.random.choice(experiments),
            'cell_line': np.random.choice(cell_lines),
            'num_cells': int(np.random.normal(800, 200)),
            'mean_area': np.random.normal(1200, 300),
            'mean_circularity': np.random.beta(8, 2),
            'quality_score': np.random.beta(9, 2)
        })
    
    return data


if __name__ == "__main__":
    # For standalone testing
    st.set_page_config(page_title="성과지표 분석", page_icon="📈", layout="wide")
    show_performance_analytics()
