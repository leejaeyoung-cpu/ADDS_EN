"""
Show Dashboard Page
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
from data.data_integrator import DataIntegrator
from utils.system_utils import clamp
import plotly.express as px
import plotly.graph_objects as go


def show_dashboard():
    """Comprehensive experiment dashboard with condition analysis"""
    st.header("📈 실험 대시보드")
    st.markdown("실험 조건별 비교 분석 및 통계 검정")
    
    try:
        db = AnalysisDatabase()
        all_analyses = db.get_all_analyses()
        
        if not all_analyses:
            st.info("📌 아직 분석 데이터가 없습니다. 이미지 분석을 시작하세요!")
            return
        
        # Convert to DataFrame
        df = pd.DataFrame(all_analyses)
        
        # === SUMMARY METRICS ===
        st.markdown("### 📊 전체 요약")
        metric_cols = st.columns(4)
        
        with metric_cols[0]:
            st.metric("전체 분석 수", len(df))
        
        with metric_cols[1]:
            if 'experiment_name' in df.columns:
                unique_experiments = df['experiment_name'].nunique()
            else:
                unique_experiments = 0
            st.metric("실험 수", unique_experiments)
        
        with metric_cols[2]:
            avg_cells = df['num_cells'].mean() if 'num_cells' in df.columns else 0
            st.metric("평균 세포 수", f"{avg_cells:.0f}")
        
        with metric_cols[3]:
            avg_quality = df['quality_score'].mean() if 'quality_score' in df.columns else 0
            st.metric("평균 품질 점수", f"{avg_quality:.2f}")
        
        st.markdown("---")
        
        # === EXPERIMENT SELECTION ===
        # Check if experiment columns exist
        if 'experiment_name' in df.columns and 'condition' in df.columns:
            experiments_with_conditions = df[
                (df['experiment_name'].notna()) & 
                (df['condition'].notna())
            ]
        else:
            experiments_with_conditions = pd.DataFrame()
        
        if len(experiments_with_conditions) > 0:
            st.markdown("### 🔬 실험별 분석")
            
            # Experiment selector
            unique_exp = experiments_with_conditions['experiment_name'].unique()
            selected_exp = st.selectbox(
                "분석할 실험 선택",
                options=unique_exp,
                index=0
            )
            
            # Filter data for selected experiment
            exp_data = experiments_with_conditions[
                experiments_with_conditions['experiment_name'] == selected_exp
            ].copy()
            
            if len(exp_data) > 0:
                # === EXPERIMENT INFO ===
                info_col1, info_col2, info_col3 = st.columns(3)
                
                with info_col1:
                    cell_line = exp_data['cell_line'].iloc[0] if exp_data['cell_line'].notna().any() else "N/A"
                    st.metric("세포주", cell_line)
                
                with info_col2:
                    treatment = exp_data['treatment'].iloc[0] if exp_data['treatment'].notna().any() else "N/A"
                    st.metric("처리", treatment)
                
                with info_col3:
                    concentration = exp_data['concentration'].iloc[0] if exp_data['concentration'].notna().any() else "N/A"
                    st.metric("농도", concentration)
                
                st.markdown("---")
                
                # === TABS FOR DIFFERENT ANALYSES ===
                tab1, tab2, tab3 = st.tabs(["📈 시간 추이", "📊 조건별 비교", "🔬 통계 검정"])
                
                with tab1:
                    st.markdown("#### 시간에 따른 변화")
                    
                    # Time-course line plots
                    metrics_to_plot = ['num_cells', 'mean_circularity', 'quality_score']
                    metric_names = {
                        'num_cells': '세포 수',
                        'mean_circularity': '원형도',
                        'quality_score': '품질 점수'
                    }
                    
                    for metric in metrics_to_plot:
                        if metric in exp_data.columns:
                            # Group by condition and calculate mean
                            condition_summary = exp_data.groupby('condition')[metric].agg(['mean', 'std', 'count']).reset_index()
                            
                            # Sort conditions (Control first, then by hour)
                            def sort_key(cond):
                                if cond == 'Control':
                                    return 0
                                match = re.search(r'(\d+)', str(cond))
                                return int(match.group(1)) if match else 999
                            
                            condition_summary['sort_key'] = condition_summary['condition'].apply(sort_key)
                            condition_summary = condition_summary.sort_values('sort_key')
                            
                            # Create line plot
                            fig = go.Figure()
                            
                            fig.add_trace(go.Scatter(
                                x=condition_summary['condition'],
                                y=condition_summary['mean'],
                                mode='lines+markers',
                                name=metric_names[metric],
                                error_y=dict(
                                    type='data',
                                    array=condition_summary['std'],
                                    visible=True
                                ),
                                line=dict(width=3),
                                marker=dict(size=10)
                            ))
                            
                            fig.update_layout(
                                title=f"{metric_names[metric]} 시간 추이",
                                xaxis_title="실험 조건",
                                yaxis_title=metric_names[metric],
                                height=300,
                                showlegend=False
                            )
                            
                            st.plotly_chart(fig, use_container_width=True)
                
                with tab2:
                    st.info("조건별 비교 기능은 준비 중입니다.")
                    st.markdown("#### 조건별 비교")
                    
                    # Box plots for each metric
                    comp_col1, comp_col2 = st.columns(2)
                    
                    with comp_col1:
                        # Cell count box plot
                        fig_cells = px.box(
                            exp_data,
                            x='condition',
                            y='num_cells',
                            title="조건별 세포 수 분포",
                            labels={'condition': '조건', 'num_cells': '세포 수'},
                            color='condition'
                        )
                        fig_cells.update_layout(showlegend=False, height=350)
                        st.plotly_chart(fig_cells, use_container_width=True)
                        
                        # Quality score box plot
                        fig_quality = px.box(
                            exp_data,
                            x='condition',
                            y='quality_score',
                            title="조건별 품질 점수 분포",
                            labels={'condition': '조건', 'quality_score': '품질 점수'},
                            color='condition'
                        )
                        fig_quality.update_layout(showlegend=False, height=350)
                        st.plotly_chart(fig_quality, use_container_width=True)
                    
                    with comp_col2:
                        # Circularity box plot
                        fig_circ = px.box(
                            exp_data,
                            x='condition',
                            y='mean_circularity',
                            title="조건별 원형도 분포",
                            labels={'condition': '조건', 'mean_circularity': '원형도'},
                            color='condition'
                        )
                        fig_circ.update_layout(showlegend=False, height=350)
                        st.plotly_chart(fig_circ, use_container_width=True)
                        
                        # Mean area box plot
                        fig_area = px.box(
                            exp_data,
                            x='condition',
                            y='mean_area',
                            title="조건별 평균 면적 분포",
                            labels={'condition': '조건', 'mean_area': '평균 면적 (px²)'},
                            color='condition'
                        )
                        fig_area.update_layout(showlegend=False, height=350)
                        st.plotly_chart(fig_area, use_container_width=True)
                    
                    # Summary table
                    st.markdown("#### 📋 조건별 요약 통계")
                    summary_stats = exp_data.groupby('condition').agg({
                        'num_cells': ['mean', 'std', 'count'],
                        'mean_circularity': ['mean', 'std'],
                        'quality_score': ['mean', 'std']
                    }).round(2)
                    
                    summary_stats.columns = ['_'.join(col).strip() for col in summary_stats.columns.values]
                    summary_stats = summary_stats.reset_index()
                    st.dataframe(summary_stats, use_container_width=True)
                
                with tab3:
                    st.markdown("#### 통계 검정")
                    st.caption("조건 간 유의성 검정 (p < 0.05 유의)")
                    
                    # Check if we have enough data for statistical tests
                    conditions = exp_data['condition'].unique()
                    
                    if len(conditions) >= 2:
                        try:
                            from scipy import stats
                            
                            # ANOVA for each metric
                            st.markdown("##### One-way ANOVA 결과")
                            
                            metrics_test = {
                                'num_cells': '세포 수',
                                'mean_circularity': '원형도',
                                'mean_area': '평균 면적',
                                'quality_score': '품질 점수'
                            }
                            
                            anova_results = []
                            
                            for metric, name in metrics_test.items():
                                if metric in exp_data.columns:
                                    # Prepare groups
                                    groups = [exp_data[exp_data['condition'] == cond][metric].dropna() 
                                             for cond in conditions]
                                    
                                    # Filter out empty groups
                                    groups = [g for g in groups if len(g) > 0]
                                    
                                    if len(groups) >= 2:
                                        # Perform ANOVA
                                        f_stat, p_value = stats.f_oneway(*groups)
                                        
                                        significance = "✅ 유의함" if p_value < 0.05 else "❌ 유의하지 않음"
                                        
                                        anova_results.append({
                                            '메트릭': name,
                                            'F-통계량': f"{f_stat:.3f}",
                                            'p-value': f"{p_value:.4f}",
                                            '유의성': significance
                                        })
                            
                            if anova_results:
                                st.dataframe(pd.DataFrame(anova_results), use_container_width=True, hide_index=True)
                                
                                st.info("💡 p-value < 0.05인 경우, 조건 간 통계적으로 유의한 차이가 있습니다.")
                            
                            # Pairwise t-tests
                            if len(conditions) >= 2:
                                st.markdown("##### Pairwise t-test")
                                
                                test_metric = st.selectbox(
                                    "검정할 메트릭 선택",
                                    options=list(metrics_test.keys()),
                                    format_func=lambda x: metrics_test[x]
                                )
                                
                                pairwise_results = []
                                
                                for i, cond1 in enumerate(conditions):
                                    for cond2 in conditions[i+1:]:
                                        group1 = exp_data[exp_data['condition'] == cond1][test_metric].dropna()
                                        group2 = exp_data[exp_data['condition'] == cond2][test_metric].dropna()
                                        
                                        if len(group1) > 1 and len(group2) > 1:
                                            t_stat, p_val = stats.ttest_ind(group1, group2)
                                            
                                            significance = "✅ 유의함" if p_val < 0.05 else "❌ 유의하지 않음"
                                            
                                            pairwise_results.append({
                                                '비교': f"{cond1} vs {cond2}",
                                                't-통계량': f"{t_stat:.3f}",
                                                'p-value': f"{p_val:.4f}",
                                                '유의성': significance
                                            })
                                
                                if pairwise_results:
                                    st.dataframe(pd.DataFrame(pairwise_results), use_container_width=True, hide_index=True)
                        
                        except ImportError:
                            st.warning("⚠️ scipy 패키지가 필요합니다: `pip install scipy`")
                        except Exception as e:
                            st.error(f"통계 검정 중 오류: {str(e)}")
                    else:
                        st.info("통계 검정을 위해서는 최소 2개 이상의 조건이 필요합니다.")
        
        else:
            st.info("📌 실험 조건 정보가 있는 분석 데이터가 없습니다. 이미지 분석 시 실험 정보를 입력하세요.")
    
    except Exception as e:
        st.error(f"대시보드 로드 실패: {str(e)}")
        import traceback
        with st.expander("오류 상세"):
            st.code(traceback.format_exc())
