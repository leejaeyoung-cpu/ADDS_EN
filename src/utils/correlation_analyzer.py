"""
Correlation Analyzer
Analyze relationships between cell metrics and quality scores
"""

import pandas as pd
import numpy as np
from scipy import stats
from typing import Dict, List, Tuple
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots


class CorrelationAnalyzer:
    """Analyze correlations between cell metrics"""
    
    def __init__(self):
        self.correlation_matrix = None
        self.significant_correlations = []
    
    def analyze_correlations(
        self,
        cell_stats: pd.DataFrame,
        metrics: List[str] = None
    ) -> Dict:
        """
        Analyze correlations between cell metrics
        
        Args:
            cell_stats: DataFrame with cell statistics
            metrics: List of metrics to analyze (default: all numeric)
        
        Returns:
            Dictionary with correlation results
        """
        if metrics is None:
            # Use all numeric columns
            metrics = cell_stats.select_dtypes(include=[np.number]).columns.tolist()
        
        # Compute correlation matrix
        corr_matrix = cell_stats[metrics].corr(method='pearson')
        self.correlation_matrix = corr_matrix
        
        # Find significant correlations
        significant = self._find_significant_correlations(
            cell_stats[metrics],
            threshold=0.5
        )
        self.significant_correlations = significant
        
        # Compute p-values
        p_values = self._compute_p_values(cell_stats[metrics])
        
        return {
            'correlation_matrix': corr_matrix,
            'significant_pairs': significant,
            'p_values': p_values,
            'summary': self._generate_summary(corr_matrix, significant)
        }
    
    def _find_significant_correlations(
        self,
        data: pd.DataFrame,
        threshold: float = 0.5
    ) -> List[Dict]:
        """Find correlations above threshold"""
        corr_matrix = data.corr()
        significant = []
        
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr_value = corr_matrix.iloc[i, j]
                
                if abs(corr_value) >= threshold:
                    metric1 = corr_matrix.columns[i]
                    metric2 = corr_matrix.columns[j]
                    
                    # Compute p-value
                    corr, p_value = stats.pearsonr(
                        data[metric1].dropna(),
                        data[metric2].dropna()
                    )
                    
                    significant.append({
                        'metric1': metric1,
                        'metric2': metric2,
                        'correlation': corr_value,
                        'p_value': p_value,
                        'strength': self._interpret_correlation(abs(corr_value)),
                        'direction': 'positive' if corr_value > 0 else 'negative'
                    })
        
        # Sort by absolute correlation
        significant.sort(key=lambda x: abs(x['correlation']), reverse=True)
        
        return significant
    
    def _compute_p_values(self, data: pd.DataFrame) -> pd.DataFrame:
        """Compute p-values for all pairs"""
        metrics = data.columns
        n_metrics = len(metrics)
        p_matrix = np.zeros((n_metrics, n_metrics))
        
        for i in range(n_metrics):
            for j in range(n_metrics):
                if i != j:
                    _, p_value = stats.pearsonr(
                        data.iloc[:, i].dropna(),
                        data.iloc[:, j].dropna()
                    )
                    p_matrix[i, j] = p_value
                else:
                    p_matrix[i, j] = 0
        
        return pd.DataFrame(p_matrix, index=metrics, columns=metrics)
    
    def _interpret_correlation(self, corr: float) -> str:
        """Interpret correlation strength"""
        if corr >= 0.7:
            return 'Strong'
        elif corr >= 0.5:
            return 'Moderate'
        elif corr >= 0.3:
            return 'Weak'
        else:
            return 'Very Weak'
    
    def _generate_summary(
        self,
        corr_matrix: pd.DataFrame,
        significant: List[Dict]
    ) -> str:
        """Generate text summary"""
        summary = f"**상관관계 분석 요약**\n\n"
        summary += f"- 분석 지표 수: {len(corr_matrix.columns)}개\n"
        summary += f"- 유의미한 상관관계: {len(significant)}쌍\n\n"
        
        if significant:
            summary += "**주요 발견:**\n"
            for i, pair in enumerate(significant[:5], 1):
                direction = "양의" if pair['direction'] == 'positive' else "음의"
                summary += (
                    f"{i}. **{pair['metric1']}** ↔ **{pair['metric2']}**: "
                    f"{direction} {pair['strength']} 상관관계 "
                    f"(r={pair['correlation']:.3f}, p<{pair['p_value']:.4f})\n"
                )
        
        return summary
    
    def create_correlation_heatmap(
        self,
        corr_matrix: pd.DataFrame = None
    ) -> go.Figure:
        """Create interactive correlation heatmap"""
        if corr_matrix is None:
            corr_matrix = self.correlation_matrix
        
        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            colorscale='RdBu_r',
            zmid=0,
            zmin=-1,
            zmax=1,
            text=corr_matrix.values,
            texttemplate='%{text:.2f}',
            textfont={"size": 10},
            colorbar=dict(title="Correlation")
        ))
        
        fig.update_layout(
            title="상관관계 히트맵",
            xaxis_title="Metrics",
            yaxis_title="Metrics",
            height=500,
            width=500
        )
        
        return fig
    
    def create_scatter_matrix(
        self,
        data: pd.DataFrame,
        metrics: List[str] = None
    ) -> go.Figure:
        """Create scatter plot matrix"""
        if metrics is None:
            metrics = data.select_dtypes(include=[np.number]).columns.tolist()
        
        # Limit to top 4 metrics to avoid overcrowding
        metrics = metrics[:4]
        
        fig = px.scatter_matrix(
            data,
            dimensions=metrics,
            title="Scatter Plot Matrix",
            height=700,
            color=metrics[0] if metrics else None
        )
        
        fig.update_traces(diagonal_visible=False, showupperhalf=False)
        
        return fig
    
    def analyze_metric_relationships(
        self,
        cell_stats: pd.DataFrame,
        metric1: str,
        metric2: str
    ) -> Dict:
        """
        Detailed analysis of relationship between two metrics
        
        Returns:
            Dictionary with relationship analysis
        """
        data1 = cell_stats[metric1].dropna()
        data2 = cell_stats[metric2].dropna()
        
        # Pearson correlation
        pearson_r, pearson_p = stats.pearsonr(data1, data2)
        
        # Spearman correlation (non-parametric)
        spearman_r, spearman_p = stats.spearmanr(data1, data2)
        
        # Linear regression
        slope, intercept, r_value, p_value, std_err = stats.linregress(data1, data2)
        
        return {
            'pearson': {
                'correlation': pearson_r,
                'p_value': pearson_p,
                'strength': self._interpret_correlation(abs(pearson_r))
            },
            'spearman': {
                'correlation': spearman_r,
                'p_value': spearman_p
            },
            'regression': {
                'slope': slope,
                'intercept': intercept,
                'r_squared': r_value**2,
                'equation': f"y = {slope:.3f}x + {intercept:.3f}"
            }
        }


def show_correlation_analysis(cell_stats: pd.DataFrame, quality_score: float = None):
    """
    Streamlit component for correlation analysis
    
    Args:
        cell_stats: DataFrame with cell statistics
        quality_score: Overall quality score (optional)
    """
    import streamlit as st
    
    st.subheader("📊 상관관계 분석")
    
    # Add quality score to dataframe if provided
    if quality_score is not None:
        cell_stats['quality_score'] = quality_score
    
    # Initialize analyzer
    analyzer = CorrelationAnalyzer()
    
    # Run analysis
    results = analyzer.analyze_correlations(cell_stats)
    
    # Display summary
    st.markdown(results['summary'])
    
    # Show correlation heatmap
    st.markdown("---")
    st.markdown("### 상관관계 히트맵")
    
    fig_heatmap = analyzer.create_correlation_heatmap(results['correlation_matrix'])
    st.plotly_chart(fig_heatmap, use_container_width=True)
    
    # Significant correlations table
    if results['significant_pairs']:
        st.markdown("---")
        st.markdown("### 유의미한 상관관계")
        
        sig_df = pd.DataFrame(results['significant_pairs'])
        st.dataframe(
            sig_df.style.background_gradient(
                subset=['correlation'],
                cmap='RdYlGn',
                vmin=-1,
                vmax=1
            ),
            use_container_width=True
        )
    
    # Detailed pairwise analysis
    st.markdown("---")
    st.markdown("### 상세 분석")
    
    metrics = cell_stats.select_dtypes(include=[np.number]).columns.tolist()
    
    col1, col2 = st.columns(2)
    
    with col1:
        metric1 = st.selectbox("지표 1", metrics, key="corr_metric1")
    
    with col2:
        metric2 = st.selectbox("지표 2", metrics, index=1 if len(metrics) > 1 else 0, key="corr_metric2")
    
    if metric1 and metric2 and metric1 != metric2:
        # Analyze relationship
        relationship = analyzer.analyze_metric_relationships(
            cell_stats,
            metric1,
            metric2
        )
        
        # Show scatter plot with regression line
        fig_scatter = px.scatter(
            cell_stats,
            x=metric1,
            y=metric2,
            title=f"{metric1} vs {metric2}",
            trendline="ols",
            trendline_color_override="red"
        )
        st.plotly_chart(fig_scatter, use_container_width=True)
        
        # Show relationship stats
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "Pearson 상관계수",
                f"{relationship['pearson']['correlation']:.3f}",
                delta=relationship['pearson']['strength']
            )
        
        with col2:
            st.metric(
                "R² (결정계수)",
                f"{relationship['regression']['r_squared']:.3f}"
            )
        
        with col3:
            st.metric(
                "P-value",
                f"{relationship['pearson']['p_value']:.4f}",
                delta="유의함" if relationship['pearson']['p_value'] < 0.05 else "유의하지 않음"
            )
        
        st.info(f"**회귀 방정식:** {relationship['regression']['equation']}")


if __name__ == "__main__":
    # For testing
    import streamlit as st
    
    st.set_page_config(page_title="Correlation Analysis", layout="wide")
    
    # Generate test data
    np.random.seed(42)
    n = 500
    
    test_data = pd.DataFrame({
        'area': np.random.normal(1000, 200, n),
        'circularity': np.random.beta(8, 2, n),
        'perimeter': np.random.normal(150, 30, n),
        'aspect_ratio': np.random.gamma(2, 0.5, n)
    })
    
    # Add some correlations
    test_data['area'] = test_data['area'] + test_data['perimeter'] * 2
    test_data['quality_score'] = test_data['circularity'] * 0.5 + np.random.normal(0.4, 0.1, n)
    
    show_correlation_analysis(test_data, quality_score=0.85)
