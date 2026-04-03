"""
Abnormal Cell Profiler
Deep analysis of cells with abnormal morphology
"""

import pandas as pd
import numpy as np
from typing import Dict, List
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots


class AbnormalCellProfiler:
    """Profile and analyze abnormal cells in detail"""
    
    def __init__(self):
        self.abnormal_cells = []
        self.profile_results = {}
    
    def profile_abnormal_cells(
        self,
        cell_stats: pd.DataFrame,
        circularity_threshold: float = 0.5,
        area_std_multiplier: float = 2.0
    ) -> Dict:
        """
        Profile cells with abnormal morphology
        
        Args:
            cell_stats: DataFrame with cell statistics
            circularity_threshold: Cells below this are considered abnormal
            area_std_multiplier: Cells beyond this many std devs are abnormal
        
        Returns:
            Dictionary with profiling results
        """
        # Find abnormal cells
        abnormal_indices = self._identify_abnormal_cells(
            cell_stats,
            circularity_threshold,
            area_std_multiplier
        )
        
        if len(abnormal_indices) == 0:
            return {
                'count': 0,
                'percentage': 0,
                'profiles': [],
                'summary': "정상 범위 내의 세포들로 구성되어 있습니다."
            }
        
        # Profile each abnormal cell
        profiles = []
        for idx in abnormal_indices:
            profile = self._create_cell_profile(cell_stats.loc[idx])
            profiles.append(profile)
        
        self.abnormal_cells = abnormal_indices
        self.profile_results = {
            'count': len(abnormal_indices),
            'percentage': len(abnormal_indices) / len(cell_stats) * 100,
            'profiles': profiles,
            'summary': self._generate_profile_summary(profiles),
            'categories': self._categorize_abnormalities(profiles)
        }
        
        return self.profile_results
    
    def _identify_abnormal_cells(
        self,
        cell_stats: pd.DataFrame,
        circ_threshold: float,
        area_std_mult: float
    ) -> List[int]:
        """Identify abnormal cell indices"""
        abnormal = set()
        
        # Low circularity
        if 'circularity' in cell_stats.columns:
            low_circ = cell_stats[cell_stats['circularity'] < circ_threshold].index
            abnormal.update(low_circ)
        
        # Extreme area
        if 'area' in cell_stats.columns:
            mean_area = cell_stats['area'].mean()
            std_area = cell_stats['area'].std()
            
            extreme_area = cell_stats[
                (cell_stats['area'] < mean_area - area_std_mult * std_area) |
                (cell_stats['area'] > mean_area + area_std_mult * std_area)
            ].index
            abnormal.update(extreme_area)
        
        # High aspect ratio (elongated)
        if 'aspect_ratio' in cell_stats.columns:
            elongated = cell_stats[cell_stats['aspect_ratio'] > 3.0].index
            abnormal.update(elongated)
        
        return sorted(list(abnormal))
    
    def _create_cell_profile(self, cell_data: pd.Series) -> Dict:
        """Create detailed profile for a single cell"""
        profile = {
            'cell_id': cell_data.name,
            'metrics': {},
            'abnormality_reasons': [],
            'severity': 'unknown'
        }
        
        # Extract all metrics
        for key, value in cell_data.items():
            if isinstance(value, (int, float, np.number)):
                profile['metrics'][key] = float(value)
        
        # Identify reasons for abnormality
        if 'circularity' in cell_data and cell_data['circularity'] < 0.5:
            profile['abnormality_reasons'].append(
                f"낮은 원형도 ({cell_data['circularity']:.3f} < 0.5)"
            )
        
        if 'aspect_ratio' in cell_data and cell_data['aspect_ratio'] > 3.0:
            profile['abnormality_reasons'].append(
                f"높은 종횡비 ({cell_data['aspect_ratio']:.2f} > 3.0) - 매우 길쭉함"
            )
        
        # Determine severity
        severity_score = len(profile['abnormality_reasons'])
        if severity_score >= 3:
            profile['severity'] = 'High'
        elif severity_score == 2:
            profile['severity'] = 'Medium'
        else:
            profile['severity'] = 'Low'
        
        return profile
    
    def _categorize_abnormalities(self, profiles: List[Dict]) -> Dict:
        """Categorize abnormalities by type"""
        categories = {
            'Irregular Shape': 0,
            'Elongated': 0,
            'Extremely Large': 0,
            'Extremely Small': 0
        }
        
        for profile in profiles:
            reasons = ' '.join(profile['abnormality_reasons'])
            
            if '원형도' in reasons:
                categories['Irregular Shape'] += 1
            if '종횡비' in reasons or '길쭉' in reasons:
                categories['Elongated'] += 1
            if '크고' in reasons or 'large' in reasons.lower():
                categories['Extremely Large'] += 1
            if '작고' in reasons or 'small' in reasons.lower():
                categories['Extremely Small'] += 1
        
        return categories
    
    def _generate_profile_summary(self, profiles: List[Dict]) -> str:
        """Generate text summary of profiles"""
        summary = f"**비정상 세포 프로파일 요약**\n\n"
        summary += f"총 {len(profiles)}개의 비정상 세포가 탐지되었습니다.\n\n"
        
        # Severity distribution
        severity_counts = {'High': 0, 'Medium': 0, 'Low': 0}
        for p in profiles:
            severity_counts[p['severity']] += 1
        
        summary += "**심각도 분포:**\n"
        for severity, count in severity_counts.items():
            if count > 0:
                summary += f"- {severity}: {count}개\n"
        
        return summary
    
    def create_abnormal_cell_visualization(
        self,
        cell_stats: pd.DataFrame,
        abnormal_indices: List[int] = None
    ) -> go.Figure:
        """Create visualization highlighting abnormal cells"""
        if abnormal_indices is None:
            abnormal_indices = self.abnormal_cells
        
        # Mark abnormal cells
        cell_stats['is_abnormal'] = False
        cell_stats.loc[abnormal_indices, 'is_abnormal'] = True
        
        # Create scatter plot
        fig = go.Figure()
        
        # Normal cells
        normal = cell_stats[~cell_stats['is_abnormal']]
        fig.add_trace(go.Scatter(
            x=normal['area'] if 'area' in normal.columns else normal.index,
            y=normal['circularity'] if 'circularity' in normal.columns else normal.index,
            mode='markers',
            name='Normal',
            marker=dict(size=6, color='lightblue', opacity=0.6)
        ))
        
        # Abnormal cells
        abnormal = cell_stats[cell_stats['is_abnormal']]
        fig.add_trace(go.Scatter(
            x=abnormal['area'] if 'area' in abnormal.columns else abnormal.index,
            y=abnormal['circularity'] if 'circularity' in abnormal.columns else abnormal.index,
            mode='markers',
            name='Abnormal',
            marker=dict(size=10, color='red', symbol='x', line=dict(width=2, color='darkred')),
            text=[f"Cell {idx}" for idx in abnormal.index],
            hovertemplate='%{text}<br>Area: %{x:.0f}<br>Circularity: %{y:.3f}<extra></extra>'
        ))
        
        fig.update_layout(
            title="Normal vs Abnormal Cells",
            xaxis_title="Area (pixels²)",
            yaxis_title="Circularity",
            height=500,
            hovermode='closest'
        )
        
        return fig


def show_abnormal_cell_analysis(cell_stats: pd.DataFrame):
    """
    Streamlit component for abnormal cell analysis
    
    Args:
        cell_stats: DataFrame with cell statistics
    """
    import streamlit as st
    
    st.subheader("🔬 비정상 세포 심층 분석")
    
    # Configuration
    col1, col2 = st.columns(2)
    
    with col1:
        circ_threshold = st.slider(
            "원형도 임계값",
            0.0, 1.0, 0.5,
            help="이 값보다 낮은 원형도는 비정상으로 간주"
        )
    
    with col2:
        area_std_mult = st.slider(
            "면적 표준편차 배수",
            1.0, 3.0, 2.0,
            step=0.5,
            help="평균에서 이 배수만큼 벗어난 면적은 비정상으로 간주"
        )
    
    # Profile abnormal cells
    profiler = AbnormalCellProfiler()
    results = profiler.profile_abnormal_cells(
        cell_stats,
        circularity_threshold=circ_threshold,
        area_std_multiplier=area_std_mult
    )
    
    # Display summary
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric(
            "비정상 세포 수",
            f"{results['count']}개",
            delta=f"{results['percentage']:.1f}%"
        )
    
    with col2:
        if results['count'] > 0:
            severity_counts = {}
            for p in results['profiles']:
                severity = p['severity']
                severity_counts[severity] = severity_counts.get(severity, 0) + 1
            
            severity_str = ", ".join([f"{k}: {v}" for k, v in severity_counts.items()])
            st.metric("심각도 분포", severity_str)
    
    st.markdown(results['summary'])
    
    if results['count'] > 0:
        # Visualization
        st.markdown("---")
        st.markdown("### 시각화")
        
        fig = profiler.create_abnormal_cell_visualization(cell_stats)
        st.plotly_chart(fig, use_container_width=True)
        
        # Category breakdown
        st.markdown("---")
        st.markdown("### 비정상 유형 분포")
        
        categories_df = pd.DataFrame(
            list(results['categories'].items()),
            columns=['Category', 'Count']
        )
        
        fig_cat = px.bar(
            categories_df,
            x='Category',
            y='Count',
            title="비정상 유형별 분포",
            color='Count',
            color_continuous_scale='Reds'
        )
        st.plotly_chart(fig_cat, use_container_width=True)
        
        # Detailed list
        with st.expander(f"📋 비정상 세포 상세 목록 ({results['count']}개)"):
            profiles_data = []
            for p in results['profiles']:
                profiles_data.append({
                    'Cell ID': p['cell_id'],
                    'Severity': p['severity'],
                    'Reasons': ', '.join(p['abnormality_reasons']),
                    'Area': p['metrics'].get('area', 'N/A'),
                    'Circularity': p['metrics'].get('circularity', 'N/A')
                })
            
            profiles_df = pd.DataFrame(profiles_data)
            st.dataframe(profiles_df, use_container_width=True)


if __name__ == "__main__":
    # For testing
    import streamlit as st
    
    st.set_page_config(page_title="Abnormal Cell Profiler", layout="wide")
    
    # Generate test data
    np.random.seed(42)
    n = 500
    
    test_data = pd.DataFrame({
        'area': np.random.normal(1000, 200, n),
        'circularity': np.random.beta(8, 2, n),
        'perimeter': np.random.normal(150, 30, n),
        'aspect_ratio': np.random.gamma(2, 0.5, n)
    })
    
    # Add some abnormal cells
    test_data.loc[10:15, 'circularity'] = 0.3
    test_data.loc[20:25, 'aspect_ratio'] = 4.5
    test_data.loc[30:32, 'area'] = 2500
    
    show_abnormal_cell_analysis(test_data)
