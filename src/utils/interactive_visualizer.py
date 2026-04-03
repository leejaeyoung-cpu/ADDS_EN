"""
Enhanced Interactive Visualizer
Interactive cell visualization with zoom, pan, and selection using Plotly
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np
import pandas as pd
from typing import Dict, Optional, List


def create_interactive_image_view(
    image: np.ndarray,
    masks: np.ndarray = None,
    cell_stats: pd.DataFrame = None,
    show_overlay: bool = True,
    show_centroids: bool = True,
    show_contours: bool = False
):
    """
    Create interactive image view with Plotly
    
    Args:
        image: Original image
        masks: Cell masks
        cell_stats: DataFrame with cell statistics
        show_overlay: Show mask overlay
        show_centroids: Show cell centroids
        show_contours: Show cell contours
    
    Returns:
        Plotly figure
    """
    fig = go.Figure()
    
    # Add base image
    fig.add_trace(go.Image(
        z=image,
        name="Original Image",
        hovertemplate='x: %{x}<br>y: %{y}<extra></extra>'
    ))
    
    # Add mask overlay if provided
    if masks is not None and show_overlay:
        # Create colored mask (transparent where no mask)
        from skimage import color
        colored_masks = color.label2rgb(masks, bg_label=0, alpha=0.3)
        
        fig.add_trace(go.Image(
            z=colored_masks,
            name="Mask Overlay",
            opacity=0.5,
            visible=show_overlay
        ))
    
    # Add cell centroids if provided
    if cell_stats is not None and show_centroids and 'centroid_y' in cell_stats.columns:
        hover_text = []
        for idx, row in cell_stats.iterrows():
            text = (
                f"Cell {idx}<br>"
                f"Area: {row.get('area', 0):.0f}<br>"
                f"Circularity: {row.get('circularity', 0):.3f}<br>"
                f"Perimeter: {row.get('perimeter', 0):.1f}"
            )
            hover_text.append(text)
        
        fig.add_trace(go.Scatter(
            x=cell_stats['centroid_x'],
            y=cell_stats['centroid_y'],
            mode='markers',
            name='Cell Centroids',
            marker=dict(
                size=8,
                color='red',
                symbol='circle',
                line=dict(color='white', width=1)
            ),
            text=hover_text,
            hovertemplate='%{text}<extra></extra>',
            visible=show_centroids
        ))
    
    # Configure layout for interactivity
    fig.update_layout(
        title="Interactive Cell View",
        xaxis=dict(
            scaleanchor='y',
            scaleratio=1,
            showgrid=False,
            zeroline=False
        ),
        yaxis=dict(
            showgrid=False,
            zeroline=False,
            autorange='reversed'  # Flip y-axis for image coordinates
        ),
        dragmode='pan',
        hovermode='closest',
        height=600,
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
            bgcolor="rgba(255,255,255,0.8)"
        )
    )
    
    return fig


def create_scatter_analysis(cell_stats: pd.DataFrame):
    """
    Create interactive scatter plots for cell analysis
    
    Args:
        cell_stats: DataFrame with cell statistics
    
    Returns:
        Plotly figure
    """
    # Create subplots
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Area vs Circularity', 'Perimeter vs Area')
    )
    
    # Scatter 1: Area vs Circularity
    fig.add_trace(
        go.Scatter(
            x=cell_stats['area'],
            y=cell_stats['circularity'],
            mode='markers',
            name='Cells',
            marker=dict(
                size=8,
                color=cell_stats['circularity'],
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="Circularity", x=0.46)
            ),
            text=[f"Cell {i}" for i in cell_stats.index],
            hovertemplate='Cell %{text}<br>Area: %{x:.0f}<br>Circularity: %{y:.3f}<extra></extra>'
        ),
        row=1, col=1
    )
    
    # Scatter 2: Perimeter vs Area
    fig.add_trace(
        go.Scatter(
            x=cell_stats['perimeter'],
            y=cell_stats['area'],
            mode='markers',
            name='Cells',
            marker=dict(
                size=8,
                color=cell_stats['area'],
                colorscale='Plasma',
                showscale=True,
                colorbar=dict(title="Area", x=1.0)
            ),
            text=[f"Cell {i}" for i in cell_stats.index],
            hovertemplate='Cell %{text}<br>Perimeter: %{x:.1f}<br>Area: %{y:.0f}<extra></extra>',
            showlegend=False
        ),
        row=1, col=2
    )
    
    # Update axes
    fig.update_xaxes(title_text="Area (pixels²)", row=1, col=1)
    fig.update_yaxes(title_text="Circularity", row=1, col=1)
    fig.update_xaxes(title_text="Perimeter (pixels)", row=1, col=2)
    fig.update_yaxes(title_text="Area (pixels²)", row=1, col=2)
    
    fig.update_layout(
        height=400,
        showlegend=False,
        hovermode='closest'
    )
    
    return fig


def show_interactive_visualizer(
    image: np.ndarray,
    masks: Optional[np.ndarray] = None,
    cell_stats: Optional[pd.DataFrame] = None
):
    """
    Display interactive visualizer in Streamlit
    
    Args:
        image: Original image
        masks: Cell masks (optional)
        cell_stats: Cell statistics DataFrame (optional)
    """
    st.subheader("🎯 Interactive Viewer")
    
    # Control toggles
    col1, col2, col3 = st.columns(3)
    
    with col1:
        show_overlay = st.checkbox("Show Mask Overlay", value=True)
    
    with col2:
        show_centroids = st.checkbox("Show Cell Centroids", value=True)
    
    with col3:
        show_contours = st.checkbox("Show Contours", value=False)
    
    # Create and display interactive figure
    fig = create_interactive_image_view(
        image,
        masks,
        cell_stats,
        show_overlay=show_overlay,
        show_centroids=show_centroids,
        show_contours=show_contours
    )
    
    st.plotly_chart(
        fig,
        use_container_width=True,
        config={
            'scrollZoom': True,
            'displayModeBar': True,
            'displaylogo': False,
            'modeBarButtonsToAdd': ['drawline', 'drawopenpath', 'eraseshape'],
            'modeBarButtonsToRemove': ['lasso2d']
        }
    )
    
    st.caption("""
    💡 **사용법:**
    - 마우스 휠: Zoom in/out
    - 드래그: Pan (이동)
    - 마커 클릭: 세포 정보 확인
    - 레전드 클릭: 레이어 on/off
    """)
    
    # Show scatter analysis if stats available
    if cell_stats is not None and len(cell_stats) > 0:
        st.markdown("---")
        st.subheader("📊 Scatter Analysis")
        
        scatter_fig = create_scatter_analysis(cell_stats)
        st.plotly_chart(scatter_fig, use_container_width=True)


if __name__ == "__main__":
    # For testing
    st.set_page_config(page_title="Interactive Visualizer", layout="wide")
    
    # Generate test data
    test_image = np.random.rand(512, 512, 3)
    test_masks = np.random.randint(0, 100, (512, 512))
    
    test_stats = pd.DataFrame({
        'centroid_x': np.random.randint(50, 450, 50),
        'centroid_y': np.random.randint(50, 450, 50),
        'area': np.random.randint(100, 1000, 50),
        'circularity': np.random.rand(50) * 0.5 + 0.5,
        'perimeter': np.random.randint(50, 200, 50)
    })
    
    show_interactive_visualizer(test_image, test_masks, test_stats)
