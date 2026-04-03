"""
Enhanced Interactive Cell Viewer
Click cells to view detailed properties and compare multiple cells
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import Dict, List, Optional


class InteractiveCellViewer:
    """Interactive viewer with cell selection and detailed property display"""
    
    def __init__(self):
        self.selected_cells = []
        self.cell_stats = None
    
    def create_interactive_viewer(
        self,
        image: np.ndarray,
        masks: np.ndarray,
        cell_stats: pd.DataFrame,
        highlight_cells: List[int] = None
    ) -> go.Figure:
        """
        Create interactive cell viewer with click selection
        
        Args:
            image: Original image
            masks: Cell masks
            cell_stats: Cell statistics DataFrame
            highlight_cells: List of cell IDs to highlight
        
        Returns:
            Plotly figure
        """
        self.cell_stats = cell_stats
        
        fig = go.Figure()
        
        # Add base image
        fig.add_trace(go.Image(
            z=image,
            name="Image",
            hoverinfo='skip'
        ))
        
        # Add cell centroids with detailed hover info
        if 'centroid_x' in cell_stats.columns:
            # Prepare hover text
            hover_text = []
            colors = []
            sizes = []
            
            for idx, row in cell_stats.iterrows():
                # Detailed hover information
                text = f"<b>Cell {idx}</b><br>"
                text += f"Area: {row.get('area', 0):.0f} px²<br>"
                text += f"Circularity: {row.get('circularity', 0):.3f}<br>"
                text += f"Perimeter: {row.get('perimeter', 0):.1f} px<br>"
                
                if 'aspect_ratio' in row:
                    text += f"Aspect Ratio: {row.get('aspect_ratio', 0):.2f}<br>"
                if 'solidity' in row:
                    text += f"Solidity: {row.get('solidity', 0):.3f}<br>"
                
                hover_text.append(text)
                
                # Color and size based on selection/highlight
                if highlight_cells and idx in highlight_cells:
                    colors.append('yellow')
                    sizes.append(15)
                elif idx in self.selected_cells:
                    colors.append('lime')
                    sizes.append(12)
                else:
                    colors.append('red')
                    sizes.append(8)
            
            fig.add_trace(go.Scatter(
                x=cell_stats['centroid_x'],
                y=cell_stats['centroid_y'],
                mode='markers',
                name='Cells',
                marker=dict(
                    size=sizes,
                    color=colors,
                    symbol='circle',
                    line=dict(color='white', width=2)
                ),
                text=hover_text,
                hovertemplate='%{text}<extra></extra>',
                customdata=cell_stats.index
            ))
        
        # Configure layout
        fig.update_layout(
            title="Interactive Cell Viewer - Click cells to view details",
            xaxis=dict(
                scaleanchor='y',
                scaleratio=1,
                showgrid=False,
                zeroline=False
            ),
            yaxis=dict(
                showgrid=False,
                zeroline=False,
                autorange='reversed'
            ),
            dragmode='pan',
            hovermode='closest',
            height=600,
            showlegend=True
        )
        
        return fig
    
    def show_cell_details(self, cell_id: int):
        """Display detailed properties of selected cell"""
        if self.cell_stats is None or cell_id not in self.cell_stats.index:
            st.warning(f"Cell {cell_id} not found")
            return
        
        cell = self.cell_stats.loc[cell_id]
        
        st.markdown(f"### 🔬 Cell {cell_id} - Detailed Properties")
        
        # Metrics in columns
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Area", f"{cell.get('area', 0):.0f} px²")
            st.metric("Perimeter", f"{cell.get('perimeter', 0):.1f} px")
        
        with col2:
            st.metric("Circularity", f"{cell.get('circularity', 0):.3f}")
            st.metric("Solidity", f"{cell.get('solidity', 0):.3f}" if 'solidity' in cell else "N/A")
        
        with col3:
            st.metric("Aspect Ratio", f"{cell.get('aspect_ratio', 0):.2f}" if 'aspect_ratio' in cell else "N/A")
            st.metric("Equivalent Diameter", f"{cell.get('equivalent_diameter', 0):.1f} px" if 'equivalent_diameter' in cell else "N/A")
        
        # Classification
        st.markdown("---")
        st.markdown("**형태 분류:**")
        
        circularity = cell.get('circularity', 0)
        aspect_ratio = cell.get('aspect_ratio', 1)
        
        if circularity > 0.85:
            shape = "🔵 **Round** (정상적인 원형)"
        elif aspect_ratio > 3.0:
            shape = "📏 **Elongated** (길쭉한 형태)"
        elif circularity < 0.5:
            shape = "⚠️ **Irregular** (불규칙한 형태)"
        else:
            shape = "⚪ **Normal** (일반적인 형태)"
        
        st.info(shape)
        
        # Additional properties
        with st.expander("📋 All Properties"):
            props_df = pd.DataFrame({
                'Property': cell.index,
                'Value': cell.values
            })
            st.dataframe(props_df, use_container_width=True)
    
    def compare_cells(self, cell_ids: List[int]):
        """Compare multiple selected cells"""
        if not cell_ids or self.cell_stats is None:
            return
        
        st.markdown("### 📊 Cell Comparison")
        
        # Create comparison dataframe
        comparison_data = []
        for cell_id in cell_ids:
            if cell_id in self.cell_stats.index:
                cell = self.cell_stats.loc[cell_id]
                comparison_data.append({
                    'Cell ID': cell_id,
                    'Area': cell.get('area', 0),
                    'Circularity': cell.get('circularity', 0),
                    'Perimeter': cell.get('perimeter', 0),
                    'Aspect Ratio': cell.get('aspect_ratio', 0) if 'aspect_ratio' in cell else 0
                })
        
        if comparison_data:
            comp_df = pd.DataFrame(comparison_data)
            
            # Display table
            st.dataframe(
                comp_df.style.background_gradient(
                    subset=['Area', 'Circularity', 'Perimeter'],
                    cmap='RdYlGn'
                ),
                use_container_width=True
            )
            
            # Radar chart for comparison
            if len(cell_ids) <= 5:  # Limit to 5 cells for readability
                fig = go.Figure()
                
                metrics = ['Area', 'Circularity', 'Perimeter', 'Aspect Ratio']
                
                for _, row in comp_df.iterrows():
                    # Normalize values for radar chart
                    values = [
                        row['Area'] / comp_df['Area'].max(),
                        row['Circularity'],
                        row['Perimeter'] / comp_df['Perimeter'].max(),
                        row['Aspect Ratio'] / max(comp_df['Aspect Ratio'].max(), 1)
                    ]
                    
                    fig.add_trace(go.Scatterpolar(
                        r=values,
                        theta=metrics,
                        fill='toself',
                        name=f"Cell {row['Cell ID']}"
                    ))
                
                fig.update_layout(
                    polar=dict(
                        radialaxis=dict(
                            visible=True,
                            range=[0, 1]
                        )
                    ),
                    showlegend=True,
                    title="Normalized Metrics Comparison"
                )
                
                st.plotly_chart(fig, use_container_width=True)


def show_interactive_cell_viewer(
    image: np.ndarray,
    masks: np.ndarray,
    cell_stats: pd.DataFrame
):
    """
    Streamlit component for interactive cell viewing
    
    Args:
        image: Original image
        masks: Cell masks
        cell_stats: Cell statistics DataFrame
    """
    st.subheader("🎯 Interactive Cell Viewer")
    
    # Initialize viewer in session state
    if 'cell_viewer' not in st.session_state:
        st.session_state.cell_viewer = InteractiveCellViewer()
    
    viewer = st.session_state.cell_viewer
    
    # Control panel
    col1, col2, col3 = st.columns([2, 2, 1])
    
    with col1:
        # Cell selection by ID
        cell_ids = cell_stats.index.tolist()
        selected_cell = st.selectbox(
            "세포 선택 (ID)",
            options=['선택 안함'] + cell_ids,
            key="cell_selector"
        )
    
    with col2:
        # Multi-select for comparison
        compare_cells = st.multiselect(
            "비교할 세포들",
            options=cell_ids,
            max_selections=5,
            key="cell_comparer"
        )
    
    with col3:
        if st.button("🔄 Reset"):
            viewer.selected_cells = []
            st.rerun()
    
    # Create and display interactive figure
    highlight_cells = compare_cells if compare_cells else None
    
    fig = viewer.create_interactive_viewer(
        image,
        masks,
        cell_stats,
        highlight_cells=highlight_cells
    )
    
    # Display figure with click events
    st.plotly_chart(
        fig,
        use_container_width=True,
        config={
            'scrollZoom': True,
            'displayModeBar': True,
            'displaylogo': False
        }
    )
    
    st.caption("""
    💡 **사용법:**
    - 🖱️ 드롭다운에서 세포 선택하여 상세 정보 확인
    - 📊 여러 세포 선택하여 비교
    - 🔍 마우스 휠로 Zoom, 드래그로 Pan
    - 💬 마우스 올려서 간단한 정보 확인
    """)
    
    # Display selected cell details
    if selected_cell and selected_cell != '선택 안함':
        st.markdown("---")
        viewer.show_cell_details(int(selected_cell))
    
    # Display comparison
    if compare_cells and len(compare_cells) > 1:
        st.markdown("---")
        viewer.compare_cells([int(c) for c in compare_cells])
    
    # Quick stats
    st.markdown("---")
    st.markdown("### 📊 Quick Statistics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Cells", len(cell_stats))
    
    with col2:
        avg_area = cell_stats['area'].mean() if 'area' in cell_stats.columns else 0
        st.metric("Avg Area", f"{avg_area:.0f} px²")
    
    with col3:
        avg_circ = cell_stats['circularity'].mean() if 'circularity' in cell_stats.columns else 0
        st.metric("Avg Circularity", f"{avg_circ:.3f}")
    
    with col4:
        if 'circularity' in cell_stats.columns:
            abnormal = len(cell_stats[cell_stats['circularity'] < 0.5])
            st.metric("Abnormal Cells", f"{abnormal} ({abnormal/len(cell_stats)*100:.1f}%)")


if __name__ == "__main__":
    # For testing
    st.set_page_config(page_title="Interactive Cell Viewer", layout="wide")
    
    # Generate test data
    np.random.seed(42)
    n = 100
    
    test_image = np.random.rand(512, 512, 3)
    test_masks = np.random.randint(0, n, (512, 512))
    
    test_stats = pd.DataFrame({
        'centroid_x': np.random.randint(50, 450, n),
        'centroid_y': np.random.randint(50, 450, n),
        'area': np.random.randint(500, 2000, n),
        'circularity': np.random.beta(8, 2, n),
        'perimeter': np.random.randint(80, 200, n),
        'aspect_ratio': np.random.gamma(2, 0.5, n),
        'solidity': np.random.beta(9, 1, n)
    })
    
    show_interactive_cell_viewer(test_image, test_masks, test_stats)
