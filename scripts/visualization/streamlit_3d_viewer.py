"""
ADDS Streamlit 3D Organ Viewer
Plotly 기반 - 즉시 작동 버전
"""

import streamlit as st
import plotly.graph_objects as go
import json
from pathlib import Path

st.set_page_config(page_title="ADDS 3D Viewer", layout="wide")

st.title("🫁 ADDS 3D Organ & Tumor Viewer")

# 메시 파일 경로 - 부드러운 메시 사용!
MESH_DIR = Path("F:/ADDS/output/meshes_smoothed")

# 장기 정의
ORGANS = {
    "fat": {"color": "#FFD700", "name": "지방"},
    "lung_tissue": {"color": "#87CEEB", "name": "폐"},
    "muscle": {"color": "#CD5C5C", "name": "근육"},
    "liver": {"color": "#8B4513", "name": "간"},
    "soft_tissue": {"color": "#FFB6C1", "name": "연조직"},
    "bone": {"color": "#FFFFFF", "name": "뼈"}
}

TUMORS = {
    "muscle_tumors": {"color": "#FF0000", "name": "근육 종양"},
    "liver_tumors": {"color": "#FF0000", "name": "간 종양"},
    "soft_tissue_tumors": {"color": "#FF0000", "name": "연조직 종양"}
}

# 사이드바 컨트롤
st.sidebar.header("⚙️ 레이어 컨트롤")

# 장기 선택
st.sidebar.subheader("장기")
selected_organs = {}
organ_opacity = {}

for organ_key, organ_info in ORGANS.items():
    col1, col2 = st.sidebar.columns([3, 1])
    with col1:
        selected_organs[organ_key] = st.checkbox(
            organ_info['name'], 
            value=True,
            key=f"organ_{organ_key}"
        )
    with col2:
        if selected_organs[organ_key]:
            organ_opacity[organ_key] = st.slider(
                "투명도",
                0.0, 1.0, 0.5,
                key=f"opacity_{organ_key}",
                label_visibility="collapsed"
            )

# 종양 선택
st.sidebar.subheader("종양")
selected_tumors = {}
tumor_opacity = {}

for tumor_key, tumor_info in TUMORS.items():
    col1, col2 = st.sidebar.columns([3, 1])
    with col1:
        selected_tumors[tumor_key] = st.checkbox(
            tumor_info['name'],
            value=True,
            key=f"tumor_{tumor_key}"
        )
    with col2:
        if selected_tumors[tumor_key]:
            tumor_opacity[tumor_key] = st.slider(
                "투명도",
                0.0, 1.0, 0.9,
                key=f"opacity_{tumor_key}",
                label_visibility="collapsed"
            )

# 메시 로딩 함수
@st.cache_data
def load_mesh(mesh_file):
    """JSON 메시 파일 로드"""
    with open(mesh_file, 'r') as f:
        return json.load(f)

# Plotly figure 생성
fig = go.Figure()

# 장기 추가
for organ_key, organ_info in ORGANS.items():
    if selected_organs.get(organ_key, False):
        mesh_file = MESH_DIR / f"{organ_key}_mesh.json"
        
        if mesh_file.exists():
            mesh_data = load_mesh(str(mesh_file))
            
            vertices = mesh_data['vertices']
            faces = mesh_data['faces']
            
            # Plotly Mesh3d 포맷으로 변환
            x = [v[0] for v in vertices]
            y = [v[1] for v in vertices]
            z = [v[2] for v in vertices]
            
            i = [f[0] for f in faces]
            j = [f[1] for f in faces]
            k = [f[2] for f in faces]
            
            fig.add_trace(go.Mesh3d(
                x=x, y=y, z=z,
                i=i, j=j, k=k,
                color=organ_info['color'],
                opacity=organ_opacity.get(organ_key, 0.5),
                name=organ_info['name'],
                showlegend=True,
                flatshading=True,
                lighting=dict(
                    ambient=0.6,
                    diffuse=0.8,
                    specular=0.2,
                    roughness=0.5
                )
            ))

# 종양 추가
for tumor_key, tumor_info in TUMORS.items():
    if selected_tumors.get(tumor_key, False):
        mesh_file = MESH_DIR / f"{tumor_key}_mesh.json"
        
        if mesh_file.exists():
            mesh_data = load_mesh(str(mesh_file))
            
            vertices = mesh_data['vertices']
            faces = mesh_data['faces']
            
            x = [v[0] for v in vertices]
            y = [v[1] for v in vertices]
            z = [v[2] for v in vertices]
            
            i = [f[0] for f in faces]
            j = [f[1] for f in faces]
            k = [f[2] for f in faces]
            
            fig.add_trace(go.Mesh3d(
                x=x, y=y, z=z,
                i=i, j=j, k=k,
                color=tumor_info['color'],
                opacity=tumor_opacity.get(tumor_key, 0.9),
                name=tumor_info['name'],
                showlegend=True,
                flatshading=True
            ))

# 레이아웃 설정
fig.update_layout(
    scene=dict(
        xaxis=dict(title='X (mm)', backgroundcolor="white", gridcolor="lightgray"),
        yaxis=dict(title='Y (mm)', backgroundcolor="white", gridcolor="lightgray"),
        zaxis=dict(title='Z (mm)', backgroundcolor="white", gridcolor="lightgray"),
        aspectmode='data',
        camera=dict(
            eye=dict(x=1.5, y=1.5, z=1.5)
        )
    ),
    width=1200,
    height=800,
    paper_bgcolor='white',
    plot_bgcolor='white',
    legend=dict(
        x=0.02,
        y=0.98,
        bgcolor='rgba(255,255,255,0.8)',
        bordercolor='gray',
        borderwidth=1
    )
)

# 3D 플롯 표시
st.plotly_chart(fig, use_container_width=True)

# 통계 정보
st.sidebar.markdown("---")
st.sidebar.subheader("📊 통계")

visible_count = sum(selected_organs.values()) + sum(selected_tumors.values())
st.sidebar.metric("표시된 레이어", visible_count)

# 사용 안내
with st.expander("📖 사용 방법"):
    st.markdown("""
    **마우스 컨트롤:**
    - **좌클릭 + 드래그**: 회전
    - **우클릭 + 드래그**: 이동
    - **스크롤**: 줌 인/아웃
    
    **레이어 컨트롤:**
    - 왼쪽 사이드바에서 장기/종양 ON/OFF
    - 투명도 슬라이더로 가시성 조절
    
    **팁:**
    - 여러 레이어를 겹쳐서 보세요
    - 종양을 강조하려면 다른 장기 투명도를 낮추세요
    """)
