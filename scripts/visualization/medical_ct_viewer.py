"""
ADDS Medical CT Viewer
3-Plane View (Axial, Sagittal, Coronal) + 3D Volume Rendering
"""

import streamlit as st
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import io
from PIL import Image

st.set_page_config(page_title="ADDS CT Viewer", layout="wide")

st.title("🏥 ADDS Medical CT Viewer")

# CT 이미지 객체 로드 (데이터는 필요할 때만)
@st.cache_resource
def load_ct_image():
    """메모리 효율적: 전체 로드 안 함"""
    organ_path = "F:/ADDS/output/organs_simple/organs_multilabel_hu.nii.gz"
    img = nib.load(organ_path)
    return img

def get_slice(img, axis, slice_idx):
    """필요한 슬라이스만 로드"""
    if axis == 'axial':
        return img.dataobj[:, :, slice_idx]
    elif axis == 'sagittal':
        return img.dataobj[slice_idx, :, :]
    elif axis == 'coronal':
        return img.dataobj[:, slice_idx, :]

# Volume Rendering 색상맵 (인체 조직 유사)
def get_tissue_colormap():
    """실제 CT 값에 따른 조직 색상"""
    # HU값별 색상 (공기 → 지방 → 연조직 → 뼈)
    colors = [
        (0.0, (0.0, 0.0, 0.0)),      # -1000 HU: 공기 (검정)
        (0.2, (0.1, 0.05, 0.0)),     # -100 HU: 지방 (어두운 갈색)
        (0.4, (0.8, 0.6, 0.5)),      # 0 HU: 연조직 (살색)
        (0.6, (0.9, 0.7, 0.6)),      # +50 HU: 근육 (밝은 살색)
        (0.8, (1.0, 0.9, 0.8)),      # +100 HU: 간 (밝은 베이지)
        (1.0, (1.0, 1.0, 1.0))       # +1000 HU: 뼈 (흰색)
    ]
    return LinearSegmentedColormap.from_list('tissue', colors)

try:
    ct_img = load_ct_image()
    shape = ct_img.shape
    
    st.sidebar.header("⚙️ 뷰어 설정")
    
    # Window/Level 설정 (CT 표준)
    st.sidebar.subheader("Window/Level")
    
    presets = {
        "연조직 (Soft Tissue)": {"window": 400, "level": 40},
        "폐 (Lung)": {"window": 1500, "level": -600},
        "뼈 (Bone)": {"window": 2000, "level": 300},
        "간 (Liver)": {"window": 150, "level": 30},
        "Custom": {"window": 400, "level": 40}
    }
    
    preset = st.sidebar.selectbox("프리셋", list(presets.keys()))
    
    if preset == "Custom":
        window = st.sidebar.slider("Window", 1, 3000, 400)
        level = st.sidebar.slider("Level", -1000, 1000, 40)
    else:
        window = presets[preset]["window"]
        level = presets[preset]["level"]
    
    st.sidebar.write(f"Window: {window}, Level: {level}")
    
    # 슬라이스 번호
    st.sidebar.subheader("슬라이스 위치")
    
    axial_slice = st.sidebar.slider(
        "Axial (가로)", 
        0, shape[2]-1, 
        shape[2]//2
    )
    
    sagittal_slice = st.sidebar.slider(
        "Sagittal (세로)", 
        0, shape[0]-1, 
        shape[0]//2
    )
    
    coronal_slice = st.sidebar.slider(
        "Coronal (정면)", 
        0, shape[1]-1, 
        shape[1]//2
    )
    
    # Window/Level 적용 함수
    def apply_window_level(image, window, level):
        """CT Window/Level 알고리즘"""
        vmin = level - window / 2
        vmax = level + window / 2
        
        # Clipping
        windowed = np.clip(image, vmin, vmax)
        
        # Normalize to 0-255
        normalized = ((windowed - vmin) / (vmax - vmin) * 255).astype(np.uint8)
        
        return normalized
    
    # 3-Plane View
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("Axial (가로 단면)")
        axial_img = np.array(get_slice(ct_img, 'axial', axial_slice))
        axial_processed = apply_window_level(axial_img, window, level)
        
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.imshow(axial_processed, cmap='gray', origin='lower')
        ax.axis('off')
        ax.set_title(f'Slice {axial_slice}/{shape[2]-1}')
        st.pyplot(fig)
        plt.close()
    
    with col2:
        st.subheader("Sagittal (세로 단면)")
        sagittal_img = np.array(get_slice(ct_img, 'sagittal', sagittal_slice))
        sagittal_processed = apply_window_level(sagittal_img, window, level)
        
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.imshow(sagittal_processed.T, cmap='gray', origin='lower')
        ax.axis('off')
        ax.set_title(f'Slice {sagittal_slice}/{shape[0]-1}')
        st.pyplot(fig)
        plt.close()
    
    with col3:
        st.subheader("Coronal (정면 단면)")
        coronal_img = np.array(get_slice(ct_img, 'coronal', coronal_slice))
        coronal_processed = apply_window_level(coronal_img, window, level)
        
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.imshow(coronal_processed.T, cmap='gray', origin='lower')
        ax.axis('off')
        ax.set_title(f'Slice {coronal_slice}/{shape[1]-1}')
        st.pyplot(fig)
        plt.close()
    
    # 3D Volume Rendering (선택한 축 중심)
    st.markdown("---")
    st.subheader("📊 Volume Rendering")
    
    render_col1, render_col2 = st.columns(2)
    
    with render_col1:
        st.write("**Maximum Intensity Projection (MIP)**")
        
        # Axial MIP (위에서 본 뷰)
        mip_thickness = st.slider("MIP 두께", 10, 100, 50)
        
        start_slice = max(0, axial_slice - mip_thickness//2)
        end_slice = min(shape[2], axial_slice + mip_thickness//2)
        
        # MIP: 슬라이스들을 하나씩 로드해서 max
        mip_slices = [np.array(get_slice(ct_img, 'axial', i)) for i in range(start_slice, end_slice)]
        mip_image = np.max(np.stack(mip_slices, axis=2), axis=2)
        mip_processed = apply_window_level(mip_image, window, level)
        
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.imshow(mip_processed, cmap='gray', origin='lower')
        ax.axis('off')
        ax.set_title('MIP - Axial View')
        st.pyplot(fig)
        plt.close()
    
    with render_col2:
        st.write("**조직별 컬러 렌더링**")
        
        # 컬러맵 적용
        tissue_cmap = get_tissue_colormap()
        
        # Normalize for colormap
        normalized = (mip_image - mip_image.min()) / (mip_image.max() - mip_image.min())
        
        fig, ax = plt.subplots(figsize=(8, 8))
        im = ax.imshow(normalized, cmap=tissue_cmap, origin='lower')
        ax.axis('off')
        ax.set_title('Tissue Colormap')
        
        # 컬러바
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Tissue Density', rotation=270, labelpad=15)
        
        st.pyplot(fig)
        plt.close()
    
    # 통계 정보
    st.sidebar.markdown("---")
    st.sidebar.subheader("📈 현재 슬라이스 통계")
    st.sidebar.metric("평균 HU", f"{axial_img.mean():.1f}")
    st.sidebar.metric("최소 HU", f"{axial_img.min():.1f}")
    st.sidebar.metric("최대 HU", f"{axial_img.max():.1f}")
    
    # 사용 안내
    with st.expander("📖 사용 방법"):
        st.markdown("""
        **Window/Level 프리셋:**
        - **연조직**: 일반적인 복부/골반 영상
        - **폐**: 폐 실질 관찰
        - **뼈**: 골절/골 구조 관찰
        - **간**: 간 병변 관찰
        
        **3-Plane View:**
        - **Axial**: 가로 단면 (발 → 머리)
        - **Sagittal**: 세로 단면 (왼쪽 → 오른쪽)
        - **Coronal**: 정면 단면 (앞 → 뒤)
        
        **Volume Rendering:**
        - **MIP**: 선택한 두께 내 최대 밝기 투영
        - **Tissue Colormap**: 조직별 색상 구분
        """)

except Exception as e:
    st.error(f"❌ 데이터 로딩 오류: {e}")
    st.info("CT 데이터 경로를 확인해주세요.")
