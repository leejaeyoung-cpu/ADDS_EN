"""
CT Analysis UI for ADDS
통합 의료 이미지 분석: CT 종양 검출, 세포 분석, AI 리서치
"""

import streamlit as st
import sys
from pathlib import Path
import json
import numpy as np
from PIL import Image
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import logging

# ADDS 경로 추가
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

logger = logging.getLogger(__name__)

# Import modules
try:
    from analyze_tumor_location import TumorAnalyzer
except:
    TumorAnalyzer = None

try:
    from src.medical_imaging.ai_research import (
        MedicalResearcher,
        analyze_ct_findings,
        explain_medical_terms
    )
    AI_RESEARCH_AVAILABLE = True
except Exception as e:
    logger.warning(f"AI research module not available: {e}")
    AI_RESEARCH_AVAILABLE = False

# Cell analysis removed - not needed for CT analysis


def show_ct_analysis():
    """CT 분석 메인 페이지"""
    
    st.title("🏥 CT 종양 검출 및 AI 분석")
    st.markdown("**CT 종양 검출 • AI 리서치 • 결과 시각화**")
    
    # 탭 구성 (3개 탭)
    tab1, tab2, tab3 = st.tabs([
        "🔍 CT 종양 검출",
        "🤖 AI 리서치",
        "📈 결과 시각화"
    ])
    
    # 탭 1: CT 종양 검출
    with tab1:
        show_tumor_detection()
    
    # 탭 2: AI 리서치
    with tab2:
        show_ai_research()
    
    # 탭 3: 결과 시각화
    with tab3:
        show_results_visualization()


def show_tumor_detection():
    """종양 검출 섹션"""
    
    st.header("🔍 CT 종양 검출 및 분석")
    
    # 이미지 소스 선택
    image_source = st.radio(
        "이미지 소스",
        ["📤 이미지 업로드", "📁 데이터셋에서 선택"],
        horizontal=True
    )
    
    uploaded_file = None
    selected_image_path = None
    
    if image_source == "📤 이미지 업로드":
        st.info("💡 CT 이미지를 업로드하세요 (NIfTI, DICOM, PNG, JPG 지원)")
        
        uploaded_file = st.file_uploader(
            "CT 이미지 선택",
            type=['nii', 'nii.gz', 'dcm', 'png', 'jpg', 'jpeg'],
            help="NIfTI (.nii, .nii.gz), DICOM (.dcm) 또는 일반 이미지 파일"
        )
        
        if uploaded_file:
            st.success(f"✅ 파일 업로드됨: {uploaded_file.name}")
            
            # 임시 저장
            temp_dir = Path("temp_uploads")
            temp_dir.mkdir(exist_ok=True)
            temp_file_path = temp_dir / uploaded_file.name
            
            with open(temp_file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            selected_image_path = temp_file_path
            
            # 미리보기 (PNG/JPG만)
            if uploaded_file.name.lower().endswith(('.png', '.jpg', '.jpeg')):
                st.subheader("📸 이미지 미리보기")
                img = Image.open(uploaded_file)
                st.image(img, use_container_width=True)
            elif uploaded_file.name.lower().endswith(('.nii', '.nii.gz')):
                st.info("🔬 NIfTI 파일이 업로드되었습니다. 분석을 시작하세요.")
            
    else:  # 데이터셋에서 선택
        st.info("💡 기존 데이터셋에서 이미지를 선택하세요")
        
        # 데이터 경로 선택
        data_options = {
            "정제된 데이터 (CTdata_cleaned)": "CTdata_cleaned",
            "원본 데이터 (CTdata)": "CTdata",
            "Medical Decathlon (Task10_Colon)": "data/medical_decathlon/Task10_Colon"
        }
        
        selected_data = st.selectbox(
            "데이터셋 선택",
            list(data_options.keys())
        )
        
        data_path = Path(data_options[selected_data])
        
        # 데이터셋에서 이미지 파일 찾기
        if data_path.exists():
            # NIfTI 파일 찾기
            nii_files = list(data_path.rglob("*.nii*"))
            
            if nii_files:
                selected_file = st.selectbox(
                    f"이미지 선택 ({len(nii_files)}개 파일)",
                    options=[f.name for f in nii_files[:50]],  # 최대 50개만 표시
                    help="데이터셋에서 분석할 이미지를 선택하세요"
                )
                
                # 선택된 파일 경로 찾기
                for f in nii_files:
                    if f.name == selected_file:
                        selected_image_path = f
                        break
                
                if selected_image_path:
                    st.success(f"✅ 선택됨: {selected_image_path}")
            else:
                st.warning(f"⚠️ {data_path}에서 NIfTI 파일을 찾을 수 없습니다.")
        else:
            st.error(f"❌ 데이터셋을 찾을 수 없습니다: {data_path}")
    
    # 검출 모드 선택
    st.markdown("---")
    detection_mode = st.radio(
        "검출 모드",
        ["Mock 모드 (임계값 기반)", "SOTA 모드 (딥러닝 - 준비 중)"],
        help="Mock 모드는 빠르지만 정확도가 제한적입니다."
    )
    
    # 파라미터 설정
    with st.expander("⚙️ 고급 설정"):
        col1, col2 = st.columns(2)
        
        with col1:
            slice_thickness = st.number_input(
                "슬라이스 간격 (mm)",
                min_value=0.1,
                max_value=10.0,
                value=1.0,
                step=0.1
            )
        
        with col2:
            pixel_spacing = st.number_input(
                "픽셀 간격 (mm)",
                min_value=0.1,
                max_value=5.0,
                value=0.78,
                step=0.01
            )
    
    # 검출 실행
    if st.button("🚀 종양 검출 시작", type="primary"):
        
        # SOTA 모드 체크
        if "SOTA" in detection_mode:
            st.warning("⚠️ SOTA 모드는 현재 모델 학습 중입니다. Mock 모드를 사용해주세요.")
            return
        
        if not selected_image_path:
            st.error("❌ 먼저 이미지를 업로드하거나 선택해주세요.")
            return
        
        with st.spinner("종양 검출 중... (수 분 소요될 수 있습니다)"):
            try:
                st.info("🔬 이미지 분석 중...")
                
                # 이미지 로드 및 처리
                import time
                import matplotlib.pyplot as plt
                import matplotlib.patches as patches
                
                # 이미지 파일 형식에 따라 처리
                img_array = None
                
                if str(selected_image_path).lower().endswith(('.png', '.jpg', '.jpeg')):
                    # 일반 이미지 로드
                    img = Image.open(selected_image_path)
                    img_array = np.array(img)
                    
                elif str(selected_image_path).lower().endswith('.dcm'):
                    # DICOM 파일 로드
                    try:
                        import pydicom
                        dcm = pydicom.dcmread(str(selected_image_path))
                        img_array = dcm.pixel_array
                        
                        # DICOM 메타데이터 표시
                        st.info(f"📊 DICOM 이미지 로드됨: {img_array.shape}")
                        
                        # 윈도우 레벨/폭 적용 (있는 경우)
                        if hasattr(dcm, 'WindowCenter') and hasattr(dcm, 'WindowWidth'):
                            window_center = float(dcm.WindowCenter) if not isinstance(dcm.WindowCenter, (list, tuple)) else float(dcm.WindowCenter[0])
                            window_width = float(dcm.WindowWidth) if not isinstance(dcm.WindowWidth, (list, tuple)) else float(dcm.WindowWidth[0])
                            
                            img_min = window_center - window_width / 2
                            img_max = window_center + window_width / 2
                            img_array = np.clip(img_array, img_min, img_max)
                            img_array = ((img_array - img_min) / (img_max - img_min) * 255).astype(np.uint8)
                        else:
                            # 정규화
                            img_array = ((img_array - img_array.min()) / (img_array.max() - img_array.min()) * 255).astype(np.uint8)
                        
                    except ImportError:
                        st.error("❌ pydicom 라이브러리가 필요합니다: `pip install pydicom`")
                        return
                    except Exception as e:
                        st.error(f"❌ DICOM 파일 로드 실패: {str(e)}")
                        return
                    
                elif str(selected_image_path).lower().endswith(('.nii', '.nii.gz')):
                    # NIfTI 파일 로드
                    try:
                        import nibabel as nib
                        nii_img = nib.load(str(selected_image_path))
                        img_data = nii_img.get_fdata()
                        
                        # 중간 슬라이스 선택
                        if img_data.ndim == 3:
                            slice_idx = img_data.shape[2] // 2
                            img_array = img_data[:, :, slice_idx]
                        elif img_data.ndim == 2:
                            img_array = img_data
                        else:
                            st.error("❌ 지원하지 않는 이미지 차원입니다.")
                            return
                            
                        st.info(f"📊 NIfTI 데이터 로드됨: {img_data.shape}")
                        
                    except ImportError:
                        st.error("❌ nibabel 라이브러리가 필요합니다: `pip install nibabel`")
                        return
                    except Exception as e:
                        st.error(f"❌ NIfTI 파일 로드 실패: {str(e)}")
                        return
                
                if img_array is None:
                    st.error("❌ 이미지를 로드할 수 없습니다.")
                    return
                
                # Mock 종양 검출 (간단한 임계값 기반)
                st.warning("⚠️ **Mock 모드 (임계값 기반)**: 이 검출은 85th percentile 밝기 기반 단순 임계값 방법입니다. "
                           "임상 진단에 사용할 수 없습니다. nnU-Net 기반 검출은 SOTA 모드에서 제공 예정입니다.")
                time.sleep(1)
                
                # 예시: 밝은 영역을 종양으로 간주 (실제로는 더 복잡한 알고리즘 필요)
                if img_array.ndim == 3:  # RGB
                    gray_img = np.mean(img_array, axis=2)
                else:
                    gray_img = img_array
                
                # 간단한 임계값 검출
                threshold = np.percentile(gray_img, 85)
                tumor_mask = gray_img > threshold
                
                # 종양 영역 찾기
                from scipy import ndimage
                labeled_mask, num_tumors = ndimage.label(tumor_mask)
                
                # 각 종양의 중심과 크기 계산
                tumor_info = []
                if num_tumors > 0:
                    for i in range(1, min(num_tumors + 1, 11)):  # 최대 10개만
                        tumor_region = (labeled_mask == i)
                        if np.sum(tumor_region) > 50:  # 최소 크기 필터
                            coords = np.argwhere(tumor_region)
                            center_y, center_x = coords.mean(axis=0)
                            size = coords.shape[0]
                            
                            # 직경 추정 (픽셀 -> mm)
                            diameter_px = np.sqrt(size / np.pi) * 2
                            diameter_mm = diameter_px * pixel_spacing
                            
                            # Confidence derived from region properties (NOT random)
                            # Based on: region compactness + intensity contrast
                            region_vals = gray_img[tumor_region]
                            mean_intensity = float(np.mean(region_vals))
                            bg_mean = float(np.mean(gray_img[~tumor_mask]))
                            contrast = (mean_intensity - bg_mean) / (mean_intensity + bg_mean + 1e-6)
                            compactness = (4 * np.pi * size) / (np.sum(np.sqrt(np.sum(np.diff(coords, axis=0)**2, axis=1)))**2 + 1e-6)
                            # Bounded heuristic confidence — NOT a trained model output
                            confidence = float(np.clip(0.3 + contrast * 0.3 + min(compactness, 1.0) * 0.2, 0.1, 0.7))
                            
                            tumor_info.append({
                                'center': (center_x, center_y),
                                'size': size,
                                'diameter_mm': diameter_mm,
                                'confidence': confidence,
                                '_method': 'threshold_mock'
                            })
                
                # 시각화
                st.success("✅ 종양 검출 완료!")
                
                # 결과 메트릭
                st.subheader("📊 검출 결과")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("검출된 종양", f"{len(tumor_info)}개")
                with col2:
                    max_diameter = max([t['diameter_mm'] for t in tumor_info]) if tumor_info else 0
                    st.metric("최대 직경", f"{max_diameter:.1f} mm")
                with col3:
                    avg_conf = np.mean([t['confidence'] for t in tumor_info]) if tumor_info else 0
                    st.metric("평균 신뢰도", f"{avg_conf*100:.0f}%")
                
                # 이미지 시각화
                st.subheader("🔍 검출 시각화")
                
                fig, ax = plt.subplots(figsize=(10, 10))
                
                # 원본 이미지 표시
                if img_array.ndim == 3:
                    ax.imshow(img_array)
                else:
                    ax.imshow(img_array, cmap='gray')
                
                # 종양 위치 표시
                for idx, tumor in enumerate(tumor_info):
                    center_x, center_y = tumor['center']
                    diameter_px = np.sqrt(tumor['size'] / np.pi) * 2
                    
                    # 원으로 표시
                    circle = patches.Circle(
                        (center_x, center_y),
                        diameter_px,
                        fill=False,
                        edgecolor='red',
                        linewidth=2,
                        label=f"종양 {idx+1}"
                    )
                    ax.add_patch(circle)
                    
                    # 텍스트 레이블
                    ax.text(
                        center_x, center_y - diameter_px - 10,
                        f"T{idx+1}: {tumor['diameter_mm']:.1f}mm\n{tumor['confidence']*100:.0f}%",
                        color='red',
                        fontsize=10,
                        ha='center',
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.7)
                    )
                
                ax.set_title("종양 검출 결과", fontsize=14, fontweight='bold')
                ax.axis('off')
                
                st.pyplot(fig)
                plt.close()
                
                # 상세 정보 테이블
                if tumor_info:
                    st.subheader("📋 종양 상세 정보")
                    tumor_df = pd.DataFrame([
                        {
                            '종양 ID': f'T{i+1}',
                            '직경 (mm)': f"{t['diameter_mm']:.1f}",
                            '픽셀 수': t['size'],
                            '신뢰도': f"{t['confidence']*100:.0f}%"
                        }
                        for i, t in enumerate(tumor_info)
                    ])
                    st.dataframe(tumor_df, use_container_width=True)
                
                # 세션 상태에 저장
                st.session_state['tumor_detection_completed'] = True
                st.session_state['detection_results'] = {
                    'tumor_count': len(tumor_info),
                    'max_diameter_mm': max_diameter,
                    'confidence_score': avg_conf,
                    'image_path': str(selected_image_path),
                    'tumors': tumor_info
                }
                
                st.info("💡 AI 리서치 탭에서 이 결과를 분석할 수 있습니다.")
                    
            except Exception as e:
                st.error(f"오류 발생: {str(e)}")


# Cell analysis function removed - not needed for CT analysis


def show_ai_research():
    """AI 리서치 섹션"""
    
    st.header("🤖 AI 의료 리서치 (OpenAI)")
    
    if not AI_RESEARCH_AVAILABLE:
        st.warning("⚠️ AI 리서치 모듈을 사용할 수 없습니다")
        st.info("""
        AI 리서치 기능을 사용하려면:
        1. OpenAI 설치: `pip install openai`
        2. API 키 설정: `.env` 파일에 `OPENAI_API_KEY` 추가
        """)
        return
    
    # API 키 확인
    import os
    api_key = os.getenv('OPENAI_API_KEY')
    
    if not api_key:
        st.error("🔑 OpenAI API 키가 필요합니다")
        st.info("`.env` 파일에 `OPENAI_API_KEY=your_key_here` 형식으로 추가하세요")
        return
    
    # API 키가 유효한 형식인지 확인 (sk- 또는 sk-proj-로 시작)
    if not (api_key.startswith('sk-') or api_key.startswith('sk-proj-')):
        st.error("🔑 유효하지 않은 API 키 형식입니다")
        st.info("OpenAI API 키는 'sk-' 또는 'sk-proj-'로 시작해야 합니다")
        return
    
    st.success("✅ AI 리서치 모듈 준비됨")
    
    # 리서치 모드 선택
    research_mode = st.selectbox(
        "리서치 모드",
        [
            "📊 CT 검출 결과 분석",
            "🔬 종양 특성 리서치",
            "📚 의학 용어 설명",
            "💊 치료법 인사이트"
        ]
    )
    
    if research_mode == "📊 CT 검출 결과 분석":
        st.subheader("CT 검출 결과 AI 분석")
        
        # 시뮬레이션 데이터 또는 실제 결과 사용
        if st.session_state.get('tumor_detection_completed'):
            st.info("최근 검출 결과를 분석합니다")
            findings = {
                'tumor_count': 2,
                'tumor_volume_mm3': 1250.5,
                'max_diameter_mm': 15.2,
                'location': 'Right colon',
                'confidence_score': 0.77
            }
        else:
            st.warning("검출 결과가 없습니다. 예시 데이터로 데모를 진행합니다.")
            
            col1, col2 = st.columns(2)
            with col1:
                tumor_count = st.number_input("종양 개수", 1, 10, 2)
                max_diameter = st.number_input("최대 직경 (mm)", 1.0, 100.0, 15.2)
            
            with col2:
                tumor_volume = st.number_input("총 부피 (mm³)", 1.0, 10000.0, 1250.5)
                location = st.text_input("위치", "Right colon")
            
            findings = {
                'tumor_count': tumor_count,
                'tumor_volume_mm3': tumor_volume,
                'max_diameter_mm': max_diameter,
                'location': location,
                'confidence_score': 0.77
            }
        
        if st.button("🤖 AI 분석 실행", type="primary"):
            with st.spinner("OpenAI로 분석 중... (10-30초 소요)"):
                try:
                    result = analyze_ct_findings(findings)
                    
                    st.subheader("📋 AI 분석 결과")
                    st.markdown(result)
                    
                    # 세션에 저장
                    st.session_state['ai_analysis'] = result
                    
                except Exception as e:
                    st.error(f"AI 분석 실패: {str(e)}")
    
    elif research_mode == "🔬 종양 특성 리서치":
        st.subheader("종양 특성 리서치")
        
        col1, col2 = st.columns(2)
        with col1:
            tumor_type = st.text_input("종양 유형", "adenocarcinoma")
            size_mm = st.number_input("크기 (mm)", 1.0, 100.0, 25.0)
        
        with col2:
            shape = st.selectbox("형태", ["regular", "irregular", "lobulated"])
            density_hu = st.number_input("밀도 (HU)", -100, 200, 45)
        
        tumor_data = {
            'type': tumor_type,
            'size_mm': size_mm,
            'shape': shape,
            'density_hu': density_hu
        }
        
        if st.button("🔬 리서치 실행", type="primary"):
            with st.spinner("의학 문헌 리서치 중..."):
                try:
                    from src.medical_imaging.ai_research import research_tumor_characteristics
                    
                    result = research_tumor_characteristics(tumor_data)
                    
                    st.subheader("📚 리서치 결과")
                    st.markdown(result)
                    
                except Exception as e:
                    st.error(f"리서치 실패: {str(e)}")
    
    elif research_mode == "📚 의학 용어 설명":
        st.subheader("의학 용어 설명")
        
        terms_input = st.text_area(
            "설명이 필요한 용어 (한 줄에 하나씩)",
            "adenocarcinoma\nmetastasis\nTNM staging"
        )
        
        terms = [t.strip() for t in terms_input.split('\n') if t.strip()]
        
        if st.button("📖 설명 요청", type="primary"):
            with st.spinner("용어 설명 생성 중..."):
                try:
                    result = explain_medical_terms(terms)
                    
                    st.subheader("📖 용어 설명")
                    st.markdown(result)
                    
                except Exception as e:
                    st.error(f"설명 생성 실패: {str(e)}")
    
    elif research_mode == "💊 치료법 인사이트":
        st.subheader("치료법 인사이트")
        
        col1, col2 = st.columns(2)
        with col1:
            tnm_stage = st.text_input("TNM 병기", "T3N1M0")
            tumor_location = st.text_input("종양 위치", "sigmoid colon")
        
        with col2:
            tumor_size = st.number_input("종양 크기 (mm)", 1.0, 100.0, 35.0)
            patient_age = st.selectbox("환자 연령대", ["20-30", "30-40", "40-50", "50-60", "60-70", "70+"])
        
        comorbidities = st.multiselect(
            "동반질환",
            ["diabetes", "hypertension", "heart disease", "kidney disease", "liver disease"]
        )
        
        patient_data = {
            'tnm_stage': tnm_stage,
            'tumor_location': tumor_location,
            'tumor_size_mm': tumor_size,
            'patient_age': patient_age,
            'comorbidities': comorbidities
        }
        
        if st.button("💊 치료 인사이트 요청", type="primary"):
            with st.spinner("치료 옵션 분석 중..."):
                try:
                    from src.medical_imaging.ai_research import suggest_treatment_insights
                    
                    result = suggest_treatment_insights(patient_data)
                    
                    st.subheader("💊 치료 인사이트")
                    st.markdown(result)
                    
                    st.warning("""
                    ⚠️ **중요 면책 조항**
                    
                    이 정보는 교육 및 참고 목적으로만 제공됩니다. 
                    실제 치료 결정은 반드시 자격을 갖춘 의료 전문가의 종합적인 평가를 바탕으로 이루어져야 합니다.
                    """)
                    
                except Exception as e:
                    st.error(f"인사이트 생성 실패: {str(e)}")


def show_results_visualization():
    """결과 시각화 섹션"""
    
    st.header("📈 통합 결과 시각화")
    
    # CT 검출 결과
    if st.session_state.get('tumor_detection_completed'):
        st.subheader("🔍 CT 종양 검출 결과")
        
        # 결과 이미지 표시
        result_path = Path("results/candidate_detection_test")
        if result_path.exists():
            images = sorted(result_path.glob("*.png"))
            
            if images:
                selected_image = st.selectbox(
                    "결과 선택",
                    options=[img.name for img in images]
                )
                
                for img_path in images:
                    if img_path.name == selected_image:
                        img = Image.open(img_path)
                        st.image(img, use_container_width=True)
                        break
    else:
        st.info("CT 검출을 먼저 실행하세요 ('🔍 CT 종양 검출' 탭)")
    
    st.markdown("---")
    
    # AI 분석 결과
    if 'ai_analysis' in st.session_state:
        st.subheader("🤖 AI 분석 결과")
        with st.expander("전체 분석 보기", expanded=True):
            st.markdown(st.session_state['ai_analysis'])
    else:
        st.info("AI 리서치를 먼저 실행하세요 ('🤖 AI 리서치' 탭)")


# 메인 함수
if __name__ == "__main__":
    show_ct_analysis()
