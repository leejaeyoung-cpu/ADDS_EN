# -*- coding: utf-8 -*-
"""
CT 종양 분석 결과 시각화 UI
Streamlit 기반 대화형 인터페이스
"""
import streamlit as st
import json
import numpy as np
from PIL import Image
from pathlib import Path
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd

# 페이지 설정
st.set_page_config(
    page_title="CT 종양 분석 시각화",
    page_icon="🏥",
    layout="wide"
)

# 스타일
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .detection-img {
        border: 2px solid #1f77b4;
        border-radius: 0.5rem;
    }
    </style>
""", unsafe_allow_html=True)

def load_analysis_results():
    """분석 결과 로드"""
    report_path = Path(r"c:\Users\brook\Desktop\ADDS\tumor_analysis_results\tumor_analysis_report.json")
    
    if not report_path.exists():
        st.error(f"분석 결과를 찾을 수 없습니다: {report_path}")
        return None
    
    with open(report_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def load_detection_image(slice_num):
    """검출 결과 이미지 로드"""
    img_path = Path(rf"c:\Users\brook\Desktop\ADDS\tumor_analysis_results\detection_{slice_num:05d}.jpg")
    
    if img_path.exists():
        return Image.open(img_path)
    return None

def main():
    st.markdown('<div class="main-header">🏥 CT 종양 위치 및 크기 분석</div>', unsafe_allow_html=True)
    
    # 분석 결과 로드
    results = load_analysis_results()
    
    if results is None:
        st.warning("먼저 `analyze_tumor_location.py`를 실행하여 분석을 완료해주세요.")
        return
    
    summary = results.get('summary', {})
    
    # 사이드바 - 전체 요약
    with st.sidebar:
        st.header("📊 분석 요약")
        
        st.metric(
            label="총 슬라이스",
            value=f"{results['total_slices']} 개"
        )
        
        st.metric(
            label="종양 검출 슬라이스",
            value=f"{summary.get('affected_slices', 0)} 개",
            delta=f"{summary.get('affected_slices', 0)/results['total_slices']*100:.1f}%"
        )
        
        st.metric(
            label="검출된 종양 영역",
            value=f"{summary.get('total_tumor_regions', 0)} 개"
        )
        
        st.markdown("---")
        
        st.subheader("🎯 종양 크기")
        st.write(f"**총 면적**: {summary.get('total_area_mm2', 0):.2f} mm²")
        st.write(f"**추정 부피**: {summary.get('total_volume_mm3', 0):.2f} mm³")
        st.write(f"**부피 (mL)**: {summary.get('total_volume_ml', 0):.4f} mL")
        
        if summary.get('bounding_box_3d'):
            st.markdown("---")
            st.subheader("📦 3D 바운딩 박스")
            bbox = summary['bounding_box_3d']
            st.write(f"**X**: {bbox['x_min']} ~ {bbox['x_max']} ({bbox['x_max']-bbox['x_min']} px)")
            st.write(f"**Y**: {bbox['y_min']} ~ {bbox['y_max']} ({bbox['y_max']-bbox['y_min']} px)")
            st.write(f"**Z**: {bbox['z_min']} ~ {bbox['z_max']} ({bbox['z_max']-bbox['z_min']} slices)")
    
    # 메인 영역 - 탭 구성
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 통계 대시보드",
        "🔍 슬라이스별 검출 결과",
        "📈 3D 분포 시각화",
        "📋 상세 데이터"
    ])
    
    # 탭 1: 통계 대시보드
    with tab1:
        st.header("종양 검출 통계")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric(
                label="검출률",
                value=f"{summary.get('affected_slices', 0)/results['total_slices']*100:.1f}%",
                help="전체 슬라이스 중 종양이 검출된 비율"
            )
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric(
                label="평균 영역 크기",
                value=f"{summary.get('total_area_pixels', 0) / max(summary.get('total_tumor_regions', 1), 1):.0f} px",
                help="검출된 종양 영역의 평균 크기"
            )
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col3:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric(
                label="슬라이스당 평균 영역",
                value=f"{summary.get('total_tumor_regions', 0) / max(summary.get('affected_slices', 1), 1):.1f} 개",
                help="종양이 있는 슬라이스당 평균 검출 영역 수"
            )
            st.markdown('</div>', unsafe_allow_html=True)
        
        # 슬라이스별 종양 분포 차트
        if results.get('tumor_detections'):
            st.subheader("슬라이스별 종양 영역 분포")
            
            # 데이터 준비
            slice_counts = {}
            for det in results['tumor_detections']:
                slice_num = det['slice_num']
                slice_counts[slice_num] = slice_counts.get(slice_num, 0) + 1
            
            df = pd.DataFrame(list(slice_counts.items()), columns=['Slice Number', 'Tumor Count'])
            df = df.sort_values('Slice Number')
            
            fig = px.bar(
                df, 
                x='Slice Number', 
                y='Tumor Count',
                title='슬라이스별 검출된 종양 영역 개수',
                labels={'Tumor Count': '종양 영역 개수', 'Slice Number': '슬라이스 번호'}
            )
            fig.update_traces(marker_color='#1f77b4')
            st.plotly_chart(fig, use_container_width=True)
            
            # 종양 크기 분포
            st.subheader("종양 크기 분포")
            
            areas = [det['area_pixels'] for det in results['tumor_detections']]
            df_area = pd.DataFrame({'Area (pixels)': areas})
            
            fig2 = px.histogram(
                df_area,
                x='Area (pixels)',
                nbins=50,
                title='종양 영역 크기 히스토그램',
                labels={'count': '빈도', 'Area (pixels)': '면적 (픽셀)'}
            )
            fig2.update_traces(marker_color='#ff7f0e')
            st.plotly_chart(fig2, use_container_width=True)
    
    # 탭 2: 슬라이스별 검출 결과
    with tab2:
        st.header("슬라이스별 종양 검출 결과")
        
        affected_slices = sorted(results.get('slices_with_tumors', []))
        
        if not affected_slices:
            st.info("검출된 종양이 없습니다.")
        else:
            # 슬라이스 선택
            col1, col2 = st.columns([3, 1])
            
            with col1:
                selected_slice = st.select_slider(
                    "슬라이스 선택",
                    options=affected_slices,
                    value=affected_slices[len(affected_slices)//2]
                )
            
            with col2:
                st.metric("현재 슬라이스", f"{selected_slice}")
            
            # 선택된 슬라이스 정보
            slice_detections = [d for d in results['tumor_detections'] if d['slice_num'] == selected_slice]
            
            st.subheader(f"슬라이스 #{selected_slice} 상세 정보")
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                # 검출 결과 이미지
                img = load_detection_image(selected_slice)
                if img:
                    st.image(img, caption=f"슬라이스 #{selected_slice} - 종양 검출 결과", use_container_width=True)
                else:
                    st.warning("이미지를 불러올 수 없습니다.")
            
            with col2:
                st.write(f"**검출된 종양 영역**: {len(slice_detections)} 개")
                
                if slice_detections:
                    total_area = sum(d['area_pixels'] for d in slice_detections)
                    st.write(f"**총 면적**: {total_area:.0f} px")
                    
                    st.markdown("---")
                    st.write("**영역별 상세 정보:**")
                    
                    for i, det in enumerate(slice_detections[:5], 1):
                        with st.expander(f"영역 #{i}"):
                            st.write(f"중심: ({det['center'][0]}, {det['center'][1]})")
                            st.write(f"면적: {det['area_pixels']:.0f} px")
                            st.write(f"등가 직경: {det['equivalent_diameter']:.1f} px")
                            x, y, w, h = det['bounding_box']
                            st.write(f"바운딩 박스: ({x}, {y}, {w}, {h})")
                    
                    if len(slice_detections) > 5:
                        st.info(f"외 {len(slice_detections) - 5} 개 영역")
    
    # 탭 3: 3D 분포 시각화
    with tab3:
        st.header("3D 종양 분포")
        
        if results.get('tumor_detections'):
            # 3D 산점도
            centers = [(d['center'][0], d['center'][1], d['slice_num']) for d in results['tumor_detections']]
            areas = [d['area_pixels'] for d in results['tumor_detections']]
            
            df_3d = pd.DataFrame(centers, columns=['X', 'Y', 'Z'])
            df_3d['Area'] = areas
            
            fig = px.scatter_3d(
                df_3d, 
                x='X', 
                y='Y', 
                z='Z',
                size='Area',
                color='Area',
                title='종양 영역 3D 분포',
                labels={'X': 'X (픽셀)', 'Y': 'Y (픽셀)', 'Z': '슬라이스 번호', 'Area': '면적 (px)'},
                color_continuous_scale='Reds'
            )
            
            fig.update_layout(height=600)
            st.plotly_chart(fig, use_container_width=True)
            
            # 바운딩 박스 표시
            if summary.get('bounding_box_3d'):
                st.subheader("3D 바운딩 박스")
                bbox = summary['bounding_box_3d']
                
                # 바운딩 박스 와이어프레임
                x = [bbox['x_min'], bbox['x_max'], bbox['x_max'], bbox['x_min'], bbox['x_min'], 
                     bbox['x_min'], bbox['x_max'], bbox['x_max'], bbox['x_min'], bbox['x_min']]
                y = [bbox['y_min'], bbox['y_min'], bbox['y_max'], bbox['y_max'], bbox['y_min'],
                     bbox['y_min'], bbox['y_min'], bbox['y_max'], bbox['y_max'], bbox['y_min']]
                z = [bbox['z_min'], bbox['z_min'], bbox['z_min'], bbox['z_min'], bbox['z_min'],
                     bbox['z_max'], bbox['z_max'], bbox['z_max'], bbox['z_max'], bbox['z_max']]
                
                fig_bbox = go.Figure(data=[go.Scatter3d(
                    x=x, y=y, z=z,
                    mode='lines',
                    line=dict(color='blue', width=5),
                    name='Bounding Box'
                )])
                
                # 종양 중심점 추가
                fig_bbox.add_trace(go.Scatter3d(
                    x=df_3d['X'],
                    y=df_3d['Y'],
                    z=df_3d['Z'],
                    mode='markers',
                    marker=dict(
                        size=3,
                        color=df_3d['Area'],
                        colorscale='Reds',
                        showscale=True
                    ),
                    name='Tumor Centers'
                ))
                
                fig_bbox.update_layout(
                    title='종양 영역과 3D 바운딩 박스',
                    scene=dict(
                        xaxis_title='X (픽셀)',
                        yaxis_title='Y (픽셀)',
                        zaxis_title='슬라이스 번호'
                    ),
                    height=600
                )
                
                st.plotly_chart(fig_bbox, use_container_width=True)
        else:
            st.info("검출된 종양이 없습니다.")
    
    # 탭 4: 상세 데이터
    with tab4:
        st.header("상세 분석 데이터")
        
        # 종합 정보
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("기본 정보")
            st.json({
                "총 슬라이스": results['total_slices'],
                "종양 검출 슬라이스": summary.get('affected_slices', 0),
                "총 종양 영역": summary.get('total_tumor_regions', 0),
                "슬라이스 범위": summary.get('slice_range', {})
            })
        
        with col2:
            st.subheader("종양 크기 정보")
            st.json({
                "총 면적 (픽셀)": summary.get('total_area_pixels', 0),
                "총 면적 (mm²)": summary.get('total_area_mm2', 0),
                "추정 부피 (mm³)": summary.get('total_volume_mm3', 0),
                "추정 부피 (mL)": summary.get('total_volume_ml', 0)
            })
        
        # 검출 데이터 테이블
        if results.get('tumor_detections'):
            st.subheader("검출된 종양 영역 데이터")
            
            df_detections = pd.DataFrame(results['tumor_detections'])
            
            # 중심 좌표 분리
            df_detections['center_x'] = df_detections['center'].apply(lambda x: x[0])
            df_detections['center_y'] = df_detections['center'].apply(lambda x: x[1])
            
            # 바운딩 박스 분리
            df_detections['bbox_x'] = df_detections['bounding_box'].apply(lambda x: x[0])
            df_detections['bbox_y'] = df_detections['bounding_box'].apply(lambda x: x[1])
            df_detections['bbox_w'] = df_detections['bounding_box'].apply(lambda x: x[2])
            df_detections['bbox_h'] = df_detections['bounding_box'].apply(lambda x: x[3])
            
            # 표시할 컬럼 선택
            display_df = df_detections[[
                'slice_num', 'center_x', 'center_y', 
                'area_pixels', 'equivalent_diameter',
                'bbox_x', 'bbox_y', 'bbox_w', 'bbox_h'
            ]].copy()
            
            display_df.columns = [
                '슬라이스', '중심_X', '중심_Y',
                '면적(px)', '등가직경',
                'BBox_X', 'BBox_Y', 'BBox_W', 'BBox_H'
            ]
            
            st.dataframe(display_df, use_container_width=True, height=400)
            
            # CSV 다운로드
            csv = display_df.to_csv(index=False, encoding='utf-8-sig')
            st.download_button(
                label="📥 CSV 다운로드",
                data=csv,
                file_name="tumor_detection_results.csv",
                mime="text/csv"
            )

if __name__ == "__main__":
    main()
