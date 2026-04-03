"""
DICOM Batch Tumor Detection - ADDS UI Integration (ADVANCED with OpenAI)
==========================================================================
Complete version with:
- 3-panel visualization for each file
- AI analysis reports with OpenAI medical interpretation
- Statistical charts and graphs
- Meaningful analysis text
"""
import streamlit as st
from pathlib import Path
import sys
import json
import time
from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import io
import os
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from src.medical_imaging.detection.candidate_detector import TumorDetector
import pydicom


# ============================================================================
# OpenAI Integration
# ============================================================================

def get_openai_client():
    """Get OpenAI client if API key is available"""
    api_key = os.getenv('OPENAI_API_KEY')
    if api_key:
        try:
            return OpenAI(api_key=api_key)
        except Exception as e:
            st.warning(f"⚠️ OpenAI 클라이언트 초기화 실패: {e}")
            return None
    return None


def generate_medical_interpretation(client, detection_data):
    """
    Generate medical interpretation using OpenAI
    
    Args:
        client: OpenAI client
        detection_data: Dict with detection results
    
    Returns:
        str: Medical interpretation text
    """
    if not client:
        return None
    
    try:
        prompt = f"""As a medical AI assistant specializing in radiology, analyze this CT tumor detection result and provide a brief clinical interpretation in Korean:

Detection Results:
- Slice ID: {detection_data['slice_name']}
- Tumor detected: {'Yes' if detection_data['has_tumor'] else 'No'}
- Number of high-confidence candidates: {detection_data['high_conf_candidates']}
- Total candidates: {detection_data['total_candidates']}
- Maximum confidence score: {detection_data['max_confidence']:.1%}

Provide:
1. Clinical significance (2-3 sentences)
2. Recommendation (1-2 sentences)

Keep it concise, professional, and in Korean. Include disclaimer that this is AI-assisted analysis requiring physician verification."""

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are an expert medical AI assistant specializing in radiology and oncology. Provide clear, evidence-based clinical interpretations in Korean."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=500,
            temperature=0.3
        )
        
        return response.choices[0].message.content
        
    except Exception as e:
        st.warning(f"⚠️ OpenAI 분석 오류: {e}")
        return None


def generate_batch_summary_interpretation(client, batch_stats):
    """
    Generate comprehensive batch analysis using OpenAI
    
    Args:
        client: OpenAI client
        batch_stats: Dict with batch statistics
    
    Returns:
        str: Comprehensive analysis text
    """
    if not client:
        return None
    
    try:
        prompt = f"""As a medical data analyst, provide a comprehensive interpretation of this DICOM batch tumor detection analysis in Korean:

Batch Statistics:
- Total cases processed: {batch_stats['total_cases']}
- Tumor detected: {batch_stats['tumor_detected']} ({batch_stats['detection_rate']:.1f}%)
- Normal cases: {batch_stats['normal_cases']}
- Average confidence: {batch_stats['avg_confidence']:.1%}
- Confidence range: {batch_stats['min_confidence']:.1%} - {batch_stats['max_confidence']:.1%}
- Average candidates per positive case: {batch_stats['avg_candidates']:.1f}

Provide a structured analysis covering:
1. **전체 평가** (Overall Assessment): 검출율과 신뢰도에 대한 평가
2. **임상적 의미** (Clinical Significance): 이 결과가 시사하는 바
3. **데이터 품질** (Data Quality): 신뢰도 분포에 대한 평가
4. **권장사항** (Recommendations): 추가 조치나 고려사항

Use professional medical language in Korean. Keep each section to 2-3 sentences. Include appropriate disclaimer."""

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are an expert medical data analyst specializing in radiology and oncology screening programs."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=800,
            temperature=0.4
        )
        
        return response.choices[0].message.content
        
    except Exception as e:
        st.warning(f"⚠️ 배치 분석 오류: {e}")
        return None


# ============================================================================
# Statistical Charts
# ============================================================================

def create_statistics_dashboard(results):
    """
    Create comprehensive statistics dashboard with multiple charts
    
    Args:
        results: List of detection results
    
    Returns:
        BytesIO: PNG image buffer
    """
    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(3, 3, figure=fig, hspace=0.4, wspace=0.3)
    
    # Extract data
    confidences = [r['max_confidence'] for r in results]
    has_tumor = [r['has_tumor'] for r in results]
    total_candidates = [r['total_candidates'] for r in results]
    high_conf_candidates = [r['high_conf_candidates'] for r in results]
    
    tumor_confidences = [r['max_confidence'] for r in results if r['has_tumor']]
    normal_confidences = [r['max_confidence'] for r in results if not r['has_tumor']]
    
    # 1. Detection Rate Pie Chart
    ax1 = fig.add_subplot(gs[0, 0])
    tumor_count = sum(has_tumor)
    normal_count = len(has_tumor) - tumor_count
    colors = ['#ff4444', '#44ff44']
    ax1.pie([tumor_count, normal_count], labels=['Tumor', 'Normal'], 
            autopct='%1.1f%%', colors=colors, startangle=90)
    ax1.set_title('Detection Results', fontsize=12, fontweight='bold')
    
    # 2. Confidence Distribution (All)
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.hist(confidences, bins=20, color='steelblue', alpha=0.7, edgecolor='black')
    ax2.axvline(np.mean(confidences), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(confidences):.1%}')
    ax2.set_xlabel('Confidence')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Confidence Distribution', fontsize=12, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Confidence by Category
    ax3 = fig.add_subplot(gs[0, 2])
    if tumor_confidences and normal_confidences:
        bp = ax3.boxplot([tumor_confidences, normal_confidences], 
                         labels=['Tumor', 'Normal'],
                         patch_artist=True)
        bp['boxes'][0].set_facecolor('#ff9999')
        bp['boxes'][1].set_facecolor('#99ff99')
    ax3.set_ylabel('Confidence')
    ax3.set_title('Confidence by Category', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='y')
    
    # 4. Candidate Count Distribution
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.hist(total_candidates, bins=15, color='orange', alpha=0.7, edgecolor='black')
    ax4.set_xlabel('Candidate Count')
    ax4.set_ylabel('Frequency')
    ax4.set_title('Candidate Distribution', fontsize=12, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    
    # 5. High-Confidence Candidates
    ax5 = fig.add_subplot(gs[1, 1])
    high_conf_count = [c for c in high_conf_candidates if c > 0]
    if high_conf_count:
        ax5.hist(high_conf_count, bins=max(10, len(set(high_conf_count))), 
                color='crimson', alpha=0.7, edgecolor='black')
    ax5.set_xlabel('High-Conf Candidates')
    ax5.set_ylabel('Frequency')
    ax5.set_title('High-Confidence Distribution', fontsize=12, fontweight='bold')
    ax5.grid(True, alpha=0.3)
    
    # 6. Cumulative Detection Rate
    ax6 = fig.add_subplot(gs[1, 2])
    sorted_conf = sorted(confidences, reverse=True)
    cumulative = np.arange(1, len(sorted_conf) + 1)
    ax6.plot(sorted_conf, cumulative, linewidth=2, color='darkblue')
    ax6.fill_between(sorted_conf, cumulative, alpha=0.3, color='lightblue')
    ax6.set_xlabel('Confidence Threshold')
    ax6.set_ylabel('Cumulative Detections')
    ax6.set_title('Cumulative Detection Curve', fontsize=12, fontweight='bold')
    ax6.grid(True, alpha=0.3)
    
    # 7. Summary Statistics Table (Smaller)
    ax7 = fig.add_subplot(gs[2, :])
    ax7.axis('off')
    
    stats_data = [
        ['Total Cases', f"{len(results)}"],
        ['Tumor Detected', f"{tumor_count} ({tumor_count/len(results)*100:.1f}%)"],
        ['Avg Confidence', f"{np.mean(confidences):.1%}"],
        ['Max Confidence', f"{max(confidences):.1%}"],
    ]
    
    table = ax7.table(cellText=stats_data, 
                     colLabels=['Metric', 'Value'],
                     cellLoc='center',
                     loc='center',
                     colWidths=[0.25, 0.15])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5)
    
    # Style table
    for i in range(len(stats_data) + 1):
        for j in range(2):
            cell = table[(i, j)]
            if i == 0:
                cell.set_facecolor('#4472C4')
                cell.set_text_props(weight='bold', color='white')
            else:
                cell.set_facecolor('#E7E6E6' if i % 2 == 0 else 'white')
    
    plt.suptitle('DICOM Batch Analysis Dashboard', fontsize=16, fontweight='bold', y=0.98)
    
    # Save to buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    plt.close()
    
    return buf


# ============================================================================
# Visualization Functions (from original)
# ============================================================================

def create_detection_visualization(hu_slice, candidates, high_conf_candidates, slice_name):
    """Create 3-panel detection visualization"""
    fig = plt.figure(figsize=(15, 5))
    gs = GridSpec(1, 3, figure=fig, hspace=0.3, wspace=0.3)
    
    # Normalize for visualization
    ct_display = np.clip(hu_slice, -160, 240)
    ct_display = (ct_display + 160) / 400
    
    # 1. Original slice
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(ct_display.T, cmap='gray', origin='lower')
    ax1.set_title(f'Original CT\n{slice_name}', fontsize=12, fontweight='bold')
    ax1.axis('off')
    
    # 2. All candidates
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.imshow(ct_display.T, cmap='gray', origin='lower')
    ax2.set_title(f'All Candidates (n={len(candidates)})', fontsize=12, fontweight='bold')
    
    for candidate in candidates[:50]:
        x, y = candidate.centroid
        radius = max(5, min(30, np.sqrt(candidate.area_pixels / np.pi)))
        conf = candidate.confidence_score
        
        color = 'yellow' if conf > 0.7 else 'orange'
        alpha = 0.3 + 0.4 * conf
        
        circle = plt.Circle((x, y), radius, color=color, fill=False, 
                          linewidth=2, alpha=alpha)
        ax2.add_patch(circle)
    ax2.axis('off')
    
    # 3. High-confidence with mask
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.imshow(ct_display.T, cmap='gray', origin='lower')
    
    has_tumor = len(high_conf_candidates) > 0
    
    if has_tumor:
        mask = np.zeros_like(hu_slice, dtype=bool)
        for candidate in high_conf_candidates:
            x, y = candidate.centroid
            radius = max(5, min(30, np.sqrt(candidate.area_pixels / np.pi)))
            
            yy, xx = np.ogrid[:hu_slice.shape[1], :hu_slice.shape[0]]
            circle_mask = (xx - x)**2 + (yy - y)**2 <= radius**2
            mask[circle_mask.T] = True
            
            circle = plt.Circle((x, y), radius, color='red', fill=False, 
                              linewidth=3, alpha=0.9)
            ax3.add_patch(circle)
            
            ax3.text(x, y - radius - 5, f'{candidate.confidence_score:.1%}', 
                    color='red', fontsize=9, fontweight='bold',
                    ha='center', va='bottom',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9))
        
        red_overlay = np.zeros((*hu_slice.shape, 4))
        red_overlay[mask.T] = [1, 0, 0, 0.3]
        ax3.imshow(red_overlay, origin='lower')
        
        ax3.set_title(f'✓ TUMOR DETECTED (n={len(high_conf_candidates)})', 
                     fontsize=12, fontweight='bold', color='red')
    else:
        ax3.set_title('✓ NO TUMOR FOUND', fontsize=12, fontweight='bold', color='green')
    
    ax3.axis('off')
    
    status = "TUMOR DETECTED" if has_tumor else "NO TUMOR"
    status_color = "red" if has_tumor else "green"
    plt.suptitle(f'{slice_name} - {status}', 
                fontsize=14, fontweight='bold', color=status_color)
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    plt.close()
    
    return buf


def generate_ai_report(slice_name, candidates, high_conf_candidates, max_confidence, medical_interpretation=None):
    """Generate AI analysis report with optional OpenAI interpretation"""
    has_tumor = len(high_conf_candidates) > 0
    
    report = f"### {'🔴 **종양 검출: 양성**' if has_tumor else '🟢 **종양 검출: 음성**'}\n\n"
    report += f"**환자 ID/슬라이스**: {slice_name}\n\n"
    
    # Basic statistics
    report += "**검출 통계**:\n"
    report += f"- 고신뢰도 종양 후보: {len(high_conf_candidates)}개\n"
    report += f"- 전체 후보: {len(candidates)}개\n"
    report += f"- 최고 신뢰도: {max_confidence:.1%}\n\n"
    
    if has_tumor and high_conf_candidates:
        report += "**주요 후보 정보**:\n"
        for idx, candidate in enumerate(high_conf_candidates[:3], 1):
            x, y = candidate.centroid
            area_mm2 = candidate.area_mm2
            conf = candidate.confidence_score
            report += f"{idx}. 위치: ({x:.0f}, {y:.0f}), 크기: {area_mm2:.1f}mm², 신뢰도: {conf:.1%}\n"
        
        if len(high_conf_candidates) > 3:
            report += f"*...외 {len(high_conf_candidates)-3}개*\n"
        report += "\n"
    
    # OpenAI medical interpretation
    if medical_interpretation:
        report += "---\n\n"
        report += "### 🤖 AI 의학적 해석\n\n"
        report += medical_interpretation + "\n\n"
    
    # Recommendations
    if has_tumor:
        report += "**권장 조치**:\n"
        report += "- 🔍 정밀 영상 검사 권장\n"
        report += "- 👨‍⚕️ 전문의 상담 필요\n"
        report += "- 📊 추적 관찰 계획 수립\n"
    else:
        report += "**권장 조치**:\n"
        report += "- ✅ 정기 검진 유지\n"
        report += "- 📅 표준 추적 관찰 일정 준수\n"
    
    return report


# ============================================================================
# Processing Functions
# ============================================================================

def process_files(files, is_uploaded, confidence, min_area, max_area, hu_range, use_openai=True):
    """Process DICOM files with full analysis"""
    
    # Initialize detector
    with st.spinner("Initializing detector..."):
        detector = TumorDetector(
            min_area_mm2=min_area,
            max_area_mm2=max_area,
            hu_range=hu_range
        )
    
    # Initialize OpenAI client if requested
    openai_client = None
    if use_openai:
        openai_client = get_openai_client()
        if openai_client:
            st.success("✅ OpenAI 클라이언트 연결됨")
        else:
            st.warning("⚠️ OpenAI API를 사용할 수 없습니다. 기본 분석만 제공됩니다.")
    
    # Progress containers
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    results = []
    start_time = time.time()
    
    # Process each file
    for idx, file_item in enumerate(files):
        try:
            # Load DICOM
            if is_uploaded:
                dcm = pydicom.dcmread(io.BytesIO(file_item.read()))
                slice_name = file_item.name.replace('.dcm', '')
            else:
                dcm = pydicom.dcmread(file_item)
                slice_name = file_item.stem
            
            # Extract HU values
            pixel_array = dcm.pixel_array.astype(float)
            slope = getattr(dcm, 'RescaleSlope', 1)
            intercept = getattr(dcm, 'RescaleIntercept', 0)
            hu_slice = pixel_array * slope + intercept
            
            # Pixel spacing
            spacing = getattr(dcm, 'PixelSpacing', [1.0, 1.0])
            pixel_spacing = (float(spacing[0]), float(spacing[1]))
            
            # Detect candidates
            candidates = detector.detect_candidates_2d(
                hu_slice=hu_slice,
                pixel_spacing=pixel_spacing,
                method='multi_threshold'
            )
            
            # Filter by confidence
            high_conf_candidates = [c for c in candidates if c.confidence_score >= confidence]
            max_conf = max([c.confidence_score for c in candidates]) if candidates else 0
            
            # Create visualization
            viz_image = create_detection_visualization(
                hu_slice, candidates, high_conf_candidates, slice_name
            )
            
            # Generate OpenAI interpretation
            medical_interpretation = None
            if openai_client:
                detection_data = {
                    'slice_name': slice_name,
                    'has_tumor': len(high_conf_candidates) > 0,
                    'total_candidates': len(candidates),
                    'high_conf_candidates': len(high_conf_candidates),
                    'max_confidence': max_conf
                }
                medical_interpretation = generate_medical_interpretation(openai_client, detection_data)
            
            # Generate AI report
            ai_report = generate_ai_report(
                slice_name, candidates, high_conf_candidates, max_conf, medical_interpretation
            )
            
            # Store result
            results.append({
                'slice_name': slice_name,
                'has_tumor': len(high_conf_candidates) > 0,
                'total_candidates': len(candidates),
                'high_conf_candidates': len(high_conf_candidates),
                'max_confidence': max_conf,
                'visualization': viz_image,
                'ai_report': ai_report,
                'medical_interpretation': medical_interpretation
            })
            
            # Update progress
            progress = (idx + 1) / len(files)
            progress_bar.progress(progress)
            status_text.text(f"Processing {slice_name}... ({idx+1}/{len(files)})")
            
        except Exception as e:
            st.warning(f"⚠️ 파일 처리 오류: {slice_name if 'slice_name' in locals() else 'unknown'} - {str(e)}")
            continue
    
    elapsed_time = time.time() - start_time
    
    # Generate statistics dashboard
    stats_chart = create_statistics_dashboard(results)
    
    # Generate batch summary with OpenAI
    batch_summary = None
    if openai_client and results:
        tumor_count = sum(1 for r in results if r['has_tumor'])
        confidences = [r['max_confidence'] for r in results]
        tumor_candidates = [r['high_conf_candidates'] for r in results if r['has_tumor']]
        
        batch_stats = {
            'total_cases': len(results),
            'tumor_detected': tumor_count,
            'normal_cases': len(results) - tumor_count,
            'detection_rate': tumor_count / len(results) * 100 if results else 0,
            'avg_confidence': np.mean(confidences),
            'min_confidence': min(confidences),
            'max_confidence': max(confidences),
            'avg_candidates': np.mean(tumor_candidates) if tumor_candidates else 0
        }
        
        batch_summary = generate_batch_summary_interpretation(openai_client, batch_stats)
    
    # Store in session state
    st.session_state['batch_results'] = results
    st.session_state['batch_elapsed_time'] = elapsed_time
    st.session_state['stats_chart'] = stats_chart
    st.session_state['batch_summary'] = batch_summary
    
    # Complete
    progress_bar.progress(1.0)
    status_text.text("✅ 배치 처리 완료!")
    st.success(f"🎉 {len(results)}개 파일 처리 완료! ({elapsed_time:.1f}초 소요)")
    st.rerun()


# ============================================================================
# Display Functions
# ============================================================================

def display_gallery_view(results, show_tumor_only=False, show_normal_only=False):
    """
    Display results in a scrollable gallery grid format
    
    Args:
        results: List of detection results
        show_tumor_only: Filter to show only tumor cases
        show_normal_only: Filter to show only normal cases
    """
    st.markdown("---")
    st.subheader("🎞️ 이미지 갤러리")
    st.caption("각 이미지를 클릭하면 상세 정보를 볼 수 있습니다")
    
    # Filter results
    filtered_results = []
    for result in results:
        if show_tumor_only and not result['has_tumor']:
            continue
        if show_normal_only and result['has_tumor']:
            continue
        filtered_results.append(result)
    
    if not filtered_results:
        st.info("필터 조건에 맞는 결과가 없습니다.")
        return
    
    # Create gallery grid (3 columns per row)
    cols_per_row = 3
    num_results = len(filtered_results)
    num_rows = (num_results + cols_per_row - 1) // cols_per_row
    
    idx = 0
    for row in range(num_rows):
        cols = st.columns(cols_per_row)
        
        for col_idx, col in enumerate(cols):
            if idx >= num_results:
                break
            
            result = filtered_results[idx]
            
            with col:
                # Status badge
                status_icon = "🔴" if result['has_tumor'] else "🟢"
                status_text = "종양 검출" if result['has_tumor'] else "정상"
                status_color = "red" if result['has_tumor'] else "green"
                
                # Display image
                st.image(result['visualization'], use_container_width=True)
                
                # Image metadata
                st.markdown(f"**{status_icon} {result['slice_name']}**")
                st.caption(f"신뢰도: {result['max_confidence']:.1%}")
                st.caption(f"후보: {result['high_conf_candidates']}개 / {result['total_candidates']}개")
                
                # Expander for quick details
                with st.expander("📋 상세 정보", expanded=False):
                    st.markdown(result['ai_report'])
            
            idx += 1
        
        # Add spacing between rows
        if row < num_rows - 1:
            st.markdown("")


def display_results():
    """Display comprehensive results with charts and AI analysis"""
    
    results = st.session_state['batch_results']
    elapsed_time = st.session_state.get('batch_elapsed_time', 0)
    stats_chart = st.session_state.get('stats_chart')
    batch_summary = st.session_state.get('batch_summary')
    
    st.markdown("---")
    st.header("📊 배치 분석 결과")
    
    # Summary metrics
    tumor_count = sum(1 for r in results if r['has_tumor'])
    confidences = [r['max_confidence'] for r in results]
    avg_confidence = np.mean(confidences)
    
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("총 케이스", len(results))
    col2.metric("종양 검출", tumor_count, delta=f"{tumor_count/len(results)*100:.1f}%")
    col3.metric("정상", len(results) - tumor_count)
    col4.metric("평균 신뢰도", f"{avg_confidence*100:.1f}%")
    col5.metric("처리 시간", f"{elapsed_time:.1f}s")
    
    # Statistics Dashboard
    if stats_chart:
        st.markdown("---")
        st.subheader("📈 통계 대시보드")
        st.image(stats_chart, use_container_width=True)
    
    # AI Batch Summary
    if batch_summary:
        st.markdown("---")
        st.subheader("🤖 AI 종합 분석")
        st.markdown(batch_summary)
    
    # View mode and filter options
    st.markdown("---")
    
    col_view, col_filter1, col_filter2 = st.columns([2, 1, 1])
    
    with col_view:
        view_mode = st.radio(
            "보기 모드",
            ["📋 상세 리스트 보기", "🎞️ 갤러리 보기"],
            horizontal=True,
            key="view_mode"
        )
    
    with col_filter1:
        show_tumor_only = st.checkbox("🔴 종양만", value=False)
    
    with col_filter2:
        show_normal_only = st.checkbox("🟢 정상만", value=False)
    
    # Display based on selected view mode
    if view_mode == "🎞️ 갤러리 보기":
        display_gallery_view(results, show_tumor_only, show_normal_only)
    else:
        # Individual results (detailed list view)
        st.markdown("---")
        st.subheader("🔬 개별 분석 결과")
        
        for idx, result in enumerate(results):
            # Apply filters
            if show_tumor_only and not result['has_tumor']:
                continue
            if show_normal_only and result['has_tumor']:
                continue
            
            # Status styling
            status_icon = "🔴" if result['has_tumor'] else "🟢"
            
            # Expandable container
            with st.expander(
                f"{status_icon} **{result['slice_name']}** - "
                f"신뢰도: {result['max_confidence']:.1%} - "
                f"후보: {result['high_conf_candidates']}개",
                expanded=(idx == 0)
            ):
                # Visualization
                st.image(result['visualization'], use_container_width=True)
                
                # AI Report
                st.markdown(result['ai_report'])
                
                # Stats
                col1, col2, col3 = st.columns(3)
                col1.metric("전체 후보", result['total_candidates'])
                col2.metric("고신뢰도 후보", result['high_conf_candidates'])
                col3.metric("최고 신뢰도", f"{result['max_confidence']:.1%}")
    
    # Download section
    st.markdown("---")
    st.subheader("💾 결과 다운로드")
    
    download_data = [{
        'slice_name': r['slice_name'],
        'has_tumor': r['has_tumor'],
        'total_candidates': r['total_candidates'],
        'high_conf_candidates': r['high_conf_candidates'],
        'max_confidence': r['max_confidence']
    } for r in results]
    
    col1, col2 = st.columns(2)
    
    with col1:
        json_data = json.dumps(download_data, indent=2, ensure_ascii=False)
        st.download_button(
            label="📄 JSON 다운로드",
            data=json_data,
            file_name=f"batch_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json",
            use_container_width=True
        )
    
    with col2:
        df = pd.DataFrame(download_data)
        csv_data = df.to_csv(index=False)
        st.download_button(
            label="📊 CSV 다운로드",
            data=csv_data,
            file_name=f"batch_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            use_container_width=True
        )


# ============================================================================
# Main UI Function
# ============================================================================

def show_dicom_batch_analysis():
    """Main Streamlit UI - ADVANCED VERSION"""
    
    st.title("📁 DICOM Batch Tumor Detection (Advanced)")
    st.markdown("---")
    st.markdown("""
    **고급 기능**:
    - 🎨 3-Panel 시각화 (각 파일)
    - 🤖 OpenAI 의학적 해석
    - 📊 통계 차트 및 그래프
    - 📈 유의미한 분석 텍스트
    """)
    
    # Settings
    st.subheader("⚙️ 입력 방법 선택")
    
    tab1, tab2 = st.tabs(["📤 파일 업로드", "📁 폴더 선택"])
    
    # Common settings
    def get_settings(prefix):
        col1, col2 = st.columns(2)
        with col1:
            confidence = st.slider(
                "신뢰도 임계값 (%)",
                min_value=50,
                max_value=99,
                value=70,
                key=f"conf_{prefix}"
            ) / 100
            
            use_openai = st.checkbox(
                "🤖 OpenAI 의학적 해석 사용",
                value=True,
                key=f"openai_{prefix}",
                help="OpenAI API를 사용하여 각 케이스에 대한 의학적 해석 제공"
            )
        
        with col2:
            st.metric("검출 파라미터", f"{confidence*100:.0f}%",
                     delta="High Sensitivity" if confidence < 0.75 else "High Precision")
        
        with st.expander("🔧 고급 설정"):
            min_area = st.number_input("최소 영역 (mm²)", value=10.0, key=f"min_{prefix}")
            max_area = st.number_input("최대 영역 (mm²)", value=10000.0, key=f"max_{prefix}")
            hu_min = st.number_input("HU 최소값", value=-50, key=f"humin_{prefix}")
            hu_max = st.number_input("HU 최대값", value=200, key=f"humax_{prefix}")
        
        return confidence, min_area, max_area, hu_min, hu_max, use_openai
    
    # Tab 1: Upload
    with tab1:
        uploaded_files = st.file_uploader(
            "DICOM 파일 선택",
            type=['dcm'],
            accept_multiple_files=True
        )
        
        if uploaded_files:
            st.success(f"✅ {len(uploaded_files)}개 파일 선택됨")
        
        confidence, min_area, max_area, hu_min, hu_max, use_openai = get_settings("upload")
        
        st.markdown("---")
        if st.button("🚀 분석 시작", type="primary", use_container_width=True, key="start_upload"):
            if uploaded_files:
                process_files(
                    uploaded_files, True, confidence,
                    min_area, max_area, (hu_min, hu_max), use_openai
                )
            else:
                st.error("❌ 파일을 선택하세요!")
    
    # Tab 2: Folder
    with tab2:
        col1, col2 = st.columns(2)
        with col1:
            folder_path = st.text_input("DICOM 폴더 경로", value="CTdcm")
        with col2:
            num_files = st.number_input("파일 개수", min_value=1, max_value=50, value=10)
        
        confidence, min_area, max_area, hu_min, hu_max, use_openai = get_settings("folder")
        
        st.markdown("---")
        if st.button("🚀 분석 시작", type="primary", use_container_width=True, key="start_folder"):
            folder = Path(folder_path)
            if not folder.is_absolute():
                folder = project_root / folder
            
            if not folder.exists():
                st.error(f"❌ 폴더를 찾을 수 없습니다: {folder}")
            else:
                dcm_files = list(folder.glob("*.dcm"))
                if not dcm_files:
                    st.error("❌ DICOM 파일이 없습니다")
                else:
                    import random
                    random.seed(42)
                    selected = random.sample(dcm_files, min(num_files, len(dcm_files)))
                    process_files(
                        selected, False, confidence,
                        min_area, max_area, (hu_min, hu_max), use_openai
                    )
    
    # Display results if available
    if 'batch_results' in st.session_state:
        display_results()


# Entry point
if __name__ == "__main__":
    show_dicom_batch_analysis()
