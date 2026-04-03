"""
Clinical EMR Dashboard Mode
Professional hospital EMR-style dashboard for daily clinical work
High information density, table-focused layout
"""

import streamlit as st
import plotly.graph_objects as go
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from ui.utils.dashboard_data import (
    get_patient_cases,
    get_system_metrics,
    get_team_activity
)


def show_home_clinical():
    """Clinical EMR-style dashboard for daily work"""
    
    # Get data
    metrics = get_system_metrics()
    patients_df = get_patient_cases(limit=5)
    team_activity = get_team_activity()
    
    # Custom CSS for Clinical EMR style
    st.markdown("""
        <style>
        /* Clinical EMR Header */
        .clinical-header {
            background: #1e3a5f;
            padding: 15px 30px;
            color: white;
            display: flex;
            justify-content: space-between;
            align-items: center;
            border-radius: 0;
            margin: -1rem -1rem 2rem -1rem;
        }
        
        .clinical-header h2 {
            margin: 0;
            font-size: 22px;
            font-weight: 600;
        }
        
        .clinical-header-right {
            display: flex;
            gap: 20px;
            align-items: center;
            font-size: 14px;
        }
        
        /* Metric cards */
        .metric-card-clinical {
            background: white;
            border: 1px solid #e0e0e0;
            border-radius: 4px;
            padding: 20px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }
        
        .metric-label-clinical {
            font-size: 13px;
            color: #666;
            font-weight: 500;
            margin-bottom: 8px;
        }
        
        .metric-value-clinical {
            font-size: 32px;
            font-weight: 700;
            color: #1e3a5f;
            margin-bottom: 4px;
        }
        
        .metric-delta-clinical {
            font-size: 12px;
            color: #4caf50;
        }
        
        .metric-delta-critical {
            color: #f44336;
        }
        
        /* Table styling */
        .dataframe {
            font-size: 13px !important;
        }
        
        /* Section headers */
        .section-header-clinical {
            font-size: 16px;
            font-weight: 600;
            color: #1e3a5f;
            margin: 20px 0 10px 0;
            padding-bottom: 8px;
            border-bottom: 2px solid #1e3a5f;
        }
        
        /* Action cards */
        .action-card-clinical {
            background: #f8f9fa;
            border: 1px solid #dee2e6;
            border-radius: 4px;
            padding: 16px;
            margin: 10px 0;
        }
        
        .action-card-clinical h4 {
            color: #1e3a5f;
            font-size: 15px;
            margin-bottom: 8px;
        }
        
        .action-card-clinical p {
            color: #666;
            font-size: 13px;
            margin: 4px 0;
        }
        
        /* Status indicators */
        .status-badge {
            display: inline-block;
            padding: 3px 8px;
            border-radius: 3px;
            font-size: 11px;
            font-weight: 600;
        }
        
        .status-stable {
            background: #d4edda;
            color: #155724;
        }
        
        .status-review {
            background: #fff3cd;
            color: #856404;
        }
        
        .status-urgent {
            background: #f8d7da;
            color: #721c24;
        }
        </style>
    """, unsafe_allow_html=True)
    
    # Header
    current_user = st.session_state.get('current_user', '이재영')
    user_info = st.session_state.get('user_info', {'role': '연구원', 'department': '바이오메디컬사이언스'})
    
    st.markdown(f"""
        <div class="clinical-header">
            <h2>🏥 ADDS Clinical Decision Support System</h2>
            <div class="clinical-header-right">
                <span>{current_user} ({user_info['role']})</span>
                <span>|</span>
                <span>Department: {user_info['department']}</span>
                <span>|</span>
                <span>🔔 3 alerts</span>
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    # Top Metrics Row - 4 columns
    st.markdown('<div class="section-header-clinical">Key Performance Indicators</div>', unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
            <div class="metric-card-clinical">
                <div class="metric-label-clinical">👥 Active Patients</div>
                <div class="metric-value-clinical">{metrics['active_patients']:,}</div>
                <div class="metric-delta-clinical">{metrics['active_patients_delta']}</div>
            </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
            <div class="metric-card-clinical">
                <div class="metric-label-clinical">📋 Pending Reviews</div>
                <div class="metric-value-clinical">{metrics['pending_reviews']}</div>
                <div class="metric-delta-critical">
                    <span class="status-badge status-urgent">{metrics['pending_priority']}</span>
                </div>
            </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
            <div class="metric-card-clinical">
                <div class="metric-label-clinical">🧠 CDSS Analyses Today</div>
                <div class="metric-value-clinical">{metrics['cdss_analyses']}</div>
                <div class="metric-delta-clinical">{metrics['cdss_accuracy']} accuracy</div>
            </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
            <div class="metric-card-clinical">
                <div class="metric-label-clinical">⚠️ Critical Alerts</div>
                <div class="metric-value-clinical">{metrics['critical_alerts']}</div>
                <div class="metric-delta-critical">
                    <span class="status-badge status-urgent">{metrics['alerts_status']}</span>
                </div>
            </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Main Content: Patient Table (60%) + System Performance (40%)
    col_left, col_right = st.columns([6, 4])
    
    with col_left:
        st.markdown('<div class="section-header-clinical">Recent Patient Cases</div>', unsafe_allow_html=True)
        
        # Style the dataframe for clinical view
        st.dataframe(
            patients_df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Status": st.column_config.TextColumn(
                    "Status",
                    help="Patient status"
                ),
                "Ki-67": st.column_config.TextColumn(
                    "Ki-67",
                    help="Ki-67 proliferation index"
                )
            }
        )
        
        # Action Buttons
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("📋 View All Patients", use_container_width=True):
                st.session_state['show_patient_list'] = not st.session_state.get('show_patient_list', False)
        
        with col2:
            if st.button("➕ New Patient", use_container_width=True):
                if st.session_state.get('current_page') != "👥 환자 관리":
                    st.session_state['current_page'] = "👥 환자 관리"
                    st.rerun()
        
        with col3:
            if st.button("📋 수술 보고서", use_container_width=True):
                st.session_state['show_surgery_report'] = not st.session_state.get('show_surgery_report', False)
        
        # Show patient list modal if toggled
        if st.session_state.get('show_patient_list', False):
            st.markdown("---")
            st.markdown("### 📋 전체 환자 목록")
            
            # Sample patient data
            all_patients_data = {
                'Patient ID': ['PT-001', 'PT-002', 'PT-003', 'PT-004', 'PT-005', 'PT-006', 'PT-007', 'PT-008', 'PT-009', 'PT-010'],
                'Name': ['김철수', '이영희', '박민수', '최지은', '정성훈', '강미래', '윤수진', '임동현', '한소영', '오준혁'],
                'Age': [58, 62, 71, 45, 68, 54, 59, 66, 51, 63],
                'Gender': ['M', 'F', 'M', 'F', 'M', 'F', 'F', 'M', 'F', 'M'],
                'Stage': ['Stage II', 'Stage III', 'Stage I', 'Stage III', 'Stage II', 'Stage IV', 'Stage I', 'Stage II', 'Stage III', 'Stage II'],
                'Last Visit': ['2026-01-28', '2026-01-27', '2026-01-29', '2026-01-25', '2026-01-30', '2026-01-26', '2026-01-29', '2026-01-28', '2026-01-24', '2026-01-30'],
                'Status': ['Active', 'Active', 'Follow-up', 'Active', 'Active', 'Critical', 'Follow-up', 'Active', 'Active', 'Active']
            }
            
            import pandas as pd
            df = pd.DataFrame(all_patients_data)
            
            # Add color coding for status
            def color_status(val):
                if val == 'Critical':
                    return 'background-color: #fee2e2; color: #991b1b'
                elif val == 'Active':
                    return 'background-color: #dbeafe; color: #1e40af'
                else:
                    return 'background-color: #d1fae5; color: #065f46'
            
            styled_df = df.style.applymap(color_status, subset=['Status'])
            
            st.dataframe(styled_df, use_container_width=True, height=400)
            
            # Action buttons
            col_a, col_b, col_c = st.columns(3)
            with col_a:
                if st.button("🔍 환자 검색", use_container_width=True):
                    if st.session_state.get('current_page') != "👥 환자 관리":
                        st.session_state['current_page'] = "👥 환자 관리"
                        st.rerun()
            with col_b:
                if st.button("📊 통계 보기", use_container_width=True):
                    st.info("환자 관리 페이지의 통계 대시보드로 이동합니다")
            with col_c:
                if st.button("✖️ 닫기", use_container_width=True):
                    st.session_state['show_patient_list'] = False
                    st.rerun()
        
        # Show surgery report modal if toggled
        if st.session_state.get('show_surgery_report', False):
            st.markdown("---")
            st.markdown("### 📋 수술 보고서 생성")
            
            report_col1, report_col2 = st.columns(2)
            
            with report_col1:
                st.selectbox("환자 선택", ['PT-001 - 김철수', 'PT-002 - 이영희', 'PT-003 - 박민수'])
                st.date_input("수술 날짜")
                st.selectbox("수술 유형", ['내시경 절제술', '복강경 수술', '개복 수술'])
            
            with report_col2:
                st.text_area("수술 소견", height=100)
                st.text_area("특이사항", height=100)
            
            col_r1, col_r2, col_r3 = st.columns(3)
            with col_r1:
                if st.button("📄 PDF 생성", use_container_width=True, type="primary"):
                    st.success("✅ 수술 보고서가 생성되었습니다!")
            with col_r2:
                if st.button("💾 저장", use_container_width=True):
                    st.success("✅ 수술 보고서가 저장되었습니다!")
            with col_r3:
                if st.button("✖️ 취소", use_container_width=True):
                    st.session_state['show_surgery_report'] = False
                    st.rerun()
    
    with col_right:
        st.markdown('<div class="section-header-clinical">System Performance</div>', unsafe_allow_html=True)
        
        # Accuracy gauge
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=metrics['accuracy_percent'],
            title={'text': "Model Accuracy", 'font': {'size': 16}},
            gauge={
                'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "#1e3a5f"},
                'bar': {'color': "#1e3a5f"},
                'bgcolor': "white",
                'borderwidth': 2,
                'bordercolor': "#e0e0e0",
                'steps': [
                    {'range': [0, 50], 'color': '#ffebee'},
                    {'range': [50, 80], 'color': '#fff3e0'},
                    {'range': [80, 100], 'color': '#e8f5e9'}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 90
                }
            }
        ))
        
        fig.update_layout(
            height=250,
            margin=dict(l=20, r=20, t=50, b=20),
            paper_bgcolor='white',
            font={'color': "#1e3a5f", 'family': "Arial"}
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # System stats
        st.markdown(f"""
            <div class="action-card-clinical">
                <p><strong>GPU Status:</strong> ✅ Active, {metrics['gpu_temperature']}</p>
                <p><strong>Processing Queue:</strong> {metrics['processing_queue']} pending</p>
                <p><strong>Model Version:</strong> {metrics['model_version']}</p>
                <p><strong>Last Update:</strong> {metrics['last_update']}</p>
            </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Bottom Section - 3 Action Cards
    st.markdown('<div class="section-header-clinical">Quick Actions</div>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
            <div class="action-card-clinical">
                <h4>🔬 Quick Analysis</h4>
                <p>Start new CDSS analysis</p>
            </div>
        """, unsafe_allow_html=True)
        
        uploaded_file = st.file_uploader(
            "Upload patient data",
            type=['dcm', 'nii', 'tiff', 'png'],
            key='clinical_upload',
            label_visibility='collapsed'
        )
        
        if st.button("Start CDSS Analysis", type="primary", use_container_width=True):
            if uploaded_file:
                st.success("✓ Analysis started")
            else:
                st.warning("Please upload a file first")
    
    with col2:
        st.markdown(f"""
            <div class="action-card-clinical">
                <h4>📈 Research Insights</h4>
                <p>Recent publications: <strong>127</strong></p>
                <p>Drug discoveries: <strong>15 new</strong></p>
                <p>Success rate: <strong>{metrics['success_rate']}%</strong></p>
            </div>
        """, unsafe_allow_html=True)
        
        if st.button("View Research Database", use_container_width=True):
            st.info("Opening research database...")
    
    with col3:
        st.markdown("""
            <div class="action-card-clinical">
                <h4>👥 Team Activity</h4>
            </div>
        """, unsafe_allow_html=True)
        
        for member in team_activity:
            st.markdown(f"""
                <p style="font-size: 13px; margin: 4px 0;">
                    <strong>{member['name']}</strong>: {member['analyses_today']} analyses today
                </p>
            """, unsafe_allow_html=True)
        
        st.markdown(f"""
            <p style="font-size: 13px; color: #4caf50; margin-top: 8px;">
                ● {len([m for m in team_activity if m['status'] == 'online'])} clinicians online
            </p>
        """, unsafe_allow_html=True)
