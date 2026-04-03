"""
AI Prediction Page
Drug combination prediction using ADDS pathway-based recommender,
pharmacokinetic modeling, and synergy analysis
"""

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import sys
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px

# Add src to path  
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from utils.adds_recommender import ADDSRecommender
from utils.drug_database import DRUG_DATABASE, get_drug_info


def show_ai_prediction():
    """AI-based drug combination prediction and optimization"""
    st.header("🤖 AI 약물 조합 예측")
    st.caption("Pathway 기반 AI 추천 + 약동학 시뮬레이션 + 시너지 분석")
    
    # Initialize recommender
    if 'recommender' not in st.session_state:
        st.session_state.recommender = ADDSRecommender()
    recommender = st.session_state.recommender
    
    # --- Patient Clinical Input ---
    with st.expander("📋 환자 임상 정보 입력", expanded=True):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            cancer_type = st.selectbox("암종", [
                "colorectal", "colon", "rectal"
            ], index=0)
            tnm_stage = st.selectbox("TNM 병기", [
                "I", "II", "IIA", "IIB", "III", "IIIA", "IIIB", "IIIC", "IV", "IVA", "IVB"
            ], index=4)
            tumor_location = st.selectbox("종양 위치", [
                "Right colon", "Left colon", "Sigmoid", "Rectum", "Cecum", "Transverse"
            ])
        
        with col2:
            msi_status = st.selectbox("MSI 상태", ["MSS", "MSI-H", "MSI-L"])
            kras_mutation = st.selectbox("KRAS 변이", ["Wild-type", "Mutant (G12D)", "Mutant (G12V)", "Mutant (G13D)", "Unknown"])
            braf_status = st.selectbox("BRAF", ["Wild-type", "V600E Mutant", "Unknown"])
        
        with col3:
            age = st.number_input("나이", 20, 100, 60)
            ki67_index = st.slider("Ki-67 (%)", 0, 100, 30)
            tumor_volume = st.number_input("종양 부피 (cm³)", 0.0, 500.0, 25.0, step=5.0)
    
    # --- Run Prediction ---
    num_drugs = st.slider("추천 약물 수", 2, 5, 3)
    
    if st.button("🔬 AI 약물 조합 예측 실행", type="primary", use_container_width=True):
        with st.spinner("Pathway 분석 및 약물 조합 최적화 중..."):
            patient_data = {
                "tnm_stage": tnm_stage,
                "msi_status": msi_status,
                "kras_mutation": kras_mutation,
                "braf_status": braf_status,
                "tumor_location": tumor_location,
                "ki67_index": ki67_index,
                "tumor_volume_cm3": tumor_volume,
                "age": age
            }
            
            # Get ADDS recommendation
            result = recommender.recommend_combination(
                cancer_type=cancer_type,
                num_drugs=num_drugs,
                patient_data=patient_data
            )
            
            # Normalize result keys to UI format
            raw_drugs = result.get('recommended_drugs', [])
            result['drugs'] = [{
                'name': d.get('drug', d.get('name', 'Unknown')),
                'class': d.get('target', ''),
                'pathways': [d.get('pathway_targeted', d.get('pathway_name', ''))],
                'mechanism': d.get('mechanism', ''),
                'sensitivity': d.get('efficacy', 0.7)
            } for d in raw_drugs]
            result['pathways'] = result.get('pathways_covered', [])
            # Confidence from pathway coverage + synergy (no artificial floor)
            n_pathways = len(result.get('pathways', []))
            pathway_cov = min(n_pathways / 6.0, 1.0)  # 6 key CRC pathways
            synergy_score = result.get('synergy_score', 0)
            synergy_conf = max(0, min(synergy_score, 1.0))
            drug_evidence = min(len(raw_drugs) / 5.0, 1.0)
            result['confidence'] = min(0.95, (
                0.20 * pathway_cov     # Pathway coverage contributes 20%
                + 0.40 * synergy_conf  # Synergy evidence contributes 40%
                + 0.25 * drug_evidence  # Drug count contributes 25%
                + 0.15 * 0.70          # Base literature evidence 15%
            ))
            result['confidence_method'] = 'pathway_coverage_synergy_composite'
            
            st.session_state.prediction_result = result
            st.session_state.patient_data = patient_data
    
    # --- Display Results ---
    if 'prediction_result' in st.session_state:
        result = st.session_state.prediction_result
        patient_data = st.session_state.patient_data
        
        st.markdown("---")
        st.subheader("📊 추천 결과")
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            score = result.get('synergy_score', 0)
            st.metric("시너지 스코어", f"{score:.3f}",
                      delta="Synergistic" if score > 0 else "Additive")
        with col2:
            n_drugs = len(result.get('drugs', []))
            st.metric("추천 약물 수", n_drugs)
        with col3:
            n_pathways = len(result.get('pathways', []))
            st.metric("표적 경로", n_pathways)
        with col4:
            confidence = result.get('confidence', 0.50)
            st.metric("신뢰도", f"{confidence:.0%}",
                      help="Pathway coverage (20%) + Synergy evidence (40%) + Drug count (25%) + Literature base (15%)")
        
        # Drug details table
        drugs = result.get('drugs', [])
        if drugs:
            st.markdown("### 💊 추천 약물 상세")
            
            drug_rows = []
            for d in drugs:
                info = get_drug_info(d.get('name', ''))
                drug_rows.append({
                    "약물명": d.get('name', 'Unknown'),
                    "분류": d.get('class', info.get('class', 'N/A') if info else 'N/A'),
                    "표적 경로": ', '.join(d.get('pathways', [])),
                    "작용 메커니즘": d.get('mechanism', info.get('mechanism', 'N/A') if info else 'N/A'),
                    "민감도": f"{d.get('sensitivity', 0.7):.0%}"
                })
            
            st.dataframe(
                pd.DataFrame(drug_rows),
                use_container_width=True,
                hide_index=True
            )
        
        # Rationale
        rationale = result.get('rationale', '')
        if rationale:
            st.markdown("### 🧬 추천 근거")
            st.info(rationale)
        
        # --- Pathway Network Visualization ---
        pathways = result.get('pathways', [])
        if pathways and drugs:
            st.markdown("### 🔗 Pathway-Drug 네트워크")
            _render_pathway_network(drugs, pathways, result)
        
        # --- PK Simulation (if available) ---
        st.markdown("### 💉 약동학 시뮬레이션")
        _render_pk_simulation(drugs, patient_data)
        
        # --- Synergy Heatmap ---
        if len(drugs) >= 2:
            st.markdown("### 🧪 약물 간 시너지 매트릭스")
            _render_synergy_matrix(drugs)
        
        # --- Custom Combination Analysis ---
        st.markdown("---")
        with st.expander("🔄 사용자 정의 약물 조합 분석"):
            _render_custom_analysis(recommender, cancer_type)


def _render_pathway_network(drugs, pathways, result):
    """Render pathway-drug interaction network using Plotly"""
    # Build simple node layout
    drug_names = [d.get('name', f'Drug {i}') for i, d in enumerate(drugs)]
    pathway_names = pathways[:8]  # Limit for readability
    
    # Create a force-directed-like layout
    n_drugs = len(drug_names)
    n_paths = len(pathway_names)
    
    # Position drugs on the left, pathways on the right
    drug_x = [0.1] * n_drugs
    drug_y = [(i + 1) / (n_drugs + 1) for i in range(n_drugs)]
    
    path_x = [0.9] * n_paths
    path_y = [(i + 1) / (n_paths + 1) for i in range(n_paths)]
    
    fig = go.Figure()
    
    # Draw edges (drug → pathway connections)
    for i, d in enumerate(drugs):
        drug_pathways = d.get('pathways', [])
        for j, p in enumerate(pathway_names):
            if p in drug_pathways:
                fig.add_trace(go.Scatter(
                    x=[drug_x[i], path_x[j]], y=[drug_y[i], path_y[j]],
                    mode='lines', line=dict(color='rgba(100,150,255,0.3)', width=2),
                    showlegend=False, hoverinfo='skip'
                ))
    
    # Draw drug nodes
    fig.add_trace(go.Scatter(
        x=drug_x, y=drug_y,
        mode='markers+text', text=drug_names,
        textposition='middle left', textfont=dict(size=12),
        marker=dict(size=20, color='#FF6B6B', symbol='circle'),
        name='약물'
    ))
    
    # Draw pathway nodes
    fig.add_trace(go.Scatter(
        x=path_x, y=path_y,
        mode='markers+text', text=pathway_names,
        textposition='middle right', textfont=dict(size=11),
        marker=dict(size=16, color='#4ECDC4', symbol='diamond'),
        name='경로'
    ))
    
    fig.update_layout(
        title="Drug-Pathway 상호작용 네트워크",
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-0.2, 1.2]),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-0.1, 1.1]),
        height=400, showlegend=True,
        plot_bgcolor='rgba(0,0,0,0)'
    )
    st.plotly_chart(fig, use_container_width=True)


def _render_pk_simulation(drugs, patient_data):
    """Run PK simulation for recommended drugs"""
    try:
        sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
        from backend.services.pharmacokinetics_service import PharmacokineticEngine, DRUG_PK_PARAMS
        
        engine = PharmacokineticEngine()
        
        drug_names = [d.get('name', '').lower() for d in drugs]
        available_drugs = [n for n in drug_names if n in DRUG_PK_PARAMS]
        
        if not available_drugs:
            st.info("추천된 약물 중 PK 파라미터를 보유한 약물이 없습니다.")
            return
        
        # Run PK for each available drug
        pk_results = {}
        for drug_name in available_drugs[:3]:  # Limit to 3
            try:
                pk = engine.analyze(
                    drug_name=drug_name,
                    tumor_volume_cm3=patient_data.get('tumor_volume_cm3', 25),
                    ki67_index=patient_data.get('ki67_index', 30),
                    age=patient_data.get('age', 60)
                )
                pk_results[drug_name] = pk
            except Exception as e:
                st.warning(f"{drug_name} PK 시뮬레이션 실패: {e}")
        
        if not pk_results:
            return
        
        # PK metrics table
        pk_rows = []
        for name, pk in pk_results.items():
            pk_rows.append({
                "약물": name.title(),
                "CL (L/h)": f"{pk.clearance:.2f}",
                "t½β (h)": f"{pk.half_life_beta:.1f}",
                "Cmax (µg/mL)": f"{pk.cmax:.3f}",
                "AUC (µg·h/mL)": f"{pk.auc:.1f}",
                "최적 용량 (mg/m²)": f"{pk.optimal_dose:.0f}",
                "투약 간격 (h)": pk.dosing_interval,
                "효능 예측": f"{pk.predicted_efficacy:.0%}",
                "독성": pk.toxicity_risk
            })
        
        st.dataframe(pd.DataFrame(pk_rows), use_container_width=True, hide_index=True)
        
        # Concentration-time profiles
        fig = go.Figure()
        colors = px.colors.qualitative.Set2
        
        for idx, (name, pk) in enumerate(pk_results.items()):
            if pk.concentration_profile:
                times = pk.concentration_profile['time_hours']
                concs = pk.concentration_profile['concentration_ug_ml']
                fig.add_trace(go.Scatter(
                    x=times, y=concs, mode='lines',
                    name=name.title(),
                    line=dict(color=colors[idx % len(colors)], width=2)
                ))
                # Therapeutic window
                therapeutic_min = pk.concentration_profile.get('therapeutic_min', 0)
                therapeutic_max = pk.concentration_profile.get('therapeutic_max', 0)
        
        # Add therapeutic window band (use last drug's values)
        if pk_results:
            last_pk = list(pk_results.values())[-1]
            if last_pk.concentration_profile:
                fig.add_hline(y=last_pk.therapeutic_window[0], 
                             line_dash="dash", line_color="green",
                             annotation_text="Cmin")
                fig.add_hline(y=last_pk.therapeutic_window[1],
                             line_dash="dash", line_color="red", 
                             annotation_text="Cmax limit")
        
        fig.update_layout(
            title="약물 농도-시간 프로파일",
            xaxis_title="시간 (hours)",
            yaxis_title="농도 (µg/mL)",
            height=350
        )
        st.plotly_chart(fig, use_container_width=True)
        
    except ImportError:
        st.info("약동학 모듈을 불러올 수 없습니다. Backend 서비스를 확인하세요.")
    except Exception as e:
        st.warning(f"PK 시뮬레이션 오류: {e}")


def _render_synergy_matrix(drugs):
    """Render drug-drug synergy potential matrix"""
    drug_names = [d.get('name', f'Drug {i}') for i, d in enumerate(drugs)]
    n = len(drug_names)
    
    # Calculate synergy based on pathway overlap
    matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i == j:
                matrix[i][j] = 1.0
            else:
                # Synergy estimate from pathway overlap
                paths_i = set(drugs[i].get('pathways', []))
                paths_j = set(drugs[j].get('pathways', []))
                
                if paths_i and paths_j:
                    # Complementary pathways → higher synergy potential
                    overlap = len(paths_i & paths_j)
                    total = len(paths_i | paths_j)
                    complementarity = 1.0 - (overlap / total if total > 0 else 0)
                    matrix[i][j] = complementarity * 0.8 + 0.1
                else:
                    matrix[i][j] = 0.5
    
    fig = go.Figure(data=go.Heatmap(
        z=matrix,
        x=drug_names,
        y=drug_names,
        colorscale='RdYlGn',
        zmin=0, zmax=1,
        text=np.round(matrix, 2),
        texttemplate="%{text}",
        textfont={"size": 12}
    ))
    fig.update_layout(
        title="약물 간 시너지 잠재력 (경로 상보성 기반)",
        height=350
    )
    st.plotly_chart(fig, use_container_width=True)


def _render_custom_analysis(recommender, cancer_type):
    """Custom drug combination analysis"""
    available = list(DRUG_DATABASE.keys()) if DRUG_DATABASE else []
    
    if available:
        selected = st.multiselect(
            "분석할 약물 선택 (2-5개)", available,
            default=available[:2] if len(available) >= 2 else available
        )
    else:
        selected_text = st.text_input("약물명 입력 (쉬표로 구분)", "5-Fluorouracil, Oxaliplatin")
        selected = [s.strip() for s in selected_text.split(",") if s.strip()]
    
    if len(selected) >= 2 and st.button("사용자 조합 분석 실행"):
        with st.spinner("분석 중..."):
            analysis = recommender.analyze_custom_combination(
                drug_names=selected,
                cancer_type=cancer_type
            )
            
            st.json(analysis)
