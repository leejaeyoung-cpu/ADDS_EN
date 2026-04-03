"""
Show Synergy Calculation Page
"""

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import sys
from datetime import datetime

# Add src to path  
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from utils.synergy_calculator import SynergyCalculator
from ui.app_core import get_synergy_calculator


def show_synergy_calculation():
    """Synergy calculation page"""
    st.header("💊 약물 시너지 스코어 계산")
    
    st.markdown("### 실험 데이터 입력")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Drug A")
        drug_a_name = st.text_input("Drug A 이름", "")
        dose_a = st.number_input("Dose A (μM)", value=0.0, min_value=0.0)
        effect_a = st.slider("Effect A (단독)", 0.0, 1.0, 0.0, 0.01)
        ic50_a = st.number_input("IC50 A (μM)", value=0.0, min_value=0.0)
    
    with col2:
        st.markdown("#### Drug B")
        drug_b_name = st.text_input("Drug B 이름", "")
        dose_b = st.number_input("Dose B (μM)", value=0.0, min_value=0.0)
        effect_b = st.slider("Effect B (단독)", 0.0, 1.0, 0.0, 0.01)
        ic50_b = st.number_input("IC50 B (μM)", value=0.0, min_value=0.0)
    
    effect_comb = st.slider("Combined Effect (조합)", 0.0, 1.0, 0.0, 0.01)
    
    if st.button("시너지 계산", type="primary"):
        # 캐시된 calculator 사용
        calculator = get_synergy_calculator()
        
        synergy_scores = calculator.calculate_all_synergies(
            dose_a=dose_a,
            dose_b=dose_b,
            effect_a=effect_a,
            effect_b=effect_b,
            effect_combination=effect_comb,
            ic50_a=ic50_a,
            ic50_b=ic50_b
        )
        
        st.markdown("### 시너지 스코어 결과")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            bliss = synergy_scores['bliss']
            st.metric("Bliss", f"{bliss:.3f}", 
                     delta="Synergistic" if bliss > 0 else "Antagonistic")
        
        with col2:
            hsa = synergy_scores['hsa']
            st.metric("HSA", f"{hsa:.3f}",
                     delta="Synergistic" if hsa > 0 else "Antagonistic")
        
        with col3:
            if 'loewe' in synergy_scores:
                loewe = synergy_scores['loewe']
                st.metric("Loewe", f"{loewe:.3f}",
                         delta="Synergistic" if loewe > 0 else "Antagonistic")
        
        with col4:
            if 'mean_synergy' in synergy_scores:
                mean = synergy_scores['mean_synergy']
                st.metric("Mean", f"{mean:.3f}")
        
        # Visualization
        methods = [k for k in synergy_scores.keys() if k not in ['is_synergistic', 'mean_synergy']]
        values = [synergy_scores[k] for k in methods]
        
        fig = go.Figure(data=[
            go.Bar(x=methods, y=values, marker_color=['green' if v > 0 else 'red' for v in values])
        ])
        fig.update_layout(title="시너지 스코어 비교", yaxis_title="Score")

    # ================================================================
    # AI-based Synergy Prediction (DeepSynergy v3 + XGBoost)
    # ================================================================
    st.markdown("---")
    st.markdown("### 🤖 AI 시너지 예측 (DeepSynergy v3)")
    st.caption("DrugComb 927K 데이터로 학습된 딥러닝 모델 예측 (약물 이름만 입력)")

    try:
        sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
        from backend.services.ml_synergy_service import MLSynergyService

        @st.cache_resource
        def _get_ml_service():
            return MLSynergyService()

        svc = _get_ml_service()
        drugs = sorted(svc.drug_smiles.keys())

        if not drugs:
            st.warning("약물 SMILES 데이터가 없습니다.")
        else:
            col_a, col_b, col_cl = st.columns(3)
            with col_a:
                ai_drug_a = st.selectbox("Drug A", drugs, key="ai_drug_a")
            with col_b:
                ai_drug_b = st.selectbox("Drug B", drugs, index=min(1, len(drugs)-1), key="ai_drug_b")
            with col_cl:
                cell_lines = ["HCT116", "A549", "MCF7", "PC3", "A375", "SKOV3", "K562"]
                ai_cl = st.selectbox("Cell Line", cell_lines, key="ai_cl")

            if st.button("🔬 AI 시너지 예측", type="primary", key="ai_predict"):
                with st.spinner("예측 중..."):
                    # DeepSynergy v3
                    ds_result = svc.predict_deep_synergy(ai_drug_a, ai_drug_b, ai_cl)

                    col1, col2 = st.columns(2)

                    with col1:
                        st.markdown("#### DeepSynergy v3")
                        if ds_result.get("status") == "success":
                            score = ds_result["synergy_score"]
                            cls = ds_result["classification"]
                            color = "🟢" if cls == "synergistic" else "🔴" if cls == "antagonistic" else "🟡"
                            st.metric(
                                f"{color} Loewe Synergy Score",
                                f"{score:.2f}",
                                delta=cls.upper()
                            )
                            st.caption("학습 데이터: DrugComb 927K (4,246 drugs)")
                        else:
                            st.error(ds_result.get("message", "예측 실패"))

                    with col2:
                        st.markdown("#### XGBoost 모델 비교")
                        xgb_result = svc.compare_models(ai_drug_a, ai_drug_b, ai_cl)
                        if xgb_result.get("status") == "success":
                            for model_name, r in xgb_result["per_model"].items():
                                if model_name == "deep_synergy_v3":
                                    continue
                                cls = r["classification"]
                                icon = "🟢" if cls == "synergistic" else "🔴" if cls == "antagonistic" else "🟡"
                                st.write(f"{icon} **{model_name}**: {r['score']:.2f} ({cls})")
                        else:
                            st.info("XGBoost 모델 없음")

                    # Model info
                    with st.expander("모델 정보"):
                        info = svc.get_model_info()
                        st.json(info)

    except Exception as e:
        st.info(f"AI 시너지 예측 모듈 로드 실패: {e}")
