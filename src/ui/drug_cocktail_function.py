def show_drug_cocktail():
    """Drug Cocktail Design and AI Recommendation Page"""
    st.header("💊 항암제 칵테일 설계 및 AI 추천")
    st.markdown("항암제 조합 설계, 시너지 분석 및 인공지능 기반 맞춤형 추천")
    
    # Create tabs
    tab1, tab2, tab3 = st.tabs(["💊 약물 조합 설계", "🔬 시너지 분석", "🤖 AI 추천"])
    
    # ===== TAB 1: Drug Combination Design =====
    with tab1:
        st.markdown("### 약물 데이터베이스")
        st.info(f"📋 **{len(DRUG_DATABASE)}개의 항암제 등록**")
        
        # Drug selection
        selected_drugs = st.multiselect(
            "조합할 약물 선택 (2-3개 권장)",
            options=list(DRUG_DATABASE.keys()),
            default=[],
            help="시너지 효과를 기대할 약물을 2-3개 선택하세요"
        )
        
        if selected_drugs:
            st.success(f"✅ {len(selected_drugs)}개 약물 선택됨")
            
            # Display drug information
            st.markdown("#### 선택한 약물 정보")
            
            cols = st.columns(min(len(selected_drugs), 3))
            for idx, drug in enumerate(selected_drugs):
                info = get_drug_info(drug)
                with cols[idx % 3]:
                    with st.expander(f"📋 {drug}", expanded=True):
                        st.markdown(f"""
                        **분류**: {info.get('class', 'N/A')}  
                        **기전**: {info.get('mechanism', 'N/A')[:50]}...  
                        **표적**: {', '.join(info.get('targets', []))}  
                        **적응증**: {', '.join(info.get('cancer_types', [])[:2])}  
                        **용량**: {info.get('typical_dose', 'N/A')}
                        """)
        else:
            st.warning("⚠️ 약물을 선택하세요")
    
    # ===== TAB 2: Synergy Analysis =====
    with tab2:
        st.markdown("### 시너지 스코어 계산")
        
        if len(selected_drugs) >= 2:
            st.info(f"분석 대상: {selected_drugs[0]} + {selected_drugs[1]}")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown(f"#### {selected_drugs[0]}")
                effect_a = st.slider(
                    f"{selected_drugs[0]} 단독 효과 (%)",
                    0, 100, 50,
                    help="단독 투여 시 세포 사멸률"
                )
            
            with col2:
                st.markdown(f"#### {selected_drugs[1]}")
                effect_b = st.slider(
                    f"{selected_drugs[1]} 단독 효과 (%)",
                    0, 100, 50,
                    help="단독 투여 시 세포 사멸률"
                )
            
            effect_comb = st.slider(
                "병용 투여 효과 (%)",
                0, 100, 75,
                help="두 약물을 함께 투여했을 때의 세포 사멸률"
            )
            
            if st.button("🔬 시너지 계산", type="primary"):
                # Calculate synergy
                calculator = get_synergy_calculator()
                synergy_scores = calculator.calculate_all_synergies(
                    dose_a=1.0,
                    dose_b=1.0,
                    effect_a=effect_a / 100.0,
                    effect_b=effect_b / 100.0,
                    effect_combination=effect_comb / 100.0,
                    ic50_a=1.0,
                    ic50_b=1.0
                )
                
                st.markdown("#### 📊 시너지 스코어 결과")
                
                metric_cols = st.columns(4)
                
                with metric_cols[0]:
                    bliss = synergy_scores.get('bliss', 0)
                    st.metric(
                        "Bliss Independence",
                        f"{bliss:.3f}",
                        delta="Synergy" if bliss > 0 else "Antagonism"
                    )
                
                with metric_cols[1]:
                    hsa = synergy_scores.get('hsa', 0)
                    st.metric(
                        "HSA",
                        f"{hsa:.3f}",
                        delta="Synergy" if hsa > 0 else "Antagonism"
                    )
                
                with metric_cols[2]:
                    if 'loewe' in synergy_scores:
                        loewe = synergy_scores['loewe']
                        st.metric("Loewe Additivity", f"{loewe:.3f}")
                
                with metric_cols[3]:
                    if 'mean_synergy' in synergy_scores:
                        mean = synergy_scores['mean_synergy']
                        st.metric("평균 시너지", f"{mean:.3f}")
                
                # Visualization
                methods = [k for k in synergy_scores.keys() 
                          if k not in ['is_synergistic', 'mean_synergy']]
                values = [synergy_scores[k] for k in methods]
                
                fig = go.Figure(data=[
                    go.Bar(
                        x=methods,
                        y=values,
                        marker_color=['green' if v > 0 else 'red' for v in values]
                    )
                ])
                fig.update_layout(
                    title="시너지 스코어 비교",
                    yaxis_title="Score",
                    height=300
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Interpretation
                if synergy_scores.get('is_synergistic', False):
                    st.success("✅ **시너지 효과 확인**: 이 조합은 단독 투여보다 우수한 효과를 보입니다.")
                else:
                    st.warning("⚠️ **시너지 효과 미미**: 조합 효과가 예상보다 낮습니다.")
        
        else:
            st.warning("⚠️ 시너지 분석을 위해 2개 이상의 약물을 선택하세요.")
    
    # ===== TAB 3: AI Recommendation =====
    with tab3:
        st.markdown("### 🤖 AI 기반 약물 추천")
        st.caption("GPT-4를 활용한 맞춤형 항암제 칵테일 추천")
        
        # Patient information input
        st.markdown("#### 환자 정보 입력")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            cancer_type = st.selectbox(
                "암 종류",
                ["Colorectal", "Breast", "Lung", "Pancreatic", "Prostate",
                 "Gastric", "Ovarian", "Liver", "Bladder", "Renal", "Esophageal", "Head_Neck"],
                help="환자의 암 종류를 선택하세요"
            )
            age = st.number_input("나이", 20, 100, 60)
        
        with col2:
            sex = st.selectbox("성별", ["Male", "Female"])
            ecog = st.selectbox(
                "ECOG Performance Status",
                [0, 1, 2, 3, 4],
                help="0:정상, 4:완전침상"
            )
        
        with col3:
            stage = st.selectbox("병기", ["I", "II", "III", "IV"])
            previous_tx = st.text_input("이전 치료", "None")
        
        if st.button("🤖 AI 추천 받기", type="primary"):
            with st.spinner("🧠 AI가 최적의 약물 조합을 분석하고 있습니다..."):
                recommender = CocktailRecommender()
                
                patient_info = {
                    "age": age,
                    "sex": sex,
                    "ecog": ecog,
                    "stage": stage,
                    "previous_treatment": previous_tx
                }
                
                recommendation = recommender.recommend_cocktail(
                    cancer_type=cancer_type,
                    patient_info=patient_info,
                    available_drugs=list(DRUG_DATABASE.keys()),
                    num_recommendations=3
                )
                
                st.markdown("---")
                st.markdown("### 🎯 AI 추천 결과")
                
                if recommendation.get("success"):
                    st.markdown(recommendation["recommendations"])
                    st.caption(f"🤖 Model: {recommendation.get('model', 'gpt-4o-mini')}")
                else:
                    if "error" in recommendation:
                        st.warning(f"API 오류: {recommendation['error']}")
                    
                    if "recommendations" in recommendation:
                        st.markdown("### 📋 기본 추천 (규칙 기반)")
                        st.markdown(recommendation["recommendations"])


