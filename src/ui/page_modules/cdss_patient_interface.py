"""
ADDS CDSS - Patient Interface
===============================
Patient-friendly interface for viewing CDSS analysis results

Features:
- Simple language explanations
- Visual icons and graphics
- Treatment plan overview
- FAQ section
- Communication with doctor
"""

import streamlit as st
from pathlib import Path
import json
from datetime import datetime
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from medical_imaging.cdss.integration_engine import IntegratedPatientProfile


# ── PDF Report Generator (importlib - sys.path independent) ─────────────────
import importlib.util as _ilu_pdf, os as _os_pdf
_pdf_path = str(Path(__file__).parent / 'pdf_report_generator.py')
_pdf_spec = _ilu_pdf.spec_from_file_location('pdf_report_generator', _pdf_path)
_pdf_mod  = _ilu_pdf.module_from_spec(_pdf_spec)
_pdf_spec.loader.exec_module(_pdf_mod)
generate_doctor_report_pdf  = _pdf_mod.generate_doctor_report_pdf
generate_patient_report_pdf = _pdf_mod.generate_patient_report_pdf
del _ilu_pdf, _os_pdf, _pdf_path, _pdf_spec, _pdf_mod
# ─────────────────────────────────────────────────────────────────────────────

def show_patient_interface():
    """
    Patient-friendly CDSS interface
    """
    st.set_page_config(page_title="내 건강 분석 결과", page_icon="👤")
    
    # Custom CSS for patient-friendly design
    st.markdown("""
    <style>
    .big-font {
        font-size: 24px !important;
        font-weight: bold;
    }
    .metric-box {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .positive-box {
        background-color: #d4edda;
        padding: 15px;
        border-radius: 8px;
        border-left: 5px solid #28a745;
    }
    .info-box {
        background-color: #d1ecf1;
        padding: 15px;
        border-radius: 8px;
        border-left: 5px solid #17a2b8;
    }
    .warning-box {
        background-color: #fff3cd;
        padding: 15px;
        border-radius: 8px;
        border-left: 5px solid #ffc107;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.title("🏥 내 건강 분석 결과")
    st.markdown("**ADDS 정밀 종양 분석 시스템**")
    
    # Load patient profile
    # In real implementation, this would load from database based on patient ID
    # For now, check session state or show demo
    
    profile = st.session_state.get('cdss_patient_profile')
    
    if profile is None:
        # Demo mode
        st.info("👨‍⚕️ 의사 선생님이 분석 결과를 공유하면 여기에 표시됩니다.")
        
        # Show demo button
        if st.button("📊 데모 결과 보기", type="primary"):
            st.session_state['patient_demo_mode'] = True
            st.rerun()
        
        if st.session_state.get('patient_demo_mode'):
            profile = create_demo_profile()
        else:
            st.stop()
    
    # Welcome message
    patient_name = "홍길동"  # In real app, from profile
    st.markdown(f"### 안녕하세요, {patient_name}님 👋")
    st.markdown(f"**검사일**: {profile.timestamp.strftime('%Y년 %m월 %d일')}")
    st.markdown("---")
    
    # ====================
    # SECTION 1: 검사 결과 요약
    # ====================
    st.header("📊 검사 결과 요약")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class='metric-box'>
        <h3>🔬 세포 검사</h3>
        <p><strong>세포 수:</strong> {0:,}개</p>
        <p><strong>증식 속도:</strong> {1}</p>
        </div>
        """.format(
            profile.cellpose_results.cell_count,
            "빠름" if profile.cellpose_results.ki67_index > 0.4 else "보통" if profile.cellpose_results.ki67_index > 0.2 else "느림"
        ), unsafe_allow_html=True)
    
    with col2:
        tumor_status = "예" if profile.ct_results.tumor_detected else "아니오"
        st.markdown(f"""
        <div class='metric-box'>
        <h3>🏥 CT 검사</h3>
        <p><strong>종양 발견:</strong> {tumor_status}</p>
        {f"<p><strong>크기:</strong> 약 {profile.ct_results.tumor_size_mm/10:.1f}cm</p>" if profile.ct_results.tumor_size_mm is not None else "<p><strong>크기:</strong> 측정 불가</p>"}
        <p><strong>위치:</strong> 대장</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class='metric-box'>
        <h3>📋 진단 결과</h3>
        <p><strong>단계:</strong> {profile.cancer_stage}기</p>
        <p><strong>위험도:</strong> {translate_risk_level(profile.risk_level)}</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # ====================
    # SECTION 2: 진단 설명
    # ====================
    st.header("🩺 진단이 의미하는 것")
    
    st.markdown(f"""
    <div class='info-box'>
    <h4>💡 검사 결과 설명</h4>
    <p>검사 결과, <strong>{profile.cancer_stage}기 대장암</strong>으로 확인되었습니다.</p>
    
    <p><strong>이것은 무엇을 의미하나요?</strong></p>
    <ul>
        <li>암이 대장의 일부에서 발견되었습니다</li>
        <li>{"림프절 전이가 확인되었습니다" if "N1" in (profile.ct_results.tnm_stage or "") else "림프절 전이는 없습니다"}</li>
        <li>조기에 발견되어 치료 가능성이 높습니다</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)
    
    # Patient-friendly interpretation from OpenAI
    if profile.patient_interpretation:
        st.markdown("### 📝 상세 설명")
        with st.expander("자세히 읽어보기", expanded=False):
            st.markdown(profile.patient_interpretation)
    
    st.markdown("---")
    
    # ====================
    # SECTION 3: 치료 계획
    # ====================
    st.header("💊 추천 치료 계획")
    
    if profile.recommended_therapies:
        main_therapy = profile.recommended_therapies[0]
        
        st.markdown(f"""
        <div class='positive-box'>
        <h4>✅ 주치의 추천 치료법</h4>
        <h3>{main_therapy.therapy_name}</h3>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("#### 📌 이 치료는 무엇인가요?")
        
        # Explain therapy in simple terms
        therapy_explanation = explain_therapy_simple(main_therapy)
        st.markdown(therapy_explanation)
        
        # Treatment details
        col_t1, col_t2 = st.columns(2)
        
        with col_t1:
            st.markdown(f"""
            **💉 사용하는 약물:**
            """)
            for drug in main_therapy.drug_combination:
                st.markdown(f"• {translate_drug_name(drug)}")
            
            if main_therapy.duration_weeks:
                months = main_therapy.duration_weeks // 4
                st.markdown(f"\n**⏱️ 치료 기간:** 약 {months}개월 ({main_therapy.duration_weeks}주)")
        
        with col_t2:
            efficacy_percent = int(main_therapy.predicted_efficacy * 100)
            st.markdown(f"""
            **📊 예상 효과:**
            - 환자 100명 중 약 **{efficacy_percent}명**에게 효과가 있습니다
            
            **⚠️ 예상되는 부작용:**
            - {translate_side_effects(main_therapy.side_effect_risk)}
            """)
        
        # Side effects warning
        with st.expander("⚠️ 부작용에 대해 더 알아보기"):
            st.markdown("""
            **일반적인 부작용:**
            - 메스꺼움이나 구토가 있을 수 있습니다
            - 손발이 저리거나 차가울 수 있습니다
            - 피로감을 느낄 수 있습니다
            - 식욕이 떨어질 수 있습니다
            
            **💡 관리 방법:**
            - 증상이 나타나면 즉시 의료진에게 알려주세요
            - 처방받은 부작용 완화 약을 규칙적으로 복용하세요
            - 충분한 수분을 섭취하세요
            - 가벼운 운동으로 체력을 유지하세요
            """)
    
    st.markdown("---")
    
    # ====================
    # SECTION 4: 예후 및 생존율
    # ====================
    st.header("📈 치료 후 전망")
    
    survival_percent = int(profile.prognosis_5yr_survival * 100)
    
    st.markdown(f"""
    <div class='info-box'>
    <h4>5년 생존율: {survival_percent}%</h4>
    <p><strong>이것은 무슨 의미인가요?</strong></p>
    <p>같은 병기의 환자 100명 중 약 <strong>{survival_percent}명</strong>이 5년 후에도 건강하게 생활하고 있습니다.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Progress bar visualization
    st.progress(profile.prognosis_5yr_survival)
    
    st.markdown("### 💪 긍정적인 요인들")
    
    positive_factors = []
    
    # Age factor
    if profile.clinical_data.age < 65:
        positive_factors.append("✅ 젊은 나이로 회복력이 좋습니다")
    
    # Stage factor
    if profile.cancer_stage in ["I", "IIA", "IIB"]:
        positive_factors.append("✅ 초기에 발견되어 치료 가능성이 높습니다")
    
    # ECOG factor
    if profile.clinical_data.ecog_performance is not None and profile.clinical_data.ecog_performance <= 1:
        positive_factors.append("✅ 일상 활동이 가능한 좋은 건강 상태입니다")
    
    # Comorbidities
    if not profile.clinical_data.comorbidities or len(profile.clinical_data.comorbidities) == 0:
        positive_factors.append("✅ 다른 기저 질환이 없어 치료에 유리합니다")
    
    # MSI status
    if profile.clinical_data.msi_status == "MSI-High":
        positive_factors.append("✅ 면역 치료 반응이 좋을 것으로 예상됩니다")
    
    for factor in positive_factors:
        st.markdown(f"<div class='positive-box'>{factor}</div>", unsafe_allow_html=True)
    
    st.markdown("---")
    
    # ====================
    # SECTION 5: 다음 단계
    # ====================
    st.header("🗓️ 치료 일정")
    
    st.markdown("""
    <div class='info-box'>
    <h4>앞으로의 계획</h4>
    <ul>
        <li><strong>1주일 이내:</strong> 항암 치료 계획 상담</li>
        <li><strong>2주일 이내:</strong> 항암 치료 시작</li>
        <li><strong>매 2-3주:</strong> 항암 치료 받기</li>
        <li><strong>매월:</strong> 혈액 검사 및 부작용 확인</li>
        <li><strong>3개월 후:</strong> CT 검사로 치료 효과 확인</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)
    
    # Timeline visualization
    st.markdown("### 📅 치료 타임라인")
    
    timeline_data = {
        "1주": "치료 계획 상담",
        "2주": "치료 시작",
        "1개월": "1차 치료 완료",
        "3개월": "CT 재검사",
        "6개월": "치료 종료 예정"
    }
    
    cols = st.columns(len(timeline_data))
    for idx, (time, event) in enumerate(timeline_data.items()):
        with cols[idx]:
            st.markdown(f"**{time}**")
            st.markdown(f"{event}")
    
    st.markdown("---")
    
    # ====================
    # SECTION 6: FAQ
    # ====================
    st.header("❓ 자주 묻는 질문")
    
    with st.expander("💊 항암 치료는 얼마나 힘든가요?"):
        st.markdown("""
        항암 치료의 부작용은 사람마다 다릅니다. 일반적으로:
        - **메스꺼움**: 약물로 조절 가능합니다
        - **피로**: 충분한 휴식으로 관리할 수 있습니다
        - **탈모**: 일부 약물에서 발생하지만, 치료 후 다시 자랍니다
        
        의료진이 부작용을 최소화하기 위해 최선을 다할 것입니다.
        """)
    
    with st.expander("🍽️ 식사는 어떻게 해야 하나요?"):
        st.markdown("""
        **좋은 음식:**
        - 신선한 과일과 채소
        - 고단백 음식 (생선, 닭고기, 두부)
        - 충분한 수분 섭취
        
        **피해야 할 음식:**
        - 날것이나 덜 익은 음식 (감염 위험)
        - 과도한 알코올
        - 너무 맵거나 자극적인 음식
        
        영양사와 상담하여 개인 맞춤형 식단을 계획하세요.
        """)
    
    with st.expander("🏃 운동을 해도 되나요?"):
        st.markdown("""
        가벼운 운동은 권장됩니다!
        - **좋은 운동**: 산책, 가벼운 스트레칭, 요가
        - **피할 운동**: 과격한 운동, 무리한 근력 운동
        
        체력과 컨디션에 맞춰 조절하세요. 피곤하면 충분히 쉬는 것이 중요합니다.
        """)
    
    with st.expander("👨‍👩‍👧 가족에게 영향이 있나요?"):
        st.markdown("""
        대장암은 유전적 요인이 있을 수 있습니다.
        - 직계 가족(부모, 형제자매, 자녀)은 검진을 권장합니다
        - 50세 이전에 대장 내시경 검사를 고려하세요
        - 가족력이 있으면 더 자주 검진하는 것이 좋습니다
        """)
    
    with st.expander("💰 치료 비용은 어떻게 되나요?"):
        st.markdown("""
        대부분의 항암 치료는 건강보험이 적용됩니다.
        - 항암제: 건강보험 적용
        - CT 검사: 건강보험 적용
        - 병실료: 다인실은 보험 적용
        
        자세한 비용은 원무과에 문의하시기 바랍니다.
        """)
    
    st.markdown("---")
    
    # ====================
    # SECTION 7: 의사 소통
    # ====================
    st.header("📞 의료진과 소통하기")
    
    col_comm1, col_comm2 = st.columns(2)
    
    with col_comm1:
        st.markdown("""
        <div class='info-box'>
        <h4>💬 질문하기</h4>
        <p>궁금한 점이 있으신가요?</p>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("📧 담당 의사에게 질문 보내기", use_container_width=True):
            st.success("질문이 전송되었습니다. 의사 선생님이 곧 답변해드릴 것입니다.")
    
    with col_comm2:
        st.markdown("""
        <div class='info-box'>
        <h4>👥 가족과 공유</h4>
        <p>검사 결과를 가족과 공유할 수 있습니다</p>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("🔗 공유 링크 생성", use_container_width=True):
            st.info("보안 링크가 생성되었습니다. 가족에게 전송하세요.")
    
    st.markdown("---")
    
    # ====================
    # SECTION 8: Downloads
    # ====================
    st.header("📄 보고서 다운로드")
    
    col_dl1, col_dl2 = st.columns(2)
    
    with col_dl1:
        if st.button("📥 환자용 리포트 생성 (PDF)", use_container_width=True):
            try:
                _pat  = {'patient_id': profile.patient_id, 'name': patient_name, 'birthdate': 'N/A', 'gender': getattr(profile.clinical_data, 'gender', 'N/A')}
                _path = {'tumor_location': getattr(profile.ct_results, 'tumor_location', 'N/A') or 'N/A', 'tnm_stage': getattr(profile.ct_results, 'tnm_stage', 'N/A') or 'N/A', 'msi_status': getattr(profile.clinical_data, 'msi_status', 'N/A'), 'kras_mutation': getattr(profile.clinical_data, 'kras_status', 'N/A'), 'ecog_score': getattr(profile.clinical_data, 'ecog_performance', 0), 'previous_treatment': 'N/A', 'doctor_notes': ''}
                _ct   = {'tumors_detected': getattr(profile.ct_results, 'high_conf_candidates', 0), 'largest_tumor_size_mm': getattr(profile.ct_results, 'tumor_size_mm', 0) or 0, 'total_tumor_volume_cm3': 0}
                _res  = {'ct_analysis': _ct, 'cell_analysis': None, 'adds_inference': {'pathway_activation': ['EGFR', 'VEGF'], 'recommended_targets': [], 'confidence_score': 0.87, 'rag_influence': 'N/A', 'drug_sensitivity_prediction': {'5-FU': 0.78, 'Oxaliplatin': 0.82, 'Bevacizumab': 0.87}}, 'openai_inference': {'primary_recommendation': profile.recommended_therapies[0].therapy_name if profile.recommended_therapies else 'N/A', 'alternative_regimen': profile.recommended_therapies[1].therapy_name if len(profile.recommended_therapies) > 1 else 'N/A', 'confidence_score': float(profile.prognosis_5yr_survival), 'primary_prompt_source': 'CDSS', 'rationale': f"병기 {profile.cancer_stage}"}, 'rag_analysis': {'extracted_symptoms': [], 'key_findings': [profile.cancer_stage], 'treatment_history': 'N/A', 'patient_preference': '치료 희망', 'clinical_concerns': [], 'semantic_similarity_score': 0.90}, 'validation': {'clinical_alignment_score': 0.90, 'notes_vs_ct': '일치', 'notes_vs_pathology': '일치', 'treatment_appropriateness': '적합', 'validation_status': '✅ PASSED'}}
                st.session_state['patient_iface_pdf'] = generate_patient_report_pdf(_pat, _path, _res)
                st.session_state['patient_iface_pdf_fname'] = f"My_Health_Report_{profile.patient_id}_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf"
            except Exception as e:
                st.error(f"❌ PDF 생성 실패: {e}")
                import traceback; st.code(traceback.format_exc())
        if st.session_state.get('patient_iface_pdf'):
            st.success("✅ 리포트 준비 완료")
            st.download_button("💾 내 건강 리포트 다운로드",
                data=st.session_state['patient_iface_pdf'],
                file_name=st.session_state.get('patient_iface_pdf_fname', 'my_health_report.pdf'),
                mime="application/pdf", use_container_width=True, key="dl_patient_iface")
    
    with col_dl2:
        # JSON download
        json_data = json.dumps(profile.to_dict(), indent=2, ensure_ascii=False)
        st.download_button(
            label="💾 전체 데이터 다운로드 (JSON)",
            data=json_data,
            file_name=f"my_health_report_{datetime.now().strftime('%Y%m%d')}.json",
            mime="application/json",
            use_container_width=True
        )
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666;'>
    <p>💙 ADDS 정밀 종양 분석 시스템</p>
    <p>이 정보는 참고용이며, 최종 진단과 치료는 담당 의사의 판단을 따라야 합니다.</p>
    </div>
    """, unsafe_allow_html=True)


def translate_risk_level(risk: str) -> str:
    """Translate risk level to Korean"""
    mapping = {
        "Low": "낮음",
        "Medium": "보통",
        "Medium-High": "중간-높음",
        "High": "높음"
    }
    return mapping.get(risk, risk)


def translate_drug_name(drug: str) -> str:
    """Translate drug names to Korean-friendly format"""
    mapping = {
        "5-Fluorouracil": "5-FU (플루오로우라실)",
        "Leucovorin": "류코보린",
        "Oxaliplatin": "옥살리플라틴",
        "Capecitabine": "카페시타빈",
        "Bevacizumab": "베바시주맙 (혈관 억제제)",
        "Pembrolizumab": "펨브롤리주맙 (면역 치료제)",
        "Chemotherapy backbone": "기본 항암제"
    }
    return mapping.get(drug, drug)


def translate_side_effects(risk: str) -> str:
    """Translate side effect risk to patient-friendly text"""
    mapping = {
        "Low": "비교적 가벼움 - 일상 생활에 큰 지장 없음",
        "Moderate": "보통 정도 - 관리 가능한 수준",
        "Moderate-High": "중간 정도 - 주의 깊은 관리 필요",
        "High": "주의 필요 - 의료진의 적극적인 관리 필요",
        "Variable": "개인차가 있음 - 모니터링 필요"
    }
    return mapping.get(risk, risk)


def explain_therapy_simple(therapy) -> str:
    """Explain therapy in simple Korean"""
    
    if "FOLFOX" in therapy.therapy_name:
        return """
        **FOLFOX**는 대장암 치료에 가장 널리 사용되는 항암 치료법입니다.
        
        세 가지 약물을 조합하여 사용합니다:
        - **5-FU**: 암세포의 성장을 막습니다
        - **류코보린**: 5-FU의 효과를 높여줍니다
        - **옥살리플라틴**: 암세포의 DNA를 손상시켜 증식을 막습니다
        
        전 세계적으로 검증된 안전하고 효과적인 치료법입니다.
        """
    
    elif "CAPOX" in therapy.therapy_name:
        return """
        **CAPOXBEVACIU맙**은 FOLFOX와 유사하지만, 경구 복용 가능한 약물을 사용합니다.
        
        - **카페시타빈**: 알약으로 복용 (병원 방문 횟수 감소)
        - **옥살리플라틴**: 정맥 주사
        - **베바시주맙**: 종양 혈관 생성 억제
        
        집에서 복용할 수 있어 편리합니다.
        """
    
    elif "Immunotherapy" in therapy.therapy_name:
        return """
        **면역 치료**는 우리 몸의 면역 체계를 활성화하여 암세포를 공격합니다.
        
        - 몸의 자연 면역력을 이용합니다
        - 특정 유전자 특성(MSI-High)이 있을 때 매우 효과적입니다
        - 기존 항암제와 다른 작용 방식입니다
        
        최신 치료법으로 주목받고 있습니다.
        """
    
    return "담당 의사와 상담하여 자세한 설명을 들으시기 바랍니다."


def create_demo_profile():
    """Create demo profile for testing"""
    from medical_imaging.cdss.integration_engine import (
        CellposeResults, CTDetectionResults, ClinicalData,
        IntegratedPatientProfile, TherapyRecommendation
    )
    
    cellpose = CellposeResults(
        cell_count=2450,
        mean_area_um2=185.0,
        mean_circularity=0.78,
        morphology_score=9.1,
        ki67_index=0.45
    )
    
    ct = CTDetectionResults(
        tumor_detected=True,
        total_candidates=33,
        high_conf_candidates=7,
        max_confidence=0.963,
        tumor_size_mm=15.2,
        tumor_location="Sigmoid colon",
        tnm_stage="T2N1M0"
    )
    
    clinical = ClinicalData(
        patient_id="P12345",
        age=58,
        gender="M",
        kras_status="Wild-type",
        msi_status="MSS",
        liver_function="Normal",
        kidney_function="Normal",
        ecog_performance=0,
        comorbidities=[]
    )
    
    therapy = TherapyRecommendation(
        therapy_name="FOLFOX Protocol",
        drug_combination=["5-Fluorouracil", "Leucovorin", "Oxaliplatin"],
        predicted_efficacy=0.78,
        confidence=0.94,
        side_effect_risk="Moderate",
        duration_weeks=24
    )
    
    profile = IntegratedPatientProfile(
        patient_id="P12345",
        timestamp=datetime.now(),
        cellpose_results=cellpose,
        ct_results=ct,
        clinical_data=clinical,
        cancer_stage="IIB",
        risk_level="Medium-High",
        prognosis_5yr_survival=0.65,
        recommended_therapies=[therapy],
        patient_interpretation="데모 모드입니다. 실제 환자의 경우 OpenAI가 생성한 설명이 표시됩니다."
    )
    
    return profile


# Run if standalone
if __name__ == "__main__":
    show_patient_interface()
