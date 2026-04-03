"""
Energy Framework UI Page for ADDS Streamlit
Connects to real backend API at /api/v1/energy/*
No demo data — all predictions from live GNN model
"""

import streamlit as st
import requests
import json
import pandas as pd
import numpy as np
from pathlib import Path

# Backend URL (same as backend_client.py)
BACKEND_URL = "http://localhost:8000"

# ─── API Calls (real backend) ─────────────────────────────────────────────────

def api_predict(drug_name, kd_nm=None, ic50_nm=None, cell_line=None,
                mutations=None, combo_drug=None, combo_ic50=None):
    """Call /api/v1/energy/predict — real GNN model."""
    payload = {"drug_name": drug_name}
    if kd_nm: payload["kd_nm"] = kd_nm
    if ic50_nm: payload["ic50_nm"] = ic50_nm
    if cell_line: payload["cell_line"] = cell_line
    if mutations: payload["mutations"] = mutations
    if combo_drug: payload["combination_drug"] = combo_drug
    if combo_ic50: payload["combination_ic50_nm"] = combo_ic50
    try:
        r = requests.post(f"{BACKEND_URL}/api/v1/energy/predict",
                         json=payload, timeout=15)
        if r.status_code == 200:
            return r.json()
        return {"error": f"API {r.status_code}: {r.text[:200]}"}
    except Exception as e:
        return {"error": str(e)}


def api_pathway_graph():
    """Call /api/v1/energy/pathway-graph — real learned edges."""
    try:
        r = requests.get(f"{BACKEND_URL}/api/v1/energy/pathway-graph", timeout=10)
        return r.json() if r.status_code == 200 else None
    except:
        return None


def api_calibration_status():
    """Call /api/v1/energy/calibration-status."""
    try:
        r = requests.get(f"{BACKEND_URL}/api/v1/energy/calibration-status", timeout=10)
        return r.json() if r.status_code == 200 else None
    except:
        return None


def api_deep_synergy(drug_a, drug_b, cell_line="HCT116"):
    """Call /api/v1/energy/deep-synergy — DeepSynergy MLP model."""
    try:
        r = requests.post(
            f"{BACKEND_URL}/api/v1/energy/deep-synergy",
            params={"drug_a": drug_a, "drug_b": drug_b, "cell_line": cell_line},
            timeout=15)
        if r.status_code == 200:
            return r.json()
        return {"error": f"API {r.status_code}: {r.text[:200]}"}
    except Exception as e:
        return {"error": str(e)}


def api_deep_synergy_drugs():
    """Get available drugs for DeepSynergy."""
    try:
        r = requests.get(f"{BACKEND_URL}/api/v1/energy/deep-synergy/drugs", timeout=10)
        return r.json() if r.status_code == 200 else None
    except:
        return None


# ─── Drug Presets (from literature) ───────────────────────────────────────────

DRUG_PRESETS = {
    "Cetuximab (anti-EGFR mAb)":    {"kd_nm": 0.5,  "ic50_nm": None, "targets": "EGFR"},
    "Panitumumab (anti-EGFR mAb)":  {"kd_nm": 0.05, "ic50_nm": None, "targets": "EGFR"},
    "5-Fluorouracil (항대사제)":     {"kd_nm": None, "ic50_nm": 5000, "targets": "TS (Thymidylate Synthase)"},
    "Oxaliplatin (백금 화합물)":     {"kd_nm": None, "ic50_nm": 1200, "targets": "DNA"},
    "Irinotecan (Top1 억제제)":     {"kd_nm": None, "ic50_nm": 2000, "targets": "Topoisomerase I"},
    "Bevacizumab (anti-VEGF)":      {"kd_nm": 0.5,  "ic50_nm": None, "targets": "VEGF"},
    "Encorafenib (BRAF 억제제)":    {"kd_nm": None, "ic50_nm": 3.2,  "targets": "BRAF V600E"},
    "Binimetinib (MEK 억제제)":     {"kd_nm": None, "ic50_nm": 12,   "targets": "MEK1/2"},
    "Pembrolizumab (anti-PD-1)":    {"kd_nm": 0.03, "ic50_nm": None, "targets": "PD-1"},
    "Nivolumab (anti-PD-1)":        {"kd_nm": 0.02, "ic50_nm": None, "targets": "PD-1"},
    "Everolimus (mTOR 억제제)":     {"kd_nm": None, "ic50_nm": 1.6,  "targets": "mTOR"},
    "Erlotinib (EGFR TKI)":         {"kd_nm": None, "ic50_nm": 2.0,  "targets": "EGFR Kinase"},
    "직접 입력": None,
}

MUTATION_OPTIONS = {
    "없음": {},
    "KRAS G12V": {"RAS": "G12V"},
    "KRAS G12D": {"RAS": "G12D"},
    "BRAF V600E": {"RAS": "V600E"},
    "PIK3CA H1047R": {"PI3K": "H1047R"},
    "TP53 R175H": {"survival": "R175H"},
    "APC Truncation": {"Wnt": "truncation"},
    "KRAS G12V + BRAF V600E": {"RAS": "G12V+V600E"},
}


# ─── DrugComb Validation Results (from disk) ──────────────────────────────────

def load_validation_results():
    """Load real DrugComb validation results from saved JSON."""
    paths = [
        Path(BASE_DIR / "models/energy/drugcomb_validation_v2.json"),
        Path(BASE_DIR / "models/energy/drugcomb_validation.json"),
    ]
    for p in paths:
        if p.exists():
            with open(p) as f:
                return json.load(f), p.name
    return None, None


# ─── Main Page ────────────────────────────────────────────────────────────────

def show_energy_framework():
    """Energy Framework page — real API, no demo data."""

    st.markdown("""
    <h2 style="text-align:center; background: linear-gradient(90deg, #f093fb 0%, #f5576c 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
    ⚡ 에너지 기반 약물 시너지 프레임워크</h2>
    """, unsafe_allow_html=True)

    # Check backend connection
    try:
        r = requests.get(f"{BACKEND_URL}/health", timeout=3)
        connected = r.status_code == 200
    except:
        connected = False

    if connected:
        st.success("🟢 백엔드 API 연결됨 (localhost:8000)")
    else:
        st.error("🔴 백엔드 API 연결 실패 — `python -m uvicorn backend.main:app --port 8000` 실행 필요")
        return

    # Tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🧬 약물 에너지 예측",
        "💊 병용 시너지 분석",
        "🕸️ Pathway Graph",
        "📊 DrugComb 검증",
        "🧠 DeepSynergy AI"
    ])

    # ── Tab 1: Single Drug Prediction ──
    with tab1:
        _show_single_prediction()

    # ── Tab 2: Combination Synergy ──
    with tab2:
        _show_combination_prediction()

    # ── Tab 3: Pathway Graph ──
    with tab3:
        _show_pathway_graph()

    # ── Tab 4: DrugComb Validation ──
    with tab4:
        _show_drugcomb_validation()

    # ── Tab 5: DeepSynergy AI ──
    with tab5:
        _show_deep_synergy()


def _show_single_prediction():
    """Single drug energy prediction tab."""
    st.markdown("### 단일 약물 에너지 예측")
    st.caption("GNN v3 모델이 약물의 결합 에너지 → pathway 전파 → 종양 억제 효과를 예측합니다.")

    col1, col2 = st.columns([1, 1])
    with col1:
        drug_sel = st.selectbox("약물 선택", list(DRUG_PRESETS.keys()), key="single_drug")
        preset = DRUG_PRESETS.get(drug_sel)

        if preset:
            kd = preset.get("kd_nm")
            ic50 = preset.get("ic50_nm")
            st.info(f"📎 타깃: {preset['targets']}")
        else:
            kd = st.number_input("KD (nM)", value=1.0, min_value=0.001, key="custom_kd")
            ic50 = st.number_input("IC50 (nM)", value=100.0, min_value=0.01, key="custom_ic50")

    with col2:
        cell_line = st.selectbox("세포주",
            ["HCT116", "SW480", "HT29", "Colo205", "DLD-1", "LoVo", "RKO", "SW620", "HCT-15", "Caco-2"],
            key="single_cell")
        mut_sel = st.selectbox("돌연변이", list(MUTATION_OPTIONS.keys()), key="single_mut")
        mutations = MUTATION_OPTIONS[mut_sel] or None

    if st.button("⚡ 에너지 예측 실행", key="btn_single", use_container_width=True, type="primary"):
        drug_name = drug_sel.split("(")[0].strip()
        with st.spinner("GNN 모델 추론 중..."):
            result = api_predict(drug_name, kd_nm=kd, ic50_nm=ic50,
                               cell_line=cell_line, mutations=mutations)

        if "error" in result:
            st.error(f"❌ {result['error']}")
            return

        # Results
        st.markdown("---")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("결합 에너지", f"{result['binding_energy_kcal']:.1f} kcal/mol")
        c2.metric("종양 억제", f"{result['predicted_tumor_suppression_pct']:.1f}%")
        c3.metric("예측 IC50", f"{result['predicted_ic50_nm']:.1f} nM")
        c4.metric("모델 버전", result.get('model_version', 'v3'))

        # Pathway energies
        if result.get("pathway_energies"):
            st.markdown("#### Pathway 에너지 분포")
            pe = result["pathway_energies"]
            df = pd.DataFrame({
                "Pathway": list(pe.keys()),
                "Energy": list(pe.values())
            }).sort_values("Energy", ascending=False)
            st.bar_chart(df.set_index("Pathway"), height=350)

            # Save to session state for comparison
            st.session_state['last_single_result'] = result


def _show_combination_prediction():
    """Combination drug synergy tab."""
    st.markdown("### 💊 병용 약물 시너지 분석")
    st.caption("두 약물의 조합 시너지를 GNN 에너지 모델로 예측합니다.")

    col1, col2 = st.columns(2)
    drug_names = [d for d in DRUG_PRESETS.keys() if d != "직접 입력"]

    with col1:
        st.markdown("**약물 A**")
        drug_a_sel = st.selectbox("약물 A", drug_names, index=0, key="combo_a")
        preset_a = DRUG_PRESETS[drug_a_sel]

    with col2:
        st.markdown("**약물 B**")
        drug_b_sel = st.selectbox("약물 B", drug_names, index=4, key="combo_b")
        preset_b = DRUG_PRESETS[drug_b_sel]

    cell_line = st.selectbox("세포주",
        ["HCT116", "SW480", "HT29", "DLD-1", "LoVo", "Colo205"],
        key="combo_cell")
    mut_sel = st.selectbox("돌연변이", list(MUTATION_OPTIONS.keys()), key="combo_mut")
    mutations = MUTATION_OPTIONS[mut_sel] or None

    if st.button("⚡ 시너지 예측 실행", key="btn_combo", use_container_width=True, type="primary"):
        name_a = drug_a_sel.split("(")[0].strip()
        name_b = drug_b_sel.split("(")[0].strip()

        combo_ic50 = preset_b.get("ic50_nm") or (preset_b.get("kd_nm", 1) * 10)

        with st.spinner("GNN 병용 추론 중..."):
            result = api_predict(
                name_a,
                kd_nm=preset_a.get("kd_nm"),
                ic50_nm=preset_a.get("ic50_nm"),
                cell_line=cell_line,
                mutations=mutations,
                combo_drug=name_b,
                combo_ic50=combo_ic50
            )

        if "error" in result:
            st.error(f"❌ {result['error']}")
            return

        st.markdown("---")

        # Synergy result
        ci = result.get("predicted_synergy_ci")
        interp = result.get("synergy_interpretation", "N/A")

        c1, c2, c3 = st.columns(3)
        c1.metric("결합 에너지", f"{result['binding_energy_kcal']:.1f} kcal/mol")

        if ci is not None:
            color = "🟢" if ci < 0.8 else ("🟡" if ci < 1.2 else "🔴")
            c2.metric("CI (Combination Index)", f"{ci:.3f} {color}")
            c3.metric("해석", interp)

            if ci < 0.3:
                st.success(f"💪 **강한 시너지** — {name_a} + {name_b}는 매우 효과적인 조합입니다.")
            elif ci < 0.8:
                st.success(f"✅ **시너지** — {name_a} + {name_b}는 상승 효과가 있습니다.")
            elif ci < 1.2:
                st.info(f"➡️ **상가 효과** — {name_a} + {name_b}는 단순 합산 수준입니다.")
            else:
                st.warning(f"⚠️ **길항 효과** — {name_a} + {name_b}는 상호 간섭할 수 있습니다.")

            # Competitive binding warning
            same_target_combos = [
                ({"Cetuximab", "Panitumumab"}, "EGFR 항체"),
                ({"5-Fluorouracil", "Capecitabine"}, "TS 억제제"),
                ({"Pembrolizumab", "Nivolumab"}, "PD-1 항체"),
            ]
            for drug_set, target in same_target_combos:
                if {name_a, name_b} == drug_set:
                    st.error(f"⚠️ **경쟁적 결합 경고**: {name_a}와 {name_b}는 동일한 {target}에 "
                            f"결합하므로 경쟁적 길항이 예상됩니다. DrugComb 실측 데이터에서도 "
                            f"이 조합은 길항적 (Loewe < 0)으로 확인되었습니다.")
        else:
            c2.metric("종양 억제", f"{result['predicted_tumor_suppression_pct']:.1f}%")
            c3.metric("예측 IC50", f"{result['predicted_ic50_nm']:.1f} nM")


def _show_pathway_graph():
    """GNN learned pathway graph visualization."""
    st.markdown("### 🕸️ GNN 학습된 Pathway 네트워크")

    graph_data = api_pathway_graph()
    if not graph_data:
        st.error("Pathway graph를 불러올 수 없습니다.")
        return

    cal_status = api_calibration_status()

    # Summary
    c1, c2, c3 = st.columns(3)
    c1.metric("노드 수", graph_data.get("n_nodes", 0))
    c2.metric("활성 엣지", graph_data.get("n_active_edges", 0))
    c3.metric("캘리브레이션",
              graph_data.get("calibration_source", "none"))

    if cal_status:
        st.markdown("#### 캘리브레이션 상태")
        cc1, cc2, cc3 = st.columns(3)
        cc1.metric("PPI 데이터", "✅ 로드됨" if cal_status.get("biogrid_ppi_loaded") else "❌")
        cc2.metric("DrugComb 행수", cal_status.get("drugcomb_rows_used", 0))
        val_r = cal_status.get("validation_r_loewe")
        cc3.metric("검증 Pearson r", f"{val_r:.3f}" if val_r else "N/A")

    # Edges table
    edges = graph_data.get("edges", [])
    if edges:
        st.markdown("#### 상위 엣지 (weight 순)")

        # Separate by source type
        ppi_edges = [e for e in edges if e.get("source_type") != "learned"]
        learned_edges = [e for e in edges if e.get("source_type") == "learned"]

        tab_ppi, tab_learned = st.tabs(
            [f"🔬 Real PPI ({len(ppi_edges)}개)", f"🧠 GNN Learned ({len(learned_edges)}개)"])

        with tab_ppi:
            if ppi_edges:
                df_ppi = pd.DataFrame(ppi_edges).sort_values("weight", ascending=False)
                df_ppi = df_ppi.rename(columns={
                    "source": "출발", "target": "도착", "weight": "강도",
                    "source_type": "출처"
                })
                st.dataframe(df_ppi[["출발", "도착", "강도", "출처"]], 
                           use_container_width=True, hide_index=True)
            else:
                st.info("PPI 엣지가 없습니다.")

        with tab_learned:
            if learned_edges:
                df_learned = pd.DataFrame(learned_edges).sort_values("weight", ascending=False)
                df_learned = df_learned.rename(columns={
                    "source": "출발", "target": "도착", "weight": "강도"
                })
                st.dataframe(df_learned[["출발", "도착", "강도"]],
                           use_container_width=True, hide_index=True)

                # Bar chart of top learned edges
                top15 = df_learned.head(15).copy()
                top15["엣지"] = top15["출발"] + " → " + top15["도착"]
                st.bar_chart(top15.set_index("엣지")["강도"], height=300)


def _show_drugcomb_validation():
    """DrugComb cross-validation results."""

import os as _os
from pathlib import Path as _Path
# ADDS_BASE_DIR environment variable overrides automatic detection
BASE_DIR = _Path(_os.environ.get("ADDS_BASE_DIR", str(_Path(__file__).resolve().parent.parent)))

    st.markdown("### 📊 DrugComb 실데이터 교차 검증")

    data, fname = load_validation_results()
    if not data:
        st.warning("검증 결과 파일이 없습니다. `validate_drugcomb_v2.py` 실행이 필요합니다.")
        return

    st.caption(f"📁 결과 파일: `models/energy/{fname}`")

    # Summary metrics
    fixes = data.get("fixes_applied", [])
    if fixes:
        st.success("✅ 적용된 개선: " + " | ".join(fixes))

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("데이터 수", data.get("n_datapoints", 0))
    c2.metric("Pearson r", f"{data.get('pearson_r', 0):.4f}")
    c3.metric("시너지 정확도", f"{data.get('synergy_accuracy', 0)*100:.1f}%")
    c4.metric("길항 정확도", f"{data.get('antagonist_accuracy', 0)*100:.1f}%")

    c5, c6, c7, c8 = st.columns(4)
    c5.metric("Cell Lines", data.get("n_cell_lines", 1))
    c6.metric("Drug Pairs", data.get("n_pairs", 0))
    c7.metric("RMSE", data.get("rmse_loewe", 0))
    three_class = data.get("three_class_accuracy")
    c8.metric("3-class 정확도", f"{three_class*100:.1f}%" if three_class else "N/A")

    # Per-pair results
    pairs = data.get("per_pair", [])
    if pairs:
        st.markdown("#### 약물 쌍별 결과")
        df = pd.DataFrame(pairs)
        df = df.rename(columns={
            "drug_a": "약물A", "drug_b": "약물B",
            "true_mean": "실측 Loewe", "pred_mean": "예측 Loewe",
            "r": "상관계수", "class_true": "실측 분류", "class_pred": "예측 분류",
            "competitive": "경쟁적 결합",
        })

        # Color coding
        def color_match(row):
            if row.get("실측 분류") == row.get("예측 분류"):
                return ["background-color: #d4edda"] * len(row)
            return ["background-color: #f8d7da"] * len(row)

        cols_show = ["약물A", "약물B", "실측 Loewe", "예측 Loewe",
                    "상관계수", "실측 분류", "예측 분류"]
        if "경쟁적 결합" in df.columns:
            cols_show.append("경쟁적 결합")

        styled = df[cols_show].style.apply(color_match, axis=1)
        st.dataframe(styled, use_container_width=True, hide_index=True)

    # Per cell line
    cell_data = data.get("per_cell_line", {})
    if cell_data:
        st.markdown("#### 세포주별 성능")
        df_cell = pd.DataFrame([
            {"세포주": cl, **vals} for cl, vals in cell_data.items()
        ]).sort_values("r", ascending=False)
        st.dataframe(df_cell, use_container_width=True, hide_index=True)


def _show_deep_synergy():
    """DeepSynergy MLP model predictions tab."""
    st.markdown("### 🧠 DeepSynergy AI — 딥러닝 약물 시너지 예측")
    st.caption(
        "Morgan Fingerprint (2048-bit) + 분자 특성 + 세포주 변이 프로파일 → "
        "MLP [4130→2048→1024→512→128→1] — "
        "18,532행 학습, CRC 592행 fine-tune (r=0.60)"
    )

    # Load available drugs
    drug_info = api_deep_synergy_drugs()
    if drug_info is None:
        st.error("DeepSynergy 모델을 불러올 수 없습니다. 백엔드를 확인하세요.")
        return

    # Model metrics banner
    metrics = drug_info.get("model_metrics", {})
    mc1, mc2, mc3 = st.columns(3)
    mc1.metric("학습 데이터", f"{metrics.get('training_data', 0):,}행")
    mc2.metric("Test Pearson r", f"{metrics.get('test_pearson_r', 0):.4f}")
    mc3.metric("CRC Pearson r", f"{metrics.get('crc_pearson_r', 0):.4f}")

    st.markdown("---")

    # Build flat drug list from grouped data
    drugs_by_class = drug_info.get("drugs_by_class", {})
    all_drugs = []
    for cls, drugs in sorted(drugs_by_class.items()):
        for d in drugs:
            label = f"{d['name']} [{cls}]" + ("" if d['has_smiles'] else " (no SMILES)")
            all_drugs.append((label, d['name']))

    cell_lines = drug_info.get("cell_lines", ["HCT116"])

    # Input
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**약물 A**")
        sel_a = st.selectbox("약물 A 선택", [x[0] for x in all_drugs], index=0, key="ds_drug_a")
        drug_a = next(d[1] for d in all_drugs if d[0] == sel_a)

    with col2:
        st.markdown("**약물 B**")
        sel_b = st.selectbox("약물 B 선택", [x[0] for x in all_drugs], index=min(3, len(all_drugs)-1), key="ds_drug_b")
        drug_b = next(d[1] for d in all_drugs if d[0] == sel_b)

    cell_line = st.selectbox("세포주", cell_lines, index=0, key="ds_cell")

    if st.button("🧠 DeepSynergy 예측 실행", key="btn_ds", use_container_width=True, type="primary"):
        with st.spinner("DeepSynergy MLP 추론 중..."):
            result = api_deep_synergy(drug_a, drug_b, cell_line)

        if "error" in result:
            st.error(f"❌ {result['error']}")
            return

        st.markdown("---")

        # Main results
        score = result.get('synergy_score', 0)
        cls = result.get('classification', 'unknown')
        conf = result.get('confidence', 0)

        rc1, rc2, rc3, rc4 = st.columns(4)
        rc1.metric("시너지 점수", f"{score:+.1f}")

        if cls == 'synergistic':
            rc2.metric("분류", "🟢 시너지")
        elif cls == 'antagonistic':
            rc2.metric("분류", "🔴 길항")
        else:
            rc2.metric("분류", "🟡 상가")

        rc3.metric("신뢰도", f"{conf:.1%}")
        rc4.metric("모델", result.get('model_version', 'N/A'))

        # Interpretation
        if cls == 'synergistic':
            st.success(
                f"💪 **시너지** — {drug_a} + {drug_b}는 {cell_line}에서 "
                f"상승 효과가 예상됩니다 (score={score:+.1f})."
            )
        elif cls == 'antagonistic':
            st.warning(
                f"⚠️ **길항** — {drug_a} + {drug_b}는 {cell_line}에서 "
                f"상호 간섭이 예상됩니다 (score={score:+.1f})."
            )
        else:
            st.info(
                f"➡️ **상가** — {drug_a} + {drug_b}는 {cell_line}에서 "
                f"단순 합산 수준입니다 (score={score:+.1f})."
            )

        # Competitive binding warning
        if result.get('competitive_binding_penalty'):
            st.error(
                f"🚫 **경쟁적 결합 감지**: {drug_a}와 {drug_b}는 동일 타깃에 "
                f"경쟁적으로 결합하여 길항 효과가 예상됩니다. "
                f"(페널티: {result['competitive_binding_penalty']:+.1f})"
            )

        # Drug details
        with st.expander("🔍 상세 정보"):
            dc1, dc2 = st.columns(2)
            smiles_a = "\u2713" if result.get('drug_a_has_smiles') else "\u2717 (항체 약물)"
            smiles_b = "\u2713" if result.get('drug_b_has_smiles') else "\u2717 (항체 약물)"
            dc1.write(f"**약물 A**: {drug_a}")
            dc1.write(f"  클래스: {result.get('drug_a_class', 'N/A')}")
            dc1.write(f"  SMILES: {smiles_a}")
            dc2.write(f"**약물 B**: {drug_b}")
            dc2.write(f"  클래스: {result.get('drug_b_class', 'N/A')}")
            dc2.write(f"  SMILES: {smiles_b}")

    # Quick comparison: run all CRC cell lines
    st.markdown("---")
    st.markdown("#### 보너스: 모든 세포주 비교")
    if st.button("📊 전체 CRC 세포주로 예측", key="ds_all_cells"):
        crc_cells = [c for c in cell_lines if c not in ['A549','MCF7','PC-3','MDA-MB-231',
                     'PANC-1','SK-OV-3','U-87','HepG2','K-562','GENERIC']]
        results = []
        with st.spinner(f"{len(crc_cells)}개 세포주 예측 중..."):
            for cl in crc_cells:
                r = api_deep_synergy(drug_a, drug_b, cl)
                if 'error' not in r:
                    results.append({
                        "세포주": cl,
                        "시너지 점수": r['synergy_score'],
                        "분류": r['classification'],
                        "신뢰도": f"{r['confidence']:.1%}",
                    })
        if results:
            df_all = pd.DataFrame(results).sort_values("시너지 점수", ascending=False)
            st.dataframe(df_all, use_container_width=True, hide_index=True)
            st.bar_chart(df_all.set_index("세포주")["시너지 점수"], height=300)
