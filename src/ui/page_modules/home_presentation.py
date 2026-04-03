"""
Patient Consultation Presentation Mode
=======================================
환자와 상담할 때 사용하는 프레젠테이션 모드.
환자가 선택되면 항암제 칵테일 선택 이유, 현재 상태, 치료 전망을
누구나 이해하기 쉬운 시각적 UI로 보여줍니다.
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import sys
from pathlib import Path
from datetime import datetime, timedelta
import random

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


# ─────────────────────────────────────────────
#  DEMO PATIENT DATA (실제 DB 연결 시 대체)
# ─────────────────────────────────────────────
DEMO_PATIENTS = {
    "PT-001 · 김철수 (58세, 남)": {
        "id": "PT-001", "name": "김철수", "age": 58, "gender": "남",
        "cancer": "대장암 (Sigmoid Colon)", "stage": "III기 (T3N1M0)",
        "kras": "KRAS G12D 변이", "prpc": "PrPc 발현 높음 (88%)",
        "ecog": 0, "diagnosis_date": "2026-01-15",
        "regimen_name": "Pritamab + FOLFOX",
        "drugs": [
            {"name": "Pritamab (항-PrPc 항체)", "role": "PrPc 신호 차단 → KRAS G12D 탈출 경로 억제", "color": "#6366F1", "icon": "🎯"},
            {"name": "Oxaliplatin (옥살리플라틴)", "role": "DNA 이중 가닥 절단 → 암세포 증식 중단", "color": "#0891B2", "icon": "⚡"},
            {"name": "5-FU (5-플루오로우라실)", "role": "DNA 복제 효소(TS) 차단 → 암세포 분열 차단", "color": "#059669", "icon": "🔬"},
        ],
        "why_this_combo": [
            ("🧬", "KRAS G12D 변이", "이 변이가 있으면 암세포가 정상적인 항암제에 저항합니다. Pritamab이 이 저항 경로를 차단합니다."),
            ("📊", "PrPc 발현 88%", "PrPc 단백질이 높게 발현될수록 Pritamab 효과가 더 강력합니다. 환자분은 최적 대상입니다."),
            ("⚡", "Bliss 시너지 +21.7",  "Pritamab + Oxaliplatin 조합은 각 약물 단독보다 21.7% 더 강한 시너지 효과를 보였습니다 (NatureComm 확인★)."),
            ("🏆", "AI 추천 1순위", "ADDS 시스템이 7개 후보 조합 중 이 조합을 가장 높은 종합 점수(89점/100점)로 선택했습니다."),
        ],
        "current_status": {
            "cycle": 3, "total_cycles": 6,
            "tumor_change": -42,   # %  (음수 = 종양 감소)
            "response": "PR",      # CR/PR/SD/PD
            "last_ct": "2026-02-20",
            "next_ct": "2026-05-20",
            "pfs_pred_months": 14.2,
            "os_pred_months": 28.5,
            "side_effects": ["Grade 1 말초신경병증", "Grade 1 오심"],
        },
        "timeline": [
            ("2026-01-15", "진단", "대장암 3기 확인"),
            ("2026-01-28", "치료 시작", "Pritamab + FOLFOX 1주기"),
            ("2026-02-20", "중간 CT", "종양 42% 감소 — 부분 반응(PR)"),
            ("2026-03-07", "오늘", "3주기 치료 중"),
            ("2026-05-20", "예정 CT", "치료 반응 재평가"),
            ("2026-07-01", "예정", "6주기 완료 후 경과 관찰"),
        ],
        "orr_historical": 74,   # %
        "pfs_5yr": 58,          # %
    },
    "PT-002 · 이영희 (62세, 여)": {
        "id": "PT-002", "name": "이영희", "age": 62, "gender": "여",
        "cancer": "직장암 (Rectal Cancer)", "stage": "II기 (T3N0M0)",
        "kras": "KRAS Wild-type", "prpc": "PrPc 발현 중간 (60%)",
        "ecog": 1, "diagnosis_date": "2026-01-22",
        "regimen_name": "mFOLFOX6",
        "drugs": [
            {"name": "Oxaliplatin (옥살리플라틴)", "role": "DNA 이중 가닥 절단 → 암세포 증식 중단", "color": "#0891B2", "icon": "⚡"},
            {"name": "5-FU (5-플루오로우라실)", "role": "DNA 복제 효소(TS) 차단 → 암세포 분열 차단", "color": "#059669", "icon": "🔬"},
            {"name": "Leucovorin (류코보린)", "role": "5-FU의 DNA 결합력을 4배 강화하는 보조제", "color": "#D97706", "icon": "💊"},
        ],
        "why_this_combo": [
            ("🧬", "KRAS 정상형", "KRAS 변이가 없어 항-EGFR 또는 표준 FOLFOX가 효과적입니다."),
            ("📊", "PrPc 발현 중간", "PrPc 발현이 중간 수준으로 Pritamab 추가는 보류하고 경과를 관찰합니다."),
            ("🏆", "IDEA 연구 근거", "3개월 FOLFOX와 6개월 FOLFOX의 생존율이 동등 (IDEA 임상 근거)."),
            ("💊", "부작용 최소화", "II기이므로 더 강한 조합 대신 부작용이 적은 mFOLFOX6을 선택했습니다."),
        ],
        "current_status": {
            "cycle": 2, "total_cycles": 6,
            "tumor_change": -28,
            "response": "PR",
            "last_ct": "2026-02-18",
            "next_ct": "2026-05-18",
            "pfs_pred_months": 11.8,
            "os_pred_months": 24.1,
            "side_effects": ["Grade 1 탈모", "Grade 2 오심"],
        },
        "timeline": [
            ("2026-01-22", "진단", "직장암 2기 확인"),
            ("2026-02-05", "치료 시작", "mFOLFOX6 1주기"),
            ("2026-02-18", "중간 CT", "종양 28% 감소 — 부분 반응(PR)"),
            ("2026-03-07", "오늘", "2주기 치료 중"),
            ("2026-05-18", "예정 CT", "치료 반응 재평가"),
        ],
        "orr_historical": 68,
        "pfs_5yr": 65,
    },
}


def show_home_presentation():
    """환자 상담 프레젠테이션 모드 — Simple & Warm Design"""

    # ── CSS ─────────────────────────────────────────────────────────
    st.markdown("""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Noto+Sans+KR:wght@300;400;500;700&display=swap');

        html, body, [class*="css"] { font-family: 'Noto Sans KR', sans-serif; }

        /* Hero */
        .pt-hero {
            background: linear-gradient(135deg, #1e3a5f 0%, #0891B2 60%, #06B6D4 100%);
            border-radius: 20px;
            padding: 36px 40px;
            color: white;
            margin-bottom: 28px;
            box-shadow: 0 8px 32px rgba(8,145,178,0.3);
        }
        .pt-hero h1 { font-size: 30px; font-weight: 700; margin: 0 0 8px 0; }
        .pt-hero p  { font-size: 15px; margin: 4px 0; opacity: 0.92; }
        .pt-hero .badge {
            display: inline-block; background: rgba(255,255,255,0.2);
            border-radius: 20px; padding: 4px 14px; font-size: 13px;
            margin: 4px 4px 0 0; border: 1px solid rgba(255,255,255,0.3);
        }

        /* Section header */
        .sec-header {
            font-size: 20px; font-weight: 700; color: #1e3a5f;
            border-left: 5px solid #0891B2; padding-left: 14px;
            margin: 32px 0 16px 0;
        }

        /* Drug card */
        .drug-card {
            border-radius: 14px; padding: 20px 22px;
            margin: 10px 0;
            border: 1.5px solid #E2E8F0;
            background: #FAFBFD;
            transition: box-shadow 0.2s;
        }
        .drug-card:hover { box-shadow: 0 4px 18px rgba(0,0,0,0.10); }
        .drug-card .drug-icon { font-size: 28px; margin-bottom: 6px; }
        .drug-card .drug-name { font-size: 16px; font-weight: 700; color: #1e3a5f; margin: 0; }
        .drug-card .drug-role { font-size: 14px; color: #475569; margin-top: 4px; }

        /* Why card */
        .why-card {
            display: flex; align-items: flex-start; gap: 14px;
            background: #F0F9FF; border-left: 4px solid #0891B2;
            border-radius: 10px; padding: 16px 18px; margin: 10px 0;
        }
        .why-icon { font-size: 26px; min-width: 36px; text-align: center; }
        .why-title { font-size: 14px; font-weight: 700; color: #1e3a5f; margin: 0 0 4px 0; }
        .why-desc  { font-size: 13px; color: #475569; margin: 0; line-height: 1.6; }

        /* Status badge */
        .status-pr  { background: #D1FAE5; color: #065F46; border-radius: 20px;
                      padding: 4px 14px; font-weight: 700; font-size: 14px; }
        .status-cr  { background: #DBEAFE; color: #1E40AF; border-radius: 20px;
                      padding: 4px 14px; font-weight: 700; font-size: 14px; }
        .status-sd  { background: #FEF3C7; color: #92400E; border-radius: 20px;
                      padding: 4px 14px; font-weight: 700; font-size: 14px; }
        .status-pd  { background: #FEE2E2; color: #991B1B; border-radius: 20px;
                      padding: 4px 14px; font-weight: 700; font-size: 14px; }

        /* Info tile */
        .info-tile {
            background: white; border: 1px solid #E2E8F0;
            border-radius: 14px; padding: 22px;
            text-align: center; box-shadow: 0 2px 8px rgba(0,0,0,0.05);
        }
        .info-tile .label { font-size: 13px; color: #64748B; margin-bottom: 6px; }
        .info-tile .value { font-size: 32px; font-weight: 800; color: #1e3a5f; }
        .info-tile .unit  { font-size: 14px; color: #94A3B8; }
        .info-tile .delta-good { color: #059669; font-size: 15px; font-weight: 600; margin-top: 4px; }
        .info-tile .delta-warn { color: #D97706; font-size: 15px; font-weight: 600; margin-top: 4px; }

        /* Timeline */
        .tl-row { display: flex; align-items: flex-start; margin: 14px 0; gap: 16px; }
        .tl-dot  { min-width: 36px; height: 36px; border-radius: 50%; display: flex;
                   align-items: center; justify-content: center;
                   font-size: 14px; font-weight: 700; }
        .tl-dot-done   { background: #DBEAFE; color: #1e3a5f; }
        .tl-dot-today  { background: #0891B2; color: white; }
        .tl-dot-future { background: #F1F5F9; color: #94A3B8; border: 1px dashed #CBD5E1; }
        .tl-body { flex: 1; }
        .tl-date  { font-size: 12px; color: #94A3B8; margin: 0; }
        .tl-title { font-size: 15px; font-weight: 700; color: #1e3a5f; margin: 2px 0; }
        .tl-desc  { font-size: 13px; color: #475569; margin: 0; }

        /* Side effect pill */
        .se-pill {
            display: inline-block; background: #FEF3C7; color: #92400E;
            border-radius: 20px; padding: 4px 12px; font-size: 13px;
            margin: 3px 3px;
        }

        /* Footer */
        .pt-footer {
            text-align: center; color: #94A3B8; font-size: 12px;
            margin-top: 40px; padding-top: 16px; border-top: 1px solid #F1F5F9;
        }
        </style>
    """, unsafe_allow_html=True)

    # ── 환자 선택 ────────────────────────────────────────────────────
    st.markdown("#### 🔍 상담할 환자를 선택하세요")
    selected_pt_key = st.selectbox(
        "환자 선택",
        list(DEMO_PATIENTS.keys()),
        label_visibility="collapsed",
        key="pt_selector"
    )
    pt = DEMO_PATIENTS[selected_pt_key]
    st.divider()

    # ── 1. HERO — 환자 기본 정보 ─────────────────────────────────────
    response_label = {
        "CR": "완전 반응 (CR) 🌟", "PR": "부분 반응 (PR) ✅",
        "SD": "안정 (SD) 💛", "PD": "진행 (PD) ⚠️"
    }.get(pt["current_status"]["response"], "평가 중")

    st.markdown(f"""
        <div class="pt-hero">
            <h1>👤 {pt['name']}님 · {pt['age']}세 · {pt['gender']}</h1>
            <p>🏥 {pt['cancer']} — {pt['stage']}</p>
            <div style="margin-top: 12px;">
                <span class="badge">🧬 {pt['kras']}</span>
                <span class="badge">🔬 {pt['prpc']}</span>
                <span class="badge">📅 진단일 {pt['diagnosis_date']}</span>
                <span class="badge">현재 {response_label}</span>
            </div>
        </div>
    """, unsafe_allow_html=True)

    # ── 2. 항암제 칵테일 구성 ────────────────────────────────────────
    st.markdown('<div class="sec-header">💊 처방된 항암제 칵테일</div>', unsafe_allow_html=True)
    st.markdown(f"**처방 레지멘:** `{pt['regimen_name']}`")
    st.caption("이 약들이 팀처럼 함께 작용하여 암세포를 공격합니다.")

    cols = st.columns(len(pt["drugs"]))
    for col, drug in zip(cols, pt["drugs"]):
        with col:
            st.markdown(f"""
                <div class="drug-card" style="border-color:{drug['color']}30; border-top: 4px solid {drug['color']};">
                    <div class="drug-icon">{drug['icon']}</div>
                    <p class="drug-name">{drug['name']}</p>
                    <p class="drug-role">{drug['role']}</p>
                </div>
            """, unsafe_allow_html=True)

    # ── 3. 왜 이 조합인가? ──────────────────────────────────────────
    st.markdown('<div class="sec-header">🤔 왜 이 약 조합을 선택했나요?</div>', unsafe_allow_html=True)

    for icon, title, desc in pt["why_this_combo"]:
        st.markdown(f"""
            <div class="why-card">
                <div class="why-icon">{icon}</div>
                <div>
                    <p class="why-title">{title}</p>
                    <p class="why-desc">{desc}</p>
                </div>
            </div>
        """, unsafe_allow_html=True)

    # ── 4. 현재 치료 상태 ────────────────────────────────────────────
    st.markdown('<div class="sec-header">📊 현재 치료 상태</div>', unsafe_allow_html=True)

    cs = pt["current_status"]
    tumor_chg = cs["tumor_change"]
    tumor_color = "#059669" if tumor_chg < 0 else "#DC2626"
    tumor_sign  = "▼" if tumor_chg < 0 else "▲"
    tumor_text  = "종양 감소 중 👍" if tumor_chg < 0 else "종양 증가 ⚠️"

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""
            <div class="info-tile">
                <div class="label">치료 진행 상황</div>
                <div class="value">{cs['cycle']}<span class="unit"> / {cs['total_cycles']} 주기</span></div>
                <div class="delta-good">{int(cs['cycle']/cs['total_cycles']*100)}% 완료</div>
            </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
            <div class="info-tile">
                <div class="label">종양 크기 변화</div>
                <div class="value" style="color:{tumor_color};">{tumor_sign}{abs(tumor_chg)}<span class="unit">%</span></div>
                <div class="delta-good">{tumor_text}</div>
            </div>
        """, unsafe_allow_html=True)
    with c3:
        st.markdown(f"""
            <div class="info-tile">
                <div class="label">예상 무진행 생존기간</div>
                <div class="value">{cs['pfs_pred_months']}<span class="unit">개월</span></div>
                <div class="delta-good">AI 예측</div>
            </div>
        """, unsafe_allow_html=True)
    with c4:
        st.markdown(f"""
            <div class="info-tile">
                <div class="label">예상 전체 생존기간</div>
                <div class="value">{cs['os_pred_months']}<span class="unit">개월</span></div>
                <div class="delta-good">AI 예측</div>
            </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # 주기별 진행 진행바 (plotly gauge)
    c_prog, c_resp = st.columns([3, 2])
    with c_prog:
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=cs["cycle"],
            delta={"reference": cs["total_cycles"], "valueformat": "d",
                   "decreasing": {"color": "#0891B2"}},
            title={"text": f"치료 주기 진행<br><span style='font-size:13px;color:#94A3B8'>"
                           f"총 {cs['total_cycles']} 주기 중 {cs['cycle']} 주기 완료</span>"},
            gauge={
                "axis": {"range": [0, cs["total_cycles"]], "tickwidth": 1},
                "bar":  {"color": "#0891B2"},
                "steps": [
                    {"range": [0, cs["total_cycles"]], "color": "#F1F5F9"},
                    {"range": [0, cs["cycle"]], "color": "#BAE6FD"},
                ],
                "threshold": {"line": {"color": "#1e3a5f", "width": 3},
                              "thickness": 0.8, "value": cs["total_cycles"]},
            }
        ))
        fig_gauge.update_layout(
            height=250, margin=dict(l=30, r=30, t=60, b=10),
            paper_bgcolor="white", font={"family": "Noto Sans KR"}
        )
        st.plotly_chart(fig_gauge, use_container_width=True)

    with c_resp:
        status_map = {"CR": "status-cr", "PR": "status-pr",
                      "SD": "status-sd", "PD": "status-pd"}
        status_cls = status_map.get(cs["response"], "status-sd")
        status_meaning = {
            "CR": ("✅ 완전 반응",  "치료제가 암세포를 영상에서 완전히 사라지게 했습니다."),
            "PR": ("✅ 부분 반응",  "종양이 30% 이상 줄어들었습니다. 치료가 잘 되고 있습니다."),
            "SD": ("💛 안정", "종양 크기가 크게 변하지 않았습니다."),
            "PD": ("⚠️ 진행",  "종양이 커지고 있습니다. 치료 변경을 검토합니다."),
        }.get(cs["response"], ("—", ""))
        st.markdown("**📋 치료 반응 평가**")
        st.markdown(f'<span class="{status_cls}">{status_meaning[0]}</span>', unsafe_allow_html=True)
        st.markdown(f"<p style='color:#475569; margin-top:10px; font-size:14px;'>{status_meaning[1]}</p>",
                    unsafe_allow_html=True)
        st.markdown(f"""
            <div style='margin-top:16px; padding:14px; background:#F8FAFC; border-radius:10px;'>
                <p style='font-size:13px; margin:4px 0;'>📅 마지막 CT: <strong>{cs['last_ct']}</strong></p>
                <p style='font-size:13px; margin:4px 0;'>📅 다음 CT 예정: <strong>{cs['next_ct']}</strong></p>
            </div>
        """, unsafe_allow_html=True)

        # 부작용
        if cs.get("side_effects"):
            st.markdown("**⚠️ 현재 부작용**")
            pills = "".join([f'<span class="se-pill">{se}</span>' for se in cs["side_effects"]])
            st.markdown(f"<div>{pills}</div>", unsafe_allow_html=True)
        else:
            st.success("✅ 현재 주요 부작용 없음")

    # ── 5. 생존율 시각화 ────────────────────────────────────────────
    st.markdown('<div class="sec-header">📈 같은 조건 환자와 비교</div>', unsafe_allow_html=True)

    orr = pt["orr_historical"]
    pfs5 = pt["pfs_5yr"]

    c5a, c5b = st.columns(2)
    with c5a:
        # ORR bar
        fig_orr = go.Figure()
        fig_orr.add_trace(go.Bar(
            y=["같은 조합으로<br>치료받은 환자 중"],
            x=[orr],
            orientation='h',
            marker_color="#0891B2",
            text=[f"  {orr}% 반응"],
            textposition="inside",
            insidetextfont=dict(color="white", size=15, family="Noto Sans KR"),
        ))
        fig_orr.add_trace(go.Bar(
            y=["같은 조합으로<br>치료받은 환자 중"],
            x=[100 - orr],
            orientation='h',
            marker_color="#E2E8F0",
            showlegend=False,
        ))
        fig_orr.update_layout(
            title=dict(text="종양 반응률 (ORR)", font=dict(size=16, color="#1e3a5f")),
            barmode='stack',
            height=180, margin=dict(l=10, r=20, t=50, b=10),
            paper_bgcolor="white", plot_bgcolor="white",
            xaxis=dict(range=[0, 100], showticklabels=False, showgrid=False, zeroline=False),
            yaxis=dict(showgrid=False),
            showlegend=False,
            font=dict(family="Noto Sans KR"),
        )
        st.plotly_chart(fig_orr, use_container_width=True)
        st.caption(f"**해석:** {orr}명 중 {int(orr*0.5)}명 이상이 종양 감소를 경험했습니다.")

    with c5b:
        fig_pfs = go.Figure(go.Indicator(
            mode="gauge+number",
            value=pfs5,
            title={"text": "5년 무진행 생존율", "font": {"size": 16, "color": "#1e3a5f"}},
            number={"suffix": "%", "font": {"size": 40, "color": "#1e3a5f"}},
            gauge={
                "axis": {"range": [0, 100]},
                "bar":  {"color": "#0891B2"},
                "steps": [
                    {"range": [0, 50],  "color": "#FEE2E2"},
                    {"range": [50, 70], "color": "#FEF3C7"},
                    {"range": [70, 100],"color": "#D1FAE5"},
                ],
            }
        ))
        fig_pfs.update_layout(
            height=200, margin=dict(l=30, r=30, t=60, b=10),
            paper_bgcolor="white",
            font=dict(family="Noto Sans KR"),
        )
        st.plotly_chart(fig_pfs, use_container_width=True)
        st.caption(f"**해석:** 같은 단계·조합으로 치료받은 환자의 {pfs5}%가 5년 후에도 재발 없이 생활합니다.")

    # ── 6. 치료 타임라인 ─────────────────────────────────────────────
    st.markdown('<div class="sec-header">🗓️ 치료 타임라인</div>', unsafe_allow_html=True)
    today_str = "2026-03-07"

    for date, title, desc in pt["timeline"]:
        if title == "오늘":
            dot_cls = "tl-dot tl-dot-today"
            dot_txt = "NOW"
        elif date > today_str:
            dot_cls = "tl-dot tl-dot-future"
            dot_txt = "예정"
        else:
            dot_cls = "tl-dot tl-dot-done"
            dot_txt = "✓"

        st.markdown(f"""
            <div class="tl-row">
                <div class="{dot_cls}">{dot_txt}</div>
                <div class="tl-body">
                    <p class="tl-date">{date}</p>
                    <p class="tl-title">{title}</p>
                    <p class="tl-desc">{desc}</p>
                </div>
            </div>
        """, unsafe_allow_html=True)

    # ── 7. 자주 묻는 질문 ────────────────────────────────────────────
    st.markdown('<div class="sec-header">❓ 자주 묻는 질문</div>', unsafe_allow_html=True)

    with st.expander("💊 왜 한 가지가 아닌 여러 약을 함께 쓰나요?"):
        st.markdown("""
        암세포는 단 하나의 약물만 사용하면 그 약을 피하는 방법(내성)을 개발하여 다시 성장합니다.
        여러 약물을 함께 사용하면:
        - 각 약이 **다른 경로**로 암세포를 공격합니다
        - 암세포가 **동시에 여러 경로를 피하기 어렵습니다**
        - 결과적으로 **더 강력하고 지속적인 치료 효과**를 얻습니다
        """)

    with st.expander("🤖 AI가 이 조합을 어떻게 결정했나요?"):
        st.markdown("""
        ADDS AI 시스템은 세 가지 데이터를 분석했습니다:

        1. **세포 현미경 이미지**: 암세포의 모양과 증식 속도
        2. **유전자 분석**: KRAS 변이, PrPc 발현 수준
        3. **CT 영상**: 종양 크기와 위치

        이 데이터를 100만 명 이상의 임상 데이터와 비교하여 **이 환자에게 가장 효과적인 조합**을 선택했습니다.
        최종 처방은 **담당 의사 선생님이 AI 결과를 검토·승인**한 것입니다.
        """)

    with st.expander("🤢 부작용이 심하면 어떻게 하나요?"):
        st.markdown("""
        부작용이 발생하면 즉시 의료진에게 알려주세요.

        - **Grade 1 (가벼움)**: 일상생활 가능, 관찰 계속
        - **Grade 2 (중간)**: 약물로 조절, 필요시 용량 조절
        - **Grade 3 이상**: 치료 일시 중단 또는 다른 약으로 변경

        의료진은 항상 **부작용과 치료 효과의 균형**을 최우선으로 고려합니다.
        """)

    with st.expander("💪 치료 기간 동안 무엇을 할 수 있나요?"):
        st.markdown("""
        | 권장 사항 | 삼가할 것 |
        |-----------|-----------|
        | ✅ 가벼운 산책 (하루 20-30분) | ❌ 격렬한 운동 |
        | ✅ 균형 잡힌 식단 | ❌ 날 음식, 날 생선 |
        | ✅ 충분한 수분 섭취 | ❌ 과도한 음주 |
        | ✅ 충분한 수면 | ❌ 흡연 |
        | ✅ 좋아하는 취미 활동 | ❌ 면역 관련 생백신 |
        """)

    # ── Footer ───────────────────────────────────────────────────────
    st.markdown("""
        <div class="pt-footer">
            <p>🏥 ADDS 정밀 종양 AI 시스템 | 이 정보는 의료진 상담을 보조하기 위한 것이며 최종 판단은 담당 의사의 권한입니다.</p>
        </div>
    """, unsafe_allow_html=True)
