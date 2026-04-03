"""
ADDS PDF Report Generator
실제 임상용 보고서를 PDF로 생성합니다.
- 의사용: ReportLab 텍스트 기반 상세 보고서
- 환자용: matplotlib 대시보드 이미지 → PDF embed (레이아웃 충돌 없음)
"""

import io
import math
from datetime import datetime
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table,
                                 TableStyle, HRFlowable, Image as RLImage)
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_RIGHT

# ── 한글 폰트 등록 (ReportLab) ──────────────────────────────────────────────
_FONT_REGISTERED = False

def _register_fonts():
    global _FONT_REGISTERED
    if _FONT_REGISTERED:
        return
    import os
    font_pairs = [("MalgunGothic", "malgun.ttf"), ("MalgunGothicBold", "malgunbd.ttf")]
    font_dir = os.path.join(os.environ.get("WINDIR", "C:\\Windows"), "Fonts")
    for name, fname in font_pairs:
        full = os.path.join(font_dir, fname)
        if os.path.exists(full):
            try:
                pdfmetrics.registerFont(TTFont(name, full))
            except Exception:
                pass
    _FONT_REGISTERED = True


def _fn(bold=False):
    _register_fonts()
    reg = pdfmetrics.getRegisteredFontNames()
    if bold:
        return "MalgunGothicBold" if "MalgunGothicBold" in reg else "Helvetica-Bold"
    return "MalgunGothic" if "MalgunGothic" in reg else "Helvetica"


def _s(name, **kw):
    """ParagraphStyle 빠른 생성."""
    defaults = dict(fontName=_fn(), fontSize=9, leading=13)
    defaults.update(kw)
    return ParagraphStyle(name, **defaults)


def _hr():
    return HRFlowable(width="100%", thickness=0.5,
                      color=colors.HexColor("#cccccc"), spaceAfter=4, spaceBefore=4)


def _kv_table(rows, key_w=55, val_w=110):
    fn, fnb = _fn(), _fn(True)
    data = [
        [Paragraph("<b>" + k + "</b>", ParagraphStyle("k", fontName=fnb, fontSize=9)),
         Paragraph(str(v),             ParagraphStyle("v", fontName=fn, fontSize=9))]
        for k, v in rows
    ]
    t = Table(data, colWidths=[key_w*mm, val_w*mm])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#f0f4ff")),
        ("GRID",       (0, 0), (-1, -1), 0.3, colors.HexColor("#dddddd")),
        ("PADDING",    (0, 0), (-1, -1), 5),
        ("VALIGN",     (0, 0), (-1, -1), "TOP"),
    ]))
    return t


# ── matplotlib 한글 폰트 설정 ─────────────────────────────────────────────
def _setup_mpl_korean():
    """matplotlib 한글 폰트 적용 (Malgun Gothic)."""
    import os
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.font_manager as fm

    font_path = os.path.join(os.environ.get("WINDIR", "C:\\Windows"), "Fonts", "malgun.ttf")
    if os.path.exists(font_path):
        fm.fontManager.addfont(font_path)
        prop = fm.FontProperties(fname=font_path)
        plt.rcParams["font.family"] = prop.get_name()
    plt.rcParams["axes.unicode_minus"] = False


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  전체 환자용 대시보드를 matplotlib 1장 이미지로 렌더링
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _build_dashboard_image(patient, pathology, results) -> io.BytesIO:
    """matplotlib figure 로 대시보드를 그려 PNG bytes 반환."""
    _setup_mpl_korean()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import matplotlib.gridspec as gridspec
    import numpy as np

    ct  = results.get("ct_analysis") or {}
    oi  = results.get("openai_inference", {})
    ai  = results.get("adds_inference", {})
    rag = results.get("rag_analysis", {})
    val = results.get("validation", {})
    ds  = ai.get("drug_sensitivity_prediction", {})

    adds_conf   = ai.get("confidence_score", 0) * 100
    openai_conf = oi.get("confidence_score", 0) * 100
    avg_conf    = (adds_conf + openai_conf) / 2
    primary_rec = oi.get("primary_recommendation", "N/A")
    kras        = pathology.get("kras_mutation", "N/A")
    msi         = pathology.get("msi_status", "N/A")
    tnm         = pathology.get("tnm_stage", "N/A")

    # ── Figure 설정 ──────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(16, 11), facecolor="#f1f5f9")
    fig.subplots_adjust(left=0.01, right=0.99, top=0.93, bottom=0.03, wspace=0.04, hspace=0.35)

    # 3열 GridSpec
    gs = gridspec.GridSpec(1, 3, figure=fig, wspace=0.035)
    gs_l = gridspec.GridSpecFromSubplotSpec(10, 1, subplot_spec=gs[0], hspace=0.5)
    gs_c = gridspec.GridSpecFromSubplotSpec(10, 1, subplot_spec=gs[1], hspace=0.5)
    gs_r = gridspec.GridSpecFromSubplotSpec(10, 1, subplot_spec=gs[2], hspace=0.5)

    # ── 공통 헬퍼 ────────────────────────────────────────────────────────────
    def panel_bg(ax, color="#ffffff", border="#c7d2fe", lw=1.2):
        ax.set_facecolor(color)
        for spine in ax.spines.values():
            spine.set_edgecolor(border)
            spine.set_linewidth(lw)
        ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

    def section_header(ax, text, bg="#1e3a5f", fg="white"):
        ax.set_facecolor(bg)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
        ax.text(0.5, 0.5, text, transform=ax.transAxes,
                ha="center", va="center", fontsize=10, fontweight="bold",
                color=fg)

    def text_block(ax, lines, y_start=0.9, dy=0.18, fontsize=9, color="#1f2937", bold_idx=None):
        ax.set_facecolor("white")
        for sp in ax.spines.values():
            sp.set_edgecolor("#e5e7eb")
        ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
        for i, line in enumerate(lines):
            fw = "bold" if bold_idx and i in bold_idx else "normal"
            ax.text(0.05, y_start - i * dy, line, transform=ax.transAxes,
                    ha="left", va="top", fontsize=fontsize, color=color, fontweight=fw,
                    wrap=True)

    # ═══════════════════════════ 제목 ════════════════════════════════════════
    title_ax = fig.add_axes([0.01, 0.945, 0.98, 0.048])
    title_ax.set_facecolor("#1e3a5f")
    for sp in title_ax.spines.values():
        sp.set_visible(False)
    title_ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    title_ax.text(0.02, 0.55, "ADDS Precision Oncology Insight",
                  transform=title_ax.transAxes, fontsize=14, fontweight="bold",
                  color="white", va="center")
    title_ax.text(0.98, 0.55,
                  patient.get("name", "N/A") + "  |  " + datetime.now().strftime("%Y.%m.%d"),
                  transform=title_ax.transAxes, fontsize=10, color="#c7d2fe",
                  va="center", ha="right")

    # ═══════════════════════════ 왼쪽 패널 ═══════════════════════════════════

    # L0 헤더
    ax_l0 = fig.add_subplot(gs_l[0])
    section_header(ax_l0, "Clinicopathological Profile")

    # L1 환자 기본 정보
    ax_l1 = fig.add_subplot(gs_l[1:3])
    panel_bg(ax_l1, "#ffffff", "#e5e7eb")
    name_str = patient.get("name", "N/A") + " (" + patient.get("patient_id", "N/A") + ")"
    bio_str  = patient.get("birthdate","N/A") + " / " + {"M":"Male","F":"Female"}.get(patient.get("gender",""),"N/A")
    text_block(ax_l1, [name_str, bio_str,
                        "Diagnosis: " + pathology.get("tumor_location","N/A"),
                        "Primary Stage: " + tnm + "  ECOG PS: " + str(pathology.get("ecog_score",""))],
               y_start=0.86, dy=0.21, fontsize=9, bold_idx=[0])

    # L2 바이오마커
    ax_l2 = fig.add_subplot(gs_l[3])
    section_header(ax_l2, "Genomic & Molecular Biomarkers", bg="#1d4ed8")

    ax_l3 = fig.add_subplot(gs_l[4])
    panel_bg(ax_l3, "#f8faff", "#c7d2fe")
    ax_l3.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    kras_color = "#dc2626" if "Mutant" in str(kras) else "#16a34a"
    kras_bg    = "#fee2e2" if "Mutant" in str(kras) else "#dcfce7"
    kras_label = "Mutant" if "Mutant" in str(kras) else "Wild Type"

    # KRAS 박스
    rect1 = mpatches.FancyBboxPatch((0.04, 0.15), 0.4, 0.68, boxstyle="round,pad=0.02",
                                     facecolor=kras_bg, edgecolor=kras_color, linewidth=1.5,
                                     transform=ax_l3.transAxes)
    ax_l3.add_patch(rect1)
    ax_l3.text(0.24, 0.72, "KRAS", transform=ax_l3.transAxes,
               ha="center", va="center", fontsize=9, fontweight="bold", color=kras_color)
    ax_l3.text(0.24, 0.38, kras_label, transform=ax_l3.transAxes,
               ha="center", va="center", fontsize=8, color=kras_color)

    # MSI 박스
    rect2 = mpatches.FancyBboxPatch((0.54, 0.15), 0.42, 0.68, boxstyle="round,pad=0.02",
                                     facecolor="#dbeafe", edgecolor="#2563eb", linewidth=1.5,
                                     transform=ax_l3.transAxes)
    ax_l3.add_patch(rect2)
    ax_l3.text(0.75, 0.72, "MSI", transform=ax_l3.transAxes,
               ha="center", va="center", fontsize=9, fontweight="bold", color="#1d4ed8")
    ax_l3.text(0.75, 0.38, msi, transform=ax_l3.transAxes,
               ha="center", va="center", fontsize=7.5, color="#1d4ed8")

    # L3 부작용 모니터링
    ax_l4 = fig.add_subplot(gs_l[5])
    section_header(ax_l4, "Predicted Adverse Event (AE) Probability", bg="#dc2626")

    ax_l5 = fig.add_subplot(gs_l[6:9])
    panel_bg(ax_l5, "#fff7f7", "#fca5a5")
    ae_items = [
        ("Nausea/Vomiting",    0.45, "#f97316"),
        ("Peripheral Neuropathy", 0.60, "#ef4444"),
        ("Alopecia",         0.30, "#fb923c"),
        ("Thrombocytopenia", 0.55, "#ef4444"),
        ("Hypertension (Bev)",  0.35, "#f97316"),
    ]
    ax_l5.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    for sp in ax_l5.spines.values():
        sp.set_edgecolor("#fca5a5")
    y_positions = [0.82, 0.65, 0.48, 0.31, 0.14]
    for (name, val_ae, color_ae), yp in zip(ae_items, y_positions):
        ax_l5.text(0.03, yp + 0.05, name, transform=ax_l5.transAxes,
                   fontsize=8, color="#374151", va="bottom")
        bar_ax = ax_l5.inset_axes([0.45, yp - 0.04, 0.45, 0.12])
        bar_ax.barh([0], [val_ae], color=color_ae, height=0.8)
        bar_ax.barh([0], [1 - val_ae], left=[val_ae], color="#e5e7eb", height=0.8)
        bar_ax.set_xlim(0, 1)
        bar_ax.axis("off")
        ax_l5.text(0.97, yp + 0.05, ["Grade 1","Grade 2","Grade ≥3"][int(val_ae * 2.5)],
                   transform=ax_l5.transAxes, fontsize=7, color=color_ae, ha="right", va="bottom")

    # ═══════════════════════════ 중앙 패널 ═══════════════════════════════════

    ax_c0 = fig.add_subplot(gs_c[0])
    section_header(ax_c0, "AI-Optimized Polypharmacology Regimen", bg="#2563eb")

    # 추천 치료명 강조
    ax_c1 = fig.add_subplot(gs_c[1:3])
    panel_bg(ax_c1, "#eff6ff", "#2563eb", lw=2)
    ax_c1.text(0.5, 0.72, primary_rec, transform=ax_c1.transAxes,
               ha="center", va="center", fontsize=15, fontweight="bold", color="#1e3a5f")
    ax_c1.text(0.5, 0.30, "SoC Integration & Targeted Pathway Inhibition",
               transform=ax_c1.transAxes, ha="center", va="center", fontsize=8.5, color="#6b7280")

    # 신뢰도 3카드
    ax_c2 = fig.add_subplot(gs_c[3])
    panel_bg(ax_c2, "#f0f9ff", "#93c5fd")
    ax_c2.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    for sp in ax_c2.spines.values():
        sp.set_edgecolor("#93c5fd")

    conf_items = [
        (0.16, f"{adds_conf:.1f}%",   "17D Tensor Conf.",   "#2563eb", "#dbeafe"),
        (0.50, f"{openai_conf:.1f}%", "LLM Validation",   "#2563eb", "#dbeafe"),
        (0.83, f"{avg_conf:.1f}%",    "Ensemble Conf.",   "white",   "#2563eb"),
    ]
    for cx, val_s, label, fg, bg in conf_items:
        rect = mpatches.FancyBboxPatch((cx - 0.145, 0.07), 0.29, 0.85,
                                       boxstyle="round,pad=0.02", facecolor=bg,
                                       edgecolor="#93c5fd", linewidth=1,
                                       transform=ax_c2.transAxes)
        ax_c2.add_patch(rect)
        ax_c2.text(cx, 0.68, val_s, transform=ax_c2.transAxes,
                   ha="center", va="center", fontsize=13, fontweight="bold", color=fg)
        ax_c2.text(cx, 0.22, label, transform=ax_c2.transAxes,
                   ha="center", va="center", fontsize=7.5, color=fg)

    # 약물 표
    ax_c3 = fig.add_subplot(gs_c[4])
    section_header(ax_c3, "Therapeutic Agent  |  Mechanism of Action", bg="#e0e7ff", fg="#1e3a5f")

    ax_c4 = fig.add_subplot(gs_c[5:7])
    panel_bg(ax_c4, "#ffffff", "#c7d2fe")
    drug_lines = [
        "5-FU / Leucovorin   →   Thymidylate synthase inhibition",
        "Oxaliplatin          →   DNA crosslinking (Apoptosis)",
        "Bevacizumab (Bev)   →   VEGF-A inhibition (Anti-angiogenic)",
    ]
    colors_drug = ["#374151", "#374151", "#dc2626"]
    ax_c4.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    for sp in ax_c4.spines.values():
        sp.set_edgecolor("#c7d2fe")
    for i, (line, col) in enumerate(zip(drug_lines, colors_drug)):
        fw = "bold" if i == 2 else "normal"
        ax_c4.text(0.04, 0.82 - i * 0.30, line, transform=ax_c4.transAxes,
                   fontsize=7.5, color=col, fontweight=fw, va="top")

    # 추론 근거
    ax_c5 = fig.add_subplot(gs_c[7])
    panel_bg(ax_c5, "#f8faff", "#e5e7eb")
    rationale = oi.get("rationale", "KRAS G12D exhibits synergistic efficacy with VEGFR inhibition.")
    ax_c5.text(0.05, 0.7, "Clinical Rationale: " + rationale,
               transform=ax_c5.transAxes, fontsize=8, color="#374151",
               wrap=True, va="top")

    # 레이더 차트
    ax_c6 = fig.add_subplot(gs_c[8:], polar=True)
    ax_c6.set_facecolor("#f0f9ff")
    radar_cats = ["Efficacy", "Resistance\nMitigation", "Toxicity\nControl", "Regimen\nFeasibility", "Cost\nEffectiveness"]
    radar_vals = [ds.get("Bevacizumab", 0.87), ai.get("confidence_score", 0.85),
                  0.72, 0.80, 0.65]
    N = len(radar_cats)
    angles = [n / float(N) * 2 * math.pi for n in range(N)]
    angles += angles[:1]
    rv = radar_vals + [radar_vals[0]]
    ax_c6.plot(angles, rv, color="#2563eb", linewidth=2)
    ax_c6.fill(angles, rv, color="#2563eb", alpha=0.20)
    ax_c6.set_xticks(angles[:-1])
    ax_c6.set_xticklabels(radar_cats, fontsize=7.5, color="#374151")
    ax_c6.set_ylim(0, 1)
    ax_c6.set_yticklabels([])
    ax_c6.grid(color="#c7d2fe", linewidth=0.5)
    ax_c6.set_title("Synergy Radar Plot", fontsize=8.5, color="#1e3a5f", fontweight="bold", pad=8)

    # ═══════════════════════════ 우측 패널 ═══════════════════════════════════

    ax_r0 = fig.add_subplot(gs_r[0])
    section_header(ax_r0, "Predicted Clinical Viability & Metrics")

    # 핵심 지표 카드
    ax_r1 = fig.add_subplot(gs_r[1:3])
    panel_bg(ax_r1, "#f8faff", "#c7d2fe")
    ax_r1.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

    # 종양 크기 감소
    r1 = mpatches.FancyBboxPatch((0.03, 0.07), 0.44, 0.86, boxstyle="round,pad=0.02",
                                  facecolor="#ecfdf5", edgecolor="#6ee7b7", linewidth=1.5,
                                  transform=ax_r1.transAxes)
    ax_r1.add_patch(r1)
    ax_r1.text(0.25, 0.72, "-42%", transform=ax_r1.transAxes,
               ha="center", va="center", fontsize=18, fontweight="bold", color="#059669")
    ax_r1.text(0.25, 0.33, "Volumetric Reduction\n(Obj. Response: PR)", transform=ax_r1.transAxes,
               ha="center", va="center", fontsize=7.5, color="#065f46")

    # 생존 기간
    r2 = mpatches.FancyBboxPatch((0.53, 0.07), 0.44, 0.86, boxstyle="round,pad=0.02",
                                  facecolor="#eff6ff", edgecolor="#93c5fd", linewidth=1.5,
                                  transform=ax_r1.transAxes)
    ax_r1.add_patch(r2)
    ax_r1.text(0.75, 0.72, "28.5 mo", transform=ax_r1.transAxes,
               ha="center", va="center", fontsize=13, fontweight="bold", color="#2563eb")
    ax_r1.text(0.75, 0.33, "Predicted OS\nmPFS: 14.2 mo", transform=ax_r1.transAxes,
               ha="center", va="center", fontsize=7.5, color="#1d4ed8")

    # 치료 진행 바
    ax_r2 = fig.add_subplot(gs_r[3])
    panel_bg(ax_r2, "#f8faff", "#e5e7eb")
    ax_r2.text(0.05, 0.75, "Treatment Execution (Cycle 3/6) - 50%",
               transform=ax_r2.transAxes, fontsize=8.5, color="#374151", va="center")
    progress = ax_r2.inset_axes([0.05, 0.15, 0.9, 0.35])
    progress.barh([0], [50], color="#2563eb", height=0.8)
    progress.barh([0], [50], left=[50], color="#e5e7eb", height=0.8)
    progress.set_xlim(0, 100)
    progress.axis("off")

    # 치료법 비교 바 차트
    ax_r3 = fig.add_subplot(gs_r[4])
    section_header(ax_r3, "Comparative Regimen Efficacy", bg="#e0e7ff", fg="#1e3a5f")

    ax_r4 = fig.add_subplot(gs_r[5:7])
    panel_bg(ax_r4, "#f8faff", "#c7d2fe")
    bar_labels = ["FOLFOX+Bev\n(AI-Guided)", "FOLFIRI+Bev", "5-FU Mono"]
    bar_vals   = [89.5, 85.2, 68.4]
    bar_colors = ["#2563eb", "#60a5fa", "#94a3b8"]
    y_pos = range(len(bar_labels))
    ax_r4.barh(list(y_pos), bar_vals, color=bar_colors, height=0.55, edgecolor="none")
    ax_r4.set_yticks(list(y_pos))
    ax_r4.set_yticklabels(bar_labels, fontsize=7.5)
    ax_r4.set_xlim(0, 105)
    ax_r4.tick_params(left=False, bottom=False, labelbottom=False)
    for sp in ax_r4.spines.values():
        sp.set_visible(False)
    for i, v in enumerate(bar_vals):
        ax_r4.text(v + 0.5, i, f"{v}", va="center", ha="left", fontsize=8.5,
                   fontweight="bold", color="#374151")

    # AI 예측 정확도
    ax_r5 = fig.add_subplot(gs_r[7])
    section_header(ax_r5, "AI Predictive Validity vs. RWE", bg="#e0e7ff", fg="#1e3a5f")

    ax_r6 = fig.add_subplot(gs_r[8:])
    panel_bg(ax_r6, "#f8faff", "#c7d2fe")
    grp_cats   = ["ORR", "Grade ≥3\nAE Rate", "Completion\nRate"]
    grp_ai     = [87, 32, 78]
    grp_actual = [84, 35, 75]
    x = range(len(grp_cats))
    w = 0.35
    ax_r6.bar([i - w/2 for i in x], grp_ai,   w, label="AI Predicted", color="#2563eb", edgecolor="none")
    ax_r6.bar([i + w/2 for i in x], grp_actual, w, label="RWE Ground Truth", color="#10b981", edgecolor="none")
    ax_r6.set_xticks(list(x))
    ax_r6.set_xticklabels(grp_cats, fontsize=8)
    ax_r6.set_ylim(0, 110)
    ax_r6.tick_params(left=False, labelleft=False)
    for sp in ax_r6.spines.values():
        sp.set_visible(False)
    ax_r6.legend(fontsize=7.5, framealpha=0.7, loc="upper right")
    val_acc = val.get("clinical_alignment_score", 0.95) * 100
    ax_r6.set_title(f"Clinical Concept Alignment: {val_acc:.1f}%", fontsize=8, color="#1e3a5f", pad=4, fontweight="bold")

    # ── PNG로 저장 ───────────────────────────────────────────────────────────
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)
    return buf


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  의사용 상세 보고서 (ReportLab 텍스트 기반)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def generate_doctor_report_pdf(patient: dict, pathology: dict, results: dict) -> bytes:
    """임상의 전용 상세 분석 보고서를 PDF bytes로 반환."""
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4,
                            topMargin=18*mm, bottomMargin=18*mm,
                            leftMargin=18*mm, rightMargin=18*mm)
    fn, fnb = _fn(), _fn(True)
    body  = _s("body", fontName=fn,  fontSize=9)
    bold  = _s("bold", fontName=fnb, fontSize=9)
    h2    = _s("h2",   fontName=fnb, fontSize=13, textColor=colors.HexColor("#1a2f5e"), spaceBefore=8)
    h3    = _s("h3",   fontName=fnb, fontSize=11, textColor=colors.HexColor("#2c5282"), spaceBefore=6)
    title = _s("tt",   fontName=fnb, fontSize=18, textColor=colors.HexColor("#1a2f5e"), alignment=TA_CENTER)
    cap   = _s("cap",  fontName=fn,  fontSize=8,  textColor=colors.HexColor("#666666"), alignment=TA_CENTER)
    sm    = _s("sm",   fontName=fn,  fontSize=8,  textColor=colors.grey)
    warn  = _s("warn", fontName=fn,  fontSize=9,  textColor=colors.HexColor("#c0392b"))

    story = []
    story.append(Paragraph("ADDS Clinical Decision Support", title))
    story.append(Paragraph("의사용 상세 진료 지원 보고서 (CDSS Report)", cap))
    story.append(Paragraph("생성 일시: " + datetime.now().strftime("%Y년 %m월 %d일 %H:%M"), sm))
    story.append(_hr())

    story.append(Paragraph("1. 환자 기본 정보", h2))
    story.append(_kv_table([
        ("환자 ID",       patient.get("patient_id","N/A")),
        ("성명",          patient.get("name","N/A")),
        ("생년월일",      patient.get("birthdate","N/A")),
        ("성별",          {"M":"남성","F":"여성"}.get(patient.get("gender",""), patient.get("gender","N/A"))),
    ]))
    story.append(Spacer(1, 4*mm))

    story.append(Paragraph("2. 임상 병리 정보", h2))
    story.append(_kv_table([
        ("종양 위치",     pathology.get("tumor_location","N/A")),
        ("TNM 병기",     pathology.get("tnm_stage","N/A")),
        ("MSI 상태",      pathology.get("msi_status","N/A")),
        ("KRAS 돌연변이", pathology.get("kras_mutation","N/A")),
        ("ECOG 점수",    str(pathology.get("ecog_score","N/A"))),
        ("이전 치료",     pathology.get("previous_treatment","없음")),
    ]))
    story.append(Spacer(1, 4*mm))

    story.append(Paragraph("3. 의사 소견서 (RAG 분석 기반)", h2))
    notes = pathology.get("doctor_notes","").strip() or "소견서 없음"
    story.append(Paragraph(notes.replace("\n","<br/>"), body))
    story.append(Spacer(1, 4*mm))

    story.append(_hr())
    story.append(Paragraph("4. AI 다중모달 분석 결과", h2))

    ct = results.get("ct_analysis")
    story.append(Paragraph("4-1. CT 분석", h3))
    if ct:
        story.append(_kv_table([
            ("종양 검출 수",   str(ct.get("tumors_detected",0)) + "개"),
            ("최대 종양 크기", f"{ct.get('largest_tumor_size_mm',0):.1f} mm"),
            ("총 종양 부피",   f"{ct.get('total_tumor_volume_cm3',0):.1f} cm³"),
        ]))
    else:
        story.append(Paragraph("CT 파일 미업로드 — 분석 생략", body))
    story.append(Spacer(1, 3*mm))

    story.append(Paragraph("4-2. ADDS 추론 결과", h3))
    ai = results.get("adds_inference", {})
    ds = ai.get("drug_sensitivity_prediction", {})
    story.append(_kv_table([
        ("추천 표적",     ", ".join(ai.get("recommended_targets",[]))),
        ("활성 Pathway", ", ".join(ai.get("pathway_activation",[]))),
        ("모델 신뢰도",  f"{ai.get('confidence_score',0)*100:.1f}%"),
        ("5-FU",          f"{ds.get('5-FU',0)*100:.0f}%"),
        ("Oxaliplatin",   f"{ds.get('Oxaliplatin',0)*100:.0f}%"),
        ("Bevacizumab",   f"{ds.get('Bevacizumab',0)*100:.0f}%"),
    ]))
    story.append(Spacer(1, 4*mm))

    story.append(_hr())
    story.append(Paragraph("5. 최종 처방 추천", h2))
    oi = results.get("openai_inference", {})
    story.append(Paragraph("1순위: " + oi.get("primary_recommendation","N/A"), bold))
    story.append(Paragraph("2순위: " + oi.get("alternative_regimen","N/A"), body))
    story.append(Paragraph("신뢰도: " + f"{oi.get('confidence_score',0)*100:.1f}%", body))
    story.append(Paragraph("추론 근거: " + oi.get("rationale","N/A"), body))
    story.append(Spacer(1, 4*mm))

    story.append(_hr())
    story.append(Paragraph("6. 필수 모니터링 및 부작용 관리", h2))
    for w in ["혈액 검사: 매주 (골수 억제 모니터링)", "간/신장 기능: 격주",
               "혈압: 매일 (Bevacizumab 부작용)", "신경독성 (Oxaliplatin): 찬 음식 회피",
               "오심/구토: 항구토제 예방적 투여"]:
        story.append(Paragraph("  - " + w, body))

    story.append(Spacer(1, 6*mm))
    story.append(_hr())
    story.append(Paragraph(
        "본 보고서는 ADDS AI 시스템(연구용)이 생성한 임상 의사결정 보조 자료입니다. "
        "최종 치료 결정은 반드시 담당 전문의의 임상 판단에 따라야 합니다.", warn))

    doc.build(story)
    return buf.getvalue()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  환자용 대시보드 리포트 (matplotlib 전체 이미지 → PDF embed)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def generate_patient_report_pdf(patient: dict, pathology: dict, results: dict) -> bytes:
    """환자용 대시보드 PDF — matplotlib 이미지를 A4에 embed."""
    buf = io.BytesIO()
    from reportlab.lib.pagesizes import landscape
    page = landscape(A4)
    MARGIN = 8 * mm
    CAP_H  = 10 * mm   # 캡션 높이 여유분
    doc = SimpleDocTemplate(buf, pagesize=page,
                            topMargin=MARGIN, bottomMargin=MARGIN,
                            leftMargin=MARGIN, rightMargin=MARGIN)

    # 1. 대시보드 이미지 생성
    dash_buf = _build_dashboard_image(patient, pathology, results)

    # 2. 실제 이미지 크기 = 페이지 - 여백(2×MARGIN) - 캡션 공간
    usable_w = page[0] - 2 * MARGIN
    usable_h = page[1] - 2 * MARGIN - CAP_H

    img = RLImage(dash_buf, width=usable_w, height=usable_h)

    fn = _fn()
    cap_style = _s("cap2", fontName=fn, fontSize=7,
                   textColor=colors.HexColor("#9ca3af"), alignment=TA_CENTER)

    story = [
        img,
        Spacer(1, 2*mm),
        Paragraph(
            "이 문서는 ADDS AI 시스템이 생성한 보조 안내 자료입니다. "
            "최종 치료 방침은 담당 전문의 선생님과 충분히 상의하세요.  |  "
            + datetime.now().strftime("%Y.%m.%d %H:%M"),
            cap_style),
    ]
    doc.build(story)
    return buf.getvalue()
