"""
Pritamab Synergy Infographic
Signal Pathway visualization for 2-drug and 3-drug combinations
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Ellipse, Arc
from matplotlib.lines import Line2D
import numpy as np

# ─── Color palette ───────────────────────────────────────────────
BG_LIGHT      = "#F7F9FC"
PANEL_BG      = "#FFFFFF"
HEADER_BLUE   = "#1B4F8A"
HEADER_LIGHT  = "#EBF2FB"
SEC_HEADER    = "#2E7D32"
SEC_LIGHT     = "#E8F5E9"
RECEPTOR_TEAL = "#007B7F"
RECEPTOR_DARK = "#005C60"
LABEL_PURPLE  = "#6A1B9A"
LABEL_LIGHT   = "#E1BEE7"
ARROW_DARK    = "#37474F"
RED_X         = "#D32F2F"
GREEN_CHECK   = "#388E3C"
TEXT_DARK     = "#212121"
TEXT_MID      = "#455A64"
GOLD          = "#F9A825"
DRUG_COLORS   = {
    "Irinotecan":  "#1565C0",
    "Oxaliplatin": "#6A1B9A",
    "TAS-102":     "#00695C",
    "FOLFOX":      "#AD1457",
    "FOLFIRI":     "#E65100",
    "FOLFOXIRI":   "#4527A0",
}
APOPTOSIS_CLR = "#B71C1C"

# ─── Layout constants ────────────────────────────────────────────
FIG_W, FIG_H = 26, 20
TOP_Y     = 0.97   # figure top
MARGIN_X  = 0.025

# Section heights (normalized)
TITLE_H   = 0.055
SEC2_H    = 0.40   # 2-drug section
SEC3_H    = 0.40   # 3-drug section
GAP       = 0.025

SEC2_TOP  = TOP_Y - TITLE_H - 0.01
SEC3_TOP  = SEC2_TOP - SEC2_H - GAP

# Column widths for 4 panels in sec2 and 3 panels in sec3
def col_centers(n, left=MARGIN_X, right=1-MARGIN_X):
    spacing = (right - left) / n
    return [left + spacing*(i+0.5) for i in range(n)]

COL4 = col_centers(4)   # sec2: Alone, +Irinotecan, +Oxaliplatin, +TAS-102
COL3 = col_centers(3)   # sec3: +FOLFOX, +FOLFIRI, +FOLFOXIRI

PANEL_W2  = 0.20
PANEL_W3  = 0.27

# ─── Helper drawing functions ────────────────────────────────────

def add_rounded_box(ax, cx, cy, w, h, color=PANEL_BG, ec="#CBD5E0",
                    lw=1.5, alpha=1.0, zorder=1, radius=0.02, transform=None):
    if transform is None:
        transform = ax.transAxes
    box = FancyBboxPatch((cx - w/2, cy - h), w, h,
                         boxstyle=f"round,pad=0,rounding_size={radius}",
                         facecolor=color, edgecolor=ec, linewidth=lw,
                         alpha=alpha, zorder=zorder, transform=transform)
    ax.add_patch(box)


def draw_receptor(ax, cx, cy, scale=1.0, label="YIGSR"):
    """Draw a simplified transmembrane receptor icon (stylized)."""
    t = ax.transAxes

    # Membrane (horizontal band)
    mem_w = 0.065 * scale
    mem_h = 0.028 * scale
    mem = FancyBboxPatch((cx - mem_w/2, cy - mem_h/2), mem_w, mem_h,
                         boxstyle="round,pad=0.002",
                         facecolor=RECEPTOR_TEAL, edgecolor=RECEPTOR_DARK,
                         linewidth=1.5, zorder=5, transform=t)
    ax.add_patch(mem)

    # Extracellular domain (circle on top)
    ec_r = 0.022 * scale
    ec_circle = Circle((cx, cy + ec_r + mem_h/2 + 0.006*scale), ec_r,
                        facecolor=RECEPTOR_TEAL, edgecolor=RECEPTOR_DARK,
                        linewidth=1.5, zorder=6, transform=t)
    ax.add_patch(ec_circle)

    # Intracellular tail (line down)
    ic_len = 0.04 * scale
    ax.annotate("", xy=(cx, cy - mem_h/2 - ic_len), xytext=(cx, cy - mem_h/2),
                xycoords=t, textcoords=t,
                arrowprops=dict(arrowstyle="-", color=RECEPTOR_DARK, lw=2))

    # YIGSR tag (purple pill)
    tag_cy = cy + ec_r*2 + mem_h/2 + 0.014*scale
    tag = FancyBboxPatch((cx - 0.038*scale, tag_cy - 0.012*scale),
                         0.076*scale, 0.024*scale,
                         boxstyle="round,pad=0.003",
                         facecolor=LABEL_PURPLE, edgecolor="#4A148C",
                         linewidth=1, zorder=7, transform=t)
    ax.add_patch(tag)
    ax.text(cx, tag_cy, label, ha="center", va="center",
            fontsize=6.5*scale, fontweight="bold", color="white",
            zorder=8, transform=t)

    return cy - mem_h/2 - ic_len   # bottom y of receptor tail


def draw_red_x(ax, cx, cy, size=0.018, lw=2.8, zorder=10):
    t = ax.transAxes
    d = size * 0.7
    for dx, dy in [(-d, -d), (d, d)]:
        for dx2, dy2 in [(d, -d), (-d, d)]:
            pass
    ax.plot([cx-d, cx+d], [cy+d, cy-d], color=RED_X, lw=lw,
            solid_capstyle="round", zorder=zorder, transform=t)
    ax.plot([cx-d, cx+d], [cy-d, cy+d], color=RED_X, lw=lw,
            solid_capstyle="round", zorder=zorder, transform=t)


def draw_arrow_down(ax, cx, top_y, bot_y, color=ARROW_DARK, lw=1.8, zorder=4, style="->"):
    t = ax.transAxes
    ax.annotate("", xy=(cx, bot_y), xytext=(cx, top_y),
                xycoords=t, textcoords=t,
                arrowprops=dict(arrowstyle=style, color=color, lw=lw,
                                mutation_scale=16))


def draw_pathway_block(ax, cx, top_y, pathways, inhibited=True, color=TEXT_MID, fontsize=7):
    """Draw a stacked list of pathway names, with optional X."""
    t = ax.transAxes
    line_h = 0.022
    for i, pw in enumerate(pathways):
        y_pos = top_y - i * line_h
        ax.text(cx, y_pos, pw, ha="center", va="center",
                fontsize=fontsize, color=color, zorder=6, transform=t,
                fontweight="bold" if i == 0 else "normal")
    if inhibited:
        mid_y = top_y - (len(pathways)-1)*line_h/2
        # strikethrough
        text_w = 0.13
        ax.plot([cx - text_w/2, cx + text_w/2], [mid_y, mid_y],
                color=RED_X, lw=1.5, zorder=7, transform=t)
    return top_y - (len(pathways)-1)*line_h


def apoptosis_box(ax, cx, cy, pct, fold=None, drug_name=None,
                  energy_vals=None, panel_top=None):
    """Draw apoptosis result box with percentage and fold efficiency."""
    t = ax.transAxes
    bw, bh = 0.18, 0.085
    # shadow
    shadow = FancyBboxPatch((cx - bw/2 + 0.003, cy - bh - 0.003), bw, bh,
                            boxstyle="round,pad=0.005",
                            facecolor="#B0BEC5", edgecolor="none",
                            linewidth=0, zorder=3, transform=t)
    ax.add_patch(shadow)
    box = FancyBboxPatch((cx - bw/2, cy - bh), bw, bh,
                         boxstyle="round,pad=0.005",
                         facecolor=APOPTOSIS_CLR, edgecolor="#7F0000",
                         linewidth=1.5, zorder=4, transform=t)
    ax.add_patch(box)

    if fold:
        ax.text(cx, cy - 0.015, f"Apoptosis {fold}-fold", ha="center", va="center",
                fontsize=7.5, fontweight="bold", color="white", zorder=5, transform=t)
        ax.text(cx, cy - 0.038, "efficiency", ha="center", va="center",
                fontsize=7, color="#FFCDD2", zorder=5, transform=t)
    else:
        ax.text(cx, cy - 0.022, "Apoptosis", ha="center", va="center",
                fontsize=7.5, fontweight="bold", color="white", zorder=5, transform=t)

    ax.text(cx, cy - 0.060, f"{pct}%", ha="center", va="center",
            fontsize=13, fontweight="bold", color="white", zorder=5, transform=t,
            fontfamily="DejaVu Sans")

    # Energy values (small footnotes)
    if energy_vals:
        for k, (label, val) in enumerate(energy_vals.items()):
            ax.text(cx, cy - bh - 0.012 - k*0.016, f"= {val}  ({label})",
                    ha="center", va="top",
                    fontsize=5.5, color=TEXT_MID, zorder=5, transform=t)


def draw_drug_badge(ax, cx, cy, name, color):
    """Draw a colored drug name badge."""
    t = ax.transAxes
    bw = 0.14
    badge = FancyBboxPatch((cx - bw/2, cy - 0.016), bw, 0.030,
                           boxstyle="round,pad=0.004",
                           facecolor=color, edgecolor="white",
                           linewidth=1.2, zorder=6, transform=t)
    ax.add_patch(badge)
    ax.text(cx, cy - 0.001, name, ha="center", va="center",
            fontsize=7, fontweight="bold", color="white", zorder=7, transform=t)


# ─── Panel drawing ───────────────────────────────────────────────

def draw_single_panel(ax, cx, panel_top, panel_h, title, drugs,
                      pathways_blocked, erk_blocked, apoptosis_pct,
                      fold=None, energy_vals=None, label=None):
    """Draw a single combination panel."""
    t = ax.transAxes

    # Panel background
    bw = PANEL_W2 if len(drugs) <= 1 else PANEL_W3
    box = FancyBboxPatch((cx - bw/2, panel_top - panel_h), bw, panel_h,
                         boxstyle="round,pad=0.008",
                         facecolor=PANEL_BG, edgecolor="#B0BEC5",
                         linewidth=1.5, zorder=2, transform=t, alpha=0.95)
    ax.add_patch(box)

    y = panel_top - 0.018

    # Panel label (A, B, C...)
    if label:
        ax.text(cx - bw/2 + 0.012, panel_top - 0.008, label,
                ha="left", va="top",
                fontsize=9, fontweight="bold", color=HEADER_BLUE,
                zorder=8, transform=t)

    # Title
    ax.text(cx, y, title, ha="center", va="top",
            fontsize=8, fontweight="bold", color=TEXT_DARK,
            zorder=6, transform=t)
    y -= 0.032

    # Drug badges
    drug_list = [d for d in drugs if d]
    n_drugs = len(drug_list)
    if n_drugs == 1:
        draw_drug_badge(ax, cx, y, drug_list[0], DRUG_COLORS.get(drug_list[0], HEADER_BLUE))
        y -= 0.030
    elif n_drugs == 2:
        offset = 0.08
        for i, d in enumerate(drug_list):
            xx = cx - offset + i * 2*offset
            draw_drug_badge(ax, xx, y, d, DRUG_COLORS.get(d, HEADER_BLUE))
        y -= 0.030

    # Receptor icon
    rec_bottom = draw_receptor(ax, cx, y, scale=1.0)
    y = rec_bottom - 0.012

    # Pathways blocked
    pw_text = pathways_blocked
    block_top = y
    for i, pw in enumerate(pw_text):
        ax.text(cx, y, pw, ha="center", va="top",
                fontsize=6.5, color=TEXT_MID, zorder=6, transform=t,
                style="italic")
        y -= 0.020

    # Red X over pathways
    mid_pw = (block_top + y + 0.020) / 2
    draw_red_x(ax, cx - 0.06, mid_pw, size=0.016)

    # Arrow down
    y -= 0.005
    draw_arrow_down(ax, cx, y, y - 0.038, color=ARROW_DARK)
    y -= 0.038

    # ERK block
    ax.text(cx + 0.025, y, "ERK", ha="left", va="center",
            fontsize=7.5, fontweight="bold", color=TEXT_DARK,
            zorder=6, transform=t)
    draw_red_x(ax, cx - 0.015, y, size=0.016)
    y -= 0.032

    # Arrow to apoptosis
    draw_arrow_down(ax, cx, y, y - 0.030, color=APOPTOSIS_CLR)
    y -= 0.030

    # Apoptosis box
    apoptosis_box(ax, cx, y, apoptosis_pct, fold=fold, energy_vals=energy_vals)


# ─── Main figure ─────────────────────────────────────────────────

fig = plt.figure(figsize=(FIG_W, FIG_H), facecolor=BG_LIGHT)
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis("off")
t = ax.transAxes

# ══ MAIN TITLE ══════════════════════════════════════════════════
title_box = FancyBboxPatch((MARGIN_X, TOP_Y - TITLE_H), 1 - 2*MARGIN_X, TITLE_H,
                           boxstyle="round,pad=0.005",
                           facecolor=HEADER_BLUE, edgecolor="#0D2E5A",
                           linewidth=2, zorder=3, transform=t)
ax.add_patch(title_box)
ax.text(0.5, TOP_Y - TITLE_H/2, "Pritamab Synergy: Signal Pathway Inhibition by Drug Combination",
        ha="center", va="center",
        fontsize=16, fontweight="bold", color="white", zorder=4, transform=t)

# ══ SECTION 1: 2-DRUG COMBINATIONS ══════════════════════════════
sec2_top = SEC2_TOP
sec2_box = FancyBboxPatch((MARGIN_X, sec2_top - SEC2_H), 1 - 2*MARGIN_X, SEC2_H,
                          boxstyle="round,pad=0.005",
                          facecolor=HEADER_LIGHT, edgecolor="#90B8E0",
                          linewidth=1.5, zorder=1, transform=t, alpha=0.5)
ax.add_patch(sec2_box)

ax.text(MARGIN_X + 0.01, sec2_top - 0.015,
        "PART 1  ·  Single-Agent & 2-Drug Combinations  (Pritamab Alone / + 1 Chemotherapy)",
        ha="left", va="top",
        fontsize=10, fontweight="bold", color=HEADER_BLUE, zorder=4, transform=t)

# Panel A: Pritamab alone
draw_single_panel(
    ax, COL4[0], sec2_top - 0.05, SEC2_H - 0.06,
    title="Pritamab",
    drugs=[],
    pathways_blocked=["ERK", "PI3K-Akt", "activity"],
    erk_blocked=True,
    apoptosis_pct=55,
    fold=None,
    energy_vals={"G": "−10 kcal/mol", "inh": "= 13.0 kcal/mol"},
    label="A"
)

# Panel B: Pritamab + Irinotecan
draw_single_panel(
    ax, COL4[1], sec2_top - 0.05, SEC2_H - 0.06,
    title="Pritamab + Irinotecan",
    drugs=["Irinotecan"],
    pathways_blocked=["YIGSR", "PI3K-Akt", "ERK activity", "Invasion"],
    erk_blocked=True,
    apoptosis_pct=75,
    fold="3.0-fold",
    energy_vals={"":  "= 13.0 kcal/mol", " ": "− 13.5 kcal/mol"},
    label="B"
)

# Panel C: Pritamab + Oxaliplatin
draw_single_panel(
    ax, COL4[2], sec2_top - 0.05, SEC2_H - 0.06,
    title="Pritamab + Oxaliplatin",
    drugs=["Oxaliplatin"],
    pathways_blocked=["YIGSR", "ERK", "Invasion"],
    erk_blocked=True,
    apoptosis_pct=75,
    fold="3.0-fold",
    energy_vals={"": "− 14.0 kcal/mol", " ": "= 18.0 kcal/mol"},
    label="C"
)

# Panel D: Pritamab + TAS-102
draw_single_panel(
    ax, COL4[3], sec2_top - 0.05, SEC2_H - 0.06,
    title="Pritamab + TAS-102",
    drugs=["TAS-102"],
    pathways_blocked=["YIGSR", "ERK-Akt", "activity", "Invasion"],
    erk_blocked=True,
    apoptosis_pct=80,
    fold="3.3-fold",
    energy_vals={"": "− 14.3 kcal/mol", " ": "= 14.3 kcal/mol"},
    label="D"
)

# ══ SECTION 2: 3-DRUG COMBINATIONS ══════════════════════════════
sec3_top = SEC3_TOP
sec3_box = FancyBboxPatch((MARGIN_X, sec3_top - SEC3_H), 1 - 2*MARGIN_X, SEC3_H,
                          boxstyle="round,pad=0.005",
                          facecolor=SEC_LIGHT, edgecolor="#A5D6A7",
                          linewidth=1.5, zorder=1, transform=t, alpha=0.5)
ax.add_patch(sec3_box)

ax.text(MARGIN_X + 0.01, sec3_top - 0.015,
        "PART 2  ·  3-Drug Combinations  (Pritamab + 2 Chemotherapy Agents)",
        ha="left", va="top",
        fontsize=10, fontweight="bold", color=SEC_HEADER, zorder=4, transform=t)

# Panel E: Pritamab + FOLFOX
draw_single_panel(
    ax, COL3[0], sec3_top - 0.05, SEC3_H - 0.06,
    title="Pritamab + FOLFOX",
    drugs=["Oxaliplatin", "FOLFOX"],
    pathways_blocked=["YIGSR", "ERK", "PI3K-Akt", "Invasion", "DNA repair"],
    erk_blocked=True,
    apoptosis_pct=85,
    fold="3.8-fold",
    energy_vals={"Ox": "− 14.0 kcal/mol", "5FU": "− 11.2 kcal/mol"},
    label="E"
)

# Panel F: Pritamab + FOLFIRI
draw_single_panel(
    ax, COL3[1], sec3_top - 0.05, SEC3_H - 0.06,
    title="Pritamab + FOLFIRI",
    drugs=["Irinotecan", "FOLFIRI"],
    pathways_blocked=["YIGSR", "ERK", "PI3K-Akt", "Topo-I inhibition", "Invasion"],
    erk_blocked=True,
    apoptosis_pct=82,
    fold="3.6-fold",
    energy_vals={"Iri": "− 13.5 kcal/mol", "5FU": "− 11.2 kcal/mol"},
    label="F"
)

# Panel G: Pritamab + FOLFOXIRI
draw_single_panel(
    ax, COL3[2], sec3_top - 0.05, SEC3_H - 0.06,
    title="Pritamab + FOLFOXIRI",
    drugs=["Oxaliplatin", "FOLFOXIRI"],
    pathways_blocked=["YIGSR", "ERK", "PI3K-Akt", "Topo-I inh.", "DNA repair", "Invasion"],
    erk_blocked=True,
    apoptosis_pct=88,
    fold="4.0-fold",
    energy_vals={"Ox": "− 14.0 kcal/mol", "Iri": "− 13.5 kcal/mol"},
    label="G"
)

# ══ LEGEND ══════════════════════════════════════════════════════
leg_y = sec3_top - SEC3_H - 0.005
legend_items = [
    (RECEPTOR_TEAL, "LRP6 Receptor (YIGSR site)"),
    (LABEL_PURPLE,  "Pritamab / YIGSR binding"),
    (RED_X,         "Pathway inhibition (×)"),
    (APOPTOSIS_CLR, "Apoptosis induction"),
    (DRUG_COLORS["Irinotecan"],  "Irinotecan"),
    (DRUG_COLORS["Oxaliplatin"], "Oxaliplatin"),
    (DRUG_COLORS["TAS-102"],     "TAS-102"),
    (DRUG_COLORS["FOLFOX"],      "FOLFOX"),
    (DRUG_COLORS["FOLFIRI"],     "FOLFIRI"),
    (DRUG_COLORS["FOLFOXIRI"],   "FOLFOXIRI"),
]
legend_patches = [mpatches.Patch(color=c, label=l) for c, l in legend_items]
ax.legend(handles=legend_patches, loc="lower center",
          bbox_to_anchor=(0.5, leg_y),
          ncol=5, fontsize=7.5, frameon=True,
          facecolor="white", edgecolor="#B0BEC5",
          bbox_transform=t)

# ══ FOOTNOTE ════════════════════════════════════════════════════
ax.text(0.5, 0.005,
        "* Fold-efficiency relative to Pritamab monotherapy (Apoptosis 55%).  "
        "Energy values from molecular docking simulations.  "
        "3-drug combination data are model-projected estimates based on published synergy scores.",
        ha="center", va="bottom",
        fontsize=6, color=TEXT_MID, zorder=6, transform=t, style="italic")

plt.savefig(r"f:\ADDS\figures\pritamab_synergy_infographic.png",
            dpi=200, bbox_inches="tight",
            facecolor=BG_LIGHT)
print("Saved: f:/ADDS/figures/pritamab_synergy_infographic.png")
plt.close()
