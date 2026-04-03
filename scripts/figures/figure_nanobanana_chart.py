"""
figure_nanobanana_chart.py
------------------------------------------------
A high-fidelity, "NanoBanana 2.0" style research proposal chart.
Focuses on clear academic content, rigorous structural layout, smooth bezier connections,
and a sophisticated vector-like aesthetic suitable for scientific grant proposals.
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, PathPatch
import matplotlib.path as mpath
import numpy as np

# ---------------------------------------------------------
# 1. Global Setup
# ---------------------------------------------------------
plt.rcParams.update({
    'font.family': 'Malgun Gothic',
    'axes.unicode_minus': False,
    'figure.facecolor': '#F8FAFC',
    'savefig.facecolor': '#F8FAFC',
})

fig, ax = plt.subplots(figsize=(24, 16), dpi=300)
ax.set_xlim(0, 24)
ax.set_ylim(0, 16)
ax.axis('off')

# ---------------------------------------------------------
# 2. Color Palette (NanoBanana 2.0 Deep Aesthetic)
# ---------------------------------------------------------
C = {
    'bg': '#F8FAFC',
    'panel_bg': '#FFFFFF',
    'text_main': '#0F172A',
    'text_sub': '#475569',
    'text_light': '#94A3B8',
    'border': '#E2E8F0',
    
    # Left: Basic/Preclinical Theme
    't1_main': '#2563EB',    # Blue 600
    't1_dark': '#1E3A8A',    # Blue 900
    't1_light': '#DBEAFE',   # Blue 100
    't1_line': '#93C5FD',    # Blue 300

    # Right: Clinical/Pathology Theme
    't2_main': '#059669',    # Emerald 600
    't2_dark': '#064E3B',    # Emerald 900
    't2_light': '#D1FAE5',   # Emerald 100
    't2_line': '#6EE7B7',    # Emerald 300

    # Center: ADDS Theme
    't3_main': '#7C3AED',    # Violet 600
    't3_dark': '#4C1D95',    # Violet 900
    't3_light': '#EDE9FE',   # Violet 100
    't3_line': '#C4B5FD',    # Violet 300
    
    't3_accent': '#D946EF',  # Fuchsia 500
}

# ---------------------------------------------------------
# 3. Helper Drawing Functions
# ---------------------------------------------------------
def draw_rounded_box(x, y, w, h, bg_color, border_color, text_main="", text_sub="", lw=1.5, radius=0.15, shadow=True, title_color='#FFFFFF', title_bg=None):
    # Perfect soft shadow
    if shadow:
        for i, alpha in enumerate(np.linspace(0.02, 0.0, 5)):
            offset = 0.02 * (i + 1)
            sb = FancyBboxPatch((x + offset, y - offset), w, h, boxstyle=f'round,pad={radius}', facecolor='#000000', edgecolor='none', alpha=alpha, zorder=1)
            ax.add_patch(sb)
    
    # Main Box
    box = FancyBboxPatch((x, y), w, h, boxstyle=f'round,pad={radius}', facecolor=bg_color, edgecolor=border_color, linewidth=lw, zorder=2)
    ax.add_patch(box)
    
    # Optional Title Ribbon
    if title_bg:
        tb = FancyBboxPatch((x, y + h - 0.5), w, 0.5, boxstyle=f'round,pad={radius}', facecolor=title_bg, edgecolor='none', zorder=3)
        # Fix rounded corners at bottom by drawing a square over it
        sq = patches.Rectangle((x-radius, y + h - 0.5), w+radius*2, radius, facecolor=title_bg, edgecolor='none', zorder=3)
        ax.add_patch(tb)
        ax.add_patch(sq)
        ax.text(x + w/2, y + h - 0.25, text_main, fontsize=12, fontweight='bold', color=title_color, ha='center', va='center', zorder=4)
        if text_sub:
            ax.text(x + w/2, y + h/2 - 0.25, text_sub, fontsize=10, color=C['text_main'], ha='center', va='center', zorder=4, linespacing=1.6)
    else:
        if text_main:
            ax.text(x + w/2, y + h/2 + 0.1, text_main, fontsize=12, fontweight='bold', color=C['text_main'], ha='center', va='center', zorder=4)
        if text_sub:
            ax.text(x + w/2, y + h/2 - 0.3, text_sub, fontsize=10, color=C['text_sub'], ha='center', va='center', zorder=4, linespacing=1.5)

def draw_bezier(x1, y1, x2, y2, color, lw=4, alpha=0.6, curvature=0.5):
    Path = mpath.Path
    # Decide direction based on delta
    dx = x2 - x1
    dy = y2 - y1
    
    if abs(dx) > abs(dy):
        # Horizontal S-curve 
        cx1, cy1 = x1 + dx*curvature, y1
        cx2, cy2 = x1 + dx*(1-curvature), y2
    else:
        # Vertical S-curve
        cx1, cy1 = x1, y1 + dy*curvature
        cx2, cy2 = x2, y1 + dy*(1-curvature)

    pp = patches.PathPatch(
        Path([(x1, y1), (cx1, cy1), (cx2, cy2), (x2, y2)], [Path.MOVETO, Path.CURVE4, Path.CURVE4, Path.CURVE4]),
        fc="none", ec=color, lw=lw, alpha=alpha, zorder=1.5
    )
    ax.add_patch(pp)
    # Add arrowhead
    ax.annotate('', xy=(x2, y2), xytext=(cx2, cy2),
                arrowprops=dict(arrowstyle="-|>,head_width=0.6,head_length=0.8", color=color, alpha=alpha), zorder=1.5)


# ---------------------------------------------------------
# 4. Global Title & Header
# ---------------------------------------------------------
ax.text(12, 15.2, "Joint Research Framework: KRAS-Mutant CRC Precision Therapy", fontsize=22, fontweight='heavy', color=C['t3_dark'], ha='center')
ax.text(12, 14.65, "Cellular Prion Protein (PrPc) Target Identification & ADDS-Driven Optimal Combination Strategy", fontsize=14, color=C['text_sub'], ha='center')

# Header Separation Line
ax.plot([2, 22], [14.2, 14.2], color=C['border'], lw=2)

# ---------------------------------------------------------
# 5. Timeline Top Axis (Chevron styled blocks)
# ---------------------------------------------------------
timeline_y = 12.8
def draw_time_block(x, w, label, desc, color):
    draw_rounded_box(x, timeline_y, w, 0.9, color, color, shadow=False, radius=0.1)
    ax.text(x + w/2, timeline_y + 0.6, label, color='#FFFFFF', fontsize=12, fontweight='bold', ha='center', va='center', zorder=4)
    ax.text(x + w/2, timeline_y + 0.25, desc, color='#FFFFFF', fontsize=9, alpha=0.9, ha='center', va='center', zorder=4)
    # Chevron notch
    if x + w < 20:
        poly = patches.Polygon([[x+w, timeline_y], [x+w+0.3, timeline_y+0.45], [x+w, timeline_y+0.9]], facecolor='#F8FAFC', zorder=3)
        ax.add_patch(poly)

draw_time_block(1.5, 6, "Phase 1: Basic & Preclinical (Ys 1-3)", "PrPc Mechanism & Resistance Pathways", C['t1_main'])
draw_time_block(7.8, 8.4, "Phase 2: Data Integration & Platform Base (Yr 4)", "ADDS Modelling & Patient Organoid Construction", C['t3_main'])
draw_time_block(16.5, 6, "Phase 3: Validation & Translation (Ys 5-6)", "PDO Validation & Combination Optimization", C['t2_main'])

# ---------------------------------------------------------
# 6. Pilar 1: Inha Basic/Preclinical Team (Left)
# ---------------------------------------------------------
x_left = 1.5
w_col = 6.0
y_start = 11.2

# Team Header
draw_rounded_box(x_left, y_start, w_col, 0.8, C['t1_light'], C['t1_main'], "PI: Prof. Sang-Hoon Lee", "Inha Univ. Basic/Preclinical Dept.", lw=2, title_bg=C['t1_main'], shadow=False)

# Modules 
modules_left = [
    ("In-Vitro PrPc Modulation", "SiRNA silencing, Overexpression\nColony Formation, Wound Healing"),
    ("Stemness & EMT Phenotyping", "CD44, LGR5, vimentin, Snail signaling\nAssays: Sphere formation, Matrigel invasion"),
    ("Therapeutic Resistance", "5-FU/oxaliplatin repeated exposure\nAUC, IC50 kinetics & molecular response")
]

cy = y_start - 1.8
for title, desc in modules_left:
    draw_rounded_box(x_left, cy, w_col, 1.4, '#FFFFFF', C['border'], text_main=title, text_sub=desc, lw=1.5, radius=0.15)
    cy -= 1.8

# Downward flow indicator
draw_bezier(x_left+w_col/2, y_start-0.1, x_left+w_col/2, y_start-0.4, C['t1_line'], lw=3, alpha=1)

# ---------------------------------------------------------
# 7. Pilar 2: Inha Clinical/Pathology Team (Right)
# ---------------------------------------------------------
x_right = 16.5
# Team Header
draw_rounded_box(x_right, y_start, w_col, 0.8, C['t2_light'], C['t2_main'], "Co-PI: Prof. Moon-Seok Choi", "Inha Univ. Hospital Surgical Dept.", lw=2, title_bg=C['t2_main'], shadow=False)

modules_right = [
    ("Patient Biospecimen Bank", "Acquisition of primary & met tissue\nEndoscopic biopsy & surgical resections"),
    ("Genomic & Pathologic Profiling", "ddPCR KRAS mutation typing\nQuantitative PrPc IHC evaluation"),
    ("Patient-Derived Organoids", "Establishment of PDO models\nMolecular congruency & propagation")
]

cy = y_start - 1.8
for title, desc in modules_right:
    draw_rounded_box(x_right, cy, w_col, 1.4, '#FFFFFF', C['border'], text_main=title, text_sub=desc, lw=1.5, radius=0.15)
    cy -= 1.8

draw_bezier(x_right+w_col/2, y_start-0.1, x_right+w_col/2, y_start-0.4, C['t2_line'], lw=3, alpha=1)


# ---------------------------------------------------------
# 8. Central Core: ADDS Platform Integration (Center to Bottom)
# ---------------------------------------------------------
# Central massive base
cx = 8.0
cw = 8.0
ch = 9.0
b_y = 1.6

# ADDS Data Fusion Core
draw_rounded_box(cx, b_y+4.2, cw, 5.0, '#FFFFFF', C['t3_main'], lw=2.5, radius=0.25, shadow=True)
# Core Title
draw_rounded_box(cx+0.1, b_y+4.2+4.0, cw-0.2, 0.9, C['t3_dark'], 'none', shadow=False, radius=0.15)
ax.text(cx + cw/2, b_y+4.2+4.5, "ADDS Platform: AI-Driven Drug Discovery System", color='#FFFFFF', fontsize=14, fontweight='bold', ha='center', va='center', zorder=4)

# Flow lines from Left/Right pillars into ADDS Core
# Connect the lowest nodes of Team 1 & 2 into ADDS
left_out_x = x_left + w_col
left_out_y = y_start - 3.6 + 0.7  # middle of middle node
draw_bezier(left_out_x, left_out_y, cx, b_y+8.0, C['t1_main'], lw=5, alpha=0.5, curvature=0.7)
draw_bezier(left_out_x, left_out_y-1.8, cx, b_y+7.0, C['t1_main'], lw=5, alpha=0.5, curvature=0.7)

right_out_x = x_right
right_out_y = y_start - 3.6 + 0.7
draw_bezier(right_out_x, right_out_y, cx+cw, b_y+8.0, C['t2_main'], lw=5, alpha=0.5, curvature=0.7)
draw_bezier(right_out_x, right_out_y-1.8, cx+cw, b_y+7.0, C['t2_main'], lw=5, alpha=0.5, curvature=0.7)

# Inside ADDS Modules
cy = b_y + 4.2 + 2.5
adds_modules = [
    ("Multi-modal Data Harmonization", "Standardizing In-vitro assay metrics with Pathology IHC scores", C['t3_main']),
    ("Energy Landscape Computation", "Mapping PrPc dynamics & KRAS mutation to activation energy (ΔG)", C['t3_main']),
    ("Therapeutic Synergy Prediction", "Prioritizing optimal drug pairs, dosage scaling, and treatment schedules", C['t3_accent'])
]

for title, desc, clr in adds_modules:
    draw_rounded_box(cx+0.5, cy, cw-1.0, 1.2, C['bg'], clr, text_main=title, text_sub=desc, lw=1.5, radius=0.15, shadow=False)
    # Downward connecting arrow between internal modules
    if cy > b_y+4.2+0.5:
        ax.annotate('', xy=(cx+cw/2, cy-0.1), xytext=(cx+cw/2, cy+0.2), arrowprops=dict(arrowstyle="->", color=C['t3_line'], lw=2), zorder=3)
    cy -= 1.6

# ---------------------------------------------------------
# 9. Output Flow & Preclinical Translation (Bottom Area)
# ---------------------------------------------------------
output_bx = cx
output_by = b_y
output_bw = cw
output_bh = 2.4

# Connection arrow from ADDS Core down to Translation Phase
ax.annotate('', xy=(cx+cw/2, output_by+output_bh), xytext=(cx+cw/2, b_y+4.2), arrowprops=dict(arrowstyle="-|>,head_width=0.8,head_length=1.2", color=C['t3_dark'], lw=4), zorder=3)

# Translation Box
draw_rounded_box(output_bx, output_by, output_bw, output_bh, C['t3_dark'], 'none', lw=0, radius=0.2, shadow=True)
ax.text(output_bx + output_bw/2, output_by + output_bh - 0.5, "Preclinical Validation & Translation", color='#D1FAE5', fontsize=13, fontweight='bold', ha='center', zorder=4)

ax.text(output_bx + output_bw/2, output_by + 0.8, "Target Validation via Patient Derived Organoids (PDOs)\nIn-Vivo Xenograft Survival & Toxicity Assessment\nFinal Stratification Criteria for KRAS Subtypes", color='#F8FAFC', fontsize=11, ha='center', linespacing=1.6, zorder=4)

# Return loop to clinical team for data feedback
draw_bezier(output_bx+output_bw, output_by+output_bh/2, x_right+w_col/2, y_start-5.4, C['t3_accent'], lw=3, alpha=0.8, curvature=0.9)
ax.text(output_bx+output_bw+1.5, output_by+output_bh/2+1.0, "Clinical Feedback Loop", color=C['t3_accent'], fontsize=10, fontweight='bold')


# ---------------------------------------------------------
# FINAL RENDER & SAVE
# ---------------------------------------------------------
plt.tight_layout()
out = r'f:\ADDS\docs\academic_nanobanana_chart.png'
plt.savefig(out, dpi=300, bbox_inches='tight')
plt.close()
print(f"Chart saved to: {out}")
