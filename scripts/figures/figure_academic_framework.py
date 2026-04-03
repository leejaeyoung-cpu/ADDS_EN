"""
figure_academic_framework.py
------------------------------------------------
A high-end, academic-level research framework figure.
Utilizes soft drop shadows, clean sans-serif typography (Malgun Gothic for Korean), 
structured grid layout, and a harmonious color palette inspired by Nature/Science review articles.
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, PathPatch
import matplotlib.path as mpath

plt.rcParams.update({
    'font.family': 'Malgun Gothic',
    'axes.unicode_minus': False,
    'figure.facecolor': '#FFFFFF',
    'savefig.facecolor': '#FFFFFF',
})

# ---------------------------------------------------------
# 1. Color Palette (Modern, Clean, Academic)
# ---------------------------------------------------------
C = {
    'bg': '#FFFFFF',
    'text_main': '#1E293B',    # Slate 800
    'text_sub': '#475569',     # Slate 600
    'text_light': '#94A3B8',   # Slate 400
    
    # Team 1 (Basic/Preclinical) - Blue Theme
    't1_dark': '#1E40AF',      # Blue 800
    't1_base': '#2563EB',      # Blue 600
    't1_light': '#DBEAFE',     # Blue 100
    't1_bg': '#EFF6FF',        # Blue 50
    
    # Team 2 (Clinical/Pathology) - Emerald/Teal Theme
    't2_dark': '#065F46',      # Emerald 800
    't2_base': '#059669',      # Emerald 600
    't2_light': '#D1FAE5',     # Emerald 100
    't2_bg': '#ECFDF5',        # Emerald 50
    
    # ADDS Integration - Purple Theme
    't3_dark': '#5B21B6',      # Violet 800
    't3_base': '#7C3AED',      # Violet 600
    't3_light': '#EDE9FE',     # Violet 100
    't3_bg': '#F5F3FF',        # Violet 50

    # Accents & Structural
    'accent_gold': '#D97706',  # Amber 600
    'panel_border': '#CBD5E1', # Slate 300
    'shadow': '#0F172A',       # Slate 900
}

# ---------------------------------------------------------
# 2. Setup Figure
# ---------------------------------------------------------
fig, ax = plt.subplots(figsize=(20, 14.5), dpi=300)
ax.set_xlim(0, 20)
ax.set_ylim(0, 14.5)
ax.axis('off')

# ---------------------------------------------------------
# 3. Drawing Utilities
# ---------------------------------------------------------
def draw_box(x, y, w, h, bg_color, border_color, lw=1.5, radius=0.2, alpha=1.0, shadow=True):
    # Shadow
    if shadow:
        sx, sy = 0.08, -0.08
        shadow_box = FancyBboxPatch(
            (x + sx, y + sy), w, h, boxstyle=f'round,pad={radius}',
            facecolor=C['shadow'], edgecolor='none', alpha=0.08, zorder=1
        )
        ax.add_patch(shadow_box)
    
    # Main Box
    box = FancyBboxPatch(
        (x, y), w, h, boxstyle=f'round,pad={radius}',
        facecolor=bg_color, edgecolor=border_color, linewidth=lw, alpha=alpha, zorder=2
    )
    ax.add_patch(box)
    return box

def add_text(x, y, text, size=10, color=C['text_main'], weight='normal', ha='center', va='center', zorder=5):
    ax.text(x, y, text, fontsize=size, color=color, fontweight=weight, ha=ha, va=va, zorder=zorder, linespacing=1.4)

def draw_arrow(x1, y1, x2, y2, color=C['text_light'], lw=2, head_w=0.15, head_l=0.2):
    ax.annotate(
        '', xy=(x2, y2), xytext=(x1, y1),
        arrowprops=dict(facecolor=color, edgecolor=color, width=lw, headwidth=head_w*50, headlength=head_l*50, shrink=0),
        zorder=3
    )

def draw_bezier_arrow(x1, y1, x2, y2, color=C['text_light'], lw=2.5, style='->'):
    Path = mpath.Path
    # Controls for an S-curve depending on direction
    dx = x2 - x1
    dy = y2 - y1
    
    if abs(dx) > abs(dy):
        # Horizontal flow
        cx1, cy1 = x1 + dx*0.5, y1
        cx2, cy2 = x1 + dx*0.5, y2
    else:
        # Vertical flow
        cx1, cy1 = x1, y1 + dy*0.5
        cx2, cy2 = x2, y1 + dy*0.5

    pp = patches.PathPatch(
        Path([(x1, y1), (cx1, cy1), (cx2, cy2), (x2, y2)],
             [Path.MOVETO, Path.CURVE4, Path.CURVE4, Path.CURVE4]),
        fc="none", ec="none"
    )
    ax.add_patch(pp)
    
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle=f'{style},head_width=0.4,head_length=0.6', color=color,
                                lw=lw, patchB=pp, connectionstyle=f"path"),
                zorder=3)

# ---------------------------------------------------------
# SECTION 1: TIMELINE (12.5 to 14.0)
# ---------------------------------------------------------
add_text(1.0, 13.8, "A | 6-Year Project Master Timeline", size=14, weight='bold', ha='left', color=C['text_main'])

time_phases = [
    ("Phase I (Ys 1-3)", "Basic Mechanism Array", "Identify PrPc regulatory networks\nand drug resistance mechanisms", C['t1_base']),
    ("Phase II (Yr 4)", "Clinical Translation Prep", "Construct Patient-Derived Organoids (PDO)\nfrom primary/metastatic colorectal cancer", C['t2_base']),
    ("Phase III (Ys 5-6)", "ADDS Integration & Validation", "Synthesize multimodal data via ADDS\nto derive optimal combination therapy", C['t3_base'])
]

tx = 1.0
tw = 5.6
gap = 0.5
for (title, subtitle, desc, clr) in time_phases:
    draw_box(tx, 12.0, tw, 1.4, '#FFFFFF', clr, lw=2, radius=0.1, shadow=True)
    # Header ribbon
    draw_box(tx+0.02, 12.0+1.4-0.4, tw-0.04, 0.38, clr, clr, lw=0, radius=0.08, shadow=False)
    
    add_text(tx + tw/2, 13.2, title, size=11, weight='bold', color='#FFFFFF')
    add_text(tx + tw/2, 12.7, subtitle, size=10, weight='bold', color=clr)
    add_text(tx + tw/2, 12.35, desc, size=9, color=C['text_sub'])
    
    # Arrow to next
    if tx < 10:
        draw_arrow(tx + tw + 0.1, 12.7, tx + tw + gap - 0.1, 12.7, color=C['panel_border'], lw=3, head_w=0.15)
        
    tx += tw + gap

# ---------------------------------------------------------
# SECTION 2: RESEARCH TEAMS (7.0 to 11.5)
# ---------------------------------------------------------
add_text(1.0, 11.3, "B | Collaborative Research Framework & Role Allocation", size=14, weight='bold', ha='left', color=C['text_main'])

# Background panel
draw_box(0.8, 6.8, 18.4, 4.1, '#F8FAFC', C['panel_border'], lw=1, radius=0.2, shadow=False)

def draw_team(cx, cy, width, height, title, subtitle, roles, base_clr, bg_clr, light_clr, align='center'):
    draw_box(cx - width/2, cy - height/2, width, height, bg_clr, base_clr, lw=2, radius=0.15, shadow=True)
    
    add_text(cx, cy + height/2 - 0.4, title, size=13, weight='bold', color=base_clr)
    add_text(cx, cy + height/2 - 0.8, subtitle, size=10, color=C['text_sub'])
    
    # Separation line
    ax.plot([cx - width/2 + 0.5, cx + width/2 - 0.5], [cy + height/2 - 1.2, cy + height/2 - 1.2], color=light_clr, lw=1.5)
    
    ry = cy + height/2 - 1.7
    for (rtitle, rdesc) in roles:
        draw_box(cx - width/2 + 0.3, ry - 0.5, width - 0.6, 0.7, '#FFFFFF', light_clr, lw=1.5, radius=0.1, shadow=False)
        add_text(cx, ry - 0.15, rtitle, size=10, weight='bold', color=base_clr)
        add_text(cx, ry - 0.45, rdesc, size=9, color=C['text_main'])
        ry -= 0.85

t1_roles = [
    ("Functional Phenotyping", "Analyze PrPc signaling, EMT, and stemness capacity"),
    ("Drug Response Assays", "Evaluate synergistic toxicity and IC50 on CRC lines"),
    ("Preclinical Validation", "Validate ADDS outputs in in vivo & organoid models")
]

t2_roles = [
    ("Clinical Biospecimen Bank", "Secure endoscopic/surgical samples (primary & met)"),
    ("Pathology & Genomics", "IHC evaluation of PrPc & KRAS mutation typing (ddPCR)"),
    ("Clinical Informatics", "Patient stratification & clinical trajectory tracking")
]

draw_team(5.5, 8.85, 8.4, 3.6, "Inha Univ. Basic/Preclinical Team", "PI: Prof. Sang-Hoon Lee", t1_roles, C['t1_base'], C['t1_bg'], C['t1_light'])
draw_team(14.5, 8.85, 8.4, 3.6, "Inha Univ. Clinical/Pathology Team", "Co-PI: Prof. Moon-Seok Choi", t2_roles, C['t2_base'], C['t2_bg'], C['t2_light'])

# Central Collaboration Hub
draw_box(9.2, 7.8, 1.6, 2.1, '#FFFFFF', C['accent_gold'], lw=2, radius=0.2, shadow=True)
add_text(10.0, 9.5, "Data\nExchange", size=10, weight='bold', color=C['accent_gold'])
add_text(10.0, 8.6, "• Biospecimens\n• Pathology Scans\n• Assays Results\n• Patient Phenotypes", size=8, color=C['text_sub'])

# Arrows to hub
draw_arrow(9.0, 8.85, 9.2, 8.85, color=C['t1_base'], lw=2)
draw_arrow(11.0, 8.85, 10.8, 8.85, color=C['t2_base'], lw=2)


# ---------------------------------------------------------
# SECTION 3: ADDS INTEGRATION PLATFORM (1.0 to 6.3)
# ---------------------------------------------------------
add_text(1.0, 6.3, "C | ADDS (AI-Driven Drug Discovery System) Analytical Flow", size=14, weight='bold', ha='left', color=C['text_main'])

# Massive Integration Background
draw_box(0.8, 1.2, 18.4, 4.8, C['t3_bg'], C['t3_base'], lw=1.5, radius=0.2, shadow=False)

# Downward flow arrows from teams into ADDS
draw_arrow(5.5, 6.8, 5.5, 5.8, color=C['t1_base'], lw=3, head_w=0.15)
draw_arrow(14.5, 6.8, 14.5, 5.8, color=C['t2_base'], lw=3, head_w=0.15)
add_text(5.6, 6.3, "In Vitro & Preclinical Data Vectors", size=9, weight='bold', color=C['t1_base'], ha='left')
add_text(14.4, 6.3, "Clinical & Genomic Vectors", size=9, weight='bold', color=C['t2_base'], ha='right')

# ADDS CORE Box (Center)
cbx, cby, cbw, cbh = 7.5, 2.5, 5.0, 2.6
draw_box(cbx, cby, cbw, cbh, '#FFFFFF', C['t3_base'], lw=2.5, radius=0.2, shadow=True)

# Header inside Core
draw_box(cbx+0.05, cby+cbh-0.65, cbw-0.1, 0.6, C['t3_base'], C['t3_base'], lw=0, radius=0.15, shadow=False)
add_text(10.0, cby+cbh-0.3, "ADDS Core Processing Unit", size=13, weight='bold', color="#FFFFFF")

add_text(10.0, 4.1, "Multi-modal Data Fusion Protocol", size=10, weight='bold', color=C['t3_dark'])
add_text(10.0, 3.6, "1. Mapping: Energy Landscape Integration\n2. Validation: Machine-learning Survival Predictor\n3. Stratification: Synergy & Toxicity Balancing", size=9, color=C['text_sub'])

# 4 Surrounding Process Nodes
nodes = [
    (2.0, 3.8, "1. Mechanistic Base", "PrPc network dynamics\nand resistance metrics", C['t1_base'], C['t1_bg']),
    (2.0, 1.7, "4. PDO Validation", "Translational validation of\ncombination regimes", C['t3_base'], '#FFFFFF'),
    (14.5, 3.8, "2. Patient Cohorts", "KRAS mutation subtypes\ncorrelated with PrPc IHC", C['t2_base'], C['t2_bg']),
    (14.5, 1.7, "3. Optimization", "Derivation of optimal drug pair,\ndose, and schedule", C['accent_gold'], '#FFFBEB'),
]

for nx, ny, ntitle, ndesc, nclr, nbg in nodes:
    nw, nh = 3.5, 1.2
    draw_box(nx, ny, nw, nh, nbg, nclr, lw=2, radius=0.15, shadow=True)
    draw_box(nx, ny+nh-0.3, nw, 0.3, nclr, nclr, lw=0, radius=0.1, shadow=False)
    add_text(nx+nw/2, ny+nh-0.15, ntitle, size=10, weight='bold', color="#FFFFFF")
    add_text(nx+nw/2, ny+nh/2-0.2, ndesc, size=9, color=C['text_main'])

# Complex arrows connecting nodes to ADDS core
def draw_connector(x1, y1, x2, y2, color, style='->'):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle=f'{style},head_width=0.3,head_length=0.4', color=color, lw=2.5),
                zorder=3)

# Node 1 -> Core
draw_connector(5.5, 4.4, cbx-0.1, 4.4, color=C['t1_base'])
# Node 2 -> Core
draw_connector(14.5, 4.4, cbx+cbw+0.1, 4.4, color=C['t2_base'])
# Core -> Node 3
draw_connector(cbx+cbw+0.1, 2.3, 14.5, 2.3, color=C['t3_base'])
# Node 3 -> Node 4 (Loopback visual)
draw_connector(14.5, 1.7, 5.5, 1.7, color=C['accent_gold'])
# Node 4 -> Upward to signify outcome
draw_connector(3.75, 2.9, 3.75, 3.7, color=C['t3_base'])


# Bottom standardized phrasing spanning the full width
add_text(10.0, 0.5, "Standardized continuous collaborative loop: Mechanism Elucidation → Patient Stratification → ADDS Validation → Preclinical Confirmation", size=10, weight='bold', color=C['t3_dark'])

# ---------------------------------------------------------
# FINAL RENDER & SAVE
# ---------------------------------------------------------
plt.tight_layout()
out = r'f:\ADDS\docs\academic_research_framework.png'
plt.savefig(out, dpi=300, bbox_inches='tight')
plt.close()
print(f"High-quality academic framework saved to: {out}")
