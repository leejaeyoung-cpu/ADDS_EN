"""
Pritamab AI-Driven Combination Therapy Pipeline — PROFESSIONAL INFOGRAPHIC
Photoshop-quality design using Pillow (PIL)
High-resolution: 3600 x 4800 px @ 300 DPI
"""
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import math, os, sys

# ── Canvas ─────────────────────────────────────────────────────────────────────
W, H = 3600, 4800
canvas = Image.new('RGBA', (W, H), (255, 255, 255, 255))
draw   = ImageDraw.Draw(canvas)

# ── Color Palette ──────────────────────────────────────────────────────────────
BG_GRAD_TOP   = (236, 245, 255)
BG_GRAD_BOT   = (220, 235, 252)

TIER1_BG  = (214, 234, 248, 230)
TIER2_BG  = (212, 239, 223, 230)
TIER3_BG  = (253, 243, 226, 230)
INFRA_BG  = (236, 239, 241, 230)

# Source block colors
COL_GENOMIC = (103, 58, 183)
COL_CELL    = (0,  150, 136)
COL_IMAGING = (229, 57,  53)
COL_CLINIC  = (25, 118, 210)

# AI module colors
COL_AI1 = (92,  107, 192)
COL_AI2 = (0,   150, 136)
COL_AI3 = (229,  57,  53)
COL_AI4 = (244, 81,  30)
COL_AI5 = (3,  169, 244)
COL_AI6 = (0,  137, 123)

COL_PIPE  = (2, 136, 209)
COL_ENS   = (56, 142, 60)

COL_PRIM  = (21,  101, 192)
COL_ALT   = (46,  125, 50)
COL_COND  = (106, 27,  154)
COL_GOLD  = (245, 127, 23)

WHITE     = (255, 255, 255)
DARK      = (26,  35,  126)
TEXT_D    = (33,  33,  33)
TEXT_M    = (66,  66,  66)
TEXT_L    = (117, 117, 117)
ARROW_C   = (84,  110, 122)
SHADOW_C  = (0, 0, 0, 60)

# ── Font Loading ───────────────────────────────────────────────────────────────
def load_font(size, bold=False):
    """Try to load a nice font, fall back to default."""
    candidates_bold = [
        "C:/Windows/Fonts/arialbd.ttf",
        "C:/Windows/Fonts/calibrib.ttf",
        "C:/Windows/Fonts/segoeuib.ttf",
    ]
    candidates_reg = [
        "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/calibri.ttf",
        "C:/Windows/Fonts/segoeui.ttf",
    ]
    candidates = candidates_bold if bold else candidates_reg
    for path in candidates:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size)
            except Exception:
                continue
    return ImageFont.load_default()

F_TITLE   = load_font(78, bold=True)
F_TITLE2  = load_font(52)
F_TIER    = load_font(52, bold=True)
F_TIER_LBL= load_font(62, bold=True)
F_HEAD    = load_font(46, bold=True)
F_SUBHEAD = load_font(38, bold=True)
F_BODY    = load_font(34)
F_BODY_B  = load_font(34, bold=True)
F_SMALL   = load_font(30)
F_BADGE   = load_font(36, bold=True)
F_DRS     = load_font(56, bold=True)
F_CAPTION = load_font(28)

# ── Helpers ────────────────────────────────────────────────────────────────────
def rgba(rgb, a=255):
    return rgb + (a,) if len(rgb) == 3 else rgb

def lerp_color(c1, c2, t):
    return tuple(int(c1[i] + (c2[i] - c1[i]) * t) for i in range(3))

def draw_gradient_rect(img, x0, y0, x1, y1, color_top, color_bot, alpha=255):
    """Draw vertical gradient rectangle."""
    band = img.crop((x0, y0, x1, y1))
    grad = Image.new('RGBA', (x1 - x0, y1 - y0))
    gd   = ImageDraw.Draw(grad)
    for row in range(y1 - y0):
        t = row / max(1, y1 - y0 - 1)
        c = lerp_color(color_top, color_bot, t)
        gd.line([(0, row), (x1 - x0, row)], fill=c + (alpha,))
    img.paste(grad, (x0, y0), grad)

def draw_rounded_rect(d, x0, y0, x1, y1, r, fill, outline=None, outline_width=4):
    """Draw filled rounded rectangle with optional outline."""
    r = min(r, (x1-x0)//2, (y1-y0)//2)
    d.rounded_rectangle([x0, y0, x1, y1], radius=r, fill=fill,
                         outline=outline, width=outline_width)

def draw_shadow_box(img, x0, y0, x1, y1, r, fill, outline, outline_w=6, shadow_offset=12):
    """Box with drop shadow."""
    shadow_layer = Image.new('RGBA', img.size, (0, 0, 0, 0))
    sd = ImageDraw.Draw(shadow_layer)
    sd.rounded_rectangle([x0+shadow_offset, y0+shadow_offset,
                           x1+shadow_offset, y1+shadow_offset],
                          radius=r, fill=(0, 0, 0, 55))
    blurred = shadow_layer.filter(ImageFilter.GaussianBlur(radius=shadow_offset*0.8))
    img.paste(blurred, (0, 0), blurred)
    d = ImageDraw.Draw(img)
    draw_rounded_rect(d, x0, y0, x1, y1, r, fill=fill,
                      outline=outline, outline_width=outline_w)

def draw_text_centered(d, cx, cy, text, font, color, line_height=None):
    """Draw centered (possibly multiline) text."""
    lines = text.split('\n')
    if line_height is None:
        line_height = font.size + 8
    total_h = len(lines) * line_height
    y = cy - total_h // 2 + line_height // 2 - 4
    for line in lines:
        bbox = d.textbbox((0, 0), line, font=font)
        tw = bbox[2] - bbox[0]
        d.text((cx - tw // 2, y), line, font=font, fill=color)
        y += line_height

def draw_text_left(d, x, y, text, font, color, line_height=None):
    """Draw left-aligned (possibly multiline) text."""
    lines = text.split('\n')
    if line_height is None:
        line_height = font.size + 8
    for line in lines:
        d.text((x, y), line, font=font, fill=color)
        y += line_height

def draw_arrow_down(img, cx, y0, y1, color, thick=8):
    d = ImageDraw.Draw(img)
    d.line([(cx, y0), (cx, y1 - 22)], fill=color, width=thick)
    # Arrow head
    pts = [(cx, y1), (cx - 18, y1 - 30), (cx + 18, y1 - 30)]
    d.polygon(pts, fill=color)

def draw_arrow_right(img, x0, x1, cy, color, thick=8):
    d = ImageDraw.Draw(img)
    d.line([(x0, cy), (x1 - 22, cy)], fill=color, width=thick)
    pts = [(x1, cy), (x1 - 30, cy - 18), (x1 - 30, cy + 18)]
    d.polygon(pts, fill=color)

def color_with_alpha(rgb, a):
    return rgb[:3] + (a,)

# ── Background Gradient ────────────────────────────────────────────────────────
draw_gradient_rect(canvas, 0, 0, W, H, BG_GRAD_TOP, BG_GRAD_BOT, alpha=255)

# ── Header ─────────────────────────────────────────────────────────────────────
# Gradient header band
draw_gradient_rect(canvas, 0, 0, W, 320, (21, 101, 192), (13, 71, 161), alpha=255)
d = ImageDraw.Draw(canvas)
draw_text_centered(d, W//2, 118,
    'AI-Driven Pritamab Combination Therapy Selection: The ADDS Framework',
    F_TITLE, WHITE)
draw_text_centered(d, W//2, 222,
    'Anti-PrPc Monoclonal Antibody Synergy Pipeline for KRAS-Mutant Precision Oncology',
    F_TITLE2, (179, 212, 255))

# Decorative accent line under header
d.rectangle([80, 288, W-80, 296], fill=(100, 180, 255, 200))

# ═══════════════════════════════════════════════════════════════════════════════
# TIER 1 — SOURCE DATA STREAMS
# y: 340 to 1060
# ═══════════════════════════════════════════════════════════════════════════════
T1_Y0, T1_Y1 = 340, 1090

# Tier background
shadow_layer = Image.new('RGBA', canvas.size, (0, 0, 0, 0))
sd = ImageDraw.Draw(shadow_layer)
sd.rounded_rectangle([60, T1_Y0, W-60, T1_Y1], radius=36, fill=(0,0,0,35))
canvas.paste(shadow_layer.filter(ImageFilter.GaussianBlur(14)), (0, 0), shadow_layer)

d = ImageDraw.Draw(canvas)
draw_rounded_rect(d, 60, T1_Y0, W-60, T1_Y1, r=36,
                   fill=(235, 245, 255, 240), outline=(21, 101, 192), outline_width=6)

# T1 header
d.rounded_rectangle([60, T1_Y0, W-60, T1_Y0+90], radius=36, fill=(21, 101, 192))
d.rectangle([60, T1_Y0+50, W-60, T1_Y0+90], fill=(21, 101, 192))

d.text((140, T1_Y0+18), 'TIER 1', font=F_TIER, fill=(179, 212, 255))
d.text((310, T1_Y0+18), '— SOURCE DATA STREAMS', font=F_TIER, fill=WHITE)

# 4 source blocks
SOURCES = [
    ('Gene & Protein\nExpression',
     ['Gene mutations', 'KRAS G12D/V/C/G13D', 'PRNP / PrPc level', 'MSI · TMB · HER2'],
     COL_GENOMIC, '[DNA]'),
    ('Cellular\nPathology Analysis',
     ['H&E segmentation', 'AI-Cellpose platform', 'PrPc IHC H-score', 'Stromal fraction'],
     COL_CELL, '[LAB]'),
    ('CT Scan\nImaging Data',
     ['Tumour detection', 'AI nnU-Net model', 'Anatomical saliency', 'TNM staging'],
     COL_IMAGING, '[IMG]'),
    ('Patient\nClinical Data',
     ['Demographics', 'Comorbidities', 'Prior treatments', 'Performance (ECOG)'],
     COL_CLINIC, '[Rx]'),
]

src_x_starts = [140, 990, 1840, 2690]
src_w, src_h = 760, 590

for (title, bullets, color, icon), sx in zip(SOURCES, src_x_starts):
    draw_shadow_box(canvas, sx, T1_Y0+120, sx+src_w, T1_Y0+120+src_h,
                    r=28, fill=WHITE, outline=color, outline_w=7, shadow_offset=10)
    d = ImageDraw.Draw(canvas)
    # Top colored band
    d.rounded_rectangle([sx, T1_Y0+120, sx+src_w, T1_Y0+120+110],
                         radius=28, fill=color)
    d.rectangle([sx, T1_Y0+120+70, sx+src_w, T1_Y0+120+110], fill=color)
    # Icon label
    d.text((sx+30, T1_Y0+135), icon, font=F_HEAD, fill=WHITE)
    # Block title
    draw_text_centered(d, sx + src_w//2, T1_Y0+285, title, F_SUBHEAD, color)
    # Bullets
    for bi, b in enumerate(bullets):
        d.text((sx+45, T1_Y0+385 + bi*68), f'• {b}', font=F_BODY, fill=TEXT_M)

# Tier1 label tag
d.rounded_rectangle([60, T1_Y0+100, 210, T1_Y0+155], radius=18, fill=COL_PRIM)
d.text((80, T1_Y0+110), 'INPUT', font=F_BADGE, fill=WHITE)

# Arrows down T1 → T2
arrow_xs = [520, 1370, 2220, 3070]
for ax_ in arrow_xs:
    draw_arrow_down(canvas, ax_, T1_Y1, T1_Y1+90, color=ARROW_C, thick=10)

# ═══════════════════════════════════════════════════════════════════════════════
# TIER 2 — ADDS INTEGRATION ENGINE
# y: 1180 to 2600
# ═══════════════════════════════════════════════════════════════════════════════
T2_Y0, T2_Y1 = T1_Y1+90, 2620

shadow_layer2 = Image.new('RGBA', canvas.size, (0, 0, 0, 0))
sd2 = ImageDraw.Draw(shadow_layer2)
sd2.rounded_rectangle([60, T2_Y0, W-60, T2_Y1], radius=36, fill=(0,0,0,35))
canvas.paste(shadow_layer2.filter(ImageFilter.GaussianBlur(14)), (0,0), shadow_layer2)

d = ImageDraw.Draw(canvas)
draw_rounded_rect(d, 60, T2_Y0, W-60, T2_Y1, r=36,
                   fill=(230, 245, 232, 240), outline=(46, 125, 50), outline_width=6)

# T2 header band
d.rounded_rectangle([60, T2_Y0, W-60, T2_Y0+90], radius=36, fill=(46, 125, 50))
d.rectangle([60, T2_Y0+50, W-60, T2_Y0+90], fill=(46, 125, 50))
d.text((140, T2_Y0+18), 'TIER 2', font=F_TIER, fill=(178, 223, 181))
d.text((310, T2_Y0+18), '— ADDS INTEGRATION ENGINE', font=F_TIER, fill=WHITE)

# ── Column 1: AI Modules ───────────────────────────────────────────────────────
COL1_X0, COL1_X1 = 100, 1130

draw_shadow_box(canvas, COL1_X0, T2_Y0+110, COL1_X1, T2_Y1-50,
                r=28, fill=(232, 234, 246, 255), outline=COL_AI1, outline_w=5, shadow_offset=8)
d = ImageDraw.Draw(canvas)
d.text((COL1_X0+30, T2_Y0+126), 'AI Modules', font=F_HEAD, fill=COL_AI1)

AI_MODULES = [
    ('AI-Cellpose\nPathology',   COL_AI2),
    ('AI-nnU-Net\nCT Imaging',   COL_AI3),
    ('AI-PRNP\nGenomics',        COL_AI1),
    ('AI-PrPc\nBiomarker',       COL_AI4),
    ('AI-XGBoost\nResponse',     COL_AI5),
    ('AI-DeepSynergy\nSynergy',  COL_AI6),
]

ai_grid = [(0,0),(1,0),(0,1),(1,1),(0,2),(1,2)]
ai_bw, ai_bh = 460, 280
ai_x_offsets = [COL1_X0+30, COL1_X0+30+ai_bw+26]
ai_y_base = T2_Y0+220

for (col, row), (label, color) in zip(ai_grid, AI_MODULES):
    bx = ai_x_offsets[col]
    by = ai_y_base + row * (ai_bh + 24)
    draw_shadow_box(canvas, bx, by, bx+ai_bw, by+ai_bh,
                    r=22, fill=WHITE, outline=color, outline_w=6, shadow_offset=6)
    d = ImageDraw.Draw(canvas)
    # Color top accent strip
    d.rounded_rectangle([bx, by, bx+ai_bw, by+38], radius=22, fill=color)
    d.rectangle([bx, by+14, bx+ai_bw, by+38], fill=color)
    draw_text_centered(d, bx+ai_bw//2, by+ai_bh//2+14, label, F_BODY_B, color)

# Arrow AI → Pipeline
ai_mid_y = T2_Y0 + (T2_Y1 - T2_Y0) // 2
draw_arrow_right(canvas, COL1_X1, COL1_X1+80, ai_mid_y, color=COL_ENS, thick=12)

# ── Column 2: Processing Pipeline ─────────────────────────────────────────────
COL2_X0, COL2_X1 = 1210, 2270

draw_shadow_box(canvas, COL2_X0, T2_Y0+110, COL2_X1, T2_Y1-50,
                r=28, fill=(227, 242, 253, 255), outline=COL_PIPE, outline_w=5, shadow_offset=8)
d = ImageDraw.Draw(canvas)
d.text((COL2_X0+30, T2_Y0+126), 'Processing Pipeline', font=F_HEAD, fill=COL_PIPE)

PIPE_STEPS = [
    ('Data Normalisation\n& Quality Control',         (2, 136, 209)),
    ('Patient Stratification\nPrPc IHC + KRAS status', (21, 101, 192)),
    ('Multi-Modal Feature\nExtraction',                 (2, 136, 209)),
    ('Energy Landscape\nModelling   \u0394G / ODE',         (230, 81, 0)),
    ('Drug Interaction\nModelling',                     (2, 136, 209)),
    ('Efficacy & Synergy\nScoring (4-Model)',           (46, 125, 50)),
]

pipe_bw, pipe_bh = 930, 200
pipe_x = COL2_X0 + 55
pipe_y_start = T2_Y0 + 230
pipe_gap = 40

for i, (label, color) in enumerate(PIPE_STEPS):
    py = pipe_y_start + i * (pipe_bh + pipe_gap)
    draw_shadow_box(canvas, pipe_x, py, pipe_x+pipe_bw, py+pipe_bh,
                    r=20, fill=WHITE, outline=color, outline_w=6, shadow_offset=6)
    d = ImageDraw.Draw(canvas)
    # Left accent bar
    d.rounded_rectangle([pipe_x, py, pipe_x+18, py+pipe_bh], radius=10, fill=color)
    draw_text_centered(d, pipe_x+pipe_bw//2+10, py+pipe_bh//2-8, label, F_BODY_B, color, line_height=48)
    # Down arrows between steps
    if i < len(PIPE_STEPS) - 1:
        cy_ = py + pipe_bh + pipe_gap//2
        draw_arrow_down(canvas, pipe_x + pipe_bw//2, py+pipe_bh, py+pipe_bh+pipe_gap, color=ARROW_C, thick=8)

# Arrow Pipeline → Ensemble
draw_arrow_right(canvas, COL2_X1, COL2_X1+80, ai_mid_y, color=COL_ENS, thick=12)

# ── Column 3: Ensemble Methods ────────────────────────────────────────────────
COL3_X0, COL3_X1 = 2350, W-100

draw_shadow_box(canvas, COL3_X0, T2_Y0+110, COL3_X1, T2_Y1-50,
                r=28, fill=(232, 245, 233, 255), outline=COL_ENS, outline_w=5, shadow_offset=8)
d = ImageDraw.Draw(canvas)
d.text((COL3_X0+30, T2_Y0+126), 'Ensemble Methods', font=F_HEAD, fill=COL_ENS)

ENSEMBLE = [
    ('Random Forest',      'Backbone eligibility\nscoring',      (67, 160, 71)),
    ('Gradient Boosting',  'Response probability\nested.',        (30, 136, 229)),
    ('Deep Learning MLP',  'Drug-drug synergy\nmodelling',        (142, 36, 170)),
    ('ODE Energy Model',   '\u0394G filter | 55.6%\noncog. rate \u2193',(251, 140, 0)),
    ('4-Model Consensus',  'Bliss+Loewe+HSA+ZIP\nthreshold 0.75',(0, 137, 123)),
    ('DRS Aggregation',    'Drug Recommendation\nScore output',   (229, 57, 53)),
]

ens_bw, ens_bh = 1080, 196
ens_x = COL3_X0 + 40
ens_y_start = T2_Y0 + 230
ens_gap = 40

for i, (label, sub, color) in enumerate(ENSEMBLE):
    ey = ens_y_start + i * (ens_bh + ens_gap)
    draw_shadow_box(canvas, ens_x, ey, ens_x+ens_bw, ey+ens_bh,
                    r=20, fill=WHITE, outline=color, outline_w=6, shadow_offset=6)
    d = ImageDraw.Draw(canvas)
    d.rounded_rectangle([ens_x, ey, ens_x+18, ey+ens_bh], radius=10, fill=color)
    d.text((ens_x+40, ey+22), label, font=F_BODY_B, fill=color)
    draw_text_left(d, ens_x+40, ey+76, sub, F_BODY, TEXT_M, line_height=46)

# Arrows T2 → T3
arrow_xs2 = [520, 1740, 2960]
for ax_ in arrow_xs2:
    draw_arrow_down(canvas, ax_, T2_Y1, T2_Y1+90, color=ARROW_C, thick=10)

# ═══════════════════════════════════════════════════════════════════════════════
# TIER 3 — OUTPUT: PRITAMAB COMBINATION RECOMMENDATIONS
# y: 2710 to 4380
# ═══════════════════════════════════════════════════════════════════════════════
T3_Y0, T3_Y1 = T2_Y1+90, 4380

shadow_layer3 = Image.new('RGBA', canvas.size, (0, 0, 0, 0))
sd3 = ImageDraw.Draw(shadow_layer3)
sd3.rounded_rectangle([60, T3_Y0, W-60, T3_Y1], radius=36, fill=(0,0,0,35))
canvas.paste(shadow_layer3.filter(ImageFilter.GaussianBlur(14)), (0,0), shadow_layer3)

d = ImageDraw.Draw(canvas)
draw_rounded_rect(d, 60, T3_Y0, W-60, T3_Y1, r=36,
                   fill=(255, 248, 225, 240), outline=(230, 81, 0), outline_width=6)

# T3 header
d.rounded_rectangle([60, T3_Y0, W-60, T3_Y0+90], radius=36, fill=(230, 81, 0))
d.rectangle([60, T3_Y0+50, W-60, T3_Y0+90], fill=(230, 81, 0))
d.text((140, T3_Y0+18), 'TIER 3', font=F_TIER, fill=(255, 213, 164))
d.text((310, T3_Y0+18), '— OUTPUT — PRITAMAB COMBINATION RECOMMENDATIONS  (DRS Ranking)',
       font=F_TIER, fill=WHITE)

# ── Three recommendation cards ─────────────────────────────────────────────────
CARDS = [
    {
        'rank': '#1',
        'header': 'PRIMARY RECOMMENDATION',
        'title': 'Pritamab + FOLFOX',
        'subtitle': '5-FU + Leucovorin + Oxaliplatin',
        'drs': 'DRS: 0.893',
        'bliss': 'Bliss Synergy: +21.0',
        'ec50': 'EC50 reduction: \u221224.0%',
        'points': [
            'Target: Pi3K/AKT · RPSA signalosome blockade',
            'Selection: PrPc-HIGH (H-score \u226550) + KRAS-mutant',
            '  G12D (H-score 142) \u00b7 G12V (138) \u00b7 G13D (124)',
            'Loewe DRI   5-FU: 1.34 \u00b7 Oxaliplatin: 1.34',
            '24% dose reduction \u2192 Cumulative toxicity \u2193',
        ],
        'footer1': 'FDA Guidelines',
        'footer2': 'NCCN Protocol',
        'color': COL_PRIM,
        'light': (219, 234, 254),
        'x0': 130,
    },
    {
        'rank': '#2',
        'header': 'ALTERNATIVE REGIMEN',
        'title': 'Pritamab + Sotorasib',
        'subtitle': 'KRAS G12C Covalent Inhibitor',
        'drs': 'DRS: 0.882',
        'bliss': 'Bliss Synergy: +22.5',
        'ec50': 'EC50 reduction: \u221224.7%',
        'points': [
            'Dual-axis: RPSA-PrPc block + G12C direct inhibition',
            'Selection: KRAS G12C-mutant (~12-13% CRC) + PrPc-HIGH',
            '  VEGF anti-angiogenic target for metastatic CRC',
            'Addresses RTK-bypass escape via RPSA route',
            'Triple: +SHP2i \u2192 DRS 0.835 \u00b7 Bliss +28.0',
        ],
        'footer1': 'FDA Approved',
        'footer2': 'ESMO Guideline',
        'color': COL_ALT,
        'light': (220, 242, 220),
        'x0': 1300,
    },
    {
        'rank': '#3',
        'header': 'CONDITIONAL REGIMEN',
        'title': 'Pritamab + FOLFOXIRI',
        'subtitle': 'Triplet Intensification Regimen',
        'drs': 'DRS: 0.784',
        'bliss': 'Bliss Synergy: +22.0',
        'ec50': 'EC50 reduction: \u221226.1%',
        'points': [
            'Triplet intensification + PrPc chemosensitisation',
            'Selection: KRAS-mutant \u00b7 PrPc-HIGH \u00b7 ECOG 0-1 only',
            '  Hepatic metastasis conversion \u00b7 Younger patients',
            'Clinical eligibility review REQUIRED',
            'Bevacizumab triplet variant (TRIBE2) optional',
        ],
        'footer1': 'ESMO Guideline',
        'footer2': 'Clinical Review Required',
        'color': COL_COND,
        'light': (243, 229, 255),
        'x0': 2470,
    },
]

card_w = 1060
card_y0 = T3_Y0 + 110
card_y1 = T3_Y1 - 50

for card in CARDS:
    cx0 = card['x0']
    cx1 = cx0 + card_w
    color = card['color']
    light = card['light']

    # Card shadow + background
    draw_shadow_box(canvas, cx0, card_y0, cx1, card_y1,
                    r=32, fill=WHITE, outline=color, outline_w=8, shadow_offset=14)
    d = ImageDraw.Draw(canvas)

    # Top gradient header band
    draw_gradient_rect(canvas, cx0+8, card_y0+8, cx1-8, card_y0+200,
                       color, lerp_color(color, (50,50,80), 0.3))
    d = ImageDraw.Draw(canvas)
    d.rounded_rectangle([cx0+8, card_y0+8, cx1-8, card_y0+200], radius=28, outline=color, width=0)

    # Rank badge
    badge_r = 70
    badge_cx = cx0 + 80
    badge_cy = card_y0 + 108
    d.ellipse([badge_cx-badge_r, badge_cy-badge_r, badge_cx+badge_r, badge_cy+badge_r],
              fill=WHITE)
    draw_text_centered(d, badge_cx, badge_cy, card['rank'], F_BADGE, color)

    # Header text
    draw_text_centered(d, cx0+card_w//2+30, card_y0+100, card['header'], F_BADGE, WHITE)

    # Combination drug title
    draw_text_centered(d, cx0+card_w//2, card_y0+260, card['title'], F_TIER_LBL, color)
    draw_text_centered(d, cx0+card_w//2, card_y0+330, card['subtitle'], F_SUBHEAD, TEXT_M)

    # DRS Score badge
    d.rounded_rectangle([cx0+60, card_y0+370, cx1-60, card_y0+458],
                         radius=18, fill=(255, 248, 225), outline=(245, 127, 23), width=5)
    draw_text_centered(d, cx0+card_w//2, card_y0+414, card['drs'], F_DRS, (200, 80, 0))

    # Synergy metrics row
    mid_x = cx0 + card_w // 2
    d.rounded_rectangle([cx0+60, card_y0+476, mid_x-10, card_y0+548],
                         radius=14, fill=light, outline=COL_ENS, width=4)
    draw_text_centered(d, (cx0+60+mid_x-10)//2, card_y0+512, card['bliss'], F_BODY_B, COL_ENS)

    d.rounded_rectangle([mid_x+10, card_y0+476, cx1-60, card_y0+548],
                         radius=14, fill=light, outline=COL_PRIM, width=4)
    draw_text_centered(d, (mid_x+10+cx1-60)//2, card_y0+512, card['ec50'], F_BODY_B, COL_PRIM)

    # Divider
    d.line([(cx0+50, card_y0+572), (cx1-50, card_y0+572)], fill=(220, 220, 220), width=3)

    # Bullet points
    for bi, pt in enumerate(card['points']):
        by = card_y0 + 596 + bi * 80
        d.text((cx0+44, by), pt, font=F_BODY, fill=TEXT_M)

    # Footer pills
    pill_y = card_y1 - 110
    pill_texts = [card['footer1'], card['footer2']]
    pill_x = cx0 + 60
    for pt in pill_texts:
        bbox = d.textbbox((0, 0), pt, font=F_SMALL)
        pw = bbox[2] - bbox[0] + 40
        d.rounded_rectangle([pill_x, pill_y, pill_x+pw, pill_y+60],
                             radius=30, fill=color)
        d.text((pill_x+20, pill_y+12), pt, font=F_SMALL, fill=WHITE)
        pill_x += pw + 24

# ═══════════════════════════════════════════════════════════════════════════════
# BOTTOM INFRASTRUCTURE BANNER
# y: 4400 to 4550
# ═══════════════════════════════════════════════════════════════════════════════
INF_Y0, INF_Y1 = T3_Y1+40, T3_Y1+180

shadow_layer4 = Image.new('RGBA', canvas.size, (0, 0, 0, 0))
sd4 = ImageDraw.Draw(shadow_layer4)
sd4.rounded_rectangle([60, INF_Y0, W-60, INF_Y1], radius=22, fill=(0,0,0,28))
canvas.paste(shadow_layer4.filter(ImageFilter.GaussianBlur(8)), (0,0), shadow_layer4)

d = ImageDraw.Draw(canvas)
draw_rounded_rect(d, 60, INF_Y0, W-60, INF_Y1, r=22,
                   fill=(236, 239, 241), outline=(144, 164, 174), outline_width=4)

infra_items = [
    'Serum PrPc Liquid Biopsy  AUC=0.777',
    'IHC Biomarker: PrPc H-score >=50 + KRAS mut.',
    'Pharmacogenomics DB: 113 drugs | 59 synergy pairs',
    'ADDS v5.3  |  TCGA n=2,285  |  KR Patent Jan 2026',
]
inf_xs = [130, 1010, 1990, 2870]
for label, ix in zip(infra_items, inf_xs):
    d.text((ix, INF_Y0+44), label, font=F_BODY, fill=TEXT_M)
    if ix < 2870:
        d.line([(ix-30, INF_Y0+22), (ix-30, INF_Y1-22)], fill=(180,190,195), width=3)

# Caption
d.text((120, INF_Y1+30),
       'Figure: AI-Driven Pritamab Pipeline | Inha University Hospital x ADDS AI Framework | Nature Communications 2026',
       font=F_CAPTION, fill=TEXT_L)

# ── Final export ───────────────────────────────────────────────────────────────
out_path = r'f:\ADDS\figures\pritamab_infographic.png'
# Convert RGBA to RGB for saving
final = canvas.convert('RGB')
final.save(out_path, 'PNG', dpi=(300, 300), optimize=False)
print(f'Saved: {out_path}')
print(f'Size: {final.size[0]} x {final.size[1]} px')
