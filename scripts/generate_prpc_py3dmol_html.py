"""
generate_prpc_py3dmol_html.py
==============================
py3Dmol으로 PDB 2W9E (PrPC + ICSM18 Fab) 3D ribbon 뷰어
HTML 파일 생성 → 브라우저에서 고해상도 스크린샷 추출

PDB 2W9E chains:
  A = PrPC  (Human prion protein 125-223)
  H = Fab Heavy Chain (VH 1-130)
  L = Fab Light Chain (VL 1-115)
"""

import py3Dmol
import os

OUT_DIR   = r"f:\ADDS\outputs\pritamab_pptx_figures"
PDB_LOCAL = os.path.join(OUT_DIR, "2W9E.pdb")
OUT_HTML  = os.path.join(OUT_DIR, "prpc_pritamab_3d.html")

# Read PDB content
with open(PDB_LOCAL, 'r') as f:
    pdb_data = f.read()

# ── py3Dmol view ────────────────────────────────────────────────────
view = py3Dmol.view(width=1400, height=1000)

# Add PDB structure
view.addModel(pdb_data, 'pdb')

# ── Color each chain distinctly ─────────────────────────────────────

# Chain A (PrPC): color by secondary structure with custom residue coloring
# First set all of chain A to loop color
view.setStyle({'chain': 'A'}, {
    'cartoon': {
        'color': '#B0BEC5',      # loop default: silver-blue
        'thickness': 0.4,
        'opacity': 1.0
    }
})

# alpha-H3 (200-223): purple
view.setStyle({'chain': 'A', 'resi': '200-223'}, {
    'cartoon': {'color': '#9C27B0', 'thickness': 0.5}
})

# alpha-H2 (172-194): teal-green
view.setStyle({'chain': 'A', 'resi': '172-194'}, {
    'cartoon': {'color': '#00C896', 'thickness': 0.5}
})

# alpha-H1 (144-154): magenta [EPITOPE REGION]
view.setStyle({'chain': 'A', 'resi': '144-154'}, {
    'cartoon': {'color': '#E91E8C', 'thickness': 0.5}
})

# beta-sheet 1 (128-131): orange
view.setStyle({'chain': 'A', 'resi': '128-131'}, {
    'cartoon': {'color': '#FF8C00', 'thickness': 0.4, 'arrows': True}
})

# beta-sheet 2 (161-163): orange
view.setStyle({'chain': 'A', 'resi': '161-163'}, {
    'cartoon': {'color': '#FF8C00', 'thickness': 0.4, 'arrows': True}
})

# Epitope highlight (142-170): bright red + surface overlay
view.addStyle({'chain': 'A', 'resi': '142-170'}, {
    'cartoon': {
        'color': '#FF3333',
        'thickness': 0.6,
        'opacity': 0.95
    }
})

# Epitope surface (translucent gold)
view.addSurface(
    py3Dmol.SES,
    {'opacity': 0.22, 'color': '#FFD700'},
    {'chain': 'A', 'resi': '142-170'}
)

# Chain H (Fab Heavy - VH domain): deep blue gradient
view.setStyle({'chain': 'H'}, {
    'cartoon': {
        'colorscheme': 'chain',
        'color': '#1565C0',
        'thickness': 0.42,
        'opacity': 0.90
    }
})

# CDR3 loop of heavy chain (residues ~95-102): bright blue highlight
view.setStyle({'chain': 'H', 'resi': '95-102'}, {
    'cartoon': {'color': '#42A5F5', 'thickness': 0.5}
})
view.addSurface(
    py3Dmol.VDW,
    {'opacity': 0.15, 'color': '#42A5F5'},
    {'chain': 'H', 'resi': '95-102'}
)

# Chain L (Fab Light - VL domain): cyan
view.setStyle({'chain': 'L'}, {
    'cartoon': {
        'color': '#00ACC1',
        'thickness': 0.42,
        'opacity': 0.90
    }
})

# CDR3 of light chain (~89-97): bright cyan
view.setStyle({'chain': 'L', 'resi': '89-97'}, {
    'cartoon': {'color': '#26C6DA', 'thickness': 0.5}
})

# ── Label key residues ──────────────────────────────────────────────
label_style = {
    'fontSize': 14,
    'fontColor': 'white',
    'backgroundOpacity': 0.75,
    'backgroundColor': '#111111',
    'borderThickness': 1,
    'borderColor': '#FFD700',
    'padding': 3
}

view.addLabel('alpha-H1 (144-154)\nEpitope core', {
    'fontSize': 13, 'fontColor': '#E91E8C',
    'backgroundOpacity': 0.80, 'backgroundColor': '#0A0A0A',
    'borderThickness': 1, 'borderColor': '#E91E8C'
}, {'chain': 'A', 'resi': '149', 'atom': 'CA'})

view.addLabel('alpha-H2 (172-194)', {
    'fontSize': 13, 'fontColor': '#00C896',
    'backgroundOpacity': 0.80, 'backgroundColor': '#0A0A0A',
    'borderThickness': 1, 'borderColor': '#00C896'
}, {'chain': 'A', 'resi': '183', 'atom': 'CA'})

view.addLabel('alpha-H3 / GlobC\n(200-223)', {
    'fontSize': 13, 'fontColor': '#CE93D8',
    'backgroundOpacity': 0.80, 'backgroundColor': '#0A0A0A',
    'borderThickness': 1, 'borderColor': '#9C27B0'
}, {'chain': 'A', 'resi': '210', 'atom': 'CA'})

view.addLabel('Binding Epitope\n142-170  Kd~0.5nM', {
    'fontSize': 14, 'fontColor': '#FFD700',
    'backgroundOpacity': 0.90, 'backgroundColor': '#1A1100',
    'borderThickness': 2, 'borderColor': '#FFD700'
}, {'chain': 'A', 'resi': '155', 'atom': 'CA'})

view.addLabel('Pritamab Fab\nHeavy Chain (VH)', {
    'fontSize': 13, 'fontColor': '#90CAF9',
    'backgroundOpacity': 0.80, 'backgroundColor': '#050E1F',
    'borderThickness': 1, 'borderColor': '#1565C0'
}, {'chain': 'H', 'resi': '50', 'atom': 'CA'})

view.addLabel('Pritamab Fab\nLight Chain (VL)', {
    'fontSize': 13, 'fontColor': '#80DEEA',
    'backgroundOpacity': 0.80, 'backgroundColor': '#05171F',
    'borderThickness': 1, 'borderColor': '#00ACC1'
}, {'chain': 'L', 'resi': '50', 'atom': 'CA'})

# ── Camera / background ─────────────────────────────────────────────
view.setBackgroundColor('#07080F')
# Zoom to fit all three chains with extra margin
view.zoomTo()
view.zoom(0.65)   # zoom out further so nothing clips
# Angle: slightly tilt to show binding interface
view.rotate(20, 'y')
view.rotate(-10, 'x')

# ── Export HTML ──────────────────────────────────────────────────────
html_raw = view._make_html()

# Inject custom CSS and screenshot button
INJECT_CSS = """
<style>
  html, body {
    margin: 0; padding: 0;
    overflow: hidden;              /* 스크롤바 완전 제거 */
    background: #07080F;
    width: 100%; height: 100%;
  }
  canvas { display: block; }
  #screenshot-btn {
    position: fixed; top: 16px; right: 16px;
    background: #FFD700; color: #000; border: none;
    padding: 10px 20px; font-size: 15px; font-weight: bold;
    border-radius: 8px; cursor: pointer; z-index: 9999;
  }
  #title-overlay {
    position: fixed; top: 16px; left: 50%;
    transform: translateX(-50%);
    text-align: center; color: white;
    font-family: 'Segoe UI', Arial, sans-serif;
    pointer-events: none; z-index: 9998;
    white-space: nowrap;
  }
  #title-overlay h2 {
    margin: 0; font-size: 20px; font-weight: 700;
    text-shadow: 0 0 12px #FFD70088;
  }
  #title-overlay p {
    margin: 4px 0 0; font-size: 11px; color: #8899BB;
    font-style: italic;
  }
  #legend {
    position: fixed; bottom: 16px; left: 16px;
    background: rgba(10,14,30,0.92);
    border: 1px solid #334455; border-radius: 8px;
    padding: 10px 14px; color: white;
    font-family: 'Segoe UI', Arial, sans-serif; font-size: 11.5px;
    z-index: 9998;
  }
  .leg-row { display: flex; align-items: center; margin: 3px 0; }
  .leg-swatch { width: 16px; height: 9px; border-radius: 2px; margin-right: 7px; flex-shrink:0; }
</style>
"""

INJECT_BODY = """
<div id="title-overlay">
  <h2>PrPC &ndash; Pritamab Antibody Binding Complex</h2>
  <p>PDB 2W9E &nbsp;|&nbsp; X-ray 2.1 &Aring; &nbsp;|&nbsp;
     Human PrP (Res 119&ndash;231) + Fab Fragment (ICSM18 / Pritamab)</p>
</div>

<button id="screenshot-btn" onclick="takeScreenshot()">&#128247; Screenshot</button>

<div id="legend">
  <div class="leg-row"><div class="leg-swatch" style="background:#E91E8C"></div>&alpha;-H1 (144&ndash;154) &larr; Epitope</div>
  <div class="leg-row"><div class="leg-swatch" style="background:#00C896"></div>&alpha;-H2 (172&ndash;194)</div>
  <div class="leg-row"><div class="leg-swatch" style="background:#9C27B0"></div>&alpha;-H3 / GlobC (200&ndash;223)</div>
  <div class="leg-row"><div class="leg-swatch" style="background:#FF8C00"></div>&beta;-sheet (&beta;1, &beta;2)</div>
  <div class="leg-row"><div class="leg-swatch" style="background:#B0BEC5"></div>Loop / Coil</div>
  <div class="leg-row"><div class="leg-swatch" style="background:#FF3333"></div>Binding Epitope (142&ndash;170) &starf;</div>
  <div class="leg-row"><div class="leg-swatch" style="background:#1565C0"></div>Fab Heavy Chain VH (Pritamab)</div>
  <div class="leg-row"><div class="leg-swatch" style="background:#00ACC1"></div>Fab Light Chain VL (Pritamab)</div>
  <div style="margin-top:8px; color:#FFD700; font-size:11px;">
    K<sub>d</sub> &asymp; 0.5 nM &nbsp;|&nbsp; &Delta;G = &minus;13.0 kcal/mol<br>
    MM-GBSA / Eyring&ndash;Evans&ndash;Polanyi TST
  </div>
</div>

<script>
function takeScreenshot() {
  // Access the 3Dmol viewer canvas
  var canvases = document.querySelectorAll('canvas');
  if (canvases.length === 0) { alert('No canvas found'); return; }
  var canvas = canvases[canvases.length - 1];
  var link = document.createElement('a');
  link.download = 'prpc_pritamab_binding_3d.png';
  link.href = canvas.toDataURL('image/png');
  link.click();
}
</script>
"""

# Insert into HTML
html_final = html_raw.replace('</head>', INJECT_CSS + '</head>')
html_final = html_final.replace('<body>', '<body>' + INJECT_BODY)

with open(OUT_HTML, 'w', encoding='utf-8') as f:
    f.write(html_final)

print(f"Saved HTML: {OUT_HTML}")
print("Opening in browser for screenshot...")
