"""
prpc_pathway_3d_pub.py
=======================
논문 게재용 PrPC 시그널 패스웨이 3D 유기구조 다이어그램
- 흰 배경 (publication-ready)
- Plotly 3D: 구형 노드, Bezier 3D 곡선 엣지
- 5층 구조: Extracellular → Membrane → Cytoplasm → Nucleus → Outcomes
- kaleido로 고해상도 PNG 저장
"""

import numpy as np
import plotly.graph_objects as go
import plotly.io as pio

OUT = r"f:\ADDS\outputs\pritamab_pptx_figures\prpc_pathway_3d_pub.png"
HTML = r"f:\ADDS\outputs\pritamab_pptx_figures\prpc_pathway_3d_pub.html"

# ══════════════════════════════════════════════════════════════════════
# 1. 노드 정의  (x, y, z, label, color, size, group)
# ══════════════════════════════════════════════════════════════════════
# Z 레벨
Z = {
    'EC':  12.0,
    'MEM':  9.5,
    'CYT1': 6.5,
    'CYT2': 4.0,
    'NUC':  1.5,
    'OUT':  -1.5,
}

# color palette (publication)
COL = {
    'prpc':    '#2196F3',  # blue  — PrPC
    'block':   '#FF8F00',  # amber — Pritamab / block
    'path':    '#E53935',  # red   — pathological
    'kinase':  '#388E3C',  # green — kinases
    'second':  '#7B1FA2',  # purple— 2nd messenger
    'tf':      '#1565C0',  # navy  — transcription factor
    'bad_tf':  '#B71C1C',  # dark red — pro-death TF
    'good_out':'#00695C',  # teal  — neuroprotection
    'bad_out': '#C62828',  # crimson— neurodegeneration
    'syn_out': '#2E7D32',  # dark green — synaptic
    'receptor':'#6A1B9A',  # violet— receptor
    'copper':  '#EF6C00',  # orange— metal binding
    'raft':    '#0277BD',  # light blue — raft
    'ros':     '#D84315',  # deep orange — oxidative
}

nodes = [
    # ── EXTRACELLULAR ──────────────────────────────────────────────
    (-1.5,  2.5, Z['EC'],  'Cu²⁺',               COL['copper'],  18, 'EC'),
    ( 1.5,  2.5, Z['EC'],  'Laminin /\nHeparin',  COL['raft'],    18, 'EC'),
    (-3.5,  0.0, Z['EC'],  'Pritamab\n(ICSM18)',  COL['block'],   32, 'EC'),
    ( 3.5,  0.0, Z['EC'],  'PrPSc\nAggregates',  COL['path'],    26, 'EC'),

    # ── MEMBRANE ──────────────────────────────────────────────────
    ( 0.0,  0.0, Z['MEM'], 'PrPC\n(GPI)',         COL['prpc'],    42, 'MEM'),
    (-2.5,  2.0, Z['MEM'], 'mGluR5',              COL['receptor'],22, 'MEM'),
    ( 2.5,  2.0, Z['MEM'], 'NCAM1',               COL['receptor'],22, 'MEM'),
    (-2.5, -2.0, Z['MEM'], 'LRP1/LRP2',           COL['receptor'],22, 'MEM'),
    ( 2.5, -2.0, Z['MEM'], 'Lipid Raft\nCaveolea', COL['raft'],   20, 'MEM'),

    # ── CYTOPLASM I : Kinases ──────────────────────────────────────
    (-4.0,  1.5, Z['CYT1'], 'Fyn\nKinase',        COL['kinase'],  26, 'CYT1'),
    (-2.0,  1.5, Z['CYT1'], 'Src\nKinase',        COL['kinase'],  22, 'CYT1'),
    ( 0.0,  1.5, Z['CYT1'], 'PI3K\n/ Akt',        COL['kinase'],  26, 'CYT1'),
    ( 2.0,  1.5, Z['CYT1'], 'ERK\n1/2',           COL['kinase'],  22, 'CYT1'),
    ( 4.0,  1.5, Z['CYT1'], 'PTEN',               COL['bad_tf'],  20, 'CYT1'),
    (-2.0, -1.5, Z['CYT1'], 'NMDAR\n(NR2B)',      COL['receptor'],24, 'CYT1'),
    ( 2.0, -1.5, Z['CYT1'], 'Caveolin-1',         COL['raft'],    20, 'CYT1'),
    ( 4.0, -1.5, Z['CYT1'], 'ROS↑\nOx.Stress',   COL['ros'],     22, 'CYT1'),
    (-4.0, -1.5, Z['CYT1'], 'SOD1\n/ SOD2',       COL['kinase'],  20, 'CYT1'),

    # ── CYTOPLASM II : 2nd Messengers ────────────────────────────
    (-3.0,  0.0, Z['CYT2'], 'mTOR',               COL['second'],  24, 'CYT2'),
    (-1.0,  0.0, Z['CYT2'], 'p38\nMAPK',          COL['second'],  22, 'CYT2'),
    ( 1.0,  0.0, Z['CYT2'], 'GSK-3β',             COL['second'],  24, 'CYT2'),
    ( 3.0,  0.0, Z['CYT2'], 'Ca²⁺\nInflux',       COL['ros'],     24, 'CYT2'),
    (-3.0, -2.0, Z['CYT2'], 'Autophagy\n/ UPS',   COL['kinase'],  20, 'CYT2'),
    ( 3.0, -2.0, Z['CYT2'], 'Mitochondria\nΔΨm↓', COL['ros'],     22, 'CYT2'),

    # ── NUCLEUS ───────────────────────────────────────────────────
    (-3.5,  0.0, Z['NUC'], 'NF-κB',               COL['tf'],      26, 'NUC'),
    (-1.5,  0.0, Z['NUC'], 'CREB',                COL['tf'],      24, 'NUC'),
    ( 0.5,  0.0, Z['NUC'], 'p53',                 COL['bad_tf'],  24, 'NUC'),
    ( 2.5,  0.0, Z['NUC'], 'Tau-P\n(tangle)',     COL['bad_tf'],  22, 'NUC'),
    ( 3.8,  0.0, Z['NUC'], 'Bcl-2\n/ Casp-3',    COL['bad_tf'],  22, 'NUC'),
    (-3.5, -2.0, Z['NUC'], 'BDNF\n/ NGF',         COL['tf'],      22, 'NUC'),

    # ── OUTCOMES ──────────────────────────────────────────────────
    (-3.5,  0.0, Z['OUT'], '🛡 Neuroprotection',  COL['good_out'],36, 'OUT'),
    ( 0.0,  0.0, Z['OUT'], '🧠 Synaptic Plasticity\n& Memory (LTP/LTD)', COL['syn_out'], 36, 'OUT'),
    ( 4.0,  0.0, Z['OUT'], '💀 Neurodegeneration\n(prion / AD / PD)',     COL['bad_out'], 36, 'OUT'),
]

# ══════════════════════════════════════════════════════════════════════
# 2. 엣지 정의 (node index 기반, activation / inhibition / block)
# ══════════════════════════════════════════════════════════════════════
# node index  빠르게 찾기
def ni(label_prefix):
    for i, n in enumerate(nodes):
        if n[3].startswith(label_prefix.split('\n')[0]):
            return i
    raise ValueError(f"Node not found: {label_prefix}")

EDGES = [
    # EC → Membrane
    ('Cu²⁺',      'PrPC',      'activate', 1.5),
    ('Laminin',   'LRP1',      'activate', 1.2),
    ('PrPSc',     'PrPC',      'activate', 2.0),  # misfolding pressure
    # Pritamab --| PrPC (block)
    ('Pritamab',  'PrPC',      'block',    2.5),

    # Membrane → Membrane
    ('PrPC',     'mGluR5',    'activate', 1.2),
    ('PrPC',     'NCAM1',     'activate', 1.2),
    ('PrPC',     'LRP1',      'activate', 1.2),
    ('PrPC',     'Lipid Raft','activate', 1.0),

    # Membrane → CYT1
    ('PrPC',     'Fyn',       'activate', 2.0),
    ('PrPC',     'Src',       'activate', 1.5),
    ('PrPC',     'PI3K',      'activate', 1.8),
    ('PrPC',     'ERK',       'activate', 1.5),
    ('mGluR5',   'NMDAR',     'activate', 1.5),
    ('Lipid Raft','Caveolin-1','activate', 1.2),
    ('LRP1',     'Fyn',       'activate', 1.2),
    ('PrPSc',    'ROS',       'activate', 2.0),

    # CYT1 → CYT1
    ('Fyn',      'PI3K',      'activate', 1.2),
    ('SOD1',     'ROS',       'inhibit',  1.2),

    # CYT1 → CYT2
    ('PI3K',     'mTOR',      'activate', 1.5),
    ('ERK',      'p38',       'activate', 1.2),
    ('PTEN',     'GSK-3β',    'activate', 1.2),
    ('NMDAR',    'Ca²⁺',      'activate', 2.0),
    ('Caveolin-1','Autophagy','activate', 1.0),
    ('ROS',      'Mitochondria','activate',1.5),
    ('ROS',      'p38',       'activate', 1.2),

    # CYT2 → NUC
    ('mTOR',     'NF-κB',     'activate', 1.5),
    ('p38',      'p53',       'activate', 1.2),
    ('GSK-3β',   'Tau-P',     'activate', 1.8),
    ('GSK-3β',   'CREB',      'inhibit',  1.2),
    ('Ca²⁺',     'Bcl-2',     'activate', 1.5),
    ('Ca²⁺',     'Tau-P',     'activate', 1.2),
    ('Autophagy','BDNF',      'activate', 1.0),
    ('Mitochondria','Bcl-2',  'activate', 1.2),
    ('mTOR',     'CREB',      'activate', 1.2),

    # NUC → Outcomes
    ('NF-κB',    'Neuroprotection','activate', 1.5),
    ('NF-κB',    'Neurodegeneration','activate',1.0),
    ('CREB',     'Synaptic',  'activate', 2.0),
    ('BDNF',     'Synaptic',  'activate', 1.5),
    ('BDNF',     'Neuroprotection','activate',1.2),
    ('p53',      'Neurodegeneration','activate',1.5),
    ('Tau-P',    'Neurodegeneration','activate',1.8),
    ('Bcl-2',    'Neurodegeneration','activate',1.5),

    # Pritamab cross-block: PrPSc conversion / neurodegeneration
    ('Pritamab', 'Neurodegeneration','block',1.5),
]

# ══════════════════════════════════════════════════════════════════════
# 3. Bezier 곡선 (3D, 층 사이 S자 경로)
# ══════════════════════════════════════════════════════════════════════
def bezier3d(p0, p3, n=60):
    """입방 베지에 곡선 — 중간 두 제어점을 Z-중간값으로 생성"""
    x0,y0,z0 = p0; x3,y3,z3 = p3
    # 제어점: 약간 바깥쪽으로 휘어짐
    mid_z = (z0+z3)/2
    ctrl_off = (z0-z3)*0.15
    x1 = x0 + (x3-x0)*0.3 + ctrl_off
    y1 = y0 + (y3-y0)*0.3
    z1 = z0 - (z0-z3)*0.35
    x2 = x0 + (x3-x0)*0.7 - ctrl_off
    y2 = y0 + (y3-y0)*0.7
    z2 = z0 - (z0-z3)*0.65
    t = np.linspace(0,1,n)
    b = lambda p0,p1,p2,p3: (
        (1-t)**3*p0 + 3*(1-t)**2*t*p1 +
        3*(1-t)*t**2*p2 + t**3*p3
    )
    return b(x0,x1,x2,x3), b(y0,y1,y2,y3), b(z0,z1,z2,z3)

# ══════════════════════════════════════════════════════════════════════
# 4. 레이어 수평 평판 (Mesh3d 반투명)
# ══════════════════════════════════════════════════════════════════════
def make_layer_mesh(z_val, color, opacity=0.07):
    x = [-5.5, 5.5, 5.5, -5.5]
    y = [-3.5, -3.5, 3.5, 3.5]
    return go.Mesh3d(
        x=x, y=y, z=[z_val]*4,
        i=[0,0], j=[1,2], k=[2,3],
        color=color,
        opacity=opacity,
        hoverinfo='skip',
        showscale=False,
    )

# ══════════════════════════════════════════════════════════════════════
# 5. Figure 구성
# ══════════════════════════════════════════════════════════════════════
fig_traces = []

# 레이어 평판
layer_info = [
    (Z['EC'],   '#90CAF9', 'Extracellular'),
    (Z['MEM'],  '#80DEEA', 'Membrane'),
    (Z['CYT1'], '#A5D6A7', 'Cytoplasm I — Kinases'),
    (Z['CYT2'], '#CE93D8', 'Cytoplasm II — 2nd Msg'),
    (Z['NUC'],  '#FFAB91', 'Nucleus'),
    (Z['OUT'],  '#B0BEC5', 'Outcomes'),
]
for z_val, col, lname in layer_info:
    fig_traces.append(make_layer_mesh(z_val, col, 0.08))

# 노드 인덱스 맵
node_pos = {n[3]: (n[0], n[1], n[2]) for n in nodes}

def find_pos(label_prefix):
    for n in nodes:
        if n[3].startswith(label_prefix.split('\n')[0].strip()):
            return (n[0], n[1], n[2])
    # partial match
    for n in nodes:
        if label_prefix.lower() in n[3].lower():
            return (n[0], n[1], n[2])
    raise ValueError(f"Cannot find: {label_prefix}")

# 엣지 — Bezier 곡선 trace
edge_style = {
    'activate': dict(color='#555555', width=2.0, dash='solid'),
    'inhibit':  dict(color='#E53935', width=2.0, dash='dash'),
    'block':    dict(color='#FF8F00', width=2.5, dash='dash'),
}

for (src, dst, etype, lw) in EDGES:
    try:
        p0 = find_pos(src); p3 = find_pos(dst)
    except ValueError:
        continue
    xs, ys, zs = bezier3d(p0, p3)
    ec = edge_style.get(etype, edge_style['activate'])
    fig_traces.append(go.Scatter3d(
        x=xs, y=ys, z=zs,
        mode='lines',
        line=dict(color=ec['color'], width=lw*ec['width']),
        hoverinfo='skip',
        showlegend=False,
    ))

# 노드 — 구형 scatter (각 그룹별 trace)
groups = ['EC','MEM','CYT1','CYT2','NUC','OUT']
group_names = {
    'EC':   'Extracellular',
    'MEM':  'Membrane',
    'CYT1': 'Cytoplasm I (Kinases)',
    'CYT2': 'Cytoplasm II (2nd Msg)',
    'NUC':  'Nucleus (TFs)',
    'OUT':  'Outcomes',
}
for grp in groups:
    gn = [n for n in nodes if n[6]==grp]
    fig_traces.append(go.Scatter3d(
        x=[n[0] for n in gn],
        y=[n[1] for n in gn],
        z=[n[2] for n in gn],
        mode='markers+text',
        marker=dict(
            size=[n[5] for n in gn],          # n[5] = marker_size
            color=[n[4] for n in gn],          # n[4] = color_hex
            opacity=0.90,
            line=dict(color='white', width=1.5),
            symbol='circle',
        ),
        text=[n[3] for n in gn],              # n[3] = label
        textposition='middle center',
        textfont=dict(size=8.5, color='#111111', family='Arial'),
        name=group_names[grp],
        hovertemplate='<b>%{text}</b><extra></extra>',
    ))

# 레이어 라벨 (3D 텍스트)
for z_val, col, lname in layer_info:
    fig_traces.append(go.Scatter3d(
        x=[-5.3], y=[3.3], z=[z_val],
        mode='text',
        text=[lname],
        textfont=dict(size=10, color=col.replace('#','#'),
                      family='Arial Black'),
        hoverinfo='skip',
        showlegend=False,
    ))

# ══════════════════════════════════════════════════════════════════════
# 6. 레이아웃 (흰 배경, 논문 스타일)
# ══════════════════════════════════════════════════════════════════════
fig = go.Figure(data=fig_traces)

fig.update_layout(
    title=dict(
        text='PrPC Signaling Pathway — 3D Layered Network',
        font=dict(size=20, family='Arial', color='#1A1A1A'),
        x=0.5, xanchor='center'
    ),
    paper_bgcolor='white',
    plot_bgcolor='white',
    scene=dict(
        bgcolor='white',
        xaxis=dict(visible=False, showgrid=False,
                   zeroline=False, showbackground=False),
        yaxis=dict(visible=False, showgrid=False,
                   zeroline=False, showbackground=False),
        zaxis=dict(
            visible=True,
            showgrid=False,
            zeroline=False,
            showbackground=False,
            tickvals=[Z['EC'], Z['MEM'], Z['CYT1'],
                      Z['CYT2'], Z['NUC'], Z['OUT']],
            ticktext=['Extracellular','Membrane',
                      'Cytoplasm I','Cytoplasm II',
                      'Nucleus','Outcomes'],
            tickfont=dict(size=10, family='Arial', color='#444'),
            title='',
        ),
        camera=dict(
            eye=dict(x=1.55, y=-1.85, z=0.55),
            center=dict(x=0, y=0, z=-0.15),
            up=dict(x=0, y=0, z=1),
        ),
        aspectmode='manual',
        aspectratio=dict(x=2.0, y=1.3, z=3.2),
    ),
    showlegend=True,
    legend=dict(
        x=0.01, y=0.98,
        bgcolor='rgba(255,255,255,0.85)',
        bordercolor='#CCCCCC',
        borderwidth=1,
        font=dict(size=10, family='Arial'),
    ),
    width=1800, height=1200,
    margin=dict(l=20, r=20, t=65, b=20),
)

# ══════════════════════════════════════════════════════════════════════
# 7. 저장
# ══════════════════════════════════════════════════════════════════════
# HTML (브라우저에서 인터랙티브)
fig.write_html(HTML)
print(f"HTML saved: {HTML}")

# PNG (kaleido)
try:
    fig.write_image(OUT, scale=2.5)
    print(f"PNG saved: {OUT}")
except Exception as e:
    print(f"kaleido error ({e}) — saving HTML only. Open HTML and screenshot manually.")
