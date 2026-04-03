"""
render_prpc_antibody_3d.py  v2
================================
PDB 2W9E  (Human PrPC 125-223) + Fab Heavy(H) + Fab Light(L)
실제 X-ray 결정구조 기반 Pritamab binding 3D Ribbon 시각화

개선사항 v2:
- VH/VL 도메인(첫 ~130 잔기)만 사용 → chain이 너무 길어지는 문제 해소
- figure annotation (ax.text) 고정 오프셋 → 레이블 겹침 해소
- elev/azim 최적화 → helix stack 잘 보이는 각도
- 구조물 zoom fit
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
import os, urllib.request
from scipy.interpolate import splprep, splev

# ── Paths ─────────────────────────────────────────────────────────────
OUT_DIR   = r"f:\ADDS\outputs\pritamab_pptx_figures"
OUT_PATH  = os.path.join(OUT_DIR, "prpc_antibody_binding_3d.png")
PDB_LOCAL = os.path.join(OUT_DIR, "2W9E.pdb")
os.makedirs(OUT_DIR, exist_ok=True)

if not os.path.exists(PDB_LOCAL):
    print("Downloading 2W9E...")
    urllib.request.urlretrieve("https://files.rcsb.org/download/2W9E.pdb", PDB_LOCAL)

# ── Parse ─────────────────────────────────────────────────────────────
from Bio.PDB import PDBParser
parser    = PDBParser(QUIET=True)
structure = parser.get_structure("2W9E", PDB_LOCAL)
model     = structure[0]

EPITOPE = (142, 170)          # ICSM18 binding epitope on PrPC
VH_MAX  = 130                 # Only show VH variable domain of heavy chain
VL_MAX  = 115                 # Only show VL variable domain of light chain

def get_ca(chain_id, res_max=9999):
    ch  = model[chain_id]
    xyz, rn = [], []
    for res in ch.get_residues():
        if res.get_id()[0] != ' ': continue
        rnum = res.get_id()[1]
        if rnum > res_max: continue
        if 'CA' in res:
            xyz.append(res['CA'].get_vector().get_array())
            rn.append(rnum)
    return np.array(xyz, float), np.array(rn)

ca_P, rn_P = get_ca('A')              # PrPC  (all 99 residues)
ca_H, rn_H = get_ca('H', VH_MAX)     # Fab VH domain only
ca_L, rn_L = get_ca('L', VL_MAX)     # Fab VL domain only

# Center on PrPC
cen = ca_P.mean(axis=0)
ca_P -= cen; ca_H -= cen; ca_L -= cen

# ── SS coloring for PrPC ──────────────────────────────────────────────
def ss_color_prpc(rn):
    if 200 <= rn <= 223: return '#9C27B0'  # alpha-H3 purple
    if 172 <= rn <= 194: return '#00C896'  # alpha-H2 teal
    if 144 <= rn <= 154: return '#E91E8C'  # alpha-H1 magenta
    if 128 <= rn <= 131: return '#FF8C00'  # beta1 orange
    if 161 <= rn <= 163: return '#FF8C00'  # beta2 orange
    return '#B0BEC5'                        # loop

# Fab: uniform color gradients
def ss_color_H(rn):
    t = rn / VH_MAX
    r = int(21 + (0-21)*t);  g = int(101 + (132-101)*t); b = int(192 + (255-192)*t)
    return f'#{r:02X}{g:02X}{b:02X}'

def ss_color_L(rn):
    t = rn / VL_MAX
    r = int(0  + (0-0)*t);   g = int(172 + (229-172)*t); b = int(193 + (255-193)*t)
    return f'#{r:02X}{g:02X}{b:02X}'

# ── Smooth spline ─────────────────────────────────────────────────────
def smooth_ca(pts, n=500, s=2.0):
    if len(pts) < 4: return pts
    tck, u = splprep(pts.T, s=s, k=3)
    return np.array(splev(np.linspace(0,1,n), tck)).T

sm_P = smooth_ca(ca_P, n=600, s=1.8)
sm_H = smooth_ca(ca_H, n=300, s=2.0)
sm_L = smooth_ca(ca_L, n=300, s=2.0)

# ── Build ribbon quads ────────────────────────────────────────────────
def ribbon_quads(smooth, res_nums, color_fn, width=0.50):
    n     = len(smooth)
    tang  = np.gradient(smooth, axis=0)
    tang /= (np.linalg.norm(tang, axis=1, keepdims=True)+1e-9)
    ref   = np.array([0.,0.,1.])
    norm  = np.cross(tang, ref)
    norm /= (np.linalg.norm(norm, axis=1, keepdims=True)+1e-9)
    L, R  = smooth + norm*width/2, smooth - norm*width/2
    u_r   = np.linspace(0,1,len(res_nums))
    u_s   = np.linspace(0,1,n)
    verts, cols = [], []
    for i in range(n-1):
        idx = int(np.clip(np.searchsorted(u_r, u_s[i]), 0, len(res_nums)-1))
        verts.append([L[i], R[i], R[i+1], L[i+1]])
        cols.append(color_fn(res_nums[idx]))
    return verts, cols

vP, cP = ribbon_quads(sm_P, rn_P, ss_color_prpc, width=0.60)
vH, cH = ribbon_quads(sm_H, rn_H, ss_color_H,    width=0.52)
vL, cL = ribbon_quads(sm_L, rn_L, ss_color_L,    width=0.52)

# ── Epitope highlight parameters ──────────────────────────────────────
u_r_P = np.linspace(0,1,len(rn_P))
u_s_P = np.linspace(0,1,len(sm_P))
epi_mask = (rn_P >= EPITOPE[0]) & (rn_P <= EPITOPE[1])
epi_idx  = np.where(epi_mask)[0]
i_lo = int(np.searchsorted(u_s_P, u_r_P[epi_idx[0]]))
i_hi = int(np.searchsorted(u_s_P, u_r_P[epi_idx[-1]]))
ep   = sm_P[i_lo:i_hi]
epi_cen = ep.mean(axis=0)

# ── Figure ────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(14, 10.5), facecolor='#060A14', dpi=220)
ax  = fig.add_subplot(111, projection='3d', facecolor='#060A14')

# PrPC ribbon
polyP = Poly3DCollection(vP, zsort='average', linewidth=0, alpha=0.93)
polyP.set_facecolors(cP); polyP.set_edgecolors(cP)
ax.add_collection3d(polyP)

# Epitope glow
for lw, al, col in [(12,0.07,'#FFD700'),(7,0.18,'#FF8800'),(3.5,0.80,'#FF2222')]:
    ax.plot(ep[:,0],ep[:,1],ep[:,2],'-',color=col,lw=lw,alpha=al,zorder=7)

# Fab Heavy
polyH = Poly3DCollection(vH, zsort='average', linewidth=0, alpha=0.87)
polyH.set_facecolors(cH); polyH.set_edgecolors(cH)
ax.add_collection3d(polyH)

# Fab Light
polyL = Poly3DCollection(vL, zsort='average', linewidth=0, alpha=0.87)
polyL.set_facecolors(cL); polyL.set_edgecolors(cL)
ax.add_collection3d(polyL)

# Centerlines
for sm, col in [(sm_P,'#FFF'),(sm_H,'#90CAF9'),(sm_L,'#80DEEA')]:
    ax.plot(sm[:,0],sm[:,1],sm[:,2],'-',color=col,lw=0.35,alpha=0.14)

# ── Interface contact cloud ───────────────────────────────────────────
# Find CDR3-like contact point (last 15 residues of VH → tip of CDR3)
fab_tip = sm_H[-20:].mean(axis=0)
rng = np.random.default_rng(7)
for _ in range(50):
    pt  = (epi_cen + fab_tip)/2 + rng.normal(0, 0.7, 3)
    r   = rng.uniform(0.3, 0.8)
    u_s = np.linspace(0,2*np.pi,10)
    xs  = pt[0]+r*np.outer(np.cos(u_s), np.sin(np.linspace(0,np.pi,6)))
    ys  = pt[1]+r*np.outer(np.sin(u_s), np.sin(np.linspace(0,np.pi,6)))
    zs  = pt[2]+r*np.outer(np.ones(10),  np.cos(np.linspace(0,np.pi,6)))
    ax.plot_surface(xs,ys,zs,color='#FFD700',alpha=0.05,linewidth=0,antialiased=False)

# ── Draw dashed binding interface line ───────────────────────────────
mid = np.linspace(epi_cen, fab_tip, 30)
ax.plot(mid[:,0],mid[:,1],mid[:,2],'--',color='#FFD700',lw=1.5,alpha=0.45,zorder=8)

# ── Annotations via fixed figure-coord offsets ────────────────────────
bbox_kw  = lambda col: dict(boxstyle='round,pad=0.20',fc='#060A14',ec=col,alpha=0.92,lw=1.1)

# Helper: find Cα position at residue number in PrPC
def pt_at(rn_t):
    idx = np.argmin(np.abs(rn_P - rn_t))
    # map to smooth
    u_r = np.linspace(0,1,len(rn_P))
    u_s = np.linspace(0,1,len(sm_P))
    i   = int(np.searchsorted(u_s, u_r[idx]))
    return sm_P[np.clip(i,0,len(sm_P)-1)]

annotations = [
    # (rn_target, label_text, color, dx, dy, dz)
    (148, 'alpha-H1  144-154\n(Epitope core)', '#E91E8C',  -6.0,  0.5,  4.0),
    (185, 'alpha-H2  172-194',                 '#00C896',   6.5, -1.5,  0.5),
    (212, 'alpha-H3 / GlobC\n200-223',         '#9C27B0',   6.0,  3.5,  0.5),
    (130, 'beta1  128-131',                    '#FF8C00',  -5.0, -3.0,  3.0),
    (162, 'beta2  161-163',                    '#FF8C00',  -5.5,  2.5,  1.5),
]
for rn_t, lbl, col, dx, dy, dz in annotations:
    pt = pt_at(rn_t)
    ax.text(pt[0]+dx, pt[1]+dy, pt[2]+dz, lbl,
            fontsize=7.5, color=col, fontweight='bold',
            va='center', ha='center', zorder=25,
            bbox=bbox_kw(col))
    ax.plot([pt[0], pt[0]+dx*0.5], [pt[1], pt[1]+dy*0.5],
            [pt[2], pt[2]+dz*0.5], '-', color=col, lw=0.8, alpha=0.5)

# Epitope star label
ax.text(epi_cen[0]-7, epi_cen[1]+1, epi_cen[2]+5,
        '  ICSM18 / Pritamab\n  Binding Epitope\n  Res 142-170\n  Kd ~ 0.5 nM',
        fontsize=9.5, color='#FFD700', fontweight='bold', zorder=30,
        bbox=dict(boxstyle='round,pad=0.28',fc='#1A1206',ec='#FFD700',alpha=0.96,lw=1.5))
ax.plot([epi_cen[0], epi_cen[0]-5.5],
        [epi_cen[1], epi_cen[1]+0.8],
        [epi_cen[2], epi_cen[2]+4.2], '-', color='#FFD700', lw=1.3, alpha=0.7)

# Pritamab Fab labels
cent_H = sm_H[len(sm_H)//4]
ax.text(cent_H[0]+6, cent_H[1]+4, cent_H[2]-1,
        'Pritamab  Fab\nHeavy Chain (VH)',
        fontsize=9, color='#90CAF9', fontweight='bold', zorder=25,
        bbox=dict(boxstyle='round,pad=0.22',fc='#060D1A',ec='#1565C0',alpha=0.94,lw=1.2))

cent_L = sm_L[len(sm_L)//4]
ax.text(cent_L[0]-6, cent_L[1]-4, cent_L[2]+1,
        'Pritamab  Fab\nLight Chain (VL)',
        fontsize=9, color='#80DEEA', fontweight='bold', zorder=25,
        bbox=dict(boxstyle='round,pad=0.22',fc='#060D1A',ec='#00ACC1',alpha=0.94,lw=1.2))

# ── Axis / view ──────────────────────────────────────────────────────
ax.set_axis_off()
ax.set_box_aspect([1.1, 1.2, 1.5])
ax.view_init(elev=15, azim=-55)

# Auto-scale to data extent
all_pts = np.vstack([sm_P, sm_H, sm_L])
lo, hi  = all_pts.min(axis=0), all_pts.max(axis=0)
mid3    = (lo+hi)/2
rng3    = (hi-lo).max()/2 * 1.15
ax.set_xlim(mid3[0]-rng3, mid3[0]+rng3)
ax.set_ylim(mid3[1]-rng3, mid3[1]+rng3)
ax.set_zlim(mid3[2]-rng3, mid3[2]+rng3)

# ── Title ────────────────────────────────────────────────────────────
fig.text(0.50, 0.975, 'PrPC - Pritamab Antibody Binding Complex',
         ha='center', va='top', fontsize=16, fontweight='bold', color='white')
fig.text(0.50, 0.948,
         'Crystal Structure: PDB 2W9E  |  X-ray 2.1 Ang  |  '
         'Human PrP (Res 119-231) + Fab Fragment (ICSM18)',
         ha='center', va='top', fontsize=9, color='#7A96BB', style='italic')
fig.text(0.50, 0.924,
         'Epitope: Res 142-170  |  DeltaG_bind = -13.0 kcal/mol'
         '  |  Kd ~ 0.5 nM  |  MM-GBSA / Eyring TST',
         ha='center', va='top', fontsize=9, color='#FFD700')

# ── Legend ───────────────────────────────────────────────────────────
leg = [
    mpatches.Patch(color='#E91E8C', label='PrPC alpha-H1 (144-154) [Epitope]'),
    mpatches.Patch(color='#00C896', label='PrPC alpha-H2 (172-194)'),
    mpatches.Patch(color='#9C27B0', label='PrPC alpha-H3 / GlobC (200-223)'),
    mpatches.Patch(color='#FF8C00', label='PrPC beta-sheet (b1, b2)'),
    mpatches.Patch(color='#B0BEC5', label='PrPC Loop / Coil'),
    mpatches.Patch(color='#FF2222', label='Binding Epitope (142-170)'),
    mpatches.Patch(color='#1565C0', label='Fab Heavy Chain VH (Pritamab)'),
    mpatches.Patch(color='#00ACC1', label='Fab Light Chain VL (Pritamab)'),
    mpatches.Patch(color='#FFD700', label='Interface contact zone'),
]
ax.legend(handles=leg, loc='lower left', fontsize=7.8,
          framealpha=0.88, facecolor='#0A0E1A',
          edgecolor='#2A3A55', labelcolor='white',
          bbox_to_anchor=(-0.02, -0.01), ncol=2, handlelength=1.4)

plt.tight_layout(rect=[0,0,1,0.92])
fig.savefig(OUT_PATH, dpi=300, bbox_inches='tight', facecolor='#060A14')
plt.close(fig)
print(f"Saved: {OUT_PATH}")
