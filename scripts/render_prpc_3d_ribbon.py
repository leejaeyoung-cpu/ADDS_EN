"""
render_prpc_3d_ribbon.py
========================
PDB 1HJM (Human PrPC 121-231) → Ribbon-style 3D rendering
Biopython  +  matplotlib  +  mpl_toolkits.mplot3d

Output: f:\ADDS\outputs\pritamab_pptx_figures\prpc_3d_ribbon.png
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection
import numpy as np
import os, urllib.request, io

# ── Output ──────────────────────────────────────────────────────────
OUT_DIR  = r"f:\ADDS\outputs\pritamab_pptx_figures"
OUT_PATH = os.path.join(OUT_DIR, "prpc_3d_ribbon.png")
os.makedirs(OUT_DIR, exist_ok=True)

# ── Download PDB 1HJM ───────────────────────────────────────────────
PDB_ID  = "1HJM"
PDB_URL = f"https://files.rcsb.org/download/{PDB_ID}.pdb"
PDB_LOCAL = os.path.join(OUT_DIR, f"{PDB_ID}.pdb")
if not os.path.exists(PDB_LOCAL):
    print(f"Downloading {PDB_ID}...")
    urllib.request.urlretrieve(PDB_URL, PDB_LOCAL)
    print("Done.")
else:
    print(f"Using cached {PDB_LOCAL}")

# ── Parse with Biopython ─────────────────────────────────────────────
from Bio.PDB import PDBParser, PPBuilder, DSSP
from Bio.PDB.DSSP import dssp_dict_from_pdb_file

parser = PDBParser(QUIET=True)
structure = parser.get_structure(PDB_ID, PDB_LOCAL)
model     = structure[0]
chain     = model['A']

# Extract Cα coordinates + residue numbers
ca_coords = []
res_nums  = []
for res in chain.get_residues():
    if res.get_id()[0] != ' ':
        continue
    if 'CA' in res:
        ca_coords.append(res['CA'].get_vector().get_array())
        res_nums.append(res.get_id()[1])

ca_coords = np.array(ca_coords, dtype=float)
res_nums  = np.array(res_nums)

# ── Secondary structure assignment (DSSP or manual for 1HJM) ────────
# 1HJM known SS for human PrPC glob domain (residues 125-228):
# Helix α-H1: 144-154   (red/crimson)
# Helix α-H2: 172-193   (green, epitope region 144-179 overlaps)
# Helix α-H3: 200-228   (purple)
# Beta β1: 128-131      (orange)
# Beta β2: 161-163      (orange)
# Rest: loop/coil       (gray)

H1_range  = (144, 154)
H2_range  = (172, 193)
H3_range  = (200, 228)
B1_range  = (128, 131)
B2_range  = (161, 163)
EPITOPE   = (144, 179)  # Pritamab binding site

def ss_color(rn):
    """Return color string by secondary structure category."""
    if H1_range[0] <= rn <= H1_range[1]:  return '#E91E8C'   # magenta α-H1
    if H2_range[0] <= rn <= H2_range[1]:  return '#00C896'   # cyan-green α-H2
    if H3_range[0] <= rn <= H3_range[1]:  return '#9C27B0'   # purple α-H3
    if B1_range[0] <= rn <= B1_range[1]:  return '#FF8C00'   # orange β1
    if B2_range[0] <= rn <= B2_range[1]:  return '#FF8C00'   # orange β2
    return '#A0AABA'                                           # loop gray

# ── Smoothing spline through Cα trace ───────────────────────────────
from scipy.interpolate import splprep, splev

def smooth_trace(pts, n_out=600):
    tck, u = splprep(pts.T, s=2.0, k=3)
    u_new  = np.linspace(0, 1, n_out)
    return np.array(splev(u_new, tck)).T

# ── Build ribbon tube (fake 3D ribbon via perpendicular offsets) ─────
def ribbon_quads(smooth, width=0.4):
    """Generate ribbon quads from smooth Cα trace."""
    n   = len(smooth)
    # Tangent
    tang = np.gradient(smooth, axis=0)
    tang /= (np.linalg.norm(tang, axis=1, keepdims=True) + 1e-9)
    # Reference up vector
    up = np.array([0, 0, 1.0])
    # Normal (perpendicular to tangent in plane with up)
    normals = np.cross(tang, up)
    nlen    = np.linalg.norm(normals, axis=1, keepdims=True) + 1e-9
    normals /= nlen
    left  = smooth + normals * width / 2
    right = smooth - normals * width / 2
    return left, right

# ── Figure setup ─────────────────────────────────────────────────────
fig = plt.figure(figsize=(10, 9), facecolor='#0A0A14', dpi=200)
ax  = fig.add_subplot(111, projection='3d', facecolor='#0A0A14')

# Center coordinates
ca_centered = ca_coords - ca_coords.mean(axis=0)

# Smooth full trace
smooth_all = smooth_trace(ca_centered, n_out=800)

# Map smoothed parameter → residue index using u-interpolation
u_res = np.linspace(0, 1, len(res_nums))
u_smo = np.linspace(0, 1, len(smooth_all))

def rnum_at(i_smo):
    """Approximate residue number at smooth index i."""
    u = u_smo[i_smo]
    idx = np.searchsorted(u_res, u)
    idx = np.clip(idx, 0, len(res_nums)-1)
    return res_nums[idx]

# ── Draw ribbon segments coloured by SS ─────────────────────────────
left_s, right_s = ribbon_quads(smooth_all, width=0.55)

# Group consecutive points by color
seg_colors = [ss_color(rnum_at(i)) for i in range(len(smooth_all))]

# Draw tube as quads
verts_list = []
col_list   = []
for i in range(len(smooth_all) - 1):
    p0l, p0r = left_s[i],   right_s[i]
    p1l, p1r = left_s[i+1], right_s[i+1]
    verts_list.append([p0l, p0r, p1r, p1l])
    col_list.append(seg_colors[i])

# Draw in batches by color for efficiency
poly = Poly3DCollection(verts_list, zsort='average',
                         linewidth=0, alpha=0.92)
poly.set_facecolors(col_list)
poly.set_edgecolors(col_list)
ax.add_collection3d(poly)

# Draw centerline (white glow effect)
ax.plot(smooth_all[:,0], smooth_all[:,1], smooth_all[:,2],
        '-', color='white', lw=0.5, alpha=0.18, zorder=1)

# ── Epitope shading (Res 144-179) ───────────────────────────────────
epi_mask = (res_nums >= EPITOPE[0]) & (res_nums <= EPITOPE[1])
epi_idx  = np.where(epi_mask)[0]
if len(epi_idx):
    # Map to smooth indices
    u_epi_lo = u_res[epi_idx[0]]
    u_epi_hi = u_res[epi_idx[-1]]
    i_lo = np.searchsorted(u_smo, u_epi_lo)
    i_hi = np.searchsorted(u_smo, u_epi_hi)
    epi_pts = smooth_all[i_lo:i_hi]
    # Draw glowing epitope tube
    ax.plot(epi_pts[:,0], epi_pts[:,1], epi_pts[:,2],
            '-', color='#FFD700', lw=4.5, alpha=0.55, zorder=5)
    ax.plot(epi_pts[:,0], epi_pts[:,1], epi_pts[:,2],
            '-', color='#FF4444', lw=2.0, alpha=0.90, zorder=6)

# ── Label key features ──────────────────────────────────────────────
def label_at(rn, text, col, offset=(0.5, 0.5, 0.5), fs=9, zorder=10):
    """Place 3D text label near a given residue."""
    idx_arr = np.where(res_nums == rn)[0]
    if len(idx_arr) == 0:
        return
    # Find smooth point near this residue
    u_t = u_res[idx_arr[0]]
    i_s = np.searchsorted(u_smo, u_t)
    i_s = np.clip(i_s, 0, len(smooth_all)-1)
    pt  = smooth_all[i_s]
    ax.text(pt[0]+offset[0], pt[1]+offset[1], pt[2]+offset[2],
            text, fontsize=fs, color=col, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.20', fc='#0A0A14',
                      ec=col, alpha=0.88, lw=1.1),
            zorder=zorder)
    # Connector
    ax.plot([pt[0], pt[0]+offset[0]*0.7],
            [pt[1], pt[1]+offset[1]*0.7],
            [pt[2], pt[2]+offset[2]*0.7],
            '-', color=col, lw=0.8, alpha=0.60)

label_at(149, 'M144 Epitope\n(Pritamab binds)', '#FF4444',
         offset=(3.5, -1.5, 2.0), fs=8)
label_at(144, 'α-H1  (144–154)', '#E91E8C',
         offset=(-4.0, -1.0, -1.0), fs=8)
label_at(182, 'α-H2  (172–193)\n[Epitope core]', '#00C896',
         offset=(4.5, 0.5, -1.0), fs=8)
label_at(214, 'α-H3 / GlobC\n(200–228)', '#9C27B0',
         offset=(4.0, 2.0, 1.0), fs=8)
label_at(130, 'β1 (128–131)', '#FF8C00',
         offset=(-3.5, -1.5, 1.5), fs=7.5)
label_at(162, 'β2 (161–163)', '#FF8C00',
         offset=(-4.0, 1.2, 0.5), fs=7.5)
label_at(165, 'Kd ≈ 0.5 nM\nΔG = −13.0 kcal/mol', '#FFD700',
         offset=(5.0, -2.0, 0.0), fs=8)

# ── Pritamab IgY schematic (arrow pointing to epitope) ─────────────
epi_center_idx = np.searchsorted(u_smo, u_res[np.where(res_nums==162)[0][0]])
ep = smooth_all[epi_center_idx]
ax.quiver(ep[0]-5.5, ep[1]+4.0, ep[2]-3.0,
          3.2, -2.5, 1.5,
          color='#FFD700', lw=2, arrow_length_ratio=0.25, zorder=8)
ax.text(ep[0]-7.0, ep[1]+5.0, ep[2]-4.0,
        'Pritamab\n(anti-PrPC IgG)', fontsize=8.5, color='#FFD700',
        fontweight='bold', zorder=9,
        bbox=dict(boxstyle='round,pad=0.22', fc='#1A1130',
                  ec='#FFD700', alpha=0.92, lw=1.3))

# ── Axes / style ─────────────────────────────────────────────────────
ax.set_axis_off()
ax.set_box_aspect([1, 1, 1.4])

# Elevation / azimuth for best view of helix stack
ax.view_init(elev=25, azim=-45)

# Title
fig.text(0.5, 0.96,
         'PrPC Globular Domain — Pritamab Binding Site',
         ha='center', va='top', fontsize=14, fontweight='bold',
         color='white')
fig.text(0.5, 0.925,
         'PDB: 1HJM  ·  Ribbon: Biopython + matplotlib  ·  '
         'Epitope Res 144–179 (highlighted)',
         ha='center', va='top', fontsize=9, color='#8899AA', style='italic')

# Legend
legend_items = [
    mpatches.Patch(color='#E91E8C', label='α-Helix 1  (144–154)'),
    mpatches.Patch(color='#00C896', label='α-Helix 2  (172–193)  ← Epitope'),
    mpatches.Patch(color='#9C27B0', label='α-Helix 3 / GlobC  (200–228)'),
    mpatches.Patch(color='#FF8C00', label='β-Sheet  (β1, β2)'),
    mpatches.Patch(color='#A0AABA', label='Loop / Coil'),
    mpatches.Patch(color='#FF4444', label='Pritamab Epitope  (144–179)'),
]
ax.legend(handles=legend_items,
          loc='lower left', fontsize=8,
          framealpha=0.85, facecolor='#0F0F1E',
          edgecolor='#334455', labelcolor='white',
          bbox_to_anchor=(-0.02, 0.0))

plt.tight_layout(rect=[0, 0, 1, 0.92])
fig.savefig(OUT_PATH, dpi=300, bbox_inches='tight', facecolor='#0A0A14')
plt.close(fig)
print(f"Saved → {OUT_PATH}")
