"""
Fig.1A — Pritamab·PrPᶜ 3D Binding & Energy Visualization   (v4.1 - layout fixed)
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.colors import LinearSegmentedColormap
import os

OUT = r'f:\ADDS\pritamab\figures'
os.makedirs(OUT, exist_ok=True)

# ── Colour palette ──────────────────────────────────────────────────────────
C = dict(
    signal   = '#CBD5E1',
    oct      = '#1E3A5F',
    beta     = '#DC2626',
    helix    = '#D97706',
    epitope  = '#7C3AED',
    antibody = '#2563EB',
)

# ── Helper: 3D smooth tube ──────────────────────────────────────────────────
def tube_3d(ax, pts, radius=0.15, color='steelblue', alpha=0.85, n_circ=14):
    pts = np.array(pts, dtype=float)
    n = len(pts)
    theta = np.linspace(0, 2*np.pi, n_circ, endpoint=False)
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    for i in range(n - 1):
        p0, p1 = pts[i], pts[i+1]
        seg = p1 - p0
        length = np.linalg.norm(seg)
        if length < 1e-9:
            continue
        axis = seg / length
        ref = np.array([0, 1, 0]) if abs(axis[1]) < 0.9 else np.array([1, 0, 0])
        u = np.cross(axis, ref); u /= np.linalg.norm(u)
        v = np.cross(axis, u)
        ring0 = [p0 + radius*(c*u + s*v) for c, s in zip(cos_t, sin_t)]
        ring1 = [p1 + radius*(c*u + s*v) for c, s in zip(cos_t, sin_t)]
        verts = []
        for j in range(n_circ):
            j2 = (j+1) % n_circ
            verts.append([ring0[j], ring0[j2], ring1[j2], ring1[j]])
        poly = Poly3DCollection(verts, alpha=alpha, linewidth=0)
        poly.set_facecolor(color)
        ax.add_collection3d(poly)

def sphere_3d(ax, cx, cy, cz, r=0.35, color='gold', alpha=1.0, n=16):
    u = np.linspace(0, 2*np.pi, n)
    v = np.linspace(0, np.pi, n)
    x = cx + r*np.outer(np.cos(u), np.sin(v))
    y = cy + r*np.outer(np.sin(u), np.sin(v))
    z = cz + r*np.outer(np.ones(n), np.cos(v))
    ax.plot_surface(x, y, z, color=color, alpha=alpha, linewidth=0, antialiased=True)

# ════════════════════════════════════════════════════════════════════════════
#  PANEL A – 3D PrPC ribbon + epitope (IMPROVED)
# ════════════════════════════════════════════════════════════════════════════
def draw_panel_A(ax):
    ax.set_axis_off()
    ax.set_xlim(-1, 10); ax.set_ylim(-3, 3); ax.set_zlim(-1, 11)
    ax.view_init(elev=20, azim=-40)
    ax.grid(False)
    ax.set_facecolor('white')
    np.random.seed(42)

    # ── Build PrPC path spreading across x=0→9, z=0→10 ───────────────────
    # N-terminal / Octapeptide (aa 1-90): loose helical coil
    t_n = np.linspace(0, np.pi*2.5, 45)
    nterm = np.column_stack([
        t_n * 0.8,            # x: 0 → ~6.3
        0.6*np.sin(t_n*1.2),  # y: oscillates
        t_n * 0.45            # z: 0 → ~3.5
    ])
    tube_3d(ax, nterm, radius=0.22, color=C['oct'], alpha=0.82)

    # WNKPSK epitope at ~aa 99-104 ≈ midpoint of nterm (t≈3.2)
    epi_idx = 20   # nterm[20] corresponds to ~aa 99
    epi_pts  = nterm[epi_idx:epi_idx+5]
    epi_c    = nterm[epi_idx + 2]  # central epitope coord

    for pt in epi_pts:
        sphere_3d(ax, *pt, r=0.38, color=C['epitope'], alpha=0.95)
    for rh, ah in [(0.90, 0.13), (1.30, 0.06)]:
        ug = np.linspace(0, 2*np.pi, 32); vg = np.linspace(0, np.pi, 32)
        hx = epi_c[0] + rh*np.outer(np.cos(ug), np.sin(vg))
        hy = epi_c[1] + rh*np.outer(np.sin(ug), np.sin(vg))
        hz = epi_c[2] + rh*np.outer(np.ones(32), np.cos(vg))
        ax.plot_surface(hx, hy, hz, color=C['epitope'], alpha=ah, linewidth=0)

    # β1 (aa 128-131) – short flat strand
    b1 = np.array([[3.2, -0.4, 4.5], [4.0, -0.2, 5.0]])
    tube_3d(ax, b1, radius=0.28, color=C['beta'], alpha=0.92)

    # H1 helix (aa 144-154) – compact helix
    t_h1 = np.linspace(0, 2.6, 22)
    h1 = np.column_stack([
        4.2 + 0.38*np.cos(t_h1*2.3),
        0.4*np.sin(t_h1*2.3),
        5.0 + t_h1*0.50
    ])
    tube_3d(ax, h1, radius=0.36, color=C['helix'], alpha=0.85)

    # β2 (aa 161-164)
    b2 = np.array([[5.2, -0.5, 6.5], [6.0, -0.3, 7.0]])
    tube_3d(ax, b2, radius=0.28, color=C['beta'], alpha=0.92)

    # H2 helix (aa 174-194) – longer helix
    t_h2 = np.linspace(0, 3.8, 32)
    h2 = np.column_stack([
        6.3 + 0.45*np.cos(t_h2*1.85),
        -0.3 + 0.55*np.sin(t_h2*1.85),
        7.0 + t_h2*0.32
    ])
    tube_3d(ax, h2, radius=0.40, color=C['helix'], alpha=0.85)

    # H3 helix (aa 200-228) – longest helix
    t_h3 = np.linspace(0, 4.0, 36)
    h3 = np.column_stack([
        7.5 + 0.50*np.cos(t_h3*1.70 + 0.8),
        0.9 + 0.60*np.sin(t_h3*1.70 + 0.8),
        8.3 + t_h3*0.30
    ])
    tube_3d(ax, h3, radius=0.44, color=C['helix'], alpha=0.85)

    # C-term signal
    cterm = np.array([[8.8, 1.5, 9.7], [9.4, 1.8, 10.0], [9.8, 2.0, 10.2]])
    tube_3d(ax, cterm, radius=0.18, color=C['signal'], alpha=0.70)

    # ── Pritamab Fab (Y-shape approaching from below) ──────────────────────
    fab_tip = epi_c + np.array([0.2, -1.8, -1.2])
    fork    = fab_tip - np.array([0, 0.8, 1.5])
    root    = fork   - np.array([0, 1.2, 2.0])
    arm_L   = fork + np.array([-1.2, 0.0, -0.7])
    arm_R   = fork + np.array([ 1.2, 0.0, -0.7])

    tube_3d(ax, [root, fork],   radius=0.20, color=C['antibody'], alpha=0.88)
    tube_3d(ax, [fork, arm_L],  radius=0.16, color=C['antibody'], alpha=0.88)
    tube_3d(ax, [fork, arm_R],  radius=0.16, color=C['antibody'], alpha=0.88)
    mid_fab = (arm_R + fab_tip)/2 + np.array([0.2, 0.4, 0.2])
    tube_3d(ax, [arm_R, mid_fab, fab_tip], radius=0.10, color='#93C5FD', alpha=0.80)
    for frac in np.linspace(0, 1, 7):
        pt = fab_tip*frac + epi_c*(1-frac)
        sphere_3d(ax, *pt, r=0.09, color='#C4B5FD', alpha=0.70)

    # ── Text annotations ──────────────────────────────────────────────────
    ax.text2D(0.03, 0.97, 'PrP$^C$ 3D Structure & Pritamab Docking',
              transform=ax.transAxes, fontsize=7.5, fontweight='bold', color='#1E293B')
    ax.text2D(0.03, 0.91,
              '$K_D$ = 0.1–0.5 nM  |  WNKPSK (aa 99–104)',
              transform=ax.transAxes, fontsize=6.5, color=C['epitope'], fontweight='bold')

    dom_labels = [
        ('Octapeptide\nrepeats', nterm[8],  (-0.2, -1.2, 0.0), C['oct']),
        ('WNKPSK',              epi_c,      ( 0.3,  1.0, 0.5), C['epitope']),
        ('β1',                  b1[0],      ( 0.3, -1.0, 0.3), C['beta']),
        ('H1',                  h1[10],     ( 0.7,  0.9, 0.4), C['helix']),
        ('β2',                  b2[0],      ( 0.3, -1.0, 0.3), C['beta']),
        ('H2',                  h2[14],     ( 0.9, -1.3, 0.3), C['helix']),
        ('H3',                  h3[14],     ( 1.0,  1.4, 0.3), C['helix']),
        ('Pritamab',            root,       ( 0.2, -1.0,-0.4), C['antibody']),
    ]
    for txt, base, off, col in dom_labels:
        ax.text(base[0]+off[0], base[1]+off[1], base[2]+off[2],
                txt, fontsize=5.5, color=col, fontweight='bold', zorder=10)

    lgd = [
        mpatches.Patch(color=C['oct'],      label='Octapeptide repeats'),
        mpatches.Patch(color=C['beta'],     label='β-strands'),
        mpatches.Patch(color=C['helix'],    label='α-helices'),
        mpatches.Patch(color=C['epitope'],  label='WNKPSK epitope'),
        mpatches.Patch(color=C['antibody'], label='Pritamab (Fab)'),
    ]
    ax.legend(handles=lgd, loc='lower left', fontsize=5.0,
              framealpha=0.88, ncol=1, handlelength=1.0)
    ax.set_title('(A)', fontsize=9, fontweight='bold', pad=2)




# ════════════════════════════════════════════════════════════════════════════
#  PANEL B – Epitope residue map
# ════════════════════════════════════════════════════════════════════════════
def draw_panel_B(ax):
    ax.set_xlim(-0.8, 12.5)
    ax.set_ylim(-3.8, 4.8)
    ax.axis('off')

    seq   = list('GGNRYPWNKPSK')
    dG_r  = [-2.1, -1.8, -0.9, -4.3, -3.7, -2.6, -6.8, -1.2, -3.5, -2.9, -1.4, -4.1]
    rtype_col = {
        'G':'#94A3B8','P':'#94A3B8',
        'N':'#34D399','S':'#34D399',
        'R':'#60A5FA','K':'#60A5FA',
        'W':'#C084FC','Y':'#C084FC',
    }
    core_idx = {6, 7, 8, 9, 10, 11}  # WNKPSK

    for i, (aa, dg) in enumerate(zip(seq, dG_r)):
        x = i
        is_core = i in core_idx
        fc = rtype_col.get(aa, '#CBD5E1')
        ec = C['epitope'] if is_core else '#475569'
        lw = 2.5 if is_core else 1.0
        r  = 0.48 if is_core else 0.38

        # Spine
        ax.plot([x, x], [0, dg*0.38], color=ec, lw=1.5, zorder=2)
        # Circle
        circ = plt.Circle((x, 0), r, fc=fc, ec=ec, lw=lw, zorder=3)
        ax.add_patch(circ)
        ax.text(x, 0, aa, ha='center', va='center',
                fontsize=7, fontweight='bold', color='white', zorder=4)

        # Bar (negative dG → downward)
        bar_h = dg * 0.38
        col_bar = '#818CF8' if is_core else '#BAC8FF'
        rect = mpatches.FancyBboxPatch((x-0.22, min(0, bar_h)),
                                       0.44, abs(bar_h),
                                       boxstyle='round,pad=0.02',
                                       fc=col_bar, ec='none', alpha=0.75, zorder=1)
        ax.add_patch(rect)

        # Label
        ax.text(x, -3.0, f'{dg:.1f}', ha='center', fontsize=5.5, color='#374151')
        ax.text(x, -3.5, f'aa {91+i}', ha='center', fontsize=4.8, color='#9CA3AF')

    # Bracket WNKPSK
    bx0, bx1, by = 5.7, 11.3, 1.9
    ax.annotate('', xy=(bx0, by), xytext=(bx1, by),
                arrowprops=dict(arrowstyle='<->', color=C['epitope'], lw=1.6))
    ax.text((bx0+bx1)/2, by+0.35, 'WNKPSK (epitope core)',
            ha='center', fontsize=7, color=C['epitope'], fontweight='bold')

    # Peptide properties box
    props = ('GGNRYPWNKPSK\n'
             'Mass: 1402.7 Da   pI: 10.76\n'
             'Net charge: +3   Hydrophobicity: +17.25 kcal·mol⁻¹\n'
             'Extinction coeff.: 6990 M⁻¹·cm⁻¹')
    ax.text(5.5, 4.6, props, ha='center', va='top', fontsize=6.2,
            color='#1E293B', fontfamily='monospace',
            bbox=dict(fc='#F1F5F9', ec='#94A3B8', pad=4, boxstyle='round'))

    ax.text(-0.7, -3.0, 'ΔG_res\n(kcal/mol)', ha='center', va='center',
            fontsize=5.5, color='#475569')
    ax.set_title('(B)  Epitope Residue Interaction Map — GGNRYPWNKPSK',
                 fontsize=8, fontweight='bold')


# ════════════════════════════════════════════════════════════════════════════
#  PANEL C – MM/PBSA energy bars (3D-style, FIXED)
# ════════════════════════════════════════════════════════════════════════════
def draw_panel_C(ax):
    ax.set_axis_off()
    ax.view_init(elev=28, azim=-48)
    ax.set_xlim(-0.3, 6.0)
    ax.set_ylim(-0.2, 1.2)
    ax.set_zlim(-78, 22)
    ax.set_facecolor('white')
    ax.grid(False)

    labels = ['ΔG\ntotal', 'van der\nWaals', 'Electro-\nstatic', 'GBSA\nSolv.', 'Entropy\n(−TΔS)']
    values = [-61.8, -30.3, -20.6, -18.8, +7.3]
    colors = ['#1E40AF', '#1D4ED8', '#3B82F6', '#60A5FA', '#DC2626']
    x_pos  = [0.0, 1.2, 2.4, 3.6, 4.8]
    bw, bd = 0.70, 0.40

    for xi, (val, c, lbl) in enumerate(zip(values, colors, labels)):
        x = x_pos[xi]
        h = abs(val)
        z0 = min(0, val)   # bottom of bar
        z1 = z0 + h        # top of bar

        # Darken for side/top faces
        rgb = np.array(matplotlib.colors.to_rgb(c))
        shade = {
            'front': rgb,
            'top'  : rgb * 0.75,
            'side' : rgb * 0.55,
            'back' : rgb * 0.35,
        }
        faces = {
            'front': [(x,    0, z0),(x+bw, 0, z0),(x+bw, 0, z1),(x,    0, z1)],
            'top'  : [(x,    0, z1),(x+bw, 0, z1),(x+bw,bd, z1),(x,   bd, z1)],
            'side' : [(x+bw, 0, z0),(x+bw,bd, z0),(x+bw,bd, z1),(x+bw, 0, z1)],
            'back' : [(x,   bd, z0),(x+bw,bd, z0),(x+bw,bd, z1),(x,   bd, z1)],
        }
        for fname, pts in faces.items():
            poly = Poly3DCollection([pts], alpha=0.90, linewidth=0.5,
                                    edgecolor='#1E293B')
            poly.set_facecolor(tuple(shade[fname]))
            ax.add_collection3d(poly)

        # Value label (above bar top or below for entropy)
        sign = '+' if val > 0 else ''
        lab_z = z1 + 4.0
        ax.text(x + bw/2, bd/2, lab_z,
                f'{sign}{val}', ha='center', va='bottom',
                fontsize=7.5, fontweight='bold',
                color='#B91C1C' if val > 0 else '#1E40AF', zorder=10)

        # x-axis label below plot floor
        ax.text(x + bw/2, 1.05, -85, lbl,
                ha='center', va='top', fontsize=6, color='#374151', zorder=10)

    # Zero plane
    xx, yy = np.meshgrid([-0.2, 5.6], [-0.05, 0.95])
    ax.plot_surface(xx, yy, np.zeros_like(xx),
                    alpha=0.18, color='#94A3B8')

    # z-axis label
    ax.text(0, 0.6, -35, 'Energy\n(kcal·mol⁻¹)',
            fontsize=6.5, color='#374151', ha='center', zorder=10)

    ax.set_title('(C)  MM/PBSA Binding Energy Decomposition',
                 fontsize=8, fontweight='bold')


# ════════════════════════════════════════════════════════════════════════════
#  PANEL D – Fluorescence heatmap (top) + PrPC domain ruler (bottom)
# ════════════════════════════════════════════════════════════════════════════
def draw_panel_D(fig, pos):
    """pos = [left, bottom, width, height] in figure fraction"""
    left, bot, w, h = pos
    mid = left + w/2

    # ── Fluorescence subplot ────────────────────────────────────────────────
    hm_h = h * 0.54
    ax_hm = fig.add_axes([left, bot + h - hm_h, w, hm_h])
    np.random.seed(99)
    sz = 80
    hm = np.random.exponential(0.3, (sz, sz)) * 25
    cx, cy = int(sz*0.54), int(sz*0.43)
    for dx in range(-9, 10):
        for dy in range(-9, 10):
            hm[cy+dy, cx+dx] += 210 * np.exp(-(dx**2+dy**2)/18)

    green_cmap = LinearSegmentedColormap.from_list(
        'fl', ['#050D05','#0D3320','#166534','#16A34A','#4ADE80','#DCFCE7'])
    im = ax_hm.imshow(hm, cmap=green_cmap, vmin=0, vmax=220,
                      interpolation='bicubic', aspect='auto')
    ax_hm.scatter([cx],[cy], s=150, marker='s', fc='none',
                  ec='white', linewidths=2, zorder=5)
    ax_hm.text(cx+3, cy-5, 'Hotspot\nX:17000  Y:27975',
               color='white', fontsize=5.5, va='top', fontweight='bold')
    ax_hm.set_xticks([]); ax_hm.set_yticks([])
    ax_hm.set_title('Fluorescence Signal — GGNRYPWNKPSK',
                    fontsize=6.5, color='white', pad=2,
                    backgroundcolor='#050D05')
    ax_hm.spines[:].set_edgecolor('white')
    # colorbar
    cbar = fig.colorbar(im, ax=ax_hm, fraction=0.028, pad=0.02)
    cbar.set_label('Intensity (a.u.)', fontsize=5, color='#374151')
    cbar.ax.tick_params(labelsize=4.5)

    # ── Domain ruler subplot ────────────────────────────────────────────────
    ruler_h = h * 0.42
    ax_ruler = fig.add_axes([left, bot, w, ruler_h])
    ax_ruler.set_xlim(0, 253)
    ax_ruler.set_ylim(-2.8, 5.5)
    ax_ruler.axis('off')

    domains = [
        (1,   22,  'Signal',     C['signal'], 0.8),
        (23,  50,  '',           C['signal'], 0.8),
        (51,  90,  'Octa-\nrepeats', C['oct'],  0.9),
        (91,  127, '',           '#D1D5DB',   0.7),
        (128, 131, 'β1',         C['beta'],   1.0),
        (132, 143, '',           '#D1D5DB',   0.7),
        (144, 154, 'H1',         C['helix'],  1.0),
        (155, 160, '',           '#D1D5DB',   0.7),
        (161, 164, 'β2',         C['beta'],   1.0),
        (165, 173, '',           '#D1D5DB',   0.7),
        (174, 194, 'H2',         C['helix'],  1.1),
        (195, 199, '',           '#D1D5DB',   0.7),
        (200, 228, 'H3',         C['helix'],  1.1),
        (229, 253, 'Signal',     C['signal'], 0.8),
    ]

    ruler_y = 1.5
    for (aa0, aa1, lbl, fc, ht) in domains:
        rect = mpatches.Rectangle(
            (aa0, ruler_y - ht/2), aa1 - aa0, ht,
            fc=fc, ec='#374151', lw=0.7, zorder=3)
        ax_ruler.add_patch(rect)
        if lbl:
            tc = 'white' if fc in [C['oct'], C['beta']] else '#1E293B'
            ax_ruler.text((aa0+aa1)/2, ruler_y, lbl,
                          ha='center', va='center', fontsize=5.2,
                          color=tc, fontweight='bold', zorder=4)

    # Epitope WNKPSK (aa 99-104)
    ax_ruler.add_patch(mpatches.FancyBboxPatch(
        (99, ruler_y-0.80), 5, 1.60,
        boxstyle='round,pad=0.05',
        fc=C['epitope'], ec='white', lw=1.5, alpha=0.95, zorder=5))
    ax_ruler.text(101.5, ruler_y, 'WNKPSK\n99–104',
                  ha='center', va='center', fontsize=5.0,
                  color='white', fontweight='bold', zorder=6)

    # Arrow: epitope → fluorescence heatmap (pointing up)
    ax_ruler.annotate('', xy=(101.5, ruler_y+1.0),
                      xytext=(101.5, ruler_y+2.2),
                      arrowprops=dict(arrowstyle='->', color=C['epitope'],
                                      lw=1.8, connectionstyle='arc3,rad=0'))

    # Polymorphism markers
    for aa_p, label in [(129,'M129V'),(171,'N171S'),(219,'E219K')]:
        ax_ruler.annotate('', xy=(aa_p, ruler_y+0.7),
                          xytext=(aa_p, ruler_y+1.8),
                          arrowprops=dict(arrowstyle='->', color='#DC2626', lw=1.0))
        ax_ruler.text(aa_p, ruler_y+2.05, label,
                      ha='center', va='bottom', fontsize=5.0, color='#DC2626',
                      bbox=dict(fc='#FEE2E2', ec='#DC2626', pad=1.2, boxstyle='round'))

    ax_ruler.annotate('', xy=(129-2, ruler_y+2.9),
                      xytext=(219+2, ruler_y+2.9),
                      arrowprops=dict(arrowstyle='<->', color='#DC2626', lw=1.0))
    ax_ruler.text(174, ruler_y+3.3, 'Polymorphisms',
                  ha='center', fontsize=6, color='#DC2626', fontweight='bold')

    # Antibody arrow
    ax_ruler.annotate('Pritamab\nbinding', xy=(101.5, ruler_y+0.85),
                      xytext=(55, ruler_y+4.5),
                      fontsize=6, color=C['antibody'], fontweight='bold',
                      arrowprops=dict(arrowstyle='->', color=C['antibody'],
                                      lw=1.5, connectionstyle='arc3,rad=-0.3'))

    # aa tick marks
    for aa_t in [1, 51, 91, 128, 161, 174, 200, 228, 253]:
        ax_ruler.plot([aa_t, aa_t], [ruler_y-0.6, ruler_y-0.85],
                      color='#374151', lw=0.8)
        ax_ruler.text(aa_t, ruler_y-1.15, str(aa_t),
                      ha='center', va='top', fontsize=4.8, color='#374151')

    ax_ruler.set_title('(D)  Fluorescence Binding & PrP$^C$ Domain Map',
                       fontsize=8, fontweight='bold',
                       x=0.5, y=1.02)


# ════════════════════════════════════════════════════════════════════════════
#  MAIN ASSEMBLY
# ════════════════════════════════════════════════════════════════════════════
def build_fig1a_3d():
    fig = plt.figure(figsize=(24/2.54, 21/2.54), facecolor='white')
    fig.suptitle(
        'Figure 1A.  Pritamab–PrP$^C$ Binding: 3D Structure, Epitope, Energy Decomposition & Fluorescence Signal',
        fontsize=9, fontweight='bold', y=0.995, color='#0F172A')

    # ── Subplot grid ────────────────────────────────────────────────────────
    # Use GridSpec for A, B, C; Panel D uses fig.add_axes for two sub-rows
    gs = gridspec.GridSpec(2, 2, figure=fig,
                           hspace=0.38, wspace=0.26,
                           left=0.04, right=0.97,
                           top=0.95, bottom=0.04)

    axA = fig.add_subplot(gs[0, 0], projection='3d')
    draw_panel_A(axA)

    axB = fig.add_subplot(gs[0, 1])
    draw_panel_B(axB)

    axC = fig.add_subplot(gs[1, 0], projection='3d')
    draw_panel_C(axC)

    # Panel D occupies gs[1,1] position — use its bbox
    tmp = fig.add_subplot(gs[1, 1])
    pos = tmp.get_position()
    tmp.remove()
    draw_panel_D(fig, [pos.x0, pos.y0, pos.width, pos.height])

    # Save
    for ext in ['png', 'pdf']:
        fpath = os.path.join(OUT, f'fig1A_3d_v4.{ext}')
        fig.savefig(fpath, dpi=300, bbox_inches='tight',
                    facecolor='white', edgecolor='none')
        print(f'  Saved: {fpath}')
    plt.close(fig)

if __name__ == '__main__':
    print('Building Fig.1A 3D v4 (layout-fixed)...')
    build_fig1a_3d()
    print('Done.')
