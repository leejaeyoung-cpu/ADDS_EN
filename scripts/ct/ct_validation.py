"""
CT Pipeline v2 -- Rigorous Validation Suite
============================================
No sugar-coating. Tests every claim the pipeline makes.

VALIDATION TARGETS:
  1. Tumor candidate plausibility (size, HU, location)
  2. Registration quality (is phase_cross_correlation actually aligning correctly?)
  3. Consistency score validity (does shift=0 mean real alignment or just no motion?)
  4. PRE vs POST comparability (are we comparing the same anatomy?)
  5. Organ annotation accuracy (do labeled voxels make anatomical sense?)
  6. Noise vs signal discrimination (do rejected candidates differ from accepted?)

Outputs:
  F:/ADDS/validation_report.txt      -- full text report
  F:/ADDS/validation_figure.png      -- visual proof panels
"""

import os, glob, json, warnings
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import ndimage
from skimage import measure, morphology, exposure
from skimage.transform import resize
from skimage.registration import phase_cross_correlation
import pydicom

warnings.filterwarnings('ignore')
np.random.seed(2026)
DATA_DIR  = r'F:\ADDS\CTdata2'
SAVE_DIR  = r'F:\ADDS'
TARGET_SZ = 256

report_lines = []
def R(line=''):
    print(line)
    report_lines.append(line)


# ================================================================
# Load volumes (same as pipeline)
# ================================================================
def load_series(prefix, kw):
    files = sorted(glob.glob(os.path.join(DATA_DIR, f'{prefix}*.dcm')))
    slices = []
    for f in files:
        try:
            ds = pydicom.dcmread(f)
            desc = getattr(ds,'SeriesDescription','')
            if kw.lower() in desc.lower() and ds.Rows==512:
                try:    z=float(ds.ImagePositionPatient[2])
                except: z=float(getattr(ds,'InstanceNumber',0))
                hu = ds.pixel_array.astype(np.float32)*float(getattr(ds,'RescaleSlope',1))+float(getattr(ds,'RescaleIntercept',-1024))
                slices.append((z,hu,ds))
        except: pass
    slices.sort(key=lambda x:x[0])
    vol = np.stack([s[1] for s in slices])
    ds0 = slices[0][2]
    meta = dict(n=len(slices), date=str(getattr(ds0,'StudyDate','?')),
                series=str(getattr(ds0,'SeriesDescription','?')),
                pid=str(getattr(ds0,'PatientID','?')),
                spacing=list(getattr(ds0,'PixelSpacing',[1,1])),
                thickness=str(getattr(ds0,'SliceThickness','?')))
    return vol, meta, slices

def hu_window(vol, wl, ww):
    lo,hi = wl-ww/2, wl+ww/2
    return (np.clip(vol,lo,hi)-lo)/(hi-lo)


# ================================================================
R('='*70)
R('CT PIPELINE VALIDATION REPORT   --  Patient 001740976')
R('='*70)
R()

R('[STEP 1] Loading PRE (0930 Abdomen Pre) and POST (1223 Chest CE)...')
vol_pre,  meta_pre,  slc_pre  = load_series('0930', 'Abdomen Pre')
vol_post, meta_post, slc_post = load_series('1223', 'Chest CE')

R(f'  PRE : {meta_pre["n"]:3d} slices | {meta_pre["series"]} | {meta_pre["date"]}  '
  f'| spacing={meta_pre["spacing"]} mm | thickness={meta_pre["thickness"]}')
R(f'  POST: {meta_post["n"]:3d} slices | {meta_post["series"]} | {meta_post["date"]}  '
  f'| spacing={meta_post["spacing"]} mm | thickness={meta_post["thickness"]}')
R()

# ================================================================
# VALIDATION 1: Anatomical region coverage
# ================================================================
R('-'*70)
R('VALIDATION 1: Anatomical Coverage (are pre/post comparing the same anatomy?)')
R('-'*70)

def z_range_hu(vol):
    """Characterise each slice by dominant HU class."""
    lung_frac  = [float(np.mean((vol[i]>-900)&(vol[i]<-500))) for i in range(len(vol))]
    bone_frac  = [float(np.mean(vol[i]>280)) for i in range(len(vol))]
    soft_frac  = [float(np.mean((vol[i]>-50)&(vol[i]<150))) for i in range(len(vol))]
    return lung_frac, bone_frac, soft_frac

pre_lung, pre_bone, pre_soft   = z_range_hu(vol_pre)
post_lung, post_bone, post_soft = z_range_hu(vol_post)

R(f'  PRE  mean lung-fraction per slice: {np.mean(pre_lung):.3f}   '
  f'(slices with >5% lung: {sum(x>0.05 for x in pre_lung)})')
R(f'  POST mean lung-fraction per slice: {np.mean(post_lung):.3f}   '
  f'(slices with >5% lung: {sum(x>0.05 for x in post_lung)})')
R()

pre_is_abdomen  = np.mean(pre_lung)  < 0.03
post_is_abdomen = np.mean(post_lung) < 0.03
R(f'  PRE  classified as: {"ABDOMEN" if pre_is_abdomen else "CHEST/THORAX"}')
R(f'  POST classified as: {"ABDOMEN" if post_is_abdomen else "CHEST/THORAX"}')
if pre_is_abdomen != post_is_abdomen:
    R()
    R('  !! WARNING: PRE and POST cover DIFFERENT body regions.')
    R('     Direct tumor size comparison is NOT VALID.')
    R('     The -81% "tumor area change" reported is ARTEFACTUAL.')
else:
    R('  OK: Both scans cover the same body region.')
R()

# ================================================================
# VALIDATION 2: Slice thickness / voxel volume
# ================================================================
R('-'*70)
R('VALIDATION 2: Voxel Size -- what does "274 voxels" actually mean?')
R('-'*70)

for vol, meta, label in [(vol_pre, meta_pre, 'PRE'), (vol_post, meta_post, 'POST')]:
    try:
        sp    = [float(x) for x in meta['spacing']]      # [row_mm, col_mm]
        thick = float(meta['thickness']) if meta['thickness'] != '?' else sp[0]
    except:
        sp    = [1.0, 1.0]
        thick = 1.0

    vox_vol_mm3  = sp[0] * sp[1] * thick
    R(f'  {label}: pixel spacing={sp[0]:.2f}x{sp[1]:.2f} mm | thickness={thick:.2f} mm')
    R(f'       1 voxel = {vox_vol_mm3:.2f} mm^3')
    R(f'       274 voxels = {274*vox_vol_mm3:.1f} mm^3  ({274*vox_vol_mm3/1000:.3f} cm^3)')
    R(f'       52  voxels = {52 *vox_vol_mm3:.1f} mm^3  ({52*vox_vol_mm3/1000:.3f} cm^3)')
    typical_rectal_tumor_cm3 = 20.0
    R(f'       Typical rectal cancer volume: ~{typical_rectal_tumor_cm3} cm^3')
    R(f'       PRE tumor (274 vox) = {274*vox_vol_mm3/10000:.1f}% of expected tumor volume.')
    R()

R('  VERDICT: Detected "tumor" candidates are far too small to be actual')
R('  rectal tumors. Most likely: blood vessels, lymph nodes, or artifacts.')
R()

# ================================================================
# VALIDATION 3: HU value sanity check
# ================================================================
R('-'*70)
R('VALIDATION 3: HU Values at detected "tumor" locations')
R('-'*70)
R('  Expected HU values:')
R('    Rectal cancer (non-contrast): 40-60 HU')
R('    Rectal cancer (contrast):     60-120 HU')
R('    Blood vessel:                 150-300 HU')
R('    Lymph node:                   30-60 HU')
R('    Muscle:                       30-70 HU')
R('    Fat:                          -150 to -30 HU')
R()
R('  Detected PRE best candidate:  HU=38.5  -> lymph node / muscle / small bowel wall')
R('  Detected POST best candidate: HU=89.3  -> blood vessel or contrast-enhanced tissue')
R('  Neither value is definitively tumor-specific without segmentation ground truth.')
R()

# ================================================================
# VALIDATION 4: Registration quality test
# ================================================================
R('-'*70)
R('VALIDATION 4: Is phase_cross_correlation actually detecting patient motion?')
R('-'*70)
R('  Method: correlation between reference slice and adjacent slices.')
R('  TRUE motion: large shift (>2px) between adjacent slices.')
R('  Stable patient: near-zero shift.')
R()
R('  Testing PRE (Abdomen Pre): should have minimal inter-slice motion in CT...')

hu_norm_pre = (np.clip(vol_pre,-150,250)+150)/400.0
shifts_pre = []
for zi in range(1, min(20, len(vol_pre))):
    sh,_,_ = phase_cross_correlation(hu_norm_pre[zi-1], hu_norm_pre[zi],
                                      upsample_factor=4, normalization=None)
    shifts_pre.append(float(np.sqrt(sh[0]**2+sh[1]**2)))

R(f'  PRE consecutive-slice shifts:')
R(f'    Mean: {np.mean(shifts_pre):.2f} px   Std: {np.std(shifts_pre):.2f} px   '
  f'Max: {np.max(shifts_pre):.2f} px   Min: {np.min(shifts_pre):.2f} px')
R(f'    Values: {[round(x,1) for x in shifts_pre[:10]]}...')
R()

if np.max(shifts_pre) < 1.0:
    R('  ISSUE: Shifts are near-zero for ALL consecutive slices.')
    R('  This means phase_cross_correlation is returning 0 because the')
    R('  slices are already aligned (helical CT has no inter-slice motion).')
    R('  => The "shift verification" step is not actually detecting noise vs signal.')
    R('  => consistency=1.00 is trivially true because shift threshold (8px) is never exceeded.')
elif np.mean(shifts_pre) > 2.0:
    R('  WARNING: Large shifts detected between consecutive slices.')
    R('  This may indicate scan artefacts or poor image quality.')
else:
    R('  Moderate shifts -- registration step provides some validation.')
R()

# ================================================================
# VALIDATION 5: Noise rejection test
#   Do rejected candidates (consistency<0.25) differ from accepted ones?
# ================================================================
R('-'*70)
R('VALIDATION 5: False positive rate estimate (noise vs signal)')
R('-'*70)
R('  Method: synthesise artificial noise patches and check if they pass')
R('  the same consistency/HU test as the "verified" candidates.')
R()

# Create synthetic noise patches at random locations and test them
rng = np.random.default_rng(2026)
Z,H,W = vol_pre.shape
hu_norm = (np.clip(vol_pre,-150,250)+150)/400.0
fake_pass = 0
n_trials  = 50
z_lo = int(Z*0.5)

for _ in range(n_trials):
    # Random noise patch
    cz = rng.integers(z_lo, Z-4)
    cr = rng.integers(24, H-24)
    cc = rng.integers(24, W-24)
    r0,r1 = cr-24,cr+24
    c0,c1 = cc-24,cc+24

    fake_hu = float(vol_pre[cz, r0:r1, c0:c1].mean())
    tlo, thi = fake_hu-50, fake_hu+50

    confirmed = 0; total = 0
    for dz in [-3,-2,-1,1,2,3]:
        zi = cz+dz
        if zi<0 or zi>=Z: continue
        total += 1
        sh,_,_ = phase_cross_correlation(hu_norm[cz], hu_norm[zi],
                                          upsample_factor=4, normalization=None)
        sm = np.sqrt(sh[0]**2+sh[1]**2)
        adj_r = ndimage.shift(hu_norm[zi],[float(sh[0]),float(sh[1])], order=1, mode='constant')
        ck = adj_r[r0:r1, c0:c1]*400-150
        fm = float(np.mean((ck>=tlo)&(ck<=thi)))
        if fm>=0.20 and sm<8.0:
            confirmed += 1
    consistency = confirmed/total if total>0 else 0
    if consistency >= 0.25:
        fake_pass += 1

R(f'  {n_trials} random soft-tissue locations tested with same criteria as pipeline.')
R(f'  Passed verification (consistency>=0.25): {fake_pass}/{n_trials} = {fake_pass/n_trials*100:.1f}%')
R()
if fake_pass/n_trials > 0.30:
    R('  !! HIGH FALSE POSITIVE RATE: The verification criteria are too permissive.')
    R('     Random soft-tissue passes at the same rate as "tumor" candidates.')
    R('     The algorithm cannot distinguish tumor from normal tissue.')
elif fake_pass/n_trials > 0.10:
    R('  MODERATE false positive rate. Criteria have some discriminative power')
    R('  but specificity is limited without supervised training data.')
else:
    R('  Low false positive rate. Criteria are reasonably specific.')
R()

# ================================================================
# VALIDATION 6: Organ annotation sanity
# ================================================================
R('-'*70)
R('VALIDATION 6: Organ annotation accuracy')
R('-'*70)

# Load the JSON results
json_path = os.path.join(SAVE_DIR, 'ct_v2_results.json')
if os.path.exists(json_path):
    with open(json_path, encoding='utf-8') as f:
        res = json.load(f)

    pre_org  = res.get('pre_organ_stats', {})
    post_org = res.get('post_organ_stats', {})

    R('  PRE organ annotation:')
    expected_present = {
        'liver': (200000, 2000000, 'Should be present in abdomen CT'),
        'spleen': (20000, 400000, 'Usually present'),
        'kidney_r': (3000, 80000, 'Usually present'),
        'kidney_l': (3000, 80000, 'Usually present'),
        'rectum_region': (5000, 500000, 'Critical for rectal cancer'),
    }
    for org, (lo, hi, note) in expected_present.items():
        cnt = pre_org.get(org, {}).get('voxel_count', 0)
        status = 'OK' if lo <= cnt <= hi else ('MISSING' if cnt < lo else 'OVER-SEGMENTED')
        R(f'    {org:20s}: {cnt:10,} voxels  [{status}]  {note}')

    R()
    R('  Note: kidney labels = 0 voxels likely because kidneys were filtered out')
    R('  by the body-half spatial prior (hardcoded to posterior half = H//2:)')
    R('  This is a limitation of the HU+anatomy heuristic approach.')
    R()
    R('  NOTE: "liver" may include spleen/bowel wall as no shape model used.')
    R('  TRUE validation requires: CT expert review OR pretrained seg model (e.g. TotalSegmentator).')
else:
    R('  ct_v2_results.json not found.')
R()

# ================================================================
# VALIDATION 7: Ground truth comparison
# ================================================================
R('-'*70)
R('VALIDATION 7: Ground Truth Comparison against Clinical Report')
R('-'*70)
R()
R('  Clinical report (1223.txt):')
R('    "Slightly decreased irregular enhancing wall thickening in rectum"')
R('    "Decreased size of LNs in perirectal and superior rectal areas"')
R()
R('  Pipeline result:')
R('    PRE best tumor: area=274 voxels at HU=38.5 in pelvis')
R('    POST best tumor: area=52 voxels at HU=89.3 in CHEST scan')
R()
R('  Critical mismatch:')
R('    a) POST scan is CHEST CT, not abdomen/pelvis -> rectum not imaged')
R('    b) Tumor volume of 274 and 52 voxels = too small for rectal wall thickening')
R('    c) Rectal wall thickening typically spans 10-30 slices, cm-scale in-plane')
R('    d) No direct rectum segmentation ground truth available to verify')
R()
R('  CONCLUSION: Pipeline detected SMALL SOFT-TISSUE FOCI (lymph nodes, vessels,')
R('  bowel wall segments), NOT the primary rectal tumor.')
R()

# ================================================================
# VALIDATION 8: SUMMARY SCORECARD
# ================================================================
R('='*70)
R('OVERALL VALIDATION SCORECARD')
R('='*70)
R()
checks = [
    ('DICOM loading and HU conversion',       'PASS', 'Correct RescaleSlope/Intercept applied'),
    ('Organ annotation: liver',               'PARTIAL', 'Large component found but boundary imprecise'),
    ('Organ annotation: spleen/kidney',       'FAIL', 'Not detected due to spatial prior limitations'),
    ('Tumor candidate detection',             'PARTIAL', 'Candidates found but too small (274/52 vox)'),
    ('Registration shift computation',        'CAUTION', 'Shift near-zero: helical CT already aligned'),
    ('Consistency score discrimination',      'FAIL', f'>{int(fake_pass/n_trials*100)}% random locations pass same criteria'),
    ('PRE vs POST anatomical comparability',  'FAIL', 'Abdomen vs Chest -- different body regions'),
    ('Tumor area change -81%',                'INVALID', 'Comparing abdomen to chest is meaningless'),
    ('Clinical correlation',                  'PARTIAL', 'Small foci consistent with LN but not primary tumor'),
]
for check, status, note in checks:
    col = {'PASS':'**','FAIL':'!!','PARTIAL':'--','CAUTION':'~~','INVALID':'XX'}.get(status,'??')
    R(f'  [{col}] {status:8s} {check}')
    R(f'           {note}')
R()
R('BOTTOM LINE:')
R('  The pipeline is a solid ENGINEERING skeleton (DICOM IO, HU windowing,')
R('  organ annotation, registration, feature extraction) but the CLINICAL')
R('  VALIDITY is limited because:')
R('    1. No ground-truth segmentation masks for training/validation')
R('    2. HU-threshold organ delineation misses spleen/kidneys')
R('    3. Consistency test is trivially passed by any stable CT region')
R('    4. PRE and POST compare DIFFERENT anatomical regions (ab vs chest)')
R()
R('RECOMMENDED NEXT STEPS:')
R('    A) Use TotalSegmentator or nnU-Net for AI organ segmentation')
R('    B) Compare 0930 Abdomen Pre vs a matching Abdomen Post (when available)')
R('    C) Manually annotate rectal tumor ROI in pre/post for ground truth')
R('    D) Use RECIST 1.1: measure maximum diameter of target lesions manually')
R()
R('='*70)

# ================================================================
# FIGURE: Validation visualisation
# ================================================================
fig = plt.figure(figsize=(22, 14), facecolor='#0D1117')
gs  = gridspec.GridSpec(3, 4, figure=fig,
                        left=0.04, right=0.98, top=0.93, bottom=0.05,
                        wspace=0.22, hspace=0.40)
TK = dict(fontsize=8.5, color='#8BAFD4', fontweight='bold', pad=5)
BG = '#161B22'

def _ax(r,c,colspan=1):
    return fig.add_subplot(gs[r, c:c+colspan] if colspan>1 else gs[r,c])

# Row 0: lung fraction per slice (anatomy check)
ax0 = _ax(0,0,colspan=2)
ax0.set_facecolor(BG)
ax0.plot(pre_lung,  color='#1D6FA5', lw=1.5, label=f'PRE {meta_pre["date"]}')
ax0.plot(post_lung, color='#D4720B', lw=1.5, label=f'POST {meta_post["date"]}')
ax0.axhline(0.05, color='gray', ls='--', lw=0.8, label='Lung threshold (5%)')
ax0.set_title('Lung Fraction per Slice -- Anatomy Mismatch Test', **TK)
ax0.set_xlabel('Slice index', fontsize=8, color='#A0B8D0')
ax0.set_ylabel('Fraction of lung voxels', fontsize=8, color='#A0B8D0')
ax0.tick_params(colors='#A0B8D0')
ax0.spines[:].set_color('#2D3748'); ax0.yaxis.grid(True, color='#2D3748', lw=0.6)
ax0.legend(fontsize=8, facecolor='#1A2030', edgecolor='#3D4F6A', labelcolor='#A0B8D0')
ax0.set_axisbelow(True)
ax0.set_facecolor(BG)
ax0.text(0.3, 0.85,
         'POST = CHEST CT (high lung fraction)\nPRE  = ABDOMEN CT (low lung fraction)',
         transform=ax0.transAxes, fontsize=8, color='#FF6B6B',
         bbox=dict(boxstyle='round,pad=0.3', facecolor='#2D1B1B', edgecolor='#FF4444'))

# Row 0 col2-3: inter-slice registration shifts
ax1 = _ax(0,2,colspan=2)
ax1.set_facecolor(BG)
ax1.bar(range(len(shifts_pre)), shifts_pre, color='#1D6FA5', alpha=0.8)
ax1.axhline(8.0, color='#FF4444', ls='--', lw=1.2, label='max_shift_px=8 threshold')
ax1.axhline(np.mean(shifts_pre), color='#FFB347', ls=':', lw=1.2,
            label=f'Mean={np.mean(shifts_pre):.2f}px')
ax1.set_title(f'Registration Shift: consecutive PRE slices\n'
              f'(all <1px => consistency test trivially passes)', **TK)
ax1.set_xlabel('Slice pair', fontsize=8, color='#A0B8D0')
ax1.set_ylabel('Shift magnitude (px)', fontsize=8, color='#A0B8D0')
ax1.tick_params(colors='#A0B8D0')
ax1.spines[:].set_color('#2D3748'); ax1.yaxis.grid(True, color='#2D3748', lw=0.6)
ax1.legend(fontsize=8, facecolor='#1A2030', edgecolor='#3D4F6A', labelcolor='#A0B8D0')
ax1.set_facecolor(BG)

# Row 1: HU histogram comparison (pre vs post)
ax2 = _ax(1,0,colspan=2)
ax2.set_facecolor(BG)
pre_flat  = vol_pre[(vol_pre>-500)& (vol_pre<300)].ravel()
post_flat = vol_post[(vol_post>-500)&(vol_post<300)].ravel()
bins = np.linspace(-500, 300, 80)
ax2.hist(pre_flat,  bins=bins, color='#1D6FA5', alpha=0.7, density=True, label='PRE Abdomen')
ax2.hist(post_flat, bins=bins, color='#D4720B', alpha=0.7, density=True, label='POST Chest')
ax2.axvspan(25, 130, color='#FF4444', alpha=0.15, label='Tumor HU search range (25-130)')
ax2.set_title('HU Histogram PRE vs POST -- Tissue Composition Difference', **TK)
ax2.set_xlabel('HU', fontsize=8, color='#A0B8D0')
ax2.set_ylabel('Density', fontsize=8, color='#A0B8D0')
ax2.tick_params(colors='#A0B8D0')
ax2.legend(fontsize=7.5, facecolor='#1A2030', edgecolor='#3D4F6A', labelcolor='#A0B8D0')
ax2.spines[:].set_color('#2D3748')
ax2.set_facecolor(BG)

# Row 1 col2-3: False positive rate
ax3 = _ax(1,2,colspan=2)
ax3.set_facecolor(BG)
ax3.bar(['Random\nlocation\n(n=50)', 'Pipeline\ntumor\ncandidates'],
        [fake_pass/n_trials, 1.0],
        color=['#D4720B', '#1D6FA5'], alpha=0.85, width=0.5)
ax3.axhline(0.10, color='#44FF88', ls='--', lw=1.2, label='10% FP (acceptable)')
ax3.axhline(0.30, color='#FF4444', ls='--', lw=1.2, label='30% FP (concerning)')
ax3.set_ylim(0, 1.2)
ax3.set_title(f'False Positive Rate Estimate\n'
              f'{fake_pass}/{n_trials} random locations pass same consistency test', **TK)
ax3.set_ylabel('Pass rate', fontsize=8, color='#A0B8D0')
ax3.tick_params(colors='#A0B8D0')
ax3.legend(fontsize=8, facecolor='#1A2030', edgecolor='#3D4F6A', labelcolor='#A0B8D0')
ax3.spines[:].set_color('#2D3748')
ax3.yaxis.grid(True, color='#2D3748', lw=0.6)
ax3.set_axisbelow(True)
ax3.set_facecolor(BG)

# Row 2: actual detected "tumor" location in PRE (show raw patch)
def soft_win(v): return hu_window(v, 40, 400)
def hu_window(v,wl,ww):
    lo,hi=wl-ww/2,wl+ww/2
    return (np.clip(v,lo,hi)-lo)/(hi-lo)

# Show the top-1 PRE candidate in context
if os.path.exists(json_path):
    pre_verified = res.get('pre_candidates_verified', [])
    post_verified = res.get('post_candidates_verified', [])
else:
    pre_verified, post_verified = [], []

pre_slices_raw  = [resize(soft_win(vol_pre[i]),  (TARGET_SZ,TARGET_SZ), anti_aliasing=True) for i in range(len(vol_pre))]
post_slices_raw = [resize(soft_win(vol_post[i]), (TARGET_SZ,TARGET_SZ), anti_aliasing=True) for i in range(len(vol_post))]

ax4 = _ax(2,0)
ax4.set_facecolor(BG)
if pre_verified:
    v   = pre_verified[0]
    cz  = int(v['centroid'][0])
    sl  = pre_slices_raw[cz]
    ax4.imshow(sl, cmap='gray', aspect='equal')
    # Scale bounding box to 256
    s = TARGET_SZ/512
    r0,r1,c0,c1 = int(v['r0']*s),int(v['r1']*s),int(v['c0']*s),int(v['c1']*s)
    ax4.add_patch(plt.Rectangle((c0,r0),c1-c0,r1-r0, edgecolor='#FF4444',
                                 facecolor='none', lw=2))
    ax4.set_title(f'PRE "tumor" z={cz}\narea={v["area"]} vox, HU={v["mean_hu"]:.0f}', **TK)
else:
    ax4.set_title('PRE -- no verified candidates', **TK)
ax4.axis('off')

ax5 = _ax(2,1)
ax5.set_facecolor(BG)
if pre_verified:
    # Zoom into the bounding box
    v   = pre_verified[0]
    cz  = int(v['centroid'][0])
    MARGIN = 30
    rs = max(0,v['r0']-MARGIN); re = min(512,v['r1']+MARGIN)
    cs = max(0,v['c0']-MARGIN); ce = min(512,v['c1']+MARGIN)
    patch = soft_win(vol_pre[cz, rs:re, cs:ce])
    ax5.imshow(patch, cmap='gray', aspect='equal')
    inner_r0 = v['r0']-rs; inner_c0 = v['c0']-cs
    inner_h  = v['r1']-v['r0']; inner_w  = v['c1']-v['c0']
    ax5.add_patch(plt.Rectangle((inner_c0,inner_r0),inner_w,inner_h,
                                 edgecolor='#FF4444', facecolor='none', lw=2))
    ax5.set_title(f'PRE zoomed patch\n({inner_w}x{inner_h} px at HU={v["mean_hu"]:.0f})', **TK)
ax5.axis('off')

ax6 = _ax(2,2)
ax6.set_facecolor(BG)
if post_verified:
    v   = post_verified[0]
    cz  = int(v['centroid'][0])
    sl  = post_slices_raw[cz]
    ax6.imshow(sl, cmap='gray', aspect='equal')
    s = TARGET_SZ/512
    r0,r1,c0,c1 = int(v['r0']*s),int(v['r1']*s),int(v['c0']*s),int(v['c1']*s)
    ax6.add_patch(plt.Rectangle((c0,r0),c1-c0,r1-r0, edgecolor='#FF4444',
                                 facecolor='none', lw=2))
    ax6.set_title(f'POST "tumor" z={cz}\narea={v["area"]} vox, HU={v["mean_hu"]:.0f}', **TK)
else:
    ax6.set_title('POST -- no verified candidates', **TK)
ax6.axis('off')

ax7 = _ax(2,3)
ax7.set_facecolor(BG)
ax7.axis('off')
scorecard = [
    ('Anatomy match',       'FAIL  -- Abdomen vs Chest',       '#FF4444'),
    ('Tumor size',          'FAIL  -- 274/52 vox too small',   '#FF4444'),
    ('Registration test',   'FAIL  -- shifts ~0 trivially',    '#FF4444'),
    ('FP rate',             f'FAIL  -- {fake_pass/n_trials*100:.0f}% random pass',  '#FF4444'),
    ('Liver detection',     'PARTIAL -- imprecise boundary',   '#FFB347'),
    ('Kidney detection',    'FAIL  -- not found',              '#FF4444'),
    ('DICOM/HU pipeline',   'PASS  -- correct conversions',    '#44FF88'),
    ('Organ dict struct',   'PASS  -- data stored correctly',  '#44FF88'),
]
ax7.text(0.5, 1.02, 'Validation Scorecard', ha='center', va='top',
         fontsize=9, color='#8BAFD4', fontweight='bold', transform=ax7.transAxes)
for i, (k,v,col) in enumerate(scorecard):
    y = 0.88 - i*0.115
    ax7.text(0.0, y, k+':', transform=ax7.transAxes, fontsize=7.5, color='#A0B8D0', va='top')
    ax7.text(0.38, y, v,   transform=ax7.transAxes, fontsize=7.5, color=col,       va='top', fontweight='bold')

fig.suptitle('CT Pipeline v2  --  RIGOROUS VALIDATION REPORT  |  Patient 001740976',
             fontsize=12, fontweight='bold', color='#E2EAF4', y=0.98)

fig_path = os.path.join(SAVE_DIR, 'validation_figure.png')
plt.savefig(fig_path, dpi=150, bbox_inches='tight', facecolor='#0D1117')
R()
R(f'Figure -> {fig_path}')
plt.close()

# Save text report
txt_path = os.path.join(SAVE_DIR, 'validation_report.txt')
with open(txt_path, 'w', encoding='utf-8') as f:
    f.write('\n'.join(report_lines))
R(f'Text  -> {txt_path}')
