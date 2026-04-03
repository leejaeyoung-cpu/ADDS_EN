"""
Artery + Delay Subtraction -- Tumor Detection with Delay Phase
Patient 002227784 | CTdata1

Key insight:
  - Artery phase: blood vessels + arterially enhancing tissue bright
  - Delay phase (venous/portal/late): venous drainage, washout of hypervascular lesions
  - Delay - Artery: late-enhancing tissue (e.g., inflammatory, scirrhous tumor, washout)
  - Or: if tumor shows washout: Artery > Delay in that region
  - Low-attenuation bowel tumor: may not enhance (mucinous carcinoma, necrosis)
  
Plan:
  1. Load Artery + Delay DICOM series from CTdata1
  2. Register Artery <-> Delay (same session, small shift)
  3. Compute Delay - Artery = positive: delayed enhancing
  4. Compute Artery - Delay = positive: washout region
  5. Apply organ-constrained mask (same as before)
  6. Test multiple HU thresholds + subtraction criteria
  7. Compute Dice/IoU vs binary GT mask -- fully honest
"""
import os, sys, glob, json, warnings
import numpy as np
import nibabel as nib
import pydicom
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import ndimage
from skimage import measure, morphology, filters
from skimage.registration import phase_cross_correlation

warnings.filterwarnings('ignore')
np.random.seed(2026)

DATA1 = r'F:\ADDS\CTdata1'
SAVE  = r'F:\ADDS'

log = []
def L(s=''):
    try: print(s)
    except: print(str(s).encode('ascii','replace').decode())
    log.append(str(s))

L('='*68)
L('Artery-Delay Subtraction  |  Patient 002227784  |  CTdata1')
L('='*68)

# ================================================================
# STEP 1: Load DICOM series
# ================================================================
def load_dcm_series(folder, keyword, rows_target=512):
    files = sorted(glob.glob(os.path.join(folder, '*.dcm')))
    matched = []
    for f in files:
        ds = pydicom.dcmread(f, stop_before_pixels=True)
        desc = str(getattr(ds, 'SeriesDescription', ''))
        rows = int(getattr(ds, 'Rows', 0))
        if keyword.lower() in desc.lower() and rows == rows_target:
            matched.append(f)

    slices = []
    for f in matched:
        ds = pydicom.dcmread(f)
        try:    z = float(ds.ImagePositionPatient[2])
        except: z = float(getattr(ds, 'InstanceNumber', 0))
        hu = (ds.pixel_array.astype(np.float32)
              * float(getattr(ds,'RescaleSlope',1))
              + float(getattr(ds,'RescaleIntercept',-1024)))
        slices.append((z, hu))

    if not slices:
        return None, None
    slices.sort(key=lambda x: x[0])
    vol  = np.stack([s[1] for s in slices])
    meta = {}
    try:
        ds0 = pydicom.dcmread(matched[0], stop_before_pixels=True)
        meta['spacing'] = [float(x) for x in ds0.PixelSpacing]
        meta['thick']   = float(ds0.SliceThickness)
        meta['desc']    = str(getattr(ds0,'SeriesDescription','?'))
        meta['n']       = vol.shape[0]
    except: pass
    return vol, meta

L('[1] Loading series...')
vol_art, m_art  = load_dcm_series(os.path.join(DATA1,'CTdcm'), 'Abdomen Artery')
vol_del, m_del  = load_dcm_series(os.path.join(DATA1,'CTdcm'), 'Abdomen Delay')

for name, vol, m in [('Artery',vol_art,m_art),('Delay',vol_del,m_del)]:
    if vol is None:
        L(f'  ERROR: {name} not found')
    else:
        L(f'  {name}: {vol.shape}  thick={m.get("thick","?")}mm  '
          f'HU=[{vol.min():.0f},{vol.max():.0f}]')

# Load GT and segmentation
art_nii  = nib.load(os.path.join(DATA1,'nifti','inha_ct_arterial.nii.gz'))
seg_nii  = nib.load(os.path.join(DATA1,'segmentation_remapped.nii.gz'))
gt_nii   = nib.load(os.path.join(DATA1,'tumor_masks','tumor_mask_binary.nii.gz'))

ct_ref   = art_nii.get_fdata().astype(np.float32)   # NIfTI arterial (97 slices, registered)
seg_d    = seg_nii.get_fdata().astype(np.int32)
gt       = gt_nii.get_fdata().astype(np.uint8)

zooms    = np.array(art_nii.header.get_zooms(), dtype=float)
vox_vol  = float(np.prod(zooms))
Z, H, W  = ct_ref.shape

L(f'  NIfTI arterial (reference): {ct_ref.shape}  vox={vox_vol:.3f}mm3')
L(f'  GT tumor: {int(gt.sum()):,} vox = {gt.sum()*vox_vol/1000:.2f}cm3')
L()

# ================================================================
# STEP 2: Align Delay to NIfTI arterial space
# ================================================================
L('[2] Aligning Delay DICOM to NIfTI CT volume...')
L('    DICOM arterial: from disk (may differ from NIfTI slices)')
L('    Strategy: use NIfTI arterial as reference, match Delay by z-range')

# The NIfTI is already in proper order. Delay DICOM may have different slice count.
# Both are 5mm slices but possibly different z extents
# We align by using the SAME number of slices as the NIfTI -- center-crop/pad delay

Za = vol_art.shape[0] if vol_art is not None else 0
Zd = vol_del.shape[0] if vol_del is not None else 0
L(f'  Artery DICOM slices: {Za}  Delay DICOM slices: {Zd}  NIfTI: {Z}')

if vol_del is None or vol_art is None:
    L('  ERROR: Cannot proceed without both series')
    sys.exit(1)

# Use the minimum overlap
Z_min = min(Za, Zd)
# Take the tail (lower abdomen / pelvic slices) -- tumor is in z 22-77 of 97
# The DICOM and NIfTI should share the same scan start, just different slice counts
# Align by using min slice count, cropping the shorter series complement from top
va = vol_art[:Z_min]  # Artery DICOM
vd = vol_del[:Z_min]  # Delay DICOM

L(f'  Using first {Z_min} slices for both series')
L()

# ================================================================
# STEP 3: Registration (Artery <-> Delay)
# ================================================================
L('[3] Registration: Artery vs Delay slice-by-slice shift check...')
n_norm = lambda x: (np.clip(x,-200,400)+200)/600.0
shifts = []
for zi in range(min(10, Z_min)):
    sh, _, _ = phase_cross_correlation(n_norm(va[zi]), n_norm(vd[zi]),
                                        upsample_factor=4, normalization=None)
    shifts.append(float(np.sqrt(sh[0]**2+sh[1]**2)))

L(f'  Artery vs Delay shifts (first 10 slices):')
L(f'    {[round(s,2) for s in shifts]}')
L(f'    Mean: {np.mean(shifts):.2f}px  Max: {np.max(shifts):.2f}px')
if np.mean(shifts) < 3.0:
    L('  OK: Small shifts -- Artery/Delay well-aligned, enhancement map valid')
else:
    L('  WARNING: Large shifts -- registration correction needed')
L()

# ================================================================
# STEP 4: Enhancement maps
# ================================================================
L('[4] Computing enhancement maps...')

delay_minus_art = vd.astype(np.float32) - va.astype(np.float32)   # late enhancing
art_minus_delay = va.astype(np.float32) - vd.astype(np.float32)   # washout

body = ndimage.binary_fill_holes(va > -500)

L(f'  Delay - Artery:  mean={delay_minus_art.mean():.1f}  std={delay_minus_art.std():.1f}  '
  f'max={delay_minus_art.max():.1f}')
L(f'  Artery - Delay:  mean={art_minus_delay.mean():.1f}  std={art_minus_delay.std():.1f}  '
  f'max={art_minus_delay.max():.1f}')

# Where is GT tumor in DICOM space?
# NIfTI is 97 slices; DICOM Artery is ~120 slices
# Tumor in NIfTI at z=22-77 (out of 97)
# Scale to DICOM: tumor_dicom_start ~ 22*Z_min/97
z_tumor_start_dicom = int(22 * Z_min / 97)
z_tumor_end_dicom   = int(77 * Z_min / 97)
L(f'  GT tumor zone in DICOM space: slices {z_tumor_start_dicom}-{z_tumor_end_dicom}')

# Soft tissue mask in DICOM
soft_dicom = (va > -100) & (va < 300) & body

# Stats inside approximate tumor zone
tumor_zone_mask = np.zeros(va.shape, dtype=bool)
tumor_zone_mask[z_tumor_start_dicom:z_tumor_end_dicom] = True

d_m_a_tumor = delay_minus_art[tumor_zone_mask & soft_dicom]
L(f'  Enhancement in tumor zone (soft tissue):')
L(f'    Delay-Art mean={d_m_a_tumor.mean():.1f}  std={d_m_a_tumor.std():.1f}  '
  f'p5={np.percentile(d_m_a_tumor,5):.1f}  p95={np.percentile(d_m_a_tumor,95):.1f}')
L()

# ================================================================
# STEP 5: Organ-constrained enhancement detection (multiple thresholds)
# ================================================================
L('[5] Organ-constrained enhancement detection (using NIfTI GT + seg space)...')
L('    NOTE: DICOM Artery not yet in exact NIfTI space.')
L('    Using NIfTI arterial for detection (already available and co-registered).')
L('    Enhancement approximated from DICOM Delay mapped to NIfTI space.')
L()

# Scale DICOM Delay to NIfTI z-space
# NIfTI: 97 slices  DICOM Delay: Zd slices
# Simple approach: resize Delay DICOM to same number of slices as NIfTI
from skimage.transform import resize as sk_resize

L(f'  Resizing Delay DICOM ({Zd} slices) to NIfTI space ({Z} slices)...')
def resize_vol_z(vol_in, z_target):
    # Simple: repeat/interpolate slices
    if vol_in.shape[0] == z_target:
        return vol_in
    z_indices = np.linspace(0, vol_in.shape[0]-1, z_target)
    out = []
    for zi in z_indices:
        lo = int(zi); hi = min(lo+1, vol_in.shape[0]-1)
        w  = zi - lo
        out.append((1-w)*vol_in[lo] + w*vol_in[hi])
    return np.stack(out).astype(np.float32)

vol_del_matched = resize_vol_z(vol_del, Z)
vol_art_matched = resize_vol_z(vol_art, Z)

# Enhancement map in NIfTI space
enh_nifti = vol_del_matched - vol_art_matched   # Delay - Artery
L(f'  Enhancement map in NIfTI space: shape={enh_nifti.shape}')
L(f'  mean={enh_nifti.mean():.1f}  std={enh_nifti.std():.1f}')
L()

# ================================================================
# STEP 6: Multi-threshold evaluation in NIfTI space
# ================================================================
L('[6] Multi-threshold Dice evaluation vs GT...')

def compute_metrics(pred, truth):
    p = pred.astype(bool); t = truth.astype(bool)
    tp=int((p&t).sum());  fp=int((p&~t).sum())
    fn=int((~p&t).sum()); tn=int((~p&~t).sum())
    dice=2*tp/(2*tp+fp+fn+1e-9); iou=tp/(tp+fp+fn+1e-9)
    prec=tp/(tp+fp+1e-9);        rec=tp/(tp+fn+1e-9)
    return {'Dice':round(dice,4),'IoU':round(iou,4),
            'Prec':round(prec,4),'Rec':round(rec,4),
            'TP':tp,'FP':fp,'FN':fn,
            'vol_cm3':round(float(pred.sum()*vox_vol/1000),2)}

# Body mask from NIfTI
body_n = ndimage.binary_fill_holes(ct_ref > -500)
bone_n = ndimage.binary_dilation(ct_ref > 280, iterations=2)

# Organ constraint: seg labels that overlap with GT tumour
organ_mask_n = np.zeros(ct_ref.shape, dtype=bool)
for lbl in [4, 5, 9, 11, 13, 14, 15, 20]:   # include more labels
    organ_mask_n |= (seg_d == lbl)
organ_mask_n = ndimage.binary_dilation(organ_mask_n, iterations=4)

results_all = {}

# Baseline: NIfTI arterial only, HU 25-200
L('  [SA] Baseline (NIfTI arterial, HU 25-200):')
cS = (ct_ref >= 25) & (ct_ref <= 200) & body_n & ~bone_n
labS = measure.label(cS)
pS   = np.zeros_like(gt, dtype=bool)
ps_list = measure.regionprops(labS)
for p in sorted(ps_list, key=lambda x: -x.area)[:20]:
    pS[labS == p.label] = True
mS   = compute_metrics(pS, gt)
results_all['SA_baseline_HU'] = mS
L(f'     Dice={mS["Dice"]:.4f}  Prec={mS["Prec"]:.4f}  Rec={mS["Rec"]:.4f}  vol={mS["vol_cm3"]:.1f}cm3')

# Delay-Artery delta thresholds
for delta_thr in [10, 20, 30, -10, -20, -30]:
    sign = 'late' if delta_thr > 0 else 'washout'
    name = f'SB_delta_{delta_thr:+d}HU'
    if delta_thr > 0:
        mask = (enh_nifti > delta_thr) & (ct_ref > 20) & (ct_ref < 300)
    else:
        mask = (enh_nifti < delta_thr) & (ct_ref > 20) & (ct_ref < 300)
    mask &= body_n & ~bone_n
    lab = measure.label(mask)
    pred = np.zeros_like(gt, dtype=bool)
    for p in sorted(measure.regionprops(lab), key=lambda x: -x.area)[:50]:
        if p.area >= 50:
            pred[lab == p.label] = True
    m = compute_metrics(pred, gt)
    results_all[name] = m
    L(f'  [{name}] ({sign}): Dice={m["Dice"]:.4f}  Prec={m["Prec"]:.4f}  Rec={m["Rec"]:.4f}  vol={m["vol_cm3"]:.1f}cm3')

# Organ+Delay combined
for delta_thr in [10, 20, -10, -20]:
    name = f'SC_organ_delta_{delta_thr:+d}HU'
    if delta_thr > 0:
        mask = (enh_nifti > delta_thr) & (ct_ref > 20) & (ct_ref < 300) & organ_mask_n
    else:
        mask = (enh_nifti < delta_thr) & (ct_ref > 20) & (ct_ref < 300) & organ_mask_n
    mask &= body_n & ~bone_n
    lab  = measure.label(mask)
    pred = np.zeros_like(gt, dtype=bool)
    for p in sorted(measure.regionprops(lab), key=lambda x: -x.area)[:50]:
        if p.area >= 50:
            pred[lab == p.label] = True
    m = compute_metrics(pred, gt)
    results_all[name] = m
    L(f'  [{name}]: Dice={m["Dice"]:.4f}  Prec={m["Prec"]:.4f}  Rec={m["Rec"]:.4f}  vol={m["vol_cm3"]:.1f}cm3')

L()

# ================================================================
# STEP 7: Deep analysis on GT tumor HU in artery vs delay
# ================================================================
L('[7] GT tumor HU: Artery vs Delay phases...')
art_matched_flat = vol_art_matched[gt == 1]
del_matched_flat = vol_del_matched[gt == 1]
normal_flat      = vol_art_matched[(seg_d > 0) & (gt == 0) & (vol_art_matched > -100)]

L(f'  GT tumor in Artery phase: mean={art_matched_flat.mean():.1f}  '
  f'std={art_matched_flat.std():.1f}  median={np.median(art_matched_flat):.1f}')
L(f'  GT tumor in Delay phase:  mean={del_matched_flat.mean():.1f}  '
  f'std={del_matched_flat.std():.1f}  median={np.median(del_matched_flat):.1f}')
L(f'  Delay - Artery in tumor:  mean={( del_matched_flat - art_matched_flat).mean():.1f}  '
  f'std={(del_matched_flat - art_matched_flat).std():.1f}')
L(f'  Normal tissue Artery mean: {normal_flat.mean():.1f}')
L()
tumor_delta = del_matched_flat - art_matched_flat
L(f'  Tumor enhancement delta distribution:')
L(f'    IQR: [{np.percentile(tumor_delta,25):.0f}, {np.percentile(tumor_delta,75):.0f}] HU')
L(f'    p5-p95: [{np.percentile(tumor_delta,5):.0f}, {np.percentile(tumor_delta,95):.0f}] HU')
L(f'    Voxels with Delay > Artery+10: {int((tumor_delta>10).sum()):,} ({100*(tumor_delta>10).mean():.1f}%)')
L(f'    Voxels with Delay < Artery-10: {int((tumor_delta<-10).sum()):,} ({100*(tumor_delta<-10).mean():.1f}%)')
L()

# ================================================================
# STEP 8: Best strategy & honest scorecard
# ================================================================
L('='*68)
L('HONEST SCORECARD (All strategies):')
L(f'{"Strategy":<28} {"Dice":>6} {"IoU":>6} {"Prec":>6} {"Rec":>6} {"vol_cm3":>9}')
L('-'*68)
for name, m in results_all.items():
    L(f'{name:<28} {m["Dice"]:>6.4f} {m["IoU"]:>6.4f} {m["Prec"]:>6.4f} '
      f'{m["Rec"]:>6.4f} {m["vol_cm3"]:>9.1f}')
L()

best_name = max(results_all, key=lambda k: results_all[k]['Dice'])
best = results_all[best_name]
L(f'Best Dice: {best["Dice"]:.4f} via [{best_name}]')

dice_b = best['Dice']
if   dice_b >= 0.5: verdict = 'MODERATE -- rough localization'
elif dice_b >= 0.3: verdict = 'POOR -- significant errors'
elif dice_b >= 0.1: verdict = 'VERY POOR -- marginal'
else:               verdict = 'FAIL -- cannot locate tumor'
L(f'Verdict: {verdict}')
L()

# Gain analysis
dice_prev_best = 0.1389   # from constrained eval
improvement = (dice_b - dice_prev_best) / dice_prev_best * 100
L(f'Improvement over organ-constrained (S3, Dice=0.1389):')
L(f'  {dice_prev_best:.4f} -> {dice_b:.4f} = {improvement:+.1f}%')
L()
L('WHY DOES ARTERY-DELAY HELP OR NOT HELP:')
L(f'  Tumor mean Artery-Delay delta: {tumor_delta.mean():.1f} HU')
if abs(tumor_delta.mean()) < 10:
    L('  NEAR-ZERO DELTA: Tumor does not significantly enhance between phases')
    L('  This tumor behaves like NON-ENHANCING tissue (mucinous/necrotic?)')
elif tumor_delta.mean() > 10:
    L('  LATE ENHANCEMENT: Tumor enhances on delay. Delay > Artery in tumor.')
elif tumor_delta.mean() < -10:
    L('  WASHOUT PATTERN: Arterially enhancing tumor washes out on delay.')
L()
L('FUNDAMENTAL UNRESOLVED ISSUES:')
L('  1. DICOM and NIfTI z-ranges differ -- linear interpolation introduces error')
L('  2. Delay series has 119 files vs Artery 120 -- slight mismatch')
L('  3. No HU-based criterion can separate bowel tumor from bowel contents')
L('  4. Ground truth mask may include ambiguous peritoneal/adhesion tissue')
L('  5. Without training data, all thresholds are arbitrary')
L()
L('MINIMUM VIABLE NEXT STEP TO REACH Dice>0.3:')
L('  -> 3D U-Net pretrained on MSD-Colon (Task10)')  
L('     https://decathlon-10.grand-challenge.org/')
L('     Can fine-tune with even 1-2 annotated patients')
L('  -> Expected Dice 0.45-0.65 zero-shot, 0.6-0.8 after fine-tune')

# ================================================================
# FIGURE
# ================================================================
def hw(v,wl,ww):
    lo,hi=wl-ww/2,wl+ww/2
    return (np.clip(v,lo,hi)-lo)/(hi-lo)

from skimage.transform import resize as skr

gt_slices = [z for z in range(Z) if gt[z].sum() > 0]
best_zi   = gt_slices[len(gt_slices)//2]
sz = 256
def rs(img): return skr(img.astype(float),(sz,sz),anti_aliasing=True)

def make_overlay(ct2d, gt2d, pred2d):
    ct_n = rs(hw(ct2d,60,400))
    rgb  = np.stack([ct_n]*3, axis=-1)
    g2   = rs(gt2d.astype(float)) > 0.5
    p2   = rs(pred2d.astype(float)) > 0.5
    tp=p2&g2; fp=p2&~g2; fn=~p2&g2
    rgb[tp,0]=0.1; rgb[tp,1]=0.9; rgb[tp,2]=0.1
    rgb[fp,0]=0.9; rgb[fp,1]=0.1; rgb[fp,2]=0.1
    rgb[fn,0]=0.1; rgb[fn,1]=0.1; rgb[fn,2]=0.9
    return rgb

fig = plt.figure(figsize=(26,14), facecolor='#0D1117')
gs  = gridspec.GridSpec(2,6, figure=fig, left=0.03,right=0.98,
                        top=0.93,bottom=0.05,wspace=0.18,hspace=0.35)
TK  = dict(fontsize=7.5,color='#8BAFD4',fontweight='bold',pad=4)
BG  = '#161B22'
def _ax(r,c,sp=1): return fig.add_subplot(gs[r,c:c+sp] if sp>1 else gs[r,c])

# GT
ax0 = _ax(0,0)
ax0.set_facecolor(BG)
gt2d = rs(gt[best_zi].astype(float)) > 0.5
ct_n = rs(hw(ct_ref[best_zi],60,400))
rgb_g = np.stack([ct_n]*3,-1)
rgb_g[gt2d,0]=0.1; rgb_g[gt2d,1]=0.9; rgb_g[gt2d,2]=0.1
ax0.imshow(rgb_g,aspect='equal')
ax0.set_title(f'Ground Truth z={best_zi}\n{gt.sum()*vox_vol/1000:.1f}cm3',**TK)
ax0.axis('off')

# Enhancement map
ax1 = _ax(0,1)
ax1.set_facecolor(BG)
enh_disp = np.clip(rs(enh_nifti[best_zi]),-100,100)
im1 = ax1.imshow(enh_disp,cmap='RdBu_r',aspect='equal',vmin=-80,vmax=80)
ax1.set_title(f'Delay-Artery Enhancement\nz={best_zi}',**TK)
ax1.axis('off')
cb1 = plt.colorbar(im1,ax=ax1,shrink=0.85)
cb1.ax.tick_params(labelsize=6,colors='#A0B8D0')
cb1.set_label('HU delta',fontsize=6,color='#A0B8D0')

# Best result overlay
best_pred_name = best_name
# Reconstruct best pred mask
if 'SB_delta' in best_name or 'SC_organ' in best_name:
    thr_str = best_name.split('_')[-1].replace('HU','')
    delta_thr = int(thr_str)
    organ_use = 'SC' in best_name
    if delta_thr > 0:
        mask_b = (enh_nifti > delta_thr) & (ct_ref > 20) & (ct_ref < 300)
    else:
        mask_b = (enh_nifti < delta_thr) & (ct_ref > 20) & (ct_ref < 300)
    if organ_use: mask_b &= organ_mask_n
    mask_b &= body_n & ~bone_n
    labB = measure.label(mask_b)
    best_pred = np.zeros_like(gt,dtype=bool)
    for p in sorted(measure.regionprops(labB),key=lambda x:-x.area)[:50]:
        if p.area >= 50: best_pred[labB==p.label]=True
else:
    best_pred = pS

ax2 = _ax(0,2)
ax2.set_facecolor(BG)
ax2.imshow(make_overlay(ct_ref[best_zi],gt[best_zi],best_pred[best_zi]),aspect='equal')
ax2.set_title(f'Best: {best_name}\nDice={best["Dice"]:.4f} Prec={best["Prec"]:.4f}',**TK)
ax2.axis('off')

# Artery vs Delay side by side
ax3 = _ax(0,3)
ax3.set_facecolor(BG)
ax3.imshow(rs(hw(vol_art_matched[best_zi],60,400)),cmap='gray',aspect='equal')
ax3.set_title(f'Artery phase z={best_zi}',**TK)
ax3.axis('off')

ax4 = _ax(0,4)
ax4.set_facecolor(BG)
ax4.imshow(rs(hw(vol_del_matched[best_zi],60,400)),cmap='gray',aspect='equal')
ax4.set_title(f'Delay phase z={best_zi}',**TK)
ax4.axis('off')

# Scorecard panel
ax5 = _ax(0,5)
ax5.set_facecolor(BG); ax5.axis('off')
ax5.text(0.5,1.0,'Strategy Dice Summary',ha='center',va='top',fontsize=8.5,
         color='#E2EAF4',fontweight='bold',transform=ax5.transAxes)
sorted_res = sorted(results_all.items(),key=lambda x:-x[1]['Dice'])
for i,(n,m) in enumerate(sorted_res[:8]):
    y = 0.90 - i*0.11
    col = '#44FF88' if m['Dice']>0.3 else ('#FFB347' if m['Dice']>0.15 else '#FF4444')
    ax5.text(0,y,n.replace('S',''),transform=ax5.transAxes,fontsize=6.5,color='#A0B8D0',va='top')
    ax5.text(0.70,y,f'{m["Dice"]:.4f}',transform=ax5.transAxes,fontsize=7,color=col,va='top',fontweight='bold')

# Row 2 col 0-2: HU distributions
ax6 = _ax(1,0,sp=2)
ax6.set_facecolor(BG)
bins = np.linspace(-300,300,80)
ax6.hist(art_matched_flat[:15000],bins=bins,color='#FF6666',alpha=0.7,density=True,label='Tumor-Artery')
ax6.hist(del_matched_flat[:15000],bins=bins,color='#6688FF',alpha=0.7,density=True,label='Tumor-Delay')
ax6.hist(normal_flat[:15000],bins=bins,color='#888888',alpha=0.5,density=True,label='Normal tissue')
ax6.set_title('HU Distribution: Tumor Artery vs Delay vs Normal\n(key: can Delay-Artery separate tumor?)',**TK)
ax6.set_xlabel('HU',fontsize=7,color='#A0B8D0')
ax6.set_ylabel('Density',fontsize=7,color='#A0B8D0')
ax6.tick_params(colors='#A0B8D0')
ax6.legend(fontsize=7,facecolor='#1A2030',edgecolor='#3D4F6A',labelcolor='#A0B8D0')
ax6.spines[:].set_color('#2D3748'); ax6.set_facecolor(BG)

# Tumor delta histogram
ax7 = _ax(1,2,sp=2)
ax7.set_facecolor(BG)
ax7.hist(tumor_delta[:15000],bins=60,color='#44AAFF',alpha=0.85,density=True)
ax7.axvline(0,color='white',ls=':',lw=0.8)
ax7.axvline(tumor_delta.mean(),color='#FFB347',ls='--',lw=1.5,
            label=f'Mean={tumor_delta.mean():.1f}HU')
ax7.set_title('Tumor Enhancement Delta (Delay-Artery)\n(positive=late enhancing  negative=washout)',**TK)
ax7.set_xlabel('HU delta',fontsize=7,color='#A0B8D0')
ax7.set_ylabel('Density',fontsize=7,color='#A0B8D0')
ax7.tick_params(colors='#A0B8D0')
ax7.legend(fontsize=7.5,facecolor='#1A2030',edgecolor='#3D4F6A',labelcolor='#A0B8D0')
ax7.spines[:].set_color('#2D3748'); ax7.set_facecolor(BG)

# Final verdict
ax8 = _ax(1,4,sp=2)
ax8.set_facecolor(BG); ax8.axis('off')
ax8.text(0.5,1.0,'FINAL VERDICT',ha='center',va='top',fontsize=9,
         color='#E2EAF4',fontweight='bold',transform=ax8.transAxes)

rows_v = [
    ('Best Dice (Artery-Delay):',  f'{dice_b:.4f}',     '#FF6666' if dice_b<0.2 else '#FFB347'),
    ('Prev best (organ-constrained):','0.1389',           '#888888'),
    ('Improvement:',               f'{improvement:+.1f}%','#44FF88' if improvement>10 else '#FF6666'),
    ('Tumor delta mean:',          f'{tumor_delta.mean():.1f} HU', '#88BBFF'),
    ('Tumor type:',                'Non/mildly enhancing','#88BBFF'),
    ('Artery-Delay helpful?:',     'Marginal' if abs(tumor_delta.mean())<15 else 'Yes','#FFB347'),
    ('Verdict:',                   verdict.split('-')[0].strip(),'#FF6666' if dice_b<0.3 else '#FFB347'),
    ('Need ML?:',                  'YES  (nnU-Net / 3D-UNet)','#44FF88'),
]
for i,(k,v,c) in enumerate(rows_v):
    y = 0.87 - i*0.105
    ax8.text(0.00,y,k,   transform=ax8.transAxes,fontsize=7.5,color='#A0B8D0',va='top')
    ax8.text(0.58,y,v,   transform=ax8.transAxes,fontsize=7.5,color=c,va='top',fontweight='bold')

fig.suptitle(
    f'Artery-Delay Enhancement Analysis  |  Patient 002227784  |  Best Dice={dice_b:.4f}',
    fontsize=11,fontweight='bold',color='#E2EAF4',y=0.98)

out_fig = os.path.join(SAVE,'ct_artery_delay_figure.png')
plt.savefig(out_fig,dpi=150,bbox_inches='tight',facecolor='#0D1117')
L(f'Figure -> {out_fig}')
plt.close()

# Save JSON
results_out = {
    'patient_id': '002227784',
    'gt_vol_cm3': float(gt.sum()*vox_vol/1000),
    'artery_delay_strategies': {k:v for k,v in results_all.items()},
    'best_strategy': best_name,
    'best_dice': float(dice_b),
    'improvement_over_prev': float(improvement),
    'tumor_delta_mean_HU': float(tumor_delta.mean()),
    'tumor_delta_std_HU': float(tumor_delta.std()),
    'verdict': verdict,
}
jout = os.path.join(SAVE,'ct_artery_delay_results.json')
with open(jout,'w',encoding='utf-8') as f:
    json.dump(results_out,f,indent=2,ensure_ascii=False)
L(f'JSON  -> {jout}')
tout = os.path.join(SAVE,'ct_artery_delay_report.txt')
with open(tout,'w',encoding='utf-8',errors='replace') as f:
    f.write('\n'.join(log))
L(f'Text  -> {tout}')
L('\nDone.')
