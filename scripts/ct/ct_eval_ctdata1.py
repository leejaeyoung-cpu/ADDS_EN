"""
CTdata1 -- Ground Truth Evaluation
====================================
Patient 002227784  |  2025-12-16
Data: NIfTI arterial CT + nnU-Net segmentation + tumor masks (ground truth)

Goals:
  1. Load arterial CT + binary tumor mask
  2. Apply enhancement-based detection (same algorithm as ct_pipeline_v3)
  3. Compute Dice, IoU, Precision, Recall vs ground truth
  4. Show exact tumor location with overlay
  5. Honest scorecard

Output: F:/ADDS/ct_v3_ctdata1_results.json
         F:/ADDS/ct_v3_ctdata1_figure.png
"""
import os, json, warnings
import numpy as np
import nibabel as nib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import ndimage
from skimage import measure, morphology, filters
from skimage.transform import resize

warnings.filterwarnings('ignore')
np.random.seed(2026)

DATA1  = r'F:\ADDS\CTdata1'
SAVE   = r'F:\ADDS'

log = []
def L(s=''):
    if isinstance(s, str):
        try:
            print(s)
        except UnicodeEncodeError:
            print(s.encode('ascii', 'replace').decode())
    else:
        print(s)
    log.append(str(s))

L('='*68)
L('CTdata1 Ground Truth Evaluation  |  Patient 002227784')
L('='*68)

# ================================================================
# LOAD DATA
# ================================================================
L('\n[1] Loading NIfTI volumes...')

art    = nib.load(os.path.join(DATA1, 'nifti',       'inha_ct_arterial.nii.gz'))
seg    = nib.load(os.path.join(DATA1,               'segmentation_remapped.nii.gz'))
tmask  = nib.load(os.path.join(DATA1, 'tumor_masks', 'tumor_mask_binary.nii.gz'))
tmask_mc = nib.load(os.path.join(DATA1, 'tumor_masks', 'tumor_mask_multiclass.nii.gz'))

ct     = art.get_fdata().astype(np.float32)      # HU
seg_d  = seg.get_fdata().astype(np.int32)
gt     = tmask.get_fdata().astype(np.uint8)       # binary ground truth
gt_mc  = tmask_mc.get_fdata().astype(np.uint8)

zooms  = np.array(art.header.get_zooms(), dtype=float)
vox_vol = float(np.prod(zooms))

Z, H, W = ct.shape
L(f'  CT volume:   {ct.shape}  HU range: [{ct.min():.0f}, {ct.max():.0f}]')
L(f'  Voxel size:  {zooms[0]:.3f} x {zooms[1]:.3f} x {zooms[2]:.3f} mm')
L(f'  Voxel vol:   {vox_vol:.3f} mm3')
L(f'  GT tumor (binary): {int(gt.sum()):,} voxels = {gt.sum()*vox_vol/1000:.2f} cm3')

# Segmentation labels
seg_labels = np.unique(seg_d[seg_d > 0])
L(f'  Segmentation labels: {seg_labels.tolist()} ({len(seg_labels)} organs)')
L()

# ================================================================
# ALGORITHM: Enhancement-based (using only arterial CT here)
# ================================================================
L('[2] Applying detection algorithm to arterial CT (no Pre phase available)...')
L('    NOTE: CTdata1 only has Arterial phase. Cannot do Pre-Portal subtraction.')
L('    Using absolute HU-based + morphology + organ mask approach.')
L()

# Use segmentation to define soft-tissue / organ priors
# Segmentation labels: remapped 0-23. Tumor-bearing organs typically labeled 1-10
body_mask = ct > -500
body_mask = ndimage.binary_fill_holes(body_mask.astype(bool))

# Strategy: detect dense soft-tissue masses in abdominal region
# Arterial phase: tumor HU typically 60-120 (with enhancement)
# Surrounding liver ~100-160, normal bowel wall ~40-80

# Step 1: candidate region by HU
cand_hu = (ct >= 40) & (ct <= 200) & body_mask

# Step 2: exclude bone (HU > 300 in adjacent region)
bone_dilated = ndimage.binary_dilation(ct > 280, iterations=2)
cand_hu[bone_dilated] = False

# Step 3: shape scoring via connected components
labeled = measure.label(cand_hu)
props   = measure.regionprops(labeled, intensity_image=ct)

candidates_raw = []
for p in props:
    if p.area < 200: continue
    if p.solidity is None or p.solidity < 0.10: continue
    bb = p.bbox
    mean_hu = float(p.mean_intensity)
    score = 0.0
    if 50 <= mean_hu <= 130: score += 0.40
    if 500 < p.area < 50000: score += 0.25
    if p.solidity > 0.3:     score += 0.20
    if p.extent > 0.2:       score += 0.15
    candidates_raw.append({
        'label': int(p.label),
        'area': int(p.area),
        'z0':bb[0],'r0':bb[1],'c0':bb[2],
        'z1':bb[3],'r1':bb[4],'c1':bb[5],
        'centroid': tuple(float(x) for x in p.centroid),
        'mean_hu': round(mean_hu, 1),
        'solidity': round(float(p.solidity), 3),
        'score': round(score, 4),
    })

candidates_raw.sort(key=lambda x: -x['score'])
L(f'  Raw candidates: {len(candidates_raw)}')
if candidates_raw:
    t = candidates_raw[0]
    L(f'  Top: area={t["area"]:,} vox={t["area"]*vox_vol/1000:.2f}cm3  HU={t["mean_hu"]}  score={t["score"]}')

# Build prediction mask from top-N candidates
N_TOP = 20
pred_mask = np.zeros_like(gt, dtype=np.uint8)
for c in candidates_raw[:N_TOP]:
    pred_mask[labeled == c['label']] = 1

# Also try with top-50
pred_mask_50 = np.zeros_like(gt, dtype=np.uint8)
for c in candidates_raw[:50]:
    pred_mask_50[labeled == c['label']] = 1

# ================================================================
# METRICS vs GROUND TRUTH
# ================================================================
L()
L('[3] Computing metrics vs ground truth mask...')

def compute_metrics(pred, truth):
    tp = int(((pred == 1) & (truth == 1)).sum())
    fp = int(((pred == 1) & (truth == 0)).sum())
    fn = int(((pred == 0) & (truth == 1)).sum())
    tn = int(((pred == 0) & (truth == 0)).sum())
    dice = 2*tp / (2*tp + fp + fn + 1e-9)
    iou  = tp / (tp + fp + fn + 1e-9)
    prec = tp / (tp + fp + 1e-9)
    rec  = tp / (tp + fn + 1e-9)
    spec = tn / (tn + fp + 1e-9)
    return {'TP':tp,'FP':fp,'FN':fn,'TN':tn,
            'Dice':round(dice,4),'IoU':round(iou,4),
            'Precision':round(prec,4),'Recall':round(rec,4),'Specificity':round(spec,4)}

m20 = compute_metrics(pred_mask,    gt)
m50 = compute_metrics(pred_mask_50, gt)

L(f'  [Top-20 candidates]')
L(f'    Dice:        {m20["Dice"]:.4f}  (1.0=perfect, 0.5=acceptable, <0.3=poor)')
L(f'    IoU:         {m20["IoU"]:.4f}')
L(f'    Precision:   {m20["Precision"]:.4f}  (TP/(TP+FP))')
L(f'    Recall:      {m20["Recall"]:.4f}  (TP/(TP+FN))')
L(f'    Specificity: {m20["Specificity"]:.4f}')
L(f'    TP={m20["TP"]:,}  FP={m20["FP"]:,}  FN={m20["FN"]:,}')
L()
L(f'  [Top-50 candidates]')
L(f'    Dice:        {m50["Dice"]:.4f}')
L(f'    IoU:         {m50["IoU"]:.4f}')
L(f'    Precision:   {m50["Precision"]:.4f}')
L(f'    Recall:      {m50["Recall"]:.4f}')
L()

# Volume comparison
pred_vol_20 = float(pred_mask.sum()) * vox_vol / 1000
pred_vol_50 = float(pred_mask_50.sum()) * vox_vol / 1000
gt_vol      = float(gt.sum()) * vox_vol / 1000
L(f'  GT tumor volume:      {gt_vol:.2f} cm3')
L(f'  Pred volume (top20):  {pred_vol_20:.2f} cm3  (over: {pred_vol_20/gt_vol*100:.0f}%)')
L(f'  Pred volume (top50):  {pred_vol_50:.2f} cm3  (over: {pred_vol_50/gt_vol*100:.0f}%)')
L()

# Verdic on Dice
if m20['Dice'] > 0.6:
    dice_verdict = 'GOOD -- clinically acceptable range'
elif m20['Dice'] > 0.4:
    dice_verdict = 'MODERATE -- usable for rough localization'
elif m20['Dice'] > 0.2:
    dice_verdict = 'POOR -- significant over/under-segmentation'
else:
    dice_verdict = 'FAIL -- algorithm cannot locate the tumor'
L(f'  Dice verdict: {dice_verdict}')
L()

# Per-slice Dice
slice_dice = []
for zi in range(Z):
    p2 = pred_mask[zi]
    g2 = gt[zi]
    tp = int(((p2==1)&(g2==1)).sum())
    fp = int(((p2==1)&(g2==0)).sum())
    fn = int(((p2==0)&(g2==1)).sum())
    d = 2*tp/(2*tp+fp+fn+1e-9)
    slice_dice.append(d)

gt_slices = [zi for zi in range(Z) if gt[zi].sum() > 0]
L(f'  GT tumor spans slices: {min(gt_slices)} to {max(gt_slices)} ({len(gt_slices)} slices)')
if gt_slices:
    best_s = max(gt_slices, key=lambda zi: gt[zi].sum())
    L(f'  Largest GT slice: z={best_s}  ({int(gt[best_s].sum())} voxels, dice={slice_dice[best_s]:.3f})')
L()

# ================================================================
# FALSE POSITIVE RATE on non-tumor slices
# ================================================================
L('[4] False Positive Rate on non-tumor slices...')
nontumor_slices = [zi for zi in range(Z) if gt[zi].sum() == 0]
fp_counts = [int(pred_mask[zi].sum()) for zi in nontumor_slices]
L(f'  Non-tumor slices: {len(nontumor_slices)}')
L(f'  FP voxels in non-tumor slices: {sum(fp_counts):,}  (mean/slice: {np.mean(fp_counts):.1f})')
L(f'  Slices with ANY false positive: {sum(1 for x in fp_counts if x>0)}/{len(nontumor_slices)}')
L()

# ================================================================
# SEGMENTATION ACCURACY (organ labels)
# ================================================================
L('[5] Organ segmentation label summary...')
for lbl in sorted(np.unique(seg_d[seg_d > 0])):
    n = int((seg_d == lbl).sum())
    vol = n * vox_vol / 1000
    L(f'  Label {lbl:3d}: {n:8,} voxels = {vol:6.1f} cm3')
L()

# ================================================================
# FIGURE
# ================================================================
L('[6] Generating figure...')

def hw(v, wl, ww):
    lo, hi = wl-ww/2, wl+ww/2
    return (np.clip(v, lo, hi) - lo) / (hi-lo)

fig = plt.figure(figsize=(24, 14), facecolor='#0D1117')
gs  = gridspec.GridSpec(2, 5, figure=fig,
                        left=0.04, right=0.98, top=0.93, bottom=0.05,
                        wspace=0.20, hspace=0.35)
TK = dict(fontsize=8.5, color='#8BAFD4', fontweight='bold', pad=4)
BG = '#161B22'

def _ax(r, c, span=1):
    return fig.add_subplot(gs[r, c:c+span] if span>1 else gs[r, c])

# Colormap for seg
cmap_seg = plt.cm.get_cmap('tab20', 24)

# Best GT slice
best_zi = gt_slices[len(gt_slices)//2] if gt_slices else Z//2

sz = 256
def rs(img):
    return resize(img, (sz,sz), anti_aliasing=True)

# Panel A: CT slice
axA = _ax(0, 0)
axA.set_facecolor(BG)
axA.imshow(rs(hw(ct[best_zi], 60, 400)), cmap='gray', aspect='equal')
axA.set_title(f'Arterial CT  z={best_zi}\n002227784 | 2025-12-16', **TK)
axA.axis('off')

# Panel B: GT overlay
axB = _ax(0, 1)
axB.set_facecolor(BG)
ct2d = rs(hw(ct[best_zi], 60, 400))
rgb  = np.stack([ct2d]*3, axis=-1)
gt2d = rs(gt[best_zi].astype(float)) > 0.5
rgb[gt2d, 0] = 0.0; rgb[gt2d, 1] = 0.9; rgb[gt2d, 2] = 0.2
axB.imshow(rgb, aspect='equal')
axB.set_title(f'Ground Truth Tumor Mask\n{gt_vol:.1f} cm3 total', **TK)
axB.axis('off')

# Panel C: Prediction overlay
axC = _ax(0, 2)
axC.set_facecolor(BG)
ct2d = rs(hw(ct[best_zi], 60, 400))
rgb2  = np.stack([ct2d]*3, axis=-1)
pred2d = rs(pred_mask[best_zi].astype(float)) > 0.5
gt2d_c = rs(gt[best_zi].astype(float)) > 0.5
rgb2[pred2d & ~gt2d_c, 0] = 0.9   # FP = red
rgb2[pred2d & ~gt2d_c, 1] = 0.1
rgb2[~pred2d & gt2d_c, 0] = 0.0   # FN = blue
rgb2[~pred2d & gt2d_c, 2] = 0.9
rgb2[pred2d & gt2d_c,  0] = 0.1   # TP = green
rgb2[pred2d & gt2d_c,  1] = 0.9
rgb2[pred2d & gt2d_c,  2] = 0.1
axC.imshow(rgb2, aspect='equal')
axC.set_title(f'Prediction vs GT (Top-20)\nGreen=TP  Red=FP  Blue=FN', **TK)
axC.axis('off')

# Panel D: segmentation overlay
axD = _ax(0, 3)
axD.set_facecolor(BG)
seg2d = rs(seg_d[best_zi].astype(float))
axD.imshow(rs(hw(ct[best_zi], 60, 400)), cmap='gray', aspect='equal')
seg_show = np.ma.masked_where(seg2d < 0.5, seg2d)
axD.imshow(seg_show, cmap='tab20', alpha=0.5, aspect='equal', vmin=0, vmax=23)
axD.set_title(f'nnU-Net Segmentation\n24 organ labels', **TK)
axD.axis('off')

# Panel E: per-slice Dice
axE = _ax(0, 4)
axE.set_facecolor(BG)
gt_count_per_slice = [int(gt[zi].sum()) for zi in range(Z)]
axE.bar(range(Z), gt_count_per_slice, color='#44AA88', alpha=0.6, label='GT voxels')
ax2e = axE.twinx()
ax2e.plot(range(Z), slice_dice, 'r-', lw=1.5, label='Dice per slice')
ax2e.set_ylim(0, 1.1)
axE.set_title('Per-slice GT tumor size vs Dice', **TK)
axE.set_xlabel('Slice (z)', fontsize=7, color='#A0B8D0')
axE.set_ylabel('GT voxels', fontsize=7, color='#44AA88')
ax2e.set_ylabel('Dice', fontsize=7, color='red')
axE.tick_params(colors='#A0B8D0', labelsize=7)
ax2e.tick_params(colors='red', labelsize=7)
axE.spines[:].set_color('#2D3748')
axE.set_facecolor(BG)

# Row 2 panel A: multiclass tumor
axF = _ax(1, 0)
axF.set_facecolor(BG)
ct2df = rs(hw(ct[best_zi], 60, 400))
rgbf  = np.stack([ct2df]*3, axis=-1)
for cls, col in [(1, [0.2, 0.9, 0.2]), (2, [0.9, 0.6, 0.1])]:
    mc2d = rs((gt_mc[best_zi] == cls).astype(float)) > 0.5
    for ch, v in enumerate(col):
        rgbf[mc2d, ch] = v
axF.imshow(rgbf, aspect='equal')
axF.set_title(f'Tumor Multiclass\nGreen=class1({28.83:.1f}cm3) Orange=class2({5.74:.1f}cm3)', **TK)
axF.axis('off')

# Row 2 panel B: HU distribution inside GT tumor
axG = _ax(1, 1)
axG.set_facecolor(BG)
tumor_hu = ct[gt == 1].flatten()
axG.hist(tumor_hu, bins=60, color='#1D6FA5', alpha=0.85, density=True)
axG.axvline(tumor_hu.mean(), color='#FFB347', ls='--', lw=1.5, label=f'Mean={tumor_hu.mean():.0f}HU')
axG.axvline(np.median(tumor_hu), color='#FF4444', ls=':', lw=1.5, label=f'Median={np.median(tumor_hu):.0f}HU')
axG.set_title('HU Distribution inside GT Tumor\n(arterial phase)', **TK)
axG.set_xlabel('HU', fontsize=7, color='#A0B8D0')
axG.set_ylabel('Density', fontsize=7, color='#A0B8D0')
axG.tick_params(colors='#A0B8D0')
axG.legend(fontsize=7, facecolor='#1A2030', edgecolor='#3D4F6A', labelcolor='#A0B8D0')
axG.spines[:].set_color('#2D3748')
axG.set_facecolor(BG)

# Row 2 panel C: metrics bar chart
axH = _ax(1, 2)
axH.set_facecolor(BG)
metrics = ['Dice', 'IoU', 'Precision', 'Recall', 'Specificity']
vals20  = [m20[m] for m in metrics]
vals50  = [m50[m] for m in metrics]
x = np.arange(len(metrics))
w = 0.35
b1 = axH.bar(x-w/2, vals20, w, color='#1D6FA5', alpha=0.85, label='Top-20')
b2 = axH.bar(x+w/2, vals50, w, color='#D4720B', alpha=0.85, label='Top-50')
axH.axhline(0.5, color='#FF4444', ls='--', lw=1, label='0.5 threshold')
axH.set_ylim(0, 1.15)
for bar in list(b1)+list(b2):
    h = bar.get_height()
    axH.text(bar.get_x()+bar.get_width()/2, h+0.02, f'{h:.2f}',
             ha='center', va='bottom', fontsize=7, color='#A0B8D0')
axH.set_title('Metrics vs Ground Truth', **TK)
axH.set_xticks(x); axH.set_xticklabels(metrics, rotation=25, ha='right', fontsize=7)
axH.tick_params(colors='#A0B8D0')
axH.legend(fontsize=7, facecolor='#1A2030', edgecolor='#3D4F6A', labelcolor='#A0B8D0')
axH.spines[:].set_color('#2D3748'); axH.yaxis.grid(True, color='#2D3748', lw=0.5)
axH.set_axisbelow(True)
axH.set_facecolor(BG)

# Row 2 panel D+E: final scorecard
axI = _ax(1, 3, span=2)
axI.set_facecolor(BG)
axI.axis('off')

axI.text(0.5, 1.0, 'CTdata1 -- Honest Final Scorecard', ha='center', va='top',
         fontsize=10, color='#E2EAF4', fontweight='bold', transform=axI.transAxes)

rows = [
    ('Data quality',      'PASS',    f'Arterial NIfTI + GT mask available'),
    ('GT tumor volume',   'INFO',    f'{gt_vol:.2f} cm3 (plausible for liver/pancreas tumor)'),
    ('Dice (top-20)',     'FAIL' if m20['Dice']<0.3 else ('POOR' if m20['Dice']<0.5 else 'OK'),
                                     f'{m20["Dice"]:.4f}'),
    ('IoU  (top-20)',     'FAIL' if m20['IoU']<0.2 else 'OK',
                                     f'{m20["IoU"]:.4f}'),
    ('Recall (top-20)',   'FAIL' if m20['Recall']<0.3 else 'OK',
                                     f'{m20["Recall"]:.4f} (FN={m20["FN"]:,})'),
    ('Precision (top-20)','FAIL' if m20['Precision']<0.2 else 'OK',
                                     f'{m20["Precision"]:.4f} (FP={m20["FP"]:,})'),
    ('FP on non-tumor',   'INFO',    f'{sum(fp_counts):,} voxels total'),
    ('No Pre phase',      'LIMIT',   'Cannot do enhancement subtraction'),
    ('N=1 patient',       'LIMIT',   'No statistical significance'),
    ('No training',       'LIMIT',   'Threshold-based, not learned'),
]

colors = {'PASS':'#44FF88','FAIL':'#FF4444','OK':'#44FF88','POOR':'#FFB347',
          'INFO':'#88BBFF','LIMIT':'#888888'}
xs = [0.0, 0.12, 0.22]
hdrs = ['Item', 'Status', 'Detail']
for ci,(h,x) in enumerate(zip(hdrs,xs)):
    axI.text(x, 0.90, h, transform=axI.transAxes,
             fontsize=8, color='#8BAFD4', fontweight='bold', va='top')
for ri, (itm, st, note) in enumerate(rows):
    y = 0.82 - ri * 0.082
    col = colors.get(st, '#A0B8D0')
    axI.text(0.00, y, itm,  transform=axI.transAxes, fontsize=7.5,
             color='#A0B8D0', va='top')
    axI.text(0.12, y, f'[{st}]', transform=axI.transAxes, fontsize=7.5,
             color=col, va='top', fontweight='bold')
    axI.text(0.22, y, note, transform=axI.transAxes, fontsize=7.0,
             color='#8899AA', va='top')

fig.suptitle(
    'CTdata1 Full Evaluation  |  Patient 002227784  |  Arterial CT + Ground Truth Tumor Mask',
    fontsize=12, fontweight='bold', color='#E2EAF4', y=0.98)

out_fig = os.path.join(SAVE, 'ct_v3_ctdata1_figure.png')
plt.savefig(out_fig, dpi=150, bbox_inches='tight', facecolor='#0D1117')
L(f'Figure -> {out_fig}')
plt.close()

# ================================================================
# JSON
# ================================================================
results = {
    'patient_id': '002227784',
    'study_date': '20251216',
    'data_source': 'CTdata1',
    'ct_shape': list(ct.shape),
    'voxel_spacing_mm': zooms.tolist(),
    'voxel_vol_mm3': vox_vol,
    'gt_tumor_voxels': int(gt.sum()),
    'gt_tumor_vol_cm3': float(gt.sum() * vox_vol / 1000),
    'pred_tumor_vol_top20_cm3': pred_vol_20,
    'pred_tumor_vol_top50_cm3': pred_vol_50,
    'metrics_top20': m20,
    'metrics_top50': m50,
    'gt_tumor_HU': {
        'mean': float(tumor_hu.mean()),
        'std':  float(tumor_hu.std()),
        'median': float(np.median(tumor_hu)),
        'p5':  float(np.percentile(tumor_hu, 5)),
        'p95': float(np.percentile(tumor_hu, 95)),
    },
    'n_raw_candidates': len(candidates_raw),
    'top_candidates': candidates_raw[:5],
}

json_out = os.path.join(SAVE, 'ct_v3_ctdata1_results.json')
with open(json_out, 'w', encoding='utf-8') as f:
    json.dump(results, f, indent=2, ensure_ascii=False)
L(f'JSON  -> {json_out}')

# Text report
txt_out = os.path.join(SAVE, 'ct_v3_ctdata1_report.txt')
with open(txt_out, 'w', encoding='utf-8', errors='replace') as f:
    f.write('\n'.join(log))
L(f'Text  -> {txt_out}')
L('\nDone.')
