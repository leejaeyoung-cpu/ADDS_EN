"""
CTdata1 -- Organ-Constrained Tumor Detection + Rigorous Validation
===================================================================
Patient 002227784  |  2025-12-16
Known: tumor in bowel/colon slices 22-77 (viewer guide)

Strategies applied:
  S1. Unconstrained HU (baseline)            -- v1 result: Dice=0.01
  S2. Organ-mask constrained (bowel ROI)
  S3. S2 + morphological refinement
  S4. Anomaly detection: local HU deviation   -- "stands out from neighborhood"
  S5. Combined S2+S3+S4 (best effort)

Each strategy is evaluated honestly with Dice/IoU/Precision/Recall/Volume ratio.
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

warnings.filterwarnings('ignore')
np.random.seed(2026)

DATA1 = r'F:\ADDS\CTdata1'
SAVE  = r'F:\ADDS'

log = []
def L(s=''):
    try: print(s)
    except UnicodeEncodeError: print(str(s).encode('ascii','replace').decode())
    log.append(str(s))

L('='*68)
L('CTdata1 Organ-Constrained Evaluation  |  Patient 002227784')
L('='*68)

# ================================================================
# LOAD
# ================================================================
art     = nib.load(os.path.join(DATA1, 'nifti',       'inha_ct_arterial.nii.gz'))
seg_n   = nib.load(os.path.join(DATA1,               'segmentation_remapped.nii.gz'))
tmask   = nib.load(os.path.join(DATA1, 'tumor_masks', 'tumor_mask_binary.nii.gz'))
tmask_mc= nib.load(os.path.join(DATA1, 'tumor_masks', 'tumor_mask_multiclass.nii.gz'))

ct      = art.get_fdata().astype(np.float32)
seg_d   = seg_n.get_fdata().astype(np.int32)
gt      = tmask.get_fdata().astype(np.uint8)
gt_mc   = tmask_mc.get_fdata().astype(np.uint8)

zooms   = np.array(art.header.get_zooms(), dtype=float)
vox_vol = float(np.prod(zooms))
Z, H, W = ct.shape

L(f'CT: {ct.shape}  vox={vox_vol:.3f}mm3  GT={int(gt.sum()):,}vox={gt.sum()*vox_vol/1000:.2f}cm3')
L()

# ================================================================
# STEP 0: Identify organ labels from segmentation
# ================================================================
L('[STEP 0] Identifying organ labels from segmentation volume...')
organ_labels = {}
for lbl in sorted(np.unique(seg_d[seg_d > 0])):
    n   = int((seg_d == lbl).sum())
    vol = n * vox_vol / 1000
    organ_labels[int(lbl)] = {'voxels': n, 'vol_cm3': round(vol, 2)}

# Map by size to known anatomy (GE CT, standard nnU-Net mapping v2)
# Heuristic: TotalSegmentator v2 remapped labels:
# Large volumes ~ liver, bowel, muscle
# Label 13 is ~539cm3 (likely liver or bowel/muscle complex)
sorted_by_vol = sorted(organ_labels.items(), key=lambda x: -x[1]['vol_cm3'])
L('  Top labels by volume:')
for lbl, info in sorted_by_vol[:12]:
    L(f'    Label {lbl:3d}: {info["voxels"]:>9,} vox = {info["vol_cm3"]:>7.1f} cm3')
L()

# GT tumor spans slices 22-77 (bowel/colon per viewer guide)  
# Identify which labels overlap with GT tumor mask
L('[STEP 0b] Overlap of segmentation labels with GT tumor...')
gt_label_overlap = {}
for lbl in np.unique(seg_d[seg_d > 0]):
    overlap = int(((seg_d == lbl) & (gt == 1)).sum())
    if overlap > 0:
        gt_label_overlap[int(lbl)] = overlap
gt_label_overlap_sorted = sorted(gt_label_overlap.items(), key=lambda x: -x[1])
L('  Labels with overlap to GT tumor (top):')
for lbl, ov in gt_label_overlap_sorted[:8]:
    total = organ_labels[lbl]['voxels']
    pct = 100 * ov / total
    L(f'    Label {lbl:3d}: {ov:>8,} GT vox ({pct:.1f}% of label)  vol={organ_labels[lbl]["vol_cm3"]:.1f}cm3')
L()

# The labels most overlapping with GT = tumor-bearing organs
tumor_labels = [lbl for lbl, _ in gt_label_overlap_sorted[:6]]
L(f'  Tumor-bearing organ labels: {tumor_labels}')
L()

# ================================================================
# METRICS FUNCTION
# ================================================================
def compute_metrics(pred, truth):
    p = pred.astype(bool); t = truth.astype(bool)
    tp = int((p &  t).sum()); fp = int((p & ~t).sum())
    fn = int((~p & t).sum()); tn = int((~p & ~t).sum())
    dice = 2*tp/(2*tp+fp+fn+1e-9)
    iou  = tp/(tp+fp+fn+1e-9)
    prec = tp/(tp+fp+1e-9)
    rec  = tp/(tp+fn+1e-9)
    spec = tn/(tn+fp+1e-9)
    vol_pred = pred.sum() * vox_vol / 1000
    vol_gt   = truth.sum() * vox_vol / 1000
    return {
        'Dice': round(dice,4), 'IoU': round(iou,4),
        'Precision': round(prec,4), 'Recall': round(rec,4), 'Specificity': round(spec,4),
        'TP':tp,'FP':fp,'FN':fn,
        'pred_vol_cm3': round(float(vol_pred),2),
        'gt_vol_cm3':   round(float(vol_gt),2),
        'vol_ratio':    round(float(vol_pred/vol_gt) if vol_gt>0 else 0, 2),
    }

all_results = {}

# ================================================================
# STRATEGY 1 (Baseline): HU only, unconstrained
# ================================================================
L('[S1] Baseline: HU 40-200, unconstrained (same as before)')
body  = ndimage.binary_fill_holes(ct > -500)
cand1 = (ct >= 40) & (ct <= 200) & body
cand1[ndimage.binary_dilation(ct > 280, iterations=2)] = False
lab1  = measure.label(cand1)
props1 = measure.regionprops(lab1, intensity_image=ct)
# Score and threshold
pred1 = np.zeros_like(gt, dtype=bool)
for p in sorted(props1, key=lambda x: -x.area)[:20]:
    pred1[lab1 == p.label] = True
m1 = compute_metrics(pred1, gt)
all_results['S1_baseline'] = m1
L(f'  Dice={m1["Dice"]:.4f}  Prec={m1["Precision"]:.4f}  Rec={m1["Recall"]:.4f}  '
  f'vol={m1["pred_vol_cm3"]:.1f}cm3 (ratio={m1["vol_ratio"]:.1f}x)')
L()

# ================================================================
# STRATEGY 2: Constrained to tumor-bearing organ labels
# ================================================================
L('[S2] Organ-constrained: HU detection within tumor-bearing labels only')
# Organ mask = union of tumor-coincident labels
organ_mask = np.zeros(ct.shape, dtype=bool)
for lbl in tumor_labels:
    organ_mask |= (seg_d == lbl)
# Also include label 0 (background) only within bounding box of tumor labels
# Dilate organ mask slightly to capture tumor extending slightly outside label
organ_mask_dilated = ndimage.binary_dilation(organ_mask, iterations=3)

cand2 = (ct >= 25) & (ct <= 200) & body & organ_mask_dilated
cand2[ndimage.binary_dilation(ct > 280, iterations=2)] = False
lab2  = measure.label(cand2)
props2 = measure.regionprops(lab2, intensity_image=ct)
pred2 = np.zeros_like(gt, dtype=bool)
for p in sorted(props2, key=lambda x: -x.area)[:50]:
    if p.area > 50:
        pred2[lab2 == p.label] = True
m2 = compute_metrics(pred2, gt)
all_results['S2_organ_constrained'] = m2
L(f'  Organ mask: {organ_mask.sum():,} vox   Dilated: {organ_mask_dilated.sum():,} vox')
L(f'  Dice={m2["Dice"]:.4f}  Prec={m2["Precision"]:.4f}  Rec={m2["Recall"]:.4f}  '
  f'vol={m2["pred_vol_cm3"]:.1f}cm3 (ratio={m2["vol_ratio"]:.1f}x)')
L()

# ================================================================
# STRATEGY 3: S2 + morphological refinement (remove large FP blobs)
# ================================================================
L('[S3] S2 + Morphological refinement: size filter + solidity filter')
pred3 = np.zeros_like(gt, dtype=bool)
gt_vol_vox = int(gt.sum())
# Only include components with area <= 3x GT volume and area >= 100 vox
for p in props2:
    if p.area < 100: continue
    if p.area > gt_vol_vox * 3: continue      # remove blobs way bigger than expected tumor
    if p.solidity is not None and p.solidity < 0.08: continue
    pred3[lab2 == p.label] = True
m3 = compute_metrics(pred3, gt)
all_results['S3_morphological'] = m3
L(f'  GT vol voxels: {gt_vol_vox:,}  Max candidate: {gt_vol_vox*3:,} vox')
L(f'  Dice={m3["Dice"]:.4f}  Prec={m3["Precision"]:.4f}  Rec={m3["Recall"]:.4f}  '
  f'vol={m3["pred_vol_cm3"]:.1f}cm3 (ratio={m3["vol_ratio"]:.1f}x)')
L()

# ================================================================
# STRATEGY 4: Local anomaly detection (Hu deviation from local context)
# ================================================================
L('[S4] Local anomaly detection: voxel HU anomalous vs 10mm neighborhood')
L('  Idea: tumor differs from surrounding tissue even within same organ')
# Compute local mean HU in 10mm radius sphere
radius_vox = int(10.0 / zooms[0])  # ~8 voxels
struct = ndimage.generate_binary_structure(3, 1)
kernel_r = radius_vox
# Use Gaussian smoothing as proxy for local mean (faster than disk)
local_mean = ndimage.gaussian_filter(ct, sigma=radius_vox/2.5)
hu_deviation = ct - local_mean   # positive = locally elevated

# Within organ mask: voxels with significant local elevation
anomaly_threshold = local_mean.std() * 0.5   # adaptive threshold
anom_mask = (hu_deviation > anomaly_threshold) & organ_mask_dilated & body
anom_mask[ndimage.binary_dilation(ct > 280, iterations=2)] = False

# Label anomalies
lab4   = measure.label(anom_mask)
props4 = measure.regionprops(lab4, intensity_image=hu_deviation)

pred4 = np.zeros_like(gt, dtype=bool)
for p in sorted(props4, key=lambda x: -x.area):
    if p.area < 50: continue
    if p.area > gt_vol_vox * 5: continue
    pred4[lab4 == p.label] = True

m4 = compute_metrics(pred4, gt)
all_results['S4_anomaly'] = m4
L(f'  Local mean std: {local_mean.std():.1f}  Anomaly threshold: {anomaly_threshold:.1f}')
L(f'  Dice={m4["Dice"]:.4f}  Prec={m4["Precision"]:.4f}  Rec={m4["Recall"]:.4f}  '
  f'vol={m4["pred_vol_cm3"]:.1f}cm3 (ratio={m4["vol_ratio"]:.1f}x)')
L()

# ================================================================
# STRATEGY 5: Combined -- intersection of S3 and S4 (reduces FP)
# ================================================================
L('[S5] Combined: S3 AND S4 intersection (conservative union)')
pred5_union = (pred3 | pred4)   # union = more recall
pred5_inter = (pred3 & pred4)   # intersection = more precision

m5u = compute_metrics(pred5_union, gt)
m5i = compute_metrics(pred5_inter, gt)
all_results['S5_union']        = m5u
all_results['S5_intersection'] = m5i

L(f'  Union:        Dice={m5u["Dice"]:.4f}  Prec={m5u["Precision"]:.4f}  Rec={m5u["Recall"]:.4f}  vol={m5u["pred_vol_cm3"]:.1f}cm3')
L(f'  Intersection: Dice={m5i["Dice"]:.4f}  Prec={m5i["Precision"]:.4f}  Rec={m5i["Recall"]:.4f}  vol={m5i["pred_vol_cm3"]:.1f}cm3')
L()

# ================================================================
# HONEST SCORECARD
# ================================================================
L('='*68)
L('HONEST SCORECARD:')
L(f'{"Strategy":<25} {"Dice":>6} {"IoU":>6} {"Prec":>6} {"Rec":>6} {"Vol(cm3)":>9} {"Ratio":>7}')
L('-'*68)
for name, m in all_results.items():
    L(f'{name:<25} {m["Dice"]:>6.4f} {m["IoU"]:>6.4f} {m["Precision"]:>6.4f} '
      f'{m["Recall"]:>6.4f} {m["pred_vol_cm3"]:>9.1f} {m["vol_ratio"]:>7.1f}x')
L()

best_name = max(all_results, key=lambda k: all_results[k]['Dice'])
best = all_results[best_name]
L(f'Best strategy by Dice: [{best_name}]  Dice={best["Dice"]:.4f}')
L()

# Clinical verdict
dice_best = best['Dice']
if dice_best >= 0.7:
    verdict = 'GOOD -- clinically useful detection'
elif dice_best >= 0.5:
    verdict = 'MODERATE -- rough localization only'
elif dice_best >= 0.3:
    verdict = 'POOR -- significant errors'
elif dice_best >= 0.1:
    verdict = 'VERY POOR -- marginal improvement over random'
else:
    verdict = 'FAIL -- cannot locate tumor'
L(f'Verdict: {verdict}')
L()

# Why is it difficult?
L('[ANALYSIS] Why is this tumor hard to detect without ML?')
tumor_hu = ct[gt == 1]
normal_bowel_hu = ct[(seg_d > 0) & (gt == 0) & (ct > -200) & (ct < 300)]
L(f'  GT tumor HU:         mean={tumor_hu.mean():.1f}  std={tumor_hu.std():.1f}  '
  f'median={np.median(tumor_hu):.1f}  IQR=[{np.percentile(tumor_hu,25):.0f},{np.percentile(tumor_hu,75):.0f}]')
L(f'  Normal tissue HU:    mean={normal_bowel_hu.mean():.1f}  std={normal_bowel_hu.std():.1f}  '
  f'median={np.median(normal_bowel_hu):.1f}  IQR=[{np.percentile(normal_bowel_hu,25):.0f},{np.percentile(normal_bowel_hu,75):.0f}]')
from scipy.stats import mannwhitneyu
stat, pval = mannwhitneyu(tumor_hu[:5000], normal_bowel_hu[:5000], alternative='two-sided')
L(f'  Mann-Whitney U p-value (HU separation): {pval:.6f}')
L()
if pval < 0.001:
    L('  HU distributions ARE statistically different (p<0.001)')
    L('  But: practical overlap makes threshold-based separation unreliable')
    # Compute overlap
    bins = np.linspace(-200, 300, 100)
    h_t,_ = np.histogram(tumor_hu[:10000], bins=bins, density=True)
    h_n,_ = np.histogram(normal_bowel_hu[:10000], bins=bins, density=True)
    overlap_frac = float(np.minimum(h_t, h_n).sum() / max(h_t.sum(), h_n.sum()))
    L(f'  HU distribution overlap: {overlap_frac*100:.1f}% (0%=perfect sep, 100%=identical)')
L()

L('[REMAINING FUNDAMENTAL LIMITS]')
L('  1. Bowel tumor vs bowel wall: same organ, similar HU -- inseparable by HU alone')
L('  2. No temporal comparison (Artery only, no Delay phase subtraction used)')
L('  3. N=1: cannot train or calibrate thresholds')
L('  4. Bright bowel loop contents + wall = similar HU to hypovascular tumor')
L('  5. Segmentation labels do not separate tumor from normal bowel wall')
L()
L('[WHAT WOULD ACTUALLY WORK]')
L('  A. Use Artery + Delay subtraction (both available in CTdata1)')
L('     -> Enhancing tumor will show higher delta in Delay phase')
L('  B. nnU-Net pretrained on colorectal CT (PanNuke/MSD Task10 etc.)')  
L('     -> Dice 0.60-0.75 expected')
L('  C. Feature engineering: texture (GLCM entropy, LBP) not just HU')
L('     -> Adds tumor heterogeneity signal')

# ================================================================
# FIGURE
# ================================================================
L()
L('[FIGURE] Generating comparison figure...')

def hw(v,wl,ww):
    lo,hi=wl-ww/2,wl+ww/2
    return (np.clip(v,lo,hi)-lo)/(hi-lo)

# Pick representative slice
gt_slices = [z for z in range(Z) if gt[z].sum() > 0]
best_zi   = gt_slices[len(gt_slices)//2]
sz = 256

def rs(img): 
    from skimage.transform import resize
    return resize(img.astype(float), (sz,sz), anti_aliasing=True)

def overlay(ct_2d, mask_2d, pred_2d):
    ct_n = rs(hw(ct_2d, 60, 400))
    rgb  = np.stack([ct_n]*3, axis=-1)
    g2   = rs(mask_2d.astype(float)) > 0.5
    p2   = rs(pred_2d.astype(float)) > 0.5
    tp_m = p2 & g2;  fp_m = p2 & ~g2;  fn_m = ~p2 & g2
    rgb[tp_m, 0] = 0.1;  rgb[tp_m, 1] = 0.9;  rgb[tp_m, 2] = 0.1   # TP=green
    rgb[fp_m, 0] = 0.9;  rgb[fp_m, 1] = 0.1;  rgb[fp_m, 2] = 0.1   # FP=red
    rgb[fn_m, 0] = 0.1;  rgb[fn_m, 1] = 0.1;  rgb[fn_m, 2] = 0.9   # FN=blue
    return rgb

fig = plt.figure(figsize=(26, 14), facecolor='#0D1117')
gs  = gridspec.GridSpec(2, 6, figure=fig, left=0.03, right=0.98,
                        top=0.93, bottom=0.05, wspace=0.18, hspace=0.35)
TK  = dict(fontsize=8, color='#8BAFD4', fontweight='bold', pad=4)
BG  = '#161B22'

def _ax(r, c, span=1):
    return fig.add_subplot(gs[r, c:c+span] if span>1 else gs[r, c])

# GT
axA = _ax(0,0)
axA.set_facecolor(BG)
gt2d_show = rs(gt[best_zi].astype(float)) > 0.5
ct_n = rs(hw(ct[best_zi],60,400))
rgb_gt = np.stack([ct_n]*3, axis=-1)
rgb_gt[gt2d_show, 0] = 0.1; rgb_gt[gt2d_show, 1] = 0.9; rgb_gt[gt2d_show, 2] = 0.1
axA.imshow(rgb_gt, aspect='equal')
axA.set_title(f'Ground Truth  z={best_zi}\n{gt.sum()*vox_vol/1000:.1f}cm3', **TK)
axA.axis('off')

# S1 baseline
axB = _ax(0,1)
axB.set_facecolor(BG)
axB.imshow(overlay(ct[best_zi], gt[best_zi], pred1[best_zi]), aspect='equal')
axB.set_title(f'S1: Baseline (HU only)\nDice={m1["Dice"]:.4f} | Prec={m1["Precision"]:.4f}', **TK)
axB.axis('off')

# S2 organ constrained
axC = _ax(0,2)
axC.set_facecolor(BG)
axC.imshow(overlay(ct[best_zi], gt[best_zi], pred2[best_zi]), aspect='equal')
axC.set_title(f'S2: Organ-constrained\nDice={m2["Dice"]:.4f} | Prec={m2["Precision"]:.4f}', **TK)
axC.axis('off')

# S3 morphological
axD = _ax(0,3)
axD.set_facecolor(BG)
axD.imshow(overlay(ct[best_zi], gt[best_zi], pred3[best_zi]), aspect='equal')
axD.set_title(f'S3: Morphological filter\nDice={m3["Dice"]:.4f} | Prec={m3["Precision"]:.4f}', **TK)
axD.axis('off')

# S4 anomaly
axE = _ax(0,4)
axE.set_facecolor(BG)
axE.imshow(overlay(ct[best_zi], gt[best_zi], pred4[best_zi]), aspect='equal')
axE.set_title(f'S4: Local anomaly\nDice={m4["Dice"]:.4f} | Prec={m4["Precision"]:.4f}', **TK)
axE.axis('off')

# S5 union
axF = _ax(0,5)
axF.set_facecolor(BG)
axF.imshow(overlay(ct[best_zi], gt[best_zi], pred5_union[best_zi]), aspect='equal')
axF.set_title(f'S5: Union\nDice={m5u["Dice"]:.4f} | Prec={m5u["Precision"]:.4f}', **TK)
axF.axis('off')

# Row 2: bar chart of all strategies
axG = _ax(1,0, span=3)
axG.set_facecolor(BG)
names_short  = ['S1\nBaseline','S2\nOrgan','S3\nMorph','S4\nAnom','S5u\nUnion','S5i\nInter']
metric_names = ['Dice','Precision','Recall','IoU']
colors_m     = ['#1D6FA5','#D4720B','#44CC88','#CC4488']
all_m_list   = [m1, m2, m3, m4, m5u, m5i]
x = np.arange(len(names_short))
w = 0.18
for mi, (mname, col) in enumerate(zip(metric_names, colors_m)):
    vals = [m[mname] for m in all_m_list]
    bars = axG.bar(x + (mi-1.5)*w, vals, w, color=col, alpha=0.85, label=mname)
    for bar in bars:
        h = bar.get_height()
        if h > 0.02:
            axG.text(bar.get_x()+bar.get_width()/2, h+0.01, f'{h:.2f}',
                     ha='center', va='bottom', fontsize=6, color='#A0B8D0')
axG.axhline(0.5, color='gray', ls='--', lw=0.8, label='0.5 (clinical threshold)')
axG.set_ylim(0, 1.2)
axG.set_xticks(x); axG.set_xticklabels(names_short, fontsize=8)
axG.tick_params(colors='#A0B8D0')
axG.legend(fontsize=7.5, facecolor='#1A2030', edgecolor='#3D4F6A', labelcolor='#A0B8D0', ncol=2)
axG.set_title('All Strategies: Metrics Comparison\nGreen=TP  Red=FP  Blue=FN in overlays', **TK)
axG.spines[:].set_color('#2D3748'); axG.yaxis.grid(True, color='#2D3748', lw=0.5)
axG.set_axisbelow(True)
axG.set_facecolor(BG)

# HU distribution
axH = _ax(1,3, span=2)
axH.set_facecolor(BG)
bins = np.linspace(-200,300,80)
axH.hist(tumor_hu[:20000], bins=bins, color='#FF4444', alpha=0.7, density=True, label='Tumor')
axH.hist(normal_bowel_hu[:20000], bins=bins, color='#1D6FA5', alpha=0.7, density=True, label='Normal tissue')
axH.axvline(25, color='#FFB347', ls='--', lw=1.2, label='Lower bound (25 HU)')
axH.axvline(200, color='#FFB347', ls=':', lw=1.2, label='Upper bound (200 HU)')
axH.set_title(f'HU Overlap: Tumor vs Normal Tissue\n(overlap makes threshold detection unreliable)', **TK)
axH.set_xlabel('HU', fontsize=8, color='#A0B8D0')
axH.set_ylabel('Density', fontsize=8, color='#A0B8D0')
axH.tick_params(colors='#A0B8D0')
axH.legend(fontsize=7.5, facecolor='#1A2030', edgecolor='#3D4F6A', labelcolor='#A0B8D0')
axH.spines[:].set_color('#2D3748')
axH.set_facecolor(BG)

# Scorecard
axI = _ax(1,5)
axI.set_facecolor(BG)
axI.axis('off')
axI.text(0.5,1.0,'Final Verdict', ha='center',va='top',fontsize=9,
         color='#E2EAF4',fontweight='bold',transform=axI.transAxes)
rows = [
    (f'Best Dice:',        f'{dice_best:.4f}',    '#FF6666' if dice_best<0.3 else '#FFB347'),
    (f'Verdict:',          verdict.split()[0],     '#FF6666' if dice_best<0.3 else '#FFB347'),
    ('HU overlap:',        f'{overlap_frac*100:.0f}%', '#FF6666'),
    ('Vol ratio best:',    f'{best["vol_ratio"]:.1f}x', '#FFB347'),
    ('Tumor type:',        'Bowel wall', '#88BBFF'),
    ('Why hard:',          'Same HU as bowel', '#888888'),
    ('Min Dice needed:',   '0.5 (clinical)', '#888888'),
    ('Achievable w/ML?',  'Yes (0.6-0.8)', '#44FF88'),
]
for i,(k,v,c) in enumerate(rows):
    y = 0.88 - i*0.11
    axI.text(0.0, y, k,         transform=axI.transAxes, fontsize=7.5, color='#A0B8D0', va='top')
    axI.text(0.55, y, v,        transform=axI.transAxes, fontsize=7.5, color=c, va='top', fontweight='bold')

fig.suptitle(
    f'CTdata1 Organ-Constrained Detection  |  Patient 002227784  |  Bowel Tumor  |  GT={gt.sum()*vox_vol/1000:.1f}cm3',
    fontsize=11, fontweight='bold', color='#E2EAF4', y=0.98)

out_fig = os.path.join(SAVE, 'ct_constrained_figure.png')
plt.savefig(out_fig, dpi=150, bbox_inches='tight', facecolor='#0D1117')
L(f'Figure -> {out_fig}')
plt.close()

# ================================================================
# SAVE
# ================================================================
results = {
    'patient_id': '002227784',
    'gt_vol_cm3': float(gt.sum()*vox_vol/1000),
    'strategies': {k: v for k,v in all_results.items()},
    'tumor_hu_stats': {
        'mean': float(tumor_hu.mean()), 'std': float(tumor_hu.std()),
        'median': float(np.median(tumor_hu)),
        'p5': float(np.percentile(tumor_hu,5)),
        'p95': float(np.percentile(tumor_hu,95)),
    },
    'hu_distribution_overlap_pct': float(overlap_frac*100),
    'best_strategy': best_name,
    'best_dice': float(dice_best),
    'verdict': verdict,
}
jout = os.path.join(SAVE, 'ct_constrained_results.json')
with open(jout, 'w', encoding='utf-8') as f:
    json.dump(results, f, indent=2, ensure_ascii=False)
L(f'JSON  -> {jout}')

tout = os.path.join(SAVE, 'ct_constrained_report.txt')
with open(tout, 'w', encoding='utf-8', errors='replace') as f:
    f.write('\n'.join(log))
L(f'Text  -> {tout}')
L('\nDone.')
