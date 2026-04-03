"""
Final Evaluation: nnU-Net Multi-Dataset + YOLO Results
=======================================================
Evaluates:
  1. CTdata1 Arterial nnU-Net (baseline)
  2. CTdata1 Delay nnU-Net
  3. CTdata2 Pre (0930) nnU-Net -- different patient/timepoint
  4. YOLO YOLOv8s-seg (spatial overlap via bbox)
"""
import os, json, glob, warnings
import numpy as np
import nibabel as nib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import ndimage
from skimage import measure
from skimage.transform import resize as sk_resize

warnings.filterwarnings('ignore')
DATA1 = r'F:\ADDS\CTdata1'
SAVE  = r'F:\ADDS'

# Load GT
art   = nib.load(os.path.join(DATA1,'nifti','inha_ct_arterial.nii.gz'))
tmask = nib.load(os.path.join(DATA1,'tumor_masks','tumor_mask_binary.nii.gz'))
ct    = art.get_fdata().astype(np.float32)
gt    = tmask.get_fdata().astype(np.uint8)
zooms = np.array(art.header.get_zooms(), dtype=float)
vox_vol = float(np.prod(zooms))
Z,H,W = ct.shape

def metrics(pred, truth):
    p=pred.astype(bool); t=truth.astype(bool)
    tp=int((p&t).sum()); fp=int((p&~t).sum()); fn=int((~p&t).sum()); tn=int((~p&~t).sum())
    dice=2*tp/(2*tp+fp+fn+1e-9); iou=tp/(tp+fp+fn+1e-9)
    prec=tp/(tp+fp+1e-9); rec=tp/(tp+fn+1e-9)
    return {'Dice':round(dice,4),'IoU':round(iou,4),'Prec':round(prec,4),
            'Rec':round(rec,4),'TP':tp,'FP':fp,'FN':fn,
            'vol_cm3':round(float(pred.sum()*vox_vol/1000),2)}

print('='*68)
print('FINAL MULTI-DATASET EVALUATION')
print('='*68)
print(f'GT: {int(gt.sum()):,} vox = {gt.sum()*vox_vol/1000:.2f} cm3')
print()

all_results = {}

# ----------------------------------------------------------------
# 1. Previous results (already known)
# ----------------------------------------------------------------
all_results['S3_organ_constrained'] = {'Dice':0.1389,'IoU':0.0747,
    'Prec':0.0747,'Rec':0.7083,'vol_cm3':173.4,'note':'best HU-based'}
all_results['nnUNet_Arterial'] = {'Dice':0.0000,'IoU':0.0000,
    'Prec':0.0000,'Rec':0.0000,'vol_cm3':3.6,'note':'domain mismatch'}

# ----------------------------------------------------------------
# 2. CTdata1 Delay nnU-Net
# ----------------------------------------------------------------
print('[1] CTdata1 Delay nnU-Net:')
delay_out = glob.glob(os.path.join(SAVE,'nnunet_delay_output','*.nii.gz'))
if delay_out:
    pred_nii = nib.load(delay_out[0])
    pred_d   = pred_nii.get_fdata().astype(np.int32)
    # Match shape to GT if needed
    if pred_d.shape != gt.shape:
        pred_d = (sk_resize(pred_d.astype(float), gt.shape, order=0,
                            anti_aliasing=False, preserve_range=True) > 0.5).astype(np.int32)
    m_delay = metrics(pred_d > 0, gt)
    all_results['nnUNet_CTdata1_Delay'] = m_delay
    print(f'  Dice={m_delay["Dice"]:.4f}  Prec={m_delay["Prec"]:.4f}  Rec={m_delay["Rec"]:.4f}  vol={m_delay["vol_cm3"]:.1f}cm3')
    print(f'  labels: {np.unique(pred_d).tolist()}')
else:
    print('  Output not ready yet.')
    all_results['nnUNet_CTdata1_Delay'] = {'Dice':0,'IoU':0,'Prec':0,'Rec':0,'vol_cm3':0,'note':'not ready'}

# ----------------------------------------------------------------
# 3. CTdata2 Pre (0930) nnU-Net - different patient, no GT match expected
# ----------------------------------------------------------------
print('\n[2] CTdata2 Pre (Abdomen 5mm) nnU-Net:')
pre_out = glob.glob(os.path.join(SAVE,'nnunet_0930_output','*.nii.gz'))
if pre_out:
    pred_nii2 = nib.load(pre_out[0])
    pred_0930 = pred_nii2.get_fdata().astype(np.int32)
    print(f'  Shape: {pred_0930.shape}  labels: {np.unique(pred_0930).tolist()}')
    fg_vox = int((pred_0930 > 0).sum())
    # Load original 0930 volume for voxel volume
    map_path = os.path.join(SAVE, 'ctdata2_nifti_mapping.json')
    with open(map_path) as f: mapping = json.load(f)
    m0930 = mapping.get('0930_rows512_n210',{}).get('meta',{})
    sp0930 = m0930.get('spacing',[1,1])
    th0930 = m0930.get('thick',5)
    vv0930 = sp0930[0]*sp0930[1]*th0930
    print(f'  Foreground voxels: {fg_vox:,} = {fg_vox*vv0930/1000:.1f} cm3')
    print(f'  NOTE: CTdata2 Pre is a different patient/time. No GT available for Dice.')
    all_results['nnUNet_CTdata2_Pre'] = {
        'Dice':'N/A','note':'no_GT',
        'fg_voxels': fg_vox,
        'vol_cm3': round(fg_vox*vv0930/1000,1),
        'labels': np.unique(pred_0930).tolist()
    }
else:
    print('  Output not ready.')
    all_results['nnUNet_CTdata2_Pre'] = {'Dice':'N/A','note':'not_ready'}

# ----------------------------------------------------------------
# 4. YOLO: Convert bounding boxes to pixelwise masks
# ----------------------------------------------------------------
print('\n[3] YOLO YOLOv8s-seg evaluation:')
try:
    from ultralytics import YOLO
    manifest_path = os.path.join(SAVE, 'yolo_manifest.json')
    with open(manifest_path) as f:
        manifest = json.load(f)

    model = YOLO('yolov8s-seg.pt')
    PNG_DIR = os.path.join(SAVE, 'yolo_crops')
    gt_slices_info = manifest['gt_slices']

    pred_vol_yolo  = np.zeros_like(gt, dtype=bool)
    n_processed = 0
    n_detections = 0

    for info in gt_slices_info:
        z   = info['z']
        r0,r1,c0,c1 = info['crop_bbox']
        zoom_path = info['zoom_path']
        if not os.path.exists(zoom_path): continue

        results = model(zoom_path, verbose=False, conf=0.05, iou=0.3)
        r = results[0]
        n_processed += 1

        if r.masks is not None and len(r.masks) > 0:
            # Each mask is 256x256 (scaled) -> rescale to H x W
            for mask_data in r.masks.data:
                mask_arr = mask_data.cpu().numpy()  # shape [h, w]
                # Map back from zoomed crop to original slice coordinates
                mask_full = sk_resize(mask_arr.astype(float),
                                      (r1-r0, c1-c0), anti_aliasing=False) > 0.5
                # Place into full slice
                tmp = np.zeros((H,W), dtype=bool)
                tmp[r0:r1, c0:c1] = mask_full
                pred_vol_yolo[z] |= tmp
                n_detections += 1
        elif r.boxes is not None and len(r.boxes) > 0:
            # No segmentation mask -- use bounding boxes
            for box in r.boxes.xyxy.cpu().numpy():
                x1,y1,x2,y2 = [int(v) for v in box]
                # Map from zoomed coordinates back to crop coordinates
                zoom_h, zoom_w = H, W
                crop_h = r1 - r0; crop_w = c1 - c0
                # Scale factor: zoom image is H x W but crop was (r1-r0) x (c1-c0)
                sx = crop_h / zoom_h; sy_c = crop_w / zoom_w
                ry1 = r0 + int(y1 * sx); ry2 = r0 + int(y2 * sx)
                cx1 = c0 + int(x1 * sy_c); cx2 = c0 + int(x2 * sy_c)
                ry1=max(0,min(ry1,H-1)); ry2=max(0,min(ry2,H-1))
                cx1=max(0,min(cx1,W-1)); cx2=max(0,min(cx2,W-1))
                pred_vol_yolo[z, ry1:ry2, cx1:cx2] = True
                n_detections += 1

    m_yolo = metrics(pred_vol_yolo, gt)
    all_results['YOLO_YOLOv8s_seg'] = m_yolo
    print(f'  Processed: {n_processed} slices  Detections: {n_detections}')
    print(f'  Dice={m_yolo["Dice"]:.4f}  Prec={m_yolo["Prec"]:.4f}  Rec={m_yolo["Rec"]:.4f}  vol={m_yolo["vol_cm3"]:.1f}cm3')

except Exception as e:
    print(f'  YOLO evaluation error: {e}')
    all_results['YOLO_YOLOv8s_seg'] = {'Dice':0,'error':str(e)}

# ----------------------------------------------------------------
# FINAL SCORECARD
# ----------------------------------------------------------------
print()
print('='*68)
print('FINAL SCORECARD -- All Methods:')
print(f'{"Method":<30} {"Dice":>7} {"Prec":>7} {"Rec":>7} {"vol":>8}  Note')
print('-'*68)
all_historical = {
    'S3_organ_constrained':0.1389,
    'HU_threshold_v1':0.0105,
    'Artery-Delay':0.0170,
    'v4_HU_corrected':0.0082,
    'nnUNet_Arterial':0.0000,
}
for name, m in all_results.items():
    d = m['Dice'] if isinstance(m['Dice'], float) else 0.0
    p = m.get('Prec',0) if isinstance(m.get('Prec',0),float) else 0.0
    r = m.get('Rec',0) if isinstance(m.get('Rec',0),float) else 0.0
    v = m.get('vol_cm3',0) if isinstance(m.get('vol_cm3',0),(int,float)) else 0.0
    note = m.get('note','')
    print(f'  {name[:28]:<30} {d:>7.4f} {p:>7.4f} {r:>7.4f} {v:>8.1f}  {note}')
print()
# Best Dice
best_key  = max((k for k in all_results if isinstance(all_results[k].get('Dice',0),float)),
                key=lambda k: all_results[k].get('Dice',0))
best_dice = all_results[best_key]['Dice']
print(f'BEST: {best_key}  Dice={best_dice:.4f}')
print(f'vs clinical-min 0.5: gap = {0.5-best_dice:.4f}')

# Save
out_json = os.path.join(SAVE, 'ct_final_result_all.json')
safe_results = {}
for k,v in all_results.items():
    safe_results[k] = {kk:(float(vv) if isinstance(vv,float) else
                            int(vv) if isinstance(vv,int) else str(vv))
                        for kk,vv in v.items() if not isinstance(vv,np.ndarray)}
with open(out_json,'w',encoding='utf-8') as f:
    json.dump(safe_results,f,indent=2,ensure_ascii=False)
print(f'\nJSON -> {out_json}')

# ----------------------------------------------------------------
# FIGURE: Full comparison
# ----------------------------------------------------------------
def hw(v,wl=60,ww=350):
    lo,hi=wl-ww/2,wl+ww/2
    return (np.clip(v,lo,hi)-lo)/(hi-lo)
def overlay(ct2d, gt2d, pred2d, sz=256):
    ct_n=sk_resize(hw(ct2d),(sz,sz),anti_aliasing=True)
    g2  =sk_resize(gt2d.astype(float),(sz,sz),anti_aliasing=False)>0.5
    p2  =sk_resize(pred2d.astype(float),(sz,sz),anti_aliasing=False)>0.5
    rgb=np.stack([ct_n]*3,-1)
    tp=p2&g2; fp=p2&~g2; fn=~p2&g2
    rgb[tp,0]=0.1; rgb[tp,1]=0.9; rgb[tp,2]=0.1
    rgb[fp,0]=0.9; rgb[fp,1]=0.1; rgb[fp,2]=0.1
    rgb[fn,0]=0.1; rgb[fn,1]=0.1; rgb[fn,2]=0.9
    return rgb

best_z = max(range(Z), key=lambda z: int(gt[z].sum()))

fig = plt.figure(figsize=(28,12), facecolor='#0D1117')
gs  = gridspec.GridSpec(2,6,figure=fig,left=0.02,right=0.99,
                        top=0.93,bottom=0.05,wspace=0.12,hspace=0.28)
TK  = dict(fontsize=7.5,color='#8BAFD4',fontweight='bold',pad=3)
BG  = '#161B22'
def _ax(r,c,sp=1): return fig.add_subplot(gs[r,c:c+sp] if sp>1 else gs[r,c])

# GT
ax=_ax(0,0); ax.set_facecolor(BG)
gt2d=sk_resize(gt[best_z].astype(float),(256,256))>0.5
ct_n=sk_resize(hw(ct[best_z]),(256,256),anti_aliasing=True)
rgb=np.stack([ct_n]*3,-1); rgb[gt2d,0]=0.1; rgb[gt2d,1]=0.9; rgb[gt2d,2]=0.1
ax.imshow(rgb,aspect='equal'); ax.axis('off')
ax.set_title(f'GT z={best_z}\n{gt.sum()*vox_vol/1000:.1f}cm3',**TK)

# YOLO
ax2=_ax(0,1); ax2.set_facecolor(BG)
ax2.imshow(overlay(ct[best_z],gt[best_z],pred_vol_yolo[best_z] if 'pred_vol_yolo' in dir() else np.zeros((H,W),bool)),aspect='equal')
ax2.axis('off')
yolo_dice = all_results.get('YOLO_YOLOv8s_seg',{}).get('Dice',0)
ax2.set_title(f'YOLO YOLOv8s-seg\nDice={yolo_dice:.4f}',**TK)

# nnU-Net Delay
ax3=_ax(0,2); ax3.set_facecolor(BG)
delay_dice = all_results.get('nnUNet_CTdata1_Delay',{}).get('Dice',0)
if delay_out:
    pred_delay_vol = nib.load(delay_out[0]).get_fdata().astype(np.int32)
    if pred_delay_vol.shape != gt.shape:
        pred_delay_vol = (sk_resize(pred_delay_vol.astype(float),gt.shape,order=0,
                                    anti_aliasing=False,preserve_range=True)>0.5).astype(np.int32)
    ax3.imshow(overlay(ct[best_z],gt[best_z],pred_delay_vol[best_z]>0),aspect='equal')
else:
    ax3.imshow(overlay(ct[best_z],gt[best_z],np.zeros((H,W),bool)),aspect='equal')
ax3.axis('off')
ax3.set_title(f'nnU-Net Delay Phase\nDice={delay_dice:.4f}',**TK)

# 2x Zoom YOLO slice
ax4=_ax(0,3); ax4.set_facecolor(BG)
rows,cols=np.where(gt[best_z]>0) if gt[best_z].sum()>0 else ([H//2],[W//2])
cr,cc=int(rows.mean()),int(cols.mean())
margin_px=int(30/zooms[0]); half_h=H//4; half_w=W//4
r0=max(0,cr-half_h-margin_px); r1=min(H,cr+half_h+margin_px)
c0=max(0,cc-half_w-margin_px); c1=min(W,cc+half_w+margin_px)
crop=sk_resize(hw(ct[best_z,r0:r1,c0:c1]),(256,256),anti_aliasing=True)
gt_crop=sk_resize(gt[best_z,r0:r1,c0:c1].astype(float),(256,256))>0.5
yolo_crop=sk_resize(pred_vol_yolo[best_z,r0:r1,c0:c1].astype(float),(256,256))>0.5 if 'pred_vol_yolo' in dir() else np.zeros((256,256),bool)
rgb2=np.stack([crop]*3,-1)
tp=yolo_crop&gt_crop; fp=yolo_crop&~gt_crop; fn=~yolo_crop&gt_crop
rgb2[tp,0]=0.1; rgb2[tp,1]=0.9; rgb2[tp,2]=0.1
rgb2[fp,0]=0.9; rgb2[fp,1]=0.1; rgb2[fp,2]=0.1
rgb2[fn,0]=0.1; rgb2[fn,1]=0.1; rgb2[fn,2]=0.9
ax4.imshow(rgb2,aspect='equal'); ax4.axis('off')
ax4.set_title(f'2x Zoom YOLO z={best_z}\n(Green=TP, Red=FP, Blue=FN)',**TK)

# CTdata2 Pre detection
ax5=_ax(0,4); ax5.set_facecolor(BG)
if pre_out:
    pred2_nii = nib.load(pre_out[0])
    pred2_vol = pred2_nii.get_fdata()
    # Show middle slice
    mid2 = pred2_vol.shape[0]//2
    ax5.imshow(pred2_vol[mid2], cmap='nipy_spectral', aspect='equal')
ax5.axis('off')
ax5.set_title(f'CTdata2 Pre nnU-Net\n(Different patient -- no GT)',**TK)

# Bar chart
ax6=fig.add_subplot(gs[0,5]); ax6.set_facecolor(BG); ax6.axis('off')
ax6.text(0.5,1.0,'SCORECARD',ha='center',va='top',fontsize=8.5,
         color='#E2EAF4',fontweight='bold',transform=ax6.transAxes)
sorted_r=sorted([(k,v) for k,v in all_results.items()
                 if isinstance(v.get('Dice',0),(int,float))],
                key=lambda x:-x[1].get('Dice',0))
for i,(n,m) in enumerate(sorted_r[:8]):
    y=0.92-i*0.113
    d=m.get('Dice',0)
    c='#44FF88' if d>0.3 else ('#FFB347' if d>0.1 else '#FF4444')
    ax6.text(0,y,n[:25],transform=ax6.transAxes,fontsize=5.5,color='#A0B8D0',va='top')
    ax6.text(0.72,y,f'{d:.4f}',transform=ax6.transAxes,fontsize=6.5,color=c,
             va='top',fontweight='bold')

# Row 2: Full comparison bar
ax7=fig.add_subplot(gs[1,0:4]); ax7.set_facecolor(BG)
method_names = [k for k,v in all_results.items() if isinstance(v.get('Dice',0),(int,float))]
dice_vals = [all_results[k].get('Dice',0) for k in method_names]
colors = ['#FF4444' if d<0.1 else '#FF8800' if d<0.3 else '#FFD700' if d<0.5 else '#44FF88'
          for d in dice_vals]
x = np.arange(len(method_names))
bars = ax7.bar(x, dice_vals, color=colors, alpha=0.85, edgecolor='#2D3748')
ax7.axhline(0.5,color='#AAFFAA',ls='--',lw=1.2,label='0.5 clinical min')
ax7.axhline(0.7,color='#FFAA00',ls=':',lw=1.0,label='0.7 publication')
ax7.set_xticks(x)
ax7.set_xticklabels([n.replace('_','\n') for n in method_names],fontsize=6.5,color='#A0B8D0')
ax7.tick_params(colors='#A0B8D0')
for bar,v in zip(bars,dice_vals):
    if v>0.01:
        ax7.text(bar.get_x()+bar.get_width()/2,v+0.01,f'{v:.4f}',
                 ha='center',fontsize=7,color='#E2EAF4',fontweight='bold')
ax7.set_ylim(0,0.8); ax7.set_ylabel('Dice',fontsize=7.5,color='#A0B8D0')
ax7.legend(fontsize=7,facecolor='#1A2030',edgecolor='#3D4F6A',labelcolor='#A0B8D0')
ax7.set_title(f'All Methods: Dice Comparison  |  Best={best_dice:.4f} ({best_key})',**TK)
ax7.spines[:].set_color('#2D3748'); ax7.yaxis.grid(True,color='#2D3748',lw=0.5)
ax7.set_axisbelow(True); ax7.set_facecolor(BG)

# Verdict
ax8=fig.add_subplot(gs[1,4:6]); ax8.set_facecolor(BG); ax8.axis('off')
ax8.text(0.5,1.0,'VERDICT & NEXT STEP',ha='center',va='top',fontsize=8,
         color='#E2EAF4',fontweight='bold',transform=ax8.transAxes)
rows_v=[
    ('Best method:',      best_key[:20], '#88BBFF'),
    ('Best Dice:',        f'{best_dice:.4f}', '#FFB347' if best_dice<0.3 else '#44FF88'),
    ('Clinical 0.5:',     'NOT MET' if best_dice<0.5 else 'MET',
                           '#FF4444' if best_dice<0.5 else '#44FF88'),
    ('YOLO (general):',   f'{yolo_dice:.4f}', '#FFB347' if yolo_dice>0.05 else '#FF4444'),
    ('nnU-Net Delay:',    f'{delay_dice:.4f}', '#FFB347' if delay_dice>0.05 else '#FF4444'),
    ('Root cause:',       'Mucinous->no boundary','#FF6666'),
    ('YOLO potential:',   'High with medical weights','#44FF88'),
    ('Next=MedSAM/SAM2:', 'Zero-shot seg model','#44FF88'),
    ('CTdata2 Pre:',      'Abdomen pre-contrast', '#88BBFF'),
]
for i,(k,v,c) in enumerate(rows_v):
    y=0.93-i*0.103
    ax8.text(0,y,k,transform=ax8.transAxes,fontsize=7,color='#A0B8D0',va='top')
    ax8.text(0.55,y,v,transform=ax8.transAxes,fontsize=7,color=c,va='top',fontweight='bold')

fig.suptitle(f'Multi-Dataset CT Pipeline: CTdata1+2 | YOLO+nnU-Net | Best Dice={best_dice:.4f}',
             fontsize=10,fontweight='bold',color='#E2EAF4',y=0.98)

out_fig = os.path.join(SAVE,'ct_final_comparison.png')
plt.savefig(out_fig,dpi=150,bbox_inches='tight',facecolor='#0D1117')
print(f'Figure -> {out_fig}')
plt.close()
print('\nDone.')
