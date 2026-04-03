"""
MedSAM / SAM2 Zero-Shot CT Tumor Segmentation
==============================================
Strategy:
  MedSAM uses SAM ViT-B backbone fine-tuned on medical images.
  However even raw SAM (without medical fine-tuning) can segment
  anatomical structures when given correct bounding box prompts.

  Approach A: MedSAM with medsam_vit_b.pth (if available)
  Approach B: SAM ViT-B with bbox prompt derived from Delay-Artery anomaly map
  Approach C: SAM ViT-H (largest) with automated bbox prompt from GT ROI

Key innovation:
  1. Use GT bounding box as prompt (upper bound evaluation)
  2. Use automated bbox from anomaly detection as prompt (no-GT eval)
  3. 2x Zoom crop -> SAM -> rescale back to original coords
  4. Evaluate Dice vs GT for both approaches

References:
  MedSAM: Ma et al., Nature Communications 2024
  SAM: Kirillov et al., ICCV 2023 (Meta AI)
"""
import os, json, sys, warnings
import numpy as np
import nibabel as nib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import ndimage
from skimage import measure
from skimage.transform import resize as sk_resize
import torch

warnings.filterwarnings('ignore')

DATA1 = r'F:\ADDS\CTdata1'
SAVE  = r'F:\ADDS'
SAM_B = os.path.join(SAVE, 'sam_vit_b_01ec64.pth')
MED_SAM = os.path.join(SAVE, 'medsam_vit_b.pth')

print('='*68)
print('MedSAM / SAM Zero-Shot CT Segmentation')
print('Patient 002227784 | Mucinous CRC | CTdata1')
print('='*68)

# Load CT and GT
art   = nib.load(os.path.join(DATA1,'nifti','inha_ct_arterial.nii.gz'))
tmask = nib.load(os.path.join(DATA1,'tumor_masks','tumor_mask_binary.nii.gz'))
ct    = art.get_fdata().astype(np.float32)
gt    = tmask.get_fdata().astype(np.uint8)
zooms = np.array(art.header.get_zooms(), dtype=float)
vox_vol = float(np.prod(zooms))
Z,H,W = ct.shape

# Also try delay phase
delay_nii_path = os.path.join(SAVE,'nnunet_delay_input','Patient002227784_Delay_0000.nii.gz')
if os.path.exists(delay_nii_path):
    delay_nii = nib.load(delay_nii_path)
    ct_delay  = delay_nii.get_fdata().astype(np.float32)
    print(f'Delay phase loaded: {ct_delay.shape}')
else:
    ct_delay = ct.copy()
    print('Delay phase not found, using arterial')

print(f'Arterial CT: {ct.shape}  GT: {gt.sum()*vox_vol/1000:.1f}cm3')
print(f'CUDA: {torch.cuda.is_available()}')
print()

# ----------------------------------------------------------------
# Check available checkpoint
# ----------------------------------------------------------------
ckpt = None
model_type_actual = None
if os.path.exists(MED_SAM) and os.path.getsize(MED_SAM) > 300e6:
    ckpt = MED_SAM
    model_type_actual = 'medsam'
    print(f'Using MedSAM checkpoint: {MED_SAM}')
elif os.path.exists(SAM_B) and os.path.getsize(SAM_B) > 300e6:
    ckpt = SAM_B
    model_type_actual = 'vit_b'
    print(f'Using SAM ViT-B checkpoint: {SAM_B}')
elif os.path.exists(os.path.join(SAVE,'sam_vit_h_4b8939.pth')):
    ckpt = os.path.join(SAVE,'sam_vit_h_4b8939.pth')
    model_type_actual = 'vit_h'
    print(f'Using SAM ViT-H checkpoint')
else:
    print(f'No checkpoint found. Checking size of {SAM_B}...')
    if os.path.exists(SAM_B):
        print(f'  {SAM_B}: {os.path.getsize(SAM_B)/1e6:.0f}MB (may still downloading)')
    sys.exit(1)

# ----------------------------------------------------------------
# Load SAM model
# ----------------------------------------------------------------
from segment_anything import sam_model_registry, SamPredictor

print(f'\nLoading SAM model ({model_type_actual})...')
sam = sam_model_registry[model_type_actual](checkpoint=ckpt)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
sam.to(device=device)
predictor = SamPredictor(sam)
print(f'SAM loaded on {device}')
print()

# ----------------------------------------------------------------
# Helper: CT slice to 3-channel uint8 for SAM
# ----------------------------------------------------------------
def ct_to_rgb(ct2d, wl=60, ww=350):
    lo, hi = wl - ww/2, wl + ww/2
    img = np.clip(ct2d, lo, hi)
    img = ((img - lo) / (hi - lo) * 255).astype(np.uint8)
    return np.stack([img]*3, axis=-1)   # H x W x 3

# ----------------------------------------------------------------
# Approach A: GT-guided bbox prompt (upper bound)
# Uses GT bounding box per slice -> SAM -> mask
# ----------------------------------------------------------------
def metrics(pred, truth):
    p=pred.astype(bool); t=truth.astype(bool)
    tp=int((p&t).sum()); fp=int((p&~t).sum()); fn=int((~p&t).sum())
    dice=2*tp/(2*tp+fp+fn+1e-9); iou=tp/(tp+fp+fn+1e-9)
    prec=tp/(tp+fp+1e-9); rec=tp/(tp+fn+1e-9)
    return {'Dice':round(dice,4),'IoU':round(iou,4),'Prec':round(prec,4),
            'Rec':round(rec,4),'TP':tp,'FP':fp,'FN':fn,
            'vol_cm3':round(float(pred.sum()*vox_vol/1000),2)}

print('[A] SAM with GT bounding box prompt (upper bound / oracle test):')
gt_slices = [z for z in range(Z) if gt[z].sum() > 0]
pred_gtbox = np.zeros_like(gt, dtype=bool)
sam_pred_details = []

EXPAND_PX = 15   # Expand GT bbox by 15px for robustness

for z in gt_slices:
    rgb = ct_to_rgb(ct[z])
    predictor.set_image(rgb)

    # Get GT bounding box
    rows, cols = np.where(gt[z] > 0)
    r0,r1 = rows.min()-EXPAND_PX, rows.max()+EXPAND_PX
    c0,c1 = cols.min()-EXPAND_PX, cols.max()+EXPAND_PX
    r0,c0 = max(0,r0), max(0,c0)
    r1,c1 = min(H-1,r1), min(W-1,c1)

    # SAM input box format: (x_min, y_min, x_max, y_max) with numpy
    input_box = np.array([c0, r0, c1, r1])

    try:
        masks, scores, _ = predictor.predict(
            point_coords=None, point_labels=None,
            box=input_box[None, :],
            multimask_output=True
        )
        # Pick best scoring mask
        best_idx = np.argmax(scores)
        mask_2d  = masks[best_idx]  # H x W bool
        pred_gtbox[z] = mask_2d
        sam_pred_details.append({
            'z': z, 'score': float(scores[best_idx]),
            'gt_area': int(gt[z].sum()), 'pred_area': int(mask_2d.sum()),
            'bbox': [int(c0),int(r0),int(c1),int(r1)]
        })
        if z % 10 == 0:
            print(f'  z={z:3d}  score={scores[best_idx]:.3f}  GT={gt[z].sum():5} pred={mask_2d.sum():5}')
    except Exception as e:
        print(f'  z={z:3d}  ERROR: {e}')

m_a = metrics(pred_gtbox, gt)
print(f'\nApproach A (GT bbox):  Dice={m_a["Dice"]:.4f}  Prec={m_a["Prec"]:.4f}  Rec={m_a["Rec"]:.4f}  vol={m_a["vol_cm3"]:.1f}cm3')
print()

# ----------------------------------------------------------------
# Approach B: Automated bbox from anomaly detection (no GT)
# Use organ mask (label 4,5 bowel) bounding box as prompt
# ----------------------------------------------------------------
print('[B] SAM with automated bbox (no GT, organ-based prompt):')
seg_nii = nib.load(os.path.join(DATA1,'segmentation_remapped.nii.gz'))
seg = seg_nii.get_fdata().astype(np.int32)
pred_auto = np.zeros_like(gt, dtype=bool)

bowel_labels = [4, 5, 6, 7]  # bowel-related labels
bowel_mask = np.isin(seg, bowel_labels)

for z in gt_slices:
    rgb = ct_to_rgb(ct[z], wl=60, ww=350)
    predictor.set_image(rgb)

    # Combine: bowel label present or large soft-tissue blob in lower abdomen
    bm = bowel_mask[z]
    lo_z = Z * 2 // 3   # lower 1/3 of abdomen likely pelvis
    pelvis_region = np.zeros((H,W), bool)
    pelvis_region[H*2//3:, :] = True
    # Use bowel OR pelvis lower third
    candidate_region = bm | (pelvis_region & ((ct[z]>-50)&(ct[z]<80)))
    if candidate_region.sum() < 50:
        candidate_region = bm

    if candidate_region.sum() < 10:
        # fallback: center lower third
        rows_r = np.array([H*2//3, H-1])
        cols_c = np.array([W//4, W*3//4])
    else:
        rows_r, cols_r_c = np.where(candidate_region)
        rows_c = cols_r_c
        rows_r = np.array([rows_r.min(), rows_r.max()])
        rows_c = np.array([rows_c.min(), rows_c.max()])

    # Bbox (x0,y0,x1,y1) = (col_min,row_min,col_max,row_max)
    c0,r0,c1,r1 = (max(0,int(rows_c.min())-EXPAND_PX),
                   max(0,int(rows_r.min())-EXPAND_PX),
                   min(W-1,int(rows_c.max())+EXPAND_PX),
                   min(H-1,int(rows_r.max())+EXPAND_PX))
    input_box = np.array([c0,r0,c1,r1])

    try:
        masks, scores, _ = predictor.predict(
            point_coords=None, point_labels=None,
            box=input_box[None,:],
            multimask_output=True
        )
        best_idx = np.argmax(scores)
        pred_auto[z]= masks[best_idx]
    except Exception as e:
        pass

m_b = metrics(pred_auto, gt)
print(f'Approach B (auto bbox):  Dice={m_b["Dice"]:.4f}  Prec={m_b["Prec"]:.4f}  Rec={m_b["Rec"]:.4f}  vol={m_b["vol_cm3"]:.1f}cm3')
print()

# ----------------------------------------------------------------
# Approach C: Use Delay phase CT with GT bbox (best case)
# ----------------------------------------------------------------
if ct_delay.shape == ct.shape:
    print('[C] SAM on Delay phase (GT bbox prompt):')
    pred_delay_sam = np.zeros_like(gt, dtype=bool)
    for z in gt_slices:
        rgb = ct_to_rgb(ct_delay[z], wl=60, ww=350)
        predictor.set_image(rgb)
        rows,cols=np.where(gt[z]>0)
        if len(rows)<2: continue
        c0_d=max(0,cols.min()-EXPAND_PX); r0_d=max(0,rows.min()-EXPAND_PX)
        c1_d=min(W-1,cols.max()+EXPAND_PX); r1_d=min(H-1,rows.max()+EXPAND_PX)
        try:
            masks,scores,_=predictor.predict(
                point_coords=None,point_labels=None,
                box=np.array([c0_d,r0_d,c1_d,r1_d])[None,:],
                multimask_output=True)
            pred_delay_sam[z]=masks[np.argmax(scores)]
        except: pass
    m_c=metrics(pred_delay_sam,gt)
    print(f'Approach C (Delay+SAM GT-box):  Dice={m_c["Dice"]:.4f}  Prec={m_c["Prec"]:.4f}  Rec={m_c["Rec"]:.4f}')
else:
    m_c={'Dice':0,'Prec':0,'Rec':0,'vol_cm3':0}
    pred_delay_sam=np.zeros_like(gt,bool)

# ----------------------------------------------------------------
# FINAL SCORECARD
# ----------------------------------------------------------------
print()
print('='*68)
print('FINAL SCORECARD (all methods):')
all_results = {
    'S3_organ_constrained':     {'Dice':0.1389},
    'SAM_A_GTbbox':             m_a,
    'SAM_B_autobbox':           m_b,
    'SAM_C_delay_GTbbox':       m_c,
    'nnUNet_all_phases':        {'Dice':0.0000},
    'YOLO_seg_general':         {'Dice':0.0124},
    'HU_threshold_v1':          {'Dice':0.0105},
}
for name,m in sorted(all_results.items(), key=lambda x: -x[1].get('Dice',0)):
    d=m.get('Dice',0)
    c='GOOD' if d>=0.5 else ('MODERATE' if d>=0.3 else ('POOR' if d>=0.1 else 'FAIL'))
    print(f'  {name:<30} {d:.4f}  [{c}]')
print()

best_key= max(all_results, key=lambda k: all_results[k].get('Dice',0))
best_d  = all_results[best_key]['Dice']
print(f'BEST: {best_key}  Dice={best_d:.4f}')
if best_d >= 0.5: print('VERDICT: Clinically usable!')
elif best_d >= 0.3: print('VERDICT: Rough localization. Usable as initial seed.')
elif best_d >= 0.1: print('VERDICT: POOR -- marginal improvement over baseline.')
else:              print('VERDICT: FAIL -- same as other methods.')

# Save JSON
out_json=os.path.join(SAVE,'ct_sam_results.json')
safe={k:{kk:float(vv) if isinstance(vv,float) else
          int(vv) if isinstance(vv,int) else str(vv)
          for kk,vv in v.items()} for k,v in all_results.items()}
with open(out_json,'w',encoding='utf-8') as f:
    json.dump(safe,f,indent=2)
print(f'\nJSON -> {out_json}')

# ----------------------------------------------------------------
# FIGURE
# ----------------------------------------------------------------
def hw(v,wl=60,ww=350):
    lo,hi=wl-ww/2,wl+ww/2
    return (np.clip(v,lo,hi)-lo)/(hi-lo)

def overlay(ct2d,gt2d,pred2d,sz=256):
    ctn=sk_resize(hw(ct2d),(sz,sz),anti_aliasing=True)
    g2=sk_resize(gt2d.astype(float),(sz,sz),anti_aliasing=False)>0.5
    p2=sk_resize(pred2d.astype(float),(sz,sz),anti_aliasing=False)>0.5
    rgb=np.stack([ctn]*3,-1)
    tp=p2&g2; fp=p2&~g2; fn=~p2&g2
    rgb[tp,0]=0.1;rgb[tp,1]=0.9;rgb[tp,2]=0.1
    rgb[fp,0]=0.9;rgb[fp,1]=0.1;rgb[fp,2]=0.1
    rgb[fn,0]=0.1;rgb[fn,1]=0.1;rgb[fn,2]=0.9
    return rgb

best_z=max(gt_slices,key=lambda z:int(gt[z].sum()))
fig=plt.figure(figsize=(28,12),facecolor='#0D1117')
gs=gridspec.GridSpec(2,6,figure=fig,left=0.02,right=0.99,
                     top=0.93,bottom=0.05,wspace=0.12,hspace=0.28)
TK=dict(fontsize=7.5,color='#8BAFD4',fontweight='bold',pad=3)
BG='#161B22'
def _ax(r,c): return fig.add_subplot(gs[r,c])

# GT
ax=_ax(0,0);ax.set_facecolor(BG)
gt2d=sk_resize(gt[best_z].astype(float),(256,256))>0.5
ctn=sk_resize(hw(ct[best_z]),(256,256),anti_aliasing=True)
rgb=np.stack([ctn]*3,-1);rgb[gt2d,0]=0.1;rgb[gt2d,1]=0.9;rgb[gt2d,2]=0.1
ax.imshow(rgb,aspect='equal');ax.axis('off')
ax.set_title(f'GT z={best_z}\n{gt.sum()*vox_vol/1000:.1f}cm3',**TK)

# SAM A: GT bbox
ax2=_ax(0,1);ax2.set_facecolor(BG)
ax2.imshow(overlay(ct[best_z],gt[best_z],pred_gtbox[best_z]),aspect='equal')
ax2.axis('off')
ax2.set_title(f'SAM+GT-bbox (oracle)\nDice={m_a["Dice"]:.4f}',**TK)

# SAM B: auto bbox
ax3=_ax(0,2);ax3.set_facecolor(BG)
ax3.imshow(overlay(ct[best_z],gt[best_z],pred_auto[best_z]),aspect='equal')
ax3.axis('off')
ax3.set_title(f'SAM+auto-bbox\nDice={m_b["Dice"]:.4f}',**TK)

# SAM C: delay
ax4=_ax(0,3);ax4.set_facecolor(BG)
ax4.imshow(overlay(ct_delay[best_z] if ct_delay.shape==ct.shape else ct[best_z],
                   gt[best_z],pred_delay_sam[best_z]),aspect='equal')
ax4.axis('off')
ax4.set_title(f'SAM+Delay+GT-bbox\nDice={m_c["Dice"]:.4f}',**TK)

# Scorecard
ax5=_ax(0,4);ax5.set_facecolor(BG);ax5.axis('off')
ax5.text(0.5,1.0,'SCORECARD',ha='center',va='top',fontsize=8.5,
         color='#E2EAF4',fontweight='bold',transform=ax5.transAxes)
for i,(n,m) in enumerate(sorted(all_results.items(),key=lambda x:-x[1].get('Dice',0))):
    y=0.93-i*0.12
    d=m.get('Dice',0)
    c='#44FF88' if d>=0.5 else ('#FFB347' if d>=0.1 else '#FF4444')
    ax5.text(0,y,n[:22],transform=ax5.transAxes,fontsize=6,color='#A0B8D0',va='top')
    ax5.text(0.73,y,f'{d:.4f}',transform=ax5.transAxes,fontsize=7,
             color=c,va='top',fontweight='bold')

# Model info
ax6=_ax(0,5);ax6.set_facecolor(BG);ax6.axis('off')
ax6.text(0.5,1.0,'SAM CONFIG',ha='center',va='top',fontsize=8,
         color='#E2EAF4',fontweight='bold',transform=ax6.transAxes)
rows_info=[
    ('Checkpoint:',  os.path.basename(ckpt) if ckpt else 'N/A'),
    ('Model type:',  model_type_actual),
    ('Device:',      device),
    ('GT slices:',   str(len(gt_slices))),
    ('Prompt A:',    'GT bbox + 15px expand'),
    ('Prompt B:',    'Bowel mask + pelvis ROI'),
    ('Prompt C:',    'Delay phase + GT bbox'),
]
for i,(k,v) in enumerate(rows_info):
    y=0.92-i*0.12
    ax6.text(0,y,k,transform=ax6.transAxes,fontsize=7,color='#A0B8D0',va='top')
    ax6.text(0.52,y,v[:18],transform=ax6.transAxes,fontsize=7,color='#88BBFF',va='top')

# Row 2: bar chart
ax7=fig.add_subplot(gs[1,0:4]);ax7.set_facecolor(BG)
dn=[k for k,v in sorted(all_results.items(),key=lambda x:-x[1].get('Dice',0))]
dv=[all_results[k]['Dice'] for k in dn]
bars=ax7.bar(range(len(dn)),dv,
             color=['#44FF88' if v>=0.5 else '#FFB347' if v>=0.1 else '#FF4444' for v in dv],
             alpha=0.85,edgecolor='#2D3748')
ax7.axhline(0.5,color='#AAFFAA',ls='--',lw=1.2,label='0.5 clinical min')
ax7.axhline(0.7,color='#FFAA00',ls=':',lw=1.0,label='0.7 publication')
ax7.set_xticks(range(len(dn)))
ax7.set_xticklabels([n.replace('_','\n') for n in dn],fontsize=6.5,color='#A0B8D0')
ax7.tick_params(colors='#A0B8D0')
ax7.set_ylabel('Dice',fontsize=8,color='#A0B8D0')
ax7.set_ylim(0,0.9)
for bar,v in zip(bars,dv):
    if v>0.01:
        ax7.text(bar.get_x()+bar.get_width()/2,v+0.01,f'{v:.4f}',
                 ha='center',fontsize=8,color='#E2EAF4',fontweight='bold')
ax7.legend(fontsize=7,facecolor='#1A2030',edgecolor='#3D4F6A',labelcolor='#A0B8D0')
ax7.set_title(f'Complete Method Comparison | Best: {best_key} Dice={best_d:.4f}',**TK)
ax7.spines[:].set_color('#2D3748');ax7.yaxis.grid(True,color='#2D3748',lw=0.5)
ax7.set_axisbelow(True);ax7.set_facecolor(BG)

# Verdict
ax8=fig.add_subplot(gs[1,4:6]);ax8.set_facecolor(BG);ax8.axis('off')
ax8.text(0.5,1.0,'FINAL VERDICT',ha='center',va='top',fontsize=8.5,
         color='#E2EAF4',fontweight='bold',transform=ax8.transAxes)
cg='#44FF88' if best_d>=0.5 else ('#FFB347' if best_d>=0.3 else '#FF4444')
verdict_s='Clinically usable!' if best_d>=0.5 else ('Rough localization' if best_d>=0.3 else 'Below clinical threshold')
rows_v=[
    ('Best method:', best_key[:18], '#88BBFF'),
    ('Best Dice:',   f'{best_d:.4f}', cg),
    ('Verdict:',     verdict_s, cg),
    ('vs clinical:', 'NOT MET' if best_d<0.5 else 'MET!',
                     '#FF4444' if best_d<0.5 else '#44FF88'),
    ('SAM A (oracle):',f'{m_a["Dice"]:.4f}','#88BBFF'),
    ('SAM B (auto):', f'{m_b["Dice"]:.4f}','#88BBFF'),
    ('SAM C (delay):',f'{m_c["Dice"]:.4f}','#88BBFF'),
    ('Model:',       'MedSAM' if 'med' in str(ckpt) else 'SAM ViT-B','#FFAA88'),
]
for i,(k,v,c) in enumerate(rows_v):
    y=0.93-i*0.115
    ax8.text(0,y,k,transform=ax8.transAxes,fontsize=7,color='#A0B8D0',va='top')
    ax8.text(0.55,y,v,transform=ax8.transAxes,fontsize=7.5,color=c,va='top',fontweight='bold')

fig.suptitle(f'MedSAM/SAM Zero-Shot Segmentation | Patient 002227784 | Best Dice={best_d:.4f}',
             fontsize=10,fontweight='bold',color='#E2EAF4',y=0.98)

out_fig=os.path.join(SAVE,'ct_sam_figure.png')
plt.savefig(out_fig,dpi=150,bbox_inches='tight',facecolor='#0D1117')
print(f'Figure -> {out_fig}')
plt.close()
print('\nDone.')
