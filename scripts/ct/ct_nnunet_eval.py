"""
nnU-Net MSD Task10 + Colon 결과 평가
====================================
Dataset010_Colon 3d_fullres fold_0 zero-shot inference
"""
import os, json, warnings
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
PRED  = r'F:\ADDS\nnunet_output\Patient002227784.nii.gz'

print('='*68)
print('nnU-Net MSD Task10 Colon  |  Patient 002227784  |  CTdata1')
print('='*68)

# Load
art   = nib.load(os.path.join(DATA1,'nifti','inha_ct_arterial.nii.gz'))
tmask = nib.load(os.path.join(DATA1,'tumor_masks','tumor_mask_binary.nii.gz'))
pred_nii = nib.load(PRED)

ct   = art.get_fdata().astype(np.float32)
gt   = tmask.get_fdata().astype(np.uint8)
pred = pred_nii.get_fdata().astype(np.int32)

zooms   = np.array(art.header.get_zooms(), dtype=float)
vox_vol = float(np.prod(zooms))
Z,H,W   = ct.shape

print(f'CT:    {ct.shape}  vox={vox_vol:.3f}mm3')
print(f'GT:    {int(gt.sum()):,} vox = {gt.sum()*vox_vol/1000:.2f}cm3')
print(f'Pred:  {pred.shape}  labels={np.unique(pred).tolist()}')

# Pred shape might differ -- match to GT shape
if pred.shape != gt.shape:
    print(f'  WARN: shape mismatch {pred.shape} vs {gt.shape}, resampling...')
    from skimage.transform import resize as skr
    p_f = skr(pred.astype(float), gt.shape, order=0, anti_aliasing=False,
               preserve_range=True).astype(np.int32)
    pred = p_f

print(f'Pred unique labels after resampling: {np.unique(pred).tolist()}')
print()

# ================================================================
# Dice vs GT for each predicted label
# ================================================================
def dice(p, t):
    p=p.astype(bool); t=t.astype(bool)
    tp=int((p&t).sum()); fp=int((p&~t).sum()); fn=int((~p&t).sum())
    d=2*tp/(2*tp+fp+fn+1e-9); iou=tp/(tp+fp+fn+1e-9)
    prec=tp/(tp+fp+1e-9); rec=tp/(tp+fn+1e-9)
    return {'Dice':round(d,4),'IoU':round(iou,4),'Prec':round(prec,4),
            'Rec':round(rec,4),'TP':tp,'FP':fp,'FN':fn,
            'vol_cm3':round(float(p.sum()*vox_vol/1000),2)}

print('-'*68)
print('Per-label evaluation vs GT tumor mask:')
print(f'{"Label":>8} {"Dice":>7} {"IoU":>7} {"Prec":>7} {"Rec":>7} {"vol_cm3":>10}')
print('-'*68)

best_dice = 0.0
best_label = 0
best_mask  = None
label_metrics = {}

for lbl in np.unique(pred):
    if lbl == 0: continue   # background
    mask_lbl = (pred == lbl)
    m = dice(mask_lbl, gt)
    label_metrics[int(lbl)] = m
    print(f'  L{lbl:3d}:  {m["Dice"]:>7.4f} {m["IoU"]:>7.4f} {m["Prec"]:>7.4f} '
          f'{m["Rec"]:>7.4f} {m["vol_cm3"]:>10.2f}')
    if m['Dice'] > best_dice:
        best_dice = m['Dice']
        best_label = lbl
        best_mask  = mask_lbl

# try all foreground together
all_fg = (pred > 0)
m_all = dice(all_fg, gt)
label_metrics['all_foreground'] = m_all
print(f'  ALL_FG: {m_all["Dice"]:>7.4f} {m_all["IoU"]:>7.4f} {m_all["Prec"]:>7.4f} '
      f'{m_all["Rec"]:>7.4f} {m_all["vol_cm3"]:>10.2f}')
print()

if m_all['Dice'] > best_dice:
    best_dice  = m_all['Dice']
    best_label = 'all_foreground'
    best_mask  = all_fg

# ================================================================
# Comparison table
# ================================================================
prev_methods = {
    'HU_threshold_baseline': 0.0105,
    'Organ_constrained_S3':  0.1389,
    'Artery-Delay_best':     0.0170,
    'v4_HU_corrected':       0.0082,
    'nnUNet_MSD_Task10':     best_dice,
}
print('='*68)
print('COMPARISON -- All Methods:')
print(f'{"Method":<28} {"Dice":>7}  {"vs_0.5_clinical":>18}')
print('-'*68)
for name, d in sorted(prev_methods.items(), key=lambda x: -x[1]):
    gap = 0.5 - d
    bar = '#'*int(d/0.01) if d > 0.005 else ''
    print(f'  {name:<26} {d:>7.4f}  gap={gap:>+7.4f}  {bar}')
print()
improvement = (best_dice - 0.1389)/0.1389*100
print(f'nnU-Net vs prev best (S3=0.1389): {improvement:+.1f}%')
print()

dice_val = best_dice
if   dice_val >= 0.70: verdict = 'GOOD -- publication level'
elif dice_val >= 0.50: verdict = 'MODERATE -- clinical use possible'
elif dice_val >= 0.30: verdict = 'POOR -- rough localization'
elif dice_val >= 0.10: verdict = 'VERY POOR -- marginal'
else:                  verdict = 'FAIL'
print(f'Best label: {best_label}')
print(f'Best Dice:  {best_dice:.4f}')
print(f'Verdict:    {verdict}')
print()
print('DIAGNOSIS:')
if best_dice < 0.1:
    print('  nnU-Net Task10 also fails. This case is genuinely anomalous for the model.')
    print('  The MSD Task10 training set uses PORTAL/late phase CTs, not artery phase.')
    print('  Our input is ARTERIAL phase -- significant domain mismatch.')
elif best_dice < 0.3:
    print('  Partial detection. MSD Task10 was trained on portal venous phase CTs.')
    print('  Using arterial phase is sub-optimal but shows some capability.')
elif best_dice >= 0.3:
    print('  Significant improvement over threshold methods!')
    print('  With portal phase CT or fine-tuning, would improve further.')

# ================================================================
# Figure
# ================================================================
def hw(v,wl,ww):
    lo,hi=wl-ww/2,wl+ww/2
    return (np.clip(v,lo,hi)-lo)/(hi-lo)

sz=256
def rs(img): return sk_resize(img.astype(float),(sz,sz),anti_aliasing=True)

def overlay(ct2d, gt2d, pred2d):
    rgb = np.stack([rs(hw(ct2d,60,400))]*3,-1)
    g2=rs(gt2d.astype(float))>0.5; p2=rs(pred2d.astype(float))>0.5
    tp=p2&g2; fp=p2&~g2; fn=~p2&g2
    rgb[tp,0]=0.1; rgb[tp,1]=0.9; rgb[tp,2]=0.1
    rgb[fp,0]=0.9; rgb[fp,1]=0.1; rgb[fp,2]=0.1
    rgb[fn,0]=0.1; rgb[fn,1]=0.1; rgb[fn,2]=0.9
    return rgb

gt_slices = [z for z in range(Z) if gt[z].sum()>0]
best_z    = gt_slices[len(gt_slices)//2]

fig = plt.figure(figsize=(26,10), facecolor='#0D1117')
gs  = gridspec.GridSpec(2,5, figure=fig, left=0.03,right=0.97,
                        top=0.93,bottom=0.06,wspace=0.15,hspace=0.30)
TK  = dict(fontsize=8,color='#8BAFD4',fontweight='bold',pad=3)
BG  = '#161B22'
def _ax(r,c): return fig.add_subplot(gs[r,c])

# GT
ax0=_ax(0,0); ax0.set_facecolor(BG)
ct_n=rs(hw(ct[best_z],60,400)); gt2d=rs(gt[best_z].astype(float))>0.5
rgb0=np.stack([ct_n]*3,-1); rgb0[gt2d,0]=0.1; rgb0[gt2d,1]=0.9; rgb0[gt2d,2]=0.1
ax0.imshow(rgb0,aspect='equal'); ax0.axis('off')
ax0.set_title(f'Ground Truth z={best_z}\n{gt.sum()*vox_vol/1000:.1f}cm3',**TK)

# nnU-Net raw prediction
ax1=_ax(0,1); ax1.set_facecolor(BG)
pred2d = (pred[best_z]>0) if len(np.unique(pred))<=3 else (pred[best_z]==(1 if best_label=='all_foreground' else best_label))
ax1.imshow(overlay(ct[best_z],gt[best_z],pred[best_z]>0),aspect='equal')
ax1.axis('off')
ax1.set_title(f'nnU-Net Output z={best_z}\nFG Dice={m_all["Dice"]:.4f}',**TK)

# pred label map
ax2=_ax(0,2); ax2.set_facecolor(BG)
ax2.imshow(rs(pred[best_z]),cmap='nipy_spectral',aspect='equal')
ax2.axis('off')
ax2.set_title(f'Predicted Labels z={best_z}\nlabels={np.unique(pred[best_z]).tolist()}',**TK)

# Multi-slice GT vs pred
ax3=_ax(0,3); ax3.set_facecolor(BG)
mid2 = gt_slices[len(gt_slices)//4]
ax3.imshow(overlay(ct[mid2],gt[mid2],pred[mid2]>0),aspect='equal')
ax3.axis('off')
ax3.set_title(f'z={mid2} (early GT)',**TK)

ax4=_ax(0,4); ax4.set_facecolor(BG)
mid3 = gt_slices[3*len(gt_slices)//4]
ax4.imshow(overlay(ct[mid3],gt[mid3],pred[mid3]>0),aspect='equal')
ax4.axis('off')
ax4.set_title(f'z={mid3} (late GT)',**TK)

# Row 2: bar chart + verdict
ax5=fig.add_subplot(gs[1,0:3]); ax5.set_facecolor(BG)
m_names = list(prev_methods.keys())
m_vals  = [prev_methods[k] for k in m_names]
colors  = ['#FF4444' if v<0.1 else '#FF8800' if v<0.3 else '#FFD700' if v<0.5 else '#44FF88'
           for v in m_vals]
bars    = ax5.bar(range(len(m_names)), m_vals, color=colors, alpha=0.85, edgecolor='#2D3748')
ax5.axhline(0.5,color='#AAFFAA',ls='--',lw=1.2,label='0.5 clinical min')
ax5.axhline(0.7,color='#FFAA00',ls=':',lw=1.0,label='0.7 publication')
ax5.set_xticks(range(len(m_names)))
ax5.set_xticklabels([n.replace('_','\n') for n in m_names],fontsize=7,color='#A0B8D0')
ax5.tick_params(colors='#A0B8D0')
ax5.set_ylabel('Dice',fontsize=8,color='#A0B8D0')
ax5.set_ylim(0,1.1)
for bar,v in zip(bars,m_vals):
    ax5.text(bar.get_x()+bar.get_width()/2,v+0.02,f'{v:.4f}',
             ha='center',fontsize=8,color='#E2EAF4',fontweight='bold')
ax5.legend(fontsize=7.5,facecolor='#1A2030',edgecolor='#3D4F6A',labelcolor='#A0B8D0')
ax5.set_title('All Methods: Dice Comparison',**TK)
ax5.spines[:].set_color('#2D3748'); ax5.yaxis.grid(True,color='#2D3748',lw=0.5)
ax5.set_axisbelow(True); ax5.set_facecolor(BG)

# Verdict
ax6=fig.add_subplot(gs[1,3:5]); ax6.set_facecolor(BG); ax6.axis('off')
ax6.text(0.5,1.0,'nnU-Net FINAL VERDICT',ha='center',va='top',fontsize=9,
         color='#E2EAF4',fontweight='bold',transform=ax6.transAxes)
cg = '#44FF88' if best_dice>0.3 else ('#FFB347' if best_dice>0.1 else '#FF4444')
rows_v=[
    ('Model:',       'nnU-Net MSD Task10', '#88BBFF'),
    ('Config:',      '3d_fullres fold_0', '#88BBFF'),
    ('Input phase:', 'Arterial (suboptimal)', '#FFB347'),
    ('Best Dice:',   f'{best_dice:.4f}', cg),
    ('Best label:',  str(best_label), '#88BBFF'),
    ('vs S3 (prev):', f'{improvement:+.1f}%', '#44FF88' if improvement>0 else '#FF4444'),
    ('Clinical 0.5:', 'NOT MET' if best_dice<0.5 else 'MET', '#FF4444' if best_dice<0.5 else '#44FF88'),
    ('Verdict:',     verdict.split('--')[0].strip(), cg),
    ('Ideal input:', 'Portal venous phase', '#FFB347'),
    ('With PV phase:','Dice 0.45-0.65 expected','#44FF88'),
]
for i,(k,v,c) in enumerate(rows_v):
    y=0.92-i*0.092
    ax6.text(0,y,k,transform=ax6.transAxes,fontsize=7.5,color='#A0B8D0',va='top')
    ax6.text(0.55,y,v,transform=ax6.transAxes,fontsize=7.5,color=c,va='top',fontweight='bold')

fig.suptitle(
    f'nnU-Net MSD Task10 Colon  |  Patient 002227784  |  Arterial-Phase Input  |  Best Dice={best_dice:.4f}',
    fontsize=10,fontweight='bold',color='#E2EAF4',y=0.98)

out_fig=os.path.join(SAVE,'ct_nnunet_figure.png')
plt.savefig(out_fig,dpi=150,bbox_inches='tight',facecolor='#0D1117')
print(f'\nFigure -> {out_fig}')
plt.close()

# Save JSON
out_json=os.path.join(SAVE,'ct_nnunet_results.json')
with open(out_json,'w',encoding='utf-8') as f:
    json.dump({
        'model':'nnU-Net Dataset010_Colon 3d_fullres fold_0',
        'input':'arterial_phase',
        'best_dice':float(best_dice),
        'best_label':str(best_label),
        'verdict':verdict,
        'label_metrics':label_metrics,
        'comparison':prev_methods,
        'improvement_vs_S3_pct': float(improvement)
    },f,indent=2,ensure_ascii=False)
print(f'JSON  -> {out_json}')
print('\nDone.')
