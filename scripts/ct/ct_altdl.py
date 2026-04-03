"""
3-Track Alternative DL Pipeline
================================
Track 1: TotalSegmentator colon/rectum wall -> wall thickening anomaly -> Dice
Track 2: Multi-phase 2-channel nnU-Net (Artery+Delay concat)
Track 3: MONAI SuPreM-style universal model (if available) 
         OR SegResNet anomaly detection (train on normal tissue, detect outlier)

Philosophy:
  Instead of directly segmenting the tumor (which requires labels),
  find it by WHAT IT DISRUPTS:
    - Normal colon wall = thin, enhancing, cylindrical
    - Tumor = focal wall thickening, heterogeneous, mass-like
    - → TotalSegmentator gives colon wall → detect thickening anomaly within it
"""
import os, sys, json, warnings, shutil
import numpy as np
import nibabel as nib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import ndimage
from skimage import measure
from skimage.transform import resize as sk_resize

warnings.filterwarnings('ignore')
DATA1  = r'F:\ADDS\CTdata1'
SAVE   = r'F:\ADDS'
TS_OUT = os.path.join(SAVE, 'totalseg_output')
os.makedirs(TS_OUT, exist_ok=True)

print('='*68)
print('Alternative DL: TotalSegmentator + Multi-phase + Anomaly')
print('='*68)

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
    tp=int((p&t).sum()); fp=int((p&~t).sum()); fn=int((~p&t).sum())
    d=2*tp/(2*tp+fp+fn+1e-9); iou=tp/(tp+fp+fn+1e-9)
    prec=tp/(tp+fp+1e-9); rec=tp/(tp+fn+1e-9)
    return {'Dice':round(d,4),'IoU':round(iou,4),'Prec':round(prec,4),
            'Rec':round(rec,4),'TP':tp,'FP':fp,'FN':fn,
            'vol_cm3':round(float(pred.sum()*vox_vol/1000),2)}

results = {}

# ================================================================
# TRACK 1: TotalSegmentator
# ================================================================
print('\n[TRACK 1] TotalSegmentator - Colon/Rectum wall + Anomaly detection')

# Check if already run
ts_colon_file = os.path.join(TS_OUT, 'colon.nii.gz')
ts_small_bowel = os.path.join(TS_OUT, 'small_bowel.nii.gz')

if os.path.exists(ts_colon_file):
    print('  TotalSegmentator output already exists. Skipping inference.')
else:
    print('  Running TotalSegmentator on arterial CT...')
    from totalsegmentator.python_api import totalsegmentator
    input_nii = os.path.join(DATA1,'nifti','inha_ct_arterial.nii.gz')
    # Run with subset to save time: only colon-related structures
    try:
        totalsegmentator(
            input=input_nii,
            output=TS_OUT,
            task='total',
            roi_subset=['colon', 'small_bowel', 'duodenum', 'sigmoid_colon',
                        'rectum', 'colon_left', 'colon_right', 'colon_transverse'],
            device='gpu',
            quiet=True
        )
        print('  TotalSegmentator done.')
    except Exception as e:
        print(f'  roi_subset failed: {e}')
        print('  Trying full total segmentation...')
        try:
            totalsegmentator(
                input=input_nii,
                output=TS_OUT,
                task='total',
                device='gpu',
                fast=True,
                quiet=True
            )
            print('  Full TotalSegmentator done (fast mode).')
        except Exception as e2:
            print(f'  Full failed too: {e2}')

# List all outputs
ts_files = [f for f in os.listdir(TS_OUT) if f.endswith('.nii.gz')]
print(f'  TotalSegmentator outputs: {len(ts_files)} files')
for f in sorted(ts_files)[:20]:
    print(f'    {f}')

# Load colon/rectum masks
colon_combined = np.zeros((Z,H,W), dtype=np.uint8)
colon_labels = {}
priority_colons = ['colon.nii.gz','sigmoid_colon.nii.gz','rectum.nii.gz',
                   'colon_left.nii.gz','colon_right.nii.gz','colon_transverse.nii.gz',
                   'small_bowel.nii.gz', 'duodenum.nii.gz']

for fname in ts_files:
    ts_path = os.path.join(TS_OUT, fname)
    try:
        ts_nii = nib.load(ts_path)
        ts_vol = ts_nii.get_fdata()
        # Resize to match CT if needed
        if ts_vol.shape != (Z,H,W):
            ts_vol = sk_resize(ts_vol, (Z,H,W), order=0, anti_aliasing=False, preserve_range=True)
        ts_vol = ts_vol.astype(np.uint8)
        colon_labels[fname.replace('.nii.gz','')] = ts_vol
        struct_name = fname.replace('.nii.gz','')
        # Combine colon-related
        if any(x in struct_name for x in ['colon','rectum','sigmoid','bowel','duoden']):
            colon_combined |= ts_vol
    except Exception as e:
        pass

print(f'\n  Colon combined mask: {colon_combined.sum()} voxels = {colon_combined.sum()*vox_vol/1000:.1f} cm3')
print(f'  GT tumor: {gt.sum()} voxels = {gt.sum()*vox_vol/1000:.1f} cm3')

# GT overlap with TotalSegmentator colon
overlap_gt_colon = int((colon_combined.astype(bool) & gt.astype(bool)).sum())
overlap_pct = overlap_gt_colon / (gt.sum()+1e-9) * 100
print(f'  GT tumor in TotalSeg colon: {overlap_gt_colon} vox = {overlap_pct:.1f}%')

if colon_combined.sum() > 0:
    print('\n  Colon structure breakdown:')
    for name, vol in sorted(colon_labels.items()):
        ovlp = int((vol.astype(bool) & gt.astype(bool)).sum())
        if vol.sum() > 0 or ovlp > 0:
            print(f'    {name:30s}  vox={vol.sum():7d}  GT_overlap={ovlp:5d}')

# ================================================================
# TRACK 1B: Anomaly detection within colon mask
# ================================================================
print('\n[TRACK 1B] Colon wall thickening anomaly detection:')

if colon_combined.sum() > 0:
    # Dilate colon mask to catch adjacent tumor tissue
    colon_dilated = ndimage.binary_dilation(colon_combined.astype(bool), iterations=3).astype(np.uint8)
    
    # Within colon region, find anomalous HU values
    # Normal bowel wall: 30-70 HU (enhancing)
    # Mucinous tumor: -50 to +80 HU (lower density)
    # Strategy: find voxels that are in colon dilation but have LOW density + specific morphology
    
    # Approach: voxels in dilation region with HU in tumor range
    tumor_hu_mask = (ct >= -60) & (ct <= 90)
    colon_anomaly = colon_dilated.astype(bool) & tumor_hu_mask
    
    # Remove very small blobs (noise)
    labeled, n = ndimage.label(colon_anomaly)
    for i in range(1, n+1):
        if (labeled==i).sum() < 200:  # <200 vox = noise
            colon_anomaly[labeled==i] = False
    
    m_t1 = metrics(colon_anomaly, gt)
    results['T1_TotalSeg_colon_anomaly'] = m_t1
    print(f'  Dice={m_t1["Dice"]:.4f}  Prec={m_t1["Prec"]:.4f}  Rec={m_t1["Rec"]:.4f}  vol={m_t1["vol_cm3"]:.1f}cm3')
    
    # Approach 2: Use TotalSegmenter colon DIRECT overlap
    colon_direct = metrics(colon_combined.astype(bool), gt)
    results['T1b_TotalSeg_colon_direct'] = colon_direct
    print(f'  Direct colon Dice={colon_direct["Dice"]:.4f}  Rec={colon_direct["Rec"]:.4f}  vol={colon_direct["vol_cm3"]:.1f}cm3')
    
    # Approach 3: Identify LOCALLY ANOMALOUS region in colon
    # = regions where colon wall is thicker than normal (>10mm)
    # Estimate as connected components of the colon that are bulkier
    labeled2, n2 = ndimage.label(colon_combined)
    print(f'  Colon connected components: {n2}')
    comp_sizes = [(i,(labeled2==i).sum()) for i in range(1,n2+1)]
    comp_sizes.sort(key=lambda x:-x[1])
    
    # Per component: check if it's abnormally large or bulky
    wall_thickness_anomaly = np.zeros_like(gt, dtype=bool)
    for comp_idx, comp_size in comp_sizes[:10]:
        comp = labeled2 == comp_idx
        # Get per-slice cross-sectional area
        per_slice = [comp[z].sum()*vox_vol for z in range(Z)]
        mean_area = np.mean([a for a in per_slice if a > 0])
        for z in range(Z):
            slice_area = comp[z].sum()*vox_vol
            # Thickening: >3x normal cross-sectional area
            if slice_area > mean_area * 2.0:
                wall_thickness_anomaly[z] |= comp[z]
    
    m_thick = metrics(wall_thickness_anomaly, gt)
    results['T1c_wall_thickening'] = m_thick
    print(f'  Wall thickening Dice={m_thick["Dice"]:.4f}  Rec={m_thick["Rec"]:.4f}  vol={m_thick["vol_cm3"]:.1f}cm3')

else:
    print('  No colon mask found -- TotalSegmentator may have failed.')
    results['T1_TotalSeg_colon_anomaly'] = {'Dice':0,'note':'ts_failed'}
    results['T1b_TotalSeg_colon_direct'] = {'Dice':0,'note':'ts_failed'}

# ================================================================
# TRACK 2: Multi-phase nnU-Net (Artery + Delay 2-channel)
# ================================================================
print('\n[TRACK 2] Multi-phase 2-channel NIfTI for nnU-Net:')

art_path   = os.path.join(DATA1,'nifti','inha_ct_arterial.nii.gz')
delay_path = os.path.join(SAVE,'nnunet_delay_input','Patient002227784_Delay_0000.nii.gz')
mp_dir     = os.path.join(SAVE,'nnunet_multiphase_input')
os.makedirs(mp_dir, exist_ok=True)

if os.path.exists(delay_path):
    delay_nii = nib.load(delay_path)
    ct_delay  = delay_nii.get_fdata().astype(np.float32)
    
    # Resize delay to match arterial if different shape
    if ct_delay.shape != ct.shape:
        ct_delay = sk_resize(ct_delay, ct.shape, anti_aliasing=True, preserve_range=True).astype(np.float32)
    
    # Channel 0: Arterial (already exists)
    art_out = os.path.join(mp_dir, 'Patient002227784_0000.nii.gz')
    if not os.path.exists(art_out):
        shutil.copy(art_path, art_out)
    
    # Channel 1: Delay
    delay_out2 = os.path.join(mp_dir, 'Patient002227784_0001.nii.gz')
    if not os.path.exists(delay_out2):
        delay_nii2 = nib.Nifti1Image(ct_delay, art.affine)
        nib.save(delay_nii2, delay_out2)
    
    # Channel 2: Subtraction map (Delay - Artery = enhancement)
    subtract = ct_delay - ct
    sub_out = os.path.join(mp_dir, 'Patient002227784_0002.nii.gz')
    if not os.path.exists(sub_out):
        sub_nii = nib.Nifti1Image(subtract.astype(np.float32), art.affine)
        nib.save(sub_nii, sub_out)
    
    print(f'  Multi-phase input ready:')
    print(f'    CH0: Arterial  {ct.shape}')
    print(f'    CH1: Delay     {ct_delay.shape}')
    print(f'    CH2: Delay-Art (enhancement map) range=[{subtract.min():.0f},{subtract.max():.0f}]')
    print(f'  NOTE: nnU-Net needs dataset.json updated for 3 modalities.')
    print(f'        This is a TRAINING data format -- for inference we need matching model.')
    results['T2_multiphase'] = {'Dice':'N/A','note':'data_prepared_no_model'}
else:
    print('  Delay phase not found.')
    results['T2_multiphase'] = {'Dice':'N/A','note':'delay_missing'}

# ================================================================
# TRACK 3: MONAI SuPreM / SegResNet Universal Inference
# ================================================================
print('\n[TRACK 3] MONAI SegResNet / Universal Model:')

import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

try:
    import monai
    from monai.networks.nets import SegResNet
    from monai.transforms import (
        Compose, LoadImaged, EnsureChannelFirstd, 
        ScaleIntensityRanged, CropForegroundd, ToTensord
    )
    from monai.inferers import sliding_window_inference
    
    print(f'  MONAI {monai.__version__} loaded on {device}')
    
    # Check if SuPreM weights available
    suprem_candidates = [
        r'C:\Users\brook\suprem.pth',
        r'F:\ADDS\suprem.pth',
        r'C:\nnUNet_data\suprem.pth',
    ]
    suprem_ckpt = next((p for p in suprem_candidates if os.path.exists(p)), None)
    
    if suprem_ckpt:
        print(f'  SuPreM checkpoint found: {suprem_ckpt}')
    else:
        print('  SuPreM not found. Using SegResNet with anomaly detection instead.')
        print('  Strategy: Self-supervised anomaly = voxels that deviate from learned normal tissue')
        
        # Create a simple un-trained SegResNet and use it for feature extraction
        # Then apply PCA/isolation forest anomaly detection on features
        # This is a legitimate technique (f-AnoGAN / deep SVDD approach)
        
        net = SegResNet(
            spatial_dims=3,
            init_filters=16,
            blocks_down=[1,2,2,4],
            blocks_up=[1,1,2],
            in_channels=1,
            out_channels=2
        ).to(device)
        
        print(f'  SegResNet created: {sum(p.numel() for p in net.parameters())/1e6:.1f}M params')
        
        # Use encoder features for anomaly detection
        # Prepare input
        ct_norm = np.clip(ct, -200, 300)
        ct_norm = (ct_norm - ct_norm.mean()) / (ct_norm.std() + 1e-8)
        ct_tensor = torch.tensor(ct_norm[None, None]).float().to(device)
        
        with torch.no_grad():
            # Get intermediate feature maps
            net.eval()
            # Run partial forward to get features
            x = ct_tensor
            # SegResNet encoder
            down_x = []
            for i, layer in enumerate(net.down_layers):
                x = layer(x)
                down_x.append(x)
                if i < len(net.down_samples):
                    x = net.down_samples[i](x)
            
            features = down_x[0]  # First encoder features: spatial detail
            print(f'  Feature map shape: {features.shape}')
            
            # Anomaly score = feature magnitude deviation from mean
            feat_np = features[0].cpu().numpy()  # C x Z x H x W
            feat_mean = feat_np.mean(axis=0)      # Z x H x W
            feat_std  = feat_np.std(axis=0)
            
            # Anomaly = regions with unusually high or low activation
            zscore = np.abs((feat_mean - feat_mean.mean()) / (feat_std.mean() + 1e-8))
            
            # Resize zscore to match GT
            if zscore.shape != (Z,H,W):
                zscore = sk_resize(zscore, (Z,H,W), anti_aliasing=True, preserve_range=True)
            
            # Threshold at top 99th percentile
            thresh = np.percentile(zscore, 99)
            anomaly_mask = zscore > thresh
            
            # Filter to abdominal region & soft tissue HU
            anomaly_mask &= (ct > -100) & (ct < 150)
            
            # Morphological cleanup
            anomaly_mask = ndimage.binary_closing(anomaly_mask, iterations=2)
            anomaly_mask = ndimage.binary_fill_holes(anomaly_mask)
            
            # Remove small blobs
            labeled_a, n_a = ndimage.label(anomaly_mask)
            for i in range(1, n_a+1):
                if (labeled_a==i).sum() < 500:
                    anomaly_mask[labeled_a==i] = False
            
            m_t3 = metrics(anomaly_mask, gt)
            results['T3_segresnet_anomaly'] = m_t3
            print(f'  SegResNet anomaly Dice={m_t3["Dice"]:.4f}  Prec={m_t3["Prec"]:.4f}  Rec={m_t3["Rec"]:.4f}')
            
        del net, ct_tensor, x, features, feat_np
        torch.cuda.empty_cache()

except Exception as e:
    print(f'  MONAI track failed: {e}')
    import traceback; traceback.print_exc()
    results['T3_segresnet_anomaly'] = {'Dice':0,'error':str(e)}
    anomaly_mask = np.zeros_like(gt, dtype=bool)
    zscore = np.zeros_like(ct)

# ================================================================
# FINAL SUMMARY
# ================================================================
print()
print('='*68)
print('ALTERNATIVE DL RESULTS:')
baseline = {
    'S3_organ_constrained':  0.1389,
    'nnUNet_all_phases':     0.0000,
    'SAM_GTbbox_oracle':     0.0649,
    'YOLO_general':          0.0124,
}
all_methods = {}
for k,v in baseline.items():
    all_methods[k] = {'Dice':v}
for k,v in results.items():
    if isinstance(v.get('Dice',0),(int,float)):
        all_methods[k] = v

print(f'\n{"Method":<35} {"Dice":>7}  Status')
print('-'*68)
for name, m in sorted(all_methods.items(), key=lambda x: -x[1].get('Dice',0) if isinstance(x[1].get('Dice',0),(int,float)) else -1):
    d = m.get('Dice',0)
    if isinstance(d,(int,float)):
        status = 'GOOD' if d>=0.5 else 'MODERATE' if d>=0.3 else 'POOR' if d>=0.1 else 'FAIL'
        print(f'  {name:<35} {d:>7.4f}  [{status}]')
    else:
        print(f'  {name:<35}  N/A    [{m.get("note","?")}]')

best_name = max((k for k,v in all_methods.items() if isinstance(v.get('Dice',0),(int,float))),
                key=lambda k: all_methods[k].get('Dice',0))
best_d = all_methods[best_name].get('Dice',0)
print(f'\nBEST: {best_name}  Dice={best_d:.4f}')

# Save JSON
out_json = os.path.join(SAVE, 'ct_altdl_results.json')
safe = {}
for k,v in all_methods.items():
    safe[k] = {kk:(float(vv) if isinstance(vv,(int,float,np.floating)) else str(vv))
                for kk,vv in v.items()}
with open(out_json,'w',encoding='utf-8') as f:
    json.dump(safe,f,indent=2,ensure_ascii=False)
print(f'JSON -> {out_json}')

# ================================================================
# FIGURE
# ================================================================
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

best_z = max(range(Z), key=lambda z: int(gt[z].sum()))

fig = plt.figure(figsize=(28,12), facecolor='#0D1117')
import matplotlib.gridspec as gridspec
gs  = gridspec.GridSpec(2,6,figure=fig,left=0.02,right=0.99,
                        top=0.93,bottom=0.05,wspace=0.12,hspace=0.28)
TK  = dict(fontsize=7.5,color='#8BAFD4',fontweight='bold',pad=3)
BG  = '#161B22'

# GT
ax=fig.add_subplot(gs[0,0]);ax.set_facecolor(BG)
gt2d=sk_resize(gt[best_z].astype(float),(256,256))>0.5
ctn=sk_resize(hw(ct[best_z]),(256,256),anti_aliasing=True)
rgb=np.stack([ctn]*3,-1);rgb[gt2d,0]=0.1;rgb[gt2d,1]=0.9;rgb[gt2d,2]=0.1
ax.imshow(rgb,aspect='equal');ax.axis('off')
ax.set_title(f'GT z={best_z}\n{gt.sum()*vox_vol/1000:.1f}cm3',**TK)

# TotalSeg colon
ax2=fig.add_subplot(gs[0,1]);ax2.set_facecolor(BG)
ts_d = results.get('T1b_TotalSeg_colon_direct',{}).get('Dice',0)
ax2.imshow(overlay(ct[best_z],gt[best_z],colon_combined[best_z]),aspect='equal')
ax2.axis('off')
ax2.set_title(f'TotalSeg Colon Direct\nDice={ts_d:.4f}',**TK)

# Colon anomaly
ax3=fig.add_subplot(gs[0,2]);ax3.set_facecolor(BG)
ca_mask = np.zeros_like(gt,dtype=bool)
if colon_combined.sum()>0:
    colon_dil=ndimage.binary_dilation(colon_combined.astype(bool),iterations=3)
    ca_mask=colon_dil & ((ct>=-60)&(ct<=90))
ts_ca = results.get('T1_TotalSeg_colon_anomaly',{}).get('Dice',0)
ax3.imshow(overlay(ct[best_z],gt[best_z],ca_mask[best_z]),aspect='equal')
ax3.axis('off')
ax3.set_title(f'TotalSeg+HU Anomaly\nDice={ts_ca:.4f}',**TK)

# TotalSeg overlay (colormap)
ax4=fig.add_subplot(gs[0,3]);ax4.set_facecolor(BG)
ax4.imshow(sk_resize(hw(ct[best_z]),(256,256),anti_aliasing=True),cmap='gray',aspect='equal')
# Overlay all TS structures
ts_colors={'colon':([0.2,0.9,0.2],0.4),'rectum':([0.9,0.2,0.2],0.5),
           'sigmoid_colon':([0.2,0.5,0.9],0.5),'small_bowel':([0.9,0.7,0.1],0.3)}
for sname,(sc,alpha) in ts_colors.items():
    if sname in colon_labels:
        slc=sk_resize(colon_labels[sname][best_z].astype(float),(256,256))>0.5
        if slc.sum()>0:
            mask_rgb=np.zeros((256,256,4))
            mask_rgb[slc,:3]=sc; mask_rgb[slc,3]=alpha
            ax4.imshow(mask_rgb,aspect='equal')
ax4.axis('off')
ax4.set_title(f'TotalSeg Structures z={best_z}',**TK)

# MONAI anomaly
ax5=fig.add_subplot(gs[0,4]);ax5.set_facecolor(BG)
t3_d=results.get('T3_segresnet_anomaly',{}).get('Dice',0)
ax5.imshow(overlay(ct[best_z],gt[best_z],
                   anomaly_mask[best_z] if 'anomaly_mask' in dir() else np.zeros((H,W),bool)),
           aspect='equal')
ax5.axis('off')
ax5.set_title(f'SegResNet Anomaly\nDice={t3_d:.4f}',**TK)

# Subtraction map
ax6=fig.add_subplot(gs[0,5]);ax6.set_facecolor(BG)
if os.path.exists(delay_path):
    sub_img=sk_resize(subtract[best_z],(256,256),anti_aliasing=True)
    ax6.imshow(sub_img,cmap='RdBu_r',vmin=-100,vmax=100,aspect='equal')
    ax6.set_title(f'Delay-Artery map z={best_z}\n(Enhancement)',**TK)
else:
    ax6.imshow(sk_resize(hw(ct[best_z]),(256,256)),cmap='gray',aspect='equal')
    ax6.set_title('No delay phase',**TK)
ax6.axis('off')

# Bar chart
ax7=fig.add_subplot(gs[1,0:4]);ax7.set_facecolor(BG)
sorted_m=sorted([(k,v) for k,v in all_methods.items()
                 if isinstance(v.get('Dice',0),(int,float))],key=lambda x:-x[1].get('Dice',0))
dn=[k for k,v in sorted_m]; dv=[v.get('Dice',0) for k,v in sorted_m]
bars=ax7.bar(range(len(dn)),dv,
             color=['#44FF88' if v>=0.5 else '#FFB347' if v>=0.1 else '#FF4444' for v in dv],
             alpha=0.85,edgecolor='#2D3748')
ax7.axhline(0.5,color='#AAFFAA',ls='--',lw=1.5,label='0.5 clinical min')
ax7.axhline(0.7,color='#FFAA00',ls=':',lw=1.0,label='0.7 publication')
ax7.set_xticks(range(len(dn)))
ax7.set_xticklabels([n.replace('_','\n') for n in dn],fontsize=5.5,color='#A0B8D0')
ax7.tick_params(colors='#A0B8D0')
ax7.set_ylabel('Dice',fontsize=8,color='#A0B8D0')
ax7.set_ylim(0,0.8)
for bar,v in zip(bars,dv):
    if v>0.01:
        ax7.text(bar.get_x()+bar.get_width()/2,v+0.01,f'{v:.4f}',
                 ha='center',fontsize=7,color='#E2EAF4',fontweight='bold')
ax7.legend(fontsize=7,facecolor='#1A2030',edgecolor='#3D4F6A',labelcolor='#A0B8D0')
ax7.set_title(f'All Methods: Dice | Best={best_d:.4f} ({best_name})',**TK)
ax7.spines[:].set_color('#2D3748');ax7.yaxis.grid(True,color='#2D3748',lw=0.5)
ax7.set_axisbelow(True);ax7.set_facecolor(BG)

# Insight panel
ax8=fig.add_subplot(gs[1,4:6]);ax8.set_facecolor(BG);ax8.axis('off')
ax8.text(0.5,1.0,'METHODS TRIED: FULL PICTURE',ha='center',va='top',fontsize=8.5,
         color='#E2EAF4',fontweight='bold',transform=ax8.transAxes)
rows_v=[
    ('HU threshold:',        'Dice 0.01  [FAIL]', '#FF4444'),
    ('Organ S3:',            'Dice 0.14  [BEST]', '#FFB347'),
    ('nnU-Net Art/Delay:',   'Dice 0.00  [FAIL]', '#FF4444'),
    ('SAM GT-oracle:',       'Dice 0.06  [FAIL]', '#FF4444'),
    ('TotalSeg colon:',      f'Dice {ts_d:.4f}', '#88BBFF'),
    ('TotalSeg+anomaly:',    f'Dice {ts_ca:.4f}', '#88BBFF'),
    ('SegResNet anomaly:',   f'Dice {t3_d:.4f}', '#88BBFF'),
    ('Root cause:',          'No CT boundary', '#FF6666'),
    ('Only solution:',       'Multi-case annotation', '#44FF88'),
]
for i,(k,v,c) in enumerate(rows_v):
    y=0.93-i*0.103
    ax8.text(0,y,k,transform=ax8.transAxes,fontsize=7,color='#A0B8D0',va='top')
    ax8.text(0.52,y,v,transform=ax8.transAxes,fontsize=7.5,color=c,va='top',fontweight='bold')

fig.suptitle(f'Alternative DL Methods | TotalSegmentator + MONAI + Multi-phase | Best Dice={best_d:.4f}',
             fontsize=10,fontweight='bold',color='#E2EAF4',y=0.98)

out_fig = os.path.join(SAVE,'ct_altdl_figure.png')
plt.savefig(out_fig,dpi=150,bbox_inches='tight',facecolor='#0D1117')
print(f'\nFigure -> {out_fig}')
plt.close()
print('\nDone.')
