"""
SegVol Direct Inference - bypass demo_data_process
===================================================
Loads the trained model, then manually preprocesses CT → tensor
and calls segvol.module.forward_decoder() directly.
Avoids all MONAI transform API compatibility issues.
"""
import os, sys, json, warnings
import numpy as np
import nibabel as nib
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import ndimage
from skimage.transform import resize as sk_resize

warnings.filterwarnings('ignore')
sys.path.insert(0, r'F:\ADDS\SegVol')

DATA1 = r'F:\ADDS\CTdata1'
SAVE  = r'F:\ADDS'
CKPT  = r'F:\ADDS\segvol_ckpt\pytorch_model.bin'

print('='*68)
print('SegVol Direct Inference (bypass data pipeline)')
print('='*68)

# Load CT and GT
art   = nib.load(os.path.join(DATA1,'nifti','inha_ct_arterial.nii.gz'))
tmask = nib.load(os.path.join(DATA1,'tumor_masks','tumor_mask_binary.nii.gz'))
ct    = art.get_fdata().astype(np.float32)
gt    = tmask.get_fdata().astype(np.uint8)
zooms = np.array(art.header.get_zooms(), dtype=float)
vox_vol = float(np.prod(zooms))
Z, H, W = ct.shape

def metrics(pred, truth):
    p=pred.astype(bool); t=truth.astype(bool)
    tp=int((p&t).sum()); fp=int((p&~t).sum()); fn=int((~p&t).sum())
    d=2*tp/(2*tp+fp+fn+1e-9); iou=tp/(tp+fp+fn+1e-9)
    prec=tp/(tp+fp+1e-9); rec=tp/(tp+fn+1e-9)
    return {'Dice':round(d,4),'IoU':round(iou,4),'Prec':round(prec,4),
            'Rec':round(rec,4),'TP':tp,'FP':fp,'FN':fn,
            'vol_cm3':round(float(pred.sum()*vox_vol/1000),2)}

# ----------------------------------------------------------------
# Manual CT preprocessing that SegVol expects
# ----------------------------------------------------------------
SPATIAL_SIZE = (32, 256, 256)
PATCH_SIZE   = (4, 16, 16)

def preprocess_ct_for_segvol(ct_vol, spatial_size):
    """Normalize CT and resize to SegVol spatial_size."""
    # Foreground normalization (SegVol default)
    flat = ct_vol.flatten()
    mean_val = np.mean(flat)
    fg = flat[flat > mean_val]
    upper = np.percentile(fg, 99.95)
    lower = np.percentile(fg, 0.05)
    mean_fg = float(np.mean(fg))
    std_fg  = float(np.std(fg))
    ct_n = np.clip(ct_vol, lower, upper)
    ct_n = (ct_n - mean_fg) / max(std_fg, 1e-8)
    # MinMax
    ct_n = ct_n - ct_n.min()
    ct_n = ct_n / max(ct_n.max(), 1e-8)
    # Resize to zoom-out size
    ct_small = sk_resize(ct_n, spatial_size, anti_aliasing=True, preserve_range=True).astype(np.float32)
    return ct_n, ct_small   # (full normalized, zoom-out)

print(f'GT tumor: {gt.sum()*vox_vol/1000:.1f} cm3  shape={ct.shape}')
print('Preprocessing CT...')
ct_norm, ct_small = preprocess_ct_for_segvol(ct, SPATIAL_SIZE)
# GT zoom-out
gt_float = gt.astype(np.float32)
gt_small = sk_resize(gt_float, SPATIAL_SIZE, order=0, anti_aliasing=False, preserve_range=True) > 0.5
print(f'CT zoom-out: {ct_small.shape}  GT zoom-out positives: {gt_small.sum()}')

# Convert to tensors
ct_tensor_small = torch.tensor(ct_small[None, None]).float()   # 1,1,D,H,W
gt_tensor_small = torch.tensor(gt_small[None, None]).float()
ct_tensor_full  = torch.tensor(ct_norm[None, None]).float()
gt_tensor_full  = torch.tensor(gt_float[None, None]).float()

# ----------------------------------------------------------------
# Load SegVol model
# ----------------------------------------------------------------
print('\nLoading SegVol model...')
from segment_anything_volumetric import sam_model_registry
from network.model import SegVol as SegVolModel

class Args:
    test_mode = True
    spatial_size = SPATIAL_SIZE
    patch_size   = PATCH_SIZE
    clip_ckpt    = r'F:\ADDS\segvol_ckpt'

args = Args()
gpu  = 0
torch.cuda.set_device(gpu)

sam_model = sam_model_registry['vit'](args=args)
segvol_model = SegVolModel(
    image_encoder  = sam_model.image_encoder,
    mask_decoder   = sam_model.mask_decoder,
    prompt_encoder = sam_model.prompt_encoder,
    clip_ckpt      = args.clip_ckpt,
    roi_size       = args.spatial_size,
    patch_size     = args.patch_size,
    test_mode      = args.test_mode,
).cuda()
segvol_model = torch.nn.DataParallel(segvol_model, device_ids=[gpu])

ckpt_data = torch.load(CKPT, map_location=f'cuda:{gpu}')
if 'model' in ckpt_data:
    segvol_model.load_state_dict(ckpt_data['model'], strict=True)
    print(f'Loaded (epoch {ckpt_data.get("epoch","?")})')
else:
    segvol_model.load_state_dict(ckpt_data, strict=False)
    print('Loaded (raw state dict)')
segvol_model.eval()
print('SegVol ready.')

from utils.monai_inferers_utils import sliding_window_inference, generate_box, logits2roi_coor, build_binary_cube

# ----------------------------------------------------------------
# Inference helpers
# ----------------------------------------------------------------
def make_gt_bbox(gt_small_tensor):
    """Generate 3D bbox from GT zoom-out label."""
    lab = gt_small_tensor[0,0]  # D x H x W tensor
    return generate_box(lab).unsqueeze(0).float().cuda()

def run_segvol(ct_small_t, text_prompt, box):
    """Run zoom-out forward pass."""
    with torch.no_grad():
        logits = segvol_model(
            ct_small_t.cuda(),
            text  = text_prompt,
            boxes = box,
            points= None,
        )
    return logits.cpu()      # 1,1,D,H,W

# ----------------------------------------------------------------
# A) GT bbox + 5 text prompts
# ----------------------------------------------------------------
print('\n[A] GT bbox + text prompt combinations:')
box_gt = make_gt_bbox(gt_tensor_small)
results = {}

text_prompts = [
    'colon cancer',
    'rectal cancer',
    'colon tumor',
    'mucinous colorectal cancer',
    'colorectal carcinoma',
]

for text in text_prompts:
    try:
        logits = run_segvol(ct_tensor_small, text, box_gt)  # 1,1,D,H,W
        logits_g = logits[0,0]   # D,H,W

        zoom_out_dice_fn = lambda l: 2*(torch.sigmoid(l)>0.5).float().sum() / \
            (((torch.sigmoid(l)>0.5).float().sum()) + gt_tensor_small[0,0].sum() + 1e-9)

        # Zoom-in
        min_d,min_h,min_w,max_d,max_h,max_w = logits2roi_coor(SPATIAL_SIZE, logits_g)
        if min_d is not None:
            # Crop full-res CT at bbox scaled to full size
            scale = [ct_norm.shape[i] / SPATIAL_SIZE[i] for i in range(3)]
            fd = max(0,int(min_d*scale[0])); td = min(Z-1,int((max_d+1)*scale[0]))
            fh = max(0,int(min_h*scale[1])); th = min(H-1,int((max_h+1)*scale[1]))
            fw = max(0,int(min_w*scale[2])); tw = min(W-1,int((max_w+1)*scale[2]))
            ct_crop = torch.tensor(ct_norm[fd:td, fh:th, fw:tw][None,None]).float()
            ct_crop_resized = F.interpolate(ct_crop, size=SPATIAL_SIZE, mode='trilinear', align_corners=False)

            # Binary cube for prompt reflection
            bc = build_binary_cube(box_gt, (1,1)+tuple(s for s in SPATIAL_SIZE))
            bc_crop = bc[0,0,min_d:max_d+1,min_h:max_h+1,min_w:max_w+1]
            global_preds = (torch.sigmoid(logits_g[min_d:max_d+1,min_h:max_h+1,min_w:max_w+1])>0.5).long()
            bc_r = F.interpolate(bc_crop.unsqueeze(0).unsqueeze(0).float(), size=SPATIAL_SIZE, mode='nearest')
            gp_r = F.interpolate(global_preds.unsqueeze(0).unsqueeze(0).float(), size=SPATIAL_SIZE, mode='nearest').long()
            prompt_refl = (bc_r, gp_r)

            with torch.no_grad():
                logits_crop = sliding_window_inference(
                    ct_crop_resized.cuda(), prompt_refl,
                    SPATIAL_SIZE, 1, segvol_model, 0.5,
                    text=text, use_box=True, use_point=False)
            # Combine
            logits_crop_np = logits_crop.cpu().squeeze()
            logits_full = F.interpolate(logits_crop_np.unsqueeze(0).unsqueeze(0),
                                         size=(td-fd, th-fh, tw-fw), mode='trilinear', align_corners=False)[0,0]
            # Upsample to full CT size
            pred_full = np.zeros(ct_norm.shape, dtype=np.float32)
            # Use zoom-out logits as fallback
            logits_g_up = F.interpolate(logits_g.unsqueeze(0).unsqueeze(0),
                                         size=ct_norm.shape, mode='trilinear', align_corners=False)[0,0]
            logits_g_up = logits_g_up.numpy()
            pred_final = (torch.sigmoid(torch.tensor(logits_g_up)) > 0.5).numpy()
        else:
            # No foreground detected → upscale zoom-out
            logits_g_up = F.interpolate(logits_g.unsqueeze(0).unsqueeze(0),
                                         size=ct_norm.shape, mode='trilinear', align_corners=False)[0,0]
            pred_final = (torch.sigmoid(logits_g_up) > 0.5).numpy()

        m = metrics(pred_final, gt)
        results[f'SegVol_{text.replace(" ","_")}'] = m
        print(f'  "{text}": Dice={m["Dice"]:.4f}  Prec={m["Prec"]:.4f}  Rec={m["Rec"]:.4f}  vol={m["vol_cm3"]:.1f}cm3')

    except Exception as e:
        print(f'  "{text}": ERROR {e}')
        import traceback; traceback.print_exc()

# ----------------------------------------------------------------
# B) No text, only GT bbox (box-only prompt)
# ----------------------------------------------------------------
print('\n[B] Box-only (no text prompt):')
try:
    logits = run_segvol(ct_tensor_small, None, box_gt)
    logits_g = F.interpolate(logits[0], size=ct_norm.shape, mode='trilinear', align_corners=False)[0]
    pred_bbox_only = (torch.sigmoid(logits_g) > 0.5).numpy()
    m_b = metrics(pred_bbox_only, gt)
    results['SegVol_bbox_only'] = m_b
    print(f'  Dice={m_b["Dice"]:.4f}  Prec={m_b["Prec"]:.4f}  Rec={m_b["Rec"]:.4f}')
except Exception as e:
    print(f'  ERROR: {e}')

# ----------------------------------------------------------------
# SUMMARY
# ----------------------------------------------------------------
baseline = {'S3_organ':0.1389,'SAM_GTbbox':0.0649,'nnUNet':0.0000,'YOLO':0.0124}
all_m = {k:{'Dice':v} for k,v in baseline.items()}
all_m.update(results)

print()
print('='*68)
print('FULL SCORECARD:')
for name, m in sorted(all_m.items(), key=lambda x: -x[1].get('Dice',0)):
    d = m.get('Dice', 0)
    flag = 'GOOD' if d>=0.5 else 'MOD' if d>=0.3 else 'POOR' if d>=0.1 else 'FAIL'
    print(f'  {name:<38} Dice={d:.4f}  [{flag}]')

best_key = max(all_m, key=lambda k: all_m[k].get('Dice',0))
best_d   = all_m[best_key].get('Dice',0)
print(f'\nBEST: {best_key}  Dice={best_d:.4f}')

# Save
safe = {k:{kk:float(vv) if isinstance(vv,(int,float,np.floating)) else str(vv)
            for kk,vv in v.items()} for k,v in all_m.items()}
with open(os.path.join(SAVE,'ct_segvol_direct.json'),'w',encoding='utf-8') as f:
    json.dump(safe,f,indent=2)
print(f'JSON -> {os.path.join(SAVE,"ct_segvol_direct.json")}')

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
    tp=p2&g2;fp=p2&~g2;fn=~p2&g2
    rgb[tp,0]=0.1;rgb[tp,1]=0.9;rgb[tp,2]=0.1
    rgb[fp,0]=0.9;rgb[fp,1]=0.1;rgb[fp,2]=0.1
    rgb[fn,0]=0.1;rgb[fn,1]=0.1;rgb[fn,2]=0.9
    return rgb

best_z = max(range(Z), key=lambda z: int(gt[z].sum()))

fig, axes = plt.subplots(1, 6, figsize=(28,5), facecolor='#0D1117')
BG='#161B22'; TK=dict(fontsize=7.5,color='#8BAFD4',fontweight='bold',pad=3)
gt2d=sk_resize(gt[best_z].astype(float),(256,256))>0.5
ctn=sk_resize(hw(ct[best_z]),(256,256),anti_aliasing=True)
rgb=np.stack([ctn]*3,-1);rgb[gt2d,0]=0.1;rgb[gt2d,1]=0.9;rgb[gt2d,2]=0.1
axes[0].imshow(rgb,aspect='equal');axes[0].axis('off');axes[0].set_facecolor(BG)
axes[0].set_title(f'GT z={best_z}\n{gt.sum()*vox_vol/1000:.1f}cm3',**TK)

# Show 4 SegVol results
seg_keys = [k for k in results][:4]
for i, key in enumerate(seg_keys):
    ax = axes[i+1]; ax.set_facecolor(BG)
    ax.imshow(sk_resize(hw(ct[best_z]),(256,256),anti_aliasing=True),cmap='gray',aspect='equal')
    ax.axis('off')
    d = results[key].get('Dice',0)
    label = key.replace('SegVol_','').replace('_',' ')
    ax.set_title(f'SegVol: "{label[:18]}"\nDice={d:.4f}',**TK)

# Bar chart
ax_bar = axes[5]; ax_bar.set_facecolor(BG)
dn=sorted(all_m, key=lambda k:-all_m[k].get('Dice',0))
dv=[all_m[k].get('Dice',0) for k in dn]
bars=ax_bar.barh(range(len(dn)),dv,
                 color=['#44FF88' if v>=0.5 else '#FFB347' if v>=0.1 else '#FF4444' for v in dv],
                 alpha=0.85,edgecolor='#2D3748')
ax_bar.set_yticks(range(len(dn)))
ax_bar.set_yticklabels([n.replace('_',' ')[:20] for n in dn],fontsize=6,color='#A0B8D0')
ax_bar.axvline(0.5,color='#AAFFAA',ls='--',lw=1.2); ax_bar.set_xlim(0,0.8)
ax_bar.tick_params(colors='#A0B8D0')
ax_bar.set_xlabel('Dice',fontsize=7,color='#A0B8D0')
ax_bar.set_title(f'Best={best_d:.4f}',**TK)
ax_bar.set_facecolor(BG); ax_bar.spines[:].set_color('#2D3748')

fig.suptitle(f'SegVol Zero-Shot | 5 text+box prompts | Best Dice={best_d:.4f}',
             fontsize=9,fontweight='bold',color='#E2EAF4')
fig.tight_layout()
out_fig = os.path.join(SAVE,'ct_segvol_direct.png')
plt.savefig(out_fig,dpi=150,bbox_inches='tight',facecolor='#0D1117')
print(f'Figure -> {out_fig}')
plt.close()
print('\nDone.')
