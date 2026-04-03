"""
SegVol Zero-Shot CT Segmentation for Mucinous CRC
==================================================
Nature Communications 2024 - BAAI/SegVol
  Text prompt: "colon cancer" / "rectal mass" / "mucinous tumor"
  Box prompt: from GT (oracle) or TotalSeg colon bbox (auto)
  Zoom-in / Zoom-out strategy built in

Requirements:
  F:\ADDS\segvol_ckpt\pytorch_model.bin
  F:\ADDS\SegVol\ (source)
"""
import os, sys, json, warnings, shutil
import numpy as np
import nibabel as nib
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import ndimage
from skimage.transform import resize as sk_resize

warnings.filterwarnings('ignore')
sys.path.insert(0, r'F:\ADDS\SegVol')

DATA1    = r'F:\ADDS\CTdata1'
SAVE     = r'F:\ADDS'
CKPT     = r'F:\ADDS\segvol_ckpt\pytorch_model.bin'
CLIP_DIR = r'F:\ADDS\segvol_ckpt'   # config.json, tokenizer etc.

print('='*68)
print('SegVol Zero-Shot Segmentation | Nature Commun. 2024 | BAAI')
print('Text + Box prompts: colon cancer, rectal mass')
print('='*68)

# Check checkpoint
if not os.path.exists(CKPT):
    print(f'ERROR: Checkpoint not found: {CKPT}')
    print('Downloading from HuggingFace...')
    from huggingface_hub import hf_hub_download
    os.makedirs(os.path.dirname(CKPT), exist_ok=True)
    hf_hub_download(repo_id='BAAI/SegVol', filename='pytorch_model.bin',
                    local_dir=os.path.dirname(CKPT))
else:
    sz = os.path.getsize(CKPT)
    print(f'Checkpoint OK: {sz/1e6:.0f} MB')

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
# Load SegVol model
# ----------------------------------------------------------------
print('\n[Loading SegVol model...]')
try:
    from segment_anything_volumetric import sam_model_registry
    from network.model import SegVol as SegVolModel

    class Args:
        test_mode = True
        resume = CKPT
        infer_overlap = 0.5
        spatial_size = (32, 256, 256)
        patch_size = (4, 16, 16)
        use_zoom_in = True
        use_text_prompt = True
        use_box_prompt = True
        use_point_prompt = False
        clip_ckpt = CLIP_DIR
        work_dir = SAVE

    args = Args()
    gpu  = 0
    torch.cuda.set_device(gpu)

    sam_model = sam_model_registry['vit'](args=args)
    segvol_model = SegVolModel(
        image_encoder=sam_model.image_encoder,
        mask_decoder=sam_model.mask_decoder,
        prompt_encoder=sam_model.prompt_encoder,
        clip_ckpt=args.clip_ckpt,
        roi_size=args.spatial_size,
        patch_size=args.patch_size,
        test_mode=args.test_mode,
    ).cuda()
    segvol_model = torch.nn.DataParallel(segvol_model, device_ids=[gpu])

    ckpt_data = torch.load(CKPT, map_location=f'cuda:{gpu}')
    if 'model' in ckpt_data:
        segvol_model.load_state_dict(ckpt_data['model'], strict=True)
        epoch = ckpt_data.get('epoch', '?')
        print(f'Loaded checkpoint (epoch {epoch})')
    else:
        segvol_model.load_state_dict(ckpt_data, strict=False)
        print('Loaded checkpoint (raw state dict)')

    segvol_model.eval()
    print('SegVol loaded successfully!')

except Exception as e:
    print(f'SegVol load failed: {e}')
    import traceback; traceback.print_exc()
    sys.exit(1)

# ----------------------------------------------------------------
# Preprocess CT for SegVol
# ----------------------------------------------------------------
print('\n[Preprocessing CT...]')
from data_process.demo_data_process import process_ct_gt

# SegVol expects: NIfTI input, normalized to [-1,1] range approx
# Prepare temp NIfTI with GT for SegVol format
art_path   = os.path.join(DATA1,'nifti','inha_ct_arterial.nii.gz')
gt_path    = os.path.join(DATA1,'tumor_masks','tumor_mask_binary.nii.gz')

# SegVol categories: text prompts to try
text_prompts = [
    'colon cancer',
    'rectal cancer',
    'colon tumor',
    'mucinous tumor',
    'colorectal carcinoma',
]

# ----------------------------------------------------------------
# GT-guided box prompt (oracle)
# ----------------------------------------------------------------
print('\n[A] Oracle test: GT bbox + text prompt combinations')

from utils.monai_inferers_utils import (
    sliding_window_inference, generate_box, select_points,
    build_binary_cube, build_binary_points, logits2roi_coor
)

results = {}

for text_prompt in text_prompts:
    print(f'\n  Prompt: "{text_prompt}"')
    try:
        data_item = process_ct_gt(art_path, gt_path, [text_prompt], args.spatial_size)
        image     = data_item['image'].float()
        gt3D      = data_item['label']
        img_zoom  = data_item['zoom_out_image'].float()
        gt_zoom   = data_item['zoom_out_label']

        image_single_resize = img_zoom
        image_single        = image[0, 0]
        ori_shape           = image_single.shape

        label_single        = gt3D[0][0]
        label_single_resize = gt_zoom[0][0]

        if torch.sum(label_single) == 0:
            print('    No GT voxels -- skipping')
            continue

        box_single = generate_box(label_single_resize).unsqueeze(0).float().cuda()
        binary_cube_resize = build_binary_cube(box_single, label_single_resize.shape)

        with torch.no_grad():
            logits_global = segvol_model(
                image_single_resize.cuda(),
                text=text_prompt,
                boxes=box_single,
                points=None,
            )

        logits_global = F.interpolate(
            logits_global.cpu(), size=ori_shape, mode='nearest')[0][0]

        # Dice zoom-out
        pred_zo = (torch.sigmoid(logits_global) > 0.5).numpy()
        gt_zo   = label_single.numpy().astype(bool)
        tp = int((pred_zo & gt_zo).sum())
        dice_zo = 2*tp / (pred_zo.sum() + gt_zo.sum() + 1e-9)
        print(f'    Zoom-out Dice: {dice_zo:.4f}')

        # Zoom-in
        min_d,min_h,min_w,max_d,max_h,max_w = logits2roi_coor(args.spatial_size, logits_global)
        if min_d is not None:
            img_crop = image_single[min_d:max_d+1, min_h:max_h+1, min_w:max_w+1].unsqueeze(0).unsqueeze(0)
            global_preds = (torch.sigmoid(logits_global[min_d:max_d+1, min_h:max_h+1, min_w:max_w+1]) > 0.5).long()
            binary_cube  = F.interpolate(binary_cube_resize.unsqueeze(0).unsqueeze(0).float(),
                                          size=ori_shape, mode='nearest')[0][0]
            bc_crop = binary_cube[min_d:max_d+1, min_h:max_h+1, min_w:max_w+1]
            prompt_refl = (bc_crop.unsqueeze(0).unsqueeze(0), global_preds.unsqueeze(0).unsqueeze(0))

            with torch.no_grad():
                logits_crop = sliding_window_inference(
                    img_crop.cuda(), prompt_refl,
                    args.spatial_size, 1, segvol_model, args.infer_overlap,
                    text=text_prompt, use_box=True, use_point=False)
            logits_global[min_d:max_d+1, min_h:max_h+1, min_w:max_w+1] = logits_crop.cpu().squeeze()

        # Final prediction
        pred_final = (torch.sigmoid(logits_global) > 0.5).numpy()
        # Resize back to original if needed
        if pred_final.shape != gt.shape:
            pred_final = sk_resize(pred_final.astype(float), gt.shape, order=0,
                                   anti_aliasing=False, preserve_range=True) > 0.5

        m = metrics(pred_final, gt)
        results[f'SegVol_{text_prompt.replace(" ","_")}'] = m
        print(f'    Final Dice={m["Dice"]:.4f}  Prec={m["Prec"]:.4f}  Rec={m["Rec"]:.4f}  vol={m["vol_cm3"]:.1f}cm3')

    except Exception as e:
        print(f'    ERROR: {e}')
        import traceback; traceback.print_exc()

# ----------------------------------------------------------------
# Auto-bbox: using existing TotalSegmentator colon output (if any)
# ----------------------------------------------------------------
print('\n[B] Auto-bbox from TotalSegmentator colon mask:')
ts_colon = os.path.join(SAVE,'totalseg_output','colon.nii.gz')
if os.path.exists(ts_colon):
    ts_nii = nib.load(ts_colon)
    ts_vol = ts_nii.get_fdata()
    if ts_vol.shape != gt.shape:
        ts_vol = sk_resize(ts_vol, gt.shape, order=0, anti_aliasing=False, preserve_range=True)
    ts_mask = ts_vol > 0.5
    if ts_mask.sum() > 0:
        print(f'  TotalSeg colon: {ts_mask.sum()*vox_vol/1000:.1f}cm3')
        # Will use this as bbox in next step
    else:
        print('  TotalSeg colon empty mask')
else:
    print('  TotalSeg colon output not found -- skipping auto-bbox')

# ----------------------------------------------------------------
# SUMMARY
# ----------------------------------------------------------------
print()
print('='*68)
print('SEGVOL RESULTS:')
baseline = {
    'S3_organ_constrained': 0.1389,
    'SAM_GTbbox':           0.0649,
    'nnUNet_Arterial':      0.0000,
}
all_m = {k: {'Dice': v} for k,v in baseline.items()}
all_m.update(results)

for name, m in sorted(all_m.items(), key=lambda x: -x[1].get('Dice',0)):
    d = m.get('Dice', 0)
    status = 'GOOD' if d>=0.5 else 'MODERATE' if d>=0.3 else 'POOR' if d>=0.1 else 'FAIL'
    print(f'  {name:<40} {d:.4f}  [{status}]')

best_key = max(all_m, key=lambda k: all_m[k].get('Dice',0))
best_d   = all_m[best_key].get('Dice',0)
print(f'\nBEST: {best_key}  Dice={best_d:.4f}')

# Save JSON
out_json = os.path.join(SAVE, 'ct_segvol_results.json')
safe = {k:{kk: float(vv) if isinstance(vv,(int,float,np.floating)) else str(vv)
            for kk,vv in v.items()} for k,v in all_m.items()}
with open(out_json,'w',encoding='utf-8') as f:
    json.dump(safe,f,indent=2)
print(f'JSON -> {out_json}')

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

best_z = max(range(Z), key=lambda z: int(gt[z].sum()))

segvol_keys = [k for k in results if 'SegVol' in k]
n_seg = max(len(segvol_keys), 1)
ncols = min(n_seg + 1, 6)

fig, axes = plt.subplots(1, ncols, figsize=(ncols*4.5, 5), facecolor='#0D1117')
axes = [axes] if ncols==1 else list(axes)
BG = '#161B22'; TK = dict(fontsize=7.5, color='#8BAFD4', fontweight='bold', pad=3)

gt2d=sk_resize(gt[best_z].astype(float),(256,256))>0.5
ctn=sk_resize(hw(ct[best_z]),(256,256),anti_aliasing=True)
rgb=np.stack([ctn]*3,-1); rgb[gt2d,0]=0.1; rgb[gt2d,1]=0.9; rgb[gt2d,2]=0.1
axes[0].imshow(rgb,aspect='equal'); axes[0].axis('off')
axes[0].set_facecolor(BG)
axes[0].set_title(f'GT z={best_z}\n{gt.sum()*vox_vol/1000:.1f}cm3',**TK)

for i, k in enumerate(segvol_keys[:ncols-1]):
    ax = axes[i+1]
    ax.set_facecolor(BG)
    m  = results[k]
    # find best pred mask from latest iteration - store if possible
    ax.imshow(sk_resize(hw(ct[best_z]),(256,256),anti_aliasing=True),cmap='gray',aspect='equal')
    ax.axis('off')
    prompt = k.replace('SegVol_','').replace('_',' ')
    ax.set_title(f'SegVol: "{prompt}"\nDice={m["Dice"]:.4f}',**TK)

# Bar chart (replace last panel if needed)
if ncols >= 4:
    fig2,ax2 = plt.subplots(figsize=(14,5),facecolor='#0D1117')
    ax2.set_facecolor('#161B22')
    dn=[k for k,v in sorted(all_m.items(),key=lambda x:-x[1].get('Dice',0))]
    dv=[all_m[k].get('Dice',0) for k in dn]
    bars=ax2.bar(range(len(dn)),dv,
                 color=['#44FF88' if v>=0.5 else '#FFB347' if v>=0.1 else '#FF4444' for v in dv],
                 alpha=0.85,edgecolor='#2D3748')
    ax2.axhline(0.5,color='#AAFFAA',ls='--',lw=1.5,label='clinical min')
    [ax2.text(b.get_x()+b.get_width()/2,v+0.005,f'{v:.4f}',
              ha='center',fontsize=7,color='#E2EAF4',fontweight='bold')
     for b,v in zip(bars,dv) if v>0.01]
    ax2.set_xticks(range(len(dn)))
    ax2.set_xticklabels([n.replace('_','\n') for n in dn],fontsize=6,color='#A0B8D0')
    ax2.set_ylabel('Dice',color='#A0B8D0'); ax2.set_ylim(0,0.8)
    ax2.legend(fontsize=7,facecolor='#1A2030',edgecolor='#3D4F6A',labelcolor='#A0B8D0')
    ax2.set_title(f'All Methods Dice | Best={best_d:.4f} ({best_key})',
                  fontsize=8,color='#8BAFD4',fontweight='bold')
    ax2.spines[:].set_color('#2D3748'); ax2.yaxis.grid(True,color='#2D3748',lw=0.5)
    ax2.set_axisbelow(True)
    fig2.tight_layout()
    out_fig2 = os.path.join(SAVE,'ct_segvol_comparison.png')
    plt.savefig(out_fig2,dpi=150,bbox_inches='tight',facecolor='#0D1117')
    print(f'Comparison figure -> {out_fig2}')
    plt.close(fig2)

fig.suptitle(f'SegVol Zero-Shot | text+box prompts | Best Dice={best_d:.4f}',
             fontsize=9,fontweight='bold',color='#E2EAF4')
fig.tight_layout()
out_fig = os.path.join(SAVE,'ct_segvol_figure.png')
plt.savefig(out_fig,dpi=150,bbox_inches='tight',facecolor='#0D1117')
print(f'Figure -> {out_fig}')
plt.close()
print('\nDone.')
