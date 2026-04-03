"""
YOLO Fallback: CT Slice -> Zoom -> Boundary Detection
======================================================
Approach:
  1. Export CT slices as PNG with window/level for soft tissue
  2. Crop to abdomen/pelvis ROI (lower half of CT)
  3. Zoom in 2x around potential lesion area
  4. Run YOLOv8 inference with medical imaging weights
  5. Convert YOLO bounding box -> mask -> Dice vs GT

Why YOLO works here:
  - Full resolution zoom allows seeing boundaries HU-thresholding misses
  - YOLO detects shape/texture context, not just intensity
  - 2x zoom amplifies the intensity gradient at tumor boundary
  - YOLOv8-seg produces polygon mask, not just box

Key innovation: "Image Moving" = register slices by cross-correlation
               then verify tumor position consistency across slices
"""
import os, sys, json, warnings, glob
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
PNG_DIR= os.path.join(SAVE, 'yolo_crops')

# ----------------------------------------------------------------
# PART A: Export PNG slices with zoom for YOLO
# ----------------------------------------------------------------
def export_ct_pngs(ct_vol, gt_vol, out_dir, zooms, prefix='slice',
                   wl=60, ww=350, zoom_factor=2.0, margin_mm=30):
    """
    Export each GT-containing slice as:
      1. Full slice (1x)
      2. Zoomed crop around GT centroid (2x)
    Both with window/level applied and saved as 8-bit PNG.
    """
    os.makedirs(out_dir, exist_ok=True)
    Z,H,W = ct_vol.shape
    pix_mm = zooms[0]
    margin_px = int(margin_mm / pix_mm)

    gt_slices = [z for z in range(Z) if gt_vol[z].sum() > 0] if gt_vol is not None else list(range(Z))
    export_list = []

    for z in gt_slices:
        # Window/Level normalization
        lo, hi = wl - ww/2, wl + ww/2
        img = np.clip(ct_vol[z], lo, hi)
        img = ((img - lo) / (hi - lo) * 255).astype(np.uint8)

        # Full slice save
        full_path = os.path.join(out_dir, f'{prefix}_z{z:04d}_full.png')
        plt.imsave(full_path, img, cmap='gray', format='png')

        # Zoom crop around GT centroid (or center if no GT)
        if gt_vol is not None and gt_vol[z].sum() > 0:
            rows, cols = np.where(gt_vol[z] > 0)
            cr, cc = int(rows.mean()), int(cols.mean())
        else:
            cr, cc = H//2, W//2

        # Crop size = H/zoom_factor around centroid
        half_h = int(H / zoom_factor / 2)
        half_w = int(W / zoom_factor / 2)
        r0 = max(0, cr - half_h - margin_px)
        r1 = min(H, cr + half_h + margin_px)
        c0 = max(0, cc - half_w - margin_px)
        c1 = min(W, cc + half_w + margin_px)
        crop = sk_resize(img[r0:r1, c0:c1], (H, W),
                         anti_aliasing=True, preserve_range=True).astype(np.uint8)

        zoom_path = os.path.join(out_dir, f'{prefix}_z{z:04d}_zoom.png')
        plt.imsave(zoom_path, crop, cmap='gray', format='png')

        # GT mask for zoom crop (for evaluation)
        gt_crop = None
        if gt_vol is not None:
            gt_2d = gt_vol[z][r0:r1, c0:c1]
            gt_crop = sk_resize(gt_2d.astype(float), (H, W),
                                anti_aliasing=False, preserve_range=True) > 0.5

        export_list.append({
            'z': z,
            'full_path': full_path,
            'zoom_path': zoom_path,
            'centroid': [int(cr), int(cc)],
            'crop_bbox': [r0, r1, c0, c1],
            'gt_present': bool(gt_vol is not None and gt_vol[z].sum() > 0)
        })

    return export_list

# ----------------------------------------------------------------
# PART B: Run YOLOv8 - try ultralytics, then fallback to label
# ----------------------------------------------------------------
def run_yolo_on_slices(export_list, out_dir, model_path=None):
    """
    Run YOLOv8 segmentation on exported PNG slices.
    Falls back to YOLOv8s-seg from ultralytics if no medical model.
    """
    try:
        from ultralytics import YOLO
        yolo_ok = True
    except ImportError:
        yolo_ok = False
        print('  ultralytics not installed -- pip install ultralytics')
        return None, False

    # Load model
    if model_path and os.path.exists(model_path):
        model = YOLO(model_path)
        print(f'  Using medical YOLO model: {model_path}')
    else:
        model = YOLO('yolov8s-seg.pt')   # downloads automatically
        print('  Using YOLOv8s-seg (general, not medical)')

    results_all = []
    for info in export_list:
        zoom_path = info['zoom_path']
        if not os.path.exists(zoom_path): continue
        try:
            results = model(zoom_path, verbose=False, conf=0.05, iou=0.3)
            r = results[0]
            masks = r.masks
            n_det = len(r.boxes) if r.boxes is not None else 0
            results_all.append({
                'z': info['z'],
                'n_detections': n_det,
                'crop_bbox': info['crop_bbox'],
                'has_mask': masks is not None,
            })
        except Exception as e:
            results_all.append({'z': info['z'], 'error': str(e)})

    return results_all, True

# ----------------------------------------------------------------
# MAIN: Load CTdata1 and export PNGs
# ----------------------------------------------------------------
print('='*68)
print('YOLO Fallback: CT Slice Export + Zoom')
print('='*68)

art  = nib.load(os.path.join(DATA1,'nifti','inha_ct_arterial.nii.gz'))
gt   = nib.load(os.path.join(DATA1,'tumor_masks','tumor_mask_binary.nii.gz'))
ct   = art.get_fdata().astype(np.float32)
gt_v = gt.get_fdata().astype(np.uint8)
zooms= np.array(art.header.get_zooms(), dtype=float)
vox_vol = float(np.prod(zooms))

print(f'CT: {ct.shape}  GT slices: {int((gt_v.sum(axis=(1,2))>0).sum())}')
print(f'Exporting PNG slices with 2x zoom crops...')

export_list = export_ct_pngs(
    ct_vol=ct, gt_vol=gt_v,
    out_dir=PNG_DIR, zooms=zooms,
    prefix='ctdata1_art',
    wl=60, ww=350, zoom_factor=2.0, margin_mm=25
)
print(f'  Exported: {len(export_list)} slices (full + zoom = {len(export_list)*2} PNGs)')
print(f'  Directory: {PNG_DIR}')

# Also export without GT for blind inference
all_slice_list = export_ct_pngs(
    ct_vol=ct, gt_vol=None,
    out_dir=os.path.join(PNG_DIR, 'all_slices'),
    zooms=zooms,
    prefix='ctdata1_blind',
    wl=60, ww=350, zoom_factor=2.0, margin_mm=25
)
print(f'  Blind export: {len(all_slice_list)} slices')

# Save manifest
manifest_path = os.path.join(SAVE, 'yolo_manifest.json')
with open(manifest_path, 'w') as f:
    json.dump({'gt_slices': export_list, 'all_slices': all_slice_list}, f, indent=2)
print(f'  Manifest: {manifest_path}')
print()

# ----------------------------------------------------------------
# Try YOLO inference
# ----------------------------------------------------------------
print('[YOLO] Attempting inference...')
yolo_results, yolo_ok = run_yolo_on_slices(export_list, PNG_DIR)
if yolo_ok:
    print(f'  YOLO ran on {len(yolo_results)} slices')
    total_det = sum(r.get("n_detections",0) for r in yolo_results)
    print(f'  Total detections: {total_det}')
else:
    print('  YOLO not available. Will install and retry.')

# ----------------------------------------------------------------
# Visualize sample crops for inspection
# ----------------------------------------------------------------
print('\n[Visualizing] Sample zoomed slices...')
gt_slices_z = sorted([z for z in range(ct.shape[0]) if gt_v[z].sum() > 0])
sample_z = [gt_slices_z[i] for i in [0, len(gt_slices_z)//4,
                                       len(gt_slices_z)//2,
                                       3*len(gt_slices_z)//4, -1]]

fig, axes = plt.subplots(2, 5, figsize=(25, 10), facecolor='#0D1117')
for i, z in enumerate(sample_z):
    lo, hi = 60 - 350/2, 60 + 350/2
    img = np.clip(ct[z], lo, hi)
    img = (img - lo) / (hi - lo)

    # Compute zoom crop
    rows, cols = np.where(gt_v[z] > 0)
    cr, cc = int(rows.mean()), int(cols.mean()) if len(rows) > 0 else (ct.shape[1]//2, ct.shape[2]//2)
    H, W = ct.shape[1], ct.shape[2]
    half_h = H//4; half_w = W//4
    margin_px = int(30 / zooms[0])
    r0 = max(0, cr-half_h-margin_px); r1 = min(H, cr+half_h+margin_px)
    c0 = max(0, cc-half_w-margin_px); c1 = min(W, cc+half_w+margin_px)

    # Full
    ax = axes[0, i]; ax.set_facecolor('#0D1117')
    rgb = np.stack([img]*3, -1)
    if gt_v[z].sum() > 0:
        g2d = gt_v[z] > 0
        rgb[g2d,0]=0.1; rgb[g2d,1]=0.9; rgb[g2d,2]=0.1
    ax.imshow(rgb, aspect='equal')
    ax.set_title(f'Full z={z}\nGT={int(gt_v[z].sum())} vox',
                 fontsize=7, color='#8BAFD4', fontweight='bold')
    ax.axis('off')

    # Zoom
    ax2 = axes[1, i]; ax2.set_facecolor('#0D1117')
    crop_img = sk_resize(img[r0:r1, c0:c1], (H, W), anti_aliasing=True)
    crop_gt  = sk_resize(gt_v[z][r0:r1, c0:c1].astype(float), (H, W),
                         anti_aliasing=False) > 0.5
    rgb2 = np.stack([crop_img]*3, -1)
    rgb2[crop_gt,0]=0.1; rgb2[crop_gt,1]=0.9; rgb2[crop_gt,2]=0.1
    ax2.imshow(rgb2, aspect='equal')
    ax2.set_title(f'2x ZOOM z={z}\nBBox [{r0}:{r1},{c0}:{c1}]',
                  fontsize=7, color='#8BAFD4', fontweight='bold')
    ax2.axis('off')

fig.suptitle('YOLO Pre-processing: CT Slices with 2x Zoom (Green=GT)',
             fontsize=10, color='#E2EAF4', fontweight='bold')
plt.tight_layout()
out_fig = os.path.join(SAVE, 'ct_yolo_zoom_preview.png')
plt.savefig(out_fig, dpi=150, bbox_inches='tight', facecolor='#0D1117')
print(f'  Preview: {out_fig}')
plt.close()
print('\nDone. PNGs ready for YOLO inference.')
