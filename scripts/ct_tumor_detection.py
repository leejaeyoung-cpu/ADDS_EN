"""
CT Tumor Detection v3 - Feature Extraction + 3D Overlay
=========================================================
Method 1: HU Anomaly-based detection with body interior masking
Method 2: nnU-Net inference (CPU fallback)
Feature extraction: Shape, HU distribution, texture (GLCM-like)
Body = natural skin/gray tones, Tumors = bright GREEN glow
"""

import os, sys, io, json, base64, gzip, time
import numpy as np

if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

import pydicom
from scipy import ndimage

DCM_DIR = r"f:\ADDS\CTdata\CTdcm"
OUTPUT_DIR = r"f:\ADDS\CTdata\ct_tumor_detection"
VOL_SIZE = 128

# nnU-Net paths
NNUNET_RESULTS = r"f:\ADDS\nnUNet_results"
DATASET = "Dataset011_ColonMasked"
TRAINER = "nnUNetTrainer__nnUNetPlans__3d_fullres"


def load_volume():
    """Load Series #604 as HU volume."""
    print("=" * 60)
    print("  CT TUMOR DETECTION v3")
    print("=" * 60)
    print("\n[1/7] Loading DICOM Series #604...")

    files = [f for f in os.listdir(DCM_DIR) if f.endswith('.dcm')]
    slices_map = {}
    for fname in files:
        try:
            ds = pydicom.dcmread(os.path.join(DCM_DIR, fname), stop_before_pixels=True)
            if int(getattr(ds, 'SeriesNumber', 0)) == 604:
                z = float(ds.ImagePositionPatient[2])
                if z not in slices_map:
                    slices_map[z] = os.path.join(DCM_DIR, fname)
        except:
            pass

    sorted_z = sorted(slices_map.keys())
    print(f"  {len(sorted_z)} slices, Z range: {sorted_z[0]:.0f} ~ {sorted_z[-1]:.0f}mm")

    hu_slices = []
    ds_ref = None
    for z in sorted_z:
        ds = pydicom.dcmread(slices_map[z])
        slope = float(getattr(ds, 'RescaleSlope', 1.0))
        intercept = float(getattr(ds, 'RescaleIntercept', 0.0))
        hu_slices.append(ds.pixel_array.astype(np.float32) * slope + intercept)
        if ds_ref is None:
            ds_ref = ds

    volume = np.stack(hu_slices, axis=0)
    pixel_spacing = float(ds_ref.PixelSpacing[0])
    z_spacing = abs(sorted_z[1] - sorted_z[0])

    print(f"  Volume: {volume.shape}, spacing: {pixel_spacing:.2f}x{pixel_spacing:.2f}x{z_spacing:.1f}mm")
    return volume, sorted_z, pixel_spacing, z_spacing, ds_ref


def create_body_mask(volume):
    """Create tight body interior mask, excluding skin/air boundary."""
    body_raw = volume > -200
    for si in range(body_raw.shape[0]):
        body_raw[si] = ndimage.binary_fill_holes(body_raw[si])
    # Erode to get interior only (15 voxels ~ 12mm)
    body_interior = ndimage.binary_erosion(body_raw, iterations=15)
    return body_raw, body_interior


# ====================================================================
#  METHOD 1: HU ANOMALY DETECTION
# ====================================================================
def hu_anomaly_detection(volume, pixel_spacing, z_spacing):
    """Detect suspicious regions based on HU anomalies."""
    print("\n[2/7] HU Anomaly Detection...")
    t0 = time.time()

    _, body_interior = create_body_mask(volume)

    print("  Computing local statistics...")
    vol64 = volume.astype(np.float64)
    local_mean = ndimage.uniform_filter(vol64, size=15)
    local_sq = ndimage.uniform_filter(vol64**2, size=15)
    local_std = np.sqrt(np.maximum(local_sq - local_mean**2, 0) + 1e-8)

    z_score = np.abs(volume - local_mean) / local_std
    z_score[~body_interior] = 0

    # Suspicious: enhanced tissue (contrast) with high z-score
    enhanced = (volume >= 80) & (volume <= 350) & body_interior
    suspicious = (z_score > 2.0) & enhanced

    # Hypo/hyper-dense soft tissue lesions
    soft = (volume >= 10) & (volume <= 90) & body_interior
    hypodense = (z_score > 2.5) & soft & (volume < local_mean - 10)
    hyperdense = (z_score > 2.5) & soft & (volume > local_mean + 15)

    combined = suspicious | hypodense | hyperdense
    combined = ndimage.binary_closing(combined, iterations=2)
    combined = ndimage.binary_opening(combined, iterations=1)

    labeled, num = ndimage.label(combined)
    print(f"  {num} candidates found")

    voxel_vol = pixel_spacing * pixel_spacing * z_spacing

    detections = []
    for i in range(1, num + 1):
        comp = labeled == i
        voxels = int(np.sum(comp))
        vol_mm3 = voxels * voxel_vol

        if vol_mm3 < 50 or vol_mm3 > 50000:
            continue

        coords = np.argwhere(comp)
        centroid = coords.mean(axis=0)
        bbox_min = coords.min(axis=0)
        bbox_max = coords.max(axis=0)

        diameter = 2 * (3 * vol_mm3 / (4 * np.pi)) ** (1/3)

        hu_vals = volume[comp]
        hu_mean = float(np.mean(hu_vals))
        hu_std = float(np.std(hu_vals))
        z_mean = float(np.mean(z_score[comp]))

        # Shape irregularity
        eroded = ndimage.binary_erosion(comp)
        surface_vox = int(np.sum(comp != eroded))
        sphere_surf = (36 * np.pi * vol_mm3**2)**(1/3)
        irregularity = (surface_vox * pixel_spacing**2) / max(sphere_surf, 1e-8)

        detections.append({
            'id': len(detections) + 1,
            'label_idx': i,
            'centroid_voxel': centroid.tolist(),
            'centroid_mm': [
                round(float(centroid[0] * z_spacing), 1),
                round(float(centroid[1] * pixel_spacing), 1),
                round(float(centroid[2] * pixel_spacing), 1),
            ],
            'bbox_min': bbox_min.tolist(),
            'bbox_max': bbox_max.tolist(),
            'voxel_count': voxels,
            'volume_mm3': round(vol_mm3, 1),
            'diameter_mm': round(diameter, 1),
            'hu_mean': round(hu_mean, 1),
            'hu_std': round(hu_std, 1),
            'hu_min': round(float(np.min(hu_vals)), 1),
            'hu_max': round(float(np.max(hu_vals)), 1),
            'z_score_mean': round(z_mean, 2),
            'irregularity': round(float(irregularity), 3),
            'method': 'HU_anomaly',
        })

    detections.sort(key=lambda d: d['z_score_mean'], reverse=True)

    # Create detection mask
    det_mask = np.zeros(volume.shape, dtype=np.uint8)
    for d in detections:
        det_mask[labeled == d['label_idx']] = min(d['id'], 255)

    elapsed = time.time() - t0
    print(f"  Result: {len(detections)} regions (in {elapsed:.1f}s)")
    for d in detections[:5]:
        print(f"    #{d['id']}: {d['diameter_mm']}mm, HU={d['hu_mean']:.0f}+/-{d['hu_std']:.0f}, Z-score={d['z_score_mean']:.2f}")

    return detections, det_mask


# ====================================================================
#  METHOD 2: nnU-Net INFERENCE
# ====================================================================
def nnunet_inference(volume, sorted_z, pixel_spacing, z_spacing, ds_ref):
    """Run nnU-Net inference with CPU fallback."""
    print("\n[3/7] nnU-Net Inference...")
    t0 = time.time()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    try:
        import nibabel as nib
        from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
        import torch

        nifti_dir = os.path.join(OUTPUT_DIR, 'nnunet_input')
        output_dir = os.path.join(OUTPUT_DIR, 'nnunet_output')
        os.makedirs(nifti_dir, exist_ok=True)
        os.makedirs(output_dir, exist_ok=True)

        # Save as NIfTI
        ipp = [float(x) for x in ds_ref.ImagePositionPatient]
        affine = np.eye(4)
        affine[0, 0], affine[1, 1], affine[2, 2] = pixel_spacing, pixel_spacing, z_spacing
        affine[0, 3], affine[1, 3], affine[2, 3] = ipp[0], ipp[1], ipp[2]

        nib.save(nib.Nifti1Image(volume.astype(np.float32), affine),
                 os.path.join(nifti_dir, 'CT_0000.nii.gz'))

        # Use CPU explicitly
        device = torch.device('cpu')
        print(f"  Device: {device}")

        predictor = nnUNetPredictor(
            tile_step_size=0.5,
            use_gaussian=True,
            use_mirroring=False,  # faster on CPU
            perform_everything_on_device=True,
            device=device,
            verbose=False,
            verbose_preprocessing=False,
            allow_tqdm=True
        )

        model_folder = os.path.join(NNUNET_RESULTS, DATASET, TRAINER)
        predictor.initialize_from_trained_model_folder(
            model_folder, use_folds=(0,), checkpoint_name='checkpoint_best.pth'
        )

        predictor.predict_from_files(
            list_of_lists_or_source_folder=nifti_dir,
            output_folder_or_list_of_truncated_output_files=output_dir,
            save_probabilities=False, overwrite=True,
            num_processes_preprocessing=1, num_processes_segmentation_export=1,
        )

        pred_files = [f for f in os.listdir(output_dir) if f.endswith('.nii.gz')]
        if pred_files:
            pred = nib.load(os.path.join(output_dir, pred_files[0])).get_fdata().astype(np.uint8)
            print(f"  Prediction: {pred.shape}, labels: {np.unique(pred)}")
            nn_det = _analyze_nn_mask(pred, volume, pixel_spacing, z_spacing)
            print(f"  nnU-Net: {len(nn_det)} detections ({time.time()-t0:.1f}s)")
            return nn_det, pred
    except Exception as e:
        print(f"  nnU-Net failed: {e}")
        print("  Continuing with HU detection only")

    return [], np.zeros(volume.shape, dtype=np.uint8)


def _analyze_nn_mask(mask, volume, pixel_spacing, z_spacing):
    detections = []
    voxel_vol = pixel_spacing * pixel_spacing * z_spacing
    for lbl in np.unique(mask):
        if lbl == 0:
            continue
        labeled, n = ndimage.label(mask == lbl)
        for j in range(1, n + 1):
            region = labeled == j
            voxels = int(np.sum(region))
            vol_mm3 = voxels * voxel_vol
            if vol_mm3 < 30:
                continue
            coords = np.argwhere(region)
            centroid = coords.mean(axis=0)
            hu_vals = volume[region]
            detections.append({
                'id': len(detections) + 1,
                'centroid_voxel': centroid.tolist(),
                'centroid_mm': [round(float(centroid[k] * [z_spacing, pixel_spacing, pixel_spacing][k]), 1) for k in range(3)],
                'voxel_count': voxels,
                'volume_mm3': round(vol_mm3, 1),
                'diameter_mm': round(2 * (3 * vol_mm3 / (4 * np.pi)) ** (1/3), 1),
                'hu_mean': round(float(np.mean(hu_vals)), 1),
                'hu_std': round(float(np.std(hu_vals)), 1),
                'method': 'nnUNet',
            })
    return detections


# ====================================================================
#  FEATURE EXTRACTION (Radiomics-style)
# ====================================================================
def extract_features(volume, detections, det_mask, pixel_spacing, z_spacing):
    """Extract radiomics-style features from each detection."""
    print("\n[4/7] Extracting Features...")

    for d in detections:
        idx = d['id']
        comp = det_mask == min(idx, 255)
        if np.sum(comp) == 0:
            continue

        hu = volume[comp]

        # ---- Shape Features ----
        coords = np.argwhere(comp)
        # Scale to mm
        coords_mm = coords.astype(float) * [z_spacing, pixel_spacing, pixel_spacing]

        # Principal component analysis for elongation
        if len(coords_mm) > 3:
            centered = coords_mm - coords_mm.mean(axis=0)
            cov = np.cov(centered.T)
            eigvals = np.sort(np.linalg.eigvalsh(cov))[::-1]
            eigvals = np.maximum(eigvals, 1e-10)
            elongation = np.sqrt(eigvals[1] / eigvals[0]) if eigvals[0] > 0 else 1.0
            flatness = np.sqrt(eigvals[2] / eigvals[0]) if eigvals[0] > 0 else 1.0
        else:
            elongation, flatness = 1.0, 1.0

        # Sphericity: ratio of surface of equivalent sphere to actual surface
        vol_mm3 = d['volume_mm3']
        equiv_r = (3 * vol_mm3 / (4 * np.pi)) ** (1/3)
        equiv_sphere_surf = 4 * np.pi * equiv_r**2

        eroded = ndimage.binary_erosion(comp)
        surface_vox = int(np.sum(comp != eroded))
        approx_surf = surface_vox * pixel_spacing * pixel_spacing  # rough
        sphericity = equiv_sphere_surf / max(approx_surf, 1e-8)
        sphericity = min(sphericity, 1.0)

        # ---- Intensity Features ----
        hu_percentiles = np.percentile(hu, [10, 25, 50, 75, 90])
        skewness = float(np.mean(((hu - np.mean(hu)) / max(np.std(hu), 1e-8))**3))
        kurtosis = float(np.mean(((hu - np.mean(hu)) / max(np.std(hu), 1e-8))**4)) - 3

        # Energy & Entropy (discretized)
        hu_clipped = np.clip(hu, -200, 400).astype(int) + 200
        bins = np.bincount(hu_clipped, minlength=601)
        probs = bins / max(bins.sum(), 1)
        probs = probs[probs > 0]
        entropy = float(-np.sum(probs * np.log2(probs)))
        energy = float(np.sum(probs**2))

        # ---- Texture Features (simplified GLCM-like) ----
        # Local contrast: mean absolute difference with neighbors
        slc = (slice(max(0, d['bbox_min'][0]-1), min(volume.shape[0], d['bbox_max'][0]+2)),
               slice(max(0, d['bbox_min'][1]-1), min(volume.shape[1], d['bbox_max'][1]+2)),
               slice(max(0, d['bbox_min'][2]-1), min(volume.shape[2], d['bbox_max'][2]+2)))
        patch = volume[slc]
        patch_mask = comp[slc]
        if patch_mask.any():
            sobelx = ndimage.sobel(patch, axis=2)
            sobely = ndimage.sobel(patch, axis=1)
            sobelz = ndimage.sobel(patch, axis=0)
            gradient_mag = np.sqrt(sobelx**2 + sobely**2 + sobelz**2)
            edge_strength = float(np.mean(gradient_mag[patch_mask]))
        else:
            edge_strength = 0.0

        # Homogeneity estimate (inverse difference)
        homogeneity = 1.0 / (1.0 + d['hu_std']**2 / 1000.0)

        # ---- Store features ----
        d['features'] = {
            'shape': {
                'sphericity': round(sphericity, 3),
                'elongation': round(elongation, 3),
                'flatness': round(flatness, 3),
                'surface_voxels': surface_vox,
            },
            'intensity': {
                'hu_p10': round(float(hu_percentiles[0]), 1),
                'hu_p25': round(float(hu_percentiles[1]), 1),
                'hu_median': round(float(hu_percentiles[2]), 1),
                'hu_p75': round(float(hu_percentiles[3]), 1),
                'hu_p90': round(float(hu_percentiles[4]), 1),
                'skewness': round(skewness, 3),
                'kurtosis': round(kurtosis, 3),
                'entropy': round(entropy, 3),
                'energy': round(energy, 4),
            },
            'texture': {
                'edge_strength': round(edge_strength, 2),
                'homogeneity': round(homogeneity, 4),
            },
        }

        print(f"  #{idx}: spher={sphericity:.2f}, elong={elongation:.2f}, "
              f"entropy={entropy:.1f}, edge={edge_strength:.0f}")

    return detections


# ====================================================================
#  3D VIEWER - Body=natural colors, Tumor=GREEN glow
# ====================================================================
def generate_viewer(volume, hu_detections, hu_mask, nn_detections, nn_mask,
                    pixel_spacing, z_spacing):
    """Generate 3D viewer with natural body + green tumor overlay."""
    print("\n[6/7] Generating 3D Viewer...")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Body mask for rendering
    body_raw, _ = create_body_mask(volume)
    body_eroded = ndimage.binary_erosion(body_raw, iterations=3)
    volume_masked = volume.copy()
    volume_masked[~body_eroded] = -1024

    # Downsample
    zoom = [VOL_SIZE / s for s in volume.shape]
    vol_ds = ndimage.zoom(volume_masked, zoom, order=3)
    vol_ds = ndimage.gaussian_filter(vol_ds, sigma=0.7)
    body_ds = ndimage.zoom(body_eroded.astype(np.float32), zoom, order=1) > 0.5
    vol_ds[~body_ds] = -1024

    # Normalize
    hu_min, hu_max = -150, 400
    norm = np.clip((vol_ds - hu_min) / (hu_max - hu_min), 0, 1)
    vol_u8 = (norm * 255).astype(np.uint8)
    vol_u8[~body_ds] = 0

    # Downsample masks
    hu_m = ndimage.zoom(hu_mask.astype(float), zoom, order=0) > 0.5
    nn_m = ndimage.zoom(nn_mask.astype(float), zoom, order=0) > 0.5

    # Tumor mask: 0=none, 85=HU, 170=nnUNet, 255=both
    tmask = np.zeros(vol_u8.shape, dtype=np.uint8)
    tmask[hu_m] = 85
    tmask[nn_m] = 170
    tmask[hu_m & nn_m] = 255
    tmask[vol_u8 < 10] = 0   # no tumors in air

    # Pack as 2-channel interleaved
    packed = np.zeros((*vol_u8.shape, 2), dtype=np.uint8)
    packed[..., 0] = vol_u8
    packed[..., 1] = tmask
    packed_t = np.ascontiguousarray(packed.transpose(2, 1, 0, 3))

    compressed = gzip.compress(packed_t.tobytes(), compresslevel=6)
    b64 = base64.b64encode(compressed).decode('ascii')

    nx, ny, nz = packed_t.shape[0], packed_t.shape[1], packed_t.shape[2]
    print(f"  Volume: {nx}x{ny}x{nz}, compressed: {len(compressed)//1024}KB")

    # Top 15 detections for overlay cards
    all_det = (hu_detections + nn_detections)[:15]
    det_json = json.dumps(all_det, default=str)

    html = _build_html(nx, ny, nz, b64, len(compressed),
                       det_json, len(hu_detections), len(nn_detections))

    path = os.path.join(OUTPUT_DIR, 'tumor_3d_viewer.html')
    with open(path, 'w', encoding='utf-8') as f:
        f.write(html)
    print(f"  Saved: {path}")
    return path


def _build_html(nx, ny, nz, b64, comp_size, det_json, n_hu, n_nn):
    return f"""<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="UTF-8">
<title>CT Tumor Detection v3</title>
<style>
*{{margin:0;padding:0;box-sizing:border-box}}
body{{background:#0d1117;font-family:'Segoe UI',system-ui,sans-serif;overflow:hidden;color:#c9d1d9}}
canvas{{display:block;width:100vw;height:100vh}}
#ui{{position:fixed;top:10px;left:10px;z-index:10;background:rgba(13,17,23,0.96);
  border-radius:14px;padding:16px;width:300px;max-height:96vh;overflow-y:auto;
  border:1px solid #30363d;box-shadow:0 8px 24px rgba(0,0,0,0.5)}}
h1{{font-size:1.2em;margin-bottom:2px;color:#58a6ff}}
.sub{{font-size:.7em;color:#484f58;margin-bottom:10px}}
.sec{{margin-bottom:12px}}
.sec h3{{font-size:.72em;color:#3fb950;text-transform:uppercase;letter-spacing:.8px;margin-bottom:6px;
  padding-bottom:2px;border-bottom:1px solid #21262d}}
.pbtn{{display:flex;gap:3px;flex-wrap:wrap;margin-bottom:6px}}
.pbtn button{{flex:1;min-width:50px;padding:6px 2px;font-size:.7em;border:1px solid #30363d;
  background:#161b22;border-radius:6px;cursor:pointer;transition:all .12s;color:#8b949e}}
.pbtn button:hover{{background:#1f6feb33;border-color:#58a6ff}}
.pbtn button.on{{background:#238636;color:#fff;border-color:#238636}}
.r{{display:flex;align-items:center;gap:5px;margin:4px 0}}
.r label{{font-size:.72em;color:#8b949e;min-width:58px}}
.r input[type=range]{{flex:1;accent-color:#3fb950}}
.r span{{font-size:.68em;color:#3fb950;min-width:26px;text-align:right}}
.r input[type=checkbox]{{accent-color:#3fb950}}
.sbox{{background:#161b22;border:1px solid #30363d;border-radius:8px;padding:10px;margin-bottom:10px}}
.sbox .row{{display:flex;justify-content:space-between;margin:2px 0;font-size:.78em}}
.sbox .lbl{{color:#8b949e}}.sbox .v1{{color:#f85149}}.sbox .v2{{color:#3fb950}}
.det{{background:#161b22;border:1px solid #30363d;border-radius:8px;padding:8px;margin:4px 0;font-size:.72em}}
.det .hdr{{display:flex;justify-content:space-between;align-items:center;margin-bottom:3px}}
.det .tag{{font-size:.62em;padding:1px 5px;border-radius:3px;font-weight:700}}
.tag-hu{{background:#f85149;color:#fff}}.tag-nn{{background:#3fb950;color:#fff}}
.det .m{{display:grid;grid-template-columns:1fr 1fr;gap:1px;font-size:.68em;color:#8b949e}}
.det .m .v{{color:#c9d1d9;font-weight:600}}
#info{{font-size:.64em;color:#484f58;margin-top:6px}}
#ld{{position:fixed;top:50%;left:50%;transform:translate(-50%,-50%);z-index:20;
  text-align:center;color:#58a6ff;background:#0d1117ee;padding:30px;border-radius:14px;
  box-shadow:0 8px 30px rgba(0,0,0,0.6)}}
.sp{{width:40px;height:40px;border:3px solid #30363d;border-top-color:#3fb950;border-radius:50%;
  animation:sp .7s linear infinite;margin:0 auto 10px}}
@keyframes sp{{to{{transform:rotate(360deg)}}}}
</style>
</head>
<body>
<div id="ld"><div class="sp"></div><div id="lm">Loading...</div></div>
<div id="ui">
  <h1>CT Tumor Detection</h1>
  <div class="sub">{nx}x{ny}x{nz} | Dual Method</div>
  <div class="sbox">
    <div class="row"><span class="lbl">HU Anomaly</span><span class="v1">{n_hu} regions</span></div>
    <div class="row"><span class="lbl">nnU-Net DL</span><span class="v2">{n_nn} regions</span></div>
  </div>
  <div class="sec"><h3>Mode</h3>
    <div class="pbtn" id="pB">
      <button class="on" onclick="P('tumor')">Tumor</button>
      <button onclick="P('abd')">Abdomen</button>
      <button onclick="P('bone')">Bone</button>
      <button onclick="P('xray')">X-Ray</button>
    </div>
  </div>
  <div class="sec"><h3>Controls</h3>
    <div class="r"><label>Tumors</label><input type="checkbox" id="cT" checked onchange="U()"></div>
    <div class="r"><label>Glow</label><input type="range" id="sG" min="0" max="100" value="90" oninput="U()"><span id="vG">90</span></div>
    <div class="r"><label>Density</label><input type="range" id="sD" min="1" max="60" value="18" oninput="U()"><span id="vD">18</span></div>
    <div class="r"><label>Bright</label><input type="range" id="sB" min="20" max="300" value="160" oninput="U()"><span id="vB">160</span></div>
    <div class="r"><label>Steps</label><input type="range" id="sS" min="50" max="400" value="200" oninput="U()"><span id="vS">200</span></div>
  </div>
  <div class="sec"><h3>Clip</h3>
    <div class="r"><label>X</label><input type="range" id="cX" min="0" max="100" value="100" oninput="U()"><span id="vX">100</span></div>
    <div class="r"><label>Y</label><input type="range" id="cY" min="0" max="100" value="100" oninput="U()"><span id="vY">100</span></div>
    <div class="r"><label>Z</label><input type="range" id="cZ" min="0" max="100" value="100" oninput="U()"><span id="vZ">100</span></div>
  </div>
  <div class="sec"><h3>Detections</h3><div id="dL"></div></div>
  <div class="r"><label>Rotate</label><input type="checkbox" id="cR" checked onchange="ctl.autoRotate=this.checked"></div>
  <div id="info">Loading...</div>
</div>

<script type="importmap">
{{"imports":{{"three":"https://cdn.jsdelivr.net/npm/three@0.160.0/build/three.module.js","three/addons/":"https://cdn.jsdelivr.net/npm/three@0.160.0/examples/jsm/"}}}}
</script>
<script type="module">
import * as THREE from 'three';
import {{OrbitControls}} from 'three/addons/controls/OrbitControls.js';

const NX={nx},NY={ny},NZ={nz};
const B64="{b64}";
const DET={det_json};
let R,scene,cam,ctl,mat;

async function dec(s){{
  const b=Uint8Array.from(atob(s),c=>c.charCodeAt(0));
  const d=new DecompressionStream('gzip');
  const w=d.writable.getWriter();w.write(b);w.close();
  const r=d.readable.getReader();const ch=[];
  while(true){{const{{done,value}}=await r.read();if(done)break;ch.push(value);}}
  let t=0;ch.forEach(c=>t+=c.length);
  const res=new Uint8Array(t);let o=0;
  ch.forEach(c=>{{res.set(c,o);o+=c.length;}});return res;
}}

async function init(){{
  document.getElementById('lm').textContent='Decompressing...';
  const pk=await dec(B64);
  document.getElementById('lm').textContent='Creating textures...';

  const sz=NX*NY*NZ;
  const vd=new Uint8Array(sz),md=new Uint8Array(sz);
  for(let i=0;i<sz;i++){{vd[i]=pk[i*2];md[i]=pk[i*2+1];}}

  R=new THREE.WebGLRenderer({{antialias:true}});
  R.setSize(innerWidth,innerHeight);
  R.setPixelRatio(Math.min(devicePixelRatio,2));
  document.body.appendChild(R.domElement);

  scene=new THREE.Scene();
  scene.background=new THREE.Color(0x0d1117);
  cam=new THREE.PerspectiveCamera(50,innerWidth/innerHeight,0.01,20);
  cam.position.set(0,0,2.5);

  ctl=new OrbitControls(cam,R.domElement);
  ctl.enableDamping=true;ctl.dampingFactor=0.05;
  ctl.autoRotate=true;ctl.autoRotateSpeed=1.5;

  const vt=new THREE.Data3DTexture(vd,NX,NY,NZ);
  vt.format=THREE.RedFormat;vt.type=THREE.UnsignedByteType;
  vt.minFilter=THREE.LinearFilter;vt.magFilter=THREE.LinearFilter;
  vt.wrapS=vt.wrapT=vt.wrapR=THREE.ClampToEdgeWrapping;vt.needsUpdate=true;

  const mt=new THREE.Data3DTexture(md,NX,NY,NZ);
  mt.format=THREE.RedFormat;mt.type=THREE.UnsignedByteType;
  mt.minFilter=THREE.NearestFilter;mt.magFilter=THREE.NearestFilter;
  mt.wrapS=mt.wrapT=mt.wrapR=THREE.ClampToEdgeWrapping;mt.needsUpdate=true;

  mat=new THREE.RawShaderMaterial({{
    glslVersion:THREE.GLSL3,
    uniforms:{{
      uV:{{value:vt}},uM:{{value:mt}},
      uSteps:{{value:200.0}},uBr:{{value:1.6}},uDn:{{value:18.0}},
      uClip:{{value:new THREE.Vector3(1,1,1)}},
      uMode:{{value:0}},uTumor:{{value:1}},uGlow:{{value:0.9}},uTime:{{value:0.0}},
    }},
    vertexShader:`
      in vec3 position;
      uniform mat4 modelViewMatrix;uniform mat4 projectionMatrix;
      out vec3 vP,vC;
      void main(){{
        vP=position+0.5;
        vC=(inverse(modelViewMatrix)*vec4(0,0,0,1)).xyz+0.5;
        gl_Position=projectionMatrix*modelViewMatrix*vec4(position,1);
      }}`,
    fragmentShader:`
      precision highp float;precision highp sampler3D;
      in vec3 vP,vC;out vec4 oC;
      uniform sampler3D uV,uM;
      uniform float uSteps,uBr,uDn,uTime,uGlow;
      uniform vec3 uClip;uniform int uMode,uTumor;

      vec2 box(vec3 o,vec3 d){{
        vec3 a=min(-o/d,(vec3(1)-o)/d),b=max(-o/d,(vec3(1)-o)/d);
        return vec2(max(max(a.x,a.y),a.z),min(min(b.x,b.y),b.z));
      }}

      // Body in NATURAL colors (beige/gray, NOT red)
      vec4 body(float v){{
        vec3 c=vec3(0);float a=0.0;
        if(uMode==0){{ // Tumor view: neutral semi-transparent body
          if(v<0.10) a=0.0;
          else if(v<0.25){{c=vec3(0.75,0.68,0.58);a=(v-0.10)*0.2;}}  // fat: warm beige
          else if(v<0.40){{c=vec3(0.70,0.55,0.50);a=0.04;}}            // tissue: muted pink
          else if(v<0.55){{c=vec3(0.65,0.50,0.45);a=0.06;}}            // organ: slightly darker
          else if(v<0.72){{c=vec3(0.60,0.48,0.42);a=0.08;}}            // dense tissue
          else{{float t=min((v-0.72)/0.28,1.0);c=mix(vec3(0.82,0.78,0.72),vec3(0.92,0.90,0.85),t);a=0.25+t*0.25;}} // bone
        }}else if(uMode==1){{ // Abdomen: full color
          if(v<0.10) a=0.0;
          else if(v<0.20){{c=vec3(0.90,0.82,0.55);a=(v-0.10)*0.8;}}
          else if(v<0.30){{c=vec3(0.85,0.65,0.50);a=0.06+(v-0.20)*0.5;}}
          else if(v<0.42){{float t=(v-0.30)/0.12;c=mix(vec3(0.80,0.48,0.40),vec3(0.70,0.32,0.28),t);a=0.15+t*0.12;}}
          else if(v<0.55){{float t=(v-0.42)/0.13;c=mix(vec3(0.65,0.30,0.25),vec3(0.75,0.28,0.22),t);a=0.25+t*0.12;}}
          else if(v<0.72){{float t=(v-0.55)/0.17;c=mix(vec3(0.80,0.35,0.25),vec3(0.88,0.42,0.28),t);a=0.35+t*0.12;}}
          else{{float t=min((v-0.72)/0.28,1.0);c=mix(vec3(0.88,0.85,0.75),vec3(0.98,0.95,0.88),t);a=0.55+t*0.35;}}
        }}else if(uMode==2){{ // Bone
          if(v<0.68) a=0.0;
          else{{float t=min((v-0.68)/0.32,1.0);c=mix(vec3(0.82,0.78,0.68),vec3(0.98,0.96,0.90),t);a=0.6+t*0.4;}}
        }}else{{ // X-Ray
          if(v<0.04) a=0.0;
          else{{a=v*v*1.2;c=vec3(v);}}
        }}
        return vec4(c,a);
      }}

      void main(){{
        vec3 rd=normalize(vP-vC);
        vec2 bx=box(vC,rd);
        if(bx.x>bx.y) discard;
        bx.x=max(bx.x,0.001);
        float dt=(bx.y-bx.x)/uSteps;
        vec3 p=vC+bx.x*rd,st=rd*dt;
        vec4 acc=vec4(0);
        float pulse=0.6+0.4*sin(uTime*2.5);

        for(float i=0.0;i<400.0;i+=1.0){{
          if(i>=uSteps)break;
          if(p.x<0.0||p.y<0.0||p.z<0.0||p.x>uClip.x||p.y>uClip.y||p.z>uClip.z){{p+=st;continue;}}

          float raw=texture(uV,p).r;
          // HARD skip: air voxels (value 0-3/255) -> completely invisible
          if(raw<0.015){{p+=st;continue;}}

          float msk=texture(uM,p).r;
          vec4 col=body(raw);

          // TUMOR OVERLAY: bright GREEN (very different from body)
          if(uTumor==1 && msk>0.05 && raw>0.10){{
            vec3 tc;float ta;
            if(msk>0.8){{ // Both methods
              tc=vec3(1.0,0.95,0.0)*pulse;ta=0.95*uGlow; // bright yellow
            }}else if(msk>0.5){{ // nnU-Net
              tc=vec3(0.0,1.0,0.8)*pulse;ta=0.85*uGlow;  // cyan-green
            }}else{{ // HU anomaly
              tc=vec3(0.0,1.0,0.3)*pulse;ta=0.80*uGlow;  // bright GREEN
            }}
            col.rgb=mix(col.rgb,tc,ta);
            col.a=max(col.a,ta*0.6);
          }}

          float sa=col.a*uDn*dt*2.0;
          acc.rgb+=(1.0-acc.a)*col.rgb*sa*uBr;
          acc.a+=(1.0-acc.a)*sa;
          if(acc.a>0.95)break;
          p+=st;
        }}
        vec3 bg=vec3(0.051,0.067,0.09);
        oC=vec4(mix(bg,acc.rgb,min(acc.a,1.0)),1.0);
      }}`,
    side:THREE.BackSide,transparent:false,
  }});

  scene.add(new THREE.Mesh(new THREE.BoxGeometry(1,1,1),mat));
  document.getElementById('ld').style.display='none';
  document.getElementById('info').innerHTML='GPU Ray Marching + Tumor Overlay';

  // Detection cards
  const dl=document.getElementById('dL');
  DET.forEach(d=>{{
    const tg=d.method==='nnUNet'?'tag-nn':'tag-hu';
    const tt=d.method==='nnUNet'?'DL':'HU';
    const f=d.features;
    const card=document.createElement('div');card.className='det';
    card.innerHTML=`<div class="hdr"><strong>#${{d.id}}</strong><span class="tag ${{tg}}">${{tt}}</span></div>
      <div class="m">
        <div>Size: <span class="v">${{d.diameter_mm}}mm</span></div>
        <div>Vol: <span class="v">${{d.volume_mm3}}mm3</span></div>
        <div>HU: <span class="v">${{d.hu_mean||'-'}}</span></div>
        <div>Z: <span class="v">${{d.z_score_mean||'-'}}</span></div>
        ${{f?'<div>Spher: <span class="v">'+f.shape.sphericity+'</span></div>':''}}
        ${{f?'<div>Entropy: <span class="v">'+f.intensity.entropy+'</span></div>':''}}
      </div>`;
    dl.appendChild(card);
  }});

  addEventListener('resize',()=>{{cam.aspect=innerWidth/innerHeight;cam.updateProjectionMatrix();R.setSize(innerWidth,innerHeight);}});
  const clk=new THREE.Clock();
  (function lp(){{requestAnimationFrame(lp);ctl.update();mat.uniforms.uTime.value=clk.getElapsedTime();R.render(scene,cam);}})();
}}

window.U=function(){{
  if(!mat)return;const u=mat.uniforms;
  const g=id=>parseInt(document.getElementById(id).value);
  const s=(id,v)=>document.getElementById(id).textContent=v;
  u.uSteps.value=g('sS');s('vS',g('sS'));
  u.uBr.value=g('sB')/100;s('vB',g('sB'));
  u.uDn.value=g('sD');s('vD',g('sD'));
  u.uGlow.value=g('sG')/100;s('vG',g('sG'));
  u.uClip.value.set(g('cX')/100,g('cY')/100,g('cZ')/100);
  s('vX',g('cX'));s('vY',g('cY'));s('vZ',g('cZ'));
  u.uTumor.value=document.getElementById('cT').checked?1:0;
}};
window.P=function(n){{
  const p={{tumor:{{s:200,b:160,d:18,m:0}},abd:{{s:200,b:160,d:28,m:1}},bone:{{s:180,b:140,d:35,m:2}},xray:{{s:300,b:200,d:12,m:3}}}};
  const v=p[n];if(!v)return;
  document.getElementById('sS').value=v.s;document.getElementById('sB').value=v.b;
  document.getElementById('sD').value=v.d;
  if(mat)mat.uniforms.uMode.value=v.m;
  document.querySelectorAll('#pB button').forEach(b=>b.classList.remove('on'));
  event.target.classList.add('on');U();
}};

init().catch(e=>{{document.getElementById('lm').textContent='Error: '+e.message;console.error(e);}});
</script>
</body>
</html>"""


def save_report(hu_det, nn_det, output_dir):
    """Save full detection report."""
    print("\n[7/7] Saving report...")
    report = {
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'hu_anomaly': {'count': len(hu_det), 'detections': hu_det},
        'nnunet': {'count': len(nn_det), 'detections': nn_det},
    }
    rpath = os.path.join(output_dir, 'detection_report.json')
    with open(rpath, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False, default=str)
    print(f"  Report: {rpath}")

    # Feature summary table
    print("\n" + "=" * 60)
    print("  DETECTION SUMMARY WITH FEATURES")
    print("=" * 60)
    print(f"  {'#':>3} {'Size':>6} {'HU':>7} {'ZScore':>6} {'Spher':>6} {'Entropy':>8} {'Edge':>6}")
    print("  " + "-" * 50)
    for d in hu_det[:10]:
        f = d.get('features', {})
        sh = f.get('shape', {})
        it = f.get('intensity', {})
        tx = f.get('texture', {})
        print(f"  {d['id']:>3} {d['diameter_mm']:>5.1f}m "
              f"{d['hu_mean']:>6.0f} {d['z_score_mean']:>6.2f} "
              f"{sh.get('sphericity','?'):>6} {it.get('entropy','?'):>8} "
              f"{tx.get('edge_strength','?'):>6}")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    volume, sorted_z, ps, zs, ds_ref = load_volume()

    # Method 1: HU anomaly
    hu_det, hu_mask = hu_anomaly_detection(volume, ps, zs)

    # Method 2: nnU-Net
    nn_det, nn_mask = nnunet_inference(volume, sorted_z, ps, zs, ds_ref)

    # Feature extraction
    hu_det = extract_features(volume, hu_det, hu_mask, ps, zs)

    # Report
    print("\n[5/7] Combining results...")
    save_report(hu_det, nn_det, OUTPUT_DIR)

    # 3D viewer
    path = generate_viewer(volume, hu_det, hu_mask, nn_det, nn_mask, ps, zs)

    print(f"\n  DONE! Open: {path}")


if __name__ == '__main__':
    main()
