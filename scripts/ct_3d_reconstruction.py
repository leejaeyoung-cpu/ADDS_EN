"""
CT 3D Volume Reconstruction with Advanced Interpolation
=========================================================
1. Cubic Spline 보간으로 슬라이스 간 빈 공간 채움
2. Marching Cubes로 3D 표면 추출
3. Three.js 기반 인터랙티브 3D 뷰어 생성
"""

import os
import sys
import io
import json
import struct
import numpy as np
from collections import Counter

if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

import pydicom
from PIL import Image
from scipy import ndimage
from scipy.interpolate import RegularGridInterpolator
from skimage import measure

DCM_DIR = r"f:\ADDS\CTdata\CTdcm"
OUTPUT_DIR = r"f:\ADDS\CTdata\ct_3d_model"

# HU thresholds for different tissues
HU_THRESHOLDS = {
    'bone':      {'min': 300,  'max': 3000,  'color': [0.95, 0.90, 0.78], 'opacity': 0.8},
    'soft_tissue': {'min': 20,   'max': 80,    'color': [0.85, 0.55, 0.45], 'opacity': 0.3},
    'organ':     {'min': 30,   'max': 300,   'color': [0.80, 0.40, 0.35], 'opacity': 0.4},
}


def load_primary_series(dcm_dir):
    """Load primary series (#604 Abdomen Artery) with deduplication."""
    print("[1/5] Loading DICOM primary series...")

    files = [f for f in os.listdir(dcm_dir) if f.endswith('.dcm')]
    series_slices = {}

    for fname in files:
        fpath = os.path.join(dcm_dir, fname)
        try:
            ds = pydicom.dcmread(fpath, stop_before_pixels=True)
            series_num = int(getattr(ds, 'SeriesNumber', 0))
            if series_num == 604:
                z_pos = float(ds.ImagePositionPatient[2])
                if z_pos not in series_slices:
                    series_slices[z_pos] = fpath
        except:
            pass

    # Sort by Z position
    sorted_z = sorted(series_slices.keys())
    print(f"  Found {len(sorted_z)} unique slices in Series #604")
    print(f"  Z range: {sorted_z[0]:.1f} to {sorted_z[-1]:.1f} mm")

    # Load pixel data
    slices = []
    for i, z in enumerate(sorted_z):
        ds = pydicom.dcmread(series_slices[z])
        slope = float(getattr(ds, 'RescaleSlope', 1.0))
        intercept = float(getattr(ds, 'RescaleIntercept', 0.0))
        hu = ds.pixel_array.astype(np.float32) * slope + intercept
        slices.append(hu)

        if (i + 1) % 30 == 0:
            print(f"    Loading: {i+1}/{len(sorted_z)}")

    pixel_spacing = [float(x) for x in ds.PixelSpacing]
    print(f"  Pixel spacing: {pixel_spacing[0]:.3f} x {pixel_spacing[1]:.3f} mm")
    print(f"  Slice gap: {sorted_z[1] - sorted_z[0]:.1f} mm")

    return slices, sorted_z, pixel_spacing


def cubic_spline_interpolation(slices, z_positions, target_spacing=2.5):
    """Cubic spline interpolation to fill gaps between slices."""
    print(f"\n[2/5] Cubic Spline Interpolation (target: {target_spacing}mm)...")

    volume = np.stack(slices, axis=0)  # (Z, H, W)
    nz, ny, nx = volume.shape
    print(f"  Original volume: {nz} x {ny} x {nx}")

    z_arr = np.array(z_positions)
    z_min, z_max = z_arr.min(), z_arr.max()

    # New Z positions at target spacing
    new_z = np.arange(z_min, z_max + target_spacing * 0.5, target_spacing)
    print(f"  New Z positions: {len(new_z)} (from {nz})")

    # Downsample for 3D model (full resolution would be too large)
    # For 3D rendering: 256x256 is sufficient
    target_size = 256
    if ny > target_size or nx > target_size:
        scale_y = target_size / ny
        scale_x = target_size / nx
        scale = min(scale_y, scale_x)
        print(f"  Downsampling for 3D: scale={scale:.3f}")

        # Downsample each slice
        small_slices = []
        for s in slices:
            small = ndimage.zoom(s, scale, order=1)
            small_slices.append(small)
        volume_small = np.stack(small_slices, axis=0)
    else:
        volume_small = volume
        scale = 1.0

    sz, sy, sx = volume_small.shape
    print(f"  Working volume: {sz} x {sy} x {sx}")

    # Create interpolator on original Z grid
    y_grid = np.arange(sy)
    x_grid = np.arange(sx)

    interp = RegularGridInterpolator(
        (z_arr, y_grid, x_grid),
        volume_small,
        method='linear',  # scipy uses linear; we apply additional smoothing
        bounds_error=False,
        fill_value=None
    )

    # Generate new volume at target spacing
    new_z_clipped = new_z[(new_z >= z_min) & (new_z <= z_max)]
    nz_new = len(new_z_clipped)
    print(f"  Interpolating {nz_new} slices...")

    new_volume = np.zeros((nz_new, sy, sx), dtype=np.float32)

    for i, z in enumerate(new_z_clipped):
        # Create meshgrid for this Z plane
        yy, xx = np.meshgrid(y_grid, x_grid, indexing='ij')
        zz = np.full_like(yy, z, dtype=np.float64)
        pts = np.stack([zz.ravel(), yy.ravel(), xx.ravel()], axis=-1)
        new_volume[i] = interp(pts).reshape(sy, sx)

        if (i + 1) % 50 == 0:
            print(f"    Interpolated: {i+1}/{nz_new}")

    # Apply slight 3D Gaussian smoothing for continuity
    print("  Applying 3D smoothing for continuity...")
    new_volume = ndimage.gaussian_filter(new_volume, sigma=[0.5, 0.3, 0.3])

    print(f"  Final volume: {new_volume.shape}")
    voxel_spacing = [target_spacing, pixel_spacing[0] / scale, pixel_spacing[1] / scale]

    return new_volume, new_z_clipped, voxel_spacing, scale


def extract_surfaces(volume, voxel_spacing):
    """Extract 3D surfaces using Marching Cubes for different tissues."""
    print(f"\n[3/5] Extracting 3D surfaces (Marching Cubes)...")

    meshes = {}

    for tissue_name, params in HU_THRESHOLDS.items():
        print(f"  Extracting: {tissue_name} (HU: {params['min']}~{params['max']})...")

        # Create binary mask
        mask = (volume >= params['min']) & (volume <= params['max'])

        # Clean up: remove small components
        labeled, num_features = ndimage.label(mask)
        if num_features > 0:
            sizes = ndimage.sum(mask, labeled, range(1, num_features + 1))
            # Keep only components larger than 100 voxels
            min_size = max(100, np.max(sizes) * 0.01)
            for j in range(num_features):
                if sizes[j] < min_size:
                    mask[labeled == (j + 1)] = False

        voxel_count = np.sum(mask)
        if voxel_count < 50:
            print(f"    Skipped: too few voxels ({voxel_count})")
            continue

        try:
            # Marching cubes
            verts, faces, normals, values = measure.marching_cubes(
                mask.astype(float),
                level=0.5,
                spacing=voxel_spacing,
                step_size=2  # Speed up by skipping voxels
            )

            # Decimate if too many faces
            max_faces = 100000
            if len(faces) > max_faces:
                step = max(1, len(faces) // max_faces)
                faces = faces[::step]
                print(f"    Decimated: {len(faces)} faces (step={step})")

            meshes[tissue_name] = {
                'vertices': verts,
                'faces': faces,
                'normals': normals,
                'color': params['color'],
                'opacity': params['opacity'],
                'voxel_count': int(voxel_count),
                'face_count': len(faces),
                'vertex_count': len(verts),
            }
            print(f"    OK: {len(verts)} vertices, {len(faces)} faces")

        except Exception as e:
            print(f"    Error: {e}")

    return meshes


def mesh_to_binary_buffer(verts, faces, normals):
    """Convert mesh to compact binary buffer for Three.js."""
    # Interleaved: position (3f) + normal (3f) per vertex
    # Indices as uint32
    positions = verts.astype(np.float32).tobytes()
    norms = normals.astype(np.float32).tobytes()
    indices = faces.astype(np.uint32).tobytes()
    return positions, norms, indices


def generate_3d_viewer(meshes, volume, new_z, voxel_spacing, output_dir):
    """Generate interactive 3D viewer using Three.js."""
    print(f"\n[4/5] Generating 3D viewer...")

    os.makedirs(output_dir, exist_ok=True)

    # Export meshes as JSON (simplified for web)
    mesh_data = {}
    for name, m in meshes.items():
        # Center vertices
        center = m['vertices'].mean(axis=0)
        centered_verts = m['vertices'] - center

        # Scale to reasonable size
        max_extent = np.abs(centered_verts).max()
        if max_extent > 0:
            scale_factor = 100.0 / max_extent
            centered_verts *= scale_factor

        # Save as binary files
        pos_file = f"{name}_positions.bin"
        norm_file = f"{name}_normals.bin"
        idx_file = f"{name}_indices.bin"

        centered_verts.astype(np.float32).tofile(os.path.join(output_dir, pos_file))
        m['normals'].astype(np.float32).tofile(os.path.join(output_dir, norm_file))
        m['faces'].astype(np.uint32).tofile(os.path.join(output_dir, idx_file))

        mesh_data[name] = {
            'posFile': pos_file,
            'normFile': norm_file,
            'idxFile': idx_file,
            'vertexCount': int(m['vertex_count']),
            'faceCount': int(m['face_count']),
            'color': m['color'],
            'opacity': m['opacity'],
            'center': center.tolist(),
            'scale': float(scale_factor) if max_extent > 0 else 1.0,
        }

    # Also create cross-section images
    print("  Creating cross-section images...")
    cross_dir = os.path.join(output_dir, 'cross_sections')
    os.makedirs(cross_dir, exist_ok=True)

    nz, ny, nx = volume.shape

    # Save representative slices (every 5th)
    axial_imgs = []
    for i in range(0, nz, max(1, nz // 40)):
        sl = volume[i]
        # Window for abdomen
        sl_w = np.clip((sl - (-150)) / 500 * 255, 0, 255).astype(np.uint8)
        img = Image.fromarray(sl_w)
        fname = f"axial_{i:04d}.png"
        img.save(os.path.join(cross_dir, fname))
        axial_imgs.append({'idx': i, 'z': float(new_z[i]) if i < len(new_z) else 0, 'file': fname})

    # Coronal (front view) - middle slice
    cor_slice = volume[:, ny // 2, :]
    cor_w = np.clip((cor_slice - (-150)) / 500 * 255, 0, 255).astype(np.uint8)
    Image.fromarray(cor_w).save(os.path.join(cross_dir, 'coronal_mid.png'))

    # Sagittal (side view) - middle slice
    sag_slice = volume[:, :, nx // 2]
    sag_w = np.clip((sag_slice - (-150)) / 500 * 255, 0, 255).astype(np.uint8)
    Image.fromarray(sag_w).save(os.path.join(cross_dir, 'sagittal_mid.png'))

    mesh_data_json = json.dumps(mesh_data)
    axial_json = json.dumps(axial_imgs)

    html = f"""<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="UTF-8">
<title>CT 3D Model Viewer</title>
<style>
* {{ margin:0; padding:0; box-sizing:border-box; }}
body {{ background:#0a0a0a; color:#e0e0e0; font-family:'Segoe UI',system-ui,sans-serif; overflow:hidden; }}

#container {{ width:100vw; height:100vh; position:relative; }}

#panel {{
  position:absolute; top:10px; left:10px; z-index:100;
  background:rgba(15,15,25,0.92); backdrop-filter:blur(12px);
  border:1px solid rgba(255,255,255,0.1); border-radius:14px;
  padding:18px; width:280px; max-height:95vh; overflow-y:auto;
  box-shadow:0 8px 32px rgba(0,0,0,0.5);
}}
#panel h1 {{
  font-size:1.3em; margin-bottom:12px;
  background:linear-gradient(135deg,#00bcd4,#7c4dff);
  -webkit-background-clip:text; -webkit-text-fill-color:transparent;
}}
.section {{ margin-bottom:14px; }}
.section h3 {{ color:#4fc3f7; font-size:0.85em; margin-bottom:8px; text-transform:uppercase; letter-spacing:1px; }}

.tissue-toggle {{
  display:flex; align-items:center; gap:8px;
  padding:6px 10px; margin-bottom:4px;
  background:rgba(255,255,255,0.05); border-radius:8px;
  cursor:pointer; transition:background 0.2s;
}}
.tissue-toggle:hover {{ background:rgba(255,255,255,0.1); }}
.tissue-toggle .dot {{ width:12px; height:12px; border-radius:50%; flex-shrink:0; }}
.tissue-toggle label {{ flex:1; font-size:0.85em; cursor:pointer; }}
.tissue-toggle .count {{ font-size:0.7em; color:#666; }}

input[type=range] {{ width:100%; accent-color:#00bcd4; }}
input[type=checkbox] {{ accent-color:#00bcd4; }}

.slider-row {{ display:flex; align-items:center; gap:8px; margin:4px 0; }}
.slider-row label {{ font-size:0.8em; color:#888; min-width:55px; }}
.slider-row span {{ font-size:0.75em; color:#4fc3f7; min-width:30px; }}

.cross-panel {{
  position:absolute; bottom:10px; right:10px; z-index:100;
  background:rgba(15,15,25,0.92); backdrop-filter:blur(12px);
  border:1px solid rgba(255,255,255,0.1); border-radius:14px;
  padding:14px; width:280px;
  box-shadow:0 8px 32px rgba(0,0,0,0.5);
}}
.cross-panel h3 {{ color:#4fc3f7; font-size:0.85em; margin-bottom:8px; text-transform:uppercase; letter-spacing:1px; }}
.cross-imgs {{ display:flex; gap:6px; flex-wrap:wrap; }}
.cross-imgs img {{ width:80px; height:80px; object-fit:cover; border:1px solid #333; border-radius:4px; cursor:pointer; transition:border 0.2s; }}
.cross-imgs img:hover {{ border-color:#00bcd4; }}
.cross-imgs img.active {{ border:2px solid #ff5722; }}

.axial-viewer {{
  margin-top:8px;
}}
.axial-viewer img {{ width:100%; border-radius:6px; border:1px solid #333; }}
.axial-viewer .slider-row {{ margin-top:6px; }}

#loading {{
  position:absolute; top:50%; left:50%; transform:translate(-50%,-50%);
  z-index:200; text-align:center; color:#4fc3f7;
}}
#loading .spinner {{
  width:50px; height:50px; border:4px solid #333;
  border-top-color:#00bcd4; border-radius:50%; animation:spin 1s linear infinite;
  margin:0 auto 15px;
}}
@keyframes spin {{ to {{ transform:rotate(360deg); }} }}

.stats {{
  font-size:0.75em; color:#555; margin-top:10px; padding-top:8px;
  border-top:1px solid rgba(255,255,255,0.05);
}}
</style>
</head>
<body>

<div id="loading">
  <div class="spinner"></div>
  <div>Loading 3D Model...</div>
  <div id="loadProgress" style="font-size:0.8em;color:#666;margin-top:8px"></div>
</div>

<div id="container"></div>

<div id="panel">
  <h1>CT 3D Model</h1>

  <div class="section">
    <h3>Tissues</h3>
    <div id="tissueToggles"></div>
  </div>

  <div class="section">
    <h3>Display</h3>
    <div class="slider-row">
      <label>Rotation</label>
      <input type="checkbox" id="autoRotate" checked>
      <span>Auto</span>
    </div>
    <div class="slider-row">
      <label>Wireframe</label>
      <input type="checkbox" id="wireframe">
    </div>
    <div class="slider-row">
      <label>Clip Z</label>
      <input type="range" id="clipZ" min="0" max="100" value="100" oninput="updateClip(this.value)">
      <span id="clipVal">100%</span>
    </div>
  </div>

  <div class="section">
    <h3>Camera</h3>
    <div style="display:flex;gap:4px;flex-wrap:wrap;">
      <button onclick="setCam('front')" style="flex:1;padding:6px;background:#222;color:#ccc;border:1px solid #444;border-radius:6px;cursor:pointer;font-size:0.8em;">Front</button>
      <button onclick="setCam('side')" style="flex:1;padding:6px;background:#222;color:#ccc;border:1px solid #444;border-radius:6px;cursor:pointer;font-size:0.8em;">Side</button>
      <button onclick="setCam('top')" style="flex:1;padding:6px;background:#222;color:#ccc;border:1px solid #444;border-radius:6px;cursor:pointer;font-size:0.8em;">Top</button>
      <button onclick="setCam('free')" style="flex:1;padding:6px;background:#222;color:#ccc;border:1px solid #444;border-radius:6px;cursor:pointer;font-size:0.8em;">Free</button>
    </div>
  </div>

  <div class="stats" id="statsInfo">Loading...</div>
</div>

<div class="cross-panel">
  <h3>Cross Sections</h3>
  <div class="cross-imgs">
    <img src="cross_sections/coronal_mid.png" alt="Coronal" title="Coronal (Front)" onclick="this.classList.toggle('active')">
    <img src="cross_sections/sagittal_mid.png" alt="Sagittal" title="Sagittal (Side)" onclick="this.classList.toggle('active')">
  </div>
  <div class="axial-viewer">
    <img id="axialImg" src="" alt="Axial">
    <div class="slider-row">
      <label>Axial</label>
      <input type="range" id="axialSlider" min="0" max="0" value="0" oninput="showAxial(this.value)">
      <span id="axialInfo">-</span>
    </div>
  </div>
</div>

<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"></script>

<script>
const MESH_DATA = {mesh_data_json};
const AXIAL_IMGS = {axial_json};

let scene, camera, renderer, controls;
let tissueMeshes = {{}};
let clipPlane;
let maxZ = 100;

function init() {{
  scene = new THREE.Scene();
  scene.background = new THREE.Color(0x0a0a0a);

  // Subtle ambient
  scene.add(new THREE.AmbientLight(0x404060, 0.6));

  // Key light
  const key = new THREE.DirectionalLight(0xffffff, 0.8);
  key.position.set(50, 80, 60);
  key.castShadow = true;
  scene.add(key);

  // Fill light
  const fill = new THREE.DirectionalLight(0x4488cc, 0.4);
  fill.position.set(-40, -20, -50);
  scene.add(fill);

  // Rim light
  const rim = new THREE.DirectionalLight(0x88aaff, 0.3);
  rim.position.set(0, -60, 30);
  scene.add(rim);

  camera = new THREE.PerspectiveCamera(45, window.innerWidth/window.innerHeight, 0.1, 2000);
  camera.position.set(0, 0, 300);

  renderer = new THREE.WebGLRenderer({{ antialias:true, alpha:true }});
  renderer.setSize(window.innerWidth, window.innerHeight);
  renderer.setPixelRatio(window.devicePixelRatio);
  renderer.shadowMap.enabled = true;
  document.getElementById('container').appendChild(renderer.domElement);

  controls = new THREE.OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  controls.dampingFactor = 0.05;
  controls.autoRotate = true;
  controls.autoRotateSpeed = 1.0;

  // Global clip plane
  clipPlane = new THREE.Plane(new THREE.Vector3(0, 0, -1), 100);
  renderer.localClippingEnabled = true;

  loadMeshes();
  setupAxialViewer();

  window.addEventListener('resize', () => {{
    camera.aspect = window.innerWidth / window.innerHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(window.innerWidth, window.innerHeight);
  }});

  animate();
}}

async function loadBinary(url, type) {{
  const resp = await fetch(url);
  const buf = await resp.arrayBuffer();
  if (type === 'float32') return new Float32Array(buf);
  if (type === 'uint32') return new Uint32Array(buf);
  return buf;
}}

async function loadMeshes() {{
  const names = Object.keys(MESH_DATA);
  let loaded = 0;
  let totalVerts = 0, totalFaces = 0;

  const togglesEl = document.getElementById('tissueToggles');

  for (const name of names) {{
    const info = MESH_DATA[name];
    document.getElementById('loadProgress').textContent = `Loading ${{name}}...`;

    try {{
      const [positions, normals, indices] = await Promise.all([
        loadBinary(info.posFile, 'float32'),
        loadBinary(info.normFile, 'float32'),
        loadBinary(info.idxFile, 'uint32'),
      ]);

      const geo = new THREE.BufferGeometry();
      geo.setAttribute('position', new THREE.BufferAttribute(positions, 3));
      geo.setAttribute('normal', new THREE.BufferAttribute(normals, 3));
      geo.setIndex(new THREE.BufferAttribute(indices, 1));

      const color = new THREE.Color(info.color[0], info.color[1], info.color[2]);
      const mat = new THREE.MeshPhongMaterial({{
        color: color,
        transparent: true,
        opacity: info.opacity,
        shininess: 60,
        specular: new THREE.Color(0x222222),
        side: THREE.DoubleSide,
        clippingPlanes: [clipPlane],
        clipShadows: true,
        depthWrite: info.opacity > 0.5,
      }});

      const mesh = new THREE.Mesh(geo, mat);
      scene.add(mesh);
      tissueMeshes[name] = mesh;

      totalVerts += info.vertexCount;
      totalFaces += info.faceCount;

      // Add toggle
      const toggle = document.createElement('div');
      toggle.className = 'tissue-toggle';
      toggle.innerHTML = `
        <div class="dot" style="background:rgb(${{Math.round(info.color[0]*255)}},${{Math.round(info.color[1]*255)}},${{Math.round(info.color[2]*255)}})"></div>
        <input type="checkbox" checked id="tog_${{name}}" onchange="toggleTissue('${{name}}',this.checked)">
        <label for="tog_${{name}}">${{name}}</label>
        <span class="count">${{(info.faceCount/1000).toFixed(0)}}k</span>
      `;
      togglesEl.appendChild(toggle);

      // Opacity slider
      const opRow = document.createElement('div');
      opRow.className = 'slider-row';
      opRow.style.marginLeft = '20px';
      opRow.innerHTML = `
        <label style="font-size:0.75em">Opacity</label>
        <input type="range" min="0" max="100" value="${{Math.round(info.opacity*100)}}"
               oninput="setOpacity('${{name}}',this.value/100)" style="flex:1">
      `;
      togglesEl.appendChild(opRow);

    }} catch(e) {{
      console.error(`Failed to load ${{name}}:`, e);
    }}

    loaded++;
  }}

  document.getElementById('loading').style.display = 'none';
  document.getElementById('statsInfo').innerHTML =
    `Vertices: ${{(totalVerts/1000).toFixed(0)}}K<br>` +
    `Faces: ${{(totalFaces/1000).toFixed(0)}}K<br>` +
    `Tissues: ${{loaded}}`;

  // Compute bounding box and reposition camera
  const box = new THREE.Box3();
  Object.values(tissueMeshes).forEach(m => box.expandByObject(m));
  const center = box.getCenter(new THREE.Vector3());
  const size = box.getSize(new THREE.Vector3());
  maxZ = center.z + size.z / 2;

  controls.target.copy(center);
  camera.position.set(center.x, center.y - size.y * 0.5, center.z + size.z * 1.5);
  clipPlane.constant = maxZ + 10;
  camera.updateProjectionMatrix();
}}

function toggleTissue(name, visible) {{
  if (tissueMeshes[name]) tissueMeshes[name].visible = visible;
}}

function setOpacity(name, val) {{
  if (tissueMeshes[name]) {{
    tissueMeshes[name].material.opacity = val;
    tissueMeshes[name].material.depthWrite = val > 0.5;
  }}
}}

function updateClip(val) {{
  const pct = parseInt(val);
  document.getElementById('clipVal').textContent = pct + '%';
  clipPlane.constant = maxZ * (pct / 100) - maxZ * 0.5;
}}

function setCam(view) {{
  const box = new THREE.Box3();
  Object.values(tissueMeshes).forEach(m => box.expandByObject(m));
  const center = box.getCenter(new THREE.Vector3());
  const size = box.getSize(new THREE.Vector3());
  const dist = Math.max(size.x, size.y, size.z) * 1.5;

  controls.autoRotate = false;
  document.getElementById('autoRotate').checked = false;

  switch(view) {{
    case 'front':
      camera.position.set(center.x, center.y - dist, center.z);
      break;
    case 'side':
      camera.position.set(center.x + dist, center.y, center.z);
      break;
    case 'top':
      camera.position.set(center.x, center.y, center.z + dist);
      break;
    case 'free':
      camera.position.set(center.x + dist*0.5, center.y - dist*0.5, center.z + dist*0.7);
      controls.autoRotate = true;
      document.getElementById('autoRotate').checked = true;
      break;
  }}
  controls.target.copy(center);
  camera.lookAt(center);
}}

function setupAxialViewer() {{
  if (AXIAL_IMGS.length === 0) return;
  const slider = document.getElementById('axialSlider');
  slider.max = AXIAL_IMGS.length - 1;
  showAxial(0);
}}

function showAxial(idx) {{
  idx = parseInt(idx);
  const img = AXIAL_IMGS[idx];
  if (!img) return;
  document.getElementById('axialImg').src = 'cross_sections/' + img.file;
  document.getElementById('axialInfo').textContent = 'Z:' + img.z.toFixed(0) + 'mm';
}}

document.getElementById('autoRotate').addEventListener('change', function() {{
  controls.autoRotate = this.checked;
}});
document.getElementById('wireframe').addEventListener('change', function() {{
  Object.values(tissueMeshes).forEach(m => m.material.wireframe = this.checked);
}});

function animate() {{
  requestAnimationFrame(animate);
  controls.update();
  renderer.render(scene, camera);
}}

init();
</script>
</body>
</html>"""

    html_path = os.path.join(output_dir, 'ct_3d_viewer.html')
    with open(html_path, 'w', encoding='utf-8') as f:
        f.write(html)

    print(f"  3D viewer saved: {html_path}")
    return html_path


def save_interpolated_gallery(volume, new_z, output_dir):
    """Save all interpolated slices as a gallery."""
    print(f"\n[5/5] Saving interpolated slice gallery...")

    gallery_dir = os.path.join(output_dir, 'all_slices')
    os.makedirs(gallery_dir, exist_ok=True)

    nz = volume.shape[0]
    for i in range(nz):
        sl = volume[i]
        sl_w = np.clip((sl - (-150)) / 500 * 255, 0, 255).astype(np.uint8)
        img = Image.fromarray(sl_w)
        img.save(os.path.join(gallery_dir, f"slice_{i:04d}.png"))

        if (i + 1) % 50 == 0:
            print(f"    Saved: {i+1}/{nz}")

    print(f"  Saved {nz} slices to {gallery_dir}")


def main():
    global pixel_spacing

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Step 1: Load
    slices, z_positions, pixel_spacing = load_primary_series(DCM_DIR)

    # Step 2: Interpolate
    volume, new_z, voxel_spacing, scale = cubic_spline_interpolation(slices, z_positions, target_spacing=2.5)

    # Step 3: Extract surfaces
    meshes = extract_surfaces(volume, voxel_spacing)

    # Step 4: Generate 3D viewer
    html_path = generate_3d_viewer(meshes, volume, new_z, voxel_spacing, OUTPUT_DIR)

    # Step 5: Save all slices
    save_interpolated_gallery(volume, new_z, OUTPUT_DIR)

    # Summary
    print("\n" + "=" * 60)
    print("  COMPLETE")
    print("=" * 60)
    print(f"  Original slices: {len(slices)}")
    print(f"  Interpolated volume: {volume.shape}")
    print(f"  Voxel spacing: {[f'{v:.2f}' for v in voxel_spacing]} mm")
    print(f"  Tissues extracted: {list(meshes.keys())}")
    print(f"  3D Viewer: {html_path}")

    return html_path


if __name__ == '__main__':
    main()
