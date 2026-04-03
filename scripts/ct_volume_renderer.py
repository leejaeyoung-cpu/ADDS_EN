"""
CT Volume Renderer - Natural Internal Organ Visualization
==========================================================
1. CT 볼륨을 3D 가우시안 스무딩으로 자연스럽게 처리
2. 3D 텍스처(uint8)로 변환하여 WebGL에서 사용
3. GLSL Ray Marching 셰이더로 내부 장기까지 자연스럽게 렌더링
"""

import os
import sys
import io
import json
import base64
import numpy as np

if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

import pydicom
from PIL import Image
from scipy import ndimage

DCM_DIR = r"f:\ADDS\CTdata\CTdcm"
OUTPUT_DIR = r"f:\ADDS\CTdata\ct_volume_render"

# Volume parameters
TARGET_SIZE = 192  # Balanced: good quality + fast rendering
TARGET_Z_SPACING = 2.5


def load_primary_series():
    """Load Series #604 (Abdomen Artery)."""
    print("[1/4] Loading DICOM series #604...")
    files = [f for f in os.listdir(DCM_DIR) if f.endswith('.dcm')]
    series_slices = {}

    for fname in files:
        fpath = os.path.join(DCM_DIR, fname)
        try:
            ds = pydicom.dcmread(fpath, stop_before_pixels=True)
            if int(getattr(ds, 'SeriesNumber', 0)) == 604:
                z = float(ds.ImagePositionPatient[2])
                if z not in series_slices:
                    series_slices[z] = fpath
        except:
            pass

    sorted_z = sorted(series_slices.keys())
    print(f"  {len(sorted_z)} slices, Z: {sorted_z[0]:.0f} ~ {sorted_z[-1]:.0f} mm")

    slices = []
    for z in sorted_z:
        ds = pydicom.dcmread(series_slices[z])
        slope = float(getattr(ds, 'RescaleSlope', 1.0))
        intercept = float(getattr(ds, 'RescaleIntercept', 0.0))
        hu = ds.pixel_array.astype(np.float32) * slope + intercept
        slices.append(hu)

    pixel_spacing = float(ds.PixelSpacing[0])
    volume = np.stack(slices, axis=0)
    print(f"  Original volume: {volume.shape}, spacing: {pixel_spacing:.3f}mm")
    return volume, sorted_z, pixel_spacing


def prepare_volume(volume, z_positions, pixel_spacing):
    """Interpolate, smooth, and normalize volume for rendering."""
    print("\n[2/4] Preparing volume for rendering...")

    nz, ny, nx = volume.shape
    z_spacing = abs(z_positions[1] - z_positions[0])

    # Step 1: Resample to isotropic voxels at target size
    target_xy = TARGET_SIZE
    target_z = int(TARGET_SIZE * (z_spacing * nz) / (pixel_spacing * max(ny, nx)))
    target_z = min(target_z, TARGET_SIZE)

    zoom_factors = [target_z / nz, target_xy / ny, target_xy / nx]
    print(f"  Resampling {volume.shape} -> ({target_z}, {target_xy}, {target_xy})")
    print(f"  Zoom factors: {[f'{z:.3f}' for z in zoom_factors]}")

    # Cubic spline resampling (order=3) for smooth interpolation
    resampled = ndimage.zoom(volume, zoom_factors, order=3)
    print(f"  Resampled: {resampled.shape}")

    # Step 2: 3D Gaussian smoothing for natural appearance
    print("  Applying 3D Gaussian smoothing (sigma=0.8)...")
    smoothed = ndimage.gaussian_filter(resampled, sigma=0.8)

    # Step 3: Apply bilateral-like filtering (edge-preserving)
    # Approximate: smooth areas with less variation more, preserve edges
    print("  Edge-preserving enhancement...")
    gradient = np.sqrt(
        ndimage.sobel(smoothed, axis=0)**2 +
        ndimage.sobel(smoothed, axis=1)**2 +
        ndimage.sobel(smoothed, axis=2)**2
    )
    # Normalize gradient to [0,1]
    grad_norm = gradient / (gradient.max() + 1e-8)
    # Blend: use more smoothed version where gradients are low (flat areas)
    extra_smooth = ndimage.gaussian_filter(smoothed, sigma=1.5)
    enhanced = smoothed * grad_norm + extra_smooth * (1 - grad_norm)

    # Step 4: Normalize HU to 0-255 uint8
    # Map HU range [-1024, 3000] -> [0, 255]
    hu_min, hu_max = -1024, 3000
    normalized = np.clip((enhanced - hu_min) / (hu_max - hu_min), 0, 1)
    volume_uint8 = (normalized * 255).astype(np.uint8)

    print(f"  Final volume: {volume_uint8.shape}, dtype: {volume_uint8.dtype}")
    print(f"  HU range mapped: [{hu_min}, {hu_max}] -> [0, 255]")

    return volume_uint8, hu_min, hu_max


def generate_volume_renderer(volume_uint8, hu_min, hu_max, output_dir):
    """Generate HTML volume renderer with Ray Marching shaders."""
    print("\n[3/4] Generating Volume Renderer...")

    os.makedirs(output_dir, exist_ok=True)

    nz, ny, nx = volume_uint8.shape

    # Save volume as raw binary
    vol_path = os.path.join(output_dir, 'volume.raw')
    volume_uint8.tofile(vol_path)
    vol_size_mb = os.path.getsize(vol_path) / (1024 * 1024)
    print(f"  Volume saved: {vol_path} ({vol_size_mb:.1f} MB)")

    # Also create preview cross-sections
    preview_dir = os.path.join(output_dir, 'previews')
    os.makedirs(preview_dir, exist_ok=True)

    # Axial previews
    axial_data = []
    for i in range(0, nz, max(1, nz // 30)):
        fname = f'axial_{i:04d}.png'
        Image.fromarray(volume_uint8[i]).save(os.path.join(preview_dir, fname))
        axial_data.append({'i': i, 'f': fname})

    # Coronal & Sagittal mid
    Image.fromarray(volume_uint8[:, ny//2, :]).save(os.path.join(preview_dir, 'coronal.png'))
    Image.fromarray(volume_uint8[:, :, nx//2]).save(os.path.join(preview_dir, 'sagittal.png'))

    axial_json = json.dumps(axial_data)

    html = f"""<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="UTF-8">
<title>CT Volume Renderer</title>
<style>
* {{ margin:0; padding:0; box-sizing:border-box; }}
body {{ background:#f0f0f0; font-family:'Segoe UI',system-ui,sans-serif; overflow:hidden; }}
canvas {{ display:block; }}

#ui {{
  position:absolute; top:10px; left:10px; z-index:10;
  background:rgba(255,255,255,0.96); border-radius:16px;
  padding:20px; width:300px; max-height:95vh; overflow-y:auto;
  border:1px solid #ddd; box-shadow:0 8px 40px rgba(0,0,0,0.12);
}}
#ui h1 {{
  font-size:1.4em; margin-bottom:4px;
  background:linear-gradient(135deg,#e53935,#1e88e5);
  -webkit-background-clip:text; -webkit-text-fill-color:transparent;
}}
#ui .subtitle {{ font-size:0.75em; color:#999; margin-bottom:16px; }}

.section {{ margin-bottom:16px; }}
.section h3 {{
  font-size:0.8em; color:#0097a7; text-transform:uppercase;
  letter-spacing:1px; margin-bottom:10px;
  padding-bottom:4px; border-bottom:1px solid #eee;
}}

.preset-btns {{ display:flex; gap:5px; flex-wrap:wrap; margin-bottom:10px; }}
.preset-btns button {{
  flex:1; min-width:60px; padding:8px 4px; font-size:0.75em;
  border:1px solid #ddd; background:#fff; border-radius:8px;
  cursor:pointer; transition:all 0.2s; color:#555;
}}
.preset-btns button:hover {{ background:#e3f2fd; border-color:#42a5f5; }}
.preset-btns button.active {{ background:#1e88e5; color:#fff; border-color:#1e88e5; }}

.ctrl {{ display:flex; align-items:center; gap:8px; margin:6px 0; }}
.ctrl label {{ font-size:0.8em; color:#666; min-width:70px; }}
.ctrl input[type=range] {{ flex:1; accent-color:#1e88e5; }}
.ctrl span {{ font-size:0.75em; color:#1e88e5; min-width:35px; text-align:right; }}
.ctrl input[type=checkbox] {{ accent-color:#1e88e5; }}

.color-row {{ display:flex; gap:4px; align-items:center; margin:4px 0; }}
.color-row .swatch {{
  width:20px; height:20px; border-radius:4px; border:1px solid #ccc;
}}
.color-row label {{ font-size:0.75em; color:#666; flex:1; }}
.color-row input[type=range] {{ width:60px; accent-color:#1e88e5; }}

#crossPanel {{
  position:absolute; bottom:10px; right:10px; z-index:10;
  background:rgba(255,255,255,0.96); border-radius:16px;
  padding:16px; width:260px; border:1px solid #ddd;
  box-shadow:0 8px 40px rgba(0,0,0,0.12);
}}
#crossPanel h3 {{
  font-size:0.8em; color:#0097a7; text-transform:uppercase;
  letter-spacing:1px; margin-bottom:8px;
}}
.cross-row {{ display:flex; gap:6px; margin-bottom:8px; }}
.cross-row img {{ width:80px; height:80px; object-fit:cover; border-radius:6px; border:1px solid #ddd; }}
#crossPanel .axial-img {{ width:100%; border-radius:8px; border:1px solid #ddd; }}

#loading {{
  position:absolute; top:50%; left:50%; transform:translate(-50%,-50%);
  z-index:20; text-align:center; color:#1e88e5;
}}
.spinner {{
  width:50px; height:50px; border:4px solid #ddd;
  border-top-color:#1e88e5; border-radius:50%;
  animation:spin 0.8s linear infinite; margin:0 auto 12px;
}}
@keyframes spin {{ to {{ transform:rotate(360deg); }} }}
.stats {{ font-size:0.7em; color:#aaa; margin-top:10px; }}
</style>

<!-- Three.js -->
<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"></script>
</head>
<body>

<div id="loading">
  <div class="spinner"></div>
  <div id="loadMsg">Loading Volume Data...</div>
</div>

<div id="ui">
  <h1>CT Volume Renderer</h1>
  <div class="subtitle">{nx}x{ny}x{nz} voxels | Ray Marching</div>

  <div class="section">
    <h3>Presets</h3>
    <div class="preset-btns">
      <button onclick="applyPreset('abdomen')" id="pre_abdomen" class="active">Abdomen</button>
      <button onclick="applyPreset('bone')" id="pre_bone">Bone</button>
      <button onclick="applyPreset('lung')" id="pre_lung">Lung</button>
      <button onclick="applyPreset('vessel')" id="pre_vessel">Vessel</button>
      <button onclick="applyPreset('muscle')" id="pre_muscle">Muscle</button>
      <button onclick="applyPreset('xray')" id="pre_xray">X-Ray</button>
    </div>
  </div>

  <div class="section">
    <h3>Rendering</h3>
    <div class="ctrl">
      <label>Steps</label>
      <input type="range" id="steps" min="64" max="512" value="256" oninput="updateParam()">
      <span id="stepsVal">256</span>
    </div>
    <div class="ctrl">
      <label>Brightness</label>
      <input type="range" id="brightness" min="0" max="200" value="100" oninput="updateParam()">
      <span id="brightnessVal">100</span>
    </div>
    <div class="ctrl">
      <label>Density</label>
      <input type="range" id="density" min="1" max="100" value="30" oninput="updateParam()">
      <span id="densityVal">30</span>
    </div>
    <div class="ctrl">
      <label>Auto Rotate</label>
      <input type="checkbox" id="autoRot" checked onchange="controls.autoRotate=this.checked">
    </div>
  </div>

  <div class="section">
    <h3>Window / Level</h3>
    <div class="ctrl">
      <label>Window Min</label>
      <input type="range" id="wmin" min="0" max="255" value="40" oninput="updateParam()">
      <span id="wminVal">40</span>
    </div>
    <div class="ctrl">
      <label>Window Max</label>
      <input type="range" id="wmax" min="0" max="255" value="200" oninput="updateParam()">
      <span id="wmaxVal">200</span>
    </div>
  </div>

  <div class="section">
    <h3>Clip Plane</h3>
    <div class="ctrl">
      <label>X Clip</label>
      <input type="range" id="clipX" min="0" max="100" value="100" oninput="updateParam()">
      <span id="clipXVal">100</span>
    </div>
    <div class="ctrl">
      <label>Y Clip</label>
      <input type="range" id="clipY" min="0" max="100" value="100" oninput="updateParam()">
      <span id="clipYVal">100</span>
    </div>
    <div class="ctrl">
      <label>Z Clip</label>
      <input type="range" id="clipZ" min="0" max="100" value="100" oninput="updateParam()">
      <span id="clipZVal">100</span>
    </div>
  </div>

  <div class="stats" id="stats">Loading...</div>
</div>

<div id="crossPanel">
  <h3>Cross Sections</h3>
  <div class="cross-row">
    <img src="previews/coronal.png" alt="Coronal" title="Coronal">
    <img src="previews/sagittal.png" alt="Sagittal" title="Sagittal">
  </div>
  <img class="axial-img" id="axialImg" src="" alt="Axial">
  <div class="ctrl" style="margin-top:6px;">
    <label>Axial</label>
    <input type="range" id="axialSlider" min="0" max="0" value="0" oninput="showAxial(this.value)">
    <span id="axialInfo">-</span>
  </div>
</div>

<script>
// ===================== SHADER CODE =====================
const vertexShader = `
varying vec3 vOrigin;
varying vec3 vDirection;
uniform mat4 modelViewMatrix;
uniform mat4 projectionMatrix;
uniform vec3 cameraPos;

attribute vec3 position;

void main() {{
  vec4 mvPos = modelViewMatrix * vec4(position, 1.0);
  vOrigin = vec3(inverse(modelViewMatrix) * vec4(0.0, 0.0, 0.0, 1.0));
  vDirection = position - vOrigin;
  gl_Position = projectionMatrix * mvPos;
}}
`;

const fragmentShader = `
precision highp float;
precision highp sampler3D;

varying vec3 vOrigin;
varying vec3 vDirection;

uniform sampler3D volumeTex;
uniform float uSteps;
uniform float uBrightness;
uniform float uDensity;
uniform float uWinMin;
uniform float uWinMax;
uniform vec3 uClip;
uniform int uPreset;

// Ray-box intersection
vec2 intersectBox(vec3 orig, vec3 dir) {{
  vec3 invDir = 1.0 / dir;
  vec3 tMin = (vec3(0.0) - orig) * invDir;
  vec3 tMax = (vec3(1.0) - orig) * invDir;
  vec3 t1 = min(tMin, tMax);
  vec3 t2 = max(tMin, tMax);
  float tNear = max(max(t1.x, t1.y), t1.z);
  float tFar = min(min(t2.x, t2.y), t2.z);
  return vec2(tNear, tFar);
}}

// Transfer function: map normalized HU to color+opacity
vec4 transferFunction(float val, int preset) {{
  float alpha = 0.0;
  vec3 color = vec3(0.0);

  if (preset == 0) {{ // Abdomen
    if (val < 0.15) {{ // Air / Fat
      alpha = 0.0;
    }} else if (val < 0.22) {{ // Fat
      color = vec3(0.95, 0.85, 0.6);
      alpha = (val - 0.15) * 2.0;
    }} else if (val < 0.27) {{ // Soft tissue
      color = vec3(0.85, 0.55, 0.45);
      alpha = 0.15;
    }} else if (val < 0.32) {{ // Organ parenchyma
      float t = (val - 0.27) / 0.05;
      color = mix(vec3(0.85, 0.45, 0.35), vec3(0.7, 0.25, 0.2), t);
      alpha = 0.25 + t * 0.15;
    }} else if (val < 0.40) {{ // Dense tissue / contrast
      float t = (val - 0.32) / 0.08;
      color = mix(vec3(0.7, 0.25, 0.2), vec3(0.9, 0.3, 0.2), t);
      alpha = 0.35;
    }} else {{ // Bone / calcification
      float t = min((val - 0.40) / 0.3, 1.0);
      color = mix(vec3(0.9, 0.85, 0.75), vec3(1.0, 1.0, 0.95), t);
      alpha = 0.6 + t * 0.4;
    }}
  }} else if (preset == 1) {{ // Bone
    if (val < 0.33) {{
      alpha = 0.0;
    }} else if (val < 0.40) {{
      color = vec3(0.8, 0.7, 0.5);
      alpha = (val - 0.33) * 5.0;
    }} else {{
      float t = min((val - 0.40) / 0.3, 1.0);
      color = mix(vec3(0.9, 0.85, 0.7), vec3(1.0, 0.98, 0.92), t);
      alpha = 0.7 + t * 0.3;
    }}
  }} else if (preset == 2) {{ // Lung
    if (val < 0.05) {{
      alpha = 0.0;
    }} else if (val < 0.18) {{ // Air in lungs
      color = vec3(0.3, 0.5, 0.7);
      alpha = (val - 0.05) * 0.8;
    }} else if (val < 0.25) {{ // Lung tissue
      color = vec3(0.6, 0.4, 0.5);
      alpha = 0.15;
    }} else if (val < 0.35) {{ // Vessels
      color = vec3(0.8, 0.2, 0.15);
      alpha = 0.4;
    }} else {{ // Bone
      color = vec3(0.95, 0.9, 0.8);
      alpha = 0.6;
    }}
  }} else if (preset == 3) {{ // Vessel
    if (val < 0.28) {{
      alpha = 0.0;
    }} else if (val < 0.38) {{ // Enhanced vessels
      float t = (val - 0.28) / 0.10;
      color = mix(vec3(0.9, 0.2, 0.1), vec3(1.0, 0.4, 0.2), t);
      alpha = t * 0.6;
    }} else if (val < 0.50) {{
      color = vec3(1.0, 0.5, 0.3);
      alpha = 0.5;
    }} else {{ // Bone
      color = vec3(0.9, 0.85, 0.75);
      alpha = 0.3;
    }}
  }} else if (preset == 4) {{ // Muscle
    if (val < 0.20) {{
      alpha = 0.0;
    }} else if (val < 0.28) {{ // Muscle
      float t = (val - 0.20) / 0.08;
      color = mix(vec3(0.7, 0.35, 0.3), vec3(0.85, 0.45, 0.35), t);
      alpha = 0.1 + t * 0.2;
    }} else if (val < 0.35) {{
      color = vec3(0.85, 0.5, 0.4);
      alpha = 0.3;
    }} else {{ // Bone
      color = vec3(0.95, 0.9, 0.8);
      alpha = 0.5;
    }}
  }} else {{ // X-Ray (preset 5)
    alpha = val * val * 0.8;
    color = vec3(val);
  }}

  return vec4(color, alpha);
}}

void main() {{
  vec3 rayDir = normalize(vDirection);
  vec2 bounds = intersectBox(vOrigin, rayDir);

  if (bounds.x > bounds.y) discard;

  bounds.x = max(bounds.x, 0.0);
  float stepSize = 1.0 / uSteps;

  vec4 accum = vec4(0.0);
  vec3 pos = vOrigin + bounds.x * rayDir;
  vec3 step = rayDir * (bounds.y - bounds.x) / uSteps;

  for (float i = 0.0; i < 512.0; i += 1.0) {{
    if (i >= uSteps) break;

    // Clip planes
    if (pos.x > uClip.x || pos.y > uClip.y || pos.z > uClip.z) {{
      pos += step;
      continue;
    }}

    // Sample volume
    float val = texture(volumeTex, pos).r;

    // Apply window
    float winVal = clamp((val * 255.0 - uWinMin) / (uWinMax - uWinMin + 0.001), 0.0, 1.0);

    // Transfer function
    vec4 tfColor = transferFunction(winVal, uPreset);

    // Accumulate with front-to-back compositing
    float sampleAlpha = tfColor.a * uDensity * stepSize;
    accum.rgb += (1.0 - accum.a) * tfColor.rgb * sampleAlpha * uBrightness;
    accum.a += (1.0 - accum.a) * sampleAlpha;

    if (accum.a > 0.98) break; // Early termination

    pos += step;
  }}

  // Background blend
  vec3 bg = vec3(0.95);
  gl_FragColor = vec4(mix(bg, accum.rgb, accum.a), 1.0);
}}
`;

// ===================== MAIN APP =====================
const VOL_DIMS = [{nx}, {ny}, {nz}];
const AXIAL_IMGS = {axial_json};

let renderer, scene, camera, controls;
let volumeMesh, volumeMaterial;
let currentPreset = 0;

const PRESETS = {{
  abdomen: {{ wmin: 30,  wmax: 200, density: 30, brightness: 100, steps: 256, preset: 0 }},
  bone:    {{ wmin: 80,  wmax: 230, density: 25, brightness: 120, steps: 200, preset: 1 }},
  lung:    {{ wmin: 10,  wmax: 180, density: 20, brightness: 110, steps: 256, preset: 2 }},
  vessel:  {{ wmin: 60,  wmax: 210, density: 40, brightness: 130, steps: 300, preset: 3 }},
  muscle:  {{ wmin: 40,  wmax: 190, density: 35, brightness: 110, steps: 256, preset: 4 }},
  xray:    {{ wmin: 0,   wmax: 255, density: 15, brightness: 150, steps: 300, preset: 5 }},
}};

async function init() {{
  // Renderer
  renderer = new THREE.WebGLRenderer({{ antialias: true }});
  renderer.setSize(window.innerWidth, window.innerHeight);
  renderer.setPixelRatio(window.devicePixelRatio);
  document.body.appendChild(renderer.domElement);

  // Scene
  scene = new THREE.Scene();
  scene.background = new THREE.Color(0xf0f0f0);

  // Camera
  camera = new THREE.PerspectiveCamera(50, window.innerWidth/window.innerHeight, 0.01, 10);
  camera.position.set(1.5, 1.0, 1.5);

  // Controls
  controls = new THREE.OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  controls.dampingFactor = 0.05;
  controls.autoRotate = true;
  controls.autoRotateSpeed = 1.5;
  controls.target.set(0.5, 0.5, 0.5);

  // Load volume
  document.getElementById('loadMsg').textContent = 'Loading volume data ({vol_size_mb:.1f} MB)...';
  const volData = await loadVolume('volume.raw');

  // Create 3D texture
  document.getElementById('loadMsg').textContent = 'Creating 3D texture...';
  const texture = new THREE.DataTexture3D(volData, VOL_DIMS[0], VOL_DIMS[1], VOL_DIMS[2]);
  texture.format = THREE.RedFormat;
  texture.type = THREE.UnsignedByteType;
  texture.minFilter = THREE.LinearFilter;
  texture.magFilter = THREE.LinearFilter;
  texture.unpackAlignment = 1;
  texture.needsUpdate = true;

  // Create volume mesh (unit cube)
  const geo = new THREE.BoxGeometry(1, 1, 1);
  volumeMaterial = new THREE.RawShaderMaterial({{
    glslVersion: THREE.GLSL3,
    uniforms: {{
      volumeTex: {{ value: texture }},
      uSteps: {{ value: 256.0 }},
      uBrightness: {{ value: 1.0 }},
      uDensity: {{ value: 30.0 }},
      uWinMin: {{ value: 40.0 }},
      uWinMax: {{ value: 200.0 }},
      uClip: {{ value: new THREE.Vector3(1.0, 1.0, 1.0) }},
      uPreset: {{ value: 0 }},
    }},
    vertexShader: `#version 300 es
      in vec3 position;
      uniform mat4 modelViewMatrix;
      uniform mat4 projectionMatrix;
      out vec3 vOrigin;
      out vec3 vDirection;
      void main() {{
        vec4 mvPos = modelViewMatrix * vec4(position, 1.0);
        vOrigin = vec3(inverse(modelViewMatrix) * vec4(0,0,0,1));
        vDirection = position - vOrigin;
        gl_Position = projectionMatrix * mvPos;
      }}`,
    fragmentShader: `#version 300 es
      precision highp float;
      precision highp sampler3D;
      in vec3 vOrigin;
      in vec3 vDirection;
      out vec4 fragColor;
      uniform sampler3D volumeTex;
      uniform float uSteps, uBrightness, uDensity, uWinMin, uWinMax;
      uniform vec3 uClip;
      uniform int uPreset;

      vec2 intersectBox(vec3 o, vec3 d) {{
        vec3 inv = 1.0/d;
        vec3 t1 = min(-o*inv, (vec3(1)-o)*inv);
        vec3 t2 = max(-o*inv, (vec3(1)-o)*inv);
        return vec2(max(max(t1.x,t1.y),t1.z), min(min(t2.x,t2.y),t2.z));
      }}

      vec4 tf(float v, int p) {{
        float a=0.0; vec3 c=vec3(0);
        if (p==0) {{ // Abdomen
          if(v<0.15) a=0.0;
          else if(v<0.22) {{ c=vec3(.95,.85,.6); a=(v-.15)*2.0; }}
          else if(v<0.27) {{ c=vec3(.85,.55,.45); a=0.15; }}
          else if(v<0.32) {{ float t=(v-.27)/.05; c=mix(vec3(.85,.45,.35),vec3(.7,.25,.2),t); a=.25+t*.15; }}
          else if(v<0.40) {{ c=mix(vec3(.7,.25,.2),vec3(.9,.3,.2),(v-.32)/.08); a=.35; }}
          else {{ float t=min((v-.40)/.3,1.0); c=mix(vec3(.9,.85,.75),vec3(1,.95,.9),t); a=.6+t*.4; }}
        }} else if(p==1) {{ // Bone
          if(v<0.33) a=0.0;
          else if(v<0.40) {{ c=vec3(.8,.7,.5); a=(v-.33)*5.0; }}
          else {{ float t=min((v-.40)/.3,1.0); c=mix(vec3(.9,.85,.7),vec3(1,.98,.92),t); a=.7+t*.3; }}
        }} else if(p==2) {{ // Lung
          if(v<0.05) a=0.0;
          else if(v<0.18) {{ c=vec3(.3,.5,.7); a=(v-.05)*.8; }}
          else if(v<0.25) {{ c=vec3(.6,.4,.5); a=.15; }}
          else if(v<0.35) {{ c=vec3(.8,.2,.15); a=.4; }}
          else {{ c=vec3(.95,.9,.8); a=.6; }}
        }} else if(p==3) {{ // Vessel
          if(v<0.28) a=0.0;
          else if(v<0.38) {{ float t=(v-.28)/.10; c=mix(vec3(.9,.2,.1),vec3(1,.4,.2),t); a=t*.6; }}
          else if(v<0.50) {{ c=vec3(1,.5,.3); a=.5; }}
          else {{ c=vec3(.9,.85,.75); a=.3; }}
        }} else if(p==4) {{ // Muscle
          if(v<0.20) a=0.0;
          else if(v<0.28) {{ float t=(v-.20)/.08; c=mix(vec3(.7,.35,.3),vec3(.85,.45,.35),t); a=.1+t*.2; }}
          else if(v<0.35) {{ c=vec3(.85,.5,.4); a=.3; }}
          else {{ c=vec3(.95,.9,.8); a=.5; }}
        }} else {{ // X-Ray
          a=v*v*.8; c=vec3(v);
        }}
        return vec4(c,a);
      }}

      void main() {{
        vec3 rd=normalize(vDirection);
        vec2 b=intersectBox(vOrigin,rd);
        if(b.x>b.y) discard;
        b.x=max(b.x,0.0);
        vec4 acc=vec4(0);
        vec3 p=vOrigin+b.x*rd;
        vec3 s=rd*(b.y-b.x)/uSteps;
        for(float i=0.0;i<512.0;i+=1.0) {{
          if(i>=uSteps) break;
          if(p.x>uClip.x||p.y>uClip.y||p.z>uClip.z) {{ p+=s; continue; }}
          if(p.x<0.0||p.y<0.0||p.z<0.0||p.x>1.0||p.y>1.0||p.z>1.0) {{ p+=s; continue; }}
          float v=texture(volumeTex,p).r;
          float w=clamp((v*255.0-uWinMin)/(uWinMax-uWinMin+.001),0.0,1.0);
          vec4 tc=tf(w,uPreset);
          float sa=tc.a*uDensity*(1.0/uSteps);
          acc.rgb+=(1.0-acc.a)*tc.rgb*sa*uBrightness;
          acc.a+=(1.0-acc.a)*sa;
          if(acc.a>0.98) break;
          p+=s;
        }}
        vec3 bg=vec3(0.95);
        fragColor=vec4(mix(bg,acc.rgb,acc.a),1.0);
      }}`,
    side: THREE.BackSide,
    transparent: false,
  }});

  volumeMesh = new THREE.Mesh(geo, volumeMaterial);
  scene.add(volumeMesh);

  // Hide loading
  document.getElementById('loading').style.display = 'none';
  document.getElementById('stats').innerHTML =
    `Volume: ${{VOL_DIMS[0]}}x${{VOL_DIMS[1]}}x${{VOL_DIMS[2]}}<br>` +
    `Size: {vol_size_mb:.1f} MB<br>Ray Marching GPU`;

  // Setup axial
  const axSlider = document.getElementById('axialSlider');
  axSlider.max = AXIAL_IMGS.length - 1;
  showAxial(0);

  // Start
  window.addEventListener('resize', onResize);
  animate();
}}

async function loadVolume(url) {{
  const resp = await fetch(url);
  const total = parseInt(resp.headers.get('content-length') || '0');
  const reader = resp.body.getReader();
  const chunks = [];
  let received = 0;

  while (true) {{
    const {{ done, value }} = await reader.read();
    if (done) break;
    chunks.push(value);
    received += value.length;
    if (total > 0) {{
      const pct = Math.round(received / total * 100);
      document.getElementById('loadMsg').textContent = `Loading... ${{pct}}%`;
    }}
  }}

  const data = new Uint8Array(received);
  let offset = 0;
  for (const chunk of chunks) {{
    data.set(chunk, offset);
    offset += chunk.length;
  }}
  return data;
}}

function updateParam() {{
  if (!volumeMaterial) return;
  const u = volumeMaterial.uniforms;
  const steps = parseInt(document.getElementById('steps').value);
  const brightness = parseInt(document.getElementById('brightness').value);
  const density = parseInt(document.getElementById('density').value);
  const wmin = parseInt(document.getElementById('wmin').value);
  const wmax = parseInt(document.getElementById('wmax').value);
  const cx = parseInt(document.getElementById('clipX').value) / 100;
  const cy = parseInt(document.getElementById('clipY').value) / 100;
  const cz = parseInt(document.getElementById('clipZ').value) / 100;

  u.uSteps.value = steps;
  u.uBrightness.value = brightness / 100;
  u.uDensity.value = density;
  u.uWinMin.value = wmin;
  u.uWinMax.value = wmax;
  u.uClip.value.set(cx, cy, cz);

  document.getElementById('stepsVal').textContent = steps;
  document.getElementById('brightnessVal').textContent = brightness;
  document.getElementById('densityVal').textContent = density;
  document.getElementById('wminVal').textContent = wmin;
  document.getElementById('wmaxVal').textContent = wmax;
  document.getElementById('clipXVal').textContent = Math.round(cx * 100);
  document.getElementById('clipYVal').textContent = Math.round(cy * 100);
  document.getElementById('clipZVal').textContent = Math.round(cz * 100);
}}

function applyPreset(name) {{
  const p = PRESETS[name];
  if (!p) return;

  document.getElementById('steps').value = p.steps;
  document.getElementById('brightness').value = p.brightness;
  document.getElementById('density').value = p.density;
  document.getElementById('wmin').value = p.wmin;
  document.getElementById('wmax').value = p.wmax;

  if (volumeMaterial) {{
    volumeMaterial.uniforms.uPreset.value = p.preset;
  }}

  document.querySelectorAll('.preset-btns button').forEach(b => b.classList.remove('active'));
  const btn = document.getElementById('pre_' + name);
  if (btn) btn.classList.add('active');

  updateParam();
}}

function showAxial(idx) {{
  idx = parseInt(idx);
  if (!AXIAL_IMGS[idx]) return;
  document.getElementById('axialImg').src = 'previews/' + AXIAL_IMGS[idx].f;
  document.getElementById('axialInfo').textContent = '#' + AXIAL_IMGS[idx].i;
}}

function onResize() {{
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight);
}}

function animate() {{
  requestAnimationFrame(animate);
  controls.update();
  renderer.render(scene, camera);
}}

init();
</script>
</body>
</html>"""

    html_path = os.path.join(output_dir, 'volume_viewer.html')
    with open(html_path, 'w', encoding='utf-8') as f:
        f.write(html)

    print(f"  Volume viewer saved: {html_path}")
    return html_path


def main():
    # Step 1: Load
    volume, z_positions, pixel_spacing = load_primary_series()

    # Step 2: Prepare
    volume_uint8, hu_min, hu_max = prepare_volume(volume, z_positions, pixel_spacing)

    # Step 3: Generate viewer
    html_path = generate_volume_renderer(volume_uint8, hu_min, hu_max, OUTPUT_DIR)

    print("\n" + "=" * 60)
    print("  COMPLETE")
    print("=" * 60)
    print(f"  Volume: {volume_uint8.shape}")
    print(f"  Viewer: {html_path}")
    print(f"  Run: cd {OUTPUT_DIR} && python -m http.server 8090")
    print(f"  Open: http://localhost:8090/volume_viewer.html")

    return html_path


if __name__ == '__main__':
    main()
