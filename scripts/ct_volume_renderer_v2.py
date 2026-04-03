"""
CT Volume Renderer v2 - Inline Volume Data
============================================
볼륨 데이터를 base64로 HTML에 직접 내장하여
fetch/CORS 문제 없이 안정적으로 렌더링.
128^3 볼륨 = ~2MB base64.
"""

import os, sys, io, json, base64, gzip
import numpy as np

if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

import pydicom
from scipy import ndimage

DCM_DIR = r"f:\ADDS\CTdata\CTdcm"
OUTPUT_DIR = r"f:\ADDS\CTdata\ct_volume_render"
VOL_SIZE = 128  # 128^3 = 2MB, fast and reliable


def load_and_prepare():
    """Load, interpolate, smooth, normalize."""
    print("[1/3] Loading Series #604...")
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
    print(f"  {len(sorted_z)} slices")

    hu_slices = []
    for z in sorted_z:
        ds = pydicom.dcmread(slices_map[z])
        slope = float(getattr(ds, 'RescaleSlope', 1.0))
        intercept = float(getattr(ds, 'RescaleIntercept', 0.0))
        hu_slices.append(ds.pixel_array.astype(np.float32) * slope + intercept)

    volume = np.stack(hu_slices, axis=0)
    print(f"  Original: {volume.shape}")

    # Resample to isotropic VOL_SIZE^3
    print(f"[2/3] Resampling to {VOL_SIZE}^3 with cubic spline...")
    zoom = [VOL_SIZE / s for s in volume.shape]
    resampled = ndimage.zoom(volume, zoom, order=3)

    # Smooth
    print("  3D Gaussian smoothing...")
    smoothed = ndimage.gaussian_filter(resampled, sigma=0.7)

    # Normalize: focus on soft tissue range
    # [-150, 400] captures: fat(-100), water(0), tissue(20-80), organ(40-80), contrast(100-300), bone(300+)
    hu_min, hu_max = -150, 400
    norm = np.clip((smoothed - hu_min) / (hu_max - hu_min), 0, 1)
    vol_u8 = (norm * 255).astype(np.uint8)

    print(f"  Final: {vol_u8.shape}, HU [{hu_min},{hu_max}] -> [0,255]")
    return vol_u8


def generate_html(vol_u8, output_dir):
    """Generate self-contained HTML with inline volume data."""
    print("[3/3] Generating HTML...")

    os.makedirs(output_dir, exist_ok=True)
    nz, ny, nx = vol_u8.shape

    # Transpose to Fortran order (X varies fastest for WebGL)
    # WebGL 3D texture expects: width=X(fastest), height=Y, depth=Z(slowest)
    vol_fortran = np.ascontiguousarray(vol_u8.transpose(2, 1, 0))

    # Compress + base64
    compressed = gzip.compress(vol_fortran.tobytes(), compresslevel=6)
    b64 = base64.b64encode(compressed).decode('ascii')
    print(f"  Volume: {len(vol_fortran.tobytes())/1024/1024:.1f}MB -> compressed: {len(compressed)/1024/1024:.1f}MB -> base64: {len(b64)/1024/1024:.1f}MB")

    # Also save as raw for the server-based viewer
    vol_fortran.tofile(os.path.join(output_dir, 'volume_v2.raw'))

    html = f"""<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="UTF-8">
<title>CT Volume Renderer</title>
<style>
*{{margin:0;padding:0;box-sizing:border-box}}
body{{background:#e8e8e8;font-family:'Segoe UI',system-ui,sans-serif;overflow:hidden}}
canvas{{display:block;width:100vw;height:100vh}}
#ui{{position:fixed;top:12px;left:12px;z-index:10;background:rgba(255,255,255,0.97);
  border-radius:16px;padding:20px;width:290px;max-height:95vh;overflow-y:auto;
  border:1px solid #ddd;box-shadow:0 6px 30px rgba(0,0,0,0.12)}}
#ui h1{{font-size:1.3em;margin-bottom:2px;
  background:linear-gradient(135deg,#e53935,#1e88e5);
  -webkit-background-clip:text;-webkit-text-fill-color:transparent}}
.sub{{font-size:.72em;color:#999;margin-bottom:14px}}
.sec{{margin-bottom:14px}}
.sec h3{{font-size:.78em;color:#0097a7;text-transform:uppercase;letter-spacing:1px;margin-bottom:8px;
  padding-bottom:3px;border-bottom:1px solid #eee}}
.pbtn{{display:flex;gap:4px;flex-wrap:wrap;margin-bottom:8px}}
.pbtn button{{flex:1;min-width:55px;padding:7px 3px;font-size:.72em;border:1px solid #ddd;
  background:#fff;border-radius:8px;cursor:pointer;transition:all .15s;color:#555}}
.pbtn button:hover{{background:#e3f2fd;border-color:#42a5f5}}
.pbtn button.on{{background:#1e88e5;color:#fff;border-color:#1e88e5}}
.r{{display:flex;align-items:center;gap:6px;margin:5px 0}}
.r label{{font-size:.78em;color:#666;min-width:65px}}
.r input[type=range]{{flex:1;accent-color:#1e88e5}}
.r span{{font-size:.72em;color:#1e88e5;min-width:30px;text-align:right}}
.r input[type=checkbox]{{accent-color:#1e88e5}}
#info{{font-size:.68em;color:#aaa;margin-top:8px}}
#loading{{position:fixed;top:50%;left:50%;transform:translate(-50%,-50%);z-index:20;
  text-align:center;color:#1e88e5;background:rgba(255,255,255,0.95);padding:40px;border-radius:16px;
  box-shadow:0 8px 40px rgba(0,0,0,0.15)}}
.sp{{width:50px;height:50px;border:4px solid #ddd;border-top-color:#1e88e5;border-radius:50%;
  animation:sp .8s linear infinite;margin:0 auto 12px}}
@keyframes sp{{to{{transform:rotate(360deg)}}}}
</style>
</head>
<body>

<div id="loading">
  <div class="sp"></div>
  <div id="lmsg">Decompressing volume data...</div>
</div>

<div id="ui">
  <h1>CT Volume Renderer</h1>
  <div class="sub">{nx}x{ny}x{nz} | GPU Ray Marching</div>

  <div class="sec">
    <h3>Preset</h3>
    <div class="pbtn" id="presetBtns">
      <button class="on" onclick="preset('abd')">Abdomen</button>
      <button onclick="preset('bone')">Bone</button>
      <button onclick="preset('lung')">Lung</button>
      <button onclick="preset('vessel')">Vessel</button>
      <button onclick="preset('xray')">X-Ray</button>
    </div>
  </div>

  <div class="sec">
    <h3>Rendering</h3>
    <div class="r"><label>Quality</label><input type="range" id="sSteps" min="50" max="400" value="200" oninput="upd()"><span id="vSteps">200</span></div>
    <div class="r"><label>Brightness</label><input type="range" id="sBright" min="20" max="300" value="150" oninput="upd()"><span id="vBright">150</span></div>
    <div class="r"><label>Density</label><input type="range" id="sDens" min="1" max="80" value="30" oninput="upd()"><span id="vDens">30</span></div>
  </div>

  <div class="sec">
    <h3>Window</h3>
    <div class="r"><label>Min</label><input type="range" id="sWMin" min="0" max="255" value="0" oninput="upd()"><span id="vWMin">0</span></div>
    <div class="r"><label>Max</label><input type="range" id="sWMax" min="0" max="255" value="255" oninput="upd()"><span id="vWMax">255</span></div>
  </div>

  <div class="sec">
    <h3>Clip (Slice View)</h3>
    <div class="r"><label>X</label><input type="range" id="sCX" min="0" max="100" value="100" oninput="upd()"><span id="vCX">100</span></div>
    <div class="r"><label>Y</label><input type="range" id="sCY" min="0" max="100" value="100" oninput="upd()"><span id="vCY">100</span></div>
    <div class="r"><label>Z</label><input type="range" id="sCZ" min="0" max="100" value="100" oninput="upd()"><span id="vCZ">100</span></div>
  </div>

  <div class="sec">
    <h3>View</h3>
    <div class="r"><label>Auto Rotate</label><input type="checkbox" id="cbRot" checked onchange="ctl.autoRotate=this.checked"></div>
    <div class="pbtn">
      <button onclick="cam('F')">Front</button>
      <button onclick="cam('S')">Side</button>
      <button onclick="cam('T')">Top</button>
    </div>
  </div>

  <div id="info">Loading...</div>
</div>

<script type="importmap">
{{
  "imports": {{
    "three": "https://cdn.jsdelivr.net/npm/three@0.160.0/build/three.module.js",
    "three/addons/": "https://cdn.jsdelivr.net/npm/three@0.160.0/examples/jsm/"
  }}
}}
</script>

<script type="module">
import * as THREE from 'three';
import {{ OrbitControls }} from 'three/addons/controls/OrbitControls.js';

const NX={nx}, NY={ny}, NZ={nz};
const B64 = "{b64}";

let renderer, scene, camera, ctl, volMat;

// Decompress volume
async function decompress(b64str) {{
  const bin = Uint8Array.from(atob(b64str), c => c.charCodeAt(0));
  const ds = new DecompressionStream('gzip');
  const writer = ds.writable.getWriter();
  writer.write(bin);
  writer.close();
  const reader = ds.readable.getReader();
  const chunks = [];
  while(true) {{
    const {{done, value}} = await reader.read();
    if(done) break;
    chunks.push(value);
  }}
  let total = 0;
  chunks.forEach(c => total += c.length);
  const result = new Uint8Array(total);
  let off = 0;
  chunks.forEach(c => {{ result.set(c, off); off += c.length; }});
  return result;
}}

async function init() {{
  // Decompress volume
  document.getElementById('lmsg').textContent = 'Decompressing ({len(compressed)//1024}KB)...';
  const volData = await decompress(B64);
  document.getElementById('lmsg').textContent = 'Creating 3D texture...';

  // Renderer
  renderer = new THREE.WebGLRenderer({{ antialias: true }});
  renderer.setSize(window.innerWidth, window.innerHeight);
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  document.body.appendChild(renderer.domElement);

  scene = new THREE.Scene();
  scene.background = new THREE.Color(0xe8e8e8);

  camera = new THREE.PerspectiveCamera(50, window.innerWidth/window.innerHeight, 0.01, 20);
  camera.position.set(0, 0, 2.5);

  ctl = new OrbitControls(camera, renderer.domElement);
  ctl.enableDamping = true;
  ctl.dampingFactor = 0.05;
  ctl.autoRotate = true;
  ctl.autoRotateSpeed = 1.5;
  ctl.target.set(0, 0, 0);

  // 3D texture
  const tex = new THREE.Data3DTexture(volData, NX, NY, NZ);
  tex.format = THREE.RedFormat;
  tex.type = THREE.UnsignedByteType;
  tex.minFilter = THREE.LinearFilter;
  tex.magFilter = THREE.LinearFilter;
  tex.wrapS = THREE.ClampToEdgeWrapping;
  tex.wrapT = THREE.ClampToEdgeWrapping;
  tex.wrapR = THREE.ClampToEdgeWrapping;
  tex.needsUpdate = true;

  // Volume material with ray marching
  volMat = new THREE.RawShaderMaterial({{
    glslVersion: THREE.GLSL3,
    uniforms: {{
      uVolume: {{ value: tex }},
      uSteps:  {{ value: 200.0 }},
      uBright: {{ value: 1.5 }},
      uDens:   {{ value: 30.0 }},
      uWMin:   {{ value: 0.0 }},
      uWMax:   {{ value: 255.0 }},
      uClip:   {{ value: new THREE.Vector3(1,1,1) }},
      uMode:   {{ value: 0 }},
    }},
    vertexShader: `
      in vec3 position;
      uniform mat4 modelViewMatrix;
      uniform mat4 projectionMatrix;
      out vec3 vPos;
      out vec3 vCamPos;
      void main() {{
        vPos = position + 0.5; // [0,1]
        vCamPos = (inverse(modelViewMatrix) * vec4(0,0,0,1)).xyz + 0.5;
        gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
      }}
    `,
    fragmentShader: `
      precision highp float;
      precision highp sampler3D;
      in vec3 vPos;
      in vec3 vCamPos;
      out vec4 outColor;

      uniform sampler3D uVolume;
      uniform float uSteps, uBright, uDens, uWMin, uWMax;
      uniform vec3 uClip;
      uniform int uMode;

      vec2 hitBox(vec3 o, vec3 d) {{
        vec3 tMin = -o / d;
        vec3 tMax = (vec3(1.0) - o) / d;
        vec3 t1 = min(tMin, tMax);
        vec3 t2 = max(tMin, tMax);
        float near = max(max(t1.x,t1.y),t1.z);
        float far  = min(min(t2.x,t2.y),t2.z);
        return vec2(near, far);
      }}

      // HU range [-150,400] mapped to [0,1]. Key values:
      // fat(-100)=0.09, water(0)=0.27, tissue(40)=0.35, organ(60)=0.38
      // contrast(200)=0.64, bone(300)=0.82, dense_bone(400)=1.0
      vec4 colorMap(float v) {{
        vec3 c = vec3(0.0); float a = 0.0;
        if (uMode == 0) {{ // Abdomen - show everything
          if      (v < 0.05) {{ a = 0.0; }} // air
          else if (v < 0.15) {{ c = vec3(0.95,0.88,0.55); a = (v-0.05)*1.0; }} // fat
          else if (v < 0.30) {{ c = vec3(0.92,0.72,0.55); a = 0.08 + (v-0.15)*0.6; }} // water/fluid
          else if (v < 0.42) {{ float t=(v-0.30)/0.12; c = mix(vec3(0.88,0.50,0.40),vec3(0.78,0.30,0.25),t); a = 0.18+t*0.15; }} // soft tissue/muscle
          else if (v < 0.55) {{ float t=(v-0.42)/0.13; c = mix(vec3(0.75,0.28,0.22),vec3(0.85,0.25,0.20),t); a = 0.30+t*0.15; }} // organs
          else if (v < 0.72) {{ float t=(v-0.55)/0.17; c = mix(vec3(0.90,0.30,0.20),vec3(0.95,0.40,0.25),t); a = 0.40+t*0.15; }} // contrast vessels
          else               {{ float t=min((v-0.72)/0.28,1.0); c = mix(vec3(0.92,0.88,0.78),vec3(1,0.97,0.92),t); a = 0.65+t*0.35; }} // bone
        }} else if (uMode == 1) {{ // Bone only
          if (v < 0.70)      {{ a = 0.0; }}
          else if (v < 0.80) {{ c = vec3(0.85,0.75,0.55); a = (v-0.70)*8.0; }}
          else               {{ float t=min((v-0.80)/0.2,1.0); c = mix(vec3(0.92,0.88,0.75),vec3(1,0.98,0.93),t); a = 0.8+t*0.2; }}
        }} else if (uMode == 2) {{ // Lung
          if      (v < 0.03) {{ a = 0.0; }}
          else if (v < 0.20) {{ c = vec3(0.25,0.50,0.75); a = (v-0.03)*1.5; }} // air in lungs
          else if (v < 0.35) {{ c = vec3(0.60,0.40,0.50); a = 0.15; }} // lung tissue
          else if (v < 0.55) {{ c = vec3(0.85,0.25,0.18); a = 0.35; }} // vessels
          else               {{ c = vec3(0.95,0.90,0.80); a = 0.55; }} // bone
        }} else if (uMode == 3) {{ // Vessel
          if (v < 0.40)      {{ a = 0.0; }}
          else if (v < 0.65) {{ float t=(v-0.40)/0.25; c = mix(vec3(0.90,0.15,0.10),vec3(1.0,0.45,0.20),t); a = t*0.6; }}
          else               {{ c = vec3(1.0,0.50,0.30); a = 0.55; }}
        }} else {{ // X-Ray
          a = v * v * 1.5;
          c = vec3(v);
        }}
        return vec4(c, a);
      }}

      void main() {{
        vec3 rayDir = normalize(vPos - vCamPos);
        vec2 bounds = hitBox(vCamPos, rayDir);
        if (bounds.x > bounds.y) discard;
        bounds.x = max(bounds.x, 0.001);

        float dt = (bounds.y - bounds.x) / uSteps;
        vec3 p = vCamPos + bounds.x * rayDir;
        vec3 step = rayDir * dt;

        vec4 acc = vec4(0.0);

        for (float i = 0.0; i < 400.0; i += 1.0) {{
          if (i >= uSteps) break;

          // Clip
          if (p.x < 0.0 || p.y < 0.0 || p.z < 0.0 ||
              p.x > uClip.x || p.y > uClip.y || p.z > uClip.z) {{
            p += step;
            continue;
          }}

          float raw = texture(uVolume, p).r; // [0,1]

          // Window
          float v = clamp((raw * 255.0 - uWMin) / max(uWMax - uWMin, 1.0), 0.0, 1.0);

          vec4 col = colorMap(v);
          float sa = col.a * uDens * dt * 2.0;

          acc.rgb += (1.0 - acc.a) * col.rgb * sa * uBright;
          acc.a   += (1.0 - acc.a) * sa;

          if (acc.a > 0.95) break;
          p += step;
        }}

        vec3 bg = vec3(0.91);
        outColor = vec4(mix(bg, acc.rgb, min(acc.a, 1.0)), 1.0);
      }}
    `,
    side: THREE.BackSide,
    transparent: false,
  }});

  const box = new THREE.BoxGeometry(1, 1, 1);
  const mesh = new THREE.Mesh(box, volMat);
  scene.add(mesh);

  document.getElementById('loading').style.display = 'none';
  document.getElementById('info').innerHTML =
    'Volume: '+NX+'x'+NY+'x'+NZ+'<br>GPU Ray Marching<br>Rendering OK';

  window.addEventListener('resize', () => {{
    camera.aspect = window.innerWidth / window.innerHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(window.innerWidth, window.innerHeight);
  }});

  (function loop() {{
    requestAnimationFrame(loop);
    ctl.update();
    renderer.render(scene, camera);
  }})();
}}

// UI functions - exposed globally
window.upd = function() {{
  if (!volMat) return;
  const u = volMat.uniforms;
  const g = id => {{ const el=document.getElementById(id); return parseInt(el.value); }};
  const s = (id,v) => document.getElementById(id).textContent = v;

  u.uSteps.value = g('sSteps'); s('vSteps', g('sSteps'));
  u.uBright.value = g('sBright')/100; s('vBright', g('sBright'));
  u.uDens.value = g('sDens'); s('vDens', g('sDens'));
  u.uWMin.value = g('sWMin'); s('vWMin', g('sWMin'));
  u.uWMax.value = g('sWMax'); s('vWMax', g('sWMax'));
  u.uClip.value.set(g('sCX')/100, g('sCY')/100, g('sCZ')/100);
  s('vCX', g('sCX')); s('vCY', g('sCY')); s('vCZ', g('sCZ'));
}};

window.preset = function(name) {{
  const presets = {{
    abd:    {{ steps:200, bright:150, dens:30, wmin:0,  wmax:255, mode:0 }},
    bone:   {{ steps:180, bright:140, dens:35, wmin:0,  wmax:255, mode:1 }},
    lung:   {{ steps:220, bright:120, dens:25, wmin:0,  wmax:255, mode:2 }},
    vessel: {{ steps:250, bright:160, dens:40, wmin:0,  wmax:255, mode:3 }},
    xray:   {{ steps:300, bright:200, dens:15, wmin:0,  wmax:255, mode:4 }},
  }};
  const p = presets[name]; if (!p) return;
  document.getElementById('sSteps').value = p.steps;
  document.getElementById('sBright').value = p.bright;
  document.getElementById('sDens').value = p.dens;
  document.getElementById('sWMin').value = p.wmin;
  document.getElementById('sWMax').value = p.wmax;
  if (volMat) volMat.uniforms.uMode.value = p.mode;
  document.querySelectorAll('.pbtn button').forEach(b => b.classList.remove('on'));
  event.target.classList.add('on');
  upd();
}};

window.cam = function(v) {{
  ctl.autoRotate = false;
  document.getElementById('cbRot').checked = false;
  const d = 2.5;
  if (v==='F') camera.position.set(0, -d, 0);
  if (v==='S') camera.position.set(d, 0, 0);
  if (v==='T') camera.position.set(0, 0, d);
  camera.lookAt(0,0,0);
}};

init().catch(e => {{
  document.getElementById('lmsg').textContent = 'Error: ' + e.message;
  console.error(e);
}});
</script>
</body>
</html>"""

    html_path = os.path.join(output_dir, 'volume_viewer_v2.html')
    with open(html_path, 'w', encoding='utf-8') as f:
        f.write(html)

    size_mb = os.path.getsize(html_path) / (1024*1024)
    print(f"  HTML saved: {html_path} ({size_mb:.1f} MB)")
    return html_path


def main():
    vol = load_and_prepare()
    html = generate_html(vol, OUTPUT_DIR)
    print(f"\n  DONE! Open directly in browser (no server needed):")
    print(f"  {html}")


if __name__ == '__main__':
    main()
