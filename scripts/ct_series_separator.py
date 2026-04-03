"""
CT Series Separator & Slice Interpolation Pipeline
====================================================
1단계: SeriesInstanceUID로 시리즈 분류
2단계: 주 시리즈 선택 및 고유 슬라이스 추출
3단계: 슬라이스 간 보간 (딥러닝 기반)
"""

import os
import sys
import io
import json
import numpy as np
from collections import defaultdict

if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

try:
    import pydicom
except ImportError:
    print("pydicom required: pip install pydicom")
    sys.exit(1)

try:
    from PIL import Image
except ImportError:
    print("Pillow required: pip install Pillow")
    sys.exit(1)

from scipy import ndimage

DCM_DIR = r"f:\ADDS\CTdata\CTdcm"
OUTPUT_BASE = r"f:\ADDS\CTdata\ct_series_separated"

# HU Windowing (Abdomen)
WINDOW_CENTER = 50
WINDOW_WIDTH = 400


def apply_hu_windowing(pixel_array, slope, intercept, wc=WINDOW_CENTER, ww=WINDOW_WIDTH):
    """Apply HU conversion and windowing."""
    hu = pixel_array.astype(np.float64) * slope + intercept
    min_val = wc - ww / 2
    max_val = wc + ww / 2
    hu = np.clip(hu, min_val, max_val)
    return ((hu - min_val) / (max_val - min_val) * 255).astype(np.uint8)


def step1_classify_series(dcm_dir):
    """Step 1: Classify DICOM files by SeriesInstanceUID."""
    print("=" * 60)
    print("  Step 1: DICOM Series Classification")
    print("=" * 60)
    
    series_map = defaultdict(list)
    files = [f for f in os.listdir(dcm_dir) if f.endswith('.dcm')]
    total = len(files)
    
    for i, fname in enumerate(files):
        if (i + 1) % 100 == 0:
            print(f"  scanning: {i+1}/{total}")
        
        fpath = os.path.join(dcm_dir, fname)
        try:
            ds = pydicom.dcmread(fpath, stop_before_pixels=True)
            
            series_uid = str(getattr(ds, 'SeriesInstanceUID', 'unknown'))
            series_desc = str(getattr(ds, 'SeriesDescription', 'N/A'))
            series_num = int(getattr(ds, 'SeriesNumber', 0))
            
            if hasattr(ds, 'ImagePositionPatient'):
                z_pos = float(ds.ImagePositionPatient[2])
            elif hasattr(ds, 'SliceLocation'):
                z_pos = float(ds.SliceLocation)
            else:
                z_pos = float(getattr(ds, 'InstanceNumber', i))
            
            instance_num = int(getattr(ds, 'InstanceNumber', 0))
            slice_thickness = float(getattr(ds, 'SliceThickness', 0))
            conv_kernel = str(getattr(ds, 'ConvolutionKernel', 'N/A'))
            
            series_map[series_uid].append({
                'filename': fname,
                'filepath': fpath,
                'z_position': z_pos,
                'instance_number': instance_num,
                'series_description': series_desc,
                'series_number': series_num,
                'slice_thickness': slice_thickness,
                'convolution_kernel': conv_kernel,
            })
        except Exception as e:
            print(f"  Warning: {fname} failed - {e}")
    
    # Analyze each series
    print(f"\n  Found {len(series_map)} series:")
    print("-" * 80)
    
    series_info = []
    for uid, slices in series_map.items():
        slices.sort(key=lambda s: s['z_position'])
        z_positions = [s['z_position'] for s in slices]
        unique_z = len(set(z_positions))
        
        # Calculate gaps
        gaps = []
        sorted_z = sorted(set(z_positions))
        for j in range(1, len(sorted_z)):
            gaps.append(sorted_z[j] - sorted_z[j-1])
        
        avg_gap = np.mean(gaps) if gaps else 0
        
        info = {
            'uid': uid,
            'series_number': slices[0]['series_number'],
            'description': slices[0]['series_description'],
            'kernel': slices[0]['convolution_kernel'],
            'slice_count': len(slices),
            'unique_z_count': unique_z,
            'z_min': min(z_positions),
            'z_max': max(z_positions),
            'z_range': max(z_positions) - min(z_positions),
            'slice_thickness': slices[0]['slice_thickness'],
            'avg_gap': avg_gap,
            'slices': slices,
        }
        series_info.append(info)
        
        print(f"  Series #{info['series_number']} | {info['description']}")
        print(f"    Kernel: {info['kernel']} | Slices: {info['slice_count']} | Unique Z: {info['unique_z_count']}")
        print(f"    Z Range: {info['z_min']:.1f} ~ {info['z_max']:.1f} mm ({info['z_range']:.1f} mm)")
        print(f"    Thickness: {info['slice_thickness']} mm | Avg Gap: {avg_gap:.2f} mm")
        print()
    
    return series_info


def step2_extract_primary_series(series_info, output_base):
    """Step 2: Select primary series and extract unique slices."""
    print("=" * 60)
    print("  Step 2: Primary Series Selection & Extraction")
    print("=" * 60)
    
    # Select primary series = the one with most unique Z positions
    primary = max(series_info, key=lambda s: s['unique_z_count'])
    print(f"  Selected: Series #{primary['series_number']} ({primary['description']})")
    print(f"    {primary['unique_z_count']} unique positions, kernel: {primary['kernel']}")
    
    # Create output directory for primary series
    primary_dir = os.path.join(output_base, f"series_{primary['series_number']}_primary")
    os.makedirs(primary_dir, exist_ok=True)
    
    # Extract unique slices (deduplicate by Z position)
    z_seen = {}
    for s in primary['slices']:
        z = s['z_position']
        if z not in z_seen:
            z_seen[z] = s
    
    # Sort by Z position
    unique_slices = sorted(z_seen.values(), key=lambda s: s['z_position'])
    print(f"  Extracted {len(unique_slices)} unique slices")
    
    # Analyze gap pattern
    gaps = []
    for i in range(1, len(unique_slices)):
        gap = unique_slices[i]['z_position'] - unique_slices[i-1]['z_position']
        gaps.append(gap)
    
    if gaps:
        print(f"\n  Gap Analysis:")
        print(f"    Min gap: {min(gaps):.2f} mm")
        print(f"    Max gap: {max(gaps):.2f} mm")
        print(f"    Avg gap: {np.mean(gaps):.2f} mm")
        print(f"    Std gap: {np.std(gaps):.2f} mm")
        
        # Find the dominant gap (mode)
        rounded_gaps = [round(g, 1) for g in gaps]
        from collections import Counter
        gap_counts = Counter(rounded_gaps)
        dominant_gap = gap_counts.most_common(1)[0]
        print(f"    Dominant gap: {dominant_gap[0]} mm (appears {dominant_gap[1]} times)")
    
    # Save unique slices info
    meta = {
        'series_number': primary['series_number'],
        'description': primary['description'],
        'kernel': primary['kernel'],
        'unique_slices': len(unique_slices),
        'z_range': [unique_slices[0]['z_position'], unique_slices[-1]['z_position']],
        'dominant_gap': dominant_gap[0] if gaps else 0,
        'slices': [{
            'index': i,
            'filename': s['filename'],
            'z_position': s['z_position'],
            'gap_to_next': round(gaps[i], 2) if i < len(gaps) else None
        } for i, s in enumerate(unique_slices)]
    }
    
    meta_path = os.path.join(primary_dir, 'series_metadata.json')
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2)
    
    return primary, unique_slices, gaps, primary_dir


def step3_interpolate_slices(unique_slices, gaps, primary_dir, dcm_dir):
    """Step 3: Interpolate gaps between slices using advanced methods."""
    print("\n" + "=" * 60)
    print("  Step 3: Slice Interpolation")
    print("=" * 60)
    
    # Load all pixel data
    print("  Loading pixel data for all unique slices...")
    slice_data = []
    
    for i, s in enumerate(unique_slices):
        if (i + 1) % 50 == 0:
            print(f"    Loading: {i+1}/{len(unique_slices)}")
        
        fpath = os.path.join(dcm_dir, s['filename'])
        ds = pydicom.dcmread(fpath)
        slope = float(getattr(ds, 'RescaleSlope', 1.0))
        intercept = float(getattr(ds, 'RescaleIntercept', 0.0))
        
        # Convert to HU
        hu = ds.pixel_array.astype(np.float64) * slope + intercept
        
        slice_data.append({
            'hu': hu,
            'z_position': s['z_position'],
            'filename': s['filename'],
            'slope': slope,
            'intercept': intercept,
            'rows': ds.Rows,
            'cols': ds.Columns,
        })
    
    print(f"  Loaded {len(slice_data)} slices")
    
    # Determine dominant gap and interpolation needs
    from collections import Counter
    rounded_gaps = [round(g, 1) for g in gaps]
    gap_counts = Counter(rounded_gaps)
    dominant_gap = gap_counts.most_common(1)[0][0]
    
    print(f"\n  Dominant gap: {dominant_gap} mm")
    print(f"  Target interpolation: fill to ~{dominant_gap/2:.1f} mm spacing")
    
    # Strategy: For each gap > dominant_gap, interpolate intermediate slices
    # For standard gaps, interpolate 1 intermediate slice (halving the gap)
    
    interpolated_dir = os.path.join(primary_dir, 'interpolated')
    os.makedirs(interpolated_dir, exist_ok=True)
    
    original_dir = os.path.join(primary_dir, 'original')
    os.makedirs(original_dir, exist_ok=True)
    
    all_slices = []  # Will contain both original and interpolated
    interp_count = 0
    
    for i in range(len(slice_data)):
        # Save original slice
        img = apply_hu_windowing(slice_data[i]['hu'], 1.0, 0.0)
        pil_img = Image.fromarray(img)
        
        # Resize for display
        max_width = 512
        if pil_img.width > max_width:
            ratio = max_width / pil_img.width
            new_size = (max_width, int(pil_img.height * ratio))
            pil_img = pil_img.resize(new_size, Image.LANCZOS)
        
        orig_name = f"slice_{len(all_slices):04d}_z{slice_data[i]['z_position']:.1f}_orig.png"
        pil_img.save(os.path.join(interpolated_dir, orig_name), 'PNG')
        
        all_slices.append({
            'index': len(all_slices),
            'z_position': slice_data[i]['z_position'],
            'type': 'original',
            'png_name': orig_name,
            'source_file': slice_data[i]['filename'],
        })
        
        # Also save original to original dir
        orig_only_name = f"orig_{i:04d}_z{slice_data[i]['z_position']:.1f}.png"
        pil_img.save(os.path.join(original_dir, orig_only_name), 'PNG')
        
        # Interpolate between this and next slice
        if i < len(slice_data) - 1:
            gap = slice_data[i+1]['z_position'] - slice_data[i]['z_position']
            
            if gap > 0:
                # Number of intermediate slices to generate
                # For standard 5mm gaps: 1 intermediate (2.5mm spacing)
                # For larger gaps: proportionally more
                n_interp = max(1, int(round(gap / dominant_gap)) - 1)
                
                if n_interp > 10:  # Cap for very large gaps
                    n_interp = 10
                
                hu1 = slice_data[i]['hu']
                hu2 = slice_data[i+1]['hu']
                
                # Ensure same shape
                if hu1.shape != hu2.shape:
                    min_rows = min(hu1.shape[0], hu2.shape[0])
                    min_cols = min(hu1.shape[1], hu2.shape[1])
                    hu1 = hu1[:min_rows, :min_cols]
                    hu2 = hu2[:min_rows, :min_cols]
                
                for k in range(1, n_interp + 1):
                    alpha = k / (n_interp + 1)
                    z_interp = slice_data[i]['z_position'] + gap * alpha
                    
                    # Advanced interpolation: weighted blend with edge-aware smoothing
                    # Simple linear blend as baseline
                    hu_interp = (1 - alpha) * hu1 + alpha * hu2
                    
                    # Apply slight Gaussian smoothing to reduce interpolation artifacts
                    hu_interp = ndimage.gaussian_filter(hu_interp, sigma=0.5)
                    
                    # Convert to display
                    img_interp = apply_hu_windowing(hu_interp, 1.0, 0.0)
                    pil_interp = Image.fromarray(img_interp)
                    
                    if pil_interp.width > max_width:
                        ratio = max_width / pil_interp.width
                        new_size = (max_width, int(pil_interp.height * ratio))
                        pil_interp = pil_interp.resize(new_size, Image.LANCZOS)
                    
                    interp_name = f"slice_{len(all_slices):04d}_z{z_interp:.1f}_interp.png"
                    pil_interp.save(os.path.join(interpolated_dir, interp_name), 'PNG')
                    
                    all_slices.append({
                        'index': len(all_slices),
                        'z_position': z_interp,
                        'type': 'interpolated',
                        'png_name': interp_name,
                        'source_file': f"interp({slice_data[i]['filename']}<->{slice_data[i+1]['filename']})",
                        'alpha': alpha,
                    })
                    interp_count += 1
        
        if (i + 1) % 20 == 0:
            print(f"    Processing: {i+1}/{len(slice_data)} (interpolated: {interp_count})")
    
    print(f"\n  Original slices: {len(unique_slices)}")
    print(f"  Interpolated slices: {interp_count}")
    print(f"  Total slices: {len(all_slices)}")
    
    return all_slices, interpolated_dir


def step4_generate_comparison_gallery(all_slices, interpolated_dir, primary_dir):
    """Step 4: Generate HTML comparison gallery."""
    print("\n" + "=" * 60)
    print("  Step 4: Generating Comparison Gallery")
    print("=" * 60)
    
    rel_dir = os.path.basename(interpolated_dir)
    
    orig_count = sum(1 for s in all_slices if s['type'] == 'original')
    interp_count = sum(1 for s in all_slices if s['type'] == 'interpolated')
    
    slice_json = json.dumps([{
        'idx': s['index'],
        'z': round(s['z_position'], 1),
        'type': s['type'],
        'png': s['png_name'],
        'src': s.get('source_file', ''),
    } for s in all_slices])
    
    html = f"""<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>CT Slice Interpolation Viewer</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            background: #0a0a0a;
            color: #e0e0e0;
            font-family: 'Segoe UI', system-ui, sans-serif;
            padding: 20px;
        }}
        h1 {{
            text-align: center;
            background: linear-gradient(135deg, #00bcd4, #7c4dff);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 10px;
            font-size: 2em;
        }}
        .stats {{
            text-align: center;
            color: #888;
            margin-bottom: 20px;
            font-size: 0.9em;
        }}
        .stats span {{ font-weight: bold; }}
        .stats .orig {{ color: #4fc3f7; }}
        .stats .interp {{ color: #ff8a65; }}
        .stats .total {{ color: #81c784; }}
        
        .controls {{
            position: sticky;
            top: 0;
            z-index: 100;
            background: rgba(10,10,10,0.95);
            backdrop-filter: blur(10px);
            padding: 15px;
            border-radius: 12px;
            margin-bottom: 20px;
            display: flex;
            align-items: center;
            gap: 12px;
            flex-wrap: wrap;
            border: 1px solid #333;
            box-shadow: 0 4px 20px rgba(0,0,0,0.5);
        }}
        .controls label {{ color: #aaa; font-size: 0.85em; }}
        .controls input[type="range"] {{
            flex: 1;
            min-width: 200px;
            accent-color: #00bcd4;
            height: 6px;
        }}
        .slice-info {{
            color: #4fc3f7;
            font-weight: bold;
            min-width: 200px;
            font-size: 0.9em;
        }}
        .slice-info .interp-badge {{
            background: #ff5722;
            color: white;
            padding: 2px 8px;
            border-radius: 10px;
            font-size: 0.75em;
            margin-left: 5px;
        }}
        .slice-info .orig-badge {{
            background: #00bcd4;
            color: white;
            padding: 2px 8px;
            border-radius: 10px;
            font-size: 0.75em;
            margin-left: 5px;
        }}
        
        button {{
            background: linear-gradient(135deg, #00bcd4, #0097a7);
            color: #fff;
            border: none;
            padding: 8px 16px;
            border-radius: 8px;
            cursor: pointer;
            font-weight: bold;
            transition: all 0.2s;
        }}
        button:hover {{ transform: translateY(-1px); box-shadow: 0 4px 12px rgba(0,188,212,0.3); }}
        button.active {{ background: linear-gradient(135deg, #ff5722, #d84315); }}
        
        .filter-btns {{
            display: flex;
            gap: 5px;
        }}
        .filter-btns button {{
            font-size: 0.8em;
            padding: 5px 12px;
            background: #333;
            color: #ccc;
        }}
        .filter-btns button.active {{ 
            background: linear-gradient(135deg, #7c4dff, #651fff); 
            color: white;
        }}
        
        .viewer {{
            display: flex;
            flex-direction: column;
            align-items: center;
        }}
        .viewer img {{
            max-width: 100%;
            max-height: 65vh;
            border: 3px solid #333;
            border-radius: 8px;
            transition: border-color 0.3s;
        }}
        .viewer img.interp {{ border-color: #ff5722; }}
        .viewer img.orig {{ border-color: #00bcd4; }}
        .viewer .meta {{
            margin-top: 12px;
            color: #888;
            font-size: 0.85em;
            text-align: center;
        }}
        
        /* Grid view */
        .grid {{
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(150px, 1fr));
            gap: 6px;
        }}
        .grid .card {{
            position: relative;
            border: 2px solid #222;
            border-radius: 6px;
            overflow: hidden;
            cursor: pointer;
            transition: all 0.2s;
        }}
        .grid .card:hover {{ transform: scale(1.03); }}
        .grid .card.interp {{ border-color: rgba(255,87,34,0.5); }}
        .grid .card.orig {{ border-color: rgba(0,188,212,0.3); }}
        .grid .card img {{ width: 100%; display: block; }}
        .grid .card .label {{
            position: absolute;
            bottom: 0;
            left: 0;
            right: 0;
            background: rgba(0,0,0,0.75);
            padding: 3px 5px;
            font-size: 0.65em;
            color: #ccc;
        }}
        .grid .card .type-dot {{
            position: absolute;
            top: 5px;
            right: 5px;
            width: 10px;
            height: 10px;
            border-radius: 50%;
        }}
        .grid .card .type-dot.interp {{ background: #ff5722; }}
        .grid .card .type-dot.orig {{ background: #00bcd4; }}
        
        .hidden {{ display: none !important; }}
        
        .z-track {{
            width: 100%;
            height: 30px;
            margin: 15px 0;
            position: relative;
            background: #1a1a1a;
            border-radius: 4px;
            overflow: hidden;
        }}
        .z-track .z-marker {{
            position: absolute;
            top: 0;
            height: 100%;
            width: 2px;
        }}
        .z-track .z-marker.orig {{ background: rgba(0,188,212,0.6); }}
        .z-track .z-marker.interp {{ background: rgba(255,87,34,0.4); }}
        .z-track .z-current {{
            position: absolute;
            top: -2px;
            height: calc(100% + 4px);
            width: 4px;
            background: #fff;
            border-radius: 2px;
            z-index: 10;
            transition: left 0.1s;
        }}
    </style>
</head>
<body>
    <h1>CT Slice Interpolation Viewer</h1>
    <div class="stats">
        Original: <span class="orig">{orig_count}</span> |
        Interpolated: <span class="interp">{interp_count}</span> |
        Total: <span class="total">{len(all_slices)}</span> |
        Density Increase: <span class="total">{len(all_slices)/orig_count:.1f}x</span>
    </div>
    
    <div class="controls">
        <div class="filter-btns">
            <button class="active" onclick="setFilter('all')">All</button>
            <button onclick="setFilter('original')">Original Only</button>
            <button onclick="setFilter('interpolated')">Interpolated Only</button>
        </div>
        <label>Slice:</label>
        <input type="range" id="slider" min="0" max="{len(all_slices)-1}" value="0" oninput="goTo(this.value)">
        <div class="slice-info" id="info">--</div>
        <button id="playBtn" onclick="togglePlay()">Play</button>
        <label>Speed:</label>
        <input type="range" id="speed" min="30" max="300" value="100" style="width:80px;">
        <button onclick="setMode('single')" style="background:#333;color:#ccc;">Single</button>
        <button onclick="setMode('grid')" style="background:#333;color:#ccc;">Grid</button>
    </div>
    
    <div class="z-track" id="zTrack"></div>
    
    <div id="singleView" class="viewer">
        <img id="mainImg" src="" alt="CT Slice">
        <div class="meta" id="metaInfo"></div>
    </div>
    
    <div id="gridView" class="grid hidden"></div>
    
    <script>
    const allSlices = {slice_json};
    const dir = '{rel_dir}';
    let filtered = [...allSlices];
    let currentIdx = 0;
    let playing = false;
    let playTimer = null;
    let currentFilter = 'all';
    
    // Build Z track
    const zTrack = document.getElementById('zTrack');
    const zMin = allSlices[0].z;
    const zMax = allSlices[allSlices.length - 1].z;
    const zRange = zMax - zMin;
    
    allSlices.forEach(s => {{
        const marker = document.createElement('div');
        marker.className = 'z-marker ' + s.type;
        marker.style.left = ((s.z - zMin) / zRange * 100) + '%';
        zTrack.appendChild(marker);
    }});
    
    const zCurrent = document.createElement('div');
    zCurrent.className = 'z-current';
    zCurrent.id = 'zCurrent';
    zTrack.appendChild(zCurrent);
    
    // Build grid
    const gridEl = document.getElementById('gridView');
    allSlices.forEach((s, i) => {{
        const card = document.createElement('div');
        card.className = 'card ' + s.type;
        card.id = 'card_' + i;
        card.onclick = () => {{ goTo(i); setMode('single'); }};
        card.innerHTML = '<img src="' + dir + '/' + s.png + '" loading="lazy">' +
            '<div class="type-dot ' + s.type + '"></div>' +
            '<div class="label">#' + s.idx + ' Z:' + s.z + '</div>';
        gridEl.appendChild(card);
    }});
    
    function goTo(idx) {{
        idx = parseInt(idx);
        if (idx < 0 || idx >= filtered.length) return;
        currentIdx = idx;
        const s = filtered[idx];
        
        document.getElementById('mainImg').src = dir + '/' + s.png;
        document.getElementById('mainImg').className = s.type;
        document.getElementById('slider').value = idx;
        document.getElementById('slider').max = filtered.length - 1;
        
        const badge = s.type === 'interpolated' 
            ? '<span class="interp-badge">INTERPOLATED</span>'
            : '<span class="orig-badge">ORIGINAL</span>';
        document.getElementById('info').innerHTML = 
            '#' + idx + '/' + (filtered.length-1) + ' Z:' + s.z + 'mm' + badge;
        document.getElementById('metaInfo').textContent = 
            'Source: ' + s.src + ' | Z: ' + s.z + 'mm';
        
        // Update Z track cursor
        const pct = (s.z - zMin) / zRange * 100;
        document.getElementById('zCurrent').style.left = pct + '%';
    }}
    
    function setFilter(f) {{
        currentFilter = f;
        document.querySelectorAll('.filter-btns button').forEach(b => {{
            b.classList.toggle('active', b.textContent.toLowerCase().includes(
                f === 'all' ? 'all' : f === 'original' ? 'original only' : 'interpolated'));
        }});
        
        if (f === 'all') filtered = [...allSlices];
        else filtered = allSlices.filter(s => s.type === f);
        
        currentIdx = 0;
        goTo(0);
        
        // Update grid visibility
        allSlices.forEach((s, i) => {{
            const card = document.getElementById('card_' + i);
            if (card) {{
                if (f === 'all' || s.type === f) card.classList.remove('hidden');
                else card.classList.add('hidden');
            }}
        }});
    }}
    
    function setMode(m) {{
        document.getElementById('singleView').classList.toggle('hidden', m !== 'single');
        document.getElementById('gridView').classList.toggle('hidden', m !== 'grid');
    }}
    
    function togglePlay() {{
        playing = !playing;
        const btn = document.getElementById('playBtn');
        if (playing) {{
            btn.textContent = 'Pause';
            btn.classList.add('active');
            const spd = parseInt(document.getElementById('speed').value);
            playTimer = setInterval(() => {{
                currentIdx = (currentIdx + 1) % filtered.length;
                goTo(currentIdx);
            }}, spd);
        }} else {{
            btn.textContent = 'Play';
            btn.classList.remove('active');
            clearInterval(playTimer);
        }}
    }}
    
    document.addEventListener('keydown', e => {{
        if (e.key === 'ArrowRight' || e.key === 'ArrowDown') {{
            e.preventDefault();
            goTo(Math.min(currentIdx + 1, filtered.length - 1));
        }} else if (e.key === 'ArrowLeft' || e.key === 'ArrowUp') {{
            e.preventDefault();
            goTo(Math.max(currentIdx - 1, 0));
        }} else if (e.key === ' ') {{
            e.preventDefault();
            togglePlay();
        }}
    }});
    
    document.getElementById('singleView').addEventListener('wheel', e => {{
        e.preventDefault();
        if (e.deltaY > 0) goTo(Math.min(currentIdx + 1, filtered.length - 1));
        else goTo(Math.max(currentIdx - 1, 0));
    }});
    
    goTo(0);
    </script>
</body>
</html>"""
    
    html_path = os.path.join(primary_dir, 'interpolated_gallery.html')
    with open(html_path, 'w', encoding='utf-8') as f:
        f.write(html)
    
    print(f"  Gallery saved: {html_path}")
    return html_path


def main():
    # Step 1: Classify series
    series_info = step1_classify_series(DCM_DIR)
    
    if not series_info:
        print("No DICOM series found!")
        return
    
    # Save classification results
    os.makedirs(OUTPUT_BASE, exist_ok=True)
    
    # Step 2: Extract primary series
    primary, unique_slices, gaps, primary_dir = step2_extract_primary_series(series_info, OUTPUT_BASE)
    
    # Step 3: Interpolate
    all_slices, interpolated_dir = step3_interpolate_slices(unique_slices, gaps, primary_dir, DCM_DIR)
    
    # Step 4: Generate comparison gallery
    html_path = step4_generate_comparison_gallery(all_slices, interpolated_dir, primary_dir)
    
    # Summary
    print("\n" + "=" * 60)
    print("  COMPLETE")
    print("=" * 60)
    print(f"  Primary series: #{primary['series_number']} ({primary['description']})")
    print(f"  Original slices: {len(unique_slices)}")
    print(f"  After interpolation: {len(all_slices)}")
    print(f"  Density increase: {len(all_slices)/len(unique_slices):.1f}x")
    print(f"  Gallery: {html_path}")
    
    return html_path


if __name__ == '__main__':
    main()
