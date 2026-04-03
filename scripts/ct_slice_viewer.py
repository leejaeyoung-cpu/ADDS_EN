"""
CT Slice Sequential Viewer
===========================
DICOM 파일을 슬라이스 위치(ImagePositionPatient[2] or SliceLocation) 순서대로
정렬하고, 각 슬라이스를 PNG로 변환하여 HTML 갤러리로 출력합니다.

슬라이스 간 간격(gap) 정보도 함께 표시하여
딥러닝 보간(interpolation)이 필요한 구간을 파악할 수 있게 합니다.
"""

import os
import sys
import io
import json
import numpy as np

# Fix Windows console encoding
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

try:
    import pydicom
except ImportError:
    print("pydicom이 필요합니다: pip install pydicom")
    sys.exit(1)

try:
    from PIL import Image
except ImportError:
    print("Pillow가 필요합니다: pip install Pillow")
    sys.exit(1)


DCM_DIR = r"f:\ADDS\CTdata\CTdcm"
OUTPUT_DIR = r"f:\ADDS\CTdata\ct_slices_ordered"
HTML_OUTPUT = r"f:\ADDS\CTdata\ct_slice_gallery.html"

# HU Windowing for abdomen
WINDOW_CENTER = 50
WINDOW_WIDTH = 400


def apply_hu_windowing(pixel_array, slope, intercept, wc, ww):
    """Apply HU conversion and windowing."""
    hu = pixel_array.astype(np.float64) * slope + intercept
    min_val = wc - ww / 2
    max_val = wc + ww / 2
    hu = np.clip(hu, min_val, max_val)
    # Normalize to 0-255
    hu = ((hu - min_val) / (max_val - min_val) * 255).astype(np.uint8)
    return hu


def load_and_sort_dicoms(dcm_dir):
    """Load all DICOM files and sort by slice location."""
    slices = []
    print(f"DICOM 파일 로딩 중: {dcm_dir}")
    
    files = [f for f in os.listdir(dcm_dir) if f.endswith('.dcm')]
    total = len(files)
    
    for i, fname in enumerate(files):
        if (i + 1) % 50 == 0:
            print(f"  로딩 진행: {i+1}/{total}")
        
        fpath = os.path.join(dcm_dir, fname)
        try:
            ds = pydicom.dcmread(fpath)
            
            # Get slice location - try multiple methods
            if hasattr(ds, 'ImagePositionPatient'):
                z_pos = float(ds.ImagePositionPatient[2])
            elif hasattr(ds, 'SliceLocation'):
                z_pos = float(ds.SliceLocation)
            else:
                # Use instance number as fallback
                z_pos = float(getattr(ds, 'InstanceNumber', i))
            
            slope = float(getattr(ds, 'RescaleSlope', 1.0))
            intercept = float(getattr(ds, 'RescaleIntercept', 0.0))
            
            slices.append({
                'filename': fname,
                'filepath': fpath,
                'z_position': z_pos,
                'instance_number': int(getattr(ds, 'InstanceNumber', 0)),
                'slice_thickness': float(getattr(ds, 'SliceThickness', 0)),
                'pixel_array': ds.pixel_array,
                'slope': slope,
                'intercept': intercept,
                'rows': ds.Rows,
                'cols': ds.Columns,
                'pixel_spacing': [float(x) for x in getattr(ds, 'PixelSpacing', [1, 1])],
            })
        except Exception as e:
            print(f"  경고: {fname} 로딩 실패 - {e}")
    
    # Sort by z_position (ascending = feet to head typically)
    slices.sort(key=lambda s: s['z_position'])
    print(f"총 {len(slices)}개 슬라이스 로딩 완료")
    return slices


def analyze_gaps(slices):
    """Analyze gaps between slices."""
    gaps = []
    for i in range(1, len(slices)):
        gap = abs(slices[i]['z_position'] - slices[i-1]['z_position'])
        gaps.append(gap)
    
    if gaps:
        print(f"\n=== 슬라이스 간격 분석 ===")
        print(f"  최소 간격: {min(gaps):.2f} mm")
        print(f"  최대 간격: {max(gaps):.2f} mm")
        print(f"  평균 간격: {np.mean(gaps):.2f} mm")
        print(f"  표준편차: {np.std(gaps):.2f} mm")
        
        # Identify large gaps (> 2x average)
        avg_gap = np.mean(gaps)
        large_gaps = [(i, g) for i, g in enumerate(gaps) if g > avg_gap * 2]
        if large_gaps:
            print(f"\n  [WARNING] 큰 간격 발견 ({len(large_gaps)}곳):")
            for idx, g in large_gaps:
                print(f"    슬라이스 {idx} → {idx+1}: {g:.2f} mm")
    
    return gaps


def save_slices_as_png(slices, output_dir):
    """Save each slice as PNG with HU windowing."""
    os.makedirs(output_dir, exist_ok=True)
    
    png_files = []
    total = len(slices)
    print(f"\nPNG 변환 중...")
    
    for i, s in enumerate(slices):
        if (i + 1) % 50 == 0 or i == 0:
            print(f"  변환 진행: {i+1}/{total}")
        
        # Apply HU windowing
        img_array = apply_hu_windowing(
            s['pixel_array'], s['slope'], s['intercept'],
            WINDOW_CENTER, WINDOW_WIDTH
        )
        
        # Resize for web display (max 512px wide)
        img = Image.fromarray(img_array)
        max_width = 512
        if img.width > max_width:
            ratio = max_width / img.width
            new_size = (max_width, int(img.height * ratio))
            img = img.resize(new_size, Image.LANCZOS)
        
        # Save with ordered filename
        png_name = f"slice_{i:04d}_z{s['z_position']:.1f}.png"
        png_path = os.path.join(output_dir, png_name)
        img.save(png_path, 'PNG')
        
        png_files.append({
            'index': i,
            'png_name': png_name,
            'original_file': s['filename'],
            'z_position': s['z_position'],
            'instance_number': s['instance_number'],
        })
    
    print(f"  {total}개 PNG 저장 완료: {output_dir}")
    return png_files


def generate_html_gallery(png_files, gaps, output_html, png_dir):
    """Generate an interactive HTML gallery of all slices."""
    
    rel_png_dir = os.path.basename(png_dir)
    avg_gap = np.mean(gaps) if gaps else 0
    
    html = f"""<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>CT Slice Sequential Viewer</title>
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
            color: #00bcd4;
            margin-bottom: 10px;
            font-size: 1.8em;
        }}
        .stats {{
            text-align: center;
            color: #888;
            margin-bottom: 20px;
            font-size: 0.9em;
        }}
        .stats span {{ color: #4fc3f7; font-weight: bold; }}
        
        .controls {{
            position: sticky;
            top: 0;
            z-index: 100;
            background: rgba(10,10,10,0.95);
            backdrop-filter: blur(10px);
            padding: 15px;
            border-radius: 10px;
            margin-bottom: 20px;
            display: flex;
            align-items: center;
            gap: 15px;
            flex-wrap: wrap;
            border: 1px solid #333;
        }}
        .controls label {{ color: #aaa; font-size: 0.85em; }}
        .controls input[type="range"] {{
            flex: 1;
            min-width: 200px;
            accent-color: #00bcd4;
        }}
        .controls .slice-info {{
            color: #4fc3f7;
            font-weight: bold;
            min-width: 180px;
        }}
        .controls button {{
            background: #00bcd4;
            color: #000;
            border: none;
            padding: 8px 16px;
            border-radius: 5px;
            cursor: pointer;
            font-weight: bold;
        }}
        .controls button:hover {{ background: #26c6da; }}
        .controls button.active {{ background: #ff5722; }}
        
        .view-mode {{ display: flex; gap: 10px; }}
        .view-mode button {{
            background: #333;
            color: #ccc;
            font-size: 0.8em;
            padding: 5px 12px;
        }}
        .view-mode button.active {{ background: #00bcd4; color: #000; }}
        
        /* Single slice view */
        .single-view {{
            display: flex;
            flex-direction: column;
            align-items: center;
        }}
        .single-view img {{
            max-width: 100%;
            max-height: 70vh;
            border: 2px solid #333;
            border-radius: 5px;
        }}
        .single-view .meta {{
            margin-top: 10px;
            color: #888;
            font-size: 0.85em;
        }}
        
        /* Grid view */
        .grid-view {{
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(180px, 1fr));
            gap: 8px;
        }}
        .grid-view .slice-card {{
            position: relative;
            border: 2px solid #222;
            border-radius: 5px;
            overflow: hidden;
            cursor: pointer;
            transition: border-color 0.2s, transform 0.2s;
        }}
        .grid-view .slice-card:hover {{
            border-color: #00bcd4;
            transform: scale(1.02);
        }}
        .grid-view .slice-card.large-gap {{
            border-color: #ff5722;
        }}
        .grid-view .slice-card img {{
            width: 100%;
            display: block;
        }}
        .grid-view .slice-card .label {{
            position: absolute;
            bottom: 0;
            left: 0;
            right: 0;
            background: rgba(0,0,0,0.7);
            padding: 4px 6px;
            font-size: 0.7em;
            color: #ccc;
        }}
        .grid-view .slice-card .label .z {{ color: #4fc3f7; }}
        
        .gap-indicator {{
            grid-column: 1 / -1;
            background: rgba(255, 87, 34, 0.15);
            border: 1px dashed #ff5722;
            border-radius: 5px;
            padding: 8px;
            text-align: center;
            color: #ff8a65;
            font-size: 0.85em;
        }}
        
        /* Film strip view */
        .filmstrip-view {{
            display: flex;
            overflow-x: auto;
            gap: 2px;
            padding: 10px 0;
        }}
        .filmstrip-view img {{
            height: 200px;
            flex-shrink: 0;
            cursor: pointer;
            border: 1px solid #333;
            transition: border-color 0.2s;
        }}
        .filmstrip-view img:hover {{ border-color: #00bcd4; }}
        .filmstrip-view img.active {{ border: 2px solid #ff5722; }}
        
        .hidden {{ display: none !important; }}
    </style>
</head>
<body>
    <h1>🔬 CT Slice Sequential Viewer</h1>
    <div class="stats">
        Total Slices: <span>{len(png_files)}</span> | 
        Avg Gap: <span>{avg_gap:.2f}mm</span> | 
        Z Range: <span>{png_files[0]['z_position']:.1f}</span> → <span>{png_files[-1]['z_position']:.1f}</span> mm |
        Window: <span>C:{WINDOW_CENTER} W:{WINDOW_WIDTH}</span> (Abdomen)
    </div>
    
    <div class="controls">
        <div class="view-mode">
            <button class="active" onclick="setView('single')">Single</button>
            <button onclick="setView('grid')">Grid</button>
            <button onclick="setView('filmstrip')">Filmstrip</button>
        </div>
        <label>Slice:</label>
        <input type="range" id="sliceSlider" min="0" max="{len(png_files)-1}" value="0" 
               oninput="showSlice(this.value)">
        <div class="slice-info" id="sliceInfo">Slice 0 / {len(png_files)-1}</div>
        <button id="playBtn" onclick="togglePlay()">▶ Play</button>
        <label>Speed:</label>
        <input type="range" id="speedSlider" min="50" max="500" value="150" style="width:80px;">
    </div>
    
    <div id="singleView" class="single-view">
        <img id="mainImage" src="{rel_png_dir}/{png_files[0]['png_name']}" alt="CT Slice">
        <div class="meta" id="sliceMeta"></div>
    </div>
    
    <div id="gridView" class="grid-view hidden">
"""
    
    # Add grid items
    for i, pf in enumerate(png_files):
        gap_class = ''
        # Check if there's a large gap before this slice
        if i > 0 and i-1 < len(gaps):
            if gaps[i-1] > avg_gap * 2:
                html += f'<div class="gap-indicator">⚠️ 큰 간격: {gaps[i-1]:.1f}mm (보간 필요)</div>\n'
                gap_class = ' large-gap'
        
        html += f"""<div class="slice-card{gap_class}" onclick="showSlice({i}); setView('single');">
    <img src="{rel_png_dir}/{pf['png_name']}" loading="lazy" alt="Slice {i}">
    <div class="label">#{i} <span class="z">Z:{pf['z_position']:.1f}</span> {pf['original_file']}</div>
</div>
"""
    
    html += """</div>
    
    <div id="filmstripView" class="filmstrip-view hidden">
"""
    for i, pf in enumerate(png_files):
        html += f'<img src="{rel_png_dir}/{pf["png_name"]}" loading="lazy" onclick="showSlice({i}); setView(\'single\');" alt="Slice {i}" id="film_{i}">\n'
    
    # Build slice metadata JSON
    slice_data_json = json.dumps([{
        'index': pf['index'],
        'png': pf['png_name'],
        'z': pf['z_position'],
        'file': pf['original_file'],
        'gap_before': round(gaps[pf['index']-1], 2) if pf['index'] > 0 and pf['index']-1 < len(gaps) else 0
    } for pf in png_files])
    
    html += f"""</div>
    
    <script>
    const sliceData = {slice_data_json};
    const pngDir = '{rel_png_dir}';
    let currentSlice = 0;
    let playing = false;
    let playInterval = null;
    
    function showSlice(idx) {{
        idx = parseInt(idx);
        currentSlice = idx;
        const s = sliceData[idx];
        document.getElementById('mainImage').src = pngDir + '/' + s.png;
        document.getElementById('sliceSlider').value = idx;
        document.getElementById('sliceInfo').textContent = 
            'Slice ' + idx + ' / {len(png_files)-1}';
        document.getElementById('sliceMeta').textContent = 
            'Z: ' + s.z.toFixed(1) + 'mm | File: ' + s.file + 
            (s.gap_before > 0 ? ' | Gap from prev: ' + s.gap_before + 'mm' : '');
        
        // Update filmstrip active
        document.querySelectorAll('.filmstrip-view img').forEach((img, i) => {{
            img.classList.toggle('active', i === idx);
        }});
    }}
    
    function setView(mode) {{
        document.getElementById('singleView').classList.toggle('hidden', mode !== 'single');
        document.getElementById('gridView').classList.toggle('hidden', mode !== 'grid');
        document.getElementById('filmstripView').classList.toggle('hidden', mode !== 'filmstrip');
        document.querySelectorAll('.view-mode button').forEach(btn => {{
            btn.classList.toggle('active', btn.textContent.toLowerCase().includes(mode));
        }});
    }}
    
    function togglePlay() {{
        playing = !playing;
        const btn = document.getElementById('playBtn');
        if (playing) {{
            btn.textContent = '⏸ Pause';
            btn.classList.add('active');
            const speed = parseInt(document.getElementById('speedSlider').value);
            playInterval = setInterval(() => {{
                currentSlice = (currentSlice + 1) % sliceData.length;
                showSlice(currentSlice);
            }}, speed);
        }} else {{
            btn.textContent = '▶ Play';
            btn.classList.remove('active');
            clearInterval(playInterval);
        }}
    }}
    
    // Keyboard navigation
    document.addEventListener('keydown', (e) => {{
        if (e.key === 'ArrowRight' || e.key === 'ArrowDown') {{
            e.preventDefault();
            showSlice(Math.min(currentSlice + 1, sliceData.length - 1));
        }} else if (e.key === 'ArrowLeft' || e.key === 'ArrowUp') {{
            e.preventDefault();
            showSlice(Math.max(currentSlice - 1, 0));
        }} else if (e.key === ' ') {{
            e.preventDefault();
            togglePlay();
        }}
    }});
    
    // Mouse wheel on single view
    document.getElementById('singleView').addEventListener('wheel', (e) => {{
        e.preventDefault();
        if (e.deltaY > 0) showSlice(Math.min(currentSlice + 1, sliceData.length - 1));
        else showSlice(Math.max(currentSlice - 1, 0));
    }});
    
    // Initial display
    showSlice(0);
    </script>
</body>
</html>"""
    
    with open(output_html, 'w', encoding='utf-8') as f:
        f.write(html)
    
    print(f"\nHTML 갤러리 생성 완료: {output_html}")


def main():
    print("=" * 60)
    print("  CT Slice Sequential Viewer")
    print("=" * 60)
    
    # 1. Load and sort DICOM files
    slices = load_and_sort_dicoms(DCM_DIR)
    
    if not slices:
        print("❌ DICOM 파일을 찾을 수 없습니다.")
        return
    
    # 2. Analyze gaps
    gaps = analyze_gaps(slices)
    
    # 3. Save as PNG
    png_files = save_slices_as_png(slices, OUTPUT_DIR)
    
    # 4. Generate HTML gallery
    generate_html_gallery(png_files, gaps, HTML_OUTPUT, OUTPUT_DIR)
    
    # 5. Save metadata
    meta_path = os.path.join(OUTPUT_DIR, 'slice_metadata.json')
    meta = {
        'total_slices': len(slices),
        'z_range': [slices[0]['z_position'], slices[-1]['z_position']],
        'avg_gap': float(np.mean(gaps)) if gaps else 0,
        'pixel_spacing': slices[0]['pixel_spacing'],
        'slice_thickness': slices[0]['slice_thickness'],
        'slices': [{
            'index': i,
            'filename': s['filename'],
            'z_position': s['z_position'],
            'gap_to_next': float(gaps[i]) if i < len(gaps) else None
        } for i, s in enumerate(slices)]
    }
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2)
    print(f"메타데이터 저장: {meta_path}")
    
    print(f"\n✅ 완료! 브라우저에서 열기: {HTML_OUTPUT}")


if __name__ == '__main__':
    main()
