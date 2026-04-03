"""
Full end-to-end verification of the Cellpose annotation pipeline.
Mirrors the exact code path used in patient_management.py Step 2 + display section.
"""
import os, sys, importlib.util, traceback
from io import BytesIO
from PIL import Image
import numpy as np

PASS = []
FAIL = []

def check(name, cond, detail=""):
    if cond:
        PASS.append(name)
        print(f"  PASS | {name}")
    else:
        FAIL.append(name)
        print(f"  FAIL | {name} {detail}")

print("=" * 60)
print("STEP 1: Import cell_annotator via importlib")
print("=" * 60)

pm_file = r"f:\ADDS\src\ui\page_modules\patient_management.py"
ann_path = os.path.abspath(os.path.join(os.path.dirname(pm_file), '..', 'cell_annotator.py'))
print(f"  Resolved path: {ann_path}")
check("cell_annotator.py exists", os.path.exists(ann_path))

try:
    spec = importlib.util.spec_from_file_location("cell_annotator", ann_path)
    ann_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ann_mod)
    annotate_cell_image = ann_mod.annotate_cell_image
    check("importlib load OK", True)
    check("annotate_cell_image callable", callable(annotate_cell_image))
except Exception as e:
    check("importlib load OK", False, str(e))
    traceback.print_exc()
    sys.exit(1)

print()
print("=" * 60)
print("STEP 2: Simulate UploadedFile bytes")
print("=" * 60)

def make_cell_image(n_cells=8, size=256):
    img = np.ones((size, size, 3), dtype=np.uint8) * 200
    rng = np.random.default_rng(42 + n_cells)
    for _ in range(n_cells):
        cx, cy = rng.integers(40, size-40, 2)
        rx, ry = rng.integers(12, 35, 2)
        val = int(rng.integers(40, 110))
        Y, X = np.ogrid[:size, :size]
        mask = ((X-cx)**2/rx**2 + (Y-cy)**2/ry**2) < 1
        img[mask] = [val, val, val]
    return Image.fromarray(img)

n_images = 3
img_bytes_list = []
for i in range(n_images):
    pil_img = make_cell_image(n_cells=4+i)
    buf = BytesIO()
    pil_img.save(buf, format="PNG")
    img_bytes_list.append(buf.getvalue())

check("Bytes generated", len(img_bytes_list) == n_images)
check("Bytes non-empty", all(len(b) > 500 for b in img_bytes_list))

print()
print("=" * 60)
print("STEP 3: annotate_cell_image() on each image")
print("=" * 60)

import warnings
warnings.filterwarnings("error", category=FutureWarning)

ann_bytes_list, df_list, sum_list = [], [], []
from PIL import Image as _PIL_s

for i, img_b in enumerate(img_bytes_list):
    try:
        ann_arr, cdf, csum = annotate_cell_image(BytesIO(img_b), pixel_size_um=0.5)
        check(f"Img{i+1}: ann_arr returned", ann_arr is not None)
        if ann_arr is not None:
            check(f"Img{i+1}: shape (H,W,3) uint8",
                  ann_arr.ndim == 3 and ann_arr.shape[2] == 3 and ann_arr.dtype == np.uint8,
                  str(ann_arr.shape) + " " + str(ann_arr.dtype))
            n = csum.get('total_cells', 0)
            check(f"Img{i+1}: cells > 0", n > 0, f"cells={n}")
            check(f"Img{i+1}: df rows == cells", len(cdf) == n, f"df={len(cdf)}, sum={n}")
            print(f"    cells={n} | mean_area={csum.get('mean_area_um2')} um2 | circ={csum.get('mean_circularity')} | irr={csum.get('irregular_count')}")

            buf = BytesIO()
            _PIL_s.fromarray(ann_arr).save(buf, format='PNG')
            png_bytes = buf.getvalue()
            check(f"Img{i+1}: PNG serialize ({len(png_bytes)} bytes)", len(png_bytes) > 500)
            ann_bytes_list.append(png_bytes)
            df_list.append(cdf)
            sum_list.append(csum)

    except FutureWarning as fw:
        check(f"Img{i+1}: No FutureWarning", False, str(fw))
    except Exception as e:
        check(f"Img{i+1}: annotate OK", False, str(e))
        traceback.print_exc()

print()
print("=" * 60)
print("STEP 4: Session state simulate + display read")
print("=" * 60)

session = {
    'cellpose_ann_bytes': ann_bytes_list,
    'cellpose_cell_dfs': df_list,
    'cellpose_summaries': sum_list,
}
read_back = session.get('cellpose_ann_bytes', [])
check("Session: ann_bytes count", len(read_back) == n_images, f"expected {n_images}, got {len(read_back)}")

for idx, png_b in enumerate(read_back):
    pil_check = Image.open(BytesIO(png_b))
    w, h = pil_check.size
    check(f"Display Img{idx+1}: BytesIO->PIL open", w > 0 and h > 0, f"{w}x{h}")

total_cells = sum(s.get('total_cells', 0) for s in sum_list)
check("Total detected cells > 0", total_cells > 0, str(total_cells))

print()
print("=" * 60)
print("STEP 5: DataFrame schema validation")
print("=" * 60)
expected_cols = ['Cell #', '중심 X', '중심 Y', '면적 (μm²)', '등가직경 (μm)', '원형도', '단축/장축', '둘레 (px)', '형태 분류']
for i, df in enumerate(df_list):
    if df is not None and len(df) > 0:
        missing = [c for c in expected_cols if c not in df.columns]
        check(f"DF{i+1}: all 9 cols present", len(missing) == 0, f"missing:{missing}")
        valid_m = all(v in ['정상', '경계', '불규칙'] for v in df['형태 분류'].unique())
        check(f"DF{i+1}: 형태분류 valid", valid_m, str(df['형태 분류'].unique().tolist()))
        print(f"    {df['형태 분류'].value_counts().to_dict()}")

print()
print("=" * 60)
print(f"최종결과: PASS {len(PASS)} / FAIL {len(FAIL)} / 전체 {len(PASS)+len(FAIL)}")
print("=" * 60)
if FAIL:
    print("실패 항목:")
    for f in FAIL:
        print(f"  - {f}")
else:
    print("모든 항목 통과!")
