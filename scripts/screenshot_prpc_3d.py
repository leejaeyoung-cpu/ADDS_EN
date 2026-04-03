"""
screenshot_prpc_3d.py
=====================
Selenium Chrome headless로 py3Dmol HTML을 렌더링하고
고해상도 PNG 스크린샷 저장
"""

import time, os
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager

HTML_PATH = r"f:\ADDS\outputs\pritamab_pptx_figures\prpc_pritamab_3d.html"
OUT_PNG   = r"f:\ADDS\outputs\pritamab_pptx_figures\prpc_antibody_binding_3d.png"

file_url = "file:///" + HTML_PATH.replace("\\", "/")

opts = Options()
opts.add_argument("--headless=new")
opts.add_argument("--window-size=1600,1100")
opts.add_argument("--disable-gpu=false")
opts.add_argument("--enable-webgl")
opts.add_argument("--use-gl=angle")
opts.add_argument("--use-angle=gl-egl")
opts.add_argument("--allow-file-access-from-files")
opts.add_argument("--disable-web-security")
opts.add_argument("--no-sandbox")
opts.add_argument("--disable-dev-shm-usage")
opts.add_argument("--force-device-scale-factor=2")
opts.add_argument("--hide-scrollbars")            # ← 스크롤바 숨기기

print("Starting Chrome...")
driver = webdriver.Chrome(
    service=Service(ChromeDriverManager().install()),
    options=opts
)

try:
    driver.set_window_size(1600, 1100)
    print(f"Opening: {file_url}")
    driver.get(file_url)

    # Wait for 3Dmol WebGL to render
    print("Waiting for WebGL render (10s)...")
    time.sleep(10)

    # Force zoom-out via 3Dmol JS API so nothing clips
    driver.execute_script("""
        try {
            var keys = Object.keys(window).filter(k => window[k] && typeof window[k] === 'object' && window[k].rotate);
            keys.forEach(function(k) {
                try {
                    window[k].zoomTo();
                    window[k].zoom(0.72);
                    window[k].render();
                } catch(e) {}
            });
        } catch(e) {}
    """)
    time.sleep(3)

    # Screenshot
    raw_path = OUT_PNG.replace('.png', '_raw.png')
    driver.save_screenshot(raw_path)
    print(f"Raw screenshot saved: {raw_path}")

finally:
    driver.quit()
    print("Chrome closed.")

# ── PIL: auto-crop dark background + save final ─────────────────────
from PIL import Image
import numpy as np

img = Image.open(raw_path).convert('RGB')
arr = np.array(img)

# Find non-dark pixels (content region)
# Dark threshold: any channel < 25 → background
mask = arr.max(axis=2) > 25          # True = content pixel
rows = np.any(mask, axis=1)
cols = np.any(mask, axis=0)

rmin, rmax = np.where(rows)[0][[0, -1]]
cmin, cmax = np.where(cols)[0][[0, -1]]

# Add 40px padding around content
pad = 40
rmin = max(0, rmin - pad)
rmax = min(arr.shape[0], rmax + pad)
cmin = max(0, cmin - pad)
cmax = min(arr.shape[1], cmax + pad)

cropped = img.crop((cmin, rmin, cmax, rmax))
cropped.save(OUT_PNG, dpi=(300, 300))
print(f"Final cropped PNG saved: {OUT_PNG}  ({cmax-cmin}x{rmax-rmin}px)")
