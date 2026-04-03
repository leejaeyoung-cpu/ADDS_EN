"""
cell_annotator.py
Standalone cell segmentation + annotation.
No heavy dependencies (no Cellpose, no integration_engine, no PipelineVisualizer).
Requires: numpy, opencv-python, Pillow, pandas, (optionally) scikit-image
"""
from __future__ import annotations

import numpy as np
import cv2
import pandas as pd
from io import BytesIO
from PIL import Image as _PIL
from typing import Optional, Tuple


# 20-color palette (RGB)
_COLORS = [
    (255, 99, 71), (30, 144, 255), (50, 205, 50), (255, 215, 0),
    (255, 20, 147), (0, 206, 209), (255, 140, 0), (147, 112, 219),
    (0, 255, 127), (255, 69, 0), (0, 191, 255), (154, 205, 50),
    (255, 105, 180), (64, 224, 208), (255, 160, 122), (100, 149, 237),
    (144, 238, 144), (255, 182, 193), (173, 255, 47), (135, 206, 250),
]


def _load_image(source) -> Optional[np.ndarray]:
    """Load image from bytes, BytesIO, file-like, or path. Returns RGB uint8 array."""
    try:
        if isinstance(source, (bytes, bytearray)):
            return np.array(_PIL.open(BytesIO(source)).convert("RGB"))
        if isinstance(source, BytesIO):
            source.seek(0)
            return np.array(_PIL.open(source).convert("RGB"))
        if hasattr(source, 'read'):
            source.seek(0)
            return np.array(_PIL.open(source).convert("RGB"))
        return np.array(_PIL.open(str(source)).convert("RGB"))
    except Exception as e:
        print(f"[cell_annotator] load error: {e}")
        return None


def _draw_label(img_bgr: np.ndarray, cx: float, cy: float, num: int, color_rgb: tuple):
    r, g, b = color_rgb
    lbl = str(num)
    font = cv2.FONT_HERSHEY_SIMPLEX
    fs = 0.38
    (tw, th), _ = cv2.getTextSize(lbl, font, fs, 1)
    lx, ly = int(cx) - tw // 2, int(cy) + th // 2
    cv2.rectangle(img_bgr, (lx - 2, ly - th - 2), (lx + tw + 2, ly + 2), (b, g, r), -1)
    cv2.putText(img_bgr, lbl, (lx, ly), font, fs, (255, 255, 255), 1, cv2.LINE_AA)


def annotate_cell_image(
    source,
    pixel_size_um: float = 0.5,
) -> Tuple[Optional[np.ndarray], pd.DataFrame, dict]:
    """
    Segment cells and produce annotated RGB image + per-cell DataFrame + summary.

    Parameters
    ----------
    source : bytes | BytesIO | file-like | path-str
        Image data.
    pixel_size_um : float
        Physical size of one pixel in micrometers.

    Returns
    -------
    annotated_rgb : np.ndarray (H, W, 3 uint8) or None on failure
    cell_df       : pd.DataFrame
    summary       : dict
    """
    img = _load_image(source)
    if img is None:
        return None, pd.DataFrame(), {"error": "load failed"}

    h, w = img.shape[:2]
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    # ── Preprocessing ────────────────────────────────────────────────────────
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    blurred = cv2.GaussianBlur(enhanced, (5, 5), 0)
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)

    # ── Watershed ────────────────────────────────────────────────────────────
    dist = cv2.distanceTransform(cleaned, cv2.DIST_L2, 5)
    if dist.max() == 0:
        return img, pd.DataFrame(), {"total_cells": 0, "image_size": f"{w}x{h}"}

    _, sure_fg = cv2.threshold(dist, 0.35 * dist.max(), 255, 0)
    sure_fg = sure_fg.astype(np.uint8)
    sure_bg = cv2.dilate(cleaned, kernel, iterations=3)
    unknown = cv2.subtract(sure_bg, sure_fg)
    _, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0
    markers_ws = cv2.watershed(img_bgr.copy(), markers)

    px_area = pixel_size_um ** 2
    records = []

    # ── Region extraction (skimage preferred) ────────────────────────────────
    try:
        from skimage import measure as _meas
        cell_mask = np.where(markers_ws > 1, markers_ws - 1, 0).astype(np.int32)
        regions = [rg for rg in _meas.regionprops(cell_mask) if rg.area >= 60]

        # Build a combined color overlay ONCE (no per-cell blending accumulation)
        overlay_bgr = img_bgr.copy()
        for i, rg in enumerate(regions):
            clr = _COLORS[i % len(_COLORS)]
            r, g, b = clr
            rrows, ccols = np.where(cell_mask == rg.label)
            # Fill cell region with color directly on overlay
            overlay_bgr[rrows, ccols] = [b, g, r]

        # Single blend: original * 0.55 + color_overlay * 0.45
        img_bgr = cv2.addWeighted(img_bgr, 0.55, overlay_bgr, 0.45, 0)

        # Draw contours, bounding boxes, and labels on top
        for i, rg in enumerate(regions):
            clr = _COLORS[i % len(_COLORS)]
            r, g, b = clr
            cy_r, cx_r = rg.centroid
            ap = rg.area
            pm = max(rg.perimeter, 1.0)
            circ = min(1.0, (4 * np.pi * ap) / (pm ** 2))
            y1, x1, y2, x2 = rg.bbox
            minor = rg.axis_minor_length
            major = max(rg.axis_major_length, 1.0)

            rb = (cell_mask == rg.label).astype(np.uint8)
            cnts, _ = cv2.findContours(rb, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(img_bgr, cnts, -1, (b, g, r), 2)
            cv2.rectangle(img_bgr, (x1, y1), (x2, y2), (b, g, r), 1)
            _draw_label(img_bgr, cx_r, cy_r, i + 1, clr)

            records.append({
                "Cell #": i + 1,
                "중심 X": round(float(cx_r), 1),
                "중심 Y": round(float(cy_r), 1),
                "면적 (μm²)": round(ap * px_area, 1),
                "등가직경 (μm)": round(rg.equivalent_diameter_area * pixel_size_um, 1),
                "원형도": round(circ, 3),
                "단축/장축": round(minor / max(major, 1.0), 3),
                "둘레 (px)": round(pm, 1),
                "형태 분류": "정상" if circ > 0.6 else ("경계" if circ > 0.4 else "불규칙"),
            })

    except ImportError:
        # OpenCV contours only (no skimage)
        cnts_all, _ = cv2.findContours(sure_fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        valid = [(c, cv2.contourArea(c)) for c in cnts_all if cv2.contourArea(c) >= 60]

        # Single-pass color fill on overlay
        overlay_bgr = img_bgr.copy()
        for i, (cnt, _) in enumerate(valid):
            clr = _COLORS[i % len(_COLORS)]
            r, g, b = clr
            cv2.drawContours(overlay_bgr, [cnt], -1, (b, g, r), cv2.FILLED)
        img_bgr = cv2.addWeighted(img_bgr, 0.55, overlay_bgr, 0.45, 0)

        for i, (cnt, ap) in enumerate(valid):
            clr = _COLORS[i % len(_COLORS)]
            r, g, b = clr
            M = cv2.moments(cnt)
            cx_r = M["m10"] / M["m00"] if M["m00"] else 0
            cy_r = M["m01"] / M["m00"] if M["m00"] else 0
            pm = max(cv2.arcLength(cnt, True), 1.0)
            circ = min(1.0, (4 * np.pi * ap) / (pm ** 2))
            bx, by, bw, bh = cv2.boundingRect(cnt)
            cv2.drawContours(img_bgr, [cnt], -1, (b, g, r), 2)
            cv2.rectangle(img_bgr, (bx, by), (bx + bw, by + bh), (b, g, r), 1)
            _draw_label(img_bgr, cx_r, cy_r, i + 1, clr)
            records.append({
                "Cell #": i + 1,
                "중심 X": round(cx_r, 1),
                "중심 Y": round(cy_r, 1),
                "면적 (μm²)": round(ap * px_area, 1),
                "등가직경 (μm)": round(2 * np.sqrt(ap / np.pi) * pixel_size_um, 1),
                "원형도": round(circ, 3),
                "단축/장축": None,
                "둘레 (px)": round(pm, 1),
                "형태 분류": "정상" if circ > 0.6 else ("경계" if circ > 0.4 else "불규칙"),
            })

    annotated_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    cell_df = pd.DataFrame(records) if records else pd.DataFrame()
    areas = [rec["면적 (μm²)"] for rec in records] if records else [0]
    circs = [rec["원형도"] for rec in records] if records else [0]
    n = len(records)

    summary = {
        "total_cells": n,
        "mean_area_um2": round(float(np.mean(areas)), 1),
        "std_area_um2": round(float(np.std(areas)), 1),
        "mean_circularity": round(float(np.mean(circs)), 3),
        "irregular_count": sum(1 for rec in records if rec.get("형태 분류") == "불규칙"),
        "normal_count": sum(1 for rec in records if rec.get("형태 분류") == "정상"),
        "image_size": f"{w}x{h}",
        "pixel_size_um": pixel_size_um,
    }
    return annotated_rgb, cell_df, summary
