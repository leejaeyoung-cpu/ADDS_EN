"""
Enhanced Cellpose Analysis with Pipeline Visualization
Integrates real Cellpose analysis with complete metadata tracking
"""

import numpy as np
from PIL import Image
import cv2
from typing import Optional, Dict, Any, Tuple
import time
from datetime import datetime
from pathlib import Path
import sys

# Add paths
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from src.medical_imaging.cdss.integration_engine import CellposeResults
from ui.utils.pipeline_visualizer import PipelineVisualizer
from ui.utils.metadata_tracker import (
    create_analysis_metadata,
    create_data_provenance,
    get_current_author_info
)

try:
    from src.preprocessing.image_processor import CellposeProcessor
    from skimage import measure, exposure
    CELLPOSE_AVAILABLE = True
except ImportError:
    CELLPOSE_AVAILABLE = False


def analyze_cellpose_with_pipeline(
    image_file,
    pixel_size_um: float = 0.5,
    diameter: Optional[int] = None,
    molecular_data: Optional[Dict[str, str]] = None
) -> Tuple[Optional[CellposeResults], Optional[PipelineVisualizer], Optional[Dict[str, Any]]]:
    """
    Analyze microscopy image with Cellpose and create complete pipeline visualization
    
    Args:
        image_file: Uploaded image file
        pixel_size_um: Pixel size in micrometers
        diameter: Cell diameter for Cellpose (None = auto-detect)
        
    Returns:
        Tuple of (CellposeResults, PipelineVisualizer, complete_metadata)
    """
    if not CELLPOSE_AVAILABLE:
        return None, None, None
    
    try:
        # === STAGE 0: Data Provenance ===
        start_time = time.time()
        
        # Get file hash for provenance
        import hashlib
        file_content = image_file.read()
        file_hash = hashlib.md5(file_content).hexdigest()
        image_file.seek(0)  # Reset file pointer
        
        provenance = {
            'source_file': image_file.name,
            'upload_timestamp': datetime.now().isoformat(),
            'uploader': get_current_author_info(),
            'file_hash': file_hash,
            'file_size_bytes': len(file_content)
        }
        
        # Initialize pipeline visualizer
        pipeline = PipelineVisualizer("Cellpose 세포 분석")
        
        # Load image
        img = np.array(Image.open(image_file))
        original_img = img.copy()
        
        # === STAGE 1: 원본 이미지 ===
        stage1_start = time.time()
        
        pipeline.add_stage(
            "원본 이미지",
            original_img,
            {
                '이미지 크기': f"{img.shape[1]}x{img.shape[0]}",
                '채널 수': img.shape[2] if len(img.shape) == 3 else 1,
                '데이터 타입': str(img.dtype),
                '픽셀 크기': f"{pixel_size_um} μm",
                '파일명': image_file.name
            }
        )
        
        stage1_duration = (time.time() - stage1_start) * 1000
        pipeline.stages[-1].set_duration(stage1_duration)
        
        # === STAGE 2: 전처리 (CLAHE + 정규화) ===
        stage2_start = time.time()
        
        # Convert to grayscale if needed
        if len(img.shape) == 3:
            img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        else:
            img_gray = img
        
        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        img_clahe = clahe.apply(img_gray.astype(np.uint8))
        
        # Normalize to 0-1
        img_normalized = img_clahe.astype(np.float32) / 255.0
        
        pipeline.add_stage(
            "전처리 (CLAHE)",
            img_clahe,
            {
                '방법': 'CLAHE + 정규화',
                'Clip Limit': 2.0,
                'Tile Size': '8x8',
                '출력 범위': '0-1',
                '평균 픽셀 값': f"{img_normalized.mean():.3f}",
                '표준편차': f"{img_normalized.std():.3f}"
            }
        )
        
        stage2_duration = (time.time() - stage2_start) * 1000
        pipeline.stages[-1].set_duration(stage2_duration)
        
        # === STAGE 3: Cellpose 모델 추론 ===
        stage3_start = time.time()
        
        # Initialize Cellpose
        processor = CellposeProcessor(model_type='cyto2', gpu=True)
        
        # Segment image
        masks, flows, metadata = processor.segment_image(
            img_normalized,
            diameter=diameter,
            channels=[0, 0]  # Grayscale
        )
        
        num_cells = metadata['num_cells']
        detected_diameter = metadata.get('diameter', diameter or 30)
        
        # Create colored mask visualization
        colored_mask = np.zeros((*masks.shape, 3), dtype=np.uint8)
        for cell_id in range(1, num_cells + 1):
            color = np.random.randint(0, 255, 3)
            colored_mask[masks == cell_id] = color
        
        pipeline.add_stage(
            "Cellpose 모델 추론",
            colored_mask,
            {
                '모델': 'cyto2',
                '검출된 세포 수': num_cells,
                'Cell Diameter': f"{detected_diameter:.1f} pixels",
                'GPU 사용': 'Yes' if processor.gpu else 'No',
                'Flow 임계값': 0.4,
                '추론 시간': f"{(time.time() - stage3_start) * 1000:.0f}ms"
            }
        )
        
        stage3_duration = (time.time() - stage3_start) * 1000
        pipeline.stages[-1].set_duration(stage3_duration)
        
        # === STAGE 4: 후처리 (특징 추출) ===
        stage4_start = time.time()
        
        # Extract cell properties using skimage
        regions = measure.regionprops(masks)
        
        # Calculate features
        pixel_area_um2 = pixel_size_um ** 2
        areas_um2 = [r.area * pixel_area_um2 for r in regions]
        mean_area = np.mean(areas_um2) if areas_um2 else 0
        
        # Calculate circularities
        circularities = []
        for r in regions:
            if r.perimeter > 0:
                circ = 4 * np.pi * r.area / (r.perimeter ** 2)
                circularities.append(min(circ, 1.0))
        mean_circularity = np.mean(circularities) if circularities else 0.5
        
        # Create overlay visualization
        overlay = original_img.copy()
        if len(overlay.shape) ==2:
            overlay = cv2.cvtColor(overlay, cv2.COLOR_GRAY2RGB)
        
        # Draw contours
        for region in regions:
            contour = region.coords
            if len(contour) > 2:
                # Draw contour
                pts = contour[:, [1, 0]].astype(np.int32)  # Swap x,y
                cv2.polylines(overlay, [pts], True, (0, 255, 0), 2)
                
                # Draw centroid
                cy, cx = region.centroid
                cv2.circle(overlay, (int(cx), int(cy)), 3, (255, 0, 0), -1)
        
        pipeline.add_stage(
            "후처리 (컨투어 추출)",
            overlay,
            {
                '추출된 세포': len(regions),
                '평균 면적': f"{mean_area:.1f} μm²",
                '평균 원형도': f"{mean_circularity:.3f}",
                '최소 면적': f"{min(areas_um2):.1f} μm²" if areas_um2 else "N/A",
                '최대 면적': f"{max(areas_um2):.1f} μm²" if areas_um2 else "N/A",
                '컨투어 색상': '녹색 (Green)',
                '중심점 색상': '빨강 (Red)'
            }
        )
        
        stage4_duration = (time.time() - stage4_start) * 1000
        pipeline.stages[-1].set_duration(stage4_duration)
        
        # === STAGE 5: 최종 결과 추출 ===
        stage5_start = time.time()
        
        # Calculate morphology score
        area_std = np.std(areas_um2) if areas_um2 else 0
        area_cv = area_std / mean_area if mean_area > 0 else 0
        morphology_score = max(0, 10 - area_cv * 20)
        
        # Ki-67 NOTE: Ki-67 proliferation index requires immunohistochemistry (IHC).
        # Cell density is NOT a valid clinical proxy for Ki-67.
        # We pass None to signal the value is unavailable from image analysis alone.
        ki67_estimate = None  # IHC required — do not infer from cell density

        
        # Create histogram visualization
        import matplotlib.pyplot as plt
        import io
        
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        
        # Area distribution
        axes[0].hist(areas_um2, bins=30, color='skyblue', edgecolor='black')
        axes[0].set_xlabel('Cell Area (μm²)')
        axes[0].set_ylabel('Frequency')
        axes[0].set_title('Cell Area Distribution')
        axes[0].axvline(mean_area, color='red', linestyle='--', label=f'Mean: {mean_area:.1f}')
        axes[0].legend()
        
        # Circularity distribution
        axes[1].hist(circularities, bins=30, color='lightgreen', edgecolor='black')
        axes[1].set_xlabel('Circularity')
        axes[1].set_ylabel('Frequency')
        axes[1].set_title('Cell Circularity Distribution')
        axes[1].axvline(mean_circularity, color='red', linestyle='--', label=f'Mean: {mean_circularity:.3f}')
        axes[1].legend()
        
        plt.tight_layout()
        
        # Convert plot to image
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=100)
        buf.seek(0)
        hist_img = np.array(Image.open(buf))
        plt.close()
        
        pipeline.add_stage(
            "결과 추출 및 통계",
            hist_img,
            {
                '총 세포 수': num_cells,
                '평균 세포 면적': f"{mean_area:.1f} μm²",
                '평균 원형도': f"{mean_circularity:.3f}",
                '형태학 점수': f"{morphology_score:.1f}/10",
                'Ki-67 증식 지표': '측정 불가 (IHC 별도 필요)',
                '면적 변이계수': f"{area_cv:.3f}"
            }
        )
        
        stage5_duration = (time.time() - stage5_start) * 1000
        pipeline.stages[-1].set_duration(stage5_duration)
        
        # === STAGE 6: 분자 진단 융합 (Multi-Modal Molecular Fusion) ===
        stage6_start = time.time()
        
        # Calculate Cancer Severity Score (0-100, higher = more severe)
        base_severity = min(100.0, float(area_cv * 150.0))
        severity_score = base_severity
        
        clinical_impact = "Standard Observation (No prominent molecular markers)"
        molecular_flags = {}
        
        if molecular_data:
            kras = molecular_data.get('kras', 'WT')
            nras = molecular_data.get('nras_braf', 'WT')
            msi = molecular_data.get('msi_mmr', 'MSS')
            
            molecular_flags['KRAS'] = kras
            molecular_flags['NRAS/BRAF'] = nras
            molecular_flags['MSI/MMR'] = msi
            
            # Severity modifiers
            if kras != 'WT':
                severity_score = min(100.0, severity_score + 25.0)
                clinical_impact = f"High Risk: {kras} Mutation detected. Targeted therapy required. High risk of early progression."
            elif nras != 'WT':
                severity_score = min(100.0, severity_score + 20.0)
                clinical_impact = f"Risk factor: {nras} Mutation. Alternative targeted therapy path necessary."
                
            if msi == 'MSI-H':
                severity_score = max(0.0, severity_score - 15.0)
                clinical_impact += " Validated MSI-H: Highly favorable candidate for Immunotherapy."
                
        # Create a visual block/text image for STAGE 6
        import matplotlib.pyplot as plt
        import io
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.set_facecolor('#1a2530')
        ax.axis('off')
        
        plt.text(0.5, 0.8, f"Cancer Severity Score: {severity_score:.1f}/100", 
                 color='cyan' if severity_score < 50 else 'orange' if severity_score < 75 else 'red',
                 fontsize=20, ha='center', fontweight='bold')
                 
        plt.text(0.5, 0.5, f"Clinical Impact:\n{clinical_impact}", 
                 color='white', fontsize=12, ha='center', wrap=True)
                 
        if molecular_flags:
            flags_txt = " | ".join([f"{k}: {v}" for k, v in molecular_flags.items()])
            plt.text(0.5, 0.2, f"Molecular: {flags_txt}", color='lightgreen', fontsize=11, ha='center')
            
        plt.tight_layout()
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=100, bbox_inches='tight', facecolor='#1a2530')
        buf.seek(0)
        stage6_img = np.array(Image.open(buf))
        plt.close(fig)
        
        pipeline.add_stage(
            "분자 진단 및 다중모달 융합",
            stage6_img,
            {
                '형태학 불안정성': f"{base_severity:.1f}%",
                '최종 진행 위험도(Severity)': f"{severity_score:.1f}/100",
                '임상 의료 결정 가이드': clinical_impact,
                'KRAS 변이 여부': molecular_flags.get('KRAS', 'Not Provided'),
                'MSI 상태': molecular_flags.get('MSI/MMR', 'Not Provided')
            }
        )
        
        stage6_duration = (time.time() - stage6_start) * 1000
        pipeline.stages[-1].set_duration(stage6_duration)
        
        # Finalize pipeline
        pipeline.finalize()
        
        # Create CellposeResults (Note: We pass severity as a dynamic attribute if dataclass is strict, or just modify what's available)
        cellpose_result = CellposeResults(
            cell_count=num_cells,
            mean_area_um2=float(mean_area),
            mean_circularity=float(mean_circularity),
            morphology_score=float(morphology_score),
            ki67_index=float(ki67_estimate) if ki67_estimate else 0.0
        )
        # Monkey patch the new properties into the result for downstream fusion
        cellpose_result.cancer_severity_score = severity_score
        cellpose_result.clinical_impact = clinical_impact
        
        # Create complete metadata
        complete_metadata = {
            'analysis_type': 'cellpose_segmentation',
            'author': get_current_author_info(),
            'provenance': provenance,
            'pipeline': pipeline.get_timeline_data(),
            'inference': {
                'model': {
                    'name': 'Cellpose',
                    'version': 'cyto2',
                    'framework': 'PyTorch',
                    'device': 'CUDA' if processor.gpu else 'CPU'
                },
                'inference_params': {
                    'diameter': detected_diameter,
                    'flow_threshold': 0.4,
                    'cellprob_threshold': 0.0,
                    'channels': [0, 0]
                },
                'executed_by': get_current_author_info(),
                'execution_time': datetime.now().isoformat()
            },
            'results': {
                'cell_count': num_cells,
                'mean_area_um2': float(mean_area),
                'mean_circularity': float(mean_circularity),
                'morphology_score': float(morphology_score),
                'cancer_severity_score': float(severity_score),
                'clinical_impact': clinical_impact,
                'molecular_flags': molecular_flags,
                'ki67_index': None
            },
            'total_processing_time_ms': pipeline.total_duration_ms,
            'created_at': datetime.now().isoformat(),
            'version': '2.0.0'
        }
        
        return cellpose_result, pipeline, complete_metadata
        
    except Exception as e:
        print(f"Error in pipeline analysis: {e}")
        import traceback
        traceback.print_exc()


def annotate_cell_image(image_file, pixel_size_um: float = 0.5):
    """
    Perform cell segmentation on an uploaded microscopy image.
    Returns annotated RGB image + per-cell DataFrame + summary dict.
    Works WITHOUT Cellpose: uses OpenCV Watershed + (optional) skimage regionprops.
    """
    import numpy as np
    import cv2
    import pandas as pd
    from PIL import Image as _PIL

    # 1. Load
    try:
        if hasattr(image_file, 'read'):
            image_file.seek(0)
            img = np.array(_PIL.open(image_file).convert("RGB"))
        else:
            img = np.array(_PIL.open(str(image_file)).convert("RGB"))
    except Exception as e:
        return None, None, {"error": str(e)}

    h, w = img.shape[:2]
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    # 2. CLAHE + Otsu threshold
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    blurred = cv2.GaussianBlur(enhanced, (5, 5), 0)
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)

    # 3. Watershed
    dist = cv2.distanceTransform(cleaned, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(dist, 0.35 * dist.max(), 255, 0)
    sure_fg = sure_fg.astype(np.uint8)
    sure_bg = cv2.dilate(cleaned, kernel, iterations=3)
    unknown = cv2.subtract(sure_bg, sure_fg)
    _, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0
    markers_ws = cv2.watershed(img_bgr.copy(), markers)

    # 4. Color palette
    COLORS = [
        (255, 99, 71), (30, 144, 255), (50, 205, 50), (255, 215, 0),
        (255, 20, 147), (0, 206, 209), (255, 140, 0), (147, 112, 219),
        (0, 255, 127), (255, 69, 0), (0, 191, 255), (154, 205, 50),
        (255, 105, 180), (64, 224, 208), (255, 160, 122), (100, 149, 237),
        (144, 238, 144), (255, 182, 193), (173, 255, 47), (135, 206, 250),
    ]
    px_area = pixel_size_um ** 2
    records = []

    def _label(img_b, cx, cy, num, color):
        r, g, b = color
        fs = 0.38
        font = cv2.FONT_HERSHEY_SIMPLEX
        lbl = str(num)
        (tw, th), _ = cv2.getTextSize(lbl, font, fs, 1)
        lx, ly = int(cx) - tw // 2, int(cy) + th // 2
        cv2.rectangle(img_b, (lx - 2, ly - th - 2), (lx + tw + 2, ly + 2), (b, g, r), -1)
        cv2.putText(img_b, lbl, (lx, ly), font, fs, (255, 255, 255), 1, cv2.LINE_AA)

    # 5. Region extraction
    try:
        from skimage import measure
        cell_mask = np.where(markers_ws > 1, markers_ws - 1, 0).astype(np.int32)
        regions = [rg for rg in measure.regionprops(cell_mask) if rg.area >= 60]
        for i, rg in enumerate(regions):
            clr = COLORS[i % len(COLORS)]
            r, g, b = clr
            cy_r, cx_r = rg.centroid
            ap = rg.area
            pm = max(rg.perimeter, 1.0)
            circ = min(1.0, (4 * np.pi * ap) / (pm ** 2))
            y1, x1, y2, x2 = rg.bbox
            minor = rg.minor_axis_length
            major = max(rg.major_axis_length, 1.0)

            rrows, ccols = np.where(cell_mask == rg.label)
            alpha_mask = np.zeros(img_bgr.shape, dtype=np.uint8)
            alpha_mask[rrows, ccols] = [b, g, r]
            img_bgr = cv2.addWeighted(img_bgr, 0.6, alpha_mask, 0.4, 0)

            rb = (cell_mask == rg.label).astype(np.uint8)
            cnts, _ = cv2.findContours(rb, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(img_bgr, cnts, -1, (b, g, r), 2)
            cv2.rectangle(img_bgr, (x1, y1), (x2, y2), (b, g, r), 1)
            _label(img_bgr, cx_r, cy_r, i + 1, clr)

            records.append({
                "Cell #": i + 1,
                "중심 X": round(float(cx_r), 1),
                "중심 Y": round(float(cy_r), 1),
                "면적 (μm²)": round(ap * px_area, 1),
                "등가직경 (μm)": round(rg.equivalent_diameter * pixel_size_um, 1),
                "원형도": round(circ, 3),
                "단축/장축": round(minor / major, 3),
                "둘레 (px)": round(pm, 1),
                "형태 분류": "정상" if circ > 0.6 else ("경계" if circ > 0.4 else "불규칙"),
            })

    except ImportError:
        # Fallback: OpenCV contours only
        cnts_all, _ = cv2.findContours(sure_fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        valid = [(c, cv2.contourArea(c)) for c in cnts_all if cv2.contourArea(c) >= 60]
        for i, (cnt, ap) in enumerate(valid):
            clr = COLORS[i % len(COLORS)]
            r, g, b = clr
            M = cv2.moments(cnt)
            cx_r = M["m10"] / M["m00"] if M["m00"] else 0
            cy_r = M["m01"] / M["m00"] if M["m00"] else 0
            pm = max(cv2.arcLength(cnt, True), 1.0)
            circ = min(1.0, (4 * np.pi * ap) / (pm ** 2))
            bx, by, bw, bh = cv2.boundingRect(cnt)
            cv2.drawContours(img_bgr, [cnt], -1, (b, g, r), 2)
            cv2.rectangle(img_bgr, (bx, by), (bx + bw, by + bh), (b, g, r), 1)
            _label(img_bgr, cx_r, cy_r, i + 1, clr)
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
