"""
Improved Tumor Visualization with Segmentation Mask
종양의 정확한 모양을 픽셀 단위로 표시
"""

import numpy as np
from PIL import Image, ImageDraw
import cv2

def visualize_tumor_segmentation(
    original_image,
    segmentation_mask,
    overlay_alpha=0.4,
    contour_thickness=2,
    tumor_color=(255, 0, 0),  # Red
    show_contour=True,
    show_filled=True
):
    """
    종양 segmentation을 원본 이미지 위에 정확하게 표시
    
    Args:
        original_image: 원본 CT 이미지 (PIL Image 또는 numpy array)
        segmentation_mask: Binary mask (numpy array, 0=배경, 1=종양)
        overlay_alpha: 투명도 (0=투명, 1=불투명)
        contour_thickness: 경계선 두께
        tumor_color: 종양 표시 색상 (R, G, B)
        show_contour: 경계선 표시 여부
        show_filled: 채워진 영역 표시 여부
    
    Returns:
        PIL.Image: 시각화된 이미지
    """
    # PIL Image → numpy array 변환
    if isinstance(original_image, Image.Image):
        img_array = np.array(original_image.convert('RGB'))
    else:
        img_array = original_image.copy()
        if len(img_array.shape) == 2:  # Grayscale
            img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
    
    # Segmentation mask 준비
    if isinstance(segmentation_mask, np.ndarray):
        mask = (segmentation_mask > 0).astype(np.uint8)
    else:
        mask = np.array(segmentation_mask).astype(np.uint8)
    
    # 결과 이미지 복사
    result = img_array.copy()
    
    # 1. 채워진 영역 표시 (투명한 overlay)
    if show_filled and np.sum(mask) > 0:
        # Color mask 생성
        color_mask = np.zeros_like(img_array)
        color_mask[mask > 0] = tumor_color
        
        # Alpha blending
        result = cv2.addWeighted(
            result,
            1 - overlay_alpha,
            color_mask,
            overlay_alpha,
            0
        )
    
    # 2. 경계선 표시 (Contour)
    if show_contour and np.sum(mask) > 0:
        # Contour 추출
        contours, _ = cv2.findContours(
            mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )
        
        # 모든 contour 그리기
        cv2.drawContours(
            result,
            contours,
            -1,  # 모든 contour
            tumor_color,
            contour_thickness
        )
    
    # numpy → PIL Image
    return Image.fromarray(result.astype(np.uint8))


def create_side_by_side_comparison(
    original_image,
    segmentation_mask,
    title_original="원본 이미지",
    title_segmented="종양 검출 결과"
):
    """
    원본과 검출 결과를 나란히 배치
    
    Args:
        original_image: 원본 이미지
        segmentation_mask: Segmentation mask
        title_original: 원본 제목
        title_segmented: 검출 결과 제목
    
    Returns:
        PIL.Image: Side-by-side 비교 이미지
    """
    from PIL import ImageFont, ImageDraw
    
    # 원본을 RGB로 변환
    if isinstance(original_image, Image.Image):
        orig_pil = original_image.convert('RGB')
    else:
        orig_pil = Image.fromarray(original_image).convert('RGB')
    
    # Segmentation 결과 생성
    seg_result = visualize_tumor_segmentation(
        original_image,
        segmentation_mask,
        overlay_alpha=0.3,
        contour_thickness=3,
        show_contour=True,
        show_filled=True
    )
    
    # 두 이미지의 크기가 같은지 확인
    if orig_pil.size != seg_result.size:
        seg_result = seg_result.resize(orig_pil.size)
    
    # Side-by-side 캔버스 생성
    width, height = orig_pil.size
    canvas_width = width * 2 + 40  # 간격 포함
    canvas_height = height + 80  # 제목 공간 포함
    
    canvas = Image.new('RGB', (canvas_width, canvas_height), color=(255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    
    # 제목 추가
    try:
        font = ImageFont.truetype("arial.ttf", 24)
    except:
        font = ImageFont.load_default()
    
    # 원본 이미지 배치
    canvas.paste(orig_pil, (20, 60))
    draw.text((20, 20), title_original, fill=(0, 0, 0), font=font)
    
    # 검출 결과 배치
    canvas.paste(seg_result, (width + 40, 60))
    draw.text((width + 40, 20), title_segmented, fill=(0, 0, 0), font=font)
    
    return canvas


def extract_tumor_statistics(segmentation_mask):
    """
    Segmentation mask에서 통계 추출
    
    Args:
        segmentation_mask: Binary mask
    
    Returns:
        dict: 종양 통계
    """
    mask = (segmentation_mask > 0).astype(np.uint8)
    
    # Contour 추출
    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )
    
    if len(contours) == 0:
        return {
            'tumor_detected': False,
            'area': 0,
            'perimeter': 0,
            'circularity': 0,
            'bounding_box': None
        }
    
    # 가장 큰 contour 선택
    largest_contour = max(contours, key=cv2.contourArea)
    
    # 통계 계산
    area = cv2.contourArea(largest_contour)
    perimeter = cv2.arcLength(largest_contour, True)
    
    # Circularity (원형도): 4π × area / perimeter²
    # 1.0 = 완벽한 원, 0에 가까울수록 불규칙
    circularity = 4 * np.pi * area / (perimeter ** 2) if perimeter > 0 else 0
    
    # Bounding box
    x, y, w, h = cv2.boundingRect(largest_contour)
    
    return {
        'tumor_detected': True,
        'area': float(area),
        'perimeter': float(perimeter),
        'circularity': float(circularity),
        'bounding_box': {'x': int(x), 'y': int(y), 'width': int(w), 'height': int(h)},
        'contour_points': len(largest_contour)
    }


# 간단한 테스트 함수
if __name__ == "__main__":
    import sys
    sys.path.insert(0, 'C:/Users/brook/Desktop/ADDS/src')
    
    from medical_imaging.ct_analyzer import CTAnalyzer

import os as _os
from pathlib import Path as _Path
# ADDS_BASE_DIR environment variable overrides automatic detection
BASE_DIR = _Path(_os.environ.get("ADDS_BASE_DIR", str(_Path(__file__).resolve().parent.parent)))

    
    # Test image
    test_image = str(BASE_DIR / "output")
    
    print("Testing improved tumor visualization...")
    
    # CT 분석
    analyzer = CTAnalyzer(use_gpu=True, use_nnunet=False)
    result = analyzer.analyze_ct_image(test_image)
    
    if result.get('status') == 'success':
        # 원본 이미지
        original = Image.open(test_image)
        
        # Mask는 analyzer 내부에서 가져와야 함 (임시로 mock)
        # 실제로는 analyzer._segment_tumor()의 결과를 반환하도록 수정 필요
        print("✅ Analysis complete")
        print(f"Tumor detected: {result['segmentation']['tumor_detected']}")
        
        # TODO: actual mask를 가져와서 시각화
        print("\n⚠️ Note: Full visualization requires mask extraction from analyzer")
