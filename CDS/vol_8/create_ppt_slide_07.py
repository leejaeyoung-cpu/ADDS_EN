import os
from PIL import Image, ImageDraw, ImageFont, ImageEnhance

def get_font(size, bold=False):
    fonts = ["malgunbd.ttf" if bold else "malgun.ttf", "applegothic.ttf"]
    for font_name in fonts:
        try:
            return ImageFont.truetype(font_name, size)
        except IOError:
            continue
    return ImageFont.load_default()

def draw_glass_rect(canvas, bounds, fill, outline=None, alpha=180, width=0, radius=25):
    overlay = Image.new('RGBA', canvas.size, (255, 255, 255, 0))
    d = ImageDraw.Draw(overlay)
    d.rounded_rectangle(bounds, radius=radius, fill=fill + (alpha,))
    if outline:
        d.rounded_rectangle(bounds, radius=radius, outline=outline + (255,), width=width)
    canvas.paste(Image.alpha_composite(canvas.convert('RGBA'), overlay))
    return canvas

def create_ppt_slide_07():
    out_dir = r"f:\ADDS\CDS"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "ADDS_PPT_Slide_07.png")
    
    bg_path = r"C:\Users\brook\.gemini\antigravity\brain\574d389b-b6ed-48ef-afb4-dd86c25c56c1\slide_07_clinical_criteria_white_1773938383752.png"
    
    W, H = 1920, 1080
    
    # 1. Load Background
    if os.path.exists(bg_path):
        canvas = Image.open(bg_path).convert('RGBA')
        canvas = canvas.resize((W, H), Image.LANCZOS)
        enhancer = ImageEnhance.Brightness(canvas)
        canvas = enhancer.enhance(1.05)
    else:
        canvas = Image.new('RGBA', (W, H), color=(250, 250, 252, 255))
        print("Background not found, using solid soft white.")

    # 2. Draw Content Pane (Right side) keeping design consistency
    pane_w = 1100
    pane_h = 800
    px = W - pane_w - 40
    py = (H - pane_h) // 2
    
    # Minimalist frameless white translucent card
    canvas = draw_glass_rect(canvas, [px, py, px+pane_w, py+pane_h], (255, 255, 255), None, alpha=235, width=0, radius=25)
    
    # 3. Text Implementation
    draw = ImageDraw.Draw(canvas)
    
    title_font = get_font(38, bold=True)
    sub_font = get_font(24, bold=False)
    point_font = get_font(26, bold=True)
    desc_font = get_font(20, bold=False)
    
    cx = px + 60
    cy = py + 60
    
    # Header
    draw.text((cx, cy), "07. 초진 임상 검사 항목의 ADDS 반영 기준", font=title_font, fill=(34, 49, 63, 255))  # Deep charcoal
    cy += 60
    draw.text((cx, cy), "다중모달 정밀 융합을 위한 초진 필수 파라미터 규격화 가이드라인", font=sub_font, fill=(108, 122, 137, 255))
    cy += 70
    draw.line((cx, cy, px + pane_w - 60, cy), fill=(200, 200, 200, 255), width=2)
    cy += 50
    
    # Point 1: Essential Parameters
    draw.ellipse([cx, cy+15, cx+15, cy+30], fill=(211, 84, 0)) # Burnt Orange
    draw.text((cx + 35, cy), "필수 임상 코어 파라미터(Core Parameters) 지정", font=point_font, fill=(44, 62, 80, 255))
    cy += 45
    draw.text((cx + 35, cy), "- 종양 표지자(CEA, CA19-9) 및 전신 염증 마커(CRP 등) 필수 수집", font=desc_font, fill=(108, 122, 137, 255))
    cy += 40
    draw.text((cx + 35, cy), "- 연령, 성별, 전신 활동도(ECOG) 등 기본 임상 페놈 프로파일링 강제", font=desc_font, fill=(108, 122, 137, 255))
    cy += 70
    
    # Point 2: Modal-Specific
    draw.ellipse([cx, cy+15, cx+15, cy+30], fill=(41, 128, 185)) # Blue
    draw.text((cx + 35, cy), "다중모달 특이적 스크리닝(Modal-Specific Screening)", font=point_font, fill=(44, 62, 80, 255))
    cy += 45
    draw.text((cx + 35, cy), "- [병리]: H&E WSI 품질 20x 배율 이상 및 조직 손상도 기준 확립", font=desc_font, fill=(108, 122, 137, 255))
    cy += 40
    draw.text((cx + 35, cy), "- [분자]: NGS 기반 돌연변이(KRAS, NRAS, BRAF, MSI) 구조화 필수 입력", font=desc_font, fill=(108, 122, 137, 255))
    cy += 70

    # Point 3: Precision Failure
    draw.ellipse([cx, cy+15, cx+15, cy+30], fill=(192, 57, 43)) # Crimson Red
    draw.text((cx + 35, cy), "전향적 코호트 기반 AI 동기화 플로우 설정", font=point_font, fill=(44, 62, 80, 255))
    cy += 45
    draw.text((cx + 35, cy), "- 수집된 원시(Raw) 임상 결과를 ADDS 17D 텐서 형식으로 실시간 변환", font=desc_font, fill=(108, 122, 137, 255))
    cy += 40
    draw.text((cx + 35, cy), "- 초진 시 누락된 비위험군 데이터는 AI 딥러닝 보간을 통해 제한적 운영", font=desc_font, fill=(192, 57, 43, 255))
    
    # Footer UI Elements
    cy += 120
    draw.rectangle([cx-15, cy, cx+8, cy+80], fill=(44, 62, 80)) # Subtle dark accent bar
    draw.text((cx + 25, cy + 5), "결론: 주관적 판단에 의존하던 초진 차트 기록을,\n객관적 시스템 구동형 융합 데이터(Standardized AI Format)로\n완전 격상(Upgrade)시키는 핵심 절차입니다.", font=get_font(20, bold=True), fill=(34, 49, 63, 255))

    canvas.convert('RGB').save(out_path, quality=98)
    print(f"PPT Slide 07 saved to {out_path}")

if __name__ == "__main__":
    create_ppt_slide_07()
