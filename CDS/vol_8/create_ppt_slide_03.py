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

def create_ppt_slide_03():
    out_dir = r"f:\ADDS\CDS"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "ADDS_PPT_Slide_03.png")
    
    bg_path = r"C:\Users\brook\.gemini\antigravity\brain\574d389b-b6ed-48ef-afb4-dd86c25c56c1\data_silo_white_left_1773936498737.png"
    
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

    # 2. Draw Content Pane (Right side) keeping design consistency with Slide 1/2
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
    draw.text((cx, cy), "03. 임상 데이터 파편화(Fragmentation) 한계", font=title_font, fill=(34, 49, 63, 255))  # Deep charcoal
    cy += 70
    draw.text((cx, cy), "진료 부서별 분절된 의료 정보로 인한 통합 분석의 단절", font=sub_font, fill=(108, 122, 137, 255))
    cy += 70
    draw.line((cx, cy, px + pane_w - 60, cy), fill=(200, 200, 200, 255), width=2)
    cy += 60
    
    # Point 1: Silos
    draw.ellipse([cx, cy+15, cx+15, cy+30], fill=(211, 84, 0)) # Burnt Orange
    draw.text((cx + 35, cy), "임상 의료 정보의 극심한 사일로(Silo)화", font=point_font, fill=(44, 62, 80, 255))
    cy += 50
    draw.text((cx + 35, cy), "환자 진단 기록이 병리과, 외과, 종양내과 등 개별 부서 단위로 고립 축적", font=desc_font, fill=(108, 122, 137, 255))
    cy += 40
    draw.text((cx + 35, cy), "➔ 물리적·시스템적 장벽으로 다학제 간 입체적 정보 교류 차단", font=desc_font, fill=(211, 84, 0, 255))
    cy += 80
    
    # Point 2: Difficult Database integration
    draw.ellipse([cx, cy+15, cx+15, cy+30], fill=(41, 128, 185)) # Blue
    draw.text((cx + 35, cy), "종합적인 환자 데이터(DB) 구축의 근본적 어려움", font=point_font, fill=(44, 62, 80, 255))
    cy += 50
    draw.text((cx + 35, cy), "복부 영상(CT), 10만 배율 디지털 병리, 분자 유전체표 등 상이한 형식", font=desc_font, fill=(108, 122, 137, 255))
    cy += 40
    draw.text((cx + 35, cy), "➔ 비정형 데이터 호환 불가로 '환자 중심 1D 통합 데이터셋' 구성 불능", font=desc_font, fill=(41, 128, 185, 255))
    cy += 80

    # Point 3: Precision Failure
    draw.ellipse([cx, cy+15, cx+15, cy+30], fill=(192, 57, 43)) # Crimson Red
    draw.text((cx + 35, cy), "최적의 맞춤형 질병 예측 및 치료법 도출 한계", font=point_font, fill=(44, 62, 80, 255))
    cy += 50
    draw.text((cx + 35, cy), "단편화된 시각으로는 다중 모달리티 데이터 간 인과관계를 유추할 수 없음", font=desc_font, fill=(108, 122, 137, 255))
    cy += 40
    draw.text((cx + 35, cy), "➔ 단일 요인 진단에만 의존하여, 포괄적 표적 시너지(효과) 도출 불가능", font=desc_font, fill=(192, 57, 43, 255))
    
    # Footer UI Elements
    cy += 120
    draw.rectangle([cx-15, cy, cx+8, cy+80], fill=(44, 62, 80)) # Subtle dark accent bar
    draw.text((cx + 25, cy + 5), "결론: 기존 파편화된 OCS/PACS 시스템 한계를 극복할\n다중모달 융합 아키텍처(ADDS V1 Multi-Modal) 도입 절실", font=get_font(20, bold=True), fill=(34, 49, 63, 255))

    canvas.convert('RGB').save(out_path, quality=98)
    print(f"PPT Slide 03 saved to {out_path}")

if __name__ == "__main__":
    create_ppt_slide_03()
