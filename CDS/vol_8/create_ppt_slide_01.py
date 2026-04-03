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

def draw_glass_rect(canvas, bounds, fill_color, outline_color, alpha=180, width=2, radius=15):
    overlay = Image.new('RGBA', canvas.size, (255, 255, 255, 0))
    d = ImageDraw.Draw(overlay)
    d.rounded_rectangle(bounds, radius=radius, fill=fill_color + (alpha,))
    if outline_color:
        d.rounded_rectangle(bounds, radius=radius, outline=outline_color + (255,), width=width)
    canvas.paste(Image.alpha_composite(canvas.convert('RGBA'), overlay))
    return canvas

def create_ppt_slide_01():
    out_dir = r"f:\ADDS\CDS"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "ADDS_PPT_Slide_01.png")
    
    bg_path = r"C:\Users\brook\.gemini\antigravity\brain\574d389b-b6ed-48ef-afb4-dd86c25c56c1\kras_modern_white_1773931293955.png"
    
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

    # 2. Draw Content Pane (Right side)
    pane_w = 1100
    pane_h = 800
    px = W - pane_w - 40
    py = (H - pane_h) // 2
    
    # Very subtle, elegant frameless white translucent card
    canvas = draw_glass_rect(canvas, [px, py, px+pane_w, py+pane_h], (255, 255, 255), None, alpha=230, width=0, radius=25)
    
    # 3. Text Implementation
    draw = ImageDraw.Draw(canvas)
    
    title_font = get_font(38, bold=True)
    sub_font = get_font(24, bold=False)
    point_font = get_font(26, bold=True)
    desc_font = get_font(20, bold=False)
    
    cx = px + 60
    cy = py + 60
    
    # Header
    draw.text((cx, cy), "01. 임상 미충족 수요 (Unmet Needs)", font=title_font, fill=(34, 49, 63, 255))  # Deep charcoal
    cy += 70
    draw.text((cx, cy), "대장암(CRC)의 임상적 한계와 표적 치료의 딜레마", font=sub_font, fill=(108, 122, 137, 255)) # Steel gray
    cy += 70
    draw.line((cx, cy, px + pane_w - 60, cy), fill=(200, 200, 200, 255), width=2)
    cy += 60
    
    # Point 1
    draw.ellipse([cx, cy+15, cx+15, cy+30], fill=(211, 84, 0)) # Burnt Orange / Red
    draw.text((cx + 35, cy), "대장암 발병률 지속적 증가", font=point_font, fill=(44, 62, 80, 255))
    cy += 50
    draw.text((cx + 35, cy), "국내외 발생률 최상위권 및 진행성/전이성 대장암의 제한적 생존율", font=desc_font, fill=(108, 122, 137, 255))
    cy += 80
    
    # Point 2
    draw.ellipse([cx, cy+15, cx+15, cy+30], fill=(41, 128, 185)) # Deep Blue
    draw.text((cx + 35, cy), "난치성 KRAS 유전자 변이 다수 동반", font=point_font, fill=(44, 62, 80, 255))
    cy += 50
    draw.text((cx + 35, cy), "전체 대장암 환자의 약 40~50%에서 발생하여 종양의 이질성 심화", font=desc_font, fill=(108, 122, 137, 255))
    cy += 80

    # Point 3
    draw.ellipse([cx, cy+15, cx+15, cy+30], fill=(192, 57, 43)) # Crimson Red
    draw.text((cx + 35, cy), "표적 치료 조합의 근본적 한계", font=point_font, fill=(44, 62, 80, 255))
    cy += 50
    draw.text((cx + 35, cy), "KRAS 변이 시 기존 항EGFR 표적항암제(Cetuximab 등) 무효화", font=desc_font, fill=(108, 122, 137, 255))
    cy += 40
    draw.text((cx + 35, cy), "➔ 대안적인 병용 요법 및 정밀 치료 옵션 극히 제한적", font=desc_font, fill=(192, 57, 43, 255))
    
    # Footer UI Elements
    cy += 120
    draw.rectangle([cx-15, cy, cx+8, cy+60], fill=(44, 62, 80)) # Subtle dark accent bar
    draw.text((cx + 25, cy + 5), "현행 표준 치료 가이드라인을 넘어서는\n새로운 다중모달 정밀 의사결정 지원 시스템 시급", font=get_font(20, bold=True), fill=(34, 49, 63, 255))

    canvas.convert('RGB').save(out_path, quality=98)
    print(f"PPT Slide 01 saved to {out_path}")

if __name__ == "__main__":
    create_ppt_slide_01()
