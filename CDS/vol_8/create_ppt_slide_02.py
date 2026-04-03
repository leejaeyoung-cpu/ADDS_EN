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

def create_ppt_slide_02():
    out_dir = r"f:\ADDS\CDS"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "ADDS_PPT_Slide_02.png")
    
    bg_path = r"C:\Users\brook\.gemini\antigravity\brain\574d389b-b6ed-48ef-afb4-dd86c25c56c1\combo_therapy_left_aligned_1773936192858.png"
    
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

    # 2. Draw Content Pane (Right side) keeping design consistency with Slide 1
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
    draw.text((cx, cy), "02. 현행 임상 병용 요법의 다중적 한계", font=title_font, fill=(34, 49, 63, 255))  # Deep charcoal
    cy += 70
    draw.text((cx, cy), "다약제 병용 투약의 필연적 부작용 및 표적(Targeting) 실패", font=sub_font, fill=(108, 122, 137, 255))
    cy += 70
    draw.line((cx, cy, px + pane_w - 60, cy), fill=(200, 200, 200, 255), width=2)
    cy += 60
    
    # Point 1: Toxicity
    draw.ellipse([cx, cy+15, cx+15, cy+30], fill=(211, 84, 0)) # Burnt Orange
    draw.text((cx + 35, cy), "무분별한 병용 요법의 한계 독성(Toxicity)", font=point_font, fill=(44, 62, 80, 255))
    cy += 50
    draw.text((cx + 35, cy), "두 개 이상의 표적/항암제 병용 시 누적되는 치명적 부작용", font=desc_font, fill=(108, 122, 137, 255))
    cy += 40
    draw.text((cx + 35, cy), "➔ 환자 순응도(Compliance) 급감 및 조기 중단 리스크", font=desc_font, fill=(211, 84, 0, 255))
    cy += 80
    
    # Point 2: Resistance
    draw.ellipse([cx, cy+15, cx+15, cy+30], fill=(142, 68, 173)) # Purple
    draw.text((cx + 35, cy), "항암제 내성 및 면역 회피 기전 심화", font=point_font, fill=(44, 62, 80, 255))
    cy += 50
    draw.text((cx + 35, cy), "복잡한 종양 미세환경 내에서 암세포의 우회적 신호전달 경로 활성화", font=desc_font, fill=(108, 122, 137, 255))
    cy += 40
    draw.text((cx + 35, cy), "➔ 단일/이중 표적 병용으로는 내성 발현 방어 불가", font=desc_font, fill=(142, 68, 173, 255))
    cy += 80

    # Point 3: Precision Missing
    draw.ellipse([cx, cy+15, cx+15, cy+30], fill=(192, 57, 43)) # Crimson Red
    draw.text((cx + 35, cy), "정밀 바이오마커 기반의 표적 실패", font=point_font, fill=(44, 62, 80, 255))
    cy += 50
    draw.text((cx + 35, cy), "동반 변이(KRAS, PrPc 등)를 다각적으로 분석하지 않은 일차원적 처방", font=desc_font, fill=(108, 122, 137, 255))
    cy += 40
    draw.text((cx + 35, cy), "➔ 환자 개개인의 복합 변이 특성을 무시한 '제한된 맹목적 투여'", font=desc_font, fill=(192, 57, 43, 255))
    
    # Footer UI Elements
    cy += 120
    draw.rectangle([cx-15, cy, cx+8, cy+60], fill=(44, 62, 80)) # Subtle dark accent bar
    draw.text((cx + 25, cy + 5), "결론: 맹목적인 '여러가지 조합'이 아닌,\nAI 기반의 복합 시너지/독성 예측(CDSS) 모델 필수", font=get_font(20, bold=True), fill=(34, 49, 63, 255))

    canvas.convert('RGB').save(out_path, quality=98)
    print(f"PPT Slide 02 saved to {out_path}")

if __name__ == "__main__":
    create_ppt_slide_02()
