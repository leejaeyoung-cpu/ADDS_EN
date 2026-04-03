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

def create_ppt_slide_05():
    out_dir = r"f:\ADDS\CDS"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "ADDS_PPT_Slide_05.png")
    bg_path = r"C:\Users\brook\.gemini\antigravity\brain\574d389b-b6ed-48ef-afb4-dd86c25c56c1\slide_05_adds_integration_1773936894342.png"
    W, H = 1920, 1080
    
    if os.path.exists(bg_path):
        canvas = Image.open(bg_path).convert('RGBA')
        canvas = canvas.resize((W, H), Image.LANCZOS)
        enhancer = ImageEnhance.Brightness(canvas)
        canvas = enhancer.enhance(1.05)
    else:
        canvas = Image.new('RGBA', (W, H), color=(250, 250, 252, 255))

    pane_w, pane_h = 900, 800
    px, py = W - pane_w - 100, (H - pane_h) // 2
    canvas = draw_glass_rect(canvas, [px, py, px+pane_w, py+pane_h], (255, 255, 255), None, alpha=235, width=0, radius=25)
    
    draw = ImageDraw.Draw(canvas)
    title_font = get_font(38, bold=True)
    sub_font = get_font(24, bold=False)
    point_font = get_font(26, bold=True)
    desc_font = get_font(20, bold=False)
    cx, cy = px + 60, py + 60
    
    draw.text((cx, cy), "05. ADDS 데이터 통합 아키텍처 및 정규화", font=title_font, fill=(34, 49, 63, 255))
    cy += 70
    draw.text((cx, cy), "상이한 비정형 데이터의 단일 텐서(1D Tensor) 파이프라인 정제", font=sub_font, fill=(108, 122, 137, 255))
    cy += 70
    draw.line((cx, cy, px + pane_w - 60, cy), fill=(200, 200, 200, 255), width=2)
    cy += 60
    
    draw.ellipse([cx, cy+15, cx+15, cy+30], fill=(241, 196, 15))
    draw.text((cx + 35, cy), "진료과별 비정형/정형 데이터의 정렬", font=point_font, fill=(44, 62, 80, 255))
    cy += 50
    draw.text((cx + 35, cy), "이미지(병리), 엑셀(임상/혈액결과) 등의 다른 포맷을 ADDS 코어 모델이 분석 가능한 규격 프레임으로 통일", font=desc_font, fill=(108, 122, 137, 255))
    cy += 80
    
    draw.ellipse([cx, cy+15, cx+15, cy+30], fill=(52, 152, 219))
    draw.text((cx + 35, cy), "환자 DB 내 결측치(Missing Data) AI 보간 체계", font=point_font, fill=(44, 62, 80, 255))
    cy += 50
    draw.text((cx + 35, cy), "현실적으로 완벽할 수 없는 환자 임상 기록의 결측치를 딥러닝 알고리즘으로 보간하여 데이터 완전성(Density) 100% 달성", font=desc_font, fill=(108, 122, 137, 255))
    cy += 80

    draw.ellipse([cx, cy+15, cx+15, cy+30], fill=(211, 84, 0))
    draw.text((cx + 35, cy), "Multi-Modal 융합 바이오마커 도출 엔진 연결", font=point_font, fill=(44, 62, 80, 255))
    cy += 50
    draw.text((cx + 35, cy), "정제된 데이터를 물리적 '단일' 데이터베이스로 합쳐, 기존에 없던 형태학+유전학 융합 수치를 새롭게 계산해냄", font=desc_font, fill=(108, 122, 137, 255))
    
    canvas.convert('RGB').save(out_path, quality=98)

if __name__ == "__main__":
    create_ppt_slide_05()
