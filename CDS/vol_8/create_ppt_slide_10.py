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

def create_ppt_slide_10():
    out_dir = r"f:\ADDS\CDS"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "ADDS_PPT_Slide_10.png")
    
    bg_path = r"C:\Users\brook\.gemini\antigravity\brain\574d389b-b6ed-48ef-afb4-dd86c25c56c1\slide_10_pdo_validation_white_1773940021544.png"
    
    W, H = 1920, 1080
    
    if os.path.exists(bg_path):
        canvas = Image.open(bg_path).convert('RGBA')
        canvas = canvas.resize((W, H), Image.LANCZOS)
        enhancer = ImageEnhance.Brightness(canvas)
        canvas = enhancer.enhance(1.05)
    else:
        canvas = Image.new('RGBA', (W, H), color=(250, 250, 252, 255))
        print("Background not found, using solid soft white.")

    pane_w = 1100
    pane_h = 800
    px = W - pane_w - 40
    py = (H - pane_h) // 2
    
    canvas = draw_glass_rect(canvas, [px, py, px+pane_w, py+pane_h], (255, 255, 255), None, alpha=235, width=0, radius=25)
    
    draw = ImageDraw.Draw(canvas)
    
    title_font = get_font(38, bold=True)
    sub_font = get_font(24, bold=False)
    point_font = get_font(26, bold=True)
    desc_font = get_font(20, bold=False)
    
    cx = px + 60
    cy = py + 60
    
    draw.text((cx, cy), "10. PDO 실증 검증 및 ADDS 선순환 성능 고도화", font=title_font, fill=(34, 49, 63, 255))
    cy += 60
    draw.text((cx, cy), "후향적(Retrospective) 임상 정밀 데이터를 통한 모델 교차 검토(Cross-Validation)", font=sub_font, fill=(108, 122, 137, 255))
    cy += 70
    draw.line((cx, cy, px + pane_w - 60, cy), fill=(200, 200, 200, 255), width=2)
    cy += 50
    
    # Point 1: Validation
    draw.ellipse([cx, cy+15, cx+15, cy+30], fill=(211, 84, 0)) # Burnt Orange
    draw.text((cx + 35, cy), "ADDS 예측 모델의 타당성(Validity) 교차 검증", font=point_font, fill=(44, 62, 80, 255))
    cy += 45
    draw.text((cx + 35, cy), "- AI 엔진이 선제 예측한 '최적 약물 조합 스코어'와", font=desc_font, fill=(108, 122, 137, 255))
    cy += 40
    draw.text((cx + 35, cy), "  실제 체외 오가노이드(PDO) 배양에서 확인된 항암/효능 실험 결과를", font=desc_font, fill=(108, 122, 137, 255))
    cy += 40
    draw.text((cx + 35, cy), "  1:1로 직접 매칭하여 강력한 예측 일치도(Concordance) 증명", font=desc_font, fill=(211, 84, 0, 255))
    cy += 70
    
    # Point 2: Retrospective Matching
    draw.ellipse([cx, cy+15, cx+15, cy+30], fill=(41, 128, 185)) # Deep Blue
    draw.text((cx + 35, cy), "후향적 임상 빅데이터(Retrospective Data) 매칭", font=point_font, fill=(44, 62, 80, 255))
    cy += 45
    draw.text((cx + 35, cy), "- 병원 내 축적된 수만 건의 과거 환자 처방 및 예후(PFS/OS) 데이터를", font=desc_font, fill=(108, 122, 137, 255))
    cy += 40
    draw.text((cx + 35, cy), "  사후 AI 시뮬레이션 결과와 역추적 대조(Back-testing)", font=desc_font, fill=(108, 122, 137, 255))
    cy += 40
    draw.text((cx + 35, cy), "- 실제 적용에 앞서 AI 알고리즘의 임상적 오류 제로화 및 안전성 완벽 확보", font=desc_font, fill=(108, 122, 137, 255))
    cy += 70

    # Point 3: Upgrade
    draw.ellipse([cx, cy+15, cx+15, cy+30], fill=(46, 204, 113)) # Green
    draw.text((cx + 35, cy), "연속적 자기 주도 성능 업그레이드(Self-Optimization) 루프", font=point_font, fill=(44, 62, 80, 255))
    cy += 45
    draw.text((cx + 35, cy), "- AI 예측 값과 체외/임상 실측 값 사이의 오차율(Loss)을 지속 반영", font=desc_font, fill=(108, 122, 137, 255))
    cy += 40
    draw.text((cx + 35, cy), "- 신규 약물이나 변이에 대한 예측 정밀도를 스스로 영구적으로 보정 및 상승시킴", font=desc_font, fill=(46, 204, 113, 255))
    
    canvas.convert('RGB').save(out_path, quality=98)
    print(f"PPT Slide 10 saved to {out_path}")

if __name__ == "__main__":
    create_ppt_slide_10()
