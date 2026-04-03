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

def create_ppt_slide_08():
    out_dir = r"f:\ADDS\CDS"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "ADDS_PPT_Slide_08.png")
    
    bg_path = r"C:\Users\brook\.gemini\antigravity\brain\574d389b-b6ed-48ef-afb4-dd86c25c56c1\slide_08_output_framework_white_1773939373271.png"
    
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
    draw.text((cx, cy), "08. ADDS 솔루션 결과 도출의 3원화(Tripartite)", font=title_font, fill=(34, 49, 63, 255))  # Deep charcoal
    cy += 60
    draw.text((cx, cy), "다중모달 17D 텐서 분석을 통한 약물 예측 및 계층적 리포팅", font=sub_font, fill=(108, 122, 137, 255))
    cy += 70
    draw.line((cx, cy, px + pane_w - 60, cy), fill=(200, 200, 200, 255), width=2)
    cy += 50
    
    # Point 1: Drug Prediction
    draw.ellipse([cx, cy+15, cx+15, cy+30], fill=(211, 84, 0)) # Burnt Orange
    draw.text((cx + 35, cy), "최적의 표적 병용 약물 조합(Combo-Synergy) 예측", font=point_font, fill=(44, 62, 80, 255))
    cy += 45
    draw.text((cx + 35, cy), "- 단일 효과를 넘어 KRAS 변이 억제율 및 약물 간 화학적 상호작용 계산", font=desc_font, fill=(108, 122, 137, 255))
    cy += 40
    draw.text((cx + 35, cy), "- 독성(Toxicity)이 최소화되고 효과가 극대화된 환자 맞춤형 복합 제안", font=desc_font, fill=(108, 122, 137, 255))
    cy += 70
    
    # Point 2: Clinician Info
    draw.ellipse([cx, cy+15, cx+15, cy+30], fill=(41, 128, 185)) # Deep Blue
    draw.text((cx + 35, cy), "임상의(Doctor) 전용 고도화된 CDSS 대시보드 표출", font=point_font, fill=(44, 62, 80, 255))
    cy += 45
    draw.text((cx + 35, cy), "- 병리 WSI 밀집도(ROI) 및 CT 종양 체적 등 다중모달 AI 분석의", font=desc_font, fill=(108, 122, 137, 255))
    cy += 40
    draw.text((cx + 35, cy), "  명확한 근거(Evidence)를 중증도(Severity) 컷오프 알림과 함께 즉시 렌더링", font=desc_font, fill=(41, 128, 185, 255))
    cy += 70

    # Point 3: Patient Report
    draw.ellipse([cx, cy+15, cx+15, cy+30], fill=(46, 204, 113)) # Green
    draw.text((cx + 35, cy), "환자(Patient) 맞춤의 직관적 시각화 안심 리포트 제공", font=point_font, fill=(44, 62, 80, 255))
    cy += 45
    draw.text((cx + 35, cy), "- 의사소통 장벽을 낮추기 위해 난해한 의학 텍스트를 배제", font=desc_font, fill=(108, 122, 137, 255))
    cy += 40
    draw.text((cx + 35, cy), "- AI가 도출한 '해당 치료법 선택 시 상대적 생존(PFS) 기대 이득'을\n  인포그래픽화 하여 환자의 치료 순응도(Compliance) 극대화", font=desc_font, fill=(46, 204, 113, 255))
    
    canvas.convert('RGB').save(out_path, quality=98)
    print(f"PPT Slide 08 saved to {out_path}")

if __name__ == "__main__":
    create_ppt_slide_08()
