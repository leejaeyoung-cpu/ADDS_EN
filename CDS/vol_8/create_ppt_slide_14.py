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

def create_ppt_slide_14():
    out_dir = r"f:\ADDS\CDS"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "ADDS_PPT_Slide_14.png")
    
    bg_path = r"C:\Users\brook\.gemini\antigravity\brain\574d389b-b6ed-48ef-afb4-dd86c25c56c1\slide_14_expansion_white_1773940655418.png"
    
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
    
    # Header
    draw.text((cx, cy), "14. ADDS 플랫폼 기대 효과 및 수평적 확장(Scalability) 로드맵", font=title_font, fill=(34, 49, 63, 255))
    cy += 60
    draw.text((cx, cy), "본 대장암(CRC) 난제 해결을 넘어 범용 고형암/희귀 질환 정복을 위한 게임 체인저", font=sub_font, fill=(108, 122, 137, 255))
    cy += 70
    draw.line((cx, cy, px + pane_w - 60, cy), fill=(200, 200, 200, 255), width=2)
    cy += 50
    
    # Point 1: Expected Impact
    draw.ellipse([cx, cy+15, cx+15, cy+30], fill=(46, 204, 113)) # Green
    draw.text((cx + 35, cy), "압도적 임상 도입 기대 효과 (Immediate Clinical Impact)", font=point_font, fill=(44, 62, 80, 255))
    cy += 45
    draw.text((cx + 35, cy), "- [의료진]: 인간적 인지 한계(Human Error) 극복 및 분석 지연 시간 80% 이상 획기적 단축", font=desc_font, fill=(108, 122, 137, 255))
    cy += 40
    draw.text((cx + 35, cy), "- [환자]: 신속 명확한 1차 표적 처방 도출을 통해 환자 중심 생존율(OS) 극대화 보장", font=desc_font, fill=(108, 122, 137, 255))
    cy += 70
    
    # Point 2: Expansion
    draw.ellipse([cx, cy+15, cx+15, cy+30], fill=(41, 128, 185)) # Blue
    draw.text((cx + 35, cy), "타 고형암종으로의 즉각적 수평 확장 (Horizontal Expansion)", font=point_font, fill=(44, 62, 80, 255))
    cy += 45
    draw.text((cx + 35, cy), "- 대장암과 동일한 종양 메커니즘을 겪는 폐암, 췌장암, 위암 등 타 핵심 암종에 발사", font=desc_font, fill=(108, 122, 137, 255))
    cy += 40
    draw.text((cx + 35, cy), "- ADDS 17D 텐서 파이프라인의 가중치를 전이 학습(Transfer Learning)하여 모델 확장 비용 0화", font=desc_font, fill=(108, 122, 137, 255))
    cy += 70

    # Point 3: Future Prevision
    draw.ellipse([cx, cy+15, cx+15, cy+30], fill=(142, 68, 173)) # Purple
    draw.text((cx + 35, cy), "희귀 난치 질환 정복 및 예측 예방 의료(Preventive Medicine) 선도", font=point_font, fill=(44, 62, 80, 255))
    cy += 45
    draw.text((cx + 35, cy), "- 단일 질환 분석을 뛰어넘어 아직 정복되지 않은 복합 희귀 질환의 암표지자 패턴을 예측", font=desc_font, fill=(108, 122, 137, 255))
    cy += 40
    draw.text((cx + 35, cy), "- 장기적으로 글로벌 의료 시장 내 최고 수준의 '다중모달 병원 원격 협진(Tele-MDT)의 척도'로 시장 선점", font=desc_font, fill=(142, 68, 173, 255))
    
    canvas.convert('RGB').save(out_path, quality=98)
    print(f"PPT Slide 14 saved to {out_path}")

if __name__ == "__main__":
    create_ppt_slide_14()
