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

def create_ppt_slide_12():
    out_dir = r"f:\ADDS\CDS"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "ADDS_PPT_Slide_12.png")
    
    bg_path = r"C:\Users\brook\.gemini\antigravity\brain\574d389b-b6ed-48ef-afb4-dd86c25c56c1\slide_12_clinical_value_white_1773940315675.png"
    
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
    draw.text((cx, cy), "12. 다학제 진료(MDT) 내 ADDS 장착의 임상적 가치", font=title_font, fill=(34, 49, 63, 255))  # Deep charcoal
    cy += 60
    draw.text((cx, cy), "표적 시너지 극대화, 막대한 의료 비용 절감 및 환자의 본질적 삶의 질 보존", font=sub_font, fill=(108, 122, 137, 255))
    cy += 70
    draw.line((cx, cy, px + pane_w - 60, cy), fill=(200, 200, 200, 255), width=2)
    cy += 50
    
    # Point 1: Efficacy
    draw.ellipse([cx, cy+15, cx+15, cy+30], fill=(46, 204, 113)) # Green
    draw.text((cx + 35, cy), "최적의 치료 골든타임 확보 및 생존율(OS/PFS) 극대화", font=point_font, fill=(44, 62, 80, 255))
    cy += 45
    draw.text((cx + 35, cy), "- 다학제 위원회(MDT)의 융합 데이터 리뷰 및 합의 결정을 혁신적으로 단축", font=desc_font, fill=(108, 122, 137, 255))
    cy += 40
    draw.text((cx + 35, cy), "- AI 기반 맞춤 타겟팅으로 1차 포합 처방의 성공률 및 종양 억제력 급증", font=desc_font, fill=(46, 204, 113, 255))
    cy += 70
    
    # Point 2: Cost & Toxicity
    draw.ellipse([cx, cy+15, cx+15, cy+30], fill=(211, 84, 0)) # Burnt Orange
    draw.text((cx + 35, cy), "고독성 오남용 방지 및 사회적 의료 비용(Cost) 절감", font=point_font, fill=(44, 62, 80, 255))
    cy += 45
    draw.text((cx + 35, cy), "- 개별 환자에게 맞지 않는 고가 비효율 항암제의 맹목적 투여(Trial & Error)를", font=desc_font, fill=(108, 122, 137, 255))
    cy += 40
    draw.text((cx + 35, cy), "  AI가 원천 차단하여, 무의미하게 낭비되는 건강보험 재정 및 개인 부담금 구제", font=desc_font, fill=(108, 122, 137, 255))
    cy += 70

    # Point 3: QoL
    draw.ellipse([cx, cy+15, cx+15, cy+30], fill=(41, 128, 185)) # Deep Blue
    draw.text((cx + 35, cy), "환자의 신체적 고통 경감 및 본질적 삶의 질(QoL) 보존", font=point_font, fill=(44, 62, 80, 255))
    cy += 45
    draw.text((cx + 35, cy), "- 무분별한 병용의 독성으로 인한 피로, 탈모, 장기 손상 등 합병증 최소화", font=desc_font, fill=(108, 122, 137, 255))
    cy += 40
    draw.text((cx + 35, cy), "- 항암 투병 과정 중에도 환자가 무너지지 않고 '인간다운 일상(Daily Life)'을", font=desc_font, fill=(108, 122, 137, 255))
    cy += 40
    draw.text((cx + 35, cy), "  온전히 유지하도록 보호하는 진정한 목적의 정밀 의료(Precision) 실현", font=desc_font, fill=(41, 128, 185, 255))
    
    canvas.convert('RGB').save(out_path, quality=98)
    print(f"PPT Slide 12 saved to {out_path}")

if __name__ == "__main__":
    create_ppt_slide_12()
