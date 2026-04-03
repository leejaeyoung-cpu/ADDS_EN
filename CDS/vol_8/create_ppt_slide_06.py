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

def create_ppt_slide_06():
    out_dir = r"f:\ADDS\CDS"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "ADDS_PPT_Slide_06.png")
    bg_path = r"C:\Users\brook\.gemini\antigravity\brain\574d389b-b6ed-48ef-afb4-dd86c25c56c1\slide_06_precision_synergy_1773936910421.png"
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
    
    draw.text((cx, cy), "06. 다중모달 데이터 통합의 임상적 필수성", font=title_font, fill=(34, 49, 63, 255))
    cy += 70
    draw.text((cx, cy), "단일 데이터 한계 극복 및 고정밀 맞춤형 의료(Precision Medicine) 실현", font=sub_font, fill=(108, 122, 137, 255))
    cy += 70
    draw.line((cx, cy, px + pane_w - 60, cy), fill=(200, 200, 200, 255), width=2)
    cy += 60
    
    draw.ellipse([cx, cy+15, cx+15, cy+30], fill=(231, 76, 60)) # Red
    draw.text((cx + 35, cy), "1차원적 진단 오류 및 표적 실패의 근본적 극복", font=point_font, fill=(44, 62, 80, 255))
    cy += 50
    draw.text((cx + 35, cy), "단순 유전자(KRAS) 검사만으로는 파악할 수 없는 종양 미세환경 이질성을 형태학(병리)+유전학 결합으로 정확하게 판별", font=desc_font, fill=(108, 122, 137, 255))
    cy += 80
    
    draw.ellipse([cx, cy+15, cx+15, cy+30], fill=(46, 204, 113)) # Green
    draw.text((cx + 35, cy), "환자 맞춤형 복합 표적 치료 가이드라인 시뮬레이션", font=point_font, fill=(44, 62, 80, 255))
    cy += 50
    draw.text((cx + 35, cy), "통합 기반의 암 진행 스코어(Severity Score)를 통해 가장 효과적인 병용 약제와 투여 시점을 사전 예측 도출", font=desc_font, fill=(108, 122, 137, 255))
    cy += 80

    draw.ellipse([cx, cy+15, cx+15, cy+30], fill=(142, 68, 173)) # Purple
    draw.text((cx + 35, cy), "불필요한 투약 독성 회피 및 생존율(OS/PFS) 극대화", font=point_font, fill=(44, 62, 80, 255))
    cy += 50
    draw.text((cx + 35, cy), "효과 없는 고독성 항암제의 맹목적 투여를 방지(Toxicity 회피)하고, 개인별 최고의 표적 시너지(Synergy) 효과 달성 보장", font=desc_font, fill=(108, 122, 137, 255))
    
    cy += 100
    draw.rectangle([cx-15, cy, cx+8, cy+60], fill=(44, 62, 80)) # Subtle dark accent bar
    draw.text((cx + 25, cy + 5), "통합 솔루션: 이 모든 복합 추론을 17D AI Tensor 엔진으로 실시간\n해결하는 것이 바로 ADDS V1 Multi-Modal 플랫폼의 가치입니다.", font=get_font(20, bold=True), fill=(34, 49, 63, 255))

    canvas.convert('RGB').save(out_path, quality=98)

if __name__ == "__main__":
    create_ppt_slide_06()
