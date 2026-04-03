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

def create_ppt_slide_09():
    out_dir = r"f:\ADDS\CDS"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "ADDS_PPT_Slide_09.png")
    
    bg_path = r"C:\Users\brook\.gemini\antigravity\brain\574d389b-b6ed-48ef-afb4-dd86c25c56c1\slide_09_organoid_resistance_1773939871971.png"
    
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
    
    draw.text((cx, cy), "09. 환자 유래 오가노이드(PDO) 기반 맞춤 약물 검증", font=title_font, fill=(34, 49, 63, 255))  # Deep charcoal
    cy += 60
    draw.text((cx, cy), "생체 외(In vitro) 아바타 모델을 통한 항암/내성 예측 및 독성 회피", font=sub_font, fill=(108, 122, 137, 255))
    cy += 70
    draw.line((cx, cy, px + pane_w - 60, cy), fill=(200, 200, 200, 255), width=2)
    cy += 50
    
    # Point 1: Organoid
    draw.ellipse([cx, cy+15, cx+15, cy+30], fill=(46, 204, 113)) # Green
    draw.text((cx + 35, cy), "환자 맞춤형 3D 종양 오가노이드(PDO) 구축", font=point_font, fill=(44, 62, 80, 255))
    cy += 45
    draw.text((cx + 35, cy), "- 환자의 실제 종양 생체조직을 배양하여 체내 종양 미세환경(TME) 및", font=desc_font, fill=(108, 122, 137, 255))
    cy += 40
    draw.text((cx + 35, cy), "  증식 기전을 생체 외(In vitro)에서 완벽히 재현하는 아바타 생성", font=desc_font, fill=(108, 122, 137, 255))
    cy += 70
    
    # Point 2: Resistance Prediction
    draw.ellipse([cx, cy+15, cx+15, cy+30], fill=(211, 84, 0)) # Burnt Orange
    draw.text((cx + 35, cy), "항암 시너지 검증 및 획득 내성(Resistance) 사전 예측", font=point_font, fill=(44, 62, 80, 255))
    cy += 45
    draw.text((cx + 35, cy), "- 인간 중심 임상 전, 아바타 모델에 여러 표적 조합 약물을 시뮬레이션", font=desc_font, fill=(108, 122, 137, 255))
    cy += 40
    draw.text((cx + 35, cy), "- 초기 무진행 생존(PFS) 달성 이후, 암세포 유전체가 약물 내성을 획득하며", font=desc_font, fill=(108, 122, 137, 255))
    cy += 40
    draw.text((cx + 35, cy), "  우회 변이(Mutation)를 일으킬 특정 유전자를 선제적으로 예측 및 대비", font=desc_font, fill=(211, 84, 0, 255))
    cy += 70

    # Point 3: Toxicity Avoidance
    draw.ellipse([cx, cy+15, cx+15, cy+30], fill=(41, 128, 185)) # Deep Blue
    draw.text((cx + 35, cy), "최소 부작용·최대 시너지 '최적 스위트스팟' 도출", font=point_font, fill=(44, 62, 80, 255))
    cy += 45
    draw.text((cx + 35, cy), "- 다중 약제 조합 방식을 PDO 스크리닝 빅데이터와 ADDS 엔진으로 결합", font=desc_font, fill=(108, 122, 137, 255))
    cy += 40
    draw.text((cx + 35, cy), "- 치명적인 약물 독성(부작용)은 피하고 항암 효과는 가장 높게 발현되는", font=desc_font, fill=(108, 122, 137, 255))
    cy += 40
    draw.text((cx + 35, cy), "  가장 이상적인 약물 조합 처방만을 의사에게 최종 제시", font=desc_font, fill=(41, 128, 185, 255))
    
    canvas.convert('RGB').save(out_path, quality=98)
    print(f"PPT Slide 09 saved to {out_path}")

if __name__ == "__main__":
    create_ppt_slide_09()
