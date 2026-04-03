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

def create_ppt_slide_13():
    out_dir = r"f:\ADDS\CDS"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "ADDS_PPT_Slide_13.png")
    
    bg_path = r"C:\Users\brook\.gemini\antigravity\brain\574d389b-b6ed-48ef-afb4-dd86c25c56c1\slide_13_samd_security_white_1773940537978.png"
    
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
    draw.text((cx, cy), "13. ADDS 상용화 로드맵: 파일럿 실증 및 SaMD 인증", font=title_font, fill=(34, 49, 63, 255))
    cy += 60
    draw.text((cx, cy), "안전한 실제 임상 현장의 타당성 확보 및 글로벌 의료기기 규제/보안 완벽 대응", font=sub_font, fill=(108, 122, 137, 255))
    cy += 70
    draw.line((cx, cy, px + pane_w - 60, cy), fill=(200, 200, 200, 255), width=2)
    cy += 50
    
    # Point 1: Pilot
    draw.ellipse([cx, cy+15, cx+15, cy+30], fill=(41, 128, 185)) # Deep Blue
    draw.text((cx + 35, cy), "원내 파일럿 테스트(Pilot Test) 기반 사용성(Usability) 실증", font=point_font, fill=(44, 62, 80, 255))
    cy += 45
    draw.text((cx + 35, cy), "- 인하대병원 등 핵심 거점 파트너십을 통한 다학제 종양내과 실근무 투입 시뮬레이션", font=desc_font, fill=(108, 122, 137, 255))
    cy += 40
    draw.text((cx + 35, cy), "- 집중적인 UI/UX 직관성 검토 및 의료진 피드백 루프를 통한 '실제 임상 사용 가능성(Feasibility)' 타진", font=desc_font, fill=(108, 122, 137, 255))
    cy += 70
    
    # Point 2: SaMD
    draw.ellipse([cx, cy+15, cx+15, cy+30], fill=(46, 204, 113)) # Green
    draw.text((cx + 35, cy), "소프트웨어 의료기기(SaMD) 인허가 및 GMP 고도화 구축", font=point_font, fill=(44, 62, 80, 255))
    cy += 45
    draw.text((cx + 35, cy), "- 식약처(MFDS) 가이드라인에 완전히 부합하는 확증 임상 시험 인허가(IND/NDA) 쾌속 트랙 준비", font=desc_font, fill=(108, 122, 137, 255))
    cy += 40
    draw.text((cx + 35, cy), "- 의료기기 수준의 강력한 품질경영시스템(ISO 13485) 및 체계적인 GMP 필수 기반 문서 시스템 내재화 완료", font=desc_font, fill=(108, 122, 137, 255))
    cy += 70

    # Point 3: Security
    draw.ellipse([cx, cy+15, cx+15, cy+30], fill=(142, 68, 173)) # Purple
    draw.text((cx + 35, cy), "민감 의료 정보 보호를 위한 무결점 사이버 보안(Cybersecurity)", font=point_font, fill=(44, 62, 80, 255))
    cy += 45
    draw.text((cx + 35, cy), "- 이기종 다중 클라우드 연동 시 환자 개인정보 식별 비식별화(De-identification) 및 최고 수준 보안 규격 달성", font=desc_font, fill=(108, 122, 137, 255))
    cy += 40
    draw.text((cx + 35, cy), "- 랜섬웨어 원천 방어 및 엔드투엔드(E2E) 암호화를 거친 데이터 결합 아키텍처로 무결성(Integrity) 절대 보장", font=desc_font, fill=(108, 122, 137, 255))
    
    canvas.convert('RGB').save(out_path, quality=98)
    print(f"PPT Slide 13 saved to {out_path}")

if __name__ == "__main__":
    create_ppt_slide_13()
