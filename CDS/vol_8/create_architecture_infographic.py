import os
from PIL import Image, ImageDraw, ImageFont

def get_font(size, bold=False):
    fonts = ["malgunbd.ttf" if bold else "malgun.ttf", "applegothic.ttf"]
    for font_name in fonts:
        try:
            return ImageFont.truetype(font_name, size)
        except IOError:
            continue
    return ImageFont.load_default()

def draw_centered_text(draw, x_center, y, text, font, fill):
    if hasattr(draw, 'textbbox'):
        w = draw.textbbox((0, 0), text, font=font)[2]
    else:
        w = draw.textsize(text, font=font)[0]
    draw.text((x_center - w/2, y), text, font=font, fill=fill)

def create_architecture_infographic():
    output_dir = r"f:\ADDS\CDS"
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "SaMD_Architecture_Concept.png")
    
    W, H = 1920, 1080
    bg_color = (250, 252, 255)         
    canvas = Image.new('RGB', (W, H), color=bg_color)
    draw = ImageDraw.Draw(canvas)
    
    # Text fonts
    title_font = get_font(52, bold=True)
    subtitle_font = get_font(30, bold=False)
    block_title = get_font(36, bold=True)
    label_font = get_font(24, bold=True)
    body_font = get_font(22, bold=False)

    # 1. Header (Top 180px)
    primary_color = (15, 30, 60)
    draw.rectangle([0, 0, W, 180], fill=primary_color)
    draw_centered_text(draw, W/2, 45, "NanoBanana AI 독립형 소프트웨어 의료기기(SaMD) 상용화 아키텍처", title_font, (255, 255, 255))
    draw_centered_text(draw, W/2, 120, "식약처 인허가 로드맵 / 국제 표준 품질관리(QMS) / 사이버보안 통합 체계", subtitle_font, (180, 220, 255))
    
    # 2. Main Architecture Layout (Center block)
    # The whole system is enclosed in a "Security Shield" (Slide 14)
    sec_x1, sec_y1 = 150, 250
    sec_x2, sec_y2 = W - 150, H - 100
    sec_color = (52, 152, 219) # Blue
    
    # Background for security perimeter
    draw.rectangle([sec_x1, sec_y1, sec_x2, sec_y2], fill=(240, 248, 255), outline=sec_color, width=6)
    draw.rectangle([sec_x1, sec_y1, sec_x2, sec_y1+50], fill=sec_color)
    draw_centered_text(draw, W/2, sec_y1 + 10, "사이버보안 및 무결성 구간 (Closed Network / End-to-End Encryption)", get_font(26, bold=True), (255, 255, 255))
    
    # 3. Inside the Security perimeter is the QMS Domain (Slide 12)
    qms_x1, qms_y1 = sec_x1 + 60, sec_y1 + 100
    qms_x2, qms_y2 = sec_x2 - 60, sec_y2 - 60
    qms_color = (22, 160, 133) # Teal
    draw.rectangle([qms_x1, qms_y1, qms_x2, qms_y2], outline=qms_color, width=4)
    # Dashed effect for QMS boundary (representing process boundary)
    for px in range(qms_x1, qms_x2, 20):
        draw.line((px, qms_y1, px+10, qms_y1), fill=(250, 252, 255), width=4)
        draw.line((px, qms_y2, px+10, qms_y2), fill=(250, 252, 255), width=4)
    for py in range(qms_y1, qms_y2, 20):
        draw.line((qms_x1, py, qms_x1, py+10), fill=(250, 252, 255), width=4)
        draw.line((qms_x2, py, qms_x2, py+10), fill=(250, 252, 255), width=4)

    draw_centered_text(draw, W/2, qms_y1 - 35, "품질 관리 (QMS / GMP) 환경 - IEC 62304 & ISO 13485 (문서 통제 / CAPA 통제)", get_font(24, bold=True), qms_color)
    
    # 4. SaMD Core Pipeline (Slide 11) Inside QMS
    # Flow: EMR -> Fusion Engine -> SaMD Outputs -> Clinical Decision
    
    block_w = 320
    block_h = 240
    gap = 80
    
    # Block 1: Input Data
    b1_x, b1_y = qms_x1 + 60, qms_y1 + 60
    draw.rectangle([b1_x, b1_y, b1_x+block_w, b1_y+block_h], fill=(236, 240, 241), outline=(189, 195, 199), width=3)
    draw.rectangle([b1_x, b1_y, b1_x+block_w, b1_y+40], fill=(127, 140, 141))
    draw_centered_text(draw, b1_x + block_w/2, b1_y + 8, "병원 망 연동 (입력부)", label_font, (255, 255, 255))
    draw_centered_text(draw, b1_x + block_w/2, b1_y + 70, "\u2022 복부 조영 CT 영상", body_font, (40, 40, 40))
    draw_centered_text(draw, b1_x + block_w/2, b1_y + 110, "\u2022 디지털 병리 (WSI)", body_font, (40, 40, 40))
    draw_centered_text(draw, b1_x + block_w/2, b1_y + 150, "\u2022 환자 임상 EMR", body_font, (40, 40, 40))
    draw_centered_text(draw, b1_x + block_w/2, b1_y + 190, "(사용적합성 UI 적용)", get_font(20), (100, 100, 100))
    
    # Arrow 1 -> 2
    arr_y = b1_y + block_h/2
    draw.line((b1_x+block_w, arr_y, b1_x+block_w+gap, arr_y), fill=(52, 73, 94), width=6)
    draw.polygon([(b1_x+block_w+gap, arr_y), (b1_x+block_w+gap-15, arr_y-10), (b1_x+block_w+gap-15, arr_y+10)], fill=(52, 73, 94))

    # Block 2: AI Core (SaMD 3등급 인증 목표)
    b2_x, b2_y = b1_x + block_w + gap, b1_y - 30
    b2_h = block_h + 60
    draw.rectangle([b2_x, b2_y, b2_x+block_w, b2_y+b2_h], fill=(255, 255, 255), outline=(241, 196, 15), width=4) # Yellow outline
    # Header
    draw.rectangle([b2_x, b2_y, b2_x+block_w, b2_y+60], fill=(241, 196, 15)) # Yellow header
    draw_centered_text(draw, b2_x + block_w/2, b2_y + 15, "NanoBanana AI SaMD 엔진", block_title, (40, 40, 40)) # Dark text on yellow
    
    draw_centered_text(draw, b2_x + block_w/2, b2_y + 90, "[ 3등급 의료기기 파이프라인 ]", label_font, (211, 84, 0)) # Orange
    
    # Load and embed the Nano Banana image
    banana_path = r"C:\Users\brook\.gemini\antigravity\brain\574d389b-b6ed-48ef-afb4-dd86c25c56c1\nano_banana_1773911334370.png"
    if os.path.exists(banana_path):
        try:
            banana_img = Image.open(banana_path).convert("RGBA")
            # Resize
            bw, bh = 140, 140
            banana_img = banana_img.resize((bw, bh), Image.LANCZOS)
            # Create a white backdrop (or paste directly)
            canvas.paste(banana_img, (int(b2_x + block_w/2 - bw/2), int(b2_y + 115)), mask=banana_img.split()[3])
        except Exception as e:
            print("Banana err:", e)

    # Divider
    draw.line((b2_x+30, b2_y+260, b2_x+block_w-30, b2_y+260), fill=(200, 200, 200), width=2)
    draw_centered_text(draw, b2_x + block_w/2, b2_y + 270, "위험관리(ISO 14971)", get_font(20, bold=True), (211, 84, 0))

    # Arrow 2 -> 3
    draw.line((b2_x+block_w, arr_y, b2_x+block_w+gap, arr_y), fill=(52, 73, 94), width=6)
    draw.polygon([(b2_x+block_w+gap, arr_y), (b2_x+block_w+gap-15, arr_y-10), (b2_x+block_w+gap-15, arr_y+10)], fill=(52, 73, 94))

    # Block 3: Decision Support
    b3_x, b3_y = b2_x + block_w + gap, b1_y
    draw.rectangle([b3_x, b3_y, b3_x+block_w, b3_y+block_h], fill=(236, 240, 241), outline=(189, 195, 199), width=3)
    draw.rectangle([b3_x, b3_y, b3_x+block_w, b3_y+40], fill=(39, 174, 96))
    draw_centered_text(draw, b3_x + block_w/2, b3_y + 8, "다학제 임상 지원 (출력부)", label_font, (255, 255, 255))
    draw_centered_text(draw, b3_x + block_w/2, b3_y + 70, "\u2022 조기 진행 위험 (Risk Score)", body_font, (40, 40, 40))
    draw_centered_text(draw, b3_x + block_w/2, b3_y + 110, "\u2022 최적 처방 융합 권고", body_font, (40, 40, 40))
    draw_centered_text(draw, b3_x + block_w/2, b3_y + 150, "\u2022 3차원 영상 결과물", body_font, (40, 40, 40))
    draw_centered_text(draw, b3_x + block_w/2, b3_y + 190, "인증 가이드라인 기준 패키징", get_font(20), (100, 100, 100))

    # 5. Bottom descriptive badges/ribbons
    badge_w = 400
    badge_h = 100
    spaces = (W - (3 * badge_w)) / 4
    
    # 3 Badges summarizing the requirements
    reqs = [
        {"title": "01. 소프트웨어 품질(GMP)", "desc": "형상관리 및 버전통제 의무화"},
        {"title": "02. SaMD 등급 분류", "desc": "잠재위해성 높은 3등급"},
        {"title": "03. 사이버보안 승인", "desc": "개인정보 비식별 / E2E 암호화"}
    ]
    
    for i, req in enumerate(reqs):
        bx = spaces + i*(badge_w + spaces)
        by = qms_y2 + 40
        
        # Draw ribbon
        draw.rectangle([bx, by, bx+badge_w, by+badge_h], fill=(255, 255, 255), outline=(200, 200, 200), width=1)
        draw.rectangle([bx, by, bx+10, by+badge_h], fill=(44, 62, 80)) # Left accent
        
        draw_centered_text(draw, bx+badge_w/2, by + 20, req["title"], get_font(24, bold=True), (44, 62, 80))
        draw_centered_text(draw, bx+badge_w/2, by + 60, req["desc"], get_font(20), (100, 100, 100))

    canvas.save(out_path, quality=95)
    print(f"Architectural Infographic successfully generated at {out_path}")

if __name__ == "__main__":
    create_architecture_infographic()
