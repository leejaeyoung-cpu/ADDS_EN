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

def draw_centered_text(draw, x_center, y, text, font, fill):
    if hasattr(draw, 'textbbox'):
        w = draw.textbbox((0, 0), text, font=font)[2]
    else:
        w = draw.textsize(text, font=font)[0]
    draw.text((x_center - w/2, y), text, font=font, fill=fill)

def draw_glass_rect(canvas, bounds, fill_color, outline_color, alpha=180, width=2, radius=15):
    overlay = Image.new('RGBA', canvas.size, (255, 255, 255, 0))
    d = ImageDraw.Draw(overlay)
    x1, y1, x2, y2 = bounds
    d.rounded_rectangle(bounds, radius=radius, fill=fill_color + (alpha,))
    if outline_color:
        d.rounded_rectangle(bounds, radius=radius, outline=outline_color + (255,), width=width)
    canvas.paste(Image.alpha_composite(canvas.convert('RGBA'), overlay))
    return canvas

def create_glass_architecture():
    output_dir = r"f:\ADDS\CDS"
    out_path = os.path.join(output_dir, "ADDS_V1_Architecture_Final.png")
    
    bg_path = r"C:\Users\brook\.gemini\antigravity\brain\574d389b-b6ed-48ef-afb4-dd86c25c56c1\ai_tech_background_1773911716030.png"
    
    W, H = 1920, 1080
    
    if os.path.exists(bg_path):
        canvas = Image.open(bg_path).convert('RGBA')
        canvas = canvas.resize((W, H), Image.LANCZOS)
        enhancer = ImageEnhance.Brightness(canvas)
        canvas = enhancer.enhance(0.85)
    else:
        canvas = Image.new('RGBA', (W, H), color=(10, 20, 40, 255))
        
    title_font = get_font(52, bold=True)
    subtitle_font = get_font(28, bold=False)
    block_title = get_font(30, bold=True)
    label_font = get_font(22, bold=True)
    body_font = get_font(20, bold=False)

    draw = ImageDraw.Draw(canvas)
    main_title = "ADDS. V1 의료기기(SaMD) 상용화 아키텍처"
    draw_centered_text(draw, W/2, 40, main_title, title_font, (255, 255, 255, 255))
    draw_centered_text(draw, W/2, 110, "식약처 인허가 로드맵 / 국제 표준 품질관리(QMS) / 다중모달 융합(Multimodal Fusion)", subtitle_font, (200, 230, 255, 255))
    
    sec_x1, sec_y1 = 80, 180
    sec_x2, sec_y2 = W - 80, H - 50
    canvas = draw_glass_rect(canvas, [sec_x1, sec_y1, sec_x2, sec_y2], (10, 40, 80), (100, 200, 255), alpha=80, width=3, radius=20)
    
    qms_x1, qms_y1 = sec_x1 + 40, sec_y1 + 80
    qms_x2, qms_y2 = sec_x2 - 40, sec_y2 - 50
    canvas = draw_glass_rect(canvas, [qms_x1, qms_y1, qms_x2, qms_y2], (0, 0, 0), (80, 220, 180), alpha=60, width=2, radius=15)

    draw_final = ImageDraw.Draw(canvas)
    
    draw_centered_text(draw_final, W/2, sec_y1 + 25, "Layer 3. 사이버 보안 / 허가 가이드라인 무결성 (Closed-Network & End-to-End Encryption)", label_font, (150, 220, 255, 255))
    draw_centered_text(draw_final, W/2, qms_y1 + 25, "Layer 2. 소프트웨어 제조 및 품질관리 체계 (QMS / GMP / ISO 13485)", label_font, (150, 255, 200, 255))

    # Center Blocks
    # Made block wider to fit specific technical texts
    block_w, block_h = 440, 480
    gap = 80
    
    b1_x = qms_x1 + 60
    b1_y = qms_y1 + 90
    
    b2_x = b1_x + block_w + gap
    b2_y = b1_y - 20
    b2_h = block_h + 40
    
    b3_x = b2_x + block_w + gap
    b3_y = b1_y
    
    # 3 Layers boxes
    canvas = draw_glass_rect(canvas, [b1_x, b1_y, b1_x+block_w, b1_y+block_h], (255, 255, 255), (200, 200, 200), alpha=180, radius=15)
    canvas = draw_glass_rect(canvas, [b2_x, b2_y, b2_x+block_w, b2_y+b2_h], (255, 250, 200), (255, 215, 0), alpha=200, width=4, radius=15)
    canvas = draw_glass_rect(canvas, [b3_x, b3_y, b3_x+block_w, b3_y+block_h], (255, 255, 255), (200, 200, 200), alpha=180, radius=15)
    
    draw_final = ImageDraw.Draw(canvas)
    dark_text = (30, 30, 30, 255)
    
    # ======== INPUT DATA (Block 1) ========
    draw_final.rounded_rectangle([b1_x, b1_y, b1_x+block_w, b1_y+60], fill=(44, 62, 80, 255), radius=15)
    draw_centered_text(draw_final, b1_x + block_w/2, b1_y + 15, "임상 입력 데이터 (H&E 및 분자)", block_title, (255, 255, 255, 255))
    
    inputs = [
        "• 환자 EMR 및 복부 CT 영상",
        "• H&E 디지털 병리 슬라이드",
        "• 분자 진단 플랫폼 데이터",
        "  └ KRAS 돌연변이 (Mutation)",
        "  └ NRAS / BRAF 변이",
        "  └ MSI / MMR 상태"
    ]
    cy = b1_y + 90
    for ipt in inputs:
        # Indent sub items
        font_to_use = body_font if "└" in ipt else label_font
        x_off = 30 if "└" in ipt else 0
        draw_final.text((b1_x + 30 + x_off, cy), ipt, font=font_to_use, fill=dark_text)
        cy += 45
        
    draw_centered_text(draw_final, b1_x + block_w/2, b1_y + block_h - 70, "[ 병원망 분리 연동 ]", get_font(20, bold=True), (100, 100, 100, 255))

    # ======== ADDS V1 SaMD ENGINE (Block 2) ========
    draw_final.rounded_rectangle([b2_x, b2_y, b2_x+block_w, b2_y+60], fill=(241, 196, 15, 255), radius=15)
    draw_centered_text(draw_final, b2_x + block_w/2, b2_y + 15, "ADDS. V1 AI 엔진 구동 (다중모달 융합)", block_title, (40, 40, 40, 255))
    
    draw_centered_text(draw_final, b2_x + block_w/2, b2_y + 80, "Layer 1. 3등급 의료기기 코어 모델", get_font(20, bold=True), (211, 84, 0, 255))
    
    core_items = [
        "Cellpose 기반 형태학적 분석",
        "└ H&E 결과 및 세포 형태 추출",
        "└ 분자(KRAS 등) 변이별 세포 양상 계량화",
        "Swin-UNETR: 종양 용적 변화 분할",
        "멀티모달 진단 (Multi-Modal Fusion)",
        "└ 위 항목들을 통합 수치화/벡터화"
    ]
    cy = b2_y + 120
    for item in core_items:
        if "└" not in item:
            draw_final.ellipse([b2_x + 30, cy + 10, b2_x + 40, cy + 20], fill=(211, 84, 0, 255))
            draw_final.text((b2_x + 55, cy), item, font=label_font, fill=dark_text)
            cy += 40
        else:
            draw_final.text((b2_x + 70, cy), item, font=body_font, fill=(80, 80, 80, 255))
            cy += 35
        
    draw_final.rounded_rectangle([b2_x + 40, cy + 30, b2_x + block_w - 40, b2_y + b2_h - 40], fill=(255, 255, 255, 180), outline=(241, 196, 15, 255), width=2, radius=10)
    draw_centered_text(draw_final, b2_x + block_w/2, cy + 45, "암 진행 정도(Severity) 스코어 계산", label_font, dark_text)
    draw_centered_text(draw_final, b2_x + block_w/2, cy + 85, "환자 맞춤형 치료 효용성 분석", body_font, dark_text)

    # ======== OUTPUT DECISION SUPPORT (Block 3) ========
    draw_final.rounded_rectangle([b3_x, b3_y, b3_x+block_w, b3_y+60], fill=(39, 174, 96, 255), radius=15)
    draw_centered_text(draw_final, b3_x + block_w/2, b3_y + 15, "임상 의료 결정 시각화", block_title, (255, 255, 255, 255))
    
    outputs = [
        "• 통합 조기 진행 위험 (Risk Score)",
        "• KRAS 변이 특이적 치료 병합 권고",
        "  (표적 항암 효과 예측)",
        "• 환자 의료 결정 영향 시뮬레이션",
        "• 다학제 시스템 리포트 출력"
    ]
    cy = b3_y + 90
    for opt in outputs:
        if "(" in opt or "  " in opt:
            draw_final.text((b3_x + 50, cy), opt, font=body_font, fill=(80, 80, 80, 255))
            cy += 40
        else:
            draw_final.text((b3_x + 30, cy), opt, font=label_font, fill=dark_text)
            cy += 50
        
    draw_centered_text(draw_final, b3_x + block_w/2, b3_y + block_h - 70, "[ IEC 62366 기반 안전 UI/UX ]", get_font(20, bold=True), (100, 100, 100, 255))

    # Draw Arrows Connecting Them
    arr1_y = b1_y + block_h/2
    draw_final.line((b1_x+block_w, arr1_y, b2_x, arr1_y), fill=(255, 255, 255, 150), width=6)
    
    arr2_y = b2_y + b2_h/2
    draw_final.line((b2_x+block_w, arr2_y, b3_x, arr2_y), fill=(255, 255, 255, 150), width=6)

    # Save
    canvas.convert('RGB').save(out_path, quality=98)
    print(f"Updated Advanced Architectural Logic Infographic saved to {out_path}")

if __name__ == "__main__":
    create_glass_architecture()
