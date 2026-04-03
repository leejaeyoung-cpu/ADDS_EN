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

def wrap_text(text, font, max_width, draw):
    lines = []
    if not text:
        return lines
    words = text.split(' ')
    current_line = words[0]
    for word in words[1:]:
        if hasattr(draw, 'textbbox'):
            w = draw.textbbox((0, 0), current_line + ' ' + word, font=font)[2]
        else:
            w, h = draw.textsize(current_line + ' ' + word, font=font)
        
        if w <= max_width:
            current_line += ' ' + word
        else:
            lines.append(current_line)
            current_line = word
    if current_line:
        lines.append(current_line)
    return lines

def draw_icon_samd(draw, x, y, size, color):
    # Abstract representation of medical device (Cross and Plus)
    cx, cy = x + size//2, y + size//2
    # Background circle
    draw.ellipse([x, y, x+size, y+size], fill=(240, 248, 255))
    # Outer ring
    draw.arc([x+5, y+5, x+size-5, y+size-5], 0, 360, fill=color, width=4)
    # Plus sign
    w = 15
    draw.rectangle([cx-w, cy-w*3, cx+w, cy+w*3], fill=color)
    draw.rectangle([cx-w*3, cy-w, cx+w*3, cy+w], fill=color)

def draw_icon_qms(draw, x, y, size, color):
    # Abstract document and gear
    cx, cy = x + size//2, y + size//2
    # Document background
    draw.rectangle([x+10, y+10, x+size-20, y+size-20], fill=(240, 248, 255), outline=color, width=4)
    # Lines
    draw.line((x+25, y+30, x+size-35, y+30), fill=color, width=3)
    draw.line((x+25, y+45, x+size-35, y+45), fill=color, width=3)
    draw.line((x+25, y+60, x+size-55, y+60), fill=color, width=3)
    # A stamp/gear overlapping
    draw.ellipse([x+size-45, y+size-45, x+size-5, y+size-5], fill=color)
    draw.ellipse([x+size-35, y+size-35, x+size-15, y+size-15], fill=(240, 248, 255))

def draw_icon_security(draw, x, y, size, color):
    # Shield icon
    cx, cy = x + size//2, y + size//2
    # Background
    # Shield shape points
    pts = [
        (cx, y+5), 
        (x+size-15, y+20), 
        (x+size-15, cy), 
        (cx, y+size-10), 
        (x+15, cy), 
        (x+15, y+20)
    ]
    draw.polygon(pts, fill=(255, 245, 230))
    # Outline
    draw.line([pts[0], pts[1], pts[2], pts[3], pts[4], pts[5], pts[0]], fill=color, width=4)
    # Inner lines or check
    draw.line((x+30, cy, cx-5, cy+20), fill=color, width=6)
    draw.line((cx-5, cy+20, x+size-25, y+30), fill=color, width=6)

def create_infographic():
    output_dir = r"f:\ADDS\CDS"
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "SaMD_Certification_Infographic_Simple.png")
    
    W, H = 1920, 1080
    bg_color = (245, 247, 250)         
    primary_color = (20, 40, 80)       
    secondary_color = (25, 160, 160)   
    card_bg = (255, 255, 255)          
    text_dark = (40, 40, 40)
    
    canvas = Image.new('RGB', (W, H), color=bg_color)
    draw = ImageDraw.Draw(canvas)
    
    title_font = get_font(56, bold=True)
    subtitle_font = get_font(32, bold=False)
    card_title_font = get_font(34, bold=True)
    card_sub_font = get_font(24, bold=False)
    bullet_font = get_font(28, bold=False)
    
    header_height = 250
    draw.rectangle([0, 0, W, header_height], fill=primary_color)
    
    main_title = "ADDS. V1 독립형 소프트웨어 의료기기(SaMD) 제품화 로드맵"
    if hasattr(draw, 'textbbox'):
        tw = draw.textbbox((0, 0), main_title, font=title_font)[2]
    else:
        tw = draw.textsize(main_title, font=title_font)[0]
    draw.text(((W - tw)/2, 60), main_title, font=title_font, fill=(255, 255, 255))
    
    main_sub = "연구 모델을 넘어, 신뢰할 수 있는 임상 지원 의료기술의 도입"
    if hasattr(draw, 'textbbox'):
        sw = draw.textbbox((0, 0), main_sub, font=subtitle_font)[2]
    else:
        sw = draw.textsize(main_sub, font=subtitle_font)[0]
    draw.text(((W - sw)/2, 160), main_sub, font=subtitle_font, fill=(180, 220, 255))
    
    cards = [
        {
            "step": "Phase 1",
            "title": "SaMD 상용화 전환",
            "sub": "식약처 인허가 가이드라인 브릿징",
            "bullets": [
                "연구용 모델 → 3등급 소프트웨어 의료기기",
                "맞춤형 치료 전략 객관적 증거 제시",
                "엄격한 임상 근거를 통한 인허가 확보"
            ],
            "color": (46, 134, 193),
            "icon_func": draw_icon_samd
        },
        {
            "step": "Phase 2",
            "title": "안전 및 품질 관리",
            "sub": "IEC 62304 / ISO 13485 규격 내재화",
            "bullets": [
                "국제 표준 기준 GMP (품질경영) 적용",
                "지속적 형상 관리, 통제된 문서 배포",
                "ISO 14971 기반 자율적 오류 제어 (CAPA)"
            ],
            "color": (25, 160, 160),
            "icon_func": draw_icon_qms
        },
        {
            "step": "Phase 3",
            "title": "클로즈드 보안 설계",
            "sub": "EMR 유출 방지 및 조작 편의성 (UX)",
            "bullets": [
                "환자 민감 임상/EMR 레코드 원천 보호",
                "병원 내부망 망분리 및 종단간(E2E) 암호화",
                "IEC 62366 기반 인간공학적 안전 설계"
            ],
            "color": (211, 84, 0),
            "icon_func": draw_icon_security
        }
    ]
    
    card_y = 310
    card_h = 650
    card_w = 540
    spacing = 60
    start_x = (W - (card_w * 3 + spacing * 2)) / 2
    
    for i, c in enumerate(cards):
        x = start_x + i * (card_w + spacing)
        
        # Shadow
        shadow_offset = 12
        draw.rectangle([x+shadow_offset, card_y+shadow_offset, x+card_w+shadow_offset, card_y+card_h+shadow_offset], fill=(225, 225, 230), outline=None)
        
        # Main Card
        draw.rectangle([x, card_y, x+card_w, card_y+card_h], fill=card_bg, outline=(210, 210, 210), width=1)
        draw.rectangle([x, card_y, x+card_w, card_y+15], fill=c["color"])
        
        cx = x + 40
        cy = card_y + 40
        
        # Step
        draw.text((cx, cy), f"{c['step']}", font=get_font(20, bold=True), fill=c["color"])
        cy += 35
        
        # Title
        draw.text((cx, cy), c["title"], font=card_title_font, fill=primary_color)
        cy += 50
        
        # Subtitle
        draw.text((cx, cy), c["sub"], font=card_sub_font, fill=c["color"])
        cy += 40
        
        # Draw Icon (Centered in a block)
        icon_size = 120
        c["icon_func"](draw, x + (card_w - icon_size)//2, cy + 20, icon_size, c["color"])
        
        cy += 180
        draw.line((cx, cy, x + card_w - 40, cy), fill=(230, 230, 230), width=2)
        cy += 40
        
        # Bullets
        for bullet in c["bullets"]:
            # Draw bullet point
            draw.ellipse([cx, cy + 12, cx + 10, cy + 22], fill=c["color"])
            
            wrapped_lines = wrap_text(bullet, bullet_font, card_w - 90, draw)
            bullet_y = cy
            for w_line in wrapped_lines:
                draw.text((cx + 25, bullet_y), w_line, font=bullet_font, fill=text_dark)
                bullet_y += 40
            cy = bullet_y + 20
            
    footer_text = "Clinical Pathology Platform - 의료기기 인증 파이프라인"
    if hasattr(draw, 'textbbox'):
        fw = draw.textbbox((0, 0), footer_text, font=get_font(20))[2]
    else:
        fw = draw.textsize(footer_text, font=get_font(20))[0]
    draw.text(((W - fw)/2, 1030), footer_text, font=get_font(20), fill=(160, 160, 160))
    
    canvas.save(out_path, quality=95)
    print(f"Infographic successfully generated at {out_path}")

if __name__ == "__main__":
    create_infographic()
