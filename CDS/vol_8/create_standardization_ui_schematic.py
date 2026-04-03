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

def draw_round_rect(draw, bounds, fill, outline=None, width=2, radius=15):
    draw.rounded_rectangle(bounds, radius=radius, fill=fill, outline=outline, width=width)

def draw_centered_text(draw, x_center, y, text, font, fill):
    if hasattr(draw, 'textbbox'):
        w = draw.textbbox((0, 0), text, font=font)[2]
    else:
        w = draw.textsize(text, font=font)[0]
    draw.text((x_center - w/2, y), text, font=font, fill=fill)

def create_schematic():
    out_dir = r"f:\ADDS\CDS"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "ADDS_V1_Data_Standardization_Viewer_Concept.png")
    
    W, H = 1920, 1080
    bg_color = (250, 252, 255)
    canvas = Image.new('RGB', (W, H), color=bg_color)
    draw = ImageDraw.Draw(canvas)
    
    title_font = get_font(48, bold=True)
    sub_font = get_font(28, bold=False)
    block_title = get_font(26, bold=True)
    body_font = get_font(20, bold=False)
    large_score_font = get_font(64, bold=True)

    # Main Header
    draw.rectangle([0, 0, W, 140], fill=(26, 42, 64))
    draw_centered_text(draw, W/2, 30, "ADDS. V1 다중모달 통합 데이터 표준화 및 임상의 뷰어(CDSS) 모식도", title_font, (255, 255, 255))
    draw_centered_text(draw, W/2, 90, "임상·CT·병리(ROI) 파이프라인 규격화부터 최종 예측 근거 시각화까지의 전주기 설계", sub_font, (180, 220, 255))
    
    # ------------------ LEFT SIDE: Data Pipeline ------------------
    left_cx = W * 0.28
    
    # 3 Input Sources
    i_w, i_h = 320, 100
    y_start = 220
    gap = 40
    
    inputs = [
        ("의료 데이터 (임상/분자)", "EMR 소견, KRAS, 바이오마커", (241, 196, 15)), # Yellow
        ("조영 CT 영상", "판독 소견 및 종양 용적 데이터", (231, 76, 60)), # Red
        ("디지털 병리 (WSI)", "자동 분할 정량 지표 및 형태학 ROI", (46, 204, 113)) # Green
    ]
    
    for i, (title, desc, color) in enumerate(inputs):
        bx = left_cx - i_w/2
        by = y_start + i*(i_h + gap)
        draw_round_rect(draw, [bx, by, bx+i_w, by+i_h], fill=(255, 255, 255), outline=color, width=3)
        draw.rectangle([bx, by, bx+i_w, by+8], fill=color)
        draw_centered_text(draw, left_cx, by + 30, title, block_title, (40, 40, 40))
        draw_centered_text(draw, left_cx, by + 65, desc, body_font, (100, 100, 100))
        
        # Arrow down
        arr_y = by + i_h
        if i < 2:
            pass # Arrows will converge

    # Central DB block
    db_w, db_h = 420, 120
    db_x = left_cx - db_w/2
    db_y = y_start + 3*(i_h + gap) + 20
    
    # Draw converging lines
    draw.line((left_cx, y_start+i_h+gap*2, left_cx, db_y), fill=(150, 150, 150), width=4)
    # Arrow head
    draw.polygon([(left_cx, db_y), (left_cx-10, db_y-15), (left_cx+10, db_y-15)], fill=(150, 150, 150))
    
    draw_round_rect(draw, [db_x, db_y, db_x+db_w, db_y+db_h], fill=(240, 248, 255), outline=(52, 152, 219), width=4)
    draw_centered_text(draw, left_cx, db_y + 20, "환자 ID (PT-ID) 연동 표준화 DB", get_font(28, bold=True), (41, 128, 185))
    draw_centered_text(draw, left_cx, db_y + 60, "각기계 이질적 포맷을 단일 JSON 텐서로 통합 적재", body_font, (40, 40, 40))
    draw_centered_text(draw, left_cx, db_y + 85, "(결측치 보간 및 ROI 정규화 처리)", body_font, (100, 100, 100))

    # Engine Block
    ai_w, ai_h = 420, 140
    ai_x = left_cx - ai_w/2
    ai_y = db_y + db_h + 80
    
    draw.line((left_cx, db_y+db_h, left_cx, ai_y), fill=(150, 150, 150), width=4)
    draw.polygon([(left_cx, ai_y), (left_cx-10, ai_y-15), (left_cx+10, ai_y-15)], fill=(150, 150, 150))
    
    draw_round_rect(draw, [ai_x, ai_y, ai_x+ai_w, ai_y+ai_h], fill=(41, 128, 185), outline=(31, 97, 141), width=4)
    draw_centered_text(draw, left_cx, ai_y + 25, "ADDS. V1 예측 모델 융합 추론", get_font(30, bold=True), (255, 255, 255))
    draw_centered_text(draw, left_cx, ai_y + 70, "1) 병리-영상-분자 17D 피쳐 벡터링 결합", body_font, (220, 240, 255))
    draw_centered_text(draw, left_cx, ai_y + 100, "2) 대규모 환자 데이터 재학습을 통한 성능 개선", body_font, (220, 240, 255))

    # Arrow from DB to UI
    arrow_x1 = left_cx + ai_w/2
    arrow_y = ai_y + ai_h/2
    ui_x_start = W * 0.52
    
    draw.line((arrow_x1, arrow_y, ui_x_start-20, arrow_y), fill=(52, 73, 94), width=8)
    draw.polygon([(ui_x_start, arrow_y), (ui_x_start-25, arrow_y-15), (ui_x_start-25, arrow_y+15)], fill=(52, 73, 94))
    
    draw_centered_text(draw, (arrow_x1 + ui_x_start)/2, arrow_y - 25, "결과 송출", block_title, (52, 73, 94))

    # ------------------ RIGHT SIDE: Presentation UI Mockup ------------------
    ui_w = 800
    ui_h = 820
    ui_y = 200
    ui_color = (255, 255, 255)
    shadow_color = (220, 220, 230)
    
    # Shadow
    draw_round_rect(draw, [ui_x_start+15, ui_y+15, ui_x_start+15+ui_w, ui_y+15+ui_h], fill=shadow_color, outline=None)
    # Device Frame
    draw_round_rect(draw, [ui_x_start, ui_y, ui_x_start+ui_w, ui_y+ui_h], fill=ui_color, outline=(200, 200, 200), width=2, radius=10)
    
    # UI Top Bar
    draw.rounded_rectangle([ui_x_start, ui_y, ui_x_start+ui_w, ui_y+60], radius=10, fill=(245, 245, 245))
    draw.rectangle([ui_x_start, ui_y+40, ui_x_start+ui_w, ui_y+60], fill=(245, 245, 245)) # remove bottom corners
    draw.line((ui_x_start, ui_y+60, ui_x_start+ui_w, ui_y+60), fill=(220, 220, 220), width=2)
    
    draw.ellipse([ui_x_start+20, ui_y+20, ui_x_start+35, ui_y+35], fill=(231, 76, 60))
    draw.ellipse([ui_x_start+45, ui_y+20, ui_x_start+60, ui_y+35], fill=(241, 196, 15))
    draw.ellipse([ui_x_start+70, ui_y+20, ui_x_start+85, ui_y+35], fill=(46, 204, 113))
    
    draw.text((ui_x_start+120, ui_y+15), "ADDS V1 Clinic Dashboard", font=get_font(22, bold=True), fill=(100, 100, 100))

    # Patient Header
    draw.rectangle([ui_x_start, ui_y+60, ui_x_start+ui_w, ui_y+130], fill=(236, 240, 241))
    draw.line((ui_x_start, ui_y+130, ui_x_start+ui_w, ui_y+130), fill=(189, 195, 199), width=2)
    draw.text((ui_x_start+40, ui_y+85), "환자 ID: PT - 10042", font=block_title, fill=(44, 62, 80))
    draw.text((ui_x_start+400, ui_y+90), "나이: 64 | 성별: M | 대장암 4기 | 항암 시작: 2026.03.11", font=body_font, fill=(100, 100, 100))

    # Output 1: Risk Score (0-100)
    score_y = ui_y + 160
    draw.text((ui_x_start+40, score_y), "다중모달 기반 최종 위험도 (Cancer Severity Score)", font=get_font(24, bold=True), fill=(44, 62, 80))
    
    score_cx = ui_x_start + 140
    score_cy = score_y + 110
    draw.arc([score_cx-70, score_cy-70, score_cx+70, score_cy+70], start=135, end=405, fill=(231, 76, 60), width=15) # Red arc
    draw_centered_text(draw, score_cx, score_cy-35, "87", large_score_font, (231, 76, 60))
    draw_centered_text(draw, score_cx, score_cy+30, "/ 100", block_title, (150, 150, 150))
    
    draw.text((score_cx+110, score_y+70), "예측 확률: 조기 진행(Progression) 위험 상위 13%", font=get_font(22, bold=True), fill=(231, 76, 60))
    draw.text((score_cx+110, score_y+105), "시스템 권고: 표준요법 불응성 극도의 위험군. 표적 항암제 또는 병용투여 전환 강력 권고.", font=body_font, fill=(40, 40, 40))

    # Output 2: Rationale (ROI/Indicators)
    ev_y = score_y + 220
    draw.text((ui_x_start+40, ev_y), "예측 판독 근거 (Representative Evidence & ROI)", font=get_font(24, bold=True), fill=(44, 62, 80))
    draw.line((ui_x_start+40, ev_y+40, ui_x_start+ui_w-40, ev_y+40), fill=(200, 200, 200), width=2)
    
    box_w = 340
    box_h = 240
    bx1 = ui_x_start + 40
    bx2 = bx1 + box_w + 40
    cy = ev_y + 60

    # Evidence A: CT ROI
    draw_round_rect(draw, [bx1, cy, bx1+box_w, cy+box_h], fill=(245, 245, 250), outline=(200, 200, 200), width=1, radius=8)
    # Placeholder CT Image Graphic
    draw.rectangle([bx1+80, cy+20, bx1+260, cy+140], fill=(40, 40, 40))
    draw.ellipse([bx1+150, cy+60, bx1+200, cy+100], outline=(231, 76, 60), width=3) # Bounding box/ROI
    draw.text((bx1+110, cy+70), "CT Slice", font=get_font(18), fill=(255, 255, 255))
    
    draw_centered_text(draw, bx1+box_w/2, cy+160, "[ 대표 CT 판독 ROI ]", get_font(20, bold=True), (44, 62, 80))
    draw_centered_text(draw, bx1+box_w/2, cy+190, "종양 체적(Volume) 32% 증가 검출", body_font, (100, 100, 100))
    draw_centered_text(draw, bx1+box_w/2, cy+215, "(Swin-UNETR 분할 지표 악화)", body_font, (150, 150, 150))

    # Evidence B: Pathology ROI
    draw_round_rect(draw, [bx2, cy, bx2+box_w, cy+box_h], fill=(245, 245, 250), outline=(200, 200, 200), width=1, radius=8)
    # Placeholder Pathology Image Graphic
    draw.rectangle([bx2+80, cy+20, bx2+260, cy+140], fill=(217, 136, 128)) # Pinkish tissue color
    # draw some cell masks
    for j in range(5):
        draw.ellipse([bx2+100+j*30, cy+40+(j%2)*20, bx2+115+j*30, cy+55+(j%2)*20], fill=(52, 152, 219))
    draw.rectangle([bx2+90, cy+30, bx2+250, cy+110], outline=(46, 204, 113), width=3) # ROI Focus
    
    draw_centered_text(draw, bx2+box_w/2, cy+160, "[ 대표 병리 세포 형태 ROI ]", get_font(20, bold=True), (44, 62, 80))
    draw_centered_text(draw, bx2+box_w/2, cy+190, "암세포 밀집도 및 비대칭 편차 64% 상승", body_font, (100, 100, 100))
    draw_centered_text(draw, bx2+box_w/2, cy+215, "(Cellpose 핵 분해 알고리즘 산출)", body_font, (150, 150, 150))

    # Evidence C: Clinical / Molecular Text Alert
    cl_y = cy + box_h + 30
    draw_round_rect(draw, [bx1, cl_y, bx2+box_w, cl_y+80], fill=(253, 235, 208), outline=(243, 156, 18), width=2, radius=8)
    draw.text((bx1+20, cl_y+25), "⚠️ 주요 분자 진단 편차 (Molecular Evidence):", font=get_font(22, bold=True), fill=(211, 84, 0))
    draw.text((bx1+500, cl_y+27), "'KRAS G12D 변이 양성' 으로 인한 높은 재발 확률(P<0.05) 보정됨", font=body_font, fill=(135, 54, 0))

    canvas.save(out_path, quality=95)
    print(f"Standardization and UI schematic saved to {out_path}")

if __name__ == "__main__":
    create_schematic()
