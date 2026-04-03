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

def create_ppt_slide_11():
    out_dir = r"f:\ADDS\CDS"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "ADDS_PPT_Slide_11.png")
    
    bg_path = r"C:\Users\brook\.gemini\antigravity\brain\574d389b-b6ed-48ef-afb4-dd86c25c56c1\slide_11_concrete_reports_white_1773940168463.png"
    
    W, H = 1920, 1080
    
    if os.path.exists(bg_path):
        canvas = Image.open(bg_path).convert('RGBA')
        canvas = canvas.resize((W, H), Image.LANCZOS)
        enhancer = ImageEnhance.Brightness(canvas)
        canvas = enhancer.enhance(1.05)
    else:
        canvas = Image.new('RGBA', (W, H), color=(250, 250, 252, 255))
        print("Background not found, using solid soft white.")

    pane_w = 1100  # Wide pane to fit text
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
    
    draw.text((cx, cy), "11. ADDS 구동 시뮬레이션: 타겟별 맞춤 결과 리포트 (UI/UX)", font=title_font, fill=(34, 49, 63, 255))
    cy += 60
    draw.text((cx, cy), "치료 효과(Synergy) 극대화를 위한 임상의-환자 쌍방향 의사결정 지원 리포팅", font=sub_font, fill=(108, 122, 137, 255))
    cy += 70
    draw.line((cx, cy, px + pane_w - 60, cy), fill=(200, 200, 200, 255), width=2)
    cy += 50
    
    # Clinician Side
    draw.ellipse([cx, cy+15, cx+15, cy+30], fill=(41, 128, 185)) # Blue
    draw.text((cx + 35, cy), "임상의(Doctor) 전용 실무 CDSS 대시보드 출력창 예시", font=point_font, fill=(44, 62, 80, 255))
    cy += 45
    draw.text((cx + 35, cy), "▶ [다중모달 융합 점수]: \"환자 PT-093, CT+위험도 25% + KRAS[G12D] = 총합 88점(위험)\"", font=desc_font, fill=(108, 122, 137, 255))
    cy += 40
    draw.text((cx + 35, cy), "▶ [표적 약물 추천]: \"1순위 약물A(MEK) + B(EGFR) 병용처방 시 사전 모델 시너지 92% 달성\"", font=desc_font, fill=(108, 122, 137, 255))
    cy += 40
    draw.text((cx + 35, cy), "▶ [의학적 근거 제공]: 분석된 원본 H&E 슬라이드 ROI 핫스팟 및 글로벌 생존 코호트 신뢰도 팝업", font=desc_font, fill=(41, 128, 185, 255))
    cy += 70

    # Patient Side
    draw.ellipse([cx, cy+15, cx+15, cy+30], fill=(46, 204, 113)) # Green
    draw.text((cx + 35, cy), "환자(Patient) 소통용 직관적 인포그래픽 리포트 예시", font=point_font, fill=(44, 62, 80, 255))
    cy += 45
    draw.text((cx + 35, cy), "▶ [현재 상태 요약]: \"환자분 암세포의 45%에서 발견된 변이로 기존 A약물의 효과 저하 위기\"", font=desc_font, fill=(108, 122, 137, 255))
    cy += 40
    draw.text((cx + 35, cy), "▶ [치료 결정 이득]: \"제시된 새 병용(A+B) 적용 시, 독성 부작용은 줄이면서 종양 억제 확률이", font=desc_font, fill=(108, 122, 137, 255))
    cy += 40
    draw.text((cx + 35, cy), "                         기존 C약물 단독 50% 수준 대비 안전하게 2.5배 연장될 가능성이 높습니다.\"", font=desc_font, fill=(108, 122, 137, 255))
    cy += 40
    draw.text((cx + 35, cy), "▶ [시각적 안심 효과]: 복잡한 p-value 수치 대신 '신호등 색상 등급'화 하여 치료 순응도 극대화", font=desc_font, fill=(46, 204, 113, 255))
    
    canvas.convert('RGB').save(out_path, quality=98)
    print(f"PPT Slide 11 saved to {out_path}")

if __name__ == "__main__":
    create_ppt_slide_11()
