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

def create_ppt_slide_04():
    out_dir = r"f:\ADDS\CDS"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "ADDS_PPT_Slide_04.png")
    bg_path = r"C:\Users\brook\.gemini\antigravity\brain\574d389b-b6ed-48ef-afb4-dd86c25c56c1\slide_04_data_collection_1773936881621.png"
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
    
    draw.text((cx, cy), "04. 선행연구 기반 멀티모달 데이터 수집", font=title_font, fill=(34, 49, 63, 255))
    cy += 70
    draw.text((cx, cy), "병원 내 산재된 다차원 데이터의 '환자 중심' 1차 결합", font=sub_font, fill=(108, 122, 137, 255))
    cy += 70
    draw.line((cx, cy, px + pane_w - 60, cy), fill=(200, 200, 200, 255), width=2)
    cy += 60
    
    draw.ellipse([cx, cy+15, cx+15, cy+30], fill=(41, 128, 185))
    draw.text((cx + 35, cy), "환자 고유 ID(PT-ID) 중심의 전주기 매핑", font=point_font, fill=(44, 62, 80, 255))
    cy += 50
    draw.text((cx + 35, cy), "병리과, 진단검사의학과, 종양내과 등 부서별 고립 데이터를 환자 기준으로 통합 호출", font=desc_font, fill=(108, 122, 137, 255))
    cy += 80
    
    draw.ellipse([cx, cy+15, cx+15, cy+30], fill=(46, 204, 113))
    draw.text((cx + 35, cy), "다차원 의료 정보의 포괄적 획득 단계", font=point_font, fill=(44, 62, 80, 255))
    cy += 50
    draw.text((cx + 35, cy), "1) H&E WSI 병리조직 지표 (형태학, 군집도 등)", font=desc_font, fill=(108, 122, 137, 255))
    cy += 40
    draw.text((cx + 35, cy), "2) 핵심 혈액 검사 및 특정 분자 진단 마커 결과", font=desc_font, fill=(108, 122, 137, 255))
    cy += 80

    draw.ellipse([cx, cy+15, cx+15, cy+30], fill=(230, 126, 34))
    draw.text((cx + 35, cy), "치료 단계별(Phase) 임상 차트 타임라인 구축", font=point_font, fill=(44, 62, 80, 255))
    cy += 50
    draw.text((cx + 35, cy), "내원진단 ➔ 외과적 수술 ➔ 항암 투여 ➔ 예우/추적 등 시계열 데이터 트래킹 보존", font=desc_font, fill=(108, 122, 137, 255))
    
    canvas.convert('RGB').save(out_path, quality=98)

if __name__ == "__main__":
    create_ppt_slide_04()
