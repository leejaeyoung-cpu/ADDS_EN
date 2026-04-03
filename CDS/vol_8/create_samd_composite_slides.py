import os
import glob
from PIL import Image, ImageDraw, ImageFont

def get_font(size):
    try:
        return ImageFont.truetype("malgun.ttf", size)
    except IOError:
        try:
            return ImageFont.truetype("arial.ttf", size)
        except IOError:
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

def create_composites():
    input_dir = r"C:\Users\brook\.gemini\antigravity\brain\44a9df00-4e33-4924-9829-577663bc3d3e"
    output_dir = "f:\\ADDS\\CDS\\final_pitch_slides"
    os.makedirs(output_dir, exist_ok=True)
    
    # Original research diagrams
    kakao_1 = "f:\\ADDS\\CDS\\KakaoTalk_20260318_110529150.png"
    kakao_2 = "f:\\ADDS\\CDS\\KakaoTalk_20260318_153749133.png"
    
    # Mapping image numbers (1 to 15) to text content
    content_map = {
        "01": {"title": "임상 미충족 수요 (Unmet Need)", "sub": "KRAS 환자 반응 이질성과 예측 한계", "bullet": "대장암 타겟 치료 시 환자마다 치료 반응이 상이합니다.\n조기 진행 위험(HR)을 예측하여 표적 치료가 무의미한 환자를 선별하는 기술이 시급합니다."},
        "02": {"title": "현재 진단 데이터의 한계", "sub": "분절된 임상, 특수 병리, 영상 정보", "bullet": "CT 영상과 디지털 병리 결과(WSI), 분자 검사(ctDNA)가 서로 연계되지 않고 단순 저장되어, 통합적인 항암 치료 효과 예측이 불가합니다."},
        "03": {"title": "ADDS V1 통합 결점 시스템", "sub": "다중모달 기반 정밀 치료 보조", "bullet": "병리조직(WSI), 조영 CT영상, 임상정보를 하나의 딥러닝 벡터로 연결하여 위험도를 통합 예측하는 솔루션(ADDS V1)을 제안합니다.", "embed": kakao_1},
        "04": {"title": "디지털 병리 정량 및 지표 추출", "sub": "Cellpose 기반 정밀 영상 분할", "bullet": "암 종양 절편 슬라이드(H&E)에서 AI 파이프라인을 구동하여\nN/C Ratio, 핵의 형태학적 왜곡 정도를 실시간 정량화합니다."},
        "05": {"title": "전신 CT 영상의학 AI 분석", "sub": "3D 종양 분할 및 체적(Volume) 평가", "bullet": "치료 전/후 복부 CT에서 Bounding Box와 Segmentation을 통해 HU 분표밀도와 종양 용적 변화율의 예측 지표화를 달성합니다."},
        "06": {"title": "ADDS V1 임상 결과 대시보드", "sub": "Risk Score & Therapy Recommendation", "bullet": "진단된 결과는 임상의에게 0-100점 척도의 조기진행위험 예측도와 함께 '최적의 병합 및 항암 요법 가이드라인'으로 직관적으로 제공됩니다."},
        "07": {"title": "체외 검증 - 환자유래 오가노이드(PDO)", "sub": "실험실 내 모델 유효도 검증", "bullet": "AI 알고리즘의 신뢰성을 위해 2차년도에는 환자 종양에서 유래된 오가노이드 배양과 로보틱스 기반 IC50 약물 스크리닝을 진행하여 실증적 교차 검토를 실시합니다."},
        "08": {"title": "약물 내성 바이오마커 발굴", "sub": "PrPc 및 ctDNA VAF 추적", "bullet": "항암제 불응성의 주요 요인으로 추정되는 PrPc 발현량 및 ctDNA 잔여 수치를 기반으로 치료 중 내성을 실시간 조기 예측합니다."},
        "09": {"title": "멀티모달 통합 Fusion 엔진", "sub": "다차원 변수 퓨전 딥러닝", "bullet": "1단계 딥러닝 Feature Extraction과 2단계 Random Forest/Gradient Boost 기반 분류기를 결합하는 다계층 앙상블 아키텍처를 도입했습니다."},
        "10": {"title": "3년 주기의 임상 실증 파이프라인", "sub": "후향적 코호트부터 전향적 임상 파일럿까지", "bullet": "과제 3년 차에 걸쳐 후향적 데이터 분석, PDO 체외 검토, 확증을 위한 소규모 전향적 파일럿 검증이라는 다계층 임상 실증 로드맵을 운영합니다."},
        "11": {"title": "독립형 시스템 (SaMD) 제품화 전환", "sub": "연구 성과의 성공적인 인허가 브릿징", "bullet": "연구용 모델로 끝나지 않습니다.\nADDS V1은 식약처 인허가 가이드라인에 맞춘 단계적 제품화 체계를 적용하여 '의료기기'로 승인받기 위해 설계되었습니다.", "embed": kakao_2},
        "12": {"title": "소프트웨어 제조 및 품질관리 (QMS/GMP)", "sub": "IEC 62304 / ISO 13485 규격", "bullet": "환자 생명과 직결되는 소프트웨어인 만큼 엄격한 문서 통제, 형상 관리, 오류 CAPA 처리 및 ISO 14971 기반의 소프트웨어 단독 위해 점검을 내재화합니다."},
        "13": {"title": "사용적합성(Usability) 엔지니어링", "sub": "인적 오류 예방을 위한 UI/UX 설계", "bullet": "임상의의 피로도와 잦은 알람으로 인한 오독을 방지하기 위하여 IEC 62366 기반으로 조작 동선을 최적화하고 명확한 경고 시각화(Risk Color Code)를 달성합니다."},
        "14": {"title": "사이버보안 및 데이터 무결성", "sub": "병원 외부 유출을 막는 클로즈드 설계", "bullet": "민감한 EMR 및 임상 레코드 보호를 최우선으로, 종단간 암호화와 병원 내부 망 기반의 터널링 방벽을 준수하는 가장 강력한 보안을 적용합니다."},
        "15": {"title": "최종 비전: 다학제(MDT) 파트너 ADDS V1", "sub": "불필요한 항암은 멈추고 생환율은 높이다", "bullet": "우리 연구의 끝 지점은 종양내과, 병리과 등 다학제 진료 테이블 위에서 확신에 찬 결정을 내릴 수 있도록 돕는 전방위 임상 지원 의료기술의 도입입니다."}
    }

    # Fetch all AI generated images
    images = sorted(glob.glob(os.path.join(input_dir, "medical_story_*.png")))
    if not images:
        print(f"No original images found in {input_dir}")
        return

    W, H = 1920, 1080
    bg_color = (250, 250, 250)
    text_color = (40, 40, 40)
    primary_color = (18, 35, 66)  # Dark Blue
    accent_color = (25, 160, 160) # Teal
    
    font_title = get_font(64)
    font_subtitle = get_font(40)
    font_body = get_font(32)

    for i, img_path in enumerate(images):
        key = f"{i+1:02d}"
        data = content_map.get(key)
        
        if not data:
            continue
            
        canvas = Image.new('RGB', (W, H), color=bg_color)
        draw = ImageDraw.Draw(canvas)
        
        # Load left AI image
        original_img = Image.open(img_path)
        img_w, img_h = original_img.size
        # Resize to fit height, preserving aspect ratio (height 800)
        new_h = 800
        new_w = int(img_w * (new_h / img_h))
        original_img = original_img.resize((new_w, new_h), Image.LANCZOS)
        
        pad_x, pad_y = 100, 140
        canvas.paste(original_img, (pad_x, pad_y))
        
        # Draw top bar numbering
        draw.text((pad_x, 50), f"Part {key} / 15", font=get_font(28), fill=accent_color)
        
        # Text positioning
        text_x = pad_x + new_w + 120
        max_text_width = W - text_x - 100
        current_y = pad_y
        
        # Title
        draw.text((text_x, current_y), data["title"], font=font_title, fill=primary_color)
        current_y += 100
        
        # Subtitle
        draw.text((text_x, current_y), data["sub"], font=font_subtitle, fill=accent_color)
        current_y += 80
        
        # Line
        draw.line((text_x, current_y, text_x + max_text_width, current_y), fill=(200, 200, 200), width=4)
        current_y += 60
        
        # Bullet
        lines = data["bullet"].split('\n')
        for line in lines:
            wrapped_lines = wrap_text(line, font_body, max_text_width, draw)
            for w_line in wrapped_lines:
                draw.text((text_x, current_y), w_line, font=font_body, fill=text_color)
                current_y += 50
            current_y += 20
            
        # Composite original data (Kakao images) if available
        embed_path = data.get("embed")
        if embed_path and os.path.exists(embed_path):
            current_y += 40
            try:
                emb_img = Image.open(embed_path).convert("RGBA")
                # Resize embed image to fit bottom right
                emb_w, emb_h = emb_img.size
                target_w = max_text_width
                target_h = int(emb_h * (target_w / emb_w))
                # Check height overflow
                if current_y + target_h > H - 100:
                    target_h = H - current_y - 100
                    target_w = int(emb_w * (target_h / emb_h))
                
                emb_img = emb_img.resize((target_w, target_h), Image.LANCZOS)
                
                # Create white layer for background of transparency and paste it
                white_backdrop = Image.new("RGB", emb_img.size, (255, 255, 255))
                white_backdrop.paste(emb_img, mask=emb_img.split()[3])
                
                canvas.paste(white_backdrop, (text_x, current_y))
                
                # Small border around embedded image
                draw.rectangle([text_x-2, current_y-2, text_x+target_w+2, current_y+target_h+2], outline=accent_color, width=2)
            except Exception as e:
                print(f"Error embedding {embed_path}: {e}")
            
        out_name = f"Slide_{key}_Composite.png"
        out_path = os.path.join(output_dir, out_name)
        canvas.save(out_path, quality=95)
        print(f"Generated Slide {key}: {out_name}")

if __name__ == '__main__':
    create_composites()
