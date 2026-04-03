import os
import glob
from PIL import Image, ImageDraw, ImageFont

def get_font(size):
    # Malgun Gothic is available on standard Windows systems
    try:
        return ImageFont.truetype("malgun.ttf", size)
    except IOError:
        # Fallback if not found
        try:
            return ImageFont.truetype("arial.ttf", size)
        except IOError:
            return ImageFont.load_default()

def wrap_text(text, font, max_width, draw):
    # simple text wrap
    lines = []
    if not text:
        return lines
    words = text.split(' ')
    current_line = words[0]
    for word in words[1:]:
        # Pillow < 10 uses textsize, >= 10 uses textbbox
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

def create_explained_figures():
    input_dir = "f:\\ADDS_CDS_PART2\\PPT_Figures\\1"
    output_dir = "f:\\ADDS_CDS_PART2\\PPT_Figures_With_Text"
    os.makedirs(output_dir, exist_ok=True)
    
    # Mapping keywords in filenames to text data
    content_map = {
        "ppt_01": {
            "title": "ADDS. V1: 정밀 항암 의료기술",
            "subtitle": "[도입] KRAS 변이 대장암 맞춤형 다중모달 통합 솔루션",
            "bullet": "본 연구는 표준 치료법에 반응하지 않는 이질적인 KRAS 변이 대장암 환자들을 위해, 임상·병리·화상 데이터를 아우르는 통합 AI 의사결정지원 시스템(ADDS. V1) 실용화를 목표로 합니다."
        },
        "ppt_02": {
            "title": "미충족 의료 수요 (Unmet Need)",
            "subtitle": "[문제 제기] 복잡한 종양 미세환경과 표준치료의 한계",
            "bullet": "• 단일 표적 항암제(EGFR 저해제 등)의 저항성 문제\n• 환자 간 치료 반응률 편차 극심\n• 막대한 치료 비용 대비 낮은 조기 진행(초기 저항) 예측률"
        },
        "ppt_03": {
            "title": "기존 진단/생검의 한계",
            "subtitle": "[한계점] 거대 종양의 공간적 이질성",
            "bullet": "수술 전 실시하는 국소적인 단일 조직 생검만으로는 거대한 암 덩어리가 가지는 공간적인 세포 이질성을 완벽하게 반영하지 못하여, 정확한 표적 항암 치료 여부를 판별하기 부족합니다."
        },
        "ppt_04": {
            "title": "우리의 해결책: ADDS. V1",
            "subtitle": "[솔루션] 임상 의사결정을 돕는 첨단 AI 플랫폼",
            "bullet": "복잡도 높은 다중모달 분석 알고리즘을 블랙박스 형태가 아닌, 의사들에게 직관적인 UI(위험도 점수, 추천 요법, XAI 히트맵)로 돌려주는 독립형 소프트웨어 서비스입니다."
        },
        "ppt_05": {
            "title": "다중모달 데이터 통합 AI",
            "subtitle": "[핵심 기술 1] 입체적인 환자 데이터 수집 체계",
            "bullet": "• 임상 정보(EMR/혈액 검사)\n• 방사선 판독 정보(CT 영상 기반 병기)\n• 분자진단 (ctDNA, 돌연변이 프로파일)\n이 모든 정보를 환자 ID 기반으로 연동하여 빅데이터 체계를 구축합니다."
        },
        "ppt_06": {
            "title": "디지털 병리(WSI) 딥러닝 분석",
            "subtitle": "[핵심 기술 2] H&E 및 IHC 슬라이드 정량화",
            "bullet": "Cellpose 알고리즘 등을 활용하여 암조직 슬라이드 상의 수많은 세포와 핵을 정밀하게 자동 분할(Segmentation)하고, 면역화학염색(IHC) 발현 강도를 픽셀 단위로 수치화합니다."
        },
        "ppt_07": {
            "title": "PrPᶜ 바이오마커 기전",
            "subtitle": "[차별점] 신규 종양 인자 발굴",
            "bullet": "PrPᶜ 단백질은 기존 항암 약물 저항성을 촉진하며 특히 주변 혈관으로의 미세 침윤(Microvascular Invasion)을 주도하는 것으로 밝혀져, 본 모델의 핵심 예측 지표로 포함됩니다."
        },
        "ppt_08": {
            "title": "환자 위험도 계층화(Stratification)",
            "subtitle": "[임상 효용 1] 조기 진행 예측 모델",
            "bullet": "각종 입력 변수를 종합하여, 항암 치료 시작 전 해당 환자가 치료에 불응하여 조기 진행할 확률(High Risk vs Low Risk)을 산출하고 카플란-마이어 생존 곡선 지표를 개선합니다."
        },
        "ppt_09": {
            "title": "환자 유래 오가노이드(PDO) 배양",
            "subtitle": "[검증 1] 체외(Ex Vivo) 실증 기반 구축",
            "bullet": "AI 모델이 내린 가설과 예측 결과를 단순히 SW 안에서 끝내지 않고, 실제 해당 환자의 조직에서 유래한 미니 장기(오가노이드)를 배양하여 신뢰도를 극대화합니다."
        },
        "ppt_10": {
            "title": "로보틱 약물 스크리닝 (Screening)",
            "subtitle": "[검증 2] 체외 약물 반응성 테스트 (IC50)",
            "bullet": "배양된 PDO에 AI가 추천한 단독 또는 병용 항암제를 투여한 뒤, 로보틱 분석 기법을 통해 실제 암세포 생존율과 약물 반응성(IC50)을 정량 평가하여 모델을 재학습시킵니다."
        },
        "ppt_11": {
            "title": "다학제(MDT) 진료 보조 리포트",
            "subtitle": "[임상 적용] 의사결정의 강력한 동반자",
            "bullet": "종합적인 분석을 마친 AI는 다학제 진료 현장에서 의사들이 참고할 수 있는 최종 리포트를 출력하여, '가장 유효성이 높은 Top1 약물 조합'을 자신 있게 제안합니다."
        },
        "ppt_12": {
            "title": "디지털 의료기기(SaMD) 상용화 단계",
            "subtitle": "[로드맵] 연구 프로토타입을 넘어선 진화",
            "bullet": "본 과제는 단순히 논문 게재가 목표가 아닙니다.\n• 1차년: 대상/시점 확정\n• 2차년: 제조/품질 통제(QS) 도입\n• 3차년: 식품의약품안전처 품목 허가 심사 추진"
        },
        "ppt_13": {
            "title": "검증 및 밸리데이션(V&V) 체계",
            "subtitle": "[품질 보증] 소프트웨어 기술문서 패키징",
            "bullet": "SaMD 허가를 위해 필수적인 요구사항 명세, 핵심 알고리즘 설명서, 데이터셋 정의서 및 철저한 위험관리(Risk) 파일을 구축하여 공인된 시험성적서를 도출합니다."
        },
        "ppt_14": {
            "title": "사용적합성 및 사이버 보안",
            "subtitle": "[보안 및 안전] 민감 의료 정보 완벽 통제",
            "bullet": "병원 진료망 도입 시 발생할 수 있는 데이터 유출 방지를 위해 종단 간 암호화를 적용하며, 의료진의 입력 오류를 최소화하는 사용적합성(Usability) 설계를 우선합니다."
        },
        "ppt_15": {
            "title": "정밀의료 확장 및 최종 비전",
            "subtitle": "[최종 비전] 불필요한 항암 축소 및 생존율 개선",
            "bullet": "결과적으로 고통스러운 무성 항암 치료를 최소화하고, 환자 개인에게 가장 적합한 약물을 조기에 투여함으로써 치료 비용 절감과 함께 생존율과 삶의 질 극대화를 도모합니다."
        }
    }

    # Fetch all AI images
    images = glob.glob(os.path.join(input_dir, "ppt_*.png"))
    if not images:
        print(f"No original images found in {input_dir}")
        return

    # Dimensions for output figure (1920x1080 for 16:9 presentation)
    W, H = 1920, 1080
    bg_color = (250, 250, 250) # Light gray/white
    text_color = (30, 30, 30)
    primary_color = (0, 75, 150) # Deep Blue
    
    font_title = get_font(72)
    font_subtitle = get_font(48)
    font_body = get_font(40)

    for img_path in images:
        filename = os.path.basename(img_path)
        key = filename[:6] # "ppt_01"
        data = content_map.get(key)
        
        if not data:
            continue
            
        # Create blank canvas
        canvas = Image.new('RGB', (W, H), color=bg_color)
        draw = ImageDraw.Draw(canvas)
        
        # Load and resize left-side image (AI generated)
        original_img = Image.open(img_path)
        # Resize to fit height, preserving aspect ratio
        img_w, img_h = original_img.size
        new_h = 900
        new_w = int(img_w * (new_h / img_h))
        original_img = original_img.resize((new_w, new_h), Image.LANCZOS)
        
        # Paste image on the left, with some padding
        pad_x, pad_y = 100, 90
        canvas.paste(original_img, (pad_x, pad_y))
        
        # Text positioning (Right side)
        text_x = pad_x + new_w + 100
        max_text_width = W - text_x - 100
        current_y = pad_y + 100
        
        # Draw Title
        draw.text((text_x, current_y), data["title"], font=font_title, fill=primary_color)
        current_y += 120
        
        # Draw Subtitle
        draw.text((text_x, current_y), data["subtitle"], font=font_subtitle, fill=(0, 150, 136)) # Teal
        current_y += 100
        
        # Draw Separator Line
        draw.line((text_x, current_y, text_x + max_text_width, current_y), fill=(200, 200, 200), width=3)
        current_y += 80
        
        # Draw Bullet Points
        lines = data["bullet"].split('\n')
        for line in lines:
            wrapped_lines = wrap_text(line, font_body, max_text_width, draw)
            for w_line in wrapped_lines:
                draw.text((text_x, current_y), w_line, font=font_body, fill=text_color)
                current_y += 60
            current_y += 20 # slight padding between paragraphs
            
        out_path = os.path.join(output_dir, f"{key}_explained.png")
        canvas.save(out_path)
        print(f"Created: {out_path}")

if __name__ == '__main__':
    create_explained_figures()
