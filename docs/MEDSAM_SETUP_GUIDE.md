# MedSAM 설치 및 다운로드 가이드

## ✅ 완료된 단계

1. ✅ segment-anything 패키지 설치됨
2. ✅ MedSAM detector 코드 작성 완료
3. ✅ models/ 디렉토리 생성

## 📥 SAM Weights 다운로드

### Option 1: ViT-H (추천) - 2.4GB
**최고 성능, RTX 5070으로 충분**

```bash
cd C:\Users\brook\Desktop\ADDS
curl -L -o models/sam_vit_h_4b8939.pth https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
```

### Option 2: ViT-L (중간) - 1.2GB
**빠름, 성능 약간 낮음**

```bash
cd C:\Users\brook\Desktop\ADDS
curl -L -o models/sam_vit_l_0b3195.pth https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth
```

### Option 3: ViT-B (가벼움) - 375MB
**가장 빠름, 성능 제한적**

```bash
cd C:\Users\brook\Desktop\ADDS
curl -L -o models/sam_vit_b_01ec64.pth https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth
```

## 🧪 테스트 실행

다운로드 완료 후:

```bash
python src/medical_imaging/medsam_detector.py
```

예상 출력:
```
Using device: cuda
Loading SAM model from models/sam_vit_h_4b8939.pth...
SAM model loaded successfully!

Loading CT slice: 20050.dcm
Image shape: (512, 512, 3)
Device: cuda

Testing with center point: [256 256]
Running inference...

Results:
  Masks generated: 3
  Scores: [0.98, 0.95, 0.89]
  Best score: 0.980

Visualization saved: CTdata/visualizations/medsam_test_result.png
```

## 📊 다음 단계

1. **테스트 결과 확인**
   - visualization 이미지 검토
   - GPU 메모리 사용량 확인

2. **Annotation Interface 생성**
   - Streamlit 기반 클릭 인터페이스
   - 종양 위치 표시
   - MedSAM 자동 segmentation

3. **5-10개 슬라이스 Annotation**
   - 명확한 종양 슬라이스 선택
   - 포인트만 클릭하면 자동 분할

4. **Fine-tuning (Optional)**
   - Annotated data로 학습
   - 정확도 향상

## 💡 Tip

**curl이 없다면:**
- 웹브라우저로 URL 직접 다운로드
- models/ 폴더에 저장

**GPU 메모리 부족 시:**
- ViT-L 또는 ViT-B 사용
- Batch size 줄이기
