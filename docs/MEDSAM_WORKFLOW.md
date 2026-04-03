# MedSAM 워크플로우 및 다음 단계

## 📥 현재 Status

### ✅ 완료
1. segment-anything 설치
2. MedSAM detector 코드 작성
3. Streamlit annotation tool 생성
4. models/ 디렉토리 생성

### ⏳ 진행 중
- **SAM ViT-H weights 다운로드** (2.4GB)
  - 예상 소요 시간: 3-10분 (인터넷 속도에 따라)
  - 파일: `models/sam_vit_h_4b8939.pth`

---

## 🎯 다운로드 완료 후 워크플로우

### Step 1: MedSAM 테스트 (5분)

```bash
# 기본 작동 확인
python src/medical_imaging/medsam_detector.py
```

**예상 결과:**
- GPU 인식 확인
- 샘플 슬라이스 segmentation
- 시각화 저장

---

### Step 2: Annotation Tool 실행 (30분-1시간)

```bash
streamlit run streamlit_apps/medsam_annotation_tool.py
```

**작업 flow:**
1. MedSAM 초기화
2. CT 슬라이스 선택
3. 종양 중심 클릭
4. 자동 segmentation 확인
5. 맞다/안맞다 표시
6. 5-10개 슬라이스 반복

**필요한 것:**
- 종양이 명확한 슬라이스 식별
- 각 슬라이스당 1-2번 클릭
- 결과 검증

---

### Step 3: Annotation 저장 및 검토

**Output:**
- `annotations/medsam_annotations.json`

**포함 내용:**
```json
{
  "20050.dcm": {
    "file": "20050.dcm",
    "point": [256, 300],
    "mask": [[0, 0, 1, ...], ...],
    "score": 0.95,
    "validated": true
  },
  ...
}
```

---

### Step 4: Fine-tuning (Optional, 1-2일)

**최소 데이터로 성능 향상**

```python
from medsam_finetuning import finetune

model = finetune(
    base_checkpoint='models/sam_vit_h_4b8939.pth',
    annotations='annotations/medsam_annotations.json',
    ct_data_dir='CTdata/CTdcm/',
    n_epochs=50,
    lr=1e-5
)
```

**예상 개선:**
- Dice: 0.75 → 0.85-0.90
- IoU: 0.65 → 0.75-0.85

---

### Step 5: Full Inference (30분)

```python
# 전체 426 슬라이스 자동 분할
python scripts/run_full_medsam_inference.py
```

**Output:**
- 각 슬라이스별 segmentation mask
- Bounding box
- 확률 점수
- 시각화 이미지

---

## 🔧 현재 진행 사항 확인

### 다운로드 확인
```bash
# PowerShell에서
ls models/
```

**완료 시:**
```
sam_vit_h_4b8939.pth (2.4GB)
```

### GPU 확인
```bash
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"
```

**예상 출력:**
```
CUDA available: True
GPU: NVIDIA GeForce RTX 5070
```

---

## 📊 예상 성능

### Zero-shot (annotation 없이)
- **Dice Score**: 0.70-0.80
- **IoU**: 0.60-0.70
- **장점**: 즉시 사용 가능
- **단점**: 일반적인 segmentation

### Few-shot (5-10 annotations)
- **Dice Score**: 0.80-0.88
- **IoU**: 0.70-0.80
- **장점**: 빠른 adaptation
- **단점**: 약간의 수동 작업

### Fine-tuned (20+ annotations)
- **Dice Score**: 0.85-0.92
- **IoU**: 0.75-0.85
- **장점**: 최고 성능
- **단점**: 더 많은 annotation 시간

---

## 💡 Tips

### Annotation 전략
1. **명확한 슬라이스부터**
   - 종양이 크고 명확한 슬라이스
   - 경계가 분명한 경우

2. **다양성 확보**
   - 다른 위치
   - 다른 크기
   - 다른 contrast

3. **Quality over Quantity**
   - 10개의 완벽한 annotation
   - 50개의 부정확한 것보다 나음

### 시간 절약
- MedSAM이 초안 자동 생성
- 의사는 확인만 (5-10초/slice)
- 전통적 수동 labeling 대비 80% 시간 절약

---

## 🚨 문제 해결

### "Model not found"
→ 다운로드 완료 대기

### "CUDA out of memory"
→ ViT-L 또는 ViT-B 사용

### "Segmentation not accurate"
→ 다른 포인트 시도
→ 5-10개 annotation 후 fine-tuning

### "Too slow"
→ GPU 사용 확인
→ Batch processing

---

## ⏭️ 다음 단계 요약

**즉시 (다운로드 완료 후):**
1. ✅ 테스트 스크립트 실행
2. ✅ GPU 작동 확인
3. ✅ 샘플 segmentation 검토

**오늘 중:**
4. Annotation tool로 5-10개 슬라이스 라벨링
5. 결과 검토 및 저장

**내일:**
6. Fine-tuning (optional)
7. 전체 426 슬라이스 inference
8. 결과 분석

**이번 주:**
9. 항외과 교수 검증
10. Ground truth 데이터셋 구축
11. 논문 작성 시작

---

**Estimated completion: 다운로드 후 1-2시간이면 첫 결과 확인 가능**
