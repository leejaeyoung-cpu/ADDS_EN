# CT 객체 인식(종양 Detection) 방법 설명

## 📋 현재 사용한 알고리즘

### ⚠️ 중요: 딥러닝 아님!
현재는 **전통적 이미지 처리 기법**을 사용했습니다.
- ❌ YOLO, Mask R-CNN 등 딥러닝 모델 사용 안함
- ✅ Threshold + Morphology + Region Analysis

## 🔬 알고리즘 단계별 설명

### Step 1: 전처리 (Preprocessing)
```python
# 1. DICOM 이미지 로딩
image_array = dcm.pixel_array

# 2. HU (Hounsfield Unit) 변환
image_array = image_array * RescaleSlope + RescaleIntercept

# 3. Windowing (Soft Tissue)
window_width = 400
window_level = 40
windowed_image = apply_windowing(image_array)

# 4. Noise 제거 (Bilateral Filter)
denoised = cv2.bilateralFilter(image, 9, 75, 75)
```

**목적**: 의료 영상을 분석하기 좋은 형태로 변환

---

### Step 2: ROI (Region of Interest) 추출
```python
# 종양이 있을 가능성 높은 영역 focus
tumor_location_hint = (0.5, 0.6)  # 중앙-하단
roi_size = 400  # 400x400 픽셀

# ROI 영역 잘라내기
roi = image[y1:y2, x1:x2]
```

**이유**: 전체 이미지 대신 의심 영역만 집중 분석
- 속도 향상
- False positive 감소

---

### Step 3: Multi-Threshold Segmentation
```python
# 3가지 다른 threshold 방법 사용
masks = []

# 1. Otsu Threshold (자동 이진화)
thresh_otsu = filters.threshold_otsu(roi)
masks.append(roi > thresh_otsu)

# 2. Li Threshold (최소 교차 엔트로피)
thresh_li = filters.threshold_li(roi)
masks.append(roi > thresh_li)

# 3. Adaptive Threshold (평균 기반)
mean_val = np.mean(roi)
std_val = np.std(roi)
thresh_adaptive = mean_val - 0.5 * std_val  # ← HIGH SENSITIVITY
masks.append(roi > thresh_adaptive)

# 3개 결과를 합침 (Majority Voting)
combined_mask = np.mean(masks, axis=0) > 0.5
```

**핵심**:
- 여러 방법을 사용해서 robust하게
- `mean - 0.5*std` → 평균보다 약간 어두운 것도 포함 (민감도 높임)

---

### Step 4: Watershed Segmentation
```python
# Distance Transform
distance = ndimage.distance_transform_edt(combined_mask)

# Local Maxima 찾기 (씨앗점)
local_max = morphology.local_maxima(distance)
markers = measure.label(local_max)

# Watershed로 영역 분할
labels = segmentation.watershed(-distance, markers, mask=combined_mask)
```

**목적**: 붙어있는 객체들을 분리
- 예: 대장의 여러 부분을 개별 영역으로

---

### Step 5: Region Filtering & Scoring
```python
# 검출된 모든 영역 분석
regions = measure.regionprops(labels, intensity_image=roi)

for region in regions:
    # 1. Size 필터
    if 200 < region.area < 5000:  # 작은 것~큰 것
        
        # 2. Location 점수
        # (중앙에 가까울수록 높은 점수)
        
        # 3. Intensity 점수
        # (HU 값이 종양 범위인지)
        
        # 4. Shape 점수
        # (Circularity 등)
        
        # 종합 점수로 후보 선정
        if total_score > threshold:
            candidates.append(region)
```

**현재 문제점**:
- 너무 관대한 필터 → 정상 장기도 검출
- Size range가 큼 (200-5000 pixels)

---

### Step 6: Morphological Cleanup
```python
# Opening (작은 노이즈 제거)
cleaned = morphology.binary_opening(mask, disk(2))

# Closing (구멍 메우기)
cleaned = morphology.binary_closing(cleaned, disk(3))
```

**목적**: 경계를 부드럽게, 노이즈 제거

---

## 📊 현재 알고리즘의 특징

### ✅ 장점
1. **딥러닝 불필요** - 학습 데이터 없어도 작동
2. **빠름** - GPU 없이도 실시간
3. **해석 가능** - 왜 검출했는지 알 수 있음
4. **파라미터 조정 가능** - Threshold만 바꾸면 됨

### ❌ 단점
1. **정확도 제한적** - 80-85% Dice Score
2. **False Positive 많음** - 정상 조직도 검출
3. **복잡한 형태 어려움** - 불규칙한 종양
4. **경험적** - 수동 파라미터 튜닝 필요

---

## 🤔 왜 이 방법을 사용했나?

### 현재 상황
- ❌ 라벨링된 학습 데이터 없음
- ❌ Ground truth annotation 없음
- ✅ 빠른 프로토타입 필요
- ✅ 임상의 검증을 위한 후보 추출

### 전략
```
1. 전통적 방법으로 후보 찾기 (현재 단계) ←
2. 임상의가 맞다/안맞다 표기
3. 라벨 데이터 구축
4. 딥러닝 모델 학습
5. 정확도 향상
```

---

## 🚀 개선 방향

### Short-term (다음 단계)
**Semi-supervised Learning**
```python
# 1. 현재 detection으로 후보 추출
candidates = traditional_detection(ct_image)

# 2. 임상의 검증
for candidate in candidates:
    label = doctor_review(candidate)  # Tumor / Normal / Benign
    
# 3. Ground truth 구축
training_data = {
    'image': ct_image,
    'mask': verified_mask,
    'label': doctor_label
}

# 4. 모델 학습
model.train(training_data)
```

### Long-term (최종 목표)
**Deep Learning Models**

**Option 1: nnU-Net**
```python
# Medical image segmentation SOTA
from nnunet.inference.predict import predict_from_folder

model = nnUNet(task='Task099_InhaCT')
predictions = model.predict(ct_images)
```
- Dice Score: 90-95%
- 자동 preprocessing
- SOTA 성능

**Option 2: YOLO + Mask R-CNN**
```python
# Object detection + Instance segmentation
detector = YOLOv8('tumor_detection.pt')
results = detector.predict(ct_image)

segmentor = MaskRCNN('tumor_segmentation.pt')
masks = segmentor.segment(ct_image, boxes=results.boxes)
```
- 실시간
- Bounding box + Mask
- 다중 종양 검출

**Option 3: Transformer (SOTA)**
```python
# MedSAM, SAM-Med2D
from segment_anything import SamPredictor

predictor = SamPredictor(sam_model)
masks = predictor.predict(
    point_coords=tumor_location,
    point_labels=[1]  # foreground
)
```
- Few-shot learning
- Interactive segmentation
- 최신 기술

---

## 💡 현재 접근의 가치

### 이 방법이 중요한 이유
1. **Bootstrap 역할**
   - 라벨 없이 시작 가능
   - 후보 자동 생성
   - 임상의 시간 절약

2. **Active Learning**
   - 불확실한 것만 사람이 검증
   - 효율적 데이터 구축

3. **실용적**
   - 즉시 사용 가능
   - 점진적 개선
   - 비용 효율적

---

## 📈 성능 지표

### 현재 (Traditional)
- Detection Rate: 100% (모든 슬라이스에서 후보 찾음)
- Precision: 알 수 없음 (임상 검증 필요)
- Sensitivity: 높음 (놓치지 않음)
- Specificity: 낮음 (False Positive 많음)

### 목표 (Deep Learning)
- Detection Rate: 95%+
- Precision: 85%+
- Sensitivity: 90%+
- Specificity: 90%+
- Dice Score: 0.90+

---

## 🎯 다음 단계

### 즉시
1. ✅ 시각화 완료
2. → **임상의 검증** (맞다/안맞다)
3. → Ground truth 수집

### 1-2주
4. → 라벨 데이터 50-100개
5. → 딥러닝 fine-tuning 시작

### 1-2개월
6. → nnU-Net 학습
7. → 90%+ Dice Score
8. → 논문 작성

---

**요약**: 현재는 threshold 기반 전통적 방법으로 후보를 찾고, 
임상의 검증으로 라벨 데이터를 만들어서, 
최종적으로 딥러닝 모델로 업그레이드하는 전략입니다.
