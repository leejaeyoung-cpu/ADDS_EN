# Cellpose 현미경 분석 파이프라인

**HUVEC 세포 형태계측학 자동화 분석 — Cellpose cyto3 모델 기반**

---

## 개요

ADDS Cellpose 파이프라인은 HUVEC (Human Umbilical Vein Endothelial Cells) 및 암세포주 현미경 이미지를 자동 분석하여 세포 형태 지표, Ki-67 증식 지수, 종양 이질성 점수를 추출합니다.

이 데이터는 CT 방사선학 지표와 융합되어 **14차원 멀티모달 특징 벡터**를 완성합니다.

**분석 규모**: n = **43,190개** 세포 (HUVEC Serum 실험 전체)  
**사용 모델**: Cellpose **cyto3** (최신 세포질 분할 모델)

---

## 실험 설계

### HUVEC Serum 실험 (핵심 데이터셋)

| 실험군 | 조건 | 이미지 수 | 분석 세포 수 |
|--------|------|---------|----------|
| Control | 기본 배양액 | 20장 | 11,717 |
| Healthy Serum | 정상인 혈청 | 20장 | 6,538 |
| HGPS Serum | 조로증 환자 혈청 | 20장 | 13,676 |
| **HGPS + MT-Exo** | **조로증 + 미토콘드리아 엑소좀** | **20장** | **11,259** |

### 분석 지표 정의

| 지표 | 계산 방법 | 임상 의미 |
|------|---------|---------|
| **Cell Density** | 세포 수 / 이미지 면적 (mm²) | 성장 속도 |
| **Elongation Ratio** | 장축 / 단축 | 이동성/침윤성 |
| **Circularity** | 4π × 면적 / 둘레² | 세포 활성화 |
| **Clark-Evans Index** | 관찰 평균 거리 / 기대 평균 거리 | 공간 군집성 |
| **Ki-67 Index** | 증식 세포 비율 추정 | 증식 속도 |
| **Heterogeneity Score** | 면적 변동계수 (CV) | 종양 이질성 |

---

## 분석 파이프라인

```
원시 현미경 이미지 (Brightfield / Fluorescence)
         │
         ▼
    ┌──────────────────────────────────────┐
    │  Step 1: 이미지 전처리               │
    │  - CLAHE (대비 향상)                 │
    │  - Gaussian Denoising (σ=1.0)        │
    │  - 채널 표준화                       │
    └──────────────────────────────────────┘
         │
         ▼
    ┌──────────────────────────────────────┐
    │  Step 2: Cellpose cyto3 추론          │
    │  - diameter=None (자동 추정)          │
    │  - flow_threshold=0.4                │
    │  - cellprob_threshold=0.0            │
    │  - GPU 가속 (RTX 5070)               │
    └──────────────────────────────────────┘
         │
         ▼
    ┌──────────────────────────────────────┐
    │  Step 3: 마스크 후처리               │
    │  - 소형 세포 필터링 (< 50 px²)       │
    │  - 경계 세포 제거                    │
    │  - 세포 인스턴스 레이블링            │
    └──────────────────────────────────────┘
         │
         ▼
    ┌──────────────────────────────────────┐
    │  Step 4: 형태 특징 추출              │
    │  scikit-image regionprops            │
    │  - area, perimeter, eccentricity     │
    │  - major/minor_axis_length           │
    │  - solidity, extent                  │
    └──────────────────────────────────────┘
         │
         ▼
    ┌──────────────────────────────────────┐
    │  Step 5: 임상 지표 계산              │
    │  - Ki-67 Index (밀도 + 형태 기반)   │
    │  - Clark-Evans Index                 │
    │  - 이질성 점수 (CV)                  │
    │  - 약물 저항성 점수                  │
    └──────────────────────────────────────┘
         │
         ▼
    출력: 7D Cell Culture Feature Vector
    + 시각화 (마스크 오버레이, 히트맵)
```

---

## 핵심 결과

### HUVEC 형태계측 결과 (n = 43,190 cells)

| 조건 | 세포 수 | 장축비 | 세포면적 (px²) | 원형도 | Ki-67 추정 |
|------|---------|--------|--------------|--------|-----------|
| Control | 11,717 | 1.831 | 696 | 0.72 | 12.3% |
| Healthy Serum | 6,538 | 1.865 | 618 | 0.74 | 10.8% |
| HGPS Serum | 13,676 | 1.902 | 756 | 0.69 | 18.5% |
| **HGPS + MT-Exo** | **11,259** | **1.992** | **775** | **0.68** | **21.2%** |

**통계 유의성**: MT-Exo 처리군 장축비 증가 **p < 0.001** (One-way ANOVA + Tukey HSD)

### TNF-α 모델 결과 (HUVEC 시간 경과)

| 시점 | 평균 면적 | 원형도 | 건강 점수 | 해석 |
|------|---------|--------|---------|------|
| Control | 1,100 px² | 0.70 | 🟢 0.75 | 정지 상태 |
| **48hr** | **900 px²** | **0.40** | **🔴 0.35** | **혈관 누출/F-Actin 스트레스** |

---

## 코드 예시

### 기본 실행

```bash
# Step 1: 전처리
python analysis/huvec/01_preprocess.py \
    --input_dir /data/microscopy/raw/ \
    --output_dir /data/microscopy/preprocessed/

# Step 2: Cellpose 세분화
python analysis/huvec/02_cellpose_run.py \
    --input_dir /data/microscopy/preprocessed/ \
    --model cyto3 \
    --gpu True \
    --diameter 0  # 자동 추정

# Step 3: 특징 추출 및 통계
python analysis/huvec/07_ppt_figures.py
python analysis/huvec/08_ppt_infographic.py
```

### Python API 사용

```python
from src.pathology.cdss_cellpose_pipeline import CellposePipeline

pipeline = CellposePipeline(model_type="cyto3", use_gpu=True)

# 이미지 분석
result = pipeline.analyze(
    image_path="huvec_hgps_mt_exo_01.tif",
    condition="HGPS_MT-Exo"
)

print(f"검출 세포 수: {result['cell_count']}")
print(f"Ki-67 지수: {result['ki67_index']:.1f}%")
print(f"이질성 점수: {result['heterogeneity_score']:.3f}")

# 14D 특징 벡터 추출
feature_vector = result['cell_culture_features']
# → PK 모델 및 ADDS 추론에 전달
```

### 세션 상태 연동 (Streamlit)

```python
# Section 2에서 업로드된 이미지를 Section 3으로 전달
st.session_state['cell_images'] = uploaded_images  # 제로복사 전달

# Section 3에서 처리
if 'cell_images' in st.session_state:
    cell_result = pipeline.analyze_batch(
        st.session_state['cell_images']
    )
```

---

## 7차원 세포 배양 특징 벡터

```python
cell_features = {
    # 1. 세포 밀도 (cells/mm²)
    "cell_density": 247.3,
    
    # 2. 약물 저항성 점수 [0-1]
    # 높을수록 약물 저항성 높음
    "drug_resistance_score": 0.34,
    
    # 3. 증식 점수 [0-1] (Ki-67 기반)
    "proliferation_score": 0.21,
    
    # 4. 미세환경 복잡도 [0-1]
    # Clark-Evans + 면적 분산 복합
    "microenv_complexity": 0.58,
    
    # 5. 평균 원형도 [0-1]
    "mean_circularity": 0.68,
    
    # 6. Clark-Evans 군집 지수
    # <1: 군집, 1: 랜덤, >1: 균등 분포
    "clark_evans_index": 0.87,
    
    # 7. 예상 세포 생존율 [0-1]
    "estimated_viability": 0.76,
}
```

---

## 환경 설정

### GPU 설정 (권장)

```bash
# CUDA 지원 Cellpose
pip install cellpose[all]

# RTX 50-series (Blackwell)
pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu128
```

### CPU 폴백

```python
# GPU 없는 환경
pipeline = CellposePipeline(
    model_type="cyto3",
    use_gpu=False,  # CPU 모드
    batch_size=1    # 메모리 절약
)
```

---

## 트러블슈팅

| 증상 | 원인 | 해결 |
|------|------|------|
| 세포 과다/과소 검출 | `diameter` 파라미터 | 실제 세포 직경으로 설정 |
| GPU OOM | 이미지 너무 큼 | `tile=True` + 타일 크기 감소 |
| 저품질 마스크 | 이미지 전처리 부족 | CLAHE + 노이즈 제거 강화 |
| 느린 속도 | CPU 실행 | CUDA 환경 확인 |

---

*참조: [ARCHITECTURE.md](ARCHITECTURE.md) | [PHARMACOKINETICS.md](PHARMACOKINETICS.md)*
