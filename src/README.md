# ADDS Source Modules

**핵심 소스 코드 디렉토리 — 26개 서브모듈**

---

## 모듈 구조

```
src/
├── adds/               ← ADDS 핵심 추론 엔진
├── ai/                 ← AI 모델 관리
├── ai_validation/      ← AI 결과 검증
├── analysis/           ← 분석 파이프라인
├── clinical/           ← 임상 데이터 관리
├── data/               ← 데이터 로더
├── evaluation/         ← 평가 메트릭
├── knowledge/          ← 문헌 지식 베이스 (311편)
├── knowledge_base/     ← 구조화된 약물·기전 DB
├── medical_ai/         ← 의료 AI 앙상블
├── medical_imaging/    ← CT 이미지 처리 핵심
│   ├── detection/          ← 종양 검출 (SimpleHUDetector)
│   ├── preprocessing/      ← DICOM 전처리
│   ├── radiomics/          ← PyRadiomics 특징 추출
│   └── segmentation/       ← 장기 분할 (nnU-Net)
├── ml/                 ← 머신러닝 모델
│   ├── fusion/             ← 14D 멀티모달 융합
│   └── survival/           ← PFS/OS 생존 분석
├── models/             ← 모델 가중치 관리
├── pathology/          ← Cellpose 현미경 분석
│   └── cdss_cellpose_pipeline.py
├── preprocessing/      ← 일반 전처리 유틸리티
├── pritamab_ml/        ← Pritamab 반응 예측
├── processing/         ← 데이터 처리 파이프라인
├── protein/            ← PrPc 단백질 분석
├── recommendation/     ← 약물 추천 엔진
├── reporting/          ← PDF 보고서 생성
├── security/           ← 데이터 보안
├── ui/                 ← Streamlit UI 컴포넌트
│   └── app.py              ← 메인 UI 진입점
├── utils/              ← 공통 유틸리티
├── visualization/      ← 데이터 시각화
└── xai/                ← 설명 가능 AI (XAI)
```

---

## 핵심 모듈 가이드

### `medical_imaging/detection/`

검증된 CT 종양 검출기:

```python
from src.medical_imaging.detection.simple_hu_detector import SimpleHUDetector

detector = SimpleHUDetector(
    hu_min=60,    # 동맥기 하한
    hu_max=120,   # 동맥기 상한
    min_size=50   # 최소 병변 크기 (mm³)
)
results = detector.detect(volume_array)
```

> ⚠️ `IntegratedCRCDetectionPipeline`에서 이 검출기를 직접 임포트합니다.  
> 로직 변경 시 반드시 양쪽 동기화 필요.

---

### `pathology/cdss_cellpose_pipeline.py`

Cellpose 기반 세포 분석:

```python
from src.pathology.cdss_cellpose_pipeline import CellposePipeline

pipeline = CellposePipeline(model_type="cyto3", use_gpu=True)
result = pipeline.analyze(image_path="huvec_001.tif")
# result["ki67_index"], result["heterogeneity_score"], ...
```

---

### `recommendation/`

약물 추천 엔진:

```python
from src.recommendation import DrugRecommender

recommender = DrugRecommender(knowledge_base_path="src/knowledge_base/")
recommendation = recommender.recommend(
    kras_status="G12D",
    msi_status="MSS",
    feature_vector_14d=features
)
```

---

### `reporting/`

PDF 보고서 생성:

```python
from src.reporting import ReportGenerator

gen = ReportGenerator()
gen.create_physician_report(patient, analysis_data, output_path="report_physician.pdf")
gen.create_patient_report(patient, analysis_data, output_path="report_patient.pdf")
```

---

### `xai/`

설명 가능 AI (의사 결정 근거 시각화):

```python
from src.xai import FeatureImportanceExplainer

explainer = FeatureImportanceExplainer()
importance = explainer.explain(model, feature_vector_14d)
explainer.plot_shap_values(importance, output_path="feature_importance.png")
```

---

## 의존성 규칙

| 모듈 | 허용 의존성 | 금지 의존성 |
|------|-----------|-----------|
| `medical_imaging/` | `preprocessing/`, `utils/` | `ui/`, `recommendation/` |
| `pathology/` | `utils/` | `medical_imaging/`, `recommendation/` |
| `recommendation/` | `knowledge/`, `knowledge_base/`, `ml/` | `ui/`, 직접 API |
| `ui/` | 모든 모듈 | 순환 임포트 금지 |

---

## 코딩 표준

### UTF-8 인코딩 (Windows 중요!)

```python
# 모든 파이썬 파일 최상단
# -*- coding: utf-8 -*-

# 또는 환경변수 설정
import os
os.environ["PYTHONIOENCODING"] = "utf-8"
```

### NumPy 직렬화 (SQLite 저장 시)

```python
from src.utils.serialization import convert_numpy_to_serializable

# SQLite에 저장 전 반드시 변환
data = convert_numpy_to_serializable(numpy_features)
db.save(json.dumps(data))
```

### 보호된 임포트 패턴

```python
try:
    import SimpleITK as sitk
    SITK_AVAILABLE = True
except ImportError:
    SITK_AVAILABLE = False
```

---

*참조: [docs/ARCHITECTURE.md](../docs/ARCHITECTURE.md)*
