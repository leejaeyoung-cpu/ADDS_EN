<div align="center">

<img src="https://img.shields.io/badge/ADDS-v3.5.0-blueviolet?style=for-the-badge&logo=python" alt="ADDS Version"/>

# ADDS — AI-Driven Drug Synergy & Diagnostic System

**정밀 종양학을 위한 멀티모달 AI 플랫폼**  
*Multimodal AI Platform for Precision Oncology*

[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x_GPU-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-Backend-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-UI-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Cellpose](https://img.shields.io/badge/Cellpose-cyto3-00C49F)](https://cellpose.readthedocs.io/)
[![nnU-Net](https://img.shields.io/badge/nnU--Net-v2-FF6B35)](https://github.com/MIC-DKFZ/nnUNet)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)
[![Institution](https://img.shields.io/badge/Institution-Inha_University_Hospital-003DA5)](https://www.inha.com/)

<br/>

> **ADDS**는 CT 방사선학, 세포 형태계측학, 약동학 모델링, 기계학습을 하나의 통합 플랫폼으로 융합하여  
> 대장암(CRC) 환자를 위한 개인화 항암 약물 칵테일을 추천하는 정밀 종양학 AI 시스템입니다.

</div>

---

## 📌 목차 / Table of Contents

- [시스템 개요](#-시스템-개요--system-overview)
- [전체 아키텍처](#-전체-아키텍처--architecture)
- [핵심 모듈](#-핵심-모듈--core-modules)
  - [CT 분석 파이프라인](#1-ct-분석-파이프라인)
  - [Cellpose 현미경 분석](#2-cellpose-현미경-분석)
  - [KRAS-PrPc 약물 시너지](#3-kras-prpc-약물-시너지)
  - [약동학 (PK/PD) 모델링](#4-약동학-pkpd-모델링)
  - [임상 의사결정 지원](#5-임상-의사결정-지원-cds)
  - [환자 관리 시스템](#6-통합-환자-관리-시스템)
- [성능 지표](#-성능-지표--performance-metrics)
- [14차원 특징 벡터](#-14차원-멀티모달-특징-벡터)
- [설치 및 실행](#-설치-및-실행--installation)
- [API 참조](#-api-참조--api-reference)
- [데이터 구조](#-데이터-구조--data-structure)
- [연구 배경](#-연구-배경--research-background)
- [인용](#-인용--citation)

---

## 🔬 시스템 개요 / System Overview

ADDS (AI-Driven Drug Synergy) 는 인하대학교병원과의 공동 연구를 통해 개발된 **정밀 종양학 AI 생태계**입니다.

### 핵심 혁신 포인트

| 혁신 | 설명 |
|------|------|
| **멀티모달 데이터 융합** | CT 방사선학 + 세포 병리학 + 임상 메타데이터를 단일 14차원 특징 벡터로 통합 |
| **이중 추론 엔진** | ADDS 경로 기반 엔진 + OpenAI GPT-4 동시 실행 및 교차 검증 |
| **RAG 기반 근거 생성** | 의사 소견서를 1순위 프롬프트로 활용하는 검색 증강 생성(RAG) 시스템 |
| **PrPc 바이오마커 발견** | TCGA 데이터(n=2,285)에서 KRAS-RPSA 시그널로솜 기반 신규 바이오마커 발견 |
| **실시간 임상 적용** | 15.67초 내 엔드-투-엔드 분석 완료 (530×751×750 볼륨 기준) |

---

## 🏗️ 전체 아키텍처 / Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                    ADDS Precision Oncology Platform v3.5             │
│                      Inha University Hospital                        │
└─────────────────────────────────────────────────────────────────────┘
                                    │
          ┌─────────────────────────┼─────────────────────────┐
          ▼                         ▼                         ▼
  ┌───────────────┐       ┌─────────────────┐       ┌────────────────┐
  │  Streamlit UI │       │  FastAPI Backend │       │  Data Layer    │
  │  (Port 8505)  │◄─────►│  (Port 8000)    │◄─────►│  SQLite / NFS  │
  │               │       │                 │       │                │
  │ • 환자 관리   │       │ /api/v1/        │       │ patients.db    │
  │ • AI 분석     │       │  ├─ patients    │       │ ct_data/       │
  │ • 약물 추천   │       │  ├─ ct          │       │ microscopy/    │
  │ • 보고서 생성 │       │  ├─ cellpose    │       │ literature/    │
  └───────────────┘       │  ├─ pharmacoki  │       └────────────────┘
                          │  ├─ adds        │
                          │  └─ openai      │
                          └─────────────────┘
                                    │
         ┌──────────────────────────┼──────────────────────────┐
         ▼                          ▼                          ▼
┌────────────────┐        ┌─────────────────┐        ┌────────────────┐
│  CT Pipeline   │        │ Cellpose Pipeline│        │  Drug Synergy  │
│  (6 Stages)    │        │                 │        │  Engine        │
│                │        │ cyto3 Model     │        │                │
│ S1: DICOM→NIfTI│        │ → Segmentation  │        │ KRAS-PrPc      │
│ S2: Organ Seg  │        │ → Ki-67 Index   │        │ Signalosome    │
│ S3: Tumor Det  │        │ → Morphology    │        │                │
│ S4: Radiomics  │        │ → Heterogeneity │        │ Pritamab       │
│ S5: Staging    │        │                 │        │ Prediction     │
│ S6: ADDS Integ │        │ n=43,190 cells  │        │                │
│                │        │ analyzed        │        │ PK/PD Modeling │
│ Acc: 98.65%    │        │                 │        │                │
└────────────────┘        └─────────────────┘        └────────────────┘
         │                          │                          │
         └──────────────────────────┼──────────────────────────┘
                                    ▼
                    ┌───────────────────────────────┐
                    │    14D Multimodal Feature      │
                    │    Vector Fusion               │
                    │                                │
                    │  CT Radiomics (7D):            │
                    │  Sphericity, Entropy,          │
                    │  Contrast, Size, Circularity,  │
                    │  Mean HU, Confidence           │
                    │                                │
                    │  Cell Culture (7D):            │
                    │  Density, Drug Resistance,     │
                    │  Proliferation, Complexity,    │
                    │  Circularity, Clark-Evans,     │
                    │  Viability                     │
                    └───────────────────────────────┘
                                    │
                    ┌───────────────┴───────────────┐
                    ▼                               ▼
         ┌─────────────────┐             ┌──────────────────┐
         │  ADDS Engine    │             │  OpenAI Engine   │
         │  (Pathway-Based)│             │  (GPT-4 Medical) │
         │                 │             │                  │
         │ KRAS/RAF/MEK/   │             │ Clinical Summary │
         │ ERK Signaling   │◄── Cross ──►│ Treatment Plan   │
         │ Synergy Scoring │  Validate   │ MDT Consensus    │
         └─────────────────┘             └──────────────────┘
                    │                               │
                    └───────────────┬───────────────┘
                                    ▼
                    ┌───────────────────────────────┐
                    │   Final Drug Cocktail          │
                    │   Recommendation               │
                    │                                │
                    │  FOLFOX + Bevacizumab          │
                    │  + PK-Optimized Dosing         │
                    │  + Outcome Simulation          │
                    │   (ORR / PFS / OS)             │
                    └───────────────────────────────┘
```

---

## ⚙️ 핵심 모듈 / Core Modules

### 1. CT 분석 파이프라인

**6단계 3D CT 종양 검출 및 방사선학 분석 파이프라인**

```
Stage 1: 3D Volume Reconstruction
    DICOM Series → 1mm³ Isotropic NIfTI Volume
    (SimpleITK, scipy 기반 리샘플링)

Stage 2: Anatomical Organ Segmentation
    nnU-Net v2 → Colon / Liver / Lymph Node Parsing

Stage 3: Tumor Detection  ← VerifiedCTDetector (98.65% Accuracy)
    HU Thresholding: 60–120 HU (Arterial Phase)
    2D Slice-by-Slice Morphological Filtering
    Min Size: 30 px (noise), 50 mm³ (clinical threshold)

Stage 4: Radiomics Extraction
    PyRadiomics → 100+ Phenotypic Features
    (Sphericity, Entropy, GLCM Contrast, Surface Area...)

Stage 5: Biomarker Prediction
    Malignancy Score / TNM Staging / MSI / KRAS Status

Stage 6: ADDS Integration
    Radiomics → PK Sensitivity Model → Drug Recommendation
```

**주요 성능 지표 (인하대학교병원 코호트)**

| 지표 | 값 |
|------|-----|
| 검출 정확도 | **98.65%** (74개 슬라이스 중 73개) |
| 처리 시간 | **15.67초** (530×751×750 볼륨) |
| 처리량 | **33.8 슬라이스/초** |
| HU 탐지 범위 | 60–120 HU (동맥기) |
| 최소 병변 크기 | 50 mm³ |

관련 스크립트:
```bash
python ct_pipeline_v4.py                    # CT 파이프라인 메인
python detect_tumors_inha_corrected.py      # 검증된 검출기 (98.65%)
python ct_crc_detection_pipeline.py         # CRC 특화 파이프라인
python batch_tumor_detection_dcm.py         # 배치 처리
```

---

### 2. Cellpose 현미경 분석

**HUVEC 세포 형태계측학 자동화 분석 (Cellpose cyto3 모델 기반)**

```
Raw Microscopy Image
       │
       ▼
CLAHE + Denoising (Preprocessing)
       │
       ▼
Cellpose cyto3 Segmentation
       │
       ├─→ Cell Count & Density
       ├─→ Elongation Ratio (장축/단축)
       ├─→ Circularity Score
       ├─→ Clark-Evans Index (군집 분포)
       ├─→ Ki-67 Proliferation Index Estimation
       └─→ Tumor Heterogeneity Score
```

**분석 결과 (HUVEC Serum 실험, n = 43,190 cells)**

| 조건 | 세포 수 | 장축비 | 세포면적 | 해석 |
|------|---------|--------|---------|------|
| Control | 11,717 | 1.831 | 696 px² | 정지 상태 |
| Healthy Serum | 6,538 | 1.865 | 618 px² | 정상 활성화 |
| HGPS Serum | 13,676 | 1.902 | 756 px² | 병리적 활성화 |
| **HGPS + MT-Exo** | **11,259** | **1.992** | **775 px²** | **최대 내피 활성화** |

> MT-Exo 처리군에서 세포 장축비 유의미한 증가 (p < 0.001) — 내피세포 이동 능력 증강 시사

관련 스크립트:
```bash
python analysis/huvec/01_preprocess.py     # 이미지 전처리
python analysis/huvec/02_cellpose_run.py   # Cellpose 세분화
python analysis/huvec/07_ppt_figures.py    # 논문용 Figure 생성
python verify_cellpose_pipeline.py          # 파이프라인 검증
```

---

### 3. KRAS-PrPc 약물 시너지

**기전 기반 약물 시너지 예측 엔진**

#### PrPc 조직-혈청 패러독스 해결

| 측정 | CRC 조직 | 혈청 | 기전 |
|------|---------|------|------|
| PRNP mRNA | ↓ 낮음 | — | 종양 억제 |
| PrPc 단백질 | — | ↑↑ 높음 | **ADAM10/17 쉐딩** |

> ADAM10/17 효소가 세포막 GPI-앵커 PrPc를 절단 → 혈류로 방출  
> TCGA 실데이터 검증: n = 2,285 (BRCA, STAD, COAD, PAAD, READ)

#### KRAS-RPSA 시그널로솜 경로

```
KRAS Mutation (G12D/G12V)
       │
       ▼
RAF → MEK → ERK Activation
       │
       ├─→ PrPc-RPSA Complex Formation
       │         │
       │         └─→ Laminin Binding (세포 침윤 촉진)
       │
       └─→ Downstream Survival Pathways
                 │
                 ├─→ mTOR Axis
                 ├─→ PI3K/AKT
                 └─→ WNT/β-catenin
```

#### 약물 지식 베이스

| 지표 | 값 |
|------|-----|
| 총 논문 수 | 311편 (Nature/Cell/Science 등 Tier-1) |
| 데이터 샘플 | 2,348 임상 샘플 |
| 등록 약물 | 113종 |
| 작용 기전 | 90개 |
| 바이오마커 | 69개 |
| 시너지 조합 | 59개 |

---

### 4. 약동학 (PK/PD) 모델링

**환자 맞춤형 항암제 용량 최적화 1-구획 모델**

$$C_{max} = \frac{D}{V_d} \cdot e^{-k_e \cdot t}$$

| 파라미터 | 공식 | 단위 |
|---------|------|------|
| **청소율 (Cl)** | $120.0 \times \max(0.7, 1.0 - \frac{V_{tumor}}{500})$ | mL/min |
| **분포용적 (Vd)** | $45.0 + (V_{tumor} \times 0.5)$ | L |
| **반감기 (t½)** | $0.693 \times \frac{V_d}{Cl \times 0.06}$ | hours |
| **최적 용량 (D)** | $200.0 \times (1.0 + \frac{Ki67}{200})$ | mg/m² |

**안전 제약 조건:**
- 투여 간격: 6h – 24h (하드 클램프)
- 최대 반응률: 95% (임상 현실성 유지)
- 신장/간 기능 대리 지표: `cl_factor` (종양 부담 기반)

---

### 5. 임상 의사결정 지원 (CDS)

**이중 추론 엔진 기반 교차 검증 시스템**

```
┌─────────────────────────────────────────────────────────┐
│            6-Step Dynamic Inference Pipeline             │
└─────────────────────────────────────────────────────────┘

Step 0: RAG Analysis
    의사 소견서 → 의미론적 임상 컨텍스트 추출
    (증상, 병력, 환자 선호도)

Step 1: CT Analysis (Live API)
    DICOM 업로드 → /api/v1/ct/analyze
    결과: 방사선학 JSON + 시각화 이미지 스트림

Step 2: Cell Analysis (조건부)
    Cellpose 세분화 → Ki-67 정량화
    (현미경 이미지 없으면 건너뜀)

Step 3: Pharmacokinetics
    CT + Cellpose 결과 → PK 최적화 파라미터

Step 4: ADDS Inference
    경로 기반 기전 추천
    (RAG 컨텍스트 + 멀티모달 데이터)

Step 5: OpenAI Inference
    GPT-4 임상 통합 (의사 소견서 1순위 프롬프트)

Step 6: Cross-Validation
    소견서 ↔ CT 결과 ↔ 병리 결과 자동 일치성 검증
```

**최종 추천 생성:**
- 🎯 항암제 칵테일 (예: FOLFOX + Bevacizumab)
- 💊 최적화된 투여량 및 경로
- 📊 예후 시뮬레이션 (ORR / PFS / OS)
- 📄 이중 보고서 (의료진 기술 보고서 + 환자 가이드)

---

### 6. 통합 환자 관리 시스템

**엔터프라이즈급 임상 데이터 관리 (IPMS)**

```python
# 환자 ID 형식
Patient ID: P-2026-001

# 핵심 임상 메타데이터
{
  "tnm_stage": "T4N0M0",
  "msi_status": "MSS",
  "kras_mutation": "G12D",
  "ecog_score": 1,
  "ki67_index": 45.2,
  "tumor_location": "Sigmoid Colon"
}
```

| 기능 | 설명 |
|------|------|
| **환자 CRUD** | P-YYYY-NNN 형식 영구 레코드 |
| **종단 추적** | 치료 경과에 따른 데이터 이력 관리 |
| **멀티모달 업로드** | CT DICOM + 현미경 이미지 + 소견서 통합 |
| **실시간 진행** | 분석 단계별 실시간 상태 추적 |
| **PDF 보고서** | 자동 생성 (의료진용 / 환자용) |

---

## 📊 성능 지표 / Performance Metrics

### CT 분석 성능
```
┌─────────────────────────────────────────────────────┐
│  CT Detection Performance (Inha University Hospital) │
│  ─────────────────────────────────────────────────── │
│  Accuracy:      ████████████████████ 98.65%         │
│  Speed:         15.67s / patient (E2E)               │
│  Throughput:    33.8 slices/sec                      │
│  Volume Size:   530 × 751 × 750 voxels               │
│  HU Range:      60 – 120 HU (arterial phase)         │
│  Min Lesion:    50 mm³                               │
└─────────────────────────────────────────────────────┘
```

### 시스템 벤치마크
| 구성 | 처리 시간 |
|------|---------|
| CT E2E 분석 (표준) | ~45.2초 |
| CT E2E 분석 (최적화) | **15.67초** |
| Cellpose (GPU, 1장) | ~3.2초 |
| 약물 추천 생성 | ~2.1초 |
| 전체 파이프라인 | **< 90초** |

### 연구 데이터 규모

| 데이터 유형 | 규모 |
|------------|------|
| HUVEC 분석 세포 수 | **43,190개** |
| TCGA PrPc 실제 샘플 | **2,285개** |
| 논문 지식 베이스 | **311편** |
| 이하 CT 코호트 볼륨 | 530×751×750 |
| 임상 샘플 (전체) | **2,348개** |

---

## 🧬 14차원 멀티모달 특징 벡터

```python
feature_vector = {
    # CT Radiomics (7D) — 거시적 영상 특징
    "sphericity":          float,  # 종양 구형도
    "energy":              float,  # GLCM 텍스처 에너지
    "contrast":            float,  # 영상 대비도
    "tumor_size_mm2":      float,  # 종양 크기 (mm²)
    "circularity":         float,  # 원형도
    "mean_hu":             float,  # 평균 하운스필드 단위
    "detection_confidence":float,  # 검출 신뢰도

    # Cell Culture (7D) — 미시적 세포 특징
    "cell_density":        float,  # 세포 밀도 (cells/mm²)
    "drug_resistance":     float,  # 약물 저항 점수
    "proliferation_score": float,  # Ki-67 기반 증식 지수
    "microenv_complexity": float,  # 미세환경 복잡도
    "mean_circularity":    float,  # 평균 세포 원형도
    "clark_evans_index":   float,  # 공간적 군집 지수
    "estimated_viability": float,  # 예상 세포 생존율
}
```

---

## 🚀 설치 및 실행 / Installation

### 시스템 요구사항

| 항목 | 최소 | 권장 |
|------|------|------|
| Python | 3.11 | 3.11+ |
| GPU | CUDA 11.x | CUDA 12.8 (RTX 50-series) |
| RAM | 16 GB | 32 GB |
| VRAM | 8 GB | 16 GB |
| 저장공간 | 50 GB | 200 GB |

### 빠른 설치

```bash
# 1. 레포지토리 클론
git clone https://github.com/leejaeyoung-cpu/ADDS.git
cd ADDS

# 2. 가상환경 생성
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 3. 의존성 설치
pip install -r requirements.txt

# 4. 환경 변수 설정
cp .env.example .env
# .env 파일에서 OPENAI_API_KEY, DB_PATH 등 설정

# 5. 데이터베이스 초기화
cd backend
python -c "from database_init import init_database; init_database()"
cd ..
```

### 시스템 실행

```bash
# ✅ 방법 1: 통합 실행 (권장)
START_ALL.bat           # 백엔드(8000) + Streamlit UI(8505) 동시 실행

# ✅ 방법 2: 수동 실행
# 터미널 1 — 백엔드
cd backend
uvicorn main:app --host 0.0.0.0 --port 8000 --reload

# 터미널 2 — Streamlit UI
python -m streamlit run src/ui/app.py --server.port 8505
```

> **접근 주소:**
> - 🖥️ 임상 UI: `http://localhost:8505`
> - 📡 API 서버: `http://localhost:8000`
> - 📚 API 문서: `http://localhost:8000/docs`

### GPU 설정 (RTX 50-series / Blackwell)

```bash
# PyTorch Nightly (cu128 지원)
pip install --pre torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/nightly/cu128

# GPU 상태 확인
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, Device: {torch.cuda.get_device_name(0)}')"
```

---

## 📁 데이터 구조 / Data Structure

```
ADDS/
├── 📂 src/                         ← 핵심 소스 모듈
│   ├── adds/                       ← ADDS 추론 엔진
│   ├── medical_imaging/            ← CT 파이프라인
│   │   ├── detection/              ← 종양 검출 (SimpleHUDetector)
│   │   ├── preprocessing/          ← DICOM 전처리
│   │   ├── radiomics/              ← 방사선학 특징 추출
│   │   └── segmentation/           ← 장기 분할
│   ├── pathology/                  ← Cellpose 현미경 분석
│   ├── clinical/                   ← 임상 데이터 관리
│   ├── ml/                         ← 머신러닝 모델
│   │   ├── fusion/                 ← 멀티모달 융합
│   │   └── survival/               ← PFS/OS 예측
│   ├── protein/                    ← PrPc 단백질 분석
│   ├── recommendation/             ← 약물 추천 엔진
│   ├── knowledge/                  ← 지식 베이스 (311편 논문)
│   ├── knowledge_base/             ← 구조화된 약물 DB
│   ├── reporting/                  ← PDF 보고서 생성
│   ├── visualization/              ← 데이터 시각화
│   ├── xai/                        ← 설명 가능 AI (XAI)
│   └── ui/                         ← Streamlit UI 컴포넌트
│
├── 📂 backend/                     ← FastAPI 백엔드
│   ├── main.py                     ← 앱 진입점
│   ├── api/                        ← REST API 라우터
│   │   ├── ct_analysis.py          ←  /api/v1/ct
│   │   ├── patients.py             ←  /api/v1/patients
│   │   ├── pharmacokinetics.py     ←  /api/v1/pharmacokinetics
│   │   ├── adds_inference.py       ←  /api/v1/adds
│   │   └── openai_inference.py     ←  /api/v1/openai
│   ├── services/                   ← 비즈니스 로직 서비스
│   │   ├── ct_pipeline_service.py
│   │   ├── cell_culture_service.py
│   │   ├── adds_service.py
│   │   └── openai_service.py
│   ├── models/                     ← SQLAlchemy ORM 모델
│   └── schemas/                    ← Pydantic 스키마
│
├── 📂 analysis/                    ← 연구 분석 스크립트
│   ├── huvec/                      ← HUVEC 세포 분석
│   ├── ct/                         ← CT 분석 파이프라인
│   └── pritamab/                   ← Pritamab 약물 시너지
│
├── 📂 figures/                     ← 논문용 Figure (300 DPI)
├── 📂 docs/                        ← 시스템 문서
├── 📂 configs/                     ← 설정 파일
├── 📂 tests/                       ← 유닛 테스트
├── 📂 notebooks/                   ← Jupyter 분석 노트북
├── 📂 data/samples/                ← 익명화된 샘플 데이터
│
├── 🐳 Dockerfile                   ← 컨테이너 이미지
├── 🐳 docker-compose.yml           ← 서비스 오케스트레이션
├── 📋 requirements.txt             ← Python 의존성
├── 📋 pyproject.toml               ← 프로젝트 설정
└── 🔑 .env.example                 ← 환경변수 템플릿
```

---

## 📡 API 참조 / API Reference

### Base URL

```
http://localhost:8000/api/v1
```

### 핵심 엔드포인트

| Method | Endpoint | 설명 |
|--------|---------|------|
| `GET` | `/health` | 시스템 상태 확인 |
| `GET` | `/patients` | 환자 목록 조회 |
| `POST` | `/patients` | 신규 환자 등록 |
| `GET` | `/patients/{id}` | 환자 상세 조회 |
| `POST` | `/ct/analyze` | CT DICOM 분석 실행 |
| `GET` | `/ct/health` | CT 파이프라인 상태 |
| `GET` | `/ct/models/status` | nnU-Net 모델 상태 |
| `POST` | `/pharmacokinetics/analyze` | PK 파라미터 계산 |
| `POST` | `/adds/infer` | ADDS 경로 기반 추론 |
| `POST` | `/openai/infer` | GPT-4 임상 추론 |

### CT 분석 요청 예시

```python
import requests

# DICOM 파일 업로드 및 분석
with open("tumor_series.dcm", "rb") as f:
    response = requests.post(
        "http://localhost:8000/api/v1/ct/analyze",
        files={"dicom_file": f},
        data={"patient_id": "P-2026-001"}
    )

result = response.json()
print(f"종양 검출: {result['tumors_detected']}개")
print(f"신뢰도: {result['confidence']:.2%}")
print(f"TNM 추정: {result['tnm_stage']}")
```

### PK 최적화 요청 예시

```python
pk_response = requests.post(
    "http://localhost:8000/api/v1/pharmacokinetics/analyze",
    json={
        "patient_id": "P-2026-001",
        "tumor_volume_mm3": 2450.5,
        "ki67_index": 45.2,
        "body_surface_area": 1.73
    }
)

pk = pk_response.json()
print(f"최적 용량: {pk['optimal_dose_mg_m2']} mg/m²")
print(f"반감기: {pk['half_life_hours']:.1f}시간")
print(f"투여 간격: {pk['dosing_interval_hours']}시간")
```

---

## 🧪 연구 배경 / Research Background

### PrPc 바이오마커 발견 여정

| 버전 | 전략 | 코호트 | 목표 | 결과 |
|------|------|--------|------|------|
| v1.0 | 단일 마커 (혈청) | n=63 | Stage III CRC | ❌ 갭 발견 |
| v2.0 | 멀티마커 패널 | 20–30개 | 일반 GI 암 | 🔄 전략 전환 |
| **v3.0** | **AI-First / 국가 바이오데이터** | **n=300–800** | **조기 검출** | ✅ **진행 중** |

### 지식 베이스 구성 (2026년 2월 기준)

```
문헌 지식 베이스 v2.0
├── Tier 1 (100편): Nature / Cell / Science / Nature Medicine
├── Tier 2 (100편): JCO / Cancer Research
└── Tier 3: The Biology of Cancer (Weinberg)

통계:
• 311편 논문 (초록 기반 GPT-4 추출)
• 2,285 실제 TCGA 샘플 (BRCA, STAD, COAD, PAAD, READ)
• 113종 약물 / 90개 기전 / 69개 바이오마커
• 59개 시너지 조합 검증
```

### 임상 파일럿 프로토콜

```
파일럿 연구 설계 (v1.0)
• 디자인: 전향적 파일럿, N=100 (증례 50, 대조 50)
• 목표: Stage I 30% + Stage II 30% (조기 검출)
• Go/No-Go 기준: AUC ≥ 0.75

3개월 로드맵:
• Month 1: IRB 제출 + 계정 설정
• Month 2: 승인 확보 + 사이트 활성화
• Month 3: 등록 + Go/No-Go 결정
```

---

## ⚠️ 데이터 가용성 / Data Availability

환자 CT 데이터 및 원시 현미경 이미지는 이 레포지토리에 **포함되지 않습니다:**

- 🔒 **PHI 규정** (Protected Health Information): 개인건강정보 보호법
- 📏 **파일 크기 제한**: GitHub 100MB 제한 (CT 볼륨은 수 GB)
- 🏥 **기관 승인 필요**: 인하대학교병원 IRB 승인 데이터

재현을 위한 데이터 접근은 저자에게 문의하세요.  
`data/samples/` 디렉토리에는 익명화된 소규모 샘플만 포함됩니다.

---

## 📄 인용 / Citation

이 코드를 연구에 사용하신다면 다음을 인용해 주세요:

```bibtex
@misc{adds2026,
  title     = {ADDS: AI-Driven Drug Synergy and Diagnostic System — 
               A Multimodal Precision Oncology Platform},
  author    = {Lee, Jaeyoung and others},
  year      = {2026},
  url       = {https://github.com/leejaeyoung-cpu/ADDS},
  note      = {Inha University Hospital, Incheon, Korea}
}
```

---

## 🤝 기여 / Contributing

기여를 환영합니다! 세부 가이드라인은 [CONTRIBUTING.md](.github/CONTRIBUTING.md)를 참조하세요.

**빠른 기여 가이드:**
1. `Fork` → `Feature Branch` 생성 (`feat/my-feature`)
2. 변경사항 작성 + 테스트 추가
3. `Pull Request` 생성 (PR 템플릿 작성 필수)

---

## 🔐 보안 / Security

보안 취약점 발견 시 공개 이슈를 생성하지 말고, [SECURITY.md](.github/SECURITY.md)의 가이드라인에 따라 비공개 보고해 주세요.

---

## ⚠️ Methodological Notes / 방법론 주석

> **Transparency Statement**: All performance metrics are reported with their methodological context and limitations. This section is intended to support scientific reproducibility and honest evaluation.

### CT Tumor Detection (98.65% Accuracy)

| Item | Detail |
|------|--------|
| **Dataset** | Inha University Hospital CRC cohort |
| **Sample size** | N = 74 CT slices (single patient, arterial phase) |
| **Method** | HU-threshold (60–120 HU) + morphological filtering + connected-component analysis |
| **Ground truth** | Manual annotation by clinical radiologist |
| **Metric** | Slice-level detection accuracy (correct slices / total slices) |
| **95% CI** | [0.949, 1.000] (Wilson score interval) |
| **⚠️ Limitation** | Single-patient pilot study. Multi-center validation with N≥200 patients is ongoing. This metric does NOT represent patient-level diagnostic accuracy. |

### Cell Morphometry (N = 43,190 cells)

| Item | Detail |
|------|--------|
| **Instrument** | Brightfield microscopy |
| **Cell lines** | HUVEC (Human Umbilical Vein Endothelial Cells) |
| **Conditions** | 4 groups: Control · Healthy Serum · HGPS Serum · HGPS + MT-Exosome |
| **Images analyzed** | 80 brightfield images |
| **Segmentation** | Cellpose v3 (cyto3 model), GPU-accelerated |
| **⚠️ Limitation** | In vitro model only. Clinical relevance requires PDO (Patient-Derived Organoid) validation. |

### Drug Synergy Models (TCGA N = 2,285)

| Item | Detail |
|------|--------|
| **Training data** | TCGA-COAD + DrugComb + OncoKB |
| **Synergy metrics** | Bliss Independence, Loewe Additivity, HSA, ZIP |
| **Model architecture** | DeepSynergy v2 (DNN) + XGBoost ensemble |
| **Validation** | 5-fold cross-validation on held-out TCGA subset |
| **⚠️ Limitation** | Synergy predictions are based on genomic/transcriptomic features. Prospective clinical validation has not been conducted. Not for clinical use without regulatory approval. |

### Reproducibility

```bash
# Verify core scientific logic (no GPU required)
pip install -r requirements-ci.txt
python -m pytest tests/test_science_core.py -v
# Expected: 18 passed
```

All statistical tests, synergy formulas, and data integrity checks in `tests/test_science_core.py` pass with zero external dependencies.

---

## 📬 연락처 / Contact

| 항목 | 내용 |
|------|------|
| **레포지토리** | [github.com/leejaeyoung-cpu/ADDS](https://github.com/leejaeyoung-cpu/ADDS) |
| **기관** | 인하대학교병원, 인천광역시, 대한민국 |
| **연구 분야** | 정밀 종양학 / AI 의료기기 (SaMD) |
| **목표 저널** | Nature Communications |

---

<div align="center">

**ADDS v3.5.0** — Built with ❤️ for Precision Oncology  
Inha University Hospital × AI Research Team | 2026

</div>
