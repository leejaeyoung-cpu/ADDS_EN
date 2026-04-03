# ADDS System Architecture

## 전체 시스템 아키텍처

ADDS는 **정밀 종양학을 위한 멀티모달 AI 플랫폼**으로, 세 가지 핵심 계층으로 구성됩니다:

1. **프레젠테이션 계층** — Streamlit 기반 임상 UI
2. **애플리케이션 계층** — FastAPI 기반 REST API 백엔드
3. **데이터/AI 계층** — 멀티모달 AI 엔진 및 데이터 스토어

---

## 계층별 아키텍처

### 1. 프레젠테이션 계층 (Streamlit UI)

```
src/ui/app.py
│
├── 탭 1: 환자 관리 (show_patient_management)
│   ├── Section 1: 환자 검색 & 선택
│   ├── Section 2: 병리 정보 & 멀티모달 업로드
│   ├── Section 3: AI 분석 (6단계 파이프라인)
│   ├── Section 4: 항암제 칵테일 추천
│   └── Section 5: 교차 검증 & 보고서
│
├── 탭 2: CT 분석 대시보드
├── 탭 3: Cellpose 분석
└── 탭 4: 시스템 상태
```

### 2. 애플리케이션 계층 (FastAPI Backend)

```
backend/main.py (FastAPI App)
│
├── /api/v1/health           — 시스템 상태
├── /api/v1/patients         — 환자 CRUD
├── /api/v1/ct/              — CT 분석
│   ├── /analyze             — CT DICOM 분석 실행
│   ├── /health              — CT 파이프라인 상태
│   └── /models/status       — nnU-Net 모델 상태
├── /api/v1/pharmacokinetics — PK 파라미터 계산
├── /api/v1/adds             — ADDS 경로 기반 추론
└── /api/v1/openai           — GPT-4 임상 추론
```

### 3. AI/데이터 계층

```
AI Engines:
├── VerifiedCTDetector (98.65% accuracy)
│   src/medical_imaging/detection/simple_hu_detector.py
├── Cellpose cyto3 Model
│   src/pathology/cdss_cellpose_pipeline.py
├── nnU-Net v2 (organ segmentation)
│   src/medical_imaging/segmentation/
├── ADDS Knowledge Engine
│   src/knowledge_base/
├── PyRadiomics (100+ features)
│   src/medical_imaging/radiomics/
└── OpenAI GPT-4 API

Data Stores:
├── SQLite (patient records)
│   backend/patients.db
├── NIfTI files (CT volumes)
│   data/outputs/{run_id}/
├── Literature Database
│   literature_database/ (311 papers)
└── Model Checkpoints
    checkpoints/ / segvol_ckpt/
```

---

## 데이터 흐름

```
의사 입력 (소견서 + 파일 업로드)
           │
           ▼
    Streamlit UI
    (st.session_state)
           │
           │ HTTP Multipart Upload
           ▼
    FastAPI Backend
    /api/v1/ct/analyze
           │
    ┌──────┴──────────────────────────┐
    │                                 │
    ▼                                 ▼
Stage 1-3                       Patient DB
(DICOM → NIfTI                  (SQLAlchemy)
 → Tumor Detection)
    │
    ▼
Stage 4-5
(Radiomics + Staging)
    │
    ▼
Stage 6 (ADDS Integration)
    │
    ├── 14D Feature Vector 생성
    │
    ├── ADDS Engine ──→ Pathway 기반 추천
    │
    └── OpenAI Engine ──→ GPT-4 임상 합성
           │
           ▼
    Cross-Validation + Consensus
           │
           ▼
    Drug Cocktail + PDF Report
```

---

## 기술 스택

| 계층 | 기술 | 버전 |
|------|------|------|
| **언어** | Python | 3.11+ |
| **백엔드** | FastAPI | 0.100+ |
| **UI** | Streamlit | 1.30+ |
| **ORM** | SQLAlchemy | 2.x |
| **DB** | SQLite | 3.x |
| **DL 프레임워크** | PyTorch | 2.x |
| **CT 분할** | nnU-Net v2 | latest |
| **세포 분석** | Cellpose | cyto3 |
| **방사선학** | PyRadiomics | 3.x |
| **의료영상** | SimpleITK | 2.x |
| **LLM** | OpenAI GPT-4 | API |
| **컨테이너** | Docker | 24+ |

---

## 배포 아키텍처

### 개발 환경 (Native)

```
Windows 11 + CUDA 12.8
├── Backend: uvicorn (Port 8000)
├── UI: streamlit (Port 8505)
└── GPU: RTX 5070 (Blackwell)
```

### 프로덕션 환경 (Docker)

```
docker-compose.clinical.yml
├── adds-backend (Port 8000)
│   Image: adds:latest
│   GPU: NVIDIA runtime
├── adds-ui (Port 8505)
│   Image: adds-ui:latest
└── adds-db (Volume mount)
    /data/clinical/
```

### 향후 병원 배포 (Hospital Appliance)

```
On-Premise Server
├── Isolated Network (PHI 보안)
├── NVIDIA A100 GPU
├── PACS Integration (DICOM 직접 연동)
└── EMR/HIS 연동 (HL7 FHIR)
```

---

## 성능 병목 포인트

| 단계 | 현재 처리 시간 | 병목 원인 | 해결 방향 |
|------|--------------|---------|---------|
| DICOM→NIfTI | ~2s | IO bound | 병렬 슬라이스 로딩 |
| 장기 분할 (nnU-Net) | ~10s | GPU inference | TensorRT 최적화 |
| 종양 검출 (CCA) | ~3s (2D) / ~15min (3D CCA) | CPU bound | CC3D C++ binding |
| Radiomics | ~5s | CPU bound | 병렬 처리 |
| ADDS 추론 | ~2s | Network (GPT-4) | 캐싱 전략 |
| **전체 E2E** | **~15.67s** | — | — |

> ⚠️ **알려진 병목**: 3D CCA (Connected Component Analysis)는 26K+ 컴포넌트 처리 시 15분+ 소요.  
> 현재 2D 슬라이스별 처리로 우회하여 **<3분**으로 단축.

---

## 모듈 의존성 그래프

```
src/ui/app.py
    ├── src/clinical/ (Patient Management)
    ├── backend/api/ct_analysis.py
    │       └── src/medical_imaging/detection/simple_hu_detector.py
    │       └── src/medical_imaging/radiomics/
    ├── backend/api/pharmacokinetics.py
    │       └── (CT Stage output + Cellpose output)
    ├── backend/api/adds_inference.py
    │       └── src/knowledge_base/
    │       └── src/recommendation/
    └── backend/api/openai_inference.py
            └── OpenAI API (외부)
```

---

*참조: [CT_PIPELINE.md](CT_PIPELINE.md) | [CELLPOSE_ANALYSIS.md](CELLPOSE_ANALYSIS.md) | [API_REFERENCE.md](API_REFERENCE.md)*
