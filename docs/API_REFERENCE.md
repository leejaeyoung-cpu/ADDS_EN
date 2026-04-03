# ADDS API Reference

**REST API 완전 명세 — ADDS Backend v3.0.0**

---

## Base URL

```
http://localhost:8000/api/v1
```

**API 문서 (Swagger UI)**: `http://localhost:8000/docs`  
**ReDoc**: `http://localhost:8000/redoc`

---

## 인증

현재 버전은 로컬 연구 환경에서 인증 없이 사용합니다.  
병원 배포 시 Bearer Token 인증이 추가됩니다.

```http
Authorization: Bearer <token>  # 향후 병원 배포 시
```

---

## 시스템 상태

### `GET /health`

시스템 전체 상태 확인.

**응답 예시**:
```json
{
  "status": "healthy",
  "version": "3.0.0",
  "timestamp": "2026-02-06T14:23:11Z",
  "components": {
    "database": "connected",
    "gpu": "available",
    "ct_pipeline": "ready",
    "cellpose": "ready",
    "openai": "connected"
  },
  "gpu_info": {
    "name": "NVIDIA GeForce RTX 5070",
    "memory_total_gb": 12.0,
    "memory_used_gb": 2.3,
    "cuda_version": "12.8"
  }
}
```

---

## 환자 관리 API

### `GET /patients`

환자 목록 조회.

**Query Parameters**:
| 파라미터 | 타입 | 설명 |
|---------|------|------|
| `search` | string | 환자 ID 또는 이름 검색 |
| `page` | int | 페이지 번호 (기본: 1) |
| `limit` | int | 페이지 당 항목 수 (기본: 20, 최대: 100) |

**응답 예시**:
```json
{
  "patients": [
    {
      "id": "P-2026-001",
      "name": "김OO",
      "age": 58,
      "diagnosis": "CRC Stage III",
      "registration_date": "2026-01-15",
      "last_analysis": "2026-02-06T14:23:11Z",
      "kras_status": "G12D"
    }
  ],
  "total": 45,
  "page": 1,
  "limit": 20
}
```

---

### `POST /patients`

신규 환자 등록.

**요청 바디**:
```json
{
  "name": "김OO",
  "birth_date": "1968-03-15",
  "sex": "M",
  "diagnosis": "Colorectal Cancer",
  "tnm_stage": "T3N1M0",
  "msi_status": "MSS",
  "kras_mutation": "G12D",
  "ecog_score": 1,
  "institution": "Inha University Hospital"
}
```

**응답**:
```json
{
  "patient_id": "P-2026-046",
  "created_at": "2026-02-06T14:23:11Z",
  "status": "registered"
}
```

---

### `GET /patients/{patient_id}`

특정 환자 상세 조회.

```http
GET /api/v1/patients/P-2026-001
```

**응답**:
```json
{
  "id": "P-2026-001",
  "demographics": {...},
  "clinical_metadata": {
    "tnm_stage": "T3N1M0",
    "msi_status": "MSS",
    "kras_mutation": "G12D",
    "ecog_score": 1,
    "ki67_index": 45.2
  },
  "analysis_history": [
    {
      "analysis_id": "ANA-001",
      "date": "2026-02-06",
      "tumors_detected": 2,
      "recommendation": "FOLFOX + Bevacizumab",
      "status": "completed"
    }
  ]
}
```

---

## CT 분석 API

### `POST /ct/analyze`

CT DICOM 분석 실행 — 6단계 파이프라인.

**요청 (Multipart Form)**:
```http
POST /api/v1/ct/analyze
Content-Type: multipart/form-data

dicom_files: <binary DICOM files>
patient_id: P-2026-001
use_verified_detector: true
```

**Python 예시**:
```python
import requests

files = [
    ("dicom_files", ("scan_001.dcm", open("scan_001.dcm", "rb"), "application/dicom")),
    ("dicom_files", ("scan_002.dcm", open("scan_002.dcm", "rb"), "application/dicom")),
]

response = requests.post(
    "http://localhost:8000/api/v1/ct/analyze",
    files=files,
    data={"patient_id": "P-2026-001"},
    timeout=120  # 큰 볼륨 대비 120초 타임아웃
)
```

**응답**:
```json
{
  "patient_id": "P-2026-001",
  "analysis_id": "ANA-001",
  "processing_time_seconds": 15.67,
  "tumors_detected": 2,
  "staging": {
    "t_stage": "T3",
    "n_stage": "N1",
    "m_stage": "M0",
    "tnm": "T3N1M0"
  },
  "radiomics": {
    "sphericity": 0.743,
    "energy": 8923.4,
    "contrast": 0.234,
    "tumor_size_mm2": 847.3,
    "circularity": 0.621,
    "mean_hu": 89.3,
    "detection_confidence": 0.94
  },
  "tumor_candidates": [
    {
      "id": 1,
      "location": "Sigmoid Colon",
      "z_slice": 370,
      "centroid_mm": [245.2, 312.7, 370.0],
      "volume_mm3": 2450.5,
      "mean_hu": 89.3,
      "confidence": 0.94
    }
  ],
  "analyzed_images": [
    {
      "slice_z": 370,
      "original": "data:image/png;base64,...",
      "heatmap": "data:image/png;base64,...",
      "overlay": "data:image/png;base64,..."
    }
  ],
  "biomarker_predictions": {
    "kras_mutation_predicted": true,
    "msi_status_predicted": "MSS",
    "malignancy_score": 0.87
  }
}
```

---

### `GET /ct/health`

CT 파이프라인 상태 확인.

```json
{
  "pipeline_status": "ready",
  "detector": "VerifiedCTDetector v1.0 (98.65% accuracy)",
  "normalization_mode": "disabled",
  "hu_range": [60, 120],
  "nnunet_available": false,
  "fallback_mode": "SimpleHUDetector"
}
```

---

### `GET /ct/models/status`

AI 모델 로드 상태 확인.

```json
{
  "nnunet": {
    "status": "not_loaded",
    "reason": "Training cohort < 20 cases",
    "fallback": "SimpleHUDetector"
  },
  "segvol": {
    "status": "available",
    "checkpoint": "segvol_ckpt/model.pt"
  },
  "sam": {
    "status": "available",
    "checkpoint": "sam_vit_b_01ec64.pth"
  }
}
```

---

## 약동학 API

### `POST /pharmacokinetics/analyze`

환자 맞춤형 PK 파라미터 계산.

**요청**:
```json
{
  "patient_id": "P-2026-001",
  "tumor_volume_mm3": 2450.5,
  "ki67_index": 45.2,
  "body_surface_area": 1.73,
  "serum_creatinine": 0.9,
  "hepatic_function": "normal",
  "target_drug": "FOLFOX"
}
```

**응답**:
```json
{
  "pk_parameters": {
    "clearance_ml_min": 84.0,
    "volume_distribution_L": 1270.0,
    "half_life_hours": 10.5,
    "elimination_rate_constant": 0.066
  },
  "dosing_recommendation": {
    "optimal_dose_mg_m2": 245.2,
    "total_dose_mg": 424.2,
    "dosing_interval_hours": 24,
    "cycle_length_days": 14,
    "route": "IV infusion"
  },
  "predicted_outcomes": {
    "response_rate": 0.73,
    "pfs_months": 8.2,
    "grade3_toxicity_risk": 0.18
  },
  "safety_flags": []
}
```

---

## ADDS 추론 API

### `POST /adds/infer`

경로 기반 기전 약물 추천.

**요청**:
```json
{
  "patient_id": "P-2026-001",
  "clinical_context": "의사 소견 RAG 분석 결과...",
  "feature_vector_14d": {
    "sphericity": 0.743,
    "energy": 8923.4,
    "contrast": 0.234,
    "tumor_size_mm2": 847.3,
    "circularity": 0.621,
    "mean_hu": 89.3,
    "detection_confidence": 0.94,
    "cell_density": 247.3,
    "drug_resistance_score": 0.34,
    "proliferation_score": 0.21,
    "microenv_complexity": 0.58,
    "mean_circularity": 0.68,
    "clark_evans_index": 0.87,
    "estimated_viability": 0.76
  },
  "kras_status": "G12D",
  "msi_status": "MSS"
}
```

**응답**:
```json
{
  "primary_regimen": "FOLFOX + Bevacizumab",
  "secondary_regimen": "FOLFIRI + Bevacizumab",
  "pathway_rationale": "KRAS G12D → RAF/MEK/ERK 상시 활성 → 옥살리플라틴 DNA 교차결합 표적 + VEGF 혈관신생 억제",
  "synergy_score": 8.7,
  "contraindications": ["Cetuximab (KRAS MUT)", "Panitumumab (KRAS MUT)"],
  "biomarker_match": {
    "KRAS G12D": "FOLFOX 적응 ✅",
    "MSS": "면역관문억제제 제한 ⚠️",
    "Ki-67 >40%": "집중 요법 권장 ✅"
  }
}
```

---

## OpenAI 추론 API

### `POST /openai/infer`

GPT-4 기반 임상 통합 추론.

**요청**:
```json
{
  "patient_id": "P-2026-001",
  "doctors_notes": "58세 남성, S자 결장 부위 통증...",
  "ct_findings_summary": "직경 67mm 종양, HU 89",
  "adds_recommendation": "FOLFOX + Bevacizumab",
  "pk_recommendation": "245 mg/m² q24h"
}
```

---

## 에러 코드

| HTTP 코드 | 에러 코드 | 설명 |
|---------|---------|------|
| 400 | `INVALID_DICOM` | DICOM 파일 형식 오류 |
| 400 | `INVALID_HU_RANGE` | HU 범위 이상 (정규화 오류 의심) |
| 404 | `PATIENT_NOT_FOUND` | 환자 ID 없음 |
| 413 | `FILE_TOO_LARGE` | 업로드 파일 너무 큼 (>2GB) |
| 422 | `VALIDATION_ERROR` | 요청 바디 검증 실패 |
| 500 | `CT_PIPELINE_ERROR` | CT 파이프라인 내부 오류 |
| 503 | `GPU_UNAVAILABLE` | GPU 메모리 부족 |
| 504 | `INFERENCE_TIMEOUT` | 추론 타임아웃 (>120초) |

---

## 타임아웃 설정

| 엔드포인트 | 권장 타임아웃 |
|---------|------------|
| `/ct/analyze` (표준) | 60초 |
| `/ct/analyze` (고해상도 >500슬라이스) | 120초 |
| `/pharmacokinetics/analyze` | 30초 |
| `/adds/infer` | 30초 |
| `/openai/infer` | 60초 |

---

*참조: [ARCHITECTURE.md](ARCHITECTURE.md) | [CLINICAL_CDS.md](CLINICAL_CDS.md)*
