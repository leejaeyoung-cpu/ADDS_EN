# ADDS Backend

**FastAPI 기반 REST API 백엔드 — ADDS v3.0.0**

---

## 아키텍처

```
backend/
├── main.py                 ← FastAPI 앱 진입점
├── config.py               ← 환경 설정
├── database_init.py        ← DB 초기화 유틸리티
├── gunicorn_config.py      ← 프로덕션 서버 설정
│
├── api/                    ← REST API 라우터
│   ├── ct_analysis.py      ← /api/v1/ct/*
│   ├── patients.py         ← /api/v1/patients/*
│   ├── pharmacokinetics.py ← /api/v1/pharmacokinetics/*
│   ├── adds_inference.py   ← /api/v1/adds/*
│   ├── openai_inference.py ← /api/v1/openai/*
│   ├── segmentation.py     ← /api/v1/segmentation/*
│   ├── features.py         ← /api/v1/features/*
│   ├── statistics.py       ← /api/v1/statistics/*
│   └── synergy.py          ← /api/v1/synergy/*
│
├── services/               ← 비즈니스 로직
│   ├── ct_pipeline_service.py
│   ├── cell_culture_service.py
│   ├── adds_service.py
│   └── openai_service.py
│
├── models/                 ← SQLAlchemy ORM 모델
│   ├── __init__.py         ← Base, Patient, CTAnalysis 노출
│   └── patient.py          ← 데이터 모델 정의
│
├── schemas/                ← Pydantic 검증 스키마
│   └── patient.py
│
├── pipeline/               ← CT 파이프라인 서비스
└── outputs/                ← 분석 결과 저장 경로
```

---

## 실행

### 개발 모드

```bash
cd backend
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

### 프로덕션 모드 (Gunicorn)

```bash
cd backend
gunicorn main:app -c gunicorn_config.py
```

---

## 데이터베이스

### 초기화

```bash
cd backend
python -c "from database_init import init_database; init_database()"
```

이 명령은 `backend/patients.db` (SQLite)를 생성하고 다음 테이블을 초기화합니다:
- `patients` — 환자 기본 정보
- `ct_analyses` — CT 분석 결과 이력
- `cell_analyses` — 세포 분석 결과

### 스키마 드리프트 방지 (중요!)

새 데이터 서비스 추가 시 반드시 독립적인 `init_database.py`를 제공하세요:

```python
# ⚠️ 필수: 모든 새 라우터에 DB 초기화 스크립트 포함
python migrate_metadata_tables.py  # 기존 테이블에 컬럼 추가 시
```

---

## 라우터 등록

`main.py`의 라우터 등록 패턴:

```python
from api import segmentation, features, statistics, synergy
from api import ct_analysis, patients, adds_inference, openai_inference, pharmacokinetics

app.include_router(patients.router,         prefix="/api/v1/patients",         tags=["Patients"])
app.include_router(ct_analysis.router,      prefix="/api/v1/ct",               tags=["CT Analysis"])
app.include_router(pharmacokinetics.router, prefix="/api/v1/pharmacokinetics", tags=["PK Model"])
app.include_router(adds_inference.router,   prefix="/api/v1/adds",             tags=["ADDS Inference"])
app.include_router(openai_inference.router, prefix="/api/v1/openai",           tags=["OpenAI"])
```

---

## 알려진 제약사항

| 제약 | 상세 | 해결 방법 |
|------|------|---------|
| CT 처리 시간 | 고해상도 볼륨: 120초+ | 클라이언트 타임아웃 설정 |
| SQLite 동시성 | 멀티 워커 충돌 | 프로덕션 배포 시 PostgreSQL 전환 |
| numpy 버전 | PyRadiomics 요구: 1.26.4 | requirements.txt 고정 |

---

*API 상세: [docs/API_REFERENCE.md](../docs/API_REFERENCE.md)*
