# Contributing to ADDS

ADDS 프로젝트에 기여해 주셔서 감사합니다! 이 가이드는 기여 프로세스를 안내합니다.

---

## 🧭 기여 전 읽어야 할 것들

- [프로젝트 README](../README.md) — 시스템 개요
- [아키텍처 문서](../docs/ARCHITECTURE.md) — 전체 시스템 설계
- [API 참조](../docs/API_REFERENCE.md) — 백엔드 API 명세

---

## 🚀 빠른 시작

### 1. Fork & Clone

```bash
git clone https://github.com/YOUR_USERNAME/ADDS.git
cd ADDS
git remote add upstream https://github.com/leejaeyoung-cpu/ADDS.git
```

### 2. 개발 환경 설정

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install -r requirements_dev.txt
```

### 3. 브랜치 생성

브랜치 이름 규칙:

| 유형 | 형식 | 예시 |
|------|------|------|
| 새 기능 | `feat/기능-이름` | `feat/pk-model-v2` |
| 버그 수정 | `fix/버그-설명` | `fix/ct-hu-normalization` |
| 문서 업데이트 | `docs/내용` | `docs/api-reference` |
| 리팩토링 | `refactor/범위` | `refactor/cellpose-pipeline` |
| 테스트 추가 | `test/대상` | `test/tumor-detection` |

```bash
git checkout -b feat/my-new-feature
```

---

## 📋 기여 영역별 가이드

### CT 파이프라인 (`src/medical_imaging/`)
- HU 값은 절대 normalize하지 마세요 (Stage 1에서 `normalize=False` 유지)
- 새 검출기 추가 시 `detect_tumors_inha_corrected.py` 기준(98.65%)으로 정확도 비교 필수
- 테스트: `python test_ct_pipeline_simple.py`

### Cellpose 분석 (`src/pathology/`)
- 새 모델 추가 시 `cyto3` 기준 성능과 비교
- n=43,190 HUVEC 데이터셋으로 검증 필수

### 백엔드 API (`backend/`)
- 새 라우터는 `backend/api/` 디렉토리에 추가
- Pydantic 스키마는 `backend/schemas/`에 별도 정의
- FastAPI 라우터 등록: `backend/main.py`에 `include_router()` 추가
- SQLAlchemy 모델 변경 시 마이그레이션 스크립트 필수

### 약동학 모델 (`backend/api/pharmacokinetics.py`)
- 투여 간격 범위: **6h – 24h** (하드 클램프 유지)
- 최대 반응률: **95%** 초과 불가
- 새 파라미터 추가 시 임상 근거 논문 인용 필수

---

## ✅ PR 제출 체크리스트

PR을 열기 전에 아래를 확인하세요:

- [ ] 변경사항이 관련 모듈의 단위 테스트를 통과함
- [ ] 새 기능에 대한 테스트를 추가함
- [ ] 의학 데이터 (PHI)가 코드에 하드코딩되지 않음
- [ ] API 변경 시 `docs/API_REFERENCE.md` 업데이트
- [ ] CUDA 의존성 변경 시 `requirements.txt` 업데이트
- [ ] `CHANGELOG.md`에 변경 내역 추가

---

## 🧪 테스트 실행

```bash
# 전체 테스트
python -m pytest tests/ -v

# 모듈별 테스트
python -m pytest tests/test_ct_pipeline.py -v
python -m pytest tests/test_cellpose.py -v
python -m pytest tests/test_api.py -v

# 빠른 스모크 테스트
python test_ct_pipeline_simple.py
python verify_cellpose_pipeline.py
```

---

## 🏥 의료 데이터 취급 원칙

> ⚠️ **중요**: ADDS는 의료 데이터를 다루는 연구 플랫폼입니다.

1. **절대 금지**: 실제 환자 CT 데이터, 병리 이미지, 환자 정보를 GitHub에 업로드
2. **데이터 경로**: 항상 `.env`의 환경변수로 관리 (`DATA_PATH`, `CT_DATA_DIR`)
3. **샘플 데이터만**: `data/samples/`에는 완전 익명화된 합성 또는 공개 데이터만 허용
4. **로그 확인**: PR 전 `git diff`로 민감 정보 누출 여부 검토

---

## 📩 질문 및 토론

- **버그 리포트**: [이슈 템플릿](.github/ISSUE_TEMPLATE/bug_report.md) 사용
- **기능 제안**: [피처 리퀘스트 템플릿](.github/ISSUE_TEMPLATE/feature_request.md) 사용
- **임상 데이터 관련**: 저자에게 직접 연락

---

감사합니다! 🙏  
*Inha University Hospital AI Research Team*
