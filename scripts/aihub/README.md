# AI-Hub 데이터 처리 스크립트

## 사용 방법

### 1. 데이터 분석
데이터를 다운로드한 후 먼저 구조를 분석합니다:

```bash
python scripts/aihub/data_loader.py data/aihub_colorectal/raw
```

**출력:**
- 데이터 유형 자동 감지 (CT/병리/내시경)
- 파일 형식 확인
- 크기 통계
- 샘플 로드
- `data/aihub_colorectal/metadata/analysis_report.json` 생성

### 2. ADDS 형식 변환
(분석 리포트 확인 후 작성 예정)

```bash
python scripts/aihub/convert_to_adds.py
```

### 3. 데이터 검증
(변환 후 작성 예정)

```bash
python scripts/aihub/verify_dataset.py
```

## 스크립트 목록

- `data_loader.py` - 데이터 구조 분석 및 샘플 로드
- `convert_to_adds.py` - ADDS 형식 변환 (작성 예정)
- `verify_dataset.py` - 데이터셋 검증 (작성 예정)
- `find_params.py` - 파라미터 최적화 (작성 예정)

## 다음 단계

1. 사용자가 데이터 다운로드
2. `data_loader.py` 실행하여 구조 파악
3. 분석 리포트 기반으로 나머지 스크립트 작성
