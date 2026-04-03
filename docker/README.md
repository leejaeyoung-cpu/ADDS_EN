# ADDS Docker Configuration

## 환경 변수 설정

```bash
# PostgreSQL
POSTGRES_PASSWORD=your_secure_password_here

# 데이터 루트
DATA_ROOT=F:/ADDS/patient_data

# GPU 설정 (NVIDIA Docker runtime 필요)
NVIDIA_VISIBLE_DEVICES=all
```

## 사용 방법

### 1. 개발 환경

기존 개발용 설정 사용:
```bash
docker-compose up -d
```

### 2. Production 환경 (Physics-based ADDS)

새로운 microservices 아키텍처:
```bash
docker-compose -f docker-compose.prod.yml up -d
```

### 3. 서비스별 실행

특정 서비스만 실행:
```bash
# 데이터베이스만
docker-compose -f docker-compose.prod.yml up -d postgres

# 데이터 수집 서비스
docker-compose -f docker-compose.prod.yml up -d data-ingestion

# CT 분석 서비스 (GPU 필요)
docker-compose -f docker-compose.prod.yml up -d ct-analysis
```

## 서비스 포트

| 서비스 | 포트 | 용도 |
|--------|------|------|
| PostgreSQL | 5432 | 데이터베이스 |
| API Gateway | 8000 | 메인 API |
| CT Analysis | 8001 | CT 분석 서비스 |
| Energy Modeling | 8002 | 에너지 계산 |
| AI Discovery | 8003 | AI 추론 |
| Redis | 6379 | 캐시 |

## 데이터베이스 초기화

```bash
# 스키마 자동 생성 (docker-entrypoint-initdb.d)
docker-compose -f docker-compose.prod.yml up -d postgres

# 수동 실행
docker exec -i adds_postgres psql -U adds_user -d adds_db < database/schema/database_schema.sql
```

## 로그 확인

```bash
# 전체 로그
docker-compose -f docker-compose.prod.yml logs -f

# 특정 서비스
docker-compose -f docker-compose.prod.yml logs -f ct-analysis
```

## GPU 확인

```bash
# NVIDIA Docker runtime 설치 확인
docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi

# CT 분석 컨테이너 내 GPU 확인
docker exec adds_ct_analysis nvidia-smi
```

## 다음 단계

1. Dockerfile 작성 필요
   - `docker/Dockerfile.data-ingestion`
   - `docker/Dockerfile.ct-analysis`
   - `docker/Dockerfile.energy-modeling`
   - `docker/Dockerfile.ai-discovery`
   - `docker/Dockerfile.api-gateway`

2. 환경별 설정
   - `.env.development`
   - `.env.production`

3. 볼륨 백업
   - `postgres_data` 자동 백업
   - `patient_data` 외부 스토리지 연결
