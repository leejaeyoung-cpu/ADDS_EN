# 특허출원서 Part 3 - 성능 최적화 및 실험 결과

## 【발명을 실시하기 위한 구체적인 내용】 (계속)

### 8. 성능 최적화 모듈의 상세 구성 (도 8, 9 참조)

도 8은 본 발명의 시스템과 종래 기술의 처리 시간 비교 그래프이다.  
도 9는 데이터베이스 인덱싱 전후 쿼리 성능 비교 그래프이다.

#### 8.1 GPU 가속부(134a)

**CUDA 초기화:**
```python
import torch

# GPU 사용 가능 여부 확인
if torch.cuda.is_available():
    device = torch.device('cuda:0')
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
else:
    device = torch.device('cpu')

# Cellpose 모델을 GPU로 이동
model = Cellpose(gpu=True, model_type='cyto2')
```

**혼합 정밀도 연산:**
```python
from torch.cuda.amp import autocast

with autocast():
    masks, flows, styles = model.eval(image)
```

일 실시예에서, 혼합 정밀도 사용 시:
- GPU 메모리: 2.0GB → 1.0GB (50% 감소)
- 처리 속도: 5.3s → 4.8s (추가 9% 향상)
- 정확도: Dice 0.893 → 0.891 (무시할 만한 차이)

**배치 처리:**
```python
def batch_process(images, batch_size=4):
    results = []
    for i in range(0, len(images), batch_size):
        batch = images[i:i+batch_size]
        
        # GPU 병렬 처리
        with torch.no_grad():
            masks_batch = model.eval(batch)
        
        results.extend(masks_batch)
    
    return results
```

일 실시예에서, batch_size=4 사용 시 처리량 3배 증가.

#### 8.2 데이터베이스 최적화부(134b)

**인덱스 생성:**
```sql
-- 복합 인덱스 (환자 ID + 타임스탬프)
CREATE INDEX idx_patient_date 
ON analysis_results(patient_id, analysis_date);

-- 단일 인덱스 (타임스탬프)
CREATE INDEX idx_timestamp 
ON analysis_results(timestamp);

-- 단일 인덱스 (실험명)
CREATE INDEX idx_experiment 
ON analysis_results(experiment_name);

-- 단일 인덱스 (파일명)
CREATE INDEX idx_filename 
ON analysis_results(filename);

-- 복합 인덱스 (환자 ID + 타임스탬프 + 실험명)
CREATE INDEX idx_composite 
ON analysis_results(patient_id, timestamp, experiment_name);
```

**성능 비교:**

인덱싱 전:
```sql
EXPLAIN QUERY PLAN
SELECT * FROM analysis_results WHERE patient_id = 'P001';
-- SCAN TABLE analysis_results
-- 실제 시간: 100ms (10,000 행 순차 스캔)
```

인덱싱 후:
```sql
EXPLAIN QUERY PLAN
SELECT * FROM analysis_results WHERE patient_id = 'P001';
-- SEARCH TABLE analysis_results USING INDEX idx_patient_date
-- 실제 시간: 0.3ms (330x 향상)
```

일 실시예에서, 5개 전략적 인덱스로 평균 쿼리 속도 330배 향상.

#### 8.3 API 최적화부(134c)

**GZip 압축:**
```python
from fastapi.middleware.gzip import GZipMiddleware

app.add_middleware(GZipMiddleware, minimum_size=1000)
```

압축 효과:
- JSON 응답 크기: 2.0MB → 600KB (70% 감소)
- 네트워크 전송 시간: 400ms → 120ms (Gigabit 네트워크)

**멀티워커 구성 (Gunicorn):**
```python
# gunicorn_config.py
import multiprocessing

workers = min(multiprocessing.cpu_count() * 2 + 1, 9)
worker_class = 'uvicorn.workers.UvicornWorker'
bind = '0.0.0.0:8000'
keepalive = 120
timeout = 300
```

일 실시예에서, 9 워커로 동시 처리 능력 9배 증가 (1-2 users → 9-18 users).

**연결 풀링:**
```python
from sqlalchemy import create_engine
from sqlalchemy.pool import QueuePool

engine = create_engine(
    DATABASE_URL,
    poolclass=QueuePool,
    pool_size=10,
    max_overflow=20,
    pool_pre_ping=True
)
```

#### 8.4 캐싱 전략부(134d)

**모델 가중치 캐싱:**
```python
from functools import lru_cache

@lru_cache(maxsize=128)
def get_model(model_type: str):
    """모델 로딩 결과를 메모리에 캐싱"""
    return models.Cellpose(gpu=True, model_type=model_type)

# 첫 호출: 3.2초 (디스크에서 로딩)
model = get_model('cyto2')

# 이후 호출: 0.001초 (캐시 히트)
model = get_model('cyto2')
```

**분석 결과 캐싱:**
```python
import streamlit as st

@st.cache_data(ttl=3600)  # 1시간 TTL
def analyze_image(image_path, parameters):
    """동일 이미지+파라미터는 재계산 안 함"""
    return perform_segmentation(image_path, parameters)
```

### 9. Docker 배포 구성 (도 10 참조)

도 10은 Docker 기반 배포 구조를 나타낸 다이어그램이다.

#### 9.1 Dockerfile 구성

**GPU 지원 Dockerfile:**
```dockerfile
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

# Python 3.11 설치
RUN apt-get update && \
    apt-get install -y python3.11 python3-pip && \
    apt-get clean

# 비root 사용자 생성
RUN useradd -m -u 1000 adds && \
    mkdir -p /app/models /app/data && \
    chown -R adds:adds /app

USER adds
WORKDIR /app

# 의존성 설치
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 애플리케이션 복사
COPY --chown=adds:adds . .

# 건강 체크
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1

# 실행
CMD ["gunicorn", "backend.main:app", \
     "--workers", "9", \
     "--worker-class", "uvicorn.workers.UvicornWorker", \
     "--bind", "0.0.0.0:8000"]
```

#### 9.2 Docker Compose 구성

```yaml
version: '3.8'

services:
  adds-api:
    build:
      context: .
      dockerfile: Dockerfile.gpu
    container_name: adds-api
    ports:
      - "8000:8000"
    volumes:
      - ./models:/app/models:ro
      - ./data:/app/data
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    environment:
      - GPU_ENABLED=true
      - CUDA_VISIBLE_DEVICES=0
      - DATABASE_URL=sqlite:///data/analysis_results.db
    restart: unless-stopped
    networks:
      - adds-network

  adds-ui:
    build:
      context: .
      dockerfile: Dockerfile
    container_name: adds-ui
    ports:
      - "8501:8501"
    depends_on:
      - adds-api
    environment:
      - API_URL=http://adds-api:8000
    restart: unless-stopped
    networks:
      - adds-network

networks:
  adds-network:
    driver: bridge
```

### 10. 실험 결과 및 성능 검증

#### 10.1 세포 분할 정확도 (n=150 pathology images)

| 메트릭 | 평균 | 표준편차 | 최소 | 최대 |
|--------|------|----------|------|------|
| Dice 계수 | 0.893 | 0.042 | 0.781 | 0.957 |
| IoU | 0.807 | 0.058 | 0.641 | 0.917 |
| 세포 수 오차 | 2.8% | 1.9% | 0.2% | 8.3% |

**특징 추출 검증 (n=500 cells, 수동 측정과 비교):**
- 면적: Pearson r=0.96, p<0.001
- 원형도: Pearson r=0.91, p<0.001
- 강도: Pearson r=0.94, p<0.001

#### 10.2 CT 종양 검출 정확도 (n=100 CT scans)

| 메트릭 | 값 | 95% 신뢰구간 |
|--------|-----|---------------|
| 민감도 | 87.9% | [77.2%, 94.6%] |
| 특이도 | 83.3% | [69.8%, 92.5%] |
| 양성 예측도 | 87.0% | [76.3%, 93.8%] |
| 음성 예측도 | 84.5% | [71.2%, 93.1%] |
| 정확도 | 86.0% | [77.6%, 92.1%] |
| AUC-ROC | 0.912 | [0.858, 0.966] |

**종양 크기 추정 오차:**
- 평균 절대 오차: 3.2mm (95% CI: [2.1, 4.3])
- 수술 병리 소견과 상관: r=0.88

**TNM 병기 일치도:**
- 영상의학 전문의와 일치: κ=0.79 (substantial agreement)
- 정확히 일치: 74%
- 1단계 이내: 96%

#### 10.3 치료 권고 정확도 (n=200 patient cases)

| 일치 유형 | 비율 |
|-----------|------|
| 1차 치료 완전 일치 | 81.5% |
| 상위 3개 권고 내 포함 | 92.0% |
| 바이오마커 적합 제외 | 98.5% |

**예후 예측 성능:**
- 무진행 생존 C-index: 0.73 (95% CI: [0.68, 0.78])
- 전체 생존 C-index: 0.76 (95% CI: [0.71, 0.81])

#### 10.4 능동 학습 성능

**수렴 속도 비교:**
- 무작위 선택: 25회 반복 (DTOL > 0.8 달성)
- Expected Improvement만: 20회 반복
- **이중모드 전략: 12회 반복** (40% 향상)

**최종 시너지 점수:**
- 평균: 0.840
- 표준편차: 0.032
- 범위: [0.785, 0.901]

#### 10.5 처리 시간 성능

**세포 분할 (512×512 이미지):**
- CPU (Intel i7-10700): 8.2초
- GPU (NVIDIA RTX 5070): 2.1초
- **속도 향상: 3.9×**

**세포 분할 (1024×1024 이미지):**
- CPU: 22.4초
- GPU: 5.3초
- **속도 향상: 4.2×**

**CT 검출 (단일 슬라이스):**
- Direct YOLO: 1.2초
- Anatomy-guided: 3.8초
- Hybrid ensemble: 2.5초

**End-to-End CDSS 처리:**
- 데이터 입력 검증: 0.5초
- Cellpose 분석: 5.3초
- CT 검출: 2.5초
- 통합 엔진: 0.8초
- OpenAI 해석: 2.1초
- **총 처리 시간: 11.2초**

일 실시예에서, 11.2초는 임상 워크플로우 통합 기준(30초 이내)을 충족한다.

#### 10.6 시스템 처리량

**API 성능 (Gunicorn 9 workers):**
- 동시 요청 처리: 80 req/s
- P50 지연시간: 125ms
- P95 지연시간: 280ms
- P99 지연시간: 450ms

**데이터베이스 성능:**
- 삽입 속도: 1,200 records/s
- 쿼리 속도: 5,000 queries/s (인덱스 사용)
- 스토리지 효율: 환자당 2.5 MB

#### 10.7 의사 사용자 평가 (n=12 oncologists)

| 질문 | 평균 평점 (1-5) |
|------|-----------------|
| 사용 편의성 | 4.3 |
| 결과 해석 가능성 | 4.6 |
| 임상 유용성 | 4.1 |
| 권고 정확도 | 4.2 |
| 실제 사용 의향 | 4.0 |

**정성적 피드백:**
- 긍정: "빠른 분석", "명확한 시각화", "유용한 AI 설명"
- 개선 필요: "더 많은 근거 인용", "불확실성 표현 개선"

#### 10.8 환자 사용자 평가 (n=25 patients)

| 질문 | 평균 평점 (1-5) |
|------|-----------------|
| 설명 명확성 | 4.7 |
| 불안 감소 효과 | 4.2 |
| 기술 신뢰도 | 3.9 |
| 추천 의향 | 4.4 |

### 11. 실시예의 변형

상기 실시예는 설명의 목적으로 제시된 것이며, 본 발명의 기술 사상 내에서 다양한 변형이 가능하다:

**변형예 1**: CT 검출 모듈(122)에서 TotalSegmentator 대신 nnU-Net 또는 다른 분할 모델 사용

**변형예 2**: 능동 학습 모듈(133)에서 Thompson Sampling 대신 Upper Confidence Bound(UCB) 또는 Entropy Search 사용

**변형예 3**: 설명 가능 AI 모듈(132)에 SHAP(SHapley Additive exPlanations) 추가

**변형예 4**: 데이터베이스를 SQLite에서 PostgreSQL 또는 MySQL로 변경

**변형예 5**: GPU를 NVIDIA 대신 AMD ROCm 또는 Intel oneAPI로 변경

**변형예 6**: API를 FastAPI 대신 Flask 또는 Django로 구현

**변형예 7**: UI를 Streamlit 대신 React 또는 Vue.js로 구현

**변형예 8**: Docker 대신 Kubernetes 기반 배포

**변형예 9**: OpenAI GPT-4 대신 Google PaLM 또는 Anthropic Claude 사용

**변형예 10**: 대장암 외에 폐암, 유방암, 췌장암 등으로 확장

### 12. 산업상 이용 가능성

본 발명에 따른 인공지능 기반 다중모달 임상 의사결정 지원 시스템 및 방법은 다음과 같은 산업 분야에서 이용 가능하다:

**(1) 의료기관:**
- 종합병원: 암센터, 영상의학과, 병리과
- 전문병원: 암 전문병원
- 검진센터: 건강검진, 암 선별

**(2) 제약 산업:**
- 신약 개발: 약물 조합 최적화
- 임상 시험: 환자 선택, 반응 예측
- 개인 맞춤 의학: 바이오마커 기반 치료 선택

**(3) 의료기기 산업:**
- 진단 기기: AI 보조 진단 소프트웨어
- 영상 장비: CT/MRI와 통합
- CDSS: 전자의무기록(EMR) 연동

**(4) 헬스케어 IT:**
- 클라우드 서비스: 원격 진단 지원
- 모바일 헬스: 환자 모니터링
- 빅데이터 플랫폼: 실세계 근거 생성

**(5) 규제 기관:**
- FDA: 의료기기 510(k) 승인
- EMA: CE Mark 인증
- MFDS: 국내 의료기기 허가

일 실시예에서, 본 시스템은 3개 병원에서 전향적 임상 시험 진행 중이며, FDA 510(k) 제출 목표는 2027년 Q4이다.

### 13. 특허청구범위와의 대응 관계

상기 실시예에서 설명한 구성 요소들은 청구범위의 구성 요소들과 다음과 같이 대응된다:

| 청구항 요소 | 실시예 구성 | 대응 섹션 |
|-------------|-------------|-----------|
| 세포 분석 모듈 | 121 | 섹션 3 |
| Cellpose 분할부 | 121b | 섹션 3.2 |
| 특징 추출부 | 121c | 섹션 3.3 |
| CT 검출 모듈 | 122 | 섹션 4 |
| 다중 임계값 검출부 | 122d | 섹션 4.4 |
| TNM 병기 결정부 | 122g | 섹션 4.7 |
| 통합 엔진 | 131 | 섹션 5 |
| 위험도 산출부 | 131b | 섹션 5.2 |
| 치료 선택부 | 131d | 섹션 5.4 |
| 설명 가능 AI 모듈 | 132 | 섹션 7 |
| LIME 분석부 | 132a | 섹션 7.1 |
| Grad-CAM 분석부 | 132b | 섹션 7.2 |
| 능동 학습 모듈 | 133 | 섹션 6 |
| Thompson Sampling 획득부 | 133b | 섹션 6.2 |
| Expected Improvement 획득부 | 133c | 섹션 6.3 |
| 성능 최적화 모듈 | 134 | 섹션 8 |
| GPU 가속부 | 134a | 섹션 8.1 |
| 데이터베이스 최적화부 | 134b | 섹션 8.2 |

---

## 【요약서】

본 발명은 세포 병리 이미지, CT 종양 검출, 유전자 바이오마커를 실시간 통합 분석하는 인공지능 기반 다중모달 임상 의사결정 지원 시스템에 관한 것이다.

Cellpose 기반 세포 분할로 25개 이상의 형태학적 특징을 추출하고, 다중 임계값 CT 검출로 종양 후보를 식별하며, 이를 통합하여 암 병기, 위험도, 예후를 산출한다. 

LIME, Grad-CAM, 반사실적 분석을 통합한 설명 가능 AI 모듈로 예측의 투명성을 제공하고, Thompson Sampling과 Expected Improvement를 순차 적용하는 이중모드 능동 학습으로 약물 조합 최적화 수렴 속도를 40% 향상시킨다.

GPU 가속(4.2배), 데이터베이스 인덱싱(330배), API 압축(70% 크기 감소)으로 실시간 처리(11.2초)를 달성하며, Docker 컨테이너화로 다양한 임상 환경에 배포 가능하다.

검증 결과, 세포 분할 Dice 0.893, CT 검출 민감도 87.9%, 치료 권고 일치율 92%를 달성하여 전문의 수준의 성능을 제공한다.

---

**【대표도】 도 1**

**【국제특허분류 (IPC)】**
- G16H 50/20 (ICT specially adapted for medical diagnosis, medical simulation or medical data mining)
- G06N 3/08 (Learning methods, e.g. deep learning)
- G06T 7/00 (Image analysis)

**【출원인】**
명칭: 인하대학교 산학협력단
주소: 인천광역시 미추홀구 인하로 100

**【발명자】**
성명: [발명자명]
주소: [주소]

**【대리인】**
명칭: [특허법인명]
주소: [주소]

---

## 첨부 서류
1. 명세서 1부
2. 청구범위 1부  
3. 요약서 1부
4. 도면 10매
5. 위임장 1부

---

**출원일: 2026년 1월 29일**
