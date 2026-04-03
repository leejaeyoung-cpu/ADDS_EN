# Setup Guide

## 환경 설정

### 1. Python 환경

Python 3.10 이상이 필요합니다.

```bash
python --version  # 3.10+ 확인
```

### 2. 가상환경 생성 (권장)

```bash
cd C:\Users\brook\Desktop\AIDATA\ADDS

# 가상환경 생성
python -m venv venv

# 활성화 (Windows)
venv\Scripts\activate

# 활성화 (Linux/Mac)
source venv/bin/activate
```

### 3. 의존성 설치

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

주요 패키지 설치 시간: 약 5-10분

## 데이터베이스 설정 (선택사항)

ADDS는 PostgreSQL을 사용합니다. 데이터베이스를 사용하지 않고도 대부분의 기능을 사용할 수 있습니다.

### PostgreSQL 설치

1. [PostgreSQL 다운로드](https://www.postgresql.org/download/)
2. 설치 후 데이터베이스 생성:

```bash
# PostgreSQL 서비스 시작
# Windows: 자동 시작
# Linux: sudo systemctl start postgresql

# 데이터베이스 생성
createdb adds_db

# 스키마 적용
psql adds_db < configs/database_schema.sql
```

### 환경 변수 설정

`.env` 파일을 프로젝트 루트에 생성:

```env
ADDS_DB_PASSWORD=your_secure_password
```

## GPU 설정 (선택사항)

Cellpose 및 딥러닝 모델에서 GPU를 사용하려면:

### CUDA 설치

1. [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads) 설치 (11.8 이상 권장)
2. PyTorch CUDA 버전 설치:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

GPU 확인:

```python
import torch
print(torch.cuda.is_available())  # True면 성공
```

## 빠른 시작

### 1. Streamlit UI 실행

```bash
streamlit run src/ui/app.py
```

브라우저에서 `http://localhost:8501` 접속

### 2. 예제 스크립트 실행

```bash
python examples/complete_pipeline.py
```

### 3. Jupyter Notebook

```bash
jupyter notebook notebooks/quick_start.ipynb
```

## 테스트 실행

```bash
# 모든 테스트 실행
python tests/test_core.py

# 또는 pytest 사용
pytest tests/ -v
```

## 디렉토리 구조 확인

```bash
# Windows
tree /F ADDS

# Linux/Mac
tree ADDS
```

예상 출력:
```
ADDS/
├── src/
│   ├── data/
│   ├── models/
│   ├── preprocessing/
│   ├── evaluation/
│   ├── utils/
│   └── ui/
├── data/
├── configs/
├── examples/
├── notebooks/
├── tests/
└── README.md
```

## 문제 해결

### 패키지 설치 오류

```bash
# pip 업그레이드
python -m pip install --upgrade pip

# 캐시 삭제 후 재설치
pip cache purge
pip install -r requirements.txt --no-cache-dir
```

### Cellpose 오류

```bash
# Cellpose 재설치
pip uninstall cellpose
pip install cellpose --no-deps
pip install -r requirements.txt
```

### GPU 인식 안됨

```bash
# PyTorch 버전 확인
python -c "import torch; print(torch.__version__)"

# CUDA 사용 가능 확인
python -c "import torch; print(torch.cuda.is_available())"
```

## 다음 단계

1. [README.md](README.md) - 전체 프로젝트 개요
2. [Walkthrough](docs/walkthrough.md) - 상세 가이드
3. [Quick Start Notebook](notebooks/quick_start.ipynb) - 실습 튜토리얼

## 지원

문제가 발생하면:
1. [GitHub Issues](https://github.com/your-org/ADDS/issues) (if applicable)
2. 인하대학병원 의생명공학과 연구팀 문의
