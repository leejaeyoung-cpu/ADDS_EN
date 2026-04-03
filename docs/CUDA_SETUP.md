# CUDA 설정 가이드

## 🎛️ CUDA 토글 기능

이제 환경 변수를 통해 CPU/GPU 모드를 쉽게 전환할 수 있습니다!

### 빠른 시작

#### 1. 토글 유틸리티 사용 (권장)

```bash
# CPU 모드로 전환 (현재 권장)
python src/utils/toggle_cuda.py cpu

# GPU 모드로 전환 (PyTorch sm_120 지원 시)
python src/utils/toggle_cuda.py gpu

# 혼합 모드 (Training은 GPU, Cellpose는 CPU)
python src/utils/toggle_cuda.py mixed

# 현재 설정 확인
python src/utils/toggle_cuda.py status
```

#### 2. 수동 .env 파일 편집

`.env` 파일을 직접 수정하여 설정할 수도 있습니다:

```env
# CPU 모드
ADDS_DEVICE=cpu
ADDS_CELLPOSE_GPU=false

# GPU 모드
ADDS_DEVICE=cuda
ADDS_CELLPOSE_GPU=true

# 혼합 모드
ADDS_DEVICE=cuda
ADDS_CELLPOSE_GPU=false
```

## 📋 모드 설명

### CPU 모드 (현재 권장)
- **설정**: `ADDS_DEVICE=cpu`, `ADDS_CELLPOSE_GPU=false`
- **사용 사례**: RTX 5070 sm_120 호환성 문제 회피
- **장점**: 안정적, 호환성 문제 없음
- **단점**: 느린 처리 속도

```bash
python src/utils/toggle_cuda.py cpu
```

### GPU 모드
- **설정**: `ADDS_DEVICE=cuda`, `ADDS_CELLPOSE_GPU=true`
- **사용 사례**: PyTorch가 sm_120을 지원하게 되면
- **장점**: 최고 성능
- **단점**: 현재 RTX 5070과 호환 불가

```bash
python src/utils/toggle_cuda.py gpu
```

### 혼합 모드
- **설정**: `ADDS_DEVICE=cuda`, `ADDS_CELLPOSE_GPU=false`
- **사용 사례**: Training은 GPU로, Cellpose만 CPU로
- **장점**: 부분적 GPU 가속
- **단점**: Cellpose는 여전히 느림

```bash
python src/utils/toggle_cuda.py mixed
```

## 🔍 시스템 확인

### GPU 상태 확인

```bash
python check_gpu.py
```

출력 예시:
```
==================================================
GPU Status Check
==================================================
PyTorch version: 2.6.0+cu124
CUDA available: True
CUDA version: 12.4
GPU count: 1
GPU name: NVIDIA GeForce RTX 5070 Laptop GPU
GPU memory: 7.96 GB
GPU is READY!
==================================================
```

### 현재 설정 확인

```bash
python src/utils/toggle_cuda.py status
```

출력 예시:
```
📊 Current Settings:
   Training Device: cpu
   Cellpose GPU: false

🔍 System Status:
   PyTorch Version: 2.6.0+cu124
   CUDA Available: True
   GPU Name: NVIDIA GeForce RTX 5070 Laptop GPU
   GPU Memory: 7.96 GB
```

## ⚙️ 작동 원리

1. **환경 변수**: `.env` 파일에서 `ADDS_DEVICE`와 `ADDS_CELLPOSE_GPU`를 읽음
2. **설정 오버라이드**: `src/utils/config.py`가 자동으로 환경 변수를 적용
3. **우선순위**: 환경 변수 > `config.yaml` 기본값

### 코드 예시

```python
from src.utils.config import config

# 자동으로 환경 변수에서 로드됨
device = config.get('training.device')  # 'cpu' or 'cuda'
cellpose_gpu = config.get('cellpose.gpu')  # True or False

print(f"Training on: {device}")
print(f"Cellpose GPU: {cellpose_gpu}")
```

## 🚨 RTX 5070 호환성 이슈

현재 PyTorch 2.6.0은 RTX 5070의 CUDA capability sm_120을 지원하지 않습니다.

### 경고 메시지:
```
NVIDIA GeForce RTX 5070 Laptop GPU with CUDA capability sm_120 is not compatible with the current PyTorch installation.
The current PyTorch install supports CUDA capabilities sm_50 sm_60 sm_61 sm_70 sm_75 sm_80 sm_86 sm_90.
```

### 해결 방법:

1. **CPU 모드 사용 (현재)**: 가장 안정적
   ```bash
   python src/utils/toggle_cuda.py cpu
   ```

2. **PyTorch 업데이트 대기**: sm_120 지원 버전 출시 시
   - PyTorch 공식 사이트 확인: https://pytorch.org/get-started/locally/
   - CUDA 12.4+ 및 sm_120 지원 버전 설치

3. **PyTorch Nightly 설치 시도** (실험적):
   ```bash
   pip install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu124
   ```

## 📝 환경 파일 설명

### `.env.example`
- 템플릿 파일
- Git에 커밋됨
- 새 설정 추가 시 여기에 문서화

### `.env`
- 실제 사용 파일
- Git에서 무시됨 (.gitignore)
- 개인 설정 포함

## 💡 팁

### 빠른 모드 전환

작업 중 모드를 바꾸려면:

```bash
# CPU로 전환
python src/utils/toggle_cuda.py cpu

# 프로그램 재시작 (Streamlit 예시)
streamlit run src/ui/app.py
```

환경 변수는 프로그램 시작 시 로드되므로, 변경 후 반드시 재시작하세요!

### 성능 비교

| 작업 | CPU (Ryzen 9 8940HX) | GPU (RTX 5070) |
|------|----------------------|----------------|
| Cellpose 세그멘테이션 | ~10s/image | ~2s/image* |
| GNN 학습 (100 epochs) | ~30min | ~5min* |
| 배치 추론 | ~5s | ~1s* |

*호환성 문제 해결 시 예상 성능

## 🔗 관련 문서

- [config.yaml](../configs/config.yaml) - 기본 설정
- [config.py](../src/utils/config.py) - 설정 로더
- [toggle_cuda.py](../src/utils/toggle_cuda.py) - 토글 유틸리티
- [check_gpu.py](../check_gpu.py) - GPU 상태 확인
