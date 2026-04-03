FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

WORKDIR /app

# 시스템 패키지
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3-pip \
    libpq-dev \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Python 패키지
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 기존 ADDS 코드 복사
COPY ../../step1_3d_connected_components.py ./utils/
COPY ../../src/visualization/pyvista_3d_viewer.py ./utils/
COPY ../../detect_tumors_inha_corrected.py ./detection/

# 애플리케이션 코드
COPY . .

CMD ["python3", "main.py"]
