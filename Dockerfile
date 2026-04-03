# Dockerfile for ADDS - CPU Version
# Multi-stage build for optimized production image

# Stage 1: Base
FROM python:3.11-slim as base

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Stage 2: Builder
FROM base as builder

# Install build dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .
COPY backend/requirements.txt ./backend/

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir -r backend/requirements.txt

# Download Cellpose models (optional, can be done at runtime)
RUN python -c "from cellpose import models; models.Cellpose(model_type='cyto2', gpu=False)"

# Stage 3: Production
FROM base

# Create non-root user
RUN useradd -m -u 1000 adds && \
    mkdir -p /app /app/data /app/models /app/logs && \
    chown -R adds:adds /app

# Copy Python packages from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin
COPY --from=builder /root/.cellpose /home/adds/.cellpose

# Set working directory
WORKDIR /app

# Copy application code
COPY --chown=adds:adds src/ ./src/
COPY --chown=adds:adds backend/ ./backend/
COPY --chown=adds:adds configs/ ./configs/
COPY --chown=adds:adds .env.example .env

# Switch to non-root user
USER adds

# Environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/home/adds/.local/bin:$PATH" \
    CELLPOSE_CACHE_DIR=/app/models

# Expose ports
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/health', timeout=5)"

# Run API server
CMD ["python", "-m", "uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
