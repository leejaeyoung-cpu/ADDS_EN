@echo off
REM ADDS Clinical System - Docker Start Script
REM 물리학 기반 항암제 최적화 시스템

echo ========================================
echo ADDS Clinical System - Docker Setup
echo ========================================
echo.

REM 1. 환경 변수 확인
if not exist .env.clinical (
    echo [ERROR] .env.clinical not found!
    echo Please create .env.clinical from .env.clinical.example
    pause
    exit /b 1
)

echo [1/5] Copying environment variables...
copy /Y .env.clinical .env
echo.

REM 2. 데이터 폴더 생성
echo [2/5] Creating data directories...
if not exist data\incoming mkdir data\incoming
if not exist data\processed mkdir data\processed
if not exist data\postgres mkdir data\postgres
if not exist data\minio mkdir data\minio
if not exist data\redis mkdir data\redis
echo Data directories created
echo.

REM 3. Docker Compose 빌드
echo [3/5] Building Docker images...
docker-compose -f docker-compose.clinical.yml build
if errorlevel 1 (
    echo [ERROR] Docker build failed!
    pause
    exit /b 1
)
echo.

REM 4. Docker Compose 시작
echo [4/5] Starting services...
docker-compose -f docker-compose.clinical.yml up -d
if errorlevel 1 (
    echo [ERROR] Docker start failed!
    pause
    exit /b 1
)
echo.

REM 5. 상태 확인
echo [5/5] Checking service health...
timeout /t 10 /nobreak
docker-compose -f docker-compose.clinical.yml ps
echo.

echo ========================================
echo Services started successfully!
echo ========================================
echo.
echo PostgreSQL: localhost:5432
echo MinIO Console: http://localhost:9001
echo API Gateway: http://localhost:8001 (coming soon)
echo.
echo Data ingestion watching: ./data/incoming
echo.
echo To view logs:
echo   docker-compose -f docker-compose.clinical.yml logs -f data_ingestion
echo.
echo To stop:
echo   docker-compose -f docker-compose.clinical.yml down
echo.

pause
