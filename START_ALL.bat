@echo off
REM ADDS 통합 시스템 시작 (UI + Backend + GPU 설정)
REM GPU 환경 변수 설정 포함 버전

echo ========================================
echo  ADDS - 통합 시스템 실행
echo  UI + Backend API + GPU 강제 설정
echo ========================================
echo.

REM Kill existing processes
echo [1/5] 기존 프로세스 종료 중...
taskkill /F /IM streamlit.exe >nul 2>&1
taskkill /F /IM python.exe >nul 2>&1
timeout /t 2 /nobreak >nul
echo   ✓ 기존 프로세스 종료 완료
echo.

REM Navigate to project directory
cd /d "%~dp0"

REM Set GPU environment variables (CRITICAL!)
echo [2/5] GPU 환경 변수 설정 중...
set CUDA_VISIBLE_DEVICES=0
set HIP_VISIBLE_DEVICES=-1
set CUDA_DEVICE_ORDER=PCI_BUS_ID
echo   ✓ CUDA_VISIBLE_DEVICES=0 (NVIDIA RTX only)
echo   ✓ HIP_VISIBLE_DEVICES=-1 (AMD disabled)
echo.

REM Start Backend in new window
echo [3/5] Backend API 시작 중...
start "ADDS Backend API" cmd /k "set CUDA_VISIBLE_DEVICES=0 && set HIP_VISIBLE_DEVICES=-1 && python -m backend.main"
echo   ✓ Backend 시작 (http://localhost:8000)
echo   ✓ API Docs: http://localhost:8000/api/docs
echo.

REM Wait for backend to initialize
echo [4/5] Backend 초기화 대기 중... (5초)
timeout /t 5 /nobreak >nul
echo   ✓ Backend 준비 완료
echo.

REM Start Streamlit UI
echo [5/5] Streamlit UI 시작 중...
echo.
echo ==================================================
echo  ADDS 시스템 실행 완료!
echo ==================================================
echo  Frontend (UI):  http://localhost:8505
echo  Backend (API):  http://localhost:8000
echo  API Docs:       http://localhost:8000/api/docs
echo  
echo  GPU: NVIDIA RTX 5070 (AMD disabled)
echo  
echo  종료: Ctrl+C (이 창) + Backend 창 닫기
echo ==================================================
echo.

streamlit run src\ui\app.py --server.port 8505

pause
