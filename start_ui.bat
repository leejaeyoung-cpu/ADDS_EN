@echo off
REM ADDS Streamlit UI Launcher
REM Kills existing Streamlit processes and starts fresh

echo ========================================
echo  ADDS - AI Anticancer Drug Discovery
echo  Streamlit UI Launcher
echo ========================================
echo.

REM Kill existing Streamlit processes
echo [1/3] Stopping existing Streamlit processes...
taskkill /F /IM streamlit.exe >nul 2>&1
for /f "tokens=2" %%i in ('tasklist ^| findstr /i "python.exe"') do (
    wmic process where "ProcessId=%%i AND CommandLine like '%%streamlit%%'" delete >nul 2>&1
)
timeout /t 2 /nobreak >nul

REM Navigate to project directory
echo [2/3] Navigating to project directory...
cd /d "%~dp0"

REM Set GPU environment variables to force NVIDIA GPU (GPU 0)
echo [2.5/3] Setting GPU environment variables...
set CUDA_VISIBLE_DEVICES=0
set HIP_VISIBLE_DEVICES=-1
set CUDA_DEVICE_ORDER=PCI_BUS_ID
echo   - CUDA_VISIBLE_DEVICES=0 (NVIDIA RTX only)
echo   - HIP_VISIBLE_DEVICES=-1 (AMD disabled)

REM Start Streamlit
echo [3/3] Starting Streamlit UI on port 8505...
echo.
echo ==================================================
echo  URL: http://localhost:8505
echo  Press Ctrl+C to stop the server
echo ==================================================
echo.

streamlit run src\ui\app.py --server.port 8505

pause
