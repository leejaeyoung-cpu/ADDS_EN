@echo off
REM Emergency GPU Fix - Force Kill and Restart with GPU Settings

echo ========================================
echo  EMERGENCY RESTART - GPU FIX
echo ========================================
echo.

REM Kill ALL Python processes
echo [1/4] Killing ALL Python processes...
taskkill /F /IM python.exe >nul 2>&1
taskkill /F /IM streamlit.exe >nul 2>&1
timeout /t 3 /nobreak >nul

REM Verify all killed
echo [2/4] Verifying processes stopped...
tasklist | findstr /i "python.exe streamlit.exe" || echo All processes stopped!
timeout /t 2 /nobreak >nul

REM Set GPU environment variables
echo [3/4] Setting GPU environment variables...
set CUDA_VISIBLE_DEVICES=0
set HIP_VISIBLE_DEVICES=-1
set CUDA_DEVICE_ORDER=PCI_BUS_ID
echo   [OK] CUDA_VISIBLE_DEVICES=%CUDA_VISIBLE_DEVICES%
echo   [OK] HIP_VISIBLE_DEVICES=%HIP_VISIBLE_DEVICES%
echo.

REM Navigate and start
cd /d "%~dp0"

echo [4/4] Starting Streamlit with NVIDIA GPU forced...
echo.
echo ==================================================
echo  GPU FORCED: NVIDIA RTX 5070 ONLY
echo  AMD GPU: BLOCKED
echo  URL: http://localhost:8505
echo ==================================================
echo.

streamlit run src\ui\app.py --server.port 8505

pause
