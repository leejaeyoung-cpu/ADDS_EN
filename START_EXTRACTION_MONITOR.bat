@echo off
REM Real-time Literature Extraction Monitor
REM Run this in a separate terminal window to watch extraction progress

echo ========================================
echo   LITERATURE EXTRACTION MONITOR
echo ========================================
echo.
echo This window will show real-time extraction progress.
echo Press Ctrl+C to stop monitoring.
echo.
echo Starting monitor in 3 seconds...
timeout /t 3 /nobreak >nul

python scripts/monitor_extraction.py --interval 2.0

pause
