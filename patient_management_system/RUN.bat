@echo off
REM Patient Management System - Quick Start
REM One-click launcher

echo ========================================
echo  ADDS Patient Management System
echo ========================================
echo.

REM Kill existing Python processes
taskkill /F /IM python.exe >nul 2>&1
timeout /t 1 /nobreak >nul

REM Navigate to directory
cd /d "%~dp0"

REM Start server
echo Starting server at http://localhost:8000
echo.
echo Press Ctrl+C to stop
echo.

python app.py

pause
