@echo off
REM ADDS Backend Startup Script

echo ======================================
echo  ADDS FastAPI Backend Server
echo ======================================
echo.

cd backend

echo [1/3] Checking Python environment...
python --version
echo.

echo [2/3] Starting FastAPI server...
echo Server will be available at: http://localhost:8000
echo API Documentation: http://localhost:8000/api/docs
echo.

echo [3/3] Launching server...
python main.py

pause
