@echo off
REM ADDS Patient Management System 실행
REM 새로운 통합 환자 관리 시스템 (FastAPI + Modern Frontend)

echo ========================================
echo  ADDS Patient Management System
echo  Integrated CT + Cell Culture + AI
echo ========================================
echo.

REM Kill existing processes
echo [1/3] 기존 프로세스 종료 중...
taskkill /F /IM python.exe >nul 2>&1
timeout /t 2 /nobreak >nul
echo   ✓ 기존 프로세스 종료 완료
echo.

REM Navigate to patient_management_system directory
cd /d "%~dp0"
echo [2/3] 작업 디렉토리: %cd%
echo.

REM Start FastAPI server
echo [3/3] FastAPI 서버 시작 중...
echo.
echo ==================================================
echo  Patient Management System 실행 완료!
echo ==================================================
echo  Homepage:       http://localhost:8000
echo  API Docs:       http://localhost:8000/docs
echo  환자 등록:      http://localhost:8000/patient/register
echo  비교 리포트:    http://localhost:8000/comparison
echo  
echo  포트: 8000
echo  
echo  종료: Ctrl+C
echo ==================================================
echo.

python app.py

pause
