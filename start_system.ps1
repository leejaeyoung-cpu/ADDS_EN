# ADDS 통합 시작 스크립트
# Frontend (Streamlit) + Backend (FastAPI) 동시 실행

Write-Host "======================================" -ForegroundColor Cyan
Write-Host "  ADDS 통합 시스템 시작" -ForegroundColor Cyan
Write-Host "======================================" -ForegroundColor Cyan
Write-Host ""

# 1. Backend 시작
Write-Host "[1/2] Starting Backend (FastAPI)..." -ForegroundColor Yellow
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd '$PSScriptRoot\backend'; python main.py"
Start-Sleep -Seconds 3

# 2. Frontend 시작
Write-Host "[2/2] Starting Frontend (Streamlit)..." -ForegroundColor Yellow
Start-Sleep -Seconds 2
streamlit run src/ui/app.py

Write-Host ""
Write-Host "System started!" -ForegroundColor Green
Write-Host "Frontend: http://localhost:8501" -ForegroundColor Cyan
Write-Host "Backend: http://localhost:8000" -ForegroundColor Cyan
Write-Host "API Docs: http://localhost:8000/api/docs" -ForegroundColor Cyan
