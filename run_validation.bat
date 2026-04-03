@echo off
REM ============================================================
REM ADDS 종합 검증 파이프라인 실행 스크립트
REM ============================================================

echo.
echo ============================================================
echo   ADDS 종합 검증 파이프라인
echo ============================================================
echo.

REM 기본 검증 (Backend 실행 불필요)
echo [모드 선택]
echo 1. 기본 검증 (API 제외 - Backend 실행 불필요)
echo 2. 전체 검증 (모든 단계 - Backend 실행 필요)
echo 3. 특정 단계만 실행
echo.

set /p mode="선택 (1-3): "

if "%mode%"=="1" (
    echo.
    echo 기본 검증 실행 중...
    python validate_adds.py --verbose --report
) else if "%mode%"=="2" (
    echo.
    echo 전체 검증 실행 중...
    echo 주의: Backend가 실행 중이어야 합니다 (start_backend.bat^)
    echo.
    pause
    python validate_adds.py --all --verbose --report
) else if "%mode%"=="3" (
    echo.
    echo 실행할 단계 번호를 입력하세요 (쉼표로 구분, 예: 1,2,3):
    echo   1. 환경 및 설정
    echo   2. 의존성
    echo   3. 핵심 모듈
    echo   4. UI 컴포넌트
    echo   5. API 엔드포인트 (Backend 필요^)
    echo   6. 통합 테스트
    echo   7. 데이터 무결성
    echo.
    set /p stages="단계: "
    python validate_adds.py --stages %stages% --verbose --report
) else (
    echo 잘못된 선택입니다.
    pause
    exit /b 1
)

echo.
echo ============================================================
echo   검증 완료!
echo ============================================================
echo.

REM HTML 리포트 자동 열기
if exist validation_report.html (
    echo 📊 HTML 리포트를 브라우저에서 여는 중...
    start validation_report.html
)

echo.
pause
