@echo off
REM ============================================================
REM ADDS Phase 2 Clinical Validation - Real-time Progress
REM ============================================================

echo.
echo ============================================================
echo   ADDS Phase 2 Clinical Validation
echo   Multi-Cancer DTOL Testing with Real-time Progress
echo ============================================================
echo.

cd /d "%~dp0"

echo Starting clinical validation tests...
echo.
echo Test Details:
echo - 5 Cancer Types: Pancreatic, Breast, Colorectal, Lung, Gastric
echo - 2 Strategies per cancer: Dual-Mode and Adaptive
echo - 3 Iterations per strategy
echo - Total: 10 test runs (5 cancers x 2 strategies)
echo.
echo Estimated time: 5-10 minutes
echo.

python tests\test_clinical_validation_realtime.py

echo.
echo ============================================================
echo   Validation Complete!
echo ============================================================
echo.

pause
