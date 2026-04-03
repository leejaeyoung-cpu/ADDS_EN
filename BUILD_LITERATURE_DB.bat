@echo off
REM Literature Database Builder
REM 대량 논문 수집 및 RAG 시스템 구축

echo ============================================================
echo ADDS Literature Database Builder
echo ============================================================
echo.

cd /d F:\ADDS\scripts

REM 의존성 설치
echo [1/4] Installing dependencies...
pip install -r requirements_literature.txt
pip install sentence-transformers faiss-cpu

echo.
echo [2/4] Collecting papers from PubMed and ArXiv...
echo This may take 30-60 minutes...
python collect_literature.py

echo.
echo [3/4] Extracting formulas and parameters...
python extract_formulas.py

echo.
echo [4/4] Building RAG system...
python rag_system.py

echo.
echo ============================================================
echo COMPLETE!
echo ============================================================
echo Database location: F:\ADDS\literature_database\
echo.

pause
