#!/bin/bash
# ADDS Streamlit 실행 스크립트 - NVIDIA GPU 전용
# GPU 0 (NVIDIA RTX 5070)만 사용하도록 설정

echo "========================================"
echo "  ADDS - AI Anticancer Drug Discovery"
echo "  GPU Mode: NVIDIA RTX 5070 (GPU 0)"
echo "========================================"
echo ""

# NVIDIA GPU만 사용하도록 환경 변수 설정
export CUDA_VISIBLE_DEVICES=0

echo "[GPU 설정]"
echo "- CUDA_VISIBLE_DEVICES = 0"
echo "- Target: NVIDIA GeForce RTX 5070"
echo ""

echo "[Streamlit 시작]"
cd "$(dirname "$0")"
streamlit run src/ui/app.py
