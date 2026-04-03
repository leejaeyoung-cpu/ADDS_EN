"""
ADDS 시스템 검증 스크립트
========================
1. 코드 품질 검증
2. 모듈 import 테스트
3. 간단한 기능 테스트
"""

import sys
import os
from pathlib import Path
import numpy as np
import torch
import traceback

print("=" * 80)
print("ADDS 시스템 검증 시작")
print("=" * 80)

# 현재 작업 디렉토리 확인
print(f"\n현재 디렉토리: {os.getcwd()}")
print(f"Python 경로: {sys.executable}")
print(f"Python 버전: {sys.version}")

# GPU 확인
print(f"\n🎮 GPU 상태:")
print(f"  CUDA 사용 가능: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"  GPU 이름: {torch.cuda.get_device_name(0)}")
    print(f"  GPU 메모리: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")

# ============================================================
# 1. 모듈 Import 테스트
# ============================================================
print("\n" + "=" * 80)
print("1️⃣  모듈 Import 테스트")
print("=" * 80)

modules_to_test = [
    ("Candidate Detector", "src.medical_imaging.detection.candidate_detector"),
    ("Optimized Detector", "src.medical_imaging.detection.optimized_detector"),
    ("Integrated Pipeline", "src.medical_imaging.integrated_pipeline"),
    ("UI CT Tumor Analysis", "ui_ct_tumor_analysis"),
]

import_results = {}
for module_name, module_path in modules_to_test:
    try:
        exec(f"import {module_path}")
        print(f"✅ {module_name} - OK")
        import_results[module_name] = True
    except Exception as e:
        print(f"❌ {module_name} - FAILED: {str(e)[:80]}")
        import_results[module_name] = False

# ============================================================
# 2. Candidate Detector 기본 테스트
# ============================================================
print("\n" + "=" * 80)
print("2️⃣  Candidate Detector 기능 테스트")
print("=" * 80)

try:
    from src.medical_imaging.detection.candidate_detector import (
        TumorDetector, TumorCandidate, CTPreprocessor
    )
    
    # 간단한 테스트 데이터 생성
    print("\n📝 테스트 데이터 생성 중...")
    test_slice = np.random.randn(512, 512) * 50 + 40  # HU 값 시뮬레이션
    pixel_spacing = (1.0, 1.0)  # mm
    
    # Detector 초기화
    print("🔧 TumorDetector 초기화 중...")
    detector = TumorDetector(
        min_area_mm2=10.0,
        max_area_mm2=10000.0,
        hu_range=(-50, 200)
    )
    
    # 검출 실행
    print("🔍 종양 후보 검출 중...")
    candidates = detector.detect_candidates_2d(
        test_slice, pixel_spacing, body_mask=None, slice_index=0
    )
    
    print(f"\n✅ 검출 완료!")
    print(f"  검출된 후보 수: {len(candidates)}")
    
    if candidates:
        top_candidate = max(candidates, key=lambda x: x.confidence_score)
        print(f"\n  최고 신뢰도 후보:")
        print(f"    - 신뢰도: {top_candidate.confidence_score:.3f}")
        print(f"    - 면적: {top_candidate.area_mm2:.2f} mm²")
        print(f"    - 평균 HU: {top_candidate.mean_hu:.1f}")
        print(f"    - 원형도: {top_candidate.circularity:.3f}")
    
except Exception as e:
    print(f"❌ Candidate Detector 테스트 실패:")
    print(traceback.format_exc())

# ============================================================
# 3. CT 데이터 확인
# ============================================================
print("\n" + "=" * 80)
print("3️⃣  CT 데이터 확인")
print("=" * 80)

ctdata_path = Path("CTdata")
if ctdata_path.exists():
    ct_files = list(ctdata_path.glob("*.nii.gz")) + list(ctdata_path.glob("*.nii"))
    print(f"✅ CTdata 폴더 존재")
    print(f"  NIfTI 파일 수: {len(ct_files)}")
    
    if ct_files:
        print(f"\n  첫 3개 파일:")
        for f in ct_files[:3]:
            size_mb = f.stat().st_size / 1024 / 1024
            print(f"    - {f.name} ({size_mb:.1f} MB)")
else:
    print(f"⚠️  CTdata 폴더 없음")

# ============================================================
# 4. 모델 파일 확인
# ============================================================
print("\n" + "=" * 80)
print("4️⃣  학습된 모델 확인")
print("=" * 80)

model_files = [
    "best_model.pth",
    "checkpoint_epoch_0.pth",
]

for model_file in model_files:
    model_path = Path(model_file)
    if model_path.exists():
        size_mb = model_path.stat().st_size / 1024 / 1024
        print(f"✅ {model_file} - {size_mb:.0f} MB")
    else:
        print(f"❌ {model_file} - 없음")

# ============================================================
# 5. 종합 결과
# ============================================================
print("\n" + "=" * 80)
print("📊 종합 검증 결과")
print("=" * 80)

total_tests = len(import_results)
passed_tests = sum(import_results.values())

print(f"\n모듈 Import: {passed_tests}/{total_tests} 성공")

# 각 모듈별 상태
print(f"\n상세 결과:")
for module, status in import_results.items():
    status_icon = "✅" if status else "❌"
    print(f"  {status_icon} {module}")

# 전체 평가
if passed_tests == total_tests:
    grade = "A+ (완벽)"
    emoji = "🎉"
elif passed_tests >= total_tests * 0.8:
    grade = "B+ (양호)"
    emoji = "👍"
elif passed_tests >= total_tests * 0.5:
    grade = "C (보통)"
    emoji = "⚠️"
else:
    grade = "D (문제 있음)"
    emoji = "❌"

print(f"\n{emoji} 종합 평가: {grade}")
print(f"성공률: {passed_tests/total_tests*100:.0f}%")

print("\n" + "=" * 80)
print("검증 완료")
print("=" * 80)
