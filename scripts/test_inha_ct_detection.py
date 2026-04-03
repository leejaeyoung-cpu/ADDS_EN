#!/usr/bin/env python3
"""
인하대병원 CT 데이터 Detection 테스트
Patient 002227784 복부 CT 종양 검출
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from medical_imaging.ct_analyzer import CTAnalyzer
import pydicom
import numpy as np
from PIL import Image
import json

def test_inha_ct_detection():
    """인하대 CT 데이터로 detection 테스트"""
    
    print("\n" + "="*70)
    print("인하대병원 CT Detection 테스트")
    print("="*70)
    
    # CT Analyzer 초기화
    analyzer = CTAnalyzer(use_gpu=True, use_nnunet=False)
    
    # DICOM 디렉토리
    dicom_dir = Path("CTdata/CTdcm")
    
    # Abdomen Artery 시리즈에서 중간 슬라이스 선택 (종양 가능성 높음)
    # 시리즈 2: Abdomen Artery (120 slices, 20001-20120)
    test_slices = [20040, 20050, 20060, 20070, 20080]  # 중간 부분 샘플링
    
    results = []
    
    for slice_num in test_slices:
        dcm_file = dicom_dir / f"{slice_num}.dcm"
        
        if not dcm_file.exists():
            continue
        
        print(f"\n{'─'*70}")
        print(f"분석 중: {dcm_file.name}")
        
        try:
            # CT 분석 실행
            result = analyzer.analyze_ct_image(
                image_path=str(dcm_file),
                cancer_type="Colorectal",
                additional_context="인하대병원 복부 CT, 동맥기"
            )
            
            # 결과 저장
            results.append({
                'slice': slice_num,
                'file': dcm_file.name,
                'result': result
            })
            
            # 간단한 요약 출력
            print(f"  Tumor Detected: {result.get('tumor_detected', False)}")
            if result.get('tumor_detected'):
                print(f"  Lesion Count: {result.get('lesion_count', 0)}")
                print(f"  Total Volume: {result.get('total_volume_ml', 0):.2f} mL")
        
        except Exception as e:
            print(f"  Error: {e}")
            results.append({
                'slice': slice_num,
                'file': dcm_file.name,
                'error': str(e)
            })
    
    # 전체 결과 요약
    print("\n" + "="*70)
    print("Detection 결과 요약")
    print("="*70)
    
    detected_count = sum(1 for r in results if r.get('result', {}).get('tumor_detected', False))
    total_tested = len([r for r in results if 'result' in r])
    
    print(f"\n테스트한 슬라이스: {total_tested}")
    print(f"종양 검출: {detected_count}")
    print(f"검출률: {detected_count/total_tested*100:.1f}%" if total_tested > 0 else "N/A")
    
    # JSON 저장
    output_file = "inha_ct_detection_results.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n결과 저장: {output_file}")
    print("="*70 + "\n")
    
    return results

if __name__ == '__main__':
    results = test_inha_ct_detection()
    
    print("\n다음 단계:")
    print("1. 검출 결과 시각화 필요")
    print("2. 3D rendering (전체 volume)")
    print("3. 전문가 검증")
