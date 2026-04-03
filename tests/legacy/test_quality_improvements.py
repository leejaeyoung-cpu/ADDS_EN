"""
Test Improved Visualization Quality
=====================================
Quick test to compare quality improvements with OpenAI reference
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(r'c:\Users\brook\Desktop\ADDS')
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / 'txt'))

from txt.tumor_detection_pipeline import TumorDetectionPipeline

def main():
    print("=" * 80)
    print("CT 종양 검출 품질 테스트 (OpenAI-level Quality)")
    print("=" * 80)
    print("")
    
    # Test with CTdcm samples
    ctdcm_dir = project_root / 'CTdcm'
    output_dir = project_root / 'outputs' / 'quality_test'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Input: {ctdcm_dir}")
    print(f"Output: {output_dir}")
    print("")
    
    # Initialize pipeline
    pipeline = TumorDetectionPipeline(
        min_area_mm2=20.0,
        max_area_mm2=5000.0,
        hu_range=(-50, 150)
    )
    
    # Test with specific files from OpenAI reference
    test_files = ['20003.dcm', '20004.dcm', '20005.dcm']
    
    results = []
    for filename in test_files:
        filepath = ctdcm_dir / filename
        if not filepath.exists():
            print(f"⚠ File not found: {filename}")
            continue
        
        print(f"\n처리중: {filename}")
        print("-" * 60)
        
        try:
            dicom_img, candidates = pipeline.process_single(
                str(filepath),
                str(output_dir)
            )
            results.append((dicom_img, candidates))
            
            print(f"✓ 완료: {len(candidates)} candidates detected")
            if candidates:
                high_conf = [c for c in candidates if c.confidence_score > 0.5]
                print(f"  High confidence (>0.5): {len(high_conf)}")
                if high_conf:
                    max_conf = max(c.confidence_score for c in high_conf)
                    print(f"  Max confidence: {max_conf:.2f}")
        except Exception as e:
            print(f"✗ Error: {e}")
    
    print("\n" + "=" * 80)
    print("테스트 완료")
    print("=" * 80)
    print(f"\n결과 확인: {output_dir}")
    print("\n비교 항목:")
    print("  1. 이미지가 선명한가? (고해상도, DPI 300)")
    print("  2. Bounding box가 명확한가? (굵은 빨간색 테두리)")
    print("  3. 텍스트가 읽기 쉬운가? (큰 폰트, 볼드)")
    print("  4. 전체적인 품질이 OpenAI 참조와 유사한가?")
    print("")

if __name__ == '__main__':
    main()
