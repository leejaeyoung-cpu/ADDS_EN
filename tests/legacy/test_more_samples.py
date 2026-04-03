"""
Test Multiple CT Samples with Improved Quality
===============================================
"""

import sys
from pathlib import Path

project_root = Path(r'c:\Users\brook\Desktop\ADDS')
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / 'txt'))

from txt.tumor_detection_pipeline import TumorDetectionPipeline

def main():
    print("=" * 80)
    print("Multiple Sample Quality Test")
    print("=" * 80)
    
    ctdcm_dir = project_root / 'CTdcm'
    output_dir = project_root / 'outputs' / 'quality_test_multi'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Test additional files from OpenAI reference
    test_files = ['20004.dcm', '20005.dcm', '20010.dcm', '20015.dcm', '20020.dcm']
    
    pipeline = TumorDetectionPipeline(
        min_area_mm2=20.0,
        max_area_mm2=5000.0,
        hu_range=(-50, 150)
    )
    
    results = []
    for filename in test_files:
        filepath = ctdcm_dir / filename
        if not filepath.exists():
            print(f"\nSkip: {filename} not found")
            continue
        
        print(f"\n[Processing] {filename}")
        print("-" * 60)
        
        try:
            dicom_img, candidates = pipeline.process_single(
                str(filepath),
                str(output_dir)
            )
            results.append((filename, len(candidates)))
            
            if candidates:
                high_conf = [c for c in candidates if c.confidence_score > 0.5]
                max_conf = max(c.confidence_score for c in candidates) if candidates else 0
                print(f"OK: {len(candidates)} candidates, {len(high_conf)} high-conf, max={max_conf:.2f}")
            else:
                print("OK: No candidates detected")
                
        except Exception as e:
            print(f"ERROR: {str(e)[:100]}")
    
    print("\n" + "=" * 80)
    print("Summary")
    print("=" * 80)
    for filename, count in results:
        print(f"  {filename}: {count} candidates")
    
    print(f"\nOutput folder: {output_dir}")
    print(f"Generated {len(results)} visualization files")

if __name__ == '__main__':
    main()
