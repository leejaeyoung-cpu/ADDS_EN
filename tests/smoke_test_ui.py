#!/usr/bin/env python
"""UI 스모크 테스트"""
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path.cwd() / 'src'))

print("="*60)
print("ADDS UI Smoke Test")
print("="*60)

def test_imports():
    """필수 모듈 import 테스트"""
    print("\n[1/4] Testing imports...")
    try:
        from ui.page_modules.image_analysis import show_image_analysis
        from ui.page_modules.home import show_home
        print("✅ All UI module imports successful")
        return True
    except Exception as e:
        print(f"❌ Import failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_tumor_analyzer():
    """TumorAnalyzer 초기화 테스트"""
    print("\n[2/4] Testing TumorAnalyzer...")
    try:
        sys.path.append(str(Path.cwd()))
        from analyze_tumor_location import TumorAnalyzer
        
        # 초기화 테스트
        test_dir = Path("CTdata_cleaned")
        if not test_dir.exists():
            test_dir = Path("CTdata")
        
        if test_dir.exists():
            analyzer = TumorAnalyzer(str(test_dir))
            print(f"✅ TumorAnalyzer initialized with {test_dir}")
            return True
        else:
            print("⚠️ No CT data directory found, skipping")
            return True
    except Exception as e:
        print(f"❌ TumorAnalyzer failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_dependencies():
    """핵심 dependencies 테스트"""
    print("\n[3/4] Testing dependencies...")
    try:
        import streamlit
        import pandas
        import numpy
        import cv2
        from PIL import Image
        import plotly
        import torch
        
        print(f"✅ streamlit: {streamlit.__version__}")
        print(f"✅ pandas: {pandas.__version__}")
        print(f"✅ numpy: {numpy.__version__}")
        print(f"✅ opencv: {cv2.__version__}")
        print(f"✅ torch: {torch.__version__}")
        return True
    except Exception as e:
        print(f"❌ Dependency check failed: {e}")
        return False

def test_file_structure():
    """파일 구조 확인"""
    print("\n[4/4] Testing file structure...")
    required_files = [
        "src/ui/app.py",
        "src/ui/page_modules/image_analysis.py",
        "analyze_tumor_location.py",
        "configs/sota_training_config.yaml"
    ]
    
    all_exist = True
    for file_path in required_files:
        path = Path(file_path)
        if path.exists():
            print(f"✅ {file_path}")
        else:
            print(f"❌ {file_path} NOT FOUND")
            all_exist = False
    
    return all_exist

if __name__ == "__main__":
    print()
    tests = [
        test_imports,
        test_tumor_analyzer,
        test_dependencies,
        test_file_structure
    ]
    
    results = []
    for test in tests:
        try:
            results.append(test())
        except Exception as e:
            print(f"❌ Test crashed: {e}")
            results.append(False)
    
    passed = sum(results)
    total = len(results)
    
    print("\n" + "="*60)
    print(f"RESULTS: {passed}/{total} tests passed")
    print("="*60)
    
    if passed == total:
        print("✅ ALL TESTS PASSED")
        sys.exit(0)
    else:
        print("⚠️ SOME TESTS FAILED")
        sys.exit(1)
