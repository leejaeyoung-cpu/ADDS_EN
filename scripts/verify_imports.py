# ADDS Phase 2 검증 스크립트

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

def test_imports():
    """핵심 모듈 import 검증"""
    results = {}
    
    # 1. UI Module
    try:
        from ui import app
        results['ui.app'] = 'OK'
    except Exception as e:
        results['ui.app'] = f'ERROR: {str(e)[:50]}'
    
    # 2. Image Processor
    try:
        from preprocessing.image_processor import CellposeProcessor
        results['image_processor'] = 'OK'
    except Exception as e:
        results['image_processor'] = f'ERROR: {str(e)[:50]}'
    
    # 3. Report Generator
    try:
        from preprocessing.report_generator import AnalysisReportGenerator
        results['report_generator'] = 'OK'
    except Exception as e:
        results['report_generator'] = f'ERROR: {str(e)[:50]}'
    
    # 4. Database
    try:
        from utils.analysis_db import AnalysisDatabase
        results['analysis_db'] = 'OK'
    except Exception as e:
        results['analysis_db'] = f'ERROR: {str(e)[:50]}'
    
    # 5. CDSS
    try:
        from medical_imaging.cdss import integration_engine
        results['cdss'] = 'OK'
    except Exception as e:
        results['cdss'] = f'ERROR: {str(e)[:50]}'
    
    return results

def print_results(results):
    """결과 출력"""
    print("\n=== Module Import Test Results ===\n")
    
    ok_count = sum(1 for v in results.values() if v == 'OK')
    total = len(results)
    
    for module, status in results.items():
        symbol = '[OK]' if status == 'OK' else '[FAIL]'
        print(f"{symbol} {module:30s} {status}")
    
    print(f"\n=== Summary: {ok_count}/{total} modules OK ===\n")
    
    return ok_count == total

if __name__ == '__main__':
    results = test_imports()
    success = print_results(results)
    sys.exit(0 if success else 1)
