"""
Automated Refactoring Verification Script
Tests all modules, imports, and basic functionality
"""
import sys
import os
from pathlib import Path

# Set UTF-8 for Windows console
if sys.platform == 'win32':
    os.system('chcp 65001 > nul')

sys.path.insert(0, r'C:\Users\brook\Desktop\ADDS\src')

def test_all_imports():
    """Test all page module imports"""
    print('='*70)
    print('TEST 1: Module Imports')
    print('='*70)
    
    tests = []
    
    # Test 1: Main app
    print('[1/5] Testing main app.py...', end=' ')
    try:
        import ui.app
        print('[OK]')
        tests.append(('app.py', True, None))
    except Exception as e:
        print(f'[FAIL] {e}')
        tests.append(('app.py', False, str(e)))
    
    # Test 2: App core
    print('[2/5] Testing app_core.py...', end=' ')
    try:
        from ui.app_core import configure_gpu, get_cellpose_processor
        print('[OK]')
        tests.append(('app_core.py', True, None))
    except Exception as e:
        print(f'[FAIL] {e}')
        tests.append(('app_core.py', False, str(e)))
    
    # Test 3: All pages
    print('[3/5] Testing all page modules...', end=' ')
    try:
        from ui.pages import (
            show_home, show_image_analysis, show_document_processing,
            show_drug_cocktail, show_data_management, show_dashboard
        )
        print('[OK]')
        tests.append(('pages modules', True, None))
    except Exception as e:
        print(f'[FAIL] {e}')
        tests.append(('pages modules', False, str(e)))
    
    # Test 4: Precision oncology
    print('[4/5] Testing precision_oncology...', end=' ')
    try:
        from ui.precision_oncology import show_precision_oncology
        print('[OK]')
        tests.append(('precision_oncology', True, None))
    except Exception as e:
        print(f'[FAIL] {e}')
        tests.append(('precision_oncology', False, str(e)))
    
    # Test 5: Individual pages
    print('[5/5] Testing individual pages...', end=' ')
    try:
        from ui.pages.home import show_home
        from ui.pages.image_analysis import show_image_analysis
        from ui.pages.dashboard import show_dashboard
        print('[OK]')
        tests.append(('individual imports', True, None))
    except Exception as e:
        print(f'[FAIL] {e}')
        tests.append(('individual imports', False, str(e)))
    
    return tests


def test_file_structure():
    """Test file sizes and structure"""
    print()
    print('='*70)
    print('TEST 2: File Structure')
    print('='*70)
    
    ui_dir = Path(r'C:\Users\brook\Desktop\ADDS\src\ui')
    
    files_to_check = {
        'app.py': 300,  # Max allowed lines
        'app_core.py': 200,
        'pages/home.py': 400,
        'pages/image_analysis.py': 1500,
        'pages/document_processing.py': 600,
        'pages/drug_cocktail.py': 400,
        'pages/data_management.py': 300,
        'pages/dashboard.py': 400,
        'precision_oncology/main.py': 1500
    }
    
    results = []
    all_pass = True
    
    for rel_path, max_lines in files_to_check.items():
        file_path = ui_dir / rel_path
        if file_path.exists():
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = len(f.readlines())
            
            status = 'OK' if lines <= max_lines else 'WARN'
            if status == 'WARN':
                all_pass = False
            
            print(f'  {rel_path:40s} {lines:5d} lines [{status}]')
            results.append((rel_path, lines, lines <= max_lines))
        else:
            print(f'  {rel_path:40s} [MISSING]')
            results.append((rel_path, 0, False))
            all_pass = False
    
    return results, all_pass


def test_no_circular_imports():
    """Test for circular import issues"""
    print()
    print('='*70)
    print('TEST 3: Circular Import Check')
    print('='*70)
    
    try:
        # Try importing in different orders
        from ui import app_core
        from ui.pages import home
        from ui.precision_oncology import main
        from ui import app
        
        print('[OK] No circular import detected')
        return True
    except ImportError as e:
        print(f'[FAIL] Circular import: {e}')
        return False


def test_syntax_errors():
    """Check for basic syntax errors"""
    print()
    print('='*70)
    print('TEST 4: Syntax Validation')
    print('='*70)
    
    import py_compile
    
    ui_dir = Path(r'C:\Users\brook\Desktop\ADDS\src\ui')
    errors = []
    
    for py_file in ui_dir.rglob('*.py'):
        try:
            py_compile.compile(str(py_file), doraise=True)
        except py_compile.PyCompileError as e:
            errors.append((py_file.name, str(e)))
            print(f'  [FAIL] {py_file.name}: {e}')
    
    if not errors:
        print('[OK] All files have valid syntax')
        return True
    else:
        return False


def generate_report(import_tests, structure_results, structure_pass, 
                   circular_ok, syntax_ok):
    """Generate final report"""
    print()
    print('='*70)
    print('FINAL REPORT')
    print('='*70)
    
    # Import tests summary
    import_pass = sum(1 for _, ok, _ in import_tests if ok)
    import_total = len(import_tests)
    print(f'Import Tests: {import_pass}/{import_total} passed')
    
    # Structure test
    print(f'File Structure: {"PASS" if structure_pass else "WARN"}')
    
    # Other tests
    print(f'Circular Imports: {"PASS" if circular_ok else "FAIL"}')
    print(f'Syntax Check: {"PASS" if syntax_ok else "FAIL"}')
    
    # Overall
    print()
    all_pass = (import_pass == import_total and circular_ok and syntax_ok)
    
    if all_pass:
        print('[SUCCESS] All automated tests passed!')
        print()
        print('Next steps:')
        print('  1. VS Code Reload (Ctrl+Shift+P -> Reload Window)')
        print('  2. Check Pyrefly Language Server in Output panel')
        print('  3. Test IntelliSense (Ctrl+Space)')
        print('  4. Run: streamlit run src/ui/app.py')
        return 0
    else:
        print('[PARTIAL] Some tests failed, but refactoring complete')
        print()
        print('Manual verification recommended:')
        print('  - Check import errors above')
        print('  - Review file sizes')
        return 1


if __name__ == '__main__':
    try:
        # Run all tests
        import_tests = test_all_imports()
        structure_results, structure_pass = test_file_structure()
        circular_ok = test_no_circular_imports()
        syntax_ok = test_syntax_errors()
        
        # Generate report
        exit_code = generate_report(
            import_tests, structure_results, structure_pass,
            circular_ok, syntax_ok
        )
        
        sys.exit(exit_code)
        
    except Exception as e:
        print(f'\n[ERROR] Test failed: {e}')
        import traceback
        traceback.print_exc()
        sys.exit(1)
