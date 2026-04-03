"""
Integration Test Suite for ADDS
Tests full system functionality including API, UI components, and Docker deployment
"""

import requests
import time
from pathlib import Path
import sys

class ADDSIntegrationTest:
    """Comprehensive integration testing for ADDS system"""
    
    def __init__(self):
        self.api_url = "http://localhost:8000"
        self.ui_url = "http://localhost:8501"
        self.results = []
    
    def log(self, test_name: str, passed: bool, message: str = ""):
        """Log test result"""
        status = "PASS" if passed else "FAIL"
        self.results.append((test_name, passed, message))
        symbol = "+" if passed else "x"
        print(f"{symbol} [{status}] {test_name}")
        if message:
            print(f"  -> {message}")
    
    def test_api_health(self):
        """Test API health endpoint"""
        try:
            response = requests.get(f"{self.api_url}/health", timeout=5)
            passed = response.status_code == 200 and response.json().get('status') == 'healthy'
            self.log("API Health Check", passed, f"Status: {response.status_code}")
        except Exception as e:
            self.log("API Health Check", False, str(e))
    
    def test_api_docs(self):
        """Test API documentation accessibility"""
        try:
            response = requests.get(f"{self.api_url}/api/docs", timeout=5)
            passed = response.status_code == 200
            self.log("API Documentation", passed)
        except Exception as e:
            self.log("API Documentation", False, str(e))
    
    def test_api_endpoints(self):
        """Test API endpoint availability"""
        try:
            response = requests.get(f"{self.api_url}/", timeout=5)
            data = response.json()
            endpoints = data.get('endpoints', {})
            
            required_endpoints = ['segmentation', 'features', 'statistics', 'synergy']
            all_present = all(ep in str(endpoints) for ep in required_endpoints)
            
            self.log("API Endpoints", all_present, f"Found: {list(endpoints.keys())}")
        except Exception as e:
            self.log("API Endpoints", False, str(e))
    
    def test_streamlit_health(self):
        """Test Streamlit UI health"""
        try:
            response = requests.get(f"{self.ui_url}/_stcore/health", timeout=5)
            passed = response.status_code == 200
            self.log("Streamlit Health", passed)
        except Exception as e:
            self.log("Streamlit Health", False, str(e))
    
    def test_database_exists(self):
        """Test database file existence"""
        db_path = Path("data/analysis_results.db")
        passed = db_path.parent.exists()
        self.log("Database Directory", passed, f"Path: {db_path.parent}")
    
    def test_model_cache_dir(self):
        """Test model cache directory"""
        models_dir = Path("models")
        passed = models_dir.exists()
        self.log("Models Directory", passed, f"Path: {models_dir}")
    
    def test_logs_dir(self):
        """Test logs directory"""
        logs_dir = Path("logs")
        passed = logs_dir.exists()
        self.log("Logs Directory", passed, f"Path: {logs_dir}")
    
    def test_docker_files(self):
        """Test Docker configuration files"""
        files = ["Dockerfile", "docker-compose.yml", ".dockerignore"]
        missing = [f for f in files if not Path(f).exists()]
        passed = len(missing) == 0
        self.log("Docker Files", passed, f"Missing: {missing}" if missing else "All present")
    
    def test_utility_modules(self):
        """Test new utility modules exist"""
        modules = [
            "src/utils/process_monitor.py",
            "src/utils/cache_manager.py",
            "src/utils/gpu_monitor.py",
            "src/utils/history_manager.py",
            "src/utils/system_utils.py"
        ]
        missing = [m for m in modules if not Path(m).exists()]
        passed = len(missing) == 0
        self.log("Utility Modules", passed, f"Missing: {missing}" if missing else "All present")
    
    def test_requirements_files(self):
        """Test requirements files exist"""
        files = ["requirements.txt", "backend/requirements.txt"]
        missing = [f for f in files if not Path(f).exists()]
        passed = len(missing) == 0
        self.log("Requirements Files", passed, f"Missing: {missing}" if missing else "All present")
    
    def run_all_tests(self):
        """Run all integration tests"""
        print("=" * 60)
        print("ADDS Integration Test Suite")
        print("=" * 60)
        print()
        
        print("File System Tests:")
        print("-" * 60)
        self.test_database_exists()
        self.test_model_cache_dir()
        self.test_logs_dir()
        self.test_docker_files()
        self.test_utility_modules()
        self.test_requirements_files()
        
        print()
        print("Service Tests:")
        print("-" * 60)
        self.test_api_health()
        self.test_api_docs()
        self.test_api_endpoints()
        self.test_streamlit_health()
        
        print()
        print("=" * 60)
        print("Test Summary")
        print("=" * 60)
        
        total = len(self.results)
        passed = sum(1 for _, p, _ in self.results if p)
        failed = total - passed
        
        print(f"Total Tests: {total}")
        print(f"Passed: {passed} ({passed/total*100:.1f}%)")
        print(f"Failed: {failed} ({failed/total*100:.1f}%)")
        
        if failed > 0:
            print()
            print("Failed Tests:")
            for name, p, msg in self.results:
                if not p:
                    print(f"  - {name}: {msg}")
        
        print()
        print("=" * 60)
        return failed == 0

if __name__ == '__main__':
    tester = ADDSIntegrationTest()
    success = tester.run_all_tests()
    sys.exit(0 if success else 1)
