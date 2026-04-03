"""
Unit Tests for CombinationTester Module

의료급 테스트:
- PK/PD 모델 검증
- 독성 등급 정확성
- RECIST 기준 준수
- QC 시스템 작동

Author: ADDS Development Team
Date: 2026-01-15
"""

import unittest
import numpy as np
from pathlib import Path
import sys

# src 디렉토리를 path에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / 'src'))

from recommendation.combination_designer import Drug, DrugCombination, PatientProfile
from recommendation.combination_tester import (
    CombinationTester, PKProfile, PDProfile, ToxicityProfile,
    PKPDSimulator, TestResult, TestStatus
)


class TestPKPDSimulator(unittest.TestCase):
    """PKPDSimulator 테스트"""
    
    def setUp(self):
        """테스트 환경 설정"""
        self.simulator = PKPDSimulator()
        
        self.drug = Drug(
            name="Cisplatin",
            min_dose=50,
            max_dose=100,
            standard_dose=75,
            route="IV",
            half_life_hours=24,
            target_proteins=["DNA"],
            known_interactions=[]
        )
        
        self.patient = PatientProfile(
            patient_id="TEST001",
            age=60,
            cancer_type="Gastric",
            stage=3,
            ecog=1,
            mutations=["HER2"],
            previous_treatments=[],
            egfr=80,
            bilirubin=0.8
        )
    
    def test_pk_simulation(self):
        """PK 시뮬레이션 정상 작동"""
        pk = self.simulator.simulate_pk(
            drug=self.drug,
            dose=75,
            patient=self.patient
        )
        
        # 필수 속성 확인
        self.assertEqual(pk.drug_name, "Cisplatin")
        self.assertGreater(pk.auc, 0)
        self.assertGreater(pk.cmax, 0)
        self.assertGreater(pk.half_life, 0)
        
        # 농도는 시간에 따라 감소
        self.assertGreater(pk.concentrations[0], pk.concentrations[-1])
    
    def test_pk_renal_adjustment(self):
        """신기능 저하 시 반감기 증가"""
        # 정상 신기능
        normal_patient = PatientProfile(
            patient_id="NORMAL",
            age=60,
            cancer_type="Gastric",
            stage=3,
            ecog=1,
            mutations=[],
            previous_treatments=[],
            egfr=90,  # 정상
            bilirubin=0.8
        )
        
        # 신기능 저하
        ckd_patient = PatientProfile(
            patient_id="CKD",
            age=60,
            cancer_type="Gastric",
            stage=3,
            ecog=1,
            mutations=[],
            previous_treatments=[],
            egfr=30,  # CKD Stage 3
            bilirubin=0.8
        )
        
        pk_normal = self.simulator.simulate_pk(self.drug, 75, normal_patient)
        pk_ckd = self.simulator.simulate_pk(self.drug, 75, ckd_patient)
        
        # CKD 환자는 반감기가 더 길어야 함
        self.assertGreater(pk_ckd.half_life, pk_normal.half_life)
    
    def test_pd_simulation(self):
        """PD 시뮬레이션 정상 작동"""
        pk = self.simulator.simulate_pk(self.drug, 75, self.patient)
        pd = self.simulator.simulate_pd([pk], self.patient)
        
        # 종양 크기는 감소해야 함 (약물 효과)
        self.assertLess(pd.tumor_volumes[-1], pd.tumor_volumes[0])
        
        # 세포 생존율 감소
        self.assertLess(pd.cell_viability[-1], pd.cell_viability[0])
    
    def test_toxicity_simulation(self):
        """독성 시뮬레이션"""
        drug2 = Drug("5-FU", 400, 1000, 800, "IV", 12, ["TS"], [])
        
        combo = DrugCombination(
            drugs=[self.drug, drug2],  # 2제 조합
            doses=[75, 800],
            rationale="Test"
        )
        
        pk1 = self.simulator.simulate_pk(self.drug, 75, self.patient)
        pk2 = self.simulator.simulate_pk(drug2, 800, self.patient)
        tox = self.simulator.simulate_toxicity([pk1, pk2], combo, self.patient)
        
        # 모든 독성 카테고리 존재
        self.assertIsInstance(tox.hematologic, dict)
        self.assertIsInstance(tox.hepatic, dict)
        self.assertIsInstance(tox.renal, dict)
        
        # Grade 범위 (0-5)
        max_grade = tox.calculate_max_grade()
        self.assertGreaterEqual(max_grade, 0)
        self.assertLessEqual(max_grade, 5)


class TestToxicityProfile(unittest.TestCase):
    """ToxicityProfile 클래스 테스트"""
    
    def test_max_grade_calculation(self):
        """최대 등급 계산"""
        tox = ToxicityProfile(
            hematologic={'Neutropenia': 3, 'Anemia': 2},
            hepatic={'AST': 1},
            renal={'Creatinine': 2},
            cardiac={'QTc': 0},
            gastrointestinal={'Nausea': 2}
        )
        
        self.assertEqual(tox.calculate_max_grade(), 3)
    
    def test_dlt_detection(self):
        """DLT (용량 제한 독성) 감지"""
        # Grade 4 혈액학적 독성 = DLT
        tox_dlt = ToxicityProfile(
            hematologic={'Neutropenia': 4},
            hepatic={},
            renal={},
            cardiac={},
            gastrointestinal={}
        )
        
        self.assertTrue(tox_dlt.is_dose_limiting_toxicity())
        
        # Grade 2는 DLT 아님
        tox_safe = ToxicityProfile(
            hematologic={'Neutropenia': 2},
            hepatic={},
            renal={},
            cardiac={},
            gastrointestinal={}
        )
        
        self.assertFalse(tox_safe.is_dose_limiting_toxicity())


class TestPDProfile(unittest.TestCase):
    """PDProfile 클래스 테스트"""
    
    def test_recist_cr(self):
        """Complete Response 판정"""
        pd = PDProfile(
            time_points=np.array([0, 7, 14, 21, 28]),
            tumor_volumes=np.array([1000, 500, 100, 10, 0]),  # 100% 감소
            cell_viability=np.array([1.0, 0.5, 0.1, 0.01, 0.0]),
            biomarker_levels={}
        )
        
        self.assertEqual(pd.calculate_response(), "CR")
    
    def test_recist_pr(self):
        """Partial Response 판정"""
        pd = PDProfile(
            time_points=np.array([0, 7, 14, 21, 28]),
            tumor_volumes=np.array([1000, 800, 600, 500, 600]),  # 40% 감소
            cell_viability=np.array([1.0, 0.8, 0.6, 0.5, 0.6]),
            biomarker_levels={}
        )
        
        self.assertEqual(pd.calculate_response(), "PR")
    
    def test_recist_pd(self):
        """Progressive Disease 판정"""
        pd = PDProfile(
            time_points=np.array([0, 7, 14, 21, 28]),
            tumor_volumes=np.array([1000, 1100, 1300, 1600, 2200]),  # 120% 증가
            cell_viability=np.array([1.0, 1.1, 1.3, 1.6, 2.2]),
            biomarker_levels={}
        )
        
        self.assertEqual(pd.calculate_response(), "PD")


class TestCombinationTester(unittest.TestCase):
    """CombinationTester 클래스 테스트"""
    
    def setUp(self):
        """테스트 환경 설정"""
        self.tester = CombinationTester(max_workers=2)
        
        self.drugs = [
            Drug("Cisplatin", 50, 100, 75, "IV", 24, ["DNA"], []),
            Drug("5-FU", 400, 1000, 800, "IV", 12, ["TS"], [])
        ]
        
        self.combo = DrugCombination(
            drugs=self.drugs,
            doses=[75, 800],
            rationale="Standard"
        )
        
        self.patient = PatientProfile(
            patient_id="TEST001",
            age=60,
            cancer_type="Colorectal",
            stage=3,
            ecog=1,
            mutations=["KRAS"],
            previous_treatments=[],
            egfr=80,
            bilirubin=0.8
        )
    
    def test_single_combination_test(self):
        """단일 조합 테스트"""
        results = self.tester.test_combinations([self.combo], self.patient)
        
        self.assertEqual(len(results), 1)
        result = results[0]
        
        # 필수 속성
        self.assertIsInstance(result, TestResult)
        self.assertEqual(result.status, TestStatus.COMPLETED)
        self.assertTrue(result.quality_passed)
        
        # 점수 범위
        self.assertGreaterEqual(result.efficacy_score, 0)
        self.assertLessEqual(result.efficacy_score, 1)
        self.assertGreaterEqual(result.safety_score, 0)
        self.assertLessEqual(result.safety_score, 1)
    
    def test_quality_check(self):
        """품질 관리 검사"""
        results = self.tester.test_combinations([self.combo], self.patient)
        
        for result in results:
            # QC 통과한 결과만 반환되어야 함
            self.assertTrue(result.quality_passed)
            
            # PK 검증
            for pk in result.pk_profiles:
                self.assertGreater(pk.auc, 0)
                self.assertFalse(np.isnan(pk.auc))
                self.assertFalse(np.isinf(pk.auc))
            
            # PD 검증
            self.assertFalse(np.any(np.isnan(result.pd_profile.tumor_volumes)))
            self.assertTrue(np.all(result.pd_profile.tumor_volumes >= 0))
    
    def test_parallel_processing(self):
        """병렬 처리 작동"""
        # 여러 조합 동시 테스트
        combos = []
        for _ in range(5):
            combos.append(DrugCombination(
                drugs=self.drugs,
                doses=[75, 800],
                rationale="Test"
            ))
        
        results = self.tester.test_combinations(combos, self.patient)
        
        # 모두 성공해야 함 (QC 통과)
        self.assertGreater(len(results), 0)
        self.assertLessEqual(len(results), 5)


def run_tests():
    """모든 테스트 실행"""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    suite.addTests(loader.loadTestsFromTestCase(TestPKPDSimulator))
    suite.addTests(loader.loadTestsFromTestCase(TestToxicityProfile))
    suite.addTests(loader.loadTestsFromTestCase(TestPDProfile))
    suite.addTests(loader.loadTestsFromTestCase(TestCombinationTester))
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    print("\n" + "="*70)
    print("TEST SUMMARY - CombinationTester")
    print("="*70)
    print(f"Tests run: {result.testsRun}")
    print(f"Successes: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    
    if result.wasSuccessful():
        print("\n[OK] All tests passed!")
    else:
        print("\n[FAIL] Some tests failed!")
    
    print("="*70)
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
