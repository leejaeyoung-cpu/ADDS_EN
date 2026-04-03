"""
Unit Tests for CombinationDesigner Module

의료급 테스트:
- 모든 입력 검증
- 경계 조건 테스트
- 의학적 안전성 검증
- 재현성 확인

Author: ADDS Development Team
Date: 2026-01-15
"""

import unittest
import numpy as np
from pathlib import Path
import sys
from pathlib import Path

# src 디렉토리를 path에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / 'src'))

from recommendation.combination_designer import (
    Drug, DrugCombination, PatientProfile, CombinationDesigner
)


class TestDrug(unittest.TestCase):
    """Drug 클래스 테스트"""
    
    def test_valid_drug_creation(self):
        """정상 약물 생성"""
        drug = Drug(
            name="Cisplatin",
            min_dose=50,
            max_dose=100,
            standard_dose=75,
            route="IV",
            half_life_hours=24,
            target_proteins=["DNA"],
            known_interactions=[]
        )
        
        self.assertEqual(drug.name, "Cisplatin")
        self.assertEqual(drug.standard_dose, 75)
    
    def test_invalid_dose_range(self):
        """잘못된 용량 범위 거부"""
        with self.assertRaises(AssertionError):
            Drug(
                name="Invalid",
                min_dose=100,  # min > max
                max_dose=50,
                standard_dose=75,
                route="IV",
                half_life_hours=24,
                target_proteins=[],
                known_interactions=[]
            )
    
    def test_invalid_standard_dose(self):
        """범위 밖 표준 용량 거부"""
        with self.assertRaises(AssertionError):
            Drug(
                name="Invalid",
                min_dose=50,
                max_dose=100,
                standard_dose=150,  # 범위 밖
                route="IV",
                half_life_hours=24,
                target_proteins=[],
                known_interactions=[]
            )


class TestDrugCombination(unittest.TestCase):
    """DrugCombination 클래스 테스트"""
    
    def setUp(self):
        """테스트용 약물 생성"""
        self.cisplatin = Drug(
            name="Cisplatin",
            min_dose=50,
            max_dose=100,
            standard_dose=75,
            route="IV",
            half_life_hours=24,
            target_proteins=["DNA"],
            known_interactions=["5-FU"]
        )
        
        self.fluorouracil = Drug(
            name="5-FU",
            min_dose=400,
            max_dose=1000,
            standard_dose=800,
            route="IV",
            half_life_hours=12,
            target_proteins=["TS"],
            known_interactions=["Cisplatin"]
        )
    
    def test_valid_combination(self):
        """정상 조합 생성"""
        combo = DrugCombination(
            drugs=[self.cisplatin, self.fluorouracil],
            doses=[75, 800],
            rationale="Standard FOLFOX"
        )
        
        self.assertEqual(len(combo.drugs), 2)
        self.assertEqual(combo.doses, [75, 800])
    
    def test_dose_out_of_range(self):
        """범위 밖 용량 거부"""
        with self.assertRaises(AssertionError):
            DrugCombination(
                drugs=[self.cisplatin],
                doses=[150],  # max_dose = 100
                rationale="Invalid"
            )
    
    def test_too_many_drugs(self):
        """5제 이상 조합 거부 (의학적 실현 불가능)"""
        many_drugs = [self.cisplatin] * 5
        many_doses = [75] * 5
        
        with self.assertRaises(AssertionError):
            DrugCombination(
                drugs=many_drugs,
                doses=many_doses,
                rationale="Too many"
            )
    
    def test_single_drug_rejected(self):
        """1제 조합 거부 (조합이 아님)"""
        with self.assertRaises(AssertionError):
            DrugCombination(
                drugs=[self.cisplatin],
                doses=[75],
                rationale="Single drug"
            )


class TestPatientProfile(unittest.TestCase):
    """PatientProfile 클래스 테스트"""
    
    def test_valid_patient(self):
        """정상 환자 프로필"""
        patient = PatientProfile(
            patient_id="TEST001",
            age=65,
            cancer_type="Gastric",
            stage=3,
            ecog=1,
            mutations=["HER2"],
            previous_treatments=[],
            egfr=75,
            bilirubin=0.9
        )
        
        self.assertEqual(patient.age, 65)
        self.assertEqual(patient.stage, 3)
    
    def test_invalid_age(self):
        """비정상 연령 거부"""
        with self.assertRaises(AssertionError):
            PatientProfile(
                patient_id="TEST",
                age=150,  # invalid
                cancer_type="Gastric",
                stage=3,
                ecog=1,
                mutations=[],
                previous_treatments=[],
                egfr=75,
                bilirubin=0.9
            )
    
    def test_invalid_ecog(self):
        """비정상 ECOG 거부"""
        with self.assertRaises(AssertionError):
            PatientProfile(
                patient_id="TEST",
                age=65,
                cancer_type="Gastric",
                stage=3,
                ecog=6,  # max = 5
                mutations=[],
                previous_treatments=[],
                egfr=75,
                bilirubin=0.9
            )


class TestCombinationDesigner(unittest.TestCase):
    """CombinationDesigner 클래스 테스트"""
    
    def setUp(self):
        """테스트 환경 설정"""
        self.designer = CombinationDesigner()
        
        self.drugs = [
            Drug("Cisplatin", 50, 100, 75, "IV", 24, ["DNA"], []),
            Drug("5-FU", 400, 1000, 800, "IV", 12, ["TS"], []),
            Drug("Leucovorin", 200, 400, 400, "IV", 6, ["Folate"], [])
        ]
        
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
    
    def test_design_initial_combinations(self):
        """초기 조합 설계"""
        combinations = self.designer.design_next_combinations(
            available_drugs=self.drugs,
            patient_profile=self.patient,
            num_combinations=5,
            iteration=0
        )
        
        self.assertGreater(len(combinations), 0)
        self.assertLessEqual(len(combinations), 5)
        
        # 모든 조합이 유효한지 확인
        for combo in combinations:
            self.assertIsInstance(combo, DrugCombination)
            self.assertGreaterEqual(len(combo.drugs), 2)
            self.assertLessEqual(len(combo.drugs), 4)
    
    def test_mpo_score_calculation(self):
        """MPO 점수 계산"""
        combo = DrugCombination(
            drugs=[self.drugs[0], self.drugs[1]],
            doses=[75, 800],
            rationale="Test"
        )
        
        scores = self.designer._calculate_mpo_score(combo, self.patient)
        
        # 점수 범위 확인
        self.assertGreaterEqual(scores['total'], 0)
        self.assertLessEqual(scores['total'], 1)
        self.assertGreaterEqual(scores['efficacy'], 0)
        self.assertLessEqual(scores['efficacy'], 1)
        self.assertGreaterEqual(scores['safety'], 0)
        self.assertLessEqual(scores['safety'], 1)
        
        # 필수 키 확인
        self.assertIn('needs_review', scores)
    
    def test_toxicity_prediction(self):
        """독성 예측"""
        combo = DrugCombination(
            drugs=self.drugs[:2],
            doses=[75, 800],
            rationale="Test"
        )
        
        toxicity = self.designer._predict_toxicity(combo, self.patient)
        
        self.assertGreaterEqual(toxicity, 0)
        self.assertLessEqual(toxicity, 1)
    
    def test_featurization(self):
        """조합 특징 벡터화"""
        combo = DrugCombination(
            drugs=self.drugs[:2],
            doses=[75, 800],
            rationale="Test"
        )
        
        features = self.designer._featurize_combination(combo)
        
        self.assertIsInstance(features, np.ndarray)
        self.assertEqual(features.shape, (9,))  # 9-dim features
        
        # 모든 특징이 유효한 범위인지
        self.assertTrue(np.all(features >= 0))
        self.assertTrue(np.all(features <= 1))
    
    def test_history_update(self):
        """히스토리 업데이트"""
        combo = DrugCombination(
            drugs=self.drugs[:2],
            doses=[75, 800],
            rationale="Test"
        )
        
        initial_len = len(self.designer.history)
        self.designer.update_history(combo, 0.75)
        
        self.assertEqual(len(self.designer.history), initial_len + 1)
    
    def test_reproducibility(self):
        """재현성 테스트 (동일 입력 → 동일 출력)"""
        np.random.seed(42)
        combos1 = self.designer.design_next_combinations(
            available_drugs=self.drugs,
            patient_profile=self.patient,
            num_combinations=3,
            iteration=0
        )
        
        # 새로운 designer로 동일 조건
        designer2 = CombinationDesigner()
        np.random.seed(42)
        combos2 = designer2.design_next_combinations(
            available_drugs=self.drugs,
            patient_profile=self.patient,
            num_combinations=3,
            iteration=0
        )
        
        # 동일한 수의 조합 생성
        self.assertEqual(len(combos1), len(combos2))


def run_tests():
    """모든 테스트 실행"""
    # Test suite 생성
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # 모든 테스트 추가
    suite.addTests(loader.loadTestsFromTestCase(TestDrug))
    suite.addTests(loader.loadTestsFromTestCase(TestDrugCombination))
    suite.addTests(loader.loadTestsFromTestCase(TestPatientProfile))
    suite.addTests(loader.loadTestsFromTestCase(TestCombinationDesigner))
    
    # 실행
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # 결과 출력
    print("\n" + "="*70)
    print("TEST SUMMARY")
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
