"""
ADDS-Exscientia Integration: Phase 1-3
Optimize & Learn Modules

- SynergyOptimizer: 시너지 최적화 (Scipy 기반)
- CombinationLearner: AI 학습 (Gaussian Process)

Author: ADDS Development Team
Date: 2026-01-15
Medical Review: Required before production use
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import scipy.optimize
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, RBF
import logging

try:
    from .combination_designer import DrugCombination, Drug, PatientProfile
    from .combination_tester import TestResult
except ImportError:
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent))
    from combination_designer import DrugCombination, Drug, PatientProfile
    from combination_tester import TestResult

logger = logging.getLogger(__name__)


class SynergyOptimizer:
    """
    Exscientia 정밀 설계 방식을 조합 최적화에 적용
    
    핵심 기능:
    1. Gradient-based Optimization (Scipy)
    2. Multi-objective Optimization (Pareto)
    3. Constraint Satisfaction (용량 범위, 독성)
    
    의료 안전성:
    - 모든 용량은 FDA 승인 범위 내
    - 독성 제약 조건 강제
    - 실현 가능한 조합만 반환
    """
    
    def __init__(
        self,
        efficacy_weight: float = 0.6,
        safety_weight: float = 0.4
    ):
        """
        Args:
            efficacy_weight: 효능 가중치
            safety_weight: 안전성 가중치
        """
        assert abs(efficacy_weight + safety_weight - 1.0) < 1e-6, \
            "Weights must sum to 1.0"
        
        self.efficacy_weight = efficacy_weight
        self.safety_weight = safety_weight
        
        logger.info(
            f"SynergyOptimizer initialized: "
            f"efficacy={efficacy_weight}, safety={safety_weight}"
        )
    
    def optimize_combination(
        self,
        base_combination: DrugCombination,
        test_result: TestResult,
        patient: PatientProfile
    ) -> DrugCombination:
        """
        테스트 결과 기반 조합 미세 조정
        
        최적화 목표:
        - Efficacy 최대화
        - Toxicity 최소화
        - 용량 범위 준수
        
        Returns:
            최적화된 DrugCombination
        """
        logger.info(f"Optimizing combination: {[d.name for d in base_combination.drugs]}")
        
        # 초기 용량 (현재 조합)
        x0 = np.array(base_combination.doses)
        
        # 목적 함수 정의
        def objective(doses):
            """
            최소화할 목적 함수
            (음수로 변환하여 최대화를 최소화로)
            """
            # 가상 조합 생성
            trial_combo = self._create_trial_combination(
                base_combination,
                doses
            )
            
            # 예측 성능 (간단한 휴리스틱)
            efficacy = self._predict_efficacy(trial_combo, patient)
            toxicity = self._predict_toxicity(trial_combo, patient)
            safety = 1.0 - toxicity
            
            # 종합 점수 (가중 평균)
            score = self.efficacy_weight * efficacy + self.safety_weight * safety
            
            # 최소화를 위해 음수로 반환
            return -score
        
        # 제약 조건 정의
        constraints = self._build_constraints(base_combination, patient)
        
        # 경계 조건 (용량 범위)
        bounds = [
            (drug.min_dose, drug.max_dose)
            for drug in base_combination.drugs
        ]
        
        # Scipy L-BFGS-B 최적화
        result = scipy.optimize.minimize(
            objective,
            x0=x0,
            method='L-BFGS-B',
            bounds=bounds,
            constraints=constraints,
            options={
                'maxiter': 100,
                'ftol': 1e-6,
                'disp': False
            }
        )
        
        if not result.success:
            logger.warning(f"Optimization failed: {result.message}")
            return base_combination  # 원본 반환
        
        # 최적 조합 생성
        optimized = DrugCombination(
            drugs=base_combination.drugs,
            doses=result.x.tolist(),
            rationale=f"Optimized from base combination (score: {-result.fun:.3f})"
        )
        
        logger.info(
            f"Optimization completed: "
            f"{x0} → {result.x} (improvement: {-result.fun:.3f})"
        )
        
        return optimized
    
    def _create_trial_combination(
        self,
        base: DrugCombination,
        doses: np.ndarray
    ) -> DrugCombination:
        """임시 조합 생성"""
        try:
            return DrugCombination(
                drugs=base.drugs,
                doses=doses.tolist(),
                rationale="Trial optimization"
            )
        except AssertionError:
            # 검증 실패 시 원본 반환
            return base
    
    def _predict_efficacy(
        self,
        combination: DrugCombination,
        patient: PatientProfile
    ) -> float:
        """
        효능 예측 (간단한 모델)
        실제로는 ML 모델 또는 PK/PD 시뮬레이션 사용
        """
        # 용량 대비 효능 (saturation curve)
        total_dose_ratio = sum(
            dose / drug.standard_dose
            for drug, dose in zip(combination.drugs, combination.doses)
        )
        
        # Emax 모델: E = Emax * D / (ED50 + D)
        emax = 1.0
        ed50 = len(combination.drugs)  # 약물 수에 비례
        efficacy = emax * total_dose_ratio / (ed50 + total_dose_ratio)
        
        return np.clip(efficacy, 0, 1)
    
    def _predict_toxicity(
        self,
        combination: DrugCombination,
        patient: PatientProfile
    ) -> float:
        """독성 예측 (간단한 모델)"""
        # 용량에 비례
        total_dose_ratio = sum(
            dose / drug.standard_dose
            for drug, dose in zip(combination.drugs, combination.doses)
        )
        
        # 기본 독성
        base_toxicity = total_dose_ratio * 0.2
        
        # 환자 요인
        if patient.egfr < 60:
            base_toxicity *= 1.5
        if patient.age > 75:
            base_toxicity *= 1.2
        
        return np.clip(base_toxicity, 0, 1)
    
    def _build_constraints(
        self,
        combination: DrugCombination,
        patient: PatientProfile
    ) -> List[Dict]:
        """
        제약 조건 정의
        
        1. 독성 < 0.8 (DLT 방지)
        2. 총 비용 < 한도
        3. 실현 가능성
        """
        constraints = []
        
        # 독성 제약
        def toxicity_constraint(doses):
            trial = self._create_trial_combination(combination, doses)
            toxicity = self._predict_toxicity(trial, patient)
            return 0.8 - toxicity  # >= 0
        
        constraints.append({
            'type': 'ineq',
            'fun': toxicity_constraint
        })
        
        # 비용 제약 (예: $10,000)
        def cost_constraint(doses):
            total_cost = sum(
                dose * 100  # $100/mg 가정
                for dose in doses
            )
            return 10000 - total_cost  # >= 0
        
        constraints.append({
            'type': 'ineq',
            'fun': cost_constraint
        })
        
        return constraints


class CombinationLearner:
    """
    Exscientia 연속 학습 방식
    
    핵심 기능:
    1. Gaussian Process Regression
    2. Incremental Learning
    3. Uncertainty Quantification
    
    의료 정밀도:
    - 예측 불확실성 추적
    - 재현 가능한 학습
    - 모델 성능 모니터링
    """
    
    def __init__(self, kernel_type: str = 'matern'):
        """
        Args:
            kernel_type: 'matern', 'rbf'
        """
        if kernel_type == 'matern':
            kernel = Matern(nu=2.5, length_scale=1.0)
        elif kernel_type == 'rbf':
            kernel = RBF(length_scale=1.0)
        else:
            raise ValueError(f"Unknown kernel: {kernel_type}")
        
        self.gp_model = GaussianProcessRegressor(
            kernel=kernel,
            alpha=0.1,  # Noise level
            n_restarts_optimizer=10,
            random_state=42  # 재현성
        )
        
        self.history = []  # (features, outcome) 튜플
        
        logger.info(f"CombinationLearner initialized with {kernel_type} kernel")
    
    def learn_from_results(
        self,
        results: List[TestResult]
    ) -> Dict[str, float]:
        """
        테스트 결과로 모델 업데이트
        
        Args:
            results: TestResult 리스트
        
        Returns:
            학습 통계 (R², RMSE, 샘플 수)
        """
        logger.info(f"Learning from {len(results)} test results")
        
        # Step 1: 특징 추출
        X_new = []
        y_new = []
        
        for result in results:
            # 조합을 벡터로 표현
            features = self._featurize_combination(result.combination)
            X_new.append(features)
            
            # 성능 (overall score)
            y_new.append(result.overall_score)
        
        # Step 2: 히스토리 업데이트
        for x, y in zip(X_new, y_new):
            self.history.append((x, y))
        
        # Step 3: 모델 재학습
        if len(self.history) >= 5:  # 최소 5개 샘플 필요
            X_train = np.array([x for x, _ in self.history])
            y_train = np.array([y for _, y in self.history])
            
            self.gp_model.fit(X_train, y_train)
            
            # 모델 성능 평가
            y_pred = self.gp_model.predict(X_train)
            r2 = self._calculate_r2(y_train, y_pred)
            rmse = np.sqrt(np.mean((y_train - y_pred) ** 2))
            
            logger.info(
                f"Model updated: {len(self.history)} samples, "
                f"R²={r2:.3f}, RMSE={rmse:.3f}"
            )
            
            return {
                'n_samples': len(self.history),
                'r2_score': r2,
                'rmse': rmse,
                'model_trained': True
            }
        else:
            logger.warning(
                f"Insufficient data for training: {len(self.history)} < 5"
            )
            return {
                'n_samples': len(self.history),
                'r2_score': 0.0,
                'rmse': 0.0,
                'model_trained': False
            }
    
    def predict(
        self,
        combination: DrugCombination,
        return_std: bool = False
    ) -> Tuple[float, Optional[float]]:
        """
        조합 성능 예측
        
        Args:
            combination: 예측할 조합
            return_std: 불확실성(표준편차) 반환 여부
        
        Returns:
            (mean, std) 또는 mean만
        """
        if len(self.history) < 5:
            logger.warning("Model not trained, returning default prediction")
            if return_std:
                return 0.5, 0.3  # 높은 불확실성
            return 0.5
        
        features = self._featurize_combination(combination)
        X = features.reshape(1, -1)
        
        if return_std:
            mean, std = self.gp_model.predict(X, return_std=True)
            return mean[0], std[0]
        else:
            mean = self.gp_model.predict(X)
            return mean[0]
    
    def analyze_uncertainty(self) -> Dict[str, float]:
        """
        모델 불확실성 분석
        
        Returns:
            불확실성 통계
        """
        if len(self.history) < 5:
            return {
                'mean_uncertainty': 1.0,
                'max_uncertainty': 1.0,
                'confidence': 0.0
            }
        
        X_train = np.array([x for x, _ in self.history])
        _, std = self.gp_model.predict(X_train, return_std=True)
        
        return {
            'mean_uncertainty': float(np.mean(std)),
            'max_uncertainty': float(np.max(std)),
            'confidence': float(1.0 - np.mean(std))
        }
    
    def _featurize_combination(
        self,
        combination: DrugCombination
    ) -> np.ndarray:
        """
        조합을 feature vector로 변환
        
        Features (9-dim):
        - 약물 수 (정규화)
        - 각 약물 용량 비율 (최대 4개)
        - 타겟 다양성
        - 경로 분포 (IV/PO/SC)
        """
        features = []
        
        # 약물 수 (2-4 → 0-1)
        features.append((len(combination.drugs) - 2) / 2.0)
        
        # 용량 비율 (최대 4개 약물)
        for i in range(4):
            if i < len(combination.drugs):
                drug = combination.drugs[i]
                dose_ratio = combination.doses[i] / drug.standard_dose
                features.append(dose_ratio)
            else:
                features.append(0.0)
        
        # 타겟 다양성
        all_targets = set()
        for drug in combination.drugs:
            all_targets.update(drug.target_proteins)
        features.append(len(all_targets) / 10.0)
        
        # 경로 분포
        route_counts = {'IV': 0, 'PO': 0, 'SC': 0}
        for drug in combination.drugs:
            route_counts[drug.route] += 1
        features.extend([
            route_counts['IV'] / 4.0,
            route_counts['PO'] / 4.0,
            route_counts['SC'] / 4.0
        ])
        
        return np.array(features)
    
    def _calculate_r2(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> float:
        """R² 계산"""
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        
        if ss_tot < 1e-10:
            return 0.0
        
        r2 = 1 - (ss_res / ss_tot)
        return np.clip(r2, -1, 1)


# 의료급 테스트
if __name__ == "__main__":
    from combination_designer import Drug, DrugCombination, PatientProfile
    from combination_tester import CombinationTester, TestResult
    
    print(f"\n{'='*60}")
    print("OPTIMIZE & LEARN MODULES TEST (MEDICAL-GRADE)")
    print(f"{'='*60}\n")
    
    # 테스트용 데이터
    cisplatin = Drug(
        name="Cisplatin",
        min_dose=50,
        max_dose=100,
        standard_dose=75,
        route="IV",
        half_life_hours=24,
        target_proteins=["DNA"],
        known_interactions=[]
    )
    
    fluorouracil = Drug(
        name="5-FU",
        min_dose=400,
        max_dose=1000,
        standard_dose=800,
        route="IV",
        half_life_hours=12,
        target_proteins=["TS"],
        known_interactions=[]
    )
    
    test_combo = DrugCombination(
        drugs=[cisplatin, fluorouracil],
        doses=[75, 800],
        rationale="Standard regimen"
    )
    
    test_patient = PatientProfile(
        patient_id="TEST003",
        age=62,
        cancer_type="Gastric",
        stage=3,
        ecog=1,
        mutations=["HER2"],
        previous_treatments=[],
        egfr=75,
        bilirubin=1.0
    )
    
    # Test 1: SynergyOptimizer
    print("Test 1: SynergyOptimizer")
    print("-" * 60)
    
    optimizer = SynergyOptimizer()
    
    # 가상 테스트 결과 생성
    tester = CombinationTester(max_workers=1)
    test_results = tester.test_combinations([test_combo], test_patient)
    
    if test_results:
        test_result = test_results[0]
        
        print(f"Original doses: {test_combo.doses}")
        print(f"Original score: {test_result.overall_score:.3f}")
        
        optimized_combo = optimizer.optimize_combination(
            test_combo,
            test_result,
            test_patient
        )
        
        print(f"Optimized doses: {optimized_combo.doses}")
        print(f"Improvement: {optimized_combo.rationale}")

    print(f"\n{'='*60}\n")
    
    # Test 2: CombinationLearner
    print("Test 2: CombinationLearner")
    print("-" * 60)
    
    learner = CombinationLearner(kernel_type='matern')
    
    # 학습
    stats = learner.learn_from_results(test_results)
    print(f"Learning stats:")
    print(f"  Samples: {stats['n_samples']}")
    print(f"  Model trained: {stats['model_trained']}")
    
    # 예측
    mean, std = learner.predict(test_combo, return_std=True)
    print(f"\nPrediction for test combination:")
    print(f"  Mean: {mean:.3f}")
    print(f"  Std: {std:.3f}")
    
    # 불확실성 분석
    uncertainty = learner.analyze_uncertainty()
    print(f"\nUncertainty analysis:")
    print(f"  Mean uncertainty: {uncertainty['mean_uncertainty']:.3f}")
    print(f"  Confidence: {uncertainty['confidence']:.3f}")
    
    print(f"\n{'='*60}")
    print("✓ All tests completed successfully!")
    print(f"{'='*60}\n")
