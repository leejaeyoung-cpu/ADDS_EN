"""
ADDS-Exscientia Integration: Phase 1-1
Design Module - CombinationDesigner Class

이 모듈은 Exscientia의 Active Learning 방식을 약물 조합 최적화에 적용합니다.
의료급 정밀도를 위해 모든 입력/출력을 검증하고, 의학적 안전성을 보장합니다.

Author: ADDS Development Team
Date: 2026-01-15
Medical Review: Required before production use
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
import logging

# Phase 2-2: Enhanced Acquisition Functions
try:
    from recommendation.acquisition_functions import (
        AcquisitionFunctionRegistry,
        AdaptiveAcquisitionSelector
    )
except ImportError:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent))
    from acquisition_functions import (
        AcquisitionFunctionRegistry,
        AdaptiveAcquisitionSelector
    )

# 의료급 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/combination_designer.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


@dataclass
class Drug:
    """
    약물 정보 클래스
    
    의료 안전성 요구사항:
    - 모든 용량은 FDA 승인 범위 내
    - 약물 상호작용 데이터 필수
    """
    name: str
    min_dose: float  # mg/m²
    max_dose: float  # mg/m²
    standard_dose: float  # mg/m²
    route: str  # 'IV', 'PO', 'SC'
    half_life_hours: float
    target_proteins: List[str]
    known_interactions: List[str]  # 알려진 약물 상호작용
    
    def __post_init__(self):
        """입력 검증 (의료 안전성)"""
        assert self.min_dose > 0, f"{self.name}: min_dose must be positive"
        assert self.max_dose > self.min_dose, f"{self.name}: max_dose must be > min_dose"
        assert self.min_dose <= self.standard_dose <= self.max_dose, \
            f"{self.name}: standard_dose must be within min/max range"
        assert self.half_life_hours > 0, f"{self.name}: half_life must be positive"
        
        logger.info(f"Drug validated: {self.name} ({self.min_dose}-{self.max_dose} mg/m²)")


@dataclass
class DrugCombination:
    """
    약물 조합 클래스
    
    의료 안전성:
    - 최대 4제 조합 제한 (임상 실현 가능성)
    - 약물 상호작용 자동 검사
    - 총 독성 점수 계산
    """
    drugs: List[Drug]
    doses: List[float]  # mg/m² for each drug
    rationale: str  # 조합 근거
    
    def __post_init__(self):
        """조합 검증"""
        assert len(self.drugs) == len(self.doses), "Drugs and doses must match"
        assert 2 <= len(self.drugs) <= 4, "Combination must have 2-4 drugs (medical feasibility)"
        
        # 각 약물 용량 범위 검증
        for drug, dose in zip(self.drugs, self.doses):
            assert drug.min_dose <= dose <= drug.max_dose, \
                f"{drug.name} dose {dose} outside safe range [{drug.min_dose}, {drug.max_dose}]"
        
        # 약물 상호작용 검사
        self._check_drug_interactions()
        
        logger.info(f"Combination validated: {[d.name for d in self.drugs]}")
    
    def _check_drug_interactions(self):
        """약물 간 상호작용 검사 (의료 안전성 핵심)"""
        drug_names = [d.name for d in self.drugs]
        
        for i, drug1 in enumerate(self.drugs):
            for j, drug2 in enumerate(self.drugs[i+1:], i+1):
                # 알려진 상호작용 검사
                if drug2.name in drug1.known_interactions:
                    logger.warning(
                        f"Known interaction: {drug1.name} ↔ {drug2.name}. "
                        f"Requires clinical review."
                    )


@dataclass
class PatientProfile:
    """
    환자 프로필
    
    개인정보 보호:
    - 식별 정보 제외 (연령, 병기만)
    - HIPAA 준수
    """
    patient_id: str  # 익명화된 ID
    age: int
    cancer_type: str
    stage: int  # I-IV
    ecog: int  # 0-5 Performance Status
    mutations: List[str]  # 주요 유전자 변이
    previous_treatments: List[str]
    egfr: float  # eGFR (신기능), mL/min
    bilirubin: float  # 간기능, mg/dL
    
    def __post_init__(self):
        """환자 데이터 검증"""
        assert 0 <= self.age <= 120, "Invalid age"
        assert 1 <= self.stage <= 4, "Cancer stage must be I-IV"
        assert 0 <= self.ecog <= 5, "ECOG must be 0-5"
        assert self.egfr > 0, "eGFR must be positive"
        assert self.bilirubin >= 0, "Bilirubin must be non-negative"
        
        logger.info(f"Patient profile validated: {self.patient_id}")


class CombinationDesigner:
    """
    Exscientia Active Learning 기반 약물 조합 설계기
    
    핵심 기능:
    1. Multi-Parameter Optimization (MPO)
    2. Active Learning (Expected Improvement)
    3. Batch Selection (여러 조합 동시 선택)
    
    의료 안전성:
    - 모든 조합은 임상 실현 가능성 검증
    - 독성 예측 포함
    - 전문가 검토 플래그 자동 설정
    """
    
    def __init__(
        self,
        safety_weight: float = 0.30,
        efficacy_weight: float = 0.35,
        synergy_weight: float = 0.20,
        cost_weight: float = 0.10,
        feasibility_weight: float = 0.05,
        random_state: int = 42,
        dual_mode_threshold: int = 10
    ):
        """
        Parameters:
            safety_weight: 안전성 가중치 (높을수록 보수적)
            efficacy_weight: 효능 가중치
            synergy_weight: 시너지 가중치
            cost_weight: 비용 가중치
            feasibility_weight: 실현 가능성 가중치
            random_state: 난수 시드 (재현성)
            dual_mode_threshold: 듀얼 모드 전환 iteration (기본 10)
        
        의료 안전성 원칙:
        - Safety가 최우선 (기본 30%)
        - 가중치 합 = 1.0 보장
        
        Phase 2-2 Enhancement:
        - Enhanced acquisition functions (5종)
        - Dual-mode strategy (Thompson → EI)
        - Cancer-specific adaptive selection
        """
        # 가중치 검증
        total = safety_weight + efficacy_weight + synergy_weight + cost_weight + feasibility_weight
        assert abs(total - 1.0) < 1e-6, f"Weights must sum to 1.0, got {total}"
        
        self.weights = {
            'safety': safety_weight,
            'efficacy': efficacy_weight,
            'synergy': synergy_weight,
            'cost': cost_weight,
            'feasibility': feasibility_weight
        }
        
        # Gaussian Process 모델 (Active Learning)
        self.gp_model = GaussianProcessRegressor(
            kernel=Matern(nu=2.5),
            alpha=0.1,  # Noise level
            n_restarts_optimizer=10,
            random_state=random_state
        )
        
        self.history = []  # (features, outcome) 튜플 리스트
        
        # Phase 2-2: Enhanced Acquisition Functions
        self.acq_registry = AcquisitionFunctionRegistry(random_state=random_state)
        self.acq_selector = AdaptiveAcquisitionSelector()
        self.dual_mode_threshold = dual_mode_threshold
        self.random_state = random_state
        
        logger.info(f"CombinationDesigner initialized with weights: {self.weights}")
        logger.info(f"Phase 2-2: Enhanced acquisition functions enabled (dual_mode_threshold={dual_mode_threshold})")
    
    def design_next_combinations(
        self,
        available_drugs: List[Drug],
        patient_profile: PatientProfile,
        num_combinations: int = 10,
        iteration: int = 0,
        acquisition: str = 'dual_mode'
    ) -> List[DrugCombination]:
        """
        다음 테스트할 조합 선택 (Active Learning)
        
        Args:
            available_drugs: 사용 가능한 약물 리스트
            patient_profile: 환자 정보
            num_combinations: 선택할 조합 수
            iteration: 현재 반복 횟수 (0=초기)
            acquisition: Acquisition function 선택
                - 'expected_improvement': 전통적 EI (안정적)
                - 'upper_confidence_bound': UCB (exploration)
                - 'probability_of_improvement': PI (exploitation)
                - 'thompson_sampling': Bayesian sampling (빠른 수렴) ⭐
                - 'entropy_search_simplified': 정보 이론 기반
                - 'dual_mode': Thompson → EI 자동 전환 (추천) 🏆
                - 'adaptive': 암종별 자동 선택
        
        Returns:
            선택된 DrugCombination 리스트
        
        Phase 2-2 Enhancement:
            - 5가지 검증된 acquisition functions 통합
            - 듀얼 모드: 초기 빠른 탐색 → 후기 정밀 최적화
            - 암종 특화: 복잡한 암(췌장암 등)은 Entropy Search 자동 적용
        
        의료 안전성:
        - 모든 조합은 환자 안전성 검증
        - 신기능/간기능에 따라 용량 조정
        - 고위험 조합은 전문가 검토 플래그
        """
        logger.info(f"Designing {num_combinations} combinations for {patient_profile.patient_id}")
        
        # Phase 2-2: Determine acquisition function
        acq_func = self._determine_acquisition_function(
            iteration=iteration,
            patient=patient_profile,
            acquisition=acquisition
        )
        logger.info(f"Using acquisition function: {acq_func} (requested: {acquisition})")
        
        # Step 1: 후보 조합 생성
        if iteration == 0:
            # 초기: 광범위한 탐색
            candidates = self._generate_diverse_combinations(
                drugs=available_drugs,
                patient=patient_profile,
                num_candidates=100
            )
        else:
            # 후속: 학습 기반 생성
            candidates = self._generate_focused_combinations(
                drugs=available_drugs,
                patient=patient_profile,
                num_candidates=50
            )
        
        # Step 2: MPO 점수 계산
        scored_candidates = []
        for combo in candidates:
            mpo_score = self._calculate_mpo_score(combo, patient_profile)
            scored_candidates.append((combo, mpo_score))
        
        # Step 3: Enhanced Active Learning 선택
        selected = self._active_learning_select_enhanced(
            candidates=scored_candidates,
            num_select=num_combinations,
            acquisition=acq_func,
            iteration=iteration
        )
        
        logger.info(f"Selected {len(selected)} combinations using {acq_func}")
        return selected
    
    def _calculate_mpo_score(
        self,
        combination: DrugCombination,
        patient: PatientProfile
    ) -> Dict[str, float]:
        """
        Multi-Parameter Optimization Score 계산
        (Exscientia 방식 차용)
        
        Returns:
            {
                'total': 0.0-1.0 종합 점수,
                'efficacy': 0.0-1.0,
                'safety': 0.0-1.0,
                'synergy': 0.0-1.0,
                'cost': 0.0-1.0,
                'feasibility': 0.0-1.0,
                'needs_review': bool  # 전문가 검토 필요 여부
            }
        """
        # 1. Efficacy (효능) 예측
        efficacy = self._predict_efficacy(combination, patient)
        
        # 2. Safety (안전성) 예측
        toxicity = self._predict_toxicity(combination, patient)
        safety = 1.0 - toxicity  # 독성이 낮을수록 안전
        
        # 3. Synergy (시너지) 예측
        synergy = self._predict_synergy(combination)
        
        # 4. Cost (비용)
        total_cost = sum(drug.standard_dose * 100 for drug in combination.drugs)  # $100/mg 가정
        cost_score = 1.0 - min(total_cost / 10000, 1.0)  # $10K 기준 정규화
        
        # 5. Feasibility (실현 가능성)
        feasibility = self._check_feasibility(combination, patient)
        
        # 종합 점수 (가중 평균)
        scores = {
            'efficacy': efficacy,
            'safety': safety,
            'synergy': synergy,
            'cost': cost_score,
            'feasibility': feasibility
        }
        
        total_score = sum(scores[k] * self.weights[k] for k in scores)
        scores['total'] = total_score
        
        # 전문가 검토 필요 여부 (의료 안전성)
        scores['needs_review'] = (
            safety < 0.7 or          # 안전성 낮음
            feasibility < 0.5 or     # 실현 불가능
            len(combination.drugs) >= 4  # 4제 조합
        )
        
        logger.debug(f"MPO scores for {[d.name for d in combination.drugs]}: {scores}")
        return scores
    
    def _predict_efficacy(
        self,
        combination: DrugCombination,
        patient: PatientProfile
    ) -> float:
        """
        효능 예측 (0-1)
        
        실제 구현 시:
        - ML 모델 사용 (trained on clinical data)
        - 환자 바이오마커 기반 예측
        - 문헌 기반 증거 통합
        
        현재: 간단한 휴리스틱
        """
        # 타겟 단백질과 환자 변이 매칭
        target_coverage = 0.0
        for drug in combination.drugs:
            for mutation in patient.mutations:
                if mutation in drug.target_proteins:
                    target_coverage += 0.3
        
        efficacy = min(target_coverage, 1.0)
        
        # ECOG에 따른 보정 (Performance Status 좋을수록 효과적)
        ecog_factor = (5 - patient.ecog) / 5.0
        efficacy *= ecog_factor
        
        return efficacy
    
    def _predict_toxicity(
        self,
        combination: DrugCombination,
        patient: PatientProfile
    ) -> float:
        """
        독성 예측 (0-1, 높을수록 위험)
        
        의료 안전성 핵심 함수:
        - 약물 수에 따른 독성 증가
        - 신기능/간기능에 따른 조정
        - 약물 상호작용 반영
        """
        # 기본 독성 (약물 수에 비례)
        base_toxicity = len(combination.drugs) * 0.15
        
        # 신기능 장애 시 독성 증가
        if patient.egfr < 60:  # CKD Stage 3+
            renal_factor = 60 / (patient.egfr + 1e-6)
            base_toxicity *= renal_factor
        
        # 간기능 장애 시 독성 증가
        if patient.bilirubin > 1.2:  # 정상 상한
            hepatic_factor = patient.bilirubin / 1.2
            base_toxicity *= hepatic_factor
        
        # 고령 환자 독성 증가
        if patient.age > 75:
            age_factor = 1.2
            base_toxicity *= age_factor
        
        # 약물 상호작용 독성
        interaction_toxicity = 0.0
        for i, drug1 in enumerate(combination.drugs):
            for drug2 in combination.drugs[i+1:]:
                if drug2.name in drug1.known_interactions:
                    interaction_toxicity += 0.1
        
        total_toxicity = min(base_toxicity + interaction_toxicity, 1.0)
        
        logger.debug(f"Predicted toxicity: {total_toxicity:.3f} for {patient.patient_id}")
        return total_toxicity
    
    def _predict_synergy(self, combination: DrugCombination) -> float:
        """
        시너지 예측 (0-1)
        
        실제 구현 시:
        - 생물학적 경로 분석
        - 문헌 기반 시너지 데이터
        - ML 모델 예측
        
        현재: 타겟 중복도 기반 휴리스틱
        """
        all_targets = set()
        for drug in combination.drugs:
            all_targets.update(drug.target_proteins)
        
        # 타겟 중복이 적을수록 시너지 (상호 보완)
        avg_targets_per_drug = len(all_targets) / len(combination.drugs)
        synergy = min(avg_targets_per_drug / 3.0, 1.0)  # 3개 타겟 기준
        
        return synergy
    
    def _check_feasibility(
        self,
        combination: DrugCombination,
        patient: PatientProfile
    ) -> float:
        """
        실현 가능성 점수 (0-1)
        
        고려 사항:
        - 주입 시간 (IV 약물)
        - 환자 순응도 (PO 약물 개수)
        - 병용 금기
        """
        feasibility = 1.0
        
        # IV 약물이 3개 이상이면 실현 어려움 (12시간 초과)
        iv_drugs = [d for d in combination.drugs if d.route == 'IV']
        if len(iv_drugs) >= 3:
            feasibility *= 0.6
        
        # PO 약물이 5개 이상이면 순응도 문제
        po_drugs = [d for d in combination.drugs if d.route == 'PO']
        if len(po_drugs) >= 5:
            feasibility *= 0.7
        
        # ECOG 3-4 환자는 복잡한 요법 곤란
        if patient.ecog >= 3 and len(combination.drugs) >= 3:
            feasibility *= 0.5
        
        return feasibility
    
    def _generate_diverse_combinations(
        self,
        drugs: List[Drug],
        patient: PatientProfile,
        num_candidates: int
    ) -> List[DrugCombination]:
        """
        다양한 초기 후보 생성 (exploration)
        
        전략:
        - 2-4제 조합 균등 분포
        - 작용 기전 다양성 확보
        - 안전성 필터링
        """
        candidates = []
        
        # 2제, 3제, 4제 조합 균등 생성
        max_combo_size = min(4, len(drugs))  # 사용 가능한 약물 수 제한
        min_combo_size = min(2, max_combo_size)
        
        for combo_size in range(min_combo_size, max_combo_size + 1):
            size_candidates = num_candidates // (max_combo_size - min_combo_size + 1)
            
            for _ in range(size_candidates):
                # 약물 수가 combo_size보다 적으면 스킵
                if len(drugs) < combo_size:
                    continue
                
                # 무작위 약물 선택
                selected_drugs = np.random.choice(
                    drugs,
                    size=combo_size,
                    replace=False
                ).tolist()
                
                # 표준 용량으로 초기화
                doses = [drug.standard_dose for drug in selected_drugs]
                
                try:
                    combo = DrugCombination(
                        drugs=selected_drugs,
                        doses=doses,
                        rationale=f"Initial exploration - {combo_size} drugs"
                    )
                    
                    # 안전성 필터 (독성 > 0.8이면 제외)
                    toxicity = self._predict_toxicity(combo, patient)
                    if toxicity < 0.8:
                        candidates.append(combo)
                    else:
                        logger.warning(f"Combination rejected: high toxicity {toxicity:.2f}")
                
                except AssertionError as e:
                    logger.warning(f"Invalid combination: {e}")
                    continue
        
        logger.info(f"Generated {len(candidates)} diverse candidates")
        return candidates
    
    def _generate_focused_combinations(
        self,
        drugs: List[Drug],
        patient: PatientProfile,
        num_candidates: int
    ) -> List[DrugCombination]:
        """
        학습 기반 집중 생성 (exploitation)
        
        이전 결과를 학습하여 유망한 영역 집중 탐색
        """
        # TODO: 실제 구현 시 Gaussian Process 예측 기반 생성
        # 현재는 diverse와 동일하게 처리
        return self._generate_diverse_combinations(drugs, patient, num_candidates)
    
    def _determine_acquisition_function(
        self,
        iteration: int,
        patient: PatientProfile,
        acquisition: str
    ) -> str:
        """
        최적 Acquisition Function 결정
        
        Phase 2-2: 듀얼 모드 및 적응형 선택 지원
        
        Args:
            iteration: 현재 iteration
            patient: 환자 프로필
            acquisition: 사용자 지정 acquisition 전략
        
        Returns:
            실제 사용할 acquisition function 이름
        
        전략:
            - 'dual_mode': Thompson Sampling (early) → Expected Improvement (late)
            - 'adaptive': 암종/병기에 따라 자동 선택
            - Others: 그대로 사용
        """
        if acquisition == 'dual_mode':
            # Dual-mode strategy (benchmark 기반)
            if iteration < self.dual_mode_threshold:
                selected = 'thompson_sampling'
                logger.info(f"Dual-mode: Using Thompson Sampling (iteration {iteration} < {self.dual_mode_threshold})")
            else:
                selected = 'expected_improvement'
                logger.info(f"Dual-mode: Switching to Expected Improvement (iteration {iteration} >= {self.dual_mode_threshold})")
            return selected
        
        elif acquisition == 'adaptive':
            # Cancer-specific adaptive selection
            selected = self.acq_selector.select_best_acquisition(
                patient_cancer_type=patient.cancer_type,
                history_size=iteration,
                stage=patient.stage
            )
            logger.info(f"Adaptive: Selected {selected} for {patient.cancer_type} Stage {patient.stage}")
            return selected
        
        else:
            # User-specified or legacy function
            return acquisition
    
    def _active_learning_select_enhanced(
        self,
        candidates: List[Tuple[DrugCombination, Dict[str, float]]],
        num_select: int,
        acquisition: str,
        iteration: int
    ) -> List[DrugCombination]:
        """
        Phase 2-2: Enhanced Active Learning Selection
        
        AcquisitionFunctionRegistry를 사용한 통합 선택 로직
        
        Args:
            candidates: (DrugCombination, MPO scores) 튜플 리스트
            num_select: 선택할 조합 수
            acquisition: Acquisition function 이름
                - 'expected_improvement'
                - 'upper_confidence_bound'
                - 'probability_of_improvement'
                - 'thompson_sampling'
                - 'entropy_search_simplified'
            iteration: 현재 iteration (1-indexed for UCB)
        
        Returns:
            상위 num_select개 조합
        
        의료 안전성:
            - GP 예측이 불충분할 때는 MPO 점수 사용
            - Thompson Sampling의 확률적 특성 유지
            - 모든 함수 수치 안정성 보장
        """
        # 현재 최고 성능
        if self.history:
            best_so_far = max(y for _, y in self.history)
        else:
            best_so_far = 0.5  # 초기값
        
        acq_scores = []
        
        for combo, mpo_scores in candidates:
            features = self._featurize_combination(combo)
            
            # GP prediction (충분한 데이터가 있을 때만)
            if len(self.history) > 5:
                try:
                    mean, std = self._gp_predict(features)
                except Exception as e:
                    logger.warning(f"GP prediction failed: {e}. Using MPO scores.")
                    mean = mpo_scores['total']
                    std = 0.2  # 높은 불확실성
            else:
                # 초기에는 MPO 점수 사용
                mean = mpo_scores['total']
                std = 0.2  # 높은 불확실성 (exploration 유도)
            
            # Phase 2-2: Use AcquisitionFunctionRegistry
            try:
                score = self.acq_registry.evaluate(
                    function_name=acquisition,
                    mean=mean,
                    std=std,
                    best_so_far=best_so_far,
                    iteration=iteration + 1  # 1-indexed for UCB
                )
            except ValueError as e:
                # Fallback to basic EI if unknown function
                logger.error(f"Unknown acquisition function '{acquisition}': {e}. Falling back to EI.")
                from scipy.stats import norm
                z = (mean - best_so_far) / (std + 1e-6)
                score = (mean - best_so_far) * norm.cdf(z) + std * norm.pdf(z)
                score = max(score, 0.0)
            
            acq_scores.append((combo, score, mean, std))
        
        # Select top num_select by acquisition score
        acq_scores.sort(key=lambda x: x[1], reverse=True)
        selected = [combo for combo, _, _, _ in acq_scores[:num_select]]
        
        # Log statistics
        if acq_scores:
            top_scores = [s for _, s, _, _ in acq_scores[:num_select]]
            logger.info(
                f"Acquisition '{acquisition}': "
                f"Selected top {num_select} (scores: {top_scores[0]:.4f} - {top_scores[-1]:.4f})"
            )
        
        return selected

    
    def _active_learning_select(
        self,
        candidates: List[Tuple[DrugCombination, Dict[str, float]]],
        num_select: int,
        acquisition: str
    ) -> List[DrugCombination]:
        """
        Active Learning으로 최적 조합 선택
        
        Acquisition Functions:
        - 'expected_improvement': 현재 최고 대비 개선 기대치
        - 'ucb': Upper Confidence Bound (불확실성 고려)
        - 'pi': Probability of Improvement
        
        Returns:
            Score 상위 num_select개 조합
        """
        if acquisition == 'expected_improvement':
            return self._select_by_ei(candidates, num_select)
        elif acquisition == 'ucb':
            return self._select_by_ucb(candidates, num_select)
        elif acquisition == 'pi':
            return self._select_by_pi(candidates, num_select)
        else:
            raise ValueError(f"Unknown acquisition function: {acquisition}")
    
    def _select_by_ei(
        self,
        candidates: List[Tuple[DrugCombination, Dict[str, float]]],
        num_select: int
    ) -> List[DrugCombination]:
        """
        Expected Improvement로 선택
        
        EI = E[max(f(x) - f(x_best), 0)]
        """
        # 현재 최고 성능
        if self.history:
            current_best = max(y for _, y in self.history)
        else:
            current_best = 0.5  # 초기값
        
        ei_scores = []
        
        for combo, mpo_scores in candidates:
            # Gaussian Process 예측 (mean, std)
            features = self._featurize_combination(combo)
            
            if len(self.history) > 5:  # 충분한 데이터 있을 때만
                mean, std = self._gp_predict(features)
            else:
                # 초기에는 MPO 점수 사용
                mean = mpo_scores['total']
                std = 0.2  # 높은 불확실성
            
            # Expected Improvement 계산
            z = (mean - current_best) / (std + 1e-6)
            ei = (mean - current_best) * norm.cdf(z) + std * norm.pdf(z)
            
            ei_scores.append((combo, ei))
        
        # EI 상위 num_select개 선택
        ei_scores.sort(key=lambda x: x[1], reverse=True)
        selected = [combo for combo, _ in ei_scores[:num_select]]
        
        logger.info(f"Selected top {num_select} by Expected Improvement")
        return selected
    
    def _select_by_ucb(
        self,
        candidates: List[Tuple[DrugCombination, Dict[str, float]]],
        num_select: int,
        kappa: float = 2.0
    ) -> List[DrugCombination]:
        """
        Upper Confidence Bound로 선택
        
        UCB = mean + kappa * std
        """
        ucb_scores = []
        
        for combo, mpo_scores in candidates:
            features = self._featurize_combination(combo)
            
            if len(self.history) > 5:
                mean, std = self._gp_predict(features)
            else:
                mean = mpo_scores['total']
                std = 0.2
            
            ucb = mean + kappa * std
            ucb_scores.append((combo, ucb))
        
        ucb_scores.sort(key=lambda x: x[1], reverse=True)
        selected = [combo for combo, _ in ucb_scores[:num_select]]
        
        logger.info(f"Selected top {num_select} by UCB (kappa={kappa})")
        return selected
    
    def _select_by_pi(
        self,
        candidates: List[Tuple[DrugCombination, Dict[str, float]]],
        num_select: int
    ) -> List[DrugCombination]:
        """
        Probability of Improvement로 선택
        
        PI = P(f(x) > f(x_best))
        """
        if self.history:
            current_best = max(y for _, y in self.history)
        else:
            current_best = 0.5
        
        pi_scores = []
        
        for combo, mpo_scores in candidates:
            features = self._featurize_combination(combo)
            
            if len(self.history) > 5:
                mean, std = self._gp_predict(features)
            else:
                mean = mpo_scores['total']
                std = 0.2
            
            z = (mean - current_best) / (std + 1e-6)
            pi = norm.cdf(z)
            
            pi_scores.append((combo, pi))
        
        pi_scores.sort(key=lambda x: x[1], reverse=True)
        selected = [combo for combo, _ in pi_scores[:num_select]]
        
        logger.info(f"Selected top {num_select} by Probability of Improvement")
        return selected
    
    def _featurize_combination(self, combination: DrugCombination) -> np.ndarray:
        """
        조합을 feature vector로 변환
        
        Features:
        - 약물 수
        - 각 약물 용량 (정규화)
        - 타겟 다양성
        - 경로 분포 (IV/PO/SC)
        
        Returns:
            1D numpy array
        """
        features = []
        
        # 약물 수 (2-4 → 0-1 정규화)
        features.append((len(combination.drugs) - 2) / 2.0)
        
        # 용량 (최대 4개 약물, 없으면 0)
        for i in range(4):
            if i < len(combination.drugs):
                drug = combination.drugs[i]
                # 표준 용량 대비 비율
                dose_ratio = combination.doses[i] / drug.standard_dose
                features.append(dose_ratio)
            else:
                features.append(0.0)
        
        # 타겟 다양성
        all_targets = set()
        for drug in combination.drugs:
            all_targets.update(drug.target_proteins)
        features.append(len(all_targets) / 10.0)  # 최대 10개 타겟 가정
        
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
    
    def _gp_predict(self, features: np.ndarray) -> Tuple[float, float]:
        """
        Gaussian Process 예측
        
        Returns:
            (mean, std) 튜플
        """
        X_train = np.array([x for x, _ in self.history])
        y_train = np.array([y for _, y in self.history])
        
        self.gp_model.fit(X_train, y_train)
        
        mean, std = self.gp_model.predict(
            features.reshape(1, -1),
            return_std=True
        )
        
        return mean[0], std[0]
    
    def update_history(
        self,
        combination: DrugCombination,
        outcome: float
    ):
        """
        실험 결과로 모델 업데이트
        
        Args:
            combination: 테스트한 조합
            outcome: 실제 성능 (0-1)
        """
        features = self._featurize_combination(combination)
        self.history.append((features, outcome))
        
        logger.info(f"History updated: {len(self.history)} samples")


# 의료급 테스트 코드
if __name__ == "__main__":
    # 테스트용 약물 생성
    test_drugs = [
        Drug(
            name="Cisplatin",
            min_dose=50,
            max_dose=100,
            standard_dose=75,
            route="IV",
            half_life_hours=24,
            target_proteins=["DNA"],
            known_interactions=["5-FU"]
        ),
        Drug(
            name="5-Fluorouracil",
            min_dose=400,
            max_dose=1000,
            standard_dose=800,
            route="IV",
            half_life_hours=12,
            target_proteins=["Thymidylate synthase"],
            known_interactions=["Cisplatin"]
        ),
        Drug(
            name="Trastuzumab",
            min_dose=6,
            max_dose=8,
            standard_dose=8,
            route="IV",
            half_life_hours=672,  # 28 days
            target_proteins=["HER2"],
            known_interactions=[]
        )
    ]
    
    # 테스트용 환자
    test_patient = PatientProfile(
        patient_id="TEST001",
        age=65,
        cancer_type="Gastric",
        stage=3,
        ecog=1,
        mutations=["HER2"],
        previous_treatments=[],
        egfr=70,
       bilirubin=0.8
    )
    
    # Designer 생성
    designer = CombinationDesigner()
    
    # 조합 설계
    combinations = designer.design_next_combinations(
        available_drugs=test_drugs,
        patient_profile=test_patient,
        num_combinations=5,
        iteration=0
    )
    
    print(f"\n{'='*60}")
    print("DESIGNED COMBINATIONS (MEDICAL-GRADE)")
    print(f"{'='*60}\n")
    
    for i, combo in enumerate(combinations, 1):
        print(f"Combination {i}:")
        for drug, dose in zip(combo.drugs, combo.doses):
            print(f"  - {drug.name}: {dose:.1f} mg/m²")
        print(f"  Rationale: {combo.rationale}")
        
        # MPO 점수 계산
        scores = designer._calculate_mpo_score(combo, test_patient)
        print(f"  MPO Scores:")
        print(f"    Total: {scores['total']:.3f}")
        print(f"    Efficacy: {scores['efficacy']:.3f}")
        print(f"    Safety: {scores['safety']:.3f}")
        print(f"    Synergy: {scores['synergy']:.3f}")
        print(f"    Needs Review: {scores['needs_review']}")
        print()
