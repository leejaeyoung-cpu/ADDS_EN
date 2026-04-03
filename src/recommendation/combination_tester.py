"""
ADDS-Exscientia Integration: Phase 1-2
Test Module - CombinationTester Class

이 모듈은 Exscientia의 자동화 Assay 방식을 환자 데이터 시뮬레이션에 적용합니다.
- 병렬 처리로 여러 조합 동시 테스트
- PK/PD 모델 기반 정밀 시뮬레이션
- 자동 품질 관리 (QC)
- 재시험 로직

Author: ADDS Development Team
Date: 2026-01-15
Medical Review: Required before production use
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
from enum import Enum

# combination_designer에서 import
try:
    from .combination_designer import DrugCombination, Drug, PatientProfile
except ImportError:
    # 테스트 실행 시
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent))
    from combination_designer import DrugCombination, Drug, PatientProfile

logger = logging.getLogger(__name__)


class TestStatus(Enum):
    """테스트 상태"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    RETEST_REQUIRED = "retest_required"


@dataclass
class PKProfile:
    """
    Pharmacokinetic (약동학) 프로필
    
    의료 정밀도:
    - 시간별 혈중 농도 추적
    - AUC (Area Under Curve) 계산
    - Cmax, Tmax 식별
    """
    drug_name: str
    time_points: np.ndarray  # 시간 (hours)
    concentrations: np.ndarray  # 농도 (μM)
    auc: float  # AUC (μM·h)
    cmax: float  # 최대 농도 (μM)
    tmax: float  # 최대 농도 도달 시간 (h)
    half_life: float  # 반감기 (h)
    
    def __post_init__(self):
        """검증"""
        assert len(self.time_points) == len(self.concentrations), \
            "Time points and concentrations must match"
        assert self.auc >= 0, "AUC must be non-negative"
        assert self.cmax >= 0, "Cmax must be non-negative"
        assert self.half_life > 0, "Half-life must be positive"


@dataclass
class PDProfile:
    """
    Pharmacodynamic (약력학) 프로필
    
    의료 정밀도:
    - 세포 사멸률 추적
    - 종양 크기 변화
    - 바이오마커 반응
    """
    time_points: np.ndarray  # 시간 (days)
    tumor_volumes: np.ndarray  # 종양 부피 (mm³)
    cell_viability: np.ndarray  # 세포 생존율 (0-1)
    biomarker_levels: Dict[str, np.ndarray]  # 바이오마커 수치
    
    def calculate_response(self) -> str:
        """
        RECIST 1.1 기준 반응 평가
        
        CR (Complete Response): 100% 감소
        PR (Partial Response): ≥30% 감소
        SD (Stable Disease): -30% ~ +20%
        PD (Progressive Disease): >20% 증가
        """
        baseline = self.tumor_volumes[0]
        nadir = self.tumor_volumes.min()  # 최소 크기
        final = self.tumor_volumes[-1]
        
        # 최선의 반응 (nadir 기준)
        change_from_baseline = (nadir - baseline) / baseline * 100
        
        # PD 판정은 nadir 대비 증가도 체크 (RECIST 1.1)
        if nadir < final:
            change_from_nadir = (final - nadir) / nadir * 100
            if change_from_nadir > 20:
                return "PD"  # Progressive Disease
        
        if change_from_baseline <= -100:
            return "CR"  # Complete Response
        elif change_from_baseline <= -30:
            return "PR"  # Partial Response
        elif change_from_baseline <= 20:
            return "SD"  # Stable Disease
        else:
            return "PD"  # Progressive Disease


@dataclass
class ToxicityProfile:
    """
    독성 프로필
    
    의료 안전성:
    - CTCAE v5.0 기준 등급화
    - 장기별 독성 추적
    - 용량 제한 독성 (DLT) 평가
    """
    hematologic: Dict[str, int]  # Neutropenia, Thrombocytopenia 등 (Grade 0-5)
    hepatic: Dict[str, int]  # AST, ALT, Bilirubin (Grade 0-5)
    renal: Dict[str, int]  # Creatinine, Proteinuria (Grade 0-5)
    cardiac: Dict[str, int]  # QTc prolongation, LVEF (Grade 0-5)
    gastrointestinal: Dict[str, int]  # Nausea, Diarrhea (Grade 0-5)
    
    def calculate_max_grade(self) -> int:
        """최대 독성 등급"""
        all_grades = []
        for toxicity_dict in [
            self.hematologic,
            self.hepatic,
            self.renal,
            self.cardiac,
            self.gastrointestinal
        ]:
            all_grades.extend(toxicity_dict.values())
        
        return max(all_grades) if all_grades else 0
    
    def is_dose_limiting_toxicity(self) -> bool:
        """
        용량 제한 독성 (DLT) 판정
        
        DLT 기준:
        - Grade 4 hematologic toxicity
        - Grade 3-4 non-hematologic toxicity
        """
        max_grade = self.calculate_max_grade()
        
        # Grade 4 혈액학적 독성
        if any(grade >= 4 for grade in self.hematologic.values()):
            return True
        
        # Grade 3-4 비혈액학적 독성
        non_hematologic = (
            list(self.hepatic.values()) +
            list(self.renal.values()) +
            list(self.cardiac.values()) +
            list(self.gastrointestinal.values())
        )
        if any(grade >= 3 for grade in non_hematologic):
            return True
        
        return False


@dataclass
class TestResult:
    """
    종합 테스트 결과
    
    의료 정밀도:
    - 완전한 추적 가능성
    - 재현 가능한 결과
    - 품질 관리 지표
    """
    combination: DrugCombination
    patient: PatientProfile
    pk_profiles: List[PKProfile]
    pd_profile: PDProfile
    toxicity_profile: ToxicityProfile
    
    # 종합 평가
    efficacy_score: float  # 0-1
    safety_score: float  # 0-1
    overall_score: float  # 0-1
    
    # 메타데이터
    status: TestStatus
    timestamp: datetime
    quality_passed: bool
    retest_count: int = 0
    
    def __post_init__(self):
        """검증"""
        assert 0 <= self.efficacy_score <= 1, "Efficacy score must be 0-1"
        assert 0 <= self.safety_score <= 1, "Safety score must be 0-1"
        assert 0 <= self.overall_score <= 1, "Overall score must be 0-1"


class PKPDSimulator:
    """
    PK/PD 시뮬레이터
    
    의료 정밀도:
    - 1-compartment PK 모델
    - Tumor growth inhibition (TGI) PD 모델
    - 개인차 반영 (CYP2D6 등)
    """
    
    def simulate_pk(
        self,
        drug: Drug,
        dose: float,
        patient: PatientProfile,
        duration_hours: float = 168  # 1주
    ) -> PKProfile:
        """
        1-compartment PK 모델 시뮬레이션
        
        C(t) = (Dose / Vd) * exp(-k * t)
        
        where:
        - Vd = Volume of distribution
        - k = Elimination rate constant = 0.693 / t1/2
        """
        # 시간 포인트 (0-168시간, 1시간 간격)
        time_points = np.arange(0, duration_hours, 1.0)
        
        # 파라미터
        t_half = drug.half_life_hours
        
        # 신기능에 따른 반감기 조정
        if patient.egfr < 60:  # CKD Stage 3+
            t_half *= (60 / patient.egfr)  # 신기능 저하 시 반감기 증가
        
        # 간기능에 따른 조정
        if patient.bilirubin > 1.2:
            t_half *= (patient.bilirubin / 1.2)
        
        k = 0.693 / t_half  # Elimination rate constant
        
        # Drug-specific Volume of distribution (literature values, L)
        DRUG_VD = {
            '5-Fluorouracil': 16.0,    # Diasio & Harris, Clin Pharmacokinet 1989
            '5-FU': 16.0,
            'Oxaliplatin': 440.0,       # Graham et al., Clin Cancer Res 2000
            'Irinotecan': 157.0,        # Mathijssen et al., Clin Cancer Res 2001
            'Capecitabine': 100.0,      # Reigner et al., Clin Pharmacokinet 2001
            'Cetuximab': 3.0,           # mAb ~plasma volume
            'Bevacizumab': 3.1,         # mAb ~plasma volume
            'Pembrolizumab': 6.0,       # Ahamadi et al., CPT Pharmacol Syst Pharmacol 2017
            'Nivolumab': 8.0,           # Bajaj et al., J Clin Pharmacol 2017
            'Regorafenib': 121.0,       # Strumberg et al., Clin Cancer Res 2012
        }
        vd = DRUG_VD.get(drug.name, 50.0)  # Default 50L if unknown
        
        # 초기 농도
        c0 = dose / vd
        
        # 농도 곡선
        concentrations = c0 * np.exp(-k * time_points)
        
        # AUC 계산 (사다리꼴 적분)
        auc = np.trapz(concentrations, time_points)
        
        # Cmax, Tmax
        cmax = concentrations[0]  # IV bolus 가정
        tmax = 0.0
        
        return PKProfile(
            drug_name=drug.name,
            time_points=time_points,
            concentrations=concentrations,
            auc=auc,
            cmax=cmax,
            tmax=tmax,
            half_life=t_half
        )
    
    def simulate_pd(
        self,
        pk_profiles: List[PKProfile],
        patient: PatientProfile,
        duration_days: int = 28
    ) -> PDProfile:
        """
        Tumor Growth Inhibition (TGI) 모델
        
        dV/dt = λV - δ(C)V
        
        where:
        - V = Tumor volume
        - λ = Growth rate
        - δ(C) = Drug-induced death rate (concentration-dependent)
        """
        time_points = np.arange(0, duration_days, 1.0)
        
        # 초기 종양 부피 (병기에 따라)
        if patient.stage == 1:
            v0 = 100  # mm³
        elif patient.stage == 2:
            v0 = 500
        elif patient.stage == 3:
            v0 = 1000
        else:  # stage 4
            v0 = 2000
        
        # 성장률 (doubling time ~100 days for solid tumors)
        lambda_growth = np.log(2) / 100  # per day
        
        # 약물 효과: Emax model — δ = Emax × C / (EC50 + C)
        # Drug-specific EC50 values (µg/mL, literature estimates)
        DRUG_EC50 = {
            '5-Fluorouracil': 0.5,    # Longley et al., Nat Rev Cancer 2003
            '5-FU': 0.5,
            'Oxaliplatin': 1.2,       # Raymond et al., Mol Cancer Ther 2002
            'Irinotecan': 2.0,        # Rivory et al., Clin Pharmacol Ther 1996
            'Capecitabine': 0.8,      # Reigner et al., 2001
            'Cetuximab': 50.0,        # Yang et al., Cancer Res 2001
            'Bevacizumab': 30.0,      # Ferrara et al., Nat Rev Drug Discov 2004
            'Pembrolizumab': 10.0,    # Freshwater et al., CPT 2017
            'Regorafenib': 3.5,       # Strumberg et al., 2012
        }
        emax = 0.02  # Maximum kill rate per day
        
        # Compute average daily drug concentration for each drug
        total_daily_kill = 0.0
        for pk in pk_profiles:
            # Mean steady-state concentration estimate
            mean_conc = pk.auc / 24.0 if pk.auc > 0 else 0
            ec50 = DRUG_EC50.get(pk.drug_name, 5.0)
            drug_kill = emax * mean_conc / (ec50 + mean_conc)
            total_daily_kill += drug_kill
        
        # 종양 부피 시뮬레이션 (Euler method)
        volumes = np.zeros(len(time_points))
        volumes[0] = v0
        
        dt = 1.0  # time step (days)
        for i in range(1, len(time_points)):
            dv_dt = lambda_growth * volumes[i-1] - total_daily_kill * volumes[i-1]
            volumes[i] = volumes[i-1] + dv_dt * dt
            
            # 음수 방지
            if volumes[i] < 0:
                volumes[i] = 0
        
        # 세포 생존율
        cell_viability = volumes / v0
        cell_viability = np.clip(cell_viability, 0, 1)
        
        # 바이오마커 (예: Ki-67, CA 19-9)
        biomarkers = {
            'Ki-67': 100 * (1 - cell_viability),  # 증식 감소
            'CA19-9': 1000 * cell_viability  # 종양 마커
        }
        
        return PDProfile(
            time_points=time_points,
            tumor_volumes=volumes,
            cell_viability=cell_viability,
            biomarker_levels=biomarkers
        )
    
    def simulate_toxicity(
        self,
        pk_profiles: List[PKProfile],
        combination: DrugCombination,
        patient: PatientProfile
    ) -> ToxicityProfile:
        """
        독성 프로필 시뮬레이션
        
        의료 안전성:
        - CTCAE v5.0 기준
        - 약물 수에 따른 독성 증가
        - 환자 취약성 반영
        """
        # 기본 독성 (AUC에 비례)
        total_auc = sum(pk.auc for pk in pk_profiles)
        base_toxicity = min(total_auc / 1000, 1.0)  # 정규화
        
        # 혈액학적 독성
        hematologic = {
            'Neutropenia': int(base_toxicity * 3),  # Grade 0-3
            'Thrombocytopenia': int(base_toxicity * 2),
            'Anemia': int(base_toxicity * 2)
        }
        
        # 간 독성
        hepatic_factor = 1.0
        if patient.bilirubin > 1.2:
            hepatic_factor = patient.bilirubin / 1.2
        
        hepatic = {
            'AST_elevation': int(base_toxicity * hepatic_factor * 2),
            'ALT_elevation': int(base_toxicity * hepatic_factor * 2),
            'Bilirubin_elevation': int((patient.bilirubin - 0.3) / 0.3)
        }
        
        # 신 독성
        renal_factor = 1.0
        if patient.egfr < 60:
            renal_factor = 60 / patient.egfr
        
        renal = {
            'Creatinine_elevation': int(base_toxicity * renal_factor * 2),
            'Proteinuria': int(base_toxicity * renal_factor)
        }
        
        # 심장 독성 (특정 약물, 예: Trastuzumab)
        cardiac_drugs = [d.name for d in combination.drugs if 'Trastuzumab' in d.name]
        cardiac_toxicity = len(cardiac_drugs) * 1  # Grade 0-1
        
        cardiac = {
            'LVEF_decline': cardiac_toxicity,
            'QTc_prolongation': int(base_toxicity)
        }
        
        # 위장관 독성
        gastrointestinal = {
            'Nausea': int(base_toxicity * 3),
            'Diarrhea': int(base_toxicity * 2),
            'Mucositis': int(base_toxicity * 2)
        }
        
        # Grade cap at 5
        for toxicity_dict in [hematologic, hepatic, renal, cardiac, gastrointestinal]:
            for key in toxicity_dict:
                toxicity_dict[key] = min(toxicity_dict[key], 5)
        
        return ToxicityProfile(
            hematologic=hematologic,
            hepatic=hepatic,
            renal=renal,
            cardiac=cardiac,
            gastrointestinal=gastrointestinal
        )


class CombinationTester:
    """
    Exscientia 자동화 Assay 방식을 환자 데이터에 적용
    
    핵심 기능:
    1. 병렬 처리 (여러 조합 동시 테스트)
    2. PK/PD 시뮬레이션
    3. 자동 품질 관리 (QC)
    4. 재시험 로직
    
    의료 안전성:
    - 모든 결과 검증
    - DLT 자동 탐지
    - 재현성 보장
    """
    
    def __init__(self, max_workers: int = 8):
        """
        Args:
            max_workers: 병렬 처리 워커 수
        """
        self.max_workers = max_workers
        self.simulator = PKPDSimulator()
        
        logger.info(f"CombinationTester initialized with {max_workers} workers")
    
    def test_combinations(
        self,
        combinations: List[DrugCombination],
        patient: PatientProfile,
        patient_cohort: Optional[List[PatientProfile]] = None
    ) -> List[TestResult]:
        """
        여러 조합을 병렬로 테스트
        
        Args:
            combinations: 테스트할 조합 리스트
            patient: 대표 환자 (단일 환자 테스트)
            patient_cohort: 환자 코호트 (다중 환자 테스트)
        
        Returns:
            TestResult 리스트
        
        의료 안전성:
        - 각 조합 독립 검증
        - 품질 관리 통과한 결과만 반환
        - 실패 시 자동 재시험
        """
        logger.info(f"Testing {len(combinations)} combinations")
        
        results = []
        
        # 병렬 처리 (Exscientia 로봇 자동화 개념)
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {}
            
            for combo in combinations:
                future = executor.submit(
                    self._run_single_test,
                    combo,
                    patient
                )
                futures[future] = combo
            
            # 결과 수집
            for future in as_completed(futures):
                combo = futures[future]
                try:
                    result = future.result()
                    
                    # 품질 관리 (QC)
                    if self._passes_quality_check(result):
                        results.append(result)
                        logger.info(
                            f"✓ {[d.name for d in combo.drugs]}: "
                            f"Efficacy={result.efficacy_score:.3f}, "
                            f"Safety={result.safety_score:.3f}"
                        )
                    else:
                        # 재시험
                        logger.warning(f"Quality check failed for {[d.name for d in combo.drugs]}, retesting...")
                        retest_result = self._run_single_test(combo, patient)
                        retest_result.retest_count += 1
                        
                        if self._passes_quality_check(retest_result):
                            results.append(retest_result)
                        else:
                            logger.error(f"Retest also failed for {[d.name for d in combo.drugs]}")
                
                except Exception as e:
                    logger.error(f"Test failed for {[d.name for d in combo.drugs]}: {e}")
        
        logger.info(f"Completed testing: {len(results)}/{len(combinations)} passed QC")
        return results
    
    def _run_single_test(
        self,
        combination: DrugCombination,
        patient: PatientProfile
    ) -> TestResult:
        """
        단일 조합 테스트 (핵심 로직)
        
        단계:
        1. PK 시뮬레이션 (각 약물)
        2. PD 시뮬레이션 (종합 효과)
        3. 독성 프로필
        4. 종합 점수 계산
        """
        # Step 1: PK 시뮬레이션
        pk_profiles = []
        for drug, dose in zip(combination.drugs, combination.doses):
            pk = self.simulator.simulate_pk(drug, dose, patient)
            pk_profiles.append(pk)
        
        # Step 2: PD 시뮬레이션
        pd_profile = self.simulator.simulate_pd(pk_profiles, patient)
        
        # Step 3: 독성 프로필
        toxicity_profile = self.simulator.simulate_toxicity(
            pk_profiles,
            combination,
            patient
        )
        
        # Step 4: 종합 점수
        efficacy_score = self._calculate_efficacy_score(pd_profile)
        safety_score = self._calculate_safety_score(toxicity_profile)
        overall_score = 0.6 * efficacy_score + 0.4 * safety_score  # 가중 평균
        
        return TestResult(
            combination=combination,
            patient=patient,
            pk_profiles=pk_profiles,
            pd_profile=pd_profile,
            toxicity_profile=toxicity_profile,
            efficacy_score=efficacy_score,
            safety_score=safety_score,
            overall_score=overall_score,
            status=TestStatus.COMPLETED,
            timestamp=datetime.now(),
            quality_passed=True  # QC는 별도 검증
        )
    
    def _calculate_efficacy_score(self, pd_profile: PDProfile) -> float:
        """
        효능 점수 계산 (0-1)
        
        기준:
        - Tumor volume reduction
        - RECIST response
        - Cell viability reduction
        """
        # 종양 크기 변화 (%)
        baseline = pd_profile.tumor_volumes[0]
        final = pd_profile.tumor_volumes[-1]
        reduction = (baseline - final) / baseline
        
        # RECIST 반응
        response = pd_profile.calculate_response()
        response_scores = {
            'CR': 1.0,
            'PR': 0.7,
            'SD': 0.3,
            'PD': 0.0
        }
        response_score = response_scores[response]
        
        # 종합 (가중 평균)
        efficacy = 0.6 * min(reduction, 1.0) + 0.4 * response_score
        
        return np.clip(efficacy, 0, 1)
    
    def _calculate_safety_score(self, toxicity_profile: ToxicityProfile) -> float:
        """
        안전성 점수 계산 (0-1)
        
        기준:
        - 최대 독성 등급
        - DLT 여부
        - 장기별 독성 분포
        """
        max_grade = toxicity_profile.calculate_max_grade()
        is_dlt = toxicity_profile.is_dose_limiting_toxicity()
        
        # 최대 등급에 따른 점수
        # Grade 0: 1.0, Grade 1: 0.8, Grade 2: 0.6, Grade 3: 0.3, Grade 4+: 0.0
        grade_scores = {0: 1.0, 1: 0.8, 2: 0.6, 3: 0.3, 4: 0.1, 5: 0.0}
        grade_score = grade_scores.get(max_grade, 0.0)
        
        # DLT 페널티
        if is_dlt:
            grade_score *= 0.5
        
        return grade_score
    
    def _passes_quality_check(self, result: TestResult) -> bool:
        """
        품질 관리 (QC) 검사
        
        기준:
        1. PK 프로필 정상 범위
        2. PD 반응 생물학적 타당성
        3. 독성 등급 합리성
        4. 수치적 안정성 (NaN, Inf 없음)
        """
        # 1. PK 검증
        for pk in result.pk_profiles:
            if pk.auc <= 0 or np.isnan(pk.auc) or np.isinf(pk.auc):
                logger.warning(f"Invalid PK profile for {pk.drug_name}: AUC={pk.auc}")
                return False
            
            if pk.cmax <= 0:
                logger.warning(f"Invalid Cmax for {pk.drug_name}: {pk.cmax}")
                return False
        
        # 2. PD 검증
        if np.any(np.isnan(result.pd_profile.tumor_volumes)):
            logger.warning("NaN in tumor volumes")
            return False
        
        if np.any(result.pd_profile.tumor_volumes < 0):
            logger.warning("Negative tumor volume")
            return False
        
        # 3. 독성 검증
        max_grade = result.toxicity_profile.calculate_max_grade()
        if max_grade < 0 or max_grade > 5:
            logger.warning(f"Invalid toxicity grade: {max_grade}")
            return False
        
        # 4. 점수 검증
        if not (0 <= result.efficacy_score <= 1):
            logger.warning(f"Invalid efficacy score: {result.efficacy_score}")
            return False
        
        if not (0 <= result.safety_score <= 1):
            logger.warning(f"Invalid safety score: {result.safety_score}")
            return False
        
        return True


# 의료급 테스트 코드
if __name__ == "__main__":
    from combination_designer import Drug, DrugCombination, PatientProfile
    
    # 테스트용 약물
    cisplatin = Drug(
        name="Cisplatin",
        min_dose=50,
        max_dose=100,
        standard_dose=75,
        route="IV",
        half_life_hours=24,
        target_proteins=["DNA"],
        known_interactions=["5-FU"]
    )
    
    fluorouracil = Drug(
        name="5-Fluorouracil",
        min_dose=400,
        max_dose=1000,
        standard_dose=800,
        route="IV",
        half_life_hours=12,
        target_proteins=["Thymidylate synthase"],
        known_interactions=["Cisplatin"]
    )
    
    # 테스트용 조합
    test_combo = DrugCombination(
        drugs=[cisplatin, fluorouracil],
        doses=[75, 800],
        rationale="Standard FOLFOX regimen"
    )
    
    # 테스트용 환자
    test_patient = PatientProfile(
        patient_id="TEST002",
        age=58,
        cancer_type="Colorectal",
        stage=3,
        ecog=1,
        mutations=["KRAS"],
        previous_treatments=[],
        egfr=85,
        bilirubin=0.9
    )
    
    # Tester 생성
    tester = CombinationTester(max_workers=4)
    
    # 테스트 실행
    results = tester.test_combinations(
        combinations=[test_combo],
        patient=test_patient
    )
    
    print(f"\n{'='*60}")
    print("COMBINATION TEST RESULTS (MEDICAL-GRADE)")
    print(f"{'='*60}\n")
    
    for result in results:
        print(f"Combination: {[d.name for d in result.combination.drugs]}")
        print(f"Patient: {result.patient.patient_id}")
        print(f"\nPK Profiles:")
        for pk in result.pk_profiles:
            print(f"  {pk.drug_name}:")
            print(f"    AUC: {pk.auc:.2f} μM·h")
            print(f"    Cmax: {pk.cmax:.2f} μM")
            print(f"    Half-life: {pk.half_life:.1f} h")
        
        print(f"\nPD Profile:")
        response = result.pd_profile.calculate_response()
        print(f"  RECIST Response: {response}")
        print(f"  Tumor reduction: {(1 - result.pd_profile.tumor_volumes[-1]/result.pd_profile.tumor_volumes[0])*100:.1f}%")
        
        print(f"\nToxicity Profile:")
        print(f"  Max Grade: {result.toxicity_profile.calculate_max_grade()}")
        print(f"  DLT: {result.toxicity_profile.is_dose_limiting_toxicity()}")
        
        print(f"\nScores:")
        print(f"  Efficacy: {result.efficacy_score:.3f}")
        print(f"  Safety: {result.safety_score:.3f}")
        print(f"  Overall: {result.overall_score:.3f}")
        print(f"\nQuality Check: {'PASSED' if result.quality_passed else 'FAILED'}")
        print(f"Status: {result.status.value}")
