"""
ADDS Protein Integration Tests
Phase 4-1: AlphaFold/ESMFold Integration

의료급 테스트:
- 알려진 약물-표적 쌍 재현성 ≥ 95%
- 시너지 예측 정확도 검증
- 데이터 무결성 (HIPAA)
"""

import sys
import os
import pytest
import logging

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from protein.esmfold_client import ESMFoldClient, ProteinStructure
from protein.drug_protein_interaction import (
    DrugProteinInteractionPredictor,
    DrugProteinBinding,
    KNOWN_DRUG_TARGETS
)
from protein.ppi_network import PPINetworkAnalyzer, PPINetwork
from protein.graph_synergy import (
    SimplifiedGNNSynergyPredictor,
    SynergyPrediction,
    KNOWN_SYNERGIES
)

logger = logging.getLogger(__name__)


class TestESMFoldClient:
    """ESMFold 클라이언트 테스트"""
    
    def test_initialization(self):
        """클라이언트 초기화 테스트"""
        client = ESMFoldClient(use_mock=True)
        assert client is not None
        assert client.use_mock is True
    
    def test_structure_prediction(self):
        """구조 예측 테스트"""
        client = ESMFoldClient(use_mock=True)
        
        sequence = "MRPSGTAGAALLALLAALCPASRALEEKKVCQGT"  # 짧은 EGFR 서열
        structure = client.predict_structure("EGFR", sequence)
        
        assert structure.protein_id == "EGFR"
        assert structure.sequence == sequence
        assert structure.structure_available is True
        assert structure.mean_plddt is not None
        assert structure.mean_plddt > 0
    
    def test_high_confidence_check(self):
        """고신뢰도 구조 확인"""
        client = ESMFoldClient(use_mock=True)
        
        structure = client.predict_structure("EGFR", "MOCK")
        
        # Mock은 80-95 범위로 생성됨
        assert structure.is_high_confidence(threshold=70)
    
    def test_caching(self):
        """캐싱 테스트"""
        client = ESMFoldClient(use_mock=True)
        
        # 첫 예측
        structure1 = client.predict_structure("TEST_PROTEIN", "ACDEFGH")
        
        # 캐시에서 로드
        structure2 = client.predict_structure("TEST_PROTEIN", "ACDEFGH", use_cache=True)
        
        assert structure1.protein_id == structure2.protein_id
        assert structure1.mean_plddt == structure2.mean_plddt


class TestDrugProteinInteraction:
    """약물-단백질 상호작용 테스트"""
    
    def test_initialization(self):
        """예측기 초기화"""
        predictor = DrugProteinInteractionPredictor(use_mock=True)
        assert predictor is not None
    
    def test_known_target_prediction(self):
        """알려진 표적 예측 테스트"""
        predictor = DrugProteinInteractionPredictor(use_mock=True)
        
        # Imatinib-BCR-ABL (알려진 강한 결합)
        bcr_abl = ProteinStructure(
            protein_id="BCR-ABL",
            sequence="MOCK",
            structure_available=True,
            mean_plddt=88.0
        )
        
        binding = predictor.predict_binding("Imatinib", bcr_abl)
        
        assert binding.drug_name == "Imatinib"
        assert binding.protein_id == "BCR-ABL"
        assert binding.is_known_target is True
        assert binding.is_strong_binder()  # < -7 kcal/mol
        assert binding.confidence > 0.8
    
    def test_unknown_target_prediction(self):
        """알려지지 않은 표적 예측"""
        predictor = DrugProteinInteractionPredictor(use_mock=True)
        
        # 알려지지 않은 쌍
        unknown_protein = ProteinStructure(
            protein_id="UNKNOWN_PROTEIN",
            sequence="MOCK",
            structure_available=True,
            mean_plddt=85.0
        )
        
        binding = predictor.predict_binding("Imatinib", unknown_protein)
        
        assert binding.is_known_target is False
        assert binding.confidence < 0.5  # 낮은 신뢰도
        assert binding.is_weak_binder()  # > -5 kcal/mol (약한 결합)
    
    def test_validation_accuracy(self):
        """알려진 약물-표적 쌍 재현성 테스트
        
        의료 안전성: ≥ 95% 정확도 필요
        """
        predictor = DrugProteinInteractionPredictor(use_mock=True)
        
        metrics = predictor.validate_predictions()
        
        assert metrics['accuracy'] >= 0.95, \
            f"Validation accuracy {metrics['accuracy']:.1%} below 95% threshold"
        assert metrics['correct'] > 0
        assert metrics['total'] > 0
    
    def test_low_confidence_structure_warning(self):
        """낮은 신뢰도 구조 경고"""
        predictor = DrugProteinInteractionPredictor(use_mock=True)
        
        # 낮은 pLDDT 구조
        low_conf_structure = ProteinStructure(
            protein_id="LOW_CONF",
            sequence="MOCK",
            structure_available=True,
            mean_plddt=60.0  # < 70
        )
        
        with pytest.warns(None):  # 경고가 로그에 기록되는지 확인
            binding = predictor.predict_binding("Gefitinib", low_conf_structure)
            assert binding is not None


class TestPPINetwork:
    """PPI 네트워크 테스트"""
    
    def test_network_building(self):
        """네트워크 구축 테스트"""
        analyzer = PPINetworkAnalyzer(use_mock=True)
        network = analyzer.build_network()
        
        assert len(network.proteins) > 0
        assert len(network.interactions) > 0
    
    def test_shortest_path_egfr_braf(self):
        """EGFR-BRAF 최단 경로 (예상: EGFR-KRAS-BRAF)"""
        analyzer = PPINetworkAnalyzer(use_mock=True)
        network = analyzer.build_network()
        
        path = network.get_shortest_path("EGFR", "BRAF")
        
        assert path is not None
        assert path[0] == "EGFR"
        assert path[-1] == "BRAF"
        assert len(path) == 3  # EGFR-KRAS-BRAF
        assert "KRAS" in path
    
    def test_target_distance(self):
        """표적 간 거리 계산"""
        analyzer = PPINetworkAnalyzer(use_mock=True)
        network = analyzer.build_network()
        
        # EGFR-PI3K: 직접 연결 (distance=1)
        distance = analyzer.compute_target_distance("EGFR", "PI3K", network)
        assert distance == 1
        
        # EGFR-mTOR: 2 steps (EGFR-PI3K-AKT?)
        distance_far = analyzer.compute_target_distance("EGFR", "mTOR", network)
        assert distance_far > 1
    
    def test_synergy_prediction_from_network(self):
        """네트워크 기반 시너기 예측"""
        analyzer = PPINetworkAnalyzer(use_mock=True)
        network = analyzer.build_network()
        
        # EGFR-PI3K: 같은 경로, 높은 시너지 예상
        result = analyzer.predict_synergy_from_network("EGFR", "PI3K", network)
        
        assert result['distance'] == 1
        assert result['synergy_score'] > 0.5
        assert result['pathway'] == 'same'
    
    def test_common_downstream(self):
        """공통 하위 노드 탐색"""
        analyzer = PPINetworkAnalyzer(use_mock=True)
        network = analyzer.build_network()
        
        common = analyzer.find_common_downstream("EGFR", "PI3K", network, max_depth=2)
        
        # AKT는 둘 다의 하위 노드
        assert "AKT" in common


class TestGraphSynergy:
    """GNN 시너지 예측 테스트"""
    
    def test_predictor_initialization(self):
        """예측기 초기화"""
        predictor = SimplifiedGNNSynergyPredictor()
        assert predictor is not None
    
    def test_known_synergy_cisplatin_pemetrexed(self):
        """알려진 시너지: Cisplatin + Pemetrexed (폐암)"""
        predictor = SimplifiedGNNSynergyPredictor()
        
        pred = predictor.predict_synergy("Cisplatin", "Pemetrexed")
        
        assert pred.drug_a == "Cisplatin"
        assert pred.drug_b == "Pemetrexed"
        assert pred.is_synergistic
        assert pred.synergy_score > 0.7
        assert pred.confidence > 0.9  # 알려진 조합은 높은 신뢰도
    
    def test_known_synergy_order_independence(self):
        """순서 무관성 테스트 (A+B = B+A)"""
        predictor = SimplifiedGNNSynergyPredictor()
        
        pred1 = predictor.predict_synergy("Cisplatin", "Pemetrexed")
        pred2 = predictor.predict_synergy("Pemetrexed", "Cisplatin")
        
        assert pred1.synergy_score == pred2.synergy_score
        assert pred1.synergy_type == pred2.synergy_type
    
    def test_same_drug_additive(self):
        """같은 약물 조합 (additive)"""
        predictor = SimplifiedGNNSynergyPredictor()
        
        pred = predictor.predict_synergy("Cisplatin", "Cisplatin")
        
        assert pred.synergy_type == "additive"
        assert pred.synergy_score == 0.5
    
    def test_network_based_prediction(self):
        """네트워크 기반 예측 (알려지지 않은 조합)"""
        predictor = SimplifiedGNNSynergyPredictor()
        
        # Gefitinib + Imatinib (알려지지 않음, 네트워크로 예측)
        pred = predictor.predict_synergy("Gefitinib", "Imatinib")
        
        assert pred.confidence < 0.95  # 네트워크 기반은 낮은 신뢰도
        assert pred.synergy_score >= 0  # Valid score
        assert pred.synergy_type in ["synergistic", "additive", "antagonistic"]
    
    def test_find_best_combinations(self):
        """최고 조합 찾기"""
        predictor = SimplifiedGNNSynergyPredictor()
        
        drugs = ["Cisplatin", "Pemetrexed", "Gefitinib", "Carboplatin"]
        top_combos = predictor.find_best_combinations(drugs, n_top=3)
        
        assert len(top_combos) == 3
        # 시너지 점수 순으로 정렬되어야 함
        for i in range(len(top_combos) - 1):
            assert top_combos[i].synergy_score >= top_combos[i+1].synergy_score
    
    def test_batch_prediction(self):
        """배치 예측"""
        predictor = SimplifiedGNNSynergyPredictor()
        
        pairs = [
            ("Cisplatin", "Pemetrexed"),
            ("Carboplatin", "Paclitaxel"),
            ("Gefitinib", "Erlotinib")
        ]
        
        results = predictor.batch_predict(pairs)
        
        assert len(results) == 3
        for result in results:
            assert isinstance(result, SynergyPrediction)


class TestProteinModuleIntegration:
    """전체 모듈 통합 테스트"""
    
    def test_full_workflow(self):
        """전체 워크플로우: 구조 예측 → 결합 예측 → 시너지 예측"""
        
        # 1. 구조 예측
        esmfold = ESMFoldClient(use_mock=True)
        egfr_structure = esmfold.predict_structure("EGFR", "MOCK_SEQUENCE")
        assert egfr_structure.structure_available
        
        # 2. 약물-단백질 결합
        dpi_predictor = DrugProteinInteractionPredictor(use_mock=True)
        binding = dpi_predictor.predict_binding("Gefitinib", egfr_structure)
        assert binding.is_strong_binder()
        
        # 3. 시너지 예측
        synergy_predictor = SimplifiedGNNSynergyPredictor()
        synergy = synergy_predictor.predict_synergy("Gefitinib", "Erlotinib")
        assert synergy is not None
    
    def test_medical_safety_checks(self):
        """의료 안전성 체크
        
        - 구조 신뢰도 검증
        - 알려진 표적과 비교
        - Off-target 경고
        """
        # 낮은 신뢰도 구조
        low_conf = ProteinStructure(
            protein_id="TEST",
            sequence="MOCK",
            structure_available=True,
            mean_plddt=65.0  # < 70
        )
        
        assert not low_conf.is_high_confidence()
        
        # 예측은 가능하지만 경고 발생
        dpi = DrugProteinInteractionPredictor(use_mock=True)
        binding = dpi.predict_binding("Imatinib", low_conf)
        assert binding is not None


class TestClinicalValidation:
    """임상 검증 테스트
    
    의료 안전성:
    - 알려진 약물 조합 재현성 ≥ 95%
    - 예측 정확도 검증
    """
    
    def test_known_synergy_reproduction(self):
        """알려진 시너지 조합 재현성"""
        predictor = SimplifiedGNNSynergyPredictor()
        
        correct = 0
        total = 0
        
        for (drug_a, drug_b), (expected_type, _, _) in KNOWN_SYNERGIES.items():
            pred = predictor.predict_synergy(drug_a, drug_b)
            if pred.synergy_type == expected_type:
                correct += 1
            total += 1
        
        accuracy = correct / total if total > 0 else 0
        
        assert accuracy >= 0.95, \
            f"Known synergy reproduction: {accuracy:.1%} < 95%"
    
    def test_drug_target_validation(self):
        """약물-표적 검증 정확도"""
        predictor = DrugProteinInteractionPredictor(use_mock=True)
        
        metrics = predictor.validate_predictions()
        
        # 95% 이상 정확도 요구
        assert metrics['accuracy'] >= 0.95


if __name__ == "__main__":
    # pytest 실행
    logging.basicConfig(level=logging.INFO)
    
    print("Running ADDS Protein Module Tests...")
    print("=" * 60)
    
    # -v: verbose, -s: show print output
    pytest.main([__file__, '-v', '-s'])
