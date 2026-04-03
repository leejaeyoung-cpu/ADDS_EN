"""
PPI (Protein-Protein Interaction) Network Analyzer
STRING DB 기반 단백질 상호작용 네트워크 분석

PPI 네트워크는 약물 시너지 예측의 핵심:
- 두 약물이 다른 경로를 타겟하면 시너지 가능성 높음
- 같은 복합체(complex)를 타겟하면 협력 효과

Data source:
- STRING DB v12.0 (https://string-db.org/)
- 9.6M proteins, 2B interactions
"""

import logging
from typing import Dict, List, Set, Tuple, Optional
from dataclasses import dataclass, field
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ProteinInteraction:
    """단백질 간 상호작용
    
    Attributes:
        protein_a: 단백질 A
        protein_b: 단백질 B
        confidence: 상호작용 신뢰도 (0-1, STRING combined score / 1000)
        interaction_type: 상호작용 유형 (binding, activation, inhibition 등)
    """
    protein_a: str
    protein_b: str
    confidence: float
    interaction_type: str = "binding"
    
    def is_high_confidence(self, threshold: float = 0.7) -> bool:
        """고신뢰도 상호작용인지 확인
        
        STRING DB 권장: confidence > 0.7 (700/1000)
        """
        return self.confidence > threshold


@dataclass
class PPINetwork:
    """PPI 네트워크
    
    Attributes:
        proteins: 단백질 노드 집합
        interactions: 상호작용 엣지 리스트
        adjacency: 인접 리스트 {protein: [neighbor1, neighbor2, ...]}
    """
    proteins: Set[str] = field(default_factory=set)
    interactions: List[ProteinInteraction] = field(default_factory=list)
    adjacency: Dict[str, List[str]] = field(default_factory=dict)
    
    def add_interaction(self, interaction: ProteinInteraction):
        """상호작용 추가"""
        self.proteins.add(interaction.protein_a)
        self.proteins.add(interaction.protein_b)
        self.interactions.append(interaction)
        
        # 인접 리스트 업데이트
        if interaction.protein_a not in self.adjacency:
            self.adjacency[interaction.protein_a] = []
        if interaction.protein_b not in self.adjacency:
            self.adjacency[interaction.protein_b] = []
        
        self.adjacency[interaction.protein_a].append(interaction.protein_b)
        self.adjacency[interaction.protein_b].append(interaction.protein_a)
    
    def get_neighbors(self, protein: str) -> List[str]:
        """특정 단백질의 이웃 노드 반환"""
        return self.adjacency.get(protein, [])
    
    def get_shortest_path(self, protein_a: str, protein_b: str) -> Optional[List[str]]:
        """두 단백질 간 최단 경로 (BFS)
        
        Returns:
            경로 리스트 [protein_a, intermediate1, ..., protein_b]
            연결되지 않으면 None
        """
        if protein_a == protein_b:
            return [protein_a]
        
        if protein_a not in self.adjacency or protein_b not in self.adjacency:
            return None
        
        # BFS
        queue = [(protein_a, [protein_a])]
        visited = {protein_a}
        
        while queue:
            current, path = queue.pop(0)
            
            for neighbor in self.get_neighbors(current):
                if neighbor == protein_b:
                    return path + [neighbor]
                
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, path + [neighbor]))
        
        return None
    
    def get_path_distance(self, protein_a: str, protein_b: str) -> int:
        """두 단백질 간 경로 거리
        
        Returns:
            경로 길이 (1 = 직접 연결, 2 = 1개 중간 노드 등)
            연결 안 되면 -1
        """
        path = self.get_shortest_path(protein_a, protein_b)
        return len(path) - 1 if path else -1


# Mock PPI 데이터 (암 관련 주요 경로)
# 실제로는 STRING DB API에서 가져와야 함
MOCK_PPI_DATA = [
    # EGFR 신호 전달 경로
    ("EGFR", "KRAS", 0.85, "activation"),
    ("KRAS", "BRAF", 0.90, "activation"),
    ("BRAF", "MEK1", 0.92, "activation"),
    ("MEK1", "ERK1", 0.88, "activation"),
    ("ERK1", "MYC", 0.75, "activation"),
    
    # PI3K-AKT 경로
    ("EGFR", "PI3K", 0.82, "activation"),
    ("PI3K", "AKT", 0.95, "activation"),
    ("AKT", "mTOR", 0.87, "activation"),
    ("mTOR", "S6K", 0.83, "activation"),
    
    # 세포사멸 경로
    ("AKT", "BAD", 0.78, "inhibition"),
    ("BAD", "BCL2", 0.80, "inhibition"),
    ("BCL2", "CASP3", 0.85, "inhibition"),
    ("CASP3", "PARP", 0.90, "activation"),
    
    # 약물 표적 간 연결
    ("BCR-ABL", "KRAS", 0.65, "activation"),
    ("EGFR", "HER2", 0.88, "binding"),  # HER2-EGFR 이합체
    
    # DNA 손상 반응
    ("P53", "MDM2", 0.92, "binding"),
    ("P53", "BAX", 0.85, "activation"),
    ("ATM", "P53", 0.88, "activation"),
]


class PPINetworkAnalyzer:
    """PPI 네트워크 분석기
    
    약물 시너지 예측을 위한 네트워크 분석:
    1. 두 약물의 표적 간 거리
    2. 표적이 속한 경로 (pathway)
    3. 공통 하위 노드 (downstream targets)
    
    Example:
        >>> analyzer = PPINetworkAnalyzer()
        >>> network = analyzer.build_network()
        >>> distance = analyzer.compute_target_distance("EGFR", "BRAF", network)
        >>> print(f"Distance: {distance} (2=EGFR-KRAS-BRAF)")
    """
    
    def __init__(self, use_mock: bool = True):
        """
        Args:
            use_mock: True이면 mock PPI 데이터 사용
                     False이면 STRING DB API 호출 (구현 예정)
        """
        self.use_mock = use_mock
        
        if not use_mock:
            logger.warning("STRING DB API not implemented. Using mock data.")
            self.use_mock = True
    
    def build_network(
        self,
        protein_list: Optional[List[str]] = None,
        confidence_threshold: float = 0.4
    ) -> PPINetwork:
        """PPI 네트워크 구축
        
        Args:
            protein_list: 관심 단백질 리스트 (None이면 전체)
            confidence_threshold: 최소 신뢰도 (STRING combined score / 1000)
            
        Returns:
            PPINetwork 객체
        """
        network = PPINetwork()
        
        if self.use_mock:
            # Mock 데이터 로드
            for protein_a, protein_b, confidence, int_type in MOCK_PPI_DATA:
                if confidence >= confidence_threshold:
                    # protein_list 필터링
                    if protein_list is not None:
                        if protein_a not in protein_list and protein_b not in protein_list:
                            continue
                    
                    interaction = ProteinInteraction(
                        protein_a=protein_a,
                        protein_b=protein_b,
                        confidence=confidence,
                        interaction_type=int_type
                    )
                    network.add_interaction(interaction)
        else:
            # TODO: STRING DB API 호출
            raise NotImplementedError("STRING DB API not implemented")
        
        logger.info(
            f"Built PPI network: {len(network.proteins)} proteins, "
            f"{len(network.interactions)} interactions"
        )
        
        return network
    
    def compute_target_distance(
        self,
        target1: str,
        target2: str,
        network: PPINetwork
    ) -> int:
        """두 약물 표적 간 네트워크 거리
        
        시너지 해석:
        - Distance = 1: 직접 상호작용 (협력 또는 경쟁 가능)
        - Distance = 2-3: 같은 경로, 시너지 가능성 높음
        - Distance > 5: 독립적 경로, 낮은 시너지
        
        Returns:
            경로 거리 (-1이면 연결 안 됨)
        """
        return network.get_path_distance(target1, target2)
    
    def predict_synergy_from_network(
        self,
        target1: str,
        target2: str,
        network: PPINetwork
    ) -> Dict:
        """PPI 네트워크 기반 시너지 예측
        
        Returns:
            {
                'distance': int,
                'synergy_score': float (0-1),
                'pathway': 'same'|'different'|'unknown',
                'mechanism': str
            }
        """
        distance = self.compute_target_distance(target1, target2, network)
        
        if distance == -1:
            # 연결 안 됨 - 독립적 경로
            return {
                'distance': -1,
                'synergy_score': 0.3,  # 낮은 시너지
                'pathway': 'different',
                'mechanism': 'Independent pathways, possible additive effect'
            }
        
        elif distance == 1:
            # 직접 상호작용
            return {
                'distance': 1,
                'synergy_score': 0.6,
                'pathway': 'same',
                'mechanism': 'Direct interaction, potential synergy or antagonism'
            }
        
        elif 2 <= distance <= 3:
            # 같은 경로 내 가까운 표적
            return {
                'distance': distance,
                'synergy_score': 0.75,  # 높은 시너지
                'pathway': 'same',
                'mechanism': f'Same pathway ({distance} steps apart), strong synergy potential'
            }
        
        else:
            # 먼 거리
            return {
                'distance': distance,
                'synergy_score': 0.4,
                'pathway': 'different',
                'mechanism': f'Distant targets ({distance} steps), moderate synergy'
            }
    
    def find_common_downstream(
        self,
        target1: str,
        target2: str,
        network: PPINetwork,
        max_depth: int = 3
    ) -> Set[str]:
        """두 표적의 공통 하위 노드 찾기
        
        공통 하위 노드가 많으면 협력 효과 가능성 높음
        
        Returns:
            공통 하위 노드 집합
        """
        def get_downstream(protein: str, depth: int) -> Set[str]:
            if depth == 0:
                return set()
            downstream = set(network.get_neighbors(protein))
            for neighbor in list(downstream):
                downstream.update(get_downstream(neighbor, depth - 1))
            return downstream
        
        downstream1 = get_downstream(target1, max_depth)
        downstream2 = get_downstream(target2, max_depth)
        
        common = downstream1.intersection(downstream2)
        
        if common:
            logger.info(
                f"{target1} and {target2} share {len(common)} downstream targets: "
                f"{list(common)[:5]}"  # 첫 5개만 표시
            )
        
        return common


if __name__ == "__main__":
    # 테스트 실행
    logging.basicConfig(level=logging.INFO)
    
    analyzer = PPINetworkAnalyzer(use_mock=True)
    network = analyzer.build_network()
    
    print(f"\n=== PPI Network ===")
    print(f"Proteins: {len(network.proteins)}")
    print(f"Interactions: {len(network.interactions)}")
    
    # EGFR-BRAF 거리 (예상: 2, EGFR-KRAS-BRAF)
    print(f"\n=== Target Distance Test ===")
    distance = analyzer.compute_target_distance("EGFR", "BRAF", network)
    path = network.get_shortest_path("EGFR", "BRAF")
    print(f"EGFR-BRAF distance: {distance}")
    print(f"Path: {' -> '.join(path) if path else 'Not connected'}")
    
    # 시너지 예측
    print(f"\n=== Synergy Prediction ===")
    synergy = analyzer.predict_synergy_from_network("EGFR", "PI3K", network)
    print(f"EGFR-PI3K synergy:")
    print(f"  Distance: {synergy['distance']}")
    print(f"  Synergy score: {synergy['synergy_score']}")
    print(f"  Mechanism: {synergy['mechanism']}")
    
    # 공통 하위 노드
    print(f"\n=== Common Downstream ===")
    common = analyzer.find_common_downstream("EGFR", "PI3K", network, max_depth=2)
    print(f"Common downstream targets: {common}")
