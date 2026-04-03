"""
ADDS Protein Module - ESMFold Integration
Phase 4-1: AlphaFold/ESMFold Integration

이 모듈은 단백질 구조 예측 및 약물-단백질 상호작용 분석을 담당합니다.
ESMFold (Meta AI)를 사용하여 AlphaFold와 유사한 성능을 로컬에서 제공합니다.

의료 안전성:
- 모든 단백질 데이터는 로컬 처리 (HIPAA 준수)
- 구조 예측 신뢰도 점수 필수 (pLDDT > 70 권장)
- 약물-표적 검증은 참고용 (임상 결정의 보조 도구)
"""

__version__ = "1.0.0"
__author__ = "ADDS Development Team"

from typing import Optional

# Optional imports - 설치되지 않은 경우 graceful degradation
try:
    from .esmfold_client import ESMFoldClient
    PROTEIN_MODULE_AVAILABLE = True
except ImportError:
    ESMFoldClient = None
    PROTEIN_MODULE_AVAILABLE = False

try:
    from .drug_protein_interaction import DrugProteinInteractionPredictor
except ImportError:
    DrugProteinInteractionPredictor = None

try:
    from .ppi_network import PPINetworkAnalyzer
except ImportError:
    PPINetworkAnalyzer = None

try:
    from .graph_synergy import GraphSynergyPredictor
except ImportError:
    GraphSynergyPredictor = None


__all__ = [
    'ESMFoldClient',
    'DrugProteinInteractionPredictor',
    'PPINetworkAnalyzer',
    'GraphSynergyPredictor',
    'PROTEIN_MODULE_AVAILABLE'
]
