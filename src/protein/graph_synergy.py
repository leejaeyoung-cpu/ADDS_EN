"""
Graph Neural Network Based Drug Synergy Predictor
PPI 네트워크를 활용한 약물 시너지 예측

GraphSynergy 스타일의 GNN 모델:
- 노드: 단백질 (features: 구조 embedding, 발현량 등)
- 엣지: PPI 상호작용 (features: 신뢰도, 유형)
- 예측: Drug A + Drug B → Synergy score

Reference:
- GraphSynergy: https://academic.oup.com/bib/article/22/5/bbab015/6145781
- DeepSynergy: https://academic.oup.com/bioinformatics/article/34/9/1538/4747884
"""

import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import numpy as np

logger = logging.getLogger(__name__)

# Optional imports
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch not available. GNN synergy predictor will use simplified model.")

try:
    from .ppi_network import PPINetwork, PPINetworkAnalyzer
except:
    from ppi_network import PPINetwork, PPINetworkAnalyzer


@dataclass
class SynergyPrediction:
    """시너지 예측 결과
    
    Attributes:
        drug_a: 약물 A
        drug_b: 약물 B
        synergy_score: 시너지 점수 (0-1, 높을수록 시너지)
        synergy_type: 'synergistic', 'additive', 'antagonistic'
        confidence: 예측 신뢰도 (0-1)
        mechanism: 시너지 메커니즘 설명
    """
    drug_a: str
    drug_b: str
    synergy_score: float
    synergy_type: str
    confidence: float
    mechanism: str = ""
    
    @property
    def is_synergistic(self) -> bool:
        """시너지 효과 있음"""
        return self.synergy_type == "synergistic"
    
    @property
    def is_antagonistic(self) -> bool:
        """길항 효과 (같이 쓰면 안 됨)"""
        return self.synergy_type == "antagonistic"


# 알려진 시너지 조합 (검증용)
KNOWN_SYNERGIES = {
    # 폐암
    ("Cisplatin", "Pemetrexed"): ("synergistic", 0.75, "DNA damage + nucleotide synthesis inhibition"),
    ("Carboplatin", "Paclitaxel"): ("synergistic", 0.80, "DNA damage + microtubule stabilization"),
    
    # 유방암
    ("Doxorubicin", "Cyclophosphamide"): ("synergistic", 0.70, "DNA intercalation + alkylation"),
    ("Trastuzumab", "Pertuzumab"): ("synergistic", 0.85, "Dual HER2 blockade"),
    
    # 대장암
    ("5-FU", "Leucovorin"): ("synergistic", 0.78, "Thymidylate synthase enhancement"),
    ("5-FU", "Oxaliplatin"): ("synergistic", 0.72, "Different mechanisms"),
    
    # 길항 효과 (negative control)
    ("Cisplatin", "Cisplatin"): ("additive", 0.50, "Same drug"),
}


class SimplifiedGNNSynergyPredictor:
    """간소화된 GNN 시너지 예측기 (PyTorch 없이)
    
    실제 GNN 없이 규칙 기반 + PPI 네트워크 거리 기반 예측
    PyTorch 사용 가능하면 실제 GNN 모델로 대체 가능
    """
    
    def __init__(self):
        self.ppi_analyzer = PPINetworkAnalyzer(use_mock=True)
        self.network = self.ppi_analyzer.build_network()
        
        # 약물-표적 매핑 (간단한 버전)
        self.drug_targets = {
            "Imatinib": ["BCR-ABL", "c-KIT"],
            "Gefitinib": ["EGFR"],
            "Erlotinib": ["EGFR"],
            "Cisplatin": ["P53"],
            "Pemetrexed": ["TYMS"],
            "Carboplatin": ["P53"],
            "Paclitaxel": ["TUBB"],  # Tubulin
            "Doxorubicin": ["TOP2A"],  # Topoisomerase
            "Cyclophosphamide": ["DNA"],
            "Trastuzumab": ["HER2"],
            "Pertuzumab": ["HER2"],
            "5-FU": ["TYMS"],
            "Leucovorin": ["TYMS"],
            "Oxaliplatin": ["P53"],
        }
    
    def predict_synergy(
        self,
        drug_a: str,
        drug_b: str,
        cancer_type: Optional[str] = None
    ) -> SynergyPrediction:
        """약물 조합 시너지 예측
        
        Args:
            drug_a: 약물 A 이름
            drug_b: 약물 B 이름
            cancer_type: 암종 (optional, 향후 암종별 모델에 사용)
            
        Returns:
            SynergyPrediction 객체
        """
        # 알려진 조합인지 확인
        known_key = (drug_a, drug_b)
        reversed_key = (drug_b, drug_a)
        
        if known_key in KNOWN_SYNERGIES:
            syn_type, score, mechanism = KNOWN_SYNERGIES[known_key]
            return SynergyPrediction(
                drug_a=drug_a,
                drug_b=drug_b,
                synergy_score=score,
                synergy_type=syn_type,
                confidence=0.95,  # 알려진 조합은 높은 신뢰도
                mechanism=mechanism
            )
        elif reversed_key in KNOWN_SYNERGIES:
            syn_type, score, mechanism = KNOWN_SYNERGIES[reversed_key]
            return SynergyPrediction(
                drug_a=drug_a,
                drug_b=drug_b,
                synergy_score=score,
                synergy_type=syn_type,
                confidence=0.95,
                mechanism=mechanism
            )
        
        # 같은 약물
        if drug_a == drug_b:
            return SynergyPrediction(
                drug_a=drug_a,
                drug_b=drug_b,
                synergy_score=0.5,
                synergy_type="additive",
                confidence=1.0,
                mechanism="Same drug - no synergy expected"
            )
        
        # PPI 네트워크 기반 예측
        return self._predict_from_network(drug_a, drug_b)
    
    def _predict_from_network(
        self,
        drug_a: str,
        drug_b: str
    ) -> SynergyPrediction:
        """PPI 네트워크 거리 기반 시너지 예측"""
        
        # 약물의 표적 가져오기
        targets_a = self.drug_targets.get(drug_a, [])
        targets_b = self.drug_targets.get(drug_b, [])
        
        if not targets_a or not targets_b:
            # 표적 정보 없음
            return SynergyPrediction(
                drug_a=drug_a,
                drug_b=drug_b,
                synergy_score=0.5,
                synergy_type="additive",
                confidence=0.2,
                mechanism="Unknown targets - cannot predict"
            )
        
        # 표적 간 최소 거리 계산
        min_distance = float('inf')
        for target_a in targets_a:
            for target_b in targets_b:
                distance = self.ppi_analyzer.compute_target_distance(
                    target_a, target_b, self.network
                )
                if distance != -1 and distance < min_distance:
                    min_distance = distance
        
        # 거리 기반 시너지 점수
        if min_distance == float('inf') or min_distance == -1:
            # 연결 안 됨 - 독립적 경로
            synergy_score = 0.4
            synergy_type = "additive"
            mechanism = "Independent pathways"
        elif min_distance == 1:
            # 직접 연결 - 협력 또는 경쟁
            synergy_score = 0.6
            synergy_type = "additive"
            mechanism = "Direct target interaction"
        elif 2 <= min_distance <= 3:
            # 가까운 경로 - 높은 시너지
            synergy_score = 0.75
            synergy_type = "synergistic"
            mechanism = f"Same pathway ({min_distance} steps apart)"
        else:
            # 먼 경로
            synergy_score = 0.45
            synergy_type = "additive"
            mechanism = f"Distant pathways ({min_distance} steps)"
        
        return SynergyPrediction(
            drug_a=drug_a,
            drug_b=drug_b,
            synergy_score=synergy_score,
            synergy_type=synergy_type,
            confidence=0.6,  # 네트워크 기반은 중간 신뢰도
            mechanism=mechanism
        )
    
    def batch_predict(
        self,
        drug_pairs: List[Tuple[str, str]]
    ) -> List[SynergyPrediction]:
        """여러 약물 조합 예측"""
        return [self.predict_synergy(a, b) for a, b in drug_pairs]
    
    def find_best_combinations(
        self,
        drug_list: List[str],
        n_top: int = 5
    ) -> List[SynergyPrediction]:
        """가장 시너지적인 조합 찾기
        
        Args:
            drug_list: 후보 약물 리스트
            n_top: 상위 N개 조합
            
        Returns:
            시너지 점수 순으로 정렬된 조합
        """
        predictions = []
        
        # 모든 쌍 생성
        for i, drug_a in enumerate(drug_list):
            for drug_b in drug_list[i+1:]:
                pred = self.predict_synergy(drug_a, drug_b)
                predictions.append(pred)
        
        # 시너지 점수 순 정렬
        predictions.sort(key=lambda x: x.synergy_score, reverse=True)
        
        return predictions[:n_top]


class GraphSynergyPredictor:
    """XGBoost 기반 시너지 예측기
    
    모델 우선순위:
    1. v3: Drug FP + CCLE expression + Bio (Pearson r=0.71)
    2. v2: Drug FP only (Pearson r=0.69)
    3. v1: label encoding (fallback)
    4. 규칙 기반: SimplifiedGNNSynergyPredictor
    """
    
    def __init__(self):
        from pathlib import Path
        import pickle
        import json
        
        self.model = None
        self.drug_fps = {}     # drug_name -> fp_array (1024)
        self.cell_expr = {}    # cell_line -> expr_array (256)
        self.cell_bio = {}     # cell_line -> bio_array (15)
        self.n_fp_bits = 1024
        self.n_expr = 256
        self.n_bio = 15
        self.version = None
        self.fallback = SimplifiedGNNSynergyPredictor()
        
        model_dir = Path(BASE_DIR / "models/synergy")
        
        try:
            import xgboost as xgb_lib
        except ImportError:
            logger.warning("xgboost not installed; using fallback predictor")
            return
        
        try:
            # --- v4: Best model (P2, LCLO=0.610) with expanded drug FPs ---
            v4_model = model_dir / "xgboost_synergy_realexpr.pkl"
            v4_fps = model_dir / "drug_fingerprints_morgan_full.pkl"
            v4_expr = model_dir / "cell_line_expression.pkl"
            
            if v4_model.exists():
                with open(v4_model, 'rb') as f:
                    self.model = pickle.load(f)
                
                # Load expanded Morgan FPs (4,246 drugs)
                if v4_fps.exists():
                    with open(v4_fps, 'rb') as f:
                        self.drug_fps = pickle.load(f)
                elif (model_dir / "drug_fingerprints.pkl").exists():
                    with open(model_dir / "drug_fingerprints.pkl", 'rb') as f:
                        self.drug_fps = pickle.load(f)
                
                # Load cell expression (256-dim per cell)
                if v4_expr.exists():
                    with open(v4_expr, 'rb') as f:
                        self.cell_expr = pickle.load(f)
                
                self.version = 'v4'
                self.n_expr = 256
                logger.info(f"Loaded synergy model v4 (P2-best, LCLO=0.610): "
                           f"{len(self.drug_fps)} drugs, {len(self.cell_expr)} cell expr")
                return

            # Load shared drug fingerprints (used by v2/v3)
            fps_file = model_dir / "drug_fingerprints.pkl"
            if fps_file.exists():
                with open(fps_file, 'rb') as f:
                    self.drug_fps = pickle.load(f)
            
            # Try v3 (Drug FP + CCLE + Bio)
            v3_model = model_dir / "xgb_synergy_v3.json"
            
            if v3_model.exists() and fps_file.exists():
                self.model = xgb_lib.XGBRegressor()
                self.model.load_model(str(v3_model))
                
                # Load cell expression (256 genes)
                if v4_expr.exists():
                    with open(v4_expr, 'rb') as f:
                        self.cell_expr = pickle.load(f)
                
                # Load bio features
                bio_file = Path(BASE_DIR / "data/ml_training/cell_line_features.csv")
                if bio_file.exists():
                    import pandas as _pd
                    bio_df = _pd.read_csv(bio_file)
                    for _, row in bio_df.iterrows():
                        cl = row['cell_line']
                        self.cell_bio[cl] = row.drop('cell_line').values.astype(np.float32)
                    self.n_bio = len(next(iter(self.cell_bio.values())))
                
                self.version = 'v3'
                logger.info(f"Loaded synergy model v3: {len(self.drug_fps)} drugs, "
                           f"{len(self.cell_expr)} cell expr, {len(self.cell_bio)} cell bio")
                return
            
            # Try v2 (Drug FP only)
            v2_model = model_dir / "xgb_synergy_v2.json"
            if v2_model.exists() and fps_file.exists():
                self.model = xgb_lib.XGBRegressor()
                self.model.load_model(str(v2_model))
                self.version = 'v2'
                logger.info(f"Loaded synergy model v2: {len(self.drug_fps)} drugs, Morgan FP")
                return
            
            # Try v1 (label encoding)
            v1_model = model_dir / "xgb_synergy_v1.json"
            v1_enc = model_dir / "encoders.pkl"
            if v1_model.exists() and v1_enc.exists():
                self.model = xgb_lib.XGBRegressor()
                self.model.load_model(str(v1_model))
                with open(v1_enc, 'rb') as f:
                    self._v1_encoders = pickle.load(f)
                v1_meta = model_dir / "model_metadata.json"
                self._v1_features = []
                if v1_meta.exists():
                    with open(v1_meta) as f:
                        self._v1_features = json.load(f).get('feature_names', [])
                self.version = 'v1'
                logger.info("Loaded synergy model v1 (label encoding)")
                return
            
            logger.info("No trained synergy model found; using rule-based fallback")
        except Exception as e:
            logger.warning(f"Failed to load synergy model: {e}; using fallback")
            self.model = None
    
    def _get_drug_fp(self, drug_name: str) -> np.ndarray:
        """Get Morgan FP for a drug, case-insensitive."""
        if drug_name in self.drug_fps:
            return self.drug_fps[drug_name]
        upper_map = {k.upper(): v for k, v in self.drug_fps.items()}
        if drug_name.upper() in upper_map:
            return upper_map[drug_name.upper()]
        return np.zeros(self.n_fp_bits, dtype=np.float32)
    
    def _get_cell_expr(self, cell_line: str) -> np.ndarray:
        """Get CCLE expression for a cell line, case-insensitive."""
        if cell_line in self.cell_expr:
            return self.cell_expr[cell_line]
        upper_map = {k.upper(): v for k, v in self.cell_expr.items()}
        if cell_line and cell_line.upper() in upper_map:
            return upper_map[cell_line.upper()]
        return np.zeros(self.n_expr, dtype=np.float32)
    
    def _get_cell_bio(self, cell_line: str) -> np.ndarray:
        """Get biological features for a cell line, case-insensitive."""
        if cell_line in self.cell_bio:
            return self.cell_bio[cell_line]
        upper_map = {k.upper(): v for k, v in self.cell_bio.items()}
        if cell_line and cell_line.upper() in upper_map:
            return upper_map[cell_line.upper()]
        return np.zeros(self.n_bio, dtype=np.float32)
    
    def predict_synergy(
        self,
        drug_a: str,
        drug_b: str,
        cancer_type: Optional[str] = None
    ) -> SynergyPrediction:
        """시너지 예측: v4 (best) -> v3 (FP+expr+bio) -> v2 (FP) -> v1 -> 규칙 기반"""
        if self.model is None:
            return self.fallback.predict_synergy(drug_a, drug_b, cancer_type)
        
        try:
            if self.version == 'v4':
                # v4: FP_A(1024) + FP_B(1024) + expr(256) = 2304
                fp_a = self._get_drug_fp(drug_a)
                fp_b = self._get_drug_fp(drug_b)
                expr = self._get_cell_expr(cancer_type or '')
                features = np.concatenate([fp_a, fp_b, expr])
            elif self.version == 'v3':
                features = self._build_features_v3(drug_a, drug_b, cancer_type)
            elif self.version == 'v2':
                features = self._build_features_v2(drug_a, drug_b)
            else:
                features = self._build_features_v1(drug_a, drug_b)
            
            loewe_score = float(self.model.predict(np.array([features]))[0])
            
            # Loewe score -> synergy classification
            if loewe_score > 10:
                syn_type = "synergistic"
                score_01 = min(0.5 + loewe_score / 60.0, 0.95)
            elif loewe_score < -10:
                syn_type = "antagonistic"
                score_01 = max(0.5 + loewe_score / 60.0, 0.05)
            else:
                syn_type = "additive"
                score_01 = 0.5 + loewe_score / 60.0
            
            # Confidence based on drug/cell line coverage
            a_known = drug_a.upper() in {k.upper() for k in self.drug_fps}
            b_known = drug_b.upper() in {k.upper() for k in self.drug_fps}
            confidence = 0.85 if (a_known and b_known) else 0.55 if (a_known or b_known) else 0.30
            
            return SynergyPrediction(
                drug_a=drug_a,
                drug_b=drug_b,
                synergy_score=round(score_01, 3),
                synergy_type=syn_type,
                confidence=round(confidence, 2),
                mechanism=f"XGBoost {self.version} (Loewe={loewe_score:.2f})"
            )
        except Exception as e:
            logger.warning(f"XGBoost prediction failed: {e}; falling back")
            return self.fallback.predict_synergy(drug_a, drug_b, cancer_type)
    
    def _build_features_v3(self, drug_a: str, drug_b: str,
                           cell_line: Optional[str] = None) -> np.ndarray:
        """Build v3 feature vector: FP_A(1024) + FP_B(1024) + expr(256) + bio(15)."""
        fp_a = self._get_drug_fp(drug_a)
        fp_b = self._get_drug_fp(drug_b)
        expr = self._get_cell_expr(cell_line or '')
        bio = self._get_cell_bio(cell_line or '')
        return np.concatenate([fp_a, fp_b, expr, bio])
    
    def _build_features_v2(self, drug_a: str, drug_b: str) -> np.ndarray:
        """Build v2 feature vector: FP_A (1024) + FP_B (1024) + cell_line_enc."""
        fp_a = self._get_drug_fp(drug_a)
        fp_b = self._get_drug_fp(drug_b)
        return np.concatenate([fp_a, fp_b, [0.0]])
    
    def _build_features_v1(self, drug_a: str, drug_b: str) -> list:
        """Build v1 feature vector (label encoding)."""
        feature_map = {
            'drug_a_enc': 0, 'drug_b_enc': 0, 'cell_line_enc': 0,
            'ic50_a_log': 0.0, 'ic50_b_log': 0.0,
            'synergy_bliss': 0.0, 'synergy_loewe': 0.0, 'synergy_hsa': 0.0,
        }
        return [feature_map.get(f, 0.0) for f in getattr(self, '_v1_features', [])] \
            or list(feature_map.values())
    
    def batch_predict(
        self,
        drug_pairs: List[Tuple[str, str]]
    ) -> List[SynergyPrediction]:
        """배치 예측"""

import os as _os
from pathlib import Path as _Path
# ADDS_BASE_DIR environment variable overrides automatic detection
BASE_DIR = _Path(_os.environ.get("ADDS_BASE_DIR", str(_Path(__file__).resolve().parent.parent)))

        return [self.predict_synergy(a, b) for a, b in drug_pairs]


if __name__ == "__main__":
    # 테스트 실행
    logging.basicConfig(level=logging.INFO)
    
    predictor = SimplifiedGNNSynergyPredictor()
    
    print("=== Known Synergy Test ===")
    # 알려진 시너지 (폐암)
    pred1 = predictor.predict_synergy("Cisplatin", "Pemetrexed")
    print(f"Cisplatin + Pemetrexed:")
    print(f"  Type: {pred1.synergy_type}")
    print(f"  Score: {pred1.synergy_score}")
    print(f"  Mechanism: {pred1.mechanism}")
    print(f"  Confidence: {pred1.confidence}")
    
    print("\n=== Network-based Prediction ===")
    # 네트워크 기반 예측
    pred2 = predictor.predict_synergy("Gefitinib", "Imatinib")
    print(f"Gefitinib + Imatinib:")
    print(f"  Type: {pred2.synergy_type}")
    print(f"  Score: {pred2.synergy_score}")
    print(f"  Mechanism: {pred2.mechanism}")
    
    print("\n=== Best Combinations ===")
    # 최고 조합 찾기
    available_drugs = [
        "Cisplatin", "Pemetrexed", "Gefitinib", 
        "Carboplatin", "Paclitaxel"
    ]
    top_combos = predictor.find_best_combinations(available_drugs, n_top=3)
    for i, pred in enumerate(top_combos, 1):
        print(f"{i}. {pred.drug_a} + {pred.drug_b}: {pred.synergy_score:.2f} ({pred.synergy_type})")
