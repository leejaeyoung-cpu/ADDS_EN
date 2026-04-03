"""
Multi-Modal Treatment Response Pipeline
========================================
유전자 발현 + CT 종양 크기 변화 + Cellpose 암세포 스트레스/사멸

3가지 독립 modality를 통합하여 치료 반응을 예측하는 모듈화된 파이프라인.

Architecture:
    ┌──────────────┐   ┌──────────────┐   ┌──────────────┐
    │ Gene Expr.   │   │ CT Tumor     │   │ Cellpose     │
    │ Feature      │   │ ΔSize        │   │ Stress/Death │
    │ Extractor    │   │ Extractor    │   │ Extractor    │
    └──────┬───────┘   └──────┬───────┘   └──────┬───────┘
           │                  │                  │
           ▼                  ▼                  ▼
    ┌─────────────────────────────────────────────────┐
    │        Multi-Modal Fusion Module                │
    │   (Feature concatenation + weighted scoring)    │
    └──────────────────────┬──────────────────────────┘
                           │
                           ▼
    ┌─────────────────────────────────────────────────┐
    │        Treatment Response Predictor             │
    │   (Responder/Non-responder classification)      │
    └─────────────────────────────────────────────────┘
"""
import numpy as np
import pandas as pd
import pickle
import json
import logging
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple, Any

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


# ============================================================
# Data Classes
# ============================================================

@dataclass
class GeneExpressionFeatures:
    """유전자 발현 기반 특성."""
    source: str = ""                    # "tcga", "geo", "custom"
    n_genes: int = 0
    expression_vector: np.ndarray = field(default_factory=lambda: np.array([]))
    top_markers: Dict[str, float] = field(default_factory=dict)  # gene -> score
    pathway_scores: Dict[str, float] = field(default_factory=dict)
    timestamp: str = ""
    
    def to_feature_vector(self) -> np.ndarray:
        return self.expression_vector


@dataclass
class CTTumorFeatures:
    """CT 기반 종양 크기 변화 특성."""
    pre_volume_ml: float = 0.0          # 치료 전 종양 부피
    post_volume_ml: float = 0.0         # 치료 후 종양 부피
    delta_volume: float = 0.0           # 부피 변화 (ml)
    delta_volume_pct: float = 0.0       # 부피 변화 (%)
    recist_category: str = ""           # CR, PR, SD, PD
    longest_diameter_mm: float = 0.0
    delta_diameter_pct: float = 0.0
    hu_mean_change: float = 0.0         # HU 평균 변화 (밀도)
    new_lesions: int = 0                # 새 병변 수
    n_slices_affected: int = 0
    sphericity: float = 0.0             # 종양 형태 (1.0 = 완전 구)
    
    def to_feature_vector(self) -> np.ndarray:
        return np.array([
            self.pre_volume_ml,
            self.post_volume_ml, 
            self.delta_volume,
            self.delta_volume_pct,
            self.longest_diameter_mm,
            self.delta_diameter_pct,
            self.hu_mean_change,
            self.new_lesions,
            self.n_slices_affected,
            self.sphericity,
            # RECIST-like scoring
            1.0 if self.recist_category == 'CR' else 0.0,
            1.0 if self.recist_category == 'PR' else 0.0,
            1.0 if self.recist_category == 'SD' else 0.0,
            1.0 if self.recist_category == 'PD' else 0.0,
        ], dtype=np.float32)


@dataclass
class CellposeFeatures:
    """Cellpose 기반 암세포 스트레스/사멸 특성."""
    total_cells: int = 0
    viable_cells: int = 0
    apoptotic_cells: int = 0            # 세포 사멸 (사멸체 형성)
    necrotic_cells: int = 0             # 괴사
    stressed_cells: int = 0             # 스트레스 (형태 변화)
    viability_ratio: float = 0.0        # 생존율
    apoptosis_rate: float = 0.0
    necrosis_rate: float = 0.0
    stress_index: float = 0.0
    mean_cell_area_um2: float = 0.0
    cell_area_cv: float = 0.0           # 면적 변동계수
    mean_circularity: float = 0.0       # 원형도 (1.0 = 원형)
    mean_solidity: float = 0.0          # 솔리디티
    cell_density_per_mm2: float = 0.0
    cluster_count: int = 0
    ki67_positive_rate: float = 0.0     # 증식 마커 (있으면)
    
    def to_feature_vector(self) -> np.ndarray:
        return np.array([
            self.total_cells,
            self.viable_cells,
            self.apoptotic_cells,
            self.necrotic_cells,
            self.stressed_cells,
            self.viability_ratio,
            self.apoptosis_rate,
            self.necrosis_rate,
            self.stress_index,
            self.mean_cell_area_um2,
            self.cell_area_cv,
            self.mean_circularity,
            self.mean_solidity,
            self.cell_density_per_mm2,
            self.cluster_count,
            self.ki67_positive_rate,
        ], dtype=np.float32)


@dataclass
class MultiModalResponse:
    """통합 치료 반응 예측 결과."""
    patient_id: str = ""
    gene_features: Optional[GeneExpressionFeatures] = None
    ct_features: Optional[CTTumorFeatures] = None
    cellpose_features: Optional[CellposeFeatures] = None
    response_probability: float = 0.0   # 반응 확률 (0~1)
    response_prediction: str = ""        # "Responder" / "Non-responder"
    confidence: float = 0.0
    modality_contributions: Dict[str, float] = field(default_factory=dict)
    timestamp: str = ""


# ============================================================
# Feature Extractors (Modular)
# ============================================================

class GeneExpressionExtractor:
    """유전자 발현 특성 추출기."""
    
    # FOLFOX 관련 약물 대사/감수성 유전자 (문헌 기반)
    FOLFOX_MARKERS = {
        # 5-FU pathway
        'TYMS': 'thymidylate_synthase',           # 5-FU 표적
        'DPYD': 'dpyd',                            # 5-FU 대사
        'MTHFR': 'mthfr',                         # 엽산 대사
        'UMPS': 'umps',                            # 5-FU 활성화
        # Oxaliplatin pathway
        'ERCC1': 'ercc1',                          # DNA repair → oxaliplatin 저항
        'ERCC2': 'ercc2',                          # DNA repair
        'XRCC1': 'xrcc1',                          # Base excision repair
        'GSTP1': 'gstp1',                          # 해독
        # General chemo response
        'TP53': 'p53',                             # 세포 사멸
        'MDR1': 'mdr1',                            # 약물 유출 펌프
        'BAX': 'bax',                              # 프로-세포사멸
        'BCL2': 'bcl2',                            # 항-세포사멸
        'VEGFA': 'vegf',                           # 혈관신생
        'EGFR': 'egfr',                            # 성장인자 수용체
        'MSH2': 'msh2',                            # Mismatch repair → MSI
        'MLH1': 'mlh1',                            # Mismatch repair
    }
    
    def __init__(self, model_path: Optional[str] = None):
        self.model = None
        self.scaler = None
        self.selected_probes = None
        
        if model_path and Path(model_path).exists():
            with open(model_path, 'rb') as f:
                data = pickle.load(f)
                self.model = data.get('model')
                self.scaler = data.get('scaler')
                self.selected_probes = data.get('selected_probes')
    
    def extract(self, expression_data: Dict[str, float], source: str = "custom") -> GeneExpressionFeatures:
        """유전자 발현 데이터에서 특성 추출."""
        features = GeneExpressionFeatures(
            source=source,
            n_genes=len(expression_data),
            timestamp=datetime.now().isoformat(),
        )
        
        # 약물 반응 관련 유전자 점수
        for gene, role in self.FOLFOX_MARKERS.items():
            if gene in expression_data:
                features.top_markers[gene] = expression_data[gene]
        
        # Pathway scores
        features.pathway_scores = self._compute_pathway_scores(expression_data)
        
        # Expression vector for model
        if self.selected_probes:
            vec = np.array([expression_data.get(p, 0.0) for p in self.selected_probes], dtype=np.float32)
        else:
            # Use marker genes as features
            vec = np.array([expression_data.get(g, 0.0) for g in self.FOLFOX_MARKERS], dtype=np.float32)
        
        features.expression_vector = vec
        return features
    
    def _compute_pathway_scores(self, expr: Dict[str, float]) -> Dict[str, float]:
        """경로별 점수 계산."""
        scores = {}
        
        # 5-FU sensitivity = TYMS↓ + DPYD↑ (good response)
        tyms = expr.get('TYMS', 0)
        dpyd = expr.get('DPYD', 0)
        scores['5fu_sensitivity'] = max(0, dpyd - tyms) / max(abs(tyms) + abs(dpyd), 1)
        
        # Oxaliplatin sensitivity = ERCC1↓ (good response)
        ercc1 = expr.get('ERCC1', 0)
        scores['oxaliplatin_sensitivity'] = -ercc1 / max(abs(ercc1), 1)
        
        # Apoptosis readiness = BAX/BCL2 ratio
        bax = expr.get('BAX', 1)
        bcl2 = expr.get('BCL2', 1)
        scores['apoptosis_readiness'] = bax / max(bcl2, 0.1)
        
        # DNA repair capacity
        repair_genes = ['ERCC1', 'ERCC2', 'XRCC1', 'MSH2', 'MLH1']
        repair_vals = [expr.get(g, 0) for g in repair_genes]
        scores['dna_repair_capacity'] = np.mean(repair_vals) if repair_vals else 0
        
        # Drug efflux
        scores['drug_efflux'] = expr.get('MDR1', 0) + expr.get('GSTP1', 0)
        
        return scores


class CTTumorExtractor:
    """CT 종양 크기 변화 특성 추출기."""
    
    def extract(self, pre_ct_results: Dict, post_ct_results: Dict) -> CTTumorFeatures:
        """치료 전후 CT 결과에서 종양 변화 특성 추출."""
        features = CTTumorFeatures()
        
        # 부피
        pre_vol = pre_ct_results.get('total_volume_ml', 0)
        post_vol = post_ct_results.get('total_volume_ml', 0)
        
        features.pre_volume_ml = pre_vol
        features.post_volume_ml = post_vol
        features.delta_volume = post_vol - pre_vol
        features.delta_volume_pct = (post_vol - pre_vol) / max(pre_vol, 0.01) * 100
        
        # Diameter
        pre_diam = pre_ct_results.get('max_diameter_mm', 0)
        post_diam = post_ct_results.get('max_diameter_mm', 0)
        features.longest_diameter_mm = post_diam
        features.delta_diameter_pct = (post_diam - pre_diam) / max(pre_diam, 0.01) * 100
        
        # RECIST classification
        features.recist_category = self._classify_recist(features.delta_diameter_pct,
                                                          post_ct_results.get('new_lesions', 0))
        
        # HU changes
        features.hu_mean_change = post_ct_results.get('mean_hu', 0) - pre_ct_results.get('mean_hu', 0)
        
        features.new_lesions = post_ct_results.get('new_lesions', 0)
        features.n_slices_affected = post_ct_results.get('affected_slices', 0)
        features.sphericity = post_ct_results.get('sphericity', 0.5)
        
        return features
    
    def extract_from_detect(self, tumor_results: Dict) -> CTTumorFeatures:
        """단일 CT 검출 결과에서 특성 추출 (종양 자체 특성)."""
        features = CTTumorFeatures()
        
        summary = tumor_results.get('summary', tumor_results)
        
        features.pre_volume_ml = summary.get('total_volume_ml', 0)
        features.n_slices_affected = summary.get('affected_slices', 0)
        features.sphericity = summary.get('sphericity', 0.5)
        
        # 3D bounding box → longest diameter 추정
        bbox = summary.get('bounding_box_3d', {})
        if bbox:
            dims = [
                bbox.get('x_max', 0) - bbox.get('x_min', 0),
                bbox.get('y_max', 0) - bbox.get('y_min', 0),
                bbox.get('z_max', 0) - bbox.get('z_min', 0),
            ]
            features.longest_diameter_mm = max(dims)
        
        return features
    
    def _classify_recist(self, delta_diam_pct: float, new_lesions: int) -> str:
        """RECIST 1.1 기준 분류."""
        if delta_diam_pct <= -100:
            return 'CR'  # Complete Response
        elif delta_diam_pct <= -30:
            return 'PR'  # Partial Response
        elif delta_diam_pct >= 20 or new_lesions > 0:
            return 'PD'  # Progressive Disease
        else:
            return 'SD'  # Stable Disease


class CellposeExtractor:
    """Cellpose 기반 암세포 스트레스/사멸 특성 추출기."""
    
    # Cellpose 분석 결과에서 스트레스/사멸 판단 기준
    CIRCULARITY_THRESHOLD = 0.7    # 원형도 < 0.7 → 스트레스
    AREA_DEVIATION = 2.0           # 평균±2σ 밖 → 비정상
    SOLIDITY_THRESHOLD = 0.85      # 솔리디티 < 0.85 → 불규칙
    
    def extract(self, cellpose_results: Dict) -> CellposeFeatures:
        """Cellpose 분석 결과에서 세포 스트레스/사멸 특성 추출."""
        features = CellposeFeatures()
        
        cells = cellpose_results.get('cells', [])
        if not cells:
            # Use summary stats if individual cell data not available
            return self._extract_from_summary(cellpose_results)
        
        features.total_cells = len(cells)
        
        areas = [c.get('area', 0) for c in cells]
        circularities = [c.get('circularity', 1.0) for c in cells]
        solidities = [c.get('solidity', 1.0) for c in cells]
        
        mean_area = np.mean(areas) if areas else 0
        std_area = np.std(areas) if areas else 0
        
        viable = 0
        apoptotic = 0
        stressed = 0
        
        for c in cells:
            circ = c.get('circularity', 1.0)
            area = c.get('area', mean_area)
            sol = c.get('solidity', 1.0)
            
            # 세포사멸 판단: 매우 작고 원형 (사멸체)
            if area < mean_area * 0.3 and circ > 0.8:
                apoptotic += 1
            # 스트레스: 불규칙한 형태
            elif circ < self.CIRCULARITY_THRESHOLD or sol < self.SOLIDITY_THRESHOLD:
                stressed += 1
            # 비정상 크기
            elif abs(area - mean_area) > self.AREA_DEVIATION * std_area:
                stressed += 1
            else:
                viable += 1
        
        features.viable_cells = viable
        features.apoptotic_cells = apoptotic
        features.stressed_cells = stressed
        features.viability_ratio = viable / max(features.total_cells, 1)
        features.apoptosis_rate = apoptotic / max(features.total_cells, 1)
        features.stress_index = stressed / max(features.total_cells, 1)
        
        features.mean_cell_area_um2 = mean_area
        features.cell_area_cv = std_area / max(mean_area, 1e-6)
        features.mean_circularity = np.mean(circularities)
        features.mean_solidity = np.mean(solidities)
        
        # 세포 밀도 (mm^2당)
        image_area = cellpose_results.get('image_area_mm2', 1.0)
        features.cell_density_per_mm2 = features.total_cells / image_area
        
        return features
    
    def _extract_from_summary(self, results: Dict) -> CellposeFeatures:
        """요약 통계에서 추출 (개별 세포 데이터 없을 때)."""
        features = CellposeFeatures()
        features.total_cells = results.get('total_cells', results.get('cell_count', 0))
        features.mean_cell_area_um2 = results.get('avg_cell_area', results.get('mean_area', 0))
        features.mean_circularity = results.get('avg_circularity', results.get('mean_circularity', 0.8))
        features.cell_density_per_mm2 = results.get('cell_density', 0)
        
        # 스트레스 비율 추정
        circ = features.mean_circularity
        features.stress_index = max(0, 1 - circ / 0.9) if circ > 0 else 0.5
        features.viability_ratio = 1 - features.stress_index * 0.5
        
        return features


# ============================================================
# Multi-Modal Fusion Module
# ============================================================

class MultiModalFusionModule:
    """다중 모달리티 통합 모듈."""
    
    # 기본 가중치 (문헌 + 데이터 기반 조정 가능)
    DEFAULT_WEIGHTS = {
        'gene_expression': 0.40,
        'ct_tumor': 0.35,
        'cellpose': 0.25,
    }
    
    def __init__(self, weights: Optional[Dict[str, float]] = None):
        self.weights = weights or self.DEFAULT_WEIGHTS.copy()
        self.gene_extractor = GeneExpressionExtractor()
        self.ct_extractor = CTTumorExtractor()
        self.cellpose_extractor = CellposeExtractor()
    
    def predict(self,
                patient_id: str,
                gene_data: Optional[Dict[str, float]] = None,
                ct_pre: Optional[Dict] = None,
                ct_post: Optional[Dict] = None,
                ct_single: Optional[Dict] = None,
                cellpose_data: Optional[Dict] = None) -> MultiModalResponse:
        """
        다중 모달리티 통합 치료 반응 예측.
        
        최소 1개 modality만 있어도 예측 가능.
        """
        response = MultiModalResponse(
            patient_id=patient_id,
            timestamp=datetime.now().isoformat(),
        )
        
        scores = {}
        contributions = {}
        available_weight_sum = 0
        
        # 1. Gene Expression
        if gene_data:
            gene_feat = self.gene_extractor.extract(gene_data)
            response.gene_features = gene_feat
            gene_score = self._gene_response_score(gene_feat)
            scores['gene_expression'] = gene_score
            available_weight_sum += self.weights['gene_expression']
        
        # 2. CT Tumor
        if ct_pre and ct_post:
            ct_feat = self.ct_extractor.extract(ct_pre, ct_post)
            response.ct_features = ct_feat
            ct_score = self._ct_response_score(ct_feat)
            scores['ct_tumor'] = ct_score
            available_weight_sum += self.weights['ct_tumor']
        elif ct_single:
            ct_feat = self.ct_extractor.extract_from_detect(ct_single)
            response.ct_features = ct_feat
            # 단일 CT: 종양 특성만으로 예후 추정
            ct_score = self._ct_prognosis_score(ct_feat)
            scores['ct_tumor'] = ct_score
            available_weight_sum += self.weights['ct_tumor'] * 0.5  # 낮은 가중치
        
        # 3. Cellpose
        if cellpose_data:
            cell_feat = self.cellpose_extractor.extract(cellpose_data)
            response.cellpose_features = cell_feat
            cell_score = self._cellpose_response_score(cell_feat)
            scores['cellpose'] = cell_score
            available_weight_sum += self.weights['cellpose']
        
        if not scores:
            response.response_prediction = "Insufficient Data"
            response.response_probability = 0.5
            response.confidence = 0.0
            return response
        
        # Fusion: 가중 평균
        weighted_sum = 0
        for modality, score in scores.items():
            w = self.weights.get(modality, 0)
            weighted_sum += w * score
            contributions[modality] = w * score / max(available_weight_sum, 1e-6)
        
        prob = weighted_sum / max(available_weight_sum, 1e-6)
        prob = np.clip(prob, 0, 1)
        
        response.response_probability = float(prob)
        response.response_prediction = "Responder" if prob > 0.5 else "Non-responder"
        response.confidence = float(abs(prob - 0.5) * 2)  # 0~1
        response.modality_contributions = contributions
        
        return response
    
    def _gene_response_score(self, feat: GeneExpressionFeatures) -> float:
        """유전자 발현 기반 반응 점수 (0=NR, 1=R)."""
        pw = feat.pathway_scores
        score = 0.5
        
        # 5FU 감수성 기여
        score += pw.get('5fu_sensitivity', 0) * 0.2
        # Oxaliplatin 감수성
        score += pw.get('oxaliplatin_sensitivity', 0) * 0.15
        # 세포사멸 준비도
        apo = pw.get('apoptosis_readiness', 1.0)
        score += min(0.15, (apo - 1.0) * 0.1) if apo > 1.0 else max(-0.15, (apo - 1.0) * 0.1)
        # DNA repair (높으면 저항)
        repair = pw.get('dna_repair_capacity', 0)
        score -= repair * 0.01
        # Drug efflux (높으면 저항)
        efflux = pw.get('drug_efflux', 0)
        score -= efflux * 0.005
        
        return np.clip(score, 0, 1)
    
    def _ct_response_score(self, feat: CTTumorFeatures) -> float:
        """CT 종양 변화 기반 반응 점수."""
        # RECIST 기반
        recist_scores = {'CR': 1.0, 'PR': 0.75, 'SD': 0.4, 'PD': 0.1}
        base = recist_scores.get(feat.recist_category, 0.5)
        
        # 부피 변화 보정
        if feat.delta_volume_pct < -50:
            base += 0.1
        elif feat.delta_volume_pct > 50:
            base -= 0.1
        
        # 새 병변
        if feat.new_lesions > 0:
            base -= 0.15 * min(feat.new_lesions, 3)
        
        return np.clip(base, 0, 1)
    
    def _ct_prognosis_score(self, feat: CTTumorFeatures) -> float:
        """단일 CT에서 예후 추정."""
        score = 0.5
        
        # 작은 종양 → 좋은 예후
        if feat.pre_volume_ml < 5:
            score += 0.15
        elif feat.pre_volume_ml > 50:
            score -= 0.15
        
        # 높은 구형도 → 덜 침습적
        score += (feat.sphericity - 0.5) * 0.2
        
        return np.clip(score, 0, 1)
    
    def _cellpose_response_score(self, feat: CellposeFeatures) -> float:
        """Cellpose 세포 분석 기반 반응 점수."""
        score = 0.5
        
        # 높은 세포사멸율 → 치료 효과
        score += feat.apoptosis_rate * 0.5
        
        # 높은 스트레스 → 치료 효과
        score += feat.stress_index * 0.3
        
        # 낮은 생존율 → 치료 효과
        score += (1 - feat.viability_ratio) * 0.2
        
        # 낮은 원형도 → 형태 변화 (치료 효과)
        if feat.mean_circularity < 0.7:
            score += 0.1
        
        return np.clip(score, 0, 1)


# ============================================================
# Pipeline Orchestrator
# ============================================================

class TreatmentResponsePipeline:
    """치료 반응 예측 통합 파이프라인."""
    
    def __init__(self, model_dir: str = BASE_DIR / "models/synergy"):
        self.model_dir = Path(model_dir)
        self.fusion = MultiModalFusionModule()
        self.results_history: List[MultiModalResponse] = []
    
    def analyze_patient(self,
                       patient_id: str,
                       gene_expression: Optional[Dict[str, float]] = None,
                       ct_pre_results: Optional[Dict] = None,
                       ct_post_results: Optional[Dict] = None,
                       ct_detection: Optional[Dict] = None,
                       cellpose_results: Optional[Dict] = None) -> MultiModalResponse:
        """환자 데이터 분석 → 치료 반응 예측."""
        logger.info(f"Analyzing patient {patient_id}...")
        
        modalities = []
        if gene_expression: modalities.append("Gene Expression")
        if ct_pre_results and ct_post_results: modalities.append("CT (Pre/Post)")
        elif ct_detection: modalities.append("CT (Single)")
        if cellpose_results: modalities.append("Cellpose")
        
        logger.info(f"  Available modalities: {modalities}")
        
        response = self.fusion.predict(
            patient_id=patient_id,
            gene_data=gene_expression,
            ct_pre=ct_pre_results,
            ct_post=ct_post_results,
            ct_single=ct_detection,
            cellpose_data=cellpose_results,
        )
        
        self.results_history.append(response)
        
        return response
    
    def generate_report(self, response: MultiModalResponse) -> Dict[str, Any]:
        """분석 결과 보고서 생성."""
        report = {
            'patient_id': response.patient_id,
            'timestamp': response.timestamp,
            'prediction': response.response_prediction,
            'probability': round(response.response_probability, 4),
            'confidence': round(response.confidence, 4),
            'modality_contributions': response.modality_contributions,
        }
        
        if response.gene_features:
            report['gene_expression'] = {
                'source': response.gene_features.source,
                'n_genes': response.gene_features.n_genes,
                'pathway_scores': response.gene_features.pathway_scores,
                'top_markers': response.gene_features.top_markers,
            }
        
        if response.ct_features:
            report['ct_tumor'] = {
                'pre_volume_ml': response.ct_features.pre_volume_ml,
                'post_volume_ml': response.ct_features.post_volume_ml,
                'delta_volume_pct': round(response.ct_features.delta_volume_pct, 1),
                'recist_category': response.ct_features.recist_category,
                'new_lesions': response.ct_features.new_lesions,
            }
        
        if response.cellpose_features:
            report['cellpose'] = {
                'total_cells': response.cellpose_features.total_cells,
                'viability_ratio': round(response.cellpose_features.viability_ratio, 3),
                'apoptosis_rate': round(response.cellpose_features.apoptosis_rate, 3),
                'stress_index': round(response.cellpose_features.stress_index, 3),
            }
        
        return report
    
    def save_report(self, report: Dict, output_path: str):
        """보고서 JSON 저장."""
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False, default=str)
        logger.info(f"  Report saved: {output_path}")


# ============================================================
# Demo & Validation
# ============================================================

def run_demo():
    """Multi-modal pipeline 데모."""

import os as _os
from pathlib import Path as _Path
# ADDS_BASE_DIR environment variable overrides automatic detection
BASE_DIR = _Path(_os.environ.get("ADDS_BASE_DIR", str(_Path(__file__).resolve().parent.parent)))

    print("=" * 70)
    print("Multi-Modal Treatment Response Pipeline — DEMO")
    print("=" * 70)
    
    pipeline = TreatmentResponsePipeline()
    
    # ===== Case 1: 반응자 (Responder) =====
    print(f"\n{'='*70}")
    print("Case 1: Expected Responder")
    print("=" * 70)
    
    resp1 = pipeline.analyze_patient(
        patient_id="DEMO-001",
        gene_expression={
            'TYMS': 2.1,     # Low TYMS → good 5-FU response
            'DPYD': 8.5,     # Normal DPYD
            'ERCC1': 1.2,    # Low ERCC1 → good oxaliplatin response
            'ERCC2': 2.0,
            'XRCC1': 3.0,
            'GSTP1': 2.5,
            'TP53': 5.0,
            'MDR1': 1.0,     # Low efflux
            'BAX': 8.0,      # High pro-apoptotic
            'BCL2': 2.0,     # Low anti-apoptotic
            'VEGFA': 3.0,
            'EGFR': 2.5,
            'MSH2': 6.0,
            'MLH1': 5.5,
        },
        ct_pre_results={
            'total_volume_ml': 25.0,
            'max_diameter_mm': 35.0,
            'mean_hu': 45.0,
        },
        ct_post_results={
            'total_volume_ml': 8.0,
            'max_diameter_mm': 20.0,
            'mean_hu': 30.0,
            'new_lesions': 0,
        },
        cellpose_results={
            'cells': [
                {'area': 200, 'circularity': 0.85, 'solidity': 0.9},
                {'area': 50, 'circularity': 0.9, 'solidity': 0.95},    # apoptotic (small+round)
                {'area': 180, 'circularity': 0.5, 'solidity': 0.7},    # stressed
                {'area': 210, 'circularity': 0.88, 'solidity': 0.92},
                {'area': 45, 'circularity': 0.92, 'solidity': 0.96},   # apoptotic
                {'area': 190, 'circularity': 0.65, 'solidity': 0.75},  # stressed
                {'area': 205, 'circularity': 0.82, 'solidity': 0.89},
                {'area': 195, 'circularity': 0.87, 'solidity': 0.91},
            ],
            'image_area_mm2': 0.5,
        }
    )
    
    report1 = pipeline.generate_report(resp1)
    print(json.dumps(report1, indent=2, ensure_ascii=False, default=str))
    
    # ===== Case 2: 비반응자 (Non-responder) =====
    print(f"\n{'='*70}")
    print("Case 2: Expected Non-responder")
    print("=" * 70)
    
    resp2 = pipeline.analyze_patient(
        patient_id="DEMO-002",
        gene_expression={
            'TYMS': 12.0,    # High TYMS → 5-FU resistance
            'DPYD': 9.0,
            'ERCC1': 8.5,    # High ERCC1 → oxaliplatin resistance
            'ERCC2': 7.0,
            'XRCC1': 7.5,
            'GSTP1': 9.0,    # High detox
            'TP53': 1.0,     # Low p53 → impaired apoptosis
            'MDR1': 8.0,     # High efflux
            'BAX': 2.0,      # Low pro-apoptotic
            'BCL2': 9.0,     # High anti-apoptotic
            'VEGFA': 8.0,
            'EGFR': 7.0,
            'MSH2': 1.5,
            'MLH1': 1.0,
        },
        ct_pre_results={
            'total_volume_ml': 30.0,
            'max_diameter_mm': 40.0,
            'mean_hu': 50.0,
        },
        ct_post_results={
            'total_volume_ml': 55.0,
            'max_diameter_mm': 55.0,
            'mean_hu': 55.0,
            'new_lesions': 2,
        },
        cellpose_results={
            'cells': [
                {'area': 200, 'circularity': 0.9, 'solidity': 0.95},
                {'area': 210, 'circularity': 0.88, 'solidity': 0.93},
                {'area': 195, 'circularity': 0.92, 'solidity': 0.94},
                {'area': 205, 'circularity': 0.87, 'solidity': 0.91},
                {'area': 215, 'circularity': 0.85, 'solidity': 0.92},
            ],
            'image_area_mm2': 0.3,
        }
    )
    
    report2 = pipeline.generate_report(resp2)
    print(json.dumps(report2, indent=2, ensure_ascii=False, default=str))
    
    # ===== Case 3: 유전자 발현만 =====
    print(f"\n{'='*70}")
    print("Case 3: Gene Expression Only")
    print("=" * 70)
    
    resp3 = pipeline.analyze_patient(
        patient_id="DEMO-003",
        gene_expression={
            'TYMS': 5.0, 'DPYD': 6.0, 'ERCC1': 4.0,
            'BAX': 5.0, 'BCL2': 5.0, 'MDR1': 3.0,
        }
    )
    
    report3 = pipeline.generate_report(resp3)
    print(json.dumps(report3, indent=2, ensure_ascii=False, default=str))
    
    # Save reports
    output_dir = Path(BASE_DIR / "outputs/multimodal_reports")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for i, report in enumerate([report1, report2, report3], 1):
        pipeline.save_report(report, str(output_dir / f"demo_case_{i}.json"))
    
    # Summary
    print(f"\n{'='*70}")
    print("PIPELINE VALIDATION SUMMARY")
    print("=" * 70)
    
    cases = [
        ("Case 1 (Expected R)", resp1),
        ("Case 2 (Expected NR)", resp2),
        ("Case 3 (Gene Only)", resp3),
    ]
    
    for name, resp in cases:
        print(f"  {name:25s}: {resp.response_prediction:15s} "
              f"(P={resp.response_probability:.3f}, Conf={resp.confidence:.3f})")
        if resp.modality_contributions:
            for mod, contrib in resp.modality_contributions.items():
                print(f"    → {mod}: {contrib:.3f}")
    
    # Validation
    correct = 0
    if resp1.response_prediction == "Responder": correct += 1
    if resp2.response_prediction == "Non-responder": correct += 1
    
    print(f"\n  Demo accuracy: {correct}/2 correct")
    print(f"  Pipeline components: 3 extractors + 1 fusion module")
    print(f"  Reports saved: {output_dir}")


if __name__ == "__main__":
    run_demo()
