"""
pathway_drug_optimizer.py
--------------------------
Step 3: 경로 감도 점수 → 약물 우선순위 → Bliss synergy 재계산 → 최적 칵테일 출력

입력: PatientEnergyProfile (imaging_to_energy.py)
출력: DrugCocktailRecommendation
  - recommended_drugs (우선순위 정렬)
  - doses_relative (EC50 배수 — 환자 프로파일 기반 조정)
  - synergy_matrix (Bliss 시너지 행렬)
  - narrative (임상 해석 텍스트)
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

# ── 약물-경로 매핑 테이블 (문헌 기반) ──────────────────────────────────────────
# Format: pathway → {drug: (EC50_alone_nM, max_E, specificity)}
#   specificity: 해당 경로에 특이적인 정도 (0~1, 1=완전 특이적)
DRUG_PATHWAY_MAP: Dict[str, Dict[str, dict]] = {
    "KRAS_ERK": {
        "Sotorasib":    {"EC50_nM": 75.0,   "maxE": 0.80, "spec": 0.90},
        "Trametinib":   {"EC50_nM": 1.0,    "maxE": 0.75, "spec": 0.85},
        "FOLFOX":       {"EC50_nM": 3750.0, "maxE": 0.60, "spec": 0.30},
        "FOLFIRI":      {"EC50_nM": 7500.0, "maxE": 0.55, "spec": 0.25},
    },
    "PI3K_mTOR": {
        "Everolimus":   {"EC50_nM": 0.2,    "maxE": 0.70, "spec": 0.92},
        "Buparlisib":   {"EC50_nM": 100.0,  "maxE": 0.65, "spec": 0.85},
        "FOLFOX":       {"EC50_nM": 3750.0, "maxE": 0.60, "spec": 0.20},
    },
    "HIF_VEGF": {
        "Bevacizumab":  {"EC50_nM": 50.0,   "maxE": 0.65, "spec": 0.88},
        "Lenvatinib":   {"EC50_nM": 4.0,    "maxE": 0.70, "spec": 0.80},
        "Regorafenib":  {"EC50_nM": 900.0,  "maxE": 0.55, "spec": 0.60},
    },
    "RhoA_INV": {
        "Fasudil":      {"EC50_nM": 330.0,  "maxE": 0.55, "spec": 0.80},
        "Y-27632":      {"EC50_nM": 14.0,   "maxE": 0.60, "spec": 0.85},
    },
    # Pritamab은 모든 경로에 상류 조절 (PrPc-RPSA 시그날로솜)
    "ALL": {
        "Pritamab":     {"EC50_nM": 0.84,   "maxE": 0.85, "spec": 0.70},
    }
}

# 임상적 사용 가능성 점수 (현재 CRC 승인/임상 여부 기반)
CLINICAL_AVAILABILITY: Dict[str, float] = {
    "Pritamab":    0.70,   # 임상 개발중 (investigational)
    "FOLFOX":      1.00,   # 표준요법
    "FOLFIRI":     1.00,
    "Sotorasib":   0.90,   # KRAS G12C 승인 (G12D는 investigational)
    "Trametinib":  0.85,
    "Bevacizumab": 0.95,
    "Everolimus":  0.90,
    "Lenvatinib":  0.85,
    "Buparlisib":  0.60,
    "Regorafenib": 0.85,
    "Fasudil":     0.50,
    "Y-27632":     0.30,
}

RT_KCAL = 0.593  # 37°C


@dataclass
class DrugCocktailRecommendation:
    """
    최적 항암 칵테일 추천 결과.

    Attributes
    ----------
    recommended_drugs : List[str]
        우선순위 정렬된 약물 목록 (최대 5종).
    pathway_priority : List[str]
        활성화 경로 순위 (imaging_to_energy 결과).
    doses_relative : Dict[str, float]
        각 약물의 권장 용량 (EC50 배수, 1.0=표준량).
    synergy_matrix : np.ndarray
        약물간 Bliss synergy 행렬 (n_drugs × n_drugs).
    drug_names_matrix : List[str]
        synergy_matrix의 행/열 이름.
    coverage_scores : Dict[str, float]
        각 활성 경로의 약물 커버리지 (0~1).
    risk_score : float
        약물 조합 독성 위험 추정 (0~1, 낮을수록 안전).
    narrative : str
        임상 해석 텍스트.
    """
    recommended_drugs:  List[str]           = field(default_factory=list)
    pathway_priority:   List[str]           = field(default_factory=list)
    doses_relative:     Dict[str, float]    = field(default_factory=dict)
    synergy_matrix:     Optional[np.ndarray]= None
    drug_names_matrix:  List[str]           = field(default_factory=list)
    coverage_scores:    Dict[str, float]    = field(default_factory=dict)
    risk_score:         float               = 0.5
    narrative:          str                 = ""

    def to_dict(self) -> dict:
        return {
            "recommended_drugs":  self.recommended_drugs,
            "pathway_priority":   self.pathway_priority,
            "doses_relative":     {k: round(v, 3) for k, v in self.doses_relative.items()},
            "coverage_scores":    {k: round(v, 3) for k, v in self.coverage_scores.items()},
            "risk_score":         round(self.risk_score, 3),
            "narrative":          self.narrative,
        }


class PathwayDrugOptimizer:
    """
    PatientEnergyProfile → 최적 항암 칵테일 추천.

    알고리즘 흐름
    1. 경로 감도 순위에서 상위 N개 경로 선택
    2. 각 경로에서 k_pathway 가중 약물 점수 계산
    3. 중복 제거 후 상위 약물 선택 (최대 5종, Pritamab 자동 포함)
    4. 환자-특이적 EC50 조정 (ddg 기반)
    5. 선택 약물간 Bliss synergy 행렬 계산
    6. 임상 narrative 생성
    """

    def __init__(self, max_drugs: int = 5, top_pathways: int = 3):
        self.max_drugs     = max_drugs
        self.top_pathways  = top_pathways

    # ── 약물 점수 계산 ─────────────────────────────────────────────────────────
    def _score_drug(self, drug: str, info: dict, k_pathway: float,
                    kras_allele: str) -> float:
        """
        pathway_k · maxE · availability · specificity · kras_boost
        """
        kras_boost = 1.0
        if drug == "Sotorasib" and kras_allele not in ("G12C",):
            kras_boost = 0.50   # G12C 특이적 — 다른 allele은 감점
        if drug == "Sotorasib" and kras_allele == "G12D":
            kras_boost = 0.40
        avail = CLINICAL_AVAILABILITY.get(drug, 0.5)
        return (k_pathway * info["maxE"] * info["spec"] * avail * kras_boost)

    # ── 환자-특이적 EC50 조정 ──────────────────────────────────────────────────
    @staticmethod
    def _adjust_ec50(base_ec50_nM: float, ddg_pathway: float) -> float:
        """
        ddg가 낮을수록 경로가 활성화 → 약물이 더 효과적임을 나타내지만
        세포가 이미 commitment했을 수 있어 EC50은 약간 증가.
        간단 모델: EC50_adj = EC50_base · (1 + 0.3·(1 - ddg/ddg_max))
        """
        DDG_MAX = 1.40
        shift = 0.3 * (1.0 - ddg_pathway / DDG_MAX)
        return base_ec50_nM * (1.0 + max(shift, 0.0))

    # ── Bliss Independence 시너지 ──────────────────────────────────────────────
    @staticmethod
    def _bliss_synergy(E1: float, E2: float,
                       k_pw1: float, k_pw2: float) -> float:
        """
        E_expected = E1 + E2 - E1·E2  (Bliss independence)
        시너지 보너스: 두 경로 k가 모두 높을 때 교차억제 증가.
        """
        E_exp = E1 + E2 - E1 * E2
        cross_bonus = 0.08 * min(k_pw1, k_pw2) / max(k_pw1, k_pw2)
        E_obs = min(E_exp * (1 + cross_bonus), 0.97)
        return round(E_obs - E_exp, 4)

    # ── 독성 위험 추정 ─────────────────────────────────────────────────────────
    @staticmethod
    def _toxicity_risk(drugs: List[str]) -> float:
        """
        약물 수와 중요 조합 독성에 기반한 간단 추정.
        임상 주의 조합: mTOR억제제 + 항VEGF (간독성 위험)
        """
        base_risk = 0.1 + 0.08 * len(drugs)
        if "Everolimus" in drugs and "Bevacizumab" in drugs:
            base_risk += 0.15
        if "Trametinib" in drugs and "Buparlisib" in drugs:
            base_risk += 0.10
        return float(np.clip(base_risk, 0, 1))

    # ── 핵심 최적화 ────────────────────────────────────────────────────────────
    def optimize(self, profile, kras_allele: str = "G12D") -> DrugCocktailRecommendation:
        """
        Parameters
        ----------
        profile : PatientEnergyProfile
        kras_allele : str
        """
        rank   = profile.sensitivity_rank
        k_pw   = profile.k_per_pathway
        ddg_pw = profile.ddg_per_pathway

        # ── 1. 경로별 약물 점수 집계 ─────────────────────────────────────────
        drug_scores: Dict[str, float] = {}
        drug_info_best: Dict[str, dict] = {}   # 최고 점수 정보 저장

        for pathway in rank[:self.top_pathways]:
            k = k_pw.get(pathway, 0.1)
            for drug, info in DRUG_PATHWAY_MAP.get(pathway, {}).items():
                s = self._score_drug(drug, info, k, kras_allele)
                if s > drug_scores.get(drug, -1):
                    drug_scores[drug] = s
                    drug_info_best[drug] = {**info, "pathway": pathway}

        # Pritamab 항상 포함 (PrPc 상류 조절)
        for drug, info in DRUG_PATHWAY_MAP["ALL"].items():
            k_mean = float(np.mean(list(k_pw.values())))
            s = self._score_drug(drug, info, k_mean, kras_allele)
            drug_scores[drug] = s
            drug_info_best[drug] = {**info, "pathway": "ALL"}

        # ── 2. 상위 max_drugs 선택 (Pritamab 강제 포함) ──────────────────────
        sorted_drugs = sorted(drug_scores, key=drug_scores.get, reverse=True)
        # Pritamab은 항상 포함 (PrPc-RPSA 상류 조절 — 전 경로에 영향)
        without_pritamab = [d for d in sorted_drugs if d != "Pritamab"]
        selected = without_pritamab[:self.max_drugs - 1] + ["Pritamab"]

        # ── 3. 환자-특이적 용량 조정 ──────────────────────────────────────────
        doses_rel: Dict[str, float] = {}
        for drug in selected:
            info = drug_info_best[drug]
            pw   = info.get("pathway", rank[0])
            ddg  = ddg_pw.get(pw, 1.0) if pw != "ALL" else profile.ddg_effective
            adj_ec50 = self._adjust_ec50(info["EC50_nM"], ddg)
            # 기준 EC50 대비 용량 비율 (1.0 = 표준, <1.0 = 감량, >1.0 = 증량)
            doses_rel[drug] = round(info["EC50_nM"] / adj_ec50, 3)

        # ── 4. Bliss synergy 행렬 ─────────────────────────────────────────────
        n = len(selected)
        syn_mat = np.zeros((n, n), dtype=np.float32)
        for i, d1 in enumerate(selected):
            for j, d2 in enumerate(selected):
                if i == j:
                    syn_mat[i, j] = 0.0
                    continue
                info1 = drug_info_best[d1]
                info2 = drug_info_best[d2]
                k1 = k_pw.get(info1.get("pathway", rank[0]), 0.1)
                k2 = k_pw.get(info2.get("pathway", rank[0]), 0.1)
                syn_mat[i, j] = self._bliss_synergy(
                    info1["maxE"], info2["maxE"], k1, k2
                )

        # ── 5. 경로 커버리지 ─────────────────────────────────────────────────
        coverage: Dict[str, float] = {}
        for pathway in rank:
            drugs_covering = [
                d for d in selected
                if drug_info_best.get(d, {}).get("pathway") in (pathway, "ALL")
            ]
            if drugs_covering:
                max_e = max(drug_info_best[d]["maxE"] for d in drugs_covering)
                coverage[pathway] = round(max_e, 3)
            else:
                coverage[pathway] = 0.0

        # ── 6. 독성 위험 ─────────────────────────────────────────────────────
        risk = self._toxicity_risk(selected)

        # ── 7. 임상 narrative ────────────────────────────────────────────────
        top_pw = rank[0] if rank else "KRAS_ERK"
        top_drug = selected[0] if selected else "Pritamab"
        narrative = (
            f"주요 활성 경로: {top_pw} (k={k_pw.get(top_pw, 0):.4f}). "
            f"ΔG‡_eff={profile.ddg_effective:.3f} kcal/mol → "
            f"표준 대비 {(1-profile.ddg_effective/1.2)*100:.1f}% 활성화 증가. "
            f"1차 표적약: {top_drug} (용량비={doses_rel.get(top_drug, 1.0):.2f}×EC50). "
            f"권장 칵테일: {', '.join(selected)}. "
            f"독성위험: {risk:.0%}."
        )

        return DrugCocktailRecommendation(
            recommended_drugs=selected,
            pathway_priority=rank,
            doses_relative=doses_rel,
            synergy_matrix=syn_mat,
            drug_names_matrix=selected,
            coverage_scores=coverage,
            risk_score=risk,
            narrative=narrative,
        )


# ── 자가 테스트 ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
    from src.pritamab_ml.imaging_to_energy import ImagingToEnergyMapper

    mapper    = ImagingToEnergyMapper()
    optimizer = PathwayDrugOptimizer(max_drugs=4)

    print("=" * 65)
    print("Case A: 고증식 (KRAS G12D) + 고침윤")
    summaries_A = [
        {"total_cells": 520, "mean_area_um2": 310.0,
         "mean_circularity": 0.50, "irregular_count": 290, "normal_count": 120},
    ] * 3
    ct_A = {"tumor_volume_cc": 45.0, "mean_hu": -8.0, "std_hu": 50.0, "necrosis_ratio": 0.20}
    profile_A = mapper.compute_profile(summaries_A, ct_A, kras_allele="G12D")
    rec_A = optimizer.optimize(profile_A, kras_allele="G12D")
    print(f"경로 순위: {rec_A.pathway_priority}")
    print(f"추천 칵테일: {rec_A.recommended_drugs}")
    print(f"용량 조정: {rec_A.doses_relative}")
    print(f"경로 커버리지: {rec_A.coverage_scores}")
    print(f"독성위험: {rec_A.risk_score:.1%}")
    print(f"Synergy 행렬:\n{rec_A.synergy_matrix.round(4)}")
    print(f"Narrative: {rec_A.narrative}")

    print()
    print("=" * 65)
    print("Case B: 저산소 주도 (KRAS WT)")
    summaries_B = [
        {"total_cells": 80, "mean_area_um2": 520.0,
         "mean_circularity": 0.75, "irregular_count": 10, "normal_count": 65},
    ] * 3
    ct_B = {"tumor_volume_cc": 12.0, "mean_hu": -35.0, "std_hu": 70.0, "necrosis_ratio": 0.45}
    profile_B = mapper.compute_profile(summaries_B, ct_B, kras_allele="WT")
    rec_B = optimizer.optimize(profile_B, kras_allele="WT")
    print(f"경로 순위: {rec_B.pathway_priority}")
    print(f"추천 칵테일: {rec_B.recommended_drugs}")
    print(f"Narrative: {rec_B.narrative}")
