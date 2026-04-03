"""
imaging_to_energy.py
--------------------
Step 1: 환자 이미징 데이터 → 경로별 활성화 에너지 ΔG‡ 추론

입력:
  - Cellpose 분석 결과 (cell_annotator.py 기반 summaries)
  - CT 분석 결과 (tumor_volume_cc, mean_hu, std_hu, necrosis_ratio)

출력:
  - PatientEnergyProfile: 4개 경로별 ddg, k_pathway, sensitivity_rank

과학적 근거:
  - Boltzmann-역추론: 경로 활성 프록시 → 유효 ΔG‡ 감소분 추정
  - Eyring 방정식: k = (kB·T/h) · exp(-ΔG‡/RT)
  - 각 경로의 k_pathway가 클수록 → 현재 활성화됨 → 우선 억제 표적
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional

# ── 열역학 상수 (37°C 생리 조건) ──────────────────────────────────────────────
RT_KCAL      = 0.593    # RT at 310 K (kcal/mol)
KB_OVER_H    = 6.25e12  # kB·T/h at 310 K (s⁻¹) — Eyring prefactor (×10⁻¹²)

# ── 경로 기준선 ΔG‡ (문헌 값, kcal/mol) ──────────────────────────────────────
BASELINE_DDG = {
    "KRAS_ERK":  1.20,   # KRAS->RAF->MEK->ERK 세포분열 경로
    "PI3K_mTOR": 1.17,   # PI3K->AKT->mTOR 단백합성/세포크기 (상향 조정: 독립 경로)
    "HIF_VEGF":  1.30,   # HIF-1a 저산소 응답 (하향: 저산소시 강한 효과)
    "RhoA_INV":  1.15,   # RhoA->ROCK->Cofilin 침윤/전이 경로
}

# ── 최대 ΔG‡ 감소분 허용치 (kcal/mol) — 물리적 하한 방지 ─────────────────────
MAX_DELTA_DDG = {k: v * 0.75 for k, v in BASELINE_DDG.items()}


@dataclass
class PatientEnergyProfile:
    """
    환자-특이적 에너지 지형.

    Attributes
    ----------
    ddg_per_pathway : Dict[str, float]
        각 경로의 유효 ΔG‡ (kcal/mol). 낮을수록 경로가 활성화됨.
    k_per_pathway : Dict[str, float]
        Eyring 방정식으로 계산된 경로별 유사-속도상수 (arbitrary units).
    sensitivity_rank : List[str]
        k 기준 내림차순 정렬 — 첫 번째가 가장 활성화된(우선 억제) 경로.
    ddg_effective : float
        전체 유효 ΔG‡ (PKPD 모듈용 단일 값, 가중 평균).
    imaging_indices : Dict[str, float]
        계산에 사용된 중간 지표들 (디버깅/로깅용).
    """
    ddg_per_pathway: Dict[str, float] = field(default_factory=dict)
    k_per_pathway:   Dict[str, float] = field(default_factory=dict)
    sensitivity_rank: List[str]       = field(default_factory=list)
    ddg_effective:    float           = 1.20   # fallback = KRAS baseline
    imaging_indices:  Dict[str, float]= field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "ddg_per_pathway":  self.ddg_per_pathway,
            "k_per_pathway":    self.k_per_pathway,
            "sensitivity_rank": self.sensitivity_rank,
            "ddg_effective":    round(self.ddg_effective, 4),
            "imaging_indices":  self.imaging_indices,
        }

    def __repr__(self):
        ranks = " > ".join(self.sensitivity_rank)
        return (f"PatientEnergyProfile("
                f"ddg_eff={self.ddg_effective:.3f} kcal/mol, "
                f"rank=[{ranks}])")


class ImagingToEnergyMapper:
    """
    Cellpose + CT 이미징 데이터를 경로별 ΔG‡로 변환합니다.

    사용 예시
    ---------
    mapper = ImagingToEnergyMapper()

    # Cellpose summaries (cell_annotator.py 출력 형식)
    cellpose_summaries = [
        {"total_cells": 527, "mean_area_um2": 320.0,
         "mean_circularity": 0.51, "irregular_count": 286,
         "normal_count": 124, "image_size": "1024x1024"},
    ]

    # CT 분석 결과 (선택사항)
    ct_data = {"tumor_volume_cc": 34.5, "mean_hu": -12.0,
               "std_hu": 45.0, "necrosis_ratio": 0.18}

    profile = mapper.compute_profile(cellpose_summaries, ct_data)
    print(profile)
    """

    def __init__(self, alpha: float = 0.35):
        """
        alpha : coupling coefficient (ADDS v1 literature value)
                ΔG‡_shift = alpha · index_normalized
        """
        self.alpha = alpha

    # ── 중간 지표 계산 ─────────────────────────────────────────────────────────

    def _proliferation_index(self, summaries: list) -> float:
        """
        KRAS-ERK 경로 활성 proxy.
        높은 세포 밀도 + 이형성 → 세포분열 속도 높음 → 유효 ΔG‡ 낮음.
        반환: 0~1 (1에 가까울수록 KRAS↑)
        """
        if not summaries:
            return 0.5
        total = sum(s.get("total_cells", 0) for s in summaries)
        irr   = sum(s.get("irregular_count", 0) for s in summaries)
        n_img = len(summaries)
        # 이미지당 세포 수 기준화 (200개 = 중간값)
        density_score = min(1.0, (total / n_img) / 300.0)
        # 불규칙 세포 비율
        irr_ratio = irr / max(total, 1)
        return float(np.clip(density_score * 0.6 + irr_ratio * 0.4, 0, 1))

    def _cell_size_index(self, summaries: list) -> float:
        """
        PI3K-mTOR 경로 활성 proxy.
        큰 세포 + 고분산 면적 → mTOR 항시 활성 → 유효 ΔG‡ 낮음.
        반환: 0~1 (1에 가까울수록 PI3K↑)
        """
        if not summaries:
            return 0.5
        areas = [s.get("mean_area_um2", 200.0) for s in summaries]
        mean_a = np.mean(areas)
        # 500 μm² = 비정상적으로 큰 세포로 간주
        size_score = min(1.0, mean_a / 500.0)
        return float(np.clip(size_score, 0, 1))

    def _invasion_score(self, summaries: list) -> float:
        """
        RhoA-Cofilin 침윤 경로 proxy.
        낮은 원형도 = 세포 골격 변형 = 침윤성 높음.
        반환: 0~1 (1에 가까울수록 RhoA↑)
        """
        if not summaries:
            return 0.3
        circs = [s.get("mean_circularity", 0.7) for s in summaries]
        mean_circ = np.mean(circs)
        # 원형도 0.5 이하 = 강한 침윤 표현형
        return float(np.clip(1.0 - mean_circ, 0, 1))

    def _hypoxia_score(self, ct_data: Optional[dict]) -> float:
        """
        HIF-1α 저산소 응답 proxy.
        CT 저음영 + 괴사 비율 → 저산소 → HIF 경로 활성.
        반환: 0~1 (1에 가까울수록 HIF↑)
        """
        if ct_data is None:
            return 0.3   # CT 없으면 moderate
        mean_hu      = ct_data.get("mean_hu", 30.0)
        std_hu       = ct_data.get("std_hu", 30.0)
        necrosis_r   = ct_data.get("necrosis_ratio", 0.0)

        # 낮은 HU = 저밀도 = 괴사/저산소 조직
        # 정상 연부조직 HU ~= 30~60; 괴사 ~= -20~10
        # 60.0 기준으로 강화 (HU=-35이면 score=(30+35)/60=1.08->clip=1.0)
        hu_score = np.clip((30.0 - mean_hu) / 60.0, 0, 1)
        # 높은 이질성 (std_hu > 60) -> 혼합 조직
        het_score = np.clip(std_hu / 70.0, 0, 1)
        return float(np.clip(
            hu_score * 0.55 + necrosis_r * 0.30 + het_score * 0.15, 0, 1
        ))

    # ── 핵심 계산 ──────────────────────────────────────────────────────────────

    def _compute_ddg(self, pathway: str, activation_index: float) -> float:
        """
        Boltzmann-역추론: 활성화 지수 → 유효 ΔG‡.

        ΔG‡_eff = ΔG‡_baseline - alpha · max_shift · activation_index

        낮은 ΔG‡ = 경로가 이미 '활성화 장벽'을 넘어 상시 켜진 상태.
        """
        baseline   = BASELINE_DDG[pathway]
        max_shift  = MAX_DELTA_DDG[pathway]
        shift      = self.alpha * max_shift * activation_index
        return float(np.clip(baseline - shift, 0.10, baseline))

    @staticmethod
    def _eyring_k(ddg_kcal: float) -> float:
        """
        Eyring 방정식 → 유사-속도상수 (정규화된 단위).
        k ∝ exp(-ΔG‡ / RT)
        """
        return float(np.exp(-ddg_kcal / RT_KCAL))

    # ── 공개 인터페이스 ────────────────────────────────────────────────────────

    def compute_profile(
        self,
        cellpose_summaries: list,
        ct_data: Optional[dict] = None,
        kras_allele: str = "G12D",
    ) -> PatientEnergyProfile:
        """
        환자 이미징 → PatientEnergyProfile.

        Parameters
        ----------
        cellpose_summaries : list of dict
            cell_annotator.py 출력 summaries 리스트.
        ct_data : dict, optional
            {"tumor_volume_cc": float, "mean_hu": float,
             "std_hu": float, "necrosis_ratio": float}
        kras_allele : str
            KRAS 변이 유형 — G12D/G12V/G12C/G13D/WT.
            G12D가 가장 강한 KRAS 활성 → proliferation_index 보정.
        """
        # ── 1. 중간 지표 계산 ─────────────────────────────────────────────────
        kras_boost = {"G12D": 0.20, "G12V": 0.15, "G12C": 0.10,
                      "G13D": 0.05, "WT": 0.0}.get(kras_allele, 0.0)

        pi = min(1.0, self._proliferation_index(cellpose_summaries) + kras_boost)
        cs = self._cell_size_index(cellpose_summaries)
        hy = self._hypoxia_score(ct_data)
        iv = self._invasion_score(cellpose_summaries)

        indices = {
            "proliferation_index": round(pi, 3),
            "cell_size_index":     round(cs, 3),
            "hypoxia_score":       round(hy, 3),
            "invasion_score":      round(iv, 3),
        }

        # ── 2. 경로별 ΔG‡ 계산 ───────────────────────────────────────────────
        pathway_map = {
            "KRAS_ERK":  pi,
            "PI3K_mTOR": cs,
            "HIF_VEGF":  hy,
            "RhoA_INV":  iv,
        }
        ddg = {pw: self._compute_ddg(pw, idx) for pw, idx in pathway_map.items()}
        k_pw = {pw: self._eyring_k(d) for pw, d in ddg.items()}

        # ── 3. 감도 순위 (k 내림차순) ─────────────────────────────────────────
        rank = sorted(k_pw, key=k_pw.get, reverse=True)

        # ── 4. 통합 유효 ΔG‡ (k 가중 평균) ───────────────────────────────────
        k_vals = np.array(list(k_pw.values()))
        d_vals = np.array([ddg[pw] for pw in k_pw])
        ddg_eff = float(np.average(d_vals, weights=k_vals))

        return PatientEnergyProfile(
            ddg_per_pathway=ddg,
            k_per_pathway={pw: round(v, 6) for pw, v in k_pw.items()},
            sensitivity_rank=rank,
            ddg_effective=round(ddg_eff, 4),
            imaging_indices=indices,
        )


# ── 자가 테스트 ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    mapper = ImagingToEnergyMapper()

    print("=" * 55)
    print("Case A: 고증식·고침윤 (KRAS G12D)")
    case_a = [
        {"total_cells": 520, "mean_area_um2": 310.0,
         "mean_circularity": 0.50, "irregular_count": 290,
         "normal_count": 120, "image_size": "1024x1024"},
    ] * 3
    ct_a = {"tumor_volume_cc": 45.0, "mean_hu": -8.0,
            "std_hu": 50.0, "necrosis_ratio": 0.20}
    pA = mapper.compute_profile(case_a, ct_a, kras_allele="G12D")
    print(pA)
    for pw, d in pA.ddg_per_pathway.items():
        print(f"  {pw:12s}: ddg={d:.3f} kcal/mol, k={pA.k_per_pathway[pw]:.4f}")

    print()
    print("=" * 55)
    print("Case B: 저증식·저산소 (KRAS WT)")
    case_b = [
        {"total_cells": 80, "mean_area_um2": 520.0,
         "mean_circularity": 0.75, "irregular_count": 10,
         "normal_count": 65, "image_size": "1024x1024"},
    ] * 3
    ct_b = {"tumor_volume_cc": 12.0, "mean_hu": -35.0,
            "std_hu": 70.0, "necrosis_ratio": 0.45}
    pB = mapper.compute_profile(case_b, ct_b, kras_allele="WT")
    print(pB)
    for pw, d in pB.ddg_per_pathway.items():
        print(f"  {pw:12s}: ddg={d:.3f} kcal/mol, k={pB.k_per_pathway[pw]:.4f}")

    print()
    print("=" * 55)
    print("Case C: CT 없음 (Cellpose만)")
    case_c = [
        {"total_cells": 200, "mean_area_um2": 400.0,
         "mean_circularity": 0.62, "irregular_count": 80,
         "normal_count": 100},
    ]
    pC = mapper.compute_profile(case_c, ct_data=None, kras_allele="G12V")
    print(pC)
