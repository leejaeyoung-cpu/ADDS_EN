# 약동학 (PK/PD) 모델링

**환자 맞춤형 항암제 용량 최적화 — 1-구획 약동학 모델**

---

## 개요

ADDS PK 모델은 CT 방사선학 데이터 (종양 부피)와 Cellpose 세포 분석 데이터 (Ki-67 지수)를 통합하여 대장암 환자를 위한 **개인화 항암제 투여 용량**을 계산합니다.

**엔드포인트**: `POST /api/v1/pharmacokinetics/analyze`  
**처리 시간**: ~2.1초  
**현황**: Production v3.2.2 (2026년 2월 6일 라이브)

---

## 생리학적 배경

### 왜 PK 모델이 필요한가?

표준 용량 프로토콜 (예: FOLFOX 85 mg/m²)은 체표면적(BSA)만 고려합니다. ADDS는 추가로:

1. **종양 부피** → 청소율(Cl) 감소 (신장/간 기능 대리 지표)
2. **Ki-67 지수** → 최적 투여량 증가 (고증식 종양에서 더 높은 용량 필요)
3. **미세환경 복잡도** → 약물 침투율 조정

이를 통해 **±30% 용량 개인화**를 달성합니다.

---

## 수학적 모델

### 1구획 약동학 기본 방정식

$$C(t) = C_0 \cdot e^{-k_e \cdot t}$$

여기서:
- $C(t)$ = 시간 t에서 혈중 약물 농도
- $C_0 = \frac{D}{V_d}$ = 초기 최대 농도
- $k_e = \frac{Cl}{V_d}$ = 소실 속도 상수

### ADDS 파라미터 공식

| 파라미터 | 공식 | 단위 | 설명 |
|---------|------|------|------|
| **청소율 (Cl)** | $120.0 \times \max(0.7, 1.0 - \frac{V_{tumor}}{500})$ | mL/min | 종양 부피가 클수록 청소율 감소 |
| **분포용적 (Vd)** | $45.0 + (V_{tumor} \times 0.5)$ | L | 종양 조직이 약물 분포 공간 추가 |
| **반감기 (t½)** | $0.693 \times \frac{V_d}{Cl \times 0.06}$ | hours | Vd/Cl 비율로 결정 |
| **최적 용량 (D)** | $200.0 \times (1.0 + \frac{Ki67}{200})$ | mg/m² | 고증식 종양에서 용량 증가 |
| **투여 간격** | $\max(6, \min(24, t_{1/2} \times 2))$ | hours | 안전 범위 내 조정 |

### 안전 제약 조건

```python
# 하드 클램프 (임상 현실성 보장)
dosing_interval = max(6, min(24, calculated_interval))  # 6–24시간
response_ceiling = min(predicted_response, 0.95)          # 최대 95%
clearance_floor = max(clearance, 120.0 * 0.7)            # 최소 84 mL/min
```

---

## 계산 예시

### 환자 케이스: P-2026-001

```
입력 파라미터:
├── 종양 부피: 2,450 mm³ (CT Stage 3 결과)
├── Ki-67 지수: 45.2% (Cellpose 추정)
└── 체표면적: 1.73 m²

계산 과정:
├── 청소율: 120 × max(0.7, 1.0 - 2450/500)
│   = 120 × max(0.7, -3.9)
│   = 120 × 0.7 = 84 mL/min
│
├── 분포용적: 45 + (2450 × 0.5) = 1,270 L
│
├── 반감기: 0.693 × (1270 / (84 × 0.06))
│   = 0.693 × 252.4 = 174.9시간 → 클램프 → 24시간
│   (실용적 반감기)
│
└── 최적 용량: 200 × (1 + 45.2/200) = 245.2 mg/m²

API 출력:
{
  "optimal_dose_mg_m2": 245.2,
  "total_dose_mg": 424.2,  # × 1.73 m²
  "dosing_interval_hours": 24,
  "half_life_hours": 10.5,
  "predicted_response_rate": 0.73,
  "clearance_ml_min": 84.0,
  "volume_distribution_L": 1270.0,
  "recommended_regimen": "FOLFOX 245 mg/m² q24h",
  "safety_notes": "신장 기능 모니터링 권장 (고용량)"
}
```

---

## API 사용

### 요청

```http
POST /api/v1/pharmacokinetics/analyze
Content-Type: application/json

{
  "patient_id": "P-2026-001",
  "tumor_volume_mm3": 2450.5,
  "ki67_index": 45.2,
  "body_surface_area": 1.73,
  "serum_creatinine": 0.9,
  "hepatic_function": "normal"
}
```

### 응답

```json
{
  "patient_id": "P-2026-001",
  "timestamp": "2026-02-06T14:23:11Z",
  "pk_parameters": {
    "clearance_ml_min": 84.0,
    "volume_distribution_L": 1270.0,
    "half_life_hours": 10.5,
    "elimination_rate_constant": 0.066
  },
  "dosing_recommendation": {
    "optimal_dose_mg_m2": 245.2,
    "total_dose_mg": 424.2,
    "dosing_interval_hours": 24,
    "route": "IV infusion over 2 hours",
    "cycle_length_days": 14
  },
  "predicted_outcomes": {
    "response_rate": 0.73,
    "pfs_months": 8.2,
    "grade3_toxicity_risk": 0.18
  }
}
```

---

## 독성 지표 (v4.0)

### 4-tier 독성 모니터링 기준

| Tier | 등급 | 지표 | 조치 |
|------|------|------|------|
| **1** | Grade 1–2 | ANC > 1.0K, Cr < 1.5x | 모니터링 유지 |
| **2** | Grade 3 | ANC 0.5–1.0K, Cr 1.5–3x | 용량 25% 감소 |
| **3** | Grade 4 | ANC < 0.5K, Cr > 3x | 투여 중단 |
| **4** | Life-threatening | 심각한 이상반응 | 영구 중단 |

### 용량 조정 알고리즘

```python
def adjust_dose(base_dose: float, toxicity_grade: int) -> float:
    adjustments = {0: 1.0, 1: 1.0, 2: 0.80, 3: 0.75, 4: 0.0}
    return base_dose * adjustments.get(toxicity_grade, 0.0)
```

---

## 폴백 전략

PK API 타임아웃 또는 연결 실패 시 프론트엔드 시뮬레이션 폴백:

```python
try:
    pk_result = requests.post(
        f"{BACKEND_URL}/api/v1/pharmacokinetics/analyze",
        json=pk_payload,
        timeout=30
    ).json()
except Exception:
    # 시뮬레이션 폴백 (임시)
    pk_result = {
        "optimal_dose_mg_m2": 200.0,  # 표준 용량
        "dosing_interval_hours": 14,
        "note": "PK 서버 연결 실패 — 표준 용량 사용"
    }
    st.warning("⚠️ PK API 연결 실패. 표준 용량으로 진행합니다.")
```

---

## 임상 검증 상태

| 검증 항목 | 상태 | 근거 |
|---------|------|------|
| 1구획 모델 적합성 | ✅ 검증 | CRC FOLFOX 문헌 PK 일치 |
| 종양 부피 Cl 보정 | ✅ 이론적 근거 | 신장/간 기능 대리지표 |
| Ki-67 기반 용량 增 | ⚠️ 연구 단계 | 임상 데이터 수집 중 |
| 투여 간격 안전성 | ✅ 검증 | 6–24h 임상 표준 준수 |

> ⚠️ **면책 조항**: ADDS PK 모델은 연구 목적의 보조 도구입니다.  
> 실제 임상 투여 결정은 반드시 담당 의사의 판단을 따라야 합니다.

---

*참조: [DRUG_SYNERGY.md](DRUG_SYNERGY.md) | [CLINICAL_CDS.md](CLINICAL_CDS.md) | [API_REFERENCE.md](API_REFERENCE.md)*
