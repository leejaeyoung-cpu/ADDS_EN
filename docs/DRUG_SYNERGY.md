# KRAS-PrPc 약물 시너지 시스템

**기전 기반 항암제 시너지 예측 — KRAS-RPSA 시그널로솜 경로**

---

## 개요

ADDS 약물 시너지 시스템은 KRAS 돌연변이를 보유한 대장암 환자에서 PrPc (Prion Protein) 의 혈청 발현 패턴과 RPSA (Ribosomal Protein SA) 와의 상호작용을 기반으로 최적의 항암제 조합을 추천합니다.

**핵심 발견**: KRAS 돌연변이 → PrPc-RPSA 복합체 형성 → 라미닌 결합 증가 → 종양 침윤성 증가

---

## 1. PrPc 바이오마커의 과학적 근거

### 1.1 조직-혈청 역설 (The Paradox)

기존 연구자들은 CRC 조직에서 PRNP mRNA가 낮다는 이유로 PrPc를 무관한 마커로 판단했습니다. ADDS 팀은 이것이 **분석 수준의 오류**임을 밝혔습니다:

```
❌ 기존 가설:
   CRC 조직 PRNP mRNA ↓ → PrPc 무관

✅ ADDS v3.0 발견:
   CRC 조직 PRNP mRNA ↓  (세포 내 발현 억제)
   CRC 혈청 PrPc 단백질 ↑↑  (세포 외 방출 증가)
   
   설명: ADAM10/17 쉐딩 효소 → GPI-앵커 PrPc 절단 → 혈류 방출
```

### 1.2 TCGA 실데이터 검증

| 암종 | 샘플 수 | 혈청 PrPc 발현 | 조직 PRNP | 상관관계 |
|------|--------|--------------|----------|---------|
| COAD (대장) | 512 | ↑↑ | ↓ | 역비례 |
| READ (직장) | 166 | ↑↑ | ↓ | 역비례 |
| STAD (위) | 415 | ↑ | ↓ | 역비례 |
| BRCA (유방) | 892 | ↑ | ↓ | 역비례 |
| PAAD (췌장) | 300 | ↑↑ | ↓ | 역비례 |
| **합계** | **2,285** | | | |

---

## 2. KRAS-RPSA 시그널로솜 경로

### 2.1 분자 신호전달 경로

```
KRAS 돌연변이 (G12D / G12V / G13D)
         │
         ▼
   GTPase 상시 활성화
   (GDP → GTP 전환 불가)
         │
         ├──────────────────────────────────┐
         ▼                                  ▼
   RAF → MEK → ERK                    PI3K → AKT → mTOR
   (세포 증식)                         (세포 생존)
         │
         ▼
   PrPc-RPSA 복합체 형성
         │
         ├── 라미닌 결합 증가 (세포 침윤)
         ├── 인테그린 신호 활성화
         └── MMP 분비 증가 (기저막 분해)
         │
         ▼
   WNT/β-catenin 활성화
   (줄기세포 특성 획득)
```

### 2.2 치료 핵심 포인트

| 경로 노드 | 임상 표적 약물 | 효과 |
|---------|--------------|------|
| KRAS G12D | Sotorasib (Lumakras) | KRAS 직접 억제 |
| MEK | Trametinib | ERK 신호 차단 |
| EGFR (KRAS WT만) | Cetuximab / Panitumumab | 수용체 차단 |
| mTOR | Everolimus | PI3K/AKT 하류 차단 |
| PrPc-RPSA | ⚠️ 미개발 (연구 중) | 라미닌 결합 차단 |
| VEGF | Bevacizumab | 혈관신생 억제 |

---

## 3. 지식 베이스

### 3.1 문헌 수집 파이프라인

```python
# 자동화 논문 수집
python ct_literature_db.py --tier 1 --cancer_type CRC

# GPT-4 기반 지식 추출
python abstract_knowledge_extractor.py \
    --input literature_database/ \
    --output binding_database/

# TCGA 실데이터 다운로드
python tcga_prnp_real_download.py \
    --cancer_types COAD READ STAD BRCA PAAD \
    --output data/tcga/
```

### 3.2 지식 베이스 통계 (2026년 2월)

```
문헌 지식 베이스 v2.0
══════════════════════════════════════════
Tier 1 (100편): Nature, Cell, Science,
                Nature Medicine, Nature Cancer
Tier 2 (100편): JCO, Cancer Research,
                Gut, Annals of Oncology  
Tier 3: The Biology of Cancer (Weinberg)
──────────────────────────────────────────
총 논문 수:      311편
임상 샘플:       2,348개
등록 약물:       113종
작용 기전:       90개
바이오마커:      69개
시너지 조합:     59개 검증
══════════════════════════════════════════
```

### 3.3 핵심 약물 DB 구조

```json
{
  "drug": "FOLFOX",
  "components": ["Oxaliplatin", "5-FU", "Leucovorin"],
  "mechanism": "DNA crosslinking + TS inhibition",
  "target_mutations": ["KRAS_WT", "KRAS_MUT"],
  "biomarker_requirements": {
    "MSI_MSS": "모두 적용",
    "KRAS": "G12D, G12V, G13D 모두 적용"
  },
  "synergy_partners": ["Bevacizumab", "Cetuximab (WT only)"],
  "response_rate": {
    "KRAS_WT": 0.62,
    "KRAS_G12D": 0.51
  },
  "pk_half_life_hours": 10.5,
  "evidence_grade": "1A",
  "references": ["NEJM 2004", "JCO 2014", "ESMO 2022"]
}
```

---

## 4. Pritamab 예측 시스템

### 4.1 Pritamab 개요

Pritamab는 PrPc-RPSA 복합체를 표적으로 하는 **연구 단계 항체 약물**입니다.  
ADDS는 CT + Cellpose 데이터를 기반으로 Pritamab의 예상 반응률을 예측합니다.

### 4.2 반응 예측 모델

```python
def predict_pritamab_response(features: dict) -> float:
    """
    Pritamab 반응률 예측 (연구용 모델)
    
    입력: 14D 멀티모달 특징 벡터
    출력: 예상 반응률 [0-1]
    """
    # 핵심 예측 인자
    kras_weight = 0.35      # KRAS 돌연변이 여부
    prpc_weight = 0.30      # 혈청 PrPc 수준 (대리 지표)
    ki67_weight = 0.20      # Ki-67 증식 지수
    sphericity_weight = 0.15  # 종양 구형도
    
    score = (
        features["kras_mut_score"] * kras_weight +
        features["proliferation_score"] * prpc_weight +
        features["ki67_index"] / 100 * ki67_weight +
        (1 - features["sphericity"]) * sphericity_weight
    )
    
    return min(score, 0.95)  # 최대 95% 클램프
```

---

## 5. 시너지 매트릭스

```
약물 시너지 매트릭스 (ADDS 예측값)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
          │ Beva │ Cetux│ TMZ  │ Evero
──────────┼──────┼──────┼──────┼──────
FOLFOX    │ +++  │ ++(WT│  +   │ ++
FOLFIRI   │ +++  │ ++(WT│  +   │ ++
Sotorasib │  ++  │  N/A │ +++  │ +++
Irinotecan│  ++  │  +(WT│  +   │  +
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
+++: 높은 시너지  ++: 중간  +: 낮음  N/A: 금기
```

---

## 6. 임상 파일럿 로드맵

### PrPc 혈청 검사 연구 (v3.0)

```
파일럿 프로토콜 (IRB 준비 중)
━━━━━━━━━━━━━━━━━━━━━━━━━━━
디자인: 전향적 파일럿
코호트: N=100 (증례 50, 대조 50)
목표: 30% Stage I + 30% Stage II
────────────────────────────────
Month 1: IRB 제출 + 계정 설정
Month 2: 승인 + 사이트 활성화
Month 3: 등록 시작
────────────────────────────────
Go/No-Go 기준: AUC ≥ 0.75
성공 시: N=500 확장 연구
```

---

*참조: [PHARMACOKINETICS.md](PHARMACOKINETICS.md) | [CLINICAL_CDS.md](CLINICAL_CDS.md)*
