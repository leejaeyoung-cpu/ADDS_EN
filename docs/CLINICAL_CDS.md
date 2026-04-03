# 임상 의사결정 지원 시스템 (CDS)

**이중 추론 엔진 기반 정밀 종양학 의사결정 지원**

---

## 개요

ADDS CDS는 의사의 임상 소견, AI 기반 영상 분석, 세포 병리학, 약동학을 통합하여 최종 항암제 칵테일 추천을 제공하는 **6단계 동적 추론 파이프라인**입니다.

**핵심 원칙**: 의사 소견서가 **1순위 프롬프트(Primary Prompt)**로 모든 AI 추론의 그라운드 트루스 역할을 수행합니다.

---

## 6단계 동적 추론 파이프라인

### Step 0: RAG 기반 임상 컨텍스트 분석

```
의사 소견서 (Doctor's Notes)
         │
         ▼
  RAG (Retrieval-Augmented Generation)
         │
    ┌────┴────────────────────────────────┐
    ▼                                     ▼
증상 추출                            병력 추출
(통증/출혈/체중감소 etc.)          (기저질환/수술력 etc.)
    │                                     │
    └──────────────────┬──────────────────┘
                       ▼
               임상 컨텍스트 JSON
               (Step 4/5의 Primary Prompt)
```

**추출 지표**:
- 주요 증상 및 발현 시기
- 관련 병력 및 수술 이력
- 환자 선호도 (독성 우려, 라이프스타일)
- 이전 치료 반응력

### Step 1: CT 분석 (Live API Integration)

```python
# Multipart Async Upload
response = await upload_ct_async(
    endpoint="/api/v1/ct/analyze",
    dicom_files=st.session_state["ct_files"],
    patient_id=patient_id
)

# 결과 스트림
ct_result = {
    "tumors_detected": 2,
    "radiomics": {...},    # 100+ 특징
    "analyzed_images": [...],  # Base64 시각화
    "tnm_stage": "T3N1",
    "confidence": 0.94
}
```

**검증 로그**:
```
✅ CT Files Count (Frontend): 3개
✅ CT Files Count (Backend received): 3개
✅ Tumor Detection: 2 candidates found
✅ HU Range: [62.3, 118.7] — 정상 범위
```

### Step 2: 세포 분석 (조건부 실행)

```python
# 현미경 이미지 있을 때만 실행
if "cell_images" in st.session_state and len(st.session_state["cell_images"]) > 0:
    cell_result = cellpose_pipeline.analyze_batch(
        images=st.session_state["cell_images"]
    )
else:
    st.info("ℹ️ 현미경 이미지 없음 — 세포 분석 건너뜀")
    cell_result = default_cell_features()
```

### Step 3: 약동학 최적화

```python
# CT + Cellpose 데이터 → PK 파라미터
pk_input = {
    "tumor_volume_mm3": ct_result["tumor_volume"],
    "ki67_index": cell_result["ki67_index"],
    "body_surface_area": patient.bsa
}
pk_result = await pk_api.analyze(pk_input)
```

### Step 4: ADDS 경로 기반 추론

```python
adds_prompt = {
    "clinical_context": rag_context,        # Step 0 출력 (1순위)
    "ct_findings": ct_result["radiomics"],   # Step 1 출력
    "cell_features": cell_result,            # Step 2 출력
    "pk_params": pk_result,                  # Step 3 출력
    "pathway_db": knowledge_base             # KRAS-PrPc 지식 베이스
}

adds_result = {
    "recommended_regimen": "FOLFOX + Bevacizumab",
    "pathway_rationale": "KRAS G12D → RAF/MEK/ERK 상시활성 → 옥살리플라틴 DNA 교차결합 표적",
    "synergy_score": 8.7,
    "biomarker_match": {
        "KRAS": "G12D (EGFR 억제제 제외)",
        "MSS": "면역관문억제제 제한적",
        "Ki-67 >40%": "집중 요법 권장"
    }
}
```

### Step 5: OpenAI GPT-4 임상 추론

```python
openai_prompt = f"""
[임상 맥락 - 최우선 참조]
{rag_context}

[CT 소견]
{ct_result["summary"]}

[세포 분석]
{cell_result["summary"]}

[PK 최적화]
{pk_result["recommendation"]}

위 정보를 종합하여 CRC 환자를 위한 임상적 항암 치료 계획을 제시하세요.
의사의 소견서에 기술된 증상과 환자 선호도를 반드시 고려하세요.
"""

openai_result = openai.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": openai_prompt}]
)
```

### Step 6: 교차 검증 (Cross-Validation)

```
소견서 설명 ←───────────→ CT 소견
"S자 결장 통증"        "Z=370mm Sigmoid 병변"
      └────────────── ✅ 일치
      
소견서 설명 ←───────────→ 병리 소견
"KRAS 양성"            "KRAS G12D 확인"
      └────────────── ✅ 일치

최종 일치도: 94.2%
```

---

## 최종 항암제 추천 생성

### 추천 구조

```json
{
  "primary_recommendation": {
    "regimen": "FOLFOX + Bevacizumab",
    "rationale": "KRAS G12D 돌연변이 → EGFR 차단제 제외\nDNA 교차결합 + 혈관신생 이중 차단",
    "dosing": {
      "oxaliplatin": "85 mg/m² (조정: 104.2 mg/m²)",
      "5fu": "400 mg/m² bolus + 2400 mg/m²/46h",
      "bevacizumab": "5 mg/kg q2w",
      "leucovorin": "400 mg/m²"
    },
    "cycle": "14일 주기, 12주기"
  },
  "secondary_recommendation": {
    "regimen": "FOLFIRI + Bevacizumab",
    "indication": "FOLFOX 내성 발생 시 2차 치료"
  },
  "predicted_outcomes": {
    "orr_percent": 73.0,
    "pfs_months": 8.2,
    "os_months": 18.5,
    "grade3_toxicity_risk": 0.18
  }
}
```

---

## 교차 검증 & 보고서

### 이중 보고서 생성

```python
# 보고서 생성 (자동화)
report_generator.create_physician_report(
    patient=patient,
    ct=ct_result,
    cell=cell_result,
    pk=pk_result,
    adds=adds_result,
    openai=openai_result,
    format="pdf",
    level="technical"  # 의사용: 원시 데이터 포함
)

report_generator.create_patient_report(
    patient=patient,
    recommendation=final_recommendation,
    format="pdf",
    level="simplified"  # 환자용: 이해하기 쉬운 설명
)
```

### 생성 보고서 예시

**의사 기술 보고서 내용**:
- CT 방사선학 원시 측정값 (100+ 특징)
- 14D 특징 벡터 전체
- ADDS 경로 점수 상세
- PK 파라미터 계산 과정
- 교차 검증 일치도 매트릭스

**환자 가이드 내용**:
- 치료 방법 (쉬운 말로)
- 예상 부작용 및 대처법
- 치료 일정
- 다음 방문 일정

---

## 시스템 텔레메트리

```python
# 실시간 상태 표시 (Streamlit)
col1, col2, col3 = st.columns(3)
col1.metric("백엔드 API", "v3.0.0 ✅")
col2.metric("GPU 사용률", f"{gpu_util}%")
col3.metric("처리 속도", f"{elapsed:.1f}초")

# 디버그 정보 (개발용)
with st.expander("🔍 API 응답 상세"):
    st.json({
        "ct_files_received": len(ct_files),
        "tumors_detected": ct_result["count"],
        "backend_url": BACKEND_URL,
        "session_state_keys": list(st.session_state.keys())
    })
```

---

## 실패 복원력 (Fallback Resilience)

### 라이브러리 미설치 보호

```python
# 보호된 임포트 패턴
try:
    import SimpleITK as sitk
    SITK_AVAILABLE = True
    SITK_BSPLINE = sitk.sitkBSpline
except ImportError:
    SITK_AVAILABLE = False
    SITK_BSPLINE = 3  # 직접 정수 상수

class RadiomicsExtractor:
    def extract(self, volume):
        if not SITK_AVAILABLE:
            # 기본 특징 폴백 (볼륨 + 중심점)
            return self._simplified_features(volume)
        return self._full_radiomics(volume)
```

### 4단계 폴백 계층

```
1. 프로덕션 엔진 (VerifiedCTDetector 98.65%)
   ↓ 실패시
2. 단순 HU 임계값 검출기
   ↓ 실패시
3. 기하학적 중심점 기반 폴백
   ↓ 실패시
4. 플레이스홀더 (분석 불가 표시)
```

---

*참조: [CT_PIPELINE.md](CT_PIPELINE.md) | [PHARMACOKINETICS.md](PHARMACOKINETICS.md) | [API_REFERENCE.md](API_REFERENCE.md)*
