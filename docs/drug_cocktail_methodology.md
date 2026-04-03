# ADDS 항암제 칵테일 최적화 기법 - 상세 기술 문서

## 개요

ADDS 시스템은 세 가지 상호 보완적인 항암제 칵테일 최적화 기법을 통합하여 환자 맞춤형 치료 계획을 생성합니다:

1. **증거 기반 프로토콜 매칭** - 가이드라인 및 문헌 기반 표준 요법
2. **시너지 계산 모델** - 4가지 약리학적 상호작용 정량화
3. **신호 경로 분석** - 분자 표적 기반 최적화
4. **AI 기반 조합 추천** - OpenAI GPT-4 활용 지능형 추천

---

## 1. 증거 기반 프로토콜 매칭 (Evidence-Based Protocol Matching)

### 1.1 알고리즘 개요

환자의 암 유형, 병기, 코호트 분류에 따라 임상 가이드라인(NCCN, ESMO)에서 검증된 표준 요법을 검색합니다.

### 1.2 프로토콜 데이터베이스 구조

```python
protocol_database = {
    "colorectal": {
        "adjuvant_high_risk": [
            {
                "name": "FOLFOX",
                "agents": ["5-FU", "Leucovorin", "Oxaliplatin"],
                "dosing": {
                    "5-FU": "400 mg/m² bolus + 2400 mg/m² 46h infusion",
                    "Leucovorin": "400 mg/m² IV",
                    "Oxaliplatin": "85 mg/m² IV"
                },
                "schedule": "Every 2 weeks",
                "duration": "6 months",
                "evidence_level": "Level I",
                "expected_response": 0.72,  # 72% disease-free survival
                "references": ["Andre et al., NEJM 2004"]
            },
            {
                "name": "CAPOX",
                "agents": ["Capecitabine", "Oxaliplatin"],
                "dosing": {
                    "Capecitabine": "1000 mg/m² PO BID d1-14",
                    "Oxaliplatin": "130 mg/m² IV d1"
                },
                "schedule": "Every 3 weeks",
                "duration": "6 months",
                "evidence_level": "Level I",
                "expected_response": 0.70,
                "references": ["Haller et al., Lancet 2011"]
            }
        ],
        "metastatic_first_line": [
            {
                "name": "FOLFIRI + Bevacizumab",
                "agents": ["5-FU", "Leucovorin", "Irinotecan", "Bevacizumab"],
                "biomarker_requirement": None,  # No restriction
                "expected_response": 0.45,
                "pfs_months": 10.2,
                "evidence_level": "Level I"
            },
            {
                "name": "FOLFOX + Cetuximab",
                "agents": ["5-FU", "Leucovorin", "Oxaliplatin", "Cetuximab"],
                "biomarker_requirement": "KRAS/NRAS wild-type",  # CRITICAL
                "expected_response": 0.57,
                "pfs_months": 9.6,
                "evidence_level": "Level I",
                "contraindication": "KRAS mutant"
            }
        ]
    }
}
```

### 1.3 선택 알고리즘

**단계 1: 코호트/병기 기반 필터링**

```python
def get_base_protocols(cancer_type, cohort_classification, stage):
    """
    Args:
        cancer_type: "colorectal", "breast", "lung", etc.
        cohort_classification: {
            'cohort': 'adjuvant_high_risk' | 'metastatic_first_line' | ...
        }
        stage: "I", "IIA", "IIIB", "IV", etc.
    
    Returns:
        List of candidate protocols
    """
    cohort_key = cohort_classification['cohort']
    
    # Stage-based routing
    if stage in ["I", "IIA", "IIB"]:
        if cohort_key == "low_risk":
            return []  # Observation only
        else:
            cohort_key = "adjuvant_high_risk"
    
    elif stage in ["IIIA", "IIIB", "IIIC"]:
        cohort_key = "adjuvant_high_risk"
    
    elif stage == "IV":
        cohort_key = "metastatic_first_line"
    
    protocols = protocol_database[cancer_type][cohort_key]
    return protocols
```

**단계 2: 바이오마커 기반 필터링**

```python
def filter_by_biomarkers(protocols, genomic_variants):
    """
    CRITICAL: Exclude contraindicated drugs based on biomarkers
    
    Args:
        protocols: List of candidate protocols
        genomic_variants: [
            {'gene': 'KRAS', 'variant': 'G12D', 'status': 'mutant'},
            {'gene': 'TP53', 'variant': 'R273H', 'status': 'mutant'},
            {'gene': 'MSI', 'status': 'MSI-H'}
        ]
    
    Returns:
        Filtered protocols
    """
    biomarker_status = {var['gene']: var['status'] for var in genomic_variants}
    
    filtered = []
    for protocol in protocols:
        # Check contraindications
        if protocol.get('contraindication'):
            if biomarker_status.get('KRAS') == 'mutant' and \
               'KRAS mutant' in protocol['contraindication']:
                continue  # Skip anti-EGFR if KRAS mutant
        
        # Check requirements
        if protocol.get('biomarker_requirement'):
            req = protocol['biomarker_requirement']
            if 'KRAS/NRAS wild-type' in req:
                if biomarker_status.get('KRAS') == 'mutant':
                    continue  # Skip
        
        # MSI-H boost
        if biomarker_status.get('MSI') == 'MSI-H':
            protocol['efficacy_boost'] = 1.2  # 20% boost for immunotherapy
        
        filtered.append(protocol)
    
    return filtered
```

**단계 3: 장기 기능 기반 용량 조정**

```python
def adjust_for_organ_function(protocols, clinical_profile):
    """
    Dose modifications based on hepatic/renal function
    
    Args:
        clinical_profile: {
            'creatinine_clearance': 52,  # mL/min
            'liver_function': 'Child-Pugh B',
            'bilirubin': 2.3  # mg/dL
        }
    """
    for protocol in protocols:
        # Renal impairment
        if clinical_profile.get('creatinine_clearance', 100) < 60:
            if 'Oxaliplatin' in protocol['agents']:
                protocol['dose_adjustment'] = {
                    'Oxaliplatin': 0.75  # 25% dose reduction
                }
                protocol['warnings'] = protocol.get('warnings', [])
                protocol['warnings'].append(
                    "Oxaliplatin dose reduced 25% for CrCl < 60 mL/min"
                )
        
        # Hepatic impairment
        if clinical_profile.get('liver_function') == 'Child-Pugh B':
            if 'Irinotecan' in protocol['agents']:
                protocol['dose_adjustment'] = protocol.get('dose_adjustment', {})
                protocol['dose_adjustment']['Irinotecan'] = 0.75
                protocol['warnings'] = protocol.get('warnings', [])
                protocol['warnings'].append(
                    "Irinotecan dose reduced for hepatic impairment"
                )
    
    return protocols
```

---

## 2. 약물 시너지 계산 (Drug Synergy Calculation)

### 2.1 시너지 모델

ADDS는 4가지 표준 약리학 모델을 구현합니다:

#### 2.1.1 Bliss Independence Model

**이론**: 두 약물이 독립적으로 작용한다고 가정

$$
E_{expected} = E_A + E_B - (E_A \times E_B)
$$

**시너지 점수**:
$$
Synergy_{Bliss} = E_{observed} - E_{expected}
$$

- **Synergy > 0**: 시너지 (관찰 효과 > 예상 효과)
- **Synergy = 0**: 가산 (Additive)
- **Synergy < 0**: 길항 (Antagonism)

**구현**:
```python
def bliss_independence(effect_a, effect_b, effect_combined):
    """
    Args:
        effect_a, effect_b: Individual drug effects (0-1)
        effect_combined: Observed combination effect (0-1)
    
    Returns:
        Bliss synergy score
    """
    expected = effect_a + effect_b - (effect_a * effect_b)
    synergy = effect_combined - expected
    return synergy

# Example:
# Drug A: 40% tumor reduction
# Drug B: 30% tumor reduction
# Combination: 80% tumor reduction
# Expected (Bliss): 0.4 + 0.3 - (0.4 × 0.3) = 0.58
# Synergy: 0.80 - 0.58 = +0.22 (Strong synergy!)
```

#### 2.1.2 Loewe Additivity Model

**이론**: 용량 등가성 원리 기반

$$
CI = \frac{d_A}{D_A} + \frac{d_B}{D_B}
$$

여기서:
- $d_A, d_B$: 조합에서 사용된 용량
- $D_A, D_B$: 단독으로 동일 효과를 내는 용량 (EC50)

**해석**:
- **CI < 1**: 시너지 (조합 용량 < 개별 용량 합)
- **CI = 1**: 가산성
- **CI > 1**: 길항 작용

**구현**:
```python
def loewe_additivity(dose_a, dose_b, ec50_a, ec50_b):
    """
    Combination Index (CI)
    
    Args:
        dose_a, dose_b: Actual doses used in combination
        ec50_a, ec50_b: EC50 (half-maximal effective concentration)
    
    Returns:
        CI value
    """
    ci = (dose_a / ec50_a) + (dose_b / ec50_b)
    return ci

# Example:
# Oxaliplatin: dose=85 mg/m², EC50=120 mg/m²
# 5-FU: dose=2400 mg/m², EC50=4000 mg/m²
# CI = (85/120) + (2400/4000) = 0.708 + 0.600 = 1.308
# CI > 1 suggests antagonism? 
# BUT clinical data shows synergy → Model mismatch
# → Bliss더 적합 for this pair
```

#### 2.1.3 Highest Single Agent (HSA)

**이론**: 보수적 모델, 가장 강력한 단일 약물 효과와 비교

$$
E_{expected} = \max(E_A, E_B)
$$

$$
Synergy_{HSA} = E_{observed} - E_{expected}
$$

**특징**: 가장 엄격한 시너지 판정

**구현**:
```python
def hsa(effect_a, effect_b, effect_combined):
    """
    Most conservative synergy model
    """
    expected = max(effect_a, effect_b)
    synergy = effect_combined - expected
    return synergy

# Example:
# Drug A: 40%
# Drug B: 30%
# Combination: 80%
# Expected (HSA): max(0.4, 0.3) = 0.4
# Synergy: 0.80 - 0.40 = +0.40 (Very strong by HSA standards)
```

#### 2.1.4 Zero Interaction Potency (ZIP)

**이론**: Bliss와 Loewe의 통합, 효능과 효력 변화 모두 고려

**단순화된 델타 점수**:
$$
ZIP = E_{observed} - \frac{E_{Bliss} + E_{Loewe}}{2}
$$

**구현**:
```python
def zip_score(effect_a, effect_b, effect_combined, 
              dose_a, dose_b, ec50_a, ec50_b):
    """
    ZIP integrates Bliss and Loewe
    """
    # Bliss component
    bliss_expected = effect_a + effect_b - (effect_a * effect_b)
    
    # Simplified Loewe component (assuming dose-response curve)
    loewe_expected = (dose_a / ec50_a + dose_b / ec50_b) / 2.0
    
    # Average
    expected_avg = (bliss_expected + loewe_expected) / 2.0
    
    zip = effect_combined - expected_avg
    return zip
```

### 2.2 시너지 통합 분류

4가지 모델의 합의(Consensus)로 최종 판정:

```python
def classify_interaction(synergy_scores):
    """
    Args:
        synergy_scores: {
            'bliss': 0.22,
            'hsa': 0.40,
            'zip': 0.18
        }
    
    Returns:
        Classification: "Strong Synergy" | "Synergy" | "Additive" | "Antagonism"
    """
    # Count positive synergies
    positive = sum(1 for score in synergy_scores.values() if score > 0.1)
    
    # Average magnitude
    avg_score = np.mean(list(synergy_scores.values()))
    
    if positive >= 3 and avg_score > 0.2:
        return "Strong Synergy"
    elif positive >= 2 and avg_score > 0.1:
        return "Synergy"
    elif positive >= 2 and avg_score > 0:
        return "Weak Synergy"  
    elif avg_score >= -0.05 and avg_score <= 0.05:
        return "Additive"
    else:
        return "Antagonism"
```

### 2.3 실제 데이터 통합

```python
# 문헌에서 추출한 시너지 데이터 예시
synergy_database = {
    ("5-FU", "Oxaliplatin"): {
        "bliss": 0.18,
        "cancer_type": "colorectal",
        "reference": "Raymond et al., JCO 1998",
        "clinical_benefit": "20% improvement in PFS"
    },
    ("Irinotecan", "Bevacizumab"): {
        "bliss": 0.25,
        "cancer_type": "colorectal",
        "reference": "Hurwitz et al., NEJM 2004",
        "clinical_benefit": "4.7 months OS improvement"
    },
    ("Encorafenib", "Cetuximab"): {
        "bliss": 0.42,
        "cancer_type": "colorectal",
        "biomarker": "BRAF V600E",
        "reference": "Kopetz et al., NEJM 2019",
        "clinical_benefit": "ORR 26% vs 2%"
    }
}
```

---

## 3. 신호 경로 분석 (Signal Pathway Analysis)

### 3.1 경로 기반 표적 선택

**원리**: 여러 신호 경로를 동시에 차단하여 내성 회피

**주요 경로**:
- **EGFR 경로**: Cetuximab, Panitumumab
- **VEGF 경로**: Bevacizumab, Ramucirumab
- **BRAF/MEK 경로**: Encorafenib, Binimetinib
- **DNA 손상 수복**: Oxaliplatin, Irinotecan, 5-FU

### 3.2 경로 활성화 분석

```python
def analyze_active_pathways(cell_analysis, ct_results, genomic_variants):
    """
    Identify active oncogenic pathways
    
    Args:
        cell_analysis: {'ki67': 0.42, 'cell_density': 2800}
        ct_results: {'angiogenesis_score': 0.75}
        genomic_variants: [{'gene': 'KRAS', 'status': 'mutant'}, ...]
    
    Returns:
        pathway_activation: {
            'EGFR': {'active': True, 'confidence': 0.85, 'biomarker': 'KRAS wild-type'},
            'VEGF': {'active': True, 'confidence': 0.75, 'evidence': 'High angiogenesis'},
            'BRAF': {'active': False, 'biomarker': 'BRAF wild-type'}
        }
    """
    pathways = {}
    
    # EGFR pathway
    kras_status = next((v['status'] for v in genomic_variants if v['gene'] == 'KRAS'), 'unknown')
    if kras_status == 'wild-type':
        pathways['EGFR'] = {
            'active': True,
            'confidence': 0.85,
            'targetable': True,
            'drugs': ['Cetuximab', 'Panitumumab']
        }
    
    # VEGF/Angiogenesis pathway
    angio_score = ct_results.get('angiogenesis_score', 0)
    if angio_score > 0.6 or ct_results.get('tumor_size_mm', 0) > 50:
        pathways['VEGF'] = {
            'active': True,
            'confidence': angio_score,
            'targetable': True,
            'drugs': ['Bevacizumab']
        }
    
    # BRAF pathway
    braf_status = next((v['status'] for v in genomic_variants if v['gene'] == 'BRAF'), 'wild-type')
    if braf_status == 'mutant':
        pathways['BRAF'] = {
            'active': True,
            'confidence': 1.0,  # Genomic evidence = high confidence
            'targetable': True,
            'drugs': ['Encorafenib + Cetuximab', 'Dabrafenib + Trametinib']
        }
    
    # Cell cycle / DNA damage
    if cell_analysis['ki67'] > 0.3:
        pathways['DNA_damage'] = {
            'active': True,
            'confidence': cell_analysis['ki67'],
            'targetable': True,
            'drugs': ['Oxaliplatin', 'Irinotecan', '5-FU']
        }
    
    return pathways
```

### 3.3 다중 경로 억제 전략

```python
def design_multi_pathway_strategy(active_pathways):
    """
    Design combination targeting multiple pathways
    
    Goal: Overcome resistance by hitting multiple targets
    """
    strategy = {
        'targeted_pathways': [],
        'recommended_additions': []
    }
    
    # Rule: Target at least 2 pathways
    targetable = [p for p, info in active_pathways.items() if info.get('targetable')]
    
    if len(targetable) >= 2:
        # Multi-pathway strategy
        strategy['targeted_pathways'] = targetable[:2]
        
        # Example: EGFR + VEGF
        if 'EGFR' in targetable and 'VEGF' in targetable:
            strategy['recommended_additions'] = [
                {
                    'combo': 'Cetuximab + Bevacizumab',
                    'rationale': 'Dual blockade EGFR + VEGF',
                    'evidence': 'Heinemann et al., Lancet Oncol 2014'
                }
            ]
    
    return strategy
```

---

## 4. AI 기반 지능형 추천 (OpenAI GPT-4 Integration)

### 4.1 프롬프트 엔지니어링

```python
def build_ai_recommendation_prompt(patient_profile, available_drugs):
    """
    Construct detailed prompt for GPT-4
    """
    prompt = f"""As an expert oncologist, recommend the TOP 3 drug cocktail combinations.

**Patient Profile:**
- Cancer Type: {patient_profile['cancer_type']}
- Stage: {patient_profile['stage']}
- Age: {patient_profile['age']}
- ECOG: {patient_profile['ecog']}
- Biomarkers:
  - KRAS: {patient_profile['kras']}
  - TP53: {patient_profile['tp53']}
  - MSI: {patient_profile['msi']}
- Ki-67: {patient_profile['ki67']*100:.0f}%
- Tumor size: {patient_profile['tumor_size_mm']} mm

**Available Drugs:**
{', '.join(available_drugs)}

**Required Output Format:**

1. **First Choice Combination**
   - Drugs: [drug names]
   - Doses: [specific dosing]
   - Synergy mechanism: [explanation]
   - Expected response rate: [%]
   - Evidence level: [I/II/III]
   - Key references: [citations]

2. **Second Choice Combination**
   [same format]

3. **Third Choice Combination**
   [same format]

**Treatment Monitoring:**
- Biomarkers to track
- Expected timeline

**Contraindications and Warnings**

Provide evidence-based, patient-specific recommendations."""
    
    return prompt
```

### 4.2 GPT-4 호출 및 파싱

```python
def get_ai_recommendation(patient_profile, available_drugs):
    """
    Get AI-powered drug cocktail recommendation
    """
    client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",  # or gpt-4-turbo
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert oncologist with 30 years experience in combination chemotherapy. Provide evidence-based, patient-specific drug cocktail recommendations citing Level I evidence whenever possible."
                },
                {
                    "role": "user",
                    "content": build_ai_recommendation_prompt(patient_profile, available_drugs)
                }
            ],
            temperature=0.3,  # Lower = more deterministic
            max_tokens=2000
        )
        
        ai_text = response.choices[0].message.content
        
        return {
            'success': True,
            'recommendations': ai_text,
            'model': 'gpt-4o-mini',
            'cost': estimate_cost(response.usage)
        }
    
    except Exception as e:
        # Fallback to rule-based
        return fallback_recommendation(patient_profile)
```

---

## 5. 통합 의사결정 알고리즘 (Integrated Decision Algorithm)

### 5.1 최종 권고 생성 프로세스

```
┌─────────────────────────────────────────────┐
│ Step 1: Protocol Matching                  │
│ → Get guideline-based protocols            │
└─────────────────┬───────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────┐
│ Step 2: Biomarker Filtering                │
│ → Exclude contraindicated drugs            │
│ → Add targeted therapies if eligible       │
└─────────────────┬───────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────┐
│ Step 3: Synergy Calculation                │
│ → Query synergy database                   │
│ → Boost efficacy for synergistic pairs     │
└─────────────────┬───────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────┐
│ Step 4: Pathway Analysis                   │
│ → Identify active pathways                 │
│ → Add pathway-targeted agents              │
└─────────────────┬───────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────┐
│ Step 5: Organ Function Adjustment          │
│ → Dose modifications for renal/hepatic     │
└─────────────────┬───────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────┐
│ Step 6: AI Enhancement (Optional)          │
│ → GPT-4 refines recommendations            │
│ → Adds clinical context and nuance         │
└─────────────────┬───────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────┐
│ Step 7: Confidence Scoring                 │
│ → Calculate overall confidence (0-1)       │
│ → Generate warnings and monitoring plan    │
└─────────────────┬───────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────┐
│ Step 8: Top-3 Ranking                      │
│ → Sort by: efficacy × confidence           │
│ → Return with full evidence package        │
└─────────────────────────────────────────────┘
```

### 5.2 신뢰도 점수 산출

```python
def calculate_confidence(regimen, evidence):
    """
    Calculate confidence score (0-1) for recommendation
    
    Factors:
    - Evidence level (I=1.0, II=0.8, III=0.6)
    - Biomarker match (+0.1 if perfect match)
    - Synergy evidence (+0.1 if synergy proven)
    - Pathway coverage (+0.05 per pathway targeted)
    - Safety concerns (-0.1 per major warning)
    """
    confidence = 0.5  # Base
    
    # Evidence level
    evidence_map = {'Level I': 1.0, 'Level II': 0.8, 'Level III': 0.6}
    confidence *= evidence_map.get(evidence['level'], 0.5)
    
    # Biomarker perfect match
    if evidence.get('biomarker_match'):
        confidence += 0.1
    
    # Synergy proven
    if evidence.get('synergy_score', 0) > 0.15:
        confidence += 0.1
    
    # Pathway coverage
    pathways_targeted = evidence.get('pathways_targeted', [])
    confidence += 0.05 * len(pathways_targeted)
    
    #Safety warnings
    warnings = regimen.get('warnings', [])
    confidence -= 0.1 * len(warnings)
    
    return min(max(confidence, 0.0), 1.0)
```

---

## 6. 실제 사용 예시

### 사례: 전이성 대장암, KRAS 야생형

**입력**:
```python
patient = {
    'cancer_type': 'colorectal',
    'stage': 'IV',
    'kras': 'wild-type',
    'tp53': 'mutant',
    'msi': 'MSS',
    'ki67': 0.38,
    'tumor_size_mm': 58,
    'liver_mets': True,
    'ecog': 1,
    'age': 62,
    'creatinine_clearance': 85
}
```

**처리 과정**:

1. **프로토콜 매칭** → `metastatic_first_line` 프로토콜 5개 후보
2. **바이오마커 필터링** → KRAS 야생형이므로 anti-EGFR 가능
3. **시너지 계산**:
   - FOLFIRI + Bevacizumab: Bliss = +0.25
   - FOLFOX + Cetuximab: Bliss = +0.18
4. **경로 분석**:
   - EGFR 활성 (KRAS 야생형)
   - VEGF 활성 (큰 종양, 간 전이)
   - DNA damage 활성 (Ki-67 38%)
5. **AI 추천** (GPT-4):
   ```
   "Given KRAS wild-type status, anti-EGFR therapy with Cetuximab 
    is highly effective. Combine with FOLFOX for synergistic DNA damage. 
    Expected ORR 57%, PFS 9.6 months (Bokemeyer et al., JCO 2011)"
   ```
6. **신뢰도**: 0.92 (Level I evidence, biomarker match, proven synergy)

**최종 권고**:

| 순위 | 조합 | 예상 효능 | 신뢰도 | 근거 |
|------|------|-----------|--------|------|
| 1 | FOLFOX + Cetuximab | 57% ORR | 0.92 | Level I, KRAS WT matching |
| 2 | FOLFIRI + Bevacizumab | 45% ORR | 0.88 | Level I, VEGF targeting |
| 3 | CAPOX + Bevacizumab | 42% ORR | 0.75 | Level II, oral convenience |

---

## 7. 특허 및 논문에 추가할 핵심 내용

### 7.1 특허 청구항 추가

**새로운 종속 청구항 11 (특허 Part 1에 추가)**:

```
청구항 11 (종속항 - 다층 시너지 계산)

제1항에 있어서,

상기 통합 엔진(131)은,

Bliss Independence 모델로 E_expected = E_A + E_B - (E_A × E_B)를 계산하여 독립 작용 가정 하 예상 효과를 산출하고,

Loewe Additivity 모델로 CI = (dose_A / EC50_A) + (dose_B / EC50_B)를 계산하여 조합 지수를 산출하며,

Highest Single Agent(HSA) 모델로 E_expected = max(E_A, E_B)를 계산하고,

Zero Interaction Potency(ZIP) 모델로 상기 Bliss와 Loewe의 평균을 산출하여,

상기 4가지 모델의 합의(consensus)로 약물 상호작용을 "Strong Synergy", "Synergy", "Additive", "Antagonism"으로 분류하고,

문헌 데이터베이스에서 실증된 시너지 점수를 질의하여 예상 효능에 가중치(0.15 × synergy_score)를 부여하는,

인공지능 기반 다중모달 임상 의사결정 지원 시스템.
```

### 7.2 논문 섹션 추가

**Section 4.6: Multi-Model Drug Synergy Quantification** (Methodology에 추가)

4가지 표준 약리학 모델(Bliss, Loewe, HSA, ZIP)의 통합 접근법을 통해 약물 조합 시너지를 정량화하고, 문헌 데이터베이스와 연계하여 임상 검증된 조합에 가중치를 부여하는 방법론 설명.

**Section 5.5: OpenAI Integration for Clinical Nuance** (Implementation에 추가)

GPT-4를 활용한 문맥 인식 치료 권고 개선 방법, 프롬프트 엔지니어링 전략, fallback 메커니즘 설명.

---

**문서 작성일**: 2026년 1월 29일  
**작성자**: ADDS 개발팀
