# 특허출원서 Part 2 - 발명의 상세한 설명

## 【발명을 실시하기 위한 구체적인 내용】

### 1. 실시예

이하, 첨부된 도면을 참조하여 본 발명의 바람직한 실시예를 상세히 설명한다. 다만, 본 발명의 기술 사상은 설명되는 일부 실시예에 한정되는 것이 아니라 서로 다른 다양한 형태로 구현될 수 있고, 본 발명의 기술 사상 범위 내에서라면, 실시예들 간 그 구성 요소들 중 하나 이상을 선택적으로 결합, 치환하여 사용할 수 있다.

### 2. 시스템 전체 구성 (도 1 참조)

도 1은 본 발명의 일 실시예에 따른 인공지능 기반 다중모달 임상 의사결정 지원 시스템(100)의 전체 아키텍처를 나타낸 블록도이다.

시스템(100)은 크게 데이터 획득 계층(110), 처리 계층(120), 통합 계층(130), 및 표현 계층(140)으로 구성된다.

**데이터 획득 계층(110)**은 다음을 포함한다:
- 병리 이미지 입력부(111): TIFF, PNG 형식의 세포 이미지를 수신 (해상도 2048×2048~4096×4096 픽셀)
- CT 영상 입력부(112): DICOM 형식의 CT 스캔을 수신 (512×512×N 슬라이스)
- 임상 데이터 입력부(113): 환자 인구통계학적 정보, 유전자 바이오마커(KRAS, TP53, MSI), 검사 결과(간/신기능), ECOG 수행 상태를 수신

**처리 계층(120)**은:
- 세포 분석 모듈(121): Cellpose 기반 세포 분할 및 특징 추출
- CT 검출 모듈(122): 해부학적 분할, 종양 후보 검출, TNM 병기 결정

**통합 계층(130)**은:
- 통합 엔진(131): 다중모달 데이터 융합, 병기 결정, 위험도 산출
- 설명 가능 AI 모듈(132): LIME, Grad-CAM, 반사실적 분석
- 능동 학습 모듈(133): 이중모드 획득 함수 기반 약물 조합 최적화

**표현 계층(140)**은:
- 의사 인터페이스(141): 상세 분석 결과 및 치료 권고 제공
- 환자 인터페이스(142): 쉬운 언어로 진단 및 치료 설명
- API 계층(143): RESTful API 엔드포인트 제공

### 3. 세포 분석 모듈의 상세 구성 (도 2 참조)

도 2는 세포 분석 모듈(121)의 상세 처리 흐름을 나타낸 순서도이다.

#### 3.1 전처리부(121a)

입력 이미지 I ∈ ℝ^(H×W×C)에 대해 다음을 수행한다:

```
(1) 정규화: I_norm = (I - μ) / σ, 여기서 μ=0, σ=1
(2) 대비 향상: CLAHE(Contrast Limited Adaptive Histogram Equalization) 적용
```

일 실시예에서, CLAHE는 타일 크기 8×8, 클립 한계 2.0으로 수행된다.

```
(3) 잡음 제거: 가우시안 필터(σ=1.0) 적용
```

#### 3.2 Cellpose 분할부(121b)

전처리된 이미지에 대해 Cellpose 신경망을 적용한다:

**네트워크 구조:**
- 인코더: ResNet 스타일, 스킵 연결
- 디코더: 업샘플링 + 인코더 특징 맵 결합
- 파라미터: 17M (cyto2 모델)

**출력:**
- 수평 흐름: F_x(p) (픽셀 p에서 세포 중심으로의 x 방향 벡터)
- 수직 흐름: F_y(p) (픽셀 p에서 세포 중심으로의 y 방향 벡터)
- 세포 확률: P_cell(p) (픽셀 p가 세포에 속할 확률)

**마스크 재구성:**
```
For each pixel p:
  1. p_current = p
  2. While not converged:
       p_current = p_current + F(p_current)
  3. cell_id[p] = cluster at p_current
```

일 실시예에서, flow_threshold=0.4, cellprob_threshold=0.0으로 설정된다.

#### 3.3 특징 추출부(121c)

분할된 각 세포 i=1..N에 대해 다음 특징을 추출한다:

**형태학적 특징 (6개):**
- 면적: A_i = Σ(p ∈ R_i) 1 [μm²]
- 둘레: P_i = contour_length(R_i) [μm]
- 원형도: C_i = 4πA_i / P_i² (0.0 ~ 1.0, 1.0 = 완전한 원)
- 편심률: E_i = sqrt(1 - b²/a²), a,b는 타원 장축/단축
- 견고성: S_i = A_i / A_convex_i
- 방향: θ_i = atan2(중심 모멘트) [라디안]

**강도 특징 (7개):**
- 평균 강도: μ_I = (1/|R_i|) Σ(p ∈ R_i) I(p)
- 표준편차: σ_I
- 분위수: Q_25, Q_50 (중앙값), Q_75
- 범위: max(I) - min(I)
- 적분 밀도: Σ(p ∈ R_i) I(p)

**텍스처 특징 (4개, GLCM):**

Gray-Level Co-occurrence Matrix P(i,j)를 오프셋 [(1,0), (0,1), (1,1), (1,-1)]에서 계산:

- 대비: Σ(i,j) (i-j)² × P(i,j)
- 상관: Pearson 상관 계수
- 에너지: Σ(i,j) P(i,j)²
- 동질성: Σ(i,j) P(i,j) / (1+|i-j|)

**공간 특징 (5개):**
- 무게중심: (x_c, y_c)
- 세포 밀도: ρ = N / A_total [cells/mm²]
- 최근접 이웃 거리: d_nn = min_j(||c_i - c_j||) [μm]
- Clark-Evans 지수: R = d̄_obs / d̄_random
- Ripley의 K-함수: K(r) = (A/N²) Σ(i≠j) 𝟙(d_ij < r)

#### 3.4 Ki-67 산출부(121d)

증식 지수는 다음과 같이 계산된다:

```
N_proliferating = count(cells where μ_I > I_80th_percentile)
Ki67_index = N_proliferating / N_total × 100%
```

일 실시예에서, 80th percentile 이상의 강도를 가진 핵을 증식 중인 것으로 분류한다.

### 4. CT 검출 모듈의 상세 구성 (도 3 참조)

도 3은 CT 검출 모듈(122)의 다중 임계값 종양 후보 검출 과정을 나타낸 도면이다.

#### 4.1 DICOM 처리부(122a)

DICOM 파일로부터 다음을 추출한다:

```
HU = pixel_value × RescaleSlope + RescaleIntercept
pixel_spacing = [PixelSpacing[0], PixelSpacing[1], SliceThickness]
```

일 실시예에서, RescaleSlope=1.0, RescaleIntercept=-1024이다.

#### 4.2 윈도잉부(122b)

소프트 조직 가시화를 위한 윈도잉 수행:

```
W = 400 HU (window width)
L = 40 HU (window level)
display_range = [L - W/2, L + W/2] = [-160, 240] HU
```

#### 4.3 해부학적 분할부(122c)

TotalSegmentator를 사용하여 104개 해부학적 구조 식별:

```
totalsegmentator -i ct_scan.nii.gz -o segmentations/ --fast
```

일 실시예에서, 대장암 검출을 위해 다음 구조에 집중한다:
- 대장(상행, 횡행, 하행, S상)
- 직장
- 국소 림프절
- 간(전이 선별용)

#### 4.4 종양 후보 검출부(122d)

다중 임계값 접근법:

```python
candidates = []
for threshold in range(-50, 200, 10):  # HU 범위, 10 HU 간격
    binary_mask = (hu_slice > threshold).astype(int)
    labeled_regions = connected_component_labeling(binary_mask)
    
    for region in labeled_regions:
        area_mm2 = region.area × pixel_spacing[0] × pixel_spacing[1]
        
        # 크기 필터: 50 mm² ~ 5000 mm²
        if 50 < area_mm2 < 5000:
            candidates.append({
                'centroid': region.centroid,
                'area': area_mm2,
                'mean_hu': region.mean_intensity,
                'circularity': 4π × area / perimeter²,
                'threshold': threshold
            })
```

#### 4.5 신뢰도 스코어링부(122e)

각 후보 c에 대해 신뢰도 점수 계산:

```python
def calculate_confidence(candidate, anatomy_mask):
    score = 0.0
    
    # (1) 크기 점수 (50-500 mm² 범위 선호)
    if 50 < candidate.area < 500:
        size_score = 0.3
    elif 500 <= candidate.area < 1000:
        size_score = 0.2
    elif 1000 <= candidate.area < 5000:
        size_score = 0.1
    else:
        size_score = 0.0
    score += size_score
    
    # (2) 형상 점수 (불규칙한 형태 = 종양 특징)
    if candidate.circularity < 0.7:
        score += 0.2
    
    # (3) 강도 점수 (연조직 HU 범위 20-80)
    if 20 < candidate.mean_hu < 80:
        score += 0.3
    elif -10 < candidate.mean_hu < 120:
        score += 0.1
    
    # (4) 해부학적 타당성 (대장 내부 위치)
    if is_inside_organ(candidate.centroid, anatomy_mask):
        score += 0.2
    
    return min(score, 1.0)
```

#### 4.6 NMS (Non-Maximum Suppression)부(122f)

중복 후보 제거:

```python
# 1. 신뢰도 기준 내림차순 정렬
candidates_sorted = sort(candidates, key=lambda c: -c.confidence)

# 2. 거리 계산
distances = euclidean_distance_matrix(centroids)

# 3. 필터링 (중심 간 거리 > 10mm인 경우만 유지)
keep = []
for i in range(len(candidates_sorted)):
    if all(distances[i][j] > 10 or j >= i for j in keep):
        keep.append(i)

candidates_filtered = [candidates_sorted[i] for i in keep]
```

#### 4.7 TNM 병기 결정부(122g)

**T-Stage (종양 크기):**
```
largest_tumor_diameter = max([c.diameter for c in candidates_filtered])

if largest_tumor_diameter <= 20:
    T_stage = "T1"
elif largest_tumor_diameter <= 50:
    T_stage = "T2"
elif largest_tumor_diameter <= 100:
    T_stage = "T3"
else:
    T_stage = "T4"
```

**N-Stage (림프절):**

림프절 검출 기준:
- 단축 직경 > 10mm
- 원형도 > 0.7 (L/S 비율 < 2)
- 중심부 저밀도(괴사)

```
lymph_node_count = count_lymph_nodes(ct_scan, criteria)

if lymph_node_count == 0:
    N_stage = "N0"
elif lymph_node_count <= 3:
    N_stage = "N1"
else:
    N_stage = "N2"
```

**M-Stage (전이):**

다중 장기 선별:
- 간: 저밀도 병변
- 폐: 5mm 이상 결절
- 복막: 복수, 복막 비후

```
liver_mets = detect_liver_metastases(liver_region)
lung_mets = detect_lung_nodules(lung_region)

if liver_mets or lung_mets:
    M_stage = "M1"
else:
    M_stage = "M0"

TNM = f"{T_stage}{N_stage}{M_stage}"
```

### 5. 통합 엔진의 상세 구성 (도 4 참조)

도 4는 통합 엔진(131)이 다중모달 데이터를 융합하여 통합 환자 프로파일을 생성하는 과정을 나타낸 도면이다.

#### 5.1 암 병기 결정부(131a)

TNM과 Ki-67을 결합하여 최종 병기 결정:

```python
def determine_cancer_stage(tnm_stage, ki67_index):
    # TNM 기본 병기
    tnm_to_stage = {
        "T1N0M0": "I",
        "T2N0M0": "IIA",
        "T3N0M0": "IIB",
        "T4N0M0": "IIC",
        "T1-2N1M0": "IIIA",
        "T3-4N1M0": "IIIB",
        "TanyN2M0": "IIIC",
        "TanyNanyM1": "IV"
    }
    
    base_stage = tnm_to_stage.get(tnm_stage, "Unknown")
    
    # Ki-67 기반 조정
    if ki67_index > 40 and base_stage in ["IIA", "IIB"]:
        # 높은 증식 지수는 병기 상향 고려
        adjusted_stage = upgrade_stage(base_stage)
    else:
        adjusted_stage = base_stage
    
    return adjusted_stage
```

#### 5.2 위험도 산출부(131b)

통합 위험도 점수 계산:

```python
def calculate_risk_level(stage, ki67, kras, tp53, ecog):
    risk_score = 0
    
    # 병기 기여 (0-3점)
    stage_scores = {"I": 0, "IIA": 1, "IIB": 1, "IIC": 2,
                    "IIIA": 2, "IIIB": 2, "IIIC": 3, "IV": 3}
    risk_score += stage_scores.get(stage, 0)
    
    # Ki-67 기여 (0-2점)
    if ki67 > 40:
        risk_score += 2
    elif ki67 > 20:
        risk_score += 1
    
    # 유전자 변이 (각 0-1점)
    if kras == "mutant":
        risk_score += 1
    if tp53 == "mutant":
        risk_score += 1
    
    # ECOG 수행 상태 (0-1점)
    if ecog >= 2:
        risk_score += 1
    
    # 최종 분류
    if risk_score <= 2:
        return "Low"
    elif risk_score <= 4:
        return "Medium"
    elif risk_score <= 6:
        return "Medium-High"
    else:
        return "High"
```

#### 5.3 예후 추정부(131c)

5년 생존율 추정:

```python
def estimate_prognosis(stage, risk_level):
    # 기본 병기별 생존율 (문헌 기반)
    stage_survival = {
        "I": 0.92,
        "IIA": 0.87, "IIB": 0.83, "IIC": 0.78,
        "IIIA": 0.74, "IIIB": 0.64, "IIIC": 0.53,
        "IV": 0.14
    }
    
    base_survival = stage_survival.get(stage, 0.5)
    
    # 위험도 기반 조정
    risk_adjustments = {
        "Low": 1.05,
        "Medium": 1.0,
        "Medium-High": 0.95,
        "High": 0.90
    }
    
    adjusted_survival = base_survival × risk_adjustments[risk_level]
    
    return min(max(adjusted_survival, 0.0), 1.0)
```

#### 5.4 치료 선택부(131d)

다중 기준 의사결정 알고리즘:

```python
def select_therapy(stage, kras, tp53, msi, liver_function, kidney_function):
    # (1) 가이드라인 기반 옵션 식별
    guideline_regimens = NCCN_guidelines[stage]
    
    # (2) 바이오마커 필터링
    filtered_regimens = []
    for regimen in guideline_regimens:
        # KRAS 돌연변이 시 anti-EGFR 제외
        if kras == "mutant" and "cetuximab" in regimen.drugs:
            continue
        if kras == "mutant" and "panitumumab" in regimen.drugs:
            continue
        
        # MSI-H 시 면역치료 우선
        if msi == "MSI-H":
            regimen.efficacy_boost = 1.2
        
        filtered_regimens.append(regimen)
    
    # (3) 장기 기능 기반 용량 조정
    for regimen in filtered_regimens:
        if kidney_function == "CrCl < 60 mL/min":
            if "oxaliplatin" in regimen.drugs:
                regimen.dose_adjustment = 0.75  # 25% 감량
        
        if liver_function == "Child-Pugh B":
            if "irinotecan" in regimen.drugs:
                regimen.dose_adjustment = 0.75
    
    # (4) 시너지 예측 (능동 학습 모듈 호출)
    for regimen in filtered_regimens:
        if len(regimen.drugs) >= 2:
            synergy = active_learning_module.predict_synergy(
                regimen.drugs[0], regimen.drugs[1]
            )
            regimen.efficacy += synergy × 0.15
    
    # (5) 신뢰도 계산
    for regimen in filtered_regimens:
        toxicity_risk = predict_toxicity(regimen, patient)
        regimen.confidence = regimen.efficacy - (toxicity_risk × 0.1)
    
    # (6) 상위 3개 반환
    top3 = sorted(filtered_regimens, key=lambda r: -r.confidence)[:3]
    
    return top3
```

#### 5.5 약물 칵테일 최적화부(131e)

다중 약물 조합의 시너지 효과를 정량화하고 최적 칵테일을 선택하는 모듈이다.

**5.5.1 4-모델 시너지 계산**

4가지 표준 약리학 모델을 병렬로 계산하여 합의 기반 시너지 판정을 수행한다:

```python
def calculate_drug_synergy(drug_a_effect, drug_b_effect, combined_effect,
                          dose_a, dose_b, ec50_a, ec50_b):
    """
    4가지 모델로 약물 시너지 계산
    
    Returns:
        synergy_scores: {
            'bliss': float,
            'loewe_ci': float,
            'hsa': float,
            'zip': float,
            'consensus': str  # "Strong Synergy" | "Synergy" | "Additive" | "Antagonism"
        }
    """
    scores = {}
    
    # (1) Bliss Independence 모델
    bliss_expected = drug_a_effect + drug_b_effect - (drug_a_effect * drug_b_effect)
    scores['bliss'] = combined_effect - bliss_expected
    
    # (2) Loewe Additivity 모델 (Combination Index)
    scores['loewe_ci'] = (dose_a / ec50_a) + (dose_b / ec50_b)
    # CI를 시너지 점수로 변환: CI < 1 → 양수 점수
    scores['loewe'] = 1.0 - scores['loewe_ci']
    
    # (3) Highest Single Agent (HSA) 모델
    hsa_expected = max(drug_a_effect, drug_b_effect)
    scores['hsa'] = combined_effect - hsa_expected
    
    # (4) ZIP (Zero Interaction Potency) 모델
    # Bliss와 Loewe의 통합
    zip_expected = (bliss_expected + (1 - scores['loewe_ci'])) / 2.0
    scores['zip'] = combined_effect - zip_expected
    
    # 합의 알고리즘
    positive_count = sum(1 for s in [scores['bliss'], scores['loewe'], 
                                     scores['hsa'], scores['zip']] if s > 0.1)
    avg_score = (scores['bliss'] + scores['loewe'] + 
                 scores['hsa'] + scores['zip']) / 4.0
    
    if positive_count >= 3 and avg_score > 0.2:
        scores['consensus'] = "Strong Synergy"
    elif positive_count >= 2 and avg_score > 0.1:
        scores['consensus'] = "Synergy"
    elif -0.05 <= avg_score <= 0.05:
        scores['consensus'] = "Additive"
    else:
        scores['consensus'] = "Antagonism"
    
    scores['average'] = avg_score
    
    return scores
```

일 실시예에서, 문헌 데이터베이스에 실증 시너지 데이터가 존재하는 경우 우선 사용한다:

```python
# 문헌 데이터베이스 예시
literature_synergy = {
    ("5-FU", "Oxaliplatin"): {
        "bliss": 0.18,
        "clinical_benefit": "20% PFS improvement",
        "reference": "Raymond et al., JCO 1998"
    },
    ("Irinotecan", "Bevacizumab"): {
        "bliss": 0.25,
        "clinical_benefit": "4.7 months OS improvement",
        "reference": "Hurwitz et al., NEJM 2004"
    },
    ("Encorafenib", "Cetuximab"): {
        "bliss": 0.42,
        "biomarker": "BRAF V600E",
        "clinical_benefit": "ORR 26% vs 2%",
        "reference": "Kopetz et al., NEJM 2019"
    }
}

def query_literature_synergy(drug_a, drug_b):
    """문헌 데이터베이스에서 시너지 질의"""
    pair = tuple(sorted([drug_a, drug_b]))
    return literature_synergy.get(pair, None)
```

**5.5.2 효능 가중치 부여**

시너지 점수에 기반하여 예상 효능을 조정한다:

```python
def apply_synergy_boost(base_efficacy, synergy_score):
    """
    시너지 점수로 효능 가중치 부여
    
    Args:
        base_efficacy: 기본 예상 효능 (0-1)
        synergy_score: Bliss 시너지 점수 (-1 ~ +1)
    
    Returns:
        adjusted_efficacy: 조정된 효능
    """
    # 양수 시너지만 가중치 적용
    if synergy_score > 0:
        boost_factor = 1 + (0.15 * synergy_score)  # 최대 15% 증가
        adjusted = base_efficacy * boost_factor
    else:
        # 길항 작용은 감소
        penalty_factor = 1 + (0.10 * synergy_score)  # 최대 10% 감소
        adjusted = base_efficacy * penalty_factor
    
    return min(adjusted, 1.0)

# 예시
# Base efficacy: 0.45 (45% response rate)
# Synergy score: 0.25 (Bliss)
# Adjusted: 0.45 × (1 + 0.15×0.25) = 0.45 × 1.0375 = 0.467 (46.7%)
```

**5.5.3 바이오마커 기반 칵테일 필터링**

환자의 바이오마커 상태에 따라 금기 약물을 제외하고 최적 조합을 선택한다:

```python
def filter_cocktails(cocktail_candidates, biomarkers, organ_function):
    """
    바이오마커 및 장기 기능 기반 필터링
    
    Args:
        cocktail_candidates: 후보 약물 조합 리스트
        biomarkers: {'KRAS': 'wild-type', 'TP53': 'mutant', 'MSI': 'MSS'}
        organ_function: {'liver': 'normal', 'kidney_crcl': 75}
    
    Returns:
        filtered_safe_cocktails: 필터링된 안전한 조합
    """
    filtered = []
    
    for cocktail in cocktail_candidates:
        skip = False
        
        # KRAS 돌연변이: anti-EGFR 제외
        if biomarkers.get('KRAS') == 'mutant':
            if any(drug in cocktail['drugs'] for drug in 
                   ['Cetuximab', 'Panitumumab']):
                cocktail['exclusion_reason'] = "KRAS mutant contraindication"
                skip = True
        
        # MSI-H: 면역치료 가중치
        if biomarkers.get('MSI') == 'MSI-H':
            if 'Pembrolizumab' in cocktail['drugs']:
                cocktail['efficacy_boost'] = 1.2  # 20% 증가
        
        # 간기능 저하: Irinotecan 용량 조정
        if organ_function.get('liver') == 'Child-Pugh B':
            if 'Irinotecan' in cocktail['drugs']:
                cocktail['dose_adjustments'] = {'Irinotecan': 0.75}
                cocktail['warnings'].append("Irinotecan 25% dose reduction")
        
        # 신기능 저하: Oxaliplatin 용량 조정
        if organ_function.get('kidney_crcl', 100) < 60:
            if 'Oxaliplatin' in cocktail['drugs']:
                cocktail['dose_adjustments'] = {'Oxaliplatin': 0.75}
                cocktail['warnings'].append("Oxaliplatin 25% dose reduction")
        
        if not skip:
            filtered.append(cocktail)
    
    return filtered
```

**5.5.4 종합 칵테일 선택 알고리즘**

모든 단계를 통합하여 최종 Top-3 칵테일을 선택한다:

```python
def optimize_drug_cocktail(patient_data, available_drugs):
    """
    종합 약물 칵테일 최적화
    
    Pipeline:
    1. 가이드라인 기반 후보 생성
    2. 시너지 계산 (4-모델)
    3. 바이오마커 필터링
    4. 효능 가중치 적용
    5. 신뢰도 점수 계산
    6. Top-3 선택
    """
    # Step 1: 후보 생성
    base_regimens = get_guideline_regimens(
        cancer_type=patient_data['cancer_type'],
        stage=patient_data['stage']
    )
    
    # Step 2: 시너지 계산
    for regimen in base_regimens:
        if len(regimen['drugs']) >= 2:
            # 문헌 우선 질의
            lit_synergy = query_literature_synergy(
                regimen['drugs'][0], regimen['drugs'][1]
            )
            
            if lit_synergy:
                regimen['synergy'] = lit_synergy['bliss']
                regimen['synergy_source'] = 'literature'
            else:
                # 4-모델 계산
                synergy_results = calculate_drug_synergy(
                    drug_a_effect=0.4,  # 모델 예측 또는 문헌
                    drug_b_effect=0.3,
                    combined_effect=0.75,
                    dose_a=regimen['doses'][0],
                    dose_b=regimen['doses'][1],
                    ec50_a=regimen['ec50s'][0],
                    ec50_b=regimen['ec50s'][1]
                )
                regimen['synergy'] = synergy_results['average']
                regimen['synergy_consensus'] = synergy_results['consensus']
                regimen['synergy_source'] = 'calculated'
    
    # Step 3: 바이오마커 필터링
    safe_regimens = filter_cocktails(
        base_regimens,
        patient_data['biomarkers'],
        patient_data['organ_function']
    )
    
    # Step 4: 효능 가중치 적용
    for regimen in safe_regimens:
        regimen['adjusted_efficacy'] = apply_synergy_boost(
            regimen['base_efficacy'],
            regimen['synergy']
        )
    
    # Step 5: 신뢰도 계산
    for regimen in safe_regimens:
        confidence = 0.5  # Base
        
        # 근거 수준
        if regimen.get('evidence_level') == 'Level I':
            confidence *= 1.0
        elif regimen.get('evidence_level') == 'Level II':
            confidence *= 0.8
        
        # 시너지 근거
        if regimen.get('synergy_source') == 'literature':
            confidence += 0.15
        elif regimen['synergy'] > 0.15:
            confidence += 0.10
        
        # 바이오마커 매칭
        if regimen.get('biomarker_matched'):
            confidence += 0.10
        
        regimen['confidence'] = min(confidence, 1.0)
    
    # Step 6: Top-3 선택
    ranked = sorted(safe_regimens, 
                   key=lambda r: r['adjusted_efficacy'] * r['confidence'],
                   reverse=True)
    
    return ranked[:3]
```

일 실시예에서, FOLFOX + Cetuximab 조합은 KRAS 야생형 환자에서 다음과 같이 최적화된다:
- Base efficacy: 0.50 (50% ORR)
- Synergy (literature): +0.18
- Adjusted efficacy: 0.50 × 1.027 = 0.514 (51.4% ORR)
- Confidence: 0.92 (Level I evidence + biomarker match)

### 6. 능동 학습 모듈의 상세 구성 (도 5 참조)

도 5는 능동 학습 모듈(133)에서 이중모드 획득 함수 전환을 나타낸 그래프이다.

#### 6.1 Surrogate 모델 초기화부(133a)

가우시안 프로세스 초기화:

```python
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel

kernel = ConstantKernel(1.0) * RBF(length_scale=1.0)
gp = GaussianProcessRegressor(
    kernel=kernel,
    alpha=1e-6,
    normalize_y=True,
    n_restarts_optimizer=10
)
```

#### 6.2 Thompson Sampling 획득부(133b)

반복 0-9에서 사용:

```python
def thompson_sampling(gp, drug_space, n_samples=1):
    # GP 사후 분포로부터 함수 샘플링
    X_candidate = generate_candidates(drug_space, n=1000)
    f_sample = gp.sample_y(X_candidate, n_samples=1).flatten()
    
    # 샘플링된 함수의 최댓값 위치 찾기
    best_idx = np.argmax(f_sample)
    x_next = X_candidate[best_idx]
    
    return x_next
```

일 실시예에서, drug_space는 분자량, LogP, 표적 단백질 수를 포함하는 3차원 공간이다.

#### 6.3 Expected Improvement 획득부(133c)

반복 10 이상에서 사용:

```python
from scipy.stats import norm

def expected_improvement(gp, X, f_best, xi=0.01):
    # GP 예측
    mu, sigma = gp.predict(X, return_std=True)
    
    # 개선량
    improvement = mu - f_best - xi
    Z = improvement / sigma
    
    # EI 계산
    EI = improvement * norm.cdf(Z) + sigma * norm.pdf(Z)
    
    # sigma=0인 경우 처리
    EI[sigma == 0.0] = 0.0
    
    return EI

# 최적화
x_next = maximize(lambda x: expected_improvement(gp, x, f_best), 
                  bounds=drug_space.bounds)
```

#### 6.4 DTOL 사이클 실행부(133d)

```python
def dtol_cycle(max_iterations=20, switch_iteration=10):
    # 초기 무작위 샘플링
    X_train = random_sample(drug_space, n=5)
    y_train = [query_synergy_database(x) for x in X_train]
    
    for iteration in range(max_iterations):
        # (1) Optimize: GP 업데이트
        gp.fit(X_train, y_train)
        
        # (2) Design: 다음 약물 조합 선택
        if iteration < switch_iteration:
            x_next = thompson_sampling(gp, drug_space)
        else:
            f_best = np.max(y_train)
            x_next = maximize(
                lambda x: expected_improvement(gp, x, f_best),
                drug_space
            )
        
        # (3) Test: 시너지 점수 질의
        y_next = query_synergy_database(x_next)
        
        # (4) Learn: 데이터 추가
        X_train = np.vstack([X_train, x_next])
        y_train = np.append(y_train, y_next)
        
        # 수렴 체크
        if y_next > 0.8:  # DTOL 점수 임계값
            print(f"Converged at iteration {iteration}")
            break
    
    # 최적 조합 반환
    best_idx = np.argmax(y_train)
    return X_train[best_idx], y_train[best_idx]
```

일 실시예에서, 수렴은 평균 12회 반복에서 달성된다(기존 20회 대비 40% 향상).

### 7. 설명 가능 AI 모듈의 상세 구성 (도 6, 7 참조)

도 6과 도 7은 설명 가능 AI 모듈(132)이 생성한 LIME 특징 중요도 및 Grad-CAM 히트맵 예시이다.

#### 7.1 LIME 분석부(132a)

```python
from lime import lime_tabular

def explain_with_lime(model, features, feature_names):
    # LIME 설명자 생성
    explainer = lime_tabular.LimeTabularExplainer(
        training_data=X_train,
        feature_names=feature_names,
        mode='classification'
    )
    
    # 개별 예측에 대한 설명
    explanation = explainer.explain_instance(
        data_row=features,
        predict_fn=model.predict_proba
    )
    
    # 특징 중요도 추출
    feature_importance = explanation.as_list()
    
    return feature_importance
```

**출력 예시:**
```
Ki-67 > 40%: +0.32 (종양 가능성 증가)
Area > 300 μm²: +0.18 (종양 가능성 증가)
Circularity < 0.6: +0.15 (불규칙 형태 = 종양)
KRAS mutant: +0.12 (불량 예후)
ECOG >= 2: -0.08 (수행 상태 저하)
```

#### 7.2 Grad-CAM 분석부(132b)

```python
import torch
import torch.nn.functional as F

def grad_cam(model, image, target_class):
    # Forward pass
    features = model.features(image)
    output = model.classifier(features)
    
    # Backward pass
    model.zero_grad()
    class_loss = output[0, target_class]
    class_loss.backward()
    
    # 기울기 추출
    gradients = model.get_activations_gradient()
    
    # 가중치 계산 (global average pooling)
    weights = torch.mean(gradients, dim=[2, 3], keepdim=True)
    
    # 가중 합
    cam = torch.sum(weights * features, dim=1, keepdim=True)
    
    # ReLU 적용 (양수 기여도만)
    cam = F.relu(cam)
    
    # 정규화
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    
    return cam
```

일 실시예에서, Grad-CAM은 Cellpose 네트워크의 마지막 합성곱 층(512 채널)에 적용된다.

#### 7.3 반사실적 분석부(132c)

```python
def counterfactual_analysis(patient_profile, feature_to_change, new_value):
    # 기존 예측
    original_prediction = model.predict(patient_profile)
    
    # 특징 변경
    modified_profile = patient_profile.copy()
    modified_profile[feature_to_change] = new_value
    
    # 새로운 예측
    new_prediction = model.predict(modified_profile)
    
    # 변화량 계산
    delta = new_prediction - original_prediction
    
    explanation = f"""
    반사실적 분석:
    만약 {feature_to_change}가 {patient_profile[feature_to_change]}에서 
    {new_value}로 변경된다면,
    
    예측 변화:
    - 종양 확률: {original_prediction:.1%} → {new_prediction:.1%} (Δ {delta:+.1%})
    - 권장 치료: {get_therapy(original_prediction)} → {get_therapy(new_prediction)}
    """
    
    return explanation
```

**출력 예시:**
```
만약 Ki-67이 42%에서 25%로 감소한다면,
- 5년 생존율: 64% → 78% (+14%)
- 권장 치료: FOLFIRI → FOLFOX (덜 공격적인 요법)
```

---
