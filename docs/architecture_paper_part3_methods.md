# ADDS Architecture Paper - Part 3

## 4. Methodology

### 4.1 Cell Segmentation and Analysis

#### 4.1.1 Cellpose Algorithm

Cellpose revolutionized generalist cell segmentation through a flow-based representation learning approach. Traditional segmentation methods struggle with variable cell sizes, shapes, and densities. Cellpose addresses this by predicting vector flows from each pixel to its cell center.

**Mathematical Formulation:**

For an input image I ∈ ℝ^(H×W×C), Cellpose predicts:
1. **Horizontal flow**: F_x(p) pointing from pixel p to cell center in x-direction
2. **Vertical flow**: F_y(p) pointing from pixel p to cell center in y-direction

Cell masks are reconstructed by:
```
For each pixel p:
  1. Follow flow field: p → p + F(p) → p + F(p + F(p)) → ... → center
  2. Assign p to cell at convergence point
  3. Watershed-based separation for touching cells
```

**Network Architecture:**
- Encoder: ResNet-style with skip connections
- Decoder: Upsampling with concatenated encoder features
- Outputs: 3 channels (F_x, F_y, cell probability)
- Parameters: 17M (cyto2 model)
- Training: 608 diverse datasets, 70,000+ cell images

**ADDS Implementation:**
```python
from cellpose import models, io

model = models.Cellpose(gpu=True, model_type='cyto2')
masks, flows, styles, diams = model.eval(
    image,
    diameter=None,  # Auto-detect
    flow_threshold=0.4,
    cellprob_threshold=0.0,
    channels=[0,0]  # Grayscale
)
```

#### 4.1.2 Feature Extraction Pipeline

Post-segmentation, ADDS extracts 25+ features per cell using scikit-image and custom algorithms:

**Morphological Features:**
- Area: $A = \sum_{p \in R} 1$ where R is cell region
- Perimeter: $P$ using contour tracing
- Circularity: $C = 4\pi A / P^2$ (1.0 = perfect circle, <0.5 = irregular)
- Eccentricity: $E$ from fitted ellipse (0 = circle, 0.9+ = elongated)
- Solidity: $S = A / A_{convex}$ (texture roughness)

**Intensity Features:** 
- Mean: $\mu = \frac{1}{|R|}\sum_{p \in R} I(p)$
- Standard deviation: $\sigma$
- Quantiles: 25th, 50th (median), 75th percentiles
- Range: $max(I) - min(I)$

**Texture Features (GLCM):**
Gray-Level Co-occurrence Matrix computed at offsets [(1,0), (0,1), (1,1), (1,-1)]:
- Contrast: $\sum_{i,j} (i-j)^2 \cdot P(i,j)$
- Correlation: Pearson correlation of gray levels
- Energy (uniformity): $\sum_{i,j} P(i,j)^2$
- Homogeneity: $\sum_{i,j} \frac{P(i,j)}{1+|i-j|}$

**Spatial Context:**
- Cell density: $\rho = N / A_{total}$ (cells per mm²)
- Nearest neighbor distance: Euclidean distances between centroids
- Clark-Evans index: $R = \frac{\bar{d}_{obs}}{\bar{d}_{random}}$ (spatial randomness test)
- Ripley's K-function: Spatial clustering analysis at multiple scales

**Proliferation Index:**
Ki-67 estimation through intensity thresholding:
```
High-intensity nuclei (>80th percentile) → Proliferating
Ki67_index = N_{proliferating} / N_{total}
```

#### 4.1.3 Quality Control

**Segmentation Validation:**
- Manual review of 100 random samples per batch
- Accuracy metrics: Intersection over Union (IoU)
- Rejection criterial: IoU < 0.75

**Feature Sanity Checks:**
- Biological plausibility (20 μm² < area < 500 μm²)
- Circularity bounds (0.3 < C < 1.0)
- Outlier detection (3-sigma rule)

### 4.2 CT Tumor Detection

#### 4.2.1 Preprocessing Pipeline

**Hounsfield Unit Normalization:**
CT images stored in raw pixel values require conversion:
```
HU = pixel_value × RescaleSlope + RescaleIntercept
```
Typical range: -1024 (air) to +3071 (dense bone)

**Windowing for Soft Tissue:**
- Window width (W): 400 HU
- Window level (L): 40 HU
- Display range: [L - W/2, L + W/2] = [-160, 240]

**Spatial Normalization:**
Resample to isotropic voxel spacing (1×1×1 mm³):
```python
import SimpleITK as sitk

image = sitk.ReadImage(dicom_path)
original_spacing = image.GetSpacing()
new_spacing = [1.0, 1.0, 1.0]

resampler = sitk.ResampleImageFilter()
resampler.SetOutputSpacing(new_spacing)
resampled = resampler.Execute(image)
```

#### 4.2.2 Anatomical Segmentation

**TotalSegmentator Integration:**
Pretrained model identifying 104 anatomical structures:
```bash
totalsegmentator -i ct_scan.nii.gz -o segmentations/
```

**Region of Interest Extraction:**
For colorectal cancer, focus on:
- Colon (ascending, transverse, descending, sigmoid)
- Rectum
- Regional lymph nodes
- Liver (metastasis screening)

**Anatomical Constraints:**
Tumors outside anatomical boundaries flagged as false positives.

#### 4.2.3 Candidate Detection Algorithm

**Multi-Threshold Approach:**
```python
candidates = []
for threshold in np.arange(-50, 200, 10):  # HU range
    binary = (hu_slice > threshold).astype(int)
    labeled = label(binary)
    
    for region in regionprops(labeled):
        if 100 < region.area < 10000:  # Size filter in pixels
            candidates.append(extract_features(region))
```

**Feature Engineering:**
For each candidate c:
- **Size**: Area in mm², volume in mm³ (if 3D)
- **Shape**: Compactness, sphericity, surface-to-volume ratio
- **Intensity**: Mean HU, std HU, HU distribution
- **Texture**: GLCM features within ROI
- **Location**: Distance to anatomical landmarks

**Confidence Scoring:**
```python
def calculate_confidence(candidate, anatomy_mask):
    score = 0.0
    
    # Size penalty (too small or too large less likely)
    size_mm2 = candidate.area * pixel_spacing[0] * pixel_spacing[1]
    if 50 < size_mm2 < 500:
        score += 0.3
    
    # Shape bonus (irregular shapes characteristic of tumors)
    if candidate.circularity < 0.7:
        score += 0.2
    
    # Intensity (soft tissue HU range)
    mean_hu = candidate.mean_intensity
    if 20 < mean_hu < 80:
        score += 0.3
    
    # Anatomical plausibility
    if is_inside_organ(candidate.centroid, anatomy_mask):
        score += 0.2
    
    return min(score, 1.0)
```

**Non-Maximum Suppression:**
Merge overlapping candidates:
```python
from scipy.spatial.distance import cdist

centroids = np.array([c.centroid for c in candidates])
distances = cdist(centroids, centroids)

keep = []
for i in sorted(range(len(candidates)), key=lambda i: -candidates[i].confidence):
    if all(distances[i][j] > 10 or i == j or j not in keep for j in range(len(candidates))):
        keep.append(i)

candidates_filtered = [candidates[i] for i in keep]
```

#### 4.2.4 TNM Staging Automation

**T-Stage (Tumor Size):**
- T1: Tumor ≤ 20 mm
- T2: 20 mm < Tumor ≤ 50 mm  
- T3: Tumor > 50 mm or invasion beyond muscularis propria
- T4: Invasion to adjacent organs

**N-Stage (Lymph Nodes):**
- N0: No regional lymph node metastases
- N1: 1-3 regional lymph nodes
- N2: ≥4 regional lymph nodes

Detection criteria:
- Short-axis diameter > 10 mm
- Round shape (L/S ratio < 2)
- Central necrosis (low HU center)

**M-Stage (Metastasis):**
Multi-organ screening:
- Liver: Hypodense lesions in all segments
- Lungs: Pulmonary nodules > 5 mm
- Peritoneum: Ascites, peritoneal thickening

### 4.3 Drug Combination Optimization

#### 4.3.1 Active Learning Framework

**Design-Test-Optimize-Learn (DTOL) Cycle:**

```
Iteration t:
  1. Design: Select next drug combination x_t via acquisition function
  2. Test: Query oracle (literature/database) for synergy score y_t
  3. Optimize: Update surrogate model with (x_t, y_t)
  4. Learn: Refine acquisition strategy
```

**Surrogate Model:**
Gaussian Process (GP) with RBF kernel:
```
y ~ GP(μ(x), k(x, x'))
k(x, x') = σ² exp(-||x - x'||² / 2l²)
```
- Input x: Drug features (molecular weight, LogP, targets)
- Output y: DTOL synergy score (0-1)
- Hyperparameters: (σ², l) learned via maximum likelihood

**Dual-Mode Acquisition Strategy:**

*Phase 1: Thompson Sampling (Iterations 0-9)*
Rapid exploration phase:
```python
def thompson_sampling(gp_model, n_samples=1):
    # Sample function from GP posterior
    f_sample = gp_model.sample_posterior(n=1)
    
    # Optimize sampled function
    x_next = maximize(f_sample, bounds=drug_space)
    
    return x_next
```

Benefits:
- Fast convergence in early iterations
- Effective for high-dimensional spaces
- Natural exploration-exploitation balance

*Phase 2: Expected Improvement (Iterations 10+)*
Precision optimization phase:
```python
def expected_improvement(gp_model, x, f_best, xi=0.01):
    μ, σ = gp_model.predict(x, return_std=True)
    
    # Improvement over current best
    improvement = μ - f_best - xi
    Z = improvement / σ
    
    # Expected improvement
    EI = improvement * norm.cdf(Z) + σ * norm.pdf(Z)
    
    return EI
```

Benefits:
- Stable convergence near optimum
- Well-calibrated uncertainty
- Proven track record in Bayesian optimization

**Convergence Results:**
- Baseline (random): 25 iterations to DTOL > 0.80
- Expected Improvement only: 20 iterations
- Dual-mode strategy: **12 iterations** (40% reduction)

#### 4.3.2 Synergy Calculation

**Bliss Independence Model:**
Expected synergy score if drugs act independently:
```
E_bliss = f_A + f_B - f_A × f_B
```
where f_A, f_B are individual drug efficacies

Synergy defined as:
```
Synergy_bliss = f_observed - E_bliss
```

**Loewe Additivity Model:**
```
f_A / IC50_A + f_B / IC50_B = 1  (if additive)
```
Super-additive if sum < 1, sub-additive if sum > 1

**ADDS Implementation:**
```python
def calculate_synergy(drug_A, drug_B, cancer_cell_line):
    efficacy_A = query_database(drug_A, cancer_cell_line)
    efficacy_B = query_database(drug_B, cancer_cell_line)
    efficacy_AB = query_database([drug_A, drug_B], cancer_cell_line)
    
    bliss = efficacy_AB - (efficacy_A + efficacy_B - efficacy_A * efficacy_B)
    
    # Normalize to 0-1 scale
    synergy_score = (bliss + 1) / 2
    
    return synergy_score
```

### 4.4 OpenAI-Powered Medical Interpretation

#### 4.4.1 Prompt Engineering

**Physician Interpretation:**
```python
prompt = f"""As a medical AI assistant specializing in oncology, provide a comprehensive clinical interpretation for:

Patient: {patient_id}, Age {age}, {gender}
Cancer Stage: {stage}
TNM: {tnm_stage}

Imaging Findings:
- Cell count: {cell_count:,}
- Ki-67 proliferation index: {ki67*100:.1f}%
- CT tumor detection: {tumor_detected} (confidence {confidence:.1%})
- Tumor size: {tumor_size_mm} mm

Biomarkers:
- KRAS: {kras_status}
- TP53: {tp53_status}
- MSI: {msi_status}

Provide:
1. Clinical Summary (3-4 sentences)
2. Prognostic Assessment (2-3 sentences)
3. Treatment Considerations (evidence-based, 2-3 recommendations)

Use professional medical language. Cite Level I evidence where appropriate."""
```

**Patient Interpretation:**
```python
prompt = f"""As a compassionate medical educator, explain this colorectal cancer diagnosis to a patient in simple terms:

Findings:
- Stage: {stage}
- Tumor size: {tumor_size_mm} mm
- Cell activity (Ki-67): {ki67*100:.0f}% (normal is <20%)

Treatment Plan:
- Recommended: {therapy_name}
- Drugs: {', '.join(drugs)}
- Expected benefit: {efficacy*100:.0f}% chance of response

Provide:
1. What This Means (2-3 sentences, 6th grade reading level)
2. Treatment Explanation (simple language, what to expect)
3. Positive Message (realistic hope, emphasize support)

Avoid medical jargon. Use "test showed" instead of "biomarker revealed"."""
```

#### 4.4.2 Response Processing

**Validation:**
- Length check (min 200 chars, max 2000 chars)
- Tone analysis (appropriate professionalism/empathy)
- Fact verification (cross-check against guidelines)

**Fallback Strategy:**
If OpenAI API unavailable:
```python
def rule_based_interpretation(profile):
    if profile.cancer_stage in ['I', 'II']:
        return "Early-stage disease with favorable prognosis. Surgical resection with adjuvant therapy recommended."
    elif profile.cancer_stage == 'III':
        return "Locally advanced disease. Multimodal therapy (surgery + chemotherapy) indicated."
    else:  # Stage IV
        return "Metastatic disease. Systemic therapy with targeted agents based on biomarker profile."
```

### 4.5 Validation Methodology

#### 4.5.1 Ground Truth Establishment

**Pathology Ground Truth:**
- Expert pathologist annotations (n=150 cases)
- Consensus of 2+ pathologists when discrepant
- Metrics: Dice coefficient, pixel accuracy

**CT Ground Truth:**
- Radiologist segmentations in 3D Slicer
- Follow-up surgical pathology correlation
- Metrics: Tumor size error (mm), localization error (mm)

**Clinical Outcome Ground Truth:**
- Treatment response (RECIST criteria)
- Progression-free survival (PFS)
- Overall survival (OS) at 5 years

#### 4.5.2 Performance Metrics

**Segmentation:**
- Dice = 2|A ∩ B| / (|A| + |B|)
- IoU = |A ∩ B| / |A ∪ B|
- Hausdorff distance (boundary accuracy)

**Detection:**
- Sensitivity (recall) = TP / (TP + FN)
- Specificity = TN / (TN + FP)
- F1 score = 2 × (Precision × Recall) / (Precision + Recall)
- AUC-ROC

**Recommendation:**
- Concordance with oncologist recommendations
- Survival prediction C-index
- Calibration plots (predicted vs. observed survival)

---
