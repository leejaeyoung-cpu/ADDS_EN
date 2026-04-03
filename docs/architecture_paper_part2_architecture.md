# ADDS Architecture Paper - Part 2

## 3. System Architecture

### 3.1 Overview

ADDS employs a multi-tier architecture integrating data acquisition, processing, analysis, and presentation layers. The system follows a modular design enabling independent scaling and maintenance of components while ensuring seamless data flow through the clinical decision support pipeline.

**Architecture Principles:**
1. **Modularity**: Each component (cell analysis, tumor detection, therapy recommendation) operates independently with well-defined interfaces
2. **Scalability**: Horizontal scaling through containerization and microservices
3. **Interoperability**: Standard medical imaging formats (DICOM, NIfTI) and HL7 FHIR clinical data exchange
4. **Explainability**: XAI integrated at each decision point, not post-hoc additions
5. **Performance**: GPU acceleration and caching strategies for real-time clinical workflows

### 3.2 System Layers

#### 3.2.1 Data Acquisition Layer

**Medical Imaging Input:**
- **Pathology Images**: TIFF/PNG formats, typical resolution 2048×2048 to 4096×4096 pixels
- **CT Scans**: DICOM format, volumetric data with 512×512×N slices
- **Metadata Extraction**: Automated parsing of imaging parameters (pixel spacing, slice thickness, acquisition protocols)

**Clinical Data Input:**
- Patient demographics (age, gender, BMI)
- Genomic biomarkers (KRAS mutation status, TP53, MSI-H/MSI-L)
- Laboratory results (liver function, kidney function, blood counts)
- Performance status (ECOG score 0-4)
- Treatment history (previous therapies, response, adverse events)

**Data Validation:**
- Format verification and DICOM tag parsing
- Quality assessment (image resolution, contrast, artifacts)
- Completeness checking (required clinical fields)
- Anonymization and de-identification for privacy compliance

#### 3.2.2 Processing Layer

**Cellpose Cell Segmentation Module:**
```
Input: Pathology image (H×W×C)
↓
Preprocessing:
  - Normalization (μ=0, σ=1)
  - Contrast enhancement (CLAHE)
  - Noise reduction (Gaussian σ=1.0)
↓
Cellpose Neural Network:
  - Architecture: U-Net with flow prediction
  - Model: cyto2 (generalist, 17M parameters)
  - GPU: CUDA-accelerated inference
↓
Post-processing:
  - Flow to mask conversion
  - Size filtering (10-10000 μm²)
  - Overlap resolution
↓
Output: Cell masks (H×W), Cell count N
```

**Feature Extraction:**
For each detected cell i=1..N:
- **Morphological**: Area (μm²), perimeter, circularity, eccentricity, solidity
- **Intensity**: Mean intensity, std intensity, 25th/50th/75th percentiles
- **Texture**: GLCM contrast, correlation, energy, homogeneity
- **Spatial**: Centroid coordinates, orientation angle

**CT Tumor Detection Module:**
```
Input: CT DICOM slice or volume
↓
Preprocessing:
  - Hounsfield Unit normalization
  - Windowing (W=400, L=40 for soft tissue)
  - Resampling to isotropic spacing
↓
Anatomical Segmentation (TotalSegmentator):
  - Organ identification (colon, liver, lungs)
  - Region of interest extraction
  - Anatomical context for tumor localization
↓
Tumor Candidate Detection:
  - Multi-threshold segmentation (-50 to +200 HU)
  - Morphological filtering (10-10000 mm²)
  - Feature extraction per candidate
↓
Confidence Scoring:
  - Size-based weighting
  - Shape analysis (circularity, compactness)
  - Intensity distribution (mean, std, range)
  - Anatomical plausibility
↓
Output: Tumor candidates with confidence scores
```

**Hybrid Detection Strategy:**
ADDS employs a dual-pathway approach:
1. **Anatomy-Guided Path**: TotalSegmentator → region-constrained detection
2. **Direct Detection Path**: YOLO-based object detection for speed
3. **Fusion**: Ensemble voting with confidence weighting

#### 3.2.3 Integration Layer

**CDSS Integration Engine:**

The core engine fuses multi-modal data through a structured pipeline:

```python
class CDSSIntegrationEngine:
    def integrate_patient_data(
        cellpose_results: CellposeResults,
        ct_results: CTDetectionResults,
        clinical_data: ClinicalData
    ) -> IntegratedPatientProfile:
        
        # Step 1: Determine cancer stage
        stage = determine_cancer_stage(ct_results, cellpose_results)
        
        # Step 2: Calculate risk level
        risk = calculate_risk_level(stage, cellpose_results, clinical_data)
        
        # Step 3: Estimate prognosis
        survival_5yr = estimate_prognosis(stage, risk)
        
        # Step 4: Select therapy
        therapies = select_therapy(stage, clinical_data, ct_results)
        
        # Step 5: Generate interpretation (OpenAI)
        interpretation = generate_medical_interpretation(...)
        
        return IntegratedPatientProfile(...)
```

**Cancer Stage Determination:**
TNM staging from integrated data:
- **T (Tumor size)**: From CT tumor detection (diameter, volume)
- **N (Nodes)**: From CT nodal analysis + clinical assessment
- **M (Metastasis)**: From clinical data and multi-organ CT screening
- **Ki-67**: From cell analysis proliferation index

**Risk Stratification Algorithm:**
```
risk_score = 0
if stage in ['IV', 'IIIC']: risk_score += 3
elif stage in ['IIIB', 'IIIA']: risk_score += 2
elif stage in ['IIB', 'IIC']: risk_score += 1

if Ki67 > 40%: risk_score += 2
elif Ki67 > 20%: risk_score += 1

if KRAS mutant: risk_score += 1
if TP53 mutant: risk_score += 1
if ECOG >= 2: risk_score += 1

risk_level = 'Low' if risk_score <= 2
           else 'Medium' if risk_score <= 4
           else 'Medium-High' if risk_score <= 6
           else 'High'
```

**Therapy Selection Engine:**

Multi-criteria decision algorithm integrating:
1. **Guidelines Matching**: NCCN/ESMO guideline lookup by stage
2. **Biomarker Filtering**: Exclude contraindicated drugs (e.g., anti-EGFR if KRAS mutant)
3. **Organ Function Adjustment**: Dose modifications for hepatic/renal impairment
4. **Synergy Prediction**: Active learning model ranking drug combinations

```
For each candidate regimen:
  base_efficacy = guideline_efficacy[stage][regimen]
  
  # Biomarker adjustments
  if biomarker_match:
    efficacy *= 1.2
  if biomarker_conflict:
    efficacy *= 0.6
    
  # Synergy bonus
  synergy = bliss_independence(drug_A, drug_B)
  efficacy += synergy * 0.15
  
  # Toxicity penalty
  toxicity_risk = predict_toxicity(regimen, patient)
  confidence = efficacy - (toxicity_risk * 0.1)

Return top 3 regimens by confidence
```

#### 3.2.4 Presentation Layer

**Physician Interface:**
Streamlit-based web application with:
- Multi-tab workflow (Data Input → Analysis → Treatment Plan)
- Interactive visualizations (Plotly, Matplotlib)
- Real-time progress tracking (5-stage pipeline monitoring)
- Export capabilities (PDF reports, JSON data, CSV results)

**Patient Interface:**
Simplified view providing:
- Condition summary in lay language (OpenAI-generated)
- Treatment plan visualization (timeline, drug names)
- Prognosis estimates with confidence intervals
- Educational resources (condition-specific)

**API Layer:**
RESTful FastAPI service offering:
```
POST /api/v1/segmentation - Cell segmentation analysis
POST /api/v1/tumor_detection - CT tumor detection
POST /api/v1/cdss/integrate - Full CDSS integration
GET /api/v1/analysis/{id} - Retrieve analysis results
GET /api/docs - Interactive API documentation
```

### 3.3 Data Flow Architecture

```
┌─────────────────────────────────────────────────┐
│  User Interface (Streamlit)                     │
│  ┌──────────────┐  ┌──────────────┐            │
│  │ Physician UI  │  │ Patient UI   │            │
│  └──────────────┘  └──────────────┘            │
└─────────────────────────────────────────────────┘
                    ↕ HTTP/WebSocket
┌─────────────────────────────────────────────────┐
│  API Gateway (FastAPI)                          │
│  ┌──────────────────────────────────────────┐  │
│  │ Authentication │ Rate Limiting │ Logging │  │
│  └──────────────────────────────────────────┘  │
└─────────────────────────────────────────────────┘
                    ↕
┌─────────────────────────────────────────────────┐
│  Service Layer                                  │
│  ┌─────────────┐ ┌─────────────┐ ┌──────────┐ │
│  │ Segmentation│ │ Detection   │ │ CDSS     │ │
│  │ Service     │ │ Service     │ │ Service  │ │
│  └─────────────┘ └─────────────┘ └──────────┘ │
└─────────────────────────────────────────────────┘
                    ↕
┌─────────────────────────────────────────────────┐
│  Model Layer                                    │
│  ┌──────────┐ ┌──────────┐ ┌──────────────┐   │
│  │ Cellpose │ │ YOLO/    │ │ Therapy      │   │
│  │ (GPU)    │ │ nnU-Net  │ │ Recommender  │   │
│  └──────────┘ └──────────┘ └──────────────┘   │
└─────────────────────────────────────────────────┘
                    ↕
┌─────────────────────────────────────────────────┐
│  Data Layer                                     │
│  ┌──────────┐ ┌──────────┐ ┌──────────────┐   │
│  │ SQLite   │ │ File     │ │ Model Cache  │   │
│  │ Database │ │ Storage  │ │ (models/)    │   │
│  └──────────┘ └──────────┘ └──────────────┘   │
└─────────────────────────────────────────────────┘
```

### 3.4 Explainable AI Integration

XAI components integrated at three levels:

**Image-Level XAI:**
- Grad-CAM heatmaps showing Cellpose attention regions
- Overlay visualizations highlighting tumor vs. normal tissue discrimination

**Feature-Level XAI:**
- LIME explanations for key morphological features driving predictions
- Individual cell contribution scores to overall diagnosis

**Decision-Level XAI:**
- Counterfactual analysis: "If Ki-67 were 25% instead of 42%, FOLFOX confidence would increase by 8%"
- Therapy pathway trees showing decision logic from biomarkers to recommendations

### 3.5 Performance Optimization Strategies

**GPU Acceleration:**
- CUDA-enabled Cellpose inference
- Batch processing for multiple images
- Mixed-precision (FP16) computation reducing memory 50%

**Caching:**
- Model weights cached in `models/` directory (persistent)
- Analysis results cached in SQLite with TTL
- Intelligent cache invalidation on parameter changes

**Database Indexing:**
- Composite indexes on (patient_id, timestamp)
- B-tree indexes on analysis_date for temporal queries
- Result: 330× query speedup (100ms → 0.3ms)

**API Optimization:**
- Gunicorn with 9 workers for 9× concurrency
- GZip compression reducing response size 70%
- Connection pooling for database access

**Result:**
- CPU mode: 20-30s per image
- GPU mode: 5-7s per image (4.2× speedup)
- API throughput: 80 requests/second
- 92% production readiness score

---
