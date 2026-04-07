<div align="center">

<img src="https://img.shields.io/badge/ADDS-v3.5.0-blueviolet?style=for-the-badge&logo=python" alt="ADDS Version"/>

# ADDS â AI-Driven Drug Synergy & Diagnostic System

**Multimodal AI Platform for Precision Oncology**

[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x_GPU-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-Backend-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-UI-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Cellpose](https://img.shields.io/badge/Cellpose-cyto3-00C49F)](https://cellpose.readthedocs.io/)
[![nnU-Net](https://img.shields.io/badge/nnU--Net-v2-FF6B35)](https://github.com/MIC-DKFZ/nnUNet)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)
[![CI](https://github.com/leejaeyoung-cpu/ADDS/actions/workflows/ci.yml/badge.svg)](https://github.com/leejaeyoung-cpu/ADDS/actions)
[![Institution](https://img.shields.io/badge/Institution-Inha_University_Hospital-003DA5)](https://www.inha.com/)

<br/>

> **ADDS** integrates CT radiomics, cell morphometrics, pharmacokinetic modeling, and machine learning  
> into a unified platform that recommends personalized anti-cancer drug cocktails for colorectal cancer (CRC) patients.

</div>

---

## Table of Contents

- [System Overview](#-system-overview)
- [Architecture](#-architecture)
- [Core Modules](#-core-modules)
  - [CT Analysis Pipeline](#1-ct-analysis-pipeline)
  - [Cellpose Microscopy Analysis](#2-cellpose-microscopy-analysis)
  - [KRAS-PrPc Drug Synergy](#3-kras-prpc-drug-synergy)
  - [Pharmacokinetic (PK/PD) Modeling](#4-pharmacokinetic-pkpd-modeling)
  - [Clinical Decision Support (CDS)](#5-clinical-decision-support-cds)
  - [Patient Management System](#6-integrated-patient-management-system)
- [Performance Metrics](#-performance-metrics)
- [14D Feature Vector](#-14-dimensional-multimodal-feature-vector)
- [Installation](#-installation)
- [API Reference](#-api-reference)
- [Data Structure](#-data-structure)
- [Research Background](#-research-background)
- [Citation](#-citation)

---

## ð¬ System Overview

ADDS (AI-Driven Drug Synergy) is a **precision oncology AI ecosystem** developed through collaborative research with Inha University Hospital.

### Core Innovations

| Innovation | Description |
|-----------|-------------|
| **Multimodal Data Fusion** | CT radiomics + cell pathology + clinical metadata unified into a single 14-dimensional feature vector |
| **Dual Inference Engine** | ADDS pathway-based engine + OpenAI GPT-4 running simultaneously with cross-validation |
| **RAG-Based Evidence Generation** | Retrieval-Augmented Generation (RAG) system using physician notes as priority-1 prompt |
| **PrPc Biomarker Discovery** | Novel biomarker discovery via KRAS-RPSA signalosome from TCGA data (n=2,285) |
| **Real-Time Clinical Application** | End-to-end analysis completed within 15.67 seconds (530Ã751Ã750 volume) |

---

## ðï¸ Architecture

```
âââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââ
â                    ADDS Precision Oncology Platform v3.5             â
â                      Inha University Hospital                        â
âââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââ
                                    â
          âââââââââââââââââââââââââââ¼ââââââââââââââââââââââââââ
          â¼                         â¼                         â¼
  âââââââââââââââââ       âââââââââââââââââââ       ââââââââââââââââââ
  â  Streamlit UI â       â  FastAPI Backend â       â  Data Layer    â
  â  (Port 8505)  ââââââââºâ  (Port 8000)    ââââââââºâ  SQLite / NFS  â
  â               â       â                 â       â                â
  â â¢ Patient Mgmtâ       â /api/v1/        â       â patients.db    â
  â â¢ AI Analysis â       â  ââ patients    â       â ct_data/       â
  â â¢ Drug Reco   â       â  ââ ct          â       â microscopy/    â
  â â¢ Reports     â       â  ââ cellpose    â       â literature/    â
  âââââââââââââââââ       â  ââ pharmacoki  â       ââââââââââââââââââ
                          â  ââ adds        â
                          â  ââ openai      â
                          âââââââââââââââââââ
                                    â
         ââââââââââââââââââââââââââââ¼âââââââââââââââââââââââââââ
         â¼                          â¼                          â¼
ââââââââââââââââââ        âââââââââââââââââââ        ââââââââââââââââââ
â  CT Pipeline   â        â Cellpose Pipelineâ        â  Drug Synergy  â
â  (6 Stages)    â        â                 â        â  Engine        â
â                â        â cyto3 Model     â        â                â
â S1: DICOMâNIfTIâ        â â Segmentation  â        â KRAS-PrPc      â
â S2: Organ Seg  â        â â Ki-67 Index   â        â Signalosome    â
â S3: Tumor Det  â        â â Morphology    â        â                â
â S4: Radiomics  â        â â Heterogeneity â        â Pritamab       â
â S5: Staging    â        â                 â        â Prediction     â
â S6: ADDS Integ â        â n=43,190 cells  â        â                â
â                â        â analyzed        â        â PK/PD Modeling â
â Acc: 98.65%    â        â                 â        â                â
ââââââââââââââââââ        âââââââââââââââââââ        ââââââââââââââââââ
         â                          â                          â
         ââââââââââââââââââââââââââââ¼âââââââââââââââââââââââââââ
                                    â¼
                    âââââââââââââââââââââââââââââââââ
                    â    14D Multimodal Feature      â
                    â    Vector Fusion               â
                    â                                â
                    â  CT Radiomics (7D):            â
                    â  Sphericity, Entropy,          â
                    â  Contrast, Size, Circularity,  â
                    â  Mean HU, Confidence           â
                    â                                â
                    â  Cell Culture (7D):            â
                    â  Density, Drug Resistance,     â
                    â  Proliferation, Complexity,    â
                    â  Circularity, Clark-Evans,     â
                    â  Viability                     â
                    âââââââââââââââââââââââââââââââââ
                                    â
                    âââââââââââââââââ´ââââââââââââââââ
                    â¼                               â¼
         âââââââââââââââââââ             ââââââââââââââââââââ
         â  ADDS Engine    â             â  OpenAI Engine   â
         â  (Pathway-Based)â             â  (GPT-4 Medical) â
         â                 â             â                  â
         â KRAS/RAF/MEK/   â             â Clinical Summary â
         â ERK Signaling   ââââ Cross âââºâ Treatment Plan   â
         â Synergy Scoring â  Validate   â MDT Consensus    â
         âââââââââââââââââââ             ââââââââââââââââââââ
                    â                               â
                    âââââââââââââââââ¬ââââââââââââââââ
                                    â¼
                    âââââââââââââââââââââââââââââââââ
                    â   Final Drug Cocktail          â
                    â   Recommendation               â
                    â                                â
                    â  FOLFOX + Bevacizumab          â
                    â  + PK-Optimized Dosing         â
                    â  + Outcome Simulation          â
                    â   (ORR / PFS / OS)             â
                    âââââââââââââââââââââââââââââââââ
```

---

## âï¸ Core Modules

### 1. CT Analysis Pipeline

**6-Stage 3D CT Tumor Detection and Radiomics Analysis Pipeline**

```
Stage 1: 3D Volume Reconstruction
    DICOM Series â 1mmÂ³ Isotropic NIfTI Volume
    (SimpleITK, scipy-based resampling)

Stage 2: Anatomical Organ Segmentation
    nnU-Net v2 â Colon / Liver / Lymph Node Parsing

Stage 3: Tumor Detection  â VerifiedCTDetector (98.65% Accuracy)
    HU Thresholding: 60â120 HU (Arterial Phase)
    2D Slice-by-Slice Morphological Filtering
    Min Size: 30 px (noise), 50 mmÂ³ (clinical threshold)

Stage 4: Radiomics Extraction
    PyRadiomics â 100+ Phenotypic Features
    (Sphericity, Entropy, GLCM Contrast, Surface Area...)

Stage 5: Biomarker Prediction
    Malignancy Score / TNM Staging / MSI / KRAS Status

Stage 6: ADDS Integration
    Radiomics â PK Sensitivity Model â Drug Recommendation
```

**Key Performance Metrics (Inha University Hospital Cohort)**

| Metric | Value |
|--------|-------|
| Detection Accuracy | **98.65%** (73 of 74 slices) |
| Processing Time | **15.67s** (530Ã751Ã750 volume) |
| Throughput | **33.8 slices/sec** |
| HU Detection Range | 60â120 HU (arterial phase) |
| Minimum Lesion Size | 50 mmÂ³ |

---

### 2. Cellpose Microscopy Analysis

**Automated HUVEC Cell Morphometry Analysis (Cellpose cyto3 Model)**

```
Raw Microscopy Image
       â
       â¼
CLAHE + Denoising (Preprocessing)
       â
       â¼
Cellpose cyto3 Segmentation
       â
       âââ Cell Count & Density
       âââ Elongation Ratio (major/minor axis)
       âââ Circularity Score
       âââ Clark-Evans Index (spatial distribution)
       âââ Ki-67 Proliferation Index Estimation
       âââ Tumor Heterogeneity Score
```

**Analysis Results (HUVEC Serum Experiment, n = 43,190 cells)**

| Condition | Cell Count | Elongation | Cell Area | Interpretation |
|-----------|-----------|------------|-----------|----------------|
| Control | 11,717 | 1.831 | 696 pxÂ² | Resting state |
| Healthy Serum | 6,538 | 1.865 | 618 pxÂ² | Normal activation |
| HGPS Serum | 13,676 | 1.902 | 756 pxÂ² | Pathological activation |
| **HGPS + MT-Exo** | **11,259** | **1.992** | **775 pxÂ²** | **Maximum endothelial activation** |

> Significant increase in cell elongation ratio in MT-Exo treated group (p < 0.001) â suggests enhanced endothelial cell migration capacity

---

### 3. KRAS-PrPc Drug Synergy

**Mechanism-Based Drug Synergy Prediction Engine**

#### Resolving the PrPc Tissue-Serum Paradox

| Measurement | CRC Tissue | Serum | Mechanism |
|-------------|-----------|-------|-----------|
| PRNP mRNA | â Low | â | Tumor suppression |
| PrPc Protein | â | ââ High | **ADAM10/17 Shedding** |

> ADAM10/17 enzymes cleave GPI-anchored PrPc from cell membrane â released into bloodstream  
> Validated with real TCGA data: n = 2,285 (BRCA, STAD, COAD, PAAD, READ)

#### KRAS-RPSA Signalosome Pathway

```
KRAS Mutation (G12D/G12V)
       â
       â¼
RAF â MEK â ERK Activation
       â
       âââ PrPc-RPSA Complex Formation
       â         â
       â         âââ Laminin Binding (promotes cell invasion)
       â
       âââ Downstream Survival Pathways
                 â
                 âââ mTOR Axis
                 âââ PI3K/AKT
                 âââ WNT/Î²-catenin
```

#### Drug Knowledge Base

| Metric | Value |
|--------|-------|
| Total Publications | 311 (Nature/Cell/Science and other Tier-1 journals) |
| Data Samples | 2,348 clinical samples |
| Registered Drugs | 113 |
| Mechanisms of Action | 90 |
| Biomarkers | 69 |
| Synergy Combinations | 59 |

---

### 4. Pharmacokinetic (PK/PD) Modeling

**Patient-Specific Anticancer Drug Dose Optimization â 1-Compartment Model**

$$C_{max} = \frac{D}{V_d} \cdot e^{-k_e \cdot t}$$

| Parameter | Formula | Unit |
|-----------|---------|------|
| **Clearance (Cl)** | $120.0 \times \max(0.7, 1.0 - \frac{V_{tumor}}{500})$ | mL/min |
| **Volume of Distribution (Vd)** | $45.0 + (V_{tumor} \times 0.5)$ | L |
| **Half-life (tÂ½)** | $0.693 \times \frac{V_d}{Cl \times 0.06}$ | hours |
| **Optimal Dose (D)** | $200.0 \times (1.0 + \frac{Ki67}{200})$ | mg/mÂ² |

**Safety Constraints:**
- Dosing interval: 6h â 24h (hard clamp)
- Maximum response rate: 95% (clinical realism)
- Renal/hepatic function proxy: `cl_factor` (tumor burden-based)

---

### 5. Clinical Decision Support (CDS)

**Dual Inference Engine Cross-Validation System**

```
âââââââââââââââââââââââââââââââââââââââââââââââââââââââââââ
â            6-Step Dynamic Inference Pipeline             â
âââââââââââââââââââââââââââââââââââââââââââââââââââââââââââ

Step 0: RAG Analysis
    Physician notes â Semantic clinical context extraction
    (symptoms, medical history, patient preferences)

Step 1: CT Analysis (Live API)
    DICOM upload â /api/v1/ct/analyze
    Output: Radiomics JSON + visualization image stream

Step 2: Cell Analysis (Conditional)
    Cellpose segmentation â Ki-67 quantification
    (skipped if no microscopy images provided)

Step 3: Pharmacokinetics
    CT + Cellpose results â PK optimization parameters

Step 4: ADDS Inference
    Pathway-based mechanistic recommendations
    (RAG context + multimodal data)

Step 5: OpenAI Inference
    GPT-4 clinical integration (physician notes as priority-1 prompt)

Step 6: Cross-Validation
    Notes â CT results â Pathology results â automated consistency check
```

**Final Recommendation Output:**
- ð¯ Drug cocktail (e.g., FOLFOX + Bevacizumab)
- ð Optimized dosage and route
- ð Outcome simulation (ORR / PFS / OS)
- ð Dual report (clinical technical report + patient guide)

---

### 6. Integrated Patient Management System

**Enterprise-Grade Clinical Data Management (IPMS)**

```python
# Patient ID format
Patient ID: P-2026-001

# Core clinical metadata
{
  "tnm_stage": "T4N0M0",
  "msi_status": "MSS",
  "kras_mutation": "G12D",
  "ecog_score": 1,
  "ki67_index": 45.2,
  "tumor_location": "Sigmoid Colon"
}
```

| Feature | Description |
|---------|-------------|
| **Patient CRUD** | Permanent records in P-YYYY-NNN format |
| **Longitudinal Tracking** | Complete data history across treatment course |
| **Multimodal Upload** | CT DICOM + microscopy images + physician notes integration |
| **Real-time Progress** | Live status tracking for each analysis stage |
| **PDF Reports** | Auto-generated (clinical / patient versions) |

---

## ð Performance Metrics

### CT Analysis Performance
```
âââââââââââââââââââââââââââââââââââââââââââââââââââââââ
â  CT Detection Performance (Inha University Hospital) â
â  âââââââââââââââââââââââââââââââââââââââââââââââââââ â
â  Accuracy:      ââââââââââââââââââââ 98.65%         â
â  Speed:         15.67s / patient (E2E)               â
â  Throughput:    33.8 slices/sec                      â
â  Volume Size:   530 Ã 751 Ã 750 voxels               â
â  HU Range:      60 â 120 HU (arterial phase)         â
â  Min Lesion:    50 mmÂ³                               â
âââââââââââââââââââââââââââââââââââââââââââââââââââââââ
```

### System Benchmark
| Configuration | Processing Time |
|--------------|----------------|
| CT E2E Analysis (standard) | ~45.2s |
| CT E2E Analysis (optimized) | **15.67s** |
| Cellpose (GPU, single image) | ~3.2s |
| Drug recommendation generation | ~2.1s |
| Full pipeline | **< 90s** |

### Research Data Scale

| Data Type | Scale |
|-----------|-------|
| HUVEC cells analyzed | **43,190** |
| TCGA PrPc real samples | **2,285** |
| Literature knowledge base | **311 papers** |
| Inha CT cohort volume | 530Ã751Ã750 |
| Clinical samples (total) | **2,348** |

---

## ð§¬ 14-Dimensional Multimodal Feature Vector

```python
feature_vector = {
    # CT Radiomics (7D) â macroscopic imaging features
    "sphericity":          float,  # Tumor sphericity
    "energy":              float,  # GLCM texture energy
    "contrast":            float,  # Image contrast
    "tumor_size_mm2":      float,  # Tumor size (mmÂ²)
    "circularity":         float,  # Circularity
    "mean_hu":             float,  # Mean Hounsfield Units
    "detection_confidence":float,  # Detection confidence score

    # Cell Culture (7D) â microscopic cellular features
    "cell_density":        float,  # Cell density (cells/mmÂ²)
    "drug_resistance":     float,  # Drug resistance score
    "proliferation_score": float,  # Ki-67-based proliferation index
    "microenv_complexity": float,  # Microenvironment complexity
    "mean_circularity":    float,  # Mean cell circularity
    "clark_evans_index":   float,  # Spatial clustering index
    "estimated_viability": float,  # Estimated cell viability
}
```

---

## ð Installation

### System Requirements

| Item | Minimum | Recommended |
|------|---------|-------------|
| Python | 3.11 | 3.11+ |
| GPU | CUDA 11.x | CUDA 12.8 (RTX 50-series) |
| RAM | 16 GB | 32 GB |
| VRAM | 8 GB | 16 GB |
| Storage | 50 GB | 200 GB |

### Quick Installation

```bash
# 1. Clone the repository
git clone https://github.com/leejaeyoung-cpu/ADDS.git
cd ADDS

# 2. Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment variables
cp .env.example .env
# Edit .env: set OPENAI_API_KEY, DB_PATH, etc.

# 5. Initialize database
cd backend
python -c "from database_init import init_database; init_database()"
cd ..
```

### Running the System

```bash
# Method 1: Unified launch (recommended)
START_ALL.bat           # Starts backend (8000) + Streamlit UI (8505) simultaneously

# Method 2: Manual launch
# Terminal 1 â Backend
cd backend
uvicorn main:app --host 0.0.0.0 --port 8000 --reload

# Terminal 2 â Streamlit UI
python -m streamlit run src/ui/app.py --server.port 8505
```

> **Access URLs:**
> - ð¥ï¸ Clinical UI: `http://localhost:8505`
> - ð¡ API Server: `http://localhost:8000`
> - ð API Docs: `http://localhost:8000/docs`

### GPU Configuration (RTX 50-series / Blackwell)

```bash
# PyTorch Nightly (cu128 support)
pip install --pre torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/nightly/cu128

# Verify GPU
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, Device: {torch.cuda.get_device_name(0)}')"
```

---

## ð Data Structure

```
ADDS/
âââ ð src/                         â Core source modules
â   âââ adds/                       â ADDS inference engine
â   âââ medical_imaging/            â CT pipeline
â   â   âââ detection/              â Tumor detection (SimpleHUDetector)
â   â   âââ preprocessing/          â DICOM preprocessing
â   â   âââ radiomics/              â Radiomics feature extraction
â   â   âââ segmentation/           â Organ segmentation
â   âââ pathology/                  â Cellpose microscopy analysis
â   âââ clinical/                   â Clinical data management
â   âââ ml/                         â Machine learning models
â   â   âââ fusion/                 â Multimodal fusion
â   â   âââ survival/               â PFS/OS prediction
â   âââ protein/                    â PrPc protein analysis
â   âââ recommendation/             â Drug recommendation engine
â   âââ knowledge/                  â Knowledge base (311 papers)
â   âââ knowledge_base/             â Structured drug database
â   âââ reporting/                  â PDF report generation
â   âââ visualization/              â Data visualization
â   âââ xai/                        â Explainable AI (XAI)
â   âââ ui/                         â Streamlit UI components
â
âââ ð backend/                     â FastAPI backend
â   âââ main.py                     â Application entry point
â   âââ api/                        â REST API routers
â   â   âââ ct_analysis.py          â  /api/v1/ct
â   â   âââ patients.py             â  /api/v1/patients
â   â   âââ pharmacokinetics.py     â  /api/v1/pharmacokinetics
â   â   âââ adds_inference.py       â  /api/v1/adds
â   â   âââ openai_inference.py     â  /api/v1/openai
â   âââ services/                   â Business logic services
â   â   âââ ct_pipeline_service.py
â   â   âââ cell_culture_service.py
â   â   âââ adds_service.py
â   â   âââ openai_service.py
â   âââ models/                     â SQLAlchemy ORM models
â   âââ schemas/                    â Pydantic schemas
â
âââ ð analysis/                    â Research analysis scripts
â   âââ huvec/                      â HUVEC cell analysis
â   âââ ct/                         â CT analysis pipeline
â   âââ pritamab/                   â Pritamab drug synergy
â
âââ ð docs/                        â System documentation
âââ ð configs/                     â Configuration files
âââ ð tests/                       â Unit tests
âââ ð notebooks/                   â Jupyter analysis notebooks
âââ ð data/samples/                â Anonymized sample data
â
âââ ð³ Dockerfile                   â Container image
âââ ð³ docker-compose.yml           â Service orchestration
âââ ð requirements.txt             â Python dependencies
âââ ð pyproject.toml               â Project configuration
âââ ð .env.example                 â Environment variable template
```

---

## ð¡ API Reference

### Base URL

```
http://localhost:8000/api/v1
```

### Core Endpoints

| Method | Endpoint | Description |
|--------|---------|-------------|
| `GET` | `/health` | System health check |
| `GET` | `/patients` | List all patients |
| `POST` | `/patients` | Register new patient |
| `GET` | `/patients/{id}` | Get patient details |
| `POST` | `/ct/analyze` | Run CT DICOM analysis |
| `GET` | `/ct/health` | CT pipeline status |
| `GET` | `/ct/models/status` | nnU-Net model status |
| `POST` | `/pharmacokinetics/analyze` | Calculate PK parameters |
| `POST` | `/adds/infer` | ADDS pathway-based inference |
| `POST` | `/openai/infer` | GPT-4 clinical inference |

### CT Analysis Request Example

```python
import requests

# Upload DICOM file and analyze
with open("tumor_series.dcm", "rb") as f:
    response = requests.post(
        "http://localhost:8000/api/v1/ct/analyze",
        files={"dicom_file": f},
        data={"patient_id": "P-2026-001"}
    )

result = response.json()
print(f"Tumors detected: {result['tumors_detected']}")
print(f"Confidence: {result['confidence']:.2%}")
print(f"TNM estimate: {result['tnm_stage']}")
```

### PK Optimization Request Example

```python
pk_response = requests.post(
    "http://localhost:8000/api/v1/pharmacokinetics/analyze",
    json={
        "patient_id": "P-2026-001",
        "tumor_volume_mm3": 2450.5,
        "ki67_index": 45.2,
        "body_surface_area": 1.73
    }
)

pk = pk_response.json()
print(f"Optimal dose: {pk['optimal_dose_mg_m2']} mg/mÂ²")
print(f"Half-life: {pk['half_life_hours']:.1f} hours")
print(f"Dosing interval: {pk['dosing_interval_hours']} hours")
```

---

## ð§ª Research Background

### PrPc Biomarker Discovery Journey

| Version | Strategy | Cohort | Goal | Result |
|---------|----------|--------|------|--------|
| v1.0 | Single marker (serum) | n=63 | Stage III CRC | â Gap discovered |
| v2.0 | Multi-marker panel | 20â30 | General GI cancer | ð Strategy pivot |
| **v3.0** | **AI-First / National Biodata** | **n=300â800** | **Early detection** | â **In progress** |

### Knowledge Base Composition (as of February 2026)

```
Literature Knowledge Base v2.0
âââ Tier 1 (100 papers): Nature / Cell / Science / Nature Medicine
âââ Tier 2 (100 papers): JCO / Cancer Research
âââ Tier 3: The Biology of Cancer (Weinberg)

Statistics:
â¢ 311 papers (abstract-based GPT-4 extraction)
â¢ 2,285 real TCGA samples (BRCA, STAD, COAD, PAAD, READ)
â¢ 113 drugs / 90 mechanisms / 69 biomarkers
â¢ 59 synergy combinations validated
```

### Clinical Pilot Protocol

```
Pilot Study Design (v1.0)
â¢ Design: Prospective pilot, N=100 (50 cases, 50 controls)
â¢ Objective: Stage I 30% + Stage II 30% (early detection)
â¢ Go/No-Go criterion: AUC â¥ 0.75

3-Month Roadmap:
â¢ Month 1: IRB submission + account setup
â¢ Month 2: Approval + site activation
â¢ Month 3: Enrollment + Go/No-Go decision
```

---

## â ï¸ Data Availability

Patient CT data and raw microscopy images are **NOT included** in this repository:

- ð **PHI Regulations** (Protected Health Information)
- ð **File Size Limit**: GitHub 100MB limit (CT volumes are several GB)
- ð¥ **Institutional Approval Required**: Inha University Hospital IRB-approved data

For data access to reproduce results, please contact the authors.  
The `data/samples/` directory contains only anonymized small-scale samples.

---

## â ï¸ Methodological Notes

> **Transparency Statement**: All performance metrics are reported with their methodological context and limitations. This section is intended to support scientific reproducibility and honest evaluation.

### CT Tumor Detection (98.65% Accuracy)

| Item | Detail |
|------|--------|
| **Dataset** | Inha University Hospital CRC cohort |
| **Sample size** | N = 74 CT slices (single patient, arterial phase) |
| **Method** | HU-threshold (60â120 HU) + morphological filtering + connected-component analysis |
| **Ground truth** | Manual annotation by clinical radiologist |
| **Metric** | Slice-level detection accuracy (correct slices / total slices) |
| **95% CI** | [0.949, 1.000] (Wilson score interval) |
| **â ï¸ Limitation** | Single-patient pilot study. Multi-center validation with Nâ¥200 patients is ongoing. This metric does NOT represent patient-level diagnostic accuracy. |

### Cell Morphometry (N = 43,190 cells)

| Item | Detail |
|------|--------|
| **Instrument** | Brightfield microscopy |
| **Cell lines** | HUVEC (Human Umbilical Vein Endothelial Cells) |
| **Conditions** | 4 groups: Control Â· Healthy Serum Â· HGPS Serum Â· HGPS + MT-Exosome |
| **Images analyzed** | 80 brightfield images |
| **Segmentation** | Cellpose v3 (cyto3 model), GPU-accelerated |
| **â ï¸ Limitation** | In vitro model only. Clinical relevance requires PDO (Patient-Derived Organoid) validation. |

### Drug Synergy Models (TCGA N = 2,285)

| Item | Detail |
|------|--------|
| **Training data** | TCGA-COAD + DrugComb + OncoKB |
| **Synergy metrics** | Bliss Independence, Loewe Additivity, HSA, ZIP |
| **Model architecture** | DeepSynergy v2 (DNN) + XGBoost ensemble |
| **Validation** | 5-fold cross-validation on held-out TCGA subset |
| **â ï¸ Limitation** | Synergy predictions are based on genomic/transcriptomic features. Prospective clinical validation has not been conducted. Not for clinical use without regulatory approval. |

### Reproducibility

```bash
# Verify core scientific logic (no GPU required)
pip install -r requirements-ci.txt
python -m pytest tests/test_science_core.py -v
# Expected: 18 passed
```

All statistical tests, synergy formulas, and data integrity checks in `tests/test_science_core.py` pass with zero external dependencies.

---

## ð Citation

If you use this code in your research, please cite:

```bibtex
@misc{adds2026,
  title     = {ADDS: AI-Driven Drug Synergy and Diagnostic System â
               A Multimodal Precision Oncology Platform},
  author    = {Lee, Jaeyoung and others},
  year      = {2026},
  url       = {https://github.com/leejaeyoung-cpu/ADDS},
  note      = {Inha University Hospital, Incheon, Korea}
}
```

---

## ð¤ Contributing

Contributions are welcome! See [CONTRIBUTING.md](.github/CONTRIBUTING.md) for detailed guidelines.

**Quick contribution guide:**
1. `Fork` â Create `Feature Branch` (`feat/my-feature`)
2. Write changes + add tests
3. Create `Pull Request` (fill in PR template)

---

## ð Security

If you discover a security vulnerability, please do NOT create a public issue. Instead, follow the private disclosure guidelines in [SECURITY.md](.github/SECURITY.md).

---

## ð¬ Contact

| Item | Details |
|------|---------|
| **Repository** | [github.com/leejaeyoung-cpu/ADDS](https://github.com/leejaeyoung-cpu/ADDS) |
| **Institution** | Inha University Hospital, Incheon, Republic of Korea |
| **Research Area** | Precision Oncology / AI Medical Device (SaMD) |
| **Target Journal** | Nature Communications |

---

<div align="center">

**ADDS v3.5.0** â Built with â¤ï¸ for Precision Oncology  
Inha University Hospital Ã AI Research Team | 2026

</div>
