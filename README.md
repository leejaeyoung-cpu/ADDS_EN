<div align="center">

![ADDS Version](https://img.shields.io/badge/ADDS-v3.6.0-blueviolet?style=for-the-badge&logo=python)

# ADDS - AI-Driven Drug Synergy & Diagnostic System

**Multimodal AI Platform for Precision Oncology**

[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x_GPU-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)
[![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.22302648-blue)](https://doi.org/10.5281/zenodo.22302648)
[![Institution](https://img.shields.io/badge/Institution-Inha_University_Hospital-003DA5)](https://www.inha.com/)

<blockquote><strong>ADDS</strong>strong> integrates CT radiomics, cellular morphometry, pharmacokinetic modelling, and machine learning into a unified platform for personalized anticancer drug-cocktail recommendation in colorectal cancer (CRC).</blockquote>blockquote>

</div>

---

## System Overview

ADDS (AI-Driven Drug Synergy) is a **precision oncology AI ecosystem** developed in collaboration with Inha University Hospital. The framework fuses a **480-dimensional** multimodal feature vector across CT radiomics, transcriptomics, PK/PD physics, and cellular imaging to rank candidate treatment regimens.

### Key Innovations

| Innovation | Description |
|---|---|
| **480-D Multimodal Fusion** | CT (64D) + Transcriptomic/biomarker (256D) + Cell imaging (128D) + PK/PD physics (32D) |
| **Dual Inference Engine** | ADDS pathway-based engine + OpenAI GPT-4 concurrent cross-validation |
| **PrPc Biomarker Discovery** | Novel KRAS-RPSA signalosome biomarker, TCGA validated (n=2,285) |
| **Real-time Clinical Application** | End-to-end analysis in 15.67 seconds (530x751x750 CT volume) |

---

## R1 Manuscript Reproducibility

All virtual-cohort outputs are **model-based projections** and should not be interpreted as prospectively observed clinical outcomes. (*Digital Medicine*, September 2026)

### Composite Ranking Score

```
Score = 0.35 * E_pred + 0.15 * S_pred - 0.10 * (T_tox / 10)
```

- `E_pred` (0-1): efficacy prediction
- - `S_pred` (0-2): synergy prediction
  - - `T_tox` (1-10): toxicity burden
    - - Role: **ranking/prioritisation only** - not a clinically calibrated HR surrogate
      -
      - ### Survival-Projection Blend: 70:30
      -
      - Regimen-prioritisation ranking invariant across all sensitivity weights (Spearman rho=1.0, Kendall tau=1.0).
      -
      - ### Bootstrap Stability (Figure 4)
      -
      - | Panel | Analysis | n |
      - |---|---|---|
      - | A | Bootstrap top-3 retention | N=5,000 resamples |
      - | B | Perturbation rank correlation | N=2,000 replicates |
      - | C | Modality dropout ablation | N=3,000 repetitions |
      -
      - ### Biomarker Definitions
      -
      - - **PrPc-high:** IHC H-score >= 50
        - - **KRAS-mutant:** codons 12/13 (G12D, G12V, G12C, G13D)
          -
          - ---
          -
          - ## Performance Metrics
          -
          - | Metric | Value |
          - |---|---|
          - | CT Detection Accuracy | **98.65%** (74 slices, single-patient pilot) |
          - | Processing Time (E2E) | **15.67 s** (530x751x750 volume) |
          - | HUVEC Cells Analysed | **43,190** |
          - | TCGA PrPc Samples | **2,285** |
          - | Literature KB | **311 papers** |
          -
          - <blockquote><strong>Note:</strong>strong> CT accuracy is from a single-patient pilot study. Multi-centre validation (N &gt;= 200) is ongoing.</blockquote>blockquote>

          ---

          ## Installation

          ```bash
          git clone https://github.com/leejaeyoung-cpu/ADDS_EN.git
          cd ADDS_EN
          python -m venv .venv && source .venv/bin/activate
          pip install -r requirements.txt
          ```

          ---

          ## Data Availability

          The ADDS framework (v3.6.0) is publicly archived at:

          **Zenodo DOI:** [10.5281/zenodo.22302648](https://doi.org/10.5281/zenodo.22302648)

          **GitHub (Korean):** [leejaeyoung-cpu/ADDS](https://github.com/leejaeyoung-cpu/ADDS)

          ---

          ## Citation

          ```bibtex
          @misc{adds2026,
            title  = {ADDS: AI-Driven Drug Synergy and Diagnostic System},
            author = {Lee, Jaeyoung and others},
            year   = {2026},
            url    = {https://github.com/leejaeyoung-cpu/ADDS_EN},
            note   = {v3.6.0, DOI: 10.5281/zenodo.22302648, Inha University Hospital}
          }
          ```

          ---

          <div align="center">
       
           **ADDS v3.6.0** | Inha University Hospital x AI Research Team | 2026
       
          </div>
          </strong></blockquote>
