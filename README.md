# ADDS - AI-Driven Drug Synergy and Diagnostic System

**Multimodal AI Platform for Precision Oncology** | v3.6.0 | Inha University Hospital

[![Python](https://img.shields.io/badge/Python-3.11+-3776AB)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.22302648-blue)](https://doi.org/10.5281/zenodo.22302648)

ADDS integrates CT radiomics, cellular morphometry, pharmacokinetic modelling, and machine learning into a unified platform for personalized anticancer drug-cocktail recommendation in colorectal cancer.

---

## System Overview

ADDS is a precision oncology AI ecosystem developed in collaboration with Inha University Hospital. The framework fuses a **480-dimensional** multimodal feature vector across CT radiomics, transcriptomics, PK/PD physics, and cellular imaging to rank candidate treatment regimens.

### Key Innovations

| Innovation | Description |
|---|---|
| 480-D Multimodal Fusion | CT (64D) + Transcriptomic (256D) + Cell imaging (128D) + PK/PD physics (32D) |
| Dual Inference Engine | ADDS pathway-based engine + OpenAI GPT-4 cross-validation |
| PrPc Biomarker Discovery | Novel KRAS-RPSA signalosome biomarker, TCGA validated (n=2,285) |
| Real-time Clinical Application | End-to-end analysis in 15.67 seconds (530x751x750 CT volume) |

---

## R1 Manuscript Reproducibility

All virtual-cohort outputs are **model-based projections** and should not be interpreted as prospectively observed clinical outcomes. (Digital Medicine, September 2026)

### Composite Ranking Score

```
Score = 0.35 * E_pred + 0.15 * S_pred - 0.10 * (T_tox / 10)
```

- **E_pred (0-1)**: efficacy prediction
- **S_pred (0-2)**: synergy prediction
- **T_tox (1-10)**: toxicity burden
- **Role**: ranking/prioritisation only, not a clinically calibrated HR surrogate

- ### Survival-Projection Blend

- Primary blend: 70:30. Regimen-prioritisation ranking invariant across all tested sensitivity weights (Spearman rho=1.0, Kendall tau=1.0).

- ### Bootstrap Stability (Figure 4)

- | Panel | Analysis | n |
- |---|---|---|
- | A | Bootstrap top-3 retention | N=5,000 resamples |
- | B | Perturbation rank correlation | N=2,000 replicates |
- | C | Modality dropout ablation | N=3,000 repetitions |

- ### Biomarker Definitions

- - **PrPc-high**: IHC H-score >= 50
  - **KRAS-mutant**: codons 12/13 activating mutations (G12D, G12V, G12C, G13D)
 
  - ---

  ## Performance Metrics

  | Metric | Value |
  |---|---|
  | CT Detection Accuracy | 98.65% (74 slices, single-patient pilot) |
  | Processing Time (E2E) | 15.67 s (530x751x750 volume) |
  | HUVEC Cells Analysed | 43,190 |
  | TCGA PrPc Samples | 2,285 |
  | Literature KB | 311 papers |

  Note: CT accuracy is from a single-patient pilot study. Multi-centre validation is ongoing.

  ---

  ## Installation

  ```bash
  git clone https://github.com/leejaeyoung-cpu/ADDS_EN.git
  cd ADDS_EN
  python -m venv .venv
  source .venv/bin/activate
  pip install -r requirements.txt
  ```

  ---

  ## Data Availability

  Zenodo archive (v3.6.0): https://doi.org/10.5281/zenodo.22302648

  GitHub (Korean version): https://github.com/leejaeyoung-cpu/ADDS

  Reproducibility materials: See paper_release/r1_2026-09/ directory for ANALYTICAL_SPECIFICATION_R1.md, DATA_PROVENANCE_R1.csv, MANIFEST_R1.tsv

  ---

  ## Citation

  ```bibtex
  @misc{adds2026,
    title  = {ADDS: AI-Driven Drug Synergy and Diagnostic System},
    author = {Lee, Jaeyoung and others},
    year   = {2026},
    url    = {https://github.com/leejaeyoung-cpu/ADDS_EN},
    note   = {v3.6.0, DOI: 10.5281/zenodo.22302648}
  }
  ```

  ---

  ADDS v3.6.0 - Inha University Hospital x AI Research Team - 2026
