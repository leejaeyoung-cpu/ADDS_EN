# ADDS - AI-Driven Drug Synergy and Diagnostic System

**Multimodal AI Platform for Precision Oncology** | v3.6.0 | Inha University Hospital

[![Python](https://img.shields.io/badge/Python-3.11+-3776AB)](https://www.python.org/)
[![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.22302648-blue)](https://doi.org/10.5281/zenodo.22302648)

ADDS integrates CT radiomics, cellular morphometry, pharmacokinetic modelling, and machine learning into a unified platform for personalized anticancer drug-cocktail recommendation in colorectal cancer.

## System Overview

ADDS is a precision oncology AI ecosystem. The framework fuses a **480-dimensional** multimodal feature vector: CT (64D) + Transcriptomic (256D) + Cell imaging (128D) + PK/PD physics (32D).

## R1 Manuscript Reproducibility

All virtual-cohort outputs are **model-based projections** and should not be interpreted as prospectively observed clinical outcomes. (Digital Medicine, September 2026)

### Composite Ranking Score

    Score = 0.35 * E_pred + 0.15 * S_pred - 0.10 * (T_tox / 10)

E_pred (0-1): efficacy prediction. S_pred (0-2): synergy prediction. T_tox (1-10): toxicity burden. Role: ranking/prioritisation only.

### Survival-Projection Blend

Primary blend: 70:30. Regimen-prioritisation ranking invariant across all tested sensitivity weights (Spearman rho=1.0, Kendall tau=1.0).

### Bootstrap Stability

| Panel | Analysis | n |
|---|---|---|
| A | Bootstrap top-3 retention | N=5,000 resamples |
| B | Perturbation rank correlation | N=2,000 replicates |
| C | Modality dropout ablation | N=3,000 repetitions |

### Biomarker Definitions

PrPc-high: IHC H-score >= 50. KRAS-mutant: codons 12/13 (G12D, G12V, G12C, G13D).

## Performance Metrics

| Metric | Value |
|---|---|
| CT Detection Accuracy | 98.65% |
| Processing Time | 15.67 s |
| HUVEC Cells Analysed | 43,190 |
| TCGA PrPc Samples | 2,285 |
| Literature KB | 311 papers |

## Installation


    git clone https://github.com/leejaeyoung-cpu/ADDS_EN.git
    cd ADDS_EN
    python -m venv .venv
    pip install -r requirements.txt

## Data Availability

Zenodo archive (v3.6.0): https://doi.org/10.5281/zenodo.22302648

GitHub (Korean): https://github.com/leejaeyoung-cpu/ADDS

## Citation

    @misc{adds2026,
      title  = {ADDS: AI-Driven Drug Synergy and Diagnostic System},
      author = {Lee, Jaeyoung and others},
      year   = {2026},
      url    = {https://github.com/leejaeyoung-cpu/ADDS_EN},
      note   = {v3.6.0, DOI: 10.5281/zenodo.22302648}
    }

ADDS v3.6.0 - Inha University Hospital x AI Research Team - 2026
