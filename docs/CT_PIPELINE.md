# CT Analysis Pipeline

**6-Stage 3D CT Tumor Detection and Radiomics Analysis Pipeline**

---

## Overview

The ADDS CT pipeline is an end-to-end clinical analysis system that takes DICOM CT series from colorectal cancer (CRC) suspected patients and predicts tumor location, size, radiological features, biomarkers, and drug sensitivity through 6 stages.

**Validated Performance**: **98.65%** tumor detection accuracy on Inha University Hospital cohort  
**Processing Speed**: **15.67 seconds** / patient (530Ã751Ã750 volume)

---

## Stage 1: 3D Volume Reconstruction

**Goal**: DICOM series â 1mmÂ³ isotropic NIfTI volume conversion

```python
# Core settings
target_spacing = (1.0, 1.0, 1.0)  # mmÂ³ isotropic
normalize = False  # â ï¸ CRITICAL: Must preserve absolute HU values!
orientation = "axial"  # Re-orient to standard direction
```

> â ï¸ **Important**: Setting `normalize=True` distorts HU values to the [0, 50] range,  
> causing the Stage 3 detector to fail completely. Never change this.

**Processing Steps**:
1. Read DICOM series (SimpleITK `ImageSeriesReader`)
2. Spacing resampling (Simpson/B-spline interpolation)
3. Orientation standardization (LPS â RAS or Axial re-orientation)
4. Save as NIfTI format (`data/outputs/{run_id}/volume.nii.gz`)

**Validation Checkpoint**:
```python
# Verify HU range (normal: -1000 ~ +3000)
volume = sitk.ReadImage("volume.nii.gz")
array = sitk.GetArrayFromImage(volume)
assert array.min() < -500, f"HU min too high: {array.min()}"
assert array.max() > 0, f"HU max too low: {array.max()}"
```

---

## Stage 2: Anatomical Organ Segmentation

**Goal**: Segment organs of interest (colon, liver, lymph nodes)

**Engine**: nnU-Net v2

```python
# nnU-Net execution example
nnUNetv2_predict \
    -i input_folder/ \
    -o output_folder/ \
    -d Dataset001_CRC \
    -c 3d_fullres \
    -f all
```

**Target Organs**:
| Organ | Label | Clinical Importance |
|-------|-------|-------------------|
| Colon | 1 | Direct analysis target |
| Liver | 2 | Metastasis assessment |
| Lymph Nodes | 3 | TNM N Stage |
| Rectum | 4 | Primary site differentiation |

**Fallback Engines**: When nnU-Net model is unavailable, uses TotalSegmentator or SAM (SegVol)

---

## Stage 3: Tumor Detection â Core Stage

**Goal**: Detect and localize malignant lesion candidates

### VerifiedCTDetector (Production Engine)

```python
class SimpleHUDetector:
    """
    Production detector validated at 98.65% accuracy
    (Inha University Hospital cohort: 73/74 slices)
    
    Algorithm: Per-slice 2D detection (bypasses 3D CCA bottleneck)
    """
    
    HU_MIN = 60   # Arterial phase lower bound
    HU_MAX = 120  # Arterial phase upper bound
    
    def detect(self, volume: np.ndarray) -> List[TumorCandidate]:
        results = []
        for z, axial_slice in enumerate(volume):
            # 1. HU range masking
            mask = (axial_slice >= self.HU_MIN) & (axial_slice <= self.HU_MAX)
            
            # 2. Morphological noise removal
            mask = morphology.remove_small_objects(mask, min_size=30)
            mask = morphology.closing(mask, morphology.disk(2))
            
            # 3. Component labeling
            labeled = label(mask)
            
            # 4. Size filtering (50 mmÂ³ = ~50 pixels @ 1mm spacing)
            for region in regionprops(labeled):
                if region.area >= 50:
                    results.append(self._score_candidate(region, z))
        return results
```

**HU Threshold Rationale**:
| HU Range | Tissue Type | Clinical Meaning |
|----------|------------|-----------------|
| -1000 ~ -100 | Air | Excluded |
| -100 ~ -10 | Fat | Excluded |
| -10 ~ 60 | Soft tissue (normal) | Excluded |
| **60 ~ 120** | **Malignant lesion (arterial phase)** | **â Detection target** |
| 120 ~ 250 | Muscle/vessels | Excluded |
| 250+ | Bone | Excluded |

**Output Format**:
```python
{
    "tumors_detected": 3,
    "candidates": [
        {
            "z_slice": 370,
            "centroid_xy": [245, 312],
            "area_mm2": 847,
            "mean_hu": 89.3,
            "confidence": 0.94,
            "overlay_image_b64": "..."  # Base64 visualization
        }
    ]
}
```

---

## Stage 4: Radiomics Extraction

**Goal**: Extract 100+ phenotypic features (PyRadiomics)

```python
extractor = featureextractor.RadiomicsFeatureExtractor()
extractor.enableAllFeatures()

features = {
    # First Order Statistics
    "Energy": ...,
    "Entropy": ...,       # Tissue heterogeneity
    "Uniformity": ...,
    
    # Shape Features
    "Sphericity": ...,    # Sphericity (malignant: low)
    "SurfaceArea": ...,
    "Circularity": ...,
    
    # GLCM Texture
    "Contrast": ...,      # Boundary sharpness
    "Correlation": ...,
    "DifferenceEntropy": ...,
    
    # GLRLM, GLDM, etc...
}
```

**Library Stability Notes**:
```python
# Pin numpy version (required)
# pip install numpy==1.26.4
# (PyRadiomics C++ binding compatibility)

# SimpleITK constant protection pattern
try:
    import SimpleITK as sitk
    SITK_BSPLINE = sitk.sitkBSpline
except ImportError:
    SITK_BSPLINE = 3  # Direct integer fallback
```

---

## Stage 5: Biomarker Prediction

**Goal**: Map imaging features â molecular biomarker predictions

| Biomarker | Prediction Method | Clinical Use |
|-----------|-----------------|--------------|
| **TNM Stage** | Size + location rule-based | Treatment planning |
| **MSI Status** | Texture feature-based | Immune checkpoint inhibitor indication |
| **KRAS Mutation** | Radiomics pattern | EGFR inhibitor exclusion |
| **Ki-67 Index** | Density + morphology estimation | Proliferation rate classification |
| **Malignancy Score** | Composite score [0-1] | Malignancy grading |

**TNM Decision Rules Example**:
```python
def predict_tnm(tumor_size_mm, location, lymph_nodes):
    if tumor_size_mm < 20:
        t_stage = "T1"
    elif tumor_size_mm < 50:
        t_stage = "T2"
    elif tumor_size_mm < 70:
        t_stage = "T3"
    else:
        t_stage = "T4"
    
    n_stage = "N0" if lymph_nodes == 0 else f"N{min(lymph_nodes, 2)}"
    return f"{t_stage}{n_stage}"
```

---

## Stage 6: ADDS Integration

**Goal**: Radiomics results â drug sensitivity model mapping

```python
# Stage 6 core output
adds_output = {
    "recommended_regimen": "FOLFOX",
    "secondary_regimen": "FOLFIRI + Bevacizumab",
    "predicted_response_rate": 0.80,   # 80%
    "mechanism": "KRAS G12D â MEK/ERK pathway blockade",
    "biomarker_rationale": {
        "KRAS": "WT â EGFR inhibitor eligible",
        "MSS": "Immune checkpoint inhibitor limited"
    }
}
```

**Inha University Hospital Case Validation Result**:
```
Patient: Stage II (T4N0)
Tumor: Sigmoid Colon, 67mm
KRAS: G12D (Mutant)
Predicted Response: FOLFOX 80% (6-month PFS)
Actual Outcome: FOLFOX first-line treatment â favorable response
```

---

## Running the Full Pipeline

```bash
# Single patient analysis
python ct_pipeline_v4.py --dicom_dir /path/to/dicom/ --patient_id P-2026-001

# Batch processing
python batch_tumor_detection_dcm.py \
    --input_dir /data/ct_cases/ \
    --output_dir /data/results/ \
    --use_verified_detector True

# API-based analysis
curl -X POST http://localhost:8000/api/v1/ct/analyze \
    -F "dicom_files=@tumor_series.dcm" \
    -F "patient_id=P-2026-001"
```

---

## Troubleshooting

| Symptom | Cause | Solution |
|---------|-------|---------|
| "No candidates found" | HU normalization error | Check `normalize=False` |
| CCA taking 15+ minutes | 3D CCA with 26K+ components | Set `USE_2D_SLICE=True` |
| PyRadiomics ImportError | numpy version conflict | `pip install numpy==1.26.4` |
| GPU OOM | Volume too large | Set `target_spacing=(2.0,2.0,2.0)` |

---

*Reference: [ARCHITECTURE.md](ARCHITECTURE.md) | [API_REFERENCE.md](API_REFERENCE.md)*
