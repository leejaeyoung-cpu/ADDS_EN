#!/usr/bin/env python3
"""
Advanced Lesion Detection System
Detect and classify lesions within each organ using multi-modal features
"""

import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from pathlib import Path
import cv2
from typing import Dict, List, Tuple
from scipy import ndimage
from skimage.feature import graycomatrix, graycoprops
from skimage.measure import label, regionprops
import json

def load_nifti(file_path: Path) -> np.ndarray:
    """Load NIfTI file"""
    nii = nib.load(str(file_path))
    return nii.get_fdata()

def extract_lesion_features(ct_image: np.ndarray, lesion_mask: np.ndarray, organ_mask: np.ndarray, baseline: Dict) -> Dict:
    """
    Extract comprehensive lesion features
    
    Features:
    - Intensity: mean, std, min, max HU
    - Shape: area, perimeter, circularity, solidity
    - Texture: GLCM features (contrast, homogeneity, correlation)
    - Context: distance to organ border, relative position
    """
    if lesion_mask.sum() == 0:
        return None
    
    # Intensity features
    lesion_pixels = ct_image[lesion_mask > 0]
    
    intensity_features = {
        'mean_hu': float(np.mean(lesion_pixels)),
        'std_hu': float(np.std(lesion_pixels)),
        'min_hu': float(np.min(lesion_pixels)),
        'max_hu': float(np.max(lesion_pixels)),
        'median_hu': float(np.median(lesion_pixels)),
        'range_hu': float(np.max(lesion_pixels) - np.min(lesion_pixels)),
    }
    
    # Deviation from baseline
    if baseline:
        intensity_features['deviation_from_mean'] = abs(intensity_features['mean_hu'] - baseline['mean_hu'])
        intensity_features['z_score'] = (intensity_features['mean_hu'] - baseline['mean_hu']) / (baseline['std_hu'] + 1e-6)
    
    # Shape features
    props = regionprops(lesion_mask.astype(int))[0]
    
    shape_features = {
        'area': float(props.area),
        'perimeter': float(props.perimeter),
        'circularity': 4 * np.pi * props.area / (props.perimeter ** 2 + 1e-6),
        'solidity': float(props.solidity),
        'eccentricity': float(props.eccentricity),
        'major_axis': float(props.major_axis_length),
        'minor_axis': float(props.minor_axis_length),
    }
    
    # Texture features (GLCM)
    # Normalize to 0-255 for GLCM
    ct_norm = cv2.normalize(ct_image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    lesion_region = ct_norm * lesion_mask
    
    # Extract region
    ys, xs = np.where(lesion_mask > 0)
    if len(ys) > 5 and len(xs) > 5:
        y_min, y_max = ys.min(), ys.max() + 1
        x_min, x_max = xs.min(), xs.max() + 1
        roi = lesion_region[y_min:y_max, x_min:x_max]
        
        # Compute GLCM
        try:
            glcm = graycomatrix(roi, distances=[1], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4],
                              levels=256, symmetric=True, normed=True)
            
            texture_features = {
                'contrast': float(graycoprops(glcm, 'contrast').mean()),
                'dissimilarity': float(graycoprops(glcm, 'dissimilarity').mean()),
                'homogeneity': float(graycoprops(glcm, 'homogeneity').mean()),
                'energy': float(graycoprops(glcm, 'energy').mean()),
                'correlation': float(graycoprops(glcm, 'correlation').mean()),
                'ASM': float(graycoprops(glcm, 'ASM').mean()),
            }
        except:
            texture_features = {
                'contrast': 0, 'dissimilarity': 0, 'homogeneity': 0,
                'energy': 0, 'correlation': 0, 'ASM': 0
            }
    else:
        texture_features = {
            'contrast': 0, 'dissimilarity': 0, 'homogeneity': 0,
            'energy': 0, 'correlation': 0, 'ASM': 0
        }
    
    # Context features
    # Distance to organ border
    organ_distance_map = ndimage.distance_transform_edt(organ_mask)
    lesion_distances = organ_distance_map[lesion_mask > 0]
    
    context_features = {
        'min_distance_to_border': float(lesion_distances.min()) if len(lesion_distances) > 0 else 0,
        'mean_distance_to_border': float(lesion_distances.mean()) if len(lesion_distances) > 0 else 0,
    }
    
    # Combine all features
    features = {
        **intensity_features,
        **shape_features,
        **texture_features,
        **context_features
    }
    
    return features

def classify_lesion_heuristic(features: Dict, organ_name: str) -> Tuple[str, float]:
    """
    Heuristic rule-based classification
    
    Returns: (class_name, confidence)
    
    Classes:
    - malignant_tumor: Cancer
    - inflammation: Inflammatory lesion
    - benign_lesion: Benign mass
    - cyst: Fluid-filled cyst
    - artifact: Imaging artifact / normal variation
    """
    
    if not features:
        return 'unknown', 0.0
    
    # Extract key features
    mean_hu = features.get('mean_hu', 0)
    z_score = abs(features.get('z_score', 0))
    circularity = features.get('circularity', 0)
    contrast = features.get('contrast', 0)
    homogeneity = features.get('homogeneity', 1)
    area = features.get('area', 0)
    
    # Rule-based classification
    scores = {
        'malignant_tumor': 0,
        'inflammation': 0,
        'benign_lesion': 0,
        'cyst': 0,
        'artifact': 0
    }
    
    # Cyst indicators (low HU, high homogeneity, circular)
    if mean_hu < 20 and homogeneity > 0.8 and circularity > 0.7:
        scores['cyst'] = 0.85
    
    # Malignant tumor indicators
    # - Moderate to high HU (enhanced)
    # - Irregular shape (low circularity)
    # - High contrast (heterogeneous)
    # - Large size
    if mean_hu > 60 and circularity < 0.6 and contrast > 80:
        scores['malignant_tumor'] = 0.75
        if area > 200:  # Large lesion
            scores['malignant_tumor'] += 0.1
    
    # Inflammation indicators
    # - Moderate HU
    # - High heterogeneity
    # - Irregular borders
    if 30 < mean_hu < 80 and homogeneity < 0.5 and contrast > 50:
        scores['inflammation'] = 0.70
    
    # Benign lesion indicators
    # - Well-defined borders (circular)
    # - Moderate size
    # - Relatively homogeneous
    if 0.6 < circularity < 0.9 and 50 < area < 500 and homogeneity > 0.6:
        scores['benign_lesion'] = 0.65
    
    # Artifact indicators
    # - Very small
    # - Extreme HU values
    # - Very high or very low z-score
    if area < 50 or z_score < 1.5 or abs(mean_hu) > 1000:
        scores['artifact'] = 0.90
    
    # Return highest scoring class
    best_class = max(scores.items(), key=lambda x: x[1])
    
    # If all scores are low, mark as unknown
    if best_class[1] < 0.3:
        return 'unknown', 0.0
    
    return best_class

def detect_and_classify_lesions(
    ct_slice: np.ndarray,
    organ_mask: np.ndarray,
    organ_name: str,
    baseline: Dict
) -> List[Dict]:
    """
    Detect and classify all lesions within an organ
    """
    if organ_mask.sum() == 0 or baseline is None:
        return []
    
    # Z-score based anomaly detection
    mean_hu = baseline['mean_hu']
    std_hu = baseline['std_hu']
    
    z_scores = np.zeros_like(ct_slice)
    organ_pixels = organ_mask > 0
    z_scores[organ_pixels] = (ct_slice[organ_pixels] - mean_hu) / (std_hu + 1e-6)
    
    # Detect high and low intensity anomalies
    high_threshold = 3.0
    low_threshold = -3.0
    
    high_anomalies = (z_scores > high_threshold) & organ_pixels
    low_anomalies = (z_scores < low_threshold) & organ_pixels
    
    # Combine
    all_anomalies = high_anomalies | low_anomalies
    
    # Clean up small artifacts
    kernel = np.ones((3, 3), np.uint8)
    all_anomalies = cv2.morphologyEx(all_anomalies.astype(np.uint8), cv2.MORPH_OPEN, kernel)
    all_anomalies = cv2.morphologyEx(all_anomalies, cv2.MORPH_CLOSE, kernel)
    
    # Label connected components
    labeled = label(all_anomalies)
    
    # Analyze each lesion
    lesions = []
    for region in regionprops(labeled):
        # Create individual lesion mask
        lesion_mask = (labeled == region.label).astype(np.uint8)
        
        # Extract features
        features = extract_lesion_features(ct_slice, lesion_mask, organ_mask, baseline)
        
        if features:
            # Classify
            lesion_class, confidence = classify_lesion_heuristic(features, organ_name)
            
            # Store lesion info
            lesion_info = {
                'mask': lesion_mask,
                'features': features,
                'class': lesion_class,
                'confidence': confidence,
                'organ': organ_name,
                'centroid': region.centroid
            }
            
            lesions.append(lesion_info)
    
    return lesions

def visualize_lesion_detection(
    ct_slice: np.ndarray,
    organ_masks: Dict[str, np.ndarray],
    lesions: List[Dict],
    slice_idx: int,
    output_dir: Path
):
    """
    Create advanced visualization with lesion classification
    """
    
    # Color scheme for lesion classes
    LESION_COLORS = {
        'malignant_tumor': (255, 0, 0),      # Bright Red
        'inflammation': (255, 255, 0),       # Yellow
        'benign_lesion': (0, 255, 0),        # Green
        'cyst': (0, 255, 255),               # Cyan
        'artifact': (128, 128, 128),         # Gray
        'unknown': (255, 0, 255)             # Magenta
    }
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 16))
    
    # 1. Original CT
    ax = axes[0, 0]
    ax.imshow(ct_slice, cmap='gray', vmin=-200, vmax=200)
    ax.set_title('Original CT', fontsize=14, fontweight='bold')
    ax.axis('off')
    
    # 2. Organs overlay
    ax = axes[0, 1]
    ct_norm = cv2.normalize(ct_slice, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    overlay = cv2.cvtColor(ct_norm, cv2.COLOR_GRAY2RGB)
    
    # Add organ outlines
    for organ_name, mask in organ_masks.items():
        if mask.sum() > 0:
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(overlay, contours, -1, (0, 255, 0), 2)
    
    ax.imshow(overlay)
    ax.set_title('Organ Boundaries', fontsize=14, fontweight='bold')
    ax.axis('off')
    
    # 3. Lesion detection
    ax = axes[1, 0]
    lesion_overlay = cv2.cvtColor(ct_norm, cv2.COLOR_GRAY2RGB)
    
    for lesion in lesions:
        color = LESION_COLORS.get(lesion['class'], (255, 0, 255))
        mask = lesion['mask']
        lesion_overlay[mask > 0] = (lesion_overlay[mask > 0] * 0.3 + np.array(color) * 0.7).astype(np.uint8)
    
    ax.imshow(lesion_overlay)
    ax.set_title(f'Lesion Detection ({len(lesions)} found)', fontsize=14, fontweight='bold')
    ax.axis('off')
    
    # 4. Classification summary
    ax = axes[1, 1]
    ax.axis('off')
    
    # Create text summary
    summary_text = "LESION CLASSIFICATION SUMMARY\n"
    summary_text += "="*40 + "\n\n"
    

    class_counts = {}
    for lesion in lesions:
        cls = lesion['class']
        class_counts[cls] = class_counts.get(cls, 0) + 1
    
    for cls, count in sorted(class_counts.items(), key=lambda x: -x[1]):
        color_hex = '#{:02x}{:02x}{:02x}'.format(*LESION_COLORS.get(cls, (255, 0, 255)))
        summary_text += f"{cls.replace('_', ' ').title()}: {count}\n"
    
    summary_text += "\n" + "-"*40 + "\n\n"
    
    # List top lesions
    summary_text += "TOP FINDINGS:\n\n"
    sorted_lesions = sorted(lesions, key=lambda x: -x['confidence'])[:5]
    
    for i, lesion in enumerate(sorted_lesions, 1):
        summary_text += f"{i}. {lesion['class'].replace('_', ' ').title()}\n"
        summary_text += f"   Confidence: {lesion['confidence']:.2f}\n"
        summary_text += f"   Size: {lesion['features']['area']:.0f} pixels\n"
        summary_text += f"   Mean HU: {lesion['features']['mean_hu']:.1f}\n"
        summary_text += f"   Organ: {lesion['organ']}\n\n"
    
    ax.text(0.05, 0.95, summary_text, transform=ax.transAxes,
           fontsize=10, verticalalignment='top', fontfamily='monospace',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.suptitle(f'Advanced Lesion Analysis - Slice {slice_idx}',
                fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    
    # Save
    output_file = output_dir / f'lesion_analysis_slice_{slice_idx:03d}.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    
    return output_file

def main():
    """Main lesion detection script"""
    
    print("\n" + "="*70)
    print("Advanced Lesion Detection System")
    print("="*70 + "\n")
    
    # Load CT
    ct_path = Path("CTdata/nifti/inha_ct_arterial.nii.gz")
    ct_volume = load_nifti(ct_path)
    print(f"CT volume: {ct_volume.shape}")
    
    # Load organs
    seg_dir = Path("CTdata/segmentation")
    priority_organs = ['colon', 'liver', 'kidney_left', 'kidney_right',
                      'spleen', 'pancreas', 'stomach', 'small_bowel']
    
    organ_masks_3d = {}
    for organ_name in priority_organs:
        organ_file = seg_dir / f"{organ_name}.nii.gz"
        if organ_file.exists():
            mask = load_nifti(organ_file)
            if mask.sum() > 0:
                organ_masks_3d[organ_name] = mask.astype(np.uint8)
                print(f"  Loaded: {organ_name}")
    
    # Analyze baselines
    print("\nComputing organ baselines...")
    organ_baselines = {}
    for organ_name, mask_3d in organ_masks_3d.items():
        voxels = ct_volume[mask_3d > 0]
        if len(voxels) > 0:
            organ_baselines[organ_name] = {
                'mean_hu': float(np.mean(voxels)),
                'std_hu': float(np.std(voxels)),
                'median_hu': float(np.median(voxels))
            }
            print(f"  {organ_name}: {organ_baselines[organ_name]['mean_hu']:.1f} ± "
                  f"{organ_baselines[organ_name]['std_hu']:.1f} HU")
    
    # Process key slices
    print("\nDetecting lesions...")
    output_dir = Path("CTdata/visualizations/lesion_detection")
    output_dir.mkdir(exist_ok=True, parents=True)
    
    n_slices = ct_volume.shape[0]
    slice_indices = [n_slices // 4, n_slices // 2, 3 * n_slices // 4]
    
    all_results = {}
    
    for slice_idx in slice_indices:
        print(f"\n  Processing slice {slice_idx}...")
        ct_slice = ct_volume[slice_idx, :, :]
        
        # Get 2D organ masks
        organs_2d = {}
        for organ_name, mask_3d in organ_masks_3d.items():
            mask_2d = mask_3d[slice_idx, :, :].astype(np.uint8)
            if mask_2d.sum() > 100:
                organs_2d[organ_name] = mask_2d
        
        # Detect lesions in each organ
        all_lesions = []
        for organ_name, mask_2d in organs_2d.items():
            baseline = organ_baselines.get(organ_name)
            lesions = detect_and_classify_lesions(ct_slice, mask_2d, organ_name, baseline)
            all_lesions.extend(lesions)
            
            if lesions:
                print(f"    {organ_name}: {len(lesions)} lesions")
        
        # Visualize
        output_file = visualize_lesion_detection(ct_slice, organs_2d, all_lesions, slice_idx, output_dir)
        print(f"    Saved: {output_file.name}")
        
        # Store results
        all_results[slice_idx] = {
            'lesion_count': len(all_lesions),
            'lesions': [
                {
                    'class': l['class'],
                    'confidence': l['confidence'],
                    'organ': l['organ'],
                    'features': {k: v for k, v in l['features'].items() if k != 'complexity'}
                }
                for l in all_lesions
            ]
        }
    
    # Save results
    results_file = output_dir / "lesion_detection_results.json"
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print("\n" + "="*70)
    print("Lesion detection complete!")
    print("="*70)
    print(f"\nOutput: {output_dir}")
    print(f"Results: {results_file}")

if __name__ == '__main__':
    main()
