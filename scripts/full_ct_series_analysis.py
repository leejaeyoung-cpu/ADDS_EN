#!/usr/bin/env python3
"""
Full CT Series Analysis
Process all slices and generate comprehensive lesion report
"""

import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from pathlib import Path
import cv2
from typing import Dict, List
from scipy import ndimage
from skimage.feature import graycomatrix, graycoprops
from skimage.measure import label, regionprops
import json
from tqdm import tqdm

def load_nifti(file_path: Path) -> np.ndarray:
    """Load NIfTI file"""
    nii = nib.load(str(file_path))
    return nii.get_fdata()

def extract_lesion_features(ct_image: np.ndarray, lesion_mask: np.ndarray, organ_mask: np.ndarray, baseline: Dict) -> Dict:
    """Extract comprehensive lesion features"""
    if lesion_mask.sum() == 0:
        return None
    
    lesion_pixels = ct_image[lesion_mask > 0]
    
    intensity_features = {
        'mean_hu': float(np.mean(lesion_pixels)),
        'std_hu': float(np.std(lesion_pixels)),
        'min_hu': float(np.min(lesion_pixels)),
        'max_hu': float(np.max(lesion_pixels)),
        'median_hu': float(np.median(lesion_pixels)),
    }
    
    if baseline:
        intensity_features['z_score'] = (intensity_features['mean_hu'] - baseline['mean_hu']) / (baseline['std_hu'] + 1e-6)
    
    props = regionprops(lesion_mask.astype(int))[0]
    
    shape_features = {
        'area': float(props.area),
        'perimeter': float(props.perimeter),
        'circularity': 4 * np.pi * props.area / (props.perimeter ** 2 + 1e-6),
        'solidity': float(props.solidity),
    }
    
    features = {**intensity_features, **shape_features}
    return features

def classify_lesion(features: Dict) -> tuple:
    """
    Enhanced multi-criteria classification
    
    Uses scoring system to reduce false negatives while maintaining specificity.
    Combines HU values, morphology, size, and statistical deviation.
    """
    if not features:
        return 'unknown', 0.0
    
    # Extract features
    mean_hu = features.get('mean_hu', 0)
    area = features.get('area', 0)
    z_score = abs(features.get('z_score', 0))
    circularity = features.get('circularity', 0)
    solidity = features.get('solidity', 1.0)
    
    # Definite artifacts: very low HU (gas)
    if mean_hu < -800:
        return 'artifact', 0.95
    
    # Multi-criteria tumor scoring
    tumor_score = 0.0
    
    # Criterion 1: HU range (TIGHTENED: 80-150 to reduce false positives)
    # Tumors typically 80+ HU (enhanced), excludes normal bowel wall ~50-70 HU
    if 80 <= mean_hu <= 150:
        tumor_score += 0.40  # Increased weight
    elif 60 <= mean_hu < 80:  # Borderline
        tumor_score += 0.15  # Reduced weight
    elif mean_hu > 150:  # Highly enhanced
        tumor_score += 0.30
    
    # Criterion 2: Size (TIGHTENED: minimum 100 pixels)
    if 100 <= area <= 10000:
        tumor_score += 0.20
    elif area > 10000:  # Very large mass
        tumor_score += 0.15
    
    # Criterion 3: Shape irregularity (tumors are often irregular)
    if circularity < 0.8:
        tumor_score += 0.20
    if solidity < 0.95:  # Complex internal structure
        tumor_score += 0.10
    
    # Criterion 4: Statistical significance (TIGHTENED: 2.5 from 1.5)
    if z_score >= 2.5:
        tumor_score += 0.25  # Increased weight for significant outliers
    elif z_score >= 1.5:
        tumor_score += 0.10  # Partial credit
    
    # Classification based on total score (TIGHTENED thresholds)
    if tumor_score >= 0.70:  # Increased from 0.60
        confidence = min(0.50 + tumor_score * 0.5, 0.95)
        return 'potential_tumor', confidence
    elif tumor_score >= 0.50:  # Increased from 0.40
        confidence = min(0.40 + tumor_score * 0.4, 0.75)
        return 'suspicious', confidence
    elif mean_hu < -500:
        return 'artifact', 0.90
    elif mean_hu < 30 and circularity > 0.8:
        return 'cyst', 0.75
    elif 30 < mean_hu < 80 and area < 200:
        return 'inflammation', 0.60
    else:
        return 'uncertain', 0.30

def detect_lesions_in_slice(ct_slice: np.ndarray, organ_mask: np.ndarray, baseline: Dict, organ_name: str) -> List[Dict]:
    """
    Enhanced lesion detection with multi-method approach
    
    Methods:
    1. Z-score based anomaly detection (relaxed threshold)
    2. Absolute HU threshold for bright masses (new)
    """
    if organ_mask.sum() == 0 or baseline is None:
        return []
    
    mean_hu = baseline['mean_hu']
    std_hu = baseline['std_hu']
    organ_pixels = organ_mask > 0
    
    # Method 1: Z-score based detection (TIGHTENED: 2.5 from 1.5)
    z_scores = np.zeros_like(ct_slice)
    z_scores[organ_pixels] = (ct_slice[organ_pixels] - mean_hu) / (std_hu + 1e-6)
    anomalies_zscore = (np.abs(z_scores) > 2.5) & organ_pixels  # Stricter
    
    # Method 2: Absolute HU threshold (TIGHTENED: 80 from 30)
    # Only detect clearly bright regions, exclude normal bowel wall (~50-70 HU)
    anomalies_hu = (ct_slice >= 80) & (ct_slice <= 200) & organ_pixels
    
    # Combine both methods
    anomalies = anomalies_zscore | anomalies_hu
    
    # Clean up noise
    kernel = np.ones((3, 3), np.uint8)
    anomalies = cv2.morphologyEx(anomalies.astype(np.uint8), cv2.MORPH_OPEN, kernel)
    anomalies = cv2.morphologyEx(anomalies, cv2.MORPH_CLOSE, kernel)
    
    # Label connected components
    labeled = label(anomalies)
    
    lesions = []
    for region in regionprops(labeled):
        lesion_mask = (labeled == region.label).astype(np.uint8)
        features = extract_lesion_features(ct_slice, lesion_mask, organ_mask, baseline)
        
        if features:
            lesion_class, confidence = classify_lesion(features)
            
            lesions.append({
                'features': features,
                'class': lesion_class,
                'confidence': confidence,
                'organ': organ_name,
                'centroid': region.centroid,
                'area': features['area']
            })
    
    return lesions

def create_summary_visualization(all_lesions: Dict, output_dir: Path):
    """Create summary visualization of all findings"""
    
    # Count by class
    class_counts = {}
    organ_counts = {}
    
    for slice_idx, data in all_lesions.items():
        for lesion in data['lesions']:
            cls = lesion['class']
            organ = lesion['organ']
            class_counts[cls] = class_counts.get(cls, 0) + 1
            organ_counts[organ] = organ_counts.get(organ, 0) + 1
    
    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Lesions per slice
    ax = axes[0, 0]
    slice_indices = sorted([int(k) for k in all_lesions.keys()])
    lesion_counts = [all_lesions[idx]['lesion_count'] for idx in slice_indices]
    
    ax.bar(slice_indices, lesion_counts, color='steelblue', alpha=0.7)
    ax.set_xlabel('Slice Index', fontsize=12)
    ax.set_ylabel('Lesion Count', fontsize=12)
    ax.set_title('Lesions per Slice', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # 2. Classification distribution
    ax = axes[0, 1]
    if class_counts:
        classes = list(class_counts.keys())
        counts = list(class_counts.values())
        colors = ['red', 'yellow', 'cyan', 'green', 'gray'][:len(classes)]
        
        ax.pie(counts, labels=classes, autopct='%1.1f%%', colors=colors, startangle=90)
        ax.set_title('Lesion Classification', fontsize=14, fontweight='bold')
    
    # 3. Organ distribution
    ax = axes[1, 0]
    if organ_counts:
        organs = list(organ_counts.keys())
        counts = list(organ_counts.values())
        
        ax.barh(organs, counts, color='forestgreen', alpha=0.7)
        ax.set_xlabel('Lesion Count', fontsize=12)
        ax.set_title('Lesions by Organ', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')
    
    # 4. Summary statistics
    ax = axes[1, 1]
    ax.axis('off')
    
    total_lesions = sum(class_counts.values()) if class_counts else 0
    total_slices = len(all_lesions)
    slices_with_lesions = sum(1 for d in all_lesions.values() if d['lesion_count'] > 0)
    
    summary_text = "COMPREHENSIVE ANALYSIS SUMMARY\n"
    summary_text += "="*45 + "\n\n"
    summary_text += f"Total Slices Analyzed: {total_slices}\n"
    summary_text += f"Slices with Lesions: {slices_with_lesions}\n"
    summary_text += f"Total Lesions Detected: {total_lesions}\n\n"
    
    summary_text += "Classification Breakdown:\n"
    for cls, count in sorted(class_counts.items(), key=lambda x: -x[1]):
        pct = 100 * count / total_lesions if total_lesions > 0 else 0
        summary_text += f"  {cls}: {count} ({pct:.1f}%)\n"
    
    summary_text += "\nOrgan Distribution:\n"
    for organ, count in sorted(organ_counts.items(), key=lambda x: -x[1]):
        pct = 100 * count / total_lesions if total_lesions > 0 else 0
        summary_text += f"  {organ}: {count} ({pct:.1f}%)\n"
    
    ax.text(0.05, 0.95, summary_text, transform=ax.transAxes,
           fontsize=11, verticalalignment='top', fontfamily='monospace',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.suptitle('Full CT Series Analysis Summary', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    output_file = output_dir / 'full_series_summary.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    
    return output_file

def main():
    """Full CT series analysis"""
    
    print("\n" + "="*70)
    print("FULL CT SERIES ANALYSIS")
    print("="*70 + "\n")
    
    # Load CT
    ct_path = Path("CTdata/nifti/inha_ct_arterial.nii.gz")
    ct_volume = load_nifti(ct_path)
    n_slices = ct_volume.shape[0]
    print(f"CT volume: {ct_volume.shape} ({n_slices} slices)")
    
    # Load organs
    seg_dir = Path("CTdata/segmentation")
    priority_organs = ['colon', 'liver', 'kidney_left', 'kidney_right',
                      'spleen', 'pancreas', 'stomach', 'small_bowel']
    
    organ_masks_3d = {}
    print("\nLoading organ masks...")
    for organ_name in priority_organs:
        organ_file = seg_dir / f"{organ_name}.nii.gz"
        if organ_file.exists():
            mask = load_nifti(organ_file)
            if mask.sum() > 0:
                organ_masks_3d[organ_name] = mask.astype(np.uint8)
                print(f"  [OK] {organ_name}")
    
    # Compute baselines
    print("\nComputing organ baselines...")
    organ_baselines = {}
    for organ_name, mask_3d in organ_masks_3d.items():
        voxels = ct_volume[mask_3d > 0]
        if len(voxels) > 0:
            organ_baselines[organ_name] = {
                'mean_hu': float(np.mean(voxels)),
                'std_hu': float(np.std(voxels)),
            }
            print(f"  {organ_name}: {organ_baselines[organ_name]['mean_hu']:.1f} ± "
                  f"{organ_baselines[organ_name]['std_hu']:.1f} HU")
    
    # Process all slices
    print(f"\nProcessing {n_slices} slices...")
    output_dir = Path("CTdata/visualizations/full_series")
    output_dir.mkdir(exist_ok=True, parents=True)
    
    all_results = {}
    potential_tumors = []
    
    for slice_idx in tqdm(range(n_slices), desc="Analyzing"):
        ct_slice = ct_volume[slice_idx, :, :]
        
        # Get 2D organ masks
        organs_2d = {}
        for organ_name, mask_3d in organ_masks_3d.items():
            mask_2d = mask_3d[slice_idx, :, :].astype(np.uint8)
            if mask_2d.sum() > 100:
                organs_2d[organ_name] = mask_2d
        
        # Detect lesions
        all_lesions = []
        for organ_name, mask_2d in organs_2d.items():
            baseline = organ_baselines.get(organ_name)
            lesions = detect_lesions_in_slice(ct_slice, mask_2d, baseline, organ_name)
            all_lesions.extend(lesions)
        
        # Store results
        all_results[slice_idx] = {
            'lesion_count': len(all_lesions),
            'lesions': all_lesions
        }
        
        # Track potential tumors
        for lesion in all_lesions:
            if lesion['class'] == 'potential_tumor':
                potential_tumors.append({
                    'slice': slice_idx,
                    'lesion': lesion
                })
    
    # Generate summary
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)
    
    total_lesions = sum(d['lesion_count'] for d in all_results.values())
    slices_with_lesions = sum(1 for d in all_results.values() if d['lesion_count'] > 0)
    
    print(f"\nTotal lesions detected: {total_lesions}")
    print(f"Slices with lesions: {slices_with_lesions}/{n_slices}")
    print(f"Potential tumors: {len(potential_tumors)}")
    
    # Create visualizations
    print("\nGenerating summary visualization...")
    summary_file = create_summary_visualization(all_results, output_dir)
    print(f"  Saved: {summary_file}")
    
    # Save detailed results
    results_file = output_dir / "full_series_results.json"
    
    # Convert for JSON serialization
    json_results = {}
    for slice_idx, data in all_results.items():
        json_results[str(slice_idx)] = {
            'lesion_count': data['lesion_count'],
            'lesions': [
                {
                    'class': l['class'],
                    'confidence': l['confidence'],
                    'organ': l['organ'],
                    'area': l['area'],
                    'mean_hu': l['features']['mean_hu'],
                    'z_score': l['features'].get('z_score', 0)
                }
                for l in data['lesions']
            ]
        }
    
    with open(results_file, 'w') as f:
        json.dump(json_results, f, indent=2)
    
    print(f"  Saved: {results_file}")
    
    # Report potential tumors
    if potential_tumors:
        print("\n" + "="*70)
        print("POTENTIAL TUMORS DETECTED")
        print("="*70)
        
        for i, item in enumerate(potential_tumors[:10], 1):  # Show top 10
            lesion = item['lesion']
            print(f"\n{i}. Slice {item['slice']}")
            print(f"   Organ: {lesion['organ']}")
            print(f"   Size: {lesion['area']:.0f} pixels")
            print(f"   Mean HU: {lesion['features']['mean_hu']:.1f}")
            print(f"   Confidence: {lesion['confidence']:.2f}")
    
    print("\n" + "="*70)
    print(f"Results saved to: {output_dir}")
    print("="*70)

if __name__ == '__main__':
    main()
