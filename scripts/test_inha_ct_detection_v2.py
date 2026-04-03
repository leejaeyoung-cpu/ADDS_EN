#!/usr/bin/env python3
"""
인하대병원 CT Detection v2 - High Sensitivity + Probability Scoring
Semi-supervised annotation workflow
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from medical_imaging.ct_analyzer import CTAnalyzer
import pydicom
import numpy as np
import json
from typing import List, Dict

def calculate_tumor_probability(region_features: Dict) -> Dict:
    """
    종양 확률 점수 계산
    
    Returns:
        {
            'probability': 0.0-1.0,
            'confidence_level': 'Low'/'Medium'/'High',
            'scores': {...individual scores...}
        }
    """
    scores = []
    score_details = {}
    
    # 1. Intensity score (HU or normalized intensity)
    intensity_mean = region_features.get('intensity_mean', 0)
    if 30 < intensity_mean < 80:  # Typical tumor HU range
        intensity_score = 0.8
    elif 20 < intensity_mean < 100:
        intensity_score = 0.5
    else:
        intensity_score = 0.2
    scores.append(intensity_score)
    score_details['intensity'] = intensity_score
    
    # 2. Size score
    volume_voxels = region_features.get('shape_volume_voxels', 0)
    if 500 < volume_voxels < 3000:  # Typical tumor size
        size_score = 0.8
    elif 200 < volume_voxels < 5000:
        size_score = 0.5
    else:
        size_score = 0.3
    scores.append(size_score)
    score_details['size'] = size_score
    
    # 3. Texture score (heterogeneity)
    intensity_std = region_features.get('intensity_std', 0)
    if intensity_std > 15:  # Heterogeneous (tumor characteristic)
        texture_score = 0.7
    elif intensity_std > 10:
        texture_score = 0.5
    else:
        texture_score = 0.3
    scores.append(texture_score)
    score_details['texture'] = texture_score
    
    # 4. Shape score (compactness)
    # Note: This would require circularity calculation
    # For now, assume moderate score
    shape_score = 0.5
    scores.append(shape_score)
    score_details['shape'] = shape_score
    
    # Calculate weighted average
    probability = sum(scores) / len(scores)
    
    # Confidence level
    if probability >= 0.7:
        confidence = 'High'
    elif probability >= 0.4:
        confidence = 'Medium'
    else:
        confidence = 'Low'
    
    return{
        'probability': float(probability),
        'confidence_level': confidence,
        'score_details': score_details
    }

def test_high_sensitivity_detection():
    """High sensitivity detection test"""
    
    print("\n" + "="*70)
    print("인하대병원 CT Detection v2 - High Sensitivity")
    print("="*70)
    
    # CT Analyzer 초기화
    analyzer = CTAnalyzer(use_gpu=True, use_nnunet=False, enable_ai_research=False)
    
    # DICOM directory
    dicom_dir = Path("CTdata/CTdcm")
    
    # Test more slices across the series
    test_slices = list(range(20030, 20090, 10))  # 20030, 20040, ..., 20080
    
    all_candidates = []
    
    for slice_num in test_slices:
        dcm_file = dicom_dir / f"{slice_num}.dcm"
        
        if not dcm_file.exists():
            continue
        
        print(f"\n{'─'*70}")
        print(f"Analyzing: {dcm_file.name}")
        
        try:
            # CT analysis
            result = analyzer.analyze_ct_image(
                image_path=str(dcm_file),
                cancer_type="Colorectal",
                additional_context="Inha Hospital Abdomen CT, Arterial phase"
            )
            
            if result['status'] != 'success':
                print(f"  Error: {result.get('error', 'Unknown')}")
                continue
            
            # Extract candidates
            segmentation = result.get('segmentation', {})
            radiomics = result.get('radiomics_features', {})
            measurements = result.get('measurements', {})
            
            if segmentation.get('tumor_detected', False):
                # Calculate probability
                prob_result = calculate_tumor_probability(radiomics)
                
                candidate = {
                    'slice': slice_num,
                    'file': dcm_file.name,
                    'probability': prob_result['probability'],
                    'confidence': prob_result['confidence_level'],
                    'bbox': segmentation.get('tumor_bounding_box', {}),
                    'volume_voxels': radiomics.get('shape_volume_voxels', 0),
                    'diameter_mm': measurements.get('longest_diameter_mm', 0),
                    'features': {
                        'intensity_mean': radiomics.get('intensity_mean', 0),
                        'intensity_std': radiomics.get('intensity_std', 0),
                        'texture_entropy': radiomics.get('texture_entropy', 0)
                    },
                    'score_details': prob_result['score_details']
                }
                
                all_candidates.append(candidate)
                
                # Print summary
                print(f"  ✓ Candidate detected!")
                print(f"    Probability: {prob_result['probability']:.2%}")
                print(f"    Confidence: {prob_result['confidence_level']}")
                print(f"    Size: {radiomics.get('shape_volume_voxels', 0)} voxels")
                print(f"    Diameter: {measurements.get('longest_diameter_mm', 0):.1f} mm")
            else:
                print(f"  No candidates detected")
        
        except Exception as e:
            print(f"  Error: {e}")
    
    # Summary
    print("\n" + "="*70)
    print("Detection Results Summary")
    print("="*70)
    
    total_slices = len(test_slices)
    detected_slices = len(all_candidates)
    
    print(f"\nTotal slices tested: {total_slices}")
    print(f"Candidates detected: {detected_slices}")
    print(f"Detection rate: {detected_slices/total_slices*100:.1f}%")
    
    if all_candidates:
        # Sort by probability
        sorted_candidates = sorted(all_candidates, key=lambda x: x['probability'], reverse=True)
        
        print(f"\n{'─'*70}")
        print("Top Candidates (sorted by probability):")
        print(f"{'─'*70}")
        
        for i, cand in enumerate(sorted_candidates[:10], 1):
            print(f"\n{i}. Slice {cand['slice']} ({cand['file']})")
            print(f"   Probability: {cand['probability']:.2%} ({cand['confidence']})")
            print(f"   Size: {cand['volume_voxels']} voxels, {cand['diameter_mm']:.1f} mm")
            print(f"   BBox: x={cand['bbox']['x']}, y={cand['bbox']['y']}, "
                  f"w={cand['bbox']['width']}, h={cand['bbox']['height']}")
        
        # Risk stratification
        high_risk = [c for c in all_candidates if c['probability'] >= 0.7]
        medium_risk = [c for c in all_candidates if 0.4 <= c['probability'] < 0.7]
        low_risk = [c for c in all_candidates if c['probability'] < 0.4]
        
        print(f"\n{'─'*70}")
        print("Risk Stratification:")
        print(f"  High Risk (≥70%): {len(high_risk)} candidates")
        print(f"  Medium Risk (40-70%): {len(medium_risk)} candidates")
        print(f"  Low Risk (<40%): {len(low_risk)} candidates")
    
    # Save results
    output_file = "inha_ct_candidates_v2.json"
    
    # Convert numpy types to native Python types for JSON serialization
    def convert_to_json_serializable(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_to_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_json_serializable(item) for item in obj]
        return obj
    
    serializable_candidates = convert_to_json_serializable(all_candidates)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump({
            'total_slices': total_slices,
            'detected_slices': detected_slices,
            'detection_rate': detected_slices/total_slices,
            'candidates': serializable_candidates
        }, f, indent=2, ensure_ascii=False)
    
    print(f"\n{'='*70}")
    print(f"Results saved: {output_file}")
    print("="*70 + "\n")
    
    print("Next Steps:")
    print("1. Review candidates in order of probability")
    print("2. Clinical validation (맞다/안맞다)")
    print("3. Build annotated dataset")
    print("4. Full 426-slice scan")
    
    return all_candidates

if __name__ == '__main__':
    candidates = test_high_sensitivity_detection()
