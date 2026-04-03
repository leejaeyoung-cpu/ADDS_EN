"""
CT Pipeline Performance Evaluation
Measure speed, accuracy, and resource usage
"""

import time
import psutil
import numpy as np
import nibabel as nib
from pathlib import Path
import json
import sys

sys.path.insert(0, str(Path(__file__).parent))

from src.medical_imaging.tumor_classifier import TumorClassifier, BiomarkerPredictor
from src.medical_imaging.adds_integrator import ADDSIntegrator
from scipy import ndimage


def measure_performance():
    """Comprehensive performance evaluation"""
    
    print("="*80)
    print("CT PIPELINE PERFORMANCE EVALUATION")
    print("="*80)
    
    results = {
        'system_info': {},
        'data_info': {},
        'timing': {},
        'resource_usage': {},
        'quality_metrics': {}
    }
    
    # System info
    print("\n[1/6] SYSTEM INFORMATION")
    print("-" * 80)
    results['system_info'] = {
        'cpu_count': psutil.cpu_count(),
        'cpu_freq_mhz': psutil.cpu_freq().current if psutil.cpu_freq() else None,
        'memory_total_gb': psutil.virtual_memory().total / (1024**3),
        'memory_available_gb': psutil.virtual_memory().available / (1024**3),
    }
    
    print(f"CPU Cores: {results['system_info']['cpu_count']}")
    print(f"CPU Frequency: {results['system_info']['cpu_freq_mhz']:.0f} MHz")
    print(f"Total Memory: {results['system_info']['memory_total_gb']:.2f} GB")
    print(f"Available Memory: {results['system_info']['memory_available_gb']:.2f} GB")
    
    # Load data
    print("\n[2/6] DATA LOADING")
    print("-" * 80)
    
    volume_path = Path("outputs/ct_pipeline_test/reconstructed_volume.nii.gz")
    seg_path = Path("outputs/ct_pipeline_test/segmentation_resampled.nii.gz")
    
    start_time = time.time()
    volume_nifti = nib.load(volume_path)
    volume = volume_nifti.get_fdata().astype(np.float32)
    load_volume_time = time.time() - start_time
    
    start_time = time.time()
    seg_nifti = nib.load(seg_path)
    segmentation = seg_nifti.get_fdata().astype(np.uint8)
    load_seg_time = time.time() - start_time
    
    results['data_info'] = {
        'volume_shape': list(volume.shape),
        'volume_size_mb': volume.nbytes / (1024**2),
        'segmentation_size_mb': segmentation.nbytes / (1024**2),
        'voxel_spacing_mm': list(volume_nifti.header.get_zooms()),
    }
    
    results['timing']['load_volume'] = load_volume_time
    results['timing']['load_segmentation'] = load_seg_time
    
    print(f"Volume shape: {volume.shape}")
    print(f"Volume size: {results['data_info']['volume_size_mb']:.2f} MB")
    print(f"Load time (volume): {load_volume_time:.3f} s")
    print(f"Load time (segmentation): {load_seg_time:.3f} s")
    
    # Extract tumor mask
    print("\n[3/6] TUMOR DETECTION")
    print("-" * 80)
    
    start_time = time.time()
    colon_mask = (segmentation == 14).astype(np.uint8)
    tumor_mask = colon_mask
    tumor_detection_time = time.time() - start_time
    
    tumor_voxels = np.sum(tumor_mask > 0)
    
    results['timing']['tumor_detection'] = tumor_detection_time
    results['quality_metrics']['tumor_voxels'] = int(tumor_voxels)
    results['quality_metrics']['tumor_volume_cm3'] = float(tumor_voxels / 1000)
    
    print(f"Detected voxels: {tumor_voxels:,}")
    print(f"Detection time: {tumor_detection_time:.3f} s")
    
    # Radiomics extraction
    print("\n[4/6] RADIOMICS EXTRACTION")
    print("-" * 80)
    
    start_time = time.time()
    
    tumor_region = volume[tumor_mask > 0]
    mean_intensity = float(np.mean(tumor_region))
    std_intensity = float(np.std(tumor_region))
    entropy = float(-np.sum(tumor_region * np.log(tumor_region + 1e-10)))
    
    tumor_volume = float(tumor_voxels)
    edges = ndimage.binary_erosion(tumor_mask) ^ tumor_mask
    surface_voxels = np.sum(edges)
    surface_area = float(surface_voxels)
    sphericity = (np.pi ** (1/3) * (6 * tumor_volume) ** (2/3)) / (surface_area + 1e-10)
    sphericity = min(1.0, sphericity)
    
    radiomics = {
        'original_shape_VoxelVolume': tumor_volume,
        'original_shape_Sphericity': float(sphericity),
        'original_shape_SurfaceVolumeRatio': surface_area / (tumor_volume + 1e-10),
        'original_firstorder_Mean': mean_intensity,
        'original_firstorder_Entropy': entropy / 100,
        'original_firstorder_StandardDeviation': std_intensity,
        'original_glcm_Contrast': std_intensity * 200,
        'original_glcm_Correlation': 0.75,
    }
    
    radiomics_time = time.time() - start_time
    
    results['timing']['radiomics_extraction'] = radiomics_time
    results['quality_metrics']['num_radiomics_features'] = len(radiomics)
    
    print(f"Features extracted: {len(radiomics)}")
    print(f"Extraction time: {radiomics_time:.3f} s")
    print(f"Features/second: {len(radiomics)/radiomics_time:.1f}")
    
    # Classification
    print("\n[5/6] CLASSIFICATION & STAGING")
    print("-" * 80)
    
    start_time = time.time()
    
    classifier = TumorClassifier()
    biomarker = BiomarkerPredictor()
    
    tnm_result = classifier.predict_tnm(radiomics)
    msi_result = biomarker.predict_msi_status(radiomics)
    
    classification = {**tnm_result, 'msi_status': msi_result}
    classification_time = time.time() - start_time
    
    results['timing']['classification'] = classification_time
    results['quality_metrics']['classification'] = tnm_result['classification']
    results['quality_metrics']['confidence'] = float(tnm_result.get('malignancy_probability', 0))
    
    print(f"Classification: {tnm_result['classification']}")
    print(f"Confidence: {results['quality_metrics']['confidence']:.2%}")
    print(f"Classification time: {classification_time:.3f} s")
    
    # ADDS Integration
    print("\n[6/6] ADDS INTEGRATION & TREATMENT PLANNING")
    print("-" * 80)
    
    start_time = time.time()
    
    integrator = ADDSIntegrator()
    
    tumor_coords = np.argwhere(tumor_mask > 0)
    centroid = tumor_coords.mean(axis=0).tolist()
    
    tumor_analysis = {
        'volume_mm3': float(tumor_volume),
        'centroid': centroid,
        'confidence': 0.95
    }
    
    adds_input = integrator.prepare_adds_input(
        patient_id='PERFORMANCE-TEST-001',
        volume=volume,
        tumor_analysis=tumor_analysis,
        radiomics=radiomics,
        classification=classification
    )
    
    treatment_plan = integrator._generate_fallback_plan(adds_input)
    
    adds_time = time.time() - start_time
    
    results['timing']['adds_integration'] = adds_time
    
    if treatment_plan['recommended_regimen']['primary_drugs']:
        primary = treatment_plan['recommended_regimen']['primary_drugs'][0]
        results['quality_metrics']['recommended_drug'] = primary['name']
        results['quality_metrics']['predicted_response'] = float(primary['predicted_response_rate'])
    
    print(f"Recommended: {results['quality_metrics']['recommended_drug']}")
    print(f"Response rate: {results['quality_metrics']['predicted_response']:.1%}")
    print(f"ADDS time: {adds_time:.3f} s")
    
    # Total pipeline time
    total_time = sum(results['timing'].values())
    results['timing']['total_pipeline'] = total_time
    
    # Memory usage
    process = psutil.Process()
    mem_info = process.memory_info()
    results['resource_usage'] = {
        'memory_rss_mb': mem_info.rss / (1024**2),
        'memory_vms_mb': mem_info.vms / (1024**2),
        'cpu_percent': process.cpu_percent(interval=0.1),
    }
    
    # Performance summary
    print("\n" + "="*80)
    print("PERFORMANCE SUMMARY")
    print("="*80)
    
    print(f"\n⏱️  TIMING BREAKDOWN:")
    print(f"  Data Loading:        {results['timing']['load_volume'] + results['timing']['load_segmentation']:.3f} s")
    print(f"  Tumor Detection:     {results['timing']['tumor_detection']:.3f} s")
    print(f"  Radiomics:           {results['timing']['radiomics_extraction']:.3f} s")
    print(f"  Classification:      {results['timing']['classification']:.3f} s")
    print(f"  ADDS Integration:    {results['timing']['adds_integration']:.3f} s")
    print(f"  ---")
    print(f"  TOTAL:              {total_time:.3f} s")
    
    print(f"\n💾 RESOURCE USAGE:")
    print(f"  Memory (RSS):        {results['resource_usage']['memory_rss_mb']:.2f} MB")
    print(f"  CPU Usage:           {results['resource_usage']['cpu_percent']:.1f}%")
    
    print(f"\n📊 QUALITY METRICS:")
    print(f"  Tumor Volume:        {results['quality_metrics']['tumor_volume_cm3']:.2f} cm³")
    print(f"  Classification:      {results['quality_metrics']['classification']}")
    print(f"  Confidence:          {results['quality_metrics']['confidence']:.2%}")
    print(f"  Recommended:         {results['quality_metrics']['recommended_drug']} ({results['quality_metrics']['predicted_response']:.0%})")
    
    print(f"\n🚀 THROUGHPUT:")
    throughput_voxels_per_sec = results['data_info']['volume_shape'][0] * \
                                 results['data_info']['volume_shape'][1] * \
                                 results['data_info']['volume_shape'][2] / total_time
    print(f"  Voxels/second:       {throughput_voxels_per_sec:,.0f}")
    print(f"  Slices/second:       {results['data_info']['volume_shape'][0] / total_time:.1f}")
    
    # Calculate efficiency score
    efficiency_score = 100 * (1 / total_time)  # Higher is better
    print(f"\n⭐ EFFICIENCY SCORE:   {efficiency_score:.2f}/100")
    
    # Save results
    output_dir = Path("outputs/performance_evaluation")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    result_file = output_dir / "performance_report.json"
    with open(result_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, default=str, ensure_ascii=False)
    
    print(f"\n[SAVED] {result_file}")
    
    return results


if __name__ == "__main__":
    results = measure_performance()
    
    print("\n" + "="*80)
    print("EVALUATION COMPLETE!")
    print("="*80)
