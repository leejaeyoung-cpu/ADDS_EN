"""
Performance Benchmarking for Perfect Reconstruction Pipeline

Measures:
- Execution time per stage
- Memory usage
- Scalability
- Bottleneck identification
"""

import time
import json
from pathlib import Path
import psutil
import os


def benchmark_pipeline():
    """Run complete pipeline with detailed timing"""
    
    print("="*80)
    print("PERFORMANCE BENCHMARKING - PERFECT RECONSTRUCTION")
    print("="*80)
    
    # Import after print to avoid early errors
    from backend.services.perfect_reconstruction import run_perfect_reconstruction_pipeline
    from backend.services.mesh_generator import MeshGenerator
    import nibabel as nib
    
    # Get process for memory tracking
    process = psutil.Process(os.getpid())
    
    # Paths
    detection_dir = Path("outputs/inha_ct_detection_with_masks")
    volume_path = Path("outputs/inha_ct_analysis/inha_ct_volume.nii.gz")
    output_dir = Path("outputs/inha_3d_analysis")
    
    # Get initial state
    start_memory = process.memory_info().rss / 1024 / 1024  # MB
    print(f"\nBaseline Memory: {start_memory:.1f} MB")
    
    # ========================================================================
    # STAGE 1-4: Perfect Reconstruction Pipeline
    # ========================================================================
    
    print("\n[STAGE 1-4] Running Perfect Reconstruction Pipeline...")
    
    stage_start = time.time()
    
    results = run_perfect_reconstruction_pipeline(
        detection_dir,
        volume_path,
        output_dir
    )
    
    pipeline_time = time.time() - stage_start
    pipeline_memory = process.memory_info().rss / 1024 / 1024
    
    num_tumors = results['summary']['total_tumors']
    
    print(f"\n  Time: {pipeline_time:.1f}s")
    print(f"  Memory: {pipeline_memory:.1f} MB (delta: +{pipeline_memory - start_memory:.1f} MB)")
    print(f"  Tumors: {num_tumors}")
    print(f"  Time per tumor: {pipeline_time / num_tumors:.2f}s")
    
    # ========================================================================
    # STAGE 5: Mesh Generation
    # ========================================================================
    
    print("\n[STAGE 5] Generating Enhanced Meshes...")
    
    stage_start = time.time()
    
    mesh_gen = MeshGenerator()
    nii = nib.load(volume_path)
    spacing = nii.header.get_zooms()
    
    tumor_meshes = mesh_gen.generate_tumor_meshes_perfect(
        results['masks_npz'],
        results['enhanced_json'],
        spacing,
        max_tumors=20
    )
    
    mesh_time = time.time() - stage_start
    mesh_memory = process.memory_info().rss / 1024 / 1024
    
    print(f"\n  Time: {mesh_time:.1f}s")
    print(f"  Memory: {mesh_memory:.1f} MB")
    print(f"  Meshes: {len(tumor_meshes)}")
    if tumor_meshes:
        print(f"  Time per mesh: {mesh_time / len(tumor_meshes):.2f}s")
    
    # ========================================================================
    # SUMMARY
    # ========================================================================
    
    total_time = pipeline_time + mesh_time
    peak_memory = mesh_memory
    
    print("\n" + "="*80)
    print("PERFORMANCE SUMMARY")
    print("="*80)
    
    print(f"\nTotal Execution Time: {total_time:.1f}s")
    print(f"  Pipeline (clustering + reconstruction + measurements): {pipeline_time:.1f}s ({pipeline_time/total_time*100:.1f}%)")
    print(f"  Mesh Generation: {mesh_time:.1f}s ({mesh_time/total_time*100:.1f}%)")
    
    print(f"\nMemory Usage:")
    print(f"  Baseline: {start_memory:.1f} MB")
    print(f"  Peak: {peak_memory:.1f} MB")
    print(f"  Delta: +{peak_memory - start_memory:.1f} MB")
    
    print(f"\nThroughput:")
    print(f"  Tumors processed: {num_tumors}")
    print(f"  Time per tumor (reconstruction): {pipeline_time / num_tumors:.2f}s")
    print(f"  Meshes generated: {len(tumor_meshes)}")
    if tumor_meshes:
        print(f"  Time per mesh: {mesh_time / len(tumor_meshes):.2f}s")
    
    # Scalability estimate
    print(f"\nScalability Estimates (linear):")
    print(f"  10 tumors:  ~{pipeline_time * 10 / num_tumors:.0f}s")
    print(f"  50 tumors:  ~{pipeline_time * 50 / num_tumors:.0f}s")
    print(f"  100 tumors: ~{pipeline_time * 100 / num_tumors:.0f}s")
    
    # Performance targets
    print(f"\nPerformance Targets:")
    target_time_per_tumor = 3.0
    target_peak_memory = 4000  # MB
    
    time_status = "PASS" if pipeline_time / num_tumors <= target_time_per_tumor else "WARN"
    memory_status = "PASS" if peak_memory <= target_peak_memory else "WARN"
    
    print(f"  [{time_status}] Time per tumor: {pipeline_time / num_tumors:.2f}s (target: <{target_time_per_tumor:.1f}s)")
    print(f"  [{memory_status}] Peak memory: {peak_memory:.0f} MB (target: <{target_peak_memory} MB)")
    
    # Save benchmark results
    benchmark_data = {
        'total_time_seconds': total_time,
        'pipeline_time_seconds': pipeline_time,
        'mesh_time_seconds': mesh_time,
        'baseline_memory_mb': start_memory,
        'peak_memory_mb': peak_memory,
        'memory_delta_mb': peak_memory - start_memory,
        'tumors_processed': num_tumors,
        'meshes_generated': len(tumor_meshes),
        'time_per_tumor_seconds': pipeline_time / num_tumors,
        'time_per_mesh_seconds': mesh_time / len(tumor_meshes) if tumor_meshes else 0,
        'performance_checks': {
            'time_per_tumor_pass': time_status == "PASS",
            'memory_usage_pass': memory_status == "PASS"
        }
    }
    
    benchmark_path = Path('outputs/inha_3d_analysis/performance_benchmark.json')
    with open(benchmark_path, 'w') as f:
        json.dump(benchmark_data, f, indent=2)
    
    print(f"\nBenchmark saved: {benchmark_path}")
    
    print("\n" + "="*80)
    print("PERFORMANCE BENCHMARKING COMPLETE")
    print("="*80)
    
    return benchmark_data


if __name__ == "__main__":
    benchmark_data = benchmark_pipeline()
    
    # Final verdict
    all_pass = (benchmark_data['performance_checks']['time_per_tumor_pass'] and
                benchmark_data['performance_checks']['memory_usage_pass'])
    
    print(f"\n{'='*80}")
    if all_pass:
        print("PERFORMANCE: PRODUCTION READY")
    else:
        print("PERFORMANCE: OPTIMIZATION RECOMMENDED")
    print(f"{'='*80}")
