"""
ADDS Cellpose Performance Benchmark
Comprehensive benchmarking of Cellpose segmentation performance
"""

import sys
import time
from pathlib import Path
import json
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import numpy as np
from PIL import Image
import torch

from utils.profiler import TimingContext, PerformanceProfiler, MemoryProfiler
from utils.config import ConfigLoader


class CellposeBenchmark:
    """Benchmark Cellpose segmentation performance"""
    
    def __init__(self):
        self.config = ConfigLoader()
        self.results = {
            'timestamp': datetime.now().isoformat(),
            'system_info': self._get_system_info(),
            'benchmarks': []
        }
    
    def _get_system_info(self):
        """Get system information"""
        import platform
        
        info = {
            'os': f"{platform.system()} {platform.release()}",
            'python': platform.python_version(),
            'torch_version': torch.__version__,
            'cuda_available': torch.cuda.is_available(),
        }
        
        if torch.cuda.is_available():
            info['cuda_version'] = torch.version.cuda
            info['gpu_name'] = torch.cuda.get_device_name(0)
            info['gpu_memory_gb'] = torch.cuda.get_device_properties(0).total_memory / 1024**3
        
        return info
    
    def _create_synthetic_image(self, size: tuple) -> np.ndarray:
        """Create synthetic test image"""
        # Create realistic cell-like image
        image = np.random.rand(*size) * 100
        
        # Add some cell-like circular structures
        y, x = np.ogrid[:size[0], :size[1]]
        num_cells = 50
        
        for _ in range(num_cells):
            cx, cy = np.random.randint(0, size[1]), np.random.randint(0, size[0])
            r = np.random.randint(10, 30)
            mask = (x - cx)**2 + (y - cy)**2 <= r**2
            image[mask] = 200 + np.random.rand() * 55
        
        return image.astype(np.uint8)
    
    def _load_cellpose_model(self, use_gpu: bool = True):
        """Load Cellpose model"""
        from cellpose import models
        
        # Disable GPU if requested or not available
        use_gpu = use_gpu and torch.cuda.is_available()
        
        print(f"Loading Cellpose model (GPU: {use_gpu})...")
        model = models.CellposeModel(gpu=use_gpu, model_type='cyto')
        
        return model, use_gpu
    
    def benchmark_image_size(self, sizes: list, use_gpu: bool = True, iterations: int = 5):
        """
        Benchmark performance across different image sizes
        
        Args:
            sizes: List of (height, width) tuples
            use_gpu: Whether to use GPU
            iterations: Number of iterations per size
        """
        print(f"\n{'='*60}")
        print(f"BENCHMARK: Image Size Performance (GPU: {use_gpu})")
        print(f"{'='*60}")
        
        model, actual_gpu = self._load_cellpose_model(use_gpu)
        
        for size in sizes:
            print(f"\nTesting size: {size[0]}x{size[1]}")
            
            # Create test image
            image = self._create_synthetic_image(size)
            
            # Warmup
            print("  Warmup...", end='', flush=True)
            _ = model.eval(image, diameter=30)
            print(" Done")
            
            # Benchmark
            times = []
            memory_usage = []
            
            for i in range(iterations):
                # Track memory before
                if actual_gpu:
                    torch.cuda.reset_peak_memory_stats()
                    mem_before = torch.cuda.memory_allocated() / 1024**2
                
                # Time the segmentation
                start = time.time()
                masks, flows, styles = model.eval(image, diameter=30)
                elapsed = time.time() - start
                
                times.append(elapsed)
                
                # Track memory after
                if actual_gpu:
                    mem_after = torch.cuda.memory_allocated() / 1024**2
                    peak_mem = torch.cuda.max_memory_allocated() / 1024**2
                    memory_usage.append({
                        'allocated': mem_after - mem_before,
                        'peak': peak_mem
                    })
                
                print(f"  Iteration {i+1}/{iterations}: {elapsed:.3f}s", end='')
                if actual_gpu:
                    print(f" (Peak GPU: {peak_mem:.1f}MB)", end='')
                print()
            
            # Calculate statistics
            mean_time = np.mean(times)
            std_time = np.std(times)
            median_time = np.median(times)
            
            result = {
                'image_size': f"{size[0]}x{size[1]}",
                'pixels': size[0] * size[1],
                'gpu_used': actual_gpu,
                'iterations': iterations,
                'mean_time_s': float(mean_time),
                'std_time_s': float(std_time),
                'median_time_s': float(median_time),
                'min_time_s': float(np.min(times)),
                'max_time_s': float(np.max(times)),
                'throughput_pixels_per_sec': size[0] * size[1] / mean_time
            }
            
            if actual_gpu and memory_usage:
                result['gpu_memory'] = {
                    'mean_allocated_mb': float(np.mean([m['allocated'] for m in memory_usage])),
                    'mean_peak_mb': float(np.mean([m['peak'] for m in memory_usage]))
                }
            
            self.results['benchmarks'].append(result)
            
            print(f"\n  Results:")
            print(f"    Mean: {mean_time:.3f}s (±{std_time:.3f}s)")
            print(f"    Median: {median_time:.3f}s")
            print(f"    Range: {np.min(times):.3f}s - {np.max(times):.3f}s")
            print(f"    Throughput: {result['throughput_pixels_per_sec']/1e6:.2f}M pixels/s")
            
            if actual_gpu and memory_usage:
                print(f"    GPU Memory (peak): {result['gpu_memory']['mean_peak_mb']:.1f}MB")
    
    def benchmark_batch_processing(self, batch_sizes: list, image_size: tuple = (512, 512)):
        """
        Benchmark batch processing performance
        
        Args:
            batch_sizes: List of batch sizes to test
            image_size: Size of each image
        """
        print(f"\n{'='*60}")
        print(f"BENCHMARK: Batch Processing Performance")
        print(f"{'='*60}")
        
        model, actual_gpu = self._load_cellpose_model(use_gpu=True)
        
        for batch_size in batch_sizes:
            print(f"\nTesting batch size: {batch_size}")
            
            # Create batch of images
            images = [self._create_synthetic_image(image_size) for _ in range(batch_size)]
            
            # Warmup
            _ = model.eval(images[0], diameter=30)
            
            # Benchmark sequential processing
            print("  Sequential processing...", end='', flush=True)
            start = time.time()
            for img in images:
                _ = model.eval(img, diameter=30)
            sequential_time = time.time() - start
            print(f" {sequential_time:.3f}s")
            
            # Calculate metrics
            avg_per_image = sequential_time / batch_size
            
            result = {
                'batch_size': batch_size,
                'image_size': f"{image_size[0]}x{image_size[1]}",
                'total_time_s': float(sequential_time),
                'avg_per_image_s': float(avg_per_image),
                'images_per_sec': batch_size / sequential_time
            }
            
            self.results['benchmarks'].append(result)
            
            print(f"  Results:")
            print(f"    Total time: {sequential_time:.3f}s")
            print(f"    Avg per image: {avg_per_image:.3f}s")
            print(f"    Throughput: {result['images_per_sec']:.2f} images/s")
    
    def benchmark_gpu_vs_cpu(self, image_size: tuple = (512, 512), iterations: int = 3):
        """
        Compare GPU vs CPU performance
        
        Args:
            image_size: Size of test image
            iterations: Number of iterations
        """
        print(f"\n{'='*60}")
        print(f"BENCHMARK: GPU vs CPU Performance")
        print(f"{'='*60}")
        
        if not torch.cuda.is_available():
            print("\nGPU not available, skipping GPU vs CPU benchmark")
            return
        
        image = self._create_synthetic_image(image_size)
        
        results_comparison = {'image_size': f"{image_size[0]}x{image_size[1]}"}
        
        for use_gpu in [False, True]:
            device_name = "GPU" if use_gpu else "CPU"
            print(f"\nTesting {device_name}...")
            
            model, actual_gpu = self._load_cellpose_model(use_gpu=use_gpu)
            
            # Warmup
            _ = model.eval(image, diameter=30)
            
            # Benchmark
            times = []
            for i in range(iterations):
                start = time.time()
                _ = model.eval(image, diameter=30)
                elapsed = time.time() - start
                times.append(elapsed)
                print(f"  Iteration {i+1}/{iterations}: {elapsed:.3f}s")
            
            mean_time = np.mean(times)
            results_comparison[device_name.lower() + '_mean_time_s'] = float(mean_time)
            
            print(f"  Mean: {mean_time:.3f}s")
        
        # Calculate speedup
        if 'cpu_mean_time_s' in results_comparison and 'gpu_mean_time_s' in results_comparison:
            speedup = results_comparison['cpu_mean_time_s'] / results_comparison['gpu_mean_time_s']
            results_comparison['gpu_speedup'] = float(speedup)
            
            print(f"\n  GPU Speedup: {speedup:.2f}x faster than CPU")
        
        self.results['benchmarks'].append(results_comparison)
    
    def analyze_bottlenecks(self, image_size: tuple = (512, 512)):
        """
        Analyze performance bottlenecks
        
        Args:
            image_size: Size of test image
        """
        print(f"\n{'='*60}")
        print(f"BENCHMARK: Bottleneck Analysis")
        print(f"{'='*60}")
        
        from cellpose import models
        
        model, actual_gpu = self._load_cellpose_model(use_gpu=True)
        image = self._create_synthetic_image(image_size)
        
        bottlenecks = {}
        
        # 1. Image loading (from disk)
        print("\n1. Testing image I/O...")
        test_path = Path("temp_test_image.png")
        Image.fromarray(image).save(test_path)
        
        times = []
        for _ in range(10):
            start = time.time()
            img = Image.open(test_path)
            arr = np.array(img)
            times.append(time.time() - start)
        
        bottlenecks['image_io_ms'] = float(np.mean(times) * 1000)
        print(f"   Image I/O: {bottlenecks['image_io_ms']:.2f}ms")
        
        test_path.unlink()  # Clean up
        
        # 2. Preprocessing
        print("\n2. Testing preprocessing...")
        times = []
        for _ in range(10):
            start = time.time()
            # Simulate normalization
            normalized = (image - np.mean(image)) / (np.std(image) + 1e-7)
            times.append(time.time() - start)
        
        bottlenecks['preprocessing_ms'] = float(np.mean(times) * 1000)
        print(f"   Preprocessing: {bottlenecks['preprocessing_ms']:.2f}ms")
        
        # 3. Model inference (full pipeline)
        print("\n3. Testing full inference...")
        times = []
        for _ in range(5):
            start = time.time()
            _ = model.eval(image, diameter=30)
            times.append(time.time() - start)
        
        bottlenecks['full_inference_ms'] = float(np.mean(times) * 1000)
        print(f"   Full inference: {bottlenecks['full_inference_ms']:.2f}ms")
        
        # 4. Memory profiling
        if actual_gpu:
            print("\n4. GPU Memory profiling...")
            torch.cuda.reset_peak_memory_stats()
            _ = model.eval(image, diameter=30)
            peak_mem = torch.cuda.max_memory_allocated() / 1024**2
            
            bottlenecks['gpu_peak_memory_mb'] = float(peak_mem)
            print(f"   Peak GPU memory: {peak_mem:.1f}MB")
        
        self.results['bottleneck_analysis'] = bottlenecks
        
        print(f"\nBottleneck Summary:")
        print(f"  Image I/O: {bottlenecks['image_io_ms']:.2f}ms ({bottlenecks['image_io_ms']/bottlenecks['full_inference_ms']*100:.1f}%)")
        print(f"  Preprocessing: {bottlenecks['preprocessing_ms']:.2f}ms ({bottlenecks['preprocessing_ms']/bottlenecks['full_inference_ms']*100:.1f}%)")
        print(f"  Full Inference: {bottlenecks['full_inference_ms']:.2f}ms (100%)")
    
    def save_results(self, output_path: str = "benchmark_results/performance_profile.json"):
        """Save benchmark results to JSON"""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        print(f"\n{'='*60}")
        print(f"Results saved to: {output_path}")
        print(f"{'='*60}")
    
    def print_summary(self):
        """Print benchmark summary"""
        print(f"\n{'='*60}")
        print(f"BENCHMARK SUMMARY")
        print(f"{'='*60}")
        
        print(f"\nSystem Information:")
        for key, value in self.results['system_info'].items():
            print(f"  {key}: {value}")
        
        print(f"\nKey Findings:")
        
        # Find slowest and fastest configurations
        size_benchmarks = [b for b in self.results['benchmarks'] if 'mean_time_s' in b and 'image_size' in b]
        if size_benchmarks:
            slowest = max(size_benchmarks, key=lambda x: x['mean_time_s'])
            fastest = min(size_benchmarks, key=lambda x: x['mean_time_s'])
            
            print(f"  Fastest config: {fastest['image_size']} - {fastest['mean_time_s']:.3f}s")
            print(f"  Slowest config: {slowest['image_size']} - {slowest['mean_time_s']:.3f}s")
        
        # GPU speedup
        gpu_comparison = [b for b in self.results['benchmarks'] if 'gpu_speedup' in b]
        if gpu_comparison:
            speedup = gpu_comparison[0]['gpu_speedup']
            print(f"  GPU Speedup: {speedup:.2f}x")
        
        # Average performance for 512x512 (standard size)
        standard_benchmarks = [b for b in size_benchmarks if '512x512' in b.get('image_size', '')]
        if standard_benchmarks:
            avg_time = np.mean([b['mean_time_s'] for b in standard_benchmarks])
            print(f"  Standard size (512x512) avg: {avg_time:.3f}s")
            print(f"  Target (< 5s): {'[PASS]' if avg_time < 5.0 else '[FAIL]'}")


def main():
    """Run comprehensive performance benchmark"""
    print("="*60)
    print("ADDS Cellpose Performance Benchmark")
    print("="*60)
    
    benchmark = CellposeBenchmark()
    
    try:
        # 1. Benchmark different image sizes
        sizes = [
            (256, 256),
            (512, 512),
            (1024, 1024),
        ]
        benchmark.benchmark_image_size(sizes, use_gpu=True, iterations=5)
        
        # 2. GPU vs CPU comparison
        benchmark.benchmark_gpu_vs_cpu(image_size=(512, 512), iterations=3)
        
        # 3. Batch processing
        benchmark.benchmark_batch_processing(
            batch_sizes=[1, 5, 10],
            image_size=(512, 512)
        )
        
        # 4. Bottleneck analysis
        benchmark.analyze_bottlenecks(image_size=(512, 512))
        
        # Save results
        benchmark.save_results()
        
        # Print summary
        benchmark.print_summary()
        
        print("\n[SUCCESS] Benchmark completed successfully!")
        
    except Exception as e:
        print(f"\n[ERROR] Benchmark failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
