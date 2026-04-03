"""
Performance Profiling Utilities
Provides tools for identifying bottlenecks and measuring performance
"""

import cProfile
import pstats
import io
from functools import wraps
import time
from typing import Callable, Any
from pathlib import Path


class PerformanceProfiler:
    """Performance profiling utility for identifying bottlenecks"""
    
    def __init__(self, output_dir: str = "logs/profiles"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    @staticmethod
    def profile_function(func: Callable) -> Callable:
        """
        Decorator to profile function execution
        
        Usage:
            @PerformanceProfiler.profile_function
            def my_function():
                # ... code ...
        """
        @wraps(func)
        def wrapper(*args, **kwargs):
            profiler = cProfile.Profile()
            profiler.enable()
            
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
            finally:
                elapsed = time.time() - start_time
                profiler.disable()
                
                # Print statistics
                s = io.StringIO()
                ps = pstats.Stats(profiler, stream=s)
                ps.sort_stats('cumulative')
                ps.print_stats(20)  # Top 20 functions
                
                print(f"\n{'='*60}")
                print(f"Profile: {func.__name__}")
                print(f"{'='*60}")
                print(s.getvalue())
                print(f"\nTotal execution time: {elapsed:.4f} seconds")
                print(f"{'='*60}\n")
            
            return result
        return wrapper
    
    @staticmethod
    def time_function(func: Callable) -> Callable:
        """
        Simple timing decorator
        
        Usage:
            @PerformanceProfiler.time_function
            def my_function():
                # ... code ...
        """
        @wraps(func)
        def wrapper(*args, **kwargs):
            start = time.time()
            result = func(*args, **kwargs)
            elapsed = time.time() - start
            
            print(f"⏱️  {func.__name__}: {elapsed:.4f}s")
            
            return result
        return wrapper
    
    def save_profile(self, func: Callable, filename: str = None) -> Callable:
        """
        Save profile to file for later analysis
        
        Usage:
            profiler = PerformanceProfiler()
            
            @profiler.save_profile
            def my_function():
                # ... code ...
        """
        @wraps(func)
        def wrapper(*args, **kwargs):
            profiler = cProfile.Profile()
            profiler.enable()
            
            try:
                result = func(*args, **kwargs)
            finally:
                profiler.disable()
                
                # Save to file
                if filename:
                    output_file = self.output_dir / filename
                else:
                    timestamp = time.strftime("%Y%m%d_%H%M%S")
                    output_file = self.output_dir / f"{func.__name__}_{timestamp}.prof"
                
                profiler.dump_stats(str(output_file))
                print(f"📊 Profile saved to: {output_file}")
                
                # Also print summary
                ps = pstats.Stats(profiler)
                ps.sort_stats('cumulative')
                print("\nTop 10 functions by cumulative time:")
                ps.print_stats(10)
            
            return result
        return wrapper


class TimingContext:
    """
    Context manager for timing code blocks
    
    Usage:
        with TimingContext("Database query"):
            # ... code to time ...
    """
    
    def __init__(self, name: str = "Operation", verbose: bool = True):
        self.name = name
        self.verbose = verbose
        self.start_time = None
        self.elapsed = None
    
    def __enter__(self):
        self.start_time = time.time()
        if self.verbose:
            print(f"⏱️  Starting: {self.name}")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.elapsed = time.time() - self.start_time
        if self.verbose:
            print(f"✓ {self.name}: {self.elapsed:.4f}s")


class MemoryProfiler:
    """Memory profiling utilities"""
    
    @staticmethod
    def get_memory_usage() -> dict:
        """Get current memory usage in MB"""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        
        return {
            'rss_mb': memory_info.rss / 1024 / 1024,  # Resident Set Size
            'vms_mb': memory_info.vms / 1024 / 1024,  # Virtual Memory Size
            'percent': process.memory_percent()
        }
    
    @staticmethod
    def monitor_memory(func: Callable) -> Callable:
        """
        Monitor memory usage during function execution
        
        Usage:
            @MemoryProfiler.monitor_memory
            def my_function():
                # ... code ...
        """
        @wraps(func)
        def wrapper(*args, **kwargs):
            import psutil
            import os
            
            process = psutil.Process(os.getpid())
            mem_before = process.memory_info().rss / 1024 / 1024
            
            result = func(*args, **kwargs)
            
            mem_after = process.memory_info().rss / 1024 / 1024
            mem_diff = mem_after - mem_before
            
            print(f"🧠 {func.__name__} memory: {mem_before:.1f}MB → {mem_after:.1f}MB "
                  f"(Δ{mem_diff:+.1f}MB)")
            
            return result
        return wrapper


def benchmark_function(func: Callable, iterations: int = 10, warmup: int = 2) -> dict:
    """
    Benchmark a function with multiple iterations
    
    Args:
        func: Function to benchmark
        iterations: Number of iterations
        warmup: Number of warmup iterations (not counted)
    
    Returns:
        Dictionary with timing statistics
    """
    import statistics
    
    # Warmup
    for _ in range(warmup):
        func()
    
    # Benchmark
    times = []
    for _ in range(iterations):
        start = time.time()
        func()
        elapsed = time.time() - start
        times.append(elapsed)
    
    return {
        'mean': statistics.mean(times),
        'median': statistics.median(times),
        'stdev': statistics.stdev(times) if len(times) > 1 else 0,
        'min': min(times),
        'max': max(times),
        'iterations': iterations
    }


# Convenience functions
def profile(func: Callable) -> Callable:
    """Convenience alias for PerformanceProfiler.profile_function"""
    return PerformanceProfiler.profile_function(func)


def time_it(func: Callable) -> Callable:
    """Convenience alias for PerformanceProfiler.time_function"""
    return PerformanceProfiler.time_function(func)


def monitor_memory(func: Callable) -> Callable:
    """Convenience alias for MemoryProfiler.monitor_memory"""
    return MemoryProfiler.monitor_memory(func)
