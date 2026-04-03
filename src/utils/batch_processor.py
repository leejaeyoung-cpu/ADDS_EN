"""
Batch Image Processor
Parallel processing of multiple images with progress tracking
"""

import asyncio
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Callable, Any
from pathlib import Path
import time
from dataclasses import dataclass, field


@dataclass
class BatchResult:
    """Result from batch processing"""
    success_count: int = 0
    failure_count: int = 0
    total_time: float = 0.0
    results: List[Dict[str, Any]] = field(default_factory=list)
    errors: List[Dict[str, str]] = field(default_factory=list)


class BatchProcessor:
    """Process multiple images in parallel"""
    
    def __init__(self, max_workers: int = 4):
        """
        Initialize batch processor
        
        Args:
            max_workers: Maximum number of parallel workers
        """
        self.max_workers = max_workers
    
    def process_batch(
        self,
        items: List[Any],
        process_func: Callable,
        progress_callback: Callable[[int, int], None] = None,
        **kwargs
    ) -> BatchResult:
        """
        Process batch of items in parallel
        
        Args:
            items: List of items to process
            process_func: Function to apply to each item
            progress_callback: Callback(current, total) for progress updates
            **kwargs: Additional arguments to pass to process_func
        
        Returns:
            BatchResult with success/failure counts and results
        """
        batch_result = BatchResult()
        start_time = time.time()
        
        total = len(items)
        completed = 0
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            future_to_item = {
                executor.submit(process_func, item, **kwargs): item
                for item in items
            }
            
            # Process completed tasks
            for future in as_completed(future_to_item):
                item = future_to_item[future]
                completed += 1
                
                try:
                    result = future.result()
                    batch_result.results.append(result)
                    batch_result.success_count += 1
                except Exception as e:
                    batch_result.errors.append({
                        'item': str(item),
                        'error': str(e)
                    })
                    batch_result.failure_count += 1
                
                # Progress callback
                if progress_callback:
                    progress_callback(completed, total)
        
        batch_result.total_time = time.time() - start_time
        return batch_result
    
    def process_images_batch(
        self,
        image_paths: List[str],
        cellpose_func: Callable,
        progress_callback: Callable[[int, int], None] = None,
        **cellpose_params
    ) -> BatchResult:
        """
        Process batch of images with Cellpose
        
        Args:
            image_paths: List of image file paths
            cellpose_func: Cellpose segmentation function
            progress_callback: Progress callback function
            **cellpose_params: Parameters for Cellpose
        
        Returns:
            BatchResult with all analysis results
        """
        def process_single_image(image_path: str) -> Dict[str, Any]:
            """Process single image"""
            from PIL import Image
            import numpy as np
            
            # Load image
            img = Image.open(image_path)
            image_array = np.array(img)
            
            # Run segmentation
            result = cellpose_func(image_array, **cellpose_params)
            
            # Add metadata
            result['image_path'] = image_path
            result['image_name'] = Path(image_path).name
            
            return result
        
        return self.process_batch(
            image_paths,
            process_single_image,
            progress_callback
        )


class AsyncBatchProcessor:
    """Async batch processor for non-blocking operations"""
    
    def __init__(self, max_concurrent: int = 4):
        self.max_concurrent = max_concurrent
        self.semaphore = asyncio.Semaphore(max_concurrent)
    
    async def process_async(
        self,
        items: List[Any],
        async_func: Callable,
        progress_callback: Callable[[int, int], None] = None,
        **kwargs
    ) -> BatchResult:
        """
        Process batch asynchronously
        
        Args:
            items: Items to process
            async_func: Async function to call for each item
            progress_callback: Progress callback
            **kwargs: Additional arguments
        
        Returns:
            BatchResult
        """
        batch_result = BatchResult()
        start_time = time.time()
        
        total = len(items)
        completed = 0
        
        async def process_with_semaphore(item):
            async with self.semaphore:
                return await async_func(item, **kwargs)
        
        # Create tasks
        tasks = [process_with_semaphore(item) for item in items]
        
        # Process tasks
        for coro in asyncio.as_completed(tasks):
            try:
                result = await coro
                batch_result.results.append(result)
                batch_result.success_count += 1
            except Exception as e:
                batch_result.errors.append({'error': str(e)})
                batch_result.failure_count += 1
            
            completed += 1
            if progress_callback:
                progress_callback(completed, total)
        
        batch_result.total_time = time.time() - start_time
        return batch_result
