"""
GPU Memory Monitor for CUDA Device Tracking
Provides real-time GPU memory usage monitoring with graceful CPU fallback
"""

import sys
from typing import Dict, Optional, List, Any
from datetime import datetime


class GPUMonitor:
    """
    Monitor GPU memory usage and availability
    Gracefully handles CPU-only environments
    """
    
    def __init__(self):
        """Initialize GPU monitor and detect CUDA availability"""
        self.cuda_available = False
        self.torch = None
        self.device_count = 0
        self.device_names = []
        
        try:
            import torch
            self.torch = torch
            self.cuda_available = torch.cuda.is_available()
            
            if self.cuda_available:
                self.device_count = torch.cuda.device_count()
                self.device_names = [torch.cuda.get_device_name(i) for i in range(self.device_count)]
        except ImportError:
            pass
    
    def is_available(self) -> bool:
        """Check if GPU is available"""
        return self.cuda_available
    
    def get_device_count(self) -> int:
        """Get number of available CUDA devices"""
        return self.device_count if self.cuda_available else 0
    
    def get_device_name(self, device_id: int = 0) -> str:
        """
        Get GPU device name
        
        Args:
            device_id: GPU device index
            
        Returns:
            Device name or "CPU" if not available
        """
        if self.cuda_available and 0 <= device_id < len(self.device_names):
            return self.device_names[device_id]
        return "CPU"
    
    def get_memory_info(self, device_id: int = 0) -> Dict[str, Any]:
        """
        Get memory information for a specific GPU
        
        Args:
            device_id: GPU device index
            
        Returns:
            Dictionary with memory statistics
        """
        if not self.cuda_available or self.torch is None:
            return {
                'available': False,
                'total_mb': 0,
                'allocated_mb': 0,
                'reserved_mb': 0,
                'free_mb': 0,
                'utilization_percent': 0,
                'device_name': 'CPU'
            }
        
        try:
            # Get memory stats in bytes
            total = self.torch.cuda.get_device_properties(device_id).total_memory
            allocated = self.torch.cuda.memory_allocated(device_id)
            reserved = self.torch.cuda.memory_reserved(device_id)
            free = total - allocated
            
            # Convert to MB
            total_mb = total / (1024 ** 2)
            allocated_mb = allocated / (1024 ** 2)
            reserved_mb = reserved / (1024 ** 2)
            free_mb = free / (1024 ** 2)
            
            utilization = (allocated / total * 100) if total > 0 else 0
            
            return {
                'available': True,
                'total_mb': total_mb,
                'allocated_mb': allocated_mb,
                'reserved_mb': reserved_mb,
                'free_mb': free_mb,
                'utilization_percent': utilization,
                'device_name': self.get_device_name(device_id)
            }
        except Exception as e:
            print(f"Error getting GPU memory info: {e}")
            return {
                'available': False,
                'total_mb': 0,
                'allocated_mb': 0,
                'reserved_mb': 0,
                'free_mb': 0,
                'utilization_percent': 0,
                'device_name': 'CPU',
                'error': str(e)
            }
    
    def format_memory(self, mb: float) -> str:
        """
        Format memory in MB to human-readable string
        
        Args:
            mb: Memory in megabytes
            
        Returns:
            Formatted string (e.g., "2.5 GB")
        """
        if mb < 1024:
            return f"{mb:.1f} MB"
        else:
            gb = mb / 1024
            return f"{gb:.1f} GB"
    
    def get_memory_summary(self, device_id: int = 0) -> str:
        """
        Get formatted memory summary string
        
        Args:
            device_id: GPU device index
            
        Returns:
            Formatted summary (e.g., "2.5 GB / 8.0 GB (31.2%)")
        """
        info = self.get_memory_info(device_id)
        
        if not info['available']:
            return "N/A (CPU Mode)"
        
        allocated = self.format_memory(info['allocated_mb'])
        total = self.format_memory(info['total_mb'])
        util = info['utilization_percent']
        
        return f"{allocated} / {total} ({util:.1f}%)"
    
    def check_memory_warning(self, device_id: int = 0, threshold_percent: float = 80.0) -> Optional[str]:
        """
        Check if GPU memory usage exceeds threshold
        
        Args:
            device_id: GPU device index
            threshold_percent: Warning threshold percentage
            
        Returns:
            Warning message if threshold exceeded, None otherwise
        """
        info = self.get_memory_info(device_id)
        
        if not info['available']:
            return None
        
        if info['utilization_percent'] >= threshold_percent:
            return f"⚠️ GPU 메모리 사용률이 높습니다: {info['utilization_percent']:.1f}%"
        
        return None
    
    def clear_cache(self, device_id: int = 0):
        """
        Clear GPU memory cache
        
        Args:
            device_id: GPU device index
        """
        if self.cuda_available and self.torch is not None:
            try:
                self.torch.cuda.empty_cache()
            except Exception as e:
                print(f"Error clearing GPU cache: {e}")
    
    def get_all_devices_info(self) -> List[Dict[str, Any]]:
        """
        Get memory information for all available GPUs
        
        Returns:
            List of memory info dictionaries
        """
        devices_info = []
        
        for i in range(self.device_count):
            info = self.get_memory_info(i)
            info['device_id'] = i
            devices_info.append(info)
        
        return devices_info
    
    def log_memory_snapshot(self, device_id: int = 0, label: str = "") -> Dict[str, Any]:
        """
        Create a memory snapshot for logging/debugging
        
        Args:
            device_id: GPU device index
            label: Optional label for the snapshot
            
        Returns:
            Dictionary with snapshot data
        """
        info = self.get_memory_info(device_id)
        
        snapshot = {
            'timestamp': datetime.now().isoformat(),
            'label': label,
            'device_id': device_id,
            'device_name': info['device_name'],
            'memory': info
        }
        
        return snapshot
    
    def get_recommended_batch_size(self, device_id: int = 0, item_size_mb: float = 10.0) -> int:
        """
        Estimate recommended batch size based on available memory
        
        Args:
            device_id: GPU device index
            item_size_mb: Estimated memory per item in MB
            
        Returns:
            Recommended batch size
        """
        info = self.get_memory_info(device_id)
        
        if not info['available']:
            return 1  # Conservative default for CPU
        
        # Use 70% of free memory for processing
        available_mb = info['free_mb'] * 0.7
        
        if item_size_mb <= 0:
            return 1
        
        batch_size = int(available_mb / item_size_mb)
        
        # Ensure at least 1, max 128
        return max(1, min(batch_size, 128))
