"""
Advanced Cache Manager for Streamlit Resource Management
Provides cache statistics, monitoring, and control utilities
"""

import os
import shutil
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
import json


class CacheManager:
    """
    Manage and monitor Streamlit cache resources
    Provides statistics and control over cached models and data
    """
    
    def __init__(self, cache_dir: str = "data/cache"):
        """
        Initialize cache manager
        
        Args:
            cache_dir: Directory to store cached resources
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Metadata file to track cache statistics
        self.metadata_file = self.cache_dir / "cache_metadata.json"
        self.metadata = self._load_metadata()
    
    def _load_metadata(self) -> Dict[str, Any]:
        """Load cache metadata from file"""
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except:
                pass
        
        # Default metadata structure
        return {
            'created_at': datetime.now().isoformat(),
            'last_cleared': None,
            'total_clears': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'items': {}
        }
    
    def _save_metadata(self):
        """Save cache metadata to file"""
        with open(self.metadata_file, 'w', encoding='utf-8') as f:
            json.dump(self.metadata, f, indent=2, ensure_ascii=False)
    
    def get_cache_size(self) -> int:
        """
        Calculate total cache size in bytes
        
        Returns:
            Total size in bytes
        """
        total_size = 0
        
        try:
            for item in self.cache_dir.rglob('*'):
                if item.is_file():
                    total_size += item.stat().st_size
        except Exception as e:
            print(f"Error calculating cache size: {e}")
        
        return total_size
    
    def format_size(self, size_bytes: int) -> str:
        """
        Format bytes into human-readable string
        
        Args:
            size_bytes: Size in bytes
            
        Returns:
            Formatted string (e.g., "125.5 MB")
        """
        for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
            if size_bytes < 1024.0:
                return f"{size_bytes:.1f} {unit}"
            size_bytes /= 1024.0
        return f"{size_bytes:.1f} PB"
    
    def get_cache_item_count(self) -> int:
        """
        Count number of cached items
        
        Returns:
            Number of files in cache directory
        """
        try:
            return sum(1 for item in self.cache_dir.rglob('*') if item.is_file())
        except:
            return 0
    
    def clear_cache(self, confirm: bool = False) -> bool:
        """
        Clear all cached resources
        
        Args:
            confirm: Safety flag to prevent accidental clearing
            
        Returns:
            True if cache was cleared, False otherwise
        """
        if not confirm:
            return False
        
        try:
            # Remove all files and subdirectories except metadata
            for item in self.cache_dir.iterdir():
                if item.name == 'cache_metadata.json':
                    continue
                
                if item.is_file():
                    item.unlink()
                elif item.is_dir():
                    shutil.rmtree(item)
            
            # Update metadata
            self.metadata['last_cleared'] = datetime.now().isoformat()
            self.metadata['total_clears'] += 1
            self.metadata['items'] = {}
            self._save_metadata()
            
            return True
        except Exception as e:
            print(f"Error clearing cache: {e}")
            return False
    
    def clear_item(self, item_name: str) -> bool:
        """
        Clear a specific cached item
        
        Args:
            item_name: Name of the item to clear
            
        Returns:
            True if item was cleared, False otherwise
        """
        try:
            item_path = self.cache_dir / item_name
            
            if item_path.exists():
                if item_path.is_file():
                    item_path.unlink()
                elif item_path.is_dir():
                    shutil.rmtree(item_path)
                
                # Update metadata
                if item_name in self.metadata['items']:
                    del self.metadata['items'][item_name]
                self._save_metadata()
                
                return True
        except Exception as e:
            print(f"Error clearing cache item {item_name}: {e}")
        
        return False
    
    def record_cache_hit(self, item_name: str):
        """Record a cache hit for statistics"""
        self.metadata['cache_hits'] += 1
        
        if item_name not in self.metadata['items']:
            self.metadata['items'][item_name] = {
                'first_access': datetime.now().isoformat(),
                'last_access': datetime.now().isoformat(),
                'hit_count': 1
            }
        else:
            self.metadata['items'][item_name]['last_access'] = datetime.now().isoformat()
            self.metadata['items'][item_name]['hit_count'] += 1
        
        self._save_metadata()
    
    def record_cache_miss(self):
        """Record a cache miss for statistics"""
        self.metadata['cache_misses'] += 1
        self._save_metadata()
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive cache statistics
        
        Returns:
            Dictionary with cache statistics
        """
        total_size = self.get_cache_size()
        item_count = self.get_cache_item_count()
        
        total_accesses = self.metadata['cache_hits'] + self.metadata['cache_misses']
        hit_rate = (self.metadata['cache_hits'] / total_accesses * 100) if total_accesses > 0 else 0
        
        return {
            'total_size': total_size,
            'total_size_formatted': self.format_size(total_size),
            'item_count': item_count,
            'cache_hits': self.metadata['cache_hits'],
            'cache_misses': self.metadata['cache_misses'],
            'hit_rate_percent': hit_rate,
            'last_cleared': self.metadata['last_cleared'],
            'total_clears': self.metadata['total_clears'],
            'created_at': self.metadata['created_at']
        }
    
    def get_top_items(self, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Get most frequently accessed cache items
        
        Args:
            limit: Maximum number of items to return
            
        Returns:
            List of item statistics
        """
        items = []
        
        for name, data in self.metadata['items'].items():
            items.append({
                'name': name,
                'hit_count': data['hit_count'],
                'first_access': data['first_access'],
                'last_access': data['last_access']
            })
        
        # Sort by hit count
        items.sort(key=lambda x: x['hit_count'], reverse=True)
        
        return items[:limit]
    
    def create_backup(self) -> Optional[Path]:
        """
        Create a timestamped backup of the cache
        
        Returns:
            Path to backup directory, or None if failed
        """
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            backup_dir = self.cache_dir.parent / f"cache_backup_{timestamp}"
            
            shutil.copytree(self.cache_dir, backup_dir)
            
            return backup_dir
        except Exception as e:
            print(f"Error creating cache backup: {e}")
            return None
    
    def get_age_days(self) -> float:
        """
        Get age of cache in days
        
        Returns:
            Number of days since cache creation
        """
        try:
            created = datetime.fromisoformat(self.metadata['created_at'])
            age = datetime.now() - created
            return age.total_seconds() / 86400  # Convert to days
        except:
            return 0.0
