"""
System Utilities
Consolidated helper functions for file handling, formatting, and configuration
"""

import os
from pathlib import Path
from typing import Any, Callable, Optional
from datetime import datetime
import functools
import traceback


def format_file_size(size_bytes: int) -> str:
    """
    Format bytes into human-readable file size
    
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


def format_timestamp(timestamp: datetime, format_type: str = 'full') -> str:
    """
    Format datetime into readable string
    
    Args:
        timestamp: DateTime object
        format_type: 'full', 'date', 'time', or 'compact'
        
    Returns:
        Formatted timestamp string
    """
    if format_type == 'full':
        return timestamp.strftime('%Y-%m-%d %H:%M:%S')
    elif format_type == 'date':
        return timestamp.strftime('%Y-%m-%d')
    elif format_type == 'time':
        return timestamp.strftime('%H:%M:%S')
    elif format_type == 'compact':
        return timestamp.strftime('%Y%m%d_%H%M%S')
    else:
        return str(timestamp)


def format_duration(seconds: float) -> str:
    """
    Format duration in seconds to readable string
    
    Args:
        seconds: Duration in seconds
        
    Returns:
        Formatted string (e.g., "2분 34초")
    """
    if seconds < 1:
        return f"{seconds*1000:.0f}ms"
    elif seconds < 60:
        return f"{seconds:.1f}초"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{minutes}분 {secs}초"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        return f"{hours}시간 {minutes}분"


def validate_path(path: str, must_exist: bool = False, create_if_missing: bool = False) -> Path:
    """
    Validate and normalize file path
    
    Args:
        path: Path string
        must_exist: Raise error if path doesn't exist
        create_if_missing: Create directory if it doesn't exist
        
    Returns:
        Validated Path object
        
    Raises:
        FileNotFoundError: If must_exist=True and path doesn't exist
    """
    p = Path(path)
    
    if must_exist and not p.exists():
        raise FileNotFoundError(f"Path does not exist: {path}")
    
    if create_if_missing and not p.exists():
        if '.' in p.name:  # Likely a file
            p.parent.mkdir(parents=True, exist_ok=True)
        else:  # Directory
            p.mkdir(parents=True, exist_ok=True)
    
    return p


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """
    Safely divide two numbers, returning default if denominator is zero
    
    Args:
        numerator: Numerator
        denominator: Denominator
        default: Default value if division by zero
        
    Returns:
        Division result or default
    """
    try:
        if denominator == 0:
            return default
        return numerator / denominator
    except:
        return default


def clamp(value: float, min_val: float, max_val: float) -> float:
    """
    Clamp value between min and max
    
    Args:
        value: Value to clamp
        min_val: Minimum value
        max_val: Maximum value
        
    Returns:
        Clamped value
    """
    return max(min_val, min(value, max_val))


def error_handler(default_return: Any = None, log_error: bool = True):
    """
    Decorator for graceful error handling
    
    Args:
        default_return: Value to return on error
        log_error: Whether to print error traceback
        
    Returns:
        Decorator function
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                if log_error:
                    print(f"Error in {func.__name__}: {str(e)}")
                    traceback.print_exc()
                return default_return
        return wrapper
    return decorator


def ensure_dir(path: str) -> Path:
    """
    Ensure directory exists, create if necessary
    
    Args:
        path: Directory path
        
    Returns:
        Path object
    """
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def get_available_filename(base_path: str, extension: str = "") -> Path:
    """
    Get available filename by appending number if file exists
    
    Args:
        base_path: Base file path
        extension: File extension (optional)
        
    Returns:
        Available file path
    """
    p = Path(base_path)
    
    if extension and not extension.startswith('.'):
        extension = f".{extension}"
    
    # If no extension in base_path but extension provided
    if extension and not p.suffix:
        p = p.with_suffix(extension)
    
    if not p.exists():
        return p
    
    # Add number suffix
    counter = 1
    stem = p.stem
    suffix = p.suffix
    parent = p.parent
    
    while True:
        new_path = parent / f"{stem}_{counter}{suffix}"
        if not new_path.exists():
            return new_path
        counter += 1


def truncate_string(text: str, max_length: int = 50, suffix: str = "...") -> str:
    """
    Truncate string to maximum length
    
    Args:
        text: String to truncate
        max_length: Maximum length
        suffix: Suffix to add if truncated
        
    Returns:
        Truncated string
    """
    if len(text) <= max_length:
        return text
    
    return text[:max_length - len(suffix)] + suffix


def parse_iso_timestamp(timestamp_str: str) -> Optional[datetime]:
    """
    Parse ISO format timestamp string
    
    Args:
        timestamp_str: ISO format timestamp
        
    Returns:
        Datetime object or None if parsing fails
    """
    try:
        return datetime.fromisoformat(timestamp_str)
    except:
        return None


def get_file_age_days(file_path: str) -> float:
    """
    Get file age in days
    
    Args:
        file_path: Path to file
        
    Returns:
        Age in days
    """
    try:
        p = Path(file_path)
        if not p.exists():
            return 0.0
        
        mtime = datetime.fromtimestamp(p.stat().st_mtime)
        age = datetime.now() - mtime
        return age.total_seconds() / 86400
    except:
        return 0.0


def create_timestamped_backup(file_path: str, backup_dir: str = "backups") -> Optional[Path]:
    """
    Create timestamped backup of a file
    
    Args:
        file_path: Path to file to backup
        backup_dir: Directory for backups
        
    Returns:
        Path to backup file, or None if failed
    """
    try:
        import shutil
        
        source = Path(file_path)
        if not source.exists():
            return None
        
        backup_path = ensure_dir(backup_dir)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        backup_file = backup_path / f"{source.stem}_{timestamp}{source.suffix}"
        
        shutil.copy2(source, backup_file)
        
        return backup_file
    except Exception as e:
        print(f"Error creating backup: {e}")
        return None
