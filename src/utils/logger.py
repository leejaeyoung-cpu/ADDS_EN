"""
Logging utility for ADDS
"""

import logging
import sys
from pathlib import Path
from typing import Optional
from datetime import datetime


class ADDSLogger:
    """
    Centralized logging for ADDS
    """
    
    _loggers = {}
    
    @classmethod
    def get_logger(
        cls,
        name: str,
        log_file: Optional[str] = None,
        level: str = "INFO",
        console_output: bool = True
    ) -> logging.Logger:
        """
        Get or create a logger
        
        Args:
            name: Logger name (usually __name__)
            log_file: Path to log file. If None, uses default
            level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
            console_output: Whether to output to console
        
        Returns:
            Configured logger
        """
        if name in cls._loggers:
            return cls._loggers[name]
        
        # Create logger
        logger = logging.getLogger(name)
        logger.setLevel(getattr(logging, level.upper()))
        
        # Remove existing handlers
        logger.handlers.clear()
        
        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # Console handler
        if console_output:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(getattr(logging, level.upper()))
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)
        
        # File handler
        if log_file:
            log_path = Path(log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            
            file_handler = logging.FileHandler(log_path, encoding='utf-8')
            file_handler.setLevel(getattr(logging, level.upper()))
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        else:
            # Default log file
            default_log_dir = Path(__file__).parent.parent.parent / "logs"
            default_log_dir.mkdir(exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d")
            default_log_file = default_log_dir / f"adds_{timestamp}.log"
            
            file_handler = logging.FileHandler(default_log_file, encoding='utf-8')
            file_handler.setLevel(getattr(logging, level.upper()))
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        
        # Prevent propagation to root logger
        logger.propagate = False
        
        cls._loggers[name] = logger
        return logger


def get_logger(name: str = __name__, level: str = "INFO") -> logging.Logger:
    """
    Convenience function to get a logger
    
    Args:
        name: Logger name
        level: Logging level
    
    Returns:
        Logger instance
    """
    return ADDSLogger.get_logger(name, level=level)
